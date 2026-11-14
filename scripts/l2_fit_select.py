"""Cross-fit LightGBM ubar scorers, sweep disjoint selection and report EV1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if os.environ.get("SOLAR_FINAL_INFERENCE") != "1":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from solar import candidates, data, metric, select

TAUS = [round(t / 100, 2) for t in range(10, 37, 2)]
BANK = {}
TABLES = {}
RECORDS = {}
FEATURES = candidates.FEATURES
MODEL_OPTIONS = {}
BASE_FEATURES = candidates.FEATURES


def read_bank(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for filename, key in (("features.parquet", "features_sha256"), ("rles.pkl", "rles_sha256")):
        if hashlib.sha256((path / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"Bank file hash mismatch: {filename}")
    with (path / "rles.pkl").open("rb") as stream:
        bank = pickle.load(stream)
    return pd.read_parquet(path / "features.parquet"), bank, manifest


def estimator(objective, workers):
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=40, reg_lambda=2.0, colsample_bytree=0.9)
    preset = MODEL_OPTIONS.get("model_preset", "default")
    if preset == "regularized":
        params.update(n_estimators=600, learning_rate=0.03, num_leaves=15,
                      min_child_samples=160, reg_lambda=10.0)
    elif preset == "capacity":
        params.update(n_estimators=900, learning_rate=0.03, num_leaves=63,
                      min_child_samples=40, reg_lambda=4.0)
    return lgb.LGBMRegressor(objective=objective, **params, random_state=261009,
                             n_jobs=workers, verbosity=-1, deterministic=True, force_col_wise=True)


def train_models(table, workers):
    core_features = BASE_FEATURES if MODEL_OPTIONS.get("hit_features_only") else FEATURES
    x = table[core_features]
    weights = table.rater_ious.map(len).to_numpy()
    direct = estimator("regression", workers).fit(x, table.ubar, sample_weight=weights)
    # Cross entropy accepts fractional Bernoulli labels (rater hit proportions).
    if MODEL_OPTIONS.get("hit_rater_level"):
        indices = np.repeat(np.arange(len(table)), weights)
        labels = np.concatenate(table.rater_ious.to_numpy()) > 0.5
        hit = estimator("binary", workers).fit(table[FEATURES].iloc[indices], labels.astype(float))
    else:
        hit = estimator("cross_entropy", workers).fit(table[FEATURES], table.hit, sample_weight=weights)
    matched = table.hit.to_numpy() > 0
    conditional = estimator("regression", workers)
    if matched.any():
        conditional.fit(x.loc[matched], table.loc[matched, "miou"],
                        sample_weight=weights[matched] * table.loc[matched, "hit"].to_numpy())
    else:
        conditional = None
    return dict(direct=direct, hit=hit, conditional=conditional, core_features=core_features)


def predict_models(models, x):
    core = x[models.get("core_features", list(x.columns))]
    direct = np.clip(models["direct"].predict(core), 0, 1)
    hit = np.clip(models["hit"].predict(x), 0, 1)
    conditional = np.clip(models["conditional"].predict(core), 0.5, 1) if models["conditional"] is not None else np.zeros(len(x))
    return direct, hit * conditional


def cross_fit(table, workers):
    table = table.copy()
    table["pred_direct"], table["pred_hurdle"] = np.nan, np.nan
    models, folds = {}, []
    for fold in sorted(int(f) for f in table.fold.unique()):
        validation = table.fold == fold
        training = ~validation
        if not training.any() or not validation.any():
            raise ValueError(f"Fold {fold} lacks train/validation candidates")
        started = time.perf_counter()
        model = train_models(table.loc[training], workers)
        direct, hurdle = predict_models(model, table.loc[validation, FEATURES])
        table.loc[validation, "pred_direct"] = direct
        table.loc[validation, "pred_hurdle"] = hurdle
        models[fold] = model
        folds.append(dict(fold=fold, training_candidates=int(training.sum()), validation_candidates=int(validation.sum()),
                          training_images=int(table.loc[training, "stem"].nunique()),
                          validation_images=int(table.loc[validation, "stem"].nunique()),
                          elapsed_seconds=time.perf_counter()-started))
        print(f"fit fold={fold} train={training.sum()} validation={validation.sum()} wall={folds[-1]['elapsed_seconds']:.1f}s", flush=True)
    return table, models, folds


def sweep_one(stem):
    image = BANK[stem]
    table = TABLES.get(stem)
    n = len(image["rles"])
    if table is None:
        scores = {name: np.empty(0) for name in ("direct", "hurdle", "oracle")}
        allowed, parents, baseline = np.zeros(0, bool), np.empty(0, int), []
    else:
        scores = dict(direct=table.pred_direct.to_numpy(), hurdle=table.pred_hurdle.to_numpy(), oracle=table.ubar.to_numpy())
        allowed, parents = table.in_a.to_numpy(), table.parent_a.to_numpy()
        baseline = np.flatnonzero(table.baseline.to_numpy()).tolist()
    conflicts = select.overlap_graph(image["rles"])
    selected, result, objectives = {}, {}, {}
    recs = RECORDS[stem]
    if image["record_ids"] != [rec.record_id for rec in recs]:
        raise ValueError(f"GT record lineage mismatch: {stem}")
    bucket = "single" if len(recs) == 1 else "multi"

    def score(ids):
        stats = metric.PQStats()
        for matrix in image["matrices"]:
            stats.add(metric.record_stats(matrix[:, ids]))
        return stats.as_dict()

    result["baseline"] = [score(baseline)]
    selected["baseline"] = [baseline]
    for scorer, values in scores.items():
        for method in ("greedy_ab", "greedy_a", "dp_a"):
            key = f"{scorer}/{method}"
            selections, summaries, sums = [], [], []
            for tau in TAUS:
                if method == "dp_a":
                    ids = select.tree_dp(values, tau, parents, allowed)
                else:
                    ids = select.greedy(values, tau, conflicts, allowed if method == "greedy_a" else None)
                # Verify the actual emitted masks are disjoint on every image/sweep.
                if len(ids) > 1 and int(conflicts[np.ix_(ids, ids)].sum()) != len(ids):
                    raise ValueError(f"Overlapping selection: {stem} {key} {tau}")
                selections.append(ids)
                summaries.append(score(ids))
                sums.append(float(np.sum(values[ids] - tau)))
            selected[key], result[key], objectives[key] = selections, summaries, sums
    return stem, bucket, result, selected, objectives, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else None)


def selection_preds(selections, key, index, bank):
    return {stem: [bank[stem]["rles"][cid] for cid in choices[key][index]] for stem, choices in selections.items()}


def write_submission(path, test_bank, test_table, models, scorer, method, tau):
    predictions = np.zeros(len(test_table))
    for model in models.values():
        if len(test_table):
            direct, hurdle = predict_models(model, test_table[FEATURES])
            predictions += direct if scorer == "direct" else hurdle
    predictions /= len(models)
    test_table = test_table.copy()
    test_table["score"] = predictions
    groups = {stem: group.sort_values("id") for stem, group in test_table.groupby("stem", sort=False)}
    path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem, image in test_bank.items():
            table = groups.get(stem)
            if table is None:
                continue
            conflicts = select.overlap_graph(image["rles"])
            if method == "dp_a":
                ids = select.tree_dp(table.score.to_numpy(), tau, table.parent_a.to_numpy(), table.in_a.to_numpy())
            else:
                ids = select.greedy(table.score.to_numpy(), tau, conflicts, table.in_a.to_numpy() if method == "greedy_a" else None)
            if len(ids) > 1 and int(conflicts[np.ix_(ids, ids)].sum()) != len(ids):
                raise ValueError(f"Overlapping submission masks: {stem}")
            for k, cid in enumerate(ids, 1):
                counts = image["rles"][cid]["counts"]
                writer.writerow([f"{stem}_{k}", counts.decode("ascii") if isinstance(counts, bytes) else counts])
                emitted += 1
    return dict(path=str(path.resolve()), instances=emitted, images=len(test_bank), scorer=scorer, method=method, tau=tau,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model=f"mean of {len(models)} cross-fit scorers",
                scorer_count=len(models))


def main():
    global BANK, TABLES, RECORDS, FEATURES, BASE_FEATURES, MODEL_OPTIONS, TAUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-scorers", action="store_true")
    parser.add_argument("--within-fold-cross-fit", action="store_true", help="Replace one OOF fold with five stem-order modulo-5 sub-folds")
    parser.add_argument("--test-bank", type=Path)
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--tau", type=float, help="Override the learned best tau for submission only")
    parser.add_argument("--scorer-dir", type=Path, help="Separate score/model/selection outputs from the input bank")
    parser.add_argument("--feature-columns", help="Comma-separated scorer feature columns; default is the manifest")
    parser.add_argument("--feature-groups", help="Comma-separated added groups from the manifest, appended to base features")
    parser.add_argument("--hit-features-only", action="store_true", help="Use added features only for hit; keep direct/miou on base features")
    parser.add_argument("--hit-rater-level", action="store_true", help="Fit hit on one binary row per rater instead of weighted proportions")
    parser.add_argument("--model-preset", choices=("default", "regularized", "capacity"), default="default")
    parser.add_argument("--tau-grid", help="Comma-separated selection thresholds; default is 0.10:0.02:0.36")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    if bool(args.test_bank) != bool(args.submission_csv):
        parser.error("test-bank and submission-csv must be supplied together")
    started = time.perf_counter()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    table, BANK, manifest = read_bank(args.bank_dir)
    FEATURES = manifest.get("feature_columns", candidates.FEATURES)
    BASE_FEATURES = manifest.get("base_feature_columns", FEATURES)
    if args.feature_groups:
        if args.feature_columns:
            parser.error("feature-groups and feature-columns are mutually exclusive")
        groups = args.feature_groups.split(",")
        declared = manifest.get("feature_groups", {})
        if not set(groups) <= set(declared):
            parser.error("feature-groups must be declared in the enriched bank manifest")
        FEATURES = BASE_FEATURES + [column for group in groups for column in declared[group]]
    if args.feature_columns:
        FEATURES = args.feature_columns.split(",")
        if not set(FEATURES) <= set(manifest.get("feature_columns", candidates.FEATURES)):
            parser.error("feature-columns must be declared in the bank manifest")
    if args.tau_grid:
        TAUS = [float(value) for value in args.tau_grid.split(",")]
    MODEL_OPTIONS = dict(model_preset=args.model_preset, hit_features_only=args.hit_features_only,
                         hit_rater_level=args.hit_rater_level)
    if manifest["split"] != "train":
        parser.error("scorer fitting requires a train bank")
    RECORDS = data.cached_records()
    stems = manifest["stems"]
    fold_path = Path(__file__).resolve().parents[1] / "configs/folds_v1.csv"
    if hashlib.sha256(fold_path.read_bytes()).hexdigest() != manifest["folds_sha256"]:
        raise ValueError("Bank fold identity differs from current folds_v1")
    if args.within_fold_cross_fit:
        if table.fold.nunique() != 1 and manifest.get("cross_fit_identity") != "stem-order-mod5":
            raise ValueError("Within-fold cross-fit requires a single original OOF fold")
        if manifest.get("cross_fit_identity") != "stem-order-mod5":
            manifest["original_oof_fold"] = int(table.fold.iloc[0])
            table["fold"] = table.stem.map({stem: i % 5 for i, stem in enumerate(sorted(stems))})
            table.to_parquet(args.bank_dir / "features.parquet", index=False)
            manifest["features_sha256"] = hashlib.sha256((args.bank_dir / "features.parquet").read_bytes()).hexdigest()
            manifest["cross_fit_identity"] = "stem-order-mod5"
            manifest["cross_fit_assignment"] = {stem: i % 5 for i, stem in enumerate(sorted(stems))}
            (args.bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    scorer_dir = args.scorer_dir or args.bank_dir
    scorer_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, model_path = scorer_dir / "oof_scores.parquet", scorer_dir / "scorers.pkl"
    if args.reuse_scorers:
        table = pd.read_parquet(prediction_path)
        with model_path.open("rb") as stream:
            bundle = pickle.load(stream)
        if bundle["features_sha256"] != manifest["features_sha256"]:
            raise ValueError("Scorers belong to a different bank")
        if bundle["features"] != FEATURES:
            raise ValueError("Scorer feature columns differ from the bank")
        if bundle.get("model_options", dict(model_preset="default", hit_features_only=False, hit_rater_level=False)) != MODEL_OPTIONS:
            raise ValueError("Scorer options differ from the requested fit")
        models, folds = bundle["models"], bundle["folds"]
        model_source_sha = bundle.get("source_sha", manifest["source_sha"])
    else:
        table, models, folds = cross_fit(table, args.workers)
        model_source_sha = source_sha
        table.to_parquet(prediction_path, index=False)
        with model_path.open("wb") as stream:
            pickle.dump(dict(models=models, folds=folds, features=FEATURES,
                             features_sha256=manifest["features_sha256"], source_sha=source_sha,
                             model_options=MODEL_OPTIONS, lightgbm_version=lgb.__version__), stream)
    if table[["pred_direct", "pred_hurdle"]].isna().any().any():
        raise ValueError("Missing cross-fit predictions")
    fit_seconds = time.perf_counter()-started
    TABLES = {stem: group.sort_values("id") for stem, group in table.groupby("stem", sort=False)}
    selections, aggregate, objectives, max_rss = {}, {}, {}, 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (stem, bucket, result, selected, objective, rss) in enumerate(pool.map(sweep_one, stems, chunksize=1), 1):
            selections[stem] = selected
            max_rss = max(max_rss, rss) if rss is not None else None
            for key, summaries in result.items():
                if key not in aggregate:
                    aggregate[key] = [{b: metric.PQStats() for b in ("all", "single", "multi")} for _ in summaries]
                for index, summary in enumerate(summaries):
                    stats = metric.PQStats(summary["iou_sum"], summary["tp"], summary["fp"], summary["fn"])
                    aggregate[key][index]["all"].add(stats)
                    aggregate[key][index][bucket].add(stats)
            for key, values in objective.items():
                objectives.setdefault(key, np.zeros(len(TAUS)))
                objectives[key] += values
            if done % 50 == 0 or done == len(stems):
                print(f"select {done}/{len(stems)} wall={time.perf_counter()-started:.1f}s max_worker_rss={format(max_rss, '.0f') if max_rss is not None else 'unavailable'}MB", flush=True)
    sweep = {key: [dict(tau=TAUS[i], **{bucket: stats.as_dict() for bucket, stats in buckets.items()}) for i, buckets in enumerate(values)] for key, values in aggregate.items() if key != "baseline"}
    best = {}
    for key, values in sweep.items():
        index = int(np.argmax([entry["all"]["pq"] for entry in values]))
        preds = selection_preds(selections, key, index, BANK)
        exact = metric.evaluate(preds, RECORDS, stems).summary()
        for bucket in exact:
            cached = values[index][bucket]
            if any(exact[bucket][field] != cached[field] for field in ("tp", "fp", "fn")) or not np.isclose(exact[bucket]["iou_sum"], cached["iou_sum"], atol=1e-8):
                raise ValueError(f"Cached sweep disagrees with metric.evaluate: {key} {bucket}")
        best[key] = dict(tau=TAUS[index], **exact)
    baseline = metric.evaluate(selection_preds(selections, "baseline", 0, BANK), RECORDS, stems).summary()
    learned_key = max((key for key in best if not key.startswith("oracle/")), key=lambda key: best[key]["all"]["pq"])
    oracle_key = max((key for key in best if key.startswith("oracle/")), key=lambda key: best[key]["all"]["pq"])
    comparisons = {}
    for scorer in ("direct", "hurdle", "oracle"):
        greedy_key, dp_key = f"{scorer}/greedy_a", f"{scorer}/dp_a"
        comparisons[scorer] = dict(best_dp_minus_best_greedy_pq=best[dp_key]["all"]["pq"]-best[greedy_key]["all"]["pq"],
                                  same_tau=[dict(tau=tau, dp_minus_greedy_pq=sweep[dp_key][i]["all"]["pq"]-sweep[greedy_key][i]["all"]["pq"],
                                                 dp_minus_greedy_objective=float(objectives[dp_key][i]-objectives[greedy_key][i])) for i, tau in enumerate(TAUS)])
        if any(entry["dp_minus_greedy_objective"] < -1e-6 for entry in comparisons[scorer]["same_tau"]):
            raise ValueError(f"DP underperforms greedy antichain objective: {scorer}")
    gaps = {method: best[f"oracle/{method}"]["all"]["pq"] - max(best[f"{scorer}/{method}"]["all"]["pq"] for scorer in ("direct", "hurdle")) for method in ("greedy_ab", "greedy_a", "dp_a")}
    report = dict(evaluation_identity="EV1", source_sha=source_sha, model_source_sha=model_source_sha,
                  model_options=MODEL_OPTIONS, scorer_dir=str(scorer_dir.resolve()),
                  feature_columns=FEATURES, cross_fit_identity=manifest.get("cross_fit_identity", "folds_v1"),
                  original_oof_fold=manifest.get("original_oof_fold"), features_sha256=manifest["features_sha256"],
                  fit_code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selector_code_hash=hashlib.sha256(Path(select.__file__).read_bytes()).hexdigest(),
                  scorers_reused=args.reuse_scorers, workers=args.workers,
                  scope=dict(images=len(stems), records=sum(len(RECORDS[s]) for s in stems),
                  instances=sum(len(rec.instances) for s in stems for rec in RECORDS[s]), single_images=sum(len(RECORDS[s]) == 1 for s in stems),
                  multi_images=sum(len(RECORDS[s]) > 1 for s in stems)), bank=dict(path=str(args.bank_dir.resolve()),
                  source_sha=manifest["source_sha"], code_hash=manifest["code_hash"], candidates=manifest["candidates"],
                  elapsed_seconds=manifest["elapsed_seconds"], candidates_per_image=manifest["candidates_per_image"],
                  max_worker_rss_mb=manifest["max_worker_rss_mb"], parent_max_rss_mb=manifest["parent_max_rss_mb"]),
                  probability_kind=manifest["probability_kind"], caveats=["Proxy scores demonstrate execution only; GT-derived inputs are not research evidence."] if "proxy" in manifest["probability_kind"] else ["Confirm dense OOF provenance separately."],
                  threshold_selection="Best tau/scorer/method selected on pooled cross-fit results; no nested validation for these choices.",
                  oracle_definition="True ubar with greedy A+B or exact A-only antichain; A+B greedy is not a global oracle upper bound.",
                  folds=folds, lightgbm_version=lgb.__version__, baseline=baseline, best=best, sweep=sweep,
                  best_learned=dict(configuration=learned_key, **best[learned_key]), best_oracle=dict(configuration=oracle_key, **best[oracle_key]),
                  oracle_minus_learned_pq=best[oracle_key]["all"]["pq"]-best[learned_key]["all"]["pq"],
                  oracle_minus_learned_by_method=gaps, greedy_vs_dp=comparisons,
                  fit_seconds=fit_seconds, max_selection_worker_rss_mb=max_rss,
                  parent_max_rss_mb=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource is not None else None))
    if args.test_bank:
        if "proxy" in manifest["probability_kind"]:
            raise ValueError("Proxy scorers cannot produce a research submission")
        test_table, test_bank, test_manifest = read_bank(args.test_bank)
        if test_manifest["split"] != "test" or set(test_manifest["stems"]) != set(data.list_test_stems()):
            raise ValueError("Submission requires a complete test bank")
        scorer, method = learned_key.split("/")
        report["submission"] = write_submission(args.submission_csv, test_bank, test_table, models, scorer, method,
                                                 args.tau if args.tau is not None else best[learned_key]["tau"])
    report["elapsed_seconds"] = time.perf_counter()-started
    report["pipeline_elapsed_seconds"] = manifest["elapsed_seconds"] + report["elapsed_seconds"]
    with (scorer_dir / "selections.pkl").open("wb") as stream:
        pickle.dump(dict(taus=TAUS, selections=selections), stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("best_learned", "best_oracle", "oracle_minus_learned_pq", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
