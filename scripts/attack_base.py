"""Rebuild and deploy full-707 EV1 CPU banks and family-admissible scorers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

if os.environ.get("SOLAR_FINAL_INFERENCE") != "1":
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.isotonic import IsotonicRegression
from solar import candidates, data, metric, select
from l2_fit_select import read_bank
cv2.setNumThreads(1)

ROOT = data.artifact_root()
OUT = Path(os.environ.get("SOLAR_ATTACK_BASE_OUT", ROOT / "attack_base_20261030"))
MARKER_OUT = Path(os.environ.get("SOLAR_ATTACK_MARKER_OUT", ROOT / "markers"))
BANK, TABLES, RECORDS, SOURCES = {}, {}, {}, {}
PAINT_PLANS = []
TAUS = [round(t / 100, 2) for t in range(10, 37, 2)]
try:
    RUN_SOURCE_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
except (OSError, subprocess.CalledProcessError):
    release = REPO / "docs/final/RELEASE_MANIFEST.json"
    RUN_SOURCE_SHA = json.loads(release.read_text())["source_sha"] if release.exists() else "unknown"
RUN_CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def source_receipt():
    return dict(source_sha=RUN_SOURCE_SHA, code_sha256=RUN_CODE_SHA, timestamp=datetime.now(timezone.utc).isoformat())


def run_script(script, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", script, *map(str, args), flush=True)
    with log.open("w") as stream:
        subprocess.run([sys.executable, str(REPO / "scripts" / script), *map(str, args)], stdout=stream, stderr=subprocess.STDOUT, check=True)


def discover():
    sources = {}
    for key, filename in (("long2x", "maps_long2x_ready.json"), ("s05", "maps_s05_ready.json"), ("gong", "maps_w45gong_ready.json"),
                          ("long2x_tta", "maps_long2x_tta_ready.json"), ("gong_tta", "maps_w45gong_tta_ready.json")):
        path = ROOT / "markers" / filename
        if not path.exists():
            continue
        marker = json.loads(path.read_text())
        assert marker["status"] == "READY", path
        if marker.get("receipt_sha256"):
            assert sha(marker["receipt"]) == marker["receipt_sha256"], path
        extra_markers = {}
        if key == "gong":
            for child in sorted((ROOT / "markers").glob("maps_w45gong_fold*_ready.json")):
                value = json.loads(child.read_text())
                assert value["status"] == "READY" and value["oof"] == marker["oof"]
                if value.get("receipt_sha256"):
                    assert sha(value["receipt"]) == value["receipt_sha256"]
                marker["folds"] = sorted(set(marker["folds"] + value["folds"]))
                marker["test_prob"].update(value["test_prob"])
                extra_markers[str(child)] = sha(child)
            if extra_markers:
                assignments = pd.read_csv(REPO / "configs/folds_v1.csv")
                marker["oof_images"] = int(assignments.fold.isin(marker["folds"]).sum())
                marker["fold_marker_sha256"] = extra_markers
        oof = Path(marker.get("oof", marker.get("oof_dir", "")))
        assert oof.is_dir(), (path, oof)
        tests = marker.get("test_prob", marker.get("test", marker.get("test_dir")))
        tests = list(tests.values()) if isinstance(tests, dict) else tests if isinstance(tests, list) else [tests]
        assert tests and all(tests), path
        fingerprint = hashlib.sha256(json.dumps(marker, sort_keys=True).encode()).hexdigest()
        destination = OUT / "maps" / key / fingerprint[:12] / "test" if extra_markers else OUT / "maps" / key / "test"
        stamp = destination.parent / "average.json"
        if not stamp.exists() or json.loads(stamp.read_text())["fingerprint"] != fingerprint:
            run_script("avg_probs.py", ["--inputs", *tests, "--out", destination], OUT / f"average_{key}.log")
            dump(stamp, dict(fingerprint=fingerprint, sources=tests, **source_receipt()))
        sources[key] = dict(oof=str(oof), test=str(destination), marker=str(path), marker_sha256=sha(path), folds=marker.get("folds"), oof_images=marker.get("oof_images"))
        if extra_markers:
            sources[key]["marker_fingerprint"] = fingerprint
            sources[key]["fold_markers"] = extra_markers
    rf_markers = sorted((ROOT / "markers").glob("rf_family_ready_fold*.json"))
    if rf_markers and "long2x" in sources:
        fingerprint = hashlib.sha256(json.dumps({str(p): sha(p) for p in rf_markers}, sort_keys=True).encode()).hexdigest()
        rf_out = OUT / "rf_intake" / fingerprint[:12]
        if not (rf_out / "intake.json").exists():
            run_script("attack_rf_intake.py", ["--out", rf_out, "--long-oof", sources["long2x"]["oof"], "--long-test", sources["long2x"]["test"], "--workers", 2], rf_out / "intake.log")
        info = json.loads((rf_out / "intake.json").read_text())
        rf_table, _, rf_manifest = read_bank(Path(info["train_bank"]))
        sources["rf"] = dict(oof=None, test=None, bank=info["train_bank"], test_bank=info["test_bank"], marker=str(rf_out / "intake.json"),
            marker_sha256=sha(rf_out / "intake.json"), oof_images=rf_manifest["images"], folds=sorted(map(int, rf_table.fold.unique())))
    # RF workers can hand off the standard features/RLE/manifest bank contract.
    for marker_path in sorted((ROOT / "markers").glob("rf_family_ready*.json"), key=lambda p: p.stat().st_mtime):
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "READY":
            continue
        train_bank = marker.get("train_bank", marker.get("oof_bank", marker.get("bank_dir", marker.get("bank"))))
        test_bank = marker.get("test_bank")
        if isinstance(train_bank, str) and isinstance(test_bank, str):
            train, _, manifest = read_bank(Path(train_bank))
            read_bank(Path(test_bank))
            assert set(train.stem) <= set(data.cached_records())
            sources["rf"] = dict(oof=None, test=None, bank=train_bank, test_bank=test_bank, marker=str(marker_path),
                                 marker_sha256=sha(marker_path), oof_images=manifest["images"], folds=sorted(map(int, train.fold.unique())))
    dump(OUT / "sources.json", sources)
    return sources


def load(path, split="train"):
    global BANK, TABLES, RECORDS, SOURCES
    table, BANK, manifest = read_bank(Path(path))
    table = table.sort_values(["stem", "id"]).reset_index(drop=True)
    TABLES = {s: f.copy() for s, f in table.groupby("stem", sort=False)}
    RECORDS = data.cached_records() if split == "train" else {}
    SOURCES = manifest["sources"]
    cv2.setNumThreads(1)
    return table, manifest


def support_one(stem):
    cv2.setNumThreads(1)
    table = TABLES.get(stem)
    if table is None:
        return stem, [], np.empty(0, int)
    entry = BANK[stem]
    rles = entry["rles"]
    iou = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
    groups = connected_components(csr_matrix(iou > .1), directed=False)[1]
    rows = [dict(stem=stem, id=i) for i in range(len(rles))]
    crops = [select.crop_decode(rle) for rle in rles]
    split = "oof" if RECORDS else "test"
    probability = {}
    for name, source in SOURCES.items():
        path = Path(source[split]) / f"{stem}_fg.png" if source.get(split) else None
        value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path is not None and path.exists() else None
        if value is not None:
            assert value.dtype == np.uint8 and value.shape == (2048, 2048), path
            probability[name] = value.astype(np.float32) / 255
        own = np.flatnonzero(table.source_name.to_numpy() == name)
        for i, (mask, x, y) in enumerate(crops):
            pixels = probability[name][y:y+mask.shape[0], x:x+mask.shape[1]][mask] if name in probability else None
            rows[i][f"support_{name}_available"] = int(pixels is not None)
            rows[i][f"support_{name}_family_available"] = int(len(own) > 0)
            rows[i][f"support_{name}_mean"] = float(pixels.mean()) if pixels is not None else np.nan
            rows[i][f"support_{name}_max"] = float(pixels.max()) if pixels is not None else np.nan
            rows[i][f"support_{name}_iou"] = float(iou[i, own].max()) if len(own) else np.nan
            rows[i][f"support_{name}_count50"] = int((iou[i, own] > .5).sum()) if len(own) else 0
            if name == "rf":
                scores = table.iloc[own].rf_score.to_numpy() if len(own) else np.empty(0)
                rows[i]["rf_max_iou_score"] = float((iou[i, own]*scores).max()) if len(own) else 0.
                rows[i]["rf_overlap_count"] = int((iou[i, own] > 0).sum()) if len(own) else 0
                rows[i]["rf_best_score"] = float(np.where(iou[i, own] > 0, scores, 0).max()) if len(own) else 0.
    for row in rows:
        for name in SOURCES:
            if name != "long2x":
                for stat in ("mean", "max"):
                    row[f"support_{name}_{stat}_delta"] = row[f"support_{name}_{stat}"] - row.get(f"support_long2x_{stat}", np.nan)
    return stem, rows, groups


def prepare(name, include, support, workers, shift=0.0):
    sources = discover()
    assert set(include + support) <= set(sources), (include, support, sources.keys())
    sources = {k: sources[k] for k in support}
    if shift:
        for family, source in sources.items():
            for map_kind in ("oof", "test"):
                if not source.get(map_kind):
                    continue
                probability = Path(source[map_kind])
                coverage = "_folds" + "".join(map(str, source["folds"])) if family == "gong" and source.get("folds") != [0, 1] else ""
                shifted = OUT / "maps" / f"{family}{coverage}_dy{shift}" / map_kind
                shifted.mkdir(parents=True, exist_ok=True)
                for input_path in sorted(probability.glob("*.png")):
                    target = shifted / input_path.name
                    if not target.exists():
                        value = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
                        value = cv2.warpAffine(value.astype(np.float32), np.array([[1, 0, 0], [0, 1, shift]], np.float32), (2048, 2048), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        assert cv2.imwrite(str(target), np.rint(value).astype(np.uint8))
                source["raw_" + map_kind], source[map_kind] = str(probability), str(shifted)
                source["dy"] = shift
    destination = OUT / name / "bank"
    if (destination / "manifest.json").exists():
        _, _, existing = read_bank(destination)
        assert existing["include"] == include and existing["shift"] == shift
        assert {k: v.get("marker_fingerprint", v["marker_sha256"]) for k, v in existing["sources"].items()} == {k: v.get("marker_fingerprint", v["marker_sha256"]) for k, v in sources.items()}, "Source markers changed; use a new run name"
        return
    tables, banks, manifests = {}, {}, {}
    for family in include:
        for split, map_kind in (("train", "oof"), ("test", "test")):
            if sources[family].get("bank"):
                bank_path = Path(sources[family]["bank" if split == "train" else "test_bank"])
                tables[family, split], banks[family, split], manifests[family, split] = read_bank(bank_path)
                continue
            prob = Path(sources[family][map_kind])
            coverage = "_folds" + "".join(map(str, sources[family]["folds"])) if family == "gong" and sources[family].get("folds") != [0, 1] else ""
            raw = OUT / "families" / f"{family}{coverage}_dy{shift}" / split
            enriched = raw.with_name(split + "_enriched")
            if not (raw / "manifest.json").exists():
                options = ["--prob-dir", prob, "--bank-dir", raw, "--split", split, "--workers", workers, "--resume"]
                if split == "train" and len(list(prob.glob("*_fg.png"))) < 707:
                    options.append("--available-only")
                run_script("l2_build_bank.py", options, OUT / f"{family}_dy{shift}_{split}_build.log")
            if not (enriched / "manifest.json").exists():
                run_script("l2_add_scorer_features.py", ["--bank-dir", raw, "--output-bank", enriched, "--workers", workers], OUT / f"{family}_dy{shift}_{split}_enrich.log")
            tables[family, split], banks[family, split], manifests[family, split] = read_bank(enriched)
    for split in ("train", "test"):
        stems = sorted(data.cached_records()) if split == "train" else data.list_test_stems()
        combined, rows = {}, []
        for stem in stems:
            entry = dict(rles=[], thresholds=[], matrices=[], record_ids=[])
            if split == "train":
                recs = data.cached_records()[stem]
                entry["record_ids"] = [r.record_id for r in recs]
                entry["matrices"] = [np.empty((len(r.instances), 0)) for r in recs]
            offset = 0
            for source_id, family in enumerate(include):
                frame = tables[family, split]
                frame = frame.loc[frame.stem == stem].copy().sort_values("id")
                if stem not in banks[family, split]:
                    continue
                part = banks[family, split][stem]
                if split == "train":
                    assert entry["record_ids"] == part["record_ids"], (family, stem)
                    entry["matrices"] = [np.concatenate([a, b], axis=1) for a, b in zip(entry["matrices"], part["matrices"])]
                frame["id"] += offset
                frame["parent_a"] = np.where(frame.parent_a >= 0, frame.parent_a + offset, -1)
                frame["source_name"] = family
                frame["source_family"] = source_id
                frame["selection_family"] = source_id * 16 + frame.family
                frame["in_a"] = frame.in_a & (source_id == 0)
                rows.append(frame)
                entry["rles"].extend(part["rles"])
                entry["thresholds"].extend(part["thresholds"])
                offset += len(part["rles"])
            combined[stem] = entry
        table = pd.concat(rows, ignore_index=True).sort_values(["stem", "id"]).reset_index(drop=True)
        global BANK, TABLES, RECORDS, SOURCES
        BANK, TABLES, RECORDS, SOURCES = combined, {s: f for s, f in table.groupby("stem", sort=False)}, data.cached_records() if split == "train" else {}, sources
        added, graphs = [], {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (stem, features, groups) in enumerate(pool.map(support_one, stems), 1):
                added.extend(features)
                graphs[stem] = groups
                if n % 100 == 0:
                    print("support", name, split, n, flush=True)
        extra = pd.DataFrame(added).set_index(["stem", "id"])
        table = table.join(extra, on=["stem", "id"], validate="one_to_one")
        path = OUT / name / ("bank" if split == "train" else "test_bank")
        path.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path / "features.parquet", index=False)
        save(path / "rles.pkl", combined)
        save(path / "groups.pkl", graphs)
        features = list(dict.fromkeys([col for family in include for col in manifests[family, split]["feature_columns"]] + ["source_family", "selection_family"] + list(extra.columns)))
        assert set(features) <= set(table.columns), "Declared family features are missing from its table"
        forbidden = {"stem", "id", "fold", "rater_ious", "ubar", "hit", "miou", "source_name"}
        assert not set(features) & forbidden
        assert not [col for col in features if any(token in col.lower() for token in ("station", "timestamp", "annotator", "rater"))]
        dump(path / "manifest.json", dict(**source_receipt(), split=split, sources=sources, include=include, shift=shift,
             feature_columns=features, images=len(stems), stems=stems, candidates=len(table), evaluation_identity="EV1" if split == "train" else None,
             features_sha256=sha(path / "features.parquet"), rles_sha256=sha(path / "rles.pkl"),
             family_inputs={k: manifests[k, split] for k in include}, groups_sha256=sha(path / "groups.pkl")))


def estimator(objective="regression", workers=4, seed=261009, trees=600, prior=False):
    return lgb.LGBMRegressor(objective=objective, n_estimators=trees, learning_rate=.03,
        num_leaves=15, min_child_samples=160 if prior else 320, reg_lambda=10 if prior else 40,
        colsample_bytree=.9, random_state=seed, n_jobs=workers, verbosity=-1, deterministic=True, force_col_wise=True)


def fit_prior(table, features, excluded, path, workers):
    if path.exists():
        return pickle.load(path.open("rb"))
    training = table.loc[~table.fold.isin(excluded)]
    weight = training.rater_ious.map(len).to_numpy(float)
    hit = estimator("cross_entropy", workers, trees=250, prior=True).fit(training[features], training.hit, sample_weight=weight)
    positive = training.hit > 0
    quality = estimator(workers=workers, trees=250, prior=True).fit(training.loc[positive, features], training.loc[positive, "miou"], sample_weight=weight[positive] * training.loc[positive, "hit"])
    model = dict(hit=hit, quality=quality, excluded=list(excluded), train_stems=sorted(training.stem.unique()))
    save(path, model)
    return model


def prior_predict(model, table, features):
    return np.column_stack([np.clip(model["hit"].predict(table[features]), 0, 1), np.clip(model["quality"].predict(table[features]), .5, 1)])


def allocate(table, utility, ranking, groups, alpha=.25):
    result = np.asarray(utility).copy()
    for stem, frame in table.groupby("stem", sort=False):
        indices = frame.index.to_numpy()
        group = groups[stem]
        for group_id in np.unique(group):
            keep = indices[group == group_id]
            values = ranking[keep] - ranking[keep].max()
            weights = np.exp(values)
            assigned = np.clip(weights / weights.sum() * result[keep].sum(), 0, 1)
            result[keep] = (1-alpha) * result[keep] + alpha * assigned
    return result


def fit(name, workers, members=5, temporal=False):
    path = OUT / name
    table, manifest = load(path / "bank")
    features = manifest["feature_columns"]
    groups = pickle.load((path / "bank/groups.pkl").open("rb"))
    if temporal:
        mapping = pd.read_csv(REPO / "configs/folds_v2_temporal.csv").set_index("stem").fold
        table["fold"] = table.stem.map(mapping).astype(int)
    kind = "temporal_fit" if temporal else "fit"
    target = path / kind
    target.mkdir(parents=True, exist_ok=True)
    result = table[["stem", "id", "fold", "selection_family"]].copy()
    feature_sets = [features] * 3 + [sorted(np.random.default_rng(seed).choice(features, max(1, int(.8*len(features))), replace=False)) for seed in (261017, 261029)]
    seeds = [261009, 261017, 261029, 261017, 261029]
    for fold in range(5):
        train = table.loc[table.fold != fold].copy()
        valid = table.loc[table.fold == fold].copy()
        nested = np.zeros((len(train), 2))
        for inner in range(5):
            if inner == fold:
                continue
            excluded = sorted([fold, inner])
            prior = fit_prior(table, features, excluded, target / f"prior_pair_{excluded[0]}_{excluded[1]}.pkl", workers)
            keep = train.fold.to_numpy() == inner
            assert not set(train.loc[keep, "stem"]) & set(prior["train_stems"])
            nested[keep] = prior_predict(prior, train.loc[keep], features)
        prior = fit_prior(table, features, [fold], target / f"prior_outer_{fold}.pkl", workers)
        assert not set(valid.stem) & set(prior["train_stems"])
        train[["critic_hit", "critic_miou"]] = nested
        valid[["critic_hit", "critic_miou"]] = prior_predict(prior, valid, features)
        weight = train.rater_ious.map(len).to_numpy(float)
        hit, ubar = train.hit.to_numpy(copy=True), train.ubar.to_numpy(copy=True)
        # Weak EB uses rater counts only to form supervised targets and weights.
        multi = weight > 1
        p = nested[:, 0]
        residual = np.mean((hit[multi]-p[multi])**2)
        sampling = np.mean(p[multi]*(1-p[multi])/weight[multi])
        precision = .5 * np.clip(np.mean(p[multi]*(1-p[multi])) / max(residual-sampling, .01) - 1, .25, 2)
        single = weight == 1
        hit[single] = (hit[single] + precision*p[single]) / (1+precision)
        ubar[single] = (ubar[single] + precision*p[single]*nested[single, 1]) / (1+precision)
        quality = ubar / np.maximum(hit, 1e-8)
        all_features = features + ["critic_hit", "critic_miou"]
        rank_path = target / f"rank_{fold}.pkl"
        if rank_path.exists():
            ranker = pickle.load(rank_path.open("rb"))
        else:
            queries = train.stem.astype(str) + "_" + np.concatenate([groups[s] for s in train.stem.unique()]).astype(str)
            ordered = train.assign(query=queries).sort_values(["query", "id"])
            ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=.03, num_leaves=15,
                min_child_samples=160, reg_lambda=10, colsample_bytree=.9, label_gain=list(range(21)), lambdarank_truncation_level=20,
                random_state=261009, n_jobs=workers, verbosity=-1, deterministic=True, force_col_wise=True)
            ranker.fit(ordered[all_features], np.rint(ordered.ubar.to_numpy()*20).astype(int), group=ordered.groupby("query", sort=False).size().to_numpy())
            save(rank_path, ranker)
        ranking = ranker.predict(valid[all_features])
        result.loc[valid.index, "rank"] = ranking
        for member in range(members):
            model_path = target / f"member{member}_fold{fold}.pkl"
            selected_features = feature_sets[member] + ["critic_hit", "critic_miou"]
            if model_path.exists():
                model = pickle.load(model_path.open("rb"))
            else:
                model = dict(features=selected_features, precision=float(precision), train_stems=sorted(train.stem.unique()))
                for head, objective, labels, weights, keep in (
                    ("direct", "regression", ubar, weight, np.ones(len(train), bool)),
                    ("hit", "cross_entropy", hit, weight, np.ones(len(train), bool)),
                    ("quality", "regression", quality, weight*hit, hit > 0)):
                    model[head] = estimator(objective, workers, seeds[member]).fit(train.loc[keep, selected_features], labels[keep], sample_weight=weights[keep])
                save(model_path, model)
            assert not set(valid.stem) & set(model["train_stems"])
            x = valid[model["features"]]
            result.loc[valid.index, f"direct_{member}"] = np.clip(model["direct"].predict(x), 0, 1)
            result.loc[valid.index, f"hurdle_{member}"] = np.clip(model["hit"].predict(x), 0, 1) * np.clip(model["quality"].predict(x), .5, 1)
        result.to_parquet(target / "scores_partial.parquet", index=False)
        print("fit", name, kind, "fold", fold, "members", members, flush=True)
    for member in range(members):
        result[f"ranked_{member}"] = allocate(table, result[f"hurdle_{member}"].to_numpy(), result["rank"].to_numpy(), groups)
    result["baseline"] = result.ranked_0
    result["family_mean"] = result[["direct_0", "hurdle_0", "ranked_0"]].mean(axis=1)
    result["ensemble"] = result[[f"{head}_{member}" for member in range(members) for head in ("direct", "hurdle", "ranked")]].mean(axis=1)
    # Outer calibration sees only other-fold labels; no test metadata is read.
    for column in ("baseline", "family_mean", "ensemble"):
        result[column + "_cal"] = np.nan
        for fold in range(5):
            calibrators = {}
            for family in sorted(table.selection_family.unique()):
                keep = (table.fold != fold) & (table.selection_family == family)
                valid = (table.fold == fold) & (table.selection_family == family)
                if keep.sum() < 80:
                    result.loc[valid, column + "_cal"] = result.loc[valid, column]
                    continue
                model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(result.loc[keep, column], table.loc[keep, "ubar"], sample_weight=table.loc[keep, "rater_ious"].map(len))
                calibrators[int(family)] = model
                if valid.any():
                    result.loc[valid, column + "_cal"] = model.predict(result.loc[valid, column])
            save(target / f"calibration_{column}_{fold}.pkl", calibrators)
    assert np.isfinite(result.drop(columns=["stem"]).to_numpy()).all()
    result.to_parquet(target / "scores.parquet", index=False)
    dump(target / "fit.json", dict(**source_receipt(), members=members, features=all_features, bank_sha256=manifest["rles_sha256"],
         folds_sha256=sha(REPO / ("configs/folds_v2_temporal.csv" if temporal else "configs/folds_v1.csv")),
         images_evaluated=707, images_with_candidates=int(table.stem.nunique()), nested_prior=True,
         limitation="Dense OOF is fixed v1; pooled policy selection and stacked calibration are not a fully nested pipeline."))


def stats_for(entry, ids):
    total = metric.PQStats()
    for matrix in entry["matrices"]:
        total.add(metric.record_stats(matrix[:, ids]))
    return total


def summary(stats):
    totals = {key: metric.PQStats() for key in ("all", "single", "multi")}
    for stem, stat in stats.items():
        totals["all"].add(stat)
        totals["single" if len(RECORDS[stem]) == 1 else "multi"].add(stat)
    return {k: v.as_dict() for k, v in totals.items()}


def snapshot(path):
    """Pin a handoff before a later policy or ensemble updates its live file."""
    path = Path(path)
    digest = sha(path)
    target = path.with_name(path.stem + "_" + digest[:12] + path.suffix)
    if not target.exists():
        os.link(path, target)
    return str(target), digest


def publish_bank(name, kind, manifest, best, phase):
    path = OUT / name
    predictions, digest = snapshot(path / kind / "best_predictions.pkl")
    ids, ids_digest = snapshot(path / kind / "best_ids.pkl")
    marker = dict(status="READY", **source_receipt(), stage=name, phase=phase, bank=str(path / "bank"),
        test_bank=str(path / "test_bank"), candidate_table=str(path / "bank/features.parquet"), scorer=str(path / kind),
        oof_selections=predictions, oof_ids=ids, oof_ids_sha256=ids_digest, images=707, candidates=manifest["candidates"],
        bank_sha256=manifest["rles_sha256"], candidate_table_sha256=manifest["features_sha256"],
        oof_selections_sha256=digest, metric=best["metric"], policy=best, families=manifest["include"],
        family_oof_coverage={k: v.get("oof_images") for k, v in manifest["sources"].items()},
        scorer_receipt=str(path / kind / "fit.json"), scorer_receipt_sha256=sha(path / kind / "fit.json"))
    export_path = path / kind / "export.json"
    if export_path.exists():
        exported = json.loads(export_path.read_text())
        if exported["policy"] == best:
            test_predictions, test_digest = snapshot(path / kind / "test_predictions.pkl")
            marker.update(test_predictions=test_predictions, test_predictions_sha256=test_digest, test_submission=exported["validation"])
    ready = MARKER_OUT / "bank_ready.json"
    existing = json.loads(ready.read_text()) if ready.exists() else {}
    evaluated = existing.get("evaluated_banks", {})
    evaluated[f"{name}:{kind}"] = {key: marker[key] for key in ("stage", "bank", "test_bank", "candidate_table", "scorer", "oof_selections", "oof_selections_sha256", "metric", "families")}
    marker["evaluated_banks"] = evaluated
    marker["available_families"] = sorted({family for item in evaluated.values() for family in item["families"]})
    if existing and existing["metric"]["all"]["pq"] > best["metric"]["all"]["pq"]:
        existing.update(evaluated_banks=evaluated, available_families=marker["available_families"], latest_evaluated_stage=name)
        dump(ready, existing)
    else:
        dump(ready, marker)


def select_one(stem, column, method, tau, fraction=0.0):
    entry, frame = BANK[stem], TABLES.get(stem)
    if frame is None:
        return [], [], stats_for(entry, [])
    scores = frame[column].to_numpy()
    graph = select.overlap_graph(entry["rles"])
    if method == "paint":
        ids, masks = select.paint(scores, tau, entry["rles"], min_fraction=fraction)
        st = metric.PQStats()
        for _, part in metric.score_image(masks, RECORDS[stem]):
            st.add(part)
    else:
        ids = select.greedy(scores, tau, graph, frame.in_a.to_numpy() if method == "old_a" else None)
        if method == "replace3":
            ids = select.replace_k(scores, tau, graph, initial=ids)
        masks, st = [entry["rles"][i] for i in ids], stats_for(entry, ids)
    return ids, masks, st


def painting_one(stem):
    entry, frame = BANK[stem], TABLES.get(stem)
    decoded, results = {}, []
    for column, tau, fraction in PAINT_PLANS:
        if frame is None:
            ids, masks = [], []
        else:
            ids, masks = select.paint(frame[column].to_numpy(), tau, entry["rles"], min_fraction=fraction, decoded=decoded)
        stat = metric.PQStats()
        for _, part in metric.score_image(masks, RECORDS[stem]):
            stat.add(part)
        results.append((ids, masks, stat))
    return stem, results


def evaluate(name, temporal=False, workers=4):
    global TABLES, PAINT_PLANS
    path = OUT / name
    table, manifest = load(path / "bank")
    kind = "temporal_fit" if temporal else "fit"
    scores = pd.read_parquet(path / kind / "scores.parquet")
    assert table[["stem", "id"]].equals(scores[["stem", "id"]])
    for column in scores:
        if column not in ("stem", "id", "fold", "selection_family"):
            table[column] = scores[column]
    TABLES = {s: f for s, f in table.groupby("stem", sort=False)}
    # Cache conflicts once across the complete policy sweep.
    stems = list(BANK)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        graphs = dict(zip(stems, pool.map(select.overlap_graph, [BANK[s]["rles"] for s in stems])))
    results, best = [], None
    def measure(column, method, tau, fraction=0.):
        nonlocal best
        chosen, predictions, stats = {}, {}, {}
        for stem, entry in BANK.items():
            frame = TABLES.get(stem)
            if frame is None:
                ids, masks, stat = [], [], stats_for(entry, [])
            elif method == "paint":
                ids, masks = select.paint(frame[column].to_numpy(), tau, entry["rles"], min_fraction=fraction)
                stat = metric.PQStats()
                for _, part in metric.score_image(masks, RECORDS[stem]):
                    stat.add(part)
            else:
                ids = select.greedy(frame[column].to_numpy(), tau, graphs[stem], frame.in_a.to_numpy() if method == "old_a" else None)
                if method == "replace3":
                    ids = select.replace_k(frame[column].to_numpy(), tau, graphs[stem], ids)
                masks, stat = [entry["rles"][i] for i in ids], stats_for(entry, ids)
            chosen[stem], predictions[stem], stats[stem] = ids, masks, stat
        result = dict(column=column, method=method, tau=tau, fraction=fraction, metric=summary(stats))
        results.append(result)
        if best is None or result["metric"]["all"]["pq"] > best["metric"]["all"]["pq"]:
            best = result
            save(path / kind / "best_ids.pkl", chosen)
            save(path / kind / "best_predictions.pkl", predictions)
            save(path / kind / "best_stats.pkl", stats)
            print("BEST", name, kind, json.dumps(result), flush=True)
        return result
    for column in ("baseline", "family_mean", "ensemble", "baseline_cal", "family_mean_cal", "ensemble_cal"):
        for method in ("old_a", "all"):
            for tau in TAUS:
                measure(column, method, tau)
    strict_best = dict(best)
    measure(strict_best["column"], "replace3", strict_best["tau"])
    exact = metric.evaluate(pickle.load((path / kind / "best_predictions.pkl").open("rb")), RECORDS)
    assert abs(exact.total.pq - best["metric"]["all"]["pq"]) < 1e-10
    publish_bank(name, kind, manifest, best, "strict evaluated; painting sweep running")
    PAINT_PLANS = []
    for column in ("baseline", "ensemble", "ensemble_cal"):
        candidates_best = max([r for r in results if r["column"] == column], key=lambda r: r["metric"]["all"]["pq"])
        for tau in sorted(set([max(.08, candidates_best["tau"]-.02), candidates_best["tau"], candidates_best["tau"]+.02])):
            for fraction in (0., .65, .85):
                PAINT_PLANS.append((column, tau, fraction))
    painting = [dict(chosen={}, masks={}, stats={}) for _ in PAINT_PLANS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, (stem, values) in enumerate(pool.map(painting_one, stems), 1):
            for item, (ids, masks, stat) in zip(painting, values):
                item["chosen"][stem], item["masks"][stem], item["stats"][stem] = ids, masks, stat
            if count % 100 == 0:
                print("painting", name, count, len(stems), flush=True)
    for (column, tau, fraction), item in zip(PAINT_PLANS, painting):
        result = dict(column=column, method="paint", tau=tau, fraction=fraction, metric=summary(item["stats"]))
        results.append(result)
        if result["metric"]["all"]["pq"] > best["metric"]["all"]["pq"]:
            best = result
            save(path / kind / "best_ids.pkl", item["chosen"])
            save(path / kind / "best_predictions.pkl", item["masks"])
            save(path / kind / "best_stats.pkl", item["stats"])
            print("BEST", name, kind, json.dumps(best), flush=True)
    exact = metric.evaluate(pickle.load((path / kind / "best_predictions.pkl").open("rb")), RECORDS)
    assert abs(exact.total.pq - best["metric"]["all"]["pq"]) < 1e-10
    baseline_reference = max([r for r in results if r["column"] == "baseline" and r["method"] == "old_a"], key=lambda r: r["metric"]["all"]["pq"])
    dump(path / kind / "evaluation.json", dict(**source_receipt(), identity="EV1 full707 / canonical COCO / fixed v1 dense OOF" + (" / temporal scorer" if temporal else ""),
         images=707, records=1154, gt=8199, baseline_reference=baseline_reference, best=best, strict_best=strict_best, policies=results, delta_vs_historical_s13=best["metric"]["all"]["pq"]-.446749945))
    publish_bank(name, kind, manifest, best, "evaluated")


def validate_csv(path, expected_masks=None):
    reference = pd.read_csv(ROOT / "submissions/sub13_support_all_tau020.csv", nrows=1)
    frame = pd.read_csv(path, keep_default_na=False)
    assert list(frame.columns) == list(reference.columns)
    assert frame.filament_id.is_unique
    tests = data.list_test_stems()
    assert len(tests) == 180
    by_stem = {}
    for row in frame.itertuples(index=False):
        stem, number = row.filament_id.rsplit("_", 1)
        assert stem in tests and int(number) >= 1
        rle = dict(size=[2048, 2048], counts=row.segmentation_rle.encode("ascii"))
        decoded = mu.decode(rle)
        assert decoded.shape == (2048, 2048) and int(decoded.sum()) >= 32
        assert mu.encode(np.asfortranarray(decoded))["counts"] == rle["counts"]
        by_stem.setdefault(stem, []).append((int(number), rle))
    for stem, masks in by_stem.items():
        assert [i for i, _ in masks] == list(range(1, len(masks)+1))
        graph = select.overlap_graph([r for _, r in masks])
        assert not np.triu(graph, 1).any(), stem
    if expected_masks is not None:
        assert set(expected_masks) == set(tests)
        for stem in tests:
            actual = [r for _, r in by_stem.get(stem, [])]
            assert [r["counts"] for r in actual] == [r["counts"] for r in expected_masks[stem]], stem
    result = dict(path=str(path), sha256=sha(path), scope_images=len(tests), images_with_predictions=len(by_stem), instances=len(frame),
                  checks=["S13 header", "180-image inference scope", "unique sequential IDs", "2048x2048 COCO RLE roundtrip", "pixel-disjoint", "exact emitted RLE identity"])
    dump(Path(path).with_suffix(".validation.json"), result)
    return result


def export(name, temporal=False):
    path = OUT / name
    kind = "temporal_fit" if temporal else "fit"
    fit_receipt = json.loads((path / kind / "fit.json").read_text())
    report = json.loads((path / kind / "evaluation.json").read_text())
    best = report["best"]
    table, manifest = load(path / "test_bank", "test")
    _, _, train_manifest = read_bank(path / "bank")
    features = train_manifest["feature_columns"]
    groups = pickle.load((path / "test_bank/groups.pkl").open("rb"))
    collected = []
    for fold in range(5):
        prior = pickle.load((path / kind / f"prior_outer_{fold}.pkl").open("rb"))
        frame = table.copy()
        frame[["critic_hit", "critic_miou"]] = prior_predict(prior, frame, features)
        ranker = pickle.load((path / kind / f"rank_{fold}.pkl").open("rb"))
        rank = ranker.predict(frame[features + ["critic_hit", "critic_miou"]])
        heads = []
        for member in range(fit_receipt["members"]):
            model = pickle.load((path / kind / f"member{member}_fold{fold}.pkl").open("rb"))
            x = frame[model["features"]]
            direct = np.clip(model["direct"].predict(x), 0, 1)
            hurdle = np.clip(model["hit"].predict(x), 0, 1)*np.clip(model["quality"].predict(x), .5, 1)
            ranked = allocate(frame, hurdle, rank, groups)
            heads.append(np.column_stack([direct, hurdle, ranked]))
        utility = heads[0][:, 2] if best["column"].startswith("baseline") else heads[0].mean(1) if best["column"].startswith("family_mean") else np.mean(heads, axis=(0, 2))
        if best["column"].endswith("_cal"):
            calibrators = pickle.load((path / kind / f"calibration_{best['column'][:-4]}_{fold}.pkl").open("rb"))
            utility = select.family_utilities(utility, frame.selection_family.to_numpy(), calibrators)
        collected.append(utility)
    table["utility"] = np.mean(collected, axis=0)
    if best.get("offsets"):
        table["utility"] = select.family_utilities(table.utility.to_numpy(), table.selection_family.to_numpy(),
            offsets={int(k): float(v) for k, v in best["offsets"].items()})
    table.to_parquet(path / kind / "test_scores.parquet", index=False)
    predictions = {s: [] for s in data.list_test_stems()}
    for stem, group in table.groupby("stem", sort=False):
        rles = BANK[stem]["rles"]
        if best["method"] == "paint":
            _, masks = select.paint(group.utility.to_numpy(), best["tau"], rles, min_fraction=best["fraction"])
        else:
            graph = select.overlap_graph(rles)
            ids = select.greedy(group.utility.to_numpy(), best["tau"], graph, group.in_a.to_numpy() if best["method"] == "old_a" else None)
            if best["method"] == "replace3":
                ids = select.replace_k(group.utility.to_numpy(), best["tau"], graph, ids)
            masks = [rles[i] for i in ids]
        predictions[stem] = masks
    save(path / kind / "test_predictions.pkl", predictions)
    csv = ROOT / "submissions" / f"attack_{name}_{kind}_20261030.csv"
    rows = [dict(filament_id=f"{stem}_{i}", segmentation_rle=rle["counts"].decode("ascii")) for stem, masks in predictions.items() for i, rle in enumerate(masks, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(csv, index=False)
    validation = validate_csv(csv, predictions)
    dump(path / kind / "export.json", dict(**source_receipt(), policy=best, validation=validation, scope_stems=list(predictions),
         caveat="OOF uses one heldout dense model per image; test geometry uses mean-fold dense maps."))
    marker_path = MARKER_OUT / "bank_ready.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("scorer") == str(path / kind) and not marker.get("postselection"):
            test_predictions, test_digest = snapshot(path / kind / "test_predictions.pkl")
            marker.update(test_predictions=test_predictions, test_predictions_sha256=test_digest, test_submission=validation,
                          phase="OOF and validated test CSV ready", timestamp=datetime.now(timezone.utc).isoformat())
            dump(marker_path, marker)
    print("EXPORTED", json.dumps(validation), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["discover", "prepare", "fit", "evaluate", "export"])
    parser.add_argument("--name", default="a0")
    parser.add_argument("--include", default="long2x")
    parser.add_argument("--support", default="long2x,s05")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--shift", type=float, default=0.)
    args = parser.parse_args()
    assert 1 <= args.workers <= 6 and 1 <= args.members <= 5
    if args.command == "discover":
        print(json.dumps(discover(), indent=2))
    elif args.command == "prepare":
        prepare(args.name, args.include.split(","), args.support.split(","), args.workers, args.shift)
    elif args.command == "fit":
        fit(args.name, args.workers, args.members, args.temporal)
    elif args.command == "evaluate":
        evaluate(args.name, args.temporal, args.workers)
    elif args.command == "export":
        export(args.name, args.temporal)


if __name__ == "__main__":
    main()
