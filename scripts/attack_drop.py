"""Learn exact keep/drop utilities from pair-excluded selected-mask states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import faulthandler
import signal

import attack_actions as actions
import attack_base as base
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from solar import data, metric, select

CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SPATIAL = {}
DROP_DIR = None
TRAIN_OFFSETS = (-.06, 0., .06)
CUTOFFS = (-.08, -.04, -.02, 0., .01, .02, .04, .06, .08, .1, .15, .2)
HEADS = ("gain7", "gain15", "decomposed", "blend")


def provenance():
    return dict(**actions.receipt(), drop_code_sha256=CODE_SHA)


def configure(name, temporal=False):
    global DROP_DIR
    path, kind = actions.configure(name, temporal)
    DROP_DIR = path / (kind + "_drop")
    DROP_DIR.mkdir(parents=True, exist_ok=True)
    return path, kind


def spatial_features(stem, state):
    """Pool raw image, mask and probability only; no label-derived embedding."""
    if stem not in SPATIAL:
        train = bool(base.RECORDS)
        image = data.load_gray(data.train_image_path(stem) if train else data.test_image_path(stem)).astype(float)
        probability = cv2.imread(str(Path(base.SOURCES["long2x"]["oof" if train else "test"]) / f"{stem}_fg.png"), 0).astype(float) / 255
        SPATIAL[stem] = image, probability, {}
    image, probability, cached = SPATIAL[stem]
    result = []
    for value in state:
        key = value["rle"]["counts"]
        if key not in cached:
            mask, x, y = select.crop_decode(value["rle"])
            x0, y0 = max(0, x-32), max(0, y-32)
            x1, y1 = min(2048, x+mask.shape[1]+32), min(2048, y+mask.shape[0]+32)
            gray = image[y0:y1, x0:x1]
            prob = probability[y0:y1, x0:x1]
            own = np.zeros_like(gray)
            own[y-y0:y-y0+mask.shape[0], x-x0:x-x0+mask.shape[1]] = mask
            z = (gray-gray.mean()) / max(gray.std(), 1.)
            channels = (z, prob, own, z*own, prob*own)
            cached[key] = np.concatenate([cv2.resize(c.astype(np.float32), (4, 4), interpolation=cv2.INTER_AREA).ravel() for c in channels])
        result.append(cached[key])
    return np.asarray(result)


def features_for(stem, state, utility):
    frame = base.TABLES[stem]
    ids = [v["cid"] for v in state]
    assert all(i is not None for i in ids)
    own = frame[actions.FEATURES].to_numpy(float)[ids]
    score = np.asarray(utility)
    selected = np.asarray([v["score"] for v in state])
    common = np.r_[actions.state_context(state), np.quantile(score, [0., .25, .5, .75, .9, 1.]), score.mean(), score.std()]
    iou = np.asarray(mu.iou(base.BANK[stem]["rles"], [v["rle"] for v in state], [0]*len(state)))
    dynamic = []
    for index, cid in enumerate(ids):
        overlaps = iou[:, index] > 0
        alternate = (iou[:, index] > .5) & (np.arange(len(score)) != cid)
        dynamic.append(np.r_[selected[index], score[cid], (score < score[cid]).mean(),
            np.max(score[alternate]) if alternate.any() else 0.,
            np.mean(score[overlaps]) if overlaps.any() else 0.,
            np.sum(overlaps), selected[index]-selected.min(), common])
    return np.column_stack([own, np.asarray(dynamic), spatial_features(stem, state)])


def training_image(task):
    stem, utility = task
    SPATIAL.clear()
    features, targets, weights = [], [], []
    decoded = {}
    for offset in TRAIN_OFFSETS:
        policy = dict(actions.POLICY, tau=actions.POLICY["tau"]+offset)
        state = actions.initial(stem, utility, policy, decoded=decoded)
        if not state:
            continue
        before = actions.exact(stem, state)
        x = features_for(stem, state, utility)
        records = len(base.RECORDS[stem])
        for i in range(len(state)):
            after = actions.exact(stem, state[:i]+state[i+1:])
            di = (after.iou_sum-before.iou_sum)/records
            dd = (actions.denominator(after)-actions.denominator(before))/records
            features.append(x[i])
            targets.append((di, dd))
            weights.append(records/len(TRAIN_OFFSETS))
    return stem, np.asarray(features), np.asarray(targets), np.asarray(weights)


def training_worker(name, temporal):
    # A fresh process avoids inheriting native thread pools after parent fits.
    configure(name, temporal)
    faulthandler.register(signal.SIGUSR1)


def fit(name, workers, temporal=False):
    path, kind = configure(name, temporal)
    lineage = {}
    for outer in range(5):
        destination = DROP_DIR / f"models_{outer}.pkl"
        if destination.exists():
            continue
        training = actions.TABLE.loc[actions.TABLE.fold != outer]
        prediction = np.zeros(len(actions.TABLE))
        for inner in range(5):
            if inner == outer:
                continue
            pair = sorted([outer, inner])
            utility = actions.nested_utility(pair, path / kind, workers)
            cache = actions.DIRECTORY / f"nested_pair_{pair[0]}_{pair[1]}.parquet"
            lineage[str(cache)] = base.sha(cache)
            nested_model = pickle.load((actions.DIRECTORY / f"nested_models_{pair[0]}_{pair[1]}.pkl").open("rb"))
            assert set(nested_model["excluded"]) == set(pair)
            valid = actions.TABLE.fold == inner
            assert not set(actions.TABLE.loc[valid, "stem"]) & set(nested_model["train_stems"])
            keys = pd.MultiIndex.from_frame(actions.TABLE.loc[valid, ["stem", "id"]])
            prediction[valid] = utility.loc[keys].to_numpy()
        tasks = [(s, prediction[f.index]) for s, f in training.groupby("stem", sort=False)]
        xs, ys, ws, stems = [], [], [], []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=training_worker, initargs=(name, temporal)) as pool:
            for n, (stem, x, y, weight) in enumerate(pool.map(training_image, tasks), 1):
                if len(x):
                    xs.append(x)
                    ys.append(y)
                    ws.append(weight)
                    stems.extend([stem]*len(x))
                if n % 100 == 0:
                    print("drop samples", outer, n, sum(map(len, xs)), flush=True)
        x, y, weights = np.vstack(xs), np.vstack(ys), np.concatenate(ws)
        gain = y[:, 0]-actions.Q*y[:, 1]
        model = dict(q=actions.Q, excluded=[outer], train_stems=sorted(training.stem.unique()),
            features=actions.FEATURES, training_rows=len(x), input_columns=x.shape[1],
            target="Exact per-record deltaI-q*deltaD after deleting one selected mask; averaged over records")
        for leaves in (7, 15):
            estimator = base.estimator(workers=workers, trees=500)
            estimator.set_params(num_leaves=leaves, min_child_samples=96, reg_lambda=40)
            model[f"gain{leaves}"] = estimator.fit(x, gain, sample_weight=weights)
        for key, index in (("di", 0), ("dd", 1)):
            estimator = base.estimator(workers=workers, trees=500)
            estimator.set_params(num_leaves=7, min_child_samples=96, reg_lambda=40)
            model[key] = estimator.fit(x, y[:, index], sample_weight=weights)
        base.save(destination, model)
        np.savez_compressed(DROP_DIR / f"training_{outer}.npz", targets=y, weights=weights,
            stems=np.asarray(stems), positive_drop=gain > 0)
        print("DROP FIT", outer, len(x), "positive", int((gain > 0).sum()), flush=True)
    lineage.update({str(p): base.sha(p) for p in actions.DIRECTORY.glob("nested_pair_*.parquet")})
    base.dump(DROP_DIR / "fit.json", dict(**provenance(), identity="EV1 full707 fixed-v1 dense; nested selected-mask drop",
        outer_folds=5, training_tau_offsets=TRAIN_OFFSETS, nested_anchor_hashes=lineage,
        labels="Exact shared-output per-rater drop delta; rater count is target/weight only",
        limitation="Dense OOF remains fixed v1; policy and q use pooled OOF. No test metadata features."))


def predict(model, x, head):
    if head == "decomposed":
        return model["di"].predict(x, num_threads=1)-model["q"]*model["dd"].predict(x, num_threads=1)
    if head == "blend":
        return (predict(model, x, "gain7")+predict(model, x, "gain15"))/2
    return model[head].predict(x, num_threads=1)


def oof_image(stem):
    SPATIAL.clear()
    state = actions.initial(stem)
    if not state:
        return stem, state, {h: np.empty(0) for h in HEADS}
    frame = base.TABLES[stem]
    fold = int(frame.fold.iloc[0])
    model = pickle.load((DROP_DIR / f"models_{fold}.pkl").open("rb"))
    assert stem not in model["train_stems"] and model["excluded"] == [fold]
    x = features_for(stem, state, frame.utility.to_numpy())
    return stem, state, {h: predict(model, x, h) for h in HEADS}


def sweep_image(task):
    stem, state, predicted = task
    masks, stats, deleted = {}, {}, {}
    for head in HEADS:
        for cutoff in CUTOFFS:
            key = f"{head}:{cutoff}"
            keep = predicted[head] <= cutoff
            result = [v for v, accepted in zip(state, keep) if accepted]
            masks[key] = [v["rle"] for v in result]
            stats[key] = actions.exact(stem, result)
            deleted[key] = int((~keep).sum())
    return stem, masks, stats, deleted


def evaluate(name, workers, temporal=False):
    configure(name, temporal)
    cache = DROP_DIR / "oof_drop_scores.pkl"
    if cache.exists():
        results = pickle.load(cache.open("rb"))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, value in enumerate(pool.map(oof_image, sorted(base.BANK)), 1):
                results.append(value)
                if n % 100 == 0:
                    print("drop predict", n, flush=True)
        base.save(cache, results)
    masks, stats, deleted = {}, {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for stem, ms, ss, dd in pool.map(sweep_image, results):
            for key in ms:
                masks.setdefault(key, {})[stem] = ms[key]
                stats.setdefault(key, {})[stem] = ss[key]
                deleted[key] = deleted.get(key, 0)+dd[key]
    trials = {k: dict(metric=base.summary(v), deleted_masks=deleted[k]) for k, v in stats.items()}
    trials["seed"] = dict(metric=actions.POLICY["metric"], deleted_masks=0)
    best = max(trials, key=lambda k: trials[k]["metric"]["all"]["pq"])
    if best == "seed":
        predictions = {s: [v["rle"] for v in state] for s, state, _ in results}
    else:
        predictions = masks[best]
    native = metric.evaluate(predictions, base.RECORDS).summary()
    assert abs(native["all"]["pq"]-trials[best]["metric"]["all"]["pq"]) < 1e-10
    base.save(DROP_DIR / "best_predictions.pkl", predictions)
    base.dump(DROP_DIR / "evaluation.json", dict(**provenance(), identity="EV1 full707 / 1154 records / 8199GT",
        best=best, metric=native, baseline=actions.POLICY["metric"], trials=trials,
        delta_all=native["all"]["pq"]-actions.Q,
        policy_selection="One global head and cutoff selected on pooled OOF; not fully nested"))
    print("DROP BEST", best, json.dumps(native), flush=True)
    print("DROP HEADS", {h: max(v["metric"]["all"]["pq"] for k, v in trials.items() if k.startswith(h+":")) for h in HEADS}, flush=True)


def export(name, temporal=False):
    path, kind = configure(name, temporal)
    report = json.loads((DROP_DIR / "evaluation.json").read_text())
    best = report["best"]
    if best == "seed":
        exported = json.loads((path / kind / "export.json").read_text())
        base.dump(DROP_DIR / "export.json", dict(**provenance(), policy="seed", validation=exported["validation"]))
        return
    head, cutoff = best.split(":")
    cutoff = float(cutoff)
    actor_directory = DROP_DIR
    actions.TABLE, _ = base.load(path / "test_bank", "test")
    scores = pd.read_parquet(path / kind / "test_scores.parquet")
    assert actions.TABLE[["stem", "id"]].equals(scores[["stem", "id"]])
    actions.TABLE["utility"] = scores.utility
    base.TABLES = {s: f for s, f in actions.TABLE.groupby("stem", sort=False)}
    models = [pickle.load((actor_directory / f"models_{fold}.pkl").open("rb")) for fold in range(5)]
    predictions = {}
    for stem in data.list_test_stems():
        SPATIAL.clear()
        state = actions.initial(stem)
        if state:
            x = features_for(stem, state, base.TABLES[stem].utility.to_numpy())
            gain = np.mean([predict(m, x, head) for m in models], axis=0)
            state = [v for v, accepted in zip(state, gain <= cutoff) if accepted]
        predictions[stem] = [v["rle"] for v in state]
    base.save(actor_directory / "test_predictions.pkl", predictions)
    csv = base.ROOT / "submissions" / f"attack_{name}_{kind}_drop_20261030.csv"
    rows = [dict(filament_id=f"{s}_{i}", segmentation_rle=r["counts"].decode("ascii")) for s, ms in predictions.items() for i, r in enumerate(ms, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(csv, index=False)
    validation = base.validate_csv(csv, predictions)
    base.dump(actor_directory / "export.json", dict(**provenance(), policy=best, validation=validation))
    publish(name, kind, report, validation)


def publish(name, kind, report, validation):
    """Keep the node-score handoff contract while publishing the stacked masks."""
    path = base.OUT / name
    actor = path / (kind + "_drop")
    manifest = json.loads((path / "bank/manifest.json").read_text())
    seed_masks = pickle.load((path / kind / "best_predictions.pkl").open("rb"))
    seed_ids = pickle.load((path / kind / "best_ids.pkl").open("rb"))
    chosen = pickle.load((actor / "best_predictions.pkl").open("rb"))
    ids = {}
    for stem, masks in chosen.items():
        lookup = {r["counts"]: i for r, i in zip(seed_masks[stem], seed_ids[stem])}
        ids[stem] = [lookup[r["counts"]] for r in masks]
    base.save(actor / "best_ids.pkl", ids)
    ready = base.MARKER_OUT / "bank_ready.json"
    existing = json.loads(ready.read_text()) if ready.exists() else {}
    marker = dict(status="READY", **provenance(), stage=name, phase="Stacked keep/drop OOF and validated test CSV ready",
        bank=str(path / "bank"), test_bank=str(path / "test_bank"), candidate_table=str(path / "bank/features.parquet"),
        bank_sha256=manifest["rles_sha256"], candidate_table_sha256=manifest["features_sha256"],
        images=707, candidates=manifest["candidates"], families=manifest["include"],
        family_oof_coverage={k: v.get("oof_images") for k, v in manifest["sources"].items()},
        scorer=str(path / kind), scorer_receipt=str(path / kind / "fit.json"),
        scorer_receipt_sha256=base.sha(path / kind / "fit.json"), policy=actions.POLICY,
        metric=report["metric"], test_submission=validation,
        postselection=dict(kind="selected_mask_drop", policy=report["best"], scorer=str(actor),
            receipt=str(actor / "fit.json"), receipt_sha256=base.sha(actor / "fit.json")))
    for field, artifact in (("oof_selections", "best_predictions.pkl"), ("oof_ids", "best_ids.pkl"), ("test_predictions", "test_predictions.pkl")):
        marker[field], marker[field + "_sha256"] = base.snapshot(actor / artifact)
    evaluated = existing.get("evaluated_banks", {})
    evaluated[f"{name}:{kind}_drop"] = {k: marker[k] for k in ("stage", "bank", "test_bank", "candidate_table", "scorer", "oof_selections", "oof_selections_sha256", "metric", "families")}
    marker["evaluated_banks"] = evaluated
    marker["available_families"] = sorted({f for value in evaluated.values() for f in value["families"]})
    if not existing or marker["metric"]["all"]["pq"] >= existing["metric"]["all"]["pq"]:
        base.dump(ready, marker)
        base.dump(base.MARKER_OUT / "attack_base_bundle_ready.json", marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["fit", "evaluate", "export"])
    parser.add_argument("--name", default="a4_gong01_tau")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args.name, args.temporal)
    else:
        globals()[args.command](args.name, args.workers, args.temporal)
