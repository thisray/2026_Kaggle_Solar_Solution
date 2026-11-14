"""Expanded native-RLE capacity and fold-safe learned set-action utilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
import json
import hashlib
from pathlib import Path
import pickle
import sys
import time

import attack_base as base
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy.spatial import cKDTree
from solar import candidates, data, metric, select

TABLE, FEATURES, SCORES, POLICY, DIRECTORY = None, [], {}, {}, None
Q = .446749945
CAPACITY_PREVIOUS = {}
ACTION_CUTOFFS = (0., .01, .025, .05, .075, .1, .15)
ACTION_KEYS = ["seed"] + [("c2_merge" if merge else "c2") + f"_cut{cutoff}" for merge in (False, True) for cutoff in ACTION_CUTOFFS]
RUN_CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GT_CACHE_KEY, GT_COLUMNS = None, {}


def receipt():
    return dict(**base.source_receipt(), action_code_sha256=RUN_CODE_SHA)


def denominator(stat):
    return stat.tp + .5 * (stat.fp + stat.fn)


def exact(stem, state):
    global GT_CACHE_KEY, GT_COLUMNS
    entry = base.BANK[stem]
    key = stem, id(entry)
    if key != GT_CACHE_KEY:
        GT_CACHE_KEY = key
        GT_COLUMNS = {rle["counts"]: [matrix[:, cid] for matrix in entry["matrices"]] for cid, rle in enumerate(entry["rles"])}
    columns = []
    for value in state:
        rle = value["rle"]
        if rle["counts"] not in GT_COLUMNS:
            GT_COLUMNS[rle["counts"]] = [metric.iou_matrix([i.rle for i in rec.instances], [rle])[:, 0] for rec in base.RECORDS[stem]]
        columns.append(GT_COLUMNS[rle["counts"]])
    total = metric.PQStats()
    for ri, rec in enumerate(base.RECORDS[stem]):
        matrix = np.column_stack([value[ri] for value in columns]) if columns else np.empty((len(rec.instances), 0))
        total.add(metric.record_stats(matrix))
    return total


def validate_exact(stem, state):
    cached, native = exact(stem, state), metric.PQStats()
    for _, stat in metric.score_image([v["rle"] for v in state], base.RECORDS[stem]):
        native.add(stat)
    assert (cached.tp, cached.fp, cached.fn) == (native.tp, native.fp, native.fn)
    assert abs(cached.iou_sum-native.iou_sum) < 1e-9
    return native


def initial(stem, utility=None, policy=None, decoded=None):
    frame = base.TABLES.get(stem)
    if frame is None:
        return []
    scores = frame["utility"].to_numpy() if utility is None else utility
    policy = policy or POLICY
    rles = base.BANK[stem]["rles"]
    if policy["method"] == "paint":
        ids, masks = select.paint(scores, policy["tau"], rles, min_fraction=policy["fraction"], decoded=decoded)
    else:
        graph = select.overlap_graph(rles)
        ids = select.greedy(scores, policy["tau"], graph, frame.in_a.to_numpy() if policy["method"] == "old_a" else None)
        if policy["method"] == "replace3":
            ids = select.replace_k(scores, policy["tau"], graph, ids)
        masks = [rles[i] for i in ids]
    return [dict(rle=r, score=float(scores[i]), cid=i) for i, r in zip(ids, masks)]


@lru_cache(maxsize=8192)
def roi(counts, size):
    return select.crop_decode(dict(counts=counts, size=list(size)))


def crop(rle):
    return roi(rle["counts"], tuple(rle["size"]))


@lru_cache(maxsize=8192)
def object_geometry(counts, size):
    mask, x, y = roi(counts, size)
    yy, xx = np.where(mask)
    xy = np.column_stack([xx+x, yy+y]).astype(float)
    if len(xy) < 3:
        axis, width = np.array([1., 0.]), 1.
    else:
        values, vectors = np.linalg.eigh(np.cov(xy.T, bias=True))
        axis, width = vectors[:, -1], max(1., 4*np.sqrt(max(0., values[0])))
    return xy, axis, width, cKDTree(xy)


def geometry(rle):
    return object_geometry(rle["counts"], tuple(rle["size"]))


def clear_geometry():
    roi.cache_clear()
    object_geometry.cache_clear()


def state_context(state):
    if not state:
        return np.zeros(6)
    scores = np.array([v["score"] for v in state])
    return np.array([len(state), scores.sum(), scores.max(), scores.mean(), scores.std(), sum(float(mu.area(v["rle"])) for v in state)])


def propose(stem, state, scores=None, restricted=False, oracle=False, allow_drop=True):
    """All families enter clip-add, replace-k and adjacent pair-merge actions."""
    frame = base.TABLES.get(stem)
    if frame is None:
        return []
    scores = frame.utility.to_numpy() if scores is None else scores
    rles = base.BANK[stem]["rles"]
    current = [v["rle"] for v in state]
    iou = np.asarray(mu.iou(rles, current, [0]*len(current))) if current else np.zeros((len(rles), 0))
    claimed = np.zeros((2048, 2048), bool)
    for rle in current:
        mask, x, y = crop(rle)
        claimed[y:y+mask.shape[0], x:x+mask.shape[1]] |= mask
    actions = []
    image = None if oracle else data.load_gray(data.train_image_path(stem) if base.RECORDS else data.test_image_path(stem))
    probability_path = Path(base.SOURCES["long2x"]["oof" if base.RECORDS else "test"]) / f"{stem}_fg.png"
    probability = None if oracle else cv2.imread(str(probability_path), cv2.IMREAD_GRAYSCALE).astype(float) / 255
    context = state_context(state)
    own_features = frame[FEATURES].to_numpy(float) if not oracle else None

    def add(kind, removed, new_rle, cid=None, link=None):
        # Labels, raters, record IDs and file metadata never enter this vector.
        score = float(scores[cid]) if cid is not None else sum(state[i]["score"] for i in removed)
        if oracle:
            actions.append(dict(kind=kind, removed=removed, rle=new_rle, cid=cid, score=score))
            return
        own = own_features[cid] if cid is not None else np.mean([own_features[state[i]["cid"]] for i in removed if state[i]["cid"] is not None], axis=0) if any(state[i]["cid"] is not None for i in removed) else np.zeros(len(FEATURES))
        removed_state = [state[i] for i in removed]
        removed_context = state_context(removed_state)
        geom = np.zeros(10)
        if new_rle is not None:
            mask, x, y = crop(new_rle)
            pix = image[y:y+mask.shape[0], x:x+mask.shape[1]][mask]
            prob = probability[y:y+mask.shape[0], x:x+mask.shape[1]][mask]
            old_area = float(mu.area(rles[cid])) if cid is not None else sum(float(mu.area(v["rle"])) for v in removed_state)
            geom = np.array([mask.sum(), mask.shape[0], mask.shape[1], mask.sum()/max(old_area, 1), pix.mean(), pix.std(), prob.mean(), prob.max(), scores[cid] if cid is not None else sum(v["score"] for v in removed_state), len(removed)])
        feature = np.r_[own, context, removed_context, geom, [float(kind == k) for k in range(5)], np.zeros(9) if link is None else link]
        actions.append(dict(kind=kind, removed=removed, rle=new_rle, cid=cid, score=score, features=feature))

    if not restricted and allow_drop:
        for i in range(len(state)):
            add(0, [i], None)
    for cid, rle in enumerate(rles):
        if not oracle and scores[cid] < .03:
            continue
        if float(mu.area(rle)) < 32:
            continue
        if any(v["rle"]["counts"] == rle["counts"] for v in state):
            continue
        removed = np.flatnonzero(iou[cid] > 0).tolist()
        if len(removed) <= (1 if restricted else 3):
            add(1, removed, rle, cid)
        if removed and not restricted:
            mask, x, y = crop(rle)
            remaining = mask & ~claimed[y:y+mask.shape[0], x:x+mask.shape[1]]
            if remaining.sum() >= 32:
                add(2, [], candidates.crop_rle(remaining, x, y, (2048, 2048)), cid)
    if restricted:
        return actions
    for i, j in combinations(range(len(state)), 2):
        a, b = current[i], current[j]
        ax, axis_a, width_a, tree_a = geometry(a)
        bx, axis_b, width_b, tree_b = geometry(b)
        if not len(ax) or not len(bx):
            continue
        # Full pixel coordinates retain very narrow filament contacts.
        distances, indices = tree_a.query(bx)
        k = int(np.argmin(distances))
        gap = float(distances[k])
        if gap > 48:
            continue
        p, q = ax[indices[k]], bx[k]
        direction = (q-p) / max(np.linalg.norm(q-p), 1)
        line = np.rint(np.linspace(p, q, max(2, int(np.ceil(gap))+1))).astype(int)
        lp = np.zeros(len(line)) if oracle else probability[line[:, 1], line[:, 0]]
        lg = np.zeros(len(line)) if oracle else image[line[:, 1], line[:, 0]].astype(float)
        link = np.array([gap, abs(axis_a @ axis_b), min(abs(axis_a @ direction), abs(axis_b @ direction)), min(width_a, width_b)/max(width_a, width_b), lp.mean(), lp.min(), lg.mean(), lg.std(), min(width_a, width_b)])
        merged = mu.merge([a, b])
        add(3, [i, j], merged, link=link)
        mask, x, y = crop(merged)
        pad = 8
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(2048, x+mask.shape[1]+pad), min(2048, y+mask.shape[0]+pad)
        bridge = np.zeros((y1-y0, x1-x0), np.uint8)
        bridge[y-y0:y-y0+mask.shape[0], x-x0:x-x0+mask.shape[1]] = mask
        cv2.line(bridge, tuple((p-[x0, y0]).astype(int)), tuple((q-[x0, y0]).astype(int)), 1, max(1, min(8, int(min(width_a, width_b)/2))))
        protected = claimed[y0:y1, x0:x1].copy()
        for rle in (a, b):
            m, xx, yy = crop(rle)
            protected[yy-y0:yy-y0+m.shape[0], xx-x0:xx-x0+m.shape[1]] &= ~m
        bridge[protected] = 0
        new_rle = candidates.crop_rle(bridge, x0, y0, (2048, 2048))
        if new_rle["counts"] != merged["counts"]:
            add(4, [i, j], new_rle, link=link)
    return actions


def apply(state, action, gain=None):
    after = [v for i, v in enumerate(state) if i not in action["removed"]]
    if action["rle"] is not None:
        score = action["score"]
        after.append(dict(rle=action["rle"], cid=action["cid"], score=score))
    return after


def fragments(stem, state):
    counts = dict(fn=0, covered_by_two=0, union_can_hit=0)
    for record in base.RECORDS[stem]:
        gts = [i.rle for i in record.instances]
        iou = metric.iou_matrix(gts, [v["rle"] for v in state])
        for gi, gt in enumerate(gts):
            if (iou[gi] > .5).any():
                continue
            counts["fn"] += 1
            parts = np.flatnonzero((iou[gi] > 0) & (iou[gi] < .5))
            if len(parts) >= 2:
                counts["covered_by_two"] += 1
                for a, b in combinations(parts, 2):
                    union = mu.merge([state[a]["rle"], state[b]["rle"]])
                    if float(mu.iou([union], [gt], [0])[0, 0]) > .5:
                        counts["union_can_hit"] += 1
                        break
    return counts


def capacity_one(stem):
    clear_geometry()
    seed = initial(stem)
    original = exact(stem, seed)
    result = dict(CAPACITY_PREVIOUS.get(stem, dict(stem=stem, original=original, fragments=fragments(stem, seed))))
    for mode in ("restricted", "expanded_nodrop", "expanded"):
        if mode in result:
            continue
        state = list(seed)
        trace = []
        for step in range(10):
            before = exact(stem, state)
            actions = propose(stem, state, restricted=mode == "restricted", oracle=True, allow_drop=mode != "expanded_nodrop")
            best = None
            for action in actions:
                after = exact(stem, apply(state, action))
                gain = after.iou_sum-before.iou_sum-Q*(denominator(after)-denominator(before))
                if gain > 1e-10 and (best is None or gain > best[0]):
                    best = gain, action, after
            if best is None:
                break
            state = apply(state, best[1])
            trace.append(dict(step=step+1, kind=best[1]["kind"], removed=len(best[1]["removed"]), gain=best[0]))
        rles = [v["rle"] for v in state]
        assert not np.triu(select.overlap_graph(rles), 1).any()
        result[mode] = validate_exact(stem, state)
        result[mode + "_trace"] = trace
    return result


def configure(name, temporal=False):
    global TABLE, FEATURES, SCORES, POLICY, DIRECTORY, Q
    path = base.OUT / name
    kind = "temporal_fit" if temporal else "fit"
    TABLE, manifest = base.load(path / "bank")
    FEATURES = manifest["feature_columns"]
    report = json.loads((path / kind / "evaluation.json").read_text())
    POLICY = report["best"]
    Q = POLICY["metric"]["all"]["pq"]
    scores = pd.read_parquet(path / kind / "scores.parquet")
    assert TABLE[["stem", "id"]].equals(scores[["stem", "id"]])
    TABLE["utility"] = scores[POLICY["column"]]
    if POLICY.get("offsets"):
        TABLE["utility"] = select.family_utilities(TABLE.utility.to_numpy(), TABLE.selection_family.to_numpy(),
            offsets={int(k): float(v) for k, v in POLICY["offsets"].items()})
    if temporal:
        TABLE["fold"] = scores["fold"]
    base.TABLES = {s: f for s, f in TABLE.groupby("stem", sort=False)}
    DIRECTORY = path / (kind + "_actions")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    return path, kind


def capacity(name, workers, temporal=False):
    global CAPACITY_PREVIOUS
    configure(name, temporal)
    previous = DIRECTORY / "capacity.pkl"
    lineage = None
    if previous.exists():
        lineage = base.snapshot(previous)
        CAPACITY_PREVIOUS = {v["stem"]: v for v in pickle.load(previous.open("rb"))}
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, value in enumerate(pool.map(capacity_one, sorted(base.BANK)), 1):
            results.append(value)
            if n % 25 == 0:
                print("capacity", n, 707, flush=True)
    save_path = DIRECTORY / "capacity.pkl"
    base.save(save_path, results)
    summaries = {key: base.summary({v["stem"]: v[key] for v in results}) for key in ("original", "restricted", "expanded_nodrop", "expanded")}
    count = {k: sum(v["fragments"][k] for v in results) for k in ("fn", "covered_by_two", "union_can_hit")}
    base.dump(DIRECTORY / "capacity.json", dict(**receipt(), identity="EV1 full707 train-GT-aware NONDEPLOYABLE", summaries=summaries,
        fragments=count, historical_s13_k10=.528069250, same_bank_comparison=True,
        previous_capacity=lineage,
        caveat="Historical K10 used the deleted EoMT/LVD S13 bank. Compare restricted versus expanded here on the rebuilt bank; neither is a global upper bound."))
    print("CAPACITY", json.dumps(summaries), count, flush=True)


def training_image(task):
    stem, outer, utility = task
    clear_geometry()
    # Inner predictions exclude outer and this image's fold, including labels.
    state = initial(stem, utility)
    before = exact(stem, state)
    actions = propose(stem, state, utility)
    r = len(base.RECORDS[stem])
    rows = []
    for action in actions:
        after = exact(stem, apply(state, action))
        di = (after.iou_sum-before.iou_sum) / r
        dd = (denominator(after)-denominator(before)) / r
        rows.append((action["features"], di, dd, action["kind"]))
    return stem, rows


def nested_utility(excluded, fit_directory, workers):
    """Match the strong-EB/rank recipe while excluding both heldout folds."""
    a, b = sorted(excluded)
    cache = DIRECTORY / f"nested_pair_{a}_{b}.parquet"
    if cache.exists():
        value = pd.read_parquet(cache)
        return value.set_index(["stem", "id"]).utility
    train = TABLE.loc[~TABLE.fold.isin(excluded)].copy()
    valid = TABLE.loc[TABLE.fold.isin(excluded)].copy()
    critic = np.zeros((len(train), 2))
    for inner in sorted(train.fold.unique()):
        heldout = sorted([*excluded, int(inner)])
        prior = base.fit_prior(TABLE, FEATURES, heldout, DIRECTORY / ("prior_triple_" + "_".join(map(str, heldout)) + ".pkl"), workers)
        keep = train.fold.to_numpy() == inner
        critic[keep] = base.prior_predict(prior, train.loc[keep], FEATURES)
    pair_prior = pickle.load((fit_directory / f"prior_pair_{a}_{b}.pkl").open("rb"))
    assert not set(valid.stem) & set(pair_prior["train_stems"])
    train[["critic_hit", "critic_miou"]] = critic
    valid[["critic_hit", "critic_miou"]] = base.prior_predict(pair_prior, valid, FEATURES)
    weight = train.rater_ious.map(len).to_numpy(float)
    h, u = train.hit.to_numpy(copy=True), train.ubar.to_numpy(copy=True)
    multi = weight > 1
    p = critic[:, 0]
    residual = np.mean((h[multi]-p[multi])**2)
    sampling = np.mean(p[multi]*(1-p[multi])/weight[multi])
    precision = .5*np.clip(np.mean(p[multi]*(1-p[multi])) / max(residual-sampling, .01)-1, .25, 2)
    single = weight == 1
    h[single] = (h[single]+precision*p[single]) / (1+precision)
    u[single] = (u[single]+precision*p[single]*critic[single, 1]) / (1+precision)
    q = u / np.maximum(h, 1e-8)
    features = FEATURES + ["critic_hit", "critic_miou"]
    models, predictions = {}, {}
    for key, objective, target, weights, keep in (
        ("direct", "regression", u, weight, np.ones(len(train), bool)),
        ("hit", "cross_entropy", h, weight, np.ones(len(train), bool)),
        ("quality", "regression", q, weight*h, h > 0)):
        model = base.estimator(objective, workers).fit(train.loc[keep, features], target[keep], sample_weight=weights[keep])
        models[key] = model
        predictions[key] = np.clip(model.predict(valid[features]), .5 if key == "quality" else 0, 1)
    groups = pickle.load((DIRECTORY.parent / "bank/groups.pkl").open("rb"))
    queries = train.stem.astype(str) + "_" + np.concatenate([groups[s] for s in train.stem.unique()]).astype(str)
    ordered = train.assign(query=queries).sort_values(["query", "id"])
    ranker = base.lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=.03, num_leaves=15,
        min_child_samples=160, reg_lambda=10, colsample_bytree=.9, label_gain=list(range(21)), lambdarank_truncation_level=20,
        random_state=261009, n_jobs=workers, verbosity=-1, deterministic=True, force_col_wise=True)
    ranker.fit(ordered[features], np.rint(ordered.ubar.to_numpy()*20).astype(int), group=ordered.groupby("query", sort=False).size().to_numpy())
    rank = ranker.predict(valid[features])
    hurdle = predictions["hit"]*predictions["quality"]
    ranked = base.allocate(valid.reset_index(drop=True), hurdle, rank, groups)
    utility = ranked if POLICY["column"].startswith("baseline") else (predictions["direct"]+hurdle+ranked)/3
    if POLICY.get("offsets"):
        utility = select.family_utilities(utility, valid.selection_family.to_numpy(), offsets={int(k): float(v) for k, v in POLICY["offsets"].items()})
    result = valid[["stem", "id", "fold"]].copy()
    result["utility"] = utility
    result.to_parquet(cache, index=False)
    base.save(DIRECTORY / f"nested_models_{a}_{b}.pkl", dict(models=models, ranker=ranker, excluded=excluded,
        train_stems=sorted(train.stem.unique()), features=features, precision=precision))
    print("nested action anchor", excluded, len(train), len(valid), flush=True)
    return result.set_index(["stem", "id"]).utility


def fit(name, workers, temporal=False):
    path, kind = configure(name, temporal)
    fit_directory = path / kind
    fragmentation = json.loads((DIRECTORY / "capacity.json").read_text())["fragments"]
    for outer in range(5):
        model_path = DIRECTORY / f"models_{outer}.pkl"
        if model_path.exists():
            continue
        train = TABLE.loc[TABLE.fold != outer]
        prediction = np.zeros(len(TABLE))
        for inner in range(5):
            if inner == outer:
                continue
            a, b = sorted([outer, inner])
            valid = TABLE.fold == inner
            utility = nested_utility([a, b], fit_directory, workers)
            keys = pd.MultiIndex.from_frame(TABLE.loc[valid, ["stem", "id"]])
            prediction[valid] = utility.loc[keys].to_numpy()
        tasks = [(s, outer, prediction[f.index]) for s, f in train.groupby("stem", sort=False)]
        x, labels, meta, weights = [], [], [], []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (stem, rows) in enumerate(pool.map(training_image, tasks), 1):
                for feature, di, dd, action_kind in rows:
                    x.append(feature)
                    labels.append((di, dd))
                    meta.append((stem, action_kind))
                    weights.append(len(base.RECORDS[stem]))
                if n % 100 == 0:
                    print("action samples", outer, n, "actions", len(x), flush=True)
        x, labels, weights = np.asarray(x), np.asarray(labels), np.asarray(weights)
        model = dict(train_stems=sorted(train.stem.unique()), features=FEATURES, nested_anchor="pair-excluded strong-EB/rank single-member recipe",
                     q=Q, fragment_count=fragmentation, training_actions=len(x))
        for key, index in (("di", 0), ("dd", 1)):
            model[key] = base.estimator(workers=workers, trees=400).fit(x, labels[:, index], sample_weight=weights)
        merge = np.array([k >= 3 for s, k in meta])
        if fragmentation["covered_by_two"] >= 300 and merge.sum() >= 100:
            model["merge"] = base.estimator("cross_entropy", workers, trees=300).fit(x[merge], (labels[merge, 0]-Q*labels[merge, 1] > 0).astype(float), sample_weight=weights[merge])
        base.save(model_path, model)
        np.savez_compressed(DIRECTORY / f"training_{outer}.npz", targets=labels, kinds=np.array([k for s, k in meta]))
        print("ACTION FIT", outer, len(x), flush=True)
    base.dump(DIRECTORY / "fit.json", dict(**receipt(), images=707, outer_folds=5, target="per-rater exact delta I and delta D; gain=deltaI-q*deltaD",
        rater_count="target normalization only; never an inference feature", anchor="pair-excluded strong-EB/rank; triple-excluded critic; fixed dense v1 OOF is not fully nested",
        transport="Inner uses a single scorer member with three utility heads; outer may use the five-member average or isotonic calibration."))


def anchors(name, workers, temporal=False):
    path, kind = configure(name, temporal)
    for excluded in combinations(range(5), 2):
        nested_utility(list(excluded), path / kind, workers)


def learned(stem, model, cutoff=0., use_merge=False, steps=3, return_states=False):
    state = initial(stem)
    states = [state]
    trace = []
    for step in range(steps):
        actions = propose(stem, state)
        if not actions:
            break
        x = np.array([a["features"] for a in actions])
        gain = model["di"].predict(x, num_threads=1) - model["q"]*model["dd"].predict(x, num_threads=1)
        if use_merge and "merge" in model:
            merge = np.array([a["kind"] >= 3 for a in actions])
            if merge.any():
                probability = np.clip(model["merge"].predict(x[merge], num_threads=1), 0, 1)
                gain[merge] *= probability
        best = int(np.argmax(gain))
        if gain[best] <= cutoff:
            break
        state = apply(state, actions[best])
        states.append(state)
        trace.append(dict(step=step+1, kind=actions[best]["kind"], estimated_gain=float(gain[best])))
    return (state, trace, states) if return_states else (state, trace)


def evaluate_one(stem):
    clear_geometry()
    frame = base.TABLES.get(stem)
    if frame is None:
        return stem, {key: ([], exact(stem, []), []) for key in ACTION_KEYS}
    fold = int(frame.fold.iloc[0])
    model = pickle.load((DIRECTORY / f"models_{fold}.pkl").open("rb"))
    assert stem not in model["train_stems"]
    seed = initial(stem)
    result = dict(seed=([v["rle"] for v in seed], validate_exact(stem, seed), []))
    for merge in (False, True):
        if merge and "merge" not in model:
            for cutoff in ACTION_CUTOFFS:
                result[f"c2_merge_cut{cutoff}"] = result[f"c2_cut{cutoff}"]
            continue
        _, trace, states = learned(stem, model, use_merge=merge, return_states=True)
        for cutoff in ACTION_CUTOFFS:
            length = next((i for i, action in enumerate(trace) if action["estimated_gain"] <= cutoff), len(trace))
            state = states[length]
            masks = [v["rle"] for v in state]
            assert not np.triu(select.overlap_graph(masks), 1).any()
            key = ("c2_merge" if merge else "c2") + f"_cut{cutoff}"
            result[key] = masks, validate_exact(stem, state), trace[:length]
    return stem, result


def evaluate(name, workers, temporal=False):
    configure(name, temporal)
    masks, stats, traces = [{key: {} for key in ACTION_KEYS} for _ in range(3)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (stem, result) in enumerate(pool.map(evaluate_one, sorted(base.BANK)), 1):
            for key, (m, s, t) in result.items():
                masks[key][stem], stats[key][stem], traces[key][stem] = m, s, t
            if n % 100 == 0:
                print("action evaluate", n, flush=True)
    metrics = {key: base.summary(value) for key, value in stats.items()}
    best = max(metrics, key=lambda k: metrics[k]["all"]["pq"])
    for key in metrics:
        base.save(DIRECTORY / f"{key}_predictions.pkl", masks[key])
        base.save(DIRECTORY / f"{key}_stats.pkl", stats[key])
        base.dump(DIRECTORY / f"{key}_trace.json", traces[key])
    base.dump(DIRECTORY / "evaluation.json", dict(**receipt(), identity="EV1 full707", metrics=metrics, best=best,
         baseline=POLICY["metric"], delta_all=metrics[best]["all"]["pq"]-POLICY["metric"]["all"]["pq"]))
    print("ACTIONS", json.dumps(metrics), flush=True)


def export(name, temporal=False):
    path, kind = configure(name, temporal)
    actor_directory = DIRECTORY
    report = json.loads((actor_directory / "evaluation.json").read_text())
    best = report["best"]
    if best == "seed":
        parent_export = json.loads((path / kind / "export.json").read_text())
        base.dump(actor_directory / "export.json", dict(**receipt(), policy="seed", validation=parent_export["validation"]))
        return
    # Test node utilities were produced by the same frozen base policy.
    global TABLE
    TABLE, manifest = base.load(path / "test_bank", "test")
    scores = pd.read_parquet(path / kind / "test_scores.parquet")
    assert TABLE[["stem", "id"]].equals(scores[["stem", "id"]])
    TABLE["utility"] = scores.utility
    base.TABLES = {s: f for s, f in TABLE.groupby("stem", sort=False)}
    models = [pickle.load((actor_directory / f"models_{f}.pkl").open("rb")) for f in range(5)]
    # Average predictions from all five actors before selecting an action.
    class MeanModel:
        def __init__(self, key):
            self.models = [v[key] for v in models if key in v]
        def predict(self, x, **kwargs):
            return np.mean([v.predict(x, num_threads=1) for v in self.models], axis=0)
    model = dict(di=MeanModel("di"), dd=MeanModel("dd"), q=Q)
    if any("merge" in m for m in models):
        model["merge"] = MeanModel("merge")
    predictions = {}
    cutoff = float(best.rsplit("_cut", 1)[1])
    for stem in data.list_test_stems():
        clear_geometry()
        state, trace = learned(stem, model, cutoff=cutoff, use_merge=best.startswith("c2_merge"))
        predictions[stem] = [v["rle"] for v in state]
    base.save(actor_directory / "test_predictions.pkl", predictions)
    csv = base.ROOT / "submissions" / f"attack_{name}_{kind}_{best}_20261030.csv"
    rows = [dict(filament_id=f"{stem}_{i}", segmentation_rle=r["counts"].decode("ascii")) for stem, masks in predictions.items() for i, r in enumerate(masks, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(csv, index=False)
    base.dump(actor_directory / "export.json", dict(**receipt(), validation=base.validate_csv(csv, predictions), policy=best))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["capacity", "anchors", "fit", "evaluate", "export"])
    parser.add_argument("--name", default="a0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    if args.command in ("capacity", "anchors", "fit", "evaluate"):
        globals()[args.command](args.name, args.workers, args.temporal)
    else:
        export(args.name, args.temporal)


if __name__ == "__main__":
    main()
