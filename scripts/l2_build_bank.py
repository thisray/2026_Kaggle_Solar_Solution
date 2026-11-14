"""Build a reusable CPU candidate bank from native-resolution probability PNGs."""
from __future__ import annotations

import argparse
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

import cv2
import numpy as np
import pandas as pd

from solar import candidates, data

RECORDS = {}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(split):
    global RECORDS
    cv2.setNumThreads(1)
    RECORDS = data.cached_records() if split == "train" else {}


def proxy(stem, records, gray, output, seed):
    """GT-derived development input, deliberately unsuitable for research claims."""
    image_seed = int.from_bytes(hashlib.sha256(stem.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed ^ image_seed)
    fg = data.soft_foreground(records)
    instances = [inst for rec in records for inst in rec.instances]
    if instances:
        erased = rng.choice(len(instances), size=max(1, int(round(0.2 * len(instances)))), replace=False)
        for index in erased:
            fg[data.decode(instances[index].rle)] = 0
    fg = cv2.GaussianBlur(fg, (0, 0), 3)
    noise = cv2.resize(rng.normal(0, 0.022, (32, 32)).astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    fg += noise
    dark = np.argwhere((gray > 10) & (gray < 100))
    if not len(dark):
        dark = np.argwhere(gray > 10)
    for _ in range(int(rng.integers(2, 6))):
        if not len(dark):
            break
        y, x = dark[int(rng.integers(len(dark)))]
        rx, ry = int(rng.integers(5, 18)), int(rng.integers(5, 18))
        x0, x1 = max(x-3*rx, 0), min(x+3*rx+1, gray.shape[1])
        y0, y1 = max(y-3*ry, 0), min(y+3*ry+1, gray.shape[0])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        fg[y0:y1, x0:x1] += rng.uniform(0.3, 0.8) * np.exp(-0.5 * (((xx-x)/rx)**2 + ((yy-y)/ry)**2))
    path = output / f"{stem}_fg.png"
    if not cv2.imwrite(str(path), np.rint(np.clip(fg, 0, 1) * 255).astype(np.uint8)):
        raise OSError(path)


def build_one(task):
    stem, prob_dir, shard_dir, split, make_proxy, seed, resume, options = task
    started = time.perf_counter()
    prob_dir, shard_dir = Path(prob_dir), Path(shard_dir)
    image_path = data.train_image_path(stem) if split == "train" else data.test_image_path(stem)
    gray = data.load_gray(image_path)
    if make_proxy:
        proxy(stem, RECORDS[stem], gray, prob_dir, seed)
    fg_path, spine_path = prob_dir / f"{stem}_fg.png", prob_dir / f"{stem}_spine.png"
    hashes = {"jpeg": sha256(image_path), "fg": sha256(fg_path), "spine": sha256(spine_path) if spine_path.exists() else None}
    head_paths = sorted(prob_dir.glob(f"{stem}_h[0-9]*.png"), key=lambda p: int(p.stem.rsplit("_h", 1)[1])) if options.get("heads") else []
    if options.get("heads"):
        if not head_paths:
            raise FileNotFoundError(f"No head PNGs for {stem}")
        hashes["heads"] = {path.name: sha256(path) for path in head_paths}
    shard_path = shard_dir / f"{stem}.pkl"
    if resume and shard_path.exists():
        with shard_path.open("rb") as stream:
            old = pickle.load(stream)
        if old["input_hashes"] == hashes and old["code_hash"] == sha256(Path(candidates.__file__)) and old.get("options") == options:
            return stem, len(old["rows"]), 0.0, old["max_rss_mb"]
    fg_u8 = cv2.imread(str(fg_path), cv2.IMREAD_UNCHANGED)
    if fg_u8 is None or fg_u8.dtype != np.uint8 or fg_u8.shape != (data.SIZE, data.SIZE):
        raise ValueError(f"Expected uint8 {data.SIZE}x{data.SIZE} foreground PNG: {fg_path}")
    fg = fg_u8.astype(np.float32) / 255
    spine = None
    if spine_path.exists():
        spine_u8 = cv2.imread(str(spine_path), cv2.IMREAD_UNCHANGED)
        if spine_u8 is None or spine_u8.dtype != np.uint8 or spine_u8.shape != fg.shape:
            raise ValueError(f"Invalid spine PNG: {spine_path}")
        spine = spine_u8.astype(np.float32) / 255
    heads = []
    for path in head_paths:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None or value.dtype != np.uint8 or value.shape != fg.shape:
            raise ValueError(f"Invalid head PNG: {path}")
        heads.append(value.astype(np.float32) / 255)
    heads = heads if head_paths else None
    nodes = candidates.generate(fg, morph=options["morph"], merge=options["merge"], heads=heads)
    rows = candidates.extract_features(nodes, fg, gray, spine, heads=heads)
    rles = [node["rle"] for node in nodes]
    target_rows, matrices = candidates.targets(rles, RECORDS.get(stem, []))
    for row, target in zip(rows, target_rows):
        row.update(target)
        row["stem"] = stem
    elapsed = time.perf_counter() - started
    rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else None)
    payload = dict(rows=rows, rles=rles, matrices=matrices,
                   thresholds=[dict(a=node["thresholds_a"], b=node["thresholds_b"]) for node in nodes],
                   record_ids=[rec.record_id for rec in RECORDS.get(stem, [])], input_hashes=hashes,
                   options=options, code_hash=sha256(Path(candidates.__file__)), elapsed_seconds=elapsed, max_rss_mb=rss)
    temporary = shard_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(shard_path)
    return stem, len(nodes), elapsed, rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--make-proxy", action="store_true")
    parser.add_argument("--seed", type=int, default=261009)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--available-only", action="store_true", help="Only stems whose fg PNG exists (partial OOF)")
    parser.add_argument("--morph", action="store_true", help="Add disk erode1/dilate1/dilate2 variants of A")
    parser.add_argument("--merge", choices=("none", "strict", "relaxed"), default="none", help="Add bridged A chains: strict=25px/35deg, relaxed=12px")
    parser.add_argument("--shape-features", action="store_true", help="Include shape metadata in scorer features, including for the control arm")
    parser.add_argument("--heads", action="store_true", help="Add per-head CC family H at thresholds 0.3/0.5/0.7 and head features")
    parser.add_argument("--head-features", action="store_true", help="Include zero head features for a matched foreground-only control")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    if args.make_proxy and args.split != "train":
        parser.error("proxy requires train annotations")
    started = time.perf_counter()
    stems = sorted(data.cached_records()) if args.split == "train" else data.list_test_stems()
    if args.limit:
        stems = stems[:args.limit]
    if args.available_only:
        stems = [stem for stem in stems if (args.prob_dir / f"{stem}_fg.png").exists()]
    if not args.make_proxy:
        missing = [stem for stem in stems if not (args.prob_dir / f"{stem}_fg.png").exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} foreground PNGs, first: {missing[0]}")
    args.prob_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.bank_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    options = dict(morph=args.morph, merge=args.merge)
    if args.heads:
        options["heads"] = True
    feature_columns = candidates.FEATURES + (candidates.SHAPE_FEATURES if args.shape_features or args.morph or args.merge != "none" else [])
    if args.heads or args.head_features:
        feature_columns += candidates.HEAD_FEATURES
    tasks = [(stem, str(args.prob_dir), str(shard_dir), args.split, args.make_proxy, args.seed, args.resume, options) for stem in stems]
    durations, rss_values, sizes = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.split,)) as pool:
        for done, (stem, count, seconds, rss) in enumerate(pool.map(build_one, tasks, chunksize=1), 1):
            durations.append(seconds)
            if rss is not None:
                rss_values.append(rss)
            sizes.append(count)
            if done % 25 == 0 or done == len(stems):
                print(f"build {done}/{len(stems)} candidates={sum(sizes)} wall={time.perf_counter()-started:.1f}s max_worker_rss={format(max(rss_values), '.0f') if rss_values else 'unavailable'}MB", flush=True)
    rows, masks, inputs = [], {}, {}
    for stem in stems:
        with (shard_dir / f"{stem}.pkl").open("rb") as stream:
            shard = pickle.load(stream)
        rows.extend(shard.pop("rows"))
        inputs[stem] = shard.pop("input_hashes")
        masks[stem] = shard
    fold_path = Path(__file__).resolve().parents[1] / "configs/folds_v1.csv"
    table = pd.DataFrame(rows, columns=None if rows else ["stem", "id", "in_a", "baseline", "parent_a", "ubar", "hit", "miou", "rater_ious", *candidates.FEATURES])
    if args.head_features and not args.heads:
        for name in candidates.HEAD_FEATURES:
            table[name] = 0.0
    if args.split == "train":
        folds = pd.read_csv(fold_path).set_index("stem")["fold"]
        table["fold"] = table["stem"].map(folds)
        if table["fold"].isna().any():
            raise ValueError("Unassigned candidate fold")
    table.to_parquet(args.bank_dir / "features.parquet", index=False)
    with (args.bank_dir / "rles.pkl").open("wb") as stream:
        pickle.dump(masks, stream, protocol=pickle.HIGHEST_PROTOCOL)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    manifest = dict(schema=2, source_sha=source, code_hash=sha256(Path(candidates.__file__)), options=options,
                    feature_columns=feature_columns, builder_code_hash=sha256(Path(__file__)),
                    family_encoding={"0": "A", "1": "B", "2": "A+B", "3": "M", "4": "G-strict", "5": "G-relaxed"},
                    variant_encoding={"0": "original", "1": "erode1", "2": "dilate1", "3": "dilate2", "4": "merge_2", "5": "merge_3", "6": "merge_4"},
                    evaluation_identity="EV1" if args.split == "train" else None,
                    probability_kind="GT-derived proxy; no research significance" if args.make_proxy else "provided dense probabilities; verify OOF lineage externally",
                    proxy_seed=args.seed if args.make_proxy else None, split=args.split, prob_dir=str(args.prob_dir.resolve()),
                    stems=stems, images=len(stems), candidates=len(table), workers=args.workers,
                    elapsed_seconds=time.perf_counter()-started, candidate_cpu_seconds=sum(durations),
                    max_worker_rss_mb=max(rss_values, default=0) if resource is not None else None, parent_max_rss_mb=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource is not None else None),
                    candidates_per_image=dict(min=min(sizes, default=0), median=float(np.median(sizes)) if sizes else 0, max=max(sizes, default=0)),
                    folds_sha256=sha256(fold_path), inputs=inputs,
                    features_sha256=sha256(args.bank_dir / "features.parquet"), rles_sha256=sha256(args.bank_dir / "rles.pkl"))
    if args.heads:
        manifest["family_encoding"]["6"] = "H"
    (args.bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("images", "candidates", "elapsed_seconds", "max_worker_rss_mb", "parent_max_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
