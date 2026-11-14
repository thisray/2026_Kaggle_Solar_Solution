"""Copy a fixed bank and append label-free scorer features without changing masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
try:
    import resource
except ImportError:
    resource = None
import shutil
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
from pycocotools import mask as mask_util
from solar import data
from l2_fit_select import read_bank

BANK = {}
TABLES = {}
MANIFEST = {}
GROUPS = {
    "stability": ["persist_count", "persist_span", "exact_threshold_count", "parent_iou", "child_iou_max", "conflict_count"],
    "context": ["nearest_cc_distance", "near_cc_64", "near_cc_128", "image_cc_area", "area_image_ratio", "image_candidate_count"],
    "appearance": ["disk_x", "disk_y", "disk_mu", "gray_std", "gray_p10", "relative_darkness_8", "relative_darkness_24", "relative_darkness_64", "ring_z_8", "ring_z_24", "ring_z_64", "axis_spine_alignment", "spine_axis_present", "image_gray_mean", "image_gray_std"],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich_one(stem):
    cv2.setNumThreads(1)
    table = TABLES.get(stem)
    if table is None:
        return [], (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else None)
    image = BANK[stem]
    rles = image["rles"]
    ious = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)))
    stable = ious > 0.9
    centers = table[["bbox_x", "bbox_y"]].to_numpy() + table[["bbox_w", "bbox_h"]].to_numpy() / 2
    baseline = np.flatnonzero(table.baseline.to_numpy())
    distances = np.linalg.norm(centers[:, None] - centers[baseline][None], axis=2)
    distances[ious[:, baseline] > 0] = np.inf
    image_area = float(table.loc[table.baseline, "area"].sum())
    gray = data.load_gray(data.train_image_path(stem) if MANIFEST["split"] == "train" else data.test_image_path(stem))
    disk = gray > 10
    dy, dx = np.where(disk)
    cx, cy, radius = dx.mean(), dy.mean(), np.sqrt(len(dx) / np.pi)
    disk_mean, disk_std = float(gray[disk].mean()), float(gray[disk].std())
    spine_path = Path(MANIFEST["prob_dir"]) / f"{stem}_spine.png"
    spine = None
    expected_spine = MANIFEST["inputs"][stem].get("spine")
    if expected_spine:
        if sha256(spine_path) != expected_spine:
            raise ValueError(f"Spine input hash mismatch: {stem}")
        spine = cv2.imread(str(spine_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    thresholds = image["thresholds"]
    rows = []
    for cid, row in enumerate(table.itertuples(index=False)):
        members = set(t for index in np.flatnonzero(stable[cid]) for t in thresholds[index]["a"])
        parent = int(row.parent_a)
        children = np.flatnonzero(table.parent_a.to_numpy() == cid)
        values = dict(stem=stem, id=cid, persist_count=len(members),
                      persist_span=max(members)-min(members) if members else 0.0,
                      exact_threshold_count=len(thresholds[cid]["a"]),
                      parent_iou=float(ious[cid, parent]) if parent >= 0 else 0.0,
                      child_iou_max=float(ious[cid, children].max()) if len(children) else 0.0,
                      conflict_count=int((ious[cid] > 0).sum()) - 1,
                      nearest_cc_distance=float(distances[cid].min()) / radius if len(baseline) else np.nan,
                      near_cc_64=int((distances[cid] < 64).sum()), near_cc_128=int((distances[cid] < 128).sum()),
                      image_cc_area=image_area, area_image_ratio=row.area / max(image_area, 1),
                      image_candidate_count=len(table), image_gray_mean=disk_mean, image_gray_std=disk_std)
        x, y, w, h = int(row.bbox_x), int(row.bbox_y), int(row.bbox_w), int(row.bbox_h)
        x0, y0, x1, y1 = max(x-64, 0), max(y-64, 0), min(x+w+64, gray.shape[1]), min(y+h+64, gray.shape[0])
        mask = mask_util.decode(rles[cid])[y0:y1, x0:x1].astype(bool)
        pixels = gray[y0:y1, x0:x1]
        inside = pixels[mask]
        yy, xx = np.where(mask)
        px, py = (xx.mean()+x0-cx) / radius, (yy.mean()+y0-cy) / radius
        values.update(disk_x=px, disk_y=py, disk_mu=float(np.sqrt(max(0, 1-px*px-py*py))),
                      gray_std=float(inside.std()), gray_p10=float(np.quantile(inside, 0.1)))
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for width in (8, 24, 64):
            ring = pixels[(distance > 0) & (distance <= width) & disk[y0:y1, x0:x1]]
            mean = float(ring.mean()) if len(ring) else np.nan
            std = float(ring.std()) if len(ring) else np.nan
            values[f"relative_darkness_{width}"] = (mean-float(inside.mean())) / max(mean, 1) if len(ring) else np.nan
            values[f"ring_z_{width}"] = (mean-float(inside.mean())) / max(std, 1) if len(ring) else np.nan
        alignment, present = np.nan, 0
        if spine is not None:
            weights = spine[y0:y1, x0:x1][mask]
            keep = weights > 0.3
            if keep.sum() >= 3 and len(xx) >= 3:
                coords = np.stack((xx, yy))
                _, axes = np.linalg.eigh(np.cov(coords, bias=True))
                _, spine_axes = np.linalg.eigh(np.cov(coords[:, keep], aweights=weights[keep], bias=True))
                alignment = float(np.dot(axes[:, -1], spine_axes[:, -1])**2)
                present = 1
        values.update(axis_spine_alignment=alignment, spine_axis_present=present)
        rows.append(values)
    return rows, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else None)


def main():
    global BANK, TABLES, MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 9:
        parser.error("workers must be between 1 and 9 (parent included in the ten-process budget)")
    if args.output_bank.exists():
        parser.error("output-bank must be new")
    started = time.perf_counter()
    table, BANK, MANIFEST = read_bank(args.bank_dir)
    TABLES = {stem: group.sort_values("id") for stem, group in table.groupby("stem", sort=False)}
    rows, max_rss = [], 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (added, rss) in enumerate(pool.map(enrich_one, MANIFEST["stems"], chunksize=1), 1):
            rows.extend(added)
            max_rss = max(max_rss, rss) if rss is not None else None
            if done % 25 == 0 or done == len(MANIFEST["stems"]):
                print(f"features {done}/{len(MANIFEST['stems'])} wall={time.perf_counter()-started:.1f}s max_worker_rss={format(max_rss, '.0f') if max_rss is not None else 'unavailable'}MB", flush=True)
    added = pd.DataFrame(rows).set_index(["stem", "id"])
    extended = table.join(added, on=["stem", "id"], validate="one_to_one")
    args.output_bank.mkdir(parents=True)
    shutil.copyfile(args.bank_dir / "rles.pkl", args.output_bank / "rles.pkl")
    extended.to_parquet(args.output_bank / "features.parquet", index=False)
    manifest = dict(MANIFEST)
    manifest.update(base_feature_columns=MANIFEST["feature_columns"], feature_groups=GROUPS,
                    feature_columns=MANIFEST["feature_columns"] + sum(GROUPS.values(), []),
                    parent_bank=str(args.bank_dir.resolve()), parent_features_sha256=MANIFEST["features_sha256"],
                    feature_source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    feature_code_sha256=sha256(Path(__file__)),
                    features_sha256=sha256(args.output_bank / "features.parquet"),
                    feature_elapsed_seconds=time.perf_counter()-started, feature_max_worker_rss_mb=max_rss)
    (args.output_bank / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(images=manifest["images"], candidates=len(extended), elapsed_seconds=manifest["feature_elapsed_seconds"], max_worker_rss_mb=max_rss)), flush=True)


if __name__ == "__main__":
    main()
