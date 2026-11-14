"""Annotation loading, canonical COCO rasterization and per-image GT caches.

Ground truth follows the official evaluator: every annotator-image record is
scored separately; the record key is ``<annotator_id>-<image_stem>``.
All categories are scored as one filament class.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

SIZE = 2048
RAW_SUBDIR = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")
ANN_NAME = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def artifact_root() -> Path:
    env = os.environ.get("SOLAR_ARTIFACTS")
    if env:
        return Path(env)
    return Path("artifacts")


def raw_root(root: Path | None = None) -> Path:
    env = os.environ.get("SOLAR_RAW")
    if env:
        return Path(env)
    return (root or artifact_root()) / RAW_SUBDIR


def train_image_path(stem: str, root: Path | None = None) -> Path:
    return raw_root(root) / "train/train_images" / f"{stem}.jpeg"


def test_image_path(stem: str, root: Path | None = None) -> Path:
    return raw_root(root) / "test/test_images" / f"{stem}.jpeg"


def list_test_stems(root: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (raw_root(root) / "test/test_images").glob("*.jpeg"))


def _polygon(seg) -> list[list[float]]:
    if isinstance(seg, str):
        seg = ast.literal_eval(seg)
    if seg and not isinstance(seg[0], (list, tuple)):
        seg = [seg]
    return [list(map(float, p)) for p in seg if len(p) >= 6]


def polygon_rle(seg, h: int = SIZE, w: int = SIZE) -> dict:
    """Canonical COCO polygon raster (floating vertices), merged into one RLE."""
    polys = _polygon(seg)
    rles = mask_util.frPyObjects(polys, h, w)
    return mask_util.merge(rles) if len(rles) > 1 else rles[0]


@dataclass
class Instance:
    ann_id: str
    category_id: int
    rle: dict
    area: int
    spine: list[float] = field(default_factory=list)


@dataclass
class Record:
    record_id: str          # "<annotator>-<stem>"
    annotator: str
    stem: str
    instances: list[Instance]


def load_records(root: Path | None = None) -> dict[str, list[Record]]:
    """Return {image_stem: [Record, ...]} for all train images."""
    path = raw_root(root) / ANN_NAME
    with open(path) as f:
        coco = json.load(f)
    by_record: dict[str, list[Instance]] = defaultdict(list)
    for ann in coco["annotations"]:
        rle = polygon_rle(ann["segmentation"])
        spine = ann.get("spine") or []
        if isinstance(spine, str):
            spine = ast.literal_eval(spine)
        by_record[ann["image_id"]].append(
            Instance(ann["id"], int(ann["category_id"]), rle, int(mask_util.area(rle)), list(spine))
        )
    out: dict[str, list[Record]] = defaultdict(list)
    for img in coco["images"]:
        rid = img["id"]
        annotator, stem = rid.split("-", 1)
        assert stem == Path(img["file_name"]).stem, rid
        out[stem].append(Record(rid, annotator, stem, by_record.get(rid, [])))
    return dict(out)


def cached_records(root: Path | None = None) -> dict[str, list[Record]]:
    root = root or artifact_root()
    cache = root / "cache/gt_records_v1.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    recs = load_records(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    return recs


def decode(rle: dict) -> np.ndarray:
    return mask_util.decode(rle).astype(bool)


def record_label_map(record: Record) -> np.ndarray:
    """int32 map, 0 = background, k = instance k (1-based). Later instances win on overlap."""
    lab = np.zeros((SIZE, SIZE), np.int32)
    for k, inst in enumerate(record.instances, 1):
        lab[decode(inst.rle)] = k
    return lab


def soft_foreground(records: list[Record]) -> np.ndarray:
    """Mean over raters of each rater's union mask, float32 in [0, 1]."""
    acc = np.zeros((SIZE, SIZE), np.float32)
    for rec in records:
        if rec.instances:
            acc += decode(mask_util.merge([i.rle for i in rec.instances]))
    return acc / max(len(records), 1)


def encode_mask(mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def load_gray(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img
