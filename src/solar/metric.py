"""Panoptic Quality matching the official v6 self-evaluation notebook.

Official semantics (azimahmadzadeh/self-evaluation-notebook v6):
- every annotator-image record is scored separately and pooled globally;
- a pair is a hit when IoU > 0.5; all hit pairs are TP (all-pairs, no 1:1 solver);
- FP = predictions with no hit in that record, FN = GT with no hit;
- PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN).
IoU is computed exactly on RLEs with pycocotools (same as dense matmul).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

THR = 0.5


@dataclass
class PQStats:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PQStats") -> "PQStats":
        self.iou_sum += other.iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0

    def as_dict(self) -> dict:
        return dict(pq=self.pq, sq=self.sq, rq=self.rq, tp=self.tp, fp=self.fp, fn=self.fn, iou_sum=self.iou_sum)


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), np.float64)
    m = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))  # (n_pred, n_gt)
    return np.asarray(m, np.float64).T


def record_stats(iou: np.ndarray) -> PQStats:
    n_gt, n_pred = iou.shape
    if n_gt == 0:
        return PQStats(fp=n_pred)
    if n_pred == 0:
        return PQStats(fn=n_gt)
    hit = iou > THR
    return PQStats(
        iou_sum=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(0) == 0).sum()),
        fn=int((hit.sum(1) == 0).sum()),
    )


def score_image(pred_rles: list[dict], records) -> list[tuple[str, PQStats]]:
    """Score one image's predictions against every annotator record of it."""
    out = []
    for rec in records:
        iou = iou_matrix([i.rle for i in rec.instances], pred_rles)
        out.append((rec.record_id, record_stats(iou)))
    return out


def label_map_to_rles(label: np.ndarray) -> list[dict]:
    ids = np.unique(label)
    ids = ids[ids > 0]
    return [mask_util.encode(np.asfortranarray((label == k).astype(np.uint8))) for k in ids]


@dataclass
class Evaluation:
    total: PQStats = field(default_factory=PQStats)
    single: PQStats = field(default_factory=PQStats)   # images with one rater
    multi: PQStats = field(default_factory=PQStats)    # images with >1 raters
    per_record: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(all=self.total.as_dict(), single=self.single.as_dict(), multi=self.multi.as_dict())


def evaluate(preds: dict[str, list[dict]], records_by_stem: dict, stems=None) -> Evaluation:
    """preds: {stem: [rle, ...]} (missing stem = no predictions)."""
    ev = Evaluation()
    for stem in (stems if stems is not None else records_by_stem):
        recs = records_by_stem[stem]
        bucket = ev.multi if len(recs) > 1 else ev.single
        for rid, st in score_image(preds.get(stem, []), recs):
            ev.per_record[rid] = st
            ev.total.add(st)
            bucket.add(st)
    return ev
