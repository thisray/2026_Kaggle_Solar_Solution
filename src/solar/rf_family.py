"""RF candidate stitching and image-only support features in native COCO RLE."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from pycocotools import mask as mu


def support_features(candidates: list[dict], rf_candidates: list[dict]) -> dict[str, np.ndarray]:
    """Return score-weighted IoU, any-pixel overlap count and best overlapping score.

    Candidate arguments may be RLE dictionaries or records containing ``rle``.
    RF candidates contain ``rle`` and ``score``. Empty images return zeros.
    Use only the held-out fold model for a training image.
    """
    n = len(candidates)
    result = {name: np.zeros(n, np.float32) for name in
              ("rf_max_iou_score", "rf_overlap_count", "rf_best_score")}
    if not n or not rf_candidates:
        return result
    rles = [item.get("rle", item) for item in candidates]
    size = tuple(rf_candidates[0]["rle"]["size"])
    if any(tuple(rle["size"]) != size for rle in rles) or any(tuple(item["rle"]["size"]) != size for item in rf_candidates):
        raise ValueError("External and RF candidate RLE dimensions differ")
    overlap = np.asarray(mu.iou(rles, [item["rle"] for item in rf_candidates],
                                [0] * len(rf_candidates)))
    scores = np.asarray([item["score"] for item in rf_candidates])
    result["rf_max_iou_score"] = (overlap * scores[None]).max(1).astype(np.float32)
    result["rf_overlap_count"] = (overlap > 0).sum(1).astype(np.float32)
    result["rf_best_score"] = np.where(overlap > 0, scores[None], 0).max(1).astype(np.float32)
    return result


def stitch_tiles(raw: list[dict], agreement: float = 0.5, min_intersection: int = 8) -> list[dict]:
    """Join reciprocal best masks in overlapping tiles without using GT.

    Agreement is intersection / smaller *overlap-band* area. A component may
    contain at most one query from each tile, preventing same-tile query chains.
    Scores are the maximum member score. Original proposals remain available to
    the caller as alternatives; this function returns the replacement pool.
    """
    if not raw:
        return []
    groups = defaultdict(list)
    for i, item in enumerate(raw):
        groups[tuple(item["tile"])].append(i)
    parents = list(range(len(raw)))
    tiles = [{tuple(item["tile"])} for item in raw]

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = []
    h, w = raw[0]["rle"]["size"]
    for ta, tb in combinations(sorted(groups), 2):
        x0, y0 = max(ta[0], tb[0]), max(ta[1], tb[1])
        x1, y1 = min(ta[0] + ta[2], tb[0] + tb[2]), min(ta[1] + ta[3], tb[1] + tb[3])
        if x1 <= x0 or y1 <= y0:
            continue
        band = mu.frPyObjects([[x0, y0, x1, y0, x1, y1, x0, y1]], h, w)[0]
        ia, ib = groups[ta], groups[tb]
        a = [mu.merge([raw[i]["rle"], band], intersect=True) for i in ia]
        b = [mu.merge([raw[i]["rle"], band], intersect=True) for i in ib]
        aa, ab = np.asarray(mu.area(a), float), np.asarray(mu.area(b), float)
        iou = np.asarray(mu.iou(a, b, [0] * len(b)))
        intersection = iou * (aa[:, None] + ab[None]) / (1 + iou)
        quality = intersection / np.maximum(1, np.minimum(aa[:, None], ab[None]))
        valid = (quality >= agreement) & (intersection >= min_intersection)
        quality = np.where(valid, quality, -1)
        best_b, best_a = quality.argmax(1), quality.argmax(0)
        for j, k in enumerate(best_b):
            if valid[j, k] and best_a[k] == j:
                edges.append((float(quality[j, k]), min(raw[ia[j]]["score"], raw[ib[k]]["score"]), ia[j], ib[k]))
    for _, _, a, b in sorted(edges, reverse=True):
        ra, rb = root(a), root(b)
        if ra == rb or tiles[ra] & tiles[rb]:
            continue
        parents[rb] = ra
        tiles[ra] |= tiles[rb]
    members = defaultdict(list)
    for i in range(len(raw)):
        members[root(i)].append(i)
    result = []
    for indices in members.values():
        best = max(indices, key=lambda i: raw[i]["score"])
        item = dict(raw[best])
        item["rle"] = mu.merge([raw[i]["rle"] for i in indices])
        item["box"] = mu.toBbox(item["rle"]).tolist()
        item["tile_provenance"] = [dict(tile=raw[i]["tile"], query=raw[i].get("query"),
                                         score=raw[i]["score"], raw_index=i) for i in indices]
        item["merged"] = len(indices) > 1
        result.append(item)
    return sorted(result, key=lambda item: -item["score"])


def candidate_alternatives(raw: list[dict], stitched: list[dict]) -> list[dict]:
    """Keep raw candidates plus novel unions for downstream learned selection."""
    originals = [dict(item, box=mu.toBbox(item["rle"]).tolist(), merged=False,
                      tile_provenance=[dict(tile=item["tile"], query=item.get("query"),
                                            score=item["score"], raw_index=i)])
                 for i, item in enumerate(raw)]
    seen = {item["rle"]["counts"] for item in originals}
    for item in stitched:
        if item["rle"]["counts"] not in seen:
            originals.append(item)
            seen.add(item["rle"]["counts"])
    return sorted(originals, key=lambda item: -item["score"])


def finalize(candidates: list[dict], tau: float = 0.5, topk: int = 256) -> list[dict]:
    """Score-ordered mask NMS followed by pixel ownership, as in W52."""
    if not candidates:
        return []
    occupied = np.zeros(candidates[0]["rle"]["size"], bool)
    result = []
    for item in sorted(candidates, key=lambda item: -item["score"]):
        if item["score"] < tau:
            continue
        if result and np.asarray(mu.iou([item["rle"]], result, [0] * len(result))).max() > 0.5:
            continue
        mask = mu.decode(item["rle"]).astype(bool) & ~occupied
        if mask.sum() < 20:
            continue
        occupied |= mask
        result.append(mu.encode(np.asfortranarray(mask.astype(np.uint8))))
        if len(result) >= topk:
            break
    return result
