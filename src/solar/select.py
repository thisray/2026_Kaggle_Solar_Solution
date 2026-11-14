"""Pixel-disjoint greedy selection and exact component-tree antichain DP."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util


def crop_decode(rle):
    """Decode compressed COCO runs directly into the native bounding box."""
    x, y, w, h = map(int, mask_util.toBbox(rle))
    raw = rle["counts"]
    if isinstance(raw, str):
        raw = raw.encode("ascii")
    counts, position = [], 0
    while position < len(raw):
        value, shift = 0, 0
        while True:
            byte = raw[position] - 48
            position += 1
            value |= (byte & 31) << (5 * shift)
            shift += 1
            if not byte & 32:
                if byte & 16:
                    value |= -1 << (5 * shift)
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    height = rle["size"][0]
    result = np.zeros((h, w), bool)
    offset = 0
    for index, count in enumerate(counts):
        end = offset + count
        if index % 2 and count:
            for column in range(max(x, offset // height), min(x + w, (end - 1) // height + 1)):
                lo = max(y, offset - column * height)
                hi = min(y + h, end - column * height)
                if hi > lo:
                    result[lo-y:hi-y, column-x] = True
        offset = end
    if offset != int(np.prod(rle["size"])):
        raise ValueError("Invalid COCO RLE length")
    return result, x, y


def overlap_graph(rles):
    """Any positive intersection is a conflict; RLE IoU avoids dense decoding."""
    if not rles:
        return np.zeros((0, 0), bool)
    return np.asarray(mask_util.iou(rles, rles, [0] * len(rles))) > 0


def greedy(scores, tau, conflicts, allowed=None):
    scores = np.asarray(scores)
    active = scores > tau
    if allowed is not None:
        active &= np.asarray(allowed, bool)
    blocked = np.zeros(len(scores), bool)
    chosen = []
    for cid in np.argsort(-scores, kind="stable"):
        if active[cid] and not blocked[cid]:
            chosen.append(int(cid))
            blocked |= conflicts[cid]
    return chosen


def family_utilities(scores, families, calibrators=None, offsets=None):
    """Calibrate every family without an A-membership admission restriction."""
    values = np.asarray(scores, float).copy()
    families = np.asarray(families)
    for family in np.unique(families):
        keep = families == family
        if calibrators and family in calibrators:
            values[keep] = calibrators[family].predict(values[keep])
        if offsets:
            values[keep] -= offsets.get(family, 0.0)
    return values


def paint(scores, tau, rles, allowed=None, min_area=32, min_fraction=0.0, decoded=None):
    """Paint in descending utility order; return emitted, pixel-disjoint RLEs.

    Candidate IDs refer to the source geometry. Emitted masks can be clipped,
    so callers must score and export the returned RLEs rather than source IDs.
    """
    scores = np.asarray(scores, float)
    active = scores > tau
    if allowed is not None:
        active &= np.asarray(allowed, bool)
    if not rles:
        return [], []
    from solar.candidates import crop_rle

    claimed = np.zeros(rles[0]["size"], bool)
    ids, emitted = [], []
    for cid in np.argsort(-scores, kind="stable"):
        if not active[cid]:
            continue
        if decoded is not None and int(cid) in decoded:
            mask, x, y = decoded[int(cid)]
        else:
            mask, x, y = crop_decode(rles[cid])
            if decoded is not None:
                decoded[int(cid)] = mask, x, y
        region = claimed[y:y+mask.shape[0], x:x+mask.shape[1]]
        remaining = mask & ~region
        area = int(remaining.sum())
        if area < min_area or area < min_fraction * int(mask.sum()):
            continue
        region |= remaining
        ids.append(int(cid))
        emitted.append(crop_rle(remaining, x, y, rles[cid]["size"]))
    return ids, emitted


def replace_k(scores, tau, conflicts, initial=None, max_remove=3, max_steps=10):
    """Improve an independent set by replacing at most three incumbents."""
    if not 1 <= max_remove <= 3:
        raise ValueError("max_remove must be in [1, 3]")
    values = np.asarray(scores, float) - tau
    chosen = list(initial if initial is not None else greedy(scores, tau, conflicts))
    for _ in range(max_steps):
        present = set(chosen)
        best = None
        for cid in np.flatnonzero(values > 0):
            if int(cid) in present:
                continue
            removed = [i for i in chosen if conflicts[cid, i]]
            if len(removed) > max_remove:
                continue
            gain = values[cid] - values[removed].sum()
            if gain > 1e-12 and (best is None or gain > best[0]):
                best = (float(gain), int(cid), removed)
        if best is None:
            break
        _, cid, removed = best
        chosen = [i for i in chosen if i not in removed] + [cid]
    return chosen


def tree_dp(scores, tau, parents, allowed):
    """Maximize sum(score-tau) on an A-only laminar forest exactly."""
    scores = np.asarray(scores)
    allowed = np.asarray(allowed, bool)
    children = [[] for _ in scores]
    roots = []
    for cid in np.flatnonzero(allowed):
        parent = int(parents[cid])
        if parent < 0:
            roots.append(int(cid))
        else:
            children[parent].append(int(cid))
    order, stack = [], roots.copy()
    while stack:
        cid = stack.pop()
        order.append(cid)
        stack.extend(children[cid])
    value = np.zeros(len(scores))
    take = np.zeros(len(scores), bool)
    for cid in reversed(order):
        descendant_value = sum(value[ch] for ch in children[cid])
        own = float(scores[cid] - tau)
        take[cid] = own > 0 and own >= descendant_value
        value[cid] = max(0.0, own, descendant_value)
    chosen, stack = [], roots.copy()
    while stack:
        cid = stack.pop()
        if take[cid]:
            chosen.append(cid)
        else:
            stack.extend(children[cid])
    return chosen
