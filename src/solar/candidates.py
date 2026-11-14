"""CPU component-tree candidates, cropped features and rater-averaged targets."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
from pycocotools import mask as mask_util
from skimage.morphology import disk, skeletonize
from skimage.measure import perimeter
from scipy.spatial import cKDTree

from solar import metric

THRESHOLDS = tuple(round(x / 100, 2) for x in range(10, 86, 5))
KERNEL = disk(3).astype(np.uint8)
FEATURES = [
    "area", "perimeter", "bbox_w", "bbox_h", "axis_ratio", "skeleton_length",
    "mean_width", "fg_mean", "fg_max", "fg_p10", "fg_p50", "fg_p90",
    "ring_fg_mean", "contrast", "spine_mean", "spine_max", "spine_fraction",
    "spine_present", "t", "family", "parent_area_ratio", "children",
    "normalized_radius", "gray_mean", "ring_gray_mean", "gray_contrast",
    "image_cc50", "image_fg_mean",
]
SHAPE_FEATURES = ["variant", "n_parts", "bridge_length"]
HEAD_FEATURES = ["head_agree", "head_prob_min", "head_prob_max", "head_frac"]


def crop_rle(mask, x, y, shape):
    """Encode a crop directly in full-image COCO coordinates without a full mask."""
    h, w = shape
    padded = np.pad(mask.astype(np.int8), ((1, 1), (0, 0)))
    changes = np.diff(padded, axis=0)
    sy, sx = np.where(changes == 1)
    ey, ex = np.where(changes == -1)
    starts = (sx.astype(np.int64) + x) * h + sy + y
    ends = (ex.astype(np.int64) + x) * h + ey + y
    starts.sort()
    ends.sort()
    keep_start = np.r_[True, starts[1:] != ends[:-1]]
    keep_end = np.r_[ends[:-1] != starts[1:], True]
    starts, ends = starts[keep_start], ends[keep_end]
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[0] = starts[0]
    counts[1:-1:2] = ends - starts
    counts[2:-1:2] = starts[1:] - ends[:-1]
    counts[-1] = h * w - ends[-1]
    return mask_util.frPyObjects({"size": [h, w], "counts": counts.tolist()}, h, w)


def generate(fg, min_area=20, morph=False, merge="none", heads=None):
    """Deduplicate masks globally, retaining both family memberships and trees.

    family: 0=A, 1=B, 2=A+B. Parents skip repeated identical threshold nodes.
    Candidate IDs are image-local positions in the returned list.
    """
    nodes, seen = [], {}
    for family in ("a", "b"):
        previous_labels = previous_ids = None
        for t in THRESHOLDS:
            binary = (fg > t).astype(np.uint8)
            if family == "b":
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, KERNEL)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            ids = np.full(count, -1, dtype=np.int32)
            for label in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1:
                x, y, w, h, area = map(int, stats[label])
                crop = labels[y:y+h, x:x+w] == label
                digest = hashlib.blake2b(crop.tobytes(), digest_size=16).digest()
                key = (x, y, w, h, area, digest)
                ry, rx = np.argwhere(crop)[0]
                parent = -1 if previous_labels is None else int(previous_ids[previous_labels[y+ry, x+rx]])
                if key not in seen:
                    cid = len(nodes)
                    seen[key] = cid
                    nodes.append(dict(id=cid, bbox=(x, y, w, h), area=area,
                                      mask=crop, rle=crop_rle(crop, x, y, fg.shape),
                                      t=t, family=0 if family == "a" else 1,
                                      parent_a=-1, parent_b=-1, thresholds_a=[], thresholds_b=[]))
                cid = seen[key]
                node = nodes[cid]
                membership = node[f"thresholds_{family}"]
                if not membership:
                    node[f"parent_{family}"] = parent
                membership.append(t)
                if node["thresholds_a"] and node["thresholds_b"]:
                    node["family"] = 2
                ids[label] = cid
            previous_labels, previous_ids = labels, ids
    for node in nodes:
        node["parent"] = node["parent_a"] if node["thresholds_a"] else node["parent_b"]
        node.update(variant=0, variant_name="original", n_parts=1, bridge_length=0.0)
    base = list(nodes)

    def add(mask, x, y, t, family, variant, name, parts, length, sources):
        yy, xx = np.where(mask)
        if not len(xx):
            return
        x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
        mask = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=bool)
        x, y = int(x+x0), int(y+y0)
        bh, bw = mask.shape
        area = int(mask.sum())
        key = (x, y, bw, bh, area, hashlib.blake2b(mask.tobytes(), digest_size=16).digest())
        if key in seen:
            return
        cid = len(nodes)
        seen[key] = cid
        nodes.append(dict(id=cid, bbox=(x, y, bw, bh), area=area, mask=mask,
                          rle=crop_rle(mask, x, y, fg.shape), t=t, family=family,
                          parent=-1, parent_a=-1, parent_b=-1, thresholds_a=[], thresholds_b=[],
                          variant=variant, variant_name=name, n_parts=parts,
                          bridge_length=float(length), source_ids=sources))

    if morph:
        for node in base:
            ts = [t for t in (0.3, 0.5, 0.7) if t in node["thresholds_a"]]
            if not ts or node["area"] < 100:
                continue
            x, y, bw, bh = node["bbox"]
            x0, y0 = max(0, x-2), max(0, y-2)
            x1, y1 = min(fg.shape[1], x+bw+2), min(fg.shape[0], y+bh+2)
            padded = np.zeros((y1-y0, x1-x0), np.uint8)
            padded[y-y0:y-y0+bh, x-x0:x-x0+bw] = node["mask"]
            for variant, name, radius, operation in ((1, "erode1", 1, cv2.erode),
                                                     (2, "dilate1", 1, cv2.dilate),
                                                     (3, "dilate2", 2, cv2.dilate)):
                mask = operation(padded, disk(radius).astype(np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                add(mask, x0, y0, ts[0], 3, variant, name, 1, 0, [node["id"]])
    if merge not in ("none", "strict", "relaxed"):
        raise ValueError(f"Unknown merge mode: {merge}")
    if merge != "none":
        geometry = {}
        for t in (0.3, 0.5):
            components = [node for node in base if t in node["thresholds_a"]]
            for node in components:
                cid = node["id"]
                if cid in geometry:
                    continue
                x, y, _, _ = node["bbox"]
                yy, xx = np.where(node["mask"])
                points = np.stack((xx+x, yy+y), axis=1)
                _, vectors = np.linalg.eigh(np.cov(points.T, bias=True))
                axis = vectors[:, -1]
                padded = np.pad(node["mask"].astype(np.uint8), 1)
                boundary = padded & (1-cv2.erode(padded, np.ones((3, 3), np.uint8)))
                by, bx = np.where(boundary[1:-1, 1:-1])
                edge = np.stack((bx+x, by+y), axis=1)
                width = node["area"] / max(int(skeletonize(padded > 0).sum()), 1)
                geometry[cid] = (edge, cKDTree(edge), axis, width)
            edges = {}
            limit = 25 if merge == "strict" else 12
            cosine = np.cos(np.deg2rad(35))
            for i, left in enumerate(components):
                ax, ay, aw, ah = left["bbox"]
                lp, lt, la, lw = geometry[left["id"]]
                for right in components[i+1:]:
                    bx, by, bw, bh = right["bbox"]
                    dx = max(ax-(bx+bw-1), bx-(ax+aw-1), 0)
                    dy = max(ay-(by+bh-1), by-(ay+ah-1), 0)
                    if np.hypot(dx, dy) > limit:
                        continue
                    rp, _, ra, rw = geometry[right["id"]]
                    if merge == "strict" and abs(float(la @ ra)) <= cosine:
                        continue
                    distances, indices = lt.query(rp, workers=1)
                    k = int(np.argmin(distances))
                    distance = float(distances[k])
                    if distance > limit:
                        continue
                    p, q = lp[indices[k]], rp[k]
                    direction = (q-p) / max(distance, 1e-12)
                    if merge == "strict" and min(abs(float(direction @ la)), abs(float(direction @ ra))) <= cosine:
                        continue
                    edges[(left["id"], right["id"])] = (distance, p, q, max(1, int(round((lw+rw)/2))))

            def emit(parts, bridges):
                padding = max(edge[3] for edge in bridges)+1
                x0 = max(0, min(nodes[c]["bbox"][0] for c in parts)-padding)
                y0 = max(0, min(nodes[c]["bbox"][1] for c in parts)-padding)
                x1 = min(fg.shape[1], max(nodes[c]["bbox"][0]+nodes[c]["bbox"][2] for c in parts)+padding)
                y1 = min(fg.shape[0], max(nodes[c]["bbox"][1]+nodes[c]["bbox"][3] for c in parts)+padding)
                mask = np.zeros((y1-y0, x1-x0), np.uint8)
                for cid in parts:
                    x, y, bw, bh = nodes[cid]["bbox"]
                    mask[y-y0:y-y0+bh, x-x0:x-x0+bw] |= nodes[cid]["mask"]
                for _, p, q, width in bridges:
                    cv2.line(mask, tuple((p-[x0, y0]).astype(int)), tuple((q-[x0, y0]).astype(int)), 1, width, cv2.LINE_8)
                add(mask, x0, y0, t, 4 if merge == "strict" else 5,
                    len(parts)+2, f"merge_{len(parts)}", len(parts), sum(e[0] for e in bridges), list(parts))

            # Seed every admissible pair, then extend only chain endpoints greedily.
            for pair, edge in sorted(edges.items(), key=lambda item: (item[1][0], item[0])):
                parts, bridges = list(pair), [edge]
                emit(parts, bridges)
                while len(parts) < 4:
                    options = [(e[0], key, e) for key, e in edges.items()
                               if ((key[0] in (parts[0], parts[-1]) and key[1] not in parts)
                                   or (key[1] in (parts[0], parts[-1]) and key[0] not in parts))]
                    if not options:
                        break
                    _, key, extension = min(options, key=lambda item: (item[0], item[1]))
                    endpoint = key[0] if key[0] in parts else key[1]
                    new = key[1] if endpoint == key[0] else key[0]
                    parts.insert(0, new) if endpoint == parts[0] else parts.append(new)
                    bridges.append(extension)
                    emit(parts, bridges)
    if heads is not None:
        head_masks = []
        for head_id, probability in enumerate(heads):
            for t in (0.5, 0.3, 0.7):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (probability > t).astype(np.uint8), connectivity=8)
                for label in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1:
                    x, y, bw, bh, area = map(int, stats[label])
                    mask = labels[y:y+bh, x:x+bw] == label
                    key = (x, y, bw, bh, area, hashlib.blake2b(mask.tobytes(), digest_size=16).digest())
                    if key not in seen:
                        cid = len(nodes)
                        seen[key] = cid
                        nodes.append(dict(id=cid, bbox=(x, y, bw, bh), area=area, mask=mask,
                                          rle=crop_rle(mask, x, y, fg.shape), t=t, family=6,
                                          parent=-1, parent_a=-1, parent_b=-1,
                                          thresholds_a=[], thresholds_b=[], variant=0,
                                          variant_name="head", n_parts=1, bridge_length=0.0))
                    node = nodes[seen[key]]
                    node.setdefault("head_sources", []).append((head_id, t))
                    head_masks.append((head_id, node))
        # Count distinct generating heads, including approximate agreement with A/B.
        for node in nodes:
            x, y, bw, bh = node["bbox"]
            possible = [(hid, other) for hid, other in head_masks
                        if min(node["area"], other["area"]) > .9 * max(node["area"], other["area"])
                        and x < other["bbox"][0] + other["bbox"][2]
                        and other["bbox"][0] < x + bw
                        and y < other["bbox"][1] + other["bbox"][3]
                        and other["bbox"][1] < y + bh]
            agreement = set()
            if possible:
                ious = mask_util.iou([node["rle"]], [other["rle"] for _, other in possible],
                                     [0] * len(possible))[0]
                agreement = {hid for (hid, _), iou in zip(possible, ious) if iou > .9}
            node["head_agree"] = len(agreement)
    return nodes


def extract_features(nodes, fg, gray, spine=None, heads=None):
    """Work on padded bounding boxes; no candidate allocates a full-image mask."""
    h, w = fg.shape
    disk_y, disk_x = np.where(gray > 10)
    cx = float(disk_x.mean()) if len(disk_x) else w / 2
    cy = float(disk_y.mean()) if len(disk_y) else h / 2
    radius = max(np.sqrt(len(disk_y) / np.pi), 1)
    image_cc50 = cv2.connectedComponents((fg > 0.5).astype(np.uint8), connectivity=8)[0] - 1
    image_fg_mean = float(fg.mean())
    child_counts = np.zeros(len(nodes), dtype=int)
    for node in nodes:
        if node["parent"] >= 0:
            child_counts[node["parent"]] += 1
    rows = []
    for node in nodes:
        x, y, bw, bh = node["bbox"]
        mask = node["mask"]
        yy, xx = np.where(mask)
        cov = np.cov(np.stack((xx, yy)), bias=True) if len(xx) > 1 else np.zeros((2, 2))
        eig = np.linalg.eigvalsh(cov)
        axis_ratio = float(np.sqrt((eig[1] + 1e-6) / (eig[0] + 1e-6)))
        skel_length = int(skeletonize(np.pad(mask, 1)).sum())
        x0, y0, x1, y1 = max(x-3, 0), max(y-3, 0), min(x+bw+3, w), min(y+bh+3, h)
        padded = np.zeros((y1-y0, x1-x0), np.uint8)
        padded[y-y0:y-y0+bh, x-x0:x-x0+bw] = mask
        ring = (cv2.dilate(padded, KERNEL) > 0) & (padded == 0)
        inside_fg = fg[y:y+bh, x:x+bw][mask]
        ring_fg = fg[y0:y1, x0:x1][ring]
        inside_gray = gray[y:y+bh, x:x+bw][mask]
        ring_gray = gray[y0:y1, x0:x1][ring]
        quantiles = np.quantile(inside_fg, [0.1, 0.5, 0.9])
        if spine is not None:
            inside_spine = spine[y:y+bh, x:x+bw][mask]
            sm, sx, sf = float(inside_spine.mean()), float(inside_spine.max()), float((inside_spine > 0.5).mean())
        else:
            sm = sx = sf = np.nan
        fg_mean = float(inside_fg.mean())
        ring_mean = float(ring_fg.mean()) if len(ring_fg) else 0.0
        gray_mean = float(inside_gray.mean())
        ring_gray_mean = float(ring_gray.mean()) if len(ring_gray) else 0.0
        parent = node["parent"]
        rows.append(dict(
            id=node["id"], family=node["family"], t=node["t"], parent=parent,
            variant=node["variant"], variant_name=node["variant_name"],
            n_parts=node["n_parts"], bridge_length=node["bridge_length"],
            parent_a=node["parent_a"], parent_b=node["parent_b"],
            in_a=bool(node["thresholds_a"]), baseline=0.5 in node["thresholds_a"],
            area=node["area"], perimeter=float(perimeter(np.pad(mask, 1), neighborhood=8)),
            bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh, axis_ratio=axis_ratio,
            skeleton_length=skel_length, mean_width=node["area"] / max(skel_length, 1),
            fg_mean=fg_mean, fg_max=float(inside_fg.max()), fg_p10=quantiles[0],
            fg_p50=quantiles[1], fg_p90=quantiles[2], ring_fg_mean=ring_mean,
            contrast=fg_mean-ring_mean, spine_mean=sm, spine_max=sx, spine_fraction=sf,
            spine_present=int(spine is not None), parent_area_ratio=node["area"] / nodes[parent]["area"] if parent >= 0 else 1.0,
            children=int(child_counts[node["id"]]), normalized_radius=float(np.hypot(xx.mean()+x-cx, yy.mean()+y-cy) / radius),
            gray_mean=gray_mean, ring_gray_mean=ring_gray_mean, gray_contrast=gray_mean-ring_gray_mean,
            image_cc50=image_cc50, image_fg_mean=image_fg_mean,
        ))
        if heads is not None:
            means = [float(probability[y:y+bh, x:x+bw][mask].mean()) for probability in heads]
            rows[-1].update(in_h=bool(node.get("head_sources")), head_agree=node["head_agree"],
                            head_prob_min=min(means), head_prob_max=max(means),
                            head_frac=node["head_agree"] / len(heads))
    return rows


def targets(rles, records):
    matrices = [metric.iou_matrix([inst.rle for inst in rec.instances], rles) for rec in records]
    maxima = np.stack([m.max(axis=0) if len(m) else np.zeros(len(rles)) for m in matrices], axis=1) if records else np.empty((len(rles), 0))
    hit = maxima > 0.5
    q = maxima * hit
    rows = []
    for i in range(len(rles)):
        n_hit = int(hit[i].sum())
        rows.append(dict(ubar=float(q[i].mean()) if records else np.nan,
                         hit=float(hit[i].mean()) if records else np.nan,
                         miou=float(q[i].sum() / n_hit) if n_hit else 0.0,
                         rater_ious=maxima[i].tolist()))
    return rows, matrices
