"""Official-record filament centers, extents and identity with frozen DINOv3."""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
import torch.nn.functional as F

from solar import data
from solar.foundation_dense import CNX, PRETRAINED_REVISIONS


def image_channels(gray):
    raw = gray.astype(np.float32) / 255
    disk = raw > .04
    pixels = raw[disk]
    normalized = (raw - pixels.mean()) / max(float(pixels.std()), .01)
    yy, xx = np.indices(raw.shape, dtype=np.float32)
    radius = np.hypot(xx - 1024, yy - 1024) / 1000
    contrast = cv2.GaussianBlur(raw, (0, 0), 8) - raw
    return np.stack([raw, normalized, contrast, radius])


def contrast_pool(gray, limit=256):
    """Image-only hard-negative/diagnostic pool; no fixed disk erosion."""
    raw = gray.astype(np.float32) / 255
    contrast = cv2.GaussianBlur(raw, (0, 0), 8) - raw
    boxes = []
    for threshold in (.02, .04, .06):
        binary = ((contrast > threshold) & (gray > 20)).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if 30 <= area <= 50000 and max(w, h) <= 1024:
                score = float(contrast[y:y+h, x:x+w][labels[y:y+h, x:x+w] == i].mean()) * math.sqrt(area)
                boxes.append(dict(box=[x, y, w, h], score=score))
    unique = []
    for proposal in sorted(boxes, key=lambda p: -p['score']):
        if not any(np.max(np.abs(np.asarray(proposal['box']) - p['box'])) < 5 for p in unique):
            unique.append(proposal)
            if len(unique) == limit:
                break
    return unique


def fixed_window(box, size):
    x, y, w, h = box
    return (int(np.clip(round(x+w/2-size/2), 0, 2048-size)),
            int(np.clip(round(y+h/2-size/2), 0, 2048-size)), size)


def covered(box, proposal, size):
    left, top, _ = fixed_window(proposal, size)
    x, y, w, h = box
    return x >= left and y >= top and x+w <= left+size and y+h <= top+size


class FilamentProposer(nn.Module):
    def __init__(self, pretrained=True, width=32, slots=2):
        super().__init__()
        import timm
        self.encoder = timm.create_model(CNX, pretrained=pretrained, in_chans=1, features_only=True,
            pretrained_cfg_overlay={'hf_hub_id': f'timm/{CNX}@{PRETRAINED_REVISIONS[CNX]}'})
        self.encoder.requires_grad_(False)
        self.slots = slots
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in self.encoder.feature_info.channels())
        self.detail = nn.Sequential(nn.Conv2d(4, width, 5, 4, 2), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(width*2, width, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.center = nn.Conv2d(width, slots, 1)
        self.geometry = nn.Conv2d(width, slots*4, 1)
        self.spine = nn.Conv2d(width, 1, 1)
        self.identity = nn.Conv2d(width, 8, 1)
        self.box_quality = nn.Sequential(nn.Linear(width+4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.center.bias, -4.6)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x[:, 1:2])
        levels = [layer(value) for layer, value in zip(self.lateral, features)]
        for i in range(len(levels)-2, -1, -1):
            levels[i] = levels[i] + F.interpolate(levels[i+1], size=levels[i].shape[-2:], mode='bilinear', align_corners=False)
        feature = self.fuse(torch.cat([levels[0], self.detail(x)], 1))
        return dict(center=self.center(feature), geometry=self.geometry(feature), spine=self.spine(feature),
                    identity=F.normalize(self.identity(feature), dim=1), feature=feature)

    def score_boxes(self, feature, boxes, size):
        """Learn empty and merged-box negatives on the same proposal distribution."""
        if not len(boxes):
            return feature.new_empty(0)
        pooled = []
        height, width = feature.shape[-2:]
        for x, y, w, h in boxes:
            x0, x1 = max(0, int(x/4)), min(width, max(1, math.ceil((x+w)/4)))
            y0, y1 = max(0, int(y/4)), min(height, max(1, math.ceil((y+h)/4)))
            if x1 <= x0 or y1 <= y0:
                raise ValueError('Box outside tile')
            geometry = feature.new_tensor([w/size, h/size, (x+w/2)/size, (y+h/2)/size])
            pooled.append(torch.cat([feature[0, :, y0:y1, x0:x1].mean((1, 2)), geometry]))
        return self.box_quality(torch.stack(pooled)).flatten()


class ProposalCrops:
    def __init__(self, records, stems, seed=261046):
        self.records = records
        self.seed = seed
        # Uniform annotation-instance sampling, not uniform foreground pixels.
        self.units = [(s, ri, ii) for s in sorted(stems) for ri, r in enumerate(records[s]) for ii in range(len(r.instances))]

    @lru_cache(maxsize=4)
    def image(self, stem):
        gray = data.load_gray(data.train_image_path(stem))
        return gray, image_channels(gray), contrast_pool(gray, 128)

    def draw(self, index):
        rng = np.random.default_rng(self.seed+index)
        stem, ri, ii = self.units[int(rng.integers(len(self.units)))]
        rec = self.records[stem][ri]
        chosen = mu.toBbox(rec.instances[ii].rle)
        size = int(rng.choice([512, 768]))
        gray, channels, proposals = self.image(stem)
        kind = 'instance'
        box = chosen.copy()
        if index % 4 == 3:
            # Entire records supervise the tile even on negative-source draws.
            if proposals and rng.random() < .75:
                box = np.asarray(proposals[int(rng.integers(len(proposals)))]['box'], dtype=float)
                kind = 'contrast'
            else:
                box = np.array([*rng.uniform(0, 2048, 2), 1, 1])
                kind = 'empty_or_random'
        box[:2] += rng.uniform(-.15, .15, 2) * np.maximum(box[2:], 32)
        if kind == 'instance':
            # Every anchor, including rare long instances, has positive exposure.
            need = 1.3*float(max(chosen[2:]))
            size = max(size, next((s for s in (512, 768, 1024, 2048) if need <= s), 2048))
        left, top, _ = fixed_window(box, size)
        n = size//4
        center = np.zeros((2, n, n), np.float32)
        spine = np.zeros((size, size), np.uint8)
        current = np.zeros((size, size), np.uint8)
        others = np.zeros_like(current)
        items = []
        occupancy = {}
        overflow = 0
        for rj, record in enumerate(self.records[stem]):
            for inst in record.instances:
                crop = mu.decode(inst.rle)[top:top+size, left:left+size]
                if rj != ri:
                    others |= crop
                    b = np.asarray(mu.toBbox(inst.rle))
                    x0, y0 = np.maximum(0, np.floor(b[:2]-[left, top])).astype(int)
                    x1, y1 = np.minimum(size, np.ceil(b[:2]+b[2:]-[left, top])).astype(int)
                    if x1 > x0 and y1 > y0:
                        # Bbox centers may lie outside a thin curved mask.
                        # Ignore those known centers as well as mask pixels.
                        others[y0:y1, x0:x1] = 1
                    continue
                current |= crop
                b = np.asarray(mu.toBbox(inst.rle), np.float32)
                b[:2] -= [left, top]
                if b[0] < 0 or b[1] < 0 or b[0]+b[2] > size or b[1]+b[3] > size:
                    continue
                cx, cy = (b[:2]+b[2:]/2)/4
                px, py = min(int(cx), n-1), min(int(cy), n-1)
                slot = occupancy.get((py, px), 0)
                occupancy[py, px] = slot+1
                if slot >= 2:
                    overflow += 1
                    continue
                yy, xx = np.indices((n, n))
                sigma = np.clip(min(b[2:])/16, 1, 4)
                heat = np.exp(-((xx-px)**2+(yy-py)**2)/(2*sigma**2))
                center[slot] = np.maximum(center[slot], heat)
                points = np.asarray(inst.spine, np.float32).reshape(-1, 2)-[left, top]
                if len(points) >= 2:
                    cv2.polylines(spine, [np.rint(points).astype(np.int32)], False, 1, 3)
                ys, xs = np.where(crop)
                ids = rng.choice(len(xs), min(32, len(xs)), replace=False)
                cells = np.unique(np.stack([ys[ids]//4, xs[ids]//4], 1), axis=0)
                items.append(dict(box=b.tolist(), cell=[slot, py, px],
                                  geometry=[cx-px, cy-py, *np.log(np.maximum(b[2:], 1))], points=cells.tolist()))
        # Other-rater-only structures are ambiguous, never hard background.
        ambiguous = (others > 0) & (current == 0)
        valid = 1-cv2.resize(ambiguous.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
        dark = cv2.resize((channels[2, top:top+size, left:left+size] > .04).astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
        valid = np.where(center.max(0) > 0.1, 1, valid)
        target = dict(center=center, valid=valid, negative_weight=1+2*dark,
                      spine=cv2.resize(spine.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA), items=items)
        boxes = [item['box'] for item in items]
        labels = [1.]*len(boxes)
        occupied = others | current
        for p in proposals:
            b = np.asarray(p['box'], np.float32)
            b[:2] -= [left, top]
            x, y, w, h = b.astype(int)
            if x >= 0 and y >= 0 and x+w <= size and y+h <= size and not occupied[y:y+h, x:x+w].any():
                boxes.append(b.tolist()); labels.append(0.)
                if len(labels)-len(items) >= 8:
                    break
        merged = 0
        for i in range(min(len(items), 4)):
            for j in range(i):
                a, b = np.asarray(items[i]['box']), np.asarray(items[j]['box'])
                lo, hi = np.minimum(a[:2], b[:2]), np.maximum(a[:2]+a[2:], b[:2]+b[2:])
                extent = hi-lo
                if extent.prod() > 1.5*max(a[2:].prod(), b[2:].prod()):
                    # Only a demonstrably non-single-instance merged extent.
                    boxes.append([*lo.tolist(), *extent.tolist()]); labels.append(0.); merged += 1
        target.update(boxes=boxes, box_labels=labels)
        x = torch.from_numpy(channels[:, top:top+size, left:left+size].copy())
        info = dict(stem=stem, record_id=rec.record_id, anchor_id=str(rec.instances[ii].ann_id),
                    size=size, window=[left, top, size], kind=kind, instances=len(items),
                    ambiguous_pixels=int(ambiguous.sum()), hard_negative_boxes=len(labels)-len(items),
                    merged_negative_boxes=merged, slot_overflow=overflow)
        return x, target, info


def proposal_loss(model, output, target, size, positive_weight=1.):
    logits = output['center'][0]
    heat = logits.new_tensor(target['center'])
    p = logits.sigmoid()
    positive = heat == 1
    valid = logits.new_tensor(target['valid'])
    weight = logits.new_tensor(target['negative_weight'])
    count = max(int(positive.sum()), 1)
    loss = (-positive_weight*F.logsigmoid(logits)*(1-p).square()*positive
            - F.logsigmoid(-logits)*p.square()*(1-heat).pow(4)*(~positive)*valid*weight).sum()/count
    geometry = output['geometry'][0].reshape(2, 4, *logits.shape[-2:])
    box_losses, identity_losses, vectors = [], [], []
    for item in target['items']:
        slot, y, x = item['cell']
        box_losses.append(F.smooth_l1_loss(geometry[slot, :, y, x], logits.new_tensor(item['geometry'])))
        cells = item['points']
        if cells:
            ids = torch.as_tensor(cells, device=logits.device)
            vector = output['identity'][0, :, ids[:, 0], ids[:, 1]].mean(1)
            vector = F.normalize(vector, dim=0)
            identity_losses.append(1-(output['identity'][0, :, ids[:, 0], ids[:, 1]]*vector[:, None]).sum(0).mean())
            vectors.append(vector)
    if box_losses:
        loss = loss+torch.stack(box_losses).mean()
    if identity_losses:
        loss = loss+.1*torch.stack(identity_losses).mean()
    if len(vectors) > 1:
        similarity = torch.stack(vectors) @ torch.stack(vectors).T
        upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        loss = loss+.1*F.relu(similarity[upper]-.3).mean()
    spine = logits.new_tensor(target['spine'])
    loss = loss+.5*(F.binary_cross_entropy_with_logits(output['spine'][0, 0], spine, reduction='none')*valid*(1+10*spine)).mean()
    if target['boxes']:
        q = model.score_boxes(output['feature'], target['boxes'], size)
        loss = loss+.5*F.binary_cross_entropy_with_logits(q, q.new_tensor(target['box_labels']))
    return loss


@torch.no_grad()
def decode_tile(model, output, size, limit=64):
    heat = output['center'].sigmoid()
    peaks = heat == F.max_pool2d(heat, 3, 1, 1)
    scores, flat = (heat*peaks).flatten().topk(min(limit, heat.numel()))
    n = heat.shape[-1]
    geometry = output['geometry'][0].reshape(2, 4, n, n)
    boxes, kept, seeds, identities = [], [], [], []
    for score, index in zip(scores.tolist(), flat.tolist()):
        if score <= 0:
            continue
        slot, rem = divmod(index, n*n)
        y, x = divmod(rem, n)
        g = geometry[slot, :, y, x].cpu().numpy()
        cx, cy = (np.asarray([x, y])+np.clip(g[:2], -1, 2))*4
        w, h = np.clip(np.exp(np.clip(g[2:], 0, math.log(2048))), 4, size)
        x0, y0 = np.clip([cx-w/2, cy-h/2], 0, size-4)
        w, h = min(w, size-x0), min(h, size-y0)
        boxes.append([float(x0), float(y0), float(w), float(h)])
        kept.append(score); seeds.append([float(cx), float(cy)])
        identities.append(output['identity'][0, :, y, x].cpu().tolist())
    quality = model.score_boxes(output['feature'], boxes, size).sigmoid().cpu().tolist()
    return [dict(box=b, center=c, objectness=s, box_quality=q, score=s*q, identity=i)
            for b, c, s, q, i in zip(boxes, seeds, kept, quality, identities)]


@torch.no_grad()
def propose(model, gray, device='cpu', limit=256):
    """Overlapping native 512/768 tiles, independent of any GT or rater ID."""
    channels = image_channels(gray)
    pool = []
    for size in (512, 768):
        starts = sorted(set([*range(0, 2048-size+1, size*3//4), 2048-size]))
        for top in starts:
            for left in starts:
                x = torch.from_numpy(channels[:, top:top+size, left:left+size].copy())[None].to(device)
                output = model(x)
                for p in decode_tile(model, output, size):
                    p['box'][0] += left; p['box'][1] += top
                    p['center'][0] += left; p['center'][1] += top
                    p.update(tile=[left, top, size], window=list(fixed_window(p['box'], size)))
                    pool.append(p)
    unique = []
    for p in sorted(pool, key=lambda p: -p['score']):
        a = np.asarray(p['box'])
        duplicate = False
        for q in unique:
            b = np.asarray(q['box'])
            overlap = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])).prod()
            iou = overlap/max(a[2:].prod()+b[2:].prod()-overlap, 1)
            same_seed = np.linalg.norm(np.asarray(p['center'])-q['center']) < 12
            if iou > .7 and same_seed and np.dot(p['identity'], q['identity']) > .5:
                duplicate = True
                break
        if not duplicate:
            unique.append(p)
            if len(unique) == limit:
                break
    for rank, p in enumerate(unique, 1):
        p['rank'] = rank
    return unique
