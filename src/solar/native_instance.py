"""Dense dynamic-query instance decoding using clean frozen foundation pixels."""
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pycocotools import mask as mu

from solar import data
from solar.foundation_dense import CNX, PRETRAINED_REVISIONS
from solar.native_roi import InstanceCrops, area_bin, make_inputs, window


class EqualInstanceCrops(InstanceCrops):
    """Sample each annotation with equal probability, retaining rater partitions."""
    def __init__(self, records, stems, seed=261042):
        super().__init__(records, stems, 'instance', seed)
        self.items = [(s, ri, ii) for s in self.stems
                      for ri, rec in enumerate(records[s]) for ii in range(len(rec.instances))]

    def draw(self, index):
        rng = np.random.default_rng(self.seed + index)
        stem, ri, ii = self.items[int(rng.integers(len(self.items)))]
        rec = self.records[stem][ri]
        instance = rec.instances[ii]
        chosen = self.masks(stem, ri)[ii]
        box = mu.toBbox(instance.rle).astype(float)
        jitter = rng.uniform(-.15, .15, 4)
        center = box[:2] + box[2:] / 2 + jitter[:2] * np.maximum(box[2:], 32)
        win = window(box, center)
        conditioned = np.r_[box[:2] + jitter[:2] * box[2:], box[2:] * (1 + jitter[2:])]
        x, context = make_inputs(self.image(stem), conditioned, win)
        left, top, size = win
        target = np.zeros((5, size, size), np.float32)
        target[0] = chosen[top:top+size, left:left+size]
        points = np.asarray(instance.spine, np.float32).reshape(-1, 2)
        if len(points) >= 2:
            p = np.rint(points - [left, top]).astype(np.int32)
            cv2.polylines(target[1], [p], False, 1, 2)
            for channel, point in zip((2, 3), (p[0], p[-1])):
                cv2.circle(target[channel], tuple(point), 4, 1, -1)
        distance = cv2.distanceTransform(target[0].astype(np.uint8), cv2.DIST_L2, 3)
        target[4] = np.minimum(distance * 2, 32) / 32
        flips = [bool(rng.random() < .5), bool(rng.random() < .5)]
        for axis, flip in zip((2, 1), flips):
            if flip:
                x = np.flip(x, axis=axis)
                target = np.flip(target, axis=axis)
                context = np.flip(context, axis=axis)
        nearby = [str(other.ann_id) for j, other in enumerate(rec.instances)
                  if j != ii and self.masks(stem, ri)[j][top:top+size, left:left+size].any()]
        info = dict(stem=stem, record_id=rec.record_id, ann_id=str(instance.ann_id),
                    area=instance.area, area_bin=area_bin(instance.area), window=list(win),
                    box=conditioned.tolist(), flips=flips, nearby_negative_ids=nearby,
                    target_pixels=int(target[0].sum()))
        assert info['target_pixels'] == instance.area
        return [torch.from_numpy(a.copy()) for a in (x, context, target)], info


def prompt_planes(mask):
    """Box center and relative coordinates inferred solely from the input box."""
    b, _, h, w = mask.shape
    ys = torch.arange(h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    mass = mask.sum((2, 3), keepdim=True).clamp_min(1)
    cx = (mask * xs).sum((2, 3), keepdim=True) / mass
    cy = (mask * ys).sum((2, 3), keepdim=True) / mass
    bw = mask.amax(2, keepdim=True).sum(3, keepdim=True).clamp_min(4)
    bh = mask.amax(3, keepdim=True).sum(2, keepdim=True).clamp_min(4)
    dx = ((xs - cx) / bw).expand(b, 1, h, w).clamp(-4, 4)
    dy = ((ys - cy) / bh).expand(b, 1, h, w).clamp(-4, 4)
    heat = torch.exp(-2 * (dx.square() + dy.square()))
    return torch.cat([mask, heat, dx, dy], 1)


class DynamicPixels(nn.Module):
    """Query-generated matrices act on every FPN pixel, rather than FiLM."""
    def __init__(self, query_dim=96, channels=36, hidden=16, output=8):
        super().__init__()
        self.channels, self.hidden, self.output = channels, hidden, output
        self.generator = nn.Linear(query_dim, channels*hidden+hidden+hidden*output+output)
        nn.init.normal_(self.generator.weight, std=.01)
        nn.init.zeros_(self.generator.bias)

    def forward(self, pixels, query):
        b, _, h, w = pixels.shape
        weights = self.generator(query)
        n1 = self.channels * self.hidden
        n2 = n1 + self.hidden
        n3 = n2 + self.hidden * self.output
        w1 = weights[:, :n1].reshape(b, self.hidden, self.channels) / math.sqrt(self.channels)
        b1 = weights[:, n1:n2, None]
        w2 = weights[:, n2:n3].reshape(b, self.output, self.hidden) / math.sqrt(self.hidden)
        b2 = weights[:, n3:, None]
        value = F.gelu(torch.bmm(w1, pixels.flatten(2)) + b1)
        return (torch.bmm(w2, value) + b2).reshape(b, self.output, h, w)


class NativeInstance(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(CNX, pretrained=pretrained, in_chans=1, features_only=True,
            pretrained_cfg_overlay={'hf_hub_id': f'timm/{CNX}@{PRETRAINED_REVISIONS[CNX]}'})
        self.encoder.requires_grad_(False)
        assert self.encoder.feature_info.reduction()[:3] == [4, 8, 16]
        channels = self.encoder.feature_info.channels()
        self.laterals = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in channels[:3]])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.GELU()) for _ in range(3)])
        self.prompt = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.context = nn.Sequential(nn.Conv2d(1, 16, 5, stride=4, padding=2), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(1))
        self.query = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 96))
        self.dynamic = nn.ModuleList([DynamicPixels() for _ in range(3)])
        self.fuse = nn.Sequential(nn.Conv2d(72, 32, 3, padding=1), nn.GroupNorm(4, 32),
                                  nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.native = nn.Sequential(nn.Conv2d(38, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 5, 1))
        # Separate random-rater hit probability and conditional hard IoU.
        self.quality = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, outer):
        with torch.no_grad():
            features = self.encoder(x[:, 1:2])
        return self.decode(x, outer, features)

    def decode(self, x, outer, features):
        planes = prompt_planes(x[:, 2:3])
        fpn = [layer(f) for layer, f in zip(self.laterals, features[:3])]
        for i in (1, 0):
            fpn[i] = fpn[i] + F.interpolate(fpn[i+1], size=fpn[i].shape[-2:], mode='bilinear', align_corners=False)
        fpn = [layer(f) for layer, f in zip(self.smooth, fpn)]
        prompts = F.interpolate(planes, size=fpn[0].shape[-2:], mode='bilinear', align_corners=False)
        encoded_prompt = self.prompt(prompts)
        pool_mask = F.interpolate(x[:, 2:3], size=fpn[2].shape[-2:], mode='area')
        pooled = (fpn[2]*pool_mask).sum((2, 3)) / pool_mask.sum((2, 3)).clamp_min(1)
        query = self.query(torch.cat([pooled, encoded_prompt.mean((2, 3)), self.context(outer).flatten(1)], 1))
        conditioned = []
        for layer, f in zip(self.dynamic, fpn):
            local_prompt = F.interpolate(planes, size=f.shape[-2:], mode='bilinear', align_corners=False)
            response = layer(torch.cat([f, local_prompt], 1), query)
            conditioned.append(F.interpolate(response, size=fpn[0].shape[-2:], mode='bilinear', align_corners=False))
        dense = self.fuse(torch.cat([fpn[0], encoded_prompt, *conditioned], 1))
        native = F.interpolate(dense, size=x.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.native(torch.cat([x[:, :2], planes, native], 1))
        mask_weights = F.interpolate(logits[:, :1].sigmoid(), size=dense.shape[-2:], mode='area')
        appearance = (dense*mask_weights).sum((2, 3)) / mask_weights.sum((2, 3)).clamp_min(1)
        return logits, self.quality(torch.cat([query, appearance], 1))


def instance_loss(logits, quality, target, quality_target=None):
    mask = target[:, 0]
    probability = logits[:, 0].sigmoid()
    intersection = (probability * mask).sum()
    union = probability.sum() + mask.sum() - intersection
    soft_iou = (intersection + 1) / (union + 1)
    loss = F.binary_cross_entropy_with_logits(logits[:, 0], mask)
    loss += 1 - (2 * intersection + 1) / (probability.sum() + mask.sum() + 1)
    near = ((soft_iou.detach() >= .35) & (soft_iou.detach() <= .60)).float()
    loss += .1 * near * F.relu(.55 - soft_iou)
    loss += .5 * F.binary_cross_entropy_with_logits(logits[:, 1], target[:, 1])
    ep = logits[:, 2:4].sigmoid()
    y = target[:, 2:4]
    loss += .25 * (F.binary_cross_entropy_with_logits(logits[:, 2:4], y, reduction='none') *
                    (y*(1-ep)+(1-y)*ep).square()).mean()
    loss += .25 * ((logits[:, 4].sigmoid()-target[:, 4]).abs()*mask).sum() / mask.sum().clamp_min(1)
    if quality_target is None:
        pred = probability.detach() > .5
        hit_iou = (pred*mask).sum() / (pred.sum()+mask.sum()-(pred*mask).sum()).clamp_min(1)
        quality_target = torch.stack([(hit_iou > .5).float(), hit_iou]).reshape(1, 2)
    loss += .2 * F.binary_cross_entropy_with_logits(quality[:, 0], quality_target[:, 0])
    if (quality_target[:, 0] > 0).any():
        loss += .1 * F.smooth_l1_loss(quality[:, 1].sigmoid()[quality_target[:, 0] > 0],
                                    quality_target[:, 1][quality_target[:, 0] > 0])
    return loss
