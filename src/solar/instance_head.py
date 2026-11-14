"""Image-conditioned dynamic instance filters with unrestricted full-image support."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as mask_util

from solar.dense import make_model


def instance_prompt(instance):
    points = np.asarray(instance.spine, dtype=np.float32).reshape(-1, 2)
    if len(points) >= 2:
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.], np.cumsum(lengths)))
        if cumulative[-1] > 0:
            midpoint = cumulative[-1] / 2
            index = min(np.searchsorted(cumulative, midpoint, side="right") - 1, len(lengths) - 1)
            point = points[index] + (points[index + 1] - points[index]) * (
                (midpoint - cumulative[index]) / max(lengths[index], 1e-6))
            return np.clip(point / 2048, 0, 1)
    x, y, width, height = mask_util.toBbox(instance.rle)
    return np.clip(np.asarray([x + width / 2, y + height / 2]) / 2048, 0, 1)


class InstanceHead(nn.Module):
    def __init__(self, mask_size=512):
        super().__init__()
        self.backbone = make_model(pretrained=False)
        self.mask_size = mask_size
        self.mask_features = nn.Conv2d(16, 8, 1)
        self.controller = nn.Sequential(nn.Linear(528, 128), nn.ReLU(), nn.Linear(128, 169))
        nn.init.normal_(self.controller[-1].weight, std=.01)
        nn.init.zeros_(self.controller[-1].bias)

    def encode(self, image):
        features = self.backbone.encoder(image)
        decoded = self.backbone.decoder(features)
        masks = F.interpolate(self.mask_features(decoded), size=(self.mask_size, self.mask_size),
                              mode="bilinear", align_corners=False)
        return masks, decoded, features[-1]

    def masks(self, encoded, prompts, output_size=None):
        shared, local, global_features = encoded
        grid = (prompts * 2 - 1)[None, :, None, :]
        sample = lambda value: F.grid_sample(value, grid, align_corners=False,
                                             padding_mode="border")[0, :, :, 0].T
        parameters = self.controller(torch.cat([sample(local), sample(global_features)], dim=1))
        count, height, width = len(prompts), shared.shape[-2], shared.shape[-1]
        yy, xx = torch.meshgrid((torch.arange(height, device=shared.device) + .5) / height,
                                (torch.arange(width, device=shared.device) + .5) / width, indexing="ij")
        relative = torch.stack([xx[None] - prompts[:, 0, None, None],
                                yy[None] - prompts[:, 1, None, None]], dim=1)
        value = torch.cat([shared.expand(count, -1, -1, -1), relative], dim=1)
        # Three instance-specific 1x1 convolutions, applied as grouped convolutions.
        offset = 0
        for in_channels, out_channels in ((10, 8), (8, 8), (8, 1)):
            length = in_channels * out_channels
            weights = parameters[:, offset:offset + length].reshape(count * out_channels, in_channels, 1, 1)
            offset += length
            bias = parameters[:, offset:offset + out_channels].reshape(-1)
            offset += out_channels
            value = F.conv2d(value.reshape(1, count * in_channels, height, width), weights, bias, groups=count)
            value = value.reshape(count, out_channels, height, width)
            if out_channels != 1:
                value = F.relu(value)
        if output_size is not None:
            value = F.interpolate(value.float(), size=(output_size, output_size), mode="bilinear", align_corners=False)
        return value[:, 0]


def instance_loss(logits, target):
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    axes = (-2, -1)
    positive = (bce * target).sum(axes) / target.sum(axes).clamp_min(1)
    negative = (bce * (1 - target)).sum(axes) / (1 - target).sum(axes).clamp_min(1)
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum(axes) + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    return (.5 * positive + .5 * negative + dice).mean()
