import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
import types
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu
from solar import data
REPO=Path(__file__).resolve().parents[1]
ROOT=data.artifact_root()
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)

def folds():
    with (REPO / "configs/folds_v1.csv").open() as handle:
        return {row["stem"]: int(row["fold"]) for row in csv.DictReader(handle)}

def model_and_criterion(args, legacy=False):
    from rfdetr import RFDETRSegNano, RFDETRSegSmall
    from rfdetr.config import SegmentationTrainConfig
    from rfdetr.models import build_criterion_and_postprocessors
    kind = "nano" if legacy else args.size
    pretrained = ROOT / "w52/rf-detr-seg-nano.pt" if kind == "nano" else args.small_weights
    wrapper = {"nano": RFDETRSegNano, "small": RFDETRSegSmall}[kind](
        resolution=516 if legacy else args.tile, device=args.device, amp=False,
        group_detr=1 if legacy else 13, gradient_checkpointing=not legacy,
        pretrain_weights=str(pretrained))
    wrapper.model.reinitialize_detection_head(2)
    config = wrapper.model.args
    config.num_classes = 1
    for key in ("mask_point_sample_ratio", "mask_ce_loss_coef", "mask_dice_loss_coef", "cls_loss_coef"):
        setattr(config, key, SegmentationTrainConfig.model_fields[key].default)
    criterion, _ = build_criterion_and_postprocessors(config)
    if legacy:
        # Match Pascal FP32 inference; RF-DETR imports otherwise enable TF32.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if not legacy:
        attach_ignore_losses(criterion)
    return wrapper.model.model, criterion, config

def attach_ignore_losses(criterion):
    """Preserve official IA-BCE and mask losses while excluding fragment regions."""
    from rfdetr.util import box_ops
    from rfdetr.models.segmentation_head import (
        calculate_uncertainty, get_uncertain_point_coords_with_randomness, point_sample)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([target["labels"][j] for target, (_, j) in zip(targets, indices)])
        target_boxes = torch.cat([target["boxes"][j] for target, (_, j) in zip(targets, indices)])
        iou = torch.diag(box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][idx].detach()),
                                       box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
        prob = logits.sigmoid()
        positive, negative = torch.zeros_like(logits), prob ** 2
        pos_idx = (*idx, labels)
        quality = torch.clamp(prob[pos_idx].pow(self.focal_alpha) * iou.pow(1 - self.focal_alpha), 0.01).detach()
        positive[pos_idx] = quality.to(positive.dtype)
        negative[pos_idx] = 1 - quality.to(negative.dtype)
        keep_negative = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        with torch.no_grad():
            for b, target in enumerate(targets):
                ignored = target["ignore_mask"].float()
                height, width = ignored.shape
                integral = F.pad(ignored.cumsum(0).cumsum(1), (1, 0, 1, 0))
                boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][b].detach())
                scale = boxes.new_tensor([width, height, width, height])
                boxes = boxes * scale
                lo = boxes[:, :2].floor().long()
                hi = boxes[:, 2:].ceil().long()
                x0, y0 = lo[:, 0].clamp(0, width), lo[:, 1].clamp(0, height)
                x1, y1 = hi[:, 0].clamp(0, width), hi[:, 1].clamp(0, height)
                mass = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
                fraction = mass / ((x1 - x0) * (y1 - y0)).clamp(min=1)
                # Box IoU also catches very thin ignored filaments in large boxes.
                ibox = target["ignore_boxes"]
                near = fraction > 0.3
                if len(ibox):
                    intersection_iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][b].detach()),
                                                       box_ops.box_cxcywh_to_xyxy(ibox))[0]
                    near |= intersection_iou.max(1).values > 0.3
                near[indices[b][0]] = False
                keep_negative[b, near] = False
        negative = negative * keep_negative[:, :, None]
        loss = (negative * logits - F.logsigmoid(logits) * (positive + negative)).sum() / num_boxes
        return {"loss_ce": loss}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        source = outputs["pred_masks"]
        if isinstance(source, torch.Tensor):
            predicted = source[idx]
        else:
            chunks = []
            for b, (queries, _) in enumerate(indices):
                if len(queries):
                    chunks.append(torch.einsum("chw,nc->nhw", source["spatial_features"][b],
                                                source["query_features"][b, queries]) + source["bias"])
            if not chunks:
                zero = source["spatial_features"].sum() * 0
                return {"loss_mask_ce": zero, "loss_mask_dice": zero}
            predicted = torch.cat(chunks)
        if not predicted.numel():
            return {"loss_mask_ce": predicted.sum(), "loss_mask_dice": predicted.sum()}
        expected = torch.cat([target["masks"][j] for target, (_, j) in zip(targets, indices)])[:, None].float()
        valid = torch.cat([(~target["ignore_mask"])[None].expand(len(j), -1, -1)
                           for target, (_, j) in zip(targets, indices)])[:, None].float()
        predicted = predicted[:, None]
        points = max(predicted.shape[-2], predicted.shape[-2] * predicted.shape[-1] // self.mask_point_sample_ratio)
        with torch.no_grad():
            coords = get_uncertain_point_coords_with_randomness(predicted, calculate_uncertainty, points, 3, 0.75)
            truth = point_sample(expected, coords, align_corners=False, mode="nearest")[:, 0]
            weight = point_sample(valid, coords, align_corners=False, mode="nearest")[:, 0]
        logits = point_sample(predicted, coords, align_corners=False)[:, 0]
        ce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weight).sum(1) / weight.sum(1).clamp(min=1)
        prob = logits.sigmoid()
        dice = 1 - (2 * (prob * truth * weight).sum(1) + 1) / ((prob * weight).sum(1) + (truth * weight).sum(1) + 1)
        return {"loss_mask_ce": ce.sum() / num_boxes, "loss_mask_dice": dice.sum() / num_boxes}

    criterion.loss_labels = types.MethodType(loss_labels, criterion)
    criterion.loss_masks = types.MethodType(loss_masks, criterion)

def normalized(gray, device):
    tensor = torch.from_numpy(np.ascontiguousarray(gray)).float().to(device) / 255
    tensor = tensor[None].expand(3, -1, -1)
    return (tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]

def augment_geometry(gray, masks, ignore, rotation, flip_x, flip_y):
    """Apply one coordinate transform to image, positives and ignore pixels."""
    def transform(value, axes):
        value = np.rot90(value, rotation, axes=axes)
        if flip_x:
            value = np.flip(value, axis=axes[1])
        if flip_y:
            value = np.flip(value, axis=axes[0])
        return value
    return transform(gray, (0, 1)), transform(masks, (1, 2)), transform(ignore, (0, 1))
