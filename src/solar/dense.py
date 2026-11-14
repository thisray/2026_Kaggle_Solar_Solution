"""Dense v1 targets, native crops, model and probability inference."""
from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from solar import data


def wait_for_gpu():
    """Yield the shared GPU while another Python CUDA process is present."""
    while True:
        pause = os.environ.get("SOLAR_GPU_PAUSE_FILE")
        if pause and Path(pause).exists():
            print(f"paused by {pause}", flush=True)
            time.sleep(45)
            continue
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv"], text=True)
        foreign = [row for row in csv.DictReader(output.splitlines(), skipinitialspace=True)
                   if int(row["pid"]) not in (os.getpid(), os.getppid())
                   and "python" in row["process_name"].lower()]
        if not foreign:
            # Establish the CUDA context before slow CPU model initialization so
            # other workers can see this process in their preflight inventory.
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            return
        print(f"waiting for shared GPU: {foreign}", flush=True)
        time.sleep(45)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_targets(records):
    root = data.artifact_root() / "cache/targets_v1"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    for index, (stem, raters) in enumerate(sorted(records.items())):
        fg_path, spine_path = root / f"{stem}_fg.png", root / f"{stem}_spine.png"
        if not fg_path.exists() or not spine_path.exists():
            fg = data.soft_foreground(raters)
            spine = np.zeros((data.SIZE, data.SIZE), np.float32)
            for rater in raters:
                union = np.zeros((data.SIZE, data.SIZE), np.uint8)
                for instance in rater.instances:
                    points = np.asarray(instance.spine, np.float32).reshape(-1, 2)
                    if len(points) >= 2:
                        points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(union, [points], False, 1, thickness=3)
                spine += union
            spine /= max(len(raters), 1)
            for path, target in ((fg_path, fg), (spine_path, spine)):
                if not cv2.imwrite(str(path), np.rint(target * 255).astype(np.uint8)):
                    raise IOError(path)
        if stem not in metadata:
            image = data.load_gray(data.train_image_path(stem))
            valid = image[image > 10].astype(np.float32) / 255
            metadata[stem] = {"mean": float(valid.mean()), "std": max(float(valid.std()), 1e-6),
                              "raters": len(raters)}
        if index % 50 == 0:
            print(f"targets {index + 1}/{len(records)}", flush=True)
            write_json(metadata_path, metadata)
    write_json(metadata_path, metadata)
    return root, metadata


class NativeCrops(Dataset):
    def __init__(self, stems, metadata, crop, length, seed, offset=0):
        self.stems = sorted(stems)
        self.metadata = metadata
        self.crop = crop
        self.length = length
        self.seed = seed
        self.offset = offset
        self.root = data.artifact_root() / "cache/targets_v1"
        self.preloaded = {}

    def preload(self):
        """Decode once in the parent; read-only arrays are shared by fork workers."""
        if sys.platform != "linux":
            return dict(enabled=False, reason="Linux fork only")
        started = time.perf_counter()
        for index, stem in enumerate(self.stems):
            values = NativeCrops.read(self, stem)
            for value in values:
                value.setflags(write=False)
            self.preloaded[stem] = values
            if index % 100 == 0:
                print(f"preload {index + 1}/{len(self.stems)}", flush=True)
        return dict(enabled=True, images=len(self.preloaded),
                    bytes=sum(v.nbytes for values in self.preloaded.values() for v in values),
                    seconds=time.perf_counter() - started, sharing="fork_copy_on_write")

    def __len__(self):
        return self.length

    @lru_cache(maxsize=8)
    def read(self, stem):
        if stem in self.preloaded:
            return self.preloaded[stem]
        image = data.load_gray(data.train_image_path(stem))
        fg = data.load_gray(self.root / f"{stem}_fg.png")
        spine = data.load_gray(self.root / f"{stem}_spine.png")
        positive = np.flatnonzero(fg)
        return image, fg, spine, positive

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index + self.offset)
        stem = self.stems[int(rng.integers(len(self.stems)))]
        image, fg, spine, positive = self.read(stem)
        height, width = image.shape
        if rng.random() < .8 and len(positive):
            center = int(positive[int(rng.integers(len(positive)))])
            cy, cx = divmod(center, width)
            top = int(np.clip(cy - self.crop // 2, 0, height - self.crop))
            left = int(np.clip(cx - self.crop // 2, 0, width - self.crop))
        else:
            top = int(rng.integers(height - self.crop + 1))
            left = int(rng.integers(width - self.crop + 1))
        slices = np.s_[top:top + self.crop, left:left + self.crop]
        x = image[slices].astype(np.float32) / 255
        y = np.stack([fg[slices], spine[slices]]).astype(np.float32) / 255
        turns = int(rng.integers(4))
        x, y = np.rot90(x, turns), np.rot90(y, turns, axes=(1, 2))
        if rng.random() < .5:
            x, y = x[:, ::-1], y[:, :, ::-1]
        x = np.clip(x * rng.uniform(.9, 1.1) + rng.uniform(-.03, .03), 0, 1)
        x = x ** rng.uniform(.9, 1.1)
        stats = self.metadata[stem]
        x = (x - stats["mean"]) / stats["std"]
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(y))


def make_model(pretrained=True, encoder="resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=2)


def dense_loss(logits, targets):
    if (targets < 0).any():
        mask = targets >= 0
        clean = targets.clamp_min(0)
        bce = (F.binary_cross_entropy_with_logits(logits, clean, reduction="none")*mask).sum(dim=(0,2,3))
        bce = bce/mask.sum(dim=(0,2,3)).clamp_min(1)
        probability = logits[:,0].sigmoid()*mask[:,0]
        dice = 1-(2*(probability*clean[:,0]).sum()+1)/(probability.sum()+clean[:,0].sum()+1)
        return bce[0]+dice+.5*bce[1]
    fg_bce = F.binary_cross_entropy_with_logits(logits[:, 0], targets[:, 0])
    probability = logits[:, 0].sigmoid()
    fg_dice = 1 - (2 * (probability * targets[:, 0]).sum() + 1) / (
        probability.sum() + targets[:, 0].sum() + 1)
    spine_bce = F.binary_cross_entropy_with_logits(logits[:, 1], targets[:, 1])
    return fg_bce + fg_dice + .5 * spine_bce


def normalized_image(path):
    image = data.load_gray(path)
    valid = image[image > 10].astype(np.float32) / 255
    mean, std = float(valid.mean()), max(float(valid.std()), 1e-6)
    return (image.astype(np.float32) / 255 - mean) / std


def translate_channels(values, dx, dy):
    """Translate channel-first soft maps in pixel coordinates; positive is right/down."""
    if dx == dy == 0:
        return values
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return np.stack([cv2.warpAffine(channel.astype(np.float32), matrix,
                                  (channel.shape[1], channel.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0) for channel in values])


@torch.inference_mode()
def tiled_probability(model, image, tile=1024, overlap=256):
    height, width = image.shape
    stride = tile - overlap
    ys = sorted(set(list(range(0, max(height - tile, 0) + 1, stride)) + [max(height - tile, 0)]))
    xs = sorted(set(list(range(0, max(width - tile, 0) + 1, stride)) + [max(width - tile, 0)]))
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3).astype(np.float32)
    result = np.zeros((2, height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    for top in ys:
        for left in xs:
            x = torch.from_numpy(np.ascontiguousarray(image[top:top+tile, left:left+tile][None, None])).cuda()
            probability = model(x).sigmoid()[0].cpu().numpy()
            result[:, top:top+tile, left:left+tile] += probability * window
            weight[top:top+tile, left:left+tile] += window
    return result / weight


@torch.inference_mode()
def predict_probability(model, image, tiled=False):
    result = np.zeros((2, *image.shape), np.float32)
    for axes in ((), (1,), (0,), (0, 1)):
        transformed = np.flip(image, axes) if axes else image
        if tiled:
            probability = tiled_probability(model, transformed)
        else:
            x = torch.from_numpy(np.ascontiguousarray(transformed[None, None])).cuda()
            try:
                probability = model(x).sigmoid()[0].cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                del x
                torch.cuda.empty_cache()
                tiled = True
                probability = tiled_probability(model, transformed)
        if axes:
            probability = np.flip(probability, tuple(axis + 1 for axis in axes))
        result += probability * .25
    return result, tiled


def save_probability(directory, stem, probability):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if probability.shape != (2, data.SIZE, data.SIZE) or not np.isfinite(probability).all():
        raise ValueError(f"Invalid probability: {stem}")
    for index, channel in enumerate(("fg", "spine")):
        path = directory / f"{stem}_{channel}.png"
        if not cv2.imwrite(str(path), np.rint(probability[index].clip(0, 1) * 255).astype(np.uint8)):
            raise IOError(path)
