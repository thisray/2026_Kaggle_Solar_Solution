"""Native-pixel, unbounded seed refinement and whole-action quality prediction."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
import torch.nn.functional as F

from solar import data,metric


def encode(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


def windows(rle, size=512, margin=96):
    """Cover the entire expanded seed, without resizing or truncating long seeds."""
    x, y, w, h = map(int, mu.toBbox(rle))
    def positions(lo, hi):
        first = max(0, min(2048-size, lo-margin))
        last = max(first, min(2048-size, hi+margin-size))
        return sorted(set(list(range(first, last+1, size-128))+[last]))
    return [(left, top, size) for top in positions(y, y+h) for left in positions(x, x+w)]


class Refiner(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None,
                            in_channels=6, classes=1, decoder_channels=(128,64,32,16,16))
        nn.init.zeros_(self.net.segmentation_head[0].weight)
        nn.init.zeros_(self.net.segmentation_head[0].bias)

    def forward(self, x):
        # No sigmoid/tanh on the residual: both add and remove logits are unbounded.
        return (2*x[:,1:2]-1)*2 + self.net(x)


class ActionQuality(nn.Module):
    """Predict changes from the final assembled mask, never a mean of tile deltas."""
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old = self.net.conv1
        self.net.conv1 = nn.Conv2d(7,64,7,2,3,bias=False)
        with torch.no_grad():
            self.net.conv1.weight.copy_(old.weight.mean(1,keepdim=True).repeat(1,7,1,1)*3/7)
        self.net.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(512+6,128),nn.ReLU(),nn.Linear(128,2))

    def forward(self, x, geometry):
        return self.head(torch.cat([self.net(x),geometry],1))


def geometry_loss(logits, target):
    p = logits.sigmoid()
    axes = (1,2,3)
    intersection = (p*target).sum(axes)
    union = (p+target-p*target).sum(axes)
    soft_iou = (intersection+1)/(union+1)
    # Empty protection examples still receive BCE and IoU loss.
    return F.binary_cross_entropy_with_logits(logits,target) + (1-soft_iou).mean()


class Cases:
    def __init__(self, root, split="train", target="mean"):
        self.root, self.split, self.target_mode = Path(root), split, target
        import json
        self.plan = json.loads((self.root/"plan.json").read_text())

    @lru_cache(maxsize=3)
    def pack(self, stem):
        with (self.root/self.split/f"{stem}.pkl").open("rb") as stream:
            return pickle.load(stream)

    @lru_cache(maxsize=3)
    def image(self, stem):
        cached = self.root/"image_cache"/self.split/f"{stem}.npy"
        if cached.exists():
            values = np.load(cached,mmap_mode="r")
            return tuple(values[i] for i in range(3))
        path = data.train_image_path(stem) if self.split == "train" else data.test_image_path(stem)
        gray = data.load_gray(path).astype(np.float32)/255
        disk = gray[gray>10/255]
        gray = (gray-float(disk.mean()))/max(float(disk.std()),1e-4)
        maps = []
        for kind in ("fg","spine"):
            values = []
            for directory in self.plan["maps"][self.split]:
                path = Path(directory)/f"{stem}_{kind}.png"
                if path.exists():
                    value = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                    assert value.shape == (2048,2048)
                    values.append(value.astype(np.float32)/255)
            if not values:
                raise FileNotFoundError(f"No {kind} maps for {stem}")
            maps.append(np.mean(values,axis=0))
        return gray,*maps

    @lru_cache(maxsize=12)
    def planes(self, stem, cid):
        pack = self.pack(stem)
        case = pack["cases"][cid]
        seed = data.decode(case["seed"])
        occupied = np.zeros((2048,2048),bool)
        for index, rle in enumerate(pack["selected"]):
            if index != case["selected_index"]:
                occupied |= data.decode(rle)
        return seed,occupied

    @lru_cache(maxsize=12)
    def targets(self, stem, cid):
        case = self.pack(stem)["cases"][cid]
        return [data.decode(r) if r is not None else np.zeros((2048,2048),bool) for r in case["targets"]]

    def input(self, stem, cid, win):
        gray,fg,spine = self.image(stem)
        seed,occupied = self.planes(stem,cid)
        left,top,size = win
        crop = np.s_[top:top+size,left:left+size]
        # A clipped normalization plane adds a robust local contrast input.
        raw = np.clip(gray[crop]/4,-1,1)
        return np.stack([gray[crop],seed[crop],fg[crop],spine[crop],occupied[crop],raw]).astype(np.float32)

    def target(self, stem, cid, win):
        left,top,size = win
        targets = self.targets(stem,cid)
        case = self.pack(stem)["cases"][cid]
        if self.target_mode == "best":
            targets = [targets[int(np.argmax(case["seed_ious"]))]]
        return np.mean([t[top:top+size,left:left+size] for t in targets],axis=0).astype(np.float32)[None]

    def quality_input(self, stem, cid, final):
        gray,fg,spine = self.image(stem)
        seed,occupied = self.planes(stem,cid)
        combined = seed|final
        yy,xx = np.where(combined)
        x0,x1 = max(0,int(xx.min())-96),min(2048,int(xx.max())+97)
        y0,y1 = max(0,int(yy.min())-96),min(2048,int(yy.max())+97)
        crop = np.s_[y0:y1,x0:x1]
        values = [gray,seed,final,fg,spine,occupied,seed^final]
        x = np.stack([cv2.resize(v[crop].astype(np.float32),(256,256),interpolation=cv2.INTER_AREA) for v in values])
        a,b = int(seed.sum()),int(final.sum())
        inter = int((seed&final).sum())
        geometry = np.asarray([np.log1p(a)/16,np.log1p(b)/16,b/max(a,1)/4,
                               inter/max(a+b-inter,1),int((final&occupied).sum())/max(b,1),
                               (a+b-2*inter)/max(a+b,1)],np.float32)
        return x,geometry

    def owned_action(self, stem, cid, final):
        pack = self.pack(stem)
        case = pack["cases"][cid]
        scores = {c["selected_index"]:c["utility"] for c in pack["cases"] if c["selected_index"]>=0}
        result = final.copy()
        for index,rle in enumerate(pack["selected"]):
            # Stable ties preserve the existing selected-mask order.
            if index!=case["selected_index"] and (scores[index]>case["utility"] or (scores[index]==case["utility"] and index<case["selected_index"])):
                result &= ~data.decode(rle)
        return result

    @lru_cache(maxsize=1)
    def records(self):
        return data.cached_records()

    def quality_target(self, stem, cid, final):
        case = self.pack(stem)["cases"][cid]
        targets = self.targets(stem,cid)
        ious = np.asarray([np.count_nonzero(final&t)/max(np.count_nonzero(final|t),1) for t in targets])
        old = np.asarray(case["seed_ious"])
        pack = self.pack(stem)
        scores = {c["selected_index"]:c["utility"] for c in pack["cases"] if c["selected_index"]>=0}
        proposals = [(scores[i],i,r) for i,r in enumerate(pack["selected"]) if i!=case["selected_index"]]
        proposals.append((case["utility"],case["selected_index"] if case["selected_index"]>=0 else len(pack["selected"]),encode(final)))
        occupied = np.zeros((2048,2048),bool)
        after = []
        for _,index,rle in sorted(proposals,key=lambda v:(-v[0],v[1])):
            mask = data.decode(rle)&~occupied
            if mask.sum()>=32:
                after.append(encode(mask));occupied|=mask
        def utility(rles):
            stats = metric.score_image(rles,self.records()[stem])
            q = self.plan["baseline"]["all"]["pq"]
            return np.mean([s.iou_sum-q*(s.tp+.5*(s.fp+s.fn)) for _,s in stats])
        return np.asarray([(ious-old).mean(),utility(after)-utility(pack["selected"])],np.float32)


@torch.inference_mode()
def refine(model, cases, stem, cid, device="cpu", batch=8):
    seed = cases.pack(stem)["cases"][cid]["seed"]
    wins = windows(seed)
    total = np.zeros((2048,2048),np.float32)
    weights = np.zeros_like(total)
    ramp = np.minimum(np.arange(512)+1,np.arange(512,0,-1)).astype(np.float32)
    taper = np.minimum(ramp/64,1)
    taper = taper[:,None]*taper[None,:]
    for first in range(0,len(wins),batch):
        chunk = wins[first:first+batch]
        x = torch.from_numpy(np.stack([cases.input(stem,cid,w) for w in chunk])).to(device)
        values = model(x).sigmoid().float().cpu().numpy()[:,0]
        for (left,top,size),p in zip(chunk,values):
            total[top:top+size,left:left+size] += p*taper
            weights[top:top+size,left:left+size] += taper
    final = data.decode(seed)
    seen = weights>0
    final[seen] = total[seen]/weights[seen]>.5
    # Retain the seed on empty output: selected replacement keeps its instance count.
    return final if final.any() else data.decode(seed)
