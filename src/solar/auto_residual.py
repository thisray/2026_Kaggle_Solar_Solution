"""Modern frozen-DINO native refinement with final-mask action supervision."""
from __future__ import annotations
from functools import lru_cache
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pycocotools import mask as mu
from solar import data, metric
from solar.native_instance import NativeInstance


def encoded(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


def stats(rles, records):
    rows = []
    for rec in records:
        st = metric.record_stats(metric.iou_matrix([i.rle for i in rec.instances], rles))
        rows.append([st.iou_sum, st.tp+.5*(st.fp+st.fn)])
    return np.asarray(rows, np.float32)


class FinalActionQuality(nn.Module):
    """Read actual owned final pixels, original seed, and all-image context."""
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(5,16,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.global_view = nn.Sequential(nn.Conv2d(4,16,5,stride=4,padding=2),nn.GELU(),
            nn.Conv2d(16,16,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(52,32),nn.GELU(),nn.Linear(32,2))

    def forward(self, local, global_view, geometry):
        return self.head(torch.cat([self.local(local).flatten(1),self.global_view(global_view).flatten(1),geometry],1))


class AutoResidual(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.native = NativeInstance(pretrained=True)
        self.support = nn.Conv2d(2,5,1)
        self.action_quality = FinalActionQuality()
        if arm == 'B2':
            # Equal initial add/remove logits preserve an exact finite-logit no-op.
            nn.init.zeros_(self.native.native[-1].weight)
            nn.init.zeros_(self.native.native[-1].bias)
            nn.init.zeros_(self.support.weight)
            nn.init.zeros_(self.support.bias)

    def forward(self, x, context):
        raw, _ = self.native(x,context)
        raw = raw+self.support(x[:,3:5])
        if self.arm == 'B2':
            seed = x[:,2:3]
            return (seed*2-1)*2 + 4*(raw[:,0:1].sigmoid()-raw[:,1:2].sigmoid())
        return raw[:,0:1]


def windows(seed_rle, arm):
    x,y,w,h = map(int,mu.toBbox(seed_rle))
    if arm == 'B1':
        size = min(2048,max(384,int(max(w,h)*1.3+48)))
        return [(int(np.clip(x+w/2-size/2,0,2048-size)),int(np.clip(y+h/2-size/2,0,2048-size)),size)]
    # Multiple original-pixel tiles preserve a single instance and its full old mask.
    def positions(low, high):
        first = max(0,min(1536,low-48))
        last = max(first,min(1536,high+48-512))
        values = list(range(first,last+1,384))
        return sorted(set(values+[last]))
    return [(left,top,512) for top in positions(y,y+h) for left in positions(x,x+w)]


class AutoCases:
    def __init__(self, out, plan, records=None):
        self.out,self.plan,self.records = out,plan,records

    @lru_cache(maxsize=4)
    def image(self,stem):
        gray = data.load_gray(data.train_image_path(stem))
        valid = gray[gray>10].astype(np.float32)/255
        normalized = (gray.astype(np.float32)/255-float(valid.mean()))/max(float(valid.std()),1e-6)
        return gray,normalized

    @lru_cache(maxsize=4)
    def auto(self,stem):
        import pickle
        with open(self.out/'auto'/f'{stem}.pkl','rb') as stream:
            return pickle.load(stream)

    @lru_cache(maxsize=4)
    def masks(self,stem):
        pack = self.auto(stem)
        masks = [data.decode(r) for r in pack['rles']]
        labels = np.zeros((2048,2048),np.int32)
        for index, mask in enumerate(masks):
            labels[mask] = index+1
        return masks,labels

    @lru_cache(maxsize=8)
    def targets(self,stem,cid):
        rle = self.auto(stem)['rles'][cid]
        matches = []
        overlaps = []
        for record in self.records[stem]:
            values = metric.iou_matrix([i.rle for i in record.instances],[rle]).ravel()
            best = int(values.argmax()) if len(values) else -1
            matches.append(record.instances[best].rle if best>=0 and values[best]>0 else None)
            overlaps.append(float(values[best]) if best>=0 else 0.)
        # Choose one observed complete shape nearest to the automatic seed.
        j = int(np.argmax(overlaps))
        return matches, matches[j], max(overlaps,default=0.)

    def input(self,stem,cid,win,arm):
        raw,gray = self.image(stem)
        seeds,labels = self.masks(stem)
        seed = seeds[cid]
        occupancy = (labels>0)&(labels!=cid+1)
        fg = self.auto(stem)['fg'].astype(np.float32)/255
        left,top,size = win
        slices = np.s_[top:top+size,left:left+size]
        x = np.stack([raw[slices].astype(np.float32)/255,gray[slices],seed[slices],fg[slices],occupancy[slices]]).astype(np.float32)
        if arm == 'B1':
            x = np.stack([cv2.resize(value,(384,384),interpolation=cv2.INTER_AREA) for value in x])
        context = cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)[None].astype(np.float32)
        return x,context

    def target(self,stem,cid,win,arm):
        matches,chosen,_ = self.targets(stem,cid)
        left,top,size = win
        if arm == 'B1':
            target = sum((data.decode(r)[top:top+size,left:left+size].astype(np.float32) for r in matches if r is not None),start=np.zeros((size,size),np.float32))/len(matches)
            return cv2.resize(target,(384,384),interpolation=cv2.INTER_AREA)
        return data.decode(chosen)[top:top+size,left:left+size].astype(np.float32) if chosen is not None else np.zeros((size,size),np.float32)

    def quality_inputs(self,stem,cid,final,win):
        _,gray = self.image(stem)
        seeds,labels = self.masks(stem)
        seed = seeds[cid]
        occupancy = (labels>0)&(labels!=cid+1)
        fg = self.auto(stem)['fg'].astype(np.float32)/255
        left,top,size = win
        slices = np.s_[top:top+size,left:left+size]
        local = np.stack([gray[slices],seed[slices],final[slices],occupancy[slices],fg[slices]]).astype(np.float32)
        if size!=512:
            local = np.stack([cv2.resize(v,(512,512),interpolation=cv2.INTER_AREA) for v in local])
        global_view = np.stack([cv2.resize(v.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA) for v in (gray,seed,final,occupancy)])
        oldarea=max(1,int(seed.sum()))
        geom=np.array([np.log1p(final.sum())/16, np.log1p(oldarea)/16,
            np.clip(final.sum()/oldarea,0,4)/4,np.count_nonzero(final!=seed)/max(1,final.sum()+oldarea)],np.float32)
        return local,global_view,geom

    def final_from_probability(self,stem,cid,win,probability):
        seeds,labels = self.masks(stem)
        final = seeds[cid].copy()
        left,top,size = win
        if probability.shape!=(size,size):
            probability=cv2.resize(probability,(size,size),interpolation=cv2.INTER_LINEAR)
        final[top:top+size,left:left+size]=probability>.5
        final[(labels>0)&(labels!=cid+1)] = False
        return final

    def action_target(self,stem,cid,final):
        rles = self.auto(stem)['rles']
        before = stats(rles,self.records[stem])
        changed = list(rles)
        if final.any():
            changed[cid] = encoded(final)
        else:
            # An empty final shape is object deletion, not an extra empty FP.
            changed.pop(cid)
        after = stats(changed,self.records[stem])
        return (after-before).mean(0)


def geometry_loss(logit,target):
    probability=logit.sigmoid()
    return F.binary_cross_entropy_with_logits(logit,target)+1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
