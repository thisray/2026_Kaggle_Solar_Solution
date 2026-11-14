"""Official-rater instance crops and a frozen DINOv3 conditioned native decoder."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pycocotools import mask as mu

from solar import data
from solar.foundation_dense import CNX, PRETRAINED_REVISIONS


def window(box, center=None):
    x, y, w, h = box
    cx, cy = center if center is not None else (x+w/2, y+h/2)
    need = max(w, h, 2*abs(cx-(x+w/2))+w, 2*abs(cy-(y+h/2))+h)
    size = next((s for s in (512,768,1024) if need <= .90*s), 2048)
    left = int(np.clip(round(cx-size/2), 0, data.SIZE-size))
    top = int(np.clip(round(cy-size/2), 0, data.SIZE-size))
    return left, top, size


def indicator(box, win, expansion=1.):
    left, top, size = win
    x,y,w,h = box
    cx,cy=x+w/2,y+h/2
    x0,y0=round(cx-w*expansion/2-left),round(cy-h*expansion/2-top)
    x1,y1=round(cx+w*expansion/2-left),round(cy+h*expansion/2-top)
    result=np.zeros((size,size),np.float32)
    result[max(0,min(size,y0)):max(0,min(size,y1)),max(0,min(size,x0)):max(0,min(size,x1))]=1
    return result


def area_bin(area):
    return '<400' if area<400 else '400-800' if area<800 else '800-2000' if area<2000 else '>=2000'


def make_inputs(gray, box, win):
    left,top,size=win
    valid=gray[gray>10].astype(np.float32)/255
    mean,std=float(valid.mean()),max(float(valid.std()),1e-6)
    crop=gray[top:top+size,left:left+size].astype(np.float32)/255
    # W30 features have restricted legacy lineage. Keep the two reserved soft
    # channels zero in this official-only pilot, never import legacy weights.
    channels=np.stack([crop,(crop-mean)/std,indicator(box,win),np.zeros_like(crop),np.zeros_like(crop)])
    cx,cy=box[0]+box[2]/2,box[1]+box[3]/2
    outer_left=int(np.clip(round(cx-512),0,1024))
    outer_top=int(np.clip(round(cy-512),0,1024))
    outer=gray[outer_top:outer_top+1024,outer_left:outer_left+1024].astype(np.float32)/255
    context=cv2.resize((outer-mean)/std,(256,256),interpolation=cv2.INTER_AREA)[None]
    return channels,context


class InstanceCrops:
    def __init__(self, records, stems, sampler, seed=261042):
        self.records=records
        self.stems=sorted(stems)
        self.sampler=sampler
        self.seed=seed

    @lru_cache(maxsize=16)
    def image(self,stem):
        return data.load_gray(data.train_image_path(stem))

    @lru_cache(maxsize=32)
    def masks(self,stem,ri):
        return [data.decode(i.rle) for i in self.records[stem][ri].instances]

    def draw(self,index):
        # Both arms draw identical source/rater and geometry RNG sequences.
        rng=np.random.default_rng(self.seed+index)
        stem=self.stems[int(rng.integers(len(self.stems)))]
        ri=int(rng.integers(len(self.records[stem])))
        rec=self.records[stem][ri]
        if not rec.instances:
            return self.draw(index+1000000)
        masks=self.masks(stem,ri)
        unit=rng.random()
        positive=rng.random()<.8
        if self.sampler=='instance':
            ii=min(int(unit*len(rec.instances)),len(rec.instances)-1)
            chosen=masks[ii]
            box=list(map(float,mu.toBbox(rec.instances[ii].rle)))
            cx,cy=box[0]+box[2]/2,box[1]+box[3]/2
        else:
            union=np.logical_or.reduce(masks)
            pixels=np.flatnonzero(union)
            pixel=int(pixels[min(int(unit*len(pixels)),len(pixels)-1)])
            cy,cx=divmod(pixel,data.SIZE)
            ii=next(j for j,m in enumerate(masks) if m[cy,cx])
            chosen=masks[ii]
            box=list(map(float,mu.toBbox(rec.instances[ii].rle)))
        jitter=rng.uniform(-.15,.15,size=4)
        # Uniform foreground controls retain the existing 80% foreground-center
        # policy; both arms retain instance identities as the supervised target.
        if not positive and self.sampler=='foreground':
            cx,cy=rng.uniform(256,1792,size=2)
        cx+=jitter[0]*max(box[2],32)
        cy+=jitter[1]*max(box[3],32)
        win=window(box,(cx,cy))
        if self.sampler=='foreground':
            # Keep the foreground/random crop at the same bbox-based native
            # resolution; do not grow a random crop to encompass a distant GT.
            size=window(box)[2]
            win=(int(np.clip(round(cx-size/2),0,data.SIZE-size)),
                 int(np.clip(round(cy-size/2),0,data.SIZE-size)),size)
        conditioned=[box[0]+jitter[0]*box[2],box[1]+jitter[1]*box[3],
                     box[2]*(1+jitter[2]),box[3]*(1+jitter[3])]
        x,context=make_inputs(self.image(stem),conditioned,win)
        left,top,size=win
        target=np.zeros((5,size,size),np.float32)
        target[0]=chosen[top:top+size,left:left+size]
        points=np.asarray(rec.instances[ii].spine,np.float32).reshape(-1,2)
        if len(points)>=2:
            p=np.rint(points-[left,top]).astype(np.int32)
            cv2.polylines(target[1],[p],False,1,2)
            for channel,point in zip((2,3),(p[0],p[-1])):
                cv2.circle(target[channel],tuple(point),4,1,-1)
        distance=cv2.distanceTransform(target[0].astype(np.uint8),cv2.DIST_L2,3)
        target[4]=np.minimum(distance*2,32)/32
        flipx,flipy=rng.random()<.5,rng.random()<.5
        for axis,flip in ((2,flipx),(1,flipy)):
            if flip:
                x=np.flip(x,axis=axis); target=np.flip(target,axis=axis);context=np.flip(context,axis=axis)
        info=dict(stem=stem,record_id=rec.record_id,ann_id=str(rec.instances[ii].ann_id),
                  area=rec.instances[ii].area,area_bin=area_bin(rec.instances[ii].area),
                  window=win,box=conditioned,nearby_negative_ids=[str(i.ann_id) for j,i in enumerate(rec.instances) if j!=ii],
                  target_pixels=int(target[0].sum()))
        return [torch.from_numpy(a.copy()) for a in (x,context,target)],info


class NativeROI(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        import timm
        self.encoder=timm.create_model(CNX,pretrained=pretrained,in_chans=1,features_only=True,
            pretrained_cfg_overlay={'hf_hub_id':f'timm/{CNX}@{PRETRAINED_REVISIONS[CNX]}'})
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        channels=self.encoder.feature_info.channels()
        self.projections=nn.ModuleList([nn.Conv2d(channels[i],64,1) for i in (0,2)])
        self.query=nn.Sequential(nn.Linear(channels[2]+4,256),nn.GELU(),nn.Linear(256,128))
        self.context=nn.Sequential(nn.Conv2d(1,16,5,stride=4,padding=2),nn.GELU(),nn.AdaptiveAvgPool2d(1))
        self.context_query=nn.Linear(16,128)
        self.coarse=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,16,3,padding=1),nn.GELU())
        self.native=nn.Sequential(nn.Conv2d(21,16,3,padding=1),nn.GELU(),nn.Conv2d(16,5,1))
        self.quality=nn.Linear(128,1)

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self,x,outer):
        with torch.no_grad():
            f=self.encoder(x[:,1:2])
        return self.decode(x,outer,f)

    def decode(self,x,outer,f):
        mask=x[:,2:3]
        small=F.interpolate(mask,size=f[2].shape[-2:],mode='area')
        pooled=(f[2]*small).sum((2,3))/small.sum((2,3)).clamp_min(1)
        geometry=torch.stack([mask.mean((1,2,3)),x[:,0].mean((1,2)),x[:,0].std((1,2)),x[:,1].std((1,2))],dim=1)
        query=self.query(torch.cat([pooled,geometry],dim=1))+self.context_query(self.context(outer).flatten(1))
        scale,bias=query.chunk(2,dim=1)
        a=self.projections[0](f[0]); b=self.projections[1](f[2])
        a=a*(1+scale[:,:,None,None])+bias[:,:,None,None]
        fused=self.coarse(torch.cat([a,F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)],1))
        logits=self.native(torch.cat([x,F.interpolate(fused,size=x.shape[-2:],mode='bilinear',align_corners=False)],1))
        return logits,self.quality(query)


def roi_loss(logits,quality,target):
    mask=target[:,0]
    p=logits[:,0].sigmoid()
    loss=F.binary_cross_entropy_with_logits(logits[:,0],mask)
    loss+=1-(2*(p*mask).sum()+1)/(p.sum()+mask.sum()+1)
    loss+=.5*F.binary_cross_entropy_with_logits(logits[:,1],target[:,1])
    ep=logits[:,2:4].sigmoid()
    y=target[:,2:4]
    loss+=.25*(F.binary_cross_entropy_with_logits(logits[:,2:4],y,reduction='none')*(y*(1-ep)+(1-y)*ep).square()).mean()
    loss+=.25*((logits[:,4].sigmoid()-target[:,4]).abs()*mask).sum()/mask.sum().clamp_min(1)
    # Quality learns the detached IoU>0.5 crossing, rather than an always-one
    # existence target; the target remains defined for empty cropped instances.
    predicted=p.detach()>.5
    iou=(predicted*mask).sum()/(predicted.sum()+mask.sum()-(predicted*mask).sum()).clamp_min(1)
    loss+=.2*F.binary_cross_entropy_with_logits(quality.flatten(),(iou>.5).float().reshape(1))
    return loss


def raw_boxes(gray,limit=200,with_scores=False):
    image=gray.astype(np.float32)/255
    contrast=cv2.GaussianBlur(image,(0,0),8)-image
    disk=(gray>20).astype(np.uint8)
    disk=cv2.erode(disk,np.ones((31,31),np.uint8))
    boxes=[]
    for threshold in (.02,.04,.06):
        binary=((contrast>threshold)&(disk>0)).astype(np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,centroids=cv2.connectedComponentsWithStats(binary)
        for k in range(1,n):
            x,y,w,h,area=map(int,stats[k])
            if area<30 or area>50000 or max(w,h)>1024:
                continue
            score=float(contrast[y:y+h,x:x+w][labels[y:y+h,x:x+w]==k].mean())*math.sqrt(area)
            boxes.append((score,[x,y,w,h]))
    boxes.sort(reverse=True)
    unique=[];scores=[]
    for score,box in boxes:
        if any(np.max(np.abs(np.asarray(box)-b))<5 for b in unique):
            continue
        unique.append(box)
        scores.append(score)
        if len(unique)==limit:
            break
    return [dict(box=box,score=score) for box,score in zip(unique,scores)] if with_scores else unique
