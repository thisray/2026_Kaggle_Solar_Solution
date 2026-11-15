import cv2
import numpy as np
from scipy import ndimage as ndi
from pycocotools import mask as mu
from solar.select import crop_decode
from solar.candidates import crop_rle
def canonical(rle):
    counts = rle['counts']
    if isinstance(counts, str):
        counts = counts.encode('ascii')
    out = dict(size=list(map(int, rle['size'])), counts=counts)
    if isinstance(counts, list):
        out = mu.frPyObjects(out, *out['size'])
    return out

def matrix(rles, gt):
    a, b = [canonical(r) for r in rles], [canonical(r) for r in gt]
    if a and b and any(r['size'] != a[0]['size'] for r in a+b):
        raise ValueError('Prediction/GT shape mismatch')
    return np.asarray(mu.iou(a, b, [0]*len(b)), float).reshape(len(a), len(b)) if a and b else np.zeros((len(a), len(b)))
def encode(mask,x,y):
    return crop_rle(mask,x,y,[2048,2048]) if mask.any() else None

class Projection:
    def __init__(self,stem,cohort,seeds,prob=None,matrices=None,scales=None):
        self.seeds=seeds; self.seed_crops=[crop_decode(r) for r in seeds]
        self.owner=np.full((2048,2048),-1,np.int16)
        for j,(a,x,y) in enumerate(self.seed_crops):
            region=self.owner[y:y+a.shape[0],x:x+a.shape[1]]
            assert np.all(region[a]<0);region[a]=j
        q=raw_prob(stem,cohort) if prob is None else prob
        matrices=np.load(PIXELS/stem/'matrices.npy') if matrices is None else matrices
        self.scales=np.load(PIXELS/stem/'scales.npy') if scales is None else scales
        self.native=[];self.cache={}
        for j,(a,m) in enumerate(zip(q,matrices)):
            full=cv2.warpAffine(a.astype(np.float32),cv2.invertAffineTransform(m),(2048,2048),flags=cv2.INTER_LINEAR)
            points=full>.2999
            seed,sx,sy=self.seed_crops[j]
            points[sy:sy+seed.shape[0],sx:sx+seed.shape[1]] |= seed
            yy,xx=np.where(points);x0,y0=int(xx.min()),int(yy.min())
            self.native.append((full[y0:int(yy.max())+1,x0:int(xx.max())+1].copy(),x0,y0))

    def proposals(self,t,protect=True):
        ck=(t,protect)
        if ck in self.cache:return self.cache[ck]
        result=[]
        for j,((q,ox,oy),seed) in enumerate(zip(self.native,self.seed_crops)):
            if self.scales[j]<.5 or not np.isfinite(q).all():result.append(seed);continue
            candidate=q>t
            # Native labels exactly reproduce core.project connectivity before owner clipping.
            lab,_=ndi.label(candidate,np.ones((3,3)))
            a,x,y=seed; touched=np.unique(lab[y-oy:y-oy+a.shape[0],x-ox:x-ox+a.shape[1]][a]); touched=touched[touched>0]
            candidate=np.isin(lab,touched) if len(touched) else np.zeros_like(candidate)
            if protect:
                owner=self.owner[oy:oy+q.shape[0],ox:ox+q.shape[1]]
                candidate &= (owner<0)|(owner==j)
            if int(candidate.sum())<96:result.append(seed);continue
            yy,xx=np.where(candidate); x0,y0=int(xx.min()),int(yy.min())
            result.append((candidate[y0:int(yy.max())+1,x0:int(xx.max())+1],x0+ox,y0+oy))
        self.cache[ck]=result;return result

    def apply(self,config,rf):
        t=config.get('threshold',.4);rule=config.get('owner','original')
        crops=list(self.proposals(t,True))
        for j in range(len(crops)):
            if config.get('condition')=='low' and rf[j]>=config['score']:crops[j]=self.seed_crops[j]
            if config.get('condition')=='high' and rf[j]<config['score']:crops[j]=self.seed_crops[j]
        order=list(range(len(crops)))
        if rule=='rf':order=sorted(order,key=lambda j:(-rf[j],j))
        if rule=='reverse':order=order[::-1]
        occupied=np.zeros((2048,2048),bool);out=[None]*len(crops)
        for j in order:
            a,x,y=crops[j];region=occupied[y:y+a.shape[0],x:x+a.shape[1]];a=a&~region
            if int(a.sum())<96:
                a,x,y=self.seed_crops[j];region=occupied[y:y+a.shape[0],x:x+a.shape[1]]
                assert not region[a].any()
            out[j]=encode(a,x,y);region|=a
        return out
def support(rles,bank):
    if not rles:return np.zeros(0)
    if not bank:return np.zeros(len(rles))
    return matrix(rles,bank).max(axis=1)

def keep_consensus(rles,rf,ss,config):
    n=sum((ss[name]>=config['iou']).astype(int) for name in config['systems'])
    return [r for j,r in enumerate(rles) if not(n[j]<config['support'] and rf[j]<config['score'])]

def aligned_scores(pred,reference,rf):
    if not pred:return []
    x=matrix(pred,reference)
    return [float(rf[j]) for j in x.argmax(1)] if reference else [0.]*len(pred)
