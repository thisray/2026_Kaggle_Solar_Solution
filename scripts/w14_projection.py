"""Parameter-free ratio-of-expectations IoU binarization on native crops."""
import numpy as np
from scipy import ndimage as ndi

BINS = 4096

def bin_ids(q):
    return np.minimum((np.clip(q, 0, 1).astype(np.float64)*BINS).astype(np.int32), BINS-1)

def optimal_mask(q):
    flat = q.ravel()
    order = np.argsort(-flat, kind='stable')
    values = flat[order].astype(np.float64)
    if not len(values) or values.sum() <= 0:
        return np.zeros_like(q, dtype=bool), 1., 0
    sums = np.cumsum(values)
    k = np.arange(1, len(values)+1)
    utility = sums/(k + sums[-1] - sums)
    # A scalar threshold must include every pixel in a tied probability block.
    ends = np.r_[values[:-1] != values[1:], True]
    ids = np.flatnonzero(ends)
    best = int(ids[np.argmax(utility[ids])])
    mask = np.zeros(len(flat), bool); mask[order[:best+1]] = True
    threshold = float(values[best])
    return mask.reshape(q.shape), threshold, best+1

class EIoUMixin:
    def proposals(self, t, protect=True):
        if isinstance(t, (float, int)):
            return super().proposals(t, protect)
        mode = t
        ck = (mode, protect)
        if ck in self.cache:
            return self.cache[ck]
        result = []; diagnostics = []
        for j, ((raw, ox, oy), seed) in enumerate(zip(self.native, self.seed_crops)):
            fallback = self.scales[j] < .5 or not np.isfinite(raw).all()
            if fallback:
                result.append(seed); diagnostics.append(dict(fallback=True, threshold=None)); continue
            q = self.calibration[bin_ids(raw)] if mode == 'isotonic' else raw
            candidate, threshold, k = optimal_mask(q)
            if mode == 'clamped':
                threshold = float(np.clip(threshold, .35, .55))
                candidate = q >= threshold
            # Retain W6 connectivity, protected seeds, area fallback and ownership.
            lab, _ = ndi.label(candidate, np.ones((3, 3)))
            a, x, y = seed
            touched = np.unique(lab[y-oy:y-oy+a.shape[0], x-ox:x-ox+a.shape[1]][a])
            touched = touched[touched > 0]
            candidate = np.isin(lab, touched) if len(touched) else np.zeros_like(candidate)
            if protect:
                owner = self.owner[oy:oy+q.shape[0], ox:ox+q.shape[1]]
                candidate &= (owner < 0) | (owner == j)
            fallback = int(candidate.sum()) < 96
            diagnostics.append(dict(fallback=fallback, threshold=threshold, optimal_k=k))
            if fallback:
                result.append(seed); continue
            yy, xx = np.where(candidate); x0, y0 = int(xx.min()), int(yy.min())
            result.append((candidate[y0:int(yy.max())+1, x0:int(xx.max())+1], x0+ox, y0+oy))
        self.diagnostics = getattr(self, 'diagnostics', {})
        self.diagnostics[mode] = diagnostics
        self.cache[ck] = result
        return result
