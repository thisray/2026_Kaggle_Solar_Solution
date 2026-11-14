import hashlib
import json
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
MODES=("seed_overlap","largest")
def canonical(rle):
    counts = rle['counts']
    if isinstance(counts, str):
        counts = counts.encode('ascii')
    out = dict(size=list(map(int, rle['size'])), counts=counts)
    if isinstance(counts, list):
        out = mu.frPyObjects(out, *out['size'])
    return out

def key(rle):
    r = canonical(rle)
    return hashlib.sha256(json.dumps(r['size']).encode()+b'|'+r['counts']).hexdigest()

def validate_disjoint(rles):
    """Reject even one intersecting pixel, including identical nonempty masks."""
    rs = [canonical(r) for r in rles]
    if not rs:
        return
    if any(r['size'] != rs[0]['size'] for r in rs):
        raise ValueError('Prediction shape mismatch')
    if any(int(mu.area(r)) == 0 for r in rs):
        raise ValueError('Empty instance is not an omitted prediction')
    if len(rs) > 1:
        ious = np.asarray(mu.iou(rs, rs, [0]*len(rs)))
        if np.any(np.triu(ious, 1) > 0):
            raise ValueError('Overlapping or duplicated predictions')
def components(rle):
    mask = mu.decode(rle).astype(bool)
    x, y, w, h = map(int, mu.toBbox(rle))
    labels, count = ndi.label(mask[y:y+h, x:x+w], np.ones((3, 3)))
    return mask, (x, y, w, h), labels, count

def refine_one(refined, seed, mode):
    if mode not in MODES:
        raise ValueError(mode)
    assert refined['size'] == seed['size'] and int(mu.area(refined)) > 0
    mask, (x, y, w, h), labels, count = components(refined)
    fallback = False
    if count > 1:
        if mode == 'seed_overlap':
            seed_mask = mu.decode(seed).astype(bool)
            touched = np.unique(labels[seed_mask[y:y+h, x:x+w]])
            touched = touched[touched > 0]
            kept = np.isin(labels, touched)
            fallback = not bool(kept.any())
        else:
            areas = np.bincount(labels.ravel(), minlength=count+1)
            areas[0] = 0
            largest = int(areas.argmax())
            kept = labels == largest
            fallback = int(areas[largest]) < 96
        if fallback:
            result = seed
        else:
            result_mask = np.zeros_like(mask, dtype=np.uint8)
            result_mask[y:y+h, x:x+w] = kept
            assert not np.any(result_mask.astype(bool) & ~mask)
            result = mu.encode(np.asfortranarray(result_mask))
    else:
        result = refined
    if key(result) == key(refined):
        after_count = count
        added = removed = 0
    else:
        after_mask, _, _, after_count = components(result)
        added = int(np.count_nonzero(after_mask & ~mask))
        removed = int(np.count_nonzero(mask & ~after_mask))
    assert fallback or added == 0
    return result, dict(fragmented_before=int(count > 1), fragmented_after=int(after_count > 1),
                        changed=int(key(result) != key(refined)), fallback=int(fallback),
                        added_pixels=added, removed_pixels=removed)

def transform(refined, seed, mode, progress=None):
    assert set(refined) == set(seed)
    totals = dict(images=len(refined), instances=0, fragmented_before=0, fragmented_after=0,
                  changed=0, fallback=0, added_pixels=0, removed_pixels=0)
    result = {}
    for index, stem in enumerate(sorted(refined)):
        rs, ss = refined[stem], seed[stem]
        assert len(rs) == len(ss), stem
        validate_disjoint(rs)
        validate_disjoint(ss)
        out = []
        for r, s in zip(rs, ss):
            value, stats = refine_one(r, s, mode)
            out.append(value)
            totals['instances'] += 1
            for name, value in stats.items():
                totals[name] += value
        validate_disjoint(out)
        assert len(out) == len(ss)
        result[stem] = out
        if progress and (index == 0 or (index+1) % 100 == 0):
            progress(images_done=index+1, totals=totals.copy())
    totals.update(disjoint=True, count_unchanged=True, GT_read=False,
                  fallback_exception='Original seed may restore pixels; every output image is revalidated')
    return result, totals
