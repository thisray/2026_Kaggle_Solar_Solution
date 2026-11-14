import numpy as np
from pycocotools import mask as mu
COLUMNS=['support_rf_iou', 'support_rf_score', 'support_rf_coverage', 'support_rf_count50']
def descriptors(rles, items):
    result = np.zeros((len(rles), 4), np.float64)
    if not rles or not items:
        return result
    support = [v['rle'] for v in items]
    overlaps = np.asarray(mu.iou(rles, support, [0] * len(support)))
    best = overlaps.argmax(1)
    result[:, 0] = overlaps[np.arange(len(rles)), best]
    result[:, 1] = np.where(result[:, 0] > 0, np.asarray([v['score'] for v in items])[best], 0.)
    result[:, 3] = (overlaps > .5).sum(1)
    confident = [v['rle'] for v in items if v['score'] >= .3]
    if confident:
        union = mu.merge(confident)
        overlap = np.asarray(mu.iou(rles, [union], [0])).ravel()
        area = np.asarray([mu.area(rle) for rle in rles], float)
        intersection = overlap * (area + float(mu.area(union))) / (1 + overlap)
        result[:, 2] = intersection / np.maximum(area, 1)
    return result
