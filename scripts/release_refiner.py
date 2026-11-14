import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
def transform(mask):
    yy, xx = np.where(mask)
    if not len(xx): raise ValueError('Empty selected seed')
    rect = cv2.minAreaRect(np.column_stack([xx, yy]).astype(np.float32))
    box = cv2.boxPoints(rect); edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1); index = int(lengths.argmax())
    axis = edges[index]; length = float(lengths[index]); width = float(lengths[(index+1)%4])
    scale = min(1., 384/max(2*length, 1), 128/max(width, 1))
    center = np.asarray(rect[0]); angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    matrix[:, 2] += np.array([192., 64.]) - center
    return matrix.astype(np.float64), scale

def inputs(gray, foreground, spine, seeds):
    if gray.shape != (2048, 2048) or foreground.shape != gray.shape or spine.shape != gray.shape:
        raise ValueError('All pixel sources must be native2048; no silent resize')
    valid = gray[gray > 10].astype(np.float32)/255
    mean, std = float(valid.mean()), max(float(valid.std()), 1e-6)
    channels = [(gray.astype(np.float32)/255-mean)/std, foreground.astype(np.float32)/255,
                spine.astype(np.float32)/255]
    masks = [mu.decode(s).astype(bool) for s in seeds]
    union = np.zeros(gray.shape, bool)
    for mask in masks:
        if np.any(union & mask): raise ValueError('Protected A seeds overlap')
        union |= mask
    crops, matrices, scales = [], [], []
    for mask in masks:
        matrix, scale = transform(mask)
        values = channels + [mask.astype(np.float32), (union & ~mask).astype(np.float32)]
        crops.append(np.stack([cv2.warpAffine(v, matrix, (384, 128), flags=cv2.INTER_LINEAR if k < 3 else cv2.INTER_NEAREST)
                               for k, v in enumerate(values)]))
        matrices.append(matrix); scales.append(scale)
    return (np.asarray(crops, np.float16).reshape(-1, 5, 128, 384),
            np.asarray(matrices, np.float64).reshape(-1, 2, 3), np.asarray(scales), dict(mean=mean, std=std))

def project(probabilities, matrices, scales, seeds, threshold, identity=False):
    """Preserve original owners, then use frozen original list order as key tie-break."""
    if identity: return list(seeds), dict(changed=0, total=len(seeds), exact_null=True)
    if len(probabilities) != len(seeds): raise ValueError('Instance alignment mismatch')
    masks = [mu.decode(s).astype(bool) for s in seeds]
    owner = np.full((2048, 2048), -1, np.int16)
    for j, mask in enumerate(masks):
        if np.any(owner[mask] >= 0): raise ValueError('Seed ownership is not disjoint')
        owner[mask] = j
    proposals, fallback = [], []
    for j, (probability, matrix, scale, mask) in enumerate(zip(probabilities, matrices, scales, masks)):
        use_seed = scale < .5 or not np.isfinite(probability).all()
        native = mask
        if not use_seed:
            native = cv2.warpAffine(probability.astype(np.float32), cv2.invertAffineTransform(matrix),
                                    (2048, 2048), flags=cv2.INTER_LINEAR) > threshold
            labels, _ = ndi.label(native, np.ones((3, 3)))
            touched = np.unique(labels[mask]); touched = touched[touched > 0]
            native = np.isin(labels, touched) if len(touched) else np.zeros_like(mask)
            native &= (owner < 0) | (owner == j)
            use_seed = int(native.sum()) < 96
        proposals.append(mask.copy() if use_seed else native); fallback.append(use_seed)
    # Fallback masks are already protected by owner, so background ties cannot erase them.
    occupied = np.zeros((2048, 2048), bool); output = []; changed = 0
    for j, (candidate, seed, mask) in enumerate(zip(proposals, seeds, masks)):
        candidate &= ~occupied
        if int(candidate.sum()) < 96: candidate = mask.copy(); fallback[j] = True
        occupied |= candidate
        unchanged = np.array_equal(candidate, mask)
        output.append(seed if unchanged else mu.encode(np.asfortranarray(candidate.astype(np.uint8))))
        changed += int(not unchanged)
    return output, dict(changed=changed, total=len(seeds), fallback=sum(fallback), exact_null=False)
def make_model(member, pretrained=True):
    import segmentation_models_pytorch as smp
    if member == 'r34':
        return smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights='imagenet' if pretrained else None,in_channels=5,classes=1)
    if member == 'convnext':
        return smp.Unet(encoder_name='tu-convnext_tiny',encoder_weights='imagenet' if pretrained else None,in_channels=5,classes=1)
    raise ValueError(member)

def predict(model, x, flips=True):
    import torch
    model.eval(); values=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            tensor=torch.from_numpy(np.asarray(x[start:start+16],np.float32)).cuda()
            total=None
            for dims in (((),(-1,),(-2,),(-1,-2)) if flips else ((),)):
                p=model(torch.flip(tensor,dims) if dims else tensor).sigmoid()
                if dims:p=torch.flip(p,dims)
                total=p if total is None else total+p
            values.append((total/(4 if flips else 1))[:,0].cpu().numpy())
    return np.concatenate(values) if values else np.zeros((0,128,384),np.float32)
