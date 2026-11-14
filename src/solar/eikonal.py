"""Independent 2-D distance/flow targets and suppressed Euler decoder.

Implemented from elementary distance transforms, gradients and Euler integration;
no Omnipose package or code is used. Fields are generated before any cropping.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def targets(label: np.ndarray) -> np.ndarray:
    """Return native-pixel [distance, vy, vx, boundary] for one whole record.

    Background distance is -1. Later-instance ownership is resolved by the caller.
    Boundary includes the inner one-pixel perimeter of each instance.
    """
    fields = np.zeros((4, *label.shape), np.float32)
    fields[0] = -1
    objects = ndi.find_objects(label)
    for identity, box in enumerate(objects, 1):
        if box is None:
            continue
        box = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2))
                    for s, n in zip(box, label.shape))
        mask = label[box] == identity
        distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        smooth = ndi.gaussian_filter(distance, .8)
        vy, vx = np.gradient(smooth)
        norm = np.hypot(vy, vx)
        vy = np.divide(vy, norm, out=np.zeros_like(vy), where=norm > 1e-6)
        vx = np.divide(vx, norm, out=np.zeros_like(vx), where=norm > 1e-6)
        boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)))
        for channel, values in enumerate((distance, vy, vx, boundary)):
            fields[channel][box][mask] = values[mask]
    return fields


def transform(fields: np.ndarray, turns: int = 0, flip: bool = False) -> np.ndarray:
    """Apply image geometry and rotate/reflect vector components consistently."""
    result = np.rot90(fields, turns, axes=(1, 2)).copy()
    for _ in range(turns % 4):
        vy, vx = result[1].copy(), result[2].copy()
        result[1], result[2] = -vx, vy
    if flip:
        result = result[:, :, ::-1].copy()
        result[2] *= -1
    return result


def decode(fields: np.ndarray, suppressed: bool = True, steps: int = 64,
           return_trace: bool = False, interface_evidence: bool = True) -> np.ndarray:
    """Integrate local flow and partition a transported pixel adjacency graph.

    Suppression decays the Euler step to retain a connected medial ribbon in
    elongated objects. Retaining local adjacency avoids quantized endpoint
    density gaps splitting a continuous ribbon. Adjacent boundary pixels with
    diverging trajectories cut the graph between touching objects. S4b requires
    native flow on both sides to point away from the interface: a transported
    gap alone can arise along the same object's outer perimeter. Zero signed
    projection is the geometric separating plane, not a fitted size threshold.
    No instance IDs, GT masks, boxes, anchors or bank are passed to the decoder.
    """
    distance, vy, vx, boundary = fields
    foreground = distance > .5
    yy, xx = np.nonzero(foreground)
    if not len(yy):
        empty = np.zeros(distance.shape, np.int32)
        return (empty, (yy, xx, np.empty((2, 0), np.float32))) if return_trace else empty
    positions = np.stack((yy, xx)).astype(np.float32)
    for step in range(steps):
        flow = np.stack([ndi.map_coordinates(channel, positions, order=1,
                                             mode="nearest") for channel in (vy, vx)])
        scale = .5 / (1 + step / 4) if suppressed else .5
        proposed = positions + scale * flow
        proposed[0] = proposed[0].clip(0, distance.shape[0] - 1)
        proposed[1] = proposed[1].clip(0, distance.shape[1] - 1)
        valid = ndi.map_coordinates(distance, proposed, order=0, mode="nearest") > .5
        positions[:, valid] = proposed[:, valid]
    index = np.full(distance.shape, -1, np.int32)
    index[yy, xx] = np.arange(len(yy))
    rows, columns = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ny, nx = yy + dy, xx + dx
        valid = (ny >= 0) & (ny < distance.shape[0]) & (nx >= 0) & (nx < distance.shape[1])
        source = np.flatnonzero(valid)
        destination = index[ny[valid], nx[valid]]
        valid = destination >= 0
        source, destination = source[valid], destination[valid]
        separation = ((positions[0, destination] - positions[0, source]) * dy +
                      (positions[1, destination] - positions[1, source]) * dx)
        both_boundary = (boundary[yy[source], xx[source]] > .5) & (boundary[yy[destination], xx[destination]] > .5)
        cut = both_boundary & (separation > dy * dy + dx * dx + .5)
        if interface_evidence:
            source_normal = vy[yy[source], xx[source]] * dy + vx[yy[source], xx[source]] * dx
            destination_normal = vy[yy[destination], xx[destination]] * dy + vx[yy[destination], xx[destination]] * dx
            cut &= (source_normal < 0) & (destination_normal > 0)
        rows.append(source[~cut])
        columns.append(destination[~cut])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = sparse.coo_matrix((np.ones(len(rows), np.uint8), (rows, columns)), shape=(len(yy), len(yy))).tocsr()
    _, identities = connected_components(graph, directed=False)
    result = np.zeros(distance.shape, np.int32)
    result[yy, xx] = identities + 1
    if return_trace:
        return result, (yy, xx, positions)
    return result
