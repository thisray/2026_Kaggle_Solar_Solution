"""Inverse orthographic solar-surface remapping with Snodgrass rotation."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=32768)
def observer(stamp):
    from astropy.coordinates import get_sun
    from astropy.time import Time
    vector = -get_sun(Time(stamp, scale="utc")).cartesian.xyz.value
    vector /= np.linalg.norm(vector)
    ra, dec = np.deg2rad([286.13, 63.87])
    pole = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    axis = np.cross([0., 0., 1.], pole)
    axis /= np.linalg.norm(axis)
    other = np.cross(pole, axis)
    return float(np.arcsin(np.dot(vector, pole))), float(np.arctan2(np.dot(vector, other), np.dot(vector, axis)))


def rotation_map(geometry, start, end, shape=(2048, 2048)):
    """Return input coordinates sampling start-time pixels on the end canvas.

    x is solar west, y is solar north. Sidereal omega(phi) is
    (2.851 - 0.343 sin(phi)^2 - 0.474 sin(phi)^4) * 1e-6 rad/s.
    Earth longitude is explicitly included; do not subtract it twice.
    Off-disk pixels retain identity; hidden surface pixels sample (-1,-1).
    """
    b0, e0 = observer(start)
    b1, e1 = observer(end)
    seconds = (end-start).total_seconds()
    earth_delta = math.atan2(math.sin(e1-e0), math.cos(e1-e0))
    mx, my = [np.empty(shape, np.float32) for _ in range(2)]
    xx = np.arange(shape[1], dtype=np.float32)[None, :]
    x = (xx-geometry["cx"])/geometry["radius"]
    for top in range(0, shape[0], 128):
        bottom = min(top+128, shape[0])
        yy = np.arange(top, bottom, dtype=np.float32)[:, None]
        y = (geometry["cy"]-yy)/geometry["radius"]
        inside = x*x+y*y < 1
        z = np.sqrt(np.maximum(0, 1-x*x-y*y))
        lat = np.arcsin(np.clip(y*math.cos(b1)+z*math.sin(b1), -1, 1))
        longitude = np.arctan2(x, z*math.cos(b1)-y*math.sin(b1))
        sn = np.sin(lat)
        omega = (2.851-.343*sn**2-.474*sn**4)*1e-6
        source_longitude = longitude-omega*seconds+earth_delta
        cosine = np.cos(lat)*np.cos(source_longitude)
        sy = sn*math.cos(b0)-cosine*math.sin(b0)
        sz = sn*math.sin(b0)+cosine*math.cos(b0)
        sx = np.cos(lat)*np.sin(source_longitude)
        mx[top:bottom] = np.where(inside, geometry["cx"]+geometry["radius"]*sx, xx)
        my[top:bottom] = np.where(inside, geometry["cy"]-geometry["radius"]*sy, yy)
        hidden = inside & (sz < 0)
        mx[top:bottom][hidden] = -1
        my[top:bottom][hidden] = -1
    return mx, my, dict(model="snodgrass_sidereal", b0_source_deg=float(np.rad2deg(b0)),
        b0_target_deg=float(np.rad2deg(b1)), observer_longitude_delta_deg=float(np.rad2deg(earth_delta)), seconds=seconds)
