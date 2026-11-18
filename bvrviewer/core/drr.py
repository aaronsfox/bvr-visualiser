"""Digitally reconstructed radiographs (DRRs) of TIF bones for the camera views (Visualiser #25).

Autoscoper's DRR adds up the volume's values along each ray from the X-ray source to a pixel. Here the same
sum is made the other way round: each piece of the volume is projected to the pixel its ray reaches and its
value (times its size along the ray) is added there. To keep playback quick:

* ``prepare`` runs once, when a TIF bone is loaded: the volume is averaged in blocks of f × f × f voxels,
  with f chosen so a bone has at most ``budget`` non-empty blocks, and only those blocks are kept.
* ``render`` projects the blocks at the bone's pose into a grid a few times coarser than the radiograph,
  with the perspective weighting a ray-caster gives (a block covers fewer pixels the further it is from
  the source), so the result is the line integral ∫ value ds in mm along each pixel's ray.

Values are the volume's own, scaled so its brightest voxel is 1. ``DrrVolume.ref`` is a typical thickest
line integral through the bone, so ``render(...) / ref`` is about 1 where the bone is thickest; the display
multiplies it by a brightness and clips to 0–1. Nothing here needs the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import index_to_aut


@dataclass
class DrrVolume:
    points: np.ndarray       # (n, 3) float32: block centres in AUT space (the bone's model space)
    values: np.ndarray       # (n,) float32: block means, 0–1 of the volume's brightest voxel
    block: float             # one block's volume, mm³
    factor: int              # voxels per block along each axis
    ref: float               # a typical thickest line integral through the bone (mm), for brightness 1
    vmin: float              # smallest non-zero block value (the bottom of the cutoff range)
    edge: float = 1.0        # a block's longest side, mm

    @property
    def n(self) -> int:
        return len(self.values)


def prepare(vol: np.ndarray, voxel: tuple[float, float, float], budget: int = 250_000) -> DrrVolume | None:
    """Averaged blocks of a bone's volume (already flipped as the bone is), for ``render``. None if it's empty."""
    vmax = float(vol.max())
    if vmax <= 0:
        return None
    nz = int(np.count_nonzero(vol))
    f = 1
    while nz / f**3 > budget:
        f += 1
    Z, Y, X = vol.shape
    Yb, Xb = -(-Y // f), -(-X // f)
    pad_y, pad_x = Yb * f - Y, Xb * f - X
    ks, js, is_, vals = [], [], [], []
    for kb, k0 in enumerate(range(0, Z, f)):
        slab = vol[k0:k0 + f].astype(np.float32)
        if slab.shape[0] < f or pad_y or pad_x:
            slab = np.pad(slab, ((0, f - slab.shape[0]), (0, pad_y), (0, pad_x)))
        blocks = slab.reshape(f, Yb, f, Xb, f).sum(axis=(0, 2, 4)) / (f**3 * vmax)
        j, i = np.nonzero(blocks)
        if len(j):
            ks.append(np.full(len(j), kb))
            js.append(j)
            is_.append(i)
            vals.append(blocks[j, i])
    if not vals:
        return None
    kb, jb, ib = (np.concatenate(a) for a in (ks, js, is_))
    values = np.concatenate(vals).astype(np.float32)
    # block centres in voxel indices, then AUT space as the bone's surface is built
    c = (f - 1) / 2.0
    pts = index_to_aut(np.column_stack([kb * f + c, jb * f + c, ib * f + c]).astype(float), vol.shape, voxel)
    dx, dy, dz = voxel
    block = float(dx * dy * dz * f**3)
    # reference: the thickest line integrals through the bone along each volume axis (99.5th percentile of the
    # non-empty columns, averaged over the three axes)
    grid = np.zeros((int(kb.max()) + 1, Yb, Xb), np.float32)
    grid[kb, jb, ib] = values
    refs = []
    for axis, length in ((0, dz), (1, dy), (2, dx)):
        cols = grid.sum(axis=axis) * length * f
        cols = cols[cols > 0]
        if len(cols):
            refs.append(float(np.percentile(cols, 99.5)))
    ref = float(np.mean(refs)) if refs else 1.0
    return DrrVolume(points=pts.astype(np.float32), values=values, block=block, factor=f, ref=max(ref, 1e-6),
                     vmin=float(values.min()), edge=float(f * max(voxel)))


def grid_scale(cal, target: int = 512) -> int:
    """How many radiograph pixels one DRR pixel spans (the DRR is about ``target`` pixels across)."""
    return max(1, round(max(cal.width, cal.height) / target))


def render(d: DrrVolume, pose: np.ndarray, cal, scale: int, cutoff: float = 0.0) -> np.ndarray:
    """The bone's DRR at ``pose`` on a grid ``scale`` times coarser than the radiograph (the whole grid)."""
    W, H = -(-cal.width // scale), -(-cal.height // scale)
    out = np.zeros((H, W), np.float32)
    r = render_region(d, pose, cal, scale, cutoff)
    if r is not None:
        img, x0, y0 = r
        out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def render_region(d: DrrVolume, pose: np.ndarray, cal, scale: int, cutoff: float = 0.0):
    """The bone's DRR at ``pose`` on a grid ``scale`` times coarser than the radiograph, as (image, x0, y0): only
    the part of the grid the bone covers, starting at grid pixel (x0, y0). None if it falls outside.

    Each pixel is the line integral of the volume along its ray, divided by ``d.ref`` (so about 1 where the bone
    is thickest), smoothed by about a block's size so it's continuous. ``cutoff`` (0–1) leaves out blocks below
    that fraction of the way from the volume's faintest to its brightest value, as Autoscoper's X-ray cutoff
    does.
    """
    from scipy.ndimage import gaussian_filter

    W, H = -(-cal.width // scale), -(-cal.height // scale)
    pts, vals = d.points, d.values
    if cutoff > 0:
        keep = vals > d.vmin + cutoff * (1.0 - d.vmin)
        pts, vals = pts[keep], vals[keep]
    if not len(vals):
        return None
    R = (cal.R @ pose[:3, :3]).astype(np.float32)          # bone -> camera
    t = (cal.R @ pose[:3, 3] + cal.t).astype(np.float32)
    xc = pts @ R.T + t
    uvw = xc @ cal.K.T.astype(np.float32)
    z = uvw[:, 2]
    ok = np.abs(z) > 1e-6
    u = uvw[ok, 0] / z[ok]
    v = uvw[ok, 1] / z[ok]
    zc = np.abs(xc[ok, 2])
    # a block of volume V at depth z covers V / a of a ray whose cross-section there is a = (z·s / f)²
    fx, fy = float(cal.K[0, 0]), float(cal.K[1, 1])
    w = vals[ok] * d.block * (abs(fx * fy) / (scale * scale)) / zc**2
    sigma = max(0.6, 0.6 * d.edge * abs(fx) / (float(np.median(zc)) * scale))
    # spread each block over the four pixels around where it lands (bilinear), on the part of the grid the
    # bone covers (with room for the smoothing)
    gu, gv = u / scale - 0.5, v / scale - 0.5
    iu, iv = np.floor(gu).astype(np.int64), np.floor(gv).astype(np.int64)
    pad = int(np.ceil(3 * sigma)) + 2
    x0, y0 = max(int(iu.min()) - pad, 0), max(int(iv.min()) - pad, 0)
    x1, y1 = min(int(iu.max()) + pad + 2, W), min(int(iv.max()) + pad + 2, H)
    if x1 <= x0 or y1 <= y0:
        return None
    w_, h_ = x1 - x0, y1 - y0
    fu, fv = (gu - iu).astype(np.float32), (gv - iv).astype(np.float32)
    img = np.zeros(w_ * h_)
    for du, dv_, wt in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        x, y = iu + du - x0, iv + dv_ - y0
        inside = (x >= 0) & (x < w_) & (y >= 0) & (y < h_)
        img += np.bincount(y[inside] * w_ + x[inside], weights=(w * wt)[inside], minlength=w_ * h_)
    img = img.reshape(h_, w_).astype(np.float32) / d.ref
    # blocks (and the slices they come from) can land more than a pixel apart: smooth by about a block's size
    # on the grid so the DRR is continuous rather than striped
    return gaussian_filter(img, sigma), x0, y0
