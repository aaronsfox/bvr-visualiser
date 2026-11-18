"""DRRs of TIF bones for the camera views (Visualiser #25)."""

from pathlib import Path

import numpy as np

from bvrviewer.core import drr as D
from bvrviewer.core.calibration import Calibration


def _ball(n=60, r=20, voxel=0.5):
    k, j, i = np.mgrid[:n, :n, :n]
    c = (n - 1) / 2
    vol = (((k - c) ** 2 + (j - c) ** 2 + (i - c) ** 2) <= r * r).astype(np.uint16) * 1000
    vol[(((k - c) ** 2 + (j - c) ** 2 + (i - c) ** 2) <= (r / 2) ** 2)] = 3000      # a denser core
    return vol, (voxel, voxel, voxel)


def _cam(f=4000.0, size=1024):
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], float)
    return Calibration(path=Path("x.json"), K=K, R=np.eye(3), t=np.zeros(3), width=size, height=size)


def _pose_at(vol, voxel, depth):
    """A pose putting the ball's centre on the optical axis at ``depth`` mm."""
    n = vol.shape[0]
    from bvrviewer.core.models import index_to_aut

    c = index_to_aut(np.array([[(n - 1) / 2] * 3]), vol.shape, voxel)[0]
    T = np.eye(4)
    T[:3, 3] = np.array([0, 0, depth]) - c
    return T


def test_prepare_averages_into_blocks_within_budget():
    vol, voxel = _ball()
    d = D.prepare(vol, voxel, budget=5000)
    assert d is not None and d.n <= 5000 and d.factor >= 2
    assert d.values.max() <= 1.0 + 1e-6 and d.values.min() > 0
    assert d.ref > 0
    assert D.prepare(np.zeros((4, 4, 4), np.uint8), voxel) is None


def test_render_is_brightest_through_the_middle_and_centred():
    vol, voxel = _ball()
    d = D.prepare(vol, voxel)
    cal = _cam()
    img = D.render(d, _pose_at(vol, voxel, 400), cal, scale=2)
    assert img.shape == (512, 512)
    y, x = np.unravel_index(np.argmax(img), img.shape)
    assert abs(x - 256) <= 3 and abs(y - 256) <= 3
    # already divided by the reference thickness: about 1 through the middle, fading to the edges
    assert 0.5 < img.max() < 2.0
    assert img[256, 256] > 3 * img[256, 256 + 40]


def test_further_from_the_source_is_smaller_but_not_dimmer():
    vol, voxel = _ball()
    d = D.prepare(vol, voxel)
    cal = _cam()
    near = D.render(d, _pose_at(vol, voxel, 400), cal, scale=2)
    far = D.render(d, _pose_at(vol, voxel, 800), cal, scale=2)
    area = lambda im: int((im > 0.2 * im.max()).sum())     # noqa: E731
    assert 3.0 < area(near) / area(far) < 5.0                # about (800 / 400)² as many pixels
    assert abs(near.max() / far.max() - 1) < 0.25            # the same thickness of bone along the middle ray


def test_cutoff_keeps_only_the_dense_core():
    vol, voxel = _ball()
    d = D.prepare(vol, voxel)
    cal = _cam()
    pose = _pose_at(vol, voxel, 400)
    full = D.render(d, pose, cal, scale=2)
    core = D.render(d, pose, cal, scale=2, cutoff=0.5)
    assert (core > 0.05 * full.max()).sum() < 0.5 * (full > 0.05 * full.max()).sum()
    assert D.render_region(d, pose, cal, scale=2, cutoff=1.0) is None


def test_region_matches_the_full_grid_and_misses_return_none():
    vol, voxel = _ball()
    d = D.prepare(vol, voxel)
    cal = _cam()
    pose = _pose_at(vol, voxel, 400)
    full = D.render(d, pose, cal, scale=2)
    img, x0, y0 = D.render_region(d, pose, cal, scale=2)
    assert np.allclose(full[y0:y0 + img.shape[0], x0:x0 + img.shape[1]], img)
    off = pose.copy()
    off[0, 3] += 5000                                        # far off to the side
    assert D.render_region(d, off, cal, scale=2) is None
    assert D.grid_scale(cal) == 2
