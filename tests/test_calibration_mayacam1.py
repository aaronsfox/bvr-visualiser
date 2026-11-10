"""MayaCam 1.x calibrations (5 lines of numbers, e.g. XMALab's *_MayaCam.csv) give the same projection as
Autoscoper itself.

Autoscoper's camera handling (libautoscoper Camera::loadMayaCam1 / loadMayaCam2, CoordFrame) is
re-implemented here step by step, independently of bvrviewer.core.calibration:
- the re-implementation must reproduce our MayaCam 2.0 projection, which was verified against the
  sample radiographs (run when BVR_SAMPLE_DATA is set);
- our MayaCam 1.x reader must then agree with it.
The two files below are a real biplane set-up (Lego calibration cube, 2016 x 2016 images); their
projections of the cube's beads were checked on the calibration images (9 Nov 2026).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bvrviewer.core.calibration import CalibrationError, load_calibration

CAM1 = """-1111.58096582,81.0771441702,1052.81858634
-0.993870320862,-53.8345571013,-1.39252393885
-40.1633538278,21.9311517231,-883.307872093
1409.63353828,788.688482769,-8833.07872093
0.1,2016,2016
"""
CAM2 = """-1270.07321942,92.873584981,-669.348957562
-179.887384356,-59.0638422328,178.653218249
-22.8113177572,6.83726655486,-839.254926731
1236.11317757,939.627334451,-8392.54926731
0.1,2016,2016
"""


# ---------------------------------------------------------------- Autoscoper's camera model, step by step

def _rotate(M, axis, deg):                       # CoordFrame::rotate (rotation_ stored row by row)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    oc = 1 - c
    x, y, z = a
    R = np.array([[oc * x * x + c, oc * x * y + s * z, oc * x * z - s * y],
                  [oc * x * y - s * z, oc * y * y + c, oc * y * z + s * x],
                  [oc * x * z + s * y, oc * y * z - s * x, oc * z * z + c]])
    return M @ R


def _from_xyzypr(v):                             # CoordFrame::from_xyzypr
    M = np.eye(3)
    M = _rotate(M, M[2], v[3])
    M = _rotate(M, M[1], v[4])
    M = _rotate(M, M[0], v[5])
    return M, np.array(v[:3], float)


def _autoscoper_mc1(text):                       # Camera::loadMayaCam1
    v = np.array([[float(x) for x in ln.split(",")] for ln in text.strip().splitlines()])
    M, T = _from_xyzypr([*v[0], v[1][2], v[1][1], v[1][0]])
    f = abs(v[3][2])
    return {"M": M, "T": T, "cx": v[3][0] - 1, "cy": v[3][1] - 1, "fx": f, "fy": f, "H": v[4][2]}


def _autoscoper_mc2(path):                       # Camera::loadMayaCam2
    L = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    size = [float(x) for x in L[1].split(",")]
    K = np.array([[float(x) for x in L[i].split(",")] for i in (3, 4, 5)]).T
    R = np.array([[float(x) for x in L[i].split(",")] for i in (7, 8, 9)]).T
    t = np.array([float(L[i]) for i in (11, 12, 13)])
    t[0], t[2] = -t[0], -t[2]
    R[:, 0], R[:, 2] = -R[:, 0], -R[:, 2]
    K[2][1] = (size[1] - 1) - K[2][1]
    R = R.T.copy()
    t = -np.array([t @ R[:, i] for i in range(3)])
    R[0], R[1] = -R[0], -R[1]
    return {"M": R, "T": t, "cx": K[2][0], "cy": K[2][1], "fx": K[0][0], "fy": K[1][1], "H": size[1]}


def _autoscoper_project(c, X):
    """Camera looks along its local -z; viewport from cx, cy, fx, fy; image rows counted from the top."""
    p = (X - c["T"]) @ c["M"].T
    u = c["cx"] + c["fx"] * p[:, 0] / -p[:, 2]
    v_up = c["cy"] + c["fy"] * p[:, 1] / -p[:, 2]
    return np.c_[u, (c["H"] - 1) - v_up]


def _points_in_front(cal, n=200, seed=0):
    rng = np.random.default_rng(seed)
    centre = cal.source + cal.R.T @ np.array([0, 0, 1.0]) * np.linalg.norm(cal.source)
    return centre + rng.normal(size=(n, 3)) * 50


# ---------------------------------------------------------------- tests

@pytest.mark.parametrize("text", [CAM1, CAM2])
def test_mayacam1_matches_autoscoper(tmp_path, text):
    f = tmp_path / "cam_MayaCam.csv"
    f.write_text(text)
    cal = load_calibration(f)
    assert cal.kind == "MayaCam 1.x" and (cal.width, cal.height) == (2016, 2016)
    assert abs(np.linalg.det(cal.R) - 1) < 1e-9
    X = _points_in_front(cal)
    np.testing.assert_allclose(cal.project(X), _autoscoper_project(_autoscoper_mc1(text), X), atol=1e-6)
    assert np.all(cal.depth(X) > 0)                     # in front of the source
    assert 1300 < np.linalg.norm(cal.source) < 1700        # ~1.5 m from the cube


def test_the_two_cameras_see_the_cube(tmp_path):
    """The calibration cube (Lego, 64 beads 64 mm apart in 4 layers 57.6 mm apart) projects inside both
    images, in roughly the same place as the beads on the calibration images."""
    cube = np.array([[x, y, z] for z in (0, 57.6, 115.2, 172.8) for y in (0, 64, 128, 192) for x in (0, 64, 128, 192)])
    for text, centre in ((CAM1, (1000, 830)), (CAM2, (1020, 900))):
        f = tmp_path / "c.csv"
        f.write_text(text)
        uv = load_calibration(f).project(cube)
        assert np.all((uv > 0) & (uv < 2016))
        assert np.linalg.norm(uv.mean(axis=0) - centre) < 120, uv.mean(axis=0)


def test_bad_mayacam1_files(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(CalibrationError, match="5 lines"):
        load_calibration(f)
    f.write_text("hello\n")
    with pytest.raises(CalibrationError, match="MayaCam"):
        load_calibration(f)


SAMPLES = os.environ.get("BVR_SAMPLE_DATA")


@pytest.mark.skipif(not SAMPLES, reason="BVR_SAMPLE_DATA not set")
@pytest.mark.parametrize("rel", ["2025-01-12-Ankle/Calibration/cam1.txt", "2025-01-12-Ankle/Calibration/cam2.txt",
                                 "2025-02-19-Wrist/Calibration/xr_pre_cube_4_cam01_MayaCam.txt",
                                 "Knee_SampleData/Calibration/Left_Hop_Camera1_Calibration.txt"])
def test_autoscoper_model_reproduces_mayacam2(rel):
    """The re-implementation above gives the same pixels as our (image-verified) MayaCam 2.0 reader."""
    path = Path(SAMPLES) / rel
    cal = load_calibration(path)
    X = _points_in_front(cal)
    np.testing.assert_allclose(_autoscoper_project(_autoscoper_mc2(path), X), cal.project(X), atol=1e-6)
