"""Tests for automatic axes from bone geometry."""

import os
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from bvrviewer.core import auto_axes as AA
from bvrviewer.core import joints as J

SAMPLE = Path(os.environ["BVR_SAMPLE_DATA"]) if os.environ.get("BVR_SAMPLE_DATA") else None
sample = pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="BVR_SAMPLE_DATA not set")


def test_cylinder_fit_recovers_axis_and_radius():
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 0.2, -0.1])
    axis /= np.linalg.norm(axis)
    u, v = AA._basis(axis)
    th = rng.uniform(-0.3, 2.2, 3000)                      # only part of the circumference, like condyles
    t = rng.uniform(-35, 35, 3000)
    p0 = np.array([5.0, -3.0, 12.0])
    P = p0 + np.outer(t, axis) + 21.0 * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    P += rng.normal(0, 0.2, P.shape)
    P[:150] += rng.normal(0, 8, (150, 3))                  # outliers
    p, ax, r, rms, keep = AA.fit_cylinder(P, axis + [0, 0.15, 0.1])
    assert abs(abs(ax @ axis) - 1) < 1e-4
    assert abs(r - 21.0) < 0.1
    d = p - p0
    assert np.linalg.norm(d - (d @ axis) * axis) < 0.2       # p is on the true axis
    assert rms < 0.4


def test_robust_sphere_ignores_outliers():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(800, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    d = d[d[:, 2] > 0]                                     # a cap
    P = np.array([1, 2, 3.0]) + 18 * d + rng.normal(0, 0.1, d.shape)
    P = np.r_[P, rng.normal(0, 30, (40, 3))]
    c, r, rms, keep = AA.robust_sphere(P)
    np.testing.assert_allclose(c, [1, 2, 3], atol=0.1)
    assert abs(r - 18) < 0.1


def _toy_bone():
    """A shaft (cylinder, along +Y) on top of a wider block at y < 0, bulging towards -X (posterior)."""
    shaft = pv.Cylinder(center=(0, 60, 0), direction=(0, 1, 0), radius=12, height=120, resolution=60)
    block = pv.Box(bounds=(-40, 10, -30, 0, -40, 40))
    shaft = shaft.triangulate().subdivide_adaptive(max_edge_len=3.0)
    block = block.triangulate().subdivide_adaptive(max_edge_len=3.0)
    return shaft.merge(block)          # overlapping solids: fine for inertia and sections here


def test_long_axis_and_overhang_on_a_toy_bone():
    bone = _toy_bone()
    P = np.asarray(bone.points)
    L = AA.long_axis(bone, P)
    assert L.axis @ [0, 1, 0] > 0.99                       # from the wide end towards the shaft
    post, ratio = AA.posterior_by_overhang(P, L, np.array([1.0, 0, 0]), (2, 25), at_low_end=True)
    assert post @ [-1, 0, 0] > 0.99 and ratio > 2
    L2 = AA.long_axis(bone, P, flip=True)
    assert L2.axis @ [0, -1, 0] > 0.9


# ---------- knee sample

def _sample():
    from test_joints import _auto_landmarks

    from bvrviewer.core.tracking import load_tracking

    K = SAMPLE / "Knee_SampleData"
    F = pv.read(K / "Models" / "AUT_Left_Femur.stl")
    T = pv.read(K / "Models" / "AUT_Left_Tibia.stl")
    Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses[0]
    Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses[0]
    return F, T, _auto_landmarks(F, T, Tf, Tt)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


@sample
@pytest.mark.parametrize("key", [k for k, v in AA.AUTO_METHODS.items()
                                 if v.bone in ("femur", "tibia") and k != "staple_femur_kai"])
def test_auto_methods_agree_with_landmarks_on_the_sample(key):
    """Each automatic method gives axes close to landmark-based ones (same direction, near the same origin)."""
    F, T, L = _sample()
    man = {"femur": J.RECIPES["isb_femur"]({k: L["femur"][k] for k in ("ME", "LE", "FS")}, "left", F),
           "tibia": J.RECIPES["knee_tibia"]({k: L["tibia"][k] for k in ("MC", "LC", "TD")}, "left", T)}
    bone = AA.AUTO_METHODS[key].bone
    r = AA.run(key, F if bone == "femur" else T, "left")
    A, M = r.acs, man[bone]
    assert np.isclose(np.linalg.det(A[:3, :3]), 1)
    for k in range(3):
        assert _angle(A[:3, k], M[:3, k]) < 20, (key, k, _angle(A[:3, k], M[:3, k]))
    assert np.linalg.norm(A[:3, 3] - M[:3, 3]) < 15
    assert r.notes == [], r.notes
    flipped = AA.run(key, F if bone == "femur" else T, "left", {"flip_ap": True}).acs
    # flipping redoes the fit on the other side's surfaces, so it's reversed but not exactly
    assert _angle(flipped[:3, 0], A[:3, 0]) > 120 and _angle(flipped[:3, 1], A[:3, 1]) < 15


@sample
def test_model_with_automatic_axes_and_scan_reference(tmp_path):
    from bvrviewer.core.models import BoneModel
    from bvrviewer.core.tracking import load_tracking
    from bvrviewer.core.transforms import model_to_aut_from_tfm

    K = SAMPLE / "Knee_SampleData"
    m = J.KinematicModel.preset("knee", "left")
    for f in ("Femur", "Tibia"):
        bm = BoneModel(name=f, path=K / "Models" / f"AUT_Left_{f}.stl")
        bm.build()
        m.segment(f).surface = bm
    assert m.segment("Femur").axes_source == "staple_femur_cylinder"
    assert m.segment("Tibia").axes_source == "staple_tibia_kai"
    m.compute()
    assert m.segment("Femur").acs is not None, m.segment("Femur").error
    assert m.checks() == []
    Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses
    Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses
    k = m.joint_kinematics("Knee", Tf, Tt)
    print("auto knee angles", k["angles"][0].round(1), "translations", k["translations"][0].round(1))
    assert np.all(np.abs(k["angles"][0]) < 20)
    # Reference from the scan: each AUT model's pose in CT space is the inverse of its DICOM2AUT
    # mapping. In this sample, frame 1 of the tracking is that same pose (rotated as a whole), so the
    # normalised knee is zero there.
    scan = {f: np.linalg.inv(model_to_aut_from_tfm(K / "Transforms" / f"Left_{f}-DICOM2AUT.tfm"))
            for f in ("Femur", "Tibia")}
    for mode in ("subtract", "realign"):
        m.set_reference_from_poses("Knee", scan["Femur"], scan["Tibia"], mode=mode, source={"kind": "scan"})
        z = m.joint_kinematics("Knee", Tf, Tt)
        np.testing.assert_allclose(z["angles"][0], 0, atol=1e-6)
        np.testing.assert_allclose(z["translations"][0], 0, atol=1e-6)
    m.save(tmp_path / "auto.bvrmodel.json")
    m2 = J.KinematicModel.load(tmp_path / "auto.bvrmodel.json")
    assert m2.segment("Femur").axes_source == "staple_femur_cylinder" and m2.joint("Knee").reference.mode == "realign"
    np.testing.assert_allclose(m2.segment("Femur").acs, m.segment("Femur").acs, atol=1e-9)
    with pytest.raises(J.JointModelError):
        m.set_axes_source("Femur", "tibia_ellipse")
    rows = m.compare_methods("Knee", Tf[0], Tt[0])
    assert len(rows) == 5 * 6          # femur: 4 methods + landmarks; tibia: 4 methods + AAFACT + landmarks
    ok = [r for r in rows if r.get("angles") is not None]
    # Kai's femur method needs the whole femur, AAFACT hasn't been run and landmarks aren't placed:
    # 3 x 4 rows have angles
    assert len(ok) == 3 * 4, [(r["proximal"], r["distal"], r.get("error")) for r in rows if r.get("angles") is None]
    flex = [r["angles"][0] for r in ok]
    print("flexion at frame 1 by method pair:", np.round(flex, 1))
    assert np.ptp(flex) < 15
