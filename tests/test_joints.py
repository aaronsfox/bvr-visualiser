"""Tests for joint models: landmarks -> axes -> joint angles."""

import os
from pathlib import Path

import numpy as np
import pytest

from bvrviewer.core import joints as J

SAMPLE = Path(os.environ["BVR_SAMPLE_DATA"]) if os.environ.get("BVR_SAMPLE_DATA") else None
sample = pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="BVR_SAMPLE_DATA not set")


def pose(R=np.eye(3), t=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, t
    return m


@pytest.mark.parametrize("seq", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
def test_cardan_round_trip(seq):
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform([-2.5, -1.4, -2.5], [2.5, 1.4, 2.5])
        np.testing.assert_allclose(J.decompose(J.compose(seq, a), seq), a, atol=1e-9)


class FakeSurface:
    mesh = None

    def to_dict(self):
        return {}


def knee_model(side):
    m = J.JointModel.new(J.KNEE.key, side)
    for e in m.bones.values():
        e.surface = FakeSurface()
        e.surface.mesh = object()
    m.bones["femur"].acs = np.eye(4)          # femur axes = world: X anterior, Y up, Z right
    m.bones["tibia"].acs = np.eye(4)
    return m


@pytest.mark.parametrize("side", ["right", "left"])
def test_knee_signs_match_anatomy(side):
    """Positive flexion, adduction (varus) and internal rotation mean the same motion on both sides."""
    m = knee_model(side)
    femur = pose()
    # 1. Knee flexed 90°: the tibia's long axis (Y, pointing up to the knee) points anteriorly (+X).
    flexed = pose(J.compose("ZXY", np.radians([-90, 0, 0])))
    assert np.allclose(flexed[:3, 1], [1, 0, 0])
    k = m.joint_kinematics("knee", np.array([femur]), np.array([flexed]))
    np.testing.assert_allclose(k["angles"][0], [90, 0, 0], atol=1e-9)
    # 2. Varus (adduction): the distal tibia moves medially, so the tibia's Y tilts laterally.
    lateral = np.array([0, 0, 1.0]) if side == "right" else np.array([0, 0, -1.0])
    tilt = 10.0
    axis = np.cross([0, 1.0, 0], lateral)               # rotate Y towards lateral
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(np.radians(tilt) * axis).as_matrix()
    k = m.joint_kinematics("knee", np.array([femur]), np.array([pose(R)]))
    np.testing.assert_allclose(k["angles"][0], [0, tilt, 0], atol=1e-9)
    # 3. Internal rotation: the tibia's anterior axis (X) turns towards medial.
    medial = -lateral
    axis = np.cross([1.0, 0, 0], medial)                # rotate X towards medial (about ±Y)
    R = Rotation.from_rotvec(np.radians(15) * axis).as_matrix()
    k = m.joint_kinematics("knee", np.array([femur]), np.array([pose(R)]))
    np.testing.assert_allclose(k["angles"][0], [0, 0, 15], atol=1e-9)


@pytest.mark.parametrize("side", ["right", "left"])
def test_knee_translations(side):
    m = knee_model(side)
    lateral = 1 if side == "right" else -1
    tib = pose(t=(5, -40, 3 * lateral))     # 5 mm anterior, 40 mm distal, 3 mm lateral
    k = m.joint_kinematics("knee", np.array([pose()]), np.array([tib]))
    np.testing.assert_allclose(k["translations"][0], [3, 5, -40], atol=1e-9)


def test_untracked_frames_give_nan():
    m = knee_model("right")
    nanpose = np.full((4, 4), np.nan)
    k = m.joint_kinematics("knee", np.array([pose(), nanpose]), np.array([pose(), pose()]))
    assert not np.isnan(k["angles"][0]).any() and np.isnan(k["angles"][1]).all()


@pytest.mark.parametrize("side", ["right", "left"])
def test_landmark_axes(side):
    """ISB femur and tibia recipes give X anterior, Y up, Z right from plausible landmarks."""
    right = np.array([0, 0, 1.0])
    lat = right if side == "right" else -right
    L_f = {"ME": -40 * lat + [2, 0, 0], "LE": 40 * lat + [-2, 0, 0], "FS": np.array([0, 120.0, 0])}
    A = J.RECIPES["isb_femur"](L_f, side, None)
    np.testing.assert_allclose(A[:3, :3], np.eye(3), atol=0.06)
    np.testing.assert_allclose(A[:3, 3], [0, 0, 0], atol=1e-9)
    L_t = {"MC": -35 * lat + [0, -20, 0], "LC": 35 * lat + [0, -20, 0], "TD": np.array([0, -140.0, 0])}
    B = J.RECIPES["knee_tibia"](L_t, side, None)
    np.testing.assert_allclose(B[:3, :3], np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(A[:3, :3]), 1) and np.isclose(np.linalg.det(B[:3, :3]), 1)


def test_inertial_axes_of_a_box():
    import pyvista as pv

    box = pv.Box(bounds=(-10, 10, -40, 40, -5, 5)).triangulate()     # long along Y, thin along Z
    A, moments = J.inertial_frame(box)
    np.testing.assert_allclose(A[:3, 3], 0, atol=1e-9)
    assert abs(abs(A[:3, 0] @ [0, 1, 0]) - 1) < 1e-9                 # axis 1 = long axis
    assert abs(abs(A[:3, 1] @ [1, 0, 0]) - 1) < 1e-9
    V = 20 * 80 * 10
    expected = sorted([V / 12 * (80 ** 2 + 10 ** 2), V / 12 * (20 ** 2 + 10 ** 2), V / 12 * (20 ** 2 + 80 ** 2)])
    np.testing.assert_allclose(moments, expected, rtol=1e-9)
    assert np.isclose(np.linalg.det(A[:3, :3]), 1)


def test_sphere_fit_and_landmark_methods():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    pts = np.array([10, -5, 30]) + 22 * d
    c, r = J.sphere_fit(pts[:60])                      # part of the sphere is enough
    np.testing.assert_allclose(c, [10, -5, 30], atol=1e-9)
    assert abs(r - 22) < 1e-9
    assert J.landmark_value("point", [[1, 2, 3], [4, 5, 6]]).tolist() == [4, 5, 6]
    np.testing.assert_allclose(J.landmark_value("region", [[0, 0, 0], [2, 4, 6]]), [1, 2, 3])
    assert J.landmark_value("region", []) is None


# ---------- knee sample data

def _auto_landmarks(femur_mesh, tibia_mesh, Tf, Tt):
    """Rough landmarks for the sample knee, found from geometry (stand-in for picking by hand)."""
    def world(T, P):
        return (np.c_[P, np.ones(len(P))] @ T.T)[:, :3]

    def model(T, P):
        return (np.c_[P, np.ones(len(P))] @ np.linalg.inv(T).T)[:, :3]

    from bvrviewer.core import auto_axes as AA

    out = {}
    # femur shaft line in its model space, then into the world
    Pm = np.asarray(femur_mesh.points, float)
    Lf = AA.long_axis(femur_mesh, Pm)
    R = Tf[:3, :3]
    up = R @ Lf.axis                                   # knee -> shaft: superior
    Pf, Pt = world(Tf, femur_mesh.points), world(Tt, tibia_mesh.points)
    # femur: shaft band near the top, epicondyles = extremes of the widest direction near the bottom
    top = Pf[Pf @ up > (Pf @ up).max() - 8]
    bot = Pf[Pf @ up < (Pf @ up).min() + 35]
    c = bot.mean(0)
    Q = bot - c
    Q -= np.outer(Q @ up, up)
    _w, V = np.linalg.eigh(np.cov(Q.T))
    ml = V[:, -1]
    # Anterior is perpendicular to the condyle line, and the condyles reach much further behind the
    # shaft line than in front of it. (Measured from the shaft line, not from "up" in the lab: the
    # sample femur leans, which would otherwise reverse the answer.)
    cand = np.cross(up, ml)
    cand /= np.linalg.norm(cand)
    post_m, _ratio = AA.posterior_by_overhang(Pm, Lf, R.T @ cand, (10, 45), at_low_end=True)
    ant = -(R @ post_m)
    right = np.cross(ant, up)                          # ISB: Z = X × Y
    ends = [bot[np.argmax(Q @ ml)], bot[np.argmin(Q @ ml)]]
    left_knee = True                                   # the sample is "Left_*"
    lateral_dir = -right if left_knee else right
    lat_pt = max(ends, key=lambda p: (p - c) @ lateral_dir)
    med_pt = min(ends, key=lambda p: (p - c) @ lateral_dir)
    out["femur"] = {"ME": model(Tf, med_pt[None])[0], "LE": model(Tf, lat_pt[None])[0],
                    "FS": model(Tf, top).mean(0), "FS_points": model(Tf, top)}
    # tibia: shaft band near the bottom, condyles = extremes along the femur's medial-lateral line
    low = Pt[Pt @ up < (Pt @ up).min() + 8]
    hi = Pt[Pt @ up > (Pt @ up).max() - 25]
    lat_t = hi[np.argmax(hi @ lateral_dir)]
    med_t = hi[np.argmin(hi @ lateral_dir)]
    out["tibia"] = {"MC": model(Tt, med_t[None])[0], "LC": model(Tt, lat_t[None])[0],
                    "TD": model(Tt, low).mean(0), "TD_points": model(Tt, low)}
    return out


@sample
def test_knee_sample_end_to_end(tmp_path):
    from bvrviewer.core.models import BoneModel
    from bvrviewer.core.tracking import load_tracking

    K = SAMPLE / "Knee_SampleData"
    m = J.JointModel.new(J.KNEE.key, "left")
    for role, f in [("femur", "Femur"), ("tibia", "Tibia")]:
        bm = BoneModel(name=f, path=K / "Models" / f"AUT_Left_{f}.stl")
        bm.build()
        m.bones[role].surface = bm
    Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses
    Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses
    L = _auto_landmarks(m.bones["femur"].surface.mesh, m.bones["tibia"].surface.mesh, Tf[0], Tt[0])
    m.bones["femur"].landmarks = {"ME": {"method": "point", "points": [L["femur"]["ME"].tolist()]},
                                  "LE": {"method": "point", "points": [L["femur"]["LE"].tolist()]},
                                  "FS": {"method": "region", "points": L["femur"]["FS_points"].tolist()}}
    m.bones["tibia"].landmarks = {"MC": {"method": "point", "points": [L["tibia"]["MC"].tolist()]},
                                  "LC": {"method": "point", "points": [L["tibia"]["LC"].tolist()]},
                                  "TD": {"method": "region", "points": L["tibia"]["TD_points"].tolist()}}
    m.compute()
    assert m.bones["femur"].acs is not None and m.bones["tibia"].acs is not None, \
        (m.bones["femur"].error, m.bones["tibia"].error)
    assert m.checks() == [], m.checks()
    k = m.joint_kinematics("knee", Tf, Tt)
    ang, tr = k["angles"][0], k["translations"][0]
    print("sample knee angles", ang.round(1), "translations", tr.round(1))
    assert np.all(np.abs(ang) < 45)                    # an upright knee: no wild angles
    assert -80 < tr[2] < 0                             # tibia origin below the femur origin
    # save, reload, same answer
    m.save(tmp_path / "knee.bvrmodel.json")
    m2 = J.JointModel.load(tmp_path / "knee.bvrmodel.json")
    np.testing.assert_allclose(m2.joint_kinematics("knee", Tf, Tt)["angles"], k["angles"], atol=1e-9)
    # generic inertial template on the same bones runs too
    g = J.JointModel.new(J.GENERIC.key)
    g.bones["proximal"].surface = m.bones["femur"].surface
    g.bones["distal"].surface = m.bones["tibia"].surface
    g.compute()
    assert np.isfinite(g.joint_kinematics("joint", Tf, Tt)["angles"]).all()
