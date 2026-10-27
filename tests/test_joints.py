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
    m = J.KinematicModel.preset("knee", side)
    for seg in m.segments:
        seg.surface = FakeSurface()
        seg.surface.mesh = object()
        seg.acs = np.eye(4)                     # axes = world: X anterior, Y up, Z right
    return m


@pytest.mark.parametrize("side", ["right", "left"])
def test_knee_signs_match_anatomy(side):
    """Positive flexion, adduction (varus) and internal rotation mean the same motion on both sides."""
    m = knee_model(side)
    femur = pose()
    # 1. Knee flexed 90°: the tibia's long axis (Y, pointing up to the knee) points anteriorly (+X).
    flexed = pose(J.compose("ZXY", np.radians([-90, 0, 0])))
    assert np.allclose(flexed[:3, 1], [1, 0, 0])
    k = m.joint_kinematics("Knee", np.array([femur]), np.array([flexed]))
    np.testing.assert_allclose(k["angles"][0], [90, 0, 0], atol=1e-9)
    # 2. Varus (adduction): the distal tibia moves medially, so the tibia's Y tilts laterally.
    lateral = np.array([0, 0, 1.0]) if side == "right" else np.array([0, 0, -1.0])
    tilt = 10.0
    axis = np.cross([0, 1.0, 0], lateral)               # rotate Y towards lateral
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(np.radians(tilt) * axis).as_matrix()
    k = m.joint_kinematics("Knee", np.array([femur]), np.array([pose(R)]))
    np.testing.assert_allclose(k["angles"][0], [0, tilt, 0], atol=1e-9)
    # 3. Internal rotation: the tibia's anterior axis (X) turns towards medial.
    medial = -lateral
    axis = np.cross([1.0, 0, 0], medial)                # rotate X towards medial (about ±Y)
    R = Rotation.from_rotvec(np.radians(15) * axis).as_matrix()
    k = m.joint_kinematics("Knee", np.array([femur]), np.array([pose(R)]))
    np.testing.assert_allclose(k["angles"][0], [0, 0, 15], atol=1e-9)


@pytest.mark.parametrize("side", ["right", "left"])
def test_knee_translations(side):
    m = knee_model(side)
    lateral = 1 if side == "right" else -1
    tib = pose(t=(5, -40, 3 * lateral))     # 5 mm anterior, 40 mm distal, 3 mm lateral
    k = m.joint_kinematics("Knee", np.array([pose()]), np.array([tib]))
    np.testing.assert_allclose(k["translations"][0], [3, 5, -40], atol=1e-9)


def test_untracked_frames_give_nan():
    m = knee_model("right")
    nanpose = np.full((4, 4), np.nan)
    k = m.joint_kinematics("Knee", np.array([pose(), nanpose]), np.array([pose(), pose()]))
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
    m = J.KinematicModel.preset("knee", "left")
    for f in ("Femur", "Tibia"):
        bm = BoneModel(name=f, path=K / "Models" / f"AUT_Left_{f}.stl")
        bm.build()
        m.segment(f).surface = bm
        m.set_axes_source(f, "landmarks")
    Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses
    Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses
    L = _auto_landmarks(m.segment("Femur").mesh, m.segment("Tibia").mesh, Tf[0], Tt[0])
    m.segment("Femur").landmarks = {"ME": {"method": "point", "points": [L["femur"]["ME"].tolist()]},
                                    "LE": {"method": "point", "points": [L["femur"]["LE"].tolist()]},
                                    "FS": {"method": "region", "points": L["femur"]["FS_points"].tolist()}}
    m.segment("Tibia").landmarks = {"MC": {"method": "point", "points": [L["tibia"]["MC"].tolist()]},
                                    "LC": {"method": "point", "points": [L["tibia"]["LC"].tolist()]},
                                    "TD": {"method": "region", "points": L["tibia"]["TD_points"].tolist()}}
    m.compute()
    assert m.segment("Femur").acs is not None and m.segment("Tibia").acs is not None, \
        (m.segment("Femur").error, m.segment("Tibia").error)
    assert m.checks() == [], m.checks()
    k = m.joint_kinematics("Knee", Tf, Tt)
    ang, tr = k["angles"][0], k["translations"][0]
    print("sample knee angles", ang.round(1), "translations", tr.round(1))
    assert np.all(np.abs(ang) < 45)                    # an upright knee: no wild angles
    assert -80 < tr[2] < 0                             # tibia origin below the femur origin
    # save, reload, same answer
    m.save(tmp_path / "knee.bvrmodel.json")
    m2 = J.KinematicModel.load(tmp_path / "knee.bvrmodel.json")
    np.testing.assert_allclose(m2.joint_kinematics("Knee", Tf, Tt)["angles"], k["angles"], atol=1e-9)
    # a model with "other bone" segments (inertial axes) and a Cardan joint runs too
    g = J.KinematicModel()
    a, b = g.add_segment("generic", "A"), g.add_segment("generic", "B")
    a.surface, b.surface = m.segment("Femur").surface, m.segment("Tibia").surface
    g.add_joint("cardan_xyz", "A", "B")
    g.compute()
    assert np.isfinite(g.joint_kinematics("Joint", Tf, Tt)["angles"]).all()


# ---------- model structure, references, segment motion

def test_segments_and_joints_structure():
    m = J.KinematicModel()
    f = m.add_segment("femur", side="left")
    m.add_segment("tibia", side="left")
    assert f.name == "Femur" and m.add_segment("femur").name == "Femur 2"
    assert f.axes_source == "staple_femur_cylinder"                # automatic by default
    with pytest.raises(J.JointModelError):
        m.add_joint("knee_grood_suntay", "Tibia", "Femur")         # wrong way round
    j = m.add_joint("knee_grood_suntay", "Femur", "Tibia")
    assert j.name == "Knee" and m.joint_side(j) == "left"
    assert [s.name for s in m.candidates("knee_grood_suntay", "proximal")] == ["Femur", "Femur 2"]
    assert m.rename_segment("Tibia", "Left tibia") == "Left tibia" and j.distal == "Left tibia"
    m.remove_segment("Left tibia")
    assert m.joints == []
    g = m.add_segment("generic", "Capitate")
    assert g.side == "none" and m.axes_sources("Capitate") == [("inertial", "Automatic: Principal axes of inertia")]


def _flexing_trial(n=30):
    """Femur fixed; tibia flexes 0 -> 58 deg about the femur Z axis, 40 mm below."""
    P = np.repeat(np.eye(4)[None], n, 0)
    D = np.empty((n, 4, 4))
    for f in range(n):
        D[f] = pose(J.compose("ZXY", np.radians([-2.0 * f, 0, 0])), (0, -40, 0))
    return P, D


@pytest.mark.parametrize("mode", ["subtract", "realign"])
def test_reference_pose_from_a_neutral_trial(mode):
    m = knee_model("right")
    # the tibia axes are off by 7 deg of flexion and 3 deg of rotation relative to the femur's
    m.segment("Tibia").acs = pose(J.compose("ZXY", np.radians([-7, 0, 3])))
    P, D = _flexing_trial()
    raw = m.joint_kinematics("Knee", P, D)["angles"]
    assert abs(raw[0, 0] - 7) < 1e-9 and abs(raw[0, 2] - 3) < 1e-9
    # neutral trial: bones in the frame-0 pose, with a little tracking noise
    rng = np.random.default_rng(0)
    Pn = np.repeat(np.eye(4)[None], 20, 0)
    Dn = np.array([pose(J.compose("XYZ", rng.normal(0, 0.002, 3)), (0, -40, 0) + rng.normal(0, 0.05, 3))
                   for _ in range(20)])
    used = m.set_reference_from_tracking("Knee", Pn, Dn, mode=mode, source={"kind": "trial"})
    assert used == 20
    k = m.joint_kinematics("Knee", P, D)
    np.testing.assert_allclose(k["angles"][0], 0, atol=0.3)
    np.testing.assert_allclose(k["translations"][0], 0, atol=0.2)
    assert abs(k["angles"][-1, 0] - 58) < 0.5                     # flexion change is kept
    np.testing.assert_allclose(m.joint_kinematics("Knee", P, D, normalise=False)["angles"], raw)


def test_reference_from_scan_poses_and_untracked_frames():
    m = knee_model("left")
    P, D = _flexing_trial()
    # scan poses: any common world placement of the frame-10 relative pose (here shifted 5 mm)
    W = pose(J.compose("XYZ", [0.3, -0.2, 1.0]), (5, 5, 5))
    m.set_reference_from_poses("Knee", W @ P[10], W @ D[10], source={"kind": "scan"})
    k = m.joint_kinematics("Knee", P, D)
    np.testing.assert_allclose(k["angles"][10], 0, atol=1e-9)
    with pytest.raises(J.JointModelError):
        m.set_reference_from_tracking("Knee", np.full((3, 4, 4), np.nan), D[:3])


def test_segment_kinematics_about_its_own_axes():
    m = knee_model("right")
    tib = m.segment("Tibia")
    tib.acs = pose(J.compose("XYZ", np.radians([10, 20, 30])), (3, 4, 5))   # axes not aligned with the lab
    n = 5
    T = np.empty((n, 4, 4))
    for f in range(n):            # rotate about the segment's own Y (long) axis, through its origin
        S = tib.acs @ pose(J.compose("ZXY", np.radians([0, 0, 10.0 * f]))) @ np.linalg.inv(tib.acs)
        T[f] = pose(t=(0, 0, 2.0 * f)) @ S
    T[2] = np.nan
    k = m.segment_kinematics("Tibia", T)
    assert np.isnan(k["angles"][2]).all()
    np.testing.assert_allclose(k["angles"][4], [0, 0, 40], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(k["translations"][4]), 8, atol=1e-9)
    lab = m.segment_kinematics("Tibia", T, "lab")
    assert np.isfinite(lab["angles"][0]).all()


def test_version1_files_are_upgraded():
    old = {"type": "bvr-joint-model", "version": 1, "name": "Old knee", "template": "knee_grood_suntay",
           "side": "left", "bones": {"femur": {"surface": None, "landmarks": {}, "axes_source": "femur_spheres"},
                                     "tibia": {"surface": None, "landmarks": {}}}}
    m = J.KinematicModel.from_dict(old)
    assert [s.name for s in m.segments] == ["Femur", "Tibia"] and m.segment("Femur").side == "left"
    assert m.segment("Femur").axes_source == "femur_spheres" and m.segment("Tibia").axes_source == "landmarks"
    # an older method from a file is still offered for that bone, after the current ones
    assert m.axes_sources("Femur")[-2][0] == "femur_spheres"
    assert m.joints[0].name == "Knee" and m.joints[0].type == "knee_grood_suntay"
    d = m.to_dict()
    assert d["version"] == 2 and J.KinematicModel.from_dict(d).joints[0].distal == "Tibia"
