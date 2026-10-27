"""Tests for the STAPLE translation (core/staple).

The reference checks need a clone of https://github.com/modenaxe/msk-STAPLE (environment variable
STAPLE_REPO) and the mat-io package to read its example bones; they compare our results with the
joint frames in STAPLE's own reference OpenSim models.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from bvrviewer.core.staple import tibia as TB
from bvrviewer.core.staple import femur as FM
from bvrviewer.core.staple.common import tri_from_pyvista
from bvrviewer.core.staple.fitting import fit_ellipse, lscylinder, sphereFit
from bvrviewer.core.staple.tri import (Tri, TriCurvature, TriDilateMesh, TriErodeMesh, TriFillPlanarHoles,
                                       TriInertiaPpties, TriPlanIntersect, TriReduceMesh, ensure_outward)

STAPLE = Path(os.environ["STAPLE_REPO"]) if os.environ.get("STAPLE_REPO") else None
SAMPLE = Path(os.environ["BVR_SAMPLE_DATA"]) if os.environ.get("BVR_SAMPLE_DATA") else None


def _sphere(r=20.0, res=60):
    return tri_from_pyvista(pv.Sphere(radius=r, theta_resolution=res, phi_resolution=res), target_faces=None)


# ---------------------------------------------------------------- mesh basics

def test_inertia_of_a_box():
    box = tri_from_pyvista(pv.Box(bounds=(0, 40, 0, 20, 0, 10)).triangulate(), target_faces=None)
    V, C, I, D, vol = TriInertiaPpties(box)
    assert np.isclose(vol, 8000)
    np.testing.assert_allclose(C, [20, 10, 5], atol=1e-9)
    assert abs(V[:, 0] @ [1, 0, 0]) > 0.999          # smallest moment along the longest side
    np.testing.assert_allclose(D[0], vol * (20 ** 2 + 10 ** 2) / 12, rtol=1e-9)


def test_inward_faces_are_reoriented():
    s = _sphere()
    flipped = Tri(s.ConnectivityList[:, [0, 2, 1]], s.Points)
    assert TriInertiaPpties(ensure_outward(flipped))[4] > 0


def test_plane_section_area_and_hole():
    s = _sphere(20.0, 120)
    curves, area = TriPlanIntersect(s, [0, 0, 1], -5.0)       # z = 5
    assert len(curves) == 1
    assert abs(area - np.pi * (20 ** 2 - 5 ** 2)) / area < 0.01
    outer, inner = _sphere(20.0, 120), _sphere(10.0, 120)
    hollow = Tri(np.vstack([outer.ConnectivityList, inner.ConnectivityList[:, [0, 2, 1]] + outer.n_points]),
                 np.vstack([outer.Points, inner.Points]))
    curves, area = TriPlanIntersect(hollow, [1, 0, 0], -0.37)   # not through vertices
    assert len(curves) == 2 and sorted(c["Hole"] for c in curves) == [-1, 1]
    assert abs(area - np.pi * (20 ** 2 - 10 ** 2)) / area < 0.01


def test_curvature_of_a_sphere():
    s = _sphere(25.0, 80)
    Cmean, Cg = TriCurvature(s)[:2]
    # outward normals: the quadric opens away from the normal, so mean curvature is -1/R
    assert abs(np.median(np.abs(Cmean)) - 1 / 25) < 0.002
    assert abs(np.median(Cg) - 1 / 625) < 2e-4


def test_morphology_and_holes():
    s = _sphere(20.0, 60)
    cap = TriReduceMesh(s, np.nonzero(s.incenter[:, 2] > 15)[0])
    grown = TriDilateMesh(s, cap, 2)
    assert grown.n_faces > cap.n_faces
    assert TriErodeMesh(grown, 2).n_faces < grown.n_faces
    closed = TriFillPlanarHoles(TriReduceMesh(s, np.nonzero(s.incenter[:, 2] < 10)[0]))
    assert len(closed.freeBoundary) == 0
    assert TriInertiaPpties(closed)[4] > 0


def test_fits():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    P = np.array([3.0, -2, 7]) + 12 * d / np.linalg.norm(d, axis=1, keepdims=True)
    C, R, _ = sphereFit(P)
    np.testing.assert_allclose(C, [3, -2, 7], atol=1e-9)
    assert np.isclose(R, 12)
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    x, y = 4 + 10 * np.cos(th), -1 + 5 * np.sin(th)
    c, s = np.cos(0.3), np.sin(0.3)
    E = fit_ellipse(c * x - s * y, s * x + c * y)
    assert np.isclose(max(E["a"], E["b"]), 10, atol=1e-6) and np.isclose(min(E["a"], E["b"]), 5, atol=1e-6)
    ax = np.array([1.0, 0.1, 0.05]) / np.linalg.norm([1.0, 0.1, 0.05])
    u = np.cross(ax, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    t = rng.uniform(-30, 30, 2000)
    a = rng.uniform(0, 2.5, 2000)
    Q = np.outer(t, ax) + 20 * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v)) + [1, 2, 3]
    x0, an, rn, _ = lscylinder(Q, [1, 2, 3], ax + [0, 0.1, 0], 18)
    assert abs(abs(an @ ax) - 1) < 1e-8 and np.isclose(rn, 20)


# ---------------------------------------------------------------- against STAPLE's reference models

def _rot_xyz(a, b, c):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _osim_frames(path):
    out = {}
    for j in ET.parse(path).getroot().iter("CustomJoint"):
        for f in j.iter("PhysicalOffsetFrame"):
            body = f.find("socket_parent").text.split("/")[-1]
            out[(j.get("name"), body)] = (_rot_xyz(*np.array(f.find("orientation").text.split(), float)),
                                          np.array(f.find("translation").text.split(), float) * 1000)
    return out


def _staple_bone(dataset, name):
    matio = pytest.importorskip("matio")
    d = matio.load_from_mat(str(STAPLE / "bone_datasets" / dataset / "tri" / f"{name}.mat"))
    t = list(d.values())[0].properties
    return Tri(np.asarray(t["ConnectivityList"], np.int64) - 1, t["Points"])


def _angles(V, R):
    return [np.degrees(np.arccos(np.clip(V[:, k] @ R[:, k], -1, 1))) for k in range(3)]


needs_staple = pytest.mark.skipif(STAPLE is None or not STAPLE.exists(), reason="STAPLE_REPO not set")


@needs_staple
@pytest.mark.parametrize("dataset", ["LHDL_CT", "TLEM2_CT", "ICL_MRI", "JIA_MRI"])
def test_giboc_femur_matches_staple(dataset):
    ref = _osim_frames(STAPLE / "tests" / "ref_models" / "paper" / f"automatic_{dataset}.osim")
    r = FM.GIBOC_femur(_staple_bone(dataset, "femur_r"), "r", "cylinder")
    for joint in ("knee_r", "hip_r"):
        R, t = ref[(joint, "femur_r")]
        assert max(_angles(r.JCS[joint]["V"], R)) < 0.5, joint
        assert np.linalg.norm(r.JCS[joint]["Origin"] - t) < 1.0, joint


@needs_staple
@pytest.mark.parametrize("dataset", ["JIA_MRI", "TLEM2_MRI"])
def test_left_and_right_femurs_match_staple(dataset):
    ref = _osim_frames(STAPLE / "tests" / "ref_models" / "bilateral" / f"{dataset}_bilateral.osim")
    for side in "lr":
        r = FM.GIBOC_femur(_staple_bone(dataset, f"femur_{side}"), side, "cylinder")
        for j in (f"knee_{side}", f"hip_{side}"):
            R, t = ref[(j, f"femur_{side}")]
            assert max(_angles(r.JCS[j]["V"], R)) < 0.5, j
            assert np.linalg.norm(r.JCS[j]["Origin"] - t) < 1.0, j


@needs_staple
@pytest.mark.parametrize("side", ["r", "l"])
def test_talus_matches_staple(side):
    from bvrviewer.core.staple.talus import STAPLE_talus
    ref = _osim_frames(STAPLE / "tests" / "ref_models" / "partial" / f"example_auto2020_ankle_{side.upper()}.osim")
    r = STAPLE_talus(_staple_bone("JIA_ANKLE_MRI", f"talus_{side}"), side)
    for j in (f"ankle_{side}", f"subtalar_{side}"):
        R, t = ref[(j, f"talus_{side}")]
        assert max(_angles(r.JCS[j]["V"], R)) < 0.01, j
        assert np.linalg.norm(r.JCS[j]["Origin"] - t) < 0.01, j


@needs_staple
def test_kai_tibia_matches_staple():
    ref = _osim_frames(STAPLE / "tests" / "ref_models" / "partial" / "example_auto2020_knee_R.osim")
    r = TB.Kai2014_tibia(_staple_bone("ICL_MRI", "tibia_r"), "r")
    assert max(_angles(r.JCS["knee_r"]["V"], ref[("knee_r", "tibia_r")][0])) < 0.05


@needs_staple
@pytest.mark.parametrize("side", ["r", "l"])
def test_medial_and_lateral_on_both_sides(side):
    from bvrviewer.core.staple.tri import TriMesh2DProperties
    T = _staple_bone("JIA_MRI", f"tibia_{side}")
    g = TB.GIBOC_tibia(T, side, "centroids")
    lat = TriMesh2DProperties(g.art_surf["plateau_lat"])["Center"]
    med = TriMesh2DProperties(g.art_surf["plateau_med"])["Center"]
    assert (lat - med) @ g.frame("knee")[:3, 2] * (1 if side == "r" else -1) > 30     # Z points right


# ---------------------------------------------------------------- partial bones (BVR sample)

@pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="BVR_SAMPLE_DATA not set")
def test_partial_knee_sample():
    K = SAMPLE / "Knee_SampleData" / "Models"
    F = tri_from_pyvista(pv.read(K / "AUT_Left_Femur.stl"))
    T = tri_from_pyvista(pv.read(K / "AUT_Left_Tibia.stl"))
    f = FM.GIBOC_femur(F, "l", "cylinder")
    assert not f.whole and "hip_l" not in f.JCS
    t = TB.GIBOC_tibia(T, "l", "ellipse")
    k = TB.Kai2014_tibia(T, "l")
    for r in (f, t, k):
        assert np.isclose(np.linalg.det(r.frame("knee")[:3, :3]), 1)
    # femur and tibia axes roughly agree in the scan pose (knee near full extension)
    from bvrviewer.core.transforms import model_to_aut_from_tfm
    tf = SAMPLE / "Knee_SampleData" / "Transforms"
    A = np.linalg.inv(model_to_aut_from_tfm(tf / "Left_Femur-DICOM2AUT.tfm")) @ f.frame("knee")
    B = np.linalg.inv(model_to_aut_from_tfm(tf / "Left_Tibia-DICOM2AUT.tfm")) @ t.frame("knee")
    assert A[:3, 2] @ B[:3, 2] > np.cos(np.radians(15)) and A[:3, 1] @ B[:3, 1] > np.cos(np.radians(25))
    assert np.degrees(np.arccos(np.clip(t.frame("knee")[:3, 2] @ k.frame("knee")[:3, 2], -1, 1))) < 10
    with pytest.raises(Exception):
        FM.Kai2014_femur(F, "l")                         # needs the femoral head
