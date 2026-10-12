"""Tests for the loaders.

Synthetic tests always run. Tests marked `sample` need the SlicerAutoscoperM sample data
unzipped into one folder, pointed to by BVR_SAMPLE_DATA (see README).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bvrviewer.core import tracking as trk
from bvrviewer.core.calibration import CalibrationError, load_calibration
from bvrviewer.core.config import read_cfg
from bvrviewer.core.images import ImageSequence, adjust, find_images
from bvrviewer.core.models import BoneModel, ModelError, load_surface
from bvrviewer.core.session import TabState, XrayCamera, load_session, save_session
from bvrviewer.core.transforms import euler_zyx_to_matrix, matrix_to_xyzypr, model_to_aut_from_tfm

SAMPLE = Path(os.environ.get("BVR_SAMPLE_DATA", "")) if os.environ.get("BVR_SAMPLE_DATA") else None
sample = pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="BVR_SAMPLE_DATA not set")
WRIST = SAMPLE / "2025-02-19-Wrist" if SAMPLE else None
ANKLE = SAMPLE / "2025-01-12-Ankle" if SAMPLE else None
KNEE = SAMPLE / "Knee_SampleData" if SAMPLE else None


# ---------- synthetic tracking files ----------

def _poses(n=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = euler_zyx_to_matrix(*rng.uniform(-1.2, 1.2, 3))
        m[:3, 3] = rng.uniform(-100, 100, 3)
        out.append(m)
    return np.array(out)


def _write(tmp_path, name, rows, sep=","):
    p = tmp_path / name
    p.write_text("\n".join(sep.join(f"{v:.10f}" if np.isfinite(v) else "NaN" for v in r) for r in rows))
    return p


def test_row_and_column_order_agree(tmp_path):
    P = _poses()
    row = _write(tmp_path, "row.tra", [m.reshape(-1) for m in P])
    col = _write(tmp_path, "col.tra", [m.T.reshape(-1) for m in P], sep=" ")
    a, b = trk.load_tracking(row), trk.load_tracking(col)
    assert a.fmt.order == "row" and b.fmt.order == "column" and b.fmt.separator == " "
    np.testing.assert_allclose(a.poses, P, atol=1e-8)
    np.testing.assert_allclose(b.poses, P, atol=1e-8)


def test_nan_rows_and_gap_filling(tmp_path):
    P = _poses(6)
    rows = [m.reshape(-1) for m in P]
    rows[2] = rows[3] = np.full(16, np.nan)
    t = trk.load_tracking(_write(tmp_path, "gap.tra", rows))
    assert t.n_untracked == 2 and t.gaps() == [(2, 3)]
    assert t.pose(2, "hide") is None
    np.testing.assert_allclose(t.pose(3, "hold"), P[1])
    mid = t.pose(2, "interpolate")
    assert mid is not None and abs(np.linalg.det(mid[:3, :3]) - 1) < 1e-9
    np.testing.assert_allclose(mid[:3, 3], P[1, :3, 3] + (P[4, :3, 3] - P[1, :3, 3]) / 3, atol=1e-9)
    assert t.pose(99, "hide") is None


def test_xyzypr_degrees_radians_and_cm(tmp_path):
    P = _poses()
    deg = [matrix_to_xyzypr(m) for m in P]
    rad = [np.r_[d[:3] / 10, np.radians(d[3:])] for d in deg]
    a = trk.load_tracking(_write(tmp_path, "deg.tra", deg))
    b = trk.load_tracking(_write(tmp_path, "rad.tra", rad), overrides={"translation_units": "cm"})
    assert a.fmt.kind == "xyzypr" and a.fmt.rotation_units == "deg"
    assert b.fmt.rotation_units == "rad"
    np.testing.assert_allclose(a.poses, P, atol=1e-7)
    np.testing.assert_allclose(b.poses, P, atol=1e-7)


def test_multi_volume_file(tmp_path):
    A, B = _poses(4, 1), _poses(4, 2)
    p = _write(tmp_path, "all.tra", [np.r_[a.reshape(-1), b.reshape(-1)] for a, b in zip(A, B)])
    t = trk.load_tracking(p, volume_index=1)
    assert t.fmt.n_volumes == 2
    np.testing.assert_allclose(t.poses, B, atol=1e-8)
    with pytest.raises(trk.TrackingError):
        trk.load_tracking(p, volume_index=2)


def test_bad_tracking_files(tmp_path):
    with pytest.raises(trk.TrackingError):
        trk.load_tracking(_write(tmp_path, "w.tra", [np.arange(7.0)]))
    (tmp_path / "e.tra").write_text("")
    with pytest.raises(trk.TrackingError):
        trk.load_tracking(tmp_path / "e.tra")


def test_calibration_errors(tmp_path):
    (tmp_path / "v1.txt").write_text("1,2,3\n4,5,6\n7,8,9\n1,2,3\n1,2,3\n")
    with pytest.raises(CalibrationError, match="1.0"):
        load_calibration(tmp_path / "v1.txt")
    (tmp_path / "c.json").write_text("{}")
    with pytest.raises(CalibrationError, match="VTK"):
        load_calibration(tmp_path / "c.json")


def test_image_ordering_and_adjust(tmp_path):
    import tifffile

    for n in [10, 9, 100]:
        tifffile.imwrite(tmp_path / f"cam_{n}.tif", (np.arange(64, dtype=np.uint16).reshape(8, 8) * n))
    (tmp_path / "Thumbs.db").write_text("x")
    files, skipped = find_images(tmp_path)
    assert [f.name for f in files] == ["cam_9.tif", "cam_10.tif", "cam_100.tif"] and skipped == ["Thumbs.db"]
    seq = ImageSequence(files=files)
    assert seq.get(0).dtype == np.uint8 and seq.label(2) == "100"
    assert adjust(np.full((2, 2), 100, np.uint8), invert=True)[0, 0] == 155


# ---------- sample data ----------

@sample
def test_sample_tracking_files():
    w = trk.load_tracking(WRIST / "Tracking_SAM" / "flx_ext_rad.tra")
    # 61 rows; the last has no trailing newline, so `wc -l` reports 60.
    assert (w.n_frames, w.fmt.kind, w.fmt.order, w.n_untracked, w.warnings) == (61, "matrix", "row", 0, [])
    a = trk.load_tracking(ANKLE / "Tracking" / "T0019_jog0001_cal.tra")   # no trailing newline
    assert a.n_frames == 19
    k = trk.load_tracking(KNEE / "Tracking" / "Left_Femur.tra")
    assert k.n_frames == 1


@sample
def test_tfm_moves_ct_mesh_into_autoscoper_space():
    src = load_surface(WRIST / "Models" / "rad.stl")
    ref = load_surface(WRIST / "Models" / "AUT_Models" / "AUT_rad.stl")
    moved = src.transform(model_to_aut_from_tfm(WRIST / "Transforms" / "rad-DICOM2AUT.tfm"), inplace=False)
    np.testing.assert_allclose(np.array(moved.bounds), np.array(ref.bounds), atol=1e-3)


@sample
def test_degenerate_stl_is_rejected():
    with pytest.raises(ModelError, match="1 mm"):
        load_surface(WRIST / "Models" / "untracked" / "mc4.stl")


@sample
@pytest.mark.parametrize("tif,stl,voxel", [
    ("2025-02-19-Wrist/Volumes/rad.tif", "2025-02-19-Wrist/Models/AUT_Models/AUT_rad.stl", (0.391, 0.391, 0.625)),
    ("Knee_SampleData/Volumes/Left_Tibia.tif", "Knee_SampleData/Models/AUT_Left_Tibia.stl", (0.223, 0.223, 0.625)),
])
def test_tif_surface_matches_autoscoper_mesh(tif, stl, voxel):
    from scipy.spatial import cKDTree

    m = BoneModel(name="x", path=SAMPLE / tif, voxel_size=voxel)
    m.build()
    ref = load_surface(SAMPLE / stl)
    d, _ = cKDTree(ref.points).query(m.mesh.points[::10])
    assert np.median(d) < 0.5, np.median(d)
    b1, b2 = np.array(m.image_data.bounds), np.array(ref.bounds)
    assert np.abs(b1 - b2).max() < 2.0


@sample
def test_cfg_and_projection_onto_radiograph():
    cfg = read_cfg(WRIST / "Wrist_Sample_Data.cfg")
    assert len(cfg.calibrations) == 2 and all(p.exists() for p in cfg.calibrations)
    assert all(d.is_dir() for d in cfg.camera_dirs)
    assert cfg.voxel_size_for("rad.tif") == (0.391, 0.391, 0.625)

    cam = XrayCamera(calibration_path=cfg.calibrations[0], source_kind="folder", source=[cfg.camera_dirs[0]])
    cam.build()
    assert len(cam.images) == 61 and cam.images.label(0) == "0195"
    bone = load_surface(WRIST / "Models" / "AUT_Models" / "AUT_rad.stl")
    T = trk.load_tracking(WRIST / "Tracking_SAM" / "flx_ext_rad.tra").poses[0]
    pts = (np.c_[bone.points, np.ones(bone.n_points)] @ T.T)[:, :3]
    uv = cam.calibration.project(pts)
    img = cam.images.get(0)
    on_bone = img[uv[:, 1].astype(int), uv[:, 0].astype(int)].mean()
    assert on_bone < img.mean() - 25          # bone shadows are darker than the image overall


@sample
def test_session_round_trip(tmp_path):
    m = BoneModel(name="Radius", path=WRIST / "Volumes" / "rad.tif", voxel_size=(0.391, 0.391, 0.625),
                  tracking_path=WRIST / "Tracking_SAM" / "flx_ext_rad.tra", color="#123456")
    st = TabState(name="Wrist", models=[m])
    save_session(st, tmp_path / "s.json")
    back = load_session(tmp_path / "s.json")
    assert back.models[0].voxel_size == (0.391, 0.391, 0.625) and back.models[0].color == "#123456"
