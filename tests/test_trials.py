"""Trials in a Visualiser tab (Visualiser #16) and setting them up from files (Visualiser #17)."""

import json
import os
from pathlib import Path

import pytest

from bvrviewer.core import trial as TR
from bvrviewer.core.models import BoneModel
from bvrviewer.core.session import (
    KinItem,
    TabState,
    Trial,
    XrayCamera,
    load_session,
    save_session,
)

SAMPLE = Path(os.environ.get("BVR_SAMPLE_DATA", "/nonexistent"))
sample = pytest.mark.skipif(not SAMPLE.exists(), reason="BVR_SAMPLE_DATA not set")


# ---------------------------------------------------------------- tab state
def test_empty_tab_has_no_trial_until_something_is_added():
    st = TabState()
    assert st.trials == [] and st.trial is None and st.models == [] and st.cameras == []
    st.models.append(BoneModel(name="tib", path=Path("tib.stl")))       # older code: adding makes "Trial 1"
    assert [t.name for t in st.trials] == ["Trial 1"] and [m.name for m in st.models] == ["tib"]


def test_lists_follow_the_trial_shown():
    st = TabState()
    a = st.add_trial("Jog")
    b = st.add_trial("Jog")
    assert b.name == "Jog (2)" and st.trial is a
    a.models.append(BoneModel(name="tib", path=Path("tib.stl")))
    st.active = 1
    assert st.models == [] and st.trial is b
    st.joints = [KinItem(model_path=Path("m.json"), kind="joint", name="Knee")]
    assert b.joints and not a.joints
    assert st.trial_of(a.models[0]) is a and st.trial_of(b) is b


def test_session_round_trip_with_trials(tmp_path):
    st = TabState(name="Session")
    a = st.add_trial("T0019 jog")
    a.models.append(BoneModel(name="tib", path=tmp_path / "tib.stl", tracking_path=tmp_path / "tib.tra"))
    a.cameras.append(XrayCamera(calibration_path=tmp_path / "c1.txt", source_kind="folder", source=[tmp_path],
                                name="X-ray 1"))
    a.cfg_path, a.tracking_dir = tmp_path / "trial.cfg", tmp_path / "Tracking"
    b = st.add_trial("Walk")
    b.models.append(BoneModel(name="tal", path=tmp_path / "tal.stl"))
    st.active = 1
    st.settings.fps, st.settings.hold_still = 250.0, "tal"
    p = tmp_path / "s.json"
    save_session(st, p)
    d = json.loads(p.read_text())
    assert d["version"] == 2 and [t["name"] for t in d["trials"]] == ["T0019 jog", "Walk"] and d["active"] == 1
    st2 = load_session(p)
    assert [t.name for t in st2.trials] == ["T0019 jog", "Walk"] and st2.trial.name == "Walk"
    assert st2.settings.fps == 250.0 and st2.settings.hold_still == "tal"        # the shown trial's own settings
    assert st2.trials[0].cfg_path == tmp_path / "trial.cfg" and st2.trials[0].cameras[0].name == "X-ray 1"


def test_version_1_session_opens_as_one_trial(tmp_path):
    d = {"version": 1, "tool": "visualiser", "name": "Old session",
         "models": [BoneModel(name="tib", path=tmp_path / "tib.stl").to_dict()], "cameras": [], "joints": [],
         "settings": {"fps": 120.0}}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(d))
    st = load_session(p)
    assert [t.name for t in st.trials] == ["Old session"] and st.trial.fps == 120.0
    assert [m.name for m in st.models] == ["tib"]


def test_saved_radiograph_names_include_the_trial_when_there_are_several(tmp_path):
    from bvrviewer.core.session import _path_fields

    st = TabState()
    for n in ("A", "B"):
        t = st.add_trial(n)
        t.cameras.append(XrayCamera(calibration_path=tmp_path / f"{n}.txt", source_kind="folder",
                                    source=[tmp_path / n], name="X-ray 1"))
    kinds = [k for _h, _k, k in _path_fields(st.to_dict())]
    assert kinds.count("calibration") == 2 and kinds.count("radiographs") == 2


# ---------------------------------------------------------------- matching tracking files
def test_tracking_files_matched_by_the_part_that_differs():
    files = [Path(f"T0019_jog0001_{b}.tra") for b in ("cal", "tal", "tib")]
    assert TR.tracking_keys(files)[files[2]] == "tib"
    assert TR.match_tracking(["tib", "tal", "cal"], files) == {"tib": files[2], "tal": files[1], "cal": files[0]}
    knee = [Path("Left_Femur.tra"), Path("Left_Tibia.tra")]
    assert TR.match_tracking(["Left_Tibia", "Left_Femur"], knee) == {"Left_Tibia": knee[1], "Left_Femur": knee[0]}
    assert TR.match_tracking(["Femur", "Patella"], knee) == {"Femur": knee[0]}            # no file for the patella
    assert TR.match_tracking(["mc3", "rad"], [Path("flx_ext_rad.tra")]) == {"rad": Path("flx_ext_rad.tra")}


def test_trial_name_from_tracking_files():
    assert TR.trial_name_from_files([Path("T0019_jog0001_cal.tra"), Path("T0019_jog0001_tib.tra")]) == "T0019_jog0001"
    assert TR.trial_name_from_files([Path("Left_Femur.tra"), Path("Left_Tibia.tra")]) is None    # only a side
    assert TR.trial_name_from_files([Path("tib.tra")]) is None


def test_model_segments_matched_to_bones():
    m = TR.match_segments([("Femur", "AUT_Left_Femur"), ("Tibia", "AUT_Left_Tibia")], ["Left_Tibia", "Left_Femur"])
    assert m == {"Femur": "Left_Femur", "Tibia": "Left_Tibia"}


# ---------------------------------------------------------------- drafts
def _files(tmp_path):
    (tmp_path / "Tracking").mkdir()
    for b in ("tib", "tal"):
        (tmp_path / f"{b}.stl").write_text("solid x\nendsolid x\n")
        (tmp_path / "Tracking" / f"S01_walk_{b}.tra").write_text("1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1\n" * 5)
    return tmp_path


def test_draft_by_hand(tmp_path):
    root = _files(tmp_path)
    d = TR.TrialDraft(name="")
    d.bones = [TR.BoneDraft(name=b, path=root / f"{b}.stl") for b in ("tib", "tal")]
    TR.set_tracking_dir(d, root / "Tracking")
    assert d.name == "S01_walk" and [b.tracking_path.name for b in d.bones] == ["S01_walk_tib.tra", "S01_walk_tal.tra"]
    assert d.problems() == [] and d.unused_tracking() == []
    d.bones[1].include = False
    assert [f.name for f in d.unused_tracking()] == ["S01_walk_tal.tra"]
    d.bones.append(TR.BoneDraft(name="tib", path=root / "tib.stl"))
    d.bones.append(TR.BoneDraft(name="vol", path=root / "vol.tif"))
    probs = " ".join(d.problems())
    assert "Two bones are called “tib”" in probs and "vol.tif wasn't found" in probs and "voxel size" in probs
    t = TR.TrialDraft(name="X", bones=d.bones[:1]).to_trial()
    assert isinstance(t, Trial) and [m.name for m in t.models] == ["tib"] and t.models[0].tracking_path is not None


def test_draft_from_a_trial_keeps_settings_and_shares_surfaces(tmp_path):
    m = BoneModel(name="tib", path=tmp_path / "tib.stl", color="#123456", opacity=0.5, tracking_path=tmp_path / "a.tra")
    m.mesh = object()                                   # stands in for a built surface
    m.axes_path, m.axes_segment = tmp_path / "m.bvrmodel.json", "Tibia"
    cam = XrayCamera(calibration_path=tmp_path / "c.txt", source_kind="folder", source=[tmp_path / "cam1"],
                     name="X-ray 1", show_plane=False)
    jl = KinItem(model_path=tmp_path / "m.bvrmodel.json", kind="joint", name="Ankle", model_map={"Tibia": "tib"})
    t = Trial(name="A", models=[m], cameras=[cam], joints=[jl])
    d = TR.draft_from_trial(t)
    d.joints = list(t.joints)
    assert d.bones[0].tracking_path == tmp_path / "a.tra" and d.cameras[0].folder == tmp_path / "cam1"
    t2 = d.to_trial()
    m2 = t2.models[0]
    assert m2 is not m and m2.mesh is m.mesh and m2.opacity == 0.5 and m2.axes_segment == "Tibia"
    assert t2.cameras[0].show_plane is False and t2.joints[0].name == "Ankle" and t2.joints[0] is not jl
    d.bones[0].voxel_size = (1, 1, 1)                   # a different surface: built again
    assert TR.bone_model(d.bones[0]).mesh is None


# ---------------------------------------------------------------- the samples' .cfg files
@sample
@pytest.mark.parametrize("cfg, folder, name, pairs", [
    ("2025-01-12-Ankle/2025-01-12-Ankle.cfg", "Tracking", "T0019_jog0001",
     [("cal", "T0019_jog0001_cal.tra"), ("tal", "T0019_jog0001_tal.tra"), ("tib", "T0019_jog0001_tib.tra")]),
    ("Knee_SampleData/Knee_Sample_Data.cfg", "Tracking", "Knee_Sample_Data",
     [("Left_Femur", "Left_Femur.tra"), ("Left_Tibia", "Left_Tibia.tra")]),
    ("2025-02-19-Wrist/Wrist_Sample_Data.cfg", "Tracking_SAM", "flx_ext",
     [("mc3", "flx_ext_mc3.tra"), ("rad", "flx_ext_rad.tra")]),
])
def test_drafts_from_the_samples(cfg, folder, name, pairs):
    path = SAMPLE / cfg
    assert TR.guess_tracking_dir(path).name == folder
    d = TR.draft_from_cfg(path, TR.guess_tracking_dir(path))
    assert d.name == name
    assert [(b.name, b.tracking_path.name) for b in d.bones] == pairs
    assert all(b.is_volume and b.voxel_size for b in d.bones) and len(d.cameras) == 2
    assert d.problems() == []


# ---------------------------------------------------------------- settings for each trial (Visualiser #23)
def test_each_trial_has_its_own_settings_but_the_tab_wide_ones_are_shared():
    from bvrviewer.core.session import TAB_WIDE

    st = TabState()
    a = st.add_trial("A")
    st.settings.xray_distance, st.settings.speed, st.settings.loop = 200, 0.5, False
    b = st.add_trial("B")                                  # starts like the trial shown
    assert b.settings is not a.settings and (b.settings.xray_distance, b.settings.speed) == (200, 0.5)
    st.active = 1
    st.settings.xray_distance = 50
    assert a.settings.xray_distance == 200
    st.set_tab_wide("background", "white")
    assert a.settings.background == b.settings.background == "white" == st._settings.background
    assert set(TAB_WIDE) == {"show_axes", "show_grid", "background", "curve_labels"}


def test_trial_settings_saved_with_each_trial(tmp_path):
    st = TabState()
    a = st.add_trial("A")
    a.settings.overlay_fill, a.settings.point_density, a.settings.camera_views = False, 0.3, True
    a.view = [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    b = st.add_trial("B")
    b.settings.xray_distance = 300
    st.set_tab_wide("curve_labels", "anatomical")
    p = tmp_path / "s.json"
    save_session(st, p)
    st2 = load_session(p)
    a2, b2 = st2.trials
    assert (a2.settings.overlay_fill, a2.settings.point_density, a2.settings.camera_views) == (False, 0.3, True)
    assert a2.view[0] == (1.0, 2.0, 3.0) and b2.settings.xray_distance == 300 and a2.settings.xray_distance == 120
    assert a2.settings.curve_labels == b2.settings.curve_labels == "anatomical"


def test_trials_saved_before_their_own_settings_use_the_tabs(tmp_path):
    d = {"version": 2, "tool": "visualiser", "name": "S", "active": 0, "settings": {"xray_distance": 250, "fps": 60},
         "trials": [{"name": "A", "models": [], "cameras": [], "joints": [], "fps": 500, "hold_still": "tib"}]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(d))
    t = load_session(p).trials[0]
    assert (t.settings.xray_distance, t.settings.fps, t.settings.hold_still) == (250, 500, "tib")


# ---------------------------------------------------------------- thinning the point cloud (Visualiser #24)
def test_even_order_spreads_any_share_of_the_points_evenly():
    import numpy as np
    from scipy.spatial import cKDTree

    from bvrviewer.app.camera_views import even_order

    rng = np.random.default_rng(3)
    pts = rng.normal(size=(3000, 3))
    pts = 20 * pts / np.linalg.norm(pts, axis=1, keepdims=True)          # points on a 20 mm sphere
    o = even_order(pts)
    assert sorted(o.tolist()) == list(range(len(pts)))
    for share in (0.05, 0.2):
        k = int(share * len(pts))
        gap_even = cKDTree(pts[o[:k]]).query(pts)[0].max()
        gap_random = cKDTree(pts[rng.permutation(len(pts))[:k]]).query(pts)[0].max()
        assert gap_even < 0.75 * gap_random, (share, gap_even, gap_random)
