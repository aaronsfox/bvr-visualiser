"""Model Builder steps in the model (Model Builder #1): bone types from file names, changing a bone's type,
axes used only once applied, three-point axes for any bone, and the joints that fit two bones."""

import numpy as np
import pytest
import pyvista as pv

from bvrviewer.core import joints as J


@pytest.mark.parametrize("name, typ", [
    ("SOL001A_tal_aligned.stl", "talus"), ("SOL001A_cal_aligned.stl", "calcaneus"),
    ("SOL001A_tib_aligned.stl", "tibia"), ("AUT_Left_Femur.stl", "femur"), ("tib_distal.stl", "tibia_distal"),
    ("P01_MT1.stl", "metatarsal1"), ("met5_R.stl", "metatarsal5"), ("med_cun.stl", "medial_cuneiform"),
    ("Cuboid_R.stl", "cuboid"), ("nav.stl", "navicular"), ("fibula.stl", "fibula"),
    ("mc3.stl", None), ("rad.stl", None), ("Patella.stl", None), ("total_knee.stl", None)])
def test_bone_type_from_file_name(name, typ):
    assert J.guess_segment_type(name) == typ


def test_side_from_file_name():
    from bvrviewer.app.model_builder import _guess_side

    assert _guess_side("AUT_Left_Femur.stl") == "left" and _guess_side("tal_R.stl") == "right"
    assert _guess_side("SOL001A_tal_aligned.stl") is None


class Box:
    """A mesh stand-in: a 40 x 20 x 10 box."""

    mesh = pv.Box(bounds=(0, 40, 0, 20, 0, 10)).triangulate().subdivide(2)

    def to_dict(self):
        return None


def model_with_box(require_apply=True, typ="generic"):
    m = J.KinematicModel(require_apply=require_apply)
    s = m.add_segment(typ, "Bone")
    s.surface = Box()
    return m, s


def place_three_points(s):
    s.landmarks = {"O": {"method": "point", "points": [[0, 0, 0]]}, "A": {"method": "point", "points": [[40, 0, 0]]},
                   "P": {"method": "point", "points": [[10, 20, 0]]}}


def test_three_point_axes_for_any_bone():
    m, s = model_with_box(require_apply=False)
    m.set_axes_source("Bone", "landmarks")
    place_three_points(s)
    m.compute()
    np.testing.assert_allclose(s.acs, np.eye(4), atol=1e-12)
    s.landmarks["P"]["points"] = [[20, 0, 0]]                       # in line: no plane
    m.compute()
    assert s.acs is None and "in line" in s.error


def test_nothing_until_applied_then_updates():
    m, s = model_with_box()
    m.set_axes_source("Bone", "landmarks")
    place_three_points(s)
    m.compute()
    assert s.draft is not None and s.acs is None and m.status("Bone") == "none"
    m.apply("Bone")
    assert m.status("Bone") == "applied" and np.allclose(s.acs, np.eye(4))
    s.landmarks["A"]["points"] = [[0, 20, 0]]                       # moved: follows straight away
    m.compute()
    assert np.allclose(s.acs[:3, 0], [0, 1, 0]) and m.status("Bone") == "applied"
    s.auto_update = False
    s.landmarks["A"]["points"] = [[40, 0, 0]]
    m.compute()
    assert np.allclose(s.acs[:3, 0], [0, 1, 0]) and m.status("Bone") == "changed"
    m.apply("Bone")
    assert np.allclose(s.acs[:3, 0], [1, 0, 0])
    s.landmarks.pop("A")
    m.compute()
    assert m.status("Bone") == "failed" and s.acs is not None       # the applied axes are kept


def test_automatic_methods_run_on_apply():
    m, s = model_with_box()
    assert s.axes_source == "inertial"
    m.compute()
    assert s.auto is None and s.acs is None
    m.apply("Bone")
    assert s.acs is not None and s.auto is not None


def test_apply_failure_keeps_things_as_they_were():
    m, s = model_with_box()
    m.set_axes_source("Bone", "landmarks")
    with pytest.raises(J.JointModelError, match="Missing"):
        m.apply("Bone")
    assert m.status("Bone") == "none" and "Missing" in s.error


def test_saved_files_record_what_was_applied(tmp_path):
    m, s = model_with_box()
    m.set_axes_source("Bone", "landmarks")
    place_three_points(s)
    m.compute()
    m.save(tmp_path / "a.json")
    back = J.KinematicModel.from_dict(m.to_dict())
    assert back.segments[0].acs is None and not back.segments[0].applied
    m.apply("Bone")
    s.auto_update = False
    back = J.KinematicModel.from_dict(m.to_dict())
    assert back.segments[0].applied and not back.segments[0].auto_update and back.segments[0].acs is not None
    old = m.to_dict()                                               # files from before Apply existed
    for sd in old["segments"]:
        sd.pop("applied")
        sd.pop("auto_update")
    back = J.KinematicModel.from_dict(old)
    assert back.segments[0].applied and back.segments[0].auto_update


def test_changing_a_bones_type():
    m = J.KinematicModel(require_apply=True)
    f = m.add_segment("femur", "Femur", "left")
    t = m.add_segment("tibia", "Tibia", "left")
    f.landmarks = {"ME": {"method": "point", "points": [[0, 0, 0]]}}
    m.add_joint("knee_grood_suntay", "Femur", "Tibia")
    m.set_segment_type("Femur", "generic")
    assert f.side == "none" and f.landmarks == {} and f.axes_source == "inertial" and m.joints == []
    m.set_segment_type("Tibia", "tibia_distal")
    assert t.side == "left" and t.axes_source == "aafact"


def test_joints_that_fit_two_bones():
    m = J.KinematicModel()
    for typ, n in (("tibia_distal", "Tibia"), ("talus", "Talus"), ("calcaneus", "Calcaneus"), ("generic", "X")):
        m.add_segment(typ, n)
    assert m.joint_types_for("Talus", "Tibia")[0] == ("ankle_isb", "Tibia", "Talus")
    assert m.joint_types_for("Calcaneus", "Tibia")[0][0] == "ankle_complex_isb"
    general = m.joint_types_for("X", "Talus")
    assert [k for k, *_ in general] == ["jcs_zxy", "cardan_xyz"] and general[0][1:] == ("X", "Talus")


# ---------- Model Builder #3, #4, #6 (11 Nov 2026)

def test_guess_bone_names_and_flags():
    from bvrviewer.app.model_builder import guess_bone

    g = guess_bone("AUT_Left_Tibia.stl")
    assert (g["type"], g["side"], g["name"], g["type_guessed"], g["side_guessed"]) == \
        ("tibia", "left", "Tibia", True, True)
    g = guess_bone("tib_distal.stl", fallback_side="left")
    assert (g["type"], g["name"], g["side"], g["side_guessed"]) == ("tibia_distal", "Tibia", "left", False)
    g = guess_bone("scan_0042.stl")
    assert (g["type"], g["name"], g["type_guessed"]) == ("generic", "scan_0042", False)


def test_bone_dropdown_lists_bone_names_only():
    from bvrviewer.app.model_builder import bone_items

    items = bone_items()
    assert [k for k, _ in items] == list(dict.fromkeys(k for k, _ in items))        # each bone once
    assert {k for k, _ in items} == set(J.SEGMENT_TYPES)
    titles = dict(items)
    assert titles["tibia"] == "Proximal tibia (knee)" and titles["tibia_distal"] == "Distal tibia (ankle)"
    assert not any("·" in t for t in titles.values())
    m = J.KinematicModel()
    assert m.add_segment("tibia_distal").name == "Tibia"                                # names stay short


@pytest.mark.parametrize("type_key", list(J.SEGMENT_TYPES))
def test_every_axes_method_has_instructions(type_key):
    from bvrviewer.core import method_guide as MG

    m = J.KinematicModel()
    s = m.add_segment(type_key)
    for key, _title in m.axes_sources(s.name):
        g = MG.guide(key, s.stype)
        assert g.needs and len(g.steps) >= 2, (type_key, key)
        assert any("Apply" in x for x in g.steps), (type_key, key)


def test_partial_tibia_caution_for_unstable_methods():
    from bvrviewer.core import method_guide as MG

    t = J.SEGMENT_TYPES["tibia"]
    for k in ("staple_tibia_ellipse", "staple_tibia_centroids"):
        assert any("Plateau or Kai" in c for c in MG.guide(k, t).cautions)
    for k in ("staple_tibia_plateau", "staple_tibia_kai"):
        assert not any("Plateau or Kai" in c for c in MG.guide(k, t).cautions)


def test_side_kept_when_a_bone_goes_through_general():
    m = J.KinematicModel()
    f = m.add_segment("femur", "Femur", "left")
    m.set_segment_type("Femur", "generic")
    assert f.side == "none"
    m.set_segment_type("Femur", "femur")
    assert f.side == "left"
    c = m.add_segment("generic", "Calc")                  # loaded as General, then said to be a calcaneus
    m.set_segment_type("Calc", "calcaneus")
    assert c.side == "left"                               # the other bones' side
