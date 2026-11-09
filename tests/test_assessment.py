"""Assessor evaluation core: pose differences, measures, statistics, comparisons and files."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bvrviewer.core import assessment as A
from bvrviewer.core.tracking import Tracking, TrackingFormat


def _T(rotvec_deg=(0, 0, 0), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T[:3, 3] = t
    return T


def _track(poses):
    poses = np.array(poses, float)
    return Tracking(path=None, poses=poses, fmt=TrackingFormat(), n_rows=len(poses))


def test_pose_difference_is_in_the_bone_frame():
    ref = np.array([_T((10, 20, 30), (5, -3, 100))])
    d = _T((1, 0, 0), (0.5, 0, 0))                  # error in the bone's own frame
    test = ref @ d
    np.testing.assert_allclose(A.pose_difference(test, ref)[0], d, atol=1e-12)


def test_helical_angle_and_axial_translation():
    # 3 deg about z plus 2 mm along z: a screw motion
    phi, axis, along = A.helical(np.array([_T((0, 0, 3), (0, 0, 2))]))
    assert phi[0] == pytest.approx(3.0) and along[0] == pytest.approx(2.0)
    np.testing.assert_allclose(axis[0], [0, 0, 1], atol=1e-12)
    # pure translation: all of it is along the axis
    phi, _axis, along = A.helical(np.array([_T((0, 0, 0), (3, 4, 0))]))
    assert phi[0] == 0 and along[0] == pytest.approx(5.0)
    # identity
    phi, _a, along = A.helical(np.array([np.eye(4)]))
    assert phi[0] == 0 and along[0] == 0
    # 180 deg is handled
    phi, _a, _t = A.helical(np.array([_T((180, 0, 0))]))
    assert phi[0] == pytest.approx(180.0)


def test_helical_translation_misses_sideways_error_but_surface_distance_does_not():
    # 0.5 deg about z plus 1 mm along x (across the axis): ~0 mm along the helical axis
    dT = np.array([_T((0, 0, 0.5), (1, 0, 0))])
    _phi, _axis, along = A.helical(dT)
    assert abs(along[0]) < 1e-9
    pts = np.random.default_rng(1).normal(size=(500, 3)) * 10
    mean, mx = A.surface_distances(dT, pts)
    assert mean[0] > 0.8 and mx[0] >= mean[0]


def test_surface_distance_of_a_translation_is_its_length():
    pts = np.random.default_rng(2).normal(size=(50, 3)) * 20
    mean, mx = A.surface_distances(np.array([_T(t=(0, 0.6, 0.8))]), pts)
    assert mean[0] == pytest.approx(1.0) and mx[0] == pytest.approx(1.0)


def test_bone_axes_components():
    acs = _T((0, 0, 90), (10, 0, 0))                # axes rotated 90 deg about z, origin at x=10
    # 2 deg about the bone's own X axis, about its origin, plus 1 mm along its Y axis
    d_local = _T((2, 0, 0), (0, 1, 0))
    dT = acs @ d_local @ np.linalg.inv(acs)
    ang, tr = A.axes_components(np.array([dT]), acs)
    np.testing.assert_allclose(ang[0], [0, 2, 0], atol=1e-9)       # about Z, X, Y
    np.testing.assert_allclose(tr[0], [0, 1, 0], atol=1e-9)


def test_statistics():
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 10.0])
    s = A.statistics(x, threshold=2.0)
    assert s["n"] == 6 and s["mean"] == pytest.approx(x.mean()) and s["sd"] == pytest.approx(x.std(ddof=1))
    assert s["median"] == pytest.approx(1.75) and s["rmse"] == pytest.approx(np.sqrt(np.mean(x ** 2)))
    assert s["loa_high"] - s["loa_low"] == pytest.approx(2 * 1.96 * s["sd"])
    assert s["over"] == 2 and s["pct_over"] == pytest.approx(100 * 2 / 6)
    assert s["outliers"] == 1                                        # 10.0
    assert s["robust_low"] == 0.0                                    # floored: magnitudes
    assert s["cumulative"] == pytest.approx(x.sum())
    signed = A.statistics(np.array([-1.0, 1.0, -2.0, 2.0]), signed=True)
    assert signed["mean"] == 0 and signed["mae"] == 1.5 and signed["robust_low"] < 0
    assert A.statistics(np.array([np.nan]))["n"] == 0


def _project():
    """Reference + raters A (0.5 deg) and B (2 deg), sessions 1 and 2, bone tib; frame 1 untracked by raters."""
    rng = np.random.default_rng(3)
    n = 20
    truth = np.array([_T(rng.normal(size=3) * 20, rng.normal(size=3) * 50) for _ in range(n)])
    p = A.AssessmentProject()
    crit = A.TrackingEntry(path=Path("crit.tra"), bone="tib", criterion=True)
    crit.tracking = _track(truth)
    p.entries.append(crit)
    for rater, err in [("A", 0.5), ("B", 2.0)]:
        for session in ("1", "2"):
            poses = np.array([t @ _T((err, 0, 0)) for t in truth])
            poses[0] = np.nan                                         # untracked frame
            e = A.TrackingEntry(path=Path(f"{rater}{session}.tra"), bone="tib", rater=rater, session=session)
            e.tracking = _track(poses)
            p.entries.append(e)
    p.sync_factors()
    return p


def test_validity_inter_and_intra_comparisons():
    p = _project()
    assert (p.bone_names(), p.raters(), p.sessions(), p.trials(), p.has_reference) == \
        (["tib"], ["A", "B"], ["1", "2"], ["Trial 1"], True)
    comps = p.comparisons(A.AnalysisSettings(mode="validity"))
    assert [c.label for c in comps] == ["A s1", "A s2", "B s1", "B s2"] and all(c.ok for c in comps)
    assert all(len(c.frames) == 19 for c in comps)                   # frame 1 untracked by raters
    np.testing.assert_allclose(comps[0].values["helical_rotation"], 0.5, atol=1e-9)
    np.testing.assert_allclose(comps[2].values["helical_rotation"], 2.0, atol=1e-9)
    inter = p.comparisons(A.AnalysisSettings(mode="inter"))
    assert [c.label for c in inter] == ["A vs B"]
    np.testing.assert_allclose(inter[0].values["helical_rotation"], 1.5, atol=1e-9)
    intra = p.comparisons(A.AnalysisSettings(mode="intra"))
    assert [c.label for c in intra] == ["A s2 vs A s1", "B s2 vs B s1"]
    np.testing.assert_allclose(intra[0].values["helical_rotation"], 0.0, atol=1e-6)
    rows = A.summarise(comps, A.AnalysisSettings(measures=("helical_rotation",), rotation_threshold=1.0))
    assert len(rows) == 4 and rows[0].stats["pct_over"] == 0.0 and rows[2].stats["pct_over"] == 100.0
    assert not p.problems()


def test_gaps_give_na_with_reasons(tmp_path):
    p = _project()
    p.add_assessor("C")                                              # an assessor with no files at all
    p.add_bone("tal")                                                # a bone nobody has tracked
    # B didn't do session 2
    p.entries = [e for e in p.entries if not (e.rater == "B" and e.session == "2")]
    assert p.filled() == (4, 2 * (1 + 3 * 2))
    st = A.AnalysisSettings(mode="validity", measures=("helical_rotation",))
    comps = p.comparisons(st)
    assert len(comps) == 2 * 3 * 2                                  # bones x assessors x sessions
    by = {(c.bone, c.label): c for c in comps}
    assert by[("tib", "A s1")].ok and by[("tib", "B s1")].ok
    assert by[("tib", "B s2")].missing == "no file for B · session 2"
    assert by[("tal", "A s1")].missing == "no file for A · session 1; no file for Reference"
    rows = {(r.bone, r.label): r for r in A.summarise(comps, st)}
    assert rows[("tib", "B s2")].na and "B · session 2" in rows[("tib", "B s2")].note
    assert not rows[("tib", "A s1")].na and rows[("tib", "A s1")].note == ""
    intra = {c.label: c for c in p.comparisons(A.AnalysisSettings(mode="intra")) if c.bone == "tib"}
    assert intra["A s2 vs A s1"].ok and not intra["B s2 vs B s1"].ok and not intra["C s2 vs C s1"].ok
    inter = [c.label for c in p.comparisons(A.AnalysisSettings(mode="inter")) if c.bone == "tib"]
    assert inter == ["A vs B", "A vs C", "B vs C"]
    # pooled over bones: tib values count, tal's missing ones are noted
    pooled = A.summarise(comps, A.AnalysisSettings(mode="validity", measures=("helical_rotation",), pool="all"))
    a1 = next(r for r in pooled if r.label == "A s1")
    assert a1.stats["n"] == 19 and a1.note == "1 of 2 comparisons missing"
    A.write_summary_csv(tmp_path / "s.csv", A.summarise(comps, st))
    text = (tmp_path / "s.csv").read_text()
    assert "Trial 1,tib,B s2,helical_rotation,deg,0,NA" in text and "no file for B · session 2" in text
    # no reference: validity has nothing to compare
    p.set_reference(False)
    assert p.comparisons(st) == [] and not any(e.criterion for e in p.entries)


def test_pooling_and_csv(tmp_path):
    p = _project()
    for e in list(p.entries):                                       # a second trial, same data
        f = A.TrackingEntry(path=e.path, trial="Trial 2", bone=e.bone, rater=e.rater, session=e.session,
                            criterion=e.criterion)
        f.tracking = e.tracking
        p.entries.append(f)
    p.sync_factors()
    assert p.trials() == ["Trial 1", "Trial 2"]
    st = A.AnalysisSettings(mode="validity", measures=("helical_rotation",), pool="trials")
    comps = p.comparisons(st)
    rows = A.summarise(comps, st)
    assert len(rows) == 4 and rows[0].trial == "All trials" and rows[0].stats["n"] == 38
    A.write_frames_csv(tmp_path / "f.csv", comps, st.measures)
    A.write_summary_csv(tmp_path / "s.csv", rows)
    lines = (tmp_path / "f.csv").read_text().splitlines()
    assert lines[0].startswith("trial,bone,comparison") and len(lines) == 1 + 8 * 19
    assert (tmp_path / "s.csv").read_text().splitlines()[1].startswith("All trials,tib,A s1,helical_rotation,deg,38")


def test_mismatched_lengths_are_reported():
    p = _project()
    p.entries[3].tracking = _track(p.entries[3].tracking.poses[:15])
    assert any("different numbers of rows" in s for s in p.problems())


def test_file_patterns(tmp_path):
    root = tmp_path
    f = root / "RaterA" / "T0019_jog0001_s2_tib.tra"
    fields = A.guess_fields(f, root, ["tib", "tal", "cal"], "{rater}/{trial}_*_s{session}_{bone}")
    assert fields == {"rater": "RaterA", "trial": "T0019", "session": "2", "bone": "tib", "criterion": False}
    g = A.guess_fields(root / "Reference" / "T0019_jog0001_cal.tra", root, ["tib", "tal", "cal"])
    assert g["bone"] == "cal" and g["criterion"]
    with pytest.raises(ValueError):
        A.pattern_regex("{nope}")


def test_matching_files_to_slots(tmp_path):
    p = A.AssessmentProject()
    for b in ("tib", "tal", "cal"):
        p.add_bone(b)
    for a in ("Expert", "Novice A"):
        p.add_assessor(a)
    p.add_session()
    p.set_reference(True)
    m = A.match_file(tmp_path / "Expert" / "T0019_jog0001_s2_tal.tra", p)
    assert m == {"bone": "tal", "who": "Expert", "session": "2", "trial": "Trial 1"}
    m = A.match_file(tmp_path / "raters" / "novice_a_session1_cal.tra", p)
    assert m == {"bone": "cal", "who": "Novice A", "session": "1", "trial": "Trial 1"}
    m = A.match_file(tmp_path / "Reference" / "T0019_jog0001_tib.tra", p)
    assert m == {"bone": "tib", "who": A.CRITERION, "trial": "Trial 1"}
    m = A.match_file(tmp_path / "x" / "unknown.tra", p)
    assert "bone" not in m and "who" not in m and "session" not in m      # two sessions: can't guess


def test_factors_rename_remove_and_slots(tmp_path):
    rng = np.random.default_rng(5)
    truth = np.array([_T(rng.normal(size=3) * 20, rng.normal(size=3) * 50) for _ in range(10)])
    tra = tmp_path / "x.tra"
    tra.write_text("\n".join(",".join(f"{v:.10f}" for v in T.reshape(-1)) for T in truth))
    p = A.AssessmentProject()
    p.add_bone("tib")
    p.add_assessor("Jo")
    with pytest.raises(ValueError):
        p.add_assessor("Jo")
    assert p.rows() == [("Jo", "1")]
    p.set_reference(True)
    assert p.rows() == [(A.CRITERION, ""), ("Jo", "1")] and p.row_label(A.CRITERION, "") == "Reference"
    p.set_entry("Trial 1", "tib", A.CRITERION, "", tra)
    e = p.set_entry("Trial 1", "tib", "Jo", "1", tra)
    assert e.tracking.n_rows == 10 and p.filled() == (2, 2)
    p.set_entry("Trial 1", "tib", "Jo", "1", tra)                     # replacing keeps one entry per slot
    assert len(p.entries) == 2
    # a file for a new combination adds its factors
    p.set_entry("Trial 2", "tal", "Sam", "2", tra)
    assert (p.trials(), p.bone_names(), p.raters(), p.sessions()) == \
        (["Trial 1", "Trial 2"], ["tib", "tal"], ["Jo", "Sam"], ["1", "2"])
    # renames carry the files; the reference's file isn't touched by assessor/session renames
    p.rename("assessor", "Jo", "Joanne")
    p.rename("session", "1", "pre")
    p.rename("bone", "tib", "tibia")
    p.rename("trial", "Trial 1", "jog")
    assert p.entry("jog", "tibia", "Joanne", "pre") is not None and p.entry("jog", "tibia", A.CRITERION) is not None
    with pytest.raises(ValueError):
        p.rename("bone", "tibia", "tal")
    p.move("assessor", "Sam", -1)
    assert p.raters() == ["Sam", "Joanne"]
    # removing a factor removes the files in its slots
    p.remove("assessor", "Sam")
    assert all(e.rater != "Sam" for e in p.entries) and p.raters() == ["Joanne"]
    p.remove("session", "pre")
    assert p.sessions() == ["2"] and all(e.criterion for e in p.entries)
    # the project file keeps the factors, including combinations without files
    p.add_assessor("Kim")
    p.save(tmp_path / "p.bvrassess.json")
    q = A.AssessmentProject.load(tmp_path / "p.bvrassess.json")
    assert (q.bone_names(), q.raters(), q.sessions(), q.trials(), q.has_reference) == \
        (p.bone_names(), p.raters(), p.sessions(), p.trials(), p.has_reference)
    assert len(q.entries) == len(p.entries)


def test_older_project_files_get_factors(tmp_path):
    tra = tmp_path / "a.tra"
    tra.write_text("\n".join(",".join(f"{v:.6f}" for v in np.eye(4).reshape(-1)) for _ in range(3)))
    d = {"tool": "assessor", "version": 1, "entries": [
        {"path": str(tra), "bone": "tib", "criterion": True},
        {"path": str(tra), "bone": "tib", "rater": "A", "session": "1"},
        {"path": str(tra), "bone": "tal", "rater": "A", "session": "2", "trial": "jog"}]}
    (tmp_path / "v1.json").write_text(__import__("json").dumps(d))
    q = A.AssessmentProject.load(tmp_path / "v1.json")
    assert (q.bone_names(), q.raters(), q.sessions(), q.has_reference) == (["tib", "tal"], ["A"], ["1", "2"], True)
    assert q.trials() == ["Trial 1", "jog"]
    d2 = {"tool": "assessor", "version": 2, "assessors": [
        {"name": "Criterion", "criterion": True, "bones": ["tib"]},
        {"name": "Jo", "session": "1", "bones": ["tib", "cal"]}], "entries": []}
    (tmp_path / "v2.json").write_text(__import__("json").dumps(d2))
    q = A.AssessmentProject.load(tmp_path / "v2.json")
    assert (q.bone_names(), q.raters(), q.sessions(), q.has_reference) == (["tib", "cal"], ["Jo"], ["1"], True)


def test_project_round_trip(tmp_path):
    tra = tmp_path / "a.tra"
    tra.write_text("\n".join(",".join(f"{v:.6f}" for v in np.eye(4).reshape(-1)) for _ in range(3)))
    p = A.AssessmentProject(pattern="{rater}/{bone}")
    p.set_entry("Trial 1", "tib", "A", "2", tra, offset=1)
    p.settings.mode = "inter"
    p.save(tmp_path / "x.bvrassess.json")
    q = A.AssessmentProject.load(tmp_path / "x.bvrassess.json")
    assert q.settings.mode == "inter" and q.pattern == "{rater}/{bone}"
    e = q.entries[0]
    assert (e.rater, e.session, e.offset) == ("A", "2", 1) and e.tracking.n_frames == 4
