"""Assessor evaluation core: pose differences, measures, statistics, comparisons and files."""

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
    rng = np.random.default_rng(3)
    n = 20
    truth = np.array([_T(rng.normal(size=3) * 20, rng.normal(size=3) * 50) for _ in range(n)])
    p = A.AssessmentProject()
    crit = A.TrackingEntry(path=__import__("pathlib").Path("crit.tra"), bone="tib", criterion=True)
    crit.tracking = _track(truth)
    p.entries.append(crit)
    for rater, err in [("A", 0.5), ("B", 2.0)]:
        for session in ("1", "2"):
            poses = np.array([t @ _T((err, 0, 0)) for t in truth])
            poses[0] = np.nan                                         # untracked frame
            e = A.TrackingEntry(path=__import__("pathlib").Path(f"{rater}{session}.tra"), bone="tib",
                                rater=rater, session=session)
            e.tracking = _track(poses)
            p.entries.append(e)
    return p


def test_validity_inter_and_intra_comparisons():
    p = _project()
    comps = p.comparisons(A.AnalysisSettings(mode="validity"))
    assert [c.label for c in comps] == ["A s1", "A s2", "B s1", "B s2"]
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


def test_pooling_and_csv(tmp_path):
    p = _project()
    for e in p.entries:                                             # a second trial, same data
        if e.trial == "Trial 1":
            f = A.TrackingEntry(path=e.path, trial="Trial 2", bone=e.bone, rater=e.rater, session=e.session,
                                criterion=e.criterion)
            f.tracking = e.tracking
            p.entries.append(f)
    st = A.AnalysisSettings(mode="validity", measures=("helical_rotation",), pool="trials")
    comps = p.comparisons(st)
    rows = A.summarise(comps, st)
    assert len(rows) == 4 and rows[0].trial == "All trials" and rows[0].stats["n"] == 38
    A.write_frames_csv(tmp_path / "f.csv", comps, st.measures)
    A.write_summary_csv(tmp_path / "s.csv", rows)
    lines = (tmp_path / "f.csv").read_text().splitlines()
    assert lines[0].startswith("trial,bone,comparison") and len(lines) == 1 + 8 * 19
    assert (tmp_path / "s.csv").read_text().splitlines()[1].startswith("All trials,tib,A s1,helical_rotation,deg,38")


def test_duplicates_and_mismatched_lengths_are_reported():
    p = _project()
    p.entries[1].session = "2"
    p.entries[3].tracking = _track(p.entries[3].tracking.poses[:15])
    probs = p.problems()
    assert any("both A, session 2" in s for s in probs)
    assert any("different numbers of rows" in s for s in probs)


def test_file_patterns(tmp_path):
    root = tmp_path
    f = root / "RaterA" / "T0019_jog0001_s2_tib.tra"
    fields = A.guess_fields(f, root, ["tib", "tal", "cal"], "{rater}/{trial}_*_s{session}_{bone}")
    assert fields == {"rater": "RaterA", "trial": "T0019", "session": "2", "bone": "tib", "criterion": False}
    g = A.guess_fields(root / "Reference" / "T0019_jog0001_cal.tra", root, ["tib", "tal", "cal"])
    assert g["bone"] == "cal" and g["criterion"]
    h = A.guess_fields(root / "Jo" / "jog_tal.tra", root, ["tib", "tal", "cal"])
    assert h["bone"] == "tal" and h["rater"] == "Jo" and not h["criterion"]
    with pytest.raises(ValueError):
        A.pattern_regex("{nope}")


def test_project_round_trip(tmp_path):
    tra = tmp_path / "a.tra"
    tra.write_text("\n".join(",".join(f"{v:.6f}" for v in np.eye(4).reshape(-1)) for _ in range(3)))
    p = A.AssessmentProject(pattern="{rater}/{bone}")
    p.entries.append(A.TrackingEntry(path=tra, bone="tib", rater="A", session="2", offset=1))
    p.settings.mode = "inter"
    p.save(tmp_path / "x.bvrassess.json")
    q = A.AssessmentProject.load(tmp_path / "x.bvrassess.json")
    assert q.settings.mode == "inter" and q.pattern == "{rater}/{bone}"
    e = q.entries[0]
    assert (e.rater, e.session, e.offset) == ("A", "2", 1) and e.tracking.n_frames == 4
