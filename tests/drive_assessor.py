"""Drive the Assessor Evaluation tab with simulated raters and save screenshots and exports.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a -s "-screen 0 1600x1000x24" python tests/drive_assessor.py OUTDIR

The ankle sample's tracking (the participant and jog trial of Welte et al. 2022) is the criterion. Three
simulated raters (two sessions each) are made by moving each bone, at every frame, by a smooth random
rotation about its centroid plus a translation, of known size:
  Expert ~0.4 deg / 0.2 mm, Novice A ~1.2 deg / 0.6 mm, Novice B ~2.5 deg / 1.2 mm with a few failed frames.
Files are laid out as Reference/<trial>_<bone>.tra and <Rater>/<trial>_s<session>_<bone>.tra.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

QApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
import pyvista as pv  # noqa: E402

from bvrviewer.app import assessor as asmod  # noqa: E402
from bvrviewer.app import dialogs  # noqa: E402
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core import assessment as A  # noqa: E402
from bvrviewer.core.tracking import load_tracking  # noqa: E402

S = Path(os.environ["BVR_SAMPLE_DATA"])
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ANK = S / "2025-01-12-Ankle"
TRIAL = "T0019_jog0001"
BONES = ("tib", "tal", "cal")
RATERS = {"Expert": (0.4, 0.2, 0), "NoviceA": (1.2, 0.6, 0), "NoviceB": (2.5, 1.2, 3)}


# ---------------------------------------------------------------- simulated raters
def write_tra(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for T in poses:
        rows.append(",".join("NaN" if not np.isfinite(v) else f"{v:.10f}" for v in T.reshape(-1)))
    path.write_text("\n".join(rows))


def smooth_noise(rng, n, size):
    """(n, 3) smooth random vectors whose norms average `size`."""
    x = gaussian_filter1d(rng.normal(size=(n, 3)), sigma=2.0, axis=0, mode="nearest")
    return x * size / np.mean(np.linalg.norm(x, axis=1))


sim = OUT / "sim"
centroids = {b: np.asarray(pv.read(str(ANK / "Meshes" / f"SOL001A_{b}_aligned.stl")).points).mean(axis=0)
             for b in BONES}
for b in BONES:
    ref = load_tracking(ANK / "Tracking" / f"{TRIAL}_{b}.tra")
    write_tra(sim / "Reference" / f"{TRIAL}_{b}.tra", ref.poses)
    n = ref.n_frames
    for k, (rater, (deg, mm, fails)) in enumerate(RATERS.items()):
        for session in (1, 2):
            rng = np.random.default_rng(100 * k + 10 * session + BONES.index(b))
            rv = smooth_noise(rng, n, np.radians(deg))
            tv = smooth_noise(rng, n, mm)
            if fails:
                rv[4:4 + fails] *= 5.0                    # a few badly tracked frames
            c = centroids[b]
            poses = ref.poses.copy()
            for f in range(n):
                D = np.eye(4)
                R = Rotation.from_rotvec(rv[f]).as_matrix()
                D[:3, :3] = R
                D[:3, 3] = c - R @ c + tv[f]             # rotation about the bone's centroid, then a shift
                poses[f] = ref.poses[f] @ D
            if rater == "NoviceA" and b == "cal":
                poses[-2:] = np.nan                       # didn't track the last frames
            write_tra(sim / rater / f"{TRIAL}_s{session}_{b}.tra", poses)
print("simulated raters written", flush=True)

# ---------------------------------------------------------------- the GUI
app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1600, 980)
win.show()


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def shot(name, widget=None):
    pump(0.6)
    (widget or win).grab().save(str(OUT / f"{name}.png"))
    print("saved", name, flush=True)


assert "assessor" in win.start.cards
file_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&File")
new_tab = next(a.menu() for a in file_menu.actions() if a.text() == "New tab")
next(a for a in new_tab.actions() if a.text() == "Assessor Evaluation").trigger()
pump()
tab = win.current()
assert tab.kind == "assessor" and win.tabs.tabText(1) == "Assessor 1"
shot("a00_empty")

# Format dialog: accept what was detected; remember it for the tab after the first file
asked = []


def accept(self):
    asked.append(self.path.name)
    self.remember.setChecked(True)
    return QDialog.Accepted


dialogs.FormatDialog.exec = accept
setup = sys.modules["bvrviewer.app.assessor_setup"]
p = tab.project

# 1. the factors: bones, assessors, sessions and the reference
for b in BONES:
    tab.add_bone(b)
for r in RATERS:
    tab.add_factor("assessor", r)
tab.add_factor("session", "2")
tab.reference.setChecked(True)
assert (p.bone_names(), p.raters(), p.sessions(), p.trials(), p.has_reference) == \
    (list(BONES), list(RATERS), ["1", "2"], ["Trial 1"], True)
g = tab.grid
assert (g.rowCount(), g.columnCount()) == (1 + 3 * 2, 3)
assert [g.verticalHeaderItem(r).text() for r in range(3)] == ["Reference", "Expert · session 1",
                                                               "Expert · session 2"]
assert g.item(0, 0).text() == "+" and tab.coverage.text() == "0 of 21 cells filled"
shot("a01_factors_empty_grid")

# 2. fill the reference cells one by one
for c, b in enumerate(BONES):
    assert tab.load_cell("Trial 1", b, A.CRITERION, "", sim / "Reference" / f"{TRIAL}_{b}.tra")
assert asked == [f"{TRIAL}_tib.tra"], asked          # asked once, then remembered for the tab
assert g.item(0, 0).text() == f"{TRIAL}_tib.tra"

# 3. fill the rest from files, matched to cells by the names in their paths; leave one gap
paths = sorted(str(x) for x in sim.rglob("*_s*_*.tra") if x.parent.name != "Reference")
gap = str(sim / "NoviceB" / f"{TRIAL}_s2_cal.tra")
paths.remove(gap)
fd = setup.FillDialog(tab, p, paths, sim)
got = {(Path(x).parent.name, Path(x).name): (t, b, w, s_) for x, t, b, w, s_ in fd.values()}
assert len(got) == 17 and got[("NoviceA", f"{TRIAL}_s2_tal.tra")] == ("Trial 1", "tal", "NoviceA", "2")
setup.FillDialog.exec = lambda self: QDialog.Accepted
assert tab.fill_from_files(paths) == 17
assert tab.coverage.text() == "20 of 21 cells filled" and not p.problems(), p.problems()
r_gap = g.slots.index(("NoviceB", "2"))
assert g.item(r_gap, 2).text() == "+" and g.item(r_gap, 0).text().endswith("_s2_tib.tra")
tab.recompute()
pump()

# validity: the errors come out in the order and size simulated; the gap is N/A
st = p.settings
assert st.mode == "validity" and st.measures == ("helical_rotation", "helical_translation")
assert len(tab.all_comps) == 18 and len(tab.comps) == 17
assert "17 of 18 comparisons available (1 N/A" in tab.headline.text(), tab.headline.text()
rows = {(r.bone, r.label, r.measure): r for r in tab.rows}
for b in BONES:
    m = [rows[(b, f"{r} s1", "helical_rotation")].stats["median"] for r in RATERS]
    assert m[0] < m[1] < m[2], (b, m)
    assert 0.2 < m[0] < 0.7 and 0.7 < m[1] < 2.0, (b, m)
na = rows[("cal", "NoviceB s2", "helical_rotation")]
assert na.na and na.note == "no file for NoviceB · session 2", na.note
assert rows[("cal", "NoviceA s1", "helical_rotation")].stats["n"] == 17      # two untracked frames dropped
r_na = next(i for i, r in enumerate(tab.rows) if r is na)
assert tab.summary.item(r_na, 5).text() == "N/A" and "no file" in tab.summary.item(r_na, tab.summary.columnCount() - 1).text()
shot("a02_grid_and_summary")

# surfaces from the .cfg (TIF partial volumes) add the surface distance
tab.add_cfg(str(ANK / "2025-01-12-Ankle.cfg"))
pump()
assert all(b.surface is not None for b in p.bones) and len(p.bones) == 3
assert tab.bones_table.item(0, 1).text() == "tib.tif"
assert tab.measure_checks["surface_mean"].isEnabled() and tab.measure_checks["surface_mean"].isChecked()
tab.recompute()
rows = {(r.bone, r.label, r.measure): r.stats for r in tab.rows}
sd = rows[("tib", "Expert s1", "surface_mean")]["median"]
assert 0.1 < sd < 2.0, sd
tab.tabs.setCurrentWidget(tab.dist)
pump()
shot("a03_distributions_validity")
tab.dist.fig.savefig(OUT / "a03_distributions_validity_figure.png", dpi=150)
tab.cap.setChecked(True)
pump()
tab.dist.fig.savefig(OUT / "a04_distributions_capped_figure.png", dpi=150)
tab.cap.setChecked(False)
tab.tabs.setCurrentWidget(tab.frames_fig)
pump()
tab.frames_fig.fig.savefig(OUT / "a05_over_frames_figure.png", dpi=150)

# inter-rater (one session): 3 pairs per bone; session 2 has the gap
tab.mode.setCurrentIndex(tab.mode.findData("inter"))
tab.recompute()
assert sorted({c.label for c in tab.all_comps}) == ["Expert vs NoviceA", "Expert vs NoviceB", "NoviceA vs NoviceB"]
assert len(tab.comps) == 9
tab.session.setCurrentIndex(tab.session.findData("2"))
tab.recompute()
assert p.settings.inter_session == "2" and len(tab.all_comps) == 9 and len(tab.comps) == 7
tab.tabs.setCurrentWidget(tab.dist)
pump()
tab.dist.fig.savefig(OUT / "a06_distributions_inter_s2_figure.png", dpi=150)

# intra-rater: session 2 vs session 1; NoviceB's calcaneus is N/A
tab.mode.setCurrentIndex(tab.mode.findData("intra"))
tab.recompute()
assert len(tab.all_comps) == 9 and len(tab.comps) == 8
assert [r.note for r in tab.rows if r.bone == "cal" and r.label == "NoviceB s2 vs NoviceB s1"][0] == \
    "no file for NoviceB · session 2"
tab.tabs.setCurrentWidget(tab.summary)
shot("a07_summary_intra")

# pooled statistics
tab.mode.setCurrentIndex(tab.mode.findData("validity"))
tab.pool.setCurrentIndex(tab.pool.findData("all"))
tab.recompute()
pooled = {r.label: r for r in tab.rows if r.measure == "helical_rotation"}
assert set(pooled) == {f"{r} s{s}" for r in RATERS for s in ("1", "2")}
assert pooled["NoviceB s2"].note == "1 of 3 comparisons missing" and not pooled["NoviceB s2"].na
tab.pool.setCurrentIndex(tab.pool.findData("none"))
tab.recompute()

# a reference in the wrong space is flagged
far = p.entry("Trial 1", "tal", A.CRITERION)
far.tracking.poses[:, :3, 3] += 200.0
tab.recompute()
assert tab.space_warn.isVisibleTo(tab) and "same bone space" in tab.space_warn.text()
far.load()
tab.recompute()
assert not tab.space_warn.isVisibleTo(tab)

# renaming and removing factors keeps the files in their cells
tab.rename_factor("assessor", "NoviceA", "Novice A")
assert p.entry("Trial 1", "tib", "Novice A", "1") is not None
tab.rename_factor("assessor", "Novice A", "NoviceA")

# exports and the project file
tab.export_summary(str(OUT / "summary.csv"))
tab.export_frames(str(OUT / "frames.csv"))
tab.export_figures(str(OUT / "fig.png"))
assert (OUT / "fig_distributions.png").exists() and (OUT / "fig_frames_helical_rotation.png").exists()
head = (OUT / "summary.csv").read_text().splitlines()
assert head[0].startswith("trial,bone,comparison,measure,unit,n,mean") and len(head) == 1 + len(tab.rows)
assert any(",NA," in line and "no file for NoviceB" in line for line in head)
frames = (OUT / "frames.csv").read_text().splitlines()
assert frames[1].startswith("Trial 1,tib,validity,Expert s1,Criterion,1,helical_rotation"), frames[1]
tab.save_project(str(OUT / "project.bvrassess.json"))
before = [(r.label, r.measure, r.note, round(r.stats.get("mean", -1), 9)) for r in tab.rows]
next(a for a in new_tab.actions() if a.text() == "Assessor Evaluation").trigger()
pump()
tab2 = win.current()
tab2.open_project(str(OUT / "project.bvrassess.json"))
tab2.recompute()
after = [(r.label, r.measure, r.note, round(r.stats.get("mean", -1), 9)) for r in tab2.rows]
assert before == after
assert tab2.coverage.text() == "20 of 21 cells filled"

# filling from a folder with a name pattern adds assessors and sessions it names
next(a for a in new_tab.actions() if a.text() == "Assessor Evaluation").trigger()
pump()
tab3 = win.current()
tab3._format = dict(p.entries[0].overrides)
for b in BONES:
    tab3.add_bone(b)
assert tab3.fill_from_folder(str(sim), "{rater}/*_s{session}_{bone}") == 21
q = tab3.project
assert (q.raters(), q.sessions(), q.has_reference) == (list(RATERS), ["1", "2"], True), (q.raters(), q.sessions())
assert tab3.coverage.text() == "21 of 21 cells filled"
print("ASSESSOR OK", flush=True)
assert asmod  # module imported for the record
