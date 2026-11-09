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
win.resize(1600, 960)
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
p = tab.project
setup = sys.modules["bvrviewer.app.assessor_setup"]

# 1. the reference: add it, add its bones one by one with their tracking files
ref = tab.add_reference(ask=False)
assert ref is p.reference and ref.criterion
for b in BONES:
    tab.add_bone(ref, b, path=sim / "Reference" / f"{TRIAL}_{b}.tra")
assert ref.bones == list(BONES) and len(p.entries_of(ref)) == 3
assert asked == [f"{TRIAL}_tib.tra"], asked          # asked once, then remembered for the tab
assert p.trials() == ["Trial 1"]

# 2. the first assessor starts with the reference's bones; its files are matched to bones by name
expert = tab.add_assessor("Expert", "1", copy_from=ref, ask=False)
assert expert.bones == list(BONES) and not p.entries_of(expert)
item = tab.tree.currentItem()
assert setup.SetupTree.data(item)[1] is expert and item.text(0) == "Expert"
assert [item.child(i).text(0) for i in range(item.childCount())] == list(BONES)
assert item.child(0).child(0).text(1).startswith("none")          # tracking not loaded yet: grey row
assert item.child(0).text(1).startswith("none")                  # shown on the bone row too (one trial)
menu = [a.text() for a in tab.build_menu(setup.SetupTree.data(item)).actions() if a.text()]
assert menu[:2] == ["Add bone…", "Load tracking files…"] and "Add another session" in menu, menu
files = [sim / "Expert" / f"{TRIAL}_s1_{b}.tra" for b in BONES]
md = setup.MatchDialog(tab, p, expert, files)
assert [r[1] for r in md.values()] == list(BONES) and {r[2] for r in md.values()} == {"Trial 1"}
setup.MatchDialog.exec = lambda self: QDialog.Accepted
tab.load_tracking_files(expert, files)
assert all(p.entry(expert, b, "Trial 1") is not None for b in BONES)

# 3. more assessors and sessions copy the bones of an earlier one
expert2 = tab.add_session(expert)
assert (expert2.name, expert2.session, expert2.bones) == ("Expert", "2", list(BONES))
assert p.display_name(expert) == "Expert · session 1" and p.display_name(expert2) == "Expert · session 2"
tab.load_tracking_files(expert2, [sim / "Expert" / f"{TRIAL}_s2_{b}.tra" for b in BONES])
for rater in ("NoviceA", "NoviceB"):
    a1 = tab.add_assessor(rater, "1", copy_from=expert, ask=False)
    tab.load_tracking_files(a1, [sim / rater / f"{TRIAL}_s1_{b}.tra" for b in BONES])
    a2 = tab.add_session(a1)
    tab.load_tracking_files(a2, [sim / rater / f"{TRIAL}_s2_{b}.tra" for b in BONES])
assert p.raters() == ["Expert", "NoviceA", "NoviceB"] and p.sessions() == ["1", "2"]
assert len(p.entries) == 3 + 3 * 2 * 3 and not p.problems(), p.problems()

# a bone under another label isn't compared until it matches the others
nb = p.find_assessor("NoviceB", "1")
p.relabel_bone(nb, "cal", "calcaneus")
tab.refresh_tables()
assert any("Only NoviceB · session 1 has a bone 'calcaneus'" in s for s in p.problems()), p.problems()
bd = setup.BoneDialog(tab, p, nb, current="calcaneus", title="Change bone label", ask_file=False)
assert bd.combo.findData("cal") >= 0                  # the label the others use is offered
p.relabel_bone(nb, "calcaneus", "cal")
tab.refresh_tables()
assert not p.problems()
tab.recompute()
pump()

# validity: the errors come out in the order and size simulated
st = p.settings
assert st.mode == "validity" and st.measures == ("helical_rotation", "helical_translation")
rows = {(r.bone, r.label, r.measure): r.stats for r in tab.rows}
for b in BONES:
    m = [rows[(b, f"{r} s1", "helical_rotation")]["median"] for r in RATERS]
    assert m[0] < m[1] < m[2], (b, m)
    assert 0.2 < m[0] < 0.7 and 0.7 < m[1] < 2.0, (b, m)
assert rows[("cal", "NoviceA s1", "helical_rotation")]["n"] == 17      # two untracked frames dropped
assert rows[("tib", "NoviceB s1", "helical_rotation")]["outliers"] >= 1 or \
    rows[("tib", "NoviceB s1", "helical_rotation")]["over"] >= 3
shot("a01_files_and_summary")

# surfaces from the .cfg (TIF partial volumes) add the surface distance
tab.add_cfg(str(ANK / "2025-01-12-Ankle.cfg"))
pump()
assert all(b.surface is not None for b in p.bones) and len(p.bones) == 3
assert tab.measure_checks["surface_mean"].isEnabled()
tab.measure_checks["surface_mean"].setChecked(True)
tab.recompute()
rows = {(r.bone, r.label, r.measure): r.stats for r in tab.rows}
sd = rows[("tib", "Expert s1", "surface_mean")]["median"]
assert 0.1 < sd < 2.0, sd
tab.tabs.setCurrentWidget(tab.dist)
pump()
shot("a02_distributions_validity")
tab.dist.fig.savefig(OUT / "a02_distributions_validity_figure.png", dpi=150)
tab.cap.setChecked(True)
pump()
tab.dist.fig.savefig(OUT / "a03_distributions_capped_figure.png", dpi=150)
tab.cap.setChecked(False)
tab.tabs.setCurrentWidget(tab.frames_fig)
pump()
shot("a04_over_frames")
tab.frames_fig.fig.savefig(OUT / "a04_over_frames_figure.png", dpi=150)

# inter-rater (session 1 of each rater): 3 pairs per bone
tab.mode.setCurrentIndex(tab.mode.findData("inter"))
tab.recompute()
labels = sorted({c.label for c in tab.comps})
assert labels == ["Expert vs NoviceA", "Expert vs NoviceB", "NoviceA vs NoviceB"], labels
tab.session.setCurrentIndex(tab.session.findData("2"))
tab.recompute()
assert p.settings.inter_session == "2" and len(tab.comps) == 9
tab.tabs.setCurrentWidget(tab.dist)
pump()
tab.dist.fig.savefig(OUT / "a05_distributions_inter_figure.png", dpi=150)

# intra-rater: session 2 vs session 1 for each rater
tab.mode.setCurrentIndex(tab.mode.findData("intra"))
tab.recompute()
assert sorted({c.label for c in tab.comps}) == ["Expert s2 vs Expert s1", "NoviceA s2 vs NoviceA s1",
                                                 "NoviceB s2 vs NoviceB s1"]
tab.tabs.setCurrentWidget(tab.summary)
shot("a06_summary_intra")

# pooled statistics
tab.mode.setCurrentIndex(tab.mode.findData("validity"))
tab.pool.setCurrentIndex(tab.pool.findData("all"))
tab.recompute()
assert {r.bone for r in tab.rows} == {"All bones"} and len({r.label for r in tab.rows}) == 6
tab.tabs.setCurrentWidget(tab.dist)
pump()
tab.dist.fig.savefig(OUT / "a07_distributions_pooled_figure.png", dpi=150)
tab.pool.setCurrentIndex(tab.pool.findData("none"))
tab.recompute()

# a criterion in the wrong space is flagged
far = [e for e in p.entries if e.criterion and e.bone == "tal"][0]
far.tracking.poses[:, :3, 3] += 200.0
tab.recompute()
assert tab.space_warn.isVisibleTo(tab) and "same bone space" in tab.space_warn.text()
far.load()
tab.recompute()
assert not tab.space_warn.isVisibleTo(tab)

# exports and the project file
tab.export_summary(str(OUT / "summary.csv"))
tab.export_frames(str(OUT / "frames.csv"))
tab.export_figures(str(OUT / "fig.png"))
assert (OUT / "fig_distributions.png").exists() and (OUT / "fig_frames_helical_rotation.png").exists()
head = (OUT / "summary.csv").read_text().splitlines()
assert head[0].startswith("trial,bone,comparison,measure,unit,n,mean") and len(head) == 1 + len(tab.rows)
frames = (OUT / "frames.csv").read_text().splitlines()
assert frames[1].startswith("Trial 1,tib,validity,Expert s1,Criterion,1,helical_rotation"), frames[1]
tab.save_project(str(OUT / "project.bvrassess.json"))
before = [(r.label, r.measure, round(r.stats["mean"], 9)) for r in tab.rows]
next(a for a in new_tab.actions() if a.text() == "Assessor Evaluation").trigger()
pump()
tab2 = win.current()
tab2.open_project(str(OUT / "project.bvrassess.json"))
tab2.recompute()
after = [(r.label, r.measure, round(r.stats["mean"], 9)) for r in tab2.rows]
assert before == after
assert win.tabs.tabText(2) == tab.state.name

# the folder import makes the same assessors, bones and tracking from file names
next(a for a in new_tab.actions() if a.text() == "Assessor Evaluation").trigger()
pump()
tab3 = win.current()
tab3._format = dict(p.entries[0].overrides)
tab3.import_folder(str(sim), "{rater}/*_s{session}_{bone}")
q = tab3.project
assert [q.display_name(a) for a in q.assessors] == ["Reference"] + [f"{r} · session {s}" for r in RATERS
                                                                    for s in ("1", "2")], \
    [q.display_name(a) for a in q.assessors]
assert all(a.bones == list(BONES) or sorted(a.bones) == sorted(BONES) for a in q.assessors)
assert len(q.entries) == len(p.entries) and not q.problems(), q.problems()
tab3.recompute()
assert sorted((r.bone, r.label, round(r.stats["mean"], 9)) for r in tab3.rows if r.measure == "helical_rotation") \
    == sorted((r.bone, r.label, round(r.stats["mean"], 9)) for r in tab.rows if r.measure == "helical_rotation")
print("ASSESSOR OK", flush=True)
assert asmod  # module imported for the record
