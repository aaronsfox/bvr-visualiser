"""Headless driver for trials in the Visualiser (Visualiser #16 and #17).

Usage: BVR_SAMPLE_DATA=... xvfb-run -a -s "-screen 0 1700x1000x24" python tests/drive_trials.py OUT_DIR

A new tab (only "Add trial…"); the Trial setup wizard on the ankle sample's .cfg and tracking folder (each
page saved as a screenshot, the camera preview, the review); a second trial duplicated from it; the knee
sample with a Model Builder model; switching trials; Configure… (rename a bone, change its colour and
tracking file, leave a camera out); saving and reopening the session; removing trials.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bvrviewer.app import trial_wizard as TW  # noqa: E402
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.navigator import ADD_TRIAL, TRIAL  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core import joints as J  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
S = Path(os.environ["BVR_SAMPLE_DATA"])
A, K = S / "2025-01-12-Ankle", S / "Knee_SampleData"

app = QApplication.instance() or QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(s=0.3):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout=180, what=""):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        pump(0.1)
    raise AssertionError(f"timed out: {what}")


def shot(w, name):
    pump(0.3)
    w.grab().save(str(OUT / f"{name}.png"))


# ---- #16: a new tab shows only "Add trial…"
tab = win.new_tab()
pump(0.5)
ERRORS = []


def _error(title, text):
    ERRORS.append(f"{title}: {text}")
    print("ERROR DIALOG:", title, "|", text, flush=True)


def no_dialogs(t):
    t._error = _error


no_dialogs(tab)
nav = tab.nav
assert tab.state.trials == [] and tab.stack.currentIndex() == 0
assert [nav.roots["trials"].child(i).data(0, Qt.UserRole) for i in range(nav.roots["trials"].childCount())] == [ADD_TRIAL]
assert nav.roots["scene"].isHidden()
acts = [a.text() for a in tab.build_menu(ADD_TRIAL, None).actions()]
assert acts == ["Import trial…", "Empty trial…"], acts
shot(win, "tr_00_new_tab")

# ---- #17: the wizard on the ankle sample's .cfg
dlg = TW.TrialSetupDialog(tab, mode="new", trials=tab.state.trials)
dlg.show()
dlg.set_cfg(A / "2025-01-12-Ankle.cfg")
d = dlg.draft
assert d.name == "T0019_jog0001" and d.tracking_dir == A / "Tracking", (d.name, d.tracking_dir)
assert [(b.name, b.tracking_path.name) for b in d.bones] == [("cal", "T0019_jog0001_cal.tra"),
                                                            ("tal", "T0019_jog0001_tal.tra"),
                                                            ("tib", "T0019_jog0001_tib.tra")]
assert all(b.voxel_size for b in d.bones) and len(d.cameras) == 2
assert not dlg.finish.isVisible() and dlg.next.isVisible()
shot(dlg, "tr_01_wizard_source")
dlg.steps.setCurrentRow(1)
pump(0.5)
checks = [r["check"].text() for r in dlg.bone_rows]
assert all(c.startswith("19 frames, all tracked") for c in checks), checks
assert "4×4 matrix" in dlg.fmt_label.text()
# the colour and tracking file can be changed per bone; leave the calcaneus out for now
dlg.bone_rows[2]["col"].colorChanged.emit("#E07A9A")
assert d.bones[2].color == "#E07A9A"
dlg.bone_rows[0]["inc"].setChecked(False)
shot(dlg, "tr_02_wizard_bones")
dlg.bone_rows[0]["inc"].setChecked(True)
dlg.steps.setCurrentRow(2)
pump(0.3)
assert all("images" in r["img"].text() for r in dlg.cam_rows), [r["img"].text() for r in dlg.cam_rows]
dlg.preview_camera(dlg.cam_rows[0])
assert dlg.preview.pixmap() is not None and "3 bones" in dlg.preview_note.text(), dlg.preview_note.text()
# #22: display settings for the trial
dlg.disp["xray_distance"].setValue(200)
dlg.disp["camera_views"].setChecked(True)
dlg.disp["overlay_fill"].setChecked(False)
assert not dlg.disp["overlay_opacity"].isEnabled() and dlg.disp["outline_width"].isEnabled()
dlg.disp["outline_width"].setValue(3)
shot(dlg, "tr_03_wizard_cameras")
dlg.steps.setCurrentRow(3)
pump(0.3)
assert dlg.finish.isVisible() and dlg.finish.isEnabled() and "Everything is ready" in dlg.issues.text()
shot(dlg, "tr_04_wizard_review")
# a name already used is a problem to fix
dlg.taken.add("T0019_jog0001")
dlg._update_review()
assert not dlg.finish.isEnabled() and "already a trial" in dlg.issues.text()
dlg.taken.clear()
dlg._update_review()
dlg._finish()
assert dlg.result() == QDialog.Accepted
t1 = tab.add_trial_from_draft(dlg.draft)
# #20: a "Loading" dialog with a progress bar while the bones are built
wait(lambda: tab._busy is not None and tab._busy.value() >= 0 and "Bone" in tab._busy.labelText(), timeout=30,
     what="loading dialog")
assert tab._busy.windowTitle() == "Loading T0019_jog0001" and tab._busy.maximum() == 4      # 3 bones + cameras
pump(0.2)
tab._busy.grab().save(str(OUT / "tr_04b_loading.png"))
wait(lambda: tab.state.trial is t1 and len(tab.state.models) == 3 and len(tab.scene._cam_actors) == 2,
     what="ankle trial")
pump(1)
s1 = t1.settings
assert (s1.xray_distance, s1.camera_views, s1.overlay_fill, s1.outline_width) == (200, True, False, 3)
assert tab.cam_views.isVisible()
v0 = tab.cam_views.views[0]
pump(0.5)
assert v0.outlines and all(it.isVisible() for it in v0.outlines.values())          # #21: outline only
assert not any(it.isVisible() for it in v0.items.values())
assert [m.name for m in t1.models] == ["cal", "tal", "tib"] and t1.models[2].color == "#E07A9A"
assert all(m.tracking is not None for m in t1.models) and tab.scene.n_frames == 19
assert all(c.show_source is False for c in t1.cameras)
tab.set_camera_views(True)
tab.timeline.slider.setValue(9)
shot(win, "tr_05_trial_shown")

# ---- Duplicate: same bones, new tracking (here the same folder again, as a stand-in for another trial)
dup = TW.TrialSetupDialog(tab, mode="new", trials=tab.state.trials, start_from=t1,
                          taken_names=[x.name for x in tab.state.trials])
dup.show()
assert dup.r_trial.isChecked() and dup.draft.name == "T0019_jog0001 (2)"
assert all(b.tracking_path is None for b in dup.draft.bones) and all(c.folder is None for c in dup.draft.cameras)
dup.set_tracking_dir(A / "Tracking")
assert dup.draft.name == "T0019_jog0001 (2)"          # the suggested name stays
assert all(b.tracking_path is not None for b in dup.draft.bones)
dup.steps.setCurrentRow(3)
pump(0.3)
assert not dup.finish.isEnabled() and "choose the radiograph folder" in dup.issues.text()
for row in dup.cam_rows:
    row["inc"].setChecked(False)                      # no radiographs for this one
dup._update_review()
assert dup.finish.isEnabled()
shot(dup, "tr_06_duplicate_review")
dup._finish()
t2 = tab.add_trial_from_draft(dup.draft)
wait(lambda: tab.state.trial is t2 and len(tab.state.models) == 3, what="duplicate")
assert all(a.mesh is b.mesh for a, b in zip(t1.models, t2.models))      # surfaces shared, not rebuilt
assert not t2.cameras and not tab.scene._cam_actors
assert t1.view is not None                                              # the viewpoint kept with the trial
# #23: each trial's own settings; background, axes, grid and curve labels are the tab's
assert t2.settings is not t1.settings and t2.settings.xray_distance == 200          # started like t1
tab._set("xray_distance", 50)
tab._set("trail", True)
tab.timeline.speed.setCurrentIndex(tab.timeline.speed.findData(0.25))
tab.timeline.loop.setChecked(False)
tab._set("background", "light")
assert (t1.settings.xray_distance, t1.settings.trail, t1.settings.speed, t1.settings.loop) == (200, False, 1.0, True)
assert (t2.settings.xray_distance, t2.settings.trail, t2.settings.speed, t2.settings.loop) == (50, True, 0.25, False)
assert t1.settings.background == "light" == t2.settings.background
# a bone's own settings are its trial's too
t2.models[0].color = "#FFFFFF"
assert t1.models[0].color != "#FFFFFF"

# #20: Cancel while a trial loads: nothing is added
n_before = len(tab.state.trials)
cancel_draft = TW.TR.draft_from_cfg(K / "Knee_Sample_Data.cfg", K / "Tracking")     # (surfaces not loaded yet)
cancel_draft.name = "Cancelled"
tab.add_trial_from_draft(cancel_draft)
wait(lambda: tab._busy is not None, timeout=30, what="loading dialog to cancel")
next(b for b in tab._busy.findChildren(QPushButton) if b.text() == "Cancel").click()
wait(lambda: tab._busy is None, timeout=120, what="cancelled")
pump(0.5)
assert len(tab.state.trials) == n_before and tab.state.trial is t2

# ---- the knee sample with a Model Builder model (axes from AUT STLs, bones from the .cfg's TIFs)
model_path = OUT / "knee.bvrmodel.json"
km = J.KinematicModel.preset("knee", "left")
for f in ("Femur", "Tibia"):
    bm = BoneModel(name=f, path=K / "Models" / f"AUT_Left_{f}.stl")
    bm.build()
    km.segment(f).surface = bm
km.compute()
have_model = all(sg.acs is not None for sg in km.segments)
km.save(model_path)
kd = TW.TrialSetupDialog(tab, mode="new", trials=tab.state.trials, taken_names=[x.name for x in tab.state.trials])
kd.show()
kd.set_cfg(K / "Knee_Sample_Data.cfg")
assert kd.draft.name == "Knee_Sample_Data"
kd.name.setText("Knee hop")
kd._name_typed("Knee hop")
kd.set_tracking_dir(K / "Tracking")
assert kd.draft.name == "Knee hop"                   # a typed name stays
if have_model:
    kd.draft.model_path = model_path
kd.read_later.setChecked(True)
assert kd.draft.read_radiographs is False
kd.steps.setCurrentRow(3)
pump(0.3)
shot(kd, "tr_07_knee_review")
kd._finish()
t3 = tab.add_trial_from_draft(kd.draft)
wait(lambda: tab.state.trial is t3 and len(tab.state.models) == 2, timeout=300, what="knee trial")
pump(1)
if have_model:
    wait(lambda: all(m.has_axes for m in t3.models) and t3.joints, what="knee model")
    print("knee axes:", [(m.name, m.axes_segment) for m in t3.models], "joints:", [j.label for j in t3.joints])
shot(win, "tr_08_knee_trial")

# ---- switching: double-click a trial, or select an item of a hidden trial
tab.show_trial(t1)
pump(0.5)
assert tab.state.trial is t1 and tab.scene.frame == 9 and len(tab.scene._cam_actors) == 2
assert np.allclose(np.asarray(tab.plotter.camera_position[0]), np.asarray(t1.view[0]))   # same viewpoint
assert tab.timeline.speed.currentData() == 1.0 and tab.timeline.loop.isChecked()        # t1's playback settings
nav.select("model", t3.models[0])
wait(lambda: tab.state.trial is t3 and tab.sel_obj is t3.models[0], what="switch by selecting a bone")
tab._nav_activated(TRIAL, t1)
pump(0.5)
assert tab.state.trial is t1
nav.select(TRIAL, t2)
pump(0.3)
shot(win, "tr_09_navigator")

# ---- Configure…: rename a bone, change its colour and tracking file, leave a camera out
cd = TW.TrialSetupDialog(tab, TW.TR.draft_from_trial(t1), mode="configure", trials=tab.state.trials,
                         taken_names=[x.name for x in tab.state.trials if x is not t1])
cd.show()
assert not cd.start_box.isVisible() and cd.finish.isVisible() and not cd.next.isVisible()
cd.steps.setCurrentRow(1)
b_tib = next(b for b in cd.draft.bones if b.name == "tib")
b_tib.name = "Tibia"
b_tib.color = "#5CC8A8"
b_cal = next(b for b in cd.draft.bones if b.name == "cal")
b_cal.tracking_path = A / "Tracking" / "T0019_jog0001_tal.tra"     # (a deliberate wrong pick, to see it reload)
cd.draft.cameras[1].include = False
cd._fill_bones()
cd._fill_cameras()
assert any("Same file as tal" in r["check"].text() for r in cd.bone_rows)
shot(cd, "tr_10_configure_bones")
cd._finish()
meshes = {m.name: m.mesh for m in t1.models}
tab.apply_trial_draft(t1, cd.draft)
wait(lambda: len(t1.cameras) == 1 and any(m.name == "Tibia" for m in t1.models), what="configure")
pump(0.5)
tib = next(m for m in t1.models if m.name == "Tibia")
assert tib.mesh is meshes["tib"] and tib.color == "#5CC8A8"          # unchanged surface: not rebuilt
cal = next(m for m in t1.models if m.name == "cal")
assert cal.tracking.path.name == "T0019_jog0001_tal.tra"
assert len(tab.scene._cam_actors) == 1

# ---- save and reopen; remove trials
s = OUT / "trials.bvrsession.json"
tab.save_session_file(str(s), radiographs=False, all_files=False)
names = [t1.name, t2.name, t3.name]
_i = win.tabs.indexOf(tab)                # close it (frees its volumes and radiographs) without the prompt
tab.close_tab()
win.tabs.removeTab(_i)
tab.deleteLater()
tab = t1 = t2 = t3 = nav = tib = cal = meshes = None
import gc  # noqa: E402

gc.collect()
pump(0.5)
tab2 = win.new_tab()
no_dialogs(tab2)
tab2.load_session_file(str(s))
wait(lambda: len(tab2.state.trials) == 3 and tab2.state.trial is not None and tab2.state.trial.name == names[0]
     and len(tab2.state.models) == 3, timeout=300, what="reopen")
pump(1)
assert [t.name for t in tab2.state.trials] == names
assert [m.name for m in tab2.state.trials[0].models] == ["cal", "tal", "Tibia"]
r0, r1 = tab2.state.trials[0].models, tab2.state.trials[1].models
assert r1[0].mesh is r0[0].mesh and r1[1].mesh is r0[1].mesh       # surfaces shared again after reopening
if have_model:
    assert tab2.state.trials[2].joints and all(m.has_axes for m in tab2.state.trials[2].models)
shot(win, "tr_11_reopened")
tab2.remove_trial(tab2.state.trials[1], ask=False)
pump(0.3)
assert [t.name for t in tab2.state.trials] == [names[0], names[2]] and tab2.state.trial.name == names[0]
tab2.remove_trial(tab2.state.trial, ask=False)
wait(lambda: tab2.state.trial is not None and tab2.state.trial.name == names[2] and tab2.state.models, what="after removal")
tab2.remove_trial(tab2.state.trial, ask=False)
pump(0.3)
assert not tab2.state.trials and tab2.stack.currentIndex() == 0
print("TRIALS OK", flush=True)
win.close()
