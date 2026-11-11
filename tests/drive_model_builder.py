"""Headless driver for the Model Builder tab and model use in the Visualiser, on the Autoscoper knee sample.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a python3 tests/drive_model_builder.py OUT_DIR

Follows the steps: bones added from their surface files (type and side from the file names), automatic
axes applied, then landmarks placed with the tools (Place point, Paint, Erase, undo; real left-button
clicks and strokes in the viewport, right-drag rotating), a Grood & Suntay knee joint from the two
selected bones, and reference poses (scan pose and a "neutral trial"). Then loads the saved model into a Visualiser tab with a synthetic flexion sweep and
checks segment and joint curves, zeroing, CSV export and sessions.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.model_builder import ReferenceDialog, guess_tfm  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core import joints as J  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402
from bvrviewer.core.session import load_session, save_session  # noqa: E402
from bvrviewer.core.tracking import load_tracking  # noqa: E402
from test_joints import _auto_landmarks  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
K = Path(os.environ["BVR_SAMPLE_DATA"]) / "Knee_SampleData"

app = QApplication.instance() or QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(s=0.15):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def shot(name):
    pump(0.2)
    win.grab().save(str(OUT / f"{name}.png"))
    print("saved", name, flush=True)


from PySide6.QtTest import QTest  # noqa: E402

QTest.mouseClick(win.start.cards["model_builder"], Qt.LeftButton)
pump(0.3)
tab = win.current()
assert tab.kind == "model_builder" and win.tabs.tabText(1) == "Model 1", win.tabs.tabText(1)
shot("mb_00_empty")
for kind in ("model", "segments", "joints"):
    assert not tab.build_menu(kind).isEmpty()

# ---- step 1: bones from their surfaces, then type, method, define, Apply ---------------------------------
added = tab.add_surfaces([K / "Models" / "AUT_Left_Femur.stl", K / "Models" / "AUT_Left_Tibia.stl"])
m = tab.model
fem, tib = added
assert (fem.type, fem.side, fem.name) == ("femur", "left", "Femur") and (tib.type, tib.side) == ("tibia", "left")
assert fem.acs is None and m.status("Femur") == "none" and fem.auto is None      # nothing until Apply
assert win.tabs.tabText(1) == "Model 1 •", win.tabs.tabText(1)                  # unsaved changes
for s, f in [(fem, "Femur"), (tib, "Tibia")]:
    tab.set_preview(s, K / "Tracking" / f"Left_{f}.tra")
tab.select("segment", fem)
assert not tab.tool_buttons["point"].isEnabled() and tab.tool_buttons["navigate"].isChecked()
shot("mb_01_femur_steps")
assert tab.apply_axes(fem) and tab.apply_axes(tib)
assert fem.axes_source == "staple_femur_cylinder" and m.status("Femur") == "applied", fem.error
assert tib.axes_source == "staple_tibia_kai" and tib.acs is not None, tib.error
assert fem.auto.quality["Bone"].startswith("partial") and fem.auto.notes == [], fem.auto.notes
tab.select("segment", fem)
shot("mb_02_femur_applied")
tab.select("segment", tib)
x_before = tib.acs[:3, 0].copy()
tab._toggle_option(tib, "flip_ap")                   # applied and updating automatically: follows at once
assert tib.acs[:3, 0] @ x_before < -0.7
tab._set_auto_update(tib, False)
tab._toggle_option(tib, "flip_ap")                   # not updating: the applied axes stay, marked changed
assert tib.acs[:3, 0] @ x_before < -0.7 and m.status("Tibia") == "changed"
tab.apply_axes(tib)
assert tib.acs[:3, 0] @ x_before > 0.7 and m.status("Tibia") == "applied"
tab._set_auto_update(tib, True)
shot("mb_02b_tibia_auto")

# landmarks instead, placed with the tools by real clicks and strokes (left button)
Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses[0]
Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses[0]
L = _auto_landmarks(fem.mesh, tib.mesh, Tf, Tt)
iren = tab.plotter.iren.interactor


def screen_of(point):
    ren = tab.plotter.renderer
    ren.SetWorldPoint(*point, 1.0)
    ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return int(round(x)), int(round(y))


def face_camera(point, mesh):
    c = np.asarray(mesh.center)
    d = np.asarray(point) - c
    d /= np.linalg.norm(d)
    cam = tab.plotter.camera
    cam.focal_point = tuple(point)
    cam.position = tuple(np.asarray(point) + d * 250)
    cam.up = tuple(np.cross(d, [1, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1, 0]))
    tab.plotter.renderer.ResetCameraClippingRange()
    tab.plotter.render()
    pump(0.1)


def click(xy, drag_to=None, button="Left"):
    iren.SetEventInformation(xy[0], xy[1], 0, 0)
    iren.InvokeEvent(f"{button}ButtonPressEvent")
    if drag_to is not None:
        for t in np.linspace(0, 1, 8):
            p = (np.array(xy) * (1 - t) + np.array(drag_to) * t).round().astype(int)
            iren.SetEventInformation(int(p[0]), int(p[1]), 0, 0)
            iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent(f"{button}ButtonReleaseEvent")
    pump(0.05)


auto_acs = {s.name: s.acs.copy() for s in (fem, tib)}
errors = []
for s, keys, role in [(fem, ("ME", "LE", "FS"), "femur"), (tib, ("MC", "LC", "TD"), "tibia")]:
    tab.set_axes_source(s, "landmarks")
    assert m.status(s.name) == "failed"                 # applied axes kept until the landmarks give new ones
    for key in keys:
        tab.edit_landmark(s, key)
        painted = key in ("FS", "TD")
        assert tab.tool == ("paint" if painted else "point"), (key, tab.tool)
        assert tab.tool_buttons["paint" if painted else "point"].isChecked()
        assert not tab.tool_buttons["point" if painted else "paint"].isEnabled()
        if painted:
            pts = L[role][f"{key}_points"]
            target = pts.mean(0)
            near = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
            face_camera(near, s.mesh)
            a = screen_of(near)
            click((a[0] - 25, a[1]), drag_to=(a[0] + 25, a[1]))
        else:
            target = L[role][key]
            face_camera(target, s.mesh)
            click(screen_of(target))
        got = m.landmark(s.name, key)
        assert got is not None, f"{s.name} {key} not placed"
        errors.append((s.name, key, float(np.linalg.norm(got - target))))
        if key == "FS":
            # erase part of the painted area, then undo it
            n0 = len(s.landmarks["FS"]["points"])
            assert tab.set_tool("erase")
            click((a[0] - 25, a[1]), drag_to=(a[0] - 5, a[1]))
            n1 = len(s.landmarks.get("FS", {}).get("points", []))
            assert n1 < n0, (n0, n1)
            tab.undo()
            assert len(s.landmarks["FS"]["points"]) == n0
            tab.set_tool("paint")
            # right-drag rotates while a tool is active, without painting
            pos = np.array(tab.plotter.camera.position)
            click((a[0], a[1] - 60), drag_to=(a[0] + 80, a[1] - 60), button="Right")
            assert not np.allclose(pos, tab.plotter.camera.position)
            assert len(s.landmarks["FS"]["points"]) == n0
            face_camera(near, s.mesh)
            iren.SetEventInformation(a[0], a[1], 0, 0)
            iren.InvokeEvent("MouseMoveEvent")
            pump(0.05)
            assert tab._cursor_actor is not None                       # the brush ring follows the mouse
            shot("mb_03_painted")
for e in errors:
    print("landmark", *e[:2], f"{e[2]:.1f} mm from target", flush=True)
assert all(e[2] < 15 for e in errors if e[1] not in ("FS", "TD")), errors
assert m.status("Femur") == m.status("Tibia") == "applied" and m.checks() == [], m.checks()  # updated as placed
tab.select("segment", fem)
tab.set_tool("pick")
face_camera(L["femur"]["ME"], fem.mesh)
click(screen_of(L["femur"]["ME"]))
assert "nearest landmark: Medial epicondyle" in tab.hint.text(), tab.hint.text()
tab.set_tool("navigate")

# ---- step 2: the joint, from the two bones selected together -------------------------------------------------
tab.select_pair(fem, tib)
assert len(tab.tree.selectedItems()) == 2
assert m.joint_types_for("Femur", "Tibia")[0] == ("knee_grood_suntay", "Femur", "Tibia")
assert m.joint_types_for("Tibia", "Femur")[0] == ("knee_grood_suntay", "Femur", "Tibia")   # either order
shot("mb_04a_pair")
from PySide6.QtWidgets import QPushButton  # noqa: E402

add_btn = next(b for b in tab.props.findChildren(QPushButton) if b.text() == "Add joint")
add_btn.click()
pump()
knee = m.joint("Knee")
assert (knee.type, knee.proximal, knee.distal) == ("knee_grood_suntay", "Femur", "Tibia")
tab.select("joint", knee)
rows = tab.compare_methods(knee)
assert len(rows) == 5 * 6 and sum(r["current"] for r in rows) == 1
rows = [r for r in rows if r["angles"] is not None]   # Kai femur needs a whole femur; AAFACT not run
assert len(rows) == 4 * 5, len(rows)                   # 3 femur methods + landmarks, 4 tibia + landmarks
for r in rows:
    print("compare", f"{r['proximal']:22s} {r['distal']:22s}", np.round(r["angles"], 1), flush=True)
shot("mb_04_joint_compare")
for s in (fem, tib):                                  # back to automatic axes
    tab.set_axes_source(s, s.stype.auto_methods[0])
assert all(np.allclose(s.acs, auto_acs[s.name]) for s in (fem, tib))

# ---- step 3: reference poses --------------------------------------------------------------------------
assert guess_tfm(fem.surface).endswith("Left_Femur-DICOM2AUT.tfm"), guess_tfm(fem.surface)
dlg = ReferenceDialog(tab, m, knee, "scan")
assert dlg.tfm_p.value().endswith("Left_Femur-DICOM2AUT.tfm") and dlg.tfm_d.value().endswith("Left_Tibia-DICOM2AUT.tfm")
dlg.apply()
assert knee.reference.source["kind"] == "scan"
tab.select("reference", knee)
tab.plotter.reset_camera()
shot("mb_05_reference_scan")
_, raw, norm = tab._preview_kinematics(knee)
print("preview frame 1: anatomical", raw["angles"][0].round(1), "relative to the scan pose", norm["angles"][0].round(3))
np.testing.assert_allclose(norm["angles"][0], 0, atol=1e-6)      # sample frame 1 = the scan pose
# a "neutral trial" from the preview tracking (all frames), realigned
dlg = ReferenceDialog(tab, m, knee, "trial")
dlg.mode.setCurrentIndex(dlg.mode.findData("realign"))
dlg.apply()
assert knee.reference.mode == "realign" and knee.reference.source["frames_used"] == 1
tab.select("joint", knee)
shot("mb_06_joint_props")
tab.select("reference", knee)
shot("mb_07_reference_props")
# back to the scan pose for the Visualiser checks
ReferenceDialog(tab, m, knee, "scan").apply()

path = OUT / "knee_left.bvrmodel.json"
assert tab._save_to(path)
assert win.tabs.tabText(1) == "Model 1" and not tab.has_unsaved()
tab2 = win.open_tool("model_builder")
tab2.load_model_file(path)
pump()
assert tab2.model.joint("Knee").reference is not None and tab2.model.segment("Femur").acs is not None
assert tab2.model.status("Femur") == "applied" and tab2.model.segment("Femur").applied
assert win.tabs.tabText(win.tabs.indexOf(tab2)) == tab2.model.name
# a bone that isn't applied is saved without axes
extra = tab2.add_surfaces([K / "Models" / "AUT_Left_Tibia.stl"])[0]
tab2.set_segment_type(extra, "generic")
assert extra.side == "none" and extra.type == "generic" and extra.acs is None
assert ("landmarks", "Landmarks (placed by hand)") in tab2.model.axes_sources(extra.name)
tab2.model.remove_segment("Tibia")
assert tab2.model.joints == []
print("MODEL BUILDER OK", flush=True)

# ---- the model in a Visualiser tab, with a synthetic flexion sweep (the sample has one frame) ----------
km = J.KinematicModel.load(path, build_surfaces=False)
Af = km.segment("Femur").acs
n = 60
flex = np.r_[np.linspace(0, 60, 40), np.linspace(60, 30, 20)]
rows_f, rows_t = [], []
for f in range(n):
    # flexion = negative rotation about the femur's Z axis on both sides (Grood & Suntay signs)
    R = np.eye(4)
    R[:3, :3] = J.compose("ZXY", np.radians([-flex[f], 0, 0]))
    W = Tf @ Af @ R @ np.linalg.inv(Af) @ np.linalg.inv(Tf)
    Pt = W @ Tt
    if 45 <= f < 48:
        Pt = np.full((4, 4), np.nan)                 # a gap
    rows_f.append(",".join(f"{v:.9g}" for v in Tf.ravel()))
    rows_t.append(",".join(f"{v:.9g}" for v in Pt.ravel()))
(OUT / "sweep_femur.tra").write_text("\n".join(rows_f) + "\n")
(OUT / "sweep_tibia.tra").write_text("\n".join(rows_t) + "\n")

vt = win.open_tool("visualiser")
vt._build_models([BoneModel(name="Femur", path=K / "Models" / "AUT_Left_Femur.stl", color="#E8D9B5",
                            tracking_path=OUT / "sweep_femur.tra"),
                  BoneModel(name="Tibia", path=K / "Models" / "AUT_Left_Tibia.stl", color="#4F9CF9",
                            tracking_path=OUT / "sweep_tibia.tra")])
end = time.time() + 120
while vt._threads and time.time() < end:
    pump(0.1)
pump(0.5)
assert len(vt.state.models) == 2
assert [a.text() for a in vt.build_menu("joints", None).actions()][0] == "Add joint…"
femur_m, tibia_m = vt.state.models
assert "Load coordinate system…" in [a.text() for a in vt.build_menu("model", femur_m).actions()]
assert "Load file…" in [a.text() for a in vt.build_menu("axes", femur_m).actions()]
# step A: a coordinate system on one model (the segment is found from the surface file)
assert vt.set_axes(femur_m, path)
assert femur_m.axes_segment == "Femur" and femur_m.acs is not None and not tibia_m.has_axes
assert vt.nav.item_for("axes", femur_m).text(0).startswith("Axes · Femur")
assert vt.nav.item_for("axes", tibia_m).text(0) == "Axes"          # nothing loaded: greyed out
fem_k = vt.joint_values(femur_m)
np.testing.assert_allclose(fem_k["angles"][~np.isnan(fem_k["angles"][:, 0])], 0, atol=1e-6)   # the femur is still
shot("vis_00_axes_on_model")
# step B: the joint; the tibia gets its coordinate system from the joint's file
items = vt.load_joint_model_file(path)
assert [i.name for i in items] == ["Knee"] and tibia_m.axes_segment == "Tibia"
knee_item = items[0]
assert knee_item.model_map == {"Femur": "Femur", "Tibia": "Tibia"}
k = vt.joint_values(knee_item)
print("visualiser flexion (relative to the scan pose)", np.round(k["angles"][[0, 20, 39, 59], 0], 1), flush=True)
ok = ~np.isnan(k["angles"][:, 0])
assert ok.sum() == n - 3
np.testing.assert_allclose(k["angles"][ok, 0], flex[ok], atol=1e-6)        # frame 0 = scan pose = zero
np.testing.assert_allclose(k["angles"][ok, 1:], 0, atol=1e-6)
vt._set_use_reference(knee_item, False)
anat = vt.joint_values(knee_item)["angles"]
assert abs(anat[0, 0] - 17.7) < 0.2, anat[0]                                 # anatomical: automatic-axes offset
vt._set_use_reference(knee_item, True)
assert vt.curves.isVisible()
vt.timeline.slider.setValue(39)
pump(0.2)
assert "Frame 40" in vt.curves.readout.text(), vt.curves.readout.text()
shot("vis_01_joint_curves")
vt.sel_kind, vt.sel_obj = "joint", knee_item
vt.refresh_all()
shot("vis_02_joint_props")
# the tibia turns by the flexion angle about its own axes
tib_k = vt.joint_values(tibia_m)
turn = np.degrees(np.arccos(np.clip((np.trace(J.compose("ZXY", np.radians(tib_k["angles"][39]))) - 1) / 2, -1, 1)))
assert abs(turn - 60) < 1e-4, turn
vt._set_curve_joint(tibia_m)
vt.sel_kind, vt.sel_obj = "axes", tibia_m
vt.refresh_all()
shot("vis_03_segment_curves")
vt._toggle_axes(tibia_m, False)
assert vt.nav.item_for("axes", tibia_m).checkState(0) == Qt.Unchecked
vt._toggle_axes(tibia_m, True)
vt._set_curve_joint(knee_item)
vt._set_reference(knee_item, 0)
assert abs(vt.joint_values(knee_item)["angles"][39, 0] - 60) < 1e-6
vt.timeline.slider.setValue(46)
pump(0.2)
assert "untracked" in vt.curves.readout.text()
vt.export_joint_csv(knee_item, str(OUT / "knee.csv"))
lines = (OUT / "knee.csv").read_text().splitlines()
assert len(lines) == n + 1 and lines[0].startswith("frame,time_s,Angle X: Adduction"), lines[0]   # X Y Z order
vt.state.settings.curve_labels = "anatomical"
vt.export_joint_csv(knee_item, str(OUT / "knee_anatomical.csv"))
assert (OUT / "knee_anatomical.csv").read_text().startswith("frame,time_s,Flexion")
vt.state.settings.curve_labels = "axes"
# rename a model: the items follow it
vt._rename(vt.state.models[1], "Left tibia")
assert knee_item.model_map["Tibia"] == "Left tibia" and vt.joint_values(knee_item) is not None
save_session(vt.state, OUT / "knee.bvrsession.json")
st = load_session(OUT / "knee.bvrsession.json")
assert [i.name for i in st.joints] == ["Knee"] and [m.axes_segment for m in st.models] == ["Femur", "Tibia"]
vt2 = win.open_tool("visualiser")
vt2.load_session_file(str(OUT / "knee.bvrsession.json"))
end = time.time() + 120
while vt2._threads and time.time() < end:
    pump(0.1)
pump(0.5)
assert len(vt2.state.joints) == 1 and vt2.joint_values(vt2.state.joints[0]) is not None
assert all(m.acs is not None for m in vt2.state.models) and vt2.joint_values(vt2.state.models[1]) is not None
for it in list(vt.state.joints):
    vt.remove_joint(it)
for mm in vt.state.models:
    vt.remove_axes(mm)
assert not vt.curves.isVisible() and not vt._joint_actors
# Load model: a new tab gets every bone (from the model's surface files), its coordinate system and the joint
vt3 = win.open_tool("visualiser")
vt3.load_full_model(str(path))
end = time.time() + 120
while vt3._threads and time.time() < end:
    pump(0.1)
pump(0.5)
assert [mm.name for mm in vt3.state.models] == ["Femur", "Tibia"], [mm.name for mm in vt3.state.models]
assert [mm.axes_segment for mm in vt3.state.models] == ["Femur", "Tibia"] and all(mm.acs is not None
                                                                                 for mm in vt3.state.models)
assert [j.name for j in vt3.state.joints] == ["Knee"]
assert vt3.nav.roots["models"].text(0) == "Bones (2)"
assert vt3.joint_values(vt3.state.joints[0]) is None                 # no tracking yet: no joint values
shot("vis_04_load_model")
# loading it into a tab that already has those bones reuses them (no duplicates)
vt3.load_full_model(str(path))
pump(0.3)
assert len(vt3.state.models) == 2 and len(vt3.state.joints) == 1
print("VISUALISER OK", flush=True)

# ---- ankle: STAPLE talus axes and AAFACT results (imported; AAFACT itself isn't run here) ----------
from test_aafact import write_aafact_xlsx  # noqa: E402

A = Path(os.environ["BVR_SAMPLE_DATA"]) / "2025-01-12-Ankle" / "Meshes"
ab = win.open_tool("model_builder")
segs = ab.add_surfaces([A / f"SOL001A_{f}_aligned.stl" for f in ("tib", "tal", "cal")])
am = ab.model
assert [s.type for s in am.segments] == ["tibia", "talus", "calcaneus"]       # guessed from the file names
ab.set_segment_type(segs[0], "tibia_distal")                                   # the ankle's (distal) tibia
tib_a, tal, cal = segs
assert [s.type for s in am.segments] == ["tibia_distal", "talus", "calcaneus"] and tib_a.axes_source == "aafact"
assert tal.axes_source == "staple_talus_ankle" and tal.acs is None and tal.auto is None
assert ab.apply_axes(tal) and tal.acs is not None, tal.error
assert tal.auto.quality["Algorithm"] == "STAPLE-Talus"
ab.select("segment", tal)
shot("mb_10_talus_staple")
assert cal.axes_source == "aafact" and cal.acs is None and "AAFACT" in cal.error
R = np.eye(3)
xl = OUT / "CoordinateSystem_bvrjob.xlsx"
write_aafact_xlsx(xl, [("CC_bvr_Calcaneus_Right", "bvr_Calcaneus_Right", "Calcaneus", "Right",
                        np.array(cal.mesh.center), R),
                       ("ST_bvr_Calcaneus_Right", "bvr_Calcaneus_Right", "Calcaneus", "Right",
                        np.array(cal.mesh.center), R[:, [1, 2, 0]] @ np.diag([1, 1, 1]))])
assert ab.import_aafact(cal, str(xl))
assert cal.acs is None and am.status("Calcaneus") == "none" and cal.imported["option"] == "CC"
assert ab.apply_axes(cal) and cal.acs is not None
ab.select("segment", cal)
shot("mb_11_calcaneus_aafact")
ab._set_aafact_option(cal, "ST")
assert cal.imported["option"] == "ST"
# joints offered for each pair of bones
assert am.joint_types_for("Talus", "Calcaneus")[0] == ("subtalar_isb", "Talus", "Calcaneus")
ab.select_pair(cal, tal)
assert "Add joint" in [b.text() for b in ab.props.findChildren(QPushButton)]
ab.add_joint(*am.joint_types_for("Calcaneus", "Talus")[0])
assert [(j.type, j.proximal, j.distal) for j in am.joints] == [("subtalar_isb", "Talus", "Calcaneus")]
ab.select_pair(tib_a, tal)                       # the tibia's axes aren't applied: no joint yet
assert any("Apply the axes of Tibia" in w.text() for w in ab.props.findChildren(type(ab.hint)))
shot("mb_12_pair_not_applied")
print("ANKLE OK", flush=True)

# ---- Model Builder requests 3-9 (11 Nov 2026) -------------------------------------------------------------
from PySide6.QtWidgets import QComboBox, QLabel  # noqa: E402

from bvrviewer.app.model_builder import ConfirmBonesDialog, bone_items  # noqa: E402

win.open_tool("model_builder")
pump(0.3)
rb = win.current()
assert rb.kind == "model_builder" and not rb.model.segments

# #4: bone names only, with the two tibias told apart
titles = [t for _, t in bone_items()]
assert not any("·" in t for t in titles) and "Proximal tibia (knee)" in titles and "Distal tibia (ankle)" in titles

# #3: what each surface was taken to be, shown to check and change
files = [K / "Models" / "AUT_Left_Femur.stl", K / "Models" / "AUT_Left_Tibia.stl",
         A / "SOL001A_tib_aligned.stl", Path("scan_0042.stl")]
dlg = ConfirmBonesDialog(rb, files)
dlg.show()
pump(0.3)
dlg.grab().save(str(OUT / "mb_20_confirm_bones.png"))
v = dlg.values()
assert [(x["name"], x["type"], x["side"]) for x in v[:2]] == [("Femur", "femur", "left"), ("Tibia", "tibia", "left")]
assert v[3]["type"] == "generic" and v[3]["name"] == "scan_0042"
assert "not recognised" in dlg.rows[3]["how"].text()
row = dlg.rows[2]                                   # the ankle's tibia guesses as a knee tibia: change it
row["type"].setCurrentIndex(row["type"].findData("tibia_distal"))
assert dlg.values()[2]["type"] == "tibia_distal" and dlg.values()[2]["name"] == "Tibia"
assert "set by you" in row["how"].text()
row["name"].setText("Shank")
row["type"].setCurrentIndex(row["type"].findData("talus"))
assert dlg.values()[2]["name"] == "Shank"           # an edited name stays
dlg.close()
segs = rb.add_surfaces(files[:2], choices=[{"name": "Femur", "type": "femur", "side": "left"},
                                           {"name": "Knee tibia", "type": "tibia", "side": "left"}])
assert [s.name for s in segs] == ["Femur", "Knee tibia"]
fem3, tib3 = segs
pump(0.3)


def nav_item(text):
    it = rb.tree.findItems(text, Qt.MatchContains | Qt.MatchRecursive)
    return it[0] if it else None


# #8: the Axes row is grey italics until applied, green once applied, amber after a change
ax = nav_item("Axes: GIBOC-Cylinder")
assert ax.foreground(0).color().name().upper() == "#7A8291" and ax.font(0).italic()
assert rb.apply_axes(fem3)
ax = nav_item("Axes: GIBOC-Cylinder")
assert ax.foreground(0).color().name().upper() == "#8FD3A9" and not ax.font(0).italic()
rb._toggle_option(fem3, "flip_ap")
fem3.auto_update = False
rb._toggle_option(fem3, "flip_ap")
rb._toggle_option(fem3, "flip_ap")
if rb.model.status("Femur") == "changed":
    assert nav_item("Axes: GIBOC-Cylinder").foreground(0).color().name().upper() == "#F2C48D"
fem3.auto_update = True
rb._toggle_option(fem3, "flip_ap")
pump(0.2)

# #9: right-clicking the hint rows opens their section's menu
from bvrviewer.app import model_builder as MBm  # noqa: E402

ph = nav_item("Ctrl+click two bones")

assert ph.data(0, MBm.KIND) == "joints"
acts = [a.text() for a in rb.build_menu(ph.data(0, MBm.KIND)).actions()]
assert acts == ["Add joint…"], acts


# #5: change dropdowns with real clicks in their popups (the sheet is rebuilt after each one)
def prop_widget(label_text):
    g = rb.props.grid
    for i in range(g.count()):
        w = g.itemAt(i).widget()
        if isinstance(w, QLabel) and w.text() == label_text:
            r, _c, _, _ = g.getItemPosition(i)
            return g.itemAtPosition(r, 1).widget()
    return None


def pick(label_text, text):
    c = prop_widget(label_text)
    assert isinstance(c, QComboBox), label_text
    idx = c.findText(text, Qt.MatchContains)
    assert idx >= 0, (text, [c.itemText(i) for i in range(c.count())])
    view = c.view()
    for _ in range(100):                 # the last popup has finished closing
        if not view.window().isVisible():
            break
        pump(0.02)
    c.showPopup()
    for _ in range(100):
        pump(0.02)
        if view.window().isVisible():
            break
    pump(0.3)
    view.scrollTo(view.model().index(idx, 0))
    pump(0.2)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, view.visualRect(view.model().index(idx, 0)).center())
    pump(0.6)
    now = prop_widget(label_text)
    assert text in now.currentText(), (label_text, text, now.currentText())


rb.select("segment", fem3)
pump(0.3)
pick("Method", "Landmarks")
assert fem3.axes_source == "landmarks"
pump(0.2)
win.grab().save(str(OUT / "mb_21_landmarks_steps.png"))
pick("Method", "GIBOC-Spheres")
assert fem3.axes_source == "staple_femur_spheres"
pick("Bone", "General")
assert fem3.type == "generic"
pick("Bone", "Femur")
assert fem3.type == "femur"
pick("Side", "Right")
assert fem3.side == "right"
pick("Side", "Left")

# #6: what each method needs and what to do, under the Method dropdown, and the full instructions
assert prop_widget("Needs") is not None and "distal femur" in prop_widget("Needs").text()
win.grab().save(str(OUT / "mb_22_method_steps.png"))
rb.show_method_instructions(fem3)
pump(0.3)
rb._instructions.grab().save(str(OUT / "mb_23_instructions.png"))
html = rb.method_instructions_html(fem3)
assert "What to do" in html and "Modenese" in html, html[:300]
rb._instructions.close()
rb.select("segment", tib3)
rb.set_axes_source(tib3, "staple_tibia_ellipse")
assert "Plateau or Kai" in rb.method_instructions_html(tib3)

# #7: automatic methods on surfaces made from the knee sample's TIF partial volumes
tifs = rb.add_surfaces([K / "Volumes" / "Left_Femur.tif", K / "Volumes" / "Left_Tibia.tif"],
                       voxel_size=(0.223, 0.223, 0.625),
                       choices=[{"name": "Femur (TIF)", "type": "femur", "side": "left"},
                                {"name": "Tibia (TIF)", "type": "tibia", "side": "left"}])
ftif, ttif = tifs
for s, methods in ((ftif, ("staple_femur_cylinder", "staple_femur_spheres", "staple_femur_ellipsoids")),
                   (ttif, ("staple_tibia_kai", "staple_tibia_plateau", "staple_tibia_ellipse",
                           "staple_tibia_centroids"))):
    for meth in methods:
        rb.set_axes_source(s, meth)
        assert rb.apply_axes(s) and s.acs is not None, (meth, s.error)
        assert "Loose pieces ignored" in s.auto.quality, s.auto.quality
        print("TIF", s.name, meth, "ok", s.auto.quality["Loose pieces ignored"], flush=True)
rb.set_axes_source(ftif, "staple_femur_kai")
assert not rb.apply_axes(ftif) and "whole femur" in ftif.error
rb.set_axes_source(ftif, "staple_femur_cylinder")
rb.apply_axes(ftif)
rb.select("segment", ttif)
pump(0.3)
win.grab().save(str(OUT / "mb_24_tif_tibia_applied.png"))
print("REQUESTS OK", flush=True)
win.close()
