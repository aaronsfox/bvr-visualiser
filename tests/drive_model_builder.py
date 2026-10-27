"""Headless driver for the Model Builder tab and model use in the Visualiser, on the Autoscoper knee sample.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a python3 tests/drive_model_builder.py OUT_DIR

Follows the three steps: segments (automatic axes, then landmarks placed by real right-click picks and
paint strokes in the viewport), a Grood & Suntay knee joint, and reference poses (scan pose and a
"neutral trial"). Then loads the saved model into a Visualiser tab with a synthetic flexion sweep and
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

# ---- step 1: segments ------------------------------------------------------------------------------
tab.add_preset("knee")
m = tab.model
fem, tib = m.segment("Femur"), m.segment("Tibia")
assert [j.name for j in m.joints] == ["Knee"]
for s in (fem, tib):
    tab._set_side(s, "left")
for s, f in [(fem, "Femur"), (tib, "Tibia")]:
    assert tab.set_surface(s, K / "Models" / f"AUT_Left_{f}.stl")
    tab.set_preview(s, K / "Tracking" / f"Left_{f}.tra")
    pump()
assert fem.axes_source == "staple_femur_cylinder" and fem.acs is not None, fem.error
assert tib.axes_source == "staple_tibia_kai" and tib.acs is not None, tib.error
assert fem.auto.quality["Bone"].startswith("partial") and fem.auto.notes == [], fem.auto.notes
tab.select("segment", fem)
shot("mb_01_femur_auto")
tab.select("segment", tib)
x_before = tib.acs[:3, 0].copy()
tab._toggle_option(tib, "flip_ap")
assert tib.acs[:3, 0] @ x_before < -0.7
tab._toggle_option(tib, "flip_ap")
shot("mb_02_tibia_auto")

# landmarks instead, placed by real picks and paint strokes
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


def right_click(xy, drag_to=None):
    iren.SetEventInformation(xy[0], xy[1], 0, 0)
    iren.InvokeEvent("RightButtonPressEvent")
    if drag_to is not None:
        for t in np.linspace(0, 1, 8):
            p = (np.array(xy) * (1 - t) + np.array(drag_to) * t).round().astype(int)
            iren.SetEventInformation(int(p[0]), int(p[1]), 0, 0)
            iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("RightButtonReleaseEvent")
    pump(0.05)


auto_acs = {s.name: s.acs.copy() for s in (fem, tib)}
errors = []
for s, keys, role in [(fem, ("ME", "LE", "FS"), "femur"), (tib, ("MC", "LC", "TD"), "tibia")]:
    tab.set_axes_source(s, "landmarks")
    for key in keys:
        tab.select("landmark", s, key)
        if key in ("FS", "TD"):
            pts = L[role][f"{key}_points"]
            target = pts.mean(0)
            near = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
            face_camera(near, s.mesh)
            a = screen_of(near)
            right_click((a[0] - 25, a[1]), drag_to=(a[0] + 25, a[1]))
        else:
            target = L[role][key]
            face_camera(target, s.mesh)
            right_click(screen_of(target))
        got = m.landmark(s.name, key)
        assert got is not None, f"{s.name} {key} not placed"
        errors.append((s.name, key, float(np.linalg.norm(got - target))))
        if key == "FS":
            shot("mb_03_painted")
for e in errors:
    print("landmark", *e[:2], f"{e[2]:.1f} mm from target", flush=True)
assert all(e[2] < 15 for e in errors if e[1] not in ("FS", "TD")), errors
assert fem.acs is not None and tib.acs is not None and m.checks() == [], m.checks()

# ---- step 2: the joint ------------------------------------------------------------------------------
knee = m.joint("Knee")
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
m.save(path)
tab2 = win.open_tool("model_builder")
tab2.load_model_file(path)
pump()
assert tab2.model.joint("Knee").reference is not None and tab2.model.segment("Femur").acs is not None
assert win.tabs.tabText(win.tabs.indexOf(tab2)) == tab2.model.name
# segments without a joint are a model too; removing a segment removes its joint
extra = tab2.add_segment("generic", "Patella")
assert extra.side == "none"
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
assert "Add coordinate system…" in [a.text() for a in vt.build_menu("model", femur_m).actions()]
assert "Add coordinate system…" in [a.text() for a in vt.build_menu("axes", femur_m).actions()]
# step A: a coordinate system on one model (the segment is found from the surface file)
assert vt.set_axes(femur_m, path)
assert femur_m.axes_segment == "Femur" and femur_m.acs is not None and not tibia_m.has_axes
assert vt.nav.item_for("axes", femur_m).text(0).startswith("Axes: Femur")
assert vt.nav.item_for("axes", tibia_m).text(0).startswith("No coordinate system")
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
assert len(lines) == n + 1 and lines[0].startswith("frame,time_s,Flexion"), lines[0]
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
ab.add_preset("ankle")
am = ab.model
assert [s.type for s in am.segments] == ["tibia_distal", "talus", "calcaneus"]
assert [j.type for j in am.joints] == ["ankle_isb", "subtalar_isb"]
for s, f in zip(am.segments, ("tib", "tal", "cal")):
    assert ab.set_surface(s, A / f"SOL001A_{f}_aligned.stl")
    pump()
tal, cal = am.segment("Talus"), am.segment("Calcaneus")
assert tal.axes_source == "staple_talus_ankle" and tal.acs is not None, tal.error
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
assert cal.acs is not None and cal.imported["option"] == "CC"
ab.select("segment", cal)
shot("mb_11_calcaneus_aafact")
ab._set_aafact_option(cal, "ST")
assert cal.imported["option"] == "ST"
print("ANKLE OK", flush=True)
win.close()
