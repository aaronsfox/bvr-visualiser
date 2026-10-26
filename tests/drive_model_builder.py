"""Headless driver for the Model Builder tab on the Autoscoper knee sample.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a python3 tests/drive_model_builder.py OUT_DIR
Places landmarks by real right-click picks/paints in the viewport (screen positions found by projecting
the target vertices), saves the model, and loads it back.
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
from bvrviewer.app.style import apply_theme  # noqa: E402
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

# side + surfaces
tab._set_side("left")
for role, f in [("femur", "Femur"), ("tibia", "Tibia")]:
    assert tab.set_surface(role, K / "Models" / f"AUT_Left_{f}.stl")
    pump()
for role, f in [("femur", "Femur"), ("tibia", "Tibia")]:
    tab.set_preview(role, K / "Tracking" / f"Left_{f}.tra")
tab.sel = ("model", None, None)
tab.refresh()
shot("mb_01_surfaces")

Tf = load_tracking(K / "Tracking" / "Left_Femur.tra").poses[0]
Tt = load_tracking(K / "Tracking" / "Left_Tibia.tra").poses[0]
L = _auto_landmarks(tab.model.bones["femur"].surface.mesh, tab.model.bones["tibia"].surface.mesh, Tf, Tt)
iren = tab.plotter.iren.interactor


def screen_of(point):
    """Display coordinates of a model-space point in the currently shown (single) bone."""
    ren = tab.plotter.renderer
    ren.SetWorldPoint(*point, 1.0)
    ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return int(round(x)), int(round(y))


def face_camera(point, mesh):
    """Point the camera at the landmark from outside the bone so it is visible."""
    c = np.asarray(mesh.center)
    d = np.asarray(point) - c
    d /= np.linalg.norm(d)
    cam = tab.plotter.camera
    cam.focal_point = tuple(point)
    cam.position = tuple(np.asarray(point) + d * 250)
    up = np.cross(d, [1, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1, 0])
    cam.up = tuple(up)
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


errors = []
for role, keys in [("femur", ("ME", "LE", "FS")), ("tibia", ("MC", "LC", "TD"))]:
    mesh = tab.model.bones[role].surface.mesh
    for key in keys:
        tab.sel = ("landmark", role, key)
        tab.refresh()
        if key in ("FS", "TD"):                       # shaft band: paint a stroke across it
            pts = L[role][f"{key}_points"]
            target = pts.mean(0)
            near = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
            face_camera(near, mesh)
            a = screen_of(near)
            right_click((a[0] - 25, a[1]), drag_to=(a[0] + 25, a[1]))
        else:
            target = L[role][key]
            face_camera(target, mesh)
            right_click(screen_of(target))
        got = tab.model.landmark(role, key)
        assert got is not None, f"{role} {key} not placed"
        errors.append((role, key, float(np.linalg.norm(got - target))))
        if key == "ME":
            shot("mb_02_point_pick")
        if key == "FS":
            shot("mb_03_painted")
for e in errors:
    print("landmark", *e[:2], f"{e[2]:.1f} mm from target", flush=True)
assert all(e[2] < 15 for e in errors if e[1] not in ("FS", "TD")), errors

tab.model.compute()
assert tab.model.bones["femur"].acs is not None and tab.model.bones["tibia"].acs is not None
print("checks", tab.model.checks(), flush=True)
tab.sel = ("bone", "femur", None)
tab.refresh()
tab.plotter.reset_camera()
shot("mb_04_femur_axes")
tab.sel = ("joint", None, "knee")
tab.refresh()
tab.plotter.reset_camera()
tab.plotter.render()
shot("mb_05_joint")
k = tab._preview_kinematics(tab.model.template.joints[0])
print("preview knee", np.round(k[1], 1), np.round(k[2], 1), flush=True)

# ---- automatic axes: per bone, drawn fits, flips, comparison with the landmark axes
tab.sel = ("bone", "femur", None)
tab.refresh()
tab.set_axes_source("femur", "femur_cylinder")
assert tab.model.bones["femur"].acs is not None and tab.model.bones["femur"].auto is not None
tab.plotter.reset_camera()
shot("mb_08_auto_femur_cylinder")
tab.sel = ("bone", "tibia", None)
tab.set_axes_source("tibia", "tibia_ellipse")
tab.plotter.reset_camera()
shot("mb_09_auto_tibia_ellipse")
x_before = tab.model.bones["tibia"].acs[:3, 0].copy()
tab._toggle_option("tibia", "flip_ap")
assert tab.model.bones["tibia"].acs[:3, 0] @ x_before < -0.7
tab._toggle_option("tibia", "flip_ap")
assert tab.model.bones["tibia"].auto_options == {}
tab.sel = ("joint", None, "knee")
tab.refresh()
rows = tab.compare_methods("knee")
assert len(rows) == 3 * 4 and sum(r["current"] for r in rows) == 1
for r in rows:
    print("compare", f"{r['proximal']:15s} {r['distal']:15s}", np.round(r["angles"], 1), flush=True)
tab.plotter.reset_camera()
shot("mb_10_compare")
auto_path = OUT / "knee_left_auto.bvrmodel.json"
tab.model.save(auto_path)
tab.landmarks_all()
assert tab.model.bones["femur"].axes_source == "landmarks"

# save and reload in a new tab
path = OUT / "knee_left.bvrmodel.json"
tab.model.save(path)
QTest.mouseClick(win.start.cards["model_builder"], Qt.LeftButton) if win.current() is win.start else None
tab2 = win.open_tool("model_builder")
tab2.load_model_file(path)
pump()
k2 = tab2._preview_kinematics(tab2.model.template.joints[0])
np.testing.assert_allclose(k2[1], k[1], atol=1e-6)
assert win.tabs.tabText(win.tabs.indexOf(tab2)) == tab2.model.name
shot("mb_06_reloaded")

# menus build for every kind
for kind, role, key in [("model", None, None), ("bone", "femur", None), ("landmark", "femur", "ME")]:
    assert not tab.build_menu(kind, role, key).isEmpty()
# generic template switch keeps surfaces
tab2.model.bones["femur"].landmarks.clear()
for e in tab2.model.bones.values():
    e.landmarks.clear()
tab2.change_template("generic_inertial")
assert all(e.acs is not None for e in tab2.model.bones.values())
shot("mb_07_generic")
print("MODEL BUILDER OK", flush=True)

# ---- the saved model in a Visualiser tab, with a synthetic flexion sweep (the sample has one frame)
from bvrviewer.core import joints as J  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402

jm = J.JointModel.load(path, build_surfaces=False)
Af = jm.bones["femur"].acs
n = 60
flex = np.r_[np.linspace(0, 60, 40), np.linspace(60, 30, 20)]
rows_f, rows_t = [], []
for f in range(n):
    # rotate the tibia about the femur's flexion axis (Z) through the femur origin.
    # Flexion is the negative rotation about the femur Z axis on both sides (Grood & Suntay signs).
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
menu = vt.build_menu("joints", None)
assert [a.text() for a in menu.actions()][0] == "Load joint model…"
links = vt.load_joint_model_file(path)
assert len(links) == 1 and links[0].role_map == {"femur": "Femur", "tibia": "Tibia"}, links[0].role_map
k = vt.joint_values(links[0])
base = k["angles"][0, 0]
print("visualiser flexion", np.round(k["angles"][[0, 20, 39, 59], 0], 1), flush=True)
ok = ~np.isnan(k["angles"][:, 0])
assert ok.sum() == n - 3
np.testing.assert_allclose((k["angles"][:, 0] - base)[ok], flex[ok], atol=1e-6)
np.testing.assert_allclose(k["angles"][ok, 1:], np.repeat(k["angles"][:1, 1:], ok.sum(), 0), atol=1e-6)
assert vt.curves.isVisible()
vt.timeline.slider.setValue(39)
pump(0.2)
assert "Frame 40" in vt.curves.readout.text(), vt.curves.readout.text()
shot("vis_01_joint_curves")
vt._set_reference(links[0], 0)
assert abs(vt.joint_values(links[0])["angles"][39, 0] - 60) < 1e-6
vt.sel_kind, vt.sel_obj = "joint", links[0]
vt.refresh_all()
shot("vis_02_joint_props")
vt.timeline.slider.setValue(46)
pump(0.2)
assert "untracked" in vt.curves.readout.text()
vt.export_joint_csv(links[0], str(OUT / "knee.csv"))
lines = (OUT / "knee.csv").read_text().splitlines()
assert len(lines) == n + 1 and lines[0].startswith("frame,time_s,Flexion"), lines[0]
# rename a model: the joint follows it
vt._rename(vt.state.models[1], "Left tibia")
assert links[0].role_map["tibia"] == "Left tibia" and vt.joint_values(links[0]) is not None
# session round trip keeps the joint
from bvrviewer.core.session import load_session, save_session  # noqa: E402

save_session(vt.state, OUT / "knee.bvrsession.json")
st = load_session(OUT / "knee.bvrsession.json")
assert st.joints[0].role_map["tibia"] == "Left tibia" and st.joints[0].reference_frame == 0
vt2 = win.open_tool("visualiser")
vt2.load_session_file(str(OUT / "knee.bvrsession.json"))
end = time.time() + 120
while vt2._threads and time.time() < end:
    pump(0.1)
pump(0.5)
assert len(vt2.state.joints) == 1 and vt2.joint_values(vt2.state.joints[0]) is not None
vt.remove_joint(links[0])
assert not vt.curves.isVisible() and not vt._joint_actors
print("VISUALISER JOINTS OK", flush=True)
win.close()
