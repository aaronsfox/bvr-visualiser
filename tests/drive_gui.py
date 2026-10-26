"""Drive the real GUI on the sample data (no file dialogs) and save screenshots.

Usage: BVR_SAMPLE_DATA=... xvfb-run -s "-screen 0 1600x1000x24" python tests/drive_gui.py OUTDIR
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import pyvista as pv  # noqa: E402

pv.global_theme.allow_empty_mesh = True
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core.config import read_cfg  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402
from bvrviewer.core.session import XrayCamera  # noqa: E402

S = Path(os.environ["BVR_SAMPLE_DATA"])
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
W = S / "2025-02-19-Wrist"
A = S / "2025-01-12-Ankle"

app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1500, 920)
win.show()


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_threads(tab, timeout=120):
    end = time.time() + timeout
    while tab._threads and time.time() < end:
        pump(0.1)
    pump(0.5)


def shot(name, tab=None):
    pump(0.4)
    win.grab().save(str(OUT / f"{name}_window.png"))
    if tab is not None and tab.state.models:
        tab.plotter.screenshot(str(OUT / f"{name}_view.png"))
    print("saved", name, flush=True)


tab = win.current()
shot("01_empty")

# Wrist: one TIF (built from the volume) and one AUT STL, each with its own tracking file,
# plus an untracked STL brought in with its .tfm (CT space) to check that path.
rad = BoneModel(name="rad", path=W / "Volumes" / "rad.tif", voxel_size=(0.391, 0.391, 0.625),
                color="#E8D9B5", tracking_path=W / "Tracking_SAM" / "flx_ext_rad.tra")
mc3 = BoneModel(name="AUT_mc3", path=W / "Models" / "AUT_Models" / "AUT_mc3.stl", color="#B78AF0",
                tracking_path=W / "Tracking_SAM" / "flx_ext_mc3.tra")
cap = BoneModel(name="cap", path=W / "Models" / "untracked" / "cap.stl", color="#5CC8A8")
tab._build_models([rad, mc3, cap])
wait_threads(tab)
assert len(tab.state.models) == 3, [m.name for m in tab.state.models]
tab.timeline.slider.setValue(30)
shot("02_wrist_models", tab)

# Rename, recolour, select, trail
# rename through the navigator, as an inline edit would
it = tab.nav.item_for("model", tab.state.models[0])
it.setText(0, "Radius")
assert tab.state.models[0].name == "Radius", tab.state.models[0].name
# untick in the navigator hides the bone, ticking shows it again
it.setCheckState(0, Qt.Unchecked)
assert not tab.state.models[0].visible
it.setCheckState(0, Qt.Checked)
tab.select(tab.state.models[1])
assert tab.sel_obj is tab.state.models[1] and tab.sel_kind == "model"
tab.state.settings.trail = True
tab.state.settings.trail_step = 6
tab._settings_changed("trail")
shot("03_trail_selected", tab)
tab.state.settings.trail = False
tab._settings_changed("trail")

# Radiographs from the .cfg (the same code the "Import from .cfg" button uses, minus the dialog)
cfg = read_cfg(W / "Wrist_Sample_Data.cfg")
for k, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
    cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder], name=f"X-ray {k}")
    cam.build()
    tab.state.cameras.append(cam)
    tab.scene.add_camera(cam)
tab.refresh_all()
for c in tab.state.cameras:
    tab._set_camera(c, "show_beams", True)
tab.timeline.slider.setValue(0)
tab.set_view("3d")
shot("04_xray_world_default", tab)
# Looking from between the two X-ray sources, as in the design mock-up
import numpy as np  # noqa: E402

c1, c2 = [c.calibration.source for c in tab.state.cameras]
mid = tab.scene._bones_centroid()
d = (c1 + c2) / 2 - mid
d /= np.linalg.norm(d)
up = np.cross(d, c1 - c2)
up /= np.linalg.norm(up)
cam = tab.plotter.camera
cam.focal_point, cam.position, cam.up, cam.view_angle = mid, mid + d * 650 + up * 200, up, 40
tab.plotter.render()
shot("04b_xray_world_source_side", tab)
# And from behind the detectors: planes should be see-through
cam.position = mid - d * 650 + up * 200
tab.plotter.render()
shot("04c_xray_world_behind", tab)
tab.set_xray_view(0)
shot("05_through_xray1", tab)
tab.timeline.slider.setValue(40)
shot("06_through_xray1_frame41", tab)
tab.set_xray_view(1)
shot("07_through_xray2_frame41", tab)

# Hold the radius still in plane view
tab.set_view("xz")
tab.hold_still(tab.state.models[0], True)
tab.refresh_all()
shot("08_hold_radius_xz", tab)
tab.hold_still(tab.state.models[0], False)

# Volume display of the TIF (radiographs hidden for this one)
for c in tab.state.cameras:
    c.visible = False
tab.state.models[0].display = "volume"
tab.state.models[1].visible = False
tab.state.models[2].visible = False
tab._appearance(tab.state.models[0])
tab.set_view("3d")
shot("09_volume_display", tab)
tab.state.models[0].display = "surface"
for m in tab.state.models:
    m.visible = True
for c in tab.state.cameras:
    c.visible = True
tab._appearance(tab.state.models[0])

# Save and reload the session in a new tab
from bvrviewer.core.session import save_session  # noqa: E402

save_session(tab.state, OUT / "wrist.bvrsession.json")
t2 = win.new_tab()
t2.load_session_file(str(OUT / "wrist.bvrsession.json"))
wait_threads(t2)
pump(1)
assert len(t2.state.models) == 3 and len(t2.state.cameras) == 2, (len(t2.state.models), len(t2.state.cameras))
t2.timeline.slider.setValue(20)
shot("10_session_reloaded", t2)

# Ankle: three AUT STLs with tracking, 16-bit radiographs
t3 = win.new_tab()
t3.state.name = "Ankle jog"
models = [BoneModel(name=b, path=A / "Meshes" / f"SOL001A_{b}_aligned.stl", color=c,
                    tracking_path=A / "Tracking" / f"T0019_jog0001_{b}.tra")
          for b, c in [("tib", "#E8D9B5"), ("tal", "#4F9CF9"), ("cal", "#5CC8A8")]]
t3._build_models(models)
wait_threads(t3)
cfg = read_cfg(A / "2025-01-12-Ankle.cfg")
for k, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
    cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder], name=f"X-ray {k}")
    cam.build()
    t3.state.cameras.append(cam)
    t3.scene.add_camera(cam)
t3.refresh_all()
t3.timeline.slider.setValue(10)
t3.set_view("3d")
shot("11_ankle_world", t3)
t3.set_xray_view(1)
shot("12_ankle_through_xray2", t3)

# ---- dialogs (constructed directly, grabbed as images, then closed)
from bvrviewer.app.dialogs import AddCameraDialog, FormatDialog, VoxelDialog  # noqa: E402

win.tabs.setCurrentIndex(0)
tab.timeline.slider.setValue(0)
dlg = AddCameraDialog(win, tab.scene.n_frames, tab._preview_bones(), existing=tab.state.cameras[0])
dlg.show()
pump(0.8)
dlg.grab().save(str(OUT / "13_add_camera_dialog.png"))
assert dlg.ok.isEnabled()
dlg.close()
dlg = VoxelDialog(win, "rad.tif", (0.391, 0.391, 0.625))
dlg.show()
pump(0.4)
dlg.grab().save(str(OUT / "14_voxel_dialog.png"))
dlg.close()
dlg = FormatDialog(win, tab.state.models[1].tracking.fmt, 0)
dlg.show()
pump(0.4)
dlg.grab().save(str(OUT / "15_format_dialog.png"))
dlg.close()
print("saved dialogs", flush=True)

# ---- a tracking file with untracked (NaN) frames, shown with each policy
from bvrviewer.core.tracking import load_tracking  # noqa: E402

lines = (W / "Tracking_SAM" / "flx_ext_mc3.tra").read_text().splitlines()
for i in range(20, 32):
    lines[i] = ",".join(["NaN"] * 16)
nan_path = OUT / "flx_ext_mc3_with_gaps.tra"
nan_path.write_text("\n".join(lines))
mc3_model = tab.state.models[1]
mc3_model.tracking_path, mc3_model.tracking = nan_path, load_tracking(nan_path)
tab._tracking_changed(mc3_model)
assert mc3_model.tracking.n_untracked == 12
tab.set_view("3d")
for c in tab.state.cameras:
    c.visible = False
tab.timeline.slider.setValue(25)
tab.state.settings.untracked = "hide"
tab._settings_changed("untracked")
assert not tab.scene._bone_actors[id(mc3_model)]["surface"].GetVisibility()
tab.state.settings.untracked = "interpolate"
tab._settings_changed("untracked")
assert tab.scene._bone_actors[id(mc3_model)]["surface"].GetVisibility()
shot("16_nan_gap_interpolated", tab)

# ---- a CT-space STL brought into Autoscoper space with its .tfm (should sit exactly on the TIF radius)
t4 = win.new_tab()
ct = BoneModel(name="rad (CT space + tfm)", path=W / "Models" / "rad.stl", color="#F0A04B",
               tfm_path=W / "Transforms" / "rad-DICOM2AUT.tfm", opacity=0.6,
               tracking_path=W / "Tracking_SAM" / "flx_ext_rad.tra")
tif = BoneModel(name="rad (TIF)", path=W / "Volumes" / "rad.tif", voxel_size=(0.391, 0.391, 0.625),
                color="#4F9CF9", opacity=0.6, tracking_path=W / "Tracking_SAM" / "flx_ext_rad.tra")
t4._build_models([ct, tif])
wait_threads(t4)
t4.timeline.slider.setValue(30)
t4.set_view("3d")
shot("17_tfm_vs_tif", t4)
b1 = np.array(t4.scene._bone_actors[id(t4.state.models[0])]["surface"].GetBounds())
b2 = np.array(t4.scene._bone_actors[id(t4.state.models[1])]["surface"].GetBounds())
print("tfm vs tif bounds max diff (mm):", round(float(np.abs(b1 - b2).max()), 2), flush=True)
assert np.abs(b1 - b2).max() < 2.0

# ---- grid follows bones when tracking is attached after the model (reported bug)
t5 = win.new_tab()
bare = BoneModel(name="mc3", path=W / "Models" / "AUT_Models" / "AUT_mc3.stl", color="#B78AF0")
t5._build_models([bare])
wait_threads(t5)
m5 = t5.state.models[0]
m5.tracking_path, m5.tracking = W / "Tracking_SAM" / "flx_ext_mc3.tra", load_tracking(W / "Tracking_SAM" / "flx_ext_mc3.tra")
t5._tracking_changed(m5)
cube = t5.plotter.renderer.cube_axes_actor
gb = np.array(cube.GetBounds())
bb = np.array(t5.scene._bone_actors[id(m5)]["surface"].GetBounds())
inside = np.all(bb[::2] >= gb[::2] - 1e-6) and np.all(bb[1::2] <= gb[1::2] + 1e-6)
print("grid contains tracked bone:", bool(inside), flush=True)
assert inside
shot("18_grid_after_tracking", t5)

# ---- navigator + properties: each kind of item, and its right-click menu
win.tabs.setCurrentIndex(0)
tab.set_view("3d")
for c in tab.state.cameras:
    c.visible = True
    tab._set_camera(c, "show_beams", False)
tab.refresh_all()
expected = [
    ("models", None, ["Add model…", "Show all", "Hide all"]),
    ("model", tab.state.models[0], ["Replace tracking…", "Tracking format…", "Detach tracking",
                                     "Hold this bone still", "Hide", "Rename", "Use file name as name",
                                     "Replace model file…", "Voxel size / surface…", "Remove model"]),
    ("tracking", tab.state.models[2], ["Attach tracking…"]),
    ("cameras", None, ["Add X-ray camera…", "Import cameras from .cfg…"]),
    ("camera", tab.state.cameras[0], ["View through this camera", "Hide", "Rename",
                                       "Change calibration or images…", "Remove camera"]),
    ("scene", None, ["Fit bones in view", "Free 3D view"]),
]
for kind, obj, names in expected:
    got = [a.text() for a in tab.build_menu(kind, obj).actions() if not a.isSeparator()]
    assert got == names, (kind, got)
print("context menus OK", flush=True)
for n, (kind, obj) in enumerate([("model", tab.state.models[0]), ("tracking", tab.state.models[1]),
                                 ("camera", tab.state.cameras[0]), ("cameras", None), ("scene", None),
                                 ("models", None)], start=19):
    tab.nav.select(kind, obj)
    pump(0.3)
    assert (tab.sel_kind, tab.sel_obj) == (kind, obj), (tab.sel_kind, kind)
    win.grab().save(str(OUT / f"{n}_props_{kind}.png"))
# properties edit the model: colour and opacity
m0 = tab.state.models[0]
tab.nav.select("model", m0)
tab._set_model(m0, "color", "#E07A9A")
tab._set_model(m0, "opacity", 0.5, nav=False)
assert tab.scene._bone_actors[id(m0)]["surface"].prop.opacity == 0.5
# per-camera view options: hide camera 2's radiograph only
c2 = tab.state.cameras[1]
tab._set_camera(c2, "show_plane", False)
assert not tab.scene._cam_actors[id(c2)]["plane"].GetVisibility()
assert tab.scene._cam_actors[id(tab.state.cameras[0])]["plane"].GetVisibility()
tab.nav.select("model", m0)
shot("25_props_model_edited", tab)
tab._set_camera(c2, "show_plane", True)
tab._set_model(m0, "opacity", 1.0, nav=False)
print("saved navigator/properties shots", flush=True)

# Video export of the wrist tab (no dialogs): reuse the export loop directly
import imageio.v2 as imageio  # noqa: E402

win.tabs.setCurrentIndex(0)
tab.set_view("3d")
writer = imageio.get_writer(str(OUT / "wrist.mp4"), fps=30, codec="libx264", quality=7, macro_block_size=1)
for f in range(tab.scene.n_frames):
    tab.scene.update_frame(f)
    img = tab.plotter.screenshot(return_img=True)
    h, w = img.shape[:2]
    writer.append_data(img[: h - h % 2, : w - w % 2])
writer.close()
print("video frames", tab.scene.n_frames, flush=True)
print("ALL OK", flush=True)
win.close()
