"""Drive the GUI through the 9 Nov 2026 requests (feature_requests/REQUESTS.md) and save screenshots.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a -s "-screen 0 1600x1000x24" python tests/drive_requests.py OUTDIR

Checks: #7 every tool opens from the menus inside a tab; #1 typed slider values; #2 curve legend labels;
#6 the tracking format prompt (and remembering it); #4 the Tracking/Axes rows and their menus, refresh;
#5 frame offsets and the length-mismatch notes; #8 the 2D camera panels.
"""

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import pyvista as pv  # noqa: E402

pv.global_theme.allow_empty_mesh = True
QApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
from bvrviewer.app import dialogs  # noqa: E402
from bvrviewer.app.camera_views import silhouette_path  # noqa: E402
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core.config import read_cfg  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402
from bvrviewer.core.session import XrayCamera  # noqa: E402

S = Path(os.environ["BVR_SAMPLE_DATA"])
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
W = S / "2025-02-19-Wrist"

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


def shot(name):
    pump(0.5)
    win.grab().save(str(OUT / f"{name}.png"))
    print("saved", name, flush=True)


# ---- #7: open any tool from the File menu or "+ New tab", all in one window
file_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&File")
new_tab = next(a.menu() for a in file_menu.actions() if a.text() == "New tab")
titles = [a.text() for a in new_tab.actions() if a.text()]
assert titles[:2] == ["Visualiser", "Model Builder"], titles
assert [a.text() for a in win.plus.menu().actions() if a.text()] == titles
next(a for a in new_tab.actions() if a.text() == "Visualiser").trigger()
pump()
tab = win.current()
next(a for a in new_tab.actions() if a.text() == "Model Builder").trigger()
pump()
assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Start", "Visualiser 1", "Model 1"]
assert len(QApplication.topLevelWidgets()) >= 1 and win.isVisible()
win.tabs.setCurrentWidget(tab)
print("#7 OK", flush=True)

# ---- #6: tracking is loaded through the format prompt (here accepted as filled in, and remembered)
rad = BoneModel(name="rad", path=W / "Volumes" / "rad.tif", voxel_size=(0.391, 0.391, 0.625), color="#E8D9B5")
mc3 = BoneModel(name="mc3", path=W / "Models" / "AUT_Models" / "AUT_mc3.stl", color="#B78AF0")
tab._build_models([rad, mc3])
wait_threads(tab)
rad, mc3 = tab.state.models

seen = []


def accept_and_remember(self):
    seen.append((self.windowTitle(), self.check.text(), self.frames.text(), not self.remember.isHidden()))
    self.remember.setChecked(True)
    return QDialog.Accepted


real_exec = dialogs.FormatDialog.exec
dialogs.FormatDialog.exec = accept_and_remember
tab.attach_tracking(rad, str(W / "Tracking_SAM" / "flx_ext_rad.tra"))
assert len(seen) == 1 and seen[0][3], seen
assert seen[0][1].startswith("✓"), seen[0][1]
assert tab.tracking_format and tab.tracking_format["kind"] == "matrix"
tab.attach_tracking(mc3, str(W / "Tracking_SAM" / "flx_ext_mc3.tra"))
assert len(seen) == 1, "a matching file should use the remembered options without asking"
assert rad.tracking.n_rows == mc3.tracking.n_rows == 61
print("#6 OK", flush=True)

# ---- #4: the Tracking and Axes rows
it_t = tab.nav.item_for("tracking", rad)
it_a = tab.nav.item_for("axes", rad)
assert it_t.text(0).startswith("Tracking · flx_ext_rad.tra") and it_t.font(0).bold(), it_t.text(0)
assert it_a.text(0) == "Axes" and it_a.font(0).italic() and not it_a.font(0).bold()
menu = lambda kind, obj: [a.text() for a in tab.build_menu(kind, obj).actions() if a.text()]  # noqa: E731
assert menu("tracking", rad)[:2] == ["Load new file…", "Refresh current file"], menu("tracking", rad)
assert menu("axes", rad) == ["Load file…"]
# refresh picks up a file saved over: shorten a copy of mc3's tracking, then refresh
copy = OUT / "mc3_copy.tra"
shutil.copy(W / "Tracking_SAM" / "flx_ext_mc3.tra", copy)
dialogs.FormatDialog.exec = lambda self: QDialog.Accepted
tab.tracking_format = None
tab.attach_tracking(mc3, str(copy))
lines = copy.read_text().splitlines()
copy.write_text("\n".join(lines[:50]))
tab.refresh_tracking(mc3)
assert mc3.tracking.n_rows == 50, mc3.tracking.n_rows
assert "50 rows" in tab.nav.item_for("tracking", mc3).text(0)
tab.nav.select("tracking", mc3)
shot("r4_tracking_rows")
print("#4 OK", flush=True)

# ---- #5: different lengths are pointed out, and a file can start later in the trial
captured = {}


def look(self):
    captured["frames"] = self.frames.text()
    self.start.setValue(12)            # first row = trial frame 12
    return QDialog.Accepted


dialogs.FormatDialog.exec = look
tab.edit_format(mc3)
assert "different" in captured["frames"], captured["frames"]
assert mc3.tracking.offset == 11 and mc3.tracking.n_frames == 61
assert mc3.tracking.pose(5) is None and mc3.tracking.pose(11) is not None
assert "from frame 12" in tab.nav.item_for("tracking", mc3).text(0)
tab.nav.select("tracking", mc3)
pump()
shot("r5_offset_properties")
dialogs.FormatDialog.exec = real_exec
# the dialog itself, as a user would see it
dlg = dialogs.FormatDialog(tab, mc3.tracking.fmt, 0, path=copy, detected=mc3.tracking.fmt, offset=11, n_rows=50,
                           others=tab._other_tracking(mc3), bone="mc3", ask_remember=True)
dlg.show()
pump()
dlg.grab().save(str(OUT / "r5_r6_format_dialog.png"))
dlg.close()
# put mc3 back to the full file
dialogs.FormatDialog.exec = lambda self: QDialog.Accepted
tab.attach_tracking(mc3, str(W / "Tracking_SAM" / "flx_ext_mc3.tra"))
dialogs.FormatDialog.exec = real_exec
print("#5 OK", flush=True)

# ---- #1: typed values next to sliders, and the timeline's frame box
tab.nav.select("model", rad)
pump()
from PySide6.QtWidgets import QSlider  # noqa: E402

sl = next(w for w in tab.props.findChildren(QSlider))
box = sl.value_box
box.setValue(40)                       # typing 40 % and pressing Enter
assert abs(rad.opacity - 0.4) < 1e-9 and sl.value() == 40, (rad.opacity, sl.value())
sl.setValue(75)
assert abs(box.value() - 75) < 1e-9
tab.timeline.frame_box.setValue(25)
assert tab.scene.frame == 24, tab.scene.frame
tab._set_model(rad, "opacity", 1.0)
print("#1 OK", flush=True)

# ---- #2: legend labels Angle/Translation X Y Z (by default), anatomical on request
from bvrviewer.core import joints as J  # noqa: E402

k = J._kinematics(type("Jt", (), {"convention": J.JOINT_CONVENTIONS["grood_suntay_knee"]})(), "right",
                  np.eye(4), np.eye(4), np.array([np.eye(4)]), np.array([np.eye(4)]))
tab.curves.setVisible(True)
tab.curves.set_data(k, "test")
legend = [it[1].text for it in tab.curves.plots[0].plotItem.legend.items]
assert legend == ["Angle X", "Angle Y", "Angle Z"], legend
legend = [it[1].text for it in tab.curves.plots[1].plotItem.legend.items]
assert legend == ["Translation X", "Translation Y", "Translation Z"], legend
tab.curves.label_style.setCurrentIndex(1)
legend = [it[1].text for it in tab.curves.plots[0].plotItem.legend.items]
assert legend == ["Flexion", "Adduction", "Int. rotation"], legend
assert tab.state.settings.curve_labels == "anatomical"
tab.curves.label_style.setCurrentIndex(0)
shot("r2_legend")
tab.curves.set_data(None)
print("#2 OK", flush=True)

# ---- #8: 2D camera panels beside the 3D view, bones in their colours
cfg = read_cfg(W / "Wrist_Sample_Data.cfg")
for n, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
    cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder], name=f"X-ray {n}")
    cam.build()
    tab.state.cameras.append(cam)
    tab.scene.add_camera(cam)
tab.refresh_all()
assert tab.cam_views_btn.isVisible() and not tab.cam_views.isVisible()
tab.set_camera_views(True)
pump(1.0)
assert tab.cam_views.isVisible() and len(tab.cam_views.views) == 2
tab.timeline.slider.setValue(20)
pump(1.0)
view = tab.cam_views.views[0]
# the silhouette of each bone lies on the image and inside where the bone projects
for m in tab.state.models:
    item = view.items[id(m)]
    assert item.isVisible(), m.name
    r = item.path().boundingRect()
    cal = view.cam.calibration
    centre = cal.project((m.tracking.pose(20) @ np.r_[np.array(m.mesh.center), 1.0])[:3])[0]
    assert r.contains(centre[0], centre[1]), (m.name, r, centre)
    assert 0 <= r.center().x() <= cal.width and 0 <= r.center().y() <= cal.height
shot("r8_camera_views_silhouette")
tab._set("overlay", "points")
pump(0.5)
assert type(view.items[id(rad)]).__name__ == "ScatterPlotItem"
shot("r8_camera_views_points")
# through-camera 3D views still work with the panels open
tab.set_xray_view(0)
pump(0.5)
assert tab.scene.view_mode == "xray:0" and tab.cam_views.isVisible()
shot("r8_through_camera_with_panels")
tab.set_view("3d")
tab._set("overlay", "silhouette")
# winding: a closed cube's projection must fill completely (front and back faces don't cancel)
cube = pv.Cube().triangulate()
uv = cube.points[:, :2] * 100 + 200
tri = np.asarray(cube.faces).reshape(-1, 4)[:, 1:]
path = silhouette_path(uv, tri)
from PySide6.QtCore import QPointF  # noqa: E402

assert path.contains(QPointF(200, 200)) and path.contains(QPointF(240, 160))
# sessions keep the layout
st = tab.state.to_dict()["settings"]
assert st["camera_views"] is True and st["overlay"] == "silhouette" and st["curve_labels"] == "axes"
assert tab.state.models[1].to_dict()["tracking_offset"] == 0
print("#8 OK", flush=True)

print("REQUESTS OK", flush=True)
