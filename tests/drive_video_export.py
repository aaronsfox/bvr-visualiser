"""Headless driver for video export (Visualiser #15) on the Autoscoper ankle sample.

Usage: BVR_SAMPLE_DATA=... xvfb-run -a -s "-screen 0 1700x1000x24" python tests/drive_video_export.py OUT_DIR

Loads the ankle jog (three bones, two cameras), shows the export dialog, then exports a short clip in
each layout and checks the videos' size and frame count. Saves a still of each layout.
"""

import os
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.app.video_export import (  # noqa: E402
    ExportOptions,
    FrameComposer,
    VideoExportDialog,
)
from bvrviewer.core.config import read_cfg  # noqa: E402
from bvrviewer.core.models import BoneModel  # noqa: E402
from bvrviewer.core.session import XrayCamera  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
A = Path(os.environ["BVR_SAMPLE_DATA"]) / "2025-01-12-Ankle"

app = QApplication.instance() or QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1500, 950)
win.show()


def pump(s=0.2):
    end = time.time() + s
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


tab = win.new_tab()
tab.state.name = "Ankle jog"
tab._build_models([BoneModel(name=b, path=A / "Meshes" / f"SOL001A_{b}_aligned.stl", color=c,
                             tracking_path=A / "Tracking" / f"T0019_jog0001_{b}.tra")
                   for b, c in [("tib", "#E8D9B5"), ("tal", "#4F9CF9"), ("cal", "#5CC8A8")]])
for _ in range(200):
    pump(0.05)
    if all(m.mesh is not None and m.tracking is not None for m in tab.state.models):
        break
cfg = read_cfg(A / "2025-01-12-Ankle.cfg")
for k, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
    cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder], name=f"X-ray {k}")
    cam.build()
    assert cam.show_source is False                 # Visualiser #14: marker off for new cameras
    tab.state.cameras.append(cam)
    tab.scene.add_camera(cam)
tab.refresh_all()
tab.set_camera_views(True)
tab.timeline.slider.setValue(10)
pump(1)
win.grab().save(str(OUT / "vid_00_tab.png"))

dlg = VideoExportDialog(tab, tab.scene.n_frames, 2)
dlg.show()
pump(0.3)
dlg.grab().save(str(OUT / "vid_01_dialog.png"))
o = dlg.options()
assert (o.content, o.layout, o.width, o.height, o.first) == ("both", "main", 1920, 1080, 0), o
dlg.close()

import imageio.v2 as imageio  # noqa: E402

n = tab.scene.n_frames
cases = [("3d", "row", 1280, 720), ("both", "row", 1920, 1080), ("both", "main", 1920, 1080),
         ("both", "grid", 1920, 1080), ("cams", "row", 1280, 720)]
for content, layout, w, h in cases:
    opts = ExportOptions(content, layout, w, h, first=5, last=14, fps=30)
    comp = FrameComposer(tab, opts)
    still = comp.frame(10)
    comp.close()
    assert still.shape == (h, w, 3) and still.std() > 5, (content, layout, still.shape)
    imageio.imwrite(OUT / f"vid_still_{content}_{layout}.png", still)
    path = OUT / f"vid_{content}_{layout}.mp4"
    assert tab.export_video(str(path), opts)
    r = imageio.get_reader(path)
    frames = [f for f in r]
    assert len(frames) == 10 and frames[0].shape == (h, w, 3), (len(frames), frames[0].shape)
    print("exported", path.name, len(frames), "frames", frames[0].shape, flush=True)
assert tab.scene.frame == 10                       # playback position restored
print("video export OK")
win.close()
