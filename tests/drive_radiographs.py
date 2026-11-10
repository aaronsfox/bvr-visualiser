"""Radiographs read up front with a progress bar (Visualiser #12), on a simulated slow network drive.

A trial of N frames per camera is made from one 2016 x 2016 radiograph per camera (links to the same
file) and every read is slowed down. The driver checks that:
- importing the .cfg shows the "Reading radiographs" progress bar and, once it's done, every image is in
  memory (playback and jumping need no more reads);
- Skip stops the up-front read, and a frame that hasn't been read yet is then read straight away.

Usage: xvfb-run -a -s "-screen 0 1600x1000x24" python tests/drive_radiographs.py TRIAL_FOLDER OUTDIR [N]
where TRIAL_FOLDER holds 003_Config.cfg, Calibration/ and RadiographImages/Camera 1|2/ (one image each).
"""

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QProgressDialog  # noqa: E402

import pyvista as pv  # noqa: E402

pv.global_theme.allow_empty_mesh = True
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core import images as IM  # noqa: E402

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
OUT.mkdir(parents=True, exist_ok=True)
DELAY = 0.06            # s per 4 MB read: ~65 MB/s per thread, like a network drive

trial = OUT / "trial"
shutil.rmtree(trial, ignore_errors=True)
shutil.copytree(SRC / "Calibration", trial / "Calibration")
for cam in ("Camera 1", "Camera 2"):
    src = next((SRC / "RadiographImages" / cam).glob("*.tif"))
    d = trial / "RadiographImages" / cam
    d.mkdir(parents=True)
    stem = src.name.rsplit(".", 2)[0]
    for i in range(1, N + 1):
        os.link(src, d / f"{stem}.{i:04d}.tif")
shutil.copy(SRC / "003_Config.cfg", trial / "003_Config.cfg")

_read = IM._read_any
reads = []


def slow_read(path, page=None):
    time.sleep(DELAY)
    reads.append(path)
    return _read(path, page)


IM._read_any = slow_read

app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1500, 920)
win.show()
tab = win.new_visualiser_tab()


def find_dialog():
    return next((w for w in app.topLevelWidgets() if isinstance(w, QProgressDialog) and w.isVisible()), None)


# ---- 1. read everything up front, with a progress bar
seen = {}


def grab():
    d = find_dialog()
    if d is not None:
        seen["value"], seen["max"], seen["text"] = d.value(), d.maximum(), d.labelText()
        win.grab().save(str(OUT / "radiographs_progress_window.png"))
        d.grab().save(str(OUT / "radiographs_progress_dialog.png"))
    else:
        QTimer.singleShot(200, grab)


QTimer.singleShot(2500, grab)
t0 = time.monotonic()
tab.import_cfg_file(str(trial / "003_Config.cfg"), quiet=True)
took = time.monotonic() - t0
cams = tab.state.cameras
assert len(cams) == 2, cams
assert seen, "no progress bar was shown"
assert seen["max"] == 2 * N and 0 < seen["value"] < 2 * N, seen
assert "MB/s" in seen["text"] and "X-ray 1" in seen["text"], seen["text"]
for c in cams:
    assert c.images.progress() == (N, 0, 0), c.images.progress()
print(f"read {2 * N} images up front in {took:.1f} s; dialog said:\n{seen['text']}", flush=True)

n_reads = len(reads)
assert all(c.images.has_display(N - 1) for c in cams)
t0 = time.monotonic()
for f in list(range(0, N, 7)) + [N - 1, 0, N // 2]:      # jumping and playing need no more reads
    tab.scene.update_frame(f)      # no bones here, so the timeline has one frame
    app.processEvents()
assert len(reads) == n_reads, len(reads) - n_reads
print(f"jumped through {len(range(0, N, 7)) + 3} frames in {time.monotonic() - t0:.2f} s with no reads", flush=True)
tab.scene.update_frame(N // 2)
app.processEvents()
win.grab().save(str(OUT / "radiographs_loaded_window.png"))

# ---- 2. Skip: stop the up-front read; unread frames are read when shown
for c in list(cams):
    tab.state.cameras.remove(c)
    tab.scene.forget_camera(c)


def skip():
    d = find_dialog()
    if d is not None:
        d.cancel()
    else:
        QTimer.singleShot(100, skip)


QTimer.singleShot(1500, skip)
t0 = time.monotonic()
tab.import_cfg_file(str(trial / "003_Config.cfg"), quiet=True)
took = time.monotonic() - t0
assert took < 5, took
cams = tab.state.cameras
time.sleep(1)
read = [c.images.progress()[0] for c in cams]
assert all(r < N for r in read), read
t0 = time.monotonic()
tab.scene.update_frame(N - 2)
app.processEvents()
assert time.monotonic() - t0 < 1.0, time.monotonic() - t0
assert all(c.images.has_display(N - 2) for c in cams)
print(f"skipped after {took:.1f} s ({read} images read); an unread frame showed in "
      f"{time.monotonic() - t0:.2f} s", flush=True)
print("RADIOGRAPHS OK", flush=True)
os._exit(0)
