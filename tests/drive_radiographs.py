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
from PySide6.QtCore import QSettings, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog  # noqa: E402

import pyvista as pv  # noqa: E402

pv.global_theme.allow_empty_mesh = True
from bvrviewer.app.main_window import MainWindow  # noqa: E402
from bvrviewer.app.style import apply_theme  # noqa: E402
from bvrviewer.core import images as IM  # noqa: E402
from bvrviewer.core import radiograph_store as RS  # noqa: E402

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "cache", ignore_errors=True)
shutil.rmtree(OUT / "config", ignore_errors=True)
os.environ["BVR_CACHE_DIR"] = str(OUT / "cache")        # a cache and settings of its own
os.environ["XDG_CONFIG_HOME"] = str(OUT / "config")
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
settings = QSettings("BVR", "BVR Motion Viewer")
settings.setValue("radiographs/cache_mode", "always")


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

# the radiographs were kept in the local cache (in the background)
end = time.monotonic() + 60
while len(RS.cache_files()) < 2 and time.monotonic() < end:
    app.processEvents()
    time.sleep(0.1)
assert len(RS.cache_files()) == 2, RS.cache_files()
print(f"cached: {RS.cache_size() / 1e6:.0f} MB for {2 * N} images", flush=True)

# ---- 2. Skip: stop the up-front read; unread frames are read when shown
settings.setValue("radiographs/cache_mode", "never")
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

# ---- 3. the same trial again: from the local cache, with no reads of the originals
settings.setValue("radiographs/cache_mode", "always")
for c in list(cams):
    tab.state.cameras.remove(c)
    tab.scene.forget_camera(c)
time.sleep(0.5)                                     # reads already under way when they were removed
n_reads = len(reads)
t0 = time.monotonic()
tab.import_cfg_file(str(trial / "003_Config.cfg"), quiet=True)
took = time.monotonic() - t0
cams = tab.state.cameras
assert all(c.images.store_kind == "cache" and c.images.progress() == (N, 0, 0) for c in cams)
assert len(reads) - n_reads <= 2, len(reads) - n_reads          # only the first image, for its size
print(f"opened again from the cache in {took:.1f} s ({len(reads) - n_reads} reads of the originals)", flush=True)
tab.sel_kind, tab.sel_obj = "camera", cams[0]
tab.refresh_all()
app.processEvents()
win.grab().save(str(OUT / "radiographs_from_cache_props.png"))

# ---- 4. save the session with its radiographs and files; open it without the originals
from bvrviewer.app.dialogs import CacheDialog, SaveSessionDialog  # noqa: E402

sess = OUT / "save" / "Trial003.bvrsession.json"
shutil.rmtree(sess.parent, ignore_errors=True)
sess.parent.mkdir()
dlg = SaveSessionDialog(tab, tab.state, sess)
dlg.all_files.setChecked(True)
assert dlg.radiographs.isChecked()                  # ticking all files ticks the radiographs too
dlg.radiographs.setChecked(False)                   # ... which can still be unticked
assert dlg.all_files.isChecked() and not dlg.radiographs.isChecked()
dlg.radiographs.setChecked(True)
dlg.show()
app.processEvents()
dlg.grab().save(str(OUT / "save_session_dialog.png"))
dlg.close()
tab.save_session_file(str(sess), radiographs=True, all_files=True)
rep = tab.last_save
assert rep["radiographs"] == 2 and rep["files"] == 2, rep
print(f"saved session with {rep['radiographs']} cameras' radiographs and {rep['files']} files: "
      f"{rep['bytes'] / 1e6:.0f} MB in {rep['data_folder'].name}", flush=True)
for c in list(cams):
    tab.state.cameras.remove(c)
    tab.scene.forget_camera(c)
moved = trial.parent / "trial_moved"
shutil.rmtree(moved, ignore_errors=True)
trial.rename(moved)                                  # the originals are no longer there
RS.clear_cache()


def close_boxes():
    for w in app.topLevelWidgets():
        if isinstance(w, QMessageBox) and w.isVisible():
            seen["box"] = w.text()
            w.grab().save(str(OUT / "open_session_notes.png"))
            w.accept()
            return
    QTimer.singleShot(200, close_boxes)


QTimer.singleShot(300, close_boxes)
t2 = win.new_visualiser_tab()
n_reads = len(reads)
t0 = time.monotonic()
t2.load_session_file(str(sess))
took = time.monotonic() - t0
cams = t2.state.cameras
assert len(cams) == 2 and all(not c.images.originals_available for c in cams), [c.note for c in cams]
assert all(c.images.progress() == (N, 0, 0) for c in cams)
assert len(reads) == n_reads
assert "weren't found" in seen.get("box", ""), seen.get("box")
assert str(cams[0].calibration_path).startswith(str(sess.parent / "Trial003_data")), cams[0].calibration_path
t2.sel_kind, t2.sel_obj = "camera", cams[0]
t2.refresh_all()
t2.scene.update_frame(N // 3)
app.processEvents()
win.grab().save(str(OUT / "session_without_originals.png"))
print(f"opened the saved session without the originals in {took:.1f} s", flush=True)
moved.rename(trial)

dlg = CacheDialog(win)
dlg.show()
app.processEvents()
dlg.grab().save(str(OUT / "cache_dialog.png"))
dlg.close()
print("RADIOGRAPHS OK", flush=True)
os._exit(0)
