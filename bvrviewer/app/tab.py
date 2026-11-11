"""The Visualiser tab: navigator + properties on the left, the 3D view and timeline on the right."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ..core import joints as J
from ..core.config import read_cfg
from ..core.models import PALETTE, BoneModel, ModelError, is_volume_file
from ..core.session import (
    TAB_WIDE,
    KinItem,
    TabState,
    Trial,
    XrayCamera,
    load_session,
    save_session,
)
from ..core.tracking import TrackingError, load_tracking
from ..core.transforms import TransformFileError
from .camera_views import OVERLAYS, CameraViews
from .dialogs import AddCameraDialog, FormatDialog, VoxelDialog
from .joint_panel import (
    JointCurves,
    SegmentMapDialog,
    curve_labels,
    guess_segment_map,
    write_csv,
    zeroed,
)
from .model_builder import add_triad
from .navigator import (
    ADD_TRIAL,
    AXES,
    CAMERA,
    CAMERAS,
    JOINT,
    JOINTS,
    MODEL,
    MODELS,
    SCENE,
    TRACKING,
    TRIAL,
    TRIALS,
    Navigator,
    PropertySheet,
)
from .panels import Timeline, h2, label
from .scene import SceneController

MODEL_FILTER = "Bone models (*.stl *.STL *.tif *.tiff *.TIF *.obj *.ply *.vtk *.vtp);;All files (*)"


class _Worker(QObject):
    done = Signal(object, object)       # result, error text
    progress = Signal(int, str)         # step done, what it's doing (for _run_steps)

    def __init__(self, fn, steps: bool = False):
        super().__init__()
        self.fn = fn
        self.steps = steps
        self.cancel = threading.Event()

    def run(self):
        try:
            if self.steps:
                self.done.emit(self.fn(self.progress.emit, self.cancel.is_set), None)
            else:
                self.done.emit(self.fn(), None)
        except Exception as e:  # noqa: BLE001 - shown to the user
            msg = str(e) if isinstance(e, (ModelError, TrackingError, TransformFileError, ValueError, OSError)) \
                else f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}"
            self.done.emit(None, msg)


class _ProgressRelay(QObject):
    """Brings a worker's progress reports onto the UI thread."""

    def __init__(self, fn, parent):
        super().__init__(parent)
        self.fn = fn

    @Slot(int, str)
    def deliver(self, step, text):
        self.fn(step, text)


class _Relay(QObject):
    """Lives on the UI thread so worker results are handled there (VTK/OpenGL must stay on the UI thread)."""

    def __init__(self, fn, parent):
        super().__init__(parent)
        self.fn = fn

    @Slot(object, object)
    def deliver(self, result, err):
        self.fn(result, err)


class EmptyState(QFrame):
    importTrial = Signal()
    emptyTrial = Signal()
    openSession = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Empty")
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        box = QWidget()
        box.setMaximumWidth(580)
        v = QVBoxLayout(box)
        v.setSpacing(18)
        title = QLabel("Add a trial to start")
        title.setObjectName("Big")
        title.setWordWrap(True)
        v.addWidget(title)
        v.addWidget(label("A trial holds its bones (with their tracking), X-ray cameras and joints. A tab can hold "
                          "several trials, listed in the navigator on the left; one is shown at a time."))
        for n, head, body in [
            (1, "Import a trial", "Choose the trial's Autoscoper .cfg (its bones and X-ray cameras) and the folder "
                                  "with its tracking files. You then check each bone's tracking file and colour, and "
                                  "the cameras, before the trial is built."),
            (2, "Or add an empty trial", "Give it a name, then add bones, tracking and cameras yourself from the "
                                         "navigator (right-click Bones or X-ray cameras)."),
            (3, "Add more trials", "Use + Add trial… in the navigator. Double-click a trial to show it."),
        ]:
            r = QHBoxLayout()
            num = QLabel(str(n))
            num.setObjectName("Step" if n == 1 else "StepOff")
            num.setFixedSize(28, 28)
            num.setAlignment(Qt.AlignCenter)
            r.addWidget(num, 0, Qt.AlignTop)
            col = QVBoxLayout()
            h = QLabel(head)
            h.setStyleSheet("font-weight: 600; font-size: 14px;")
            col.addWidget(h)
            col.addWidget(label(body))
            r.addLayout(col, 1)
            v.addLayout(r)
        r = QHBoxLayout()
        imp = QPushButton("+  Import trial…")
        imp.setObjectName("Primary")
        imp.setMinimumHeight(40)
        imp.setToolTip("From an Autoscoper .cfg and its tracking files")
        imp.clicked.connect(self.importTrial.emit)
        emp = QPushButton("Empty trial…")
        emp.setMinimumHeight(40)
        emp.setToolTip("A trial with a name only: add its bones and cameras yourself")
        emp.clicked.connect(self.emptyTrial.emit)
        op = QPushButton("Open saved session…")
        op.setMinimumHeight(40)
        op.clicked.connect(self.openSession.emit)
        r.addWidget(imp)
        r.addWidget(emp)
        r.addWidget(op)
        r.addStretch(1)
        v.addLayout(r)
        self.trial_note = label("")
        self.trial_note.setVisible(False)
        v.addWidget(self.trial_note)
        outer.addWidget(box)

    def set_trial(self, name: str | None) -> None:
        """Note when the shown trial exists but has no bones yet."""
        self.trial_note.setVisible(bool(name))
        if name:
            self.trial_note.setText(f"“{name}” has no bones yet: right-click Bones under it in the navigator to add "
                                    "one, or use Configure… on the trial.")


PCT = (100, 0, "%")     # slider value box for 0–1 fractions shown as percentages


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class VisualiserTab(QWidget):
    """One trial: its models, tracking, radiographs, 3D view and timeline."""

    kind = "visualiser"
    titleChanged = Signal(str)

    def __init__(self, state: TabState | None = None):
        super().__init__()
        self.state = state or TabState()
        self.cfg = None                     # last .cfg used, for voxel sizes
        self._threads: list = []
        self._busy: QProgressDialog | None = None
        self.selected: BoneModel | None = None          # model used for trail / coverage
        self.sel_kind: str = MODELS                     # what the navigator has selected
        self.sel_obj = None
        self.curve_joint = None                         # model (its coordinate system) or joint whose curves show
        self.show_curves = True
        self._kin: dict = {}                            # id(item) -> kinematics (before zeroing at a frame)
        self._joint_actors: dict = {}                   # id(item) -> [(actor list, model, acs)]
        self._kin_models: dict = {}                     # model file path -> KinematicModel
        self.tracking_format: dict | None = None        # tracking options remembered for this tab (not saved)
        self._scene_trial: Trial | None = None          # the trial whose bones and cameras are in the 3D scene

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)

        # ---- left: navigator over properties
        left = QSplitter(Qt.Vertical)
        left.setHandleWidth(1)
        nav_box = QWidget()
        nav_box.setObjectName("Panel")
        nb = QVBoxLayout(nav_box)
        nb.setContentsMargins(0, 0, 0, 0)
        nb.setSpacing(0)
        head = QWidget()
        head.setObjectName("PropsHead")
        head.setAttribute(Qt.WA_StyledBackground, True)
        hh = QHBoxLayout(head)
        hh.setContentsMargins(14, 10, 10, 8)
        hh.addWidget(h2("Navigator"), 1)
        hh.addWidget(label("right-click for actions", wrap=False))
        nb.addWidget(head)
        self.nav = Navigator()
        nb.addWidget(self.nav, 1)
        left.addWidget(nav_box)
        self.props = PropertySheet()
        left.addWidget(self.props)
        left.setSizes([330, 520])
        left.setMinimumWidth(340)
        split.addWidget(left)

        # ---- right: view buttons, 3D view, info strip, timeline
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        self.view_bar = QWidget()
        self.view_bar.setObjectName("Panel")
        vb = QHBoxLayout(self.view_bar)
        vb.setContentsMargins(10, 6, 10, 6)
        vb.addWidget(label("View", wrap=False))
        vb.addStretch(1)
        self.view_buttons: dict[str, QPushButton] = {}
        for key, text in [("3d", "Free 3D"), ("xy", "XY plane"), ("yz", "YZ plane"), ("xz", "XZ plane")]:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setObjectName("Small")
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            self.view_buttons[key] = b
            vb.addWidget(b)
        self.flip_btn = QPushButton("Flip ↔")
        self.flip_btn.setObjectName("Small")
        self.flip_btn.setToolTip("Look from the opposite side")
        self.flip_btn.clicked.connect(self._flip)
        vb.addWidget(self.flip_btn)
        self.xray_btn_box = QHBoxLayout()
        vb.addLayout(self.xray_btn_box)
        self.cam_views_btn = QPushButton("Camera views")
        self.cam_views_btn.setObjectName("Small")
        self.cam_views_btn.setCheckable(True)
        self.cam_views_btn.setToolTip("Show each X-ray camera's radiograph beside the 3D view, with the bones "
                                      "drawn over it in their colours")
        self.cam_views_btn.clicked.connect(lambda on: self.set_camera_views(on))
        vb.addWidget(self.cam_views_btn)
        cv.addWidget(self.view_bar)

        self.stack = QStackedWidget()
        self.empty = EmptyState()
        self.empty.importTrial.connect(lambda: self.import_trial())
        self.empty.emptyTrial.connect(lambda: self.add_empty_trial())
        self.empty.openSession.connect(self.open_session)
        self.stack.addWidget(self.empty)
        self.plotter = QtInteractor(self.stack)
        self.stack.addWidget(self.plotter.interactor)
        self.view_split = QSplitter(Qt.Horizontal)       # 3D view | 2D camera panels
        self.view_split.setChildrenCollapsible(False)
        self.view_split.addWidget(self.stack)
        cv.addWidget(self.view_split, 1)
        self.hint = QWidget()
        self.hint.setObjectName("Panel")
        hb = QHBoxLayout(self.hint)
        hb.setContentsMargins(10, 4, 10, 4)
        self.frame_info = label("", "Mono", wrap=False)
        hb.addWidget(self.frame_info, 1)
        self.status_msg = label("", "Ok", wrap=False)
        hb.addWidget(self.status_msg)
        hb.addWidget(label("Drag: rotate · Shift-drag: pan · Scroll: zoom · Right-click: select bone", wrap=False))
        cv.addWidget(self.hint)
        self.curves = JointCurves()
        self.curves.setVisible(False)
        cv.addWidget(self.curves)
        self.timeline = Timeline(self.state.settings)
        cv.addWidget(self.timeline)
        split.addWidget(center)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1120])
        root.addWidget(split)

        self.scene = SceneController(self.plotter, self.state)
        self.cam_views = CameraViews(self.scene)
        self.cam_views.setVisible(False)
        self.view_split.addWidget(self.cam_views)
        self.view_split.setStretchFactor(0, 3)
        self.view_split.setStretchFactor(1, 2)
        self.scene.on_frame = self.cam_views.update_frame
        self.plotter.enable_mesh_picking(self._picked, use_actor=True, show=False, show_message=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timeline.play.toggled.connect(self._play)
        self.timeline.frameChanged.connect(self._frame_changed)
        self.curves.frameClicked.connect(lambda f: self.timeline.slider.setValue(f))
        self.curves.set_style(self.state.settings.curve_labels)
        self.curves.labelStyleChanged.connect(self._curve_style_changed)
        self.nav.itemSelected.connect(self._nav_selected)
        self.nav.menuRequested.connect(self._show_menu)
        self.nav.visibilityToggled.connect(self._toggle_visible)
        self.nav.axesToggled.connect(self._toggle_axes)
        self.nav.renamed.connect(self._rename)
        self.nav.activated.connect(self._nav_activated)
        for key, fn in [(Qt.Key_Space, lambda: self.timeline.play.toggle()),
                        (Qt.Key_Right, lambda: self.timeline.step(1)), (Qt.Key_Left, lambda: self.timeline.step(-1))]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._negative = False
        self.refresh_all()

    # ---------------------------------------------------------------- helpers
    def _read_radiographs(self, cams, title: str | None = None) -> None:
        """Read every radiograph of these cameras up front, with a progress bar.

        Reading while playing is too slow when the images are large or on a network drive, so they are
        all read (and shrunk for display) before the cameras are shown. Skip leaves the rest to be read
        as each frame is shown.
        """
        import time

        from ..core import radiograph_store as RS
        from .dialogs import cache_settings

        cams = [c for c in cams if c.images is not None]
        for c in cams:
            c._read_once = True                     # (a trial's cameras are read the first time it's shown)
        mode, limit = cache_settings()
        for c in cams:                              # a copy in the local cache, if the originals are unchanged
            im = c.images
            if mode != "never" and im.store_kind is None and im.originals_available:
                try:
                    p = RS.find_cached(im)
                    if p is not None:
                        im.attach_store(p, "cache")
                except Exception:  # noqa: BLE001 - a broken cache never stops the originals being read
                    pass
        t_start = time.monotonic()
        for c in cams:
            c.images.preload(c.first_image)
        total = sum(len(c.images) - c.first_image for c in cams)

        def counts():
            return [c.images.progress() for c in cams]

        if not total or all(p[2] == 0 for p in counts()):
            self._cache_radiographs(cams, time.monotonic() - t_start)
            return
        dlg = QProgressDialog("Reading radiographs…", "Skip", 0, total, self)
        dlg.setWindowTitle(title or "Reading radiographs")
        text = QLabel()
        text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        dlg.setLabel(text)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setMinimumWidth(460)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        t0, b0 = time.monotonic(), sum(c.images.bytes_read for c in cams)
        while True:
            ps = counts()
            done = sum(min(r + f, len(c.images) - c.first_image) for c, (r, f, _) in zip(cams, ps))
            left = sum(p[2] for p in ps)
            if left == 0 or dlg.wasCanceled():
                break
            dt = time.monotonic() - t0
            mb = (sum(c.images.bytes_read for c in cams) - b0) / 1e6
            lines = [f"{c.name}: {r} of {len(c.images) - c.first_image} images" + (f" ({f} unreadable)" if f else "")
                     + (f", from {c.images.where_from()}" if c.images.store_kind else "")
                     for c, (r, f, _) in zip(cams, ps)]
            if dt > 1 and mb > 0:
                rate = mb / dt
                eta = left * (mb / max(done, 1)) / rate
                lines.append(f"{rate:.0f} MB/s · about {eta:.0f} s left" if eta < 90 else
                             f"{rate:.0f} MB/s · about {eta / 60:.0f} min left")
            lines.append("\nSkip to start now: the rest are then read as each frame is shown.")
            dlg.setLabelText("Reading the radiographs into memory so they play smoothly…\n\n" + "\n".join(lines))
            dlg.setValue(done)
            QApplication.processEvents()
            QThread.msleep(40)
        if dlg.wasCanceled():
            for c in cams:
                c.images.cancel_preload()
        dlg.close()
        self._cache_radiographs(cams, time.monotonic() - t_start)
        bad = [(c, len(c.images.failed)) for c in cams if c.images.failed]
        if bad:
            first = bad[0][0].images
            i, err = next(iter(first.failed.items()))
            self._error("Some radiographs couldn't be read",
                        "\n".join(f"{c.name}: {n} image(s)" for c, n in bad)
                        + f"\n\nFor example {first.name(i)}: {err}\nThey'll be tried again when shown.")

    def _cache_radiographs(self, cams, elapsed: float) -> None:
        """Keep a copy of fully read radiographs in the local cache (in the background), if they were slow
        to read (or always / never, as set in Radiographs → Radiograph cache)."""
        import threading

        from ..core import radiograph_store as RS
        from .dialogs import cache_settings

        mode, limit = cache_settings()
        if mode == "never":
            return
        for c in cams:
            im = c.images
            if im.store_kind is not None or not im.originals_available or im.failed or not im.fits_in_memory:
                continue
            if im.progress()[0] < len(im):
                continue                                    # skipped: not all in memory
            if mode == "slow" and not (elapsed > 5 or RS.is_network_path(im.source)):
                continue

            def work(im=im):
                try:
                    RS.save_to_cache(im, limit, cancelled=lambda: im._closed)
                except Exception:  # noqa: BLE001 - the cache is only a convenience
                    pass
            threading.Thread(target=work, name="radiograph-cache", daemon=True).start()

    def _with_progress(self, title: str, fn) -> object:
        """Run fn(progress, cancelled) on a thread with a modal progress dialog; returns its result or
        raises its exception. progress(text, done, total) updates the dialog."""
        import threading

        st = {"text": title + "…", "done": 0, "total": 0, "result": None, "error": None, "finished": False}
        stop = threading.Event()

        def progress(text, done, total):
            st.update(text=text, done=done, total=total)

        def run():
            try:
                st["result"] = fn(progress, stop.is_set)
            except BaseException as e:  # noqa: BLE001 - re-raised on the UI thread
                st["error"] = e
            st["finished"] = True

        th = threading.Thread(target=run, daemon=True)
        th.start()
        dlg = QProgressDialog(title + "…", "Cancel", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setMinimumWidth(420)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        while not st["finished"]:
            if dlg.wasCanceled():
                stop.set()
            dlg.setLabelText(st["text"])
            if st["total"]:
                dlg.setMaximum(st["total"])
                dlg.setValue(min(st["done"], st["total"]))
            QApplication.processEvents()
            QThread.msleep(40)
        dlg.close()
        if st["error"] is not None:
            raise st["error"]
        return st["result"]

    def _run(self, text: str, fn, on_done) -> None:
        """Run slow loading off the UI thread with a busy dialog."""
        self._busy = QProgressDialog(text, None, 0, 0, self)
        self._busy.setWindowTitle("Loading")
        self._busy.setWindowModality(Qt.WindowModal)
        self._busy.setMinimumDuration(300)
        thread = QThread(self)
        worker = _Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def finished(result, err):
            if self._busy:
                self._busy.close()
                self._busy = None
            thread.quit()
            thread.wait()
            entry = (thread, worker, relay)
            if entry in self._threads:
                self._threads.remove(entry)
            on_done(result, err)

        relay = _Relay(finished, self)
        worker.done.connect(relay.deliver, Qt.QueuedConnection)
        self._threads.append((thread, worker, relay))
        thread.start()

    def _run_steps(self, title: str, total: int, fn, on_done) -> None:
        """Like _run, with a progress bar and Cancel (Visualiser #20).

        ``fn(report, cancelled)`` runs on a worker thread: ``report(step, text)`` moves the bar to ``step`` of
        ``total`` and says what it's doing; ``cancelled()`` is True once Cancel is pressed (stop at the next
        convenient point). ``on_done(result, error)`` runs on the UI thread.
        """
        dlg = QProgressDialog(title, "Cancel", 0, max(total, 1), self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setMinimumWidth(460)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        self._busy = dlg
        thread = QThread(self)
        worker = _Worker(fn, steps=True)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def progress(step, text):
            if self._busy is dlg:
                dlg.setValue(min(step, dlg.maximum()))
                dlg.setLabelText(text)

        def cancel():
            worker.cancel.set()
            dlg.setLabelText("Cancelling after the current step…")

        dlg.canceled.connect(cancel)

        def finished(result, err):
            if self._busy is dlg:
                self._busy = None
            dlg.canceled.disconnect(cancel)          # (closing the dialog would emit it)
            dlg.close()
            thread.quit()
            thread.wait()
            entry = (thread, worker, relay)
            if entry in self._threads:
                self._threads.remove(entry)
            on_done(result, err)

        relay = _Relay(finished, self)
        prog = _ProgressRelay(progress, self)
        worker.progress.connect(prog.deliver, Qt.QueuedConnection)
        worker.done.connect(relay.deliver, Qt.QueuedConnection)
        self._threads.append((thread, worker, relay))
        self._progress_relays = getattr(self, "_progress_relays", []) + [prog]
        thread.start()

    def _status(self, text: str, seconds: float = 5.0) -> None:
        """A short message in the bar under the 3D view that clears itself."""
        self.status_msg.setText(text)
        QTimer.singleShot(int(seconds * 1000), lambda: self.status_msg.text() == text and self.status_msg.setText(""))

    def _error(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _preview_bones(self):
        out = []
        for m in self.state.models:
            if m.tracking is None or m.mesh is None:
                continue
            T = m.tracking.pose(0, "hold")
            if T is None:
                continue
            pts = np.asarray(m.mesh.points)[:: max(1, m.mesh.n_points // 3000)]
            out.append(((np.c_[pts, np.ones(len(pts))] @ T.T)[:, :3], m.color))
        return out

    # ---------------------------------------------------------------- refresh
    def refresh_all(self) -> None:
        """Bring the navigator, properties, timeline and view buttons in line with the state."""
        if self.timeline.s is not self.state.settings:      # e.g. "Trial 1" made by adding bones to an empty tab
            self._use_trial_settings()
        has = bool(self.state.models)
        self.stack.setCurrentIndex(1 if has else 0)
        self.empty.set_trial(self.state.trial.name if self.state.trial is not None and not has else None)
        self.view_bar.setVisible(has)
        self.hint.setVisible(has)
        self.timeline.set_count(self.scene.n_frames)
        # (identity checks: models and cameras are dataclasses, so `in` would compare their fields)
        if self.sel_kind in (MODEL, TRACKING, AXES) and not any(m is self.sel_obj for m in self.state.models):
            self.sel_kind, self.sel_obj = MODELS, None
        if self.sel_kind == CAMERA and not any(c is self.sel_obj for c in self.state.cameras):
            self.sel_kind, self.sel_obj = CAMERAS, None
        if self.sel_kind == JOINT and not any(j is self.sel_obj for j in self.state.joints):
            self.sel_kind, self.sel_obj = JOINTS, None
        if self.sel_kind == TRIAL and not any(t is self.sel_obj for t in self.state.trials):
            self.sel_kind, self.sel_obj = TRIALS, None
        if self.sel_kind in (MODELS, CAMERAS, JOINTS) and self.sel_obj is not self.state.trial:
            self.sel_kind, self.sel_obj = (TRIALS, None) if self.state.trial is None else (self.sel_kind, None)
        self.refresh_joints()
        self.nav.rebuild(self.state, self.sel_kind, self.sel_obj)
        self.show_properties()
        self._refresh_coverage()
        self._refresh_xray_buttons()
        self._update_frame_info()

    def _refresh_coverage(self) -> None:
        m = self.selected
        self.timeline.coverage.set_data(m.tracking.valid if m and m.tracking else None, self.scene.n_frames)

    def _refresh_xray_buttons(self) -> None:
        while self.xray_btn_box.count():
            w = self.xray_btn_box.takeAt(0).widget()
            if w:
                w.setParent(None)
        for i, cam in enumerate(self.state.cameras):
            b = QPushButton(f"Through {cam.name}")
            b.setObjectName("Small")
            b.setCheckable(True)
            b.setChecked(self.scene.view_mode == f"xray:{i}")
            b.clicked.connect(lambda _=False, k=i: self.set_xray_view(k))
            self.xray_btn_box.addWidget(b)
        self._apply_camera_views()

    def set_camera_views(self, on: bool) -> None:
        """Show or hide the 2D camera panels beside the 3D view (the through-camera 3D views stay available)."""
        self.state.settings.camera_views = bool(on)
        self._apply_camera_views()
        if self.sel_kind == CAMERAS:
            self.show_properties()

    def _apply_camera_views(self) -> None:
        has = bool(self.state.cameras) and bool(self.state.models)
        on = has and self.state.settings.camera_views
        self.cam_views_btn.setVisible(bool(self.state.cameras))
        self.cam_views_btn.setChecked(on)
        was = self.cam_views.isVisible()
        self.cam_views.setVisible(on)
        if on:
            if not was:
                total = max(self.view_split.width(), 600)
                self.view_split.setSizes([int(total * 0.6), int(total * 0.4)])
            self.cam_views.update_frame(self.scene.frame)

    def _update_frame_info(self) -> None:
        f = self.scene.frame
        txt = f"Frame {f + 1} / {self.scene.n_frames}"
        hs = self.state.settings.hold_still
        txt += f"   ·   Held still: {hs}" if hs else "   ·   Autoscoper world frame"
        for cam in self.state.cameras:
            i = cam.image_index(f)
            if i is not None and cam.visible:
                txt += f"   ·   {cam.name}: {cam.images.label(i)}"
        self.frame_info.setText(txt)

    # ---------------------------------------------------------------- navigator
    def _trial_for(self, kind: str, obj) -> Trial | None:
        """The trial a navigator item belongs to (None for Trials, Add trial and Scene)."""
        if kind in (MODELS, CAMERAS, JOINTS, TRIAL):
            return obj
        if kind in (MODEL, TRACKING, AXES, CAMERA, JOINT):
            return self.state.trial_of(obj)
        return None

    def _nav_activated(self, kind: str, obj) -> None:
        if kind == TRIAL:
            self.show_trial(obj)
        elif kind == CAMERA:
            if self.state.trial_of(obj) is not self.state.trial:
                self.show_trial(self.state.trial_of(obj), select=(CAMERA, obj))
            self.set_xray_view(self.state.cameras.index(obj))

    def _nav_selected(self, kind: str, obj) -> None:
        t = self._trial_for(kind, obj)
        if kind != TRIAL and t is not None and t is not self.state.trial:
            # an item of a trial that isn't shown: show that trial, then select the item
            QTimer.singleShot(0, lambda: self.show_trial(t, select=(kind, obj)))
            return
        if kind in (MODELS, CAMERAS, JOINTS):
            obj = None                              # the shown trial's section
        self.sel_kind, self.sel_obj = kind, obj
        if kind in (MODEL, TRACKING) and obj is not self.selected:
            self.select(obj, from_nav=True)
        if kind == JOINT or (kind == AXES and obj is not None and obj.has_axes):
            if obj is not self.curve_joint:
                self.curve_joint = obj
            self._update_curves()
        self.show_properties()

    def _toggle_visible(self, obj, on: bool) -> None:
        obj.visible = on
        if isinstance(obj, BoneModel):
            self._appearance(obj)
            self.nav.refresh_obj(MODEL, obj)
        elif isinstance(obj, KinItem):
            self.nav.refresh_obj(JOINT, obj)
            self._draw_joint_axes()
        else:
            self.nav.refresh_obj(CAMERA, obj)
            self.scene.update_frame(self.scene.frame)
            self._update_frame_info()
        if obj is self.sel_obj:
            self.show_properties()

    def _rename(self, obj, name: str) -> None:
        if isinstance(obj, Trial):
            name = name.strip()
            if name and name != obj.name:
                others = TabState(trials=[t for t in self.state.trials if t is not obj])
                obj.name = others.unique_trial_name(name)
            self.nav.refresh_obj(TRIAL, obj)
            if obj is self.sel_obj:
                self.props.title.setText(obj.name)
            return
        if isinstance(obj, KinItem):
            obj.label = name
            self.nav.refresh_obj(JOINT, obj)
            self._update_curves()
            if obj is self.sel_obj:
                self.props.title.setText(name)
            return
        if name == obj.name:
            return
        old, obj.name = obj.name, name
        if isinstance(obj, BoneModel):
            if self.state.settings.hold_still == old:
                self.state.settings.hold_still = name
            for jl in self.state.joints:              # segments and joints refer to models by name
                for r, n in list(jl.model_map.items()):
                    if n == old:
                        jl.model_map[r] = name
            self.nav.refresh_obj(MODEL, obj)

        else:
            self.nav.refresh_obj(CAMERA, obj)
            self._refresh_xray_buttons()
        self._update_frame_info()
        if obj is self.sel_obj:
            self.props.title.setText(name)

    def _show_menu(self, kind: str, obj, pos) -> None:
        t = self._trial_for(kind, obj)
        if kind != TRIAL and t is not None and t is not self.state.trial:
            self.show_trial(t, select=(kind, obj))
        if kind in (MODELS, CAMERAS, JOINTS):
            obj = None
        menu = self.build_menu(kind, obj)
        if not menu.isEmpty():
            menu.exec(pos)

    def build_menu(self, kind: str, obj) -> QMenu:
        """The right-click menu for a navigator level or item."""
        menu = QMenu(self)

        def add(text, fn, enabled=True, checked=None):
            a = QAction(text, menu)
            a.triggered.connect(lambda _=False: fn())
            a.setEnabled(enabled)
            if checked is not None:
                a.setCheckable(True)
                a.setChecked(checked)
            menu.addAction(a)
            return a

        if kind in (TRIALS, ADD_TRIAL):
            add("Import trial…", lambda: self.import_trial())
            add("Empty trial…", lambda: self.add_empty_trial())
        elif kind == TRIAL:
            t: Trial = obj
            add("Show this trial", lambda: self.show_trial(t), enabled=t is not self.state.trial)
            add("Configure…", lambda: self.configure_trial(t))
            add("Duplicate for another trial…", lambda: self.duplicate_trial(t))
            menu.addSeparator()
            add("Rename", lambda: self.nav.start_rename(TRIAL, t))
            add("Remove trial", lambda: self.remove_trial(t))
        elif kind == MODELS:
            add("Add bone…", self.add_models)
            add("Load model (bones, coordinate systems, joints)…", self.load_full_model)
            menu.addSeparator()
            add("Show all", lambda: self._show_all(True), enabled=bool(self.state.models))
            add("Hide all", lambda: self._show_all(False), enabled=bool(self.state.models))
        elif kind == TRACKING:
            m: BoneModel = obj
            if m.tracking_path is None:
                add("Load file…", lambda: self.attach_tracking(m))
            else:
                add("Load new file…", lambda: self.attach_tracking(m))
                add("Refresh current file", lambda: self.refresh_tracking(m))
                menu.addSeparator()
                add("Tracking format and frames…", lambda: self.edit_format(m), enabled=m.tracking is not None)
                add("Hold this bone still", lambda: self.hold_still(m, self.state.settings.hold_still != m.name),
                    enabled=m.tracking is not None, checked=self.state.settings.hold_still == m.name)
                menu.addSeparator()
                add("Remove tracking", lambda: self.detach_tracking(m))
        elif kind == MODEL:
            m: BoneModel = obj
            add("Load new tracking file…" if m.tracking_path else "Load tracking file…", lambda: self.attach_tracking(m))
            add("Load new coordinate system…" if m.has_axes else "Load coordinate system…", lambda: self.add_axes(m))
            menu.addSeparator()
            add("Hold this bone still", lambda: self.hold_still(m, self.state.settings.hold_still != m.name),
                enabled=m.tracking is not None, checked=self.state.settings.hold_still == m.name)
            add("Hide" if m.visible else "Show", lambda: self._toggle_visible(m, not m.visible))
            add("Rename", lambda: self.nav.start_rename(MODEL, m))
            add("Use file name as name", lambda: self._rename(m, m.path.stem))
            menu.addSeparator()
            add("Replace surface file…", lambda: self.replace_model(m))
            if m.is_volume:
                add("Voxel size / surface…", lambda: self.edit_voxel(m))
            elif m.tfm_path:
                add("Remove .tfm transform", lambda: self.clear_tfm(m))
            else:
                add("Apply .tfm transform…", lambda: self.load_tfm(m))
            menu.addSeparator()
            add("Remove bone", lambda: self.remove_model(m))
        elif kind == AXES:
            m: BoneModel = obj
            if not m.has_axes:
                add("Load file…", lambda: self.add_axes(m))
            else:
                add("Load new file…", lambda: self.add_axes(m))
                add("Refresh current file", lambda: self.refresh_axes(m))
                menu.addSeparator()
                add("Show curves for this bone", lambda: self._set_curve_joint(m))
                add("Hide axes" if m.axes_visible else "Show axes", lambda: self._toggle_axes(m, not m.axes_visible))
                add("Zero at the current frame", lambda: self._set_reference(m, self.scene.frame))
                add("Clear zero", lambda: self._set_reference(m, None), enabled=m.axes_zero is not None)
                add("Export values (CSV)…", lambda: self.export_joint_csv(m))
                menu.addSeparator()
                add("Remove coordinate system", lambda: self.remove_axes(m))
        elif kind == CAMERAS:
            add("Add X-ray camera…", self.add_camera)
            add("Import cameras from .cfg…", self.import_cfg)
        elif kind == CAMERA:
            cam: XrayCamera = obj
            i = self.state.cameras.index(cam)
            add("View through this camera", lambda: self.set_xray_view(i))
            add("Hide" if cam.visible else "Show", lambda: self._toggle_visible(cam, not cam.visible))
            add("Rename", lambda: self.nav.start_rename(CAMERA, cam))
            add("Change calibration or images…", lambda: self.edit_camera(i))
            menu.addSeparator()
            add("Remove camera", lambda: self.remove_camera(i))
        elif kind == JOINTS:
            add("Add joint…", self.load_joint_model, enabled=len(self.state.models) >= 2)
            add("Hide curves" if self.show_curves else "Show curves", self._toggle_curves,
                enabled=bool(self.state.joints))
        elif kind == JOINT:
            jl: KinItem = obj
            add("Show curves for this joint", lambda: self._set_curve_joint(jl))
            add("Hide axes" if jl.visible else "Show axes", lambda: self._toggle_visible(jl, not jl.visible))
            add("Relative to the reference pose", lambda: self._set_use_reference(jl, not jl.use_reference),
                enabled=self._has_reference(jl), checked=jl.use_reference and self._has_reference(jl))
            add("Zero at the current frame", lambda: self._set_reference(jl, self.scene.frame))
            add("Clear zero", lambda: self._set_reference(jl, None), enabled=jl.zero_frame is not None)
            add("Export values (CSV)…", lambda: self.export_joint_csv(jl))
            menu.addSeparator()
            add("Rename", lambda: self.nav.start_rename(JOINT, jl))
            add("Change bones…", lambda: self.remap_joint(jl))
            add("Remove", lambda: self.remove_joint(jl))
        elif kind == SCENE:
            add("Fit bones in view", self.scene.zoom_to_bones, enabled=bool(self.state.models))
            add("Free 3D view", lambda: self.set_view("3d"), enabled=bool(self.state.models))
        return menu

    def _show_all(self, on: bool) -> None:
        for m in self.state.models:
            m.visible = on
            self.scene.refresh_model_style(m)
            self.nav.refresh_obj(MODEL, m)
        self.scene.update_frame(self.scene.frame)
        if self.sel_kind == MODEL:
            self.show_properties()

    # ---------------------------------------------------------------- properties
    def show_properties(self) -> None:
        """Fill the properties sheet for whatever is selected in the navigator."""
        kind, obj = self.sel_kind, self.sel_obj
        p = self.props
        if kind == MODEL and obj is not None:
            self._props_model(obj)
        elif kind == AXES and obj is not None:
            self._props_axes(obj)
        elif kind == TRACKING and obj is not None:
            self._props_tracking(obj)
        elif kind == CAMERA and obj is not None:
            self._props_camera(obj)
        elif kind == CAMERAS:
            self._props_cameras()
        elif kind == SCENE:
            self._props_scene()
        elif kind == JOINT and obj is not None:
            self._props_joint(obj)
        elif kind == JOINTS:
            self._props_joints()
        elif kind == TRIAL and obj is not None:
            self._props_trial(obj)
        elif kind in (TRIALS, ADD_TRIAL) or self.state.trial is None:
            self._props_trials()
        else:
            self._props_models()
        p.finish()

    def _props_trials(self) -> None:
        p = self.props
        p.clear("Trials")
        n = len(self.state.trials)
        p.value("Trials", str(n) if n else "None yet")
        if self.state.trial is not None:
            p.value("Shown", self.state.trial.name)
        p.note("A trial holds its bones (with tracking), X-ray cameras and joints. One trial is shown at a time: "
               "double-click a trial in the navigator to show it.")
        p.buttons([("Import trial…", lambda: self.import_trial()), ("Empty trial…", lambda: self.add_empty_trial())],
                  primary="Import trial…")

    def _props_trial(self, t: Trial) -> None:
        p = self.props
        shown = t is self.state.trial
        p.clear(t.name, icon_color="#8FD3A9" if shown else None)
        p.line("Name", t.name, lambda v: self._rename(t, v))
        p.value("Status", "Shown" if shown else "Not shown", kind="Ok" if shown else "Muted")
        tracked = sum(1 for m in t.models if m.tracking is not None)
        p.value("Bones", f"{len(t.models)} ({tracked} with tracking)" if t.models else "None")
        frames = max((m.tracking.n_frames for m in t.models if m.tracking is not None), default=0)
        if frames:
            p.value("Frames", str(frames))
        p.value("X-ray cameras", str(len(t.cameras)) if t.cameras else "None")
        p.value("Joints", str(len(t.joints)) if t.joints else "None")
        if t.cfg_path:
            p.value("Imported from", Path(t.cfg_path).name, tip=str(t.cfg_path))
        if t.tracking_dir:
            p.value("Tracking folder", Path(t.tracking_dir).name, tip=str(t.tracking_dir))
        items = [] if shown else [("Show", lambda: self.show_trial(t))]
        items += [("Configure…", lambda: self.configure_trial(t)),
                  ("Duplicate…", lambda: self.duplicate_trial(t)),
                  ("Remove", lambda: self.remove_trial(t))]
        btns = p.buttons(items, danger="Remove", primary=None if shown else "Show")
        for b, tip in zip(btns, ([] if shown else ["Show this trial"]) +
                          ["Change its name, bones, tracking files and cameras",
                           ("A new trial with the same bones, coordinate systems, joints and camera calibrations, "
                            "for another trial's tracking and radiographs"), "Remove this trial from the tab"]):
            b.setToolTip(tip)

    def _props_models(self) -> None:
        p = self.props
        p.clear("Bones")
        n = len(self.state.models)
        tracked = sum(1 for m in self.state.models if m.tracking)
        p.value("Bones", str(n))
        p.value("With tracking", f"{tracked} of {n}" if n else "–")
        if n:
            p.value("Frames", str(self.scene.n_frames))
        p.note("Select a bone in the navigator to edit it. Right-click a bone for its actions.")
        p.buttons([("Add bone…", self.add_models), ("Load model…", self.load_full_model)] +
                  ([("Show all", lambda: self._show_all(True)), ("Hide all", lambda: self._show_all(False))]
                   if n else []))

    def _props_model(self, m: BoneModel) -> None:
        p = self.props
        p.clear(m.name)
        p.section("Appearance")
        p.line("Name", m.name, lambda v: self._rename(m, v))
        p.color("Colour", m.color, lambda c: self._set_model(m, "color", c))
        p.slider("Opacity", m.opacity, 0, 100, 100, _fmt_pct, lambda v: self._set_model(m, "opacity", v, nav=False),
                 spin=PCT)
        p.check("Visible", m.visible, lambda on: self._toggle_visible(m, on))
        p.combo("Display as", [("surface", "Surface"), ("volume", "Volume", m.image_data is not None),
                               ("both", "Surface and volume", m.image_data is not None)],
                m.display, lambda v: self._set_model(m, "display", v),
                tip=None if m.image_data is not None else "Volume display needs a TIF surface.")
        p.section("Motion")
        if m.tracking:
            p.value("Tracking", f"{m.tracking.path.name} · {m.tracking.n_rows} rows", tip=str(m.tracking.path))
        else:
            p.value("Tracking", "None. Shown untracked, in its own model space.", kind="Warn")
        p.check("Hold still", self.state.settings.hold_still == m.name, lambda on: self.hold_still(m, on),
                caption="Others move relative to it", enabled=m.tracking is not None)
        p.buttons([("Load new file…" if m.tracking_path else "Load tracking file…", lambda: self.attach_tracking(m))]
                  + ([("Refresh current file", lambda: self.refresh_tracking(m))] if m.tracking_path else []))
        p.section("Surface file")
        p.value("File", m.path.name, tip=str(m.path))
        if m.is_volume:
            vs = " × ".join(f"{x:g}" for x in m.voxel_size) if m.voxel_size else "?"
            p.value("Voxel size", f"{vs} mm")
            p.value("Surface", m.surface_note)
            p.buttons([("Replace…", lambda: self.replace_model(m)), ("Voxel size / surface…", lambda: self.edit_voxel(m))])
        else:
            p.value("Space", f"Moved into Autoscoper space with {Path(m.tfm_path).name} (inverted)" if m.tfm_path
                    else "Assumed to be in Autoscoper space")
            p.buttons([("Replace…", lambda: self.replace_model(m)),
                       ("Remove .tfm", lambda: self.clear_tfm(m)) if m.tfm_path else
                       ("Apply .tfm…", lambda: self.load_tfm(m))])
        p.buttons([("Remove bone", lambda: self.remove_model(m))], danger="Remove bone")

    def _props_tracking(self, m: BoneModel) -> None:
        p = self.props
        t = m.tracking
        p.clear(f"Tracking · {m.name}")
        if t is None:
            p.note("No tracking file loaded, so this bone stays untracked, in its own model space.", "Warn")
            p.buttons([("Load file…", lambda: self.attach_tracking(m))])
            return
        p.section("Tracking file")
        p.value("File", t.path.name, tip=str(t.path))
        p.value("Rows in file", str(t.n_rows))
        first, last = t.offset + 1, t.offset + t.n_rows
        p.value("Trial frames", f"{max(first, 1)}–{last}" + (f" (first {1 - first} rows dropped)" if first < 1 else ""),
                tip="Rows are matched to trial frames by number (Autoscoper files have no timestamps). Change "
                    "where the file starts with Format and frames….")
        spans = {(o.name, o.tracking.offset, o.tracking.n_rows) for o in self.state.models
                 if o is not m and o.tracking is not None}
        diff = [f"{n} {a + 1}–{a + r}" for n, a, r in sorted(spans) if (a, r) != (t.offset, t.n_rows)]
        if diff:
            p.note("Other bones cover different frames: " + ", ".join(diff) + ". Check that their first rows are "
                   "the same moment as this file's.", "Warn")
        gaps = t.gaps()
        if gaps:
            shown = ", ".join(f"{a + 1}–{b + 1}" if b > a else f"{a + 1}" for a, b in gaps[:6])
            p.value("Untracked", f"{t.n_untracked} frames ({shown}{' …' if len(gaps) > 6 else ''})", kind="Warn")
        else:
            p.value("Untracked", "None")
        p.section("Format")
        p.value("Read as", t.fmt.describe())
        if t.fmt.n_volumes > 1:
            p.value("Bone in file", f"Volume {t.volume_index + 1} of {t.fmt.n_volumes}")
        for w in t.warnings:
            p.note(w, "Warn")
        p.buttons([("Format and frames…", lambda: self.edit_format(m)),
                   ("Refresh current file", lambda: self.refresh_tracking(m))])
        p.buttons([("Load new file…", lambda: self.attach_tracking(m)), ("Remove", lambda: self.detach_tracking(m))])

    def _props_camera(self, cam: XrayCamera) -> None:
        p = self.props
        i = self.state.cameras.index(cam)
        p.clear(cam.name)
        p.section("Camera")
        p.line("Name", cam.name, lambda v: self._rename(cam, v))
        p.check("Visible", cam.visible, lambda on: self._toggle_visible(cam, on))
        p.section("Show in the 3D scene")
        p.check("Radiograph", cam.show_plane, lambda on: self._set_camera(cam, "show_plane", on),
                caption="On detector plane")
        p.check("X-ray source", cam.show_source, lambda on: self._set_camera(cam, "show_source", on),
                caption="Show marker")
        p.check("Beam outline", cam.show_beams, lambda on: self._set_camera(cam, "show_beams", on),
                caption="Lines to image corners")
        p.buttons([("View through this camera", lambda: self.set_xray_view(i))])
        p.section("Images")
        n = len(cam.images)
        p.value("Images", f"{n} · {cam.images.describe()}", tip=str(cam.images.source))
        w, h = cam.images.display_size
        p.value("Shown at", f"{w} × {h}" + (f" (1/{cam.images.display_step} size)" if cam.images.display_step > 1 else ""),
                tip="Radiographs are shrunk for display so that every image fits in memory "
                    f"(up to {cam.images.display_budget_mb} MB per camera)")
        p.value("Folder", cam.images.source.name, tip=str(cam.images.source))
        p.value("Read from", cam.images.where_from().split(" (")[0].capitalize(),
                tip="Radiographs come from the original images, the local cache (Radiographs → Radiograph "
                    "cache), or the copies saved with the session")
        if cam.note:
            p.note(cam.note, "Warn")
        if self.state.models and n != self.scene.n_frames:
            p.note(f"{n} images for {self.scene.n_frames} tracked frames (image 1 is shown at frame 1).", "Warn")
        p.section("Calibration")
        p.value("File", cam.calibration_path.name, tip=str(cam.calibration_path))
        p.value("Details", cam.calibration.summary())
        p.buttons([("Change files…", lambda: self.edit_camera(i))])
        p.buttons([("Remove camera", lambda: self.remove_camera(i))], danger="Remove camera")

    def _props_cameras(self) -> None:
        p = self.props
        s = self.state.settings
        p.clear("X-ray cameras")
        if not self.state.cameras:
            p.note("Optional: not needed to view motion. Add each X-ray camera's calibration and images to show "
                   "the radiographs behind the bones, or to look through a camera at its radiograph.")
            p.buttons([("Add X-ray camera…", self.add_camera), ("Import from .cfg…", self.import_cfg)])
            return
        p.note("These settings apply to every camera. Select a camera for its own options.")
        p.section("Radiographs in the 3D scene")
        p.slider("Image opacity", s.xray_opacity, 0, 100, 100, _fmt_pct, lambda v: self._set("xray_opacity", v),
                 spin=PCT)
        p.slider("Distance behind", s.xray_distance, 0, 800, 1, lambda x: f"{x:.0f} mm",
                 lambda v: self._set("xray_distance", v), on_release=True, spin=(1, 0, " mm"))
        p.section("Through-X-ray views")
        p.slider("Bone opacity", s.xray_bone_opacity, 0, 100, 100, _fmt_pct, lambda v: self._set("xray_bone_opacity", v),
                 spin=PCT)
        p.check("Wireframe bones", s.xray_outline, lambda on: self._set("xray_outline", on), caption="Draw as wireframe")
        p.section("Camera views (2D)")
        p.check("Show", s.camera_views, lambda on: self.set_camera_views(on), caption="Beside the 3D view",
                tip="Each camera's radiograph in its own 2D panel, with the bones projected onto it")
        p.combo("Bones drawn as", OVERLAYS, s.overlay, lambda v: (self._set("overlay", v), self.show_properties()))
        if s.overlay == "silhouette":
            p.check("Fill", s.overlay_fill, lambda on: (self._set("overlay_fill", on), self.show_properties()),
                    caption="Show fill")
            if s.overlay_fill:
                p.slider("Fill opacity", s.overlay_opacity, 5, 100, 100, _fmt_pct,
                         lambda v: self._set("overlay_opacity", v), spin=PCT)
            p.check("Outline", s.overlay_outline, lambda on: (self._set("overlay_outline", on), self.show_properties()),
                    caption="Show outline", tip="The bone's outline in its colour, at full opacity, plus the edges "
                                                "inside it where the surface turns away from the X-ray source")
            if s.overlay_outline:
                p.slider("Outline weight", s.outline_width, 1, 6, 1, lambda x: f"{x:.0f} px",
                         lambda v: self._set("outline_width", int(round(v))), spin=(1, 0, " px"))
        elif s.overlay == "points":
            p.slider("Point opacity", s.overlay_opacity, 5, 100, 100, _fmt_pct,
                     lambda v: self._set("overlay_opacity", v), spin=PCT)
        p.section("Image")
        p.slider("Brightness", s.brightness, -100, 100, 100, lambda x: f"{x:+.2f}", lambda v: self._set("brightness", v),
                 spin=(1, 2, ""))
        p.slider("Contrast", s.contrast, 20, 300, 100, lambda x: f"{x:.1f}×", lambda v: self._set("contrast", v),
                 spin=(1, 1, "×"))
        p.check("Invert", s.invert, lambda on: self._set("invert", on))
        p.buttons([("Add X-ray camera…", self.add_camera), ("Import from .cfg…", self.import_cfg)])

    def _props_scene(self) -> None:
        p = self.props
        s = self.state.settings
        p.clear("Scene")
        p.section("Display")
        p.check("Axes", s.show_axes, lambda on: self._set("show_axes", on), caption="Global axes at origin")
        p.check("Grid", s.show_grid, lambda on: self._set("show_grid", on), caption="Grid with mm scale")
        p.combo("Background", [("dark", "Dark"), ("light", "Light grey"), ("white", "White (for figures)")],
                s.background, lambda v: self._set("background", v))
        p.section("Motion trail")
        p.check("Show trail", s.trail, lambda on: self._set("trail", on), caption="Selected bone")
        p.spin("Every", s.trail_step, 1, 100, lambda v: self._set("trail_step", v), suffix=" frames")
        p.spin("Copies", s.trail_count, 1, 8, lambda v: self._set("trail_count", v))
        p.section("Untracked frames")
        p.combo("Show as", [("hide", "Hide the bone"), ("hold", "Show last tracked pose"),
                                              ("interpolate", "Interpolate across gaps")],
                s.untracked, lambda v: self._set("untracked", v))

    # ---- property setters
    def _set(self, key: str, value) -> None:
        if key in TAB_WIDE:
            self.state.set_tab_wide(key, value)          # the same for every trial
        else:
            setattr(self.state.settings, key, value)     # the shown trial's own
        self._settings_changed(key)

    def _set_model(self, m: BoneModel, key: str, value, nav: bool = True) -> None:
        setattr(m, key, value)
        self._appearance(m)
        if nav:
            self.nav.refresh_obj(MODEL, m)

    def _set_camera(self, cam: XrayCamera, key: str, value) -> None:
        setattr(cam, key, value)
        self.scene.update_frame(self.scene.frame)

    # ---------------------------------------------------------------- models
    def add_models(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add bones (surface files)", "", MODEL_FILTER)
        if not paths:
            return
        models = []
        for p in paths:
            p = Path(p)
            m = BoneModel(name=p.stem, path=p, color=PALETTE[(len(self.state.models) + len(models)) % len(PALETTE)])
            if is_volume_file(p):
                guess = self.cfg.voxel_size_for(p) if self.cfg else None
                dlg = VoxelDialog(self, p.name, guess or (self._last_voxel() or None),
                                  flip=self.cfg.flip_for(p) if self.cfg else (0, 0, 0))
                if dlg.exec() != QDialog.Accepted:
                    continue
                m.voxel_size, m.flip, m.threshold = dlg.values()
            models.append(m)
        if models:
            self._build_models(models)

    def _last_voxel(self):
        for m in reversed(self.state.models):
            if m.voxel_size:
                return m.voxel_size
        return None

    def _build_models(self, models: list[BoneModel], then=None) -> None:
        def work():
            built, errors = [], []
            for m in models:
                try:
                    m.build()
                    if m.tracking_path:
                        m.load_tracking()
                    built.append(m)
                except (ModelError, TrackingError, TransformFileError, OSError, ValueError) as e:
                    errors.append(f"{m.path.name}: {e}")
            return built, errors

        names = ", ".join(m.path.name for m in models[:3]) + (" …" if len(models) > 3 else "")
        self._run(f"Loading {names}", work, lambda res, err: self._models_built(res, err, then))

    def _models_built(self, res, err, then) -> None:
        if err:
            self._error("Couldn't load the bones", err)
            return
        built, errors = res
        first = not self.state.models
        for m in built:
            self.state.models.append(m)          # (adds "Trial 1" first in a tab without trials)
            self.scene.add_model(m)
        self._scene_trial = self.state.trial
        if built:
            self.select(built[-1], refresh=False)
            self.sel_kind, self.sel_obj = MODEL, built[-1]
        if errors:
            self._error("Some bones couldn't be loaded", "\n\n".join(errors))
        self.refresh_all()
        self.scene.apply_settings()
        if first and built:
            self.set_view("3d")
        if then:
            then()

    def select(self, m: BoneModel | None, refresh: bool = True, from_nav: bool = False) -> None:
        """Make m the selected bone (trail, timeline coverage) and show it in the navigator."""
        self.selected = m
        self.scene.selected = m
        self.scene.rebuild_trail()
        if refresh:
            self._refresh_coverage()
            self.scene.update_frame(self.scene.frame)
            if not from_nav and m is not None:
                self.nav.select(MODEL, m)

    def _picked(self, actor) -> None:
        for m in self.state.models:
            e = self.scene._bone_actors.get(id(m))
            if e and actor in (e["surface"], e["volume"]):
                self.select(m)
                return

    def _appearance(self, m: BoneModel) -> None:
        self.scene.refresh_model_style(m)
        self.scene.update_frame(self.scene.frame)

    def replace_model(self, m: BoneModel) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Replace surface file", str(m.path.parent), MODEL_FILTER)
        if not path:
            return
        new = BoneModel.from_dict({**m.to_dict(), "path": path, "tfm_path": None})
        if is_volume_file(path):
            dlg = VoxelDialog(self, Path(path).name, m.voxel_size or self._last_voxel())
            if dlg.exec() != QDialog.Accepted:
                return
            new.voxel_size, new.flip, new.threshold = dlg.values()
        self._rebuild(m, new)

    def _rebuild(self, old: BoneModel, new: BoneModel) -> None:
        def work():
            new.build()
            new.axes_model = old.axes_model if new.has_axes else None
            if new.tracking_path:
                new.tracking = old.tracking if old.tracking_path == new.tracking_path and \
                    old.tracking_overrides == new.tracking_overrides and old.tracking_volume == new.tracking_volume \
                    and old.tracking_offset == new.tracking_offset else new.load_tracking()
            return new

        def done(res, err):
            if err:
                self._error("Couldn't update the bone", err)
                return
            i = self.state.models.index(old)
            self.state.models[i] = res
            self.scene.remove_model(old)
            self.scene.add_model(res)
            self.select(res, refresh=False)
            self.sel_kind, self.sel_obj = MODEL, res
            self.refresh_all()
            self.scene.apply_settings()

        self._run(f"Loading {new.path.name}", work, done)

    def load_tfm(self, m: BoneModel) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Transform into Autoscoper space", str(m.path.parent),
                                              "ITK transform (*.tfm *.txt);;All files (*)")
        if path:
            self._rebuild(m, BoneModel.from_dict({**m.to_dict(), "tfm_path": path}))

    def clear_tfm(self, m: BoneModel) -> None:
        self._rebuild(m, BoneModel.from_dict({**m.to_dict(), "tfm_path": None}))

    def edit_voxel(self, m: BoneModel) -> None:
        dlg = VoxelDialog(self, m.path.name, m.voxel_size, m.threshold, m.flip)
        if dlg.exec() == QDialog.Accepted:
            voxel, flip, level = dlg.values()
            self._rebuild(m, BoneModel.from_dict({**m.to_dict(), "voxel_size": voxel, "flip": flip,
                                                  "threshold": level}))

    def remove_model(self, m: BoneModel) -> None:
        if QMessageBox.question(self, "Remove bone", f"Remove {m.name} from this tab?") != QMessageBox.Yes:
            return
        self.scene.remove_model(m)
        self.state.models.remove(m)
        if self.state.settings.hold_still == m.name:
            self.state.settings.hold_still = None
        nxt = self.state.models[-1] if self.state.models else None
        self.select(nxt, refresh=False)
        self.sel_kind, self.sel_obj = (MODEL, nxt) if nxt else (MODELS, None)
        self.refresh_all()
        self.scene.frame = min(self.scene.frame, self.scene.n_frames - 1)
        self.scene.apply_settings()

    def hold_still(self, m: BoneModel, on: bool) -> None:
        self.state.settings.hold_still = m.name if on else None
        self.scene.apply_settings()          # grid follows the new frame of reference
        self._update_frame_info()
        self.scene.zoom_to_bones()
        if self.sel_obj is m:
            self.show_properties()

    # ---------------------------------------------------------------- tracking
    def attach_tracking(self, m: BoneModel, path: str | None = None) -> None:
        """Load a tracking file for a bone (a new one, or replacing the one it has)."""
        if path is None:
            start = m.tracking_path.parent if m.tracking_path else m.path.parent
            path, _ = QFileDialog.getOpenFileName(self, f"Tracking file for {m.name}", str(start),
                                                  "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if not path:
                return
        try:
            detected = load_tracking(path)
        except (TrackingError, OSError) as e:
            self._error("Couldn't read the tracking file", str(e))
            return
        overrides, vol, offset = self._remembered_format(m, detected)
        if overrides is None:
            dlg = self._format_dialog(m, path, detected, detected.fmt, 0, 0, ask_remember=True)
            if dlg.exec() != QDialog.Accepted:
                return
            (overrides, vol), offset = dlg.values(), dlg.offset
            if dlg.remember.isChecked():
                self.tracking_format = dict(overrides)
        try:
            t = load_tracking(path, vol, overrides, offset)
        except (TrackingError, OSError) as e:
            self._error("Couldn't read the tracking file", str(e))
            return
        m.tracking_path, m.tracking_volume, m.tracking_overrides, m.tracking_offset = Path(path), vol, overrides, offset
        m.tracking = t
        self._tracking_changed(m)
        if t.warnings:
            self._error("Check the tracking format", "\n".join(t.warnings) +
                        "\n\nRight-click Tracking under the bone and choose Tracking format and frames… to correct it.")

    def _other_tracking(self, m: BoneModel) -> list:
        return [(o.name, o.tracking.offset, o.tracking.n_rows) for o in self.state.models
                if o is not m and o.tracking is not None]

    def _radiograph_frames(self) -> int | None:
        for cam in self.state.cameras:
            if cam.images is not None:
                return len(cam.images) - cam.first_image
        return None

    def _format_dialog(self, m: BoneModel, path, detected, fmt, vol: int, offset: int, ask_remember=False):
        return FormatDialog(self, fmt, vol, path=path, detected=detected.fmt, offset=offset,
                            n_rows=detected.n_rows, others=self._other_tracking(m),
                            n_images=self._radiograph_frames(), bone=m.name, ask_remember=ask_remember)

    def _remembered_format(self, m: BoneModel, detected):
        """The tab's remembered options, when they fit this file without asking; else (None, 0, 0)."""
        rem = self.tracking_format
        if not rem or detected.fmt.n_volumes != 1 or rem.get("kind") != detected.fmt.kind:
            return None, 0, 0
        spans = {(off, rows) for _n, off, rows in self._other_tracking(m)}
        if spans and spans != {(0, detected.n_rows)}:
            return None, 0, 0           # different lengths: ask, so the frames can be lined up
        try:
            if load_tracking(detected.path, 0, rem).warnings:
                return None, 0, 0
        except TrackingError:
            return None, 0, 0
        return dict(rem), 0, 0

    def refresh_tracking(self, m: BoneModel) -> None:
        """Re-read the bone's tracking file (e.g. after saving over it in Autoscoper), keeping its options."""
        if m.tracking_path is None:
            return
        try:
            t = m.load_tracking()
        except (TrackingError, OSError) as e:
            self._error("Couldn't refresh the tracking", f"{m.tracking_path}\n\n{e}")
            return
        self._tracking_changed(m)
        self._status(f"Refreshed {m.tracking_path.name}: {t.n_rows} rows")
        if t.warnings:
            self._error("Check the tracking format", "\n".join(t.warnings))

    def _tracking_changed(self, m: BoneModel) -> None:
        if self.state.cameras:
            self.scene.reposition_cameras()
        self.select(m, refresh=False)
        if self.sel_obj is not m:
            self.sel_kind, self.sel_obj = MODEL, m
        self.refresh_all()
        self.scene.frame = min(self.scene.frame, self.scene.n_frames - 1)
        self.scene.apply_settings()          # grid and axes now cover where the bones move
        self.scene.zoom_to_bones()

    def detach_tracking(self, m: BoneModel) -> None:
        m.tracking_path, m.tracking = None, None
        if self.state.settings.hold_still == m.name:
            self.state.settings.hold_still = None
        self._tracking_changed(m)

    def edit_format(self, m: BoneModel) -> None:
        if not m.tracking:
            return
        try:
            detected = load_tracking(m.tracking_path)
        except (TrackingError, OSError) as e:
            self._error("Couldn't read the tracking file", str(e))
            return
        dlg = self._format_dialog(m, m.tracking_path, detected, m.tracking.fmt, m.tracking_volume, m.tracking_offset)
        if dlg.exec() != QDialog.Accepted:
            return
        (overrides, vol), offset = dlg.values(), dlg.offset
        try:
            t = load_tracking(m.tracking_path, vol, overrides, offset)
        except TrackingError as e:
            self._error("That format doesn't fit this file", str(e))
            return
        m.tracking, m.tracking_volume, m.tracking_overrides, m.tracking_offset = t, vol, overrides, offset
        self._tracking_changed(m)

    # ---------------------------------------------------------------- playback
    def _play(self, on: bool) -> None:
        self.timeline.play.setText("❚❚" if on else "▶")
        if on:
            self.timer.start(int(1000 / (self.state.settings.fps * self.state.settings.speed)))
        else:
            self.timer.stop()

    def _tick(self) -> None:
        self.timer.setInterval(int(1000 / (self.state.settings.fps * self.state.settings.speed)))
        if not self.timeline.step(1):
            self.timeline.play.setChecked(False)

    def _frame_changed(self, f: int) -> None:
        self.scene.update_frame(f)
        self._update_frame_info()
        self._pose_joint_axes()
        self.curves.set_frame(f)
        if self.sel_kind == JOINT and not self.timer.isActive():
            self.show_properties()

    # ---------------------------------------------------------------- views & settings
    def set_view(self, key: str) -> None:
        self._negative = False
        self._current_view = key
        for k, b in self.view_buttons.items():
            b.setChecked(k == key)
        self.scene.set_view(key)
        self._refresh_xray_buttons()

    def _flip(self) -> None:
        if self.scene.view_mode.startswith("xray"):
            return
        self._negative = not self._negative
        self.scene.set_view(getattr(self, "_current_view", "3d"), self._negative)

    def set_xray_view(self, i: int) -> None:
        for b in self.view_buttons.values():
            b.setChecked(False)
        self.scene.set_xray_view(i)
        self._refresh_xray_buttons()

    def _settings_changed(self, key: str) -> None:
        if key == "xray_distance":
            self.scene.reposition_cameras()
        elif key in ("show_axes", "show_grid", "background", "trail", "trail_step", "trail_count"):
            self.scene.apply_settings()
        else:
            for m in self.state.models:
                self.scene.refresh_model_style(m)
            self.scene.update_frame(self.scene.frame)
        self._update_frame_info()

    def _sync_timeline_settings(self) -> None:
        self.timeline.loop.setChecked(self.state.settings.loop)
        i = self.timeline.speed.findData(self.state.settings.speed)
        if i >= 0:
            self.timeline.speed.setCurrentIndex(i)

    # ---------------------------------------------------------------- radiographs
    def add_camera(self) -> None:
        dlg = AddCameraDialog(self, self.scene.n_frames, self._preview_bones(), index=len(self.state.cameras) + 1)
        if dlg.exec() == QDialog.Accepted and dlg.result_camera:
            self._read_radiographs([dlg.result_camera])
            self.state.cameras.append(dlg.result_camera)
            self.scene.add_camera(dlg.result_camera)
            self._scene_trial = self.state.trial
            self.sel_kind, self.sel_obj = CAMERA, dlg.result_camera
            self.refresh_all()
            self.scene.update_frame(self.scene.frame)

    def edit_camera(self, i: int) -> None:
        old = self.state.cameras[i]
        dlg = AddCameraDialog(self, self.scene.n_frames, self._preview_bones(), existing=old)
        if dlg.exec() == QDialog.Accepted and dlg.result_camera:
            new = dlg.result_camera
            new.visible, new.show_plane, new.show_source, new.show_beams = \
                old.visible, old.show_plane, old.show_source, old.show_beams
            if new.images is not old.images:
                self.scene.forget_camera(old)
                self._read_radiographs([new])
            else:
                self.scene.remove_camera(old)
            self.state.cameras[i] = new
            self.scene.add_camera(new)
            self.sel_kind, self.sel_obj = CAMERA, new
            self.refresh_all()
            self.scene.update_frame(self.scene.frame)

    def remove_camera(self, i: int) -> None:
        cam = self.state.cameras[i]
        if QMessageBox.question(self, "Remove camera", f"Remove {cam.name} from this tab?") != QMessageBox.Yes:
            return
        self.state.cameras.pop(i)
        self.scene.forget_camera(cam)
        if self.scene.view_mode.startswith("xray"):
            self.set_view("3d")
        self.sel_kind, self.sel_obj = CAMERAS, None
        self.refresh_all()
        self.scene.update_frame(self.scene.frame)

    # ---------------------------------------------------------------- trials (Visualiser #16, #17)
    def show_trial(self, t: Trial, select: tuple | None = None) -> None:
        """Show this trial: its bones, cameras and joints in the 3D view, camera views, timeline and curves.

        The trial shown before keeps everything it has loaded (surfaces, tracking, radiographs), so showing
        it again is instant; its settings (Visualiser #23), frame and 3D viewpoint are kept with it. A trial's
        radiographs are read (with the progress bar) the first time it is shown.
        """
        st = self.state
        if not any(x is t for x in st.trials):
            return
        if t is not self._scene_trial:
            self.timeline.play.setChecked(False)
            old = self._scene_trial
            if old is not None:
                if old is st.trial:
                    old.frame, old.view = self.scene.frame, list(self.plotter.camera_position)
                for m in old.models:
                    self.scene.remove_model(m)
                for c in old.cameras:
                    self.scene.remove_camera(c)
            st.active = next(i for i, x in enumerate(st.trials) if x is t)
            self._scene_trial = t
            self._use_trial_settings()
            self.selected = None
            self.curve_joint = None
            self.scene.selected = None
            if self.scene.view_mode != "3d":
                self.scene.view_mode = "3d"
            for m in t.models:
                if m.mesh is not None:
                    self.scene.add_model(m)
            unread = [c for c in t.cameras if c.images is not None and not getattr(c, "_read_once", False)]
            if unread:
                self._read_radiographs(unread, title=f"Loading {t.name}: radiographs")
            for c in t.cameras:
                if c.images is not None:
                    self.scene.add_camera(c)
            for m in t.models:
                self.scene.refresh_model_style(m)
        self.sel_kind, self.sel_obj = select if select is not None else (TRIAL, t)
        if self.sel_kind in (MODELS, CAMERAS, JOINTS):
            self.sel_obj = None
        if self.sel_kind in (MODEL, TRACKING) and self.sel_obj is not None:
            self.selected = self.sel_obj
            self.scene.selected = self.sel_obj
        self.refresh_all()
        frame = min(t.frame, max(self.scene.n_frames - 1, 0))
        self.timeline.slider.setValue(frame)
        self.scene.update_frame(frame)
        if t.view:
            self.plotter.camera_position = t.view          # where it was looked at from
            self.plotter.render()
        elif t.models:
            self.scene.zoom_to_bones()

    def _use_trial_settings(self) -> None:
        """Point the timeline, scene and curves at the shown trial's settings (Visualiser #23)."""
        self.timeline.s = self.state.settings
        self._sync_timeline_settings()
        self.scene.apply_settings()
        self.curves.set_style(self.state.settings.curve_labels)

    def import_trial(self, start_from: Trial | None = None) -> bool:
        """Open the Trial setup wizard (Visualiser #17) and add the trial it sets up."""
        from .trial_wizard import TrialSetupDialog

        dlg = TrialSetupDialog(self, mode="new", trials=self.state.trials, start_from=start_from,
                               taken_names=[t.name for t in self.state.trials],
                               settings=(start_from or self.state).settings)
        self._trial_dialog = dlg                # (kept for scripts and screenshots)
        if dlg.exec() != QDialog.Accepted:
            return False
        self.add_trial_from_draft(dlg.draft)
        return True

    def duplicate_trial(self, t: Trial) -> bool:
        """A new trial with this trial's bones, coordinate systems, joints and camera calibrations: the wizard
        opens on it to choose the new tracking files and radiographs."""
        return self.import_trial(start_from=t)

    def add_trial_from_draft(self, draft, then=None) -> Trial:
        """Build and show the trial a Trial setup draft describes (then add its model's axes and joints)."""
        t = draft.to_trial()
        t.name = self.state.unique_trial_name(t.name)
        t.settings = self.state.settings.copy()          # starts like the trial shown (Visualiser #23) ...
        t.settings.hold_still = None
        for k, v in draft.display.items():               # ... with the display chosen in the dialog (#22)
            setattr(t.settings, k, v)
        if not draft.read_radiographs:
            for c in t.cameras:
                c._read_once = True           # read as each frame is shown

        def loaded():
            if draft.model_path:
                self.apply_model_to_trial(draft.model_path)
            if then:
                then()

        self.load_trial(t, then=loaded)
        return t

    def apply_model_to_trial(self, path) -> list[str]:
        """Give the shown trial's bones the coordinate systems of a Model Builder model's bones with matching
        names (or surface files), and add the model's joints between them. Returns the bones given axes."""
        from ..core.trial import match_segments

        try:
            self._kin_models.pop(str(Path(path).resolve()), None)
            km = self._kin_model(path)
        except Exception as e:  # noqa: BLE001
            self._error("Couldn't open the model", str(e))
            return []
        models = list(self.state.models)
        smap = guess_segment_map(km, models)
        rest = [sg for sg in km.segments if sg.acs is not None and sg.name not in smap]
        free = [m.name for m in models if m.name not in smap.values()]
        smap.update(match_segments([(sg.name, Path(sg.surface.path).stem if sg.surface else sg.name) for sg in rest],
                                   free))
        done = []
        for sname, mname in smap.items():
            m = self._model_named(mname)
            if m is None or km.segment(sname).acs is None:
                continue
            m.axes_path, m.axes_segment, m.axes_model = Path(path), sname, km
            m.axes_reference, m.axes_zero = "first", None
            done.append(mname)
        if any(j.proximal in smap and j.distal in smap for j in km.joints):
            self.load_joint_model_file(path, segment_map=smap, ask=False)
        else:
            self.refresh_all()
        return done

    def configure_trial(self, t: Trial) -> bool:
        """Change a trial in the Trial setup dialog: its name, bones (surfaces, colours, tracking files), cameras."""
        from ..core.trial import draft_from_trial
        from .trial_wizard import TrialSetupDialog

        draft = draft_from_trial(t)
        draft.read_radiographs = True
        dlg = TrialSetupDialog(self, draft, mode="configure", trials=self.state.trials,
                               taken_names=[x.name for x in self.state.trials if x is not t])
        self._trial_dialog = dlg
        if dlg.exec() != QDialog.Accepted:
            return False
        self.apply_trial_draft(t, dlg.draft)
        return True

    def apply_trial_draft(self, t: Trial, d, then=None) -> None:
        """Bring a trial in line with an edited draft. Bones and cameras whose files are unchanged keep what
        they have loaded (surfaces, radiographs); only new or changed files are read."""
        from ..core import trial as TR

        shown = t is self._scene_trial
        models, rebuild, retrack = [], [], []
        renamed = {}
        for b in (x for x in d.bones if x.include):
            src = b.source
            if src is not None and TR.same_surface(b, src):
                m = src
                if m.name != b.name:
                    renamed[m.name] = b.name
                m.name, m.color = b.name, b.color
                new_trk = (Path(b.tracking_path) if b.tracking_path else None, dict(b.tracking_overrides),
                           b.tracking_volume, b.tracking_offset)
                if new_trk != (Path(m.tracking_path) if m.tracking_path else None, dict(m.tracking_overrides),
                               m.tracking_volume, m.tracking_offset):
                    m.tracking_path, m.tracking_overrides, m.tracking_volume, m.tracking_offset = new_trk
                    m.tracking = None
                    retrack.append(m)
            else:
                m = TR.bone_model(b)
                if src is not None and src.name != b.name:
                    renamed[src.name] = b.name
                rebuild.append(m)
            models.append(m)
        removed = [m for m in t.models if not any(m is x for x in models)]
        cams, new_cams = [], []
        for c in (x for x in d.cameras if x.include):
            src = c.source_cam
            if src is not None and Path(src.calibration_path) == Path(c.calibration_path) and \
                    [Path(p) for p in src.source] == TR.camera_source(c) and src.source_kind == c.source_kind:
                src.name, src.show_source = c.name, c.show_source
                cams.append(src)
            else:
                cam = TR.camera(c)
                new_cams.append(cam)
                cams.append(cam)
        gone_cams = [c for c in t.cameras if not any(c is x for x in cams)]
        for jl in t.joints:                          # joints refer to bones by name
            for r, n in list(jl.model_map.items()):
                if n in renamed:
                    jl.model_map[r] = renamed[n]
        if t.hold_still in renamed:
            t.hold_still = renamed[t.hold_still]
        if shown and self.state.settings.hold_still in renamed:
            self.state.settings.hold_still = renamed[self.state.settings.hold_still]
        t.name = self.state.unique_trial_name(d.name) if d.name != t.name else t.name
        t.tracking_dir = d.tracking_dir
        for k, v in d.display.items():                   # display settings from the dialog (Visualiser #22)
            setattr(t.settings, k, v)
        if shown:
            for m in removed + rebuild:
                self.scene.remove_model(m)
        for c in gone_cams:
            self.scene.forget_camera(c)

        def work():
            errors = []
            for m in rebuild:
                try:
                    if m.mesh is None:
                        m.build()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{m.name}: {e}")
            for m in rebuild + retrack:
                if m.tracking_path:
                    try:
                        m.load_tracking()
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{m.name} tracking: {e}")
            return errors

        def done(res, err):
            problems = [str(err)] if err else list(res)
            t.models = [m for m in models if m.mesh is not None]
            ok = []
            for cam in cams:
                if any(cam is x for x in new_cams):
                    try:
                        cam.build()
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{cam.name}: {e}")
                        continue
                ok.append(cam)
            t.cameras = ok
            for m in t.models:
                if m.has_axes and m.axes_model is None:
                    try:
                        m.axes_model = self._kin_model(m.axes_path)
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{m.name} coordinate system: {e}")
            if shown:
                st = self.state.settings
                t.frame, t.view = self.scene.frame, list(self.plotter.camera_position)
                self._scene_trial = None              # put the trial's bones and cameras back in the scene
                self.show_trial(t, select=(TRIAL, t))
            else:
                self.refresh_all()
            if problems:
                self._error("Some files couldn't be used", "\n".join(problems))
            if then:
                then()

        if rebuild or retrack:
            self._run(f"{t.name}: loading changes", work, done)
        else:
            done([], None)

    @staticmethod
    def _surface_key(m: BoneModel) -> tuple:
        """What a bone's built surface depends on (bones with the same key can share it)."""
        return (str(Path(m.path).resolve()), tuple(m.voxel_size or ()), tuple(m.flip), m.threshold,
                str(m.tfm_path) if m.tfm_path else None)

    def add_empty_trial(self, name: str | None = None) -> Trial | None:
        """Add a trial with a name only (asked for unless given), and show it."""
        if name is None:
            name, ok = QInputDialog.getText(self, "Empty trial", "Trial name:",
                                            text=self.state.unique_trial_name(f"Trial {len(self.state.trials) + 1}"))
            if not ok or not name.strip():
                return None
        t = self.state.add_trial(name.strip())
        self.show_trial(t)
        return t

    def remove_trial(self, t: Trial, ask: bool = True) -> None:
        """Remove a trial and everything in it (its files are untouched)."""
        if ask and QMessageBox.question(
                self, "Remove trial", f"Remove “{t.name}” and its {len(t.models)} bones, {len(t.cameras)} X-ray "
                                      "cameras and joints from this tab? (The files aren't changed.)") \
                != QMessageBox.Yes:
            return
        st = self.state
        shown = t is st.trial
        if t is self._scene_trial:
            for m in t.models:
                self.scene.remove_model(m)
            self._scene_trial = None
        for c in t.cameras:
            self.scene.forget_camera(c)
        i = next(k for k, x in enumerate(st.trials) if x is t)
        st.trials.pop(i)
        if not shown and i < st.active:
            st.active -= 1                     # the shown trial moved up one place
        if not st.trials:
            st.active = -1
            self.sel_kind, self.sel_obj = TRIALS, None
            self.refresh_all()
            self.scene.update_frame(0)
        elif shown:
            st.active = -1
            self.show_trial(st.trials[min(i, len(st.trials) - 1)])
        else:
            self.refresh_all()

    def import_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Autoscoper trial configuration", "", "Config (*.cfg)")
        if path:
            self.import_cfg_file(path)

    def import_cfg_file(self, path: str, quiet: bool = False) -> None:
        cfg = read_cfg(path)
        self.cfg = cfg
        if not cfg.calibrations:
            self._error("No cameras", "That .cfg doesn't list any cameras (mayaCam_csv lines).")
            return
        cams, problems = [], []
        for k, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
            cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder],
                             name=f"X-ray {len(self.state.cameras) + len(cams) + 1}")
            try:
                cam.build()
                cams.append(cam)
            except Exception as e:  # noqa: BLE001 - reported to the user
                problems.append(f"Camera {k}: {e}\n  calibration: {cal}\n  images: {folder}")
        self._read_radiographs(cams)
        for cam in cams:
            self.state.cameras.append(cam)
            self.scene.add_camera(cam)
            self._scene_trial = self.state.trial
        if cams:
            self.sel_kind, self.sel_obj = CAMERAS, None
        self.refresh_all()
        self.scene.update_frame(self.scene.frame)
        msg = f"Added {len(cams)} camera(s) from {Path(path).name}."
        counts = {len(c.images) for c in cams}
        if cams and self.state.models and any(n != self.scene.n_frames for n in counts):
            msg += f"\n\nImage counts ({', '.join(map(str, sorted(counts)))}) differ from the {self.scene.n_frames} " \
                   "tracked frames (image 1 is shown at frame 1)."
        msg += "\n\nIts voxel sizes will be offered when you add TIF volumes."
        if problems:
            self._error("Some cameras couldn't be added", msg + "\n\n" + "\n\n".join(problems))
        elif not quiet:
            QMessageBox.information(self, "Cameras imported", msg)

    # ---------------------------------------------------------------- sessions & export
    def save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save tab as session", f"{self.state.name}.bvrsession.json",
                                              "Session (*.json)")
        if path:
            self.save_session_file(path)

    def save_session_file(self, path: str, radiographs: bool | None = None, all_files: bool | None = None) -> None:
        """Save the tab, asking what to save with it (unless both choices are given)."""
        from .dialogs import SaveSessionDialog, _mb

        if radiographs is None or all_files is None:
            dlg = SaveSessionDialog(self, self.state, path)
            if dlg.exec() != QDialog.Accepted:
                return
            radiographs, all_files = dlg.choices()
        try:
            if radiographs or all_files:
                rep = self._with_progress("Saving session", lambda progress, cancelled: save_session(
                    self.state, path, radiographs=radiographs, all_files=all_files, progress=progress,
                    cancelled=cancelled))
            else:
                rep = save_session(self.state, path)
        except InterruptedError:
            return
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._error("Couldn't save the session", str(e))
            return
        self.last_save = rep
        if rep["data_folder"] is not None:
            bits = ([f"the radiographs of {rep['radiographs']} camera(s)"] if rep["radiographs"] else []) + \
                   ([f"{rep['files']} other file(s)"] if rep["files"] else [])
            self._toast(f"Saved {Path(path).name} with {' and '.join(bits)} in {rep['data_folder'].name} "
                        f"({_mb(rep['bytes'])}).")

    def _toast(self, text: str) -> None:
        """A short confirmation in the main window's status bar (or a message box without one)."""
        win = self.window()
        bar = win.statusBar() if hasattr(win, "statusBar") else None
        if bar is not None:
            bar.showMessage(text, 10000)

    def open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open saved session", "", "Session (*.json)")
        if path:
            self.load_session_file(path)

    def load_session_file(self, path: str) -> None:
        """Open a saved session: its trials are added to this tab, and the one shown when it was saved is shown."""
        try:
            st = load_session(path)
        except Exception as e:  # noqa: BLE001
            self._error("Couldn't open the session", str(e))
            return
        if not self.state.trials:
            self.state.name = st.name
            self.titleChanged.emit(st.name)
            self.state._settings = st._settings          # the tab's settings (each trial brings its own)
            self._use_trial_settings()
        self.state.originals.update(st.originals)
        trials, shown = st.trials, st.trial
        problems: list[str] = []

        def step(k: int) -> None:
            if k == len(trials):
                if shown is not None:
                    self.show_trial(shown)
                else:
                    self.refresh_all()
                if problems:
                    self._error("Some files couldn't be used as saved", "\n".join(problems))
                return
            self.load_trial(trials[k], problems, then=lambda: step(k + 1), show=False)

        step(0)

    def load_trial(self, t: Trial, problems: list | None = None, then=None, show: bool = True) -> None:
        """Read a trial's files and add it to the tab.

        Its bones and their tracking are read on a worker thread; then its X-ray cameras (calibration and
        image list), the bones' coordinate systems and the joints. Files that can't be used are left out
        and listed in ``problems`` (or shown when it isn't given). Radiographs are read when the trial is
        first shown.
        """
        report = problems is None
        problems = [] if problems is None else problems
        models = list(t.models)
        loaded = {self._surface_key(m): m for x in self.state.trials for m in x.models if m.mesh is not None}
        for m in models:                    # a surface already loaded in this tab (e.g. a duplicated trial) is shared
            same = loaded.get(self._surface_key(m)) if m.mesh is None else None
            if same is not None:
                m.mesh, m.image_data, m.surface_note = same.mesh, same.image_data, same.surface_note

        cams_in = list(t.cameras)
        n_steps = len(models) + (1 if cams_in else 0)

        def work(report, cancelled):
            """Each bone's surface and tracking, then the cameras (calibration and image list)."""
            built, errors, cams, cam_errors = [], [], [], []
            for k, m in enumerate(models):
                if cancelled():
                    return built, errors, cams, cam_errors, True
                what = "building the surface" if m.mesh is None and m.is_volume else (
                    "reading the surface" if m.mesh is None else "using its surface")
                report(k, f"{t.name}\n\nBone {k + 1} of {len(models)}: {m.name} ({Path(m.path).name}), {what}…")
                try:
                    if m.mesh is None:              # (a duplicated bone shares its surface)
                        m.build()
                    if m.tracking_path:
                        report(k, f"{t.name}\n\nBone {k + 1} of {len(models)}: {m.name}, reading its tracking…")
                        try:
                            m.load_tracking()
                        except (TrackingError, OSError, ValueError) as e:
                            errors.append(f"{t.name} · {m.name} tracking: {e}")
                    built.append(m)
                except (ModelError, TrackingError, TransformFileError, OSError, ValueError) as e:
                    errors.append(f"{t.name} · {m.path.name}: {e}")
            if cams_in:
                if cancelled():
                    return built, errors, cams, cam_errors, True
                report(len(models), f"{t.name}\n\nX-ray cameras: reading {len(cams_in)} calibration(s) and listing "
                                    "the radiographs…")
                for cam in cams_in:
                    try:
                        cam.build()
                        cams.append(cam)
                    except Exception as e:  # noqa: BLE001
                        cam_errors.append(f"{t.name} · {cam.name}: {e}")
            report(n_steps, f"{t.name}\n\nCoordinate systems and joints…")
            return built, errors, cams, cam_errors, False

        def done(res, err):
            if err:
                problems.append(f"{t.name}: {err}")
                built, cams, cancelled = [], [], False
            else:
                built, errors, cams, cam_errors, cancelled = res
                problems.extend(errors + cam_errors)
            if cancelled:
                for c in cams:
                    if c.images is not None:
                        c.images.close()
                self._status(f"Loading {t.name} was cancelled")
                if then:
                    then()
                return
            t.models = built
            problems.extend(f"{t.name} · {c.name}: {c.note}" for c in cams if c.note)
            t.cameras = cams
            for m in t.models:                       # coordinate systems on the bones
                if m.has_axes:
                    try:
                        m.axes_model = self._kin_model(m.axes_path)
                        m.axes_model.segment(m.axes_segment)
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{t.name} · {m.name} coordinate system: {e}")
            joints = []
            for jl in t.joints:
                if jl.kind == "segment":                # older sessions listed segments separately
                    m = next((x for x in t.models if x.name == jl.model_map.get(jl.name)), None)
                    if m is not None and not m.has_axes:
                        try:
                            km = self._kin_model(jl.model_path)
                            km.segment(jl.name)
                            m.axes_path, m.axes_segment, m.axes_model = Path(jl.model_path), jl.name, km
                            m.axes_visible = jl.visible
                        except Exception as e:  # noqa: BLE001
                            problems.append(f"{t.name} · {m.name} coordinate system: {e}")
                    continue
                try:
                    jl.model = self._kin_model(jl.model_path)
                    for n in jl.segment_names():          # checks the segment / joint still exists
                        jl.model.segment(n)
                    jl.label = jl.label or jl.name
                    joints.append(jl)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{t.name} · {jl.name or jl.model_path.name}: {e}")
            t.joints = joints
            if not any(x is t for x in self.state.trials):
                self.state.add_trial(trial=t)
            if show:
                self.show_trial(t)
            else:
                self.refresh_all()
            if report and problems:
                self._error("Some files couldn't be used", "\n".join(problems))
            if then:
                then()

        if n_steps:
            self._run_steps(f"Loading {t.name}", n_steps, work, done)
        else:
            done(work(lambda *_: None, lambda: False), None)

    # ---------------------------------------------------------------- coordinate systems and joints
    MODEL_FILE_FILTER = "Model from the Model Builder (*.bvrmodel.json *.json)"

    def _kin_model(self, path):
        from ..core.joints import KinematicModel

        key = str(Path(path).resolve())
        if key not in self._kin_models:
            self._kin_models[key] = KinematicModel.load(path, build_surfaces=False)
        return self._kin_models[key]

    def add_axes(self, m: BoneModel) -> None:
        """Right-click Axes under a bone → Load file…: one of the bones of a Model Builder model."""
        start = str(m.axes_path.parent) if m.axes_path else str(m.path.parent)
        path, _ = QFileDialog.getOpenFileName(self, f"Coordinate system for {m.name}", start, self.MODEL_FILE_FILTER)
        if path:
            try:
                self.set_axes(m, path)
            except Exception as e:  # noqa: BLE001 - shown to the user
                self._error("Couldn't add the coordinate system", str(e))

    def set_axes(self, m: BoneModel, path, segment: str | None = None, warn: bool = True) -> bool:
        """Give a model the axes of a segment in a model file. Asks which segment when it isn't clear."""
        from ..core.joints import JointModelError

        self._kin_models.pop(str(Path(path).resolve()), None)       # re-read: the file may have been re-saved
        km = self._kin_model(path)
        segs = [sg for sg in km.segments if sg.acs is not None]
        if not segs:
            raise JointModelError("No bone in this file has axes yet. Define them in the Model Builder and "
                                  "save again.")
        if segment is None:
            guess = guess_segment_map(km, [m]).items()
            segment = next((sname for sname, mname in guess if mname == m.name), None)
            if segment is None and len(segs) == 1:
                segment = segs[0].name
            if segment is None:
                names = [sg.name for sg in segs]
                choice, ok = QInputDialog.getItem(self, "Which bone?", f"{Path(path).name} has several "
                                                  f"bones. Which one is {m.name}?", names, 0, False)
                if not ok:
                    return False
                segment = choice
        km.segment(segment)                                            # raises if missing
        m.axes_path, m.axes_segment, m.axes_model = Path(path), segment, km
        m.axes_reference, m.axes_zero = "first", None
        if warn:
            sg = km.segment(segment)
            if sg.surface is not None and Path(sg.surface.path).name != Path(m.path).name:
                self._error("Check the bone", f"{segment} was defined on {Path(sg.surface.path).name}, but this bone "
                            f"is {Path(m.path).name}. The axes only fit if both use the same surface (same model "
                            "space).")
        self.curve_joint = m
        self.sel_kind, self.sel_obj = AXES, m
        self.refresh_all()
        return True

    def refresh_axes(self, m: BoneModel) -> None:
        """Re-read the bone's model file (e.g. after re-saving it in the Model Builder). Every bone and joint
        in this tab that uses the same file picks up the new version."""
        if not m.has_axes:
            return
        key = str(Path(m.axes_path).resolve())
        self._kin_models.pop(key, None)
        try:
            km = self._kin_model(m.axes_path)
            km.segment(m.axes_segment)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._error("Couldn't refresh the coordinate system", f"{m.axes_path}\n\n{e}")
            return
        n = 0
        for o in self.state.models:
            if o.has_axes and str(Path(o.axes_path).resolve()) == key:
                o.axes_model = km
                n += 1
        for jl in self.state.joints:
            if str(Path(jl.model_path).resolve()) == key:
                jl.model = km
                n += 1
        self.refresh_all()
        self._status(f"Refreshed {Path(m.axes_path).name}" + (f" ({n} items use it)" if n > 1 else ""))

    def load_full_model(self, path=None) -> None:
        """Load a model: all its bones (reusing bones already here with the same surface file), each with its
        coordinate system, and all its joints."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load model", "", self.MODEL_FILE_FILTER)
            if not path:
                return
        self._kin_models.pop(str(Path(path).resolve()), None)
        try:
            km = self._kin_model(path)
        except Exception as e:  # noqa: BLE001
            self._error("Couldn't open the model", str(e))
            return
        segs = [sg for sg in km.segments if sg.acs is not None and sg.surface is not None]
        if not segs:
            self._error("Nothing to load", "None of the model's bones has a surface and axes yet. Finish them in the "
                                           "Model Builder and save again.")
            return
        pairs, new = [], []
        taken = {m.name for m in self.state.models}
        for sg in segs:
            same = next((m for m in self.state.models
                         if Path(m.path).resolve() == Path(sg.surface.path).resolve()), None)
            if same is None:
                d = sg.surface.to_dict()
                name, k = sg.name, 2
                while name in taken:
                    name, k = f"{sg.name} {k}", k + 1
                taken.add(name)
                same = BoneModel.from_dict({**d, "name": name,
                                            "color": PALETTE[(len(self.state.models) + len(new)) % len(PALETTE)]})
                new.append(same)
            pairs.append((sg.name, same))

        def finish():
            here = [(s, m) for s, m in pairs if any(m is x for x in self.state.models)]
            for sname, m in here:
                m.axes_path, m.axes_segment, m.axes_model = Path(path), sname, km
                m.axes_reference, m.axes_zero = "first", None
            smap = {s: m.name for s, m in here}
            if any(j.proximal in smap and j.distal in smap for j in km.joints):
                self.load_joint_model_file(path, segment_map=smap, ask=False)
            else:
                if here:
                    self.curve_joint = here[0][1]
                    self.sel_kind, self.sel_obj = AXES, here[0][1]
                self.refresh_all()
            self.scene.apply_settings()
            self.scene.zoom_to_bones()

        if new:
            self._build_models(new, then=finish)
        else:
            finish()

    def remove_axes(self, m: BoneModel) -> None:
        m.axes_path = m.axes_segment = m.axes_model = None
        if self.curve_joint is m:
            self.curve_joint = None
        self.refresh_all()

    def _toggle_axes(self, m: BoneModel, on: bool) -> None:
        m.axes_visible = bool(on)
        self.nav.refresh_obj(AXES, m)
        self._draw_joint_axes()
        if self.sel_obj is m:
            self.show_properties()

    def load_joint_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add joints from a model", "", self.MODEL_FILE_FILTER)
        if path:
            self.load_joint_model_file(path)

    def load_joint_model_file(self, path, segment_map: dict | None = None, ask: bool = True) -> list[KinItem]:
        """Add a model file's joints. Their segments are matched to this tab's models (first to models whose
        coordinate system is that segment, then by surface file); models without a coordinate system get the
        segment's."""
        try:
            self._kin_models.pop(str(Path(path).resolve()), None)
            km = self._kin_model(path)
        except Exception as e:  # noqa: BLE001
            self._error("Couldn't open the model", str(e))
            return []
        joints = [j for j in km.joints
                  if km.segment(j.proximal).acs is not None and km.segment(j.distal).acs is not None]
        if not joints:
            self._error("No joints", "This file has no joints with both bones' axes defined. Add joints in the "
                                     "Model Builder (step 2) and save again.")
            return []
        if segment_map is None:
            smap = {}
            for m in self.state.models:                      # models already using this file's segments
                if m.has_axes and m.axes_path and Path(m.axes_path).resolve() == Path(path).resolve():
                    smap[m.axes_segment] = m.name
            for sname, mname in guess_segment_map(km, [m for m in self.state.models
                                                       if m.name not in smap.values()]).items():
                smap.setdefault(sname, mname)
            needed = {n for j in joints for n in (j.proximal, j.distal)}
            if ask and not needed <= set(smap):
                dlg = SegmentMapDialog(self, km, self.state.models, smap, Path(path))
                if dlg.exec() != QDialog.Accepted:
                    return []
                smap = dlg.values()
        else:
            smap = segment_map
        added = []
        for j in joints:
            dup = any(Path(x.model_path).resolve() == Path(path).resolve() and x.name == j.name and
                      x.model_map == {n: smap.get(n) for n in (j.proximal, j.distal)} for x in self.state.joints)
            if dup:
                continue                                      # already in this tab
            if j.proximal in smap and j.distal in smap:
                added.append(KinItem(Path(path), "joint", j.name, {n: smap[n] for n in (j.proximal, j.distal)},
                                     label=j.name, model=km))
        if not added:
            if not any(Path(x.model_path).resolve() == Path(path).resolve() for x in self.state.joints):
                self._error("Nothing to add", "The joints' bones weren't matched to bones in this tab.")
            else:
                self.refresh_all()
            return []
        for jl in added:                                     # give bones without axes the joint's segments
            for sname, mname in jl.model_map.items():
                m = self._model_named(mname)
                if m is not None and not m.has_axes:
                    m.axes_path, m.axes_segment, m.axes_model = Path(path), sname, km
        self.state.joints.extend(added)
        self.curve_joint = added[0]
        self.sel_kind, self.sel_obj = JOINT, added[0]
        self.refresh_all()
        for jl in added:
            self._check_joint_axes(jl)
        return added

    def _check_joint_axes(self, jl: KinItem) -> None:
        """Warn when a joint's bone shows a different coordinate system from the one the joint uses."""
        odd = []
        for sname, mname in jl.model_map.items():
            m = self._model_named(mname)
            if m is None:
                continue
            seg = jl.model.segment(sname)
            if m.has_axes and (Path(m.axes_path).resolve() != Path(jl.model_path).resolve() or m.axes_segment != sname):
                odd.append(f"{m.name} shows {m.axes_segment} from {Path(m.axes_path).name}; the joint uses {sname} "
                           f"from {jl.model_path.name}.")
            if seg.surface is not None and Path(seg.surface.path).name != Path(m.path).name:
                odd.append(f"{sname} was defined on {Path(seg.surface.path).name}, but {m.name} is {Path(m.path).name}.")
        if odd:
            self._error("Check the joint's bones", "\n".join(odd) + "\n\nThe joint always uses its own file's axes.")

    def _model_named(self, name):
        return next((m for m in self.state.models if m.name == name), None)

    def _poses(self, m: BoneModel, n: int) -> np.ndarray:
        if m.tracking is None:
            return np.repeat(np.eye(4)[None], n, axis=0)
        P = m.tracking.poses
        if len(P) < n:
            P = np.concatenate([P, np.full((n - len(P), 4, 4), np.nan)])
        return P[:n]

    @staticmethod
    def _has_reference(jl) -> bool:
        return isinstance(jl, KinItem) and jl.model is not None and jl.model.joint(jl.name).reference is not None

    def _kin_key(self, obj):
        return ("axes", id(obj)) if isinstance(obj, BoneModel) else ("joint", id(obj))

    def joint_values(self, obj, zero: bool = True) -> dict | None:
        """Values over the trial of a model's coordinate system (segment motion) or of a joint."""
        k = self._kin.get(self._kin_key(obj))
        if k is None:
            return None
        z = obj.axes_zero if isinstance(obj, BoneModel) else obj.zero_frame
        return zeroed(k, z) if zero else k

    def refresh_joints(self) -> None:
        from ..core.joints import JointModelError

        self._kin.clear()
        n = max(self.scene.n_frames, 1)
        for m in self.state.models:
            if m.has_axes and m.acs is not None:
                try:
                    self._kin[self._kin_key(m)] = m.axes_model.segment_kinematics(
                        m.axes_segment, self._poses(m, n), m.axes_reference)
                except JointModelError:
                    pass
        for jl in self.state.joints:
            models = [self._model_named(x) for x in jl.tab_models()]
            if any(mm is None or mm.tracking is None for mm in models):
                continue                  # untracked bones sit in unrelated model spaces: no joint values
            try:
                self._kin[self._kin_key(jl)] = jl.model.joint_kinematics(
                    jl.name, self._poses(models[0], n), self._poses(models[1], n), normalise=jl.use_reference)
            except JointModelError:
                continue
        live = [*self.state.joints, *[m for m in self.state.models if m.has_axes]]
        if self.curve_joint is not None and not any(o is self.curve_joint for o in live):
            self.curve_joint = None
        if self.curve_joint is None and live:
            self.curve_joint = live[0]
        self._update_curves()
        self._draw_joint_axes()

    def _update_curves(self) -> None:
        obj = self.curve_joint
        k = self.joint_values(obj) if obj is not None else None
        self.curves.setVisible(self.show_curves and k is not None)
        if k is not None:
            name = obj.name if isinstance(obj, BoneModel) else obj.label
            self.curves.set_data(k, f"{name} · {self.curves_title(obj)}")
            self.curves.set_frame(self.scene.frame)
        else:
            self.curves.set_data(None)

    def curves_title(self, obj) -> str:
        notes = []
        if isinstance(obj, BoneModel):
            ref = obj.axes_reference
            notes.append("about its own axes, relative to " +
                         ("the lab axes" if ref == "lab" else "its first tracked frame" if ref == "first"
                          else f"frame {int(ref) + 1}"))
            z = obj.axes_zero
        else:
            a, b = obj.tab_models()
            notes.append(f"{b} relative to {a}")
            if obj.use_reference and self._has_reference(obj):
                notes.append("relative to the reference pose")
            z = obj.zero_frame
        if z is not None:
            notes.append(f"zeroed at frame {z + 1}")
        return " · ".join(notes)

    def _draw_joint_axes(self) -> None:
        """Axes on bones: each model's own coordinate system, plus any a ticked joint uses that differ."""
        for entries in self._joint_actors.values():
            for actors, _m, _acs in entries:
                for a in actors:
                    self.plotter.remove_actor(a, render=False)
        self._joint_actors.clear()
        drawn = set()

        def draw(key, m, km, seg, acs):
            if m is None or acs is None or (id(m), id(km), seg) in drawn:
                return None
            drawn.add((id(m), id(km), seg))
            size = float(np.linalg.norm(np.ptp(np.asarray(m.mesh.points), axis=0))) if m.mesh is not None else 100
            self._joint_actors.setdefault(key, []).append((add_triad(self.plotter, acs, max(size * 0.3, 15)), m, acs))

        for m in self.state.models:
            if m.has_axes and m.axes_visible:
                draw(self._kin_key(m), m, m.axes_model, m.axes_segment, m.acs)
        for jl in self.state.joints:
            if jl.visible:
                for seg, tab_name in zip(jl.segment_names(), jl.tab_models()):
                    m = self._model_named(tab_name)
                    if m is not None and m.has_axes and not m.axes_visible:
                        continue                         # the user hid this bone's axes
                    draw(self._kin_key(jl), m, jl.model, seg, jl.model.segment(seg).acs)
        self._pose_joint_axes()

    def _pose_joint_axes(self) -> None:
        f = self.scene.frame
        for entries in self._joint_actors.values():
            for actors, m, acs in entries:
                pose = self.scene.model_pose(m, f)
                for a in actors:
                    a.SetVisibility(pose is not None and m.visible)
                    if pose is not None:
                        # the arrows are built at the ACS, so move them by the bone's pose
                        a.user_matrix = pose
        self.plotter.render()

    def _set_curve_joint(self, obj) -> None:
        self.curve_joint = obj
        self.show_curves = True
        self._update_curves()

    def _toggle_curves(self) -> None:
        self.show_curves = not self.show_curves
        self._update_curves()
        if self.sel_kind in (JOINT, JOINTS, AXES):
            self.show_properties()

    def _set_reference(self, obj, frame: int | None) -> None:
        if isinstance(obj, BoneModel):
            obj.axes_zero = frame
        else:
            obj.zero_frame = frame
        self._update_curves()
        if self.sel_obj is obj:
            self.show_properties()

    def _set_use_reference(self, jl: KinItem, on: bool) -> None:
        jl.use_reference = bool(on)
        self.refresh_joints()
        if self.sel_obj is jl:
            self.show_properties()

    def _set_seg_reference(self, m: BoneModel, ref) -> None:
        m.axes_reference = ref
        self.refresh_joints()
        if self.sel_obj is m:
            self.show_properties()

    def remap_joint(self, jl: KinItem) -> None:
        dlg = SegmentMapDialog(self, jl.model, self.state.models, jl.model_map, jl.model_path)
        for name, cb in dlg.combos.items():
            cb.setEnabled(cb.isEnabled() and name in jl.segment_names())
        if dlg.exec() == QDialog.Accepted:
            vals = dlg.values()
            if all(n in vals for n in jl.segment_names()):
                jl.model_map = {n: vals[n] for n in jl.segment_names()}
                self.refresh_all()
                self._check_joint_axes(jl)

    def remove_joint(self, jl: KinItem) -> None:
        self.state.joints = [j for j in self.state.joints if j is not jl]
        if self.sel_obj is jl:
            self.sel_kind, self.sel_obj = JOINTS, None
        self.refresh_all()

    def export_joint_csv(self, obj, path: str | None = None) -> None:
        k = self.joint_values(obj)
        if k is None:
            self._error("Nothing to export", "Its bones aren't all in this tab.")
            return
        if path is None:
            name = obj.name if isinstance(obj, BoneModel) else obj.label
            path, _ = QFileDialog.getSaveFileName(self, "Export values", f"{self.state.name}_{name}.csv",
                                                  "CSV (*.csv)")
            if not path:
                return
        write_csv(path, k, self.state.settings.fps, self.state.settings.curve_labels)

    def _frame_values(self, p, k, missing: str = "Its bones need to be in this tab, with axes.") -> None:
        p.section(f"Frame {self.scene.frame + 1}")
        if k is None:
            p.note(missing, "Warn")
            return
        f = min(self.scene.frame, len(k["angles"]) - 1)
        if np.isnan(k["angles"][f]).any():
            p.value("Values", "untracked")
            return
        la, lt = curve_labels(k, self.state.settings.curve_labels)
        for c, n, full, _col in la:
            v = k["angles"][f][c]
            p.value(n, f"{(0.0 if abs(v) < 0.05 else v):+.1f}°", tip=full)
        for c, n, full, _col in lt:
            v = k["translations"][f][c]
            p.value(n, f"{(0.0 if abs(v) < 0.05 else v):+.1f} mm", tip=full)

    def _curve_style_changed(self, style: str) -> None:
        self.state.set_tab_wide("curve_labels", style)
        if self.sel_kind in (JOINT, AXES):
            self.show_properties()

    def _props_axes(self, m: BoneModel) -> None:
        p = self.props
        p.clear(f"{m.name}: coordinate system")
        if not m.has_axes:
            p.note("No coordinate system yet. Add one from a model saved in the Model Builder: pick the file, and "
                   "the model's bone that matches this one is chosen (or you're asked). Its axes then show on the "
                   "bone, its motion about its own axes is plotted, and joints can use it.")
            p.buttons([("Load file…", lambda: self.add_axes(m))])
            return
        seg = m.axes_model.segment(m.axes_segment) if m.acs is not None else None
        p.value("Model", Path(m.axes_path).name, tip=str(m.axes_path))
        p.value("Model bone", m.axes_segment if seg is None else
                f"{seg.name} ({seg.stype.title.lower()}" + (f", {seg.side})" if seg.side != "none" else ")"))
        if seg is None:
            p.note("This bone isn't in the file any more, or has no axes.", "Warn")
        else:
            p.value("Axes from", J.axes_source_title(seg.axes_source))
            if seg.surface is not None:
                same = Path(seg.surface.path).name == Path(m.path).name
                p.value("Defined on", Path(seg.surface.path).name, kind="Value" if same else "Warn",
                        tip=None if same else "Not this model's file: the axes only fit the same surface.")
        p.buttons([("Load new file…", lambda: self.add_axes(m)), ("Refresh current file", lambda: self.refresh_axes(m))])
        p.buttons([("Remove", lambda: self.remove_axes(m))])
        p.section("Display")
        p.check("Axes", m.axes_visible, lambda on: self._toggle_axes(m, on), "Show the axes on the bone")
        p.check("Curves", self.show_curves and self.curve_joint is m,
                lambda on: self._set_curve_joint(m) if on else (self.show_curves and self._toggle_curves()),
                "Show this bone's motion")
        p.section("Motion relative to")
        cur = m.axes_reference if m.axes_reference in ("first", "lab") else "frame"
        items = [("first", "Its first tracked frame"),
                 ("frame", f"Frame {int(m.axes_reference) + 1}" if cur == "frame" else "The current frame"),
                 ("lab", "The lab (tracking) axes")]
        p.combo("Reference", items, cur, lambda v: self._set_seg_reference(m, self.scene.frame if v == "frame" else v))
        p.note("Rotations are about the bone's own axes (Z, then X, then Y); translations are the origin's movement "
               "along the reference axes.")
        p.section("Zero at a frame")
        p.value("Zero", f"frame {m.axes_zero + 1}" if m.axes_zero is not None else "none")
        p.buttons([("Zero at this frame", lambda: self._set_reference(m, self.scene.frame)),
                   ("Clear", lambda: self._set_reference(m, None))])
        self._frame_values(p, self.joint_values(m))
        p.buttons([("Export CSV…", lambda: self.export_joint_csv(m))])

    def _props_joints(self) -> None:
        p = self.props
        p.clear("Joints")
        p.note("Joints come from a file saved in the Model Builder. Each pairs two bones with a joint definition, "
               "optionally relative to the file's reference pose. Bones without a coordinate system get the joint's "
               "bones.")
        if self.state.joints:
            p.check("Curves", self.show_curves, lambda on: self.show_curves != on and self._toggle_curves(),
                    "Show curves under the view")
        p.buttons([("Add joint…", self.load_joint_model)])

    def _props_joint(self, jl: KinItem) -> None:
        p = self.props
        km = jl.model
        j = km.joint(jl.name)
        p.clear(jl.label)
        p.value("File", jl.model_path.name, tip=str(jl.model_path))
        p.value("Definition", j.jtype.title)
        p.section("Bones")
        for seg, name in zip(jl.segment_names(), jl.tab_models()):
            ok = self._model_named(name) is not None
            p.value(seg, name if ok else f"{name or 'not set'} (missing)", kind="Value" if ok else "Warn")
        p.buttons([("Change bones…", lambda: self.remap_joint(jl))])
        p.section("Display")
        p.check("Axes", jl.visible, lambda on: self._toggle_visible(jl, on), "Show the joint's axes on its bones")
        p.check("Curves", self.show_curves and self.curve_joint is jl,
                lambda on: self._set_curve_joint(jl) if on else (self.show_curves and self._toggle_curves()),
                "Show these curves")
        p.section("Reference pose")
        if j.reference is None:
            p.note("The file has no reference pose for this joint, so zero is anatomical (the bones' axes lined "
                   "up). Add one in the Model Builder, or zero at a frame below.")
        else:
            what = {"trial": "a neutral trial", "scan": "the scan pose", "scan_space": "the scan pose"}.get(
                j.reference.source.get("kind"), "set")
            p.check("Use it", jl.use_reference, lambda on: self._set_use_reference(jl, on),
                    f"Relative to the reference pose ({what})")
        p.section("Zero at a frame")
        p.value("Zero", f"frame {jl.zero_frame + 1}" if jl.zero_frame is not None else "none",
                tip="Values at this frame of the trial are subtracted.")
        p.buttons([("Zero at this frame", lambda: self._set_reference(jl, self.scene.frame)),
                   ("Clear", lambda: self._set_reference(jl, None))])
        self._frame_values(p, self.joint_values(jl), "Joint values need both bones in this tab, each with its "
                                                     "tracking loaded (right-click Tracking under a bone → Load file…).")
        p.buttons([("Export CSV…", lambda: self.export_joint_csv(jl)), ("Remove", lambda: self.remove_joint(jl))])

    def screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", f"{self.state.name}_frame{self.scene.frame + 1}.png",
                                              "PNG image (*.png)")
        if path:
            self.scene.screenshot(path)

    def export_video(self, path=None, options=None) -> bool:
        """Export the playback as an MP4: the 3D view and/or the X-ray camera views (Visualiser #15).

        Asks for the options (``video_export.VideoExportDialog``) and the file unless given.
        """
        from .video_export import ExportOptions, FrameComposer, VideoExportDialog

        n = self.scene.n_frames
        ncam = sum(c.calibration is not None and c.images is not None for c in self.state.cameras)
        if options is None:
            last = getattr(self, "_video_options", None)
            if last is not None:
                last.first, last.last = min(last.first, n - 1), min(last.last, n - 1)
            dlg = VideoExportDialog(self, n, ncam, last or ExportOptions(content="both" if ncam else "3d",
                                                                          last=n - 1))
            if dlg.exec() != QDialog.Accepted:
                return False
            options = dlg.options()
            self._video_options = options
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export video", f"{self.state.name}.mp4", "MP4 video (*.mp4)")
            if not path:
                return False
        import imageio.v2 as imageio

        frames = range(options.first, options.last + 1)
        prog = QProgressDialog("Exporting video…", "Cancel", 0, len(frames), self)
        prog.setWindowModality(Qt.WindowModal)
        start = self.scene.frame
        comp = FrameComposer(self, options)
        ok = True
        try:
            writer = imageio.get_writer(path, fps=options.fps, codec="libx264", quality=8, macro_block_size=1)
            try:
                for k, f in enumerate(frames):
                    if prog.wasCanceled():
                        ok = False
                        break
                    writer.append_data(comp.frame(f))
                    prog.setValue(k + 1)
                    QApplication.processEvents()
            finally:
                writer.close()
        except Exception as e:  # noqa: BLE001
            self._error("Video export failed", str(e))
            ok = False
        finally:
            comp.close()
            prog.close()
            self.scene.update_frame(start)
        return ok

    def close_tab(self) -> None:
        self.timer.stop()
        for cam in self.state.cameras:
            if cam.images is not None:
                cam.images.close()
        self.plotter.close()


TrialTab = VisualiserTab   # older name, kept for scripts that used it
