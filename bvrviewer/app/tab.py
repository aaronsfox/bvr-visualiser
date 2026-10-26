"""One tab: navigator + properties on the left, the 3D view and timeline on the right."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
    QProgressDialog, QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from ..core.config import read_cfg
from ..core.models import PALETTE, BoneModel, ModelError, is_volume_file
from ..core.session import TabState, XrayCamera, load_session, save_session
from ..core.tracking import TrackingError, load_tracking
from ..core.transforms import TransformFileError
from .dialogs import AddCameraDialog, FormatDialog, VoxelDialog
from .navigator import CAMERA, CAMERAS, MODEL, MODELS, SCENE, TRACKING, Navigator, PropertySheet
from .panels import Timeline, h2, label
from .scene import SceneController

MODEL_FILTER = "Bone models (*.stl *.STL *.tif *.tiff *.TIF *.obj *.ply *.vtk *.vtp);;All files (*)"


class _Worker(QObject):
    done = Signal(object, object)       # result, error text

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:  # noqa: BLE001 - shown to the user
            msg = str(e) if isinstance(e, (ModelError, TrackingError, TransformFileError, ValueError, OSError)) \
                else f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}"
            self.done.emit(None, msg)


class _Relay(QObject):
    """Lives on the UI thread so worker results are handled there (VTK/OpenGL must stay on the UI thread)."""

    def __init__(self, fn, parent):
        super().__init__(parent)
        self.fn = fn

    @Slot(object, object)
    def deliver(self, result, err):
        self.fn(result, err)


class EmptyState(QFrame):
    addModel = Signal()
    openSession = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Empty")
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        box = QWidget()
        box.setMaximumWidth(560)
        v = QVBoxLayout(box)
        v.setSpacing(18)
        title = QLabel("Build up this tab one bone at a time")
        title.setObjectName("Big")
        title.setWordWrap(True)
        v.addWidget(title)
        v.addWidget(label("Everything in the tab is listed in the navigator on the left. Right-click an item "
                          "there for its actions, and select it to edit its properties below."))
        for n, head, body in [
            (1, "Add a model", "Right-click Models (or use the button below). STL surfaces or TIF partial volumes; "
                               "the file name becomes the bone's name, which you can change."),
            (2, "Attach its tracking file", "Right-click the model and choose Attach tracking. The .tra format "
                                            "is detected automatically."),
            (3, "Set how it looks", "Select the model to change its colour, opacity and display in Properties."),
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
        add = QPushButton("+  Add model (STL / TIF)")
        add.setObjectName("Primary")
        add.setMinimumHeight(40)
        add.clicked.connect(self.addModel.emit)
        op = QPushButton("Open saved session…")
        op.setMinimumHeight(40)
        op.clicked.connect(self.openSession.emit)
        r.addWidget(add)
        r.addWidget(op)
        r.addStretch(1)
        v.addLayout(r)
        v.addWidget(label("Tip: select several files at once and each is added as a model."))
        outer.addWidget(box)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class TrialTab(QWidget):
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
        cv.addWidget(self.view_bar)

        self.stack = QStackedWidget()
        self.empty = EmptyState()
        self.empty.addModel.connect(self.add_models)
        self.empty.openSession.connect(self.open_session)
        self.stack.addWidget(self.empty)
        self.plotter = QtInteractor(self.stack)
        self.stack.addWidget(self.plotter.interactor)
        cv.addWidget(self.stack, 1)
        self.hint = QWidget()
        self.hint.setObjectName("Panel")
        hb = QHBoxLayout(self.hint)
        hb.setContentsMargins(10, 4, 10, 4)
        self.frame_info = label("", "Mono", wrap=False)
        hb.addWidget(self.frame_info, 1)
        hb.addWidget(label("Drag: rotate · Shift-drag: pan · Scroll: zoom · Right-click: select bone", wrap=False))
        cv.addWidget(self.hint)
        self.timeline = Timeline(self.state.settings)
        cv.addWidget(self.timeline)
        split.addWidget(center)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1120])
        root.addWidget(split)

        self.scene = SceneController(self.plotter, self.state)
        self.plotter.enable_mesh_picking(self._picked, use_actor=True, show=False, show_message=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timeline.play.toggled.connect(self._play)
        self.timeline.frameChanged.connect(self._frame_changed)
        self.nav.itemSelected.connect(self._nav_selected)
        self.nav.menuRequested.connect(self._show_menu)
        self.nav.visibilityToggled.connect(self._toggle_visible)
        self.nav.renamed.connect(self._rename)
        self.nav.activated.connect(lambda _k, cam: self.set_xray_view(self.state.cameras.index(cam)))
        for key, fn in [(Qt.Key_Space, lambda: self.timeline.play.toggle()),
                        (Qt.Key_Right, lambda: self.timeline.step(1)), (Qt.Key_Left, lambda: self.timeline.step(-1))]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._negative = False
        self.refresh_all()

    # ---------------------------------------------------------------- helpers
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
        has = bool(self.state.models)
        self.stack.setCurrentIndex(1 if has else 0)
        self.view_bar.setVisible(has)
        self.hint.setVisible(has)
        self.timeline.set_count(self.scene.n_frames)
        # (identity checks: models and cameras are dataclasses, so `in` would compare their fields)
        if self.sel_kind in (MODEL, TRACKING) and not any(m is self.sel_obj for m in self.state.models):
            self.sel_kind, self.sel_obj = MODELS, None
        if self.sel_kind == CAMERA and not any(c is self.sel_obj for c in self.state.cameras):
            self.sel_kind, self.sel_obj = CAMERAS, None
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
    def _nav_selected(self, kind: str, obj) -> None:
        self.sel_kind, self.sel_obj = kind, obj
        if kind in (MODEL, TRACKING) and obj is not self.selected:
            self.select(obj, from_nav=True)
        self.show_properties()

    def _toggle_visible(self, obj, on: bool) -> None:
        obj.visible = on
        if isinstance(obj, BoneModel):
            self._appearance(obj)
            self.nav.refresh_obj(MODEL, obj)
        else:
            self.nav.refresh_obj(CAMERA, obj)
            self.scene.update_frame(self.scene.frame)
            self._update_frame_info()
        if obj is self.sel_obj:
            self.show_properties()

    def _rename(self, obj, name: str) -> None:
        if name == obj.name:
            return
        old, obj.name = obj.name, name
        if isinstance(obj, BoneModel):
            if self.state.settings.hold_still == old:
                self.state.settings.hold_still = name
            self.nav.refresh_obj(MODEL, obj)
        else:
            self.nav.refresh_obj(CAMERA, obj)
            self._refresh_xray_buttons()
        self._update_frame_info()
        if obj is self.sel_obj:
            self.props.title.setText(name)

    def _show_menu(self, kind: str, obj, pos) -> None:
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

        if kind == MODELS:
            add("Add model…", self.add_models)
            menu.addSeparator()
            add("Show all", lambda: self._show_all(True), enabled=bool(self.state.models))
            add("Hide all", lambda: self._show_all(False), enabled=bool(self.state.models))
        elif kind in (MODEL, TRACKING):
            m: BoneModel = obj
            add("Replace tracking…" if m.tracking else "Attach tracking…", lambda: self.attach_tracking(m))
            if m.tracking:
                add("Tracking format…", lambda: self.edit_format(m))
                add("Detach tracking", lambda: self.detach_tracking(m))
            if kind == MODEL:
                menu.addSeparator()
                add("Hold this bone still", lambda: self.hold_still(m, self.state.settings.hold_still != m.name),
                    enabled=m.tracking is not None, checked=self.state.settings.hold_still == m.name)
                add("Hide" if m.visible else "Show", lambda: self._toggle_visible(m, not m.visible))
                add("Rename", lambda: self.nav.start_rename(MODEL, m))
                add("Use file name as name", lambda: self._rename(m, m.path.stem))
                menu.addSeparator()
                add("Replace model file…", lambda: self.replace_model(m))
                if m.is_volume:
                    add("Voxel size / surface…", lambda: self.edit_voxel(m))
                elif m.tfm_path:
                    add("Remove .tfm transform", lambda: self.clear_tfm(m))
                else:
                    add("Apply .tfm transform…", lambda: self.load_tfm(m))
                menu.addSeparator()
                add("Remove model", lambda: self.remove_model(m))
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
        elif kind == TRACKING and obj is not None:
            self._props_tracking(obj)
        elif kind == CAMERA and obj is not None:
            self._props_camera(obj)
        elif kind == CAMERAS:
            self._props_cameras()
        elif kind == SCENE:
            self._props_scene()
        else:
            self._props_models()
        p.finish()

    def _props_models(self) -> None:
        p = self.props
        p.clear("Models")
        n = len(self.state.models)
        tracked = sum(1 for m in self.state.models if m.tracking)
        p.value("Models", str(n))
        p.value("With tracking", f"{tracked} of {n}" if n else "–")
        if n:
            p.value("Frames", str(self.scene.n_frames))
        p.note("Select a model in the navigator to edit it. Right-click a model for its actions.")
        p.buttons([("Add model…", self.add_models)] +
                  ([("Show all", lambda: self._show_all(True)), ("Hide all", lambda: self._show_all(False))]
                   if n else []))

    def _props_model(self, m: BoneModel) -> None:
        p = self.props
        p.clear(m.name)
        p.section("Appearance")
        p.line("Name", m.name, lambda v: self._rename(m, v))
        p.color("Colour", m.color, lambda c: self._set_model(m, "color", c))
        p.slider("Opacity", m.opacity, 0, 100, 100, _fmt_pct, lambda v: self._set_model(m, "opacity", v, nav=False))
        p.check("Visible", m.visible, lambda on: self._toggle_visible(m, on))
        p.combo("Display as", [("surface", "Surface"), ("volume", "Volume", m.image_data is not None),
                               ("both", "Surface and volume", m.image_data is not None)],
                m.display, lambda v: self._set_model(m, "display", v),
                tip=None if m.image_data is not None else "Volume display needs a TIF model.")
        p.section("Motion")
        if m.tracking:
            p.value("Tracking", f"{m.tracking.path.name} · {m.tracking.n_frames} frames", tip=str(m.tracking.path))
        else:
            p.value("Tracking", "None. Shown in its model pose.", kind="Warn")
        p.check("Hold still", self.state.settings.hold_still == m.name, lambda on: self.hold_still(m, on),
                caption="Others move relative to it", enabled=m.tracking is not None)
        p.buttons([("Replace tracking…" if m.tracking else "Attach tracking…", lambda: self.attach_tracking(m))])
        p.section("Model file")
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
        p.buttons([("Remove model", lambda: self.remove_model(m))], danger="Remove model")

    def _props_tracking(self, m: BoneModel) -> None:
        p = self.props
        t = m.tracking
        p.clear(f"Tracking · {m.name}")
        if t is None:
            p.note("No tracking file attached, so this model stays in its model pose.", "Warn")
            p.buttons([("Attach tracking…", lambda: self.attach_tracking(m))])
            return
        p.section("Tracking file")
        p.value("File", t.path.name, tip=str(t.path))
        p.value("Frames", str(t.n_frames))
        gaps = t.gaps()
        if gaps:
            shown = ", ".join(f"{a + 1}–{b + 1}" if b > a else f"{a + 1}" for a, b in gaps[:6])
            p.value("Untracked", f"{t.n_untracked} frames ({shown}{' …' if len(gaps) > 6 else ''})", kind="Warn")
        else:
            p.value("Untracked", "None")
        p.section("Format")
        p.value("Detected", t.fmt.describe())
        if t.fmt.n_volumes > 1:
            p.value("Bone in file", f"Volume {t.volume_index + 1} of {t.fmt.n_volumes}")
        for w in t.warnings:
            p.note(w, "Warn")
        p.buttons([("Format…", lambda: self.edit_format(m)), ("Replace…", lambda: self.attach_tracking(m)),
                   ("Detach", lambda: self.detach_tracking(m))])

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
        p.value("Folder", cam.images.source.name, tip=str(cam.images.source))
        p.combo("Frame 1 image", [(k, cam.images.label(k)) for k in range(n)], cam.first_image,
                lambda k: self._set_first_image(cam, k),
                tip="Which radiograph matches the first tracked frame.")
        shown = n - cam.first_image
        if self.state.models and shown != self.scene.n_frames:
            p.note(f"{shown} images from here for {self.scene.n_frames} tracked frames.", "Warn")
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
        p.slider("Image opacity", s.xray_opacity, 0, 100, 100, _fmt_pct, lambda v: self._set("xray_opacity", v))
        p.slider("Distance behind", s.xray_distance, 0, 800, 1, lambda x: f"{x:.0f} mm",
                 lambda v: self._set("xray_distance", v), on_release=True)
        p.section("Through-X-ray views")
        p.slider("Bone opacity", s.xray_bone_opacity, 0, 100, 100, _fmt_pct, lambda v: self._set("xray_bone_opacity", v))
        p.check("Wireframe bones", s.xray_outline, lambda on: self._set("xray_outline", on), caption="Draw as wireframe")
        p.section("Image")
        p.slider("Brightness", s.brightness, -100, 100, 100, lambda x: f"{x:+.2f}", lambda v: self._set("brightness", v))
        p.slider("Contrast", s.contrast, 20, 300, 100, lambda x: f"{x:.1f}×", lambda v: self._set("contrast", v))
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
        setattr(self.state.settings, key, value)
        self._settings_changed(key)

    def _set_model(self, m: BoneModel, key: str, value, nav: bool = True) -> None:
        setattr(m, key, value)
        self._appearance(m)
        if nav:
            self.nav.refresh_obj(MODEL, m)

    def _set_camera(self, cam: XrayCamera, key: str, value) -> None:
        setattr(cam, key, value)
        self.scene.update_frame(self.scene.frame)

    def _set_first_image(self, cam: XrayCamera, k: int) -> None:
        cam.first_image = int(k)
        cam.images.prefetch(range(cam.first_image, len(cam.images)))
        self.scene.update_frame(self.scene.frame)
        self._update_frame_info()

    # ---------------------------------------------------------------- models
    def add_models(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add bone models", "", MODEL_FILTER)
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
                        m.tracking = load_tracking(m.tracking_path, m.tracking_volume, m.tracking_overrides)
                    built.append(m)
                except (ModelError, TrackingError, TransformFileError, OSError, ValueError) as e:
                    errors.append(f"{m.path.name}: {e}")
            return built, errors

        names = ", ".join(m.path.name for m in models[:3]) + (" …" if len(models) > 3 else "")
        self._run(f"Loading {names}", work, lambda res, err: self._models_built(res, err, then))

    def _models_built(self, res, err, then) -> None:
        if err:
            self._error("Couldn't load models", err)
            return
        built, errors = res
        first = not self.state.models
        for m in built:
            self.state.models.append(m)
            self.scene.add_model(m)
        if built:
            self.select(built[-1], refresh=False)
            self.sel_kind, self.sel_obj = MODEL, built[-1]
        if errors:
            self._error("Some models couldn't be loaded", "\n\n".join(errors))
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
        path, _ = QFileDialog.getOpenFileName(self, "Replace model", str(m.path.parent), MODEL_FILTER)
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
            if new.tracking_path:
                new.tracking = old.tracking if old.tracking_path == new.tracking_path and \
                    old.tracking_overrides == new.tracking_overrides and old.tracking_volume == new.tracking_volume \
                    else load_tracking(new.tracking_path, new.tracking_volume, new.tracking_overrides)
            return new

        def done(res, err):
            if err:
                self._error("Couldn't update the model", err)
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
        if QMessageBox.question(self, "Remove model", f"Remove {m.name} from this tab?") != QMessageBox.Yes:
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
    def attach_tracking(self, m: BoneModel) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Tracking file for {m.name}", str(m.path.parent),
                                              "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            t = load_tracking(path)
            vol = 0
            if t.fmt.n_volumes > 1:
                items = [f"Volume {i + 1}" for i in range(t.fmt.n_volumes)]
                choice, ok = QInputDialog.getItem(self, "Which bone?",
                                                  f"{Path(path).name} holds {t.fmt.n_volumes} bones side by side.\n"
                                                  f"Which one is {m.name}?", items, 0, False)
                if not ok:
                    return
                vol = items.index(choice)
                t = load_tracking(path, vol)
        except (TrackingError, OSError) as e:
            self._error("Couldn't read the tracking file", str(e))
            return
        m.tracking_path, m.tracking_volume, m.tracking_overrides, m.tracking = Path(path), vol, {}, t
        self._tracking_changed(m)
        if t.warnings:
            self._error("Check the tracking format", "\n".join(t.warnings) +
                        "\n\nRight-click the model and choose Tracking format… to correct it.")

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
        dlg = FormatDialog(self, m.tracking.fmt, m.tracking_volume)
        if dlg.exec() != QDialog.Accepted:
            return
        overrides, vol = dlg.values()
        try:
            t = load_tracking(m.tracking_path, vol, overrides)
        except TrackingError as e:
            self._error("That format doesn't fit this file", str(e))
            return
        m.tracking, m.tracking_volume, m.tracking_overrides = t, vol, overrides
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
            self.state.cameras.append(dlg.result_camera)
            self.scene.add_camera(dlg.result_camera)
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
        for cam in cams:
            self.state.cameras.append(cam)
            self.scene.add_camera(cam)
        if cams:
            self.sel_kind, self.sel_obj = CAMERAS, None
        self.refresh_all()
        self.scene.update_frame(self.scene.frame)
        msg = f"Added {len(cams)} camera(s) from {Path(path).name}."
        counts = {len(c.images) for c in cams}
        if cams and self.state.models and any(n != self.scene.n_frames for n in counts):
            msg += f"\n\nImage counts ({', '.join(map(str, sorted(counts)))}) differ from the {self.scene.n_frames} " \
                   "tracked frames. Select a camera to choose which image is frame 1."
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
            save_session(self.state, path)

    def open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open saved session", "", "Session (*.json)")
        if path:
            self.load_session_file(path)

    def load_session_file(self, path: str) -> None:
        try:
            st = load_session(path)
        except Exception as e:  # noqa: BLE001
            self._error("Couldn't open the session", str(e))
            return
        self.state.name = st.name
        self.titleChanged.emit(st.name)
        for k, v in st.settings.__dict__.items():
            setattr(self.state.settings, k, v)
        self._sync_timeline_settings()
        cams = st.cameras

        def add_cams():
            problems = []
            for cam in cams:
                try:
                    cam.build()
                    self.state.cameras.append(cam)
                    self.scene.add_camera(cam)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{cam.name}: {e}")
            self.refresh_all()
            self.scene.update_frame(0)
            if problems:
                self._error("Some radiographs couldn't be loaded", "\n".join(problems))

        if st.models:
            self._build_models(st.models, then=add_cams)
        else:
            add_cams()

    def screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", f"{self.state.name}_frame{self.scene.frame + 1}.png",
                                              "PNG image (*.png)")
        if path:
            self.scene.screenshot(path)

    def export_video(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export video", f"{self.state.name}.mp4", "MP4 video (*.mp4)")
        if not path:
            return
        fps, ok = QInputDialog.getInt(self, "Video frame rate", "Frames per second in the video:", 30, 1, 120)
        if not ok:
            return
        import imageio.v2 as imageio

        n = self.scene.n_frames
        prog = QProgressDialog("Exporting video…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModal)
        start = self.scene.frame
        try:
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
            for f in range(n):
                if prog.wasCanceled():
                    break
                self.scene.update_frame(f)
                img = self.plotter.screenshot(return_img=True)
                h, w = img.shape[:2]
                writer.append_data(img[: h - h % 2, : w - w % 2])
                prog.setValue(f + 1)
                QApplication.processEvents()
            writer.close()
        except Exception as e:  # noqa: BLE001
            self._error("Video export failed", str(e))
        prog.close()
        self.scene.update_frame(start)

    def close_tab(self) -> None:
        self.timer.stop()
        for cam in self.state.cameras:
            if cam.images is not None:
                cam.images.close()
        self.plotter.close()
