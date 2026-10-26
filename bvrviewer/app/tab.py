"""The Visualiser tab: navigator + properties on the left, the 3D view and timeline on the right."""

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

from ..core.auto_axes import AUTO_METHODS
from ..core.config import read_cfg
from ..core.models import PALETTE, BoneModel, ModelError, is_volume_file
from ..core.session import KinItem, TabState, XrayCamera, load_session, save_session
from ..core.tracking import TrackingError, load_tracking
from ..core.transforms import TransformFileError
from .dialogs import AddCameraDialog, FormatDialog, VoxelDialog
from .joint_panel import JointCurves, SegmentMapDialog, guess_segment_map, write_csv, zeroed
from .model_builder import add_triad
from .navigator import AXES, CAMERA, CAMERAS, JOINT, JOINTS, MODEL, MODELS, SCENE, TRACKING, Navigator, PropertySheet
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
    loadModel = Signal()
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
            (1, "Add a bone", "Right-click Bones (or use the button below). STL surfaces or TIF partial volumes; "
                              "the file name becomes the bone's name, which you can change. Or load a model from the "
                              "Model Builder to bring in all its bones, coordinate systems and joints at once."),
            (2, "Attach its tracking file", "Right-click the bone and choose Attach tracking. The .tra format "
                                            "is detected automatically."),
            (3, "Set how it looks", "Select the bone to change its colour, opacity and display in Properties."),
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
        add = QPushButton("+  Add bone (STL / TIF)")
        add.setObjectName("Primary")
        add.setMinimumHeight(40)
        add.clicked.connect(self.addModel.emit)
        lm = QPushButton("Load model…")
        lm.setMinimumHeight(40)
        lm.setToolTip("A model from the Model Builder: its bones, coordinate systems and joints")
        lm.clicked.connect(self.loadModel.emit)
        op = QPushButton("Open saved session…")
        op.setMinimumHeight(40)
        op.clicked.connect(self.openSession.emit)
        r.addWidget(add)
        r.addWidget(lm)
        r.addWidget(op)
        r.addStretch(1)
        v.addLayout(r)
        v.addWidget(label("Tip: select several files at once and each is added as a bone."))
        outer.addWidget(box)


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
        self.empty.loadModel.connect(self.load_full_model)
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
        self.plotter.enable_mesh_picking(self._picked, use_actor=True, show=False, show_message=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timeline.play.toggled.connect(self._play)
        self.timeline.frameChanged.connect(self._frame_changed)
        self.curves.frameClicked.connect(lambda f: self.timeline.slider.setValue(f))
        self.nav.itemSelected.connect(self._nav_selected)
        self.nav.menuRequested.connect(self._show_menu)
        self.nav.visibilityToggled.connect(self._toggle_visible)
        self.nav.axesToggled.connect(self._toggle_axes)
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
        if self.sel_kind in (MODEL, TRACKING, AXES) and not any(m is self.sel_obj for m in self.state.models):
            self.sel_kind, self.sel_obj = MODELS, None
        if self.sel_kind == CAMERA and not any(c is self.sel_obj for c in self.state.cameras):
            self.sel_kind, self.sel_obj = CAMERAS, None
        if self.sel_kind == JOINT and not any(j is self.sel_obj for j in self.state.joints):
            self.sel_kind, self.sel_obj = JOINTS, None
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
            add("Add bone…", self.add_models)
            add("Load model (bones, coordinate systems, joints)…", self.load_full_model)
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
                add("Replace coordinate system…" if m.has_axes else "Add coordinate system…",
                    lambda: self.add_axes(m))
                if m.has_axes:
                    add("Remove coordinate system", lambda: self.remove_axes(m))
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
            add("Replace coordinate system…" if m.has_axes else "Add coordinate system…", lambda: self.add_axes(m))
            if m.has_axes:
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
        else:
            self._props_models()
        p.finish()

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
        p.slider("Opacity", m.opacity, 0, 100, 100, _fmt_pct, lambda v: self._set_model(m, "opacity", v, nav=False))
        p.check("Visible", m.visible, lambda on: self._toggle_visible(m, on))
        p.combo("Display as", [("surface", "Surface"), ("volume", "Volume", m.image_data is not None),
                               ("both", "Surface and volume", m.image_data is not None)],
                m.display, lambda v: self._set_model(m, "display", v),
                tip=None if m.image_data is not None else "Volume display needs a TIF surface.")
        p.section("Motion")
        if m.tracking:
            p.value("Tracking", f"{m.tracking.path.name} · {m.tracking.n_frames} frames", tip=str(m.tracking.path))
        else:
            p.value("Tracking", "None. Shown untracked, in its own model space.", kind="Warn")
        p.check("Hold still", self.state.settings.hold_still == m.name, lambda on: self.hold_still(m, on),
                caption="Others move relative to it", enabled=m.tracking is not None)
        p.buttons([("Replace tracking…" if m.tracking else "Attach tracking…", lambda: self.attach_tracking(m))])
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
            p.note("No tracking file attached, so this bone stays untracked, in its own model space.", "Warn")
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
                        m.tracking = load_tracking(m.tracking_path, m.tracking_volume, m.tracking_overrides)
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
            self.state.models.append(m)
            self.scene.add_model(m)
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
                    else load_tracking(new.tracking_path, new.tracking_volume, new.tracking_overrides)
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
                        "\n\nRight-click the bone and choose Tracking format… to correct it.")

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
            for m in self.state.models:                  # coordinate systems on the models
                if m.has_axes:
                    try:
                        m.axes_model = self._kin_model(m.axes_path)
                        m.axes_model.segment(m.axes_segment)
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{m.name} coordinate system: {e}")
            for jl in st.joints:
                if jl.kind == "segment":                # older sessions listed segments separately
                    m = self._model_named(jl.model_map.get(jl.name))
                    if m is not None and not m.has_axes:
                        try:
                            self.set_axes(m, jl.model_path, jl.name, warn=False)
                            m.axes_visible = jl.visible
                        except Exception as e:  # noqa: BLE001
                            problems.append(f"{m.name} coordinate system: {e}")
                    continue
                try:
                    jl.model = self._kin_model(jl.model_path)
                    for n in jl.segment_names():          # checks the segment / joint still exists
                        jl.model.segment(n)
                    jl.label = jl.label or jl.name
                    self.state.joints.append(jl)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{jl.name or jl.model_path.name}: {e}")
            self.refresh_all()
            self.scene.update_frame(0)
            if problems:
                self._error("Some radiographs or joints couldn't be loaded", "\n".join(problems))

        if st.models:
            self._build_models(st.models, then=add_cams)
        else:
            add_cams()

    # ---------------------------------------------------------------- coordinate systems and joints
    MODEL_FILE_FILTER = "Model from the Model Builder (*.bvrmodel.json *.json)"

    def _kin_model(self, path):
        from ..core.joints import KinematicModel

        key = str(Path(path).resolve())
        if key not in self._kin_models:
            self._kin_models[key] = KinematicModel.load(path, build_surfaces=False)
        return self._kin_models[key]

    def add_axes(self, m: BoneModel) -> None:
        """Right-click a bone → Add coordinate system: one of the bones of a Model Builder model."""
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
        write_csv(path, k, self.state.settings.fps)

    def _frame_values(self, p, k, missing: str = "Its bones need to be in this tab, with axes.") -> None:
        p.section(f"Frame {self.scene.frame + 1}")
        if k is None:
            p.note(missing, "Warn")
            return
        f = min(self.scene.frame, len(k["angles"]) - 1)
        if np.isnan(k["angles"][f]).any():
            p.value("Values", "untracked")
            return
        for n, v in zip(k["short"], k["angles"][f]):
            p.value(n, f"{(0.0 if abs(v) < 0.05 else v):+.1f}°")
        for n, v in zip(k["t_short"], k["translations"][f]):
            p.value(n, f"{(0.0 if abs(v) < 0.05 else v):+.1f} mm")

    def _props_axes(self, m: BoneModel) -> None:
        p = self.props
        p.clear(f"{m.name}: coordinate system")
        if not m.has_axes:
            p.note("No coordinate system yet. Add one from a model saved in the Model Builder: pick the file, and "
                   "the model's bone that matches this one is chosen (or you're asked). Its axes then show on the "
                   "bone, its motion about its own axes is plotted, and joints can use it.")
            p.buttons([("Add coordinate system…", lambda: self.add_axes(m))])
            return
        seg = m.axes_model.segment(m.axes_segment) if m.acs is not None else None
        p.value("Model", Path(m.axes_path).name, tip=str(m.axes_path))
        p.value("Model bone", m.axes_segment if seg is None else
                f"{seg.name} ({seg.stype.title.lower()}" + (f", {seg.side})" if seg.side != "none" else ")"))
        if seg is None:
            p.note("This bone isn't in the file any more, or has no axes.", "Warn")
        else:
            p.value("Axes from", "Landmarks" if seg.axes_source == "landmarks" else
                    AUTO_METHODS[seg.axes_source].title)
            if seg.surface is not None:
                same = Path(seg.surface.path).name == Path(m.path).name
                p.value("Defined on", Path(seg.surface.path).name, kind="Value" if same else "Warn",
                        tip=None if same else "Not this model's file: the axes only fit the same surface.")
        p.buttons([("Replace…", lambda: self.add_axes(m)), ("Remove", lambda: self.remove_axes(m))])
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
                                                     "tracking attached (right-click a bone → Attach tracking).")
        p.buttons([("Export CSV…", lambda: self.export_joint_csv(jl)), ("Remove", lambda: self.remove_joint(jl))])

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


TrialTab = VisualiserTab   # older name, kept for scripts that used it
