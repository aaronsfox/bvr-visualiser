"""One tab: its models, tracking, radiographs, 3D view and timeline."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QProgressDialog,
    QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from ..core.config import read_cfg
from ..core.models import PALETTE, BoneModel, ModelError, is_volume_file
from ..core.session import TabState, XrayCamera, load_session, save_session
from ..core.tracking import TrackingError, load_tracking
from ..core.transforms import TransformFileError
from .dialogs import AddCameraDialog, FormatDialog, VoxelDialog
from .panels import ModelRow, ModelsPanel, OptionsPanel, Timeline, label
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
        v.addWidget(label("Each model gets its own tracking file from Autoscoper. Add as many models as you need, "
                          "then press play."))
        for n, head, body in [
            (1, "Add a model", "An STL surface or a TIF partial volume. Its file name becomes the bone's name, "
                               "which you can change."),
            (2, "Set how it looks", "Colour, opacity, and surface or volume display."),
            (3, "Attach its tracking file", "The .tra exported for that bone. The format is detected automatically."),
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


class TrialTab(QWidget):
    titleChanged = Signal(str)

    def __init__(self, state: TabState | None = None):
        super().__init__()
        self.state = state or TabState()
        self.cfg = None                     # last .cfg used, for voxel sizes
        self._threads: list = []
        self._busy: QProgressDialog | None = None
        self.selected: BoneModel | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)
        self.models_panel = ModelsPanel()
        self.models_panel.setMinimumWidth(370)
        split.addWidget(self.models_panel)

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
        tip = label("Drag: rotate · Shift-drag: pan · Scroll: zoom · Right-click: select bone", wrap=False)
        hb.addWidget(tip)
        cv.addWidget(self.hint)
        split.addWidget(center)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1120])
        root.addWidget(split, 1)
        self.timeline = Timeline(self.state.settings)
        root.addWidget(self.timeline)
        # Options live in a foldable, tabbed panel along the bottom (Scene, Radiographs).
        self._root = root
        self.settings_panel = self._make_settings_panel()
        root.addWidget(self.settings_panel)

        self.scene = SceneController(self.plotter, self.state)
        self.plotter.enable_mesh_picking(self._picked, use_actor=True, show=False, show_message=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timeline.play.toggled.connect(self._play)
        self.timeline.frameChanged.connect(self._frame_changed)
        self.models_panel.addModel.connect(self.add_models)
        c = self.models_panel.card
        c.replaceModel.connect(self.replace_model)
        c.loadTfm.connect(self.load_tfm)
        c.clearTfm.connect(self.clear_tfm)
        c.editVoxel.connect(self.edit_voxel)
        c.attachTracking.connect(self.attach_tracking)
        c.detachTracking.connect(self.detach_tracking)
        c.editFormat.connect(self.edit_format)
        c.displayChanged.connect(self._appearance)
        c.holdStill.connect(self.hold_still)
        c.remove.connect(self.remove_model)
        c.resetName.connect(self._reset_name)
        for key, fn in [(Qt.Key_Space, lambda: self.timeline.play.toggle()),
                        (Qt.Key_Right, lambda: self.timeline.step(1)), (Qt.Key_Left, lambda: self.timeline.step(-1))]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self._negative = False
        self.refresh_all()

    def _make_settings_panel(self, collapsed: bool = False, page: int = 0) -> OptionsPanel:
        p = OptionsPanel(self.state.settings, collapsed=collapsed, page=page)
        p.changed.connect(self._settings_changed)
        p.addCamera.connect(self.add_camera)
        p.importCfg.connect(self.import_cfg)
        p.editCamera.connect(self.edit_camera)
        p.removeCamera.connect(self.remove_camera)
        return p

    def _rebuild_settings_panel(self) -> None:
        """Recreate the settings panel so it shows settings loaded from a session."""
        old = self.settings_panel
        self.settings_panel = self._make_settings_panel(old.collapsed, old.stack.currentIndex())
        self._root.replaceWidget(old, self.settings_panel)
        old.setParent(None)
        old.deleteLater()
        self.timeline.loop.setChecked(self.state.settings.loop)
        i = self.timeline.speed.findData(self.state.settings.speed)
        if i >= 0:
            self.timeline.speed.setCurrentIndex(i)

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
        has = bool(self.state.models)
        self.stack.setCurrentIndex(1 if has else 0)
        self.view_bar.setVisible(has)
        self.hint.setVisible(has)
        self.timeline.set_count(self.scene.n_frames)
        self.models_panel.set_selected(self.selected)
        self.models_panel.card.show_model(self.selected, self.state.settings.hold_still)
        for r in self.models_panel.rows:
            r.refresh()
        self._refresh_coverage()
        self._refresh_xray_buttons()
        self.settings_panel.set_cameras(self.state.cameras)
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
            row = ModelRow(m)
            row.selected.connect(self.select)
            row.changed.connect(self._appearance)
            row.renamed.connect(self._renamed)
            row.attachTracking.connect(self.attach_tracking)
            self.models_panel.add_row(row)
        if built:
            self.select(built[-1], refresh=False)
        if errors:
            self._error("Some models couldn't be loaded", "\n\n".join(errors))
        self.refresh_all()
        self.scene.apply_settings()
        if first and built:
            self.set_view("3d")
        if then:
            then()

    def select(self, m: BoneModel | None, refresh: bool = True) -> None:
        self.selected = m
        self.scene.selected = m
        self.scene.rebuild_trail()
        if refresh:
            self.models_panel.set_selected(m)
            self.models_panel.card.show_model(m, self.state.settings.hold_still)
            self._refresh_coverage()
            self.scene.update_frame(self.scene.frame)

    def _picked(self, actor) -> None:
        for m in self.state.models:
            e = self.scene._bone_actors.get(id(m))
            if e and actor in (e["surface"], e["volume"]):
                self.select(m)
                return

    def _appearance(self, m: BoneModel) -> None:
        self.scene.refresh_model_style(m)
        self.scene.update_frame(self.scene.frame)
        if m is self.selected:
            self.models_panel.card.show_model(m, self.state.settings.hold_still)

    def _renamed(self, old: str, new: str) -> None:
        if self.state.settings.hold_still == old:
            self.state.settings.hold_still = new
            self._update_frame_info()

    def _reset_name(self, m: BoneModel) -> None:
        old = m.name
        m.name = m.path.stem
        if self.state.settings.hold_still == old:
            self.state.settings.hold_still = m.name
        row = self.models_panel.row_for(m)
        if row:
            row.name.setText(m.name)
        self.models_panel.card.show_model(m, self.state.settings.hold_still)

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
            row = self.models_panel.row_for(old)
            row.model = res
            self.select(res, refresh=False)
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
        self.models_panel.remove_row(m)
        if self.state.settings.hold_still == m.name:
            self.state.settings.hold_still = None
        self.select(self.state.models[-1] if self.state.models else None, refresh=False)
        self.refresh_all()
        self.scene.frame = min(self.scene.frame, self.scene.n_frames - 1)
        self.scene.apply_settings()

    def hold_still(self, m: BoneModel, on: bool) -> None:
        self.state.settings.hold_still = m.name if on else None
        self.scene.apply_settings()          # grid follows the new frame of reference
        self._update_frame_info()
        self.scene.zoom_to_bones()

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
                        "\n\nUse “Format…” on the selected model to correct it.")

    def _tracking_changed(self, m: BoneModel) -> None:
        if self.state.cameras:
            self.scene.reposition_cameras()
        self.select(m, refresh=False)
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

    # ---------------------------------------------------------------- radiographs
    def add_camera(self) -> None:
        dlg = AddCameraDialog(self, self.scene.n_frames, self._preview_bones(), index=len(self.state.cameras) + 1)
        if dlg.exec() == QDialog.Accepted and dlg.result_camera:
            self.state.cameras.append(dlg.result_camera)
            self.scene.add_camera(dlg.result_camera)
            self.refresh_all()
            self.settings_panel.show_page(1)
            self.scene.update_frame(self.scene.frame)

    def edit_camera(self, i: int) -> None:
        old = self.state.cameras[i]
        dlg = AddCameraDialog(self, self.scene.n_frames, self._preview_bones(), existing=old)
        if dlg.exec() == QDialog.Accepted and dlg.result_camera:
            if dlg.result_camera.images is not old.images:
                self.scene.forget_camera(old)
            else:
                self.scene.remove_camera(old)
            self.state.cameras[i] = dlg.result_camera
            self.scene.add_camera(dlg.result_camera)
            self.refresh_all()
            self.scene.update_frame(self.scene.frame)

    def remove_camera(self, i: int) -> None:
        cam = self.state.cameras.pop(i)
        self.scene.forget_camera(cam)
        if self.scene.view_mode.startswith("xray"):
            self.set_view("3d")
        self.refresh_all()
        self.scene.update_frame(self.scene.frame)

    def import_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Autoscoper trial configuration", "", "Config (*.cfg)")
        if not path:
            return
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
        self.refresh_all()
        self.scene.update_frame(self.scene.frame)
        if cams:
            self.settings_panel.show_page(1)
        msg = f"Added {len(cams)} camera(s) from {Path(path).name}."
        counts = {len(c.images) for c in cams}
        if cams and self.state.models and any(n != self.scene.n_frames for n in counts):
            msg += f"\n\nImage counts ({', '.join(map(str, sorted(counts)))}) differ from the {self.scene.n_frames} " \
                   "tracked frames. Use Edit on a camera to choose which image is frame 1."
        msg += "\n\nIts voxel sizes will be offered when you add TIF volumes."
        if problems:
            self._error("Some cameras couldn't be added", msg + "\n\n" + "\n\n".join(problems))
        else:
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
        self._rebuild_settings_panel()
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
