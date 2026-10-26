"""Side panels and the timeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTabBar, QVBoxLayout, QWidget,
)

from ..core.models import BoneModel
from ..core.session import SceneSettings


def h2(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("H2")
    return lab


def label(text: str, kind: str = "Muted", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    return lab


def small_button(text: str, kind: str = "Small") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    return b


class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Swatch")
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Change colour")
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color: str) -> None:
        self.color = color
        self.setStyleSheet(f"QPushButton#Swatch {{ background: {color}; }}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self.color), self, "Bone colour")
        if c.isValid():
            self.set_color(c.name())
            self.colorChanged.emit(c.name())


# ---------------------------------------------------------------- models list

class ModelRow(QFrame):
    selected = Signal(object)
    changed = Signal(object)            # appearance changed
    renamed = Signal(str, str)          # old name, new name
    attachTracking = Signal(object)

    def __init__(self, model: BoneModel):
        super().__init__()
        self.model = model
        self.setObjectName("Row")
        self.setProperty("selected", False)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        top = QHBoxLayout()
        self.visible = QCheckBox()
        self.visible.setChecked(model.visible)
        self.visible.setToolTip("Show or hide")
        self.visible.toggled.connect(self._on_visible)
        self.color = ColorButton(model.color)
        self.color.colorChanged.connect(self._on_color)
        self.name = QLineEdit(model.name)
        self.name.setObjectName("NameEdit")
        self.name.setToolTip("Click to rename")
        self.name.editingFinished.connect(self._on_name)
        top.addWidget(self.visible)
        top.addWidget(self.color)
        top.addWidget(self.name, 1)
        v.addLayout(top)

        self.file_label = label("", "Mono", wrap=False)
        self.track_row = QHBoxLayout()
        self.track_label = label("", "Mono", wrap=False)
        self.attach = small_button("Attach .tra…", "Attach")
        self.attach.clicked.connect(lambda: self.attachTracking.emit(self.model))
        self.track_row.addWidget(self.track_label, 1)
        self.track_row.addWidget(self.attach)
        info = QVBoxLayout()
        info.setContentsMargins(26, 0, 0, 0)
        info.setSpacing(2)
        info.addWidget(self.file_label)
        info.addLayout(self.track_row)
        v.addLayout(info)

        op = QHBoxLayout()
        op.setContentsMargins(26, 0, 0, 0)
        op.addWidget(label("Opacity", wrap=False))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(int(model.opacity * 100))
        self.opacity.valueChanged.connect(self._on_opacity)
        self.op_value = label(f"{int(model.opacity * 100)}%", "Mono", wrap=False)
        self.op_value.setFixedWidth(36)
        op.addWidget(self.opacity, 1)
        op.addWidget(self.op_value)
        v.addLayout(op)
        self.refresh()

    def refresh(self) -> None:
        m = self.model
        self.file_label.setText(f"▣ {m.path.name}")
        self.file_label.setToolTip(str(m.path))
        if m.tracking is not None:
            t = m.tracking
            extra = f" · {t.n_untracked} untracked" if t.n_untracked else ""
            self.track_label.setText(f"↗ {t.path.name} · {t.n_frames} frames{extra}")
            self.track_label.setStyleSheet("color: #8FD3A9;")
            self.track_label.setToolTip(str(t.path))
            self.attach.hide()
        else:
            self.track_label.setText("No tracking · shown in model pose")
            self.track_label.setStyleSheet("color: #F2C48D;")
            self.attach.show()

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):  # noqa: N802
        self.selected.emit(self.model)
        super().mousePressEvent(e)

    def _on_visible(self, on: bool) -> None:
        self.model.visible = on
        self.changed.emit(self.model)

    def _on_color(self, c: str) -> None:
        self.model.color = c
        self.changed.emit(self.model)

    def _on_name(self) -> None:
        text = self.name.text().strip()
        if text and text != self.model.name:
            old, self.model.name = self.model.name, text
            self.renamed.emit(old, text)
            self.changed.emit(self.model)
        elif not text:
            self.name.setText(self.model.name)

    def _on_opacity(self, v: int) -> None:
        self.model.opacity = v / 100
        self.op_value.setText(f"{v}%")
        self.changed.emit(self.model)


class SelectedCard(QFrame):
    """Details of the selected model: files, model space, tracking format, display, hold still."""

    replaceModel = Signal(object)
    loadTfm = Signal(object)
    clearTfm = Signal(object)
    editVoxel = Signal(object)
    attachTracking = Signal(object)
    detachTracking = Signal(object)
    editFormat = Signal(object)
    displayChanged = Signal(object)
    holdStill = Signal(object, bool)
    remove = Signal(object)
    resetName = Signal(object)

    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        self.model: BoneModel | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)
        head = QHBoxLayout()
        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.reset_name = small_button("Use file name")
        self.reset_name.clicked.connect(lambda: self.resetName.emit(self.model))
        head.addWidget(self.dot)
        head.addWidget(self.title, 1)
        head.addWidget(self.reset_name)
        v.addLayout(head)

        v.addWidget(label("Model", wrap=False))
        r = QHBoxLayout()
        self.model_file = label("", "Mono", wrap=False)
        b = small_button("Replace…")
        b.clicked.connect(lambda: self.replaceModel.emit(self.model))
        r.addWidget(self.model_file, 1)
        r.addWidget(b)
        v.addLayout(r)
        self.space_note = label("")
        v.addWidget(self.space_note)
        self.space_row = QHBoxLayout()
        self.tfm_btn = small_button("Apply .tfm…")
        self.tfm_btn.setToolTip("For surfaces not in Autoscoper space, e.g. *-DICOM2AUT.tfm from SlicerAutoscoperM.")
        self.tfm_btn.clicked.connect(lambda: self.loadTfm.emit(self.model))
        self.tfm_clear = small_button("Remove .tfm")
        self.tfm_clear.clicked.connect(lambda: self.clearTfm.emit(self.model))
        self.voxel_btn = small_button("Voxel size / surface…")
        self.voxel_btn.clicked.connect(lambda: self.editVoxel.emit(self.model))
        for w in (self.tfm_btn, self.tfm_clear, self.voxel_btn):
            self.space_row.addWidget(w)
        self.space_row.addStretch(1)
        v.addLayout(self.space_row)

        v.addWidget(label("Tracking", wrap=False))
        r = QHBoxLayout()
        self.track_file = label("", "Mono", wrap=False)
        self.track_attach = small_button("Attach…")
        self.track_attach.clicked.connect(lambda: self.attachTracking.emit(self.model))
        self.track_detach = small_button("Detach")
        self.track_detach.clicked.connect(lambda: self.detachTracking.emit(self.model))
        r.addWidget(self.track_file, 1)
        r.addWidget(self.track_attach)
        r.addWidget(self.track_detach)
        v.addLayout(r)
        r = QHBoxLayout()
        self.track_fmt = label("")
        self.fmt_btn = small_button("Format…")
        self.fmt_btn.clicked.connect(lambda: self.editFormat.emit(self.model))
        r.addWidget(self.track_fmt, 1)
        r.addWidget(self.fmt_btn, 0, Qt.AlignTop)
        v.addLayout(r)
        self.track_warn = label("", "Warn")
        v.addWidget(self.track_warn)

        v.addWidget(label("Display as", wrap=False))
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, text in [("surface", "Surface"), ("volume", "Volume"), ("both", "Both")]:
            b = QPushButton(text)
            b.setCheckable(True)
            self.mode_group.addButton(b)
            self.mode_buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self._on_mode(k))
            seg.addWidget(b)
        v.addLayout(seg)
        self.hold = QCheckBox("Hold this bone still (view others relative to it)")
        self.hold.toggled.connect(lambda on: self.model and self.holdStill.emit(self.model, on))
        v.addWidget(self.hold)
        rm = small_button("Remove model", "Danger")
        rm.clicked.connect(lambda: self.remove.emit(self.model))
        v.addWidget(rm, 0, Qt.AlignLeft)

    def show_model(self, m: BoneModel | None, hold_name: str | None) -> None:
        self.model = m
        self.setVisible(m is not None)
        if m is None:
            return
        self.dot.setStyleSheet(f"background: {m.color}; border-radius: 5px;")
        self.title.setText(m.name)
        self.model_file.setText(m.path.name)
        self.model_file.setToolTip(str(m.path))
        if m.is_volume:
            vs = " × ".join(f"{x:g}" for x in m.voxel_size) if m.voxel_size else "?"
            self.space_note.setText(f"TIF volume · voxel {vs} mm · surface from {m.surface_note}")
        elif m.tfm_path:
            self.space_note.setText(f"Surface moved into Autoscoper space with {Path(m.tfm_path).name} (inverted)")
        else:
            self.space_note.setText("Surface assumed to be in Autoscoper space")
        self.tfm_btn.setVisible(not m.is_volume)
        self.tfm_clear.setVisible(not m.is_volume and m.tfm_path is not None)
        self.voxel_btn.setVisible(m.is_volume)
        t = m.tracking
        self.track_file.setText(t.path.name if t else "None attached")
        self.track_attach.setText("Replace…" if t else "Attach…")
        self.track_detach.setVisible(t is not None)
        self.fmt_btn.setVisible(t is not None)
        if t:
            vol = f" · volume {t.volume_index + 1} of {t.fmt.n_volumes}" if t.fmt.n_volumes > 1 else ""
            self.track_fmt.setText(f"Detected: {t.fmt.describe()} · {t.n_frames} frames · "
                                   f"{t.n_untracked} untracked{vol}")
        else:
            self.track_fmt.setText("")
        self.track_warn.setText("\n".join(t.warnings) if t else "")
        self.track_warn.setVisible(bool(t and t.warnings))
        for k, b in self.mode_buttons.items():
            b.setChecked(m.display == k)
            b.setEnabled(k == "surface" or m.image_data is not None)
        self.hold.blockSignals(True)
        self.hold.setChecked(hold_name == m.name)
        self.hold.setEnabled(t is not None)
        self.hold.blockSignals(False)

    def _on_mode(self, key: str) -> None:
        if self.model:
            self.model.display = key
            self.displayChanged.emit(self.model)


class ModelsPanel(QWidget):
    addModel = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.rows: list[ModelRow] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("Panel")
        self.v = QVBoxLayout(inner)
        self.v.setContentsMargins(14, 14, 14, 14)
        self.v.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(h2("Models in this tab"), 1)
        self.show_all = small_button("Show all")
        self.hide_all = small_button("Hide all")
        head.addWidget(self.show_all)
        head.addWidget(self.hide_all)
        self.v.addLayout(head)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(4)
        self.v.addLayout(self.rows_box)
        add = QPushButton("+  Add another model")
        add.setObjectName("Dashed")
        add.clicked.connect(self.addModel.emit)
        self.v.addWidget(add)
        self.card = SelectedCard()
        self.v.addWidget(self.card)
        self.v.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.show_all.clicked.connect(lambda: self._set_all(True))
        self.hide_all.clicked.connect(lambda: self._set_all(False))

    def _set_all(self, on: bool) -> None:
        for r in self.rows:
            r.visible.setChecked(on)

    def add_row(self, row: ModelRow) -> None:
        self.rows.append(row)
        self.rows_box.addWidget(row)

    def remove_row(self, model: BoneModel) -> None:
        for r in list(self.rows):
            if r.model is model:
                self.rows.remove(r)
                r.setParent(None)
                r.deleteLater()

    def row_for(self, model: BoneModel) -> ModelRow | None:
        return next((r for r in self.rows if r.model is model), None)

    def set_selected(self, model: BoneModel | None) -> None:
        for r in self.rows:
            r.set_selected(r.model is model)


# ---------------------------------------------------------------- options (bottom panel)

class OptionsPanel(QWidget):
    """Tabbed options along the bottom of the window, like an IDE's tool windows.

    Click a tab to open it; click the open tab again (or Hide) to fold the panel away,
    leaving just the tab strip.
    """

    changed = Signal(str)          # name of what changed
    addCamera = Signal()
    importCfg = Signal()
    editCamera = Signal(int)
    removeCamera = Signal(int)
    collapsedChanged = Signal(bool)

    BODY_HEIGHT = 170

    def __init__(self, settings: SceneSettings, collapsed: bool = False, page: int = 0):
        super().__init__()
        self.setObjectName("OptionsBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.s = settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(8, 0, 8, 0)
        self.bar = QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.addTab("Scene")
        self.bar.addTab("Radiographs")
        self.bar.tabBarClicked.connect(self._tab_clicked)
        head.addWidget(self.bar)
        head.addStretch(1)
        self.fold = small_button("Hide ▾")
        self.fold.setToolTip("Fold the options panel away (click a tab to bring it back)")
        self.fold.clicked.connect(lambda: self.set_collapsed(not self.collapsed))
        head.addWidget(self.fold)
        outer.addLayout(head)

        self.stack = QStackedWidget()
        self.stack.setObjectName("OptionsBody")
        self.stack.setAttribute(Qt.WA_StyledBackground, True)
        self.stack.setFixedHeight(self.BODY_HEIGHT)
        self.stack.addWidget(self._scroll(self._scene_page()))
        self.stack.addWidget(self._scroll(self._xray_page()))
        outer.addWidget(self.stack)
        self.collapsed = False
        self.bar.setCurrentIndex(page)
        self.stack.setCurrentIndex(page)
        self.set_collapsed(collapsed)

    # ---- layout helpers
    @staticmethod
    def _scroll(page: QWidget) -> QScrollArea:
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page.setObjectName("OptionsBody")
        page.setAttribute(Qt.WA_StyledBackground, True)
        sc.setWidget(page)
        return sc

    @staticmethod
    def _column(row: QHBoxLayout, title: str, width: int = 250) -> QVBoxLayout:
        box = QWidget()
        box.setFixedWidth(width)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(h2(title))
        row.addWidget(box, 0, Qt.AlignTop)
        return v

    @staticmethod
    def _divider(row: QHBoxLayout) -> None:
        line = QFrame()
        line.setFixedWidth(1)
        line.setStyleSheet("background: #262A33;")
        row.addWidget(line)

    def _scene_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(22)
        v = self._column(row, "Display", 230)
        self._check(v, "Global axes at origin", "show_axes")
        self._check(v, "Grid with mm scale", "show_grid")
        r = QHBoxLayout()
        r.addWidget(QLabel("Background"))
        self.bg = QComboBox()
        for k, t in [("dark", "Dark"), ("light", "Light grey"), ("white", "White (for figures)")]:
            self.bg.addItem(t, k)
        self.bg.setCurrentIndex(["dark", "light", "white"].index(self.s.background))
        self.bg.currentIndexChanged.connect(lambda _: self._set("background", self.bg.currentData()))
        r.addWidget(self.bg, 1)
        v.addLayout(r)
        v.addStretch(1)
        self._divider(row)

        v = self._column(row, "Motion trail", 250)
        self._check(v, "Show trail for the selected bone", "trail")
        r = QHBoxLayout()
        r.addWidget(label("Every", wrap=False))
        self.trail_step = QSpinBox()
        self.trail_step.setRange(1, 100)
        self.trail_step.setValue(self.s.trail_step)
        self.trail_step.valueChanged.connect(lambda x: self._set("trail_step", x))
        r.addWidget(self.trail_step)
        r.addWidget(label("frames, last", wrap=False))
        self.trail_count = QSpinBox()
        self.trail_count.setRange(1, 8)
        self.trail_count.setValue(self.s.trail_count)
        self.trail_count.valueChanged.connect(lambda x: self._set("trail_count", x))
        r.addWidget(self.trail_count)
        r.addStretch(1)
        v.addLayout(r)
        v.addStretch(1)
        self._divider(row)

        v = self._column(row, "Untracked frames", 240)
        grp = QButtonGroup(page)
        for key, text in [("hide", "Hide the bone"), ("hold", "Show last tracked pose"),
                          ("interpolate", "Interpolate across gaps")]:
            rb = QRadioButton(text)
            rb.setChecked(self.s.untracked == key)
            rb.toggled.connect(lambda on, k=key: on and self._set("untracked", k))
            grp.addButton(rb)
            v.addWidget(rb)
        v.addStretch(1)
        row.addStretch(1)
        return page

    def _xray_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(22)
        v = self._column(row, "X-ray cameras", 360)
        r = QHBoxLayout()
        b = small_button("+ Add X-ray camera")
        b.clicked.connect(self.addCamera.emit)
        r.addWidget(b)
        b = small_button("Import from .cfg…")
        b.clicked.connect(self.importCfg.emit)
        r.addWidget(b)
        r.addStretch(1)
        v.addLayout(r)
        self.no_cams = label("Optional: not needed to view motion. Add each X-ray camera's calibration and "
                             "images to show the radiographs too.")
        v.addWidget(self.no_cams)
        self.cam_box = QVBoxLayout()
        self.cam_box.setSpacing(6)
        v.addLayout(self.cam_box)
        v.addStretch(1)

        self.xray_opts = QWidget()
        opts = QHBoxLayout(self.xray_opts)
        opts.setContentsMargins(0, 0, 0, 0)
        opts.setSpacing(22)
        self._divider(opts)
        c = self._column(opts, "In the 3D scene", 300)
        self._check(c, "Radiographs on detector planes", "xray_in_scene")
        r = QHBoxLayout()
        self._check(r, "X-ray sources", "xray_sources")
        self._check(r, "Beam outlines", "xray_beams")
        r.addStretch(1)
        c.addLayout(r)
        self._slider(c, "Image opacity", "xray_opacity", 0, 100, 100, "%")
        self._slider(c, "Distance behind", "xray_distance", 0, 800, 1, " mm", reposition=True)
        c.addStretch(1)
        self._divider(opts)
        c = self._column(opts, "Through-X-ray views", 260)
        self._slider(c, "Bone opacity", "xray_bone_opacity", 0, 100, 100, "%")
        self._check(c, "Draw bones as wireframe", "xray_outline")
        c.addStretch(1)
        self._divider(opts)
        c = self._column(opts, "Image", 260)
        self._slider(c, "Brightness", "brightness", -100, 100, 100, "")
        self._slider(c, "Contrast", "contrast", 20, 300, 100, "×")
        self._check(c, "Invert", "invert")
        c.addStretch(1)
        row.addWidget(self.xray_opts, 0, Qt.AlignTop)
        row.addStretch(1)
        self.xray_opts.setVisible(False)
        return page

    # ---- folding
    def _tab_clicked(self, i: int) -> None:
        if i < 0:
            return
        if i == self.stack.currentIndex() and not self.collapsed:
            self.set_collapsed(True)
        else:
            self.stack.setCurrentIndex(i)
            self.set_collapsed(False)

    def set_collapsed(self, on: bool) -> None:
        self.collapsed = on
        self.stack.setVisible(not on)
        self.fold.setText("Show ▴" if on else "Hide ▾")
        self.collapsedChanged.emit(on)

    def show_page(self, i: int) -> None:
        self.bar.setCurrentIndex(i)
        self.stack.setCurrentIndex(i)
        self.set_collapsed(False)

    # ---- controls
    def _set(self, key: str, value) -> None:
        setattr(self.s, key, value)
        self.changed.emit(key)

    def _check(self, layout, text: str, key: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(getattr(self.s, key)))
        cb.toggled.connect(lambda on: self._set(key, on))
        layout.addWidget(cb)
        return cb

    def _slider(self, layout, text: str, key: str, lo: int, hi: int, scale: float, unit: str,
                reposition: bool = False) -> None:
        r = QHBoxLayout()
        lab = QLabel(text)
        lab.setFixedWidth(110)
        r.addWidget(lab)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(int(round(getattr(self.s, key) * scale)))
        val = label("", "Mono", wrap=False)
        val.setFixedWidth(56)

        def show(x: int) -> None:
            v = x / scale
            val.setText(f"{x}{unit}" if unit in ("%", " mm") else (f"{v:.1f}{unit}" if unit else f"{v:+.2f}"))

        show(sl.value())

        def moved(x: int) -> None:
            show(x)
            if not reposition:
                self._set(key, x / scale)

        sl.valueChanged.connect(moved)
        if reposition:
            sl.sliderReleased.connect(lambda: self._set(key, sl.value() / scale))
        r.addWidget(sl, 1)
        r.addWidget(val)
        layout.addLayout(r)

    def set_cameras(self, cameras) -> None:
        while self.cam_box.count():
            w = self.cam_box.takeAt(0).widget()
            if w:
                w.setParent(None)
        for i, cam in enumerate(cameras):
            card = QFrame()
            card.setObjectName("Card")
            cv = QHBoxLayout(card)
            cv.setContentsMargins(10, 6, 8, 6)
            cv.setSpacing(6)
            cb = QCheckBox(cam.name)
            cb.setChecked(cam.visible)
            cb.setStyleSheet("font-weight: 600;")
            n = len(cam.images) if cam.images else 0
            first = cam.images.label(cam.first_image) if cam.images else "?"
            cb.setToolTip(f"{cam.calibration_path.name}\n"
                          f"{cam.images.source.name if cam.images else '?'}\n"
                          f"{n} images · frame 1 = image {first}")

            def toggle(on, c=cam):
                c.visible = on
                self.changed.emit("cameras")

            cb.toggled.connect(toggle)
            cv.addWidget(cb)
            cv.addWidget(label(f"{n} images · starts at {first}", "Mono", wrap=False), 1)
            e = small_button("Edit")
            e.clicked.connect(lambda _=False, k=i: self.editCamera.emit(k))
            x = small_button("Remove")
            x.clicked.connect(lambda _=False, k=i: self.removeCamera.emit(k))
            cv.addWidget(e)
            cv.addWidget(x)
            self.cam_box.addWidget(card)
        self.xray_opts.setVisible(bool(cameras))
        self.no_cams.setVisible(not cameras)
        self.bar.setTabText(1, f"Radiographs ({len(cameras)})" if cameras else "Radiographs")


# ---------------------------------------------------------------- timeline

class CoverageStrip(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(6)
        self.valid: np.ndarray | None = None
        self.n = 1

    def set_data(self, valid: np.ndarray | None, n: int) -> None:
        self.valid, self.n = valid, max(n, 1)
        self.setToolTip("Tracked (green) and untracked (orange) frames for the selected bone"
                        if valid is not None else "")
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#262A33"))
        if self.valid is None:
            return
        n = self.n
        for i in range(n):
            ok = i < len(self.valid) and self.valid[i]
            x0, x1 = int(w * i / n), int(w * (i + 1) / n)
            p.fillRect(x0, 0, max(x1 - x0, 1), h, QColor("#3B6E52" if ok else "#D9822B"))


class Timeline(QWidget):
    frameChanged = Signal(int)

    def __init__(self, settings: SceneSettings):
        super().__init__()
        self.setObjectName("Panel")
        self.s = settings
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        r = QHBoxLayout()
        self.first = QPushButton("|‹")
        self.prev = QPushButton("‹")
        self.play = QPushButton("▶")
        self.play.setObjectName("Primary")
        self.play.setCheckable(True)
        self.next = QPushButton("›")
        self.last = QPushButton("›|")
        for b, tip in [(self.first, "First frame"), (self.prev, "Previous frame (←)"), (self.play, "Play / pause (space)"),
                       (self.next, "Next frame (→)"), (self.last, "Last frame")]:
            b.setFixedSize(42, 38)
            b.setToolTip(tip)
            b.setStyleSheet("font-size: 17px; font-weight: 600;" if b is not self.play else "")
            r.addWidget(b)
        mid = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.coverage = CoverageStrip()
        mid.addWidget(self.slider)
        mid.addWidget(self.coverage)
        r.addLayout(mid, 1)
        self.counter = label("Frame 1 / 1", "Mono", wrap=False)
        self.counter.setMinimumWidth(110)
        r.addWidget(self.counter)
        r.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for x in (0.1, 0.25, 0.5, 1.0, 2.0):
            self.speed.addItem(f"{x:g}×", x)
        self.speed.setCurrentIndex(3)
        self.speed.currentIndexChanged.connect(lambda _: setattr(self.s, "speed", self.speed.currentData()))
        r.addWidget(self.speed)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(settings.loop)
        self.loop.toggled.connect(lambda on: setattr(self.s, "loop", on))
        r.addWidget(self.loop)
        v.addLayout(r)
        self.slider.valueChanged.connect(self._changed)
        self.first.clicked.connect(lambda: self.slider.setValue(0))
        self.last.clicked.connect(lambda: self.slider.setValue(self.slider.maximum()))
        self.prev.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))

    def _changed(self, f: int) -> None:
        self.counter.setText(f"Frame {f + 1} / {self.slider.maximum() + 1}")
        self.frameChanged.emit(f)

    def set_count(self, n: int) -> None:
        f = min(self.slider.value(), n - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(f)
        self.slider.blockSignals(False)
        self.counter.setText(f"Frame {f + 1} / {n}")

    def step(self, d: int) -> bool:
        """Advance; returns False when it stops at the end."""
        f = self.slider.value() + d
        n = self.slider.maximum()
        if f > n or f < 0:
            if not self.s.loop:
                return False
            f = f % (n + 1)
        self.slider.setValue(f)
        return True
