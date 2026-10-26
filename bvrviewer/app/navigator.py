"""Navigator tree (what's in the tab) and the properties sheet (settings of the selected item).

Laid out like OpenSim: the navigator lists Models, X-ray cameras and Scene; right-clicking a level
or an item offers its actions; selecting an item fills the properties sheet below it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit,
    QPushButton,
    QScrollArea, QSlider, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..core.models import BoneModel
from ..core.session import KinItem, TabState, XrayCamera
from .panels import ColorButton, h2, label

KIND_ROLE = Qt.UserRole
OBJ_ROLE = Qt.UserRole + 1

# Item kinds
MODELS, MODEL, TRACKING, CAMERAS, CAMERA, SCENE = "models", "model", "tracking", "cameras", "camera", "scene"
JOINTS, JOINT = "joints", "joint"
AXES = "axes"


def _dot_icon(color: str, size: int = 14, ring: bool = False) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if ring:
        p.setPen(QColor(color))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(2, 2, size - 5, size - 5)
    else:
        p.setPen(QColor("#00000055"))
        p.setBrush(QColor(color))
        p.drawEllipse(1, 1, size - 3, size - 3)
    p.end()
    return QIcon(pm)


class Navigator(QTreeWidget):
    """Tree of the tab's contents. Emits what the user does; the tab decides what that means."""

    itemSelected = Signal(str, object)              # kind, object (None for levels)
    menuRequested = Signal(str, object, QPoint)     # kind, object, global position
    visibilityToggled = Signal(object, bool)        # model, camera or joint
    axesToggled = Signal(object, bool)              # model whose coordinate-system axes are shown/hidden
    renamed = Signal(object, str)                   # model or camera, new name
    activated = Signal(str, object)                 # double-click on a camera: view through it

    def __init__(self):
        super().__init__()
        self.setObjectName("Navigator")
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setIconSize(QSize(14, 14))
        self.setIndentation(18)
        self.setUniformRowHeights(True)
        self.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context)
        self.currentItemChanged.connect(self._current_changed)
        self.itemChanged.connect(self._item_changed)
        self.itemDoubleClicked.connect(self._double_clicked)
        self._building = False
        self.roots: dict[str, QTreeWidgetItem] = {}
        bold = QFont()
        bold.setBold(True)
        for kind, text in [(MODELS, "Bones"), (CAMERAS, "X-ray cameras"), (JOINTS, "Joints"), (SCENE, "Scene")]:
            it = QTreeWidgetItem([text])
            it.setData(0, KIND_ROLE, kind)
            it.setFont(0, bold)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.addTopLevelItem(it)
            it.setExpanded(True)
            self.roots[kind] = it
        self.roots[MODELS].setToolTip(0, "Right-click to add a bone, or load a model")
        self.roots[CAMERAS].setToolTip(0, "Optional. Right-click to add an X-ray camera")
        self.roots[JOINTS].setToolTip(0, "Optional. Right-click to add joints from a model")
        self.roots[SCENE].setToolTip(0, "Axes, grid, background, motion trail, untracked frames")

    # ---- building
    def rebuild(self, state: TabState, select_kind: str | None = None, select_obj=None) -> None:
        """Rebuild the item lists from the tab's state, keeping expansion and the selection."""
        self._building = True
        expanded = {id(self._obj(it)) for it in self._all_items() if it.isExpanded() and self._obj(it) is not None}
        first_build = not any(self._obj(it) is not None for it in self._all_items())
        for root in (self.roots[MODELS], self.roots[CAMERAS], self.roots[JOINTS]):
            root.takeChildren()
        n_models, n_cams = len(state.models), len(state.cameras)
        self.roots[MODELS].setText(0, f"Bones ({n_models})" if n_models else "Bones")
        self.roots[CAMERAS].setText(0, f"X-ray cameras ({n_cams})" if n_cams else "X-ray cameras")
        if not n_models:
            self._placeholder(self.roots[MODELS], "Right-click Bones to add one, or load a model")
        for m in state.models:
            it = QTreeWidgetItem([m.name])
            self._set(it, MODEL, m)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            self.roots[MODELS].addChild(it)
            self.refresh_item(it)
            child = QTreeWidgetItem([""])
            self._set(child, TRACKING, m)
            child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.addChild(child)
            self.refresh_item(child)
            axes = QTreeWidgetItem([""])
            self._set(axes, AXES, m)
            axes.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            it.addChild(axes)
            self.refresh_item(axes)
            it.setExpanded(first_build or id(m) in expanded or m.tracking is None)
        if not n_cams:
            self._placeholder(self.roots[CAMERAS], "Optional: right-click to add")
        for cam in state.cameras:
            it = QTreeWidgetItem([cam.name])
            self._set(it, CAMERA, cam)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            self.roots[CAMERAS].addChild(it)
            self.refresh_item(it)
        n_j = len(state.joints)
        self.roots[JOINTS].setText(0, f"Joints ({n_j})" if n_j else "Joints")
        if not n_j:
            self._placeholder(self.roots[JOINTS], "Optional: right-click to add a joint")
        for jl in state.joints:
            it = QTreeWidgetItem([jl.label])
            self._set(it, JOINT, jl)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            self.roots[JOINTS].addChild(it)
            self.refresh_item(it)
        self._building = False
        if select_kind is not None:
            self.select(select_kind, select_obj)

    def _placeholder(self, root: QTreeWidgetItem, text: str) -> None:
        it = QTreeWidgetItem([text])
        it.setFlags(Qt.ItemIsEnabled)
        it.setForeground(0, QBrush(QColor("#7A8291")))
        f = it.font(0)
        f.setItalic(True)
        it.setFont(0, f)
        root.addChild(it)

    @staticmethod
    def _set(it: QTreeWidgetItem, kind: str, obj) -> None:
        it.setData(0, KIND_ROLE, kind)
        it.setData(0, OBJ_ROLE, obj)

    @staticmethod
    def _kind(it: QTreeWidgetItem | None) -> str | None:
        return it.data(0, KIND_ROLE) if it is not None else None

    @staticmethod
    def _obj(it: QTreeWidgetItem | None):
        return it.data(0, OBJ_ROLE) if it is not None else None

    def _all_items(self):
        out = []
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            it = stack.pop()
            out.append(it)
            stack.extend(it.child(i) for i in range(it.childCount()))
        return out

    def item_for(self, kind: str, obj=None) -> QTreeWidgetItem | None:
        if obj is None and kind in self.roots:
            return self.roots[kind]
        for it in self._all_items():
            if self._kind(it) == kind and self._obj(it) is obj:
                return it
        return None

    def refresh_item(self, it: QTreeWidgetItem | None) -> None:
        """Update one item's text, icon and tick box from its object (no rebuild)."""
        if it is None:
            return
        was, self._building = self._building, True
        kind, obj = self._kind(it), self._obj(it)
        if kind == MODEL:
            m: BoneModel = obj
            it.setText(0, m.name)
            it.setIcon(0, _dot_icon(m.color))
            it.setCheckState(0, Qt.Checked if m.visible else Qt.Unchecked)
            it.setToolTip(0, f"{m.path}\nTick to show, untick to hide. Right-click for actions.")
            it.setForeground(0, QBrush(QColor("#E6E8EC" if m.visible else "#7A8291")))
        elif kind == TRACKING:
            m = obj
            if m.tracking is not None:
                t = m.tracking
                gaps = f" · {t.n_untracked} untracked" if t.n_untracked else ""
                it.setText(0, f"{t.path.name} · {t.n_frames} frames{gaps}")
                it.setForeground(0, QBrush(QColor("#8FD3A9")))
                it.setToolTip(0, str(t.path))
            else:
                it.setText(0, "No tracking (right-click to attach)")
                it.setForeground(0, QBrush(QColor("#F2C48D")))
                it.setToolTip(0, "This bone stays untracked (in its own model space) until a .tra file is attached.")
        elif kind == AXES:
            m = obj
            if m.has_axes:
                ok = m.acs is not None
                it.setText(0, f"Axes: {m.axes_segment} · {Path(m.axes_path).name}" + ("" if ok else " (not found)"))
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if m.axes_visible else Qt.Unchecked)
                it.setForeground(0, QBrush(QColor("#5CC8A8" if ok else "#F2C48D")))
                it.setToolTip(0, f"Coordinate system from {m.axes_path}\nTick to show the axes. Select for this "
                                 "bone's motion about its own axes.")
            else:
                it.setText(0, "No coordinate system (right-click to add)")
                it.setData(0, Qt.CheckStateRole, None)
                it.setForeground(0, QBrush(QColor("#7A8291")))
                it.setToolTip(0, "Optional: axes from a model give this bone's motion about its own "
                                 "axes, and are used by joints.")
        elif kind == CAMERA:
            cam: XrayCamera = obj
            it.setText(0, cam.name)
            it.setIcon(0, _dot_icon("#F2C48D", ring=True))
            it.setCheckState(0, Qt.Checked if cam.visible else Qt.Unchecked)
            n = len(cam.images) if cam.images else 0
            it.setToolTip(0, f"{cam.calibration_path.name}\n{n} images · double-click to view through it")
            it.setForeground(0, QBrush(QColor("#E6E8EC" if cam.visible else "#7A8291")))
        elif kind == JOINT:
            jl: KinItem = obj
            it.setText(0, jl.label)
            it.setIcon(0, _dot_icon("#5CC8A8" if jl.kind == "segment" else "#B78AF0"))
            it.setCheckState(0, Qt.Checked if jl.visible else Qt.Unchecked)
            bones = " · ".join(str(n) for n in jl.tab_models()) if jl.model is not None else ""
            what = "Segment" if jl.kind == "segment" else "Joint"
            it.setToolTip(0, f"{what} from {jl.model_path.name}\n{bones}\nTick to show the axes.")
            it.setForeground(0, QBrush(QColor("#E6E8EC" if jl.visible else "#7A8291")))
        self._building = was

    def refresh_obj(self, kind: str, obj) -> None:
        self.refresh_item(self.item_for(kind, obj))

    def select(self, kind: str, obj=None) -> None:
        it = self.item_for(kind, obj)
        if it is not None and it is not self.currentItem():
            self.setCurrentItem(it)
        elif it is not None:
            self.itemSelected.emit(kind, obj)

    def start_rename(self, kind: str, obj) -> None:
        it = self.item_for(kind, obj)
        if it is not None:
            self.setCurrentItem(it)
            self.editItem(it, 0)

    # ---- events
    def _current_changed(self, cur: QTreeWidgetItem | None, _prev) -> None:
        if self._building or cur is None or self._kind(cur) is None:
            return
        self.itemSelected.emit(self._kind(cur), self._obj(cur))

    def _item_changed(self, it: QTreeWidgetItem, _col: int) -> None:
        if self._building:
            return
        kind, obj = self._kind(it), self._obj(it)
        if kind == AXES:
            if obj.has_axes and it.data(0, Qt.CheckStateRole) is not None:
                on = it.checkState(0) == Qt.Checked
                if on != obj.axes_visible:
                    self.axesToggled.emit(obj, on)
            return
        if kind not in (MODEL, CAMERA, JOINT):
            return
        on = it.checkState(0) == Qt.Checked
        if on != obj.visible:
            self.visibilityToggled.emit(obj, on)
        text = it.text(0).strip()
        current = obj.label if kind == JOINT else obj.name
        if text and text != current:
            self.renamed.emit(obj, text)
        elif not text:
            self.refresh_item(it)

    def _double_clicked(self, it: QTreeWidgetItem, _col: int) -> None:
        if self._kind(it) == CAMERA:
            self.activated.emit(CAMERA, self._obj(it))

    def _context(self, pos: QPoint) -> None:
        it = self.itemAt(pos)
        if it is None or self._kind(it) is None:
            it = None
        if it is not None:
            self.setCurrentItem(it)
        kind = self._kind(it) if it else MODELS
        self.menuRequested.emit(kind, self._obj(it) if it else None, self.viewport().mapToGlobal(pos))


class PropertySheet(QWidget):
    """Two-column sheet of editable properties, rebuilt for whatever is selected in the navigator."""

    LABEL_WIDTH = 104

    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        head = QWidget()
        head.setObjectName("PropsHead")
        head.setAttribute(Qt.WA_StyledBackground, True)
        hv = QVBoxLayout(head)
        hv.setContentsMargins(14, 10, 14, 8)
        hv.setSpacing(2)
        hv.addWidget(h2("Properties"))
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")
        hv.addWidget(self.title)
        outer.addWidget(head)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll, 1)
        self.body = None
        self.grid = None
        self.row_i = 0
        self.clear("")

    # ---- building blocks
    def clear(self, title: str, icon_color: str | None = None) -> None:
        self.title.setText(title)
        if self.body is not None:
            self.body.setParent(None)
            self.body.deleteLater()
        self.body = QWidget()
        self.body.setObjectName("Panel")
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(14, 6, 14, 14)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(7)
        self.grid.setColumnStretch(1, 1)
        # rows keep their full height (wrapped text included) and the sheet scrolls, rather than squashing
        self.grid.setSizeConstraint(QLayout.SetMinimumSize)
        self.row_i = 0
        self.scroll.setWidget(self.body)

    def finish(self) -> None:
        self.grid.setRowStretch(self.row_i, 1)
        self._fit_wrapped()

    def _fit_wrapped(self) -> None:
        """Give word-wrapped labels the height their text needs at the sheet's width.

        QGridLayout doesn't apply height-for-width to wrapped labels inside a scroll area, so long
        notes would otherwise be clipped to two lines.
        """
        if self.body is None:
            return
        m = self.grid.contentsMargins()
        sb = self.scroll.verticalScrollBar()
        # measure as if the scrollbar is showing (it often appears once the labels grow)
        vw = self.scroll.width() - sb.sizeHint().width() - 2
        full = max(vw - m.left() - m.right(), 120)
        value_w = full - self.LABEL_WIDTH - self.grid.horizontalSpacing()
        for i in range(self.grid.count()):
            w = self.grid.itemAt(i).widget()
            if not isinstance(w, QLabel) or not w.wordWrap():
                continue
            _r, col, _rs, cs = self.grid.getItemPosition(i)
            width = full if cs > 1 else (self.LABEL_WIDTH if col == 0 else value_w)
            w.setMinimumHeight(w.heightForWidth(max(width - 10, 60)) + 4)   # a little slack for rounding
        self.grid.invalidate()

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._fit_wrapped()

    def section(self, text: str) -> None:
        lab = h2(text)
        lab.setContentsMargins(0, 8 if self.row_i else 2, 0, 0)
        self.grid.addWidget(lab, self.row_i, 0, 1, 2)
        self.row_i += 1

    def row(self, text: str, widget: QWidget, tip: str | None = None) -> QWidget:
        lab = QLabel(text)
        lab.setObjectName("PropLabel")
        lab.setWordWrap(True)
        lab.setFixedWidth(self.LABEL_WIDTH)     # same label column on every page; long labels wrap
        if tip:
            lab.setToolTip(tip)
            widget.setToolTip(tip)
        self.grid.addWidget(lab, self.row_i, 0, Qt.AlignTop | Qt.AlignLeft)
        self.grid.addWidget(widget, self.row_i, 1)
        self.row_i += 1
        return widget

    def wide(self, widget: QWidget) -> QWidget:
        self.grid.addWidget(widget, self.row_i, 0, 1, 2)
        self.row_i += 1
        return widget

    def note(self, text: str, kind: str = "Muted") -> QLabel:
        return self.wide(label(text, kind))

    def value(self, text: str, value: str, tip: str | None = None, kind: str = "Value") -> QLabel:
        lab = QLabel(value)
        lab.setObjectName(kind)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return self.row(text, lab, tip)

    def buttons(self, items, danger: str | None = None) -> None:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 2, 0, 0)
        h.setSpacing(6)
        for text, fn in items:
            b = QPushButton(text)
            b.setObjectName("Danger" if text == danger else "Small")
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        self.wide(box)

    # ---- editors
    def line(self, text: str, value: str, on_done) -> QLineEdit:
        ed = QLineEdit(value)

        def done():
            v = ed.text().strip()
            if v:
                on_done(v)
            else:
                ed.setText(value)

        ed.editingFinished.connect(done)
        return self.row(text, ed)

    def color(self, text: str, value: str, on_change) -> ColorButton:
        b = ColorButton(value)
        b.colorChanged.connect(on_change)
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(b)
        h.addWidget(label(value.upper(), "Mono", wrap=False))
        b.colorChanged.connect(lambda c: h.itemAt(1).widget().setText(c.upper()))
        h.addStretch(1)
        self.row(text, box)
        return b

    def check(self, text: str, value: bool, on_change, caption: str = "", tip: str | None = None,
              enabled: bool = True) -> QCheckBox:
        cb = QCheckBox(caption)
        cb.setChecked(bool(value))
        cb.setEnabled(enabled)
        cb.toggled.connect(on_change)
        return self.row(text, cb, tip)

    def combo(self, text: str, items, current, on_change, tip: str | None = None) -> QComboBox:
        c = QComboBox()
        c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)   # long captions mustn't widen the sheet
        c.setMinimumContentsLength(8)
        keys = []
        for key, caption, *rest in items:
            c.addItem(caption, key)
            keys.append(key)
            if rest and rest[0] is False:
                c.model().item(c.count() - 1).setEnabled(False)
        if current in keys:
            c.setCurrentIndex(keys.index(current))
        c.currentIndexChanged.connect(lambda _: on_change(c.currentData()))
        return self.row(text, c, tip)

    def spin(self, text: str, value: int, lo: int, hi: int, on_change, suffix: str = "") -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(value))
        if suffix:
            s.setSuffix(suffix)
        s.valueChanged.connect(on_change)
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(s)
        h.addStretch(1)
        self.row(text, box)
        return s

    def slider(self, text: str, value: float, lo: int, hi: int, scale: float, fmt, on_change,
               on_release: bool = False) -> QSlider:
        """A slider for value*scale in [lo, hi]. fmt(x) formats the number shown beside it."""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(int(round(value * scale)))
        val = label(fmt(sl.value() / scale), "Mono", wrap=False)
        val.setFixedWidth(58)
        sl.valueChanged.connect(lambda x: val.setText(fmt(x / scale)))
        if on_release:
            sl.sliderReleased.connect(lambda: on_change(sl.value() / scale))
            sl.valueChanged.connect(lambda x: None if sl.isSliderDown() else on_change(x / scale))
        else:
            sl.valueChanged.connect(lambda x: on_change(x / scale))
        h.addWidget(sl, 1)
        h.addWidget(val)
        self.row(text, box)
        return sl
