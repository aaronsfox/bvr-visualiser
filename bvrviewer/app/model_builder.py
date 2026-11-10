"""Model Builder tab: segments (axes on bone surfaces) -> joints (optional) -> reference poses (optional)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QRadioButton, QSlider, QSpinBox, QSplitter,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ..core import auto_axes as AA
from ..core import joints as J
from ..core.models import BoneModel, ModelError, is_volume_file
from ..core.tracking import TrackingError, load_tracking
from ..core.transforms import TransformFileError, model_to_aut_from_tfm
from .dialogs import VoxelDialog
from .navigator import PropertySheet
from .panels import h2, label
from .scene import BACKGROUNDS

MODEL_FILTER = "Bone models (*.stl *.STL *.tif *.tiff *.TIF *.obj *.ply *.vtk *.vtp);;All files (*)"
JOINT_MODEL_FILTER = "Model (*.bvrmodel.json *.json)"
TRA_FILTER = "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)"
TFM_FILTER = "ITK transform (*.tfm *.txt);;All files (*)"
AXIS_COLOURS = ("#E5534B", "#57AB5A", "#539BF5")        # X red, Y green, Z blue
BONE_COLOURS = ("#E8D9B5", "#C9B98F", "#B9C7D6", "#D6B9C9")
KIND, OBJ, KEY = Qt.UserRole, Qt.UserRole + 1, Qt.UserRole + 2
SHORT_SOURCE = {"landmarks": "Landmarks", "staple_femur_cylinder": "Cylinder", "staple_femur_spheres": "Spheres",
                "staple_femur_ellipsoids": "Ellipsoids", "staple_femur_kai": "Kai", "staple_tibia_kai": "Kai",
                "staple_tibia_ellipse": "Ellipse", "staple_tibia_centroids": "Centroids",
                "staple_tibia_plateau": "Plateau", "femur_cylinder": "Cylinder (old)",
                "femur_spheres": "Spheres (old)", "tibia_ellipse": "Ellipse (old)", "tibia_centroids": "Centroids (old)",
                "tibia_inertia": "Plateau inertia (old)", "inertial": "Inertial",
                "staple_talus_ankle": "Talocrural", "staple_talus_subtalar": "Subtalar", "aafact": "AAFACT"}


BONE_GROUPS = (("Knee", ("femur", "tibia")),
               ("Ankle and foot", ("tibia_distal", "fibula", "talus", "calcaneus", "navicular", "cuboid",
                                   "medial_cuneiform", "intermediate_cuneiform", "lateral_cuneiform",
                                   "metatarsal1", "metatarsal2", "metatarsal3", "metatarsal4", "metatarsal5")),
               ("Any bone", ("generic",)))


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings("BVR", "BVR Motion Viewer")


class _State:
    """Minimal state object so the main window can name and close this tab like the others."""

    def __init__(self):
        self.name = "Model"
        self.models: list = []


def add_triad(plotter, acs: np.ndarray, length: float, opacity: float = 1.0, pose: np.ndarray | None = None):
    """Draw an X/Y/Z axis triad for a 4x4 coordinate system; returns the actors."""
    M = acs if pose is None else pose @ acs
    o = M[:3, 3]
    actors = []
    for k in range(3):
        arrow = pv.Arrow(start=o, direction=M[:3, k], scale=length, tip_length=0.18, tip_radius=0.06,
                         shaft_radius=0.022)
        actors.append(plotter.add_mesh(arrow, color=AXIS_COLOURS[k], opacity=opacity, pickable=False,
                                       reset_camera=False, smooth_shading=True))
    return actors


def scan_pose_from_tfm(path) -> np.ndarray:
    """Pose of an Autoscoper model in the scan (CT) space, from SlicerAutoscoperM's *-DICOM2AUT.tfm."""
    return np.linalg.inv(model_to_aut_from_tfm(path))


def guess_tfm(surface) -> str | None:
    """The *-DICOM2AUT.tfm that belongs with a surface, if one is easy to find next to it."""
    if surface is None:
        return None
    if getattr(surface, "tfm_path", None):
        return str(surface.tfm_path)
    p = Path(surface.path)
    stem = p.stem.lower().removeprefix("aut_")
    for folder in (p.parent, p.parent.parent / "Transforms", p.parent.parent / "transforms"):
        if folder.is_dir():
            for f in sorted(folder.glob("*DICOM2AUT*.tfm")):
                if f.stem.lower().split("-dicom2aut")[0] == stem:
                    return str(f)
    return None


def _guess_side(filename: str) -> str | None:
    """'left' / 'right' from a file name (AUT_Left_Femur.stl, tal_R.stl), or None."""
    import re

    words = [w for w in re.split(r"[^a-z]+", Path(filename).stem.lower()) if w]
    if any(w in ("left", "lt", "l") for w in words):
        return "left"
    if any(w in ("right", "rt", "r") for w in words):
        return "right"
    return None


# ==================================================================== tools of the 3D view

@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    key: str
    tip: str


TOOLS = (
    Tool("navigate", "Navigate", "N", "Left-drag rotates, middle-drag pans, scroll or right-drag zooms (as in the "
                                      "Visualiser)."),
    Tool("point", "Place point", "P", "Click the surface to place the selected landmark (click again to move it)."),
    Tool("paint", "Paint", "B", "Drag over the surface to paint the selected landmark's area."),
    Tool("erase", "Erase", "E", "Drag over the painted area to remove it."),
    Tool("pick", "Inspect", "I", "Click the surface to see the point's coordinates and the nearest landmark."),
)
TOOL_NAV_TIP = "While a tool is active: right-drag rotates, middle-drag pans, scroll zooms. Esc returns to Navigate."


def tool_icon(name: str, color: str = "#D5DAE3") -> QIcon:
    """Simple line icons for the tool bar, drawn here so they match the theme."""
    pm = QPixmap(48, 48)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 3.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    if name == "navigate":                      # orbit: an ellipse around a centre, with an arrow head
        p.drawEllipse(QPointF(24, 24), 17, 9)
        p.drawEllipse(QPointF(24, 24), 9, 17)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(24, 24), 3, 3)
    elif name == "point":                       # cross-hair with a dot
        for a, b in (((24, 4), (24, 16)), ((24, 32), (24, 44)), ((4, 24), (16, 24)), ((32, 24), (44, 24))):
            p.drawLine(QPointF(*a), QPointF(*b))
        p.setBrush(QColor("#F0A04B"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(24, 24), 6, 6)
    elif name == "paint":                       # brush: handle and a filled tip
        p.drawLine(QPointF(38, 8), QPointF(22, 26))
        path = QPainterPath(QPointF(22, 24))
        path.cubicTo(QPointF(12, 22), QPointF(10, 34), QPointF(6, 42))
        path.cubicTo(QPointF(16, 42), QPointF(28, 38), QPointF(26, 28))
        path.closeSubpath()
        p.setBrush(QColor("#F0A04B"))
        p.drawPath(path)
    elif name == "erase":                       # eraser block
        path = QPainterPath(QPointF(18, 40))
        for x, y in ((6, 28), (26, 8), (42, 24), (26, 40), (18, 40)):
            path.lineTo(QPointF(x, y))
        p.drawPath(path)
        p.drawLine(QPointF(14, 20), QPointF(34, 40))
        p.drawLine(QPointF(18, 40), QPointF(42, 40))
    elif name == "pick":                        # "i" in a circle
        p.drawEllipse(QPointF(24, 24), 18, 18)
        p.drawLine(QPointF(24, 21), QPointF(24, 34))
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(24, 14), 1.8, 1.8)
    elif name == "undo":                        # curved arrow
        path = QPainterPath(QPointF(14, 18))
        path.cubicTo(QPointF(26, 8), QPointF(42, 16), QPointF(40, 30))
        path.cubicTo(QPointF(38, 38), QPointF(30, 40), QPointF(24, 40))
        p.drawPath(path)
        p.drawLine(QPointF(14, 18), QPointF(16, 6))
        p.drawLine(QPointF(14, 18), QPointF(26, 20))
    p.end()
    return QIcon(pm)


# ==================================================================== dialogs

class AddJointDialog(QDialog):
    """Pick a joint definition and the two segments it connects."""

    def __init__(self, parent, model: J.KinematicModel, type_key: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add a joint")
        self.model = model
        v = QVBoxLayout(self)
        v.addWidget(label("A joint uses the axes already defined on its two bones. Its angles are the distal "
                          "bone's axes relative to the proximal bone's, by the definition chosen here."))
        form = QFormLayout()
        self.type = QComboBox()
        self._cands = lambda k, end: [s for s in model.candidates(k, end) if s.acs is not None]
        for k, t in J.JOINT_TYPES.items():
            if len({s.name for s in self._cands(k, "proximal") + self._cands(k, "distal")}) >= 2:
                self.type.addItem(t.title, k)
        if type_key is not None and self.type.findData(type_key) >= 0:
            self.type.setCurrentIndex(self.type.findData(type_key))
        self.prox, self.dist = QComboBox(), QComboBox()
        self.name = QLineEdit()
        self.summary = label("", "Muted")
        form.addRow("Definition", self.type)
        form.addRow("Proximal bone", self.prox)
        form.addRow("Distal bone", self.dist)
        form.addRow("Name", self.name)
        v.addLayout(form)
        v.addWidget(self.summary)
        self.warn = QLabel("")
        self.warn.setObjectName("Warn")
        v.addWidget(self.warn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        for b in bb.buttons():
            b.setIcon(QIcon())
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.type.currentIndexChanged.connect(self._type_changed)
        self._type_changed()

    def _type_changed(self) -> None:
        k = self.type.currentData()
        if k is None:
            return
        t = J.JOINT_TYPES[k]
        for cb, end in ((self.prox, "proximal"), (self.dist, "distal")):
            cb.clear()
            for s in self._cands(k, end):
                cb.addItem(s.name, s.name)
        if self.dist.count() > 1 and self.dist.currentData() == self.prox.currentData():
            self.dist.setCurrentIndex(1)
        self.name.setText(t.default_name)
        self.summary.setText(t.summary)

    def _ok(self) -> None:
        if self.prox.currentData() == self.dist.currentData():
            self.warn.setText("Choose two different bones.")
            return
        self.accept()

    def values(self):
        return self.type.currentData(), self.prox.currentData(), self.dist.currentData(), self.name.text().strip()


class _FileRow(QWidget):
    def __init__(self, caption: str, filt: str, value: str | None = None):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value or "")
        b = QPushButton("Browse…")
        b.setObjectName("Small")
        b.clicked.connect(lambda: self._browse(caption, filt))
        h.addWidget(self.edit, 1)
        h.addWidget(b)

    def _browse(self, caption, filt):
        start = str(Path(self.edit.text()).parent) if self.edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(self, caption, start, filt)
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()


class ReferenceDialog(QDialog):
    """Choose where a joint's reference pose comes from."""

    def __init__(self, parent, model: J.KinematicModel, joint: J.Joint, start: str = "trial"):
        super().__init__(parent)
        self.setWindowTitle(f"Reference pose for {joint.name}")
        self.model, self.joint = model, joint
        p, d = model.segment(joint.proximal), model.segment(joint.distal)
        v = QVBoxLayout(self)
        v.addWidget(label("The joint's values in this pose are taken as zero. Use a neutral or static trial "
                          "tracked like any other, or the pose the bones had in the scan. Without a reference, "
                          "zero is anatomical (the two bones' axes lined up)."))
        self.group = QButtonGroup(self)
        self.r_trial = QRadioButton("Neutral / static trial (tracking files)")
        self.r_scan = QRadioButton("Pose in the scan (each bone's *-DICOM2AUT.tfm)")
        self.r_same = QRadioButton("Pose in the scan (both surfaces are already in the scan's space)")
        for r in (self.r_trial, self.r_scan, self.r_same):
            self.group.addButton(r)
        v.addWidget(self.r_trial)
        f1 = QFormLayout()
        self.tra_p = _FileRow(f"Tracking for {p.name}", TRA_FILTER, p.preview_tracking)
        self.tra_d = _FileRow(f"Tracking for {d.name}", TRA_FILTER, d.preview_tracking)
        self.all_frames = QRadioButton("Every frame both are tracked (averaged)")
        self.some_frames = QRadioButton("Frames")
        self.all_frames.setChecked(True)
        self.f_from, self.f_to = QSpinBox(), QSpinBox()
        for s in (self.f_from, self.f_to):
            s.setRange(1, 1_000_000)
        self.f_to.setValue(1)
        fr = QHBoxLayout()
        fr.addWidget(self.some_frames)
        fr.addWidget(self.f_from)
        fr.addWidget(QLabel("to"))
        fr.addWidget(self.f_to)
        fr.addStretch(1)
        f1.addRow(p.name, self.tra_p)
        f1.addRow(d.name, self.tra_d)
        f1.addRow("Use", self.all_frames)
        f1.addRow("", fr)
        v.addLayout(f1)
        v.addWidget(self.r_scan)
        f2 = QFormLayout()
        self.tfm_p = _FileRow(f"Scan transform for {p.name}", TFM_FILTER, guess_tfm(p.surface))
        self.tfm_d = _FileRow(f"Scan transform for {d.name}", TFM_FILTER, guess_tfm(d.surface))
        f2.addRow(p.name, self.tfm_p)
        f2.addRow(d.name, self.tfm_d)
        v.addLayout(f2)
        v.addWidget(self.r_same)
        f3 = QFormLayout()
        self.mode = QComboBox()
        for k, t in J.REFERENCE_MODES.items():
            self.mode.addItem(t, k)
        if joint.reference is not None:
            self.mode.setCurrentIndex(self.mode.findData(joint.reference.mode))
        f3.addRow("Normalise by", self.mode)
        v.addLayout(f3)
        self.warn = QLabel("")
        self.warn.setObjectName("Warn")
        self.warn.setWordWrap(True)
        v.addWidget(self.warn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        for b in bb.buttons():
            b.setIcon(QIcon())
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        {"trial": self.r_trial, "scan": self.r_scan, "same": self.r_same}.get(start, self.r_trial).setChecked(True)

    def _ok(self) -> None:
        try:
            self.apply()
        except (J.JointModelError, TrackingError, TransformFileError, OSError, ValueError) as err:
            self.warn.setText(str(err))
            return
        self.accept()

    def apply(self) -> None:
        """Set the joint's reference from the dialog's choices (raises on bad input)."""
        m, j, mode = self.model, self.joint, self.mode.currentData()
        if self.r_trial.isChecked():
            if not self.tra_p.value() or not self.tra_d.value():
                raise J.JointModelError("Choose a tracking file for each bone.")
            P, D = load_tracking(self.tra_p.value()).poses, load_tracking(self.tra_d.value()).poses
            frames = None
            if self.some_frames.isChecked():
                a, b = sorted((self.f_from.value(), self.f_to.value()))
                frames = range(a - 1, b)
            src = {"kind": "trial", "proximal": self.tra_p.value(), "distal": self.tra_d.value(),
                   "frames": "all" if frames is None else [frames.start + 1, frames.stop]}
            m.set_reference_from_tracking(j.name, P, D, frames, mode, src)
        elif self.r_scan.isChecked():
            if not self.tfm_p.value() or not self.tfm_d.value():
                raise J.JointModelError("Choose the scan transform (.tfm) for each bone.")
            m.set_reference_from_poses(j.name, scan_pose_from_tfm(self.tfm_p.value()),
                                       scan_pose_from_tfm(self.tfm_d.value()), mode,
                                       {"kind": "scan", "proximal": self.tfm_p.value(), "distal": self.tfm_d.value()})
        else:
            m.set_reference_from_poses(j.name, np.eye(4), np.eye(4), mode, {"kind": "scan_space"})


# ==================================================================== the tab

class ModelBuilderTab(QWidget):
    kind = "model_builder"
    titleChanged = Signal(str)

    def __init__(self, state=None):
        super().__init__()
        self.state = state or _State()
        self.model = J.KinematicModel(name="Model", require_apply=True)
        self.sel = ("model", None, None)          # (kind, Segment/Joint/(a, b) or None, landmark key)
        self.brush = 4.0                          # mm
        self.tool = "navigate"                    # the left button's tool: see TOOLS
        self._painting = False                    # a paint / erase stroke is under way
        self._rotating_right = False              # right button rotating while a tool is active
        self._painted: dict = {}                  # (id(segment), key) -> set of vertex indices (this session)
        self._trees: dict = {}                    # id(mesh) -> cKDTree of its points
        self._actors: list = []
        self._bone_actors: dict = {}
        self._cursor_actor = None
        self._undo: list = []                     # (segment, landmarks before an edit)
        self._dirty = False
        self.show_fit = True                      # draw what an automatic method fitted
        self._compare = None                      # (joint, frame, rows) from Compare methods

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)
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
        self.tree = QTreeWidget()
        self.tree.setObjectName("Navigator")
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setToolTip("Ctrl+click two bones to add a joint between them")
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        nb.addWidget(self.tree, 1)
        left.addWidget(nav_box)
        self.props = PropertySheet()
        left.addWidget(self.props)
        left.setSizes([300, 560])
        left.setMinimumWidth(360)
        split.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("Panel")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 6, 10, 6)
        bl.setSpacing(6)
        for text, fn, tip in (("Add bones…", self.add_surfaces, ("Load bone surfaces (STL, or TIF partial volumes): "
                                                                 "each file becomes a bone")),
                              ("Open model…", self.open_model, "Open a saved model (.bvrmodel.json)"),
                              ("Save", self.save_model, "Save the model (Ctrl+S)"),
                              ("Save as…", self.save_model_as, "Save the model to a new file (Ctrl+Shift+S)")):
            b = QPushButton(text)
            b.setObjectName("Small")
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            bl.addWidget(b)
        bl.addSpacing(10)
        self.showing = label("", wrap=False)
        bl.addWidget(self.showing, 1)
        views = QToolButton()
        views.setText("Views")
        views.setPopupMode(QToolButton.InstantPopup)
        vm = QMenu(views)
        for text, args in (("Looking along +X", ("x", 1)), ("Looking along −X", ("x", -1)),
                           ("Looking along +Y", ("y", 1)), ("Looking along −Y", ("y", -1)),
                           ("Looking along +Z", ("z", 1)), ("Looking along −Z", ("z", -1))):
            a = vm.addAction(text)
            a.triggered.connect(lambda _=False, args=args: self.view_along(*args))
        views.setMenu(vm)
        views.setToolTip("Standard views, in the bone's model space (the axes of the corner triad)")
        bl.addWidget(views)
        fit = QPushButton("Fit view")
        fit.setObjectName("Small")
        fit.clicked.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        bl.addWidget(fit)
        cv.addWidget(bar)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(0)
        view_row.addWidget(self._make_toolbar())
        self.plotter = QtInteractor(center)
        self.plotter.set_background(BACKGROUNDS["dark"])
        self.plotter.add_axes(color="#C3C8D2", line_width=2)
        view_row.addWidget(self.plotter.interactor, 1)
        cv.addLayout(view_row, 1)
        self.hint = QLabel("")
        self.hint.setObjectName("HintBar")
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(12, 8, 12, 8)
        cv.addWidget(self.hint)
        split.addWidget(center)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1100])
        root.addWidget(split)

        style = self.plotter.iren.interactor.GetInteractorStyle()
        self._style = style
        for ev, fn in (("LeftButtonPressEvent", self._left_down), ("LeftButtonReleaseEvent", self._left_up),
                       ("RightButtonPressEvent", self._right_down), ("RightButtonReleaseEvent", self._right_up),
                       ("MouseMoveEvent", self._mouse_move)):
            style.AddObserver(ev, fn)
        for key, fn in (("Ctrl+Z", self.undo), ("Escape", lambda: self.set_tool("navigate")),
                        ("Ctrl+S", self.save_model), ("Ctrl+Shift+S", self.save_model_as)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        for t in TOOLS:
            sc = QShortcut(QKeySequence(t.key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda k=t.name: self.set_tool(k))
        self.refresh(rebuild_view=True)

    # ================================================================ tool bar
    def _make_toolbar(self) -> QWidget:
        box = QWidget()
        box.setObjectName("Panel")
        v = QVBoxLayout(box)
        v.setContentsMargins(5, 8, 5, 8)
        v.setSpacing(4)
        self.tool_buttons: dict = {}
        group = QButtonGroup(box)
        group.setExclusive(True)
        for t in TOOLS:
            b = QToolButton()
            b.setCheckable(True)
            b.setIcon(tool_icon(t.name))
            b.setIconSize(QSize(22, 22))
            b.setFixedSize(38, 38)
            b.setAutoRaise(True)
            b.setToolTip(f"{t.title} ({t.key})\n{t.tip}")
            b.clicked.connect(lambda _=False, k=t.name: self.set_tool(k))
            group.addButton(b)
            v.addWidget(b)
            self.tool_buttons[t.name] = b
        v.addSpacing(8)
        self.undo_btn = QToolButton()
        self.undo_btn.setIcon(tool_icon("undo"))
        self.undo_btn.setIconSize(QSize(22, 22))
        self.undo_btn.setFixedSize(38, 38)
        self.undo_btn.setAutoRaise(True)
        self.undo_btn.setToolTip("Undo the last landmark change (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        v.addWidget(self.undo_btn)
        v.addSpacing(8)
        self.brush_label = QLabel("Brush")
        self.brush_label.setObjectName("Muted")
        self.brush_label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.brush_label)
        self.brush_slider = QSlider(Qt.Vertical)
        self.brush_slider.setRange(1, 15)
        self.brush_slider.setValue(int(self.brush))
        self.brush_slider.setFixedHeight(110)
        self.brush_slider.setToolTip("Brush size (mm)")
        self.brush_slider.valueChanged.connect(self._set_brush)
        v.addWidget(self.brush_slider, 0, Qt.AlignHCenter)
        self.brush_value = QLabel(f"{self.brush:.0f} mm")
        self.brush_value.setObjectName("Muted")
        self.brush_value.setAlignment(Qt.AlignCenter)
        v.addWidget(self.brush_value)
        v.addStretch(1)
        return box

    def _tool_target(self):
        """(segment, landmark key, method) the point / paint / erase tools act on, or None."""
        kind, obj, key = self.sel
        if kind != "landmark" or obj is None or obj.surface is None or obj.axes_source != "landmarks":
            return None
        d = next(x for x in obj.stype.landmarks if x.key == key)
        return obj, key, (obj.landmarks.get(key) or {"method": d.methods[0]})["method"]

    def tool_available(self, name: str) -> tuple[bool, str]:
        """Whether a tool can be used now, and why not."""
        if name in ("navigate",):
            return True, ""
        if name == "pick":
            ok = bool(self._shown_segments())
            return ok, "" if ok else "Add a bone first."
        tgt = self._tool_target()
        if tgt is None:
            return False, "Choose a landmark first: Place or Paint in the bone's steps, or select it in the navigator."
        method = tgt[2]
        if name == "point" and method != "point":
            return False, "This landmark is painted (its method is a region or sphere): use Paint."
        if name in ("paint", "erase") and method == "point":
            return False, "This landmark is a single point: use Place point."
        return True, ""

    def _update_tools(self) -> None:
        if self.tool != "navigate" and not self.tool_available(self.tool)[0]:
            self.tool = "navigate"
        for t in TOOLS:
            b = self.tool_buttons[t.name]
            ok, why = self.tool_available(t.name)
            b.setEnabled(ok)
            b.setChecked(self.tool == t.name)
            b.setToolTip(f"{t.title} ({t.key})\n{t.tip}" + ("" if ok else f"\n\n{why}"))
        brush = self.tool in ("paint", "erase")
        for w in (self.brush_label, self.brush_slider, self.brush_value):
            w.setVisible(brush)
        self.undo_btn.setEnabled(bool(self._undo))
        self.plotter.interactor.setCursor(Qt.CrossCursor if self.tool in ("point", "pick") else
                                          Qt.BlankCursor if brush else Qt.ArrowCursor)
        if not brush:
            self._hide_cursor()

    def set_tool(self, name: str) -> bool:
        ok, why = self.tool_available(name)
        if not ok:
            self.hint.setText(why)
            self._update_tools()
            return False
        self.tool = name
        self._update_tools()
        self._update_hint()
        return True

    def view_along(self, axis: str, sign: int) -> None:
        pl = self.plotter
        d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        up = (0, 0, 1) if axis != "z" else (0, 1, 0)
        c = np.array(pl.camera.focal_point)
        dist = np.linalg.norm(np.array(pl.camera.position) - c) or 300.0
        pl.camera.position = tuple(c - sign * np.array(d, float) * dist)
        pl.camera.up = up
        pl.reset_camera()
        pl.render()

    # ================================================================ navigator
    def refresh(self, rebuild_view: bool = True) -> None:
        self._rebuild_tree()
        self.show_properties()
        if rebuild_view:
            self.draw()

    def _item(self, text, kind, obj=None, key=None, bold=False, color=None, tip=None):
        it = QTreeWidgetItem([text])
        it.setData(0, KIND, kind)
        it.setData(0, OBJ, obj)
        it.setData(0, KEY, key)
        if bold:
            f = QFont()
            f.setBold(True)
            it.setFont(0, f)
        if color:
            it.setForeground(0, QBrush(QColor(color)))
        if tip:
            it.setToolTip(0, tip)
        return it

    def _placeholder(self, parent, text):
        it = self._item(text, None, color="#7A8291")
        f = it.font(0)
        f.setItalic(True)
        it.setFont(0, f)
        parent.addChild(it)

    STATUS_MARK: ClassVar[dict] = {"applied": "✓ ", "changed": "◐ ", "failed": "◐ ", "none": "○ "}

    def _segment_text(self, s: J.Segment) -> str:
        side = f" ({s.side})" if s.side in ("right", "left") else ""
        return self.STATUS_MARK[self.model.status(s.name)] + f"{s.name}{side} · {s.stype.title}"

    def _segment_tip(self, s: J.Segment) -> str:
        st = self.model.status(s.name)
        surf = Path(s.surface.path).name if s.surface is not None else "no surface"
        return {"applied": "Axes applied", "changed": "Changed since the axes were applied: Apply to update",
                "failed": "The axes were applied, but the current definition doesn't give axes: " + s.error,
                "none": "Axes not applied yet"}[st] + f"\nSurface: {surf}\nCtrl+click another bone to add a joint"

    def _joint_text(self, j: J.Joint) -> str:
        ok = all(self.model.segment(n).acs is not None for n in (j.proximal, j.distal))
        return ("✓ " if ok else "○ ") + f"{j.name}: {j.proximal} → {j.distal}"

    @staticmethod
    def _reference_text(j: J.Joint) -> str:
        r = j.reference
        if r is None:
            return "Reference pose: none (anatomical zero)"
        kind = r.source.get("kind")
        what = {"trial": f"neutral trial, {r.source.get('frames_used', '?')} frame(s)", "scan": "scan pose",
                "scan_space": "scan pose"}.get(kind, "set")
        return f"Reference pose: {what}"

    def _rebuild_tree(self) -> None:
        m = self.model
        self.tree.blockSignals(True)
        self.tree.clear()
        current = None
        top = self._item(m.name, "model", bold=True)
        self.tree.addTopLevelItem(top)
        segs = self._item("1  Bones", "segments", bold=True, tip="Right-click to add bones from surface files")
        self.tree.addTopLevelItem(segs)
        if not m.segments:
            self._placeholder(segs, "Right-click to add bones from their surfaces")
        pair = self.sel[1] if self.sel[0] == "pair" else ()
        pair_items = []
        for s in m.segments:
            auto = s.axes_source != "landmarks"
            si = self._item(self._segment_text(s), "segment", s,
                            color="#E6E8EC" if s.surface is not None else "#F2C48D", tip=self._segment_tip(s))
            segs.addChild(si)
            if self.sel[0] == "segment" and self.sel[1] is s:
                current = si
            if any(x is s for x in pair):
                pair_items.append(si)
            if auto:
                am = AA.AUTO_METHODS.get(s.axes_source)
                si.addChild(self._item("Axes: " + J.axes_source_title(s.axes_source), "axesinfo", s,
                                       color="#B78AF0", tip=am.summary if am else J.AAFACT_TITLE))
            for d in s.stype.landmarks:
                done = self.model.landmark(s.name, d.key) is not None
                if auto and not done:
                    continue                      # unused and empty: keep the tree short
                lm = self._item(("✓ " if done else "○ ") + d.name, "landmark", s, d.key,
                                color=("#5E8F72" if auto else "#8FD3A9") if done else "#C3C8D2",
                                tip=("Not used while the axes are automatic. " if auto else "") + d.hint)
                si.addChild(lm)
                if self.sel[0] == "landmark" and self.sel[1] is s and self.sel[2] == d.key:
                    current = lm
            si.setExpanded(True)
        joints = self._item("2  Joints (optional)", "joints", bold=True,
                            tip="Ctrl+click two bones with applied axes to add a joint between them")
        self.tree.addTopLevelItem(joints)
        if not m.joints:
            self._placeholder(joints, "Ctrl+click two bones with axes to join them")
        for j in m.joints:
            ji = self._item(self._joint_text(j), "joint", j, tip=j.jtype.title)
            joints.addChild(ji)
            if self.sel[0] == "joint" and self.sel[1] is j:
                current = ji
            ri = self._item("3  " + self._reference_text(j), "reference", j,
                            color="#8FD3A9" if j.reference is not None else "#A3AAB8",
                            tip="Optional: a pose whose joint values are taken as zero")
            ji.addChild(ri)
            if self.sel[0] == "reference" and self.sel[1] is j:
                current = ri
            ji.setExpanded(True)
        segs.setExpanded(True)
        joints.setExpanded(True)
        if len(pair_items) == 2:
            self.tree.clearSelection()
            self.tree.setCurrentItem(pair_items[1])
            for it in pair_items:
                it.setSelected(True)
        else:
            if current is None:
                current = {"model": top, "segments": segs, "joints": joints}.get(self.sel[0], top)
            self.tree.setCurrentItem(current)
        self.tree.blockSignals(False)

    def _tree_selected(self, cur, _prev) -> None:
        if cur is None or cur.data(0, KIND) is None:
            return
        if len(self.tree.selectedItems()) > 1:
            return                                   # several selected: see _selection_changed
        kind = cur.data(0, KIND)
        if kind == "axesinfo":
            kind = "segment"
        self.sel = (kind, cur.data(0, OBJ), cur.data(0, KEY))
        self._after_select()

    def _selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if len(items) < 2:
            return
        segs = [it.data(0, OBJ) for it in items if it.data(0, KIND) == "segment"]
        if len(segs) == 2 and len(items) == 2:
            self.sel = ("pair", (segs[0], segs[1]), None)
            self._after_select()
        else:
            self.hint.setText("Select exactly two bones (Ctrl+click) to add a joint between them.")

    def _after_select(self) -> None:
        kind, obj, _key = self.sel
        if kind == "landmark" and obj is not None and obj.axes_source == "landmarks":
            tgt = self._tool_target()
            if tgt is not None and self.tool in ("navigate", "point", "paint", "erase"):
                self.tool = "point" if tgt[2] == "point" else ("erase" if self.tool == "erase" else "paint")
        self.show_properties()
        self.draw()

    def select_pair(self, a: J.Segment, b: J.Segment) -> None:
        self.sel = ("pair", (a, b), None)
        self.refresh()

    def select(self, kind, obj=None, key=None) -> None:
        self.sel = (kind, obj, key)
        self.refresh()

    def _context(self, pos) -> None:
        if self.sel[0] == "pair" and len(self.tree.selectedItems()) == 2:
            menu = self.build_menu("pair", self.sel[1])
            menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        it = self.tree.itemAt(pos)
        kind = (it.data(0, KIND) if it else None) or "model"
        obj, key = (it.data(0, OBJ), it.data(0, KEY)) if it else (None, None)
        if it is not None and it.data(0, KIND) is not None:
            self.tree.setCurrentItem(it)
        menu = self.build_menu(kind, obj, key)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def build_menu(self, kind, obj=None, key=None) -> QMenu:
        menu = QMenu(self)

        def add(text, fn, enabled=True, checked=None, parent=menu):
            a = QAction(text, parent)
            a.triggered.connect(lambda _=False: fn())
            a.setEnabled(enabled)
            if checked is not None:
                a.setCheckable(True)
                a.setChecked(checked)
            parent.addAction(a)
            return a

        if kind in ("model", "segments"):
            add("Add bones from surfaces…", self.add_surfaces)
        if kind in ("model", "joints"):
            add("Add joint…", self.add_joint, enabled=len(self._with_axes()) >= 2)
        if kind == "model":
            menu.addSeparator()
            add("Save", self.save_model)
            add("Save as…", self.save_model_as)
            add("Open model…", self.open_model)
        elif kind == "pair":
            a, b = obj
            ok = a.acs is not None and b.acs is not None
            add(f"Add joint between {a.name} and {b.name}…", lambda: self.select_pair(a, b), enabled=ok)
        elif kind in ("segment", "axesinfo"):
            s: J.Segment = obj
            add("Replace surface…" if s.surface else "Load surface…", lambda: self.load_surface(s))
            add("Preview tracking (.tra)…", lambda: self.attach_preview(s), enabled=s.surface is not None)
            menu.addSeparator()
            add("Apply axes", lambda: self.apply_axes(s), enabled=s.surface is not None)
            for src, title in self.model.axes_sources(s.name):
                add("Axes from landmarks" if src == "landmarks" else title,
                    lambda src=src: self.set_axes_source(s, src), checked=s.axes_source == src)
            menu.addSeparator()
            others = [o for o in self._with_axes() if o is not s]
            if s.acs is not None and others:
                sub = menu.addMenu("Add joint with")
                for o in others:
                    add(o.name, lambda o=o: self.select_pair(s, o), parent=sub)
            add("Clear all landmarks", lambda: self.clear_landmarks(s), enabled=bool(s.landmarks))
            add("Remove bone", lambda: self.remove_segment(s))
        elif kind == "landmark":
            add("Clear this landmark", lambda: self.clear_landmark(obj, key),
                enabled=self.model.landmark(obj.name, key) is not None)
        elif kind == "joint":
            add("Reference pose…", lambda: self.set_reference(obj))
            add("Remove joint", lambda: self.remove_joint(obj))
        elif kind == "reference":
            add("From a neutral trial…", lambda: self.set_reference(obj, "trial"))
            add("From the scan pose…", lambda: self.set_reference(obj, "scan"))
            add("Clear reference pose", lambda: self.clear_reference(obj), enabled=obj.reference is not None)
        return menu

    # ================================================================ properties
    def show_properties(self) -> None:
        kind, obj, key = self.sel
        if kind == "segment" and obj is not None:
            self._props_segment(obj)
        elif kind == "landmark" and obj is not None:
            self._props_segment(obj, key)
        elif kind == "pair" and obj is not None:
            self._props_pair(*obj)
        elif kind == "joint" and obj is not None:
            self._props_joint(obj)
        elif kind == "reference" and obj is not None:
            self._props_reference(obj)
        elif kind == "segments":
            self._props_segments()
        elif kind == "joints":
            self._props_joints()
        else:
            self._props_model()
        self.props.finish()
        self._update_tools()
        self._update_hint()

    def _props_model(self) -> None:
        p = self.props
        m = self.model
        p.clear(m.name)
        p.line("Name", m.name, self._rename)
        p.section("Steps")
        p.note("1  Bones: add each bone's surface, say which bone it is, choose how its axes are made, define "
               "them (place or paint landmarks, or run an automatic method) and Apply. On its own this gives each "
               "bone's motion about its own axes in the Visualiser.")
        p.note("2  Joints (optional): Ctrl+click two bones with applied axes and choose a definition, e.g. Grood & "
               "Suntay for femur → tibia.")
        p.note("3  Reference pose (optional, per joint): a neutral trial or the scan pose whose joint values are "
               "taken as zero. Without one, zero is anatomical (the bones' axes lined up).")
        p.buttons([("Add bones…", self.add_surfaces), ("Open model…", self.open_model)], primary="Add bones…")
        p.section("Contents")
        p.value("Bones", ", ".join(f"{s.name} ({self.model.status(s.name).replace('none', 'not applied')})"
                                   for s in m.segments) or "none")
        p.value("Joints", ", ".join(j.name for j in m.joints) or "none")
        p.section("File")
        p.value("Saved as", (m.path.name if m.path else "Not saved yet") + (" · unsaved changes" if self._dirty else ""),
                tip=str(m.path) if m.path else None)
        p.buttons([("Save", self.save_model), ("Save as…", self.save_model_as)])

    def _props_segments(self) -> None:
        p = self.props
        p.clear("Bones")
        p.note("Each bone is one surface with its own axes. Add the surfaces first (STL, or TIF partial volumes: "
               "the same files and model space you track with); each file becomes a bone, named and typed from its "
               "file name where possible. Then, for each bone: say which bone it is, choose the axes method, define "
               "it and Apply.")
        p.buttons([("Add bones…", self.add_surfaces)], primary="Add bones…")

    def _props_joints(self) -> None:
        p = self.props
        p.clear("Joints")
        p.note("Joints are optional. A joint pairs two bones whose axes are applied: Ctrl+click the two bones in "
               "the navigator (or right-click a bone → Add joint with). The definitions offered fit the two bones:")
        for t in J.JOINT_TYPES.values():
            p.note(f"• {t.title}")
        p.buttons([("Add joint…", self.add_joint, len(self._with_axes()) >= 2,
                    "Apply the axes of at least two bones first")])

    def _with_axes(self) -> list:
        return [s for s in self.model.segments if s.acs is not None]

    def _props_pair(self, a: J.Segment, b: J.Segment) -> None:
        p = self.props
        m = self.model
        p.clear(f"{a.name} + {b.name}")
        missing = [x.name for x in (a, b) if x.acs is None]
        if missing:
            p.note(f"Apply the axes of {' and '.join(missing)} first: a joint uses both bones' applied axes.", "Warn")
            return
        p.note("Add a joint between these two bones. The definitions listed fit them; the general ones work for "
               "any two bones (swap which is proximal if needed).")
        opts = m.joint_types_for(a.name, b.name)
        if not hasattr(self, "_pair_choice") or self._pair_choice[0] != (id(a), id(b)):
            self._pair_choice = ((id(a), id(b)), 0, False)
        idx, swapped = self._pair_choice[1], self._pair_choice[2]
        idx = min(idx, len(opts) - 1)
        k, prox, dist = opts[idx]
        jt = J.JOINT_TYPES[k]
        general = not jt.proximal_types and not jt.distal_types
        if general and swapped:
            prox, dist = dist, prox
        p.combo("Definition", [(i, J.JOINT_TYPES[o[0]].title) for i, o in enumerate(opts)], idx,
                lambda i: self._set_pair_choice(a, b, i, swapped))
        p.note(jt.summary)
        p.value("Proximal", prox)
        p.value("Distal", dist)
        if general:
            p.buttons([("Swap proximal and distal", lambda: self._set_pair_choice(a, b, idx, not swapped))])
        name = p.line("Name", m._unique(jt.default_name, {j.name for j in m.joints}), lambda n: None)
        p.buttons([("Add joint", lambda: self.add_joint(k, prox, dist, name.text().strip() or None))],
                  primary="Add joint")

    def _set_pair_choice(self, a, b, idx, swapped) -> None:
        self._pair_choice = ((id(a), id(b)), idx, swapped)
        self.show_properties()

    STATUS_TEXT: ClassVar[dict] = {"applied": ("Applied", "Value"), "changed": ("Changed since applied: Apply to update", "Warn"),
                   "failed": ("Applied, but the current definition doesn't give axes", "Warn"),
                   "none": ("Not applied yet", "Warn")}

    def _props_segment(self, s: J.Segment, active: str | None = None) -> None:
        p = self.props
        t = s.stype
        m = self.model
        p.clear(s.name)
        p.line("Name", s.name, lambda n: self._rename_segment(s, n))
        # 1 surface
        p.section("1  Surface")
        if s.surface is None:
            p.note("Load this bone's surface (STL, or a TIF partial volume): the same file and model space you "
                   "track with.", "Warn")
            p.buttons([("Load surface…", lambda: self.load_surface(s))], primary="Load surface…")
            return
        p.value("File", Path(s.surface.path).name, tip=str(s.surface.path))
        if s.surface.surface_note:
            p.value("Space", s.surface.surface_note)
        p.buttons([("Replace…", lambda: self.load_surface(s))])
        # 2 bone
        p.section("2  Bone")
        items = []
        for title, keys in BONE_GROUPS:
            items += [(k, (f"{title} · " if title != "Any bone" else "") + J.SEGMENT_TYPES[k].title) for k in keys]
        p.combo("Bone", items, s.type, lambda k: self.set_segment_type(s, k),
                tip="Which bone this is: it decides the methods offered for its axes. General works for any bone.")
        if t.anatomical:
            p.combo("Side", [("right", "Right"), ("left", "Left")], s.side, lambda v: self._set_side(s, v),
                    tip="Sets which way is right (ISB Z) for landmark axes, and joint angle signs on the left side. "
                        "The automatic (STAPLE) methods also use it to tell medial from lateral.")
        # 3 method
        p.section("3  Axes method")
        sources = m.axes_sources(s.name)
        p.combo("Method", sources, s.axes_source, lambda src: self.set_axes_source(s, src),
                tip="An automatic method fitted to the surface, or landmarks you place")
        # 4 define
        p.section("4  Define")
        if s.axes_source == "landmarks":
            p.note(t.axes_note or "")
            for ref in t.references:
                p.note(ref)
            for d in t.landmarks:
                lm = s.landmarks.get(d.key) or {"method": d.methods[0], "points": []}
                v = m.landmark(s.name, d.key)
                is_active = active == d.key
                verb = "Place" if lm["method"] == "point" else "Paint"
                status = ("✓ placed" if lm["method"] == "point" else
                          f"✓ painted ({len(lm.get('points') or [])} points)") if v is not None else "○ not set"
                p.value(("▸ " if is_active else "") + d.name, status, tip=d.hint,
                        kind="Value" if v is not None else "Warn")
                p.buttons([(verb, lambda d=d: self.edit_landmark(s, d.key)),
                           ("Clear", lambda d=d: self.clear_landmark(s, d.key), v is not None or bool(lm.get("points")))],
                          primary=verb if is_active else None)
                if is_active:
                    p.note(d.hint)
                    if len(d.methods) > 1:
                        p.combo("Method", [(mm, J.LANDMARK_METHODS[mm]) for mm in d.methods], lm["method"],
                                lambda mm, d=d: self._set_method(s, d.key, mm))
                    if lm["method"] == "sphere" and v is not None and len(lm.get("points") or []) >= 4:
                        p.value("Sphere radius", f"{J.sphere_fit(np.asarray(lm['points']))[1]:.1f} mm")
        elif s.axes_source == "aafact":
            self._props_aafact(s)
        else:
            am = AA.AUTO_METHODS[s.axes_source]
            p.note("Fitted to the surface when you Apply (a few seconds the first time). How it works: see "
                   "About this method, below.")
            whole = bool(s.auto is not None and str(s.auto.quality.get("Bone", "")).startswith("whole"))
            if not whole and t.anatomical:
                p.buttons([("Flip anterior/posterior", lambda: self._toggle_option(s, "flip_ap")),
                           ("Flip superior/inferior", lambda: self._toggle_option(s, "flip_long"))])
            flips = [n for k, n in (("flip_ap", "anterior/posterior"), ("flip_long", "superior/inferior"))
                     if s.auto_options.get(k)]
            if flips:
                p.note("Flipped: " + ", ".join(flips))
        # 5 apply
        p.section("5  Apply")
        st = m.status(s.name)
        text, kind = self.STATUS_TEXT[st]
        p.value("Axes", text, kind=kind)
        if s.error and st != "applied" and s.error != "Not applied yet.":
            p.note(s.error, "Warn")
        p.buttons([("Apply", lambda: self.apply_axes(s))], primary="Apply" if st != "applied" else None)
        p.check("After applying", s.auto_update, lambda on: self._set_auto_update(s, on),
                caption="Update automatically",
                tip="Once applied, changing landmarks (e.g. repainting an area), flips or options updates the axes "
                    "straight away. Untick to keep the applied axes until you Apply again.")
        if s.acs is not None:
            if s.auto is not None:
                for k, v in s.auto.quality.items():
                    p.value(k, f"{v:.2f}" if isinstance(v, float) else str(v))
                for n in s.auto.notes:
                    p.note(n, "Warn")
                p.check("Show fit", self.show_fit, self._set_show_fit, "Draw the fitted shape and points")
            names = ("X", "Y", "Z") if t.anatomical and s.axes_source != "inertial" else ("1", "2", "3")
            for k, n in enumerate(names):
                p.value(f"Axis {n}", "  ".join(f"{v:+.3f}" for v in s.acs[:3, k]))
            p.value("Origin", "  ".join(f"{v:.1f}" for v in s.acs[:3, 3]) + " mm")
            others = [o for o in self._with_axes() if o is not s]
            if others:
                p.note("Ctrl+click another bone with axes in the navigator to add a joint.")
        p.section("Preview tracking (optional)")
        p.value("Tracking", Path(s.preview_tracking).name if s.preview_tracking else "None",
                tip="Lets joints be previewed in a tracked pose, and offered as the reference trial.")
        p.buttons([("Attach .tra…", lambda: self.attach_preview(s))])
        if s.axes_source in AA.AUTO_METHODS:
            am = AA.AUTO_METHODS[s.axes_source]
            p.section("About this method")
            p.note(am.summary)
            for ref in am.references:
                p.note(ref)
        p.section("")
        p.buttons([("Remove bone", lambda: self.remove_segment(s))], danger="Remove bone")

    def _props_aafact(self, s: J.Segment) -> None:
        from ..core import aafact as AF

        p = self.props
        p.note("AAFACT aligns the bone to a template and builds its axes from regional centroids: X = AP "
               "(anterior), Y = SI (superior), Z = ML (pointing right). It is a separate program: install the "
               "compiled AAFACT (with the MATLAB Runtime) from github.com/Lenz-Lab/AAFACT, then run it here, or "
               "import the CoordinateSystem_….xlsx it writes. Set the bone's side first.")
        p.note(AF.PETERSON)
        exe = _settings().value("aafact/exe", "", str)
        p.value("AAFACT program", Path(exe).name if exe else "Not set", tip=exe or None,
                kind="Value" if exe else "Warn")
        p.buttons([("Run AAFACT on this bone", lambda: self.run_aafact(s)),
                   ("Import results (.xlsx)…", lambda: self.import_aafact(s))])
        p.buttons([("Set AAFACT program…", self._choose_aafact_exe)])
        imp = s.imported if s.imported and s.imported.get("method") == "aafact" else None
        if imp:
            opts = imp["options"]
            if len(opts) > 1:
                p.combo("Coordinate system", [(k, v["title"]) for k, v in opts.items()], imp["option"],
                        lambda k: self._set_aafact_option(s, k),
                        tip="AAFACT offers several systems for this bone")
            cur = opts[imp["option"]]
            p.value("From", Path(imp["file"]).name if imp["file"] else "AAFACT", tip=imp["file"] or None)
            p.value("Origin", cur.get("origin", ""))
            dev = float(cur.get("orthogonality", 0))
            p.value("Axes orthogonality", f"{dev:.2f}° from 90°", kind="Value" if dev < 2 else "Warn",
                    tip="AAFACT's axes are made exactly orthogonal here (nearest rotation)")

    def _choose_aafact_exe(self) -> str:
        from PySide6.QtWidgets import QFileDialog
        st = _settings()
        path, _ = QFileDialog.getOpenFileName(self, "Find the compiled AAFACT program", st.value("aafact/exe", "", str),
                                              "AAFACT program (AAFACT.exe AAFACT *.exe *);;All files (*)")
        if path:
            st.setValue("aafact/exe", path)
            self._changed()
        return path

    def _set_aafact_option(self, s: J.Segment, key: str) -> None:
        self.model.set_imported_option(s.name, key)
        self._changed()

    def import_aafact(self, s: J.Segment, path: str | None = None) -> bool:
        from PySide6.QtWidgets import QFileDialog
        from ..core import aafact as AF
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "AAFACT results", "", "AAFACT results (*.xlsx);;All files (*)")
            if not path:
                return False
        try:
            res = AF.read_results(path)
            self.model.set_imported(s.name, res, file=path)
        except (AF.AafactError, J.JointModelError) as err:
            QMessageBox.warning(self, "AAFACT results", str(err))
            return False
        s.applied = s.applied or not self.model.require_apply
        self._changed()
        return True

    def run_aafact(self, s: J.Segment) -> bool:
        import tempfile
        from PySide6.QtGui import QGuiApplication
        from ..core import aafact as AF
        if s.mesh is None:
            QMessageBox.information(self, "Run AAFACT", "Load this bone's surface first.")
            return False
        exe = _settings().value("aafact/exe", "", str) or self._choose_aafact_exe()
        if not exe:
            return False
        bone = AF.BONES.get(s.type)
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            work = Path(tempfile.mkdtemp(prefix="aafact_"))
            res = AF.run(exe, s.mesh, bone, s.side, work)
            self.model.set_imported(s.name, res, file=str(next(work.glob("*/CoordinateSystem_*.xlsx"), "")))
        except (AF.AafactError, J.JointModelError) as err:
            QMessageBox.warning(self, "Run AAFACT", str(err))
            return False
        finally:
            QGuiApplication.restoreOverrideCursor()
        self._changed()
        return True

    def _fmt(self, v) -> str:
        return f"{(0.0 if abs(v) < 0.05 else v):+.1f}"

    def _props_joint(self, j: J.Joint) -> None:
        p = self.props
        m = self.model
        conv = j.convention
        p.clear(j.name)
        p.line("Name", j.name, lambda n: self._rename_joint(j, n))
        p.value("Definition", j.jtype.title)
        p.note(j.jtype.summary)
        for ref in j.jtype.references:
            p.note(ref)
        p.section("Bones")
        p.combo("Proximal", [(s.name, s.name) for s in m.candidates(j.type, "proximal")], j.proximal,
                lambda n: self._set_joint_segment(j, "proximal", n))
        p.combo("Distal", [(s.name, s.name) for s in m.candidates(j.type, "distal")], j.distal,
                lambda n: self._set_joint_segment(j, "distal", n))
        p.section("Angles")
        for n in conv["angles"]:
            p.note("• " + n)
        p.section("Preview")
        k = self._preview_kinematics(j)
        if k is None:
            p.note("Attach preview tracking to both bones to see the joint in a tracked pose.")
        else:
            f, raw, norm = k
            p.value("Frame", str(f + 1))
            p.note("Anatomical values" + (" (then relative to the reference pose)" if norm is not None else "") + ":")
            for i, name in enumerate(conv["short"]):
                p.value(name, f"{self._fmt(raw['angles'][0, i])}°" +
                        (f"   ref: {self._fmt(norm['angles'][0, i])}°" if norm is not None else ""))
            for i, name in enumerate(conv["short_t"]):
                p.value(name, f"{self._fmt(raw['translations'][0, i])} mm" +
                        (f"   ref: {self._fmt(norm['translations'][0, i])} mm" if norm is not None else ""))
        p.section("Reference pose (optional)")
        p.value("Reference", self._reference_text(j).split(": ", 1)[1])
        p.buttons([("Set reference pose…", lambda: self.set_reference(j))])
        if any(len(m.axes_sources(n)) > 1 for n in (j.proximal, j.distal)):
            p.section("Compare axis methods")
            cmp = self._compare if self._compare and self._compare[0] is j else None
            if cmp is None:
                p.note("Anatomical joint angles at the preview frame for every combination of the bones' axis "
                       "methods (landmarks, if placed, and each automatic method).")
            else:
                p.note(f"Frame {cmp[1] + 1}. " + " / ".join(conv["short"]) + " (°). ▸ marks the methods in use.")
                rows = [r for r in cmp[2] if r["angles"] is not None]
                for row in rows:
                    name = f"{'▸ ' if row['current'] else ''}{SHORT_SOURCE.get(row['proximal'], row['proximal'])} · " \
                           f"{SHORT_SOURCE.get(row['distal'], row['distal'])}"
                    p.value(name, "  ".join(self._fmt(v) for v in row["angles"]),
                            tip=f"{row['proximal_title']}\n{row['distal_title']}")
                if len(rows) < len(cmp[2]):
                    p.note(f"{len(cmp[2]) - len(rows)} combination(s) left out: their axes can't be made yet "
                           "(e.g. landmarks not placed).")
            p.buttons([("Compare methods", lambda: self.compare_methods(j))])
        warns = [w for w in m.checks() if w.startswith((j.name + ":", j.proximal + ":", j.distal + ":"))]
        if warns:
            p.section("Checks")
            for w in warns:
                p.note(w, "Warn")
        p.section("")
        p.buttons([("Remove joint", lambda: self.remove_joint(j))])

    def _props_reference(self, j: J.Joint) -> None:
        p = self.props
        r = j.reference
        p.clear(f"{j.name}: reference pose")
        p.note("Optional. The joint's values in this pose are taken as zero, e.g. a standing/neutral trial, or the "
               "pose the bones had in the scan. It's stored as the two bones' relative pose, so it stays valid if "
               "either bone's axes are changed later.")
        if r is None:
            p.value("Reference", "None: zero is anatomical (the bones' axes lined up)")
        else:
            src = r.source
            kind = src.get("kind")
            if kind == "trial":
                p.value("Source", "Neutral / static trial")
                p.value(j.proximal, Path(src.get("proximal", "")).name, tip=src.get("proximal"))
                p.value(j.distal, Path(src.get("distal", "")).name, tip=src.get("distal"))
                fr = src.get("frames")
                p.value("Frames", "all tracked" if fr == "all" else f"{fr[0]}–{fr[1]}" if fr else "?")
                p.value("Frames used", str(src.get("frames_used", "?")))
            elif kind == "scan":
                p.value("Source", "Pose in the scan")
                p.value(j.proximal, Path(src.get("proximal", "")).name, tip=src.get("proximal"))
                p.value(j.distal, Path(src.get("distal", "")).name, tip=src.get("distal"))
            else:
                p.value("Source", "Pose in the scan (surfaces share its space)")
            p.combo("Normalise by", list(J.REFERENCE_MODES.items()), r.mode, lambda md: self._set_ref_mode(j, md))
            try:
                v = self.model.reference_values(j.name, "subtract")
                p.section("Anatomical values at the reference")
                for name, x in zip(j.convention["short"], v["angles"][0]):
                    p.value(name, f"{self._fmt(x)}°")
                for name, x in zip(j.convention["short_t"], v["translations"][0]):
                    p.value(name, f"{self._fmt(x)} mm")
                p.note("These are subtracted from the joint's values (or, when realigning, the distal axes are "
                       "turned to remove the rotations).")
            except J.JointModelError as err:
                p.note(str(err), "Warn")
        p.buttons([("From a neutral trial…", lambda: self.set_reference(j, "trial")),
                   ("From the scan pose…", lambda: self.set_reference(j, "scan"))])
        if r is not None:
            p.buttons([("Clear reference", lambda: self.clear_reference(j))])

    # ================================================================ actions
    def _changed(self, rebuild_view: bool = True) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)     # automatic methods take a few seconds the first time
        try:
            self.model.compute()
        finally:
            QApplication.restoreOverrideCursor()
        self._set_dirty(True)
        self.refresh(rebuild_view)

    def _set_dirty(self, on: bool) -> None:
        if on != self._dirty:
            self._dirty = on
            self.titleChanged.emit(self.state.name + (" •" if on else ""))

    def has_unsaved(self) -> bool:
        return self._dirty and bool(self.model.segments or self.model.joints)

    def _rename(self, name: str) -> None:
        self.model.name = name
        self.state.name = name
        self.titleChanged.emit(name + (" •" if self._dirty else ""))
        self._rebuild_tree()

    # ---- bones from surfaces
    def add_surfaces(self, paths=None, voxel_size=None) -> list:
        """Add a bone for each surface file (asking for files if none are given). The bone's type and side are
        guessed from the file name (e.g. AUT_Left_Femur.stl: a left femur); General if it can't be told."""
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Add bones: choose their surfaces", "", MODEL_FILTER)
            if not paths:
                return []
        added = []
        for path in paths:
            bm = self._build_surface(path, voxel_size)
            if bm is None:
                continue
            typ = J.guess_segment_type(Path(path).name) or "generic"
            side = _guess_side(Path(path).name) or next(
                (x.side for x in self.model.segments if x.side in ("right", "left")), "right")
            name = J.SEGMENT_TYPES[typ].title if typ != "generic" else Path(path).stem
            seg = self.model.add_segment(typ, name, side)
            seg.surface = bm
            added.append(seg)
        if added:
            self.sel = ("segment", added[0], None)
            self._changed()
            self.plotter.reset_camera()
        return added

    def _build_surface(self, path, voxel_size=None):
        p = Path(path)
        bm = BoneModel(name=p.stem, path=p)
        if is_volume_file(p):
            if voxel_size is None:
                dlg = VoxelDialog(self, p.name, None)
                if dlg.exec() != QDialog.Accepted:
                    return None
                bm.voxel_size, bm.flip, bm.threshold = dlg.values()
            else:
                bm.voxel_size = tuple(voxel_size)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bm.build()
        except (ModelError, OSError, ValueError) as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't load the surface", f"{p.name}: {err}")
            return None
        QApplication.restoreOverrideCursor()
        return bm

    def set_segment_type(self, s: J.Segment, type_key: str) -> None:
        if type_key == s.type:
            return
        gone = [j.name for j in self.model.joints if s.name in (j.proximal, j.distal)]
        self._push_undo(s)
        self.model.set_segment_type(s.name, type_key)
        gone = [n for n in gone if n not in {j.name for j in self.model.joints}]
        if gone:
            QMessageBox.information(self, "Bone changed", f"Joint(s) {', '.join(gone)} no longer fit this bone and "
                                    "were removed.")
        self._compare = None
        self.sel = ("segment", s, None)
        self._changed()

    def apply_axes(self, s: J.Segment) -> bool:
        """Make the bone's axes from its current definition and use them."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.model.apply(s.name)
            ok = True
        except J.JointModelError:
            ok = False
        finally:
            QApplication.restoreOverrideCursor()
        self._compare = None
        self._set_dirty(True)
        if self.sel[0] not in ("segment", "landmark") or self.sel[1] is not s:
            self.sel = ("segment", s, None)
        self.refresh()
        if not ok:
            self.hint.setText(f"{s.name}'s axes couldn't be made: {s.error}")
        return ok

    def _set_auto_update(self, s: J.Segment, on: bool) -> None:
        s.auto_update = bool(on)
        self._changed()

    def edit_landmark(self, s: J.Segment, key: str) -> None:
        """Select a landmark to place or paint, and the tool for it."""
        self.sel = ("landmark", s, key)
        self.tool = "navigate"
        self._rebuild_tree()
        self._after_select()

    def add_segment(self, type_key: str, name: str | None = None, side: str | None = None) -> J.Segment:
        side = side or next((s.side for s in self.model.segments if s.side in ("right", "left")), "right")
        s = self.model.add_segment(type_key, name, side)
        self.sel = ("segment", s, None)
        self._changed()
        return s

    def add_preset(self, key: str) -> None:
        side = next((s.side for s in self.model.segments if s.side in ("right", "left")), "right")
        p = J.PRESETS[key]
        names = {}
        for t, n in p["segments"]:
            names[n] = self.model.add_segment(t, n, side).name
        for t, a, b, n in p["joints"]:
            self.model.add_joint(t, names[a], names[b], n)
        self.sel = ("segment", self.model.segment(names[p["segments"][0][1]]), None)
        self._changed()

    def remove_segment(self, s: J.Segment) -> None:
        used = [j.name for j in self.model.joints if s.name in (j.proximal, j.distal)]
        msg = f"Remove {s.name}?" + (f" Its joint(s) {', '.join(used)} will be removed too." if used else "")
        if QMessageBox.question(self, "Remove bone", msg) != QMessageBox.Yes:
            return
        self.model.remove_segment(s.name)
        self.sel = ("segments", None, None)
        self._changed()

    def _rename_segment(self, s: J.Segment, name: str) -> None:
        self.model.rename_segment(s.name, name)
        self._rebuild_tree()
        self.props.title.setText(s.name)

    def _rename_joint(self, j: J.Joint, name: str) -> None:
        self.model.rename_joint(j.name, name)
        self._rebuild_tree()
        self.props.title.setText(j.name)

    def _set_side(self, s: J.Segment, side: str) -> None:
        s.side = side
        self._changed()

    def _set_brush(self, mm: float) -> None:
        self.brush = float(mm)
        if hasattr(self, "brush_value"):
            self.brush_value.setText(f"{self.brush:.0f} mm")
            if self.brush_slider.value() != round(mm):
                self.brush_slider.setValue(round(mm))

    def _set_method(self, s, key, method) -> None:
        self._push_undo(s)
        lm = s.landmarks.setdefault(key, {"method": method, "points": []})
        lm["method"] = method
        self._changed()

    def set_axes_source(self, s: J.Segment, source: str) -> None:
        if s.axes_source == source:
            return
        self.model.set_axes_source(s.name, source)
        self._compare = None
        self._changed()

    def _toggle_option(self, s: J.Segment, key: str) -> None:
        o = s.auto_options
        o[key] = not o.get(key, False)
        if not o[key]:
            o.pop(key)
        self._compare = None
        self._changed()

    def _set_show_fit(self, on: bool) -> None:
        self.show_fit = bool(on)
        self.draw()

    def add_joint(self, type_key: str | None = None, proximal: str | None = None, distal: str | None = None,
                  name: str | None = None) -> J.Joint | None:
        if proximal is None or distal is None:
            dlg = AddJointDialog(self, self.model, type_key)
            if dlg.type.count() == 0:
                QMessageBox.information(self, "Add a joint", "Apply the axes of at least two bones first.")
                return None
            if dlg.exec() != QDialog.Accepted:
                return None
            type_key, proximal, distal, name = dlg.values()
        try:
            j = self.model.add_joint(type_key, proximal, distal, name or None)
        except J.JointModelError as err:
            QMessageBox.warning(self, "Add a joint", str(err))
            return None
        self.sel = ("joint", j, None)
        self._changed()
        return j

    def _set_joint_segment(self, j: J.Joint, end: str, name: str) -> None:
        other = j.distal if end == "proximal" else j.proximal
        if name == other:
            QMessageBox.warning(self, "Joint bones", "A joint needs two different bones.")
            self.show_properties()
            return
        setattr(j, end, name)
        self._compare = None
        self._changed()

    def remove_joint(self, j: J.Joint) -> None:
        self.model.remove_joint(j.name)
        self.sel = ("joints", None, None)
        self._changed()

    def set_reference(self, j: J.Joint, start: str = "trial") -> bool:
        dlg = ReferenceDialog(self, self.model, j, start)
        if dlg.exec() != QDialog.Accepted:
            return False
        self.sel = ("reference", j, None)
        self._set_dirty(True)
        self.refresh()
        return True

    def clear_reference(self, j: J.Joint) -> None:
        j.reference = None
        self._set_dirty(True)
        self.refresh()

    def _set_ref_mode(self, j: J.Joint, mode: str) -> None:
        if j.reference is not None:
            j.reference.mode = mode
            self._set_dirty(True)
            self.show_properties()

    def compare_methods(self, j: J.Joint) -> list | None:
        pp = self._preview_poses([self.model.segment(j.proximal), self.model.segment(j.distal)])
        if pp is None:
            QMessageBox.information(self, "Compare methods", "Attach a preview tracking file to both bones first "
                                    "(right-click each bone → Preview tracking).")
            return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rows = self.model.compare_methods(j.name, pp[1][0], pp[1][1])
        finally:
            QApplication.restoreOverrideCursor()
        self._compare = (j, pp[0], rows)
        self.show_properties()
        return rows

    # ---- files
    def load_surface(self, s: J.Segment) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Surface for {s.name}", "", MODEL_FILTER)
        if path:
            self.set_surface(s, path)

    def set_surface(self, s: J.Segment, path, voxel_size=None) -> bool:
        bm = self._build_surface(path, voxel_size)
        if bm is None:
            return False
        s.surface = bm
        self._painted = {k: v for k, v in self._painted.items() if k[0] != id(s)}
        self.sel = ("segment", s, None)
        self._changed()
        self.plotter.reset_camera()
        return True

    def attach_preview(self, s: J.Segment) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Tracking for {s.name} (preview)", "", TRA_FILTER)
        if path:
            self.set_preview(s, path)

    def set_preview(self, s: J.Segment, path) -> None:
        try:
            load_tracking(path)
        except (TrackingError, OSError) as err:
            QMessageBox.warning(self, "Couldn't read the tracking file", str(err))
            return
        s.preview_tracking = str(path)
        self._changed()

    def save_model(self) -> bool:
        """Save to the model's file (asking for one the first time)."""
        if self.model.path is None:
            return self.save_model_as()
        return self._save_to(self.model.path)

    def save_model_as(self) -> bool:
        default = str(self.model.path or f"{self.model.name}.bvrmodel.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save model as", default, JOINT_MODEL_FILTER)
        if not path:
            return False
        if not path.endswith(".json"):
            path += ".bvrmodel.json"
        return self._save_to(path)

    def _save_to(self, path) -> bool:
        try:
            self.model.save(path)
        except OSError as err:
            QMessageBox.warning(self, "Couldn't save the model", str(err))
            return False
        self._set_dirty(False)
        todo = [s.name for s in self.model.segments if s.acs is None]
        self.hint.setText(f"Saved {Path(path).name}." + (f" Not applied, so saved without axes: {', '.join(todo)}."
                                                        if todo else ""))
        self.show_properties()
        self.hint.setText(f"Saved {Path(path).name}." + (f" Not applied, so saved without axes: {', '.join(todo)}."
                                                        if todo else ""))
        return True

    def open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open model", "", JOINT_MODEL_FILTER)
        if path:
            self.load_model_file(path)

    def load_model_file(self, path) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = J.KinematicModel.load(path, require_apply=True)
        except Exception as err:  # noqa: BLE001 - shown to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't open the model", str(err))
            return
        QApplication.restoreOverrideCursor()
        self.model = m
        self.state.name = m.name
        self.titleChanged.emit(m.name)
        self._painted.clear()
        self._undo.clear()
        self.sel = ("model", None, None)
        self._dirty = True
        self._set_dirty(False)
        self.refresh()
        self.plotter.reset_camera()

    def _push_undo(self, s: J.Segment) -> None:
        import copy

        self._undo.append((s, s.type, copy.deepcopy(s.landmarks)))
        del self._undo[:-50]

    def undo(self) -> None:
        if not self._undo:
            self.hint.setText("Nothing to undo.")
            return
        s, typ, lms = self._undo.pop()
        if s not in self.model.segments:
            return self.undo()
        if s.type != typ:
            self.model.set_segment_type(s.name, typ)
        s.landmarks = lms
        self._painted = {k: v for k, v in self._painted.items() if k[0] != id(s)}
        self._changed()

    def clear_landmark(self, s, key) -> None:
        self._push_undo(s)
        s.landmarks.pop(key, None)
        self._painted.pop((id(s), key), None)
        self._changed()

    def clear_landmarks(self, s) -> None:
        self._push_undo(s)
        s.landmarks.clear()
        for k in [k for k in self._painted if k[0] == id(s)]:
            self._painted.pop(k)
        self._changed()

    # ================================================================ 3D view
    def _shown(self):
        """(segments to draw, joint being shown or None)."""
        kind, obj, _ = self.sel
        if kind in ("segment", "landmark") and obj is not None:
            return [obj], None
        if kind in ("joint", "reference") and obj is not None:
            return [self.model.segment(obj.proximal), self.model.segment(obj.distal)], obj
        if kind == "pair" and obj is not None:
            return list(obj), None
        return [s for s in self.model.segments if s.surface is not None], None

    def _preview_poses(self, segs):
        """Tracking poses at the first frame every listed segment is tracked, or None."""
        tr = []
        for s in segs:
            if not s.preview_tracking:
                return None
            try:
                tr.append(load_tracking(s.preview_tracking).poses)
            except (TrackingError, OSError):
                return None
        if not tr:
            return None
        n = min(len(t) for t in tr)
        for f in range(n):
            if all(not np.isnan(t[f]).any() for t in tr):
                return f, [t[f] for t in tr]
        return None

    def _preview_kinematics(self, j: J.Joint):
        p, d = self.model.segment(j.proximal), self.model.segment(j.distal)
        if p.acs is None or d.acs is None:
            return None
        pp = self._preview_poses([p, d])
        if pp is None:
            return None
        f, (P, D) = pp
        raw = self.model.joint_kinematics(j.name, P[None], D[None], normalise=False)
        norm = self.model.joint_kinematics(j.name, P[None], D[None]) if j.reference is not None else None
        return f, raw, norm

    def draw(self) -> None:
        pl = self.plotter
        for a in self._actors:
            pl.remove_actor(a, render=False)
        self._actors = []
        self._bone_actors = {}
        segs, joint = self._shown()
        segs = [s for s in segs if s.surface is not None]
        many = len(segs) > 1
        poses, label_ = None, ""
        if joint is not None and self.sel[0] == "reference" and joint.reference is not None and len(segs) == 2:
            poses = [np.eye(4), joint.reference.G0]       # the reference pose itself
            label_ = f"Showing: {joint.name} in its reference pose"
        elif many:
            pp = self._preview_poses(segs)
            if pp is not None:
                poses = pp[1]
                label_ = f"Showing: preview tracking, frame {pp[0] + 1}"
        offset = 0.0
        for i, s in enumerate(segs):
            mesh = s.mesh
            if poses is not None:
                T = poses[i]
            else:
                T = np.eye(4)
                if many:                               # no common pose: lay the bones side by side
                    b = np.array(mesh.bounds)
                    T[0, 3] = offset - b[0]
                    offset += (b[1] - b[0]) + 25
            act = pl.add_mesh(mesh, color=BONE_COLOURS[i % len(BONE_COLOURS)], smooth_shading=True, specular=0.2,
                              pickable=True, reset_camera=False,
                              opacity=1.0 if self.sel[0] == "landmark" and not many else 0.55)
            act.user_matrix = T
            self._actors.append(act)
            self._bone_actors[id(s)] = act
            size = float(np.linalg.norm(np.ptp(np.asarray(mesh.points), axis=0)))
            if s.acs is not None:
                self._actors += add_triad(pl, s.acs, max(size * 0.28, 15), pose=T)
            if s.auto is not None and self.show_fit and self.sel[0] == "segment":
                self._draw_fit(s.auto, T)
            self._draw_landmarks(s, T, size)
        if self.sel[0] in ("segment", "landmark") and segs:
            label_ = f"Showing: {segs[0].name} (its own model space)"
        elif many and poses is None:
            label_ = "Showing: bones side by side (no common pose: attach preview tracking, or set a reference)"
        self.showing.setText(label_)
        shown = tuple(id(s) for s in segs) + (poses is not None, self.sel[0] == "reference")
        if self._actors and shown != getattr(self, "_last_shown", None):   # new set of bones: fit them
            pl.reset_camera()
            self._last_shown = shown
        pl.render()

    def _draw_fit(self, res, T) -> None:
        """Draw an automatic method's fitted shapes and the surface points it used."""
        pl = self.plotter

        def add(mesh, **kw):
            a = pl.add_mesh(mesh, pickable=False, reset_camera=False, **kw)
            a.user_matrix = T
            self._actors.append(a)

        if res.points is not None and len(res.points):
            pts = np.asarray(res.points)[:: max(1, len(res.points) // 3000)]
            add(pv.PolyData(pts), color="#F0A04B", point_size=3, render_points_as_spheres=True)
        for prim in res.primitives:
            k = prim["kind"]
            if k == "cylinder":
                add(pv.Cylinder(center=prim["center"], direction=prim["direction"], radius=prim["radius"],
                                height=prim["height"] + 10, resolution=48, capping=False),
                    color="#B78AF0", opacity=0.3, style="wireframe", line_width=1)
                ax = prim["direction"] * (prim["height"] / 2 + 25)
                add(pv.Line(prim["center"] - ax, prim["center"] + ax), color="#B78AF0", line_width=3)
            elif k == "sphere":
                add(pv.Sphere(radius=prim["radius"], center=prim["center"], theta_resolution=24, phi_resolution=16),
                    color="#B78AF0", opacity=0.3, style="wireframe" if prim["radius"] > 5 else "surface")
            elif k == "ellipse":
                th = np.linspace(0, 2 * np.pi, 97)
                ring = prim["center"] + np.outer(np.cos(th) * prim["a"], prim["major"]) + \
                    np.outer(np.sin(th) * prim["b"], prim["minor"])
                add(pv.lines_from_points(ring), color="#B78AF0", line_width=3)
            elif k == "surface":
                F = np.asarray(prim["faces"], np.int64)
                if len(F):
                    add(pv.PolyData(np.asarray(prim["points"], float), np.c_[np.full(len(F), 3), F].ravel()),
                        color=prim.get("color", "#F0A04B"), opacity=0.85)
            elif k == "ellipsoid":
                e = pv.ParametricEllipsoid(*np.abs(prim["radii"]))
                M = np.eye(4)
                M[:3, :3] = prim["axes"]
                M[:3, 3] = prim["center"]
                add(e.transform(M, inplace=False), color="#B78AF0", opacity=0.3, style="wireframe", line_width=1)
            elif k == "loop" and len(prim["points"]) >= 3:
                c = np.asarray(prim["points"], float)
                add(pv.lines_from_points(np.vstack([c, c[:1]])), color="#B78AF0", line_width=3)
            elif k == "points" and len(prim["points"]):
                add(pv.PolyData(np.asarray(prim["points"], float)), color="#F0A04B", point_size=3,
                    render_points_as_spheres=True)
            elif k == "line" and prim["points"] is not None and len(prim["points"]) >= 2:
                c = np.asarray(prim["points"])
                add(pv.PolyData(c), color="#5CC8A8", point_size=7, render_points_as_spheres=True)
                add(pv.lines_from_points(c), color="#5CC8A8", line_width=2)

    def _draw_landmarks(self, s: J.Segment, T, size) -> None:
        pl = self.plotter
        active = self.sel[0] == "landmark" and self.sel[1] is s
        if s.axes_source != "landmarks" and not active:
            return
        rad = max(size * 0.012, 1.2)
        for d in s.stype.landmarks:
            lm = s.landmarks.get(d.key)
            if not lm:
                continue
            is_active = active and self.sel[2] == d.key
            pts = np.asarray(lm.get("points") or [], float).reshape(-1, 3)
            if lm["method"] != "point" and len(pts) and (is_active or not active):
                a = pl.add_mesh(pv.PolyData(pts), color="#F0A04B" if is_active else "#8A92A1", point_size=6,
                                render_points_as_spheres=True, pickable=False, reset_camera=False)
                a.user_matrix = T
                self._actors.append(a)
            v = self.model.landmark(s.name, d.key)
            if v is None:
                continue
            sp = pl.add_mesh(pv.Sphere(radius=rad * (1.6 if is_active else 1.0), center=v),
                             color="#F0A04B" if is_active else "#4F9CF9", pickable=False, reset_camera=False)
            sp.user_matrix = T
            self._actors.append(sp)
            if np.allclose(T, np.eye(4)):
                lab = pl.add_point_labels(pv.PolyData(v[None]), [d.key], font_size=11, text_color="#E6E8EC",
                                          shape_color="#1C1F26", shape_opacity=0.8, show_points=False,
                                          always_visible=True, pickable=False, reset_camera=False)
                self._actors.append(lab)

    def _update_hint(self) -> None:
        kind, obj, key = self.sel
        tool_tip = {"point": "Place point: click the surface to place “{n}” (click again to move it).",
                    "paint": "Paint: drag over the surface to paint “{n}”; it adds to the area. Brush {b:.0f} mm.",
                    "erase": "Erase: drag over “{n}”'s painted area to remove it. Brush {b:.0f} mm.",
                    "pick": "Inspect: click the surface to see the point's coordinates and nearest landmark."}
        if self.tool != "navigate":
            n = ""
            if kind == "landmark" and obj is not None:
                n = next(x for x in obj.stype.landmarks if x.key == key).name
            self.hint.setText(tool_tip[self.tool].format(n=n, b=self.brush) + "  " + TOOL_NAV_TIP)
        elif kind == "landmark" and obj is not None:
            if obj.surface is None:
                self.hint.setText(f"Load {obj.name}'s surface first.")
            elif obj.axes_source != "landmarks":
                self.hint.setText(f"{obj.name}'s axes are automatic: choose Landmarks as its method to place this.")
            else:
                self.hint.setText("Choose Place point or Paint in the tool bar (left of the view) to edit this "
                                  "landmark.")
        elif kind == "segment" and obj is not None and obj.axes_source != "landmarks":
            self.hint.setText("Automatic axes: purple = fitted shape, orange/pink = articular surfaces found "
                              "(lateral/medial), teal = shaft line. X red, Y green, Z blue. Left-drag rotates.")
        elif kind == "segment":
            self.hint.setText("Place or paint each landmark (Place / Paint in the steps), then Apply.")
        elif kind == "pair":
            self.hint.setText("Both bones are shown. Choose the joint definition on the left and Add joint.")
        elif kind == "reference":
            self.hint.setText("Shown in the reference pose when one is set. Its joint values are taken as zero.")
        elif kind == "joint":
            self.hint.setText("Both bones' axes are shown. X red (anterior), Y green (superior), Z blue (right).")
        else:
            self.hint.setText("1 Add bones from their surfaces and apply their axes · 2 Ctrl+click two bones to add a "
                              "joint (optional) · 3 set a reference pose (optional).")

    # ---- the mouse: the left button does the active tool; navigation stays on the other buttons
    def _shown_segments(self) -> list:
        segs, _ = self._shown()
        return [s for s in segs if s.surface is not None]

    def pick_at(self, x: int, y: int, segment: J.Segment | None = None):
        """Surface point under display position (x, y) in a bone's model space: (segment, point) or None.
        With `segment`, only that bone is picked."""
        cands = [segment] if segment is not None else self._shown_segments()
        actors = [(s, self._bone_actors.get(id(s))) for s in cands]
        actors = [(s, a) for s, a in actors if a is not None]
        if not actors:
            return None
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        for _, a in actors:
            picker.AddPickList(a)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        hit = picker.GetActor()
        s, a = next(((s, a) for s, a in actors if a is hit), actors[0])
        world = np.array(picker.GetPickPosition())
        M = np.array(a.user_matrix)
        return s, (np.linalg.inv(M) @ np.r_[world, 1.0])[:3]

    def apply_pick(self, point: np.ndarray, finish: bool = True, erase: bool = False) -> None:
        """Place (point), add to (paint) or remove from (erase) the selected landmark at a surface point."""
        s, key, _method = self._tool_target()
        d = next(x for x in s.stype.landmarks if x.key == key)
        lm = s.landmarks.setdefault(key, {"method": d.methods[0], "points": []})
        mesh = s.mesh
        tree = self._tree(mesh)
        if lm["method"] == "point":
            _, i = tree.query(point)
            lm["points"] = [np.asarray(mesh.points[i], float).tolist()]
        else:
            idx = tree.query_ball_point(point, self.brush)
            done = self._painted.setdefault((id(s), key), set())
            if not done and lm.get("points"):
                # continuing a region loaded from file: recover its vertex indices
                _, prev = tree.query(np.asarray(lm["points"]))
                done.update(int(i) for i in np.atleast_1d(prev))
            if erase:
                done.difference_update(int(i) for i in idx)
            else:
                done.update(int(i) for i in idx)
            lm["points"] = np.asarray(mesh.points)[sorted(done)].tolist()
            if not lm["points"]:
                s.landmarks.pop(key, None)
        self._set_dirty(True)
        if finish:
            self._changed()
        else:
            self.model.compute()
            self.draw()

    def _tree(self, mesh) -> cKDTree:
        if id(mesh) not in self._trees:
            self._trees[id(mesh)] = cKDTree(np.asarray(mesh.points))
        return self._trees[id(mesh)]

    def _event_xy(self, obj):
        return obj.GetInteractor().GetEventPosition()

    def _left_down(self, obj, _event) -> None:
        if self.tool == "navigate":
            obj.OnLeftButtonDown()
            return
        x, y = self._event_xy(obj)
        if self.tool == "pick":
            self.inspect_at(x, y)
            return
        tgt = self._tool_target()
        if tgt is None:
            return
        hit = self.pick_at(x, y, tgt[0])
        if hit is None:
            return
        self._push_undo(tgt[0])
        if self.tool == "point":
            self.apply_pick(hit[1], finish=True)
        else:
            self._painting = True
            self.apply_pick(hit[1], finish=False, erase=self.tool == "erase")

    def _left_up(self, obj, _event) -> None:
        if self.tool == "navigate":
            obj.OnLeftButtonUp()
            return
        if self._painting:
            self._painting = False
            self._changed()

    def _right_down(self, obj, _event) -> None:
        if self.tool == "navigate":
            obj.OnRightButtonDown()                 # zoom, as in the Visualiser
        else:
            self._rotating_right = True             # with a tool active, the right button rotates
            obj.OnLeftButtonDown()

    def _right_up(self, obj, _event) -> None:
        if self._rotating_right:
            self._rotating_right = False
            obj.OnLeftButtonUp()
        else:
            obj.OnRightButtonUp()

    def _mouse_move(self, obj, _event) -> None:
        if self.tool in ("paint", "erase"):
            x, y = self._event_xy(obj)
            tgt = self._tool_target()
            hit = self.pick_at(x, y, tgt[0]) if tgt is not None and not self._rotating_right else None
            if self._painting and hit is not None:
                self.apply_pick(hit[1], finish=False, erase=self.tool == "erase")
            self._show_cursor(hit)
            if self._painting:
                return
        obj.OnMouseMove()

    def _show_cursor(self, hit) -> None:
        """A ring the size of the brush on the surface under the mouse."""
        self._hide_cursor(render=False)
        if hit is None:
            self.plotter.render()
            return
        s, pt = hit
        act = self._bone_actors.get(id(s))
        M = np.array(act.user_matrix) if act is not None else np.eye(4)
        world = (M @ np.r_[pt, 1.0])[:3]
        cam = self.plotter.camera
        normal = np.array(cam.position) - np.array(cam.focal_point)
        ring = pv.Polygon(center=world, radius=self.brush, normal=normal, n_sides=48, fill=False)
        self._cursor_actor = self.plotter.add_mesh(ring, color="#F0A04B" if self.tool == "paint" else "#E5534B",
                                                   line_width=2, pickable=False, reset_camera=False,
                                                   style="wireframe")
        self.plotter.render()

    def _hide_cursor(self, render: bool = True) -> None:
        if self._cursor_actor is not None:
            self.plotter.remove_actor(self._cursor_actor, render=render)
            self._cursor_actor = None

    def inspect_at(self, x: int, y: int):
        """Show the coordinates of the surface point under (x, y) and the nearest landmark."""
        hit = self.pick_at(x, y)
        if hit is None:
            self.hint.setText("Inspect: no surface there.")
            return None
        s, pt = hit
        text = f"{s.name}: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}) mm in its model space"
        best = None
        for d in s.stype.landmarks:
            v = self.model.landmark(s.name, d.key)
            if v is not None:
                dist = float(np.linalg.norm(v - pt))
                if best is None or dist < best[0]:
                    best = (dist, d.name)
        if best is not None:
            text += f" · nearest landmark: {best[1]}, {best[0]:.1f} mm away"
        if s.acs is not None:
            local = np.linalg.solve(s.acs, np.r_[pt, 1.0])[:3]
            names = "XYZ" if s.stype.anatomical and s.axes_source != "inertial" else "123"
            text += " · in its axes: " + ", ".join(f"{n} {v:+.1f}" for n, v in zip(names, local))
        self.hint.setText(text)
        return pt

    # ================================================================ tab interface
    def close_tab(self) -> None:
        self.plotter.close()
