"""Model Builder tab: segments (axes on bone surfaces) -> joints (optional) -> reference poses (optional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QRadioButton, QSpinBox, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
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
SHORT_SOURCE = {"landmarks": "Landmarks", "femur_cylinder": "Cylinder", "femur_spheres": "Spheres",
                "tibia_ellipse": "Ellipse", "tibia_centroids": "Centroids", "tibia_inertia": "Plateau inertia",
                "inertial": "Inertial"}


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
        for k, t in J.JOINT_TYPES.items():
            if model.candidates(k, "proximal") and model.candidates(k, "distal"):
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
            for s in self.model.candidates(k, end):
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
        self.model = J.KinematicModel(name="Model")
        self.sel = ("model", None, None)          # (kind, Segment/Joint or None, landmark key)
        self.brush = 4.0                          # mm
        self._painting = False
        self._painted: dict = {}                  # (id(segment), key) -> set of vertex indices (this session)
        self._trees: dict = {}                    # id(mesh) -> cKDTree of its points
        self._actors: list = []
        self._bone_actors: dict = {}
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
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context)
        self.tree.currentItemChanged.connect(self._tree_selected)
        nb.addWidget(self.tree, 1)
        left.addWidget(nav_box)
        self.props = PropertySheet()
        left.addWidget(self.props)
        left.setSizes([340, 520])
        left.setMinimumWidth(340)
        split.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("Panel")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 6, 10, 6)
        self.showing = label("", wrap=False)
        bl.addWidget(self.showing, 1)
        fit = QPushButton("Fit view")
        fit.setObjectName("Small")
        fit.clicked.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        bl.addWidget(fit)
        cv.addWidget(bar)
        self.plotter = QtInteractor(center)
        self.plotter.set_background(BACKGROUNDS["dark"])
        self.plotter.add_axes(color="#C3C8D2", line_width=2)
        cv.addWidget(self.plotter.interactor, 1)
        self.hint = QLabel("")
        self.hint.setObjectName("HintBar")
        self.hint.setWordWrap(True)
        self.hint.setContentsMargins(12, 8, 12, 8)
        cv.addWidget(self.hint)
        split.addWidget(center)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1120])
        root.addWidget(split)

        style = self.plotter.iren.interactor.GetInteractorStyle()
        style.AddObserver("RightButtonPressEvent", self._right_down)
        style.AddObserver("RightButtonReleaseEvent", self._right_up)
        style.AddObserver("MouseMoveEvent", self._mouse_move)
        self.refresh(rebuild_view=True)

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

    def _segment_text(self, s: J.Segment) -> str:
        side = f" ({s.side})" if s.side in ("right", "left") else ""
        surf = Path(s.surface.path).name if s.surface is not None else "no surface"
        return ("✓ " if s.acs is not None else "○ ") + f"{s.name}{side} · {surf}"

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
        segs = self._item("1  Bones", "segments", bold=True, tip="Right-click to add a bone")
        self.tree.addTopLevelItem(segs)
        if not m.segments:
            self._placeholder(segs, "Right-click to add a bone (femur, tibia, other bone)")
        for s in m.segments:
            auto = s.axes_source != "landmarks"
            si = self._item(self._segment_text(s), "segment", s,
                            color="#E6E8EC" if s.surface is not None else "#F2C48D",
                            tip="Axes complete" if s.acs is not None else (s.error or ""))
            segs.addChild(si)
            if self.sel[0] == "segment" and self.sel[1] is s:
                current = si
            if auto:
                si.addChild(self._item("Axes: " + AA.AUTO_METHODS[s.axes_source].title, "axesinfo", s,
                                       color="#B78AF0", tip=AA.AUTO_METHODS[s.axes_source].summary))
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
        joints = self._item("2  Joints (optional)", "joints", bold=True, tip="Right-click to add a joint")
        self.tree.addTopLevelItem(joints)
        if not m.joints:
            self._placeholder(joints, "Right-click to add a joint between two bones")
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
        if current is None:
            current = {"model": top, "segments": segs, "joints": joints}.get(self.sel[0], top)
        self.tree.setCurrentItem(current)
        self.tree.blockSignals(False)

    def _tree_selected(self, cur, _prev) -> None:
        if cur is None or cur.data(0, KIND) is None:
            return
        kind = cur.data(0, KIND)
        if kind == "axesinfo":
            kind = "segment"
        self.sel = (kind, cur.data(0, OBJ), cur.data(0, KEY))
        self.show_properties()
        self.draw()

    def select(self, kind, obj=None, key=None) -> None:
        self.sel = (kind, obj, key)
        self.refresh()

    def _context(self, pos) -> None:
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
            sub = menu.addMenu("Add bone")
            for k, t in J.SEGMENT_TYPES.items():
                add(t.title, lambda k=k: self.add_segment(k), parent=sub)
            for k, p in J.PRESETS.items():
                add("Add " + p["title"], lambda k=k: self.add_preset(k))
        if kind in ("model", "joints"):
            add("Add joint…", self.add_joint, enabled=len(self.model.segments) >= 2)
        if kind == "model":
            menu.addSeparator()
            add("Save model…", self.save_model)
            add("Open model…", self.open_model)
        elif kind in ("segment", "axesinfo"):
            s: J.Segment = obj
            add("Replace surface…" if s.surface else "Load surface…", lambda: self.load_surface(s))
            add("Preview tracking (.tra)…", lambda: self.attach_preview(s), enabled=s.surface is not None)
            menu.addSeparator()
            for src, title in self.model.axes_sources(s.name):
                add("Axes from landmarks" if src == "landmarks" else title,
                    lambda src=src: self.set_axes_source(s, src), checked=s.axes_source == src)
            menu.addSeparator()
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
            self._props_landmark(obj, key)
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
        self._update_hint()

    def _props_model(self) -> None:
        p = self.props
        m = self.model
        p.clear(m.name)
        p.line("Name", m.name, self._rename)
        p.section("Steps")
        p.note("1  Bones: load each bone's surface and define its axes (landmarks, or an automatic method). "
               "On its own this gives each bone's motion about its own axes in the Visualiser.")
        p.note("2  Joints (optional): pair two bones with a published joint definition, e.g. Grood & Suntay "
               "for femur → tibia.")
        p.note("3  Reference pose (optional, per joint): a neutral trial or the scan pose whose joint values are "
               "taken as zero. Without one, zero is anatomical (the bones' axes lined up).")
        p.buttons([("Add bone…", self._add_segment_menu), ("Add joint…", self.add_joint)])
        p.buttons([("Start a knee (femur + tibia + joint)", lambda: self.add_preset("knee"))])
        p.section("Contents")
        p.value("Bones", ", ".join(s.name for s in m.segments) or "none")
        p.value("Joints", ", ".join(j.name for j in m.joints) or "none")
        p.section("File")
        p.value("Saved as", m.path.name if m.path else "Not saved yet", tip=str(m.path) if m.path else None)
        p.buttons([("Save model…", self.save_model), ("Open model…", self.open_model)])

    def _props_segments(self) -> None:
        p = self.props
        p.clear("Bones")
        p.note("Each bone is one surface with its own axes. Choose its type: the type decides which "
               "landmarks and automatic methods it can use.")
        for k, t in J.SEGMENT_TYPES.items():
            p.buttons([(f"Add {t.title.lower()}", lambda k=k: self.add_segment(k))])

    def _props_joints(self) -> None:
        p = self.props
        p.clear("Joints")
        p.note("Joints are optional. Each pairs two existing bones with a definition:")
        for t in J.JOINT_TYPES.values():
            p.note(f"• {t.title}")
        p.buttons([("Add joint…", self.add_joint)])

    def _props_segment(self, s: J.Segment) -> None:
        p = self.props
        t = s.stype
        p.clear(s.name)
        p.line("Name", s.name, lambda n: self._rename_segment(s, n))
        p.value("Type", t.title)
        if t.anatomical:
            p.combo("Side", [("right", "Right"), ("left", "Left")], s.side, lambda v: self._set_side(s, v),
                    tip="Sets which way is right (ISB Z) for landmark axes, and joint angle signs on the left side. "
                        "Automatic axes find right from the bone's shape.")
        p.section("Surface")
        if s.surface is None:
            p.note("Load this bone's surface (STL, or a TIF partial volume): the same file and model space you "
                   "track with.", "Warn")
            p.buttons([("Load surface…", lambda: self.load_surface(s))])
        else:
            p.value("File", Path(s.surface.path).name, tip=str(s.surface.path))
            if s.surface.surface_note:
                p.value("Space", s.surface.surface_note)
            p.buttons([("Replace…", lambda: self.load_surface(s))])
        p.section("Axes")
        sources = self.model.axes_sources(s.name)
        if len(sources) > 1:
            p.combo("Axes from", sources, s.axes_source, lambda src: self.set_axes_source(s, src),
                    tip="An automatic method fitted to the surface, or landmarks you place")
        if s.axes_source == "landmarks":
            p.note(t.axes_note or "")
            for ref in t.references:
                p.note(ref)
        else:
            am = AA.AUTO_METHODS[s.axes_source]
            p.note(am.summary)
            for ref in am.references:
                p.note(ref)
            if s.auto is not None:
                for k, v in s.auto.quality.items():
                    p.value(k, f"{v:.2f}" if isinstance(v, float) else str(v))
                for n in s.auto.notes:
                    p.note(n, "Warn")
            p.check("Show fit", self.show_fit, self._set_show_fit, "Draw the fitted shape and points")
            p.buttons([("Flip anterior/posterior", lambda: self._toggle_option(s, "flip_ap")),
                       ("Flip superior/inferior", lambda: self._toggle_option(s, "flip_long"))])
            flips = [n for k, n in (("flip_ap", "anterior/posterior"), ("flip_long", "superior/inferior"))
                     if s.auto_options.get(k)]
            if flips:
                p.note("Flipped: " + ", ".join(flips))
        if s.acs is not None:
            names = ("X", "Y", "Z") if t.anatomical and s.axes_source != "inertial" else ("1", "2", "3")
            for k, n in enumerate(names):
                p.value(f"Axis {n}", "  ".join(f"{v:+.3f}" for v in s.acs[:3, k]))
            p.value("Origin", "  ".join(f"{v:.1f}" for v in s.acs[:3, 3]) + " mm")
        else:
            p.value("Status", s.error or "Not computed", kind="Warn")
        if t.landmarks and s.axes_source == "landmarks":
            p.section("Landmarks")
            p.note("Select a landmark in the navigator, then right-click the surface to place it.")
            for d in t.landmarks:
                v = self.model.landmark(s.name, d.key)
                p.value(d.name, "set" if v is not None else "missing", kind="Value" if v is not None else "Warn")
        p.section("Preview tracking (optional)")
        p.value("Tracking", Path(s.preview_tracking).name if s.preview_tracking else "None",
                tip="Lets joints be previewed in a tracked pose, and offered as the reference trial.")
        p.buttons([("Attach .tra…", lambda: self.attach_preview(s))])
        p.section("")
        p.buttons([("Remove bone", lambda: self.remove_segment(s))])

    def _props_landmark(self, s: J.Segment, key: str) -> None:
        p = self.props
        d = next(x for x in s.stype.landmarks if x.key == key)
        lm = s.landmarks.get(key) or {"method": d.methods[0], "points": []}
        p.clear(d.name)
        p.note(d.hint)
        if s.axes_source != "landmarks":
            p.note(f"{s.name}'s axes are automatic, so this landmark isn't used (it can still be placed, e.g. to "
                   "compare methods).", "Warn")
        p.section("How to place it")
        p.combo("Method", [(mm, J.LANDMARK_METHODS[mm]) for mm in d.methods], lm["method"],
                lambda mm: self._set_method(s, key, mm))
        if lm["method"] != "point":
            p.slider("Brush size", self.brush, 1, 15, 1, lambda x: f"{x:.0f} mm", self._set_brush)
        p.section("Value")
        v = self.model.landmark(s.name, key)
        n = len(lm.get("points") or [])
        if v is None:
            p.value("Status", "Not placed yet", kind="Warn")
        else:
            p.value("Position", "  ".join(f"{x:.1f}" for x in v) + " mm")
            if lm["method"] != "point":
                p.value("Painted", f"{n} surface points")
            if lm["method"] == "sphere" and n >= 4:
                p.value("Sphere radius", f"{J.sphere_fit(np.asarray(lm['points']))[1]:.1f} mm")
        p.buttons([("Clear", lambda: self.clear_landmark(s, key))])

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
        self._dirty = True
        self.refresh(rebuild_view)

    def _rename(self, name: str) -> None:
        self.model.name = name
        self.state.name = name
        self.titleChanged.emit(name)
        self._rebuild_tree()

    def _add_segment_menu(self) -> None:
        menu = self.build_menu("segments")
        menu.exec(self.cursor().pos())

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

    def _set_method(self, s, key, method) -> None:
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
                QMessageBox.information(self, "Add a joint", "Add at least two bones first.")
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
        self._dirty = True
        self.refresh()
        return True

    def clear_reference(self, j: J.Joint) -> None:
        j.reference = None
        self._dirty = True
        self.refresh()

    def _set_ref_mode(self, j: J.Joint, mode: str) -> None:
        if j.reference is not None:
            j.reference.mode = mode
            self._dirty = True
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
        p = Path(path)
        bm = BoneModel(name=p.stem, path=p)
        if is_volume_file(p):
            if voxel_size is None:
                dlg = VoxelDialog(self, p.name, None)
                if dlg.exec() != QDialog.Accepted:
                    return False
                bm.voxel_size, bm.flip, bm.threshold = dlg.values()
            else:
                bm.voxel_size = tuple(voxel_size)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bm.build()
        except (ModelError, OSError, ValueError) as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't load the surface", str(err))
            return False
        QApplication.restoreOverrideCursor()
        s.surface = bm
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

    def save_model(self) -> None:
        default = str(self.model.path or f"{self.model.name}.bvrmodel.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save model", default, JOINT_MODEL_FILTER)
        if path:
            if not path.endswith(".json"):
                path += ".bvrmodel.json"
            self.model.save(path)
            self._dirty = False
            self.show_properties()

    def open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open model", "", JOINT_MODEL_FILTER)
        if path:
            self.load_model_file(path)

    def load_model_file(self, path) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = J.KinematicModel.load(path)
        except Exception as err:  # noqa: BLE001 - shown to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't open the model", str(err))
            return
        QApplication.restoreOverrideCursor()
        self.model = m
        self.state.name = m.name
        self.titleChanged.emit(m.name)
        self._painted.clear()
        self.sel = ("model", None, None)
        self._dirty = False
        self.refresh()
        self.plotter.reset_camera()

    def clear_landmark(self, s, key) -> None:
        s.landmarks.pop(key, None)
        self._painted.pop((id(s), key), None)
        self._changed()

    def clear_landmarks(self, s) -> None:
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
                              pickable=True, reset_camera=False, opacity=1.0 if self.sel[0] == "landmark" else 0.55)
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
        if kind == "landmark":
            d = next(x for x in obj.stype.landmarks if x.key == key)
            if obj.surface is None:
                self.hint.setText(f"Load {obj.name}'s surface first.")
                return
            method = (obj.landmarks.get(key) or {"method": d.methods[0]})["method"]
            if method == "point":
                self.hint.setText(f"Right-click the surface to place “{d.name}”. Right-click again to move it. "
                                  "Left-drag rotates, scroll zooms.")
            else:
                self.hint.setText(f"Right-click and drag over the surface to paint “{d.name}”. Painting adds "
                                  f"to the region; Clear starts again. Brush: {self.brush:.0f} mm.")
        elif kind == "segment" and obj.axes_source != "landmarks":
            self.hint.setText("Automatic axes: purple = fitted shape, orange = surface points used, teal = shaft "
                              "centres. X red, Y green, Z blue.")
        elif kind == "segment":
            self.hint.setText("Select a landmark under this bone in the navigator, then right-click the surface "
                              "to place it.")
        elif kind == "reference":
            self.hint.setText("Shown in the reference pose when one is set. Its joint values are taken as zero.")
        elif kind == "joint":
            self.hint.setText("Both bones' axes are shown. X red (anterior), Y green (superior), Z blue (right).")
        else:
            self.hint.setText("1 Add bones and define their axes · 2 add joints (optional) · 3 set a reference "
                              "pose (optional). Right-click in the navigator for actions.")

    # ---- picking and painting (right mouse button; left button still rotates)
    def _armed(self):
        kind, obj, key = self.sel
        if kind != "landmark" or obj is None or obj.surface is None:
            return None
        return obj, key

    def pick_at(self, x: int, y: int) -> np.ndarray | None:
        """Surface point under display position (x, y) on the shown bone, in its model space."""
        armed = self._armed()
        if armed is None:
            return None
        actor = self._bone_actors.get(id(armed[0]))
        if actor is None:
            return None
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(actor)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        p = np.array(picker.GetPickPosition())
        M = np.array(actor.user_matrix)
        return (np.linalg.inv(M) @ np.r_[p, 1.0])[:3]

    def apply_pick(self, point: np.ndarray, finish: bool = True) -> None:
        s, key = self._armed()
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
            done.update(int(i) for i in idx)
            lm["points"] = np.asarray(mesh.points)[sorted(done)].tolist()
        self._dirty = True
        if finish:
            self._changed()
        else:
            self.model.compute()
            self.draw()

    def _tree(self, mesh) -> cKDTree:
        if id(mesh) not in self._trees:
            self._trees[id(mesh)] = cKDTree(np.asarray(mesh.points))
        return self._trees[id(mesh)]

    def _right_down(self, obj, _event) -> None:
        armed = self._armed()
        if armed is None:
            return
        x, y = obj.GetInteractor().GetEventPosition()
        p = self.pick_at(x, y)
        if p is None:
            return
        s, key = armed
        method = (s.landmarks.get(key) or {}).get("method")
        if method is None:
            method = next(d for d in s.stype.landmarks if d.key == key).methods[0]
        self._painting = method != "point"
        self.apply_pick(p, finish=not self._painting)

    def _right_up(self, _obj, _event) -> None:
        if self._painting:
            self._painting = False
            self._changed()

    def _mouse_move(self, obj, _event) -> None:
        if self._painting:
            x, y = obj.GetInteractor().GetEventPosition()
            p = self.pick_at(x, y)
            if p is not None:
                self.apply_pick(p, finish=False)
        else:
            obj.OnMouseMove()

    # ================================================================ tab interface
    def close_tab(self) -> None:
        self.plotter.close()
