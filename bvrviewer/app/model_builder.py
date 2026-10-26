"""Model Builder tab: apply a joint template to bone surfaces by picking or painting landmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton, QSplitter,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ..core import joints as J
from ..core.models import BoneModel, ModelError, is_volume_file
from ..core.tracking import TrackingError, load_tracking
from .dialogs import VoxelDialog
from .navigator import PropertySheet
from .panels import h2, label
from .scene import BACKGROUNDS

MODEL_FILTER = "Bone models (*.stl *.STL *.tif *.tiff *.TIF *.obj *.ply *.vtk *.vtp);;All files (*)"
JOINT_MODEL_FILTER = "Joint model (*.bvrmodel.json *.json)"
AXIS_COLOURS = ("#E5534B", "#57AB5A", "#539BF5")        # X red, Y green, Z blue
KIND, ROLE, KEY = Qt.UserRole, Qt.UserRole + 1, Qt.UserRole + 2


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


class ModelBuilderTab(QWidget):
    kind = "model_builder"
    titleChanged = Signal(str)

    def __init__(self, state=None):
        super().__init__()
        self.state = state or _State()
        self.model = J.JointModel.new(J.KNEE.key, "right")
        self.sel = ("model", None, None)          # (kind, role, landmark key)
        self.brush = 4.0                          # mm
        self._painting = False
        self._painted: dict = {}                  # (role, key) -> set of vertex indices (this session)
        self._trees: dict = {}                    # role -> cKDTree of mesh points
        self._actors: list = []
        self._dirty = False

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
        left.setSizes([330, 520])
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
        self._style = style
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

    def _item(self, text, kind, role=None, key=None, bold=False, color=None, tip=None):
        it = QTreeWidgetItem([text])
        it.setData(0, KIND, kind)
        it.setData(0, ROLE, role)
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

    def _rebuild_tree(self) -> None:
        t = self.model.template
        self.tree.blockSignals(True)
        self.tree.clear()
        top = self._item(self.model.name, "model", bold=True, tip=t.title)
        self.tree.addTopLevelItem(top)
        bones = self._item("Bones", "bones", bold=True)
        self.tree.addTopLevelItem(bones)
        current = None
        for role in t.bones:
            e = self.model.bones[role.key]
            ok = e.acs is not None
            surf = e.surface.path.name if e.surface is not None else "no surface (right-click to load)"
            b = self._item(f"{role.title}: {surf}", "bone", role.key,
                           color="#E6E8EC" if e.surface is not None else "#F2C48D",
                           tip="Axes complete" if ok else (e.error or ""))
            b.setText(0, ("✓ " if ok else "○ ") + b.text(0))
            bones.addChild(b)
            if self.sel[:2] == ("bone", role.key):
                current = b
            for d in role.landmarks:
                done = self.model.landmark(role.key, d.key) is not None
                lm = self._item(("✓ " if done else "○ ") + d.name, "landmark", role.key, d.key,
                                color="#8FD3A9" if done else "#C3C8D2", tip=d.hint)
                b.addChild(lm)
                if self.sel == ("landmark", role.key, d.key):
                    current = lm
            if not role.landmarks:
                b.addChild(self._item("Automatic axes (no landmarks needed)", "info", role.key, color="#7A8291"))
            b.setExpanded(True)
        joints = self._item("Joints", "joints", bold=True)
        self.tree.addTopLevelItem(joints)
        for j in t.joints:
            ready = all(self.model.bones[r].acs is not None for r in (j.proximal, j.distal))
            ji = self._item(("✓ " if ready else "○ ") + j.title, "joint", key=j.key,
                            color="#E6E8EC" if ready else "#A3AAB8")
            joints.addChild(ji)
            if self.sel[0] == "joint" and self.sel[2] == j.key:
                current = ji
        bones.setExpanded(True)
        joints.setExpanded(True)
        if current is None and self.sel[0] == "model":
            current = top
        if current is not None:
            self.tree.setCurrentItem(current)
        self.tree.blockSignals(False)

    def _tree_selected(self, cur, _prev) -> None:
        if cur is None:
            return
        kind = cur.data(0, KIND)
        if kind in ("bones", "joints", "info"):
            kind = "model" if kind != "info" else "bone"
        self.sel = (kind, cur.data(0, ROLE), cur.data(0, KEY))
        self.show_properties()
        self.draw()

    def _context(self, pos) -> None:
        it = self.tree.itemAt(pos)
        kind = it.data(0, KIND) if it else "model"
        role, key = (it.data(0, ROLE), it.data(0, KEY)) if it else (None, None)
        if it is not None:
            self.tree.setCurrentItem(it)
        menu = self.build_menu(kind, role, key)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def build_menu(self, kind, role=None, key=None) -> QMenu:
        menu = QMenu(self)

        def add(text, fn, enabled=True):
            a = QAction(text, menu)
            a.triggered.connect(lambda _=False: fn())
            a.setEnabled(enabled)
            menu.addAction(a)

        if kind in ("model", "bones", "joints"):
            add("Save model…", self.save_model)
            add("Open model…", self.open_model)
            menu.addSeparator()
            add("Change template…", self.change_template_menu)
        elif kind == "bone" or kind == "info":
            e = self.model.bones[role]
            add("Replace surface…" if e.surface else "Load surface…", lambda: self.load_surface(role))
            add("Preview tracking (.tra)…", lambda: self.attach_preview(role), enabled=e.surface is not None)
            add("Clear all landmarks", lambda: self.clear_landmarks(role), enabled=bool(e.landmarks))
        elif kind == "landmark":
            add("Clear this landmark", lambda: self.clear_landmark(role, key),
                enabled=self.model.landmark(role, key) is not None)
        return menu

    # ================================================================ properties
    def show_properties(self) -> None:
        kind, role, key = self.sel
        p = self.props
        if kind == "bone" and role:
            self._props_bone(role)
        elif kind == "landmark" and role:
            self._props_landmark(role, key)
        elif kind == "joint":
            self._props_joint(key)
        else:
            self._props_model()
        p.finish()
        self._update_hint()

    def _props_model(self) -> None:
        p = self.props
        t = self.model.template
        p.clear(self.model.name)
        p.section("Model")
        p.line("Name", self.model.name, self._rename)
        p.combo("Template", [(k, v.title) for k, v in J.TEMPLATES.items()], t.key, self.change_template)
        if t.sided:
            p.combo("Side", [("right", "Right"), ("left", "Left")], self.model.side, self._set_side,
                    tip="Sets which way is 'right' (ISB Z) and the signs of left-side angles.")
        p.note(t.summary)
        p.section("References")
        for r in t.references:
            p.note(r)
        p.section("File")
        p.value("Saved as", self.model.path.name if self.model.path else "Not saved yet",
                tip=str(self.model.path) if self.model.path else None)
        p.buttons([("Save model…", self.save_model), ("Open model…", self.open_model)])

    def _props_bone(self, role: str) -> None:
        p = self.props
        r = self.model.template.role(role)
        e = self.model.bones[role]
        p.clear(r.title)
        p.section("Surface")
        if e.surface is None:
            p.note("Load this bone's surface (STL, or a TIF partial volume) — the same file and model space "
                   "you track with.", "Warn")
            p.buttons([("Load surface…", lambda: self.load_surface(role))])
        else:
            p.value("File", e.surface.path.name, tip=str(e.surface.path))
            if e.surface.surface_note:
                p.value("Space", e.surface.surface_note)
            p.buttons([("Replace…", lambda: self.load_surface(role))])
        p.section("Axes")
        p.note(r.axes_note or J.RECIPES[r.recipe].__doc__.strip().split("\n")[0])
        if e.acs is not None:
            names = ("X", "Y", "Z") if r.recipe != "inertial" else ("1", "2", "3")
            for k, n in enumerate(names):
                p.value(f"Axis {n}", "  ".join(f"{v:+.3f}" for v in e.acs[:3, k]))
            p.value("Origin", "  ".join(f"{v:.1f}" for v in e.acs[:3, 3]) + " mm")
        else:
            p.value("Status", e.error or "Not computed", kind="Warn")
        if r.landmarks:
            p.section("Landmarks")
            for d in r.landmarks:
                v = self.model.landmark(role, d.key)
                p.value(d.name, "set" if v is not None else "missing", kind="Value" if v is not None else "Warn")
        p.section("Preview tracking (optional)")
        p.value("Tracking", Path(e.preview_tracking).name if e.preview_tracking else "None",
                tip="Lets the Joints view show both bones in a tracked pose, with their axes.")
        p.buttons([("Attach .tra…", lambda: self.attach_preview(role))])

    def _props_landmark(self, role: str, key: str) -> None:
        p = self.props
        r = self.model.template.role(role)
        d = next(x for x in r.landmarks if x.key == key)
        lm = self.model.bones[role].landmarks.get(key) or {"method": d.methods[0], "points": []}
        p.clear(d.name)
        p.note(d.hint)
        p.section("How to place it")
        p.combo("Method", [(m, J.LANDMARK_METHODS[m]) for m in d.methods], lm["method"],
                lambda m: self._set_method(role, key, m))
        if lm["method"] != "point":
            p.slider("Brush size", self.brush, 1, 15, 1, lambda x: f"{x:.0f} mm", self._set_brush)
        p.section("Value")
        v = self.model.landmark(role, key)
        n = len(lm.get("points") or [])
        if v is None:
            p.value("Status", "Not placed yet", kind="Warn")
        else:
            p.value("Position", "  ".join(f"{x:.1f}" for x in v) + " mm")
            if lm["method"] != "point":
                p.value("Painted", f"{n} surface points")
            if lm["method"] == "sphere" and n >= 4:
                p.value("Sphere radius", f"{J.sphere_fit(np.asarray(lm['points']))[1]:.1f} mm")
        p.buttons([("Clear", lambda: self.clear_landmark(role, key))])

    def _props_joint(self, key: str) -> None:
        p = self.props
        j = next(x for x in self.model.template.joints if x.key == key)
        conv = J.JOINT_CONVENTIONS[j.convention]
        p.clear(j.title)
        p.value("Proximal", self.model.template.role(j.proximal).title)
        p.value("Distal", self.model.template.role(j.distal).title)
        p.section("Convention")
        p.note(conv["title"])
        for n in conv["angles"]:
            p.note("• " + n)
        p.section("Preview")
        k = self._preview_kinematics(j)
        if k is None:
            p.note("Attach a preview tracking file to both bones to see the joint in a tracked pose.")
        else:
            f, ang, tr = k
            p.value("Frame", str(f + 1))
            for name, v in zip(conv["short"], ang):
                p.value(name, f"{(0.0 if abs(v) < 0.05 else v):+.1f}°")
            for name, v in zip(conv["short_t"], tr):
                p.value(name, f"{(0.0 if abs(v) < 0.05 else v):+.1f} mm")
        warns = self.model.checks()
        if warns:
            p.section("Checks")
            for w in warns:
                p.note(w, "Warn")
        p.note("Save the model, then load it into a Visualiser tab (right-click Joints) for joint angles "
               "throughout a trial.")
        p.buttons([("Save model…", self.save_model)])

    # ---- setters
    def _rename(self, name: str) -> None:
        self.model.name = name
        self.state.name = name
        self.titleChanged.emit(name)
        self._rebuild_tree()

    def _set_side(self, side: str) -> None:
        self.model.side = side
        self._changed()

    def _set_brush(self, mm: float) -> None:
        self.brush = float(mm)

    def _set_method(self, role, key, method) -> None:
        lm = self.model.bones[role].landmarks.setdefault(key, {"method": method, "points": []})
        lm["method"] = method
        self._changed()

    def change_template(self, key: str) -> None:
        if key == self.model.template_key:
            return
        has = any(e.landmarks for e in self.model.bones.values())
        if has and QMessageBox.question(self, "Change template",
                                        "Changing the template clears the landmarks. Continue?") != QMessageBox.Yes:
            self.show_properties()
            return
        surfaces = [e.surface for e in self.model.bones.values()]
        self.model = J.JointModel.new(key, self.model.side)
        for e, s in zip(self.model.bones.values(), surfaces):   # keep loaded surfaces in order
            e.surface = s
        self._painted.clear()
        self._trees.clear()
        self.sel = ("model", None, None)
        self._changed()

    def change_template_menu(self) -> None:
        self.sel = ("model", None, None)
        self.refresh()

    def _changed(self, rebuild_view: bool = True) -> None:
        self.model.compute()
        self._dirty = True
        self.refresh(rebuild_view)

    # ================================================================ files
    def load_surface(self, role: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Bone surface", "", MODEL_FILTER)
        if path:
            self.set_surface(role, path)

    def set_surface(self, role: str, path, voxel_size=None) -> bool:
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
        e = self.model.bones[role]
        e.surface = bm
        self._trees.pop(role, None)
        self.sel = ("bone", role, None)
        self._changed()
        self.plotter.reset_camera()
        return True

    def attach_preview(self, role: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Tracking file for the preview", "",
                                              "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
        if path:
            self.set_preview(role, path)

    def set_preview(self, role: str, path) -> None:
        try:
            load_tracking(path)
        except (TrackingError, OSError) as err:
            QMessageBox.warning(self, "Couldn't read the tracking file", str(err))
            return
        self.model.bones[role].preview_tracking = str(path)
        self._changed()

    def save_model(self) -> None:
        default = str(self.model.path or f"{self.model.name}.bvrmodel.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save joint model", default, JOINT_MODEL_FILTER)
        if path:
            if not path.endswith(".json"):
                path += ".bvrmodel.json"
            self.model.save(path)
            self._dirty = False
            self.show_properties()

    def open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open joint model", "", JOINT_MODEL_FILTER)
        if path:
            self.load_model_file(path)

    def load_model_file(self, path) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = J.JointModel.load(path)
        except Exception as err:  # noqa: BLE001 - shown to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't open the model", str(err))
            return
        QApplication.restoreOverrideCursor()
        self.model = m
        self.state.name = m.name
        self.titleChanged.emit(m.name)
        self._painted.clear()
        self._trees.clear()
        self.sel = ("model", None, None)
        self._dirty = False
        self.refresh()
        self.plotter.reset_camera()

    def clear_landmark(self, role, key) -> None:
        self.model.bones[role].landmarks.pop(key, None)
        self._painted.pop((role, key), None)
        self._changed()

    def clear_landmarks(self, role) -> None:
        self.model.bones[role].landmarks.clear()
        for k in [k for k in self._painted if k[0] == role]:
            self._painted.pop(k)
        self._changed()

    # ================================================================ 3D view
    def _shown_roles(self) -> list[str]:
        kind, role, _ = self.sel
        if kind in ("bone", "landmark") and role:
            return [role]
        return [r.key for r in self.model.template.bones if self.model.bones[r.key].surface is not None]

    def _preview_poses(self, roles):
        """Tracking poses at the first frame every listed bone is tracked, or None."""
        tr = []
        for r in roles:
            p = self.model.bones[r].preview_tracking
            if not p:
                return None
            try:
                tr.append(load_tracking(p).poses)
            except (TrackingError, OSError):
                return None
        n = min(len(t) for t in tr)
        for f in range(n):
            if all(not np.isnan(t[f]).any() for t in tr):
                return f, [t[f] for t in tr]
        return None

    def _preview_kinematics(self, j):
        if any(self.model.bones[r].acs is None for r in (j.proximal, j.distal)):
            return None
        pp = self._preview_poses([j.proximal, j.distal])
        if pp is None:
            return None
        f, (P, D) = pp
        k = self.model.joint_kinematics(j.key, P[None], D[None])
        return f, k["angles"][0], k["translations"][0]

    def draw(self) -> None:
        pl = self.plotter
        for a in self._actors:
            pl.remove_actor(a, render=False)
        self._actors = []
        self._bone_actors = {}
        roles = self._shown_roles()
        many = len(roles) > 1
        poses = self._preview_poses(roles) if many else None
        offset = 0.0
        for i, role in enumerate(roles):
            e = self.model.bones[role]
            if e.surface is None:
                continue
            mesh = e.surface.mesh
            if poses is not None:
                T = poses[1][i]
            else:
                T = np.eye(4)
                if many:                               # no preview pose: lay the bones side by side
                    b = np.array(mesh.bounds)
                    T[0, 3] = offset - b[0]
                    offset += (b[1] - b[0]) + 25
            colour = "#E8D9B5" if i == 0 else "#C9B98F"
            act = pl.add_mesh(mesh, color=colour, smooth_shading=True, specular=0.2, pickable=True,
                              reset_camera=False, opacity=1.0 if self.sel[0] == "landmark" else 0.55)
            act.user_matrix = T
            self._actors.append(act)
            self._bone_actors[role] = act
            size = float(np.linalg.norm(np.ptp(np.asarray(mesh.points), axis=0)))
            if e.acs is not None:
                self._actors += add_triad(pl, e.acs, max(size * 0.28, 15), pose=T)
            self._draw_landmarks(role, T, size)
        if self.sel[0] in ("bone", "landmark"):
            self.showing.setText(f"Showing: {self.model.template.role(self.sel[1]).title} (its own model space)")
        elif poses is not None:
            self.showing.setText(f"Showing: all bones at preview frame {poses[0] + 1}")
        elif many:
            self.showing.setText("Showing: all bones side by side (attach preview tracking to see them in place)")
        else:
            self.showing.setText("")
        shown = tuple(roles) + ((poses is not None),)
        if self._actors and shown != getattr(self, "_last_shown", None):   # new set of bones: fit them
            pl.reset_camera()
            self._last_shown = shown
        pl.render()

    def _draw_landmarks(self, role, T, size) -> None:
        pl = self.plotter
        r = self.model.template.role(role)
        active = self.sel[0] == "landmark" and self.sel[1] == role
        rad = max(size * 0.012, 1.2)
        for d in r.landmarks:
            lm = self.model.bones[role].landmarks.get(d.key)
            if not lm:
                continue
            is_active = active and self.sel[2] == d.key
            pts = np.asarray(lm.get("points") or [], float).reshape(-1, 3)
            if lm["method"] != "point" and len(pts) and (is_active or not active):
                cloud = pv.PolyData(pts)
                a = pl.add_mesh(cloud, color="#F0A04B" if is_active else "#8A92A1", point_size=6,
                                render_points_as_spheres=True, pickable=False, reset_camera=False)
                a.user_matrix = T
                self._actors.append(a)
            v = self.model.landmark(role, d.key)
            if v is None:
                continue
            s = pl.add_mesh(pv.Sphere(radius=rad * (1.6 if is_active else 1.0), center=v),
                            color="#F0A04B" if is_active else "#4F9CF9", pickable=False, reset_camera=False)
            s.user_matrix = T
            self._actors.append(s)
            if T is not None and np.allclose(T, np.eye(4)):
                lab = pl.add_point_labels(pv.PolyData(v[None]), [d.key], font_size=11, text_color="#E6E8EC",
                                          shape_color="#1C1F26", shape_opacity=0.8, show_points=False,
                                          always_visible=True, pickable=False, reset_camera=False)
                self._actors.append(lab)

    def _update_hint(self) -> None:
        kind, role, key = self.sel
        if kind == "landmark":
            e = self.model.bones[role]
            d = next(x for x in self.model.template.role(role).landmarks if x.key == key)
            if e.surface is None:
                self.hint.setText(f"Load the {self.model.template.role(role).title.lower()} surface first.")
                return
            method = (e.landmarks.get(key) or {"method": d.methods[0]})["method"]
            if method == "point":
                self.hint.setText(f"Right-click the surface to place “{d.name}”. Right-click again to move it. "
                                  "Left-drag rotates, scroll zooms.")
            else:
                self.hint.setText(f"Right-click and drag over the surface to paint “{d.name}”. Painting adds "
                                  f"to the region; Clear starts again. Brush: {self.brush:.0f} mm.")
        elif kind == "bone":
            self.hint.setText("Select a landmark under this bone in the navigator, then right-click the surface "
                              "to place it.")
        elif kind == "joint":
            self.hint.setText("Both bones' axes are shown. X red (anterior), Y green (superior), Z blue (right).")
        else:
            self.hint.setText("Right-click a bone in the navigator to load its surface, then place each landmark.")

    # ---- picking and painting (right mouse button; left button still rotates)
    def _armed(self):
        kind, role, key = self.sel
        if kind != "landmark" or self.model.bones[role].surface is None:
            return None
        return role, key

    def pick_at(self, x: int, y: int) -> np.ndarray | None:
        """Surface point under display position (x, y) on the shown bone, in its model space."""
        armed = self._armed()
        if armed is None:
            return None
        actor = self._bone_actors.get(armed[0])
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
        role, key = self._armed()
        r = self.model.template.role(role)
        d = next(x for x in r.landmarks if x.key == key)
        lm = self.model.bones[role].landmarks.setdefault(key, {"method": d.methods[0], "points": []})
        mesh = self.model.bones[role].surface.mesh
        if lm["method"] == "point":
            tree = self._tree(role)
            _, i = tree.query(point)
            lm["points"] = [np.asarray(mesh.points[i], float).tolist()]
        else:
            tree = self._tree(role)
            idx = tree.query_ball_point(point, self.brush)
            done = self._painted.setdefault((role, key), set())
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

    def _tree(self, role) -> cKDTree:
        if role not in self._trees:
            self._trees[role] = cKDTree(np.asarray(self.model.bones[role].surface.mesh.points))
        return self._trees[role]

    def _right_down(self, obj, _event) -> None:
        if self._armed() is None:
            return
        x, y = obj.GetInteractor().GetEventPosition()
        p = self.pick_at(x, y)
        if p is None:
            return
        method = (self.model.bones[self.sel[1]].landmarks.get(self.sel[2]) or {}).get("method")
        if method is None:
            d = next(d for d in self.model.template.role(self.sel[1]).landmarks if d.key == self.sel[2])
            method = d.methods[0]
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
