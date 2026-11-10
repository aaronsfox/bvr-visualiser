"""Assessor Evaluation tab: reliability of raters' tracking, and validity against a criterion.

Left: the set-up (app/assessor_setup.py): a tree of the bones (optional surfaces for surface distance,
optional coordinate systems from a model for bone-axes errors), the reference (criterion) and the
assessors, each with their bones and each bone's tracking file; and the analysis (validity / inter-rater /
intra-rater, measures, thresholds, pooling). Right: the results, recomputed whenever the set-up changes: a summary table,
violin plots of the errors (Welte et al. 2022, figure 4) and the error at every frame (figure 5).

The calculations are in core/assessment.py; see the notes there on what each measure means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import assessment as A
from ..core.config import read_cfg
from ..core.models import BoneModel, is_volume_file
from ..core.tracking import TrackingError, load_tracking
from .assess_plots import apply_style, draw_distributions, draw_over_frames
from .assessor_setup import FactorList, TrackingGrid
from .dialogs import FormatDialog
from .panels import label, small_button

SURFACES = "Surfaces (*.stl *.obj *.ply *.vtk *.vtp *.tif *.tiff);;All files (*)"


class _State:
    def __init__(self):
        self.name = "Assessor"


def _item(text: str, editable: bool = False, tip: str = "") -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
    if editable:
        flags |= Qt.ItemIsEditable
    it.setFlags(flags)
    if tip:
        it.setToolTip(tip)
    return it


def _fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


class FigurePanel(QWidget):
    """A matplotlib figure with its toolbar (zoom, pan, save as PNG/SVG/PDF)."""

    def __init__(self, controls: QWidget | None = None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        bar = QHBoxLayout()
        bar.addWidget(NavigationToolbar2QT(self.canvas, self))
        if controls is not None:
            bar.addWidget(controls)
        bar.addStretch(1)
        v.addLayout(bar)
        v.addWidget(self.canvas, 1)


NEEDS = {"surface": ("Not available yet: load a surface for at least one bone (Bones: Surfaces… or From .cfg…, "
                     "or double-click a bone's Surface)."),
         "axes": "Not available yet: load coordinate systems from a Model Builder model (Bones: Axes…)."}


class AssessorTab(QWidget):
    kind = "assessor"
    titleChanged = Signal(str)

    def __init__(self, state=None):
        super().__init__()
        self.state = state or _State()
        self.project = A.AssessmentProject()
        self.comps: list = []
        self.all_comps: list = []
        self.rows: list = []
        self._format: dict | None = None          # tracking options remembered for this tab (when ticked)
        self._last_dir = ""
        self._trial = "Trial 1"                     # the trial shown in the grid
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.recompute)
        apply_style()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)
        root.addWidget(split)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_results())
        split.setSizes([640, 960])
        self.refresh_tables()
        self.recompute()

    # ================================================================ set-up panel
    def _build_setup(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(600)
        w = QWidget()
        w.setObjectName("Panel")
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        scroll.setWidget(w)

        top = QHBoxLayout()
        top.addWidget(label("Assessor Evaluation", "H2", wrap=False))
        top.addStretch(1)
        for text, fn in [("Open…", self.open_project), ("Save…", self.save_project)]:
            b = small_button(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            top.addWidget(b)
        v.addLayout(top)

        # ---- 1 factors
        v.addWidget(label("1  What was tracked", "H2", wrap=False))
        v.addWidget(label("List the bones, assessors and sessions (and trials, if more than one). Every "
                          "combination becomes a cell to fill with a tracking file below.", "Muted"))
        bh = QHBoxLayout()
        bh.addWidget(label("Bones", "PropLabel", wrap=False))
        bh.addStretch(1)
        for text, fn, tip in [("+", self.add_bone, "Add a bone"),
                              ("Surfaces…", self.add_surfaces,
                               "Bone surfaces (STL, OBJ, PLY, VTK) add the surface distance"),
                              ("From .cfg…", self.add_cfg, "Surfaces from an Autoscoper .cfg (TIF volumes)"),
                              ("Axes…", self.add_axes,
                               "Coordinate systems from a Model Builder model add errors about each bone's axes")]:
            b = small_button(text)
            b.setToolTip(tip)
            if text == "+":
                b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, f=fn: f())
            bh.addWidget(b)
        v.addLayout(bh)
        self.bones_table = QTableWidget(0, 3)
        self.bones_table.setHorizontalHeaderLabels(["Bone", "Surface (optional)", "Axes (optional)"])
        self.bones_table.verticalHeader().setVisible(False)
        self.bones_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bones_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bones_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bones_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.bones_table.setMinimumHeight(90)
        self.bones_table.setMaximumHeight(150)
        self.bones_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bones_table.customContextMenuRequested.connect(self._bones_menu)
        self.bones_table.cellDoubleClicked.connect(self._bones_double_clicked)
        v.addWidget(self.bones_table)

        lists = QHBoxLayout()
        self.lists = {}
        for factor, title, tip in [
                ("reference", "Reference", ("Optional: the criterion (gold-standard) tracking, e.g. from beads, "
                                            "that assessors are compared with for validity. Only one.")),
                ("assessor", "Assessors", "The people who tracked the bones (raters)"),
                ("session", "Sessions", "Repeats by the same assessor, for intra-rater reliability"),
                ("trial", "Trials", "Trials (or participants) each tracked by the assessors")]:
            fl = FactorList(title, tip, max_items=1 if factor == "reference" else None)
            fl.added.connect(lambda f=factor: self.add_factor(f))
            fl.renamed.connect(lambda old, f=factor: self.rename_factor(f, old))
            fl.removed.connect(lambda name, f=factor: self.remove_factor(f, name))
            fl.moved.connect(lambda name, step, f=factor: self.move_factor(f, name, step))
            lists.addWidget(fl)
            self.lists[factor] = fl
        v.addLayout(lists)

        # ---- 2 tracking files
        v.addWidget(label("2  Tracking files", "H2", wrap=False))
        gh = QHBoxLayout()
        self.coverage = label("", "Muted", wrap=False)
        gh.addWidget(self.coverage, 1)
        self.grid_trial = QComboBox()
        self.grid_trial.setToolTip("The trial whose files are shown")
        self.grid_trial.currentIndexChanged.connect(self._grid_trial_changed)
        gh.addWidget(self.grid_trial)
        v.addLayout(gh)
        v.addWidget(label("Double-click a cell, or drop a file on it, to load its tracking file. Cells can stay "
                          "empty: comparisons that need them are N/A.", "Muted"))
        self.grid = TrackingGrid()
        self.grid.cellDoubleClicked.connect(self._cell_double_clicked)
        self.grid.customContextMenuRequested.connect(self._grid_menu)
        self.grid.fileDropped.connect(self._file_dropped)
        v.addWidget(self.grid, 1)
        self.problems = label("", "Warn")
        v.addWidget(self.problems)

        # ---- 3 analysis
        v.addWidget(label("3  Analysis", "H2", wrap=False))
        form = QFormLayout()
        self.mode = QComboBox()
        for k, t in A.MODES:
            self.mode.addItem(t, k)
        self.mode.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Compare", self.mode)
        self.session = QComboBox()
        self.session.setToolTip("Inter-rater: which session of each assessor to compare")
        self.session.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Session", self.session)
        self.measure_box = QWidget()
        mg = QGridLayout(self.measure_box)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setHorizontalSpacing(14)
        mg.setVerticalSpacing(2)
        self.measure_checks = {}
        boxes = [("helical_rotation", "Helical rotation", "Rotation about the helical axis (Welte et al. 2022)"),
                 ("helical_translation", "Helical translation",
                  ("Translation along the helical axis (Welte et al. 2022). Not the bone's displacement: "
                   "translation across the axis isn't counted")),
                 ("surface_mean", "Surface distance (mean)", ("Needs bone surfaces: how far the surface is from "
                                                               "where it should be, on average")),
                 ("surface_max", "Surface distance (max)", "Needs bone surfaces: the largest distance"),
                 ("centroid", "Centroid displacement", "Needs bone surfaces: how far the surface's centroid moved"),
                 ("axes", "Bone axes (6 components)", ("Needs coordinate systems from a model: rotation about Z, X, "
                                                        "Y and translation along X, Y, Z of the bone's own axes"))]
        for i, (key, text, tip) in enumerate(boxes):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.setProperty("tip", tip)
            c.setChecked(key in A.DEFAULT_MEASURES)
            c.toggled.connect(self._settings_changed)
            mg.addWidget(c, i // 2, i % 2)
            self.measure_checks[key] = c
        form.addRow("Measures", self.measure_box)
        self.rot_thr = self._spin(2.0, " °", "Rotation threshold (Welte et al. 2022 used 2°)")
        self.tr_thr = self._spin(1.0, " mm", "Translation and distance threshold (Welte et al. 2022 used 1 mm)")
        thr = QHBoxLayout()
        thr.addWidget(self.rot_thr)
        thr.addWidget(self.tr_thr)
        thr.addStretch(1)
        form.addRow("Thresholds", thr)
        self.pool = QComboBox()
        for k, t in [("none", "Each trial and bone separately"), ("trials", "Pool trials (per bone)"),
                     ("all", "Pool trials and bones")]:
            self.pool.addItem(t, k)
        self.pool.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Statistics", self.pool)
        v.addLayout(form)
        return scroll

    def _spin(self, value: float, suffix: str, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.01, 1000)
        s.setDecimals(2)
        s.setSingleStep(0.5)
        s.setValue(value)
        s.setSuffix(suffix)
        s.setToolTip(tip)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(self._settings_changed)
        return s

    # ================================================================ results panel
    def _build_results(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        head = QHBoxLayout()
        self.headline = label("", "Muted")
        head.addWidget(self.headline, 1)
        self.cap = QCheckBox("Limit plots to 3 × threshold")
        self.cap.setToolTip("Hide very large errors (e.g. failed frames) so the rest can be seen; the number "
                            "hidden is noted on each plot")
        self.cap.toggled.connect(self.redraw)
        head.addWidget(self.cap)
        exp = QPushButton("Export ▾")
        exp.setObjectName("Small")
        menu = QMenu(exp)
        menu.addAction("Summary statistics (CSV)…", self.export_summary)
        menu.addAction("Per-frame values (CSV)…", self.export_frames)
        menu.addAction("Figures (PNG, SVG or PDF)…", self.export_figures)
        exp.setMenu(menu)
        head.addWidget(exp)
        v.addLayout(head)
        self.space_warn = label("", "Warn")
        self.space_warn.setVisible(False)
        v.addWidget(self.space_warn)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

        self.summary = QTableWidget(0, 0)
        self.summary.verticalHeader().setVisible(False)
        self.summary.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary.setSortingEnabled(False)
        self.tabs.addTab(self.summary, "Summary")

        self.dist = FigurePanel()
        self.tabs.addTab(self.dist, "Distributions")

        ctl = QWidget()
        cl = QHBoxLayout(ctl)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(label("Measure", wrap=False))
        self.frame_measure = QComboBox()
        self.frame_measure.currentIndexChanged.connect(self.redraw)
        cl.addWidget(self.frame_measure)
        cl.addWidget(label("Trial", wrap=False))
        self.frame_trial = QComboBox()
        self.frame_trial.currentIndexChanged.connect(self.redraw)
        cl.addWidget(self.frame_trial)
        self.frames_fig = FigurePanel(ctl)
        self.tabs.addTab(self.frames_fig, "Over frames")
        self.tabs.currentChanged.connect(self.redraw)
        return w

    # ================================================================ factors
    def _changed(self) -> None:
        self.refresh_tables()
        self.schedule()

    def add_factor(self, factor: str, name: str | None = None) -> str | None:
        p = self.project
        if factor == "reference":
            return self.add_reference(name)
        if name is None:
            default = {"assessor": p.next_assessor_name(),
                       "session": next(str(i) for i in range(1, 1000) if str(i) not in p.sessions()),
                       "trial": next(f"Trial {i}" for i in range(1, 1000) if f"Trial {i}" not in p.trials()),
                       "bone": ""}[factor]
            name, ok = QInputDialog.getText(self, f"Add {factor}", f"{factor.capitalize()} name:", text=default)
            if not ok:
                return None
        try:
            name = {"bone": p.add_bone, "assessor": p.add_assessor, "session": p.add_session,
                    "trial": p.add_trial}[factor](name)
        except ValueError as e:
            QMessageBox.warning(self, f"Add {factor}", str(e))
            return None
        if factor == "trial":
            self._trial = name
        self._changed()
        return name

    def add_bone(self, name: str | None = None) -> str | None:
        return self.add_factor("bone", name)

    def rename_factor(self, factor: str, old: str, new: str | None = None) -> None:
        if new is None:
            new, ok = QInputDialog.getText(self, f"Rename {factor}", f"New name for {factor} '{old}':", text=old)
            if not ok:
                return
        try:
            if factor == "reference":
                self.project.rename_reference(new)
            else:
                self.project.rename(factor, old, new)
        except ValueError as e:
            QMessageBox.warning(self, f"Rename {factor}", str(e))
            return
        if factor == "trial" and self._trial == old:
            self._trial = new.strip()
        self._changed()

    def remove_factor(self, factor: str, name: str, ask: bool = True) -> None:
        p = self.project
        if factor == "reference":
            self.remove_reference(ask)
            return
        attr = A._FIELDS[factor]
        n = sum(getattr(e, attr) == name and not (e.criterion and factor in ("assessor", "session"))
                for e in p.entries)
        if ask and n and QMessageBox.question(
                self, "Remove", f"Remove {factor} '{name}' and the {n} tracking file{'s' * (n != 1)} in its cells? "
                                "(The files themselves are kept.)") != QMessageBox.Yes:
            return
        p.remove(factor, name)
        self._changed()

    def move_factor(self, factor: str, name: str, step: int) -> None:
        if factor != "reference":
            self.project.move(factor, name, step)
            self._changed()

    def add_reference(self, name: str | None = None) -> str | None:
        """The reference (criterion) tracking: only one."""
        p = self.project
        if p.has_reference:
            QMessageBox.information(self, "Reference", "There can be only one reference. Rename or remove the "
                                                       "one there.")
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Add reference", "Name for the reference (criterion) tracking, "
                                                                   "e.g. Beads:", text="Reference")
            if not ok:
                return None
        name = name.strip() or "Reference"
        if name in p.raters():
            QMessageBox.warning(self, "Add reference", f"There is already an assessor '{name}'.")
            return None
        p.set_reference(True, name)
        self._changed()
        return name

    def remove_reference(self, ask: bool = True) -> None:
        p = self.project
        n = sum(e.criterion for e in p.entries)
        if ask and n and QMessageBox.question(
                self, "Remove", f"Remove the reference '{p.reference_name}' and its {n} tracking "
                                f"file{'s' * (n != 1)}? (The files themselves are kept.)") != QMessageBox.Yes:
            return
        p.set_reference(False)
        self._changed()

    # ================================================================ bones table
    def _bones_menu(self, pos) -> None:
        r = self.bones_table.rowAt(pos.y())
        m = QMenu(self)
        m.addAction("Add bone…", self.add_bone)
        if 0 <= r < len(self.project.bones):
            bone = self.project.bones[r].name
            info = self.project.bones[r]
            m.addAction("Rename…", lambda: self.rename_factor("bone", bone))
            m.addAction("Move up", lambda: self.move_factor("bone", bone, -1))
            m.addAction("Move down", lambda: self.move_factor("bone", bone, 1))
            m.addSeparator()
            m.addAction("Load surface…", lambda: self.add_surfaces(bone=bone))
            act = m.addAction("Remove surface", lambda: self.remove_surface(bone))
            act.setEnabled(info.surface is not None)
            m.addAction("Axes from a model…", self.add_axes)
            act = m.addAction("Remove axes", lambda: self.remove_axes(bone))
            act.setEnabled(bool(info.axes_path))
            m.addSeparator()
            m.addAction("Remove bone", lambda: self.remove_factor("bone", bone))
        m.exec(self.bones_table.viewport().mapToGlobal(pos))

    def _bones_double_clicked(self, r: int, c: int) -> None:
        bone = self.project.bones[r].name
        if c == 0:
            self.rename_factor("bone", bone)
        elif c == 1:
            self.add_surfaces(bone=bone)
        else:
            self.add_axes()

    # ================================================================ the grid
    def _slot(self, r: int, c: int):
        who, ses = self.grid.slots[r]
        return self.grid.trial, self.grid.bones[c], who, ses

    def _grid_trial_changed(self, *_) -> None:
        if self._busy:
            return
        self._trial = self.grid_trial.currentData() or self._trial
        self.refresh_tables()

    def _cell_double_clicked(self, r: int, c: int) -> None:
        trial, bone, who, ses = self._slot(r, c)
        e = self.project.entry(trial, bone, who, ses)
        if e is not None and e.tracking is not None:
            self.edit_format(e)
        else:
            self.load_cell(trial, bone, who, ses)

    def _grid_menu(self, pos) -> None:
        idx = self.grid.indexAt(pos)
        if not idx.isValid():
            return
        cells = {(i.row(), i.column()) for i in self.grid.selectedIndexes()} | {(idx.row(), idx.column())}
        entries = [e for r, c in cells if (e := self.project.entry(*self._slot(r, c))) is not None]
        trial, bone, who, ses = self._slot(idx.row(), idx.column())
        e = self.project.entry(trial, bone, who, ses)
        m = QMenu(self)
        m.addAction("Load new file…" if e else "Load file…", lambda: self.load_cell(trial, bone, who, ses))
        if e is not None:
            m.addAction("Refresh current file", lambda: self.refresh_tracking(e))
            m.addAction("Format and frames…", lambda: self.edit_format(e))
        if entries:
            m.addSeparator()
            m.addAction(f"Clear {len(entries)} cell{'s' * (len(entries) != 1)}", lambda: self.clear_cells(entries))
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def _file_dropped(self, r: int, c: int, path: str) -> None:
        if r >= 0 and c >= 0:
            self.load_cell(*self._slot(r, c), path=path)

    def load_cell(self, trial: str, bone: str, who: str, session: str, path=None, more: int = 0) -> bool:
        """Put a tracking file in one cell (choosing it if not given)."""
        p = self.project
        where = f"{p.row_label(who, session)} · {bone}" + (f" · {trial}" if len(p.trials()) > 1 else "")
        if path is None:
            old = p.entry(trial, bone, who, session)
            start = str(old.path.parent) if old is not None else self._last_dir
            path, _ = QFileDialog.getOpenFileName(self, f"Tracking file: {where}", start,
                                                  "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if not path:
                return False
        path = Path(path)
        self._last_dir = str(path.parent)
        fmt = self._choose_format(path, where, more)
        if fmt is None:
            return False
        p.set_entry(trial, bone, who, session, path, *fmt)
        self._changed()
        return True

    def clear_cells(self, entries) -> None:
        for e in entries:
            if e in self.project.entries:
                self.project.entries.remove(e)
        self._changed()

    def refresh_tracking(self, e: A.TrackingEntry) -> None:
        e.load()
        self._changed()

    def _choose_format(self, path: Path, what: str, more: int = 0):
        """(overrides, volume, offset) for a file: the remembered options if they fit, else ask."""
        try:
            detected = load_tracking(path)
        except (TrackingError, OSError) as err:
            QMessageBox.warning(self, "Couldn't read the tracking file", f"{path}\n\n{err}")
            return None
        rem = self._format
        if rem and detected.fmt.n_volumes == 1 and rem.get("kind") == detected.fmt.kind:
            try:
                if not load_tracking(path, 0, rem).warnings:
                    return dict(rem), 0, 0
            except TrackingError:
                pass
        dlg = FormatDialog(self, detected.fmt, 0, path=path, detected=detected.fmt, offset=0,
                           n_rows=detected.n_rows, bone=what, ask_remember=True)
        dlg.remember.setText("Use these options for the other tracking files in this tab without asking"
                             + (f" (the other {more} being added now, too)" if more else ""))
        dlg.remember.setChecked(more > 0)
        if dlg.exec() != QDialog.Accepted:
            return None
        (overrides, volume), offset = dlg.values(), dlg.offset
        if dlg.remember.isChecked():
            self._format = dict(overrides)
        return overrides, volume, offset

    def edit_format(self, e: A.TrackingEntry) -> None:
        p = self.project
        try:
            detected = load_tracking(e.path)
        except (TrackingError, OSError) as err:
            QMessageBox.warning(self, "Couldn't read the file", str(err))
            return
        name = lambda o: p.row_label(A.CRITERION if o.criterion else o.rater, o.session)
        fmt = e.tracking.fmt if e.tracking is not None else detected.fmt
        others = [(name(o), o.tracking.offset, o.tracking.n_rows) for o in p.entries
                  if o is not e and o.tracking is not None and o.trial == e.trial and o.bone == e.bone]
        dlg = FormatDialog(self, fmt, e.volume, path=e.path, detected=detected.fmt, offset=e.offset,
                           n_rows=detected.n_rows, others=others, bone=f"{name(e)} · {e.bone}")
        if dlg.exec() != QDialog.Accepted:
            return
        (e.overrides, e.volume), e.offset = dlg.values(), dlg.offset
        e.load()
        self._changed()

    # ================================================================ bones: surfaces and axes
    def _match_bone(self, stem: str) -> str | None:
        parts = [x for x in stem.lower().replace("-", "_").split("_") if x]
        for name in sorted(self.project.bone_names(), key=len, reverse=True):
            if name.lower() in parts or stem.lower() == name.lower():
                return name
        return None

    def add_surfaces(self, paths=None, bone: str | None = None) -> None:
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Bone surfaces (in the space the tracking maps)", "",
                                                    SURFACES)
            if not paths:
                return
        for x in map(Path, paths):
            name = bone or self._match_bone(x.stem)
            if name is None:
                name, ok = QInputDialog.getItem(self, "Which bone?", f"Bone label for {x.name}:",
                                                self.project.bone_names() or [x.stem], 0, True)
                if not ok or not name.strip():
                    continue
                name = name.strip()
            if is_volume_file(x):
                QMessageBox.information(self, "TIF volume", "For TIF volumes use 'Bone surfaces from an "
                                                            "Autoscoper .cfg…', which reads their voxel size.")
                continue
            surf = BoneModel(name=name, path=x)
            if not self._build(surf):
                continue
            b = self.project.bone_info_for(name)
            b.surface, b._points = surf, None
        self._changed()

    def add_cfg(self, path=None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Autoscoper trial configuration", "",
                                                  "Autoscoper config (*.cfg);;All files (*)")
            if not path:
                return
        try:
            cfg = read_cfg(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Couldn't read the .cfg", str(e))
            return
        for vol in cfg.volumes:
            name = self._match_bone(vol.path.stem) or vol.path.stem
            surf = BoneModel(name=name, path=vol.path, voxel_size=vol.voxel_size, flip=vol.flip)
            if not vol.voxel_size or not self._build(surf):
                continue
            b = self.project.bone_info_for(name)
            b.surface, b._points = surf, None
        self._changed()

    def _build(self, surf: BoneModel) -> bool:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            surf.build()
            return True
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Couldn't load the surface", f"{surf.path}\n\n{e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def add_axes(self, path=None) -> None:
        """Coordinate systems from a Model Builder model, matched to bones by label or surface file."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Model with the bones' coordinate systems", "",
                                                  "BVR model (*.bvrmodel.json *.json);;All files (*)")
            if not path:
                return
        from ..core.joints import KinematicModel

        try:
            km = KinematicModel.load(path, build_surfaces=False)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Couldn't read the model", str(e))
            return
        matched, missing = [], []
        for name in self.project.bone_names():
            seg = None
            for s in km.segments:
                stem = Path(s.surface.path).stem if s.surface is not None else ""
                if s.acs is not None and (s.name.lower() == name.lower() or self._match_bone(stem) == name):
                    seg = s
                    break
            if seg is None:
                missing.append(name)
                continue
            b = self.project.bone_info_for(name)
            b.axes_path, b.axes_segment = str(path), seg.name
            A.load_bone_axes(b)
            matched.append(f"{name} → {seg.name}")
        self._changed()
        QMessageBox.information(self, "Coordinate systems",
                                ("Matched: " + ", ".join(matched) if matched else "No bones matched.")
                                + (f"\nNo coordinate system for: {', '.join(missing)}" if missing else ""))

    def remove_surface(self, bone: str) -> None:
        b = self.project.bone_info_for(bone)
        b.surface, b._points = None, None
        self._changed()

    def remove_axes(self, bone: str) -> None:
        b = self.project.bone_info_for(bone)
        b.axes_path = b.axes_segment = None
        b.acs = None
        self._changed()

    # ================================================================ refresh
    def refresh_tables(self, *_):
        """Redraw the set-up from the project. The analysis controls ignore changes while this runs, so it
        always lets them go again, even if something fails part-way."""
        self._busy = True
        try:
            self._refresh_tables()
        finally:
            self._busy = False
        self._sync_settings_widgets()

    def _refresh_tables(self) -> None:
        p = self.project
        # factors
        self.bones_table.setRowCount(len(p.bones))
        for r, b in enumerate(p.bones):
            it = QTableWidgetItem(b.name)
            it.setToolTip("Double-click to rename; right-click for more")
            self.bones_table.setItem(r, 0, it)
            s = b.surface
            si = QTableWidgetItem(Path(s.path).name if s is not None else "none")
            si.setToolTip(str(s.path) if s is not None else "Optional: double-click to load a surface")
            si.setForeground(QColor("#8FD3A9" if s is not None else "#5E6573"))
            self.bones_table.setItem(r, 1, si)
            ax = (f"{b.axes_segment} ({Path(b.axes_path).name})" if b.acs is not None
                  else ("not found" if b.axes_path else "none"))
            ai = QTableWidgetItem(ax)
            ai.setToolTip(b.axes_path or "Optional: double-click to take coordinate systems from a model")
            ai.setForeground(QColor("#5CC8A8" if b.acs is not None else ("#F2C48D" if b.axes_path else "#5E6573")))
            self.bones_table.setItem(r, 2, ai)
        self.lists["assessor"].set_items(p.raters())
        self.lists["session"].set_items(p.sessions())
        self.lists["trial"].set_items(p.trials())
        self.lists["reference"].set_items([p.reference_name] if p.has_reference else [])
        # grid
        if self._trial not in p.trials():
            self._trial = p.trials()[0]
        self.grid_trial.clear()
        for t in p.trials():
            self.grid_trial.addItem(t, t)
        self.grid_trial.setCurrentIndex(max(self.grid_trial.findData(self._trial), 0))
        self.grid_trial.setVisible(len(p.trials()) > 1)
        self.grid.fill(p, self._trial)
        have, n = p.filled()
        if not p.bones or not p.rows():
            self.coverage.setText("Add bones and assessors (or the reference) to get cells to fill.")
        else:
            self.coverage.setText(f"{have} of {n} cells filled" + (f" across {len(p.trials())} trials"
                                                                  if len(p.trials()) > 1 else ""))
        probs = p.problems()
        self.problems.setText("\n".join("⚠ " + s for s in probs[:8]) + (f"\n… and {len(probs) - 8} more"
                                                                         if len(probs) > 8 else ""))
        self.problems.setVisible(bool(probs))
        # analysis options that depend on the factors
        avail = set(p.available_measures())
        for key, c in self.measure_checks.items():
            ok = key in avail or (key == "axes" and "axes_rz" in avail)
            c.setEnabled(ok)
            need = "axes" if key == "axes" else A.MEASURES[key].needs
            c.setToolTip(c.property("tip") + ("" if ok else "\n\n" + NEEDS[need]))
        cur = self.session.currentData()
        self.session.blockSignals(True)
        self.session.clear()
        for s in p.sessions():
            self.session.addItem(f"Session {s}", s)
        i = self.session.findData(cur if cur is not None else p.settings.inter_session)
        self.session.setCurrentIndex(max(i, 0))
        self.session.blockSignals(False)

    def _sync_settings_widgets(self) -> None:
        mode = self.mode.currentData()
        multi = len(self.project.sessions()) > 1
        self.session.setEnabled(mode == "inter" and multi)
        why = ("Inter-rater: which session of each assessor to compare." if mode == "inter" and multi else
               "Only used for inter-rater comparisons (Compare: Between assessors)." if mode != "inter" else
               "There is only one session. Add sessions (Sessions: +) to choose which one to compare.")
        self.session.setToolTip(why)

    # ================================================================ settings and results
    def _settings_changed(self, *_):
        if self._busy:
            return
        self._sync_settings_widgets()
        self.schedule()

    def _measure_check(self, key: str) -> QCheckBox:
        return self.measure_checks["axes" if key.startswith("axes_") else key]

    def _measure_on(self, key: str) -> bool:
        return self._measure_check(key).isChecked()

    def read_settings(self) -> A.AnalysisSettings:
        avail = set(self.project.available_measures())
        st = A.AnalysisSettings(
            mode=self.mode.currentData(),
            measures=tuple(k for k in A.MEASURES if k in avail and self._measure_on(k)),
            rotation_threshold=self.rot_thr.value(), translation_threshold=self.tr_thr.value(),
            inter_session=self.session.currentData() or "", pool=self.pool.currentData())
        self.project.settings = st
        return st

    def apply_settings(self, st: A.AnalysisSettings) -> None:
        self._busy = True
        try:
            self.mode.setCurrentIndex(max(self.mode.findData(st.mode), 0))
            for k, c in self.measure_checks.items():
                c.setChecked(k in st.measures or (k == "axes" and any(m.startswith("axes_") for m in st.measures)))
            self.rot_thr.setValue(st.rotation_threshold)
            self.tr_thr.setValue(st.translation_threshold)
            self.pool.setCurrentIndex(max(self.pool.findData(st.pool), 0))
        finally:
            self._busy = False
        self.refresh_tables()
        i = self.session.findData(st.inter_session)
        self.session.setCurrentIndex(max(i, 0))

    def schedule(self) -> None:
        self._timer.start(50)

    def recompute(self) -> None:
        st = self.read_settings()
        self.all_comps = self.project.comparisons(st)          # including N/A ones (missing files)
        self.comps = [c for c in self.all_comps if c.ok]       # with values: plots and exports
        self.rows = A.summarise(self.all_comps, st)
        self._headline(st)
        self._space_check(st)
        self._fill_summary(st)
        # measure / trial choices for the frames plot
        for combo, items in [(self.frame_measure, [(A.MEASURES[k].short, k) for k in st.measures]),
                             (self.frame_trial, [(t, t) for t in self.project.trials()])]:
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for t, k in items:
                combo.addItem(t, k)
            combo.setCurrentIndex(max(combo.findData(cur), 0))
            combo.blockSignals(False)
        self.frame_trial.setVisible(len(self.project.trials()) > 1)
        self.redraw()

    def _headline(self, st: A.AnalysisSettings) -> None:
        p = self.project
        n_all, n_ok = len(self.all_comps), len(self.comps)
        text = (f"{n_ok} of {n_all} comparison{'s' * (n_all != 1)} available"
                + (f" ({n_all - n_ok} N/A: files missing)" if n_ok < n_all else "")) if n_all else ""
        if not n_all:
            text = {"validity": "Validity compares each assessor with the reference: tick Reference and add "
                                "bones and assessors.",
                    "inter": "Inter-rater compares assessors with each other: add bones and two or more assessors.",
                    "intra": "Intra-rater compares each assessor's sessions: add bones, assessors and two or "
                             "more sessions."}[st.mode]
            if st.mode == "validity" and p.has_reference and p.bones and not p.raters():
                text = "Add the assessors to compare with the reference."
        self.headline.setText(text)

    def _space_check(self, st: A.AnalysisSettings) -> None:
        """Very large, steady differences usually mean the two files map different bone spaces."""
        bad = []
        for c in self.comps:
            phi = c.values.get("helical_rotation")
            if phi is not None and len(phi) and (np.median(phi) > 20 or np.median(np.abs(
                    c.values.get("helical_translation", [0]))) > 50):
                bad.append(f"{c.label} ({c.bone})")
        self.space_warn.setText("Very large differences for " + ", ".join(bad[:4]) + (" …" if len(bad) > 4 else "")
                                + ". Check that both trackings map the same bone space (e.g. a criterion in CT "
                                  "space and raters in Autoscoper space), the same rows (frames) and the "
                                  "same format.")
        self.space_warn.setVisible(bool(bad))

    SUMMARY_COLS: tuple = (("n", "Frames", 0), ("tracked", "Tracked", 0), ("mean", "Mean", 2), ("sd", "SD", 2), ("median", "Median", 2),
                    ("iqr", "IQR", 2), ("rmse", "RMS", 2), ("max", "Max", 2), ("loa", "LoA (95%)", 2),
                    ("robust", "Robust LoA", 2), ("mad", "MAD", 2), ("outliers", "Outliers", 0),
                    ("pct_over", "% > thr.", 0))

    def _fill_summary(self, st: A.AnalysisSettings) -> None:
        head = ["Trial", "Bone", {"validity": "Assessor", "inter": "Assessors", "intra": "Sessions"}[st.mode], "Measure"]
        cols = self.SUMMARY_COLS
        self.summary.clear()
        self.summary.setColumnCount(len(head) + len(cols) + 1)
        self.summary.setHorizontalHeaderLabels(head + [c[1] for c in cols] + ["Note"])
        tips = {"Frames": "Frames compared: only frames tracked in both files (rows of NaN are untracked)",
                "Tracked": "Frames tracked in the test file / in the file it's compared with",
                "LoA (95%)": "Bland–Altman limits of agreement: mean ± 1.96 SD (± half-width of their 95% CI)",
                "Robust LoA": "Median ± 1.45 × IQR (Williams et al. 2026); lower limit floored at 0 for magnitudes",
                "MAD": "Median absolute deviation from the median",
                "Outliers": "Hampel identifier: more than 3.5 scaled MADs from the median",
                "% > thr.": "Share of frames above the threshold (rotation or translation)",
                "RMS": "Root mean square", "Max": "Largest absolute value"}
        for i, (_k, t, _d) in enumerate(cols):
            if t in tips:
                self.summary.horizontalHeaderItem(len(head) + i).setToolTip(tips[t])
        self.summary.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            m = A.MEASURES[row.measure]
            s = row.stats
            u = A.unit_symbol(m.unit)
            vals = [row.trial, row.bone, row.label, f"{m.short} ({u})"]
            if row.na:
                vals += ["0"] + ["N/A"] * (len(cols) - 1)
                cols_iter = []
            else:
                cols_iter = cols
            for k, _t, d in cols_iter:
                if k == "iqr":
                    vals.append(_fmt(s.get("q3", np.nan) - s.get("q1", np.nan), d))
                elif k == "loa":
                    vals.append(f"{_fmt(s.get('loa_low'), d)} to {_fmt(s.get('loa_high'), d)} "
                                f"(±{_fmt(s.get('loa_ci'), d)})" if s.get("n", 0) > 1 else "–")
                elif k == "robust":
                    vals.append(f"{_fmt(s.get('robust_low'), d)} to {_fmt(s.get('robust_high'), d)}")
                elif k == "tracked":
                    vals.append(f"{row.frames_test} / {row.frames_ref}")
                elif k == "pct_over":
                    vals.append(f"{s.get('pct_over', 0):.0f}% ({s.get('over', 0)})")
                else:
                    vals.append(_fmt(s.get(k), d) if d else str(s.get(k, "–")))
            for c, text in enumerate(vals):
                it = _item(text)
                if c >= len(head):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if row.na:
                    it.setForeground(QColor("#5E6573"))
                if row.note:
                    it.setToolTip(row.note)
                self.summary.setItem(r, c, it)
            self.summary.setItem(r, len(vals), _item(row.note))
            if row.na:
                self.summary.item(r, len(vals)).setForeground(QColor("#5E6573"))
            if s.get("pct_over", 0) > 0:
                self.summary.item(r, len(head) + len(cols) - 1).setForeground(QColor("#F2C48D"))
        self.summary.resizeColumnsToContents()

    def redraw(self, *_):
        st = self.project.settings
        tab = self.tabs.currentWidget()
        if tab is self.dist:
            n_bones = 1 if st.pool == "all" else len({c.bone for c in self.comps}) or 1
            self.dist.fig.set_size_inches(8, max(3.2, 2.6 * n_bones))
            draw_distributions(self.dist.fig, self.all_comps, st, cap=self.cap.isChecked())
            self.dist.canvas.draw_idle()
        elif tab is self.frames_fig:
            key = self.frame_measure.currentData()
            if key is None:
                self.frames_fig.fig.clear()
            else:
                trial = self.frame_trial.currentData() if len(self.project.trials()) > 1 else None
                draw_over_frames(self.frames_fig.fig, self.comps, st, key, trial, cap=self.cap.isChecked())
            self.frames_fig.canvas.draw_idle()

    # ================================================================ export and project files
    def _save_path(self, title, default, filt):
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        return path

    def export_summary(self, path=None) -> None:
        path = path or self._save_path("Summary statistics", "assessor_summary.csv", "CSV (*.csv)")
        if path:
            A.write_summary_csv(path, self.rows)

    def export_frames(self, path=None) -> None:
        path = path or self._save_path("Per-frame values", "assessor_frames.csv", "CSV (*.csv)")
        if path:
            A.write_frames_csv(path, self.comps, self.project.settings.measures)

    def export_figures(self, path=None) -> None:
        """Both figures, as <name>_distributions.<ext> and <name>_frames_<measure>.<ext>."""
        path = path or self._save_path("Figures", "assessor.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        p = Path(path)
        st = self.project.settings
        fig = Figure(figsize=(8, max(3.2, 2.6 * max(1, len({c.bone for c in self.comps})))), facecolor="white")
        draw_distributions(fig, self.all_comps, st, cap=self.cap.isChecked())
        fig.savefig(p.with_name(p.stem + "_distributions" + p.suffix))
        for key in st.measures:
            for trial in self.project.trials() or [None]:
                fig = Figure(figsize=(8, max(3.2, 2.2 * max(1, len({c.bone for c in self.comps})))),
                             facecolor="white")
                draw_over_frames(fig, self.comps, st, key, trial if len(self.project.trials()) > 1 else None,
                                 cap=self.cap.isChecked())
                tag = f"_{trial}" if len(self.project.trials()) > 1 else ""
                fig.savefig(p.with_name(f"{p.stem}_frames_{key}{tag}{p.suffix}"))

    def save_project(self, path=None) -> None:
        path = path or self._save_path("Save assessor project", f"{self.state.name}.bvrassess.json",
                                       "Assessor project (*.bvrassess.json)")
        if path:
            self.read_settings()
            self.project.name = self.state.name
            self.project.save(path)

    def open_project(self, path=None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open assessor project", "",
                                                  "Assessor project (*.bvrassess.json *.json)")
            if not path:
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.project = A.AssessmentProject.load(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Couldn't open the project", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.state.name = self.project.name
        self.titleChanged.emit(self.state.name)
        self.apply_settings(self.project.settings)
        self.schedule()

    def close_tab(self) -> None:
        self._timer.stop()
