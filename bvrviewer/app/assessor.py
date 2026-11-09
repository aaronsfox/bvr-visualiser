"""Assessor Evaluation tab: reliability of raters' tracking, and validity against a criterion.

Left: the set-up. Bones (optional surfaces for surface distance, optional coordinate systems from a model
for bone-axes errors), the tracking files (trial, bone, rater, session, criterion; read from file and
folder names with a pattern), and the analysis (validity / inter-rater / intra-rater, measures,
thresholds, pooling). Right: the results, recomputed whenever the set-up changes: a summary table,
violin plots of the errors (Welte et al. 2022, figure 4) and the error at every frame (figure 5).

The calculations are in core/assessment.py; see the notes there on what each measure means.
"""

from __future__ import annotations

import os
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
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLineEdit,
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
from .assess_plots import draw_distributions, draw_over_frames
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


class AssessorTab(QWidget):
    kind = "assessor"
    titleChanged = Signal(str)

    def __init__(self, state=None):
        super().__init__()
        self.state = state or _State()
        self.project = A.AssessmentProject()
        self.comps: list = []
        self.rows: list = []
        self._format: dict | None = None          # tracking options remembered while importing
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.recompute)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)
        root.addWidget(split)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_results())
        split.setSizes([520, 1000])
        self.refresh_tables()
        self.recompute()

    # ================================================================ set-up panel
    def _build_setup(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
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
        v.addWidget(label("Load each rater's Autoscoper tracking of the same trial (and, for validity, the "
                          "criterion tracking, e.g. from beads). Results update as you go."))

        # ---- tracking files
        v.addWidget(label("1  Tracking files", "H2", wrap=False))
        pr = QHBoxLayout()
        pr.addWidget(label("Name pattern", wrap=False))
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText("e.g. {rater}/{trial}_*_{bone}   (optional)")
        self.pattern.setToolTip(
            "How trial, rater, session and bone appear in the file paths, relative to the folder you add.\n"
            "Fields: {trial} {rater} {session} {bone}; * matches anything; / separates folders.\n"
            "Example: Raters/Jo/T0019_jog0001_s2_tib.tra with pattern {rater}/{trial}_*_s{session}_{bone}.\n"
            "Without a pattern: bone = a bone name in the file name, rater = the folder name, and folders "
            "or files named 'reference', 'criterion', 'gold' or 'bead' are the criterion.\n"
            "Anything can be corrected in the table (double-click a cell).")
        self.pattern.editingFinished.connect(self._pattern_changed)
        pr.addWidget(self.pattern, 1)
        v.addLayout(pr)
        self.files = QTableWidget(0, 7)
        self.files.setHorizontalHeaderLabels(["Trial", "Bone", "Rater", "Session", "Criterion", "Frames", "File"])
        self.files.verticalHeader().setVisible(False)
        self.files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.files.horizontalHeader().setStretchLastSection(True)
        self.files.setMinimumHeight(220)
        self.files.itemChanged.connect(self._file_cell_changed)
        self.files.itemDoubleClicked.connect(self._file_double_clicked)
        self.files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files.customContextMenuRequested.connect(self._files_menu)
        v.addWidget(self.files, 1)
        fb = QHBoxLayout()
        for text, fn in [("Add folder…", self.add_folder), ("Add files…", self.add_files),
                         ("Format and frames…", self.edit_format), ("Remove", self.remove_files)]:
            b = small_button(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            fb.addWidget(b)
        fb.addStretch(1)
        v.addLayout(fb)
        self.problems = label("", "Warn")
        v.addWidget(self.problems)

        # ---- bones
        v.addWidget(label("2  Bones (optional)", "H2", wrap=False))
        v.addWidget(label("A surface per bone adds the surface distance (how far the bone's surface is from "
                          "where it should be). A coordinate system from a Model Builder model adds errors "
                          "about the bone's own axes. Both must be in the space the tracking maps (Autoscoper "
                          "space for SlicerAutoscoperM)."))
        self.bones = QTableWidget(0, 4)
        self.bones.setHorizontalHeaderLabels(["Bone", "Trial", "Surface", "Axes"])
        self.bones.verticalHeader().setVisible(False)
        self.bones.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bones.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.bones.horizontalHeader().setStretchLastSection(True)
        self.bones.setMinimumHeight(110)
        v.addWidget(self.bones)
        bb = QHBoxLayout()
        for text, fn in [("Add surfaces…", self.add_surfaces), ("From .cfg…", self.add_cfg),
                         ("Axes from model…", self.add_axes), ("Remove", self.remove_bones)]:
            b = small_button(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            bb.addWidget(b)
        bb.addStretch(1)
        v.addLayout(bb)

        # ---- analysis
        v.addWidget(label("3  Analysis", "H2", wrap=False))
        form = QFormLayout()
        self.mode = QComboBox()
        for k, t in A.MODES:
            self.mode.addItem(t, k)
        self.mode.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Compare", self.mode)
        self.session = QComboBox()
        self.session.setToolTip("Inter-rater: which session of each rater to compare")
        self.session.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Session", self.session)
        self.measure_box = QWidget()
        mv = QVBoxLayout(self.measure_box)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(2)
        self.measure_checks = {}
        for key, m in A.MEASURES.items():
            c = QCheckBox(m.label + {"surface": "  (needs surfaces)", "axes": "  (needs axes)"}.get(m.needs, ""))
            c.setChecked(key in A.DEFAULT_MEASURES)
            c.toggled.connect(self._settings_changed)
            mv.addWidget(c)
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
        v.addStretch(1)
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

    # ================================================================ tracking files
    def add_folder(self, folder: str | None = None) -> None:
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Folder with the tracking files (searched with its "
                                                            "subfolders)")
            if not folder:
                return
        root = Path(folder)
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".tra", ".csv", ".txt") and p.is_file())
        if not paths:
            QMessageBox.information(self, "No tracking files", f"No .tra files in {root}.")
            return
        self._add(paths, root)

    def add_files(self, paths=None) -> None:
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Tracking files", "",
                                                    "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if not paths:
                return
        paths = [Path(p) for p in paths]
        root = Path(os.path.commonpath([str(p.parent) for p in paths])).parent
        self._add(paths, root)

    def _add(self, paths, root: Path) -> None:
        known = {e.path.resolve() for e in self.project.entries}
        new = [p for p in paths if p.resolve() not in known]
        names = self.project.bone_names()
        self._format = None
        added = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for i, p in enumerate(new):
                f = A.guess_fields(p, root, self.project.bone_names() or names, self.project.pattern)
                e = A.TrackingEntry(path=p, trial=f.get("trial", "Trial 1"), bone=f.get("bone", ""),
                                    rater=f.get("rater", ""), session=f.get("session", "1"),
                                    criterion=f.get("criterion", False))
                if f.get("trial") is None and self.project.trials():
                    e.trial = self.project.trials()[0]
                QApplication.restoreOverrideCursor()
                ok = self._choose_format(e, more=len(new) - i - 1)
                QApplication.setOverrideCursor(Qt.WaitCursor)
                if ok is None:          # cancelled: stop importing
                    break
                e.load()
                self.project.entries.append(e)
                added.append(e)
        finally:
            QApplication.restoreOverrideCursor()
        # bones learnt from other files (e.g. through the pattern) help name the rest
        names = self.project.bone_names()
        for e in added:
            if not e.bone:
                e.bone = A.guess_fields(e.path, root, names).get("bone", "")
        self.refresh_tables()
        self.schedule()

    def _choose_format(self, e: A.TrackingEntry, more: int = 0):
        """Format options for a file: the ones remembered during this import if they fit, else ask."""
        try:
            detected = load_tracking(e.path)
        except (TrackingError, OSError) as err:
            e.error = str(err)
            return True
        rem = self._format
        if rem and detected.fmt.n_volumes == 1 and rem.get("kind") == detected.fmt.kind:
            try:
                if not load_tracking(e.path, 0, rem).warnings:
                    e.overrides = dict(rem)
                    return True
            except TrackingError:
                pass
        dlg = FormatDialog(self, detected.fmt, 0, path=e.path, detected=detected.fmt, offset=e.offset,
                           n_rows=detected.n_rows, bone=e.bone or e.path.stem, ask_remember=more > 0)
        dlg.remember.setText(f"Use these options for the other {more} files being added (you're asked again "
                             "for any file they don't fit)")
        dlg.remember.setChecked(more > 0)
        if dlg.exec() != QDialog.Accepted:
            return None
        (e.overrides, e.volume), e.offset = dlg.values(), dlg.offset
        if dlg.remember.isChecked():
            self._format = dict(e.overrides)
        return True

    def edit_format(self) -> None:
        rows = sorted({i.row() for i in self.files.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Format and frames", "Select a tracking file in the table first.")
            return
        e = self.project.entries[rows[0]]
        try:
            detected = load_tracking(e.path)
        except (TrackingError, OSError) as err:
            QMessageBox.warning(self, "Couldn't read the file", str(err))
            return
        fmt = e.tracking.fmt if e.tracking is not None else detected.fmt
        others = [(f"{o.who} {o.bone}", o.tracking.offset, o.tracking.n_rows) for o in self.project.entries
                  if o is not e and o.tracking is not None and o.trial == e.trial and o.bone == e.bone]
        dlg = FormatDialog(self, fmt, e.volume, path=e.path, detected=detected.fmt, offset=e.offset,
                           n_rows=detected.n_rows, others=others, bone=f"{e.who} · {e.bone}",
                           ask_remember=len(rows) > 1)
        if len(rows) > 1:
            dlg.remember.setText(f"Apply to all {len(rows)} selected files")
            dlg.remember.setChecked(True)
        if dlg.exec() != QDialog.Accepted:
            return
        (ov, vol), off = dlg.values(), dlg.offset
        targets = [self.project.entries[r] for r in rows] if dlg.remember.isChecked() else [e]
        for t in targets:
            t.overrides, t.volume, t.offset = dict(ov), vol, off
            t.load()
        self.refresh_tables()
        self.schedule()

    def remove_files(self) -> None:
        rows = sorted({i.row() for i in self.files.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.project.entries[r]
        self.refresh_tables()
        self.schedule()

    def _files_menu(self, pos) -> None:
        rows = sorted({i.row() for i in self.files.selectedIndexes()})
        if not rows:
            return
        m = QMenu(self)
        m.addAction("Set as criterion", lambda: self._set_field(rows, "criterion", True))
        m.addAction("Set as rater's tracking", lambda: self._set_field(rows, "criterion", False))
        for f, title in [("trial", "Trial"), ("bone", "Bone"), ("rater", "Rater"), ("session", "Session")]:
            m.addAction(f"Set {title.lower()} of selected…", lambda f=f, t=title: self._ask_field(rows, f, t))
        m.addSeparator()
        m.addAction("Format and frames…", self.edit_format)
        m.addAction("Refresh from file", lambda: self._reload(rows))
        m.addAction("Remove", self.remove_files)
        m.exec(self.files.viewport().mapToGlobal(pos))

    def _ask_field(self, rows, field_name, title) -> None:
        cur = getattr(self.project.entries[rows[0]], field_name)
        text, ok = QInputDialog.getText(self, title, f"{title} for the {len(rows)} selected files:", text=cur)
        if ok and text.strip():
            self._set_field(rows, field_name, text.strip())

    def _set_field(self, rows, field_name, value) -> None:
        for r in rows:
            setattr(self.project.entries[r], field_name, value)
        self.refresh_tables()
        self.schedule()

    def _reload(self, rows) -> None:
        for r in rows:
            self.project.entries[r].load()
        self.refresh_tables()
        self.schedule()

    def _pattern_changed(self) -> None:
        text = self.pattern.text().strip()
        if text == self.project.pattern:
            return
        try:
            A.pattern_regex(text)
        except ValueError as e:
            QMessageBox.warning(self, "Name pattern", str(e))
            return
        self.project.pattern = text
        if text and self.project.entries and QMessageBox.question(
                self, "Name pattern", "Re-read trial, rater, session and bone of the files already added "
                                      "with this pattern?") == QMessageBox.Yes:
            root = Path(os.path.commonpath([str(e.path.parent) for e in self.project.entries]))
            for _ in range(4):                      # try a few folder levels up for the pattern's root
                hits = 0
                for e in self.project.entries:
                    rel = e.path.relative_to(root).with_suffix("").as_posix()
                    hits += bool(A.pattern_regex(text).match(rel))
                if hits == len(self.project.entries) or root.parent == root:
                    break
                root = root.parent
            names = self.project.bone_names()
            for e in self.project.entries:
                f = A.guess_fields(e.path, root, names, text)
                for k in ("trial", "rater", "session", "bone"):
                    if k in f:
                        setattr(e, k, f[k])
            self.refresh_tables()
            self.schedule()

    def _file_cell_changed(self, it: QTableWidgetItem) -> None:
        if self._busy:
            return
        e = self.project.entries[it.row()]
        col = it.column()
        if col == 4:
            e.criterion = it.checkState() == Qt.Checked
        elif col in (0, 1, 2, 3):
            setattr(e, ("trial", "bone", "rater", "session")[col], it.text().strip())
        self.refresh_tables()
        self.schedule()

    def _file_double_clicked(self, it: QTableWidgetItem) -> None:
        if it.column() in (5, 6):
            self.files.selectRow(it.row())
            self.edit_format()

    # ================================================================ bones
    def _bone(self, name: str, trial: str = "") -> A.BoneInfo:
        for b in self.project.bones:
            if b.name == name and b.trial == trial:
                return b
        b = A.BoneInfo(name=name, trial=trial)
        self.project.bones.append(b)
        return b

    def _match_bone(self, stem: str) -> str | None:
        parts = [p for p in stem.lower().replace("-", "_").split("_") if p]
        for name in sorted(self.project.bone_names(), key=len, reverse=True):
            if name.lower() in parts or stem.lower() == name.lower():
                return name
        return None

    def add_surfaces(self, paths=None) -> None:
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Bone surfaces (in the space the tracking maps)", "",
                                                    SURFACES)
            if not paths:
                return
        for p in map(Path, paths):
            name = self._match_bone(p.stem)
            if name is None:
                name, ok = QInputDialog.getText(self, "Which bone?", f"Bone name for {p.name}:", text=p.stem)
                if not ok or not name.strip():
                    continue
                name = name.strip()
            surf = BoneModel(name=name, path=p)
            if is_volume_file(p):
                QMessageBox.information(self, "TIF volume", "For TIF volumes use From .cfg…, which reads "
                                                            "their voxel size.")
                continue
            if not self._build(surf):
                continue
            b = self._bone(name)
            b.surface, b._points = surf, None
        self.refresh_tables()
        self.schedule()

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
            b = self._bone(name)
            b.surface, b._points = surf, None
        self.refresh_tables()
        self.schedule()

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
        """Coordinate systems from a Model Builder model, matched to bones by name or surface file."""
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
            b = self._bone(name)
            b.axes_path, b.axes_segment = str(path), seg.name
            A.load_bone_axes(b)
            matched.append(f"{name} → {seg.name}")
        self.refresh_tables()
        self.schedule()
        QMessageBox.information(self, "Coordinate systems",
                                ("Matched: " + ", ".join(matched) if matched else "No bones matched.")
                                + (f"\nNo coordinate system for: {', '.join(missing)}" if missing else ""))

    def remove_bones(self) -> None:
        rows = sorted({i.row() for i in self.bones.selectedIndexes()}, reverse=True)
        shown = self._bone_rows()
        for r in rows:
            b = shown[r]
            if b in self.project.bones:
                self.project.bones.remove(b)
        self.refresh_tables()
        self.schedule()

    def _bone_rows(self) -> list:
        """BoneInfo rows to show: those with something set, plus a placeholder for other bones in the files."""
        rows = list(self.project.bones)
        have = {b.name for b in rows}
        for name in self.project.bone_names():
            if name not in have:
                rows.append(A.BoneInfo(name=name))
        return rows

    # ================================================================ tables
    def refresh_tables(self) -> None:
        self._busy = True
        p = self.project
        self.pattern.setText(p.pattern)
        self.files.setRowCount(len(p.entries))
        for r, e in enumerate(p.entries):
            self.files.setItem(r, 0, _item(e.trial, True))
            self.files.setItem(r, 1, _item(e.bone, True))
            self.files.setItem(r, 2, _item("" if e.criterion else e.rater, not e.criterion))
            self.files.setItem(r, 3, _item("" if e.criterion else e.session, not e.criterion))
            c = _item("")
            c.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            c.setCheckState(Qt.Checked if e.criterion else Qt.Unchecked)
            c.setToolTip("The criterion (gold standard) tracking that raters are compared with")
            self.files.setItem(r, 4, c)
            if e.tracking is not None:
                t = e.tracking
                first, last = t.offset + 1, t.offset + t.n_rows
                text = f"{max(first, 1)}–{last}" + (f" · {t.n_untracked} untracked" if t.n_untracked else "")
                fr = _item(text, tip=f"{t.fmt.describe()}\nDouble-click to change the format or frames")
                if t.warnings:
                    fr.setForeground(QColor("#F2C48D"))
                    fr.setToolTip("\n".join(t.warnings))
            else:
                fr = _item("can't read", tip=e.error)
                fr.setForeground(QColor("#F0A3A3"))
            self.files.setItem(r, 5, fr)
            self.files.setItem(r, 6, _item(e.path.name, tip=str(e.path)))
            if e.criterion:
                for col in range(4):
                    self.files.item(r, col).setForeground(QColor("#8FD3A9"))
        bones = self._bone_rows()
        self.bones.setRowCount(len(bones))
        for r, b in enumerate(bones):
            self.bones.setItem(r, 0, _item(b.name))
            self.bones.setItem(r, 1, _item(b.trial or "all"))
            s = b.surface
            self.bones.setItem(r, 2, _item(Path(s.path).name if s is not None else "none",
                                           tip=str(s.path) if s is not None else "Add surfaces… or From .cfg…"))
            ax = f"{b.axes_segment} ({Path(b.axes_path).name})" if b.acs is not None else (
                "not found" if b.axes_path else "none")
            self.bones.setItem(r, 3, _item(ax, tip=b.axes_path or "Axes from model…"))
        probs = p.problems()
        self.problems.setText("\n".join("⚠ " + s for s in probs[:8]) + (f"\n… and {len(probs) - 8} more"
                                                                         if len(probs) > 8 else ""))
        self.problems.setVisible(bool(probs))
        # analysis options that depend on the files
        avail = set(p.available_measures())
        for key, c in self.measure_checks.items():
            c.setEnabled(key in avail)
        sessions = p.sessions()
        cur = self.session.currentData()
        self.session.blockSignals(True)
        self.session.clear()
        self.session.addItem("Each rater's first session", "")
        for s in sessions:
            self.session.addItem(f"Session {s}", s)
        i = self.session.findData(cur if cur is not None else p.settings.inter_session)
        self.session.setCurrentIndex(max(i, 0))
        self.session.blockSignals(False)
        self._busy = False
        self._sync_settings_widgets()

    def _sync_settings_widgets(self) -> None:
        mode = self.mode.currentData()
        self.session.setEnabled(mode == "inter" and len(self.project.sessions()) > 1)

    # ================================================================ settings and results
    def _settings_changed(self, *_):
        if self._busy:
            return
        self._sync_settings_widgets()
        self.schedule()

    def read_settings(self) -> A.AnalysisSettings:
        avail = set(self.project.available_measures())
        st = A.AnalysisSettings(
            mode=self.mode.currentData(),
            measures=tuple(k for k, c in self.measure_checks.items() if c.isChecked() and k in avail),
            rotation_threshold=self.rot_thr.value(), translation_threshold=self.tr_thr.value(),
            inter_session=self.session.currentData() or "", pool=self.pool.currentData())
        self.project.settings = st
        return st

    def apply_settings(self, st: A.AnalysisSettings) -> None:
        self._busy = True
        self.mode.setCurrentIndex(max(self.mode.findData(st.mode), 0))
        for k, c in self.measure_checks.items():
            c.setChecked(k in st.measures)
        self.rot_thr.setValue(st.rotation_threshold)
        self.tr_thr.setValue(st.translation_threshold)
        self.pool.setCurrentIndex(max(self.pool.findData(st.pool), 0))
        self._busy = False
        self.refresh_tables()
        i = self.session.findData(st.inter_session)
        self.session.setCurrentIndex(max(i, 0))

    def schedule(self) -> None:
        self._timer.start(50)

    def recompute(self) -> None:
        st = self.read_settings()
        self.comps = [c for c in self.project.comparisons(st) if len(c.frames)]
        self.rows = A.summarise(self.comps, st)
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
        n_crit = sum(e.criterion for e in p.entries)
        bits = [f"{len(p.raters())} rater{'s' * (len(p.raters()) != 1)}",
                f"{len(p.bone_names())} bone{'s' * (len(p.bone_names()) != 1)}",
                f"{len(p.trials())} trial{'s' * (len(p.trials()) != 1)}"]
        if n_crit:
            bits.append("criterion loaded")
        text = " · ".join(bits) + f" — {len(self.comps)} comparison{'s' * (len(self.comps) != 1)}"
        if not self.comps:
            text += {"validity": ". Validity needs a criterion tracking and at least one rater's tracking of "
                                 "the same bone and trial (tick Criterion for the reference file).",
                     "inter": ". Inter-rater needs two or more raters' tracking of the same bone and trial.",
                     "intra": ". Intra-rater needs a rater's tracking from two or more sessions."}[st.mode]
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

    SUMMARY_COLS: tuple = (("n", "Frames", 0), ("mean", "Mean", 2), ("sd", "SD", 2), ("median", "Median", 2),
                    ("iqr", "IQR", 2), ("rmse", "RMS", 2), ("max", "Max", 2), ("loa", "LoA (95%)", 2),
                    ("robust", "Robust LoA", 2), ("mad", "MAD", 2), ("outliers", "Outliers", 0),
                    ("pct_over", "% > thr.", 0))

    def _fill_summary(self, st: A.AnalysisSettings) -> None:
        head = ["Trial", "Bone", {"validity": "Rater", "inter": "Raters", "intra": "Sessions"}[st.mode], "Measure"]
        cols = self.SUMMARY_COLS
        self.summary.clear()
        self.summary.setColumnCount(len(head) + len(cols))
        self.summary.setHorizontalHeaderLabels(head + [c[1] for c in cols])
        tips = {"LoA (95%)": "Bland–Altman limits of agreement: mean ± 1.96 SD (± half-width of their 95% CI)",
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
            for k, _t, d in cols:
                if k == "iqr":
                    vals.append(_fmt(s.get("q3", np.nan) - s.get("q1", np.nan), d))
                elif k == "loa":
                    vals.append(f"{_fmt(s.get('loa_low'), d)} to {_fmt(s.get('loa_high'), d)} "
                                f"(±{_fmt(s.get('loa_ci'), d)})" if s.get("n", 0) > 1 else "–")
                elif k == "robust":
                    vals.append(f"{_fmt(s.get('robust_low'), d)} to {_fmt(s.get('robust_high'), d)}")
                elif k == "pct_over":
                    vals.append(f"{s.get('pct_over', 0):.0f}% ({s.get('over', 0)})")
                else:
                    vals.append(_fmt(s.get(k), d) if d else str(s.get(k, "–")))
            for c, text in enumerate(vals):
                it = _item(text)
                if c >= len(head):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary.setItem(r, c, it)
            if s.get("pct_over", 0) > 0:
                self.summary.item(r, self.summary.columnCount() - 1).setForeground(QColor("#F2C48D"))
        self.summary.resizeColumnsToContents()

    def redraw(self, *_):
        st = self.project.settings
        tab = self.tabs.currentWidget()
        if tab is self.dist:
            n_bones = 1 if st.pool == "all" else len({c.bone for c in self.comps}) or 1
            self.dist.fig.set_size_inches(8, max(3.2, 2.6 * n_bones))
            draw_distributions(self.dist.fig, self.comps, st, cap=self.cap.isChecked())
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
        draw_distributions(fig, self.comps, st, cap=self.cap.isChecked())
        fig.savefig(p.with_name(p.stem + "_distributions" + p.suffix), dpi=200)
        for key in st.measures:
            for trial in self.project.trials() or [None]:
                fig = Figure(figsize=(8, max(3.2, 2.2 * max(1, len({c.bone for c in self.comps})))),
                             facecolor="white")
                draw_over_frames(fig, self.comps, st, key, trial if len(self.project.trials()) > 1 else None,
                                 cap=self.cap.isChecked())
                tag = f"_{trial}" if len(self.project.trials()) > 1 else ""
                fig.savefig(p.with_name(f"{p.stem}_frames_{key}{tag}{p.suffix}"), dpi=200)

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
