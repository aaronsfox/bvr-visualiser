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
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
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
from .assessor_setup import (
    ABONE,
    ASSESSOR,
    ASSESSORS,
    AXES,
    BONE,
    BONES,
    REFERENCE,
    SURFACE,
    TRACK,
    AssessorDialog,
    BoneDialog,
    MatchDialog,
    SetupTree,
)
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
        self._format: dict | None = None          # tracking options remembered for this tab (when ticked)
        self._last_dir = ""
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
        v.addWidget(label("Add each assessor, then their bones, then each bone's tracking file. Bones are matched "
                          "across assessors by label. Add the reference (criterion) tracking for validity."))

        bar = QHBoxLayout()
        for text, fn, tip in [("+ Add assessor", self.add_assessor, "An assessor (rater) in one session"),
                              ("+ Add reference", self.add_reference,
                               "The criterion (gold-standard) tracking, e.g. bead-based")]:
            b = small_button(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, f=fn: f())
            bar.addWidget(b)
        more = QPushButton("More ▾")
        more.setObjectName("Small")
        m = QMenu(more)
        m.addAction("Import a folder of tracking files…", self.import_folder)
        m.addSeparator()
        m.addAction("Bone surfaces…", self.add_surfaces)
        m.addAction("Bone surfaces from an Autoscoper .cfg…", self.add_cfg)
        m.addAction("Bone coordinate systems from a model…", self.add_axes)
        more.setMenu(m)
        bar.addWidget(more)
        bar.addStretch(1)
        v.addLayout(bar)

        self.tree = SetupTree()
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.itemDoubleClicked.connect(self._tree_double_clicked)
        v.addWidget(self.tree, 1)
        self.problems = label("", "Warn")
        v.addWidget(self.problems)

        # ---- analysis
        v.addWidget(label("Analysis", "H2", wrap=False))
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

    # ================================================================ assessors and the reference
    def _changed(self, select: tuple | None = None) -> None:
        self.refresh_tables(select)
        self.schedule()

    def add_assessor(self, name: str | None = None, session: str = "1", copy_from=None, ask: bool = True):
        """Add an assessor (one rater in one session), optionally starting with another one's bones."""
        p = self.project
        if ask:
            n = len(p.raters_list()) + 1
            default = name or next(f"Assessor {i}" for i in range(n, n + 100)
                                   if p.find_assessor(f"Assessor {i}", session) is None)
            dlg = AssessorDialog(self, p, name=default, session=session, copy_default=copy_from)
            if dlg.exec() != QDialog.Accepted:
                return None
            name, session, copy_from = dlg.values()
        try:
            a = p.add_assessor(name, session, copy_from)
        except ValueError as e:
            QMessageBox.warning(self, "Add assessor", str(e))
            return None
        self._changed((ASSESSOR, a, None, None))
        return a

    def add_session(self, a: A.Assessor):
        """Another session of the same assessor, with the same bones (for intra-rater reliability)."""
        sessions = [o.session for o in self.project.raters_list() if o.name == a.name]
        nxt = next(str(i) for i in range(2, 100) if str(i) not in sessions)
        return self.add_assessor(a.name, nxt, a, ask=False)

    def add_reference(self, copy_from=None, ask: bool = True):
        p = self.project
        if p.reference is not None:
            QMessageBox.information(self, "Reference", "There is already a reference. Add bones or tracking to it, "
                                                       "or remove it first.")
            return p.reference
        if ask:
            dlg = AssessorDialog(self, p, reference=True, copy_default=copy_from)
            if dlg.exec() != QDialog.Accepted:
                return None
            copy_from = dlg.values()[2]
        a = p.add_reference(copy_from)
        self._changed((REFERENCE, a, None, None))
        return a

    def rename_assessor(self, a: A.Assessor) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Rename assessor")
        f = QFormLayout(dlg)
        name = QLineEdit(a.name)
        sess = QLineEdit(a.session)
        f.addRow("Name", name)
        f.addRow("Session", sess)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not name.text().strip() or not sess.text().strip():
            return
        try:
            self.project.rename_assessor(a, name.text().strip(), sess.text().strip())
        except ValueError as e:
            QMessageBox.warning(self, "Rename assessor", str(e))
            return
        self._changed((ASSESSOR, a, None, None))

    def remove_assessor(self, a: A.Assessor) -> None:
        p = self.project
        n = len(p.entries_of(a))
        if n and QMessageBox.question(self, "Remove", f"Remove {p.display_name(a)} and its {n} tracking "
                                                      f"file{'s' * (n != 1)}? (The files themselves are kept.)"
                                      ) != QMessageBox.Yes:
            return
        p.remove_assessor(a)
        self._changed()

    # ================================================================ an assessor's bones and tracking
    def add_bone(self, a: A.Assessor, label_text: str | None = None, path=None, trial: str | None = None):
        """Add a bone to an assessor (picking a label others use, or a new one), then its tracking file."""
        p = self.project
        want_file = path is not None
        if label_text is None:
            dlg = BoneDialog(self, p, a)
            if dlg.exec() != QDialog.Accepted or not dlg.value():
                return
            label_text, want_file = dlg.value(), dlg.file_now.isChecked()
        try:
            p.add_bone(a, label_text)
        except ValueError as e:
            QMessageBox.warning(self, "Add bone", str(e))
            return
        self._changed((ABONE, a, label_text, None))
        if want_file:
            self.load_tracking(a, label_text, trial or p.trials()[0], path)

    def relabel_bone(self, a: A.Assessor, bone: str) -> None:
        dlg = BoneDialog(self, self.project, a, current=bone, title="Change bone label", ask_file=False)
        if dlg.exec() != QDialog.Accepted or not dlg.value():
            return
        try:
            self.project.relabel_bone(a, bone, dlg.value())
        except ValueError as e:
            QMessageBox.warning(self, "Change bone label", str(e))
            return
        self._changed((ABONE, a, dlg.value(), None))

    def remove_bone(self, a: A.Assessor, bone: str) -> None:
        self.project.remove_bone(a, bone)
        self._changed((ASSESSOR if not a.criterion else REFERENCE, a, None, None))

    def load_tracking(self, a: A.Assessor, bone: str, trial: str, path=None, more: int = 0) -> bool:
        """Choose (or replace) the tracking file of an assessor's bone in a trial."""
        p = self.project
        if path is None:
            old = p.entry(a, bone, trial)
            start = str(old.path.parent) if old is not None else self._last_dir
            path, _ = QFileDialog.getOpenFileName(self, f"Tracking file: {p.display_name(a)} · {bone}", start,
                                                  "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if not path:
                return False
        path = Path(path)
        self._last_dir = str(path.parent)
        fmt = self._choose_format(path, f"{p.display_name(a)} · {bone}", more)
        if fmt is None:
            return False
        overrides, volume, offset = fmt
        p.set_tracking(a, bone, trial, path, overrides, volume, offset)
        self._changed((TRACK, a, bone, trial) if len(p.trials()) > 1 else (ABONE, a, bone, None))
        return True

    def load_tracking_files(self, a: A.Assessor, paths=None) -> None:
        """Several files for one assessor at once: each matched to a bone (by label in its name) and a trial."""
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, f"Tracking files: {self.project.display_name(a)}",
                                                    self._last_dir,
                                                    "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if not paths:
                return
        dlg = MatchDialog(self, self.project, a, paths)
        if dlg.exec() != QDialog.Accepted:
            return
        rows = [r for r in dlg.values() if r[1]]
        for i, (path, bone, trial) in enumerate(rows):
            if bone not in a.bones:
                self.project.add_bone(a, bone)
            if not self.load_tracking(a, bone, trial, path, more=len(rows) - i - 1):
                break

    def add_trial_tracking(self, a: A.Assessor, bone: str) -> None:
        trials = self.project.trials()
        trial, ok = QInputDialog.getItem(self, "Another trial", "Trial name (a new name adds a trial):",
                                         trials + [f"Trial {len(trials) + 1}"], len(trials), True)
        if ok and trial.strip():
            self.load_tracking(a, bone, trial.strip())

    def refresh_tracking(self, e: A.TrackingEntry) -> None:
        e.load()
        self._changed()

    def remove_tracking(self, e: A.TrackingEntry) -> None:
        self.project.entries.remove(e)
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
        fmt = e.tracking.fmt if e.tracking is not None else detected.fmt
        others = [(f"{p.display_name(p.assessor_of(o)) if p.assessor_of(o) else o.who}", o.tracking.offset,
                   o.tracking.n_rows) for o in p.entries
                  if o is not e and o.tracking is not None and o.trial == e.trial and o.bone == e.bone]
        a = p.assessor_of(e)
        dlg = FormatDialog(self, fmt, e.volume, path=e.path, detected=detected.fmt, offset=e.offset,
                           n_rows=detected.n_rows, others=others,
                           bone=f"{p.display_name(a) if a else e.who} · {e.bone}")
        if dlg.exec() != QDialog.Accepted:
            return
        (e.overrides, e.volume), e.offset = dlg.values(), dlg.offset
        e.load()
        self._changed()

    # ================================================================ folder import (optional shortcut)
    def import_folder(self, folder: str | None = None, pattern: str | None = None) -> None:
        """Make assessors, bones and tracking from a folder of files, using a name pattern."""
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Folder with the tracking files (and subfolders)")
            if not folder:
                return
        if pattern is None:
            pattern, ok = QInputDialog.getText(
                self, "Name pattern",
                "How assessor, session, trial and bone appear in the paths (relative to the folder):\n"
                "{rater} {session} {trial} {bone}, * for anything, / between folders.\n"
                "Example: {rater}/{trial}_s{session}_{bone}  for  Jo/T0019_jog_s2_tib.tra\n\n"
                "Leave empty to guess: rater = folder name, bone = a known bone label in the file name,\n"
                "folders or files named reference/criterion/gold/bead = the reference.",
                text=self.project.pattern)
            if not ok:
                return
        try:
            A.pattern_regex(pattern)
        except ValueError as e:
            QMessageBox.warning(self, "Name pattern", str(e))
            return
        self.project.pattern = pattern
        root = Path(folder)
        paths = sorted(x for x in root.rglob("*") if x.suffix.lower() in (".tra", ".csv", ".txt") and x.is_file())
        if not paths:
            QMessageBox.information(self, "No tracking files", f"No .tra files in {root}.")
            return
        p = self.project
        guessed = [(x, A.guess_fields(x, root, p.bone_names(), pattern)) for x in paths]
        names = A._unique(p.bone_names() + [f["bone"] for _x, f in guessed if f.get("bone")])
        default_trial = p.trials()[0]
        skipped = []
        for i, (x, f) in enumerate(guessed):
            if not f.get("bone"):
                f = {**f, **{k: v for k, v in A.guess_fields(x, root, names).items() if k == "bone"}}
            if not f.get("bone"):
                skipped.append(x.name)
                continue
            if f.get("criterion"):
                a = p.reference or p.add_reference()
            else:
                rater, session = f.get("rater") or x.parent.name, f.get("session", "1")
                a = p.find_assessor(rater, session) or p.add_assessor(rater, session)
            if f["bone"] not in a.bones:
                p.add_bone(a, f["bone"])
            fmt = self._choose_format(x, f"{p.display_name(a)} · {f['bone']}", more=len(guessed) - i - 1)
            if fmt is None:
                break
            p.set_tracking(a, f["bone"], f.get("trial", default_trial), x, *fmt)
        self._changed()
        if skipped:
            QMessageBox.information(self, "Some files weren't added",
                                    "No bone could be found in these file names (use the pattern's {bone}, or add "
                                    "them under an assessor):\n" + "\n".join(skipped[:20]))

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

    def rename_bone_everywhere(self, bone: str) -> None:
        new, ok = QInputDialog.getText(self, "Rename bone", f"New label for '{bone}' (for every assessor and the "
                                                            "reference):", text=bone)
        if not ok or not new.strip():
            return
        try:
            self.project.rename_bone(bone, new.strip())
        except ValueError as e:
            QMessageBox.warning(self, "Rename bone", str(e))
            return
        self._changed((BONE, None, new.strip(), None))

    def remove_surface(self, bone: str) -> None:
        b = self.project.bone_info_for(bone)
        b.surface, b._points = None, None
        self._changed()

    def remove_axes(self, bone: str) -> None:
        b = self.project.bone_info_for(bone)
        b.axes_path = b.axes_segment = None
        b.acs = None
        self._changed()

    # ================================================================ the tree's menus
    def _selected(self) -> list:
        return [SetupTree.data(it) for it in self.tree.selectedItems()]

    def build_menu(self, data) -> QMenu:
        """The right-click menu for a tree item (also used by tests)."""
        kind, a, bone, trial = data
        p = self.project
        m = QMenu(self)

        def add(text, fn, enabled=True):
            act = m.addAction(text, fn)
            act.setEnabled(enabled)
            return act

        if kind == ASSESSORS:
            add("Add assessor…", self.add_assessor)
            add("Import a folder of tracking files…", self.import_folder)
        elif kind == ASSESSOR or kind == REFERENCE:
            add("Add bone…", lambda: self.add_bone(a))
            add("Load tracking files…", lambda: self.load_tracking_files(a))
            m.addSeparator()
            if kind == ASSESSOR:
                add("Add another session", lambda: self.add_session(a))
                add("Add assessor with the same bones…", lambda: self.add_assessor(copy_from=a))
                add("Rename…", lambda: self.rename_assessor(a))
                m.addSeparator()
                add("Remove assessor", lambda: self.remove_assessor(a))
            else:
                add("Remove reference", lambda: self.remove_assessor(a))
        elif kind == ABONE:
            e = p.entry(a, bone, p.trials()[0])
            add("Load new tracking file…" if e else "Load tracking file…",
                lambda: self.load_tracking(a, bone, p.trials()[0]))
            add("Tracking for another trial…", lambda: self.add_trial_tracking(a, bone))
            m.addSeparator()
            add("Change label…", lambda: self.relabel_bone(a, bone))
            add("Remove bone from " + p.display_name(a), lambda: self.remove_bone(a, bone))
        elif kind == TRACK:
            e = p.entry(a, bone, trial)
            add("Load new file…" if e else "Load file…", lambda: self.load_tracking(a, bone, trial))
            if e is not None:
                add("Refresh current file", lambda: self.refresh_tracking(e))
                add("Format and frames…", lambda: self.edit_format(e))
                m.addSeparator()
                add("Remove tracking", lambda: self.remove_tracking(e))
        elif kind == BONES:
            add("Bone surfaces…", self.add_surfaces)
            add("Bone surfaces from an Autoscoper .cfg…", self.add_cfg)
            add("Bone coordinate systems from a model…", self.add_axes)
        elif kind == BONE:
            add("Rename everywhere…", lambda: self.rename_bone_everywhere(bone))
            add("Load surface…", lambda: self.add_surfaces(bone=bone))
            add("Coordinate systems from a model…", self.add_axes)
        elif kind == SURFACE:
            info = p.bone_info_for(bone)
            add("Load new surface…" if info.surface else "Load surface…", lambda: self.add_surfaces(bone=bone))
            add("Remove surface", lambda: self.remove_surface(bone), enabled=info.surface is not None)
        elif kind == AXES:
            info = p.bone_info_for(bone)
            add("Coordinate systems from a model…", self.add_axes)
            add("Remove coordinate system", lambda: self.remove_axes(bone), enabled=bool(info.axes_path))
        return m

    def _tree_menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if it is None:
            m = QMenu(self)
            m.addAction("Add assessor…", self.add_assessor)
            m.addAction("Add reference…", self.add_reference)
        else:
            m = self.build_menu(SetupTree.data(it))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _tree_double_clicked(self, it, _col) -> None:
        kind, a, bone, trial = SetupTree.data(it)
        p = self.project
        if kind == TRACK:
            e = p.entry(a, bone, trial)
            (self.edit_format(e) if e is not None else self.load_tracking(a, bone, trial))
        elif kind == ABONE and p.entry(a, bone, p.trials()[0]) is None:
            self.load_tracking(a, bone, p.trials()[0])
        elif kind == ASSESSOR:
            self.rename_assessor(a)
        elif kind == SURFACE:
            self.add_surfaces(bone=bone)
        elif kind == AXES:
            self.add_axes()

    # ================================================================ refresh
    def refresh_tables(self, select: tuple | None = None) -> None:
        self._busy = True
        p = self.project
        self.tree.rebuild(p, select)
        probs = p.problems()
        self.problems.setText("\n".join("⚠ " + s for s in probs[:8]) + (f"\n… and {len(probs) - 8} more"
                                                                         if len(probs) > 8 else ""))
        self.problems.setVisible(bool(probs))
        avail = set(p.available_measures())
        for key, c in self.measure_checks.items():
            c.setEnabled(key in avail or (key == "axes" and "axes_rz" in avail))
        sessions = p.sessions()
        cur = self.session.currentData()
        self.session.blockSignals(True)
        self.session.clear()
        self.session.addItem("Each assessor's first session", "")
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
        self.mode.setCurrentIndex(max(self.mode.findData(st.mode), 0))
        for k, c in self.measure_checks.items():
            c.setChecked(k in st.measures or (k == "axes" and any(m.startswith("axes_") for m in st.measures)))
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
