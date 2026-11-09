"""The set-up side of the Assessor Evaluation tab: a tree of bones, the reference and the assessors.

    Bones                       one row per bone label, with its optional Surface and Axes
      tib
        Surface · tib.tif
        Axes · Tibia (knee.bvrmodel.json)
    Reference                   the criterion (e.g. bead tracking), if added
      tib
        Tracking · ref_tib.tra (19 rows)
    Assessors
      Jo                        one assessor in one session ("Jo · session 2" for another session)
        tib
          Tracking · jo_tib.tra (19 rows)
        tal
          Tracking (none: right-click to load)

Bones are matched across the reference and the assessors by their label, so each assessor's bones
are added by picking (or typing) a label; new assessors can start with another assessor's bones.
With several trials, each bone has one Tracking row per trial.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..core import assessment as A
from .panels import label

# tree item kinds
BONES, BONE, SURFACE, AXES = "bones", "bone", "surface", "axes"
REFERENCE, ASSESSORS, ASSESSOR, ABONE, TRACK = "reference", "assessors", "assessor", "abone", "track"

GREEN, TEAL, AMBER, GREY, RED = "#8FD3A9", "#5CC8A8", "#F2C48D", "#5E6573", "#F0A3A3"


def _bold(it: QTreeWidgetItem, colour: str | None = None, col: int = 0) -> None:
    f = QFont()
    f.setBold(True)
    it.setFont(col, f)
    if colour:
        it.setForeground(col, QBrush(QColor(colour)))


def _grey(it: QTreeWidgetItem, col: int = 0) -> None:
    f = QFont()
    f.setItalic(True)
    it.setFont(col, f)
    it.setForeground(col, QBrush(QColor(GREY)))


class SetupTree(QTreeWidget):
    """The tree. Each item carries (kind, assessor or None, bone label or None, trial or None)."""

    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setUniformRowHeights(True)
        self.setMinimumHeight(260)
        self._expanded: set = set()

    @staticmethod
    def data(it: QTreeWidgetItem):
        return it.data(0, Qt.UserRole) if it is not None else None

    def _item(self, parent, text: str, data: tuple, detail: str = "") -> QTreeWidgetItem:
        it = QTreeWidgetItem([text, detail])
        it.setData(0, Qt.UserRole, data)
        (parent.addChild(it) if isinstance(parent, QTreeWidgetItem) else self.addTopLevelItem(it))
        return it

    def _key(self, data) -> tuple:
        kind, a, bone, trial = data
        return (kind, a.key() if a is not None else None, bone, trial)

    def rebuild(self, p: A.AssessmentProject, select: tuple | None = None) -> None:
        """Redraw from the project, keeping expanded items and the selection."""
        if self.topLevelItemCount():
            self._expanded = set()
            stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
            while stack:
                it = stack.pop()
                if it.isExpanded():
                    self._expanded.add(self._key(self.data(it)))
                stack += [it.child(i) for i in range(it.childCount())]
            if select is None and self.currentItem() is not None:
                select = self.data(self.currentItem())
        self.clear()
        trials = p.trials()
        many_trials = len(trials) > 1
        to_select = None
        want = self._key(select) if select is not None else None

        # ---- bones
        names = p.bone_names()
        top = self._item(self, "Bones", (BONES, None, None, None),
                         f"{len(names)} bone{'s' * (len(names) != 1)}" if names else "")
        _bold(top)
        top.setToolTip(0, "Bone labels used by the reference and assessors. A surface adds the surface "
                          "distance; a coordinate system from a model adds errors about the bone's own axes.")
        for name in names:
            info = p.bone_info_for(name)
            users = [p.display_name(a) for a in p.assessors if name in a.bones]
            b = self._item(top, name, (BONE, None, name, None), ", ".join(users))
            b.setToolTip(1, "Used by: " + (", ".join(users) or "nobody yet"))
            s = info.surface
            si = self._item(b, "Surface", (SURFACE, None, name, None), Path(s.path).name if s is not None else "")
            (_bold(si, GREEN) if s is not None else _grey(si))
            si.setToolTip(0, str(s.path) if s is not None else "Optional: right-click → Load surface…")
            ax = self._item(b, "Axes", (AXES, None, name, None),
                            f"{info.axes_segment} ({Path(info.axes_path).name})" if info.axes_path else "")
            if info.acs is not None:
                _bold(ax, TEAL)
            elif info.axes_path:
                _bold(ax, AMBER)
                ax.setText(1, ax.text(1) + " not found")
            else:
                _grey(ax)
            ax.setToolTip(0, "Optional: right-click → Axes from model…")
            for it in (b, si, ax):
                if want is not None and self._key(self.data(it)) == want:
                    to_select = it

        # ---- reference and assessors
        def assessor_items(parent, a: A.Assessor):
            nonlocal to_select
            for bone in a.bones:
                bi = self._item(parent, bone, (ABONE, a, bone, None))
                n_ok = 0
                for trial in trials:
                    e = p.entry(a, bone, trial)
                    title = f"Tracking · {trial}" if many_trials else "Tracking"
                    if e is None:
                        ti = self._item(bi, title, (TRACK, a, bone, trial), "none: double-click to load")
                        _grey(ti)
                        _grey(ti, 1)
                    elif e.tracking is None:
                        ti = self._item(bi, title, (TRACK, a, bone, trial), f"{e.path.name}: can't read")
                        _bold(ti, RED)
                        ti.setToolTip(1, e.error)
                    else:
                        t = e.tracking
                        extra = (f", from frame {t.offset + 1}" if t.offset else "") + \
                                (f", {t.n_untracked} untracked" if t.n_untracked else "")
                        ti = self._item(bi, title, (TRACK, a, bone, trial), f"{e.path.name} ({t.n_rows} rows{extra})")
                        _bold(ti, AMBER if t.warnings else GREEN)
                        ti.setToolTip(1, f"{e.path}\n{t.fmt.describe()}" + ("\n" + "\n".join(t.warnings)
                                                                            if t.warnings else ""))
                        n_ok += 1
                    if want is not None and self._key(self.data(ti)) == want:
                        to_select = ti
                if not many_trials:                 # one trial: the bone row shows its file directly
                    ti = bi.child(0)
                    bi.setText(1, ti.text(1))
                    bi.setToolTip(1, ti.toolTip(1))
                    bi.setForeground(1, ti.foreground(1))
                    bi.setFont(1, ti.font(1))
                else:
                    bi.setText(1, f"{n_ok} of {len(trials)} trials" if n_ok < len(trials) else
                               f"{len(trials)} trials")
                    if n_ok < len(trials):
                        bi.setForeground(1, QBrush(QColor(GREY)))
                bi.setToolTip(0, "Right-click to load or change its tracking, change its label or remove it. "
                                 "Expand for its tracking " + ("rows per trial" if many_trials else "row") + ".")
                if want is not None and self._key(self.data(bi)) == want:
                    to_select = bi

        ref = p.reference
        if ref is not None:
            r = self._item(self, "Reference", (REFERENCE, ref, None, None), "criterion")
            _bold(r, GREEN)
            r.setToolTip(0, "The criterion (gold standard) tracking, e.g. bead-based. Raters are compared with "
                            "it for validity.")
            assessor_items(r, ref)
            if want is not None and self._key(self.data(r)) == want:
                to_select = r
        raters = p.raters_list()
        top = self._item(self, "Assessors", (ASSESSORS, None, None, None),
                         f"{len(raters)} assessor{'s' * (len(raters) != 1)}" if raters else "right-click to add")
        _bold(top)
        for a in raters:
            ai = self._item(top, p.display_name(a), (ASSESSOR, a, None, None),
                            f"{len(a.bones)} bone{'s' * (len(a.bones) != 1)}")
            _bold(ai)
            assessor_items(ai, a)
            if want is not None and self._key(self.data(ai)) == want:
                to_select = ai

        # expand: remembered, or everything down to bones on first show
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            it = stack.pop()
            kind = self.data(it)[0]
            k = self._key(self.data(it))
            first_time = not self._expanded
            it.setExpanded(k in self._expanded or (first_time and kind in (BONES, REFERENCE, ASSESSORS, ASSESSOR)))
            stack += [it.child(i) for i in range(it.childCount())]
        if to_select is not None:
            self.setCurrentItem(to_select)
            self.scrollToItem(to_select)


class AssessorDialog(QDialog):
    """Add an assessor (name, session) or the reference, optionally starting with another one's bones."""

    def __init__(self, parent, p: A.AssessmentProject, reference: bool = False, name: str = "",
                 session: str = "1", copy_default: A.Assessor | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add reference" if reference else "Add assessor")
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        if reference:
            v.addWidget(label("The reference is the criterion (gold-standard) tracking that assessors are compared "
                              "with for validity, e.g. bead-based tracking.", "Muted"))
        form = QFormLayout()
        self.name = QLineEdit(name)
        self.session = QComboBox()
        self.session.setEditable(True)
        for s in A._unique(["1", "2", "3"] + p.sessions()):
            self.session.addItem(s)
        self.session.setCurrentText(session)
        self.session.setToolTip("Sessions are repeats by the same assessor (for intra-rater reliability)")
        if not reference:
            form.addRow("Name", self.name)
            form.addRow("Session", self.session)
        self.copy = QComboBox()
        self.copy.addItem("No bones yet", None)
        for a in p.assessors:
            if a.bones:
                self.copy.addItem(f"Same bones as {p.display_name(a)} ({', '.join(a.bones)})", a)
        if copy_default is not None:
            self.copy.setCurrentIndex(max(self.copy.findData(copy_default), 0))
        elif self.copy.count() > 1:
            self.copy.setCurrentIndex(self.copy.count() - 1)
        form.addRow("Bones", self.copy)
        v.addLayout(form)
        self.note = label("", "Warn")
        v.addWidget(self.note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.p, self.reference = p, reference
        self.name.textChanged.connect(self._check)
        self.session.currentTextChanged.connect(self._check)
        self._ok = bb.button(QDialogButtonBox.Ok)
        self._check()

    def _check(self, *_):
        if self.reference:
            return
        n, s = self.name.text().strip(), self.session.currentText().strip()
        clash = self.p.find_assessor(n, s) is not None
        self.note.setText(f"There is already an assessor {n}, session {s}." if clash else
                          ("" if n else "Give the assessor a name."))
        self._ok.setEnabled(bool(n) and bool(s) and not clash)

    def values(self):
        return self.name.text().strip(), self.session.currentText().strip(), self.copy.currentData()


class BoneDialog(QDialog):
    """Pick a bone label for an assessor: one already used elsewhere, or a new one."""

    def __init__(self, parent, p: A.AssessmentProject, a: A.Assessor, current: str = "", title: str = "Add bone",
                 ask_file: bool = True):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: {p.display_name(a)}")
        self.setMinimumWidth(380)
        v = QVBoxLayout(self)
        v.addWidget(label("Bones are matched across the reference and assessors by label, so pick the label the "
                          "others use for this bone, or type a new one.", "Muted"))
        self.combo = QComboBox()
        self.combo.setEditable(True)
        others = [b for b in p.bone_names() if b not in a.bones or b == current]
        for b in others:
            users = [p.display_name(o) for o in p.assessors if b in o.bones and o is not a]
            self.combo.addItem(b + (f"   (used by {', '.join(users)})" if users else ""), b)
        self.combo.setCurrentIndex(max(self.combo.findData(current), 0) if current else 0)
        if not others:
            self.combo.setCurrentText("")
        self.combo.lineEdit().setPlaceholderText("e.g. tib")
        f = QFormLayout()
        f.addRow("Bone", self.combo)
        v.addLayout(f)
        self.file_now = QCheckBox("Choose its tracking file now")
        self.file_now.setChecked(True)
        self.file_now.setVisible(ask_file)
        v.addWidget(self.file_now)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def value(self) -> str:
        text = self.combo.currentText().strip()
        i = self.combo.currentIndex()
        if i >= 0 and text == self.combo.itemText(i):
            return self.combo.itemData(i)
        return text.split("   (used by")[0].strip()


class MatchDialog(QDialog):
    """Several tracking files for one assessor: which bone (and trial) each one is."""

    def __init__(self, parent, p: A.AssessmentProject, a: A.Assessor, paths):
        super().__init__(parent)
        self.setWindowTitle(f"Tracking files: {p.display_name(a)}")
        self.setMinimumWidth(620)
        v = QVBoxLayout(self)
        v.addWidget(label("Each file's bone is guessed from its name (the labels this assessor and the others "
                          "use). Change any that are wrong; a new label adds a bone.", "Muted"))
        self.table = QTableWidget(len(paths), 3)
        self.table.setHorizontalHeaderLabels(["File", "Bone", "Trial"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        labels = A._unique(list(a.bones) + p.bone_names())
        trials = p.trials()
        self.paths = [Path(x) for x in paths]
        for r, path in enumerate(self.paths):
            it = QTableWidgetItem(path.name)
            it.setFlags(Qt.ItemIsEnabled)
            it.setToolTip(str(path))
            self.table.setItem(r, 0, it)
            bone = QComboBox()
            bone.setEditable(True)
            bone.addItems(labels)
            guess = A.guess_fields(path, None, labels).get("bone") or path.stem.split("_")[-1]
            bone.setCurrentText(guess)
            self.table.setCellWidget(r, 1, bone)
            trial = QComboBox()
            trial.setEditable(True)
            trial.addItems(trials)
            self.table.setCellWidget(r, 2, trial)
        v.addWidget(self.table)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def values(self) -> list[tuple]:
        return [(path, self.table.cellWidget(r, 1).currentText().strip(), self.table.cellWidget(r, 2).currentText()
                 .strip() or "Trial 1") for r, path in enumerate(self.paths)]
