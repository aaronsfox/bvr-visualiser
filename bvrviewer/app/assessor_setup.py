"""Set-up widgets of the Assessor Evaluation tab: the factors, and the grid of tracking files to fill.

1. Factors: Bones (each with an optional surface and coordinate system), Assessors, Sessions and Trials
   (short editable lists), and whether there is a Reference (criterion).
2. Tracking files: for each trial, a grid with one row per assessor × session (and a Reference row) and
   one column per bone. Every cell is one tracking file; fill the gaps by double-clicking a cell, dropping
   a file on it, or filling many at once from files whose names say which cell they belong to.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import assessment as A
from .panels import label, small_button

GREEN, TEAL, AMBER, GREY, RED = "#8FD3A9", "#5CC8A8", "#F2C48D", "#5E6573", "#F0A3A3"
EMPTY_BG = "#1A1D24"


class FactorList(QWidget):
    """A titled, short list of names (assessors, sessions or trials) with add / rename / remove / move."""

    added = Signal()
    renamed = Signal(str)               # the old name (the new one is asked by the tab)
    removed = Signal(str)
    moved = Signal(str, int)

    def __init__(self, title: str, tip: str):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        head = QHBoxLayout()
        t = label(title, "PropLabel", wrap=False)
        t.setToolTip(tip)
        head.addWidget(t)
        head.addStretch(1)
        self.add_btn = small_button("+")
        self.add_btn.setToolTip(f"Add {title.lower().rstrip('s')}")
        self.add_btn.setFixedWidth(28)
        self.add_btn.clicked.connect(self.added.emit)
        head.addWidget(self.add_btn)
        v.addLayout(head)
        self.list = QListWidget()
        self.list.setToolTip(tip + "\nDouble-click to rename; right-click for more.")
        self.list.setMinimumHeight(70)
        self.list.setMaximumHeight(120)
        self.list.itemDoubleClicked.connect(lambda it: self.renamed.emit(it.text()))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        v.addWidget(self.list)

    def set_items(self, names) -> None:
        cur = self.list.currentItem().text() if self.list.currentItem() else None
        self.list.clear()
        for n in names:
            it = QListWidgetItem(n)
            self.list.addItem(it)
            if n == cur:
                self.list.setCurrentItem(it)

    def _menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        it = self.list.itemAt(pos)
        m = QMenu(self)
        m.addAction("Add", self.added.emit)
        if it is not None:
            name = it.text()
            m.addAction("Rename…", lambda: self.renamed.emit(name))
            m.addAction("Move up", lambda: self.moved.emit(name, -1))
            m.addAction("Move down", lambda: self.moved.emit(name, 1))
            m.addSeparator()
            m.addAction("Remove", lambda: self.removed.emit(name))
        m.exec(self.list.viewport().mapToGlobal(pos))


class TrackingGrid(QTableWidget):
    """Rows: Reference and assessor × session; columns: bones. Each cell is one tracking file.

    Files dropped on a cell fill it (several files dropped at once go to the bulk-fill dialog).
    """

    filesDropped = Signal(int, int, list)      # row, column, paths

    def __init__(self):
        super().__init__(0, 0)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setMinimumSectionSize(90)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideMiddle)      # keep the end of the name (…_s2_tal.tra) visible
        self.setMinimumHeight(150)
        self.slots: list[tuple] = []       # row -> (who, session)
        self.bones: list[str] = []         # column -> bone
        self.trial = ""

    def fill(self, p: A.AssessmentProject, trial: str) -> None:
        self.trial = trial
        self.slots, self.bones = p.rows(), p.bone_names()
        self.setRowCount(len(self.slots))
        self.setColumnCount(len(self.bones))
        self.setHorizontalHeaderLabels(self.bones)
        self.setVerticalHeaderLabels([p.row_label(w, s) for w, s in self.slots])
        if p.has_reference and self.slots:
            h = self.verticalHeaderItem(0)
            f = QFont()
            f.setBold(True)
            h.setFont(f)
            h.setForeground(QBrush(QColor(GREEN)))
            h.setToolTip("The reference (criterion) tracking, e.g. bead-based")
        for r, (who, ses) in enumerate(self.slots):
            for c, bone in enumerate(self.bones):
                e = p.entry(trial, bone, who, ses)
                it = QTableWidgetItem()
                it.setTextAlignment(Qt.AlignCenter)
                where = f"{p.row_label(who, ses)} · {bone}" + (f" · {trial}" if len(p.trials()) > 1 else "")
                if e is None:
                    it.setText("+")
                    it.setForeground(QBrush(QColor(GREY)))
                    it.setBackground(QBrush(QColor(EMPTY_BG)))
                    it.setToolTip(f"{where}\nNo file yet: double-click to choose one, or drop a file here")
                elif e.tracking is None:
                    it.setText(f"⚠ {e.path.name}")
                    it.setForeground(QBrush(QColor(RED)))
                    it.setToolTip(f"{where}\n{e.path}\n{e.error}")
                else:
                    t = e.tracking
                    it.setText(e.path.name)
                    it.setForeground(QBrush(QColor(AMBER if t.warnings else GREEN)))
                    extra = (f", first row = frame {t.offset + 1}" if t.offset else "") + \
                            (f", {t.n_untracked} untracked" if t.n_untracked else "")
                    it.setToolTip(f"{where}\n{e.path}\n{t.n_rows} rows{extra}\n{t.fmt.describe()}"
                                  + ("\n" + "\n".join(t.warnings) if t.warnings else "")
                                  + "\nDouble-click to change the format or frames; right-click for more")
                self.setItem(r, c, it)

    # ---- drops
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):
        if not ev.mimeData().hasUrls():
            return super().dropEvent(ev)
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        idx = self.indexAt(ev.position().toPoint())
        self.filesDropped.emit(idx.row(), idx.column(), paths)
        ev.acceptProposedAction()


class FillDialog(QDialog):
    """Fill many cells at once: each file is matched to a cell from its name; check and correct."""

    def __init__(self, parent, p: A.AssessmentProject, paths, root: Path | None = None, pattern: str = "",
                 trial: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Fill from files")
        self.setMinimumWidth(820)
        self.setMinimumHeight(420)
        self.p = p
        v = QVBoxLayout(self)
        v.addWidget(label("Each file's cell is worked out from the names in its folder and file name (bones, "
                          "assessors, trials, a session like s2 or session_2, and reference/criterion/gold/bead "
                          "for the reference). Check them, correct any that are wrong, and leave a file's bone "
                          "empty to skip it. A name that isn't in the lists adds it.", "Muted"))
        self.table = QTableWidget(len(paths), 5)
        self.table.setHorizontalHeaderLabels(["File", "Assessor", "Session", "Bone", "Trial"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.paths = [Path(x) for x in paths]
        who_items = ["Reference"] + p.raters()
        for r, path in enumerate(self.paths):
            m = A.match_file(path, p, root, pattern)
            it = QTableWidgetItem(path.name)
            it.setFlags(Qt.ItemIsEnabled)
            it.setToolTip(str(path))
            self.table.setItem(r, 0, it)
            who = m.get("who")
            cells = [(1, who_items, "Reference" if who == A.CRITERION else (who or "")),
                     (2, p.sessions(), m.get("session", "")),
                     (3, p.bone_names(), m.get("bone", "")),
                     (4, p.trials(), m.get("trial", trial or ""))]
            for col, items, value in cells:
                cb = QComboBox()
                cb.setEditable(True)
                cb.addItems(items)
                cb.setCurrentText(value)
                cb.currentTextChanged.connect(self._check)
                self.table.setCellWidget(r, col, cb)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        self.note = label("", "Warn")
        v.addWidget(self.note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._check()

    def _row(self, r: int):
        get = lambda c: self.table.cellWidget(r, c).currentText().strip()
        who = get(1)
        who = A.CRITERION if who == "Reference" else who
        return self.paths[r], get(4), get(3), who, ("" if who == A.CRITERION else get(2))

    def _check(self, *_):
        seen, problems = {}, []
        skipped = 0
        for r in range(len(self.paths)):
            path, trial, bone, who, ses = self._row(r)
            if not bone:
                skipped += 1
                continue
            if not who or (who != A.CRITERION and not ses) or not trial:
                problems.append(f"{path.name}: choose the assessor, session and trial")
                continue
            key = (trial, bone, who, ses)
            if key in seen:
                problems.append(f"{path.name} and {seen[key]} are for the same cell")
            seen[key] = path.name
            if self.p.entry(trial, bone, who, ses) is not None:
                problems.append(f"{path.name} replaces the file already in "
                                f"{self.p.row_label(who, ses)} · {bone}")
        if skipped:
            problems.append(f"{skipped} file{'s' * (skipped != 1)} without a bone will be skipped")
        self.note.setText("\n".join(problems[:6]) + (f"\n… and {len(problems) - 6} more" if len(problems) > 6 else ""))

    def values(self) -> list[tuple]:
        """(path, trial, bone, who, session) for each file to load (files without a bone are skipped)."""
        out = []
        for r in range(len(self.paths)):
            path, trial, bone, who, ses = self._row(r)
            if bone and who and trial and (who == A.CRITERION or ses):
                out.append((path, trial, bone, who, ses))
        return out
