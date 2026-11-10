"""Set-up widgets of the Assessor Evaluation tab: the factors, and the grid of tracking files to fill.

1. Factors: Bones (each with an optional surface and coordinate system), and short editable lists of the
   Reference (criterion; at most one), Assessors, Sessions and Trials.
2. Tracking files: for each trial, a grid with one row per assessor × session (and a Reference row) and
   one column per bone. Every cell is one tracking file; fill the gaps by double-clicking a cell or
   dropping a file on it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
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

    def __init__(self, title: str, tip: str, max_items: int | None = None):
        super().__init__()
        self.max_items = max_items
        self.title = title
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
        self._add_tip = self.add_btn.toolTip()
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
        full = self.max_items is not None and len(names) >= self.max_items
        self.add_btn.setEnabled(not full)
        self.add_btn.setToolTip(f"There can be only one {self.title.lower()}: rename or remove it" if full
                                else self._add_tip)

    def _menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        it = self.list.itemAt(pos)
        m = QMenu(self)
        m.addAction("Add", self.added.emit).setEnabled(self.add_btn.isEnabled())
        if it is not None:
            name = it.text()
            m.addAction("Rename…", lambda: self.renamed.emit(name))
            if self.max_items != 1:
                m.addAction("Move up", lambda: self.moved.emit(name, -1))
                m.addAction("Move down", lambda: self.moved.emit(name, 1))
            m.addSeparator()
            m.addAction("Remove", lambda: self.removed.emit(name))
        m.exec(self.list.viewport().mapToGlobal(pos))


class TrackingGrid(QTableWidget):
    """Rows: Reference and assessor × session; columns: bones. Each cell is one tracking file.

    A single file dropped on a cell fills it.
    """

    fileDropped = Signal(int, int, str)        # row, column, path

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
    @staticmethod
    def _one_file(ev) -> str | None:
        urls = ev.mimeData().urls() if ev.mimeData().hasUrls() else []
        return urls[0].toLocalFile() if len(urls) == 1 and urls[0].isLocalFile() else None

    def dragEnterEvent(self, ev):
        if self._one_file(ev):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        idx = self.indexAt(ev.position().toPoint())
        if self._one_file(ev) and idx.isValid():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        path = self._one_file(ev)
        idx = self.indexAt(ev.position().toPoint())
        if path and idx.isValid():
            self.fileDropped.emit(idx.row(), idx.column(), path)
            ev.acceptProposedAction()
        else:
            ev.ignore()
