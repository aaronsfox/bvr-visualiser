"""The Start page: one card per tool. Clicking an available card opens that tool in a new tab."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import __version__
from .panels import label
from .tools import TOOLS, ToolType


class ToolIcon(QWidget):
    """A small line drawing for each tool (no image files needed)."""

    def __init__(self, key: str, color: str, enabled: bool):
        super().__init__()
        self.key, self.color, self.enabled = key, QColor(color), enabled
        self.setFixedSize(44, 44)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self.color)
        if not self.enabled:
            c.setAlpha(110)
        bg = QColor(c)
        bg.setAlpha(36)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(0, 0, 44, 44), 10, 10)
        pen = QPen(c, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if self.key == "visualiser":            # a cube: 3D view
            p.drawPolygon([QPointF(22, 10), QPointF(33, 16), QPointF(33, 29), QPointF(22, 35),
                           QPointF(11, 29), QPointF(11, 16)])
            p.drawPolyline([QPointF(11, 16), QPointF(22, 22), QPointF(33, 16)])
            p.drawLine(QPointF(22, 22), QPointF(22, 35))
        elif self.key == "reliability":         # two overlapping traces: repeated tracking
            p.drawPolyline([QPointF(9, 30), QPointF(16, 20), QPointF(23, 25), QPointF(35, 12)])
            dashed = QPen(c, 2.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(dashed)
            p.drawPolyline([QPointF(9, 33), QPointF(16, 24), QPointF(23, 28), QPointF(35, 16)])
        elif self.key == "model_builder":       # two segments joined at a joint, with an axis
            p.drawLine(QPointF(10, 34), QPointF(20, 22))
            p.drawLine(QPointF(24, 22), QPointF(34, 10))
            p.drawEllipse(QPointF(22, 22), 3.2, 3.2)
            p.drawLine(QPointF(10, 34), QPointF(10, 27))
            p.drawLine(QPointF(10, 34), QPointF(17, 34))
        elif self.key == "validity":            # a target: agreement with a criterion
            p.drawEllipse(QPointF(22, 22), 12, 12)
            p.drawEllipse(QPointF(22, 22), 6, 6)
            p.setBrush(c)
            p.drawEllipse(QPointF(22, 22), 1.8, 1.8)
        else:                                   # anything added later
            p.drawEllipse(QPointF(22, 22), 9, 9)
        p.end()


class ToolCard(QFrame):
    clicked = Signal(str)

    def __init__(self, t: ToolType):
        super().__init__()
        self.tool = t
        self.setObjectName("ToolCard")
        self.setProperty("available", t.available)
        self.setFixedWidth(340)
        self.setMinimumHeight(310)
        if t.available:
            self.setCursor(Qt.PointingHandCursor)
            self.setFocusPolicy(Qt.StrongFocus)
            self.setToolTip(f"Open a new {t.title} tab")
        else:
            self.setToolTip("Coming soon")
        self.setAccessibleName(t.title + ("" if t.available else " (coming soon)"))
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 22, 22, 20)
        v.setSpacing(10)
        top = QHBoxLayout()
        top.addWidget(ToolIcon(t.key, t.color, t.available))
        top.addStretch(1)
        if not t.available:
            badge = QLabel("Coming soon")
            badge.setObjectName("SoonBadge")
            top.addWidget(badge, 0, Qt.AlignTop)
        v.addLayout(top)
        title = QLabel(t.title)
        title.setObjectName("CardTitle" if t.available else "CardTitleOff")
        v.addWidget(title)
        blurb = label(t.blurb, "CardBlurb" if t.available else "CardMuted")
        v.addWidget(blurb)
        for d in t.details:
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = QLabel("•")
            dot.setObjectName("CardMuted")
            dot.setStyleSheet(f"color: {t.color};" if t.available else "")
            row.addWidget(dot, 0, Qt.AlignTop)
            row.addWidget(label(d, "CardMuted"), 1)
            v.addLayout(row)
        v.addStretch(1)
        foot = QLabel(f"Open a {t.title} tab  →" if t.available else "Not available yet")
        foot.setObjectName("CardAction" if t.available else "CardMuted")
        if t.available:
            foot.setStyleSheet(f"color: {t.color};")
        v.addWidget(foot)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self.tool.available and e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.tool.key)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):  # noqa: N802
        if self.tool.available and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.tool.key)
        else:
            super().keyPressEvent(e)


class StartPage(QWidget):
    """Shown when the app opens, and whenever + New tab is clicked."""

    kind = "start"
    openTool = Signal(str)
    openSession = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self.state = None
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        box = QWidget()
        v = QVBoxLayout(box)
        v.setSpacing(8)
        title = QLabel("BVR Motion Viewer")
        title.setObjectName("Big")
        v.addWidget(title)
        v.addWidget(label("Choose a tool to start. Each opens in its own tab, and you can have several open at "
                          "once. Use + New tab to come back here.", wrap=True))
        v.addSpacing(18)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        self.cards: dict[str, ToolCard] = {}
        cols = 2 if len(TOOLS) == 4 else 3          # 2 × 2 fits smaller laptop screens
        for i, t in enumerate(TOOLS):
            card = ToolCard(t)
            card.clicked.connect(self.openTool.emit)
            self.cards[t.key] = card
            grid.addWidget(card, i // cols, i % cols)
        v.addLayout(grid)
        v.addSpacing(16)
        r = QHBoxLayout()
        op = QPushButton("Open a saved Visualiser session…")
        op.setObjectName("Small")
        op.clicked.connect(self.openSession.emit)
        r.addWidget(op)
        r.addStretch(1)
        r.addWidget(label(f"Version {__version__} · prototype", "Muted", wrap=False))
        v.addLayout(r)
        outer.addWidget(box, 0, Qt.AlignCenter)

    def close_tab(self) -> None:   # same interface as tool tabs
        pass
