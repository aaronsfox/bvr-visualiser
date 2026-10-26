"""Shared widgets: small helpers, the colour button and the timeline."""

from __future__ import annotations


import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from ..core.session import SceneSettings


def h2(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("H2")
    return lab


def label(text: str, kind: str = "Muted", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    return lab


def small_button(text: str, kind: str = "Small") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    return b


class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Swatch")
        self.setFixedSize(26, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Change colour")
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color: str) -> None:
        self.color = color
        self.setStyleSheet(f"QPushButton#Swatch {{ background: {color}; }}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self.color), self, "Bone colour")
        if c.isValid():
            self.set_color(c.name())
            self.colorChanged.emit(c.name())


# ---------------------------------------------------------------- timeline

class CoverageStrip(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(6)
        self.valid: np.ndarray | None = None
        self.n = 1

    def set_data(self, valid: np.ndarray | None, n: int) -> None:
        self.valid, self.n = valid, max(n, 1)
        self.setToolTip("Tracked (green) and untracked (orange) frames for the selected bone"
                        if valid is not None else "")
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#262A33"))
        if self.valid is None:
            return
        n = self.n
        for i in range(n):
            ok = i < len(self.valid) and self.valid[i]
            x0, x1 = int(w * i / n), int(w * (i + 1) / n)
            p.fillRect(x0, 0, max(x1 - x0, 1), h, QColor("#3B6E52" if ok else "#D9822B"))


class Timeline(QWidget):
    frameChanged = Signal(int)

    def __init__(self, settings: SceneSettings):
        super().__init__()
        self.setObjectName("Panel")
        self.s = settings
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        r = QHBoxLayout()
        self.first = QPushButton("|‹")
        self.prev = QPushButton("‹")
        self.play = QPushButton("▶")
        self.play.setObjectName("Primary")
        self.play.setCheckable(True)
        self.next = QPushButton("›")
        self.last = QPushButton("›|")
        for b, tip in [(self.first, "First frame"), (self.prev, "Previous frame (←)"), (self.play, "Play / pause (space)"),
                       (self.next, "Next frame (→)"), (self.last, "Last frame")]:
            b.setFixedSize(42, 38)
            b.setToolTip(tip)
            b.setStyleSheet("font-size: 17px; font-weight: 600;" if b is not self.play else "")
            r.addWidget(b)
        mid = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.coverage = CoverageStrip()
        mid.addWidget(self.slider)
        mid.addWidget(self.coverage)
        r.addLayout(mid, 1)
        self.counter = label("Frame 1 / 1", "Mono", wrap=False)
        self.counter.setMinimumWidth(110)
        r.addWidget(self.counter)
        r.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for x in (0.1, 0.25, 0.5, 1.0, 2.0):
            self.speed.addItem(f"{x:g}×", x)
        self.speed.setCurrentIndex(3)
        self.speed.currentIndexChanged.connect(lambda _: setattr(self.s, "speed", self.speed.currentData()))
        r.addWidget(self.speed)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(settings.loop)
        self.loop.toggled.connect(lambda on: setattr(self.s, "loop", on))
        r.addWidget(self.loop)
        v.addLayout(r)
        self.slider.valueChanged.connect(self._changed)
        self.first.clicked.connect(lambda: self.slider.setValue(0))
        self.last.clicked.connect(lambda: self.slider.setValue(self.slider.maximum()))
        self.prev.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))

    def _changed(self, f: int) -> None:
        self.counter.setText(f"Frame {f + 1} / {self.slider.maximum() + 1}")
        self.frameChanged.emit(f)

    def set_count(self, n: int) -> None:
        f = min(self.slider.value(), n - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(f)
        self.slider.blockSignals(False)
        self.counter.setText(f"Frame {f + 1} / {n}")

    def step(self, d: int) -> bool:
        """Advance; returns False when it stops at the end."""
        f = self.slider.value() + d
        n = self.slider.maximum()
        if f > n or f < 0:
            if not self.s.loop:
                return False
            f = f % (n + 1)
        self.slider.setValue(f)
        return True
