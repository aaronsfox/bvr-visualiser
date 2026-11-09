"""Joint angle curves for the Visualiser, and the dialog that matches a joint model's bones to tab models."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from .panels import label

CURVE_COLOURS = ("#E5534B", "#57AB5A", "#539BF5")
AXIS_COLOURS = {"X": CURVE_COLOURS[0], "Y": CURVE_COLOURS[1], "Z": CURVE_COLOURS[2],    # as the 3D axes
                "1": CURVE_COLOURS[0], "2": CURVE_COLOURS[1], "3": CURVE_COLOURS[2]}


def curve_labels(k: dict, style: str = "axes"):
    """Order and labels of the three angles and three translations of kinematics k.

    style "axes": 'Angle X/Y/Z' and 'Translation X/Y/Z', listed X, Y, Z and coloured like the 3D axes.
    style "anatomical": the convention's own names (Flexion, ...) in its own order.
    Returns [(index, label, full name, colour)] for the angles and for the translations.
    """
    out = []
    for vals, short, full, axes, word in [(k["angles"], k["short"], k["names"], k.get("axes"), "Angle"),
                                          (k["translations"], k["t_short"], k["t_names"], k.get("t_axes"),
                                           "Translation")]:
        if style == "axes" and axes:
            order = sorted(range(3), key=lambda c: axes[c])
            out.append([(c, f"{word} {axes[c]}", full[c], AXIS_COLOURS.get(axes[c], CURVE_COLOURS[c]))
                        for c in order])
        else:
            out.append([(c, short[c], full[c], CURVE_COLOURS[c]) for c in range(3)])
    return out


# ---------------------------------------------------------------- matching bones to models

def _norm(p) -> str:
    try:
        return str(Path(p).resolve()).lower()
    except OSError:
        return str(p).lower()


def guess_segment_map(model, models) -> dict:
    """segment name -> tab model name, by surface file path first and then by name; unmatched are left out."""
    out = {}
    used = set()
    for seg in model.segments:
        if seg.surface is None or seg.acs is None:
            continue
        hit = next((m for m in models if id(m) not in used and _norm(m.path) == _norm(seg.surface.path)), None)
        if hit is None:
            stem = Path(seg.surface.path).stem.lower()
            hit = next((m for m in models if id(m) not in used and
                        (m.name.lower() == stem or Path(m.path).stem.lower() == stem)), None)
        if hit is None:
            key = seg.name.lower()
            hit = next((m for m in models if id(m) not in used and (key in m.name.lower() or seg.type in
                                                                       m.name.lower())), None)
        if hit is not None:
            out[seg.name] = hit.name
            used.add(id(hit))
    return out


NOT_HERE = "(not in this tab)"


class SegmentMapDialog(QDialog):
    """Choose which of the tab's models is each segment of a model, and what to show."""

    def __init__(self, parent, model, models, guess: dict, path: Path):
        super().__init__(parent)
        self.setWindowTitle("Match the model's bones to this tab")
        v = QVBoxLayout(self)
        v.addWidget(label(f"{path.name}: pick which bone in this tab is each of the model's bones. It must be the same surface file "
                          "(same model space) the bone was defined on. Bones left out aren't shown, nor are "
                          "joints that use them.", wrap=True))
        form = QFormLayout()
        self.combos = {}
        for seg in model.segments:
            cb = QComboBox()
            cb.addItem(NOT_HERE, None)
            for m in models:
                cb.addItem(m.name, m.name)
            if seg.acs is None:
                cb.setEnabled(False)
                cb.setToolTip("This bone has no axes in the model yet.")
            elif seg.name in guess:
                cb.setCurrentIndex(max(cb.findData(guess[seg.name]), 0))
            surf = f"  ({Path(seg.surface.path).name})" if seg.surface is not None else ""
            form.addRow(seg.name + surf, cb)
            self.combos[seg.name] = cb
        v.addLayout(form)
        self.warn = QLabel("")
        self.warn.setObjectName("Warn")
        v.addWidget(self.warn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _ok(self) -> None:
        names = [n for n in self.values().values()]
        if len(set(names)) < len(names):
            self.warn.setText("Choose a different bone in this tab for each.")
            return
        if not names:
            self.warn.setText("Match at least one bone.")
            return
        self.accept()

    def values(self) -> dict:
        return {k: cb.currentData() for k, cb in self.combos.items() if cb.currentData() is not None}


# ---------------------------------------------------------------- curves

class JointCurves(QWidget):
    """Angles and translations of one joint across the trial, with a cursor at the current frame."""

    frameClicked = Signal(int)
    labelStyleChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.setMinimumHeight(210)
        pg.setConfigOptions(antialias=True)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(4)
        head = QHBoxLayout()
        self.title = label("", "H2", wrap=False)
        head.addWidget(self.title, 1)
        self.readout = label("", "Mono", wrap=False)
        head.addWidget(self.readout)
        head.addSpacing(12)
        head.addWidget(QLabel("Labels"))
        self.label_style = QComboBox()
        self.label_style.addItem("Angle / Translation X Y Z", "axes")
        self.label_style.addItem("Anatomical names", "anatomical")
        self.label_style.setToolTip("Legend labels: the axis each value is about (X, Y, Z of the coordinate "
                                    "system), or the joint convention's anatomical names")
        self.label_style.currentIndexChanged.connect(self._style_changed)
        head.addWidget(self.label_style)
        v.addLayout(head)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.plots = []
        self.cursors = []
        for unit in ("deg", "mm"):
            pw = pg.PlotWidget(background="#14171C")
            pw.setMenuEnabled(False)
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.setLabel("left", "Angle (°)" if unit == "deg" else "Translation (mm)", color="#A3AAB8")
            pw.setLabel("bottom", "Frame", color="#A3AAB8")
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setTextPen("#A3AAB8")
                pw.getAxis(ax).setPen("#3A404C")
            pw.addLegend(offset=(6, 4), labelTextColor="#C3C8D2", brush=pg.mkBrush("#1C1F26CC"))
            cur = pg.InfiniteLine(angle=90, pen=pg.mkPen("#F0A04B", width=1.5))
            pw.addItem(cur)
            pw.scene().sigMouseClicked.connect(lambda ev, w=pw: self._clicked(ev, w))
            row.addWidget(pw, 1)
            self.plots.append(pw)
            self.cursors.append(cur)
        v.addLayout(row, 1)
        self.data = None
        self.title_text = ""
        self.frame = 0

    @property
    def style(self) -> str:
        return self.label_style.currentData()

    def set_style(self, style: str) -> None:
        i = self.label_style.findData(style)
        if i >= 0 and i != self.label_style.currentIndex():
            self.label_style.blockSignals(True)
            self.label_style.setCurrentIndex(i)
            self.label_style.blockSignals(False)
            self.set_data(self.data, self.title_text)

    def _style_changed(self, _i) -> None:
        self.set_data(self.data, self.title_text)
        self.labelStyleChanged.emit(self.style)

    def set_data(self, k: dict | None, title: str = "") -> None:
        """k: output of JointModel.joint_kinematics (after any zeroing); None clears."""
        self.data = k
        self.title_text = title
        self.title.setText(title)
        for pw, cur in zip(self.plots, self.cursors):
            for item in list(pw.listDataItems()):
                pw.removeItem(item)
            pw.plotItem.legend.clear()
        if k is None:
            self.readout.setText("")
            return
        x = np.arange(len(k["angles"]))
        la, lt = curve_labels(k, self.style)
        for pw, vals, labels, span in [(self.plots[0], k["angles"], la, 10.0),
                                       (self.plots[1], k["translations"], lt, 5.0)]:
            for c, name, _full, colour in labels:
                pw.plot(x, vals[:, c], pen=pg.mkPen(colour, width=2), name=name, connect="finite")
            pw.setXRange(0, max(len(x) - 1, 1), padding=0.02)
            fin = vals[np.isfinite(vals)]
            if fin.size:                    # at least `span` tall, so flat curves don't magnify round-off
                lo, hi = float(fin.min()), float(fin.max())
                mid, half = (lo + hi) / 2, max(hi - lo, span) / 2
                pw.setYRange(mid - half, mid + half, padding=0.08)
        self.set_frame(self.frame)

    def set_frame(self, f: int) -> None:
        self.frame = int(f)
        for cur in self.cursors:
            cur.setValue(self.frame)
        k = self.data
        if k is None or not len(k["angles"]):
            self.readout.setText("")
            return
        i = min(self.frame, len(k["angles"]) - 1)
        a, t = k["angles"][i], k["translations"][i]
        if np.isnan(a).any():
            self.readout.setText(f"Frame {self.frame + 1}: untracked")
        else:
            la, lt = curve_labels(k, self.style)
            parts = [f"{n} {(0.0 if abs(a[c]) < 0.05 else a[c]):+.1f}°" for c, n, _f, _col in la] + \
                    [f"{n} {(0.0 if abs(t[c]) < 0.05 else t[c]):+.1f} mm" for c, n, _f, _col in lt]
            self.readout.setText(f"Frame {self.frame + 1}:  " + "   ".join(parts))

    def _clicked(self, ev, pw) -> None:
        if self.data is None or ev.button() != Qt.LeftButton:
            return
        vb = pw.plotItem.vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = vb.mapSceneToView(ev.scenePos()).x()
        n = len(self.data["angles"])
        self.frameClicked.emit(int(np.clip(round(x), 0, n - 1)))


def zeroed(k: dict, frame: int | None) -> dict:
    """Subtract the values at a reference frame (e.g. a standing trial frame); NaN reference leaves k alone."""
    if frame is None or frame >= len(k["angles"]) or np.isnan(k["angles"][frame]).any():
        return k
    out = dict(k)
    out["angles"] = k["angles"] - k["angles"][frame]
    out["translations"] = k["translations"] - k["translations"][frame]
    return out


def write_csv(path, k: dict, fps: float | None = None, style: str = "anatomical") -> None:
    """Columns in the order and with the labels the curves use (full anatomical names kept in the header)."""
    la, lt = curve_labels(k, style)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["frame"] + (["time_s"] if fps else [])
        for labels, unit in ((la, "deg"), (lt, "mm")):
            head += [f"{n}: {full} ({unit})" if style == "axes" and n != full else f"{full} ({unit})"
                     for _c, n, full, _col in labels]
        w.writerow(head)
        cols = [c for c, *_ in la] + [3 + c for c, *_ in lt]
        for f, (a, t) in enumerate(zip(k["angles"], k["translations"])):
            row = [f + 1] + ([f"{f / fps:.5f}"] if fps else [])
            vals = (*a, *t)
            row += ["" if np.isnan(vals[c]) else f"{vals[c]:.4f}" for c in cols]
            w.writerow(row)
