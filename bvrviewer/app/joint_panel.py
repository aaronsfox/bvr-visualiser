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

from ..core.session import JointLink
from .panels import label

CURVE_COLOURS = ("#E5534B", "#57AB5A", "#539BF5")


# ---------------------------------------------------------------- matching bones to models

def _norm(p) -> str:
    try:
        return str(Path(p).resolve()).lower()
    except OSError:
        return str(p).lower()


def guess_role_map(joint_model, models) -> dict:
    """role -> tab model name, by surface file path first and then by name; missing roles are left out."""
    out = {}
    used = set()
    for role in joint_model.template.bones:
        e = joint_model.bones[role.key]
        if e.surface is None:
            continue
        hit = next((m for m in models if id(m) not in used and _norm(m.path) == _norm(e.surface.path)), None)
        if hit is None:
            stem = Path(e.surface.path).stem.lower()
            hit = next((m for m in models if id(m) not in used and
                        (m.name.lower() == stem or Path(m.path).stem.lower() == stem)), None)
        if hit is None:
            hit = next((m for m in models if id(m) not in used and role.key in m.name.lower()), None)
        if hit is not None:
            out[role.key] = hit.name
            used.add(id(hit))
    return out


class RoleMapDialog(QDialog):
    """Choose which of the tab's models is each bone of the joint."""

    def __init__(self, parent, link: JointLink, models):
        super().__init__(parent)
        self.setWindowTitle("Match the joint's bones")
        self.link = link
        v = QVBoxLayout(self)
        j = link.joint()
        v.addWidget(label(f"{j.title} from {link.model_path.name}. Pick the tab model for each bone. It must be the "
                          "same surface file (same model space) the joint model was built on.", wrap=True))
        form = QFormLayout()
        self.combos = {}
        jm = link.joint_model
        for role_key in (j.proximal, j.distal):
            role = jm.template.role(role_key)
            cb = QComboBox()
            for m in models:
                cb.addItem(m.name, m.name)
            cur = link.role_map.get(role_key)
            if cur is not None:
                cb.setCurrentIndex(max(cb.findData(cur), 0))
            surf = jm.bones[role_key].surface
            form.addRow(f"{role.title}" + (f"  ({Path(surf.path).name})" if surf is not None else ""), cb)
            self.combos[role_key] = cb
        v.addLayout(form)
        self.warn = QLabel("")
        self.warn.setObjectName("Warn")
        v.addWidget(self.warn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _ok(self) -> None:
        names = [cb.currentData() for cb in self.combos.values()]
        if len(set(names)) < len(names):
            self.warn.setText("Choose a different model for each bone.")
            return
        self.accept()

    def values(self) -> dict:
        return {k: cb.currentData() for k, cb in self.combos.items()}


# ---------------------------------------------------------------- curves

class JointCurves(QWidget):
    """Angles and translations of one joint across the trial, with a cursor at the current frame."""

    frameClicked = Signal(int)

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
        self.frame = 0

    def set_data(self, k: dict | None, title: str = "") -> None:
        """k: output of JointModel.joint_kinematics (after any zeroing); None clears."""
        self.data = k
        self.title.setText(title)
        for pw, cur in zip(self.plots, self.cursors):
            for item in list(pw.listDataItems()):
                pw.removeItem(item)
            pw.plotItem.legend.clear()
        if k is None:
            self.readout.setText("")
            return
        x = np.arange(len(k["angles"]))
        for pw, vals, names, span in [(self.plots[0], k["angles"], k["short"], 10.0),
                                      (self.plots[1], k["translations"], k["t_short"], 5.0)]:
            for c in range(3):
                pw.plot(x, vals[:, c], pen=pg.mkPen(CURVE_COLOURS[c], width=2), name=names[c], connect="finite")
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
            parts = [f"{n} {(0.0 if abs(v) < 0.05 else v):+.1f}°" for n, v in zip(k["short"], a)] + \
                    [f"{n} {(0.0 if abs(v) < 0.05 else v):+.1f} mm" for n, v in zip(k["t_short"], t)]
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


def write_csv(path, k: dict, fps: float | None = None) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["frame"] + (["time_s"] if fps else [])
        head += [f"{n} (deg)" for n in k["names"]] + [f"{n} (mm)" for n in k["t_names"]]
        w.writerow(head)
        for f, (a, t) in enumerate(zip(k["angles"], k["translations"])):
            row = [f + 1] + ([f"{f / fps:.5f}"] if fps else [])
            row += ["" if np.isnan(v) else f"{v:.4f}" for v in (*a, *t)]
            w.writerow(row)
