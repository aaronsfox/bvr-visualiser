"""Dialogs: voxel size, tracking format, and adding an X-ray camera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from ..core.calibration import CalibrationError, load_calibration
from ..core.config import read_cfg
from ..core.images import ImageSequence, ImageSequenceError, find_images
from ..core.session import XrayCamera
from ..core.tracking import TrackingFormat
from .panels import label, small_button


class VoxelDialog(QDialog):
    """Voxel size (not stored in TIFs) and the surface threshold."""

    def __init__(self, parent, name: str, voxel=None, threshold=None, flip=(0, 0, 0)):
        super().__init__(parent)
        self.setWindowTitle(f"Volume settings · {name}")
        v = QVBoxLayout(self)
        v.addWidget(label("TIF files don't store voxel size. Enter it, or read it from the trial's .cfg "
                          "(the VoxelSize line)."))
        form = QFormLayout()
        row = QHBoxLayout()
        self.spins = []
        for val in (voxel or (0.39, 0.39, 0.625)):
            s = QDoubleSpinBox()
            s.setDecimals(4)
            s.setRange(0.001, 50)
            s.setSingleStep(0.01)
            s.setValue(val)
            s.setSuffix(" mm")
            self.spins.append(s)
            row.addWidget(s)
        form.addRow("Voxel size x, y, z", row)
        frow = QHBoxLayout()
        self.flips = []
        for ax, on in zip("XYZ", flip):
            cb = QCheckBox(ax)
            cb.setChecked(bool(on))
            self.flips.append(cb)
            frow.addWidget(cb)
        frow.addStretch(1)
        form.addRow("VolumeFlip", frow)
        trow = QHBoxLayout()
        self.auto = QCheckBox("Automatic")
        self.auto.setChecked(threshold is None)
        self.level = QDoubleSpinBox()
        self.level.setRange(-1e6, 1e6)
        self.level.setDecimals(1)
        self.level.setValue(threshold if threshold is not None else 700)
        self.level.setEnabled(threshold is not None)
        self.auto.toggled.connect(lambda on: self.level.setEnabled(not on))
        trow.addWidget(self.auto)
        trow.addWidget(self.level, 1)
        form.addRow("Surface threshold", trow)
        v.addLayout(form)
        v.addWidget(label("Automatic uses the edge of the masked partial volume (SlicerAutoscoperM output), "
                          "or an Otsu threshold for unmasked volumes."))
        b = small_button("Read voxel size from .cfg…")
        b.clicked.connect(lambda: self._from_cfg(name))
        v.addWidget(b, 0, Qt.AlignLeft)
        self.cfg_msg = label("")
        v.addWidget(self.cfg_msg)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _from_cfg(self, name: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Autoscoper trial configuration", "", "Config (*.cfg)")
        if not path:
            return
        cfg = read_cfg(path)
        vs = cfg.voxel_size_for(name)
        if vs is None and cfg.volumes and cfg.volumes[0].voxel_size:
            vs = cfg.volumes[0].voxel_size
            self.cfg_msg.setText(f"{name} isn't listed; used the first volume's voxel size.")
        if vs is None:
            self.cfg_msg.setText("No VoxelSize found in that file.")
            return
        for s, x in zip(self.spins, vs):
            s.setValue(x)
        for cb, f in zip(self.flips, cfg.flip_for(name)):
            cb.setChecked(bool(f))
        if not self.cfg_msg.text():
            self.cfg_msg.setText(f"Read from {Path(path).name}.")

    def values(self):
        voxel = tuple(s.value() for s in self.spins)
        flip = tuple(int(cb.isChecked()) for cb in self.flips)
        level = None if self.auto.isChecked() else self.level.value()
        return voxel, flip, level


class FormatDialog(QDialog):
    """Override what was detected in a .tra file."""

    def __init__(self, parent, fmt: TrackingFormat, volume_index: int):
        super().__init__(parent)
        self.setWindowTitle("Tracking file format")
        v = QVBoxLayout(self)
        v.addWidget(label("These were detected from the file. Change them if the bone moves oddly "
                          "(units can't be detected and default to Autoscoper's mm and degrees)."))
        form = QFormLayout()
        self.kind = self._combo([("matrix", "4×4 matrix"), ("xyzypr", "x y z yaw pitch roll")], fmt.kind)
        self.order = self._combo([("row", "Row order"), ("column", "Column order")], fmt.order)
        self.tunits = self._combo([("mm", "Millimetres"), ("cm", "Centimetres")], fmt.translation_units)
        self.runits = self._combo([("deg", "Degrees"), ("rad", "Radians")], fmt.rotation_units)
        self.volume = QComboBox()
        for i in range(fmt.n_volumes):
            self.volume.addItem(f"Volume {i + 1}", i)
        self.volume.setCurrentIndex(volume_index)
        form.addRow("Values", self.kind)
        form.addRow("Matrix order", self.order)
        form.addRow("Translation units", self.tunits)
        form.addRow("Rotation units", self.runits)
        form.addRow("Bone in file", self.volume)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    @staticmethod
    def _combo(items, current):
        c = QComboBox()
        for k, t in items:
            c.addItem(t, k)
        c.setCurrentIndex([k for k, _ in items].index(current))
        return c

    def values(self):
        overrides = {"kind": self.kind.currentData(), "order": self.order.currentData(),
                     "translation_units": self.tunits.currentData(), "rotation_units": self.runits.currentData()}
        return overrides, self.volume.currentData()


class AddCameraDialog(QDialog):
    """Pick a calibration file and radiographs, with a preview of the bones on the first image."""

    def __init__(self, parent, n_frames: int, preview_bones, existing: XrayCamera | None = None, index: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Edit X-ray camera" if existing else "Add X-ray camera")
        self.setMinimumWidth(720)
        self.n_frames = n_frames
        self.preview_bones = preview_bones        # list of (world points (N,3), colour)
        self.calibration = None
        self.images: ImageSequence | None = None
        self.source_kind = "folder"
        self.source: list[Path] = []
        self.result_camera: XrayCamera | None = None
        v = QVBoxLayout(self)
        v.setSpacing(10)

        self.name = QLineEdit(existing.name if existing else f"X-ray {index}")
        r = QHBoxLayout()
        r.addWidget(QLabel("Name"))
        r.addWidget(self.name, 1)
        v.addLayout(r)

        v.addWidget(self._step(1, "Calibration file"))
        r = QHBoxLayout()
        self.cal_path = QLineEdit()
        self.cal_path.setReadOnly(True)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_cal)
        r.addWidget(self.cal_path, 1)
        r.addWidget(b)
        v.addLayout(r)
        self.cal_msg = label("")
        v.addWidget(self.cal_msg)

        v.addWidget(self._step(2, "Radiograph images"))
        r = QHBoxLayout()
        grp = QButtonGroup(self)
        self.kind_buttons = {}
        for k, t in [("folder", "A folder of images"), ("files", "Selected files"),
                     ("multipage", "One multi-page TIFF")]:
            rb = QRadioButton(t)
            rb.setChecked(k == "folder")
            rb.toggled.connect(lambda on, kk=k: on and setattr(self, "source_kind", kk))
            grp.addButton(rb)
            self.kind_buttons[k] = rb
            r.addWidget(rb)
        r.addStretch(1)
        v.addLayout(r)
        r = QHBoxLayout()
        self.src_path = QLineEdit()
        self.src_path.setReadOnly(True)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_images)
        r.addWidget(self.src_path, 1)
        r.addWidget(b)
        v.addLayout(r)
        box = QHBoxLayout()
        self.preview = QLabel()
        self.preview.setFixedSize(200, 200)
        self.preview.setStyleSheet("background: #0B0D11; border-radius: 6px;")
        self.preview.setAlignment(Qt.AlignCenter)
        box.addWidget(self.preview)
        info = QVBoxLayout()
        self.img_msg = label("")
        self.img_detail = label("", "Mono")
        self.img_note = label("Preview shows the tracked bones on the image for frame 1. If they don't sit on "
                              "their shadows, check the calibration file and the frame matching below.")
        info.addWidget(self.img_msg)
        info.addWidget(self.img_detail)
        info.addWidget(self.img_note)
        info.addStretch(1)
        box.addLayout(info, 1)
        v.addLayout(box)

        v.addWidget(self._step(3, "Match images to tracking frames"))
        r = QHBoxLayout()
        r.addWidget(QLabel("Tracking frame 1 uses image"))
        self.first = QComboBox()
        self.first.currentIndexChanged.connect(self._update_match)
        r.addWidget(self.first)
        r.addStretch(1)
        v.addLayout(r)
        self.match_msg = label("")
        v.addWidget(self.match_msg)

        self.error = label("", "Error")
        v.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setText("Save camera" if existing else "Add camera")
        self.ok.setObjectName("Primary")
        for btn in bb.buttons():
            btn.setIcon(QIcon())
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        if existing:
            self._set_calibration(existing.calibration_path)
            self.kind_buttons[existing.source_kind].setChecked(True)
            self._set_images(existing.source_kind, existing.source)
            self.first.setCurrentIndex(existing.first_image)
        self._refresh_ok()

    @staticmethod
    def _step(n: int, text: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 6, 0, 0)
        num = QLabel(str(n))
        num.setObjectName("Step")
        num.setFixedSize(28, 28)
        num.setAlignment(Qt.AlignCenter)
        t = QLabel(text)
        t.setStyleSheet("font-weight: 600; font-size: 14px;")
        h.addWidget(num)
        h.addWidget(t, 1)
        return w

    def _browse_cal(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Calibration file", "",
                                              "Calibration (*.txt *.csv *.json);;All files (*)")
        if path:
            self._set_calibration(Path(path))

    def _set_calibration(self, path: Path) -> None:
        self.cal_path.setText(str(path))
        try:
            self.calibration = load_calibration(path)
            self.cal_msg.setObjectName("Ok")
            self.cal_msg.setText("✓ " + self.calibration.summary())
        except (CalibrationError, OSError) as e:
            self.calibration = None
            self.cal_msg.setObjectName("Error")
            self.cal_msg.setText(str(e))
        self.cal_msg.style().polish(self.cal_msg)
        self._check_sizes()
        self._draw_preview()
        self._refresh_ok()

    def _browse_images(self) -> None:
        k = self.source_kind
        if k == "folder":
            p = QFileDialog.getExistingDirectory(self, "Folder of radiographs")
            paths = [Path(p)] if p else []
        elif k == "files":
            ps, _ = QFileDialog.getOpenFileNames(self, "Radiograph images", "",
                                                 "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
            paths = [Path(p) for p in ps]
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Multi-page TIFF", "", "TIFF (*.tif *.tiff)")
            paths = [Path(p)] if p else []
        if paths:
            self._set_images(k, paths)

    def _set_images(self, kind: str, paths: list[Path]) -> None:
        self.source_kind, self.source = kind, paths
        self.src_path.setText(str(paths[0]) if len(paths) == 1 else f"{len(paths)} files in {paths[0].parent}")
        skipped = []
        try:
            if kind == "folder":
                files, skipped = find_images(paths[0])
                self.images = ImageSequence(files=files)
            elif kind == "multipage":
                self.images = ImageSequence(multipage=paths[0])
            else:
                self.images = ImageSequence(files=sorted(paths))
        except (ImageSequenceError, OSError, ValueError) as e:
            self.images = None
            self.img_msg.setObjectName("Error")
            self.img_msg.setText(str(e))
            self.img_msg.style().polish(self.img_msg)
            self._refresh_ok()
            return
        n = len(self.images)
        self.img_msg.setObjectName("Ok")
        self.img_msg.setText(f"✓ {n} images found, ordered by the number in each file name")
        self.img_msg.style().polish(self.img_msg)
        detail = f"{self.images.name(0)} → {self.images.name(n - 1)}\n{self.images.describe()}"
        if skipped:
            detail += f"\nIgnored: {', '.join(skipped[:4])}" + (" …" if len(skipped) > 4 else "")
        self.img_detail.setText(detail)
        self.first.blockSignals(True)
        self.first.clear()
        for i in range(n):
            self.first.addItem(self.images.label(i) + ("  (first image)" if i == 0 else ""), i)
        self.first.blockSignals(False)
        self._check_sizes()
        self._update_match()
        self._refresh_ok()

    def _check_sizes(self) -> None:
        if self.images is None or self.calibration is None:
            return
        if (self.images.width, self.images.height) != (self.calibration.width, self.calibration.height):
            self.img_detail.setText(self.img_detail.text().split("\n⚠")[0] +
                                    f"\n⚠ Images are {self.images.width} × {self.images.height} but the "
                                    f"calibration expects {self.calibration.width} × {self.calibration.height}.")

    def _update_match(self) -> None:
        if self.images is None:
            return
        first = self.first.currentData() or 0
        shown = len(self.images) - first
        if shown == self.n_frames:
            self.match_msg.setObjectName("Ok")
            self.match_msg.setText(f"✓ {shown} images match the {self.n_frames} tracked frames")
        elif shown > self.n_frames:
            self.match_msg.setObjectName("Warn")
            self.match_msg.setText(f"{shown} images from here but {self.n_frames} tracked frames: "
                                   f"the last {shown - self.n_frames} images won't be shown.")
        else:
            self.match_msg.setObjectName("Warn")
            self.match_msg.setText(f"Only {shown} images from here for {self.n_frames} tracked frames: "
                                   f"the last {self.n_frames - shown} frames will have no radiograph.")
        self.match_msg.style().polish(self.match_msg)
        self._draw_preview()

    def _draw_preview(self) -> None:
        if self.images is None:
            return
        idx = self.first.currentData() or 0
        img = np.ascontiguousarray(self.images.get(idx))
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).convertToFormat(QImage.Format_RGB32)
        if self.calibration is not None and self.preview_bones:
            painter = QPainter(qimg)
            dot = max(3, w // 180)            # stays visible once scaled down to the thumbnail
            for pts, color in self.preview_bones:
                uv = self.calibration.project(pts)
                c = QColor(color)
                c.setAlpha(140)
                painter.setPen(Qt.NoPen)
                painter.setBrush(c)
                for u, vv in uv[:: max(1, len(uv) // 1500)]:
                    if 0 <= u < w and 0 <= vv < h:
                        painter.drawEllipse(int(u) - dot // 2, int(vv) - dot // 2, dot, dot)
            painter.end()
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_ok(self) -> None:
        self.ok.setEnabled(self.calibration is not None and self.images is not None)

    def _accept(self) -> None:
        cam = XrayCamera(calibration_path=Path(self.cal_path.text()), source_kind=self.source_kind,
                         source=self.source, first_image=self.first.currentData() or 0,
                         name=self.name.text().strip() or "X-ray")
        cam.calibration, cam.images = self.calibration, self.images
        self.result_camera = cam
        self.accept()
