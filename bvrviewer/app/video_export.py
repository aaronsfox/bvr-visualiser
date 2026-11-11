"""Video export from a Visualiser tab: the 3D view and/or the X-ray camera views, in a chosen layout
(Visualiser #15).

``ExportOptions`` holds the choices from ``VideoExportDialog``; ``tile_rects`` places the views on the
video frame; ``FrameComposer`` draws one video frame (3D render, camera views with the bones drawn
as on screen, captions and frame label) as an RGB array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

CONTENTS = (("3d", "3D view only"), ("both", "3D view and X-ray cameras"), ("cams", "X-ray cameras only"))
LAYOUTS = (("row", "Side by side"), ("main", "Main view large, the others stacked beside it"), ("grid", "Grid"))
SIZES = ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (1080, 1080))


@dataclass
class ExportOptions:
    content: str = "both"            # "3d", "both" or "cams"
    layout: str = "main"             # "row", "main" or "grid"
    width: int = 1920
    height: int = 1080
    first: int = 0                   # frames, 0-based, inclusive
    last: int = 0
    fps: int = 30
    frame_label: bool = True         # "Frame 12 / 300 · 0.367 s" in the corner
    captions: bool = True            # camera name and image on each camera view


def tile_rects(n: int, layout: str, width: int, height: int, gap: int = 6) -> list[QRect]:
    """Where each of ``n`` views goes on a ``width`` × ``height`` frame, the main view first.

    row: one row of equal columns. main: the first view on the left (60% of the width), the rest
    stacked on the right. grid: as square a grid as fits, filled row by row.
    """
    if n <= 0:
        return []
    if n == 1:
        return [QRect(0, 0, width, height)]
    if layout == "row":
        w = (width - gap * (n - 1)) / n
        return [QRect(round(i * (w + gap)), 0, round(w), height) for i in range(n)]
    if layout == "main":
        wm = round(width * 0.6)
        ws = width - wm - gap
        k = n - 1
        h = (height - gap * (k - 1)) / k
        return [QRect(0, 0, wm, height)] + [QRect(wm + gap, round(i * (h + gap)), ws, round(h)) for i in range(k)]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w = (width - gap * (cols - 1)) / cols
    h = (height - gap * (rows - 1)) / rows
    out = []
    for i in range(n):
        row, col = divmod(i, cols)
        in_row = min(cols, n - row * cols)
        x0 = (width - in_row * w - gap * (in_row - 1)) / 2        # a part-filled last row is centred
        out.append(QRect(round(x0 + col * (w + gap)), round(row * (h + gap)), round(w), round(h)))
    return out


def fit_into(img_w: int, img_h: int, rect: QRect) -> QRect:
    """The largest rectangle with the image's aspect ratio centred in ``rect``."""
    s = min(rect.width() / img_w, rect.height() / img_h)
    w, h = round(img_w * s), round(img_h * s)
    return QRect(rect.x() + (rect.width() - w) // 2, rect.y() + (rect.height() - h) // 2, w, h)


class VideoExportDialog(QDialog):
    """What goes in the video, its layout, size, frames and labels."""

    def __init__(self, parent, n_frames: int, n_cameras: int, start: ExportOptions | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        o = start or ExportOptions(last=n_frames - 1)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.content = QComboBox()
        for k, t in CONTENTS:
            self.content.addItem(t, k)
            if k != "3d" and n_cameras == 0:
                self.content.model().item(self.content.count() - 1).setEnabled(False)
        self.content.setCurrentIndex(self.content.findData(o.content if n_cameras else "3d"))
        if n_cameras == 0:
            self.content.setToolTip("Add X-ray cameras (with calibration and radiographs) to include them")
        form.addRow("Include", self.content)
        self.layout_box = QComboBox()
        for k, t in LAYOUTS:
            self.layout_box.addItem(t, k)
        self.layout_box.setCurrentIndex(self.layout_box.findData(o.layout))
        form.addRow("Layout", self.layout_box)
        self.size = QComboBox()
        for w, h in SIZES:
            self.size.addItem(f"{w} × {h}" + ("  (HD)" if (w, h) == (1920, 1080) else ""))
        self.size.setCurrentIndex(SIZES.index((o.width, o.height)) if (o.width, o.height) in SIZES else 1)
        form.addRow("Size", self.size)
        rng = QWidget()
        h = QHBoxLayout(rng)
        h.setContentsMargins(0, 0, 0, 0)
        self.first = QSpinBox()
        self.last = QSpinBox()
        for s in (self.first, self.last):
            s.setRange(1, max(1, n_frames))
        self.first.setValue(o.first + 1)
        self.last.setValue(min(o.last, n_frames - 1) + 1)
        h.addWidget(self.first)
        h.addWidget(QLabel("to"))
        h.addWidget(self.last)
        h.addWidget(QLabel(f"of {n_frames}"))
        h.addStretch(1)
        form.addRow("Frames", rng)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(o.fps)
        self.fps.setSuffix(" frames/s")
        form.addRow("Video speed", self.fps)
        self.frame_label = QCheckBox("Frame number and time in the corner")
        self.frame_label.setChecked(o.frame_label)
        form.addRow("Labels", self.frame_label)
        self.captions = QCheckBox("Camera name and image on each camera view")
        self.captions.setChecked(o.captions)
        form.addRow("", self.captions)
        v.addLayout(form)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(150)
        v.addWidget(self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export…")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.n_cameras = n_cameras
        for w in (self.content, self.layout_box, self.size):
            w.currentIndexChanged.connect(self._update)
        self.first.valueChanged.connect(lambda x: self.last.setValue(max(self.last.value(), x)))
        self.last.valueChanged.connect(lambda x: self.first.setValue(min(self.first.value(), x)))
        self._update()

    def options(self) -> ExportOptions:
        w, h = SIZES[self.size.currentIndex()]
        return ExportOptions(self.content.currentData(), self.layout_box.currentData(), w, h,
                             self.first.value() - 1, self.last.value() - 1, self.fps.value(),
                             self.frame_label.isChecked(), self.captions.isChecked())

    def _update(self) -> None:
        o = self.options()
        n = (o.content != "cams") + (self.n_cameras if o.content != "3d" else 0)
        self.layout_box.setEnabled(n > 1)
        self.captions.setEnabled(o.content != "3d")
        # a small sketch of the layout
        W, H = 240, round(240 * o.height / o.width)
        img = QImage(W, H, QImage.Format_RGB32)
        img.fill(QColor("#0B0D11"))
        p = QPainter(img)
        f = QFont()
        f.setPixelSize(11)
        p.setFont(f)
        names = (["3D"] if o.content != "cams" else []) + [f"Camera {i + 1}" for i in range(n - (o.content != "cams"))]
        for name, r in zip(names, tile_rects(n, o.layout, W, H, 3)):
            p.fillRect(r, QColor("#2A3140" if name == "3D" else "#3A3F4B"))
            p.setPen(QColor("#C3C8D2"))
            p.drawText(r, Qt.AlignCenter, name)
        p.end()
        from PySide6.QtGui import QPixmap
        self.preview.setPixmap(QPixmap.fromImage(img))


class FrameComposer:
    """Draws video frames for a Visualiser tab with the chosen options."""

    def __init__(self, tab, opts: ExportOptions):
        from .camera_views import CameraView

        self.tab, self.o = tab, opts
        st = tab.state
        self.bg = QColor(_background(st.settings.background))
        self.fg = QColor("#E6E8EC" if self.bg.lightness() < 128 else "#1C1F26")
        self.cams = ([c for c in st.cameras if c.calibration is not None and c.images is not None]
                     if opts.content != "3d" else [])
        n = (opts.content != "cams") + len(self.cams)
        self.rects = tile_rects(n, opts.layout, opts.width, opts.height)
        cam_rects = self.rects[1:] if opts.content != "cams" else self.rects
        on_screen = {id(v.cam): v for v in tab.cam_views.views} if tab.cam_views.isVisible() else {}
        self.views = []
        for cam, r in zip(self.cams, cam_rects):
            view = CameraView(cam)
            view.layout().setContentsMargins(0, 0, 0, 0)
            for w in (view.title, view.info):
                w.hide()
            view.gv.setBackground(self.bg)
            view.setAttribute(Qt.WA_DontShowOnScreen)
            view.resize(r.width(), r.height())
            view.show()
            shown = on_screen.get(id(cam))
            if shown is not None:          # keep the zoom and pan the camera view has on screen
                (x0, x1), (y0, y1) = shown.vb.viewRange()
                view.vb.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0)
                view._fitted = True
            self.views.append(view)

    def close(self) -> None:
        for v in self.views:
            v.close()
            v.deleteLater()
        self.views = []

    def frame(self, f: int) -> np.ndarray:
        """Video frame ``f`` (0-based tracking frame) as an H×W×3 uint8 array."""
        from PySide6.QtWidgets import QApplication

        tab, o = self.tab, self.o
        tab.scene.update_frame(f)
        canvas = QImage(o.width, o.height, QImage.Format_RGB888)
        canvas.fill(self.bg)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rects = list(self.rects)
        if o.content != "cams":
            r = rects.pop(0)
            shot = np.ascontiguousarray(tab.plotter.screenshot(return_img=True)[..., :3])
            img = QImage(shot.data, shot.shape[1], shot.shape[0], 3 * shot.shape[1], QImage.Format_RGB888)
            p.drawImage(fit_into(img.width(), img.height(), r), img)
        if self.views:
            cv = tab.cam_views
            args = cv.render_args()
            for view, r in zip(self.views, rects):
                view.update_view(f, *args)
                QApplication.processEvents()
                p.drawImage(r.topLeft(), view.gv.grab().toImage().scaled(
                    r.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                if o.captions:
                    i = view.cam.image_index(f)
                    text = view.cam.name + (f"  ·  {view.cam.images.label(i)}" if i is not None else "  ·  no image")
                    _label(p, _image_rect(view, r).adjusted(8, 6, -8, -6), text, Qt.AlignTop | Qt.AlignLeft,
                           o.height)
        if o.frame_label:
            fps = tab.state.settings.fps
            text = f"Frame {f + 1} / {tab.scene.n_frames}" + (f"  ·  {f / fps:.3f} s" if fps else "")
            corner = self.rects[0] if self.rects else QRect(0, 0, o.width, o.height)
            _label(p, corner.adjusted(8, 6, -8, -6), text, Qt.AlignBottom | Qt.AlignLeft, o.height)
        p.end()
        ptr = canvas.constBits()
        a = np.frombuffer(ptr, np.uint8, canvas.sizeInBytes()).reshape(o.height, canvas.bytesPerLine())
        return a[:, : 3 * o.width].reshape(o.height, o.width, 3).copy()


def _image_rect(view, tile: QRect) -> QRect:
    """Where the radiograph sits within ``tile`` (the view grabbed and scaled to it), clipped to the tile."""
    cal = view.cam.calibration
    poly = view.vb.mapViewToScene(QRectF(0, 0, cal.width, cal.height))
    br = view.gv.mapFromScene(poly).boundingRect()
    sx, sy = tile.width() / max(view.gv.width(), 1), tile.height() / max(view.gv.height(), 1)
    r = QRect(tile.x() + round(br.x() * sx), tile.y() + round(br.y() * sy), round(br.width() * sx),
              round(br.height() * sy))
    return r.intersected(tile) if r.intersects(tile) else tile


def _label(p: QPainter, rect: QRect, text: str, align, height: int) -> None:
    """White text on a dark translucent box, sized for the video height."""
    f = QFont()
    f.setPixelSize(max(12, round(height / 45)))
    f.setBold(True)
    p.setFont(f)
    br = p.boundingRect(rect, int(align), text).adjusted(-6, -3, 6, 3)
    p.fillRect(br, QColor(0, 0, 0, 150))
    p.setPen(QColor("#FFFFFF"))
    p.drawText(rect, int(align), text)


def _background(name: str) -> str:
    from .scene import BACKGROUNDS

    c = BACKGROUNDS.get(name, BACKGROUNDS["dark"])
    if isinstance(c, str):
        return c
    return QColor.fromRgbF(*c[:3]).name()
