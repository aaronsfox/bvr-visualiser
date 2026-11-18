"""2D camera panels: each X-ray camera's radiograph with the tracked bones projected onto it.

Shown beside the 3D view (one panel per camera, stacked). These are plain 2D image views, not 3D
windows: each bone's surface is projected through the camera calibration (pixel = K·(R·X + t)/z) at
the current frame, and drawn over the image in the bone's colour, either as a point cloud of its
surface or as a silhouette (closest to Autoscoper's overlay). The point cloud can be thinned evenly.

The bones are placed with their raw tracking (the Autoscoper world frame the cameras are calibrated
in), so "hold this bone still" in the 3D view doesn't affect these panels.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QHBoxLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.models import BoneModel
from ..core.session import XrayCamera
from .panels import label

MAX_POINTS = 4000           # surface points drawn per bone in point-cloud mode
MAX_TRIANGLES = 6000        # triangles projected per bone in silhouette mode

OVERLAYS = [("drr", "DRR (TIF bones)"), ("silhouette", "Silhouette"), ("points", "Point cloud"), ("off", "Off")]


def overlays_for(models) -> list:
    """The overlay choices for these bones: DRR only when some are TIF volumes (Visualiser #25)."""
    has_drr = any(getattr(m, "drr", None) is not None for m in models)
    return [o for o in OVERLAYS if o[0] != "drr" or has_drr]


def even_order(pts: np.ndarray) -> np.ndarray:
    """An order of the points in which any first share of them is spread evenly over the surface (Visualiser #24).

    Farthest-point order: each next point is the one farthest from those already chosen, so showing the
    first 10% thins the cloud evenly rather than leaving patches. Done once per bone (a fraction of a second for
    4000 points).
    """
    n = len(pts)
    if n < 3:
        return np.arange(n)
    pts = np.asarray(pts, float)
    order = np.empty(n, dtype=np.int64)
    order[0] = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    x, y, z = (np.ascontiguousarray(pts[:, c]) for c in range(3))
    dist = (x - x[order[0]]) ** 2 + (y - y[order[0]]) ** 2 + (z - z[order[0]]) ** 2      # squared: same order
    d = np.empty(n)
    for k in range(1, n):
        i = int(dist.argmax())
        order[k] = i
        np.subtract(x, x[i], out=d)
        d *= d
        dist2 = d.copy()
        np.subtract(y, y[i], out=d)
        d *= d
        dist2 += d
        np.subtract(z, z[i], out=d)
        d *= d
        dist2 += d
        np.minimum(dist, dist2, out=dist)
    return order


class _BoneShape:
    """A bone's surface reduced for fast projection, cached for as long as its mesh doesn't change."""

    def __init__(self, m: BoneModel):
        self.mesh_id = id(m.mesh)
        pts = np.asarray(m.mesh.points, float)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), MAX_POINTS, replace=False) if len(pts) > MAX_POINTS else np.arange(len(pts))
        self.points = pts[idx][even_order(pts[idx])]      # any first share of them covers the bone evenly
        tri = m.mesh.triangulate()
        if tri.n_cells > MAX_TRIANGLES:
            try:
                tri = tri.decimate(1.0 - MAX_TRIANGLES / tri.n_cells)
            except Exception:  # noqa: BLE001, S110 - keep the full mesh if decimation fails
                pass
        self.tri_points = np.asarray(tri.points, float)
        faces = np.asarray(tri.faces).reshape(-1, 4)
        self.triangles = faces[faces[:, 0] == 3, 1:]


def silhouette_path(uv: np.ndarray, triangles: np.ndarray) -> QPainterPath:
    """One path covering every projected triangle.

    Front- and back-facing triangles project with opposite windings, which would cancel out under a
    non-zero fill, so every triangle is first made counter-clockwise.
    """
    a, b, c = uv[triangles[:, 0]], uv[triangles[:, 1]], uv[triangles[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area < 0
    b2 = np.where(flip[:, None], c, b)
    c2 = np.where(flip[:, None], b, c)
    quad = np.stack([a, b2, c2, a], axis=1)                       # (n, 4, 2): closed triangles
    ok = np.isfinite(quad).all(axis=(1, 2))
    quad = quad[ok]
    connect = np.tile(np.array([1, 1, 1, 0], dtype=np.int32), len(quad))
    path = pg.arrayToQPath(quad[:, :, 0].ravel(), quad[:, :, 1].ravel(), connect=connect)
    path.setFillRule(Qt.WindingFill)
    return path


class CameraView(QWidget):
    """One camera: its radiograph at the current frame with the bones drawn over it."""

    def __init__(self, cam: XrayCamera):
        super().__init__()
        self.cam = cam
        self.setObjectName("Panel")
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(2)
        head = QHBoxLayout()
        self.title = label(cam.name, "H2", wrap=False)
        head.addWidget(self.title)
        head.addStretch(1)
        self.info = label("", "Mono", wrap=False)
        head.addWidget(self.info)
        v.addLayout(head)
        self.gv = pg.GraphicsLayoutWidget()
        self.gv.setBackground("#0B0D11")
        self.vb = self.gv.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)                       # image rows run downwards, like pixel v
        self.vb.setMenuEnabled(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.image)
        self.drr_item = pg.ImageItem(axisOrder="row-major")   # DRRs of TIF bones, in their colours (#25)
        self.drr_item.setZValue(5)
        self.vb.addItem(self.drr_item)
        self.gv.setToolTip("Drag: pan · Scroll: zoom · Double-click: fit the image")
        self.gv.scene().sigMouseClicked.connect(self._clicked)
        v.addWidget(self.gv, 1)
        self.items: dict[int, object] = {}          # id(model) -> scatter or path item
        self._image_key = None
        self._fitted = False

    def _clicked(self, ev) -> None:
        if ev.double():
            self.fit()

    def fit(self) -> None:
        cal = self.cam.calibration
        self.vb.setRange(QRectF(0, 0, cal.width, cal.height), padding=0.02)

    def update_view(self, frame: int, models, shapes: dict, lut: np.ndarray, overlay: str, alpha: float,
                    fill: bool = True, density: float = 1.0, drr: tuple = (0.7, 1.0, 0.0)) -> None:
        """Draw this frame: the radiograph, and each bone as a silhouette (filled at ``alpha``, or not shown when
        ``fill`` is off), a point cloud (``density``: the share of its points shown, thinned evenly), a DRR
        (TIF bones, with ``drr`` = (opacity, brightness, cutoff); other bones as silhouettes), or nothing."""
        cam, cal = self.cam, self.cam.calibration
        self.title.setText(cam.name)
        i = cam.image_index(frame)
        key = (i, id(lut))
        if key != self._image_key:
            self._image_key = key
            if i is None:
                self.image.clear()
                self.info.setText("no image for this frame")
            else:
                self.image.setImage(lut[cam.images.get_display(i)], levels=(0, 255), autoLevels=False)
                self.image.setRect(QRectF(0, 0, cal.width, cal.height))     # display images may be shrunk
                self.info.setText(cam.images.label(i))
                cam.images.prefetch(range(i + 1, i + 40))
            if not self._fitted:
                self.fit()
                self._fitted = True
        live = set()
        drr_layers = []                              # (DRR grid, colour) of each TIF bone shown as a DRR
        for m in models:
            pose = m.tracking.pose(frame, "hide") if m.tracking is not None else None
            if overlay == "drr" and getattr(m, "drr", None) is not None:
                old = self.items.pop(id(m), None)
                if old is not None:
                    self.vb.removeItem(old)
                live.add(id(m))
                if pose is not None and m.visible:
                    from ..core import drr as D

                    r = D.render_region(m.drr, pose, cal, D.grid_scale(cal), drr[2])
                    if r is not None:
                        drr_layers.append((*r, m.color))
                continue
            item = self._bone_item(m, "silhouette" if overlay == "drr" else overlay)
            live.add(id(m))
            pose = m.tracking.pose(frame, "hide") if m.tracking is not None else None
            if item is None:
                continue
            if overlay == "off" or pose is None or not m.visible or m.mesh is None:
                item.setVisible(False)
                continue
            shape = shapes[id(m)]
            colour = QColor(m.color)
            if overlay == "points":
                k = max(1, round(len(shape.points) * min(max(density, 0.0), 1.0)))
                uv = cal.project(shape.points[:k] @ pose[:3, :3].T + pose[:3, 3])
                colour.setAlphaF(min(1.0, alpha + 0.3))
                item.setData(pos=uv, brush=pg.mkBrush(colour), pen=None, size=2.5)
                item.setVisible(True)
                continue
            if fill:
                uv = cal.project(shape.tri_points @ pose[:3, :3].T + pose[:3, 3])
                item.setPath(silhouette_path(uv, shape.triangles))
                colour.setAlphaF(alpha)
                item.setBrush(QBrush(colour))
            item.setVisible(bool(fill))
        for k in [k for k in self.items if k not in live]:
            self.vb.removeItem(self.items.pop(k))
        self._show_drr(drr_layers, drr[0], drr[1])

    def _show_drr(self, layers, opacity: float, brightness: float) -> None:
        """Lay the bones' DRRs over the radiograph: each in its bone's colour, as opaque as its DRR value. Only the
        part of the image the bones cover is drawn."""
        if not layers:
            self.drr_item.setVisible(False)
            return
        x0 = min(x for _img, x, _y, _c in layers)
        y0 = min(y for _img, _x, y, _c in layers)
        x1 = max(x + img.shape[1] for img, x, _y, _c in layers)
        y1 = max(y + img.shape[0] for img, _x, y, _c in layers)
        H, W = y1 - y0, x1 - x0
        rgb = np.zeros((H, W, 3), np.float32)
        a_out = np.zeros((H, W), np.float32)
        for img, x, y, colour in layers:
            c = QColor(colour)
            a = np.clip(img * brightness, 0.0, 1.0) * opacity
            sl = (slice(y - y0, y - y0 + img.shape[0]), slice(x - x0, x - x0 + img.shape[1]))
            col = np.array([c.redF(), c.greenF(), c.blueF()], np.float32)
            rgb[sl] = rgb[sl] * (1 - a[..., None]) + col * a[..., None]
            a_out[sl] = a_out[sl] + a * (1 - a_out[sl])
        rgba = np.empty((H, W, 4), np.uint8)
        colour = rgb / np.maximum(a_out[..., None], 1e-6)
        rgba[..., :3] = np.clip(colour * 255, 0, 255)
        rgba[..., 3] = np.clip(a_out * 255, 0, 255)
        from ..core.drr import grid_scale

        s = grid_scale(self.cam.calibration)
        self.drr_item.setImage(rgba, levels=(0, 255), autoLevels=False)
        self.drr_item.setRect(QRectF(x0 * s, y0 * s, W * s, H * s))
        self.drr_item.setVisible(True)

    def _bone_item(self, m: BoneModel, overlay: str):
        want = pg.ScatterPlotItem if overlay == "points" else QGraphicsPathItem
        item = self.items.get(id(m))
        if item is not None and not isinstance(item, want):
            self.vb.removeItem(item)
            item = None
        if item is None:
            if overlay == "points":
                item = pg.ScatterPlotItem(pxMode=True)
            else:
                item = QGraphicsPathItem()
                item.setPen(QPen(Qt.NoPen))
            item.setZValue(10)
            self.vb.addItem(item)
            self.items[id(m)] = item
        return item


class CameraViews(QWidget):
    """The camera panels, one per X-ray camera, stacked top to bottom."""

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        v.addWidget(self.split)
        self.views: list[CameraView] = []
        self._shapes: dict[int, _BoneShape] = {}

    def _sync_cameras(self) -> None:
        cams = [c for c in self.scene.state.cameras if c.calibration is not None and c.images is not None]
        if len(cams) == len(self.views) and all(v.cam is c for v, c in zip(self.views, cams)):
            return
        for view in self.views:
            view.setParent(None)
            view.deleteLater()
        self.views = [CameraView(c) for c in cams]
        for view in self.views:
            self.split.addWidget(view)

    def _shape(self, m: BoneModel) -> _BoneShape | None:
        if m.mesh is None:
            return None
        s = self._shapes.get(id(m))
        if s is None or s.mesh_id != id(m.mesh):
            s = self._shapes[id(m)] = _BoneShape(m)
        return s

    def render_args(self) -> tuple:
        """(models, shapes, lut, overlay, opacity, fill, point density, DRR settings) for CameraView.update_view, as
        the panels draw them."""
        st = self.scene.state.settings
        models = [m for m in self.scene.state.models if m.tracking is not None and self._shape(m) is not None]
        live = {id(m) for m in self.scene.state.models}
        for k in [k for k in self._shapes if k not in live]:
            del self._shapes[k]
        shapes = {id(m): self._shapes[id(m)] for m in models}
        return (models, shapes, self.scene._lut(), st.overlay, st.overlay_opacity, st.overlay_fill, st.point_density,
                (st.drr_opacity, st.drr_brightness, st.drr_cutoff))

    def update_frame(self, frame: int) -> None:
        if not self.isVisible():
            return
        self._sync_cameras()
        args = self.render_args()
        for view in self.views:
            view.update_view(frame, *args)

    def fit_all(self) -> None:
        for view in self.views:
            view.fit()
