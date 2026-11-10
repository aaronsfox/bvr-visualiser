"""X-ray camera calibrations.

MayaCam 2.0 text files (the format in all SlicerAutoscoperM sample data):

    image size / W,H / camera matrix / K (3 rows) / rotation / R (3 rows) / translation / t (3 rows)

R and t map world to camera: x_cam = R X + t; pixel = K x_cam / z (no image flip).
Verified by projecting tracked bones onto the wrist and ankle sample radiographs.

MayaCam 1.x files (5 lines of 3 comma-separated numbers, e.g. XMALab's *_MayaCam.csv exports):

    1: camera position in world space
    2: rotations (degrees) about the camera's local x, y and z axes, applied z, then y, then x
    3: film-plane position in camera space (derived from lines 4 and 5; not needed)
    4: u0, v0, Z  (principal point in 1-based pixels, and -principal distance in pixels)
    5: scale (mm per pixel), image width, image height (0 = 1024)

They are converted to the same K, R, t as MayaCam 2.0 following Autoscoper's own reader
(libautoscoper Camera::loadMayaCam1 and CoordFrame::from_xyzypr): the camera frame is built from
the position and z-y-x rotations, it looks along its local -z axis, and image rows run downwards
from the top (v = H - 1 - v_up). Replicating Autoscoper's internal camera model reproduces the
MayaCam 2.0 projections of all three SlicerAutoscoperM samples exactly; the 1.x conversion uses
that same model (checked on a Lego calibration cube, tests/test_calibration_mayacam1.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CalibrationError(ValueError):
    pass


@dataclass
class Calibration:
    path: Path
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    kind: str = "MayaCam 2.0"

    @property
    def source(self) -> np.ndarray:
        """X-ray source (camera centre) in world coordinates."""
        return -self.R.T @ self.t

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Camera-space depth (z) of world points."""
        return (np.atleast_2d(points) @ self.R.T + self.t)[:, 2]

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (N,3) to pixel coordinates (N,2)."""
        xc = np.atleast_2d(points) @ self.R.T + self.t
        uv = xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        return self.source + self.R.T @ (ray * depth)

    def plane_corners(self, depth: float) -> np.ndarray:
        """World positions of the image corners (0,0), (W,0), (W,H), (0,H) at a camera depth."""
        W, H = self.width, self.height
        return np.array([self.pixel_to_world(u, v, depth) for u, v in [(0, 0), (W, 0), (W, H), (0, H)]])

    def view_angle(self) -> float:
        """Vertical field of view (degrees) that just fits the image height."""
        return float(np.degrees(2 * np.arctan(self.height / 2 / self.K[1, 1])))

    def summary(self) -> str:
        return f"{self.kind} · image {self.width} × {self.height} · source {np.linalg.norm(self.source):.0f} mm from origin"


def _rows(lines: list[str], key: str, n: int) -> list[list[float]]:
    try:
        i = lines.index(key)
    except ValueError:
        raise CalibrationError(f"Missing '{key}' section.") from None
    out = []
    for ln in lines[i + 1:i + 1 + n]:
        out.append([float(x) for x in ln.replace(",", " ").split()])
    return out


def load_calibration(path: str | Path) -> Calibration:
    path = Path(path)
    if path.suffix.lower() == ".json":
        raise CalibrationError("VTK camera (.json) calibrations aren't supported yet. Please use a MayaCam file.")
    text = path.read_text(errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise CalibrationError("The file is empty.")
    if lines[0].lower() != "image size":
        return _load_mayacam1(path, lines)
    try:
        (size,) = _rows(lines, "image size", 1)
        K = np.array(_rows(lines, "camera matrix", 3))
        R = np.array(_rows(lines, "rotation", 3))
        t = np.array(_rows(lines, "translation", 3)).reshape(-1)
    except (ValueError, IndexError) as e:
        if isinstance(e, CalibrationError):
            raise
        raise CalibrationError("Couldn't read the numbers in this calibration file.") from None
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or len(size) != 2:
        raise CalibrationError("The calibration sections don't have the expected sizes.")
    if abs(np.linalg.det(R) - 1) > 1e-3:
        raise CalibrationError("The rotation section isn't a valid rotation matrix.")
    return Calibration(path=path, K=K, R=R, t=t, width=int(size[0]), height=int(size[1]))


def _xyzypr_frame(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rows = the camera's local x, y, z axes in world space, as Autoscoper's CoordFrame::from_xyzypr
    builds them: rotate about the frame's z axis by rz, then its y axis by ry, then its x axis by rx."""
    def rotate(M, axis, deg):
        a = axis / np.linalg.norm(axis)
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        oc = 1 - c
        x, y, z = a
        R = np.array([[oc * x * x + c, oc * x * y + s * z, oc * x * z - s * y],
                      [oc * x * y - s * z, oc * y * y + c, oc * y * z + s * x],
                      [oc * x * z + s * y, oc * y * z - s * x, oc * z * z + c]])
        return M @ R

    M = np.eye(3)
    M = rotate(M, M[2].copy(), rz)
    M = rotate(M, M[1].copy(), ry)
    return rotate(M, M[0].copy(), rx)


def mayacam1_to_krt(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """K, R, t, width, height (MayaCam 2.0 convention) from the 5 x 3 numbers of a MayaCam 1.x file."""
    v = np.asarray(values, float)
    position, (rx, ry, rz) = v[0], v[1]
    u0, v0, z = v[3][0] - 1.0, v[3][1] - 1.0, v[3][2]          # 1-based (MATLAB) pixels -> 0-based
    W = round(v[4][1]) or 1024
    H = round(v[4][2]) or 1024
    f = abs(z)
    if f == 0:
        raise CalibrationError("The principal distance (line 4, third value) is zero.")
    M = _xyzypr_frame(rx, ry, rz)
    R = np.diag([1.0, -1.0, -1.0]) @ M            # camera looks along local -z; image rows run downwards
    t = -R @ position
    K = np.array([[f, 0.0, u0], [0.0, f, (H - 1) - v0], [0.0, 0.0, 1.0]])
    return K, R, t, W, H


def _load_mayacam1(path: Path, lines: list[str]) -> Calibration:
    rows = []
    for ln in lines:
        if ln.lower().startswith(("version", "mayacam")):
            continue
        try:
            vals = [float(x) for x in ln.replace(",", " ").split()]
        except ValueError:
            raise CalibrationError(
                "This isn't a MayaCam calibration: expected 'image size' (MayaCam 2.0) or 5 lines of 3 numbers "
                "(MayaCam 1.x).") from None
        rows.append(vals)
    if len(rows) < 5 or any(len(r) != 3 for r in rows[:5]):
        raise CalibrationError("A MayaCam 1.x file needs 5 lines of 3 numbers (camera position, rotations, "
                               "film plane, u0 v0 Z, scale width height).")
    K, R, t, W, H = mayacam1_to_krt(np.array(rows[:5]))
    return Calibration(path=path, K=K, R=R, t=t, width=W, height=H, kind="MayaCam 1.x")
