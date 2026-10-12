"""X-ray camera calibrations.

Supports MayaCam 2.0 text files (the format in all SlicerAutoscoperM sample data):

    image size / W,H / camera matrix / K (3 rows) / rotation / R (3 rows) / translation / t (3 rows)

R and t map world to camera: x_cam = R X + t; pixel = K x_cam / z (no image flip).
Verified by projecting tracked bones onto the wrist and ankle sample radiographs.
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
        raise CalibrationError("VTK camera (.json) calibrations aren't supported yet. Please use a MayaCam 2.0 file.")
    text = path.read_text(errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise CalibrationError("The file is empty.")
    if lines[0].lower() != "image size":
        raise CalibrationError(
            "This looks like a MayaCam 1.0 file (5 lines of numbers). Only MayaCam 2.0 is supported so far."
        )
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
