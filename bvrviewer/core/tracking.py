"""Reading Autoscoper tracking (.tra) files.

Autoscoper writes one row per frame. Each row holds, for every exported volume,
either 16 values (a 4x4 matrix) or 6 values (x, y, z, yaw, pitch, roll).
Untracked frames are written as "NaN" tokens. Export options (see
AutoscoperMainWindow::save_tracking_results in the Autoscoper source):

* matrix, row order:    R00 R01 R02 tx  R10 R11 R12 ty  R20 R21 R22 tz  0 0 0 1
* matrix, column order: R00 R10 R20 0   R01 R11 R21 0   R02 R12 R22 0   tx ty tz 1
* xyzypr: translation, then yaw/pitch/roll as Z-Y-X Euler angles,
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
* separator: comma or space; translation in mm or cm; rotation in degrees or radians.

Units can't be detected from the numbers, so they default to mm/degrees
(Autoscoper's defaults) and can be overridden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .transforms import euler_zyx_to_matrix


class TrackingError(ValueError):
    pass


@dataclass
class TrackingFormat:
    kind: str = "matrix"            # "matrix" or "xyzypr"
    order: str = "row"              # "row" or "column" (matrix only)
    separator: str = ","            # "," or " "
    n_volumes: int = 1              # volumes side by side in each row
    translation_units: str = "mm"   # "mm" or "cm"
    rotation_units: str = "deg"     # "deg" or "rad" (xyzypr only)

    def describe(self) -> str:
        if self.kind == "matrix":
            s = f"4×4 matrix · {self.order} order"
        else:
            s = f"x y z yaw pitch roll · {self.rotation_units}"
        s += " · comma" if self.separator == "," else " · space"
        s += f" · {self.translation_units}"
        if self.n_volumes > 1:
            s += f" · {self.n_volumes} volumes in file"
        return s


@dataclass
class Tracking:
    path: Path
    poses: np.ndarray               # (n_frames, 4, 4); untracked frames are all-NaN
    fmt: TrackingFormat
    volume_index: int = 0
    warnings: list[str] = field(default_factory=list)
    _filled: dict = field(default_factory=dict, repr=False)

    @property
    def n_frames(self) -> int:
        return self.poses.shape[0]

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.poses).any(axis=(1, 2))

    @property
    def n_untracked(self) -> int:
        return int((~self.valid).sum())

    def gaps(self) -> list[tuple[int, int]]:
        """Runs of untracked frames as (first, last) inclusive, 0-based."""
        out, start = [], None
        for i, ok in enumerate(self.valid):
            if not ok and start is None:
                start = i
            elif ok and start is not None:
                out.append((start, i - 1))
                start = None
        if start is not None:
            out.append((start, self.n_frames - 1))
        return out

    def pose(self, frame: int, policy: str = "hide") -> np.ndarray | None:
        """Pose at a 0-based frame, or None if the bone should be hidden.

        policy: "hide" (None on untracked frames), "hold" (last tracked pose,
        or the next one before tracking starts) or "interpolate" (linear
        translation + spherical rotation interpolation across gaps; holds at the ends).
        Frames past the end of the file count as untracked.
        """
        if frame < 0 or frame >= self.n_frames:
            if policy == "hide" or not self.valid.any():
                return None
            frame = min(max(frame, 0), self.n_frames - 1)
        if policy == "hide":
            p = self.poses[frame]
            return None if np.isnan(p).any() else p
        filled = self._get_filled(policy)
        p = filled[frame]
        return None if np.isnan(p).any() else p

    def _get_filled(self, policy: str) -> np.ndarray:
        if policy not in self._filled:
            self._filled[policy] = fill_gaps(self.poses, policy)
        return self._filled[policy]


def fill_gaps(poses: np.ndarray, policy: str) -> np.ndarray:
    valid = ~np.isnan(poses).any(axis=(1, 2))
    out = poses.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    if policy == "hold":
        last = idx[0]
        for i in range(len(poses)):
            if valid[i]:
                last = i
            else:
                out[i] = poses[last]
        return out
    if policy == "interpolate":
        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_matrix(poses[idx, :3, :3])
        slerp = Slerp(idx, rots)
        missing = np.flatnonzero(~valid)
        inner = missing[(missing > idx[0]) & (missing < idx[-1])]
        if len(inner):
            out[inner, :3, :3] = slerp(inner).as_matrix()
            for k in range(3):
                out[inner, k, 3] = np.interp(inner, idx, poses[idx, k, 3])
            out[inner, 3] = [0, 0, 0, 1]
        out[missing[missing < idx[0]]] = poses[idx[0]]
        out[missing[missing > idx[-1]]] = poses[idx[-1]]
        return out
    raise ValueError(f"Unknown policy {policy!r}")


_NUM = re.compile(r"[,\s;]+")


def read_raw(path: str | Path) -> tuple[np.ndarray, str]:
    """Read a .tra file into a (rows, columns) float array plus the separator used."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise TrackingError("The file is empty.")
    sep = "," if "," in lines[0] else " "
    rows = []
    for n, ln in enumerate(lines, 1):
        toks = [t for t in _NUM.split(ln) if t]
        try:
            rows.append([float(t) for t in toks])          # float() accepts "NaN"/"nan"
        except ValueError:
            raise TrackingError(f"Line {n} contains something that isn't a number.") from None
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise TrackingError(f"Rows have different numbers of values ({sorted(widths)}).")
    return np.array(rows, dtype=float), sep


def _looks_row_major(block: np.ndarray) -> bool:
    return np.allclose(block[:, 12:15], 0, atol=1e-6) and np.allclose(block[:, 15], 1, atol=1e-6)


def _looks_col_major(block: np.ndarray) -> bool:
    return np.allclose(block[:, [3, 7, 11]], 0, atol=1e-6) and np.allclose(block[:, 15], 1, atol=1e-6)


def detect_format(raw: np.ndarray, sep: str) -> TrackingFormat:
    ncol = raw.shape[1]
    valid_rows = raw[~np.isnan(raw).any(axis=1)]
    fmt = TrackingFormat(separator=sep)
    as_matrix = ncol % 16 == 0
    as_xyz = ncol % 6 == 0
    if as_matrix and len(valid_rows):
        b = valid_rows[:, :16]
        as_matrix = _looks_row_major(b) or _looks_col_major(b)
    if as_matrix:
        fmt.kind = "matrix"
        fmt.n_volumes = ncol // 16
        if len(valid_rows):
            b = valid_rows[:, :16]
            fmt.order = "row" if _looks_row_major(b) else "column"
        return fmt
    if as_xyz:
        fmt.kind = "xyzypr"
        fmt.n_volumes = ncol // 6
        if len(valid_rows):
            ang = np.abs(valid_rows[:, 3:6])
            # Autoscoper's yaw/pitch/roll are within ±180°; values all under 2π suggest radians.
            if ang.max() <= 2 * np.pi + 1e-6 and ang.max() > 0 and np.median(ang) < 1.0:
                fmt.rotation_units = "rad"
        return fmt
    raise TrackingError(
        f"Each row has {ncol} values. Expected 16 per bone (matrix) or 6 per bone (x y z yaw pitch roll)."
    )


def to_poses(raw: np.ndarray, fmt: TrackingFormat, volume_index: int = 0) -> np.ndarray:
    width = 16 if fmt.kind == "matrix" else 6
    if not 0 <= volume_index < fmt.n_volumes:
        raise TrackingError(f"Volume {volume_index + 1} requested but the file holds {fmt.n_volumes}.")
    block = raw[:, volume_index * width:(volume_index + 1) * width]
    n = block.shape[0]
    poses = np.full((n, 4, 4), np.nan)
    tscale = 10.0 if fmt.translation_units == "cm" else 1.0
    for i in range(n):
        v = block[i]
        if np.isnan(v).any():
            continue
        if fmt.kind == "matrix":
            m = v.reshape(4, 4)
            if fmt.order == "column":
                m = m.T
            m = m.copy()
        else:
            ang = v[3:6] if fmt.rotation_units == "rad" else np.radians(v[3:6])
            m = euler_zyx_to_matrix(*ang)
            m[:3, 3] = v[:3]
        m[:3, 3] *= tscale
        m[3] = [0, 0, 0, 1]
        poses[i] = m
    return poses


def load_tracking(path: str | Path, volume_index: int = 0, overrides: dict | None = None) -> Tracking:
    """Load a .tra file. `overrides` can set any TrackingFormat field the user corrects."""
    path = Path(path)
    raw, sep = read_raw(path)
    fmt = detect_format(raw, sep)
    for k, v in (overrides or {}).items():
        setattr(fmt, k, v)
    poses = to_poses(raw, fmt, volume_index)
    t = Tracking(path=path, poses=poses, fmt=fmt, volume_index=volume_index)
    valid = t.valid
    if not valid.any():
        t.warnings.append("No tracked frames in this file (every row is NaN).")
    else:
        R = poses[valid, :3, :3]
        err = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
        if err > 1e-3:
            t.warnings.append(
                "Rotations aren't orthonormal. The format (row/column order or angles) may be wrong."
            )
    return t
