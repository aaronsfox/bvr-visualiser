"""Small geometry helpers: Euler angles and ITK transform (.tfm) files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def euler_zyx_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Autoscoper's yaw/pitch/roll (radians): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    m = np.eye(4)
    m[:3, :3] = Rz @ Ry @ Rx
    return m


def matrix_to_xyzypr(m: np.ndarray) -> np.ndarray:
    """Inverse of the above, matching CoordFrame::to_xyzypr in Autoscoper (degrees)."""
    R = m[:3, :3]
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return np.array([m[0, 3], m[1, 3], m[2, 3], yaw, pitch, roll])


class TransformFileError(ValueError):
    pass


def read_itk_tfm(path: str | Path) -> np.ndarray:
    """Read a single affine transform from an ITK .tfm file as a 4x4 matrix (as stored).

    ITK affine: y = A (x - c) + t + c, with A and t in `Parameters` and c in `FixedParameters`.
    """
    params = fixed = None
    kind = ""
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("Transform:"):
            if kind:
                raise TransformFileError("This file holds more than one transform; only single affine transforms are supported.")
            kind = line.split(":", 1)[1].strip()
        elif line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), float)
    if params is None or len(params) != 12:
        raise TransformFileError(f"Expected a 3D affine transform (12 parameters), found {kind or 'nothing usable'}.")
    if not any(k in kind for k in ("Affine", "MatrixOffset", "Euler3D", "Versor", "Similarity")):
        raise TransformFileError(f"Unsupported transform type: {kind}")
    A = params[:9].reshape(3, 3)
    t = params[9:12]
    c = fixed[:3] if fixed is not None and len(fixed) >= 3 else np.zeros(3)
    m = np.eye(4)
    m[:3, :3] = A
    m[:3, 3] = t + c - A @ c
    return m


def model_to_aut_from_tfm(path: str | Path) -> np.ndarray:
    """Matrix taking model coordinates into Autoscoper space for a SlicerAutoscoperM *-DICOM2AUT.tfm.

    ITK files store the resampling direction, so the matrix is inverted. Verified on the
    wrist sample: inv(rad-DICOM2AUT) maps rad.stl onto AUT_rad.stl with 0.0 mm error.
    """
    return np.linalg.inv(read_itk_tfm(path))
