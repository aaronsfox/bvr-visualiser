# Part of the STAPLE port (see NOTICE.md). bodySide2Sign is ported from msk-STAPLE (CC BY-NC 4.0,
# Copyright (c) 2020-2021 Modenese L., Renault J.-B.); the partial-bone helpers are additions for bones
# imaged by biplanar videoradiography, not part of STAPLE.
"""Shared pieces: results, body side, mesh conversion and the partial-bone helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fitting import PlanPolygonCentroid3D
from .tri import StapleError, Tri, TriPlanIntersect, normalizeV


def bodySide2Sign(side):
    s = str(side).strip().lower()[:1]
    if s == "r":
        return 1, "r"
    if s == "l":
        return -1, "l"
    raise StapleError("Specify the body side: right ('r') or left ('l').")


@dataclass
class BoneResult:
    """What a STAPLE algorithm found on one bone.

    JCS: joint frames on this bone, keyed like STAPLE (e.g. "knee_r", "hip_r"): {"V": 3x3 columns X Y Z,
    "Origin": (3,)}. aux: STAPLE's AuxCSInfo (fitted centres, radii, axes). art_surf: the articular
    surfaces found (Tri). notes: anything the user should know (fallbacks, partial-bone substitutes).
    """
    bone: str
    method: str
    side: str
    JCS: dict
    aux: dict = field(default_factory=dict)
    art_surf: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    whole: bool = True

    def __post_init__(self):
        for k, j in self.JCS.items():
            if not (np.all(np.isfinite(j["V"])) and np.all(np.isfinite(j["Origin"]))):
                raise StapleError(f"{self.method} on the {self.bone} gave an invalid {k.split('_')[0]} frame "
                                  "(the surface may be incomplete or of poor quality).")

    def frame(self, joint: str) -> np.ndarray:
        """4x4 frame (columns X, Y, Z, origin) of the joint frame named e.g. "knee" or "knee_r"."""
        key = joint if joint in self.JCS else f"{joint}_{self.side}"
        j = self.JCS[key]
        m = np.eye(4)
        m[:3, :3] = j["V"]
        m[:3, 3] = j["Origin"]
        return m


def tri_from_pyvista(mesh, target_faces: int | None = 40000) -> Tri:
    """A Tri from a PyVista surface: triangulated, cleaned, outward-facing, and optionally decimated.

    STAPLE was developed on meshes of ~10-30k faces; its morphological operations scale with the mean
    edge length, so decimating very dense scans keeps the run time reasonable without changing the method.
    """
    m = mesh.triangulate().clean()
    if target_faces and m.n_cells > target_faces:
        m = m.decimate_pro(1 - target_faces / m.n_cells, preserve_topology=True).clean()
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True, split_vertices=False,
                          point_normals=False, cell_normals=True)
    faces = np.asarray(m.faces).reshape(-1, 4)[:, 1:]
    return Tri(faces, np.asarray(m.points, float))


def partial_long_axis(tri: Tri, axis, epiphysis_at: str):
    """PARTIAL. Long-axis direction of a partial long bone from where its widest cross-section is.

    epiphysis_at="low": return the direction pointing away from the wide (epiphyseal) end, i.e.
    distal-to-proximal for a distal femur. "high": pointing towards it (a proximal tibia).
    """
    axis = normalizeV(axis)
    z = tri.Points @ axis
    alt = np.linspace(z.min() + 0.5, z.max() - 0.5, 60)
    areas = []
    for a in alt:
        curves, _ = TriPlanIntersect(tri, axis, -a)
        areas.append(max([c["Area"] for c in curves], default=0.0))
    areas = np.array(areas)
    k = int(np.argmax(areas))
    if k > 0.66 * len(alt):
        wide_at_plus = True
    elif k < 0.33 * len(alt):
        wide_at_plus = False
    else:
        q = len(alt) // 4
        wide_at_plus = areas[-q:].mean() > areas[:q].mean()
    toward = axis if wide_at_plus else -axis
    return -toward if epiphysis_at == "low" else toward


def shaft_line(tri: Tri, Z0, Zepi, epiphysis: str, margin=10.0, step=2.0):
    """PARTIAL. The line through the centroids of the shaft's cross-sections, from ``margin`` mm beyond
    the epiphysis to 3 mm before the cut end. Returns (unit direction along Z0, point on the line, notes)."""
    Z0 = normalizeV(Z0)
    z = tri.Points @ Z0
    if epiphysis == "distal":
        lo, hi = Zepi + margin, z.max() - 3
    else:
        lo, hi = z.min() + 3, Zepi - margin
    notes = []
    cents = []
    for a in np.arange(lo, hi, step):
        curves, _ = TriPlanIntersect(tri, Z0, -a)
        if not curves:
            continue
        c = max(curves, key=lambda c: c["Area"])
        cents.append(PlanPolygonCentroid3D(c["Pts"])[0])
    cents = np.array(cents)
    if len(cents) < 5 or np.ptp(cents @ Z0) < 20:
        notes.append("The shaft is too short for its own axis; used the bone's principal axis instead.")
        sel = tri.Points[(z > lo) & (z < hi)] if np.any((z > lo) & (z < hi)) else tri.Points
        return Z0, sel.mean(0), notes
    _, _, Vt = np.linalg.svd(cents - cents.mean(0))
    d = Vt[0]
    return np.sign(d @ Z0) * d, cents.mean(0), notes


def shaft_axis(tri: Tri, Z0, Zepi, epiphysis: str, margin=10.0, step=2.0):
    """PARTIAL. Unit direction (along Z0) of the shaft line (see shaft_line)."""
    d, _, notes = shaft_line(tri, Z0, Zepi, epiphysis, margin, step)
    return d, notes
