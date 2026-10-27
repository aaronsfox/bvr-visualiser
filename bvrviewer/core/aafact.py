"""AAFACT (Peterson, Kruger & Lenz 2023): run the user's compiled AAFACT program and read its results.

AAFACT computes anatomical coordinate systems for 14 foot and ankle bones by aligning each bone to a
template (ICP) and using regional centroids. It is distributed under a licence that does not allow
modified copies, so nothing of AAFACT is included here: the user installs the compiled program
(github.com/Lenz-Lab/AAFACT, with the MATLAB Runtime) and the viewer runs it, or imports the Excel
file AAFACT writes.

AAFACT's axes: AP positive anterior (distal for the metatarsals), SI positive superior (dorsal), ML
positive to the subject's right on both sides. They map onto ISB names as X = AP, Y = SI, Z = ML.

Results layout (one sheet per bone and coordinate system; sheet name = file name, with a prefix for
bones that have several systems: TN_/TT_/ST_ talus, CC_/ST_ calcaneus, V_/R_ cuboid, lateral
cuneiform and metatarsals):
    A1:B3   Subject / Bone Model / Side
    A6:D6   "<origin> Origin", x, y, z        (in the bone file's own coordinates)
    A7:D7   "AP Axis", end point of the unit AP axis (origin + AP)
    A8:D8   "SI Axis", origin + SI
    A9:D9   "ML Axis", origin + ML
    A11:D14 the same in AAFACT's template space (not used here)

Reference: Peterson AC, Kruger KM, Lenz AL (2023) Automatic anatomical foot and ankle coordinate
toolbox. Front Bioeng Biotechnol 11:1255464.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PETERSON = ("Peterson AC, Kruger KM, Lenz AL (2023) Automatic anatomical foot and ankle coordinate toolbox. "
            "Front Bioeng Biotechnol 11:1255464 (AAFACT; github.com/Lenz-Lab/AAFACT)")

# AAFACT's bone names (as it detects them in file names) by our segment type key
BONES = {
    "talus": "Talus", "calcaneus": "Calcaneus", "navicular": "Navicular", "cuboid": "Cuboid",
    "medial_cuneiform": "Medial_Cuneiform", "intermediate_cuneiform": "Intermediate_Cuneiform",
    "lateral_cuneiform": "Lateral_Cuneiform", "metatarsal1": "Metatarsal1", "metatarsal2": "Metatarsal2",
    "metatarsal3": "Metatarsal3", "metatarsal4": "Metatarsal4", "metatarsal5": "Metatarsal5",
    "tibia": "Tibia", "tibia_distal": "Tibia", "fibula": "Fibula",
}
_BONE_ALIASES = {  # AAFACT's second list of names, which it also recognises
    "Talus": ("talus",), "Calcaneus": ("calcaneus",), "Navicular": ("navicular",), "Cuboid": ("cuboid",),
    "Medial_Cuneiform": ("medial_cuneiform", "med_cuneiform"),
    "Intermediate_Cuneiform": ("intermediate_cuneiform", "int_cuneiform"),
    "Lateral_Cuneiform": ("lateral_cuneiform", "lat_cuneiform"),
    "Metatarsal1": ("metatarsal1", "first_metatarsal"), "Metatarsal2": ("metatarsal2", "second_metatarsal"),
    "Metatarsal3": ("metatarsal3", "third_metatarsal"), "Metatarsal4": ("metatarsal4", "fourth_metatarsal"),
    "Metatarsal5": ("metatarsal5", "fifth_metatarsal"), "Tibia": ("tibia",), "Fibula": ("fibula",),
}

# The coordinate systems AAFACT offers per bone, by sheet prefix
OPTIONS = {
    "Talus": {"TN": "Talonavicular (whole talus; AP towards the talar head)",
              "TT": "Tibiotalar (from the talar dome)", "ST": "Subtalar (from the subtalar facets)"},
    "Calcaneus": {"CC": "Calcaneocuboid (AP along the calcaneus to the CC joint)",
                  "ST": "Subtalar (from the subtalar facets)"},
    "Cuboid": {"V": "Vertical (SI perpendicular to the plantar plane)",
               "R": "Radial (SI perpendicular to the bone's dorsal surface in the arch)"},
    "Lateral_Cuneiform": {"V": "Vertical", "R": "Radial"},
    **{f"Metatarsal{i}": {"V": "Vertical", "R": "Radial"} for i in range(1, 6)},
}


class AafactError(RuntimeError):
    pass


@dataclass
class AafactCS:
    sheet: str
    subject: str
    bone: str                  # AAFACT's bone name, e.g. "Talus"
    side: str                  # "right" / "left"
    option: str                # "TN", "TT", "ST", "CC", "V", "R" or "" (bones with one system)
    origin_name: str           # e.g. "Center"
    acs: np.ndarray            # 4x4: columns X (AP), Y (SI), Z (ML, right), origin; orthonormalised
    raw_axes: np.ndarray = field(default=None)   # 3x3 columns AP, SI, ML as written by AAFACT
    orthogonality: float = 0.0  # largest deviation from 90 degrees between AAFACT's axes

    @property
    def title(self) -> str:
        opt = OPTIONS.get(self.bone, {}).get(self.option, "")
        return f"{self.bone.replace('_', ' ')}" + (f" – {self.option}: {opt}" if self.option else "")


def _orthonormal(A):
    """Closest rotation to A (polar decomposition), so small non-orthogonality doesn't skew the axes."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _cells(ws):
    rows = {}
    for row in ws.iter_rows(min_row=1, max_row=20, max_col=4, values_only=True):
        rows[len(rows) + 1] = list(row) + [None] * (4 - len(row))
    return rows


def read_results(path) -> list[AafactCS]:
    """All coordinate systems in an AAFACT results workbook (CoordinateSystem_<folder>.xlsx)."""
    try:
        import openpyxl
    except ImportError as err:                      # pragma: no cover - openpyxl is a dependency
        raise AafactError("Reading AAFACT results needs the openpyxl package.") from err
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as err:  # noqa: BLE001
        raise AafactError(f"Couldn't open {Path(path).name}: {err}") from err
    out = []
    for ws in wb.worksheets:
        r = _cells(ws)
        try:
            subject, bone, side = (str(r[i][1]).strip() for i in (1, 2, 3))
            vals = [np.array([float(v) for v in r[i][1:4]]) for i in (6, 7, 8, 9)]
        except (KeyError, TypeError, ValueError):
            continue                                    # not an AAFACT sheet
        if not str(r[7][0] or "").lower().startswith("ap"):
            continue
        o = vals[0]
        raw = np.c_[vals[1] - o, vals[2] - o, vals[3] - o]
        raw = raw / np.linalg.norm(raw, axis=0)
        dev = max(abs(90 - np.degrees(np.arccos(np.clip(raw[:, i] @ raw[:, j], -1, 1))))
                  for i, j in ((0, 1), (0, 2), (1, 2)))
        acs = np.eye(4)
        acs[:3, :3] = _orthonormal(raw)
        acs[:3, 3] = o
        m = re.match(r"^(TN|TT|ST|CC|V|R)_", ws.title)
        opt = m.group(1) if m and bone in OPTIONS and m.group(1) in OPTIONS[bone] else ""
        origin_name = re.sub(r"\s*Origin$", "", str(r[6][0] or "")).strip() or "Origin"
        out.append(AafactCS(ws.title, subject, bone, side.lower(), opt, origin_name, acs, raw, float(dev)))
    wb.close()
    if not out:
        raise AafactError(f"No AAFACT coordinate systems found in {Path(path).name}.")
    return out


def bone_from_name(text: str) -> str | None:
    """AAFACT's bone name found in a file name, as AAFACT itself would find it (first match wins)."""
    t = text.lower()
    for bone, names in _BONE_ALIASES.items():
        if any(n in t for n in names):
            return bone
    return None


def run(exe, surface, bone: str, side: str, workdir, timeout: float = 900) -> list[AafactCS]:
    """Run the compiled AAFACT program on one bone surface and return its coordinate systems.

    exe: AAFACT's executable (AAFACT.exe on Windows). surface: a PyVista surface or a surface file, in
    the coordinates the axes should be returned in. bone: AAFACT's bone name (see BONES). The surface
    is written into a fresh sub-folder of workdir, named so AAFACT recognises the bone and side.
    """
    exe = Path(exe)
    if not exe.exists():
        raise AafactError(f"AAFACT program not found: {exe}")
    if bone not in _BONE_ALIASES:
        raise AafactError(f"AAFACT doesn't handle the bone {bone!r}.")
    if side not in ("right", "left"):
        raise AafactError("AAFACT needs the bone's side (right or left).")
    work = Path(workdir)
    job = work / "bvrjob"                       # folder name must not contain bone or side words
    k = 0
    while job.exists():
        k += 1
        job = work / f"bvrjob{k}"
    job.mkdir(parents=True)
    dst = job / f"bvr_{bone}_{side.capitalize()}.stl"
    if isinstance(surface, (str, Path)):
        import pyvista as pv
        surface = pv.read(str(surface))
    surface_for_aafact(surface, dst)
    try:
        proc = subprocess.run([str(exe), str(job)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as err:
        raise AafactError(f"AAFACT didn't finish within {timeout:.0f} s.") from err
    except OSError as err:
        raise AafactError(f"Couldn't start AAFACT: {err}") from err
    xl = job / f"CoordinateSystem_{job.name}.xlsx"
    if not xl.exists():
        found = sorted(job.glob("CoordinateSystem_*.xlsx"))
        if not found:
            raise AafactError("AAFACT finished without writing results."
                              + (f"\n\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}" if proc.stdout or proc.stderr
                                 else ""))
        xl = found[-1]
    return read_results(xl)


def surface_for_aafact(mesh, path) -> Path:
    """Write a PyVista surface as an STL that AAFACT can read (triangles only)."""
    p = Path(path)
    surf = mesh if mesh.is_all_triangles and hasattr(mesh, "faces") else mesh.extract_surface(algorithm=None)
    surf.triangulate().save(str(p))
    return p
