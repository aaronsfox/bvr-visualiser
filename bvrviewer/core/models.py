"""Bone models: surfaces (STL etc.) and Autoscoper partial volumes (TIF).

Everything is brought into Autoscoper ("AUT") volume space, the frame the .tra poses act on:

* TIF partial volume, read as (z, y, x) = (k, j, i) with voxel size (dx, dy, dz):
      x = i*dx,  y = (ny - j)*dy,  z = -k*dz
  Verified on the wrist and knee samples against the AUT_*.stl meshes.
* Surfaces are assumed to already be in AUT space (e.g. SlicerAutoscoperM's AUT_*.stl),
  unless a .tfm is given (e.g. *-DICOM2AUT.tfm), whose inverse is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SURFACE_EXTS = {".stl", ".obj", ".ply", ".vtk", ".vtp"}
VOLUME_EXTS = {".tif", ".tiff"}

PALETTE = ["#E8D9B5", "#4F9CF9", "#F0A04B", "#5CC8A8", "#B78AF0", "#E07A9A", "#C9B98F", "#8FA3BF",
           "#7FC8F8", "#F2C48D"]


class ModelError(ValueError):
    pass


def is_volume_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VOLUME_EXTS


def is_surface_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SURFACE_EXTS


def load_surface(path: str | Path):
    import pyvista as pv

    path = Path(path)
    try:
        mesh = pv.read(path)
    except Exception as e:  # noqa: BLE001 - surface readers raise many types
        raise ModelError(f"Couldn't read {path.name}: {e}") from None
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points == 0:
        raise ModelError(f"{path.name} contains no geometry.")
    b = np.array(mesh.bounds).reshape(3, 2)
    if np.linalg.norm(b[:, 1] - b[:, 0]) < 1.0:
        raise ModelError(
            f"{path.name} is smaller than 1 mm across. It may be empty or saved in different units."
        )
    return mesh.triangulate().clean()


def load_volume_array(path: str | Path) -> np.ndarray:
    import tifffile

    try:
        vol = tifffile.imread(path)
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"Couldn't read {Path(path).name}: {e}") from None
    vol = np.squeeze(vol)
    if vol.ndim == 4 and vol.shape[-1] in (3, 4):
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ModelError(f"{Path(path).name} isn't a 3D stack (shape {vol.shape}).")
    return vol


def apply_flips(vol: np.ndarray, flip: tuple[int, int, int]) -> np.ndarray:
    """Autoscoper VolumeFlip x y z. Untested: all sample data uses 0 0 0."""
    fx, fy, fz = flip
    if fx:
        vol = vol[:, :, ::-1]
    if fy:
        vol = vol[:, ::-1, :]
    if fz:
        vol = vol[::-1, :, :]
    return vol


def index_to_aut(verts_kji: np.ndarray, shape: tuple[int, int, int], voxel: tuple[float, float, float]) -> np.ndarray:
    nz, ny, nx = shape
    dx, dy, dz = voxel
    k, j, i = verts_kji[:, 0], verts_kji[:, 1], verts_kji[:, 2]
    return np.column_stack([i * dx, (ny - j) * dy, -k * dz])


def volume_image_data(vol: np.ndarray, voxel: tuple[float, float, float]):
    """A pyvista ImageData placed in AUT space, for volume rendering."""
    import pyvista as pv

    nz, ny, nx = vol.shape
    dx, dy, dz = voxel
    img = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dy, dz), origin=(0.0, ny * dy, 0.0))
    img.direction_matrix = np.diag([1.0, -1.0, -1.0])
    img.point_data["intensity"] = np.ascontiguousarray(vol).ravel(order="C")
    return img


def volume_surface(vol: np.ndarray, voxel: tuple[float, float, float], level: float | None = None,
                   max_voxels: int = 40_000_000):
    """Bone surface from a partial volume, in AUT space.

    SlicerAutoscoperM partial volumes are masked (zero outside the bone), so by default
    the surface is the boundary of the non-zero region, which reproduces the AUT_*.stl
    meshes. For unmasked volumes an Otsu threshold is used instead.
    Returns (mesh, description).
    """
    import pyvista as pv
    from skimage.filters import threshold_otsu
    from skimage.measure import marching_cubes

    step = 1
    while vol.size / step**3 > max_voxels:
        step += 1
    v = vol[::step, ::step, ::step]
    zero_frac = float((v == 0).mean())
    if level is None and zero_frac > 0.2:
        data, lvl, how = (v > 0).astype(np.float32), 0.5, "masked volume boundary"
    else:
        data = v.astype(np.float32)
        if level is None:
            sample = data[::2, ::2, ::2].ravel()
            lvl = float(threshold_otsu(sample[sample > sample.min()] if (sample > sample.min()).any() else sample))
            how = f"Otsu threshold {lvl:.0f}"
        else:
            lvl, how = float(level), f"threshold {level:g}"
    padded = np.pad(data, 1, mode="constant", constant_values=float(data.min()) if lvl > data.min() else 0)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=lvl)
    except (ValueError, RuntimeError):
        raise ModelError("No surface found at that threshold.") from None
    verts = (verts - 1) * step
    pts = index_to_aut(verts, vol.shape, voxel)
    mesh = pv.PolyData(pts, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    mesh = mesh.clean()
    if mesh.n_cells > 400_000:
        mesh = mesh.decimate(1 - 400_000 / mesh.n_cells)
    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.1)
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True, split_vertices=False)
    if step > 1:
        how += f", downsampled ×{step}"
    return mesh, how


@dataclass
class BoneModel:
    """Everything needed to show one bone. Paths and settings are saved with the tab."""

    name: str
    path: Path
    color: str = "#E8D9B5"
    opacity: float = 1.0
    visible: bool = True
    display: str = "surface"                  # "surface", "volume" or "both" (volume needs a TIF)
    voxel_size: tuple[float, float, float] | None = None
    flip: tuple[int, int, int] = (0, 0, 0)
    tfm_path: Path | None = None              # surfaces in another space
    threshold: float | None = None            # TIF surface level (None = automatic)
    tracking_path: Path | None = None
    tracking_volume: int = 0
    tracking_overrides: dict = field(default_factory=dict)
    # coordinate system: a segment of a model file from the Model Builder (.bvrmodel.json)
    axes_path: Path | None = None
    axes_segment: str | None = None
    axes_visible: bool = True
    axes_reference: object = "first"          # segment motion relative to: "first", "lab" or a frame index
    axes_zero: int | None = None              # also subtract the values at this frame

    # runtime only
    mesh: object = field(default=None, repr=False)
    image_data: object = field(default=None, repr=False)
    tracking: object = field(default=None, repr=False)
    axes_model: object = field(default=None, repr=False)     # the loaded KinematicModel
    surface_note: str = ""

    @property
    def has_axes(self) -> bool:
        return self.axes_path is not None and self.axes_segment is not None

    @property
    def acs(self):
        """This bone's coordinate system (4x4, in its model space), or None."""
        if self.axes_model is None or self.axes_segment is None:
            return None
        try:
            return self.axes_model.segment(self.axes_segment).acs
        except Exception:  # noqa: BLE001 - segment removed from the file since
            return None

    @property
    def is_volume(self) -> bool:
        return is_volume_file(self.path)

    @property
    def file_kind(self) -> str:
        return "TIF" if self.is_volume else self.path.suffix.upper().lstrip(".")

    def build(self) -> None:
        """Load the geometry into AUT space. Requires voxel_size for TIFs."""
        if self.is_volume:
            if not self.voxel_size:
                raise ModelError("Voxel size is needed for a TIF volume.")
            vol = apply_flips(load_volume_array(self.path), self.flip)
            self.mesh, self.surface_note = volume_surface(vol, self.voxel_size, self.threshold)
            self.image_data = volume_image_data(vol, self.voxel_size)
        else:
            mesh = load_surface(self.path)
            if self.tfm_path:
                from .transforms import model_to_aut_from_tfm

                mesh = mesh.transform(model_to_aut_from_tfm(self.tfm_path), inplace=False)
                self.surface_note = f"moved into Autoscoper space with {Path(self.tfm_path).name}"
            mesh = mesh.compute_normals(auto_orient_normals=False, split_vertices=False)
            self.mesh = mesh
            self.image_data = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "path": str(self.path), "color": self.color, "opacity": self.opacity,
            "visible": self.visible, "display": self.display,
            "voxel_size": list(self.voxel_size) if self.voxel_size else None,
            "flip": list(self.flip), "tfm_path": str(self.tfm_path) if self.tfm_path else None,
            "threshold": self.threshold,
            "tracking_path": str(self.tracking_path) if self.tracking_path else None,
            "tracking_volume": self.tracking_volume, "tracking_overrides": self.tracking_overrides,
            **({"axes": {"path": str(self.axes_path), "segment": self.axes_segment, "visible": self.axes_visible,
                         "reference": self.axes_reference, "zero": self.axes_zero}} if self.has_axes else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoneModel":
        return cls(
            name=d["name"], path=Path(d["path"]), color=d.get("color", "#E8D9B5"),
            opacity=d.get("opacity", 1.0), visible=d.get("visible", True), display=d.get("display", "surface"),
            voxel_size=tuple(d["voxel_size"]) if d.get("voxel_size") else None,
            flip=tuple(d.get("flip", (0, 0, 0))),
            tfm_path=Path(d["tfm_path"]) if d.get("tfm_path") else None,
            threshold=d.get("threshold"),
            tracking_path=Path(d["tracking_path"]) if d.get("tracking_path") else None,
            tracking_volume=d.get("tracking_volume", 0), tracking_overrides=d.get("tracking_overrides", {}),
            axes_path=Path(d["axes"]["path"]) if d.get("axes") else None,
            axes_segment=d["axes"]["segment"] if d.get("axes") else None,
            axes_visible=d.get("axes", {}).get("visible", True),
            axes_reference=d.get("axes", {}).get("reference", "first"),
            axes_zero=d.get("axes", {}).get("zero"),
        )
