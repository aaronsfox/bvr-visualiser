"""Reading Autoscoper trial configuration (.cfg) files.

Only used as a shortcut: to pick up camera calibrations, radiograph folders,
voxel sizes and volume flips. Paths in a .cfg are relative to the .cfg itself
and often use Windows backslashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CfgVolume:
    path: Path
    flip: tuple[int, int, int] = (0, 0, 0)
    voxel_size: tuple[float, float, float] | None = None


@dataclass
class TrialConfig:
    path: Path
    calibrations: list[Path] = field(default_factory=list)
    camera_dirs: list[Path] = field(default_factory=list)
    volumes: list[CfgVolume] = field(default_factory=list)

    def voxel_size_for(self, tif_path: str | Path) -> tuple[float, float, float] | None:
        name = Path(tif_path).name.lower()
        for v in self.volumes:
            if v.path.name.lower() == name and v.voxel_size:
                return v.voxel_size
        return None

    def flip_for(self, tif_path: str | Path) -> tuple[int, int, int]:
        name = Path(tif_path).name.lower()
        for v in self.volumes:
            if v.path.name.lower() == name:
                return v.flip
        return (0, 0, 0)


def _resolve(base: Path, value: str) -> Path:
    p = Path(value.strip().strip('"').replace("\\", "/"))
    return p if p.is_absolute() else (base / p)


def read_cfg(path: str | Path) -> TrialConfig:
    path = Path(path)
    base = path.parent
    cfg = TrialConfig(path=path)
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        value = value.strip()
        if key == "mayaCam_csv":
            cfg.calibrations.append(_resolve(base, value))
        elif key == "CameraRootDir":
            cfg.camera_dirs.append(_resolve(base, value))
        elif key == "VolumeFile":
            cfg.volumes.append(CfgVolume(path=_resolve(base, value)))
        elif key == "VolumeFlip" and cfg.volumes:
            cfg.volumes[-1].flip = tuple(int(float(x)) for x in value.split()[:3])
        elif key == "VoxelSize" and cfg.volumes:
            cfg.volumes[-1].voxel_size = tuple(float(x) for x in value.split()[:3])
    return cfg
