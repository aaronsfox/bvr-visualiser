"""What a tab holds, and saving/loading it as a JSON session file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import BoneModel

SESSION_VERSION = 1


@dataclass
class XrayCamera:
    calibration_path: Path
    source_kind: str                 # "folder", "files" or "multipage"
    source: list[Path]               # the folder, the files, or the single TIFF
    first_image: int = 0             # index of the image shown at tracking frame 1
    visible: bool = True
    name: str = "X-ray"
    show_plane: bool = True          # radiograph on its detector plane in the 3D scene
    show_source: bool = True         # X-ray source marker
    show_beams: bool = False         # lines from the source to the image corners

    # runtime only
    calibration: object = field(default=None, repr=False)
    images: object = field(default=None, repr=False)

    def build(self) -> None:
        from .calibration import load_calibration
        from .images import ImageSequence, find_images

        self.calibration = load_calibration(self.calibration_path)
        if self.source_kind == "folder":
            files, _ = find_images(self.source[0])
            self.images = ImageSequence(files=files)
        elif self.source_kind == "multipage":
            self.images = ImageSequence(multipage=self.source[0])
        else:
            self.images = ImageSequence(files=sorted(self.source))

    def image_index(self, frame: int) -> int | None:
        i = self.first_image + frame
        return i if self.images is not None and 0 <= i < len(self.images) else None

    def to_dict(self) -> dict:
        return {"name": self.name, "calibration_path": str(self.calibration_path), "source_kind": self.source_kind,
                "source": [str(p) for p in self.source], "first_image": self.first_image, "visible": self.visible,
                "show_plane": self.show_plane, "show_source": self.show_source, "show_beams": self.show_beams}

    @classmethod
    def from_dict(cls, d: dict) -> "XrayCamera":
        return cls(calibration_path=Path(d["calibration_path"]), source_kind=d["source_kind"],
                   source=[Path(p) for p in d["source"]], first_image=d.get("first_image", 0),
                   visible=d.get("visible", True), name=d.get("name", "X-ray"),
                   show_plane=d.get("show_plane", True), show_source=d.get("show_source", True),
                   show_beams=d.get("show_beams", False))


@dataclass
class SceneSettings:
    show_axes: bool = True
    show_grid: bool = True
    background: str = "dark"          # "dark", "light", "white"
    trail: bool = False
    trail_step: int = 5
    trail_count: int = 3
    untracked: str = "hide"           # "hide", "hold", "interpolate"
    hold_still: str | None = None     # name of the bone kept still
    xray_opacity: float = 0.9
    xray_distance: float = 120.0      # mm beyond the bones
    xray_bone_opacity: float = 0.5
    xray_outline: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    invert: bool = False
    speed: float = 1.0
    loop: bool = True
    fps: float = 30.0


@dataclass
class TabState:
    name: str = "Untitled tab"
    models: list[BoneModel] = field(default_factory=list)
    cameras: list[XrayCamera] = field(default_factory=list)
    settings: SceneSettings = field(default_factory=SceneSettings)

    def to_dict(self) -> dict:
        return {"version": SESSION_VERSION, "name": self.name,
                "models": [m.to_dict() for m in self.models],
                "cameras": [c.to_dict() for c in self.cameras],
                "settings": self.settings.__dict__.copy()}

    @classmethod
    def from_dict(cls, d: dict) -> "TabState":
        s = SceneSettings()
        for k, v in d.get("settings", {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return cls(name=d.get("name", "Untitled tab"),
                   models=[BoneModel.from_dict(m) for m in d.get("models", [])],
                   cameras=[XrayCamera.from_dict(c) for c in d.get("cameras", [])], settings=s)


def save_session(state: TabState, path: str | Path) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), indent=2))


def load_session(path: str | Path) -> TabState:
    return TabState.from_dict(json.loads(Path(path).read_text()))
