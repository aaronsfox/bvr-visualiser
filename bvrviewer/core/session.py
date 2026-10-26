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


@dataclass(eq=False)
class KinItem:
    """A segment or joint from a saved model (.bvrmodel.json), shown in a Visualiser tab.

    model_map maps the model's segment names to the names of this tab's models.
    """

    model_path: Path
    kind: str                                         # "segment" or "joint"
    name: str                                         # the segment's or joint's name in the model
    model_map: dict = field(default_factory=dict)     # segment name -> tab model name
    label: str = ""
    visible: bool = True                              # draw the axes in the 3D view
    use_reference: bool = True                        # joints: relative to the model's reference pose, if any
    zero_frame: int | None = None                     # subtract the values at this trial frame (0-based)
    seg_reference: object = "first"                   # segments: "first", "lab" or a frame index

    # runtime
    model: object = None                              # core.joints.KinematicModel (surfaces not rebuilt)

    def to_dict(self) -> dict:
        return {"model_path": str(self.model_path), "kind": self.kind, "name": self.name,
                "model_map": dict(self.model_map), "label": self.label, "visible": self.visible,
                "use_reference": self.use_reference, "zero_frame": self.zero_frame,
                "seg_reference": self.seg_reference}

    @classmethod
    def from_dict(cls, d: dict) -> "KinItem":
        if "joint_key" in d:                          # version-1 sessions held joints of a template model
            names = {"femur": "Femur", "tibia": "Tibia", "proximal": "Proximal bone", "distal": "Distal bone"}
            return cls(model_path=Path(d["model_path"]), kind="joint",
                       name={"knee": "Knee", "joint": "Joint"}.get(d["joint_key"], d["joint_key"]),
                       model_map={names.get(r, r): n for r, n in d.get("role_map", {}).items()},
                       label=d.get("name", ""), visible=d.get("visible", True), zero_frame=d.get("reference_frame"))
        return cls(model_path=Path(d["model_path"]), kind=d["kind"], name=d["name"], model_map=d.get("model_map", {}),
                   label=d.get("label", ""), visible=d.get("visible", True),
                   use_reference=d.get("use_reference", True), zero_frame=d.get("zero_frame"),
                   seg_reference=d.get("seg_reference", "first"))

    def segment_names(self) -> list[str]:
        if self.kind == "segment":
            return [self.name]
        j = self.model.joint(self.name)
        return [j.proximal, j.distal]

    def tab_models(self) -> list:
        return [self.model_map.get(n) for n in self.segment_names()]


@dataclass
class TabState:
    name: str = "Untitled tab"
    models: list[BoneModel] = field(default_factory=list)
    cameras: list[XrayCamera] = field(default_factory=list)
    settings: SceneSettings = field(default_factory=SceneSettings)
    joints: list[KinItem] = field(default_factory=list)     # segments and joints from models

    def to_dict(self) -> dict:
        return {"version": SESSION_VERSION, "tool": "visualiser", "name": self.name,
                "models": [m.to_dict() for m in self.models],
                "cameras": [c.to_dict() for c in self.cameras],
                "joints": [j.to_dict() for j in self.joints],
                "settings": self.settings.__dict__.copy()}

    @classmethod
    def from_dict(cls, d: dict) -> "TabState":
        s = SceneSettings()
        for k, v in d.get("settings", {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return cls(name=d.get("name", "Untitled tab"),
                   models=[BoneModel.from_dict(m) for m in d.get("models", [])],
                   cameras=[XrayCamera.from_dict(c) for c in d.get("cameras", [])], settings=s,
                   joints=[KinItem.from_dict(j) for j in d.get("joints", [])])


def save_session(state: TabState, path: str | Path) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), indent=2))


def load_session(path: str | Path) -> TabState:
    return TabState.from_dict(json.loads(Path(path).read_text()))
