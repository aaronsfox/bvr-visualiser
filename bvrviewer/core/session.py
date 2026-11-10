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
    first_image: int = 0             # image shown at tracking frame 1: always the first (setting removed 10 Nov 2026)
    visible: bool = True
    name: str = "X-ray"
    show_plane: bool = True          # radiograph on its detector plane in the 3D scene
    show_source: bool = True         # X-ray source marker
    show_beams: bool = False         # lines from the source to the image corners
    store_path: Path | None = None   # display copies saved with the session (core.radiograph_store)

    # runtime only
    calibration: object = field(default=None, repr=False)
    images: object = field(default=None, repr=False)
    note: str = field(default="", repr=False)         # how the radiographs were found, if not as usual

    def build(self) -> None:
        """Load the calibration and the radiographs.

        With radiographs saved with the session, their copies are used when the originals are unchanged
        (or can't be found); if the originals have changed since, the originals are read instead.
        """
        from .calibration import load_calibration
        from .images import ImageSequence, find_images
        from .radiograph_store import StoreError, matches, read_meta

        self.calibration = load_calibration(self.calibration_path)
        self.note = ""
        store = Path(self.store_path) if self.store_path and Path(self.store_path).exists() else None
        meta = None
        if store is not None:
            try:
                meta = read_meta(store)
            except StoreError as e:
                self.note = f"The radiographs saved with the session couldn't be used: {e}"
                store = None
        files, multipage, missing = None, None, None
        try:
            if self.source_kind == "folder":
                files, _ = find_images(self.source[0])
            elif self.source_kind == "multipage":
                multipage = self.source[0]
                if not Path(multipage).exists():
                    raise FileNotFoundError(f"{multipage} wasn't found.")
            else:
                files = sorted(self.source)
                if not all(Path(f).exists() for f in files):
                    raise FileNotFoundError("Some of the image files weren't found.")
        except (OSError, ValueError) as e:
            missing = e
        if missing is not None:
            if store is None:
                raise missing
            self.images = ImageSequence(store=store, store_kind="session")
            self.note = "The original radiographs weren't found, so the copies saved with the session are shown."
            return
        if store is not None:
            seq = ImageSequence(files=files, multipage=multipage, store=store, store_kind="session")
            if matches(meta, seq):
                self.images = seq
                return
            seq.close()
            self.note = ("The original radiographs have changed since the session was saved, so they're read "
                         "again (the saved copies aren't used).")
        self.images = ImageSequence(files=files, multipage=multipage)

    def image_index(self, frame: int) -> int | None:
        i = self.first_image + frame
        return i if self.images is not None and 0 <= i < len(self.images) else None

    def to_dict(self) -> dict:
        return {"name": self.name, "calibration_path": str(self.calibration_path), "source_kind": self.source_kind,
                "source": [str(p) for p in self.source], "first_image": self.first_image, "visible": self.visible,
                "show_plane": self.show_plane, "show_source": self.show_source, "show_beams": self.show_beams,
                **({"radiographs": str(self.store_path)} if self.store_path else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "XrayCamera":
        return cls(calibration_path=Path(d["calibration_path"]), source_kind=d["source_kind"],
                   source=[Path(p) for p in d["source"]], first_image=0,
                   visible=d.get("visible", True), name=d.get("name", "X-ray"),
                   show_plane=d.get("show_plane", True), show_source=d.get("show_source", True),
                   show_beams=d.get("show_beams", False),
                   store_path=Path(d["radiographs"]) if d.get("radiographs") else None)


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
    camera_views: bool = False        # 2D camera panels beside the 3D view
    overlay: str = "silhouette"       # bones on those panels: "silhouette", "points" or "off"
    overlay_opacity: float = 0.5
    curve_labels: str = "axes"        # curve legends: "axes" (Angle X/Y/Z) or "anatomical" (Flexion, ...)


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
    originals: dict = field(default_factory=dict)  # files used from a session's data folder -> where they came from

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


# ---------------------------------------------------------------- saving and opening
# Path fields of a session (dict form), with the data-folder subfolder each kind of file is copied to.
def _path_fields(d: dict):
    """(holder, key, kind) for every file path in a session dict."""
    for m in d.get("models", []):
        yield m, "path", "bones"
        yield m, "tfm_path", "transforms"
        yield m, "tracking_path", "tracking"
        if m.get("axes"):
            yield m["axes"], "path", "models"
    for c in d.get("cameras", []):
        yield c, "calibration_path", "calibration"
        for i in range(len(c.get("source", []))):
            yield c["source"], i, "radiographs"
        if c.get("radiographs"):
            yield c, "radiographs", "saved radiographs"
    for j in d.get("joints", []):
        yield j, "model_path", "models"


def data_folder(path: str | Path) -> Path:
    """The folder saved alongside a session for its files: 'Trial.bvrsession.json' -> 'Trial_data'."""
    path = Path(path)
    stem = path.name
    for ext in (".bvrsession.json", ".json"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return path.parent / f"{stem}_data"


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in name).strip() or "camera"


def _rel(p: Path, base: Path) -> str:
    import os

    return Path(os.path.relpath(p, base)).as_posix()


def session_files(state: TabState) -> list[Path]:
    """The files (other than radiographs) that 'Include all files' copies, each once."""
    out = []
    for holder, key, kind in _path_fields(state.to_dict()):
        v = holder[key]
        if v and kind not in ("radiographs", "saved radiographs") and Path(v) not in out:
            out.append(Path(v))
    return out


def save_session(state: TabState, path: str | Path, radiographs: bool = False, all_files: bool = False,
                 progress=None, cancelled=None) -> dict:
    """Save a tab as a JSON session file.

    radiographs: also save each camera's radiographs (display copies, one compressed file per camera) in
    the session's data folder, so the session opens quickly anywhere, even without the originals.
    all_files: also copy the bones, volumes, transforms, tracking, calibration and model files there, so
    the session and its folder are self-contained. The session refers to copied files relative to its own
    folder, and keeps their original locations (used if a copy goes missing).

    progress(text, done, total) reports what's being saved; cancelled() returning True stops.
    Returns {"data_folder": Path or None, "files": n copied, "radiographs": n cameras, "bytes": n}.
    """
    import shutil

    from .radiograph_store import SUFFIX, write_store

    path = Path(path)
    base = path.parent
    data = data_folder(path)
    d = state.to_dict()
    originals: dict[str, str] = {}
    report = {"data_folder": None, "files": 0, "radiographs": 0, "bytes": 0}
    cams = [c for c in state.cameras if c.images is not None] if radiographs else []
    copies = session_files(state) if all_files else []
    total = len(copies) + sum(len(c.images) for c in cams)
    done = 0

    def tick(text, n=1):
        nonlocal done
        done += n
        if progress is not None:
            progress(text, done, total)
        if cancelled is not None and cancelled():
            raise InterruptedError("Saving was cancelled.")

    if all_files:
        where: dict[Path, Path] = {}
        used: set[Path] = set()
        for holder, key, kind in _path_fields(d):
            v = holder[key]
            if not v or kind in ("radiographs", "saved radiographs"):
                continue
            src = Path(v)
            if src not in where:
                dst = data / "files" / kind / src.name
                k = 2
                while dst in used:
                    dst = data / "files" / kind / f"{src.stem}_{k}{src.suffix}"
                    k += 1
                used.add(dst)
                if not src.exists():
                    raise FileNotFoundError(f"{src} wasn't found, so it can't be copied.")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not (dst.exists() and dst.resolve() == src.resolve()):
                    shutil.copy2(src, dst)
                report["files"] += 1
                report["bytes"] += dst.stat().st_size
                where[src] = dst
                tick(f"Copying {src.name}")
            rel = _rel(where[src], base)
            holder[key] = rel
            originals[rel] = state.originals.get(str(src), str(src))
    for cam, cd in zip(state.cameras, d["cameras"]):
        cd.pop("radiographs", None)
        if cam not in cams:
            continue
        dst = data / "radiographs" / f"{_safe(cam.name)}{SUFFIX}"
        cur = getattr(cam.images, "_store", None)
        if cur is not None and cam.images.store_kind == "session" and cur.path.resolve() == dst.resolve():
            tick(f"{cam.name}: already saved", len(cam.images))          # re-saving the same session
        else:
            n0 = done
            write_store(cam.images, dst, progress=lambda i, n, c=cam, n0=n0: tick(f"Saving {c.name} radiographs",
                                                                                     n0 + i - done),
                        cancelled=cancelled)
        report["radiographs"] += 1
        report["bytes"] += dst.stat().st_size
        cd["radiographs"] = _rel(dst, base)
    if originals:
        d["originals"] = originals
    if report["files"] or report["radiographs"]:
        report["data_folder"] = data
    path.write_text(json.dumps(d, indent=2))
    return report


def load_session(path: str | Path) -> TabState:
    """Open a session file. Paths relative to the session (files in its data folder) are resolved; a
    copied file that has gone missing falls back to where it originally came from."""
    path = Path(path)
    d = json.loads(path.read_text())
    originals = d.get("originals", {})
    used: dict[str, str] = {}
    for holder, key, _kind in _path_fields(d):
        v = holder[key]
        if not v or Path(v).is_absolute():
            continue
        full = path.parent / v
        if not full.exists() and v in originals:
            holder[key] = originals[v]
        else:
            holder[key] = str(full)
            if v in originals:
                used[str(full)] = originals[v]
    st = TabState.from_dict(d)
    st.originals = used
    return st
