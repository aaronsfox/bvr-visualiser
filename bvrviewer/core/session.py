"""What a tab holds, and saving/loading it as a JSON session file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import BoneModel

SESSION_VERSION = 2              # 2: trials (Visualiser #16)


@dataclass
class XrayCamera:
    calibration_path: Path
    source_kind: str                 # "folder", "files" or "multipage"
    source: list[Path]               # the folder, the files, or the single TIFF
    first_image: int = 0             # image shown at tracking frame 1: always the first (setting removed 10 Nov 2026)
    visible: bool = True
    name: str = "X-ray"
    show_plane: bool = True          # radiograph on its detector plane in the 3D scene
    show_source: bool = False        # X-ray source marker (off for new cameras: Visualiser #14)
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
    def from_dict(cls, d: dict) -> XrayCamera:
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
    def from_dict(cls, d: dict) -> KinItem:
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


@dataclass(eq=False)
class Trial:
    """One trial in a Visualiser tab: its bones (with tracking), X-ray cameras and joints (Visualiser #16).

    Each trial keeps its own bones, model and cameras, so trials from different participants can sit in one
    tab. ``fps``, ``hold_still`` and ``frame`` are the trial's own scene settings, restored when it's shown.
    """

    name: str = "Trial 1"
    models: list[BoneModel] = field(default_factory=list)
    cameras: list[XrayCamera] = field(default_factory=list)
    joints: list[KinItem] = field(default_factory=list)
    fps: float = 30.0
    hold_still: str | None = None
    frame: int = 0
    cfg_path: Path | None = None          # the Autoscoper .cfg it was imported from
    tracking_dir: Path | None = None      # the folder its tracking files were chosen from

    def to_dict(self) -> dict:
        return {"name": self.name, "models": [m.to_dict() for m in self.models],
                "cameras": [c.to_dict() for c in self.cameras], "joints": [j.to_dict() for j in self.joints],
                "fps": self.fps, "hold_still": self.hold_still, "frame": self.frame,
                "cfg_path": str(self.cfg_path) if self.cfg_path else None,
                "tracking_dir": str(self.tracking_dir) if self.tracking_dir else None}

    @classmethod
    def from_dict(cls, d: dict) -> Trial:
        return cls(name=d.get("name", "Trial 1"), models=[BoneModel.from_dict(m) for m in d.get("models", [])],
                   cameras=[XrayCamera.from_dict(c) for c in d.get("cameras", [])],
                   joints=[KinItem.from_dict(j) for j in d.get("joints", [])], fps=d.get("fps", 30.0),
                   hold_still=d.get("hold_still"), frame=int(d.get("frame", 0) or 0),
                   cfg_path=Path(d["cfg_path"]) if d.get("cfg_path") else None,
                   tracking_dir=Path(d["tracking_dir"]) if d.get("tracking_dir") else None)


class _NoTrialList(list):
    """Stands in for a list of the trial being shown when there is no trial: empty to read, and adding to it
    first adds a trial ("Trial 1"), so code written for one set of bones keeps working."""

    def __init__(self, state, key):
        super().__init__()
        self._state, self._key = state, key

    def _real(self) -> list:
        return getattr(self._state.trial or self._state.add_trial("Trial 1"), self._key)

    def append(self, x):
        self._real().append(x)

    def extend(self, xs):
        self._real().extend(xs)

    def insert(self, i, x):
        self._real().insert(i, x)


class TabState:
    """A Visualiser tab: its trials (one shown at a time) and the tab-wide scene settings.

    ``models``, ``cameras`` and ``joints`` are those of the trial being shown (when there is no trial they read
    as empty, and adding to them first adds "Trial 1"). Older code and sessions that hold one set
    of bones, cameras and joints become a single trial.
    """

    def __init__(self, name: str = "Untitled tab", models=None, cameras=None, joints=None,
                 settings: SceneSettings | None = None, trials: list[Trial] | None = None, active: int | None = None):
        self.name = name
        self.settings = settings or SceneSettings()
        self.trials: list[Trial] = list(trials or [])
        if models or cameras or joints:
            self.trials.append(Trial(name="Trial 1", models=list(models or []), cameras=list(cameras or []),
                                     joints=list(joints or []), fps=self.settings.fps,
                                     hold_still=self.settings.hold_still))
        self.active = (0 if self.trials else -1) if active is None else active
        self.originals: dict = {}      # files used from a session's data folder -> where they came from

    # ---- the trial being shown
    @property
    def trial(self) -> Trial | None:
        return self.trials[self.active] if 0 <= self.active < len(self.trials) else None

    def _list(self, key):
        t = self.trial
        return getattr(t, key) if t is not None else _NoTrialList(self, key)

    def _set_list(self, key, value):
        setattr(self.trial or self.add_trial("Trial 1"), key, list(value))

    models = property(lambda self: self._list("models"), lambda self, v: self._set_list("models", v))
    cameras = property(lambda self: self._list("cameras"), lambda self, v: self._set_list("cameras", v))
    joints = property(lambda self: self._list("joints"), lambda self, v: self._set_list("joints", v))

    def unique_trial_name(self, name: str) -> str:
        names = {t.name for t in self.trials}
        if name not in names:
            return name
        k = 2
        while f"{name} ({k})" in names:
            k += 1
        return f"{name} ({k})"

    def add_trial(self, name: str = "Trial 1", trial: Trial | None = None) -> Trial:
        """Add a trial (a new empty one called ``name``, or ``trial``); the first one becomes the one shown."""
        t = trial or Trial(name=name, fps=self.settings.fps)
        t.name = self.unique_trial_name(t.name)
        self.trials.append(t)
        self.active = max(self.active, 0)
        return t

    def trial_of(self, obj) -> Trial | None:
        """The trial holding a bone, camera, joint or trial."""
        for t in self.trials:
            if obj is t or any(obj is x for x in (*t.models, *t.cameras, *t.joints)):
                return t
        return None

    # ---- files
    def to_dict(self) -> dict:
        t = self.trial
        if t is not None:                     # the shown trial's settings live in the scene settings
            t.fps, t.hold_still = self.settings.fps, self.settings.hold_still
        return {"version": SESSION_VERSION, "tool": "visualiser", "name": self.name,
                "trials": [x.to_dict() for x in self.trials], "active": self.active,
                "settings": self.settings.__dict__.copy()}

    @classmethod
    def from_dict(cls, d: dict) -> TabState:
        s = SceneSettings()
        for k, v in d.get("settings", {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        name = d.get("name", "Untitled tab")
        if "trials" in d:
            trials = [Trial.from_dict(t) for t in d["trials"]]
            active = d.get("active", 0) if trials else -1
        elif d.get("models") or d.get("cameras") or d.get("joints"):
            # version 1: one set of bones, cameras and joints -> one trial named after the session
            trials = [Trial(name=name, models=[BoneModel.from_dict(m) for m in d.get("models", [])],
                            cameras=[XrayCamera.from_dict(c) for c in d.get("cameras", [])],
                            joints=[KinItem.from_dict(j) for j in d.get("joints", [])], fps=s.fps,
                            hold_still=s.hold_still)]
            active = 0
        else:
            trials, active = [], -1
        st = cls(name=name, settings=s, trials=trials, active=min(active, len(trials) - 1))
        if st.trial is not None:
            s.fps, s.hold_still = st.trial.fps, st.trial.hold_still
        return st


# ---------------------------------------------------------------- saving and opening
# Path fields of a session (dict form), with the data-folder subfolder each kind of file is copied to.
def _path_fields(d: dict):
    """(holder, key, kind) for every file path in a session dict (every trial's, or a version-1 session's)."""
    for t in d.get("trials", []):
        yield from _path_fields(t)
        if t.get("cfg_path"):
            yield t, "cfg_path", "config"
        if t.get("tracking_dir"):
            yield t, "tracking_dir", "folder"
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
        if v and kind not in ("radiographs", "saved radiographs", "folder") and Path(v) not in out:
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
    all_cams = [(t, c) for t in state.trials for c in t.cameras]
    cams = [c for _t, c in all_cams if c.images is not None] if radiographs else []
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
            if not v or kind in ("radiographs", "saved radiographs", "folder"):
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
    cam_dicts = [cd for td in d["trials"] for cd in td["cameras"]]
    several = len(state.trials) > 1
    for (trial, cam), cd in zip(all_cams, cam_dicts, strict=True):
        cd.pop("radiographs", None)
        if not any(cam is c for c in cams):
            continue
        stem = f"{_safe(trial.name)} - {_safe(cam.name)}" if several else _safe(cam.name)
        dst = data / "radiographs" / f"{stem}{SUFFIX}"
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
