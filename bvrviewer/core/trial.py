"""Setting up a trial from its files (Visualiser #17): an Autoscoper .cfg and a folder of tracking files.

``TrialDraft`` is what the Trial setup dialog edits: the trial's name, its bones (surface, colour, tracking
file) and X-ray cameras, with checks. ``draft_from_cfg`` fills one from a .cfg (its volumes and cameras, as
listed); ``match_tracking`` pairs tracking files with bones by name; ``TrialDraft.to_trial`` makes the
``Trial`` the Visualiser loads. Nothing here needs the GUI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import read_cfg
from .models import PALETTE, BoneModel
from .session import KinItem, Trial, XrayCamera

TRACKING_SUFFIXES = (".tra",)

# Display settings the Trial setup dialog sets for a trial (Visualiser #22); each trial has its own (#23)
DISPLAY_KEYS = ("xray_opacity", "xray_distance", "xray_bone_opacity", "camera_views", "overlay", "overlay_fill",
                "overlay_opacity", "point_density")


def display_of(settings) -> dict:
    return {k: getattr(settings, k) for k in DISPLAY_KEYS}


@dataclass
class BoneDraft:
    name: str
    path: Path
    color: str = PALETTE[0]
    include: bool = True
    voxel_size: tuple[float, float, float] | None = None
    flip: tuple[int, int, int] = (0, 0, 0)
    tracking_path: Path | None = None
    tracking_overrides: dict = field(default_factory=dict)
    tracking_volume: int = 0
    tracking_offset: int = 0
    axes_path: Path | None = None          # a Model Builder model giving this bone's coordinate system
    axes_segment: str | None = None
    source: object = None                  # the BoneModel it came from (Configure… / Duplicate)

    @property
    def is_volume(self) -> bool:
        return Path(self.path).suffix.lower() in (".tif", ".tiff")


@dataclass
class CameraDraft:
    name: str
    calibration_path: Path
    folder: Path | None = None             # radiograph folder (the .cfg's CameraRootDir)
    source_kind: str = "folder"
    source: list = field(default_factory=list)
    include: bool = True
    show_source: bool = False
    first_image: int = 0
    source_cam: object = None              # the XrayCamera it came from (Configure… / Duplicate)


@dataclass
class TrialDraft:
    name: str = "Trial 1"
    cfg_path: Path | None = None
    tracking_dir: Path | None = None
    bones: list[BoneDraft] = field(default_factory=list)
    cameras: list[CameraDraft] = field(default_factory=list)
    model_path: Path | None = None         # Model Builder model whose bones (matched by name) give axes and joints
    read_radiographs: bool = True          # read every radiograph when the trial is first shown
    joints: list = field(default_factory=list)   # joints to copy (from the trial a new one started from)
    display: dict = field(default_factory=dict)  # display settings for the trial (DISPLAY_KEYS)

    def tracking_files(self) -> list[Path]:
        return list_tracking(self.tracking_dir) if self.tracking_dir else []

    def match(self) -> None:
        """Give each included bone the tracking file that matches its name (bones that have one keep it)."""
        free = [b for b in self.bones if b.tracking_path is None]
        taken = {Path(b.tracking_path) for b in self.bones if b.tracking_path is not None}
        files = [f for f in self.tracking_files() if f not in taken]
        for b, f in match_tracking([b.name for b in free], files).items():
            next(x for x in free if x.name == b).tracking_path = f

    def unused_tracking(self) -> list[Path]:
        used = {Path(b.tracking_path) for b in self.bones if b.include and b.tracking_path}
        return [f for f in self.tracking_files() if f not in used]

    def problems(self) -> list[str]:
        """What stops the trial being built as it is (files missing, voxel sizes, duplicate names)."""
        out = []
        names = [b.name for b in self.bones if b.include]
        for n in sorted({n for n in names if names.count(n) > 1}):
            out.append(f"Two bones are called “{n}”: rename one.")
        for b in self.bones:
            if not b.include:
                continue
            if not Path(b.path).exists():
                out.append(f"{b.name}: {Path(b.path).name} wasn't found.")
            if b.is_volume and not b.voxel_size:
                out.append(f"{b.name}: a TIF volume needs its voxel size.")
            if b.tracking_path is not None and not Path(b.tracking_path).exists():
                out.append(f"{b.name}: tracking file {Path(b.tracking_path).name} wasn't found.")
        for c in self.cameras:
            if not c.include:
                continue
            if not Path(c.calibration_path).exists():
                out.append(f"{c.name}: calibration {Path(c.calibration_path).name} wasn't found.")
            if c.source_kind == "folder" and c.folder is not None and not Path(c.folder).is_dir():
                out.append(f"{c.name}: radiograph folder {Path(c.folder).name} wasn't found.")
        return out

    def to_trial(self) -> Trial:
        """The trial to load: its bones, cameras (radiographs read when it is first shown) and joints."""
        models = [bone_model(b) for b in self.bones if b.include]
        cams = [camera(c) for c in self.cameras if c.include]
        names = {m.name for m in models}
        joints = [KinItem.from_dict(j.to_dict()) for j in self.joints
                  if all(n in names for n in j.model_map.values())]
        return Trial(name=self.name, models=models, cameras=cams, joints=joints, cfg_path=self.cfg_path,
                     tracking_dir=self.tracking_dir)


def same_surface(b: BoneDraft, m: BoneModel) -> bool:
    """Whether a bone's surface is the one a loaded bone already has (so it needn't be built again)."""
    return (Path(b.path) == Path(m.path) and tuple(b.voxel_size or ()) == tuple(m.voxel_size or ())
            and tuple(b.flip) == tuple(m.flip))


def bone_model(b: BoneDraft) -> BoneModel:
    """A new bone (not yet built) from a draft row; a surface already built for its source bone is reused."""
    m = BoneModel(name=b.name, path=Path(b.path), color=b.color, voxel_size=b.voxel_size, flip=tuple(b.flip),
                  tracking_path=Path(b.tracking_path) if b.tracking_path else None,
                  tracking_volume=b.tracking_volume, tracking_overrides=dict(b.tracking_overrides),
                  tracking_offset=b.tracking_offset)
    src = b.source
    if src is not None:                 # keep what the bone it came from had set
        for key in ("opacity", "visible", "display", "threshold", "tfm_path", "axes_path", "axes_segment",
                    "axes_visible", "axes_reference", "axes_zero"):
            setattr(m, key, getattr(src, key))
        if same_surface(b, src) and src.mesh is not None:
            m.mesh, m.image_data, m.surface_note = src.mesh, src.image_data, src.surface_note
    if b.axes_path is not None:
        m.axes_path, m.axes_segment = Path(b.axes_path), b.axes_segment
    return m


def camera_source(c: CameraDraft) -> list:
    return [Path(c.folder)] if c.source_kind == "folder" and c.folder is not None else [Path(p) for p in c.source]


def camera(c: CameraDraft) -> XrayCamera:
    cam = XrayCamera(calibration_path=Path(c.calibration_path), source_kind=c.source_kind, source=camera_source(c),
                     name=c.name, show_source=c.show_source)
    src = c.source_cam
    if src is not None:
        for key in ("visible", "first_image", "show_plane", "show_beams"):
            setattr(cam, key, getattr(src, key))
    return cam


# ---------------------------------------------------------------- .cfg
def draft_from_cfg(path: str | Path, tracking_dir: str | Path | None = None) -> TrialDraft:
    """A trial from an Autoscoper .cfg: a bone for each volume it lists (the TIF, with the .cfg's voxel size
    and flip) and an X-ray camera for each calibration and radiograph folder. With ``tracking_dir``, each
    bone is given the tracking file whose name matches it."""
    path = Path(path)
    cfg = read_cfg(path)
    bones = []
    for k, v in enumerate(cfg.volumes):
        bones.append(BoneDraft(name=Path(v.path).stem, path=Path(v.path), color=PALETTE[k % len(PALETTE)],
                               voxel_size=v.voxel_size, flip=tuple(v.flip)))
    cams = [CameraDraft(name=f"X-ray {k}", calibration_path=Path(cal), folder=Path(d))
            for k, (cal, d) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1)]
    d = TrialDraft(name=path.stem, cfg_path=path, bones=bones, cameras=cams)
    if tracking_dir is not None:
        set_tracking_dir(d, tracking_dir)
    return d


def set_tracking_dir(d: TrialDraft, folder: str | Path | None) -> None:
    """Use this folder's tracking files: bones are matched to them again, and the trial is named after them
    when its name is still the default."""
    old_guess = trial_name_from_files(d.tracking_files()) if d.tracking_dir else None
    d.tracking_dir = Path(folder) if folder else None
    for b in d.bones:
        if b.tracking_path is not None and (d.tracking_dir is None or Path(b.tracking_path).parent != d.tracking_dir):
            b.tracking_path = None
    d.match()
    guess = trial_name_from_files(d.tracking_files())
    default = Path(d.cfg_path).stem if d.cfg_path else None
    if guess and d.name in (default, old_guess, "Trial 1", ""):
        d.name = guess


def guess_tracking_dir(cfg_path: str | Path) -> Path | None:
    """The folder next to a .cfg most likely to hold its tracking files ("Tracking", "Tracking_SAM", …)."""
    base = Path(cfg_path).parent
    dirs = [p for p in base.iterdir() if p.is_dir() and list_tracking(p)] if base.is_dir() else []
    dirs.sort(key=lambda p: (not p.name.lower().startswith("track"), p.name.lower()))
    return dirs[0] if dirs else None


# ---------------------------------------------------------------- tracking files
def list_tracking(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TRACKING_SUFFIXES),
                  key=lambda p: p.name.lower())


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def _common_affixes(stems: list[str]) -> tuple[str, str]:
    """The prefix and suffix every name shares, cut back to a separator (so 'tib'/'tal' keep their 't')."""
    if len(stems) < 2:
        return "", ""
    pre = os.path.commonprefix(stems)
    m = re.match(r"^(.*[^a-zA-Z0-9])", pre)
    pre = m.group(1) if m else ""
    suf = os.path.commonprefix([s[::-1] for s in stems])[::-1]
    m = re.search(r"([^a-zA-Z0-9].*)$", suf)
    suf = m.group(1) if m else ""
    return pre, suf


def tracking_keys(files: list[Path]) -> dict[Path, str]:
    """Each file's distinguishing part: the name without the prefix and suffix all the files share
    ('T0019_jog0001_tib.tra' -> 'tib')."""
    stems = [Path(f).stem for f in files]
    pre, suf = _common_affixes(stems)
    out = {}
    for f, s in zip(files, stems):
        k = s[len(pre):len(s) - len(suf) if suf else None]
        out[Path(f)] = k or s
    return out


def _score(bone: str, key: str, stem: str) -> int:
    b, k = bone.lower(), key.lower()
    if b == k:
        return 100
    bw, kw, sw = _words(bone), _words(key), _words(stem)
    if bw and bw == kw:
        return 90
    if kw and set(kw) <= set(bw):          # 'Femur' in 'Left_Femur'
        return 70 + len(kw)
    if bw and set(bw) <= set(sw):          # 'tib' in 'T0019_jog0001_tib'
        return 60 + len(bw)
    shared = set(bw) & set(sw)
    if shared:
        return 30 + len(shared)
    if len(b) >= 3 and (b in k or k in b):
        return 20
    return 0


def match_tracking(bones: list[str], files: list[Path]) -> dict[str, Path]:
    """Pair bone names with tracking files by name, each file used once (best matches first). Bones with no
    plausible file are left out."""
    files = [Path(f) for f in files]
    keys = tracking_keys(files)
    pairs = sorted(((_score(b, keys[f], f.stem), i, j) for i, b in enumerate(bones) for j, f in enumerate(files)),
                   key=lambda x: (-x[0], x[1], x[2]))
    out: dict[str, Path] = {}
    used: set[int] = set()
    for score, i, j in pairs:
        if score <= 0:
            break
        if bones[i] in out or j in used:
            continue
        out[bones[i]] = files[j]
        used.add(j)
    return out


def match_segments(segments: list[tuple[str, str]], bones: list[str]) -> dict[str, str]:
    """Pair a model's segments, given as (segment name, its surface file's stem), with bone names, each used once
    ('Femur' / 'AUT_Left_Femur' -> 'Left_Femur')."""
    pairs = sorted(((max(_score(b, seg, stem), _score(b, stem, stem)), i, j)
                    for i, (seg, stem) in enumerate(segments) for j, b in enumerate(bones)),
                   key=lambda x: (-x[0], x[1], x[2]))
    out: dict[str, str] = {}
    used: set[int] = set()
    for score, i, j in pairs:
        if score <= 0:
            break
        if segments[i][0] in out or j in used:
            continue
        out[segments[i][0]] = bones[j]
        used.add(j)
    return out


def trial_name_from_files(files: list[Path]) -> str | None:
    """The trial's name from its tracking files: the part all their names share ('T0019_jog0001')."""
    if len(files) < 2:
        return None
    pre, _ = _common_affixes([Path(f).stem for f in files])
    name = pre.strip(" _-.")
    if not name or set(_words(name)) <= {"left", "right", "l", "r", "lt", "rt", "aut"}:
        return None                       # a side word alone ('Left_Femur', 'Left_Tibia') isn't a trial name
    return name


# ---------------------------------------------------------------- from an existing trial
def draft_from_trial(t: Trial) -> TrialDraft:
    """A draft of an existing trial (to change it, or as the start of another trial with the same bones)."""
    bones = [BoneDraft(name=m.name, path=Path(m.path), color=m.color, voxel_size=m.voxel_size, flip=tuple(m.flip),
                       tracking_path=Path(m.tracking_path) if m.tracking_path else None,
                       tracking_overrides=dict(m.tracking_overrides), tracking_volume=m.tracking_volume,
                       tracking_offset=m.tracking_offset, source=m) for m in t.models]
    cams = []
    for c in t.cameras:
        folder = Path(c.source[0]) if c.source_kind == "folder" and c.source else None
        cams.append(CameraDraft(name=c.name, calibration_path=Path(c.calibration_path), folder=folder,
                                source_kind=c.source_kind, source=list(c.source), show_source=c.show_source,
                                source_cam=c))
    return TrialDraft(name=t.name, cfg_path=t.cfg_path, tracking_dir=t.tracking_dir, bones=bones, cameras=cams,
                      display=display_of(t.settings))
