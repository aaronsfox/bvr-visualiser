"""Assessor evaluation: how far raters' tracking is from a criterion, and from each other.

Every analysis is a comparison of a *test* tracking with a *reference* tracking of the same bone in the
same trial:
- validity:     each rater's tracking (test) against the criterion (reference), e.g. bead-based poses;
- inter-rater:  one rater against another (every pair);
- intra-rater:  one rater's session against another session of the same rater (every pair).

Per frame, the difference between the two poses is the bone-frame transform

    dT = inv(T_ref) · T_test        (both map bone space -> X-ray/world space)

as in Welte et al. 2022 (JoVE e63535) and Williams et al. 2026 (Med Eng Phys 147:02NT01). It is reported as:
- helical rotation: the angle of dT about its helical (screw) axis, in degrees;
- helical translation: the absolute translation along that axis, in mm. This is what Welte and Williams
  report, but note that it is NOT the bone's displacement: with any rotation error, translation across the
  axis shows up as a shift of the axis instead (so it can be ~0 while the bone is millimetres away);
- surface distance: mean (and max) distance between the bone's surface points under the two poses, in mm,
  i.e. how far the surface actually is from where it should be;
- centroid displacement: distance between the bone's surface centroid under the two poses, in mm;
- bone axes (optional, when the bone has a coordinate system from a model): dT expressed in the bone's
  axes, as rotations about Z, X, Y (Z-X-Y Cardan, ISB) and the displacement of the axes' origin along X, Y, Z.

Only frames tracked in both trackings are compared. Statistics follow the papers reviewed in
claude/assessor-evaluation-review.md (bias, SD, Bland-Altman limits with CIs, MAE, RMSE, median/IQR,
robust limits of Williams 2026, median absolute deviation, cumulative error, Hampel outliers, and the
share of frames over a threshold, Welte 2022: 2 deg / 1 mm).

The helical decomposition is implemented from its definition (Panjabi 1982; Spoor & Veldpaus 1980), not
taken from Welte et al.'s (AGPL) code.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .joints import decompose
from .tracking import Tracking, TrackingError, load_tracking

CRITERION = "Criterion"


# ---------------------------------------------------------------- measures

@dataclass(frozen=True)
class Measure:
    key: str
    label: str              # long name (tables, axis titles)
    short: str              # short name (plot titles)
    unit: str               # "deg" or "mm"
    signed: bool = False    # signed components have a meaningful bias; magnitudes are >= 0
    kind: str = "rotation"  # "rotation" or "translation": which threshold applies
    needs: str = ""         # "", "surface" or "axes"


MEASURES: dict[str, Measure] = {m.key: m for m in [
    Measure("helical_rotation", "Rotation about the helical axis", "Helical rotation", "deg"),
    Measure("helical_translation", "Translation along the helical axis", "Helical translation", "mm",
            kind="translation"),
    Measure("surface_mean", "Mean surface distance", "Surface distance (mean)", "mm", kind="translation",
            needs="surface"),
    Measure("surface_max", "Maximum surface distance", "Surface distance (max)", "mm", kind="translation",
            needs="surface"),
    Measure("centroid", "Centroid displacement", "Centroid displacement", "mm", kind="translation",
            needs="surface"),
    Measure("axes_rz", "Rotation about Z (bone axes)", "Rotation Z", "deg", signed=True, needs="axes"),
    Measure("axes_rx", "Rotation about X (bone axes)", "Rotation X", "deg", signed=True, needs="axes"),
    Measure("axes_ry", "Rotation about Y (bone axes)", "Rotation Y", "deg", signed=True, needs="axes"),
    Measure("axes_tx", "Translation along X (bone axes)", "Translation X", "mm", signed=True,
            kind="translation", needs="axes"),
    Measure("axes_ty", "Translation along Y (bone axes)", "Translation Y", "mm", signed=True,
            kind="translation", needs="axes"),
    Measure("axes_tz", "Translation along Z (bone axes)", "Translation Z", "mm", signed=True,
            kind="translation", needs="axes"),
]}
DEFAULT_MEASURES = ("helical_rotation", "helical_translation", "surface_mean")


def unit_symbol(unit: str) -> str:
    return "°" if unit == "deg" else unit


# ---------------------------------------------------------------- pose differences

def pose_difference(T_test: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """dT = inv(T_ref) @ T_test for stacks of 4x4 poses (n, 4, 4): the test pose seen from the reference."""
    R_ref, t_ref = T_ref[:, :3, :3], T_ref[:, :3, 3]
    Rt = np.transpose(R_ref, (0, 2, 1))
    dT = np.zeros_like(T_test)
    dT[:, :3, :3] = Rt @ T_test[:, :3, :3]
    dT[:, :3, 3] = np.einsum("nij,nj->ni", Rt, T_test[:, :3, 3] - t_ref)
    dT[:, 3, 3] = 1.0
    return dT


def helical(dT: np.ndarray):
    """Helical (screw) parameters of each transform: rotation angle (deg, 0-180), unit axis (n, 3) and
    signed translation along the axis (mm). A pure translation has its axis along the translation."""
    rotvec = Rotation.from_matrix(dT[:, :3, :3]).as_rotvec()
    phi = np.linalg.norm(rotvec, axis=1)
    t = dT[:, :3, 3]
    axis = np.zeros_like(t)
    rot = phi > 1e-9
    axis[rot] = rotvec[rot] / phi[rot, None]
    tn = np.linalg.norm(t, axis=1)
    pure = ~rot & (tn > 0)
    axis[pure] = t[pure] / tn[pure, None]
    along = np.einsum("ni,ni->n", t, axis)
    return np.degrees(phi), axis, along


def surface_distances(dT: np.ndarray, points: np.ndarray):
    """Mean and max distance (n,) between the points under dT and where they should be (identity)."""
    moved = np.einsum("nij,pj->npi", dT[:, :3, :3], points) + dT[:, None, :3, 3]
    d = np.linalg.norm(moved - points[None], axis=2)
    return d.mean(axis=1), d.max(axis=1)


def axes_components(dT: np.ndarray, acs: np.ndarray):
    """dT in the bone's axes: rotations about Z, X, Y (Z-X-Y Cardan, deg) and origin displacement (X, Y, Z)."""
    inv = np.linalg.inv(acs)
    dA = inv[None] @ dT @ acs[None]
    ang = np.degrees(np.array([decompose(m[:3, :3], "ZXY") for m in dA]))     # (n, 3): about Z, X, Y
    return ang, dA[:, :3, 3]


def measure_values(T_test: np.ndarray, T_ref: np.ndarray, points: np.ndarray | None = None,
                   acs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every available measure, per frame, for frames where both poses are given (no NaN)."""
    dT = pose_difference(T_test, T_ref)
    phi, _axis, along = helical(dT)
    out = {"helical_rotation": phi, "helical_translation": np.abs(along)}
    if points is not None and len(points):
        mean, mx = surface_distances(dT, points)
        out["surface_mean"], out["surface_max"] = mean, mx
        c = points.mean(axis=0)
        out["centroid"] = np.linalg.norm(np.einsum("nij,j->ni", dT[:, :3, :3], c) + dT[:, :3, 3] - c, axis=1)
    if acs is not None:
        ang, tr = axes_components(dT, acs)
        out["axes_rz"], out["axes_rx"], out["axes_ry"] = ang[:, 0], ang[:, 1], ang[:, 2]
        out["axes_tx"], out["axes_ty"], out["axes_tz"] = tr[:, 0], tr[:, 1], tr[:, 2]
    return out


# ---------------------------------------------------------------- statistics

STAT_COLUMNS = [
    ("n", "Frames"), ("mean", "Mean"), ("sd", "SD"), ("median", "Median"), ("q1", "Q1"), ("q3", "Q3"),
    ("mae", "Mean abs."), ("rmse", "RMS"), ("max", "Max"),
    ("loa_low", "LoA low"), ("loa_high", "LoA high"), ("loa_ci", "LoA 95% CI ±"),
    ("robust_low", "Robust LoA low"), ("robust_high", "Robust LoA high"),
    ("mad", "Median abs. deviation"), ("cumulative", "Cumulative"),
    ("outliers", "Hampel outliers"), ("over", "Over threshold"), ("pct_over", "% over threshold"),
]


def statistics(x: np.ndarray, threshold: float | None = None, signed: bool = False) -> dict:
    """Summary statistics of per-frame values.

    - mean (= bias for signed components), SD, Bland-Altman limits of agreement mean ± 1.96 SD and the
      half-width of their 95% CI, 1.96·sqrt(3·SD²/n) (Bland & Altman 1986);
    - mean absolute value, RMS, max |x|;
    - median, Q1, Q3, robust limits median ± 1.45·IQR (Williams 2026; lower limit floored at 0 for
      magnitudes), median absolute deviation, cumulative error (sum of |x|);
    - Hampel outliers: |x - median| > 3.5 × 1.4826 × MAD;
    - frames with |x| above the threshold, and their share.
    Frames are treated as independent observations, which they aren't (neighbouring frames are correlated),
    so the CIs are optimistic.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    med = float(np.median(x))
    q1, q3 = (float(v) for v in np.percentile(x, [25, 75]))
    iqr = q3 - q1
    mad = float(np.median(np.abs(x - med)))
    ax = np.abs(x)
    rlow = med - 1.45 * iqr
    out = {
        "n": n, "mean": mean, "sd": sd, "median": med, "q1": q1, "q3": q3,
        "mae": float(ax.mean()), "rmse": float(np.sqrt(np.mean(x ** 2))), "max": float(ax.max()),
        "loa_low": mean - 1.96 * sd, "loa_high": mean + 1.96 * sd,
        "loa_ci": 1.96 * float(np.sqrt(3 * sd ** 2 / n)) if n > 1 else float("nan"),
        "robust_low": rlow if signed else max(rlow, 0.0), "robust_high": med + 1.45 * iqr,
        "mad": mad, "cumulative": float(ax.sum()),
        "outliers": int(np.sum(np.abs(x - med) > 3.5 * 1.4826 * mad)) if mad > 0 else 0,
    }
    if threshold is not None:
        over = int(np.sum(ax > threshold))
        out["over"], out["pct_over"] = over, 100.0 * over / n
    return out


# ---------------------------------------------------------------- the project

@dataclass(eq=False)
class TrackingEntry:
    """One tracking file: one bone, in one trial, by one rater in one session (or the criterion)."""
    path: Path
    trial: str = "Trial 1"
    bone: str = ""
    rater: str = ""
    session: str = "1"
    criterion: bool = False
    overrides: dict = field(default_factory=dict)
    volume: int = 0
    offset: int = 0
    tracking: Tracking | None = None
    error: str = ""

    @property
    def who(self) -> str:
        """How this tracking is named in results: the rater (plus session when it matters)."""
        return CRITERION if self.criterion else self.rater

    def load(self) -> None:
        try:
            self.tracking = load_tracking(self.path, self.volume, self.overrides, self.offset)
            self.error = ""
        except (TrackingError, OSError) as e:
            self.tracking, self.error = None, str(e)

    def to_dict(self) -> dict:
        return {"path": str(self.path), "trial": self.trial, "bone": self.bone, "rater": self.rater,
                "session": self.session, "criterion": self.criterion, "overrides": self.overrides,
                "volume": self.volume, "offset": self.offset}

    @classmethod
    def from_dict(cls, d: dict) -> TrackingEntry:
        return cls(path=Path(d["path"]), trial=d.get("trial", "Trial 1"), bone=d.get("bone", ""),
                   rater=d.get("rater", ""), session=str(d.get("session", "1")),
                   criterion=bool(d.get("criterion", False)), overrides=d.get("overrides", {}),
                   volume=int(d.get("volume", 0)), offset=int(d.get("offset", 0)))


@dataclass(eq=False)
class BoneInfo:
    """A bone's surface (for surface distance) and optional coordinate system (for bone-axes errors).

    `trial` empty = the same surface for every trial (one participant); otherwise only for that trial.
    """
    name: str
    trial: str = ""
    surface: object = None                  # core.models.BoneModel, built, or None
    axes_path: str | None = None
    axes_segment: str | None = None
    acs: np.ndarray | None = None
    _points: np.ndarray | None = None

    def points(self, max_points: int = 4000) -> np.ndarray | None:
        """Surface points used for distances (a fixed random subset of the mesh vertices)."""
        if self._points is None and self.surface is not None and getattr(self.surface, "mesh", None) is not None:
            pts = np.asarray(self.surface.mesh.points, float)
            if len(pts) > max_points:
                pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
            self._points = pts
        return self._points

    def to_dict(self) -> dict:
        s = self.surface
        return {"name": self.name, "trial": self.trial, "surface": s.to_dict() if s is not None else None,
                "axes_path": self.axes_path, "axes_segment": self.axes_segment}


@dataclass(eq=False)
class Assessor:
    """An assessor (one rater in one session) or the reference (criterion), and the bones it has.

    Bones are matched across assessors by their label, so 'tib' under each assessor is the same bone.
    Each bone can have one tracking file per trial (TrackingEntry with this assessor's name, session and
    criterion flag).
    """
    name: str
    session: str = "1"
    criterion: bool = False
    bones: list = field(default_factory=list)       # bone labels, in order

    def key(self) -> tuple:
        return (True, "", "") if self.criterion else (False, self.name, self.session)

    def to_dict(self) -> dict:
        return {"name": self.name, "session": self.session, "criterion": self.criterion, "bones": list(self.bones)}

    @classmethod
    def from_dict(cls, d: dict) -> Assessor:
        return cls(name=d.get("name", ""), session=str(d.get("session", "1")),
                   criterion=bool(d.get("criterion", False)), bones=list(d.get("bones", [])))


def _entry_key(e: TrackingEntry) -> tuple:
    return (True, "", "") if e.criterion else (False, e.rater, e.session)


@dataclass
class AnalysisSettings:
    mode: str = "validity"                  # "validity", "inter" or "intra"
    measures: tuple = DEFAULT_MEASURES
    rotation_threshold: float = 2.0         # deg (Welte 2022)
    translation_threshold: float = 1.0      # mm
    inter_session: str = ""                 # inter-rater: which session ("" = each rater's first)
    pool: str = "none"                      # "none", "trials" (pool a bone's trials) or "all" (pool bones too)

    def threshold(self, m: Measure) -> float:
        return self.rotation_threshold if m.kind == "rotation" else self.translation_threshold


MODES = [("validity", "Raters vs criterion (validity)"), ("inter", "Between raters (inter-rater)"),
         ("intra", "Within raters, between sessions (intra-rater)")]


@dataclass(eq=False)
class Comparison:
    """Per-frame values of one test tracking against one reference tracking."""
    mode: str
    trial: str
    bone: str
    test: str               # label of the test tracking (rater, maybe with session)
    ref: str                # label of the reference tracking
    frames: np.ndarray      # trial frames (0-based) tracked in both
    values: dict            # measure key -> (n,) array, aligned with frames
    n_test: int = 0         # frames tracked in the test tracking
    n_ref: int = 0

    @property
    def label(self) -> str:
        return self.test if self.mode == "validity" else f"{self.test} vs {self.ref}"


@dataclass(eq=False)
class AssessmentProject:
    name: str = "Assessor evaluation"
    entries: list = field(default_factory=list)       # TrackingEntry
    bones: list = field(default_factory=list)         # BoneInfo (surface / axes per bone label)
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    pattern: str = ""
    assessors: list = field(default_factory=list)     # Assessor, the reference (if any) first

    # ---- lookups
    def trials(self) -> list[str]:
        return _unique(e.trial for e in self.entries) or ["Trial 1"]

    def bone_names(self) -> list[str]:
        return _unique([b.name for b in self.bones] + [x for a in self.assessors for x in a.bones]
                       + [e.bone for e in self.entries if e.bone])

    # ---- assessors and their bones
    @property
    def reference(self) -> Assessor | None:
        return next((a for a in self.assessors if a.criterion), None)

    def raters_list(self) -> list[Assessor]:
        return [a for a in self.assessors if not a.criterion]

    def display_name(self, a: Assessor) -> str:
        if a.criterion:
            return "Reference"
        multi = sum(o.name == a.name for o in self.raters_list()) > 1 or a.session != "1"
        return f"{a.name} · session {a.session}" if multi else a.name

    def assessor_of(self, e: TrackingEntry) -> Assessor | None:
        k = _entry_key(e)
        return next((a for a in self.assessors if a.key() == k), None)

    def entries_of(self, a: Assessor, bone: str | None = None) -> list[TrackingEntry]:
        k = a.key()
        return [e for e in self.entries if _entry_key(e) == k and (bone is None or e.bone == bone)]

    def entry(self, a: Assessor, bone: str, trial: str) -> TrackingEntry | None:
        return next((e for e in self.entries_of(a, bone) if e.trial == trial), None)

    def find_assessor(self, name: str, session: str = "1") -> Assessor | None:
        return next((a for a in self.raters_list() if a.name == name and a.session == session), None)

    def bone_info_for(self, name: str) -> BoneInfo:
        """The bone's surface/axes record (made if missing)."""
        for b in self.bones:
            if b.name == name and not b.trial:
                return b
        b = BoneInfo(name=name)
        self.bones.append(b)
        return b

    def add_assessor(self, name: str, session: str = "1", copy_bones_from: Assessor | None = None) -> Assessor:
        if self.find_assessor(name, session) is not None:
            raise ValueError(f"There is already an assessor {name}, session {session}.")
        a = Assessor(name=name, session=session, bones=list(copy_bones_from.bones) if copy_bones_from else [])
        self.assessors.append(a)
        return a

    def add_reference(self, copy_bones_from: Assessor | None = None) -> Assessor:
        if self.reference is not None:
            return self.reference
        a = Assessor(name=CRITERION, criterion=True, bones=list(copy_bones_from.bones) if copy_bones_from else [])
        self.assessors.insert(0, a)
        return a

    def add_bone(self, a: Assessor, label: str) -> None:
        label = label.strip()
        if not label:
            raise ValueError("A bone needs a label.")
        if label in a.bones:
            raise ValueError(f"{self.display_name(a)} already has a bone '{label}'.")
        a.bones.append(label)
        self.bone_info_for(label)

    def remove_bone(self, a: Assessor, label: str) -> None:
        if label in a.bones:
            a.bones.remove(label)
        self.entries = [e for e in self.entries if not (_entry_key(e) == a.key() and e.bone == label)]
        self._drop_unused_bones()

    def remove_assessor(self, a: Assessor) -> None:
        k = a.key()
        self.entries = [e for e in self.entries if _entry_key(e) != k]
        self.assessors.remove(a)
        self._drop_unused_bones()

    def rename_assessor(self, a: Assessor, name: str, session: str | None = None) -> None:
        if a.criterion:
            return
        session = a.session if session is None else session
        other = self.find_assessor(name, session)
        if other is not None and other is not a:
            raise ValueError(f"There is already an assessor {name}, session {session}.")
        for e in self.entries_of(a):
            e.rater, e.session = name, session
        a.name, a.session = name, session

    def relabel_bone(self, a: Assessor, old: str, new: str) -> None:
        """Change the label of one assessor's bone (e.g. to match the other assessors')."""
        new = new.strip()
        if new == old:
            return
        if new in a.bones:
            raise ValueError(f"{self.display_name(a)} already has a bone '{new}'.")
        a.bones[a.bones.index(old)] = new
        for e in self.entries_of(a, old):
            e.bone = new
        self.bone_info_for(new)
        self._drop_unused_bones()

    def rename_bone(self, old: str, new: str) -> None:
        """Rename a bone label everywhere (every assessor, the reference and the bone's surface/axes)."""
        new = new.strip()
        if new == old:
            return
        if new in self.bone_names():
            raise ValueError(f"There is already a bone '{new}'.")
        for a in self.assessors:
            a.bones = [new if x == old else x for x in a.bones]
        for e in self.entries:
            if e.bone == old:
                e.bone = new
        for b in self.bones:
            if b.name == old:
                b.name = new

    def set_tracking(self, a: Assessor, bone: str, trial: str, path, overrides=None, volume: int = 0,
                     offset: int = 0) -> TrackingEntry:
        """Attach (or replace) the tracking file of an assessor's bone in a trial, and read it."""
        if bone not in a.bones:
            self.add_bone(a, bone)
        old = self.entry(a, bone, trial)
        if old is not None:
            self.entries.remove(old)
        e = TrackingEntry(path=Path(path), trial=trial, bone=bone, rater="" if a.criterion else a.name,
                          session="1" if a.criterion else a.session, criterion=a.criterion,
                          overrides=dict(overrides or {}), volume=volume, offset=offset)
        e.load()
        self.entries.append(e)
        return e

    def sync_assessors(self) -> None:
        """Make sure every tracking entry has its assessor and bone (e.g. after a folder import)."""
        for e in self.entries:
            a = self.assessor_of(e)
            if a is None:
                if e.criterion:
                    a = self.add_reference()
                elif e.rater:
                    a = Assessor(name=e.rater, session=e.session)
                    self.assessors.append(a)
                else:
                    continue
            if e.bone and e.bone not in a.bones:
                a.bones.append(e.bone)
            if e.bone:
                self.bone_info_for(e.bone)

    def _drop_unused_bones(self) -> None:
        used = set(self.bone_names_in_use())
        self.bones = [b for b in self.bones if b.name in used or b.surface is not None or b.axes_path]

    def bone_names_in_use(self) -> list[str]:
        return _unique([x for a in self.assessors for x in a.bones] + [e.bone for e in self.entries if e.bone])

    def raters(self) -> list[str]:
        return _unique([a.name for a in self.raters_list()]
                       + [e.rater for e in self.entries if not e.criterion and e.rater])

    def sessions(self) -> list[str]:
        return _unique([a.session for a in self.raters_list()] + [e.session for e in self.entries if not e.criterion])

    def bone_info(self, trial: str, bone: str) -> BoneInfo | None:
        for b in self.bones:
            if b.name == bone and b.trial == trial:
                return b
        for b in self.bones:
            if b.name == bone and not b.trial:
                return b
        return None

    def available_measures(self) -> list[str]:
        has_surface = any(b.surface is not None for b in self.bones)
        has_axes = any(b.acs is not None for b in self.bones)
        return [k for k, m in MEASURES.items()
                if not m.needs or (m.needs == "surface" and has_surface) or (m.needs == "axes" and has_axes)]

    # ---- checks
    def problems(self) -> list[str]:
        """Things that stop entries being compared or look suspicious."""
        out = []
        for e in self.entries:
            if e.error:
                out.append(f"{e.path.name} ({e.bone}): {e.error}")
            elif not e.bone:
                out.append(f"{e.path.name}: no bone set")
            elif not e.criterion and not e.rater:
                out.append(f"{e.path.name}: no rater set")
        keys = {}
        for e in self.entries:
            k = (e.trial, e.bone, e.who, "" if e.criterion else e.session)
            if k in keys:
                out.append(f"{e.path.name} and {keys[k].path.name} are both {e.who}"
                           + ("" if e.criterion else f", session {e.session}") + f", {e.bone}, {e.trial}")
            keys[k] = e
        if len(self.assessors) > 1:
            for bone in self.bone_names_in_use():
                having = [a for a in self.assessors if bone in a.bones]
                if len(having) == 1:
                    out.append(f"Only {self.display_name(having[0])} has a bone '{bone}', so it can't be compared: "
                               "give it the same label as the others' bone")
        for group in _groupby(self.entries, lambda e: (e.trial, e.bone)).values():
            spans = {(e.tracking.offset, e.tracking.n_rows) for e in group if e.tracking is not None}
            if len(spans) > 1:
                out.append(f"{group[0].bone}, {group[0].trial}: the files have different numbers of rows "
                           f"({', '.join(str(r) for _o, r in sorted(spans))}); rows are matched by number")
        return out

    # ---- comparisons
    def comparisons(self, settings: AnalysisSettings | None = None) -> list[Comparison]:
        s = settings or self.settings
        out = []
        for (trial, bone), group in _groupby(self.entries, lambda e: (e.trial, e.bone)).items():
            group = [e for e in group if e.tracking is not None and e.bone]
            crit = [e for e in group if e.criterion]
            raters = [e for e in group if not e.criterion and e.rater]
            multi_session = len({e.session for e in raters}) > 1
            pairs = []
            if s.mode == "validity":
                for c in crit[:1]:
                    pairs += [(e, c, _label(e, multi_session), CRITERION) for e in raters]
            elif s.mode == "inter":
                chosen = {}
                for e in sorted(raters, key=lambda e: _session_key(e.session)):
                    if s.inter_session and e.session != s.inter_session:
                        continue
                    chosen.setdefault(e.rater, e)
                for a, b in itertools.combinations(chosen.values(), 2):
                    pairs.append((a, b, a.rater, b.rater))
            elif s.mode == "intra":
                for mine in _groupby(raters, lambda e: e.rater).values():
                    mine = sorted(mine, key=lambda e: _session_key(e.session))
                    for a, b in itertools.combinations(mine, 2):
                        pairs.append((b, a, f"{b.rater} s{b.session}", f"{a.rater} s{a.session}"))
            info = self.bone_info(trial, bone)
            pts = info.points() if info is not None else None
            acs = info.acs if info is not None else None
            for test, ref, tl, rl in pairs:
                out.append(compare(test.tracking, ref.tracking, s.mode, trial, bone, tl, rl, pts, acs))
        order = {b: i for i, b in enumerate(self.bone_names())}
        trials = {t: i for i, t in enumerate(self.trials())}
        out.sort(key=lambda c: (trials.get(c.trial, 0), order.get(c.bone, 0)))
        return out

    # ---- file
    def to_dict(self) -> dict:
        st = self.settings
        return {"tool": "assessor", "version": 2, "name": self.name, "pattern": self.pattern,
                "assessors": [a.to_dict() for a in self.assessors],
                "entries": [e.to_dict() for e in self.entries], "bones": [b.to_dict() for b in self.bones],
                "settings": {"mode": st.mode, "measures": list(st.measures),
                             "rotation_threshold": st.rotation_threshold,
                             "translation_threshold": st.translation_threshold,
                             "inter_session": st.inter_session, "pool": st.pool}}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path, build_surfaces: bool = True) -> AssessmentProject:
        """Read a project file. Tracking files are read; surfaces are built when build_surfaces."""
        from .models import BoneModel

        d = json.loads(Path(path).read_text())
        if d.get("tool") != "assessor":
            raise ValueError("This isn't an Assessor Evaluation project file.")
        st = d.get("settings", {})
        p = cls(name=d.get("name", "Assessor evaluation"), pattern=d.get("pattern", ""),
                settings=AnalysisSettings(mode=st.get("mode", "validity"),
                                          measures=tuple(st.get("measures", DEFAULT_MEASURES)),
                                          rotation_threshold=float(st.get("rotation_threshold", 2.0)),
                                          translation_threshold=float(st.get("translation_threshold", 1.0)),
                                          inter_session=st.get("inter_session", ""), pool=st.get("pool", "none")))
        p.assessors = [Assessor.from_dict(a) for a in d.get("assessors", [])]
        for ed in d.get("entries", []):
            e = TrackingEntry.from_dict(ed)
            e.load()
            p.entries.append(e)
        for bd in d.get("bones", []):
            surf = BoneModel.from_dict(bd["surface"]) if bd.get("surface") else None
            if surf is not None and build_surfaces:
                try:
                    surf.build()
                except Exception:  # noqa: BLE001 - reported as a bone without a surface
                    surf = None
            b = BoneInfo(name=bd["name"], trial=bd.get("trial", ""), surface=surf,
                         axes_path=bd.get("axes_path"), axes_segment=bd.get("axes_segment"))
            load_bone_axes(b)
            p.bones.append(b)
        p.sync_assessors()                  # version 1 files had no assessors
        return p


def load_bone_axes(b: BoneInfo) -> None:
    """Read the bone's coordinate system from its model file (if it has one)."""
    b.acs = None
    if not b.axes_path or not b.axes_segment:
        return
    from .joints import KinematicModel

    try:
        acs = KinematicModel.load(b.axes_path, build_surfaces=False).segment(b.axes_segment).acs
    except Exception:  # noqa: BLE001 - shown as "no axes"
        acs = None
    b.acs = None if acs is None else np.asarray(acs, float)


def compare(test: Tracking, ref: Tracking, mode: str, trial: str, bone: str, test_label: str, ref_label: str,
            points=None, acs=None) -> Comparison:
    n = min(test.n_frames, ref.n_frames)
    both = test.valid[:n] & ref.valid[:n]
    frames = np.flatnonzero(both)
    vals = measure_values(test.poses[frames], ref.poses[frames], points, acs) if len(frames) else {}
    return Comparison(mode, trial, bone, test_label, ref_label, frames, vals,
                      int(test.valid.sum()), int(ref.valid.sum()))


def _label(e: TrackingEntry, multi_session: bool) -> str:
    return f"{e.rater} s{e.session}" if multi_session else e.rater


def _session_key(s: str):
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)


def _unique(items) -> list:
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _groupby(items, key) -> dict:
    out: dict = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


# ---------------------------------------------------------------- summaries

@dataclass
class SummaryRow:
    trial: str
    bone: str
    label: str              # comparison label (rater, or "A vs B")
    measure: str
    stats: dict
    frames_test: int = 0
    frames_ref: int = 0


def summarise(comps: list[Comparison], settings: AnalysisSettings) -> list[SummaryRow]:
    """Statistics per comparison and measure, pooled over trials (and bones) if the settings say so."""
    def group_key(c: Comparison):
        if settings.pool == "all":
            return ("All trials", "All bones", c.label)
        if settings.pool == "trials":
            return ("All trials", c.bone, c.label)
        return (c.trial, c.bone, c.label)

    rows = []
    for (trial, bone, label), group in _groupby(comps, group_key).items():
        for key in settings.measures:
            m = MEASURES[key]
            arrays = [c.values[key] for c in group if key in c.values]
            if not arrays:
                continue
            x = np.concatenate(arrays)
            rows.append(SummaryRow(trial, bone, label, key, statistics(x, settings.threshold(m), m.signed),
                                   sum(c.n_test for c in group), sum(c.n_ref for c in group)))
    return rows


def write_frames_csv(path, comps: list[Comparison], measures) -> None:
    """Long format: one row per comparison, frame and measure (for R, SPSS, ...)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "bone", "comparison", "test", "reference", "frame", "measure", "value", "unit"])
        for c in comps:
            for key in measures:
                if key not in c.values:
                    continue
                unit = MEASURES[key].unit
                for f, v in zip(c.frames, c.values[key]):
                    w.writerow([c.trial, c.bone, c.mode, c.test, c.ref, int(f) + 1, key, f"{v:.6f}", unit])


def write_summary_csv(path, rows: list[SummaryRow]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "bone", "comparison", "measure", "unit"] + [k for k, _ in STAT_COLUMNS])
        for r in rows:
            m = MEASURES[r.measure]
            w.writerow([r.trial, r.bone, r.label, r.measure, m.unit]
                       + [_fmt_csv(r.stats.get(k)) for k, _ in STAT_COLUMNS])


def _fmt_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.6g}"


# ---------------------------------------------------------------- importing files

TOKENS = ("trial", "rater", "session", "bone")


def pattern_regex(pattern: str) -> re.Pattern:
    """A file-name pattern like '{rater}/{trial}_{bone}' as a regex on the path (no extension) relative to
    the chosen folder. Tokens: {trial} {rater} {session} {bone}, and * for anything."""
    out, i = "", 0
    for m in re.finditer(r"\{(\w+)\}|\*", pattern):
        out += re.escape(pattern[i:m.start()])
        if m.group(0) == "*":
            out += r"[^/]*?"
        elif m.group(1) in TOKENS:
            out += rf"(?P<{m.group(1)}>[^/]+?)"
        else:
            raise ValueError(f"Unknown field {{{m.group(1)}}}: use {', '.join('{' + t + '}' for t in TOKENS)}")
        i = m.end()
    out += re.escape(pattern[i:])
    return re.compile(rf"^(?:.*/)?{out}$")


CRITERION_WORDS = ("criterion", "reference", "gold", "bead", "marker")


def guess_fields(path: Path, root: Path | None, bone_names, pattern: str = "") -> dict:
    """Trial, rater, session, bone and criterion for a file: from the pattern if it matches, otherwise
    guessed (bone = a known bone name in the file name; rater = the folder name)."""
    rel = path.relative_to(root) if root is not None and path.is_relative_to(root) else Path(path.name)
    stem = rel.with_suffix("").as_posix()
    out = {}
    if pattern:
        m = pattern_regex(pattern).match(stem)
        if m:
            out = {k: v for k, v in m.groupdict().items() if v}
    low = stem.lower()
    if "bone" not in out:
        parts = re.split(r"[^a-z0-9]+", path.stem.lower())
        for b in sorted(bone_names, key=len, reverse=True):
            if b.lower() in parts or path.stem.lower().endswith(b.lower()):
                out["bone"] = b
                break
    out["criterion"] = any(w in low for w in CRITERION_WORDS) and "rater" not in out
    if "rater" not in out and not out["criterion"]:
        out["rater"] = path.parent.name
    return out
