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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .joints import decompose
from .tracking import Tracking, TrackingError, load_tracking

CRITERION = "Criterion"
_FIELDS = {"bone": "bone", "assessor": "rater", "session": "session", "trial": "trial"}


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


@dataclass
class AnalysisSettings:
    mode: str = "validity"                  # "validity", "inter" or "intra"
    measures: tuple = DEFAULT_MEASURES
    rotation_threshold: float = 2.0         # deg (Welte 2022)
    translation_threshold: float = 1.0      # mm
    inter_session: str = ""                 # inter-rater: which session ("" = the first)
    pool: str = "none"                      # "none", "trials" (pool a bone's trials) or "all" (pool bones too)

    def threshold(self, m: Measure) -> float:
        return self.rotation_threshold if m.kind == "rotation" else self.translation_threshold


MODES = [("validity", "Assessors vs reference (validity)"), ("inter", "Between assessors (inter-rater)"),
         ("intra", "Between sessions (intra-rater)")]


@dataclass(eq=False)
class Comparison:
    """Per-frame values of one test tracking against one reference tracking, or why it can't be made."""
    mode: str
    trial: str
    bone: str
    test: str               # label of the test tracking (assessor, maybe with session)
    ref: str                # label of the reference tracking
    frames: np.ndarray      # trial frames (0-based) tracked in both
    values: dict            # measure key -> (n,) array, aligned with frames
    n_test: int = 0         # frames tracked in the test tracking
    n_ref: int = 0
    missing: str = ""       # why there are no values (a file not loaded, unreadable, no common frames)

    @property
    def label(self) -> str:
        return self.test if self.mode == "validity" else f"{self.test} vs {self.ref}"

    @property
    def ok(self) -> bool:
        return not self.missing and len(self.frames) > 0


@dataclass(eq=False)
class AssessmentProject:
    """The factors of an evaluation and the tracking file in each of their combinations.

    Factors: bones (with optional surface/axes), assessors, sessions, trials, and whether there is a
    reference (criterion). Every combination trial × bone × (reference, or assessor × session) is a slot
    for one tracking file; slots can be left empty, and comparisons that need an empty slot are N/A.
    """
    name: str = "Assessor evaluation"
    entries: list = field(default_factory=list)          # TrackingEntry, one per filled slot
    bones: list = field(default_factory=list)            # BoneInfo, in order: the bone factor
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    assessor_names: list = field(default_factory=list)
    session_names: list = field(default_factory=lambda: ["1"])
    trial_names: list = field(default_factory=lambda: ["Trial 1"])
    has_reference: bool = False
    reference_name: str = "Reference"

    # ---- factors
    def trials(self) -> list[str]:
        return list(self.trial_names)

    def bone_names(self) -> list[str]:
        return [b.name for b in self.bones]

    def raters(self) -> list[str]:
        return list(self.assessor_names)

    def sessions(self) -> list[str]:
        return list(self.session_names)

    def bone_info_for(self, name: str) -> BoneInfo:
        for b in self.bones:
            if b.name == name:
                return b
        b = BoneInfo(name=name)
        self.bones.append(b)
        return b

    def bone_info(self, trial: str, bone: str) -> BoneInfo | None:
        return next((b for b in self.bones if b.name == bone), None)

    def _add(self, names: list, name: str, what: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError(f"The {what} needs a name.")
        if name in names:
            raise ValueError(f"There is already a{'n' * (what[0] in 'aeiou')} {what} '{name}'.")
        names.append(name)
        return name

    def add_bone(self, name: str) -> str:
        name = self._add(self.bone_names(), name, "bone")
        self.bones.append(BoneInfo(name=name))
        return name

    def add_assessor(self, name: str) -> str:
        if name.strip() and self.has_reference and name.strip() == self.reference_name:
            raise ValueError(f"'{name.strip()}' is the reference's name.")
        return self._add(self.assessor_names, name, "assessor")

    def add_session(self, name: str | None = None) -> str:
        if name is None:
            name = next(str(i) for i in range(1, 1000) if str(i) not in self.session_names)
        return self._add(self.session_names, name, "session")

    def add_trial(self, name: str | None = None) -> str:
        if name is None:
            name = next(f"Trial {i}" for i in range(1, 1000) if f"Trial {i}" not in self.trial_names)
        return self._add(self.trial_names, name, "trial")

    def next_assessor_name(self) -> str:
        return next(f"Assessor {i}" for i in range(1, 1000) if f"Assessor {i}" not in self.assessor_names)

    def _names(self, factor: str) -> list:
        return {"bone": self.bone_names(), "assessor": self.assessor_names, "session": self.session_names,
                "trial": self.trial_names}[factor]

    def rename(self, factor: str, old: str, new: str) -> None:
        """Rename a bone, assessor, session or trial; its tracking files follow."""
        new = new.strip()
        if new == old:
            return
        names = self._names(factor)
        if not new:
            raise ValueError(f"The {factor} needs a name.")
        if new in names:
            raise ValueError(f"There is already a{'n' * (factor[0] in 'aeiou')} {factor} '{new}'.")
        attr = _FIELDS[factor]
        for e in self.entries:
            if getattr(e, attr) == old and not (e.criterion and factor in ("assessor", "session")):
                setattr(e, attr, new)
        if factor == "bone":
            self.bone_info_for(old).name = new
        else:
            names[names.index(old)] = new

    def remove(self, factor: str, name: str) -> None:
        """Remove a bone, assessor, session or trial and the tracking files in its slots."""
        attr = _FIELDS[factor]
        self.entries = [e for e in self.entries if not (getattr(e, attr) == name and
                                                         not (e.criterion and factor in ("assessor", "session")))]
        if factor == "bone":
            self.bones = [b for b in self.bones if b.name != name]
        else:
            self._names(factor).remove(name)
            if factor in ("session", "trial") and not self._names(factor):
                self._names(factor).append("1" if factor == "session" else "Trial 1")

    def set_reference(self, on: bool, name: str | None = None) -> None:
        """Add (on) or remove the reference; removing it removes its tracking files."""
        self.has_reference = bool(on)
        if name is not None and name.strip():
            self.reference_name = name.strip()
        if not on:
            self.entries = [e for e in self.entries if not e.criterion]

    def rename_reference(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("The reference needs a name.")
        if name in self.assessor_names:
            raise ValueError(f"There is already an assessor '{name}'.")
        self.reference_name = name

    def move(self, factor: str, name: str, step: int) -> None:
        lst = self.bones if factor == "bone" else self._names(factor)
        i = next(k for k, x in enumerate(lst) if (x.name if factor == "bone" else x) == name)
        j = max(0, min(len(lst) - 1, i + step))
        lst.insert(j, lst.pop(i))

    # ---- slots and their tracking
    def rows(self) -> list[tuple]:
        """The grid's rows: (who, session); who is CRITERION for the reference."""
        out = [(CRITERION, "")] if self.has_reference else []
        return out + [(a, s) for a in self.assessor_names for s in self.session_names]

    def row_label(self, who: str, session: str) -> str:
        if who == CRITERION:
            return self.reference_name
        return who if len(self.session_names) == 1 else f"{who} · session {session}"

    def entry(self, trial: str, bone: str, who: str, session: str = "") -> TrackingEntry | None:
        crit = who == CRITERION
        for e in self.entries:
            if e.trial == trial and e.bone == bone and e.criterion == crit and \
                    (crit or (e.rater == who and e.session == session)):
                return e
        return None

    def set_entry(self, trial: str, bone: str, who: str, session: str, path, overrides=None, volume: int = 0,
                  offset: int = 0) -> TrackingEntry:
        """Put a tracking file in a slot (replacing any there), adding factors it names, and read it."""
        if trial not in self.trial_names:
            self.add_trial(trial)
        if bone not in self.bone_names():
            self.add_bone(bone)
        crit = who == CRITERION
        if crit:
            self.has_reference = True
        else:
            if who not in self.assessor_names:
                self.add_assessor(who)
            if session not in self.session_names:
                self.add_session(session)
        old = self.entry(trial, bone, who, session)
        if old is not None:
            self.entries.remove(old)
        e = TrackingEntry(path=Path(path), trial=trial, bone=bone, rater="" if crit else who,
                          session="" if crit else session, criterion=crit, overrides=dict(overrides or {}),
                          volume=volume, offset=offset)
        e.load()
        self.entries.append(e)
        return e

    def filled(self) -> tuple[int, int]:
        """(slots with a file, all slots)."""
        n = len(self.trial_names) * len(self.bones) * len(self.rows())
        have = sum(self.entry(t, b, w, s) is not None for t in self.trial_names for b in self.bone_names()
                   for w, s in self.rows())
        return have, n

    def sync_factors(self) -> None:
        """Add any bone, assessor, session or trial that tracking entries name (e.g. older project files)."""
        for e in self.entries:
            if e.trial and e.trial not in self.trial_names:
                self.trial_names.append(e.trial)
            if e.bone and e.bone not in self.bone_names():
                self.bones.append(BoneInfo(name=e.bone))
            if e.criterion:
                self.has_reference = True
                e.session = ""
            else:
                if e.rater and e.rater not in self.assessor_names:
                    self.assessor_names.append(e.rater)
                if e.session and e.session not in self.session_names:
                    self.session_names.append(e.session)
        for lst, used in ((self.trial_names, {e.trial for e in self.entries}),
                          (self.session_names, {e.session for e in self.entries if not e.criterion})):
            default = "Trial 1" if lst is self.trial_names else "1"
            if default in lst and default not in used and len(lst) > 1:
                lst.remove(default)

    def available_measures(self) -> list[str]:
        has_surface = any(b.surface is not None for b in self.bones)
        has_axes = any(b.acs is not None for b in self.bones)
        return [k for k, m in MEASURES.items()
                if not m.needs or (m.needs == "surface" and has_surface) or (m.needs == "axes" and has_axes)]

    # ---- checks
    def problems(self) -> list[str]:
        """Files that can't be read, and files of one bone and trial with different numbers of rows."""
        out = []
        for e in self.entries:
            if e.error:
                out.append(f"{e.path.name} ({self.row_label(e.who if not e.criterion else CRITERION, e.session)}, "
                           f"{e.bone}): {e.error}")
        for group in _groupby(self.entries, lambda e: (e.trial, e.bone)).values():
            spans = {(e.tracking.offset, e.tracking.n_rows) for e in group if e.tracking is not None}
            if len(spans) > 1:
                out.append(f"{group[0].bone}, {group[0].trial}: the files have different numbers of rows "
                           f"({', '.join(str(r) for _o, r in sorted(spans))}); rows are matched by number")
        return out

    # ---- comparisons
    def test_label(self, who: str, session: str) -> str:
        return who if len(self.session_names) == 1 else f"{who} s{session}"

    def comparisons(self, settings: AnalysisSettings | None = None) -> list[Comparison]:
        """Every comparison the factors call for, in order: computed where both files are there, otherwise
        with `missing` saying which file is missing."""
        s = settings or self.settings
        out = []
        for trial in self.trial_names:
            for bone in self.bone_names():
                info = self.bone_info_for(bone)
                pts, acs = info.points(), info.acs
                for test, ref, tl, rl in self._pairs(s, trial, bone):
                    out.append(self._compare(s.mode, trial, bone, test, ref, tl, rl, pts, acs))
        return out

    def _pairs(self, s: AnalysisSettings, trial: str, bone: str):
        """(test slot, reference slot, test label, reference label); slots are (who, session)."""
        if s.mode == "validity":
            if self.has_reference:
                for a in self.assessor_names:
                    for ses in self.session_names:
                        yield (a, ses), (CRITERION, ""), self.test_label(a, ses), self.reference_name
        elif s.mode == "inter":
            ses = s.inter_session if s.inter_session in self.session_names else self.session_names[0]
            for a, b in itertools.combinations(self.assessor_names, 2):
                yield (a, ses), (b, ses), a, b
        elif s.mode == "intra":
            for a in self.assessor_names:
                for s1, s2 in itertools.combinations(self.session_names, 2):
                    yield (a, s2), (a, s1), f"{a} s{s2}", f"{a} s{s1}"

    def _compare(self, mode, trial, bone, test, ref, tl, rl, pts, acs) -> Comparison:
        et, er = self.entry(trial, bone, *test), self.entry(trial, bone, *ref)
        missing = []
        for e, slot in ((et, test), (er, ref)):
            name = self.row_label(*slot)
            if e is None:
                missing.append(f"no file for {name}")
            elif e.tracking is None:
                missing.append(f"{name}'s file can't be read")
        if missing:
            return Comparison(mode, trial, bone, tl, rl, np.zeros(0, int), {}, missing="; ".join(missing))
        c = compare(et.tracking, er.tracking, mode, trial, bone, tl, rl, pts, acs)
        if not len(c.frames):
            c.missing = "no frames tracked in both"
        return c

    # ---- file
    def to_dict(self) -> dict:
        st = self.settings
        return {"tool": "assessor", "version": 3, "name": self.name,
                "factors": {"bones": self.bone_names(), "assessors": self.assessor_names,
                            "sessions": self.session_names, "trials": self.trial_names,
                            "reference": self.has_reference, "reference_name": self.reference_name},
                "entries": [e.to_dict() for e in self.entries], "bones": [b.to_dict() for b in self.bones],
                "settings": {"mode": st.mode, "measures": list(st.measures),
                             "rotation_threshold": st.rotation_threshold,
                             "translation_threshold": st.translation_threshold,
                             "inter_session": st.inter_session, "pool": st.pool}}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path, build_surfaces: bool = True) -> AssessmentProject:
        """Read a project file (any version). Tracking files are read; surfaces built when build_surfaces."""
        from .models import BoneModel

        d = json.loads(Path(path).read_text())
        if d.get("tool") != "assessor":
            raise ValueError("This isn't an Assessor Evaluation project file.")
        st = d.get("settings", {})
        p = cls(name=d.get("name", "Assessor evaluation"),
                settings=AnalysisSettings(mode=st.get("mode", "validity"),
                                          measures=tuple(st.get("measures", DEFAULT_MEASURES)),
                                          rotation_threshold=float(st.get("rotation_threshold", 2.0)),
                                          translation_threshold=float(st.get("translation_threshold", 1.0)),
                                          inter_session=st.get("inter_session", ""), pool=st.get("pool", "none")))
        f = d.get("factors")
        if f:
            p.assessor_names = list(f.get("assessors", []))
            p.session_names = list(f.get("sessions", ["1"])) or ["1"]
            p.trial_names = list(f.get("trials", ["Trial 1"])) or ["Trial 1"]
            p.has_reference = bool(f.get("reference", False))
            p.reference_name = f.get("reference_name", "Reference") or "Reference"
            p.bones = [BoneInfo(name=b) for b in f.get("bones", [])]
        for a in d.get("assessors", []):                  # version 2: assessors with their bones
            if a.get("criterion"):
                p.has_reference = True
            else:
                if a["name"] not in p.assessor_names:
                    p.assessor_names.append(a["name"])
                if str(a.get("session", "1")) not in p.session_names:
                    p.session_names.append(str(a.get("session", "1")))
            for b in a.get("bones", []):
                if b not in p.bone_names():
                    p.bones.append(BoneInfo(name=b))
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
            b = p.bone_info_for(bd["name"])
            b.surface, b.axes_path, b.axes_segment = surf, bd.get("axes_path"), bd.get("axes_segment")
            load_bone_axes(b)
        p.sync_factors()
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
    label: str              # comparison label (assessor, or "A vs B")
    measure: str
    stats: dict             # {"n": 0} when N/A
    frames_test: int = 0
    frames_ref: int = 0
    note: str = ""          # why it's N/A, or how many of the pooled comparisons are missing

    @property
    def na(self) -> bool:
        return self.stats.get("n", 0) == 0


def summarise(comps: list[Comparison], settings: AnalysisSettings) -> list[SummaryRow]:
    """Statistics per comparison and measure (pooled over trials, and bones, if the settings say so).
    Comparisons without values give N/A rows that say why."""
    def group_key(c: Comparison):
        if settings.pool == "all":
            return ("All trials", "All bones", c.label)
        if settings.pool == "trials":
            return ("All trials", c.bone, c.label)
        return (c.trial, c.bone, c.label)

    rows = []
    for (trial, bone, label), group in _groupby(comps, group_key).items():
        ok = [c for c in group if c.ok]
        missing = [c for c in group if not c.ok]
        if not ok:
            note = "; ".join(_unique(c.missing for c in missing))
        elif missing:
            note = f"{len(missing)} of {len(group)} comparisons missing"
        else:
            note = ""
        # only frames tracked in both files are compared: say so when that leaves out many of them
        both = sum(len(c.frames) for c in ok)
        fewer = sum(min(c.n_test, c.n_ref) for c in ok)
        if ok and both < 0.8 * fewer:
            note = "; ".join(x for x in (note, f"{both} of {fewer} tracked frames in common") if x)
        for key in settings.measures:
            m = MEASURES[key]
            arrays = [c.values[key] for c in ok if key in c.values]
            if ok and not arrays:
                continue                    # measure not available for this bone (no surface / axes)
            stats = statistics(np.concatenate(arrays), settings.threshold(m), m.signed) if arrays else {"n": 0}
            rows.append(SummaryRow(trial, bone, label, key, stats, sum(c.n_test for c in ok),
                                   sum(c.n_ref for c in ok), note))
    return rows


def write_frames_csv(path, comps: list[Comparison], measures) -> None:
    """Long format: one row per comparison, frame and measure (for R, SPSS, ...)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "bone", "comparison", "test", "reference", "frame", "measure", "value", "unit"])
        for c in comps:
            if not c.ok:
                continue
            for key in measures:
                if key not in c.values:
                    continue
                unit = MEASURES[key].unit
                for f, v in zip(c.frames, c.values[key]):
                    w.writerow([c.trial, c.bone, c.mode, c.test, c.ref, int(f) + 1, key, f"{v:.6f}", unit])


def write_summary_csv(path, rows: list[SummaryRow]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "bone", "comparison", "measure", "unit"] + [k for k, _ in STAT_COLUMNS]
                   + ["frames_tracked_test", "frames_tracked_reference", "note"])
        for r in rows:
            m = MEASURES[r.measure]
            vals = ["0"] + ["NA"] * (len(STAT_COLUMNS) - 1) if r.na else [_fmt_csv(r.stats.get(k))
                                                                         for k, _ in STAT_COLUMNS]
            tracked = ["NA", "NA"] if r.na else [str(r.frames_test), str(r.frames_ref)]
            w.writerow([r.trial, r.bone, r.label, r.measure, m.unit] + vals + tracked + [r.note])


def _fmt_csv(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.6g}"
