"""Kinematic models: segment axes on bone surfaces, optional joints, optional reference poses.

A model is built in three steps (the Model Builder follows them):

1. **Segments.** Each segment is one bone surface with an anatomical coordinate system (ACS),
   made from landmarks picked or painted on the surface, or by an automatic method fitted to it.
   The ACS lives in the surface's own model space (the space its .tra poses act on), so it never
   depends on where any other bone is. A segment alone already gives its motion in a trial.
2. **Joints (optional).** A joint pairs two segments with a published definition (e.g. Grood &
   Suntay for the knee: femur -> tibia). Angles are the distal ACS relative to the proximal ACS at
   each frame, so their zero is anatomical: all angles are 0 when the two ACSs line up.
3. **Reference pose (optional, per joint).** A pose of the two bones (from a neutral/static
   trial's tracking, or the pose they had in the scan) whose joint values are taken as zero,
   either by subtracting them or by realigning the distal axes to the proximal ones there.

Segment types (femur, tibia, other) say which landmarks and automatic methods a segment can use;
joint types say which segment types they connect and their angle convention. One model file
serves every trial of a participant.

Coordinates and conventions
---------------------------
An ACS is a 4x4 matrix whose columns are the X, Y, Z axes and the origin, in the bone's model space.
At a frame where the bone's tracking pose is T, the axes in the world are T @ ACS.

ISB axis names are used throughout: X anterior, Y superior (proximal), Z to the subject's right
(Wu et al. 2002, 2005). Because Z points right on both sides, left-side angles are sign-flipped so
that positive means the same anatomical motion on either side (see JOINT_CONVENTIONS).

Joint angles use a joint coordinate system (Grood & Suntay 1983): e1 is fixed in the proximal bone,
e3 in the distal bone, and e2 = e3 x e1 floats. For orthonormal ACSs this equals the intrinsic
Cardan sequence (e1, e2, e3) of the distal ACS relative to the proximal ACS, which is how it is
computed here. Translations are the components of the vector from the proximal origin to the
distal origin along e1, e2 and e3 (non-orthogonal components, as in Grood & Suntay).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_VERSION = 2


class JointModelError(ValueError):
    pass


# ---------------------------------------------------------------- landmark geometry

def sphere_fit(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere through points: returns (centre, radius)."""
    P = np.asarray(points, float)
    if len(P) < 4:
        raise JointModelError("A sphere fit needs at least 4 points. Paint a larger area.")
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    return c, r


LANDMARK_METHODS = {
    "point": "Single point",
    "region": "Centre of a painted region",
    "sphere": "Centre of a sphere fitted to a painted region",
}


def landmark_value(method: str, points) -> np.ndarray | None:
    """The 3D position a landmark stands for, from its picked or painted points."""
    P = np.asarray(points, float).reshape(-1, 3) if points is not None and len(points) else None
    if P is None or len(P) == 0:
        return None
    if method == "point":
        return P[-1]
    if method == "region":
        return P.mean(0)
    if method == "sphere":
        return sphere_fit(P)[0]
    raise JointModelError(f"Unknown landmark method {method!r}")


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise JointModelError(f"Can't form the {what}: the landmarks coincide or are in line.")
    return v / n


def frame_from_axes(origin, y_axis, z_hint) -> np.ndarray:
    """ISB-style frame: Y given, Z = the part of z_hint perpendicular to Y, X = Y x Z."""
    y = _unit(np.asarray(y_axis, float), "Y axis")
    z = np.asarray(z_hint, float)
    z = _unit(z - (z @ y) * y, "Z axis (it may be parallel to Y)")
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def inertial_frame(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Principal inertial axes of a closed surface treated as a solid of uniform density.

    Returns (ACS, principal moments). Axis 1 is the long axis (smallest moment), then axis 2, 3,
    making a right-handed frame at the centroid. Signs are fixed so axis 1 points from the centroid
    towards the farther end of the bone and axis 2 towards the bone's bulkier side; they are the
    same every time for the same surface, but carry no anatomical meaning.
    """
    tri = mesh.triangulate()
    pts = np.asarray(tri.points, float)
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    ref = pts.mean(0)
    a, b, c = a - ref, b - ref, c - ref
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))            # 6 × signed tetra volume
    V = vol6.sum() / 6
    if abs(V) < 1e-6:
        raise JointModelError("The surface encloses no volume (is it closed?).")
    cen = ((a + b + c) * vol6[:, None]).sum(0) / 24 / V          # centroid relative to ref
    # second moments of each tetra (origin, a, b, c): integral of x_i x_j over the tetra
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (vol6 / 120 * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                                    + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j]
                                    + a[:, j] * c[:, i] + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum()
    S = S - V * np.outer(cen, cen)                               # about the centroid
    inertia = np.trace(S) * np.eye(3) - S
    w, vecs = np.linalg.eigh(inertia)                            # ascending: smallest = long axis
    if V < 0:
        w = -w[::-1]
        vecs = vecs[:, ::-1]
    centroid = ref + cen
    rel = pts - centroid
    ax = [vecs[:, k].copy() for k in range(3)]
    for k in (0, 1):                                             # deterministic signs
        proj = rel @ ax[k]
        if abs(proj.max()) < abs(proj.min()):
            ax[k] = -ax[k]
    ax[2] = np.cross(ax[0], ax[1])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = ax[0], ax[1], ax[2], centroid
    return m, np.abs(w)


# ---------------------------------------------------------------- definitions

@dataclass(frozen=True)
class LandmarkDef:
    key: str
    name: str
    hint: str
    methods: tuple = ("point", "region")


@dataclass(frozen=True)
class SegmentType:
    key: str
    title: str
    landmarks: tuple = ()           # LandmarkDefs, for landmark axes
    recipe: str | None = None       # name in RECIPES (landmarks -> ACS)
    auto_methods: tuple = ()        # keys in core.auto_axes.AUTO_METHODS
    anatomical: bool = True         # axes follow ISB names (X anterior, Y superior, Z right)
    axes_note: str = ""
    references: tuple = ()


@dataclass(frozen=True)
class JointType:
    key: str
    title: str
    proximal_types: tuple           # segment type keys allowed (empty = any)
    distal_types: tuple
    convention: str                 # name in JOINT_CONVENTIONS
    summary: str = ""
    references: tuple = ()
    default_name: str = "Joint"


def _femur_isb(L: dict, side: str, mesh) -> np.ndarray:
    """ISB femur axes (Wu et al. 2002, hip section), with the origin at the knee.

    Y: from the epicondyle midpoint to the superior point (hip centre, or shaft centre on a
    partial femur). Z: perpendicular to Y in the plane of the epicondyles, pointing right.
    X = Y × Z, pointing anteriorly. Origin: epicondyle midpoint (the knee end, for knee joints).
    """
    me, le, sup = L["ME"], L["LE"], L["FS"]
    mid = (me + le) / 2
    right = (le - me) if side == "right" else (me - le)
    return frame_from_axes(mid, sup - mid, right)


def _tibia_knee(L: dict, side: str, mesh) -> np.ndarray:
    """Tibia axes for the knee (ISB axis names; condyle-based, as ISB's tibial plateau option).

    Y: from the distal point (intermalleolar midpoint, or shaft centre on a partial tibia) to the
    midpoint of the condyles. Z: perpendicular to Y in the plane of the condyles, pointing right.
    X = Y × Z, pointing anteriorly. Origin: midpoint of the condyles.
    """
    mc, lc, dist = L["MC"], L["LC"], L["TD"]
    mid = (mc + lc) / 2
    right = (lc - mc) if side == "right" else (mc - lc)
    return frame_from_axes(mid, mid - dist, right)


def _inertial(L: dict, side: str, mesh) -> np.ndarray:
    return inertial_frame(mesh)[0]


RECIPES = {"isb_femur": _femur_isb, "knee_tibia": _tibia_knee, "inertial": _inertial}

# Joint conventions: rotation sequence (intrinsic, axes of the distal ACS relative to the proximal),
# names of the three angles and three translations, and sign flips per side so that positive is
# the same anatomical motion on both sides.
JOINT_CONVENTIONS = {
    "grood_suntay_knee": {
        "title": "Grood & Suntay joint coordinate system (e1 = femur Z, e3 = tibia Y)",
        "sequence": "ZXY",
        "angles": ("Flexion (+) / extension", "Adduction (+) / abduction", "Internal (+) / external rotation"),
        "short": ("Flexion", "Adduction", "Int. rotation"),
        "translations": ("Lateral (+) / medial", "Anterior (+) / posterior", "Proximal (+) / distal"),
        "short_t": ("Lateral", "Anterior", "Proximal"),
        "signs": {"right": ((-1, 1, 1), (1, 1, 1)), "left": ((-1, -1, -1), (-1, 1, 1))},
    },
    "cardan_xyz": {
        "title": "Cardan sequence X-Y-Z of the distal axes relative to the proximal axes",
        "sequence": "XYZ",
        "angles": ("Rotation about axis 1", "Rotation about axis 2", "Rotation about axis 3"),
        "short": ("Axis 1", "Axis 2", "Axis 3"),
        "translations": ("Along axis 1", "Along axis 2", "Along axis 3"),
        "short_t": ("Along 1", "Along 2", "Along 3"),
        "signs": {"right": ((1, 1, 1), (1, 1, 1)), "left": ((1, 1, 1), (1, 1, 1))},
    },
}


# Joint conventions: rotation sequence (intrinsic, axes of the distal ACS relative to the proximal),
# names of the three angles and three translations, and sign flips per side so that positive is
# the same anatomical motion on both sides.
JOINT_CONVENTIONS = {
    "grood_suntay_knee": {
        "title": "Grood & Suntay joint coordinate system (e1 = femur Z, e3 = tibia Y)",
        "sequence": "ZXY",
        "angles": ("Flexion (+) / extension", "Adduction (+) / abduction", "Internal (+) / external rotation"),
        "short": ("Flexion", "Adduction", "Int. rotation"),
        "translations": ("Lateral (+) / medial", "Anterior (+) / posterior", "Proximal (+) / distal"),
        "short_t": ("Lateral", "Anterior", "Proximal"),
        "signs": {"right": ((-1, 1, 1), (1, 1, 1)), "left": ((-1, -1, -1), (-1, 1, 1))},
    },
    "jcs_zxy": {
        "title": "Joint coordinate system: e1 = proximal Z, e3 = distal Y, e2 floating (ISB style)",
        "sequence": "ZXY",
        "angles": ("About proximal Z (medial–lateral)", "About the floating axis", "About distal Y (long axis)"),
        "short": ("About Z", "Floating", "About Y"),
        "translations": ("Along proximal Z", "Along the floating axis", "Along distal Y"),
        "short_t": ("Along Z", "Floating", "Along Y"),
        "signs": {"right": ((1, 1, 1), (1, 1, 1)), "left": ((1, 1, 1), (1, 1, 1))},
    },
    "cardan_xyz": {
        "title": "Cardan sequence X-Y-Z of the distal axes relative to the proximal axes",
        "sequence": "XYZ",
        "angles": ("Rotation about axis 1", "Rotation about axis 2", "Rotation about axis 3"),
        "short": ("Axis 1", "Axis 2", "Axis 3"),
        "translations": ("Along axis 1", "Along axis 2", "Along axis 3"),
        "short_t": ("Along 1", "Along 2", "Along 3"),
        "signs": {"right": ((1, 1, 1), (1, 1, 1)), "left": ((1, 1, 1), (1, 1, 1))},
    },
}

# How a segment's own motion is reported: rotation about its own axes relative to a reference
# orientation, as an intrinsic Z-X-Y sequence (medial-lateral axis first, long axis last).
SEGMENT_SEQUENCE = "ZXY"
SEGMENT_NAMES = {True: (("About Z (medial–lateral)", "About X (anterior–posterior)", "About Y (long axis)"),
                        ("About Z", "About X", "About Y"),
                        ("Along X (anterior)", "Along Y (superior)", "Along Z (right)"),
                        ("Along X", "Along Y", "Along Z")),
                 False: (("About axis 3", "About axis 1", "About axis 2"), ("Axis 3", "Axis 1", "Axis 2"),
                         ("Along axis 1", "Along axis 2", "Along axis 3"), ("Along 1", "Along 2", "Along 3"))}

WU2002 = "Wu G et al. (2002) ISB recommendation, part I: ankle, hip and spine. J Biomech 35:543–548"
GROOD = "Grood ES, Suntay WJ (1983) J Biomech Eng 105:136–144"

SEGMENT_TYPES: dict[str, SegmentType] = {t.key: t for t in [
    SegmentType("femur", "Femur", (
        LandmarkDef("ME", "Medial epicondyle", "Most prominent point of the medial epicondyle.", ("point", "region")),
        LandmarkDef("LE", "Lateral epicondyle", "Most prominent point of the lateral epicondyle.", ("point", "region")),
        LandmarkDef("FS", "Superior point (hip centre or shaft centre)",
                    "Hip joint centre: paint the femoral head and use a sphere fit. On a partial femur, paint a band "
                    "around the most proximal part of the shaft and use its centre.", ("region", "sphere", "point")),
    ), "isb_femur", ("femur_cylinder", "femur_spheres"), True,
        "Origin at the epicondyle midpoint; Y towards the superior point; Z right; X anterior.", (WU2002,)),
    SegmentType("tibia", "Tibia", (
        LandmarkDef("MC", "Medial condyle", "Most medial point of the medial tibial condyle (plateau rim).",
                    ("point", "region")),
        LandmarkDef("LC", "Lateral condyle", "Most lateral point of the lateral tibial condyle (plateau rim).",
                    ("point", "region")),
        LandmarkDef("TD", "Distal point (malleoli midpoint or shaft centre)",
                    "Midpoint between the malleoli if the model includes them. On a partial tibia, paint a band "
                    "around the most distal part of the shaft and use its centre.", ("region", "point")),
    ), "knee_tibia", ("tibia_ellipse", "tibia_centroids", "tibia_inertia"), True,
        "Origin at the condyle midpoint; Y from the distal point to the origin; Z right; X anterior.", (WU2002,)),
    SegmentType("generic", "Other bone", (), None, ("inertial",), False,
                "Principal axes of inertia: axis 1 is the long axis. No anatomical names."),
]}

JOINT_TYPES: dict[str, JointType] = {t.key: t for t in [
    JointType("knee_grood_suntay", "Knee (tibiofemoral): Grood & Suntay", ("femur",), ("tibia",), "grood_suntay_knee",
              "Flexion about the femur's Z axis, internal rotation about the tibia's Y axis, ad/abduction about the "
              "floating axis between them. Left knees are mirrored so positive values mean the same motion on both "
              "sides. ISB doesn't define the knee; this is the convention most BVR knee studies use.",
              (GROOD, WU2002), "Knee"),
    JointType("jcs_zxy", "Joint coordinate system (ISB style, any two segments)", ("femur", "tibia"),
              ("femur", "tibia"), "jcs_zxy",
              "Rotation about the proximal segment's Z axis, then the floating axis, then the distal segment's Y axis. "
              "No clinical names or side mirroring.", (GROOD,)),
    JointType("cardan_xyz", "Cardan X-Y-Z (any two segments)", (), (), "cardan_xyz",
              "Rotations of the distal axes relative to the proximal axes as an X-Y-Z Cardan sequence. Works for any "
              "segments, including inertial axes; the angles have no clinical names."),
]}

PRESETS = {
    "knee": {"title": "Knee: femur + tibia + Grood & Suntay knee",
             "segments": [("femur", "Femur"), ("tibia", "Tibia")],
             "joints": [("knee_grood_suntay", "Femur", "Tibia", "Knee")]},
}


# ---------------------------------------------------------------- rotations

def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def compose(sequence: str, angles_rad) -> np.ndarray:
    """Intrinsic rotation sequence: R = R_a1 @ R_a2 @ R_a3."""
    R = np.eye(3)
    for ax, a in zip(sequence, angles_rad):
        R = R @ _rot(ax, a)
    return R


def decompose(R: np.ndarray, sequence: str) -> np.ndarray:
    """Angles (radians) of an intrinsic Cardan sequence (all three axes different)."""
    idx = {"X": 0, "Y": 1, "Z": 2}
    i, j, k = (idx[c] for c in sequence)
    if len({i, j, k}) != 3:
        raise ValueError("Only Cardan sequences (three different axes) are supported.")
    # sign of the permutation: +1 for cyclic (XYZ, YZX, ZXY), -1 otherwise
    eps = 1.0 if (j - i) % 3 == 1 else -1.0
    b = np.arcsin(np.clip(eps * R[i, k], -1.0, 1.0))
    a = np.arctan2(-eps * R[j, k], R[k, k])
    c = np.arctan2(-eps * R[i, j], R[i, i])
    return np.array([a, b, c])


# ---------------------------------------------------------------- model

def _rotation_mean(Rs) -> np.ndarray:
    """Chordal mean of rotation matrices (projection of their average onto SO(3))."""
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _pose_mean(Ts) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rotation_mean([t[:3, :3] for t in Ts])
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
    return T


@dataclass(eq=False)
class Segment:
    name: str
    type: str                             # key in SEGMENT_TYPES
    side: str = "right"                   # "right", "left" (anatomical segments) or "none"
    surface: object = None                # a core.models.BoneModel (surface file + model space settings)
    axes_source: str = "landmarks"        # "landmarks", or an automatic method key (core.auto_axes)
    landmarks: dict = field(default_factory=dict)    # key -> {"method": str, "points": [[x,y,z], ...]}
    auto_options: dict = field(default_factory=dict)  # e.g. {"flip_ap": True}
    preview_tracking: str | None = None   # optional .tra used to preview the segment and its joints

    # runtime
    acs: np.ndarray | None = None
    error: str = ""
    auto: object = None                   # core.auto_axes.AutoResult when axes_source is automatic

    @property
    def stype(self) -> SegmentType:
        return SEGMENT_TYPES[self.type]

    @property
    def mesh(self):
        return getattr(self.surface, "mesh", None) if self.surface is not None else None


@dataclass(eq=False)
class Reference:
    """A reference pose of a joint's two bones: their relative pose G0 = inv(T_prox) @ T_dist.

    G0 relates the two surfaces' model spaces, so it stays valid if either segment's axes are
    redefined later. mode: "subtract" (values at the reference are subtracted) or "realign" (the
    distal axes are rotated to line up with the proximal axes at the reference, then translations
    are subtracted). source records where it came from, for the record.
    """

    G0: np.ndarray
    mode: str = "subtract"
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"G0": self.G0.tolist(), "mode": self.mode, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        return cls(np.array(d["G0"], float), d.get("mode", "subtract"), d.get("source", {}))


REFERENCE_MODES = {
    "subtract": "Subtract the reference values (angles and translations are 0 there)",
    "realign": "Realign the distal axes to the proximal axes at the reference",
}


@dataclass(eq=False)
class Joint:
    name: str
    type: str                             # key in JOINT_TYPES
    proximal: str                         # segment names
    distal: str
    reference: Reference | None = None

    @property
    def jtype(self) -> JointType:
        return JOINT_TYPES[self.type]

    @property
    def convention(self) -> dict:
        return JOINT_CONVENTIONS[self.jtype.convention]


@dataclass
class KinematicModel:
    name: str = "Model"
    segments: list = field(default_factory=list)     # [Segment]
    joints: list = field(default_factory=list)       # [Joint]
    path: Path | None = None
    _auto_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---- building
    @classmethod
    def preset(cls, key: str, side: str = "right") -> "KinematicModel":
        p = PRESETS[key]
        m = cls(name=p["title"].split(":")[0])
        for t, n in p["segments"]:
            m.add_segment(t, n, side)
        for t, a, b, n in p["joints"]:
            m.add_joint(t, a, b, n)
        return m

    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise JointModelError(f"No segment named {name!r}.")

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise JointModelError(f"No joint named {name!r}.")

    def _unique(self, base: str, taken) -> str:
        name, k = base, 2
        while name in taken:
            name, k = f"{base} {k}", k + 1
        return name

    def add_segment(self, type_key: str, name: str | None = None, side: str = "right") -> Segment:
        t = SEGMENT_TYPES[type_key]
        seg = Segment(name=self._unique(name or t.title, {s.name for s in self.segments}), type=type_key,
                      side=side if t.anatomical else "none")
        seg.axes_source = t.auto_methods[0] if t.auto_methods else "landmarks"
        self.segments.append(seg)
        return seg

    def remove_segment(self, name: str) -> None:
        self.segments = [s for s in self.segments if s.name != name]
        self.joints = [j for j in self.joints if name not in (j.proximal, j.distal)]

    def rename_segment(self, old: str, new: str) -> str:
        new = new.strip()
        if not new or new == old:
            return old
        new = self._unique(new, {s.name for s in self.segments if s.name != old})
        self.segment(old).name = new
        for j in self.joints:
            j.proximal = new if j.proximal == old else j.proximal
            j.distal = new if j.distal == old else j.distal
        return new

    def candidates(self, joint_type: str, end: str) -> list[Segment]:
        """Segments that can be the proximal or distal end of a joint type."""
        jt = JOINT_TYPES[joint_type]
        allowed = jt.proximal_types if end == "proximal" else jt.distal_types
        return [s for s in self.segments if not allowed or s.type in allowed]

    def add_joint(self, type_key: str, proximal: str, distal: str, name: str | None = None) -> Joint:
        jt = JOINT_TYPES[type_key]
        p, d = self.segment(proximal), self.segment(distal)
        if p is d:
            raise JointModelError("A joint needs two different segments.")
        if jt.proximal_types and p.type not in jt.proximal_types:
            raise JointModelError(f"{jt.title} needs a {' or '.join(jt.proximal_types)} as the proximal segment.")
        if jt.distal_types and d.type not in jt.distal_types:
            raise JointModelError(f"{jt.title} needs a {' or '.join(jt.distal_types)} as the distal segment.")
        j = Joint(self._unique(name or jt.default_name, {x.name for x in self.joints}), type_key, p.name, d.name)
        self.joints.append(j)
        return j

    def remove_joint(self, name: str) -> None:
        self.joints = [j for j in self.joints if j.name != name]

    def rename_joint(self, old: str, new: str) -> str:
        new = new.strip()
        if not new or new == old:
            return old
        new = self._unique(new, {j.name for j in self.joints if j.name != old})
        self.joint(old).name = new
        return new

    def joint_side(self, j: Joint) -> str:
        """Side used for a joint's sign convention: the distal segment's side (right if it has none)."""
        side = self.segment(j.distal).side
        return side if side in ("right", "left") else "right"

    # ---- landmarks
    def landmark(self, seg: str, key: str) -> np.ndarray | None:
        lm = self.segment(seg).landmarks.get(key)
        if not lm:
            return None
        return landmark_value(lm["method"], lm.get("points"))

    def uses_landmarks(self, seg: str) -> bool:
        return self.segment(seg).axes_source == "landmarks"

    def missing(self, seg: str) -> list[str]:
        if not self.uses_landmarks(seg):
            return []
        return [d.name for d in self.segment(seg).stype.landmarks if self.landmark(seg, d.key) is None]

    # ---- axes
    def axes_sources(self, seg: str) -> list[tuple[str, str]]:
        """(key, title) of every way this segment's axes can be made: automatic methods, then landmarks."""
        from .auto_axes import AUTO_METHODS

        t = self.segment(seg).stype
        out = [(k, "Automatic: " + AUTO_METHODS[k].title) for k in t.auto_methods]
        return out + ([("landmarks", "Landmarks (placed by hand)")] if t.landmarks else [])

    def set_axes_source(self, seg: str, source: str) -> None:
        if source not in dict(self.axes_sources(seg)):
            raise JointModelError(f"{source!r} isn't a way to make axes for this segment.")
        self.segment(seg).axes_source = source

    def _auto_result(self, s: Segment, source: str, options: dict):
        from . import auto_axes

        key = (s.type, source, tuple(sorted(options.items())), id(s.mesh))
        if key not in self._auto_cache:
            try:
                self._auto_cache[key] = auto_axes.run(source, s.mesh, s.side, options)
            except JointModelError as err:
                self._auto_cache[key] = err
            except Exception as err:  # noqa: BLE001 - geometry can fail in many ways; report it
                self._auto_cache[key] = JointModelError(f"The automatic method failed: {err}")
        return self._auto_cache[key]

    def axes_for(self, seg: str, source: str):
        """A segment's ACS from a given source, without changing the model. Returns (acs or None, error)."""
        s = self.segment(seg)
        if s.mesh is None:
            return None, "No surface loaded."
        if source == "landmarks":
            t = s.stype
            L = {d.key: self.landmark(seg, d.key) for d in t.landmarks}
            miss = [d.name for d in t.landmarks if L[d.key] is None]
            if miss:
                return None, "Missing: " + ", ".join(miss)
            try:
                return RECIPES[t.recipe](L, s.side if s.side in ("right", "left") else "right", s.mesh), ""
            except JointModelError as err:
                return None, str(err)
        res = self._auto_result(s, source, s.auto_options if source == s.axes_source else {})
        return (None, str(res)) if isinstance(res, Exception) else (res.acs.copy(), "")

    def compute(self) -> None:
        """(Re)compute every segment's ACS; failures go in Segment.error."""
        for s in self.segments:
            s.acs, s.error, s.auto = None, "", None
            if s.mesh is None:
                s.error = "No surface loaded."
                continue
            if s.axes_source != "landmarks":
                res = self._auto_result(s, s.axes_source, s.auto_options)
                if isinstance(res, Exception):
                    s.error = str(res)
                else:
                    s.auto, s.acs = res, res.acs.copy()
                continue
            s.acs, s.error = self.axes_for(s.name, "landmarks")

    def checks(self) -> list[str]:
        """Plain-language warnings about axes, landmarks and joints that look wrong."""
        out = []
        for s in self.segments:
            if s.auto is not None:
                out += [f"{s.name}: {n}" for n in s.auto.notes]
            if s.type in ("femur", "tibia") and self.uses_landmarks(s.name):
                a, b, far = ("ME", "LE", "FS") if s.type == "femur" else ("MC", "LC", "TD")
                pa, pb, pf = self.landmark(s.name, a), self.landmark(s.name, b), self.landmark(s.name, far)
                if pa is not None and pb is not None:
                    w = np.linalg.norm(pb - pa)
                    if not 35 <= w <= 130:
                        out.append(f"{s.name}: medial and lateral points are {w:.0f} mm apart (expected about "
                                   "35–130 mm). Check they're on opposite sides.")
                    if pf is not None:
                        long = pf - (pa + pb) / 2
                        if np.linalg.norm(long) < 30:
                            out.append(f"{s.name}: the long-axis point is only {np.linalg.norm(long):.0f} mm from "
                                       "the joint end. Put it further along the shaft.")
                        elif abs(np.dot(_unit(long, "axis"), _unit(pb - pa, "axis"))) > 0.5:
                            out.append(f"{s.name}: the long axis is far from perpendicular to the medial–lateral "
                                       "line.")
        for j in self.joints:
            p, d = self.segment(j.proximal), self.segment(j.distal)
            if j.jtype.convention == "grood_suntay_knee" and p.side != d.side:
                out.append(f"{j.name}: {p.name} is {p.side} but {d.name} is {d.side}.")
        return out

    # ---- kinematics
    def _joint_acs(self, j: Joint, normalise: bool):
        A, B = self.segment(j.proximal).acs, self.segment(j.distal).acs
        if A is None or B is None:
            raise JointModelError(f"{j.name}: both segments need complete axes first.")
        B = B.copy()
        if normalise and j.reference is not None and j.reference.mode == "realign":
            Q0 = np.linalg.inv(A) @ j.reference.G0 @ B          # distal ACS in proximal ACS at the reference
            B[:3, :3] = B[:3, :3] @ Q0[:3, :3].T
        return A, B

    def reference_values(self, joint: str, normalise_mode: str | None = None) -> dict:
        """Joint angles/translations at the reference pose (before normalising)."""
        j = self.joint(joint)
        if j.reference is None:
            raise JointModelError(f"{j.name} has no reference pose.")
        A, B = self._joint_acs(j, normalise=(normalise_mode or j.reference.mode) == "realign")
        return _kinematics(j, self.joint_side(j), A, B, np.eye(4)[None], j.reference.G0[None])

    def joint_kinematics(self, joint: str, prox_poses: np.ndarray, dist_poses: np.ndarray,
                         normalise: bool = True) -> dict:
        """Joint angles (degrees) and translations (mm) per frame; NaN where either bone is untracked.

        prox_poses, dist_poses: (n, 4, 4) tracking poses of the two bones. With normalise and a
        reference pose, the values are relative to the reference (see Reference).
        """
        j = self.joint(joint)
        A, B = self._joint_acs(j, normalise)
        k = _kinematics(j, self.joint_side(j), A, B, prox_poses, dist_poses)
        if normalise and j.reference is not None:
            r = _kinematics(j, self.joint_side(j), A, B, np.eye(4)[None], j.reference.G0[None])
            if j.reference.mode == "subtract":
                k["angles"] = k["angles"] - r["angles"][0]
            k["translations"] = k["translations"] - r["translations"][0]
            k["normalised"] = j.reference.mode
        return k

    def segment_kinematics(self, seg: str, poses: np.ndarray, reference="first") -> dict:
        """A segment's own motion: rotation about its own axes and displacement of its origin.

        reference: "first" (first tracked frame), a frame index, or "lab" (the tracking's world axes).
        Rotations are the segment's orientation relative to the reference orientation, as an intrinsic
        Z-X-Y sequence about the segment's axes; translations are the origin's displacement, along the
        reference axes (or the lab axes).
        """
        s = self.segment(seg)
        if s.acs is None:
            raise JointModelError(f"{seg}: no axes yet.")
        n = len(poses)
        W = np.einsum("nij,jk->nik", np.nan_to_num(poses, nan=0.0), s.acs)
        ok = ~np.isnan(poses).any(axis=(1, 2))
        if isinstance(reference, str) and reference == "lab":
            R0, o0 = np.eye(3), np.zeros(3)
        else:
            r = int(np.argmax(ok)) if reference == "first" else int(reference)
            if not 0 <= r < n or not ok[r]:
                raise JointModelError("The reference frame isn't tracked.")
            R0, o0 = W[r, :3, :3], W[r, :3, 3]
        ang = np.full((n, 3), np.nan)
        tr = np.full((n, 3), np.nan)
        for f in np.flatnonzero(ok):
            ang[f] = np.degrees(decompose(R0.T @ W[f, :3, :3], SEGMENT_SEQUENCE))
            tr[f] = R0.T @ (W[f, :3, 3] - o0)
        names, short, tn, ts = SEGMENT_NAMES[s.stype.anatomical and s.axes_source != "inertial"]
        return {"angles": ang, "translations": tr, "names": names, "short": short, "t_names": tn, "t_short": ts,
                "title": "Segment rotation about its own axes (Z-X-Y)"}

    def compare_methods(self, joint: str, prox_pose: np.ndarray, dist_pose: np.ndarray) -> list[dict]:
        """Joint angles at one pose for every combination of the two segments' axis sources (no reference)."""
        j = self.joint(joint)
        rows = []
        for ps, pt in self.axes_sources(j.proximal):
            A, _ = self.axes_for(j.proximal, ps)
            for ds, dt in self.axes_sources(j.distal):
                B, _ = self.axes_for(j.distal, ds)
                row = {"proximal": ps, "distal": ds, "proximal_title": pt, "distal_title": dt,
                       "current": ps == self.segment(j.proximal).axes_source
                       and ds == self.segment(j.distal).axes_source}
                if A is None or B is None:
                    row["angles"] = row["translations"] = None
                else:
                    k = _kinematics(j, self.joint_side(j), A, B, prox_pose[None], dist_pose[None])
                    row["angles"], row["translations"] = k["angles"][0], k["translations"][0]
                rows.append(row)
        return rows

    # ---- reference poses
    def set_reference_from_tracking(self, joint: str, prox_poses, dist_poses, frames=None, mode: str = "subtract",
                                    source: dict | None = None) -> int:
        """Reference = the average relative pose over the given frames (default: every frame both are tracked).

        Returns the number of frames used.
        """
        P, D = np.asarray(prox_poses, float), np.asarray(dist_poses, float)
        n = min(len(P), len(D))
        idx = range(n) if frames is None else [f for f in frames if 0 <= f < n]
        G = [np.linalg.inv(P[f]) @ D[f] for f in idx if not (np.isnan(P[f]).any() or np.isnan(D[f]).any())]
        if not G:
            raise JointModelError("No frame has both bones tracked.")
        self.joint(joint).reference = Reference(_pose_mean(G), mode, dict(source or {}, frames_used=len(G)))
        return len(G)

    def set_reference_from_poses(self, joint: str, prox_pose, dist_pose, mode: str = "subtract",
                                 source: dict | None = None) -> None:
        """Reference = one pose of each bone, e.g. where they were in the scan (from their .tfm files)."""
        G0 = np.linalg.inv(np.asarray(prox_pose, float)) @ np.asarray(dist_pose, float)
        self.joint(joint).reference = Reference(G0, mode, dict(source or {}))

    # ---- files
    def to_dict(self) -> dict:
        return {
            "type": "bvr-joint-model", "version": MODEL_VERSION, "name": self.name,
            "segments": [{"name": s.name, "type": s.type, "side": s.side,
                          "surface": s.surface.to_dict() if s.surface is not None else None,
                          "axes_source": s.axes_source, "auto_options": s.auto_options, "landmarks": s.landmarks,
                          "preview_tracking": s.preview_tracking,
                          "acs": s.acs.tolist() if s.acs is not None else None} for s in self.segments],
            "joints": [{"name": j.name, "type": j.type, "proximal": j.proximal, "distal": j.distal,
                        "reference": j.reference.to_dict() if j.reference is not None else None}
                       for j in self.joints],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        self.path = Path(path)

    @classmethod
    def from_dict(cls, d: dict) -> "KinematicModel":
        from .models import BoneModel

        if d.get("type") != "bvr-joint-model":
            raise JointModelError("This isn't a joint model file.")
        if d.get("version", 1) < 2:
            d = _upgrade_v1(d)
        m = cls(name=d.get("name", "Model"))
        for sd in d.get("segments", []):
            if sd.get("type") not in SEGMENT_TYPES:
                raise JointModelError(f"Unknown segment type {sd.get('type')!r}.")
            s = Segment(name=sd["name"], type=sd["type"], side=sd.get("side", "right"),
                        surface=BoneModel.from_dict(sd["surface"]) if sd.get("surface") else None,
                        landmarks=sd.get("landmarks") or {}, auto_options=dict(sd.get("auto_options") or {}),
                        preview_tracking=sd.get("preview_tracking"))
            src = sd.get("axes_source", "landmarks")
            m.segments.append(s)
            s.axes_source = src if src in dict(m.axes_sources(s.name)) else m.axes_sources(s.name)[0][0]
            if sd.get("acs") is not None:
                s.acs = np.array(sd["acs"], float)
        for jd in d.get("joints", []):
            if jd.get("type") not in JOINT_TYPES:
                raise JointModelError(f"Unknown joint type {jd.get('type')!r}.")
            j = Joint(jd["name"], jd["type"], jd["proximal"], jd["distal"])
            if jd.get("reference"):
                j.reference = Reference.from_dict(jd["reference"])
            m.joints.append(j)
        return m

    @classmethod
    def load(cls, path: str | Path, build_surfaces: bool = True) -> "KinematicModel":
        m = cls.from_dict(json.loads(Path(path).read_text()))
        m.path = Path(path)
        if build_surfaces:
            for s in m.segments:
                if s.surface is not None:
                    s.surface.build()
            m.compute()
        return m


def _upgrade_v1(d: dict) -> dict:
    """Version-1 files (one template, bones by role) -> segments and joints."""
    side = d.get("side", "right")
    if d.get("template") == "knee_grood_suntay":
        roles = {"femur": ("Femur", "femur"), "tibia": ("Tibia", "tibia")}
        joints = [{"name": "Knee", "type": "knee_grood_suntay", "proximal": "Femur", "distal": "Tibia"}]
    elif d.get("template") == "generic_inertial":
        roles = {"proximal": ("Proximal bone", "generic"), "distal": ("Distal bone", "generic")}
        joints = [{"name": "Joint", "type": "cardan_xyz", "proximal": "Proximal bone", "distal": "Distal bone"}]
        side = "none"
    else:
        raise JointModelError(f"Unknown template {d.get('template')!r}.")
    segs = []
    for role, (name, typ) in roles.items():
        b = d.get("bones", {}).get(role, {})
        src = b.get("axes_source", "landmarks" if typ != "generic" else "inertial")
        segs.append({"name": name, "type": typ, "side": side, "surface": b.get("surface"), "axes_source": src,
                     "auto_options": b.get("auto_options"), "landmarks": b.get("landmarks"),
                     "preview_tracking": b.get("preview_tracking"), "acs": b.get("acs")})
    return {"type": "bvr-joint-model", "version": 2, "name": d.get("name", "Model"), "segments": segs,
            "joints": joints}


JointModel = KinematicModel      # older name


def _kinematics(j: "Joint", side: str, A: np.ndarray, B: np.ndarray, prox_poses, dist_poses) -> dict:
    """Angles and translations of joint j for proximal/distal ACSs A, B over (n, 4, 4) poses."""
    conv = j.convention
    n = min(len(prox_poses), len(dist_poses))
    ang = np.full((n, 3), np.nan)
    tr = np.full((n, 3), np.nan)
    sa, st = conv["signs"][side]
    seq = conv["sequence"]
    idx = {"X": 0, "Y": 1, "Z": 2}
    for f in range(n):
        P, D = prox_poses[f], dist_poses[f]
        if np.isnan(P).any() or np.isnan(D).any():
            continue
        Pw, Dw = P @ A, D @ B                                  # axes in the world
        R = Pw[:3, :3].T @ Dw[:3, :3]                          # distal relative to proximal
        ang[f] = np.degrees(decompose(R, seq)) * sa
        e1 = Pw[:3, idx[seq[0]]]
        e3 = Dw[:3, idx[seq[2]]]
        e2 = np.cross(e3, e1)
        nn = np.linalg.norm(e2)
        e2 = e2 / nn if nn > 1e-9 else Pw[:3, idx[seq[1]]]
        d = Dw[:3, 3] - Pw[:3, 3]
        q = np.linalg.solve(np.column_stack([e1, e2, e3]), d)  # d = q1 e1 + q2 e2 + q3 e3
        tr[f] = q * st
    return {"angles": ang, "translations": tr, "names": conv["angles"], "short": conv["short"],
            "t_names": conv["translations"], "t_short": conv["short_t"], "title": conv["title"]}
