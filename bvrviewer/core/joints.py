"""Joint models: landmarks on bone surfaces -> anatomical coordinate systems -> joint angles.

Vocabulary
----------
* **Template**: what a model needs. Bone roles (e.g. femur, tibia), the landmarks for each role,
  the recipe that turns landmarks into the bone's anatomical coordinate system (ACS), and the
  joints between roles with their angle convention. Built-in templates are in TEMPLATES.
* **Joint model**: a template applied to one participant's bone surfaces. It stores which surface
  fills each role and the landmarks picked or painted on it (in that surface's own model space,
  the space the .tra poses act on), so one model serves every trial of that participant.

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

MODEL_VERSION = 1


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


# ---------------------------------------------------------------- templates

@dataclass(frozen=True)
class LandmarkDef:
    key: str
    name: str
    hint: str
    methods: tuple = ("point", "region")


@dataclass(frozen=True)
class BoneRole:
    key: str
    title: str
    recipe: str                     # name in RECIPES
    landmarks: tuple = ()
    axes_note: str = ""
    auto_methods: tuple = ()        # keys in core.auto_axes.AUTO_METHODS that can replace the landmarks


@dataclass(frozen=True)
class JointDef:
    key: str
    title: str
    proximal: str
    distal: str
    convention: str                 # name in JOINT_CONVENTIONS


@dataclass(frozen=True)
class Template:
    key: str
    title: str
    summary: str
    references: tuple
    bones: tuple
    joints: tuple
    sided: bool = True

    def role(self, key: str) -> BoneRole:
        for b in self.bones:
            if b.key == key:
                return b
        raise KeyError(key)


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

TEMPLATES: dict[str, Template] = {}


def _register(t: Template) -> Template:
    TEMPLATES[t.key] = t
    return t


KNEE = _register(Template(
    key="knee_grood_suntay",
    title="Knee (tibiofemoral): Grood & Suntay, ISB femur and tibia axes",
    summary=("Femur and tibia axes named as ISB (X anterior, Y superior, Z right), joined by the Grood & "
             "Suntay joint coordinate system: flexion about the femur's Z axis, internal rotation about "
             "the tibia's Y axis, ad/abduction about the floating axis between them. Works with partial "
             "bones from biplanar videoradiography: where the hip centre or ankle isn't in the model, use the "
             "centre of a band painted around the end of the shaft. Each bone's axes can instead come from an "
             "automatic method fitted to its surface (no landmarks)."),
    references=("Grood ES, Suntay WJ (1983) J Biomech Eng 105:136–144",
                "Wu G et al. (2002) ISB recommendation, part I: ankle, hip and spine. J Biomech 35:543–548"),
    bones=(
        BoneRole("femur", "Femur", "isb_femur", (
            LandmarkDef("ME", "Medial epicondyle", "Most prominent point of the medial epicondyle.",
                        ("point", "region")),
            LandmarkDef("LE", "Lateral epicondyle", "Most prominent point of the lateral epicondyle.",
                        ("point", "region")),
            LandmarkDef("FS", "Superior point (hip centre or shaft centre)",
                        "Hip joint centre: paint the femoral head and use a sphere fit. On a partial femur, "
                        "paint a band around the most proximal part of the shaft and use its centre.",
                        ("region", "sphere", "point")),
        ), "Origin at the epicondyle midpoint; Y towards the superior point; Z right; X anterior.",
            ("femur_cylinder", "femur_spheres")),
        BoneRole("tibia", "Tibia", "knee_tibia", (
            LandmarkDef("MC", "Medial condyle", "Most medial point of the medial tibial condyle (plateau rim).",
                        ("point", "region")),
            LandmarkDef("LC", "Lateral condyle", "Most lateral point of the lateral tibial condyle (plateau rim).",
                        ("point", "region")),
            LandmarkDef("TD", "Distal point (malleoli midpoint or shaft centre)",
                        "Midpoint between the malleoli if the model includes them. On a partial tibia, paint a "
                        "band around the most distal part of the shaft and use its centre.",
                        ("region", "point")),
        ), "Origin at the condyle midpoint; Y from the distal point to the origin; Z right; X anterior.",
            ("tibia_ellipse", "tibia_centroids", "tibia_inertia")),
    ),
    joints=(JointDef("knee", "Knee", "femur", "tibia", "grood_suntay_knee"),),
))

GENERIC = _register(Template(
    key="generic_inertial",
    title="Generic joint: two bones, automatic inertial axes",
    summary=("No landmarks needed: each bone's axes are its principal axes of inertia (axis 1 is the long "
             "axis). Angles are a Cardan X-Y-Z sequence of the distal bone relative to the proximal bone. "
             "Useful for any pair of bones (e.g. carpal bones) and for checking a workflow; the angles have "
             "no clinical names."),
    references=("Principal inertial axes, as used for carpal bones: Coburn JC et al. (2007) J Biomech 40:203–209",),
    bones=(BoneRole("proximal", "Proximal bone", "inertial"), BoneRole("distal", "Distal bone", "inertial")),
    joints=(JointDef("joint", "Joint", "proximal", "distal", "cardan_xyz"),),
    sided=False,
))


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


# ---------------------------------------------------------------- joint model

@dataclass
class BoneEntry:
    role: str
    surface: object = None                # a core.models.BoneModel (surface file + model space settings)
    landmarks: dict = field(default_factory=dict)    # key -> {"method": str, "points": [[x,y,z], ...]}
    preview_tracking: str | None = None   # optional .tra to preview the joint at a frame
    axes_source: str = "landmarks"        # "landmarks", or an automatic method key (core.auto_axes)
    auto_options: dict = field(default_factory=dict)  # e.g. {"flip_ap": True, "flip_long": False}

    # runtime
    acs: np.ndarray | None = None
    error: str = ""
    auto: object = None                   # core.auto_axes.AutoResult when axes_source is automatic


@dataclass
class JointModel:
    template_key: str
    side: str = "right"
    name: str = "Joint model"
    bones: dict = field(default_factory=dict)        # role -> BoneEntry
    path: Path | None = None
    _auto_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def template(self) -> Template:
        return TEMPLATES[self.template_key]

    @classmethod
    def new(cls, template_key: str, side: str = "right") -> "JointModel":
        t = TEMPLATES[template_key]
        m = cls(template_key=template_key, side=side if t.sided else "right", name=t.title.split(":")[0])
        for role in t.bones:
            m.bones[role.key] = BoneEntry(role=role.key)
        return m

    # ---- landmarks
    def landmark(self, role: str, key: str) -> np.ndarray | None:
        lm = self.bones[role].landmarks.get(key)
        if not lm:
            return None
        return landmark_value(lm["method"], lm.get("points"))

    def uses_landmarks(self, role: str) -> bool:
        return self.bones[role].axes_source == "landmarks"

    def set_axes_source(self, role: str, source: str) -> None:
        if source != "landmarks" and source not in self.template.role(role).auto_methods:
            raise JointModelError(f"{source!r} isn't an automatic method for this bone.")
        self.bones[role].axes_source = source

    def missing(self, role: str) -> list[str]:
        if not self.uses_landmarks(role):
            return []
        r = self.template.role(role)
        return [d.name for d in r.landmarks if self.landmark(role, d.key) is None]

    # ---- coordinate systems
    def compute(self) -> None:
        """(Re)compute every bone's ACS from its landmarks or automatic method; failures go in BoneEntry.error."""
        for role in self.template.bones:
            e = self.bones[role.key]
            e.acs, e.error, e.auto = None, "", None
            if e.surface is None or getattr(e.surface, "mesh", None) is None:
                e.error = "No surface loaded."
                continue
            if e.axes_source != "landmarks":
                self._compute_auto(role, e)
                continue
            miss = self.missing(role.key)
            if miss:
                e.error = "Missing: " + ", ".join(miss)
                continue
            L = {d.key: self.landmark(role.key, d.key) for d in role.landmarks}
            try:
                e.acs = RECIPES[role.recipe](L, self.side, e.surface.mesh)
            except JointModelError as err:
                e.error = str(err)
        return None

    def _auto_result(self, role_key: str, source: str, options: dict):
        """AutoResult (or the JointModelError it raised) for a bone and automatic method, cached."""
        from . import auto_axes

        e = self.bones[role_key]
        key = (role_key, source, tuple(sorted(options.items())), id(e.surface.mesh))
        if key not in self._auto_cache:
            try:
                self._auto_cache[key] = auto_axes.run(source, e.surface.mesh, self.side, options)
            except JointModelError as err:
                self._auto_cache[key] = err
            except Exception as err:  # noqa: BLE001 - geometry can fail in many ways; report it
                self._auto_cache[key] = JointModelError(f"The automatic method failed: {err}")
        return self._auto_cache[key]

    def _compute_auto(self, role, e) -> None:
        res = self._auto_result(role.key, e.axes_source, e.auto_options)
        if isinstance(res, Exception):
            e.error = str(res)
        else:
            e.auto, e.acs = res, res.acs.copy()

    def axes_sources(self, role_key: str) -> list[tuple[str, str]]:
        """(key, title) of every way this bone's axes can be made: landmarks, then automatic methods."""
        from .auto_axes import AUTO_METHODS

        r = self.template.role(role_key)
        out = [("landmarks", "Landmarks (placed by hand)")] if r.landmarks else []
        return out + [(k, "Automatic: " + AUTO_METHODS[k].title) for k in r.auto_methods]

    def axes_for(self, role_key: str, source: str):
        """This bone's ACS from a given source, without changing the model. Returns (acs or None, error)."""
        e = self.bones[role_key]
        if e.surface is None or getattr(e.surface, "mesh", None) is None:
            return None, "No surface loaded."
        if source == "landmarks":
            r = self.template.role(role_key)
            L = {d.key: self.landmark(role_key, d.key) for d in r.landmarks}
            miss = [d.name for d in r.landmarks if L[d.key] is None]
            if miss:
                return None, "Missing: " + ", ".join(miss)
            try:
                return RECIPES[r.recipe](L, self.side, e.surface.mesh), ""
            except JointModelError as err:
                return None, str(err)
        opts = e.auto_options if source == e.axes_source else {}
        res = self._auto_result(role_key, source, opts)
        return (None, str(res)) if isinstance(res, Exception) else (res.acs.copy(), "")

    def compare_methods(self, joint_key: str, prox_pose: np.ndarray, dist_pose: np.ndarray) -> list[dict]:
        """Joint angles at one pose for every combination of the two bones' axis sources."""
        j = next(j for j in self.template.joints if j.key == joint_key)
        rows = []
        for ps, pt in self.axes_sources(j.proximal):
            A, _ = self.axes_for(j.proximal, ps)
            for ds, dt in self.axes_sources(j.distal):
                B, _ = self.axes_for(j.distal, ds)
                row = {"proximal": ps, "distal": ds, "proximal_title": pt, "distal_title": dt,
                       "current": ps == self.bones[j.proximal].axes_source and ds == self.bones[j.distal].axes_source}
                if A is None or B is None:
                    row["angles"] = row["translations"] = None
                else:
                    k = _kinematics(j, self.side, A, B, prox_pose[None], dist_pose[None])
                    row["angles"], row["translations"] = k["angles"][0], k["translations"][0]
                rows.append(row)
        return rows

    def checks(self) -> list[str]:
        """Plain-language warnings about landmarks that look wrong."""
        out = []
        t = self.template
        for r in t.bones:
            e = self.bones[r.key]
            if e.auto is not None:
                out += [f"{r.title}: {n}" for n in e.auto.notes]
        if t.key == KNEE.key:
            for role, a, b, far, lo, hi in [("femur", "ME", "LE", "FS", 35, 130), ("tibia", "MC", "LC", "TD", 35, 130)]:
                if not self.uses_landmarks(role):
                    continue
                pa, pb, pf = self.landmark(role, a), self.landmark(role, b), self.landmark(role, far)
                if pa is not None and pb is not None:
                    w = np.linalg.norm(pb - pa)
                    if not lo <= w <= hi:
                        out.append(f"{role.title()}: medial and lateral points are {w:.0f} mm apart (expected "
                                   f"about {lo}–{hi} mm). Check they're on opposite sides.")
                    if pf is not None:
                        mid = (pa + pb) / 2
                        long = pf - mid
                        if np.linalg.norm(long) < 30:
                            out.append(f"{role.title()}: the long-axis point is only {np.linalg.norm(long):.0f} mm "
                                       "from the joint end. Put it further along the shaft.")
                        else:
                            cosang = abs(np.dot(_unit(long, 'axis'), _unit(pb - pa, 'axis')))
                            if cosang > 0.5:
                                out.append(f"{role.title()}: the long axis is {np.degrees(np.arccos(cosang)):.0f}° "
                                           "from the medial–lateral line (expected near 90°).")
        return out

    # ---- kinematics
    def joint_kinematics(self, joint_key: str, prox_poses: np.ndarray, dist_poses: np.ndarray) -> dict:
        """Joint angles (degrees) and translations (mm) for each frame.

        prox_poses, dist_poses: (n, 4, 4) tracking poses of the two bones (NaN where untracked).
        Returns {"angles": (n,3), "translations": (n,3), "names": ..., "short": ...}; NaN where either
        bone is untracked.
        """
        j = next(j for j in self.template.joints if j.key == joint_key)
        A = self.bones[j.proximal].acs
        B = self.bones[j.distal].acs
        if A is None or B is None:
            raise JointModelError("Both bones need complete axes first.")
        return _kinematics(j, self.side, A, B, prox_poses, dist_poses)

    # ---- files
    def to_dict(self) -> dict:
        return {
            "type": "bvr-joint-model", "version": MODEL_VERSION, "name": self.name,
            "template": self.template_key, "side": self.side,
            "bones": {role: {"surface": e.surface.to_dict() if e.surface is not None else None,
                             "landmarks": e.landmarks, "preview_tracking": e.preview_tracking,
                             "axes_source": e.axes_source, "auto_options": e.auto_options,
                             "acs": e.acs.tolist() if e.acs is not None else None}
                      for role, e in self.bones.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        self.path = Path(path)

    @classmethod
    def from_dict(cls, d: dict) -> "JointModel":
        from .models import BoneModel

        if d.get("type") != "bvr-joint-model":
            raise JointModelError("This isn't a joint model file.")
        if d.get("template") not in TEMPLATES:
            raise JointModelError(f"Unknown template {d.get('template')!r}.")
        m = cls.new(d["template"], d.get("side", "right"))
        m.name = d.get("name", m.name)
        for role, b in d.get("bones", {}).items():
            if role not in m.bones:
                continue
            e = m.bones[role]
            e.surface = BoneModel.from_dict(b["surface"]) if b.get("surface") else None
            e.landmarks = b.get("landmarks", {}) or {}
            e.preview_tracking = b.get("preview_tracking")
            src = b.get("axes_source", "landmarks")
            e.axes_source = src if src == "landmarks" or src in m.template.role(role).auto_methods else "landmarks"
            e.auto_options = dict(b.get("auto_options") or {})
            if b.get("acs") is not None:
                e.acs = np.array(b["acs"], float)
        return m

    @classmethod
    def load(cls, path: str | Path, build_surfaces: bool = True) -> "JointModel":
        m = cls.from_dict(json.loads(Path(path).read_text()))
        m.path = Path(path)
        if build_surfaces:
            for e in m.bones.values():
                if e.surface is not None:
                    e.surface.build()
            m.compute()
        return m


def _kinematics(j: JointDef, side: str, A: np.ndarray, B: np.ndarray, prox_poses, dist_poses) -> dict:
    """Angles and translations of joint j for proximal/distal ACSs A, B over (n, 4, 4) poses."""
    conv = JOINT_CONVENTIONS[j.convention]
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
