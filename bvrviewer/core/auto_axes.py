"""Automatic anatomical axes from bone surface geometry (no landmarks).

Each method is our own implementation, written from the published descriptions of the algorithms
compared by Modenese & Renault (2021, STAPLE) and of their sources. None of the original code is
used. The methods are adapted to the partial bones typical of biplanar videoradiography: the long
axis comes from the shaft that is present (cross-section centroids), never from the hip or ankle.

Every method returns an AutoResult: the ACS (ISB axis names: X anterior, Y superior, Z right; in
the surface's model space) plus what was fitted, so the Model Builder can draw it and report the
fit quality.

Which way is anterior can't be read from a fitted cylinder or ellipse alone. It is taken from the
posterior overhang of the joint end over the shaft line: the femoral condyles reach about twice as
far behind the shaft as in front of it, and the tibial plateau overhangs posteriorly. Z (right) then
follows from X and Y, so the side of the limb isn't needed to build the axes. When the overhang is
not clear-cut the result carries a note, and the user can flip anterior/posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .joints import JointModelError, inertial_frame, sphere_fit


@dataclass
class AutoResult:
    acs: np.ndarray
    primitives: list = field(default_factory=list)   # dicts: {"kind": "cylinder"/"sphere"/"ellipse"/"line", ...}
    points: np.ndarray | None = None                  # surface points the fit used (for display)
    quality: dict = field(default_factory=dict)       # e.g. {"RMS residual (mm)": 0.6}
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class AutoMethod:
    key: str
    title: str
    bone: str                 # "femur" or "tibia"
    summary: str
    references: tuple
    fn: object


# ---------------------------------------------------------------- shared geometry

def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise JointModelError("Degenerate direction while building axes.")
    return v / n


def _surface(mesh, max_points: int = 60000, seed: int = 0):
    """Points and outward unit normals of a surface (subsampled for speed)."""
    m = mesh.triangulate().compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True,
                                           consistent_normals=True, split_vertices=False)
    P = np.asarray(m.points, float)
    N = np.asarray(m.point_data["Normals"], float)
    if len(P) > max_points:
        idx = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P, N = P[idx], N[idx]
    return P, N


def _section(mesh, origin, normal):
    """Cross-section contour points of the surface in a plane."""
    s = mesh.slice(normal=normal, origin=origin)
    return np.asarray(s.points, float)


def _basis(n):
    n = _unit(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(n, a))
    return u, np.cross(n, u)


def _hull_area_centroid(pts2):
    from scipy.spatial import ConvexHull

    h = ConvexHull(pts2)
    v = pts2[h.vertices]
    x, y = v[:, 0], v[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    A = cr.sum() / 2
    cx = ((x + xs) * cr).sum() / (6 * A)
    cy = ((y + ys) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy]), v


@dataclass
class _Long:
    """The bone's long axis from its shaft, pointing from the joint end (epiphysis) to the shaft end."""
    axis: np.ndarray          # unit, joint end -> shaft end
    point: np.ndarray         # a point on the shaft line
    t_joint: float            # position of the joint-end tip along axis (relative to point)
    t_shaft: float            # position of the shaft-end tip
    sizes: list               # (t, width) samples used to find the ends
    shaft_centroids: np.ndarray
    from_shaft: bool


def long_axis(mesh, P, flip: bool = False) -> _Long:
    """Shaft line of a long bone (whole or partial).

    The first principal inertial axis gives the direction. Cross-sections along it find the joint end
    (the wider end), and the centroids of the narrow, shaft part give the line. flip swaps the ends,
    for whole bones where both ends are wide.
    """
    A, _ = inertial_frame(mesh)
    ax, c = A[:3, 0], A[:3, 3]
    t = (P - c) @ ax
    u, v = _basis(ax)
    lo, hi = t.min(), t.max()
    sizes = []
    for tt in np.arange(lo + 2, hi - 1.5, 3.0):
        sl = P[np.abs(t - tt) < 1.5]
        if len(sl) < 10:
            continue
        q = np.c_[(sl - c) @ u, (sl - c) @ v]
        sizes.append((tt, float(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))))
    if len(sizes) < 6:
        raise JointModelError("The bone is too short to find its long axis.")
    ts, ws = np.array(sizes).T
    k = max(2, len(ts) // 4)
    joint_at_low = ws[:k].mean() > ws[-k:].mean()
    if flip:
        joint_at_low = not joint_at_low
    if not joint_at_low:
        ax = -ax
        t = -t
        ts, ws = -ts[::-1], ws[::-1]
    # shaft: from the shaft end inwards while the section stays narrow
    wmin = ws[len(ws) // 2:].min()
    shaft_t = [tt for tt, w in zip(ts, ws) if w < 1.6 * wmin and tt > ts[len(ts) // 3]]
    shaft_t = [tt for tt in shaft_t if tt < t.max() - 4]
    cents = []
    for tt in shaft_t:
        sec = _section(mesh, c + ax * tt, ax)
        if len(sec) > 20:
            cents.append(sec.mean(0))
    cents = np.array(cents)
    if len(cents) >= 4 and np.ptp((cents - c) @ ax) > 20:
        cm = cents.mean(0)
        _, _, vt = np.linalg.svd(cents - cm)
        d = vt[0] if vt[0] @ ax > 0 else -vt[0]
        point, from_shaft = cm, True
    else:
        d, point, from_shaft = ax, c, False
    tt = (P - point) @ d
    return _Long(d, point, float(tt.min()), float(tt.max()), sizes, cents, from_shaft)


def _project_perp(v, n):
    return v - (v @ n) * n


def posterior_by_overhang(P, L: _Long, ap, from_end: tuple, at_low_end: bool):
    """Which way along ±ap the joint end overhangs the shaft line more (that side is posterior).

    from_end: (near, far) distances from the joint-end tip, in mm, of the band of surface to use.
    Returns (posterior unit vector, ratio of the larger to the smaller overhang).
    """
    d = P - L.point
    t = d @ L.axis
    off = (d - np.outer(t, L.axis)) @ ap
    tip = t.min() if at_low_end else t.max()
    dist = np.abs(t - tip)
    m = (dist >= from_end[0]) & (dist < from_end[1])
    if m.sum() < 50:
        raise JointModelError("Too little bone near the joint end to tell anterior from posterior.")
    plus, minus = float(off[m].max()), float(-off[m].min())
    ratio = max(plus, minus) / max(min(plus, minus), 1e-6)
    return (ap if plus > minus else -ap), ratio


def _frame_z_primary(origin, z, y_hint):
    """Z given exactly (e.g. a fitted flexion axis), Y = y_hint made perpendicular, X = Y x Z."""
    z = _unit(z)
    y = _unit(_project_perp(np.asarray(y_hint, float), z))
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def _frame_y_primary(origin, y, z_hint):
    y = _unit(y)
    z = _unit(_project_perp(np.asarray(z_hint, float), y))
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


# ---------------------------------------------------------------- fitting

def fit_cylinder(P, axis0, point0=None, radius0=None, trim: float = 2.5):
    """Least-squares cylinder (robust, then trimmed). Returns (point on axis, unit axis, radius, rms, inliers)."""
    P = np.asarray(P, float)
    a0 = _unit(np.asarray(axis0, float))
    u, v = _basis(a0)
    if point0 is None or radius0 is None:      # circle fit in the plane across the axis
        q = np.c_[P @ u, P @ v]
        A = np.c_[2 * q, np.ones(len(q))]
        sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        cc = sol[:2]
        radius0 = float(np.sqrt(max(sol[2] + cc @ cc, 1e-6)))
        point0 = cc[0] * u + cc[1] * v + (P @ a0).mean() * a0
    point0 = np.asarray(point0, float)

    def unpack(x):
        ax = _unit(a0 + x[0] * u + x[1] * v)
        p = point0 + x[2] * u + x[3] * v
        return p, ax, x[4]

    def res(x, Q):
        p, ax, r = unpack(x)
        d = Q - p
        return np.linalg.norm(d - np.outer(d @ ax, ax), axis=1) - r

    x0 = np.array([0, 0, 0, 0, radius0], float)
    keep = np.ones(len(P), bool)
    for _ in range(3):
        sol = least_squares(res, x0, args=(P[keep],), loss="soft_l1", f_scale=1.0)
        x0 = sol.x
        r_all = np.abs(res(x0, P))
        mad = np.median(r_all[keep]) * 1.4826 + 1e-6
        keep = r_all < max(trim * mad, 0.75)
        if keep.sum() < 50:
            raise JointModelError("The cylinder fit found too few condyle points.")
    p, ax, r = unpack(x0)
    rms = float(np.sqrt(np.mean(res(x0, P[keep]) ** 2)))
    return p, ax, float(abs(r)), rms, keep


def robust_sphere(P, trim: float = 2.5):
    """Sphere fit that tolerates stray points: algebraic start, robust geometric fit, then trimming."""
    P = np.asarray(P, float)
    c0, r0 = sphere_fit(P)

    def res(x, Q):
        return np.linalg.norm(Q - x[:3], axis=1) - x[3]

    x = np.r_[c0, r0]
    keep = np.ones(len(P), bool)
    for _ in range(3):
        x = least_squares(res, x, args=(P[keep],), loss="soft_l1", f_scale=0.5).x
        d = np.abs(res(x, P))
        mad = np.median(d[keep]) * 1.4826 + 1e-6
        keep = d < max(trim * mad, 0.75)
        if keep.sum() < 30:
            raise JointModelError("The sphere fit found too few condyle points.")
    rms = float(np.sqrt(np.mean(res(x, P[keep]) ** 2)))
    return x[:3], float(abs(x[3])), rms, keep


# ---------------------------------------------------------------- femur

def _femur_condyles(mesh, options):
    """Shared femur steps: long axis, epiphysis, medial-lateral and anterior directions, condyle points."""
    P, N = _surface(mesh)
    L = long_axis(mesh, P, flip=options.get("flip_long", False))
    sup = L.axis                                        # joint end -> shaft = superior for the femur
    s = (P - L.point) @ sup                             # height along the shaft line
    s0 = s.min()
    epi = s < s0 + 45                                   # distal 45 mm: the condyles
    if epi.sum() < 200:
        raise JointModelError("Couldn't find the femoral condyles at the wider end of the bone.")
    E = P[epi]
    q = np.array([_project_perp(p - L.point, sup) for p in E])
    w, V = np.linalg.eigh(np.cov(q.T))
    ml = _unit(_project_perp(V[:, -1], sup))           # widest direction across the condyles
    ap = np.cross(sup, ml)
    # the condyles reach much further behind the shaft line than in front of it
    post, ratio = posterior_by_overhang(P, L, ap, (10, 45), at_low_end=True)
    if options.get("flip_ap", False):
        post = -post
    ant = -post
    right = np.cross(ant, sup)                          # ISB: Z = X x Y
    width = float(np.ptp(E @ ml))
    mlc = (E @ ml).min() + width / 2                    # notch centre (medial-lateral)
    c_post = (E @ post).mean()
    # articular condyle surface: posterior or distal facing, posterior half, away from the notch
    En = N[epi]
    facing = (En @ post > 0.35) | (En @ (-sup) > 0.5)
    behind = (E @ post) > c_post - 0.15 * np.ptp(E @ post)
    off_notch = np.abs(E @ ml - mlc) > 0.12 * width
    low = (s[epi] - s0) < 35
    cond = facing & behind & off_notch & low
    if cond.sum() < 200:
        raise JointModelError("Too few points on the posterior condyles.")
    return dict(P=P, L=L, sup=sup, ml=ml, ant=ant, post=post, right=right, C=E[cond],
                Cside=np.sign(E[cond] @ ml - mlc), width=width, ratio=ratio, flipped=options.get("flip_ap", False))


def femur_cylinder(mesh, side, options) -> AutoResult:
    g = _femur_condyles(mesh, options)
    p, ax, r, rms, keep = fit_cylinder(g["C"], g["ml"])
    if ax @ g["right"] < 0:
        ax = -ax
    t = (g["C"][keep] - p) @ ax
    origin = p + ax * (t.min() + t.max()) / 2
    acs = _frame_z_primary(origin, ax, g["sup"])
    res = AutoResult(acs, points=g["C"][keep])
    res.primitives = [{"kind": "cylinder", "center": origin, "direction": ax, "radius": r,
                       "height": float(t.max() - t.min())},
                      {"kind": "line", "points": g["L"].shaft_centroids}]
    res.quality = {"Radius": f"{r:.1f} mm", "RMS residual": f"{rms:.2f} mm", "Points used": int(keep.sum()),
                   "Long axis": "shaft centroids" if g["L"].from_shaft else "inertia"}
    _femur_notes(res, g)
    return res


def femur_spheres(mesh, side, options) -> AutoResult:
    g = _femur_condyles(mesh, options)
    C, sd = g["C"], g["Cside"]
    # posterior condyle surfaces only, as in sphere fits to the posterior condyles
    post_only = ((C - C.mean(0)) @ g["post"]) > -0.05 * g["width"]
    fits = []
    for k in (-1, 1):
        pts = C[(sd == k) & post_only]
        c, r, rms, keep = robust_sphere(pts)
        fits.append((c, r, rms, pts[keep]))
    (c1, r1, e1, k1), (c2, r2, e2, k2) = fits
    ax = _unit(c2 - c1)
    if ax @ g["right"] < 0:
        ax = -ax
    origin = (c1 + c2) / 2
    acs = _frame_z_primary(origin, ax, g["sup"])
    res = AutoResult(acs, points=np.r_[k1, k2])
    res.primitives = [{"kind": "sphere", "center": c1, "radius": r1}, {"kind": "sphere", "center": c2, "radius": r2},
                      {"kind": "line", "points": g["L"].shaft_centroids}]
    res.quality = {"Radii": f"{r1:.1f}, {r2:.1f} mm", "RMS residual": f"{np.hypot(e1, e2) / np.sqrt(2):.2f} mm",
                   "Points used": int(len(k1) + len(k2)),
                   "Long axis": "shaft centroids" if g["L"].from_shaft else "inertia"}
    _femur_notes(res, g)
    return res


def _femur_notes(res, g):
    res.quality["Overhang ratio"] = f"{g['ratio']:.2f} (posterior : anterior)"
    if g["ratio"] < 1.3 and not g["flipped"]:
        res.notes.append(f"The condyles don't clearly overhang the shaft posteriorly (ratio {g['ratio']:.2f}), so "
                         "anterior may be wrong. Check the red X axis points forwards; use Flip anterior/posterior "
                         "if not.")
    if not g["L"].from_shaft:
        res.notes.append("Too little shaft to fit a shaft line; the long axis is the inertial axis.")
    if not 50 <= g["width"] <= 110:
        res.notes.append(f"Condyle width {g['width']:.0f} mm is unusual for a femur. Check the bone and the "
                         "superior/inferior direction.")


# ---------------------------------------------------------------- tibia

def _tibia_common(mesh, options):
    P, N = _surface(mesh)
    L = long_axis(mesh, P, flip=options.get("flip_long", False))
    sup = -L.axis                                       # shaft -> joint end = superior for the tibia
    s = (P - L.point) @ sup
    top = s.max()
    return P, N, L, sup, s, top


def _tibia_anterior(mesh, P, L, sup, s, top, ml, options):
    """Anterior from the posterior overhang of the plateau over the shaft line."""
    ap = _unit(np.cross(ml, sup))
    # L.axis points from the joint end to the shaft, so the joint end is at its low end
    post, ratio = posterior_by_overhang(P, L, ap, (5, 30), at_low_end=True)
    ant = -post
    if options.get("flip_ap", False):
        ant = -ant
    return ant, ratio


def tibia_ellipse(mesh, side, options) -> AutoResult:
    """Largest proximal cross-section, fitted with an ellipse."""
    P, N, L, sup, s, top = _tibia_common(mesh, options)
    u, v = _basis(sup)
    best = None
    for d in np.arange(2, 31, 1.5):
        sec = _section(mesh, L.point + sup * (top - d), sup)
        if len(sec) < 30:
            continue
        q = np.c_[sec @ u, sec @ v]
        area, cen, hull = _hull_area_centroid(q)
        if best is None or area > best[0]:
            best = (area, cen, q, top - d, sec)
    if best is None:
        raise JointModelError("Couldn't find the tibial plateau cross-section.")
    area, cen, q, h, sec = best
    # ellipse from the second moments of the contour (principal directions and semi-axes)
    w, V = np.linalg.eigh(np.cov((q - q.mean(0)).T))
    major2 = V[:, -1]
    ml = _unit(major2[0] * u + major2[1] * v)
    origin = cen[0] * u + cen[1] * v + (L.point @ sup + h) * sup      # centroid, on the section plane
    ant, sk = _tibia_anterior(mesh, P, L, sup, s, top, ml, options)
    right = np.cross(ant, sup)
    acs = _frame_y_primary(origin, sup, right)
    a_len, b_len = np.sqrt(2 * w[-1]), np.sqrt(2 * w[0])    # semi-axes: points on an ellipse have var = a²/2
    res = AutoResult(acs, points=sec)
    res.primitives = [{"kind": "ellipse", "center": origin, "major": ml, "minor": np.cross(sup, ml),
                       "a": float(a_len), "b": float(b_len)},
                      {"kind": "line", "points": L.shaft_centroids}]
    res.quality = {"Section area": f"{area:.0f} mm²", "Section depth": f"{top - h:.1f} mm below the top",
                   "Ellipse": f"{2 * a_len:.0f} × {2 * b_len:.0f} mm",
                   "Overhang ratio": f"{sk:.2f} (posterior : anterior)",
                   "Long axis": "shaft centroids" if L.from_shaft else "inertia"}
    _tibia_notes(res, L, sk, options)
    return res


def _plateau_points(P, N, sup, s, top, depth=15.0):
    m = (s > top - depth) & (N @ sup > 0.7)
    if m.sum() < 200:
        raise JointModelError("Couldn't find the upward-facing plateau surfaces.")
    return P[m]


def tibia_centroids(mesh, side, options) -> AutoResult:
    """Centroids of the medial and lateral plateau articular surfaces."""
    P, N, L, sup, s, top = _tibia_common(mesh, options)
    T = _plateau_points(P, N, sup, s, top)
    q = np.array([_project_perp(p - L.point, sup) for p in T])
    w, V = np.linalg.eigh(np.cov(q.T))
    ml0 = _unit(_project_perp(V[:, -1], sup))
    x = T @ ml0
    # two clusters along the medial-lateral direction (1D 2-means)
    c1, c2 = np.percentile(x, 25), np.percentile(x, 75)
    for _ in range(20):
        lab = np.abs(x - c1) > np.abs(x - c2)
        c1, c2 = x[~lab].mean(), x[lab].mean()
    A, B = T[~lab].mean(0), T[lab].mean(0)
    ml = _unit(_project_perp(B - A, sup))
    origin = (A + B) / 2
    ant, sk = _tibia_anterior(mesh, P, L, sup, s, top, ml, options)
    right = np.cross(ant, sup)
    acs = _frame_y_primary(origin, sup, right)
    res = AutoResult(acs, points=T)
    res.primitives = [{"kind": "sphere", "center": A, "radius": 2.0}, {"kind": "sphere", "center": B, "radius": 2.0},
                      {"kind": "line", "points": L.shaft_centroids}]
    res.quality = {"Separation": f"{np.linalg.norm(B - A):.1f} mm", "Plateau points": int(len(T)),
                   "Overhang ratio": f"{sk:.2f} (posterior : anterior)",
                   "Long axis": "shaft centroids" if L.from_shaft else "inertia"}
    _tibia_notes(res, L, sk, options)
    return res


def tibia_plateau_inertia(mesh, side, options) -> AutoResult:
    """Principal axes of the proximal block of bone under the plateau."""
    P, N, L, sup, s, top = _tibia_common(mesh, options)
    depth = 15.0
    # keep the part above the plane (clip_closed_surface keeps the side the normal points to)
    block = mesh.triangulate().clip_closed_surface(normal=sup, origin=L.point + sup * (top - depth))
    if block.n_points == 0 or not block.volume:
        raise JointModelError("Couldn't cut the top of the tibia into a closed block.")
    A, mom = inertial_frame(block)
    # choose the transverse principal axis with the smallest moment (longest direction)
    cand = [(mom[k], A[:3, k]) for k in range(3) if abs(A[:3, k] @ sup) < 0.6]
    if not cand:
        raise JointModelError("Couldn't find a medial-lateral axis of the plateau.")
    ml = _unit(_project_perp(min(cand, key=lambda c: c[0])[1], sup))
    origin = A[:3, 3]
    ant, sk = _tibia_anterior(mesh, P, L, sup, s, top, ml, options)
    right = np.cross(ant, sup)
    acs = _frame_y_primary(origin, sup, right)
    res = AutoResult(acs, points=np.asarray(block.points)[:: max(1, block.n_points // 4000)])
    res.primitives = [{"kind": "line", "points": L.shaft_centroids}]
    res.quality = {"Block": f"top {depth:.0f} mm, {block.volume / 1000:.1f} cm³",
                   "Overhang ratio": f"{sk:.2f} (posterior : anterior)",
                   "Long axis": "shaft centroids" if L.from_shaft else "inertia"}
    _tibia_notes(res, L, sk, options)
    return res


def _tibia_notes(res, L, ratio, options=None):
    if not L.from_shaft:
        res.notes.append("Too little shaft to fit a shaft line; the long axis is the inertial axis.")
    if ratio < 1.2 and not (options or {}).get("flip_ap"):
        res.notes.append(f"The plateau doesn't clearly overhang the shaft posteriorly (ratio {ratio:.2f}), so "
                         "anterior may be wrong. Check the red X axis points forwards; use Flip "
                         "anterior/posterior if not.")


# ---------------------------------------------------------------- registry

MIRANDA = "Miranda DL et al. (2010) J Biomech 43:1623–1626"
KAI = "Kai S et al. (2014) J Biomech 47:1229–1233"
RENAULT = "Renault JB et al. (2018) Articular-surface-based automatic anatomical coordinate systems for the knee bones. J Biomech (GIBOC)"
MODENESE = "Modenese L, Renault JB (2021) J Biomech 116:110186 (STAPLE)"

AUTO_METHODS: dict[str, AutoMethod] = {m.key: m for m in [
    AutoMethod("femur_cylinder", "Cylinder fitted to the posterior condyles", "femur",
               "Flexion axis (Z) = axis of a cylinder fitted to the posterior and distal condyle surfaces; origin at "
               "its middle. Y from the shaft line (centroids of shaft cross-sections), made perpendicular to Z. "
               "After Miranda et al. (2010) and GIBOC-Cylinder; the most repeatable femur method in Renault et "
               "al. (2018).", (MIRANDA, RENAULT, MODENESE), femur_cylinder),
    AutoMethod("femur_spheres", "Spheres fitted to each posterior condyle", "femur",
               "A sphere is fitted to each posterior condyle; Z joins their centres, the origin is midway. Y from the "
               "shaft line, made perpendicular to Z. After Kai et al. (2014) and GIBOC-Spheres (closest to manual "
               "axes in Modenese & Renault 2021).", (KAI, RENAULT, MODENESE), femur_spheres),
    AutoMethod("tibia_ellipse", "Ellipse of the largest plateau cross-section", "tibia",
               "The largest cross-section near the top of the tibia is fitted with an ellipse: Z along its major "
               "axis, origin at its centroid. Y = the shaft line. After Kai et al. (2014) and GIBOC-Ellipse.",
               (KAI, RENAULT, MODENESE), tibia_ellipse),
    AutoMethod("tibia_centroids", "Centroids of the two plateau surfaces", "tibia",
               "The upward-facing medial and lateral plateau surfaces are found; Z joins their centroids and the "
               "origin is midway. Y = the shaft line. After GIBOC-Centroids.", (RENAULT, MODENESE),
               tibia_centroids),
    AutoMethod("tibia_inertia", "Principal axes of the proximal tibia", "tibia",
               "The top 15 mm of the tibia is treated as a solid: Z along its widest principal axis, origin at its "
               "centroid. Y = the shaft line. After Miranda et al. (2010).", (MIRANDA, MODENESE),
               tibia_plateau_inertia),
]}


def run(method_key: str, mesh, side: str, options: dict | None = None) -> AutoResult:
    m = AUTO_METHODS[method_key]
    return m.fn(mesh, side, options or {})
