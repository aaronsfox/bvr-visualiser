# Ported from msk-STAPLE (https://github.com/modenaxe/msk-STAPLE), GIBOC-core/SubFunctions
# (FittingFun, GeometricFun). Original MATLAB code: Copyright (c) 2020-2021 Modenese L., Renault J.-B.
# Licensed under CC BY-NC 4.0 (see LICENSE_STAPLE.txt). Changes from the original are listed in NOTICE.md.
"""Fitting and geometry helpers used by the STAPLE / GIBOC algorithms."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.spatial import ConvexHull, cKDTree

from .tri import (StapleError, Tri, TriFillPlanarHoles, TriInertiaPpties, TriMesh2DProperties, TriReduceMesh,
                  _inpolygon, normalizeV)


def quantile(x, p):
    """MATLAB's quantile (piecewise linear with p(i) = (i - 0.5)/n)."""
    return float(np.quantile(np.asarray(x, float), p, method="hazen"))


def sphereFit(X):
    """Least-squares sphere (algebraic, as GIBOC-core's sphereFit). Returns Center, Radius, ErrorDist."""
    X = np.asarray(X, float)
    m = X.mean(0)
    A = np.array([
        [np.mean(X[:, 0] * (X[:, 0] - m[0])), 2 * np.mean(X[:, 0] * (X[:, 1] - m[1])), 2 * np.mean(X[:, 0] * (X[:, 2] - m[2]))],
        [0, np.mean(X[:, 1] * (X[:, 1] - m[1])), 2 * np.mean(X[:, 1] * (X[:, 2] - m[2]))],
        [0, 0, np.mean(X[:, 2] * (X[:, 2] - m[2]))]])
    A = A + A.T
    s = (X ** 2).sum(1)
    B = np.array([np.mean(s * (X[:, k] - m[k])) for k in range(3)])
    C = np.linalg.solve(A, B)
    r2 = ((X - C) ** 2).sum(1)
    R = np.sqrt(r2.mean())
    return C, R, r2 - R ** 2


def ellipsoid_fit(X):
    """General ellipsoid fit (Y. Petrov's ellipsoid_fit, unconstrained case). Returns center, radii, evecs, v, chi2."""
    X = np.asarray(X, float)
    if len(X) < 9:
        raise StapleError("Must have at least 9 points to fit a unique ellipsoid.")
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    D = np.c_[x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y, 2 * x * y, 2 * x * z, 2 * y * z,
              2 * x, 2 * y, 2 * z, np.ones_like(x)]
    d2 = x * x + y * y + z * z
    u = np.linalg.solve(D.T @ D, D.T @ d2)
    v = np.zeros(10)
    v[0] = u[0] + u[1] - 1
    v[1] = u[0] - 2 * u[1] - 1
    v[2] = u[1] - 2 * u[0] - 1
    v[3:10] = u[2:9]
    A = np.array([[v[0], v[3], v[4], v[6]], [v[3], v[1], v[5], v[7]], [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])
    center = -np.linalg.solve(A[:3, :3], v[6:9])
    T = np.eye(4)
    T[3, :3] = center
    R = T @ A @ T.T
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evals, evecs = np.real(evals), np.real(evecs)
    radii = np.sqrt(1 / np.abs(evals)) * np.sign(evals)
    dd = ((X - center) @ evecs) / radii
    chi2 = np.sum(np.abs(1 - (dd ** 2 * np.sign(evals)).sum(1)))
    return center, radii, evecs, v, chi2


def fit_ellipse(x, y):
    """Least-squares conic fit of a 2D ellipse (O. Gal's fit_ellipse). Returns a dict like ellipse_t."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    tol = 1e-3
    mean_x, mean_y = x.mean(), y.mean()
    x = x - mean_x
    y = y - mean_y
    X = np.c_[x ** 2, x * y, y ** 2, x, y]
    a, b, c, d, e = np.linalg.lstsq(X.T @ X, X.sum(0), rcond=None)[0]
    if min(abs(b / a), abs(b / c)) > tol:
        phi = 0.5 * np.arctan(b / (c - a))
        cp, sp = np.cos(phi), np.sin(phi)
        a, b, c, d, e = (a * cp ** 2 - b * cp * sp + c * sp ** 2, 0.0, a * sp ** 2 + b * cp * sp + c * cp ** 2,
                         d * cp - e * sp, d * sp + e * cp)
        mean_x, mean_y = cp * mean_x - sp * mean_y, sp * mean_x + cp * mean_y
    else:
        phi = 0.0
        cp, sp = 1.0, 0.0
    if a * c <= 0:
        raise StapleError("fit_ellipse: did not locate an ellipse.")
    if a < 0:
        a, c, d, e = -a, -c, -d, -e
    X0 = mean_x - d / 2 / a
    Y0 = mean_y - e / 2 / c
    F = 1 + d ** 2 / (4 * a) + e ** 2 / (4 * c)
    A, B = np.sqrt(F / a), np.sqrt(F / c)
    R = np.array([[cp, sp], [-sp, cp]])
    X0_in, Y0_in = R @ [X0, Y0]
    th = np.linspace(0, 2 * np.pi, 1000)
    data = R @ np.vstack([X0 + A * np.cos(th), Y0 + B * np.sin(th)])
    return {"a": A, "b": B, "phi": phi, "X0": X0, "Y0": Y0, "X0_in": X0_in, "Y0_in": Y0_in,
            "long_axis": 2 * max(A, B), "short_axis": 2 * min(A, B), "data": data}


def lsplane(X, a0=None):
    """Least-squares plane: (centroid, unit normal), the normal signed to agree with a0 if given."""
    X = np.asarray(X, float)
    if len(X) < 3:
        raise StapleError("At least 3 data points required for a plane.")
    x0 = X.mean(0)
    _, S, Vt = np.linalg.svd(X - x0, full_matrices=False)
    a = Vt[np.argmin(S)]
    if a0 is not None:
        a = np.sign(np.asarray(a0, float).ravel() @ a) * a
    return x0, a


def _rot_to_z(a):
    """A rotation U with U @ a = [0, 0, 1]."""
    a = normalizeV(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = normalizeV(np.cross(t, a))
    v = np.cross(a, u)
    return np.vstack([u, v, a])


def lscylinder(X, x0, a0, r0, tolp=1e-3, tolg=1e-3):
    """Geometric least-squares cylinder (as NPL LSGE's lscylinder): returns point on axis (nearest the
    data centroid), unit axis, radius, residuals.

    The parameters are the same as in LSGE (an offset and two small rotations of the axis in a frame
    where the starting axis is z, plus the radius); the minimisation uses scipy's Levenberg-Marquardt
    instead of LSGE's Gauss-Newton solver, run to tighter tolerances.
    """
    X = np.asarray(X, float)
    if len(X) < 5:
        raise StapleError("At least 5 data points required for a cylinder.")
    xb = X.mean(0)
    R0 = _rot_to_z(a0)
    x1 = R0 @ np.asarray(x0, float).ravel()
    xb1 = R0 @ xb
    t = x1 + (xb1[2] - x1[2]) * np.array([0, 0, 1.0])
    X2 = X @ R0.T - t

    def rot(al, be):
        ca, sa, cb, sb = np.cos(al), np.sin(al), np.cos(be), np.sin(be)
        Rx = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]])
        Ry = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
        return Ry @ Rx

    def res(p):
        R3 = rot(p[2], p[3])
        Q = (X2 - [p[0], p[1], 0]) @ R3.T
        return np.hypot(Q[:, 0], Q[:, 1]) - p[4]

    sol = least_squares(res, np.array([0, 0, 0, 0, float(r0)]), method="lm", xtol=1e-10, ftol=1e-10, gtol=1e-10)
    a = sol.x
    R3 = rot(a[2], a[3])
    an = R0.T @ R3.T @ np.array([0, 0, 1.0])
    p = R3 @ (-np.array([a[0], a[1], 0]) + (xb1 - t))
    x0n = R0.T @ (t + np.array([a[0], a[1], 0]) + R3.T @ np.array([0, 0, p[2]]))
    return x0n, an, abs(a[4]), sol.fun


def fitCSA(Z, Area):
    """Find where the diaphysis and the epiphysis start from the cross-section area profile.

    A two-term Gaussian is fitted to the normalised areas, then one Gaussian plus a straight line.
    Returns Zdiaph (2,), Zepi, or (+1/-1: whether the epiphysis is at the high end).
    """
    Z = np.asarray(Z, float).ravel()
    Area = np.asarray(Area, float).ravel()
    ok = np.isfinite(Z) & np.isfinite(Area)
    Z, Area = Z[ok], Area[ok]
    Zm = Z.mean()
    Z = Z - Zm
    Area = Area / Area.mean()
    imax = int(np.argmax(Area))
    AreaMax = Area[imax]
    orr = -1 if Z[imax] < Z.mean() else 1

    def gauss2(x, a1, b1, c1, a2, b2, c2):
        return a1 * np.exp(-((x - b1) / c1) ** 2) + a2 * np.exp(-((x - b2) / c2) ** 2)

    p0 = [AreaMax, Z[imax], 20, Area.mean(), Z.mean(), 75]
    lo = [-np.inf, -np.inf, 1e-9, -np.inf, -np.inf, 1e-9]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p1, _ = curve_fit(gauss2, Z, Area, p0=p0, bounds=(lo, np.inf), maxfev=20000)
        except (RuntimeError, ValueError):
            p1 = np.array(p0, float)

        def g1(x, a1, b1, c1, d, e):
            return a1 * np.exp(-((x - b1) / c1) ** 2) + d * x + e

        try:
            p2, _ = curve_fit(g1, Z, Area, p0=[p1[0], p1[1], p1[2], 0, Z.mean()], maxfev=20000)
        except (RuntimeError, ValueError):
            p2 = np.array([p1[0], p1[1], p1[2], 0, Z.mean()], float)
    a1, b1, c1, d, e = p2
    dist = Area - (d * Z + e)
    Zkept = Z[np.abs(dist) < abs(0.1 * np.median(Area))]
    Zkept = Zkept[2:-2] if len(Zkept) > 4 else Zkept
    if len(Zkept):
        z_start = Zkept.max() if orr == -1 else Zkept.min()
    else:
        z_start = np.nan
    z_end = b1 - 3 * orr * c1
    Zdiaph = Zm + np.array([z_start, z_end])
    Zepi = Zm + b1 - 1.5 * orr * c1
    return Zdiaph, Zepi, orr


def PlanPolygonCentroid3D(Pts):
    """Centroid and area of a planar polygon in 3D (closed or not)."""
    Pts = np.asarray(Pts, float)
    if len(Pts) == 0:
        return np.full(3, np.nan), np.nan
    if np.all(Pts[0] != Pts[-1]):
        Pts = np.vstack([Pts, Pts[:1]])
    C0 = Pts[:-1].mean(0)
    mid = Pts[:-1] + np.diff(Pts, axis=0) / 2
    tc = mid - (mid - C0) / 3
    w, V = np.linalg.eigh(np.cov(Pts[:-1].T))
    nrm = V[:, 0]
    ta = 0.5 * np.cross(np.diff(Pts, axis=0), -(Pts[:-1] - C0)) @ nrm
    C = (tc * ta[:, None]).sum(0) / ta.sum()
    return C, abs(ta.sum())


def ProjectOnPlan(Pts, n, Op):
    """Project points on the plane with normal n through Op (or n.x + Op = 0 when Op is a number)."""
    n = np.asarray(n, float).ravel()
    Op = np.asarray(Op, float).ravel()
    if len(Op) == 1:
        Op = np.array([0, 0, -Op[0] / n[2]])
    Pts = np.atleast_2d(np.asarray(Pts, float))
    return Pts - np.outer((Pts - Op) @ n, n)


def getLargerPlanarSect(Curves):
    areas = [c["Area"] for c in Curves]
    k = int(np.argmax(areas))
    return Curves[k], len(Curves), areas


def LargestEdgeConvHull(Pts):
    """Edges of the convex hull, longest first: (vertex index pairs, lengths)."""
    Pts = np.asarray(Pts, float)
    K = ConvexHull(Pts).simplices
    e = np.vstack([K[:, [0, 1]], K[:, [1, 2]], K[:, [2, 0]]])
    # MATLAB lists all first edges, then all second, then all third (merged), sorted descending (stable)
    L = np.linalg.norm(Pts[e[:, 0]] - Pts[e[:, 1]], axis=1)
    order = np.argsort(-L, kind="stable")
    return e[order], L[order]


def PCRegionGrowing(Pts, S, r):
    """Points reached from the seed S by hops of at most r."""
    Pts = np.asarray(Pts, float)
    tree = cKDTree(Pts)
    _, i0 = tree.query(np.atleast_2d(S))
    reached = np.zeros(len(Pts), bool)
    front = np.unique(i0)
    reached[front] = True
    while len(front):
        nb = tree.query_ball_point(Pts[front], r)
        new = np.unique(np.concatenate([np.asarray(n, np.int64) for n in nb])) if len(nb) else np.zeros(0, np.int64)
        new = new[~reached[new]]
        reached[new] = True
        front = new
    return Pts[reached], reached


def cutLongBoneMesh(TrLB: Tri, U_0, L_ratio=0.33):
    """Proximal and distal ends (L_ratio of the length each) of a long bone, holes filled."""
    V_all = TriInertiaPpties(TrLB)[0]
    Z0 = V_all[:, 0]
    Z0 = np.sign(np.asarray(U_0, float) @ Z0) * Z0
    z = TrLB.Points @ Z0
    L = z.max() - z.min()
    ic = TrLB.incenter @ Z0
    prox = TriFillPlanarHoles(TriReduceMesh(TrLB, np.nonzero(ic > z.max() - L_ratio * L)[0]))
    dist = TriFillPlanarHoles(TriReduceMesh(TrLB, np.nonzero(ic < z.min() + L_ratio * L)[0]))
    return prox, dist


def computeTriCoeffMorpho(TriObj: Tri):
    """0.5 / mean edge length: scales the mesh-morphology operations to the mesh's resolution."""
    tot = TriMesh2DProperties(TriObj)["TotalArea"]
    mean_edge = np.sqrt(4 / np.sqrt(3) * tot / TriObj.n_faces)
    return 0.5 / mean_edge


def _convhull2(P2):
    """Counter-clockwise hull vertex indices, closed (first repeated), as MATLAB's convhull."""
    h = ConvexHull(P2)
    v = list(h.vertices)
    return np.array(v + v[:1])


def PtsOnCondylesFemur(PtsCondyle_0, Pts_Epiphysis, CutAngle, InSetRatio, ellip_dilat_fact):
    """Points of the epiphysis on one condyle, from an ellipse fitted to the condyle's trace (in the
    sagittal plane, coordinates 3 and 1 of the condyle frame)."""
    P0 = np.asarray(PtsCondyle_0, float)
    Pe = np.asarray(Pts_Epiphysis, float)
    E = fit_ellipse(P0[:, 2], P0[:, 0])
    phi = E["phi"]
    Ux = np.array([np.cos(phi), -np.sin(phi)])
    Uy = np.array([np.sin(phi), np.cos(phi)])
    R = np.c_[Ux, Uy]
    th = np.linspace(0, 2 * np.pi, 36)
    ex = E["X0"] + InSetRatio * E["a"] * np.cos(th)
    ey = E["Y0"] + InSetRatio * E["b"] * np.sin(th)
    rot = (R @ np.vstack([ex, ey])).T
    q = Pe[:, [2, 0]]
    OUT = ~_inpolygon(q[:, 0], q[:, 1], rot[:, 0], rot[:, 1])
    K = _convhull2(P0[:, [2, 0]])
    c = np.array([E["X0_in"], E["Y0_in"]])
    hx = P0[K, 2] + ellip_dilat_fact * (P0[K, 2] - c[0])
    hy = P0[K, 0] + ellip_dilat_fact * (P0[K, 0] - c[1])
    IN_CH = _inpolygon(q[:, 0], q[:, 1], hx, hy)
    rel = q - c
    sq = (rel @ Ux) ** 2 + (rel @ Uy) ** 2
    I = int(np.argmax(sq))
    U = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    lim = np.cos(np.deg2rad(90 - CutAngle))
    if rel[I] @ Uy < 0:
        ext = (U @ Uy < -lim) | ((U @ Uy < 0) & (U @ Ux > 0))
    else:
        ext = (U @ Uy > lim) | ((U @ Uy > 0) & (U @ Ux > 0))
    keep = OUT & IN_CH & ~ext
    return Pe[keep], np.nonzero(keep)[0]


def fitEllipseOnTibialCondylesEdge(Tr: Tri, n, x0):
    """Ellipse fitted to the outline of the tibial plateau seen along n. Returns Xel, Yel, ellipsePts, ellipse."""
    n = np.asarray(n, float).ravel()
    x0 = np.asarray(x0, float).ravel()
    d = -(n @ x0) if len(x0) > 1 else float(x0[0])
    E = Tr.Points[Tr.free_boundary_vertices()]
    top = E[E @ n + d > -10]
    proj = ProjectOnPlan(top, n, d)
    w, V = np.linalg.eigh(np.cov(proj.T))
    p2 = proj @ V
    K = _convhull2(p2[:, 1:3])
    el = fit_ellipse(p2[K, 1], p2[K, 2])
    pts2 = np.vstack([np.full(el["data"].shape[1], p2[0, 0]), el["data"]])
    Yel = V @ np.array([0, np.sin(el["phi"]), np.cos(el["phi"])])
    Xel = V @ np.array([0, np.cos(el["phi"]), -np.sin(el["phi"])])
    return Xel, Yel, (V @ pts2).T, el


def ankleSurfFit(TR_AnkleSurf: Tri, TR_Tibia: Tri, V):
    """Faces of the distal tibia lying within ~1 mm of a quadric fitted to the ankle surface."""
    P0 = TR_AnkleSurf.Points @ V
    PT = TR_Tibia.incenter @ V
    m = P0.mean(0)
    PT = PT - m
    P0 = P0 - m
    x, y, z = P0[:, 1], P0[:, 2], P0[:, 0]
    A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    xt, yt = PT[:, 1], PT[:, 2]
    zt = np.c_[np.ones_like(xt), xt, yt, xt * xt, xt * yt, yt * yt] @ c
    return np.nonzero(np.abs(zt - PT[:, 0]) < 1.05)[0]


def kmeans2(P, seed=0):
    """Two-cluster k-means labels (0/1)."""
    from scipy.cluster.vq import kmeans2 as _km
    _, lab = _km(np.asarray(P, float), 2, minit="++", seed=seed)
    return lab
