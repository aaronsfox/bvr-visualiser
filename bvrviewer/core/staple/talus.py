# Ported from msk-STAPLE (https://github.com/modenaxe/msk-STAPLE): algorithms/STAPLE_talus.m,
# CS_talus_trochleaCylinder.m, CS_talus_subtalarSpheres.m and GIBOC-core fitCSATalus.m, fitQuadriTalus.m.
# Original MATLAB code: Copyright (c) 2020-2021 Modenese L., Renault J.-B. Licensed under CC BY-NC 4.0
# (see LICENSE_STAPLE.txt). Changes from the original are listed in NOTICE.md.
"""STAPLE-Talus: ankle (talocrural) and subtalar joint axes from the talus surface.

Ankle frame: Z = axis of a cylinder fitted to the talar trochlea (pointing right on both sides),
origin on that axis. Subtalar frame: its third axis joins the centres of spheres fitted to the
talocalcaneal (posterior facet) and talonavicular surfaces; origin at the talocalcaneal centre.
The talus is a small bone, usually imaged whole, so no partial-bone adaptation is needed.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

from .common import BoneResult, bodySide2Sign
from .fitting import computeTriCoeffMorpho, lscylinder, quantile, sphereFit
from .tri import (Tri, TriCloseMesh, TriCurvature, TriDilateMesh, TriErodeMesh, TriInertiaPpties,
                  TriKeepLargestPatch, TriReduceMesh, TriSliceObjAlongAxis, ensure_outward, normalizeV)


def fitQuadriTalus(Tr: Tri, V_all):
    """Initial superior (Z0) and lateral-ish (Y0) directions from the quadrilateral outline of the talus
    seen along its long axis: the edge opposite the longest one is the superior (trochlear) side."""
    P2 = (Tr.Points @ V_all)[:, 1:3]
    hv = ConvexHull(P2).vertices
    K = np.r_[hv, hv[:1]]                                     # closed, as MATLAB's convhull
    Q = P2[K]

    def far_pair(X):
        """(row, col) of the largest pairwise distance, the first one in MATLAB's column-major order."""
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        k = int(np.argmax(D.T.ravel()))
        return k % len(X), k // len(X)

    r1, c1 = far_pair(Q)
    U1 = normalizeV(P2[K[c1]] - P2[K[r1]])
    Ru = np.array([[U1[0], -U1[1]], [U1[1], U1[0]]])
    Qs = (Ru @ np.diag([0.5, 1]) @ Ru.T @ Q.T).T
    r2, c2 = far_pair(Qs)
    idx = sorted([r1, c1, r2, c2])
    quad = K[idx + [idx[0]]]
    edges = [np.linalg.norm(P2[quad[i + 1]] - P2[quad[i]]) for i in range(4)]
    imax = int(np.argmax(edges)) + 1                          # 1-based, as in MATLAB
    i_sup = (imax + 2) % 4
    if i_sup == 0:                                            # MATLAB would index 0 here; the opposite edge is 4
        i_sup = 4
    e = quad[i_sup - 1:i_sup + 1]
    U = P2[e[1]] - P2[e[0]]
    Z0p = np.array([-U[1], U[0]])
    U0 = P2[e].mean(0) - P2.mean(0)
    Z0p = normalizeV(np.sign(U0 @ Z0p) * Z0p)
    Z0 = V_all @ np.r_[0, Z0p]
    Y0 = np.cross(Z0, V_all[:, 0])
    return Z0, Y0


def fitCSATalus(Alt, Area):
    """Where the talonavicular surface, the neck and the trochlea start along the long axis, from a
    two-Gaussian fit to the cross-section areas. Returns (or, alt_TlNvc_start, alt_TlNeck_start,
    alt_TlTib_start)."""
    Alt = np.asarray(Alt, float)
    Area = np.asarray(Area, float)
    Length = np.ptp(Alt)
    keep = Area >= 0.2 * Area.max()
    Alt, Area = Alt[keep], Area[keep]
    M = Area.max()

    def gauss2(x, a1, b1, c1, a2, b2, c2):
        return a1 * np.exp(-((x - b1) / c1) ** 2) + a2 * np.exp(-((x - b2) / c2) ** 2)

    lo = [0.1 * M, -np.inf, 1e-9, 0.1 * M, -np.inf, 1e-9]
    hi = [10 * M, np.inf, 0.2 * Length, 10 * M, np.inf, 0.2 * Length]
    # MATLAB's fit picks its own start point for gauss2 from the data's peaks; start from the two halves
    q = np.quantile(Alt, [0.25, 0.75])
    p0 = [np.clip(M, lo[0], hi[0]), q[0], 0.1 * Length, np.clip(M, lo[3], hi[3]), q[1], 0.1 * Length]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = curve_fit(gauss2, Alt, Area, p0=p0, bounds=(lo, hi), maxfev=20000)
    a1, b1, c1, a2, b2, c2 = p
    test_a, test_b = a1 < a2, b1 < b2
    if test_a:
        nvc = b1
        orr = -2 * int(test_b) + 1
        tib = orr * (b2 - orr * 1.0 * c2)
    else:
        nvc = b2
        orr = 2 * int(test_b) - 1
        tib = orr * (b1 - orr * 1.0 * c1)
    return orr, orr * nvc, 0.5 * orr * (b1 + b2), tib


def CS_talus_subtalarSpheres(Talus: Tri, side, CS, alt_TlNvc_start, alt_TlNeck_start, CoeffMorpho, curv):
    sgn, _ = bodySide2Sign(side)
    X0, Y0, Z0, C = CS["X0"], CS["Y0"], CS["Z0"], CS["CenterVol"]
    k2 = curv[5]
    alt_end = (Talus.incenter @ X0).max()
    L = alt_end - alt_TlNvc_start
    nav = TriReduceMesh(Talus, np.nonzero(Talus.incenter @ X0 > alt_TlNvc_start + 0.25 * L)[0])
    V = np.c_[X0, Y0, Z0]
    R = (nav.Points - C) @ V
    keep = np.nonzero(R[:, 2] < 0)[0]
    nav_pts = R[keep] @ V.T + C
    c_nav, r_nav, e_nav = sphereFit(nav_pts)
    nav = TriReduceMesh(nav, NodesKept=keep)
    U1 = normalizeV(-Z0 + 0.5 * Y0 - 0.4 * X0)
    U2 = normalizeV(-Z0 - 0.5 * Y0 - 0.3 * X0)
    vn = Talus.vertexNormal
    acosd = lambda x: np.degrees(np.arccos(np.clip(x, -1, 1)))  # noqa: E731
    ok0 = np.nonzero((k2 > quantile(k2, 0.33)) & ((acosd(vn @ U1) < 30) | (acosd(vn @ U2) < 30))
                     & (Talus.Points @ X0 < C @ X0) & (vn @ Z0 < 0))[0]
    s0 = TriKeepLargestPatch(TriReduceMesh(Talus, NodesKept=ok0))
    c0 = s0.Points.mean(0)
    u0 = normalizeV(c0 - C)
    U = U1 if U1 @ u0 >= U2 @ u0 else U2
    cen = c0 + 1.0 * U
    CP = Talus.Points - cen
    nC = -CP / np.linalg.norm(CP, axis=1, keepdims=True)
    ok1 = np.nonzero((k2 > quantile(k2, 0.33)) & (acosd(vn @ U) < 60) & ((vn * nC).sum(1) > 0)
                     & (vn @ Z0 < 0) & (Talus.Points @ X0 < alt_TlNeck_start))[0]
    s1 = TriKeepLargestPatch(TriReduceMesh(Talus, NodesKept=ok1))
    s1 = TriCloseMesh(Talus, s1, 2 * CoeffMorpho)
    c_cc, r_cc, e_cc = sphereFit(s1.incenter)
    axis = normalizeV(c_nav - c_cc)
    X1 = axis
    Z1 = normalizeV(np.cross(X1, Y0)) * sgn
    Y1 = normalizeV(np.cross(Z1, X1))
    CS.update(X1=X1, Y1=Y1, Z1=Z1, V_subtalar=np.c_[Y1, Z1, X1], subtalar_axis=axis,
              subtalar_axis_centre=(c_nav + c_cc) / 2, talocalc_centre=c_cc, talocalc_radius=r_cc,
              talonav_centre=c_nav, talonav_radius=r_nav,
              talocalc_rmse=float(np.mean(np.abs(e_cc))), talonav_rmse=float(np.mean(np.abs(e_nav))))
    return CS, s1, nav


def CS_talus_trochleaCylinder(Talus: Tri, side, CS, alt_TlNeck_start, alt_TlTib_start, CoeffMorpho, curv):
    sgn, _ = bodySide2Sign(side)
    X0, Y0, Z0, Y1 = CS["X0"], CS["Y0"], CS["Z0"], CS["Y1"]
    k1, k2 = curv[4], curv[5]
    mac = np.maximum(np.abs(k1), np.abs(k2))
    vn = Talus.vertexNormal
    acosd = lambda x: np.degrees(np.arccos(np.clip(x, -1, 1)))  # noqa: E731
    ok = np.nonzero((mac < quantile(mac, 0.5)) & (acosd(vn @ Z0) < 60) & (acosd(vn @ Y0) > 60)
                    & (Talus.Points @ X0 < alt_TlNeck_start))[0]
    s = TriReduceMesh(Talus, NodesKept=ok)
    s = TriCloseMesh(Talus, s, 1 * CoeffMorpho)
    s = TriKeepLargestPatch(s)
    s = TriErodeMesh(s, 2 * CoeffMorpho)
    s = TriKeepLargestPatch(s)
    s = TriCloseMesh(Talus, s, 3 * CoeffMorpho)
    s = TriDilateMesh(Talus, s, 2 * CoeffMorpho)
    c0, r0, _ = sphereFit(s.incenter)
    x0n, an, rn, d = lscylinder(s.incenter, c0, Y0, r0)
    axis = normalizeV(an)
    ok = np.nonzero((d > -0.05 * rn) & (np.abs(d) < 0.1 * rn) & (s.incenter @ X0 > alt_TlTib_start))[0]
    s1 = TriKeepLargestPatch(TriReduceMesh(s, ok))
    x0n, an, rn, d = lscylinder(s1.incenter, x0n, axis, rn)
    axis = normalizeV(an)
    Z2 = normalizeV(np.sign(-Y1 @ axis) * axis) * sgn
    Y2 = normalizeV(np.cross(Z2, X0))
    X2 = normalizeV(np.cross(Y2, Z2))
    CS.update(Z2=Z2, Y2=Y2, X2=X2, V_ankle=np.c_[X2, Y2, Z2], ankle_cyl_radius=rn, ankle_cyl_centre=x0n,
              ankle_cyl_axis=axis, ankle_cyl_rmse=float(np.sqrt(np.mean(d ** 2))),
              ankle_cyl_range=float(np.ptp((s1.incenter - x0n) @ axis)))
    return CS, s1


def STAPLE_talus(talusTri: Tri, side="r", **_ignored) -> BoneResult:
    """STAPLE-Talus (Modenese & Renault 2021). Returns the ankle (child) and subtalar (parent) frames."""
    talusTri = ensure_outward(talusTri)
    sgn, low = bodySide2Sign(side)
    CoeffMorpho = computeTriCoeffMorpho(talusTri)
    V_all, CenterVol, Inertia, D = TriInertiaPpties(talusTri)[:4]
    aux = {"CenterVol": CenterVol, "D": D, "X0": V_all[:, 0]}
    aux["Z0"], aux["Y0"] = fitQuadriTalus(talusTri, V_all)
    Areas, Alt = TriSliceObjAlongAxis(talusTri, aux["X0"], 0.3, 0.3)[:2]
    orr, nvc, neck, tib = fitCSATalus(Alt, Areas)
    aux["X0"] = orr * aux["X0"]
    aux["Y0"] = orr * aux["Y0"]
    curv = TriCurvature(talusTri)
    aux, talocalc, talonav = CS_talus_subtalarSpheres(talusTri, side, aux, nvc, neck, CoeffMorpho, curv)
    aux, troch = CS_talus_trochleaCylinder(talusTri, side, aux, neck, tib, CoeffMorpho, curv)
    JCS = {"ankle_" + low: {"V": aux["V_ankle"], "Origin": aux["ankle_cyl_centre"]},
           "subtalar_" + low: {"V": aux["V_subtalar"], "Origin": aux["talocalc_centre"]}}
    art = {"talar_trochlea": troch, "talo_calcn": talocalc, "talo_navic": talonav}
    return BoneResult(bone="talus", method="STAPLE-Talus", side=low, JCS=JCS, aux=aux, art_surf=art, notes=[])
