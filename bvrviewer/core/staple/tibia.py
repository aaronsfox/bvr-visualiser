# Ported from msk-STAPLE (https://github.com/modenaxe/msk-STAPLE): algorithms/GIBOC_tibia.m, Kai2014_tibia.m,
# CS_tibia_*.m and their private functions. Original MATLAB code: Copyright (c) 2020-2021 Modenese L.,
# Renault J.-B. Licensed under CC BY-NC 4.0 (see LICENSE_STAPLE.txt). Changes are listed in NOTICE.md,
# including the partial-bone adaptations marked "PARTIAL" below, which are not part of STAPLE.
"""Tibia coordinate systems: GIBOC-Tibia (plateau centroids, ellipse, plateau layer) and Kai et al. (2014).

Axis names follow STAPLE / ISB: X anterior, Y superior (exact: the mechanical axis), Z to the
subject's right.
"""

from __future__ import annotations

import numpy as np

from .common import BoneResult, bodySide2Sign, partial_long_axis, shaft_line
from .femur import GIBOC_isolate_epiphysis, _pca
from .fitting import (PlanPolygonCentroid3D, ProjectOnPlan, ankleSurfFit, computeTriCoeffMorpho, cutLongBoneMesh,
                      fit_ellipse, fitEllipseOnTibialCondylesEdge, getLargerPlanarSect, kmeans2, lsplane, quantile)
from .tri import (StapleError, Tri, ensure_outward, TriCloseMesh, TriConnectedPatch, TriCurvature, TriDifferenceMesh,
                  TriDilateMesh, TriErodeMesh, TriInertiaPpties, TriMesh2DProperties, TriOpenMesh, TriPlanIntersect,
                  TriReduceMesh, TriSliceObjAlongAxis, TriSplitObjects, TriUnite, _inpolygon, normalizeV)


# ---------------------------------------------------------------- orientation and side

def tibia_guess_CS(Tibia: Tri):
    """Distal-to-proximal direction: the largest cross-section is near the plateau (top third)."""
    V_all = TriInertiaPpties(Tibia)[0]
    Z0 = V_all[:, 0]
    z = Tibia.Points @ Z0
    alt = np.linspace(z.min() + 0.5, z.max() - 0.5, 100)
    area = []
    for a in alt:
        curves, _ = TriPlanIntersect(Tibia, Z0, -a)
        area.append(max([PlanPolygonCentroid3D(c["Pts"])[1] for c in curves], default=0.0))
    i = int(np.argmax(area))
    notes = []
    if i > 0.66 * len(alt):
        pass
    elif i < 0.33 * len(alt):
        Z0 = -Z0
    else:
        notes.append("Identification of the tibia's distal-to-proximal axis went wrong; check the geometry.")
    return Z0, notes


def tibia_identify_lateral_direction(DistTib: Tri, Z0):
    """Lateral direction from the fibula (if part of the mesh) or the medial malleolus."""
    C = TriInertiaPpties(DistTib)[1]
    curves, _ = TriPlanIntersect(DistTib, Z0, -(C @ Z0))
    if len(curves) > 2:
        raise StapleError(f"There are {len(curves)} cross-sections of the distal tibia (expected 1 or 2).")
    just_tibia = len(curves) != 2
    i = int(np.argmax(DistTib.Points @ -Z0))
    MostDistal = DistTib.Points[i]
    if just_tibia:
        U = C - MostDistal
    else:
        if curves[0]["Area"] > curves[1]["Area"]:
            U = curves[1]["Pts"].mean(0) - curves[0]["Pts"].mean(0)
        else:
            U = -(curves[1]["Pts"].mean(0) - curves[0]["Pts"].mean(0))
    return U, MostDistal, just_tibia


def removeFibulaFromProxTibia(ProxTib: Tri):
    parts = TriSplitObjects(ProxTib)
    if len(parts) == 2:
        return parts[0][0], []
    if len(parts) > 2:
        return ProxTib, ["The proximal tibia has more than two separate parts; kept the geometry as it is."]
    return ProxTib, []


# ---------------------------------------------------------------- Kai et al. 2014

def Kai2014_tibia(tibiaTri: Tri, side="r", whole=None, flip_ap=False, flip_long=False) -> BoneResult:
    """Kai et al. (2014): ellipse fitted to the largest cross-section of the tibia, as in STAPLE."""
    tibiaTri = ensure_outward(tibiaTri)
    sgn, low = bodySide2Sign(side)
    notes = []
    V_all = _pca(tibiaTri.Points)
    Vi, CenterVol = TriInertiaPpties(tibiaTri)[:2]
    if whole is None:
        whole = np.ptp(tibiaTri.Points @ Vi[:, 0]) > 250
    if whole:
        U, n = tibia_guess_CS(tibiaTri)
        notes += n
        Prox, Dist = cutLongBoneMesh(tibiaTri, U)
        Y0 = V_all[:, 0]
        Y0 = np.sign((Prox.Points.mean(0) - Dist.Points.mean(0)) @ Y0) * Y0
    else:
        # PARTIAL: the long axis is the shaft line of the tibia that is present (the principal axis of a
        # proximal tibia is dominated by the plateau); slices and the ellipse are taken across it.
        Y0 = partial_long_axis(tibiaTri, Vi[:, 0], epiphysis_at="high") * (-1 if flip_long else 1)
        Zepi = GIBOC_isolate_epiphysis(tibiaTri, Y0, "proximal")[1]
        sd, sp, sn = shaft_line(tibiaTri, Y0, Zepi, "proximal")
        notes += sn
        Y0 = sd
        e2 = normalizeV(V_all[:, 1] - (V_all[:, 1] @ Y0) * Y0)
        V_all = np.c_[Y0, e2, np.cross(Y0, e2)]
    AltAtMax = TriSliceObjAlongAxis(tibiaTri, Y0, 1.0)[4]
    curves, _ = TriPlanIntersect(tibiaTri, Y0, -AltAtMax)
    sect, ncur, _ = getLargerPlanarSect(curves)
    if ncur > 2:
        raise StapleError(f"There are {ncur} sections at the largest tibial slice (expected tibia and maybe fibula).")
    P = sect["Pts"] @ V_all
    E = fit_ellipse(P[:, 1], P[:, 2])
    if E["a"] > E["b"]:
        Zel = V_all @ np.array([0, np.cos(E["phi"]), -np.sin(E["phi"])])
    else:
        Zel = V_all @ np.array([0, np.sin(E["phi"]), np.cos(E["phi"])])
    Center = V_all @ np.array([P[:, 0].mean(), E["X0_in"], E["Y0_in"]])
    aux = {"Y0": Y0, "CenterVol": CenterVol, "EllipseCenter": Center, "Ellipse": E, "SectionPts": sect["Pts"],
           "whole": bool(whole)}
    if whole:
        U_tmp, MostDistal, just_tibia = tibia_identify_lateral_direction(Dist, Y0)
        U_tmp = sgn * U_tmp
    else:
        # PARTIAL: no fibula or malleolus to tell lateral from medial
        U_tmp = sgn * _partial_lateral(tibiaTri, Y0, Zel, (sd, sp), side, flip_ap, notes)
    Zt = normalizeV(U_tmp - (U_tmp @ Y0) * Y0)
    Zel = np.sign(Zt @ Zel) * Zel
    Y = normalizeV(Y0)
    Z = normalizeV(Zel)
    X = normalizeV(np.cross(Y, Z))
    JCS = {"knee_" + low: {"V": np.c_[X, normalizeV(np.cross(Z, X)), Z], "Origin": Center}}
    if not whole:
        aux["shaft_axis"], aux["shaft_point"] = sd, sp
        aux["adaptation"] = ("Partial tibia: Y is the shaft axis of the tibia that is present instead of the "
                             "whole bone's long axis.")
    return BoneResult(bone="tibia", method="Kai2014", side=low, JCS=JCS, aux=aux, notes=notes, whole=bool(whole))


# ---------------------------------------------------------------- GIBOC: distal (ankle) part

def GIBOC_tibia_DistMaxSectCentre(DistTib: Tri, Z0):
    maxAlt = TriSliceObjAlongAxis(DistTib, Z0, 0.3, 1.0)[4]
    curves, _ = TriPlanIntersect(DistTib, Z0, -maxAlt)
    c, n, _ = getLargerPlanarSect(curves)
    if n > 2:
        raise StapleError(f"There are {n} section areas at the distal tibia.")
    return PlanPolygonCentroid3D(c["Pts"])[0]


def GIBOC_tibia_DistArtSurf(DistTib: Tri, Z0, V_all, CoeffMorpho):
    Cmean = TriCurvature(DistTib)[0]
    ang = np.degrees(np.arccos(np.clip(-DistTib.vertexNormal @ Z0, -1, 1)))
    ok = np.nonzero((Cmean > quantile(Cmean, 0.65)) & (Cmean < quantile(Cmean, 0.95)) & (ang < 30))[0]
    s0 = TriReduceMesh(DistTib, NodesKept=ok)
    s0 = TriCloseMesh(DistTib, s0, 6 * CoeffMorpho)
    s0 = TriOpenMesh(DistTib, s0, 4 * CoeffMorpho)
    s0 = TriConnectedPatch(s0, s0.Points.mean(0))
    ids = ankleSurfFit(s0, DistTib, V_all)
    s = TriReduceMesh(DistTib, ids)
    s = TriErodeMesh(s, 2)
    s = TriCloseMesh(DistTib, s, 6 * CoeffMorpho)
    s = TriOpenMesh(DistTib, s, 4 * CoeffMorpho)
    s = TriCloseMesh(DistTib, s, 2 * CoeffMorpho)
    pp = TriMesh2DProperties(s)
    Zs, C = pp["meanNormal"], pp["onMeshCenter"]
    term = s.incenter - C
    ok = (np.degrees(np.arccos(np.clip(s.faceNormal @ Zs, -1, 1))) < 35) & \
         ((term * s.faceNormal).sum(1) / np.linalg.norm(term, axis=1) < 0.1)
    s = TriReduceMesh(s, np.nonzero(ok)[0])
    s = TriOpenMesh(DistTib, s, 4 * CoeffMorpho)
    return TriCloseMesh(DistTib, s, 10 * CoeffMorpho)


# ---------------------------------------------------------------- GIBOC: plateau

def GIBOC_tibia_FullProxArtSurf(EpiTib: Tri, CSs, CoeffMorpho, angle_thresh=35, curv_quartile=0.25):
    Z0, Y0 = CSs["Z0"], CSs["Y0"]
    Cmean, Cg = TriCurvature(EpiTib)[:2]
    Curv = np.sqrt(np.maximum(4 * Cmean ** 2 - 2 * Cg, 0))
    ang = np.degrees(np.arccos(np.clip(EpiTib.vertexNormal @ Z0, -1, 1)))
    ok = np.nonzero((ang < angle_thresh) & (Curv < quantile(Curv, curv_quartile)))[0]
    oLSP, Ztp = lsplane(EpiTib.Points[ok], Z0)
    AS = TriReduceMesh(EpiTib, NodesKept=ok)
    AS = TriCloseMesh(EpiTib, AS, 6 * CoeffMorpho)
    Xel, Yel, epts, E = fitEllipseOnTibialCondylesEdge(AS, Ztp, oLSP)
    a, b = E["a"], E["b"]
    Xel = np.sign(Xel @ Y0) * Xel
    Yel = np.sign(Y0 @ Yel) * Yel
    Ec = epts.mean(0)
    rel = AS.Points - Ec
    width = np.ptp(rel @ Yel)
    AS = TriReduceMesh(AS, NodesKept=np.nonzero(np.abs(rel @ Yel) > 0.025 * width)[0])
    m0 = Ec + 2 / 3 * b * Yel
    med = TriConnectedPatch(AS, np.vstack([m0, m0 - a / 3 * Xel, m0 + a / 3 * Xel]))
    l0 = Ec - 2 / 3 * b * Yel
    lat = TriConnectedPatch(AS, np.vstack([l0, l0 - a / 3 * Xel, l0 + a / 3 * Xel]))
    return TriUnite(med, lat), oLSP, Ztp


def GIBOC_tibia_removePartBetweenRidges_it1(ProxTib, EpiTib, CSs, CoeffMorpho, ELP1, Ztp, which):
    Z0 = CSs["Z0"]
    a, b, Xel, Yel, epts = ELP1["a"], ELP1["b"], ELP1["Xel"], ELP1["Yel"], ELP1["ellipsePts"]
    em = epts.mean(0)
    d = -(em @ Xel + 0.5 * a)
    curves, _ = TriPlanIntersect(ProxTib, Xel, d)
    if not curves:
        return None
    C = getLargerPlanarSect(curves)[0]
    term = C["Pts"] - em
    if which == "medial":
        tmp = C["Pts"][term @ Yel > 0]
        if len(tmp) == 0:
            return None
        pmax = tmp[int(np.argmax(tmp @ Z0))]
        coeff = 1
    else:
        tmp = C["Pts"][(term @ Yel < 0) & (term @ Yel > -b / 3) & (np.abs(term @ Z0) < a / 2)]
        if len(tmp) == 0:
            return None
        pmax = tmp[int(np.argmin(tmp @ Z0))]
        coeff = -1
    U = pmax - em
    np_ = normalizeV(coeff * np.sign(np.cross(Xel, Yel) @ Z0) * np.cross(U, Ztp))
    dp = -em @ np_
    nm = coeff * Yel
    dm = -em @ nm
    ids = np.nonzero(np.sign(EpiTib.Points @ np_ + dp) + np.sign(EpiTib.Points @ nm + dm) > 0.1)[0]
    if len(ids) == 0:
        return None
    R = TriReduceMesh(EpiTib, NodesKept=ids)
    if which == "lateral":
        R = TriDilateMesh(EpiTib, R, 5 * CoeffMorpho)
    return R


def GIBOC_tibia_ProxArtSurf_it1(ProxTib, EpiTib, EpiTibAS, CSs, Ztp, oLSP, CoeffMorpho):
    Y0 = CSs["Y0"]
    Xel, Yel, epts, E = fitEllipseOnTibialCondylesEdge(EpiTibAS, Ztp, oLSP)
    ELP1 = {"a": E["a"], "b": E["b"], "Xel": -np.sign(Xel @ Y0) * Xel, "Yel": -np.sign(Yel @ Y0) * Yel,
            "ellipsePts": epts}
    rm = GIBOC_tibia_removePartBetweenRidges_it1(ProxTib, EpiTib, CSs, CoeffMorpho, ELP1, Ztp, "medial")
    rl = GIBOC_tibia_removePartBetweenRidges_it1(ProxTib, EpiTib, CSs, CoeffMorpho, ELP1, Ztp, "lateral")
    parts = [p for p in (rl, rm) if p is not None]
    if parts:
        ridge = parts[0] if len(parts) == 1 else TriUnite(parts[0], parts[1])
        EpiTibAS = TriDifferenceMesh(EpiTibAS, ridge)
    em = epts.mean(0)
    m0 = em + 2 / 3 * ELP1["b"] * ELP1["Yel"]
    l0 = em - 2 / 3 * ELP1["b"] * ELP1["Yel"]
    a, X = ELP1["a"], ELP1["Xel"]
    seeds = np.vstack([m0, m0 - a / 3 * X, m0 + a / 3 * X, l0, l0 - a / 3 * X, l0 + a / 3 * X])
    return TriConnectedPatch(EpiTibAS, seeds)


def GIBOC_tibia_ProxArtSurf_it2(EpiTib, EpiTibAS, CSs, CoeffMorpho):
    Z0, Y0 = CSs["Z0"], CSs["Y0_GIBOC"]
    oLSP, Ztp = lsplane(EpiTibAS.Points, Z0)
    d = -oLSP @ Ztp
    Xel, Yel, epts, E = fitEllipseOnTibialCondylesEdge(EpiTibAS, Ztp, oLSP)
    a, b = E["a"], E["b"]
    Xel = np.sign(Xel @ Y0) * Xel
    Yel = np.sign(Yel @ Y0) * Yel
    em = epts.mean(0)
    s, seeds = [], []
    for coeff in (1, -1):
        p = em + coeff * 2 / 3 * b * Yel
        P3 = np.vstack([p, p - a / 3 * Xel, p + a / 3 * Xel])
        seeds.append(P3)
        c = TriConnectedPatch(EpiTibAS, P3)
        ok = (np.abs(c.incenter @ Ztp + d) < 5) & (c.faceNormal @ Ztp > 0.9)
        c = TriReduceMesh(c, np.nonzero(ok)[0])
        c = TriOpenMesh(EpiTib, c, 2 * CoeffMorpho)
        c = TriConnectedPatch(c, P3)
        c = TriCloseMesh(EpiTib, c, 10 * CoeffMorpho)
        s.append(c)
    it2 = TriUnite(s[0], s[1])
    oLSP, Ztp = lsplane(it2.Points, Z0)
    d = -oLSP @ Ztp
    out = []
    for k, thr in enumerate((5, 3)):
        c = s[k]
        ok = (np.abs(c.incenter @ Ztp + d) < thr) & (c.faceNormal @ Ztp > 0.95)
        c = TriReduceMesh(c, np.nonzero(ok)[0])
        c = TriOpenMesh(EpiTib, c, 2 * CoeffMorpho)
        c = TriConnectedPatch(c, seeds[k])
        c = TriCloseMesh(EpiTib, c, 10 * CoeffMorpho)
        out.append(c)
    return out[0], out[1], it2


# ---------------------------------------------------------------- coordinate systems

def _tibia_jcs(Y, Z, Origin, low):
    X = normalizeV(np.cross(Y, Z))
    return {"knee_" + low: {"V": np.c_[X, Y, normalizeV(np.cross(X, Y))], "Origin": Origin}}


def _mech_y(CS, KneeCenter, ankle_key):
    """Y: from the ankle centre to the knee centre (STAPLE) or, for a partial tibia, along the shaft (PARTIAL)."""
    if CS.get(ankle_key) is not None:
        return normalizeV(KneeCenter - CS[ankle_key])
    return normalizeV(CS["shaft_axis"])


def CS_tibia_ArtSurfCentroids(Med: Tri, Lat: Tri, CS, side):
    sgn, low = bodySide2Sign(side)
    cl = TriMesh2DProperties(Lat)["Center"]
    cm = TriMesh2DProperties(Med)["Center"]
    K = 0.5 * (cm + cl)
    CS.update(Centroid_AS_med=cm, Centroid_AS_lat=cl)
    Y = _mech_y(CS, K, "AnkleCenter")
    Z = normalizeV(cl - cm) * sgn
    return CS, _tibia_jcs(Y, Z, K, low)


def CS_tibia_Ellipse(AS: Tri, CS, side):
    sgn, low = bodySide2Sign(side)
    oLSP, Ztp = lsplane(AS.Points, CS["Z0"])
    _, Yel, epts, E = fitEllipseOnTibialCondylesEdge(AS, Ztp, oLSP)
    K = epts.mean(0)
    CS.update(ElpsMaxPtVect=Yel, ElpsPts=epts, Ellipse=E, PlateauNormal=Ztp)
    Z = normalizeV(np.sign(Yel @ CS["Y0"]) * Yel) * sgn
    Y = _mech_y(CS, K, "AnkleCenter")
    return CS, _tibia_jcs(Y, Z, K, low)


def CS_tibia_PlateauLayer(EpiTib: Tri, AS: Tri, CS, side):
    sgn, low = bodySide2Sign(side)
    oLSP, Ztp = lsplane(AS.Points, CS["Z0"])
    d = -oLSP @ Ztp
    Xel, Yel, epts, E = fitEllipseOnTibialCondylesEdge(AS, Ztp, oLSP)
    Xel = np.sign(Xel @ CS["Y0"]) * Xel
    Yel = np.sign(Yel @ CS["Y0"]) * Yel
    maxArea = TriSliceObjAlongAxis(EpiTib, CS["Z0"], 1.0)[2]
    H = 0.1 * np.sqrt(4 * 0.75 * maxArea / np.pi)
    alts = np.linspace(-d - H, -d + 0.5 * H, 20)
    space = np.mean(np.diff(alts))
    B = np.c_[Xel, Yel, Ztp]
    layer = []
    for alt in alts:
        curves, _ = TriPlanIntersect(EpiTib, Ztp, -alt)
        for c in curves:
            q = c["Pts"] @ B
            xm = np.arange(q[:, 0].min() - 0.1, q[:, 0].max() + 0.1 + 1e-9, space)
            ym = np.arange(q[:, 1].min() - 0.1, q[:, 1].max() + 0.1 + 1e-9, space)
            XX, YY = np.meshgrid(xm, ym)
            inn = _inpolygon(XX.ravel(), YY.ravel(), q[:, 0], q[:, 1])
            if inn.any():
                g = np.c_[XX.ravel()[inn], YY.ravel()[inn], np.full(inn.sum(), alt)]
                layer.append(g @ B.T)
    L = np.vstack(layer)
    w, V = np.linalg.eigh(np.cov(L.T))
    Xtp, Ytp = V[:, 1], V[:, 2]
    Xtp = np.sign(Xtp @ CS["Y0"]) * Xtp
    Ytp = np.sign(Ytp @ CS["Y0"]) * Ytp
    lab = kmeans2(L)
    cm = ProjectOnPlan(L[lab == 0].mean(0), Ztp, d)[0]
    cl = ProjectOnPlan(L[lab == 1].mean(0), Ztp, d)[0]
    K = 0.5 * (cm + cl)
    CS.update(Ztibplat=Ztp, Ytibplat=Ytp, Xtibplat=Xtp, PlateauLayerPts=L)
    Y = _mech_y(CS, K, "CenterAnkleInside")
    Z = normalizeV(Ytp) * sgn
    return CS, _tibia_jcs(Y, Z, K, low)


# ---------------------------------------------------------------- PARTIAL: lateral direction without the ankle

def _section_ap_skewness(tri: Tri, d, p0, ap, level):
    """Skewness, along ap, of the area of the bone's cross-section at ``level`` along the shaft line."""
    curves, _ = TriPlanIntersect(tri, d, -(level + p0 @ d))
    if not curves:
        return None
    c = max(curves, key=lambda c: c["Area"])
    ml = np.cross(d, ap)
    q = np.c_[(c["Pts"] - p0) @ ap, (c["Pts"] - p0) @ ml]
    xs = np.linspace(q[:, 0].min(), q[:, 0].max(), 100)
    ys = np.linspace(q[:, 1].min(), q[:, 1].max(), 100)
    X, Y = np.meshgrid(xs, ys)
    inside = _inpolygon(X.ravel(), Y.ravel(), q[:, 0], q[:, 1])
    if inside.sum() < 50:
        return None
    x = X.ravel()[inside]
    x = x - x.mean()
    return float((x ** 3).mean() / (x ** 2).mean() ** 1.5)


def _partial_lateral(tibiaTri: Tri, Z0, ml, line, side, flip_ap, notes):
    """PARTIAL (not in STAPLE). A proximal tibia has no fibula or malleolus to tell medial from lateral.

    The plateau's medio-lateral line ``ml`` is known. Anterior comes from the shape of the shaft's
    cross-sections 35-55 mm below the top of the plateau, at the tibial tuberosity and the start of the
    anterior crest: roughly triangular with the apex anterior, so the section's area is skewed towards
    anterior. (On STAPLE's ten example tibias this was right every time; the posterior overhang of the
    plateau, used as a fallback when the bone is too short, was wrong on two.) Lateral then follows from
    the side: lateral = side_sign * (anterior x superior). flip_ap reverses the anterior guess.
    """
    sgn, _ = bodySide2Sign(side)
    Z0 = normalizeV(Z0)
    ml = normalizeV(ml - (ml @ Z0) * Z0)
    d, p0 = normalizeV(line[0]), np.asarray(line[1], float)
    ap = normalizeV(np.cross(ml, d))
    rel = tibiaTri.Points - p0
    t = rel @ d
    top, bottom = t.max(), t.min()
    sk = [_section_ap_skewness(tibiaTri, d, p0, ap, top - dep) for dep in (35, 40, 45, 50, 55)
          if top - dep > bottom + 5]
    sk = [v for v in sk if v is not None]
    if sk and abs(np.mean(sk)) >= 0.04:
        ant = ap if np.mean(sk) > 0 else -ap
        if abs(np.mean(sk)) < 0.08:
            notes.append(f"Anterior/posterior was not clear-cut on this partial tibia (section skewness "
                         f"{abs(np.mean(sk)):.2f}); check the axes.")
    else:
        band = (t < top - 5) & (t > top - 30)
        if band.sum() < 50:
            raise StapleError("Too little bone below the plateau to tell anterior from posterior.")
        off = (rel[band] - np.outer(t[band], d)) @ ap
        plus, minus = off.max(), -off.min()
        ant = -ap if plus > minus else ap
        notes.append("The tibia is too short below the plateau to use the tuberosity; anterior was taken from "
                     "the plateau's posterior overhang, which is less reliable. Check the axes.")
    if flip_ap:
        ant = -ant
    right = np.cross(ant, Z0)
    return np.sign((sgn * right) @ ml) * ml


# ---------------------------------------------------------------- GIBOC tibia

TIBIA_FIT_METHODS = ("centroids", "ellipse", "plateau")


def GIBOC_tibia(tibiaTri: Tri, side="r", fit_method="ellipse", whole=None, flip_ap=False,
                flip_long=False) -> BoneResult:
    """GIBOC-Tibia (Renault et al. 2018) as implemented in STAPLE (Modenese & Renault 2021)."""
    if fit_method not in TIBIA_FIT_METHODS:
        raise ValueError(f"fit_method must be one of {TIBIA_FIT_METHODS}")
    tibiaTri = ensure_outward(tibiaTri)
    sgn, low = bodySide2Sign(side)
    notes = []
    V_all, CenterVol, Inertia = TriInertiaPpties(tibiaTri)[:3]
    if whole is None:
        whole = np.ptp(tibiaTri.Points @ V_all[:, 0]) > 250
    CoeffMorpho = computeTriCoeffMorpho(tibiaTri)
    aux = {"CenterVol": CenterVol, "V_all": V_all, "whole": bool(whole)}
    Z0 = V_all[:, 0]
    if whole:
        U, n = tibia_guess_CS(tibiaTri)
        notes += n
        Prox, Dist = cutLongBoneMesh(tibiaTri, U)
        Z0 = np.sign((Prox.Points.mean(0) - Dist.Points.mean(0)) @ Z0) * Z0
        aux["Z0"] = Z0
        aux["CenterAnkleInside"] = GIBOC_tibia_DistMaxSectCentre(Dist, Z0)
        ankle = GIBOC_tibia_DistArtSurf(Dist, Z0, V_all, CoeffMorpho)
        o, nA = lsplane(ankle.Points, Z0)
        thick = 5.0
        curves, _ = TriPlanIntersect(Dist, nA, o + thick * nA)
        sec = getLargerPlanarSect(curves)[0]
        aux["AnkleCenter"] = PlanPolygonCentroid3D(sec["Pts"])[0] - thick * nA
        U_tmp, MostDistal, just_tibia = tibia_identify_lateral_direction(Dist, Z0)
        aux["Y0"] = normalizeV(U_tmp - (U_tmp @ Z0) * Z0)
        aux["MostDistalPt"] = MostDistal
        art_extra = {"tibiotalar": ankle, "dist_tibia": Dist}
    else:
        # PARTIAL: the whole (proximal) bone stands in for STAPLE's proximal third; no ankle centre
        # (Y along the shaft instead) and no fibula/malleolus (lateral from the plateau, see _partial_lateral).
        Z0 = partial_long_axis(tibiaTri, Z0, epiphysis_at="high") * (-1 if flip_long else 1)
        aux["Z0"] = Z0
        Prox = tibiaTri
        aux["AnkleCenter"] = aux["CenterAnkleInside"] = None
        art_extra = {}
    Prox, n = removeFibulaFromProxTibia(Prox)
    notes += n
    Epi, Zepi = GIBOC_isolate_epiphysis(Prox, Z0, "proximal")
    aux["Zepi"] = Zepi
    if not whole:
        sd, sp, sn = shaft_line(tibiaTri, Z0, Zepi, "proximal")
        aux["shaft_axis"], aux["shaft_point"] = sd, sp
        notes += sn
        # provisional lateral direction from the plateau outline, then signed from the overhang
        y2 = V_all[:, 1] - (V_all[:, 1] @ Z0) * Z0
        aux["Y0"] = normalizeV(y2)
        AS0, oLSP0, Ztp0 = GIBOC_tibia_FullProxArtSurf(Epi, aux, CoeffMorpho)
        Xel0, Yel0, ep0, _ = fitEllipseOnTibialCondylesEdge(AS0, Ztp0, oLSP0)
        aux["Y0"] = _partial_lateral(tibiaTri, Z0, Yel0, (sd, sp), side, flip_ap, notes)
        aux["adaptation"] = ("Partial tibia: Y follows the shaft axis instead of the line from the ankle "
                             "centre (mechanical axis).")
    AS, oLSP, Ztp = GIBOC_tibia_FullProxArtSurf(Epi, aux, CoeffMorpho)
    AS = GIBOC_tibia_ProxArtSurf_it1(Prox, Epi, AS, aux, Ztp, oLSP, CoeffMorpho)
    AS = TriOpenMesh(Epi, AS, 15 * CoeffMorpho)
    AS = TriCloseMesh(Epi, AS, 30 * CoeffMorpho)
    aux["Y0_GIBOC"] = -aux["Y0"]
    Med, Lat, _ = GIBOC_tibia_ProxArtSurf_it2(Epi, AS, aux, CoeffMorpho)
    Plateau = TriUnite(Med, Lat)
    if fit_method == "ellipse":
        aux, JCS = CS_tibia_Ellipse(Plateau, aux, side)
    elif fit_method == "centroids":
        aux, JCS = CS_tibia_ArtSurfCentroids(Med, Lat, aux, side)
    else:
        aux, JCS = CS_tibia_PlateauLayer(Epi, Plateau, aux, side)
    art = {"plateau": Plateau, "plateau_med": Med, "plateau_lat": Lat, "epiphysis": Epi, **art_extra}
    return BoneResult(bone="tibia", method="GIBOC-" + fit_method, side=low, JCS=JCS, aux=aux, art_surf=art,
                      notes=notes, whole=bool(whole))
