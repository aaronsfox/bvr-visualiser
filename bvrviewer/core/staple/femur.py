# Ported from msk-STAPLE (https://github.com/modenaxe/msk-STAPLE): algorithms/GIBOC_femur.m, Kai2014_femur.m,
# CS_femur_*.m and their private functions. Original MATLAB code: Copyright (c) 2020-2021 Modenese L.,
# Renault J.-B. Licensed under CC BY-NC 4.0 (see LICENSE_STAPLE.txt). Changes are listed in NOTICE.md,
# including the partial-bone adaptations marked "PARTIAL" below, which are not part of STAPLE.
"""Femur coordinate systems: GIBOC-Femur (spheres, cylinder, ellipsoids) and Kai et al. (2014).

Axis names follow STAPLE / ISB: X anterior, Y superior, Z to the subject's right (both sides).
The knee frame has Z exact (the condylar axis) and is what the joint uses; the hip frame has Y exact
(the mechanical axis) and needs the femoral head.
"""

from __future__ import annotations

import numpy as np

from .fitting import (LargestEdgeConvHull, PCRegionGrowing, PtsOnCondylesFemur, computeTriCoeffMorpho,
                      cutLongBoneMesh, ellipsoid_fit, fitCSA, lscylinder, lsplane, PlanPolygonCentroid3D,
                      sphereFit)
from .tri import (StapleError, Tri, ensure_outward, TriCloseMesh, TriConnectedPatch, TriCurvature, TriDifferenceMesh,
                  TriDilateMesh, TriFillPlanarHoles, TriInertiaPpties, TriKeepLargestPatch, TriMesh2DProperties,
                  TriOpenMesh, TriPlanIntersect, TriReduceMesh, TriSliceObjAlongAxis, TriUnite, normalizeV)
from .common import BoneResult, bodySide2Sign, partial_long_axis, shaft_line


# ---------------------------------------------------------------- orientation guess

def femur_guess_CS(Femur: Tri):
    """Distal-to-proximal direction: the distal epiphysis sits closer to the diaphysis axis than the
    proximal one (because of the femoral head), more so once the bone is stretched medio-laterally."""
    V_all, CenterVol = TriInertiaPpties(Femur)[:2]
    Z0 = V_all[:, 0]
    P = Femur.Points - CenterVol
    Pd = (P @ V_all) @ np.diag([1, 2, 1]) @ V_all.T
    F2 = Tri(Femur.ConnectivityList, Pd + CenterVol)
    Epi1, Epi2 = cutLongBoneMesh(F2, Z0, 0.10)
    z = F2.Points @ Z0
    L = z.max() - z.min()
    t1 = TriFillPlanarHoles(TriReduceMesh(F2, np.nonzero(F2.incenter @ Z0 < z.max() - 0.2 * L)[0]))
    dia = TriFillPlanarHoles(TriReduceMesh(t1, np.nonzero(t1.incenter @ Z0 > z.min() + 0.2 * L)[0]))
    Vd, Cd = TriInertiaPpties(dia)[:2]
    Z0d = Vd[:, 0]
    C1 = TriInertiaPpties(Epi1)[1]
    C2 = TriInertiaPpties(Epi2)[1]
    d1 = np.linalg.norm(np.cross(C1 - Cd, Z0d))
    d2 = np.linalg.norm(np.cross(C2 - Cd, Z0d))
    U = C2 - C1 if d1 < d2 else C1 - C2
    notes = []
    if abs(d1 - d2) / (d1 + d2) < 0.2:
        notes.append("The two femoral epiphyses were at similar distances from the shaft axis; the "
                     "distal-to-proximal direction could be wrong.")
    return np.sign(U @ Z0) * Z0, notes


# ---------------------------------------------------------------- femoral head

def GIBOC_femur_fitSphere2FemHead(ProxFem: Tri, CSs: dict, CoeffMorpho: float):
    Z0, CenterVol = CSs["Z0"], CSs["CenterVol"]
    i = int(np.argmax(ProxFem.incenter @ Z0))
    ids = np.r_[i, ProxFem.neighbors[i]]
    ids = ids[ids >= 0]
    top = TriDilateMesh(ProxFem, TriReduceMesh(ProxFem, ids), 40 * CoeffMorpho)
    OT = top.Points.mean(0) - CenterVol
    CSs["Y0"] = normalizeV(np.cross(np.cross(Z0, OT), Z0))
    i = int(np.argmax(ProxFem.incenter @ CSs["Y0"]))
    ids = np.r_[i, ProxFem.neighbors[i]]
    ids = ids[ids >= 0]
    mm = TriDilateMesh(ProxFem, TriReduceMesh(ProxFem, ids), 40 * CoeffMorpho)
    FemHead0 = TriUnite(mm, top)
    C, R, err = sphereFit(FemHead0.Points)
    dil = TriDilateMesh(ProxFem, FemHead0, round(1.5 * R * CoeffMorpho))
    CFH, Rdil, _ = sphereFit(dil.Points)
    v = dil.incenter - CFH
    nv = v / np.linalg.norm(v, axis=1, keepdims=True)
    c1 = (nv * dil.faceNormal).sum(1) > 0.975
    c2 = np.abs(np.linalg.norm(dil.incenter - CFH, axis=1) - R) < 0.1 * R
    single = (c1 & c2).sum() <= 20
    applied = c1 if single else (c1 & c2)
    head = TriReduceMesh(dil, np.nonzero(applied)[0])
    if single:
        head = TriKeepLargestPatch(head)
    head = TriOpenMesh(ProxFem, head, 3 * CoeffMorpho)
    CFH, R, errf = sphereFit(head.Points)
    if not np.all(np.isfinite(CFH)):
        raise StapleError("The femoral head sphere fit failed.")
    CSs["CenterFH_Renault"] = CFH
    CSs["RadiusFH_Renault"] = R
    CSs["FemHead_RMSE"] = float(np.mean(np.abs(errf)))
    return CSs, head


def Kai2014_femur_fitSphere2FemHead(ProxFem: Tri, CS: dict):
    sect_pts_limit = 15
    corr_dir = -1
    i = int(np.argmax(ProxFem.Points @ CS["Z0"]))
    MostProx = ProxFem.Points[i]
    CS["Y0"] = normalizeV(np.cross(np.cross(CS["Z0"], MostProx - CS["CenterVol"]), CS["Z0"]))
    CS["X0"] = np.cross(CS["Y0"], CS["Z0"])
    ok, ok_med = [], []
    d = MostProx @ CS["Z0"] - 0.25
    max_so_far = 0.0
    for _ in range(2000):
        curves, _a = TriPlanIntersect(ProxFem, corr_dir * CS["Z0"], d)
        n = len(curves)
        if n == 1 and ok_med:
            break
        d = d - 1
        if n == 1:
            ok.append(curves[0]["Pts"])
        elif n > 1:
            if n == 2 and len(curves[1]["Pts"]) < sect_pts_limit:
                continue
            areas = [c["Area"] for c in curves]
            k = int(np.argmax(areas))
            ok_med.append(curves[k]["Pts"])
            if areas[k] >= max_so_far:
                max_so_far = areas[k]
            else:
                break
        elif n == 0 and (ok or ok_med):
            break
    pts = np.vstack(ok + ok_med)
    pts = pts[(pts - MostProx) @ CS["Y0"] > 0]
    C, R, err = sphereFit(pts)
    CS["CenterFH_Kai"] = C
    CS["RadiusFH_Kai"] = R
    return CS, MostProx


# ---------------------------------------------------------------- distal epiphysis and condyles

def GIBOC_isolate_epiphysis(TriObj: Tri, Z0, prox_epi: str):
    Areas, Alt = TriSliceObjAlongAxis(TriObj, Z0, 1.0)[:2]
    _, Zepi, _ = fitCSA(Alt, Areas)
    ic = TriObj.incenter @ Z0
    keep = ic > Zepi if prox_epi == "proximal" else ic < Zepi
    return TriReduceMesh(TriObj, np.nonzero(keep)[0]), Zepi


def GIBOC_femur_processEpiPhysis(EpiFem: Tri, CSs: dict, Inertia_Vects, edge_threshold, axes_dev_thresh):
    pairs, Edges = LargestEdgeConvHull(EpiFem.Points)
    bnd = np.zeros(EpiFem.n_points, bool)
    bnd[EpiFem.free_boundary_vertices()] = True
    target = int((Edges > edge_threshold * Edges[0]).sum())
    kept = []
    i = 0
    while len(kept) != target and i < len(pairs):
        if not bnd[pairs[i, 0]] and not bnd[pairs[i, 1]]:
            kept.append(i)
        i += 1
    Id = pairs[kept]
    Axes = EpiFem.Points[Id[:, 0]] - EpiFem.Points[Id[:, 1]]
    keep = Axes @ CSs["Y0"] >= 0
    Id, Axes = Id[keep], Axes[keep]
    U = Axes / np.linalg.norm(Axes, axis=1, keepdims=True)
    keep = np.abs(U @ Inertia_Vects[:, 1]) >= axes_dev_thresh
    Id, U = Id[keep], U[keep]
    if len(U) == 0:
        raise StapleError("No medio-lateral edges found across the femoral condyles.")
    _, good = PCRegionGrowing(U, normalizeV(U.mean(0)), 0.1)
    Id, U = Id[good], U[good]
    orientation = np.round(np.mean(np.sign(U @ CSs["Y0"])))
    med_lat = [1, 0] if orientation < 0 else [0, 1]
    return Id, U, med_lat


def GIBOC_femur_getCondyleMostProxPoint(EpiFem: Tri, CSs: dict, PtsTrace, U):
    from scipy.spatial import cKDTree
    sphere_r, thick = 7.5, 2.5
    P_c, nrm = lsplane(PtsTrace)
    dm = -P_c @ nrm
    P = EpiFem.Points
    on_plane = (np.abs(P @ nrm + dm) < thick) & (P @ CSs["Z0"] > np.max(PtsTrace @ CSs["Z0"] - thick))
    near = np.zeros(len(P), bool)
    for lst in cKDTree(P).query_ball_point(PtsTrace, sphere_r):
        near[lst] = True
    IOK = np.nonzero(on_plane & near)[0]
    if len(IOK) == 0:
        raise StapleError("Couldn't find the top of a femoral condyle.")
    k = int(np.argmax(EpiFem.vertexNormal[IOK] @ U))
    return P[IOK[k]]


def GIBOC_femur_filterCondyleSurf(EpiFem: Tri, CSs: dict, PtsCondyle, Pts_0_C, CoeffMorpho):
    Y1 = CSs["Y1"]
    Center = sphereFit(PtsCondyle)[0]
    Condyle = TriReduceMesh(EpiFem, NodesPts=PtsCondyle)
    Condyle = TriCloseMesh(EpiFem, Condyle, 4 * CoeffMorpho)
    Cmean, Cgauss = TriCurvature(Condyle)[:2]
    Curv = np.sqrt(np.maximum(4 * Cmean ** 2 - 2 * Cgauss, 0))
    Cyl = Condyle.Points - Center
    Ui = Cyl - np.outer(Cyl @ Y1, Y1)
    Ui /= np.linalg.norm(Ui, axis=1, keepdims=True)
    vn = Condyle.vertexNormal
    alpha = np.abs(90 - np.degrees(np.arccos(np.clip((vn * Ui).sum(1), -1, 1))))
    gamma = np.degrees(np.arccos(np.clip(vn @ CSs["Z0"], -1, 1)))
    pa = 1 / (1 + np.exp((alpha - 50) / 10))
    pa /= pa.max()
    pc = 1 / (1 + np.exp(-(Curv - 0.25) / 0.05))
    pc /= pc.max()
    pf = 1 / (1 + np.exp((gamma - 45) / 15))
    pf /= pf.max()
    pe = 0.6 * np.sqrt(pa * pc) + 0.05 * pc + 0.15 * pa + 0.2 * pf
    edges = TriReduceMesh(Condyle, NodesKept=np.nonzero(pc * pa > 0.5)[0])
    out = TriReduceMesh(Condyle, NodesKept=np.nonzero(pe < 0.20)[0])
    out = TriConnectedPatch(out, Pts_0_C)
    out = TriCloseMesh(EpiFem, out, 10 * CoeffMorpho)
    out = TriKeepLargestPatch(out)
    if edges.n_faces:
        out = TriDifferenceMesh(out, edges)
    return out


def GIBOC_femur_smoothCondyles(EpiFem: Tri, PtsFullCondyle, CoeffMorpho):
    c = TriReduceMesh(EpiFem, NodesPts=PtsFullCondyle)
    c = TriCloseMesh(EpiFem, c, 5 * CoeffMorpho)
    return TriOpenMesh(EpiFem, c, 15 * CoeffMorpho)


_ART = {"full_condyles": (70, 85, 0.8, 0.025), "post_condyles": (10, 25, 0.6, 0.025),
        "pat_groove": (10, 25, 0.6, 0.025)}


def GIBOC_femur_ArticSurf(EpiFem: Tri, CSs: dict, CoeffMorpho, art_surface: str):
    CutLat, CutMed, InSet, dil = _ART[art_surface]
    V_all, Z0 = CSs["V_all"], CSs["Z0"]
    edge_threshold, axes_dev = 0.5, 0.75
    Id, U_Axes, ml = GIBOC_femur_processEpiPhysis(EpiFem, CSs, V_all, edge_threshold, axes_dev)
    PMed = EpiFem.Points[Id[:, ml[0]]]
    PLat = EpiFem.Points[Id[:, ml[1]]]
    Y1 = normalizeV(U_Axes.sum(0))
    X1 = normalizeV(np.cross(Y1, Z0))
    Z1 = np.cross(X1, Y1)
    VC = np.c_[X1, Y1, Z1]
    n_in = int(np.ceil(len(Id) * edge_threshold))
    mid_post = np.mean(0.5 * (PMed[:n_in] + PLat[:n_in]), axis=0)
    mid_epi = EpiFem.Points.mean(0)
    X1 = np.sign((mid_epi - mid_post) @ X1) * X1
    U = normalizeV(3 * Z0 - X1)
    top_med = GIBOC_femur_getCondyleMostProxPoint(EpiFem, CSs, PMed, U)
    top_lat = GIBOC_femur_getCondyleMostProxPoint(EpiFem, CSs, PLat, U)
    mid = np.mean(0.5 * (PMed + PLat), axis=0)
    axis_proj = mid @ VC
    epi2 = EpiFem.Points @ VC
    pl = np.vstack([PLat, top_lat, top_lat]) @ VC
    pm = np.vstack([PMed, top_med, top_med]) @ VC
    P0C1 = pl @ VC.T
    P0C2 = pm @ VC.T
    c1 = epi2[epi2[:, 1] - axis_proj[1] < 0]
    c2 = epi2[epi2[:, 1] - axis_proj[1] > 0]
    AS_lat = PtsOnCondylesFemur(pl, c1, CutLat, InSet, dil)[0] @ VC.T
    AS_med = PtsOnCondylesFemur(pm, c2, CutMed, InSet, dil)[0] @ VC.T
    mid2 = np.vstack([AS_lat, AS_med]).mean(0)
    X1 = np.sign((mid_epi - mid2) @ X1) * X1
    U = normalizeV(-Z0 - 3 * X1)
    nodes_ok = EpiFem.Points[EpiFem.vertexNormal @ U > 0.98]
    U = normalizeV(Z0 - 3 * X1)
    if len(nodes_ok) == 0:
        raise StapleError("Couldn't find the intercondylar notch.")
    notch = nodes_ok[int(np.argmin(nodes_ok @ U))]
    CSs.setdefault("BL", {})["PtNotch"] = notch
    CSs["X1"], CSs["Y1"], CSs["Z1"] = X1, Y1, Z1
    if art_surface == "full_condyles":
        lat = GIBOC_femur_smoothCondyles(EpiFem, AS_lat, CoeffMorpho)
        med = GIBOC_femur_smoothCondyles(EpiFem, AS_med, CoeffMorpho)
    elif art_surface == "post_condyles":
        lim = notch @ X1
        AS_lat = AS_lat[AS_lat @ X1 <= lim]
        AS_med = AS_med[AS_med @ X1 <= lim]
        P0C1 = P0C1[P0C1 @ X1 <= lim]
        P0C2 = P0C2[P0C2 @ X1 <= lim]
        lat = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, AS_lat, P0C1, CoeffMorpho)
        med = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, AS_med, P0C2, CoeffMorpho)
    else:
        raise StapleError(f"Unsupported articular surface {art_surface!r}.")
    return med, lat, CSs


# ---------------------------------------------------------------- coordinate systems on the condyles

def _mech_y(CS, KneeCenter):
    """Y: towards the femoral head centre (STAPLE) or, for a partial femur, along the shaft (PARTIAL)."""
    if CS.get("CenterFH_Renault") is not None:
        return normalizeV(CS["CenterFH_Renault"] - KneeCenter)
    return normalizeV(CS["shaft_axis"])


def _femur_jcs(CS, X_from_YZ, Y, Z, KneeCenter, side_low):
    X = normalizeV(np.cross(Y, Z))
    JCS = {}
    if CS.get("CenterFH_Renault") is not None:
        JCS["hip_" + side_low] = {"V": np.c_[X, Y, normalizeV(np.cross(X, Y))], "Origin": CS["CenterFH_Renault"]}
    JCS["knee_" + side_low] = {"V": np.c_[X, normalizeV(np.cross(Z, X)), Z], "Origin": KneeCenter}
    return JCS


def CS_femur_SpheresOnCondyles(postLat: Tri, postMed: Tri, CS: dict, side):
    sgn, low = bodySide2Sign(side)
    cl, rl, el = sphereFit(postLat.Points)
    cm, rm, em = sphereFit(postMed.Points)
    K = 0.5 * (cl + cm)
    CS.update(sphere_center_lat=cl, sphere_radius_lat=rl, sphere_center_med=cm, sphere_radius_med=rm,
              sphere_rmse=float(np.sqrt(np.mean(np.r_[(np.sqrt(el + rl ** 2) - rl) ** 2,
                                                      (np.sqrt(em + rm ** 2) - rm) ** 2]))))
    Z = normalizeV(cl - cm) * sgn
    Y = _mech_y(CS, K)
    return CS, _femur_jcs(CS, None, Y, Z, K, low)


def CS_femur_CylinderOnCondyles(Lat: Tri, Med: Tri, CS: dict, side, tolp=1e-3, tolg=1e-3):
    sgn, low = bodySide2Sign(side)
    Pts = np.vstack([Lat.Points, Med.Points])
    CSs, JS = CS_femur_SpheresOnCondyles(Lat, Med, dict(CS), side)
    Axe0 = CSs["sphere_center_lat"] - CSs["sphere_center_med"]
    C0 = 0.5 * (CSs["sphere_center_lat"] + CSs["sphere_center_med"])
    R0 = 0.5 * (CSs["sphere_radius_lat"] + CSs["sphere_radius_med"])
    Zdir = JS["knee_" + low]["V"][:, 2]
    x0n, an, rn, res = lscylinder(Pts, C0, Axe0, R0, tolp, tolg)
    Y2 = normalizeV(an)
    cl = TriMesh2DProperties(Lat)["Center"]
    cm = TriMesh2DProperties(Med)["Center"]
    on_l = x0n + ((cl - x0n) @ Y2) * Y2
    on_m = x0n + ((cm - x0n) @ Y2) * Y2
    K = 0.5 * on_l + 0.5 * on_m
    proj = (Pts - x0n) @ Y2
    CS.update(Cyl_Y=Y2, Cyl_Pt=x0n, Cyl_Radius=rn, Cyl_Range=float(np.ptp(proj)),
              Cyl_rmse=float(np.sqrt(np.mean(res ** 2))))
    Y = _mech_y(CS, K)
    Z = normalizeV(np.sign(Y2 @ Zdir) * Y2)
    return CS, _femur_jcs(CS, None, Y, Z, K, low)


def CS_femur_EllipsoidsOnCondyles(Lat: Tri, Med: Tri, CS: dict, side):
    sgn, low = bodySide2Sign(side)
    cl, rl, el = ellipsoid_fit(Lat.Points)[:3]
    cm, rm, em = ellipsoid_fit(Med.Points)[:3]
    CS.update(ellips_centre_med=cm, ellips_centre_lat=cl, ellips_radii_med=rm, ellips_radii_lat=rl,
              ellips_evec_med=em, ellips_evec_lat=el)
    K = 0.5 * (cm + cl)
    Z = normalizeV(cl - cm) * sgn
    Y = _mech_y(CS, K)
    return CS, _femur_jcs(CS, None, Y, Z, K, low)


# ---------------------------------------------------------------- PARTIAL: medial direction without the head

def _partial_medial_direction(EpiFem: Tri, CSs: dict, side):
    """PARTIAL (not in STAPLE). Without the femoral head there is no medial direction Y0. The convex-hull
    edges across the condyles still give the medio-lateral line, and STAPLE's own test (the epiphysis
    centroid lies anterior to the posterior condyles) gives anterior, X1. Medial then follows from the
    side: medial = -side_sign * (X1 x Z0)."""
    sgn, _ = bodySide2Sign(side)
    Id, U, _ = GIBOC_femur_processEpiPhysis(EpiFem, CSs, CSs["V_all"], 0.5, 0.75)
    Y1 = normalizeV(U.sum(0))
    X1 = normalizeV(np.cross(Y1, CSs["Z0"]))
    n_in = int(np.ceil(len(Id) * 0.5))
    P = EpiFem.Points
    mid_post = np.mean(0.5 * (P[Id[:n_in, 0]] + P[Id[:n_in, 1]]), axis=0)
    X1 = np.sign((P.mean(0) - mid_post) @ X1) * X1
    right = normalizeV(np.cross(X1, CSs["Z0"]))
    return -sgn * right


# ---------------------------------------------------------------- GIBOC femur

FEMUR_FIT_METHODS = ("spheres", "cylinder", "ellipsoids")


def GIBOC_femur(femurTri: Tri, side="r", fit_method="cylinder", whole=None, flip_ap=False,
                flip_long=False) -> BoneResult:
    """GIBOC-Femur (Renault et al. 2018) as implemented in STAPLE (Modenese & Renault 2021).

    whole: True for a complete femur (STAPLE as published), False for a distal femur (PARTIAL
    adaptation), None to decide from the bone's length. flip_ap / flip_long reverse the partial-bone
    guesses of anterior and of the distal-to-proximal direction (they don't apply to whole bones).
    """
    if fit_method not in FEMUR_FIT_METHODS:
        raise ValueError(f"fit_method must be one of {FEMUR_FIT_METHODS}")
    femurTri = ensure_outward(femurTri)
    sgn, low = bodySide2Sign(side)
    notes = []
    V_all, CenterVol, Inertia = TriInertiaPpties(femurTri)[:3]
    if whole is None:
        whole = np.ptp(femurTri.Points @ V_all[:, 0]) > 300
    CoeffMorpho = computeTriCoeffMorpho(femurTri)
    Z0 = V_all[:, 0]
    aux = {"CenterVol": CenterVol, "V_all": V_all, "whole": bool(whole)}
    head = None
    if whole:
        U, n = femur_guess_CS(femurTri)
        notes += n
        Prox, Dist = cutLongBoneMesh(femurTri, U)
        Z0 = np.sign((Prox.Points.mean(0) - Dist.Points.mean(0)) @ Z0) * Z0
        aux["Z0"] = Z0
        try:
            aux, head = GIBOC_femur_fitSphere2FemHead(Prox, aux, CoeffMorpho)
        except Exception:                                   # STAPLE falls back to Kai's fit
            notes.append("Renault2018 head fitting failed; used Kai et al.'s femoral head fit.")
            aux, _ = Kai2014_femur_fitSphere2FemHead(Prox, aux)
            aux["CenterFH_Renault"], aux["RadiusFH_Renault"] = aux["CenterFH_Kai"], aux["RadiusFH_Kai"]
        aux["X0"] = np.cross(aux["Y0"], Z0)
        Epi, Zepi = GIBOC_isolate_epiphysis(Dist, Z0, "distal")
    else:
        # PARTIAL: the whole (distal) bone stands in for STAPLE's distal third; Z0 points away from the
        # wide (condylar) end; Y0 is found from the condyles instead of the head.
        Z0 = partial_long_axis(femurTri, V_all[:, 0], epiphysis_at="low")
        if flip_long:
            Z0 = -Z0
        aux["Z0"] = Z0
        Dist = femurTri
        Epi, Zepi = GIBOC_isolate_epiphysis(Dist, Z0, "distal")
        y2 = V_all[:, 1] - (V_all[:, 1] @ Z0) * Z0
        aux["Y0"] = normalizeV(y2)
        aux["Y0"] = _partial_medial_direction(Epi, aux, side) * (-1 if flip_ap else 1)
        aux["X0"] = np.cross(aux["Y0"], Z0)
        aux["CenterFH_Renault"] = None
        aux["shaft_axis"], aux["shaft_point"], sn = shaft_line(femurTri, Z0, Zepi, "distal")
        notes += sn
        aux["adaptation"] = ("Partial femur: Y follows the shaft axis instead of the line to the femoral head "
                             "(mechanical axis); they differ by a few degrees in the frontal plane.")
    aux["Zepi"] = Zepi
    full_med, full_lat, aux = GIBOC_femur_ArticSurf(Epi, aux, CoeffMorpho, "full_condyles")
    post_med, post_lat, aux = GIBOC_femur_ArticSurf(Epi, aux, CoeffMorpho, "post_condyles")
    if fit_method == "spheres":
        aux, JCS = CS_femur_SpheresOnCondyles(post_lat, post_med, aux, side)
        used = (post_lat, post_med)
    elif fit_method == "cylinder":
        aux, JCS = CS_femur_CylinderOnCondyles(post_lat, post_med, aux, side)
        used = (post_lat, post_med)
    else:
        aux, JCS = CS_femur_EllipsoidsOnCondyles(full_lat, full_med, aux, side)
        used = (full_lat, full_med)
    art = {"med_cond": full_med, "lat_cond": full_lat, "post_med_cond": post_med, "post_lat_cond": post_lat,
           "epiphysis": Epi, "fitted_lat": used[0], "fitted_med": used[1]}
    if head is not None:
        art["hip"] = head
    return BoneResult(bone="femur", method="GIBOC-" + fit_method, side=low, JCS=JCS, aux=aux, art_surf=art,
                      notes=notes, whole=bool(whole))


# ---------------------------------------------------------------- Kai et al. 2014

def sliceFemoralCondyles(DistFem: Tri, CS: dict, depth=0):
    X0, Z0 = CS["X0"], CS["Z0"]
    area_limit, sections_limit = 4, 15
    i = int(np.argmax(DistFem.Points @ -X0))
    MostPost = DistFem.Points[i]
    med, lat = [], []
    d = MostPost @ -X0 - 0.25
    top = (DistFem.Points @ Z0).max()
    for _ in range(2000):
        curves, _a = TriPlanIntersect(DistFem, X0, d)
        n = len(curves)
        for c in curves:
            if abs((c["Pts"] @ Z0).max() - top) < 8:
                if depth > 0:
                    raise StapleError("Couldn't slice the femoral condyles in either direction.")
                CS["X0"] = -CS["X0"]
                return sliceFemoralCondyles(DistFem, CS, depth + 1)
        if n == 1 and lat:
            break
        d = d - 1
        if n > 2 or n == 0:
            if n == 0 and (med or lat):
                break
            continue
        if n == 1:
            med.append(curves[0]["Pts"])
        else:
            if len(curves[1]["Pts"]) < sections_limit:
                continue
            _, a2 = PlanPolygonCentroid3D(curves[1]["Pts"])
            if abs(a2) < area_limit:
                continue
            cen = [c["Pts"].mean(0) for c in curves]
            dd = [np.linalg.norm(MostPost - c) for c in cen]
            k = int(np.argmin(dd))
            med.append(curves[k]["Pts"])
            lat.append(curves[1 - k]["Pts"])
    return CS, np.vstack(med) if med else np.zeros((0, 3)), np.vstack(lat) if lat else np.zeros((0, 3))


def Kai2014_femur_fitSpheres2Condyles(DistFem: Tri, CS: dict):
    """Slices from the back of the condyles forwards; two sections = the two condyles. (STAPLE's Kai
    condyle slicer and sliceFemoralCondyles differ only in a centroid-distance check, kept here.)"""
    X0, Z0 = CS["X0"], CS["Z0"]
    area_limit, sections_limit, dist_thr = 4, 15, 20
    i = int(np.argmax(DistFem.Points @ -X0))
    MostPost = DistFem.Points[i]
    med, lat = [], []
    d = MostPost @ -X0 - 0.25
    top = (DistFem.Points @ Z0).max()
    for _ in range(2000):
        curves, _a = TriPlanIntersect(DistFem, X0, d)
        n = len(curves)
        flipped = False
        for c in curves:
            if abs((c["Pts"] @ Z0).max() - top) < 8:
                CS["X0"] = -CS["X0"]
                CS, m2, l2 = sliceFemoralCondyles(DistFem, CS)
                med, lat = [m2], [l2]
                flipped = True
                break
        if flipped or (n == 1 and lat):
            break
        d = d - 1
        if n > 2 or n == 0:
            if n == 0 and (med or lat):
                break
            continue
        if n == 1:
            med.append(curves[0]["Pts"])
        else:
            if len(curves[1]["Pts"]) < sections_limit:
                continue
            c2, a2 = PlanPolygonCentroid3D(curves[1]["Pts"])
            if abs(a2) < area_limit:
                continue
            c1, _ = PlanPolygonCentroid3D(curves[0]["Pts"])
            if np.linalg.norm(c1 - c2) < dist_thr:
                continue
            cen = [c["Pts"].mean(0) for c in curves]
            k = int(np.argmin([np.linalg.norm(MostPost - c) for c in cen]))
            med.append(curves[k]["Pts"])
            lat.append(curves[1 - k]["Pts"])
    med = np.vstack(med) if med else np.zeros((0, 3))
    lat = np.vstack(lat) if lat else np.zeros((0, 3))
    if len(med) < 10 or len(lat) < 10:
        raise StapleError("Kai2014: couldn't separate the two femoral condyles.")
    cm, rm, _ = sphereFit(med)
    cl, rl, _ = sphereFit(lat)
    K = 0.5 * (cm + cl)
    if np.sign(CS["Y0"] @ (cm - cl)) < 0:            # Y0 points medially
        cm, rm, cl, rl = cl, rl, cm, rm
        CS.setdefault("notes", []).append("Kai2014: medial and lateral condyles were swapped (corrected).")
    CS.update(Center_Lat=cl, Radius_Lat=rl, Center_Med=cm, Radius_Med=rm, KneeCenter=K,
              FC_Med_Pts=med, FC_Lat_Pts=lat)
    return CS


def _pca(P):
    P = np.asarray(P, float)
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return Vt.T


def Kai2014_femur(femurTri: Tri, side="r", whole=None, flip_ap=False, flip_long=False) -> BoneResult:
    """Kai et al. (2014) femur, as implemented in STAPLE.

    Whole femurs only: the condyles are sliced across a direction set by the femoral head. (Estimating
    that direction on partial femurs moved the result by 10-90 degrees in testing.)"""
    femurTri = ensure_outward(femurTri)
    sgn, low = bodySide2Sign(side)
    notes = []
    V_all = _pca(femurTri.Points)
    Vi, CenterVol = TriInertiaPpties(femurTri)[:2]
    if whole is None:
        whole = np.ptp(femurTri.Points @ Vi[:, 0]) > 300
    aux = {"CenterVol": CenterVol, "whole": bool(whole)}
    if whole:
        U, n = femur_guess_CS(femurTri)
        notes += n
        Prox, Dist = cutLongBoneMesh(femurTri, U)
        Z0 = V_all[:, 0]
        Z0 = np.sign((Prox.Points.mean(0) - Dist.Points.mean(0)) @ Z0) * Z0
        aux["Z0"] = Z0
        aux, _ = Kai2014_femur_fitSphere2FemHead(Prox, aux)
        head = aux["CenterFH_Kai"]
    else:
        raise StapleError("Kai et al.'s femur method needs the whole femur (it starts from the femoral head). "
                          "Use a GIBOC method for a partial femur.")
    aux = Kai2014_femur_fitSpheres2Condyles(Dist, aux)
    notes += aux.pop("notes", [])
    Z = normalizeV(aux["Center_Lat"] - aux["Center_Med"]) * sgn
    Y = normalizeV(head - aux["KneeCenter"])
    X = normalizeV(np.cross(Y, Z))
    JCS = {"hip_" + low: {"V": np.c_[X, Y, normalizeV(np.cross(X, Y))], "Origin": head}}
    JCS["knee_" + low] = {"V": np.c_[X, normalizeV(np.cross(Z, X)), Z], "Origin": aux["KneeCenter"]}
    return BoneResult(bone="femur", method="Kai2014", side=low, JCS=JCS, aux=aux, art_surf={}, notes=notes,
                      whole=bool(whole))
