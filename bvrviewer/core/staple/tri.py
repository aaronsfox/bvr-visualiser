# Ported from msk-STAPLE (https://github.com/modenaxe/msk-STAPLE), GIBOC-core/SubFunctions/TriangulationFun
# Original MATLAB code: Copyright (c) 2020-2021 Modenese L., Renault J.-B. (GIBOC-core: Renault J.-B.)
# Licensed under CC BY-NC 4.0 (see LICENSE_STAPLE.txt). Changes from the original are listed in NOTICE.md.
"""Triangulated surfaces and the GIBOC-core mesh operations (Tri*.m).

``Tri`` stands in for MATLAB's ``triangulation`` object: the same queries (faceNormal, incenter,
vertexNormal, neighbors, freeBoundary, vertexAttachments, nearestNeighbor) with 0-based indices.
The Tri* functions keep the MATLAB names so the ported algorithms read like the originals.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


class StapleError(RuntimeError):
    """An automatic method could not finish on this bone."""


class Tri:
    """A triangulation: Points (n x 3) and ConnectivityList (m x 3, 0-based)."""

    def __init__(self, faces, points):
        self.ConnectivityList = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        self.Points = np.asarray(points, dtype=float).reshape(-1, 3)
        self._cache = {}

    # ---------------------------------------------------------------- basic queries
    @property
    def n_faces(self):
        return len(self.ConnectivityList)

    @property
    def n_points(self):
        return len(self.Points)

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _cross(self):
        def f():
            P, F = self.Points, self.ConnectivityList
            return np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        return self._get("cross", f)

    @property
    def faceNormal(self):
        def f():
            c = self._cross()
            n = np.linalg.norm(c, axis=1, keepdims=True)
            return c / np.where(n == 0, 1, n)
        return self._get("faceNormal", f)

    @property
    def incenter(self):
        """Centre of each triangle's inscribed circle (as MATLAB's triangulation.incenter)."""
        def f():
            P, F = self.Points, self.ConnectivityList
            A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
            a = np.linalg.norm(B - C, axis=1)[:, None]
            b = np.linalg.norm(C - A, axis=1)[:, None]
            c = np.linalg.norm(A - B, axis=1)[:, None]
            s = a + b + c
            s[s == 0] = 1
            return (a * A + b * B + c * C) / s
        return self._get("incenter", f)

    @property
    def vertexNormal(self):
        """Unit normal at each vertex: the mean of the attached faces' unit normals (unweighted, which
        reproduces MATLAB's triangulation.vertexNormal exactly on STAPLE's reference data)."""
        def f():
            c = self.faceNormal
            n = np.zeros_like(self.Points)
            for k in range(3):
                np.add.at(n, self.ConnectivityList[:, k], c)
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            return n / np.where(ln == 0, 1, ln)
        return self._get("vertexNormal", f)

    def _vf(self):
        """Sparse vertex x face incidence."""
        def f():
            F = self.ConnectivityList
            m = len(F)
            rows = F.ravel()
            cols = np.repeat(np.arange(m), 3)
            return sparse.csr_matrix((np.ones(3 * m, bool), (rows, cols)), shape=(self.n_points, m))
        return self._get("vf", f)

    def vertexAttachments(self, ids):
        """Indices of the faces attached to any of the vertices ``ids`` (sorted, unique)."""
        ids = np.unique(np.asarray(ids, dtype=np.int64).ravel())
        if len(ids) == 0:
            return np.zeros(0, np.int64)
        sub = self._vf()[ids]
        return np.unique(sub.indices).astype(np.int64)

    def _edges(self):
        """Per face, the edge opposite each vertex, plus a map edge-key -> faces."""
        def f():
            F = self.ConnectivityList
            m = len(F)
            # edge opposite local vertex j: (j+1, j+2)
            e = np.stack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], axis=1)       # m x 3 x 2
            key = np.sort(e, axis=2)
            flat = key.reshape(-1, 2)
            uniq, inv, cnt = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
            return e, inv.reshape(m, 3), cnt, uniq
        return self._get("edges", f)

    @property
    def neighbors(self):
        """m x 3: face sharing the edge opposite each vertex, or -1 (MATLAB NaN)."""
        def f():
            e, inv, cnt, _ = self._edges()
            m = len(inv)
            nb = -np.ones((m, 3), np.int64)
            flat = inv.ravel()
            order = np.argsort(flat, kind="stable")
            sf = flat[order]
            # pairs of half-edges with the same key
            starts = np.r_[0, np.nonzero(np.diff(sf))[0] + 1]
            ends = np.r_[starts[1:], len(sf)]
            two = (ends - starts) >= 2
            a = order[starts[two]]
            b = order[starts[two] + 1]
            fa, ja = a // 3, a % 3
            fb, jb = b // 3, b % 3
            nb[fa, ja] = fb
            nb[fb, jb] = fa
            return nb
        return self._get("neighbors", f)

    def _face_adj(self):
        def f():
            nb = self.neighbors
            m = len(nb)
            r = np.repeat(np.arange(m), 3)
            c = nb.ravel()
            ok = c >= 0
            return sparse.csr_matrix((np.ones(ok.sum(), bool), (r[ok], c[ok])), shape=(m, m))
        return self._get("fadj", f)

    @property
    def freeBoundary(self):
        """Boundary edges (k x 2), oriented as in their face."""
        def f():
            e, inv, cnt, _ = self._edges()
            free = cnt[inv] == 1
            return e[free].reshape(-1, 2)
        return self._get("freeBoundary", f)

    def free_boundary_vertices(self):
        fb = self.freeBoundary
        return np.unique(fb) if len(fb) else np.zeros(0, np.int64)

    def nearestNeighbor(self, pts):
        tree = self._get("kdtree", lambda: cKDTree(self.Points))
        _, idx = tree.query(np.atleast_2d(np.asarray(pts, float)))
        return np.asarray(idx, np.int64)

    def face_neighbours_of(self, faces):
        """Unique faces adjacent (by an edge) to any of ``faces``."""
        faces = np.asarray(faces, np.int64)
        if len(faces) == 0:
            return faces
        nb = self.neighbors[faces].ravel()
        return np.unique(nb[nb >= 0])


# ---------------------------------------------------------------- helpers

def normalizeV(v):
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    if n == 0:
        raise StapleError("Zero-length vector.")
    return v / n


def _match_rows(A, B):
    """Indices into A of rows that also appear (exactly) in B."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, np.int64)
    a = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * 3))).ravel()
    b = np.ascontiguousarray(B).view(np.dtype((np.void, B.dtype.itemsize * 3))).ravel()
    return np.nonzero(np.isin(a, b))[0]


def trace_loops(segments):
    """Chain 2-column segments into curves, the way STAPLE's loops do.

    Starts from the first unused segment, follows its second node from segment to segment, and stops
    when no unused segment holds the current node. A closed loop ends on its first node again.
    Returns a list of node lists and a list of the segments used by each curve.
    """
    segs = np.asarray(segments, np.int64).reshape(-1, 2)
    n = len(segs)
    used = np.zeros(n, bool)
    where = {}
    for i, (a, b) in enumerate(segs):
        where.setdefault(int(a), []).append(i)
        where.setdefault(int(b), []).append(i)
    curves, curve_segs = [], []
    for s in range(n):
        if used[s]:
            continue
        used[s] = True
        nodes = [int(segs[s, 0]), int(segs[s, 1])]
        cs = [s]
        cur = nodes[-1]
        while True:
            nxt = None
            for k in where.get(cur, ()):
                if not used[k]:
                    nxt = k
                    break
            if nxt is None:
                break
            used[nxt] = True
            cs.append(nxt)
            a, b = segs[nxt]
            cur = int(b) if a == cur else int(a)
            nodes.append(cur)
        curves.append(nodes)
        curve_segs.append(cs)
    return curves, curve_segs


# ---------------------------------------------------------------- Tri* functions

def TriReduceMesh(TR: Tri, ElmtsKept=None, NodesKept=None, NodesPts=None) -> Tri:
    """Sub-mesh made of the given faces, or of the faces attached to the given vertices.

    NodesKept are vertex indices; NodesPts are coordinates (snapped to the nearest vertices), the
    two cases MATLAB's TriReduceMesh tells apart by whether the values are whole numbers.
    """
    if NodesPts is not None:
        NodesKept = np.unique(TR.nearestNeighbor(NodesPts))
    if NodesKept is not None:
        ElmtsKept = TR.vertexAttachments(NodesKept)
    ElmtsKept = np.asarray(ElmtsKept).ravel()
    if ElmtsKept.dtype == bool:
        ElmtsKept = np.nonzero(ElmtsKept)[0]
    ElmtsKept = ElmtsKept.astype(np.int64)
    F = TR.ConnectivityList[ElmtsKept]
    nodes = np.unique(F)
    remap = -np.ones(TR.n_points, np.int64)
    remap[nodes] = np.arange(len(nodes))
    return Tri(remap[F], TR.Points[nodes])


def TriUnite(TR1: Tri, TR2: Tri) -> Tri:
    return Tri(np.vstack([TR1.ConnectivityList, TR2.ConnectivityList + TR1.n_points]),
               np.vstack([TR1.Points, TR2.Points]))


def _grow(TR: Tri, faces, n):
    S = np.zeros(TR.n_faces, bool)
    S[faces] = True
    A = TR._face_adj()
    for _ in range(int(n)):
        S = S | (A @ S.astype(np.int8) > 0)
    return np.nonzero(S)[0]


def TriDilateMesh(TRsup: Tri, TRin: Tri, nbElmts) -> Tri:
    nb = int(np.ceil(nbElmts))
    ia = _match_rows(TRsup.Points, TRin.Points)
    faces = TRsup.vertexAttachments(ia)
    return TriReduceMesh(TRsup, _grow(TRsup, faces, nb))


def TriErodeMesh(TRin: Tri, nbElmts) -> Tri:
    nb = int(np.ceil(nbElmts))
    border = TRin.free_boundary_vertices()
    isb = np.zeros(TRin.n_points, bool)
    isb[border] = True
    faces = np.nonzero(isb[TRin.ConnectivityList].any(axis=1))[0]
    if nb > 1:
        faces = _grow(TRin, faces, nb - 1)
    keep = np.ones(TRin.n_faces, bool)
    keep[faces] = False
    return TriReduceMesh(TRin, np.nonzero(keep)[0])


def TriCloseMesh(TRsup: Tri, TRin: Tri, nbElmts) -> Tri:
    return TriErodeMesh(TriDilateMesh(TRsup, TRin, nbElmts), nbElmts)


def TriOpenMesh(TRsup: Tri, TRin: Tri, nbElmts) -> Tri:
    return TriDilateMesh(TRsup, TriErodeMesh(TRin, nbElmts), nbElmts)


def TriConnectedPatch(TR: Tri, PtsInitial) -> Tri:
    """The part of TR connected (through shared edges) to the vertices nearest PtsInitial."""
    nodes = np.unique(TR.nearestNeighbor(PtsInitial))
    faces = TR.vertexAttachments(nodes)
    ncomp, lab = sparse.csgraph.connected_components(TR._face_adj(), directed=False)
    keep = np.isin(lab, np.unique(lab[faces]))
    return TriReduceMesh(TR, np.nonzero(keep)[0])


def TriMesh2DProperties(TR: Tri) -> dict:
    if TR.n_faces == 0:
        raise StapleError("An articular surface came out empty.")
    area = 0.5 * np.linalg.norm(TR._cross(), axis=1)
    tot = area.sum()
    center = (TR.incenter * area[:, None]).sum(0) / tot
    mn = (TR.faceNormal * area[:, None]).sum(0) / tot
    nm = np.linalg.norm(mn)
    mn = mn / nm if nm > 0 else mn
    return {"Area": area, "TotalArea": tot, "Center": center, "meanNormal": mn,
            "onMeshCenter": TR.Points[TR.nearestNeighbor(center)[0]]}


def TriKeepLargestPatch(TRin: Tri) -> Tri:
    Trin2 = TriErodeMesh(TRin, 1)
    curves, _ = trace_loops(Trin2.freeBoundary) if Trin2.n_faces else ([], [])
    if len(curves) > 1:
        best, best_a = None, -1
        for c in curves:
            patch = TriConnectedPatch(TRin, Trin2.Points[c])
            a = TriMesh2DProperties(patch)["TotalArea"]
            if a > best_a:
                best, best_a = patch, a
        return best
    return TriDilateMesh(TRin, Trin2, 1)


def TriDifferenceMesh(TR1: Tri, TR2: Tri) -> Tri:
    ia = _match_rows(TR1.Points, TR2.Points)
    if len(ia) == 0:
        ia = _match_rows(np.round(TR1.Points, 5), np.round(TR2.Points, 5))
    drop = TR1.vertexAttachments(ia)
    keep = np.ones(TR1.n_faces, bool)
    keep[drop] = False
    return TriReduceMesh(TR1, np.nonzero(keep)[0])


def TriFillPlanarHoles(TRin: Tri) -> Tri:
    """Close each hole with a fan of triangles around the hole's mean point."""
    FB = TRin.freeBoundary
    if len(FB) == 0:
        return TRin
    curves, csegs = trace_loops(FB)
    P = TRin.Points.copy()
    F = [TRin.ConnectivityList]
    tri_center = TRin.Points.mean(0)
    pts = [P]
    n = len(P)
    for nodes, segs in zip(curves, csegs):
        ids = nodes[:-1] if nodes[0] == nodes[-1] else nodes
        hole_c = TRin.Points[ids].mean(0)
        new = n
        n += 1
        pts.append(hole_c[None])
        fb = FB[segs]
        el = np.c_[fb, np.full(len(fb), new)]
        allp = np.vstack(pts)
        U = normalizeV(hole_c - tri_center)
        nr = np.cross(allp[el[:, 1]] - allp[el[:, 0]], allp[el[:, 2]] - allp[el[:, 0]])
        nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-300)
        if np.mean(nr @ U) < 0:
            el = np.c_[fb[:, 0], np.full(len(fb), new), fb[:, 1]]
        F.append(el)
    return Tri(np.vstack(F), np.vstack(pts))


def TriInertiaPpties(Tr: Tri):
    """Principal axes (columns, ascending moments), centroid, inertia matrix, moments and volume of a closed surface."""
    if len(Tr.freeBoundary):
        raise StapleError("The inertia properties are for hole-free triangulations.")
    pc = Tr.Points.mean(0)
    N = Tr.Points - pc
    F = Tr.ConnectivityList
    P1, P2, P3 = N[F[:, 0]], N[F[:, 1]], N[F[:, 2]]
    CP = np.cross(P2 - P1, P3 - P1)
    d0, d1, d2 = CP[:, 0], CP[:, 1], CP[:, 2]

    def sub(w0, w1, w2):
        temp0 = w0 + w1
        f1 = temp0 + w2
        temp1 = w0 * w0
        temp2 = temp1 + w1 * temp0
        f2 = temp2 + w2 * f1
        f3 = w0 * temp1 + w1 * temp2 + w2 * f2
        g0 = f2 + w0 * (f1 + w0)
        g1 = f2 + w1 * (f1 + w1)
        g2 = f2 + w2 * (f1 + w2)
        return f1, f2, f3, g0, g1, g2

    x0, x1, x2 = P1[:, 0], P2[:, 0], P3[:, 0]
    y0, y1, y2 = P1[:, 1], P2[:, 1], P3[:, 1]
    z0, z1, z2 = P1[:, 2], P2[:, 2], P3[:, 2]
    f1x, f2x, f3x, g0x, g1x, g2x = sub(x0, x1, x2)
    f1y, f2y, f3y, g0y, g1y, g2y = sub(y0, y1, y2)
    f1z, f2z, f3z, g0z, g1z, g2z = sub(z0, z1, z2)
    intg = np.array([
        (d0 * f1x).sum(), (d0 * f2x).sum(), (d1 * f2y).sum(), (d2 * f2z).sum(),
        (d0 * f3x).sum(), (d1 * f3y).sum(), (d2 * f3z).sum(),
        (d0 * (y0 * g0x + y1 * g1x + y2 * g2x)).sum(),
        (d1 * (z0 * g0y + z1 * g1y + z2 * g2y)).sum(),
        (d2 * (x0 * g0z + x1 * g1z + x2 * g2z)).sum()])
    intg *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    mass = intg[0]
    C = np.array([intg[1], intg[2], intg[3]]) / mass
    I = np.zeros((3, 3))
    I[0, 0] = intg[5] + intg[6] - mass * (C[[1, 2]] @ C[[1, 2]])
    I[1, 1] = intg[4] + intg[6] - mass * (C[[2, 0]] @ C[[2, 0]])
    I[2, 2] = intg[4] + intg[5] - mass * (C[[0, 1]] @ C[[0, 1]])
    I[0, 1] = -(intg[7] - mass * C[0] * C[1])
    I[1, 2] = -(intg[8] - mass * C[1] * C[2])
    I[2, 0] = -(intg[9] - mass * C[2] * C[0])
    I[1, 0], I[2, 1], I[0, 2] = I[0, 1], I[1, 2], I[2, 0]
    D, V = np.linalg.eigh(I)
    return V, C + pc, I, D, mass


def ensure_outward(TR: Tri) -> Tri:
    """The same surface with its faces wound so the normals point outwards (positive enclosed volume)."""
    P, F = TR.Points, TR.ConnectivityList
    vol = np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6
    return Tri(F[:, [0, 2, 1]], P) if vol < 0 else TR


def _polyarea(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inpolygon(px, py, xv, yv):
    """Points inside (or on) a polygon, as MATLAB's inpolygon (even-odd rule)."""
    px = np.atleast_1d(np.asarray(px, float))
    py = np.atleast_1d(np.asarray(py, float))
    xv = np.asarray(xv, float)
    yv = np.asarray(yv, float)
    if xv[0] != xv[-1] or yv[0] != yv[-1]:
        xv, yv = np.r_[xv, xv[0]], np.r_[yv, yv[0]]
    x1, y1, x2, y2 = xv[:-1], yv[:-1], xv[1:], yv[1:]
    inside = np.zeros(len(px), bool)
    on = np.zeros(len(px), bool)
    scale = max(np.ptp(xv), np.ptp(yv), 1e-12)
    for k in range(0, len(px), 4096):
        X, Y = px[k:k + 4096, None], py[k:k + 4096, None]
        crosses = ((y1 > Y) != (y2 > Y)) & (X < (x2 - x1) * (Y - y1) / np.where(y2 != y1, y2 - y1, 1) + x1)
        inside[k:k + 4096] = crosses.sum(1) % 2 == 1
        # on an edge: collinear and within the segment
        cr = (x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1)
        dot = (X - x1) * (x2 - x1) + (Y - y1) * (y2 - y1)
        L2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        on[k:k + 4096] = ((np.abs(cr) <= 1e-12 * scale * np.sqrt(L2 + 1e-300)) & (dot >= 0) & (dot <= L2)).any(1)
    return inside | on


def TriPlanIntersect(Tr: Tri, n, d):
    """Cross-section curves of a closed surface with the plane n.x + d = 0 (d a number) or the plane
    through the point d (3 numbers) with normal n.

    Returns (curves, total area); each curve is a dict with Pts (closed: last point = first), Area
    and Hole (1 for a hole, -1 for filled).
    """
    n = normalizeV(n)
    d = np.asarray(d, float).ravel()
    if len(d) > 2:
        d = -d @ n
    else:
        d = float(d[0])
    P = Tr.Points
    s = P @ n + d
    ou = (s > 0).astype(np.int8) - (s < 0).astype(np.int8)
    F = Tr.ConnectivityList
    score = ou[F].sum(1)
    fi = np.nonzero(np.abs(score) < 3)[0]
    if len(fi) == 0:
        return [], 0.0
    Fi = F[fi]
    # the two edges of each face whose ends lie on opposite sides
    E = np.stack([Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [2, 0]]], axis=1)       # k x 3 x 2
    st = ou[E].sum(2) == 0
    keys = {}
    inter = []
    segs = []
    for f in range(len(Fi)):
        js = np.nonzero(st[f])[0]
        if len(js) != 2:
            continue
        ids = []
        for j in js:
            a, b = E[f, j]
            k = (a, b) if a < b else (b, a)
            if k not in keys:
                p0, p1 = P[k[0]], P[k[1]]
                den = (p1 - p0) @ n
                r = 0.0 if den == 0 else -(p0 @ n + d) / den
                keys[k] = len(inter)
                inter.append(p0 + r * (p1 - p0))
            ids.append(keys[k])
        segs.append(ids)
    if not segs:
        return [], 0.0
    inter = np.array(inter)
    loops, _ = trace_loops(np.array(segs))
    curves = []
    frames = []
    for nodes in loops:
        pts = inter[nodes]
        core = pts[:-1] if len(pts) > 1 and nodes[0] == nodes[-1] else pts
        w, V = np.linalg.eigh(np.cov(core.T)) if len(core) > 2 else (None, np.eye(3))
        q = pts @ V
        curves.append({"Pts": pts, "Area": _polyarea(q[:, 1], q[:, 2])})
        frames.append(V)
    inout = np.zeros((len(curves), len(curves)))
    for i, ci in enumerate(curves):
        V = frames[i]
        p0 = ci["Pts"][0] @ V
        for j, cj in enumerate(curves):
            if i != j:
                q = cj["Pts"] @ V
                if _inpolygon(p0[1], p0[2], q[:, 1], q[:, 2])[0]:
                    inout[i, j] = 1
    tot = 0.0
    for i, c in enumerate(curves):
        add = 1 - 2 * (int(inout[i].sum()) % 2)
        c["Hole"] = -add
        tot -= c["Hole"] * c["Area"]
    return curves, tot


def TriSliceObjAlongAxis(TriObj: Tri, Axis, step, cut_offset=0.5):
    """Cross-section areas at regular steps along an axis. Returns Areas, Alt, maxArea, maxAreaInd, maxAlt."""
    Axis = np.asarray(Axis, float).ravel()
    z = TriObj.Points @ Axis
    lo, hi = z.min() + cut_offset, z.max() - cut_offset
    nstep = int(np.floor((hi - lo) / step + 1e-10))
    Alt = lo + step * np.arange(nstep + 1)
    Areas = np.array([TriPlanIntersect(TriObj, Axis, -a)[1] for a in Alt])
    k = int(np.argmax(Areas))
    return Areas, Alt, Areas[k], k, Alt[k]


def TriSplitObjects(TrIn: Tri):
    """Connected parts of a surface, largest volume first: list of (Tri, volume)."""
    ncomp, lab = sparse.csgraph.connected_components(TrIn._face_adj(), directed=False)
    parts = []
    for c in range(ncomp):
        t = TriReduceMesh(TrIn, np.nonzero(lab == c)[0])
        try:
            vol = TriInertiaPpties(t)[4]
        except StapleError:
            vol = 0.0
        parts.append((t, vol))
    parts.sort(key=lambda tv: -tv[1])
    return parts


def TriCurvature(TR: Tri):
    """Mean and Gaussian curvature at each vertex from a quadric fitted to its two-ring neighbourhood.

    As GIBOC-core's TriCurvature (after D. Kroon's patchcurvature): local frame (normal, two tangents),
    f = a x^2 + b y^2 + c xy + d x + e y + g fitted by least squares, principal curvatures from the
    Hessian. Coordinates are centred on each vertex before the fit (same result, better conditioned).
    Returns Cmean, Cgaussian, Dir1, Dir2, Lambda1, Lambda2.
    """
    nv = TR.n_points
    N = TR.vertexNormal
    VF = TR._vf().astype(np.int32)
    FV = VF.T.tocsr()
    ring1 = (VF @ FV)                      # vertices of the faces attached to each vertex
    ring1.data[:] = 1
    ring2 = ((ring1 @ VF) > 0).astype(np.int32) @ FV
    ring2 = ring2.tocoo()
    i, j = ring2.row, ring2.col
    # local frames: v = normal, l, k tangents (any orthonormal pair: curvature doesn't depend on it)
    a = np.where(np.abs(N[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    l = np.cross(a, N)
    l /= np.linalg.norm(l, axis=1, keepdims=True)
    k = np.cross(l, N)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    D = TR.Points[j] - TR.Points[i]
    f = np.einsum("ij,ij->i", D, N[i])
    x = np.einsum("ij,ij->i", D, l[i])
    y = np.einsum("ij,ij->i", D, k[i])
    FM = np.stack([x * x, y * y, x * y, x, y, np.ones_like(x)], 1)
    A = np.zeros((nv, 6, 6))
    b = np.zeros((nv, 6))
    np.add.at(A, i, FM[:, :, None] * FM[:, None, :])
    np.add.at(b, i, FM * f[:, None])
    sol = np.zeros((nv, 6))
    try:
        sol = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        for v in range(nv):
            sol[v] = np.linalg.lstsq(A[v], b[v], rcond=None)[0]
    Dxx, Dyy, Dxy = 2 * sol[:, 0], 2 * sol[:, 1], sol[:, 2]
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
    v2x, v2y = 2 * Dxy, Dyy - Dxx + tmp
    mag = np.sqrt(v2x ** 2 + v2y ** 2)
    nz = mag != 0
    v2x = np.where(nz, v2x / np.where(nz, mag, 1), v2x)
    v2y = np.where(nz, v2y / np.where(nz, mag, 1), v2y)
    v1x, v1y = -v2y, v2x
    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)
    sw = np.abs(mu1) < np.abs(mu2)
    L1 = np.where(sw, mu1, mu2)
    L2 = np.where(sw, mu2, mu1)
    I1 = np.where(sw[:, None], np.c_[v2x, v2y], np.c_[v1x, v1y])
    I2 = np.where(sw[:, None], np.c_[v1x, v1y], np.c_[v2x, v2y])
    Dir1 = I1[:, :1] * l + I1[:, 1:] * k
    Dir2 = I2[:, :1] * l + I2[:, 1:] * k
    return (L1 + L2) / 2, L1 * L2, Dir1, Dir2, L1, L2
