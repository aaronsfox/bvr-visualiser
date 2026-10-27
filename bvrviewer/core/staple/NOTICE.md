# STAPLE port: notice

This folder (`bvrviewer/core/staple/`) is a Python translation of parts of the **STAPLE** MATLAB toolbox
(Shared Tools for Automatic Personalised Lower Extremity modelling):

- Repository: https://github.com/modenaxe/msk-STAPLE (translated from commit `eec7db1`, 8 Feb 2024)
- Copyright (c) 2020-2021 Luca Modenese and Jean-Baptiste Renault
- Licence: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); full text in
  `LICENSE_STAPLE.txt`, summary at https://creativecommons.org/licenses/by-nc/4.0/

STAPLE includes GIBOC-core, a minimal version of GIBOC-Knee (Copyright (c) 2020 Jean-Baptiste Renault,
MIT licence, `third_party_licenses/LICENSE_GIBOC-core_MIT.txt`), which in turn uses functions by other
authors under BSD licences (`third_party_licenses/`): ellipsoid_fit (Yury Petrov), patchcurvature
(Dirk-Jan Kroon, used by TriCurvature), fit_ellipse (Ohad Gal) and sphereFit (Alan Jennings).

**Non-commercial use only.** Because of the CC BY-NC licence, this folder, and any build of the BVR
Motion Viewer that includes it, may not be used for commercial purposes. The rest of the viewer does not
depend on it: without this folder the Model Builder still offers landmark-based axes.

STAPLE's licensors do not endorse this translation or the BVR Motion Viewer.

## Please cite

- Modenese L, Renault J-B (2021). Automatic generation of personalised skeletal models of the lower limb
  from three-dimensional bone geometries. *Journal of Biomechanics* 116:110186.
  https://doi.org/10.1016/j.jbiomech.2020.110186
- Renault J-B, Aüllo-Rasser G, Donnez M, Parratte S, Chabrand P (2018).
  Articular-surface-based automatic anatomical coordinate systems for the knee bones. *Journal of
  Biomechanics* 80:171-178. https://doi.org/10.1016/j.jbiomech.2018.08.028 (GIBOC methods)
- Kai S, Sato T, Koga Y, Omori G, Kobayashi K, Sakamoto M, Tanabe Y (2014). Automatic construction of an
  anatomical coordinate system for three-dimensional bone models of the lower extremities - pelvis,
  femur, and tibia. *Journal of Biomechanics* 47:1229-1233. (Kai methods)

## What was translated

| Python | STAPLE MATLAB source |
|---|---|
| `tri.py` | GIBOC-core `TriangulationFun/`: TriReduceMesh, TriDilateMesh, TriErodeMesh, TriOpenMesh, TriCloseMesh, TriConnectedPatch, TriKeepLargestPatch, TriDifferenceMesh, TriUnite, TriFillPlanarHoles, TriInertiaPpties (+Subexpressions), TriMesh2DProperties, TriPlanIntersect, TriSliceObjAlongAxis, TriSplitObjects, TriCurvature; MATLAB's `triangulation` queries |
| `fitting.py` | GIBOC-core `FittingFun/` and `GeometricFun/`: sphereFit, ellipsoid_fit, fit_ellipse, lsplane, lscylinder (see changes), fitCSA, PlanPolygonCentroid3D, ProjectOnPlan, getLargerPlanarSect, LargestEdgeConvHull, PCRegionGrowing, PtsOnCondylesFemur, cutLongBoneMesh, computeTriCoeffMorpho, fitEllipseOnTibialCondylesEdge, ankleSurfFit |
| `femur.py` | `algorithms/`: GIBOC_femur, Kai2014_femur, CS_femur_SpheresOnCondyles, CS_femur_CylinderOnCondyles, CS_femur_EllipsoidsOnCondyles; `algorithms/private/`: femur_guess_CS, GIBOC_femur_fitSphere2FemHead, Kai2014_femur_fitSphere2FemHead, Kai2014_femur_fitSpheres2Condyles, sliceFemoralCondyles, GIBOC_isolate_epiphysis, GIBOC_femur_ArticSurf, GIBOC_femur_processEpiPhysis, GIBOC_femur_getCondyleMostProxPoint, GIBOC_femur_filterCondyleSurf, GIBOC_femur_smoothCondyles |
| `tibia.py` | `algorithms/`: GIBOC_tibia, Kai2014_tibia, CS_tibia_ArtSurfCentroids, CS_tibia_Ellipse, CS_tibia_PlateauLayer; `algorithms/private/`: tibia_guess_CS, tibia_identify_lateral_direction, removeFibulaFromProxTibia, GIBOC_tibia_DistMaxSectCentre, GIBOC_tibia_DistArtSurf, GIBOC_tibia_FullProxArtSurf, GIBOC_tibia_ProxArtSurf_it1, GIBOC_tibia_ProxArtSurf_it2, GIBOC_tibia_removePartBetweenRidges_it1 |
| `common.py` | bodySide2Sign; the rest is new (see below) |

## Changes from the original

Translation changes (intended to give the same results):

1. Indices are 0-based; MATLAB `triangulation` is replaced by the `Tri` class, which provides the same
   queries.
2. Plotting, printing and the `in_mm`/`dim_fact` unit options are left out; lengths are in mm and joint
   origins are returned in mm.
3. Results are returned as a `BoneResult` (joint frames as 3x3 axes + origin) instead of MATLAB structs;
   the OpenSim-specific angle sequences (`computeXYZAngleSeq`) and landmarking (`landmarkBoneGeom`) are
   not translated.
4. `lscylinder`: the NPL LSGE code is not translated. The cylinder is fitted with the same geometric
   least-squares objective and the same starting values (from the condyle spheres), using
   scipy's Levenberg-Marquardt solver instead of LSGE's Gauss-Newton solver.
5. `fitCSA`: MATLAB Curve Fitting Toolbox fits (`gauss2`, then one Gaussian plus a line) are done with
   `scipy.optimize.curve_fit` from the same starting points and bounds.
6. `TriCurvature`: neighbourhood coordinates are centred on each vertex before the quadric fit (same
   curvatures, better conditioned), and the fits are solved together. The tangent directions used in the
   local frame are fixed rather than random (curvatures do not depend on them).
7. `quantile` uses MATLAB's definition (`method="hazen"`); `kmeans` uses scipy's `kmeans2` with a fixed
   seed (only the mean of the two cluster centres is used).
8. Slicing loops (Kai2014 femoral head and condyles) stop when a slice no longer meets the bone. When
   Kai's condyle slicing has to reverse direction, the reversed slicing's result is used as it is.
9. `pca` (Kai methods) is computed with an SVD of the centred points.

Partial-bone adaptations (new, marked `PARTIAL` in the code; not part of STAPLE). STAPLE expects whole
bones. Bones from biplanar videoradiography are usually partial (a distal femur, a proximal tibia), so
when a bone is shorter than a whole bone (femur < 300 mm, tibia < 250 mm along its principal axis), or
when `whole=False` is passed:

10. The distal-to-proximal direction is taken from where the widest cross-section is (at the condyles of a
    distal femur, at the plateau of a proximal tibia), instead of `femur_guess_CS` / `tibia_guess_CS`'s
    whole-bone tests.
11. The whole partial bone stands in for STAPLE's distal (femur) or proximal (tibia) third.
12. Femur: without the femoral head there is no medial direction (STAPLE's Y0). The medio-lateral line
    comes from the same convex-hull edges across the condyles that GIBOC uses, anterior from GIBOC's own
    test (the epiphysis centroid is anterior to the posterior condyles), and medial from the body side.
13. Tibia: without the fibula or malleoli there is no lateral direction. The medio-lateral line comes
    from the plateau outline (GIBOC's ellipse); anterior from the posterior overhang of the bone 5-30 mm
    below the plateau relative to the shaft line; lateral from the body side.
14. The mechanical axis (femur: knee centre to femoral head centre; tibia: ankle centre to knee centre)
    is replaced by the shaft axis: the line through the centroids of the shaft's cross-sections beyond
    the epiphysis. This changes the Y axis (and so X) by a few degrees in the frontal plane compared
    with a whole-bone model; the knee's flexion axis (femur Z) is unaffected. The femur's hip frame is
    not produced.

## Validation

`tests/test_staple.py` checks the translation against the joint frames in STAPLE's own reference
OpenSim models (`tests/ref_models/` in the STAPLE repository) when a clone of STAPLE is available
(environment variable `STAPLE_REPO`; reading its `.mat` bones needs the `mat-io` package):

- GIBOC-Cylinder femur, knee and hip frames: LHDL_CT, TLEM2_CT, ICL_MRI and JIA_MRI right femurs
  (`paper/`) and JIA_MRI and TLEM2_MRI left and right femurs (`bilateral/`) agree to within 0.25 degrees
  and 0.7 mm; the femoral head centre agrees exactly.
- Kai2014 tibia (ICL_MRI, `partial/example_auto2020_knee_R.osim`): exact.
- STAPLE-Talus ankle and subtalar frames (JIA_ANKLE_MRI, left and right, `partial/`): exact.

The `paper/` reference models were made with an older STAPLE version for the tibia and talus, so only
their femur frames are used. STAPLE's reference models contain no GIBOC tibia frames; those were
checked for consistency with Kai2014 and for the medial/lateral assignment on both sides.

The partial-bone adaptations were checked by cutting ten of STAPLE's whole example femurs and tibias (LHDL_CT,
TLEM2_CT left and right, ICL_MRI, JIA_MRI left and right, VAKHUM_CT, TLEM2_MRI left and right, MC22) down
to a distal femur (115 mm above the knee centre) and a proximal tibia (105 mm below it), and comparing
the knee frames with the whole-bone results (largest axis difference per bone; median, maximum):

| Method | Axes (degrees) | Origin (mm) |
|---|---|---|
| GIBOC-Cylinder femur | 2.1, 2.8 | 2.2, 4.4 |
| GIBOC-Spheres femur | 2.3, 4.0 | 2.7, 4.8 |
| GIBOC-Ellipsoids femur | 2.9, 6.0 | 0.4, 2.1 |
| Kai2014 tibia | 3.5, 11.5 | 1.0, 2.0 |
| GIBOC-Ellipse tibia | 3.5, 10.4 | 0.2, 2.9 |
| GIBOC-Centroids tibia | 3.6, 10.0 | 0.3, 3.8 |
| GIBOC-Plateau tibia | 2.7, 10.8 | 0.2, 1.6 |

None failed, and medial and lateral (so the sign of every axis) were right on every bone. The
differences are mostly the shaft axis standing in for the mechanical (or whole-bone) axis, which moves
Y and X; the knee's flexion axis (femur Z) moved least. Kai's femur method isn't offered for partial
femurs: estimating its slicing direction without the head moved it by 10-90 degrees.
