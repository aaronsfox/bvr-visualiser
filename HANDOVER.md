# BVR Motion Viewer — handover (27 Oct 2026)

For continuing development in a new chat. Repository: https://github.com/aaronsfox/bvr-visualiser
(branch `main`, last commit `4384e47`, working tree clean). Project docs in the claude.ai project
"BVR Visualiser Tool": `claude/roadmap.md`, `claude/automatic-axes-implementation.md`,
`claude/joint-definitions-review.md`, `claude/autoscoper-data-conventions.md`.

## 1. What the app is

A PySide6 + PyVista desktop app to view and analyse bone motion tracked in Autoscoper /
SlicerAutoscoperM (biplanar videoradiography). It opens on a Start page with one card per tool, and each tool opens in its own tab:

| Tool | Status |
|---|---|
| **Visualiser** | Built. Loads bones (STL/OBJ/PLY/VTK or TIF partial volumes) with .tra tracking. Optional radiographs. Each bone can carry a coordinate system, and joints can be added. *Load model…* brings in a whole model. Shows curves and exports CSV, sessions, screenshots and MP4. |
| **Model Builder** | Built. Three steps: **1 Bones** (axes) → **2 Joints** (optional) → **3 Reference pose** (optional). Saves `.bvrmodel.json`. |
| Reliability Analysis | Planned (card only). |
| Validity Analysis | Planned (card only). |

Terminology agreed with Aaron:
- **Bone** = one surface (plus its tracking and coordinate system).
- **Model** = only what the Model Builder saves.
- Internally the classes are still `BoneModel` (Visualiser bone) and `KinematicModel` / `Segment` (Model Builder).

## 2. Work in this session (27 Oct 2026)

1. **Terminology and *Load model…*** in the Visualiser (commit `52a61ce`).
2. **Review of the 10 papers in `resources/`.** It covers ISB (Wu 2002/2005), Kai 2014, Renault 2018 (GIBOC), Modenese & Renault 2021 (STAPLE), Carson, Leardini, Maharaj, Peterson 2023 (AAFACT) and Behling 2024. Saved as `claude/joint-definitions-review.md`.
3. **STAPLE translated to Python**, as Aaron decided ("port it faithfully"). It lives in `bvrviewer/core/staple/`:
   - **Algorithms:** GIBOC femur (cylinder, spheres, ellipsoids), Kai 2014 femur (whole femurs only), Kai 2014 tibia, GIBOC tibia (ellipse, centroids, plateau), and STAPLE-Talus (talocrural cylinder, subtalar spheres).
   - **Licence:** CC BY-NC 4.0, **non-commercial**. The folder holds `LICENSE_STAPLE.txt`, the MIT/BSD notices in `third_party_licenses/`, and `NOTICE.md` (credits, citations, file mapping, every change, validation).
   - **Validated against STAPLE's own reference models:** the femur agrees within 0.25° / 0.7 mm; Kai tibia and talus are exact. Reaching exact agreement required matching MATLAB's `vertexNormal`, which is the unweighted mean of face normals.
   - **Partial-bone adaptations** for BVR bones (marked `PARTIAL` in the code):
     - Y follows the shaft axis instead of the mechanical axis.
     - Femur: medial/lateral comes from the condyles plus the body side.
     - Tibia: anterior comes from the cross-section skewness 35–55 mm below the plateau, then lateral from the side.
     - Tested by cutting 10 STAPLE example bones down to partial bones: median 2–3.6° from the whole-bone results, worst case 6° (femur) and ~11° (tibia), no failures, medial/lateral always right.
4. **AAFACT runs externally**, as Aaron decided; nothing from AAFACT is copied into the repo. `bvrviewer/core/aafact.py` either runs the user's compiled AAFACT on one bone or imports its `CoordinateSystem_*.xlsx`. AAFACT's AP/SI/ML axes become X/Y/Z, and per-bone options are offered (TN/TT/ST, CC/ST, V/R).
5. **Model Builder additions:**
   - STAPLE methods as the femur/tibia/talus options. Defaults are GIBOC-Cylinder for the femur and Kai for the tibia, as in STAPLE. The older simplified methods are kept only so old files still open.
   - New bone types: distal tibia (ISB malleoli landmarks MM, LM, TP), fibula, talus, calcaneus, navicular, cuboid, the 3 cuneiforms and metatarsals 1–5.
   - New ISB joints: ankle (talocrural), subtalar and ankle complex.
   - "Start an ankle" preset and an AAFACT panel (program path stored in QSettings).

## 3. Code map (key files)

| File | Contents |
|---|---|
| `bvrviewer/core/joints.py` | `KinematicModel`: segment/joint types, landmark recipes, joint conventions (G&S knee, ISB ankle, JCS ZXY, Cardan XYZ), kinematics, references, AAFACT results on a bone (`Segment.imported`), file v2 |
| `bvrviewer/core/auto_axes.py` | `AUTO_METHODS` registry. Its `staple_*` entries wrap `core/staple`; older own methods are flagged `legacy=True` |
| `bvrviewer/core/staple/` | `tri.py` (mesh ops), `fitting.py`, `femur.py`, `tibia.py`, `talus.py`, `common.py` (`BoneResult`, partial-bone helpers, `tri_from_pyvista` which decimates to 40k faces) |
| `bvrviewer/core/aafact.py` | `read_results()`, `run()`, `BONES`, `OPTIONS` |
| `bvrviewer/app/model_builder.py` | Model Builder tab: navigator, properties, fits drawing, AAFACT panel, reference dialog |
| `bvrviewer/app/tab.py` | Visualiser tab |
| `bvrviewer/app/joint_panel.py` | Joint curves and bone matching |
| `bvrviewer/core/session.py`, `models.py`, `tracking.py`, `transforms.py` | Sessions, bones, .tra, .tfm |

Conventions:
- **Axes:** ISB, with X anterior, Y superior and Z right on both sides. Left-side angles are mirrored through the per-convention `signs`.
- **Poses:** pose = .tra row × model space. A bone's scan pose is the inverse of its `*-DICOM2AUT.tfm`.
- **Reference:** stored as G0 = inv(T_prox)·T_dist, with a "subtract" or "realign" mode.

## 4. Testing

```
pip install -e ".[dev]"            # dev extras include pytest, ruff, pyinstaller, mat-io
pytest                              # synthetic tests (~2 s)
BVR_SAMPLE_DATA=/path/to/sample_data pytest            # + Autoscoper knee/ankle/wrist samples (~4 min)
STAPLE_REPO=/path/to/msk-STAPLE pytest tests/test_staple.py   # translation vs STAPLE reference models
BVR_SAMPLE_DATA=... xvfb-run -a python3 tests/drive_model_builder.py OUT_DIR   # GUI: knee, Visualiser, ankle
```

- **Sample data:** the SlicerAutoscoperM sample-data release, with `Knee_SampleData`, `2025-01-12-Ankle` and `2025-02-19-Wrist` in one folder.
- **STAPLE repo:** https://github.com/modenaxe/msk-STAPLE. Its `.mat` bones are read with `mat-io`.
- **Last full run:** all passing (unit and sample tests, STAPLE reference checks, and the GUI driver with "MODEL BUILDER OK", "VISUALISER OK" and "ANKLE OK").
- **Slow tests:** STAPLE methods on the dense sample meshes take ~3–10 s each, after a one-off ~6–9 s decimation.

Commit conventions used:
- Commit straight to `main`.
- Messages end with `Co-Authored-By` and `Claude-Session` lines.
- Push with `git push origin main`.

## 5. Current status / known limitations

- **AAFACT has never been run for real.** It was only tested with generated .xlsx files and a stand-in program. Its xlsx layout was taken from `aafact_cli.m`.
- **Distal tibia (ankle work):** the STAPLE tibia methods are knee-only. A distal tibia needs the malleoli landmarks or AAFACT.
- **Partial tibia accuracy:** the shaft axis stands in for the mechanical axis, which can move Y/X by up to ~11°. Report this, or use a reference pose.
- **GIBOC tibia** has no STAPLE reference output to validate against. It was checked only against Kai and for medial/lateral being right on both sides.
- **Running the STAPLE methods freezes the UI** for a few seconds per bone (no worker thread).
- **Kai femur** is refused for partial femurs.
- **Packaging:** `packaging/bvr_viewer.spec` bundles the STAPLE licence files and openpyxl. A real PyInstaller build hasn't been re-tested since these changes.
- **Ruff:** the repo has ~120 pre-existing lint warnings, and ruff is not enforced.
- **Licence:** the repo has no top-level LICENSE file yet. Any release that includes `core/staple/` must stay non-commercial.

## 6. Suggested next actions (in rough priority)

1. **Aaron to try on real data:**
   - Run AAFACT from the Model Builder on a foot bone (Windows, MATLAB Runtime) and confirm the import. If the sheet layout differs, adjust `aafact.read_results`.
   - Check the STAPLE axes on several of his own partial femurs/tibias: anterior and medial direction, and how close they are to landmark axes.
2. **Choose a repo licence** compatible with the CC BY-NC STAPLE module (or package that module as an optional extra).
3. **Run STAPLE methods in a worker thread**, with progress, so the UI doesn't freeze.
4. **Remaining review items (`joint-definitions-review.md` §Implications):**
   - New axes sources: *aligned to another bone in a reference pose* (ISB calcaneus, carpals) and *lab-aligned at a frame* (Maharaj 2022, Y-Z-X).
   - Behling 2024 talus morphological system (sphere/cylinder fits already exist in `core/staple`).
   - STAPLE foot (calcaneus) and pelvis, and the patella.
   - Hip and upper limb (Wu 2002/2005), Euler sequences (Y-X-Y), and helical axes in the Visualiser.
5. **User-defined joints**, and importing landmarks from 3D Slicer (.mrk.json/.fcsv).
6. **Reliability and Validity tools.** The specification is in `roadmap.md`.

## 7. Open questions for Aaron

- Does AAFACT run and import correctly on his machine?
- Which axis method and normalisation (anatomical, neutral trial, scan pose) should his studies standardise on?
- Methods still needed from Welte et al. (PMID 35635454), and the talus/calcaneus definitions and Euler sequence from Maharaj et al. 2020.
- Which criterion data formats (e.g. XMALab) and reliability statistics does he prefer?
