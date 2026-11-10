# BVR Motion Viewer (prototype)

A desktop app for watching bone motion tracked in **Autoscoper / SlicerAutoscoperM** from biplanar videoradiography.

Each tab is one trial. In a tab you add bones (surface files), attach each bone's own tracking file, and press play. Bones can also come from a model built in the Model Builder, with their coordinate systems and joints. Radiographs are optional: add the X-ray cameras' calibration files and images to see the radiographs behind the bones in 3D, or to look "through" an X-ray camera with the bones overlaid on the image.

## Running it

**Easiest (needs Python 3.10+ from python.org):**

- Windows: double-click `run_windows.bat`
- macOS / Linux: run `./run_mac_linux.sh` in a terminal

The first run sets up a private Python environment in a `.venv` folder (a few minutes, about 1 GB). Later runs start straight away.

**By hand:** `pip install -e .` then `python -m bvrviewer` (or `bvr-viewer`).

**In an IDE such as PyCharm:** create a virtual environment, run `pip install -e .` in it, and add a Python run configuration with the *module* `bvrviewer` and the repository folder as the working directory. Don't run `bvrviewer/__main__.py` as a script; it fails with a relative-import error. `packaging/launcher.py` can be run as a script instead.

### Windows: use Python from python.org, not Anaconda/Miniconda

Two problems come from Anaconda/Miniconda on Windows:

- **A virtual environment built from conda's Python** fails at start-up with `ImportError: DLL load failed while importing QtWidgets: The specified procedure could not be found`. Conda's `python.exe` loads its own older Microsoft C++ runtime, which PySide6 can't use. Check with `python -c "import sys; print(sys.base_prefix)"`: if it prints a `miniconda3` or `anaconda3` folder, recreate the environment from a python.org Python.
- **A conda environment from conda-forge** can fail to install with `InvalidArchiveError … viskores` (a VTK dependency), because its folders exceed Windows' path-length limit unless long paths are enabled.

The reliable route is Python 3.12 from https://www.python.org/downloads/windows/ (it installs per user, without admin rights), a virtual environment based on it, and `pip install -e .`.

**As a stand-alone app (no Python needed for users):** see [Building the app](#building-the-app).

## The Start page and tools

The app opens on a **Start** page with one box per tool. Click a box to open that tool in a new tab; you can have several tabs open, of the same or different tools. **+ New tab** (top right) and **File → New tab** list every tool, so a new tab of any kind can be opened from inside another tab. Tabs are named after their tool ("Visualiser 1", …) and can be renamed by double-clicking.

| Tool | Status | What it's for |
|---|---|---|
| **Visualiser** | available | Watch tracked bone motion in 3D, with radiographs if you have them, and joint angle curves from a joint model (below). |
| **Model Builder** | available (knee first) | Define axes on bone surfaces, optionally joints between them and reference poses, and save a model for the other tools. |
| **Assessor Evaluation** | available | Load several raters' tracking of a trial (and, for validity, the criterion tracking) and get inter- and intra-rater reliability and validity: helical and surface errors per frame, statistics, violin plots and CSV export. |

Adding a tool: describe it in `bvrviewer/app/tools.py` (it then appears on the Start page as "Coming soon"), write its tab widget, and register it in `MainWindow.TAB_CLASSES`.

## Using the Visualiser

The Visualiser tab is laid out like OpenSim. On the left, the **Navigator** lists what's in the tab under **Bones**, **X-ray cameras**, **Joints** and **Scene**. Below it, **Properties** shows the settings of whatever is selected there. The 3D view and timeline are on the right. **Right-click** any level or item in the navigator for its actions.

1. **Open a Visualiser tab** from the Start page (or File → New tab → Visualiser, Ctrl+T; any tool can be opened this way, or from "+ New tab"). Each tab starts empty.
2. **Add a bone:** right-click **Bones → Add bone…** (or File → Add bone). STL, OBJ, PLY, VTK or TIF; you can select several at once. Each bone is listed by name, which starts as its file name.
   - Or **load a model** built in the Model Builder: right-click **Bones → Load model…** (or File → Load model, Ctrl+M). This brings in all of the model's bones (from the surface files it was built on; bones already in the tab with the same surface file are reused), each with its coordinate system, and all its joints. Then load each bone's tracking for the trial.
   - **TIF partial volumes** ask for the voxel size, which TIFs don't store. Type it in, or read it from the trial's `.cfg`. The surface is built from the edge of the masked volume, which reproduces SlicerAutoscoperM's `AUT_*.stl` meshes. TIF bones can also be shown as a volume rendering (Display as: Volume).
   - **STL surfaces** are assumed to be in Autoscoper space, like SlicerAutoscoperM's `AUT_*.stl`. If a surface is in CT space instead (e.g. SlicerAutoscoperM's `Models/rad.stl`), right-click it and choose **Apply .tfm transform…** with its `*-DICOM2AUT.tfm`.
3. **Load tracking:** each bone has two rows under it, **Tracking** and **Axes**, greyed out until something is loaded and bold in colour once it is. Right-click **Tracking → Load file…**. You're asked how the file was saved from Autoscoper (matrix or x y z yaw pitch roll, row or column order, mm or cm, degrees or radians, and which bone if the file holds several), pre-filled from what the numbers suggest and checked as you change them. Tick **Use these options for this tab's other tracking files** to skip the question for the rest of the tab's files (you're still asked when a file doesn't fit them). Untracked frames (`NaN`) show in orange on the timeline.
   - **Different lengths:** Autoscoper files have no timestamps, so rows are matched to trial frames by number. When a file's frames differ from the other bones' (or from the radiographs), the dialog and the tracking's properties say so. **First row is trial frame** places a file that starts later (or earlier) in the trial.
   - Right-click a loaded row for **Load new file…** or **Refresh current file** (re-reads it with the same options, e.g. after saving over it in Autoscoper or the Model Builder), and **Tracking format and frames…** to change the options.
4. **Change how it looks:** select the bone and use **Properties**: name, colour, opacity, visible, and surface or volume display. The tick box beside each bone in the navigator shows or hides it, and F2 (or clicking a selected name) renames it.
5. **Look around:** drag to rotate, Shift-drag to pan, scroll to zoom, right-click a bone in the 3D view to select it. Use the Free 3D / XY / YZ / XZ buttons, and **Flip** to look from the other side.
6. **Play** with Space, step with ← →. Speed and Loop are on the timeline.

**What's where**

| Select… | Properties | Right-click |
|---|---|---|
| **Bones** | count, frames | Add bone, Load model, Show all, Hide all |
| a **bone** | name, colour, opacity, visible, display; hold this bone still; surface file details | load tracking, load coordinate system, hold still, show/hide, rename, replace surface file, .tfm or voxel size, remove |
| its **Tracking** | file, rows, trial frames it covers, untracked frames, format | load file / load new file, refresh current file, format and frames, hold still, remove |
| **X-ray cameras** | settings for all radiographs: image opacity, distance behind bones, bone opacity and wireframe in through-X-ray views, camera views (2D) and their overlay, brightness, contrast, invert | Add X-ray camera, Import from .cfg |
| a **camera** | name, visible; what it shows in 3D (radiograph, source marker, beam outline); images and the size they're shown at; files | view through it, show/hide, rename, change files, remove |
| its **Axes** (coordinate system) | model file and the model's bone, axes and curves on/off, motion relative to, zero at a frame, values at this frame | load file / load new file, refresh current file, show curves, show/hide axes, zero, export CSV, remove |
| **Joints** | show curves | Add joint, show/hide curves |
| a **joint** | model file, which bones in the tab it joins, axes and curves on/off, use the reference pose, zero at a frame, values at this frame | show curves, show/hide axes, reference pose, zero, export CSV, rename, change bones, remove |
| **Scene** | axes, grid, background, motion trail, untracked frames | fit bones in view |

- **Hold this bone still** (bone properties or right-click): view the other bones' motion relative to it.
- **Camera views** (view bar, View → Show/hide camera views, Ctrl+K, or X-ray cameras properties): each X-ray camera's radiograph in its own 2D panel beside the 3D view, with the tracked bones projected onto it in their colours, as a filled silhouette (like Autoscoper's overlay) or a point cloud. Scroll to zoom, drag to pan, double-click to fit. The **Through X-ray** buttons still look through a camera in the 3D view.
- Every slider has a box beside it to type the value (press Enter), and the timeline's frame number can be typed too.
- Curves are labelled **Angle X/Y/Z** and **Translation X/Y/Z** (the coordinate system's axes, coloured like the 3D axes) by default; **Labels** above the curves switches to the anatomical names (Flexion, …). CSV exports follow the same order and add the anatomical name to each column.
- **Screenshot** (Ctrl+P) and **Export video** (Ctrl+E, MP4).
- **Save tab as session** (Ctrl+S) saves file paths and every setting to a `.json`. **Open saved session** restores it.

### Coordinate systems and joints (optional)

Both come from a **model** built in the **Model Builder** (next section): one `.bvrmodel.json` holds all of a model's bones, each with its coordinate system, and its joints. Build one model with every bone and joint you need; **Load model…** (above) then brings it all in at once. They can also be added piece by piece:

- **Coordinate system on a bone:** right-click **Axes** under the bone → **Load file…** and pick the model; the model's bone that matches (by surface file) is chosen, or you're asked. It's listed under the bone, next to its tracking. Its axes are drawn on the bone (tick box), and selecting it plots the bone's motion about its own axes (rotation as a Z-X-Y sequence, and the origin's movement), relative to its first tracked frame, a chosen frame, or the lab (tracking) axes.
- **Joints:** right-click **Joints → Add joint…** and pick the model. Its joints are matched to the tab's bones (bones already using that model first, then by surface file; you're asked if that's ambiguous, and can change it later with **Change bones…**). Bones without a coordinate system get the model's. A joint gives its angles and translations, anatomical or relative to the model's **reference pose** if it has one (**Use it** in Properties). Joint values need both bones tracked.

The selected coordinate system's or joint's curves are plotted under the 3D view with a cursor at the current frame (click a plot to jump there). **Zero at this frame** also subtracts the values at a frame of this trial. **Export CSV** writes every frame. All of this is saved with the session.

The axes only fit a bone loaded from the **same surface file in the same model space** as in the model, so use the same STL/TIF (and the same voxel size or .tfm) in both tabs.

**Terms.** A **bone** is one surface (plus its tracking and, optionally, its coordinate system). A **model** is what the Model Builder saves: bones with coordinate systems, joints between them, and reference poses.

### Radiographs (optional)

Right-click **X-ray cameras → Add X-ray camera…** (or use the Radiographs menu) and pick:

1. the calibration file: MayaCam 2.0 text (as in SlicerAutoscoperM's sample data) or MayaCam 1.x (5 lines of numbers, e.g. XMALab's `*_MayaCam.csv` exports);
2. the radiographs: a folder, selected files, or one multi-page TIFF. Images are ordered by the last number in each file name, so any naming works;
3. the first image is shown at tracking frame 1, the second at frame 2, and so on; the dialog says if the number of images and tracked frames differ.

The dialog previews your tracked bones on that image. If they sit on their shadows, everything is matched up. Or use **Import cameras from .cfg…** to add every camera listed in an Autoscoper trial file at once.

With cameras added, the radiographs appear on detector planes behind the bones. The planes are see-through from behind, so they don't hide the bones. The **Through X-ray** buttons (or double-clicking a camera in the navigator) look from each X-ray source with the bones overlaid on the image.

When a camera is added, all its radiographs are read into memory first, with a progress bar showing each camera's images, the read speed and the time left, so that playing and jumping between frames need no more reading. This matters most for large images on a network drive (500 uncompressed 2016 × 2016 frames per camera is about 2 GB to fetch). **Skip** starts straight away instead, and images not yet read are then read as each frame is shown. Radiographs are shown at up to 1024 pixels across, and smaller still if a whole sequence wouldn't fit in 1.2 GB per camera (the camera's properties show the size used).

## The Model Builder

A model is built in three steps; the navigator is laid out the same way (**1 Bones**, **2 Joints**, **3 Reference pose** under each joint), with **Properties** for the selected item. Right-click for actions. **Start a knee** (on the model's properties) adds a femur, a tibia and a Grood & Suntay knee in one go; **Start an ankle** adds a distal tibia, a talus, a calcaneus and ISB-style talocrural and subtalar joints.

**1. Bones: axes on each bone.** Right-click **Bones → Add bone** and choose its type: **Femur** or **Tibia** (knee), the ankle and foot bones (**Tibia (distal: ankle)**, fibula, talus, calcaneus, navicular, cuboid, cuneiforms, metatarsals), or **Other bone**. The type decides how its axes can be made.
- Load its surface (right-click → **Load surface…**): the same file, in the same model space, that you track with. Set its **side**.
- **Axes from**: an automatic method fitted to the surface (the default; see below), or **landmarks** placed by hand: select a landmark under the bone, then **right-click the surface** (point), or right-click and drag to paint a region (its centre, or a sphere fitted to it). Left-drag still rotates.
- Anatomical bones use ISB axis names: X anterior, Y superior, Z right on both sides (Wu et al. 2002). "Other bone" uses its principal axes of inertia (axis 1 = long axis), with no anatomical names.

A bone's axes live in its own model space, so they don't depend on where any other bone is. Saved on their own, bones already give each bone's motion about its own axes in the Visualiser (right-click **Axes** under a bone → **Load file…**, or **Load model…**).

**2. Joints (optional).** Right-click **Joints → Add joint…**, choose a definition and its proximal and distal bones:
- **Knee (tibiofemoral): Grood & Suntay**, femur → tibia. ISB doesn't define the knee; this is the convention most BVR knee studies use. Left knees are mirrored, so positive values mean the same motion on both sides: flexion, adduction (varus), internal rotation; lateral, anterior and proximal translation of the tibia.
- **Ankle (talocrural): ISB**, tibia → talus, **Subtalar: ISB style**, talus → calcaneus, and **Ankle complex: ISB**, tibia → calcaneus (Wu et al. 2002): dorsiflexion about the proximal bone's Z, internal rotation about the distal bone's Y, inversion about the floating axis; left sides mirrored like the knee.
- **Joint coordinate system (ISB style)** between any two bones (proximal Z, floating axis, distal Y), without clinical names.
- **Cardan X-Y-Z** between any two bones, e.g. two carpals with inertial axes.

With no reference pose, a joint's zero is **anatomical**: all angles are 0 when the two bones' axes line up. No neutral pose is needed for that, so the overlapping model spaces you see when surfaces are loaded don't matter. Tracking places each bone at every frame, and the angles come from the axes' relative orientation.

**3. Reference pose (optional, per joint).** Select **Reference pose** under the joint, then:
- **From a neutral trial…**: the .tra files of a static/neutral trial for the two bones (the bones' preview tracking is offered), averaged over every frame both are tracked, or over a range of frames.
- **From the scan pose…**: where the bones were in the CT/MRI. For SlicerAutoscoperM surfaces, pick each bone's `*-DICOM2AUT.tfm` (found automatically when it sits in the usual `Transforms` folder). If both surfaces are already in the scan's space, choose that option instead.
- **Normalise by**: *subtract the reference values* (everything is 0 in the reference pose), or *realign the distal axes to the proximal axes in the reference pose* (rotations are exactly 0 there; translations are subtracted).

The reference is stored as the two bones' relative pose, so it stays valid if a bone's axes are changed later. The view shows the joint in its reference pose, and Properties lists the anatomical values that are being removed.

**Preview and checks.** Attach a tracking file to a bone (right-click → **Preview tracking**) to see joints in a tracked pose with their values, anatomical and relative to the reference. **Compare methods** lists the joint's anatomical angles for every combination of the bones' axis methods. **Save model…** writes a `.bvrmodel.json`: bones (surface, axes method, landmarks, axes), joints and reference poses. Open it again here to edit it. In a Visualiser tab, **Load model…** brings in its bones, coordinate systems and joints together. Version-1 files (one template) are converted on opening.

### Automatic axes (no landmarks)

Each bone's axes can come from an automatic method **or** from landmarks you place. Select the bone and choose **Axes from** in Properties (or right-click it).

**Femur and tibia: STAPLE.** The knee methods are those compared by Modenese & Renault (2021) in STAPLE, run through a Python translation of STAPLE's own MATLAB code (`bvrviewer/core/staple/`, see the licence note below). Each finds the articular surfaces from the bone's shape and fits a primitive to them:

| Bone | Method | Z (pointing right) and origin | Source |
|---|---|---|---|
| Femur | **GIBOC-Cylinder** (default) | axis of a cylinder fitted to both posterior condyle surfaces; origin midway between the condyles along it | Renault et al. 2018 (most repeatable femur method there; STAPLE's default) |
| Femur | GIBOC-Spheres | line joining spheres fitted to each posterior condyle surface; origin midway | Renault et al. 2018 (closest to manual axes in Modenese & Renault 2021) |
| Femur | GIBOC-Ellipsoids | line joining ellipsoids fitted to each whole condyle surface | Renault et al. 2018 |
| Femur | Kai et al. 2014 | line joining spheres fitted to slices of the condyles; **whole femurs only** | Kai et al. 2014 |
| Tibia | **Kai et al. 2014** (default) | major axis of an ellipse fitted to the tibia's largest cross-section; origin at its centre | Kai et al. 2014 (STAPLE's default; most robust on poorer surfaces) |
| Tibia | GIBOC-Ellipse | major axis of an ellipse fitted to the outline of the two plateau surfaces | Renault et al. 2018 |
| Tibia | GIBOC-Centroids | line joining the centroids of the medial and lateral plateau surfaces | Renault et al. 2018 |
| Tibia | GIBOC-Plateau | principal axis of a layer of bone just below the plateau | Renault et al. 2018 (PIAASL, the most repeatable tibia variant there) |

On a whole femur, Y points from the knee centre to the femoral head centre; on a whole tibia, from the ankle centre to the knee centre (the mechanical axes), and X = Y × Z.

**Partial bones.** Autoscoper surfaces usually stop part way up the femur and down the tibia, and STAPLE expects whole bones. A bone shorter than a whole one (femur < 300 mm, tibia < 250 mm) is processed with documented adaptations (marked `PARTIAL` in the code, listed in `bvrviewer/core/staple/NOTICE.md`): Y follows the **shaft axis** (the line through the centres of the shaft's cross-sections) instead of the mechanical axis; on the femur, medial and lateral come from the condyles and the body side instead of the femoral head; on the tibia, anterior comes from the shape of the shaft just below the tuberosity (its cross-section is skewed towards the anterior crest) and lateral from the side. Set each bone's **side** correctly. Tested by cutting ten of STAPLE's example femurs and tibias down to partial bones, the knee axes stayed within a median of 2–3° of the whole-bone results (femur at most 6°; tibia at most about 11°, mostly the shaft axis standing in for the mechanical axis), with medial and lateral right on every bone. **Flip anterior/posterior** and **Flip superior/inferior** correct a partial bone where a guess fails. Properties shows the fit quality and what was changed for a partial bone; the view shows the articular surfaces found (orange, pink) and the fitted shapes (purple).

**Talus: STAPLE-Talus.** *Talocrural axis*: Z = axis of a cylinder fitted to the talar trochlea (dome), X along the talus's long axis, Y superior. *Subtalar axis*: X joins the centres of spheres fitted to the talocalcaneal (posterior facet) and talonavicular surfaces, origin at the talocalcaneal centre. The talus is usually imaged whole, so no adaptation is needed.

**Foot and ankle bones: AAFACT.** AAFACT (Peterson, Kruger & Lenz 2023) gives axes for 14 foot and ankle bones (tibia, fibula, talus, calcaneus, navicular, cuboid, cuneiforms, metatarsals) by aligning each bone to a template and using regional centroids; several bones have more than one system (talus: talonavicular, tibiotalar or subtalar; calcaneus: calcaneocuboid or subtalar; cuboid, lateral cuneiform and metatarsals: vertical or radial). Its licence doesn't allow modified copies, so it isn't included: install the compiled AAFACT (with the MATLAB Runtime) from https://github.com/Lenz-Lab/AAFACT. Then, with **Axes from: AAFACT** selected, **Run AAFACT on this bone** (point to the program once; it's remembered) or **Import results (.xlsx)** from an AAFACT run of your own, and choose the coordinate system. AAFACT's AP, SI and ML axes become X, Y and Z.

**Licence of the STAPLE translation.** `bvrviewer/core/staple/` is translated from STAPLE (https://github.com/modenaxe/msk-STAPLE, Copyright (c) 2020-2021 Luca Modenese and Jean-Baptiste Renault), licensed **CC BY-NC 4.0: non-commercial use only**; the rest of the viewer doesn't depend on it. Its `NOTICE.md` lists what was translated, every change, and the validation: on STAPLE's example bones the translation reproduces the joint frames in STAPLE's own reference models (femur within 0.25° and 0.7 mm; Kai tibia and talus exactly). Please cite Modenese & Renault (2021) J Biomech 116:110186, and Renault et al. (2018) J Biomech 80:171–178 or Kai et al. (2014) J Biomech 47:1229–1233 for the method used.

Models saved with the earlier, simplified automatic methods still open; those methods are listed as "older" for those bones only.

**Compare methods** (select the joint, with preview tracking on both bones) lists the joint angles at the preview frame for every combination of the bones' methods, including landmarks if placed. Different definitions give different offsets (a few degrees between the STAPLE methods, more against landmarks), so report the method you used, and use a reference pose when comparing across definitions.

Partial bones and landmarks: where a landmark definition needs the hip centre, the malleoli or the tibial condyles that aren't in the surface, paint a band around the shaft end instead — the shaft centre gives a long axis that follows the bone. Report that substitution with your results.

**Planned:** multi-segment foot joint definitions, the hip and upper limb (ISB), and your own joint definitions.

## Assessor Evaluation

How reliable is each assessor's tracking, and how close is it to a reference (e.g. bead-based tracking)? List what was tracked, fill in the tracking files, and the results update as you go.

1. **What was tracked:** list the factors.
   - **Bones**: **+** adds a bone label (e.g. `tib`). Each bone can also have a **surface**, adding the **surface distance** (**Surfaces…** for STL, OBJ, PLY or VTK; **From .cfg…** for the trial's TIF partial volumes and voxel sizes), and **axes**: **Axes…** takes the bones' coordinate systems from a Model Builder model (matched by label or surface file), adding errors about each bone's own axes. Surfaces and axes must be in the space the tracking maps (Autoscoper space for SlicerAutoscoperM).
   - **Assessors**, **Sessions** (repeats by the same assessor, for intra-rater reliability) and **Trials** (if more than one): short lists; **+** adds, double-click renames, right-click to move or remove.
   - **Reference** (optional, at most one): the criterion tracking (e.g. bead-based), for validity. Add it with **+** like an assessor and give it a name (e.g. *Beads*); its name labels its row and the results.
2. **Tracking files:** every combination becomes a cell to fill: a grid with one row per assessor × session (and a Reference row) and one column per bone, for each trial (pick the trial above the grid). Fill the gaps:
   - double-click a cell, or drop one file on it, to load its tracking file (double-click a filled cell to change its format or frames; right-click to load a new file, refresh it, or clear cells);
   - The format is asked for the first file (as in the Visualiser); tick the box to use the same options for the tab's other files.
   - Cells can stay empty. Every comparison the factors call for is still listed in the results, as **N/A** with the reason (e.g. *no file for Jo · session 2*), so an assessor who skipped a bone or a second session simply has N/A for those.
3. **Analysis:**
   - **Compare:** assessors vs reference (validity), between assessors (inter-rater: every pair, in the chosen session) or between sessions (intra-rater: every pair of each assessor's sessions).
   - **Measures**, per frame, from the difference transform inv(T_reference)·T_test (in the bone's frame, as Welte et al. 2022):
     - *helical rotation* (default): the rotation about the helical axis, in degrees;
     - *helical translation* (default): the translation along the helical axis, as reported by Welte et al. 2022 and Williams et al. 2026. It is **not** the bone's displacement: with any rotation error, translation across the axis moves the axis instead, so it can be near 0 while the bone is millimetres away;
     - *mean surface distance* (default when surfaces are loaded) and *maximum surface distance*: how far the bone's surface is from where it should be; *centroid displacement*;
     - *bone axes* (with a model): rotation about Z, X, Y (Z-X-Y) and translation of the axes' origin along X, Y, Z, signed.
   - **Thresholds:** 2° and 1 mm by default (Welte et al. 2022).
   - **Statistics:** for each trial and bone separately, or pooled over trials, or over trials and bones.
4. **Results:**
   - **Summary:** one row per comparison and measure, N/A where a file is missing (the Note says which). Only frames tracked in **both** files are compared (rows of NaN are untracked, e.g. when only keyframes were tracked): **Frames** is how many were compared, **Tracked** how many each file has, and the Note says when they have few in common. Then mean, SD, median, IQR, RMS, max, Bland–Altman limits of agreement (mean ± 1.96 SD, with the half-width of their 95% CI), robust limits (median ± 1.45 IQR, Williams et al. 2026), median absolute deviation, Hampel outliers and the share of frames over the threshold. Frames are treated as independent, which they aren't, so the CIs are optimistic.
   - **Distributions:** a violin per rater (or pair) for each bone and measure, with every frame as a point, median, IQR, mean and threshold (as Welte et al. 2022, figure 4).
   - **Over frames:** the error at every frame for each rater, with frames above the threshold circled (as Welte et al. 2022, figure 5), to see where and how often tracking went wrong.
   - Figures use one style (bold Arial, 9 pt, inward ticks, no top/right spines, grid; saved at 300 dpi with embedded TrueType fonts in PDFs). Where Arial isn't installed, Liberation Sans or DejaVu Sans stands in.
   - **Limit plots to 3 × threshold** hides failed frames so the rest can be seen (the number hidden is shown).
   - Very large, steady differences are flagged: they usually mean the files map different bone spaces (e.g. a criterion in CT space), different rows or a wrong format.
5. **Export:** summary statistics and per-frame values (long format: trial, bone, comparison, test, reference, frame, measure, value, unit) as CSV, and the figures as PNG, SVG or PDF. **Save…** keeps everything in a `.bvrassess.json` project.

## How the files are interpreted

These conventions were checked against the SlicerAutoscoperM sample data (by projecting tracked bones onto the real radiographs) and against the Autoscoper source code. The tests check them.

- **Tracking:** pose at frame *f* = the .tra row as a 4×4 matrix × the model in Autoscoper volume space. xyzypr uses R = Rz(yaw)·Ry(pitch)·Rx(roll).
- **TIF → Autoscoper space:** a voxel at column *i*, row *j*, slice *k* sits at x = *i*·dx, y = (rows − *j*)·dy, z = −*k*·dz.
- **.tfm files** (ITK) store the resampling direction, so they're inverted before use.
- **Calibration (MayaCam 2.0):** x_cam = R·X + t, pixel = K·x_cam / z. **MayaCam 1.x** (camera position; rotations about the camera's x, y, z axes applied z, y, x; film plane; u0, v0, Z; scale, width, height) is converted to the same K, R, t following Autoscoper's own reader (`Camera::loadMayaCam1`): 1-based u0, v0, camera looking along its local −z, image rows counted from the top.

## Testing

```
pip install -e ".[dev]"
pytest                                   # synthetic tests
BVR_SAMPLE_DATA=/path/to/sample_data pytest   # plus the real-data tests
STAPLE_REPO=/path/to/msk-STAPLE pytest tests/test_staple.py   # the STAPLE translation vs STAPLE's reference models
```

`BVR_SAMPLE_DATA` is a folder holding the three unzipped SlicerAutoscoperM sample datasets (`2025-02-19-Wrist`, `2025-01-12-Ankle`, `Knee_SampleData`). Download them from https://github.com/BrownBiomechanics/Autoscoper/releases/tag/sample-data.

`tests/drive_gui.py` drives the real window on the wrist and ankle data and saves screenshots. `tests/drive_assessor.py` simulates three raters (two sessions each) from the ankle sample's tracking, fills the grid (leaving one gap), and checks the Assessor Evaluation tab in every mode, including the N/A results, exports and the project file. `tests/test_calibration_mayacam1.py` checks MayaCam 1.x against an independent re-implementation of Autoscoper's camera model. `tests/drive_requests.py` checks the features from `feature_requests/REQUESTS_VISUALISER.md` (tab menus, typed slider values, curve labels, the tracking prompt, Tracking/Axes rows, frame offsets, camera views) on the wrist data. `tests/drive_model_builder.py` builds a knee model (bones with automatic and landmark axes, the joint, the method comparison and reference poses, then loading it in the Visualiser) on the knee sample by clicking and painting in the viewport, then checks the joint angles in a Visualiser tab against a known flexion sweep. `python -m bvrviewer --selftest SAMPLE_DIR out.png` checks that a build can load and render the wrist sample.

## Building the app

```
pip install -e ".[dev]"
cd packaging
pyinstaller bvr_viewer.spec --noconfirm
```

This makes `packaging/dist/BVR Motion Viewer/` (Windows, Linux) or `BVR Motion Viewer.app` (macOS). Build on each operating system you want to support. Pushing a tag such as `v0.1.0` to GitHub runs `.github/workflows/build.yml`, which builds all three. Unsigned builds show a Windows SmartScreen or macOS Gatekeeper warning the first time they open.

The Linux build is about 1.1 GB unzipped, most of it VTK.

## Known limitations

- Calibration: MayaCam 2.0 and 1.x. VTK camera (.json) files are detected and reported as unsupported.
- `VolumeFlip` values other than 0 0 0 are applied but haven't been tested against real data.
- Units in .tra files can't be detected; they default to Autoscoper's mm and degrees.
- The through-X-ray view frames the image approximately (bones and image always line up exactly).
