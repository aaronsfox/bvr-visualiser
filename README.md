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

The app opens on a **Start** page with one box per tool. Click a box to open that tool in a new tab; you can have several tabs open, of the same or different tools. **+ New tab** (top right) brings you back to the Start page. Tabs are named after their tool ("Visualiser 1", …) and can be renamed by double-clicking.

| Tool | Status | What it's for |
|---|---|---|
| **Visualiser** | available | Watch tracked bone motion in 3D, with radiographs if you have them, and joint angle curves from a joint model (below). |
| **Model Builder** | available (knee first) | Define axes on bone surfaces, optionally joints between them and reference poses, and save a model for the other tools. |
| **Reliability Analysis** | coming soon | Load bones and repeated tracking of the same trial, by the same tester or different testers, and get accuracy and reliability metrics across them. |
| **Validity Analysis** | coming soon | Load bones, criterion tracking data and testers' tracking, and get accuracy against the criterion. |

Adding a tool: describe it in `bvrviewer/app/tools.py` (it then appears on the Start page as "Coming soon"), write its tab widget, and register it in `MainWindow.TAB_CLASSES`.

## Using the Visualiser

The Visualiser tab is laid out like OpenSim. On the left, the **Navigator** lists what's in the tab under **Bones**, **X-ray cameras**, **Joints** and **Scene**. Below it, **Properties** shows the settings of whatever is selected there. The 3D view and timeline are on the right. **Right-click** any level or item in the navigator for its actions.

1. **Open a Visualiser tab** from the Start page (or File → New Visualiser tab, Ctrl+T). Each tab starts empty.
2. **Add a bone:** right-click **Bones → Add bone…** (or File → Add bone). STL, OBJ, PLY, VTK or TIF; you can select several at once. Each bone is listed by name, which starts as its file name.
   - Or **load a model** built in the Model Builder: right-click **Bones → Load model…** (or File → Load model, Ctrl+M). This brings in all of the model's bones (from the surface files it was built on; bones already in the tab with the same surface file are reused), each with its coordinate system, and all its joints. Then attach each bone's tracking for the trial.
   - **TIF partial volumes** ask for the voxel size, which TIFs don't store. Type it in, or read it from the trial's `.cfg`. The surface is built from the edge of the masked volume, which reproduces SlicerAutoscoperM's `AUT_*.stl` meshes. TIF bones can also be shown as a volume rendering (Display as: Volume).
   - **STL surfaces** are assumed to be in Autoscoper space, like SlicerAutoscoperM's `AUT_*.stl`. If a surface is in CT space instead (e.g. SlicerAutoscoperM's `Models/rad.stl`), right-click it and choose **Apply .tfm transform…** with its `*-DICOM2AUT.tfm`.
3. **Attach tracking:** right-click the bone → **Attach tracking…**. The tracking file then shows under the bone. The format is detected: matrix or x y z yaw pitch roll, row or column order, comma or space, and several bones in one file (you'll be asked which). Untracked frames (`NaN`) show in orange on the timeline. If something looks wrong, right-click → **Tracking format…** to change the units (mm/cm, degrees/radians) or the order.
4. **Change how it looks:** select the bone and use **Properties**: name, colour, opacity, visible, and surface or volume display. The tick box beside each bone in the navigator shows or hides it, and F2 (or clicking a selected name) renames it.
5. **Look around:** drag to rotate, Shift-drag to pan, scroll to zoom, right-click a bone in the 3D view to select it. Use the Free 3D / XY / YZ / XZ buttons, and **Flip** to look from the other side.
6. **Play** with Space, step with ← →. Speed and Loop are on the timeline.

**What's where**

| Select… | Properties | Right-click |
|---|---|---|
| **Bones** | count, frames | Add bone, Load model, Show all, Hide all |
| a **bone** | name, colour, opacity, visible, display; hold this bone still; surface file details | attach/replace/detach tracking, tracking format, add/remove coordinate system, hold still, show/hide, rename, replace surface file, .tfm or voxel size, remove |
| its **tracking** | file, frames, untracked frames, detected format | attach/replace, format, detach |
| **X-ray cameras** | settings for all radiographs: image opacity, distance behind bones, bone opacity and wireframe in through-X-ray views, brightness, contrast, invert | Add X-ray camera, Import from .cfg |
| a **camera** | name, visible; what it shows in 3D (radiograph, source marker, beam outline); which image is frame 1; files | view through it, show/hide, rename, change files, remove |
| its **coordinate system** | model file and the model's bone, axes and curves on/off, motion relative to, zero at a frame, values at this frame | add/replace, show curves, show/hide axes, zero, export CSV, remove |
| **Joints** | show curves | Add joint, show/hide curves |
| a **joint** | model file, which bones in the tab it joins, axes and curves on/off, use the reference pose, zero at a frame, values at this frame | show curves, show/hide axes, reference pose, zero, export CSV, rename, change bones, remove |
| **Scene** | axes, grid, background, motion trail, untracked frames | fit bones in view |

- **Hold this bone still** (bone properties or right-click): view the other bones' motion relative to it.
- **Screenshot** (Ctrl+P) and **Export video** (Ctrl+E, MP4).
- **Save tab as session** (Ctrl+S) saves file paths and every setting to a `.json`. **Open saved session** restores it.

### Coordinate systems and joints (optional)

Both come from a **model** built in the **Model Builder** (next section): one `.bvrmodel.json` holds all of a model's bones, each with its coordinate system, and its joints. Build one model with every bone and joint you need; **Load model…** (above) then brings it all in at once. They can also be added piece by piece:

- **Coordinate system on a bone:** right-click the bone → **Add coordinate system…** and pick the model; the model's bone that matches (by surface file) is chosen, or you're asked. It's listed under the bone, next to its tracking. Its axes are drawn on the bone (tick box), and selecting it plots the bone's motion about its own axes (rotation as a Z-X-Y sequence, and the origin's movement), relative to its first tracked frame, a chosen frame, or the lab (tracking) axes.
- **Joints:** right-click **Joints → Add joint…** and pick the model. Its joints are matched to the tab's bones (bones already using that model first, then by surface file; you're asked if that's ambiguous, and can change it later with **Change bones…**). Bones without a coordinate system get the model's. A joint gives its angles and translations, anatomical or relative to the model's **reference pose** if it has one (**Use it** in Properties). Joint values need both bones tracked.

The selected coordinate system's or joint's curves are plotted under the 3D view with a cursor at the current frame (click a plot to jump there). **Zero at this frame** also subtracts the values at a frame of this trial. **Export CSV** writes every frame. All of this is saved with the session.

The axes only fit a bone loaded from the **same surface file in the same model space** as in the model, so use the same STL/TIF (and the same voxel size or .tfm) in both tabs.

**Terms.** A **bone** is one surface (plus its tracking and, optionally, its coordinate system). A **model** is what the Model Builder saves: bones with coordinate systems, joints between them, and reference poses.

### Radiographs (optional)

Right-click **X-ray cameras → Add X-ray camera…** (or use the Radiographs menu) and pick:

1. the calibration file (MayaCam 2.0 text, as in SlicerAutoscoperM's sample data);
2. the radiographs: a folder, selected files, or one multi-page TIFF. Images are ordered by the last number in each file name, so any naming works;
3. which image is tracking frame 1 (also changeable later in the camera's properties).

The dialog previews your tracked bones on that image. If they sit on their shadows, everything is matched up. Or use **Import cameras from .cfg…** to add every camera listed in an Autoscoper trial file at once.

With cameras added, the radiographs appear on detector planes behind the bones. The planes are see-through from behind, so they don't hide the bones. The **Through X-ray** buttons (or double-clicking a camera in the navigator) look from each X-ray source with the bones overlaid on the image.

Radiographs are decoded in the background as soon as a camera is added, so playback may be slower for the first few seconds of a long trial. They're shown at up to 1024 pixels across, and hidden cameras aren't decoded at all.

## The Model Builder

A model is built in three steps; the navigator is laid out the same way (**1 Bones**, **2 Joints**, **3 Reference pose** under each joint), with **Properties** for the selected item. Right-click for actions. **Start a knee** (on the model's properties) adds a femur, a tibia and a Grood & Suntay knee in one go.

**1. Bones: axes on each bone.** Right-click **Bones → Add bone** and choose its type: **Femur**, **Tibia** or **Other bone**. The type decides how its axes can be made.
- Load its surface (right-click → **Load surface…**): the same file, in the same model space, that you track with. Set its **side**.
- **Axes from**: an automatic method fitted to the surface (the default; see below), or **landmarks** placed by hand: select a landmark under the bone, then **right-click the surface** (point), or right-click and drag to paint a region (its centre, or a sphere fitted to it). Left-drag still rotates.
- Femur and tibia axes use ISB names: X anterior, Y superior, Z right (Wu et al. 2002). "Other bone" uses its principal axes of inertia (axis 1 = long axis), with no anatomical names.

A bone's axes live in its own model space, so they don't depend on where any other bone is. Saved on their own, bones already give each bone's motion about its own axes in the Visualiser (right-click a bone → **Add coordinate system…**, or **Load model…**).

**2. Joints (optional).** Right-click **Joints → Add joint…**, choose a definition and its proximal and distal bones:
- **Knee (tibiofemoral): Grood & Suntay**, femur → tibia. ISB doesn't define the knee; this is the convention most BVR knee studies use. Left knees are mirrored, so positive values mean the same motion on both sides: flexion, adduction (varus), internal rotation; lateral, anterior and proximal translation of the tibia.
- **Joint coordinate system (ISB style)** between any two femur/tibia-type bones (proximal Z, floating axis, distal Y), without clinical names.
- **Cardan X-Y-Z** between any two bones, e.g. two carpals with inertial axes.

With no reference pose, a joint's zero is **anatomical**: all angles are 0 when the two bones' axes line up. No neutral pose is needed for that, so the overlapping model spaces you see when surfaces are loaded don't matter. Tracking places each bone at every frame, and the angles come from the axes' relative orientation.

**3. Reference pose (optional, per joint).** Select **Reference pose** under the joint, then:
- **From a neutral trial…**: the .tra files of a static/neutral trial for the two bones (the bones' preview tracking is offered), averaged over every frame both are tracked, or over a range of frames.
- **From the scan pose…**: where the bones were in the CT/MRI. For SlicerAutoscoperM surfaces, pick each bone's `*-DICOM2AUT.tfm` (found automatically when it sits in the usual `Transforms` folder). If both surfaces are already in the scan's space, choose that option instead.
- **Normalise by**: *subtract the reference values* (everything is 0 in the reference pose), or *realign the distal axes to the proximal axes in the reference pose* (rotations are exactly 0 there; translations are subtracted).

The reference is stored as the two bones' relative pose, so it stays valid if a bone's axes are changed later. The view shows the joint in its reference pose, and Properties lists the anatomical values that are being removed.

**Preview and checks.** Attach a tracking file to a bone (right-click → **Preview tracking**) to see joints in a tracked pose with their values, anatomical and relative to the reference. **Compare methods** lists the joint's anatomical angles for every combination of the bones' axis methods. **Save model…** writes a `.bvrmodel.json`: bones (surface, axes method, landmarks, axes), joints and reference poses. Open it again here to edit it. In a Visualiser tab, **Load model…** brings in its bones, coordinate systems and joints together. Version-1 files (one template) are converted on opening.

### Automatic axes (no landmarks)

Each bone's axes can come from an automatic method that fits shapes to the bone surface **or** from landmarks you place. Select the bone and choose **Axes from** in Properties (or right-click it). For the knee:

| Bone | Method | What it fits | After |
|---|---|---|---|
| Femur | Cylinder fitted to the posterior condyles | Z = cylinder axis through the posterior/distal condyle surfaces; origin at its middle | Miranda et al. 2010; GIBOC-Cylinder (Renault et al. 2018) |
| Femur | Spheres fitted to each posterior condyle | Z joins the two sphere centres; origin midway | Kai et al. 2014; GIBOC-Spheres |
| Tibia | Ellipse of the largest plateau cross-section | Z = ellipse major axis; origin at the section centroid | Kai et al. 2014; GIBOC-Ellipse |
| Tibia | Centroids of the two plateau surfaces | Z joins the centroids of the upward-facing medial and lateral plateau surfaces | GIBOC-Centroids |
| Tibia | Principal axes of the proximal tibia | Z = widest inertial axis of the top 15 mm; origin at its centroid | Miranda et al. 2010 |

In every method Y is the **shaft line** (a line through the centres of shaft cross-sections), so the methods work on partial bones without the hip or ankle. Anterior is found from the joint end's posterior overhang of the shaft line (the femoral condyles and tibial plateau reach further behind the shaft than in front), and right then follows from anterior and superior. **Flip anterior/posterior** and **Flip superior/inferior** correct a bone where that fails (e.g. a whole bone, where both ends are wide); a note appears when the overhang isn't clear-cut. With the bone selected, the view shows what was fitted (purple), the surface points used (orange) and the shaft centres (teal), and Properties shows the fit quality.

These are our own implementations written from the published descriptions (Modenese & Renault 2021 compared these algorithm families in STAPLE); STAPLE's code is not used.

**Compare methods** (select the joint, with preview tracking on both bones) lists the joint angles at the preview frame for every combination of the bones' methods, including landmarks if placed. Different definitions give different offsets: on the knee sample, the femur and tibia method choices each shift flexion by about 8–9°, mainly because the landmark long axis runs from the joint centre to the shaft centre while the automatic methods use the shaft line itself. Report the method you used, and use a reference pose when comparing across definitions.

Partial bones: Autoscoper surfaces often stop part way up the femur and down the tibia. Where the landmark definition needs the hip centre or the malleoli, paint a band around the shaft end instead — the shaft centre gives a long axis that follows the bone. Report that substitution with your results.

**Planned:** more bone and joint types (ISB ankle/hip, multi-segment foot models), automatic foot and ankle axes (as in AAFACT, Peterson et al. 2023), and your own joint definitions.

## How the files are interpreted

These conventions were checked against the SlicerAutoscoperM sample data (by projecting tracked bones onto the real radiographs) and against the Autoscoper source code. The tests check them.

- **Tracking:** pose at frame *f* = the .tra row as a 4×4 matrix × the model in Autoscoper volume space. xyzypr uses R = Rz(yaw)·Ry(pitch)·Rx(roll).
- **TIF → Autoscoper space:** a voxel at column *i*, row *j*, slice *k* sits at x = *i*·dx, y = (rows − *j*)·dy, z = −*k*·dz.
- **.tfm files** (ITK) store the resampling direction, so they're inverted before use.
- **Calibration (MayaCam 2.0):** x_cam = R·X + t, pixel = K·x_cam / z.

## Testing

```
pip install -e ".[dev]"
pytest                                   # synthetic tests
BVR_SAMPLE_DATA=/path/to/sample_data pytest   # plus the real-data tests
```

`BVR_SAMPLE_DATA` is a folder holding the three unzipped SlicerAutoscoperM sample datasets (`2025-02-19-Wrist`, `2025-01-12-Ankle`, `Knee_SampleData`). Download them from https://github.com/BrownBiomechanics/Autoscoper/releases/tag/sample-data.

`tests/drive_gui.py` drives the real window on the wrist and ankle data and saves screenshots. `tests/drive_model_builder.py` builds a knee model (bones with automatic and landmark axes, the joint, the method comparison and reference poses, then loading it in the Visualiser) on the knee sample by clicking and painting in the viewport, then checks the joint angles in a Visualiser tab against a known flexion sweep. `python -m bvrviewer --selftest SAMPLE_DIR out.png` checks that a build can load and render the wrist sample.

## Building the app

```
pip install -e ".[dev]"
cd packaging
pyinstaller bvr_viewer.spec --noconfirm
```

This makes `packaging/dist/BVR Motion Viewer/` (Windows, Linux) or `BVR Motion Viewer.app` (macOS). Build on each operating system you want to support. Pushing a tag such as `v0.1.0` to GitHub runs `.github/workflows/build.yml`, which builds all three. Unsigned builds show a Windows SmartScreen or macOS Gatekeeper warning the first time they open.

The Linux build is about 1.1 GB unzipped, most of it VTK.

## Known limitations

- Calibration: only MayaCam 2.0 so far. MayaCam 1.0 and VTK camera (.json) files are detected and reported as unsupported.
- `VolumeFlip` values other than 0 0 0 are applied but haven't been tested against real data.
- Units in .tra files can't be detected; they default to Autoscoper's mm and degrees.
- One viewport per tab. A split layout (3D plus both X-ray views at once) is in the design but not built yet.
- The through-X-ray view frames the image approximately (bones and image always line up exactly).
