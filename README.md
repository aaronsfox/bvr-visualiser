# BVR Motion Viewer (prototype)

A desktop app for watching bone motion tracked in **Autoscoper / SlicerAutoscoperM** from biplanar videoradiography.

Each tab is one trial. In a tab you add bone models, attach each model's own tracking file, and press play. Radiographs are optional: add the X-ray cameras' calibration files and images to see the radiographs behind the bones in 3D, or to look "through" an X-ray camera with the bones overlaid on the image.

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
| **Model Builder** | available (knee first) | Define bone axes and joints on your surfaces from a template, and save a joint model for the other tools. |
| **Reliability Analysis** | coming soon | Load models and repeated tracking of the same trial, by the same tester or different testers, and get accuracy and reliability metrics across them. |
| **Validity Analysis** | coming soon | Load models, criterion tracking data and testers' tracking, and get accuracy against the criterion. |

Adding a tool: describe it in `bvrviewer/app/tools.py` (it then appears on the Start page as "Coming soon"), write its tab widget, and register it in `MainWindow.TAB_CLASSES`.

## Using the Visualiser

The Visualiser tab is laid out like OpenSim. On the left, the **Navigator** lists what's in the tab under **Models**, **X-ray cameras** and **Scene**. Below it, **Properties** shows the settings of whatever is selected there. The 3D view and timeline are on the right. **Right-click** any level or item in the navigator for its actions.

1. **Open a Visualiser tab** from the Start page (or File → New Visualiser tab, Ctrl+T). Each tab starts empty.
2. **Add a model:** right-click **Models → Add model…** (or File → Add model). STL, OBJ, PLY, VTK or TIF; you can select several at once. Each model is listed by name, which starts as its file name.
   - **TIF partial volumes** ask for the voxel size, which TIFs don't store. Type it in, or read it from the trial's `.cfg`. The surface is built from the edge of the masked volume, which reproduces SlicerAutoscoperM's `AUT_*.stl` meshes. TIF models can also be shown as a volume rendering (Display as: Volume).
   - **STL surfaces** are assumed to be in Autoscoper space, like SlicerAutoscoperM's `AUT_*.stl`. If a surface is in CT space instead (e.g. SlicerAutoscoperM's `Models/rad.stl`), right-click it and choose **Apply .tfm transform…** with its `*-DICOM2AUT.tfm`.
3. **Attach tracking:** right-click the model → **Attach tracking…**. The tracking file then shows under the model. The format is detected: matrix or x y z yaw pitch roll, row or column order, comma or space, and several bones in one file (you'll be asked which). Untracked frames (`NaN`) show in orange on the timeline. If something looks wrong, right-click → **Tracking format…** to change the units (mm/cm, degrees/radians) or the order.
4. **Change how it looks:** select the model and use **Properties**: name, colour, opacity, visible, and surface or volume display. The tick box beside each model in the navigator shows or hides it, and F2 (or clicking a selected name) renames it.
5. **Look around:** drag to rotate, Shift-drag to pan, scroll to zoom, right-click a bone in the 3D view to select it. Use the Free 3D / XY / YZ / XZ buttons, and **Flip** to look from the other side.
6. **Play** with Space, step with ← →. Speed and Loop are on the timeline.

**What's where**

| Select… | Properties | Right-click |
|---|---|---|
| **Models** | count, frames | Add model, Show all, Hide all |
| a **model** | name, colour, opacity, visible, display; hold this bone still; model file details | attach/replace/detach tracking, tracking format, hold still, show/hide, rename, replace file, .tfm or voxel size, remove |
| its **tracking** | file, frames, untracked frames, detected format | attach/replace, format, detach |
| **X-ray cameras** | settings for all radiographs: image opacity, distance behind bones, bone opacity and wireframe in through-X-ray views, brightness, contrast, invert | Add X-ray camera, Import from .cfg |
| a **camera** | name, visible; what it shows in 3D (radiograph, source marker, beam outline); which image is frame 1; files | view through it, show/hide, rename, change files, remove |
| **Joints** | show joint curves | Load joint model, show/hide curves |
| a **joint** | joint model, which tab models are its bones, axes and curves on/off, zero at a frame, values at this frame | show curves, show/hide axes, zero, export CSV, rename, change bones, remove |
| **Scene** | axes, grid, background, motion trail, untracked frames | fit bones in view |

- **Hold this bone still** (model properties or right-click): view the other bones' motion relative to it.
- **Screenshot** (Ctrl+P) and **Export video** (Ctrl+E, MP4).
- **Save tab as session** (Ctrl+S) saves file paths and every setting to a `.json`. **Open saved session** restores it.

### Joint angles (optional)

Build a joint model in the **Model Builder** (next section) and save it. Then, in a Visualiser tab with those bones and their tracking, right-click **Joints → Load joint model…**. The joint's bones are matched to the tab's models by surface file (then by name); you'll be asked if that's ambiguous, and can change it later (**Change bones…**). Each bone's axes are drawn on it, and the joint's angles and translations are plotted under the 3D view, with a cursor at the current frame (click a plot to jump there). **Zero at this frame** subtracts the values at a reference frame, e.g. a neutral pose. **Export CSV** writes every frame. Joints are saved with the session.

The axes only fit a model built from the **same surface file in the same model space** as the joint model, so use the same STL/TIF (and the same voxel size or .tfm) in both tabs.

### Radiographs (optional)

Right-click **X-ray cameras → Add X-ray camera…** (or use the Radiographs menu) and pick:

1. the calibration file (MayaCam 2.0 text, as in SlicerAutoscoperM's sample data);
2. the radiographs: a folder, selected files, or one multi-page TIFF. Images are ordered by the last number in each file name, so any naming works;
3. which image is tracking frame 1 (also changeable later in the camera's properties).

The dialog previews your tracked bones on that image. If they sit on their shadows, everything is matched up. Or use **Import cameras from .cfg…** to add every camera listed in an Autoscoper trial file at once.

With cameras added, the radiographs appear on detector planes behind the bones. The planes are see-through from behind, so they don't hide the bones. The **Through X-ray** buttons (or double-clicking a camera in the navigator) look from each X-ray source with the bones overlaid on the image.

Radiographs are decoded in the background as soon as a camera is added, so playback may be slower for the first few seconds of a long trial. They're shown at up to 1024 pixels across, and hidden cameras aren't decoded at all.

## The Model Builder

The Model Builder applies a joint definition to your own bone surfaces. Its navigator lists the template's **Bones** (each with its landmarks) and **Joints**; **Properties** shows the selected item.

1. **Choose a template** (select the model at the top). For now:
   - **Knee: Grood & Suntay** — ISB doesn't define the knee, so this uses the Grood & Suntay (1983) joint coordinate system with ISB-style axes on each bone (X anterior, Y superior, Z right; Wu et al. 2002 for the femur). Set the **side**: left knees are mirrored so positive values mean the same motion on both sides (flexion, adduction/varus, internal rotation; lateral, anterior, proximal translations of the tibia).
   - **Generic: inertial axes** — no landmarks: each bone's axes are its principal axes of inertia. Useful for small bones, or as a check.
2. **Load each bone's surface:** right-click the bone → **Load surface…**. Use the same file (and model space) you track with.
3. **Place each landmark:** select it in the navigator, then **right-click the surface**. Depending on the landmark's method (Properties):
   - **Point** — right-click places it on the surface; right-click again to move it.
   - **Centre of a painted region** — right-click and drag to paint; the landmark is the region's centre. Good for shaft centres on partial bones (a band around the top of the femur, the bottom of the tibia).
   - **Sphere fit** — paint part of a round surface (e.g. the femoral head); the landmark is the fitted sphere's centre.
   Left-drag still rotates. Done landmarks get a tick; the bone's axes appear as soon as it has all of them (red X, green Y, blue Z).
4. **Check the joint:** select **Joints → Knee**. With a tracking file attached to each bone (right-click the bone → **Preview tracking**), both bones are shown in place with the joint's angles and translations at the first tracked frame. Sanity checks (e.g. the epicondyles being implausibly close) are listed there.
5. **Save model…** writes a `.bvrmodel.json` (surfaces, landmark points, axes). Open it again here to edit it, or load it into a Visualiser tab.

### Automatic axes (no landmarks)

Each bone's axes can come from landmarks you place **or** from an automatic method that fits shapes to the bone surface. Select the bone and choose **Axes from** in Properties (or right-click the bone; **All automatic** / **All landmarks** on the model). For the knee:

| Bone | Method | What it fits | After |
|---|---|---|---|
| Femur | Cylinder fitted to the posterior condyles | Z = cylinder axis through the posterior/distal condyle surfaces; origin at its middle | Miranda et al. 2010; GIBOC-Cylinder (Renault et al. 2018) |
| Femur | Spheres fitted to each posterior condyle | Z joins the two sphere centres; origin midway | Kai et al. 2014; GIBOC-Spheres |
| Tibia | Ellipse of the largest plateau cross-section | Z = ellipse major axis; origin at the section centroid | Kai et al. 2014; GIBOC-Ellipse |
| Tibia | Centroids of the two plateau surfaces | Z joins the centroids of the upward-facing medial and lateral plateau surfaces | GIBOC-Centroids |
| Tibia | Principal axes of the proximal tibia | Z = widest inertial axis of the top 15 mm; origin at its centroid | Miranda et al. 2010 |

In every method Y is the **shaft line** (a line through the centres of shaft cross-sections), so the methods work on partial bones without the hip or ankle. Anterior is found from the joint end's posterior overhang of the shaft line (the femoral condyles and tibial plateau reach further behind the shaft than in front), and right then follows from anterior and superior. **Flip anterior/posterior** and **Flip superior/inferior** correct a bone where that fails (e.g. a whole bone, where both ends are wide); a note appears when the overhang isn't clear-cut. With the bone selected, the view shows what was fitted (purple), the surface points used (orange) and the shaft centres (teal), and Properties shows the fit quality.

These are our own implementations written from the published descriptions (Modenese & Renault 2021 compared these algorithm families in STAPLE); STAPLE's code is not used.

**Compare methods** (select the joint, with preview tracking on both bones) lists the joint angles at the preview frame for every combination of the bones' methods, including landmarks if placed. Different definitions give different offsets: on the knee sample, the femur and tibia method choices each shift flexion by about 8–9°, mainly because the landmark long axis runs from the joint centre to the shaft centre while the automatic methods use the shaft line itself. Report the method you used, and zero to a reference posture when comparing across definitions.

Partial bones: Autoscoper surfaces often stop part way up the femur and down the tibia. Where the template's definition needs the hip centre or the malleoli, paint a band around the shaft end instead — the shaft centre gives a long axis that follows the bone. Report that substitution with your results.

**Planned:** more templates (ISB ankle/hip, multi-segment foot models), automatic foot and ankle axes (as in AAFACT, Peterson et al. 2023), and your own joint definitions.

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

`tests/drive_gui.py` drives the real window on the wrist and ankle data and saves screenshots. `tests/drive_model_builder.py` builds a knee model (landmarks, then every automatic method and the method comparison) on the knee sample by clicking and painting in the viewport, then checks the joint angles in a Visualiser tab against a known flexion sweep. `python -m bvrviewer --selftest SAMPLE_DIR out.png` checks that a build can load and render the wrist sample.

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
