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

## Using it

1. **New tab** (top right, or Ctrl+T). Each tab starts empty. Double-click a tab to rename it.
2. **Add model** (STL, OBJ, PLY, VTK or TIF). You can select several at once. The file name becomes the bone's name; click the name to change it.
   - **TIF partial volumes** ask for the voxel size, which TIFs don't store. Type it in, or read it from the trial's `.cfg`. The surface is built from the edge of the masked volume, which reproduces SlicerAutoscoperM's `AUT_*.stl` meshes. TIF models can also be shown as a volume rendering (Display as: Volume or Both).
   - **STL surfaces** are assumed to be in Autoscoper space, like SlicerAutoscoperM's `AUT_*.stl`. If a surface is in CT space instead (e.g. SlicerAutoscoperM's `Models/rad.stl`), use **Apply .tfm…** with its `*-DICOM2AUT.tfm`.
3. **Attach .tra** to each model. The format is detected: matrix or x y z yaw pitch roll, row or column order, comma or space, and several bones in one file (you'll be asked which). Untracked frames (`NaN`) show in orange on the timeline. If something looks wrong, use **Format…** to change the units (mm/cm, degrees/radians) or the order.
4. **Look around:** drag to rotate, Shift-drag to pan, scroll to zoom, right-click a bone to select it. Use the Free 3D / XY / YZ / XZ buttons, and **Flip** to look from the other side.
5. **Play** with Space, step with ← →. Speed and Loop are on the timeline.

**Other options**

Scene and radiograph options are in the tabbed panel along the bottom of the window. Click a tab to open it; click the open tab again (or **Hide**) to fold the panel away.

- **Hold this bone still** (on the selected model): view the other bones' motion relative to it.
- **Scene tab:** axes and grid; background (dark, light grey, or white for figures); a motion trail of ghost copies of the selected bone at earlier frames; and what to do on untracked frames (hide the bone, show its last tracked pose, or interpolate across the gap).
- **Screenshot** (Ctrl+P) and **Export video** (Ctrl+E, MP4).
- **Save tab as session** (Ctrl+S) saves file paths and every setting to a `.json`. **Open saved session** restores it.

### Radiographs (optional)

In the **Radiographs** tab of the bottom panel (or the Radiographs menu), choose **Add X-ray camera** and pick:

1. the calibration file (MayaCam 2.0 text, as in SlicerAutoscoperM's sample data);
2. the radiographs: a folder, selected files, or one multi-page TIFF. Images are ordered by the last number in each file name, so any naming works;
3. which image is tracking frame 1.

The dialog previews your tracked bones on that image. If they sit on their shadows, everything is matched up. Or use **Import from .cfg…** to add every camera listed in an Autoscoper trial file at once.

With cameras added, the radiographs appear on detector planes behind the bones. The planes are see-through from behind, so they don't hide the bones. The **Through X-ray** buttons look from each X-ray source with the bones overlaid on the image.

Radiographs are decoded in the background as soon as a camera is added, so playback may be slower for the first few seconds of a long trial. They're shown at up to 1024 pixels across, and hidden cameras aren't decoded at all.

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

`tests/drive_gui.py` drives the real window on the wrist and ankle data and saves screenshots. `python -m bvrviewer --selftest SAMPLE_DIR out.png` checks that a build can load and render the wrist sample.

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
