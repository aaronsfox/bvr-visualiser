# PyInstaller spec: builds a double-clickable app with no Python install needed.
#   pip install -e ".[dev]"
#   pyinstaller packaging/bvr_viewer.spec --noconfirm
# Output: dist/BVR Motion Viewer/ (Windows/Linux folder) or dist/BVR Motion Viewer.app (macOS).
# Build on each operating system you want to support; PyInstaller can't cross-compile.

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# VTK modules PyVista loads lazily (rendering backends, file readers used by pv.read).
VTK = ["vtkRenderingOpenGL2", "vtkRenderingVolumeOpenGL2", "vtkRenderingFreeType", "vtkRenderingAnnotation",
       "vtkRenderingContextOpenGL2", "vtkInteractionStyle", "vtkInteractionWidgets", "vtkIOGeometry",
       "vtkIOPLY", "vtkIOLegacy", "vtkIOXML", "vtkFiltersModeling", "vtkFiltersHybrid", "vtkFiltersTexture",
       "vtkRenderingVolume", "vtkRenderingLabel", "vtkRenderingUI", "vtkImagingCore", "vtkFiltersGeometry"]
hidden = (
    [f"vtkmodules.{m}" for m in VTK]
    + ["vtkmodules.qt.QVTKRenderWindowInteractor", "vtkmodules.util.numpy_support"]
    + collect_submodules("imagecodecs")
    + collect_submodules("pyvistaqt")
    + ["scipy.spatial.transform._rotation_groups", "skimage.measure", "skimage.filters"]
)
datas = collect_data_files("pyvista") + collect_data_files("imageio_ffmpeg")

# Large Qt and Python parts the app never uses.
EXCLUDE = ["tkinter", "matplotlib.tests", "PyQt5", "PyQt6", "PySide2", "IPython", "jupyter", "cv2",
           "vtkmodules.web", "vtkmodules.test", "trame", "notebook", "pandas", "sympy", "numba", "dask",
           "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
           "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml",
           "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization",
           "PySide6.QtBluetooth", "PySide6.QtSql", "PySide6.QtDesigner", "PySide6.QtQuick3D"]

a = Analysis(
    ["launcher.py"],
    pathex=[".."],
    hiddenimports=hidden,
    datas=datas,
    excludes=EXCLUDE,
)
# Drop bundled Qt libraries for the excluded modules (PyInstaller collects some anyway).
DROP = ("WebEngine", "Qt6Quick", "Qt6Qml", "Qt63D", "Qt6Pdf", "Qt6Multimedia", "Qt6Charts", "Qt6Designer",
        "Qt6DataVisualization", "Qt6Quick3D", "Qt6Bluetooth", "Qt6Sql", "translations", "opencv")
a.binaries = [b for b in a.binaries if not any(d in b[0] for d in DROP)]
a.datas = [d for d in a.datas if not any(x in d[0] for x in DROP)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BVR Motion Viewer", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="BVR Motion Viewer")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="BVR Motion Viewer.app", bundle_identifier="org.bvr.motionviewer",
                 info_plist={"NSHighResolutionCapable": True})
