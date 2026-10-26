"""`--selftest SAMPLE_DIR OUT.png`: load the wrist sample with radiographs, render, save a screenshot, exit.

Used to check that a packaged build can read every file type and render. SAMPLE_DIR is the folder
holding the unzipped SlicerAutoscoperM sample data (the one containing 2025-02-19-Wrist).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer


def run(win, sample_dir: str, out: str) -> None:
    from ..core.config import read_cfg
    from ..core.models import BoneModel
    from ..core.session import XrayCamera

    w = Path(sample_dir) / "2025-02-19-Wrist"
    tab = win.new_visualiser_tab()
    models = [
        BoneModel(name="Radius", path=w / "Volumes" / "rad.tif", voxel_size=(0.391, 0.391, 0.625),
                  tracking_path=w / "Tracking_SAM" / "flx_ext_rad.tra"),
        BoneModel(name="MC3", path=w / "Models" / "AUT_Models" / "AUT_mc3.stl", color="#B78AF0",
                  tracking_path=w / "Tracking_SAM" / "flx_ext_mc3.tra"),
    ]

    def after_models():
        cfg = read_cfg(w / "Wrist_Sample_Data.cfg")
        for k, (cal, folder) in enumerate(zip(cfg.calibrations, cfg.camera_dirs), start=1):
            cam = XrayCamera(calibration_path=cal, source_kind="folder", source=[folder], name=f"X-ray {k}")
            cam.build()
            tab.state.cameras.append(cam)
            tab.scene.add_camera(cam)
        tab.refresh_all()
        tab.timeline.slider.setValue(40)
        tab.set_xray_view(0)

        def finish():
            tab.plotter.screenshot(out)
            ok = len(tab.state.models) == 2 and all(m.tracking for m in tab.state.models)
            print("SELFTEST", "OK" if ok else "FAILED", out, flush=True)
            win.close()
            sys.exit(0 if ok else 1)

        QTimer.singleShot(1500, finish)

    tab._build_models(models, then=after_models)
