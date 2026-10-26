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
        # a joint (inertial axes, no landmarks) exercises the joint maths and the curve plots
        import tempfile

        from ..core import joints as J

        jm = J.KinematicModel(name="Self-test")
        jm.add_segment("generic", "Radius").surface = tab.state.models[0]
        jm.add_segment("generic", "MC3").surface = tab.state.models[1]
        jm.add_joint("cardan_xyz", "Radius", "MC3", "Wrist")
        jm.compute()
        jpath = Path(tempfile.gettempdir()) / "bvr_selftest.bvrmodel.json"
        jm.save(jpath)
        tab.load_joint_model_file(jpath, ask=False)
        tab.timeline.slider.setValue(40)
        tab.set_xray_view(0)

        def finish():
            tab.plotter.screenshot(out)
            ok = len(tab.state.models) == 2 and all(m.tracking for m in tab.state.models)
            items = {i.name: i for i in tab.state.joints}
            k = tab.joint_values(items["Wrist"]) if "Wrist" in items else None
            ok = ok and all(m.has_axes for m in tab.state.models) and tab.joint_values(tab.state.models[1]) is not None
            ok = ok and k is not None and not tab.curves.isHidden()
            print("SELFTEST", "OK" if ok else "FAILED", out, flush=True)
            win.close()
            sys.exit(0 if ok else 1)

        QTimer.singleShot(1500, finish)

    tab._build_models(models, then=after_models)
