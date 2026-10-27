"""AAFACT results import and the external-program runner (with a stand-in for AAFACT)."""

import os
import stat
import sys
import textwrap

import numpy as np
import pytest
import pyvista as pv

from bvrviewer.core import aafact as AF
from bvrviewer.core import joints as J

openpyxl = pytest.importorskip("openpyxl")


def _rot(ax, deg):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    a = np.radians(deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def write_aafact_xlsx(path, sheets):
    """sheets: [(sheet name, subject, bone, side, origin, R (3x3 columns AP, SI, ML))] in AAFACT's layout."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, subject, bone, side, o, R in sheets:
        ws = wb.create_sheet(name)
        for i, (a, b) in enumerate([("Subject", subject), ("Bone Model", bone), ("Side", side)], start=1):
            ws.cell(i, 1, a)
            ws.cell(i, 2, b)
        rows = ["Coordinate System at Original Orientation", "Center Origin", "AP Axis", "SI Axis", "ML Axis",
                "Coordinate System at (0,0,0)", "Center Origin", "AP Axis", "SI Axis", "ML Axis"]
        for i, t in enumerate(rows, start=5):
            ws.cell(i, 1, t)
        for c, t in enumerate("XYZ", start=2):
            ws.cell(5, c, t)
            ws.cell(10, c, t)
        pts = [o, o + R[:, 0], o + R[:, 1], o + R[:, 2]]
        for r, p in zip((6, 7, 8, 9), pts):
            for c in range(3):
                ws.cell(r, c + 2, float(p[c]))
        for r, p in zip((11, 12, 13, 14), [np.zeros(3), *np.eye(3)]):
            for c in range(3):
                ws.cell(r, c + 2, float(p[c]))
    wb.save(path)


def test_read_results(tmp_path):
    R1 = _rot([1, 2, 3], 25)
    R2 = _rot([0, 1, 0], 40)
    Rskew = R1.copy()
    Rskew[:, 2] = _rot(R1[:, 0], 1.5) @ Rskew[:, 2]          # ML 1.5 degrees off orthogonal
    xl = tmp_path / "CoordinateSystem_job.xlsx"
    write_aafact_xlsx(xl, [("TT_bvr_Talus_Right", "bvr_Talus_Right", "Talus", "Right", np.array([1.0, 2, 3]), R1),
                           ("ST_bvr_Talus_Right", "bvr_Talus_Right", "Talus", "Right", np.array([1.0, 2, 3]), Rskew),
                           ("bvr_Navicular_Right", "bvr_Navicular_Right", "Navicular", "Right", np.zeros(3), R2)])
    res = AF.read_results(xl)
    assert [(r.bone, r.option, r.side) for r in res] == [("Talus", "TT", "right"), ("Talus", "ST", "right"),
                                                         ("Navicular", "", "right")]
    np.testing.assert_allclose(res[0].acs[:3, :3], R1, atol=1e-9)
    np.testing.assert_allclose(res[0].acs[:3, 3], [1, 2, 3])
    assert abs(res[1].orthogonality - 1.5) < 0.05
    assert np.allclose(res[1].acs[:3, :3].T @ res[1].acs[:3, :3], np.eye(3), atol=1e-9)
    assert "Tibiotalar" in res[0].title
    assert AF.bone_from_name("S01_First_Metatarsal_L.stl") == "Metatarsal1"


def test_model_with_aafact_axes(tmp_path):
    R = _rot([0.3, 1, 0.2], 30)
    xl = tmp_path / "CoordinateSystem_job.xlsx"
    write_aafact_xlsx(xl, [(f"{o}_bvr_Talus_Left", "bvr_Talus_Left", "Talus", "Left", np.array([5.0, 0, 0]),
                            _rot([0, 0, 1], k * 10) @ R) for k, o in enumerate(("TN", "TT", "ST"))])
    m = J.KinematicModel.preset("ankle", "left")
    tal = m.segment("Talus")
    assert tal.axes_source == "staple_talus_ankle"
    with pytest.raises(J.JointModelError):
        m.set_imported("Calcaneus", AF.read_results(xl))           # no calcaneus in these results
    m.set_imported("Talus", AF.read_results(xl), file=str(xl), option="TT")
    assert tal.axes_source == "aafact" and tal.imported["option"] == "TT"

    class _Surf:                                                   # stands in for a loaded BoneModel
        mesh = pv.Sphere()

        def to_dict(self):
            return None
    tal.surface = _Surf()
    m.compute()
    np.testing.assert_allclose(tal.acs[:3, :3], _rot([0, 0, 1], 10) @ R, atol=1e-9)
    m.set_imported_option("Talus", "ST")
    m.compute()
    np.testing.assert_allclose(tal.acs[:3, :3], _rot([0, 0, 1], 20) @ R, atol=1e-9)
    tal.surface = None
    m.save(tmp_path / "ankle.bvrmodel.json")
    m2 = J.KinematicModel.load(tmp_path / "ankle.bvrmodel.json", build_surfaces=False)
    assert m2.segment("Talus").axes_source == "aafact" and m2.segment("Talus").imported["option"] == "ST"
    assert [j.type for j in m2.joints] == ["ankle_isb", "subtalar_isb"]
    right = J.KinematicModel.preset("ankle", "right")
    with pytest.raises(J.JointModelError):
        right.set_imported("Talus", AF.read_results(xl))           # left results on a right bone


@pytest.mark.skipif(sys.platform.startswith("win"), reason="uses a POSIX shell script as the stand-in program")
def test_runner_with_a_stand_in_program(tmp_path):
    fake = tmp_path / "AAFACT"
    helper = tmp_path / "fake_aafact.py"
    helper.write_text(textwrap.dedent(f'''
        import sys, pathlib, numpy as np
        sys.path[:0] = [{str(os.path.dirname(__file__))!r}, {str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))!r}]
        from test_aafact import write_aafact_xlsx
        job = pathlib.Path(sys.argv[1])
        files = [p.name for p in job.iterdir()]
        assert files == ["bvr_Navicular_Right.stl"], files
        write_aafact_xlsx(job / f"CoordinateSystem_{{job.name}}.xlsx",
                          [("bvr_Navicular_Right", "bvr_Navicular_Right", "Navicular", "Right", np.zeros(3), np.eye(3))])
    '''))
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {helper} \"$1\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    res = AF.run(fake, pv.Sphere(radius=10), "Navicular", "right", tmp_path / "work")
    assert len(res) == 1 and res[0].bone == "Navicular"
    with pytest.raises(AF.AafactError):
        AF.run(tmp_path / "missing.exe", pv.Sphere(), "Navicular", "right", tmp_path / "work")
