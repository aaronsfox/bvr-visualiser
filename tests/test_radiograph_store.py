"""Radiographs kept for later (Visualiser #13): the local cache, and saving radiographs and other files
with a session so it opens quickly anywhere, even without the originals."""

import os
import shutil
import time

import numpy as np
import pytest
import tifffile

from bvrviewer.core import radiograph_store as RS
from bvrviewer.core.images import ImageSequence, find_images
from bvrviewer.core.models import BoneModel
from bvrviewer.core.session import (
    TabState,
    XrayCamera,
    data_folder,
    load_session,
    save_session,
)

CAL = """-1111.58096582,81.0771441702,1052.81858634
-0.993870320862,-53.8345571013,-1.39252393885
-40.1633538278,21.9311517231,-883.307872093
1409.63353828,788.688482769,-8833.07872093
0.1,2016,2016
"""


def make_images(folder, n=12, shape=(90, 120), dtype=np.uint8):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = (rng.integers(0, 50, shape) + i * 10).astype(dtype)
        tifffile.imwrite(folder / f"cam1_UND.{i + 1:04d}.tif", img)
    return sorted(folder.glob("*.tif"))


def seq_of(folder, **kw):
    files, _ = find_images(folder)
    return ImageSequence(files=files, **kw)


def all_display(seq):
    return np.stack([seq.get_display(i) for i in range(len(seq))])


def test_store_round_trip(tmp_path):
    make_images(tmp_path / "img")
    seq = seq_of(tmp_path / "img", display_max=64)
    assert seq.display_step == 2
    out = RS.write_store(seq, tmp_path / "x" / f"c{RS.SUFFIX}")
    meta = RS.read_meta(out)
    assert meta["n"] == 12 and meta["display_step"] == 2 and RS.matches(meta, seq)
    again = seq_of(tmp_path / "img", store=out)
    assert again.store_kind == "session" and again.display_step == 2
    np.testing.assert_array_equal(all_display(again), all_display(seq))
    assert out.stat().st_size < 12 * 45 * 60 * 1.5


def test_store_without_the_originals(tmp_path):
    make_images(tmp_path / "img")
    seq = seq_of(tmp_path / "img")
    out = RS.write_store(seq, tmp_path / f"c{RS.SUFFIX}")
    ref = all_display(seq)
    seq.close()
    shutil.rmtree(tmp_path / "img")
    only = ImageSequence(store=out)
    assert not only.originals_available and len(only) == 12
    assert only.label(4) == "0005" and (only.width, only.height) == (120, 90)
    np.testing.assert_array_equal(all_display(only), ref)
    np.testing.assert_array_equal(only.get(3), ref[3])          # full size falls back to the copy
    assert "weren't found" in only.where_from()


def test_changed_originals_are_detected(tmp_path):
    files = make_images(tmp_path / "img")
    out = RS.write_store(seq_of(tmp_path / "img"), tmp_path / f"c{RS.SUFFIX}")
    time.sleep(1.1)
    tifffile.imwrite(files[5], np.zeros((90, 120), np.uint8) + 7, compression="zlib")    # new size and time
    assert not RS.matches(RS.read_meta(out), seq_of(tmp_path / "img"))
    assert not seq_of(tmp_path / "img").attach_store(out)


def test_local_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BVR_CACHE_DIR", str(tmp_path / "cache"))
    make_images(tmp_path / "img")
    seq = seq_of(tmp_path / "img")
    assert RS.find_cached(seq) is None
    seq.preload()
    p = RS.save_to_cache(seq)
    assert p.parent == tmp_path / "cache" and RS.cache_size() == p.stat().st_size
    fresh = seq_of(tmp_path / "img")
    assert RS.find_cached(fresh) == p
    assert fresh.attach_store(p, "cache") and fresh.where_from() == "the local cache"
    np.testing.assert_array_equal(all_display(fresh), all_display(seq))
    # other trials push the least recently used one out when the cache is full
    make_images(tmp_path / "img2")
    os.utime(p, (time.time() - 100, time.time() - 100))
    q = RS.save_to_cache(seq_of(tmp_path / "img2"), limit_bytes=p.stat().st_size + 10)
    assert q.exists() and not p.exists()
    RS.clear_cache()
    assert RS.cache_files() == []


def test_16_bit_images_keep_their_contrast(tmp_path):
    make_images(tmp_path / "img", dtype=np.uint16)
    seq = seq_of(tmp_path / "img")
    out = RS.write_store(seq, tmp_path / f"c{RS.SUFFIX}")
    np.testing.assert_array_equal(all_display(seq_of(tmp_path / "img", store=out)), all_display(seq))


def test_network_paths():
    assert RS.is_network_path(r"\\server\share\trial\img.tif")
    assert not RS.is_network_path(os.getcwd())


# ---------------------------------------------------------------- sessions
def make_tab(root):
    import pyvista as pv

    make_images(root / "Radiographs" / "Camera 1")
    (root / "Calibration").mkdir()
    (root / "Calibration" / "cam1.csv").write_text(CAL)
    (root / "Models").mkdir()
    pv.Sphere(radius=10).save(root / "Models" / "tal.stl")
    (root / "Tracking").mkdir()
    (root / "Tracking" / "tal.tra").write_text("1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1\n" * 12)
    cam = XrayCamera(calibration_path=root / "Calibration" / "cam1.csv", source_kind="folder",
                     source=[root / "Radiographs" / "Camera 1"], name="X-ray 1")
    cam.build()
    st = TabState(name="Trial")
    st.models.append(BoneModel(name="tal", path=root / "Models" / "tal.stl", tracking_path=root / "Tracking" / "tal.tra"))
    st.cameras.append(cam)
    return st


def test_data_folder_name(tmp_path):
    assert data_folder(tmp_path / "Trial003.bvrsession.json") == tmp_path / "Trial003_data"
    assert data_folder(tmp_path / "x.json") == tmp_path / "x_data"


def test_session_with_radiographs_only(tmp_path):
    st = make_tab(tmp_path / "orig")
    s = tmp_path / "save" / "Trial.bvrsession.json"
    s.parent.mkdir()
    rep = save_session(st, s, radiographs=True)
    assert rep["radiographs"] == 1 and rep["files"] == 0
    assert (tmp_path / "save" / "Trial_data" / "radiographs" / f"X-ray 1{RS.SUFFIX}").exists()
    back = load_session(s)
    assert back.models[0].path == tmp_path / "orig" / "Models" / "tal.stl"     # other files: where they were
    cam = back.cameras[0]
    cam.build()
    assert cam.images.store_kind == "session" and cam.note == ""
    np.testing.assert_array_equal(all_display(cam.images), all_display(st.cameras[0].images))
    # originals changed since: they're read again, with a note
    time.sleep(1.1)
    f = min((tmp_path / "orig" / "Radiographs" / "Camera 1").glob("*.tif"))
    tifffile.imwrite(f, np.zeros((90, 120), np.uint8), compression="zlib")
    cam.build()
    assert cam.images.store_kind is None and "changed" in cam.note


def test_self_contained_session_moves_anywhere(tmp_path):
    st = make_tab(tmp_path / "orig")
    ref = all_display(st.cameras[0].images)
    s = tmp_path / "save" / "Trial.bvrsession.json"
    s.parent.mkdir()
    seen = []
    rep = save_session(st, s, radiographs=True, all_files=True, progress=lambda t, d, n: seen.append((d, n)))
    assert rep["files"] == 3 and rep["radiographs"] == 1          # stl, tra, calibration
    assert seen[-1][0] == seen[-1][1] == 3 + 12
    # move the session and its folder, and lose the originals
    shutil.move(str(tmp_path / "save"), str(tmp_path / "elsewhere"))
    st.cameras[0].images.close()
    shutil.rmtree(tmp_path / "orig")
    back = load_session(tmp_path / "elsewhere" / "Trial.bvrsession.json")
    m = back.models[0]
    assert m.path == tmp_path / "elsewhere" / "Trial_data" / "files" / "bones" / "tal.stl" and m.path.exists()
    assert m.tracking_path.exists() and back.originals[str(m.path)].endswith("tal.stl")
    m.build()
    cam = back.cameras[0]
    cam.build()
    assert not cam.images.originals_available and "weren't found" in cam.note
    np.testing.assert_array_equal(all_display(cam.images), ref)
    # saving it again in place keeps everything (files and radiographs aren't copied onto themselves)
    rep = save_session(back, tmp_path / "elsewhere" / "Trial.bvrsession.json", radiographs=True, all_files=True)
    assert rep["files"] == 3
    again = load_session(tmp_path / "elsewhere" / "Trial.bvrsession.json")
    assert again.originals[str(again.models[0].path)] == back.originals[str(m.path)]


def test_missing_copy_falls_back_to_the_original(tmp_path):
    st = make_tab(tmp_path / "orig")
    s = tmp_path / "Trial.bvrsession.json"
    save_session(st, s, all_files=True)
    (tmp_path / "Trial_data" / "files" / "bones" / "tal.stl").unlink()
    back = load_session(s)
    assert back.models[0].path == tmp_path / "orig" / "Models" / "tal.stl"


def test_cancel_saving(tmp_path):
    st = make_tab(tmp_path / "orig")
    with pytest.raises(InterruptedError):
        save_session(st, tmp_path / "T.json", radiographs=True, all_files=True, cancelled=lambda: True)


def test_cancel_while_saving_radiographs(tmp_path):
    st = make_tab(tmp_path / "orig")
    with pytest.raises(InterruptedError):
        save_session(st, tmp_path / "T.json", radiographs=True,
                     cancelled=lambda: len(list((tmp_path / "T_data").rglob("*.partial"))) > 0)
    assert not list((tmp_path / "T_data").rglob("*.tif*")) and not (tmp_path / "T.json").exists()
