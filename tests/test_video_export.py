"""Video export layouts (Visualiser #15) and the X-ray source marker default (Visualiser #14)."""

import pytest
from PySide6.QtCore import QRect

from bvrviewer.app.video_export import fit_into, tile_rects
from bvrviewer.core.session import XrayCamera


def _inside(r: QRect, w: int, h: int) -> bool:
    return r.left() >= 0 and r.top() >= 0 and r.right() < w + 1 and r.bottom() < h + 1


@pytest.mark.parametrize("layout", ["row", "main", "grid"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_tiles_fit_the_frame_without_overlapping(layout, n):
    rects = tile_rects(n, layout, 1920, 1080)
    assert len(rects) == n
    assert all(_inside(r, 1920, 1080) and r.width() > 100 and r.height() > 100 for r in rects)
    for i in range(n):
        for j in range(i + 1, n):
            assert not rects[i].intersects(rects[j]), (layout, n, i, j)


def test_main_layout_gives_the_first_view_most_room():
    main, *others = tile_rects(3, "main", 1920, 1080)
    assert main.width() * main.height() > 2 * max(r.width() * r.height() for r in others)
    assert all(r.left() > main.right() for r in others)


def test_grid_centres_a_part_filled_row():
    rects = tile_rects(3, "grid", 1920, 1080)
    last = rects[2]
    assert abs((last.left() + last.right()) / 2 - 960) < 3


def test_fit_into_keeps_the_aspect_ratio():
    r = fit_into(1000, 500, QRect(0, 0, 400, 400))
    assert (r.width(), r.height(), r.top()) == (400, 200, 100)


def test_source_marker_off_for_new_cameras_but_kept_for_old_sessions(tmp_path):
    assert XrayCamera(calibration_path=tmp_path / "c.txt", source_kind="folder", source=[]).show_source is False
    d = {"calibration_path": str(tmp_path / "c.txt"), "source_kind": "folder", "source": []}
    assert XrayCamera.from_dict(d).show_source is True        # saved before the setting existed: as it was
    assert XrayCamera.from_dict({**d, "show_source": False}).show_source is False
