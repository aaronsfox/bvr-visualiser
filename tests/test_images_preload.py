"""Radiographs read up front (Visualiser #12): every display copy is read in the background and kept,
a queued frame can be shown without waiting for the queue, and the read can be skipped.

Reads are slowed down here to stand in for a network drive.
"""

import time

import numpy as np
import pytest
import tifffile

from bvrviewer.core.images import ImageSequence


@pytest.fixture
def folder(tmp_path):
    for i in range(60):
        tifffile.imwrite(tmp_path / f"img.{i:04d}.tif", np.full((200, 300), i, np.uint8))
    return tmp_path


def slow(seq, delay=0.02):
    raw = seq._raw

    def _raw(i):
        time.sleep(delay)
        return raw(i)

    seq._raw = _raw
    return seq


def test_preload_reads_every_image(folder):
    seq = slow(ImageSequence.from_folder(folder))
    seq.preload()
    assert seq.progress()[2] > 0
    t0 = time.monotonic()
    while seq.progress()[2] and time.monotonic() - t0 < 10:
        time.sleep(0.01)
    assert seq.progress() == (60, 0, 0)
    assert seq.bytes_read > 60 * 200 * 300
    assert all(seq.has_display(i) for i in range(60))
    assert seq.get_display(37)[0, 0] == 37
    seq.close()


def test_a_queued_frame_is_read_straight_away(folder):
    seq = slow(ImageSequence.from_folder(folder), delay=0.05)
    seq.preload()
    t0 = time.monotonic()
    img = seq.get_display(59)              # last in the queue: ~60 × 0.05 / 4 s if it waited its turn
    assert img[0, 0] == 59
    assert time.monotonic() - t0 < 0.3
    seq.close()


def test_skip_drops_the_queue(folder):
    seq = slow(ImageSequence.from_folder(folder), delay=0.05)
    seq.preload()
    seq.cancel_preload()
    time.sleep(0.2)
    read, _, left = seq.progress()
    assert left == 0 and read < 60
    assert seq.get_display(50)[0, 0] == 50     # read when shown
    seq.close()


def test_unreadable_images_are_reported(folder):
    (folder / "img.0007.tif").write_bytes(b"not a tiff")
    seq = ImageSequence.from_folder(folder)
    seq.preload()
    t0 = time.monotonic()
    while seq.progress()[2] and time.monotonic() - t0 < 10:
        time.sleep(0.01)
    assert seq.progress()[:2] == (59, 1) and 7 in seq.failed
    seq.close()


def test_display_copies_shrink_to_fit_the_memory_budget(folder):
    seq = ImageSequence.from_folder(folder)
    assert seq.display_step == 1 and seq.fits_in_memory
    small = ImageSequence.from_folder(folder)
    small.__init__(files=[p for p, _ in small.items], display_budget_mb=1)     # 60 × 60 kB > 1 MB
    assert small.display_step == 2 and small.display_size == (150, 100) and small.fits_in_memory
