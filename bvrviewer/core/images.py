"""Radiograph image sequences: a folder, a list of files, or one multi-page TIFF."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_LAST_NUMBER = re.compile(r"(\d+)(?!.*\d)")


class ImageSequenceError(ValueError):
    pass


def frame_number(path: str | Path) -> int | None:
    """The last number in a file name (e.g. 195 for 'xr_cam01_UND.0195.tif')."""
    m = _LAST_NUMBER.search(Path(path).stem)
    return int(m.group(1)) if m else None


def _natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def find_images(folder: str | Path) -> tuple[list[Path], list[str]]:
    """Image files in a folder ordered by their numbers, plus names of files that were skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ImageSequenceError(f"{folder} isn't a folder.")
    images, skipped = [], []
    for p in folder.iterdir():
        if p.is_dir() or p.name.startswith("."):
            continue
        (images if p.suffix.lower() in IMAGE_EXTS else skipped).append(p)
    if not images:
        raise ImageSequenceError("No image files (.tif, .png, .jpg, .bmp) found in this folder.")
    images.sort(key=_natural_key)
    return images, sorted(s.name for s in skipped)


def _read_any(path: Path, page: int | None = None) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        if page is None:
            return tifffile.imread(path)
        with tifffile.TiffFile(path) as tf:
            return tf.pages[page].asarray()
    import imageio.v3 as iio

    return iio.imread(path)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] >= 3:
            if np.array_equal(img[..., 0], img[..., 1]):
                return img[..., 0]
            return (img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
        return img[..., 0]
    return img


class ImageSequence:
    """Lazily-read sequence of radiographs, returned as 8-bit grayscale.

    `get(i)` gives the full-resolution image. `get_display(i)` gives a copy shrunk so its longest
    side is at most `display_max` pixels, which is what the 3D view shows. Display images are
    decoded ahead of time on background threads (`prefetch`) and kept in a memory-limited cache,
    because decoding compressed TIFFs is far too slow to do while playing.
    """

    def __init__(self, files: list[Path] | None = None, multipage: Path | None = None, cache: int = 8,
                 display_max: int = 1024, display_budget_mb: int = 400):
        if multipage is not None:
            import tifffile

            with tifffile.TiffFile(multipage) as tf:
                n = len(tf.pages)
            if n < 2:
                raise ImageSequenceError("This TIFF has only one page.")
            self.items = [(Path(multipage), i) for i in range(n)]
            self.source = Path(multipage)
        else:
            if not files:
                raise ImageSequenceError("No images given.")
            self.items = [(Path(f), None) for f in files]
            self.source = Path(files[0]).parent
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache
        self._window: tuple[float, float] | None = None
        self._lock = threading.Lock()
        first = self._raw(0)
        self.height, self.width = first.shape[:2]
        self.dtype = str(first.dtype)
        self.channels = 1 if first.ndim == 2 else first.shape[2]
        self.display_step = max(1, -(-max(self.width, self.height) // display_max))   # ceiling division
        frame_bytes = -(-self.width // self.display_step) * -(-self.height // self.display_step)
        self._display: OrderedDict[int, np.ndarray] = OrderedDict()
        self._display_limit = max(16, display_budget_mb * 1_000_000 // max(frame_bytes, 1))
        self._pending: dict[int, Future] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._set_window(first)

    @classmethod
    def from_folder(cls, folder: str | Path) -> "ImageSequence":
        files, _ = find_images(folder)
        return cls(files=files)

    def __len__(self) -> int:
        return len(self.items)

    def name(self, i: int) -> str:
        p, page = self.items[i]
        return p.name if page is None else f"{p.name} [page {page + 1}]"

    def label(self, i: int) -> str:
        """Short label: the file's frame number if it has one, else its position."""
        p, page = self.items[i]
        if page is None:
            n = frame_number(p)
            if n is not None:
                m = _LAST_NUMBER.search(p.stem)
                return m.group(1)
        return str(i + 1)

    def _raw(self, i: int) -> np.ndarray:
        p, page = self.items[i]
        return _read_any(p, page)

    def _set_window(self, first: np.ndarray) -> None:
        """16-bit images use one contrast window, from the first image, for the whole sequence."""
        g = _to_gray(first)
        if g.dtype != np.uint8:
            lo, hi = np.percentile(g.astype(np.float32), [0.5, 99.5])
            self._window = (float(lo), float(max(hi, lo + 1)))

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        img = _to_gray(img)
        if img.dtype == np.uint8:
            return img
        lo, hi = self._window or (float(img.min()), float(img.max()) + 1)
        return np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    def get(self, i: int) -> np.ndarray:
        """Full-resolution 8-bit grayscale image i."""
        with self._lock:
            if i in self._cache:
                self._cache.move_to_end(i)
                return self._cache[i]
        img = self._to_uint8(self._raw(i))
        with self._lock:
            self._cache[i] = img
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return img

    # ---- display copies, decoded in the background
    def _load_display(self, i: int) -> np.ndarray:
        s = self.display_step
        img = np.ascontiguousarray(self._to_uint8(self._raw(i))[::s, ::s])
        with self._lock:
            self._display[i] = img
            self._pending.pop(i, None)
            while len(self._display) > self._display_limit:
                self._display.popitem(last=False)
        return img

    def get_display(self, i: int) -> np.ndarray:
        """Image i shrunk for display. Waits for a background decode if one is under way."""
        with self._lock:
            if i in self._display:
                self._display.move_to_end(i)
                return self._display[i]
            fut = self._pending.get(i)
        if fut is not None:
            try:
                return fut.result()
            except Exception:  # noqa: BLE001 - fall back to decoding here
                pass
        return self._load_display(i)

    def has_display(self, i: int) -> bool:
        with self._lock:
            return i in self._display

    def prefetch(self, indices) -> None:
        """Start decoding these display images in the background (already cached ones are skipped)."""
        if self._closed:
            return
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="radiographs")
            todo = [i for i in indices if 0 <= i < len(self.items) and i not in self._display
                    and i not in self._pending][: self._display_limit]
            for i in todo:
                self._pending[i] = self._pool.submit(self._load_display, i)

    def close(self) -> None:
        """Stop background decoding (call when the camera is removed)."""
        self._closed = True
        with self._lock:
            pool, self._pool = self._pool, None
            self._pending.clear()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def describe(self) -> str:
        bits = "8-bit" if self.dtype == "uint8" else f"{self.dtype}"
        col = "RGB" if self.channels >= 3 else "grayscale"
        return f"{bits} {col} · {self.width} × {self.height}"


def adjust_lut(brightness: float = 0.0, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    """256-entry lookup table: brightness in [-1, 1], contrast as a multiplier about mid-grey."""
    f = (np.arange(256, dtype=np.float32) - 127.5) * contrast + 127.5 + brightness * 255
    if invert:
        f = 255 - f
    return np.clip(f, 0, 255).astype(np.uint8)


def adjust(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    """Apply brightness/contrast/invert to an 8-bit image (via a lookup table, so it's fast)."""
    return adjust_lut(brightness, contrast, invert)[img]
