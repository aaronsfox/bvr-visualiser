"""Radiograph image sequences: a folder, a list of files, or one multi-page TIFF."""

from __future__ import annotations

import re
from collections import OrderedDict
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
    """Lazily-read, cached sequence of radiographs, returned as 8-bit grayscale."""

    def __init__(self, files: list[Path] | None = None, multipage: Path | None = None, cache: int = 48):
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
        first = self._raw(0)
        self.height, self.width = first.shape[:2]
        self.dtype = str(first.dtype)
        self.channels = 1 if first.ndim == 2 else first.shape[2]

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

    def get(self, i: int) -> np.ndarray:
        """8-bit grayscale image i. 16-bit images use one contrast window for the whole sequence."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        img = _to_gray(self._raw(i))
        if img.dtype != np.uint8:
            f = img.astype(np.float32)
            if self._window is None:
                lo, hi = np.percentile(f, [0.5, 99.5])
                self._window = (float(lo), float(max(hi, lo + 1)))
            lo, hi = self._window
            img = np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        self._cache[i] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def describe(self) -> str:
        bits = "8-bit" if self.dtype == "uint8" else f"{self.dtype}"
        col = "RGB" if self.channels >= 3 else "grayscale"
        return f"{bits} {col} · {self.width} × {self.height}"


def adjust(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    """Brightness in [-1, 1], contrast as a multiplier about mid-grey."""
    f = img.astype(np.float32)
    f = (f - 127.5) * contrast + 127.5 + brightness * 255
    if invert:
        f = 255 - f
    return np.clip(f, 0, 255).astype(np.uint8)
