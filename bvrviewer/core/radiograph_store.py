"""Radiographs kept for later: one compressed multi-page TIFF of display copies per camera.

The same file format serves two purposes (Visualiser #13):
- the **local cache**: radiographs that were slow to read (e.g. from a network drive) are saved on this
  computer after they've been read, so opening the same trial again takes seconds;
- **saving radiographs with a session**: the copies go in the session's data folder, so the session
  opens quickly anywhere, even without the original images.

Each page is one display copy (8-bit, shrunk as shown in the app; zlib-compressed). The first page's
description holds JSON metadata: the original images' size and type, the shrink step, the sources
and a fingerprint (name, size and modified time of each original file) to tell whether the originals
have changed since.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

STORE_VERSION = 1
SUFFIX = ".bvrimages.tif"


class StoreError(ValueError):
    pass


# ---------------------------------------------------------------- fingerprints
def fingerprint(paths) -> list:
    """[name, size, modified second] of each distinct file, listing each folder once.

    Folders are listed with os.scandir, which on Windows returns sizes and times with the listing, so a
    network folder of 500 images costs one round trip rather than 500.
    """
    by_dir: dict[Path, list[str]] = {}
    order = []
    for p in paths:
        p = Path(p)
        if p not in order:
            order.append(p)
            by_dir.setdefault(p.parent, []).append(p.name)
    info = {}
    for d, names in by_dir.items():
        wanted = set(names)
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name in wanted:
                        st = e.stat()
                        info[d / e.name] = (st.st_size, int(st.st_mtime))
        except OSError:
            pass
    return [[p.name, *info.get(p, (None, None))] for p in order]


def _files(seq) -> list[Path]:
    return [p for p, _ in seq.items]


# ---------------------------------------------------------------- writing
def write_store(seq, path: str | Path, progress=None, cancelled=None) -> Path:
    """Save every display copy of an ImageSequence (reading any not yet in memory) to `path`.

    progress(i, n) is called after each image; cancelled() returning True stops early (no file).
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(seq)
    meta = {"bvr_radiographs": STORE_VERSION, "n": n, "width": seq.width, "height": seq.height,
            "dtype": seq.dtype, "channels": seq.channels, "display_step": seq.display_step,
            "source": str(seq.source), "multipage": seq.items[0][1] is not None,
            "names": [p.name for p, _ in seq.items], "pages": [pg for _, pg in seq.items],
            "fingerprint": seq.fingerprint(), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path.with_name(path.name + ".partial")
    try:
        with tifffile.TiffWriter(tmp) as tw:
            for i in range(n):
                if cancelled is not None and cancelled():
                    raise InterruptedError("Saving the radiographs was cancelled.")
                tw.write(seq.get_display(i), compression="zlib", compressionargs={"level": 1},
                         description=json.dumps(meta) if i == 0 else None, metadata=None)
                if progress is not None:
                    progress(i + 1, n)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


# ---------------------------------------------------------------- reading
def read_meta(path: str | Path) -> dict:
    import tifffile

    try:
        with tifffile.TiffFile(path) as tf:
            desc = tf.pages[0].description
            n = len(tf.pages)
        meta = json.loads(desc)
    except Exception as e:  # noqa: BLE001 - any unreadable file is reported the same way
        raise StoreError(f"{Path(path).name} isn't a saved radiograph file ({e}).") from None
    if not isinstance(meta, dict) or "bvr_radiographs" not in meta:
        raise StoreError(f"{Path(path).name} isn't a saved radiograph file.")
    if meta["n"] != n:
        raise StoreError(f"{Path(path).name} is incomplete ({n} of {meta['n']} images).")
    return meta


class StoreReader:
    """Reads pages of a saved radiograph file; safe to use from several threads."""

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = meta or read_meta(path)
        self._local = threading.local()
        self._handles = []
        self._lock = threading.Lock()

    def _tf(self):
        tf = getattr(self._local, "tf", None)
        if tf is None:
            import tifffile

            tf = tifffile.TiffFile(self.path)
            self._local.tf = tf
            with self._lock:
                self._handles.append(tf)
        return tf

    def page(self, i: int) -> np.ndarray:
        return self._tf().pages[i].asarray()

    def page_bytes(self, i: int) -> int:
        try:
            return int(sum(self._tf().pages[i].databytecounts))
        except Exception:  # noqa: BLE001
            return 0

    def close(self) -> None:
        with self._lock:
            hs, self._handles = self._handles, []
        for tf in hs:
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass


def matches(meta: dict, seq) -> bool:
    """Whether a saved file holds these original images, unchanged."""
    return (meta.get("n") == len(seq) and meta.get("width") == seq.width and meta.get("height") == seq.height
            and meta.get("fingerprint") == seq.fingerprint())


# ---------------------------------------------------------------- local cache
def cache_dir() -> Path:
    """Where cached radiographs go (BVR_CACHE_DIR overrides)."""
    if os.environ.get("BVR_CACHE_DIR"):
        return Path(os.environ["BVR_CACHE_DIR"])
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BVR Motion Viewer"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "BVR Motion Viewer"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "bvr-motion-viewer"
    return base / "radiographs"


def cache_key(seq) -> str:
    src = [[str(p), pg] for p, pg in seq.items]
    return hashlib.sha1(json.dumps([src, seq.display_step]).encode()).hexdigest()[:20]


def cache_path(seq) -> Path:
    return cache_dir() / f"{cache_key(seq)}{SUFFIX}"


def find_cached(seq) -> Path | None:
    """The cached copy of these radiographs, if there's one and the originals haven't changed."""
    p = cache_path(seq)
    if not p.exists():
        return None
    try:
        ok = matches(read_meta(p), seq)
    except StoreError:
        ok = False
    if not ok:
        try:
            p.unlink()                     # out of date or broken
        except OSError:
            pass
        return None
    try:
        os.utime(p)                        # most recently used: kept longest
    except OSError:
        pass
    return p


def save_to_cache(seq, limit_bytes: int | None = None, cancelled=None) -> Path:
    p = write_store(seq, cache_path(seq), cancelled=cancelled)
    if limit_bytes is not None:
        prune_cache(limit_bytes, keep=p)
    return p


def cache_files() -> list[Path]:
    d = cache_dir()
    return sorted(d.glob(f"*{SUFFIX}")) if d.is_dir() else []


def cache_size() -> int:
    return sum(p.stat().st_size for p in cache_files() if p.exists())


def prune_cache(limit_bytes: int, keep: Path | None = None) -> None:
    """Delete the least recently used cached trials until the cache is within limit_bytes."""
    files = sorted(cache_files(), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= limit_bytes:
            break
        if keep is not None and p == keep:
            continue
        total -= p.stat().st_size
        try:
            p.unlink()
        except OSError:
            pass
    for p in cache_dir().glob("*.partial") if cache_dir().is_dir() else []:
        if time.time() - p.stat().st_mtime > 3600:      # left by a write that was interrupted
            p.unlink(missing_ok=True)


def clear_cache() -> None:
    for p in cache_files():
        try:
            p.unlink()
        except OSError:
            pass


def is_network_path(path: str | Path) -> bool:
    """Best guess at whether a file is on a network drive (UNC path, mapped drive, NFS / SMB mount)."""
    s = str(path)
    if s.startswith(("\\\\", "//")):
        return True
    if sys.platform.startswith("win"):
        try:
            import ctypes

            drive = os.path.splitdrive(os.path.abspath(s))[0] + "\\"
            return ctypes.windll.kernel32.GetDriveTypeW(drive) == 4         # DRIVE_REMOTE
        except Exception:  # noqa: BLE001
            return False
    try:
        real = os.path.realpath(s)
        best, fstype = "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and (real == parts[1] or real.startswith(parts[1].rstrip("/") + "/")) \
                        and len(parts[1]) > len(best):
                    best, fstype = parts[1], parts[2]
        return fstype.startswith(("nfs", "cifs", "smb", "fuse.sshfs", "9p", "afs"))
    except OSError:
        pass
    if sys.platform == "darwin":
        import subprocess

        try:
            out = subprocess.run(["df", "-P", s], capture_output=True, text=True, timeout=3, check=False).stdout.splitlines()
            src = out[-1].split()[0] if len(out) > 1 else ""
            return src.startswith("//") or ":" in src
        except Exception:  # noqa: BLE001
            return False
    return False
