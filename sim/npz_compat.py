"""Read and write npz files that numpy 1.24 can open.

``connectome_mb_v1.npz`` stores roles and other names as object arrays.
Those were pickled by numpy 2, whose unpickler asks for ``numpy._core``.
The robot has numpy 1.24.4 and cannot grow new packages. A pickle-free
copy keeps the same numbers and stores the strings as fixed ``<U`` arrays.
Older files still open: this module points ``numpy._core`` at ``numpy.core``
before pickle runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CORE_NAMES = (
    "multiarray",
    "umath",
    "_multiarray_umath",
    "numerictypes",
    "numeric",
    "fromnumeric",
    "overrides",
    "_internal",
    "einsumfunc",
    "shape_base",
)


def install_numpy2_pickle_alias() -> None:
    """Let numpy 1.x unpickle files written by numpy 2."""
    try:
        import numpy._core  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    import numpy.core as core

    sys.modules["numpy._core"] = core
    for name in _CORE_NAMES:
        dotted = "numpy._core." + name
        if dotted in sys.modules:
            continue
        mod = getattr(core, name, None)
        if mod is None:
            try:
                mod = __import__("numpy.core." + name, fromlist=[name])
            except Exception:
                continue
        sys.modules[dotted] = mod


def _needs_pickle(exc: BaseException) -> bool:
    text = str(exc)
    return "allow_pickle" in text or "Object arrays" in text


def open_npz(path: str | Path):
    """Load an npz. Object arrays use the numpy 2 alias. Others need no pickle."""
    dest = str(path)
    loaded = np.load(dest, allow_pickle=False)
    try:
        for key in loaded.files:
            loaded[key]
    except ValueError as exc:
        loaded.close()
        if not _needs_pickle(exc):
            raise
        install_numpy2_pickle_alias()
        return np.load(dest, allow_pickle=True)
    return loaded


def as_plain(value) -> np.ndarray:
    """Numeric array unchanged. Object array of names becomes ``<U``."""
    arr = np.array(value, copy=True)
    if arr.dtype != object:
        return arr
    flat = ["" if item is None else str(item) for item in arr.ravel()]
    width = 1
    for item in flat:
        if len(item) > width:
            width = len(item)
    return np.asarray(flat, dtype="<U%d" % width).reshape(arr.shape)


def write_pickle_free(src: str | Path, dest: str | Path) -> Path:
    """Copy every array into ``dest`` so ``allow_pickle=False`` can read it."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_npz(src) as loaded:
        payload = {key: as_plain(loaded[key]) for key in loaded.files}
    np.savez(out, **payload)
    return out


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == () or arr.size == 1:
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return ""
