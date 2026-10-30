"""Read-only checks for the private legacy input layout.

The release does not ship any input archive.  These checks only inspect the
caller-supplied paths with ``allow_pickle=False`` and return dimensions and
cryptographic hashes; they never write beside the inputs or emit identifiers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


DATA15_KEYS = ("x", "medical", "ci", "ids", "centers", "groups", "input_qc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_data15(path: str | Path) -> dict[str, Any]:
    """Validate the public schema of a private ``DATA15.npz`` archive."""

    data_path = Path(path).resolve()
    with np.load(data_path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        missing = [key for key in DATA15_KEYS if key not in archive]
        if missing:
            raise ValueError(f"DATA15 is missing required keys: {missing}")
        shapes = {key: list(np.asarray(archive[key]).shape) for key in DATA15_KEYS}
        dtypes = {key: str(np.asarray(archive[key]).dtype) for key in DATA15_KEYS}
        n = int(np.asarray(archive["x"]).shape[0])
        if shapes["x"] != [n, 4] or shapes["medical"] != [n, 4] or shapes["ci"] != [n]:
            raise ValueError(f"unexpected DATA15 row/feature shapes: {shapes}")
        if any(shapes[key] != [n] for key in ("ids", "centers", "groups", "input_qc")):
            raise ValueError(f"unexpected DATA15 vector shapes: {shapes}")
        if not np.isfinite(np.asarray(archive["x"], dtype=float)).all():
            raise ValueError("DATA15 x contains non-finite values")
    return {
        "path": str(data_path),
        "sha256": sha256(data_path),
        "keys": list(keys),
        "shapes": shapes,
        "dtypes": dtypes,
        "records": n,
        "allow_pickle": False,
        "read_only": True,
    }


def inspect_prefix(path: str | Path, *, expected_records: int = 480) -> dict[str, Any]:
    """Validate the private 5/10-minute prefix archive without identifiers."""

    prefix_path = Path(path).resolve()
    with np.load(prefix_path, allow_pickle=False) as archive:
        if "horizon_seconds" not in archive or "x" not in archive:
            raise ValueError("PREFIX_X.npz must contain horizon_seconds and x")
        horizons = np.asarray(archive["horizon_seconds"])
        x = np.asarray(archive["x"])
        if horizons.tolist() != [300, 600]:
            raise ValueError(f"unexpected prefix horizons: {horizons.tolist()}")
        if x.shape != (2, expected_records, 4):
            raise ValueError(f"unexpected prefix shape: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("PREFIX_X.npz contains non-finite values")
    return {
        "path": str(prefix_path),
        "sha256": sha256(prefix_path),
        "horizons_seconds": [int(v) for v in horizons.tolist()],
        "shape": list(x.shape),
        "allow_pickle": False,
        "read_only": True,
    }
