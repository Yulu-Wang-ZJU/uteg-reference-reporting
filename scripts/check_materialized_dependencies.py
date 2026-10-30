#!/usr/bin/env python3
"""Import the materialized historical dependency closure without fitting.

Each probe runs in a fresh Python process so that the historical modules named
``runtime`` cannot collide with one another.  Importing a module defines the
source functions and loads only code; this command never calls a training or
prediction entrypoint and rejects private-data/model file extensions in the
materialized tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROBES = (
    (
        "build_prefix_inputs",
        "outputs/uteg_legacy_aligned_threecenter_20261022_155614/build_prefix_inputs.py",
    ),
    (
        "prefix_source",
        "outputs/uteg_legacy_aligned_threecenter_20261022_155614/prefix_source.py",
    ),
    (
        "evaluate_prefix",
        "outputs/uteg_legacy_aligned_threecenter_20261022_155614/evaluate_prefix.py",
    ),
    (
        "platform_process",
        "outputs/uteg_platform_filter_pilot_20261006_124636/process.py",
    ),
    (
        "platform_curves",
        "outputs/uteg_platform_param_compare_20261006_125428/curves.py",
    ),
    (
        "zero5_refit",
        "outputs/uteg_zero5_med4_20261006_113257/refit.py",
    ),
    (
        "fit_ma_gates",
        "outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py",
    ),
    (
        "trust_threshold_core",
        "outputs/uteg_trust_threshold90_20261006_161248/core.py",
    ),
    (
        "scalar_ordinal_core",
        "outputs/uteg_scalar_ordinal_trust_20261006_181750/core.py",
    ),
    (
        "d2_ci_engine",
        "outputs/uteg_d2_ci_native85_20261007_012322/engine.py",
    ),
)

# This historical QC script executes ``np.load`` at module import time.  It
# remains an exact snapshot, but is intentionally not imported by this guard:
# importing it would require its private STANDARDIZED_INPUT.npz before any
# function could be selected.
SKIPPED_DATA_EAGER = (
    "outputs/uteg_hubei_qc_repair_20261021_171418/run.py",
)

FORBIDDEN_SUFFIXES = {
    ".npz",
    ".npy",
    ".joblib",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
}

IMPORT_CODE = r"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["UTEG_NOFIT_PROBE"] = "1"
sys.dont_write_bytecode = True
path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location(sys.argv[2], path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to create import spec for {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("IMPORT_OK", sys.argv[2])
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", required=True, type=Path)
    args = parser.parse_args()
    target = args.target_root.expanduser().resolve()
    if not target.is_dir():
        raise SystemExit(f"materialized target is not a directory: {target}")

    forbidden = [
        str(path.relative_to(target))
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise SystemExit(
            "private/model files are present in the materialized code root: "
            + ", ".join(forbidden)
        )

    results: list[dict[str, object]] = []
    for name, relative in PROBES:
        path = target / Path(*Path(relative).parts)
        if not path.is_file():
            results.append({"name": name, "status": "MISSING", "path": relative})
            continue
        env = os.environ.copy()
        env["UTEG_NOFIT_PROBE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", IMPORT_CODE, str(path), name],
            cwd=str(target),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        result: dict[str, object] = {
            "name": name,
            "path": relative,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
        }
        if completed.stdout.strip():
            result["stdout_tail"] = completed.stdout.strip().splitlines()[-4:]
        if completed.stderr.strip():
            result["stderr_tail"] = completed.stderr.strip().splitlines()[-8:]
        results.append(result)

    failures = [item for item in results if item["status"] != "PASS"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "target_root": str(target),
        "fit_or_private_data_access": False,
        "skipped_data_eager_modules": list(SKIPPED_DATA_EAGER),
        "probes": results,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
