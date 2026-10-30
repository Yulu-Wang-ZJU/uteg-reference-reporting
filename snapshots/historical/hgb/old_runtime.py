"""Run-local runtime for the MA tail-balanced gate round.

This module is intentionally small and write-confined.  It establishes all
cache locations before importing numerical libraries, exposes read-only
historical paths, and provides fail-closed JSON/ledger helpers.  It does not
import the old training runtime or any test evaluator.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve()
ROOT = HERE.parent
GOAL_ROOT = ROOT.parent
FROZEN_ROOT = GOAL_ROOT / "ma_tail_weight_gate_dedup_20261026_140315"
BASE_ROOT = GOAL_ROOT / "ridge10_fullchain_20261026_122551"
DATA_ROOT = ROOT.parents[1] / "uteg_legacy_aligned_threecenter_20261022_155614"
DATA_PATH = DATA_ROOT / "DATA15.npz"
ROLE_PATH = DATA_ROOT / "ROLE_FREEZE.json"

CACHE = ROOT / "cache"
for _name in ("tmp", "joblib", "numba", "torch", "hf", "mpl", "xdg", "pycache"):
    (CACHE / _name).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "TEMP": str(CACHE / "tmp"),
    "TMP": str(CACHE / "tmp"),
    "TMPDIR": str(CACHE / "tmp"),
    "PYTHONPYCACHEPREFIX": str(CACHE / "pycache"),
    "JOBLIB_TEMP_FOLDER": str(CACHE / "joblib"),
    "NUMBA_CACHE_DIR": str(CACHE / "numba"),
    "TORCH_HOME": str(CACHE / "torch"),
    "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "torch"),
    "HF_HOME": str(CACHE / "hf"),
    "MPLCONFIGDIR": str(CACHE / "mpl"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
})
sys.dont_write_bytecode = True

SCIENCE_CAP = 720
REPAIR_CAP = 40
TOTAL_CAP = 760
AGREEMENT_MIN_STRICT = 0.75
COVERAGE_MIN_STRICT = 0.20
SEEDS = tuple(range(1000, 1020))
DIRECTIONS = (
    "HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB",
    "HZ_to_SA", "HB_to_SA", "SA_to_HZ", "HB_to_HZ", "SA_to_HB", "HZ_to_HB",
)
TARGET_BY_DIRECTION = {
    "HZ+HB_to_SA": "SAHZU", "SA+HB_to_HZ": "Huzhou", "SA+HZ_to_HB": "Hubei",
    "HZ_to_SA": "SAHZU", "HB_to_SA": "SAHZU", "SA_to_HZ": "Huzhou",
    "HB_to_HZ": "Huzhou", "SA_to_HB": "Hubei", "HZ_to_HB": "Hubei",
}
MA_CLASSES = (-1, 0, 1)
LEDGER_DIR = ROOT / "ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
LOCK_PATH = ROOT / "RUNNER.lock"


def json_clean(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if np is not None and isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if np is not None and isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any) -> Path:
    p = abs_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_object_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def frozen_hashes() -> dict[str, str]:
    names = ("EXECUTION_PROMPT_ZH.md", "PROTOCOL.md", "STATE.json", "PROTOCOL_REVIEW_RECEIPT.json")
    return {name: sha256(FROZEN_ROOT / name) for name in names}


def base_hashes() -> dict[str, str]:
    names = ("MODEL_IDENTITY.json", "FULL_MODELS_FROZEN.json", "THRESHOLD_FREEZE.json",
             "OUTER_B_THRESHOLD_CONTRACT.json")
    return {name: sha256(BASE_ROOT / name) for name in names}


def load_metadata_only_data() -> dict[str, Any]:
    """Read only x/ids/centers/QC from DATA15; never access ``medical``."""
    import numpy as np

    with np.load(DATA_PATH, allow_pickle=False) as z:
        required = {"x", "ids", "centers", "input_qc"}
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(f"DATA15 missing non-medical keys: {missing}")
        return {key: np.asarray(z[key]) for key in required}


def acquire_lock(label: str) -> None:
    if LOCK_PATH.exists():
        raise RuntimeError(f"run lock exists; refusing concurrent execution: {LOCK_PATH}")
    LOCK_PATH.write_text(json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}, indent=2), encoding="utf-8")


def release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _ledger_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LEDGER_DIR.glob("*.json")):
        try:
            row = read_json(path)
        except Exception as exc:
            raise RuntimeError(f"corrupt ledger row: {path}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"ledger row is not an object: {path}")
        if row.get("status") not in {"PENDING", "RUNNING", "COMPLETE", "FAILED"}:
            raise RuntimeError(f"invalid ledger row status: {path}")
        if not isinstance(row.get("units"), int) or int(row["units"]) != 1:
            raise RuntimeError(f"candidate unit must be exactly one fit call: {path}")
        if row.get("kind") not in {"science", "repair", "check"}:
            raise RuntimeError(f"invalid ledger row kind: {path}")
        rows.append(row)
    return rows


def ledger_totals() -> dict[str, int]:
    rows = _ledger_rows()
    science = sum(int(r["units"]) for r in rows if r["kind"] == "science")
    repair = sum(int(r["units"]) for r in rows if r["kind"] in {"repair", "check"})
    return {"science": science, "repair": repair, "total": science + repair,
            "running": sum(r["status"] == "RUNNING" for r in rows)}


def update_ledger_summary(**extra: Any) -> dict[str, Any]:
    path = ROOT / "FIT_LEDGER.json"
    ledger = read_json(path)
    totals = ledger_totals()
    ledger.update({"science_units": totals["science"], "repair_units": totals["repair"],
                   "total_units": totals["total"], "fit_started": totals["total"] > 0})
    ledger.update(json_clean(extra))
    write_json(path, ledger)
    return ledger


def register_attempt(key: str, kind: str, *, scope: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Persist one unit before an optimizer call; duplicate keys fail closed."""
    if kind not in {"science", "repair", "check"}:
        raise ValueError(kind)
    totals = ledger_totals()
    if kind == "science" and totals["science"] >= SCIENCE_CAP:
        raise RuntimeError("science cap reached")
    if kind in {"repair", "check"} and totals["repair"] >= REPAIR_CAP:
        raise RuntimeError("repair/check cap reached")
    if totals["total"] >= TOTAL_CAP:
        raise RuntimeError("hard cap reached")
    safe_key = str(key).replace("/", "_").replace("\\", "_")
    path = LEDGER_DIR / f"{safe_key}.json"
    if path.exists():
        raise RuntimeError(f"duplicate attempt key; explicit retry key required: {key}")
    row = {"schema": "ma_tail_gate_attempt_v1", "key": key, "kind": kind, "scope": scope,
           "units": 1, "status": "PENDING", "pid": os.getpid(), "registered_at": time.time(),
           "metadata": json_clean(dict(metadata or {}))}
    write_json(path, row)
    update_ledger_summary(current_phase="FIT_ATTEMPT_REGISTERED", status="FIT_ATTEMPT_PENDING")
    return row


def begin_attempt(key: str) -> dict[str, Any]:
    """Transition a persisted PENDING unit to RUNNING immediately pre-fit."""
    path = LEDGER_DIR / f"{str(key).replace('/', '_').replace(chr(92), '_')}.json"
    if not path.exists():
        raise RuntimeError(f"attempt key was not persisted before optimizer: {key}")
    row = read_json(path)
    if row.get("status") != "PENDING":
        raise RuntimeError(f"attempt must be PENDING before optimizer: {key} status={row.get('status')}")
    row.update({"status": "RUNNING", "started_at": time.time()})
    write_json(path, row)
    update_ledger_summary(current_phase="OPTIMIZER_RUNNING", status="FIT_IN_PROGRESS")
    return row


def finish_attempt(key: str, status: str, **extra: Any) -> dict[str, Any]:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(status)
    path = LEDGER_DIR / f"{str(key).replace('/', '_').replace(chr(92), '_')}.json"
    row = read_json(path)
    if row.get("status") != "RUNNING":
        raise RuntimeError(f"attempt is not running: {key}")
    row.update({"status": status, "finished_at": time.time(), **json_clean(extra)})
    write_json(path, row)
    update_ledger_summary(current_phase="FIT_AVAILABLE_FOR_REVIEW" if status == "COMPLETE" else "FIT_FAILURE_RECORDED")
    return row

