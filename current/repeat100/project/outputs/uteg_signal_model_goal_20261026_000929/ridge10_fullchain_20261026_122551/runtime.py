"""Task-local runtime for the fixed Ridge-10 fullchain round.

Historical definitions are imported through this shim only after all cache and
thread variables point into this run directory.  No historical runtime or
ledger is imported for writing.
"""
from __future__ import annotations

import ast
import atexit
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
ROOT = HERE.parent
OUT = ROOT.parent
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

# Read-only historical dependency roots.  They are deliberately outside ROOT.
BASE = OUT.parent / "uteg_legacy_aligned_threecenter_20261022_155614"
EXP = OUT.parent / "uteg_hill_exp_matched_20261012_153337"
NINE = OUT.parent / "uteg_nine_direction_ma_20261025_195548"
OLD = OUT.parent / "uteg_med4_early_decision_20261004_191744"
CAP = OUT.parent / "uteg_med4_capacity_20261006_001701"
CLIN = OUT.parent / "uteg_med4_source_clinical_20261006_104632"
MAP = OUT.parent / "uteg_fourparam_mapping_trust_20261006_132600"
ZERO = OUT.parent / "uteg_zero5_med4_20261006_113257"
PLAT = OUT.parent / "uteg_platform_param_compare_20261006_125428"
SC = OUT.parent / "uteg_scalar_ordinal_trust_20261006_181750"
TH = OUT.parent / "uteg_trust_threshold90_20261006_161248"
GATE = OUT.parent / "uteg_trust_gate_selection_20261004_205412"
D1 = OUT.parent / "uteg_cohort399_gate_revalidation_20261006_203700"
D2 = OUT.parent / "uteg_c60_new_trust_20261007_170322"
AD = OUT.parent / "uteg_d2_ci_native85_20261007_012322"

SCIENCE_FIT_CAP = 12720
TOTAL_FIT_CAP = 13000
REPAIR_FIT_CAP = 280
NONSCIENCE_KINDS = {"repair", "audit", "test"}
CURRENT_SCOPE = ""
CURRENT_REPAIR = False
LOCK_PATH = ROOT / "RUNNER.lock"


def _json_clean(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if np is not None and isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if np is not None and isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write(name: str | Path, value: Any) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_clean(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract(path: str | Path, names: list[str], namespace: dict[str, Any]) -> None:
    """AST-extract definitions from a historical source file."""
    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names):
        raise RuntimeError(f"missing definitions in {source_path}: {names}")
    namespace.setdefault("__name__", "ridge10_fullchain_definitions")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)


def load(name: str, path: str | Path):
    source_path = Path(path)
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_scope(scope: str, repair: bool = False) -> None:
    global CURRENT_SCOPE, CURRENT_REPAIR
    CURRENT_SCOPE = str(scope).replace("/", "_").replace("\\", "_")
    CURRENT_REPAIR = bool(repair)


def _ledger_rows() -> list[dict[str, Any]]:
    rows = []
    for p in sorted((ROOT / "ledger").glob("*.json")):
        try:
            row = read(p)
        except Exception as exc:
            raise RuntimeError(f"corrupt ledger row: {p}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"corrupt ledger row (not an object): {p}")
        if not isinstance(row.get("units"), int) or row["units"] <= 0:
            raise RuntimeError(f"corrupt ledger row (invalid units): {p}")
        if row.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
            raise RuntimeError(f"corrupt ledger row (invalid status): {p}")
        if not isinstance(row.get("kind"), str) or not row["kind"]:
            raise RuntimeError(f"corrupt ledger row (invalid kind): {p}")
        rows.append(row)
    return rows


def _totals() -> dict[str, int]:
    out = {"science": 0, "repair": 0, "total": 0}
    for row in _ledger_rows():
        n = int(row.get("units", 0))
        k = str(row.get("kind", ""))
        out["total"] += n
        if k in NONSCIENCE_KINDS:
            if k in {"repair", "audit"}:
                out["repair"] += n
        else:
            out["science"] += n
    return out


_LEDGER_TOTALS = _totals()


def register_attempt(key: str, kind: str, units: int = 1, **meta: Any) -> Path:
    """Persist an attempt before the corresponding numerical fit."""
    scope = f"{CURRENT_SCOPE}__" if CURRENT_SCOPE else ""
    path = ROOT / "ledger" / f"{scope}{key}.json"
    if path.exists():
        raise RuntimeError(f"attempt key already exists; use a distinct retry key: {path.name}")
    totals = dict(_LEDGER_TOTALS)
    amount = int(units)
    k = "repair" if CURRENT_REPAIR else str(kind)
    next_total = totals["total"] + amount
    next_science = totals["science"] + (0 if k in NONSCIENCE_KINDS else amount)
    next_repair = totals["repair"] + (amount if k in {"repair", "audit"} else 0)
    if next_total > TOTAL_FIT_CAP or next_science > SCIENCE_FIT_CAP or next_repair > REPAIR_FIT_CAP:
        raise RuntimeError(f"fit cap exceeded before attempt: {totals} + {amount} {k}")
    payload = {
        "status": "RUNNING", "kind": k, "requested_kind": str(kind), "units": amount,
        "scope": CURRENT_SCOPE, "started": time.time(), **meta,
    }
    write(path.relative_to(ROOT), payload)
    _LEDGER_TOTALS.update({"science": next_science, "repair": next_repair, "total": next_total})
    return path


def finish_attempt(path: Path, status: str, **meta: Any) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(status)
    row = read(path)
    row.update(status=status, finished=time.time(), **meta)
    write(path.relative_to(ROOT), row)


def finish_scope(scope: str, status: str, **meta: Any) -> None:
    """Commit or fail all attempt rows belonging to one outer target fit."""
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(status)
    for path in sorted((ROOT / "ledger").glob(f"{scope}__*.json")):
        row = read(path)
        if row.get("status") == "RUNNING":
            row.update(status=status, finished=time.time(), **meta)
            write(path.relative_to(ROOT), row)


def ledger(key: str, kind: str, units: int = 1, **metadata: Any) -> None:
    """Compatibility hook for legacy modules.

    The legacy fitter calls this immediately before a numerical fit and has no
    success callback.  Leave the row RUNNING until the outer runner commits
    the enclosing target object; this preserves a truthful interrupted-run
    marker without allowing duplicate keys.
    """
    scope = f"{CURRENT_SCOPE}__" if CURRENT_SCOPE else ""
    path = ROOT / "ledger" / f"{scope}{key}.json"
    if path.exists():
        raise RuntimeError(f"legacy ledger key already exists; retry namespace required: {path.name}")
    register_attempt(key, kind, units, legacy_hook=True, **metadata)


def guard() -> dict[str, int]:
    size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    free = shutil.disk_usage(ROOT).free
    if size > 2 * 2**30:
        raise RuntimeError(f"new artifacts exceed 2 GiB: {size}")
    if free < 6 * 2**30:
        raise RuntimeError(f"E: free space below 6 GiB: {free}")
    # Totals are maintained once at process start and incremented atomically
    # by register_attempt.  Re-scanning every ledger JSON here made a long
    # fit quadratic in the number of attempts and could silently disagree
    # with the persisted pre-fit totals.
    return {"bytes": size, "free": free, **dict(_LEDGER_TOTALS)}


def current_totals() -> dict[str, int]:
    return dict(_LEDGER_TOTALS)


def acquire_run_lock(command: str = "") -> Path:
    """Prevent two fit processes from racing the persistent ledger.

    A stale lock left by an interrupted process is recoverable when its PID is
    no longer alive.  A live lock is always fail-closed; callers must not
    start a second process against the same run directory.
    """
    if LOCK_PATH.exists():
        try:
            old = read(LOCK_PATH)
            pid = int(old.get("pid", -1))
        except Exception as exc:
            raise RuntimeError(f"cannot inspect existing runner lock: {LOCK_PATH}") from exc
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except PermissionError:
                # An inaccessible PID must be treated as live; deleting its
                # lock would defeat the fail-closed concurrency contract.
                alive = True
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(f"another fit process owns the run lock: pid={pid}")
        LOCK_PATH.unlink()
    payload = {"pid": os.getpid(), "started": time.time(), "command": str(command)}
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except FileExistsError as exc:
        raise RuntimeError(f"runner lock appeared concurrently: {LOCK_PATH}") from exc
    atexit.register(release_run_lock)
    return LOCK_PATH


def release_run_lock() -> None:
    try:
        if not LOCK_PATH.exists():
            return
        row = read(LOCK_PATH)
        if int(row.get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink()
    except Exception:
        # Never mask a fit exception during interpreter shutdown.  A stale
        # lock is safely reclaimed by the next process after PID liveness check.
        return


sys.modules.setdefault("runtime", sys.modules[__name__])
