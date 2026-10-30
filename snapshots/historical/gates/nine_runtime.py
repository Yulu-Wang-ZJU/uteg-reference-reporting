"""Task-local runtime shim for legacy model definitions.

Only definitions are loaded from historical modules through this shim.  ROOT,
ledger, temporary directories, and all output paths belong to the new nine-
direction run, so historical directories remain read-only.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
RUN_ROOT = HERE.parents[1]
OUT = RUN_ROOT.parent
ROOT = RUN_ROOT
CACHE = RUN_ROOT / "cache"
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

# Historical source definitions are read-only dependencies.  These constants
# intentionally point out of RUN_ROOT; no function below writes there.
OLD = OUT / "uteg_med4_early_decision_20261004_191744"
CAP = OUT / "uteg_med4_capacity_20261006_001701"
CLIN = OUT / "uteg_med4_source_clinical_20261006_104632"
MAP = OUT / "uteg_fourparam_mapping_trust_20261006_132600"
ZERO = OUT / "uteg_zero5_med4_20261006_113257"
PLAT = OUT / "uteg_platform_param_compare_20261006_125428"
SC = OUT / "uteg_scalar_ordinal_trust_20261006_181750"
TH = OUT / "uteg_trust_threshold90_20261006_161248"
GATE = OUT / "uteg_trust_gate_selection_20261004_205412"
D1 = OUT / "uteg_cohort399_gate_revalidation_20261006_203700"
D2 = OUT / "uteg_c60_new_trust_20261007_170322"
AD = OUT / "uteg_d2_ci_native85_20261007_012322"
PREV = CAP

CURRENT_SCOPE = ""
CURRENT_REPAIR = False
SCIENCE_FIT_CAP = 16_000
TOTAL_FIT_CAP = 16_200
REPAIR_FIT_CAP = 200
NONSCIENCE_KINDS = {"test", "repair", "audit"}


def _json_clean(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write(name: str | Path, value: Any) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_clean(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract(path: str | Path, names: list[str], namespace: dict[str, Any]) -> None:
    """AST-extract definitions only; historical module top-level is not run."""
    source_path = Path(path)
    nodes = [
        node for node in ast.parse(source_path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    if len(nodes) != len(names):
        raise RuntimeError(f"missing AST definitions in {source_path}: {names}")
    namespace.setdefault("__name__", "nine_definitions_only")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)


def load(name: str, path: str | Path):
    """Import a historical definition module under the task-local runtime."""
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
    CURRENT_SCOPE = str(scope).strip().replace("/", "_").replace("\\", "_")
    CURRENT_REPAIR = bool(repair)


def _ledger_totals() -> dict[str, int]:
    totals = {"science": 0, "total": 0, "signal": 0, "repair": 0}
    ledger_dir = ROOT / "ledger"
    if not ledger_dir.exists():
        return totals
    for path in ledger_dir.glob("*.json"):
        try:
            row = read(path)
            units = int(row.get("units", 0))
            kind = str(row.get("kind", ""))
        except Exception:
            continue
        totals["total"] += units
        if kind not in NONSCIENCE_KINDS:
            totals["science"] += units
        if kind in {"signal", "signal_fit"}:
            totals["signal"] += units
        if kind in {"repair", "audit"}:
            totals["repair"] += units
    return totals


_LEDGER_TOTALS = _ledger_totals()


def ledger(key: str, kind: str, units: int = 1, **metadata: Any) -> None:
    """Record one fit unit and enforce the nine-direction hard caps."""
    scope_key = f"{CURRENT_SCOPE}__{key}" if CURRENT_SCOPE else str(key)
    path = ROOT / "ledger" / f"{scope_key}.json"
    if path.exists():
        old = read(path)
        expected_kind = "repair" if CURRENT_REPAIR else str(kind)
        if str(old.get("kind")) != expected_kind or int(old.get("units", -1)) != int(units):
            raise RuntimeError(f"ledger key conflicts with existing record: {path}")
        return
    amount = int(units)
    original_kind = str(kind)
    if CURRENT_REPAIR:
        kind = "repair"
        metadata = {"original_kind": original_kind, **metadata}
    nxt = dict(_LEDGER_TOTALS)
    nxt["total"] += amount
    if str(kind) not in NONSCIENCE_KINDS:
        nxt["science"] += amount
    if str(kind) in {"repair", "audit"}:
        nxt["repair"] += amount
    if nxt["science"] > SCIENCE_FIT_CAP:
        raise RuntimeError(f"science hard cap exceeded: {nxt}")
    if nxt["total"] > TOTAL_FIT_CAP:
        raise RuntimeError(f"total hard cap exceeded: {nxt}")
    if nxt["repair"] > REPAIR_FIT_CAP:
        raise RuntimeError(f"repair/check hard cap exceeded: {nxt}")
    payload = {"kind": str(kind), "units": amount, "scope": CURRENT_SCOPE, **metadata}
    write(path.relative_to(ROOT), payload)
    _LEDGER_TOTALS.update(nxt)


def guard() -> dict[str, int]:
    size = sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file())
    free = shutil.disk_usage(RUN_ROOT).free
    if size > 2 * 2**30:
        raise RuntimeError(f"new artifacts exceed 2 GiB: {size}")
    if free < 6 * 2**30:
        raise RuntimeError(f"E: free space below 6 GiB: {free}")
    return {"bytes": size, "free": free, **_LEDGER_TOTALS}


def legacy_seed(direction: int, seed: int, fold: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"legacy_aligned_20261022|{direction}|{seed}|{fold}".encode()).digest()[:8],
        "little",
    ) % 2_147_483_647


# Historical model files use ``import runtime as r``.  Register this shim
# under that name before any such file is loaded.
sys.modules.setdefault("runtime", sys.modules[__name__])

