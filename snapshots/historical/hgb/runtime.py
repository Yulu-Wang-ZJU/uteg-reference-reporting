"""Run-local runtime and contracts for the unexecuted Ridge+MA13D HGB candidate.

This module deliberately keeps all temporary/cache locations below this new
prepare directory.  It exposes the future fit/ledger API, but the current
round has no authorization record, so the fit entrypoint fails closed before
importing a scientific fitter or writing an attempt row.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve()
ROOT = HERE.parent
OLD_RUN = ROOT.parent
GOAL_ROOT = OLD_RUN.parent
BASE = GOAL_ROOT.parent / "uteg_legacy_aligned_threecenter_20261022_155614"
NINE = GOAL_ROOT.parent / "uteg_nine_direction_ma_20261025_195548"
RIDGE_RUN = GOAL_ROOT / "ridge10_fullchain_20261026_122551"
AUDIT_RUN = OLD_RUN / "ridge_ma13d_hgb_history_audit_20261026_150233"

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

import numpy as np


SEEDS = tuple(range(1000, 1020))
DIRECTIONS = (
    "HZ+HB_to_SA", "SA+HB_to_HZ", "SA+HZ_to_HB",
    "HZ_to_SA", "HB_to_SA", "SA_to_HZ", "HB_to_HZ", "SA_to_HB", "HZ_to_HB",
)
POSITIONS = (0, 1, 2, 3)
FEATURE_DIM = 13
MA_CLASSES = (-1, 0, 1)
SELECTOR_RUNNER_SHA256 = "1aa7ff6afc2e7a9c8cf9b31c680d3adff224ae17ce97506c30671b932a6c2da6"
SCIENCE_CAP = 720
REPAIR_CAP = 112
TOTAL_CAP = SCIENCE_CAP + REPAIR_CAP
AUTH_PATH = ROOT / "USER_AUTHORIZATION_RECORD.json"
DRIVER_PATH = ROOT / "run_hgb_ma13d_driver.py"
RECOVERY_MANIFEST = ROOT / "PILOT_RECOVERY_MANIFEST_832.json"
LOCK_PATH = ROOT / "RUNNER.lock"
LEDGER_DIR = ROOT / "ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

# A future authorization must bind every object that can change this
# candidate's meaning.  Hashes are computed at the authorization check rather
# than trusted from a stale local snapshot.
AUTH_BINDING_PATHS = {
    "protocol": AUDIT_RUN / "CANDIDATE_PROTOCOL_DRAFT.md",
    "model_identity": ROOT / "MODEL_IDENTITY.json",
    "model_manifest": RIDGE_RUN / "FULL_MODELS_FROZEN.json",
    "data_manifest": BASE / "ROLE_FREEZE.json",
    "data15": BASE / "DATA15.npz",
    "selector": OLD_RUN / "DEVELOPMENT_SELECTOR.json",
    "runtime": HERE,
    "driver": DRIVER_PATH,
    "preflight": ROOT / "PREFLIGHT_RESULT.json",
    "recovery_manifest": RECOVERY_MANIFEST,
}


def json_clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p) if p.exists() else None}


def authorization_binding_hashes() -> dict[str, str]:
    """Return the exact hash contract required by a future authorization."""
    missing = [name for name, path in AUTH_BINDING_PATHS.items() if not path.is_file()]
    if missing:
        raise PermissionError(f"CANDIDATE_NOT_AUTHORIZED: missing binding files {missing}")
    return {name: sha256_file(path) for name, path in AUTH_BINDING_PATHS.items()}


def _ledger_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LEDGER_DIR.glob("*.json")):
        try:
            row = read_json(path)
        except Exception as exc:
            raise RuntimeError(f"corrupt ledger row: {path}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"corrupt ledger row (not object): {path}")
        if row.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}:
            raise RuntimeError(f"corrupt ledger row status: {path}")
        if not isinstance(row.get("units"), int) or row["units"] <= 0:
            raise RuntimeError(f"corrupt ledger row units: {path}")
        if row.get("kind") not in {"science", "repair"}:
            raise RuntimeError(f"corrupt ledger row kind: {path}")
        rows.append(row)
    return rows


def ledger_totals() -> dict[str, int]:
    rows = _ledger_rows()
    science = sum(int(row["units"]) for row in rows if row["kind"] == "science")
    repair = sum(int(row["units"]) for row in rows if row["kind"] == "repair")
    return {"science": science, "repair": repair, "total": science + repair,
            "running": sum(row["status"] == "RUNNING" for row in rows)}


def register_attempt(key: str, kind: str = "science", units: int = 1, **meta: Any) -> Path:
    """Persist one attempt before any future HGB.fit call.

    Duplicate keys and cap overflow are fatal.  The current preflight never
    calls this function; it is present so a future authorized runner cannot
    silently fit outside the ledger.
    """
    if kind not in {"science", "repair"}:
        raise ValueError(kind)
    amount = int(units)
    if amount != 1:
        raise ValueError("one position is exactly one fit unit")
    required_position_meta = {"direction", "seed", "position"}
    if not required_position_meta.issubset(meta):
        raise ValueError("every HGB attempt must bind direction, seed, and position")
    position_key = validate_position_key(meta["direction"], meta["seed"], meta["position"])
    if not str(key).startswith(position_key + "__HGB"):
        raise ValueError(f"attempt key is not bound to fixed position key: {key!r}")
    path = LEDGER_DIR / f"{key}.json"
    if path.exists():
        raise RuntimeError(f"duplicate attempt key: {path.name}")
    totals = ledger_totals()
    if totals["science"] + (amount if kind == "science" else 0) > SCIENCE_CAP:
        raise RuntimeError(f"science cap exceeded: {totals}")
    if totals["repair"] + (amount if kind == "repair" else 0) > REPAIR_CAP:
        raise RuntimeError(f"repair cap exceeded: {totals}")
    if totals["total"] + amount > TOTAL_CAP:
        raise RuntimeError(f"total cap exceeded: {totals}")
    write_json(path, {"schema": "ridge_ma13d_hgb_attempt_v1", "status": "RUNNING",
                      "kind": kind, "units": amount, "key": key,
                      "started": time.time(), **meta})
    return path


def finish_attempt(path: str | Path, status: str, **meta: Any) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(status)
    p = Path(path)
    row = read_json(p)
    if row.get("status") != "RUNNING":
        raise RuntimeError(f"attempt is not RUNNING: {p}")
    row.update({"status": status, "finished": time.time(), **meta})
    write_json(p, row)


def _pid_alive(pid: int) -> bool:
    """Query PID liveness without sending a signal on Windows."""
    if int(pid) <= 0:
        raise RuntimeError(f"invalid lock PID: {pid}")
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise RuntimeError(f"cannot query lock PID {pid}; fail closed") from exc
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        err = ctypes.get_last_error()
        raise RuntimeError(f"cannot query lock PID {pid} on Windows (error {err}); fail closed")
    code = wintypes.DWORD()
    try:
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            err = ctypes.get_last_error()
            raise RuntimeError(f"cannot read lock PID {pid} on Windows (error {err}); fail closed")
        return int(code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def acquire_lock(command: str) -> None:
    if LOCK_PATH.exists():
        try:
            old = read_json(LOCK_PATH)
            pid = int(old.get("pid", -1))
            if _pid_alive(pid):
                raise RuntimeError(f"active runner lock exists pid={pid}")
        except RuntimeError:
            raise
        else:
            LOCK_PATH.unlink()
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started": time.time(), "command": command}, f, indent=2)
    atexit.register(release_lock)


def release_lock() -> None:
    try:
        if LOCK_PATH.exists() and int(read_json(LOCK_PATH).get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink()
    except Exception:
        pass


def lock_owned_by_current_process() -> bool:
    """Check ownership without probing or terminating another process."""
    if not LOCK_PATH.exists():
        return False
    try:
        row = read_json(LOCK_PATH)
        return int(row.get("pid", -1)) == os.getpid()
    except Exception:
        return False


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    """Exact original 13D MA feature construction, without fitting."""
    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    ym = np.asarray(ym, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pred_ma = med[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((
        x,
        (med - ym[None, :]) / ys[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float),
    ))


def fit_fold_preprocessor(features: Any) -> dict[str, Any]:
    """Exact original LR fold-local median/mean/sd contract."""
    f = np.asarray(features, dtype=float)
    if f.ndim != 2 or f.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected [n,13] features, got {f.shape}")
    finite = np.isfinite(f)
    median = np.asarray([np.median(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    mean = np.asarray([np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    sd = np.asarray([max(np.std(f[finite[:, j], j]), 1e-6) if finite[:, j].any() else 1.0 for j in range(f.shape[1])])
    return {"median": median, "mean": mean, "sd": sd, "clip": (-10.0, 10.0), "n": int(len(f)), "p": FEATURE_DIM}


def apply_fold_preprocessor(features: Any, pre: Mapping[str, Any]) -> np.ndarray:
    f = np.asarray(features, dtype=float)
    med = np.asarray(pre["median"], dtype=float)
    mean = np.asarray(pre["mean"], dtype=float)
    sd = np.asarray(pre["sd"], dtype=float)
    if f.ndim != 2 or f.shape[1] != FEATURE_DIM or med.shape != (FEATURE_DIM,):
        raise ValueError("MA13D preprocessor shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("invalid original fold preprocessor")
    z = (np.where(np.isfinite(f), f, med[None, :]) - mean[None, :]) / sd[None, :]
    return np.clip(z, -10.0, 10.0)


def fallback_reason(n: int, errors: int, correct: int) -> str | None:
    if int(n) < 8:
        return "small_n"
    if int(errors) < 2:
        return "errors_lt2"
    if int(correct) < 2:
        return "correct_lt2"
    return None


def constant_model(n: int, errors: int, correct: int) -> dict[str, Any]:
    reason = fallback_reason(n, errors, correct)
    if reason is None:
        raise ValueError("constant fallback requested for a supported group")
    n = int(n); errors = int(errors); correct = int(correct)
    return {"schema": "ridge_ma13d_hgb_constant_fallback_v1", "status": "FALLBACK_CONSTANT",
            "fallback_reason": reason, "n": n, "errors": errors, "correct": correct,
            "constant_score": float((errors + 0.5) / (n + 1.0)) if n else 0.5,
            "sample_weight": "equal_row_weight_1", "fit_calls": 0}


def action_for(pred_class: int, score: float, threshold: float | None, *,
               score_valid: bool, qc_valid: bool) -> int:
    """Preserve low/high action, but fail closed for invalid score/QC rows."""
    if not isinstance(score_valid, (bool, np.bool_)) or not isinstance(qc_valid, (bool, np.bool_)):
        raise ValueError("score_valid and qc_valid must be explicit boolean flags")
    if not bool(score_valid) or not bool(qc_valid) or not np.isfinite(score):
        return 9
    pred_class = int(pred_class)
    if pred_class not in MA_CLASSES:
        return 9
    if pred_class != 0:
        return pred_class
    if threshold is None or not np.isfinite(score):
        return 9
    return 0 if float(score) <= float(threshold) else 9


def hgb_config(seed: int) -> dict[str, Any]:
    return {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 4,
            "max_depth": 2, "min_samples_leaf": 10, "l2_regularization": 10.0,
            "max_bins": 255, "early_stopping": False, "random_state": int(seed)}


def validate_position_key(direction: Any, seed: Any, position: Any) -> str:
    """Validate the fixed 9×20×4 position universe before ledger registration."""
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError(f"invalid direction key: {direction!r}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed key must be an integer: {seed!r}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed outside fixed universe: {seed!r}")
    if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
        raise ValueError(f"position key must be an integer: {position!r}")
    if int(position) not in POSITIONS:
        raise ValueError(f"position outside fixed universe: {position!r}")
    return f"{direction}__seed{int(seed)}__pos{int(position)}"


def validate_training_masks(n: int, *, known_label_mask: Any, qc_mask: Any,
                           score_valid_mask: Any, selected_indices: Any | None = None) -> dict[str, Any]:
    """Bind legal row masks to the exact X/y rows supplied to the fitter.

    The builder must pass the original full-row masks and the exact selected
    row indices used to construct X/y.  Finite feature values alone are not a
    substitute for this contract.
    """
    n = int(n)
    out: dict[str, Any] = {}
    full_masks: dict[str, np.ndarray] = {}
    for name, value in (("known_label_mask", known_label_mask),
                        ("qc_mask", qc_mask),
                        ("score_valid_mask", score_valid_mask)):
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.dtype != np.bool_:
            raise ValueError(f"{name} must be a one-dimensional bool mask")
        full_masks[name] = arr
    lengths = {len(arr) for arr in full_masks.values()}
    if len(lengths) != 1:
        raise ValueError("known/QC/score masks must share one full-row length")
    full_n = lengths.pop()
    if selected_indices is None:
        selected = np.arange(full_n, dtype=np.int64)
    else:
        selected = np.asarray(selected_indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("selected_indices must be a one-dimensional integer array")
        selected = selected.astype(np.int64, copy=False)
        if len(selected) != n or len(np.unique(selected)) != len(selected):
            raise ValueError("selected_indices must be unique and aligned to X/y")
        if np.any(selected < 0) or np.any(selected >= full_n):
            raise ValueError("selected_indices outside full-row mask range")
    legal = full_masks["known_label_mask"] & full_masks["qc_mask"] & full_masks["score_valid_mask"]
    legal_indices = np.flatnonzero(legal).astype(np.int64)
    if not np.array_equal(selected, legal_indices):
        raise ValueError("X/y rows do not exactly equal the original known/QC/score legal rows")
    out.update({name: {"full_n": int(full_n), "legal_n": int(len(legal_indices)),
                       "true_n": int(arr.sum()), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
               for name, arr in full_masks.items()})
    out["selected_indices_sha256"] = hashlib.sha256(selected.tobytes()).hexdigest()
    out["contract"] = "original_full_masks_plus_exact_legal_row_indices"
    return out


def require_fit_authorized() -> dict[str, Any]:
    """Fail closed; this file is intentionally absent in the current round."""
    if not AUTH_PATH.exists():
        raise PermissionError("CANDIDATE_NOT_AUTHORIZED: no user authorization record")
    auth = read_json(AUTH_PATH)
    if auth.get("status") != "AUTHORIZED" or auth.get("round") != ROOT.name:
        raise PermissionError("CANDIDATE_NOT_AUTHORIZED: authorization binding mismatch")
    if auth.get("runner_sha256") != sha256_file(HERE):
        raise PermissionError("CANDIDATE_NOT_AUTHORIZED: runner hash mismatch")
    expected = authorization_binding_hashes()
    supplied = auth.get("bindings")
    if not isinstance(supplied, dict):
        raise PermissionError("CANDIDATE_NOT_AUTHORIZED: missing hash-bound bindings")
    mismatches = [name for name, digest in expected.items() if supplied.get(name) != digest]
    if mismatches:
        raise PermissionError(f"CANDIDATE_NOT_AUTHORIZED: binding hash mismatch {mismatches}")
    return auth


def fit_hgb_position(X_train: Any, y_train: Any, *, direction: str, seed: int, position: int,
                     known_label_mask: Any, qc_mask: Any, score_valid_mask: Any,
                     selected_indices: Any | None = None, attempt_kind: str = "science",
                     attempt_suffix: str = "") -> dict[str, Any]:
    """Future fit path; never reachable in this prepare round without auth.

    A failed optimizer is recorded as FAILED and is never converted to the
    constant fallback.  Fallback is selected only before fitting from the
    frozen original n/errors/correct support rule.  The three explicit legal
    masks must be aligned to the already-prefiltered X/y rows.
    """
    require_fit_authorized()
    position_key = validate_position_key(direction, seed, position)
    if not lock_owned_by_current_process():
        raise RuntimeError("FIT_REQUIRES_ACTIVE_RUNNER_LOCK")
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM or y.shape != (len(X),):
        raise ValueError("invalid HGB input shape")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("binary original error label required")
    mask_contract = validate_training_masks(
        len(y), known_label_mask=known_label_mask, qc_mask=qc_mask,
        score_valid_mask=score_valid_mask, selected_indices=selected_indices)
    n = len(y); errors = int(y.sum()); correct = n - errors
    reason = fallback_reason(n, errors, correct)
    key = f"{position_key}__HGB{attempt_suffix}"
    attempt = register_attempt(key, attempt_kind, 1, direction=direction, seed=int(seed), position=int(position),
                               n=n, errors=errors, correct=correct, preprocessing="original_MA13D_LR_fold_local",
                               sample_weight="all_ones", training_masks=mask_contract,
                               test_labels_read=False)
    if reason is not None:
        result = constant_model(n, errors, correct)
        result.update({"fit_calls": 1, "attempt_kind": attempt_kind, "attempt_key": key,
                       "training_masks": mask_contract})
        finish_attempt(attempt, "COMPLETE", result_status="FALLBACK_CONSTANT", fallback=True)
        return result
    pre = fit_fold_preprocessor(X)
    z = apply_fold_preprocessor(X, pre)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(**hgb_config(seed))
        clf.fit(z, y, sample_weight=np.ones(n, dtype=float))
        result = {"schema": "ridge_ma13d_hgb_model_v1", "status": "COMPLETE", "direction": direction,
                  "seed": int(seed), "position": int(position), "n": n, "errors": errors, "correct": correct,
                  "pre": pre, "config": hgb_config(seed), "model": clf, "sample_weight": "all_ones",
                  "attempt_kind": attempt_kind, "attempt_key": key, "training_masks": mask_contract,
                  "fit_calls": 1}
        finish_attempt(attempt, "COMPLETE", result_status="COMPLETE")
        return result
    except Exception as exc:
        finish_attempt(attempt, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def fit_hgb_batch(rows: Any) -> list[dict[str, Any]]:
    """Future authorized batch entry; every position runs under one lock.

    This interface is intentionally not called in the current prepare round.
    It accepts prefiltered rows only; ``fit_hgb_position`` additionally checks
    the process-owned lock and the fixed position universe.
    """
    require_fit_authorized()
    if not isinstance(rows, (list, tuple)):
        raise ValueError("batch rows must be a list/tuple")
    acquire_lock("authorized_hgb_batch")
    try:
        results = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("batch row must be a mapping")
            results.append(fit_hgb_position(**dict(row)))
        return results
    finally:
        release_lock()


def disk_guard() -> dict[str, int]:
    bytes_used = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    free = shutil.disk_usage(ROOT).free
    if bytes_used > 2 * 2**30:
        raise RuntimeError("new prepare artifacts exceed 2 GiB")
    if free < 6 * 2**30:
        raise RuntimeError("E: free space below 6 GiB")
    return {"bytes": int(bytes_used), "free": int(free), **ledger_totals()}
