"""Authorized development-only Ridge+MA13D HGB driver.

The driver consumes only the frozen Ridge target objects' development roles:
outer-A/inner-V OOF predictions train the gate, and outer-B predictions are
held out for later selector replay.  It never opens the target object's test
field or DATA15 medical labels.  HGB fitting is reachable only after the
hash-bound authorization record and independent implementation receipt exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import runtime as r  # establishes run-local caches before numerical imports
import joblib
import numpy as np


TARGET_BY_DIRECTION = {
    "HZ+HB_to_SA": "SAHZU", "SA+HB_to_HZ": "Huzhou", "SA+HZ_to_HB": "Hubei",
    "HZ_to_SA": "SAHZU", "HB_to_SA": "SAHZU", "SA_to_HZ": "Huzhou",
    "HB_to_HZ": "Huzhou", "SA_to_HB": "Hubei", "HZ_to_HB": "Hubei",
}
MANIFEST = r.RIDGE_RUN / "FULL_MODELS_FROZEN.json"
DATA15 = r.BASE / "DATA15.npz"
SELECTOR = r.OLD_RUN / "DEVELOPMENT_SELECTOR.json"
PREFLIGHT = r.ROOT / "HGB_DRIVER_PREFLIGHT.json"
FIT_REPORT = r.ROOT / "HGB_FIT_RUN_REPORT.json"
MODEL_ROOT = r.ROOT / "models_hgb"
PARTIAL_ROOT = r.ROOT / "partial_hgb_models"
RECOVERY_MANIFEST = r.ROOT / "PILOT_RECOVERY_MANIFEST_832.json"
PILOT_SEEDS = (1000, 1001)


def _hash_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _hash_ids(values: Any) -> str:
    return _hash_bytes("\n".join(map(str, np.asarray(values).astype(str).tolist())).encode())


def _path_from_manifest(rel: str) -> Path:
    return r.RIDGE_RUN / Path(str(rel))


def _load_metadata() -> dict[str, np.ndarray]:
    """Read only non-medical DATA15 keys; no medical/test array access."""
    with np.load(DATA15, allow_pickle=False) as z:
        required = {"x", "ids", "input_qc"}
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(f"DATA15 missing metadata keys: {missing}")
        out = {key: np.asarray(z[key]) for key in required}
    if len(out["x"]) != len(out["ids"]) or len(out["x"]) != len(out["input_qc"]):
        raise RuntimeError("DATA15 metadata lengths disagree")
    return out


def _manifest_paths() -> dict[Path, str]:
    manifest = r.read_json(MANIFEST)
    if manifest.get("config") != "RIDGE-10" or manifest.get("model_family") != "RIDGE":
        raise RuntimeError("Ridge-10 manifest identity mismatch")
    if manifest.get("test_labels_read") or manifest.get("test_evaluated"):
        raise RuntimeError("Ridge manifest is not test blind")
    rows = manifest.get("target_object_hashes")
    if not isinstance(rows, dict) or len(rows) != 180:
        raise RuntimeError("expected exactly 180 target object hashes")
    out: dict[Path, str] = {}
    for rel, expected in rows.items():
        path = _path_from_manifest(rel)
        if not path.is_file():
            raise RuntimeError(f"missing target object: {path}")
        got = r.sha256_file(path)
        if got.lower() != str(expected).lower():
            raise RuntimeError(f"target object hash mismatch: {path}")
        out[path] = str(expected).lower()
    return out


def _recovery_scope() -> set[tuple[str, int, int]]:
    """Return the exact previously-missing pilot positions allowed to repair."""
    manifest = r.read_json(RECOVERY_MANIFEST)
    if manifest.get("schema") != "ridge_ma13d_hgb_pilot_recovery_manifest_v2":
        raise RuntimeError("recovery manifest identity mismatch")
    if manifest.get("recovery_contract", {}).get("only_missing_scope_may_use_repair") is not True:
        raise RuntimeError("recovery manifest does not enforce missing-scope repairs")
    scope = manifest.get("pilot_scope", {})
    dirs = scope.get("directions")
    seeds = scope.get("seeds")
    positions = scope.get("positions")
    if dirs != list(r.DIRECTIONS) or seeds != [1000, 1001] or positions != [0, 1, 2, 3]:
        raise RuntimeError("recovery manifest pilot scope drift")
    expected = {(str(d), int(s), int(pos)) for d in dirs for s in seeds for pos in positions}
    observed = manifest.get("observed", {})
    if int(scope.get("expected_position_attempts", -1)) != len(expected):
        raise RuntimeError("recovery manifest position count drift")
    if int(observed.get("artifact_missing", -1)) != len(expected):
        raise RuntimeError("recovery manifest missing-artifact count drift")
    if int(manifest.get("recovery_contract", {}).get("repair_cap", -1)) != r.REPAIR_CAP:
        raise RuntimeError("recovery manifest repair cap drift")
    return expected


def _object_path(paths: Mapping[Path, str], direction: str, seed: int) -> tuple[Path, str]:
    target = TARGET_BY_DIRECTION[direction]
    matches = [path for path in paths if direction in path.parts and path.name == f"{target}_CFULL_seed{int(seed)}.joblib"]
    if len(matches) != 1:
        raise RuntimeError(f"object lookup failed for {direction} seed={seed}: {len(matches)}")
    return matches[0], paths[matches[0]]


def _truth_map(obj: Mapping[str, Any]) -> dict[int, int]:
    truth: dict[int, int] = {}
    for pos in (0, 1, 2):
        q = obj["positions"][pos]
        data = obj["gate_models"][pos]["outer_B_data"]
        pool_index = [int(v) for v in data["pool_index"]]
        values = [int(v) for v in data["truth_class"]]
        if [int(v) for v in q["B"]] != pool_index or len(values) != len(pool_index):
            raise RuntimeError("outer-B role/order mismatch")
        for idx, value in zip(pool_index, values):
            if idx in truth or value not in r.MA_CLASSES:
                raise RuntimeError("duplicate/invalid development truth mapping")
            truth[idx] = value
    pool = {int(v) for v in obj["pool"]}
    if set(truth) != pool:
        raise RuntimeError("outer-B union does not exactly cover development pool")
    return truth


def _check_inner_role(q: Mapping[str, Any], pos: int) -> dict[str, Any]:
    A = np.asarray(q["A"], dtype=np.int64)
    B = np.asarray(q["B"], dtype=np.int64)
    if len(A) == 0 or len(np.unique(A)) != len(A) or len(np.unique(B)) != len(B) or set(A) & set(B):
        raise RuntimeError(f"invalid outer A/B role at position {pos}")
    inner = q.get("inner")
    if not isinstance(inner, list) or len(inner) != 3:
        raise RuntimeError(f"position {pos} lacks three inner folds")
    seen: list[int] = []
    folds: list[dict[str, Any]] = []
    for fold in inner:
        T = np.asarray(fold["T"], dtype=np.int64)
        V = np.asarray(fold["V"], dtype=np.int64)
        if not len(T) or not len(V) or set(T) & set(V) or set(T) | set(V) != set(A):
            raise RuntimeError(f"inner T/V leakage or incomplete partition at position {pos}")
        seen.extend(V.tolist())
        folds.append({"T_n": int(len(T)), "V_n": int(len(V)),
                      "T_hash": _hash_ids(np.sort(T)), "V_hash": _hash_ids(np.sort(V))})
    if sorted(seen) != sorted(A.tolist()):
        raise RuntimeError(f"inner V rows do not cover outer A at position {pos}")
    return {"A_n": int(len(A)), "B_n": int(len(B)), "A_hash": _hash_ids(A),
            "B_hash": _hash_ids(B), "inner_folds": folds}


def _score_hgb(model: Mapping[str, Any], features: np.ndarray, pred: np.ndarray,
               qc: np.ndarray, score_valid: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    qc = np.asarray(qc, dtype=bool)
    score_valid = np.asarray(score_valid, dtype=bool)
    if f.ndim != 2 or f.shape[1] != r.FEATURE_DIM or qc.shape != (len(f),) or score_valid.shape != (len(f),):
        raise RuntimeError("HGB scoring shape contract failed")
    # Position 3 is the full-C gate and has no outer-B rows.  Its fitted
    # model is still needed for the later frozen deployment path, but the
    # development-side outer score is an empty vector and must remain a
    # valid empty output rather than entering sklearn preprocessing.
    if len(f) == 0:
        return np.empty(0, dtype=float)
    if model.get("status") == "FALLBACK_CONSTANT":
        # Constant fallback has no fitted fold preprocessor.  Keep the
        # feature-finiteness/QC fail-closed contract below, but do not try to
        # access the absent ``pre`` payload before constructing its constant
        # score.
        score = np.full(len(f), float(model["constant_score"]), dtype=float)
    else:
        z = r.apply_fold_preprocessor(f, model["pre"])
        estimator = model["model"]
        score = np.asarray(estimator.predict_proba(z)[:, 1], dtype=float)
    valid = qc & score_valid & np.isin(pred, r.MA_CLASSES) & np.isfinite(f).all(axis=1) & np.isfinite(score)
    return np.where(valid, score, np.inf)


def _context(obj: Mapping[str, Any], data: Mapping[str, np.ndarray], pos: int,
             truth_map: Mapping[int, int]) -> dict[str, Any]:
    q = obj["positions"][pos]
    gm = obj["gate_models"][pos]
    adapter = str(gm["adapter"])
    cross_map = q.get("cross") or {}
    vv_map = q.get("vv") or {}
    if adapter not in cross_map:
        raise RuntimeError(f"adapter {adapter} missing from position {pos}")
    A = np.asarray(q["A"], dtype=np.int64)
    B = np.asarray(q["B"], dtype=np.int64)
    cross = np.asarray(cross_map[adapter], dtype=float)
    vv = np.asarray(vv_map[adapter], dtype=float) if len(B) else np.empty((0, 4), dtype=float)
    if cross.shape != (len(A), 4) or vv.shape != (len(B), 4):
        raise RuntimeError(f"adapter prediction shape mismatch at position {pos}")
    source = obj["source_model"]
    ym = np.asarray(source["ym"], dtype=float)
    ys = np.asarray(source["ys"], dtype=float)
    if ym.shape != (4,) or ys.shape != (4,) or not np.isfinite(ym).all() or not np.isfinite(ys).all() or np.any(ys == 0):
        raise RuntimeError("source ym/ys contract failed")
    f_a = r.ma_features(data["x"][A], cross, ym, ys)
    pred_a = r.ma_class(cross[:, 2])
    truth_a = np.asarray([truth_map[int(idx)] for idx in A], dtype=np.int8)
    qc_a = np.asarray(data["input_qc"][A], dtype=bool)
    known_a = np.isin(pred_a, r.MA_CLASSES) & np.isin(truth_a, r.MA_CLASSES)
    score_valid_a = np.isfinite(cross[:, 2]) & np.isin(pred_a, r.MA_CLASSES)
    legal_a = known_a & qc_a & score_valid_a
    if not np.array_equal(legal_a, known_a):
        raise RuntimeError("original MA known-label legal rows differ from QC/score masks; refusing silent row shrink")
    train_indices = np.flatnonzero(legal_a).astype(np.int64)
    if len(train_indices) == 0:
        raise RuntimeError(f"no legal gate-training rows at position {pos}")
    f_b = r.ma_features(data["x"][B], vv, ym, ys)
    pred_b = r.ma_class(vv[:, 2])
    truth_b = np.asarray(gm["outer_B_data"]["truth_class"], dtype=np.int8)
    ids_b = np.asarray(gm["outer_B_data"]["record_id"]).astype(str)
    expected_ids = np.asarray(data["ids"][B]).astype(str)
    if not np.array_equal(ids_b, expected_ids):
        raise RuntimeError(f"outer-B record identity mismatch at position {pos}")
    qc_b = np.asarray(data["input_qc"][B], dtype=bool)
    score_valid_b = np.isfinite(vv[:, 2]) & np.isin(pred_b, r.MA_CLASSES)
    base_score = np.asarray(gm["outer_B_data"]["score"], dtype=float)
    if base_score.shape != (len(B),):
        raise RuntimeError(f"saved baseline outer-B score shape mismatch at position {pos}")
    return {
        "position": pos, "adapter": adapter, "A": A, "B": B,
        "features": f_a, "pred": pred_a, "truth": truth_a,
        "qc": qc_a, "known": known_a, "score_valid": score_valid_a,
        "legal": legal_a, "train_indices": train_indices,
        "outer_features": f_b, "outer_pred": pred_b, "outer_truth": truth_b,
        "outer_qc": qc_b, "outer_score_valid": score_valid_b,
        "outer_score": base_score, "outer_index": B, "outer_record_id": ids_b,
        "source_ym": ym, "source_ys": ys,
        "role": _check_inner_role(q, pos),
    }


def _artifact_path(direction: str, seed: int) -> Path:
    target = TARGET_BY_DIRECTION[direction]
    return MODEL_ROOT / direction / "target15" / f"{target}_HGB_seed{int(seed)}.joblib"


def _partial_model_path(direction: str, seed: int, pos: int, attempt_key: str) -> Path:
    """Atomic per-position checkpoint written before outer-B scoring.

    This is a recovery artifact, not a deployment object and never contains
    test predictions.  It prevents a post-fit score exception (notably an
    empty outer-B vector at position 3) from losing an already completed fit.
    """
    key = str(attempt_key)
    if "__retry" in key:
        tag = "retry" + key.rsplit("__retry", 1)[1]
    else:
        tag = "science"
    # Keep the Windows path comfortably below MAX_PATH; the full ledger key
    # remains in the checkpoint payload and the ledger itself.
    return PARTIAL_ROOT / direction / f"seed{int(seed)}_pos{int(pos)}_{tag}.joblib"


def _atomic_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(value, tmp, compress=3)
    tmp.replace(path)


def _stop_after_postfit_failure(failures: list[Mapping[str, Any]]) -> bool:
    """Fail-fast predicate kept pure so it can be checked without fitting."""
    return bool(failures)


def _scope_rows(scope: str) -> list[dict[str, Any]]:
    return [row for row in r._ledger_rows()
            if str(row.get("key", "")) == scope or str(row.get("key", "")).startswith(scope + "__retry")
            or row.get("scope") == scope]


def _next_attempt(direction: str, seed: int, pos: int) -> tuple[str, str, str]:
    base = f"{direction}__seed{int(seed)}__pos{int(pos)}__HGB"
    rows = _scope_rows(base)
    if any(row.get("status") in {"RUNNING", "PENDING"} for row in rows):
        raise RuntimeError(f"unfinished attempt blocks rerun: {base}")
    if any(row.get("status") == "COMPLETE" for row in rows):
        raise RuntimeError(f"completed attempt already exists: {base}")
    retries = sum(1 for row in rows if row.get("kind") == "repair")
    if not rows:
        return base, "science", ""
    retry_number = retries + 1
    return f"{base}__retry{retry_number}", "repair", f"__retry{retry_number}"


def _save_artifact(direction: str, seed: int, obj: Mapping[str, Any], base_sha: str,
                   gates: Mapping[int, Mapping[str, Any]]) -> Path:
    out = {
        "schema": "ridge_ma13d_hgb_target_object_v1", "candidate": "RIDGE-10+MA13D-HGB",
        "direction": direction, "target": TARGET_BY_DIRECTION[direction], "seed": int(seed),
        "base_object_sha256": base_sha, "external_test_evaluated": False,
        "test_labels_hidden": True, "test_labels_read": False,
        "candidate_runtime_sha256": r.sha256_file(r.HERE),
        "candidate_driver_sha256": r.sha256_file(Path(__file__)),
        "gate_input": "original numeric-MA 13D", "sample_weight": "all_ones",
        "source_family": obj.get("source_family"), "source_centers": obj.get("source_centers"),
        "gates": {str(pos): dict(value) for pos, value in gates.items()},
        "fit_units": 4,
    }
    path = _artifact_path(direction, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(out, tmp, compress=3)
    tmp.replace(path)
    return path


def _base_score_replay(ctx: Mapping[str, Any]) -> dict[str, Any]:
    score = np.asarray(ctx["outer_score"], dtype=float)
    pred = np.asarray(ctx["outer_pred"], dtype=np.int8)
    valid = np.asarray(ctx["outer_qc"], dtype=bool) & np.asarray(ctx["outer_score_valid"], dtype=bool)
    if np.any(valid & ~np.isfinite(score)):
        raise RuntimeError("baseline score is invalid on a QC/score-valid outer-B row")
    if np.any(~valid & np.isfinite(score)):
        raise RuntimeError("baseline score is finite on an invalid outer-B row")
    return {"valid_n": int(valid.sum()), "invalid_n": int((~valid).sum()),
            "pred_class_valid_n": int(np.isin(pred, r.MA_CLASSES).sum())}


def _load_partial_model(direction: str, seed: int, pos: int, base_sha: str) -> dict[str, Any] | None:
    """Load a valid per-position checkpoint without consuming another fit."""
    candidates = sorted((PARTIAL_ROOT / direction).glob(f"seed{int(seed)}_pos{int(pos)}_*.joblib"))
    if not candidates:
        return None
    path = candidates[-1]
    payload = joblib.load(path)
    if (payload.get("schema") != "ridge_ma13d_hgb_partial_model_v1"
            or payload.get("direction") != direction
            or int(payload.get("seed", -1)) != int(seed)
            or int(payload.get("position", -1)) != int(pos)
            or payload.get("base_object_sha256") != base_sha
            or payload.get("test_labels_read") is not False
            or payload.get("postfit_state") != "FIT_COMPLETE_BEFORE_OUTER_SCORE"):
        raise RuntimeError(f"partial checkpoint identity mismatch: {path}")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("status") not in {"COMPLETE", "FALLBACK_CONSTANT"}:
        raise RuntimeError(f"partial checkpoint model invalid: {path}")
    return {"path": path, "model": model}


def _fit_one(direction: str, seed: int, pos: int, obj: Mapping[str, Any], data: Mapping[str, np.ndarray],
             truth_map: Mapping[int, int], base_sha: str, attempt_kind: str, attempt_suffix: str,
             model_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    ctx = _context(obj, data, pos, truth_map)
    if model_override is None:
        y = (ctx["pred"][ctx["train_indices"]] != ctx["truth"][ctx["train_indices"]]).astype(np.int8)
        model = r.fit_hgb_position(
            ctx["features"][ctx["train_indices"]], y,
            direction=direction, seed=seed, position=pos,
            known_label_mask=ctx["known"], qc_mask=ctx["qc"], score_valid_mask=ctx["score_valid"],
            selected_indices=ctx["train_indices"], attempt_kind=attempt_kind,
            attempt_suffix=attempt_suffix)
        # Persist the completed position fit before any outer-B scoring.  A
        # score failure must not discard an optimizer result already counted
        # by the ledger; a later repair run can score this checkpoint.
        _atomic_dump({
            "schema": "ridge_ma13d_hgb_partial_model_v1",
            "direction": direction, "seed": int(seed), "position": int(pos),
            "attempt_key": model.get("attempt_key"), "base_object_sha256": base_sha,
            "postfit_state": "FIT_COMPLETE_BEFORE_OUTER_SCORE",
            "test_labels_read": False, "model": model,
        }, _partial_model_path(direction, seed, pos, str(model.get("attempt_key"))))
    else:
        model = dict(model_override)
    candidate_within = _score_hgb(model, ctx["outer_features"], ctx["outer_pred"],
                                  ctx["outer_qc"], ctx["outer_score_valid"])
    candidate_score = np.where(ctx["outer_pred"] == 0, candidate_within, ctx["outer_score"])
    gate = {
        "position": int(pos), "adapter": ctx["adapter"], "A": ctx["A"], "B": ctx["B"],
        "role": ctx["role"], "source_ym": ctx["source_ym"], "source_ys": ctx["source_ys"],
        "train_indices": ctx["train_indices"], "train_n": int(len(ctx["train_indices"])),
        "training_mask_contract": {
            "known_true_n": int(ctx["known"].sum()), "qc_true_n": int(ctx["qc"].sum()),
            "score_valid_true_n": int(ctx["score_valid"].sum()),
            "legal_n": int(ctx["legal"].sum()), "legal_indices_sha256": _hash_bytes(np.ascontiguousarray(ctx["train_indices"]).tobytes()),
            "known_mask_sha256": _hash_bytes(np.ascontiguousarray(ctx["known"]).tobytes()),
            "qc_mask_sha256": _hash_bytes(np.ascontiguousarray(ctx["qc"]).tobytes()),
            "score_valid_mask_sha256": _hash_bytes(np.ascontiguousarray(ctx["score_valid"]).tobytes()),
        },
        "model": model,
        "outer_index": ctx["outer_index"], "outer_record_id": ctx["outer_record_id"],
        "outer_pred": ctx["outer_pred"], "outer_truth": ctx["outer_truth"],
        "outer_qc": ctx["outer_qc"], "outer_score_valid": ctx["outer_score_valid"],
        "base_score": ctx["outer_score"], "candidate_within_score": candidate_within,
        "candidate_score": candidate_score, "baseline_replay": _base_score_replay(ctx),
        "test_labels_read": False,
    }
    return gate


def _write_fit_ledger_status(**extra: Any) -> None:
    path = r.ROOT / "FIT_LEDGER.json"
    ledger = r.read_json(path)
    totals = r.ledger_totals()
    ledger.update({"science_units": totals["science"], "repair_units": totals["repair"],
                   "total_units": totals["total"], "fit_calls_actual": totals["total"],
                   "optimizer_calls": totals["total"], "test_labels_read": False,
                   "candidate_test_read": False})
    ledger.update(extra)
    r.write_json(path, ledger)


def _review_receipt() -> dict[str, Any]:
    path = r.ROOT / "IMPLEMENTATION_PREFLIGHT_PASS.json"
    if not path.is_file():
        raise RuntimeError("independent implementation preflight receipt required before pilot")
    receipt = r.read_json(path)
    if receipt.get("status") not in {"PASS", "PASS_0FIT"}:
        raise RuntimeError(f"implementation receipt not PASS: {receipt.get('status')}")
    if receipt.get("driver_sha256") != r.sha256_file(Path(__file__)):
        raise RuntimeError("driver hash drift since independent preflight")
    if receipt.get("runtime_sha256") != r.sha256_file(r.HERE):
        raise RuntimeError("runtime hash drift since independent preflight")
    if receipt.get("preflight_sha256") != r.sha256_file(PREFLIGHT):
        raise RuntimeError("driver preflight hash drift since independent preflight")
    if receipt.get("recovery_manifest_sha256") != r.sha256_file(RECOVERY_MANIFEST):
        raise RuntimeError("recovery manifest hash drift since independent preflight")
    if receipt.get("test_labels_read") or receipt.get("test_evaluation_started"):
        raise RuntimeError("implementation receipt is not test closed")
    return receipt


def run_preflight() -> dict[str, Any]:
    """Read-only object/role/mask/identity preflight; no HGB import or fit."""
    paths = _manifest_paths()
    data = _load_metadata()
    selector = r.read_json(SELECTOR)
    if selector.get("test_labels_read") or selector.get("test_evaluated"):
        raise RuntimeError("selector is not test closed")
    rows: list[dict[str, Any]] = []
    for direction in r.DIRECTIONS:
        for seed in r.SEEDS:
            path, base_sha = _object_path(paths, direction, seed)
            obj = joblib.load(path)
            if obj.get("test_labels_hidden") is not True or obj.get("external_test_evaluated") is not False:
                raise RuntimeError(f"target object not test blind: {path}")
            truth_map = _truth_map(obj)
            for pos in r.POSITIONS:
                ctx = _context(obj, data, pos, truth_map)
                replay = _base_score_replay(ctx)
                rows.append({"direction": direction, "seed": int(seed), "position": int(pos),
                             "base_object_sha256": base_sha, "adapter": ctx["adapter"],
                             "feature_dim": int(ctx["features"].shape[1]), "A_n": int(len(ctx["A"])),
                             "B_n": int(len(ctx["B"])), "legal_n": int(len(ctx["train_indices"])),
                             "known_n": int(ctx["known"].sum()), "qc_n": int(ctx["qc"].sum()),
                             "score_valid_n": int(ctx["score_valid"].sum()),
                             "inner_folds": ctx["role"]["inner_folds"], "A_hash": ctx["role"]["A_hash"],
                             "B_hash": ctx["role"]["B_hash"], "train_indices_sha256": _hash_bytes(np.ascontiguousarray(ctx["train_indices"]).tobytes()),
                             "feature_hash": _hash_bytes(np.ascontiguousarray(ctx["features"]).tobytes()),
                             "baseline_replay": replay, "test_labels_read": False})
    expected = {(d, s, p) for d in r.DIRECTIONS for s in r.SEEDS for p in r.POSITIONS}
    actual = {(row["direction"], row["seed"], row["position"]) for row in rows}
    if actual != expected or len(rows) != 720:
        raise RuntimeError("driver preflight position universe mismatch")
    report = {
        "schema": "ridge_ma13d_hgb_driver_preflight_v1", "status": "PASS_0FIT",
        "fit_calls_actual": 0, "optimizer_calls": 0, "test_labels_read": False,
        "test_reference_rows_used": False, "medical_key_accessed": False,
        "objects": 180, "positions": 720, "directions": list(r.DIRECTIONS), "seeds": list(r.SEEDS),
        "runner_sha256": r.sha256_file(Path(__file__)), "runtime_sha256": r.sha256_file(r.HERE),
        "manifest_sha256": r.sha256_file(MANIFEST), "data15_sha256": r.sha256_file(DATA15),
        "selector_sha256": r.sha256_file(SELECTOR), "rows": rows,
        "candidate": "Ridge-10 + original MA13D equal-weight HistGradientBoostingClassifier",
        "training_contract": {
            "source": "frozen target object positions cross[adapter] outer-A/inner-V OOF",
            "outer_B": "held out for later selector only; never gate fit input",
            "feature_dim": 13, "sample_weight": "all_ones",
            "known_qc_score_masks": "full-row masks and exact legal indices are stored per position",
            "low_high_action": "original action unchanged; HGB score replaces only predicted-within score",
            "test": "not loaded, not evaluated, not used for selection",
        },
        "source_target_identity": {"source_full_expected": 70, "source_payload_expected": 120,
                                   "target_objects": 180, "target_object_hashes_checked": 180},
        "selector_hash": r.SELECTOR_RUNNER_SHA256,
        "fit_budget": {"science_cap": r.SCIENCE_CAP, "repair_cap": r.REPAIR_CAP,
                        "total_cap": r.TOTAL_CAP, "actual_science": 0, "actual_repair": 0},
    }
    r.write_json(PREFLIGHT, report)
    r.write_json(r.ROOT / "HGB_DRIVER_PREFLIGHT_ROWS.json", rows)
    r.write_json(r.ROOT / "HGB_DRIVER_STATE.json", {
        "schema": "ridge_ma13d_hgb_driver_state_v1",
        "status": "READY_FOR_INDEPENDENT_PREFLIGHT",
        "fit_calls_actual": 0, "optimizer_calls": 0,
        "test_labels_read": False, "test_evaluation_started": False,
        "pilot_scope": "9 directions × seeds 1000/1001 × positions 0-3 = 72",
        "full_scope": "9 directions × seeds 1000-1019 × positions 0-3 = 720",
        "preflight": "HGB_DRIVER_PREFLIGHT.json",
        "driver_sha256": report["runner_sha256"], "runtime_sha256": report["runtime_sha256"],
        "authorization_record": "USER_AUTHORIZATION_RECORD.json (human-authorized; independent receipt still required)",
    })
    _write_fit_ledger_status(current_phase="DRIVER_PREFLIGHT_PASS_0FIT", status="DRIVER_PREFLIGHT_PASS_0FIT",
                             driver_preflight_sha256=r.sha256_file(PREFLIGHT))
    return report


def run_fit(*, seeds: list[int], directions: list[str], pilot: bool) -> dict[str, Any]:
    receipt = _review_receipt()
    auth = r.require_fit_authorized()
    recovery_scope = _recovery_scope()
    if any(seed not in r.SEEDS for seed in seeds) or any(direction not in r.DIRECTIONS for direction in directions):
        raise ValueError("requested scope outside frozen 9×20 universe")
    r.acquire_lock("run_hgb_ma13d_driver")
    started = time.time()
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        paths = _manifest_paths()
        data = _load_metadata()
        for direction in directions:
            for seed in seeds:
                path, base_sha = _object_path(paths, direction, seed)
                artifact = _artifact_path(direction, seed)
                if artifact.exists():
                    prior = joblib.load(artifact)
                    if (prior.get("base_object_sha256") == base_sha and prior.get("test_labels_read") is False
                            and set(prior.get("gates", {})) == {"0", "1", "2", "3"}):
                        skipped.append({"direction": direction, "seed": int(seed), "reason": "completed_artifact_reused"})
                        continue
                    raise RuntimeError(f"artifact identity drift or incomplete artifact: {artifact}")
                obj = joblib.load(path)
                truth_map = _truth_map(obj)
                gates: dict[int, dict[str, Any]] = {}
                for pos in r.POSITIONS:
                    scope = f"{direction}__seed{int(seed)}__pos{int(pos)}__HGB"
                    rows = _scope_rows(scope)
                    position_tuple = (direction, int(seed), int(pos))
                    complete_rows = [row for row in rows if row.get("status") == "COMPLETE"]
                    if complete_rows and position_tuple not in recovery_scope:
                        raise RuntimeError(f"completed ledger unit outside recovery manifest: {scope}")
                    if complete_rows and artifact.exists():
                        raise RuntimeError(f"completed ledger unit has unexpected incomplete artifact state: {scope}")
                    repair_n = sum(1 for row in rows if row.get("kind") == "repair")
                    kind = "repair" if complete_rows or rows else "science"
                    suffix = "" if not rows else f"__retry{repair_n + 1}"
                    try:
                        partial = _load_partial_model(direction, int(seed), int(pos), base_sha)
                        gate = _fit_one(direction, int(seed), int(pos), obj, data, truth_map, base_sha, kind, suffix,
                                        model_override=None if partial is None else partial["model"])
                        gates[pos] = gate
                    except Exception as exc:
                        failures.append({"direction": direction, "seed": int(seed), "position": int(pos),
                                         "kind": kind, "error": f"{type(exc).__name__}: {exc}"})
                        break
                if len(gates) == 4:
                    saved = _save_artifact(direction, int(seed), obj, base_sha, gates)
                    completed.append({"direction": direction, "seed": int(seed), "artifact": str(saved)})
                if _stop_after_postfit_failure(failures):
                    # Preserve the first failure with its ledger state and
                    # stop immediately.  Continuing would create a large
                    # batch of unusable post-fit failures before the repair
                    # path has been reviewed.
                    break
            if _stop_after_postfit_failure(failures):
                break
        totals = r.ledger_totals()
        report = {"schema": "ridge_ma13d_hgb_fit_run_v1", "status": "COMPLETE" if not failures else "FAILED_UNITS_RETAINED",
                  "pilot": bool(pilot), "requested_directions": directions, "requested_seeds": seeds,
                  "completed_objects": completed, "skipped_objects": skipped, "failures": failures,
                  "fit_units": totals, "test_labels_read": False, "test_evaluation_started": False,
                  "driver_sha256": r.sha256_file(Path(__file__)), "runtime_sha256": r.sha256_file(r.HERE),
                  "preflight_sha256": r.sha256_file(PREFLIGHT), "receipt_sha256": r.sha256_file(r.ROOT / "IMPLEMENTATION_PREFLIGHT_PASS.json"),
                  "authorization_source": auth.get("authorization_source"), "elapsed_seconds": float(time.time() - started)}
        r.write_json(FIT_REPORT, report)
        _write_fit_ledger_status(status="HGB_FIT_COMPLETE_WITH_FAILURES_RETAINED" if failures else "HGB_FIT_COMPLETE",
                                 current_phase="HGB_FIT_COMPLETE", fit_run_report_sha256=r.sha256_file(FIT_REPORT),
                                 test_labels_read=False, candidate_test_read=False)
        return report
    finally:
        r.release_lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--pilot72", action="store_true")
    ap.add_argument("--complete720", action="store_true")
    ap.add_argument("--approved-fit", action="store_true")
    args = ap.parse_args()
    modes = [args.preflight, args.pilot72, args.complete720]
    if sum(bool(v) for v in modes) != 1:
        raise RuntimeError("choose exactly one mode")
    if args.preflight:
        report = run_preflight()
        print(json.dumps({"status": report["status"], "objects": report["objects"],
                          "positions": report["positions"], "fit_calls_actual": 0,
                          "test_labels_read": False}, ensure_ascii=False))
        return
    if not args.approved_fit:
        raise RuntimeError("fit requires --approved-fit plus independent receipt")
    seeds = list(PILOT_SEEDS) if args.pilot72 else list(r.SEEDS)
    report = run_fit(seeds=seeds, directions=list(r.DIRECTIONS), pilot=args.pilot72)
    print(json.dumps({"status": report["status"], "fit_units": report["fit_units"],
                      "completed_objects": len(report["completed_objects"]),
                      "failures": len(report["failures"]), "test_labels_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
