"""Implementation and 0-fit preflight for MA-TAIL-BALANCED-ERROR-LR.

The default command is a read-only preflight.  Optimisation is deliberately
guarded by an implementation-preflight receipt and --approved-fit so that
writing this runner can never accidentally start the 720-position round.
No test array or medical label is accessed by the preflight.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import runtime as r  # cache/thread setup must precede numerical imports
import joblib
import numpy as np


PROTOCOL_SHA = r.sha256(r.FROZEN_ROOT / "PROTOCOL.md")
PROMPT_SHA = r.sha256(r.FROZEN_ROOT / "EXECUTION_PROMPT_ZH.md")
STATE_SHA = r.sha256(r.FROZEN_ROOT / "STATE.json")
RECEIPT_SHA = r.sha256(r.FROZEN_ROOT / "PROTOCOL_REVIEW_RECEIPT.json")
BASE_MANIFEST = r.BASE_ROOT / "FULL_MODELS_FROZEN.json"
BASE_IDENTITY = r.BASE_ROOT / "MODEL_IDENTITY.json"
BASE_THRESHOLD = r.BASE_ROOT / "THRESHOLD_FREEZE.json"
BASE_CONTRACT = r.BASE_ROOT / "OUTER_B_THRESHOLD_CONTRACT.json"
OLD_RUNNER = r.BASE_ROOT / "run_ridge10_fullchain.py"


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    """The exact 13D numeric-MA feature contract, not CI features."""
    x = np.asarray(x, dtype=float)
    med = np.asarray(med, dtype=float)
    pred_ma = med[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((
        x,
        (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float),
    ))


def preprocess_fit(f: np.ndarray) -> dict[str, Any]:
    f = np.asarray(f, dtype=float)
    finite = np.isfinite(f)
    median = np.asarray([np.median(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    mean = np.asarray([np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    sd = np.asarray([max(np.std(f[finite[:, j], j]), 1e-6) if finite[:, j].any() else 1.0 for j in range(f.shape[1])])
    raw = (np.where(finite, f, median[None, :]) - mean[None, :]) / sd[None, :]
    return {"mean": mean, "sd": sd, "median": median, "ddof": 0, "floor": 1e-6,
            "clip": [-10.0, 10.0], "n": int(len(f)), "p": int(f.shape[1]),
            "missing_fraction": float(np.mean(~finite)) if f.size else 0.0,
            "clip_fraction": float(np.mean(np.abs(raw) > 10.0)) if f.size else 0.0}


def empty_pre(p: int) -> dict[str, Any]:
    return {"mean": np.zeros(p), "sd": np.ones(p), "median": np.zeros(p), "ddof": 0,
            "floor": 1e-6, "clip": [-10.0, 10.0], "n": 0, "p": int(p),
            "missing_fraction": 0.0, "clip_fraction": 0.0}


def _score_model(model: Mapping[str, Any], f: np.ndarray, pred: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    pre = model["pre"]
    finite = np.isfinite(f)
    z = np.clip((np.where(finite, f, np.asarray(pre["median"])[None, :])
                 - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    if model.get("fallback"):
        score = np.full(len(f), float(model["constant_score"]))
    else:
        eta = z @ np.asarray(model["coef"], dtype=float) + float(model["intercept"])
        score = np.where(eta >= 0.0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
    valid = np.isin(np.asarray(pred), r.MA_CLASSES) & np.isfinite(f).all(axis=1)
    return np.where(valid, score, np.inf)


def _reference_function(name: str):
    """AST extract one old pure definition without importing its runtime."""
    tree = ast.parse(OLD_RUNNER.read_text(encoding="utf-8-sig"), filename=str(OLD_RUNNER))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"expected exactly one old {name} definition")
    ns: dict[str, Any] = {"np": np, "Any": Any, "ma_class": ma_class}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(OLD_RUNNER), "exec"), ns)
    return ns[name], ast.dump(nodes[0], include_attributes=False)


def group_weights(pred: np.ndarray, truth: np.ndarray, f: np.ndarray) -> dict[str, Any]:
    """Compute the fixed three-group weights without fabricating absent groups."""
    pred = np.asarray(pred, dtype=np.int8)
    truth = np.asarray(truth, dtype=np.int8)
    f = np.asarray(f, dtype=float)
    # Match the original MA-ERROR-LR legal mask: labels decide membership;
    # finite feature values are imputed by the fold-fitted preprocessor.
    legal = np.isin(pred, r.MA_CLASSES) & np.isin(truth, r.MA_CLASSES)
    idx = np.flatnonzero(legal)
    p = pred[legal]
    t = truth[legal]
    names = np.where((t == -1) & (p == 0), "G_low", np.where((t == 1) & (p == 0), "G_high", "G_other"))
    present = [name for name in ("G_low", "G_high", "G_other") if np.any(names == name)]
    n = int(len(idx))
    k = len(present)
    weights = np.zeros(n, dtype=float)
    counts: dict[str, int] = {}
    if k:
        for name in present:
            mask = names == name
            count = int(mask.sum())
            counts[name] = count
            weights[mask] = 1.0 / (k * count)
        mean = float(weights.mean()) if n else 1.0
        if n and mean > 0:
            weights /= mean
    return {"legal_mask": legal, "indices": idx, "groups": names, "present": present,
            "counts": counts, "n": n, "k": k, "weights": weights,
            "sumw": float(weights.sum()), "meanw": float(weights.mean()) if n else 1.0,
            "errors": int(np.sum(p != t)), "correct": int(np.sum(p == t))}


def weighted_bce_objective(theta: np.ndarray, z: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Single weighted BCE+L2 objective used by both fit and FD preflight."""
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    p = z.shape[1]
    coef = theta[:p]
    eta = z @ coef + theta[p]
    prob = np.where(eta >= 0.0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
    n = len(y)
    loss = float(np.sum(weights * (np.logaddexp(0.0, eta) - y * eta)) / n + 0.1 * np.sum(coef ** 2))
    residual = weights * (prob - y)
    grad_coef = z.T @ residual / n + 0.2 * coef
    grad_intercept = float(np.sum(residual) / n)
    return loss, np.r_[grad_coef, grad_intercept]


def weighted_bce_contract(weights: np.ndarray, y: np.ndarray, eta: np.ndarray, coef: np.ndarray,
                          z: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Compatibility wrapper for the pure contract test; no optimizer."""
    weights = np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=float)
    coef = np.asarray(coef, dtype=float)
    if z is None:
        z = np.zeros((len(y), len(coef)), dtype=float)
    theta = np.r_[coef, float(np.mean(eta) if len(eta) else 0.0)]
    return weighted_bce_objective(theta, z, y, weights)


def fit_weighted_gate(f: np.ndarray, pred: np.ndarray, truth: np.ndarray, fit_id: str, seed: int, attempt_key: str) -> dict[str, Any]:
    """Future fit entrypoint; never reachable from preflight.

    The caller must register attempt_key before invoking this function.
    Small/one-sided data use the frozen constant fallback.  Optimizer errors
    are FAILED rather than silently converted to fallback.
    """
    # Re-check the complete receipt here as well as in the outer driver: a
    # direct caller must not bypass runner/runtime/preflight hash binding.
    _current_preflight_receipt()
    # The attempt must already be persisted as PENDING.  This transition is
    # the only legal route into either the fallback result or scipy optimizer.
    r.begin_attempt(attempt_key)
    from scipy.optimize import minimize
    from scipy.special import expit

    f = np.asarray(f, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    truth = np.asarray(truth, dtype=np.int8)
    info = group_weights(pred, truth, f)
    n = info["n"]
    errors = info["errors"]
    correct = info["correct"]
    constant = float((errors + 0.5) / (n + 1.0)) if n else 0.5
    out: dict[str, Any] = {"api_version": "ma-tail-balanced-error-lr-v1", "id": "MA-TAIL-BALANCED-ERROR-LR",
        "fit_id": fit_id, "fit_seed": int(seed), "feature_dim": int(f.shape[1]), "regularizer": 0.1,
        "n": n, "errors": errors, "correct": correct, "groups": info["counts"], "present_groups": info["present"],
        "weights_sum": info["sumw"], "weights_mean": info["meanw"], "pre": preprocess_fit(f[info["indices"]]) if n else empty_pre(f.shape[1]),
        "constant_score": constant, "fallback": False, "success": False, "status": "not_started",
        "coef": np.zeros(f.shape[1]), "intercept": float(math.log(constant / (1.0 - constant))) if 0 < constant < 1 else 0.0}
    if n < 8 or errors < 2 or correct < 2:
        out.update({"fallback": True, "fallback_reason": "small_or_one_sided", "success": True, "status": "fallback_constant"})
        return out
    legal = f[info["indices"]]
    y = (pred[info["indices"]] != truth[info["indices"]]).astype(float)
    w = info["weights"]
    pre = out["pre"]
    z = np.clip((np.where(np.isfinite(legal), legal, np.asarray(pre["median"])[None, :])
                 - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    p = z.shape[1]
    n_float = float(n)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return weighted_bce_objective(theta, z, y, w)

    sol = minimize(objective, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
    theta = np.asarray(sol.x, dtype=float)
    if not bool(sol.success) or not np.isfinite(theta).all() or not np.isfinite(sol.fun):
        raise RuntimeError(f"weighted gate optimizer failed: {sol.message}")
    out.update({"success": True, "status": str(sol.message), "objective": float(sol.fun),
                "iterations": int(getattr(sol, "nit", 0)), "coef": theta[:-1], "intercept": float(theta[-1])})
    return out


def _path_from_manifest(rel: str) -> Path:
    return r.BASE_ROOT / Path(*rel.replace("/", "\\").split("\\"))


def _assert_hashes() -> dict[str, Any]:
    manifest = r.read_json(BASE_MANIFEST)
    if manifest.get("status") != "ALL_MODELS_AND_THRESHOLDS_FROZEN_TEST_BLIND":
        raise RuntimeError(f"base manifest status is not test-blind frozen: {manifest.get('status')}")
    if manifest.get("test_evaluated") or manifest.get("test_labels_read") or manifest.get("threshold_selection_from_test"):
        raise RuntimeError("base manifest indicates test access/evaluation")
    hashes = manifest.get("target_object_hashes")
    if not isinstance(hashes, dict) or len(hashes) != 180:
        raise RuntimeError("base manifest does not contain exactly 180 target object hashes")
    missing: list[str] = []
    mismatched: list[str] = []
    for rel, expected in sorted(hashes.items()):
        path = _path_from_manifest(rel)
        if not path.exists():
            missing.append(rel)
            continue
        got = r.sha256(path)
        if got.lower() != str(expected).lower():
            mismatched.append(rel)
    if missing or mismatched:
        raise RuntimeError(f"base target identity mismatch; missing={len(missing)} mismatched={len(mismatched)}")
    return {"manifest_status": manifest["status"], "target_objects": len(hashes), "missing": missing,
            "mismatched": mismatched, "manifest_sha256": r.sha256(BASE_MANIFEST)}


def _truth_map(obj: Mapping[str, Any]) -> tuple[dict[int, int], dict[str, Any]]:
    truth: dict[int, int] = {}
    rows: list[int] = []
    for pos in (0, 1, 2):
        q = obj["positions"][pos]
        b = [int(v) for v in q["B"]]
        data = obj["gate_models"][pos]["outer_B_data"]
        pool_index = [int(v) for v in data["pool_index"]]
        values = [int(v) for v in data["truth_class"]]
        if b != pool_index or len(values) != len(b):
            raise RuntimeError("outer-B role/order does not match saved outer_B_data")
        for idx, value in zip(b, values):
            if idx in truth:
                raise RuntimeError(f"duplicate outer-B union row: {idx}")
            if value not in r.MA_CLASSES:
                raise RuntimeError(f"invalid development MA class: {value}")
            truth[idx] = value
            rows.append(idx)
    if len(truth) != len(obj["pool"]):
        raise RuntimeError(f"outer-B union does not cover pool: {len(truth)} vs {len(obj['pool'])}")
    if set(truth) != set(map(int, obj["pool"])):
        raise RuntimeError("outer-B union and pool indices differ")
    return truth, {"outer_B_union_n": len(truth), "outer_B_unique": len(set(rows)), "pool_n": len(obj["pool"])}


def _check_role_split(obj: Mapping[str, Any], pos: int, truth_map: Mapping[int, int]) -> dict[str, Any]:
    q = obj["positions"][pos]
    A = [int(v) for v in q["A"]]
    B = [int(v) for v in q["B"]]
    inner = q["inner"]
    if len(A) == 0 or len(set(A)) != len(A):
        raise RuntimeError(f"position {pos} has empty/duplicate A")
    if set(A) & set(B):
        raise RuntimeError(f"position {pos} A/B overlap")
    if len(inner) != 3:
        raise RuntimeError(f"position {pos} does not have exactly 3 inner folds")
    seen_v: list[int] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in inner:
        T = [int(v) for v in fold["T"]]
        V = [int(v) for v in fold["V"]]
        if set(T) & set(V) or not T or not V:
            raise RuntimeError(f"position {pos} inner T/V overlap/empty")
        if set(T) | set(V) != set(A):
            raise RuntimeError(f"position {pos} inner T/V does not partition A")
        seen_v.extend(V)
        fold_rows.append({"T_n": len(T), "V_n": len(V),
                          "T_hash": r.hash_object_bytes(",".join(map(str, sorted(T))).encode()),
                          "V_hash": r.hash_object_bytes(",".join(map(str, sorted(V))).encode())})
    if sorted(seen_v) != sorted(A):
        raise RuntimeError(f"position {pos} validation folds do not cover A exactly once")
    if sorted(set(A) - set(truth_map)):
        raise RuntimeError(f"position {pos} A rows lack C-fold truth mapping")
    return {"A_n": len(A), "B_n": len(B), "inner_folds": fold_rows,
            "A_hash": r.hash_object_bytes(",".join(map(str, A)).encode()),
            "B_hash": r.hash_object_bytes(",".join(map(str, B)).encode())}


def _replay_one(obj: Mapping[str, Any], data: Mapping[str, Any], pos: int, truth_map: Mapping[int, int], reference_ma_features, reference_preprocess) -> dict[str, Any]:
    q = obj["positions"][pos]
    gm = obj["gate_models"][pos]
    adapter = str(gm["adapter"])
    A = np.asarray(q["A"], dtype=int)
    B = np.asarray(q["B"], dtype=int)
    cross = np.asarray(q["cross"][adapter], dtype=float)
    if cross.shape != (len(A), 4):
        raise RuntimeError(f"position {pos} adapter cross shape mismatch")
    src = obj["source_model"]
    ym = np.asarray(src["ym"], dtype=float)
    ys = np.asarray(src["ys"], dtype=float)
    if ym.shape != (4,) or ys.shape != (4,) or not np.isfinite(ym).all() or not np.isfinite(ys).all() or np.any(ys == 0):
        raise RuntimeError(f"position {pos} invalid source scaler")
    f = ma_features(data["x"][A], cross, ym, ys)
    f_ref = np.asarray(reference_ma_features(data["x"][A], cross, ym, ys), dtype=float)
    if f.shape != (len(A), 13) or not np.array_equal(f, f_ref):
        raise RuntimeError(f"position {pos} candidate 13D does not match AST-extracted old ma_features")
    if not np.isfinite(f).all():
        raise RuntimeError(f"position {pos} has nonfinite numeric-MA feature")
    pred = ma_class(cross[:, 2])
    truth = np.asarray([truth_map[int(v)] for v in A], dtype=np.int8)
    if not np.isin(pred, r.MA_CLASSES).all() or not np.isin(truth, r.MA_CLASSES).all():
        raise RuntimeError(f"position {pos} invalid MA class in gate rows")
    info = group_weights(pred, truth, f)
    p_ref = reference_preprocess(f)
    p_new = preprocess_fit(f)
    if not np.array_equal(np.asarray(p_ref["mean"]), np.asarray(p_new["mean"])) or not np.array_equal(np.asarray(p_ref["sd"]), np.asarray(p_new["sd"])):
        raise RuntimeError(f"position {pos} preprocessor drift")
    baseline_score_ok = True
    baseline_threshold_ok = True
    threshold_summary: dict[str, Any] | None = None
    if len(B):
        vv = q.get("vv", {}).get(adapter)
        if vv is None:
            raise RuntimeError(f"position {pos} missing its own outer-B prediction")
        vv = np.asarray(vv, dtype=float)
        if vv.shape != (len(B), 4):
            raise RuntimeError(f"position {pos} outer-B med shape mismatch")
        f_b = ma_features(data["x"][B], vv, ym, ys)
        pred_b = ma_class(vv[:, 2])
        replay = _score_model(gm["model"], f_b, pred_b)
        saved = np.asarray(gm["outer_B_data"]["score"], dtype=float)
        if replay.shape != saved.shape:
            raise RuntimeError(f"position {pos} baseline score shape mismatch")
        equal = (np.isinf(replay) & np.isinf(saved)) | (np.isfinite(replay) & np.isfinite(saved) & np.isclose(replay, saved, rtol=0.0, atol=1e-12))
        baseline_score_ok = bool(np.all(equal))
        if not baseline_score_ok:
            raise RuntimeError(f"position {pos} baseline score replay mismatch")
        threshold = gm.get("threshold")
        if not isinstance(threshold, dict) or "threshold" not in threshold or not isinstance(threshold.get("candidates"), list):
            raise RuntimeError(f"position {pos} missing frozen threshold candidates")
        if not any(c.get("threshold") is None for c in threshold["candidates"] if isinstance(c, dict)):
            raise RuntimeError(f"position {pos} threshold candidates lack all-withhold")
        tau = threshold.get("threshold")
        # The frozen MA summary counts every valid predicted class; downstream
        # action keeps low/high as their original class.  The new candidate
        # may replace only the predicted-within score, never this baseline
        # validity/action contract.
        release = np.isfinite(saved) & np.isin(pred_b, r.MA_CLASSES) & (saved <= float(tau) if tau is not None else False)
        true_b = np.asarray(gm["outer_B_data"]["truth_class"], dtype=np.int8)
        errors = int(np.sum(release & (pred_b != true_b)))
        if errors != int(threshold.get("errors", errors)) or int(release.sum()) != int(threshold.get("released", int(release.sum()))):
            baseline_threshold_ok = False
            raise RuntimeError(f"position {pos} baseline threshold action replay mismatch direction={obj.get('direction')} seed={obj.get('seed')} calc_released={int(release.sum())} saved_released={threshold.get('released')} calc_errors={errors} saved_errors={threshold.get('errors')}")
        threshold_summary = {"threshold": tau, "released": int(release.sum()), "errors": errors,
                             "risk": threshold.get("risk"), "threshold_source": threshold.get("threshold_source")}
    else:
        if gm.get("outer_B_data", {}).get("pool_index") not in ([], None):
            raise RuntimeError(f"position {pos} empty B has nonempty outer_B_data")
        threshold_summary = {"threshold": gm.get("threshold", {}).get("threshold") if isinstance(gm.get("threshold"), dict) else None,
                             "released": 0, "errors": 0, "threshold_source": gm.get("threshold_source")}
    return {"position": pos, "adapter": adapter, "A_n": len(A), "B_n": len(B),
            "feature_dim": int(f.shape[1]), "feature_hash": r.hash_object_bytes(np.ascontiguousarray(f).tobytes()),
            "pred_counts": {str(k): int(np.sum(pred == k)) for k in r.MA_CLASSES},
            "truth_counts": {str(k): int(np.sum(truth == k)) for k in r.MA_CLASSES},
            "group_counts": info["counts"], "present_groups": info["present"], "legal_n": info["n"],
            "errors": info["errors"], "correct": info["correct"], "sumw": info["sumw"], "meanw": info["meanw"],
            "fallback_rule": {"small_n": info["n"] < 8, "errors_lt2": info["errors"] < 2, "correct_lt2": info["correct"] < 2},
            "baseline_score_replay": baseline_score_ok, "baseline_threshold_replay": baseline_threshold_ok,
            "baseline_threshold": threshold_summary}


def _synthetic_contract_tests() -> dict[str, Any]:
    pred = np.asarray([-1, 0, 0, 1, 0, 0], dtype=np.int8)
    truth = np.asarray([-1, -1, 1, 1, 0, 0], dtype=np.int8)
    f = np.arange(78, dtype=float).reshape(6, 13)
    info = group_weights(pred, truth, f)
    if info["present"] != ["G_low", "G_high", "G_other"] or abs(info["sumw"] - 6.0) > 1e-12 or abs(info["meanw"] - 1.0) > 1e-12:
        raise RuntimeError("synthetic group-weight contract failed")
    z = np.linspace(-0.4, 0.7, 6 * 13, dtype=float).reshape(6, 13)
    theta = np.linspace(-0.2, 0.3, 14, dtype=float)
    loss, grad = weighted_bce_objective(theta, z, (pred != truth).astype(float), info["weights"])
    if not np.isfinite(loss) or grad.shape != (14,):
        raise RuntimeError("synthetic weighted BCE contract failed")
    fd = np.empty_like(theta)
    h = 1e-6
    for j in range(len(theta)):
        plus = theta.copy(); minus = theta.copy(); plus[j] += h; minus[j] -= h
        fd[j] = (weighted_bce_objective(plus, z, (pred != truth).astype(float), info["weights"])[0]
                 - weighted_bce_objective(minus, z, (pred != truth).astype(float), info["weights"])[0]) / (2.0 * h)
    if float(np.max(np.abs(fd - grad))) > 2e-5:
        raise RuntimeError("synthetic weighted BCE finite-difference gradient failed")
    info2 = group_weights(np.zeros(4, dtype=np.int8), np.zeros(4, dtype=np.int8), np.zeros((4, 13)))
    if info2["present"] != ["G_other"] or abs(info2["sumw"] - 4.0) > 1e-12:
        raise RuntimeError("synthetic missing-group contract failed")
    f_bad = f.copy(); f_bad[0, 0] = np.nan
    info_bad = group_weights(pred, truth, f_bad)
    if info_bad["n"] != len(pred) or abs(info_bad["sumw"] - len(pred)) > 1e-12:
        raise RuntimeError("synthetic nonfinite feature legal-mask contract failed")
    pre_bad = preprocess_fit(f_bad)
    if not np.isfinite(np.asarray(pre_bad["mean"])).all() or not np.isfinite(np.asarray(pre_bad["sd"])).all():
        raise RuntimeError("synthetic nonfinite feature preprocessing failed")
    # Import the independent selector module only for its no-fit contract
    # checks; it does not open DATA15 or any target test field.
    from selector_weighted_gate import synthetic_selector_contract_tests
    selector_synth = synthetic_selector_contract_tests()
    return {"weighted_bce": "PASS", "present_group": "PASS", "absent_group_no_fabrication": "PASS",
            "nonfinite_feature_imputed": "PASS", "finite_difference_gradient": "PASS",
            "selector_contract": selector_synth, "optimizer_calls": 0}


def run_preflight() -> dict[str, Any]:
    started = time.time()
    if not r.FROZEN_ROOT.exists() or not r.BASE_ROOT.exists():
        raise RuntimeError("frozen protocol/base directory missing")
    frozen = r.frozen_hashes()
    base_hash = r.base_hashes()
    identity = _assert_hashes()
    base_identity = r.read_json(BASE_IDENTITY)
    if base_identity.get("model_family") != "RIDGE" or base_identity.get("config") != "RIDGE-10" or float(base_identity.get("alpha")) != 10.0:
        raise RuntimeError("fixed Ridge-10 identity mismatch")
    if base_identity.get("target_test_evaluated") or base_identity.get("test_labels_read"):
        raise RuntimeError("base identity is not test blind")
    if not r.DATA_PATH.exists() or not r.ROLE_PATH.exists():
        raise RuntimeError("frozen DATA15/ROLE_FREEZE missing")
    data = r.load_metadata_only_data()
    if "medical" in data:
        raise RuntimeError("medical key unexpectedly loaded into preflight data")
    if len(data["x"]) != len(data["ids"]) or len(data["x"]) != len(data["input_qc"]):
        raise RuntimeError("DATA15 metadata arrays have inconsistent lengths")
    if len(set(map(str, data["ids"].tolist()))) != len(data["ids"]):
        raise RuntimeError("DATA15 IDs are not unique")
    reference_ma_features, ma_ast = _reference_function("ma_features")
    reference_preprocess, pre_ast = _reference_function("_preprocess_fit")
    synth = _synthetic_contract_tests()
    manifest = r.read_json(BASE_MANIFEST)
    rows: list[dict[str, Any]] = []
    object_count = 0
    position_count = 0
    for rel in sorted(manifest["target_object_hashes"]):
        path = _path_from_manifest(rel)
        obj = joblib.load(path)
        object_count += 1
        if obj.get("test_labels_hidden") is not True or obj.get("external_test_evaluated") is not False:
            raise RuntimeError(f"target object is not test-blind: {rel}")
        direction = str(obj.get("direction"))
        seed = int(obj.get("seed"))
        if direction not in r.DIRECTIONS or seed not in r.SEEDS:
            raise RuntimeError(f"invalid direction/seed in object: {rel}")
        if not obj.get("protocol_sha256"):
            raise RuntimeError(f"missing object protocol identity: {rel}")
        truth_map, _union = _truth_map(obj)
        for pos in range(4):
            split = _check_role_split(obj, pos, truth_map)
            replay = _replay_one(obj, data, pos, truth_map, reference_ma_features, reference_preprocess)
            rows.append({"direction": direction, "seed": seed, **split, **replay})
            position_count += 1
    if object_count != 180 or position_count != 720:
        raise RuntimeError(f"object/position count mismatch: {object_count}/{position_count}")
    if len({row["direction"] + str(row["seed"]) for row in rows}) != 180:
        raise RuntimeError("direction/seed object identity is not unique")
    if not all(abs(float(row["sumw"]) - float(row["legal_n"])) <= 1e-10 and abs(float(row["meanw"]) - 1.0) <= 1e-12 for row in rows):
        raise RuntimeError("weighted rows do not satisfy sumw=N/meanw=1")
    report = {
        "schema": "ma_tail_weight_gate_implementation_preflight_v1",
        "status": "PASS_0FIT" if rows else "FAIL",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": "MA-TAIL-BALANCED-ERROR-LR",
        "fit_units": {"science": 0, "repair": 0, "total": 0},
        "optimizer_calls": 0, "test_labels_read": False, "test_reference_rows_used": False,
        "medical_key_accessed": False, "test_container_fields_accessed": False,
        "objects": object_count, "positions": position_count,
        "directions": list(r.DIRECTIONS), "seeds": list(r.SEEDS),
        "frozen_protocol_hashes": frozen, "base_hashes": base_hash,
        "base_data_sha256": r.sha256(r.DATA_PATH), "role_sha256": r.sha256(r.ROLE_PATH),
        "base_manifest_identity": identity, "base_identity": base_identity,
        "runner_sha256": r.sha256(Path(__file__)), "runtime_sha256": r.sha256(Path(r.__file__)),
        "old_runner_sha256": r.sha256(OLD_RUNNER), "old_ma_features_ast_sha256": r.hash_object_bytes(ma_ast.encode()),
        "old_preprocess_ast_sha256": r.hash_object_bytes(pre_ast.encode()),
        "selector_runner_sha256": r.sha256(r.ROOT / "selector_weighted_gate.py"),
        "source_scaler_contract": "target object source_model ym/ys, four outputs; finite nonzero",
        "role_contract": "pos0-2 outer-A inner-V OOF for gate training; pos3 full-C inner-V OOF; outer-B only heldout replay/selector",
        "selector_contract": {
            "pool": "three-fold merged outer-B positions0-2 only; inner-V/pos3 training scores prohibited",
            "arms": ["BASE_FROZEN", "BASE_JOINT", "WEIGHTED_JOINT"],
            "strict_goal": "agreement>0.75 and coverage>0.20; low/high-to-within=0; correct-within distinct>=5; both tails supported",
            "same_coverage": "descriptive only; no fractional tie enters actions or promotion",
        },
        "effective_goal_contract_sha256": r.sha256(r.ROOT / "EFFECTIVE_GOAL_CONTRACT.json"),
        "baseline_replay": {"score": "PASS", "frozen_threshold_action_dev_pool": "PASS", "test": "NOT_RUN"},
        "synthetic_contract_tests": synth,
        "row_summary": {"present_group_counts": {name: int(sum(name in row["present_groups"] for row in rows)) for name in ("G_low", "G_high", "G_other")},
                        "fallback_positions": int(sum(any(row["fallback_rule"].values()) for row in rows)),
                        "all_positions_sumw_ok": True, "all_objects_test_blind": True},
        "elapsed_seconds": float(time.time() - started),
        "next_step": "Independent nine_ma_verify review required before any optimizer; this file does not authorize fit.",
    }
    r.write_json("IMPLEMENTATION_PREFLIGHT.json", report)
    r.write_json("PREFLIGHT_ROWS.json", rows)
    r.write_json("CURRENT_STATUS.json", {"status": "IMPLEMENTATION_PREFLIGHT_PASS_0FIT", "test_labels_read": False,
                                         "test_evaluation_started": False, "science_units": 0, "repair_units": 0,
                                         "total_units": 0, "preflight": "IMPLEMENTATION_PREFLIGHT.json",
                                         "runner_sha256": report["runner_sha256"], "runtime_sha256": report["runtime_sha256"]})
    r.update_ledger_summary(status="IMPLEMENTATION_PREFLIGHT_PASS_0FIT", current_phase="IMPLEMENTATION_PREFLIGHT_PASS_0FIT",
                            preflight_sha256=r.sha256(r.ROOT / "IMPLEMENTATION_PREFLIGHT.json"))
    return report


def _current_preflight_receipt() -> dict[str, Any]:
    receipt_path = r.ROOT / "IMPLEMENTATION_PREFLIGHT_PASS.json"
    if not receipt_path.exists():
        raise RuntimeError("independent implementation preflight receipt is required before fit")
    receipt = r.read_json(receipt_path)
    if receipt.get("status") not in {"PASS", "PASS_0FIT"}:
        raise RuntimeError(f"implementation receipt is not PASS: {receipt.get('status')}")
    current_runner = r.sha256(Path(__file__))
    current_runtime = r.sha256(Path(r.__file__))
    preflight = r.ROOT / "IMPLEMENTATION_PREFLIGHT.json"
    if receipt.get("runner_sha256") != current_runner or receipt.get("runtime_sha256") != current_runtime:
        raise RuntimeError("implementation receipt runner/runtime hash drift")
    if not preflight.exists() or receipt.get("preflight_sha256") != r.sha256(preflight):
        raise RuntimeError("implementation receipt preflight hash drift")
    if receipt.get("test_labels_read") or receipt.get("test_evaluation_started"):
        raise RuntimeError("implementation receipt has test access")
    return receipt


def _base_object_manifest() -> dict[str, str]:
    manifest = r.read_json(BASE_MANIFEST)
    out: dict[str, str] = {}
    for rel in manifest["target_object_hashes"]:
        path = _path_from_manifest(rel)
        out[str(path)] = str(manifest["target_object_hashes"][rel]).lower()
    return out


def _base_path_for(direction: str, seed: int, manifest: Mapping[str, str]) -> tuple[Path, str]:
    target = r.TARGET_BY_DIRECTION[direction]
    suffix = f"{target}_CFULL_seed{int(seed)}.joblib"
    matches = [Path(path) for path in manifest if Path(path).parts[-2:] == ("target15", suffix)]
    # Direction is part of the path and avoids ambiguous target names.
    matches = [path for path in matches if direction in path.parts]
    if len(matches) != 1:
        raise RuntimeError(f"base object lookup failed for {direction} seed{seed}: {len(matches)}")
    return matches[0], manifest[str(matches[0])]


def _candidate_gate_context(obj: Mapping[str, Any], data: Mapping[str, Any], pos: int,
                            truth_map: Mapping[int, int]) -> dict[str, Any]:
    q = obj["positions"][pos]
    gm = obj["gate_models"][pos]
    adapter = str(gm["adapter"])
    A = np.asarray(q["A"], dtype=int)
    B = np.asarray(q["B"], dtype=int)
    src = obj["source_model"]
    f = ma_features(data["x"][A], np.asarray(q["cross"][adapter], dtype=float), src["ym"], src["ys"])
    pred = ma_class(np.asarray(q["cross"][adapter], dtype=float)[:, 2])
    truth = np.asarray([truth_map[int(v)] for v in A], dtype=np.int8)
    if not np.isfinite(f).all() or not np.isin(pred, r.MA_CLASSES).all() or not np.isin(truth, r.MA_CLASSES).all():
        # The legal mask preserves nonfinite features for original imputation;
        # class labels remain fail-closed and all score rows must be finite.
        if not np.isin(pred, r.MA_CLASSES).all() or not np.isin(truth, r.MA_CLASSES).all():
            raise RuntimeError(f"invalid MA class in candidate context pos{pos}")
    out: dict[str, Any] = {"position": pos, "adapter": adapter, "A": A, "B": B,
                            "features": f, "pred": pred, "truth": truth,
                            "group_info": group_weights(pred, truth, f)}
    if len(B):
        vv = np.asarray(q.get("vv", {}).get(adapter), dtype=float)
        f_b = ma_features(data["x"][B], vv, src["ym"], src["ys"])
        pred_b = ma_class(vv[:, 2])
        base_score = np.asarray(gm["outer_B_data"]["score"], dtype=float)
        if base_score.shape != (len(B),):
            raise RuntimeError(f"outer-B baseline score shape mismatch pos{pos}")
        out.update({"outer_features": f_b, "outer_pred": pred_b, "outer_truth": np.asarray(gm["outer_B_data"]["truth_class"], dtype=np.int8),
                    "outer_score": base_score, "outer_index": B,
                    "outer_record_id": [str(v) for v in gm["outer_B_data"]["record_id"]]})
    else:
        out.update({"outer_features": np.empty((0, 13), dtype=float), "outer_pred": np.empty(0, dtype=np.int8),
                    "outer_truth": np.empty(0, dtype=np.int8), "outer_score": np.empty(0, dtype=float),
                    "outer_index": np.empty(0, dtype=int), "outer_record_id": []})
    return out


def _artifact_path(direction: str, seed: int) -> Path:
    target = r.TARGET_BY_DIRECTION[direction]
    return r.ROOT / "models" / direction / "target15" / f"{target}_WTG_seed{int(seed)}.joblib"


def _attempt_rows_for_scope(scope: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(r.LEDGER_DIR.glob("*.json")):
        row = r.read_json(path)
        if row.get("scope") == scope:
            rows.append(row)
    return rows


def _next_attempt(scope: str) -> tuple[str, str]:
    rows = _attempt_rows_for_scope(scope)
    for row in rows:
        if row.get("status") in {"PENDING", "RUNNING"}:
            raise RuntimeError(f"unfinished attempt blocks rerun: {row.get('key')}")
    if any(row.get("status") == "COMPLETE" for row in rows):
        raise RuntimeError(f"completed attempt already exists for scope: {scope}")
    if not rows:
        return scope, "science"
    retries = sum(1 for row in rows if row.get("kind") == "repair") + 1
    return f"{scope}__retry{retries}", "repair"


def _save_candidate_artifact(direction: str, seed: int, obj: Mapping[str, Any], base_sha: str,
                             gates: Mapping[int, Mapping[str, Any]], receipt: Mapping[str, Any]) -> Path:
    out = {"schema": "ma_tail_weight_gate_target_object_v1", "candidate": "MA-TAIL-BALANCED-ERROR-LR",
           "direction": direction, "target": r.TARGET_BY_DIRECTION[direction], "seed": int(seed),
           "external_test_evaluated": False, "test_labels_hidden": True, "test_labels_read": False,
           "base_object_sha256": base_sha, "base_protocol_sha256": str(obj.get("protocol_sha256")),
           "candidate_runner_sha256": r.sha256(Path(__file__)), "candidate_runtime_sha256": r.sha256(Path(r.__file__)),
           "implementation_preflight_sha256": r.sha256(r.ROOT / "IMPLEMENTATION_PREFLIGHT.json"),
           "gates": {str(k): dict(v) for k, v in gates.items()},
           "source_family": obj.get("source_family"), "source_centers": obj.get("source_centers"),
           "fit_units": 4, "fit_failures": 0}
    path = _artifact_path(direction, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(out, tmp, compress=3)
    tmp.replace(path)
    return path


def run_fits(*, seeds: list[int], directions: list[str], pilot: bool) -> dict[str, Any]:
    receipt = _current_preflight_receipt()
    if any(seed not in r.SEEDS for seed in seeds) or any(direction not in r.DIRECTIONS for direction in directions):
        raise ValueError("requested direction/seed outside frozen scope")
    r.acquire_lock("run_weighted_gate.py fit")
    started = time.time(); failures: list[dict[str, Any]] = []; completed: list[dict[str, Any]] = []; skipped: list[dict[str, Any]] = []
    try:
        data = r.load_metadata_only_data()
        manifest = _base_object_manifest()
        for direction in directions:
            for seed in seeds:
                base_path, base_sha = _base_path_for(direction, seed, manifest)
                obj = joblib.load(base_path)
                truth_map, _ = _truth_map(obj)
                gates: dict[int, dict[str, Any]] = {}
                artifact = _artifact_path(direction, seed)
                if artifact.exists():
                    prior = joblib.load(artifact)
                    scopes_complete = all(
                        any(row.get("status") == "COMPLETE" for row in _attempt_rows_for_scope(
                            f"{direction}__seed{int(seed)}__pos{pos}__MA_TAIL_GATE"))
                        for pos in range(4)
                    )
                    if (prior.get("base_object_sha256") == base_sha
                            and prior.get("candidate") == "MA-TAIL-BALANCED-ERROR-LR"
                            and prior.get("test_labels_read") is False
                            and set(map(str, prior.get("gates", {}).keys())) == {"0", "1", "2", "3"}
                            and scopes_complete):
                        skipped.append({"direction": direction, "seed": int(seed), "reason": "completed_artifact_reused"})
                        continue
                    raise RuntimeError(f"candidate artifact/ledger identity drift or incomplete cache: {artifact}")
                for pos in range(4):
                    ctx = _candidate_gate_context(obj, data, pos, truth_map)
                    scope = f"{direction}__seed{int(seed)}__pos{pos}__MA_TAIL_GATE"
                    key, kind = _next_attempt(scope)
                    r.register_attempt(key, kind, scope=scope, metadata={"direction": direction, "seed": int(seed), "position": pos,
                                                                           "requested_kind": "MA-TAIL-BALANCED-ERROR-LR", "fit_stage": "pilot" if pilot else "complete"})
                    try:
                        model = fit_weighted_gate(ctx["features"], ctx["pred"], ctx["truth"], scope, seed, key)
                        candidate_within = _score_model(model, ctx["outer_features"], ctx["outer_pred"])
                        candidate_score = np.where(ctx["outer_pred"] == 0, candidate_within, ctx["outer_score"])
                        gates[pos] = {"model": model, "adapter": ctx["adapter"], "train_A": ctx["A"],
                                      "outer_B": ctx["B"], "outer_score": ctx["outer_score"],
                                      "candidate_score": candidate_score, "candidate_within_score": candidate_within,
                                      "outer_pred": ctx["outer_pred"], "outer_truth": ctx["outer_truth"],
                                      "outer_index": ctx["outer_index"], "outer_record_id": ctx["outer_record_id"],
                                      "base_score_nonwithin": ctx["outer_score"], "test_labels_read": False}
                        r.finish_attempt(key, "COMPLETE", output="in_memory_candidate_gate", fallback=bool(model.get("fallback")))
                    except Exception as exc:
                        try:
                            r.finish_attempt(key, "FAILED", error=f"{type(exc).__name__}: {exc}")
                        except Exception:
                            pass
                        failures.append({"direction": direction, "seed": int(seed), "position": pos,
                                         "key": key, "kind": kind, "error": f"{type(exc).__name__}: {exc}"})
                        break
                if len(gates) == 4:
                    path = _save_candidate_artifact(direction, seed, obj, base_sha, gates, receipt)
                    completed.append({"direction": direction, "seed": int(seed), "artifact": str(path)})
                elif failures and failures[-1]["direction"] == direction and failures[-1]["seed"] == int(seed):
                    continue
        totals = r.ledger_totals()
        report = {"schema": "ma_tail_weight_gate_fit_run_v1", "status": "COMPLETE" if not failures else "FAILED_UNITS_RETAINED",
                  "pilot": bool(pilot), "requested_directions": directions, "requested_seeds": seeds,
                  "completed_objects": completed, "skipped_objects": skipped, "failures": failures,
                  "fit_units": totals, "test_labels_read": False, "test_evaluation_started": False,
                  "elapsed_seconds": float(time.time() - started), "runner_sha256": r.sha256(Path(__file__)),
                  "receipt_sha256": r.sha256(r.ROOT / "IMPLEMENTATION_PREFLIGHT_PASS.json")}
        r.write_json("FIT_RUN_REPORT.json", report)
        r.update_ledger_summary(status="FIT_COMPLETE_WITH_FAILURES_RETAINED" if failures else "FIT_COMPLETE", current_phase="FIT_COMPLETE",
                                fit_run_report_sha256=r.sha256(r.ROOT / "FIT_RUN_REPORT.json"))
        return report
    finally:
        r.release_lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="run read-only 180-object implementation preflight")
    ap.add_argument("--pilot72", action="store_true", help="run the authorized 72-position pilot after independent PASS")
    ap.add_argument("--complete720", action="store_true", help="run all 720 science positions after independent PASS")
    ap.add_argument("--selector", action="store_true", help="replay the development-only outer-B selector after all fits")
    ap.add_argument("--approved-fit", action="store_true")
    args = ap.parse_args()
    if sum(bool(v) for v in (args.preflight, args.pilot72, args.complete720, args.selector)) > 1:
        raise RuntimeError("choose exactly one execution mode")
    if args.selector:
        from selector_weighted_gate import run_selector
        result = run_selector()
        print(json.dumps({"status": "DEVELOPMENT_SELECTOR_COMPLETE_NO_TEST",
                          "promotion": result["promotion"], "test_labels_read": False}, ensure_ascii=False))
        return
    if args.pilot72 or args.complete720:
        if not args.approved_fit:
            raise RuntimeError("fit requires explicit --approved-fit in addition to independent receipt")
        if args.pilot72:
            report = run_fits(seeds=[1000, 1001], directions=list(r.DIRECTIONS), pilot=True)
        else:
            report = run_fits(seeds=list(r.SEEDS), directions=list(r.DIRECTIONS), pilot=False)
        print(json.dumps({"status": report["status"], "fit_units": report["fit_units"], "failures": len(report["failures"])}, ensure_ascii=False))
        return
    if not args.preflight:
        print(json.dumps({"status": "READY_NO_FIT", "run": str(r.ROOT), "next": "python run_weighted_gate.py --preflight"}, ensure_ascii=False))
        return
    report = run_preflight()
    print(json.dumps({"status": report["status"], "objects": report["objects"], "positions": report["positions"],
                      "science_units": 0, "repair_units": 0, "test_labels_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

