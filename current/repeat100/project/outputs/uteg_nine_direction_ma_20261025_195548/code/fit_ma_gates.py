"""Fit the six new MA-ERROR-LR gate bundles for the frozen nine directions.

The three dual-source directions point at the already frozen endpoint-pair
gates and are recorded as zero-fit reuse.  The six single-source directions
fit only on target pool outer-A rows and select thresholds only on outer-B
rows.  Position three uses one merged position-0/1/2 outer-B threshold in
frozen pool order; its full-cross threshold is retained as a diagnostic.
No target-test label is indexed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import nine_runtime as r

# nine_runtime establishes all task-local caches before numerical imports.
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402


ROOT = r.RUN_ROOT
OUT = ROOT.parent
BASE = OUT / "uteg_legacy_aligned_threecenter_20261022_155614"
EXP = OUT / "uteg_hill_exp_matched_20261012_153337"
OLD_GATE = OUT / "uteg_ma_endpoint_pair_20261024_143347"
DATA_PATH = BASE / "DATA15.npz"
ROLE_PATH = BASE / "ROLE_FREEZE.json"
SEEDS = tuple(range(1000, 1020))
CENTERS = ("Huzhou", "SAHZU", "Hubei")
MA_CLASSES = (-1, 0, 1)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_ids(ids: Any) -> str:
    return hashlib.sha256("\n".join(map(str, np.asarray(ids).astype(str).tolist())).encode()).hexdigest()


def ma_class(value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    return np.where(~np.isfinite(x), 9, np.where(x < 50, -1, np.where(x <= 70, 0, 1))).astype(np.int8)


def ma_features(x: Any, med: Any, ym: Any, ys: Any) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.asarray(med, dtype=np.float64)
    pred_ma = med[:, 2]
    pc = ma_class(pred_ma)
    return np.column_stack((
        x,
        (med - np.asarray(ym)[None, :]) / np.asarray(ys)[None, :],
        (pred_ma - 60.0) / 10.0,
        np.minimum(np.abs(pred_ma - 50.0), np.abs(pred_ma - 70.0)) / 10.0,
        (pc == -1).astype(float), (pc == 0).astype(float), (pc == 1).astype(float),
    ))


def margin_scores(pred_ma: Any) -> np.ndarray:
    x = np.asarray(pred_ma, dtype=np.float64)
    out = np.full(x.shape, np.inf, dtype=np.float64)
    ok = np.isfinite(x)
    out[ok] = -np.minimum(np.abs(x[ok] - 50.0), np.abs(x[ok] - 70.0)) / 10.0
    return out


def safe_median(f: np.ndarray) -> np.ndarray:
    out = np.zeros(f.shape[1], dtype=float)
    for j in range(f.shape[1]):
        keep = np.isfinite(f[:, j])
        if keep.any():
            out[j] = float(np.median(f[keep, j]))
    return out


def preprocess_fit(f: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    f = np.asarray(f, dtype=float)
    median = safe_median(f)
    finite = np.isfinite(f)
    mean = np.asarray([np.mean(f[finite[:, j], j]) if finite[:, j].any() else 0.0 for j in range(f.shape[1])])
    sd = np.asarray([np.std(f[finite[:, j], j]) if finite[:, j].any() else 1.0 for j in range(f.shape[1])])
    sd = np.maximum(sd, 1e-6)
    imp = np.where(finite, f, median[None, :])
    raw = (imp - mean[None, :]) / sd[None, :]
    return np.clip(raw, -10.0, 10.0), {
        "mean": mean, "sd": sd, "median": median, "ddof": 0,
        "floor": 1e-6, "clip": [-10.0, 10.0], "n": int(len(f)),
        "p": int(f.shape[1]), "missing_fraction": float(np.mean(~finite)) if f.size else 0.0,
        "clip_fraction": float(np.mean(np.abs(raw) > 10.0)) if f.size else 0.0,
        "scale_source": "finite_column_values",
    }


def preprocess_apply(f: np.ndarray, pre: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f, dtype=float)
    finite = np.isfinite(f)
    imp = np.where(finite, f, np.asarray(pre["median"])[None, :])
    z = np.clip((imp - np.asarray(pre["mean"])[None, :]) / np.asarray(pre["sd"])[None, :], -10.0, 10.0)
    return z, finite.all(axis=1)


def bce(theta: np.ndarray, z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    p = z.shape[1]
    eta = z @ theta[:p] + theta[p]
    prob = expit(eta)
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.1 * np.sum(theta[:p] ** 2))
    residual = prob - y
    grad = np.r_[z.T @ residual / len(y) + 0.2 * theta[:p], np.mean(residual)]
    return loss, grad


def fit_ma_error_lr(f: np.ndarray, pred_class: np.ndarray, true_class: np.ndarray, fit_id: str, seed: int) -> dict[str, Any]:
    known = np.isin(true_class, MA_CLASSES) & np.isin(pred_class, MA_CLASSES)
    legal_f = np.asarray(f[known], dtype=float)
    y = (pred_class[known] != true_class[known]).astype(float)
    if len(legal_f):
        z, pre = preprocess_fit(legal_f)
    else:
        z, pre = np.empty((0, f.shape[1])), {"mean": np.zeros(f.shape[1]), "sd": np.ones(f.shape[1]),
            "median": np.zeros(f.shape[1]), "ddof": 0, "floor": 1e-6, "clip": [-10.0, 10.0],
            "n": 0, "p": int(f.shape[1]), "missing_fraction": 0.0, "clip_fraction": 0.0}
    n = int(len(y)); errors = int(y.sum()); correct = n - errors
    constant = float((errors + 0.5) / (n + 1.0))
    out: dict[str, Any] = {
        "api_version": "ma-error-lr-v1", "id": "MA-ERROR-LR", "family": "MA-ERROR-LR",
        "fit_id": fit_id, "fit_seed": int(seed), "feature_dim": int(f.shape[1]), "regularizer": 0.1,
        "n": n, "errors": errors, "correct": correct, "error_rate": float(errors / n) if n else None,
        "legal_count": n, "finite_input_rows": int(np.isfinite(f).all(axis=1).sum()), "pre": pre,
        "constant_score": constant, "fallback": False, "fallback_reason": None, "success": False,
        "finite": True, "iterations": 0, "nfev": 0, "optimizer_steps": 0, "status": "not_started",
        "objective": None, "coef": np.zeros(f.shape[1]),
        "intercept": float(math.log(constant / (1.0 - constant))) if 0 < constant < 1 else 0.0,
        "train_pred_class_counts": {str(k): int(np.sum(pred_class[known] == k)) for k in MA_CLASSES},
        "train_true_class_counts": {str(k): int(np.sum(true_class[known] == k)) for k in MA_CLASSES},
    }
    if n < 8 or errors < 2 or correct < 2:
        out.update(fallback=True, fallback_reason="small_or_one_sided", success=True, status="fallback_constant")
        return out
    try:
        sol = minimize(lambda th: bce(th, z, y), np.zeros(f.shape[1] + 1), jac=True,
                       method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
        theta = np.asarray(sol.x, dtype=float)
        finite = bool(np.isfinite(theta).all() and np.isfinite(sol.fun))
        if not finite:
            out.update(fallback=True, fallback_reason="nonfinite_optimization", success=True, finite=False,
                       status="fallback_nonfinite", optimizer_steps=int(getattr(sol, "nit", 0)),
                       iterations=int(getattr(sol, "nit", 0)), nfev=int(getattr(sol, "nfev", 0)))
            return out
        out.update(success=bool(sol.success), finite=True, status=str(sol.message),
                   iterations=int(getattr(sol, "nit", 0)), nfev=int(getattr(sol, "nfev", 0)),
                   optimizer_steps=int(getattr(sol, "nit", 0)), objective=float(sol.fun),
                   coef=theta[:-1], intercept=float(theta[-1]))
    except Exception as exc:
        out.update(fallback=True, fallback_reason="optimization_exception:" + type(exc).__name__,
                   success=True, finite=False, status="fallback_exception")
    return out


def score_model(model: Mapping[str, Any], f: np.ndarray, pred_valid: np.ndarray) -> np.ndarray:
    z, _ = preprocess_apply(f, model["pre"])
    if bool(model.get("fallback", False)):
        score = np.full(len(f), float(model["constant_score"]))
    else:
        score = expit(z @ np.asarray(model["coef"]) + float(model["intercept"]))
    score = np.asarray(score, dtype=float)
    score[~np.asarray(pred_valid, dtype=bool)] = np.inf
    return score


def summary(score: np.ndarray, pred: np.ndarray, truth: np.ndarray, threshold: float, reason: str) -> dict[str, Any]:
    known = np.isin(truth, MA_CLASSES)
    valid = known & np.isin(pred, MA_CLASSES) & np.isfinite(score)
    release = valid & (score <= threshold) if np.isfinite(threshold) else np.zeros(len(score), dtype=bool)
    wrong = release & (pred != truth)
    abnormal = known & (truth != 0)
    afn = int(np.sum(release & abnormal & (pred == 0)))
    return {"threshold": float(threshold) if np.isfinite(threshold) else None,
            "rule": "score_leq_threshold_complete_ties" if np.isfinite(threshold) else "all_withhold",
            "reason": reason, "known_reference": int(known.sum()), "valid_ranking": int(valid.sum()),
            "released": int(release.sum()), "errors": int(wrong.sum()),
            "risk": float(wrong.sum() / release.sum()) if release.any() else None,
            "coverage": float(release.sum() / known.sum()) if known.any() else 0.0,
            "abnormal_support": int(abnormal.sum()), "abnormal_false_normal": afn,
            "abnormal_false_normal_rate": float(afn / abnormal.sum()) if abnormal.any() else None}


def select_threshold(score: np.ndarray, pred: np.ndarray, truth: np.ndarray, max_risk: float) -> dict[str, Any]:
    known = np.isin(truth, MA_CLASSES) & np.isin(pred, MA_CLASSES) & np.isfinite(score)
    candidates = [summary(score, pred, truth, np.nan, "all_withhold_candidate")]
    for threshold in np.unique(score[known]) if known.any() else np.asarray([]):
        candidates.append(summary(score, pred, truth, float(threshold), "threshold_candidate"))
    eligible = [row for row in candidates[1:] if row["released"] >= 5 and row["coverage"] >= 0.20
                and row["risk"] is not None and row["risk"] <= max_risk + 1e-12]
    if eligible:
        chosen = min(eligible, key=lambda row: (-float(row["coverage"]), float(row["risk"]),
                                                 float(row["abnormal_false_normal_rate"])
                                                 if row["abnormal_false_normal_rate"] is not None else np.inf,
                                                 float(row["threshold"])))
        chosen = dict(chosen); chosen["eligible"] = True
    else:
        chosen = dict(candidates[0]); chosen["eligible"] = False; chosen["reason"] = "no_eligible_threshold"
    chosen["qualification"] = {"min_released": 5, "min_coverage": 0.20,
                                "max_risk": max_risk, "candidate_count": len(candidates) - 1,
                                "eligible_count": len(eligible)}
    chosen["candidates"] = candidates
    return chosen


def get_adapter(center: str, obj: Mapping[str, Any], code: int) -> str:
    if center == "Huzhou":
        return "H"
    adapter = obj["gate_models"][code].get("adapter")
    return "H" if adapter is None else str(adapter)


def target_spec(direction: str, target: str, source: str, source_kind: str, mode: str) -> dict[str, Any]:
    if mode == "dual_reuse":
        return {"direction": direction, "target": target, "source_centers": source,
                "source_kind": source_kind, "mode": mode}
    # ``single_new`` targets are always the newly fitted DATA15 objects even
    # when their source model is legitimately reused from an EXP D1/D2 source
    # bundle.  Only the two ``single_exp`` directions use EXP target rows.
    if mode == "single_new":
        object_pattern = ROOT / "models" / direction / "target15" / f"{target}_CFULL_seed{{seed}}.joblib"
        data_path = DATA_PATH
    elif source_kind == "exp_D1":
        object_pattern = EXP / "runs" / "D1_seed{seed}.joblib"
        data_path = EXP / "EXP_DATA.npz"
    elif source_kind == "exp_D2":
        object_pattern = EXP / "runs" / "D2_seed{seed}.joblib"
        data_path = EXP / "EXP_DATA.npz"
    else:
        object_pattern = ROOT / "models" / direction / "target15" / f"{target}_CFULL_seed{{seed}}.joblib"
        data_path = DATA_PATH
    return {"direction": direction, "target": target, "source_centers": source,
            "source_kind": source_kind, "mode": mode, "object_pattern": object_pattern,
            "data_path": data_path}


SPECS = [
    target_spec("HZ+HB_to_SA", "SAHZU", ("Huzhou", "Hubei"), "legacy_dual", "dual_reuse"),
    target_spec("SA+HB_to_HZ", "Huzhou", ("SAHZU", "Hubei"), "legacy_dual", "dual_reuse"),
    target_spec("SA+HZ_to_HB", "Hubei", ("SAHZU", "Huzhou"), "legacy_dual", "dual_reuse"),
    target_spec("HZ_to_SA", "SAHZU", ("Huzhou",), "exp_D2", "single_exp"),
    target_spec("HB_to_SA", "SAHZU", ("Hubei",), "new_hubei", "single_new"),
    target_spec("SA_to_HZ", "Huzhou", ("SAHZU",), "exp_D1", "single_exp"),
    target_spec("HB_to_HZ", "Huzhou", ("Hubei",), "new_hubei", "single_new"),
    target_spec("SA_to_HB", "Hubei", ("SAHZU",), "exp_D1", "single_new"),
    target_spec("HZ_to_HB", "Hubei", ("Huzhou",), "exp_D2", "single_new"),
]


def load_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def validate_object(spec: Mapping[str, Any], obj: Mapping[str, Any], data: Mapping[str, Any], role: Mapping[str, Any]) -> dict[str, Any]:
    target = str(spec["target"])
    seed = int(obj.get("seed", 0))
    if obj.get("target") is not None:
        assert str(obj["target"]) == target and seed in SEEDS
    if obj.get("test_labels_hidden", True) is not True:
        raise RuntimeError(f"target labels are not hidden: {spec['direction']}")
    pool = np.asarray(obj["pool"], dtype=np.int64)
    test = np.asarray(obj["test"], dtype=np.int64)
    role_target = role["roles"][target]
    base_data = load_data(DATA_PATH)
    actual_ids = np.asarray(data["ids"]).astype(str)
    expected_pool_ids = set(np.asarray(base_data["ids"])[np.asarray(role_target["pool_full"], dtype=int)].astype(str))
    expected_test_ids = set(np.asarray(base_data["ids"])[np.asarray(role_target["test"], dtype=int)].astype(str))
    if set(actual_ids[pool]) != expected_pool_ids or set(actual_ids[test]) != expected_test_ids:
        raise RuntimeError(f"target pool/test identity mismatch: {spec['direction']} seed={seed}")
    expected_test_n = {"Huzhou": 55, "SAHZU": 201, "Hubei": 41}[target]
    expected_pool_n = {"Huzhou": 40, "SAHZU": 60, "Hubei": 40}[target]
    assert len(pool) == expected_pool_n and len(test) == expected_test_n and set(pool).isdisjoint(set(test))
    ym = np.asarray(obj["source_model"]["ym"], dtype=float)
    ys = np.asarray(obj["source_model"]["ys"], dtype=float)
    assert ym.shape == (4,) and ys.shape == (4,) and np.isfinite(ym).all() and np.all(ys > 0)
    pool_loc = {int(g): i for i, g in enumerate(pool.tolist())}
    positions: dict[str, Any] = {}
    for code in range(4):
        p = obj["positions"][code]
        A = np.asarray(p["A"], dtype=np.int64)
        B = np.asarray(p["B"], dtype=np.int64)
        assert set(A).isdisjoint(set(B)) and set(A).issubset(set(pool)) and set(B).issubset(set(pool))
        if code < 3:
            assert sorted(np.r_[A, B].tolist()) == sorted(pool.tolist())
        else:
            assert len(B) == 0 and sorted(A.tolist()) == sorted(pool.tolist())
        assert len(p["inner"]) == 3
        inner_union: list[int] = []
        for fold in p["inner"]:
            T = np.asarray(fold["T"], dtype=np.int64)
            V = np.asarray(fold["V"], dtype=np.int64)
            assert set(T).isdisjoint(set(V)) and set(np.r_[T, V]) == set(A)
            assert fold["token"] == ",".join(str(int(x)) for x in sorted(T.tolist()))
            assert sorted(map(int, obj["cache"][fold["token"]]["train"])) == sorted(T.tolist())
            inner_union.extend(V.tolist())
        assert sorted(inner_union) == sorted(A.tolist())
        adapter = get_adapter(target, obj, code)
        cross = np.asarray(p["cross"][adapter], dtype=float)
        assert cross.shape == (len(A), 4)
        assert np.asarray(obj["OOF"][adapter], dtype=float).shape == (len(pool), 4)
        positions[str(code)] = {"A": A, "B": B, "adapter": adapter, "cross": cross}
    return {"pool": pool, "test": test, "ym": ym, "ys": ys, "positions": positions,
            "pool_ids": actual_ids[pool], "test_ids": actual_ids[test]}


def append_oof(rows: list[dict[str, Any]], center: str, direction: str, seed: int, code: int,
               role_name: str, indices: np.ndarray, adapter: str, pred_ma: np.ndarray,
               pred_class: np.ndarray, truth_class: np.ndarray, score: np.ndarray) -> None:
    for j, idx in enumerate(indices.tolist()):
        rows.append({"direction": direction, "target": center, "seed": int(seed), "position": int(code),
                     "outer_role": role_name, "global_index": int(idx), "adapter": adapter,
                     "pred_ma": float(pred_ma[j]) if np.isfinite(pred_ma[j]) else None,
                     "pred_class": int(pred_class[j]), "true_class": int(truth_class[j]),
                     "score": float(score[j]) if np.isfinite(score[j]) else None})


def fit_single(spec: Mapping[str, Any], role: Mapping[str, Any]) -> dict[str, Any]:
    direction = str(spec["direction"])
    target = str(spec["target"])
    source_data = load_data(Path(spec["data_path"]))
    outdir = ROOT / "models" / direction / "ma_gates"
    outdir.mkdir(parents=True, exist_ok=True)
    r.set_scope(f"MA_GATE_{direction}")
    model_hashes: dict[str, str] = {}
    thresholds: dict[str, Any] = {}
    object_hashes: dict[str, str] = {}
    oof_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        object_path = Path(str(spec["object_pattern"]).format(seed=seed))
        if not object_path.exists():
            raise FileNotFoundError(object_path)
        object_hashes[str(object_path)] = sha(object_path)
        obj = joblib.load(object_path)
        meta = validate_object(spec, obj, source_data, role)
        pool = meta["pool"]
        pool_loc = {int(g): i for i, g in enumerate(pool.tolist())}
        outer_b: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        per_seed_models: list[dict[str, Any]] = []
        for code in range(4):
            p = meta["positions"][str(code)]
            A, B, adapter, cross = p["A"], p["B"], p["adapter"], p["cross"]
            pred_ma_A = cross[:, 2]
            pred_class_A = ma_class(pred_ma_A)
            truth_class_A = ma_class(source_data["medical"][A, 2])
            f_A = ma_features(source_data["x"][A], cross, meta["ym"], meta["ys"])
            fit_id = f"{direction}__{target}_seed{seed}_pos{code}"
            model_path = outdir / f"{target}_seed{seed}_pos{code}_MA_ERROR_LR.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                if model.get("fit_id") != fit_id:
                    raise RuntimeError(f"existing gate fit identity mismatch: {model_path}")
            else:
                r.ledger(f"gate_{target}_seed{seed}_pos{code}", "gate", 1,
                         direction=direction, target=target, seed=seed, position=code,
                         source_kind=spec["source_kind"], target_test_evaluated=False)
                model = fit_ma_error_lr(f_A, pred_class_A, truth_class_A, fit_id, seed)
            score_A = score_model(model, f_A, np.isfinite(pred_ma_A))
            if len(B):
                b_local = np.asarray([pool_loc[int(g)] for g in B], dtype=int)
                pred_B = np.asarray(obj["OOF"][adapter], dtype=float)[b_local]
                pred_ma_B = pred_B[:, 2]
                pred_class_B = ma_class(pred_ma_B)
                truth_class_B = ma_class(source_data["medical"][B, 2])
                f_B = ma_features(source_data["x"][B], pred_B, meta["ym"], meta["ys"])
                score_B = score_model(model, f_B, np.isfinite(pred_ma_B))
                b_order = np.asarray([pool_loc[int(g)] for g in B], dtype=int)
                outer_b.append((b_order, pred_class_B.copy(), truth_class_B.copy(), score_B.copy(),
                                margin_scores(pred_ma_B)))
            else:
                pred_ma_B = np.empty(0); pred_class_B = np.empty(0, dtype=np.int8)
                truth_class_B = np.empty(0, dtype=np.int8); score_B = np.empty(0)
            model_info = {"direction": direction, "target": target, "seed": int(seed), "position": int(code),
                          "adapter": adapter, "outer_A_n": int(len(A)), "outer_B_n": int(len(B)),
                          "source_kind": spec["source_kind"], "target_test_evaluated": False,
                          "target_pool_ids_sha256": hash_ids(meta["pool_ids"]),
                          "target_test_ids_sha256": hash_ids(meta["test_ids"]),
                          "threshold_source": "outer_B"}
            model.update(model_info)
            per_seed_models.append({"model": model, "model_path": model_path, "A": A, "B": B,
                                    "pred_ma_A": pred_ma_A, "pred_class_A": pred_class_A,
                                    "truth_class_A": truth_class_A, "f_A": f_A, "score_A": score_A,
                                    "pred_ma_B": pred_ma_B, "pred_class_B": pred_class_B,
                                    "truth_class_B": truth_class_B, "score_B": score_B})
        if len(outer_b) != 3:
            raise RuntimeError(f"expected three outer-B rows: {direction} seed={seed}")
        order = np.argsort(np.concatenate([row[0] for row in outer_b]), kind="mergesort")
        threshold_pred = np.concatenate([row[1] for row in outer_b])[order]
        threshold_true = np.concatenate([row[2] for row in outer_b])[order]
        threshold_score = np.concatenate([row[3] for row in outer_b])[order]
        margin_threshold_score = np.concatenate([row[4] for row in outer_b])[order]
        max_risk = 0.10 if target == "Huzhou" else 0.15
        for item in per_seed_models:
            code = int(item["model"]["position"])
            if code < 3:
                tp, tt, ts = item["pred_class_B"], item["truth_class_B"], item["score_B"]
                ms = margin_scores(item["pred_ma_B"])
                source_name = "outer_B"
            else:
                tp, tt, ts, ms = threshold_pred, threshold_true, threshold_score, margin_threshold_score
                source_name = "outer_B_merged_positions_0_1_2"
            model = item["model"]
            threshold = select_threshold(ts, tp, tt, max_risk)
            margin_threshold = select_threshold(ms, tp, tt, max_risk)
            model.update({"threshold": threshold, "threshold_source": source_name,
                          "margin_threshold": margin_threshold, "margin_threshold_source": source_name,
                          "full_cross_diagnostic_threshold": select_threshold(item["score_A"], item["pred_class_A"], item["truth_class_A"], max_risk)
                          if code == 3 else None,
                          "threshold_support": {"n": int(np.isin(tt, MA_CLASSES).sum()),
                                                 "valid": int(np.isin(tt, MA_CLASSES).sum()), "source": source_name},
                          "margin_threshold_support": {"n": int(np.isin(tt, MA_CLASSES).sum()),
                                                        "source": source_name}})
            joblib.dump(model, item["model_path"], compress=3)
            rel = str(item["model_path"].relative_to(ROOT))
            model_hashes[rel] = sha(item["model_path"])
            fit_id = str(model["fit_id"])
            thresholds[fit_id] = {"threshold": threshold, "margin_threshold": margin_threshold,
                                  "threshold_source": source_name, "center": target, "seed": int(seed),
                                  "position": code, "adapter": model["adapter"],
                                  "fallback": bool(model["fallback"]), "success": bool(model["success"]),
                                  "finite": bool(model["finite"])}
            append_oof(oof_rows, target, direction, seed, code, "A", item["A"], item["model"]["adapter"],
                       item["pred_ma_A"], item["pred_class_A"], item["truth_class_A"], item["score_A"])
            if len(item["B"]):
                append_oof(oof_rows, target, direction, seed, code, "B", item["B"], item["model"]["adapter"],
                           item["pred_ma_B"], item["pred_class_B"], item["truth_class_B"], item["score_B"])
        if seed % 5 == 0:
            r.guard()
    oof_path = ROOT / "oof" / f"{direction}_OOF.npz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_path, **{key: np.asarray([row[key] for row in oof_rows]) for key in (
        "direction", "target", "seed", "position", "outer_role", "global_index", "adapter",
        "pred_ma", "pred_class", "true_class", "score")})
    manifest = {"version": "nine_direction_ma_gate_manifest_v1", "direction": direction,
                "target": target, "source_centers": list(spec["source_centers"]),
                "source_kind": spec["source_kind"], "new_fit_positions": len(model_hashes),
                "models": model_hashes, "thresholds": thresholds, "object_hashes": object_hashes,
                "oof_path": str(oof_path), "oof_sha256": sha(oof_path),
                "target_test_evaluated": False, "target_test_labels_read": False,
                "threshold_rule": "outer-B; position-3 merged positions-0/1/2 in frozen pool order",
                "source_contract": "only listed source centers; omitted third center excluded from fit/scaler/selection"}
    write_json(ROOT / "models" / direction / "MA_GATE_MANIFEST.json", manifest)
    return manifest


def reuse_dual(spec: Mapping[str, Any], role: Mapping[str, Any]) -> dict[str, Any]:
    freeze = read_json(OLD_GATE / "ALL_GATES_FROZEN.json")
    target = str(spec["target"]); direction = str(spec["direction"])
    base_data = load_data(DATA_PATH)
    role_target = role["roles"][target]
    models: dict[str, str] = {}; thresholds: dict[str, Any] = {}
    for seed in SEEDS:
        for code in range(4):
            rel_path = Path("models") / f"{target}_seed{seed}_pos{code}_MA_ERROR_LR.joblib"
            old_path = OLD_GATE / rel_path
            freeze_key = str(rel_path)
            if not old_path.exists() or sha(old_path) != freeze["models"].get(freeze_key):
                raise RuntimeError(f"dual gate reuse hash mismatch: {old_path}")
            models[str(old_path)] = sha(old_path)
            key = f"{target}_seed{seed}_pos{code}"
            thresholds[key] = freeze["thresholds"][key]
    target_pool_ids = np.asarray(base_data["ids"])[np.asarray(role_target["pool_full"], dtype=int)]
    target_test_ids = np.asarray(base_data["ids"])[np.asarray(role_target["test"], dtype=int)]
    manifest = {"version": "nine_direction_ma_gate_manifest_v1", "direction": direction,
                "target": target, "source_centers": list(spec["source_centers"]),
                "source_kind": spec["source_kind"], "new_fit_positions": 0, "reused_positions": 80,
                "models": models, "thresholds": thresholds, "target_pool_ids_sha256": hash_ids(target_pool_ids),
                "target_test_ids_sha256": hash_ids(target_test_ids), "target_test_evaluated": False,
                "target_test_labels_read": False, "reused_from": str(OLD_GATE),
                "reuse_freeze_sha256": sha(OLD_GATE / "ALL_GATES_FROZEN.json"),
                "threshold_rule": "frozen original dual endpoint-pair gate"}
    write_json(ROOT / "models" / direction / "MA_GATE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    role = read_json(ROLE_PATH)
    manifests: dict[str, Any] = {}
    for spec in SPECS:
        if spec["mode"] == "dual_reuse":
            manifests[spec["direction"]] = reuse_dual(spec, role)
            print("GATE_REUSE", spec["direction"], flush=True)
        else:
            manifests[spec["direction"]] = fit_single(spec, role)
            print("GATE_COMPLETE", spec["direction"], flush=True)
    all_models: dict[str, str] = {}
    new_count = 0; reused_count = 0
    for direction, manifest in manifests.items():
        all_models.update({f"{direction}:{path}": digest for path, digest in manifest["models"].items()})
        new_count += int(manifest.get("new_fit_positions", 0)); reused_count += int(manifest.get("reused_positions", 0))
    freeze = {"version": "nine_direction_ma_all_gates_frozen_v1", "created": time.time(),
              "before_target_ma_evaluation": True, "target_evaluation": False,
              "target_test_labels_read": False, "direction_count": len(SPECS),
              "model_count": len(all_models), "new_fit_positions": new_count,
              "reused_positions": reused_count, "models": all_models,
              "manifests": {d: sha(ROOT / "models" / d / "MA_GATE_MANIFEST.json") for d in manifests},
              "protocol_sha256": sha(ROOT / "PROTOCOL.md"), "execution_prompt_sha256": sha(ROOT / "EXECUTION_PROMPT_ZH.md"),
              "threshold_spec": "complete ties; >=5 releases; >=20% known-pool coverage; risk <=0.10 Huzhou or <=0.15 SAHZU/Hubei; position-3 outer-B merged",
              "fit_budget": {"new_gate_positions": new_count, "reused_gate_positions": reused_count, "total_gate_positions": len(all_models)}}
    write_json(ROOT / "ALL_GATES_FROZEN.json", freeze)
    write_json(ROOT / "MA_GATE_STATE.json", {"status": "ALL_GATES_FROZEN_BEFORE_TEST_EVALUATION",
                                               "target_evaluation": False, "new_fit_positions": new_count,
                                               "reused_positions": reused_count, "ledger": r.guard()})
    print(json.dumps({"status": "ALL_GATES_FROZEN", "new_fit_positions": new_count,
                      "reused_positions": reused_count, "ledger": r.guard(),
                      "target_evaluation": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
