"""Frozen target-only 5/10-minute development training.

Importing this module only exposes loaders and pure historical functions.
Fitting occurs only via ``python train_prefix.py train``.  The implementation
is a mechanical prefix adaptation of focused_reporting_20261030/run.py.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# Keep these globals explicit for historical helpers that expect ROOT/G/OUT.
# No imported historical runner is allowed to rebind them or write old outputs.
ROOT = Path(__file__).resolve().parent
G = ROOT.parent
OUT = G.parent
BASE = OUT / "uteg_legacy_aligned_threecenter_20261022_155614"
FOCUSED = G / "focused_reporting_20261030"
OLD = OUT / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551"
DATA = BASE / "DATA15.npz"
PREFIX_X = BASE / "PREFIX_X.npz"
PREFIX_PARAMETERS = BASE / "PREFIX_PARAMETERS.npz"
PREFIX_IDENTITY = BASE / "PREFIX_X_IDENTITY.json"
CONTRACT = OUT / "uteg_ridge_dual_head_prepare_20261026_172952" / "dual_head_contract.py"
SELECTOR = OUT / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "run_unified_selector.py"
RIDGE = G / "target_ridge" / "run.py"
OLD_RUN = OLD / "run_ridge10_fullchain.py"

DIRS = ["HZ_to_SA", "HZ+HB_to_SA"]
SEEDS = list(range(1000, 1020))
ALPHAS = [0.1, 1.0, 10.0, 100.0]
MA_CLASSES = (-1, 0, 1)
HORIZONS = (300, 600, 900)


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(name: str, value: Mapping[str, Any]) -> None:
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False, default=_json_value) + "\n",
                    encoding="utf-8")


def _write_table(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty development table {name}")
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _original_partition(direction: str, seed: int) -> Path:
    if direction not in DIRS:
        raise ValueError(f"unsupported frozen partition direction: {direction}")
    if int(seed) not in SEEDS:
        raise ValueError(f"unsupported frozen seed: {seed}")
    return OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{int(seed)}.joblib"


def load_source_partition(seed: int, direction: str = DIRS[0]) -> dict[str, Any]:
    """Load frozen pool/test indices and nested folds, without outcomes."""
    path = _original_partition(direction, seed)
    obj = joblib.load(path)
    pool = np.asarray(obj["pool"], dtype=np.int32)
    test = np.asarray(obj["test"], dtype=np.int32)
    if pool.shape != (60,) or test.shape != (201,):
        raise AssertionError("frozen target partition size drift")
    return {"path": str(path.resolve()), "direction": direction, "seed": int(seed),
            "pool": pool.copy(), "test": test.copy(), "positions": obj["positions"]}


def load_frozen_partition(seed: int, direction: str = DIRS[0]) -> dict[str, Any]:
    """Alias for downstream partition audits."""
    return load_source_partition(seed, direction)


def load_checkpoint(horizon: int) -> dict[str, np.ndarray | int]:
    """Return x/medical/ids/groups/input_qc in DATA15's 480-row order.

    Prefix QC is derived only from the observed prefix representation.  No
    future/full-sequence QC veto is applied.  Non-finite values are not
    imputed; the explicit ``finite_valid`` mask is returned and callers must
    fail closed.  The frozen prefix package is fully finite and is asserted.
    """
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    with np.load(DATA, allow_pickle=False) as base:
        medical = np.asarray(base["medical"], dtype=np.float64).copy()
        ids = np.asarray(base["ids"]).copy()
        groups = np.asarray(base["groups"]).copy()
        if horizon == 900:
            x = np.asarray(base["x"], dtype=np.float64).copy()
            input_qc = np.asarray(base["input_qc"], dtype=bool).copy()
        else:
            with np.load(PREFIX_X, allow_pickle=False) as prefix:
                available = [int(v) for v in np.asarray(prefix["horizon_seconds"]).tolist()]
                if horizon not in available:
                    raise AssertionError(f"missing frozen prefix horizon {horizon}")
                x = np.asarray(prefix["x"][available.index(horizon)], dtype=np.float64).copy()
            input_qc = np.isfinite(x).all(axis=1)
        if x.shape != (480, 4) or medical.shape != (480, 4):
            raise AssertionError("checkpoint shape drift")
        if ids.shape != (480,) or groups.shape != (480,):
            raise AssertionError("checkpoint identity shape drift")
        finite_valid = np.isfinite(x).all(axis=1)
        if horizon in (300, 600) and not bool(finite_valid.all()):
            raise AssertionError("frozen prefix contains non-finite rows")
        if not np.isfinite(medical).all():
            raise AssertionError("DATA15 medical reference contains non-finite values")
    return {"x": x, "medical": medical, "ids": ids, "groups": groups,
            "input_qc": input_qc, "finite_valid": finite_valid,
            "horizon_seconds": horizon}


# Import only the six pure historical functions.  Loading the full historical
# runner could rebind ROOT/G/OUT or write to its output tree.
_tree = ast.parse(OLD_RUN.read_text(encoding="utf-8"))
_pure_names = {"ma_class", "ma_features", "_preprocess_fit", "_empty_gate_pre",
               "_fit_ma_gate", "score_ma_gate"}
_pure_nodes = [node for node in _tree.body
               if isinstance(node, ast.FunctionDef) and node.name in _pure_names]
exec(compile(ast.Module(body=_pure_nodes, type_ignores=[]), str(OLD_RUN), "exec"), globals())

contract = _load_module("target_prefix_dual_contract", CONTRACT)
selector = _load_module("target_prefix_original_selector", SELECTOR)
ridge = _load_module("target_prefix_ridge", RIDGE)


def _head_scores(heads: Mapping[str, Any], x: np.ndarray, med: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for name in ("low", "high"):
        model = heads[name]
        if model is None:
            values.append(np.full(len(x), np.inf, dtype=float))
            continue
        pre = model["pre"]
        features = np.column_stack([x[:, 1], med[:, 2]])
        filled = np.where(np.isfinite(features), features, pre["median"])
        z = np.clip((filled - pre["mean"]) / pre["sd"], -10.0, 10.0)
        values.append(contract.score_head(model["theta"], z))
    return np.maximum(values[0], values[1])


def _action(threshold: Mapping[str, Any], pred: np.ndarray, base_score: np.ndarray,
            dual_score: np.ndarray, qc: np.ndarray) -> np.ndarray:
    released = np.zeros(len(pred), dtype=bool)
    if threshold["all_withhold"]:
        return released
    if threshold["normal"] is not None:
        released |= (pred == 0) & np.isfinite(dual_score) & (dual_score <= threshold["normal"])
    if threshold["tail"] is not None:
        released |= np.isin(pred, [-1, 1]) & np.isfinite(base_score) & (base_score <= threshold["tail"])
    return released & np.asarray(qc, dtype=bool)


def _choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [None] + sorted({float(row["dual_score"]) for row in rows
                               if row["pred_class"] == 0 and row["dual_score"] is not None
                               and np.isfinite(float(row["dual_score"]))})
    candidates = [{"threshold": threshold, "candidate_index": index, "n": len(rows),
                   "release_record_ids": [row["record_id"] for row in rows
                       if threshold is not None and row["pred_class"] == 0
                       and row["dual_score"] is not None
                       and np.isfinite(float(row["dual_score"]))
                       and float(row["dual_score"]) <= threshold]}
                  for index, threshold in enumerate(values)]
    normal = selector.choose_normal(candidates, rows, {row["record_id"]: row for row in rows})
    tail = selector.choose_common_tail(rows, normal)
    return {"normal": normal["selected"]["threshold"] if normal["selected"] else None,
            "tail": tail["selected"]["threshold"] if tail["selected"] else None,
            "all_withhold": tail["selected"] is None,
            "normal_status": normal["status"], "tail_status": tail["status"]}


def _synthetic_checks() -> dict[str, Any]:
    rng = np.random.default_rng(381)
    x = rng.normal(size=(30, 4)); med = rng.normal(size=(30, 4)) * 3.0 + 60.0
    a = ma_features(x, med, np.zeros(4), np.ones(4))
    b = ma_features(x, med, np.arange(4), np.arange(4) + 1.0)
    pa = _preprocess_fit(a); pb = _preprocess_fit(b)
    za = np.clip((a - pa["mean"]) / pa["sd"], -10.0, 10.0)
    zb = np.clip((b - pb["mean"]) / pb["sd"], -10.0, 10.0)
    assert np.allclose(za, zb, atol=1e-12)
    assert np.array_equal(ma_class([49.9, 50, 70, 70.1, np.nan]), [-1, 0, 0, 1, 9])
    assert not _action({"all_withhold": True, "normal": None, "tail": None},
                       np.array([0]), np.array([0.0]), np.array([0.0]), np.array([True])).any()
    assert np.isinf(_head_scores({"low": None, "high": None}, x, med)).all()
    assert contract.support_summary(np.array([0, 0, 0, 0, 0, 0, 0, 1]))["status"] == "UNSUPPORTED"
    return {"normalization_max_difference": float(np.max(np.abs(za - zb))),
            "synthetic_contract": True, "selector": selector._synthetic_tests()}


def _check_partition_identity(seed: int) -> dict[str, Any]:
    first = load_source_partition(seed, DIRS[0]); second = load_source_partition(seed, DIRS[1])
    if not np.array_equal(first["pool"], second["pool"]) or not np.array_equal(first["test"], second["test"]):
        raise AssertionError(f"partition mismatch for seed {seed}")
    # The prefix checkpoints deliberately reuse one frozen target partition;
    # materialize the equality check for every checkpoint so the freeze record
    # cannot merely assert this from a single horizon.
    checkpoint_partitions = {h: load_source_partition(seed, DIRS[0]) for h in HORIZONS}
    checkpoint_equal = {}
    for horizon in HORIZONS:
        current = checkpoint_partitions[horizon]
        if not np.array_equal(first["pool"], current["pool"]) or not np.array_equal(first["test"], current["test"]):
            raise AssertionError(f"cross-checkpoint pool/test mismatch for seed {seed}, horizon {horizon}")
        checkpoint_equal[str(horizon)] = {"pool": True, "test": True}
    position_checks = []
    for position in range(4):
        p = first["positions"][position]; q = second["positions"][position]
        if p["A"] != q["A"] or p["B"] != q["B"]:
            raise AssertionError(f"outer fold mismatch for seed {seed}, position {position}")
        if [(v["T"], v["V"]) for v in p["inner"]] != [(v["T"], v["V"]) for v in q["inner"]]:
            raise AssertionError(f"inner fold mismatch for seed {seed}, position {position}")
        for horizon, current in checkpoint_partitions.items():
            cp = current["positions"][position]
            if p["A"] != cp["A"] or p["B"] != cp["B"]:
                raise AssertionError(f"cross-checkpoint outer mismatch for seed {seed}, position {position}, horizon {horizon}")
            if [(v["T"], v["V"]) for v in p["inner"]] != [(v["T"], v["V"]) for v in cp["inner"]]:
                raise AssertionError(f"cross-checkpoint inner mismatch for seed {seed}, position {position}, horizon {horizon}")
        position_checks.append({"position": position, "A_B_equal": True, "inner_T_V_equal": True})
    return {"pool_n": 60, "test_n": 201, "positions": position_checks,
            "pool_test_equal_across_checkpoints": checkpoint_equal,
            "outer_and_inner_equal": True}


def _input_hashes() -> dict[str, str]:
    paths: list[Path] = [DATA, PREFIX_X, PREFIX_PARAMETERS, PREFIX_IDENTITY,
        ROOT / "PROTOCOL.md", FOCUSED / "PROTOCOL.md", FOCUSED / "run.py",
        OLD_RUN, CONTRACT, SELECTOR, RIDGE, Path(__file__)]
    for direction in DIRS:
        for seed in SEEDS:
            paths.append(_original_partition(direction, seed))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {str(path.resolve()): sha(path) for path in paths}


def _fit_horizon(horizon: int, log: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                   list[dict[str, Any]], dict[str, int], dict[str, str]]:
    data = load_checkpoint(horizon)
    x = np.asarray(data["x"], dtype=float); medical = np.asarray(data["medical"], dtype=float)
    ids = np.asarray(data["ids"]); groups = np.asarray(data["groups"])
    qc = np.asarray(data["input_qc"], dtype=bool)
    if x.shape != (480, 4) or medical.shape != (480, 4) or not np.isfinite(x).all() or not np.isfinite(medical).all():
        raise AssertionError("training checkpoint shape or finite-value drift")

    fold_rows: list[dict[str, Any]] = []; head_rows: list[dict[str, Any]] = []
    working_rows: list[dict[str, Any]] = []; counts = {"ridge": 0, "error": 0, "dual": 0}
    model_hashes: dict[str, str] = {}
    for seed in SEEDS:
        part = load_source_partition(seed, DIRS[0]); other = load_source_partition(seed, DIRS[1])
        if not np.array_equal(part["pool"], other["pool"]) or not np.array_equal(part["test"], other["test"]):
            raise AssertionError(f"frozen partition mismatch for seed {seed}")
        pool_global = np.asarray(part["pool"], dtype=np.int32); test_global = np.asarray(part["test"], dtype=np.int32)
        if len(pool_global) != 60 or len(test_global) != 201:
            raise AssertionError("frozen partition cardinality drift")
        if set(groups[pool_global]) & set(groups[test_global]):
            raise AssertionError(f"development/test group overlap for seed {seed}")
        lookup = {int(v): i for i, v in enumerate(pool_global.tolist())}
        final: dict[str, Any] | None = None; rows: list[dict[str, Any]] = []

        for position in range(4):
            p = part["positions"][position]
            A = np.asarray([lookup[int(v)] for v in p["A"]], dtype=np.int32)
            B = np.asarray([lookup[int(v)] for v in p["B"]], dtype=np.int32)
            if set(A.tolist()) & set(B.tolist()) or set(groups[pool_global[A]]) & set(groups[pool_global[B]]):
                raise AssertionError(f"outer fold/group overlap for seed {seed}, position {position}")
            loc = {int(v): i for i, v in enumerate(A.tolist())}
            cross = {alpha: np.full((len(A), 4), np.nan) for alpha in ALPHAS}; covered: list[int] = []
            for inner in p["inner"]:
                T = np.asarray([lookup[int(v)] for v in inner["T"]], dtype=np.int32)
                V = np.asarray([lookup[int(v)] for v in inner["V"]], dtype=np.int32)
                if set(T.tolist()) & set(V.tolist()) or set(groups[pool_global[T]]) & set(groups[pool_global[V]]):
                    raise AssertionError(f"inner fold/group overlap for seed {seed}, position {position}")
                if set(T.tolist()) | set(V.tolist()) != set(A.tolist()):
                    raise AssertionError(f"inner fold coverage drift for seed {seed}, position {position}")
                covered.extend(V.tolist())
                for alpha in ALPHAS:
                    fitted = ridge.fit(x[pool_global[T]], medical[pool_global[T]], alpha); counts["ridge"] += 1
                    cross[alpha][[loc[int(v)] for v in V.tolist()]] = ridge.predict(fitted, x[pool_global[V]])
            if sorted(covered) != sorted(A.tolist()):
                raise AssertionError(f"inner validation coverage drift for seed {seed}, position {position}")
            losses = {alpha: float(np.mean(np.abs(cross[alpha][:, 2] - medical[pool_global[A], 2]))) for alpha in ALPHAS}
            best = min(ALPHAS, key=lambda alpha: (losses[alpha], -alpha)); median_cross = cross[best]
            model = ridge.fit(x[pool_global[A]], medical[pool_global[A]], best); counts["ridge"] += 1
            pred_class = ma_class(median_cross[:, 2]); truth_class = ma_class(medical[pool_global[A], 2])
            ym = medical[pool_global[A]].mean(0); ys = np.maximum(medical[pool_global[A]].std(0), 1e-6)
            gate = _fit_ma_gate(ma_features(x[pool_global[A]], median_cross, ym, ys), pred_class, truth_class,
                                f"target_only_prefix_{horizon}_{seed}_{position}", seed)
            counts["error"] += 1
            if not gate["success"] and not gate["fallback"]:
                raise RuntimeError("error gate optimizer did not converge")
            heads: dict[str, Any] = {}
            for head in ("low", "high"):
                mask, labels = contract.build_head_mask(truth_class, pred_class, median_cross[:, 2], qc[pool_global[A]], head)
                support = contract.support_summary(labels, ids[pool_global[A]][mask])
                head_rows.append({"horizon": horizon, "seed": seed, "position": position, "head": head,
                                  "n": support["n"], "class0": support["class0_n"], "class1": support["class1_n"],
                                  "status": support["status"]})
                if support["status"] != "SUPPORTED":
                    heads[head] = None; continue
                z, pre = contract.build_features(x[pool_global[A], 1][mask], median_cross[:, 2][mask],
                                                  np.ones(int(mask.sum()), dtype=bool))
                weights = contract.class_balanced_weights(labels)
                optimum = minimize(contract.weighted_bce_objective, np.zeros(3), args=(z, labels, weights, 0.1),
                                   jac=contract.weighted_bce_gradient, method="L-BFGS-B",
                                   options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
                counts["dual"] += 1
                if not optimum.success or not np.isfinite(optimum.x).all():
                    raise RuntimeError("dual head optimizer did not converge")
                heads[head] = {"theta": optimum.x, "pre": pre, "support": support}
            fold_rows.append({"horizon": horizon, "seed": seed, "position": position, "training_n": len(A),
                              "heldout_n": len(B), "alpha": best, "inner_MA_MAE": losses[best],
                              "error_fallback": bool(gate["fallback"])})
            if len(B):
                if set(A.tolist()) | set(B.tolist()) != set(range(60)):
                    raise AssertionError(f"outer fold does not cover pool for seed {seed}, position {position}")
                predicted = ridge.predict(model, x[pool_global[B]]); predicted_class = ma_class(predicted[:, 2])
                base_score = score_ma_gate(gate, ma_features(x[pool_global[B]], predicted, ym, ys), predicted_class)
                dual_score = _head_scores(heads, x[pool_global[B]], predicted)
                for j, local_index in enumerate(B.tolist()):
                    global_index = int(pool_global[local_index]); valid = bool(qc[global_index] and np.isfinite(predicted[j]).all())
                    rows.append({"record_id": str(ids[global_index]), "pred_class": int(predicted_class[j]),
                                 "truth_class": int(ma_class(medical[global_index, 2])),
                                 "base_score": float(base_score[j]) if valid and np.isfinite(base_score[j]) else None,
                                 "dual_score": float(dual_score[j]) if valid and predicted_class[j] == 0 and np.isfinite(dual_score[j]) else None})
            else:
                final = {"predictor": model, "error": gate, "heads": heads, "ym": ym, "ys": ys,
                         "pool": pool_global.copy(), "test": test_global.copy(), "seed": seed}
        if final is None or len(rows) != 60 or len({row["record_id"] for row in rows}) != 60:
            raise AssertionError(f"outer development records are not exhaustive for seed {seed}")
        final["threshold"] = _choose_threshold(rows)
        path = ROOT / f"T{horizon}" / f"model_{seed}.joblib"
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(final, path, compress=3); model_hashes[str(path.resolve())] = sha(path)
        working_rows.append({"horizon": horizon, "seed": seed, "normal": final["threshold"]["normal"],
                             "tail": final["threshold"]["tail"], "all_withhold": final["threshold"]["all_withhold"],
                             "normal_status": final["threshold"]["normal_status"], "tail_status": final["threshold"]["tail_status"]})
        log.write(json.dumps({"horizon": horizon, "seed": seed, "fit_counts": counts.copy(),
                              "all_withhold": final["threshold"]["all_withhold"]}) + "\n"); log.flush()
        print("FROZEN", horizon, seed, final["threshold"], flush=True)
    if counts["ridge"] != 1040 or counts["error"] != 80 or counts["dual"] > 160:
        raise AssertionError(f"unexpected fit counts: {counts}")
    return fold_rows, head_rows, working_rows, counts, model_hashes


def train() -> None:
    """Fit and freeze only 300/600 target-only models."""
    if (ROOT / "FREEZE.json").exists() or list(ROOT.glob("T*/model_*.joblib")):
        raise RuntimeError("target-only prefix outputs already exist")
    if any((ROOT / name).exists() for name in ("development_folds.csv", "development_heads.csv",
                                                "development_working_points.csv", "training.log")):
        raise RuntimeError("partial target-only prefix outputs already exist")
    inputs = _input_hashes(); synthetic = _synthetic_checks()
    partition_checks = {str(seed): _check_partition_identity(seed) for seed in SEEDS}
    d15 = load_checkpoint(900); prefix_checks: dict[str, Any] = {}
    for horizon in (300, 600):
        current = load_checkpoint(horizon)
        if not np.array_equal(current["ids"], d15["ids"]) or not np.array_equal(current["groups"], d15["groups"]):
            raise AssertionError(f"record identity drift at {horizon}")
        if not np.array_equal(current["medical"], d15["medical"]):
            raise AssertionError(f"reference identity drift at {horizon}")
        finite = np.asarray(current["finite_valid"], dtype=bool)
        if not bool(finite.all()):
            raise AssertionError(f"non-finite prefix row at {horizon}")
        prefix_checks[str(horizon)] = {"records": 480, "features": 4, "finite_rows": int(finite.sum()),
                                       "input_qc_true": int(np.asarray(current["input_qc"], dtype=bool).sum()),
                                       "checkpoint_specific_finite_qc": True}
    if np.array_equal(load_checkpoint(300)["x"], load_checkpoint(600)["x"]):
        raise AssertionError("300/600 prefix representations unexpectedly identical")

    training_log = ROOT / "training.log"
    with training_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"stage": "TRAINING_START", "horizons": [300, 600], "seeds": SEEDS}) + "\n")
        fold_rows_all: list[dict[str, Any]] = []; head_rows_all: list[dict[str, Any]] = []
        working_rows_all: list[dict[str, Any]] = []; totals = {"ridge": 0, "error": 0, "dual": 0}; model_hashes: dict[str, str] = {}
        for horizon in (300, 600):
            folds, heads, working, counts, hashes = _fit_horizon(horizon, log)
            fold_rows_all.extend(folds); head_rows_all.extend(heads); working_rows_all.extend(working)
            for key in totals: totals[key] += counts[key]
            model_hashes.update(hashes)
        log.write(json.dumps({"stage": "TRAINING_COMPLETE", "fit_counts": totals}) + "\n")
    _write_table("development_folds.csv", fold_rows_all)
    _write_table("development_heads.csv", head_rows_all)
    _write_table("development_working_points.csv", working_rows_all)
    for seed in SEEDS:
        path = FOCUSED / f"model_{seed}.joblib"; model_hashes[str(path.resolve())] = sha(path)
    if len(model_hashes) != 60 or any(sha(Path(path)) != digest for path, digest in model_hashes.items()):
        raise AssertionError("model hashes changed before FREEZE creation")
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise AssertionError("input hashes changed before FREEZE creation")
    checks = {"synthetic": synthetic, "partition_checks": partition_checks, "prefix_checks": prefix_checks,
              "record_order_matches_DATA15": True, "medical_order_matches_DATA15": True, "pool_n": 60,
              "test_n": 201, "seed_n": 20, "ridge_penalties": ALPHAS,
              "fit_counts_per_horizon": {"ridge": 1040, "error": 80, "dual_max": 160},
              "total_fit_counts": totals, "all_20_seed_pool_test_cross_checkpoint_equal": True,
              "all_20_seed_four_outer_A_B_and_inner_T_V_equal": True,
              "original_outer_inner_folds_equal": True,
              "finite_handling": "raw finite prefix only; invalid values remain fail-closed; no imputation",
              "unsupported_heads_preserved": True, "zero_release_preserved": True,
              "test_labels_read_for_selection": False, "external_test_evaluated": False,
              "nine_hundred_models_reused_unchanged": True, "training_log_sha256": sha(training_log)}
    _write_json("FREEZE.json", {"inputs": inputs, "models": model_hashes, "checks": checks, "test_evaluated": False})
    print("FREEZE_CREATED", ROOT / "FREEZE.json", "models", len(model_hashes), flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"check", "train"}:
        raise SystemExit("usage: train_prefix.py {check|train}")
    if sys.argv[1] == "check":
        print(json.dumps(_synthetic_checks(), indent=2, allow_nan=False))
    else:
        train()


if __name__ == "__main__":
    main()
