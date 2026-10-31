"""Frozen 20/40/60 local-label budget comparison.

This script is intentionally local to the new experiment directory.  It reads
the existing C20/C60 transfer artifacts and source payloads, fits only the
new C40 target-side adapters and target-only Ridge controls, and evaluates the
saved predictors on the already fixed SAHZU test cohort.  No reporting gate is
fit or scored here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
HIST = OUT / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551"
S8 = OUT / "uteg_supplement_completion_20261027_203140"
BASE = OUT / "uteg_legacy_aligned_threecenter_20261022_155614"
DATA_PATH = BASE / "DATA15.npz"
ROLE_PATH = BASE / "ROLE_FREEZE.json"
TARGET_ONLY_DIR = ROOT / "target_only"
TRANSFER40_DIR = ROOT / "transfer40"
PRED_DIR = ROOT / "predictions"
DIRS = ("HZ_to_SA", "HZ+HB_to_SA")
SEEDS = tuple(range(1000, 1020))
BUDGETS = (20, 40, 60)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
PARAMS = ("R", "K", "MA", "alpha")
CLASS_NAMES = {-1: "low", 0: "within", 1: "high"}
SOURCE_KEY = {"HZ_to_SA": "exp_D2_HZ", "HZ+HB_to_SA": "dual_SA"}
BOOTSTRAP_SEED = 20261030
BOOTSTRAP_REPS = 5000
POOL_HASH_PREFIX = "uteg-minimal-label-budget-v1|"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = _json_safe(value)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_data() -> dict[str, np.ndarray]:
    return load_npz(DATA_PATH)


def c20_model_path(direction: str, seed: int) -> Path:
    return S8 / "runs" / f"budget_C20__{direction}" / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"


def full_model_path(direction: str, seed: int) -> Path:
    return HIST / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"


def c20_prediction_path(direction: str) -> Path:
    return S8 / "predictions" / f"budget_C20__{direction}.npz"


def c60_prediction_path(direction: str) -> Path:
    return HIST / "eval" / "predictions" / f"{direction}.npz"


def source_payload_path(direction: str, seed: int) -> Path:
    return HIST / "source_payloads" / SOURCE_KEY[direction] / f"seed{seed}.joblib"


def target_only_path(direction: str, budget: int, seed: int) -> Path:
    return TARGET_ONLY_DIR / direction / f"targetonly_b{budget}_seed{seed}.joblib"


def transfer40_model_path(direction: str, seed: int) -> Path:
    return TRANSFER40_DIR / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"


def id_hash(record_id: str) -> str:
    return hashlib.sha256((POOL_HASH_PREFIX + str(record_id)).encode()).hexdigest()


def get_budget_pools(data: dict[str, np.ndarray], direction: str, seed: int) -> dict[str, Any]:
    c20 = joblib.load(c20_model_path(direction, seed))
    full = joblib.load(full_model_path(direction, seed))
    pool20 = [int(v) for v in c20["pool"]]
    pool60 = [int(v) for v in full["pool"]]
    test20 = [int(v) for v in c20["test"]]
    test60 = [int(v) for v in full["test"]]
    if test20 != test60 or len(test60) != 201:
        raise AssertionError(f"test mismatch: {direction} seed {seed}")
    if len(pool20) != 20 or len(pool60) != 60 or not set(pool20) <= set(pool60):
        raise AssertionError(f"pool mismatch: {direction} seed {seed}")
    remaining = sorted(set(pool60) - set(pool20), key=lambda i: id_hash(str(data["ids"][i])))
    add20 = remaining[:20]
    pool40 = sorted(pool20 + add20)
    if len(pool40) != 40 or set(pool20) & set(add20):
        raise AssertionError("invalid C40 union")
    # Group-disjointness is the leakage check used by the historical target fit.
    test_groups = set(data["groups"][test60].astype(str).tolist())
    for label, pool in (("20", pool20), ("40", pool40), ("60", pool60)):
        if set(data["groups"][pool].astype(str).tolist()) & test_groups:
            raise AssertionError(f"group leakage {direction} {seed} {label}")
    if set(full["source_rows"]) & (set(pool60) | set(test60)):
        raise AssertionError(f"source row overlap {direction} {seed}")
    return {
        "pool20": pool20,
        "pool40": pool40,
        "pool60": pool60,
        "add20": add20,
        "test": test60,
        "source_rows": [int(v) for v in full["source_rows"]],
        "c20_obj": c20,
        "full_obj": full,
    }


def freeze_inputs() -> None:
    freeze_path = ROOT / "FREEZE.json"
    if freeze_path.exists():
        raise RuntimeError("FREEZE.json already exists; frozen input design is immutable")
    data = load_data()
    role = json.loads(ROLE_PATH.read_text(encoding="utf-8"))
    records: dict[str, Any] = {}
    hashes: dict[str, str] = {
        str(DATA_PATH): sha(DATA_PATH),
        str(ROLE_PATH): sha(ROLE_PATH),
        str(Path(__file__)): sha(Path(__file__)),
        str(ROOT / "ANALYSIS_PLAN.md"): sha(ROOT / "ANALYSIS_PLAN.md"),
        str(HIST / "run_ridge10_fullchain.py"): sha(HIST / "run_ridge10_fullchain.py"),
        str(HIST / "runtime.py"): sha(HIST / "runtime.py"),
        str(BASE / "train_target_legacy.py"): sha(BASE / "train_target_legacy.py"),
        str(OUT / "uteg_hill_exp_matched_20261012_153337" / "models.py"): sha(OUT / "uteg_hill_exp_matched_20261012_153337" / "models.py"),
    }
    for direction in DIRS:
        for seed in SEEDS:
            pools = get_budget_pools(data, direction, seed)
            c20mp = c20_model_path(direction, seed)
            fullmp = full_model_path(direction, seed)
            c20pp = c20_prediction_path(direction)
            fullpp = c60_prediction_path(direction)
            srcp = source_payload_path(direction, seed)
            for p in (c20mp, fullmp, srcp):
                hashes[str(p)] = sha(p)
            # Prediction hashes are direction-level inputs, but retain them in
            # every record so a row can be audited without inference output.
            hashes[str(c20pp)] = sha(c20pp)
            hashes[str(fullpp)] = sha(fullpp)
            key = f"{direction}::{seed}"
            records[key] = {
                "direction": direction,
                "seed": int(seed),
                "pool20": pools["pool20"],
                "pool40": pools["pool40"],
                "pool60": pools["pool60"],
                "add20_from_remaining40": pools["add20"],
                "test": pools["test"],
                "source_rows": pools["source_rows"],
                "pool20_sha256": canonical_sha(pools["pool20"]),
                "pool40_sha256": canonical_sha(pools["pool40"]),
                "pool60_sha256": canonical_sha(pools["pool60"]),
                "selected_addition_hashes": [id_hash(str(data["ids"][i])) for i in pools["add20"]],
                "test_sha256": canonical_sha(pools["test"]),
                "c20_model_path": str(c20mp),
                "c60_model_path": str(fullmp),
                "c20_prediction_path": str(c20pp),
                "c60_prediction_path": str(fullpp),
                "source_payload_path": str(srcp),
                "source_model_digest_c60": str(pools["full_obj"].get("source_model_digest", "")),
                "source_labels_read_for_fit": False,
                "test_labels_used_for_fit": False,
            }
    payload = {
        "version": "minimal_local_label_budget_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "directions": list(DIRS),
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "alphas": list(ALPHAS),
        "selection_rule": "C40 = exact per-seed S8 C20 pool + first 20 remaining C60 rows sorted by sha256(uteg-minimal-label-budget-v1|record_id)",
        "record_identity_hash_prefix": POOL_HASH_PREFIX,
        "bootstrap": {"unit": "identity-proxy group", "repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "groups_kept_paired_across_20_seeds": True},
        "test_cohort": {"records": 201, "labels_masked_during_fit": True, "gate_not_fit_or_scored": True},
        "hashes": hashes,
        "records": records,
        "role_sha256": sha(ROLE_PATH),
        "data_sha256": sha(DATA_PATH),
        "test_evaluated_before_freeze": False,
    }
    write_json(freeze_path, payload)
    write_json(ROOT / "INPUT_MANIFEST.json", {"version": payload["version"], "hashes": hashes, "record_keys": sorted(records), "private_pool_manifest": "FREEZE.json", "test_evaluated_before_freeze": False})
    write_json(ROOT / "ENVIRONMENT.json", {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "cwd": str(Path.cwd()),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    })
    print(json.dumps({"status": "FROZEN", "records": len(records), "hashes": len(hashes)}, indent=2))


def load_freeze() -> dict[str, Any]:
    p = ROOT / "FREEZE.json"
    if not p.exists():
        raise FileNotFoundError("run freeze first")
    return json.loads(p.read_text(encoding="utf-8"))


def restricted_positions(obj: dict[str, Any], pool: list[int]) -> list[dict[str, Any]]:
    poolset = set(int(v) for v in pool)
    positions: list[dict[str, Any]] = []
    values = obj["positions"].values() if isinstance(obj["positions"], dict) else obj["positions"]
    for code0, p0 in enumerate(values):
        code = int(p0.get("code", code0))
        p = {"code": code}
        p["A"] = [int(v) for v in p0["A"] if int(v) in poolset]
        p["B"] = [int(v) for v in p0["B"] if int(v) in poolset]
        p["inner"] = []
        for f0 in p0.get("inner", []):
            train = [int(v) for v in f0["T"] if int(v) in poolset]
            valid = [int(v) for v in f0["V"] if int(v) in poolset]
            if valid:
                if not train or set(train) & set(valid):
                    raise AssertionError(f"invalid restricted inner fold code={code}")
                p["inner"].append({"T": train, "V": valid})
        positions.append(p)
    if len(positions) != 4:
        raise AssertionError("expected four target positions")
    for p in positions[:3]:
        if set(p["A"]) | set(p["B"]) != poolset or set(p["A"]) & set(p["B"]):
            raise AssertionError(f"outer fold does not cover budget pool: {p['code']}")
    if set(positions[3]["A"]) != poolset or positions[3]["B"]:
        raise AssertionError("position 3 is not a complete budget fit")
    if set(v for p in positions[:3] for v in p["B"]) != poolset:
        raise AssertionError("outer B folds do not partition the budget pool")
    return positions


def fit_target_only() -> dict[str, Any]:
    freeze = load_freeze()
    data = load_data()
    model_hashes: dict[str, str] = {}
    selection_rows: list[dict[str, Any]] = []
    by_key = freeze["records"]
    for direction in DIRS:
        for seed in SEEDS:
            rec = by_key[f"{direction}::{seed}"]
            # Select from the masked fit copy; no test labels are reachable by
            # this path, even accidentally through a debug calculation.
            D = {k: np.array(v, copy=True) for k, v in data.items()}
            test = np.asarray(rec["test"], dtype=int)
            D["medical"][test] = np.nan
            D["ci"][test] = np.nan
            for budget in BUDGETS:
                pool = np.asarray(rec[f"pool{budget}"], dtype=int)
                obj = joblib.load(c20_model_path(direction, seed) if budget == 20 else full_model_path(direction, seed))
                positions = restricted_positions(obj, pool.tolist())
                scores: list[float] = []
                for alpha in ALPHAS:
                    errors: list[float] = []
                    for fold in positions[:3]:
                        A = np.asarray(fold["A"], dtype=int)
                        B = np.asarray(fold["B"], dtype=int)
                        m = ridge_fit(D["x"][A], D["medical"][A], alpha)
                        pred = ridge_predict(m, D["x"][B])
                        errors.extend(np.abs(pred[:, 2] - D["medical"][B, 2]).tolist())
                    if not errors:
                        raise AssertionError("empty development errors")
                    score = float(np.mean(errors))
                    scores.append(score)
                    selection_rows.append({"direction": direction, "budget": budget, "seed": seed, "alpha": alpha, "development_MA_MAE": score, "pool_n": len(pool)})
                best_i = min(range(len(ALPHAS)), key=lambda i: (scores[i], -ALPHAS[i]))
                alpha = float(ALPHAS[best_i])
                m = ridge_fit(D["x"][pool], D["medical"][pool], alpha)
                m.update({"direction": direction, "seed": int(seed), "budget": int(budget), "pool": pool, "test": test, "development_scores": scores, "test_labels_used_for_fit": False, "source_labels_used": False})
                dest = target_only_path(direction, budget, seed)
                dest.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(m, dest, compress=3)
                model_hashes[str(dest)] = sha(dest)
    with (ROOT / "target_only_development_selection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(selection_rows[0]))
        w.writeheader(); w.writerows(selection_rows)
    out = {"version": "target_only_ridge_budget_v1", "model_hashes": model_hashes, "models": len(model_hashes), "test_labels_used_for_fit": False, "gate_fit": False}
    write_json(ROOT / "TARGET_ONLY_FREEZE.json", out)
    return out


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    xm = x.mean(0); xs = np.maximum(x.std(0), 1e-6)
    ym = y.mean(0); ys = np.maximum(y.std(0), 1e-6)
    xx = (x - xm) / xs; yy = (y - ym) / ys
    coef = np.linalg.solve(xx.T @ xx + alpha * np.eye(x.shape[1]), xx.T @ yy)
    return {"xm": xm, "xs": xs, "ym": ym, "ys": ys, "coef": coef, "alpha": float(alpha)}


def project(y: np.ndarray) -> np.ndarray:
    out = np.array(y, copy=True)
    out[..., :3] = np.maximum(out[..., :3], 0)
    out[..., 3] = np.clip(out[..., 3], 0, 89.999999)
    return out


def ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return project(np.asarray(model["ym"]) + np.asarray(model["ys"]) * (((x - model["xm"]) / model["xs"]) @ model["coef"]))


def module_source(name: str, path: Path, source: str) -> types.ModuleType:
    mod = types.ModuleType(name); mod.__file__ = str(path); sys.modules[name] = mod
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def load_fit_modules() -> tuple[Any, Any, Any]:
    """Load the historical numerical definitions while redirecting writes."""
    TRANSFER40_DIR.mkdir(parents=True, exist_ok=True)
    runtime_text = (HIST / "runtime.py").read_text(encoding="utf-8")
    runtime_text = runtime_text.replace("OUT = ROOT.parent", f"OUT = Path({str(OUT)!r})")
    runtime_text = runtime_text.replace("OUT.parent /", f"Path({str(OUT)!r}) /")
    runtime = module_source("runtime", ROOT / "runtime_dynamic.py", runtime_text)
    runtime.ROOT = TRANSFER40_DIR
    (TRANSFER40_DIR / "ledger").mkdir(parents=True, exist_ok=True)
    runtime._LEDGER_TOTALS.update(runtime._totals())
    runner_text = (HIST / "run_ridge10_fullchain.py").read_text(encoding="utf-8")
    # The requested run is a predictive-layer arm.  Preserve the same mapper
    # and nested adapter selection but replace the endpoint MA gate block with
    # a no-op, as in the historical C20 budget artifact.
    start = runner_text.index('            r.ledger(f"{key}_MA_GATE_{code}"')
    end = runner_text.index("            outer_b = _outer_b_metadata", start)
    runner_text = runner_text[:start] + '''            gate = None
            med_B = _position_outer_prediction(p, adapter) if B else np.empty((0,4))
            pred_ma_B = med_B[:,2]
            pred_B = ma_class(pred_ma_B)
            truth_B = ma_class(D["medical"][B,2])
            score_B = np.full(len(B), np.nan)
''' + runner_text[end:]
    runner_text = runner_text.replace('"arm": "FULL", "budget": "FULL"', '"arm": "B40", "budget": "40"')
    runner = module_source("minimal_budget_runner", ROOT / "runner_dynamic.py", runner_text)
    target_text = (BASE / "train_target_legacy.py").read_text(encoding="utf-8")
    target_text = target_text.replace('OLD_MODELS = r.ROOT.parent / "uteg_hill_exp_matched_20261012_153337" / "models.py"', f'OLD_MODELS = Path({str(OUT / "uteg_hill_exp_matched_20261012_153337" / "models.py")!r})')
    target = module_source("minimal_budget_target", ROOT / "target_dynamic.py", target_text)
    runtime.write("RUNNER_FREEZE.json", {"version": "minimal_budget_transfer40_runner_v1", "protocol_sha256": sha(ROOT / "ANALYSIS_PLAN.md"), "runner_source_sha256": sha(HIST / "run_ridge10_fullchain.py"), "test_labels_read": False, "test_evaluated": False})
    runtime.write("ROUND_START_LEDGER.json", {"version": "minimal_budget_transfer40_ledger_v1", "input_manifest_sha256": sha(ROOT / "FREEZE.json"), "test_labels_read": False, "test_evaluated": False})
    return runtime, runner, target


def fit_transfer40() -> dict[str, Any]:
    freeze = load_freeze(); data = load_data(); role = json.loads(ROLE_PATH.read_text(encoding="utf-8"))
    runtime, runner, target = load_fit_modules()
    D = {k: np.array(v, copy=True) for k, v in data.items()}
    target._load_role = lambda: role
    target._load_data = lambda data_path=None, mode="T900": dict(D)
    target._freeze_before_fit = lambda *args, **kwargs: None
    model_hashes: dict[str, str] = {}
    protocol_sha = sha(ROOT / "ANALYSIS_PLAN.md")
    for direction in DIRS:
        for seed in SEEDS:
            rec = freeze["records"][f"{direction}::{seed}"]
            full = joblib.load(full_model_path(direction, seed))
            pool = list(rec["pool40"])
            positions = restricted_positions(full, pool)
            target._spec = (lambda role0, target0, seed0, budget0, positions=positions, pool=pool, test=rec["test"]: {
                "target": target0, "arm": "FULL", "seed": int(seed0), "d": 2 if target0 == "SAHZU" else 1,
                "pool": list(pool), "test": list(test), "positions": positions,
            })
            source_path = source_payload_path(direction, seed)
            runtime.set_scope(f"C40_{direction}_{seed}")
            output = runner.fit_target_one(target, direction, seed, source_path, role, D, rec["source_rows"], protocol_sha)
            obj = joblib.load(output)
            if obj.get("budget") != "40" or len(obj.get("pool", [])) != 40 or obj.get("external_test_evaluated") is not False:
                raise AssertionError(f"C40 object contract failed: {output}")
            model_hashes[str(output)] = sha(output)
    out = {"version": "transfer_c40_nested_adapter_v1", "model_hashes": model_hashes, "models": len(model_hashes), "source_refit": False, "test_labels_used_for_fit": False, "gate_fit": False}
    write_json(ROOT / "TRANSFER40_FREEZE.json", out)
    return out


def fit_all() -> None:
    freeze = load_freeze()
    target_result = fit_target_only()
    transfer_result = fit_transfer40()
    write_json(ROOT / "FIT_COMPLETE.json", {
        "version": "minimal_local_label_budget_fit_complete_v1",
        "target_only": target_result,
        "transfer40": transfer_result,
        "test_evaluated": False,
        "test_labels_used_for_fit": False,
        "source_refit": False,
        "gate_fit": False,
        "freeze_sha256": sha(ROOT / "FREEZE.json"),
    })
    print(json.dumps({"status": "FIT_COMPLETE", "target_only_models": target_result["models"], "transfer40_models": transfer_result["models"]}, indent=2))


def infer_transfer40() -> dict[tuple[str, int], dict[str, np.ndarray]]:
    data = load_data(); out: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    # Numerical prediction needs only the frozen historical mapper definitions;
    # no labels are passed to this function.
    runtime, _runner, target = load_fit_modules()
    test_by_key = {k: np.asarray(v["test"], dtype=int) for k, v in load_freeze()["records"].items()}
    for direction in DIRS:
        for seed in SEEDS:
            obj = joblib.load(transfer40_model_path(direction, seed))
            idx = test_by_key[f"{direction}::{seed}"]
            src = obj["source_model"]
            raw = target.m.predict(src, data["x"][idx])
            cache = obj["cache"][obj["positions"][3]["base"]]
            pure = target.m.calibrated(cache["cal"], raw)
            adapter = obj["final_selection"]["adapter"]
            final = pure if adapter == "H" else target.m.ans["adapt"](cache["adapters"][adapter], data["x"][idx], raw, pure, src)
            if not (np.isfinite(raw).all() and np.isfinite(pure).all() and np.isfinite(final).all()):
                raise AssertionError(f"non-finite C40 prediction {direction} {seed}")
            out[(direction, seed)] = {"raw_med4": np.asarray(raw), "pure_huber_med4": np.asarray(pure), "final_med4": np.asarray(final), "index": idx}
    return out


def normalize_saved_prediction(path: Path, data: dict[str, np.ndarray], expected_index: list[int]) -> dict[str, np.ndarray]:
    z = load_npz(path)
    if "index" in z:
        idx = np.asarray(z["index"], dtype=int)
    else:
        idx = np.asarray(z["test_global_index"], dtype=int)
    if not np.array_equal(idx, np.asarray(expected_index, dtype=int)):
        raise AssertionError(f"test index mismatch in {path}")
    ids = z["ids"] if "ids" in z else z["test_ids"]
    groups = z["groups"] if "groups" in z else data["groups"][idx]
    if not np.array_equal(ids.astype(str), data["ids"][idx].astype(str)) or not np.array_equal(groups.astype(str), data["groups"][idx].astype(str)):
        raise AssertionError(f"test identity mismatch in {path}")
    return {"seed": np.asarray(z["seed"], dtype=int), "ids": ids.astype(str), "groups": groups.astype(str), "index": idx, "raw_med4": z["raw_med4"], "pure_huber_med4": z["pure_huber_med4"], "final_med4": z["final_med4"]}


def evaluate_predictions() -> None:
    if not (ROOT / "FIT_COMPLETE.json").exists():
        raise RuntimeError("fit first")
    freeze = load_freeze(); data = load_data(); c40 = infer_transfer40()
    pred_hashes: dict[str, str] = {}
    for direction in DIRS:
        expected = freeze["records"][f"{direction}::1000"]["test"]
        c20 = normalize_saved_prediction(c20_prediction_path(direction), data, expected)
        c60 = normalize_saved_prediction(c60_prediction_path(direction), data, expected)
        for budget, saved in ((20, c20), (60, c60)):
            # Re-save only aggregate-indexed private arrays inside this run so
            # all later arithmetic uses one common schema.
            out = PRED_DIR / f"{direction}_transfer_b{budget}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, seed=saved["seed"], ids=saved["ids"], groups=saved["groups"], index=saved["index"], raw_med4=saved["raw_med4"], pure_huber_med4=saved["pure_huber_med4"], final_med4=saved["final_med4"])
            pred_hashes[str(out)] = sha(out)
        c40_final = np.stack([c40[(direction, seed)]["final_med4"] for seed in SEEDS])
        c40_raw = np.stack([c40[(direction, seed)]["raw_med4"] for seed in SEEDS])
        c40_pure = np.stack([c40[(direction, seed)]["pure_huber_med4"] for seed in SEEDS])
        out = PRED_DIR / f"{direction}_transfer_b40.npz"
        np.savez_compressed(out, seed=np.asarray(SEEDS, dtype=int), ids=data["ids"][expected], groups=data["groups"][expected], index=np.asarray(expected, dtype=int), raw_med4=c40_raw, pure_huber_med4=c40_pure, final_med4=c40_final)
        pred_hashes[str(out)] = sha(out)
        for budget in BUDGETS:
            models = [joblib.load(target_only_path(direction, budget, seed)) for seed in SEEDS]
            idx = np.asarray(expected, dtype=int)
            final = np.stack([ridge_predict(m, data["x"][idx]) for m in models])
            out = PRED_DIR / f"{direction}_targetonly_b{budget}.npz"
            np.savez_compressed(out, seed=np.asarray(SEEDS, dtype=int), ids=data["ids"][idx], groups=data["groups"][idx], index=idx, final_med4=final, alpha=np.asarray([m["alpha"] for m in models], dtype=float))
            pred_hashes[str(out)] = sha(out)
    write_json(ROOT / "PREDICTIONS_FREEZE.json", {"version": "minimal_local_label_budget_predictions_v1", "files": pred_hashes, "test_evaluated": True, "gate_fit": False, "test_tuning": False, "source_refit": False})
    print(json.dumps({"status": "PREDICTIONS_EXPORTED", "files": len(pred_hashes)}, indent=2))


class GroupBootstrap:
    def __init__(self, groups: np.ndarray):
        self.groups = np.asarray(groups).astype(str)
        self.unique = np.unique(self.groups)
        self.draw = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(self.unique), (BOOTSTRAP_REPS, len(self.unique)))

    def ratio(self, numerator: np.ndarray, denominator: np.ndarray | float = 1.0) -> dict[str, Any]:
        num = np.asarray(numerator, dtype=float)
        den = np.broadcast_to(np.asarray(denominator, dtype=float), num.shape)
        gn = np.asarray([num[:, self.groups == g].sum() for g in self.unique], dtype=float)
        gd = np.asarray([den[:, self.groups == g].sum() for g in self.unique], dtype=float)
        bden = gd[self.draw].sum(axis=1)
        ok = bden > 0
        point = float(gn.sum() / gd.sum()) if gd.sum() else None
        values = gn[self.draw].sum(axis=1)[ok] / bden[ok] if ok.any() else np.array([])
        ci = np.quantile(values, [0.025, 0.975]).tolist() if len(values) else [None, None]
        return {"estimate": point, "lo": ci[0], "hi": ci[1], "valid_resamples": int(ok.sum())}


def class_code(ma: np.ndarray) -> np.ndarray:
    return np.where(ma < 50, -1, np.where(ma > 70, 1, 0)).astype(np.int8)


def evaluate_metrics() -> None:
    if not (ROOT / "PREDICTIONS_FREEZE.json").exists():
        raise RuntimeError("run evaluate first")
    data = load_data(); rows: list[dict[str, Any]] = []; paired: list[dict[str, Any]] = []; flows: list[dict[str, Any]] = []; summaries: dict[str, Any] = {}
    for direction in DIRS:
        rec0 = load_freeze()["records"][f"{direction}::1000"]
        idx = np.asarray(rec0["test"], dtype=int); Y = data["medical"][idx]; truth = class_code(Y[:, 2]); groups = data["groups"][idx].astype(str); stat = GroupBootstrap(groups)
        arms: dict[tuple[str, int], dict[str, Any]] = {}
        for budget in BUDGETS:
            for arm in ("transfer", "target-only"):
                z = load_npz(PRED_DIR / f"{direction}_{arm.replace('-', '')}_b{budget}.npz")
                if not np.array_equal(z["seed"], np.asarray(SEEDS, dtype=int)) or not np.array_equal(z["index"], idx):
                    raise AssertionError(f"prediction schema mismatch {direction} {arm} {budget}")
                arms[(arm, budget)] = z
        direction_summary: dict[str, Any] = {"records": int(len(idx)), "groups": int(len(np.unique(groups))), "seeds": len(SEEDS), "arms": {}, "paired": {}}
        for (arm, budget), z in arms.items():
            P = z["final_med4"]
            E = np.abs(P - Y[None, :, :])
            key = f"{arm}_b{budget}"
            mm: dict[str, Any] = {"budget": budget, "arm": arm, "parameters": {}}
            for j, param in enumerate(PARAMS):
                q = stat.ratio(E[:, :, j], 1.0)
                mm["parameters"][param] = q
            pc = class_code(P[:, :, 2]); mm["class_destinations"] = {}
            for true_state in (-1, 0, 1):
                denom = int(np.sum(truth == true_state) * len(SEEDS))
                for pred_state in (-1, 0, 1):
                    n = int(np.sum((truth[None, :] == true_state) & (pc == pred_state)))
                    flows.append({"direction": direction, "arm": arm, "budget": budget, "true_state": CLASS_NAMES[true_state], "predicted_state": CLASS_NAMES[pred_state], "n": n, "N": denom, "proportion": n / denom if denom else None})
                    mm["class_destinations"].setdefault(CLASS_NAMES[true_state], {})[CLASS_NAMES[pred_state]] = n
            direction_summary["arms"][key] = mm
            for k, seed in enumerate(SEEDS):
                row = {"direction": direction, "arm": arm, "budget": budget, "seed": seed}
                for j, param in enumerate(PARAMS): row[param + "_MAE"] = float(E[k, :, j].mean())
                row["MA_class_accuracy"] = float(np.mean(pc[k] == truth))
                row["MA_class_n"] = int(np.sum(pc[k] == truth))
                row["low_to_within"] = int(np.sum((truth == -1) & (pc[k] == 0)))
                row["high_to_within"] = int(np.sum((truth == 1) & (pc[k] == 0)))
                row["selected_alpha"] = float(z["alpha"][k]) if "alpha" in z else None
                rows.append(row)
        contrasts = []
        for budget in BUDGETS:
            A = arms[("target-only", budget)]["final_med4"]; B = arms[("transfer", budget)]["final_med4"]
            for j, param in enumerate(PARAMS):
                q = stat.ratio(np.abs(A[:, :, j] - Y[None, :, j]) - np.abs(B[:, :, j] - Y[None, :, j]), 1.0)
                name = f"target-only_minus_transfer_b{budget}"
                paired.append({"direction": direction, "contrast": name, "parameter": param, **q})
                contrasts.append({"contrast": name, "parameter": param, **q})
        for arm in ("transfer", "target-only"):
            for b0, b1 in ((20, 40), (40, 60), (20, 60)):
                A = arms[(arm, b1)]["final_med4"]; B = arms[(arm, b0)]["final_med4"]
                for j, param in enumerate(PARAMS):
                    q = stat.ratio(np.abs(A[:, :, j] - Y[None, :, j]) - np.abs(B[:, :, j] - Y[None, :, j]), 1.0)
                    name = f"{arm}_b{b1}_minus_b{b0}"
                    paired.append({"direction": direction, "contrast": name, "parameter": param, **q})
                    contrasts.append({"contrast": name, "parameter": param, **q})
        direction_summary["paired"] = contrasts
        summaries[direction] = direction_summary
    fields = list(rows[0]);
    with (ROOT / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    fields = list(paired[0]);
    with (ROOT / "paired_intervals.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(paired)
    fields = list(flows[0]);
    with (ROOT / "class_destinations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flows)
    numeric_audit = []
    max_abs_error = 0.0
    for direction in DIRS:
        rec0 = load_freeze()["records"][f"{direction}::1000"]
        idx = np.asarray(rec0["test"], dtype=int)
        Y = data["medical"][idx]
        for budget in BUDGETS:
            tr = load_npz(PRED_DIR / f"{direction}_transfer_b{budget}.npz")["final_med4"]
            to = load_npz(PRED_DIR / f"{direction}_targetonly_b{budget}.npz")["final_med4"]
            for j, param in enumerate(PARAMS):
                transfer_point = float(np.abs(tr[:, :, j] - Y[None, :, j]).mean())
                target_point = float(np.abs(to[:, :, j] - Y[None, :, j]).mean())
                expected_diff = target_point - transfer_point
                got = next(x for x in paired if x["direction"] == direction and x["contrast"] == f"target-only_minus_transfer_b{budget}" and x["parameter"] == param)
                err_point = abs(float(got["estimate"]) - expected_diff)
                max_abs_error = max(max_abs_error, err_point)
                numeric_audit.append({"direction": direction, "budget": budget, "parameter": param, "transfer_point": transfer_point, "target_only_point": target_point, "expected_target_only_minus_transfer": expected_diff, "paired_interval_estimate": float(got["estimate"]), "absolute_error": err_point})
    if max_abs_error > 1e-12:
        raise AssertionError(f"paired point does not equal direct marginal difference: {max_abs_error}")
    write_json(ROOT / "NUMERIC_AUDIT.json", {"version": "minimal_local_label_budget_numeric_audit_v1", "status": "PASS", "max_absolute_point_error": max_abs_error, "rows": numeric_audit, "note": "Each paired point is independently checked against direct target-only MAE minus transfer MAE using the saved prediction arrays."})
    write_json(ROOT / "RESULTS.json", {"version": "minimal_local_label_budget_results_v1", "bootstrap": {"unit": "identity-proxy group", "groups": 190, "repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "all_seeds_paired": True}, "directions": summaries, "test_cohort_records": 201, "gate_evaluated": False, "raw_ids_public": False})
    write_results_zh(summaries)
    write_json(ROOT / "CHECKS.json", {"freeze_sha256": sha(ROOT / "FREEZE.json"), "predictions_freeze_sha256": sha(ROOT / "PREDICTIONS_FREEZE.json"), "numeric_audit_sha256": sha(ROOT / "NUMERIC_AUDIT.json"), "numeric_audit_status": "PASS", "numeric_audit_max_absolute_point_error": max_abs_error, "all_20_seeds": True, "directions": list(DIRS), "budgets": list(BUDGETS), "source_refit": False, "gate_fit_or_scored": False, "test_tuning": False, "private_ids_in_results_zh": False, "static_check_not_reproduction": True})
    print(json.dumps({d: {k: {p: round(v["parameters"][p]["estimate"], 5) for p in PARAMS} for k, v in s["arms"].items()} for d, s in summaries.items()}, indent=2, ensure_ascii=False))


def write_results_zh(summaries: dict[str, Any]) -> None:
    lines = ["# 最小本地标签预算实验结果", "", "本报告只评价四参数预测层；没有重新拟合或评价报告门，因此不把结果表述为完整 reporting-label efficiency。测试集为固定 201 条记录，20 个 seed 的结果按 identity-proxy group 配对 bootstrap（5,000 次，190 组，seed=20261030）。", "", "## MAE（mm；区间为组 bootstrap 95% CI）", "", "|方向|臂|20 条 MA|40 条 MA|60 条 MA|", "|---|---|---:|---:|---:|"]
    for direction, summary in summaries.items():
        for arm in ("transfer", "target-only"):
            vals = []
            for b in BUDGETS:
                q = summary["arms"][f"{arm}_b{b}"]["parameters"]["MA"]
                vals.append(f"{q['estimate']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}]")
            lines.append(f"|{direction}|{arm}|" + "|".join(vals) + "|")
    lines += ["", "## 同预算 target-only − transfer（正值表示 target-only 误差更高）", "", "|方向|预算|MA 差值|R 差值|K 差值|alpha 差值|", "|---|---:|---:|---:|---:|---:|"]
    for direction, summary in summaries.items():
        for b in BUDGETS:
            row = {x["parameter"]: x for x in summary["paired"] if x["contrast"] == f"target-only_minus_transfer_b{b}"}
            # Keep the displayed order identical to the header.  The internal
            # parameter order is R,K,MA,alpha; the original report accidentally
            # printed those values under an MA,R,K,alpha header.
            display_order = ("MA", "R", "K", "alpha")
            lines.append(f"|{direction}|{b}|" + "|".join(f"{row[p]['estimate']:.3f} [{row[p]['lo']:.3f}, {row[p]['hi']:.3f}]" for p in display_order) + "|")
    lines += ["", "## 如何读取", "", "* C20 transfer 与 C60 transfer 是原先冻结的 S8/fullchain 预测；本轮新增 C40 transfer 只重拟合 target-side adapter，source predictor 不变。", "* target-only 在每个预算使用同一批 target records，alpha 只由三折开发折 MA 误差选择，随后在该预算池重拟合。", "* 这些区间反映当前测试 cohort 与 identity proxies 的描述性不确定性；seed 是配对重复，不是额外患者。", "* 这是预测层的预算曲线，不能单独证明报告覆盖率、错误释放风险或临床部署收益。", "", "原始 record IDs、逐记录预测和 fitted objects 保留在本地实验目录，不进入本摘要。"]
    (ROOT / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("freeze", "fit", "evaluate", "metrics", "all"))
    args = ap.parse_args()
    if args.command == "freeze": freeze_inputs()
    elif args.command == "fit": fit_all()
    elif args.command == "evaluate": evaluate_predictions()
    elif args.command == "metrics": evaluate_metrics()
    else:
        if not (ROOT / "FREEZE.json").exists(): freeze_inputs()
        fit_all(); evaluate_predictions(); evaluate_metrics()


if __name__ == "__main__":
    main()
