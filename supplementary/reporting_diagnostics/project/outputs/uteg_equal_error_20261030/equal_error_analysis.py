from __future__ import annotations

"""Retrospective equal-error-burden diagnostics on frozen UTEG predictions.

This module never fits a predictor.  It imports only pure scoring helpers from
the completed reporting-increment analysis, builds reporting policies from the
60 development OOF rows, and evaluates them on the old 201-record test queue.
The test-label-aware frontier and the cross-calibrated sensitivity analysis are
explicitly exploratory and are kept separate from the prior frozen results.
"""

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
OUT = BASE / "outputs"
PREV = OUT / "uteg_reporting_increment_20261030"
DATA = OUT / "uteg_legacy_aligned_threecenter_20261022_155614" / "DATA15.npz"
FOCUSED = OUT / "uteg_nc_goal_20261030" / "focused_reporting_20261030"
SOURCE_ACTIONS = OUT / "uteg_ridge_dual_head_external_eval_20261026_191117" / "actions"
MANUSCRIPT = OUT / "uteg_nc_goal_20261030" / "manuscript"
DIRS = ["HZ_to_SA", "HZ+HB_to_SA"]
METHODS = ["target_only", *DIRS]
SEEDS = list(range(1000, 1020))
FIXED_CAPS = [0.0, 0.05, 0.10, 0.15, 0.20]
ERROR_BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.075, 0.10]
N_BOOT = 5000
N_CROSS_BOOT = 1000
TAG = "uteg-equal-error-20261030"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): canonical(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canonical(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.write_text(json.dumps(canonical(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(canonical(rows))


def load_old_module():
    p = PREV / "increment_experiment.py"
    spec = importlib.util.spec_from_file_location("frozen_increment_helpers", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inc = load_old_module()


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def test_indices() -> np.ndarray:
    m = joblib.load(FOCUSED / "model_1000.joblib")
    idx = np.asarray(m["test"], dtype=int)
    assert len(idx) == 201 and len(np.unique(idx)) == 201
    return idx


def group_digest(g: str) -> str:
    return hashlib.sha256(f"{TAG}|{g}".encode()).hexdigest()


def input_paths() -> list[Path]:
    paths = [
        DATA,
        PREV / "development_oof_private.json",
        PREV / "increment_experiment.py",
        PREV / "EXPERIMENT_PLAN.md",
        PREV / "frozen_thresholds.json",
        PREV / "validation.json",
        PREV / "root_original_file_hashes.json",
        ROOT / "ANALYSIS_PLAN.md",
        FOCUSED / "run.py",
        MANUSCRIPT / "main.pdf",
        MANUSCRIPT / "supplement.pdf",
        MANUSCRIPT / "Supplementary_Data_1_to_7.zip",
    ]
    for d in DIRS:
        paths.append(SOURCE_ACTIONS / f"{d}.npz")
    for s in SEEDS:
        paths.append(FOCUSED / f"model_{s}.joblib")
    return paths


def manifest() -> dict[str, str]:
    missing = [str(p) for p in input_paths() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + "; ".join(missing))
    return {str(p): sha(p) for p in input_paths()}


def _rows_by_method_seed() -> dict[tuple[str, int], list[dict[str, Any]]]:
    raw = json.loads((PREV / "development_oof_private.json").read_text(encoding="utf-8"))
    assert raw.get("test_read") is False and len(raw["rows"]) == 3600
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in raw["rows"]:
        key = (str(r["method"]), int(r["seed"]))
        out.setdefault(key, []).append(r)
    assert set(k[0] for k in out) == set(METHODS)
    assert all(len(v) == 60 for v in out.values())
    return out


def _threshold_key(v: float | None) -> tuple[int, float]:
    return (0, -math.inf) if v is None else (1, float(v))


def _policy_action(rows: list[dict[str, Any]], normal: float | None, tail: float | None) -> np.ndarray:
    a = np.zeros(len(rows), dtype=bool)
    for j, r in enumerate(rows):
        p = int(r["pred_class"])
        b = r.get("base_score")
        d = r.get("dual_score")
        if p == 0 and normal is not None and d is not None and np.isfinite(float(d)) and float(d) <= normal:
            a[j] = True
        if p in (-1, 1) and tail is not None and b is not None and np.isfinite(float(b)) and float(b) <= tail:
            a[j] = True
    return a


def _stats(rows: list[dict[str, Any]], a: np.ndarray) -> dict[str, Any]:
    p = np.asarray([int(r["pred_class"]) for r in rows])
    t = np.asarray([int(r["truth_class"]) for r in rows])
    rel = int(a.sum())
    err = int((a & (p != t)).sum())
    correct = int((a & (p == t)).sum())
    within = a & (p == 0)
    return {
        "released": rel,
        "errors": err,
        "correct": correct,
        "error_rate": float(err / rel) if rel else 0.0,
        "correct_within": int((within & (t == 0)).sum()),
        "low_to_within": int((within & (t == -1)).sum()),
        "high_to_within": int((within & (t == 1)).sum()),
    }


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal = [None] + sorted({float(r["dual_score"]) for r in rows if int(r["pred_class"]) == 0 and r.get("dual_score") is not None and np.isfinite(float(r["dual_score"]))})
    tail = [None] + sorted({float(r["base_score"]) for r in rows if int(r["pred_class"]) in (-1, 1) and r.get("base_score") is not None and np.isfinite(float(r["base_score"]))})
    truth = {int(r["truth_class"]) for r in rows}
    out = []
    for n in normal:
        for t in tail:
            a = _policy_action(rows, n, t)
            st = _stats(rows, a)
            if st["released"] < 5:
                continue
            if n is not None and (not {-1, 1}.issubset(truth) or st["correct_within"] < 5):
                continue
            out.append({
                "normal": n,
                "tail": t,
                **st,
                "eligible": True,
            })
    return out


def _select_at_cap(candidates: list[dict[str, Any]], cap: float) -> dict[str, Any]:
    eligible = [c for c in candidates if c["error_rate"] <= cap + 1e-12]
    if not eligible:
        return {"normal": None, "tail": None, "released": 0, "errors": 0, "correct": 0, "error_rate": 0.0, "status": "NO_FEASIBLE_PAIR"}
    best = min(eligible, key=lambda c: (-c["released"], c["errors"], _threshold_key(c["normal"]), _threshold_key(c["tail"])))
    return {**best, "status": "SELECTED"}


def _threshold_signature(sel_by_seed: dict[int, dict[str, Any]]) -> tuple:
    return tuple((int(s), sel_by_seed[s].get("normal"), sel_by_seed[s].get("tail"), sel_by_seed[s].get("status")) for s in SEEDS)


def freeze() -> None:
    assert not (ROOT / "EVALUATION_COMPLETE.json").exists()
    rows = _rows_by_method_seed()
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    rates: set[float] = set(float(x) for x in FIXED_CAPS)
    for method in METHODS:
        candidates[method] = {}
        for s in SEEDS:
            cs = _candidate_rows(rows[(method, s)])
            candidates[method][str(s)] = cs
            rates.update(float(c["error_rate"]) for c in cs if 0.0 <= float(c["error_rate"]) <= 0.20 + 1e-12)
    r_grid = sorted(set(round(float(x), 14) for x in rates if 0.0 <= x <= 0.20 + 1e-12))
    if 0.0 not in r_grid:
        r_grid.insert(0, 0.0)
    if 0.2 not in r_grid:
        r_grid.append(0.2)
    configs: dict[str, list[dict[str, Any]]] = {}
    config_lookup: dict[str, dict[str, str]] = {}
    for method in METHODS:
        configs[method] = []
        config_lookup[method] = {}
        seen: dict[tuple, str] = {}
        for r in r_grid:
            sels = {s: _select_at_cap(candidates[method][str(s)], r) for s in SEEDS}
            sig = _threshold_signature(sels)
            if sig not in seen:
                cid = f"{method}__cfg{len(seen):04d}"
                seen[sig] = cid
                configs[method].append({
                    "config_id": cid,
                    "first_r": float(r),
                    "last_r": float(r),
                    "selections": {str(s): sels[s] for s in SEEDS},
                })
            else:
                cid = seen[sig]
                for c in configs[method]:
                    if c["config_id"] == cid:
                        c["last_r"] = float(r)
                        break
            config_lookup[method][f"{r:.14g}"] = seen[sig]
    # Verify the five pre-existing cap selections exactly match the prior run.
    old_freeze = json.loads((PREV / "frozen_thresholds.json").read_text(encoding="utf-8"))
    parity = []
    for method in METHODS:
        for s in SEEDS:
            cs = {float(c["cap"]): c for c in old_freeze["thresholds"] if c["method"] == method and int(c["seed"]) == s}
            for cap in FIXED_CAPS:
                new = _select_at_cap(candidates[method][str(s)], cap)
                old = cs[cap]
                parity.append({"method": method, "seed": s, "cap": cap,
                               "normal_identical": new.get("normal") == old.get("normal"),
                               "tail_identical": new.get("tail") == old.get("tail"),
                               "old_status": old.get("status"), "new_status": new.get("status")})
    assert all(x["normal_identical"] and x["tail_identical"] for x in parity)
    m = manifest()
    write_json(ROOT / "input_manifest.json", {"protocol": TAG, "inputs": m, "methods": METHODS, "seeds": SEEDS})
    write_json(ROOT / "development_candidates.json", {"protocol": TAG, "test_read": False, "candidates": candidates})
    write_json(ROOT / "strategy_freeze.json", {
        "protocol": TAG,
        "test_read": False,
        "r_grid": r_grid,
        "fixed_error_budgets": ERROR_BUDGETS,
        "configs": configs,
        "config_lookup": config_lookup,
        "parity_with_previous_caps": parity,
        "candidate_rule": "same empirical development cap selector as previous common-cap analysis; candidates with error_rate <= 0.20",
        "mixing": "only after test-label-aware retrospective diagnostic; random system-policy mixture, not deployment threshold",
        "inputs": m,
        "input_manifest_sha256": sha(ROOT / "input_manifest.json"),
    })
    write_json(ROOT / "FREEZE.json", {"protocol": TAG, "test_evaluated": False,
                                      "input_manifest_sha256": sha(ROOT / "input_manifest.json"),
                                      "strategy_freeze_sha256": sha(ROOT / "strategy_freeze.json"),
                                      "script_sha256": sha(Path(__file__)),
                                      "r_count": len(r_grid),
                                      "config_counts": {k: len(v) for k, v in configs.items()}})
    rows_out = []
    for method in METHODS:
        for c in configs[method]:
            for s in SEEDS:
                sel = c["selections"][str(s)]
                rows_out.append({"method": method, "config_id": c["config_id"], "first_r": c["first_r"], "last_r": c["last_r"], "seed": s,
                                 "normal": sel.get("normal"), "tail": sel.get("tail"), "released_dev": sel.get("released", 0),
                                 "correct_dev": sel.get("correct", 0), "errors_dev": sel.get("errors", 0), "error_rate_dev": sel.get("error_rate", 0.0),
                                 "status": sel.get("status")})
    write_csv(ROOT / "strategy_freeze.csv", rows_out)
    write_json(ROOT / "FREEZE_PARITY.json", {"all_old_caps_identical": True, "checks": parity})
    print(json.dumps({"r_grid": len(r_grid), "configs": {k: len(v) for k, v in configs.items()}, "test_evaluated": False}, indent=2))


def _target_scores(D: dict[str, np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred, base, dual, qc = inc._target_test_scores(D)
    assert pred.shape == base.shape == dual.shape == qc.shape == (20, 201)
    return pred.astype(int), base.astype(float), dual.astype(float), qc.astype(bool)


def _source_scores(direction: str, D: dict[str, np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return inc._source_test_scores(direction, D, idx)


def _apply_policy(pred: np.ndarray, base: np.ndarray, dual: np.ndarray, qc: np.ndarray, valid: np.ndarray | None,
                  selections: dict[str, Any]) -> np.ndarray:
    out = np.zeros((20, 201), dtype=bool)
    for j, s in enumerate(SEEDS):
        sel = selections[str(s)]
        n, t = sel.get("normal"), sel.get("tail")
        p = pred[j]
        finite_base = np.isfinite(base[j])
        finite_dual = np.isfinite(dual[j])
        if n is None:
            normal = np.zeros_like(p, dtype=bool)
        else:
            normal = (p == 0) & finite_dual & (dual[j] <= float(n))
            if valid is not None:
                normal &= valid[j]
        if t is None:
            tail = np.zeros_like(p, dtype=bool)
        else:
            tail = np.isin(p, [-1, 1]) & finite_base & (base[j] <= float(t))
        out[j] = (normal | tail) & qc[j]
    return out


def evaluate() -> None:
    fr = json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))
    assert fr["test_evaluated"] is False
    assert sha(ROOT / "input_manifest.json") == fr["input_manifest_sha256"]
    assert sha(ROOT / "strategy_freeze.json") == fr["strategy_freeze_sha256"]
    assert sha(Path(__file__)) == fr["script_sha256"]
    freeze_obj = json.loads((ROOT / "strategy_freeze.json").read_text(encoding="utf-8"))
    D = load_data(); idx = test_indices(); truth = inc.focused.ma_class(D["medical"][idx, 2]).astype(int)
    groups_raw = D["groups"][idx].astype(str); _, groups = np.unique(groups_raw, return_inverse=True)
    target = _target_scores(D, idx)
    sources = {d: _source_scores(d, D, idx) for d in DIRS}
    scores = {"target_only": target}
    for d in DIRS:
        pred, base, dual, valid, old = sources[d]
        with np.load(SOURCE_ACTIONS / f"{d}.npz", allow_pickle=False) as z:
            qc = np.tile(z["input_qc"].astype(bool)[None, :], (20, 1))
        scores[d] = (pred.astype(int), base.astype(float), dual.astype(float), qc, valid.astype(bool))
    arrays: dict[str, np.ndarray] = {"truth": truth.astype(np.int8), "group": groups.astype(np.int16)}
    config_map = {}
    for method in METHODS:
        config_map[method] = {}
        for c in freeze_obj["configs"][method]:
            pred, base, dual, qc, *rest = scores[method]
            valid = rest[0] if rest else None
            a = _apply_policy(pred, base, dual, qc, valid, c["selections"])
            safe = c["config_id"].replace("+", "plus")
            arrays[f"pred__{safe}"] = pred.astype(np.int8)
            arrays[f"action__{safe}"] = a.astype(np.uint8)
            config_map[method][c["config_id"]] = {"method": method, "config_id": c["config_id"], "first_r": c["first_r"], "last_r": c["last_r"]}
    np.savez_compressed(ROOT / "scores_actions_private.npz", **arrays)
    old_counts = {}
    old = json.loads((PREV / "validation.json").read_text(encoding="utf-8"))
    old_counts["previous_s14"] = old["s14_counts"]
    # Recompute S14 and all five cap points using the prior frozen actions as a strict guard.
    old_actions = {}
    for method in METHODS:
        if method == "target_only":
            pred, base, dual, qc = target
            valid = None
        else:
            pred, base, dual, qc, valid = scores[method]
        old_action_rows = []
        for s in SEEDS:
            if method == "target_only":
                m = joblib.load(FOCUSED / f"model_{s}.joblib")
                old_action_rows.append(inc.focused.action(m["threshold"], pred[s-1000], base[s-1000], dual[s-1000], qc[s-1000]) & qc[s-1000])
            else:
                with np.load(SOURCE_ACTIONS / f"{method}.npz", allow_pickle=False) as z:
                    old_action_rows.append(z["unified_action"].astype(bool)[s-1000] & qc[s-1000])
        old_actions[method] = np.asarray(old_action_rows)
    expected = {"target_only": (426, 23), "HZ_to_SA": (485, 51), "HZ+HB_to_SA": (620, 44)}
    s14_check = {}
    for method, a in old_actions.items():
        pred = scores[method][0]
        cc, ee = int((a & (pred == truth[None, :])).sum()), int((a & (pred != truth[None, :])).sum())
        s14_check[method] = {"correct": cc, "errors": ee, "expected": expected[method], "pass": (cc, ee) == expected[method]}
    assert all(v["pass"] for v in s14_check.values())
    write_json(ROOT / "EVALUATION_COMPLETE.json", {"protocol": TAG, "test_evaluated": True, "scores_file": "scores_actions_private.npz",
                                                    "s14_check": s14_check, "config_counts": {m: len(freeze_obj["configs"][m]) for m in METHODS},
                                                    "test_records": 201, "groups": int(len(np.unique(groups_raw))),
                                                    "qc_false_records": {m: int((~scores[m][3][0]).sum()) for m in METHODS},
                                                    "freeze_verified_before_test": True})
    write_json(ROOT / "config_map.json", config_map)
    print(json.dumps({"s14_check": s14_check, "config_counts": {m: len(freeze_obj["configs"][m]) for m in METHODS}}, indent=2))




# Clean delivery entry points. Deprecated D2/D3 functions are intentionally omitted.
def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "help"
    if stage == "freeze":
        freeze()
    elif stage == "evaluate":
        evaluate()
    else:
        raise SystemExit("Use only: freeze | evaluate. D2/D3 are in equal_error_postprocess.py.")


if __name__ == "__main__":
    main()
