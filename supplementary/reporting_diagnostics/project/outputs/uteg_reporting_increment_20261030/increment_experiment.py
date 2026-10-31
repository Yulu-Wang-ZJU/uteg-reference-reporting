from __future__ import annotations

"""Small, frozen, post-hoc reporting-increment experiment.

The script deliberately separates preparation/development from test evaluation.
It never writes to the historical experiment directories.  The test-set subset
is selected from group hashes before any test labels or model scores are used;
it is a sensitivity analysis of the old test cohort, not an independent test.
"""

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
OUT = BASE / "outputs"
DATA = OUT / "uteg_legacy_aligned_threecenter_20261022_155614" / "DATA15.npz"
FOCUSED = OUT / "uteg_nc_goal_20261030" / "focused_reporting_20261030"
OLD = OUT / "uteg_signal_model_goal_20261026_000929" / "ridge10_fullchain_20261026_122551"
SOURCE_OOF = OUT / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "UNIFIED_SELECTION_RECORDS.json"
SOURCE_FREEZE = OUT / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "THRESHOLDS_FREEZE.json"
SOURCE_ACTIONS = OUT / "uteg_ridge_dual_head_external_eval_20261026_191117" / "actions"
CONTRACT = OUT / "uteg_ridge_dual_head_prepare_20261026_172952" / "dual_head_contract.py"
SELECTOR = OUT / "uteg_ridge_dual_head_unified_selector_20261026_190142" / "run_unified_selector.py"
TARGET_RIDGE = OUT / "uteg_nc_goal_20261030" / "target_ridge" / "run.py"
FOCUSED_RUN = FOCUSED / "run.py"
OLD_RIDGE_RUN = OLD / "run_ridge10_fullchain.py"
DIRS = ["HZ_to_SA", "HZ+HB_to_SA"]
SEEDS = list(range(1000, 1020))
ALPHAS = [0.1, 1.0, 10.0, 100.0]
CAPS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_BOOT = 5000
PROTOCOL_TAG = "uteg-increment-20261030"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.write_text(json.dumps(canonical(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(canonical(rows))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import the historical pure functions without invoking its train/evaluate CLI.
focused = load_module("increment_focused_reporting", FOCUSED_RUN)


def record_key(global_index: int) -> str:
    return hashlib.sha256(f"{PROTOCOL_TAG}|record|{int(global_index)}".encode()).hexdigest()[:20]


def group_key(group: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_TAG}|group|{group}".encode()).hexdigest()[:24]


def threshold_key(v: float | None) -> tuple[int, float]:
    # None means branch disabled and is stricter than every observed score.
    return (0, -math.inf) if v is None else (1, float(v))


def threshold_json(v: float | None) -> float | None:
    return None if v is None else float(v)


def input_paths() -> list[Path]:
    paths = [DATA, SOURCE_OOF, SOURCE_FREEZE, CONTRACT, SELECTOR, TARGET_RIDGE, FOCUSED_RUN, OLD_RIDGE_RUN,
             ROOT / "EXPERIMENT_PLAN.md", ROOT / "EXECUTION_PROMPT.md", ROOT / "ACCEPTANCE_PROMPT.md",
             FOCUSED / "development_folds.csv", FOCUSED / "development_heads.csv", FOCUSED / "development_working_points.csv",
             BASE / "outputs/uteg_nc_goal_20261030/manuscript/main.pdf",
             BASE / "outputs/uteg_nc_goal_20261030/manuscript/supplement.pdf",
             BASE / "outputs/uteg_nc_goal_20261030/manuscript/Supplementary_Data_1_to_7.zip"]
    for d in DIRS:
        paths.append(SOURCE_ACTIONS / f"{d}.npz")
        for seed in SEEDS:
            paths.append(OLD / "models" / d / "target15" / f"SAHZU_CFULL_seed{seed}.joblib")
    for seed in SEEDS:
        paths.append(FOCUSED / f"model_{seed}.joblib")
    return paths


def manifest() -> dict[str, str]:
    paths = input_paths()
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input(s): " + "; ".join(missing))
    return {str(p): sha(p) for p in paths}


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def test_indices() -> np.ndarray:
    m = joblib.load(FOCUSED / "model_1000.joblib")
    idx = np.asarray(m["test"], dtype=int)
    assert len(idx) == 201 and len(np.unique(idx)) == 201
    return idx


def prepare() -> None:
    D = load_data()
    idx = test_indices()
    ids = D["ids"].astype(str)
    groups = D["groups"].astype(str)
    test_groups = sorted(set(groups[idx]))
    assert len(test_groups) == 190
    ranked = sorted(test_groups, key=lambda g: hashlib.sha256(f"{PROTOCOL_TAG}|{g}".encode()).hexdigest())
    selected = ranked[: math.ceil(0.30 * len(ranked))]
    selected_set = set(selected)
    subset_mask = np.asarray([groups[i] in selected_set for i in idx], dtype=bool)
    # 57 proxy groups contain 60 records in this test cohort; group integrity is primary.
    assert int(subset_mask.sum()) == 60
    comp_mask = ~subset_mask
    assert int(comp_mask.sum()) == 141
    source_ids = {}
    for d in DIRS:
        with np.load(SOURCE_ACTIONS / f"{d}.npz", allow_pickle=False) as z:
            source_ids[d] = z["test_ids"].astype(str)
            assert np.array_equal(source_ids[d], ids[idx])
    record_index = {str(ids[i]): int(i) for i in idx}
    assert len(record_index) == 201
    write_json(ROOT / "input_manifest.json", {"protocol": PROTOCOL_TAG, "inputs": manifest(), "test_records": 201,
                                               "test_proxy_groups": 190, "test_indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest()})
    write_json(ROOT / "subset_mask.json", {
        "protocol": PROTOCOL_TAG,
        "selection": "sort SHA256('uteg-increment-20261030|'+group_id), take first ceil(0.30*190)",
        "uses_labels_or_scores": False,
        "test_n": 201,
        "proxy_group_n": 190,
        "selected_proxy_group_n": 57,
        "selected_record_n": int(subset_mask.sum()),
        "complement_record_n": int(comp_mask.sum()),
        "selected_group_hashes": [group_key(g) for g in selected],
        "all_group_hashes_sorted": [group_key(g) for g in ranked],
        "record_mask_by_global_index": {str(int(i)): bool(v) for i, v in zip(idx, subset_mask)},
        "record_hash_by_global_index": {str(int(i)): record_key(int(i)) for i in idx},
        "selected_global_indices": [int(i) for i in idx[subset_mask]],
        "complement_global_indices": [int(i) for i in idx[comp_mask]],
        "id_alignment_sha256": {d: hashlib.sha256("\n".join(source_ids[d]).encode()).hexdigest() for d in DIRS},
    })
    write_json(ROOT / "PREPARE_COMPLETE.json", {"protocol": PROTOCOL_TAG, "test_evaluated": False,
                                                 "manifest_sha256": sha(ROOT / "input_manifest.json"),
                                                 "subset_sha256": sha(ROOT / "subset_mask.json")})
    print(json.dumps({"test_records": 201, "proxy_groups": 190, "subset_records": int(subset_mask.sum()),
                      "complement_records": int(comp_mask.sum()), "test_evaluated": False}, indent=2))


def _row(global_index: int, pred: int, truth: int, base_score: float | None, dual_score: float | None,
         groups: np.ndarray) -> dict[str, Any]:
    return {"global_index": int(global_index), "record_key": record_key(global_index),
            "group_key": group_key(str(groups[global_index])), "pred_class": int(pred),
            "truth_class": int(truth), "base_score": None if base_score is None else float(base_score),
            "dual_score": None if dual_score is None else float(dual_score)}


def source_oof_rows() -> dict[str, dict[int, list[dict[str, Any]]]]:
    D = load_data()
    ids = D["ids"].astype(str)
    id_to_idx = {x: i for i, x in enumerate(ids)}
    raw = json.loads(SOURCE_OOF.read_text(encoding="utf-8"))["records"]
    out: dict[str, dict[int, list[dict[str, Any]]]] = {d: {} for d in DIRS}
    for d in DIRS:
        for s in SEEDS:
            rs = [r for r in raw if r["direction"] == d and int(r["seed"]) == s]
            assert len(rs) == 60, (d, s, len(rs))
            rs = sorted(rs, key=lambda r: id_to_idx[str(r["record_id"])])
            rows = []
            for r in rs:
                i = id_to_idx[str(r["record_id"])]
                assert not bool(r.get("test_read", False))
                rows.append(_row(i, int(r["pred_class"]), int(r["truth_class"]),
                                 None if r.get("base_score") is None else float(r["base_score"]),
                                 None if r.get("dual_score") is None else float(r["dual_score"]), D["groups"]))
            assert len({r["global_index"] for r in rows}) == 60
            out[d][s] = rows
    return out


def _rebuild_target_seed(seed: int, D: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = joblib.load(OLD / "models" / DIRS[0] / "target15" / f"SAHZU_CFULL_seed{seed}.joblib")
    pool = np.asarray(source["pool"], dtype=int)
    lookup = {int(g): i for i, g in enumerate(pool)}
    x = D["x"][pool].copy(); y = D["medical"][pool].copy(); ids = D["ids"][pool].astype(str)
    qc = D["input_qc"][pool].copy()
    rows: list[dict[str, Any]] = []
    fit_count = {"ridge": 0, "error": 0, "dual": 0}
    fold_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for pos in range(3):
        part = source["positions"][pos]
        A = np.asarray([lookup[int(i)] for i in part["A"]], dtype=int)
        B = np.asarray([lookup[int(i)] for i in part["B"]], dtype=int)
        loc = {int(v): j for j, v in enumerate(A)}
        cross = {a: np.full((len(A), 4), np.nan) for a in ALPHAS}
        covered: list[int] = []
        for f in part["inner"]:
            T = np.asarray([lookup[int(i)] for i in f["T"]], dtype=int)
            V = np.asarray([lookup[int(i)] for i in f["V"]], dtype=int)
            covered.extend(V.tolist())
            assert set(T).isdisjoint(set(V)) and set(T) | set(V) == set(A)
            for alpha in ALPHAS:
                model = focused.ridge.fit(x[T], y[T], alpha)
                cross[alpha][[loc[int(v)] for v in V]] = focused.ridge.predict(model, x[V])
                fit_count["ridge"] += 1
        assert sorted(covered) == sorted(A.tolist())
        losses = {a: float(np.abs(cross[a][:, 2] - y[A, 2]).mean()) for a in ALPHAS}
        best = min(ALPHAS, key=lambda a: (losses[a], -a))
        med = cross[best]
        model = focused.ridge.fit(x[A], y[A], best)
        fit_count["ridge"] += 1
        pc = focused.ma_class(med[:, 2]); truth = focused.ma_class(y[A, 2])
        ym = y[A].mean(0); ys = np.maximum(y[A].std(0), 1e-6)
        gate = focused._fit_ma_gate(focused.ma_features(x[A], med, ym, ys), pc, truth,
                                    f"target_only_{seed}_{pos}", seed)
        fit_count["error"] += 1
        assert gate["success"] or gate["fallback"]
        heads = {}
        for head in ["low", "high"]:
            mask, hy = focused.c.build_head_mask(truth, pc, med[:, 2], qc[A], head)
            support = focused.c.support_summary(hy, ids[A][mask])
            head_rows.append({"seed": seed, "position": pos, "head": head, "n": int(support["n"]),
                              "class0": int(support["class0_n"]), "class1": int(support["class1_n"]),
                              "status": str(support["status"])})
            if support["status"] != "SUPPORTED":
                heads[head] = None
                continue
            z, pre = focused.c.build_features(x[A, 1][mask], med[:, 2][mask], np.ones(int(mask.sum()), bool))
            weights = focused.c.class_balanced_weights(hy)
            from scipy.optimize import minimize
            opt = minimize(focused.c.weighted_bce_objective, np.zeros(3), args=(z, hy, weights, .1),
                           jac=focused.c.weighted_bce_gradient, method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
            assert opt.success and np.isfinite(opt.x).all()
            fit_count["dual"] += 1
            heads[head] = {"theta": opt.x, "pre": pre, "support": support}
        fold_rows.append({"seed": seed, "position": pos, "training_n": len(A), "heldout_n": len(B),
                          "alpha": float(best), "inner_MA_MAE": float(losses[best]),
                          "error_fallback": bool(gate["fallback"])})
        pb = focused.ridge.predict(model, x[B]); cb = focused.ma_class(pb[:, 2])
        bs = focused.score_ma_gate(gate, focused.ma_features(x[B], pb, ym, ys), cb)
        ds = focused.head_scores(heads, x[B], pb)
        for j, i in enumerate(B):
            base = float(bs[j]) if qc[i] and np.isfinite(bs[j]) else None
            dual = float(ds[j]) if qc[i] and cb[j] == 0 and np.isfinite(ds[j]) else None
            rows.append(_row(int(pool[i]), int(cb[j]), int(focused.ma_class(y[i, 2])), base, dual, D["groups"]))
    assert len(rows) == 60 and len({r["global_index"] for r in rows}) == 60
    # This is the exact historical reconstruction check, not a new test fit.
    original = joblib.load(FOCUSED / f"model_{seed}.joblib")
    rows_for_old_choose = [{**r, "record_id": str(D["ids"][r["global_index"]])} for r in rows]
    assert focused.choose(rows_for_old_choose) == original["threshold"], seed
    return rows, {"seed": seed, **fit_count, "historical_threshold_identical": True,
                  "fold_rows": fold_rows, "head_rows": head_rows}


def target_oof_rows() -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    D = load_data()
    out = {}; fits = []
    all_folds = []; all_heads = []
    for seed in SEEDS:
        out[seed], stats = _rebuild_target_seed(seed, D)
        all_folds.extend(stats.pop("fold_rows")); all_heads.extend(stats.pop("head_rows"))
        fits.append(stats)
        print("rebuild target OOF", seed, stats, flush=True)
    assert all(len(out[s]) == 60 for s in SEEDS)
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    old_folds = [r for r in read_csv(FOCUSED / "development_folds.csv") if int(r["position"]) < 3]
    old_heads = [r for r in read_csv(FOCUSED / "development_heads.csv") if int(r["position"]) < 3]
    assert len(old_folds) == len(all_folds) == 60 and len(old_heads) == len(all_heads) == 120
    for a, b in zip(sorted(all_folds, key=lambda r: (r["seed"], r["position"])),
                    sorted(old_folds, key=lambda r: (int(r["seed"]), int(r["position"])) )):
        assert int(a["seed"]) == int(b["seed"]) and int(a["position"]) == int(b["position"])
        assert int(a["training_n"]) == int(b["training_n"]) and int(a["heldout_n"]) == int(b["heldout_n"])
        assert float(a["alpha"]) == float(b["alpha"]) and np.isclose(float(a["inner_MA_MAE"]), float(b["inner_MA_MAE"]), rtol=0, atol=1e-12)
        assert str(a["error_fallback"]).lower() == b["error_fallback"].lower()
    for a, b in zip(sorted(all_heads, key=lambda r: (r["seed"], r["position"], r["head"])),
                    sorted(old_heads, key=lambda r: (int(r["seed"]), int(r["position"]), r["head"]))):
        for k in ["seed", "position", "head", "n", "class0", "class1", "status"]:
            assert str(a[k]) == str(b[k]), (k, a, b)
    write_csv(ROOT / "target_development_folds.csv", all_folds)
    write_csv(ROOT / "target_development_heads.csv", all_heads)
    write_json(ROOT / "target_development_comparison.json", {"outer_positions_checked": [0, 1, 2],
                                                               "fold_rows": len(all_folds), "head_rows": len(all_heads),
                                                               "historical_folds_identical": True, "historical_heads_identical": True,
                                                               "historical_working_point_thresholds_checked": True})
    return out, fits


def _policy_action(rows: list[dict[str, Any]], normal: float | None, tail: float | None) -> np.ndarray:
    out = np.zeros(len(rows), dtype=bool)
    for j, r in enumerate(rows):
        p = int(r["pred_class"]); b = r["base_score"]; d = r["dual_score"]
        if p == 0 and normal is not None and d is not None and np.isfinite(d) and d <= normal:
            out[j] = True
        if p in (-1, 1) and tail is not None and b is not None and np.isfinite(b) and b <= tail:
            out[j] = True
    return out


def _policy_stats(rows: list[dict[str, Any]], action: np.ndarray) -> dict[str, Any]:
    p = np.asarray([r["pred_class"] for r in rows]); t = np.asarray([r["truth_class"] for r in rows])
    correct = action & (p == t); within = action & (p == 0)
    return {"n": int(len(rows)), "released": int(action.sum()), "correct": int(correct.sum()),
            "errors": int((action & (p != t)).sum()), "error_rate": float((action & (p != t)).sum() / action.sum()) if action.sum() else None,
            "coverage": float(action.mean()), "correct_within": int((within & (t == 0)).sum()),
            "low_to_within": int((within & (t == -1)).sum()), "high_to_within": int((within & (t == 1)).sum()),
            "released_distinct": int(len({r["global_index"] for r, a in zip(rows, action) if a})),
            "correct_within_distinct": int(len({r["global_index"] for r, a in zip(rows, action) if a and r["pred_class"] == 0 and r["truth_class"] == 0}))}


def select_cap(rows: list[dict[str, Any]], cap: float) -> dict[str, Any]:
    normal_values = [None] + sorted({float(r["dual_score"]) for r in rows if int(r["pred_class"]) == 0 and r["dual_score"] is not None and np.isfinite(r["dual_score"])})
    tail_values = [None] + sorted({float(r["base_score"]) for r in rows if int(r["pred_class"]) in (-1, 1) and r["base_score"] is not None and np.isfinite(r["base_score"])})
    truth = [int(r["truth_class"]) for r in rows]
    has_low = -1 in truth; has_high = 1 in truth
    candidates = []
    for normal in normal_values:
        for tail in tail_values:
            action = _policy_action(rows, normal, tail)
            stats = _policy_stats(rows, action)
            if stats["released"] < 5:
                continue
            if normal is not None and (not has_low or not has_high or stats["correct_within"] < 5):
                continue
            if stats["errors"] / stats["released"] > cap + 1e-12:
                continue
            # Maximize release; then minimize errors; then choose stricter thresholds.
            key = (-stats["released"], stats["errors"], threshold_key(normal), threshold_key(tail))
            candidates.append((key, normal, tail, stats))
    if not candidates:
        chosen = {"normal": None, "tail": None, "all_withhold": True}
        stats = _policy_stats(rows, np.zeros(len(rows), bool))
        status = "NO_FEASIBLE_PAIR"
    else:
        _, normal, tail, stats = min(candidates, key=lambda x: x[0])
        chosen = {"normal": threshold_json(normal), "tail": threshold_json(tail), "all_withhold": False}
        status = "SELECTED"
    return {**chosen, "cap": float(cap), "status": status, "candidate_count": int(len(normal_values) * len(tail_values)),
            "eligible_pair_count": int(len(candidates)), "selected_metrics": stats,
            "support": {"has_low": bool(has_low), "has_high": bool(has_high), "min_records": 5,
                         "within_requires_correct": 5, "error_cap_is_empirical_development_only": True}}


def anonymize_rows(rows_by_method: dict[str, dict[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    out = []
    for method, by_seed in rows_by_method.items():
        for seed, rows in by_seed.items():
            for r in rows:
                out.append({"method": method, "seed": int(seed), **{k: r[k] for k in ["global_index", "record_key", "group_key", "pred_class", "truth_class", "base_score", "dual_score"]}})
    return out


def develop() -> None:
    assert (ROOT / "PREPARE_COMPLETE.json").exists()
    assert not (ROOT / "EVALUATION_COMPLETE.json").exists()
    m = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    assert m["inputs"] == manifest()
    source = source_oof_rows()
    target, fit_stats = target_oof_rows()
    rows = {"target_only": target, **source}
    write_json(ROOT / "development_oof_private.json", {"protocol": PROTOCOL_TAG, "test_read": False,
                                                         "rows": anonymize_rows(rows), "fit_stats": fit_stats})
    thresholds = []
    for method in ["target_only", *DIRS]:
        for seed in SEEDS:
            for cap in CAPS:
                entry = select_cap(rows[method][seed], cap)
                thresholds.append({"method": method, "seed": int(seed), "cap": float(cap), **entry})
    assert len(thresholds) == 3 * 20 * len(CAPS)
    freeze = {"protocol": PROTOCOL_TAG, "description": "new common empirical development error-cap policy; separate from S14", "test_read": False,
              "caps": CAPS, "thresholds": thresholds, "inputs": m["inputs"], "input_manifest_sha256": sha(ROOT / "input_manifest.json"),
              "target_oof_reconstruction": {"outer_positions": [0, 1, 2], "inner_alphas": ALPHAS, "final_models_retrained": False,
                                             "source_models_retrained": False, "historical_thresholds_reproduced": True}}
    write_json(ROOT / "frozen_thresholds.json", freeze)
    write_json(ROOT / "FREEZE.json", {"protocol": PROTOCOL_TAG, "test_evaluated": False,
                                      "frozen_thresholds_sha256": sha(ROOT / "frozen_thresholds.json"),
                                      "input_manifest_sha256": sha(ROOT / "input_manifest.json"),
                                      "script_sha256": sha(Path(__file__)), "threshold_count": len(thresholds)})
    write_csv(ROOT / "development_working_points.csv", [
        {"method": e["method"], "seed": e["seed"], "cap": e["cap"], "normal": e["normal"], "tail": e["tail"],
         "status": e["status"], "released": e["selected_metrics"]["released"], "correct": e["selected_metrics"]["correct"],
         "errors": e["selected_metrics"]["errors"], "error_rate": e["selected_metrics"]["error_rate"],
         "correct_within": e["selected_metrics"]["correct_within"], "low_to_within": e["selected_metrics"]["low_to_within"],
         "high_to_within": e["selected_metrics"]["high_to_within"]} for e in thresholds])
    print(json.dumps({"target_oof_rows": 20 * 60, "source_oof_rows_each": 20 * 60, "frozen_thresholds": len(thresholds), "test_evaluated": False}, indent=2))


def _target_test_scores(D: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = test_indices(); pred = []; base = []; dual = []; qc = np.tile(D["input_qc"][idx][None, :], (len(SEEDS), 1))
    for s in SEEDS:
        m = joblib.load(FOCUSED / f"model_{s}.joblib")
        assert np.array_equal(np.asarray(m["test"], dtype=int), idx)
        med = focused.ridge.predict(m["predictor"], D["x"][idx])
        pc = focused.ma_class(med[:, 2])
        bs = focused.score_ma_gate(m["error"], focused.ma_features(D["x"][idx], med, m["ym"], m["ys"]), pc)
        ds = focused.head_scores(m["heads"], D["x"][idx], med)
        pred.append(pc); base.append(bs); dual.append(ds)
    return np.asarray(pred), np.asarray(base), np.asarray(dual), qc


def _source_test_scores(direction: str, D: dict[str, np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SOURCE_ACTIONS / f"{direction}.npz", allow_pickle=False) as z:
        ids = z["test_ids"].astype(str); expected = D["ids"][idx].astype(str)
        assert np.array_equal(ids, expected)
        pred = z["final_ma_class"].astype(int); base = z["lr_score"].astype(float); dual = z["dual_score"].astype(float)
        valid = z["dual_valid"].astype(bool); qc = np.tile(z["input_qc"].astype(bool)[None, :], (20, 1))
        old_action = z["unified_action"].astype(bool)
        assert pred.shape == base.shape == dual.shape == (20, 201)
    return pred, base, dual, valid, old_action


def _threshold_lookup(freeze: dict[str, Any], method: str, seed: int, cap: float) -> dict[str, Any]:
    for e in freeze["thresholds"]:
        if e["method"] == method and int(e["seed"]) == seed and abs(float(e["cap"]) - cap) < 1e-12:
            return e
    raise KeyError((method, seed, cap))


def evaluate() -> None:
    assert (ROOT / "FREEZE.json").exists() and not (ROOT / "EVALUATION_COMPLETE.json").exists()
    prep = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    assert prep["inputs"] == manifest()
    freeze = json.loads((ROOT / "frozen_thresholds.json").read_text(encoding="utf-8"))
    freeze_record = json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["test_read"] is False and freeze_record["test_evaluated"] is False
    assert sha(ROOT / "input_manifest.json") == freeze_record["input_manifest_sha256"]
    assert sha(ROOT / "frozen_thresholds.json") == freeze_record["frozen_thresholds_sha256"]
    assert sha(Path(__file__)) == freeze_record["script_sha256"]
    D = load_data(); idx = test_indices(); truth = focused.ma_class(D["medical"][idx, 2]).astype(int)
    groups = D["groups"][idx].astype(str)
    target_pred, target_base, target_dual, target_qc = _target_test_scores(D)
    source_scores = {d: _source_test_scores(d, D, idx) for d in DIRS}
    arms: dict[str, dict[str, np.ndarray]] = {}
    # E0 fixed S14 actions.  These are copied from the historical frozen outputs.
    target_s14_action = []
    for j, s in enumerate(SEEDS):
        m = joblib.load(FOCUSED / f"model_{s}.joblib")
        target_s14_action.append(focused.action(m["threshold"], target_pred[j], target_base[j], target_dual[j], target_qc[j]))
    arms["Target-only Ridge|S14"] = {"pred": target_pred, "action": np.asarray(target_s14_action) & target_qc}
    for d in DIRS:
        pred, base, dual, valid, old_action = source_scores[d]
        source_qc = np.tile(np.load(SOURCE_ACTIONS / f"{d}.npz", allow_pickle=False)["input_qc"].astype(bool)[None, :], (20, 1))
        arms[f"{d}|S14"] = {"pred": pred, "action": old_action & source_qc}
    # E1 frozen development thresholds.
    for cap in CAPS:
        acts = []
        for j, s in enumerate(SEEDS):
            e = _threshold_lookup(freeze, "target_only", s, cap)
            acts.append(_policy_action([{"pred_class": int(target_pred[j, k]), "base_score": float(target_base[j, k]) if np.isfinite(target_base[j, k]) else None,
                                        "dual_score": float(target_dual[j, k]) if np.isfinite(target_dual[j, k]) else None} for k in range(201)], e["normal"], e["tail"]) & target_qc[j])
        arms[f"target_only|cap={cap:.2f}"] = {"pred": target_pred, "action": np.asarray(acts)}
        for d in DIRS:
            pred, base, dual, valid, _ = source_scores[d]; acts = []
            source_qc = np.tile(np.load(SOURCE_ACTIONS / f"{d}.npz", allow_pickle=False)["input_qc"].astype(bool)[None, :], (20, 1))
            for j, s in enumerate(SEEDS):
                e = _threshold_lookup(freeze, d, s, cap)
                rr = [{"pred_class": int(pred[j, k]), "base_score": float(base[j, k]) if np.isfinite(base[j, k]) else None,
                       "dual_score": float(dual[j, k]) if bool(valid[j, k]) and np.isfinite(dual[j, k]) else None} for k in range(201)]
                acts.append(_policy_action(rr, e["normal"], e["tail"]) & source_qc[j])
            arms[f"{d}|cap={cap:.2f}"] = {"pred": pred, "action": np.asarray(acts)}
    # All score/action arrays are private local artifacts; no patient identifiers are written.
    arrays = {"truth": truth, "group_global": np.unique(groups, return_inverse=True)[1], "group_count": np.asarray([len(np.unique(groups))])}
    for name, arm in arms.items():
        safe = name.replace("|", "__").replace("+", "plus").replace("=", "_").replace(".", "p")
        arrays[f"pred__{safe}"] = arm["pred"].astype(np.int8)
        arrays[f"action__{safe}"] = arm["action"].astype(np.uint8)
    np.savez_compressed(ROOT / "evaluation_private.npz", **arrays)
    write_json(ROOT / "arm_name_map.json", {name.replace("|", "__").replace("+", "plus").replace("=", "_").replace(".", "p"): name for name in arms})
    e0 = {}
    for name in ["Target-only Ridge|S14", "HZ_to_SA|S14", "HZ+HB_to_SA|S14"]:
        p = arms[name]["pred"]; a = arms[name]["action"]; e0[name] = {"released": int(a.sum()), "correct": int((a & (p == truth)).sum()), "errors": int((a & (p != truth)).sum())}
    assert e0["Target-only Ridge|S14"] == {"released": 449, "correct": 426, "errors": 23}
    assert e0["HZ_to_SA|S14"] == {"released": 536, "correct": 485, "errors": 51}
    assert e0["HZ+HB_to_SA|S14"] == {"released": 664, "correct": 620, "errors": 44}
    write_json(ROOT / "EVALUATION_COMPLETE.json", {"protocol": PROTOCOL_TAG, "test_evaluated": True,
                                                     "freeze_verified_before_test": True, "test_records": 201,
                                                     "proxy_groups": int(len(np.unique(groups))), "e0_reproduction": e0,
                                                     "private_array_file": "evaluation_private.npz",
                                                     "private_array_sha256": sha(ROOT / "evaluation_private.npz"),
                                                     "target_qc_all_true": bool(target_qc.all()),
                                                     "source_qc_all_true": {d: bool(np.tile(np.load(SOURCE_ACTIONS / f"{d}.npz", allow_pickle=False)["input_qc"].astype(bool)[None, :], (20, 1)).all()) for d in DIRS}})
    print(json.dumps(e0, indent=2))


def _arm_stats(pred: np.ndarray, action: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    p = pred[:, mask]; a = action[:, mask]; t = truth[mask][None, :]
    correct = a & (p == t); incorrect = a & (p != t); unreported = ~a; within = a & (p == 0)
    out = {"records": int(mask.sum()), "record_seed_n": int(a.size), "correct_yield": int(correct.sum()), "incorrect_yield": int(incorrect.sum()),
           "unreported": int(unreported.sum()), "coverage": float(a.mean()), "agreement": float(correct.sum() / a.sum()) if a.sum() else None,
           "low_to_within": int((within & (t == -1)).sum()), "high_to_within": int((within & (t == 1)).sum()),
           "correct_low": int((a & (p == -1) & (t == -1)).sum()), "correct_high": int((a & (p == 1) & (t == 1)).sum()),
           "correct_within": int((a & (p == 0) & (t == 0)).sum()), "zero_report_seeds": int((a.sum(axis=1) == 0).sum()),
           "within_reports": int(within.sum())}
    assert out["correct_yield"] + out["incorrect_yield"] + out["unreported"] == out["record_seed_n"]
    return out


def _bootstrap_difference(pred_a: np.ndarray, act_a: np.ndarray, pred_b: np.ndarray, act_b: np.ndarray,
                          truth: np.ndarray, groups: np.ndarray, mask: np.ndarray, unit: str, rng_seed: int) -> dict[str, Any]:
    # A paired bootstrap keeps all 20 seeds and resamples only records or proxy groups.
    rec = np.where(mask)[0]
    unique_groups = np.unique(groups[rec])
    rng = np.random.default_rng(rng_seed)
    samples = []
    if unit == "proxy_group":
        for _ in range(N_BOOT):
            picks = rng.integers(0, len(unique_groups), size=len(unique_groups))
            weights = np.bincount(picks, minlength=len(unique_groups))
            rec_weights = np.zeros(len(groups), dtype=float)
            for g, w in zip(unique_groups, weights): rec_weights[groups == g] = w
            samples.append(rec_weights)
    else:
        for _ in range(N_BOOT):
            picks = rng.integers(0, len(rec), size=len(rec))
            weights = np.bincount(picks, minlength=len(rec))
            rw = np.zeros(len(groups), dtype=float); rw[rec] = weights
            samples.append(rw)
    samples = np.asarray(samples)
    t = truth[None, :]
    ca = act_a & (pred_a == t); cb = act_b & (pred_b == t)
    ea = act_a & (pred_a != t); eb = act_b & (pred_b != t)
    ua = ~act_a; ub = ~act_b
    result = {}
    for key, aa, bb in [("correct_yield", ca, cb), ("incorrect_yield", ea, eb), ("unreported", ua, ub)]:
        diff = np.einsum("sr,br->b", (bb.astype(float) - aa.astype(float)), samples)
        denom = samples[:, mask].sum(axis=1) * len(SEEDS)
        ydiff = diff / denom
        count_stat = {"point": float((bb[:, mask].sum() - aa[:, mask].sum())), "bootstrap_mean": float(diff.mean()),
                      "ci95_low": float(np.percentile(diff, 2.5)), "ci95_high": float(np.percentile(diff, 97.5)),
                      "unit": "record-seed-count", "n_boot": N_BOOT}
        yield_stat = {"point": float((bb[:, mask].sum() - aa[:, mask].sum()) / (mask.sum() * len(SEEDS))),
                      "bootstrap_mean": float(ydiff.mean()), "ci95_low": float(np.percentile(ydiff, 2.5)),
                      "ci95_high": float(np.percentile(ydiff, 97.5)), "point_percent": float(100 * (bb[:, mask].sum() - aa[:, mask].sum()) / (mask.sum() * len(SEEDS))),
                      "ci95_low_percent": float(100 * np.percentile(ydiff, 2.5)), "ci95_high_percent": float(100 * np.percentile(ydiff, 97.5)),
                      "unit": "record-seed-fraction (percent fields are percentage points)", "n_boot": N_BOOT}
        result[key] = {"count_delta": count_stat, "yield_delta": yield_stat}
    # Descriptive coverage and agreement differences use the same paired resamples.
    # Each bootstrap replicate has its own sampled-record multiplicity.
    denom = samples[:, mask].sum(axis=1) * len(SEEDS)
    covdiff = np.einsum("sr,br->b", (act_b.astype(float) - act_a.astype(float)), samples) / denom
    result["coverage"] = {"point": float((act_b[:, mask].mean() - act_a[:, mask].mean())), "bootstrap_mean": float(covdiff.mean()),
                           "ci95_low": float(np.percentile(covdiff, 2.5)), "ci95_high": float(np.percentile(covdiff, 97.5)), "unit": unit, "n_boot": N_BOOT}
    result["note"] = "Conditional descriptive paired interval; the error interval crossing zero is not a non-inferiority result."
    return result


def summarize() -> None:
    assert (ROOT / "EVALUATION_COMPLETE.json").exists()
    D = load_data(); idx = test_indices(); groups_raw = D["groups"][idx].astype(str); group_values, group_codes = np.unique(groups_raw, return_inverse=True)
    subset = json.loads((ROOT / "subset_mask.json").read_text(encoding="utf-8")); selected_idx = set(subset["selected_global_indices"])
    masks = {"full201": np.ones(201, bool), "subset57groups": np.asarray([int(i) in selected_idx for i in idx]), "complement133groups": np.asarray([int(i) not in selected_idx for i in idx])}
    z = np.load(ROOT / "evaluation_private.npz", allow_pickle=False)
    arm_map = json.loads((ROOT / "arm_name_map.json").read_text(encoding="utf-8"))
    def safe(name: str) -> str:
        return name.replace("|", "__").replace("+", "plus").replace("=", "_").replace(".", "p")
    results = []; count_rows = []; dest_rows = []
    truth = focused.ma_class(D["medical"][idx, 2]).astype(int)
    for name in sorted(arm_map):
        display = arm_map[name]
        pred = z["pred__" + name].astype(int); act = z["action__" + name].astype(bool)
        if display.endswith("|S14"):
            method, protocol = display.rsplit("|", 1); cap = None
        else:
            method, capstr = display.rsplit("|cap=", 1); protocol = "empirical_cap"; cap = float(capstr)
        for subset_name, mask in masks.items():
            st = _arm_stats(pred, act, truth, mask)
            row = {"arm": display, "method": method, "protocol": protocol, "cap": cap, "subset": subset_name, **st}
            count_rows.append(row)
            for true_state in [-1, 0, 1]:
                for report_state in [-1, 0, 1, 9]:
                    report = np.where(act, pred, 9)
                    n = int(((report[:, mask] == report_state) & (truth[mask][None, :] == true_state)).sum())
                    dest_rows.append({"arm": display, "method": method, "protocol": protocol, "cap": cap, "subset": subset_name,
                                      "true_state": true_state, "report_state": report_state, "n": n,
                                      "N": int(mask.sum() * len(SEEDS)), "proportion": n / int(mask.sum() * len(SEEDS))})
            results.append(row)
    write_csv(ROOT / "results.csv", count_rows)
    write_csv(ROOT / "class_destinations.csv", dest_rows)
    # Per-seed table is full-cohort only, retaining all seeds including zero-report seeds.
    per_seed = []
    for name in sorted(arm_map):
        display = arm_map[name]
        pred = z["pred__" + name].astype(int); act = z["action__" + name].astype(bool)
        for j, seed in enumerate(SEEDS):
            a = act[j]; p = pred[j]; t = truth
            per_seed.append({"arm": display, "seed": seed, "released": int(a.sum()), "correct": int((a & (p == t)).sum()),
                             "incorrect": int((a & (p != t)).sum()), "unreported": int((~a).sum()),
                             "agreement": float((a & (p == t)).sum() / a.sum()) if a.sum() else None})
    write_csv(ROOT / "per_seed.csv", per_seed)
    # Paired primary and secondary contrasts for every pre-specified cap and subset.
    pair_rows = {}
    for cap in CAPS:
        for subset_name, mask in masks.items():
            for direction in DIRS:
                target = z["pred__" + safe(f"target_only|cap={cap:.2f}")].astype(int); target_a = z["action__" + safe(f"target_only|cap={cap:.2f}")].astype(bool)
                src = z["pred__" + safe(f"{direction}|cap={cap:.2f}")].astype(int); src_a = z["action__" + safe(f"{direction}|cap={cap:.2f}")].astype(bool)
                key = f"{direction}|cap={cap:.2f}|{subset_name}"
                pair_rows[key] = {"comparison": f"{direction} minus target_only", "cap": cap, "subset": subset_name,
                                  "group_bootstrap": _bootstrap_difference(target, target_a, src, src_a, truth, group_codes, mask, "proxy_group", 20261030 + int(cap * 1000) + len(subset_name)),
                                  "record_bootstrap": _bootstrap_difference(target, target_a, src, src_a, truth, group_codes, mask, "record", 20301030 + int(cap * 1000) + len(subset_name))}
    write_json(ROOT / "paired_contrasts.json", {"protocol": PROTOCOL_TAG, "n_boot": N_BOOT, "comparisons": pair_rows,
                                                 "primary": "HZ+HB_to_SA|cap=0.15|full201", "secondary": "HZ_to_SA|cap=0.15|full201",
                                                 "interpretation": "Intervals are conditional descriptive paired intervals; no non-inferiority claim."})
    validation = {
        "s14_counts": {r["arm"]: {k: r[k] for k in ["correct_yield", "incorrect_yield", "unreported", "coverage", "agreement"]} for r in count_rows if r["subset"] == "full201" and r["protocol"] == "S14"},
        "s14_expected": {"Target-only Ridge|S14": {"correct_yield": 426, "incorrect_yield": 23}, "HZ_to_SA|S14": {"correct_yield": 485, "incorrect_yield": 51}, "HZ+HB_to_SA|S14": {"correct_yield": 620, "incorrect_yield": 44}},
        "s14_reproduced": True, "subset_selected_without_labels_or_scores": True, "subset_records": int(masks["subset57groups"].sum()),
        "complement_records": int(masks["complement133groups"].sum()), "all_caps_prespecified": CAPS,
        "all_seeds_retained": True, "bootstrap_unit_keeps_all_seeds": True, "unsupported_fail_closed_checked": True,
        "freeze_before_evaluation": True, "test_label_used_only_after_freeze": True,
        "independent_validation": False, "old_test_cohort_reused": True,
        "input_manifest_unchanged": json.loads((ROOT / "input_manifest.json").read_text(encoding="utf-8"))["inputs"] == manifest(),
        "freeze_input_sha_verified": sha(ROOT / "input_manifest.json") == json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))["input_manifest_sha256"],
        "freeze_threshold_sha_verified": sha(ROOT / "frozen_thresholds.json") == json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))["frozen_thresholds_sha256"],
        "freeze_script_sha_verified": sha(Path(__file__)) == json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))["script_sha256"],
        "target_development_folds_heads_identical": json.loads((ROOT / "target_development_comparison.json").read_text(encoding="utf-8"))["historical_folds_identical"] and json.loads((ROOT / "target_development_comparison.json").read_text(encoding="utf-8"))["historical_heads_identical"],
    }
    write_json(ROOT / "validation.json", validation)
    # A compact, non-identifying Chinese handoff for the parent.
    main = [r for r in count_rows if r["subset"] == "full201" and r["protocol"] == "empirical_cap" and abs(float(r["cap"]) - .15) < 1e-12]
    def find(arm): return next(r for r in main if r["arm"] == arm)
    zh = ["# 本轮增量实验结果（仅新分析）", "", "本轮为 post-hoc 探索性补充；固定测试队列的 57/190 proxy-group 子集只作敏感性复核，不是独立验证。", "",
          "## S14 复现", "", "- target-only：426 正确、23 错误、3571 未报告。", "- HZ→SA：485 正确、51 错误、3484 未报告。", "- HZ+HB→SA：620 正确、44 错误、3356 未报告。", ""]
    zh += ["## 主预设 cap=0.15（full201）", ""]
    for arm in ["target_only|cap=0.15", "HZ_to_SA|cap=0.15", "HZ+HB_to_SA|cap=0.15"]:
        r = find(arm); zh.append(f"- {arm}：正确 {r['correct_yield']}，错误 {r['incorrect_yield']}，未报告 {r['unreported']}，coverage {r['coverage']:.4f}，agreement {r['agreement'] if r['agreement'] is not None else 'NA'}。")
    zh += ["", "## 如何解释", "", "共同开发端经验错误约束检验的是：在同一报告规则下，双源系统能否带来更多报告。测试端错误若增加，必须按 trade-off 报告；错误差区间跨零不等于安全性不劣。当前结果仍不能替代真正未接触开发判断的独立病例验证，也不自动保证接收或六分。", "本轮仍只分析 15 min；没有新增报告时间（例如 5/10 min）的时序验证。", ""]
    (ROOT / "RESULTS_ZH.md").write_text("\n".join(zh), encoding="utf-8")
    print(json.dumps({"rows": len(count_rows), "paired_comparisons": len(pair_rows), "validation": validation}, ensure_ascii=False, indent=2))


def plot() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(ROOT / "results.csv")
    full = df[df["subset"] == "full201"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    styles = {"target_only": ("#555555", "o"), "HZ_to_SA": ("#2b6cb0", "s"), "HZ+HB_to_SA": ("#c53030", "^"),
              "Target-only Ridge": ("#555555", "o"), "HZ_to_SA": ("#2b6cb0", "s"), "HZ+HB_to_SA": ("#c53030", "^")}
    for method in ["target_only", "HZ_to_SA", "HZ+HB_to_SA"]:
        sub = full[(full["method"] == method) & (full["protocol"] == "empirical_cap")]
        color, marker = styles[method]
        ax.plot(sub["incorrect_yield"], sub["correct_yield"], marker=marker, color=color, linewidth=1.2, label=f"{method} (new cap)")
        for _, r in sub.iterrows(): ax.annotate(f"{r['cap']:.2f}", (r["incorrect_yield"], r["correct_yield"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for method in ["Target-only Ridge", "HZ_to_SA", "HZ+HB_to_SA"]:
        r = full[(full["method"] == method) & (full["protocol"] == "S14")].iloc[0]
        color, marker = styles[method]; ax.scatter([r["incorrect_yield"]], [r["correct_yield"]], facecolors="none", edgecolors=color, marker=marker, s=60, label=f"{method} (S14)")
    ax.set_xlabel("Incorrect reports (record–seed count)"); ax.set_ylabel("Correct reports (record–seed count)")
    ax.set_title("Fixed development-cap policies on the existing test cohort")
    ax.grid(alpha=.22); ax.legend(fontsize=7, frameon=False, ncol=2); fig.tight_layout()
    fig.savefig(ROOT / "increment_yield_tradeoff.png", bbox_inches="tight"); fig.savefig(ROOT / "increment_yield_tradeoff.pdf", bbox_inches="tight"); plt.close(fig)
    write_json(ROOT / "PLOT_COMPLETE.json", {"protocol": PROTOCOL_TAG, "source": "results.csv", "test_optimal_point_selection": False,
                                              "figure": ["increment_yield_tradeoff.png", "increment_yield_tradeoff.pdf"]})


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "develop", "evaluate", "summarize", "plot"}:
        raise SystemExit("usage: increment_experiment.py prepare|develop|evaluate|summarize|plot")
    {"prepare": prepare, "develop": develop, "evaluate": evaluate, "summarize": summarize, "plot": plot}[sys.argv[1]]()


if __name__ == "__main__":
    main()
