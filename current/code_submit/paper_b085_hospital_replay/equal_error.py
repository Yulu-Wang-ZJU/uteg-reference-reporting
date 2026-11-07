"""Isolated equal-error-burden diagnostics for the revised-B paper replay.

This file deliberately does not modify the historical equal-error run.  It
recreates the development-side policy rows from the revised-B target-only
models and reads the newly frozen transfer rows, then evaluates the same
retrospective policy family on the 201-record test cohort.  The D2 frontier
and D3 five-fold cross-calibration are exploratory, test-label-aware
diagnostics; they are not deployment policies or independent validation.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_B085_hospital_replay_20261102/equal_error"
DATA = ROOT / "outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz"
OLD = ROOT / "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551"
FOCUSED = ROOT / "outputs/uteg_nc_goal_20261030/focused_reporting_20261030/run.py"
TARGET_OUT = ROOT / "outputs/paper_B085_hospital_replay_20261102/target_only"
TRANSFER_RUNS = ROOT / "outputs/paper_B085_hospital_replay_20261102/transfer/runs"
TRANSFER_PREDS = ROOT / "outputs/paper_B085_hospital_replay_20261102/transfer/predictions"
SEEDS = tuple(range(1000, 1020))
METHODS = ("target_only", "HZ_to_SA", "HZ+HB_to_SA")
DIRS = METHODS[1:]
FIXED_CAPS = (0.0, 0.05, 0.10, 0.15, 0.20)
ERROR_BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.075, 0.10)
N_BOOT = 5000
TAG = "uteg-equal-error-paper-B085-replay-20261102"
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k): canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [canon(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    assert path.resolve().is_relative_to(OUT.resolve())
    path.write_text(json.dumps(canon(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert path.resolve().is_relative_to(OUT.resolve())
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(canon(rows))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# The focused runner exposes the exact MA-state, gate and head contracts used
# by the frozen target-only replay.  Importing it does not run its CLI.
target_mod = load_module("paper_b085_target_only_for_equal_error", ROOT / "code_submit/paper_b085_hospital_replay/target_only.py")
fr = target_mod.fr


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    assert d["x"].shape == (480, 4) and bool(d["B_from_original_data"])
    return d


def original_model(direction: str, seed: int) -> Path:
    return OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"


def _threshold_key(v: float | None) -> tuple[int, float]:
    return (0, -math.inf) if v is None else (1, float(v))


def policy_action(rows: list[dict[str, Any]], normal: float | None, tail: float | None) -> np.ndarray:
    out = np.zeros(len(rows), dtype=bool)
    for j, r in enumerate(rows):
        p = int(r["pred_class"]); b = r.get("base_score"); d = r.get("dual_score")
        if p == 0 and normal is not None and d is not None and np.isfinite(float(d)) and float(d) <= normal:
            out[j] = True
        if p in (-1, 1) and tail is not None and b is not None and np.isfinite(float(b)) and float(b) <= tail:
            out[j] = True
    return out


def stats(rows: list[dict[str, Any]], action: np.ndarray) -> dict[str, Any]:
    p = np.asarray([int(r["pred_class"]) for r in rows]); t = np.asarray([int(r["truth_class"]) for r in rows])
    within = action & (p == 0)
    rel = int(action.sum()); err = int((action & (p != t)).sum())
    return dict(released=rel, errors=err, correct=int(rel - err), error_rate=float(err / rel) if rel else 0.0,
                correct_within=int((within & (t == 0)).sum()), low_to_within=int((within & (t == -1)).sum()),
                high_to_within=int((within & (t == 1)).sum()))


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal = [None] + sorted({float(r["dual_score"]) for r in rows if int(r["pred_class"]) == 0 and r.get("dual_score") is not None and np.isfinite(float(r["dual_score"]))})
    tail = [None] + sorted({float(r["base_score"]) for r in rows if int(r["pred_class"]) in (-1, 1) and r.get("base_score") is not None and np.isfinite(float(r["base_score"]))})
    truth_states = {int(r["truth_class"]) for r in rows}; out = []
    for n in normal:
        for t in tail:
            st = stats(rows, policy_action(rows, n, t))
            if st["released"] < 5: continue
            if n is not None and (not {-1, 1}.issubset(truth_states) or st["correct_within"] < 5): continue
            out.append(dict(normal=n, tail=t, **st, eligible=True))
    return out


def select_at_cap(candidates: list[dict[str, Any]], cap: float) -> dict[str, Any]:
    eligible = [c for c in candidates if c["error_rate"] <= cap + 1e-12]
    if not eligible:
        return dict(normal=None, tail=None, released=0, errors=0, correct=0, error_rate=0.0, status="NO_FEASIBLE_PAIR")
    best = min(eligible, key=lambda c: (-c["released"], c["errors"], _threshold_key(c["normal"]), _threshold_key(c["tail"])))
    return {**best, "status": "SELECTED"}


def _fit_target_development(D: dict[str, np.ndarray], direction: str, seed: int) -> list[dict[str, Any]]:
    """Recompute only the 60 development OOF rows, never using test labels."""
    old = joblib.load(original_model(direction, seed)); x, y = D["x"], D["medical"]
    rows: list[dict[str, Any]] = []
    for pos in range(3):
        part = old["positions"][pos]; A = np.asarray(part["A"], int); B = np.asarray(part["B"], int)
        loc = {int(v): j for j, v in enumerate(A)}; cross = {a: np.full((len(A), 4), np.nan) for a in ALPHAS}; covered = []
        for inner in part["inner"]:
            T = np.asarray(inner["T"], int); V = np.asarray(inner["V"], int); covered.extend(V.tolist())
            for a in ALPHAS: cross[a][[loc[int(v)] for v in V]] = target_mod.predict(target_mod.fit(x[T], y[T], a), x[V])
        assert sorted(covered) == sorted(A.tolist())
        losses = {a: float(np.mean(np.abs(cross[a][:, 2] - y[A, 2]))) for a in ALPHAS}
        best = min(ALPHAS, key=lambda a: (losses[a], -a)); med = cross[best]; model = target_mod.fit(x[A], y[A], best)
        pred_cls = fr.ma_class(med[:, 2]); truth_cls = fr.ma_class(y[A, 2]); ym = y[A].mean(0); ys = np.maximum(y[A].std(0), 1e-6)
        gate = fr._fit_ma_gate(fr.ma_features(x[A], med, ym, ys), pred_cls, truth_cls, f"paper_b085_equal_error_{direction}_{seed}_{pos}", seed)
        if not gate["success"] and not gate["fallback"]: raise RuntimeError("MA gate failed")
        heads = {}
        for head in ("low", "high"):
            mask, hy = fr.c.build_head_mask(truth_cls, pred_cls, med[:, 2], D["input_qc"][A], head)
            support = fr.c.support_summary(hy, D["ids"][A][mask])
            if support["status"] != "SUPPORTED": heads[head] = None; continue
            z, pre = fr.c.build_features(x[A, 1][mask], med[:, 2][mask], np.ones(mask.sum(), bool)); w = fr.c.class_balanced_weights(hy)
            opt = minimize(fr.c.weighted_bce_objective, np.zeros(3), args=(z, hy, w, .1), jac=fr.c.weighted_bce_gradient,
                           method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-9})
            if not opt.success or not np.isfinite(opt.x).all(): raise RuntimeError(f"head fit failed: {direction} {seed} {pos} {head}")
            heads[head] = dict(theta=opt.x, pre=pre, support=support)
        if len(B):
            pb = target_mod.predict(model, x[B]); cb = fr.ma_class(pb[:, 2]); bs = fr.score_ma_gate(gate, fr.ma_features(x[B], pb, ym, ys), cb); ds = fr.head_scores(heads, x[B], pb)
            for j, i in enumerate(B):
                rows.append(dict(record_id=str(D["ids"][i]), pred_class=int(cb[j]), truth_class=int(fr.ma_class(y[[i], 2])[0]),
                                 base_score=float(bs[j]) if D["input_qc"][i] and np.isfinite(bs[j]) else None,
                                 dual_score=float(ds[j]) if D["input_qc"][i] and cb[j] == 0 and np.isfinite(ds[j]) else None))
    assert len(rows) == 60 and len({r["record_id"] for r in rows}) == 60
    return rows


def development_rows(D: dict[str, np.ndarray]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    out: dict[str, dict[int, list[dict[str, Any]]]] = {m: {} for m in METHODS}
    for seed in SEEDS:
        a = joblib.load(original_model("HZ_to_SA", seed)); b = joblib.load(original_model("HZ+HB_to_SA", seed))
        assert np.array_equal(a["pool"], b["pool"]) and np.array_equal(a["test"], b["test"])
        for pos in range(4):
            pa, pb = a["positions"][pos], b["positions"][pos]
            assert set(pa["A"]) == set(pb["A"]) and set(pa["B"]) == set(pb["B"])
            assert all(set(x["T"]) == set(y["T"]) and set(x["V"]) == set(y["V"]) for x, y in zip(pa["inner"], pb["inner"]))
    for method in DIRS:
        for seed in SEEDS:
            obj = joblib.load(TRANSFER_RUNS / f"prefix_900__{method}" / "dual" / f"{method}_{seed}.joblib")
            rows = [dict(record_id=str(r["record_id"]), pred_class=int(r["pred_class"]), truth_class=int(r["truth_class"]),
                         base_score=float(r["base_score"]) if r.get("base_score") is not None else None,
                         dual_score=float(r["dual_score"]) if r.get("dual_score") is not None else None) for r in obj["outer"]]
            assert len(rows) == 60 and len({r["record_id"] for r in rows}) == 60
            id_to_i = {str(v): i for i, v in enumerate(D["ids"])}
            assert all(r["record_id"] in id_to_i for r in rows)
            assert all(int(r["truth_class"]) == int(fr.ma_class(D["medical"][id_to_i[r["record_id"]], 2])) for r in rows)
            out[method][seed] = rows
    # The two original target-control objects have identical grouped pools,
    # positions and inner folds; use one direction only and keep the control
    # arm shared across both source comparisons.
    for seed in SEEDS:
        out["target_only"][seed] = _fit_target_development(D, "HZ_to_SA", seed)
    # Target-only uses a single target fit per seed; the rows are shared by
    # the two transfer directions in the paper's target-only control.
    return out


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_data(); dev = development_rows(D)
    rows_json = {m: {str(s): dev[m][s] for s in SEEDS} for m in METHODS}
    write_json(OUT / "DEVELOPMENT_ROWS.json", {"protocol": TAG, "test_read": False, "rows": rows_json})
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}; rates = set(float(x) for x in FIXED_CAPS)
    for method in METHODS:
        candidates[method] = {}
        for s in SEEDS:
            cs = candidate_rows(dev[method][s]); candidates[method][str(s)] = cs
            rates.update(float(c["error_rate"]) for c in cs if 0 <= float(c["error_rate"]) <= .20 + 1e-12)
    grid = sorted({round(float(x), 14) for x in rates if 0 <= x <= .20 + 1e-12}); grid = sorted(set(grid) | {0.0, .20})
    configs: dict[str, list[dict[str, Any]]] = {}; lookup = {}
    for method in METHODS:
        configs[method] = []; lookup[method] = {}; seen = {}
        for cap in grid:
            sels = {s: select_at_cap(candidates[method][str(s)], cap) for s in SEEDS}
            sig = tuple((s, sels[s].get("normal"), sels[s].get("tail"), sels[s].get("status")) for s in SEEDS)
            if sig not in seen:
                cid = f"{method}__cfg{len(seen):04d}"; seen[sig] = cid
                configs[method].append(dict(config_id=cid, first_r=float(cap), last_r=float(cap), selections={str(s): sels[s] for s in SEEDS}))
            else:
                cid = seen[sig]
                for c in configs[method]:
                    if c["config_id"] == cid: c["last_r"] = float(cap); break
            lookup[method][f"{cap:.14g}"] = seen[sig]
    write_json(OUT / "DEVELOPMENT_CANDIDATES.json", {"protocol": TAG, "test_read": False, "candidates": candidates})
    write_json(OUT / "STRATEGY_FREEZE.json", {"protocol": TAG, "test_read": False, "r_grid": grid, "fixed_caps": FIXED_CAPS,
        "error_budgets": ERROR_BUDGETS, "configs": configs, "config_lookup": lookup,
        "candidate_rule": "same empirical development cap selector; candidate error rate <=0.20",
        "source_scores": "revised transfer dual outer rows; target-only rows recomputed from development positions only",
        "base_policy_selection_test_labels_used": False,
        "d2_full_test_labels_used_after_base_freeze": True,
        "d3_calibration_fold_labels_used_for_selection": True,
        "data_sha256": sha(DATA)})
    write_json(OUT / "FREEZE.json", {"protocol": TAG, "test_evaluated": False, "data_sha256": sha(DATA),
        "development_rows_sha256": sha(OUT / "DEVELOPMENT_ROWS.json"), "strategy_freeze_sha256": sha(OUT / "STRATEGY_FREEZE.json"),
        "config_counts": {m: len(configs[m]) for m in METHODS}, "seed_count": len(SEEDS), "development_rows_per_method_seed": 60,
        "base_policy_selection_test_labels_used": False,
        "d2_full_test_labels_used_after_base_freeze": True,
        "d3_calibration_fold_labels_used_for_selection": True,
        "B_from_original_data": True})
    rows = []
    for method in METHODS:
        for c in configs[method]:
            for s in SEEDS:
                sel = c["selections"][str(s)]
                rows.append(dict(method=method, config_id=c["config_id"], first_cap=c["first_r"], last_cap=c["last_r"], seed=s,
                                 normal=sel.get("normal"), tail=sel.get("tail"), released_dev=sel.get("released", 0),
                                 correct_dev=sel.get("correct", 0), errors_dev=sel.get("errors", 0), error_rate_dev=sel.get("error_rate", 0), status=sel.get("status")))
    write_csv(OUT / "STRATEGY_FREEZE.csv", rows)
    cap_rows = []
    for method in METHODS:
        for s in SEEDS:
            for cap in FIXED_CAPS:
                sel = select_at_cap(candidates[method][str(s)], cap)
                cap_rows.append(dict(method=method, seed=s, cap=cap, normal=sel.get("normal"), tail=sel.get("tail"),
                                     released_dev=sel.get("released", 0), correct_dev=sel.get("correct", 0),
                                     errors_dev=sel.get("errors", 0), error_rate_dev=sel.get("error_rate", 0.0), status=sel.get("status")))
    write_csv(OUT / "FIXED_CAPS_DEVELOPMENT.csv", cap_rows)
    write_json(OUT / "DEVELOPMENT_AUDIT.json", {"protocol": TAG, "target_only_recomputed": True, "target_only_models": 40,
        "transfer_outer_rows_loaded": 40, "transfer_outer_truth_validated_against_revised_MA": True, "rows_per_seed": 60, "test_labels_used": False, "fit_scope": "development positions only",
        "input_data_sha256": sha(DATA), "target_model_sha256": {f"{m}_{s}": sha(TARGET_OUT / "models" / f"{m}_{s}.joblib") for m in DIRS for s in SEEDS}})
    print(json.dumps({"configs": {m: len(configs[m]) for m in METHODS}, "r_grid": len(grid), "test_evaluated": False}, indent=2))


def load_test_scores(D: dict[str, np.ndarray], method: str):
    # Target-only controls are refit once per seed; the replay writes the
    # same target arm alongside each source direction.  Use the first copy
    # and verify its test index/seed axis below.
    p = TARGET_OUT / "PREDICTIONS_HZ_to_SA.npz" if method == "target_only" else TRANSFER_PREDS / f"prefix_900__{method}.npz"
    with np.load(p, allow_pickle=False) as z:
        test_idx = z["test_idx"] if "test_idx" in z.files else z["test"]
        seed = z["seed"].astype(int); assert np.array_equal(seed, np.asarray(SEEDS))
        if method == "target_only":
            pred = fr.ma_class(z["med4"][:, :, 2]).astype(int); base = z["base_risk"].astype(float); dual = z["dual_risk"].astype(float); qc = z["input_qc"].astype(bool) if "input_qc" in z.files else np.broadcast_to(D["input_qc"][test_idx], pred.shape)
        else:
            pred = fr.ma_class(z["final_med4"][:, :, 2]).astype(int); base = z["base_score"].astype(float); dual = z["dual_score"].astype(float); qc = z["input_qc"].astype(bool)
        assert pred.shape == base.shape == dual.shape == qc.shape == (20, 201)
        return test_idx.astype(int), pred, base, dual, qc


def apply_policy(pred: np.ndarray, base: np.ndarray, dual: np.ndarray, qc: np.ndarray, sel_by_seed: dict[str, Any]) -> np.ndarray:
    out = np.zeros_like(pred, dtype=bool)
    for j, seed in enumerate(SEEDS):
        n = sel_by_seed[str(seed)].get("normal"); t = sel_by_seed[str(seed)].get("tail")
        normal = (pred[j] == 0) & np.isfinite(dual[j]) & (dual[j] <= float(n)) if n is not None else np.zeros(pred.shape[1], bool)
        tail = np.isin(pred[j], (-1, 1)) & np.isfinite(base[j]) & (base[j] <= float(t)) if t is not None else np.zeros(pred.shape[1], bool)
        out[j] = (normal | tail) & qc[j]
    return out


def evaluate() -> None:
    freeze_obj = json.loads((OUT / "STRATEGY_FREEZE.json").read_text(encoding="utf-8")); D = load_data()
    idx0 = None; scores = {}; truth = None; raw_groups = None
    for method in METHODS:
        idx, pred, base, dual, qc = load_test_scores(D, method); idx0 = idx if idx0 is None else idx0; assert np.array_equal(idx, idx0)
        scores[method] = (pred, base, dual, qc)
    truth = fr.ma_class(D["medical"][idx0, 2]).astype(int); raw_groups = D["groups"][idx0].astype(str); group_names = sorted(set(raw_groups)); group_to_i = {g:i for i,g in enumerate(group_names)}; groups = np.asarray([group_to_i[g] for g in raw_groups], int)
    arrays = {"truth": truth.astype(np.int8), "group": groups.astype(np.int16)}; action_map = {}
    for method in METHODS:
        action_map[method] = {}
        for cfg in freeze_obj["configs"][method]:
            pred, base, dual, qc = scores[method]; act = apply_policy(pred, base, dual, qc, cfg["selections"]); cid = cfg["config_id"].replace("+", "plus")
            arrays[f"pred__{method}__{cid}"] = pred.astype(np.int8); arrays[f"action__{method}__{cid}"] = act.astype(np.uint8); action_map[method][cfg["config_id"]] = {"first_cap": cfg["first_r"], "last_cap": cfg["last_r"]}
    np.savez_compressed(OUT / "SCORES_ACTIONS.npz", **arrays)
    write_json(OUT / "EVALUATION_COMPLETE.json", {"protocol": TAG, "test_evaluated": True, "test_records": 201, "seed_count": 20,
        "group_count": len(group_names), "truth_source": "revised-B DATA15 medical labels; used only after base policy freeze",
        "test_label_aware_diagnostic": True, "base_policy_selection_test_labels_used": False,
        "d2_full_test_labels_used_after_base_freeze": True,
        "d3_calibration_fold_labels_used_for_selection": True,
        "config_counts": {m: len(freeze_obj["configs"][m]) for m in METHODS}})
    write_json(OUT / "CONFIG_MAP.json", action_map)
    # Report the pre-specified cap points separately from the denser frontier.
    dev = json.loads((OUT / "DEVELOPMENT_CANDIDATES.json").read_text(encoding="utf-8"))["candidates"]
    cap_rows = []
    for method in METHODS:
        pred, base, dual, qc = scores[method]
        for cap in FIXED_CAPS:
            sels = {str(s): select_at_cap(dev[method][str(s)], cap) for s in SEEDS}
            act = apply_policy(pred, base, dual, qc, sels); correct = act & (pred == truth[None, :]); err = act & ~correct
            cap_rows.append(dict(method=method, cap=cap, released=int(act.sum()), correct=int(correct.sum()), errors=int(err.sum()),
                                 coverage=float(act.mean()), released_agreement=float(correct.sum()/act.sum()) if act.sum() else None,
                                 all_report_agreement=float(np.mean(pred == truth[None, :]))))
    write_csv(OUT / "FIXED_CAPS_TEST.csv", cap_rows)
    print(json.dumps({"test_records": 201, "groups": len(group_names), "configs": {m: len(freeze_obj["configs"][m]) for m in METHODS}}, indent=2))


def metric_arrays(method: str, freeze_obj: dict[str, Any], z: Any, groups: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    configs = freeze_obj["configs"][method]; group_ids = np.unique(groups); gi = {int(g):j for j,g in enumerate(group_ids)}; S, G = len(configs), len(group_ids)
    cg = np.zeros((S,G)); eg = np.zeros((S,G)); rg = np.zeros((S,G)); class_cg = {s:np.zeros((S,G)) for s in (-1,0,1)}; class_eg = {s:np.zeros((S,G)) for s in (-1,0,1)}
    for si,c in enumerate(configs):
        safe = c["config_id"].replace("+", "plus"); pred = z[f"pred__{method}__{safe}"].astype(int); act = z[f"action__{method}__{safe}"].astype(bool); correct = act & (pred == truth[None,:]); error = act & (pred != truth[None,:])
        for g in group_ids:
            m = groups == g; j = gi[int(g)]; cg[si,j] = correct[:,m].sum(); eg[si,j] = error[:,m].sum(); rg[si,j] = act[:,m].sum()
            for st in (-1,0,1):
                tm = m & (truth == st); class_cg[st][si,j] = correct[:,tm].sum(); class_eg[st][si,j] = error[:,tm].sum()
    record_count = np.asarray([(groups == g).sum() for g in group_ids], float)
    return {"configs":configs,"group_ids":group_ids,"record_count_group":record_count,"correct_group":cg,"error_group":eg,"release_group":rg,"class_correct_group":class_cg,"class_error_group":class_eg}


def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def upper_hull(errors: np.ndarray, correct: np.ndarray) -> list[int]:
    best = {}
    for i,x in enumerate(errors):
        k=float(x)
        if k not in best or correct[i] > correct[best[k]] + 1e-15: best[k] = i
    hull=[]
    for x,i in sorted(best.items()):
        while len(hull)>=2 and cross((float(errors[hull[-2]]),float(correct[hull[-2]])),(float(errors[hull[-1]]),float(correct[hull[-1]])),(float(errors[i]),float(correct[i]))) >= -1e-14: hull.pop()
        hull.append(i)
    return hull


def budget_mix(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    h = upper_hull(errors, correct); candidates=[]
    for i in h:
        if errors[i] <= budget + 1e-12: candidates.append(dict(strategy_i=i,strategy_j=i,weight_j=0.,error_yield=float(errors[i]),correct_yield=float(correct[i]),exact=False))
    for i,j in zip(h[:-1],h[1:]):
        ei,ej=float(errors[i]),float(errors[j])
        if ei <= budget + 1e-12 <= ej and ej-ei > 1e-15:
            w=(budget-ei)/(ej-ei); candidates.append(dict(strategy_i=i,strategy_j=j,weight_j=float(w),error_yield=float(budget),correct_yield=float((1-w)*correct[i]+w*correct[j]),exact=True))
    return max(candidates,key=lambda v:(v["correct_yield"],-v["error_yield"],-v["strategy_i"],-v["strategy_j"])) if candidates else None


def exact_mix(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    h=upper_hull(errors,correct)
    if not h or budget < errors[h[0]]-1e-12 or budget > errors[h[-1]]+1e-12: return None
    for i in h:
        if abs(errors[i]-budget)<=1e-12: return dict(strategy_i=i,strategy_j=i,weight_j=0.,error_yield=float(errors[i]),correct_yield=float(correct[i]),exact=True)
    for i,j in zip(h[:-1],h[1:]):
        ei,ej=float(errors[i]),float(errors[j])
        if ei-1e-12 <= budget <= ej+1e-12 and ej-ei>1e-15:
            w=(budget-ei)/(ej-ei); return dict(strategy_i=i,strategy_j=j,weight_j=float(w),error_yield=float(budget),correct_yield=float((1-w)*correct[i]+w*correct[j]),exact=True)
    return None


def deterministic_best(errors, correct, budget):
    ids=np.where(errors<=budget+1e-12)[0]
    if not len(ids): return None
    i=max(ids.tolist(),key=lambda k:(float(correct[k]),-float(errors[k]),-k)); return dict(strategy_i=int(i),strategy_j=int(i),weight_j=0.,error_yield=float(errors[i]),correct_yield=float(correct[i]),exact=False)


def strategy_vectors(ma: dict[str, Any], weights: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    denom=float(np.dot(weights,ma["record_count_group"])*20.)
    c=(weights@ma["correct_group"].T)/denom; e=(weights@ma["error_group"].T)/denom; r=(weights@ma["release_group"].T)/denom
    return c,e,r,denom


def bootstrap_method(ma: dict[str, Any], budgets: tuple[float,...], n_boot: int, seed: int) -> dict[str, Any]:
    rng=np.random.default_rng(seed); G=len(ma["group_ids"]); out={str(b):[] for b in budgets}
    for _ in range(n_boot):
        picks=rng.integers(0,G,size=G); w=np.bincount(picks,minlength=G).astype(float); c,e,_,_=strategy_vectors(ma,w)
        for b in budgets:
            m=budget_mix(e,c,b); out[str(b)].append(float(m["correct_yield"]) if m else float("nan"))
    ret={}
    for b,vals in out.items():
        a=np.asarray(vals); a=a[np.isfinite(a)]; ret[b]={"mean":float(a.mean()) if len(a) else None,"ci95_low":float(np.percentile(a,2.5)) if len(a) else None,"ci95_high":float(np.percentile(a,97.5)) if len(a) else None,"valid":int(len(a))}
    return ret


def paired_bootstrap(target: dict[str, Any], source: dict[str, Any], budgets: tuple[float, ...], n_boot: int, seed: int) -> dict[str, Any]:
    """Paired group bootstrap for source-minus-target budget mixtures."""
    assert np.array_equal(target["record_count_group"], source["record_count_group"])
    rng = np.random.default_rng(seed); G = len(target["group_ids"]); vals = {str(b): {"correct": [], "error": []} for b in budgets}
    for _ in range(n_boot):
        picks = rng.integers(0, G, size=G); weights = np.bincount(picks, minlength=G).astype(float)
        ct, et, _, _ = strategy_vectors(target, weights); cs, es, _, _ = strategy_vectors(source, weights)
        for b in budgets:
            mt = budget_mix(et, ct, b); ms = budget_mix(es, cs, b)
            if mt is None or ms is None: continue
            vals[str(b)]["correct"].append(float(ms["correct_yield"] - mt["correct_yield"]))
            vals[str(b)]["error"].append(float(ms["error_yield"] - mt["error_yield"]))
    out = {}
    for b, fields in vals.items():
        out[b] = {}
        for field, raw in fields.items():
            a = np.asarray(raw, float); out[b][field] = {"ci95_low": float(np.percentile(a, 2.5)) if len(a) else None,
                "ci95_high": float(np.percentile(a, 97.5)) if len(a) else None, "mean": float(a.mean()) if len(a) else None, "valid": int(len(a))}
    return out


def d2() -> None:
    assert (OUT/"EVALUATION_COMPLETE.json").exists(); frz=json.loads((OUT/"STRATEGY_FREEZE.json").read_text(encoding="utf-8")); z=np.load(OUT/"SCORES_ACTIONS.npz",allow_pickle=False); truth=z["truth"].astype(int); groups=z["group"].astype(int); denom=len(truth)*20
    all_ma={}; point=[]; matching={}
    for method in METHODS:
        ma=metric_arrays(method,frz,z,groups,truth); all_ma[method]=ma; c,e,r,_=strategy_vectors(ma,np.ones(len(ma["group_ids"])))
        for i,cfg in enumerate(ma["configs"]): point.append(dict(method=method,strategy_index=i,config_id=cfg["config_id"],first_cap=cfg["first_r"],last_cap=cfg["last_r"],correct=float(c[i]*denom),error=float(e[i]*denom),release=float(r[i]*denom),correct_yield=float(c[i]),error_yield=float(e[i]),coverage=float(r[i])))
        matching[method]={}
        for b in ERROR_BUDGETS:
            ex=exact_mix(e,c,b); bm=budget_mix(e,c,b); det=deterministic_best(e,c,b); matching[method][str(b)]={"exact":ex,"budget_mix":bm,"deterministic":det,
                "exact_strategy_i":ma["configs"][ex["strategy_i"]]["config_id"] if ex else None,"exact_strategy_j":ma["configs"][ex["strategy_j"]]["config_id"] if ex else None,
                "budget_strategy_i":ma["configs"][bm["strategy_i"]]["config_id"] if bm else None,"budget_strategy_j":ma["configs"][bm["strategy_j"]]["config_id"] if bm else None}
    write_csv(OUT/"strategy_points.csv",point); write_json(OUT/"D2_MATCHING.json",{"protocol":TAG,"test_label_aware_diagnostic":True,"error_budgets":ERROR_BUDGETS,"matching":matching})
    boots={m:bootstrap_method(all_ma[m],ERROR_BUDGETS,N_BOOT,20261102+i) for i,m in enumerate(METHODS)}
    paired={}
    for source in DIRS:
        paired[source]={}
        for b in ERROR_BUDGETS:
            t=matching["target_only"][str(b)]["budget_mix"]; s=matching[source][str(b)]["budget_mix"]
            paired[source][str(b)]={"point_correct_delta":float(s["correct_yield"]-t["correct_yield"]) if s and t else None,"point_error_delta":float(s["error_yield"]-t["error_yield"]) if s and t else None,"target":t,"source":s}
        paired[source]["bootstrap"] = paired_bootstrap(all_ma["target_only"], all_ma[source], ERROR_BUDGETS, N_BOOT, 20461102 + DIRS.index(source))
    write_json(OUT/"D2_BOOTSTRAP.json",{"protocol":TAG,"n_boot":N_BOOT,"method_bootstrap":boots,"paired_source_minus_target":paired,"test_label_aware":True})
    dest=[]
    for method in METHODS:
        ma=all_ma[method]
        for b in ERROR_BUDGETS:
            m=matching[method][str(b)]["budget_mix"]
            if not m: continue
            i,j,w=int(m["strategy_i"]),int(m["strategy_j"]),float(m["weight_j"])
            for st in (-1,0,1): dest.append(dict(method=method,budget=b,true_state=st,correct_expected=(1-w)*ma["class_correct_group"][st][i].sum()+w*ma["class_correct_group"][st][j].sum(),error_expected=(1-w)*ma["class_error_group"][st][i].sum()+w*ma["class_error_group"][st][j].sum(),strategy_i=ma["configs"][i]["config_id"],strategy_j=ma["configs"][j]["config_id"],weight_j=w))
    write_csv(OUT/"D2_CLASS_DESTINATIONS.csv",dest); write_json(OUT/"D2_COMPLETE.json",{"protocol":TAG,"test_label_aware_diagnostic":True,"strategies":{m:len(all_ma[m]["configs"]) for m in METHODS},"error_budgets":ERROR_BUDGETS,"n_boot":N_BOOT})
    print(json.dumps({"d2": {m: len(all_ma[m]["configs"]) for m in METHODS}}, indent=2))


def fold_assign(raw_groups: np.ndarray) -> tuple[np.ndarray,list[str]]:
    names=sorted(set(raw_groups.astype(str)),key=lambda g:hashlib.sha256(f"{TAG}|{g}".encode()).hexdigest()); f={g:i%5 for i,g in enumerate(names)}; return np.asarray([f[g] for g in raw_groups.astype(str)],int),names


def d3() -> None:
    assert (OUT/"D2_COMPLETE.json").exists(); frz=json.loads((OUT/"STRATEGY_FREEZE.json").read_text(encoding="utf-8")); z=np.load(OUT/"SCORES_ACTIONS.npz",allow_pickle=False); truth=z["truth"].astype(int); groups=z["group"].astype(int); D=load_data(); idx=np.load(TARGET_OUT/"PREDICTIONS_HZ_to_SA.npz",allow_pickle=False)["test_idx"]; folds,names=fold_assign(D["groups"][idx].astype(str)); fold_rows=[]
    for method in METHODS:
        ma=metric_arrays(method,frz,z,groups,truth); gids=ma["group_ids"]
        for fold in range(5):
            hold=np.asarray([int(g) for g in gids if folds[groups==g][0]==fold],int); cal=np.asarray([int(g) for g in gids if folds[groups==g][0]!=fold],int); hold_m=np.isin(gids,hold); cal_m=np.isin(gids,cal)
            for b in ERROR_BUDGETS:
                c,e,_,den_cal=strategy_vectors(ma,cal_m.astype(float)); m=budget_mix(e,c,b)
                if m is None: continue
                i,j,w=int(m["strategy_i"]),int(m["strategy_j"]),float(m["weight_j"]); den_hold=float(np.dot(hold_m.astype(float),ma["record_count_group"])*20.); ci=ma["correct_group"][i][hold_m].sum(); cj=ma["correct_group"][j][hold_m].sum(); ei=ma["error_group"][i][hold_m].sum(); ej=ma["error_group"][j][hold_m].sum(); ri=ma["release_group"][i][hold_m].sum(); rj=ma["release_group"][j][hold_m].sum()
                fold_rows.append(dict(method=method,fold=fold,budget=b,strategy_i=ma["configs"][i]["config_id"],strategy_j=ma["configs"][j]["config_id"],weight_j=w,cal_correct_yield=float(m["correct_yield"]),cal_error_yield=float(m["error_yield"]),holdout_correct=float((1-w)*ci+w*cj),holdout_error=float((1-w)*ei+w*ej),holdout_release=float((1-w)*ri+w*rj),holdout_denom=den_hold,holdout_correct_yield=float(((1-w)*ci+w*cj)/den_hold),holdout_error_yield=float(((1-w)*ei+w*ej)/den_hold)))
    write_csv(OUT/"D3_FOLD_RESULTS.csv",fold_rows); agg=[]
    for method in METHODS:
        for b in ERROR_BUDGETS:
            rr=[r for r in fold_rows if r["method"]==method and float(r["budget"])==b]
            if len(rr)!=5: agg.append(dict(method=method,budget=b,folds=len(rr),correct_yield=None,error_yield=None)); continue
            den=sum(r["holdout_denom"] for r in rr); c=sum(r["holdout_correct"] for r in rr); e=sum(r["holdout_error"] for r in rr); rel=sum(r["holdout_release"] for r in rr); agg.append(dict(method=method,budget=b,folds=5,correct=c,error=e,release=rel,denom=den,correct_yield=c/den,error_yield=e/den,coverage=rel/den))
    write_csv(OUT/"D3_AGGREGATE.csv",agg); write_json(OUT/"D3_COMPLETE.json",{"protocol":TAG,"fold_rule":"sort SHA256(TAG|group), round-robin 5 folds","fold_count":5,"budgets":ERROR_BUDGETS,"choices_use_calibration_only":True,"test_label_aware":True,"calibration_fold_labels_used_for_selection":True,"heldout_fold_labels_used_for_selection":False,"aggregate":agg})
    print(json.dumps({"d3_fold_rows":len(fold_rows),"aggregate_rows":len(agg)},indent=2))


def results() -> None:
    d2=json.loads((OUT/"D2_MATCHING.json").read_text(encoding="utf-8")); d3=json.loads((OUT/"D3_COMPLETE.json").read_text(encoding="utf-8")); lines=["# Revised-B equal-error diagnostic","","This is a retrospective, exploratory diagnostic on the pre-existing 201-record test cohort. The base policy family is selected from 60 development OOF rows without using the test cohort. D2 then uses the full test labels to construct a descriptive equal-error frontier. D3 uses labels from the four calibration folds of this same historical test cohort to choose a threshold or mixture, and reports it on the held-out fifth fold; the held-out fold labels are not used for that fold's selection. Neither analysis is independent validation or a deployment policy.","","## D2 budget-matched frontier","","The `budget_mix` rows are expected yields from a convex random mixture of two frozen policy strategies. They are not a deterministic patient-level threshold. A confidence interval crossing zero is not evidence of non-inferiority.",""]
    for source in DIRS:
        lines.append(f"### {source} minus target-only"); lines.append("")
        for b in ERROR_BUDGETS:
            t=d2["matching"]["target_only"][str(b)]["budget_mix"]; s=d2["matching"][source][str(b)]["budget_mix"]; dc=(s["correct_yield"]-t["correct_yield"]) if s and t else None; de=(s["error_yield"]-t["error_yield"]) if s and t else None
            dc_text = f"{100*dc:.3f} pp" if dc is not None else "not estimable"; de_text = f"{100*de:.3f} pp" if de is not None else "not estimable"
            lines.append(f"- error budget {100*b:.1f}%: correct-yield delta {dc_text}; error-yield delta {de_text}.")
        lines.append("")
    lines += ["## D3 five-fold cross-calibration","","Each fold uses the other four group folds' labels to choose the strategy or mixture, then reports the expected yield on the held-out fifth fold. Thus D3 is a cross-calibration sensitivity analysis that uses labels from the historical test cohort for its calibration folds; it does not use the held-out fold labels for selection.","","## Boundary","","The study authors report that experimenters at the participating hospitals identified the supplied U-TEG MA values and curves as original instrument measurements. The original model and base deployment-policy selection did not use test labels, but the D2/D3 diagnostics are explicitly test-label-aware after that freeze. No result here should be described as independent validation, non-inferiority, or a deployment threshold.",""]
    (OUT/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    write_json(OUT/"RESULTS.json", {"protocol":TAG,"test_label_aware_diagnostic":True,"exploratory":True,"independent_validation":False,
        "base_policy_selection_test_labels_used":False,"d2_full_test_labels_used_after_base_freeze":True,"d3_calibration_fold_labels_used_for_selection":True,
        "fixed_caps":list(FIXED_CAPS),"error_budgets":list(ERROR_BUDGETS),"d2":d2,"d3":d3})


def verify() -> None:
    """Independent read-back checks for lineage, dimensions and denominators."""
    D = load_data(); frz = json.loads((OUT / "STRATEGY_FREEZE.json").read_text(encoding="utf-8")); ev = json.loads((OUT / "EVALUATION_COMPLETE.json").read_text(encoding="utf-8"))
    dev = json.loads((OUT / "DEVELOPMENT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    development_ok = all(len(dev[m][str(s)]) == 60 and len({r["record_id"] for r in dev[m][str(s)]}) == 60 for m in METHODS for s in SEEDS)
    z = np.load(OUT / "SCORES_ACTIONS.npz", allow_pickle=False); assert z["truth"].shape == (201,) and z["group"].shape == (201,)
    test_idx = z["truth"]
    prediction_checks = {}
    for method in METHODS:
        p = TARGET_OUT / "PREDICTIONS_HZ_to_SA.npz" if method == "target_only" else TRANSFER_PREDS / f"prefix_900__{method}.npz"
        with np.load(p, allow_pickle=False) as q:
            pred = fr.ma_class(q["med4"][:, :, 2]).astype(int) if method == "target_only" else fr.ma_class(q["final_med4"][:, :, 2]).astype(int)
            saved = q["pred_class"].astype(int); prediction_checks[method] = bool(np.array_equal(pred, saved) and pred.shape == (20, 201))
    d3 = json.loads((OUT / "D3_COMPLETE.json").read_text(encoding="utf-8")); groups = z["group"].astype(int)
    fold_group_sets = []
    raw = D["groups"][np.load(TARGET_OUT / "PREDICTIONS_HZ_to_SA.npz", allow_pickle=False)["test_idx"]].astype(str)
    folds, names = fold_assign(raw)
    for f in range(5): fold_group_sets.append({g for g in names if folds[raw == g][0] == f})
    disjoint = all(fold_group_sets[i].isdisjoint(fold_group_sets[j]) for i in range(5) for j in range(i)) and len(set().union(*fold_group_sets)) == len(names)
    checks = {"protocol":TAG,"data_sha256_matches":frz["data_sha256"] == sha(DATA),"development_rows_60_each":development_ok,
        "test_records":int(ev["test_records"]),"seed_count":int(ev["seed_count"]),"prediction_classes_match_saved_med4":prediction_checks,
        "d2_denominator":201*20,"d3_fold_count":int(d3["fold_count"]),"d3_group_folds_disjoint_and_cover":disjoint,
        "base_policy_selection_test_labels_used":False,"d2_full_test_labels_used_after_base_freeze":True,
        "d3_calibration_fold_labels_used_for_selection":True,"d3_heldout_fold_labels_used_for_selection":False,
        "test_label_aware_after_base_freeze":True,"B_from_original_data":True,
        "no_manuscript_writes":True}
    write_json(OUT / "VERIFICATION.json", checks)
    assert checks["data_sha256_matches"] and checks["development_rows_60_each"] and all(prediction_checks.values()) and disjoint
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def independent_audit() -> None:
    """Read-back audit of the large D2 deltas and target-only support.

    This intentionally reads only frozen CSV/JSON artifacts; it does not fit
    a predictor, touch test labels for selection, or regenerate predictions.
    """
    freeze_obj = json.loads((OUT / "STRATEGY_FREEZE.json").read_text(encoding="utf-8"))
    matching = json.loads((OUT / "D2_MATCHING.json").read_text(encoding="utf-8"))["matching"]
    dev = json.loads((OUT / "DEVELOPMENT_ROWS.json").read_text(encoding="utf-8"))["rows"]["target_only"]
    target_dev_rows = [r for seed_rows in dev.values() for r in seed_rows]
    high_support = {"SUPPORTED": 0, "UNSUPPORTED": 0}
    heads_path = TARGET_OUT / "development_heads.csv"
    if heads_path.exists():
        import csv
        with heads_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("head") == "high": high_support[row.get("status", "UNSUPPORTED")] = high_support.get(row.get("status", "UNSUPPORTED"), 0) + 1
    out = {"protocol": TAG, "read_only": True, "base_policy_selection_test_labels_used": False,
           "d2_full_test_labels_used_after_base_freeze": True, "d3_calibration_fold_labels_used_for_selection": True,
           "target_only_strategy_count": len(freeze_obj["configs"]["target_only"]),
           "target_only_development_rows": len(target_dev_rows),
           "target_only_development_dual_score_finite": int(sum(r.get("dual_score") is not None and np.isfinite(float(r["dual_score"])) for r in target_dev_rows)),
           "target_only_development_base_score_finite": int(sum(r.get("base_score") is not None and np.isfinite(float(r["base_score"])) for r in target_dev_rows)),
           "target_only_high_head_support": high_support, "denominator_events": 201 * 20, "d2": {}}
    for source in DIRS:
        t = matching["target_only"]["0.05"]["budget_mix"]; s = matching[source]["0.05"]["budget_mix"]
        if t is None or s is None: raise RuntimeError(f"D2 5% comparison missing: {source}")
        t_correct = float(t["correct_yield"]); s_correct = float(s["correct_yield"])
        t_error = float(t["error_yield"]); s_error = float(s["error_yield"])
        out["d2"][source] = {"budget": 0.05, "target_correct_yield": t_correct, "source_correct_yield": s_correct,
            "correct_yield_delta": s_correct - t_correct, "target_error_yield": t_error, "source_error_yield": s_error,
            "error_yield_delta": s_error - t_error, "target_expected_correct_events": t_correct * 4020,
            "source_expected_correct_events": s_correct * 4020, "source_strategy_i": s["strategy_i"], "source_strategy_j": s["strategy_j"]}
    write_json(OUT / "INDEPENDENT_AUDIT.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage in ("freeze","all"): freeze()
    if stage in ("evaluate","all"): evaluate()
    if stage in ("d2","all"): d2()
    if stage in ("d3","all"): d3()
    if stage in ("results","all"): results()
    if stage in ("verify","all"): verify()
    if stage in ("audit","all"): independent_audit()


if __name__ == "__main__":
    with threadpool_limits(limits=1): main()
