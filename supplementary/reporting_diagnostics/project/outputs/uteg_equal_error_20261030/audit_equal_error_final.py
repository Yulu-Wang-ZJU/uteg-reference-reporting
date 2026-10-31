from __future__ import annotations

"""Independent audit of the corrected equal-error post-processing outputs.

This script intentionally does not import equal_error_postprocess.py or its
frontier routines. It reads saved actions/tables and recomputes core counts,
upper-hull matching, freeze parity and the reported bootstrap availability.
It never writes outside the equal-error output directory.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ["target_only", "HZ_to_SA", "HZ+HB_to_SA"]
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.075, 0.10]


def load_json(name: str) -> Any:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hull(errors: np.ndarray, correct: np.ndarray) -> list[int]:
    best: dict[float, int] = {}
    for i, x in enumerate(errors):
        x = float(x)
        if x not in best or correct[i] > correct[best[x]] + 1e-15 or (abs(correct[i] - correct[best[x]]) <= 1e-15 and i < best[x]):
            best[x] = int(i)
    ordered = sorted((x, i) for x, i in best.items())
    keep: list[tuple[float, int]] = []
    for x, i in ordered:
        while len(keep) >= 2:
            x0, i0 = keep[-2]; x1, i1 = keep[-1]
            cr = (x1-x0)*(correct[i]-correct[i1]) - (correct[i1]-correct[i0])*(x-x1)
            if cr >= -1e-14: keep.pop()
            else: break
        keep.append((x, i))
    return [i for _, i in keep]


def exact(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    h = hull(errors, correct)
    if not h or budget < errors[h[0]] - 1e-12 or budget > errors[h[-1]] + 1e-12: return None
    for i in h:
        if abs(float(errors[i]) - budget) <= 1e-12:
            return {"strategy_i": i, "strategy_j": i, "weight_j": 0.0, "correct_yield": float(correct[i]), "error_yield": float(errors[i])}
    for i, j in zip(h[:-1], h[1:]):
        if errors[i] - 1e-12 <= budget <= errors[j] + 1e-12 and errors[j] - errors[i] > 1e-15:
            w = (budget-errors[i])/(errors[j]-errors[i])
            return {"strategy_i": i, "strategy_j": j, "weight_j": float(w), "correct_yield": float((1-w)*correct[i]+w*correct[j]), "error_yield": float(budget)}
    return None


def main() -> int:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    required = ["strategy_freeze.json", "scores_actions_private.npz", "policy_actions_private.npz", "d2_all_strategy_points.csv", "d2_post_full.json", "d2_post_bootstrap.json", "D2_POST_COMPLETE.json", "d3_fold_results.csv", "d3_aggregate.csv", "d3_cross_bootstrap.json", "D3_COMPLETE.json", "EVALUATION_COMPLETE.json"]
    missing = [x for x in required if not (ROOT / x).exists()]
    checks["required_artifacts"] = {"pass": not missing, "missing": missing}; failures += ["missing:" + x for x in missing]
    if missing:
        result = {"protocol": "uteg-equal-error-20261030", "status": "FAIL", "checks": checks, "failures": failures}
        (ROOT / "AUDIT_FINAL.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return 2
    fr = load_json("strategy_freeze.json"); ev = load_json("EVALUATION_COMPLETE.json"); pts = load_csv("d2_all_strategy_points.csv"); saved = load_json("d2_post_full.json"); boot = load_json("d2_post_bootstrap.json"); folds = load_csv("d3_fold_results.csv"); agg = load_csv("d3_aggregate.csv"); z = np.load(ROOT/"scores_actions_private.npz",allow_pickle=False); za = np.load(ROOT/"policy_actions_private.npz",allow_pickle=False); truth = z["truth"].astype(int); group = z["group"].astype(int); fold = za["fold"].astype(int)
    checks["freeze_before_test"] = {"pass": fr.get("test_read") is False and ev.get("freeze_verified_before_test") is True, "detail": {"test_read": fr.get("test_read"), "freeze_verified_before_test": ev.get("freeze_verified_before_test")}}
    checks["fixed_cap_parity"] = {"pass": len(fr.get("parity_with_previous_caps", [])) == 300 and all(x.get("normal_identical") and x.get("tail_identical") for x in fr.get("parity_with_previous_caps", [])), "rows": len(fr.get("parity_with_previous_caps", []))}
    checks["s14_reproduction"] = {"pass": all(v.get("pass") for v in ev.get("s14_check", {}).values()), "detail": ev.get("s14_check")}
    checks["shape_alignment"] = {"pass": z["truth"].shape == (201,) and z["group"].shape == (201,) and fold.shape == (201,), "truth": list(z["truth"].shape), "group": list(z["group"].shape), "fold": list(fold.shape)}
    config_counts = {m: len(fr["configs"][m]) for m in METHODS}
    expected_points = sum(config_counts.values()) + len(METHODS)  # one explicit all-withhold policy per method
    checks["all_withhold_export"] = {"pass": all(any(bool(x.get("all_withhold")) for x in load_json("policy_actions_map.json").values() if x.get("method") == m) for m in METHODS), "config_counts": config_counts}
    point_fail = []
    point_map = {(r["method"], r["config_id"]): r for r in pts}
    for m in METHODS:
        for c in fr["configs"][m]:
            cid = c["config_id"]; safe = cid.replace("+", "plus"); p = za["pred__"+safe].astype(int); a = za["action__"+safe].astype(bool); cor = a & (p == truth[None,:]); err = a & (p != truth[None,:]); row = point_map.get((m,cid))
            if row is None: point_fail.append(f"missing:{m}:{cid}"); continue
            for k,v in [("correct_yield",cor.sum()/4020), ("error_yield",err.sum()/4020), ("coverage_yield",a.sum()/4020)]:
                if abs(float(row[k])-float(v)) > 1e-12: point_fail.append(f"{m}:{cid}:{k}")
    checks["point_recalculation"] = {"pass": not point_fail and len(pts) == expected_points, "rows": len(pts), "expected": expected_points, "failures": point_fail[:20]}
    # Recompute corrected exact matching, independently of postprocess code.
    match_fail=[]
    for m in METHODS:
        rows=[r for r in pts if r["method"]==m]; errors=np.asarray([float(r["error_yield"]) for r in rows]); correct=np.asarray([float(r["correct_yield"]) for r in rows]);
        for b in BUDGETS:
            ref=exact(errors,correct,b); got=saved["matching"][m][str(b)]["exact"]
            if (ref is None) != (got is None): match_fail.append(f"{m}:{b}:reachability")
            elif ref is not None and (abs(ref["correct_yield"]-float(got["correct_yield"]))>1e-10 or abs(ref["error_yield"]-float(got["error_yield"]))>1e-10): match_fail.append(f"{m}:{b}:value")
    checks["exact_hull_recalculation"] = {"pass": not match_fail, "failures": match_fail}
    boot_fail=[]
    for m in ["HZ_to_SA","HZ+HB_to_SA"]:
        for b in BUDGETS:
            v=boot["comparisons"][m][str(b)]
            for key in ["exact_c","exact_e","budget_c","budget_e"]:
                if not (0 < int(v[key]["valid"]) <= 5000): boot_fail.append(f"{m}:{b}:{key}:{v[key]['valid']}")
    checks["d2_bootstrap_5000"] = {"pass": not boot_fail, "failures": boot_fail}
    fold_fail=[]
    for m in METHODS:
        for b in BUDGETS:
            rr=[r for r in folds if r["method"]==m and abs(float(r["budget"])-b)<1e-12]
            if len(rr)!=5: fold_fail.append(f"{m}:{b}:folds={len(rr)}")
            for r in rr:
                if not (0.0 <= float(r["weight_j"]) <= 1.0): fold_fail.append(f"{m}:{b}:weight")
    checks["d3_fold_integrity"] = {"pass": not fold_fail and len(agg)==len(METHODS)*len(BUDGETS), "failures": fold_fail, "aggregate_rows": len(agg)}
    # Recompute every held-out fold row from the exported policy×seed×record
    # tensor, including the expected random-mixture arithmetic.
    fold_value_fail = []
    fmap = {c["config_id"]: c for m in METHODS for c in fr["configs"][m]}
    for r in folds:
        m = r["method"]; i = r["strategy_i"]; j = r["strategy_j"]; w = float(r["weight_j"]); f = int(r["fold"]); safe_i = i.replace("+", "plus"); safe_j = j.replace("+", "plus")
        if "action__" + safe_i not in za.files or "action__" + safe_j not in za.files:
            fold_value_fail.append(f"missing_action:{m}:{i}:{j}"); continue
        hm = fold == f; den = float(hm.sum() * 20)
        ai = za["action__"+safe_i].astype(bool); aj = za["action__"+safe_j].astype(bool); pi = za["pred__"+safe_i].astype(int); pj = za["pred__"+safe_j].astype(int)
        ci = float((ai[:,hm] & (pi[:,hm] == truth[hm][None,:])).sum()); cj = float((aj[:,hm] & (pj[:,hm] == truth[hm][None,:])).sum())
        ei = float((ai[:,hm] & (pi[:,hm] != truth[hm][None,:])).sum()); ej = float((aj[:,hm] & (pj[:,hm] != truth[hm][None,:])).sum())
        exp_c = ((1-w)*ci+w*cj)/den; exp_e = ((1-w)*ei+w*ej)/den
        if abs(float(r["holdout_correct_yield"])-exp_c)>1e-10 or abs(float(r["holdout_error_yield"])-exp_e)>1e-10:
            fold_value_fail.append(f"value:{m}:{r['fold']}:{r['budget']}")
    checks["d3_fold_value_recalculation"] = {"pass": not fold_value_fail, "failures": fold_value_fail[:20], "failure_count": len(fold_value_fail)}
    d3=load_json("d3_cross_bootstrap.json"); d3_fail=[]
    for m in ["HZ_to_SA","HZ+HB_to_SA"]:
        for b in BUDGETS:
            for key in ["correct_delta","error_delta"]:
                if int(d3["paired_source_minus_target"][m][str(b)][key]["valid"]) != 1000: d3_fail.append(f"{m}:{b}:{key}")
    checks["d3_bootstrap_1000"] = {"pass": not d3_fail, "failures": d3_fail}
    original = load_json("root_original_hashes.json"); unchanged=[]
    for item in original:
        p=Path(item["path"]); unchanged.append({"path":str(p),"pass":p.exists() and sha(p).lower()==item["sha256"].lower()})
    checks["original_inputs_unchanged"] = {"pass": all(x["pass"] for x in unchanged), "files": unchanged}
    checks["fold_partition"] = {"pass": len(np.unique(fold))==5 and all(int((fold==k).sum())>0 for k in range(5)), "counts": {str(k):int((fold==k).sum()) for k in range(5)}}
    status = "PASS" if not failures and all(v.get("pass",False) for v in checks.values()) else "FAIL"
    result={"protocol":"uteg-equal-error-20261030","status":status,"checks":checks,"failures":failures}
    (ROOT/"AUDIT_FINAL.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# 等错误诊断独立验收", "", f"状态：**{status}**", "", "独立脚本未调用等错误后处理的凸包或匹配函数；重新核对了冻结、动作、全不报告点、策略点计数、上凸包精确匹配、5000次D2区间可用性、5折留出计数、1000次D3区间可用性和原输入哈希。", ""]
    for k,v in checks.items(): md.append(f"- {k}: {'通过' if v.get('pass') else '失败'}")
    (ROOT/"AUDIT_FINAL.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if status=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
