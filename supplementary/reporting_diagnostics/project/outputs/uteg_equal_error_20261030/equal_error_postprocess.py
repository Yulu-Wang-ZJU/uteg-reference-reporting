from __future__ import annotations

"""Corrected, read-only post-processing for the frozen equal-error run.

The first implementation snapshot is preserved as
``equal_error_analysis_initial_snapshot.py``.  This script does not alter D1
thresholds or the saved test actions.  It fixes only frontier construction and
per-bootstrap matching: exact matching uses the upper hull (including downward
segments), while budget matching maximizes correct yield subject to error
yield <= budget.  It also runs the pre-specified five-fold calibration check.
"""

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
TAG = "uteg-equal-error-20261030"
METHODS = ["target_only", "HZ_to_SA", "HZ+HB_to_SA"]
SEEDS = list(range(1000, 1020))
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.075, 0.10]
N_BOOT = 5000
N_CROSS_BOOT = 1000


def load_core():
    spec = importlib.util.spec_from_file_location("equal_error_core", ROOT / "equal_error_analysis.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


core = load_core()


def canonical(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k): canonical(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [canonical(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.write_text(json.dumps(canonical(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert path.resolve().is_relative_to(ROOT.resolve())
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(canonical(rows))


def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def upper_hull(errors: np.ndarray, correct: np.ndarray) -> list[int]:
    """Return strategy indices on the upper convex hull sorted by error.

    Duplicate x coordinates keep the highest-correct point.  A nonnegative
    cross product removes a middle point whose slope would increase; negative
    cross products retain descending segments, including a genuine decline in
    correct yield at larger error.  This is the upper frontier for maximizing
    correct yield, not a monotone envelope that deletes declining points.
    """
    best_at_x: dict[float, int] = {}
    for i, x in enumerate(errors):
        key = float(x)
        old = best_at_x.get(key)
        if old is None or float(correct[i]) > float(correct[old]) + 1e-15 or (abs(float(correct[i])-float(correct[old])) <= 1e-15 and i < old):
            best_at_x[key] = int(i)
    ordered = sorted(((x, i) for x, i in best_at_x.items()), key=lambda z: z[0])
    hull: list[tuple[float, int]] = []
    for x, i in ordered:
        p = (float(x), float(correct[i]))
        while len(hull) >= 2:
            p0 = (hull[-2][0], float(correct[hull[-2][1]])); p1 = (hull[-1][0], float(correct[hull[-1][1]]))
            if cross(p0, p1, p) >= -1e-14:
                hull.pop()
            else:
                break
        hull.append((float(x), int(i)))
    return [i for _, i in hull]


def exact_mix(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    h = upper_hull(errors, correct)
    if not h or budget < float(errors[h[0]]) - 1e-12 or budget > float(errors[h[-1]]) + 1e-12:
        return None
    # Exact vertex first (there may be multiple strategies at the same error,
    # but duplicate x coordinates were reduced in upper_hull).
    for i in h:
        if abs(float(errors[i]) - budget) <= 1e-12:
            return {"strategy_i": i, "strategy_j": i, "weight_j": 0.0,
                    "error_yield": float(errors[i]), "correct_yield": float(correct[i]), "exact": True}
    for i, j in zip(h[:-1], h[1:]):
        ei, ej = float(errors[i]), float(errors[j])
        if ei - 1e-12 <= budget <= ej + 1e-12 and ej - ei > 1e-15:
            w = (budget - ei) / (ej - ei)
            return {"strategy_i": i, "strategy_j": j, "weight_j": float(w),
                    "error_yield": float(budget), "correct_yield": float((1-w)*correct[i] + w*correct[j]), "exact": True}
    return None


def budget_mix(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    """Maximize correct yield over convex mixes with error <= budget."""
    h = upper_hull(errors, correct)
    if not h:
        return None
    candidates: list[dict[str, Any]] = []
    for i in h:
        if float(errors[i]) <= budget + 1e-12:
            candidates.append({"strategy_i": i, "strategy_j": i, "weight_j": 0.0,
                               "error_yield": float(errors[i]), "correct_yield": float(correct[i]), "exact": False})
    for i, j in zip(h[:-1], h[1:]):
        ei, ej = float(errors[i]), float(errors[j])
        if ei - 1e-12 <= budget <= ej + 1e-12 and ej - ei > 1e-15:
            w = (budget - ei) / (ej - ei)
            candidates.append({"strategy_i": i, "strategy_j": j, "weight_j": float(w),
                               "error_yield": float(budget), "correct_yield": float((1-w)*correct[i] + w*correct[j]), "exact": True})
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["correct_yield"]), -float(x["error_yield"]), -int(x["strategy_i"]), -int(x["strategy_j"])))


def deterministic_best(errors: np.ndarray, correct: np.ndarray, budget: float) -> dict[str, Any] | None:
    ids = np.where(errors <= budget + 1e-12)[0]
    if not len(ids): return None
    i = max(ids.tolist(), key=lambda k: (float(correct[k]), -float(errors[k]), -k))
    return {"strategy_i": int(i), "strategy_j": int(i), "weight_j": 0.0, "error_yield": float(errors[i]), "correct_yield": float(correct[i]), "exact": False}


def metric_vectors(ma: dict[str, Any], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    denom = float(np.dot(weights, ma["record_count_group"]) * 20.0)
    if denom <= 0: return np.full(len(ma["configs"]), np.nan), np.full(len(ma["configs"]), np.nan), denom
    return (weights @ ma["correct_group"].T) / denom, (weights @ ma["error_group"].T) / denom, denom


def load_state():
    fr = json.loads((ROOT / "strategy_freeze.json").read_text(encoding="utf-8"))
    z = np.load(ROOT / "scores_actions_private.npz", allow_pickle=False)
    truth = z["truth"].astype(int); groups = z["group"].astype(int)
    ma = {m: core._metric_arrays(m, fr, z, groups, truth) for m in METHODS}
    # D1's original freeze did not materialize the explicitly pre-specified
    # all-withhold policy. Add it here as a zero-action virtual policy without
    # changing the D1 freeze or any saved action. This supplies the (0,0)
    # frontier anchor needed for low-error budgets.
    for m in METHODS:
        mm = ma[m]
        cid = f"{m}__all_withhold"
        G = len(mm["group_ids"])
        mm["configs"] = list(mm["configs"]) + [{"config_id": cid, "first_r": 0.0, "last_r": 0.0, "virtual_all_withhold": True}]
        for k in ["correct_group", "error_group", "release_group"]:
            mm[k] = np.vstack([mm[k], np.zeros((1, G), dtype=float)])
        for k in ["class_correct_group", "class_error_group"]:
            for st in [-1, 0, 1]:
                mm[k][st] = np.vstack([mm[k][st], np.zeros((1, G), dtype=float)])
    # Export an explicit audit-friendly tensor with one row per frozen policy,
    # seed and record. Existing D1 actions are copied; the virtual row is zero.
    D = core.load_data(); idx = core.test_indices(); raw_groups = D["groups"][idx].astype(str)
    export = {"truth": truth.astype(np.int8), "group": groups.astype(np.int16), "fold": fold_assignment(raw_groups).astype(np.int8)}
    action_map = {}
    for m in METHODS:
        for c in ma[m]["configs"]:
            cid = c["config_id"]; safe = cid.replace("+", "plus")
            src = "action__" + safe
            if src in z.files:
                export[f"pred__{safe}"] = z["pred__" + safe].astype(np.int8)
                export[f"action__{safe}"] = z[src].astype(np.uint8)
            else:
                export[f"pred__{safe}"] = np.zeros((20, len(truth)), dtype=np.int8)
                export[f"action__{safe}"] = np.zeros((20, len(truth)), dtype=np.uint8)
            action_map[safe] = {"method": m, "config_id": cid, "all_withhold": bool(c.get("virtual_all_withhold", False))}
    np.savez_compressed(ROOT / "policy_actions_private.npz", **export)
    write_json(ROOT / "policy_actions_map.json", action_map)
    return fr, z, truth, groups, ma


def strategy_name(ma: dict[str, Any], idx: int) -> str:
    return ma["configs"][int(idx)]["config_id"]


def destination_for_mix(ma: dict[str, Any], mix: dict[str, Any], denom: float) -> dict[str, Any]:
    i, j, w = int(mix["strategy_i"]), int(mix["strategy_j"]), float(mix["weight_j"])
    d = {"strategy_i": strategy_name(ma, i), "strategy_j": strategy_name(ma, j), "weight_j": w}
    for st in [-1, 0, 1]:
        ci = float(ma["class_correct_group"][st][i].sum()); cj = float(ma["class_correct_group"][st][j].sum())
        ei = float(ma["class_error_group"][st][i].sum()); ej = float(ma["class_error_group"][st][j].sum())
        d[f"correct_true_{st}"] = ((1-w)*ci + w*cj) / denom
        d[f"error_true_{st}"] = ((1-w)*ei + w*ej) / denom
    return d


def d2() -> None:
    fr, z, truth, groups, ma = load_state(); full_weights = np.ones(len(ma["target_only"]["group_ids"]), dtype=float)
    point_rows=[]; all_point_rows=[]; match_rows=[]; classes=[]; full_match={}
    for method in METHODS:
        c,e,den=metric_vectors(ma[method],full_weights); h=upper_hull(e,c)
        all_point_rows.extend([{"method":method,"strategy_index":i,"config_id":strategy_name(ma[method],i),"error_yield":float(e[i]),"correct_yield":float(c[i]),"coverage_yield":float(ma[method]["release_group"][i].sum()/den)} for i in range(len(c))])
        write_hull = [{"method":method,"hull_position":k,"strategy_index":i,"config_id":strategy_name(ma[method],i),"error_yield":float(e[i]),"correct_yield":float(c[i])} for k,i in enumerate(h)]
        point_rows.extend(write_hull)
        full_match[method]={"hull":write_hull}
        for b in BUDGETS:
            ex=exact_mix(e,c,b); le=budget_mix(e,c,b); det=deterministic_best(e,c,b)
            row={"method":method,"budget":b,"exact":ex,"budget_mix":le,"deterministic":det}
            for label,m in [("exact",ex),("budget_mix",le),("deterministic",det)]:
                if m is not None:
                    m=dict(m); m["strategy_i_config_id"]=strategy_name(ma[method],m["strategy_i"]); m["strategy_j_config_id"]=strategy_name(ma[method],m["strategy_j"]); row[label]=m
            match_rows.append(row)
            full_match[method][str(b)] = row
            if ex is not None:
                classes.append({"method":method,"budget":b,"kind":"exact","denom":den,**destination_for_mix(ma[method],ex,den)})
    write_csv(ROOT/"d2_all_strategy_points.csv",all_point_rows); write_csv(ROOT/"d2_hull_points.csv",point_rows); write_csv(ROOT/"d2_matching.csv",match_rows)
    write_csv(ROOT/"d2_matching_classes.csv",classes)
    write_json(ROOT/"d2_post_full.json",{"protocol":TAG,"budgets":BUDGETS,"matching":full_match,"notes":"exact uses upper-hull equality/mixing; budget_mix uses error<=budget; both are test-label-aware diagnostics"})
    # Paired bootstrap: one group resample simultaneously drives target and both source methods.
    G=len(ma["target_only"]["group_ids"]); rng=np.random.default_rng(20261030); vals={s:{str(b):{"exact_c":[],"exact_e":[],"budget_c":[],"budget_e":[]} for b in BUDGETS} for s in ["HZ_to_SA","HZ+HB_to_SA"]}
    for rep in range(N_BOOT):
        picks=rng.integers(0,G,size=G); w=np.bincount(picks,minlength=G).astype(float)
        vec={m:metric_vectors(ma[m],w)[:2] for m in METHODS}
        for source in vals:
            for b in BUDGETS:
                for kind,fn in [("exact_c",exact_mix),("budget_c",budget_mix)]:
                    mt=fn(vec["target_only"][1],vec["target_only"][0],b); ms=fn(vec[source][1],vec[source][0],b)
                    if mt is None or ms is None: continue
                    vals[source][str(b)][kind].append(float(ms["correct_yield"]-mt["correct_yield"]))
                    vals[source][str(b)][kind.replace("_c","_e")].append(float(ms["error_yield"]-mt["error_yield"]))
    out={"protocol":TAG,"n_boot":N_BOOT,"budgets":BUDGETS,"comparisons":{}}
    for source in vals:
        out["comparisons"][source]={}
        for b in BUDGETS:
            item={}
            for kind in ["exact","budget"]:
                for what in ["c","e"]:
                    arr=np.asarray(vals[source][str(b)][f"{kind}_{what}"],dtype=float); item[f"{kind}_{what}"]={"valid":int(len(arr)),"ci95_low":float(np.percentile(arr,2.5)) if len(arr) else None,"ci95_high":float(np.percentile(arr,97.5)) if len(arr) else None,"mean":float(arr.mean()) if len(arr) else None}
            fm=full_match[source][str(b)]; tm=full_match["target_only"][str(b)]
            item["point"]={"exact_correct_delta":float(fm["exact"]["correct_yield"]-tm["exact"]["correct_yield"]) if fm["exact"] and tm["exact"] else None,"exact_error_delta":float(fm["exact"]["error_yield"]-tm["exact"]["error_yield"]) if fm["exact"] and tm["exact"] else None,"budget_correct_delta":float(fm["budget_mix"]["correct_yield"]-tm["budget_mix"]["correct_yield"]) if fm["budget_mix"] and tm["budget_mix"] else None,"budget_error_delta":float(fm["budget_mix"]["error_yield"]-tm["budget_mix"]["error_yield"]) if fm["budget_mix"] and tm["budget_mix"] else None}
            out["comparisons"][source][str(b)]=item
    write_json(ROOT/"d2_post_bootstrap.json",out)
    write_json(ROOT/"D2_POST_COMPLETE.json",{"protocol":TAG,"n_boot":N_BOOT,"budgets":BUDGETS,"methods":{m:len(ma[m]["configs"]) for m in METHODS},"matching_is_test_label_aware":True})


def fold_assignment(groups_raw: np.ndarray) -> np.ndarray:
    names=sorted(set(groups_raw.astype(str)),key=lambda g:hashlib.sha256(f"{TAG}|{g}".encode()).hexdigest())
    fm={g:i%5 for i,g in enumerate(names)}
    return np.asarray([fm[g] for g in groups_raw.astype(str)],dtype=int)


def cross_d3() -> None:
    assert (ROOT/"D2_POST_COMPLETE.json").exists()
    fr,z,truth,groups,ma=load_state(); za=np.load(ROOT/"policy_actions_private.npz",allow_pickle=False); D=core.load_data(); idx=core.test_indices(); raw=D["groups"][idx].astype(str); folds=fold_assignment(raw); group_ids=ma["target_only"]["group_ids"]
    choices=[]; fold_out=[]; seed_out=[]; classes=[]
    for method in METHODS:
        mm=ma[method]
        for fold in range(5):
            cal=np.asarray([not bool(np.all(folds[groups==g]==fold)) for g in group_ids],dtype=bool)
            hold=~cal
            for b in BUDGETS:
                c_cal, e_cal, _ = metric_vectors(mm, cal.astype(float))
                m=budget_mix(e_cal, c_cal, b)
                if m is None: continue
                i,j,w=int(m["strategy_i"]),int(m["strategy_j"]),float(m["weight_j"]); den=float(np.dot(hold.astype(float),mm["record_count_group"])*20)
                ci=mm["correct_group"][i][hold].sum(); cj=mm["correct_group"][j][hold].sum(); ei=mm["error_group"][i][hold].sum(); ej=mm["error_group"][j][hold].sum(); ri=mm["release_group"][i][hold].sum(); rj=mm["release_group"][j][hold].sum()
                row={"method":method,"fold":fold,"budget":b,"strategy_i":strategy_name(mm,i),"strategy_j":strategy_name(mm,j),"weight_j":w,"cal_error_yield":m["error_yield"],"cal_correct_yield":m["correct_yield"],"holdout_correct":(1-w)*ci+w*cj,"holdout_error":(1-w)*ei+w*ej,"holdout_release":(1-w)*ri+w*rj,"holdout_denom":den,"holdout_correct_yield":((1-w)*ci+w*cj)/den,"holdout_error_yield":((1-w)*ei+w*ej)/den}
                choices.append(row); fold_out.append(row)
                # Class and seed accounting for the expected random mixture.
                for st in [-1,0,1]:
                    tmask=(truth==st)&np.isin(groups,group_ids[hold]);
                    c_i=((za["action__"+mm["configs"][i]["config_id"].replace("+","plus")].astype(bool)&(za["pred__"+mm["configs"][i]["config_id"].replace("+","plus")] == truth[None,:]))[:,tmask]).sum()
                    c_j=((za["action__"+mm["configs"][j]["config_id"].replace("+","plus")].astype(bool)&(za["pred__"+mm["configs"][j]["config_id"].replace("+","plus")] == truth[None,:]))[:,tmask]).sum()
                    classes.append({"method":method,"fold":fold,"budget":b,"true_state":st,"holdout_correct_expected":((1-w)*c_i+w*c_j),"holdout_denom":int(tmask.sum()*20)})
                for sj,s in enumerate(SEEDS):
                    p_i=za["pred__"+mm["configs"][i]["config_id"].replace("+","plus")][sj].astype(int); a_i=za["action__"+mm["configs"][i]["config_id"].replace("+","plus")][sj].astype(bool); p_j=za["pred__"+mm["configs"][j]["config_id"].replace("+","plus")][sj].astype(int); a_j=za["action__"+mm["configs"][j]["config_id"].replace("+","plus")][sj].astype(bool); hm=np.isin(groups,group_ids[hold]);
                    seed_out.append({"method":method,"fold":fold,"budget":b,"seed":s,"holdout_correct_expected":(1-w)*int((a_i[hm]&(p_i[hm]==truth[hm])).sum())+w*int((a_j[hm]&(p_j[hm]==truth[hm])).sum()),"holdout_error_expected":(1-w)*int((a_i[hm]&(p_i[hm]!=truth[hm])).sum())+w*int((a_j[hm]&(p_j[hm]!=truth[hm])).sum())})
    write_csv(ROOT/"d3_fold_results.csv",fold_out); write_csv(ROOT/"d3_fold_classes.csv",classes); write_csv(ROOT/"d3_fold_seed.csv",seed_out)
    agg=[]
    for method in METHODS:
        for b in BUDGETS:
            rr=[r for r in fold_out if r["method"]==method and float(r["budget"])==b]
            if len(rr)!=5: agg.append({"method":method,"budget":b,"folds":len(rr),"correct_yield":None,"error_yield":None}); continue
            den=sum(r["holdout_denom"] for r in rr); c=sum(r["holdout_correct"] for r in rr); e=sum(r["holdout_error"] for r in rr); rel=sum(r["holdout_release"] for r in rr); agg.append({"method":method,"budget":b,"folds":5,"correct":c,"error":e,"release":rel,"denom":den,"correct_yield":c/den,"error_yield":e/den,"coverage":rel/den})
    write_csv(ROOT/"d3_aggregate.csv",agg)
    # 1,000 weighted re-selection bootstrap. Fixed fold membership, selection re-run each draw.
    G=len(group_ids); rng=np.random.default_rng(20301030); paired={s:{str(b):{"correct":[],"error":[]} for b in BUDGETS} for s in ["HZ_to_SA","HZ+HB_to_SA"]}; method_boot={m:{str(b):{"correct":[],"error":[]} for b in BUDGETS} for m in METHODS}
    for rep in range(N_CROSS_BOOT):
        picks=rng.integers(0,G,size=G); ww=np.bincount(picks,minlength=G).astype(float); aggregate={m:{str(b):[0.0,0.0,0.0] for b in BUDGETS} for m in METHODS}; rep_valid={m:{str(b):True for b in BUDGETS} for m in METHODS}
        for method in METHODS:
            mm=ma[method]
            for fold in range(5):
                cal=np.asarray([not bool(np.all(folds[groups==g]==fold)) for g in group_ids],dtype=bool); hold=~cal
                cal_w=ww*cal; hold_w=ww*hold
                for b in BUDGETS:
                    c_cal, e_cal, _ = metric_vectors(mm, cal_w)
                    m=budget_mix(e_cal, c_cal, b)
                    if m is None:
                        rep_valid[method][str(b)] = False
                        continue
                    i,j,w=int(m["strategy_i"]),int(m["strategy_j"]),float(m["weight_j"]); den=float(np.dot(hold_w,mm["record_count_group"])*20)
                    if den<=0:
                        rep_valid[method][str(b)] = False
                        continue
                    ci=mm["correct_group"][i]@hold_w; cj=mm["correct_group"][j]@hold_w; ei=mm["error_group"][i]@hold_w; ej=mm["error_group"][j]@hold_w
                    aggregate[method][str(b)][0]+=((1-w)*ci+w*cj); aggregate[method][str(b)][1]+=((1-w)*ei+w*ej); aggregate[method][str(b)][2]+=den
        rep_values={m:{} for m in METHODS}
        for method in METHODS:
            for b in BUDGETS:
                c,e,d=aggregate[method][str(b)]
                if rep_valid[method][str(b)] and d>0:
                    rep_values[method][str(b)] = (c/d, e/d)
                    method_boot[method][str(b)]["correct"].append(c/d); method_boot[method][str(b)]["error"].append(e/d)
        for source in paired:
            for b in BUDGETS:
                if str(b) in rep_values[source] and str(b) in rep_values["target_only"]:
                    cs, es = rep_values[source][str(b)]; ct, et = rep_values["target_only"][str(b)]
                    paired[source][str(b)]["correct"].append(cs-ct); paired[source][str(b)]["error"].append(es-et)
    def ci(arr):
        a=np.asarray(arr,dtype=float); return {"valid":int(len(a)),"ci95_low":float(np.percentile(a,2.5)) if len(a) else None,"ci95_high":float(np.percentile(a,97.5)) if len(a) else None,"mean":float(a.mean()) if len(a) else None}
    paired_out={}
    for source in paired:
        paired_out[source]={str(b):{"correct_delta":ci(paired[source][str(b)]["correct"]),"error_delta":ci(paired[source][str(b)]["error"])} for b in BUDGETS}
    write_json(ROOT/"d3_cross_bootstrap.json",{"protocol":TAG,"n_boot":N_CROSS_BOOT,"paired_source_minus_target":paired_out,"method_bootstrap":{m:{str(b):{"correct":ci(method_boot[m][str(b)]["correct"]),"error":ci(method_boot[m][str(b)]["error"])} for b in BUDGETS} for m in METHODS}})
    write_json(ROOT/"D3_COMPLETE.json",{"protocol":TAG,"fold_rule":"sort SHA256(TAG|group), round-robin 5 folds","fold_count":5,"budgets":BUDGETS,"choices_use_calibration_only":True,"n_boot":N_CROSS_BOOT,"aggregate":agg})


def results() -> None:
    d2=json.loads((ROOT/"d2_post_bootstrap.json").read_text(encoding="utf-8")); d3=json.loads((ROOT/"d3_cross_bootstrap.json").read_text(encoding="utf-8")); lines=["# 等错误负担诊断结果（修正后）","","本附件是既有201例测试队列上的回顾性、测试标签可见诊断。所有策略先由60例开发OOF分数冻结；测试标签仅用于构造描述性等错误前沿和内部交叉校准。它不是独立验证或部署保证。","","## 如何读表","","exact是上方凸包在给定错误预算处的精确错误yield混合；budget是错误yield不超过预算时的最高期望正确yield。两策略混合是每个record–seed独立随机选择整套20-seed策略的期望，不按病例真值选动作。错误区间跨零不能称非劣。","", "## 回顾性前沿的双源相对target-only"]
    for b in BUDGETS:
        v=d2["comparisons"]["HZ+HB_to_SA"][str(b)]; p=v["point"]; ex=v["exact_c"]; be=v["budget_c"]; lines.append(f"- 错误预算{100*b:.1f}%：exact正确差={None if p['exact_correct_delta'] is None else 100*p['exact_correct_delta']:.3f} pp，CI=[{None if ex['ci95_low'] is None else 100*ex['ci95_low']:.3f},{None if ex['ci95_high'] is None else 100*ex['ci95_high']:.3f}]；budget正确差={None if p['budget_correct_delta'] is None else 100*p['budget_correct_delta']:.3f} pp，CI=[{None if be['ci95_low'] is None else 100*be['ci95_low']:.3f},{None if be['ci95_high'] is None else 100*be['ci95_high']:.3f}]。")
    lines += ["", "## 内部五折交叉校准", "", "D3每折只用其余四折选择两套策略或混合，再在留出组报告实际错误yield；实际错误可与预算不同。1000次分组重抽样包含重新选择。详见d3_aggregate.csv和d3_cross_bootstrap.json。", "", "## 科学边界", "", "如果双源只在较高错误预算下有优势，应写成覆盖率–错误代价权衡；如果多个预算的exact诊断及D3都稳定为正，才支持更窄的‘在该既有队列和冻结策略族中，等错误负担下存在经验增量’。这仍不能替代独立病例。", ""]
    (ROOT/"RESULTS_ZH.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage=="d2": d2()
    elif stage=="d3": cross_d3()
    elif stage=="results": results()
    elif stage=="all": d2(); cross_d3(); results()
    else: raise SystemExit(stage)


if __name__=="__main__": main()
