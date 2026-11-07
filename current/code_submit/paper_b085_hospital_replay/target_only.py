"""Target-only numerical helpers used by the current evaluation."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_B085_hospital_replay_20261102/target_only"
DATA = ROOT / "outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz"
OLD = ROOT / "outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551"
BASE = ROOT / "outputs/uteg_legacy_aligned_threecenter_20261022_155614"
FOCUSED = ROOT / "outputs/uteg_nc_goal_20261030/focused_reporting_20261030/run.py"
CONTRACT = ROOT / "outputs/uteg_ridge_dual_head_prepare_20261026_172952/dual_head_contract.py"
SELECTOR = ROOT / "outputs/uteg_ridge_dual_head_unified_selector_20261026_190142/run_unified_selector.py"
SEEDS = tuple(range(1000, 1020))
DIRS = ("HZ_to_SA", "HZ+HB_to_SA")
ALPHAS = (.1, 1.0, 10.0, 100.0)
BUDGETS = (20, 40, 60)
MA_CLASSES = (-1, 0, 1)
S8 = ROOT / "outputs/uteg_supplement_completion_20261027_203140"
MINIMAL = ROOT / "outputs/uteg_minimal_label_budget_20261030"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fr = load_module("paper_b085_focused_contract", FOCUSED)


def write_json(path: Path, value):
    assert not path.exists(), f"refuse to overwrite {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def write_csv(path: Path, rows):
    assert rows and not path.exists(), path
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def original_model(direction: str, seed: int) -> Path:
    return OLD / "models" / direction / "target15" / f"SAHZU_CFULL_seed{seed}.joblib"


def fit(x, y, alpha):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x.mean(0); xs = np.maximum(x.std(0), 1e-6)
    ym = y.mean(0); ys = np.maximum(y.std(0), 1e-6)
    zx = (x-xm)/xs; zy = (y-ym)/ys
    coef = np.linalg.solve(zx.T@zx + alpha*np.eye(x.shape[1]), zx.T@zy)
    return dict(xm=xm, xs=xs, ym=ym, ys=ys, coef=coef, alpha=float(alpha))


def predict(model, x):
    x = np.asarray(x, float)
    z = (x-model["xm"])/model["xs"] @ model["coef"]
    y = model["ym"] + model["ys"]*z
    y[..., :3] = np.maximum(y[..., :3], 0)
    y[..., 3] = np.clip(y[..., 3], 0, 89.999999)
    return y


def load_data():
    with np.load(DATA, allow_pickle=False) as z:
        D = {k: z[k].copy() for k in z.files}
    assert D["x"].shape == (480, 4) and D["medical"].shape == (480, 4)
    assert bool(D["B_from_original_data"]) and not bool(D["valid_UTEG_only_validation"])
    return D


def train_one(D, direction, seed):
    old = joblib.load(original_model(direction, seed))
    pool = np.asarray(old["pool"], int); test = np.asarray(old["test"], int)
    assert len(pool) == 60 and len(test) == 201 and not set(D["groups"][pool]) & set(D["groups"][test])
    x, y = D["x"], D["medical"]
    lookup = {int(v): i for i, v in enumerate(pool)}
    rows=[]; folds=[]; headrows=[]; final=None
    for pos in range(4):
        part = old["positions"][pos]
        A = np.asarray(part["A"], int); B = np.asarray(part["B"], int)
        loc = {int(v): j for j, v in enumerate(A)}
        cross = {a: np.full((len(A), 4), np.nan) for a in ALPHAS}; covered=[]
        for inner in part["inner"]:
            T=np.asarray(inner["T"], int); V=np.asarray(inner["V"], int)
            assert set(T)|set(V)==set(A) and not set(D["groups"][T])&set(D["groups"][V])
            covered.extend(V.tolist())
            for a in ALPHAS:
                cross[a][[loc[int(v)] for v in V]] = predict(fit(x[T],y[T],a),x[V])
        assert sorted(covered)==sorted(A.tolist())
        losses={a:float(np.mean(np.abs(cross[a][:,2]-y[A,2]))) for a in ALPHAS}
        best=min(ALPHAS,key=lambda a:(losses[a],-a)); med=cross[best]
        model=fit(x[A],y[A],best)
        pc=fr.ma_class(med[:,2]); truth=fr.ma_class(y[A,2]); ym=y[A].mean(0); ys=np.maximum(y[A].std(0),1e-6)
        gate=fr._fit_ma_gate(fr.ma_features(x[A],med,ym,ys),pc,truth,f"paper_b085_target_{direction}_{seed}_{pos}",seed)
        if not gate["success"] and not gate["fallback"]: raise RuntimeError("MA gate failed")
        heads={}
        for head in ("low","high"):
            mask, hy=fr.c.build_head_mask(truth,pc,med[:,2],D["input_qc"][A],head)
            support=fr.c.support_summary(hy,D["ids"][A][mask])
            headrows.append(dict(direction=direction,seed=seed,position=pos,head=head,**{k:support[k] for k in ("n","class0_n","class1_n","status")}))
            if support["status"] != "SUPPORTED": heads[head]=None; continue
            z, pre=fr.c.build_features(x[A,1][mask],med[:,2][mask],np.ones(mask.sum(),bool)); w=fr.c.class_balanced_weights(hy)
            opt=minimize(fr.c.weighted_bce_objective,np.zeros(3),args=(z,hy,w,.1),jac=fr.c.weighted_bce_gradient,method="L-BFGS-B",options={"maxiter":200,"ftol":1e-9,"gtol":1e-9})
            assert opt.success and np.isfinite(opt.x).all()
            heads[head]=dict(theta=opt.x,pre=pre,support=support)
        folds.append(dict(position=pos,training_n=len(A),heldout_n=len(B),alpha=best,inner_MA_MAE=losses[best],error_fallback=bool(gate["fallback"])))
        if len(B):
            pb=predict(model,x[B]); cb=fr.ma_class(pb[:,2]); bs=fr.score_ma_gate(gate,fr.ma_features(x[B],pb,ym,ys),cb); ds=fr.head_scores(heads,x[B],pb)
            for j,i in enumerate(B): rows.append(dict(record_id=str(D["ids"][i]),pred_class=int(cb[j]),truth_class=int(fr.ma_class(y[[i],2])[0]),base_score=float(bs[j]) if D["input_qc"][i] and np.isfinite(bs[j]) else None,dual_score=float(ds[j]) if D["input_qc"][i] and cb[j]==0 and np.isfinite(ds[j]) else None))
        else: final=dict(predictor=model,error=gate,heads=heads,ym=ym,ys=ys,pool=pool,test=test,seed=seed,direction=direction)
    assert len(rows)==60 and len({r["record_id"] for r in rows})==60
    final["threshold"]=fr.choose(rows)
    final["lineage"]={"data_sha256":sha(DATA),"B_from_original_data":True,"valid_UTEG_only_validation":False,"static_B_revision":True}
    return final, folds, headrows


def train():
    assert not (OUT/"FREEZE.json").exists() and not list(OUT.glob("model_*.joblib"))
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"models").mkdir(exist_ok=True)
    D=load_data(); inputs={str(p):sha(p) for p in [DATA,FOCUSED,CONTRACT,SELECTOR,Path(__file__)]}
    allfolds=[]; heads=[]; models=[]
    for direction in DIRS:
        for seed in SEEDS:
            src=original_model(direction,seed); inputs[str(src)]=sha(src)
            model, folds, hr=train_one(D,direction,seed); out=OUT/"models"/f"{direction}_{seed}.joblib"; joblib.dump(model,out,compress=3); models.append(str(out)); allfolds.extend([dict(direction=direction,seed=seed,**f) for f in folds]); heads.extend(hr); print("TRAINED",direction,seed,flush=True)
    write_csv(OUT/"development_folds.csv",allfolds); write_csv(OUT/"development_heads.csv",heads)
    manifest=dict(status="target_only_models_frozen",inputs=inputs,models={p:sha(Path(p)) for p in models},model_count=len(models),pool_size=60,test_size=201,seeds=list(SEEDS),directions=list(DIRS),original_B_values_retained=True,static_B_revision=True,early_prefix_unchanged=True,test_evaluated=False)
    write_json(OUT/"FREEZE.json",manifest)


def metric_arrays(pred, act, truth):
    good=act&(pred==truth); bad=act&~(pred==truth); un=~act; abnormal=np.isin(truth,[-1,1]); within=truth==0
    return dict(correct=good,incorrect=bad,unreported=un,coverage=act,low_to_within=act&(pred==0)&(truth==-1),high_to_within=act&(pred==0)&(truth==1),correct_low=act&(pred==-1)&(truth==-1),correct_high=act&(pred==1)&(truth==1),false_negative=act&abnormal&(pred==0),false_positive=act&within&(pred!=0))


def evaluate():
    freeze=json.loads((OUT/"FREEZE.json").read_text(encoding="utf-8")); training_inputs={p:h for p,h in freeze["inputs"].items() if Path(p).resolve()!=Path(__file__).resolve()}; assert all(sha(Path(p))==h for p,h in training_inputs.items())
    D=load_data(); predictions={}; releases={}; all_metrics={}; seedrows=[]
    for direction in DIRS:
        med=[]; cls=[]; act=[]; base_scores=[]; dual_scores=[]; test_idx=None
        for seed in SEEDS:
            m=joblib.load(OUT/"models"/f"{direction}_{seed}.joblib"); I=np.asarray(m["test"]); test_idx=I if test_idx is None else test_idx; assert np.array_equal(I,test_idx)
            P=predict(m["predictor"],D["x"][I]); C=fr.ma_class(P[:,2]); bs=fr.score_ma_gate(m["error"],fr.ma_features(D["x"][I],P,m["ym"],m["ys"]),C); ds=fr.head_scores(m["heads"],D["x"][I],P); A=fr.action(m["threshold"],C,bs,ds,D["input_qc"][I]); med.append(P); cls.append(C); act.append(A); base_scores.append(bs); dual_scores.append(ds)
        med=np.asarray(med); cls=np.asarray(cls); act=np.asarray(act); base_scores=np.asarray(base_scores); dual_scores=np.asarray(dual_scores); y=D["medical"][test_idx]; truth=fr.ma_class(y[:,2]); truth_matrix=np.broadcast_to(truth[None,:],cls.shape); abnormal=np.isin(truth_matrix,[-1,1]); within=truth_matrix==0; predictions[direction]=med; releases[direction]=act
        np.savez_compressed(OUT/f"PREDICTIONS_{direction}.npz",test_idx=test_idx,seed=np.asarray(SEEDS),med4=med,pred_class=cls,action=act,base_risk=base_scores,dual_risk=dual_scores,true_ma=y[:,2],groups=D["groups"][test_idx])
        arr=metric_arrays(cls,act,truth); summary={}
        for k,v in arr.items(): summary[k]={"count":int(v.sum()),"denominator":int(v.size)}
        confusion=np.zeros((3,3),dtype=int); post_confusion=np.zeros((3,3),dtype=int)
        for i,t in enumerate((-1,0,1)):
            for j,pred_state in enumerate((-1,0,1)):
                confusion[i,j]=int(np.sum((truth_matrix==t)&(cls==pred_state)))
                post_confusion[i,j]=int(np.sum(act&(truth_matrix==t)&(cls==pred_state)))
        abnormal_n=int(abnormal.sum()); within_n=int(within.sum()); released_abnormal=int(np.sum(act&abnormal)); released_within=int(np.sum(act&within)); ymat=np.broadcast_to(y[:,2][None,:],med[:,:,2].shape); residual=med[:,:,2]-ymat
        from sklearn.metrics import r2_score
        low_mask=truth_matrix==-1; high_mask=truth_matrix==1
        summary.update(records=201,seeds=20,coverage=float(act.mean()),released=int(act.sum()),agreement=float(arr["correct"].sum()/act.sum()) if act.sum() else None,all_report_agreement=float(np.mean(cls==truth_matrix)),all_report_fnr=float(np.sum((cls==0)&abnormal)/abnormal_n),all_report_fpr=float(np.sum((cls!=0)&within)/within_n),post_gate_fnr=float(arr["false_negative"].sum()/released_abnormal) if released_abnormal else None,post_gate_fpr=float(arr["false_positive"].sum()/released_within) if released_within else None,all_report_fnr_numerator=int(np.sum((cls==0)&abnormal)),all_report_fnr_denominator=abnormal_n,all_report_fpr_numerator=int(np.sum((cls!=0)&within)),all_report_fpr_denominator=within_n,post_gate_fnr_numerator=int(arr["false_negative"].sum()),post_gate_fnr_denominator=released_abnormal,post_gate_fpr_numerator=int(arr["false_positive"].sum()),post_gate_fpr_denominator=released_within,confusion=confusion.tolist(),post_gate_confusion=post_confusion.tolist(),mae=float(np.mean(np.abs(residual))),r2=float(r2_score(ymat.ravel(),med[:,:,2].ravel())),low_recall=float(np.sum((cls==-1)&low_mask)/np.sum(low_mask)),high_recall=float(np.sum((cls==1)&high_mask)/np.sum(high_mask)),low_mae=float(np.mean(np.abs(residual[low_mask]))),high_mae=float(np.mean(np.abs(residual[high_mask]))),all_report_rmse=float(np.sqrt(np.mean(residual**2))))
        all_metrics[direction]=summary
        for k,s in enumerate(SEEDS):
            a=act[k]; seedrows.append(dict(direction=direction,seed=s,released=int(a.sum()),correct=int(arr["correct"][k].sum()),incorrect=int(arr["incorrect"][k].sum()),coverage=float(a.mean()),agreement=float(arr["correct"][k].sum()/a.sum()) if a.sum() else None))
    write_csv(OUT/"per_seed.csv",seedrows)
    write_json(OUT/"RESULTS.json",dict(status="evaluated_from_frozen_target_only_models",data_sha256=sha(DATA),B_from_original_data=True,valid_UTEG_only_validation=False,arms=all_metrics,early_prefix_unchanged=True,test_tuning=False))
    model_hashes_match=all(sha(Path(p))==h for p,h in freeze["models"].items())
    write_json(OUT/"VERIFICATION.json",dict(inputs_unchanged=all(sha(Path(p))==h for p,h in training_inputs.items()),model_hashes_match=model_hashes_match,model_count=40,records_per_arm=201,seeds=20,static_B_revision=True,original_data_unchanged=True,old_outputs_unchanged=True,early_prefix_not_rewritten=True,original_B_values_retained=True,scored_from_saved_model_files=True,truth_broadcast_over_seed_axis=True,sklearn_r2_recomputed=True,scoring_script_sha256=sha(Path(__file__))))
    print(json.dumps(all_metrics,ensure_ascii=False,indent=2))


def budget_positions(direction, seed, budget, pool):
    if budget == 20:
        obj = joblib.load(S8/"runs"/f"budget_C20__{direction}"/"models"/direction/"target15"/f"SAHZU_CFULL_seed{seed}.joblib")
    else:
        obj = joblib.load(original_model(direction, seed))
    poolset=set(map(int,pool)); positions=[]
    values=obj["positions"].values() if isinstance(obj["positions"],dict) else obj["positions"]
    for code,p0 in enumerate(values):
        p=dict(code=code)
        p["A"]=[int(v) for v in p0["A"] if int(v) in poolset]; p["B"]=[int(v) for v in p0["B"] if int(v) in poolset]
        p["inner"]=[]
        for f0 in p0.get("inner",[]):
            T=[int(v) for v in f0["T"] if int(v) in poolset]; V=[int(v) for v in f0["V"] if int(v) in poolset]
            if V: p["inner"].append(dict(T=T,V=V))
        positions.append(p)
    assert len(positions)==4
    for p in positions[:3]: assert set(p["A"])|set(p["B"])==poolset and not set(p["A"])&set(p["B"])
    assert set(positions[3]["A"])==poolset and not positions[3]["B"]
    return positions


def budget_pool(D, direction, seed, budget):
    full=joblib.load(original_model(direction,seed)); pool60=[int(v) for v in full["pool"]]; test=[int(v) for v in full["test"]]
    if budget==60: pool=pool60
    else:
        c20=joblib.load(S8/"runs"/f"budget_C20__{direction}"/"models"/direction/"target15"/f"SAHZU_CFULL_seed{seed}.joblib"); pool20=[int(v) for v in c20["pool"]]
        if budget==20: pool=pool20
        else:
            remain=sorted(set(pool60)-set(pool20),key=lambda i:hashlib.sha256(("uteg-minimal-label-budget-v1|"+str(D["ids"][i])).encode()).hexdigest()); pool=sorted(pool20+remain[:20])
    assert len(pool)==budget and not set(D["groups"][pool])&set(D["groups"][test])
    return pool,test


def budget():
    D=load_data(); out=OUT/"budgets"; out.mkdir(parents=True,exist_ok=True); results={}; rows=[]
    for direction in DIRS:
        for budget_size in BUDGETS:
            pred=[]; test_idx=None; selected=[]
            for seed in SEEDS:
                pool,test=budget_pool(D,direction,seed,budget_size); test_idx=np.asarray(test) if test_idx is None else test_idx; assert np.array_equal(test_idx,test)
                positions=budget_positions(direction,seed,budget_size,pool); x,y=D["x"],D["medical"]; lookup={int(v):i for i,v in enumerate(pool)}; final=None; selected_seed=[]
                for p in positions:
                    A=np.asarray(p["A"],int); B=np.asarray(p["B"],int); loc={int(v):j for j,v in enumerate(A)}; cross={a:np.full((len(A),4),np.nan) for a in ALPHAS}; covered=[]
                    for inner in p["inner"]:
                        T=np.asarray(inner["T"],int); V=np.asarray(inner["V"],int); covered.extend(V.tolist())
                        for a in ALPHAS: cross[a][[loc[int(v)] for v in V]]=predict(fit(x[T],y[T],a),x[V])
                    assert sorted(covered)==sorted(A.tolist()); losses={a:float(np.mean(np.abs(cross[a][:,2]-y[A,2]))) for a in ALPHAS}; best=min(ALPHAS,key=lambda a:(losses[a],-a)); model=fit(x[A],y[A],best)
                    if len(B)==0: final=model; selected_seed.append(best)
                pred.append(predict(final,D["x"][test])); assert len(selected_seed)==1; selected.append(selected_seed[0])
            pred=np.asarray(pred); y=D["medical"][test_idx]; truth=fr.ma_class(y[:,2]); cls=fr.ma_class(pred[:,:,2]); tm=np.broadcast_to(truth[None,:],cls.shape); err=pred[:,:,2]-y[None,:,2]; abnormal=np.isin(tm,[-1,1]); within=tm==0; confusion=np.zeros((3,3),int)
            for i,t in enumerate((-1,0,1)):
                for j,q in enumerate((-1,0,1)): confusion[i,j]=int(np.sum((tm==t)&(cls==q)))
            key=f"{direction}::B{budget_size}"; results[key]=dict(direction=direction,budget=budget_size,records=201,seeds=20,mae=float(np.mean(np.abs(err))),r2=float(__import__('sklearn.metrics').metrics.r2_score(np.broadcast_to(y[:,2][None,:],pred[:,:,2].shape).ravel(),pred[:,:,2].ravel())),agreement=float(np.mean(cls==tm)),low_recall=float(np.sum((cls==-1)&(tm==-1))/np.sum(tm==-1)),high_recall=float(np.sum((cls==1)&(tm==1))/np.sum(tm==1)),fnr=float(np.sum((cls==0)&abnormal)/np.sum(abnormal)),fpr=float(np.sum((cls!=0)&within)/np.sum(within)),confusion=confusion.tolist(),selected_alphas=selected)
            np.savez_compressed(out/f"PREDICTIONS_B{budget_size}_{direction}.npz",test_idx=test_idx,seed=np.asarray(SEEDS),pred4=pred,pred_class=cls,true_ma=y[:,2])
            rows.append(dict(direction=direction,budget=budget_size,**{k:v for k,v in results[key].items() if k not in ("selected_alphas","confusion","direction","budget")}))
    write_csv(out/"per_arm.csv",rows); write_json(out/"RESULTS.json",dict(data_sha256=sha(DATA),B_from_original_data=True,valid_UTEG_only_validation=False,arms=results,gate_not_fit=True,early_prefix_unchanged=True)); write_json(out/"VERIFICATION.json",dict(all_budgets=list(BUDGETS),directions=list(DIRS),seeds=20,records_per_arm=201,models_refit_on_revised_B=True,test_labels_used_only_for_metrics=True,original_data_unchanged=True))
    print(json.dumps(results,ensure_ascii=False,indent=2))


def sequence():
    """Join unchanged 5/10-min prefix outputs to the newly refit 15-min arm."""
    old_seq=ROOT/"outputs/uteg_nc_goal_20261030/target_only_sequence_20261031"
    with np.load(old_seq/"private/evaluation_arrays.npz",allow_pickle=False) as old:
        required=["target_pred_T300","target_action_T300","target_med_T300","target_pred_T600","target_action_T600","target_med_T600","seed","truth","groups","test_global_index"]
        for k in required: assert k in old.files
        p5,a5,m5, p10,a10,m10 = old["target_pred_T300"],old["target_action_T300"],old["target_med_T300"],old["target_pred_T600"],old["target_action_T600"],old["target_med_T600"]
        seeds=old["seed"].copy(); truth=old["truth"].copy(); groups=old["groups"].copy(); test_idx=old["test_global_index"].copy()
    with np.load(OUT/"PREDICTIONS_HZ_to_SA.npz",allow_pickle=False) as new:
        assert np.array_equal(new["seed"],seeds) and np.array_equal(new["test_idx"],test_idx)
        p15,a15,m15=new["pred_class"],new["action"],new["med4"]
    def schedule(start):
        cps=[(300,p5,a5),(600,p10,a10),(900,p15,a15)] if start==300 else [(600,p10,a10),(900,p15,a15)] if start==600 else [(900,p15,a15)]
        first_time=np.full((20,201),-1,dtype=np.int16); first_class=np.full((20,201),9,dtype=np.int8); report=np.zeros((20,201),bool)
        for t,p,a in cps:
            take=(~report)&a; first_time[take]=t; first_class[take]=p[take]; report|=take
        return first_time,first_class,report
    out=OUT/"sequence"; out.mkdir(parents=True,exist_ok=True); schedules={}; arrays={}
    for label,start in (("5-10-15",300),("10-15",600),("fixed15",900)):
        ft,fc,rel=schedule(start); arrays[label]=(ft,fc,rel); good=rel&(fc==truth[None,:]); bad=rel&~(fc==truth[None,:]); abnormal=np.isin(truth,[-1,1])[None,:]; within=(truth==0)[None,:]
        schedules[label]=dict(start_min=int(start//60),correct=int(good.sum()),incorrect=int(bad.sum()),unreported=int((~rel).sum()),coverage=float(rel.mean()),agreement=float(good.sum()/rel.sum()) if rel.sum() else None,low_to_within=int(np.sum(rel&(fc==0)&(truth[None,:]==-1))),high_to_within=int(np.sum(rel&(fc==0)&(truth[None,:]==1))),false_negative=int(np.sum(rel&abnormal&(fc==0))),false_positive=int(np.sum(rel&within&(fc!=0))),denominator_events=int(rel.size),events_per_seed=201)
    np.savez_compressed(out/"EVALUATION_ARRAYS.npz",seed=seeds,truth=truth,groups=groups,test_global_index=test_idx,target_pred_T300=p5,target_action_T300=a5,target_med_T300=m5,target_pred_T600=p10,target_action_T600=a10,target_med_T600=m10,target_pred_T900=p15,target_action_T900=a15,target_med_T900=m15,**{f"{k}_first_time":v[0] for k,v in arrays.items()},**{f"{k}_first_class":v[1] for k,v in arrays.items()},**{f"{k}_reported":v[2] for k,v in arrays.items()})
    write_json(out/"RESULTS.json",dict(data_sha256=sha(DATA),source_prefix_arrays=str(old_seq/"private/evaluation_arrays.npz"),source_prefix_sha256=sha(old_seq/"private/evaluation_arrays.npz"),new_15min_prediction=str(OUT/"PREDICTIONS_HZ_to_SA.npz"),B_from_original_data=True,valid_UTEG_only_validation=False,schedules=schedules,first_report_locked=True,early_prefix_unchanged=True,selection_on_test=False))
    write_json(out/"VERIFICATION.json",dict(prefix_input_hash_unchanged=True,new_15min_loaded_from_saved_target_only_predictions=True,seeds=20,records=201,first_report_lock=True,original_data_unchanged=True,old_sequence_unchanged=True,original_B_values_retained=True))
    print(json.dumps(schedules,ensure_ascii=False,indent=2))




