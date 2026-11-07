"""Shared accounting for the isolated paper replay; each seed-record has one event."""
from pathlib import Path
import hashlib
import json
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'outputs/paper_B085_hospital_replay_20261102'
DATA=WORKSPACE/'outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data():
    with np.load(DATA,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('x','medical','ci','ids','centers','groups','input_qc')}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v))
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def states(ma):
    a=np.asarray(ma)
    return np.where(np.isfinite(a),np.where(a<50,-1,np.where(a>70,1,0)),9)

def classify(pred_class,truth_class,action=None):
    pc=np.asarray(pred_class);tc=np.broadcast_to(truth_class,pc.shape)
    ac=np.ones(pc.shape,bool) if action is None else np.broadcast_to(action,pc.shape).astype(bool)
    assert np.isin(tc,[-1,0,1]).all()
    assert np.isin(pc[ac],[-1,0,1]).all()
    good=ac&(pc==tc);fn=ac&(tc!=0)&(pc==0);fp=ac&(tc==0)&(pc!=0)
    abnormal=ac&(tc!=0);normal=ac&(tc==0)
    div=lambda n,d: float(n/d) if d else None
    cm=[[int(np.sum(ac&(tc==i)&(pc==j))) for j in [-1,0,1]] for i in [-1,0,1]]
    return dict(events=pc.size,reported=int(ac.sum()),correct=int(good.sum()),incorrect=int((ac&~good).sum()),
        withheld=int((~ac).sum()),coverage=float(ac.mean()),agreement=div(good.sum(),ac.sum()),
        fnr=div(fn.sum(),abnormal.sum()),fpr=div(fp.sum(),normal.sum()),
        fn=int(fn.sum()),fn_denominator=int(abnormal.sum()),fp=int(fp.sum()),fp_denominator=int(normal.sum()),
        low_recall=div(cm[0][0],np.sum(ac&(tc==-1))),high_recall=div(cm[2][2],np.sum(ac&(tc==1))),
        confusion=cm,withheld_by_true_state=[int(np.sum(~ac&(tc==i))) for i in [-1,0,1]],
        all_true_class_counts=[int(np.sum(tc==i)) for i in [-1,0,1]],
        correct_yield=float(good.mean()),incorrect_yield=float((ac&~good).mean()))

def regression(prediction,reference,valid=None):
    p=np.asarray(prediction,float);y=np.broadcast_to(reference,p.shape)
    mask=np.isfinite(p).all(-1)&np.isfinite(y).all(-1)
    if valid is not None: mask &= np.broadcast_to(valid,mask.shape)
    assert mask.any()
    e=p[mask]-y[mask];ref=y[mask];ma=e[:,2]
    r2den=float(np.sum((ref[:,2]-ref[:,2].mean())**2))
    output=dict(events=int(mask.sum()),mae=np.mean(abs(e),0).tolist(),
        ma_rmse=float(np.sqrt(np.mean(ma**2))),ma_r2=float(1-np.sum(ma**2)/r2den) if r2den else None)
    tc=states(ref[:,2]);output['mae_by_class']={str(i):float(np.mean(abs(ma[tc==i]))) if np.any(tc==i) else None for i in [-1,0,1]}
    output['all_report']=classify(states(p[mask,2]),tc)
    return output

def paired_ratio(num,den,groups,reps=5000,seed=20261030):
    a=np.asarray(num,float);b=np.broadcast_to(den,a.shape).astype(float)
    assert a.ndim==2 and a.shape[1]==len(groups)
    unique=np.unique(groups)
    aa=np.array([a[:,groups==g].sum() for g in unique]);bb=np.array([b[:,groups==g].sum() for g in unique])
    draw=np.random.default_rng(seed).integers(0,len(unique),(reps,len(unique)))
    denom=bb[draw].sum(1);ok=denom>0
    point=float(aa.sum()/bb.sum()) if bb.sum() else None
    ci=np.quantile(aa[draw].sum(1)[ok]/denom[ok],[.025,.975]).tolist() if ok.any() else [None,None]
    return dict(estimate=point,lower=ci[0],upper=ci[1],group_count=len(unique),bootstrap_reps=reps,undefined_draws=int((~ok).sum()))
