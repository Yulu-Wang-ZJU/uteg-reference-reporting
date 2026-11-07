"""Nested development-only reporting controls on the new fixed MA branch."""
import argparse
import shutil
import numpy as np
import joblib
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from . import core as c
from code_submit.paper_b085_hospital_replay import target_only as historical
from code_submit.paper_b085_hospital_replay import reporting_controls as rc
fr=historical.fr

def safe(obj):
    if isinstance(obj,dict):return {str(k):safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [safe(v) for v in obj]
    if isinstance(obj,np.ndarray):return safe(obj.tolist())
    if isinstance(obj,np.generic):return safe(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def clean_write(path,obj):c.write(path,safe(obj))

def fields(rows,b,p,train,h):
    x=c.feature(train,rows,h,'four')[rows].copy();x[:,1]=b
    med=np.zeros((len(rows),4));med[:,2]=p
    # Three absent auxiliary model outputs are constant slots, never reference
    # values. All gates receive the same physical4 + MA + margin/class features.
    return x,med,fr.ma_features(x,med,np.zeros(4),np.ones(4))

def fit_score(out,h,seed,direction,pos):
    d,*_=c.context();a=out['train'];b=out['query'];p=out['inner'];pq=out['prediction']
    x,med,f=fields(a,out['inner_b'],p,a,h)
    xq,mq,fq=fields(b,out['query_b'],pq,a,h)
    pc=fr.ma_class(p);t=fr.ma_class(d['y'][a,2]);qpc=fr.ma_class(pq);valid=np.isfinite(pq)&c.qc_at(h)[b]
    lr=fr._fit_ma_gate(f,pc,t,f'inward_{direction}_{h}_{seed}_{pos}',seed)
    assert lr['success'] or lr['fallback']
    ls=fr.score_ma_gate(lr,fq,qpc);ls[~valid]=np.inf
    hgb=rc.fit_hgb(f,pc,t,np.ones(len(a),bool),direction=direction,seed=seed,pos=pos)
    hs=rc.score_hgb(hgb,fq,qpc,valid)
    heads={};support={}
    for head in ['low','high']:
        mask,y=fr.c.build_head_mask(t,pc,p,np.ones(len(a),bool),head)
        info=fr.c.support_summary(y,d['ids'][a][mask]);support[head]=info
        if info['status']!='SUPPORTED':heads[head]=None;continue
        z,pre=fr.c.build_features(x[:,1][mask],p[mask],np.ones(mask.sum(),bool));w=fr.c.class_balanced_weights(y)
        sol=minimize(fr.c.weighted_bce_objective,np.zeros(3),args=(z,y,w,.1),jac=fr.c.weighted_bce_gradient,
            method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9,'gtol':1e-9})
        assert sol.success and np.isfinite(sol.x).all()
        heads[head]=dict(theta=sol.x,pre=pre,support=info)
    ds=fr.head_scores(heads,xq,mq);ds[~valid]=np.inf
    return dict(LR=ls,HGB=hs,DUAL=ds,BOUNDARY=rc.margin_scores(pq)),dict(lr=lr,hgb=hgb,heads=heads,support=support)

def run_task(direction,h):
    path=c.OUT/'gates'/f'{direction}_T{h}.npz'
    if path.exists() and path.with_suffix('.metrics.json').exists() and path.with_suffix('.development.json').exists():return
    d,*_=c.context();saved=np.load(c.OUT/'primary'/f'{direction}_T{h}_B0.npz',allow_pickle=False)
    results={k:[] for k in ['LR','HGB','TWO_SIDED','BOUNDARY_PAPER']};scores={k:[] for k in ['LR','HGB','DUAL','BOUNDARY']};thresholds=[];support=[];devs=[]
    for si,seed in enumerate(c.SEEDS):
        spec=c.paper_spec(direction,seed);rows=[];dev_score={k:[] for k in scores};dev_pred=[];dev_true=[]
        for pos,part in enumerate(spec['positions'][:3]):
            out=c.evaluate(part['A'],part['B'],part['inner'],direction,h)
            ss,model=fit_score(out,h,seed,direction,pos);pc=fr.ma_class(out['prediction']);t=fr.ma_class(d['y'][out['query'],2])
            for k in dev_score:dev_score[k].extend(ss[k])
            dev_pred.extend(pc);dev_true.extend(t)
            for j,i in enumerate(out['query']):
                rows.append(dict(record_id=str(d['ids'][i]),pred_class=int(pc[j]),truth_class=int(t[j]),
                    base_score=float(ss['LR'][j]) if np.isfinite(ss['LR'][j]) else None,
                    dual_score=float(ss['DUAL'][j]) if np.isfinite(ss['DUAL'][j]) else None))
        assert len(rows)==len(spec['pool']) and len({r['record_id'] for r in rows})==len(rows)
        train=saved['training'][si]
        out=dict(train=train,query=saved['test'],prediction=saved['prediction'][si],inner=saved['inner'][si],
                 inner_b=saved['inner_b'][si],query_b=saved['query_b'][si])
        ss,model=fit_score(out,h,seed,direction,3)
        pc=fr.ma_class(out['prediction']);valid=np.isfinite(out['prediction'])&c.qc_at(h)[out['query']]
        th={k:rc.select_threshold(np.asarray(dev_score[k]),np.asarray(dev_pred),np.asarray(dev_true),.15) for k in ['LR','HGB','BOUNDARY']}
        dual=fr.choose(rows)
        for key,score in [('LR','LR'),('HGB','HGB'),('BOUNDARY_PAPER','BOUNDARY')]:
            v=th[score].get('threshold');results[key].append(np.zeros(len(pc),bool) if v is None else (ss[score]<=v)&np.isfinite(ss[score])&valid)
        results['TWO_SIDED'].append(fr.action(dual,pc,ss['LR'],ss['DUAL'],valid))
        for k in scores:scores[k].append(ss[k])
        thresholds.append(dict(seed=seed,base=th,dual=dual));support.append(dict(seed=seed,**model['support']))
        devs.append(dict(seed=seed,pred=dev_pred,truth=dev_true,scores=dev_score,rows=rows))
        modelpath=c.OUT/'gates/models'/f'{direction}_T{h}_{seed}.joblib';modelpath.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,modelpath,compress=3)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k+'_action':v for k,v in results.items()},**{k+'_score':v for k,v in scores.items()},
        prediction=saved['prediction'],test=saved['test'],seed=c.SEEDS,boundary95_action=saved['accepted'])
    clean_write(path.with_suffix('.thresholds.json'),thresholds);clean_write(path.with_suffix('.support.json'),support);clean_write(path.with_suffix('.development.json'),devs)
    clean_write(path.with_suffix('.metrics.json'),{k:c.summarize(saved['prediction'],d['y'][saved['test'],2],v) for k,v in results.items()})
    print('GATES_COMPLETE',direction,h,flush=True)

def run(reconstructed=False):
    parent=c.OUT
    if reconstructed:
        from .reconstruction import activate
        activate()
    clean_write(c.OUT/'GATE_PROTOCOL.json',dict(primary_gate='Boundary95 stays fixed',controls=['LR','HGB','TWO_SIDED','BOUNDARY_PAPER'],
        training='Original3 outer development positions, each gate fits upstream inner-OOF errors; thresholds chosen from outer development predictions only.',
        features='Original13slot interface: physical4 inputs, predicted MA,margin,class indicators; missing R/K/angle model-output slots constant0, never reference values.',
        selector='Historical select_threshold max_risk .15; original two-sided normal/tail selector and support requirements.',
        B_from_original_data=True))
    for direction in c.TARGET:
        for h in [900,300,600]:
            if reconstructed and h==900:
                src=parent/'gates';dest=c.OUT/'gates';dest.mkdir(parents=True,exist_ok=True)
                for p in src.glob(f'{direction}_T900.*'):
                    q=dest/p.name
                    if not q.exists():shutil.copy2(p,q)
                    assert c.sha(p)==c.sha(q)
                assert (dest/f'{direction}_T900.npz').exists()
            else:run_task(direction,h)
    clean_write(c.OUT/'GATES_COMPLETE.json',dict(directions=12,horizons=3,seed_count=20,thresholds_training_only=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reconstructed',action='store_true');a=ap.parse_args()
    with threadpool_limits(limits=1):run(a.reconstructed)
