"""Finite development-selected FN-control comparison; manuscript untouched."""
import run as r
import argparse,json,time,inspect
import numpy as np
from threadpoolctl import threadpool_limits
from code_submit.ma_target_optimization import full_curve_cost_sensitive as solver
from code_submit.ma_target_optimization.asymmetric_ma_mapping import loss_terms as original_loss

O=r.OUT/'fn_control'
CANDIDATES=[('residual',4),('residual',8),('residual',16),('residual',32),('boundary',4),('boundary',16)]
ARMS=['target_only_SA','HZ_to_SA','HZ+HB_to_SA']
RULES=['Boundary','Logistic','HGB']
GUARDS=[0.,1.,2.]

def configure(kind,cost):
    r.c.selected.COST=float(cost)
    def loss(p,y,target,unused,c):return original_loss(p,y,target,kind,c)
    r.c.selected.loss_terms=loss
    solver.loss_terms=loss
    assert r.c.selected.models.solve.__globals__ is solver.solve.__globals__
    r.c._SOURCE.clear()

def draw(item,plan):
    included=set(sum(item['sources'].values(),[]))
    r.c.source_rows=lambda arm,h:np.array([i for i in plan['source_rows'].get(arm,[]) if i in included],int)
    r.c._SOURCE.clear()

def collect(d,sp,arm,seed):
    parts=[];original=r.scoring
    def capture(out,data,seed0,arm0,pos):
        scores,info=original(out,data,seed0,arm0,pos)
        if pos<3:parts.append(dict(pred=out['prediction'].copy(),truth=data['medical'][out['query'],2].copy(),rows=np.array(out['query']),scores=scores))
        return scores,info
    r.scoring=capture
    try:o,actions,audit,scores=r.reporting(d,sp,arm,seed)
    finally:r.scoring=original
    dp=np.concatenate([v['pred'] for v in parts]);dy=np.concatenate([v['truth'] for v in parts])
    ids=np.concatenate([v['rows'] for v in parts])
    assert set(ids)==set(sp['pool']) and len(ids)==len(set(ids))
    return dict(pred=o['prediction'],actions=actions,audit=audit,scores=scores,dp=dp,dy=dy,dev_rows=ids,ds={k:np.concatenate([v['scores'][k] for v in parts]) for k in RULES})

def guard(pred,margin):
    normal=(pred>=50)&(pred<=70)
    return ~normal|((pred>=50+margin)&(pred<=70-margin))

def choose(score,pred,truth):
    pc=r.state(pred);tc=r.state(truth);options=[]
    for margin in GUARDS:
        allowed=guard(pred,margin)
        for threshold in np.unique(score[np.isfinite(score)]):
            action=allowed&np.isfinite(score)&(score<=threshold)
            n=int(action.sum());ab=action&(tc!=1)
            if not n or not ab.any():continue
            fn=int(np.sum(ab&(pc==1)));acc=float(np.mean(pc[action]==tc[action]))
            if fn or acc<=.8:continue
            options.append(dict(threshold=float(threshold),margin=margin,reported=n,accuracy=acc,reporting_rate=n/len(pred),FN=fn,abnormal_released=int(ab.sum()),joint=n/len(pred)>.75))
    if not options:return dict(feasible=False)
    pick=max(options,key=lambda x:(x['reported'],x['accuracy'],-x['margin'],-x['threshold']))
    return dict(pick,feasible=True)

def apply(x,rule,pick):
    return guard(x['pred'],pick['margin'])&np.isfinite(x['scores'][rule])&(x['scores'][rule]<=pick['threshold'])

def passed(v):
    return bool(v and v['FNR']==0 and v['accuracy'] is not None and v['accuracy']>.8 and v['reporting_rate']>.75)

def protocol():
    return dict(candidates=CANDIDATES,normal_guard_mm=GUARDS,rules=RULES,arms=ARMS,
        splits_sha256=r.sha(r.OUT/'private/SPLITS.json'),alpha=.01,
        fixed='Same source/development/test samples, train-only inverse-class weights, RBF basis, 15-minute inputs, nested development gate fits; original class boundaries 50 and 70 mm unchanged.',
        loss='Residual: asymmetric squared errors towards normal. Boundary: MA squared residual plus class-boundary squared hinges, 0.5 mm outward abnormal margin. Only training reference values enter losses.',
        selection='For each rule, all model candidates and guard margins use only outer-development predictions. Require FN=0, three-class accuracy>80%, at least one abnormal release; maximize reporting rate then accuracy; ties prefer earlier candidate and smaller margin. Joint feasibility additionally requires reporting rate>75%. No feasible policy stays NA, never a fabricated zero FNR.',
        reporting='Retain all fixed candidates under original 15% development-error thresholds, all candidate zero-FN-threshold results, and per-rule development-selected candidates. Test labels used only for final outcome computation.',
        scope='Exploratory follow-up; 100 overlapping subsamples are not independent external validation. Parameter search is separate from unchanged statistical baseline.')

def run_one(item,d,plan):
    dest=O/'private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);r.c.OUT=dest;start=time.time();draw(item,plan)
    sp=item['targets']['SAHZU'];fixed=[];candidate=[];selected=[]
    for arm in ARMS:
        store=[]
        for index,(kind,cost) in enumerate(CANDIDATES):
            configure(kind,cost);x=collect(d,sp,arm,item['seed']);picks={k:choose(x['ds'][k],x['dp'],x['dy']) for k in RULES}
            name=f'{arm}_{kind}{cost}'
            np.savez_compressed(dest/(name+'.npz'),test=sp['test'],prediction=x['pred'],dev_rows=x['dev_rows'],dev_prediction=x['dp'],**{'test_score_'+k:v for k,v in x['scores'].items()},**{'dev_score_'+k:v for k,v in x['ds'].items()})
            r.write(dest/(name+'_selection.json'),dict(picks=picks,original_gate=x['audit']))
            store.append((x,picks))
        # Select policies before reading any test label to compute results.
        decisions={}
        for rule in RULES:
            valid=[i for i,(_,picks) in enumerate(store) if picks[rule]['feasible']]
            decisions[rule]=max(valid,key=lambda i:(store[i][1][rule]['reported'],store[i][1][rule]['accuracy'],-i)) if valid else None
        truth=d['medical'][sp['test'],2]
        for index,((kind,cost),(x,picks)) in enumerate(zip(CANDIDATES,store)):
            for rule,action in x['actions'].items():
                v=r.metrics(x['pred'],truth,action)
                fixed.append(dict(rep=item['rep'],arm=arm,kind=kind,cost=cost,rule=rule,test=v,joint_test_pass=passed(v)))
            for rule,pick in picks.items():
                v=r.metrics(x['pred'],truth,apply(x,rule,pick)) if pick['feasible'] else None
                candidate.append(dict(rep=item['rep'],arm=arm,kind=kind,cost=cost,rule=rule,development=pick,test=v,joint_test_pass=passed(v)))
        for rule,index in decisions.items():
            if index is None:
                selected.append(dict(rep=item['rep'],arm=arm,rule=rule,choice=None,development=dict(feasible=False),test=None,joint_test_pass=False));continue
            x,picks=store[index];v=r.metrics(x['pred'],truth,apply(x,rule,picks[rule]))
            selected.append(dict(rep=item['rep'],arm=arm,rule=rule,choice=CANDIDATES[index],development=picks[rule],test=v,joint_test_pass=passed(v)))
    r.write(dest/'FIXED.json',fixed);r.write(dest/'CANDIDATE.json',candidate);r.write(dest/'SELECTED.json',selected)
    r.write(dest/'COMPLETE.json',dict(rep=item['rep'],seconds=time.time()-start))
    print('FN ROUND',item['rep'],round(time.time()-start,1),flush=True)

def summarize(rows,keys):
    groups={}
    for x in rows:groups.setdefault(tuple(x[k] for k in keys),[]).append(x)
    summary=[]
    for key,rr in groups.items():
        valid=[x for x in rr if x['test'] is not None];v=dict(zip(keys,key));v.update(rounds=len(rr),valid=len(valid),zero_FN_rounds=sum(x['test']['FNR']==0 for x in valid),nonzero_FN_rounds=sum(x['test']['FN']>0 for x in valid),FNR_undefined_rounds=sum(x['test']['FNR'] is None for x in valid),development_infeasible_rounds=len(rr)-len(valid),joint_test_pass=sum(x['joint_test_pass'] for x in rr),development_joint_pass=sum(x.get('development',{}).get('joint',False) for x in rr))
        for name in ['accuracy','reporting_rate','FNR','FPR','MAE','R2','low_recall','high_recall','low_retained_correct_rate','high_retained_correct_rate','FN','FP','reported']:
            values=[x['test'][name] for x in valid if x['test'][name] is not None]
            v[name]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None,min=float(min(values)),max=float(max(values)),n=len(values)) if values else None
        if 'choice' in rr[0]:
            v['choices']={str(c):sum(x['choice']==c for x in rr) for c in [None,*[list(t) for t in CANDIDATES]]}
        summary.append(v)
    return summary

def aggregate():
    allrows={k:[] for k in ['FIXED','CANDIDATE','SELECTED']}
    for p in sorted((O/'private').glob('round_*/COMPLETE.json')):
        for k in allrows:allrows[k]+=json.loads((p.parent/(k+'.json')).read_text(encoding='utf8'))
    for k,rows in allrows.items():
        r.write(O/(k+'_ROUNDS.json'),rows)
        r.write(O/(k+'_SUMMARY.json'),summarize(rows,['arm','rule'] if k=='SELECTED' else ['arm','kind','cost','rule']))
    print('FN AGGREGATED',len(allrows['SELECTED']),flush=True)

def smoke(d,plan,items):
    item=items[0];draw(item,plan);sp=item['targets']['SAHZU'];arm='HZ+HB_to_SA';configure('residual',4)
    x=collect(d,sp,arm,item['seed']);old=np.load(r.OUT/'private/round_00'/(arm+'_reporting.npz'))
    np.testing.assert_allclose(x['pred'],old['prediction'],atol=1e-9,rtol=0)
    for k in RULES:np.testing.assert_allclose(x['scores'][k],old['score_'+k],atol=1e-9,rtol=0)
    configure('boundary',16);x=collect(d,sp,arm,item['seed']);picks={k:choose(x['ds'][k],x['dp'],x['dy']) for k in RULES}
    truth=d['medical'][sp['test']].copy()
    try:
        d['medical'][sp['test']]=9999.;configure('boundary',16);z=collect(d,sp,arm,item['seed'])
        np.testing.assert_allclose(x['pred'],z['pred'],atol=1e-9,rtol=0)
        for k in RULES:
            np.testing.assert_allclose(x['scores'][k],z['scores'][k],atol=1e-9,rtol=0)
            assert picks[k]==choose(z['ds'][k],z['dp'],z['dy'])
    finally:d['medical'][sp['test']]=truth
    p=np.array([-1.4,-.8,.2,.7,1.4]);y=np.array([-1.2,-1.2,0,1.3,1.3]);eps=1e-6
    for kind,cost in CANDIDATES:
        f,g,h=original_loss(p,y,np.ones(5,bool),kind,cost)
        plus=original_loss(p+eps,y,np.ones(5,bool),kind,cost)[0];minus=original_loss(p-eps,y,np.ones(5,bool),kind,cost)[0]
        np.testing.assert_allclose(g,(plus-minus)/(2*eps),atol=1e-6,rtol=1e-6)
    assert not choose(np.zeros(3),np.ones(3)*60,np.ones(3)*60)['feasible']
    r.write(O/'SMOKE.json',dict(cost4_reproduces_original=True,boundary_loss_gradients_checked=True,test_labels_do_not_affect_fit_scores_or_selection=True,no_abnormal_development_releases_not_zero_FNR=True))
    print('FN SMOKE PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);a=ap.parse_args()
    if a.stage=='aggregate':aggregate();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    pr=protocol();file=O/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text(encoding='utf8'))==r.safe(pr)
    else:r.write(file,pr)
    if a.stage=='smoke':smoke(d,plan,items)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[a.start:a.stop]:run_one(item,d,plan)
        aggregate()
    inputs=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes']
    assert all(r.sha(r.current.ROOT/'data'/p)==h for p,h in inputs.items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
