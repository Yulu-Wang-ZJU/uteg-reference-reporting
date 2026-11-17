"""Finite train-support controls, using existing100 paired draws in two budgets."""
import run as r
import fn_candidates as f
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
O=r.OUT/'support_followup'
MODES=['baseline','residual_clamp','anchored_tails']
GUARDS=['none','support','label0','label0.1','label0.2']
ARMS=['HZ_to_SA','HZ+HB_to_SA']
original_predict=r.c.predict
MODE='baseline'

def prediction(direction,h,train,query,representation='B',loss='inward4',prior_override=None):
    if MODE=='baseline':return original_predict(direction,h,train,query,representation,loss,prior_override)
    assert representation=='B' and loss=='inward4' and prior_override is None
    d,*_=r.c.context();train=np.asarray(train,int);query=np.asarray(query,int)
    x=r.c.feature(train,query,h,representation)
    shadow=dict(d,y=np.full_like(d['y'],np.nan));shadow['y'][train]=d['y'][train]
    z,q,y,_,audit=r.c.selected.models.design(shadow,[x],train,query,'target_only')
    w=r.c.selected.models.weights(y,0,loss);src=r.c.source_model(direction,h,representation,loss)
    lo=x[train,0].min();hi=x[train,0].max();clipped=x.copy();clipped[query,0]=np.clip(x[query,0],lo,hi)
    zc,qc,_,_,_=r.c.selected.models.design(shadow,[clipped],train,query,'target_only')
    np.testing.assert_allclose(z,zc,atol=1e-12,rtol=0)
    base=(r.c.source_predict(src,x[train])-60)/10 if src is not None else np.zeros(len(train))
    bq=(r.c.source_predict(src,x[query])-60)/10 if src is not None else np.zeros(len(query))
    bc=(r.c.source_predict(src,clipped[query])-60)/10 if src is not None else np.zeros(len(query))
    coef,info=r.c.selected.solve_offset(z,(y-60)/10,w,base)
    raw=60+10*(bq+q@coef)
    if MODE=='residual_clamp':p=60+10*(bq+qc@coef)
    else:
        endpoint=60+10*(bc+qc@coef);p=raw.copy()
        below=x[query,0]<lo;above=x[query,0]>hi
        p[below]=np.minimum(p[below],endpoint[below]);p[above]=np.maximum(p[above],endpoint[above])
    p=np.maximum(p,0.)
    assert np.isfinite(p).all()
    audit.update(mode=MODE,training_feature_range=[float(lo),float(hi)],solver=info,coefficient_sha256=r.c.selected.ahash(coef))
    return p,audit,x

def configure(mode):
    global MODE
    MODE=mode;f.configure('residual',4);r.c.predict=prediction

def guard_masks(d,train,query,pred):
    a=np.asarray(train,int);q=np.asarray(query,int);x=d['x'][:,1];y=d['medical'][a,2]
    normal=r.state(pred)==1
    ans={'none':np.ones(len(q),bool),'support':~normal|((x[q]>=x[a].min())&(x[q]<=x[a].max()))}
    lo=float(x[a][y<50].max()) if np.any(y<50) else -np.inf
    hi=float(x[a][y>70].min()) if np.any(y>70) else np.inf
    scale=float(np.quantile(x[a],.75)-np.quantile(x[a],.25))
    for m in [0,.1,.2]:ans['label'+str(m)]=~normal|((x[q]>lo+m*scale)&(x[q]<hi-m*scale))
    return ans

def collect(d,sp,arm,seed):
    parts=[];original=r.scoring
    def capture(out,data,seed0,arm0,pos):
        scores,info=original(out,data,seed0,arm0,pos)
        part=dict(pred=out['prediction'].copy(),rows=np.array(out['query']),scores=scores,guards=guard_masks(data,out['train'],out['query'],out['prediction']))
        if pos<3:part['truth']=data['medical'][out['query'],2].copy()
        parts.append(part)
        return scores,info
    r.scoring=capture
    try:o,actions,audit,scores=r.reporting(d,sp,arm,seed)
    finally:r.scoring=original
    dev=parts[:3];last=parts[3]
    return dict(pred=o['prediction'],scores=scores,actions=actions,audit=audit,guards=last['guards'],dp=np.concatenate([v['pred'] for v in dev]),dy=np.concatenate([v['truth'] for v in dev]),dev_rows=np.concatenate([v['rows'] for v in dev]),ds={k:np.concatenate([v['scores'][k] for v in dev]) for k in f.RULES},dg={k:np.concatenate([v['guards'][k] for v in dev]) for k in GUARDS})

def choose(x,rule):
    p=r.state(x['dp']);y=r.state(x['dy']);score=x['ds'][rule];options=[]
    for gi,guard in enumerate(GUARDS):
        for threshold in np.unique(score[np.isfinite(score)]):
            action=x['dg'][guard]&np.isfinite(score)&(score<=threshold)
            ab=action&(y!=1);n=int(action.sum())
            if not ab.any() or not n:continue
            fn=int(np.sum(ab&(p==1)));acc=float(np.mean(p[action]==y[action]))
            if fn or acc<=.8:continue
            options.append(dict(guard=guard,guard_index=gi,threshold=float(threshold),reported=n,accuracy=acc,reporting_rate=n/len(p),abnormal_released=int(ab.sum()),FN=0,joint=n/len(p)>.75))
    if not options:return dict(feasible=False)
    pick=max(options,key=lambda q:(q['reported'],q['accuracy'],-q['guard_index'],-q['threshold']))
    return dict(pick,feasible=True)

def action(x,rule,pick):return x['guards'][pick['guard']]&np.isfinite(x['scores'][rule])&(x['scores'][rule]<=pick['threshold'])

def spec(item,budget):
    if budget==54:return item['targets']['SAHZU']
    return json.loads((r.OUT/'full60_control/private'/f"round_{item['rep']:02d}"/'SPLIT.json').read_text(encoding='utf8'))

def run_one(item,d,plan,budget):
    dest=O/f'dev{budget}'/'private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    start=time.time();dest.mkdir(parents=True,exist_ok=True);r.c.OUT=dest;f.draw(item,plan);sp=spec(item,budget)
    fixed=[];candidates=[];selected=[]
    for arm in ARMS:
        results=[]
        for mode in MODES:
            configure(mode);x=collect(d,sp,arm,item['seed']);picks={k:choose(x,k) for k in f.RULES};results.append((x,picks))
            np.savez_compressed(dest/(arm+'_'+mode+'.npz'),test=sp['test'],dev_rows=x['dev_rows'],prediction=x['pred'],dev_prediction=x['dp'],**{'score_'+k:v for k,v in x['scores'].items()},**{'dev_score_'+k:v for k,v in x['ds'].items()},**{'guard_'+k:v for k,v in x['guards'].items()},**{'dev_guard_'+k:v for k,v in x['dg'].items()})
            r.write(dest/(arm+'_'+mode+'_selection.json'),dict(picks=picks,original=x['audit']))
        decisions={}
        for rule in f.RULES:
            valid=[i for i,(_,p) in enumerate(results) if p[rule]['feasible']]
            decisions[rule]=max(valid,key=lambda i:(results[i][1][rule]['reported'],results[i][1][rule]['accuracy'],-i)) if valid else None
        truth=d['medical'][sp['test'],2]
        for mode,(x,picks) in zip(MODES,results):
            for rule,act in x['actions'].items():
                for guard in GUARDS if rule!='All report' else ['none']:
                    v=r.metrics(x['pred'],truth,act&x['guards'][guard])
                    fixed.append(dict(rep=item['rep'],budget=budget,arm=arm,mode=mode,rule=rule,guard=guard,test=v,joint_test_pass=f.passed(v)))
            for rule,pick in picks.items():
                v=r.metrics(x['pred'],truth,action(x,rule,pick)) if pick['feasible'] else None
                candidates.append(dict(rep=item['rep'],budget=budget,arm=arm,mode=mode,rule=rule,development=pick,test=v,joint_test_pass=f.passed(v)))
        for rule,i in decisions.items():
            if i is None:v=None;pick=dict(feasible=False);mode=None
            else:
                x,picks=results[i];pick=picks[rule];mode=MODES[i];v=r.metrics(x['pred'],truth,action(x,rule,pick))
            selected.append(dict(rep=item['rep'],budget=budget,arm=arm,rule=rule,mode=mode,development=pick,test=v,joint_test_pass=f.passed(v)))
    r.write(dest/'FIXED.json',fixed);r.write(dest/'CANDIDATE.json',candidates);r.write(dest/'SELECTED.json',selected)
    r.write(dest/'COMPLETE.json',dict(rep=item['rep'],budget=budget,seconds=time.time()-start));print('SUPPORT',budget,item['rep'],round(time.time()-start,2),flush=True)

def aggregate():
    for budget in [54,60]:
        for key in ['FIXED','CANDIDATE','SELECTED']:
            rr=[v for p in sorted((O/f'dev{budget}'/'private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/(key+'.json')).read_text(encoding='utf8'))]
            r.write(O/f'dev{budget}'/(key+'_ROUNDS.json'),rr)
            keys=['arm','rule'] if key=='SELECTED' else ['arm','mode','rule']+(['guard'] if key=='FIXED' else [])
            r.write(O/f'dev{budget}'/(key+'_SUMMARY.json'),f.summarize(rr,keys))

def smoke(d,plan,items):
    item=items[58];f.draw(item,plan);sp=spec(item,60);arm='HZ_to_SA'
    configure('baseline');x=collect(d,sp,arm,item['seed']);old=np.load(r.OUT/'full60_control/private/round_58/HZ_to_SA.npz')
    np.testing.assert_allclose(x['pred'],old['prediction'],atol=1e-9,rtol=0)
    for k in f.RULES:np.testing.assert_allclose(x['scores'][k],old['score_'+k],atol=1e-9,rtol=0)
    for mode in MODES[1:]:
        configure(mode);x=collect(d,sp,arm,item['seed']);picks={k:choose(x,k) for k in f.RULES}
        truth=d['medical'][sp['test']].copy()
        try:
            d['medical'][sp['test']]=9999.;configure(mode);z=collect(d,sp,arm,item['seed'])
            np.testing.assert_allclose(x['pred'],z['pred'],atol=1e-9,rtol=0)
            for k in f.RULES:
                np.testing.assert_allclose(x['scores'][k],z['scores'][k],atol=1e-9,rtol=0);assert picks[k]==choose(z,k)
            for k in GUARDS:np.testing.assert_array_equal(x['guards'][k],z['guards'][k])
        finally:d['medical'][sp['test']]=truth
        # These controls leave the trained predictor unchanged within local support.
        train=np.asarray(sp['pool']);query=np.asarray(sp['test']);configure(mode);p=prediction(arm,900,train,query)[0];baseline=original_predict(arm,900,train,query)[0]
        b=d['x'][:,1];inside=(b[query]>=b[train].min())&(b[query]<=b[train].max())
        np.testing.assert_allclose(p[inside],baseline[inside],atol=1e-9,rtol=0)
    r.write(O/'SMOKE.json',dict(original_reproduced=True,test_label_poison_invariance=True,in_range_predictions_unchanged=True,train_only_support_bounds=True))
    print('SUPPORT SMOKE PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);a=ap.parse_args()
    if a.stage=='aggregate':aggregate();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    proto=dict(modes=MODES,guards=GUARDS,arms=ARMS,repeats=100,budgets=[54,60],cost=4,alpha=.01,split_sha256=r.sha(r.OUT/'private/SPLITS.json'),
        residual_clamp='Keep source prior at actual B; evaluate local residual correction at B clipped to current training min/max.',
        anchored_tails='Outside local train B range, cap lower-tail final prediction at lower endpoint prediction and floor upper-tail prediction at upper endpoint; inside range unchanged.',
        support='Only normal predictions require B inside current train range; abnormal predictions unchanged.',
        label_support='Only normal predictions require B>max training low B +m*training IQR and B<min training high B -m*training IQR; m=0,0.1,0.2. No query labels used.',
        selection='Per rule, use nested development predictions only: FN=0, accuracy>80%, nonzero abnormal denominator; maximize reporting rate then accuracy; ties earlier mapping, simpler guard. Reporting rate>75% checked separately. No feasible policy is NA.',
        scope='Exploratory follow-up to observed errors, not new independent validation. All candidates and all100 draws retained. Original paper and data unchanged.')
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==proto
    else:r.write(path,proto)
    if a.stage=='smoke':smoke(d,plan,items)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[a.start:a.stop]:run_one(item,d,plan,a.budget)
    original=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes']
    assert all(r.sha(r.current.ROOT/'data'/name)==h for name,h in original.items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
