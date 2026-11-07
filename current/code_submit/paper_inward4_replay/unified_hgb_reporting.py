"""Use the Table 2 HGB error gate consistently across observation times.

Reuse frozen predictions and HGB scores. The 85% development criterion is the
existing Table 2 selector (maximum development error 0.15). Higher criteria use
the same selector. No evaluation labels enter thresholds, ranking or timing.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

WS=Path(__file__).resolve().parents[2]
ROOT=WS/'outputs/current_ma_20261103'
OUT=ROOT/'unified_hgb_20261105'
PRIVATE=OUT/'private'
ARMS=['target_only_SA','HZ_to_SA','HZ+HB_to_SA']
TIMES=[5,10,15]
QUOTAS=[.10,.20,.40]
TARGETS=[.85,.90,.95]
INPUTS={}

def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(safe(v),indent=2)+'\n',encoding='utf8')
def read(p):
    INPUTS[str(p.relative_to(WS))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf8'))
def load(p):
    INPUTS[str(p.relative_to(WS))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def state(p):return np.where(~np.isfinite(p),9,np.where(p<50,-1,np.where(p<=70,0,1)))
def selector():
    p=WS/'outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py'
    INPUTS[str(p.relative_to(WS))]=hashlib.sha256(p.read_bytes()).hexdigest()
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['summary','select_threshold']]
    ns={'np':np,'MA_CLASSES':(-1,0,1),'Any':object}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns['select_threshold']
SELECT=selector()

def initialize():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    contract=dict(rule='Table 2 HGB error gate at every checkpoint',minutes=TIMES,coverage_levels=QUOTAS,agreement_levels=TARGETS,
        choice='HGB fixed before this time comparison, following the requested Table 2 alignment. No new model or hyperparameter search.',
        threshold='Original select_threshold: at least five releases and 20% development coverage; largest development coverage with error <= 1 - target. Target .85 reproduces Table 2 HGB exactly.',
        capability='Retain the existing increasing isotonic development score-to-error calibration for cross-time comparison, now using HGB scores at every checkpoint. For each record choose the available prediction with the lowest calibrated score. A 1e-6 training percentile breaks plateaus; equal scores retain the earlier prediction. Select floor(quota*N) by score then record identifier, as in Fig. 3. Records can be reselected.',
        cumulative='Apply the original HGB threshold selector to raw HGB scores separately at each checkpoint, then lock the first accepted prediction. No consensus restriction, boundary gate or test-label decision. Isotonic calibration is used only to compare prediction options in the fixed-coverage panels.',
        denominator='Coverage: all record-seed events. Agreement: released reports. FNR/FPR: released abnormal/normal reference events. Low/high states: MA <50 / >70 mm.',
        paper='Original 60 development / 201 evaluation records; 20 development partitions.',
        internal='Separate frozen 301-record grouped five-fold evaluation; one existing inner-partition setting. Do not pool with the paper cohort or the older three-repeat analysis.',
        uncertainty='Exploratory alignment after prior inspection. No independent validation or monotonic observed performance guarantee.',
        input='Stored input arrays and mapping predictions.')
    p=OUT/'PROTOCOL.json'
    if p.exists():assert read(p)==contract
    else:write(p,contract)

def threshold(dev,target):
    fit=SELECT(np.asarray(dev['score'],float),np.asarray(dev['pred']),np.asarray(dev['truth']),1-target)
    return fit.get('threshold')
def apply(score,pred,cutoff):
    return np.zeros(len(score),bool) if cutoff is None or not np.isfinite(cutoff) else np.isfinite(score)&np.isfinite(pred)&(score<=cutoff)
def quota(score,pred,ids,fraction):
    n=int(np.floor(fraction*len(ids)+1e-10));eligible=np.flatnonzero(np.isfinite(score)&np.isfinite(pred))
    assert len(eligible)>=n
    order=eligible[np.lexsort((np.asarray(ids).astype(str)[eligible],score[eligible]))]
    a=np.zeros(len(ids),bool);a[order[:n]]=True;return a
def calibrated_score(dev,query):
    from sklearn.isotonic import IsotonicRegression
    score=np.asarray(dev['score'],float);error=np.asarray(dev['pred'])!=np.asarray(dev['truth'])
    fit=np.isfinite(score);finite=np.isfinite(query);out=np.full(len(query),np.inf)
    xs=score[fit];ys=error[fit].astype(float)
    if not len(xs):return out
    risk=np.full(int(finite.sum()),ys.mean()) if len(np.unique(xs))==1 else IsotonicRegression(increasing=True,out_of_bounds='clip',y_min=0,y_max=1).fit(xs,ys).predict(query[finite])
    out[finite]=risk+1e-6*np.searchsorted(np.sort(xs),query[finite],side='right')/len(xs)
    return out
def decisions(query,development,ids):
    n=len(ids);hp=np.full(n,np.nan);hs=np.full(n,np.inf);ht=np.full(n,-1,int)
    seq={q:(np.full(n,np.nan),np.zeros(n,bool),np.full(n,-1,int)) for q in TARGETS}
    result={};fits={}
    for t in TIMES:
        p=np.asarray(query[t]['prediction']);s=np.asarray(query[t]['score'])
        calibrated=calibrated_score(development[t],s)
        update=np.isfinite(p)&np.isfinite(calibrated)&(calibrated<hs-1e-12)
        hp[update]=p[update];hs[update]=calibrated[update];ht[update]=t
        for q in QUOTAS:
            result['coverage',q,t]=(hp.copy(),quota(hs,hp,ids,q),ht.copy())
            result['current_coverage',q,t]=(p.copy(),quota(s,p,ids,q),np.full(n,t))
        for q in TARGETS:
            cutoff=threshold(development[t],q);a=apply(s,p,cutoff);fits[q,t]=cutoff
            result['pointwise',q,t]=(p.copy(),a.copy(),np.full(n,t))
            pp,aa,tt=seq[q];new=a&~aa
            pp[new]=p[new];tt[new]=t;aa|=new
            result['agreement_target',q,t]=(pp.copy(),aa.copy(),tt.copy())
    return result,fits

def metrics(pred,reference,action):
    p=np.asarray(pred);y=np.broadcast_to(reference,p.shape);a=np.asarray(action)&np.isfinite(p)&np.isfinite(y)
    pc=state(p);tc=state(y);classes=[-1,0,1]
    cm=np.array([[np.sum(a&(tc==i)&(pc==j)) for j in classes] for i in classes],int)
    count=int(a.sum());correct=int(np.trace(cm));ab=a&(tc!=0);normal=a&(tc==0)
    fn=int(np.sum(ab&(pc==0)));fp=int(np.sum(normal&(pc!=0)))
    ratio=lambda num,den:float(num/den) if den else None
    err=p[a]-y[a];sst=np.sum((y[a]-np.mean(y[a]))**2) if count else 0
    out=dict(events=int(p.size),reported=count,correct=correct,incorrect=count-correct,withheld=int(p.size-count),
        coverage=ratio(count,p.size),agreement=ratio(correct,count),fn=fn,fn_denominator=int(ab.sum()),fnr=ratio(fn,ab.sum()),
        fp=fp,fp_denominator=int(normal.sum()),fpr=ratio(fp,normal.sum()),confusion=cm.tolist(),
        all_true_class_counts=[int(np.sum(tc==i)) for i in classes],withheld_by_true_state=[int(np.sum(~a&(tc==i))) for i in classes],
        mae=float(np.mean(np.abs(err))) if count else None,rmse=float(np.sqrt(np.mean(err**2))) if count else None,
        r2=float(1-np.sum(err**2)/sst) if sst>0 else None,correct_yield=ratio(correct,p.size),incorrect_yield=ratio(count-correct,p.size))
    for i,label in [(0,'low'),(1,'normal'),(2,'high')]:
        mask=a&(tc==classes[i]);out[label+'_mae']=float(np.mean(np.abs(p[mask]-y[mask]))) if mask.any() else None
        out[label+'_recall']=ratio(cm[i,i],cm[i].sum());out[label+'_correct_yield_full_class']=ratio(cm[i,i],np.sum(tc==classes[i]))
    return out

def paper_context(arm,rep,data):
    query={};development={};test=None
    for t in TIMES:
        z=load(ROOT/'gates'/f'{arm}_T{t*60}.npz');v=read(ROOT/'gates'/f'{arm}_T{t*60}.development.json')[rep]
        if test is not None:np.testing.assert_array_equal(test,z['test'])
        test=z['test'];query[t]=dict(prediction=z['prediction'][rep],score=z['HGB_score'][rep])
        development[t]=dict(score=np.array([np.inf if x is None else x for x in v['scores']['HGB']]),pred=v['pred'],truth=v['truth'])
        np.testing.assert_array_equal(apply(query[t]['score'],query[t]['prediction'],threshold(development[t],.85)),z['HGB_action'][rep])
        train_ids={r['record_id'] for r in v['rows']};assert not train_ids&set(data['ids'][test])
    return query,development,test

def internal_context(arm,fi):
    from . import core as c,current_suite as suite,gates
    d,_,_,_,_,folds,local=c.context();f=folds[fi]
    z=load(ROOT/'early_subset_20261103/private'/f'all301_{arm}_fold{fi}.npz')
    train=z['dev_rows'];test=z['test'];development={};query={}
    assert not set(d['groups'][train])&set(d['groups'][test])
    for t in [5,10]:
        query[t]=dict(prediction=z[f'query_ma{t}'],score=z[f'query_HGB_{t}'])
        development[t]=dict(score=z[f'dev_HGB_{t}'],pred=z[f'dev_pred{t}'],truth=z['dev_truth'])
    cache=PRIVATE/f'internal15_{arm}_{fi}.npz'
    if cache.exists():
        q=load(cache)
    else:
        scores=[];preds=[];truth=[]
        for pi,sub in enumerate(f['inner']):
            inn=suite.subfolds(sub['T'],22000+fi*3+pi)
            out=c.evaluate(sub['T'],sub['V'],inn,arm,900);ss,_=gates.fit_score(out,900,1000+fi,arm,pi)
            scores.extend(ss['HGB']);preds.extend(state(out['prediction']));truth.extend(state(d['y'][out['query'],2]))
        out=c.evaluate(f['A'],f['B'],f['inner'],arm,900);ss,_=gates.fit_score(out,900,1000+fi,arm,3)
        q=dict(prediction=out['prediction'],score=ss['HGB'],dev_score=scores,dev_pred=preds,dev_truth=truth,test=test)
        np.savez_compressed(cache,**q);print('INTERNAL15',arm,fi,flush=True)
    query[15]=dict(prediction=q['prediction'],score=q['score']);development[15]=dict(score=q['dev_score'],pred=q['dev_pred'],truth=q['dev_truth'])
    for t in TIMES:
        saved=load(ROOT/'all301_gates'/f'{arm}_T{t*60}.npz');pos={int(v):i for i,v in enumerate(saved['local'])};ix=[pos[int(i)] for i in test]
        np.testing.assert_array_equal(saved['prediction'][ix],query[t]['prediction'])
        np.testing.assert_array_equal(saved['HGB'][ix],apply(query[t]['score'],query[t]['prediction'],threshold(development[t],.85)))
    return query,development,test

def smoke():
    data=load(ROOT/'data/DATA15_CURRENT.npz');q,dev,idx=paper_context('HZ+HB_to_SA',0,data)
    a,ta=decisions(q,dev,data['ids'][idx]);poison={t:{**v,'truth':np.full(len(idx),999)} for t,v in q.items()}
    b,tb=decisions(poison,dev,data['ids'][idx]);assert ta==tb
    for k in a:
        for x,y in zip(a[k],b[k]):np.testing.assert_array_equal(x,y)
    altered={t:dict(v) for t,v in q.items()};altered[15]=dict(prediction=np.full(len(idx),10000.),score=np.zeros(len(idx)))
    future,_=decisions(altered,dev,data['ids'][idx])
    for k in a:
        if k[2]<15:
            for x,y in zip(a[k],future[k]):np.testing.assert_array_equal(x,y)
    for target in TARGETS:
        old=np.zeros(len(idx),bool);pold=np.full(len(idx),np.nan)
        for t in TIMES:
            p,action,_=a['agreement_target',target,t]
            assert not (old&~action).any();np.testing.assert_array_equal(p[old],pold[old]);old=action;pold=p
    truth=np.array([45.,60.,75.,45.]);p=np.array([60.,75.,75.,45.]);m=metrics(p,truth,np.ones(4,bool))
    assert m['fn']==1 and m['fn_denominator']==3 and m['fp']==1 and m['fp_denominator']==1 and m['agreement']==.5
    write(OUT/'SMOKE.json',dict(existing_HGB_actions_reproduced=True,test_label_fields_ignored=True,no_future_scores_for_early_decisions=True,first_reports_locked=True,metrics_denominators=True))
    print('SMOKE_PASS',flush=True)

def evaluate(protocol):
    assert (OUT/'SMOKE.json').exists()
    data=load(ROOT/'data/DATA15_CURRENT.npz');folds=read(ROOT/'data/FROZEN_SPLITS.json');rows=[];per=[];thresholds=[]
    for arm in ARMS:
        contexts=range(20) if protocol=='paper' else range(5)
        idx=load(ROOT/'gates'/f'{arm}_T900.npz')['test'] if protocol=='paper' else np.sort(np.r_[folds[0]['A'],folds[0]['B']])
        where={int(v):i for i,v in enumerate(idx)};reps=20 if protocol=='paper' else 1;n=len(idx);arrays={}
        seen=np.zeros((reps,n),int)
        for k in contexts:
            q,dev,test=paper_context(arm,k,data) if protocol=='paper' else internal_context(arm,k)
            decisions_out,fit=decisions(q,dev,data['ids'][test]);rep=k if protocol=='paper' else 0;jj=[where[int(i)] for i in test];seen[rep,jj]+=1
            for key,(p,a,when) in decisions_out.items():
                if key not in arrays:arrays[key]=(np.full((reps,n),np.nan),np.zeros((reps,n),bool),np.full((reps,n),-1,int))
                for dest,src in zip(arrays[key],(p,a,when)):dest[rep,jj]=src
            thresholds.extend(dict(protocol=protocol,arm=arm,repeat=rep,fold=k if protocol!='paper' else 0,target=target,minute=t,threshold=v) for (target,t),v in fit.items())
        assert (seen==1).all()
        for (mode,target,t),(p,a,when) in arrays.items():
            m=metrics(p,data['medical'][idx,2],a)
            row=dict(protocol=protocol,arm=arm,rule='HGB',mode=mode,target=target,minute=t,repeats=reps,**m)
            if mode in ['agreement_target','pointwise']:row['actual_target_met']=m['agreement'] is not None and m['agreement']+1e-12>=target
            row['prediction_source_counts']={str(t0):int(np.sum(a&(when==t0))) for t0 in TIMES}
            row['using_earlier_prediction']=int(np.sum(a&(when<t)));rows.append(row)
            for rep in range(reps):per.append(dict(protocol=protocol,arm=arm,mode=mode,target=target,minute=t,repeat=rep,**metrics(p[rep],data['medical'][idx,2],a[rep])))
            np.savez_compressed(PRIVATE/f'{protocol}_{arm}_{mode}_{target:.2f}_{t}.npz',prediction=p,action=a,source_time=when,test=idx)
        print('ARM_COMPLETE',protocol,arm,flush=True)
    write(OUT/f'{protocol}_METRICS.json',rows);write(OUT/f'{protocol}_PER_REPEAT.json',per);write(PRIVATE/f'{protocol}_THRESHOLDS.json',thresholds)
    write(OUT/f'{protocol}_INPUT_MANIFEST.json',INPUTS)
    for row in rows:
        if row['arm']=='HZ+HB_to_SA' and row['mode'] in ['coverage','agreement_target','pointwise']:
            print({k:row[k] for k in ['mode','target','minute','coverage','agreement','fnr','reported','incorrect']},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','paper','internal301']);arg=parser.parse_args();initialize()
    if arg.stage=='smoke':smoke()
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):evaluate(arg.stage)
