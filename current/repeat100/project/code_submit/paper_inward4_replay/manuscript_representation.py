"""Matched downstream EXP3/EXP4 refit controls on the acquired U-TEG signals."""
import argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,gates,analysis

OUT=suite.ROOT/'manuscript_sync/experiments/representation'

def activate(rep):
    suite.initialize();d,*_=c.context()
    z=analysis.load(suite.ROOT/'manuscript_sync/experiments/signal/REFITS.npz')
    x=z[rep].copy();qc=z['qc'].astype(bool)&d['input_qc'].astype(bool)
    np.testing.assert_array_equal(z['ids'],d['ids'])
    def features(train,query,h,representation='B'):
        return x[:,{'B':[1],'three':[0,1,2],'four':[0,1,2,3]}[representation]].copy()
    c.feature=features;c.source_feature=lambda direction,h,representation='B':features([],[],h,representation)
    plan=c.context()[3]
    def rows(direction,h):
        r=np.asarray(plan['source_rows'].get(direction,[]),int);return r[qc[r]]
    c.source_rows=rows;c.qc_at=lambda h:qc;c._SOURCE.clear()
    return 'three' if rep=='EXP3' else 'four'

def one(train,test,parts,arm,seed,rep):
    d,*_=c.context();dev={k:[] for k in ['LR','HGB','DUAL','BOUNDARY']};dp=[];dt=[];rows=[];timing=[]
    for pos,f in enumerate(parts[:3]):
        t=time.perf_counter();out=c.evaluate(f['A'],f['B'],f['inner'],arm,900,representation=rep)
        ss,_=gates.fit_score(out,900,seed,arm,pos);timing.append(time.perf_counter()-t)
        pc=gates.fr.ma_class(out['prediction']);tc=gates.fr.ma_class(d['y'][out['query'],2])
        for key in dev:dev[key].extend(ss[key])
        dp.extend(pc);dt.extend(tc)
        rows.extend(dict(record_id=str(d['ids'][i]),pred_class=int(pc[j]),truth_class=int(tc[j]),
            base_score=float(ss['LR'][j]) if np.isfinite(ss['LR'][j]) else None,
            dual_score=float(ss['DUAL'][j]) if np.isfinite(ss['DUAL'][j]) else None) for j,i in enumerate(out['query']))
    t=time.perf_counter();out=c.evaluate(train,test,parts[3]['inner'],arm,900,representation=rep)
    ss,model=gates.fit_score(out,900,seed,arm,3);timing.append(time.perf_counter()-t)
    dual=gates.fr.choose(rows);valid=np.isfinite(out['prediction']);pc=gates.fr.ma_class(out['prediction'])
    th={k:gates.rc.select_threshold(np.asarray(dev[k]),np.asarray(dp),np.asarray(dt),.15) for k in ['LR','HGB','BOUNDARY']}
    actions={}
    for name,key in [('LR','LR'),('HGB','HGB'),('BOUNDARY_PAPER','BOUNDARY')]:
        v=th[key].get('threshold');actions[name]=np.zeros(len(test),bool) if v is None else valid&np.isfinite(ss[key])&(ss[key]<=v)
    actions['TWO_SIDED']=gates.fr.action(dual,pc,ss['LR'],ss['DUAL'],valid)
    actions['BOUNDARY95']=out['accepted']
    return dict(prediction=out['prediction'],**{k+'_action':v for k,v in actions.items()},**{k+'_score':v for k,v in ss.items()}),dict(seed=seed,thresholds=th,dual=dual,support=model['support'],seconds=timing)

def smoke():
    rep=activate('EXP3');d,*_=c.context();sp=c.paper_spec('HZ_to_SA',1000);f=sp['positions'][0]
    a=c.evaluate(f['A'],f['B'],f['inner'],'HZ_to_SA',900,representation=rep)
    original=d['y'].copy()
    try:
        d['y'][f['B']]=1e6;c._SOURCE.clear()
        b=c.evaluate(f['A'],f['B'],f['inner'],'HZ_to_SA',900,representation=rep)
        np.testing.assert_array_equal(a['prediction'],b['prediction'])
        np.testing.assert_array_equal(a['accepted'],b['accepted'])
    finally:d['y'][:]=original
    OUT.mkdir(parents=True,exist_ok=True)
    gates.clean_write(OUT/'SMOKE.json',dict(evaluation_labels_cannot_affect_fit_or_gate=True,EXP3_mu_fixed=1,not_column_deletion=True))
    print('REPRESENTATION_SMOKE_PASSED',flush=True)

def run(rep):
    kind=activate(rep);OUT.mkdir(parents=True,exist_ok=True);assert (OUT/'SMOKE.json').exists();d,*_=c.context()
    for arm in c.TARGET:
        if arm.startswith('target_only'):continue
        path=OUT/f'{rep}_{arm}.npz'
        if path.exists():continue
        acc=[];audits=[]
        for seed in c.SEEDS:
            small=OUT/'private'/f'{rep}_{arm}_{seed}.npz';small.parent.mkdir(exist_ok=True)
            sp=c.paper_spec(arm,seed)
            if small.exists():values=analysis.load(small);au=c.read(small.with_suffix('.json'))
            else:
                values,au=one(sp['pool'],sp['test'],sp['positions'],arm,seed,kind)
                np.savez_compressed(small,**values);gates.clean_write(small.with_suffix('.json'),au)
            acc.append(values);audits.append(au)
        np.savez_compressed(path,**{k:np.asarray([v[k] for v in acc]) for k in acc[0]},test=sp['test'],seed=c.SEEDS)
        gates.clean_write(path.with_suffix('.audit.json'),audits)
        gates.clean_write(path.with_suffix('.metrics.json'),{k:c.summarize(np.array([v['prediction'] for v in acc]),d['y'][sp['test'],2],None if k=='ALL_REPORT' else np.array([v[k+'_action'] for v in acc])) for k in ['ALL_REPORT','BOUNDARY95','LR','TWO_SIDED']})
        print('REPRESENTATION_PAPER_COMPLETE',rep,arm,flush=True)
    folds=c.context()[5];local=c.context()[6];where={int(v):i for i,v in enumerate(local)}
    for arm in c.MAIN:
        path=OUT/f'{rep}_all301_{arm}.npz'
        if path.exists():continue
        accum={};audits=[]
        for fi,f in enumerate(folds):
            parts=[dict(A=s['T'],B=s['V'],inner=suite.subfolds(s['T'],44000+fi*3+j)) for j,s in enumerate(f['inner'])]
            parts.append(dict(inner=f['inner']))
            v,au=one(f['A'],f['B'],parts,arm,1000+fi,kind);ix=[where[i] for i in f['B']]
            for k,a in v.items():
                if k not in accum:accum[k]=np.zeros(len(local),dtype=a.dtype)
                accum[k][ix]=a
            audits.append(au)
        np.savez_compressed(path,**accum,local=local);gates.clean_write(path.with_suffix('.audit.json'),audits)
        gates.clean_write(path.with_suffix('.metrics.json'),{k:c.summarize(accum['prediction'],d['y'][local,2],None if k=='ALL_REPORT' else accum[k+'_action']) for k in ['ALL_REPORT','BOUNDARY95','LR','TWO_SIDED']})
        print('REPRESENTATION_301_COMPLETE',rep,arm,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','EXP3','EXP4']);a=ap.parse_args()
    with threadpool_limits(limits=1):
        if a.stage=='smoke':smoke()
        else:run(a.stage)
