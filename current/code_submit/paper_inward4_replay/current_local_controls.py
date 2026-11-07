"""Additional local301 budget/calibration controls, using frozen outer folds."""
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c, current_suite as suite, mapping
from .gates import clean_write

ROOT=suite.ROOT


def budget_rows(rows, count, d):
    # Deterministic group sampling, with no label-based enrichment. Exact label
    # budgets are retained when possible without splitting an identity group.
    groups=sorted(set(d['groups'][rows]), key=lambda g:hashlib.sha256(
        ('current301-budget-v1|'+str(g)).encode()).hexdigest())
    chosen=[]
    for group in groups:
        block=[int(i) for i in rows if d['groups'][i]==group]
        if len(chosen)+len(block)<=count:chosen.extend(block)
        if len(chosen)==count:break
    assert len(chosen)==count
    return np.sort(chosen)


def run():
    suite.initialize();m=suite.numerical_models()
    d,_,_,_,_,folds,local=c.context();pos={int(v):i for i,v in enumerate(local)}
    out=ROOT/'all301_local_controls';out.mkdir(exist_ok=True)
    protocol=dict(budgets=[20,40,60],full='original outer train,240or241',
        selection='Fixed identity-group hash order, no outcome enrichment; same rows for all arms. Original inner folds restricted to the selected pool.',
        calibration=['raw','affine','joint','huber','complete'],
        auxiliary='Source Ridge supplies R,K,angle; B-only source prior supplies MA. Complete updates MA only.',
        data_sha256=c.sha(ROOT/'data/DATA15_CURRENT.npz'),B_from_original_data=True,valid_UTEG_only_validation=False)
    path=out/'PROTOCOL.json'
    if path.exists():assert c.read(path)==protocol
    else:clean_write(path,protocol)
    for budget in protocol['budgets']:
        path=out/f'budget{budget}.npz'
        if path.exists():continue
        prediction=np.full((3,len(local)),np.nan);accepted=np.zeros_like(prediction,bool);audits=[]
        for fi,f in enumerate(folds):
            train=budget_rows(np.asarray(f['A'],int),budget,d);keep=set(train)
            inner=[dict(T=[i for i in s['T'] if i in keep],V=[i for i in s['V'] if i in keep]) for s in f['inner']]
            ix=[pos[i] for i in f['B']]
            for j,arm in enumerate(c.MAIN):
                v=c.evaluate(train,f['B'],inner,arm,900)
                prediction[j,ix]=v['prediction'];accepted[j,ix]=v['accepted']
                audits.append(dict(fold=fi,arm=arm,train=train,inner=inner,
                    class_counts=[int(np.sum(c.accounting.states(d['y'][train,2])==k)) for k in [-1,0,1]],fits=v['audits']))
        np.savez_compressed(path,prediction=prediction,accepted=accepted,local=local)
        clean_write(path.with_suffix('.audit.json'),audits)
        clean_write(path.with_suffix('.metrics.json'),{arm:dict(all_report=c.summarize(prediction[j],d['y'][local,2]),
            gate=c.summarize(prediction[j],d['y'][local,2],accepted[j])) for j,arm in enumerate(c.MAIN)})
        print('CURRENT301_BUDGET_COMPLETE',budget,flush=True)
    for arm in c.MAIN[1:]:
        path=out/f'calibration_{arm}.npz'
        if path.exists():continue
        arrays={k:np.full((len(local),4),np.nan) for k in protocol['calibration']};audits=[]
        source=c.source_model(arm,900);aux=mapping.source(arm,'RIDGE-10',1000);sr=c.source_rows(arm,900)
        scale={'ym':d['medical'][sr].mean(0),'ys':np.maximum(d['medical'][sr].std(0),1e-6)}
        for fi,f in enumerate(folds):
            a=np.asarray(f['A']);b=np.asarray(f['B']);ix=[pos[int(i)] for i in b]
            x=c.feature(a,b,900);x4=c.feature(a,b,900,'four')
            pt=m.predict(aux,x4[a]);pq=m.predict(aux,x4[b]);pt[:,2]=c.source_predict(source,x[a]);pq[:,2]=c.source_predict(source,x[b])
            ca={k:mapping.fit_cal(pt,d['medical'][a],scale,k) for k in ['affine','joint']}
            hc=m.fit_cal(pt,d['medical'][a],scale['ym'],scale['ys']);assert hc['finite']
            complete=pq.copy();complete[:,2]=c.predict(arm,900,a,b)[0]
            vals=dict(raw=pq,affine=mapping.pred_cal(ca['affine'],pq),joint=mapping.pred_cal(ca['joint'],pq),
                huber=m.project(m.predict_cal(hc,pq)),complete=complete)
            for k,v in vals.items():arrays[k][ix]=v
            audits.append(dict(fold=fi,train=a,query=b,huber=hc))
        with np.load(ROOT/'all301/T900.npz') as z:
            np.testing.assert_array_equal(arrays['complete'][:,2],z['prediction'][c.MAIN.index(arm)])
        np.savez_compressed(path,**arrays,local=local)
        clean_write(path.with_suffix('.audit.json'),audits)
        clean_write(path.with_suffix('.metrics.json'),{k:c.summarize(v[:,2],d['y'][local,2]) for k,v in arrays.items()})
        print('CURRENT301_CALIBRATION_COMPLETE',arm,flush=True)
    clean_write(out/'COMPLETE.json',dict(budget_bundles=3,calibration_bundles=2,full_primary_exact_reproduced=True))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
