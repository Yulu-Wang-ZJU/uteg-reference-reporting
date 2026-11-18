"""Controlled member-exchange diagnosis and fixed development-only normal guards."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import os,sys,json,time,importlib.util,csv
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
P=Path(str(_repeat_root / 'experiments' / 'uteg_sahzu60_test244_20261118'))
spec=importlib.util.spec_from_file_location('sahzu_extension_runner',P/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
import numpy as np
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent
r=q.r;c=q.c;e=q.e
VARIANTS=['baseline','symmetric90','prediction_bounds','both']

def diagnosis(d):
    old=json.loads(q.OLD.read_text());role=json.loads((r.current.ROOT/'data/ROLE_FREEZE.json').read_text())['roles']['SAHZU']
    original=np.array(role['pool_full']);proposal=json.loads((P/'SPLIT_AUDIT_AND_PROPOSAL.json').read_text())['candidate']
    index={str(v):i for i,v in enumerate(d['ids'])}
    changes=list(zip([index[v] for v in proposal['removed']],[index[v] for v in proposal['added']]))
    variants={}
    for name in ['original','normal_swaps_only','low_swap_only','all_three_swaps']:
        pairs=[(a,b) for a,b in changes if name=='all_three_swaps' or (name=='low_swap_only' and d['medical'][a,2]<50) or (name=='normal_swaps_only' and 50<=d['medical'][a,2]<=70)]
        variants[name]=np.sort(list(set(original)-{a for a,b in pairs}|{b for a,b in pairs}))
    assert np.array_equal(variants['all_three_swaps'],proposal['development'])
    rows=[];details=[]
    for item in old:
        q.f.activate(item,d)
        for arm in q.ARMS:
            for name,train in variants.items():
                pred,_,_=c.predict(arm,900,train,[5]);p=float(pred[0])
                rows.append(dict(rep=item['rep'],arm=arm,variant=name,prediction=p,reference=float(d['medical'][5,2])))
                if item['rep']==0:
                    # Compare singleton query with a group-disjoint full query batch.
                    allsa=np.flatnonzero(d['centers']=='SAHZU');query=allsa[~np.isin(d['groups'][allsa],d['groups'][train])]
                    batch,_,_=c.predict(arm,900,train,query)
                    np.testing.assert_allclose(batch[np.flatnonzero(query==5)],pred,atol=1e-9,rtol=0)
                    sp=dict(pool=train,test=query,positions=q.positions(d,train,query,item['seed']))
                    it=dict(item,targets={'SAHZU':sp});q.f.activate(it,d)
                    z,dr,qr,au,rr,picks,act=q.f.bundle(d,it,arm,900)
                    j=int(np.flatnonzero(z['test']==5)[0]);pc=r.state(z['dev_prediction'])
                    details.append(dict(arm=arm,variant=name,prediction=p,accepted=bool(act['common_B'][j]),risk=qr[j],
                        normal_gate=picks['common_B']['normal'],dev_low=[dict(record=str(d['ids'][i]),B=float(d['x'][i,1]),MA=float(d['medical'][i,2]),oof=float(v)) for i,v in zip(z['dev_rows'],z['dev_prediction']) if d['medical'][i,2]<50]))
                    if name in ['original','all_three_swaps']:
                        path=(e.O/'nine/round_00'/('full_local_'+arm+'.npz')) if name=='original' else P/'private/dev60/round_000'/(arm+'.npz')
                        with np.load(path) as saved:
                            k=np.flatnonzero(saved['test']==5);np.testing.assert_allclose(p,saved['prediction'][k],atol=1e-9,rtol=0)
        if item['rep']%20==0:print('DIAGNOSIS',item['rep']+1,flush=True)
    r.write(O/'DIAGNOSIS.json',dict(repetitions=100,rows=rows,first_repeat_details=details,
        interpretation='Controlled swaps are diagnostic only, not alternate selected evaluation cohorts. Query record label is never supplied to model fitting.'))

def fit_guard(z,d):
    p=z['dev_prediction'];truth=d['medical'][z['dev_rows'],2];norm=r.state(p)==1
    return dict(low_risk90=float(np.quantile(z['dev_risk'][norm,0],.9)) if norm.any() else None,
        lower=max(50.,float(np.max(p[truth<50]))) if np.any(truth<50) else None,
        upper=min(70.,float(np.min(p[truth>70]))) if np.any(truth>70) else None)
def apply_guard(name,p,risk,base,g):
    normal=r.state(p)==1;ok=np.ones(len(p),bool)
    if name in ['symmetric90','both']:
        ok&=False if g['low_risk90'] is None else risk[:,0]<=g['low_risk90']
    if name in ['prediction_bounds','both']:
        ok&=False if g['lower'] is None or g['upper'] is None else (p>g['lower'])&(p<g['upper'])
    return np.asarray(base,bool)&(~normal|ok)

def controls(d):
    rows=[];policies=[];checks=0;events=[]
    for budget in [60,54]:
        for rep in range(100):
            folder=P/'private'/f'dev{budget}'/f'round_{rep:03d}'
            for arm in q.ARMS:
                with np.load(folder/(arm+'.npz')) as archive:z={k:archive[k].copy() for k in archive.files}
                saved=json.loads((folder/(arm+'.json')).read_text())['policies']['common_B']
                ds={k:z['dev_score_'+k] for k in e.a.RULES};drisk=np.c_[-d['x'][z['dev_rows'],1],d['x'][z['dev_rows'],1]]
                base_dev=e.cal.action(saved,z['dev_prediction'],z['dev_risk'],drisk,ds) if saved['feasible'] else np.zeros(len(z['dev_rows']),bool)
                g=fit_guard(z,d);q.separate(d,z['dev_rows'],z['test'])
                # Smoke guard fitting with every nondevelopment reference altered.
                if rep==0:
                    altered=dict(d,medical=d['medical'].copy());forbidden=np.setdiff1d(np.arange(len(d['x'])),z['dev_rows']);altered['medical'][forbidden]=12345
                    assert g==fit_guard(z,altered)
                    reversedz=dict(z,prediction=z['prediction'][::-1],test_risk=z['test_risk'][::-1]);assert g==fit_guard(reversedz,d)
                    checks+=1
                policies.append(dict(budget=budget,rep=rep,arm=arm,**g))
                for name in VARIANTS:
                    da=apply_guard(name,z['dev_prediction'],z['dev_risk'],base_dev,g)
                    a=apply_guard(name,z['prediction'],z['test_risk'],z['action_common_B'],g)
                    assert not np.any(a&~z['action_common_B'])
                    assert np.array_equal(a[r.state(z['prediction'])!=1],z['action_common_B'][r.state(z['prediction'])!=1])
                    met=e.n.extended_metrics(z['prediction'],z['reference'],a)
                    dev=e.n.extended_metrics(z['dev_prediction'],d['medical'][z['dev_rows'],2],da)
                    assert dev['FN']==0
                    rows.append(dict(budget=budget,rep=rep,arm=arm,rule=name,test=met,development=dev))
                    fn=a&(r.state(z['prediction'])==1)&(r.state(z['reference'])!=1)
                    for j in np.flatnonzero(fn):events.append(dict(budget=budget,rep=rep,arm=arm,rule=name,record=str(d['ids'][z['test'][j]]),MA=float(z['reference'][j]),prediction=float(z['prediction'][j])))
            if rep%25==0:print('GUARDS',budget,rep+1,flush=True)
    r.write(O/'ROUNDS.json',rows);r.write(O/'POLICIES.json',policies);r.write(O/'FN_EVENTS.json',events)
    r.write(O/'CHECKS.json',dict(passed=True,forbidden_label_and_query_invariance_checks=checks,all_dev_FN_zero=True,only_normal_release_tightened=True,all100_retained=True,no_test_threshold_selection=True))
    summary=[]
    for budget in [60,54]:
        for arm in q.ARMS:
            for name in VARIANTS:
                rr=[v for v in rows if (v['budget'],v['arm'],v['rule'])==(budget,arm,name)];assert len(rr)==100
                v=dict(budget=budget,arm=arm,rule=name,FN_rounds=sum(x['test']['FN']>0 for x in rr),FN_events=sum(x['test']['FN'] for x in rr),
                    no_normal_rounds=sum(x['test']['reported_as_normal']==0 for x in rr),dev_accuracy_below80=sum(x['development']['accuracy'] is None or x['development']['accuracy']<=.8 for x in rr))
                for k in ['accuracy','reporting_rate','FNR','FPR','MAE','R2']:
                    a=np.array([x['test'][k] for x in rr if x['test'][k] is not None]);v[k+'_mean']=float(a.mean()) if len(a) else None;v[k+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None;v[k+'_n']=len(a)
                summary.append(v)
    r.write(O/'SUMMARY.json',summary)
    with (O/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as s:
        w=csv.DictWriter(s,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('DONE',len(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        d=q.init();c.OUT=O/'runtime';r.suite.ROOT=O/'runtime'
        r.write(O/'PROTOCOL.json',dict(primary_candidate='symmetric90',controls=['prediction_bounds','both'],
            fixed='Existing mapping and base gate decisions unchanged; supplementary guards only withhold predicted-normal reports; no forced abnormal reassignment.',
            symmetric90='Apply low-MA risk90th percentile of development OOF predicted-normal records, mirroring existing high-risk90 cutoff; no percentile search.',
            prediction_bounds='Release normal only above maximum OOF prediction among development true-low and below minimum among true-high, intersected with clinical normal interval; no test-reference-derived margin.',
            selection='All fixed candidates and every repeat reported; no best-test candidate silently substituted. Developer metrics reported after normal-only tightening.',
            diagnosis='Four controlled training memberships; original / normal swaps only / low swap only / all swaps. Diagnostic, not optimization of cohort members.',
            paper_unchanged=True))
        before=q.f.frozen_files();before.update({str(p):r.sha(p) for p in P.glob('*.json')})
        diagnosis(d);controls(d)
        assert all(r.sha(Path(p))==h for p,h in before.items());r.write(O/'PROTECTED_HASHES.json',before)
