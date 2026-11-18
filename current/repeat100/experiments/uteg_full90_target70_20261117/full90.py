"""Full-centre stratified 90% sampling, target 70/30, locked estimators.

Private research output only. No manuscript or previous experiment mutation.
"""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import os, sys, json, time, argparse, hashlib, csv, traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode = True
WS=Path(str(_repeat_root / 'project' / ''))
O=Path(__file__).resolve().parent
sys.path.insert(0,str(WS))
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_locked_expansion_20261117'))
import experiment as e
import fast_policy
import numpy as np
from threadpoolctl import threadpool_limits
r=e.r; c=r.c
CENTRES=['SAHZU','Huzhou','Hubei']
SHORT={'SA':'SAHZU','HZ':'Huzhou','HB':'Hubei'}
ARMS=list(c.TARGET)
TIMES=[300,600,900]
BASE_SEED=202611170
ORIGINAL_NORMAL=e.cal.choose_normal
e.cal.choose_normal=fast_policy.choose_normal

def init():
    # activate loads the current dataset/prefix arrays, but does not fit or write.
    d,*_=r.current.activate()
    c.OUT=O/'runtime'; r.suite.ROOT=O/'runtime'
    e.a.configure('B')
    return d

def quota(ix,d,fraction):
    ct=np.bincount(r.state(d['medical'][ix,2]),minlength=3)
    n=int(np.floor(fraction*len(ix)+.5)); ideal=ct*n/len(ix)
    q=np.floor(ideal).astype(int)
    for k in sorted(range(3),key=lambda k:(-(ideal[k]-q[k]),k))[:n-int(q.sum())]:q[k]+=1
    return q

def sample(ix,d,rng,fraction):
    ix=np.asarray(ix,int); q=quota(ix,d,fraction); ys=r.state(d['medical'][:,2]); gg=d['groups'][ix]
    units=[ix[gg==g] for g in np.unique(gg)]
    multi=[u for u in units if len(u)>1]
    singles=[np.array([u[0] for u in units if len(u)==1 and ys[u[0]]==k],int) for k in range(3)]
    # Reject only infeasible group allocations, never according to predictions.
    for attempt in range(10000):
        take=[u for u in multi if rng.random()<fraction]
        used=np.concatenate(take) if take else np.array([],int)
        rem=q-np.bincount(ys[used],minlength=3)
        if any(rem[k]<0 or rem[k]>len(singles[k]) for k in range(3)):continue
        out=np.sort(np.r_[used,*[rng.choice(singles[k],int(rem[k]),replace=False) for k in range(3)]])
        assert np.array_equal(np.bincount(ys[out],minlength=3),q)
        assert not set(d['groups'][out])&set(d['groups'][np.setdiff1d(ix,out)])
        return out
    raise RuntimeError('Group-respecting quota is not feasible; do not change quotas silently')

def source_rows(item,arm,h,d):
    if arm.startswith('target_only'):return np.array([],int)
    names=arm.split('_to_')[0].split('+')
    ix=np.asarray(sum([item['sources'][SHORT[n]] for n in names],[]),int)
    assert all(d['centers'][ix]!=c.TARGET[arm])
    assert c.qc_at(h)[ix].all()
    return ix

def activate(item,d):
    c._SOURCE.clear()
    c.source_rows=lambda arm,h:source_rows(item,arm,h,d)

def frozen_files():
    files=[r.current.ROOT/'data'/n for n in ['DATA15_CURRENT.npz','CURRENT_SCALED_CURVES.npz']]
    files += [e.LOCK/'SELECTED_POLICY.json',e.LOCK/'FILE_MANIFEST.json']
    files += list((e.O/'summary').glob('*'))
    manuscript=WS/'outputs/uteg_nc_goal_20261030/manuscript'
    files += [manuscript/n for n in ['main.tex','supplement.tex','main.pdf','supplement.pdf','Supplementary_Materials.zip']]
    return {str(p):r.sha(p) for p in files if p.is_file()}

def audit_splits(d,items):
    counts=[]; all_checks=0
    expected={'SAHZU':([11,230,33],[8,161,23],[3,69,10]),'Huzhou':([27,59,0],[19,41,0],[8,18,0]),'Hubei':([13,33,25],[9,23,18],[4,10,7])}
    for item in items:
        for ct in CENTRES:
            allix=np.flatnonzero(d['centers']==ct); src=np.asarray(item['sources'][ct]);sp=item['targets'][ct]
            pool=np.asarray(sp['pool']);test=np.asarray(sp['test']);unused=np.setdiff1d(allix,src)
            assert set(pool)|set(test)==set(src)
            assert not set(pool)&set(test)
            for p,q in [(pool,test),(src,unused)]:assert not set(d['groups'][p])&set(d['groups'][q])
            assert len(src)==len(set(src))
            for name,ix,exp in zip(['selected90','development70','test30'],[src,pool,test],expected[ct]):
                cls=np.bincount(r.state(d['medical'][ix,2]),minlength=3).tolist();assert cls==exp,(ct,name,cls)
                counts.append(dict(rep=item['rep'],centre=ct,role=name,n=len(ix),low=cls[0],normal=cls[1],high=cls[2]))
            outer=[]
            for pos,part in enumerate(sp['positions']):
                a=np.asarray(part['A']);b=np.asarray(part['B'])
                assert not set(d['groups'][a])&set(d['groups'][b])
                if pos<3:assert set(a)|set(b)==set(pool);outer.extend(b.tolist())
                else:assert set(a)==set(pool) and set(b)==set(test)
                inner=[]
                for fold in part['inner']:
                    tr=np.asarray(fold['T']);va=np.asarray(fold['V'])
                    assert set(tr)|set(va)==set(a)
                    assert not set(d['groups'][tr])&set(d['groups'][va])
                    assert not set(tr)&set(test)
                    inner.extend(va.tolist());all_checks+=1
                assert sorted(inner)==sorted(a)
            assert sorted(outer)==sorted(pool)
        for arm in ARMS:
            src=source_rows(item,arm,900,d);sp=item['targets'][c.TARGET[arm]]
            assert not set(d['groups'][src])&set(d['groups'][np.r_[sp['pool'],sp['test']]])
    return counts,all_checks

def prepare(d):
    path=O/'private/SPLITS.json'
    if path.exists():return json.loads(path.read_text())
    assert len(d['x'])==478 and d['input_qc'].all()
    for h in TIMES:assert c.qc_at(h).all() and np.isfinite(c.feature([],[],h,'four')).all()
    items=[]
    for rep in range(100):
        seed=BASE_SEED+rep;rng=np.random.default_rng(seed);sources={};targets={}
        for ct in CENTRES:
            src=sample(np.flatnonzero(d['centers']==ct),d,rng,.9)
            pool=sample(src,d,rng,.7);test=np.setdiff1d(src,pool)
            positions=[]
            for j,f in enumerate(r.folds(pool,d,seed)):
                positions.append(dict(A=f['T'],B=f['V'],inner=r.folds(f['T'],d,seed+101+j)))
            positions.append(dict(A=pool,B=test,inner=r.folds(pool,d,seed+201)))
            sources[ct]=src.tolist();targets[ct]=dict(pool=pool,test=test,positions=positions)
        items.append(r.safe(dict(rep=rep,seed=seed,sources=sources,targets=targets)))
    counts,checks=audit_splits(d,items)
    r.write(path,items);r.write(O/'SAMPLE_COUNTS.json',counts)
    r.write(O/'SPLIT_AUDIT.json',dict(passed=True,repeats=100,inner_splits_checked=checks,group_disjoint=True,exact_class_quotas=True,splits_sha256=r.sha(path)))
    r.write(O/'PROTECTED_FILES.json',frozen_files())
    # Snapshot the exact loaded numerical source definitions, without data.
    codes={}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if not f:continue
        p=Path(f)
        if p.suffix=='.py' and p.is_file() and (str(WS).lower() in str(p).lower() or str(p).replace('\\','/').startswith(str(_repeat_root / 'experiments' / ''))):
            sha=r.sha(p);codes[str(p)]=sha
            dest=O/'code_snapshot'/f'{sha}_{p.name}';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    r.write(O/'CODE_MANIFEST.json',codes)
    r.write(O/'PROTOCOL.json',dict(version='full-centre90_target70_v1',centres=CENTRES,repeats=100,seeds=[BASE_SEED,BASE_SEED+99],arms=ARMS,times_seconds=TIMES,
        sampling='Entire eligible centre, low<50/normal50-70/high>70 stratification; round-half-up total and largest-remainder class quotas. Groups stay intact. Target70/30 split after90% draw. Unselected10% unused in that repeat.',
        matching='Same selected centre pools, target development/test roles and nested folds across arms and times within each repeat. Source-only centres used in their entirety after90% sampling. All source models refitted; no old prediction reuse.',
        fitting='Existing B-input RBF, residual inward4 cost,alpha0.01,training class weights,anchored-tail correction. Nested3x3 development scoring; final fit on full development. Existing Boundary/Logistic/HGB/B95 and locked common_B normal check with90th percentile cap/reserve0. No hyperparameter changes.',
        primary_policy='common_B; other rules are unchanged controls, never selected using test performance.',
        scope='MA mapping and reporting:9transfer directions+3target-only,3timepoints,100repeats. No manuscript rewrite, no old60/54 overwrite.',
        metrics='accuracy/FNR/FPR conditional on released records; reporting_rate uses complete target test set. All-report errors and recalls also reported. FN=abnormal predictednormal, FP=normal predictedabnormal. Missing denominators remain undefined.',
        statistics='Mean and sampleSD across all100repeats; not95%confidence intervals or100 independent cohorts. Infeasible policies retained with0reports and undefinedconditionalmetrics.',
        source_provenance='Use current active input files as supplied. This experiment audits partition/fit isolation, not independent verification of the upstream measurement provenance.'))
    return items

def bundle(d,item,arm,h):
    sp=item['targets'][c.TARGET[arm]]
    z=e.collect(d,sp,arm,item['seed'],h)
    dr,qr,audit=e.risks(d,sp,z['dev_rows'],h)
    rows,picks,actions,curves=e.evaluate_rules(d,z,dr,qr,h)
    for au in audit:
        assert not set(au['train'])&set(sp['test'])
        assert not set(d['groups'][au['train']])&set(d['groups'][au['query']])
    if picks['common_B']['feasible']:
        met=next(v['development'] for v in rows if v['rule']=='common_B')
        assert met['FN']==0 and met['accuracy']>.8 and met['reported_as_normal']>0 and met['abnormal_released']>0
    for v in rows:
        v.update(rep=item['rep'],seed=item['seed'],arm=arm,target=c.TARGET[arm],seconds=h,development_n=len(sp['pool']),test_n=len(sp['test']),source_n=len(c.source_rows(arm,h)))
    return z,dr,qr,audit,rows,picks,actions

def compare_bundle(left,right):
    for key in left[0]:np.testing.assert_allclose(left[0][key],right[0][key],rtol=0,atol=1e-11)
    for j in [1,2]:np.testing.assert_allclose(left[j],right[j],rtol=0,atol=1e-11)
    assert r.safe(left[5])==r.safe(right[5])
    for key in left[6]:np.testing.assert_array_equal(left[6][key],right[6][key])

def smoke(d,items):
    item=items[0];activate(item,d);start=time.time();checks=[]
    for arm,h in [('target_only_SA',900),('HZ_to_SA',900),('HZ+HB_to_SA',900),('SA+HB_to_HZ',900),('SA+HZ_to_HB',900),('HZ+HB_to_SA',300),('HZ+HB_to_SA',600)]:
        activate(item,d);left=bundle(d,item,arm,h)
        sp=item['targets'][c.TARGET[arm]]
        allowed=np.r_[sp['pool'],c.source_rows(arm,h)]
        forbidden=np.setdiff1d(np.arange(len(d['x'])),allowed);old=d['medical'].copy();oldy=d['y'].copy()
        try:
            d['medical'][forbidden]=12345;d['y'][forbidden]=12345
            activate(item,d);right=bundle(d,item,arm,h);compare_bundle(left,right)
        finally:d['medical'][:]=old;d['y'][:]=oldy;c._SOURCE.clear()
        # Independently reproduce reported counts from confusion matrices.
        for row in left[4]:
            m=row['test'];cm=np.asarray(m['confusion']);assert cm.sum()==m['reported'] and cm.trace()==m['correct']
            assert int(cm[0,1]+cm[2,1])==m['FN'];assert int(cm[1,0]+cm[1,2])==m['FP']
        if h==900 and arm=='HZ_to_SA':
            z,dr,qr,_,_,picks,_=left;dp=z['dev_prediction'];normal=r.state(dp)==1
            cap=float(np.quantile(dr[normal,1],.9));risk=np.c_[-d['x'][z['dev_rows'],1],d['x'][z['dev_rows'],1]]
            original=ORIGINAL_NORMAL(dp,d['medical'][z['dev_rows'],2],dr,risk,{k:z['dev_score_'+k] for k in e.a.RULES},cap)
            assert r.safe(original)==r.safe(picks['common_B']['normal'])
        if h<900:
            prefix=c.context()[1];later={t:prefix[t].copy() for t in TIMES if t>h};full=d['x'].copy()
            try:
                for t in later:prefix[t][:]=7777
                d['x'][:]=7777;activate(item,d);future=bundle(d,item,arm,h);compare_bundle(left,future)
            finally:
                for t in later:prefix[t][:]=later[t]
                d['x'][:]=full;c._SOURCE.clear()
        checks.append(dict(arm=arm,seconds=h,poisoned_forbidden_n=len(forbidden),prediction_scores_thresholds_actions_unchanged=True,later_parameter_arrays_unused=h<900))
        print('SMOKE',arm,h,'PASS',round(time.time()-start,1),flush=True)
    r.write(O/'SMOKE.json',dict(passed=True,cases=checks,fast_selector_matches_original=True,seconds=time.time()-start))

def run_round(d,item):
    folder=O/'private'/f'round_{item["rep"]:03d}'
    if (folder/'COMPLETE.json').exists():return
    activate(item,d);folder.mkdir(parents=True,exist_ok=True);start=time.time();rows=[];policies={}
    for h in TIMES:
        for arm in ARMS:
            key=f'{arm}_T{h}';path=folder/(key+'.json')
            if path.exists():
                saved=json.loads(path.read_text());rows+=saved['rows'];policies[key]=saved['policies'];continue
            z,dr,qr,audit,rr,pp,actions=bundle(d,item,arm,h)
            np.savez_compressed(folder/(key+'.npz'),**z,dev_risk=dr,test_risk=qr,reference=d['medical'][z['test'],2],**{'action_'+k.replace(' ','_'):v for k,v in actions.items()})
            r.write(path,dict(rows=rr,policies=pp,risk_training=audit));rows+=rr;policies[key]=pp
    assert len(rows)==len(ARMS)*len(TIMES)*6
    r.write(folder/'RESULTS.json',rows);r.write(folder/'COMPLETE.json',dict(rep=item['rep'],seconds=time.time()-start,n_models=36,n_rule_results=len(rows)))
    print('ROUND',item['rep']+1,'COMPLETE',round(time.time()-start,1),flush=True)

def aggregate():
    allrows=[]
    for rep in range(100):
        folder=O/'private'/f'round_{rep:03d}'
        assert (folder/'COMPLETE.json').exists(),rep
        allrows+=json.loads((folder/'RESULTS.json').read_text())
    summary=[]
    metrics=['accuracy','reporting_rate','FNR','FPR','MAE','RMSE','R2','low_recall','high_recall','low_MAE','high_MAE','low_retained_correct_rate','high_retained_correct_rate']
    for arm in ARMS:
        for h in TIMES:
            for rule in ['All report','common_B',*e.a.RULES,'B95']:
                rr=[v for v in allrows if v['arm']==arm and v['seconds']==h and v['rule']==rule]
                assert len(rr)==100 and len({v['rep'] for v in rr})==100
                v=dict(arm=arm,target=c.TARGET[arm],seconds=h,rule=rule,repeats=100,feasible_rounds=sum(x['feasible'] for x in rr),
                    zero_FN_rounds=sum(x['test']['FN']==0 and x['test']['abnormal_released']>0 for x in rr),
                    undefined_FNR_rounds=sum(x['test']['FNR'] is None for x in rr),total_FN=sum(x['test']['FN'] for x in rr),
                    accuracy_gt80_rounds=sum(x['test']['accuracy'] is not None and x['test']['accuracy']>.8 for x in rr),reporting_gt75_rounds=sum(x['test']['reporting_rate']>.75 for x in rr))
                for k in metrics:
                    a=np.array([x['test'][k] for x in rr if x['test'][k] is not None],float)
                    v[k+'_mean']=float(a.mean()) if len(a) else None;v[k+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None;v[k+'_n']=len(a)
                summary.append(v)
    r.write(O/'ALL_ROUNDS.json',allrows);r.write(O/'SUMMARY.json',summary)
    with (O/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    failures=[v for v in allrows if v['rule']=='common_B' and (v['test']['FN']>0 or v['test']['accuracy'] is None or v['test']['accuracy']<=.8 or v['test']['reporting_rate']<=.75)]
    r.write(O/'ALL_FLAGGED_ROUNDS.json',failures)
    protected=json.loads((O/'PROTECTED_FILES.json').read_text());assert all(r.sha(Path(p))==sha for p,sha in protected.items())
    r.write(O/'FINAL_CHECKS.json',dict(all100_retained=True,model_evaluations=3600,rule_results=len(allrows),original_files_unchanged=True,mean_sd_not_confidence_intervals=True))
    print('AGGREGATE PASS',len(allrows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','run','aggregate']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=='aggregate':aggregate()
        else:
            d=init();items=prepare(d)
            if args.stage=='prepare':print('PREPARED',len(items),flush=True)
            elif args.stage=='smoke':smoke(d,items)
            else:
                assert json.loads((O/'SMOKE.json').read_text())['passed']
                for item in items[args.start:args.stop]:run_round(d,item)
