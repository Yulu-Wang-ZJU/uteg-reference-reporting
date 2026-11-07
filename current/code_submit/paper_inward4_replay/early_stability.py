"""Diagnose early-release instability without changing active paper artifacts.

Downstream fits use training data. These are exploratory retrospective runs
using the measured signals available at each checkpoint.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import early_release_subset as e
c,suite,gates,analysis,check=e.c,e.suite,e.gates,e.analysis,e.check
ROOT=suite.ROOT/'early_stability_20261103'
PRIVATE=ROOT/'private'
POLICIES=['original','fold_stable','consensus2','consensus3','fold_stable_abnormal','consensus2_abnormal','consensus3_abnormal']

def initialize():
    e.initialize();PRIVATE.mkdir(parents=True,exist_ok=True)
    spec=dict(question='Separate early mapper sample size, cutoff sample size, cohort composition and cutoff/model instability.',
        current_data_sha=c.sha(suite.ROOT/'data/DATA15_CURRENT.npz'),
        primary_protocols='Unchanged original60/201 x20 seeds and frozen301 grouped5outer3inner.',
        matched_budget='In each frozen301 outer training set, take one stratified-group quarter (~60 records), with3 frozen split seeds. Evaluate on the SAME outer holdout. Compare full pipeline at small budget with full pipeline at full budget. Separately thin only the cutoff-selection OOF records while holding all predictions/risk models/15min fallback fixed.',
        expansion='Supplementary diagnostic: original201 holdout, train on all100 disjoint eligible local records,5 seeds1000..1004. Compare with the same5 original60 seeds. Not merged into original60 results.',
        policies=POLICIES,
        fold_stable='Fit original class-specific LR cutoff and three leave-one-development-fold-out cutoffs; enable a branch only if all supported, using their minimum. Same support conditions and empirical q90; predicted-normal branch zero observed FN.',
        consensus='Keep original selected first-report time; additionally require2/3 or3/3 development models to predict the same state and pass that stage cutoff on the query. A vetoed earlier report waits until15; it is not opportunistically reassigned at10.',
        abnormal='Additionally disable predicted-normal early release. Early binary FN=0 by construction; assess precision, class direction and final outcomes, not that tautological zero.',
        downstream_isolation='No outer truth used for model, score, threshold, agreement vote, or model selection. Every fixed policy reported, no external-result selection of thresholds.',
        fallback='Original15min prediction/gate for unchanged protocols; same-budget15min fit for full-pipeline budget diagnostics. Cutoff-only thinning holds full-budget fallback fixed.',
        evidence='Exploratory reuse of previously studied data.',
        B_from_original_data=True,valid_UTEG_only_validation=False,manuscript_changed=False)
    p=ROOT/'PROTOCOL.json'
    if p.exists():assert c.read(p)==spec
    else:gates.clean_write(p,spec)

def save_context(path,dev,q,test):
    np.savez_compressed(path,**{'dev_'+k:v for k,v in dev.items()},**{'query_'+k:v for k,v in q.items()},test=test)

def read_context(path):
    z=analysis.load(path)
    return {k[4:]:v for k,v in z.items() if k.startswith('dev_')},{k[6:]:v for k,v in z.items() if k.startswith('query_')},z['test']

def build_context(key,arm,train,test,parts,seed,existing=None):
    path=PRIVATE/(key+'.npz')
    if path.exists():return read_context(path)
    d,*_=c.context();train=np.asarray(train,int);test=np.asarray(test,int);where={int(i):j for j,i in enumerate(train)}
    assert not set(d['groups'][train])&set(d['groups'][test])
    dev=dict(rows=train,truth=e.state(d['y'][train,2]),groups=d['groups'][train],fold=np.full(len(train),-1,int));q={}
    for h in [300,600]:
        m=h//60;dev['pred'+str(m)]=np.full(len(train),9,int);dev['ma'+str(m)]=np.full(len(train),np.nan)
        for score in e.SCORES:dev[f'{score}_{m}']=np.full(len(train),np.inf)
        mp=[];ms=[]
        for pi,part in enumerate(parts):
            a,b=np.asarray(part['T'],int),np.asarray(part['V'],int)
            out=c.evaluate(a,b,part['inner'],arm,h);ss,_=gates.fit_score(out,h,seed,arm,pi)
            jj=[where[int(i)] for i in out['query']];dev['fold'][jj]=pi;dev['ma'+str(m)][jj]=out['prediction'];dev['pred'+str(m)][jj]=e.state(out['prediction'])
            for score in e.SCORES:dev[f'{score}_{m}'][jj]=ss[score]
            pm,_,xf=c.predict(arm,h,a,test)
            other=dict(out,query=test,prediction=pm,query_b=xf[test,0])
            sm,_=gates.fit_score(other,h,seed,arm,pi);mp.append(e.state(pm));ms.append(sm['LR'])
        q['member_pred'+str(m)]=np.asarray(mp);q['member_LR_'+str(m)]=np.asarray(ms)
        if existing is None:
            out=c.evaluate(train,test,[dict(T=p['T'],V=p['V']) for p in parts],arm,h);ss,_=gates.fit_score(out,h,seed,arm,3)
            q['ma'+str(m)]=out['prediction'];q['pred'+str(m)]=e.state(out['prediction'])
            for score in e.SCORES:q[f'{score}_{m}']=ss[score]
        else:
            olddev,oldq,oldtest=existing;np.testing.assert_array_equal(test,oldtest)
            loc={int(i):j for j,i in enumerate(olddev['rows'])};jj=[loc[int(i)] for i in train]
            np.testing.assert_array_equal(dev['pred'+str(m)],olddev['pred'+str(m)][jj])
            for score in e.SCORES:np.testing.assert_allclose(dev[f'{score}_{m}'],olddev[f'{score}_{m}'][jj],rtol=1e-12,atol=1e-12)
            for k in ['ma'+str(m),'pred'+str(m),'LR_'+str(m),'HGB_'+str(m)]:q[k]=oldq[k].copy()
    if existing is None:
        out=c.evaluate(train,test,[dict(T=p['T'],V=p['V']) for p in parts],arm,900)
        q['ma15']=out['prediction'];q['a15']=out['accepted']
    else:q['ma15']=existing[1]['ma15'].copy();q['a15']=existing[1]['a15'].copy()
    assert (dev['fold']>=0).all()
    save_context(path,dev,q,test);print('CONTEXT',key,flush=True)
    return dev,q,test

def base_context(protocol,arm,i):
    if protocol=='paper':
        seed=1000+i;sp=c.paper_spec(arm,seed);parts=[dict(T=p['A'],V=p['B'],inner=p['inner']) for p in sp['positions'][:3]]
        return build_context(f'base_{protocol}_{arm}_{i}',arm,np.sort(sp['pool']),sp['test'],parts,seed,e.paper_data(arm,seed))
    f=c.context()[5][i];parts=[dict(**p,inner=suite.subfolds(p['T'],22000+i*3+j)) for j,p in enumerate(f['inner'])]
    return build_context(f'base_{protocol}_{arm}_{i}',arm,f['A'],f['B'],parts,1000+i,e.internal_data(arm,i))

def new_parts(train,seed):
    return [dict(**p,inner=suite.subfolds(p['T'],seed+100+j)) for j,p in enumerate(suite.subfolds(train,seed))]

def subset_dev(dev,indices):
    return {k:v[indices] for k,v in dev.items()}

def stable_fit(dev):
    p=e.fit_policy(dev,'LR','early5_10',.9);p['stages']={};a5=np.zeros(len(dev['truth']),bool)
    for minute in [5,10]:
        pred=dev['pred'+str(minute)];score=dev['LR_'+str(minute)];eligible=np.isin(pred,[-1,0,1])&~a5
        stages={}
        for b in [-1,0,1]:
            fits=[]
            for omit in [None,*np.unique(dev['fold'])]:
                keep=np.ones(len(pred),bool) if omit is None else dev['fold']!=omit
                fits.append(e.threshold(score[keep],pred[keep],dev['truth'][keep],eligible[keep],dev['groups'][keep],dev['fold'][keep],.9,b))
            stages[str(b)]=None if any(t is None for t in fits) else dict(fits[0],threshold=min(t['threshold'] for t in fits),omitted_fold_thresholds=[t['threshold'] for t in fits[1:]])
        p['stages'][str(minute)]=stages
        if minute==5:a5=e.apply_stage(stages,pred,score,eligible)
    return p

def choose_policy(dev,q,name):
    stable=name.startswith('fold_stable');p=stable_fit(dev) if stable else e.fit_policy(dev,'LR','early5_10',.9)
    a5,a10=e.apply_policy(p,q)
    masks=[a5,a10]
    if name.startswith('consensus'):
        votes=int(name[len('consensus')]);assert votes in [2,3]
        for j,m in enumerate([5,10]):
            member=q['member_pred'+str(m)];risk=q['member_LR_'+str(m)];eligible=np.zeros_like(member,bool)
            for b,v in p['stages'][str(m)].items():
                if v is not None:eligible|=(member==int(b))&(member==q['pred'+str(m)])&np.isfinite(risk)&(risk<=v['threshold'])
            masks[j]=masks[j]&(eligible.sum(0)>=votes)
    if name.endswith('_abnormal'):
        masks=[a&np.isin(q['pred'+str(m)],[-1,1]) for a,m in zip(masks,[5,10])]
    return e.outcomes(q,*masks),p

def evaluate_set(protocol,arm,contexts,nrun,tests,policies=POLICIES):
    path=PRIVATE/f'pred_{protocol}_{arm}.npz'
    if path.exists():return
    names=[*policies,'fixed15'];where={int(i):j for j,i in enumerate(tests)};seen=np.zeros((nrun,len(tests)),int)
    out={k+'__'+v:np.full((nrun,len(tests)),np.nan if v=='prediction' else -1,dtype=float if v=='prediction' else int) for k in names for v in ['prediction','time']};fits=[]
    for row,dev,q,test in contexts:
        jj=[where[int(i)] for i in test];seen[row,jj]+=1
        for policy in policies:
            z,p=choose_policy(dev,q,policy)
            for k in ['prediction','time']:out[policy+'__'+k][row,jj]=z[k]
            fits.append(dict(row=row,policy=policy,train_n=len(dev['rows']),train_class=[int((dev['truth']==b).sum()) for b in [-1,0,1]],fitted=p))
        out['fixed15__prediction'][row,jj]=q['ma15'];out['fixed15__time'][row,jj]=np.where(q['a15'],15,-1)
    assert (seen==1).all()
    for name in names:out[name+'__action']=out[name+'__time']>0
    np.savez_compressed(path,**out,test=tests);gates.clean_write(path.with_suffix('.selection.json'),fits)

def baseline():
    initialize();d,_,_,_,_,folds,local=c.context()
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            def gen():
                for i in range(20 if protocol=='paper' else 5):
                    dev,q,test=base_context(protocol,arm,i)
                    yield i if protocol=='paper' else 0,dev,q,test
            tests=c.paper_spec(arm,1000)['test'] if protocol=='paper' else local
            evaluate_set(protocol,arm,gen(),20 if protocol=='paper' else 1,tests)
    summarize()

def quarter(train,fi,rep):
    d,*_=c.context();train=np.asarray(train,int)
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=31000+rep*10+fi)
    _,ii=next(splitter.split(train,e.state(d['y'][train,2]),d['groups'][train]))
    return train[ii]

def budgets():
    initialize();d,_,_,_,_,folds,local=c.context()
    for arm in c.MAIN:
        def thin_gen():
            for rep in range(3):
                for fi,f in enumerate(folds):
                    dev,q,test=base_context('all301',arm,fi);small=quarter(f['A'],fi,rep)
                    ii=np.flatnonzero(np.isin(dev['rows'],small));yield rep,subset_dev(dev,ii),q,test
        evaluate_set('all301_cutoff_only_quarter',arm,thin_gen(),3,local,['original'])
        def small_gen():
            for rep in range(3):
                for fi,f in enumerate(folds):
                    small=quarter(f['A'],fi,rep);seed=38000+rep*10+fi
                    dev,q,test=build_context(f'budget60_{arm}_{rep}_{fi}',arm,small,f['B'],new_parts(small,seed),1000+fi)
                    yield rep,dev,q,test
        evaluate_set('all301_pipeline_quarter',arm,small_gen(),3,local,['original'])
        test=c.paper_spec(arm,1000)['test'];train=np.setdiff1d(local,test)
        assert len(train)==100 and not set(d['groups'][train])&set(d['groups'][test])
        def expanded():
            for rep in range(5):
                dev,q,idx=build_context(f'expanded100_{arm}_{rep}',arm,train,test,new_parts(train,46000+rep),1000+rep)
                yield rep,dev,q,idx
        evaluate_set('paper_expanded100_5seeds',arm,expanded(),5,test,['original'])
    summarize()

def repeat301():
    initialize();d,_,_,_,_,folds,local=c.context()
    spec=dict(status='Additional stability audit after the first candidate results; all7 candidates repeated, no threshold tuning.',
        fixed='Outer5fold membership and training records unchanged.',
        repetitions='Original inner splits plus2 independently seeded3fold inner partitions; seeds57000+fold and58000+fold. Each record evaluated3times, not3 independent patients.',
        B_from_original_data=True,valid_UTEG_only_validation=False)
    path=ROOT/'REPEAT_PROTOCOL.json'
    if path.exists():assert c.read(path)==spec
    else:gates.clean_write(path,spec)
    for arm in c.MAIN:
        def contexts():
            for rep in range(3):
                for fi,f in enumerate(folds):
                    if rep==0:dev,q,test=base_context('all301',arm,fi)
                    else:
                        seed=56000+1000*rep+fi
                        dev,q,test=build_context(f'repeat301_{arm}_{rep}_{fi}',arm,f['A'],f['B'],new_parts(f['A'],seed),1000+100*rep+fi)
                    yield rep,dev,q,test
        evaluate_set('all301_repeated_inner3',arm,contexts(),3,local)
    summarize()

def summarize():
    initialize();d,*_=c.context();rows=[];stability=[];deltas=[]
    for path in sorted(PRIVATE.glob('pred_*.npz')):
        protocol,arm=next((path.stem[5:-(len(a)+1)],a) for a in c.MAIN if path.stem.endswith('_'+a))
        z=analysis.load(path);test=z['test'];y=d['y'][test,2];tc=e.state(y);names=[k[:-12] for k in z if k.endswith('__prediction')]
        for name in names:
            p=z[name+'__prediction'];t=z[name+'__time'];correct=e.state(p)==tc
            for part,mask in [('at5',t==5),('new_at10',t==10),('early',(t>0)&(t<=10)),('final',t>0)]:
                rows.append(dict(protocol=protocol,arm=arm,policy=name,subset=part,**check.score(p,y,mask)))
            per=[]
            for i in range(len(p)):
                early=(t[i]>0)&(t[i]<=10);m=check.score(p[i],y,early)
                per.append({k:m[k] for k in ['reported','correct','incorrect','agreement','fn','fn_denominator','fnr']})
            stability.append(dict(protocol=protocol,arm=arm,policy=name,repeats=len(p),per_repeat=per,
                early_every_repeat=int(np.all((t>0)&(t<=10),axis=0).sum()),correct_early_every_repeat=int(np.all((t>0)&(t<=10)&correct,axis=0).sum())))
            fixed=z['fixed15__action'];fc=e.state(z['fixed15__prediction'])==tc
            deltas.append(dict(protocol=protocol,arm=arm,policy=name,correct_count_difference=int(((t>0)&correct).sum()-(fixed&fc).sum()),error_count_difference=int(((t>0)&~correct).sum()-(fixed&~fc).sum())))
    gates.clean_write(ROOT/'RESULTS.json',dict(rows=rows,stability=stability,fixed15_deltas=deltas))
    if rows:
        flat=[{k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]
        with (ROOT/'AGGREGATE.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('SUMMARY',len(rows),flush=True)

def diagnostics():
    initialize();d,_,_,_,_,folds,local=c.context();records=[];cohort=[]
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            for i in range(20 if protocol=='paper' else 5):
                dev,q,test=base_context(protocol,arm,i);t=e.state(d['y'][test,2]);p=e.fit_policy(dev,'LR','early5_10',.9)
                da=e.apply_policy(p,dev);qa=e.apply_policy(p,q)
                for j,m in enumerate([5,10]):
                    for role,z,true,mask in [('development',dev,dev['truth'],da[j]),('heldout',q,t,qa[j])]:
                        pc=z['pred'+str(m)];s=z['LR_'+str(m)];correct=pc==true
                        auc=roc_auc_score(~correct,s) if len(np.unique(correct))==2 and np.isfinite(s).all() else None
                        records.append(dict(protocol=protocol,arm=arm,context=i,minute=m,role=role,n=len(true),class_counts=[int((true==b).sum()) for b in [-1,0,1]],all_correct=int(correct.sum()),risk_auc=auc,selected=int(mask.sum()),selected_correct=int((mask&correct).sum()),selected_fn=int((mask&(true!=0)&(pc==0)).sum()),selected_true_abnormal=int((mask&(true!=0)).sum()),enabled_branches={b:v for b,v in p['stages'][str(m)].items()}))
            if protocol=='all301':
                z=analysis.load(PRIVATE/f'pred_all301_{arm}.npz');ix=z['test'];orig=c.paper_spec(arm,1000)['test'];mask=np.isin(ix,orig)
                for subset,keep in [('original201',mask),('other100',~mask)]:
                    for name in ['original','consensus2_abnormal','fold_stable']:
                        pp=z[name+'__prediction'][:,keep];tt=z[name+'__time'][:,keep]
                        cohort.append(dict(arm=arm,cohort=subset,policy=name,**check.score(pp,d['y'][ix[keep],2],(tt>0)&(tt<=10))))
    gates.clean_write(ROOT/'DIAGNOSTICS.json',dict(contexts=records,all301_original201_subset=cohort))
    print('DIAGNOSTICS',len(records),flush=True)

def smoke():
    initialize();dev,q,idx=base_context('paper','HZ_to_SA',0);d,*_=c.context();backup=d['y'].copy()
    first={name:choose_policy(dev,q,name) for name in POLICIES}
    try:
        d['y'][idx]=1e6
        poison=dict(q,truth=np.full(len(idx),999),ma15=np.full(len(idx),-1e6),a15=~q['a15'])
        for name in POLICIES:
            z,p=choose_policy(dev,poison,name);old,op=first[name];assert p==op
            np.testing.assert_array_equal((z['time']>0)&(z['time']<=10),(old['time']>0)&(old['time']<=10))
            if name.endswith('_abnormal'):
                early=(z['time']>0)&(z['time']<=10);assert np.isin(e.state(z['prediction'])[early],[-1,1]).all()
    finally:d['y'][:]=backup
    # A consensus veto cannot transfer a selected5min report to10min.
    orig=first['original'][0]
    for name in ['consensus2','consensus3']:
        z=first[name][0];early=(z['time']>0)&(z['time']<=10);np.testing.assert_array_equal(z['time'][early],orig['time'][early])
    # The small-budget sample contains whole identity groups and no holdout group.
    for fi,f in enumerate(c.context()[5]):
        for rep in range(3):
            small=quarter(f['A'],fi,rep);groups=set(d['groups'][small])
            assert not groups&set(d['groups'][f['B']])
            assert set(np.asarray(f['A'])[np.isin(d['groups'][f['A']],list(groups))])==set(small)
    gates.clean_write(ROOT/'SMOKE.json',dict(outer_label_and_15min_poison_invariant=True,first_report_time_preserved=True,abnormal_filter_uses_predictions=True,whole_group_budget_sampling=True,upstream_lineage_unchanged=True))
    print('STABILITY_SMOKE_PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','smoke','baseline','budgets','diagnostics','summary','repeat301']);args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=='init':initialize()
        elif args.stage=='smoke':smoke()
        elif args.stage=='summary':summarize()
        elif args.stage=='diagnostics':diagnostics()
        elif args.stage=='repeat301':repeat301()
        else:
            assert (ROOT/'SMOKE.json').exists()
            (baseline if args.stage=='baseline' else budgets)()
