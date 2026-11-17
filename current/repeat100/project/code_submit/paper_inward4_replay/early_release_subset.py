"""Training-side selection of early subsets; fixed mappings and 15-min fallback.

This is an exploratory retrospective analysis of acquired U-TEG signals.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,gates,analysis,current_finalize as check

ROOT=suite.ROOT/'early_subset_20261103'
PRIVATE=ROOT/'private'
SCORES=['LR','HGB']
MODES=['early5_10','defer5_then10','stable10']
TARGETS=[.90,.95]

def initialize():
    suite.initialize();PRIVATE.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='early_subset_20261103_v1',data_sha256=c.sha(suite.ROOT/'data/DATA15_CURRENT.npz'),
        question='Can a development-selected gate identify a useful early-report subset, keeping fixed15 as the fallback?',
        models='Unchanged current inward4 B-only mapping; existing LR/HGB error scores, no new feature or mapper search.',
        primary='LR_early5_10_q90',scores=SCORES,modes=MODES,agreement_targets=TARGETS,
        thresholds='Separate thresholds for predicted low/within/high. Maximize selected training-OOF count subject to each branch empirical agreement>=q; within-range branch permits no observed abnormal-to-normal releases.',
        support='At least max(5,ceil(.05*training_pool)) selected records, at least5 identity-proxy groups and at least2 development folds in each enabled predicted-state branch.',
        defer5_then10='Release nothing at5. At10 consider only records rejected by the original same-score .15-risk5min gate. Refit10min class thresholds on this development subgroup.',
        stable10='Release nothing at5. At10 require equal predicted states at5 and10, then fit class thresholds on that development subgroup.',
        early5_10='Fit5min thresholds first; fit10min thresholds among development records deferred by that5min policy. Lock first report.',
        fallback='For all policies at15 use the unchanged primary Boundary95 action/prediction on still-unreported records.',
        evaluation='Original60/201,20seeds and separate301 grouped5outer3inner; all12 fixed candidate procedures shown; no external-result winner promotion.',
        early_coverage='No75percent minimum for the early subset; final coverage and agreement assessed separately.',
        fnr='True abnormal reported as within-range / released true abnormal; undefined when denominator0. Also report per-original-class withheld counts and 3class confusion.',
        evidence='Exploratory conditional fixed-input analysis; datasets previously inspected; no independent clinical or prospective guarantee.',
        B_from_original_data=True,manuscript_changed=False)
    p=ROOT/'PROTOCOL.json'
    if p.exists():assert c.read(p)==protocol
    else:gates.clean_write(p,protocol)

def state(y):return gates.fr.ma_class(y)

def threshold(score,pred,truth,eligible,groups,folds,q,branch):
    available=eligible&np.isfinite(score)&(pred==branch)
    minimum=max(5,int(math.ceil(.05*len(pred))));pick=None
    for t in np.unique(score[available]):
        a=available&(score<=t);n=int(a.sum())
        if n<minimum or len(np.unique(groups[a]))<5 or len(np.unique(folds[a]))<2:continue
        correct=int(np.sum(pred[a]==truth[a]));fn=int(np.sum((truth[a]!=0)&(pred[a]==0)))
        if correct/n+1e-12<q or (branch==0 and fn):continue
        candidate=dict(threshold=float(t),selected=n,correct=correct,errors=n-correct,fn=fn,groups=len(np.unique(groups[a])),folds=len(np.unique(folds[a])))
        if pick is None or n>pick['selected']:pick=candidate
    return pick

def apply_stage(stages,pred,score,eligible):
    a=np.zeros(len(pred),bool)
    for branch,v in stages.items():
        if v is not None:a|=eligible&(pred==int(branch))&np.isfinite(score)&(score<=v['threshold'])
    return a

def old5_threshold(dev,score):
    v=gates.rc.select_threshold(dev[score+'_5'],dev['pred5'],dev['truth'],.15).get('threshold')
    return None if v is None else float(v)

def old5_action(data,score,threshold_value):
    return np.zeros(len(data['pred5']),bool) if threshold_value is None else np.isfinite(data[score+'_5'])&(data[score+'_5']<=threshold_value)&np.isin(data['pred5'],[-1,0,1])

def fit_policy(dev,score,mode,q):
    p=dict(score=score,mode=mode,q=q,old5_threshold=old5_threshold(dev,score),stages={})
    valid5=np.isin(dev['pred5'],[-1,0,1]);a5=np.zeros(len(valid5),bool)
    if mode=='early5_10':
        stages={str(b):threshold(dev[score+'_5'],dev['pred5'],dev['truth'],valid5,dev['groups'],dev['fold'],q,b) for b in [-1,0,1]}
        p['stages']['5']=stages;a5=apply_stage(stages,dev['pred5'],dev[score+'_5'],valid5)
    valid10=np.isin(dev['pred10'],[-1,0,1])
    if mode=='early5_10':eligible=valid10&~a5
    elif mode=='defer5_then10':eligible=valid10&~old5_action(dev,score,p['old5_threshold'])
    else:eligible=valid10&valid5&(dev['pred5']==dev['pred10'])
    p['stages']['10']={str(b):threshold(dev[score+'_10'],dev['pred10'],dev['truth'],eligible,dev['groups'],dev['fold'],q,b) for b in [-1,0,1]}
    return p

def apply_policy(p,query):
    score=p['score'];valid5=np.isin(query['pred5'],[-1,0,1]);valid10=np.isin(query['pred10'],[-1,0,1]);a5=np.zeros(len(valid5),bool)
    if p['mode']=='early5_10':a5=apply_stage(p['stages']['5'],query['pred5'],query[score+'_5'],valid5)
    if p['mode']=='early5_10':eligible=valid10&~a5
    elif p['mode']=='defer5_then10':eligible=valid10&~old5_action(query,score,p['old5_threshold'])
    else:eligible=valid10&valid5&(query['pred5']==query['pred10'])
    a10=apply_stage(p['stages']['10'],query['pred10'],query[score+'_10'],eligible)
    assert not np.any(a5&a10)
    return a5,a10

def paper_data(arm,seed):
    d,*_=c.context();sp=c.paper_spec(arm,seed);si=c.SEEDS.index(seed);pool=np.sort(sp['pool']);ids={str(v):i for i,v in enumerate(d['ids'])}
    positions={int(i):j for j,p in enumerate(sp['positions'][:3]) for i in p['B']}
    dev=dict(truth=state(d['y'][pool,2]),groups=d['groups'][pool],fold=np.asarray([positions[int(i)] for i in pool]),rows=pool)
    query={};test=sp['test']
    for h in [300,600]:
        minutes=h//60;v=c.read(suite.ROOT/'gates'/f'{arm}_T{h}.development.json')[si]
        assert v['seed']==seed
        order={ids[row['record_id']]:j for j,row in enumerate(v['rows'])};ix=[order[int(i)] for i in pool]
        np.testing.assert_array_equal(np.asarray(v['truth'])[ix],dev['truth']);dev['pred'+str(minutes)]=np.asarray(v['pred'])[ix]
        z=analysis.load(suite.ROOT/'gates'/f'{arm}_T{h}.npz');np.testing.assert_array_equal(z['test'],test)
        query['pred'+str(minutes)]=state(z['prediction'][si]);query['ma'+str(minutes)]=z['prediction'][si]
        for score in SCORES:
            dev[f'{score}_{minutes}']=np.asarray([np.inf if a is None else a for a in v['scores'][score]])[ix]
            query[f'{score}_{minutes}']=z[score+'_score'][si]
            if h==300:query[score+'_old5']=z[score+'_action'][si]
    z=analysis.load(suite.ROOT/'gates'/f'{arm}_T900.npz');query['ma15']=z['prediction'][si];query['a15']=z['boundary95_action'][si]
    for score in SCORES:np.testing.assert_array_equal(old5_action(query,score,old5_threshold(dev,score)),query[score+'_old5'])
    assert not set(d['groups'][pool])&set(d['groups'][test])
    return dev,query,test

def internal_data(arm,fi):
    path=PRIVATE/f'all301_{arm}_fold{fi}.npz'
    if path.exists():
        z=analysis.load(path);return {k[4:]:v for k,v in z.items() if k.startswith('dev_')},{k[6:]:v for k,v in z.items() if k.startswith('query_')},z['test']
    d,_,_,_,_,folds,local=c.context();f=folds[fi];train=np.asarray(f['A'],int);test=np.asarray(f['B'],int);where={int(i):j for j,i in enumerate(train)}
    dev=dict(truth=state(d['y'][train,2]),groups=d['groups'][train],fold=np.full(len(train),-1,int),rows=train);query={}
    for h in [300,600]:
        minutes=h//60;dev['pred'+str(minutes)]=np.full(len(train),9,int)
        for score in SCORES:dev[f'{score}_{minutes}']=np.full(len(train),np.inf)
        for pi,sub in enumerate(f['inner']):
            inn=suite.subfolds(sub['T'],22000+fi*3+pi);out=c.evaluate(sub['T'],sub['V'],inn,arm,h);ss,_=gates.fit_score(out,h,1000+fi,arm,pi)
            ix=[where[int(i)] for i in out['query']];dev['pred'+str(minutes)][ix]=state(out['prediction']);dev['fold'][ix]=pi
            for score in SCORES:dev[f'{score}_{minutes}'][ix]=ss[score]
        out=c.evaluate(train,test,f['inner'],arm,h);ss,_=gates.fit_score(out,h,1000+fi,arm,3)
        query['pred'+str(minutes)]=state(out['prediction']);query['ma'+str(minutes)]=out['prediction']
        for score in SCORES:query[f'{score}_{minutes}']=ss[score]
        saved=analysis.load(suite.ROOT/'all301_gates'/f'{arm}_T{h}.npz');loc={int(i):j for j,i in enumerate(saved['local'])};ix=[loc[int(i)] for i in test]
        np.testing.assert_array_equal(out['prediction'],saved['prediction'][ix])
        for score in SCORES:
            th=gates.rc.select_threshold(dev[f'{score}_{minutes}'],dev['pred'+str(minutes)],dev['truth'],.15).get('threshold')
            a=np.zeros(len(test),bool) if th is None else np.isfinite(ss[score])&(ss[score]<=th)
            np.testing.assert_array_equal(a,saved[score][ix])
            if h==300:query[score+'_old5']=a
    assert (dev['fold']>=0).all() and not set(d['groups'][train])&set(d['groups'][test])
    saved=analysis.load(suite.ROOT/'all301_gates'/f'{arm}_T900.npz');loc={int(i):j for j,i in enumerate(saved['local'])};ix=[loc[int(i)] for i in test]
    query['ma15']=saved['prediction'][ix];query['a15']=saved['BOUNDARY95'][ix]
    np.savez_compressed(path,**{'dev_'+k:v for k,v in dev.items()},**{'query_'+k:v for k,v in query.items()},test=test)
    print('EARLY_SCORE_CACHE',arm,fi,flush=True)
    return dev,query,test

def policy_name(score,mode,q):return f'{score}_{mode}_q{int(q*100)}'

def outcomes(query,a5,a10):
    a15=query['a15']&~a5&~a10
    pred=np.where(a5,query['ma5'],np.where(a10,query['ma10'],query['ma15']))
    when=np.where(a5,5,np.where(a10,10,np.where(a15,15,-1)))
    return dict(prediction=pred,time=when,action=when>0,fixed15_prediction=query['ma15'],fixed15_action=query['a15'])

def test():
    initialize();g=np.array(['g'+str(i) for i in range(20)]);fold=np.arange(20)%3
    dev=dict(pred5=np.zeros(20,int),pred10=np.ones(20,int),truth=np.r_[np.zeros(10,int),np.ones(10,int)],groups=g,fold=fold,LR_5=np.arange(20,dtype=float),LR_10=np.arange(20,dtype=float)[::-1])
    p=fit_policy(dev,'LR','early5_10',.9);q={k:v.copy() for k,v in dev.items() if k not in ['truth','groups','fold']};a,b=apply_policy(p,q)
    assert a[:10].all() and not a[10:].any() and b[10:].all() and not b[:10].any()
    assert p['stages']['5']['0']['fn']==0
    assert threshold(np.array([1.,1.,1.]),np.zeros(3,int),np.zeros(3,int),np.ones(3,bool),g[:3],fold[:3],.9,0) is None
    # Scores tied at a threshold must be accepted together; no identity-based tie picking.
    tied=threshold(np.ones(20),np.zeros(20,int),np.r_[np.zeros(19,int),1],np.ones(20,bool),g,fold,.9,0);assert tied is None
    q['ma5']=np.full(20,55.);q['ma10']=np.full(20,75.);q['ma15']=np.full(20,65.);q['a15']=np.ones(20,bool)
    z=outcomes(q,a,b);assert (z['prediction'][:10]==55).all() and (z['prediction'][10:]==75).all()
    # A label field in the query is neither accepted nor consulted by the policy.
    q['truth']=np.full(20,999);aa,bb=apply_policy(p,q);np.testing.assert_array_equal(aa,a);np.testing.assert_array_equal(bb,b)
    real,query,test_rows=paper_data('HZ_to_SA',1000);selected=fit_policy(real,'LR','early5_10',.9);first=apply_policy(selected,query)
    d,*_=c.context();save=d['y'].copy()
    try:
        d['y'][test_rows]=1e6;other,query2,rows=paper_data('HZ_to_SA',1000);second=fit_policy(other,'LR','early5_10',.9)
        assert selected==second
        for x,y in zip(first,apply_policy(second,query2)):np.testing.assert_array_equal(x,y)
    finally:d['y'][:]=save
    # Rebuild one actual301 context with poisoned outer labels, including mapper and scores.
    f=c.context()[5][0];arm='HZ_to_SA';h=600
    before=c.evaluate(f['A'],f['B'],f['inner'],arm,h);s1,_=gates.fit_score(before,h,1000,arm,3)
    try:
        d['y'][f['B']]=1e6;after=c.evaluate(f['A'],f['B'],f['inner'],arm,h);s2,_=gates.fit_score(after,h,1000,arm,3)
    finally:d['y'][:]=save
    np.testing.assert_array_equal(before['prediction'],after['prediction'])
    for key in SCORES:np.testing.assert_array_equal(s1[key],s2[key])
    gates.clean_write(ROOT/'SMOKE.json',dict(synthetic_selection=True,tie_handling=True,minimum_support=True,first_report_lock=True,paper_test_labels_not_used=True,all301_outer_poison_invariant=True,upstream_input_lineage_unchanged=True))
    print('EARLY_SUBSET_SMOKE_PASS',flush=True)

def run(protocol):
    initialize();assert (ROOT/'SMOKE.json').exists()
    d,_,_,_,_,folds,local=c.context();policies=[(s,m,q) for s in SCORES for m in MODES for q in TARGETS]
    for arm in c.MAIN:
        dest=PRIVATE/f'{protocol}_{arm}_PREDICTIONS.npz'
        if dest.exists():continue
        tests=c.paper_spec(arm,1000)['test'] if protocol=='paper' else local
        nrun=20 if protocol=='paper' else 1;where={int(v):j for j,v in enumerate(tests)}
        names=[policy_name(*p) for p in policies]+['existing_LR_toB95','existing_HGB_toB95','primary_B95','fixed15']
        output={name:dict(prediction=np.full((nrun,len(tests)),np.nan),time=np.full((nrun,len(tests)),-1,int),action=np.zeros((nrun,len(tests)),bool)) for name in names}
        selections=[]
        contexts=list(enumerate(c.SEEDS)) if protocol=='paper' else [(fi,fi) for fi in range(5)]
        for ci,seed in contexts:
            dev,query,idx=paper_data(arm,seed) if protocol=='paper' else internal_data(arm,ci)
            row=ci if protocol=='paper' else 0;cols=np.array([where[int(i)] for i in idx]);reference=None
            for score,mode,q in policies:
                fitted=fit_policy(dev,score,mode,q);a5,a10=apply_policy(fitted,query);z=outcomes(query,a5,a10);name=policy_name(score,mode,q)
                selections.append(dict(context=ci,policy=name,development_records=len(dev['truth']),fitted=fitted))
                for k in ['prediction','time','action']:output[name][k][row,cols]=z[k]
            for score in SCORES:
                th5=old5_threshold(dev,score);a5=old5_action(query,score,th5)
                th10=gates.rc.select_threshold(dev[score+'_10'],dev['pred10'],dev['truth'],.15).get('threshold')
                a10=np.zeros(len(idx),bool) if th10 is None else np.isfinite(query[score+'_10'])&(query[score+'_10']<=th10)&~a5
                z=outcomes(query,a5,a10)
                for k in ['prediction','time','action']:output['existing_'+score+'_toB95'][k][row,cols]=z[k]
            if protocol=='paper':
                zz=analysis.load(suite.ROOT/'sequence'/f'{arm}.npz');key='BOUNDARY95__5-10-15'
                for k in ['prediction','time','action']:output['primary_B95'][k][row,cols]=zz[key+'_'+k][ci]
                output['primary_B95']['time'][row,cols]//=60
            else:
                pp={};aa={}
                for h in c.HORIZONS:
                    zz=analysis.load(suite.ROOT/'all301_gates'/f'{arm}_T{h}.npz');pp[h]=zz['prediction'][cols];aa[h]=zz['BOUNDARY95'][cols]
                p,t,a=analysis.replay(pp,aa,[300,600,900]);output['primary_B95']['prediction'][row,cols]=p;output['primary_B95']['time'][row,cols]=np.where(t>0,t//60,-1);output['primary_B95']['action'][row,cols]=a
            output['fixed15']['prediction'][row,cols]=query['ma15'];output['fixed15']['time'][row,cols]=np.where(query['a15'],15,-1);output['fixed15']['action'][row,cols]=query['a15']
            print('EARLY_CONTEXT',protocol,arm,ci,flush=True)
        np.savez_compressed(dest,**{name+'__'+k:v for name,m in output.items() for k,v in m.items()},test=tests)
        gates.clean_write(dest.with_suffix('.selection.json'),selections)
    summarize()

def summarize():
    d,*_=c.context();rows=[];contrasts=[];first=[];stable=[]
    from .manuscript_results import ratio,paired_ratio_difference
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            path=PRIVATE/f'{protocol}_{arm}_PREDICTIONS.npz'
            if not path.exists():continue
            z=analysis.load(path);ix=z['test'];truth=d['y'][ix,2];tc=state(truth);gg=d['groups'][ix]
            names=[k[:-len('__prediction')] for k in z if k.endswith('__prediction')]
            fp=z['fixed15__prediction'];fa=z['fixed15__action'];fc=state(fp)==tc
            for name in names:
                p=z[name+'__prediction'];t=z[name+'__time'];a=z[name+'__action'];correct=state(p)==tc
                for label,mask in [('at5',t==5),('new_at10',t==10),('by10',(t>0)&(t<=10)),('final',a)]:
                    m=check.score(p,truth,mask);row=dict(protocol=protocol,arm=arm,policy=name,subset=label,**m)
                    v=ratio(mask&correct,mask,gg,2000);row['agreement_interval']=v
                    rows.append(row)
                early=(t>0)&(t<=10)
                first.append(dict(protocol=protocol,arm=arm,policy=name,early_events=int(early.sum()),early_correct_also_correct_at15=int((early&correct&fa&fc).sum()),early_correct_otherwise_wrong15=int((early&correct&fa&~fc).sum()),early_correct_otherwise_withheld15=int((early&correct&~fa).sum()),early_wrong_otherwise_correct15=int((early&~correct&fa&fc).sum()),early_wrong_otherwise_wrong15=int((early&~correct&fa&~fc).sum()),early_wrong_otherwise_withheld15=int((early&~correct&~fa).sum())))
                for key,x,y in [('correct_yield',a&correct,fa&fc),('incorrect_yield',a&~correct,fa&~fc),('coverage',a,fa)]:
                    contrasts.append(dict(protocol=protocol,arm=arm,policy=name,metric=key,**ratio(x.astype(int)-y,np.ones_like(x),gg,2000)))
                for state_name,b in [('low',-1),('within',0),('high',1)]:
                    stable.append(dict(protocol=protocol,arm=arm,policy=name,true_state=state_name,records=int((tc==b).sum()),early_in_every_repeat=int(np.sum(np.all(early,axis=0)&(tc==b))),early_correct_in_every_repeat=int(np.sum(np.all(early&correct,axis=0)&(tc==b)))))
    gates.clean_write(ROOT/'RESULTS.json',dict(rows=rows,paired_fixed15=contrasts,early_vs_fixed15=first,record_repeat_stability=stable,definitions='Counts are record-seed events in paper protocol,301 held-out record outcomes in internal protocol. State-specific recalls/FNR are conditional on release; withheld counts remain explicit.'))
    public=[]
    for row in rows:
        public.append({k:json.dumps(gates.safe(v),separators=(',',':')) if isinstance(v,(dict,list,np.ndarray)) else v for k,v in row.items()})
    if public:
        with (ROOT/'AGGREGATE_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(public[0]));w.writeheader();w.writerows(public)
    print('EARLY_AGGREGATED',len(rows),flush=True)

def abnormal_early_check():
    """One post-result ablation: disable early normal releases, without refitting.

    The zero early binary FN is structural and is not evidence of safety.
    Assess early precision, directional errors and final outcomes instead.
    """
    initialize()
    name='LR_early5_10_q90_abnormal_only_POSTHOC';base='LR_early5_10_q90'
    spec=dict(base_policy=base,derived_policy=name,status='Post-result exploratory branch ablation; not a prespecified primary result.',
        rule='Keep the frozen primary thresholds. Release early only when the primary policy selected a predicted low or high state; all other records use the original fixed15 fallback. No refit or new cutoff search.',
        rationale='Primary early errors included abnormal-to-normal releases; assess a smaller early abnormal-report subset.',
        warning='Early binary FNR is zero by construction, not a learned guarantee. Report early precision, low/high direction errors, normal false releases and final full-cohort outcomes.',
        B_from_original_data=True)
    path=ROOT/'POSTHOC_PROTOCOL.json'
    if path.exists():assert c.read(path)==spec
    else:gates.clean_write(path,spec)
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            path=PRIVATE/f'{protocol}_{arm}_PREDICTIONS.npz';z=analysis.load(path)
            p=z[base+'__prediction'];t=z[base+'__time'];early=(t>0)&(t<=10)&np.isin(state(p),[-1,1])
            fp=z['fixed15__prediction'];fa=z['fixed15__action']
            z[name+'__prediction']=np.where(early,p,fp)
            z[name+'__time']=np.where(early,t,np.where(fa,15,-1))
            z[name+'__action']=early|fa
            np.savez_compressed(path,**z)
    summarize()

def audit_and_report():
    initialize();d,_,_,_,_,folds,local=c.context();x=c.read(ROOT/'RESULTS.json')
    primary='LR_early5_10_q90';post=primary+'_abnormal_only_POSTHOC'
    arms=c.MAIN;names=dict(zip(arms,['Target-only','单源','双源']))
    report_rows=[];checked=0
    for protocol in ['paper','all301']:
        for arm in arms:
            z=analysis.load(PRIVATE/f'{protocol}_{arm}_PREDICTIONS.npz');idx=z['test'];truth=d['y'][idx,2];tc=state(truth)
            assert len(idx)==len(np.unique(idx))==(201 if protocol=='paper' else 301)
            if protocol=='all301':np.testing.assert_array_equal(idx,local)
            for name in [k[:-12] for k in z if k.endswith('__prediction')]:
                p=z[name+'__prediction'];t=z[name+'__time'];a=z[name+'__action']
                assert np.isin(t,[-1,5,10,15]).all();np.testing.assert_array_equal(a,t>0)
                assert np.isfinite(p[a]).all()
                if name not in ['primary_B95','fixed15']:
                    deferred=(t<0)|(t==15)
                    np.testing.assert_array_equal(p[deferred],z['fixed15__prediction'][deferred])
                    np.testing.assert_array_equal(a[deferred],z['fixed15__action'][deferred])
                for subset,mask in [('at5',t==5),('new_at10',t==10),('by10',(t>0)&(t<=10)),('final',a)]:
                    saved=next(r for r in x['rows'] if (r['protocol'],r['arm'],r['policy'],r['subset'])==(protocol,arm,name,subset))
                    cm=np.array([[np.sum(mask&(tc==u)&(state(p)==v)) for v in [-1,0,1]] for u in [-1,0,1]])
                    np.testing.assert_array_equal(cm,saved['confusion']);assert cm.trace()==saved['correct'];assert cm.sum()==saved['reported']
                    assert saved['reported']+saved['withheld']==p.size
                    checked+=1
                if protocol=='all301' and name in [primary,post]:
                    for j in np.flatnonzero((t[0]>0)&(t[0]<=10)):
                        fi=next(i for i,f in enumerate(folds) if int(idx[j]) in f['B'])
                        report_rows.append(dict(policy=name,arm=arm,local_record_index=int(j),outer_fold=fi,minute=int(t[0,j]),predicted_ma=float(p[0,j]),reference_ma=float(truth[j]),predicted_state=int(state(p)[0,j]),reference_state=int(tc[j]),correct=bool(state(p)[0,j]==tc[j]),fixed15_ma=float(z['fixed15__prediction'][0,j]),fixed15_reported=bool(z['fixed15__action'][0,j])))
            # Posthoc restriction must be exactly the predicted-abnormal subset.
            early=(z[primary+'__time']>0)&(z[primary+'__time']<=10)&np.isin(state(z[primary+'__prediction']),[-1,1])
            np.testing.assert_array_equal(early,(z[post+'__time']>0)&(z[post+'__time']<=10))
            contexts=list(enumerate(c.SEEDS)) if protocol=='paper' else [(i,i) for i in range(5)]
            seen=np.zeros(len(idx),int);where={int(v):j for j,v in enumerate(idx)}
            for ci,seed in contexts:
                dev,query,test_idx=paper_data(arm,seed) if protocol=='paper' else internal_data(arm,ci)
                assert not set(d['groups'][dev['rows']])&set(d['groups'][test_idx])
                p=fit_policy(dev,'LR','early5_10',.9);before=apply_policy(p,query)
                poisoned=dict(query,ma15=np.full(len(test_idx),-1e6),a15=~query['a15'],truth=np.full(len(test_idx),999))
                for a,b in zip(before,apply_policy(p,poisoned)):np.testing.assert_array_equal(a,b)
                seen[[where[int(i)] for i in test_idx]]+=1
            assert (seen==(20 if protocol=='paper' else 1)).all()
    with (PRIVATE/'EARLY_RECORDS_ALL301.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(report_rows[0]));w.writeheader();w.writerows(report_rows)
    manuscript=Path('outputs/uteg_nc_goal_20261030/manuscript');manifest=c.read(manuscript/'CURRENT_MANIFEST.json')
    for path,sha in manifest['files'].items():assert c.sha(manuscript/path)==sha,path
    gates.clean_write(ROOT/'AUDIT.json',dict(metric_rows_recomputed=checked,outer_evaluation_once_per_record=True,identity_groups_disjoint=True,early_actions_invariant_to_query_labels_and_15min_values=True,original_model_predictions_and_risk_actions_reproduced=True,posthoc_uses_predicted_states_only=True,manuscript_hashes_unchanged=len(manifest['files']),original_B_values_retained=True))
    def get(protocol,arm,policy=primary,subset='by10'):
        return next(r for r in x['rows'] if (r['protocol'],r['arm'],r['policy'],r['subset'])==(protocol,arm,policy,subset))
    def pct(v):return '不适用' if v is None else f'{100*v:.2f}%'
    def num(v):return '不适用' if v is None else f'{v:.3f}'
    def frac(r,k):return f"{r[k]}/{r[k+'_denominator']}（{pct(r[k+'r'])}）"
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    lines=['# 5/10分钟早报子集：现有输入版本的探索性结果','',
        '确实存在按开发侧规则筛出的早报记录。301条内部评价的单源主方案在5/10分钟放行85条，其中81条正确、4条错误；最终覆盖率96.01%、一致率86.85%。同一程序在原60/201划分中的早报一致率仅79.24%，因此尚未建立跨评价方案稳定的早报能力。','',
        '本文只记录本轮实验，正文、附录、图表、PDF及补充材料包均未修改。','',
        '## 1. 怎么筛选','',
        '保持现有映射、风险模型和15分钟门不变。用训练侧交叉拟合的LR或HGB风险分数，按预测低、正常、高分别选阈值。90%或95%是开发侧经验一致率条件，不能当作评价集保证；预测正常分支要求开发侧零异常漏判。每个启用分支还须达到最小记录、身份组及折数支持。5分钟已报告的结果锁定，其余到10分钟继续判断，仍未报告的到15分钟使用原门。','',
        '冻结了12种候选程序，主程序是LR_early5_10_q90。没有依据评价集挑阈值或只保留预测正确的病例。两套评价分别运行：原60条开发/201条评价×20种子（4020个记录—种子事件）；301条按组5折外层/3折内层（每条留出一次）。','',
        '## 2. 301条内部评价：预先指定的主程序','',
        table(['流程','5分钟正确/错误','10分钟新增正确/错误','早报占301条','早报一致率','早报FNR','早报FPR'],[[names[a],f"{get('all301',a,subset='at5')['correct']}/{get('all301',a,subset='at5')['incorrect']}",f"{get('all301',a,subset='new_at10')['correct']}/{get('all301',a,subset='new_at10')['incorrect']}",pct(get('all301',a)['coverage']),pct(get('all301',a)['agreement']),frac(get('all301',a),'fn'),frac(get('all301',a),'fp')] for a in arms]),'',
        '单源早报85条中，真实正常72条（70正确、2判高）；真实高MA12条（11正确、1判正常）；真实低MA1条（判正常）。另外10条低MA仍等待。因此，早报95.29%一致率并不意味着低MA早期识别已经解决。','',
        table(['流程','最终正确/错误/未报告','最终覆盖','最终一致率','最终FNR','最终FPR','MAE mm','R²'],[[names[a],f"{(r:=get('all301',a,subset='final'))['correct']}/{r['incorrect']}/{r['withheld']}",pct(r['coverage']),pct(r['agreement']),frac(r,'fn'),frac(r,'fp'),num(r['mae']),num(r['r2'])] for a in arms]),'',
        '单源相对只在15分钟报告：正确数244→251，错误39→38，未报告18→12；但是异常判正常2→4，门后FNR由2/46（4.35%）升为4/47（8.51%）。总错误少1条与FNR升高可以同时发生，不能只看总体一致率。','',
        '单源81次正确早报中，73次等待到15分钟也会正确报告，3次15分钟会报错，5次15分钟会暂缓。这说明收益主要包括把已有的正确报告提前，同时也有少量新增正确报告；不能把81次都算作新增正确报告。','',
        '## 3. 原60/201划分：同一主程序','',
        table(['流程','早报正确/错误','早报覆盖','早报一致率','早报FNR','最终覆盖','最终一致率','最终FNR'],[[names[a],f"{(r:=get('paper',a))['correct']}/{r['incorrect']}",pct(r['coverage']),pct(r['agreement']),frac(r,'fn'),pct((f:=get('paper',a,subset='final'))['coverage']),pct(f['agreement']),frac(f,'fn')] for a in arms]),'',
        '这4020项不是4020名患者；它们来自201条记录、20次开发划分。置信区间按身份代理组重采样，不能把重复预测当成独立病例。两套方案的训练量和样本用途不同，差异不能直接归因于某一项因素。','',
        '## 4. 一个追加的探索性检查：仅预测异常时提前报告','',
        '看到主程序的早期异常判正常错误后，追加一次分支消融：阈值完全不变，只关闭早期预测正常的放行分支，保留预测低/高的早报；其他记录按原15分钟门处理。这不是预先指定的主结果，也没有再搜索阈值。早期二分类FNR按定义为零，不能把这个零当作安全性能证据。应看提前异常报告有多少是真的、方向是否正确，以及最终全流程错误。','']
    for protocol,label in [('all301','301条内部评价'),('paper','原60/201评价，记录—种子事件')]:
        lines += [f'### {label}','',table(['流程','早报正确/错误','早报覆盖','早报一致率','最终正确/错误/未报告','最终覆盖','最终一致率','最终FNR','最终FPR'],[[names[a],f"{(r:=get(protocol,a,post))['correct']}/{r['incorrect']}",pct(r['coverage']),pct(r['agreement']),f"{(f:=get(protocol,a,post,'final'))['correct']}/{f['incorrect']}/{f['withheld']}",pct(f['coverage']),pct(f['agreement']),frac(f,'fn'),frac(f,'fp')] for a in arms]),'']
    lines += ['## 5. 其余预先冻结候选，全部列出','',
        'early5_10：5分钟先筛、10分钟在剩余记录中筛；defer5_then10：5分钟不报告，10分钟只在原5分钟风险门会拒绝的记录中筛；stable10：5分钟不报告，只考虑5/10分钟预测类别相同的记录。q90/q95是开发侧经验一致率阈值。无记录放行时一致率、FNR均为“不适用”。以下不依据外层结果指定优胜者。','']
    for protocol in ['paper','all301']:
        for arm in arms:
            rows=[]
            for s in SCORES:
                for mode in MODES:
                    for q in TARGETS:
                        name=policy_name(s,mode,q);r=get(protocol,arm,name);f=get(protocol,arm,name,'final')
                        rows.append([name,f"{r['correct']}/{r['incorrect']}",pct(r['coverage']),pct(r['agreement']),frac(r,'fn'),pct(f['coverage']),pct(f['agreement']),frac(f,'fn')])
            lines += [f'### {protocol} / {names[arm]}','',table(['程序','早报正确/错误','早报覆盖','早报一致率','早报FNR','最终覆盖','最终一致率','最终FNR'],rows),'']
    lines += ['## 6. 指标与证据范围','',
        '- 一致率=正确报告/已报告；覆盖率=已报告/完整评价集合。FNR=已报告真实异常中判正常的比例；FPR=已报告真实正常中判异常的比例。低/高召回也同时给出已报告分母；完整类别总数、未报告数和混淆矩阵保留在JSON/CSV，不把暂缓记作正确。',
        '- 本次新增门的阈值拟合和应用未读取外层MA。检查了身份组隔离、标签扰动、15分钟输入扰动、首次报告锁定、阈值同分处理及逐项混淆矩阵。',
        '- B为原始U-TEG数据。5分钟和10分钟分析使用相应时段的实测信号，完整U-TEG记录为15分钟。本分析为回顾性探索。',
        '- 这些数据已多轮用于研究探索；本轮程序内的留出评价仍属于内部/探索性比较。不能据此宣称临床最优时间或独立临床安全保证。',
        '- 早期覆盖无需达到75%；最终覆盖与一致率分别检查。已有正确早报提示门有值得验证的用途，早期漏判、开发样本量敏感性和前缀输入可用性仍需要一起处理。','',
        '## 7. 文件','',
        '- `PROTOCOL.json`：12种候选及主程序冻结说明。',
        '- `POSTHOC_PROTOCOL.json`：追加异常早报分支消融，明确标记事后探索。',
        '- `AGGREGATE_RESULTS.csv` / `RESULTS.json`：全部程序、各时间点、各类别误差及配对15分钟比较。',
        '- `SMOKE.json` / `AUDIT.json`：测试和复核结果。',
        '- `private/EARLY_RECORDS_ALL301.csv`：主程序及追加检查的全部早报记录，含正确与错误；仅本地保留，不进入GitHub或补充材料。','']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('EARLY_AUDIT_REPORT_PASS',checked,len(report_rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','test','paper','all301','summary','abnormal','report']);a=ap.parse_args()
    with threadpool_limits(limits=1):
        if a.stage=='init':initialize()
        elif a.stage=='test':test()
        elif a.stage=='summary':initialize();summarize()
        elif a.stage=='abnormal':abnormal_early_check()
        elif a.stage=='report':audit_and_report()
        else:run(a.stage)
