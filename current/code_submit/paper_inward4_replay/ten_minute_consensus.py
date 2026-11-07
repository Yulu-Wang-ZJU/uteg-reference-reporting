"""Dedicated10min three-model release, with a training-only low-B normal veto.

The 10-minute analysis uses the measured U-TEG signal acquired over the first 10 minutes.
No candidate changes the manuscript or the15min mapping/reporting fallback.
"""
import argparse,csv,json
import numpy as np
from threadpoolctl import threadpool_limits
from . import early_stability as s
c,e,gates,check=s.c,s.e,s.gates,s.check
ROOT=s.suite.ROOT/'ten_minute_consensus_20261103'
PRIVATE=ROOT/'private'
BASE_POLICIES=['legacy5_10','legacy10_only','class10','class10_lowguard','sharedLR10','sharedLR10_lowguard','margin10','margin10_lowguard']
EXTRA_POLICIES=['margin10_fence_fixed','margin10_fence_refit']
POLICIES=BASE_POLICIES+EXTRA_POLICIES
PRIMARY='sharedLR10_lowguard'

def initialize():
    s.initialize();PRIVATE.mkdir(parents=True,exist_ok=True)
    spec=dict(question='Improve stable10min release while avoiding lowMA-to-normal errors and retaining useful early reports.',
        primary=PRIMARY,policies=BASE_POLICIES,
        protocols='Original60/201 x20 seeds; separate frozen301 outer5fold x3 inner-repeat evaluations.',
        changes='New candidate schedules release only at10, with unchanged15min fallback. Original5/10 consensus and its10-only subset are controls.',
        consensus='All3 development-trained models and the full-training mapper must agree on the10min predicted state. LR candidates require all4 risk scores below the training-selected cutoff.',
        class10='Original per-predicted-class cutoff/support/q90 at10, fitted using all development OOF records rather than5min deferrals; predicted-normal branch zero observed FN.',
        shared='One developmentOOF cutoff across predicted states, q90 agreement, zero observed abnormal-to-normal release; minimum max(10,ceil(.1*n)) records,5 distinct proxy groups,2 folds.',
        lowguard='For predicted-normal releases only, require current10min B strictly above the largest10min B among local training lowMA records. Below this limit defer; never relabel as low. If no training lowMA, disable early normal releases. Bound is supervised training fit, not an OOF guarantee.',
        guard_ablation='Apply the same low guard before fitting a cutoff and at application. All nonguarded counterparts included. No test-label-informed numerical bound or cutoff search.',
        margin='Alternative shared score is negative distance to50/70 state boundary; query uses the worst distance among full and3 development models. Same class consensus, shared cutoff and optional low guard.',
        evidence='Exploratory reuse of previously inspected data; all frozen candidates and empty-release repetitions reported. No candidate automatically replaces manuscript or deployment model.',
        data_sha=c.sha(s.suite.ROOT/'data/DATA15_CURRENT.npz'),B_from_original_data=True,valid_UTEG_only_validation=False,manuscript_changed=False)
    p=ROOT/'PROTOCOL.json'
    if p.exists():assert c.read(p)==spec
    else:gates.clean_write(p,spec)

def context(protocol,arm,rep,fi=0):
    if protocol=='paper':return s.base_context('paper',arm,rep)
    if rep==0:return s.base_context('all301',arm,fi)
    return s.read_context(s.PRIVATE/f'repeat301_{arm}_{rep}_{fi}.npz')

def parts(protocol,arm,rep,fi):
    if protocol=='paper':
        sp=c.paper_spec(arm,1000+rep)
        return [np.asarray(p['A'],int) for p in sp['positions'][:3]]
    f=c.context()[5][fi]
    if rep==0:return [np.asarray(p['T'],int) for p in f['inner']]
    return [np.asarray(p['T'],int) for p in s.new_parts(f['A'],56000+1000*rep+fi)]

def augment(protocol,arm,rep,fi,dev,q,test):
    d,*_=c.context();out=dict(q);out['B10']=c.feature(dev['rows'],test,600)[test,0]
    dv=dict(dev);dv['B10']=c.feature(dev['rows'],test,600)[dev['rows'],0]
    path=PRIVATE/f'members_{protocol}_{arm}_{rep}_{fi}.npz'
    if path.exists():pm=s.analysis.load(path)['ma']
    else:
        pm=np.asarray([c.predict(arm,600,train,test)[0] for train in parts(protocol,arm,rep,fi)])
        np.savez_compressed(path,ma=pm)
    np.testing.assert_array_equal(e.state(pm),q['member_pred10']);out['member_ma10']=pm
    return dv,out

def guard_fit(dev,fence=False):
    values=dev['B10'][(dev['truth']==-1)&np.isfinite(dev['B10'])]
    bound=None if len(values)==0 else float(np.max(values))
    if fence and len(values):
        q1,q3=np.quantile(values,[.25,.75]);bound=max(bound,float(q3+1.5*(q3-q1)))
    return dict(bound=bound,low_training_records=int(len(values)),
        low_training_groups=int(len(np.unique(dev['groups'][dev['truth']==-1]))))

def guard_apply(p,data):
    if p['bound'] is None:return data['pred10']!=0
    return (data['pred10']!=0)|(np.isfinite(data['B10'])&(data['B10']>p['bound']))

def distance(p):return -np.minimum(abs(p-50),abs(p-70))

def shared_threshold(score,pred,truth,eligible,groups,folds):
    pick=None;minimum=max(10,int(np.ceil(.1*len(pred))))
    for t in np.unique(score[eligible&np.isfinite(score)]):
        a=eligible&np.isfinite(score)&(score<=t);n=int(a.sum())
        if n<minimum or len(np.unique(groups[a]))<5 or len(np.unique(folds[a]))<2:continue
        good=int((pred[a]==truth[a]).sum());fn=int((a&(truth!=0)&(pred==0)).sum())
        if good/n+1e-12<.9 or fn:continue
        v=dict(threshold=float(t),selected=n,correct=good,incorrect=n-good,fn=fn,
            predicted_counts=[int((a&(pred==b)).sum()) for b in [-1,0,1]],true_counts=[int((a&(truth==b)).sum()) for b in [-1,0,1]])
        if pick is None or n>pick['selected']:pick=v
    return pick

def fit(dev,policy):
    assert policy not in ['legacy5_10','legacy10_only']
    if policy=='margin10_fence_fixed':
        p=fit(dev,'margin10_lowguard');p['policy']=policy;p['original_guard']=p['guard'];p['guard']=guard_fit(dev,True)
        return p
    p=dict(policy=policy,guard=guard_fit(dev,policy.endswith('fence_refit')) if policy.endswith(('lowguard','fence_refit')) else None)
    eligible=np.isin(dev['pred10'],[-1,0,1])
    if p['guard'] is not None:eligible&=guard_apply(p['guard'],dev)
    score=distance(dev['ma10']) if policy.startswith('margin') else dev['LR_10']
    if policy.startswith('class'):
        p['cutoffs']={str(b):e.threshold(score,dev['pred10'],dev['truth'],eligible,dev['groups'],dev['fold'],.9,b) for b in [-1,0,1]}
    else:p['cutoff']=shared_threshold(score,dev['pred10'],dev['truth'],eligible,dev['groups'],dev['fold'])
    return p

def apply(p,q):
    eligible=np.isin(q['pred10'],[-1,0,1])&np.all(q['member_pred10']==q['pred10'],axis=0)
    if p['guard'] is not None:eligible&=guard_apply(p['guard'],q)
    if p['policy'].startswith('margin'):
        score=np.maximum(distance(q['ma10']),np.max(distance(q['member_ma10']),axis=0))
    else:score=np.maximum(q['LR_10'],np.max(q['member_LR_10'],axis=0))
    if 'cutoffs' in p:return e.apply_stage(p['cutoffs'],q['pred10'],score,eligible)
    v=p['cutoff'];return np.zeros(len(eligible),bool) if v is None else eligible&np.isfinite(score)&(score<=v['threshold'])

def outcome(dev,q,policy):
    if policy.startswith('legacy'):
        z,p=s.choose_policy(dev,q,'consensus3')
        if policy=='legacy5_10':return z,p
        return e.outcomes(q,np.zeros(len(q['pred10']),bool),z['time']==10),p
    p=fit(dev,policy);a10=apply(p,q)
    return e.outcomes(q,np.zeros(len(a10),bool),a10),p

def smoke():
    initialize();dev,q,test=context('paper','HZ_to_SA',0);dev,q=augment('paper','HZ_to_SA',0,0,dev,q,test)
    # Guard strictly defers an observed-low-range normal output; does not relabel it.
    g=guard_fit(dev);assert g['bound'] is not None
    demo=dict(pred10=np.array([0,0,-1,1]),B10=np.array([g['bound'],g['bound']+1e-6,g['bound']-1,g['bound']-1]))
    np.testing.assert_array_equal(guard_apply(g,demo),[False,True,True,True])
    assert not guard_apply(dict(bound=None),dict(pred10=np.array([0]),B10=np.array([3.])))[0]
    d,*_=c.context();y=d['y'].copy()
    for policy in POLICIES:
        z,p=outcome(dev,q,policy)
        try:
            d['y'][test]=1e6
            poison=dict(q,truth=np.full(len(test),999),ma15=np.full(len(test),-1e6),a15=~q['a15'])
            zz,pp=outcome(dev,poison,policy);assert p==pp
            np.testing.assert_array_equal((z['time']>0)&(z['time']<=10),(zz['time']>0)&(zz['time']<=10))
        finally:d['y'][:]=y
        if policy not in ['legacy5_10','legacy10_only']:
            assert not (z['time']==5).any();a=z['time']==10
            assert np.all(q['member_pred10'][:,a]==q['pred10'][a])
    gates.clean_write(ROOT/'SMOKE.json',dict(guard_boundary_correct=True,no_low_training_disables_normal_only=True,outer_label_and_15min_poison_invariant=True,no_query_label_argument_in_fit=True,class_consensus_enforced=True,upstream_lineage_unchanged=True))
    print('TEN_MINUTE_SMOKE_PASS',flush=True)

def run():
    initialize();assert (ROOT/'SMOKE.json').exists();d,_,_,_,_,folds,local=c.context()
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            path=PRIVATE/f'pred_{protocol}_{arm}.npz'
            if path.exists():continue
            test=c.paper_spec(arm,1000)['test'] if protocol=='paper' else local
            runs=20 if protocol=='paper' else 3;where={int(i):j for j,i in enumerate(test)}
            output={p+'__'+k:np.full((runs,len(test)),np.nan if k=='prediction' else -1,dtype=float if k=='prediction' else int) for p in [*POLICIES,'fixed15'] for k in ['prediction','time']}
            specs=[];seen=np.zeros((runs,len(test)),int)
            for rep in range(runs):
                for fi in range(1 if protocol=='paper' else 5):
                    dev,q,idx=context(protocol,arm,rep,fi);dev,q=augment(protocol,arm,rep,fi,dev,q,idx)
                    jj=[where[int(i)] for i in idx];seen[rep,jj]+=1
                    for policy in POLICIES:
                        z,p=outcome(dev,q,policy)
                        for k in ['prediction','time']:output[policy+'__'+k][rep,jj]=z[k]
                        specs.append(dict(repeat=rep,fold=fi,policy=policy,train_n=len(dev['rows']),fitted=p))
                    output['fixed15__prediction'][rep,jj]=q['ma15'];output['fixed15__time'][rep,jj]=np.where(q['a15'],15,-1)
            assert (seen==1).all()
            for p in [*POLICIES,'fixed15']:output[p+'__action']=output[p+'__time']>0
            np.savez_compressed(path,**output,test=test);gates.clean_write(path.with_suffix('.selection.json'),specs)
            print('TEN_MINUTE_COMPLETE',protocol,arm,flush=True)
    summarize()

def extend():
    initialize()
    spec=dict(status='Post-result exploratory refinement; all previous results preserved.',
        issue='A held-out lowMA can exceed the maximum lowMA B in its training fold; a maximum-only veto is not a bound on unseen lowMA.',
        bound='max(observed training lowMA B maximum, training lowMA B Q3+1.5*(Q3-Q1)); quantiles use numpy linear interpolation;1.5 fixed, no coefficient grid.',
        candidates=EXTRA_POLICIES,
        fixed='Keep the already fitted margin10_lowguard cutoff; apply the stronger normal-only veto. Strict subset, so any improvement cannot be due to newly selected cases.',
        refit='Refit the same training-only margin cutoff after applying the stronger veto to development candidates; tests whether some report count can be recovered.',
        interpretation='Training heuristic, not a confidence/tolerance guarantee. A deferred lowMA is not a correctly identified lowMA.',B_from_original_data=True,valid_UTEG_only_validation=False)
    path=ROOT/'FENCE_PROTOCOL.json'
    if path.exists():assert c.read(path)==spec
    else:gates.clean_write(path,spec)
    d,_,_,_,_,folds,local=c.context()
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            path=PRIVATE/f'pred_{protocol}_{arm}.npz';z=s.analysis.load(path);specpath=path.with_suffix('.selection.json');specs=c.read(specpath)
            if all(p+'__prediction' in z for p in EXTRA_POLICIES):continue
            for p in EXTRA_POLICIES:
                z[p+'__prediction']=np.full_like(z['fixed15__prediction'],np.nan)
                z[p+'__time']=np.full_like(z['fixed15__time'],-1)
            where={int(i):j for j,i in enumerate(z['test'])}
            for rep in range(20 if protocol=='paper' else 3):
                for fi in range(1 if protocol=='paper' else 5):
                    dev,q,test=context(protocol,arm,rep,fi);dev,q=augment(protocol,arm,rep,fi,dev,q,test);jj=[where[int(i)] for i in test]
                    for p in EXTRA_POLICIES:
                        out,model=outcome(dev,q,p)
                        for key in ['prediction','time']:z[p+'__'+key][rep,jj]=out[key]
                        specs.append(dict(repeat=rep,fold=fi,policy=p,train_n=len(dev['rows']),fitted=model))
            for p in EXTRA_POLICIES:z[p+'__action']=z[p+'__time']>0
            old=z['margin10_lowguard__time']==10;new=z['margin10_fence_fixed__time']==10
            assert not np.any(new&~old)
            np.savez_compressed(path,**z);gates.clean_write(specpath,specs)
    summarize()

def summarize():
    initialize();d,*_=c.context();rows=[];per=[];stability=[]
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            path=PRIVATE/f'pred_{protocol}_{arm}.npz'
            if not path.exists():continue
            z=s.analysis.load(path);y=d['y'][z['test'],2];tc=e.state(y)
            for p in [*POLICIES,'fixed15']:
                pred=z[p+'__prediction'];t=z[p+'__time'];correct=e.state(pred)==tc
                for part,mask in [('at5',t==5),('at10',t==10),('early',(t>0)&(t<=10)),('final',t>0)]:
                    rows.append(dict(protocol=protocol,arm=arm,policy=p,subset=part,**check.score(pred,y,mask)))
                for ri in range(len(pred)):
                    for part,mask in [('at10',t[ri]==10),('final',t[ri]>0)]:
                        per.append(dict(protocol=protocol,arm=arm,policy=p,repeat=1000+ri if protocol=='paper' else ri+1,subset=part,**check.score(pred[ri],y,mask)))
                early=t==10
                stability.append(dict(protocol=protocol,arm=arm,policy=p,repeats=len(pred),nonempty_repeats=int(np.any(early,axis=1).sum()),ever_early_records=int(np.any(early,axis=0).sum()),early_every_repeat=int(np.all(early,axis=0).sum()),correct_early_every_repeat=int(np.all(early&correct,axis=0).sum()),always_correct_by_class=[int(np.sum(np.all(early&correct,axis=0)&(tc==b))) for b in [-1,0,1]]))
    gates.clean_write(ROOT/'RESULTS.json',dict(aggregate=rows,per_repeat=per,stability=stability))
    for filename,rr in [('AGGREGATE.csv',rows),('PER_REPEAT.csv',per)]:
        if not rr:continue
        flat=[{k:json.dumps(gates.safe(v),separators=(',',':')) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()} for r in rr]
        with (ROOT/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('TEN_MINUTE_SUMMARY',len(rows),len(per),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','smoke','run','summary','extend']);a=ap.parse_args()
    with threadpool_limits(limits=1):
        {'init':initialize,'smoke':smoke,'run':run,'summary':summarize,'extend':extend}[a.stage]()
