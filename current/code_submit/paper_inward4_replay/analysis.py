"""Reporting accounting and current diagnostic helpers."""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,r2_score
from threadpoolctl import threadpool_limits
from . import core as c
from .gates import clean_write
from code_submit.paper_b085_hospital_replay.equal_error import budget_mix,exact_mix

SCHEDULES={'5-10-15':[300,600,900],'10-15':[600,900],'fixed15':[900]}
RULES=['ALL_REPORT','BOUNDARY95','LR','HGB','TWO_SIDED','BOUNDARY_PAPER']

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def replay(pred,actions,times):
    first=np.full_like(pred[900],np.nan);when=np.full(pred[900].shape,-1,int);reported=np.zeros(pred[900].shape,bool)
    for h in times:
        add=actions[h]&np.isfinite(pred[h])&~reported
        first[add]=pred[h][add];when[add]=h;reported|=add
    assert np.array_equal(when>=0,reported)
    return first,when,reported

def sequences(root):
    out=root/'sequence';out.mkdir(exist_ok=True);d,*_=c.context();results={}
    for direction in c.TARGET:
        points={h:load(root/'primary'/f'{direction}_T{h}_B0.npz') for h in c.HORIZONS}
        gates={h:load(root/'gates'/f'{direction}_T{h}.npz') for h in c.HORIZONS}
        idx=points[900]['test'];y=d['y'][idx,2]
        assert all(np.array_equal(points[h]['test'],idx) for h in c.HORIZONS)
        result={};saved={};pp={h:points[h]['prediction'] for h in c.HORIZONS}
        for rule in RULES:
            ac={h:np.isfinite(pp[h]) if rule=='ALL_REPORT' else points[h]['accepted'] if rule=='BOUNDARY95' else gates[h][rule+'_action'] for h in c.HORIZONS}
            result[rule]={}
            for label,times in SCHEDULES.items():
                p,t,a=replay(pp,ac,times);m=c.summarize(p,y,a)
                m['by_time']={str(h):dict(reported=int(np.sum(t==h)),correct=int(np.sum((t==h)&(c.accounting.states(p)==c.accounting.states(y))))) for h in times}
                result[rule][label]=m
                saved[f'{rule}__{label}_prediction']=p;saved[f'{rule}__{label}_time']=t;saved[f'{rule}__{label}_action']=a
        np.savez_compressed(out/f'{direction}.npz',**saved,test=idx)
        results[direction]=result
    clean_write(out/'RESULTS.json',results)
    # all301 remains distinct; its accepted boundary95 scheme is the selected one.
    points={h:load(root/'all301'/f'T{h}.npz') for h in c.HORIZONS};local=points[900]['local'];yy=d['y'][local,2]
    result={};saved={}
    for j,direction in enumerate(c.MAIN):
        result[direction]={}
        for label,times in SCHEDULES.items():
            p,t,a=replay({h:points[h]['prediction'][j] for h in c.HORIZONS},{h:points[h]['accepted'][j] for h in c.HORIZONS},times)
            result[direction][label]=c.summarize(p,yy,a)
            saved[f'{direction}__{label}_prediction']=p;saved[f'{direction}__{label}_time']=t;saved[f'{direction}__{label}_action']=a
    np.savez_compressed(out/'all301.npz',**saved,local=local)
    clean_write(out/'ALL301_RESULTS.json',result)
    return results,result

def diagnostics(root):
    out=root/'reporting_diagnostics';out.mkdir(exist_ok=True);d,*_=c.context();comparisons={};policies={};test=None
    for direction in c.MAIN:
        g=load(root/'gates'/f'{direction}_T900.npz');p=g['prediction'];test=g['test'];y=d['y'][test,2];truth=c.accounting.states(y);pred=c.accounting.states(p)
        dev=c.read(root/'gates'/f'{direction}_T900.development.json')
        lr_actions=[];dual_actions=[];eligible=[]
        for si in range(20):
            candidates=np.flatnonzero((pred[si]==0)&np.isfinite(g['LR_score'][si])&np.isfinite(g['DUAL_score'][si]))
            n=int(np.floor(.2*len(candidates)));eligible.append(dict(seed=1000+si,common_candidates=len(candidates),selected=n))
            for key,dest in [('LR',lr_actions),('DUAL',dual_actions)]:
                a=np.zeros(len(y),bool)
                if n:
                    ordered=sorted(candidates,key=lambda i:(g[key+'_score'][si,i],str(d['ids'][test[i]])))
                    a[ordered[:n]]=True
                dest.append(a)
        la=np.array(lr_actions);da=np.array(dual_actions);ab=truth!=0
        comparisons[direction]=dict(scope='Common finite-score predicted-normal candidates; exact floor20% quota within each seed.',
            eligible=eligible,LR=dict(released=int(la.sum()),abnormal_as_normal=int(np.sum(la&ab))),
            DUAL=dict(released=int(da.sum()),abnormal_as_normal=int(np.sum(da&ab))),
            paired_error_fraction_difference=c.accounting.paired_ratio((da&ab).astype(int)-(la&ab),la,d['groups'][test]))
        # Policies are defined from DEVELOPMENT score quantiles only. Test labels
        # below assess attainable error/yield combinations retrospectively.
        configs=[('none',0,0),('all',0,0),('boundary95',0,0)]
        configs += [('LR',q,0) for q in np.linspace(0,1,21)]
        configs += [('DUAL',q,t) for q in np.linspace(0,1,11) for t in np.linspace(0,1,11)]
        actions=np.zeros((len(configs),20,len(y)),bool)
        for si in range(20):
            scores=dev[si]['scores']
            def threshold(key,q):
                v=np.array([np.nan if x is None else x for x in scores[key]],float);v=v[np.isfinite(v)]
                return float(np.quantile(v,q,method='higher')) if len(v) else -np.inf
            for k,(kind,q,t) in enumerate(configs):
                if kind=='none':continue
                if kind=='all':a=np.isfinite(p[si])
                elif kind=='boundary95':a=g['boundary95_action'][si]
                elif kind=='LR':a=(g['LR_score'][si]<=threshold('LR',q))&np.isfinite(g['LR_score'][si])
                else:a=((pred[si]==0)&(g['DUAL_score'][si]<=threshold('DUAL',q))&np.isfinite(g['DUAL_score'][si]))|((pred[si]!=0)&(g['LR_score'][si]<=threshold('LR',t))&np.isfinite(g['LR_score'][si]))
                actions[k,si]=a&np.isfinite(p[si])
        correct=actions&(pred==truth);errors=actions&(pred!=truth)
        ey=errors.mean((1,2));cy=correct.mean((1,2));budgets=(.005,.01,.02,.05,.075,.10)
        matches={str(b):dict(budget_mix=budget_mix(ey,cy,b),exact_mix=exact_mix(ey,cy,b)) for b in budgets}
        policies[direction]=dict(configs=configs,actions=actions,correct=correct,errors=errors,ey=ey,cy=cy,matches=matches)
        np.savez_compressed(out/f'{direction}_policy_arrays.npz',actions=actions,correct=correct,errors=errors,test=test)
    clean_write(out/'MATCHED20.json',comparisons)
    groups=d['groups'][test];uniq=np.unique(groups);rng=np.random.default_rng(20261102);cross={}
    # Same10% identity groups for every method; remaining groups assess selected
    # policy. Repeated partitions are exploratory, not untouched external data.
    for direction,m in policies.items():
        rows=[]
        for repeat in range(20):
            chosen=np.random.default_rng(20261102+repeat).permutation(uniq)[:max(1,int(np.ceil(.1*len(uniq))))]
            cal=np.isin(groups,chosen);held=~cal
            ey=m['errors'][:,:,cal].mean((1,2));cy=m['correct'][:,:,cal].mean((1,2))
            ids=np.flatnonzero(ey<=.10+1e-12);best=min(ids,key=lambda i:(-cy[i],ey[i],i))
            a=m['actions'][best,:,held];err=m['errors'][best,:,held];cor=m['correct'][best,:,held]
            rows.append(dict(repeat=repeat,calibration_records=int(cal.sum()),held_records=int(held.sum()),policy_index=int(best),
                correct_yield=float(cor.mean()),incorrect_yield=float(err.mean()),coverage=float(a.mean()),
                conditional_error=float(err.sum()/a.sum()) if a.any() else None))
        cross[direction]=rows
    paired={}
    target=policies['target_only_SA']
    for arm in c.MAIN[1:]:
        src=policies[arm];paired[arm]={}
        for b in (.005,.01,.02,.05,.075,.10):
            left=target['matches'][str(b)]['budget_mix'];right=src['matches'][str(b)]['budget_mix']
            if left is None or right is None:paired[arm][str(b)]=None;continue
            def mix(m,sel,key):return (1-sel['weight_j'])*m[key][sel['strategy_i']]+sel['weight_j']*m[key][sel['strategy_j']]
            diff=mix(src,right,'correct').astype(float)-mix(target,left,'correct')
            paired[arm][str(b)]=dict(correct_yield_difference=c.accounting.paired_ratio(diff,np.ones_like(diff),groups),
                target_actual_error=left['error_yield'],source_actual_error=right['error_yield'],
                intervals='Descriptive group bootstrap conditional on retrospective chosen mixtures; not selection-adjusted.')
    clean_write(out/'EXPLORATORY_ERROR_MATCHING.json',dict(test_label_aware=True,denominator='incorrect reports / full test cohort, not conditional error',
        policies={k:dict(configs=v['configs'],error_yield=v['ey'],correct_yield=v['cy'],matching=v['matches']) for k,v in policies.items()},paired=paired,
        limitation='Retrospective convex combinations are not validated deployment policies or error non-inferiority.'))
    clean_write(out/'EXTRA_LABEL_CROSS_CALIBRATION.json',dict(calibration_fraction=.10,error_yield_cap=.10,partitions=20,
        grouping='identity proxy; same subsets for all methods',results=cross,independent_validation=False))

def paired_primary(root):
    d,*_=c.context();result={}
    z=load(root/'primary/target_only_SA_T900_B0.npz');y=d['y'][z['test'],2];t=c.accounting.states(y);groups=d['groups'][z['test']]
    base=c.accounting.states(z['prediction']);ab=t!=0;normal=t==0
    for arm in c.MAIN[1:]:
        p=load(root/'primary'/f'{arm}_T900_B0.npz');pred=c.accounting.states(p['prediction'])
        result[arm]=dict(FNR_difference=c.accounting.paired_ratio(((pred==0)&ab).astype(int)-((base==0)&ab),np.broadcast_to(ab,base.shape),groups),
            FPR_difference=c.accounting.paired_ratio(((pred!=0)&normal).astype(int)-((base!=0)&normal),np.broadcast_to(normal,base.shape),groups),
            MAE_difference=c.accounting.paired_ratio(abs(p['prediction']-y)-abs(z['prediction']-y),np.ones_like(base),groups))
    clean_write(root/'PAIRED_PRIMARY_INTERVALS.json',result)

def table(head,rows):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def pct(x):return 'NA' if x is None else f'{x:.2%}'
def num(x):return 'NA' if x is None else f'{x:.3f}'




