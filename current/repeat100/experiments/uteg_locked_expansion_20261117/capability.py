"""Repeat the original HGB capability diagnostic on the updated fixed predictions."""
import experiment as e
import aggregate as ag
import numpy as np
import json
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

def calibrated(z,query,d):
    score=z['dev_score_HGB'];error=e.r.state(z['dev_prediction'])!=e.r.state(d['medical'][z['dev_rows'],2]);fit=np.isfinite(score);xs=score[fit];ys=error[fit].astype(float)
    if len(np.unique(xs))==1:risk=np.full(len(query),ys.mean())
    else:risk=IsotonicRegression(increasing=True,out_of_bounds='clip',y_min=0,y_max=1).fit(xs,ys).predict(query)
    return risk+1e-6*np.searchsorted(np.sort(xs),query,side='right')/len(xs)

def matched(counts,target):
    accuracy=counts[:,1]/counts[:,0];good=np.flatnonzero(accuracy+1e-12>=target)
    if not len(good):return None,'unattainable'
    i=int(good[-1])
    if abs(accuracy[i]-target)<1e-12:return counts[i].copy(),'exact_integer_count'
    if i==len(counts)-1:return None,'full_rate_above_target_no_exact_match'
    diff=counts[i+1]-counts[i];fraction=(target*counts[i,0]-counts[i,1])/(diff[1]-target*diff[0]);assert -1e-10<=fraction<=1+1e-10
    value=counts[i]+fraction*diff;assert abs(value[1]/value[0]-target)<1e-12;return value,'interpolated_counts'

def metric(c,n):
    if c is None:return dict(reporting_rate=None,accuracy=None,FNR=None,FPR=None)
    reported,correct,fn,ab,fp,normal=c
    return dict(reported=float(reported),correct=float(correct),FN=float(fn),abnormal_released=float(ab),FP=float(fp),normal_released=float(normal),reporting_rate=float(reported/n),accuracy=float(correct/reported),FNR=float(fn/ab) if ab else None,FPR=float(fp/normal) if normal else None)

def main():
    d,*_=e.init();rows=[];pooled={};curves=[]
    for budget in [54,60]:
        for rep in range(100):
            for arm in e.ARMS:
                z={h:np.load(e.O/f'primary/dev{budget}/time{h}/round_{rep:02d}/{arm}.npz') for h in [300,600,900]};idx=z[300]['test'];truth=e.r.state(d['medical'][idx,2]);ids=d['ids'][idx];n=len(idx)
                for h in [300,600,900]:
                    p=z[h]['prediction'].copy();score=z[h]['score_HGB'].copy()
                    if h==600:
                        use=z[300]['action_HGB']&~z[600]['action_HGB'];p[use]=z[300]['prediction'][use];score=np.where(use,calibrated(z[300],z[300]['score_HGB'],d),calibrated(z[600],z[600]['score_HGB'],d))
                    order=np.lexsort((ids.astype(str),score));pc=e.r.state(p)[order];tc=truth[order]
                    events=np.c_[np.ones(n),pc==tc,(pc==1)&(tc!=1),tc!=1,(pc!=1)&(tc==1),tc==1];counts=np.cumsum(events,0);key=budget,arm,h
                    pooled[key]=pooled.get(key,np.zeros_like(counts))+counts
                    for rate in [.1,.2,.4]:
                        k=int(np.floor(rate*n));rows.append(dict(budget=budget,rep=rep,arm=arm,time=h,condition='fixed_reporting_rate',level=rate,status='integer_count',test=metric(counts[k-1],n)))
                    for acc in [.85,.9,.95]:
                        value,status=matched(counts,acc);rows.append(dict(budget=budget,rep=rep,arm=arm,time=h,condition='matched_observed_accuracy',level=acc,status=status,test=metric(value,n)))
                for zz in z.values():zz.close()
    pooledrows=[]
    for (budget,arm,h),counts in pooled.items():
        n=18100
        for rate in [.1,.2,.4]:pooledrows.append(dict(budget=budget,arm=arm,time=h,condition='fixed_reporting_rate',level=rate,test=metric(counts[int(np.floor(rate*181))-1],n)))
        for acc in [.85,.9,.95]:
            value,status=matched(counts,acc);pooledrows.append(dict(budget=budget,arm=arm,time=h,condition='matched_observed_accuracy',level=acc,status=status,test=metric(value,n)))
        curves.extend(dict(budget=budget,arm=arm,time=h,rank=i+1,**metric(v,n)) for i,v in enumerate(counts))
    ag.write('Fig4_HGB_capability',rows,['budget','arm','time','condition','level']);e.r.write(ag.S/'Fig4_HGB_capability_pooled.json',pooledrows);ag.csvwrite(ag.S/'Fig4_HGB_capability_pooled.csv',pooledrows);ag.csvwrite(ag.S/'Fig4_HGB_rank_curves_pooled.csv',curves)
    e.r.write(e.O/'CAPABILITY_PROTOCOL.json',dict(rule='Original manuscript HGB diagnostic, not a scalar score invented for the fixed common_B composite.',selection='5=current;10=retain5only if5passes original developer HGB gate and10fails;15=current. Development-only isotonic cross-time scores; ties by record ID.',levels=dict(reporting_rate=[.1,.2,.4],observed_accuracy=[.85,.9,.95]),accuracy_matching='Retrospective count interpolation using evaluation outcomes, not selection of a deployment threshold. Per-round infeasible exact matches retained as missing; valid-count summaries provided. Pooled100points separate from mean of100matched points.',no_monotonicity_enforcement=True,no_forced_zero_FNR=True,no_manuscript_changes=True));print('CAPABILITY COMPLETE',len(rows))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
