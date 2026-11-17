"""Fixed-cohort figure data and descriptive associations; no selection for plot accuracy."""
import experiment as e
import additional as ad
import aggregate as ag
import numpy as np
import pandas as pd
import json,copy
from threadpoolctl import threadpool_limits
r=e.r;O=e.O

def reference(d,plan,role,models):
    folder=O/'reference201'
    if (folder/'COMPLETE.json').exists():return
    seed=1000;item=dict(rep=0,seed=seed,sources={ct:np.flatnonzero(d['centers']==ct).tolist() for ct in ['SAHZU','Huzhou','Hubei']})
    e.activate(item,plan,folder);train=np.asarray(role['roles']['SAHZU']['pool_full']);test=np.asarray(role['roles']['SAHZU']['test']);positions=[]
    for j,f in enumerate(r.folds(train,d,seed)):positions.append(dict(A=f['T'],B=f['V'],inner=r.folds(f['T'],d,seed+101+j)))
    positions.append(dict(A=train,B=test,inner=r.folds(train,d,seed+201)));sp=dict(pool=train,test=test,positions=positions)
    assert len(sp['pool'])==60 and len(sp['test'])==201;rows=[];policies={}
    for arm in e.ARMS:
        z=e.collect(d,sp,arm,seed,900);dr,qr,au=e.risks(d,sp,z['dev_rows'],900);rr,pol,acts,_=e.evaluate_rules(d,z,dr,qr,900);four,_=ad.primary_four(d,models,arm,sp,z['prediction']);rows.extend(dict(v,arm=arm) for v in rr);policies[arm]=pol
        np.savez_compressed(folder/(arm+'.npz'),**z,prediction_four=four,truth_four=d['medical'][z['test']],**{'action_'+k.replace(' ','_'):v for k,v in acts.items()})
    r.write(folder/'RESULTS.json',rows);r.write(folder/'POLICIES.json',policies);r.write(folder/'SPLIT.json',sp);r.write(folder/'COMPLETE.json',dict(seed=seed,all_original_source_records=True,local_development=60,test=201,display_cohort_selected_before_prediction=True))

def associations(d):
    boundary=[];early=[]
    path=r.current.ROOT/'data/CURRENT_SCALED_CURVES.npz'
    with np.load(path,allow_pickle=False) as zz:
        np.testing.assert_array_equal(zz['ids'],d['ids']);x=zz['scaled_zeroed'][:,:300];mask=zz['observed_mask'][:,:300]
    growth=np.full(len(d['medical']),np.nan)
    for i,(xx,mm) in enumerate(zip(x,mask)):
        obs=xx[mm];span=np.percentile(obs,95)-np.percentile(obs,5)
        if len(obs)>=60 and span>0:growth[i]=(np.median(obs[-30:])-np.median(obs[-60:-30]))/span
    for budget in [54,60]:
        for rep in range(100):
            for arm in e.ARMS:
                for h in [300,900]:
                    with np.load(O/f'primary/dev{budget}/time{h}/round_{rep:02d}/{arm}.npz') as z:
                        idx=z['test'];y=d['medical'][idx,2];wrong=r.state(z['prediction'])!=r.state(y);distance=np.minimum(abs(y-50),abs(y-70))
                        base=dict(budget=budget,rep=rep,arm=arm)
                        if h==900:
                            boundary.append(dict(**base,errors=int(wrong.sum()),**{f'within_{t}mm':int(np.sum(wrong&(distance<=t))) for t in [2,3,5]},fraction_within_3mm=float(np.mean(distance[wrong]<=3)) if wrong.any() else None));continue
                        normal=r.state(y)==1;assert np.isfinite(growth[idx][normal]).all();bands=pd.qcut(growth[idx][normal],3,labels=False);groups=np.full(len(y),-1);groups[normal]=bands
                        for rule in ['common_B','HGB']:
                            act=z['action_'+rule]
                            for b,name in enumerate(['Lower','Middle','Upper']):
                                keep=groups==b;n=int(keep.sum());early.append(dict(**base,rule=rule,growth_third=name,records=n,errors=int(wrong[keep].sum()),withheld=int((~act[keep]).sum()),error_rate=float(wrong[keep].mean()),withheld_rate=float((~act[keep]).mean())))
    ag.write('boundary_associations',boundary,['budget','arm']);ag.write('early_growth_associations',early,['budget','arm','rule','growth_third'])
    r.write(O/'ASSOCIATION_PROTOCOL.json',dict(source_sha256=r.sha(path),same_growth_formula=True,strata='Within each real test subset, reference-normal records divided into thirds; descriptive only, no gate training or selection.',point='Retains new fixed gate and original HGB separately; no causal mechanism claim.',internal301='15-minute boundary summaries can be computed from internal301; no new5-minute internal nested run claimed.'))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        d,plan,role,models,items=e.init();e.GLOBAL_D=d;reference(d,plan,role,models);associations(d);print('REFERENCE AND ASSOCIATIONS COMPLETE')
