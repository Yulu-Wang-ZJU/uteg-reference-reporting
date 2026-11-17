"""Fixed linear sensitivity, saved refit representations, and independent local internal repetitions."""
import experiment as e
import additional as ad
import numpy as np
import argparse,json,time
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedGroupKFold
r=e.r;O=e.O;LINEAR_CACHE={}

def linear_predict(arm,h,train,query,representation='B',loss='inward4',prior_override=None):
    assert representation=='B' and loss=='inward4' and prior_override is None
    d,*_=r.c.context();tr=np.asarray(train);qu=np.asarray(query);x=r.c.feature([],[],h,'B');y=d['medical'][:,2]
    assert not set(d['groups'][tr])&set(d['groups'][qu])
    source=r.c.source_rows(arm,h)
    def basis(ix):
        mean=x[ix].mean(0);scale=np.maximum(x[ix].std(0),1e-6)
        return np.c_[(x-mean)/scale,np.ones(len(x))]
    key=(arm,h)
    if key not in LINEAR_CACHE:
        if len(source):
            zs=basis(source);w=np.zeros(len(source))
            for ct in np.unique(d['centers'][source]):
                mask=d['centers'][source]==ct;w[mask]=r.c.selected.models.weights(y[source][mask],0,'inward4')/mask.sum()
            w/=w.mean();coef,info=r.c.selected.models.solve(zs[source],(y[source]-60)/10,w,'residual',4.,alpha=.01)
            LINEAR_CACHE[key]=(zs,coef)
        else:LINEAR_CACHE[key]=None
    src=LINEAR_CACHE[key];z=basis(tr);base=np.maximum(60+10*src[0][tr]@src[1],0) if src is not None else np.full(len(tr),60.)
    bq=np.maximum(60+10*src[0][qu]@src[1],0) if src is not None else np.full(len(qu),60.)
    w=r.c.selected.models.weights(y[tr],0,'inward4');coef,info=r.c.selected.solve_offset(z[tr],(y[tr]-60)/10,w,(base-60)/10)
    pred=bq+10*z[qu]@coef
    # Apply the same directional endpoint guard to the complete linear prediction.
    mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),1e-6);lo=x[tr,0].min();hi=x[tr,0].max();clip=np.clip(x[qu,0],lo,hi)[:,None]
    zc=np.c_[(clip-mean)/scale,np.ones(len(qu))]
    if src is None:bc=np.full(len(qu),60.)
    else:
        sm=x[source].mean(0);ss=np.maximum(x[source].std(0),1e-6);bc=np.maximum(60+10*np.c_[(clip-sm)/ss,np.ones(len(qu))]@src[1],0)
    endpoint=bc+10*zc@coef;below=x[qu,0]<lo;above=x[qu,0]>hi;pred[below]=np.minimum(pred[below],endpoint[below]);pred[above]=np.maximum(pred[above],endpoint[above])
    return np.maximum(pred,0),dict(alpha=.01,cost=4,solver=info),x

def variant_round(d,plan,role,item,variant):
    dest=O/f'{variant}/round_{item["rep"]:02d}'
    if (dest/'COMPLETE.json').exists():return
    e.activate(item,plan,dest);LINEAR_CACHE.clear();original_feature=r.c.feature;original_source=r.c.source_feature
    if variant=='linear':r.c.predict=linear_predict
    else:
        path=r.current.ROOT/'manuscript_sync/experiments/signal/REFITS.npz'
        with np.load(path,allow_pickle=True) as z:
            np.testing.assert_array_equal(z['ids'],d['ids']);x=z[variant].copy();assert np.isfinite(x).all()
        def feature(train,query,h,representation='B'):
            return x[:,{'B':[1],'three':[0,1,2],'four':[0,1,2,3]}[representation]].copy()
        r.c.feature=feature;r.c.source_feature=lambda arm,h,representation='B':feature([],[],h,representation);r.c._SOURCE.clear()
    rows=[];policies={};start=time.time()
    try:
        for budget in [54,60]:
            sp=e.specification(item,budget,role)
            for arm in e.ARMS:
                z=e.collect(d,sp,arm,item['seed'],900);dr,qr,au=e.risks(d,sp,z['dev_rows'],900);rr,pol,acts,_=e.evaluate_rules(d,z,dr,qr,900)
                rows.extend(dict(v,rep=item['rep'],budget=budget,arm=arm,variant=variant) for v in rr);policies[f'{budget}_{arm}']=pol
                np.savez_compressed(dest/f'dev{budget}_{arm}.npz',**z,**{'action_'+k.replace(' ','_'):v for k,v in acts.items()})
        r.write(dest/'RESULTS.json',rows);r.write(dest/'POLICIES.json',policies);r.write(dest/'COMPLETE.json',dict(rep=item['rep'],seconds=time.time()-start));print(variant,item['rep']+1,round(time.time()-start,1),flush=True)
    finally:r.c.feature=original_feature;r.c.source_feature=original_source

def internal_round(d,plan,role,item):
    rep=item['rep'];dest=O/f'internal301/round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    e.activate(item,plan,dest);local=np.asarray(r.c.context()[6]);sample=r.sample(local,d,np.random.default_rng(item['seed']+301000));assert len(sample)==271
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=item['seed']);splits=[];rows=[];policies={};preds={a:np.full(len(sample),np.nan) for a in e.ARMS};actions={a:{q:np.zeros(len(sample),bool) for q in ['All report','common_B','HGB','Logistic','Boundary','B95']} for a in e.ARMS};where={int(v):i for i,v in enumerate(sample)};feasible={a:[] for a in e.ARMS};start=time.time()
    for fi,(aa,bb) in enumerate(outer.split(sample,r.state(d['medical'][sample,2]),d['groups'][sample])):
        train=sample[aa];test=sample[bb];assert not set(d['groups'][train])&set(d['groups'][test])
        inner=r.folds(train,d,item['seed']+100+fi)
        positions=[dict(A=f['T'],B=f['V'],inner=r.folds(f['T'],d,item['seed']+1000+fi*10+j)) for j,f in enumerate(inner)]+[dict(A=train,B=test,inner=inner)]
        sp=dict(pool=train,test=test,positions=positions);splits.append(sp)
        for arm in e.ARMS:
            z=e.collect(d,sp,arm,item['seed']+fi,900);dr,qr,au=e.risks(d,sp,z['dev_rows'],900);rr,pol,acts,_=e.evaluate_rules(d,z,dr,qr,900);ix=[where[int(v)] for v in test];preds[arm][ix]=z['prediction']
            for q,act in acts.items():actions[arm][q][ix]=act
            policies[f'{fi}_{arm}']=pol;feasible[arm].append(pol['common_B']['feasible'])
            np.savez_compressed(dest/f'fold{fi}_{arm}.npz',**z,dev_risk=dr,test_risk=qr,**{'action_'+k.replace(' ','_'):v for k,v in acts.items()})
    for arm in e.ARMS:
        assert np.isfinite(preds[arm]).all()
        for q,act in actions[arm].items():rows.append(dict(rep=rep,arm=arm,rule=q,feasible_folds=sum(feasible[arm]),test=e.n.extended_metrics(preds[arm],d['medical'][sample,2],act)))
        np.savez_compressed(dest/f'{arm}.npz',test=sample,prediction=preds[arm],**{'action_'+k.replace(' ','_'):v for k,v in actions[arm].items()})
    r.write(dest/'SPLITS.json',splits);r.write(dest/'POLICIES.json',policies);r.write(dest/'RESULTS.json',rows);r.write(dest/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start));print('INTERNAL301',rep+1,round(time.time()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['linear','EXP3','EXP4','internal301']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    protocol=dict(stage=args.stage,repeats=100,linear='One amplitude input, same asymmetric cost4,alpha.01,class weights,source weighting and directional endpoint guard. No hyperparameter selection.',representation='Saved EXP3 and EXP4 refits; current one-amplitude MA branch for both, all other rules fixed. This tests the representation feeding the current branch, not the historical three-feature vs four-feature predictor.',internal301='Separate exploratory internal evaluation. Sample90% of301 without replacement with grouped class quotas;271 records per repetition,outer5 folds and inner3 folds. Source90% draws shared with primary.100 repetitions,all retained; not54/60 development comparison. Whole-research adaptive reuse is not removed by nested folds.',no_manuscript_changes=True)
    pp=O/(args.stage+'_PROTOCOL.json')
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:r.write(pp,protocol)
    d,plan,role,models,items=e.init();e.GLOBAL_D=d
    if args.stage=='internal301':
        import fast_policy
        check=json.loads((O/'EXACT_SELECTOR_CHECK.json').read_text());assert check['policies_exactly_identical'] and check['cases']==615
        e.cal.choose_normal=fast_policy.choose_normal
    for item in items[args.start:args.stop]:
        if args.stage=='internal301':internal_round(d,plan,role,item)
        else:variant_round(d,plan,role,item,args.stage)
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in e.cal.protocol()['input_hashes'].items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
