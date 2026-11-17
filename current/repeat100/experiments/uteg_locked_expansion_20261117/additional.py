"""Predeclared supplementary repetitions; locked methods, no test tuning."""
import experiment as e
import argparse,json,time,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
r=e.r;O=e.O

def spec_for(item,role,target,mode):
    return e.specification(item,'sampled' if mode=='sampled90' else 'full',role,target)

def primary_four(d,models,arm,sp,pred,h=900):
    tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);src=r.c.source_rows(arm,h);x=r.c.feature([],[],h,'four')
    source=src if len(src) else tr;masked=np.full_like(d['medical'],np.nan);masked[source]=d['medical'][source];data=dict(d,x=x,medical=masked,ci=np.full_like(d['ci'],np.nan));data['ci'][source]=d['ci'][source]
    model=models.fit_source(data,source,[],'RIDGE-10',0)
    pt=models.predict(model,x[tr]);pq=models.predict(model,x[te]);raw=pq.copy()
    if len(src):
        hgb=models.fit_cal(pt,d['medical'][tr],model['ym'],model['ys']);assert hgb['finite'];pq=models.project(models.predict_cal(hgb,pq));prior=r.c.source_model(arm,h);raw[:,2]=r.c.source_predict(prior,x[te,1:2])
    pq[:,2]=pred
    return pq,raw

def nine_round(d,plan,role,models,item):
    rep=item['rep'];folder=O/f'nine/round_{rep:02d}'
    if (folder/'COMPLETE.json').exists():return
    start=time.time();e.activate(item,plan,folder);rows=[];mapping=[];rules={}
    for mode in ['sampled90','full_local']:
        for arm,target in r.c.TARGET.items():
            sp=spec_for(item,role,target,mode);size=len(sp['pool']);h=900
            if arm in e.ARMS:
                budget=54 if mode=='sampled90' else 60;src=O/f'primary/dev{budget}/time900/round_{rep:02d}';z=np.load(src/(arm+'.npz'));rr=json.loads((src/'RESULTS.json').read_text());out=[x for x in rr if x['arm']==arm];picks=json.loads((src/'POLICIES.json').read_text())[arm]
                dr=z['dev_risk'];qr=z['test_risk'];actions={x['rule']:z['action_'+x['rule'].replace(' ','_')] for x in out}
            else:
                z=e.collect(d,sp,arm,item['seed'],h);dr,qr,audit=e.risks(d,sp,z['dev_rows'],h);out,picks,actions,_=e.evaluate_rules(d,z,dr,qr,h);r.write(folder/f'{mode}_{arm}_risk_training.json',audit)
            pp,raw=primary_four(d,models,arm,sp,z['prediction']);truth=d['medical'][np.asarray(sp['test'])]
            mapping.append(dict(rep=rep,mode=mode,arm=arm,development_n=size,**r.four_metrics(pp,truth),all_report=r.metrics(pp[:,2],truth[:,2],np.ones(len(truth),bool))))
            if not arm.startswith('target_only'):mapping.append(dict(rep=rep,mode=mode,arm=arm,development_n=size,stage='unadapted',**r.four_metrics(raw,truth)))
            rows += [dict(v,rep=rep,mode=mode,arm=arm,development_n=size,time=900) for v in out];rules[mode+'_'+arm]=picks
            np.savez_compressed(folder/f'{mode}_{arm}.npz',test=np.asarray(sp['test']),dev_rows=z['dev_rows'],prediction=z['prediction'],dev_prediction=z['dev_prediction'],prediction_four=pp,source_prediction_four=raw,dev_risk=dr,test_risk=qr,**{k:z[k] for k in ['dev_score_'+q for q in e.a.RULES]+['score_'+q for q in e.a.RULES]},**{'action_'+k.replace(' ','_'):v for k,v in actions.items()})
            if hasattr(z,'close'):z.close()
    r.write(folder/'RESULTS.json',rows);r.write(folder/'MAPPING.json',mapping);r.write(folder/'POLICIES.json',rules);r.write(folder/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start));print('NINE',rep+1,round(time.time()-start,1),flush=True)

def subset(pool,d,size,seed):
    pool=np.asarray(pool,int);states=r.state(d['medical'][pool,2]);counts=np.bincount(states,minlength=3);goal=counts*size/len(pool);quota=np.floor(goal).astype(int)
    for k in sorted(range(3),key=lambda k:(-(goal[k]-quota[k]),k))[:size-quota.sum()]:quota[k]+=1
    units=[pool[d['groups'][pool]==g] for g in np.unique(d['groups'][pool])];multi=[u for u in units if len(u)>1];singles=[np.array([u[0] for u in units if len(u)==1 and r.state(d['medical'][u,2])[0]==k],int) for k in range(3)];rng=np.random.default_rng(seed)
    for _ in range(10000):
        chosen=[u for u in multi if rng.random()<size/len(pool)];chosen=np.concatenate(chosen) if chosen else np.array([],int);remaining=quota-np.bincount(r.state(d['medical'][chosen,2]),minlength=3)
        if any(remaining[k]<0 or remaining[k]>len(singles[k]) for k in range(3)):continue
        out=np.sort(np.r_[chosen,*[rng.choice(singles[k],remaining[k],replace=False) for k in range(3)]])
        assert len(out)==size;return out
    raise RuntimeError('No grouped sample supports requested quotas')

def ridge_fit(d,train,query,seed):
    y=d['medical'][:,2];x=d['x'];folds=r.folds(train,d,seed);errors=[]
    for alpha in [.1,1,10,100]:
        dif=[]
        for fold in folds:
            tr=fold['T'];va=fold['V'];scale=StandardScaler().fit(x[tr]);model=Ridge(alpha=alpha).fit(scale.transform(x[tr]),y[tr]);dif.extend(abs(model.predict(scale.transform(x[va]))-y[va]))
        errors.append(float(np.mean(dif)))
    idx=min(range(4),key=lambda i:(errors[i],-i));alpha=[.1,1,10,100][idx];scale=StandardScaler().fit(x[train]);model=Ridge(alpha=alpha).fit(scale.transform(x[train]),y[train]);return np.maximum(model.predict(scale.transform(x[query])),0),dict(alpha=alpha,development_MAE=errors)

def budgets_round(d,plan,role,models,item):
    rep=item['rep'];folder=O/f'budgets/round_{rep:02d}'
    if (folder/'COMPLETE.json').exists():return
    e.activate(item,plan,folder);base=np.asarray(role['roles']['SAHZU']['pool_full']);test=np.asarray(item['targets']['SAHZU']['test']);rows=[];saved={};spec={}
    for size in [20,40,60]:
        train=base if size==60 else subset(base,d,size,item['seed']+size*1000);spec[str(size)]=train
        for arm in e.ARMS:
            pred,_,_=r.c.predict(arm,900,train,test);rows.append(dict(rep=rep,budget=size,arm=arm,**r.metrics(pred,d['medical'][test,2],np.ones(len(test),bool))));saved[f'{arm}_{size}']=pred
        pred,audit=ridge_fit(d,train,test,item['seed']);rows.append(dict(rep=rep,budget=size,arm='target_only_Ridge',**r.metrics(pred,d['medical'][test,2],np.ones(len(test),bool)),selection=audit));saved[f'Ridge_{size}']=pred
    np.savez_compressed(folder/'PREDICTIONS.npz',test=test,**saved);r.write(folder/'TRAINING.json',spec);r.write(folder/'RESULTS.json',rows);r.write(folder/'COMPLETE.json',dict(rep=rep));print('BUDGET',rep+1,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['nine','budgets']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    proto=dict(parent=r.sha(O/'PROTOCOL.json'),stage=args.stage,repeats=100,nine='Nine transfer configurations plus3target-only;90% sampled localpool and full original localpool. Main local sizes54/60,other hospitals36/40. Same samples and seeds. Primary4readout companion Ridge/Huber, matched MA rule; no retuning.',budgets='20/40/60 sampled exclusively from original60local pool, quotas based on development labels and grouped units. Same source/test90% draws. Fixed RBF/asymmetric method; original Ridge alpha selection [.1,1,10,100] by developer3foldMAE, ties favour larger alpha. No reporting-selection bias in MA errors.')
    path=O/(args.stage+'_PROTOCOL.json')
    if path.exists():assert json.loads(path.read_text())==proto
    else:r.write(path,proto)
    d,plan,role,models,items=e.init();e.GLOBAL_D=d
    for item in items[args.start:args.stop]:
        (nine_round if args.stage=='nine' else budgets_round)(d,plan,role,models,item)
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in e.cal.protocol()['input_hashes'].items())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
