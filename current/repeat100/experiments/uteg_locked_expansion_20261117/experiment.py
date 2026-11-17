"""Matched100-repeat completion under locked common_B. Never select from test."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import sys,json,time,copy,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_risk_calibration_20261117'))
import calibrate as cal
import numpy as np
from threadpoolctl import threadpool_limits
import joblib
r=cal.r;a=cal.a;n=cal.n;b=cal.b;c=cal.c
O=Path(__file__).parent
LOCK=Path(str(_repeat_root / 'experiments' / 'uteg_locked_commonB_20261117'))
ARMS=['target_only_SA','HZ_to_SA','HZ+HB_to_SA']

def init():
    d,plan,role,models=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text());return d,plan,role,models,items

def activate(item,plan,dest):
    r.c.OUT=dest;dest.mkdir(parents=True,exist_ok=True)
    a.f.draw(item,plan);a.configure('B')

def specification(item,budget,role,target='SAHZU'):
    if target=='SAHZU' and budget in [54,60]:return cal.support.spec(item,budget)
    if budget=='sampled':return item['targets'][target]
    pool=np.asarray(role['roles'][target]['pool_full']);test=np.asarray(item['targets'][target]['test']);seed=item['seed'];pos=[]
    for j,fold in enumerate(r.folds(pool,GLOBAL_D,seed)):
        pos.append(dict(A=fold['T'],B=fold['V'],inner=r.folds(fold['T'],GLOBAL_D,seed+101+j)))
    pos.append(dict(A=pool,B=test,inner=r.folds(pool,GLOBAL_D,seed+201)))
    return dict(pool=pool,test=test,positions=pos)

def score(out,d,seed,arm,pos,h):
    train=out['train'];query=out['query'];p=out['inner'];pq=out['prediction']
    _,_,ff=r.gates.fields(train,out['inner_b'],p,train,h);_,_,fq=r.gates.fields(query,out['query_b'],pq,train,h)
    pc=r.gates.fr.ma_class(p);tc=r.gates.fr.ma_class(d['medical'][train,2]);qc=r.gates.fr.ma_class(pq)
    lr=r.gates.fr._fit_ma_gate(ff,pc,tc,f'inward_{arm}_{h}_{seed}_{pos}',seed)
    hg=r.gates.rc.fit_hgb(ff,pc,tc,np.ones(len(train),bool),direction=arm,seed=seed,pos=pos)
    return dict(Logistic=r.gates.fr.score_ma_gate(lr,fq,qc),HGB=r.gates.rc.score_hgb(hg,fq,qc,np.ones(len(query),bool)),Boundary=r.gates.rc.margin_scores(pq))

def collect(d,sp,arm,seed,h):
    parts=[]
    for pos,part in enumerate(sp['positions']):
        out=r.c.evaluate(part['A'],part['B'],part['inner'],arm,h);ss=score(out,d,seed,arm,pos,h)
        parts.append((out,ss))
    devrows=np.concatenate([x[0]['query'] for x in parts[:3]]);assert set(devrows)==set(sp['pool'])
    z=dict(dev_rows=devrows,test=np.asarray(sp['test']),dev_prediction=np.concatenate([x[0]['prediction'] for x in parts[:3]]),prediction=parts[3][0]['prediction'])
    for k in a.RULES:z['dev_score_'+k]=np.concatenate([x[1][k] for x in parts[:3]]);z['score_'+k]=parts[3][1][k]
    return z

def risks(d,sp,devrows,h):
    x=r.c.feature([],[],h,'four');dy=d['medical'][:,2];loc={int(v):i for i,v in enumerate(devrows)};dr=np.zeros((len(devrows),2));audit=[]
    for part in sp['positions'][:3]:
        values,au=n.risk_fit(x,dy,d['groups'],part['A'],part['B'],[1]);dr[[loc[int(v)] for v in part['B']]]=values;audit.append(au)
    qr,au=n.risk_fit(x,dy,d['groups'],sp['pool'],sp['test'],[1]);audit.append(au)
    return dr,qr,audit

def evaluate_rules(d,z,dr,qr,h):
    dp=z['dev_prediction'];dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2];ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
    x=r.c.feature([],[],h,'four');dB=x[z['dev_rows'],1];qB=x[z['test'],1];devrisk=np.c_[-dB,dB];queryrisk=np.c_[-qB,qB]
    norm=r.state(dp)==1
    if norm.any():
        cap=float(np.quantile(dr[norm,1],.9));pick=cal.choose(dp,dy,dr,devrisk,ds,cap)
    else:pick=dict(feasible=False,reason='No development predicted-normal cases')
    da=cal.action(pick,dp,dr,devrisk,ds) if pick['feasible'] else np.zeros(len(dp),bool)
    act=cal.action(pick,z['prediction'],qr,queryrisk,qs) if pick['feasible'] else np.zeros(len(ty),bool)
    actions={'All report':np.ones(len(ty),bool),'common_B':act};devactions={'All report':np.ones(len(dp),bool),'common_B':da};picks={'common_B':pick}
    for rule in a.RULES:
        pp=r.gates.rc.select_threshold(ds[rule],r.gates.fr.ma_class(dp),r.gates.fr.ma_class(dy),.15);t=pp.get('threshold')
        actions[rule]=np.zeros(len(ty),bool) if t is None else np.isfinite(qs[rule])&(qs[rule]<=t)
        devactions[rule]=np.zeros(len(dp),bool) if t is None else np.isfinite(ds[rule])&(ds[rule]<=t);picks[rule]=pp
    t=float(np.quantile(ds['Boundary'][np.isfinite(ds['Boundary'])],.95));picks['B95']=dict(threshold=t)
    actions['B95']=qs['Boundary']<=t;devactions['B95']=ds['Boundary']<=t
    rows=[];pc=r.state(z['prediction']);tc=r.state(ty)
    for name,mask in actions.items():
        met=n.extended_metrics(z['prediction'],ty,mask);dev=n.extended_metrics(dp,dy,devactions[name]);incorrect=pc!=tc;FP=(tc==1)&(pc!=1);FN=(tc!=1)&(pc==1)
        rows.append(dict(rule=name,feasible=bool(pick['feasible']) if name=='common_B' else True,test=met,development=dev,
            disposition=dict(before_correct=int((~incorrect).sum()),before_incorrect=int(incorrect.sum()),correct_released=int((mask&~incorrect).sum()),correct_withheld=int((~mask&~incorrect).sum()),incorrect_released=int((mask&incorrect).sum()),incorrect_withheld=int((~mask&incorrect).sum()),FP_withheld=int((~mask&FP).sum()),FN_withheld=int((~mask&FN).sum()))))
    curves=[]
    for name in a.RULES:
        order=np.argsort(qs[name],kind='stable')
        for percent in range(5,101,5):
            k=int(np.floor(len(ty)*percent/100));mask=np.zeros(len(ty),bool);mask[order[:k]]=True
            curves.append(dict(rule=name,nominal_rate=percent/100,test=r.metrics(z['prediction'],ty,mask)))
    return rows,picks,actions,curves

def reporting_round(d,plan,role,item,budget,h,arms=ARMS,folder=None):
    rep=item['rep'];folder=folder or O/f'primary/dev{budget}/time{h}/round_{rep:02d}'
    if (folder/'COMPLETE.json').exists():return
    start=time.time();activate(item,plan,folder);sp=specification(item,budget,role);rows=[];curves=[];policies={};saved={}
    for arm in arms:
        if h==900 and arm in a.ARMS:
            source,rr,_=b.load(budget,rep,arm);z={k:source[k] for k in source.files};dr,qr=rr['dev_risk_B'],rr['test_risk_B']
        else:
            z=collect(d,sp,arm,item['seed'],h);dr,qr,audit=risks(d,sp,z['dev_rows'],h);r.write(folder/(arm+'_risk_training.json'),audit)
        out,picks,actions,cc=evaluate_rules(d,z,dr,qr,h)
        if h==900 and arm in a.ARMS:
            frozen=json.loads((LOCK/f'private/dev{budget}/round_{rep:02d}/POLICIES.json').read_text())[arm]
            assert r.safe(picks['common_B'])==r.safe(frozen)
            with np.load(LOCK/f'private/dev{budget}/round_{rep:02d}/{arm}.npz') as zz:np.testing.assert_array_equal(actions['common_B'],zz['test_action']);np.testing.assert_array_equal(z['prediction'],zz['test_prediction'])
        np.savez_compressed(folder/(arm+'.npz'),**z,dev_risk=dr,test_risk=qr,**{'action_'+k.replace(' ','_'):v for k,v in actions.items()})
        rows += [dict(v,budget=budget,rep=rep,arm=arm,time=h) for v in out];curves += [dict(v,budget=budget,rep=rep,arm=arm,time=h) for v in cc];policies[arm]=picks
    r.write(folder/'RESULTS.json',rows);r.write(folder/'CURVES.json',curves);r.write(folder/'POLICIES.json',policies);r.write(folder/'SPLIT.json',sp)
    r.write(folder/'COMPLETE.json',dict(seconds=time.time()-start,rep=rep,budget=budget,time=h));print('REPORTING',budget,h,rep+1,round(time.time()-start,1),flush=True)

def correction(d,models,model,train,query):
    """Same locked B-local asymmetric residual and endpoint safeguard for each source control."""
    train=np.asarray(train);query=np.asarray(query);x=d['x'][:,1:2].copy();shadow=dict(d,y=np.full_like(d['y'],np.nan));shadow['y'][train]=d['y'][train]
    z,q,y,_,_=r.c.selected.models.design(shadow,[x],train,query,'target_only');w=r.c.selected.models.weights(y,0,'inward4')
    pt=models.predict(model,d['x'][train])[:,2];pq=models.predict(model,d['x'][query])[:,2];base=(pt-60)/10
    coef,_=r.c.selected.solve_offset(z,(y-60)/10,w,base);raw=60+10*((pq-60)/10+q@coef)
    clipped=x.copy();lo=x[train,0].min();hi=x[train,0].max();clipped[query,0]=np.clip(clipped[query,0],lo,hi)
    zz,qc,*_=r.c.selected.models.design(shadow,[clipped],train,query,'target_only');np.testing.assert_allclose(z,zz,rtol=0,atol=1e-12)
    full=d['x'][query].copy();full[:,1]=clipped[query,0];boundary=models.predict(model,full)[:,2]+10*qc@coef
    below=x[query,0]<lo;above=x[query,0]>hi;pred=raw.copy();pred[below]=np.minimum(pred[below],boundary[below]);pred[above]=np.maximum(pred[above],boundary[above])
    return np.maximum(pred,0),dict(n_lower=int(below.sum()),n_upper=int(above.sum()),raw_prediction=raw)

def table1_round(d,plan,role,models,item):
    rep=item['rep'];dest=O/f'table1/round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    start=time.time();activate(item,plan,dest);result=[];audits=[]
    for arm in a.ARMS:
        srcrows=r.c.source_rows(arm,900);masked=np.full_like(d['medical'],np.nan);masked[srcrows]=d['medical'][srcrows];data=dict(d,x=d['x'],medical=masked,ci=np.full_like(d['ci'],np.nan));data['ci'][srcrows]=d['ci'][srcrows]
        fitted={}
        for method in r.mapping.METHODS:
            cache=dest/(arm+'_'+method+'_source.joblib')
            if cache.exists():model=joblib.load(cache)
            else:
                model=models.fit_source(data,srcrows,[],method,item['seed'],steps=500 if method in r.mapping.METHODS[-2:] else None);assert model['success'];joblib.dump(model,cache,compress=3)
            fitted[method]=model
            for budget in [54,60]:
                sp=specification(item,budget,role);tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);y=d['medical'][te]
                pt=models.predict(model,d['x'][tr]);pq=models.predict(model,d['x'][te]);hc=models.fit_cal(pt,d['medical'][tr],model['ym'],model['ys']);assert hc['finite'];pred=models.project(models.predict_cal(hc,pq));pred[:,2],extra=correction(d,models,model,tr,te)
                result.append(dict(rep=rep,budget=budget,arm=arm,panel='mapping',method=method,**r.four_metrics(pred,y)))
                np.savez_compressed(dest/f'{arm}_{method}_dev{budget}.npz',test=te,prediction=pred,MA_before_endpoint=extra['raw_prediction'])
                audits.append(dict(budget=budget,arm=arm,method=method,n_lower=extra['n_lower'],n_upper=extra['n_upper']))
        ridge=fitted['RIDGE-10'];src=r.c.source_model(arm,900)
        for budget in [54,60]:
            sp=specification(item,budget,role);tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);y=d['medical'][te]
            pt=models.predict(ridge,d['x'][tr]);pq=models.predict(ridge,d['x'][te]);pt[:,2]=r.c.source_predict(src,d['x'][tr,1:2]);pq[:,2]=r.c.source_predict(src,d['x'][te,1:2])
            scale=dict(ym=d['medical'][srcrows].mean(0),ys=np.maximum(d['medical'][srcrows].std(0),1e-6));hc=models.fit_cal(pt,d['medical'][tr],scale['ym'],scale['ys']);ha=models.fit_cal(pt,d['medical'][tr],ridge['ym'],ridge['ys']);assert hc['finite'] and ha['finite']
            final=models.project(models.predict_cal(ha,pq));z,_,_=b.load(budget,rep,arm);final[:,2]=z['prediction'];assert np.array_equal(te,z['test'])
            vals={'No adaptation':pq,'Independent affine':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'affine'),pq),'Joint linear':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'joint'),pq),'Joint Huber':models.project(models.predict_cal(hc,pq)),'Asymmetric correction':final}
            result.append(dict(rep=rep,budget=budget,arm=arm,panel='mapping',method='Hybrid regression',**r.four_metrics(final,y)))
            for kind,pred in vals.items():result.append(dict(rep=rep,budget=budget,arm=arm,panel='calibration',method=kind,**r.four_metrics(pred,y)))
            np.savez_compressed(dest/f'{arm}_calibration_dev{budget}.npz',test=te,**{k.replace(' ','_'):v for k,v in vals.items()})
    r.write(dest/'RESULTS.json',result);r.write(dest/'ENDPOINT_AUDIT.json',audits);r.write(dest/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start));print('TABLE1',rep+1,round(time.time()-start,1),flush=True)

def prepare():
    proto=dict(locked_policy_sha=r.sha(LOCK/'SELECTED_POLICY.json'),original_split_sha=r.sha(r.OUT/'private/SPLITS.json'),repeats=100,primary_budgets=[54,60],
        table1='Six original source-model definitions/steps and five calibration procedures. Identical source draws shared by54/60; identical current B-local asymmetric correction and directional endpoint safeguard across source controls, with other3 inputs held unchanged when B is clipped for a four-input source prior. Primary saved MA predictions copied exactly.',
        table2='All report,Boundary,Logistic,HGB,B95 on same primary prediction arrays; original developer15%error baseline criterion; common_B locked selector separately. No parameter search or test-dependent adjustment.',
        figure2='All-report MA metrics and confusion matrices; keep repeat counts separate from unique records. Target-only re-evaluated with same anchored model/loss and same common_B selector.',
        figure3='Raw-score rank curves at5%-100% nominal quotas, floor(n*q) with stable ordering; new composite gate shown only at its real workpoint, no invented scalar ranking. Full FN/FP/correct/incorrect dispositions.',
        time='Available300/600/900-point prefix features, same locked rule recipe; baseline classifier inputs use matching time. No future features in prefix classifiers. Earliest and capability analyses will be labelled distinctly.',
        scope='Do not change manuscript,figures,PDFs,publicZIP/GitHub or lockedpackage. Save all rounds including infeasible rules/undefinedFNR.',input_hashes=cal.protocol()['input_hashes'])
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(proto)
    else:r.write(path,proto)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['table1','reporting','smoke']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);ap.add_argument('--time',type=int,default=900);args=ap.parse_args();prepare()
    global GLOBAL_D
    d,plan,role,models,items=init();GLOBAL_D=d
    if args.stage=='smoke':
        item=items[0];activate(item,plan,O/'smoke');sp=specification(item,54,role)
        z=collect(d,sp,'HZ_to_SA',item['seed'],900);old,rr,_=b.load(54,0,'HZ_to_SA')
        for k in z:np.testing.assert_allclose(z[k],old[k],atol=1e-10,rtol=1e-10)
        dr,qr,_=risks(d,sp,z['dev_rows'],900);np.testing.assert_allclose(dr,rr['dev_risk_B'],atol=1e-10,rtol=0);np.testing.assert_allclose(qr,rr['test_risk_B'],atol=1e-10,rtol=0)
        ycopy=d['medical'].copy();d['medical'][sp['test']]=9999
        try:
            zz=collect(d,sp,'HZ_to_SA',item['seed'],900)
            for k in z:np.testing.assert_allclose(z[k],zz[k],atol=1e-10,rtol=1e-10)
        finally:d['medical'][:]=ycopy
        r.write(O/'SMOKE.json',dict(primary_predictions_and_gate_scores_reproduced=True,risk_scores_reproduced=True,test_label_poison_does_not_change_fitting=True));print('SMOKE PASS',flush=True)
    else:
        for item in items[args.start:args.stop]:
            if args.stage=='table1':table1_round(d,plan,role,models,item)
            else:
                for budget in [54,60]:reporting_round(d,plan,role,item,budget,args.time)
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in cal.protocol()['input_hashes'].items())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
