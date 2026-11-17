"""Matched source-family and adaptation controls for the separate internal repeats."""
import experiment as e
import numpy as np
import json,joblib,time
from threadpoolctl import threadpool_limits
r=e.r;O=e.O

def one(d,plan,role,models,item):
    rep=item['rep'];dest=O/f'internal_controls/round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    e.activate(item,plan,dest);base=O/f'internal301/round_{rep:02d}';splits=json.loads((base/'SPLITS.json').read_text());local=np.sort(np.concatenate([sp['test'] for sp in splits]));where={int(v):i for i,v in enumerate(local)};truth=d['medical'][local];rows=[];start=time.time()
    for arm in e.a.ARMS:
        src=r.c.source_model(arm,900);srcrows=r.c.source_rows(arm,900);scale=dict(ym=d['medical'][srcrows].mean(0),ys=np.maximum(d['medical'][srcrows].std(0),1e-6));bank={}
        for method in r.mapping.METHODS:
            model=joblib.load(O/f'table1/round_{rep:02d}/{arm}_{method}_source.joblib');allp=np.full((len(local),4),np.nan)
            for sp in splits:
                tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);pt=models.predict(model,d['x'][tr]);pq=models.predict(model,d['x'][te]);hc=models.fit_cal(pt,d['medical'][tr],model['ym'],model['ys']);assert hc['finite'];pred=models.project(models.predict_cal(hc,pq));pred[:,2],_=e.correction(d,models,model,tr,te);allp[[where[int(v)] for v in te]]=pred
            rows.append(dict(rep=rep,arm=arm,panel='mapping',method=method,**r.four_metrics(allp,truth),MA_state=r.metrics(allp[:,2],truth[:,2],np.ones(len(local),bool))));bank[method]=allp
        ridge=joblib.load(O/f'table1/round_{rep:02d}/{arm}_RIDGE-10_source.joblib');calbank={name:np.full((len(local),4),np.nan) for name in ['No adaptation','Independent affine','Joint linear','Joint Huber','Asymmetric correction']}
        for fi,sp in enumerate(splits):
            tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);pt=models.predict(ridge,d['x'][tr]);pq=models.predict(ridge,d['x'][te]);pt[:,2]=r.c.source_predict(src,d['x'][tr,1:2]);pq[:,2]=r.c.source_predict(src,d['x'][te,1:2]);hc=models.fit_cal(pt,d['medical'][tr],scale['ym'],scale['ys']);ha=models.fit_cal(pt,d['medical'][tr],ridge['ym'],ridge['ys']);assert hc['finite'] and ha['finite'];final=models.project(models.predict_cal(ha,pq))
            with np.load(base/f'fold{fi}_{arm}.npz') as z:np.testing.assert_array_equal(z['test'],te);final[:,2]=z['prediction']
            vals={'No adaptation':pq,'Independent affine':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'affine'),pq),'Joint linear':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'joint'),pq),'Joint Huber':models.project(models.predict_cal(hc,pq)),'Asymmetric correction':final}
            for k,v in vals.items():calbank[k][[where[int(i)] for i in te]]=v
        for name,p in calbank.items():rows.append(dict(rep=rep,arm=arm,panel='calibration',method=name,**r.four_metrics(p,truth),MA_state=r.metrics(p[:,2],truth[:,2],np.ones(len(local),bool))))
        bank.update({k.replace(' ','_'):v for k,v in calbank.items()});np.savez_compressed(dest/(arm+'.npz'),test=local,**bank)
    r.write(dest/'RESULTS.json',rows);r.write(dest/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start));print('INTERNAL CONTROLS',rep+1,round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        d,plan,role,models,items=e.init()
        for item in items:one(d,plan,role,models,item)
