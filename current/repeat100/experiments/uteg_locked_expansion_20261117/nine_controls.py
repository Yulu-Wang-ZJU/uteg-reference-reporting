"""Complete supplementary six-model/five-calibration controls, same100 draws."""
import experiment as e
import additional as ad
import numpy as np
import json,time,argparse,joblib
from threadpoolctl import threadpool_limits
r=e.r;O=e.O

def run(d,plan,role,models,item):
    rep=item['rep'];dest=O/f'nine_controls/round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    e.activate(item,plan,dest);start=time.time();rows=[];fitted={}
    for arm in e.a.ARMS:
        signature=tuple(r.c.source_rows(arm,900))
        for method in r.mapping.METHODS:fitted[(signature,method)]=joblib.load(O/f'table1/round_{rep:02d}/{arm}_{method}_source.joblib')
    for arm,target in r.c.TARGET.items():
        if arm.startswith('target_only') or arm in e.a.ARMS:continue
        srcrows=r.c.source_rows(arm,900);sig=tuple(srcrows);masked=np.full_like(d['medical'],np.nan);masked[srcrows]=d['medical'][srcrows];data=dict(d,x=d['x'],medical=masked,ci=np.full_like(d['ci'],np.nan));data['ci'][srcrows]=d['ci'][srcrows]
        for method in r.mapping.METHODS:
            key=(sig,method)
            if key not in fitted:
                path=dest/f'{arm}_{method}_source.joblib'
                if path.exists():fitted[key]=joblib.load(path)
                else:
                    model=models.fit_source(data,srcrows,[],method,item['seed'],steps=500 if method in r.mapping.METHODS[-2:] else None);assert model['success'];fitted[key]=model;joblib.dump(model,path,compress=3)
            model=fitted[key]
            for mode in ['sampled90','full_local']:
                sp=ad.spec_for(item,role,target,mode);tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);pt=models.predict(model,d['x'][tr]);pq=models.predict(model,d['x'][te]);hc=models.fit_cal(pt,d['medical'][tr],model['ym'],model['ys']);assert hc['finite'];pred=models.project(models.predict_cal(hc,pq));pred[:,2],extra=e.correction(d,models,model,tr,te)
                rows.append(dict(rep=rep,mode=mode,arm=arm,panel='mapping',method=method,**r.four_metrics(pred,d['medical'][te]),MA_state=r.metrics(pred[:,2],d['medical'][te,2],np.ones(len(te),bool))))
                np.savez_compressed(dest/f'{mode}_{arm}_{method}.npz',test=te,prediction=pred)
        ridge=fitted[(sig,'RIDGE-10')];src=r.c.source_model(arm,900)
        for mode in ['sampled90','full_local']:
            sp=ad.spec_for(item,role,target,mode);tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);y=d['medical'][te];pt=models.predict(ridge,d['x'][tr]);pq=models.predict(ridge,d['x'][te]);pt[:,2]=r.c.source_predict(src,d['x'][tr,1:2]);pq[:,2]=r.c.source_predict(src,d['x'][te,1:2]);scale=dict(ym=d['medical'][srcrows].mean(0),ys=np.maximum(d['medical'][srcrows].std(0),1e-6));hc=models.fit_cal(pt,d['medical'][tr],scale['ym'],scale['ys']);ha=models.fit_cal(pt,d['medical'][tr],ridge['ym'],ridge['ys']);assert hc['finite'] and ha['finite'];final=models.project(models.predict_cal(ha,pq))
            with np.load(O/f'nine/round_{rep:02d}/{mode}_{arm}.npz') as z:assert np.array_equal(z['test'],te);final[:,2]=z['prediction']
            vals={'No adaptation':pq,'Independent affine':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'affine'),pq),'Joint linear':r.pred_cal(r.fit_cal(pt,d['medical'][tr],scale,'joint'),pq),'Joint Huber':models.project(models.predict_cal(hc,pq)),'Asymmetric correction':final}
            rows.append(dict(rep=rep,mode=mode,arm=arm,panel='mapping',method='Hybrid regression',**r.four_metrics(final,y),MA_state=r.metrics(final[:,2],y[:,2],np.ones(len(te),bool))))
            for kind,pred in vals.items():rows.append(dict(rep=rep,mode=mode,arm=arm,panel='calibration',method=kind,**r.four_metrics(pred,y),MA_state=r.metrics(pred[:,2],y[:,2],np.ones(len(te),bool))))
            np.savez_compressed(dest/f'{mode}_{arm}_calibration.npz',test=te,**{k.replace(' ','_'):v for k,v in vals.items()})
    r.write(dest/'RESULTS.json',rows);r.write(dest/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start));print('NINE CONTROLS',rep+1,round(time.time()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args();d,plan,role,models,items=e.init();e.GLOBAL_D=d
    for item in items[args.start:args.stop]:run(d,plan,role,models,item)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
