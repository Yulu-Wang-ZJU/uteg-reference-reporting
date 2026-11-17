"""Four-readout companion errors for the saved EXP3/EXP4 sensitivity."""
import experiment as e
import aggregate as ag
import numpy as np
from threadpoolctl import threadpool_limits

def main():
    d,plan,role,models,items=e.init();rows=[];r=e.r
    with np.load(r.current.ROOT/'manuscript_sync/experiments/signal/REFITS.npz',allow_pickle=True) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']);xx={name:z[name].copy() for name in ['EXP3','EXP4']}
    for item in items:
        dest=e.O/f'representation_mapping/round_{item["rep"]:02d}';e.activate(item,plan,dest);rr=[]
        for rep in ['EXP3','EXP4']:
            # EXP3 has a constant fourth coefficient (mu=1); after train-side
            # centring it is zero, equivalent to a three-variable Ridge fit.
            x=xx[rep]
            if rep=='EXP3':np.testing.assert_allclose(x[:,3],1.,atol=1e-12,rtol=0)
            for arm in e.ARMS:
                source=r.c.source_rows(arm,900);fitted=None
                if len(source):
                    masked=np.full_like(d['medical'],np.nan);masked[source]=d['medical'][source];data=dict(d,x=x,medical=masked);fitted=models.fit_source(data,source,[],'RIDGE-10',item['seed'])
                for budget in [54,60]:
                    sp=e.specification(item,budget,role);tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);model=fitted
                    if model is None:
                        masked=np.full_like(d['medical'],np.nan);masked[tr]=d['medical'][tr];model=models.fit_source(dict(d,x=x,medical=masked),tr,[],'RIDGE-10',item['seed'])
                    pq=models.predict(model,x[te])
                    if len(source):
                        pt=models.predict(model,x[tr]);cal=models.fit_cal(pt,d['medical'][tr],model['ym'],model['ys']);assert cal['finite'];pq=models.project(models.predict_cal(cal,pq))
                    with np.load(e.O/f'{rep}/round_{item["rep"]:02d}/dev{budget}_{arm}.npz') as z:np.testing.assert_array_equal(z['test'],te);pq[:,2]=z['prediction']
                    rr.append(dict(rep=item['rep'],representation=rep,budget=budget,arm=arm,**r.four_metrics(pq,d['medical'][te])));np.savez_compressed(dest/f'{rep}_dev{budget}_{arm}.npz',test=te,prediction=pq)
        r.write(dest/'RESULTS.json',rr);r.write(dest/'COMPLETE.json',dict(rep=item['rep']));rows+=rr
    ag.write('representation_four_readouts',rows,['representation','budget','arm']);print('REPRESENTATION MAPPING COMPLETE',len(rows))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
