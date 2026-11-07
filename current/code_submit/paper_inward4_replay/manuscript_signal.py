"""Matched EXP3(mu=1) and EXP4 refits of acquired U-TEG signals.

This is a representation control. It never changes the authoritative four
parameters or the primary fixed-B predictions.
"""
import argparse,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,reconstruction as rec
from .gates import clean_write

OUT=suite.ROOT/'manuscript_sync/experiments/signal'


def exp3(y):
    n=len(y);t=np.arange(1,n+1.);limit=min(899,n-2);scale=max(np.std(y),1e-6)
    on=np.flatnonzero(y>.05*max(float(np.median(y[-min(n,30):])),0))
    onset=float(on[0]) if len(on) else 0.
    best=None;logs=[]
    for j,(t0,k) in enumerate([(0,120),(.5*onset,180),(onset,360),(1.5*onset,600),(0,900)]):
        base=rec.shape([min(t0,limit),1,k,1],t);b=max(1e-6,float(base@y/max(base@base,1e-20)))
        x=np.array([min(t0,limit),b,np.log(k)]);lo=[0,0,0];hi=[limit,np.inf,np.log(900)]
        x=np.maximum(np.asarray(lo)+1e-9,np.minimum(x,np.asarray(hi)-1e-9));start=time.perf_counter()
        def unpack(v):return np.array([v[0],v[1],np.exp(v[2]),1.])
        sol=least_squares(lambda v:(rec.shape(unpack(v),t)-y)/scale,x,bounds=(lo,hi),method='trf',
            max_nfev=300,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac')
        q=unpack(sol.x);sse=float(np.sum((rec.shape(q,t)-y)**2))
        logs.append(dict(start=j,converged=bool(sol.success),nfev=int(sol.nfev),sse=sse,seconds=time.perf_counter()-start))
        if best is None or sse<best[0]:best=(sse,q,bool(sol.success))
    return dict(q=best[1],SSE=best[0],selected_converged=best[2],starts=logs)


def job(args):
    i,y=args
    with threadpool_limits(limits=1):
        amp=max(float(np.max(abs(y))),1e-10);v=y/amp;start=time.perf_counter()
        fit3=exp3(v);seconds3=time.perf_counter()-start;start=time.perf_counter()
        fit4=rec.fitter()(v,'F4');seconds4=time.perf_counter()-start
        result=dict(row=i,observed=len(y))
        for name,fit,seconds in [('EXP3',fit3,seconds3),('EXP4',fit4,seconds4)]:
            q=np.asarray(fit['q']).copy();q[1]*=amp;p=rec.shape(q,np.arange(1,len(y)+1.));e=p-y
            result[name]=dict(parameters=q,seconds=seconds,mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),
                nrmse=float(np.sqrt(np.mean(e**2))/max(np.ptp(y),1e-10)),r2=float(1-np.sum(e**2)/max(np.sum((y-y.mean())**2),1e-20)),
                converged=bool(fit['selected_converged']))
        return result


def run(smoke=False):
    suite.initialize();OUT.mkdir(parents=True,exist_ok=True);d,*_=c.context()
    with np.load(suite.ROOT/'data/CURRENT_SCALED_CURVES.npz') as z:
        curves=z['scaled_zeroed'].copy();mask=z['observed_mask'].copy()
    protocol=dict(input_sha256=c.sha(suite.ROOT/'data/CURRENT_SCALED_CURVES.npz'),
        model='EXP B*(1-exp(-((t-t0)+/k)^mu)), first10 median centered; EXP3 fixes mu=1; EXP4 refits4.',
        fitting='Observed samples at each checkpoint, five starts, amplitude normalized within the fitted signal; QC retained.',
        primary_unchanged=True,original_B_values_retained=True)
    p=OUT/'PROTOCOL.json'
    if p.exists():assert c.read(p)==protocol
    else:clean_write(p,protocol)
    if smoke:
        t=np.arange(1,901.);y=rec.shape([80,2.5,250,1],t)
        q=exp3(y)['q'];assert np.max(abs(rec.shape(q,t)-y))<1e-4
        q2=exp3(y*1.7)['q'];assert np.max(abs(rec.shape(q2,t)-1.7*y))<1e-4
        for i in [0,303,304,400,479]:
            a=job((i,curves[i,mask[i]]));assert all(np.isfinite(a[k]['parameters']).all() for k in ['EXP3','EXP4'])
        clean_write(OUT/'SMOKE.json',dict(synthetic_exact_fit=True,positive_scale_equivariance=True,five_current_records_finite=True))
        print('SIGNAL_SMOKE_PASS',flush=True);return
    assert (OUT/'SMOKE.json').exists()
    jobs=[(i,curves[i,mask[i]]) for i in range(len(curves)) if not (OUT/f'fit_{i}.json').exists()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for j,r in enumerate(pool.map(job,jobs,chunksize=2)):
            clean_write(OUT/f"fit_{r['row']}.json",r)
            if j%40==0:print('SIGNAL_REFIT',j+1,'of',len(jobs),flush=True)
    rows=[c.read(OUT/f'fit_{i}.json') for i in range(len(curves))]
    np.savez_compressed(OUT/'REFITS.npz',**{k:np.array([r[k]['parameters'] for r in rows]) for k in ['EXP3','EXP4']},
        qc=d['input_qc'],ids=d['ids'],observed=mask.sum(1),B_from_original_data=True)
    clean_write(OUT/'SUMMARY.json',{k:dict(records=int(d['input_qc'].sum()),
        median_nrmse=float(np.median([r[k]['nrmse'] for i,r in enumerate(rows) if d['input_qc'][i]])),
        converged=sum(r[k]['converged'] for r in rows),median_seconds=float(np.median([r[k]['seconds'] for r in rows]))) for k in ['EXP3','EXP4']})
    print('SIGNAL_REFITS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.smoke)
