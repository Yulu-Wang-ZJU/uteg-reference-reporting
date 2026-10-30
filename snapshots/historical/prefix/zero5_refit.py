import runtime as r
import numpy as np,joblib,argparse,time
from scipy.optimize import least_squares
from scipy.special import expit

def shape(q,t):
    h=q[0]*np.exp(-t/q[1])+q[2]*expit(q[4]*(np.log(t)-np.log(q[3])));return h-np.median(h[:10])
def unpack(v):return np.array([v[0],np.exp(v[2]),v[1],np.exp(v[3]),np.exp(v[4])])
def pack(q):return np.array([q[0],q[2],np.log(q[1]),np.log(q[3]),np.log(q[4])])
def fit_one(y,q0):
    t=np.arange(1,len(y)+1.);scale=max(float(np.std(y)),1e-8);lo=np.array([-np.inf,-np.inf,0,0,np.log(.2)]);hi=np.array([np.inf,np.inf,np.log(900),np.log(900),np.log(12)])
    starts=[pack(q0)]
    for tau,t50,nu in [(30,180,2),(60,360,4),(180,600,2),(480,900,5)]:
        a=np.column_stack([np.exp(-t/tau),expit(nu*(np.log(t)-np.log(t50)))]);a-=a[:10].mean(0);coef=np.linalg.lstsq(a,y,rcond=None)[0];starts.append(pack(np.array([coef[0],tau,coef[1],t50,nu])))
    init=np.minimum(np.maximum(starts[0],lo),hi);bestq=unpack(init);best=float(np.sum((shape(bestq,t)-y)**2));selected=-1;logs=[]
    for k,v in enumerate(starts):
        v=np.minimum(np.maximum(v,lo+1e-10),hi-1e-10);calls=0;begin=time.time()
        def residual(a):
            nonlocal calls
            calls+=1
            if calls>2000:raise RuntimeError('residual cap')
            return (shape(unpack(a),t)-y)/scale
        try:
            sol=least_squares(residual,v,bounds=(lo,hi),method='trf',max_nfev=300,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac');q=unpack(sol.x);sse=float(np.sum((shape(q,t)-y)**2));row=dict(start=k,success=bool(sol.success),status=int(sol.status),message=sol.message,nfev=int(sol.nfev),njev=int(sol.njev or 0),calls=calls,sse=sse,optimality=float(sol.optimality),active_mask=sol.active_mask,seconds=time.time()-begin)
            if np.isfinite(sse) and sse<best:best=sse;bestq=q;selected=k
        except RuntimeError as e:row=dict(start=k,success=False,status='call_cap',message=str(e),calls=calls,seconds=time.time()-begin)
        logs.append(row)
    pred=shape(bestq,t);h=bestq[0]*np.exp(-t/bestq[1])+bestq[2]*expit(bestq[4]*(np.log(t)-np.log(bestq[3])));return dict(q=bestq,G0_derived=-np.median(h[:10]),SSE=best,MSE=best/len(y),MAE=float(np.mean(abs(pred-y))),R2=1-best/max(float(np.sum((y-y.mean())**2)),1e-30),selected_start=selected,selected_converged=logs[selected]['success'] if selected>=0 else None,boundaries=np.isclose(bestq[[1,3,4]],[900,900,12],rtol=0,atol=1e-4),starts=logs,n=len(y),initial_SSE=float(np.sum((shape(unpack(init),t)-y)**2)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--pilot',action='store_true');p.add_argument('--assemble',action='store_true');a=p.parse_args();data=dict(np.load(r.ROOT/'private/input_archive.npz'))
    if a.assemble:
        files=[r.ROOT/f'curve_fits/{i:03d}.joblib' for i in range(401)];assert all(p.exists() for p in files);fits=[joblib.load(p) for p in files];q=np.array([f['q'] for f in fits]);assert np.isfinite(q).all();base=dict(np.load(r.PREV/'private/data.npz'));np.savez_compressed(r.ROOT/'private/data.npz',x=base['x'],x_zero6=data['x_zero6'],x_zero5=q,medical=base['medical'],ci=base['ci'],ids=base['ids'],groups=base['groups']);r.write('CURVE_FIT_SUMMARY.json',dict(records=401,selected_initial=sum(f['selected_start']==-1 for f in fits),selected_not_converged=sum(f['selected_start']>=0 and not f['selected_converged'] for f in fits),optimizer_calls=sum(len(f['starts']) for f in fits),nfev=sum(z.get('nfev',0) for f in fits for z in f['starts']),residual_calls=sum(z['calls'] for f in fits for z in f['starts']),R2_quantiles=np.quantile([f['R2'] for f in fits],[0,.05,.5,.95,1]),R2_below_095=sum(f['R2']<.95 for f in fits),by_center={name:dict(n=len(ix),R2_quantiles=np.quantile([fits[i]['R2'] for i in ix],[0,.05,.5,.95,1]),R2_below_095=sum(fits[i]['R2']<.95 for i in ix),boundary_counts=np.sum([fits[i]['boundaries'] for i in ix],0)) for name,ix in [('SAHZU',range(306)),('Huzhou',range(306,401))]}));r.write('FITTED_INPUT_FREEZE.json',dict(time=time.time(),data_sha256=r.sha(r.ROOT/'private/data.npz'),fit_hashes={p.name:r.sha(p) for p in files},target_evaluated=False));print('ASSEMBLED',flush=True);return
    for i in ([0] if a.pilot else range(a.shard,401,a.shards)):
        path=r.ROOT/f'curve_fits/{i:03d}.joblib'
        if path.exists():continue
        mask=np.isfinite(data['zeroed'][i]);assert np.array_equal(np.where(mask)[0],np.arange(sum(mask)));y=data['zeroed'][i,mask];key=f'curve{i:03d}';r.ledger('curve_record',key,index=i);begin=time.time();fit=fit_one(y,data['x_zero6'][i,1:]);path.parent.mkdir(exist_ok=True);joblib.dump(fit,path,compress=3);r.finish(key,seconds=time.time()-begin,success=True,selected_converged=fit['selected_converged'],optimizer_stages=5,residual_calls=sum(z['calls'] for z in fit['starts']));
        if i%25==0 or a.pilot:print('CURVE',i,fit['R2'],time.time()-begin,flush=True);r.guard()

if __name__=='__main__':main()
