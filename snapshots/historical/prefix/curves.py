import runtime as r
import numpy as np,joblib,time,argparse
from scipy.optimize import least_squares
from scipy.special import expit
five=dict(np=np,time=time,least_squares=least_squares,expit=expit)
r.extract(r.ZERO/'refit.py',['shape','unpack','pack','fit_one'],five)
def shape(q,t,family):
    if family=='F5':return five['shape'](q,t)
    t0,B,k,nu=q if family=='F4' else [0,*q];z=np.maximum(t-t0,0);v=np.zeros(len(t));good=z>0;v[good]=B*expit(nu*(np.log(z[good])-np.log(k)));return v-np.median(v[:10])
def fit_simple(y,family):
    t=np.arange(1,len(y)+1.);scale=max(float(y.std()),1e-8);n=len(y);H=np.median(y[-30:]);hit=np.flatnonzero(y>=.05*H);on=float(t[hit[0]]) if len(hit) else 0.;maxon=min(899,n-2.)
    specs=[(0,120,2),(0,180,4),(0,360,2),(0,600,4),(0,900,5)] if family=='F3' else [(0,300,2),(.5*on,180,2),(on,360,4),(1.5*on,600,2),(0,900,5)]
    lo=np.array([0,0,0,np.log(.2)]) if family=='F4' else np.array([0,0,np.log(.2)]);hi=np.array([maxon,np.inf,np.log(900),np.log(12)]) if family=='F4' else np.array([np.inf,np.log(900),np.log(12)])
    def unpack(v):return np.array([v[0],v[1],np.exp(v[2]),np.exp(v[3])]) if family=='F4' else np.array([v[0],np.exp(v[1]),np.exp(v[2])])
    def pack(q):return np.array([q[0],q[1],np.log(q[2]),np.log(q[3])]) if family=='F4' else np.array([q[0],np.log(q[1]),np.log(q[2])])
    logs=[];best=np.inf;bestq=None;selected=None
    for si,(t0,k,nu) in enumerate(specs):
        t0=min(t0,maxon);q=np.array([t0,1.,k,nu] if family=='F4' else [1.,k,nu]);basis=shape(q,t,family);B=max(0.,float(basis@y/max(basis@basis,1e-20)));q[1 if family=='F4' else 0]=B;v=np.minimum(np.maximum(pack(q),lo+1e-10),hi-1e-10);q=unpack(v);sse=float(np.square(shape(q,t,family)-y).sum())
        if sse<best:best=sse;bestq=q;selected=-(si+1)
        calls=0;begin=time.time()
        def residual(v):
            nonlocal calls
            calls+=1
            if calls>2000:raise RuntimeError('residual cap')
            return (shape(unpack(v),t,family)-y)/scale
        try:
            sol=least_squares(residual,v,bounds=(lo,hi),method='trf',max_nfev=300,ftol=1e-8,xtol=1e-8,gtol=1e-8,x_scale='jac');q=unpack(sol.x);sse=float(np.square(shape(q,t,family)-y).sum());row=dict(start=si,success=bool(sol.success),status=int(sol.status),message=sol.message,nfev=int(sol.nfev),calls=calls,sse=sse,active_mask=sol.active_mask,seconds=time.time()-begin)
            if np.isfinite(sse) and sse<best:best=sse;bestq=q;selected=si
        except RuntimeError as e:row=dict(start=si,success=False,status='call_cap',calls=calls,seconds=time.time()-begin)
        logs.append(row)
    pred=shape(bestq,t,family);return dict(q=bestq,SSE=best,MSE=best/n,MAE=float(np.mean(abs(pred-y))),R2=1-best/max(float(np.square(y-y.mean()).sum()),1e-30),selected_start=selected,selected_converged=logs[selected]['success'] if selected>=0 else None,starts=logs,n=n)
def path_for(p,f,i):
    if i>=306 and f=='F5':return r.ZERO/f'curve_fits/{i:03d}.joblib'
    if i>=306:p='HZ'
    return r.ROOT/f'curve_fits/{p}_{f}_{i:03d}.joblib'
def jobs():return [(p,f,i) for p in r.PREPS for f in r.FAMILIES for i in range(306)]+[('FILTER',f,i) for f in ['F4','F3'] for i in range(306,401)]
def one(p,f,i,signals,base):
    path=path_for(p,f,i)
    if path.exists():return
    obs=np.isfinite(signals[p][i]);y=signals[p][i,obs];key=f'curve_{p}_{f}_{i:03d}';r.ledger('curve',key,index=i,preprocessing=p,family=f);begin=time.time();fit=five['fit_one'](y,base['BASE5'][i]) if f=='F5' else fit_simple(y,f);fit.update(index=i,family=f,preprocessing=p,signal_hash=r.hashlib.sha256(y.tobytes()).hexdigest());path.parent.mkdir(exist_ok=True);joblib.dump(fit,path,compress=3);r.finish(key,seconds=time.time()-begin,success=bool(fit['selected_converged']),optimizer_stages=5,residual_calls=sum(v['calls'] for v in fit['starts']),selected_start=fit['selected_start']);return fit
def assemble(signals,base):
    data=dict(base);original=np.load(r.ZERO/'private/input_archive.npz')['zeroed'];rows=[];files={}
    for p in r.PREPS:
        for f in r.FAMILIES:
            arm=p+'_'+f;fits=[]
            for i in range(401):
                path=path_for(p,f,i);z=joblib.load(path);files[str(path)]=r.sha(path);fits.append(z['q']);mask=np.isfinite(signals[p][i]);y=signals[p][i,mask];t=np.arange(1,len(y)+1.);pred=shape(z['q'],t,f);raw=original[i,mask];v=z['q'];bounds=dict(time_or_shape_upper=bool(np.any(np.isclose(v[-2:],[900,12],atol=1e-4,rtol=0))))
                if f=='F5':bounds['tau_upper']=bool(np.isclose(v[1],900,atol=1e-4,rtol=0))
                if f=='F4':bounds['onset_zero']=bool(v[0]<1e-4)
                rows.append(dict(arm=arm,index=i,R2=1-np.square(pred-y).sum()/max(np.square(y-y.mean()).sum(),1e-30),MAE=float(np.mean(abs(pred-y))),raw_MAE=float(np.mean(abs(pred-raw))),raw_R2=1-np.square(pred-raw).sum()/max(np.square(raw-raw.mean()).sum(),1e-30),selected_converged=z['selected_converged'],selected_start=z['selected_start'],bounds=bounds))
            data[arm]=np.array(fits);assert np.isfinite(data[arm]).all()
    np.savez_compressed(r.ROOT/'private/data.npz',**data);r.write('private/CURVE_METRICS.json',rows);summ={}
    for arm in r.ARMS[1:]:
        summ[arm]={}
        for center,low,high in [('SAHZU',0,306),('Huzhou',306,401)]:
            z=[v for v in rows if v['arm']==arm and low<=v['index']<high];summ[arm][center]=dict(n=len(z),R2_quantiles=np.quantile([v['R2'] for v in z],[0,.05,.5,.95,1]),MAE_quantiles=np.quantile([v['MAE'] for v in z],[0,.05,.5,.95,1]),raw_R2_quantiles=np.quantile([v['raw_R2'] for v in z],[0,.05,.5,.95,1]),below095=sum(v['R2']<.95 for v in z),selected_not_converged=sum(v['selected_converged'] is not True for v in z))
    r.write('CURVE_SUMMARY.json',summ);r.write('FITTED_INPUT_FREEZE.json',dict(time=time.time(),data_sha256=r.sha(r.ROOT/'private/data.npz'),files=files,target_evaluated=False));print('ASSEMBLED',len(files),'unique fits',flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);parser.add_argument('--pilot',action='store_true');parser.add_argument('--assemble',action='store_true');args=parser.parse_args();signals={p:np.load(r.PILOT/f'private/{n}.npz')['signal'] for p,n in r.FILES.items()};base=dict(np.load(r.ROOT/'private/base.npz'))
    if args.assemble:return assemble(signals,base)
    for j,(p,f,i) in enumerate(jobs()):
        if args.pilot and (p,f,i)!=('FILTER','F4',0):continue
        if not args.pilot and j%args.shards!=args.shard:continue
        z=one(p,f,i,signals,base)
        if z is not None and (j%100==0 or args.pilot):print('CURVE',j,p,f,i,z['R2'],flush=True)
        if j%300==0:r.guard()
    if args.pilot:r.write('CURVE_PILOT.json',dict(fit=r.read('ledger/curve_FILTER_F4_000.json'),resources=r.guard()))
if __name__=='__main__':main()
