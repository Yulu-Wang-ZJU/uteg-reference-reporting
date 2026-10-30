import runtime as r
import numpy as np
from scipy.optimize import minimize
import time,json

FLOOR=1e-6
COSTS=(.05,.1,.2,.3)
def project(m):
    a=np.asarray(m,dtype=float).copy();a[..., :3]=np.maximum(a[..., :3],0);a[...,3]=np.clip(a[...,3],0,89.999999)
    a[~np.isfinite(a).all(axis=-1)]=np.nan
    return a
def ci(m):return np.asarray(m)@np.array([-.6516,-.3772,.1224,.0759])-7.7922
def labels(j):
    j=np.asarray(j);return np.where(~np.isfinite(j),9,np.where(j<-3,-1,np.where(j>3,1,0))).astype(int)
def features(x,m):
    j=ci(project(m));l=labels(j)
    return np.column_stack([x,j/3,np.minimum(abs(j+3),abs(j-3))/3,*[(l==v).astype(float) for v in (-1,0,1)]])
def load_data():
    a=dict(np.load(r.DATA,allow_pickle=False));ref=dict(np.load(r.MED/'private/reference_medical.npz',allow_pickle=False))
    a.update(medical=ref['medical'],ci=ref['ci']);return a
def fit_cal(raw_med,reference_med,mu,scale):
    started=time.time();mu=np.asarray(mu,float);scale=np.maximum(scale,FLOOR)
    v=(project(raw_med)-mu)/scale;u=(np.asarray(reference_med)-mu)/scale
    vm=v.mean(0);um=u.mean(0);vc=v-vm;uc=u-um
    init=np.r_[np.linalg.solve(vc.T@vc+10*np.eye(4),vc.T@uc+10*np.eye(4)).ravel(),np.zeros(4)]
    def objective(th):
        M=th[:16].reshape(4,4);b=th[16:];e=vc@M+b-uc;ae=abs(e);delta=1.345
        value=np.where(ae<=delta,e*e,2*delta*ae-delta**2).sum()+10*np.sum((M-np.eye(4))**2)
        psi=2*np.clip(e,-delta,delta)
        grad=np.r_[(vc.T@psi+20*(M-np.eye(4))).ravel(),psi.sum(0)]
        return float(value),grad
    sol=minimize(objective,init,jac=True,method='L-BFGS-B',options=dict(maxiter=200,ftol=1e-12,gtol=1e-8))
    return dict(matrix=sol.x[:16].reshape(4,4),intercept=sol.x[16:],vbar=vm,ubar=um,mu=mu,scale=scale,
                success=bool(sol.success),finite=bool(np.isfinite(sol.x).all()),status=str(sol.message),nit=int(sol.nit),nfev=int(sol.nfev),
                objective=float(sol.fun),initial_objective=objective(init)[0],seconds=time.time()-started,params=20,n=len(v))
def predict_cal(model,raw_med):
    v=(project(raw_med)-model['mu'])/model['scale']
    return model['mu']+model['scale']*(model['ubar']+(v-model['vbar'])@model['matrix']+model['intercept'])
class Ledger:
    def __init__(self,name):
        self.path=r.ROOT/'ledgers'/f'{name}.json';self.d=r.read(self.path) if self.path.exists() else dict(fits=[],requests=0)
    def fit(self,kind,actual=1,**kw):
        self.d['fits'].append(dict(kind=kind,actual=int(actual),**r.clean(kw)))
        if len(self.d['fits'])%25==0:self.flush()
    def flush(self):
        r.write(self.path,self.d);return ledger_total()
def ledger_total():
    files=list((r.ROOT/'ledgers').glob('*.json'));rows=[r.read(p) for p in files]
    n=sum(len(d.get('fits',[])) for d in rows);actual=sum(sum(f.get('actual',1) for f in d.get('fits',[])) for d in rows)
    calls=sum(d.get('requests',0) for d in rows)
    if n>6640 or actual>7480 or calls>1500000:raise RuntimeError(f'Hard budget exceeded {n}/{actual}/{calls}')
    return dict(positions=n,actual=actual,requests=calls,files=len(files))
