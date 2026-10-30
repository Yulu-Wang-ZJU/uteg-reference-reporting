import runtime as r
import numpy as np,torch,time,joblib
from scipy.special import expit,softmax
from scipy.optimize import minimize
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CONFIGS=['SOURCE-CI-ORDINAL','HUBER-CI-ORDINAL']
def project(x):
    x=np.asarray(x,float).copy();x[...,:3]=np.maximum(x[...,:3],0);x[...,3]=np.clip(x[...,3],0,89.999999)
    x[~np.isfinite(x).all(-1)]=np.nan;return x
r.extract(r.GATE/'independent_statistics.py',['areas'],globals())
r.extract(r.TH/'core.py',['ci','labels','actions','threshold','select','ratio','risk_at','metrics','bootstrap'],globals())
r.extract(r.ZERO/'verify_independent.py',['source'],globals());source_tt=source
r.extract(r.MAP/'verify.py',['source'],globals())
def logp(theta,z):
    eta=theta[0]*z;a=theta[1];gap=torch.nn.functional.softplus(theta[2])+1e-6;b=a+gap;ls=torch.nn.functional.logsigmoid
    return torch.stack((ls(a-eta),ls(b-eta)+ls(eta-a)+torch.log(-torch.expm1(-gap)),ls(eta-b)),dim=1)
def objective(theta,z,y):
    t=torch.tensor(theta,requires_grad=True);L=-logp(t,z)[torch.arange(len(y)),y].mean()+.1*t[0]**2
    L.backward();return float(L.detach()),t.grad.detach().numpy().copy()
def fit(name,j,y):
    start=time.time();j=np.asarray(j,float);y=np.asarray(y,int);finite=np.isfinite(j)&np.isin(y,[-1,0,1]);j=j[finite];y=y[finite]
    counts=np.array([(y==k).sum() for k in [-1,0,1]]);prior=(counts+.5)/(len(y)+1.5)
    mean=float(j.mean()) if len(j) else 0.;sd=max(float(j.std()),1e-6) if len(j) else 1.
    m=dict(name=name,mean=mean,sd=sd,n=len(y),class_counts=counts,class_prior=prior,params=3,success=False,finite=True,
      fallback=None,actual_optimization=False,iterations=0,nfev=0,seconds=0)
    if len(y)<8 or (counts>=2).sum()<2:m.update(fallback='insufficient_clinical_support',status='prior',effective_params=0)
    else:
        cum=np.cumsum(prior)[:2];cuts=np.log(cum/(1-cum));gap=max(cuts[1]-cuts[0]-1e-6,1e-8)
        initial=np.array([0.,cuts[0],np.log(np.expm1(gap))]);z=torch.tensor(np.clip((j-mean)/sd,-10,10));yi=torch.tensor(y+1,dtype=torch.long)
        out=minimize(objective,initial,args=(z,yi),jac=True,method='L-BFGS-B',bounds=[(0,None),(None,None),(None,None)],options=dict(maxiter=200,ftol=1e-9,gtol=1e-9))
        good=bool(np.isfinite(out.x).all() and np.isfinite(out.fun));m.update(theta=out.x,initial=initial,objective=float(out.fun),
          success=bool(out.success),finite=good,status=str(out.message),actual_optimization=True,iterations=int(out.nit),nfev=int(out.nfev),effective_params=3)
        if not good:m.update(fallback='nonfinite_optimizer',effective_params=0)
    m['seconds']=time.time()-start;return m
def probabilities(m,j):
    if m.get('fallback'):return np.tile(m['class_prior'],(len(j),1))
    z=torch.tensor(np.clip((np.asarray(j)-m['mean'])/m['sd'],-10,10));return logp(torch.tensor(m['theta']),z).exp().numpy()
def score(m,j,yh):
    ok=np.isfinite(j)&np.isin(yh,[-1,0,1]);out=np.full(len(j),np.inf)
    if ok.any():p=probabilities(m,np.asarray(j)[ok]);out[ok]=1-p[np.arange(ok.sum()),np.asarray(yh)[ok]+1]
    return out
