import runtime as r
import numpy as np,torch,joblib,time,math
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CONFIGS={'TT-SMALL':(12,2,1,32)};ARMS=['BASE','CI-VALUE','CI-CLIN'];NEW=ARMS[1:];FLOOR=1e-6
# Import definitions only; never execute a legacy module's runtime/cache setup.
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.PREV/'core.py',['TT','newnet','digest','predict','calibrated','metrics','dump'],globals())
def weights(J):
    known=np.isfinite(J);y=labels(J);w=np.ones(len(J));counts={str(k):int(sum(y==k)) for k in [-1,0,1,9]}
    if known.any() and len(np.unique(y[known]))>1:
        rho=np.array([(counts[str(k)]+1)**(-.5) for k in y[known]]);lo=0.;hi=1.
        while np.mean(np.clip(hi*rho,.5,3))<1:hi*=2
        for _ in range(100):
            a=(lo+hi)/2
            if np.mean(np.clip(a*rho,.5,3))<1:lo=a
            else:hi=a
        w[known]=np.clip((lo+hi)/2*rho,.5,3)
    return w,counts
def loss_parts(pred,target,ym,ys,J,w):
    med=torch.cat([torch.clamp((ym+ys*pred)[:,:3],min=0),torch.clamp((ym+ys*pred)[:,3:],min=0,max=89.999999)],1)
    ci_hat=med@torch.tensor([-.6516,-.3772,.1224,.0759])-7.7922;valid=torch.isfinite(J)
    lm=torch.mean((pred-target)**2);lv=lm*0;lc=lm*0
    if valid.any():
        err=(ci_hat[valid]-J[valid])/3;lv=torch.where(abs(err)<=1,.5*err**2,abs(err)-.5).mean();lo=(ci_hat[valid]+3)/.5;hi=(ci_hat[valid]-3)/.5
        lp=torch.stack([F.logsigmoid(-lo),F.logsigmoid(lo)+F.logsigmoid(-hi)+math.log(-math.expm1(-12)),F.logsigmoid(hi)],1)
        lab=torch.where(J[valid]<-3,0,torch.where(J[valid]>3,2,1));lc=-(lp[torch.arange(len(lab)),lab]*w[valid]).mean()
    return lm,lv,lc,med,ci_hat
def fit(data,T,V,arm,seed,steps=None):
    began=time.time();T=np.asarray(T,int);V=np.asarray(V,int);X=data['x'][T];Y=data['medical'][T];J=data['ci'][T];xm=X.mean(0);xs=np.maximum(X.std(0),1e-6);ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6);w,counts=weights(J)
    net=newnet('TT-SMALL',seed);init={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};xt=torch.tensor((X-xm)/xs);yt=torch.tensor((Y-ym)/ys);ymt=torch.tensor(ym);yst=torch.tensor(ys);jt=torch.tensor(J);wt=torch.tensor(w)
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001);best=math.inf;beststep=steps or 500;chosen=None;stale=0;trace=[];finite=True
    for step in range(1,(steps or 500)+1):
        net.train();opt.zero_grad();pred=net(xt);lm,lv,lc,med,jh=loss_parts(pred,yt,ymt,yst,jt,wt);loss=lm if arm=='BASE' else lm+lv+(.25*lc if arm=='CI-CLIN' else 0)
        if not torch.isfinite(loss):finite=False;break
        loss.backward();grad=float(nn.utils.clip_grad_norm_(net.parameters(),5));opt.step()
        if step%10==0:
            row=dict(step=step,loss=float(loss.detach()),MED4_loss=float(lm.detach()),CI_value_loss=float(lv.detach()),class_loss=float(lc.detach()),gradient=grad,projection_fraction=float(torch.any(abs(med-(ymt+yst*pred))>1e-12,1).double().mean().detach()))
            if len(V) and step>=50:
                net.eval()
                with torch.no_grad():yp=project(ym+ys*net(torch.tensor((data['x'][V]-xm)/xs)).numpy())
                error=float(np.mean(abs(yp-data['medical'][V])/ys));row['validation_MED4_MAE']=error;row['validation_clinical']=metrics(yp,data['medical'][V],data['ci'][V])
                if np.isfinite(error) and error<best-1e-10:best=error;beststep=step;chosen={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
                else:stale+=1
            trace.append(row)
            if len(V) and step>=50 and stale>=5:break
    if len(V) and chosen is not None:net.load_state_dict(chosen)
    if len(V) and chosen is None:finite=False
    state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    return dict(name='TT-SMALL',arm=arm,train=T,validation=V,xm=xm,xs=xs,ym=ym,ys=ys,seed=seed,state=state,params=1680,active=1608,steps=beststep,actual_steps=step,seconds=time.time()-began,success=finite,trace=trace,initial_hash=digest(init),final_hash=digest(state),CI_weights=w,CI_support=counts,weight_neff=float(w.sum()**2/(w@w)))
