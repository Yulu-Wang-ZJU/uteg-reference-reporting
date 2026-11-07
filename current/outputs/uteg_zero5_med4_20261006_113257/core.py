import runtime as r
import numpy as np,torch,joblib,time,math
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CONFIGS={'TT-SMALL':(12,2,1,32)};ARMS=['RAW6','ZERO6','ZERO5'];NEW=ARMS[1:];FLOOR=1e-6
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.CAP/'core.py',['TT','digest','calibrated','metrics','dump'],globals())
r.extract(r.PREV/'core.py',['weights','loss_parts'],globals())
def inputs(data,arm):return data[{'RAW6':'x','ZERO6':'x_zero6','ZERO5':'x_zero5'}[arm]]
def newnet(seed,dim=6):
    torch.manual_seed(seed);net=TT(CONFIGS['TT-SMALL'],seed)
    if dim==5:net.val=nn.Parameter(net.val[1:].detach().clone());net.feat=nn.Parameter(net.feat[1:].detach().clone())
    assert dim in [5,6];return net
def fit(data,T,V,arm,seed,steps=None):
    began=time.time();T=np.asarray(T,int);V=np.asarray(V,int);allX=inputs(data,arm);X=allX[T];Y=data['medical'][T];J=data['ci'][T];xm=X.mean(0);xs=np.maximum(X.std(0),1e-6);ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6);w,counts=weights(J)
    net=newnet(seed,X.shape[1]);init={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};xt=torch.tensor((X-xm)/xs);yt=torch.tensor((Y-ym)/ys);ymt=torch.tensor(ym);yst=torch.tensor(ys);jt=torch.tensor(J);wt=torch.tensor(w)
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001);best=math.inf;beststep=steps or 500;chosen=None;stale=0;trace=[];finite=True
    for step in range(1,(steps or 500)+1):
        net.train();opt.zero_grad();pred=net(xt);lm,lv,lc,med,jh=loss_parts(pred,yt,ymt,yst,jt,wt);loss=lm+lv+.25*lc
        if not torch.isfinite(loss):finite=False;break
        loss.backward();grad=float(nn.utils.clip_grad_norm_(net.parameters(),5));opt.step()
        if step%10==0:
            row=dict(step=step,loss=float(loss.detach()),MED4_loss=float(lm.detach()),CI_value_loss=float(lv.detach()),class_loss=float(lc.detach()),gradient=grad)
            if len(V) and step>=50:
                net.eval()
                with torch.no_grad():yp=project(ym+ys*net(torch.tensor((allX[V]-xm)/xs)).numpy())
                error=float(np.mean(abs(yp-data['medical'][V])/ys));row['validation_MED4_MAE']=error;row['validation_clinical']=metrics(yp,data['medical'][V],data['ci'][V])
                if np.isfinite(error) and error<best-1e-10:best=error;beststep=step;chosen={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
                else:stale+=1
            trace.append(row)
            if len(V) and step>=50 and stale>=5:break
    if len(V) and chosen is not None:net.load_state_dict(chosen)
    if len(V) and chosen is None:finite=False
    state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};params=sum(p.numel() for p in net.parameters());return dict(name='TT-SMALL',arm=arm,dim=X.shape[1],train=T,validation=V,xm=xm,xs=xs,ym=ym,ys=ys,seed=seed,state=state,params=params,active=params-net.feat.numel(),steps=beststep,actual_steps=step,seconds=time.time()-began,success=finite,trace=trace,initial_hash=digest(init),final_hash=digest(state),CI_weights=w,CI_support=counts,weight_neff=float(w.sum()**2/(w@w)))
def predict(m,X,raw=False):
    if not m['success']:return np.full((len(X),4),np.nan)
    with torch.random.fork_rng():
        net=newnet(m['seed'],len(m['xm']));net.load_state_dict({k:torch.tensor(v) for k,v in m['state'].items()});net.eval()
        with torch.no_grad():y=m['ym']+m['ys']*net(torch.tensor((X-m['xm'])/m['xs'])).numpy()
    return y if raw else project(y)
