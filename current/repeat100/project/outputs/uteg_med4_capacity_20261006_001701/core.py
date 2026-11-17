import runtime as r
import numpy as np,torch,joblib,time,math
from torch import nn
from scipy.optimize import minimize
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CONFIGS={'TT-ORIGINAL':(24,4,2,64),'TT-SHALLOW':(24,4,1,64),'TT-SMALL':(12,2,1,32),'TT-TINY':(8,2,1,16),'MLP-8':None,'RIDGE-10':None}
PRIORITY=['RIDGE-10','MLP-8','TT-TINY','TT-SMALL','TT-SHALLOW','TT-ORIGINAL']
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
FLOOR=1e-6
class TT(nn.Module):
    def __init__(self,spec,seed):
        super().__init__();d,h,L,f=spec;self.val=nn.Parameter(torch.randn(6,d)*.05);self.feat=nn.Parameter(torch.randn(6,d)*.05)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,h,f,dropout=.1,batch_first=True),L,enable_nested_tensor=False);self.out=nn.Linear(d,3)
        with torch.random.fork_rng():torch.manual_seed(seed+700000);self.out=nn.Linear(d,4)
    def forward(self,x):return self.out(self.enc(x[:,:,None]*self.val[None]).mean(1))
def newnet(name,seed):
    torch.manual_seed(seed)
    return TT(CONFIGS[name],seed) if name.startswith('TT') else nn.Sequential(nn.Linear(6,8),nn.SiLU(),nn.Linear(8,4))
def digest(state):return r.hashlib.sha256(b''.join(np.asarray(v).tobytes() for k,v in sorted(state.items()))).hexdigest()
def fit(data,T,V,name,seed,steps=None):
    began=time.time();T=np.asarray(T,int);V=np.asarray(V,int);X=data['x'][T];Y=data['medical'][T];assert np.isfinite(X).all() and np.isfinite(Y).all()
    xm=X.mean(0);xs=np.maximum(X.std(0),1e-6);ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6);Z=(X-xm)/xs;target=(Y-ym)/ys
    base=dict(name=name,train=T,validation=V,xm=xm,xs=xs,ym=ym,ys=ys,seed=seed)
    if name=='RIDGE-10':
        zm=Z.mean(0);tm=target.mean(0);coef=np.linalg.solve((Z-zm).T@(Z-zm)+10*np.eye(6),(Z-zm).T@(target-tm));return dict(base,coef=coef,intercept=tm-zm@coef,params=28,active=28,steps=0,actual_steps=0,seconds=time.time()-began,success=True,trace=[])
    net=newnet(name,seed);init={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};params=sum(v.numel() for v in net.parameters());active=params-(net.feat.numel() if name.startswith('TT') else 0)
    xt=torch.tensor(Z);yt=torch.tensor(target);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001);best=math.inf;beststep=steps or 500;chosen=None;stale=0;trace=[];finite=True
    for step in range(1,(steps or 500)+1):
        net.train();opt.zero_grad();loss=((net(xt)-yt)**2).mean()
        if not torch.isfinite(loss):finite=False;break
        loss.backward();grad=float(nn.utils.clip_grad_norm_(net.parameters(),5));opt.step()
        if step%10==0:
            row=dict(step=step,loss=float(loss.detach()),gradient=grad)
            if len(V) and step>=50:
                net.eval()
                with torch.no_grad():yp=project(ym+ys*net(torch.tensor((data['x'][V]-xm)/xs)).numpy())
                error=float(np.mean(abs(yp-data['medical'][V])/ys));row['validation_MED4_MAE']=error
                if np.isfinite(error) and error<best-1e-10:best=error;beststep=step;chosen={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
                else:stale+=1
            trace.append(row)
            if len(V) and step>=50 and stale>=5:break
    if len(V) and chosen is not None:net.load_state_dict(chosen)
    if len(V) and chosen is None:finite=False
    state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    return dict(base,state=state,params=params,active=active,steps=beststep,actual_steps=step,seconds=time.time()-began,success=finite,trace=trace,initial_hash=digest(init),final_hash=digest(state))
def predict(m,X,raw=False):
    if not m['success']:return np.full((len(X),4),np.nan)
    z=(X-m['xm'])/m['xs']
    if m['name']=='RIDGE-10':y=m['ym']+m['ys']*(z@m['coef']+m['intercept'])
    else:
        with torch.random.fork_rng():
            net=newnet(m['name'],m['seed']);net.load_state_dict({k:torch.tensor(v) for k,v in m['state'].items()});net.eval()
            with torch.no_grad():y=m['ym']+m['ys']*net(torch.tensor(z)).numpy()
    return y if raw else project(y)
def calibrated(a,raw):
    if not(a['success'] and a['finite']):return np.full_like(raw,np.nan)
    return project(predict_cal(a,raw))
def metrics(med,Y,J):
    p=labels(ci(med));truth=labels(J);known=np.isfinite(J);good=known&np.isfinite(ci(med));classes=[k for k in [-1,0,1] if np.any(truth==k)]
    return dict(n=len(J),accuracy=float(np.mean(p[known]==truth[known])),BA=float(np.mean([np.mean(p[truth==k]==k) for k in classes])),CI_MAE=float(np.mean(abs(ci(med)[good]-J[good]))) if good.any() else np.nan,finite=float(np.isfinite(ci(med)).mean()),MED4_MAE=np.nanmean(abs(med-Y),0),recall={str(k):float(np.mean(p[truth==k]==k)) if np.any(truth==k) else None for k in [-1,0,1]},normal_FP=float(np.mean(p[truth==0]!=0)) if np.any(truth==0) else np.nan)
def select(OOF,Y,J):
    scores={}
    for name,med in OOF.items():
        z=metrics(med,Y,J);ok=np.isfinite(ci(med)).all();scores[name]=dict(S=.5*z['CI_MAE']/3+.5*(1-z['BA']) if ok else np.inf,eligible=bool(ok),**z)
    best=min(x['S'] for x in scores.values());name=next((n for n in PRIORITY if scores[n]['eligible'] and scores[n]['S']<=best+.001),None)
    return name,scores
def dump(p,obj):
    p=r.ROOT/p;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.tmp');joblib.dump(obj,q,compress=3);q.replace(p)
