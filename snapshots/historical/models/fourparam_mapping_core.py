import runtime as r
import numpy as np,torch,joblib,time,math,warnings
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
FLOOR=1e-6
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.CAP/'core.py',['TT','digest','calibrated','metrics','dump'],globals())
r.extract(r.CLIN/'core.py',['weights','loss_parts'],globals())
FAMILIES=['RIDGE','PLS','KRR','SVR','MLP','TT']
CONFIGS={**{f'RIDGE-{v}':dict(family='RIDGE',alpha=v) for v in [100,10,1]},**{f'PLS-{v}':dict(family='PLS',components=v) for v in [2,3]},**{f'KRR-{v}':dict(family='KRR',alpha=v) for v in [10,1]},**{f'SVR-{v}':dict(family='SVR',C=v) for v in [1,10]},'MLP-8':dict(family='MLP'),'TT-SMALL':dict(family='TT')}
def newnet(family,seed):
    torch.manual_seed(seed)
    if family=='MLP':return nn.Sequential(nn.Linear(4,8),nn.SiLU(),nn.Linear(8,4))
    net=TT((12,2,1,32),seed);net.val=nn.Parameter(net.val[[2,3,4,5]].detach().clone());net.feat=nn.Parameter(net.feat[[2,3,4,5]].detach().clone());return net
def kernel(a,b):return np.exp(-.25*np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0))
def fit(data,T,V,name,seed,steps=None):
    start=time.time();T=np.array(T,int);V=np.array(V,int);X=data['x'][T];Y=data['medical'][T];J=data['ci'][T];xm=X.mean(0);xs=np.maximum(X.std(0),1e-6);ym=Y.mean(0);ys=np.maximum(Y.std(0),1e-6);Z=(X-xm)/xs;target=(Y-ym)/ys;cfg=CONFIGS[name];fam=cfg['family'];base=dict(name=name,family=fam,config=cfg,train=T,validation=V,xm=xm,xs=xs,ym=ym,ys=ys,seed=seed,steps=0,actual_steps=0,trace=[],success=True)
    with warnings.catch_warnings(record=True) as ww:
        if fam=='RIDGE':
            coef=np.linalg.solve(Z.T@Z+cfg['alpha']*np.eye(4),Z.T@target);base.update(coef=coef,intercept=np.zeros(4),params=20,active=20,df=float(np.trace(np.linalg.solve(Z.T@Z+cfg['alpha']*np.eye(4),Z.T@Z))))
        elif fam=='PLS':
            obj=PLSRegression(n_components=cfg['components'],scale=False,max_iter=500,tol=1e-6).fit(Z,target);intercept=obj.predict(np.zeros((1,4)))[0];coef=obj.predict(np.eye(4))-intercept;base.update(coef=coef,intercept=intercept,params=20,active=20,latent_components=cfg['components'],latent_x_rotations=obj.x_rotations_,latent_y_loadings=obj.y_loadings_,latent_x_mean=obj._x_mean,latent_y_mean=obj._y_mean,iterations=obj.n_iter_)
        elif fam=='KRR':
            K=kernel(Z,Z);dual=np.linalg.solve(K+cfg['alpha']*np.eye(len(Z)),target);base.update(train_z=Z,dual=dual,params=int(dual.size),active=int(dual.size),df=float(np.trace(np.linalg.solve(K+cfg['alpha']*np.eye(len(Z)),K))))
        elif fam=='SVR':
            mods=[]
            for k in range(4):
                sv=SVR(C=cfg['C'],epsilon=.1,gamma=.25,kernel='rbf',tol=1e-3,max_iter=10000).fit(Z,target[:,k]);mods.append(dict(support=sv.support_vectors_,dual=sv.dual_coef_[0],intercept=float(sv.intercept_[0]),status=int(sv.fit_status_),iterations=int(sv.n_iter_)))
            base.update(svm=mods,params=sum(len(m['dual'])+1 for m in mods),active=sum(len(m['dual'])+1 for m in mods),success=all(m['status']==0 for m in mods))
        else:
            net=newnet(fam,seed);init={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};w,counts=weights(J);xx=torch.tensor(Z);yy=torch.tensor(target);ymt=torch.tensor(ym);yst=torch.tensor(ys);jt=torch.tensor(J);wt=torch.tensor(w);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001);best=np.inf;beststep=steps or 500;chosen=None;stale=0
            for step in range(1,(steps or 500)+1):
                net.train();opt.zero_grad();lm,lv,lc,_,_=loss_parts(net(xx),yy,ymt,yst,jt,wt);loss=lm+lv+.25*lc
                if not torch.isfinite(loss):base['success']=False;break
                loss.backward();grad=float(nn.utils.clip_grad_norm_(net.parameters(),5));opt.step()
                if step%10==0:
                    row=dict(step=step,loss=float(loss.detach()),gradient=grad)
                    if len(V) and step>=50:
                        net.eval()
                        with torch.no_grad():p=project(ym+ys*net(torch.tensor((data['x'][V]-xm)/xs)).numpy())
                        error=float(np.mean(abs(p-data['medical'][V])/ys));row['validation_MED4_MAE']=error
                        if error<best-1e-10:best=error;beststep=step;chosen={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
                        else:stale+=1
                    base['trace'].append(row)
                    if len(V) and step>=50 and stale>=5:break
            if len(V) and chosen is not None:net.load_state_dict(chosen)
            if len(V) and chosen is None:base['success']=False
            state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};params=sum(v.numel() for v in net.parameters());base.update(state=state,params=params,active=params-(net.feat.numel() if fam=='TT' else 0),steps=beststep,actual_steps=step,CI_weights=w,CI_support=counts,initial_hash=digest(init),final_hash=digest(state))
        base['warnings']=[str(w.message) for w in ww]
    base['seconds']=time.time()-start;return base
def predict(m,X,raw=False):
    fam=m.get('family','TT');z=(np.atleast_2d(X)-m['xm'])/m['xs']
    if not m['success']:return np.full((len(z),4),np.nan)
    if fam in ['RIDGE','PLS']:p=z@m['coef']+m['intercept']
    elif fam=='KRR':p=kernel(z,m['train_z'])@m['dual']
    elif fam=='SVR':p=np.column_stack([kernel(z,v['support'])@v['dual']+v['intercept'] for v in m['svm']])
    else:
        with torch.random.fork_rng():
            net=newnet(fam,m['seed']);net.load_state_dict({k:torch.tensor(v) for k,v in m['state'].items()});net.eval()
            with torch.no_grad():p=net(torch.tensor(z)).numpy()
    y=m['ym']+m['ys']*p;return y if raw else project(y)
def choose(oof,Y,J):
    scores={}
    for name,p in oof.items():
        v=metrics(p,Y,J);scores[name]=dict(S=.5*v['CI_MAE']/3+.5*(1-v['BA']) if np.isfinite(p).all() else np.inf,**v)
    eligible=[n for n,z in scores.items() if np.isfinite(z['S'])]
    if not eligible:return None,scores
    best=min(scores[n]['S'] for n in eligible);return next(n for n in oof if n in eligible and scores[n]['S']<=best+.001),scores
