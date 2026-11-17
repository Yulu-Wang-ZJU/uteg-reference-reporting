import runtime as r
import numpy as np
from scipy.optimize import minimize,linprog
from scipy.special import expit,logit
NAMES=['MARGIN','ERROR-LR','NORMAL-LR','TRUST-SCORE','OSP','QCI']
PARAMS=[0,14,28,0,42,28]
def ci(m):return np.asarray(m)@np.array([-.6516,-.3772,.1224,.0759])-7.7922
def labels(j):return np.where(~np.isfinite(j),9,np.where(j < -3,-1,np.where(j>3,1,0)))
def features(x,m,source):
    j=ci(m);y=labels(j)
    return np.column_stack([x,(m-source['ym'])/source['ys'],j/3,np.minimum(abs(j+3),abs(j-3))/3,*[(y==k).astype(float) for k in [-1,0,1]]])
def preprocess(F):
    med=[];mu=[];sd=[]
    for col in F.T:
        v=col[np.isfinite(col)];med.append(np.median(v) if len(v) else 0);mu.append(np.mean(v) if len(v) else 0);sd.append(max(np.std(v),1e-6) if len(v) else 1)
    return dict(median=np.array(med),mean=np.array(mu),sd=np.array(sd))
def transform(pre,F):return np.clip((np.where(np.isfinite(F),F,pre['median'])-pre['mean'])/pre['sd'],-10,10)
def supported(e):return len(e)>=8 and sum(e)>=2 and sum(1-e)>=2
def lr(F,e):
    p=preprocess(F);m=dict(pre=p,n=len(e),errors=sum(e),constant_score=(sum(e)+.5)/(len(e)+1),status='fallback_const')
    if not supported(e):return m
    z=transform(p,F);n=len(e)
    def fun(t):
        h=z@t[:-1]+t[-1];d=expit(h)-e
        return np.mean(np.logaddexp(0,h)-e*h)+.1*sum(t[:-1]**2),np.r_[z.T@d/n+.2*t[:-1],d.mean()]
    fit=minimize(fun,np.zeros(F.shape[1]+1),jac=True,method='L-BFGS-B',options=dict(maxiter=200,ftol=1e-9,gtol=1e-9))
    m.update(success=bool(fit.success),message=fit.message,iterations=fit.nit)
    if np.isfinite(fit.x).all():m.update(coef=fit.x,status='fit' if fit.success else 'finite_unconverged')
    return m
def lr_score(m,F):
    if 'coef' not in m or m.get('status') in ['fallback_const','exception_fallback'] or m.get('finite') is False:return np.full(len(F),m['constant_score'])
    z=transform(m['pre'],F);return expit(z@m['coef'][:-1]+m['coef'][-1])
def osp_head(z,e,N,eps):
    m=dict(n=len(e),errors=sum(e),eps=eps,constant_score=(sum(e)+.5)/(len(e)+1),status='fallback_const')
    if not supported(e):return m
    x=np.column_stack([z,np.ones(len(z))]);t=np.r_[np.zeros(z.shape[1]),logit(.01)]
    def obj(t):
        g=expit(x@t);d=g*(1-g)
        return -g.sum()/N+.1*sum(t[:-1]**2),-x.T@d/N+np.r_[.2*t[:-1],0]
    def con(t):return eps-np.sum(expit(x@t)*e)/N
    def cj(t):
        g=expit(x@t);return -x.T@(g*(1-g)*e)/N
    out=minimize(obj,t,jac=True,method='SLSQP',bounds=[(-5,5)]*z.shape[1]+[(-12,12)],constraints=[dict(type='ineq',fun=con,jac=cj)],options=dict(maxiter=200,ftol=1e-9))
    m.update(success=bool(out.success),message=out.message,iterations=out.nit,constraint=float(con(out.x)),objective=float(out.fun))
    if np.isfinite(out.x).all() and con(out.x)>=-1e-8:m.update(coef=out.x,status='fit' if out.success else 'finite_unconverged')
    else:m['status']='failed_fallback'
    return m
def quantile(z,y,q):
    n,p=z.shape;m=dict(q=q,n=n,status='fallback',constant=float(np.quantile(y,q)) if n else 0.)
    if n<8:return m
    # w=w_plus-w_minus; b unrestricted; residual=u-v.
    cost=np.r_[np.full(2*p,.05),0,np.full(n,q/n),np.full(n,(1-q)/n)]
    eq=np.column_stack([z,-z,np.ones(n),np.eye(n),-np.eye(n)])
    out=linprog(cost,A_eq=eq,b_eq=y,bounds=[(0,None)]*(2*p)+[(None,None)]+[(0,None)]*(2*n),method='highs')
    m.update(success=bool(out.success),message=out.message)
    if out.success and np.isfinite(out.x).all():m.update(status='fit',coef=np.r_[out.x[:p]-out.x[p:2*p],out.x[2*p]],objective=float(out.fun))
    return m
def fit_all(F,med,J,global_lr,key):
    yp=labels(ci(med));yt=labels(J);ok=(yp!=9)&(yt!=9);F=F[ok];yp=yp[ok];yt=yt[ok];J=J[ok];e=(yp!=yt).astype(float)
    pre=preprocess(F);z=transform(pre,F)
    r.ledger(key+'_normal','NORMAL-LR',n=int(sum(yp==0)));normal=lr(F[yp==0],e[yp==0])
    r.ledger(key+'_trust','TRUST-SCORE-support',n=len(F))
    trustpre=preprocess(F[:,:4]);trust=dict(pre=trustpre,z=transform(trustpre,F[:,:4]),y=yt)
    heads={}
    for k in [-1,0,1]:
        ix=yp==k;r.ledger(key+'_osp_'+str(k),'OSP-head',n=int(sum(ix)));heads[k]=osp_head(z[ix],e[ix],len(F),.05*(sum(ix)+1)/(len(F)+3))
    residual=(J-F[:,8]*3)/3;qs=[]
    for q in [.1,.9]:
        r.ledger(key+'_quantile_'+str(q),'QCI-head',n=len(F));qs.append(quantile(z,residual,q))
    return {'MARGIN':{},'ERROR-LR':global_lr,'NORMAL-LR':dict(normal=normal,global_lr=global_lr),'TRUST-SCORE':trust,'OSP':dict(pre=pre,heads=heads),'QCI':dict(pre=pre,heads=qs)}
def score(name,m,F):
    j=F[:,8]*3;yp=labels(j);allowed=yp!=9;aux={}
    if name=='MARGIN':s=-np.minimum(abs(j+3),abs(j-3))
    elif name=='ERROR-LR':s=lr_score(m,F)
    elif name=='NORMAL-LR':
        s=lr_score(m['global_lr'],F);ix=yp==0;s[ix]=lr_score(m['normal'],F[ix])
    elif name=='TRUST-SCORE':
        z=transform(m['pre'],F[:,:4]);s=np.ones(len(F));d={}
        for k in [-1,0,1]:
            lib=m['z'][m['y']==k]
            d[k]=np.sort(np.sqrt(np.maximum(((z[:,None,:]-lib[None,:,:])**2).sum(2),0)),axis=1)[:,:min(3,len(lib))].mean(1) if len(lib) else np.full(len(F),np.inf)
        for i,k in enumerate(yp):
            a=d.get(k,np.full(len(F),np.inf))[i];b=min(d[c][i] for c in [-1,0,1] if c!=k)
            if not np.isfinite(a+b):allowed[i]=False
            else:s[i]=a/(a+b) if a+b>0 else .5
    elif name=='OSP':
        z=transform(m['pre'],F);s=np.ones(len(F))
        for k,h in m['heads'].items():
            ix=yp==k;s[ix]=1-expit(z[ix]@h['coef'][:-1]+h['coef'][-1]) if 'coef' in h else h['constant_score']
    elif name=='QCI':
        z=transform(m['pre'],F);v=np.column_stack([z@h['coef'][:-1]+h['coef'][-1] if 'coef' in h else np.full(len(F),h['constant']) for h in m['heads']])
        aux['crossed']=v[:,0]>v[:,1];lo=j+3*v.min(1);hi=j+3*v.max(1)
        s=np.where(yp==0,np.maximum((-3-lo)/3,(hi-3)/3),np.where(yp==-1,(hi+3)/3,(3-lo)/3))
        allowed&=np.where(yp==0,(lo>=-3)&(hi<=3),np.where(yp==-1,hi < -3,lo>3))
        aux.update(lower=lo,upper=hi)
    else:raise ValueError(name)
    return np.where(yp!=9,s,np.inf),allowed,aux
