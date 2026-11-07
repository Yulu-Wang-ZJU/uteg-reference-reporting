import runtime as r
import numpy as np,joblib,importlib.util
from scipy.special import expit
sp=importlib.util.spec_from_file_location('frozen_numpy_gate',r.GATE/'independent_models.py');im=importlib.util.module_from_spec(sp);sp.loader.exec_module(im);im.N_FEATURES=13
r.extract(r.GATE/'independent_statistics.py',['areas'],globals())
def ci(m):return np.asarray(m)@np.array([-.6516,-.3772,.1224,.0759])-7.7922
def labels(j):return np.where(~np.isfinite(j),9,np.where(j < -3,-1,np.where(j>3,1,0)))
def features(x,m,b):
    j=ci(m);y=labels(j);return np.column_stack([x,(m-b['ym'])/b['ys'],j/3,np.minimum(abs(j+3),abs(j-3))/3,*[(y==k).astype(float) for k in [-1,0,1]]])
def score(mod,F,yh):
    out=np.full(len(F),np.inf);ok=yh!=9
    if not ok.any():return out
    if mod['family']=='ORDINAL-MED4':
        if mod.get('fallback'):p=np.tile(mod['class_prior'],(sum(ok),1))
        else:
            pre=mod['pre'];x=F[ok,4:8];z=np.clip((np.where(np.isfinite(x),x,pre['median'])-pre['mean'])/pre['sd'],-10,10);t=mod['theta'];a=t[4]-z@t[:4];b=a+np.logaddexp(0,t[5])+1e-6;lo=expit(a);hi=expit(b);p=np.column_stack([lo,hi-lo,1-hi])
        out[ok]=1-p[np.arange(sum(ok)),yh[ok]+1]
    else:out[ok]=im.predict_independent(mod,F[ok],yh[ok])
    return out
def actions(s,yh,t):return np.zeros(len(s),bool) if t is None else np.isfinite(s)&(yh!=9)&(s<=t)
def threshold(s,yh,yt,limit):
    known=yt!=9;valid=known&(yh!=9)&np.isfinite(s);area=areas(s[valid],(yh[valid]!=yt[valid]).astype(float),np.ones((1,sum(valid))))[0] if valid.any() else np.nan;options=[]
    for t in np.unique(s[valid]):
        a=known&actions(s,yh,t);n=int(a.sum());risk=float((yh[a]!=yt[a]).mean());cov=n/sum(known);ab=known&(yt!=0)
        if n>=5 and cov>=.2 and risk<=limit+1e-12:options.append(dict(threshold=float(t),n=n,risk=risk,coverage=cov,abnormal_false_normal=float(sum(a&ab&(yh==0))/sum(ab)) if ab.any() else None))
    if not options:return dict(eligible=False,threshold=None,n=0,risk=None,coverage=0.,aurc=area)
    return dict(min(options,key=lambda z:(-z['coverage'],z['risk'],1 if z['abnormal_false_normal'] is None else z['abnormal_false_normal'],z['threshold'])),eligible=True,aurc=area)
def select(entries,yh,yt,limit):
    rows={name:dict(threshold(z['score'],yh,yt,limit),config=name,family=z['family'],params=z['params'],order=z['order']) for name,z in entries.items()}
    def choose(vs):
        good=[v for v in vs if v['eligible']]
        if good:return min(good,key=lambda v:(-round(v['coverage'],12),round(v['risk'],12),round(v['abnormal_false_normal'] if v['abnormal_false_normal'] is not None else 1,12),v['aurc'] if np.isfinite(v['aurc']) else np.inf,v['params'],v['order']))
        z=min(vs,key=lambda v:(v['aurc'] if np.isfinite(v['aurc']) else np.inf,v['params'],v['order']));return dict(z,eligible=False,threshold=None,n=0,risk=None,coverage=0.)
    families={f:choose([v for v in rows.values() if v['family']==f]) for f in dict.fromkeys(v['family'] for v in rows.values())}
    return dict(configurations=rows,families=families,selected=choose(list(families.values())))
def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=np.asarray(b)>0)
def risk_at(s,e,q):
    order=np.argsort(s,kind='stable');s=s[order];e=e[order];need=q*len(s);mass=err=0.
    for value in np.unique(s):
        ix=s==value;take=min(float(sum(ix)),need-mass);err+=take*float(e[ix].mean());mass+=take
        if mass>=need-1e-12:break
    return err/mass if mass else np.nan
def metrics(med,s,a,J):
    yt=labels(J);yp=labels(ci(med));known=yt!=9;e=yp!=yt;n=a[:,known].sum();risk=float((a[:,known]*e[:,known]).sum()/n) if n else np.nan
    seeds=[];ars=[];qrows=[]
    for k in range(20):
        nn=a[k,known].sum();rr=float((a[k,known]*e[k,known]).sum()/nn) if nn else np.nan;ok=known&(yp[k]!=9)&np.isfinite(s[k]);ar=areas(s[k,ok],e[k,ok].astype(float),np.ones((1,sum(ok))))[0] if ok.any() else np.nan;ars.append(ar);qrows.append({str(q):risk_at(s[k,ok],e[k,ok],q) for q in [.1,.25,.5,.75,1]})
        seeds.append(dict(seed=1000+k,coverage=a[k].mean(),risk=rr,n=nn,correct=(a[k,known]*(~e[k,known])).sum(),goal90=bool(a[k].mean()>=.2 and rr<=.1+1e-12),goal95=bool(a[k].mean()>=.2 and rr<=.05+1e-12)))
    classes={}
    for cl in [-1,0,1]:
        ix=yt==cl
        classes[str(cl)]=dict(n=int(ix.sum()),early_correct=float((a[:,ix]*(yp[:,ix]==cl)).mean()),early_false_normal=float((a[:,ix]*(yp[:,ix]==0)).mean()) if cl else None,other_error=float((a[:,ix]*(yp[:,ix]!=cl)*(yp[:,ix]!=0)).mean()) if cl else float((a[:,ix]*(yp[:,ix]!=0)).mean()),wait=float((1-a[:,ix]).mean())) if ix.any() else None
    return dict(accuracy=1-risk,risk=risk,coverage=a.mean(),early_correct=(a[:,known]*(~e[:,known])).sum()/20,early_wrong=(a[:,known]*e[:,known]).sum()/20,classes=classes,seeds=seeds,seeds90=sum(z['goal90'] for z in seeds),seeds95=sum(z['goal95'] for z in seeds),AURC=np.mean(ars),risk_coverage={str(q):np.mean([z[str(q)] for z in qrows]) for q in [.1,.25,.5,.75,1]},J_10=np.where(a[:,known]>0,a[:,known]*e[:,known]+(1-a[:,known])*.1,.1).mean(),finite_prediction_fraction=float((yp!=9).mean()),unknown_reference=int(sum(~known)))
def bootstrap(med,a,J,W):
    y=labels(J);ok=y!=9;e=labels(ci(med))!=y;aa=a.mean(0);ae=(a*e).mean(0)
    return dict(risk=ratio(W[:,ok]@ae[ok],W[:,ok]@aa[ok]),coverage=ratio(W@aa,W.sum(1)))
def dump(p,z):
    p=r.ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(z,p,compress=3)
