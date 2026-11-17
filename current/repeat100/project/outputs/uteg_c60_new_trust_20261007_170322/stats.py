import runtime as r
import numpy as np
from models import labels,ci
def areas(s,e,W):
    # Integral over fractional, uniformly sampled tie blocks.
    ix=np.argsort(s,kind='stable');s=s[ix];e=e[ix];W=W[:,ix]
    total=W.sum(1);lo=.1*total;mass=np.zeros(len(W));err=mass.copy();out=mass.copy()
    cuts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        w=W[:,a:b].sum(1);ee=W[:,a:b]@e[a:b];d=np.divide(ee,w,out=np.zeros(len(W)),where=w>0);left=np.maximum(mass,lo);right=mass+w;ok=(right>left)&(left>0)&(w>0)
        out[ok]+=d[ok]*(right[ok]-left[ok])+(err[ok]-d[ok]*mass[ok])*np.log(right[ok]/left[ok]);mass+=w;err+=ee
    return np.divide(out,.9*total,out=np.full(len(W),np.nan),where=total>0)
def threshold(s,yh,yt,allowed=None):
    if allowed is None:allowed=np.ones(len(s),bool)
    known=yt!=9;valid=known&(yh!=9)&np.isfinite(s);area=areas(s[valid],(yh[valid]!=yt[valid]).astype(float),np.ones((1,sum(valid))))[0] if valid.any() else np.inf
    rows=[]
    for t in np.unique(s[valid&allowed]):
        a=known&allowed&np.isfinite(s)&(yh!=9)&(s<=t);n=sum(a);risk=np.mean(yh[a]!=yt[a]) if n else np.nan;cov=n/sum(known);ab=known&(yt!=0)
        if n>=5 and cov>=.2 and risk<=.15+1e-12:rows.append(dict(threshold=float(t),n=int(n),coverage=float(cov),risk=float(risk),abnormal_false_normal=float(sum(a&ab&(yh==0))/sum(ab)) if ab.any() else None))
    if not rows:return dict(eligible=False,threshold=None,n=0,coverage=0.,risk=None,aurc=area)
    return dict(min(rows,key=lambda z:(-z['coverage'],z['risk'],1 if z['abnormal_false_normal'] is None else z['abnormal_false_normal'],z['threshold'])),eligible=True,aurc=area)
def choose(rows):
    eligible=[x for x in rows if x['eligible']]
    if eligible:return min(eligible,key=lambda z:(-round(z['coverage'],12),round(z['risk'],12),round(1 if z['abnormal_false_normal'] is None else z['abnormal_false_normal'],12),z['aurc'],z['params'],z['order']))
    return dict(min(rows,key=lambda z:(z['aurc'],z['params'],z['order'])),eligible=False,threshold=None,n=0,coverage=0.,risk=None)
def actions(s,allow,t):return np.zeros(len(s),bool) if t is None else allow&np.isfinite(s)&(s<=t)
def risk_at(s,e,q):
    mass=0.;err=0.;need=q*len(s)
    for value in np.unique(s):
        ix=s==value;take=min(sum(ix),need-mass);err+=take*e[ix].mean();mass+=take
        if mass>=need-1e-12:break
    return err/mass if mass else np.nan
def metrics(med,s,a,J):
    yt=labels(J);yp=labels(ci(med));known=yt!=9;e=yp!=yt;den=a[:,known].sum();risk=(a[:,known]*e[:,known]).sum()/den if den else np.nan
    seeds=[];ars=[];rc=[]
    for k in range(len(a)):
        n=a[k,known].sum();rr=(a[k,known]&e[k,known]).sum()/n if n else np.nan
        finite=known&(yp[k]!=9);assert np.isfinite(s[k,finite]).all()
        ars.append(areas(s[k,finite],e[k,finite].astype(float),np.ones((1,sum(finite))))[0])
        rc.append([risk_at(s[k,finite],e[k,finite],q) for q in [.1,.25,.5,.75,1]])
        seeds.append(dict(seed=1000+k,coverage=a[k].mean(),accuracy=1-rr,n=n,wrong=sum(a[k,known]&e[k,known]),goal80=bool(a[k].mean()>=.2 and rr<=.2+1e-12)))
    classes={}
    for k in [-1,0,1]:
        ix=yt==k
        classes[str(k)]=dict(n=sum(ix),early_correct=np.mean(a[:,ix]&(yp[:,ix]==k)),early_false_normal=np.mean(a[:,ix]&(yp[:,ix]==0)) if k else None,other_error=np.mean(a[:,ix]&(yp[:,ix]!=k)&(yp[:,ix]!=0)) if k else np.mean(a[:,ix]&(yp[:,ix]!=k)),wait=np.mean(~a[:,ix])) if ix.any() else None
    return dict(accuracy=1-risk,risk=risk,coverage=a.mean(),early_correct=np.sum(a[:,known]&~e[:,known])/len(a),early_wrong=np.sum(a[:,known]&e[:,known])/len(a),classes=classes,seeds=seeds,seeds80=sum(v['goal80'] for v in seeds),AURC=np.mean(ars),risk_coverage=dict(zip(['0.1','0.25','0.5','0.75','1.0'],np.mean(rc,axis=0))),J10=np.mean(np.where(a[:,known],e[:,known],.1)),base_accuracy=np.mean(~e[:,known]),unknown_reference=sum(~known))
