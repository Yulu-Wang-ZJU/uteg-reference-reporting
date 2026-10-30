import runtime as r
import numpy as np,joblib
from scipy.special import expit

def ci(m):
 a=np.asarray(m);return -.6516*a[...,0]-.3772*a[...,1]+.1224*a[...,2]+.0759*a[...,3]-7.7922
def label(j):return np.where(np.isfinite(j),np.where(j<-3,-1,np.where(j>3,1,0)),9)
def project(m):
 a=np.array(m,dtype=float,copy=True);a[...,:3]=np.maximum(a[...,:3],0);a[...,3]=np.maximum(0,np.minimum(89.999999,a[...,3]));a[~np.isfinite(a).all(-1)]=np.nan;return a
def cal(m,raw):
 if not (m['success'] and m['finite']):return np.full_like(raw,np.nan)
 v=(project(raw)-m['mu'])/m['scale'];return project(m['mu']+m['scale']*(m['ubar']+(v-m['vbar'])@m['matrix']+m['intercept']))
def gate(m,cfg,F,yh,rawj):
 out=np.full(len(yh),np.inf);ok=yh!=9
 if cfg['family']=='SCALAR':
  j=rawj if cfg['id'].startswith('SOURCE') else F[:,8]*3;ok &=np.isfinite(j)
  if m.get('fallback'):p=np.tile(m['class_prior'],(sum(ok),1))
  else:
   z=np.clip((j[ok]-m['mean'])/m['sd'],-10,10);t=m['theta'];lo=expit(t[1]-t[0]*z);hi=expit(t[1]+np.logaddexp(0,t[2])+1e-6-t[0]*z);p=np.column_stack([lo,hi-lo,1-hi])
  out[ok]=1-p[np.arange(sum(ok)),yh[ok]+1];return out
 if not ok.any():return out
 pre=m['pre'];z=np.clip((np.where(np.isfinite(F[ok]),F[ok],pre['median'])-pre['mean'])/pre['sd'],-10,10)
 if m.get('fallback') or not m['finite']:v=np.full(len(z),m.get('constant_score',1.))
 elif cfg['family']=='ERROR-LR':v=expit(z@m['coef'][:-1]+m['coef'][-1])
 elif cfg['family']=='GP-RISK':
  dist=np.maximum(((z[:,None,:]-m['train_z'][None,:,:])**2).sum(2),0);v=np.clip(m['constant_mean']+np.exp(-dist/(2*m['length']**2))@m['alpha'],0,1)
 else:
  st=m['state'];h=z@st['fc1.weight'].T+st['fc1.bias'];v=1-expit((h*expit(h))@st['out.weight'].T+st['out.bias']).ravel()
 out[ok]=np.clip(v,0,1);return out

def area(s,e,W):
 # Vectorized independent integration over each distinct score's tie interval.
 good=np.isfinite(s)&np.isfinite(e);s=s[good];e=e[good];W=np.asarray(W[:,good],float)
 vals,inv=np.unique(s,return_inverse=True);mass=np.stack([W[:,inv==k].sum(1) for k in range(len(vals))],axis=1);err=np.stack([W[:,inv==k]@e[inv==k] for k in range(len(vals))],axis=1)
 end=np.cumsum(mass,axis=1);start=end-mass;prior=np.cumsum(err,axis=1)-err;N=mass.sum(1);left=np.maximum(start,.1*N[:,None]);active=(end>left)&(mass>0)&(left>0);density=np.divide(err,mass,out=np.zeros_like(err),where=mass>0);ratio=np.divide(end,left,out=np.ones_like(end),where=active)
 integrand=np.where(active,density*(end-left)+(prior-density*start)*np.log(ratio),0)
 return np.divide(integrand.sum(1),.9*N,out=np.full(len(W),np.nan),where=N>0)

def pick(entries,yh,yt):
 result={};known=yt!=9;ab=known&(yt!=0)
 for name,z in entries.items():
  s=z['score'];valid=known&(yh!=9)&np.isfinite(s);ar=area(s[valid],(yh[valid]!=yt[valid]).astype(float),np.ones((1,sum(valid))))[0];options=[]
  for t in sorted(set(s[valid])):
   a=valid&(s<=t);n=sum(a);risk=np.mean(yh[a]!=yt[a]);cov=n/sum(known);fn=sum(a&ab&(yh==0))/sum(ab) if ab.any() else None
   if n>=5 and cov>=.2 and risk<=.1+1e-12:options.append(dict(threshold=float(t),risk=risk,coverage=cov,abnormal_false_normal=fn,n=n))
  v=min(options,key=lambda q:(-q['coverage'],q['risk'],q['abnormal_false_normal'] if q['abnormal_false_normal'] is not None else 1,q['threshold'])) if options else dict(threshold=None,coverage=0.,risk=None,n=0)
  result[name]=dict(v,eligible=bool(options),aurc=ar,config=name,params=z['params'],order=z['order'],family=z['family'])
 def choose(vs):
  good=[v for v in vs if v['eligible']]
  if good:return min(good,key=lambda v:(-round(v['coverage'],12),round(v['risk'],12),round(v['abnormal_false_normal'] if v['abnormal_false_normal'] is not None else 1,12),v['aurc'],v['params'],v['order']))
  return min(vs,key=lambda v:(v['aurc'],v['params'],v['order']))
 fam={f:choose([v for v in result.values() if v['family']==f]) for f in set(v['family'] for v in result.values())};return choose(list(fam.values()))

def preflight():
 import core as c
 D=dict(np.load(r.ROOT/'training.npz',allow_pickle=True));loc={int(i):j for j,i in enumerate(D['indices'])};rows=lambda ix:[loc[i] for i in ix];checks=0;mx=0.;fail=[]
 def same(a,b,where):
  nonlocal checks,mx
  a=np.asarray(a);b=np.asarray(b);checks+=a.size
  if not np.array_equal(np.isfinite(a),np.isfinite(b)):fail.append(where+'finite');return
  ok=np.isfinite(a)&np.isfinite(b);err=float(np.max(abs(a[ok]-b[ok]))) if ok.any() else 0.;mx=max(mx,err)
  if err>1e-8:fail.append(where)
 for arm in ['C40','C60']:
  for s in range(1000,1020):
   z=joblib.load(r.ROOT/f'runs/{arm}_seed{s}.joblib');pool=z['pool'];pl={i:j for j,i in enumerate(pool)}
   for code,ex in z['examples'].items():
    A,B=ex['A'],ex['B'];assert not set(D['groups'][rows(A)])&set(D['groups'][rows(B)])
    for part in ex['nested']:
     T,V=part['T'],part['V'];assert not set(D['groups'][rows(T)])&set(D['groups'][rows(V+B)])
     cm=z['calibrators'][part['cal']];assert set(cm['A'])==set(T)
    cm=z['calibrators'][ex['base_cal']];assert set(cm['A'])==set(A)
    F=ex['F'];yh=ex['yhat'];rawj=ci(project(z['raw_pool'][[pl[i] for i in A]]))
    for cfg in c.CONFIGS:
     m=z['models'][code][cfg['id']];a=c.score(m,cfg,F,yh,rawj);b=gate(m,cfg,F,yh,rawj);same(a,b,'gate')
     if code==3:
      single=np.concatenate([c.score(m,cfg,F[i:i+1],yh[i:i+1],rawj[i:i+1]) for i in range(len(F))]);same(a,single,'single')
     if code<3:
      vf=ex['validation_F'];vy=label(ci(ex['validation_med']));vj=ci(project(z['raw_pool'][[pl[i] for i in B]]));same(gate(m,cfg,vf,vy,vj),z['entries'][cfg['id']]['score'][[pl[i] for i in B]],'OOFscore')
    same(cal(cm['cal'],z['raw_pool']),c.calibrated(cm['cal'],z['raw_pool']),'cal')
   ys=label(D['ci'][rows(pool)]);sel=pick(z['entries'],label(ci(z['OOF_med'])),ys);assert sel['config']==z['selection']['selected']['config'] and sel['threshold']==z['selection']['selected']['threshold']
   for name,entry in z['entries'].items():
    err=(label(ci(z['OOF_med']))!=ys).astype(float);same(area(entry['score'],err,np.ones((1,len(pool)))),c.areas(entry['score'],err,np.ones((1,len(pool)))),'AURC')
 # Constant tie blocks give random risk; both clinical boundaries are range-in.
 same(area(np.zeros(10),np.arange(10)%2,np.ones((1,10))),[.5],'constant');assert np.array_equal(label(np.array([-3.,3.])),[0,0])
 out=dict(passed=not fail,checks=checks,max_abs_difference=mx,tolerance=1e-8,failures=fail,independent_model_replay=True,new_fit_count=0)
 r.write('PRE_TARGET_VERIFICATION.json',out);print(out,flush=True);assert out['passed']
if __name__=='__main__':preflight()
