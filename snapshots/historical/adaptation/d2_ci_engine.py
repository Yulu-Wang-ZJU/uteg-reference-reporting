import runtime as r
import numpy as np,joblib,time,copy
from scipy.special import expit
c=r.load('old60core',r.BASE/'core.py')
A=['H','L100','L10','K100','K10']
CFG=dict(id='ERROR-LR',family='ERROR-LR',regularizer=.1)
V=np.array([-.6516,-.3772,.1224,.0759])
def cond(X,raw,source):return np.column_stack([X,(c.project(raw)-source['ym'])/source['ys']])
def kernel(x,z):return np.exp(-np.maximum(((x[:,None,:]-z[None,:,:])**2).sum(-1),0)/16)
def fit_adapter(name,X,raw,H,J,source):
 valid=np.isfinite(J)&np.isfinite(H).all(1)&np.isfinite(X).all(1)&np.isfinite(raw).all(1)
 F=cond(X[valid],raw[valid],source);e=(J[valid]-c.ci(H[valid]))/3;mu=F.mean(0) if len(F) else np.zeros(8);sd=np.maximum(F.std(0),1e-6) if len(F) else np.ones(8);z=np.clip((F-mu)/sd,-10,10);rb=float(e.mean()) if len(e) else 0.;lam=float(name[1:]);w=source['ys']**2*V/(np.sum(source['ys']**2*V**2))
 m=dict(name=name,mean=mu,sd=sd,valid=valid,n=len(e),rb=rb,lambda_=lam,w=w,train_z=z,target=e,fallback=len(e)<8)
 if not m['fallback']:
  if name[0]=='L':m['coef']=np.linalg.solve(z.T@z+lam*np.eye(8),z.T@(e-rb))
  else:m['coef']=np.linalg.solve(kernel(z,z)+lam*np.eye(len(z)),e-rb)
 return m
def adapt(m,X,raw,H,source):
 if m is None or m['fallback']:return H.copy()
 z=np.clip((cond(X,raw,source)-m['mean'])/m['sd'],-10,10);pred=z@m['coef'] if m['name'][0]=='L' else kernel(z,m['train_z'])@m['coef'];delta=3*(m['rb']+pred);return c.project(H+delta[:,None]*m['w'])
def select(entries,yh,yt):
 # Entries may have different predictions, unlike old fixed-predictor selection.
 rows={}
 for name,z in entries.items():rows[name]=dict(c.threshold(z['score'],yh[z['adapter']],yt,.15),config=name,adapter=z['adapter'],gate=z['gate'],params=z['params'],order=z['order'])
 def choose(vs):
  good=[v for v in vs if v['eligible']]
  if good:return min(good,key=lambda v:(-round(v['coverage'],12),round(v['risk'],12),round(v.get('abnormal_false_normal') if v.get('abnormal_false_normal') is not None else 1,12),v['aurc'],v['params'],v['order']))
  v=min(vs,key=lambda v:(v['aurc'],v['params'],v['order']));return dict(v,threshold=None,eligible=False)
 fam={a:choose([v for v in rows.values() if v['adapter']==a]) for a in A};return dict(configurations=rows,adapters=fam,selected=choose(list(fam.values())))
def dump(path,z):c.dump(path,z)

def prepare():
 if (r.ROOT/'PROTOCOL_FREEZE.json').exists():return
 old=r.read(r.BASE/'ROLE_FREEZE.json');D=np.load(r.COHORT/'private/data.npz',allow_pickle=True);ids=D['ids'];g=D['groups'];test=old['test'];gs=sorted(set(g[test]),key=lambda group:min(str(ids[i]) for i in test if g[i]==group));np.random.default_rng(20261092).shuffle(gs);paths={0:[]}
 for group in gs:
  ix=sorted([i for i in test if g[i]==group],key=lambda i:str(ids[i]))
  for n,v in sorted(list(paths.items()),reverse=True):
   if n+len(ix)<=40 and n+len(ix) not in paths:paths[n+len(ix)]=v+ix
  if 40 in paths:break
 assert 40 in paths
 add=paths[40];pool=old['C60']+add;test=[i for i in test if i not in add];assert len(pool)==100 and len(test)==201
 for x,y in [(old['source'],pool),(pool,test),(old['source'],test)]:assert not set(g[x])&set(g[y])
 role=dict(source=old['source'],C60=old['C60'],C100=pool,added=add,test=test,added_ids=ids[add],blind_selection=True,seed=20261092,time=time.time());r.write('ROLE_FREEZE.json',role)
 np.savez_compressed(r.ROOT/'training.npz',indices=pool,x=D['x'][pool],medical=D['medical'][pool],ci=D['ci'][pool],groups=g[pool],ids=ids[pool])
 splits={}
 for s in range(1000,1020):
  outer=c.folds(pool,g,20261007+300000+100*s+99);positions=[]
  for code,f in enumerate(outer+[dict(T=pool,V=[])]):positions.append(dict(code=code,A=f['T'],B=f['V'],inner=outer if code==3 else c.folds(f['T'],g,20261007+300000+100*s+10*code)))
  splits[str(s)]=positions
 r.write('NESTED_SPLIT_MANIFEST.json',splits)
 r.write('TARGET_AMENDMENT.json',dict(D1_agreement=.90,D2_agreement=.85,coverage=.20,user_authorized=True,prior_90_not_claimed_met=True,clinical_safety=False))
 deps=[r.COHORT/'private/data.npz',r.BASE/'ROLE_FREEZE.json',r.BASE/'core.py',r.BASE/'runtime.py',r.BASE/'audit.py',r.OLD/'common.py',r.GATE/'risk_models.py',r.GATE/'independent_statistics.py',r.SC/'core.py',r.CF/'core.py',r.TH/'core.py']+[p/f'runs/{key}_seed{s}.joblib' for p,key in [(r.BASE,'C60'),(r.MAP,'D2')] for s in range(1000,1020)]
 r.write('PROTOCOL_FREEZE.json',dict(time=time.time(),protocol=r.sha(r.ROOT/'PROMPT.txt'),code={p.name:r.sha(p) for p in r.ROOT.glob('*.py')},dependencies={str(p):r.sha(p) for p in deps},roles=r.sha(r.ROOT/'ROLE_FREEZE.json'),training=r.sha(r.ROOT/'training.npz')))
 print('ROLE/PROTOCOL FROZEN C100/test201',flush=True)

def train_one(s,arm):
 key=f'{arm}_seed{s}'
 if (r.ROOT/f'freezes/{key}.json').exists():return
 started=time.time();D=dict(np.load(r.ROOT/'training.npz',allow_pickle=True));loc={int(i):j for j,i in enumerate(D['indices'])};rows=lambda ix:[loc[i] for i in ix];role=r.read('ROLE_FREEZE.json');pool=role[arm];pl={i:j for j,i in enumerate(pool)};base=joblib.load(r.MAP/f'runs/D2_seed{s}.joblib');source=base['families'][base['source_selection']]['model'];raw=c.source(source,D['x'][rows(pool)]);cache={}
 if arm=='C60':
  old=joblib.load(r.BASE/f'runs/C60_seed{s}.joblib');positions=[dict(code=code,A=ex['A'],B=ex['B'],inner=ex['nested']) for code,ex in old['examples'].items()];lookup={tuple(sorted(z['A'])):z['cal'] for z in old['calibrators'].values()}
 else:positions=r.read('NESTED_SPLIT_MANIFEST.json')[str(s)];lookup={}
 def get(Ai):
  token=r.hashlib.sha256(repr(tuple(sorted(Ai))).encode()).hexdigest()[:16]
  if token in cache:return token,cache[token]
  ix=[pl[i] for i in Ai];x=D['x'][rows(Ai)];J=D['ci'][rows(Ai)];rr=raw[ix]
  if tuple(sorted(Ai)) in lookup:cal=copy.deepcopy(lookup[tuple(sorted(Ai))])
  else:
   lk=f'{key}_{token}_H';r.ledger(lk,'Huber',train=Ai);cal=c.fit_cal(rr,D['medical'][rows(Ai)],source['ym'],source['ys']);r.finish(lk,success=cal['success'],finite=cal['finite'])
  H=c.calibrated(cal,rr);adapters={}
  for name in A[1:]:
   lk=f'{key}_{token}_{name}';r.ledger(lk,'CI-adapter',train=Ai);m=fit_adapter(name,x,rr,H,J,source);adapters[name]=m;r.finish(lk,success=not m['fallback'],finite=True)
  cache[token]=dict(train=Ai,Huber=cal,adapters=adapters);return token,cache[token]
 def predictions(obj,I):
  rr=raw[[pl[i] for i in I]];xx=D['x'][rows(I)];H=c.calibrated(obj['Huber'],rr);return dict(H=H,**{name:adapt(m,xx,rr,H,source) for name,m in obj['adapters'].items()})
 models={};examples={};oof={name:np.full((len(pool),4),np.nan) for name in A};entries={}
 for ai,name in enumerate(A):
  for gi,gate in enumerate(['ERROR-LR','MARGIN']):entries[name+'|'+gate]=dict(score=np.full(len(pool),np.nan),adapter=name,gate=gate,params=(0 if name=='H' else 9 if name.startswith('L') else len(pool)+1)+(14 if gate=='ERROR-LR' else 0),order=ai*2+gi)
 for pos in positions:
  code,Ai,Bi=pos['code'],pos['A'],pos['B'];al={i:j for j,i in enumerate(Ai)};med={name:np.full((len(Ai),4),np.nan) for name in A};links=[]
  for f in pos['inner']:
   T,V=f['T'],f['V'];assert not set(D['groups'][rows(T)])&set(D['groups'][rows(V+Bi)]);tok,obj=get(T);pp=predictions(obj,V)
   for name in A:med[name][[al[i] for i in V]]=pp[name]
   links.append(dict(T=T,V=V,token=tok))
  tok,obj=get(Ai);models[code]={};yt=c.labels(D['ci'][rows(Ai)]);ex=dict(A=Ai,B=Bi,nested=links,base=tok,med=med,features={},ytrue=yt)
  val=predictions(obj,Bi) if code<3 else None
  for name in A:
   F=c.features(D['x'][rows(Ai)],med[name],source);yh=c.labels(c.ci(med[name]));valid=(yh!=9)&(yt!=9);ex['features'][name]=F
   if name=='H' and arm=='C60':m=copy.deepcopy(old['models'][code]['LR-0.1']);assert np.allclose(F,old['examples'][code]['F'],equal_nan=True)
   else:
    lk=f'{key}_gate_{code}_{name}';r.ledger(lk,'gate',train=np.asarray(Ai)[valid]);m=c.fit(CFG,F[valid],yh[valid],yt[valid],20261007+200000+100*s+code);r.finish(lk,success=m['success'],finite=m['finite'],fallback=m.get('fallback'))
   models[code][name]=m
   if code<3:
    ix=[pl[i] for i in Bi];vv=val[name];vf=c.features(D['x'][rows(Bi)],vv,source);vy=c.labels(c.ci(vv));oof[name][ix]=vv;entries[name+'|ERROR-LR']['score'][ix]=c.score(m,CFG,vf,vy,np.zeros(len(Bi)));jj=c.ci(vv);entries[name+'|MARGIN']['score'][ix]=-np.minimum(abs(jj+3),abs(jj-3))
  examples[code]=ex
 yy=c.labels(D['ci'][rows(pool)]);selection=select(entries,{n:c.labels(c.ci(oof[n])) for n in A},yy)
 z=dict(pool=pool,raw=raw,cache=cache,models=models,examples=examples,oof=oof,entries=entries,selection=selection,source_family=base['source_selection']);dump(f'runs/{key}.joblib',z);r.write(f'freezes/{key}.json',dict(model_hash=r.sha(r.ROOT/f'runs/{key}.joblib'),selected=selection['selected'],seconds=time.time()-started,new_test_evaluated=False));print('FROZEN',key,selection['selected']['config'],selection['selected']['eligible'],round(time.time()-started,1),flush=True)

if __name__=='__main__':
 import sys
 prepare();fr=r.read('PROTOCOL_FREEZE.json')
 for p,h in fr['dependencies'].items():assert r.sha(p)==h
 for s in ([1000] if '--pilot' in sys.argv else range(1000,1020)):
  for arm in ['C60','C100']:train_one(s,arm)
 if '--pilot' not in sys.argv:r.write('ALL_MODELS_FREEZE.json',dict(time=time.time(),runs={p.stem:r.read(p) for p in (r.ROOT/'freezes').glob('*.json')},target_evaluated=False))
