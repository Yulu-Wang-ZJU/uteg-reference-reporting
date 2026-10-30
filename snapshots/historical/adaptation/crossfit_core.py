import runtime as r
import numpy as np,torch,joblib,time
from scipy.optimize import minimize
from scipy.special import softmax,expit
FLOOR=1e-6
rm=r.load('risk13',r.GATE/'risk_models.py');rm.N_FEATURES=13
rm4=r.load('risk4',r.GATE/'risk_models.py');rm4.N_FEATURES=4
ordinal=dict(r=r,np=np,torch=torch,time=time,minimize=minimize,rm=rm4)
r.extract(r.LOW/'models.py',['ordinal_logp','objective','fit','predict'],ordinal)
CONFIGS=rm.configs()+[dict(id='ORDINAL-MED4',family='ORDINAL-MED4')]
ORDER=['ERROR-LR','CI-MARGIN','ORDINAL-MED4','GP-RISK','FLAT-CLINICAL','FACTORED-CLINICAL','DOS-GATE','SELECTIVE-GATE']
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.ZERO/'verify_independent.py',['source'],globals());source_tt=source
r.extract(r.MAP/'verify.py',['source'],globals())
r.extract(r.GATE/'independent_statistics.py',['areas'],globals())
r.extract(r.TH/'core.py',['features','actions','threshold','select','ratio','risk_at','metrics','bootstrap','dump'],globals())
def calibrated(cal,raw):return project(predict_cal(cal,raw)) if cal['success'] and cal['finite'] else np.full_like(raw,np.nan)
def fit_gate(cfg,F,yh,yt,seed):
    if cfg['family']=='ORDINAL-MED4':return ordinal['fit'](cfg,F[:,4:8],yh,yt,seed)
    m=rm.fit_model(cfg,F,yh,yt,seed)
    if cfg['family']=='DOS-GATE':m['parameter_count_registered']=121
    elif cfg['family'] in ['FLAT-CLINICAL','FACTORED-CLINICAL','SELECTIVE-GATE']:m['parameter_count_registered']=130
    if 'state' in m:
        n=sum(v.size for v in m['state'].values());assert n==m['parameter_count_registered'];m['parameter_count_effective']=n-(9 if m.get('beta_frozen') else 0)
    m['model_hash']=rm._hash_value({k:v for k,v in m.items() if k!='model_hash'});return m
def score(m,F,yh):
    if m['family']=='ORDINAL-MED4':return ordinal['predict'](m,F[:,4:8],yh)
    out=np.full(len(yh),np.inf);ok=yh!=9
    if ok.any():out[ok]=rm.predict_model(m,F[ok],yh[ok])
    return out
def folds(indices,groups,seed):
    ids=list(indices);gs=list(dict.fromkeys(str(groups[i]) for i in ids));np.random.default_rng(seed).shuffle(gs);ff=[[],[],[]]
    for g in gs:
        members=[i for i in ids if str(groups[i])==g];k=min(range(3),key=lambda j:(len(ff[j]),j));ff[k]+=members
    return [dict(T=[i for i in ids if i not in V],V=sorted(V)) for V in ff]
