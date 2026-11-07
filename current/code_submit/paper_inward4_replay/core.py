"""Numerical models using the original U-TEG input coefficients."""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.metrics import r2_score
from code_submit.ma_target_optimization import all301_b085_inward4_transfer as selected
from code_submit.paper_b085_hospital_replay import common as accounting

WS = Path(__file__).resolve().parents[2]
OUT = WS/'outputs/current_ma_20261103'
BASE = OUT/'data'
OLD = WS/'outputs/paper_B085_hospital_replay_20261102'
SPLIT = BASE/'FROZEN_SPLITS.json'
SEEDS = list(range(1000,1020))
TARGET = {'HZ_to_SA':'SAHZU','HB_to_SA':'SAHZU','HZ+HB_to_SA':'SAHZU',
    'SA_to_HZ':'Huzhou','HB_to_HZ':'Huzhou','SA+HB_to_HZ':'Huzhou',
    'SA_to_HB':'Hubei','HZ_to_HB':'Hubei','SA+HZ_to_HB':'Hubei',
    'target_only_SA':'SAHZU','target_only_HZ':'Huzhou','target_only_HB':'Hubei'}
MAIN=['target_only_SA','HZ_to_SA','HZ+HB_to_SA']
HORIZONS=[300,600,900]
_CONTEXT=None
_SOURCE={}
RECONSTRUCTED=False

def original_B_at(h):return True

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):accounting.write(path,value)

def context():
    if _CONTEXT is not None:return _CONTEXT
    from . import current_version
    return current_version.activate()

def source_rows(direction,h):
    d,prefix,_,plan,*_=context()
    rows=np.asarray(plan['source_rows'].get(direction,[]),int)
    qc=d['input_qc'] if h==900 else np.isfinite(prefix[h]).all(1)
    return rows[qc[rows]]

def qc_at(h):
    d,prefix,*_=context()
    return d['input_qc'].astype(bool) if h==900 else np.isfinite(prefix[h]).all(1)

def feature(train,query,h,representation='B'):
    _,prefix,*_=context()
    x=prefix[h].copy()
    cols={'B':[1],'three':[0,1,2],'four':[0,1,2,3]}[representation]
    return x[:,cols]

def source_feature(direction,h,representation='B'):
    return feature([],[],h,representation)

def basis_fit(x):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    t=(x-mean)/scale/np.sqrt(x.shape[1]);n=np.sum(t*t,1)
    k=np.exp(-np.maximum(n[:,None]+n[None,:]-2*t@t.T,0))
    v,a=np.linalg.eigh((k+k.T)/2);keep=v>max(float(v.max())*1e-12,1e-12)
    root=np.sqrt(v[keep]);u=a[:,keep]
    return np.c_[u*root,np.ones(len(x))],dict(mean=mean,scale=scale,train=t,project=u/root,p=x.shape[1])

def basis_query(m,x):
    t=(x-m['mean'])/m['scale']/np.sqrt(m['p']);a=m['train']
    k=np.exp(-np.maximum(np.sum(t*t,1)[:,None]+np.sum(a*a,1)[None,:]-2*t@a.T,0))
    return np.c_[k@m['project'],np.ones(len(x))]

def source_model(direction,h,representation='B',loss='inward4'):
    key=(direction,h,representation,loss)
    if key in _SOURCE:return _SOURCE[key]
    rows=source_rows(direction,h)
    if not len(rows):return None
    d,*_=context();x=source_feature(direction,h,representation)
    if representation=='B' and loss=='inward4':
        m,a=selected.fit_source(d,x,rows);m['p']=1
    else:
        z,m=basis_fit(x[rows]);y=d['y'][rows,2];w=np.zeros(len(rows))
        for center in np.unique(d['centers'][rows]):
            mask=d['centers'][rows]==center
            w[mask]=selected.models.weights(y[mask],0,loss)/mask.sum()
        w/=w.mean();kind,cost=('residual',4.) if loss=='inward4' else ('control',1.)
        coef,a=selected.models.solve(z,(y-60)/10,w,kind,cost,alpha=.01);m['coef']=coef
    m['audit']=a;_SOURCE[key]=m
    return m

def source_predict(m,x):return np.maximum(60+10*basis_query(m,x)@m['coef'],0.)

def predict(direction,h,train,query,representation='B',loss='inward4',prior_override=None):
    d,*_=context();train=np.asarray(train,int);query=np.asarray(query,int)
    assert len(train)>1 and not set(d['groups'][train])&set(d['groups'][query])
    x=feature(train,query,h,representation)
    shadow=dict(d,y=np.full_like(d['y'],np.nan));shadow['y'][train]=d['y'][train]
    z,q,y,_,audit=selected.models.design(shadow,[x],train,query,'target_only')
    w=selected.models.weights(y,0,loss)
    src=source_model(direction,h,representation,loss) if prior_override is None else prior_override
    if src is None:base=np.zeros(len(train));bq=np.zeros(len(query))
    elif isinstance(src,dict) and 'predict_rows' in src:
        pt,pq=src['predict_rows'](train,query,h)
        base=(pt-60)/10;bq=(pq-60)/10
    elif callable(src):base=(src(x[train])-60)/10;bq=(src(x[query])-60)/10
    else:base=(source_predict(src,x[train])-60)/10;bq=(source_predict(src,x[query])-60)/10
    if loss=='inward4':coef,info=selected.solve_offset(z,(y-60)/10,w,base)
    else:
        penalty=np.r_[np.ones(z.shape[1]-1)*.01,0.]
        coef=np.linalg.solve(z.T@(w[:,None]*z)/len(y)+np.diag(penalty)+1e-12*np.eye(z.shape[1]),z.T@(w*((y-60)/10-base))/len(y))
        info={'objective_kind':'weighted_squared_final_MA'}
    p=np.maximum(60+10*(bq+q@coef),0.)
    assert np.isfinite(p).all()
    return p,dict(**audit,solver=info,coefficient_sha256=selected.ahash(coef),weights_sha256=selected.ahash(w)),x

def paper_spec(direction,seed,budget=0):
    d,_,_,_,role,*_=context();r=role['roles'][TARGET[direction]]
    pool=np.asarray(r['pool_full'] if budget!=20 else r['c20_by_seed'][str(seed)],int)
    positions=r['positions_full'][str(seed)] if budget!=20 else r['positions_c20'][str(seed)]
    if budget==40:
        initial=list(r['c20_by_seed'][str(seed)])
        remain=sorted(set(pool)-set(initial),key=lambda i:hashlib.sha256(('uteg-minimal-label-budget-v1|'+str(d['ids'][i])).encode()).hexdigest())
        pool=np.sort(initial+remain[:20])
    keep=set(pool);pos=[]
    for p in positions:
        pos.append(dict(A=[int(i) for i in p['A'] if i in keep],B=[int(i) for i in p['B'] if i in keep],
            inner=[dict(T=[int(i) for i in f['T'] if i in keep],V=[int(i) for i in f['V'] if i in keep]) for f in p['inner']]))
    return dict(pool=pool,test=np.asarray(r['test'],int),positions=pos)

def evaluate(train,query,inner,direction,h,representation='B',loss='inward4',prior_override=None):
    d,*_=context();qc=qc_at(h);train=np.asarray(train,int);query=np.asarray(query,int)
    train=train[qc[train]];valid=qc[query];qq=query[valid]
    assert not set(d['groups'][train])&set(d['groups'][query])
    srcrows=source_rows(direction,h)
    assert not set(d['groups'][srcrows])&set(d['groups'][query])
    pos={int(i):j for j,i in enumerate(train)};oof=np.full(len(train),np.nan);oof_b=np.full(len(train),np.nan)
    audits=[]
    for f in inner:
        a=np.asarray(f['T'],int);b=np.asarray(f['V'],int);a=a[qc[a]];b=b[qc[b]]
        if not len(b):continue
        assert set(a)|set(b)==set(train)
        p,au,x=predict(direction,h,a,b,representation,loss,prior_override)
        ix=[pos[int(v)] for v in b];oof[ix]=p;oof_b[ix]=x[b,0 if representation=='B' else 1];audits.append(au)
    assert np.isfinite(oof).all()
    p,au,x=predict(direction,h,train,qq,representation,loss,prior_override);audits.append(au)
    prediction=np.full(len(query),np.nan);prediction[valid]=p
    accepted=np.zeros(len(query),bool);a,t=selected.models.boundary_gate(oof[None,:],p[None,:]);accepted[valid]=a[0]
    return dict(prediction=prediction,accepted=accepted,inner=oof,inner_b=oof_b,
        query_b=x[query,0 if representation=='B' else 1],train=train,query=query,threshold=float(t[0]),audits=audits)

def summarize(pred,ref,accepted=None):
    pred=np.asarray(pred,float);ref=np.broadcast_to(ref,pred.shape);valid=np.isfinite(pred)&np.isfinite(ref)
    a=valid if accepted is None else np.asarray(accepted,bool)&valid
    pc=accounting.states(pred);truth=accounting.states(ref)
    m=accounting.classify(pc,truth,a)
    m['r2']=float(r2_score(ref[a],pred[a])) if a.sum()>1 and np.ptp(ref[a])>0 else None
    m['mae']=float(np.mean(abs(pred[a]-ref[a]))) if a.any() else None
    m['rmse']=float(np.sqrt(np.mean((pred[a]-ref[a])**2))) if a.any() else None
    m['class_mae']={k:float(np.mean(abs(pred[a&(truth==v)]-ref[a&(truth==v)]))) if np.any(a&(truth==v)) else None for k,v in [('low',-1),('normal',0),('high',1)]}
    m['binary_confusion']=[[m['fp_denominator']-m['fp'],m['fp']],[m['fn'],m['fn_denominator']-m['fn']]]
    m['B_from_original_data']=True;m['valid_UTEG_only_validation']=False
    return m
