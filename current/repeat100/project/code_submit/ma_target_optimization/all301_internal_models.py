"""The24 prespecified MA regressors and one fixed historical reference."""
import hashlib
import itertools
import numpy as np
from . import abnormal_direction_mapping as reference_model
from .density_reweight_mapping import label_weights
from .full_curve_cost_sensitive import solve
from .metrics import states

REPRESENTATIONS=['fitted4','amp6','base_amp6']
ARMS=['target_only','HB_to_SA']
LOSSES=['uniform','class','inward4','inward16']
GRID=[dict(representation=r,arm=a,loss=l) for r,a,l in itertools.product(REPRESENTATIONS,ARMS,LOSSES)]
GRID.append(dict(reference='frozen_direction_corrected_composite'))


def training_view(data,train):
    allowed=np.r_[data['source_rows']['HB_to_SA'],train]
    y=np.full_like(data['y'],np.nan);y[allowed]=data['y'][allowed]
    assert np.isfinite(y[allowed]).all()
    return dict(data,y=y)


def design(data,blocks,train,query,arm):
    source=data['source_rows']['HB_to_SA'] if arm=='HB_to_SA' else np.array([],int)
    rows=np.r_[source,train];ns=len(source)
    if set(data['groups'][rows]).intersection(data['groups'][query]):raise ValueError('Training/query group overlap')
    tx=[];qx=[];stats=[]
    for x in blocks:
        mean=x[rows].mean(0);scale=np.maximum(x[rows].std(0),1e-6)
        tx.append((x[rows]-mean)/scale/np.sqrt(x.shape[1]));qx.append((x[query]-mean)/scale/np.sqrt(x.shape[1]))
        stats.append(hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest())
    tx=np.concatenate(tx,axis=1)/np.sqrt(len(blocks));qx=np.concatenate(qx,axis=1)/np.sqrt(len(blocks))
    norms=np.sum(tx*tx,axis=1)
    K=np.exp(-np.maximum(norms[:,None]+norms[None,:]-2*tx@tx.T,0))
    Q=np.exp(-np.maximum(np.sum(qx*qx,axis=1)[:,None]+norms[None,:]-2*qx@tx.T,0))
    values,vectors=np.linalg.eigh((K+K.T)/2);keep=values>max(float(values.max())*1e-12,1e-12)
    root=np.sqrt(values[keep]);v=vectors[:,keep]
    z=v*root;q=Q@(v/root)
    if ns:z=np.c_[z,np.r_[np.ones(ns),np.zeros(len(train))]];q=np.c_[q,np.zeros(len(query))]
    z=np.c_[z,np.ones(len(rows))];q=np.c_[q,np.ones(len(query))]
    y=data['y'][rows,2]
    assert np.isfinite(z).all() and np.isfinite(q).all() and np.isfinite(y).all()
    return z,q,y,ns,dict(scaling_sha256=stats,n_source=ns,n_local=len(train),kernel_rank=int(keep.sum()),
        local_rows_sha256=hashlib.sha256(np.asarray(train,dtype='int64').tobytes()).hexdigest())


def weights(y,n_source,kind):
    w=np.zeros(len(y));parts=[np.arange(n_source),np.arange(n_source,len(y))] if n_source else [np.arange(len(y))]
    for rows in parts:
        local=label_weights(y[rows],'class_inverse',20.)[0] if kind!='uniform' else np.ones(len(rows))
        w[rows]=local/len(rows)
    return w/w.mean()


def fit_all(data,blocks,cur,smooth,train,query):
    data=training_view(data,train)
    assert np.isnan(data['y'][query]).all()
    result=[];ledger=[]
    for rep in REPRESENTATIONS:
        for arm in ARMS:
            z,q,y,ns,diag=design(data,blocks[rep],train,query,arm)
            for loss in LOSSES:
                w=weights(y,ns,loss)
                kind,cost=('residual',float(loss[6:])) if loss.startswith('inward') else ('control',1.)
                coef,info=solve(z,(y-60)/10,w,kind,cost,alpha=.01)
                result.append(np.maximum(60+10*q@coef,0))
                ledger.append(dict(**diag,solver=info,weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                    coefficient_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
                    local_classes=np.bincount(states(data['y'][train,2]),minlength=3).tolist()))
    p,raw,diag=reference_model.predict_pinned(data,blocks,cur,smooth,train,query)
    result.append(p);ledger.append(dict(reference=diag,local_rows_sha256=hashlib.sha256(np.asarray(train,dtype='int64').tobytes()).hexdigest(),n_local=len(train)))
    return np.asarray(result),ledger


def boundary_gate(training_predictions,query_predictions):
    train_score=-np.minimum(abs(training_predictions-50),abs(training_predictions-70))
    threshold=np.quantile(train_score,.95,axis=-1,method='higher')
    query_score=-np.minimum(abs(query_predictions-50),abs(query_predictions-70))
    return query_score<=np.asarray(threshold)[...,None],threshold
