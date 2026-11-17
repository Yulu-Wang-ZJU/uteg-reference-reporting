"""Full-waveform state classifiers with training-only conditional MA experts.

Non-neural controls for the jointly trained full-curve network. These are new
piecewise mappings; numerical MA errors use the very same selected expert output.
"""
import argparse
import hashlib
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .full_curve_cost_sensitive import load_waveforms,kernel_design
from .feature_combination_audit import assemble,detailed
from .full_curve_joint_state import decode,choose_threshold,POLICIES
from .metrics import states,selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

RECIPES=['target_class','HB_dense']
MODELS=[(kind,C) for kind in ['three_class','abnormal_then_direction'] for C in [1.,10.]]


def classifier(x,label,w,C):
    classes=np.unique(label)
    if len(classes)==1:return None,int(classes[0])
    model=LogisticRegression(C=C,max_iter=5000,tol=1e-7,solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,label,sample_weight=w)
    return model,None


def probability(model,constant,query,nclasses):
    p=np.zeros((len(query),nclasses))
    if model is None:p[:,constant]=1.
    else:
        p[:,model.classes_.astype(int)]=np.maximum(model.predict_proba(query),1e-12)
        p/=p.sum(1,keepdims=True)
    return p


def signature(model,constant):
    if model is None:return str(constant)
    return hashlib.sha256(model.coef_.tobytes()+model.intercept_.tobytes()+model.classes_.tobytes()).hexdigest()


def fit_predict(data,x,a,b,recipe):
    tx,y,w,mean,scale,ns=assemble(data,x,a,recipe);tx=tx[:,:-1]
    qx=(x[b]-mean)/scale
    if ns:qx=np.c_[qx,np.zeros(len(b))]
    z,q,_=kernel_design(tx,qx);z=z[:,:-1];q=q[:,:-1]
    labels=states(y)
    K=rbf_kernel(qx,tx,gamma=1/tx.shape[1])
    conditional=np.empty((len(b),3));present=np.unique(labels)
    for k in range(3):
        keep=labels==k
        # Prefer local training responses for the MA expert. Source responses
        # supply a tail expert only when that class is absent locally.
        target=keep & (np.arange(len(y))>=ns)
        if target.any():keep=target
        if not keep.any():
            conditional[:,k]=[45.,60.,75.][k]
            continue
        ww=K[:,keep]*w[keep][None,:]
        ww=np.maximum(ww,1e-300);ww/=ww.sum(1,keepdims=True)
        conditional[:,k]=np.clip(ww@y[keep],y[keep].min(),y[keep].max())
    probabilities=[];diagnostics=[]
    for kind,C in MODELS:
        if kind=='three_class':
            model,constant=classifier(z,labels,w,C)
            p=probability(model,constant,q,3);digest=[signature(model,constant)]
        else:
            abnormal=labels!=1
            detector,constant=classifier(z,abnormal.astype(int),w,C)
            ab=probability(detector,constant,q,2)[:,1]
            direction,dc=classifier(z[abnormal],(labels[abnormal]==2).astype(int),w[abnormal],C)
            high=probability(direction,dc,q,2)[:,1]
            p=np.c_[ab*(1-high),1-ab,ab*high];digest=[signature(detector,constant),signature(direction,dc)]
        p[:,[k for k in range(3) if k not in present]]=0
        p/=p.sum(1,keepdims=True)
        probabilities.append(p)
        diagnostics.append(dict(kind=kind,C=C,model_sha256=digest,training_counts=np.bincount(labels,minlength=3).tolist(),
            training_scale_sha256=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest(),n_source=ns,n_target=len(a)))
    return np.asarray(probabilities),conditional,diagnostics


def run(seeds,name):
    out=ROOT/'outputs/uteg_mapping_target_runs'/name;(out/'private').mkdir(parents=True,exist_ok=True)
    data,parts,extra,reps,mask,hashes=load_waveforms(seeds);x=reps['raw900']
    grid=[dict(recipe=r,kind=k,C=c,policy=p) for r in RECIPES for k,c in MODELS for p in POLICIES]
    protocol=dict(scope='C100, same60 outer target development queries, extra40 training-only, frozen3x3 grouped splits; original201 masked.',
        seeds=list(parts),candidates=grid,
        input='Complete first-channel900 baseline-corrected values plus observed fraction; train-only standardization, RBFgamma1/p exact training kernel eigenfeatures. Hubei uses last-value completion with observed fraction and source indicator.',
        classifier='L2 logistic C1/10, three-class or binary abnormal plus separate low/high direction. Training-domain inverse-class or HB/target DenseWeight weights inherited from existing comparison. Returned scores not claimed clinically calibrated.',
        experts='Conditional MA is RBF-weighted mean of actual same-class training MA values; target records preferred, source fallback only for absent target class. Class absent from all training has zero probability and cannot be selected. No queried labels enter experts.',
        policies='Mixture mean, argmax, FNcost4/16, inner-selected abnormal threshold. Classification and MA metrics use same output; wrong tail errors retained.',
        selection='Inner OOF predictions determine threshold and overall model/policy. Outer results evaluate only. Strict goal flag unchanged; ranking never creates false positives solely for FNR<FPR.',
        extension='5seed screen extends if joint pass, or agreement>80,FNR<37%,low recall>=25%, or R2>.35.',
        input_hashes=hashes,code_sha256=sha256(__file__),selection_code_sha256=sha256(ROOT/'code_submit/ma_target_optimization/metrics.py'),test_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=parts[1000]['pool'];pos={int(v):j for j,v in enumerate(pool)}
    full=np.full((len(grid),seeds,len(pool)),np.nan);nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']}
    ledger=[];thresholds=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            a0,b=fold['A'],fold['B'];ad=extra[~np.isin(data['groups'][extra],data['groups'][b])];a=np.r_[a0,ad]
            assert not set(data['groups'][a]).intersection(data['groups'][b])
            ix=[pos[int(v)] for v in b];ip={int(v):j for j,v in enumerate(a0)}
            inner=np.full((len(grid),len(a0)),np.nan);number=0
            for recipe in RECIPES:
                pp,ee,diag=fit_predict(data,x,a,b,recipe)
                if si==fi==0:
                    poison=dict(data);poison['y']=data['y'].copy();poison['y'][b]=np.nan
                    changed=x.copy();changed[b]=1e4
                    _,_,d2=fit_predict(poison,changed,a,b,recipe);assert d2==diag
                iprob=np.full((len(MODELS),len(a0),3),np.nan);iexpert=np.full((len(a0),3),np.nan)
                for f in fold['inner']:
                    tr,va=f['T'],f['V'];add=ad[~np.isin(data['groups'][ad],data['groups'][va])];tr=np.r_[tr,add]
                    assert not set(data['groups'][tr]).intersection(data['groups'][va])
                    p0,e0,_=fit_predict(data,x,tr,va,recipe);ji=[ip[int(v)] for v in va]
                    iprob[:,ji]=p0;iexpert[ji]=e0
                for mi,(kind,C) in enumerate(MODELS):
                    threshold,tm=choose_threshold(data['y'][a0,2],iprob[mi],iexpert)
                    thresholds.append(dict(seed=seed,fold=fi,recipe=recipe,kind=kind,C=C,threshold=threshold,inner_metrics=tm))
                    for policy in POLICIES:
                        full[number,si,ix]=decode(pp[mi],ee,policy,threshold)
                        inner[number]=decode(iprob[mi],iexpert,policy,threshold);number+=1
            yy=data['y'][a0,2];mm=[detailed(yy,p) for p in inner]
            chosen={'mse':int(np.argmin(np.mean((inner-yy)**2,axis=1))),'safety':min(range(len(grid)),key=lambda j:(selection_key(mm[j]),j))}
            for key,j in chosen.items():nested[key][si,ix]=full[j,si,ix];ledger.append(dict(seed=seed,fold=fi,objective=key,candidate=grid[j],inner_metrics=mm[j]))
        print('STATE_SEED',seed,'seconds',round(time.time()-begin,1),flush=True)
    yy=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    records=[dict(candidate=c,metrics=detailed(yy,p)) for c,p in zip(grid,full)]
    selected=[dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()]
    for row,p in zip(records+selected,list(full)+list(nested.values())):
        a=states(yy);b=states(p)
        assert [[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]==row['metrics']['confusion']
        assert abs(row['metrics']['r2']-(1-np.sum((p-yy)**2)/np.sum((yy-yy.mean())**2)))<1e-12
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=full,reference=yy[0],pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    (out/'private/THRESHOLDS.json').write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
    promising=[z for z in records if z['metrics']['mapping_target_pass'] or z['metrics']['r2']>.35 or (z['metrics']['agreement']>.8 and z['metrics']['fnr']<.37 and z['metrics']['low_recall']>=.25)]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=records,nested=selected,promising=promising,
        checks=dict(query_labels_inputs_do_not_fit=True,independent_metrics_verified=True,input_hashes_unchanged=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='full_curve_state_experts_screen_20261102');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output)
