"""Two-stage state diagnostics and train-only functional PCA.

The numerical outputs are training-class means. This is explicitly a test of
state separability, not evidence of improved precise continuous MA mapping.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256
from .metrics import states,assess
from .search import ROOT,make_summary


def load_waveforms(data,partitions):
    paths=[ROOT/'outputs/uteg_cohort399_20261006_201800/private/data.npz',ROOT/'outputs/uteg_platform_filter_pilot_20261006_124636/private/CUT_5.npz']
    with np.load(paths[0],allow_pickle=False) as z:
        original=z['original_row_index'];assert np.array_equal(z['ids'],data['ids'][:399])
    with np.load(paths[1],allow_pickle=False) as z:
        signal=z['signal'];observed=z['observed']
    pool=next(iter(partitions.values()))['pool'];raw=np.full((len(data['x']),90),np.nan)
    for i in pool:
        if i>=399:raise ValueError('This target-only loader is restricted to SAHZU')
        s=signal[original[i],:900];obs=observed[original[i],:900]
        for j in range(90):
            valid=obs[j*10:(j+1)*10];v=s[j*10:(j+1)*10][valid]
            if len(v):raw[i,j]=np.median(v)
    amplitude=np.nanmax(raw[pool],axis=1)-np.nanmin(raw[pool],axis=1)
    shape=raw.copy();shape[pool]=raw[pool]/np.maximum(amplitude[:,None],1e-6)
    return raw,shape,{str(p):sha256(p) for p in paths}


def encode(train,test,components):
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    a=imputer.fit_transform(train);b=imputer.transform(test)
    # Equal physical amplitude scale at each time; PCA centers on training only.
    if components:
        pca=PCA(n_components=min(components,len(a)-1,a.shape[1]),svd_solver='full')
        a=pca.fit_transform(a);b=pca.transform(b)
    scaler=StandardScaler();return scaler.fit_transform(a),scaler.transform(b)


def fit_predict(xtrain,ytrain,xeval,c):
    a,b=encode(xtrain,xeval,c['components']);label=states(ytrain);abnormal=label!=1
    if len(np.unique(abnormal))<2:raise ValueError('Both binary classes required')
    w=np.where(abnormal,c['cost'],1.);w/=w.mean()
    if c['learner']=='logistic':
        detector=LogisticRegression(C=c['C'],max_iter=3000).fit(a,abnormal,sample_weight=w)
        score=detector.predict_proba(b)[:,1]
    elif c['learner']=='svm':
        detector=SVC(C=c['C'],kernel='rbf',gamma='scale').fit(a,abnormal,sample_weight=w)
        score=expit(detector.decision_function(b))
    elif c['learner']=='lda':
        priors=np.array([np.sum(~abnormal),np.sum(abnormal)*c['cost']],float);priors/=priors.sum()
        detector=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=priors).fit(a,abnormal)
        score=detector.predict_proba(b)[:,1]
    else:raise ValueError(c['learner'])
    abnormal_label=label[abnormal]
    if len(np.unique(abnormal_label))==1:
        direction=np.full(len(b),abnormal_label[0])
    else:
        branch=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000).fit(a[abnormal],abnormal_label)
        direction=branch.predict(b).astype(int)
    means=np.array([ytrain[label==i].mean() if np.any(label==i) else np.nan for i in range(3)])
    return score,direction,means


def grid():
    ans=[]
    for rep,pc in [('parameters4',0),('wave90',4),('wave90',8),('shape90',4),('shape90',8),('combined94',8)]:
        for learner,regularization in [('logistic',.1),('logistic',1.),('svm',1.),('svm',10.),('lda',1.)]:
            for cost in (1.,4.):
                c=dict(family='two_stage_class_mean',representation=rep,components=pc,learner=learner,C=regularization,cost=cost)
                c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12];ans.append(c)
    return ans


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_functional_screen_v1';out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1005))
    data,partitions,hashes=load_development_inputs(ROOT,cfg);pool=next(iter(partitions.values()))['pool'];test=next(iter(partitions.values()))['test']
    raw,shape,more=load_waveforms(data,partitions);hashes.update(more);data['x'][test]=np.nan
    matrices={'parameters4':data['x'],'wave90':raw,'shape90':shape,'combined94':np.c_[data['x'],raw]}
    thresholds=[.05,.1,.2,.35,.5,.65,.8,.9];candidates=grid();pos={int(v):i for i,v in enumerate(pool)}
    protocol={'seeds':cfg['seeds'],'candidates':candidates,'thresholds':thresholds,'input_hashes':hashes,'code_hash':sha256(__file__),
              'scope':'development-only threshold/candidate selection; not unbiased validation',
              'representation':'90 time-bin medians from observed 15-minute curve; imputation/PCA/scaling fit only on each training fold',
              'score':'probability for logistic/LDA; sigmoid margin for SVM, not calibrated clinical probability',
              'numerical_output':'training class mean; diagnostic only, no continuous mapping claim',
              'goals':'all-report agreement>80%, FNR<5%, FNR<FPR; trust gate coverage>75% separately'}
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');path=out/'CANDIDATES.jsonl'
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [];done={r['candidate'].get('base_id') for r in records}
    for number,c in enumerate(candidates):
        if c['id'] in done:continue
        start=time.time()
        try:
            scores=[];directions=[];class_means=[]
            for seed,part in partitions.items():
                score=np.full(len(pool),np.nan);direction=np.full(len(pool),-1);means=np.full((len(pool),3),np.nan)
                for fold in part['folds']:
                    A,B=fold['A'],fold['B'];ix=[pos[int(i)] for i in B]
                    s,d,m=fit_predict(matrices[c['representation']][A],data['y'][A,2],matrices[c['representation']][B],c)
                    score[ix]=s;direction[ix]=d;means[ix]=m
                scores.append(score);directions.append(direction);class_means.append(means)
            scores=np.asarray(scores);directions=np.asarray(directions);class_means=np.asarray(class_means);y=data['y'][pool,2]
            np.savez_compressed(out/'private'/f"{c['id']}.npz",score=scores,direction=directions,class_means=class_means,reference=y,pool=pool,seeds=cfg['seeds'])
            for threshold in thresholds:
                selected=np.where(scores>=threshold,directions,1)
                predictions=np.take_along_axis(class_means,selected[...,None],axis=-1)[...,0]
                candidate={**c,'base_id':c['id'],'id':c['id']+'_'+str(threshold),'threshold':threshold}
                row={'candidate':candidate,'status':'complete','metrics':assess(np.broadcast_to(y,predictions.shape),predictions),'seconds':time.time()-start}
                records.append(row)
                with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        except Exception as error:
            row={'candidate':{**c,'base_id':c['id']},'status':'failed','error':str(error)};records.append(row)
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        if (number+1)%10==0 or number+1==len(candidates):
            s=make_summary(records,out);print(json.dumps({'base_models':number+1,'threshold_comparisons':s['completed'],'failed':s['failed'],'feasible':s['feasible_count']}),flush=True)
    return make_summary(records,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
