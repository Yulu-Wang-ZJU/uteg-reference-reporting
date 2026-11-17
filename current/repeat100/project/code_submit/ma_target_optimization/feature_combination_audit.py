"""Single-feature, every-pair and semantic-group U-TEG mapping audit.

All feature extraction is per-record and reference-free. Nested group folds
select combinations without looking at outer query labels. Discovery rankings
of fixed combinations are kept separate from nested selection performance.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .asymmetric_ma_mapping import solve as solve_asymmetric
from .common_clock import load_preprocessor
from .density_reweight_mapping import label_weights
from .hubei_source_search import add_hubei_arm
from .learning_curve import load_extra
from .metrics import assess, selection_key
from .search import ROOT
from .voltage_scale import recover_hubei_baselines
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


GROUPS={
    'amplitude':['raw_peak','smooth_peak','endpoint','amp_60','amp_180','amp_300','amp_600','amp_900','area_mean','q95','baseline'],
    'timing':['t10','t50','t90','rise10_90','time_peak'],
    'growth':['maximum_30s_growth','time_maximum_growth','growth_early','growth_middle','growth_late'],
    'shape':['relative_area','fraction_300','fraction_600','end_peak_ratio','monotone_fraction','noise_relative','decline_relative','observed_fraction'],
    'fitted4':['fit_t0','fit_amplitude','fit_k','fit_nu']}
NAMES=sum(GROUPS.values(),[])
RECIPES=['target_uniform','target_class','HB_dense']


def extract(signal,observed,fitted,preprocess):
    n=int(np.asarray(observed,bool)[:900].sum())
    if n<11 or not np.array_equal(np.flatnonzero(observed[:900]),np.arange(n)):
        raise ValueError('Invalid observed prefix')
    raw=np.asarray(signal,float)[:n]
    if not np.isfinite(raw).all():raise ValueError('Invalid voltage')
    baseline=float(np.median(raw[:10]));clean,_=preprocess(raw,q=None)
    padded=np.pad(clean,(0,900-n),mode='edge');peak=float(clean.max());den=max(abs(peak),1e-6)
    def level(end):return float(np.median(padded[max(0,end-30):end]))
    def slope(a,b):
        tt=np.arange(a,b)/60.;vv=padded[a:b]
        return float(np.dot(tt-tt.mean(),vv-vv.mean())/np.dot(tt-tt.mean(),tt-tt.mean()))
    levels={t:level(t) for t in [60,180,300,600,900]}
    times=[]
    for fraction in [.1,.5,.9]:
        hit=np.flatnonzero(np.convolve((padded>=fraction*peak).astype(int),np.ones(5,int),mode='valid')==5)
        times.append(float(hit[0]) if len(hit) and peak>1e-6 else 900.)
    growth=(padded[30:]-padded[:-30])*2
    j=int(np.argmax(growth))
    end=levels[900];area=float(np.trapz(padded,dx=1)/899)
    values=[float(raw.max()-baseline),peak,end,*[levels[t] for t in [60,180,300,600,900]],area,float(np.quantile(clean,.95)),baseline,
            *times,times[2]-times[0],float(np.argmax(padded)),
            float(growth[j]),float(j+15),slope(0,300),slope(300,600),slope(600,900),
            area/den,levels[300]/den,levels[600]/den,end/den,float(np.mean(np.diff(clean)>=0)),
            float(np.sqrt(np.mean(((raw-baseline)-clean)**2)))/den,(peak-end)/den,n/900.,*np.asarray(fitted,float)]
    x=np.asarray(values,float)
    if len(x)!=len(NAMES) or not np.isfinite(x).all():raise ValueError('Feature extraction failed')
    return x


def subsets():
    found={}
    def add(ix,kind):
        key=tuple(sorted(set(ix)))
        if key not in found:found[key]=dict(columns=list(key),names=[NAMES[i] for i in key],categories=[])
        if kind not in found[key]['categories']:found[key]['categories'].append(kind)
    for i in range(len(NAMES)):add([i],'single')
    for pair in itertools.combinations(range(len(NAMES)),2):add(pair,'pair')
    fit=[NAMES.index(n) for n in GROUPS['fitted4']]
    for count in range(1,5):
        for combination in itertools.combinations(fit,count):add(combination,'fitted4_subset')
    names=list(GROUPS)
    for bits in range(1,2**len(names)):
        groups=[names[j] for j in range(len(names)) if bits&(1<<j)]
        add([NAMES.index(n) for g in groups for n in GROUPS[g]],'groups:'+','.join(groups))
    answer=[]
    for c in found.values():
        c['id']=hashlib.sha256(json.dumps(c['names']).encode()).hexdigest()[:12];answer.append(c)
    return answer


def assemble(data,x,target_rows,recipe):
    if recipe not in RECIPES:raise ValueError(recipe)
    source=data['source_rows']['HB_to_SA'] if recipe=='HB_dense' else np.array([],int)
    rows=np.r_[source,target_rows]
    values=x[rows];y=data['y'][rows,2]
    if not np.isfinite(values).all() or not np.isfinite(y).all():raise ValueError('Training only finite arrays')
    method={'target_uniform':'uniform','target_class':'class_inverse','HB_dense':'denseweight'}[recipe]
    w=np.empty(len(rows));nt=len(target_rows);ns=len(source)
    w[ns:]=label_weights(y[ns:],method,20.)[0]/nt
    if ns:w[:ns]=label_weights(y[:ns],method,5.)[0]/ns
    w/=w.mean()
    mean=values.mean(0);scale=np.maximum(values.std(0),1e-6)
    z=(values-mean)/scale
    if ns:z=np.c_[z,np.r_[np.ones(ns),np.zeros(nt)]]
    z=np.c_[z,np.ones(len(z))]
    return z,y,w,mean,scale,ns


def all_linear_predictions(data,x,a,q,recipe,candidates):
    z,y,w,mean,scale,ns=assemble(data,x,a,recipe)
    query=(x[q]-mean)/scale
    if ns:query=np.c_[query,np.zeros(len(q))]
    query=np.c_[query,np.ones(len(q))]
    gram=z.T@(w[:,None]*z);rhs=z.T@(w*(y-60)/10)
    extra=list(range(x.shape[1],z.shape[1]))
    batches={}
    for number,c in enumerate(candidates):batches.setdefault(len(c['columns']),[]).append(number)
    result=np.empty((len(candidates),len(q)))
    for numbers in batches.values():
        indices=np.array([candidates[j]['columns']+extra for j in numbers])
        lhs=gram[indices[:,:,None],indices[:,None,:]].copy();b=rhs[indices]
        penalty=np.r_[np.ones(indices.shape[1]-1)*len(y)*.01,0.]+1e-12
        lhs+=np.diag(penalty)[None,:,:]
        coefficients=np.linalg.solve(lhs,b[...,None])[...,0]
        result[numbers]=np.maximum(60+10*np.einsum('qmk,mk->mq',query[:,indices],coefficients),0)
    return result


def detailed(y,p):
    m=assess(y,p)
    fp,fn=m['false_positive_count'],m['false_negative_count']
    m['binary_confusion_rows_normal_abnormal']=[[m['within_n']-fp,fp],[fn,m['low_n']+m['high_n']-fn]]
    m['r2']=float(1-np.sum((np.asarray(y)-p)**2)/np.sum((np.asarray(y)-np.mean(y))**2))
    return m


def choose(y,predictions,candidates,objective):
    # Every metric here uses inner heldout predictions only. These labels never
    # train the models which produced the corresponding predictions.
    metrics=[assess(y,p) for p in predictions]
    if objective=='mae':key=lambda j:(metrics[j]['mae'],len(candidates[j]['columns']),j)
    elif objective=='safety':
        # Add opposite-tail error as an explicit tie-breaker before FPR.
        def key(j):
            m=metrics[j];base=selection_key(m)
            if base[0]<=1:return (base[0],m['fnr'],m['opposite_abnormal_count'],m['fpr'],m['macro_mae'],len(candidates[j]['columns']),j)
            return (2,-m['agreement'],m['fnr'],m['opposite_abnormal_count'],m['macro_mae'],len(candidates[j]['columns']),j)
    else:raise ValueError(objective)
    selected=min(range(len(candidates)),key=key)
    return selected,metrics[selected]


def special_prediction(data,x,a,b,recipe,columns,kind):
    xx=x[:,columns];z,y,w,mean,scale,ns=assemble(data,xx,a,recipe)
    query=(xx[b]-mean)/scale
    if ns:query=np.c_[query,np.zeros(len(b))]
    query=np.c_[query,np.ones(len(b))]
    if kind=='asymmetric4':
        penalty=np.diag(np.r_[np.ones(z.shape[1]-1),0.])
        initial=np.linalg.solve(z.T@(w[:,None]*z)+len(y)*.01*penalty+1e-12*np.eye(z.shape[1]),z.T@(w*(y-60)/10))
        target=np.arange(len(y))>=ns
        coefficient,info=solve_asymmetric(z,(y-60)/10,w,target,'residual',4.,initial)
        return np.maximum(60+10*query@coefficient,0),info
    if kind=='hgb':
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=5,
                  learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=20261101)
        model.fit(z[:,:-1],y,sample_weight=w)
        return np.maximum(model.predict(query[:,:-1]),0),dict(iterations=100)
    raise ValueError(kind)


def outer(data,x,fold,extra,recipe,candidates,special=True):
    a0,b=fold['A'],fold['B'];outer_extra=extra[~np.isin(data['groups'][extra],data['groups'][b])]
    a=np.r_[a0,outer_extra];assert not set(data['groups'][a]).intersection(data['groups'][b])
    ordinary=all_linear_predictions(data,x,a,b,recipe,candidates)
    pos={int(v):i for i,v in enumerate(a0)}
    inner=np.full((len(candidates),len(a0)),np.nan)
    for f in fold['inner']:
        t,v=f['T'],f['V'];add=outer_extra[~np.isin(data['groups'][outer_extra],data['groups'][v])]
        t=np.r_[t,add];assert not set(data['groups'][t]).intersection(data['groups'][v])
        inner[:,[pos[int(i)] for i in v]]=all_linear_predictions(data,x,t,v,recipe,candidates)
    assert np.isfinite(inner).all()
    output={};selection=[]
    for objective in ['mae','safety']:
        index,m=choose(data['y'][a0,2],inner,candidates,objective)
        candidate=candidates[index]
        output[objective+'_ridge']=ordinary[index]
        info=dict(objective=objective,subset=candidate,inner_metrics=m,query_rows_sha256=hashlib.sha256(b.tobytes()).hexdigest())
        if special:
            # Extra estimators are evaluated on the same inner-selected feature
            # subset. Their outputs remain separate; never select on outer B.
            for kind in ['asymmetric4','hgb']:
                p,diagnostic=special_prediction(data,x,a,b,recipe,candidate['columns'],kind)
                output[objective+'_'+kind]=p
                info[kind]=diagnostic
        selection.append(info)
    return ordinary,output,selection


def load(seeds):
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds']=list(range(1000,1000+seeds))
    data,parts,hashes=load_development_inputs(ROOT,cfg)
    fitted=data['x'].copy();extra,more=load_extra(data,parts,cfg);hashes.update(more);add_hubei_arm(data,parts)
    rows=data['allowed_training_indices'];test=parts[1000]['test']
    path=ROOT/'outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz'
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(saved['ids'],data['ids'])
        raw,observed=saved['original'][rows],saved['observed'][rows]
    offsets,_,more=recover_hubei_baselines(data,rows,raw,observed);hashes.update(more)
    preprocess,prepath=load_preprocessor();hashes.update({str(p):sha256(p) for p in [path,prepath]})
    x=np.full((len(data['x']),len(NAMES)),np.nan)
    for row,signal,mask,offset in zip(rows,raw,observed,offsets):
        x[row]=extract(signal+offset,mask,fitted[row],preprocess)
    assert np.isnan(x[test]).all() and np.isnan(data['y'][test]).all()
    return data,parts,extra,x,hashes


def run(name,seeds=20,budget=60):
    out=ROOT/'outputs/uteg_mapping_target_runs'/name;(out/'private').mkdir(parents=True,exist_ok=True)
    data,parts,extra,x,hashes=load(seeds);extra=extra if budget==100 else np.array([],int)
    candidates=subsets();pool=parts[1000]['pool'];pos={int(v):i for i,v in enumerate(pool)}
    protocol=dict(features=NAMES,groups=GROUPS,feature_count=len(NAMES),subsets=candidates,
                  recipes=RECIPES,seeds=list(parts),budget=budget,input_hashes=hashes,code_hash=sha256(__file__),
                  objective='Identify additional UTEG-only information beyond maximum amplitude; continuous MA regression and abnormal direction evaluated together.',
                  scope='Same60 grouped target development queries; C100 adds historical40 only within actual training groups. Source79Hubei only in HB_dense. Original201 test masked.',
                  feature_source='33 explicit raw-first-channel readouts including4 existing frozen EXP parameters(t0,amplitude,k,nu). Prefix900, same existing SG11/order2, no P5 cutoff. Other sensor columns and referenceTEG readouts are never predictors.',
                  feature_definitions='Raw maximum and raw baseline use first10 median. Smoothed endpoint/checkpoints use trailing30-point medians. Area is trapezoidal mean over900 samples. Fractional times require5 successive points above10/50/90% smoothed peak. Maximum growth uses30-second difference in V/min. Early/middle/late slopes use0:300/300:600/600:900. Shape ratios use peak floor1e-6. Short curves last-value padded; observed fraction retained.',
                  subsets_policy='Every single parameter and every two-parameter combination; all15 nonempty subsets of original4 fitted parameters; all31 nonempty combinations of5 semantic groups, deduplicated. Not every possible high-order subset of33 variables.',
                  ridge='alpha.01 with intercept; standardization and label weighting training-only. Target uniform/class-inverse; pooledHB DenseWeight sourcecap5/targetcap20, equal base domain mass, source intercept.',
                  selection='Nested frozen inner group folds select separately by MAE and accuracy-qualified lowFNR, breaking ties using opposite-tail errors. Outer query labels used only after predictions. Full fixed-subset rankings remain exploratory discovery results.',
                  special='On each inner-selected subset, also evaluate target-inward asymmetric residualcost4 and fixed HGB100trees/7leaves/minleaf5/lr.05/l2=1/noearlystop. Outcomes kept separate; no outer selection of winner.',
                  control='Parameter-only and peak-only are explicit subset controls. Existing broader curve/CNN/multitask experiments were reviewed, not misrepresented as untried.',
                  interpretation='Development research; repeated folds do not add patients and historical development reuse precludes untouched validation claims.',
                  test_evaluated=False,gate_evaluated=False,manuscript_changed=False,
                  model_reference='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html')
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists():assert (out/'PROTOCOL.json').read_text(encoding='utf-8')==text
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8')
    np.savez_compressed(out/'private/FEATURES.npz',x=x[pool],reference=data['y'][pool,2],pool=pool,names=NAMES)
    fixed_rows=[];nested_rows=[];selections=[]
    for recipe in RECIPES:
        full=np.full((len(candidates),seeds,len(pool)),np.nan);nested={}
        start=time.time()
        for si,(seed,part) in enumerate(parts.items()):
            for number,fold in enumerate(part['folds']):
                predictions,adaptive,chosen=outer(data,x,fold,extra,recipe,candidates)
                ix=[pos[int(v)] for v in fold['B']];full[:,si,ix]=predictions
                for key,p in adaptive.items():
                    if key not in nested:nested[key]=np.full((seeds,len(pool)),np.nan)
                    nested[key][si,ix]=p
                selections.extend([dict(recipe=recipe,seed=seed,fold=number,**item) for item in chosen])
            print(json.dumps(dict(recipe=recipe,seed=seed,seconds=round(time.time()-start,1),subsets=len(candidates))),flush=True)
        assert np.isfinite(full).all()
        y=data['y'][pool,2];reference=np.broadcast_to(y,full.shape[1:])
        for i,c in enumerate(candidates):
            fixed_rows.append(dict(recipe=recipe,subset=c,metrics=detailed(reference,full[i]),interpretation='fixed-subset discovery; ranking not independently validated'))
        for key,p in nested.items():
            assert np.isfinite(p).all()
            nested_rows.append(dict(recipe=recipe,procedure=key,metrics=detailed(reference,p),nested_feature_selection=True))
        np.savez_compressed(out/'private'/f'{recipe}_predictions.npz',fixed=full,reference=y,pool=pool,seeds=list(parts),**nested)
        (out/'FIXED_SUBSETS.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in fixed_rows),encoding='utf-8')
        (out/'NESTED_RESULTS.json').write_text(json.dumps(nested_rows,indent=2,allow_nan=False),encoding='utf-8')
        (out/'private/SELECTIONS.json').write_text(json.dumps(selections,indent=2,allow_nan=False),encoding='utf-8')
    for path,digest in hashes.items():assert sha256(path)==digest
    summary=dict(feature_count=len(NAMES),subset_count=len(candidates),fixed_results=len(fixed_rows),nested_procedures=len(nested_rows),
                 fixed_joint_passes=sum(r['metrics']['mapping_target_pass'] for r in fixed_rows),nested_joint_passes=sum(r['metrics']['mapping_target_pass'] for r in nested_rows),
                 best_fixed_mae=sorted(fixed_rows,key=lambda r:r['metrics']['mae'])[:8],
                 best_fixed_safety=sorted(fixed_rows,key=lambda r:selection_key(r['metrics']))[:8],
                 nested=nested_rows,input_hashes_unchanged=True,test_evaluated=False,gate_evaluated=False)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='goal_feature_combinations_C60_v1');p.add_argument('--seeds',type=int,default=20);p.add_argument('--budget',type=int,default=60,choices=[60,100]);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.seeds,args.budget)
