"""Training-only inverse-frequency/density losses on fixed common300 mapping.

These are imbalance importance weights, not identified causal propensity
weights. Source/target sample counts, feature transform, alpha and MA boundaries
are fixed; only training loss weights vary. No classifier-to-MA conversion.
"""
import argparse
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .cost_sensitive_detection import assemble
from .models import Features, weights
from .voltage_scale import PooledVoltage, load_inputs
from .search import ROOT, training_view
from .hubei_source_search import add_hubei_arm
from .metrics import assess
from code_submit.ma_imbalance_mapping.models import _ma_weights, _silverman_bandwidth, _normalise_and_cap_weights
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


METHODS=('uniform','class_sqrt','class_inverse','kde_sqrt_inverse','kde_inverse','lds_sqrt','denseweight')


def label_weights(y, method, cap):
    y=np.asarray(y,float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('Finite training MA required')
    if method=='class_inverse':
        w=weights(y,cap);info={'formula':'inverse class count with ratio cap'}
    elif method=='uniform':
        w=np.ones(len(y));info={'formula':'uniform'}
    elif method in ('kde_sqrt_inverse','kde_inverse'):
        bandwidth=_silverman_bandwidth(y)
        distance=(y[:,None]-y[None,:])/bandwidth
        density=np.exp(-.5*distance**2).mean(1)/(np.sqrt(2*np.pi)*bandwidth)
        gamma=.5 if method=='kde_sqrt_inverse' else 1.
        w,info=_normalise_and_cap_weights(density**(-gamma),cap)
        info.update(formula='Gaussian KDE density^(-gamma), training labels only',gamma=gamma,bandwidth=bandwidth)
    else:
        definitions={'class_sqrt':('class_weighted',{'gamma':.5}),
                     'lds_sqrt':('lds',{'sigma':2.,'bin_width':1.,'kernel_size':5}),
                     'denseweight':('denseweight',{'alpha':1.})}
        if method not in definitions:raise ValueError('Unknown weighting method')
        name,parameter=definitions[method];w,info=_ma_weights(y,name,parameter,cap)
    state=np.where(y<50,0,np.where(y>70,2,1))
    info.update(method=method,cap=cap,mean=float(w.mean()),ratio=float(w.max()/w.min()),
                effective_sample_size=float(w.sum()**2/(w@w)),
                class_counts=[int(np.sum(state==k)) for k in range(3)],
                class_loss_mass=[float(w[state==k].sum()/w.sum()) for k in range(3)])
    return w,info


def fit(train,c):
    x,y,codes,blocks=assemble(train,c['arm'])
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite training inputs required')
    w=np.empty(len(y));diagnostics=[]
    for block in np.unique(blocks):
        keep=blocks==block
        source=c['arm']!='target_only' and block==0
        local,info=label_weights(y[keep],c['weight_method'],5. if source else 20.)
        w[keep]=local/keep.sum()
        diagnostics.append({'block':int(block),'role':'source' if source else 'target',**info})
    w/=w.mean()
    transform=Features.fit(x,'linear')
    design=np.c_[transform.transform(x),codes,np.ones(len(x))]
    penalty=np.diag(np.r_[np.ones(design.shape[1]-1),0.])
    lhs=design.T@(w[:,None]*design)+len(y)*.01*penalty
    coef=np.linalg.solve(lhs+1e-12*np.eye(len(lhs)),design.T@(w*(y-60)/10))
    for info in diagnostics:info['total_loss_mass']=float(w[blocks==info['block']].sum()/w.sum())
    return PooledVoltage(transform,coef,codes.shape[1],{'weights':diagnostics,'alpha':.01,
        'training_n':len(y),'target_n':len(train['x']),'source_n':len(y)-len(train['x']),
        'coef_hash':hashlib.sha256(coef.tobytes()).hexdigest()})


def grid():
    result=[]
    for arm in ('target_only','HB_to_SA'):
        for method in METHODS:
            c={'family':'density_reweighted_voltage_pool','arm':arm,'weight_method':method,'alpha':.01,
               'horizon':300,'voltage_mode':'p5','source_ratio_cap':5.,'target_ratio_cap':20.}
            c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12];result.append(c)
    return result


def run(out,seeds,ids=None):
    out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds']=list(range(1000,1000+seeds))
    data,parts,hashes=load_development_inputs(ROOT,cfg);add_hubei_arm(data,parts)
    matrices,audit,more=load_inputs(data,parts);hashes.update(more);data['x']=matrices[(300,'p5')]
    candidates=grid()
    if ids is not None:candidates=[c for c in candidates if c['id'] in ids];assert len(candidates)==len(ids)
    protocol={'candidates':candidates,'seeds':cfg['seeds'],'input_hashes':hashes,
        'code_hashes':{str(p):sha256(p) for p in [__file__,ROOT/'code_submit/ma_imbalance_mapping/models.py',ROOT/'code_submit/ma_target_optimization/voltage_scale.py',ROOT/'code_submit/ma_target_optimization/models.py',ROOT/'code_submit/ma_target_optimization/cost_sensitive_detection.py']},
        'scope':'same60 target records grouped outer folds, only approved79 Hubei when pooled; original201 test features/labels excluded',
        'hypothesis':'whether continuous-label weighting improves the current best fixed18-feature linear mapping over existing inverse3class weighting',
        'weights':'each center computed from its own training MA only; mean1 within block and equal total source/target loss mass; max/min ratio limited to5 source and20 target',
        'kde':'Gaussian evaluated at training labels, h=1.06*std*n^(-.2), fallback1 for constant labels; inverse and inverse-square-root density; study adaptation of smoothed-label weighting, not causal IPW',
        'lds':'existing audited author-style sqrt(count) then Gaussian ks5 sigma2 on1mm bins then reciprocal; training-only bins',
        'denseweight':'existing audited Gaussian-KDE DenseWeight alpha1, not itself reciprocal density',
        'controls':'ordinary and existing inverse3class weighted continuous mapping; exact replay required',
        'fixed':'same feature scaling, source indicator, alpha.01, no threshold search or gate; numerical MA evaluated at unchanged50/70 boundaries',
        'acceptance':'all-report3class agreement>.8, abnormal binaryFNR<.05 andFNR<FPR; post-gate coverage>.75 remains a separate unverified requirement',
        'limit':'adaptive development screening, not untouched external validation; seeds do not add independent cases',
        'references':['https://proceedings.mlr.press/v139/yang21m.html','https://doi.org/10.1007/s10994-021-06023-5']}
    text=json.dumps(protocol,indent=2,allow_nan=False)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text()!=text:raise RuntimeError('Changed frozen protocol')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');(out/'INPUT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    path=out/'CANDIDATES.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    pool=next(iter(parts.values()))['pool'];position={int(v):i for i,v in enumerate(pool)}
    for c in candidates:
        if any(r['candidate']['id']==c['id'] for r in rows):continue
        start=time.time();predictions=[];fold_audits=[]
        for seed,part in parts.items():
            pred=np.full(len(pool),np.nan)
            for number,fold in enumerate(part['folds']):
                assert not set(data['groups'][fold['A']])&set(data['groups'][fold['B']])
                model=fit(training_view(data,fold['A']),c)
                pred[[position[int(i)] for i in fold['B']]]=model.predict(data['x'][fold['B']])
                fold_audits.append({'seed':seed,'fold':number,**model.diagnostics})
            assert np.isfinite(pred).all();predictions.append(pred)
        pred=np.asarray(predictions);y=data['y'][pool,2]
        np.savez_compressed(out/'private'/f"{c['id']}.npz",pred=pred,reference=y,pool=pool,seeds=cfg['seeds'])
        m=assess(np.broadcast_to(y,pred.shape).ravel(),pred.ravel())
        row={'candidate':c,'status':'complete','metrics':m,'continuous_MA_evaluated':True,'gate_evaluated':False}
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);(out/'private'/f"{c['id']}_audit.json").write_text(json.dumps(fold_audits,indent=2),encoding='utf-8')
        print(json.dumps({'method':c['weight_method'],'arm':c['arm'],'seconds':round(time.time()-start,2),**{k:m[k] for k in ('agreement','fnr','fpr','mae','low_recall','high_recall')}}),flush=True)
    summary={'complete':len(rows),'mapping_joint_passes':sum(r['metrics']['mapping_target_pass'] for r in rows),
             'accuracy_qualified':sorted([r for r in rows if r['metrics']['agreement']>.8],key=lambda r:r['metrics']['fnr']),
             'test_evaluated':False,'gate_evaluated':False}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='goal_density_reweight_v1');p.add_argument('--ids',nargs='*');a=p.parse_args()
    with threadpool_limits(limits=1):run(ROOT/'outputs/uteg_mapping_target_runs'/a.output,a.seeds,a.ids)
