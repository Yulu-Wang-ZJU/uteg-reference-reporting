"""Grouped nested mapping from frozen time-series representations to MA."""
import argparse
import hashlib
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .frozen_moment_features import OUTPUT as FEATURE_DIR
from .feature_combination_audit import load,NAMES,all_linear_predictions,detailed
from .current_low_support import load_current_representations
from .full_curve_cost_sensitive import solve
from .density_reweight_mapping import label_weights
from .metrics import states,selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

REPRESENTATIONS=['embedding','embedding_amp6','embedding_current18','random_amp6','amp6']
RECIPES=['target_uniform','target_class','HB_dense']
ALPHAS=[.001,.01,.1]
LOSSES=[('control',1.),('boundary',4.),('boundary',16.)]
GRID=[dict(representation=r,recipe=w,basis=b,alpha=a,loss=l,cost=c)
      for r,w,b,a,(l,c) in itertools.product(REPRESENTATIONS,RECIPES,['linear','rbf'],ALPHAS,LOSSES)]


def load_inputs(seeds,larger=False):
    data,parts,extra,x,hashes=load(seeds)
    current,more,_=load_current_representations(data,parts);hashes.update(more)
    path=FEATURE_DIR/'private/FEATURES.npz';z=np.load(path)
    assert np.array_equal(z['allowed'],data['allowed_training_indices'])
    assert np.array_equal(z['pool'],parts[1000]['pool'])
    columns=[NAMES.index(k) for k in ['raw_peak','amp_60','amp_300','amp_600','endpoint','area_mean']]
    amp=x[:,columns];emb=z['pretrained'];random=z['random'];cur=current['current18']
    blocks={'embedding':[emb],'embedding_amp6':[emb,amp],
            'embedding_current18':[emb,cur],'random_amp6':[random,amp],'amp6':[amp]}
    if larger:
        base_dir=ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_base_features_20261102'
        bz=np.load(base_dir/'private/FEATURES.npz')
        assert np.array_equal(bz['allowed'],data['allowed_training_indices'])
        assert np.array_equal(bz['pool'],parts[1000]['pool'])
        blocks.update(base_amp6=[bz['pretrained'],amp],base_current18=[bz['pretrained'],cur],
                      base_random_amp6=[bz['random'],amp])
        for p in [base_dir/'private/FEATURES.npz',base_dir/'PROTOCOL.json',base_dir/'VERIFICATION.json']:
            hashes[str(p)]=sha256(p)
    for arrays in blocks.values():
        for v in arrays:assert np.isnan(v[parts[1000]['test']]).all()
    for p in [path,FEATURE_DIR/'PROTOCOL.json',FEATURE_DIR/'VERIFICATION.json']:hashes[str(p)]=sha256(p)
    return data,parts,extra,blocks,cur,hashes


def design(data,blocks,a,b,recipe,basis):
    source=data['source_rows']['HB_to_SA'] if recipe=='HB_dense' else np.array([],int)
    tr=np.r_[source,a];ns=len(source)
    assert not set(data['groups'][tr]).intersection(data['groups'][b])
    y=data['y'][tr,2]
    method={'target_uniform':'uniform','target_class':'class_inverse','HB_dense':'denseweight'}[recipe]
    w=np.ones(len(tr));w[ns:]=label_weights(y[ns:],method,20.)[0]/len(a)
    if ns:w[:ns]=label_weights(y[:ns],method,5.)[0]/ns
    w/=w.mean()
    train,query=[],[];stats=[]
    for x in blocks:
        mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),1e-6)
        train.append((x[tr]-mean)/scale/np.sqrt(x.shape[1]))
        query.append((x[b]-mean)/scale/np.sqrt(x.shape[1]))
        stats.append(hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest())
    # Equal average squared norm per block prevents1539 embeddings from
    # drowning out the6 raw amplitude features or the18 existing features.
    tx=np.concatenate(train,axis=1)/np.sqrt(len(blocks))
    qx=np.concatenate(query,axis=1)/np.sqrt(len(blocks))
    K=tx@tx.T;Q=qx@tx.T
    if basis=='rbf':
        norms=np.sum(tx*tx,axis=1)
        K=np.exp(-np.maximum(norms[:,None]+norms[None,:]-2*K,0))
        Q=np.exp(-np.maximum(np.sum(qx*qx,axis=1)[:,None]+norms[None,:]-2*Q,0))
    elif basis!='linear':raise ValueError(basis)
    values,vectors=np.linalg.eigh((K+K.T)/2)
    keep=values>max(1e-12,float(values.max())*1e-12)
    root=np.sqrt(values[keep]);v=vectors[:,keep]
    z,zq=v*root,Q@(v/root)
    if ns:
        z=np.c_[z,np.r_[np.ones(ns),np.zeros(len(a))]];zq=np.c_[zq,np.zeros(len(b))]
    z,zq=np.c_[z,np.ones(len(tr))],np.c_[zq,np.ones(len(b))]
    return z,zq,y,w,dict(training_scaling_sha256=stats,n_source=ns,n_target=len(a))


def predict_all(data,blocks,cur,a,b):
    predictions=[];audit=[]
    for rep in REPRESENTATIONS:
        for recipe in RECIPES:
            for basis in ['linear','rbf']:
                z,q,y,w,diag=design(data,blocks[rep],a,b,recipe,basis)
                for alpha in ALPHAS:
                    for loss,cost in LOSSES:
                        coef,info=solve(z,(y-60)/10,w,loss,cost,alpha=alpha)
                        predictions.append(np.maximum(60+10*q@coef,0))
                        audit.append(dict(coef_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),**diag,**info))
    #Frozen existing parameter-model control, without representation changes.
    p=all_linear_predictions(data,cur,a,b,'HB_dense',[dict(columns=list(range(cur.shape[1])))])[0]
    predictions.append(p)
    return np.asarray(predictions),audit


def evaluate(data,blocks,cur,fold,extra):
    a0,b=fold['A'],fold['B'];ad=extra[~np.isin(data['groups'][extra],data['groups'][b])]
    fixed,audit=predict_all(data,blocks,cur,np.r_[a0,ad],b)
    inner=np.full((len(GRID)+1,len(a0)),np.nan);pos={int(v):j for j,v in enumerate(a0)}
    for f in fold['inner']:
        t,v=f['T'],f['V'];add=ad[~np.isin(data['groups'][ad],data['groups'][v])]
        p,_=predict_all(data,blocks,cur,np.r_[t,add],v)
        inner[:,[pos[int(j)] for j in v]]=p
    y=data['y'][a0,2];mm=[detailed(y,p) for p in inner]
    chosen=dict(mse=int(np.argmin(np.mean((inner-y)**2,axis=1))),
        safety=min(range(len(mm)),key=lambda j:(selection_key(mm[j]),j)))
    return fixed,chosen,audit,inner


def run(seeds,name,smoke=False,larger=False):
    global REPRESENTATIONS,GRID
    data,parts,extra,blocks,cur,hashes=load_inputs(seeds,larger=larger)
    if larger:
        REPRESENTATIONS=['base_amp6','base_current18','base_random_amp6','embedding_amp6','amp6']
        GRID=[dict(representation=r,recipe=w,basis=b,alpha=a,loss=l,cost=c)
              for r,w,b,a,(l,c) in itertools.product(REPRESENTATIONS,RECIPES,['linear','rbf'],ALPHAS,LOSSES)]
    out=ROOT/'outputs/uteg_mapping_target_runs'/name;(out/'private').mkdir(parents=True,exist_ok=True)
    grid=GRID+[dict(representation='frozen_current18_control',recipe='HB_dense')]
    protocol=dict(candidates=grid,seeds=list(parts),
        scope='C100 same60 grouped outer queries with40 historical records training-only; original201 features/labels excluded.',
        encoder=('Frozen AutonLab/MOMENT-1-base with matched base-random, small-pretrained and amplitude-only controls.' if larger else
            'Frozen AutonLab/MOMENT-1-small, pinned checkpoint/implementation and independent random architecture control.')+' Three views cover all900 first-column UTEG points; real observation masks retained. No UTEG label updates encoder.',
        representation=('Base2307D or small1539D frozen embedding blocks, plus6raw amplitude or current18 features.' if larger else
            '1539D =3*512 embeddings +3 observed fractions. Embedding alone, plus6raw amplitude features, pluscurrent18, random encoder+amplitudes, amplitude-only matched controls.'),
        downstream='Per-feature train-only standardization within each block and inverse sqrt block dimension; equal block norm. Exact training Gram linear or RBF exp(-mean block squared distance). PooledHB has source indicator. Free global intercept, normalized penalty .001/.01/.1.',
        loss='Uniform or class-inverse target weights cap20, or HB DenseWeight cap5/source cap20/target with equal domain mass; continuous MA MSE or correct-side boundary4/16 penalty and normal-side penalty1. No withholding or label-dependent postprocessing.',
        selection='Frozen3outer/3inner group folds choose among all candidates or existing parameter control. Every downstream scaler, kernel basis, weight and model fit within its own training fold. Fixed candidates exploratory.',
        extension=('5seed larger screen extends if strict pass, or agreement>80,FNR<10%,low recall>=80%, or R2>.35.' if larger else
            '5seed screen extends if strict pass or agreement>80,FNR<37%,low recall>=25%,or R2>.35.'),
        input_hashes=hashes,code_sha256=sha256(__file__),test_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=parts[1000]['pool'];pos={int(v):j for j,v in enumerate(pool)}
    old=np.load(ROOT/'outputs/uteg_mapping_target_runs/systematic_shift_audit_20261101/private/current300_HB_dense.npz')
    assert np.array_equal(old['pool'],pool)
    fixed=np.full((len(grid),seeds,len(pool)),np.nan);nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']}
    ledger=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            cache=out/f'private/{seed}_{fi}.npz';afile=out/f'private/{seed}_{fi}.json'
            if cache.exists() and afile.exists():
                z=np.load(cache);p=z['fixed'];inner=z['inner'];a=json.loads(afile.read_text(encoding='utf-8'));choice=a['choice'];audit=a['audit']
            else:
                p,choice,audit,inner=evaluate(data,blocks,cur,fold,extra)
                np.savez_compressed(cache,fixed=p,inner=inner)
                afile.write_text(json.dumps(dict(choice=choice,audit=audit),indent=2),encoding='utf-8')
            ix=[pos[int(v)] for v in fold['B']]
            assert np.max(abs(p[-1]-old['none'][si,ix]))<1e-7
            if smoke:
                poison=dict(data);poison['y']=data['y'].copy();poison['y'][fold['B']]=np.nan
                changed={k:[v.copy() for v in arrays] for k,arrays in blocks.items()}
                for arrays in changed.values():
                    for v in arrays:v[fold['B']]=1e4
                cc=cur.copy();cc[fold['B']]=1e4
                _,choice2,audit2,inner2=evaluate(poison,changed,cc,fold,extra)
                assert choice2==choice and audit2==audit;np.testing.assert_array_equal(inner2,inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_input_label_poison_invariant=True,control_replay=True),indent=2))
                print('SMOKE_COMPLETE',round(time.time()-begin,1),flush=True);return
            fixed[:,si,ix]=p
            for key,j in choice.items():nested[key][si,ix]=p[j]
            ledger.append(dict(seed=seed,fold=fi,selection={k:grid[j] for k,j in choice.items()}))
            print('MOMENT_FOLD',seed,fi,round(time.time()-begin,1),flush=True)
    yy=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    rows=[dict(candidate=c,metrics=detailed(yy,p)) for c,p in zip(grid,fixed)]
    selected=[dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()]
    for row,p in zip(rows+selected,list(fixed)+list(nested.values())):
        a,b=states(yy),states(p)
        assert [[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]==row['metrics']['confusion']
        assert abs(row['metrics']['r2']-(1-np.sum((p-yy)**2)/np.sum((yy-yy.mean())**2)))<1e-12
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=fixed,reference=yy[0],pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    promising=[r for r in rows if r['metrics']['mapping_target_pass'] or r['metrics']['r2']>.35 or
        (r['metrics']['agreement']>.8 and r['metrics']['fnr']<(.1 if larger else .37) and r['metrics']['low_recall']>=(.8 if larger else .25))]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows,nested=selected,promising=promising,
        checks=dict(base_control_replayed=True,metrics_independently_verified=True,input_hashes_unchanged=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='frozen_moment_mapping_screen_20261102');p.add_argument('--smoke',action='store_true');p.add_argument('--larger',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output,a.smoke,a.larger)
