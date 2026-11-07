"""Last-value padded and observed-progress kinetics, same target label budget.

The user explicitly approved completing short curves with their last value.
Padding is a model input convention, never counted as new measured signal.
Observed-progress features retain the recorded duration and make no claim that
record termination equals a physiological plateau.
"""
import hashlib
import itertools
import json
import time
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from .search import ROOT,training_view,make_summary
from .signal_neighborhood import kinetics
from .models import fit_candidate,Features,weights
from .center_tree_search import fit as fit_tree
from .hierarchical_search import HierarchicalModel,objective
from .hubei_source_search import add_hubei_arm
from .metrics import assess
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs,sha256

REPRESENTATIONS=['padded17','padded_with_length18','observed_absolute18','observed_relative18']
TIME_COLUMNS=[3,4,5,6,8]
GROWTH_COLUMNS=[7,9]


def features(signal,representation):
    signal=np.asarray(signal,float);n=len(signal)
    if n<90 or n>900 or not np.isfinite(signal).all():raise ValueError('Invalid observed prefix')
    if representation.startswith('padded'):
        completed=np.pad(signal,(0,900-n),mode='edge')
        f=kinetics(np.median(completed.reshape(90,10),axis=1))
        return f if representation=='padded17' else np.r_[f,n/900.]
    # Equal clock intervals (up to one sample), rather than front-loading
    # all larger bins as np.array_split does for non-divisible lengths.
    edges=np.linspace(0,n,91).astype(int)
    wave=np.asarray([np.median(signal[a:b]) for a,b in zip(edges[:-1],edges[1:])])
    f=kinetics(wave)
    if representation=='observed_absolute18':
        f[TIME_COLUMNS]*=n/900.
        f[GROWTH_COLUMNS]*=900./n
    elif representation=='observed_relative18':
        f[TIME_COLUMNS]/=900.
        f[GROWTH_COLUMNS]*=15.
    else:raise ValueError(representation)
    return np.r_[f,n/900.]


def load_features(data,parts):
    paths=[ROOT/'outputs/uteg_cohort399_20261006_201800/private/data.npz',
           ROOT/'outputs/uteg_platform_filter_pilot_20261006_124636/private/CUT_5.npz',
           ROOT/'outputs/uteg_legacy_aligned_threecenter_20261022_155614/HUBEI_SIGNAL_INPUTS.npz']
    with np.load(paths[0],allow_pickle=False) as z:
        original=z['original_row_index'];assert np.array_equal(z['ids'],data['ids'][:399])
    with np.load(paths[1],allow_pickle=False) as z:old=z['signal'];old_obs=z['observed']
    with np.load(paths[2],allow_pickle=False) as z:hb=z['denoise_p5'];hb_obs=z['observed']
    matrices={name:np.full((len(data['x']),17 if name=='padded17' else 18),np.nan) for name in REPRESENTATIONS}
    lengths=np.zeros(len(data['x']),int)
    for i in data['allowed_training_indices']:
        row,mask=(old[original[i]],old_obs[original[i]]) if i<399 else (hb[i-399],hb_obs[i-399])
        n=int(mask.sum());assert np.array_equal(np.flatnonzero(mask),np.arange(n))
        signal=row[:n];lengths[i]=n
        for rep in REPRESENTATIONS:matrices[rep][i]=features(signal,rep)
    test=next(iter(parts.values()))['test']
    assert all(np.isnan(v[test]).all() for v in matrices.values())
    return matrices,lengths,{str(p):sha256(p) for p in paths}


def fit_partial(data,A,c):
    source=data['source_rows'][c['arm']]
    centers=[source[data['centers'][source]==name] for name in np.unique(data['centers'][source])]+[A]
    transforms=[Features.fit(data['x'][rows],'linear') for rows in centers]
    width=data['x'].shape[1]+1;blocks=[];labels=[];importance=[]
    for number,(rows,transform) in enumerate(zip(centers,transforms)):
        x=data['x'][rows];y=data['y'][rows,2];v=np.c_[transform.transform(x),np.ones(len(x))]
        z=np.zeros((len(x),width*(1+len(centers))));z[:,:width]=v;z[:,width*(number+1):width*(number+2)]=v
        blocks.append(z);labels.append(y)
        importance.append(weights(y,c['cap'])/len(y)*(4. if number==len(centers)-1 else 1.))
    z=np.vstack(blocks);y=np.concatenate(labels);w=np.concatenate(importance);w/=w.mean()
    penalty=np.ones(z.shape[1]);penalty[:width]=.01;penalty[width-1]=0
    coef=np.linalg.solve(z.T@(w[:,None]*z)+len(y)*np.diag(penalty)+1e-10*np.eye(z.shape[1]),z.T@(w*(y-60)/10))
    if c['boundary_strength']:
        result=minimize(objective,coef,args=(z,y,w,penalty,c['boundary_strength'],16.),jac=True,
                        method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-11,'gtol':1e-8,'maxls':50})
        if not result.success:raise RuntimeError(str(result.message))
        coef=result.x
    return HierarchicalModel(transforms,coef)


def grid():
    models=[dict(family='ridge',arm='target_only',features='linear',alpha=.01,cap=cap) for cap in [1.,20.]]
    for arm in ['HZ_to_SA','HB_to_SA','HZ+HB_to_SA']:
        for cap in [1.,20.]:
            models.append(dict(family='transfer',arm=arm,mode='joint',source_alpha=.1,source_cap=5.,cap=cap,adapter_penalty=10.))
            models.append(dict(family='center_tree',arm=arm,center_scale=True,loss='squared_error',leaf=1,target_mass=4.,cap=cap))
        for boundary,cap in [(0.,1.),(4.,5.)]:
            models.append(dict(family='duration_partial',arm=arm,boundary_strength=boundary,cap=cap))
    result=[]
    for rep,model in itertools.product(REPRESENTATIONS,models):
        c={**model,'representation':rep};c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12];result.append(c)
    return result


def fit(data,A,c,cache):
    if c['family']=='center_tree':return fit_tree(data,A,c)
    if c['family']=='duration_partial':return fit_partial(data,A,c)
    return fit_candidate(c,training_view(data,A),cache)


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_duration_transfer_v1';out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1005))
    data,parts,hashes=load_development_inputs(ROOT,cfg);add_hubei_arm(data,parts)
    matrices,lengths,more=load_features(data,parts);hashes.update(more)
    pool=next(iter(parts.values()))['pool'];pos={int(v):i for i,v in enumerate(pool)};candidates=grid()
    protocol={'candidates':candidates,'seeds':cfg['seeds'],'input_hashes':hashes,
              'code_hashes':{name:sha256(ROOT/'code_submit/ma_target_optimization'/name) for name in ['duration_transfer.py','signal_neighborhood.py','models.py','center_tree_search.py','hierarchical_search.py']},
              'scope':'development-only; original target60 and permitted sources; test features and labels remain unavailable',
              'authorization':'user says short records may be completed; use last observed value to fill to900 points',
              'padding':'no synthetic measurements claimed; effective observed count and padded status retained; no new sequential checkpoint inferred from padding',
              'representations':{'padded17':'last-value-completed900-point kinetics',
                 'padded_with_length18':'same17 plus observed fraction',
                 'observed_absolute18':'90 equal-progress bins over true observed prefix; times scaled by actual duration, rates per actual minute; plus duration',
                 'observed_relative18':'same observed profile, times as record fractions and rates per record progress; plus duration'},
              'caveat':'profile endpoint is recorded endpoint, not verified physiological plateau; observed-profile thirds are fractions of available record, not necessarily5/10minutes',
              'same_target_control':'all target records have900 observed points; expected target-only predictions invariant under constant feature rescaling/constant duration column'}
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');np.savez_compressed(out/'private/OBSERVATION_MASK.npz',observed_length=lengths,permitted_rows=data['allowed_training_indices'])
    path=out/'CANDIDATES.jsonl';records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [];done={r['candidate']['id'] for r in records}
    caches={rep:{} for rep in REPRESENTATIONS}
    for number,c in enumerate(candidates):
        if c['id'] in done:continue
        data['x']=matrices[c['representation']];cache=caches[c['representation']];start=time.time()
        try:
            predictions=[]
            for seed,part in parts.items():
                p=np.full(len(pool),np.nan)
                for fold in part['folds']:
                    A,B=fold['A'],fold['B'];model=fit(data,A,c,cache)
                    p[[pos[int(v)] for v in B]]=model.predict(data['x'][B])
                predictions.append(p)
            p=np.asarray(predictions);y=data['y'][pool,2]
            np.savez_compressed(out/'private'/f"{c['id']}.npz",prediction=p,reference=y,pool=pool,seeds=cfg['seeds'])
            row={'candidate':c,'status':'complete','metrics':assess(np.broadcast_to(y,p.shape),p),'seconds':time.time()-start}
        except Exception as error:row={'candidate':c,'status':'failed','error':str(error)}
        records.append(row)
        with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if (number+1)%10==0 or number+1==len(candidates):
            s=make_summary(records,out);print(json.dumps({'done':number+1,'total':len(candidates),'failed':s['failed'],'feasible':s['feasible_count']}),flush=True)
    return make_summary(records,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
