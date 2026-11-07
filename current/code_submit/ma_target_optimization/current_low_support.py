"""Current300 representation/support diagnostic; query labels never fit geometry.

The complete90-point curve is the exact waveform summarized by current18
features. Distances are descriptive, not a trained classifier or an estimate
of irreducible error. Only authorized development/source rows are loaded.
"""
import json
import numpy as np
from .search import ROOT
from .common_clock import load_preprocessor, prefix_features
from .duration_transfer import TIME_COLUMNS,GROWTH_COLUMNS
from .signal_neighborhood import kinetics
from .hubei_source_search import add_hubei_arm
from .metrics import states
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs,sha256


def prefix_wave(signal,observed,horizon=300,preprocess=None):
    signal,observed=np.asarray(signal,float),np.asarray(observed,bool)
    if signal.ndim!=1 or signal.shape!=observed.shape:raise ValueError('Mismatched signal/mask')
    n=int(observed[:horizon].sum())
    if n<11 or not np.array_equal(np.flatnonzero(observed[:horizon]),np.arange(n)):
        raise ValueError('Expected contiguous observed prefix')
    values=signal[:n]
    if not np.isfinite(values).all():raise ValueError('Nonfinite visible signal')
    if preprocess is None:preprocess,_=load_preprocessor()
    processed,_=preprocess(values,q=.05)
    completed=np.pad(processed,(0,horizon-n),mode='edge')
    edges=np.linspace(0,horizon,91).astype(int)
    return np.array([np.median(completed[a:b]) for a,b in zip(edges[:-1],edges[1:])]),n/horizon


def load_current_representations(data,parts):
    path=ROOT/'outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz'
    preprocess,prep_path=load_preprocessor();rows=data['allowed_training_indices']
    wave=np.full((len(data['x']),90),np.nan);features=np.full((len(data['x']),18),np.nan)
    with np.load(path,allow_pickle=False) as a:
        assert np.array_equal(a['ids'],data['ids']) and np.array_equal(a['center'],data['centers'])
        raw,observed=a['original'][rows],a['observed'][rows]
    error=0.
    for row,signal,mask in zip(rows,raw,observed):
        curve,fraction=prefix_wave(signal,mask,preprocess=preprocess);wave[row]=curve
        f=kinetics(curve);f[TIME_COLUMNS]*=300/900;f[GROWTH_COLUMNS]*=900/300
        features[row]=np.r_[f,fraction]
        error=max(error,float(np.max(np.abs(features[row]-prefix_features(signal,mask,300,preprocess)))))
    assert error<1e-12
    test=next(iter(parts.values()))['test'];assert np.isnan(wave[test]).all() and np.isnan(features[test]).all()
    return {'current18':features,'current_curve90':wave}, {str(p):sha256(p) for p in (path,prep_path)}, {'current18_replay_max_error':error,'test_features_nan':True}


def distance(train,query,representation):
    train,query=np.asarray(train,float),np.asarray(query,float)
    if not np.isfinite(train).all() or not np.isfinite(query).all():raise ValueError('Finite signal features required')
    mean=train.mean(0)
    scale=max(float(train.std()),1e-6) if representation=='current_curve90' else np.maximum(train.std(0),1e-6)
    return np.sqrt(np.mean((((train-mean)/scale)-((query-mean)/scale))**2,axis=1))


def describe(d,training_labels,query_class,rng):
    order=np.argsort(d);same=training_labels==query_class;normal=training_labels==1
    result={'nearest_class':int(training_labels[order[0]]),
            'nearest5_counts':np.bincount(training_labels[order[:5]],minlength=3).tolist(),
            'nearest_distance':float(d[order[0]]),
            'same_min_distance':float(d[same].min()) if same.any() else None,
            'normal_min_distance':float(d[normal].min()) if normal.any() else None}
    if query_class!=1 and same.any() and normal.any():
        k=min(3,int(same.sum()),int(normal.sum()))
        a=np.flatnonzero(same);b=np.flatnonzero(normal);wins=[]
        for _ in range(200):wins.append(d[rng.choice(b,k,replace=False)].min()<d[rng.choice(a,k,replace=False)].min())
        result['normal_closer_equal_count_fraction']=float(np.mean(wins))
    return result


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_current_low_support_v1';out.mkdir(exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1020))
    data,parts,hashes=load_development_inputs(ROOT,cfg);add_hubei_arm(data,parts)
    matrices,more,audit=load_current_representations(data,parts);hashes.update(more)
    protocol={'code_hash':sha256(__file__),'input_hashes':hashes,'seeds':cfg['seeds'],
        'scope':'same60 target development records grouped outer folds; approved Hubei/Huzhou source pools; no test inputs or labels',
        'question':'whether current18 summaries lose local support present in the exact90-point prefix curve, and whether available source lowMA records are nearby',
        'geometry':'per-feature standardization for18 summaries; one amplitude scale for90-point curve; fitted only on each permitted reference pool',
        'labels':'query labels only describe geometry after distances fixed; never used for scaling, nearest selection or feature extraction',
        'balanced_diagnostic':'200 deterministic draws per query, up to3 reference records each from same and normal class; counts balanced, not training resampling',
        'limits':'distance descriptions are not validated classifiers or proof that raw signals contain no predictive information'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=next(iter(parts.values()))['pool'];y=data['y'][:,2];labels=states(y[pool]);rng=np.random.default_rng(20261101)
    source={name:np.array([i for i in data['allowed_training_indices'] if data['centers'][i]==name]) for name in ('Huzhou','Hubei')}
    rows=[]
    for rep,x in matrices.items():
        for seed,part in parts.items():
            for foldnum,fold in enumerate(part['folds']):
                for name,train_rows in {'target_training':fold['A'],**source}.items():
                    train_labels=states(y[train_rows])
                    for query in fold['B']:
                        keep=data['groups'][train_rows]!=data['groups'][query];T=train_rows[keep]
                        d=distance(x[T],x[query],rep);qclass=int(states([y[query]])[0])
                        entry={'representation':rep,'seed':seed,'fold':foldnum,'reference_pool':name,'query_row':int(query),
                               'query_class':qclass,'query_ma':float(y[query]),'training_class_counts':np.bincount(train_labels[keep],minlength=3).tolist(),
                               'nearest_row':int(T[np.argmin(d)]),'nearest_ma':float(y[T[np.argmin(d)]]),**describe(d,train_labels[keep],qclass,rng)}
                        rows.append(entry)
    best=json.loads((ROOT/'outputs/uteg_mapping_target_runs/BEST_CURRENT_VERIFIED_20261101.json').read_text(encoding='utf-8'))
    with np.load(best['prediction_path']) as p:
        np.testing.assert_array_equal(p['pool'],pool);pred=p['pred'];reference=p['reference']
    low_indices=np.flatnonzero(reference<50);low_summaries=[]
    for index in low_indices:
        query=int(pool[index]);record={'reference_ma':float(reference[index]),'correct_low_out_of20':int((pred[:,index]<50).sum()),
            'mean_predicted_ma':float(pred[:,index].mean()),'mean_absolute_error':float(np.abs(pred[:,index]-reference[index]).mean()),'support':{}}
        for rep in matrices:
            record['support'][rep]={}
            for name in ('target_training','Huzhou','Hubei'):
                selected=[r for r in rows if r['query_row']==query and r['representation']==rep and r['reference_pool']==name]
                record['support'][rep][name]={'nearest_low_fraction':float(np.mean([r['nearest_class']==0 for r in selected])),
                    'nearest_normal_fraction':float(np.mean([r['nearest_class']==1 for r in selected])),
                    'mean_nearest5_low_count':float(np.mean([r['nearest5_counts'][0] for r in selected])),
                    'mean_normal_closer_equal_count_fraction':float(np.mean([r['normal_closer_equal_count_fraction'] for r in selected])),
                    'median_nearest_ma':float(np.median([r['nearest_ma'] for r in selected])),
                    'same_to_normal_min_distance_ratio':float(np.median([r['same_min_distance']/max(r['normal_min_distance'],1e-12) for r in selected]))}
        low_summaries.append(record)
    summary={'scope':'development signal-support diagnostic, not a new model or impossibility bound','records':60,'low_records':4,
             'counts':np.bincount(labels,minlength=3).tolist(),'input_audit':audit,'low_records_without_ids':low_summaries,
             'model_id':best['candidate']['id'],'test_not_evaluated':True}
    (out/'private/NEIGHBORHOODS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(out/'private/REPRESENTATIONS.npz',current18=matrices['current18'],curve90=matrices['current_curve90'],pool=pool,allowed=data['allowed_training_indices'])
    print(json.dumps(summary))


if __name__=='__main__':run()
