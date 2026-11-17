"""Tail-direction correction with binary abnormal decisions held invariant."""
import argparse
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import pinned_pls_mapping as baseline
from . import low_rank_curve_mapping as shared
from .full_curve_state_experts import classifier, probability, signature
from .high_ma_specialist import binary_weights
from .feature_combination_audit import detailed
from .metrics import states
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

REPS = ['base_amp6', 'embedding_amp6', 'amp6', 'current18']
MODELS = [dict(representation=r, domain=d, basis=b, C=c) for r, d, b, c in itertools.product(
    REPS, ['target', 'HB_target'], ['linear', 'rbf'], [1., 10.])]
RULES = [dict(scope='low_only', threshold=.5), dict(scope='low_only', threshold=.75), dict(scope='both', threshold=.5)]
GRID = [dict(**m, **rule) for m in MODELS for rule in RULES] + [dict(control='pinned_pls_rescue')]
# Fixed after the5-seed development screen, before its20-seed extension.
PINNED_DIRECTION = dict(representation='amp6', domain='target', basis='rbf', C=10., scope='low_only', threshold=.5)


def correct(prediction, high_score, means, scope, threshold):
    p = prediction.copy(); s = states(p)
    high = (s==0)&(high_score>=threshold)
    p[high] = means[2]
    if scope=='both': p[(s==2)&(high_score<threshold)] = means[0]
    elif scope!='low_only': raise ValueError(scope)
    assert np.array_equal(states(p)==1, s==1)
    return p


def predict_pinned(data, blocks, cur, smooth, a, b):
    base, raw, base_audit = baseline.predict(data, blocks, cur, smooth, a, b)
    local = a[states(data['y'][a, 2])!=1]
    z, q, y, ignored, diag = shared.mapping.design(data, blocks['amp6'], local, b, 'target_uniform', 'rbf')
    label = (y>70).astype(int); w = binary_weights(label, 0)
    model, constant = classifier(z[:, :-1], label, w, 10.)
    score = probability(model, constant, q[:, :-1], 2)[:, 1]
    means, counts = shared.calibration.conditional_means(data, a)
    p = correct(base, score, means, 'low_only', .5)
    return p, raw, dict(baseline=base_audit, direction=diag, direction_sha256=signature(model,constant),
                       means=means.tolist(), counts=counts)


def predict_all(data, blocks, cur, smooth, a, b, n_main):
    base, raw, diag = baseline.predict(data, blocks, cur, smooth, a, b)
    local = a[states(data['y'][a,2])!=1]
    source = data['source_rows']['HB_to_SA']; source = source[states(data['y'][source,2])!=1]
    shadow = dict(data); shadow['source_rows'] = dict(data['source_rows'], HB_to_SA=source)
    means, counts = shared.calibration.conditional_means(data, a)
    predictions = []; audit = []
    for rep in REPS:
        for domain in ['target', 'HB_target']:
            for basis in ['linear', 'rbf']:
                recipe = 'target_uniform' if domain=='target' else 'HB_dense'
                z, q, y, ignored_weights, info = shared.mapping.design(shadow, blocks[rep], local, b, recipe, basis)
                assert np.all(states(y)!=1)
                label = (y>70).astype(int); w = binary_weights(label, info['n_source'])
                for C in [1., 10.]:
                    model, constant = classifier(z[:,:-1], label, w, C)
                    score = probability(model, constant, q[:,:-1], 2)[:,1]
                    audit.append(dict(model_sha256=signature(model,constant), training_tail_counts=np.bincount(label,minlength=2).tolist(), **info))
                    for rule in RULES:
                        predictions.append(correct(base,score,means,**rule))
    predictions.append(base)
    return np.asarray(predictions), dict(baseline=diag, means=means.tolist(), counts=counts, heads=audit)


def evaluate(data, blocks, cur, smooth, fold, extra):
    return shared.evaluate(data, blocks, cur, smooth, fold, extra, predictor=predict_all, grid=GRID)


def run(seeds,name,smoke=False):
    data,parts,extra,blocks,cur,hashes=shared.mapping.load_inputs(seeds,larger=True); blocks['current18']=[cur]
    d2,p2,ex2,reps,mask,more=shared.load_waveforms(seeds); hashes.update(more)
    assert np.array_equal(extra,ex2) and np.array_equal(parts[1000]['pool'],p2[1000]['pool'])
    out=ROOT/'outputs/uteg_mapping_target_runs'/name; (out/'private').mkdir(parents=True,exist_ok=True)
    protocol=dict(candidates=GRID,seeds=list(parts),pinned_after5seed=PINNED_DIRECTION if seeds>5 else None,
        scope='Same60 main grouped development queries; extra40 training-only; original201 excluded.',
        hypothesis='Some already-abnormal outputs have the wrong direction. A high-vs-low head trained only on actual abnormal training records may correct that direction without manipulating binary FNR/FPR.',
        training='Standardization and linear/RBF Gram fitted only on actual low/high TRAINING records. L2 logistic C1/10 with equal binary class mass within each training domain and equal HB/target domain mass. Normal training labels only define exclusion from this tail head; they remain in the unchanged baseline mapping.',
        policy='Existing predicted-normal rows unchanged. Low-only correction thresholds .5/.75, or both directions at.5. Changed MA is corresponding target TRAINING class mean, source fallback only if absent locally. Same final MA determines all class/numeric metrics. FNR/FPR are mathematically invariant and checked.',
        selection='Each inner fold refits baseline, tail normalization/classifier and class means; inner predictions choose MSE/safety; outer labels score only. Fixed candidates exploratory.',
        extension='Extend5seed only if agreement>=baseline and worst-tail recall>=baseline and MAE decreases>.05mm; or agreement>baseline and MAE<=baseline and both recalls>=80%; or R2>.35.',
        input_hashes=hashes,code_sha256=sha256(__file__),baseline_sha256=sha256(baseline.__file__),
        test_evaluated=False,gate_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    old=np.load(ROOT/'outputs/uteg_mapping_target_runs/pinned_pls_gate_20seed_20261102/private/PREDICTIONS.npz')
    pool=parts[1000]['pool']; assert np.array_equal(old['pool'],pool);pos={int(v):j for j,v in enumerate(pool)}
    fixed=np.full((len(GRID),seeds,len(pool)),np.nan);nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']};ledger=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            pred,choice,audit,inner=evaluate(data,blocks,cur,reps['smooth900'],fold,extra)
            ix=[pos[int(v)] for v in fold['B']];np.testing.assert_allclose(pred[-1],old['prediction'][si,ix],rtol=0,atol=1e-7)
            if smoke:
                poison=dict(data);poison['y']=data['y'].copy();poison['y'][fold['B']]=np.nan
                changed={k:[v.copy() for v in arrays] for k,arrays in blocks.items()}
                for arrays in changed.values():
                    for v in arrays:v[fold['B']]=1e4
                cc=cur.copy();cc[fold['B']]=1e4;ss=reps['smooth900'].copy();ss[fold['B']]=1e4
                _,ch2,au2,in2=evaluate(poison,changed,cc,ss,fold,extra)
                assert ch2==choice and au2==audit;np.testing.assert_array_equal(in2,inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True,baseline_replayed=True),indent=2),encoding='utf-8')
                print('SMOKE_COMPLETE',round(time.time()-begin,1),flush=True);return
            fixed[:,si,ix]=pred
            for key,j in choice.items():nested[key][si,ix]=pred[j]
            ledger.append(dict(seed=seed,fold=fi,choice={k:GRID[j] for k,j in choice.items()},audit=audit))
        print('DIRECTION_SEED',seed,round(time.time()-begin,1),flush=True)
    yy=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    rows=[dict(candidate=c,metrics=detailed(yy,p)) for c,p in zip(GRID,fixed)]
    selected=[dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()];base=rows[-1]['metrics']
    for row,p in zip(rows+selected,list(fixed)+list(nested.values())):
        np.testing.assert_array_equal(states(p)==1,states(fixed[-1])==1)
        assert row['metrics']['fnr']==base['fnr'] and row['metrics']['fpr']==base['fpr']
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=fixed,reference=yy[0],pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    worst=lambda m:min(m['low_recall'],m['high_recall'])
    promising=[r for r in rows[:-1] if
        (r['metrics']['agreement']>=base['agreement'] and worst(r['metrics'])>=worst(base) and r['metrics']['mae']<base['mae']-.05) or
        (r['metrics']['agreement']>base['agreement'] and r['metrics']['mae']<=base['mae'] and worst(r['metrics'])>=.8) or r['metrics']['r2']>.35]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows,nested=selected,promising=promising,
        checks=dict(binary_decisions_unchanged=True,baseline_replayed=True,input_hashes_unchanged=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='abnormal_direction_mapping_screen_20261102');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output,a.smoke)
