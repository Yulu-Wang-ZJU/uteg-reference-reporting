"""Grouped train-only high-MA detector added to the fixed low-rescue mapping.

One-vs-rest inverse-frequency logistic loss specifically learns high MA. It may
correct a primary low prediction, unlike earlier normal-only rescue rules.
"""
import argparse
import itertools
import json
import time
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from . import specialist_boundary_calibration as calibration
from . import frozen_moment_mapping as mapping
from .full_curve_cost_sensitive import load_waveforms
from .full_curve_state_experts import classifier, probability, signature
from .feature_combination_audit import detailed
from .metrics import states, selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

REPS = ['embedding_amp6', 'embedding_current18', 'smooth900', 'amp6']
MODELS = [dict(representation=r, domain=d, basis=b, C=c)
          for r, d, b, c in itertools.product(REPS, ['target', 'HB_target'],
                                              ['linear', 'rbf'], [1., 10.])]
GRID = [dict(**m, threshold=t, scope=s) for m in MODELS
        for t in [.5, .75] for s in ['normal_only', 'all_nonhigh']]
GRID.append(dict(control='calibrated_low_rescue'))
BASE = dict(low_offset=1., high_offset=0., priority='low_first', numeric='conditional_mean')


def binary_weights(label, ns):
    """Equal observed class mass within each training domain; equal domain mass."""
    label = np.asarray(label, int)
    w = np.empty(len(label))
    slices = [slice(0, ns), slice(ns, len(label))] if ns else [slice(0, len(label))]
    for s in slices:
        yy = label[s]
        counts = np.bincount(yy, minlength=2)
        present = int(np.count_nonzero(counts))
        w[s] = 1. / (present * counts[yy])
    return w / w.mean()


def apply(base, score, high_mean, threshold, scope):
    out = base.copy()
    take = (states(base) == 1) if scope == 'normal_only' else (states(base) != 2)
    if scope not in ['normal_only', 'all_nonhigh']:
        raise ValueError(scope)
    out[take & (score >= threshold)] = high_mean
    return out


def predict_all(data, blocks, cur, smooth, a, b):
    raw, audit = calibration.source.models(data, blocks, cur, smooth, a, b)
    means, counts = calibration.conditional_means(data, a)
    base = calibration.predict(raw, BASE, means)
    predictions, scores, diagnostics = [], [], []
    for rep in REPS:
        for domain in ['target', 'HB_target']:
            for basis in ['linear', 'rbf']:
                recipe = 'target_uniform' if domain == 'target' else 'HB_dense'
                z, q, y, unused_weights, diag = mapping.design(data, blocks[rep], a, b, recipe, basis)
                label = (y > 70).astype(int)
                w = binary_weights(label, diag['n_source'])
                for C in [1., 10.]:
                    model, constant = classifier(z[:, :-1], label, w, C)
                    score = probability(model, constant, q[:, :-1], 2)[:, 1]
                    scores.append(score)
                    diagnostics.append(dict(model_sha256=signature(model, constant),
                        class_counts=np.bincount(label, minlength=2).tolist(),
                        weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(), **diag))
                    for threshold in [.5, .75]:
                        for scope in ['normal_only', 'all_nonhigh']:
                            predictions.append(apply(base, score, means[2], threshold, scope))
    predictions.append(base)
    return np.asarray(predictions), dict(base=audit, conditional_means=means.tolist(),
        conditional_counts=counts, high_heads=diagnostics), np.asarray(scores)


def evaluate(data, blocks, cur, smooth, fold, extra):
    a0, b = fold['A'], fold['B']
    ad = extra[~np.isin(data['groups'][extra], data['groups'][b])]
    fixed, audit, score = predict_all(data, blocks, cur, smooth, np.r_[a0, ad], b)
    inner = np.full((len(GRID), len(a0)), np.nan)
    pos = {int(v): j for j, v in enumerate(a0)}
    for f in fold['inner']:
        t, v = f['T'], f['V']
        add = ad[~np.isin(data['groups'][ad], data['groups'][v])]
        pp, _, _ = predict_all(data, blocks, cur, smooth, np.r_[t, add], v)
        inner[:, [pos[int(j)] for j in v]] = pp
    y = data['y'][a0, 2]
    mm = [detailed(y, p) for p in inner]
    choice = dict(mse=int(np.argmin(np.mean((inner-y)**2, axis=1))),
        safety=min(range(len(GRID)), key=lambda j: (selection_key(mm[j]), j)))
    return fixed, choice, audit, inner, score


def run(seeds, name, smoke=False, larger=False):
    global REPS, MODELS, GRID
    data, parts, extra, blocks, cur, hashes = mapping.load_inputs(seeds)
    d2, p2, e2, reps, mask, more = load_waveforms(seeds)
    hashes.update(more)
    assert np.array_equal(extra, e2) and np.array_equal(parts[1000]['pool'], p2[1000]['pool'])
    blocks['smooth900'] = [reps['smooth900']]
    if larger:
        feature_dir = ROOT / 'outputs/uteg_mapping_target_runs/frozen_moment_base_features_20261102'
        z = np.load(feature_dir / 'private/FEATURES.npz')
        assert np.array_equal(z['allowed'], data['allowed_training_indices'])
        assert np.array_equal(z['pool'], parts[1000]['pool'])
        blocks['base_amp6'] = [z['pretrained'], blocks['amp6'][0]]
        blocks['base_current18'] = [z['pretrained'], cur]
        blocks['base_random_amp6'] = [z['random'], blocks['amp6'][0]]
        REPS = ['base_amp6', 'base_current18', 'base_random_amp6', 'embedding_amp6', 'amp6']
        MODELS = [dict(representation=r, domain=d, basis=b, C=c)
                  for r, d, b, c in itertools.product(REPS, ['target', 'HB_target'], ['linear', 'rbf'], [1., 10.])]
        GRID = [dict(**m, threshold=t, scope=s) for m in MODELS
                for t in [.5, .75] for s in ['normal_only', 'all_nonhigh']] + [dict(control='calibrated_low_rescue')]
        for p in [feature_dir / 'private/FEATURES.npz', feature_dir / 'PROTOCOL.json', feature_dir / 'VERIFICATION.json']:
            hashes[str(p)] = sha256(p)
        for key in REPS:
            for x in blocks[key]:
                assert np.isnan(x[parts[1000]['test']]).all()
    out = ROOT / 'outputs/uteg_mapping_target_runs' / name
    (out / 'private').mkdir(parents=True, exist_ok=True)
    protocol = dict(candidates=GRID, seeds=list(parts), baseline=BASE,
        question='Can a dedicated high-vs-rest classifier correct remaining high-to-normal and high-to-low errors without destroying low recall or normal agreement?',
        scope='Same C100 grouped 60-query development protocol; extra40 training only; original201 features/labels excluded.',
        classifier='L2 logistic C1/10 in linear/RBF training Gram space. Training-only binary inverse counts give equal high/non-high class mass within each domain, then equal HB/target domain mass. Sample weights normalized to mean1. No test/query-label weighting.',
        representation=('Frozen MOMENT-base plus amp6/current18, same-base random+amp6, frozen-small+amp6 and amp6-only controls.' if larger else
            'Frozen MOMENT-small plus amp6/current18, full smooth900 and matched amp6-only.') + ' All standardization and Gram bases fit on actual training rows.',
        policies='Fixed score thresholds .5/.75; apply either to current normal only or all non-high. Existing high predictions preserved. Return same-class target-training high-MA mean; source fallback only if absent locally. Same output used for numerical MA and class metrics. Scores not claimed calibrated.',
        selection=f'All{len(GRID)} configurations including existing control selected on inner3fold; outer3fold evaluates. Fixed results exploratory. No independent test or patient-specific rules.',
        extension='Extend5seed to20 only if strict goal pass, or agreement>80,FNR<10%,low recall>=80%,high recall>75%; or agreement>=control,FNR<=control,MAE<control.',
        source='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
        input_hashes=hashes, code_sha256=sha256(__file__), test_evaluated=False, gate_evaluated=False, manuscript_changed=False)
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    old = np.load(ROOT / 'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/private/PREDICTIONS.npz')
    od = json.loads((ROOT / 'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/RESULTS.json').read_text())
    oi = next(i for i, r in enumerate(od['fixed']) if r['candidate'] == BASE)
    pool = parts[1000]['pool']
    assert np.array_equal(old['pool'], pool)
    pos = {int(v): j for j, v in enumerate(pool)}
    fixed = np.full((len(GRID), seeds, len(pool)), np.nan)
    scores = np.full((len(MODELS), seeds, len(pool)), np.nan)
    nested = {k: np.full((seeds, len(pool)), np.nan) for k in ['mse', 'safety']}
    ledger = []
    begin = time.time()
    for si, (seed, part) in enumerate(parts.items()):
        for fi, fold in enumerate(part['folds']):
            pred, choice, audit, inner, score = evaluate(data, blocks, cur, reps['smooth900'], fold, extra)
            ix = [pos[int(v)] for v in fold['B']]
            np.testing.assert_allclose(pred[-1], old['fixed'][oi, si, ix], rtol=0, atol=1e-7)
            if smoke:
                poison = dict(data)
                poison['y'] = data['y'].copy()
                poison['y'][fold['B']] = np.nan
                changed = {k: [v.copy() for v in values] for k, values in blocks.items()}
                for values in changed.values():
                    for v in values:
                        v[fold['B']] = 1e4
                cc = cur.copy(); cc[fold['B']] = 1e4
                ss = reps['smooth900'].copy(); ss[fold['B']] = 1e4
                _, ch2, au2, in2, _ = evaluate(poison, changed, cc, ss, fold, extra)
                assert ch2 == choice and au2 == audit
                np.testing.assert_array_equal(in2, inner)
                (out / 'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True,
                    previous_candidate_replayed=True), indent=2), encoding='utf-8')
                print('SMOKE_COMPLETE', round(time.time()-begin, 1), flush=True)
                return
            fixed[:, si, ix] = pred
            scores[:, si, ix] = score
            for key, j in choice.items():
                nested[key][si, ix] = pred[j]
            ledger.append(dict(seed=seed, fold=fi, choice={k: GRID[j] for k, j in choice.items()}, audit=audit))
        print('HIGH_SPECIALIST_SEED', seed, round(time.time()-begin, 1), flush=True)
    yy = np.broadcast_to(data['y'][pool, 2], (seeds, len(pool)))
    rows = [dict(candidate=c, metrics=detailed(yy, p)) for c, p in zip(GRID, fixed)]
    selected = [dict(objective=k, metrics=detailed(yy, p)) for k, p in nested.items()]
    for row, p in zip(rows + selected, list(fixed) + list(nested.values())):
        t, q = states(yy), states(p)
        assert [[int(((t == i) & (q == j)).sum()) for j in range(3)] for i in range(3)] == row['metrics']['confusion']
    for path, digest in hashes.items():
        assert sha256(path) == digest
    np.savez_compressed(out / 'private/PREDICTIONS.npz', fixed=fixed, scores=scores,
                        reference=yy[0], pool=pool, seeds=list(parts), **nested)
    (out / 'private/SELECTIONS.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    control = rows[-1]['metrics']
    promising = [r for r in rows[:-1] if r['metrics']['mapping_target_pass'] or
        (r['metrics']['agreement'] > .8 and r['metrics']['fnr'] < .1 and
         r['metrics']['low_recall'] >= .8 and r['metrics']['high_recall'] > .75) or
        (r['metrics']['agreement'] >= control['agreement'] and r['metrics']['fnr'] <= control['fnr']
         and r['metrics']['mae'] < control['mae']-1e-6)]
    (out / 'RESULTS.json').write_text(json.dumps(dict(fixed=rows, nested=selected, promising=promising,
        checks=dict(previous_candidate_replayed=True, input_hashes_unchanged=True),
        test_evaluated=False, gate_evaluated=False), indent=2), encoding='utf-8')
    print('COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--output', default='high_ma_specialist_screen_20261102')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--larger', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.seeds, a.output, a.smoke, a.larger)
