"""Train-only principal-component constraints on frozen curve-to-MA mappings."""
import argparse
import hashlib
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import frozen_moment_mapping as mapping
from . import specialist_boundary_calibration as calibration
from .full_curve_cost_sensitive import load_waveforms, solve
from .density_reweight_mapping import label_weights
from .feature_combination_audit import detailed
from .metrics import states, selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

REPS = ['base_amp6', 'embedding_amp6', 'base_random_amp6']
DIMS = [2, 4, 8, 16]
RECIPES = ['target_uniform', 'target_class', 'HB_dense']
ALPHAS = [.001, .01, .1]
LOSSES = [('control', 1.), ('boundary', 16.)]
GRID = [dict(representation=r, components=k, recipe=w, alpha=a, loss=l, cost=c)
        for r, k, w, a, (l, c) in itertools.product(REPS, DIMS, RECIPES, ALPHAS, LOSSES)]
CONTROLS = ['full_base_amp6', 'amp6_only', 'current18_primary', 'calibrated_low_rescue']
GRID += [dict(control=c) for c in CONTROLS]
BASE = dict(low_offset=1., high_offset=0., priority='low_first', numeric='conditional_mean')


def principal_blocks(embedding, amplitude, train, query, dimensions=DIMS):
    em = embedding[train].mean(0); es = np.maximum(embedding[train].std(0), 1e-6)
    x = (embedding[train]-em)/es; q = (embedding[query]-em)/es
    _, singular, directions = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(singular > max(1e-12, singular[0]*1e-10)))
    am = amplitude[train].mean(0); ass = np.maximum(amplitude[train].std(0), 1e-6)
    ax = (amplitude[train]-am)/ass/np.sqrt(amplitude.shape[1])
    aq = (amplitude[query]-am)/ass/np.sqrt(amplitude.shape[1])
    answer = {}
    for wanted in dimensions:
        k = min(wanted, rank, len(train)-1)
        v = directions[:k]
        # Do not whiten individual PCs: preserve their relative variance.
        xx, qq = x@v.T, q@v.T
        norm = max(float(np.sqrt(np.mean(np.sum(xx*xx, axis=1)))), 1e-12)
        tx = np.concatenate([xx/norm, ax], axis=1)/np.sqrt(2)
        qx = np.concatenate([qq/norm, aq], axis=1)/np.sqrt(2)
        fraction = float(np.sum(singular[:k]**2)/np.sum(singular**2)) if rank else 0.
        digest = hashlib.sha256(em.tobytes()+es.tobytes()+am.tobytes()+ass.tobytes()+v.tobytes()+np.asarray(norm).tobytes()).hexdigest()
        answer[wanted] = (tx, qx, dict(components=k, explained_variance_fraction=fraction, transform_sha256=digest))
    return answer


def training_rows_weights(data, a, recipe):
    source = data['source_rows']['HB_to_SA'] if recipe == 'HB_dense' else np.array([], int)
    rows = np.r_[source, a]; ns = len(source); y = data['y'][rows, 2]
    method = {'target_uniform': 'uniform', 'target_class': 'class_inverse', 'HB_dense': 'denseweight'}[recipe]
    w = np.ones(len(rows)); w[ns:] = label_weights(y[ns:], method, 20.)[0]/len(a)
    if ns:
        w[:ns] = label_weights(y[:ns], method, 5.)[0]/ns
    return rows, ns, y, w/w.mean()


def predict_all(data, blocks, cur, smooth, a, b, n_main):
    result = []; audit = []; cache = {}
    for rep in REPS:
        for components in DIMS:
            for recipe in RECIPES:
                tr, ns, y, w = training_rows_weights(data, a, recipe)
                assert not set(data['groups'][tr]).intersection(data['groups'][b])
                key = (rep, bool(ns))
                if key not in cache:
                    cache[key] = principal_blocks(*blocks[rep], tr, b)
                z, q, diag = cache[key][components]
                if ns:
                    z = np.c_[z, np.r_[np.ones(ns), np.zeros(len(a))]]
                    q = np.c_[q, np.zeros(len(b))]
                z = np.c_[z, np.ones(len(tr))]; q = np.c_[q, np.ones(len(b))]
                for alpha in ALPHAS:
                    for loss, cost in LOSSES:
                        coef, info = solve(z, (y-60)/10, w, loss, cost, alpha=alpha)
                        result.append(np.maximum(60+10*q@coef, 0))
                        keep = slice(ns, ns+n_main)
                        train_metric = detailed(y[keep], np.maximum(60+10*z[keep]@coef, 0))
                        audit.append(dict(**diag, **info, coef_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
                            n_source=ns, n_target=len(a), training_main_metrics=train_metric))
    controls, control_audit = control_predictions(data, blocks, cur, smooth, a, b, n_main)
    result.extend(controls); audit.extend(control_audit)
    return np.asarray(result), audit


def control_predictions(data, blocks, cur, smooth, a, b, n_main):
    result = []; audit = []
    for name, block, basis in [('full_base_amp6', blocks['base_amp6'], 'linear'), ('amp6_only', blocks['amp6'], 'rbf')]:
        z, q, y, w, diag = mapping.design(data, block, a, b, 'target_uniform', basis)
        coef, info = solve(z, (y-60)/10, w, 'control', 1., alpha=.001)
        result.append(np.maximum(60+10*q@coef, 0))
        audit.append(dict(control=name, **diag, **info, coef_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
            training_main_metrics=detailed(y[:n_main], np.maximum(60+10*z[:n_main]@coef, 0))))
    raw, diag = calibration.source.models(data, blocks, cur, smooth, a, b)
    means, counts = calibration.conditional_means(data, a)
    result.extend([raw[:, 0], calibration.predict(raw, BASE, means)])
    audit.extend([dict(control='current18_primary', **diag), dict(control='calibrated_low_rescue', **diag, means=means.tolist(), counts=counts)])
    return np.asarray(result), audit


def evaluate(data, blocks, cur, smooth, fold, extra, predictor=None, grid=None):
    predictor = predict_all if predictor is None else predictor
    grid = GRID if grid is None else grid
    a0, b = fold['A'], fold['B']; ad = extra[~np.isin(data['groups'][extra], data['groups'][b])]
    fixed, audit = predictor(data, blocks, cur, smooth, np.r_[a0, ad], b, len(a0))
    inner = np.full((len(grid), len(a0)), np.nan); pos = {int(v): j for j, v in enumerate(a0)}
    for f in fold['inner']:
        t, v = f['T'], f['V']; add = ad[~np.isin(data['groups'][ad], data['groups'][v])]
        pp, _ = predictor(data, blocks, cur, smooth, np.r_[t, add], v, len(t))
        inner[:, [pos[int(j)] for j in v]] = pp
    y = data['y'][a0, 2]; mm = [detailed(y, p) for p in inner]
    choices = dict(mse=int(np.argmin(np.mean((inner-y)**2, axis=1))),
        safety=min(range(len(grid)), key=lambda j: (selection_key(mm[j]), j)))
    return fixed, choices, audit, inner


def run(seeds, name, smoke=False):
    data, parts, extra, blocks, cur, hashes = mapping.load_inputs(seeds, larger=True)
    d2, p2, ex2, reps, mask, more = load_waveforms(seeds); hashes.update(more)
    assert np.array_equal(extra, ex2) and np.array_equal(parts[1000]['pool'], p2[1000]['pool'])
    out = ROOT/'outputs/uteg_mapping_target_runs'/name; (out/'private').mkdir(parents=True, exist_ok=True)
    protocol = dict(candidates=GRID, seeds=list(parts),
        scope='Same60 main development queries and historical40 training-only. Frozen3outer/3inner grouped folds. Original201 features/labels excluded.',
        question='Does reducing frozen waveform representation dimension improve abnormal generalization without throwing away absolute amplitude?',
        representation='Frozen MOMENT-base, MOMENT-small, or matched base-random embeddings; PCA2/4/8/16 on standardized actual TRAINING rows only. Cap rank at training rank/n-1. All6 amplitude features retained separately. SVD non-whitened PCs preserve relative variance; each projected embedding block is normalized to unit TRAINING RMS norm, combined equally with standardized amplitude block.',
        model='Linear constrained mapping with free intercept, optional HB source indicator, normalized Ridge penalty .001/.01/.1; ordinary MSE or boundary16 inward-abnormal-error surrogate. Existing training-only uniform/class-inverse/DenseWeight recipes unchanged.',
        controls='Exact full-base embedding+amp6, amp6-only, existing current18 primary, calibrated low rescue. Same MA used for all continuous/class metrics. No withheld cases.',
        isolation='Each inner split independently fits scaling/PCA/rank/normalization/weights/model. Inner predictions select hyperparameters, outer predictions evaluate. Training resubstitution logged only for diagnosis and never used to select.',
        extension='5seed screen extends if strict goal pass or agreement>80,FNR<10%,low recall>=80%, or R2>.35.',
        input_hashes=hashes, code_sha256=sha256(__file__),
        source='https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
        test_evaluated=False, gate_evaluated=False, manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    old = np.load(ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/private/PREDICTIONS.npz')
    od = json.loads((ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/RESULTS.json').read_text())
    oi = next(i for i, r in enumerate(od['fixed']) if r['candidate'] == BASE)
    prior = np.load(ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_base_mapping_screen_20261102/private/PREDICTIONS.npz')
    pd = json.loads((ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_base_mapping_screen_20261102/RESULTS.json').read_text())
    match = [next(i for i, r in enumerate(pd['fixed']) if r['candidate'] == dict(representation=rep,
        recipe='target_uniform', basis=basis, alpha=.001, loss='control', cost=1.))
        for rep, basis in [('base_amp6', 'linear'), ('amp6', 'rbf')]]
    pool = parts[1000]['pool']; pos = {int(v): j for j, v in enumerate(pool)}
    assert np.array_equal(old['pool'], pool) and np.array_equal(prior['pool'], pool)
    fixed = np.full((len(GRID), seeds, len(pool)), np.nan)
    nested = {k: np.full((seeds, len(pool)), np.nan) for k in ['mse', 'safety']}
    ledger = []; begin = time.time()
    for si, (seed, part) in enumerate(parts.items()):
        for fi, fold in enumerate(part['folds']):
            pred, choice, audit, inner = evaluate(data, blocks, cur, reps['smooth900'], fold, extra)
            ix = [pos[int(v)] for v in fold['B']]
            np.testing.assert_allclose(pred[-1], old['fixed'][oi, si, ix], rtol=0, atol=1e-7)
            if si < 5:
                for offset, number in [(-4, match[0]), (-3, match[1]), (-2, len(pd['fixed'])-1)]:
                    np.testing.assert_allclose(pred[offset], prior['fixed'][number, si, ix], rtol=0, atol=1e-7)
            if smoke:
                poison = dict(data); poison['y'] = data['y'].copy(); poison['y'][fold['B']] = np.nan
                changed = {k: [v.copy() for v in values] for k, values in blocks.items()}
                for values in changed.values():
                    for v in values: v[fold['B']] = 1e4
                cc = cur.copy(); cc[fold['B']] = 1e4
                ss = reps['smooth900'].copy(); ss[fold['B']] = 1e4
                _, ch2, au2, in2 = evaluate(poison, changed, cc, ss, fold, extra)
                assert ch2 == choice and au2 == audit
                np.testing.assert_array_equal(in2, inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True, all4controls_replayed=True), indent=2), encoding='utf-8')
                print('SMOKE_COMPLETE', round(time.time()-begin, 1), flush=True); return
            fixed[:, si, ix] = pred
            for key, j in choice.items(): nested[key][si, ix] = pred[j]
            (out/f'private/{seed}_{fi}_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
            ledger.append(dict(seed=seed, fold=fi, choice={k: GRID[j] for k, j in choice.items()}))
        print('LOW_RANK_SEED', seed, round(time.time()-begin, 1), flush=True)
    yy = np.broadcast_to(data['y'][pool, 2], (seeds, len(pool)))
    rows = [dict(candidate=c, metrics=detailed(yy, p)) for c, p in zip(GRID, fixed)]
    selected = [dict(objective=k, metrics=detailed(yy, p)) for k, p in nested.items()]
    for row, p in zip(rows+selected, list(fixed)+list(nested.values())):
        t, q = states(yy), states(p)
        assert [[int(((t==i)&(q==j)).sum()) for j in range(3)] for i in range(3)] == row['metrics']['confusion']
    for path, digest in hashes.items(): assert sha256(path) == digest
    np.savez_compressed(out/'private/PREDICTIONS.npz', fixed=fixed, reference=yy[0], pool=pool, seeds=list(parts), **nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    promising = [r for r in rows[:-4] if r['metrics']['mapping_target_pass'] or r['metrics']['r2'] > .35 or
        (r['metrics']['agreement'] > .8 and r['metrics']['fnr'] < .1 and r['metrics']['low_recall'] >= .8)]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows, nested=selected, promising=promising,
        checks=dict(controls_replayed=True, input_hashes_unchanged=True), test_evaluated=False, gate_evaluated=False), indent=2), encoding='utf-8')
    print('COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--output', default='low_rank_curve_mapping_screen_20261102'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1): run(a.seeds, a.output, a.smoke)
