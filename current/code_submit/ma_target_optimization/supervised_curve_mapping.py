"""Train-only PLS directions for a constrained waveform-to-MA regression."""
import argparse
import hashlib
import itertools
import json
import time
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from . import low_rank_curve_mapping as shared
from .full_curve_cost_sensitive import solve
from .feature_combination_audit import detailed
from .metrics import states
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

REPS = ['base_amp6', 'embedding_amp6', 'base_random_amp6']
TARGETS = ['ma', 'ma_state']
DIMS = [2, 4, 8]
ALPHAS = [.001, .01]
GRID = [dict(representation=r, target=t, components=k, recipe=w, alpha=a, loss=l, cost=c)
        for r, t, k, w, a, (l, c) in itertools.product(REPS, TARGETS, DIMS, shared.RECIPES, ALPHAS, shared.LOSSES)]
GRID += [dict(control=c) for c in shared.CONTROLS]


def pls_blocks(embedding, amplitude, reference, train, query, target, dimensions=DIMS):
    em = embedding[train].mean(0); es = np.maximum(embedding[train].std(0), 1e-6)
    x = (embedding[train]-em)/es; q = (embedding[query]-em)/es
    y = np.asarray(reference)[train]
    if target == 'ma': yy = y[:, None]
    elif target == 'ma_state': yy = np.c_[y, (y<50).astype(float), (y>70).astype(float)]
    else: raise ValueError(target)
    ym = yy.mean(0); ys = np.maximum(yy.std(0), 1e-6); yy = (yy-ym)/ys
    am = amplitude[train].mean(0); ass = np.maximum(amplitude[train].std(0), 1e-6)
    ax = (amplitude[train]-am)/ass/np.sqrt(amplitude.shape[1])
    aq = (amplitude[query]-am)/ass/np.sqrt(amplitude.shape[1])
    answer = {}
    for wanted in dimensions:
        k = min(wanted, len(train)-1, embedding.shape[1])
        model = PLSRegression(n_components=k, scale=False, max_iter=2000, tol=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x, yy)
        tx = model.transform(x); qq = model.transform(q)
        norm = max(float(np.sqrt(np.mean(np.sum(tx*tx, axis=1)))), 1e-12)
        z = np.c_[tx/norm, ax]/np.sqrt(2); zq = np.c_[qq/norm, aq]/np.sqrt(2)
        assert np.isfinite(z).all() and np.isfinite(zq).all()
        digest = hashlib.sha256(em.tobytes()+es.tobytes()+am.tobytes()+ass.tobytes()+ym.tobytes()+ys.tobytes()+
                                model.x_rotations_.tobytes()+np.asarray(norm).tobytes()).hexdigest()
        answer[wanted] = (z, zq, dict(components=k, target=target, transform_sha256=digest,
            pls_iterations=[int(v) for v in model.n_iter_], effective_components=int(np.sum(np.std(tx, axis=0)>1e-10))))
    return answer


def predict_all(data, blocks, cur, smooth, a, b, n_main):
    result = []; audit = []; cache = {}
    for rep in REPS:
        for target in TARGETS:
            for components in DIMS:
                for recipe in shared.RECIPES:
                    tr, ns, y, w = shared.training_rows_weights(data, a, recipe)
                    assert not set(data['groups'][tr]).intersection(data['groups'][b])
                    key = (rep, target, bool(ns))
                    if key not in cache:
                        cache[key] = pls_blocks(*blocks[rep], data['y'][:, 2], tr, b, target)
                    z, q, diag = cache[key][components]
                    if ns:
                        z = np.c_[z, np.r_[np.ones(ns), np.zeros(len(a))]]; q = np.c_[q, np.zeros(len(b))]
                    z = np.c_[z, np.ones(len(tr))]; q = np.c_[q, np.ones(len(b))]
                    for alpha in ALPHAS:
                        for loss, cost in shared.LOSSES:
                            coef, info = solve(z, (y-60)/10, w, loss, cost, alpha=alpha)
                            result.append(np.maximum(60+10*q@coef, 0))
                            keep = slice(ns, ns+n_main)
                            audit.append(dict(**diag, **info, n_source=ns, n_target=len(a),
                                coef_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),
                                training_main_metrics=detailed(y[keep], np.maximum(60+10*z[keep]@coef, 0))))
    controls, control_audit = shared.control_predictions(data, blocks, cur, smooth, a, b, n_main)
    result.extend(controls); audit.extend(control_audit)
    return np.asarray(result), audit


def evaluate(data, blocks, cur, smooth, fold, extra):
    return shared.evaluate(data, blocks, cur, smooth, fold, extra, predictor=predict_all, grid=GRID)


def run(seeds, name, smoke=False):
    data, parts, extra, blocks, cur, hashes = shared.mapping.load_inputs(seeds, larger=True)
    d2, p2, ex2, reps, mask, more = shared.load_waveforms(seeds); hashes.update(more)
    assert np.array_equal(extra, ex2) and np.array_equal(parts[1000]['pool'], p2[1000]['pool'])
    out = ROOT/'outputs/uteg_mapping_target_runs'/name; (out/'private').mkdir(parents=True, exist_ok=True)
    protocol = dict(candidates=GRID, seeds=list(parts),
        scope='Same60 main grouped development queries; historical40 training-only; original201 features/labels excluded.',
        question='Does supervised low-dimensional curve projection preserve predictive tail information better than variance-based PCA?',
        projection='sklearn PLSRegression(scale=False,n_components2/4/8,max_iter2000,tol1e-9), fit separately at each dimension on actual training rows only. Inputs train-standardized; targets train-standardized MA alone or MA+low/high indicators. Projection is unweighted; subsequent mapping uses the same preexisting loss weights. Query reference MA never enters projection.',
        head='PLS scores retain relative variance and use training RMS block normalization;6 raw amplitude features separately retained, equal block mass. Linear Ridge .001/.01 or boundary16 surrogate, free intercept/source indicator. Same continuous output used for state metrics.',
        isolation='Each inner fold independently refits PLS including all supervised target scaling, loss weights, and head. Inner predictions select model; outer labels only evaluate. Existing4controls exactly matched. Training metrics diagnostic only.',
        extension='5seed screen extends if strict goal pass or agreement>80,FNR<10%,low recall>=80%, or R2>.35.',
        input_hashes=hashes, code_sha256=sha256(__file__), shared_code_sha256=sha256(shared.__file__),
        source='https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html',
        test_evaluated=False, gate_evaluated=False, manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    old = np.load(ROOT/'outputs/uteg_mapping_target_runs/low_rank_curve_mapping_screen_20261102/private/PREDICTIONS.npz')
    rescue = np.load(ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/private/PREDICTIONS.npz')
    rd = json.loads((ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/RESULTS.json').read_text())
    ri = next(i for i, r in enumerate(rd['fixed']) if r['candidate'] == shared.BASE)
    pool = parts[1000]['pool']; assert np.array_equal(old['pool'], pool) and np.array_equal(rescue['pool'], pool)
    pos = {int(v): j for j, v in enumerate(pool)}
    fixed = np.full((len(GRID), seeds, len(pool)), np.nan)
    nested = {k: np.full((seeds, len(pool)), np.nan) for k in ['mse', 'safety']}; ledger = []; begin = time.time()
    for si, (seed, part) in enumerate(parts.items()):
        for fi, fold in enumerate(part['folds']):
            pred, choice, audit, inner = evaluate(data, blocks, cur, reps['smooth900'], fold, extra)
            ix = [pos[int(v)] for v in fold['B']]
            if si < 5: np.testing.assert_allclose(pred[-4:], old['fixed'][-4:, si, ix], rtol=0, atol=1e-7)
            np.testing.assert_allclose(pred[-1], rescue['fixed'][ri, si, ix], rtol=0, atol=1e-7)
            if smoke:
                poison = dict(data); poison['y'] = data['y'].copy(); poison['y'][fold['B']] = np.nan
                changed = {k: [v.copy() for v in values] for k, values in blocks.items()}
                for values in changed.values():
                    for v in values: v[fold['B']] = 1e4
                cc = cur.copy(); cc[fold['B']] = 1e4; ss = reps['smooth900'].copy(); ss[fold['B']] = 1e4
                _, ch2, au2, in2 = evaluate(poison, changed, cc, ss, fold, extra)
                assert ch2 == choice and au2 == audit
                np.testing.assert_array_equal(in2, inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True, all4controls_replayed=True), indent=2), encoding='utf-8')
                print('SMOKE_COMPLETE', round(time.time()-begin, 1), flush=True); return
            fixed[:, si, ix] = pred
            for key, j in choice.items(): nested[key][si, ix] = pred[j]
            (out/f'private/{seed}_{fi}_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
            ledger.append(dict(seed=seed, fold=fi, choice={k: GRID[j] for k, j in choice.items()}))
        print('PLS_SEED', seed, round(time.time()-begin, 1), flush=True)
    yy = np.broadcast_to(data['y'][pool, 2], (seeds, len(pool)))
    rows = [dict(candidate=c, metrics=detailed(yy, p)) for c, p in zip(GRID, fixed)]
    selected = [dict(objective=k, metrics=detailed(yy, p)) for k, p in nested.items()]
    for row, pred in zip(rows+selected, list(fixed)+list(nested.values())):
        t, q = states(yy), states(pred)
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
    p.add_argument('--output', default='supervised_curve_mapping_screen_20261102'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    with threadpool_limits(limits=1): run(a.seeds, a.output, a.smoke)
