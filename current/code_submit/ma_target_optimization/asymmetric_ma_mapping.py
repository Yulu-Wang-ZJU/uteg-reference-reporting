"""Continuous MA mapping with explicit target-side error-direction costs.

This is a study-specific use of asymmetric least squares and squared hinges,
not a reproduction of a published clinical model or a population-FNR guarantee.
Only approved training labels determine weights and error directions.
"""
import argparse
import hashlib
import itertools
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .cost_sensitive_detection import assemble
from .density_reweight_mapping import fit as control, label_weights
from .hubei_source_search import add_hubei_arm
from .learning_curve import load_extra
from .metrics import assess, selection_key
from .models import Features
from .search import ROOT, training_view
from .voltage_scale import PooledVoltage, load_inputs
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def loss_terms(prediction, reference, target, kind, cost):
    """Return per-row loss, d/d(prediction), and half the curvature.

    All coordinates are in (MA-60)/10 units. The residual mode multiplies only
    target-low overestimation / target-high underestimation by cost. The hinge
    mode adds a cost-weighted penalty for crossing the correct abnormal-side
    boundary (0.5 mm inward margin), and unit-cost normal-to-abnormal penalties.
    Wrong-tail predictions remain penalized; they are never rewarded as merely
    'abnormal'. The base MA residual remains present for every record.
    """
    p, y, target = np.asarray(prediction), np.asarray(reference), np.asarray(target, bool)
    if kind not in ('control', 'residual', 'boundary') or cost < 1:
        raise ValueError('Unknown loss or cost below one')
    r = p-y
    value, derivative, curvature = r*r, 2*r, np.ones(len(y))
    if kind == 'control':
        return value, derivative, curvature
    low, high = target & (y < -1), target & (y > 1)
    if kind == 'residual':
        risk = (low & (r > 0)) | (high & (r < 0))
        factor = 1+(cost-1)*risk
        return factor*value, factor*derivative, factor*curvature
    # Surrogate for missed low/high states, including opposite-tail mistakes.
    # It is deliberately not called an exact binary 0-1 FN loss.
    normal = target & ~low & ~high
    for mask, sign, boundary, multiplier in (
            (low, 1., -1.05, cost), (high, -1., 1.05, cost),
            (normal, -1., -1., 1.), (normal, 1., 1., 1.)):
        violation = np.maximum(sign*(p-boundary), 0)*mask
        value += multiplier*violation**2
        derivative += 2*multiplier*sign*violation
        curvature += multiplier*(violation > 0)
    return value, derivative, curvature


def objective(coef, design, y, w, target, kind, cost):
    value, derivative, curvature = loss_terms(design@coef, y, target, kind, cost)
    penalty = np.r_[np.ones(len(coef)-1)*.01, 0.]+1e-12/len(y)
    loss = float(np.mean(w*value)+np.sum(penalty*coef**2))
    gradient = design.T@(w*derivative)/len(y)+2*penalty*coef
    hessian = 2*design.T@((w*curvature)[:, None]*design)/len(y)+2*np.diag(penalty)
    return loss, gradient, hessian


def solve(design, y, w, target, kind, cost, initial):
    """Damped piecewise-quadratic Newton, independently checked stationarity."""
    coef = initial.copy()
    for iteration in range(200):
        loss, gradient, hessian = objective(coef, design, y, w, target, kind, cost)
        if np.max(np.abs(gradient)) < 1e-8:
            break
        direction = np.linalg.solve(hessian, -gradient)
        slope = float(gradient@direction)
        if slope >= 0:
            raise RuntimeError('Non-descent Newton direction')
        step = 1.
        for _ in range(50):
            proposed = coef+step*direction
            next_loss = objective(proposed, design, y, w, target, kind, cost)[0]
            if next_loss <= loss+1e-4*step*slope+1e-14:
                coef = proposed
                break
            step *= .5
        else:
            raise RuntimeError('Line search failed')
    else:
        raise RuntimeError('Direction-sensitive fit did not converge')
    loss, gradient, hessian = objective(coef, design, y, w, target, kind, cost)
    if np.max(np.abs(gradient)) >= 1e-7 or not np.isfinite(coef).all():
        raise RuntimeError('Unverified stationarity')
    return coef, dict(iterations=iteration, objective=loss,
                      gradient_max=float(np.max(np.abs(gradient))), stationary_verified=True)


def fit(train, c):
    if c['kind'] == 'control':
        return control(train, c)
    x, y, codes, blocks = assemble(train, c['arm'])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training data required')
    w = np.empty(len(y)); details = []
    for block in np.unique(blocks):
        keep = blocks == block
        source = c['arm'] != 'target_only' and block == 0
        local, info = label_weights(y[keep], c['weight_method'], 5. if source else 20.)
        w[keep] = local/keep.sum()
        details.append(dict(role='source' if source else 'target', **info))
    # These are the original fixed base weights. Do not normalize away the new
    # directional cost or let query outcomes change domain/regularization mass.
    w /= w.mean()
    transform = Features.fit(x, 'linear')
    design = np.c_[transform.transform(x), codes, np.ones(len(x))]
    penalty = np.diag(np.r_[np.ones(design.shape[1]-1), 0.])
    initial = np.linalg.solve(design.T@(w[:, None]*design)+len(y)*.01*penalty+
                              1e-12*np.eye(design.shape[1]), design.T@(w*(y-60)/10))
    target = np.arange(len(y)) >= len(y)-len(train['x'])
    coef, info = solve(design, (y-60)/10, w, target, c['kind'], c['cost'], initial)
    info.update(weights=details, target_n=len(train['x']), source_n=len(y)-len(train['x']),
                cost=c['cost'], kind=c['kind'], directional_cost_scope='target_training_only',
                coef_hash=hashlib.sha256(coef.tobytes()).hexdigest())
    return PooledVoltage(transform, coef, codes.shape[1], info)


def grid():
    candidates = []
    losses = [('control', 1.)]+[('residual', v) for v in (2., 4., 8., 16., 32.)]+[
        ('boundary', v) for v in (1., 4., 16., 64.)]
    for arm, method, budget in itertools.product(('target_only', 'HB_to_SA'),
                                               ('class_inverse', 'denseweight'), (60, 100)):
        for kind, cost in losses:
            c = dict(family='direction_sensitive_continuous_MA', arm=arm, weight_method=method,
                     budget=budget, kind=kind, cost=cost, alpha=.01, horizon=300, voltage_mode='p5')
            c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
            candidates.append(c)
    return candidates


def run(name, seeds=20, requested=None):
    out = ROOT/'outputs/uteg_mapping_target_runs'/name
    (out/'private').mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1000+seeds))
    data, parts, hashes = load_development_inputs(ROOT, cfg)
    extra, more = load_extra(data, parts, cfg); hashes.update(more)
    add_hubei_arm(data, parts)
    matrices, audit, more = load_inputs(data, parts); hashes.update(more)
    data['x'] = matrices[(300, 'p5')]
    pool, test = parts[1000]['pool'], parts[1000]['test']
    assert np.isnan(data['x'][test]).all() and np.isnan(data['y'][test]).all()
    candidates = grid()
    if requested is not None:
        candidates = [c for c in candidates if c['id'] in requested]
        assert len(candidates) == len(requested)
    protocol = dict(candidates=candidates, seeds=cfg['seeds'], input_hashes=hashes,
                    reference_confirmed_by_user='Instrument-exported TEG parameters at all three centres; correct pairing; UTEG900 samples at1s and TEG540 samples at5s confirmed.',
                    scope='Same grouped60-development queries, 79 approved Hubei source rows; C60 and C100 budgets separate. Original201 test excluded. No gate.',
                    source_anchor='Source loss remains symmetric; new directional costs apply to target-training rows only.',
                    residual='Base weighted squared MA residual; multiply target-low overestimation and target-high underestimation by cost. Other residual directions unchanged.',
                    boundary='Base weighted squared MA residual plus target squared hinges: cost*(low relu(pred-49.5)^2+high relu(70.5-pred)^2)+normal relu(50-pred)^2+normal relu(pred-70)^2; all MA distances divided by10.',
                    interpretation='Smooth convex surrogates prioritizing abnormal-to-normal errors; also penalize low/high swaps. They do not equal or guarantee binaryFNR. Not vanilla fixed-tau expectile regression.',
                    comparison='Base class_inverse/DenseWeight train-only weights, equal base source/target mass, same feature scaling, source offset, ridge.01 and fixed50/70 boundaries. Increased cost intentionally changes the relative objective, never post-hoc class thresholds.',
                    budget='C100 adds only historical40 development rows, excluding query groups. No extra lowMA records. No cross-budget method claim.',
                    selection='All declared costs reported on the same development folds; summaries are adaptive development comparisons, not nested unbiased model-selection estimates or untouched validation.',
                    acceptance='All-report3class agreement>.8, abnormalFNR<.05 andFNR<FPR. Gate coverage>.75 remains a separate requirement.',
                    references=['https://ijcai.org/proceedings/2001/133', 'https://doi.org/10.2307/1911031'],
                    code_hashes={str(p): sha256(p) for p in [__file__, ROOT/'code_submit/ma_target_optimization/density_reweight_mapping.py', ROOT/'code_submit/ma_target_optimization/metrics.py', ROOT/'code_submit/ma_target_optimization/learning_curve.py', ROOT/'code_submit/ma_target_optimization/voltage_scale.py']})
    content = json.dumps(protocol, indent=2)
    if (out/'PROTOCOL.json').exists():
        assert (out/'PROTOCOL.json').read_text(encoding='utf-8') == content
    (out/'PROTOCOL.json').write_text(content, encoding='utf-8')
    (out/'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    positions = {int(v): j for j, v in enumerate(pool)}
    result_path = out/'CANDIDATES.jsonl'
    rows = [json.loads(s) for s in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
    for c in candidates:
        if any(r['candidate']['id'] == c['id'] for r in rows):
            continue
        start = time.time(); predictions = []; diagnostics = []; train_y = []; train_p = []
        for seed, part in parts.items():
            pred = np.full(len(pool), np.nan)
            for number, fold in enumerate(part['folds']):
                b = fold['B']
                add = extra[~np.isin(data['groups'][extra], data['groups'][b])] if c['budget'] == 100 else np.array([], int)
                a = np.r_[fold['A'], add]
                assert len(np.unique(a)) == len(a)
                assert not set(data['groups'][a]).intersection(data['groups'][b])
                assert not set(a).intersection(test)
                model = fit(training_view(data, a), c)
                query = model.predict(data['x'][b]); pred[[positions[int(i)] for i in b]] = query
                train_y.append(data['y'][a, 2]); train_p.append(model.predict(data['x'][a]))
                if seed == 1000 and number == 0:
                    poisoned = dict(data); poisoned['x'] = data['x'].copy(); poisoned['y'] = data['y'].copy()
                    poisoned['x'][b] = 1e9; poisoned['y'][b] = np.nan
                    alternate = fit(training_view(poisoned, a), c)
                    np.testing.assert_array_equal(model.coef, alternate.coef)
                    np.testing.assert_allclose(query, [model.predict(data['x'][i:i+1])[0] for i in b], atol=1e-10, rtol=0)
                diagnostics.append(dict(seed=seed, fold=number, extra_n=len(add),
                                        training_rows_sha256=hashlib.sha256(a.tobytes()).hexdigest(), **model.diagnostics))
            assert np.isfinite(pred).all(); predictions.append(pred)
        pred = np.array(predictions); y = data['y'][pool, 2]
        m = assess(np.broadcast_to(y, pred.shape), pred)
        fn, fp = m['false_negative_count'], m['false_positive_count']
        m['binary_confusion_rows_normal_abnormal'] = [[m['within_n']-fp, fp], [fn, m['low_n']+m['high_n']-fn]]
        np.savez_compressed(out/'private'/f"{c['id']}.npz", pred=pred, reference=y, pool=pool, seeds=cfg['seeds'])
        (out/'private'/f"{c['id']}_folds.json").write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        row = dict(candidate=c, status='complete', metrics=m, training=assess(np.concatenate(train_y), np.concatenate(train_p)),
                   heldout_poison_invariant=True, query_batch_invariant=True, test_evaluated=False, gate_evaluated=False,
                   seconds=time.time()-start)
        rows.append(row)
        with result_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(done=len(rows), total=len(candidates), budget=c['budget'], arm=c['arm'], method=c['weight_method'], kind=c['kind'], cost=c['cost'],
                              **{k: m[k] for k in ('agreement', 'fnr', 'fpr', 'low_recall', 'high_recall', 'mae')})), flush=True)
    for path, h in hashes.items():
        assert sha256(path) == h
    summary = dict(complete=len(rows), declared=len(candidates), mapping_joint_passes=sum(r['metrics']['mapping_target_pass'] for r in rows),
                   input_hashes_unchanged=True, heldout_poison_checks=len(rows), test_evaluated=False, gate_evaluated=False,
                   best_by_budget={str(b): sorted([r for r in rows if r['candidate']['budget'] == b], key=lambda r: selection_key(r['metrics']))[:5] for b in (60, 100)})
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='goal_asymmetric_ma_v1')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.seeds, args.ids)
