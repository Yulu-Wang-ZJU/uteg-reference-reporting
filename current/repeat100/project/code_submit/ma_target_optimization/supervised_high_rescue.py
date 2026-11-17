"""Validate train-selected high-MA complements to the fixed low-rescue model."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import supervised_curve_mapping as source
from . import low_rank_curve_mapping as shared
from .feature_combination_audit import detailed
from .metrics import states
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

GRID = [dict(pls=c, scope=s) for c in source.GRID[:-4] for s in ['normal_only', 'all_nonhigh']]
GRID += [dict(control='calibrated_low_rescue')]
PINNED = dict(pls=dict(representation='base_amp6', target='ma_state', components=8,
    recipe='HB_dense', alpha=.01, loss='control', cost=1.), scope='normal_only')


def combine(base, prediction, scope):
    allowed = (states(base)==1) if scope=='normal_only' else (states(base)!=2)
    if scope not in ['normal_only', 'all_nonhigh']: raise ValueError(scope)
    out = base.copy(); take = allowed & (prediction>70)
    out[take] = prediction[take]
    return out


def combine_all(raw):
    base = raw[-1]
    return np.asarray([combine(base, p, scope) for p in raw[:-4]
        for scope in ['normal_only', 'all_nonhigh']] + [base.copy()])


def predict_all(data, blocks, cur, smooth, a, b, n_main):
    raw, audit = source.predict_all(data, blocks, cur, smooth, a, b, n_main)
    return combine_all(raw), audit


def evaluate(data, blocks, cur, smooth, fold, extra):
    return shared.evaluate(data, blocks, cur, smooth, fold, extra, predictor=predict_all, grid=GRID)


def run(seeds, name, smoke=False):
    data, parts, extra, blocks, cur, hashes = shared.mapping.load_inputs(seeds, larger=True)
    d2, p2, ex2, reps, mask, more = shared.load_waveforms(seeds); hashes.update(more)
    assert np.array_equal(extra, ex2) and np.array_equal(parts[1000]['pool'], p2[1000]['pool'])
    out = ROOT/'outputs/uteg_mapping_target_runs'/name; (out/'private').mkdir(parents=True, exist_ok=True)
    protocol = dict(candidates=GRID, pinned_development_candidate=PINNED, seeds=list(parts),
        origin='PLS standalone5seed screen was negative, but a post-screen complementarity diagnostic identified this fixed normal-only high-rescue candidate. Adaptive development choice, not preregistered or external validation.',
        scope='Same60 main grouped development queries, extra40 training-only; original201 excluded.20 repeated grouped partitions, first5 exactly replayed. Repeated partitions are not new patients.',
        rule='Start with existing calibrated low rescue. Accept actual continuous PLS output only if >70, applied to current normal only or all non-high. Existing high outputs preserved. No label-dependent query rules, threshold shifts, clipping to70.1, or withholding.',
        selection='Each outer training fold obtains complete inner3fold predictions, then selects among432complements plus unchanged control by MSE or safety. Each inner PLS, target scaling, weighting and MA head fitted only on its own training rows.',
        goal='All-report3class agreement>80,FNR<5,FNR<FPR; eventual trust-gate coverage>75. All opposite-tail errors and numerical MA errors retained.',
        input_hashes=hashes, code_sha256=sha256(__file__), pls_code_sha256=sha256(source.__file__),
        test_evaluated=False, gate_evaluated=False, manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    previous = np.load(ROOT/'outputs/uteg_mapping_target_runs/supervised_curve_mapping_screen_20261102/private/PREDICTIONS.npz')
    old_rescue = np.load(ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/private/PREDICTIONS.npz')
    rd = json.loads((ROOT/'outputs/uteg_mapping_target_runs/specialist_boundary_calibration_20seed_20261102/RESULTS.json').read_text())
    ri = next(i for i, r in enumerate(rd['fixed']) if r['candidate']==shared.BASE)
    early = np.asarray([combine_all(previous['fixed'][:, si]) for si in range(5)]).transpose(1, 0, 2)
    pool = parts[1000]['pool']; assert np.array_equal(previous['pool'], pool) and np.array_equal(old_rescue['pool'], pool)
    pos = {int(v): j for j, v in enumerate(pool)}
    fixed = np.full((len(GRID), seeds, len(pool)), np.nan)
    nested = {k: np.full((seeds, len(pool)), np.nan) for k in ['mse', 'safety']}; ledger = []; begin = time.time()
    for si, (seed, part) in enumerate(parts.items()):
        for fi, fold in enumerate(part['folds']):
            pred, choice, audit, inner = evaluate(data, blocks, cur, reps['smooth900'], fold, extra)
            ix = [pos[int(v)] for v in fold['B']]
            if si<5: np.testing.assert_allclose(pred, early[:, si, ix], rtol=0, atol=1e-7)
            np.testing.assert_allclose(pred[-1], old_rescue['fixed'][ri, si, ix], rtol=0, atol=1e-7)
            if smoke:
                poison = dict(data); poison['y'] = data['y'].copy(); poison['y'][fold['B']] = np.nan
                changed = {k: [v.copy() for v in values] for k, values in blocks.items()}
                for values in changed.values():
                    for v in values: v[fold['B']] = 1e4
                cc = cur.copy(); cc[fold['B']] = 1e4; ss = reps['smooth900'].copy(); ss[fold['B']] = 1e4
                _, ch2, au2, in2 = evaluate(poison, changed, cc, ss, fold, extra)
                assert ch2==choice and au2==audit; np.testing.assert_array_equal(in2, inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True, first5_candidates_replayed=True, baseline_replayed=True), indent=2), encoding='utf-8')
                print('SMOKE_COMPLETE', round(time.time()-begin, 1), flush=True); return
            fixed[:, si, ix] = pred
            for key, j in choice.items(): nested[key][si, ix] = pred[j]
            ledger.append(dict(seed=seed, fold=fi, choice={k: GRID[j] for k, j in choice.items()}))
        print('PLS_RESCUE_SEED', seed, round(time.time()-begin, 1), flush=True)
    yy = np.broadcast_to(data['y'][pool, 2], (seeds, len(pool)))
    rows = [dict(candidate=c, metrics=detailed(yy, p)) for c, p in zip(GRID, fixed)]
    selected = [dict(objective=k, metrics=detailed(yy, p)) for k, p in nested.items()]
    for row, pred in zip(rows+selected, list(fixed)+list(nested.values())):
        t, q = states(yy), states(pred)
        assert [[int(((t==i)&(q==j)).sum()) for j in range(3)] for i in range(3)]==row['metrics']['confusion']
    for path, digest in hashes.items(): assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz', fixed=fixed, reference=yy[0], pool=pool, seeds=list(parts), **nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    pi = GRID.index(PINNED)
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows, nested=selected, pinned=rows[pi],
        strict_passes=[r for r in rows if r['metrics']['mapping_target_pass']],
        checks=dict(first5_candidates_replayed=True, baseline_replayed=True, input_hashes_unchanged=True),
        test_evaluated=False, gate_evaluated=False), indent=2), encoding='utf-8')
    print('COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--output', default='supervised_high_rescue_20seed_20261102'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    with threadpool_limits(limits=1): run(a.seeds, a.output, a.smoke)
