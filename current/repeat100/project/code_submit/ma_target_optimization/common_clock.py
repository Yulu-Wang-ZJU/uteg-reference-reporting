"""Paired fixed-clock mapping experiment with causal, uniform preprocessing.

The 180-sample prefix is observed in every permitted record. Longer prefixes
retain last-value padding and its observed fraction. No extra measurements,
reference outputs, second channels, or test records enter these features.
"""
import argparse
import ast
import hashlib
import json
import time
import numpy as np
from scipy.signal import savgol_filter
from threadpoolctl import threadpool_limits
from .search import ROOT, training_view, make_summary
from .duration_transfer import load_features as load_frozen, TIME_COLUMNS, GROWTH_COLUMNS
from .signal_neighborhood import kinetics
from .hubei_source_search import add_hubei_arm
from .models import fit_candidate
from .metrics import assess
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def load_preprocessor():
    path = ROOT/'outputs/uteg_platform_filter_pilot_20261006_124636/process.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply']
    assert len(nodes) == 1
    namespace = {'np': np, 'savgol_filter': savgol_filter}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['apply'], path


def prefix_features(raw, observed, horizon, preprocess):
    raw, observed = np.asarray(raw, float), np.asarray(observed, bool)
    if raw.shape != observed.shape or raw.ndim != 1 or horizon not in (180, 300, 900):
        raise ValueError('Invalid input shape or horizon')
    # All operations, including the endpoint-dependent P5 rule, see the prefix only.
    visible = observed[:horizon]
    n = int(visible.sum())
    if n < 11 or not np.array_equal(np.flatnonzero(visible), np.arange(n)):
        raise ValueError('Expected an observed contiguous prefix')
    values = raw[:n]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite observed signal')
    processed, _ = preprocess(values, q=.05)
    completed = np.pad(processed, (0, horizon-n), mode='edge')
    edges = np.linspace(0, horizon, 91).astype(int)
    wave = np.array([np.median(completed[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    f = kinetics(wave)
    f[TIME_COLUMNS] *= horizon/900.
    f[GROWTH_COLUMNS] *= 900./horizon
    return np.r_[f, n/horizon]


def load_inputs(data, parts):
    existing, lengths, hashes = load_frozen(data, parts)
    raw_path = ROOT/'outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz'
    preprocess, prep_path = load_preprocessor()
    matrices = {'frozen_padded900': existing['padded_with_length18']}
    for horizon in (180, 300, 900):
        matrices[f'common{horizon}'] = np.full((len(data['x']), 18), np.nan)
    rows = data['allowed_training_indices']
    with np.load(raw_path, allow_pickle=False) as z:
        assert np.array_equal(z['ids'], data['ids'])
        assert np.array_equal(z['center'], data['centers'])
        # Load only approved rows and the approved first channel in this archive.
        raw, observed = z['original'][rows], z['observed'][rows]
    assert np.all(observed.sum(1) == lengths[rows]) and int(observed.sum(1).min()) >= 180
    for row, signal, mask in zip(rows, raw, observed):
        for horizon in (180, 300, 900):
            matrices[f'common{horizon}'][row] = prefix_features(signal, mask, horizon, preprocess)
    pool = next(iter(parts.values()))['pool']
    # The target full-window preprocessing must replay the earlier target control.
    delta = float(np.max(np.abs(matrices['common900'][pool]-matrices['frozen_padded900'][pool])))
    assert delta < 1e-10, delta
    test = next(iter(parts.values()))['test']
    assert all(np.isnan(x[test]).all() for x in matrices.values())
    audit = {'all_180_samples_observed': True, 'target900_replay_max_difference': delta,
             'input_only_first_approved_channel': True, 'test_features_all_nan': True,
             'by_center': {}}
    for center in np.unique(data['centers'][rows]):
        ix = rows[data['centers'][rows] == center]
        audit['by_center'][str(center)] = {
            'n': len(ix), 'padded_records': {str(h): int(np.sum(lengths[ix] < h)) for h in (180, 300, 900)},
            'uniform900_vs_frozen_feature_max_difference': float(np.max(np.abs(matrices['common900'][ix]-matrices['frozen_padded900'][ix])))}
    hashes.update({str(p): sha256(p) for p in (raw_path, prep_path)})
    return matrices, audit, hashes


def grid():
    base = [dict(family='ridge', arm='target_only', features='linear', alpha=.01, cap=cap) for cap in (1., 20.)]
    base.append(dict(family='boundary', arm='target_only', features='linear', alpha=.01, cap=5., boundary_strength=4., abnormal_cost=16.))
    base += [dict(family='transfer', arm=arm, mode='joint', source_alpha=.1, source_cap=5., cap=20., adapter_penalty=10.)
             for arm in ('HZ_to_SA', 'HB_to_SA', 'HZ+HB_to_SA')]
    candidates = []
    for representation in ('frozen_padded900', 'common900', 'common300', 'common180'):
        for model in base:
            c = {**model, 'representation': representation}
            c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
            candidates.append(c)
    return candidates


def run(out, seed_count, ids=None):
    out.mkdir(parents=True, exist_ok=True); (out/'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1000+seed_count))
    data, parts, hashes = load_development_inputs(ROOT, cfg); add_hubei_arm(data, parts)
    matrices, audit, more = load_inputs(data, parts); hashes.update(more)
    candidates = grid()
    if ids is not None:
        candidates = [c for c in candidates if c['id'] in ids]
        assert len(candidates) == len(ids), 'Unknown or repeated candidate ids'
    protocol = {'candidates': candidates, 'seeds': cfg['seeds'], 'input_hashes': hashes,
                'code_hashes': {name: sha256(ROOT/'code_submit/ma_target_optimization'/name) for name in
                               ('common_clock.py', 'duration_transfer.py', 'signal_neighborhood.py', 'models.py', 'metrics.py')},
                'mapping_code_hash': sha256(ROOT/'code_submit/ma_imbalance_mapping/models.py'),
                'hypothesis': 'center/class-dependent observation duration may confound mapping; compare fixed clock prefixes with uniform prefix-only preprocessing',
                'scope': 'same target60 grouped folds and approved source174; development selection only; no test labels or features',
                'selection': 'fixed24-candidate screen' if ids is None else 'selected300-sample candidates and matched900-sample controls from the5-seed screen; same20-seed confirmation',
                'first_channel_only': True, 'second_channel': 'ignored as requested by user',
                'preprocessing': 'existing baseline10/SG11-order2/rezero/P5 rule applied identically per available prefix before padding',
                'representations': 'frozen900 control; uniformly processed900; uniformly processed300; all-observed180',
                'padding': 'last observed processed value; retain observed fraction; no additional real observation claimed',
                'time_features': 'actual seconds and growth per minute; relative third/two-thirds amplitude features refer to the chosen horizon',
                'caveat': 'device signal clock from prior1s/sample convention; not clinical collection-to-report time; no horizon selected on test outcomes',
                'controls': 'frozen900 vs uniform900 isolates preprocessing; uniform180/300 vs uniform900 holds preprocessing algorithm and models fixed'}
    text = json.dumps(protocol, indent=2, allow_nan=False)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8') != text:
        raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text, encoding='utf-8')
    (out/'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    path = out/'CANDIDATES.jsonl'
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {r['candidate']['id'] for r in records}
    caches = {rep: {} for rep in matrices}
    pool = next(iter(parts.values()))['pool']; pos = {int(v): i for i, v in enumerate(pool)}
    for c in candidates:
        if c['id'] in done: continue
        start = time.time(); data['x'] = matrices[c['representation']]
        try:
            predictions = []
            for seed, part in parts.items():
                p = np.full(len(pool), np.nan)
                for fold in part['folds']:
                    A, B = fold['A'], fold['B']
                    assert not set(data['groups'][A]) & set(data['groups'][B])
                    model = fit_candidate(c, training_view(data, A), caches[c['representation']])
                    p[[pos[int(v)] for v in B]] = model.predict(data['x'][B])
                assert np.isfinite(p).all()
                predictions.append(p)
            p = np.asarray(predictions); y = data['y'][pool, 2]
            np.savez_compressed(out/'private'/f"{c['id']}.npz", prediction=p, reference=y, pool=pool, seeds=cfg['seeds'])
            row = {'candidate': c, 'status': 'complete', 'metrics': assess(np.broadcast_to(y, p.shape), p), 'seconds': time.time()-start}
        except Exception as error:
            row = {'candidate': c, 'status': 'failed', 'error': str(error), 'seconds': time.time()-start}
        records.append(row)
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps({'done': len(records), 'total': len(candidates), 'representation': c['representation'],
                          'arm': c['arm'], 'family': c['family'], 'status': row['status'],
                          **({k: row['metrics'][k] for k in ('agreement','fnr','low_recall','high_recall')} if row['status']=='complete' else {'error': row['error']})}), flush=True)
    return make_summary(records, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--output', default='goal_common_clock_v1'); args = parser.parse_args()
    with threadpool_limits(limits=1): run(ROOT/'outputs/uteg_mapping_target_runs'/args.output, args.seeds, args.ids)
