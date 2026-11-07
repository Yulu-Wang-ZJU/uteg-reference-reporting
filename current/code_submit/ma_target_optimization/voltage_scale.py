"""Preserve the confirmed common voltage scale in development-only mapping.

Voltage is the recorded sensor response, not a calibrated modulus in Pa.
The user confirmed common first-channel units/export rules across centers.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .common_clock import load_preprocessor, prefix_features
from .duration_transfer import TIME_COLUMNS, GROWTH_COLUMNS
from .signal_neighborhood import kinetics
from .search import ROOT, training_view, make_summary
from .hubei_source_search import add_hubei_arm
from .models import Features, fit_candidate, weights
from .metrics import assess
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def voltage_features(raw, observed, horizon, mode, preprocess):
    if mode == 'p5':
        return prefix_features(raw, observed, horizon, preprocess)
    if mode not in ('uncut', 'uncut_baseline') or horizon not in (300, 900):
        raise ValueError('Unknown representation')
    raw, observed = np.asarray(raw, float), np.asarray(observed, bool)
    if raw.ndim != 1 or raw.shape != observed.shape:
        raise ValueError('Mismatched signal and mask')
    n = int(observed[:horizon].sum())
    if n < 11 or not np.array_equal(np.flatnonzero(observed[:horizon]), np.arange(n)):
        raise ValueError('Expected contiguous observed prefix')
    signal = raw[:n]
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite observed voltage')
    smooth, _ = preprocess(signal, q=None)
    completed = np.pad(smooth, (0, horizon-n), mode='edge')
    edges = np.linspace(0, horizon, 91).astype(int)
    wave = np.array([np.median(completed[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    f = kinetics(wave)
    f[TIME_COLUMNS] *= horizon/900.
    f[GROWTH_COLUMNS] *= 900./horizon
    f = np.r_[f, n/horizon]
    if mode == 'uncut_baseline':
        f = np.r_[f, np.median(signal[:10])]
    return f


@dataclass
class PooledVoltage:
    transform: Features
    coef: np.ndarray
    source_count: int
    diagnostics: dict

    def predict(self, x):
        z = self.transform.transform(x)
        # Target is the reference center; its source-offset indicators are zero.
        design = np.c_[z, np.zeros((len(z), self.source_count)), np.ones(len(z))]
        return np.maximum(60+10*(design@self.coef), 0)


def fit_pooled(train, c):
    names = ['Hubei'] if c['arm'] == 'HB_to_SA' else ['Huzhou', 'Hubei']
    if c['arm'] not in ('HB_to_SA', 'HZ+HB_to_SA'):
        raise ValueError('Unsupported source arm')
    xs, ys, codes = [], [], []
    for j, name in enumerate(names):
        x, y = train['source_centers'][name]
        xs.append(np.asarray(x, float)); ys.append(np.asarray(y, float)[:, 2])
        code = np.zeros((len(x), len(names))); code[:, j] = 1
        codes.append(code)
    sx, sy = np.vstack(xs), np.concatenate(ys)
    tx, ty = np.asarray(train['x'], float), np.asarray(train['y'], float)[:, 2]
    x, y = np.vstack([sx, tx]), np.r_[sy, ty]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Training inputs must be finite')
    # The whole source pool and target A each receive total loss mass one.
    w = np.r_[weights(sy, c['source_cap'])/len(sy), weights(ty, c['cap'])/len(ty)]
    w /= w.mean()
    transform = Features.fit(x, 'linear')
    design = np.c_[transform.transform(x), np.vstack(codes+[np.zeros((len(tx), len(names)))]), np.ones(len(x))]
    penalty = np.diag(np.r_[np.ones(design.shape[1]-1), 0.])
    lhs = design.T@(w[:, None]*design)+len(y)*c['alpha']*penalty
    coef = np.linalg.solve(lhs+1e-12*np.eye(len(lhs)), design.T@(w*(y-60)/10))
    return PooledVoltage(transform, coef, len(names), {
        'source_n': len(sy), 'target_n': len(ty),
        'source_weight_mass': float(w[:len(sy)].sum()/w.sum()),
        'target_weight_mass': float(w[len(sy):].sum()/w.sum()),
        'scale': 'one shared training-only affine feature scale; no center-wise standardization',
        'source_offsets': names})


def fit(c, train, cache):
    return fit_pooled(train, c) if c['family'] == 'voltage_pool' else fit_candidate(c, train, cache)


def grid():
    models = [dict(family='ridge', arm='target_only', features='linear', alpha=.01, cap=cap)
              for cap in (1., 20.)]
    for arm in ('HB_to_SA', 'HZ+HB_to_SA'):
        models.append(dict(family='transfer', arm=arm, mode='joint', source_alpha=.1,
                           source_cap=5., cap=20., adapter_penalty=10.))
        models.append(dict(family='voltage_pool', arm=arm, alpha=.01, source_cap=5., cap=20.))
    result = []
    for horizon in (300, 900):
        for mode in ('p5', 'uncut', 'uncut_baseline'):
            for model in models:
                c = {**model, 'horizon': horizon, 'voltage_mode': mode,
                     'baseline_version': 'hubei_raw_first_signal_restored_v2'}
                c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
                result.append(c)
    return result


def recover_hubei_baselines(data, rows, raw, observed):
    """Verify the original first signal, recovering only the archived offset.

    Hubei's intermediate `wave` had already had its baseline subtracted.
    No reference channels or other sensor columns are used here.
    """
    manifest_path = ROOT/'outputs/uteg_hubei_curve_qc_fixedlength_20261021_170606/INPUT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    lookup = {r['audit_id']: r for r in manifest if r['column'] == 1}
    baselines = np.zeros(len(rows)); deltas = []; hashes = {str(manifest_path): sha256(manifest_path)}
    for i, row in enumerate(rows):
        if data['centers'][row] != 'Hubei': continue
        item = lookup[str(data['ids'][row]).removeprefix('Hubei_')]
        source = Path(item['source']); digest = sha256(source)
        assert digest == item['source_sha256'], 'Changed raw source'
        content = source.read_bytes(); lines = None
        for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
            try:
                decoded = content.decode(encoding)
                if '信号1' in decoded.splitlines()[0]:
                    lines = list(csv.reader(decoded.splitlines())); break
            except UnicodeError:
                continue
        if lines is None: raise ValueError('Unrecognized first-signal export')
        n = int(observed[i].sum())
        assert n == item['observed_used']
        # Other numeric columns are neither parsed nor used.
        values = np.array([float(line[0]) for line in lines[1:n+1]])
        base = float(np.median(values[:10]))
        assert abs(base-item['baseline_first10']) < 1e-12
        delta = float(np.max(np.abs(values-base-raw[i, :n])))
        assert delta < 1e-12, 'Archive does not replay original first signal minus baseline'
        baselines[i] = base; deltas.append(delta); hashes[str(source)] = digest
    assert len(deltas) == 79
    return baselines, {'verified_raw_files': len(deltas), 'max_zeroed_replay_difference': max(deltas),
                       'offset_recovery': 'restore Hubei first10 median from original first signal; no second signal or reference channel'}, hashes


def load_inputs(data, parts):
    path = ROOT/'outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz'
    preprocess, prep_path = load_preprocessor()
    rows = data['allowed_training_indices']
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive['ids'], data['ids'])
        assert np.array_equal(archive['center'], data['centers'])
        raw, observed = archive['original'][rows], archive['observed'][rows]
    recovered, recovery_audit, recovery_hashes = recover_hubei_baselines(data, rows, raw, observed)
    matrices = {}
    for horizon in (300, 900):
        for mode in ('p5', 'uncut', 'uncut_baseline'):
            width = 19 if mode == 'uncut_baseline' else 18
            x = np.full((len(data['x']), width), np.nan)
            for local, (row, signal, mask) in enumerate(zip(rows, raw, observed)):
                x[row] = voltage_features(signal, mask, horizon, mode, preprocess)
                if mode == 'uncut_baseline':
                    # Zeroed kinetics are offset invariant. Keep their exact
                    # historical arithmetic and restore only the baseline term.
                    x[row, -1] += recovered[local]
            matrices[(horizon, mode)] = x
    test = next(iter(parts.values()))['test']
    assert all(np.isnan(x[test]).all() for x in matrices.values())
    audit = {'first_channel_only': True, 'test_features_nan': True, 'allowed_records': len(rows),
             'user_confirmation': '2026-11-01: same first-column units and export rules across centers; voltage reflecting dissipative/viscoelastic response',
             'physical_calibration': 'not supplied; no voltage-to-modulus conversion or assumed monotonic MA relation',
             'hubei_baseline_recovery': recovery_audit,
             'center_signal_summaries': {}}
    for center in np.unique(data['centers'][rows]):
        keep = data['centers'][rows] == center
        baseline = np.median(raw[keep, :10], axis=1)+recovered[keep]
        audit['center_signal_summaries'][str(center)] = {
            'n': int(keep.sum()), 'baseline_voltage_export_quantiles': np.quantile(baseline, [0, .25, .5, .75, 1]).tolist(),
            'observed_length_quantiles': np.quantile(observed[keep].sum(1), [0, .5, 1]).tolist()}
    return matrices, audit, {**recovery_hashes, **{str(p): sha256(p) for p in (path, prep_path)}}


def run(out, seeds, ids=None):
    out.mkdir(parents=True, exist_ok=True); (out/'private').mkdir(exist_ok=True)
    cfg = json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'))
    cfg['seeds'] = list(range(1000, 1000+seeds))
    data, parts, hashes = load_development_inputs(ROOT, cfg); add_hubei_arm(data, parts)
    matrices, audit, more = load_inputs(data, parts); hashes.update(more)
    candidates = grid()
    if ids is not None:
        candidates = [c for c in candidates if c['id'] in ids]
        assert len(candidates) == len(ids)
    protocol = {'candidates': candidates, 'seeds': cfg['seeds'], 'input_hashes': hashes,
                'code_hashes': {name: sha256(ROOT/'code_submit/ma_target_optimization'/name) for name in
                                ('voltage_scale.py', 'common_clock.py', 'models.py', 'metrics.py', 'signal_neighborhood.py')},
                'mapping_code_hash': sha256(ROOT/'code_submit/ma_imbalance_mapping/models.py'),
                'scope': 'same target60 grouped development folds; approved source174; no original test labels/features',
                'hypothesis': 'with a common voltage export scale, preserve baseline and pre-P5 signal; compare direct shared-slope pooling with source mapping plus target adaptation',
                'models': 'fixed existing target/adapter controls; pooled shared slopes and regularized source offsets, equal total source/target loss mass',
                'comparison': '36 predeclared configurations, no new hyperparameter grid; 300/900 prefixes matched within each comparison',
                'representation': 'p5 control; identical smoothing without P5 truncation; same uncut features plus initial voltage baseline',
                'baseline_recovery': 'Hubei archive was already zeroed; raw first-signal baseline restored from 79 hash-verified original CSVs; fixed frozen arrays unchanged',
                'padding': 'last observed value after prefix-only preprocessing; observed fraction retained; no added observations',
                'interpretation': 'voltage not converted to physical modulus; no unmeasured elastic component is inferred',
                'selection_scope': 'development screening, not unbiased validation'}
    text = json.dumps(protocol, indent=2, allow_nan=False)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8') != text:
        raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text, encoding='utf-8')
    (out/'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    path = out/'CANDIDATES.jsonl'
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = {r['candidate']['id'] for r in records}; caches = {key: {} for key in matrices}
    pool = next(iter(parts.values()))['pool']; pos = {int(row): i for i, row in enumerate(pool)}
    for c in candidates:
        if c['id'] in done: continue
        start = time.time(); key = (c['horizon'], c['voltage_mode']); data['x'] = matrices[key]
        try:
            predictions = []
            for seed, part in parts.items():
                p = np.full(len(pool), np.nan)
                for fold in part['folds']:
                    A, B = fold['A'], fold['B']
                    assert not set(data['groups'][A]) & set(data['groups'][B])
                    model = fit(c, training_view(data, A), caches[key])
                    p[[pos[int(row)] for row in B]] = model.predict(data['x'][B])
                assert np.isfinite(p).all()
                predictions.append(p)
            p = np.asarray(predictions); y = data['y'][pool, 2]
            np.savez_compressed(out/'private'/f"{c['id']}.npz", prediction=p, reference=y, pool=pool, seeds=cfg['seeds'])
            row = {'candidate': c, 'status': 'complete', 'metrics': assess(np.broadcast_to(y, p.shape), p), 'seconds': time.time()-start}
        except Exception as error:
            row = {'candidate': c, 'status': 'failed', 'error': str(error), 'seconds': time.time()-start}
        records.append(row)
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        m = row.get('metrics', {})
        print(json.dumps({'done': len(records), 'total': len(candidates), 'mode': c['voltage_mode'], 'horizon': c['horizon'],
                          'family': c['family'], 'arm': c['arm'], 'status': row['status'],
                          **{k: m[k] for k in ('agreement', 'fnr', 'low_recall', 'high_recall') if k in m}}), flush=True)
    return make_summary(records, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--ids', nargs='*'); parser.add_argument('--output', default='goal_voltage_scale_v2')
    args = parser.parse_args()
    with threadpool_limits(limits=1): run(ROOT/'outputs/uteg_mapping_target_runs'/args.output, args.seeds, args.ids)
