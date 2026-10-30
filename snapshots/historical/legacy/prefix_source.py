"""Fit frozen-config source mappers for the 5/10-min U-TEG prefixes.

The 15-min source family/configuration is already frozen in ``source/*.joblib``.
This file fits only that same selected configuration on the corresponding
prefix parameters; no prefix-time model selection is performed.
"""
import json
import time
from pathlib import Path

import runtime as r
import numpy as np
import joblib

m = r.load('legacy_prefix_models', r.ROOT.parent / 'uteg_hill_exp_matched_20261012_153337' / 'models.py')
SEEDS = tuple(range(1000, 1020))
TARGETS = ('Huzhou', 'SAHZU', 'Hubei')
HORIZONS = (300, 600)


def make_prefix_x():
    packaged = r.ROOT / 'PREFIX_PARAMETERS.npz'
    if packaged.exists():
        z = np.load(packaged, allow_pickle=True)
        out = np.asarray(z['x'], dtype=float)
        assert out.shape == (len(HORIZONS), 480, 4) and np.isfinite(out).all()
        assert np.array_equal(np.asarray(z['horizon_seconds']), np.asarray(HORIZONS))
        r.write('PREFIX_X_IDENTITY.json', dict(source=str(packaged), source_sha256=r.sha(packaged),
                                               horizons=list(HORIZONS), records=480,
                                               shape=list(out.shape), new_signal_fit_positions=0,
                                               true_observed_segment_only=True,
                                               packaged_parameter_archive=True))
        np.savez_compressed(r.ROOT / 'PREFIX_X.npz', horizon_seconds=np.asarray(HORIZONS), x=out)
        return out
    src = r.ROOT / 'PREFIX_SIGNAL_FITS.json'
    rows = r.read(src)
    out = np.full((len(HORIZONS), 480, 4), np.nan, dtype=float)
    pos = {h: i for i, h in enumerate(HORIZONS)}
    seen = set()
    for z in rows['fits']:
        key = (int(z['horizon']), int(z['index']))
        assert key not in seen, key
        assert key[0] in pos and 0 <= key[1] < 480, key
        q = np.asarray(z['q'], dtype=float)
        assert q.shape == (4,) and np.isfinite(q).all(), key
        out[pos[key[0]], key[1]] = q
        seen.add(key)
    assert len(seen) == len(HORIZONS) * 480
    assert np.isfinite(out).all()
    np.savez_compressed(r.ROOT / 'PREFIX_X.npz', horizon_seconds=np.asarray(HORIZONS), x=out)
    r.write('PREFIX_X_IDENTITY.json', dict(source=str(src), source_sha256=r.sha(src),
                                           horizons=list(HORIZONS), records=480,
                                           shape=list(out.shape), new_signal_fit_positions=0,
                                           true_observed_segment_only=True))
    return out


def main():
    if not (r.ROOT / 'SOURCE_FREEZE.json').exists():
        raise RuntimeError('source models must be frozen before prefix source fits')
    D0 = dict(np.load(r.ROOT / 'DATA15.npz', allow_pickle=True))
    role = r.read(r.ROOT / 'ROLE_FREEZE.json')
    X = make_prefix_x()
    done = []
    for target in TARGETS:
        S = list(role['roles'][target]['source'])
        for seed in SEEDS:
            old = joblib.load(r.ROOT / 'source' / f'{target}_seed{seed}.joblib')
            family = old['source_selection']
            config = old['families'][family]['config']
            for hi, horizon in enumerate(HORIZONS):
                # Supersede the interrupted draft that used the default
                # optimizer budget; protocol-compliant fits use a new namespace.
                path = r.ROOT / 'prefix_source_v2' / f'{target}_t{horizon}_seed{seed}.joblib'
                if path.exists():
                    done.append(path)
                    continue
                data = dict(D0)
                data['x'] = X[hi]
                source_mask = np.zeros(len(data['x']), dtype=bool)
                source_mask[S] = True
                data['medical'] = np.asarray(D0['medical'], dtype=float).copy()
                data['ci'] = np.asarray(D0['ci'], dtype=float).copy()
                data['medical'][~source_mask] = np.nan
                data['ci'][~source_mask] = np.nan
                # m.fit_source reads only x/medical/ci and keeps the selected
                # source identity; no target labels are used for fitting.
                units = 4 if m.CONFIGS[config]['family'] == 'SVR' else 1
                key = f'prefix_source_v2_{target}_t{horizon}_seed{seed}'
                r.ledger(key, 'prefix_source', units, target=target, horizon=horizon,
                         seed=seed, config=config, source=S)
                started = time.time()
                source_model = old['families'][family]['model']
                fit_seed = int(source_model['seed'])
                steps = int(source_model.get('steps') or source_model.get('actual_steps') or 0) or None
                model = m.fit_source(data, S, [], config, fit_seed, steps=steps)
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(dict(target=target, seed=seed, horizon=horizon,
                                 source=S, source_family=family, source_config=config,
                                 model=model, fit_seed=fit_seed, steps=steps,
                                 source_model_sha=r.sha(r.ROOT / 'source' / f'{target}_seed{seed}.joblib'),
                                 training_seconds=time.time() - started,
                                 prefix_x_sha=r.sha(r.ROOT / 'PREFIX_X.npz')),
                            path, compress=3)
                print('PREFIX_SOURCE', target, horizon, seed, family, config,
                      round(time.time() - started, 2), flush=True)
                done.append(path)
    assert len(done) == len(TARGETS) * len(SEEDS) * len(HORIZONS)
    r.write('PREFIX_SOURCE_FREEZE_V2.json', dict(version='legacy_aligned_prefix_source_v2',
                                              files={p.name: r.sha(p) for p in done},
                                              count=len(done), horizons=list(HORIZONS),
                                              source=r.sha(r.ROOT / 'SOURCE_FREEZE.json'),
                                              new_test_evaluated=False))
    rows = [r.read(p) for p in (r.ROOT / 'ledger').glob('*.json')]
    r.write('RESOURCE_LEDGER_PREFIX_SOURCE.json', dict(fits=sum(int(z['units']) for z in rows),
                                                       kinds={k: sum(int(z['units']) for z in rows if z['kind'] == k)
                                                              for k in sorted(set(z['kind'] for z in rows))},
                                                       resources=r.guard()))


if __name__ == '__main__':
    main()
