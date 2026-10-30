"""Legacy-aligned 15-min calibration/gate fits for the three-center audit.

This module deliberately reuses the 20261012 generalized-exponential source
mapping and the old HZ/D2 calibration/gate identities.  It only changes the
role manifest and target center; it does not introduce a new model family.
"""
import runtime as r
import numpy as np
import joblib
import time

# Load the old implementation after the task-local runtime has installed all
# cache paths.  The old module's ``import runtime`` therefore resolves to r.
m = r.load('legacy_aligned_models', r.ROOT.parent / 'uteg_hill_exp_matched_20261012_153337' / 'models.py')

TARGETS = ('Huzhou', 'SAHZU', 'Hubei')
SEEDS = tuple(range(1000, 1020))


def fit_lr(F, yh, yt):
    ok = (yh != 9) & (yt != 9)
    return m.rm.fit_model(dict(id='ERROR-LR', family='ERROR-LR', regularizer=.1), F[ok], yh[ok], yt[ok], 0)


def lr_score(mod, F, yh):
    score = np.full(len(yh), np.inf)
    ok = yh != 9
    if ok.any():
        score[ok] = m.rm.predict_model(mod, F[ok], yh[ok])
    return score


def _spec(role, target, seed, arm):
    z = role['roles'][target]
    if arm == 'FULL':
        pool = list(z['pool_full'])
        positions = z['positions_full'][str(seed)]
    elif arm == 'C20':
        pool = list(z['c20_by_seed'][str(seed)])
        positions = z['positions_c20'][str(seed)]
    else:
        raise ValueError(arm)
    return dict(target=target, arm=arm, seed=seed, d=1 if target == 'Huzhou' else 2,
                pool=pool, test=list(z['test']), positions=positions)


def run_one(D, role, spec):
    target, arm, seed = spec['target'], spec['arm'], spec['seed']
    key = f"{target}_{arm}_seed{seed}"
    path = r.ROOT / 'runs' / f'{key}.joblib'
    if path.exists():
        return r.read(path) if path.suffix == '.json' else joblib.load(path)

    started = time.time()
    old = joblib.load(r.ROOT / 'source' / f'{target}_seed{seed}.joblib')
    src = old['families'][old['source_selection']]['model']
    pool = spec['pool']
    loc = {v: k for k, v in enumerate(pool)}
    raw = m.predict(src, D['x'][pool])
    cache, positions, scores = {}, {}, {}
    d = spec['d']
    As = ['H'] if d == 1 else m.ans['A']
    OOF = {a: np.full((len(pool), 4), np.nan) for a in As}
    smodels, gate_models, extra = {}, {}, {}

    def get(T):
        token = ','.join(map(str, sorted(T)))
        if token not in cache:
            ix = [loc[v] for v in T]
            rr = raw[ix]
            r.ledger(key + '_H_' + r.hashlib.sha256(token.encode())[:12], 'Huber', train=T)
            cal = m.fit_cal(rr, D['medical'][T], src['ym'], src['ys'])
            H = m.calibrated(cal, rr)
            ads = {}
            if d == 2:
                for a in As[1:]:
                    r.ledger(key + '_' + a + '_' + r.hashlib.sha256(token.encode())[:12], 'adapter', train=T)
                    ads[a] = m.ans['fit_adapter'](a, D['x'][T], rr, H, D['ci'][T], src)
            cache[token] = dict(train=T, cal=cal, adapters=ads)
        return token

    def preds(token, I):
        ca = cache[token]
        rr = raw[[loc[v] for v in I]]
        H = m.calibrated(ca['cal'], rr)
        return dict(H=H, **{a: m.ans['adapt'](mod, D['x'][I], rr, H, src)
                            for a, mod in ca['adapters'].items()})

    # Build nested target-side calibration examples.  These are the same
    # three inner folds plus a complete-C20 position used by the old workflow.
    for pos in spec['positions']:
        code, A, B = int(pos['code']), list(pos['A']), list(pos['B'])
        al = {v: k for k, v in enumerate(A)}
        cross = {a: np.full((len(A), 4), np.nan) for a in As}
        deps = []
        for f in pos['inner']:
            token = get(list(f['T']))
            pp = preds(token, list(f['V']))
            for a in As:
                cross[a][[al[v] for v in f['V']]] = pp[a]
            deps.append(dict(T=list(f['T']), V=list(f['V']), token=token))
        base = get(A)
        vv = preds(base, B) if B else None
        positions[code] = dict(A=A, B=B, inner=deps, base=base, cross=cross)
        smodels[code], gate_models[code] = {}, {}
        yt = m.labels(m.ci(D['ci'][A]))

        if d == 1:
            yp = m.labels(m.ci(cross['H']))
            ok = (yt != 9) & (yp != 9)
            for j, name in enumerate(['SOURCE-CI-ORDINAL', 'HUBER-CI-ORDINAL']):
                ciin = m.ci(raw[[loc[v] for v in A]]) if j == 0 else m.ci(cross['H'])
                r.ledger(key + f'_scalar{code}_{j}', 'ordinal', train=A)
                sm = m.scalar['fit'](name, ciin[ok], yt[ok])
                smodels[code][name] = sm
                scores.setdefault(name, dict(score=np.full(len(pool), np.nan), family=name,
                                              params=3, order=j))
                if B:
                    bi = [loc[v] for v in B]
                    ciB = m.ci(raw[bi]) if j == 0 else m.ci(vv['H'])
                    scores[name]['score'][bi] = m.scalar['score'](
                        sm, ciB, m.labels(m.ci(vv['H'])))
        else:
            Fs, lrs = {}, {}
            for a in As:
                F = m.gm.features(D['x'][A], cross[a], src)
                Fs[a] = F
                yp = m.labels(m.ci(cross[a]))
                r.ledger(key + f'_LR_{code}_{a}', 'LR', train=A)
                lr = fit_lr(F, yp, yt)
                lrs[a] = lr
                for gi, gate in enumerate(['ERROR-LR', 'MARGIN']):
                    nm = a + '|' + gate
                    params = (0 if a == 'H' else 9 if a.startswith('L') else len(pool) + 1)
                    params += 14 if gate == 'ERROR-LR' else 0
                    scores.setdefault(nm, dict(score=np.full(len(pool), np.nan), adapter=a,
                                               gate=gate, params=params,
                                               order=As.index(a) * 2 + gi))
                    if B:
                        vp = m.labels(m.ci(vv[a]))
                        VF = m.gm.features(D['x'][B], vv[a], src)
                        sc = lr_score(lr, VF, vp) if gate == 'ERROR-LR' else \
                            -np.minimum(abs(m.ci(vv[a]) + 3), abs(m.ci(vv[a]) - 3))
                        scores[nm]['score'][[loc[v] for v in B]] = sc
            positions[code].update(features=Fs, lrs=lrs)
        if B:
            for a in As:
                OOF[a][[loc[v] for v in B]] = vv[a]

    yt_pool = m.labels(m.ci(D['ci'][pool]))
    if d == 1:
        selection = m.st['select'](scores, m.labels(m.ci(OOF['H'])), yt_pool, .1)
        final_selection = selection['selected']
    else:
        selection = m.ans['select'](scores,
                                    {a: m.labels(m.ci(p)) for a, p in OOF.items()}, yt_pool)
        gate_scores = {n: np.full(len(pool), np.nan) for n in m.gm.NAMES}
        allowed = {n: np.zeros(len(pool), bool) for n in m.gm.NAMES}
        oof = np.full((len(pool), 4), np.nan)
        for code, p in positions.items():
            A, B, cross = p['A'], p['B'], p['cross']
            yi = m.labels(D['ci'][A])
            if code == 3:
                a = selection['selected']['adapter']
            else:
                rows = []
                for order, a0 in enumerate(As):
                    j = m.ci(cross[a0])
                    rows.append(dict(m.gs['threshold'](
                        -np.minimum(abs(j + 3), abs(j - 3)), m.labels(j), yi),
                        name=a0, params=0 if a0 == 'H' else 9 if a0.startswith('L') else len(A) + 1,
                        order=order))
                a = m.gs['choose'](rows)['name']
            mods = m.gm.fit_all(p['features'][a], cross[a], D['ci'][A], p['lrs'][a],
                                key + f'_sixgate{code}')
            gate_models[code] = dict(adapter=a, models=mods)
            if B:
                bi = [loc[v] for v in B]
                med = OOF[a][bi]
                VF = m.gm.features(D['x'][B], med, src)
                oof[bi] = med
                for n in m.gm.NAMES:
                    gate_scores[n][bi], allowed[n][bi], _ = m.gm.score(n, mods[n], VF)
        yy = m.labels(m.ci(oof))
        thresholds = {n: dict(m.gs['threshold'](gate_scores[n], yy, yt_pool, allowed[n]),
                              name=n, params=m.gm.PARAMS[j], order=j)
                      for j, n in enumerate(m.gm.NAMES)}
        final_selection = m.gs['choose'](list(thresholds.values()))
        extra = dict(gate_oof=oof, gate_scores=gate_scores, allowed=allowed,
                     thresholds=thresholds)

    out = dict(target=target, arm=arm, seed=seed, source_family=old['source_selection'],
               pool=pool, test=spec['test'], cache=cache, positions=positions,
               scalar_models=smodels, gate_models=gate_models, source_model=src,
               selection=selection, final_selection=final_selection, OOF=OOF,
               scores=scores, extra=extra)
    m.dump(path.relative_to(r.ROOT), out)
    print('DOWNSTREAM', key, final_selection.get('config', final_selection.get('name')),
          round(time.time() - started, 1), flush=True)
    r.guard()
    return out


def main():
    if not (r.ROOT / 'SOURCE_FREEZE.json').exists():
        raise RuntimeError('SOURCE_FREEZE.json is required before downstream fits')
    D = dict(np.load(r.ROOT / 'DATA15.npz', allow_pickle=True))
    role = r.read(r.ROOT / 'ROLE_FREEZE.json')
    all_keys = []
    for target in TARGETS:
        for seed in SEEDS:
            for arm in ('C20', 'FULL'):
                spec = _spec(role, target, seed, arm)
                run_one(D, role, spec)
                all_keys.append(f'{target}_{arm}_seed{seed}')
    files = [r.ROOT / 'runs' / f'{k}.joblib' for k in all_keys]
    assert all(p.exists() for p in files)
    rows = [r.read(p) for p in (r.ROOT / 'ledger').glob('*.json')]
    total = sum(int(z['units']) for z in rows)
    r.write('ALL_MODELS_FREEZE.json', dict(time=time.time(), files={p.name: r.sha(p) for p in files},
                                           source=r.sha(r.ROOT / 'SOURCE_FREEZE.json'),
                                           roles=r.sha(r.ROOT / 'ROLE_FREEZE.json'),
                                           code={p.name: r.sha(p) for p in r.ROOT.glob('*.py')},
                                           fits=total, new_test_evaluated=False))
    r.write('RESOURCE_LEDGER_15MIN.json', dict(fits=total,
                                               kinds={k: sum(int(z['units']) for z in rows if z['kind'] == k)
                                                      for k in sorted(set(z['kind'] for z in rows))},
                                               resources=r.guard()))


if __name__ == '__main__':
    main()
