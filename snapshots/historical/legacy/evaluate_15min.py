"""Frozen 15-min external-test replay and independent summary.

The target worker never reads external-test labels.  This evaluator reads the
labels only after TARGET_ALL_FROZEN.json exists, applies the frozen selected
calibrator/gate to the unchanged test rows, and saves row-level predictions
and actions for an independent re-computation.
"""
import json
import shutil
import time
from pathlib import Path

import runtime as r
import joblib
import numpy as np

m = r.load('legacy_eval_models', r.ROOT.parent / 'uteg_hill_exp_matched_20261012_153337' / 'models.py')
TARGETS = ('Huzhou', 'SAHZU', 'Hubei')
SEEDS = tuple(range(1000, 1020))


def _labels(j):
    return m.labels(m.ci(j)) if np.asarray(j).ndim > 1 else m.labels(j)


def _load_predictions(D, role, target, budget, seed, out):
    z = role['roles'][target]
    test = list(z['test'])
    src = out['source_model']
    raw = m.predict(src, D['x'][test])
    base_obj = out['cache'][out['positions'][3]['base']]
    H = m.calibrated(base_obj['cal'], raw)
    d = 1 if target == 'Huzhou' else 2
    As = ['H'] if d == 1 else list(m.ans['A'])
    med = {'H': H}
    for a, mod in base_obj['adapters'].items():
        med[a] = m.ans['adapt'](mod, D['x'][test], raw, H, src)

    final = out['final_selection']
    if d == 1:
        name = final.get('config', final.get('name'))
        threshold = final.get('threshold')
        sm = out['scalar_models'][3][name]
        # SOURCE refers to the gate's scalar evidence, not its emitted class.
        # The historical pipeline always reports the calibrated Huber MED4.
        j = m.ci(med['H'])
        ci_evidence = m.ci(raw) if name == 'SOURCE-CI-ORDINAL' else j
        yp = m.labels(j)
        score = m.scalar['score'](sm, ci_evidence, yp)
        allowed = np.isfinite(score) & (yp != 9)
        action = m.st['actions'](score, yp, threshold)
        selected = 'H'
    else:
        selected = out['gate_models'][3]['adapter']
        name = final.get('name')
        threshold = final.get('threshold')
        j = m.ci(med[selected])
        yp = m.labels(j)
        F = m.gm.features(D['x'][test], med[selected], src)
        score, allowed, aux = m.gm.score(name, out['gate_models'][3]['models'][name], F)
        action = m.gs['actions'](score, allowed, threshold)

    reported_med = med[selected]
    return dict(test=np.asarray(test, dtype=int), raw=raw, med=np.asarray(reported_med),
                huber_med=np.asarray(med[selected]),
                pred_ci=np.asarray(j), pred_class=np.asarray(yp), score=np.asarray(score),
                allowed=np.asarray(allowed, dtype=bool), action=np.asarray(action, dtype=bool),
                qc_forced_wait=np.zeros(len(test), dtype=bool),
                selected_adapter=selected, selected_gate=name, threshold=threshold,
                reference_ci=np.asarray(D['ci'][test]), reference_class=m.labels(D['ci'][test]))


def _summary(z, target, budget):
    yp, yt = z['pred_class'], z['reference_class']
    known = yt != 9
    e = (yp != yt) & known
    a = z['action'] & known
    abnormal = known & (yt != 0)
    abnormal_reported = a & abnormal
    normal_false = abnormal_reported & (yp == 0)
    # An abnormal prediction is correct only when it matches the reference
    # abnormal class (low or high); merely avoiding the normal class includes
    # low<->high reversals and would overstate abnormal recall.
    correct_abnormal = abnormal_reported & (yp == yt)
    denom = int(a.sum())
    return dict(target=target, budget=budget, n_records=int(len(yt)), known_n=int(known.sum()),
                base_accuracy=float(np.mean(~e)) if known.any() else None,
                base_error=float(np.mean(e)) if known.any() else None,
                coverage=float(np.mean(a)),
                early_n=denom,
                early_accuracy=float(np.mean(~e[a])) if denom else None,
                early_error=float(np.mean(e[a])) if denom else None,
                wait_fraction=float(np.mean(~a)),
                abnormal_n=int(abnormal.sum()),
                abnormal_early_n=int(abnormal_reported.sum()),
                abnormal_false_normal_n=int(normal_false.sum()),
                abnormal_correct_n=int(correct_abnormal.sum()),
                abnormal_false_normal_rate=float(normal_false.sum() / abnormal.sum()) if abnormal.any() else None,
                abnormal_early_recall=float(correct_abnormal.sum() / abnormal.sum()) if abnormal.any() else None,
                J10=float(np.mean(np.where(a, e, .10))),
                selected_gate=list(z['selected_gate']),
                selected_adapter=list(z['selected_adapter']))


def main():
    if not (r.ROOT / 'TARGET_ALL_FROZEN.json').exists():
        raise RuntimeError('TARGET_ALL_FROZEN.json required before external replay')
    D = {k: v for k, v in np.load(r.ROOT / 'DATA15.npz', allow_pickle=True).items()}
    role = r.read(r.ROOT / 'ROLE_FREEZE.json')
    outdir = r.ROOT / 'eval15'
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for target in TARGETS:
        test = list(role['roles'][target]['test'])
        for budget, label in [('C20', '20'), ('CFULL', 'FULL')]:
            rows = []
            for seed in SEEDS:
                p = r.ROOT / 'target15' / f'{target}_{budget}_seed{seed}.joblib'
                if not p.exists():
                    raise FileNotFoundError(p)
                out = joblib.load(p)
                q = _load_predictions(D, role, target, budget, seed, out)
                # Hubei's two 15-min QC failures remain in the test list but
                # are forced to wait.  This QC is a 15-min availability check;
                # it must never leak into the 5/10-min prefix branches.
                bad = (target == 'Hubei') & (~np.asarray(D['input_qc'][test], dtype=bool))
                q['allowed'][bad] = False
                q['action'][bad] = False
                q['qc_forced_wait'][bad] = True
                rows.append(q)
            keys = ['raw', 'med', 'huber_med', 'pred_ci', 'pred_class', 'score', 'allowed', 'action', 'qc_forced_wait', 'reference_ci', 'reference_class']
            payload = {k: np.stack([q[k] for q in rows], axis=0) for k in keys}
            payload.update(seed=np.asarray(SEEDS), test=np.asarray(test, dtype=int),
                           test_ids=np.asarray(D['ids'][test]),
                           selected_adapter=np.asarray([q['selected_adapter'] for q in rows]),
                           selected_gate=np.asarray([q['selected_gate'] for q in rows]),
                           threshold=np.asarray([np.nan if q['threshold'] is None else q['threshold'] for q in rows]))
            np.savez_compressed(outdir / f'{target}_{label}.npz', **payload)
            # Aggregate first across seeds as indicator/error arrays; do not
            # treat seeds as independent patients.
            agg = {k: payload[k].mean(axis=0) if payload[k].dtype.kind in 'fiu' else payload[k] for k in []}
            # Summary uses the seed-averaged indicators, preserving the old
            # protocol's record-level reporting convention.
            az = dict(pred_class=payload['pred_class'].mean(axis=0),
                      reference_class=payload['reference_class'][0],
                      action=payload['action'].mean(axis=0),
                      selected_gate=payload['selected_gate'],
                      selected_adapter=payload['selected_adapter'])
            # For an interpretable point estimate, report expected indicators
            # rather than a seed vote; thresholded indicators are retained in
            # the NPZ for exact independent recomputation.
            ytrue = payload['reference_class'][0]
            yp = payload['pred_class']
            act = payload['action']
            known = ytrue != 9
            e = yp != ytrue[None, :]
            abnormal = known & (ytrue != 0)
            early = act & known[None, :]
            # Axis 0 is seed and axis 1 is record.  Counts are first summed
            # per seed and then averaged; taking the mean over records here
            # would under-report the number of released cases by a factor of
            # 20 (the superseded v1 summary made exactly that mistake).
            n_early = early.sum(axis=1)
            wrong = (early & e).sum(axis=1)
            ab_false = (early & abnormal[None, :] & (yp == 0)).sum(axis=1)
            # Separate abnormal false-normal from correctly reported abnormal
            # cases.  The latter must match the reference class; low/high
            # reversals remain errors and are not counted as correct abnormal.
            ab_correct = (early & abnormal[None, :] & (yp == ytrue[None, :])).sum(axis=1)
            summary.append(dict(target=target, budget=budget, n_records=len(test), known_n=int(known.sum()),
                                base_accuracy=float(1-e[:, known].mean()),
                                coverage=float(early.mean()), early_n=float(n_early.mean()),
                                early_error=float(wrong.sum()/early.sum()) if early.sum() else None,
                                wait_fraction=float(1-early.mean()), abnormal_n=int(abnormal.sum()),
                                qc_forced_wait_n=float(payload['qc_forced_wait'].sum(axis=1).mean()),
                                abnormal_false_normal_n=float(ab_false.mean()),
                                abnormal_correct_n=float(ab_correct.mean()),
                                abnormal_false_normal_rate=float(ab_false.mean()/abnormal.sum()) if abnormal.any() else None,
                                abnormal_early_recall=float(ab_correct.mean()/abnormal.sum()) if abnormal.any() else None,
                                J10=float(np.mean(np.where(early, e, .10))),
                                selected_gate_counts={str(k): int(v) for k,v in zip(*np.unique(payload['selected_gate'],return_counts=True))},
                                selected_adapter_counts={str(k): int(v) for k,v in zip(*np.unique(payload['selected_adapter'],return_counts=True))}))
    # V2 is retained as the no-forced-QC replay.  V3 is the protocol-correct
    # 15-min report in which the two Hubei QC failures remain in the test
    # denominator but cannot be released early.
    if (r.ROOT / 'EVALUATION_15MIN_SUMMARY_V2.json').exists():
        shutil.copy2(r.ROOT / 'EVALUATION_15MIN_SUMMARY_V2.json', r.ROOT / 'EVALUATION_15MIN_SUMMARY_V2_NO_QC_SUPERSEDED.json')
    r.write('EVALUATION_15MIN_SUMMARY_V4.json', dict(version='legacy_aligned_threecenter_eval15_v4',
                                                  time=time.time(), rows=summary,
                                                  source=r.sha(r.ROOT/'SOURCE_FREEZE.json'),
                                                  supersedes='EVALUATION_15MIN_SUMMARY_V3.json',
                                                  legacy_semantics='SOURCE ordinal scores source CI against calibrated class; every report emits calibrated/adapted MED4; original actions functions including None thresholds',
                                                  target=r.sha(r.ROOT/'TARGET_ALL_FROZEN.json'),
                                                  test_labels_read_after_freeze=True,
                                                  fifteen_min_qc_forced_wait='Hubei input_qc=False rows remain in denominator and action=False',
                                                  curve_metrics_not_evaluated=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
