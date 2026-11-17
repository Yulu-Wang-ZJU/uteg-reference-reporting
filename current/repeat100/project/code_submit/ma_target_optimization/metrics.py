"""Explicit denominators; never treat withheld cases as correct classifications."""
import numpy as np


def states(ma):
    ma = np.asarray(ma, float)
    if not np.isfinite(ma).all():
        raise ValueError('Non-finite MA values')
    return np.where(ma < 50, 0, np.where(ma <= 70, 1, 2))


def assess(reference, prediction, accepted=None):
    y, p = np.asarray(reference, float).ravel(), np.asarray(prediction, float).ravel()
    if y.shape != p.shape or not len(y):
        raise ValueError('Reference and prediction must be nonempty and aligned')
    a = np.ones(len(y), bool) if accepted is None else np.asarray(accepted, bool).ravel()
    if a.shape != y.shape:
        raise ValueError('Acceptance mask is not aligned')
    t, q = states(y), states(p)
    c = np.zeros((3, 3), int)
    np.add.at(c, (t[a], q[a]), 1)
    safe = lambda num, den: float(num / den) if den else None
    abnormal, normal = t != 1, t == 1
    fn = int(np.sum(abnormal & (q == 1) & a))
    fp = int(np.sum(normal & (q != 1) & a))
    out = dict(n=len(y), released=int(a.sum()), coverage=float(a.mean()),
               agreement=safe(np.sum((t == q) & a), a.sum()),
               correct_yield=float(np.mean((t == q) & a)),
               fnr=safe(fn, np.sum(abnormal & a)), fpr=safe(fp, np.sum(normal & a)),
               false_negative_count=fn, false_positive_count=fp,
               false_negative_yield=fn/len(y), false_positive_yield=fp/len(y),
               opposite_abnormal_count=int(np.sum(abnormal & (q != 1) & (q != t) & a)),
               confusion=c.tolist(), mae=safe(np.abs(p[a]-y[a]).sum(), a.sum()))
    for i, name in enumerate(('low', 'within', 'high')):
        mask = (t == i) & a
        out[name+'_n'] = int(np.sum(t == i))
        out[name+'_released'] = int(mask.sum())
        out[name+'_coverage'] = safe(mask.sum(), np.sum(t == i))
        out[name+'_recall'] = safe(np.sum(mask & (q == i)), mask.sum())
        out[name+'_precision'] = safe(c[i, i], c[:, i].sum())
        out[name+'_mae'] = safe(np.abs(p[mask]-y[mask]).sum(), mask.sum())
        out[name+'_bias'] = safe((p[mask]-y[mask]).sum(), mask.sum())
    values = [out[k+'_mae'] for k in ('low', 'within', 'high')]
    out['macro_mae'] = float(np.mean(values)) if all(v is not None for v in values) else None
    out['mapping_target_pass'] = bool(out['agreement'] is not None and out['fnr'] is not None
                                      and out['fpr'] is not None and out['coverage'] == 1
                                      and out['agreement'] > .8 and out['fnr'] < .05
                                      and out['fnr'] < out['fpr'])
    out['gate_coverage_pass'] = out['coverage'] > .75
    return out


def selection_key(metrics):
    """Feasible first; otherwise reduce FNR among models meeting accuracy.

    Never reward an increase of FPR merely to satisfy the strict inequality.
    No averaging of class counts with independent-seed assumptions.
    """
    m=metrics
    if any(m.get(k) is None for k in ('agreement','fnr','fpr','macro_mae')):
        return (3, float('inf'), float('inf'), float('inf'), float('inf'))
    worst_recall = min(m['low_recall'], m['high_recall'])
    # Keep the strict FNR<FPR condition in the acceptance flag, but never
    # reward introducing false positives merely to satisfy that inequality.
    # In particular, a perfect FNR=FPR=0 solution must outrank an otherwise
    # identical solution with avoidable false positives.
    if m['agreement'] > .8 and m['fnr'] < .05:
        return (0, m['fnr'], m['fpr'], -worst_recall, m['macro_mae'])
    if m['agreement'] > .8:
        return (1, m['fnr'], m['fpr'], -worst_recall, m['macro_mae'])
    # Failed-accuracy candidates remain visible on the tradeoff frontier.
    return (2, -m['agreement'], m['fnr'], -worst_recall, m['macro_mae'])
