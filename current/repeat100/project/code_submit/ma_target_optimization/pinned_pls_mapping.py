"""Single frozen development candidate, reusable for gate/direction validation."""
import hashlib
import numpy as np
from . import supervised_curve_mapping as pls
from . import low_rank_curve_mapping as shared
from .supervised_high_rescue import PINNED, combine
from .full_curve_cost_sensitive import solve


def predict(data, blocks, cur, smooth, train, query):
    assert not set(data['groups'][train]).intersection(data['groups'][query])
    raw, audit = shared.calibration.source.models(data, blocks, cur, smooth, train, query)
    means, counts = shared.calibration.conditional_means(data, train)
    low = shared.calibration.predict(raw, shared.BASE, means)
    c = PINNED['pls']
    rows, ns, y, w = shared.training_rows_weights(data, train, c['recipe'])
    assert not set(data['groups'][rows]).intersection(data['groups'][query])
    z, q, diag = pls.pls_blocks(*blocks[c['representation']], data['y'][:, 2], rows, query,
                               c['target'], [c['components']])[c['components']]
    if ns:
        z = np.c_[z, np.r_[np.ones(ns), np.zeros(len(train))]]
        q = np.c_[q, np.zeros(len(query))]
    z = np.c_[z, np.ones(len(rows))]; q = np.c_[q, np.ones(len(query))]
    coef, info = solve(z, (y-60)/10, w, c['loss'], c['cost'], alpha=c['alpha'])
    high = np.maximum(60+10*q@coef, 0)
    final = combine(low, high, PINNED['scope'])
    return final, np.c_[raw, high], dict(base=audit, means=means.tolist(), counts=counts,
        pls=diag, solver=info, pls_coefficient_sha256=hashlib.sha256(coef.tobytes()).hexdigest())
