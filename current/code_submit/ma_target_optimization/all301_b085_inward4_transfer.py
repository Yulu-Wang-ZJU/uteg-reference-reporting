"""B-only source fitting and local correction using original input coefficients."""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import all301_internal_models as models
from .asymmetric_ma_mapping import loss_terms
from .metrics import states
ALPHA=.01
COST=4.

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def offset_objective(c, z, y, w, offset, alpha=ALPHA):
    penalty = np.r_[np.ones(z.shape[1]-1)*alpha, 0.]
    value, gradient, curvature = loss_terms(offset + z@c, y, np.ones(len(y), bool), 'residual', COST)
    f = np.mean(w*value) + np.sum(penalty*c*c)
    g = z.T@(w*gradient)/len(y) + 2*penalty*c
    h = 2*z.T@((w*curvature)[:, None]*z)/len(y) + 2*np.diag(penalty)
    return float(f), g, h

def solve_offset(z, y, w, offset):
    if np.all(offset == 0):
        return models.solve(z, y, w, 'residual', COST, alpha=ALPHA)
    penalty = np.r_[np.ones(z.shape[1]-1)*ALPHA, 0.]
    coef = np.linalg.solve(z.T@(w[:, None]*z)/len(y) + np.diag(penalty) + 1e-12*np.eye(z.shape[1]),
                           z.T@(w*(y-offset))/len(y))
    for iteration in range(200):
        value, g, h = offset_objective(coef, z, y, w, offset)
        if np.max(np.abs(g)) < 1e-8:
            break
        step = np.linalg.solve(h + 1e-12*np.eye(len(coef)), -g)
        slope = float(g@step)
        if slope >= 0:
            raise RuntimeError('Non-descent offset fit')
        rate = 1.
        for _ in range(50):
            proposal = coef + rate*step
            if offset_objective(proposal, z, y, w, offset)[0] <= value + 1e-4*rate*slope + 1e-14:
                coef = proposal
                break
            rate *= .5
        else:
            raise RuntimeError('Offset line search failed')
    value, g, _ = offset_objective(coef, z, y, w, offset)
    if np.max(np.abs(g)) >= 1e-7 or not np.isfinite(coef).all():
        raise RuntimeError('Offset model not stationary')
    return coef, dict(iterations=iteration, objective=value, gradient_max=float(np.max(np.abs(g))))

def source_basis(x):
    mean = x.mean(0); scale = np.maximum(x.std(0), 1e-6)
    t = (x-mean)/scale
    norms = np.sum(t*t, axis=1)
    k = np.exp(-np.maximum(norms[:, None]+norms[None, :]-2*t@t.T, 0))
    eig, vec = np.linalg.eigh((k+k.T)/2)
    keep = eig > max(float(eig.max())*1e-12, 1e-12)
    root = np.sqrt(eig[keep]); v = vec[:, keep]
    z = np.c_[v*root, np.ones(len(t))]
    return z, dict(mean=mean, scale=scale, train=t, project=v/root)

def source_predict(model, x):
    t = (x-model['mean'])/model['scale']; a = model['train']
    k = np.exp(-np.maximum(np.sum(t*t, axis=1)[:, None] + np.sum(a*a, axis=1)[None, :] - 2*t@a.T, 0))
    q = np.c_[k@model['project'], np.ones(len(x))]
    return np.maximum(60+10*q@model['coef'], 0.)

def fit_source(data, feature, rows):
    z, model = source_basis(feature[rows])
    y = data['y'][rows, 2]
    w = np.zeros(len(rows))
    for center in np.unique(data['centers'][rows]):
        mask = data['centers'][rows] == center
        w[mask] = models.weights(y[mask], 0, 'inward4')/mask.sum()
    w /= w.mean()  # Equal total mass for each source hospital; no local labels.
    coef, info = models.solve(z, (y-60)/10, w, 'residual', COST, alpha=ALPHA)
    model['coef'] = coef
    audit = dict(rows=rows.tolist(), n=len(rows), states=np.bincount(states(y), minlength=3).tolist(),
                 solver=info, weights_sha256=ahash(w), model_sha256=ahash(coef),
                 basis_sha256={k:ahash(v) for k,v in model.items() if k != 'coef'})
    return model, audit

def local_predict(data, x, train, query, source=None):
    # Labels outside current training rows cannot enter fitting or weights.
    y = np.full_like(data['y'], np.nan); y[train] = data['y'][train]
    shadow = dict(data, y=y)
    assert np.isnan(shadow['y'][query]).all()
    z, q, yy, ns, audit = models.design(shadow, [x], train, query, 'target_only')
    assert ns == 0
    w = models.weights(yy, 0, 'inward4')
    if source is None:
        base_train = np.zeros(len(train)); base_query = np.zeros(len(query))
    else:
        base_train = (source_predict(source, x[train])-60)/10
        base_query = (source_predict(source, x[query])-60)/10
    coef, info = solve_offset(z, (yy-60)/10, w, base_train)
    pred = np.maximum(60+10*(base_query+q@coef), 0.)
    assert np.isfinite(pred).all()
    audit.update(solver=info, weights_sha256=ahash(w), coefficient_sha256=ahash(coef),
                 offset_train_sha256=ahash(base_train), train_states=np.bincount(states(yy), minlength=3).tolist())
    return pred, audit
