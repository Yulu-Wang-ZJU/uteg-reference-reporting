"""Full-waveform RKHS regression with explicit abnormal inward-error costs.

RBF basis/normalization/weights use training records only. Ordinary MSE is an
exact kernel-ridge control; the clinical directional losses are study-specific
surrogates, not a claim to an established FNR guarantee.
"""
import argparse
import hashlib
import json
import time
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from .feature_combination_audit import load, assemble, detailed
from .common_clock import load_preprocessor
from .asymmetric_ma_mapping import loss_terms
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

LOSSES = [('control', 1.), ('residual', 4.), ('boundary', 4.), ('boundary', 16.), ('boundary', 64.)]
RECIPES = ['target_uniform', 'target_class', 'HB_dense']


def load_waveforms(seeds):
    data, parts, extra, x, hashes = load(seeds)
    path = ROOT/'outputs/uteg_three_center_validation_20261021_174409/private/INPUT_RAW_AND_REFERENCE.npz'
    rows = data['allowed_training_indices']
    with np.load(path, allow_pickle=False) as z:
        raw, observed = z['original'][rows], z['observed'][rows]
    pre, prepath = load_preprocessor()
    hashes.update({str(path):sha256(path), str(prepath):sha256(prepath)})
    reps = {k:np.full((len(x), 901), np.nan) for k in ['raw900', 'smooth900']}
    mask = np.zeros((len(x), 900), bool)
    for row, signal, obs in zip(rows, raw, observed):
        n = int(obs[:900].sum())
        assert np.array_equal(np.flatnonzero(obs[:900]), np.arange(n))
        raw_values = signal[:n]-np.median(signal[:10])
        smooth, _ = pre(signal[:n], q=None)
        mask[row, :n] = True
        for key, v in [('raw900', raw_values), ('smooth900', smooth)]:
            reps[key][row] = np.r_[np.pad(v, (0, 900-n), mode='edge'), n/900]
    assert all(np.isnan(v[parts[1000]['test']]).all() for v in reps.values())
    return data, parts, extra, reps, mask, hashes


def kernel_design(train, query):
    gamma = 1/train.shape[1]
    K = rbf_kernel(train, gamma=gamma)
    values, vectors = np.linalg.eigh(K)
    keep = values > max(1e-12, values.max()*1e-12)
    root = np.sqrt(values[keep])
    V = vectors[:, keep]
    z = np.c_[V*root, np.ones(len(train))]
    q = np.c_[rbf_kernel(query, train, gamma=gamma)@(V/root), np.ones(len(query))]
    return z, q, K


def solve(z, y, w, kind, cost, initial=None, alpha=.01):
    penalty = np.r_[np.ones(z.shape[1]-1)*alpha, 0.]
    if initial is None:
        initial = np.linalg.solve(z.T@(w[:,None]*z)/len(y)+np.diag(penalty)+1e-12*np.eye(z.shape[1]), z.T@(w*y)/len(y))
    coef = initial.copy()
    all_rows = np.ones(len(y), bool)

    def objective(c):
        value, gradient, curvature = loss_terms(z@c, y, all_rows, kind, cost)
        f = np.mean(w*value)+np.sum(penalty*c*c)
        g = z.T@(w*gradient)/len(y)+2*penalty*c
        H = 2*z.T@((w*curvature)[:,None]*z)/len(y)+2*np.diag(penalty)
        return float(f), g, H

    for iteration in range(200):
        value, g, H = objective(coef)
        if max(abs(g)) < 1e-8:
            break
        step = np.linalg.solve(H+1e-12*np.eye(len(coef)), -g)
        slope = g@step
        if slope >= 0:
            raise RuntimeError('Non-descent kernel fit')
        rate = 1.
        for _ in range(50):
            proposal = coef+rate*step
            if objective(proposal)[0] <= value+1e-4*rate*slope+1e-14:
                coef = proposal
                break
            rate *= .5
        else:
            raise RuntimeError('Kernel line search failed')
    value, g, _ = objective(coef)
    if max(abs(g)) >= 1e-7 or not np.isfinite(coef).all():
        raise RuntimeError('Kernel fit not stationary')
    return coef, dict(iterations=iteration, objective=value, gradient_max=float(max(abs(g))))


def fit_predict(data, x, a, b, recipe):
    train, y, w, mean, scale, ns = assemble(data, x, a, recipe)
    train = train[:, :-1]
    query = (x[b]-mean)/scale
    if ns:
        query = np.c_[query, np.zeros(len(b))]
    z, query, _ = kernel_design(train, query)
    yy = (y-60)/10
    initial, _ = solve(z, yy, w, 'control', 1.)
    predictions = []
    diagnostics = []
    for kind, cost in LOSSES:
        coef, info = solve(z, yy, w, kind, cost, initial)
        predictions.append(np.maximum(60+10*query@coef, 0))
        info['coefficient_sha256'] = hashlib.sha256(coef.tobytes()).hexdigest()
        info['training_low_n'] = int(np.sum(y<50))
        info['training_high_n'] = int(np.sum(y>70))
        diagnostics.append(info)
    return np.asarray(predictions), diagnostics


def configs():
    return [dict(representation=r, recipe=w, kind=k, cost=c)
            for r in ['raw900','smooth900'] for w in RECIPES for k,c in LOSSES]


def all_predict(data, reps, a, b):
    results = []
    diagnostics = []
    for x in reps.values():
        for recipe in RECIPES:
            p, d = fit_predict(data, x, a, b, recipe)
            results.append(p)
            diagnostics.extend(d)
    return np.concatenate(results), diagnostics


def run(seeds, name):
    out = ROOT/'outputs/uteg_mapping_target_runs'/name
    (out/'private').mkdir(parents=True, exist_ok=True)
    data, parts, extra, reps, mask, hashes = load_waveforms(seeds)
    grid = configs()
    protocol = dict(scope='C100: same60 target outer queries, additional40 historical training-only, fixed3x3 grouped folds. Original201 masked.',
        seeds=list(parts), candidates=grid,
        waveform='All900 first-channel values after own first10 baseline; optional SG11, no P5. Preserve voltage and time. Short source curves use last-value storage completion plus observed fraction, not invented measurements. All target curves fully observed.',
        kernel='RBF gamma1/p, eigendecomposition of TRAINING kernel only, exact RKHS norm penalty alpha=.01 with free intercept. Target-only uniform/inverse-class and HB79+target DenseWeight recipes; equal base source/target loss mass and source indicator.',
        loss='Mean weighted squared MA error in (MA-60)/10 units; residual4 increases inward errors for low/high; boundary4/16/64 adds hinges toward correct side of49.5/70.5 and unit normal-outside50/70 hinge. Applies to both source and target training labels. Opposite-tail errors remain penalized.',
        selection='Inner grouped predictions select separately by MSE and by MAE qualified agreement>80/FNR<5/FNR<FPR ordering; unqualified accuracy>80 selects smallest FNR then opposite-tail errors/FPR. Outer labels evaluate only.',
        interpretation='Exploratory development candidates, not untouched validation; repeated seeds not new patients.',
        input_hashes=hashes, code_sha256=sha256(__file__), test_evaluated=False, manuscript_changed=False,
        reference='https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    pool = parts[1000]['pool']
    pos = {int(v):j for j,v in enumerate(pool)}
    allpred = np.full((len(grid), seeds, len(pool)), np.nan)
    nested = {k:np.full((seeds,len(pool)), np.nan) for k in ['mse','safety']}
    ledger = []
    diagnostics = []
    begin = time.time()
    from .metrics import selection_key
    for si, (seed, part) in enumerate(parts.items()):
        for fi, fold in enumerate(part['folds']):
            a0, b = fold['A'], fold['B']
            additional = extra[~np.isin(data['groups'][extra], data['groups'][b])]
            a = np.r_[a0, additional]
            assert not set(data['groups'][a]).intersection(data['groups'][b])
            prediction, diag = all_predict(data, reps, a, b)
            ix = [pos[int(v)] for v in b]
            allpred[:,si,ix] = prediction
            diagnostics.append(dict(seed=seed,fold=fi,fits=diag))
            ipos = {int(v):j for j,v in enumerate(a0)}
            inner = np.full((len(grid),len(a0)),np.nan)
            for f in fold['inner']:
                tr, va = f['T'], f['V']
                ad = additional[~np.isin(data['groups'][additional],data['groups'][va])]
                tr = np.r_[tr,ad]
                assert not set(data['groups'][tr]).intersection(data['groups'][va])
                pp, _ = all_predict(data, reps, tr, va)
                inner[:,[ipos[int(v)] for v in va]] = pp
            assert np.isfinite(inner).all()
            y = data['y'][a0,2]
            mm = [detailed(y,p) for p in inner]
            bestmse = int(np.argmin(np.mean((inner-y)**2,axis=1)))
            bestsafety = min(range(len(grid)),key=lambda j:(selection_key(mm[j]),j))
            for key,j in [('mse',bestmse),('safety',bestsafety)]:
                nested[key][si,ix] = prediction[j]
                ledger.append(dict(seed=seed,fold=fi,objective=key,selected=grid[j],inner_metrics=mm[j]))
            if si==fi==0:
                poison = dict(data)
                poison['y'] = data['y'].copy()
                poison['y'][b] = np.nan
                changed = {k:v.copy() for k,v in reps.items()}
                for v in changed.values():
                    v[b] = 1e4
                _, replay = all_predict(poison,changed,a,b)
                assert replay == diag
        print('KERNEL_SEED',seed,'seconds',round(time.time()-begin,1),flush=True)
    yy = np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    records = [dict(candidate=c,metrics=detailed(yy,p)) for c,p in zip(grid,allpred)]
    selected = [dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()]
    for row,p in zip(records+selected,list(allpred)+list(nested.values())):
        true=np.where(yy<50,0,np.where(yy>70,2,1));pr=np.where(p<50,0,np.where(p>70,2,1))
        cm=[[int(np.sum((true==a)&(pr==b))) for b in range(3)] for a in range(3)]
        assert cm==row['metrics']['confusion']
        assert abs(row['metrics']['r2']-(1-np.sum((p-yy)**2)/np.sum((yy-yy.mean())**2)))<1e-12
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=allpred,reference=yy[0],pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    (out/'private/FIT_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    for path,digest in hashes.items():
        assert sha256(path)==digest
    results=dict(fixed=records,nested=selected,joint_passes=sum(v['metrics']['mapping_target_pass'] for v in records),
        checks=dict(query_inputs_labels_do_not_fit=True,independent_metrics=True,all_optimizers_stationary=True,input_hashes_unchanged=True),test_evaluated=False)
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--output',default='full_curve_cost_sensitive_20261102')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.seeds,args.output)
