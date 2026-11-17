"""Matched one-input linear/RBF MA comparison; standalone numerical runtime.

Read current saved inputs and frozen splits. Do not regenerate features, call
the historical experiment runners, or alter manuscript/delivery artifacts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, f1_score, r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/current_ma_20261103'
OUT = BASE / 'linear_rbf_ma_20261106'
MAN = ROOT / 'outputs/uteg_nc_goal_20261030/manuscript'
MODEL_DIR = ROOT / 'code_submit/ma_target_optimization'
ARMS = ['target_only_SA', 'HZ_to_SA', 'HZ+HB_to_SA']
FAMILIES = ['linear', 'rbf']
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
FIXED = ALPHAS.index(0.01)
SEEDS = list(range(1000, 1020))
COST = 4.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, Path): return str(obj)
    raise TypeError(type(obj))


def write(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=serial,
                               allow_nan=False) + '\n', encoding='utf8')


def states(y):
    y = np.asarray(y)
    assert np.isfinite(y).all()
    return np.where(y < 50, 0, np.where(y <= 70, 1, 2))


def functions(path, names, env):
    """Reuse only named numerical definitions, without heavy runner imports."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def numerical_runtime():
    common = dict(np=np, hashlib=hashlib, states=states)
    wenv = functions(MODEL_DIR/'models.py', ['weights'], dict(common))
    lenv = functions(MODEL_DIR/'density_reweight_mapping.py', ['label_weights'],
                     dict(common, weights=wenv['weights']))
    loss = functions(MODEL_DIR/'asymmetric_ma_mapping.py', ['loss_terms'], dict(common))
    solver = functions(MODEL_DIR/'full_curve_cost_sensitive.py', ['solve'],
                       dict(common, loss_terms=loss['loss_terms']))
    design = functions(MODEL_DIR/'all301_internal_models.py', ['design', 'weights'],
                       dict(common, label_weights=lenv['label_weights']))
    models = SimpleNamespace(solve=solver['solve'], weights=design['weights'], design=design['design'])
    runtimes = {}
    for alpha in ALPHAS:
        env = dict(common, models=models, loss_terms=loss['loss_terms'], ALPHA=alpha,
                   COST=COST, ahash=ahash)
        runtimes[alpha] = functions(MODEL_DIR/'all301_b085_inward4_transfer.py',
            ['offset_objective', 'solve_offset', 'source_basis', 'source_predict', 'fit_source'], env)
    return models, loss['loss_terms'], runtimes, wenv['weights']


MODELS, LOSS, NUM, BASE_WEIGHTS = numerical_runtime()


def load_inputs():
    pointer = read(ROOT/'outputs/ACTIVE_EXPERIMENT_DATASET.json')
    path = Path(pointer['dataset_path'])
    assert path.resolve() == (BASE/'data/DATA15_CURRENT.npz').resolve()
    assert sha(path) == pointer['sha256']
    with np.load(path, allow_pickle=False) as z:
        d = {k:z[k].copy() for k in ['x', 'medical', 'ids', 'groups', 'centers', 'input_qc']}
    d['y'] = d['medical']
    folds = read(BASE/'data/FROZEN_SPLITS.json')
    role = read(BASE/'data/ROLE_FREEZE.json')['roles']['SAHZU']
    plan = read(BASE/'data/SOURCE_PLAN.json')['source_rows']
    source = {a:np.asarray(plan.get(a, []), dtype=int) for a in ARMS}
    source = {a:rows[d['input_qc'][rows]] for a, rows in source.items()}
    local = np.sort(np.r_[folds[0]['A'], folds[0]['B']])
    assert len(local) == 301 and np.all(d['input_qc'][local])
    assert len(role['pool_full']) == 60 and len(role['test']) == 201
    for a in ARMS:
        assert not set(d['groups'][source[a]]) & set(d['groups'][local])
    seen = np.zeros(len(d['x']), int)
    for f in folds:
        a, b = np.asarray(f['A']), np.asarray(f['B'])
        assert not set(d['groups'][a]) & set(d['groups'][b])
        assert set(a) | set(b) == set(local)
        seen[b] += 1
        validate_inner(d, a, f['inner'])
    assert np.all(seen[local] == 1)
    for seed in SEEDS:
        validate_inner(d, np.asarray(role['pool_full']), role['positions_full'][str(seed)][3]['inner'])
    return d, folds, role, source, local


def validate_inner(d, train, inner):
    seen = {int(i):0 for i in train}
    for sub in inner:
        a, b = np.asarray(sub['T']), np.asarray(sub['V'])
        assert set(a) | set(b) == set(train)
        assert not set(d['groups'][a]) & set(d['groups'][b])
        for i in b: seen[int(i)] += 1
    assert all(v == 1 for v in seen.values())


def validation_weights(train_y, val_y):
    """Apply class costs estimated from inner TRAINING records to validation."""
    c = np.bincount(states(train_y), minlength=3)
    inverse = 1/np.maximum(c, 1)
    maximum_present = np.max(inverse[c > 0])
    inverse = np.maximum(inverse, maximum_present/20.)
    inverse /= np.mean(inverse[states(train_y)])
    return inverse[states(val_y)]


class Engine:
    def __init__(self, data, sources):
        self.d = data
        self.x = data['x'][:, 1:2].copy()
        self.y = data['medical'][:, 2].copy()
        self.sources = sources
        self.source_cache = {}
        self.design_cache = {}
        self.fit_count = 0
        self.max_gradient = 0.

    def source(self, family, arm):
        key = (family, arm)
        if key in self.source_cache: return self.source_cache[key]
        rows = self.sources[arm]
        if not len(rows):
            self.source_cache[key] = None
            return None
        y = self.y[rows]
        if family == 'rbf':
            model, audit = NUM[.01]['fit_source'](dict(self.d, y=self.d['medical']), self.x, rows)
        else:
            mean = self.x[rows].mean(0)
            scale = np.maximum(self.x[rows].std(0), 1e-6)
            z = np.c_[(self.x[rows]-mean)/scale, np.ones(len(rows))]
            w = np.zeros(len(rows))
            for center in np.unique(self.d['centers'][rows]):
                mask = self.d['centers'][rows] == center
                w[mask] = MODELS.weights(y[mask], 0, 'inward4')/mask.sum()
            w /= w.mean()
            coef, info = MODELS.solve(z, (y-60)/10, w, 'residual', COST, alpha=.01)
            model = dict(mean=mean, scale=scale, coef=coef)
            audit = dict(rows=rows, states=np.bincount(states(y), minlength=3), solver=info,
                         weights_sha256=ahash(w), model_sha256=ahash(coef))
        self.source_cache[key] = (model, audit)
        self.max_gradient = max(self.max_gradient, audit['solver']['gradient_max'])
        return model, audit

    def source_predictions(self, family, source, rows):
        if source is None: return np.full(len(rows), 60.)
        model, _ = source
        if family == 'rbf': return NUM[.01]['source_predict'](model, self.x[rows])
        q = np.c_[(self.x[rows]-model['mean'])/model['scale'], np.ones(len(rows))]
        return np.maximum(60+10*q@model['coef'], 0.)

    def design(self, family, train, query):
        key = (family, tuple(train), tuple(query))
        if key in self.design_cache: return self.design_cache[key]
        assert not set(self.d['groups'][train]) & set(self.d['groups'][query])
        if family == 'rbf':
            labels = np.full_like(self.d['medical'], np.nan)
            labels[train] = self.d['medical'][train]
            z, q, y, ns, au = MODELS.design(dict(self.d, y=labels), [self.x], train, query, 'target_only')
            assert ns == 0
        else:
            mean = self.x[train].mean(0)
            scale = np.maximum(self.x[train].std(0), 1e-6)
            z = np.c_[(self.x[train]-mean)/scale, np.ones(len(train))]
            q = np.c_[(self.x[query]-mean)/scale, np.ones(len(query))]
            y = self.y[train]
            au = dict(scaling_sha256=[hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()],
                      n_source=0, n_local=len(train), n_basis=1,
                      local_rows_sha256=ahash(np.asarray(train, dtype='int64')))
        self.design_cache[key] = z, q, y, au
        return z, q, y, au

    def predict(self, family, arm, train, query, alpha):
        train, query = np.asarray(train, int), np.asarray(query, int)
        assert np.all(self.d['input_qc'][train]) and np.all(self.d['input_qc'][query])
        assert not set(self.d['groups'][self.sources[arm]]) & set(self.d['groups'][query])
        z, q, y, au = self.design(family, train, query)
        w = MODELS.weights(y, 0, 'inward4')
        source = self.source(family, arm)
        offset = (self.source_predictions(family, source, train)-60)/10
        bq = (self.source_predictions(family, source, query)-60)/10
        coef, info = NUM[alpha]['solve_offset'](z, (y-60)/10, w, offset)
        prediction = np.maximum(60+10*(bq+q@coef), 0.)
        self.fit_count += 1
        self.max_gradient = max(self.max_gradient, info['gradient_max'])
        assert np.isfinite(prediction).all()
        return prediction, dict(**au, solver=info, alpha=alpha, weights_sha256=ahash(w),
                                coefficient_sha256=ahash(coef)), coef

    def evaluate(self, family, arm, train, query, inner):
        train, query = np.asarray(train, int), np.asarray(query, int)
        pos = {int(v):i for i,v in enumerate(train)}
        oof = np.full((len(ALPHAS), len(train)), np.nan)
        val_w = np.full(len(train), np.nan)
        audits = []
        for sub in inner:
            a, b = np.asarray(sub['T']), np.asarray(sub['V'])
            ix = [pos[int(v)] for v in b]
            val_w[ix] = validation_weights(self.y[a], self.y[b])
            for j, alpha in enumerate(ALPHAS):
                p, au, _ = self.predict(family, arm, a, b, alpha)
                oof[j, ix] = p
                audits.append(au)
        assert np.isfinite(oof).all() and np.isfinite(val_w).all()
        loss = [float(np.average(LOSS((p-60)/10, (self.y[train]-60)/10,
                    np.ones(len(train), bool), 'residual', COST)[0], weights=val_w)) for p in oof]
        best = min(loss)
        chosen = max(j for j, v in enumerate(loss) if v <= best + 1e-12)
        prediction = []
        for alpha in ALPHAS:
            p, au, _ = self.predict(family, arm, train, query, alpha)
            prediction.append(p)
            audits.append(au)
        return dict(prediction=np.asarray(prediction), oof=oof, chosen=chosen,
                    selected_alpha=ALPHAS[chosen], development_loss=loss, audits=audits,
                    validation_weights_sha256=ahash(val_w))


def metrics(y, p):
    p = np.asarray(p, float)
    y = np.broadcast_to(np.asarray(y, float), p.shape).ravel()
    p = p.ravel()
    assert np.isfinite(p).all()
    t, q = states(y), states(p)
    matrix = confusion_matrix(t, q, labels=[0,1,2])
    fn = int(np.sum((t != 1) & (q == 1)))
    fp = int(np.sum((t == 1) & (q != 1)))
    n_ab, n_no = int(np.sum(t != 1)), int(np.sum(t == 1))
    recalls = matrix.diagonal()/matrix.sum(1)
    r2 = float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))
    assert abs(r2-r2_score(y,p)) < 1e-12
    assert np.trace(matrix) == np.sum(t == q)
    return dict(n=len(y), mae=float(np.mean(abs(p-y))), r2=r2,
                rmse=float(np.sqrt(np.mean((p-y)**2))), agreement=float(np.mean(t==q)),
                low_recall=float(recalls[0]), high_recall=float(recalls[2]),
                low_correct=int(matrix[0,0]), high_correct=int(matrix[2,2]),
                fn=fn, fn_denominator=n_ab, fnr=fn/n_ab, fp=fp, fp_denominator=n_no, fpr=fp/n_no,
                confusion=matrix, binary_confusion=[[n_no-fp,fp],[fn,n_ab-fn]],
                balanced_accuracy=float(recalls.mean()), macro_f1=float(f1_score(t,q,average='macro')),
                class_mae={name:float(np.mean(abs(p[t==i]-y[t==i]))) for i,name in enumerate(['low','normal','high'])})


def freeze(d, folds, role, sources, local):
    OUT.mkdir(exist_ok=True)
    paths = [BASE/'data'/n for n in ['DATA15_CURRENT.npz','FROZEN_SPLITS.json','ROLE_FREEZE.json','SOURCE_PLAN.json']]
    paths += [ROOT/'outputs/ACTIVE_EXPERIMENT_DATASET.json', Path(__file__)]
    paths += [MODEL_DIR/n for n in ['models.py','density_reweight_mapping.py','asymmetric_ma_mapping.py',
                                  'full_curve_cost_sensitive.py','all301_internal_models.py','all301_b085_inward4_transfer.py']]
    paths += [BASE/'all301/T900.npz']+[BASE/'primary'/f'{a}_T900_B0.npz' for a in ARMS]
    protocol = dict(status='FROZEN_BEFORE_NEW_FITTING', horizon_minutes=15, input='Stored U-TEG MA, x[:,1], only; unchanged active file.',
        families=FAMILIES, arms=ARMS, fixed_alpha=.01, source_alpha=.01, local_alpha_grid=ALPHAS,
        fixed_comparison='Source and local coefficients use alpha=.01 for both families; RBF must reproduce saved baseline.',
        development_selected_comparison='For both families, select LOCAL alpha by training-weighted asymmetric validation loss over the frozen inner folds; source alpha remains .01. Ties within1e-12 prefer larger alpha.',
        source='Same source rows; source-only standardization, class weighting within each hospital, equal hospital mass. Linear source plus linear local correction versus RBF source plus RBF local correction.',
        loss='Same mean weighted squared loss in (MA-60)/10 coordinates. Multiply low-MA overestimation and high-MA underestimation by4. Apply to final prediction, not residual target.',
        weights='Same inverse class frequency weights, ratio cap20, mean1; no query labels in weights. Validation class weights estimated from each inner training set.',
        regularization='L2 coefficient penalty and free intercept. RBF RKHS coordinates; standardized linear slope. Same alpha is a controlled setting, not a claim of identical effective complexity.',
        protocols=dict(paper='Original60 development +201 test; seeds1000-1019 retained; repeated events are not independent records.',
                       internal301='Existing grouped5outer/3inner; each of301 local records predicted once per family/arm/procedure.'),
        selection_scope='Development data only. All fixed and selected results reported; no test-based winner promotion.',
        reporting='All-report mapping evaluation; no gate, no reporting threshold fitting. MA boundaries50 and70 unchanged.',
        source_counts={a:len(r) for a,r in sources.items()}, local_classes=np.bincount(states(d['medical'][local,2]),minlength=3),
        input_and_code_hashes={str(p):sha(p) for p in paths}, manuscript_changed=False)
    if (OUT/'PROTOCOL.json').exists():
        assert read(OUT/'PROTOCOL.json') == json.loads(json.dumps(protocol,default=serial))
    else:
        write(OUT/'PROTOCOL.json',protocol)
        write(OUT/'DELIVERY_BEFORE.json',{str(p):sha(p) for p in MAN.rglob('*') if p.is_file()})
    return protocol


def synthetic_checks():
    rng=np.random.default_rng(20261106)
    x=np.linspace(-2,2,35)
    z=np.c_[x,np.ones(len(x))]
    y=1.2*x+0.2+rng.normal(0,.1,len(x))
    w=MODELS.weights(60+10*y,0,'inward4')
    offset=.3*x
    coef,au=NUM[.01]['solve_offset'](z,y,w,offset)
    fun=NUM[.01]['offset_objective']
    analytic=fun(coef,z,y,w,offset)[1]
    eps=1e-6
    finite=np.array([(fun(coef+np.eye(2)[i]*eps,z,y,w,offset)[0]-fun(coef-np.eye(2)[i]*eps,z,y,w,offset)[0])/(2*eps) for i in range(2)])
    np.testing.assert_allclose(analytic,finite,atol=1e-7,rtol=0)
    independent=minimize(lambda b:fun(b,z,y,w,offset)[:2],np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-10})
    np.testing.assert_allclose(z@coef,z@independent.x,atol=1e-7,rtol=0)
    truth=np.array([-2.,-2.,2.,2.,0.])
    pred=truth+np.array([.2,-.2,-.2,.2,.2])
    np.testing.assert_allclose(LOSS(pred,truth,np.ones(5,bool),'residual',4)[0],[.16,.04,.16,.04,.04])
    np.testing.assert_allclose(validation_weights(60+10*y,60+10*y),w,atol=1e-14,rtol=0)
    return dict(gradient_finite_difference=True,independent_bfgs_agrees=True,loss_directions_correct=True,
                validation_weights_from_training_only=True)


def smoke(d, folds, role, sources, local):
    assert not (OUT/'RESULTS.json').exists()
    check=synthetic_checks()
    engine=Engine(d,sources)
    f=folds[0]; a=np.asarray(f['A']); b=np.asarray(f['B'])
    with np.load(BASE/'all301/T900.npz') as z:
        assert np.array_equal(z['local'],local)
        index={int(v):i for i,v in enumerate(local)}
        ix=[index[int(i)] for i in b]
        for j,arm in enumerate(ARMS):
            p,_,_=engine.predict('rbf',arm,a,b,.01)
            np.testing.assert_array_equal(p,z['prediction'][j,ix])
    for arm in ARMS:
        with np.load(BASE/'primary'/f'{arm}_T900_B0.npz') as z:
            a0=np.asarray(role['pool_full']); b0=np.asarray(role['test'])
            assert np.array_equal(a0,z['training'][0]) and np.array_equal(b0,z['test'])
            p,_,_=engine.predict('rbf',arm,a0,b0,.01)
            np.testing.assert_array_equal(p,z['prediction'][0])
    # Fit and select from fresh engines; no cached coefficients hide leakage.
    for family in FAMILIES:
        for arm in ARMS:
            one=Engine(d,sources).evaluate(family,arm,a,b,f['inner'])
            allowed=np.unique(np.r_[a,sources[arm]])
            poisoned={k:v.copy() for k,v in d.items()}
            forbidden=np.setdiff1d(np.arange(len(d['x'])),allowed)
            poisoned['medical'][forbidden]=1e6
            poisoned['y']=poisoned['medical']
            two=Engine(poisoned,sources).evaluate(family,arm,a,b,f['inner'])
            for key in ['prediction','oof']:np.testing.assert_array_equal(one[key],two[key])
            assert one['chosen']==two['chosen'] and one['development_loss']==two['development_loss']
            assert one['audits']==two['audits']
            # Query features may affect predictions, but not fit/selection.
            poisoned['x'][b,1]+=100
            three=Engine(poisoned,sources).evaluate(family,arm,a,b,f['inner'])
            assert one['audits']==three['audits'] and one['development_loss']==three['development_loss']
            assert one['chosen']==three['chosen']
            # The other three input coefficients have no route into this fit.
            other={k:v.copy() for k,v in d.items()};other['x'][:,[0,2,3]]=1e6
            four=Engine(other,sources).evaluate(family,arm,a,b,f['inner'])
            np.testing.assert_array_equal(one['prediction'],four['prediction'])
            assert one['development_loss']==four['development_loss']
            print('SMOKE',family,arm,flush=True)
    check.update(rbf_first_outer_fold_exact=True,rbf_original201_exact=True,forbidden_label_poison_invariant=True,
                 query_features_do_not_fit_transform_or_select_alpha=True,other_three_coefficients_unused=True,
                 all_groups_disjoint=True,code_sha256=sha(__file__))
    write(OUT/'SMOKE.json',check)
    return check


def fit_all(d, folds, role, sources, local):
    assert read(OUT/'SMOKE.json')['code_sha256']==sha(__file__)
    assert not (OUT/'PREDICTIONS_SEALED.json').exists()
    engine=Engine(d,sources)
    (OUT/'private').mkdir(exist_ok=True)
    full=np.full((2,3,5,len(local)),np.nan)
    chosen=np.full((2,3,len(local)),-1,int)
    pos={int(v):i for i,v in enumerate(local)}
    audits=[]; started=time.time()
    for fi,f in enumerate(folds):
        ix=np.array([pos[int(v)] for v in f['B']])
        for mi,family in enumerate(FAMILIES):
            for ai,arm in enumerate(ARMS):
                v=engine.evaluate(family,arm,f['A'],f['B'],f['inner'])
                full[mi,ai][:,ix]=v['prediction'];chosen[mi,ai,ix]=v['chosen']
                audits.append(dict(protocol='internal301',fold=fi,family=family,arm=arm,
                    train=f['A'],query=f['B'],**{k:val for k,val in v.items() if k not in ['prediction','oof']}))
        print('OUTER_FOLD_DONE',fi+1,'seconds',round(time.time()-started,1),flush=True)
    selected=np.take_along_axis(full,chosen[:,:,None,:],axis=2)[:,:,0,:]
    assert np.isfinite(full).all()
    with np.load(BASE/'all301/T900.npz') as z:np.testing.assert_array_equal(full[1,:,FIXED],z['prediction'])
    np.savez_compressed(OUT/'private/INTERNAL301.npz',prediction=full,selected=selected,chosen=chosen,
                        rows=local,reference=d['medical'][local,2],alphas=ALPHAS)
    pfull=np.full((2,3,len(SEEDS),5,len(role['test'])),np.nan)
    pchosen=np.full((2,3,len(SEEDS)),-1,int)
    for si,seed in enumerate(SEEDS):
        inner=role['positions_full'][str(seed)][3]['inner']
        for mi,family in enumerate(FAMILIES):
            for ai,arm in enumerate(ARMS):
                v=engine.evaluate(family,arm,role['pool_full'],role['test'],inner)
                pfull[mi,ai,si]=v['prediction'];pchosen[mi,ai,si]=v['chosen']
                audits.append(dict(protocol='paper',seed=seed,family=family,arm=arm,
                    train=role['pool_full'],query=role['test'],**{k:val for k,val in v.items() if k not in ['prediction','oof']}))
        print('PAPER_SEED_DONE',seed,'seconds',round(time.time()-started,1),flush=True)
    pselected=np.take_along_axis(pfull,np.broadcast_to(pchosen[:,:,:,None,None],(2,3,20,1,201)),axis=3)[:,:,:,0,:]
    for ai,arm in enumerate(ARMS):
        with np.load(BASE/'primary'/f'{arm}_T900_B0.npz') as z:np.testing.assert_array_equal(pfull[1,ai,:,FIXED],z['prediction'])
    np.savez_compressed(OUT/'private/PAPER201.npz',prediction=pfull,selected=pselected,chosen=pchosen,
                        rows=role['test'],reference=d['medical'][role['test'],2],seeds=SEEDS,alphas=ALPHAS)
    write(OUT/'private/FIT_AUDIT.json',audits)
    write(OUT/'PREDICTIONS_SEALED.json',dict(files={str(p):sha(p) for p in (OUT/'private').glob('*')},
        rbf_fixed_predictions_exact=True,fit_count=engine.fit_count,max_gradient=engine.max_gradient,
        elapsed_seconds=time.time()-started,scoring_started=False))


def paired(y, linear, rbf):
    t=states(np.broadcast_to(y,linear.shape));l=states(linear);r=states(rbf)
    lf=(t!=1)&(l==1);rf=(t!=1)&(r==1)
    return dict(linear_recovers_rbf_fn=int(np.sum(rf&~lf)),linear_new_fn=int(np.sum(lf&~rf)),
                linear_mae_minus_rbf=float(np.mean(abs(linear-y)-abs(rbf-y))),
                state_disagreements=int(np.sum(l!=r)))


def score():
    sealed=read(OUT/'PREDICTIONS_SEALED.json')
    assert all(sha(p)==h for p,h in sealed['files'].items())
    result={}; deltas={}; choices={}
    for protocol,file in [('paper','PAPER201.npz'),('internal301','INTERNAL301.npz')]:
        with np.load(OUT/'private'/file) as z:
            y=z['reference'];p=z['prediction'];s=z['selected'];pick=z['chosen']
            fixed=p[:,:,0,FIXED] if protocol=='paper' else p[:,:,FIXED]
            if protocol=='paper':np.testing.assert_array_equal(p[:,:,:,FIXED],np.repeat(fixed[:,:,None,:],20,axis=2))
            result[protocol]={};deltas[protocol]={};choices[protocol]={}
            for mode,values in [('fixed',fixed),('development_selected',s)]:
                result[protocol][mode]={a:{family:metrics(y,values[mi,ai]) for mi,family in enumerate(FAMILIES)} for ai,a in enumerate(ARMS)}
                deltas[protocol][mode]={a:paired(y,values[0,ai],values[1,ai]) for ai,a in enumerate(ARMS)}
            for mi,family in enumerate(FAMILIES):
                choices[protocol][family]={}
                for ai,arm in enumerate(ARMS):
                    rows={str(alpha):int(np.sum(pick[mi,ai]==j)) for j,alpha in enumerate(ALPHAS)}
                    choices[protocol][family][arm]=rows
            if protocol=='paper':
                result[protocol]['selected_seed_ranges']={a:{family:{metric:[float(min(v[metric] for v in ms)),float(max(v[metric] for v in ms))] for metric in ['mae','r2','agreement','low_recall','high_recall','fnr','fpr']} for family,ms in [(family,[metrics(y,s[mi,ai,k]) for k in range(20)]) for mi,family in enumerate(FAMILIES)]} for ai,a in enumerate(ARMS)}
    write(OUT/'RESULTS.json',result);write(OUT/'PAIRED_CHANGES.json',deltas);write(OUT/'ALPHA_SELECTION.json',choices)
    fields=['protocol','comparison','arm','model','n','mae','r2','rmse','agreement','low_recall','high_recall','fn','fn_denominator','fnr','fp','fp_denominator','fpr','balanced_accuracy','macro_f1']
    with (OUT/'RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for protocol in result:
            for mode in ['fixed','development_selected']:
                for arm in ARMS:
                    for family in FAMILIES:
                        m=result[protocol][mode][arm][family]
                        writer.writerow(dict(protocol=protocol,comparison=mode,arm=arm,model=family,**{k:m[k] for k in fields[4:]}))
    names={'target_only_SA':'Target-only','HZ_to_SA':'单源 Huzhou→SAHZU','HZ+HB_to_SA':'双源 Huzhou+Hubei→SAHZU'}
    lines=['# 只用U-TEG MA的线性/RBF公平对照','',
        '输入直接读取当前活动数据文件的U-TEG MA列；未调整数据，论文和交付材料未修改。全部结果为15分钟、门前all-report评价，覆盖率100%。',
        '线性和RBF沿用相同划分、训练记录、类别权重、4倍方向性损失、50/70 mm状态边界以及非负输出约束。迁移各使用本模型的源预测和同类型本地加性校正；两院源训练总权重相同。',
        '主比较固定源模型和本地校正的正则系数0.01。辅助比较固定源系数0.01，双方均从0.0001、0.001、0.01、0.1、1中按内层加权不对称验证损失选择本地系数；验证权重的类别频率只来自对应内层训练折。',
        '固定相同正则系数用于控制比较条件，不代表线性斜率惩罚与RBF的RKHS惩罚具有完全相同的有效复杂度。RBF带宽沿用当前设定，不进行额外搜索。','']
    for protocol,label in [('paper','原论文：60条开发＋201条评价'),('internal301','301条分组内部评价')]:
        lines += ['## '+label,'']
        lines += [('固定参数主表中每条201测试记录只计一次：7低、170正常、24高。双方20次最终固定参数预测均相同。开发侧选参结果按20个开发划分汇总；4,020为重复记录事件，不能视为独立样本。' if protocol=='paper' else '沿用5折外层、3折内层：11低、253正常、37高。每条记录恰好得到一次外层预测；不与原201条评价合并。'),'']
        for mode,title in [('fixed','固定现有参数'),('development_selected','双方均由开发侧选择本地正则系数')]:
            lines += ['### '+title,'','|流程|模型|MAE(mm)|R²|低MA召回|高MA召回|FNR|FPR|三分类一致率|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for arm in ARMS:
                for family in FAMILIES:
                    m=result[protocol][mode][arm][family]
                    lines += [f"|{names[arm]}|{family}|{m['mae']:.3f}|{m['r2']:.3f}|{m['low_recall']:.2%}|{m['high_recall']:.2%}|{m['fnr']:.2%} ({m['fn']}/{m['fn_denominator']})|{m['fpr']:.2%} ({m['fp']}/{m['fp_denominator']})|{m['agreement']:.2%}|"]
            lines += ['']
        lines += ['### 固定参数的逐类MAE与三分类矩阵','', '矩阵行是真实低/正常/高，列为预测低/正常/高。','']
        for arm in ARMS:
            for family in FAMILIES:
                m=result[protocol]['fixed'][arm][family]
                lines += [f"- {names[arm]} / {family}：低、正常、高MAE分别为 {m['class_mae']['low']:.3f}、{m['class_mae']['normal']:.3f}、{m['class_mae']['high']:.3f} mm；矩阵 `{m['confusion'].tolist()}`。"]
        lines += ['']
    lines += ['## 检验与口径','',
        '- 在完整两套评价中，固定参数RBF逐值复现既有预测；没有用近似重实现替代现有基线。',
        '- 小范围测试包括损失方向、有限差分梯度、独立优化器一致性，以及留出标签/输入扰动隔离。',
        '- FNR＝真实低或高MA却预测为正常的比例；FPR＝真实正常却预测为低或高的比例。低、高召回分别要求预测方向正确。',
        '- 原论文20个开发划分的选参范围见RESULTS.json中的selected_seed_ranges；逐折、逐seed选参记录见private/FIT_AUDIT.json。',
        '- 新比较仍使用已被反复研究的既有队列，属于补充的回顾性内部比较。',
        '- 表中包括所有预先指定的固定/开发侧选参方案，未按测试表现挑选参数或隐藏不占优方案。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result['paper']['fixed'],ensure_ascii=False,default=serial),flush=True)
    print(json.dumps(result['internal301']['fixed'],ensure_ascii=False,default=serial),flush=True)


def verify_unchanged():
    protocol=read(OUT/'PROTOCOL.json')
    assert all(sha(p)==h for p,h in protocol['input_and_code_hashes'].items())
    assert all(sha(p)==h for p,h in read(OUT/'DELIVERY_BEFORE.json').items())
    write(OUT/'VERIFICATION.json',dict(input_and_existing_predictions_unchanged=True,manuscript_and_delivery_unchanged=True,
        fixed_rbf_exact_replay_both_protocols=True,smoke=read(OUT/'SMOKE.json'),seal=read(OUT/'PREDICTIONS_SEALED.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','full','score']);args=ap.parse_args()
    d,folds,role,sources,local=load_inputs()
    freeze(d,folds,role,sources,local)
    if args.phase=='smoke':smoke(d,folds,role,sources,local)
    elif args.phase=='full':
        fit_all(d,folds,role,sources,local)
        score();verify_unchanged()
    else:score();verify_unchanged()


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
