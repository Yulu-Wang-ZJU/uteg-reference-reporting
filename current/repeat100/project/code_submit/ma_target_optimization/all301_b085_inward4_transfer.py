"""Matched B-only/inward4 source-pretraining diagnostic, not U-TEG validation.

Reuse the selected all301 target-only protocol exactly. Transfer uses a frozen
source RBF prior plus a local RBF correction. The asymmetric loss acts on FINAL
MA, not on the residual label. No outside label fits a model or report gate;
the deliberately reference-MA-derived B input remains explicitly an oracle.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, f1_score
from threadpoolctl import threadpool_limits

from . import all301_b_only_mix085 as old
from .asymmetric_ma_mapping import loss_terms
from .metrics import states

previous = old.previous
models = previous.models
ROOT = previous.contract.ROOT
OUT = ROOT / 'outputs/uteg_ma_mix_oracle_diagnostic_20261102/b_only085_inward4_transfer_all301'
DATA = ROOT / previous.contract.DEFAULT_DATA_PATH
ARMS = ('target_only', 'single_HZ', 'dual_HZ_HB')
ALPHA = .01
COST = 4.


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


def load():
    data, local, _, peak, hashes, _ = previous._load()
    with np.load(DATA, allow_pickle=False) as z:
        ids, centers, groups = z['ids'], z['centers'], z['groups']
        labels, qc = z['medical'].copy(), z['input_qc']
    np.testing.assert_array_equal(ids, data['ids'])
    np.testing.assert_array_equal(groups, data['groups'])
    sources = {c:np.flatnonzero((centers == c)&qc) for c in ('Huzhou', 'Hubei')}
    assert len(sources['Huzhou']) == 95 and len(sources['Hubei']) == 79
    allowed = np.r_[local, sources['Huzhou'], sources['Hubei']]
    masked = np.full_like(labels, np.nan); masked[allowed] = labels[allowed]
    data = dict(data, y=masked, centers=centers)
    peak = previous._raw_peak(data, allowed)
    folds = json.loads((previous.contract.OUT/'private/FROZEN_SPLITS.json').read_text(encoding='utf-8'))
    previous.contract.validate_splits(local, groups, np.r_[sources['Huzhou'], sources['Hubei']], folds)
    source_feature = np.full((len(ids), 1), np.nan)
    ranges = {}
    for center, rows in sources.items():
        feature, meta = old.feature(peak, masked[:, 2], rows, np.array([], int), .85)
        source_feature[rows] = feature[rows]
        ranges[center] = meta
    hashes[str(DATA)] = sha(DATA)
    hashes[str(old.OUT/'private/PREDICTIONS.npz')] = sha(old.OUT/'private/PREDICTIONS.npz')
    return data, local, peak, folds, sources, source_feature, ranges, hashes


def evaluate_fold(data, peak, fold, source_models):
    previous._validate_fold(data, fold)
    a, b = np.asarray(fold['A']), np.asarray(fold['B'])
    pos = {int(r):i for i,r in enumerate(a)}
    inner = np.full((3, len(a)), np.nan); outer = np.full((3, len(b)), np.nan)
    raw_prior = np.full((2, len(b)), np.nan)
    audits = []
    stages = [('inner'+str(i), np.asarray(f['T']), np.asarray(f['V'])) for i,f in enumerate(fold['inner'])]
    stages += [('outer', a, b)]
    for stage, train, query in stages:
        x, meta = old.feature(peak, data['y'][:, 2], train, query, .85)
        for j, arm in enumerate(ARMS):
            source = None if j == 0 else source_models[arm]
            pred, au = local_predict(data, x, train, query, source)
            if stage == 'outer':
                outer[j] = pred
                if j:
                    raw_prior[j-1] = source_predict(source, x[query])
            else:
                inner[j, [pos[int(r)] for r in query]] = pred
            audits.append(dict(stage=stage, arm=arm, feature=meta, **au))
    assert np.isfinite(inner).all() and np.isfinite(outer).all()
    accepted, threshold = models.boundary_gate(inner, outer)
    return outer, accepted, inner, raw_prior, dict(fits=audits, threshold=threshold.tolist())


def synthetic_tests():
    rng = np.random.default_rng(8831)
    z = np.c_[rng.normal(size=(12, 4)), np.ones(12)]
    y = np.array([-1.7, -1.4, -1.2, -.6, -.2, 0, .4, .8, 1.1, 1.3, 1.6, 1.9])
    w = np.linspace(.5, 1.5, 12); offset = rng.normal(0, .2, 12)
    c = rng.normal(0, .3, 5)
    value, gradient, hessian = offset_objective(c, z, y, w, offset)
    eps = 1e-6; eye = np.eye(5)
    numerical_g = np.array([(offset_objective(c+eps*e,z,y,w,offset)[0]-offset_objective(c-eps*e,z,y,w,offset)[0])/(2*eps) for e in eye])
    numerical_h = np.column_stack([(offset_objective(c+eps*e,z,y,w,offset)[1]-offset_objective(c-eps*e,z,y,w,offset)[1])/(2*eps) for e in eye])
    np.testing.assert_allclose(gradient, numerical_g, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hessian, numerical_h, rtol=1e-6, atol=1e-7)
    c0, _ = solve_offset(z, y, w, np.zeros(12))
    c1, _ = models.solve(z, y, w, 'residual', 4., alpha=.01)
    np.testing.assert_array_equal(c0, c1)
    final, info = solve_offset(z, y, w, offset)
    assert info['gradient_max'] < 1e-7
    # Finite nonzero offsets must retain true final-MA tail membership.
    low = np.array([-1.5, 1.5]); inward = np.array([-1., 1.])
    np.testing.assert_allclose(loss_terms(inward, low, np.ones(2,bool), 'residual', 4.)[0], [1., 1.])
    return dict(offset_gradient_hessian_finite_difference_pass=True, zero_prior_exact_original_solver=True,
                nonzero_prior_stationarity=True, loss_applies_to_final_ma=True)


def describe(y, p, accepted):
    row = previous._describe(y, p, accepted, variant='oracle08_4')
    row.pop('teacher_point_criteria_pass')
    for key, mask in [('all_report', np.ones(len(y),bool)), ('post_gate', accepted)]:
        m = row[key]
        m['r2'] = float(r2_score(y[mask], p[mask]))
        m['rmse'] = float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
        m['macro_f1'] = float(f1_score(states(y[mask]), states(p[mask]), labels=[0,1,2], average='macro', zero_division=0))
        cm = confusion_matrix(states(y[mask]), states(p[mask]), labels=[0,1,2])
        assert m['confusion'] == cm.tolist()
        assert m['false_negative_count'] == int(cm[0,1]+cm[2,1])
        assert m['false_positive_count'] == int(cm[1,0]+cm[1,2])
        m['balanced_accuracy'] = float(np.mean(np.diag(cm)/cm.sum(axis=1)))
        m['class_errors'] = {}
        for i,name in enumerate(('low','normal','high')):
            r = (states(y)==i)&mask; err = p[r]-y[r]
            m['class_errors'][name] = dict(n=int(r.sum()), mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(np.mean(err**2))), bias=float(np.mean(err)),
                abs_error_median=float(np.median(np.abs(err))), abs_error_p90=float(np.quantile(np.abs(err),.9)))
    m = row['all_report']
    row['relaxed_numeric_checks_pass'] = bool(m['agreement']>.8 and m['fnr']<m['fpr'] and row['post_gate']['coverage']>.75)
    row['independent_predictive_goal_achieved'] = False
    return row


def report(result):
    names = {'target_only':'Target-only', 'single_HZ':'单源：湖州→本地', 'dual_HZ_HB':'双源：湖州＋湖北→本地'}
    lines = ['# B-only＋inward4：单源、双源与8.33%基准的匹配诊断', '',
        '本轮固定0.85混合B、单B输入、RBF、正则系数0.01、类别逆频率权重（上限比20）和inward4。原301条、287组、5外层×3内层划分不变；每条记录只记一次外层输出。', '',
        '本地B逐折沿用旧实验的训练侧MA标尺，确保8.33%基准可精确复现；外院按各自训练记录的标尺构建B。此次没有使用后来静态全院/全三院标尺，未改写工作数据集。', '',
        '单源先在湖州95条上拟合；双源先在湖州95条＋湖北79条上拟合，两院损失总权重相等。随后用同一当前本地训练折拟合RBF残差修正，并对最终MA施加inward4。没有沿用论文的四参数CI适配，不能把本轮称为原论文流程复现。', '',
        '各记录参考MA仍进入其混合B，故本轮为标签信息参与输入的诊断，不能证明U-TEG独立预测效果。模型系数、类权重、标准化、核基和门阈值只由当前训练侧决定。', '',
        '## 门前：优先比较异常漏判', '',
        '|流程|FNR（漏判/48）|FPR（误报/253）|三分类一致率|低MA正确/11|高MA正确/37|MAE mm|R²|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = result['arms'][arm]['all_report']; cm=m['confusion']
        lines.append(f"|{names[arm]}|{m['fnr']:.2%}（{m['false_negative_count']}/48）|{m['fpr']:.2%}（{m['false_positive_count']}/253）|{m['agreement']:.2%}|{cm[0][0]}|{cm[2][2]}|{m['mae']:.3f}|{m['r2']:.3f}|")
    lines += ['', '## 门后：未报告不算正确', '',
        '|流程|覆盖率|已报告一致率|已报告异常FNR|已报告正常FPR|未报告低/正常/高|',
        '|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        row=result['arms'][arm]; m=row['post_gate']
        lines.append(f"|{names[arm]}|{m['coverage']:.2%}|{m['agreement']:.2%}|{m['fnr']:.2%}（{m['false_negative_count']}/{m['low_released']+m['high_released']}）|{m['fpr']:.2%}（{m['false_positive_count']}/{m['within_released']}）|{row['withheld_by_true_state']}|")
    lines += ['', '门使用各外层训练集内3折OOF预测的95%边界距离分位点，三个方案规则一致，但各自阈值由各自训练预测计算。门后FNR的分母仅为已报告异常，应与门前48条完整分母分开看。', '', '## 配对漏判变化', '']
    for arm in ARMS[1:]:
        m=result['paired_vs_target'][arm]
        lines.append(f"- {names[arm]}：相对Target-only，修复{m['recovered_false_negatives']}条原漏判，新增{m['new_false_negatives']}条漏判；正常误报变化{m['false_positive_count_delta']:+d}条。")
    lines += ['', '## 三分类混淆矩阵', '', '行是真实低/正常/高，列是预测低/正常/高。']
    for arm in ARMS:
        lines += ['', names[arm], '', '```text']
        lines += [str(r) for r in result['arms'][arm]['all_report']['confusion']]
        lines += ['```']
    lines += ['', '## 分类别MA误差', '', '|流程|真实类别|MAE mm|RMSE mm|偏差 mm|', '|---|---|---:|---:|---:|']
    for arm in ARMS:
        for cls,m in result['arms'][arm]['all_report']['class_errors'].items():
            lines.append(f"|{names[arm]}|{cls}|{m['mae']:.3f}|{m['rmse']:.3f}|{m['bias']:+.3f}|")
    lines += ['', '所有固定方案均报告，不按外层FNR挑选新参数。FNR<5%仍单列观察，但不再作为用户已放宽的硬性通过条件。原数据、历史结果、论文和GitHub均未修改。',
              '逐记录输出与漏判身份索引仅保存在本地private目录；完整指标、源模型未适配诊断和逐折分布见RESULTS.json。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run():
    assert not OUT.exists(), 'Refuse to overwrite an existing experiment'
    checks = synthetic_tests()
    data, local, peak, folds, sources, source_feature, ranges, hashes = load()
    code = [Path(__file__), Path(old.__file__), Path(previous.__file__), Path(models.__file__),
            Path(inspect.getsourcefile(models.solve)), Path(inspect.getsourcefile(loss_terms))]
    hashes.update({str(p):sha(p) for p in code})
    OUT.mkdir(parents=True); (OUT/'private').mkdir()
    protocol = dict(status='FROZEN_BEFORE_FITTING_AND_SCORING', arms=list(ARMS), loss='inward4', alpha=ALPHA,
        local_normalization='Exact previous .85 mixture using current local training fold min/max; query own reference MA remains in B.',
        source_normalization=ranges, source_counts={c:len(r) for c,r in sources.items()},
        transfer='Source-only B RBF inward4 prior; local B RBF additive correction, same loss on final MA. No blending coefficient or hyperparameter search.',
        source_weight='Within each source hospital class_inverse cap20; equal total hospital weight, then mean weight1.',
        local_weight='Original class_inverse cap20 mean weight1; alpha=.01, free intercept, same final nonnegative MA clip.',
        evaluation='Frozen301 grouped5outer/3inner; all48 abnormal +253 normal; source data excluded from target evaluation.',
        gate='Fixed original95th percentile boundary-distance from inner OOF, no test-label threshold selection.',
        priority='All-report FNR and count first; FPR, correct low/high direction,3class agreement and continuous MA all reported.',
        oracle_input_present=True, valid_UTEG_only_validation=False, independent_predictive_goal_achieved=False,
        manuscript_changed=False, data_modified=False, input_hashes=hashes)
    write(OUT/'FROZEN_PROTOCOL.json', protocol)
    source_models={}; source_audits={}
    for arm, rows in [('single_HZ',sources['Huzhou']), ('dual_HZ_HB',np.r_[sources['Huzhou'],sources['Hubei']])]:
        model, au = fit_source(data, source_feature, rows)
        source_models[arm]=model; source_audits[arm]=au
        np.savez_compressed(OUT/f'private/SOURCE_MODEL_{arm}.npz', **model, rows=rows)
    write(OUT/'private/SOURCE_FIT_AUDIT.json', source_audits)
    first = evaluate_fold(data, peak, folds[0], source_models)
    with np.load(old.OUT/'private/FOLD_0.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(first[0][0], saved['prediction'][2])
        np.testing.assert_array_equal(first[1][0], saved['accepted'][2])
        np.testing.assert_array_equal(first[2][0], saved['inner'][2])
    # Query labels are unavailable to fitting. Freeze the oracle feature, then
    # separately verify a query-feature change cannot change fitted parameters.
    a=np.array(folds[0]['A']); b=np.array(folds[0]['B'])
    x, meta = old.feature(peak, data['y'][:,2], a, b, .85)
    poisoned=dict(data, y=data['y'].copy()); poisoned['y'][b]=1e6
    xchanged, meta2=old.feature(peak, poisoned['y'][:,2], a, b, .85)
    assert meta == meta2
    np.testing.assert_array_equal(x[a], xchanged[a])
    for arm in ARMS:
        model=None if arm=='target_only' else source_models[arm]
        p, au=local_predict(data,x,a,b,model)
        p2, au2=local_predict(poisoned,x,a,b,model)
        np.testing.assert_array_equal(p,p2); assert au == au2
        p3, au3=local_predict(poisoned,xchanged,a,b,model)
        assert au == au3 and not np.array_equal(p,p3)
    checks.update(first_fold_baseline_exact=True, fit_query_label_poison_invariant_with_fixed_input=True,
        oracle_query_label_dependence_confirmed=True, training_parameters_independent_of_query_features=True,
        frozen_group_isolation=True)
    write(OUT/'SMOKE_VERIFIED.json', checks)
    print('SMOKE_PASSED', json.dumps(checks), flush=True)
    positions={int(r):i for i,r in enumerate(local)}
    pred=np.full((3,len(local)),np.nan); acc=np.zeros((3,len(local)),bool)
    prior=np.full((2,len(local)),np.nan); seen=np.zeros(len(local),int); fold_ids=np.full(len(local),-1,int)
    audits=[]; started=time.time()
    for i, fold in enumerate(folds):
        p, accepted, inner, raw, au=first if i==0 else evaluate_fold(data,peak,fold,source_models)
        ix=[positions[int(r)] for r in fold['B']]
        pred[:,ix]=p; acc[:,ix]=accepted; prior[:,ix]=raw; seen[ix]+=1; fold_ids[ix]=i
        audits.append(au)
        np.savez_compressed(OUT/f'private/FOLD_{i}.npz', prediction=p, accepted=accepted, inner=inner,
            unadapted_source=raw, training=fold['A'], query=fold['B'])
        print('FOLD_COMPLETE',i+1,'seconds',round(time.time()-started,2),flush=True)
    assert np.all(seen==1) and np.isfinite(pred).all()
    with np.load(old.OUT/'private/PREDICTIONS.npz',allow_pickle=False) as baseline:
        np.testing.assert_array_equal(local,baseline['local'])
        np.testing.assert_array_equal(pred[0],baseline['predictions'][2])
        np.testing.assert_array_equal(acc[0],baseline['accepted'][2])
    path=OUT/'private/PREDICTIONS.npz'
    np.savez_compressed(path,predictions=pred,accepted=acc,source_prior=prior,local=local,
        groups=data['groups'][local],true_ma=data['y'][local,2],fold=fold_ids)
    write(OUT/'private/FIT_AUDIT.json',audits)
    write(OUT/'PREDICTION_SEAL.json',dict(predictions_sha256=sha(path),protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'),scoring_started=False))
    with np.load(path,allow_pickle=False) as saved:
        y=saved['true_ma']; pred=saved['predictions']; acc=saved['accepted']; t=states(y)
        arms={arm:describe(y,pred[j],acc[j]) for j,arm in enumerate(ARMS)}
        paired={}; private_changes={}
        baseline_fn=(t!=1)&(states(pred[0])==1)
        for j,arm in enumerate(ARMS[1:],1):
            fn=(t!=1)&(states(pred[j])==1)
            recovered=baseline_fn&~fn; new=~baseline_fn&fn
            paired[arm]=dict(recovered_false_negatives=int(recovered.sum()),new_false_negatives=int(new.sum()),
                false_negative_count_delta=int(fn.sum()-baseline_fn.sum()),
                false_positive_count_delta=arms[arm]['all_report']['false_positive_count']-arms['target_only']['all_report']['false_positive_count'])
            private_changes[arm]=dict(recovered_global_rows=local[recovered].tolist(),new_missed_global_rows=local[new].tolist())
        fold_metrics={str(i):{arm:describe(y[fold_ids==i],pred[j,fold_ids==i],acc[j,fold_ids==i]) for j,arm in enumerate(ARMS)} for i in range(5)}
        result=dict(arms=arms,paired_vs_target=paired,fold_metrics=fold_metrics,
            unadapted_source={arm:describe(y,prior[j],np.ones(len(y),bool)) for j,arm in enumerate(ARMS[1:])},
            class_counts=np.bincount(t,minlength=3).tolist(),source_audit=source_audits,
            oracle_input_present=True,valid_UTEG_only_validation=False,independent_predictive_goal_achieved=False)
    write(OUT/'private/PAIRED_FALSE_NEGATIVE_ROWS.json',private_changes)
    write(OUT/'RESULTS.json',result); report(result)
    assert all(sha(p)==h for p,h in hashes.items())
    checks.update(all301_each_once=True,baseline_all_predictions_and_gate_exact=True,
        independently_recomputed_confusion_and_r2=True,input_and_old_results_unchanged=True,
        source_and_local_optimizers_stationary=True,predictions_sealed_before_scoring=True)
    write(OUT/'VERIFICATION.json',checks)
    print('COMPLETE',str(OUT),flush=True)
    print(json.dumps({arm:{k:row['all_report'][k] for k in ('fnr','fpr','false_negative_count','agreement','low_recall','high_recall','mae','r2')} for arm,row in arms.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
