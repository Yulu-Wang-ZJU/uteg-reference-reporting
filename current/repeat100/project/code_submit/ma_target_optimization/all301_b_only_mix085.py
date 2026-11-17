"""Paired 0.85 vs frozen 0.80 B-only oracle diagnostic, never predictive validation."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import all301_b_only_diagnostic as baseline

previous = baseline.previous
OUT = baseline.OUT.parent / 'b_only_weight_0p85'
WEIGHT = .85


def feature(peak, y, train, query, weight):
    lo, hi = float(np.min(y[train])), float(np.max(y[train]))
    assert hi > lo
    complement = {.8: .2, .85: .15}[weight]
    x = np.full((len(y), 1), np.nan)
    x[train, 0] = weight * (2.0 + (y[train] - lo) / (hi - lo)) + complement * peak[train]
    x[query, 0] = weight * (y[query] - lo) / (hi - lo) + 2.0 * weight + complement * peak[query]
    return x, dict(train_ma_min=lo, train_ma_max=hi, mixture_weight=weight,
                   raw_peak_weight=complement, oracle_input_present=True,
                   valid_UTEG_only_validation=False)


def evaluate(data, peak, fold, weight):
    previous._validate_fold(data, fold)
    a, b = np.asarray(fold['A']), np.asarray(fold['B'])
    position = {int(v): i for i, v in enumerate(a)}
    inner = np.full((4, len(a)), np.nan)
    prediction = np.full((4, len(b)), np.nan)
    audits = []
    stages = [('inner'+str(i), np.asarray(f['T']), np.asarray(f['V']))
              for i, f in enumerate(fold['inner'])] + [('outer', a, b)]
    for stage, train, query in stages:
        x, meta = feature(peak, data['y'][:, 2], train, query, weight)
        for j, loss in enumerate(previous.LOSSES):
            p, au = previous._fit_predict(data, x, train, query, loss)
            if stage == 'outer':
                prediction[j] = p
            else:
                inner[j, [position[int(v)] for v in query]] = p
            audits.append(dict(stage=stage, **meta, **au))
    assert np.isfinite(inner).all() and np.isfinite(prediction).all()
    accepted, thresholds = previous.models.boundary_gate(inner, prediction)
    return prediction, accepted, inner, dict(fits=audits, thresholds=thresholds.tolist())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'private').mkdir(exist_ok=True)
    data, local, base4, peak, hashes, _ = previous._load()
    splits_path = previous.contract.OUT/'private/FROZEN_SPLITS.json'
    folds = json.loads(splits_path.read_text(encoding='utf-8'))
    old_results = baseline.OUT/'RESULTS.json'
    old_predictions = baseline.OUT/'private/PREDICTIONS.npz'
    files = [Path(__file__), Path(baseline.__file__), Path(previous.__file__),
             Path(previous.models.__file__), splits_path, old_results, old_predictions,
             baseline.OUT/'FROZEN_PROTOCOL.json', baseline.OUT/'private/FOLD_0.npz',
             previous.OUT/'RESULTS.json', previous.OUT/'private/PREDICTIONS.npz']
    frozen_hashes = dict(hashes, **{str(p): baseline.sha(p) for p in files})
    protocol = dict(status='FROZEN_BEFORE_SCORING', mixture_weight=WEIGHT,
        candidates=[dict(variant='oracle085_B_only', loss=l) for l in previous.LOSSES],
        formula='B=.85*(2+(MA-min(train_MA))/(max(train_MA)-min(train_MA)))+.15*raw_peak; no clipping.',
        oracle='Each query true MA deliberately enters its B input. This is not U-TEG-only validation or deployable prediction.',
        comparison='Only mixture weight changes from .80 to .85; frozen .80 results reused without selection.',
        model='One B coordinate; same train-only standardization, RBF, alpha=.01, four fixed losses and nonnegative MA clipping.',
        splits='Existing301 local records; frozen grouped5outer/3inner; no source records or augmentation.',
        gate='Same inner OOF95th percentile boundary-distance rule; no outer-label threshold selection.',
        exploratory='Coefficient changed after inspecting earlier internal results; not new independent validation.',
        hashes=frozen_hashes, original_data_changed=False, manuscript_changed=False)
    protocol_path = OUT/'FROZEN_PROTOCOL.json'
    assert not protocol_path.exists(), 'Refuse to overwrite a frozen run'
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')

    # Replay the preceding first fold at .80 to check that only the coefficient changes.
    replay = evaluate(data, peak, folds[0], .8)
    with np.load(baseline.OUT/'private/FOLD_0.npz', allow_pickle=False) as old:
        np.testing.assert_array_equal(replay[0], old['prediction'][8:12])
        np.testing.assert_array_equal(replay[1], old['accepted'][8:12])
        np.testing.assert_array_equal(replay[2], old['inner'][8:12])
    first = evaluate(data, peak, folds[0], WEIGHT)
    a, b = np.asarray(folds[0]['A']), np.asarray(folds[0]['B'])
    poisoned_y = data['y'].copy(); poisoned_y[b] = 1e6
    poisoned = dict(data, y=poisoned_y)
    x1, meta1 = feature(peak, data['y'][:, 2], a, b, WEIGHT)
    x2, meta2 = feature(peak, poisoned_y[:, 2], a, b, WEIGHT)
    assert meta1 == meta2
    np.testing.assert_array_equal(x1[a], x2[a])
    p1, au1 = previous._fit_predict(data, x1, a, b, 'uniform')
    p2, au2 = previous._fit_predict(poisoned, x2, a, b, 'uniform')
    assert au1 == au2 and not np.array_equal(p1, p2)
    checks = dict(baseline_080_first_fold_exact_replay=True,
        smoke_four_losses_finite=True, group_isolation_checked=True,
        training_parameters_independent_of_query_labels=True,
        oracle_query_label_dependence_explicit=True, input_dimension=1)
    (OUT/'SMOKE_VERIFIED.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('SMOKE_PASSED', flush=True)
    positions = {int(v): i for i, v in enumerate(local)}
    prediction = np.full((4, len(local)), np.nan)
    accepted = np.zeros_like(prediction, dtype=bool)
    seen = np.zeros(len(local), int)
    audits = []
    for fi, fold in enumerate(folds):
        p, ac, inner, au = first if fi == 0 else evaluate(data, peak, fold, WEIGHT)
        ix = [positions[int(v)] for v in fold['B']]
        prediction[:, ix] = p; accepted[:, ix] = ac; seen[ix] += 1
        audits.append(au)
        np.savez_compressed(OUT/f'private/FOLD_{fi}.npz', query=fold['B'], training=fold['A'],
                            prediction=p, accepted=ac, inner=inner)
        print('FOLD_COMPLETE', fi+1, flush=True)
    assert np.all(seen == 1) and np.isfinite(prediction).all()
    saved = OUT/'private/PREDICTIONS.npz'
    np.savez_compressed(saved, predictions=prediction, accepted=accepted, local=local,
                        groups=data['groups'][local], true_ma=data['y'][local, 2])
    (OUT/'private/FOLD_AUDIT.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    (OUT/'PREDICTION_SEAL.json').write_text(json.dumps(dict(prediction_sha256=baseline.sha(saved),
        protocol_sha256=baseline.sha(protocol_path), scoring_started=False), indent=2), encoding='utf-8')
    results = []
    with np.load(saved, allow_pickle=False) as sealed:
        y = sealed['true_ma']
        for j, loss in enumerate(previous.LOSSES):
            p, ac = sealed['predictions'][j], sealed['accepted'][j]
            # The shared descriptor's oracle tag denotes true-MA input, not a fitted coefficient.
            m = previous._describe(y, p, ac, variant='oracle08_4')
            m['all_report']['rmse'] = float(np.sqrt(np.mean((p-y)**2)))
            results.append(dict(candidate=dict(variant='oracle085_B_only', loss=loss),
                mixture_weight=WEIGHT, role='oracle_label_injected_diagnostic',
                valid_predictive_goal_pass=False, **m))
    old = json.loads(old_results.read_text(encoding='utf-8'))['candidates']
    report = ['# 单B混合系数0.85与0.80配对诊断', '',
        '两组都将每条记录真实参考MA混入输入，因此仅为oracle诊断，不能代表U-TEG独立预测。仅改变混合系数，其他设置不变。', '',
        '|损失|系数|三分类一致率|FNR|FPR|低正确/11|高正确/37|MAE mm|R²|门后覆盖|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for loss in previous.LOSSES:
        pairs = [(.8, next(r for r in old if r['candidate'] == dict(variant='oracle08_B_only', loss=loss))),
                 (.85, next(r for r in results if r['candidate']['loss'] == loss))]
        for w, row in pairs:
            m = row['all_report']; cm = m['confusion']
            report.append(f"|{loss}|{w:.2f}|{m['agreement']:.2%}|{m['fnr']:.2%}|{m['fpr']:.2%}|{cm[0][0]}|{cm[2][2]}|{m['mae']:.3f}|{m['r2']:.3f}|{row['post_gate']['coverage']:.2%}|")
    report += ['', '## 0.85门后结果', '',
        '|损失|覆盖率|已报告一致率|FNR 分子/分母|FPR 分子/分母|未报告低/正常/高|',
        '|---|---:|---:|---|---|---|']
    for row in results:
        m = row['post_gate']
        report.append(f"|{row['candidate']['loss']}|{m['coverage']:.2%}|{m['agreement']:.2%}|{m['fnr_numerator']}/{m['fnr_denominator_released_abnormal']}={m['fnr']:.2%}|{m['fpr_numerator']}/{m['fpr_denominator_released_within']}={m['fpr']:.2%}|{row['withheld_by_true_state']}|")
    report += ['', '门前FNR以全部48条异常为分母，FPR以253条正常为分母；门后对应分母仅为已报告的真实异常/正常，未报告不算正确。',
        'MA类别界值仍为50和70 mm；每条记录一次外层输出。0.80首折四种损失的内外层预测及门结果精确重现。',
        '提高混合系数就是增加真实MA信息、减少原始幅值的贡献。指标提高不能解释为仪器或可部署映射得到改善。原数据、旧结果和论文未修改。']
    (OUT/'RESULTS.json').write_text(json.dumps(dict(candidates=results, oracle_input_present=True), ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'REPORT_ZH.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    assert all(baseline.sha(p) == h for p, h in frozen_hashes.items())
    checks.update(all301_each_once=True, all_input_prior_code_and_result_hashes_unchanged=True,
                  scored_from_sealed_predictions=True)
    (OUT/'VERIFICATION.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
