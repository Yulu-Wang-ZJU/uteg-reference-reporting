"""One-coordinate B ablation; oracle MA-injected inputs remain diagnostics."""
from pathlib import Path
import json
import hashlib

import numpy as np
from threadpoolctl import threadpool_limits

from . import all301_oracle_amplitude_substitution as previous

OUT = previous.OUT.parent / 'b_only_diagnostic'
VARIANTS = [('fitted_B_only', 'original4'), ('raw_peak_only', 'rawpeak4'), ('oracle08_B_only', 'oracle08_4')]
GRID = [dict(variant=name, loss=loss) for name, _ in VARIANTS for loss in previous.LOSSES]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(data, base4, peak, fold):
    previous._validate_fold(data, fold)
    a, b = np.asarray(fold['A']), np.asarray(fold['B'])
    pos = {int(row): j for j, row in enumerate(a)}
    y = data['y'][:, 2]
    inner = np.full((12, len(a)), np.nan)
    pred = np.full((12, len(b)), np.nan)
    audit = []
    for stage, train, query in [('inner'+str(i), np.asarray(f['T']), np.asarray(f['V'])) for i, f in enumerate(fold['inner'])] + [('outer', a, b)]:
        for vi, (name, oldname) in enumerate(VARIANTS):
            full = previous._feature(oldname, base4, peak, y, train, query)
            feature = full[:, 1:2].copy()
            assert feature.shape == (len(base4), 1)
            np.testing.assert_array_equal(feature[np.r_[train, query], 0], full[np.r_[train, query], 1])
            for li, loss in enumerate(previous.LOSSES):
                p, au = previous._fit_predict(data, feature, train, query, loss)
                j = vi*4+li
                if stage == 'outer':
                    pred[j] = p
                else:
                    inner[j, [pos[int(r)] for r in query]] = p
                audit.append(dict(stage=stage, variant=name, input_columns=1,
                    **previous._oracle_audit(y, oldname, train), **au))
    assert np.isfinite(inner).all() and np.isfinite(pred).all()
    accepted, thresholds = previous.models.boundary_gate(inner, pred)
    return pred, accepted, inner, dict(fits=audit, thresholds=thresholds.tolist())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'private').mkdir(exist_ok=True)
    data, local, base4, peak, hashes, meta = previous._load()
    splits_path = previous.contract.OUT/'private/FROZEN_SPLITS.json'
    folds = json.loads(splits_path.read_text(encoding='utf-8'))
    old_predictions = previous.OUT/'private/PREDICTIONS.npz'
    old_results = previous.OUT/'RESULTS.json'
    paths = [Path(__file__), Path(previous.__file__), Path(previous.models.__file__), old_predictions, old_results, splits_path]
    allhash = dict(hashes, **{str(p):sha(p) for p in paths})
    protocol = dict(status='FROZEN_BEFORE_NEW_SCORING', candidates=GRID,
        question='Drop t0,k,nu and keep only the exact B coordinate from the preceding four-parameter experiment.',
        model='Same train-only standardized RBF convention and alpha=.01; width changes4to1, so kernel width normalization follows the original1/n_features convention. No hyperparameter tuning.',
        oracle='oracle08_B_only uses true MA for each training and query input. Min/max are fit on current training records, but query MA remains explicitly leaked by design. Not U-TEG-only validation.',
        formula='Bmix=.8*(2+(MA-min(train_MA))/(max(train_MA)-min(train_MA)))+.2*raw_peak. No clipping to2–3.',
        selection='All12fixed results reported, no selection on outer scores. Inner folds provide only the fixed boundary gate thresholds.',
        splits='Existing301 records,287groups,5outer/3inner; same records, no source center, no augmentation.',
        gate='Existing boundary distance, innerOOF95th percentile,higher;50/70category thresholds unchanged.',
        hashes=allhash, manuscript_changed=False, original_data_changed=False)
    prot_path = OUT/'FROZEN_PROTOCOL.json'
    assert not prot_path.exists(), 'Use a new output folder rather than overwrite a scored run'
    prot_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    first = evaluate(data, base4, peak, folds[0])
    a, b = np.asarray(folds[0]['A']), np.asarray(folds[0]['B'])
    poison_y = data['y'].copy(); poison_y[b] = 1e6
    poisoned = dict(data, y=poison_y)
    for name, oldname in VARIANTS:
        f1 = previous._feature(oldname, base4, peak, data['y'][:,2], a, b)[:,1:2]
        f2 = previous._feature(oldname, base4, peak, poison_y[:,2], a, b)[:,1:2]
        p1, au1 = previous._fit_predict(data, f1, a, b, 'uniform')
        p2, au2 = previous._fit_predict(poisoned, f2, a, b, 'uniform')
        assert au1 == au2
        if oldname != 'oracle08_4':
            np.testing.assert_array_equal(p1, p2)
        else:
            assert not np.array_equal(f1[b], f2[b]) and not np.array_equal(p1,p2)
        # Removed coordinates have no route into any one-dimensional design.
        changed = base4.copy(); changed[:,[0,2,3]] = 1e6
        f3 = previous._feature(oldname, changed, peak, data['y'][:,2], a, b)[:,1:2]
        np.testing.assert_array_equal(f1, f3)
    verification = dict(first_outer_all12_finite=True, removed_coordinates_cannot_change_input=True,
        training_parameters_independent_of_query_labels=True, legal_query_label_poison_unchanged=True,
        oracle_query_label_changes_are_expected=True, existing_fold_group_checks=True)
    (OUT/'SMOKE_VERIFIED.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('SMOKE_PASSED',flush=True)
    position = {int(v):j for j,v in enumerate(local)}
    predictions=np.full((12,len(local)),np.nan);accepted=np.zeros_like(predictions,dtype=bool)
    seen=np.zeros(len(local),int);audits=[]
    for fi, fold in enumerate(folds):
        p, ac, inner, au = first if fi==0 else evaluate(data,base4,peak,fold)
        ix=[position[int(v)] for v in fold['B']]
        predictions[:,ix]=p;accepted[:,ix]=ac;seen[ix]+=1;audits.append(au)
        np.savez_compressed(OUT/f'private/FOLD_{fi}.npz',query=fold['B'],training=fold['A'],prediction=p,accepted=ac,inner=inner)
        print('FOLD_COMPLETE',fi+1,flush=True)
    assert np.all(seen==1) and np.isfinite(predictions).all()
    saved=OUT/'private/PREDICTIONS.npz'
    np.savez_compressed(saved,predictions=predictions,accepted=accepted,local=local,groups=data['groups'][local])
    (OUT/'private/FOLD_AUDIT.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    (OUT/'PREDICTION_SEAL.json').write_text(json.dumps(dict(prediction_sha256=sha(saved),protocol_sha256=sha(prot_path),scoring_started=False),indent=2),encoding='utf-8')
    y=data['y'][local,2];results=[]
    for j,c in enumerate(GRID):
        oldname=VARIANTS[j//4][1]
        m=previous._describe(y,predictions[j],accepted[j],variant=oldname)
        m['all_report']['rmse']=float(np.sqrt(np.mean((predictions[j]-y)**2)))
        m['all_report']['prediction_input_dimension']=1
        m['role']='oracle_label_injected_diagnostic' if oldname=='oracle08_4' else 'UTEG_only_exploratory_internal'
        m['valid_predictive_goal_pass']=oldname!='oracle08_4' and m['teacher_point_criteria_pass']
        results.append(dict(candidate=c,**m))
    old=json.loads(old_results.read_text(encoding='utf-8'))['candidates']
    report=['# 只用B的MA映射与诊断', '',
        '只保留B一列，去掉t0/k/nu。其余模型、4种损失、分组与门设置沿用前轮。0.8混合B仍包含每条记录真实MA，只能解释为诊断实验。', '',
        '## 全部报告：301条，每条一次外层输出', '',
        '|幅值|损失|一致率|FNR|FPR|低正确/11|高正确/37|MAE mm|R²|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        c=row['candidate'];m=row['all_report'];cm=m['confusion']
        report.append(f"|{c['variant']}|{c['loss']}|{m['agreement']:.2%}|{m['fnr']:.2%}|{m['fpr']:.2%}|{cm[0][0]}|{cm[2][2]}|{m['mae']:.3f}|{m['r2']:.3f}|")
    report+=['','## 同一混合B：四参数与单B配对','','|损失|输入|一致率|FNR|FPR|低正确|高正确|MAE mm|R²|门后覆盖|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for loss in previous.LOSSES:
        pairs=[('四参数',next(v for v in old if v['candidate']==dict(variant='oracle08_4',loss=loss))),
               ('单B',next(v for v in results if v['candidate']==dict(variant='oracle08_B_only',loss=loss)))]
        for label,row in pairs:
            m=row['all_report'];cm=m['confusion']
            report.append(f"|{loss}|{label}|{m['agreement']:.2%}|{m['fnr']:.2%}|{m['fpr']:.2%}|{cm[0][0]}/11|{cm[2][2]}/37|{m['mae']:.3f}|{m['r2']:.3f}|{row['post_gate']['coverage']:.2%}|")
    report+=['','## 单B门后结果','','|幅值|损失|覆盖|已报告一致率|FNR 分子/分母|FPR 分子/分母|未报告低/正常/高|','|---|---|---:|---:|---|---|---|']
    for row in results:
        c=row['candidate'];m=row['post_gate']
        report.append(f"|{c['variant']}|{c['loss']}|{m['coverage']:.2%}|{m['agreement']:.2%}|{m['fnr_numerator']}/{m['fnr_denominator_released_abnormal']}={m['fnr']:.2%}|{m['fpr_numerator']}/{m['fpr_denominator_released_within']}={m['fpr']:.2%}|{row['withheld_by_true_state']}|")
    report+=['','## 解读边界','','普通B与实测峰值对照仅使用U-TEG，属于反复探索后的内部评价；混合B组直接使用查询真实MA，不是可部署预测。门后FNR/FPR只在已报告的对应真实类别内计算；未报告不计正确。',
        '维度4变1后，按同一RBF实现以特征数归一化距离；本轮没有重新搜索带宽、正则化、损失、分类阈值或门。全部固定配置均报告，没有根据外层结果选一个新赢家。',
        '所有输入、旧四参数预测、论文和公开材料均未修改。']
    (OUT/'RESULTS.json').write_text(json.dumps(dict(candidates=results,protocol_role=protocol['oracle']),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    assert all(sha(p)==h for p,h in allhash.items())
    verification.update(all301_each_once=True,all_input_and_prior_result_hashes_unchanged=True,predictions_sealed_before_scoring=True)
    (OUT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps([dict(candidate=r['candidate'],all_report={k:r['all_report'][k] for k in ['agreement','fnr','fpr','low_recall','high_recall','mae','r2','confusion']},coverage=r['post_gate']['coverage']) for r in results],ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
