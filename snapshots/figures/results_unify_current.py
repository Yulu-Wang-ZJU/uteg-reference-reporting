"""Rebuild the gallery with current-policy evidence only; no fitting or selection."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
old = (HERE / 'build_figures.py').read_text(encoding='utf-8')
start = old.index('# Figure 4: keep the historical MA time replay')
end = old.index('# Full nine-direction context;', start)
# Execute the unchanged endpoint plots, excluding all historical time-data reads.
source = old[:start] + old[end:]
source = source.replace("plan='''Results", "captions.pop('Figure4_observation')\nplan='''Results", 1)
ns = {'__file__': str(HERE / 'build_figures.py'), '__name__': '__main__', '_CURRENT_ONLY': True}
exec(compile(source, str(HERE / 'build_figures.py'), 'exec'), ns)

policy = {
    'schema': 'current_figure_policy_v1',
    'candidate': 'RIDGE_B_PREDMA_DUAL_WITHIN_RISK',
    'observation_minutes_verified': [15],
    'representation': 'Existing generalized-exponential U-TEG four-parameter representation',
    'mapping': 'Frozen Ridge-10, alpha=10, four medical outputs',
    'adaptation': 'Unchanged saved target Huber/adapter chain; no coefficient refit',
    'reporting_within': 'Two class-balanced logistic risk heads, input [B, adapted predicted MA]; max(low risk, high risk)',
    'reporting_tails': 'Original MA-ERROR-LR risk score; low/high share one tail threshold within each direction-seed',
    'thresholds': 'Frozen outer-B development thresholds; distinct normal and common-tail thresholds per direction-seed',
    'B_is_new_signal': False,
    'reference_labels': 'MA <50 low; 50<=MA<=70 within; MA>70 high',
    'seeds': list(range(1000, 1020)),
    'calibration_budget': {'Huzhou':40, 'SAHZU':60, 'Hubei':40},
    'directions': 9,
    'temporal_status': 'MATCHED_5_10_MINUTE_FULLCHAIN_RESULTS_NOT_FOUND',
    'figure4_status': 'RESERVED_NOT_A_RESULT',
    'fit_units': 0,
    'test_selection': False,
}
manifest_paths = [
    'outputs/uteg_ridge_dual_head_external_eval_20261026_191117/EXTERNAL_ACTION_MANIFEST.json',
    'outputs/uteg_ridge_dual_head_unified_selector_20261026_190142/THRESHOLDS_FREEZE.json',
    'outputs/uteg_ridge_dual_head_prepare_20261026_172952/PROTOCOL.md',
]
policy['identity_hashes'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in manifest_paths}
(HERE/'CURRENT_POLICY.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')

header = '''<section><h2>统一的当前主流程</h2><p>广义指数U-TEG四参数 → 固定Ridge-10四医学参数映射 → 原Huber及适配链 → MA可信报告。B已经是四参数中的一列，不称作新增测量信号。</p><p>预测为范围内：用[B，适配后预测MA]输入两个逻辑回归风险头，分别识别潜在低MA、高MA，取两者较大风险。预测为低/高MA：沿用原MA-ERROR-LR风险分数。范围内门槛与异常门槛分开；低/高MA共用异常门槛。</p><p>每方向、每seed的数值门槛均来自冻结开发侧选择，不是九方向共用一个数值。统一20个seed（1000–1019）、九方向、湖州C40／浙二C60／湖北C40；标签50/70不变。</p><p><a href="CURRENT_POLICY.json">当前模型与规则身份</a></p></section>'''
pending = '''<section id="observation"><h2>Figure 4 · 继续观察：位置保留，当前无匹配结果</h2><p class="note">旧版MA时序图已从本次主图展示撤下，没有替换标题冒充新版。当前已核验的主流程证据为15分钟；没有找到同时匹配Ridge-10、原适配、B＋预测MA双风险头及当前放行规则的5、10分钟结果。</p><p>后续该图仍按A–C展示三中心5／10／15分钟累计正确、错误、不予报告，D展示新增正确和错误报告的配对区间。但在匹配验证完成前不画数据，不声称当前反馈回路已获验证。</p><p>匹配验证必须只用各时点真实前缀重新获得表示，并在原合法开发划分内形成相同时点的映射、适配和风险留出预测；不能把15分钟特征、参数、风险模型或门槛直接当作前缀验证。保持模型家族、配置、信号定义和选择规则不变，不依据测试调整。</p></section>'''
page = (HERE/'index.html').read_text(encoding='utf-8')
a = page.index('<h1>'); b = page.index('<section>', a)
header = header.replace('<section><h2>统一的当前主流程</h2>', '<section><details><summary>方法与版本信息（展开）</summary>')
header = header.replace('</section>', '</details></section>')
page = page[:a] + '<h1>Results：医学转译与可信报告</h1><p>Figure 1保留。图中只突出问题与结果；详细方法、统计口径见图注。</p>' + header + page[b:]
page = page.replace('</main>', pending + '</main>')
audit_note = '''<section><details><summary>数值核查与完整源数据</summary><p>Figure 2A湖北R的MAE：映射1.154分钟 → 单独校准1.144分钟 → 最终适配1.216分钟。最终相对映射增加0.062分钟（约3.7秒）；病例配对95%区间为−0.191至0.321分钟，跨零。39例QC合格测试记录，20个seed；12/20个seed误差增加。差异主要出现在后续适配，而非图例或分母画反，当前不能据此断言稳定恶化，也不能称所有读出均改善。未根据测试结果撤销适配。</p><p><a href="source_data/adaptation_R_audit.json">R逐阶段配对核查</a> · <a href="source_data/figure3_class_disposition.csv">完整三状态报告去向</a> · <a href="source_data/nine_direction_summary.csv">九方向结果</a></p><p>Figure 3的C、D按信息重复原则删除，错误数据未删除。全部真实低MA中仍有15/140、19/140事件被放行为范围内，不能仅由较高总体一致率推断异常识别良好。</p></details></section>'''
page = page.replace('</main>', audit_note + '</main>')
(HERE/'index.html').write_text(page, encoding='utf-8')

plan = '''当前主图一致性更新

唯一主流程：广义指数四参数表示 → Ridge-10(alpha=10)四医学参数映射 → 原Huber及适配链 → MA报告。
预测范围内使用[B, adapted predicted MA]双侧逻辑回归风险头；取max风险。预测异常使用原MA-ERROR-LR分数。每direction×seed用原开发侧冻结的normal阈值及low/high共用tail阈值。B不是新输入信号。
九方向、20seed1000–1019、C40/C60/C40与参考MA50/70标签保持一致。

Figure 2：当前映射和适配的四医学参数误差及MA关系。
Figure 3：仅保留A一致率—覆盖率、B低/高MA报范围内；删除与B重复的C/D，完整去向留源数据表。原LR门只是相同预测器上的具名对照，不混作主结果。图面仅标Previous rule/Current rule。
Figure S1：同一主流程的完整九方向结果，不筛去失败方向。
Figure 4：从主图展示中撤下旧MA时序分支，仅保留位置。匹配当前流程的5/10分钟结果未找到；当前15分钟单时点结果不能充当累计时序15分钟终点。

继续观察匹配验证要求：保持模型配置、信号定义、划分、校准预算和20seed；各时点只用真实前缀；拟合、适配、OOF风险分数及冻结边界须与该时点一致；沿用先冻结后测试；首报后不回改，15分钟仍未通过则不报。精确新增fit账本须在执行拟合前核算，本次未新拟合。
旧Figure4导出文件只作历史保留，不在当前页面、图注、来源账本或主证据中引用。

范例逻辑保持：Ghetmiri的预测到医学读出用于Fig2；Bai的收益—代价权衡用于Fig3；Rapp的观察—决策—行动用于未来匹配Fig4。Times New Roman、白底、矢量和源数据保持原设置。
未修改Figure1、main.tex、原实验结果或模型；当前只完成图组身份统一，尚未完成新版时序验证。
'''
(HERE/'FIGURE_PLAN_ZH.txt').write_text(plan, encoding='utf-8')
en = (HERE/'CAPTIONS_EN.txt').read_text(encoding='utf-8')
# Retain the English legacy caption only as an explicitly archived section.
if 'ARCHIVED — not part of the current figure set:' not in en:
    en = en.replace('Figure 4', 'ARCHIVED — not part of the current figure set: Figure 4')
(HERE/'CAPTIONS_EN.txt').write_text(en, encoding='utf-8')
prov = json.loads((HERE/'PROVENANCE.json').read_text(encoding='utf-8'))
assert not any('FIRST_REPORT' in p or 'PAIRED_INTERVALS' in p for p in prov['sources'])
prov.update(time_branch_separate=False, temporal_result_included=False, current_policy='CURRENT_POLICY.json')
prov['sources'].update(policy['identity_hashes'])
(HERE/'PROVENANCE.json').write_text(json.dumps(prov,indent=2), encoding='utf-8')
assert 'Figure4_observation.svg' not in page
assert len(ns['qa']) == 3
assert all(not v['off_canvas_text'] for v in ns['qa'].values())
print('CURRENT-POLICY GALLERY PASS: three matched quantitative figures; temporal slot pending; zero fits.')
import sys
sys.path.insert(0,str(HERE))
from publish_current import publish
publish(ns)
