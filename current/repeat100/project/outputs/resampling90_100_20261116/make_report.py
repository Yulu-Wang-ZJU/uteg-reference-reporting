"""Summarize completed finite experiments; no manuscript modification."""
from pathlib import Path
import json
P=Path(__file__).parent
def read(name):return json.loads((P/name).read_text(encoding='utf8'))
def fmt(q,key,pc=True):
    x=q.get(key)
    if not x or x['mean'] is None:return 'NA'
    s=100 if pc else 1
    return f"{x['mean']*s:.2f} ± {x['sd']*s:.2f}" if x['sd'] is not None else f"{x['mean']*s:.2f}"
names={'HZ_to_SA':'单源','HZ+HB_to_SA':'双源','target_only_SA':'Target-only'}
fixed=read('fn_control/FIXED_SUMMARY.json');full=read('full60_control/FIXED_SUMMARY.json')
text=['# 90%抽样100次：Table 1/2与FN控制结果','',
'原论文、PDF、图表、GitHub未修改。原始输入哈希不变。100次名单中的前20次与此前完全一致；原Table 1/2流程补跑80次，另做六种映射候选和完整60条开发记录对照。',
'', '## 核心判断','',
'本次有限调整尚未使两个流程在全部100次重复中同时达到FNR=0、accuracy>80%、reporting rate>75%。加强惩罚没有稳定解决问题。保留原60条开发记录后，双源100次均没有FN，单源在第59次（程序rep=58）出现2条低MA→normal漏判。这个对照说明本地开发数据数量/组成是重要敏感点，但它不包含本地开发样本抽样的不确定性，不能替代54条方案。',
'', '## 1. HGB的主要结果','',
'百分比指标为100次中有定义轮次的均值±样本SD。FNR=异常被放行为normal的数量/实际放行的真实异常数量；未放行异常时为NA，不算零FNR。三分类accuracy的分母是实际放行数，reporting rate的分母是每次181条测试记录。',
'', '| 本地开发记录 | 流程 | Accuracy (%) | Reporting rate (%) | FNR (%) | FPR (%) | FNR=0轮次 | FNR无定义轮次 | 三项同时达标轮次 |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for label,rr in [('抽90%，54条',fixed),('完整保留60条',full)]:
    for q in rr:
        if q['arm']=='target_only_SA' or q['rule']!='HGB' or q.get('kind','residual')!='residual' or q.get('cost',4)!=4:continue
        text.append('| '+' | '.join([label,names[q['arm']],*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']],str(q['zero_FN_rounds'])+'/100',str(q['FNR_undefined_rounds']),str(q['joint_test_pass'])+'/100'])+' |')
text+=['', '完整60条对照中，双源all-report、Boundary和Logistic的FNR均为100/100次可定义且为0；HGB为98次可定义且为0，另2次没有异常记录放行。不能把这两次补成0。双源三项同时达标轮次分别为100、89、95、88次；这些是不同工作点，不代表HGB全面最优。',
'', '## 2. 有限映射调整：同一54条开发协议','',
'所有候选均采用相同来源/开发/测试名单、类别权重、RBF和固定正则参数0.01。Residual提高异常向normal方向误差的惩罚；Boundary在连续误差之外加入分类边界平方铰链惩罚。下面保留原Table 2阈值选择协议，用于固定候选对照。不能据此事后挑选测试成绩最好的一行。',
'', '| 流程 | 损失 | 惩罚倍数 | HGB accuracy (%) | HGB reporting rate (%) | HGB FNR (%) | 零FNR轮次 | 三项同时达标轮次 |','|---|---|---:|---:|---:|---:|---:|---:|']
for q in fixed:
    if q['arm']=='target_only_SA' or q['rule']!='HGB':continue
    text.append('| '+' | '.join([names[q['arm']],q['kind'],str(q['cost']),*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR']],str(q['zero_FN_rounds'])+'/100',str(q['joint_test_pass'])+'/100'])+' |')
text+=['', '## 3. 开发侧零FN联合选型','',
'每条规则分别在六种映射及0/1/2 mm normal边界暂缓区间中选择。仅用嵌套开发预测，要求FN=0、accuracy>80%、有异常被放行，再最大化reporting rate；测试标签不参与选择。无可行设置记为无解。',
'', '| 流程 | 规则 | 开发侧无可行设置 | 测试零FNR轮次 | 测试FNR无定义轮次 | 测试三项同时达标轮次 |','|---|---|---:|---:|---:|---:|']
for q in read('fn_control/SELECTED_SUMMARY.json'):
    if q['arm']=='target_only_SA':continue
    text.append('| '+' | '.join([names[q['arm']],q['rule'],str(q['development_infeasible_rounds']),str(q['zero_FN_rounds']),str(q['FNR_undefined_rounds']),str(q['joint_test_pass'])+'/100'])+' |')
text+=['', '开发侧零FN并不自动产生测试侧零FN。该方案多数时候过于保守或无法选择，不能作为已达标方案替换原Table 2。',
'', '## 4. 已完成与文件','',
'- 原协议Table 1：100次，2400行逐轮映射/校正汇总；Table 2：100次，1200行逐轮结果。见 `Table1_rounds.csv`、`Table2_rounds.csv`、`RESULTS.md`。',
'- 六候选×三流程×100次：1800套MA映射预测；四种规则共7200行固定候选结果。见 `fn_control/FIXED_ROUNDS.json`、`FIXED_SUMMARY.json`。',
'- 开发侧联合选择：三流程×三规则×100次，900条选择记录。见 `fn_control/SELECTED_ROUNDS.json`、`SELECTED_SUMMARY.json`。',
'- 完整60条开发记录：100次，另存 `full60_control/`；不与54条结果合并。',
'- 核查：原4倍惩罚在全部100次均复现基线；选择只从开发侧结果重算；各轮数据角色和分组隔离、分母、输入哈希均通过。见 `FINAL_QA.json`、`fn_control/SMOKE.json`。',
'', '## 5. 建议后续顺序','',
'先决定全文统计补充要回答哪个问题：若要覆盖本地开发样本变化，保留54条方案及其FN波动；若要保持论文原60条开发预算，则将完整60条设为单独的来源/测试抽样敏感性分析。不能因为后者更好就把两者说成同一个方案。方法调整与统计波动补充仍需分开。随后按 `FULL_EXPERIMENT_INVENTORY.md` 完成九配置、时间比较、预算与表示对照、301条内部评价等。',
'', '100次重叠子样本的均值±SD描述当前协议下的波动，不是95%置信区间，也不等于新增100个独立验证队列。']
(P/'REPORT_100_CN.md').write_text('\n'.join(text)+'\n',encoding='utf8')

inventory='''# 全文待补实验清单（100次版本）

当前仅完成两个重点迁移配置及本地target-only的15分钟Table 1/2首批实验。原稿不动。各结果先统一统计协议，再同步图表；不把原均值与新SD拼在一起。

| 优先级 | 实验组 | 对应位置 | 需要做什么 |
|---|---|---|---|
| 1 | 九配置映射与报告 | Table 1/2、Fig. 2/3、S2–S6、S9a/c、S14 | 扩展其余七种迁移配置；同一抽样下比较方法；保存MAE、R²、三分类accuracy、reporting rate、FN/FP和类别分母 |
| 1 | 5/10/15分钟与历史信息 | Fig. 1小图、Fig. 4、S7、S9b、S18/S19 | 分开能力比较与锁定首次报告；沿用旧预测/更新选择的既定规则；不把15分钟零FN要求机械套到早期 |
| 2 | 20/40/60本地预算 | S8/S17 | 保留标签预算定义，重新做target-only Ridge与迁移配对比较；不能从54条池中取60条 |
| 2 | 线性/RBF与三/四参数消融 | Data 2/8、S13 | 训练名单、权重、损失与选型条件匹配；不能只给新方法加范围 |
| 2 | 301条内部评价 | S20、S7c相关部分、Data 8 | 单独确定重复分组协议，与原60开发/201测试方案分开；不合并统计 |
| 3 | 已保留探索性分支 | S10b–g、Data 5 | 适应强度、侧别保护、重配门等按各自原定义重复；探索性质不变 |
| 3 | 关联、稳定性与配对差异 | S6b、S11及相关区间 | 多数直接使用上述预测重算；边界/末段增长关联不能写成因果机制 |

按实际计算可细分为九组：上述第一行拆为映射和报告两组，第四行拆为线性/RBF和表示消融两组。附录共有36个编号表项：3项固定说明不加SD，3项计时需要重复计时，3项逐轮记录由输出生成，其余27项按所属实验更新。这不是36套独立训练。

若覆盖9种迁移+3种target-only、3个时间点、100次，核心共有3600个“流程×时间×重复”单元；当前15分钟首批完成300个。六种来源模型完整15分钟对照为9×6×100=5400次来源拟合，当前已完成1200次。各单元内部还含开发侧拟合；预算、301和探索分支另计。

样本数、设备参数、伦理、阈值定义、视频和示例图不应人为增加±。性能以本次各轮均值和SD一起更新，并明确不是置信区间。
'''
(P/'FULL_EXPERIMENT_INVENTORY.md').write_text(inventory,encoding='utf8')
print('REPORT WRITTEN')
