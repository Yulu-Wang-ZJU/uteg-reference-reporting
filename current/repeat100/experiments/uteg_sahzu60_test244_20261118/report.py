from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import json,csv,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
WS=Path(str(_repeat_root / 'project' / ''))
summary=json.loads((O/'SUMMARY.json').read_text())
rows=json.loads((O/'ALL_ROUNDS.json').read_text())
with np.load(WS/'outputs/current_ma_20261103/data/DATA15_CURRENT.npz') as z:d={k:z[k].copy() for k in z.files}
role=json.loads((WS/'outputs/current_ma_20261103/data/ROLE_FREEZE.json').read_text())['roles']['SAHZU']
oldtest=set(role['test']);olddev=set(role['pool_full'])
names={'target_only_SA':'Target-only','HZ_to_SA':'湖州→浙二','HB_to_SA':'湖北→浙二','HZ+HB_to_SA':'双源→浙二'}
def cell(v,k,pct=False):
    if v[k+'_mean'] is None:return '未定义'
    scale=100 if pct else 1
    return f"{v[k+'_mean']*scale:.2f} ± {v[k+'_sd']*scale:.2f}"+('%' if pct else '')
lines=['# 浙二60／244固定池：100次重复评价','',
    '仅调整浙二目标侧；三条开发记录交换经用户批准。保留60／244总数和各类别数量。同组不跨开发与测试。交换由类别和记录编号哈希确定，不使用预测效果。',
    '开发池低／正常／高为4／48／8，测试池为8／207／29。每轮测试220条（7／187／26），54条开发抽样为4／43／7；60条设置使用完整开发池。',
    '两套设置各100轮，共用相同测试抽样和原先外院抽样：湖州86／95、湖北71／79。15分钟模型、损失、超参数及common_B复合门算法不变；每轮重新拟合，门阈值仅在开发侧确定。',
    '表中为100轮均值±样本标准差，不是95%置信区间。Accuracy为放行记录中的三分类正确率；FNR分母是放行的真实异常记录，FPR分母是放行的真实正常记录；报告率分母为每轮全部220条。',
    '所有轮次保留；原论文、附录、图表和既有结果不动。','']
for budget in [60,54]:
    lines += [f'## {budget}条开发＋220条测试：复合门后','',
        '| 配置 | Accuracy | 报告率 | FNR | FPR | 出现FN的轮次 | FN累计事件 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in names:
        v=next(v for v in summary if v['budget']==budget and v['arm']==arm and v['rule']=='common_B')
        lines += [f"| {names[arm]} | {cell(v,'accuracy',True)} | {cell(v,'reporting_rate',True)} | {cell(v,'FNR',True)} | {cell(v,'FPR',True)} | {v['FN_rounds']}/100 | {v['total_FN']} |"]
    lines += ['','### 门前映射（全部测试记录）','',
        '| 配置 | MAE (mm) | R² | 低MA召回 | 高MA召回 | 低MA MAE | 高MA MAE |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in names:
        v=next(v for v in summary if v['budget']==budget and v['arm']==arm and v['rule']=='All report')
        lines += [f"| {names[arm]} | {cell(v,'MAE')} | {cell(v,'R2')} | {cell(v,'low_recall',True)} | {cell(v,'high_recall',True)} | {cell(v,'low_MAE')} | {cell(v,'high_MAE')} |"]
    lines+=['']
# Count every repeated FN event and its unique record; diagnostic only.
fn_events=[];countchecks=0
for budget in [54,60]:
    for arm in names:
        for rep in range(100):
            with np.load(O/'private'/f'dev{budget}'/f'round_{rep:03d}'/(arm+'.npz')) as z:
                ix=z['test'];y=z['reference'];p=z['prediction'];a=z['action_common_B'];truth=np.where(y<50,0,np.where(y>70,2,1));pred=np.where(p<50,0,np.where(p>70,2,1))
                fn=a&(truth!=1)&(pred==1)
                rr=next(v for v in rows if v['budget']==budget and v['arm']==arm and v['rep']==rep and v['rule']=='common_B')
                assert int(fn.sum())==rr['test']['FN'];assert int(a.sum())==rr['test']['reported'];countchecks+=1
                for j in np.flatnonzero(fn):
                    i=int(ix[j]);fn_events.append(dict(budget=budget,arm=arm,rep=rep,index=i,record=str(d['ids'][i]),MA=float(y[j]),prediction=float(p[j]),
                        old_role='old_test201' if i in oldtest else 'old_development60' if i in olddev else 'other_local_records'))
(O/'FN_EVENTS.json').write_text(json.dumps(fn_events,ensure_ascii=False,indent=2),encoding='utf-8')
lines+=['## FN位置核对','', '以下仅用于解释新结果，未据此修改门或选取轮次。FN事件数包含同一记录在多轮中重复出现，不能视为独立病例数。','',
    '| 开发数 | 配置 | 出现FN的不同记录 | 低MA FN事件 | 高MA FN事件 |','|---|---|---:|---:|---:|']
for budget in [60,54]:
    for arm in names:
        a=[v for v in fn_events if v['budget']==budget and v['arm']==arm]
        lines+=[f"| {budget} | {names[arm]} | {len(set(v['index'] for v in a))} | {sum(v['MA']<50 for v in a)} | {sum(v['MA']>70 for v in a)} |"]
lines+=['','## 与正文原方案的关系','',
    '正文主结果使用原60条开发池和201条测试池，每轮测试181条。当前复合门的正文双源结果为84.24% accuracy、93.08%报告率和100轮零FN。',
    '本次既扩大了测试池，又按分组隔离要求交换了开发池的3条成员。因此新旧性能差异不能全部归因于测试例数；60／54两套新结果之间则共用开发池、测试抽样和来源抽样，可比较减少开发记录的影响。',
    '复算原第1轮、两种开发规模和四种配置，预测、风险分数、阈值及放行决定均与原保存结果一致（ORIGINAL_REPLAY_CHECK.json）。',
    '本轮不能仅凭均值较好就替代原文零FN结论；新结果按实际FN逐轮保留。','',
    '## 文件','', 'SUMMARY.csv：所有规则及两套设置的均值、SD和有效轮数。ALL_ROUNDS.json：全部逐轮指标。private：预测、开发侧规则及完整分组划分。FN_EVENTS.json：逐条FN诊断。','']
(O/'RESULTS_CN.md').write_text('\n'.join(lines),encoding='utf-8')
checks=json.loads((O/'FINAL_CHECKS.json').read_text());checks.update(prediction_count_checks=countchecks,original_replay_passed=json.loads((O/'ORIGINAL_REPLAY_CHECK.json').read_text())['passed'])
(O/'FINAL_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('\n'.join(lines[:32]))
