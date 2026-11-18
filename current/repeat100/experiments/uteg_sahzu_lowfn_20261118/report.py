from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import json,csv,hashlib,collections,sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;P=Path(str(_repeat_root / 'experiments' / 'uteg_sahzu60_test244_20261118'));F=O/'symmetric90_reselected'
diag=json.loads((O/'DIAGNOSIS.json').read_text());s=json.loads((F/'SUMMARY.json').read_text());events=json.loads((F/'FN_EVENTS.json').read_text())
base=json.loads((P/'SUMMARY.json').read_text());rows=json.loads((F/'ROUNDS.json').read_text());policy=json.loads((F/'POLICIES.json').read_text())
arms={'target_only_SA':'Target-only','HZ_to_SA':'湖州→浙二','HB_to_SA':'湖北→浙二','HZ+HB_to_SA':'双源→浙二'}
def fmt(v,k):return f"{100*v[k+'_mean']:.2f} ± {100*v[k+'_sd']:.2f}%"
lines=['# 浙二低MA漏判定位与最小放行修正','',
    '结论：主要原因是低MA开发记录替换引起映射及正常放行边界变化，不是测试例数本身改变模型。补上与既有高侧对称的低MA风险检查，再按原程序重新选择异常分支门后，60条设置在四种配置的全部100轮中均观察到零FN，accuracy和报告率均值均超过80%。54条设置仍有高MA漏判，单独保留。','',
    '## 1. 控制变量定位','',
    '被核查记录为SAHZU_R0006，参考MA45.6641 mm，原来就在201条测试池。它不参与本轮开发；记录编号和参考MA仅用于诊断，未进入新规则。',
    '用相同100次外院抽样、相同模型和超参数分别保留原60条、只交换2条正常记录、只交换1条低MA记录、交换全部3条。以下为双源对该记录的预测。该实验仅定位原因，不据测试成绩选择开发成员。','',
    '| 开发池变化 | 预测MA均值 (mm) | 100次范围 (mm) | 判定 |','|---|---:|---:|---|']
labels={'original':'原60条','normal_swaps_only':'只交换2条正常记录','low_swap_only':'只交换1条低MA记录','all_three_swaps':'交换全部3条'}
for k,label in labels.items():
    a=np.array([x['prediction'] for x in diag['rows'] if x['arm']=='HZ+HB_to_SA' and x['variant']==k]);assert len(a)==100
    lines.append(f'| {label} | {a.mean():.4f} | {a.min():.4f}–{a.max():.4f} | '+('低MA' if (a<50).all() else '正常')+' |')
lines+=['',
    '原低MA开发记录R0036的参考MA为45.8516 mm，替换记录R0173为48.0078 mm；原开发低MA最小值45.8516变为48.0078 mm。两者类别相同，但提供的低端数值信息不同。当前只有4条低MA开发记录，替换一条足以改变局部映射。上述受控结果支持其为此次跨过50 mm边界的主要原因。',
    '第1轮双源中，正常放行的U-TEG幅值下界同时由约2.11661 V降至2.05732 V；该记录幅值2.07299 V，因此改判正常后也通过了幅值检查。原来的独立风险分数上限只约束高MA风险，未对低MA风险施加同样的上限。',
    '对相同模型分别单条查询和批量查询，预测一致；原／新完整划分的预测均与既有保存文件对应。扩大查询池本身没有改变拟合。','',
    '## 2. 最小修正','',
    '保留现有模型、幅值边界、高MA风险90%分位检查及正常分支门。对所有预测正常的记录，再检查低MA风险：其分数不得超过开发侧OOF正常预测中的第90百分位。90%直接沿用既有高侧设置，没有搜索分位数。',
    '收紧正常分支后，原异常分支阈值不一定仍满足开发accuracy>80%，因此沿用原来的候选门与选择程序重新确定低、高状态门：开发FN=0、accuracy>80%、存在正常及真实异常报告，最大化开发放行数量。阈值和规则选择均只使用本轮开发数据。',
    '此处没有把该低MA记录强行改判为低MA：其连续MA预测不变，正常报告被暂缓。因此恢复的是放行端零FN，不是门前映射零错误。','',
    '## 3. 60条开发／220条测试：100次结果','',
    '| 配置 | Accuracy | 报告率 | FNR | FPR | FN轮次 |','|---|---:|---:|---:|---:|---:|']
for v in s:
    if v['budget']==60:lines.append(f"| {arms[v['arm']]} | {fmt(v,'accuracy')} | {fmt(v,'reporting_rate')} | {fmt(v,'FNR')} | {fmt(v,'FPR')} | {v['FN_rounds']}/100 |")
lines+=['',
    '全部100轮保留，全部轮次存在正常报告和真实异常报告，FNR分母非零。双源95轮accuracy>80%，73轮报告率>80%；均值达标，不声称每轮两项均达标。',
    '双源报告率由94.09%降至82.15%，accuracy由84.97%变为84.37%，FPR由17.23%变为18.75%。FPR是放行真实正常记录中的误报比例，不等于新增误报计数。','',
    '## 4. 54条开发／220条测试：100次结果','',
    '| 配置 | Accuracy | 报告率 | FNR | FPR | FN轮次／事件 |','|---|---:|---:|---:|---:|---:|']
for v in s:
    if v['budget']==54:lines.append(f"| {arms[v['arm']]} | {fmt(v,'accuracy')} | {fmt(v,'reporting_rate')} | {fmt(v,'FNR')} | {fmt(v,'FPR')} | {v['FN_rounds']}/100；{v['FN_events']} |")
lines+=['',
    '双源原102个FN重复事件减少至22个，21轮仍有FN。低MA漏判已被阻止，剩余为高MA；未针对这些记录继续调门。',
    '## 5. 其他有限对照','',
    '还保留了按开发异常OOF预测限定正常预测区间、以及其与低风险检查叠加的对照；没有因为测试效果不佳而删除。单纯叠加低风险门且不重选异常门时，部分开发轮次accuracy低于80%，因此它仅是诊断结果，不作为保持原可行性要求的最终候选。全部对照见本目录SUMMARY.csv，保持开发约束的候选见symmetric90_reselected/SUMMARY.csv。',
    '## 6. 验证与边界','',
    '800个配置／轮次逐一重新核算预测类别、放行数、正确数、FN和FP，并检查正常放行是原正常放行的子集。更改所有非开发记录的参考值不影响新规则拟合。原数据、原结果、论文和图表文件哈希保持不变。',
    'Accuracy、FNR、FPR均在放行记录中计算；报告率分母是每轮220条。数值为100轮均值±样本SD。当前零FN是这套已有数据反复内部评价的观察结果，不是未来病例的零漏判保证。',
    '不自动替换正文结果。新候选已完整保存，可据此决定下一步。','']
verification=[]
for budget in [60,54]:
    for rep in range(100):
        with np.load(F/'private'/f'dev{budget}'/f'round_{rep:03d}.npz') as acts:
            for arm in arms:
                with np.load(P/'private'/f'dev{budget}'/f'round_{rep:03d}'/(arm+'.npz')) as z:
                    p=z['prediction'];y=z['reference'];pc=np.where(p<50,0,np.where(p>70,2,1));tc=np.where(y<50,0,np.where(y>70,2,1));a=acts[arm+'_action'];b=z['action_common_B']
                    np.testing.assert_array_equal(acts[arm+'_test'],z['test']);assert not np.any(a&(pc==1)&~b)
                    rr=next(v for v in rows if (v['budget'],v['rep'],v['arm'])==(budget,rep,arm));m=rr['test']
                    assert int(a.sum())==m['reported'];assert int((a&(pc==tc)).sum())==m['correct'];assert int((a&(tc!=1)&(pc==1)).sum())==m['FN'];assert int((a&(tc==1)&(pc!=1)).sum())==m['FP']
                    oldfn=b&(tc!=1)&(pc==1);newfn=a&(tc!=1)&(pc==1)
                    verification.append(dict(budget=budget,rep=rep,arm=arm,prevented_FN=int((oldfn&~newfn).sum()),new_FN=int((newfn&~oldfn).sum()),normal_reports=m['reported_as_normal'],true_abnormal_reports=m['abnormal_released']))
assert all(x['new_FN']==0 for x in verification)
protected=json.loads((O/'PROTECTED_HASHES.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
(O/'FINAL_VERIFICATION.json').write_text(json.dumps(dict(passed=True,count_checks=len(verification),new_FN_events=0,all_original_files_unchanged=True,details=verification),ensure_ascii=False),encoding='utf-8')
(O/'RESULTS_CN.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[:40]))
