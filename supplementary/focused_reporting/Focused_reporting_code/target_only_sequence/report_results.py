from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
d=json.loads((R/'RESULTS.json').read_text(encoding='utf-8'))
names={'Target-only Ridge':'Target-only','HZ_to_SA':'湖州→浙二','HZ+HB_to_SA':'湖州＋湖北→浙二'}
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
def interval(x):return f"{100*x['estimate']:+.2f} [{100*x['lo']:+.2f}, {100*x['hi']:+.2f}]"
lines=['# Target-only 完整序贯对照：结果记录','',
'本轮仅补实验与独立结果记录；正文、附录、补充材料包和公开仓库未改动。','',
'固定设置：同一批60条目标中心标注，201条测试记录，20个种子（1000–1019）；三个检查点共用60条标注。5/10分钟使用对应前缀并按原嵌套开发程序拟合与选阈值，15分钟直接复用已冻结的target-only模型。首次报告锁定类别与时间。',
'计数单位为4,020个record–seed events，不是4,020名患者。测试集包含7条低MA、170条范围内和24条高MA记录。95%区间为5,000次成对历史identity-proxy-group bootstrap，保留组内全部种子，是既有队列上的条件性描述区间。','',
'## 1. 完整5/10/15分钟序贯流程','',
'| 流程 | 正确报告 | 错误报告 | 未报告 | 覆盖率 | 报告内一致率 | 零报告种子 |',
'|---|---:|---:|---:|---:|---:|---:|']
for a,n in names.items():
    s=d['arms'][a]['5-10-15'];c=s['counts'];lines.append(f"| {n} | {c['correct']} | {c['incorrect']} | {c['unreported']} | {pct(s['yields']['coverage']['estimate'])} | {pct(s['agreement']['estimate'])} | {s['zero_report_seeds']}/20 |")
lines+=['','## 2. 同时间计划的成对比较','',
'差值方向固定为迁移流程减target-only；单位为全测试集事件比例的百分点。正确报告增量与错误报告增量分别解释，不合成为未预设的临床净收益。','',
'| 对比 | 正确报告差值[95%区间] | 错误报告差值[95%区间] | 覆盖率差值[95%区间] |','|---|---:|---:|---:|']
for a in ['HZ_to_SA','HZ+HB_to_SA']:
    c=d['source_minus_target'][a]['5-10-15'];lines.append(f"| {names[a]} − Target-only | {interval(c['correct'])} | {interval(c['incorrect'])} | {interval(c['coverage'])} |")
lines+=['','## 3. 各检查点累计结果','',
'| 流程 | 累计至 | 正确 | 错误 | 未报告 | 覆盖率 | 报告内一致率 |','|---|---|---:|---:|---:|---:|---:|']
for a,n in names.items():
    for p,t in [('5','5分钟'),('5-10','10分钟'),('5-10-15','15分钟')]:
        s=d['arms'][a][p];c=s['counts'];lines.append(f"| {n} | {t} | {c['correct']} | {c['incorrect']} | {c['unreported']} | {pct(s['yields']['coverage']['estimate'])} | {pct(s['agreement']['estimate'])} |")
lines+=['','## 4. 不同报告计划','',
'| 流程 | 计划（分钟） | 正确 | 错误 | 未报告 | 覆盖率 | 报告内一致率 |','|---|---|---:|---:|---:|---:|---:|']
for a,n in names.items():
    for p in ['15','10-15','5-10-15']:
        s=d['arms'][a][p];c=s['counts'];lines.append(f"| {n} | {p} | {c['correct']} | {c['incorrect']} | {c['unreported']} | {pct(s['yields']['coverage']['estimate'])} | {pct(s['agreement']['estimate'])} |")
lines+=['','## 5. 完整序贯流程的异常状态去向','',
'| 流程 | 低MA正确报告 / 140 | 低MA报为范围内 / 140 | 高MA正确报告 / 480 | 高MA报为范围内 / 480 |','|---|---:|---:|---:|---:|']
for a,n in names.items():
    c=d['arms'][a]['5-10-15']['counts'];lines.append(f"| {n} | {c['correct_low']} | {c['low_to_within']} | {c['correct_high']} | {c['high_to_within']} |")
lines+=['','## 6. Target-only 中加入早期报告的效果','',
'差值方向为完整5/10/15计划减只在15分钟报告。单位为百分点。','']
for k,label in [('correct','正确报告'),('incorrect','错误报告'),('coverage','覆盖率')]:
    lines.append(f"- {label}：{interval(d['schedule_contrasts']['Target-only Ridge']['5-10-15 minus 15'][k])}。")
rows=list(csv.DictReader((R/'paired_seeds.csv').open(encoding='utf-8-sig')))
lines+=['','## 7. 各种子方向','',
'种子是同一患者队列上的重复拟合，不作为独立样本检验。','',
'| 对比 | 正确更多 / 相同 / 更少 | 错误更多 / 相同 / 更少 |','|---|---|---|']
for a in ['HZ_to_SA','HZ+HB_to_SA']:
    rr=[r for r in rows if r['comparison']==a+' minus target-only' and r['schedule']=='5-10-15'];out=[]
    for k in ['correct','incorrect']:
        v=[int(r[k]) for r in rr];out.append(' / '.join(str(sum(test(x) for x in v)) for test in [lambda x:x>0,lambda x:x==0,lambda x:x<0]))
    lines.append(f"| {names[a]} − Target-only | {out[0]} | {out[1]} |")
lines+=['','## 文件与验证','',
'- summary.csv：单时点与累计/计划结果。',
'- paired_contrasts.csv：同计划迁移减target-only的成对区间。',
'- first_report_times.csv、class_destinations.csv：完整时间与分类去向。',
'- per_seed.csv、paired_seeds.csv：全部种子的结果与差值。',
'- RESULTS.json：完整统计结果及计划内比较区间。',
'- FREEZE.json、EVALUATION_INPUT_HASHES.json、CHECKS.json、VERIFICATION.json：冻结、溯源和核验。','',
'首次报告时间是U-TEG信号观察窗口，不是与常规TEG实际工作流比较得到的节省时间。模型和逐记录验证数组仅保存在本地private/及模型目录，不在公开交付范围。']
p=R/'RESULTS_REPORT.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(p)
