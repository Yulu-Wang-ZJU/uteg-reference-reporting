"""Readable complete-result inventory, from saved arrays only."""
import complete as m
from pathlib import Path
import numpy as np
O=m.O;S=O/'summary';r=m.r
def read(n):return m.read(S/(n+'.json'))
def one(rows,**kw):return next(x for x in rows if all(x.get(k)==v for k,v in kw.items()))
def fmt(x,key,pct=False):
    v=x.get(key)
    if v is None:return '未定义'
    a=100 if pct else 1
    text=f"{v['mean']*a:.2f} ± {v['sd']*a:.2f}" if v['sd'] is not None else f"{v['mean']*a:.2f}"
    return text+(f"（有效{v['n']}轮）" if v.get('n',100)<100 else '')
names={'target_only_SA':'Target-only','HZ_to_SA':'湖州→浙二','HB_to_SA':'湖北→浙二','HZ+HB_to_SA':'双源→浙二'}
mapping=read('all_report_mapping_summary');reporting=read('early_reporting_summary');abl=read('gate_ablation_summary');tab=read('Table1_summary');linear=read('linear_control_summary');raw=read('early_reporting_rounds')
audit=m.read(O/'FINAL_AUDIT.json');assert audit['passed']
lines=['# 固定60条方案：缺失实验补齐与结果核对','',
'本次只补算和整理实验，未修改正文、附录、图表、PDF、老师用包、补充ZIP或GitHub。模型、损失、风险分位数和门选择程序均沿用已固定方案。所有轮次保留，没有按测试结果选参数或筛轮次。','',
'## 设置','',
'- 主方案：浙二60条开发、244条测试池；每轮测试分层抽220条（低7、正常187、高26），共100轮。','- 敏感性：同一开发池抽54条，与主方案共享来源与测试抽样，单独100轮。','- 外院来源仍为湖州86／95、湖北71／79。其他两中心作为目标的结果及301条内部评价未重跑。','- 下表为100轮均值±样本标准差；并非95%置信区间，也不是100个独立队列。','- 门后三分类accuracy以已放行数为分母；reporting rate以当轮全部220条为分母。门后FNR以已放行的真实异常数为分母，FPR以已放行的真实正常数为分母。暂缓不是正确识别。','',
'## 补齐清单','',
'|内容|本次完成情况|','|---|---|',
'|Table 1|三种迁移配置；6种来源模型＋Hybrid regression、5种校正；54／60分别100轮；4个TEG读数，共7,200条表项结果|',
'|Table 2与Fig. 3|相同预测上的All report、Boundary、Logistic、HGB、B95与固定新门；逐条放行去向、FN／FP、曲线及工作点|',
'|Fig. 2|固定seed1000、全部来源记录、60／244参考拟合；4读数、散点及混淆矩阵数据；10例按MA范围选取并重建曲线|',
'|5／10／15分钟|两套×100轮×4配置×3时点；使用对应时间参数；保留不可行或无报告的轮次|',
'|新门消融|完整门及移除低风险、高风险、幅值边界、normal评分门、双风险、所有normal检查；固定其余阈值，不重选|',
'|Fig. 4能力比较|HGB原设计：固定10／20／40%报告率，匹配85／90／95%观测accuracy；不能达到的匹配保留缺失标记|',
'|时间策略|5–10–15、10–15、固定15分钟；首次报告锁定与允许更新后15分钟使用当前结果分别计算|',
'|公平支持对照|20／40／60开发预算；Target-only Ridge；固定相同损失、权重和惩罚的线性／RBF比较|',
'|其他清单项|重算边界及5分钟增长关联；九方向合并仅替换浙二三行；计算耗时单列，不冒充临床周转时间|','',
'## 60条主方案：门前映射（15分钟）','',
'|配置|MAE mm|R²|三分类accuracy %|低MA召回 %|高MA召回 %|FNR %|FPR %|','|---|---|---|---|---|---|---|---|']
for arm in m.q.ARMS:
    x=one(mapping,budget=60,arm=arm,time=900);lines.append('|'+names[arm]+'|'+'|'.join(fmt(x,k,p) for k,p in [('MAE',False),('R2',False),('accuracy',True),('low_recall',True),('high_recall',True),('FNR',True),('FPR',True)])+'|')
lines+=['','## 60条主方案：固定新门后（15分钟）','','|配置|accuracy %|报告率 %|FNR %|FPR %|出现FN轮数|normal报告数范围|','|---|---|---|---|---|---|---|']
for arm in m.q.ARMS:
    x=one(reporting,budget=60,arm=arm,time=900,rule='common_B');rr=[v for v in raw if v['budget']==60 and v['arm']==arm and v['time']==900 and v['rule']=='common_B'];ns=[v['test']['reported_as_normal'] for v in rr]
    lines.append('|'+names[arm]+'|'+'|'.join(fmt(x,k,True) for k in ['accuracy','reporting_rate','FNR','FPR'])+f"|{sum(v['test']['FN']>0 for v in rr)}/100|{min(ns)}–{max(ns)}|")
lines+=['','这里的零FN限于本次60条开发设置、15分钟、固定新门的100轮观察结果，不代表门前预测零FN，也不代表早期或54条均零FN。每轮normal及真实异常报告均非空。','',
'## 5／10／15分钟：双源固定新门','','这是各时点独立判断，不是首次报告锁定的累计结果。','','|开发条数|时间|accuracy %|报告率 %|FNR %|FPR %|出现FN轮数|无报告轮数|','|---|---|---|---|---|---|---|---|']
for b in [60,54]:
    for h in [300,600,900]:
        x=one(reporting,budget=b,arm='HZ+HB_to_SA',time=h,rule='common_B');rr=[v for v in raw if v['budget']==b and v['arm']=='HZ+HB_to_SA' and v['time']==h and v['rule']=='common_B']
        lines.append(f'|{b}|{h//60}分钟|'+'|'.join(fmt(x,k,True) for k in ['accuracy','reporting_rate','FNR','FPR'])+f"|{sum(v['test']['FN']>0 for v in rr)}/100|{sum(v['test']['reported']==0 for v in rr)}/100|")
lines+=['','## 固定阈值消融：双源','','|开发条数|移除项|accuracy %|报告率 %|FNR %|FPR %|FN总事件|','|---|---|---|---|---|---|---|']
ar=read('gate_ablation_rounds');labels={'full':'完整门','without_low_risk':'低风险检查','without_high_risk':'高风险检查','without_B_bounds':'幅值边界','without_normal_score':'normal评分门','without_both_risks':'低／高风险检查','without_normal_checks':'全部normal检查'}
for b in [60,54]:
    for key,label in labels.items():
        x=one(abl,budget=b,arm='HZ+HB_to_SA',ablation=key);rr=[v for v in ar if v['budget']==b and v['arm']=='HZ+HB_to_SA' and v['ablation']==key]
        lines.append(f'|{b}|{label}|'+'|'.join(fmt(x,k,True) for k in ['accuracy','reporting_rate','FNR','FPR'])+f"|{sum(v['test']['FN'] for v in rr)}|")
lines+=['','消融不重新选异常门，因而与“先添加检查，再重新选异常门”的开发过程诊断不是同一种比较。重复事件不是独立患者数。','',
'## Table 1主方案数值','','|配置|比较|方法|R MAE min|K MAE min|MA MAE mm|角度MAE degrees|','|---|---|---|---|---|---|---|']
for arm in m.q.ARMS[1:]:
    for x in [v for v in tab if v['budget']==60 and v['arm']==arm]:lines.append('|'+names[arm]+'|'+x['panel']+'|'+x['method']+'|'+'|'.join(fmt(x,k) for k in ['R_MAE','K_MAE','MA_MAE','angle_MAE'])+'|')
lines+=['','## 线性／RBF公平对照（60条、15分钟）','','|配置|映射|门|MAE mm|R²|accuracy %|报告率 %|FNR %|FPR %|','|---|---|---|---|---|---|---|---|---|']
for arm in m.q.ARMS:
    for name,data in [('RBF',reporting),('Linear',linear)]:
        for rule in ['All report','common_B']:
            x=one(data,**dict(budget=60,arm=arm,rule=rule,**({'time':900} if name=='RBF' else {})))
            lines.append('|'+names[arm]+'|'+name+'|'+rule+'|'+'|'.join(fmt(x,k,p) for k,p in [('MAE',False),('R2',False),('accuracy',True),('reporting_rate',True),('FNR',True),('FPR',True)])+'|')
lines+=['','## Fig. 2：完整244条参考评价','','本项为完整来源记录、固定seed1000的一次参考拟合，用于逐记录展示；不代替100轮均值。三个主展示配置均使用同一244条测试记录。','','|配置|门前MAE mm|门前R²|门前accuracy %|门前FN|门后FN|','|---|---|---|---|---|---|']
fr=m.read(O/'reference244/RESULTS.json')
for arm in m.q.ARMS:
    x=one(fr,arm=arm,rule='All report')['test'];y=one(fr,arm=arm,rule='common_B')['test'];lines.append(f"|{names[arm]}|{x['MAE']:.3f}|{x['R2']:.3f}|{x['accuracy']*100:.2f}|{x['FN']}|{y['FN']}|")
lines+=['','## 核对与下一步','','- 小范围测试：修改非训练标签，预测、分数和开发侧门槛不变；早期计算只请求对应时间的参数。','- Table 1六种来源模型的校正通过标签隔离测试；600组Hybrid最终MA与固定主预测逐条相同。','- 800组15分钟预测／放行结果与已固定60／54结果完全一致。','- 逐条重算14,400行主时点指标，混淆矩阵、FN／FP、报告数与已保存结果一致。','- 保留所有100轮，包括无报告、匹配不可达和未达均值目标的个别轮次；没有替换或删除失败轮次。','- 正文、附录、PDF、ZIP及固定输入的哈希未改变。其他目标中心复用原结果，并在九方向合集标明门版本。','- 结果齐备后可进入最小文字／数值替换；不能继续保留旧的“门前零FN”“54条全零FN”“复合门比HGB报告更多”等断言。5／10分钟及首次报告的FN须按各自结果说明。','',
'## 文件入口','','- 完整轮次及汇总CSV／JSON：`summary/`','- 固定时点逐条预测、风险分数、阈值与动作：`primary/`','- Table 1控制预测：`table1/`；线性对照：`linear/`；预算：`budgets/`','- Fig. 2逐记录结果及十例曲线：`reference244/`','- 固定阈值消融逐条动作：`ablation/`','- 检查证据：`SMOKE.json`、`CONTROL_AUDIT.json`、`FINAL_AUDIT.json`、`SOURCE_CACHE_MANIFEST.json`。','']
(O/'RESULTS_CN.md').write_text('\n'.join(lines),encoding='utf-8')
r.write(O/'DELIVERABLE_INDEX.json',dict(status='Experiments completed; manuscript unchanged',primary60=True,sensitivity54_separate=True,all100_retained=True,files=[str(p.relative_to(O)) for p in sorted(S.glob('*'))],audit=audit))
print('REPORT READY',O/'RESULTS_CN.md')
