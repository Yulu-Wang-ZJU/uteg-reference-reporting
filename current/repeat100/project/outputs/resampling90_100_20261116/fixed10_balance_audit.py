"""Audit every repeat and report all fixed10% gate controls without selecting rounds."""
import json,csv,collections
import numpy as np
from threadpoolctl import threadpool_limits
import run as r
import aux_followup as a
import normal_hard_fn as n
import high_risk_followup as h
import fixed10_gate_balance as b
import fixed10_gap_end as g
import fixed10_classwise as c

def main():
    d,*_=r.init();outcomes=0;failures=[];paired=[];source_hashes={};combined={}
    for budget in [54,60]:
        parents=json.loads((b.O/f'dev{budget}/ROUNDS.json').read_text());gaprows=json.loads((g.O/f'dev{budget}/ROUNDS.json').read_text());classrows=json.loads((c.O/f'dev{budget}/ROUNDS.json').read_text())
        rows=parents+gaprows+classrows;assert len(rows)==1800
        index={(v['rep'],v['arm'],v['policy']):v for v in rows};assert len(index)==1800
        previous=json.loads((h.O/f'dev{budget}/FIXED_ROUNDS.json').read_text());oldindex={(v['rep'],v['arm']):v for v in previous if v['policy']=='B_cap_0.9'}
        for rep in range(100):
            for root in [b.O,g.O,c.O]:assert (root/f'dev{budget}/private/round_{rep:02d}/COMPLETE.json').exists()
            acts0=np.load(b.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS.npz');acts1=np.load(g.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS.npz');acts2=np.load(c.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS.npz')
            gap=json.loads((g.O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text());classes=json.loads((c.O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text())
            for arm in a.ARMS:
                z,risk,base=b.load(budget,rep,arm);dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2];ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
                assert len(z['dev_rows'])==budget and len(z['test'])==181;assert not set(d['groups'][z['dev_rows']])&set(d['groups'][z['test']])
                normaldev=r.state(z['dev_prediction'])==1;normaltest=r.state(z['prediction'])==1
                for saved in [acts0,acts1,acts2]:np.testing.assert_array_equal(saved['test'],z['test']);np.testing.assert_array_equal(saved['dev_rows'],z['dev_rows'])
                picks=json.loads((b.O/f'dev{budget}/private/round_{rep:02d}/{arm}_POLICIES.json').read_text());recalc=b.decisions(z,dy,risk['dev_risk_B'],base);assert r.safe(recalc)==picks
                expectedcap=float(np.quantile(risk['dev_risk_B'][normaldev,1],.9,method='linear'));assert base['cap_score']==expectedcap
                baseact=n.action_for(base,z['prediction'],qs,{'B':risk['test_risk_B']});np.testing.assert_array_equal(baseact,acts0[arm+'_baseline_fixed10'])
                assert index[(rep,arm,'baseline_fixed10')]['test']==oldindex[(rep,arm)]['test']
                assert r.safe(n.extended_metrics(z['prediction'],ty,baseact))==oldindex[(rep,arm)]['test'];outcomes+=1
                allchecks=[]
                for name,p in picks.items():allchecks.append((name,p,b.apply,acts0))
                for floor in [80,85]:
                    name='normal_gap_end_'+str(floor);gp=gap[arm+'_'+name];pp,aa=g.adjust(picks['separate_gates_'+str(floor)],z['dev_prediction'],risk['dev_risk_B'],ds)
                    assert r.safe(pp)==gp['policy'] and r.safe(aa)==gp['adjustments'];allchecks.append((name,gp['policy'],b.apply,acts1))
                normal=gap[arm+'_normal_gap_end_80']['policy']['normal'];nm=b.normal_action(normal,z['dev_prediction'],risk['dev_risk_B'],ds)
                for reserve in [0,1]:
                    name='classwise_reserve_'+str(reserve);pick=classes[arm+'_'+name];assert pick['normal']==normal
                    cp=c.choose(z['dev_prediction'],dy,ds,nm,reserve);assert r.safe(cp)==pick['abnormal'];allchecks.append((name,pick,c.apply,acts2))
                for name,p,apply,saved in allchecks:
                    row=index[(rep,arm,name)];devact=apply(p,z['dev_prediction'],risk['dev_risk_B'],ds);testact=apply(p,z['prediction'],risk['test_risk_B'],qs)
                    if testact is None:
                        assert row['test'] is None and row['development'] is None;assert not saved[arm+'_'+name].any()
                    else:
                        np.testing.assert_array_equal(testact,saved[arm+'_'+name])
                        assert r.safe(n.extended_metrics(z['dev_prediction'],dy,devact))==row['development']
                        assert r.safe(n.extended_metrics(z['prediction'],ty,testact))==row['test']
                        assert row['development']['FN']==0 and row['development']['reported_as_normal']>0 and row['development']['abnormal_released']>0
                        assert not np.any(testact&normaltest&(risk['test_risk_B'][:,1]>expectedcap))
                        assert not np.any(devact&normaldev&(risk['dev_risk_B'][:,1]>expectedcap))
                        if name.startswith('locked_normal'):np.testing.assert_array_equal(testact&normaltest,baseact&normaltest)
                    outcomes+=1
                source=a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz';source_hashes[str(source)]=r.sha(source)
        for arm in a.ARMS:
            for name in dict.fromkeys(v['policy'] for v in rows):
                rr=[v for v in rows if v['arm']==arm and v['policy']==name];assert len(rr)==100
                bad=[]
                for v in rr:
                    if not v['joint_test_pass']:
                        met=v['test'];bad.append(dict(rep0=v['rep'],repeat=v['rep']+1,feasible=v['feasible'],accuracy=None if met is None else met['accuracy'],reporting_rate=None if met is None else met['reporting_rate'],FNR=None if met is None else met['FNR'],normal_reports=None if met is None else met['reported_as_normal'],developer_reporting_rate=(v['development'] or {}).get('reporting_rate')))
                failures.append(dict(budget=budget,arm=arm,policy=name,infeasible=sum(not v['feasible'] for v in rr),test_FN=[v['rep']+1 for v in rr if v['test'] and v['test']['FN']>0],test_rate_fail=[v['rep']+1 for v in rr if v['test'] and v['test']['reporting_rate']<=.75],test_accuracy_fail=[v['rep']+1 for v in rr if v['test'] and (v['test']['accuracy'] is None or v['test']['accuracy']<=.8)],test_no_normal=[v['rep']+1 for v in rr if v['test'] and v['test']['no_normal_reports']],test_FNR_undefined=[v['rep']+1 for v in rr if v['test'] and v['test']['FNR'] is None],failure_details=bad))
                changes=[]
                for v in rr:
                    original=index[(v['rep'],arm,'baseline_fixed10')]
                    if v['test'] and original['test']:
                        changes.append(dict(rep=v['rep'],correct=v['test']['correct']-original['test']['correct'],incorrect=v['test']['incorrect']-original['test']['incorrect'],normal=v['test']['reported_as_normal']-original['test']['reported_as_normal'],reporting_rate=v['test']['reporting_rate']-original['test']['reporting_rate'],FN=v['test']['FN']-original['test']['FN']))
                paired.append(dict(budget=budget,arm=arm,policy=name,changes=changes))
        combined[budget]=h.summarize(rows);r.write(b.O/f'combined/dev{budget}/ROUNDS.json',rows);r.write(b.O/f'combined/dev{budget}/SUMMARY.json',combined[budget]);flat=[]
        for v in rows:
            record={k:v[k] for k in ['budget','rep','arm','policy','feasible','development_joint_pass','joint_test_pass']}
            for side in ['development','test']:
                for key in ['accuracy','reporting_rate','FNR','FPR','FN','FP','correct','incorrect','reported_as_normal','reported_as_low','reported_as_high','abnormal_released']:
                    record[side+'_'+key]=(v[side] or {}).get(key)
            flat.append(record)
        with (b.O/f'combined/dev{budget}/ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as file:
            wr=csv.DictWriter(file,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
        print('BALANCE AUDIT',budget,'PASS',flush=True)
    previous=json.loads((h.O/'SOURCE_PREDICTION_HASHES.json').read_text())
    assert all(previous[key]==value for key,value in source_hashes.items())
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in b.protocol()['input_hashes'].items())
    r.write(b.O/'REMAINING_FAILURES.json',failures);r.write(b.O/'PAIRED_CHANGES.json',paired);r.write(b.O/'SOURCE_PREDICTION_HASHES.json',source_hashes)
    r.write(b.O/'FINAL_QA.json',dict(outcomes_recomputed=outcomes,budgets=[54,60],repeats_per_budget=100,all_policy_selection_reproduced=True,development_FN_zero_for_feasible_policies=True,original_fixed10_caps_unchanged=True,original_risk_scores_and_MA_predictions_unchanged=True,normal_release_masks_and_abnormal_masks_checked=True,baseline_reproduced=True,input_hashes_unchanged=True))
    report(combined,failures,paired);print('BALANCE FINAL QA PASS',outcomes,flush=True)

def report(summaries,failures,paired):
    names={'baseline_fixed10':'原固定10%方案','locked_normal_80':'锁定normal，单独选异常门，80%','locked_normal_85':'锁定normal，单独选异常门，85%','separate_gates_80':'normal/异常分开选门，80%','separate_gates_85':'normal/异常分开选门，85%','normal_gap_end_80':'分开选门＋normal阈值区间端点，80%','normal_gap_end_85':'分开选门＋normal阈值区间端点，85%','classwise_reserve_0':'低/normal/高分别选门，80%','classwise_reserve_1':'分三类选门＋少允许一次开发错误'}
    text=['# 固定10%高风险策略：稳定accuracy和reporting rate','',
        '论文、图表、MA映射、风险模型、三分类边界以及开发normal风险90分位上限均未改变。54条和60条各100次，单源/双源分开；全部固定候选和失败轮次保留。',
        '', '## 主要发现','',
        '固定10%风险限制后，低报告率并不全部来自normal安全检查。统一门还拦住大量预测为异常的记录，其中混有正确异常报告和FP。低accuracy轮次则放行了较多FP。因而分别选择normal、低MA和高MA的门，比简单提高整体accuracy门槛更有针对性。',
        '', '预设的三类分别选门、开发accuracy>80%对照，在完整60条单源的100次测试中全部达到FNR=0、accuracy>80%、reporting rate>75%。其余设置仍有失败轮次，不宣称全任务达标。',
        '', '## 固定流程与选择规则','',
        '1. normal继续接受原幅值风险检查，高MA风险阈值不超过原开发90分位。低/高风险界限和normal门只用开发侧零FN条件选择。',
        '2. normal阈值端点对照只把阈值移到下一被排除开发分数之前，开发侧每条记录的放行状态完全不变。10%风险上限不放宽。',
        '3. 低MA和高MA分别从All、HGB、Logistic、Boundary中选门，阈值只来自对应开发预测类别。要求总体开发accuracy>80%、FN=0，且normal及真实异常报告非空，再最大化报告数。',
        '4. 同时保留85%开发accuracy和“少允许一次开发错误”对照，没有按测试结果逐轮更换策略。无可行规则记NA；低于75%报告率记失败。',
        '', '本轮主要重选已有分数的门规则，没有重新训练MA模型。均值±标准差描述100次重叠抽样的波动，不是独立外部验证或95%置信区间。',
        '', '## 三类分别选门（80%）的结果','', '|设置|流程|Accuracy(%)|Reporting rate(%)|FPR(%)|测试零FN|测试三项达标|开发与测试同时三项达标|','|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(q,k):
        v=q[k]
        return 'NA' if v is None else f"{100*v['mean']:.2f} ± {100*v['sd']:.2f}" if v['sd'] is not None else f"{100*v['mean']:.2f}"
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['policy']=='classwise_reserve_0':text.append('|'+ '|'.join([str(budget),q['arm'],*[fmt(q,k) for k in ['accuracy','reporting_rate','FPR']],f"{q['zero_FN']}/100",f"{q['test_joint']}/100",f"{q['both_joint']}/100"])+'|')
    text+=['','三项为FNR=0、accuracy>80%、reporting rate>75%。所有此对照的可行开发规则均为零FN；“开发与测试同时达标”另外检查开发报告率是否也>75%。不得把仅测试达标数替代两侧同时达标数。',
        '', '## 仍未达标的轮次','', '|条数|流程|报告率不足75%的轮次|accuracy不足或等于80%的轮次|最低报告率|最低accuracy|','|---|---|---|---|---:|---:|']
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['policy']!='classwise_reserve_0':continue
            fail=next(v for v in failures if v['budget']==budget and v['arm']==q['arm'] and v['policy']==q['policy'])
            text.append(f"|{budget}|{q['arm']}|{fail['test_rate_fail']}|{fail['test_accuracy_fail']}|{100*q['reporting_rate']['min']:.2f}%|{100*q['accuracy']['min']:.2f}%|")
    text+=['','上表轮次从1开始。所有轮次原样保留，没有按表现删除、重新抽样或强制放行补数。',
        '', '## 放行增加了什么','', '|条数|流程|平均新增正确报告|平均新增错误报告|报告率平均增加(百分点)|每轮normal报告范围|','|---|---|---:|---:|---:|---|']
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['policy']!='classwise_reserve_0':continue
            delta=next(v for v in paired if v['budget']==budget and v['arm']==q['arm'] and v['policy']==q['policy'])['changes'];nr=q['reported_as_normal']
            text.append(f"|{budget}|{q['arm']}|{np.mean([v['correct'] for v in delta]):.2f}|{np.mean([v['incorrect'] for v in delta]):.2f}|{100*np.mean([v['reporting_rate'] for v in delta]):.2f}|{int(nr['min'])}—{int(nr['max'])}|")
    text+=['','这是相对于上一轮原固定10%方案的配对变化。新增错误主要是FP；没有改动连续MA，也没有把暂缓异常算成正确识别。',
        '', '## 全部固定对照','', '|条数|流程|方案|Accuracy(%)|Reporting rate(%)|FPR(%)|零FN轮数|测试三项达标|','|---|---|---|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        for q in summaries[budget]:text.append('|'+ '|'.join([str(budget),q['arm'],names[q['policy']],*[fmt(q,k) for k in ['accuracy','reporting_rate','FPR']],f"{q['zero_FN']}/100",f"{q['test_joint']}/100"])+'|')
    text+=['','## 本轮判断','',
        '- 把normal与异常的放行职责分开，能在不放宽10%风险限制的情况下找回报告率。',
        '- 完整60条单源已在本轮100次测试中同时达标；60条双源及54条两种迁移仍需解决少数失败轮次。',
        '- 提高开发accuracy到85%没有稳定提高联合达标数，反而压低报告率；不能把更严格的单一阈值当成通用解决办法。',
        '- 各固定策略均完整报告，不按某轮测试表现临时切换。研究历史已反复查看该测试队列，当前结果仍属探索性改进，不能承诺未来FNR绝对为零。',
        '', '## 文件','',
        '- combined/dev54/ROUNDS.csv、combined/dev60/ROUNDS.csv：每套1800行，包含9种策略×2种来源配置×100轮；开发和测试指标分列。',
        '- combined/dev54/SUMMARY.json、combined/dev60/SUMMARY.json：所有指标均值、SD、最小最大、有效数、失败数。',
        '- REMAINING_FAILURES.json：逐策略失败轮次和具体指标，轮次从1开始。',
        '- DIAGNOSTIC_ALL_ROUNDS.json：原固定10%方案中风险限制与整体门的暂缓贡献。',
        '- PAIRED_CHANGES.json：相对原方案的正确/错误报告与normal报告变化。',
        '- FINAL_QA.json：3600个结果重算、开发选型复现、10%风险上限及输入哈希检查。']
    (b.O/'REPORT_CN.md').write_text('\n'.join(text)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
