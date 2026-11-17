"""Recompute three-policy outcomes, inspect failures, write a concise audit report."""
import json, collections
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
import normal_hard_fn as n
import run as r
import aux_followup as a
import support_followup as s

def main():
    d,*_=r.init();y=d['medical'][:,2];items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    hashes={};checked=0;models=0;diagnostics={};overlap=[]
    for budget in [54,60]:
        records=json.loads((n.O/f'dev{budget}/ROUNDS.json').read_text(encoding='utf8'))
        assert len(records)==600 and sorted(set(v['rep'] for v in records))==list(range(100))
        previous=json.loads((a.O/f'threshold_interval/dev{budget}/ROUNDS.json').read_text(encoding='utf8'))
        old={(v['rep'],v['arm']):v for v in previous if v['rule']=='B_development_gate' and v['variant']=='midpoint'}
        tally={arm:dict(original_FN_events=0,withheld_original_FN=0,remaining_original_FN=0,new_FN_events=0,original_FN_rounds=[],remaining_FN_rounds=[],development_FN_records=collections.Counter(),original_FN_missing206=[],feasibility={}) for arm in a.ARMS}
        for rep in range(100):
            folder=n.O/f'dev{budget}/private/round_{rep:02d}';sp=s.spec(items[rep],budget)
            assert (folder/'COMPLETE.json').exists()
            saved=np.load(folder/'ACTIONS_RISKS.npz');audits=json.loads((folder/'RISK_TRAINING.json').read_text(encoding='utf8'));lookup={int(v):i for i,v in enumerate(saved['dev_rows'])}
            assert len(saved['dev_rows'])==budget and set(saved['dev_rows'])==set(sp['pool']) and len(saved['test'])==181
            np.testing.assert_array_equal(saved['test'],sp['test'])
            for model,au in audits.items():
                dev=np.full((budget,2),np.nan);query=None
                for pos,fit in enumerate(au):
                    train=np.array(fit['train']);qq=np.array(fit['query']);cols=fit['columns']
                    assert not set(d['groups'][train])&set(d['groups'][qq])
                    part=sp['positions'][pos];assert set(train)==set(part['A']) and set(qq)==set(part['B'])
                    x=d['x'][train][:,cols];mean=np.array(fit['mean']);scale=np.array(fit['scale'])
                    np.testing.assert_allclose(mean,x.mean(0),rtol=1e-12,atol=1e-12)
                    std=x.std(0);std[std==0]=1;np.testing.assert_allclose(scale,std,rtol=1e-12,atol=1e-12)
                    score=[]
                    for head in fit['heads']:
                        assert head['positive_count']==int(np.sum(r.state(y[train])==head['class_index']))
                        if head['status']=='fit':
                            p=expit(((d['x'][qq][:,cols]-mean)/scale)@np.array(head['coef'])[0]+head['intercept'][0])
                        else:p=np.ones(len(qq))
                        score.append(p);models+=1
                    score=np.array(score).T
                    if pos<3:dev[[lookup[int(v)] for v in qq]]=score
                    else:query=score
                np.testing.assert_allclose(dev,saved['dev_risk_'+model],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(query,saved['test_risk_'+model],atol=1e-12,rtol=1e-12)
            risks={k:(saved['dev_risk_'+k],saved['test_risk_'+k]) for k in n.FEATURES}
            for arm in a.ARMS:
                source=a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz';hashes[str(source)]=r.sha(source)
                z=np.load(source);sel=json.loads((folder/(arm+'_SELECTION.json')).read_text(encoding='utf8'))
                picks,candidates=n.decide(z,y[z['dev_rows']],risks)
                assert r.safe(picks)==sel['policies'] and r.safe(candidates)==sel['normal_check_candidates']
                qs={k:z['score_'+k] for k in a.RULES};ds={k:z['dev_score_'+k] for k in a.RULES}
                rr={v['policy']:v for v in records if v['rep']==rep and v['arm']==arm}
                for policy,pick in picks.items():
                    action=n.action_for(pick,z['prediction'],qs,{k:v[1] for k,v in risks.items()})
                    if action is None:
                        assert rr[policy]['test'] is None and not saved[arm+'_'+policy+'_action'].any()
                    else:
                        np.testing.assert_array_equal(action,saved[arm+'_'+policy+'_action'])
                        assert r.safe(n.extended_metrics(z['prediction'],y[z['test']],action))==rr[policy]['test']
                        da=n.action_for(pick,z['dev_prediction'],ds,{k:v[0] for k,v in risks.items()})
                        assert r.safe(n.extended_metrics(z['dev_prediction'],y[z['dev_rows']],da))==rr[policy]['development']
                        if policy!='existing':assert rr[policy]['development']['FN']==0 and rr[policy]['development']['reported_as_normal']>0 and rr[policy]['development']['abnormal_released']>0
                    checked+=1
                for key,value in old[(rep,arm)]['test'].items():assert rr['existing']['test'][key]==value
                devFN=(r.state(z['dev_prediction'])==1)&(r.state(y[z['dev_rows']])!=1)
                for ix in z['dev_rows'][devFN]:tally[arm]['development_FN_records'][str(ix)]+=1
                aa=saved[arm+'_existing_action'];bb=saved[arm+'_normal_check_action'];fn=(r.state(z['prediction'])==1)&(r.state(y[z['test']])!=1)
                original=aa&fn;remaining=bb&fn
                tally[arm]['original_FN_events']+=int(original.sum());tally[arm]['withheld_original_FN']+=int(np.sum(original&~bb));tally[arm]['remaining_original_FN']+=int(np.sum(original&bb));tally[arm]['new_FN_events']+=int(np.sum(remaining&~original))
                if original.any():
                    tally[arm]['original_FN_rounds'].append(rep);tally[arm]['original_FN_missing206'].append(206 not in sp['pool'])
                if remaining.any():tally[arm]['remaining_FN_rounds'].append(rep)
        for arm in a.ARMS:
            tally[arm]['development_FN_records']=dict(tally[arm]['development_FN_records'])
            for policy in n.POLICIES:
                rr=[v for v in records if v['arm']==arm and v['policy']==policy]
                q=dict(budget=budget,arm=arm,policy=policy,
                    no_FN_safe_rule=[v['rep'] for v in rr if not v['feasible']],
                    developer_rate_below_target=[v['rep'] for v in rr if v['feasible'] and v['development']['reporting_rate']<=.75],
                    developer_nonzero_FN=[v['rep'] for v in rr if v['feasible'] and v['development']['FN']>0],
                    test_FN=[v['rep'] for v in rr if v['test'] and v['test']['FN']>0],
                    test_undefined_FNR=[v['rep'] for v in rr if v['test'] and v['test']['FNR'] is None],
                    test_rate_below_target=[v['rep'] for v in rr if v['test'] and v['test']['reporting_rate']<=.75],
                    test_accuracy_below_target=[v['rep'] for v in rr if v['test'] and (v['test']['accuracy'] is None or v['test']['accuracy']<=.8)],
                    test_no_normal_reports=[v['rep'] for v in rr if v['test'] and v['test']['reported_as_normal']==0],
                    development_and_test_joint=sum(v['development_joint_pass'] and v['joint_test_pass'] for v in rr))
                overlap.append(q)
        diagnostics[str(budget)]=tally
        print('AUDIT',budget,'PASS',flush=True)
    proto=n.protocol();assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in proto['input_hashes'].items())
    r.write(n.O/'SOURCE_PREDICTION_HASHES.json',hashes);r.write(n.O/'FAILURE_ROUNDS.json',overlap);r.write(n.O/'PAIRED_FN_AUDIT.json',diagnostics)
    r.write(n.O/'FINAL_QA.json',dict(saved_policy_outcomes_recomputed=checked,risk_heads_reconstructed=models,complete_repeats_per_budget=100,source_B_predictions_unchanged=True,existing_policy_reproduced=True,all_hard_alternatives_have_development_FN_zero=True,no_normal_release_not_success=True,undefined_FNR_not_zero=True,group_isolation_checked=True,threshold_selection_reproduced=True,source_input_hashes_unchanged=True))
    report(overlap,diagnostics)
    print('FINAL AUDIT PASS',checked,models,flush=True)

def report(overlap,diagnostics):
    names={'existing':'现有规则（参照）','hard_zero_FN':'统一阈值＋开发零FN硬条件','normal_check':'零FN硬条件＋normal专门检查'}
    out=['# 开发侧零FN硬条件：三方案、两套100次对照','',
        '论文、图表、原始输入与现有MA映射均未修改。本轮复用已经完成的MA模型和嵌套门分数，重新训练normal检查的低/高MA风险头；两套各100次，不是重新训练全部映射模型。',
        '', '## 结论','',
        'normal专门检查能在开发侧零FN约束下明显保住报告率，但本轮没有得到54条设置全部100次零FN的方案。60条的测试零FN仍可保持，报告率稳定性则比旧方案下降。不能把开发零FN解释成测试零FN，也不能把旧方案测试100次零FN说成它满足开发零FN硬条件。',
        '', '## 三方案如何比较','',
        '1. 现有规则：原开发侧FN优先选择HGB/Logistic/Boundary，再用相邻开发分数中点阈值。保留作参照，可能存在开发FN。',
        '2. 零FN硬条件：开发FN必须为0、accuracy>80%，且至少放行一条预测normal和一条真实异常；在合格规则中最大化报告数。低于75%报告率的最好候选保留并标明未完成联合目标。没有这种规则则为NA。',
        '3. normal专门检查：在同样约束下，对准备报告normal的记录再检查低、高MA风险。比较仅幅值与四参数的两个正则化Logistic头，开发侧选择；无根据的normal只能暂缓，不改成异常、不改连续MA。',
        '', '新增风险头使用既定分组开发折：训练折拟合标准化、类别权重和模型，验证折只取得分。最终风险头使用完整当前开发池。所有阈值/特征选择只用开发折外预测；测试标签只用于计分。C=0.1固定，没有继续根据测试结果搜索参数。',
        '', '## 单源主结果','',
        '均值±标准差仅对各指标有定义的轮次计算；分母不是独立患者数。FN/FNR为放行的真实异常被判成normal；未放行任何真实异常时FNR为NA。下表同时给出可行数，未将失败轮次删掉，SUMMARY.json保留每个指标的有效轮数。',
        '', '|开发条数|方案|有零FN硬条件候选的轮次|开发三项达标|测试三项达标|测试Accuracy(%)|测试Reporting rate(%)|测试FNR(%)|测试FPR(%)|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(q,key,scale=100):
        v=q[key]
        if v is None:return 'NA'
        sd=v['sd'];return f"{scale*v['mean']:.2f}"+(f" ± {scale*sd:.2f}" if sd is not None else '')
    sums={}
    for budget in [54,60]:
        sums[budget]=json.loads((n.O/f'dev{budget}/SUMMARY.json').read_text(encoding='utf8'))
        for q in sums[budget]:
            if q['arm']!='HZ_to_SA':continue
            eligible=f"{q['feasible']}/100" if q['policy']!='existing' else '不强制，仅参照'
            out.append('|'+ '|'.join([str(budget),names[q['policy']],eligible,str(q['development_joint_pass'])+'/100',str(q['test_joint_pass'])+'/100',*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']]])+'|')
    out+=['','开发三项达标指开发FNR=0、accuracy>80%、reporting rate>75%；测试三项达标独立计算。二者同时达标的轮次不能由两列直接取较小值。','', '|条数|方案|开发、测试同时达标|无可行规则|测试FNR无定义|','|---|---|---:|---|---|']
    for q in overlap:
        if q['arm']=='HZ_to_SA':out.append(f"|{q['budget']}|{names[q['policy']]}|{q['development_and_test_joint']}/100|{[i+1 for i in q['no_FN_safe_rule']]}|{[i+1 for i in q['test_undefined_FNR']]}|")
    out+=['','## normal报告保留情况','', '第三种方案每轮都保留normal报告；第二种方案中出现的空报告轮次均计为失败。', '', '|条数|方案|每轮放行normal平均条数|最少—最多|门前normal预测保留率(%)|','|---|---|---:|---|---:|']
    for budget in [54,60]:
        for q in sums[budget]:
            if q['arm']!='HZ_to_SA':continue
            v=q['reported_as_normal'];out.append(f"|{budget}|{names[q['policy']]}|{v['mean']:.2f}|{int(v['min'])}—{int(v['max'])}|{fmt(q,'predicted_normal_retention')}|")
    out+=['','## 54条的17次漏判到底有没有解决','']
    q=diagnostics['54']['HZ_to_SA']
    out+=[f"现有规则在{len(q['original_FN_rounds'])}次出现FN，累计{q['original_FN_events']}个重复事件。normal检查暂缓其中{q['withheld_original_FN']}个，仍放行{q['remaining_original_FN']}个；另新增放行FN {q['new_FN_events']}个。最终仍有{len(q['remaining_FN_rounds'])}次出现FN。记录可跨重复出现，这不是独立患者数。",
        f"原FN轮次是否全部缺少开发记录206：{all(q['original_FN_missing206'])}。门前开发FN的记录分布：{q['development_FN_records']}。这里检验的是已记录关联，不把单条记录的作用宣称为普遍机制。",
        '在缺少这条高MA记录的开发集里，原映射的开发预测本来就没有FN，因此“开发零FN”不会自动要求门拒绝这类未被开发样本揭示的错误。normal专门检查减少了部分事件，但其他三个参数尚未提供足够稳定的补充区分。',
        '', '## 双源对照','', '|条数|方案|开发三项达标|测试三项达标|Accuracy(%)|Reporting rate(%)|FNR(%)|FPR(%)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        for q in sums[budget]:
            if q['arm']=='HZ+HB_to_SA':out.append('|'+ '|'.join([str(budget),names[q['policy']],str(q['development_joint_pass'])+'/100',str(q['test_joint_pass'])+'/100',*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']]])+'|')
    out+=['','## 如何读本轮结果','',
        '- 不能用统一阈值不断收紧来保证实用性：它显著降低报告率，少数轮次甚至没有满足基本硬条件的规则。',
        '- normal专门检查优于单纯收紧门的开发侧取舍，但当前方案不满足54条100次全部零FN的目标，不应作为已解决问题的结果写入论文。',
        '- 60条旧方案测试100次零FN与开发侧存在FN并不矛盾：开发是折外预测，每折训练量更少；最终测试映射用完整60条训练。新硬条件改变的是选门资格，不是推翻已记录的测试结果。',
        '- 报告率、正常报告数和真实低/高MA暂缓数均已保留。没有把暂缓解释成正确识别，也没有把空异常分母计为零FNR。',
        '- 本轮阈值搜索和风险头只用开发标签，但整个研究已多次查看这份测试集，因此这些仍是探索性结果。',
        '', '## 文件','',
        '- dev54/ROUNDS.csv、dev60/ROUNDS.csv：三方案逐次开发与测试指标，各600行（单源/双源×3方案×100次）。',
        '- dev54/SUMMARY.json、dev60/SUMMARY.json：均值、标准差、最小最大、可行数、风险头选择。',
        '- private/round_XX：风险头参数/训练编号、全部选择记录、风险分数和放行掩码。',
        '- FAILURE_ROUNDS.json：不可行、报告率不足、FN、FNR无定义的所有轮次（JSON用0起始，本文表用1起始）。',
        '- PAIRED_FN_AUDIT.json、FINAL_QA.json：FN事件配对和最终核查。',
        '- 父目录3次可选检查试运行独立保留，不混入这里的100次。',
        '- 新输出保存在D盘，因为项目E盘可用空间较少；没有删除旧实验。']
    (n.O/'REPORT_CN.md').write_text('\n'.join(out)+'\n',encoding='utf8')
    print('REPORT COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
