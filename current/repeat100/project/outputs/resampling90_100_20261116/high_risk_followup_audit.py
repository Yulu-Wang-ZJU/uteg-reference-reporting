"""Audit all stored high-risk controls and paired FN/reporting-rate outcomes."""
import json,collections
import numpy as np
from threadpoolctl import threadpool_limits
import high_risk_followup as h
import normal_hard_fn as n
import aux_followup as a
import support_followup as s
import run as r

def main():
    d,*_=r.init();y=d['medical'][:,2];items=json.loads((r.OUT/'private/SPLITS.json').read_text())
    compared=0;heads=0;failures=[];paired=[];source_hashes={}
    for budget in [54,60]:
        selected=json.loads((h.O/f'dev{budget}/SELECTED_ROUNDS.json').read_text());fixed=json.loads((h.O/f'dev{budget}/FIXED_ROUNDS.json').read_text())
        assert len(selected)==1000 and len(fixed)==2400
        assert sorted(set(v['rep'] for v in selected))==list(range(100))
        tally=collections.defaultdict(lambda:dict(old_FN=0,blocked_old_FN=0,remaining_old_FN=0,new_FN=0,old_FN_rounds=[],remaining_FN_rounds=[],FN_records=collections.Counter(),reporting_rate_change=[],normal_reports_change=[],newly_withheld_correct_normal=0,newly_withheld_correct_abnormal=0))
        for rep in range(100):
            folder=h.O/f'dev{budget}/private/round_{rep:02d}';assert (folder/'COMPLETE.json').exists();sp=s.spec(items[rep],budget)
            store=np.load(folder/'RISK_ACTIONS.npz');old=np.load(n.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS_RISKS.npz')
            np.testing.assert_array_equal(store['test'],sp['test']);assert set(store['dev_rows'])==set(sp['pool']) and len(store['dev_rows'])==budget
            lookup={int(ix):i for i,ix in enumerate(store['dev_rows'])};dev=np.full((budget,2),np.nan)
            audits=json.loads((folder/'RBF_TRAINING.json').read_text())
            for pos,au in enumerate(audits):
                train=np.array(au['train']);query=np.array(au['query']);mean=np.array(au['mean']);scale=np.array(au['scale'])
                assert set(train)==set(sp['positions'][pos]['A']) and set(query)==set(sp['positions'][pos]['B'])
                assert not set(d['groups'][train])&set(d['groups'][query])
                np.testing.assert_allclose(mean,d['x'][train].mean(0),rtol=1e-12,atol=1e-12)
                sd=d['x'][train].std(0);sd[sd==0]=1;np.testing.assert_allclose(sd,scale,rtol=1e-12,atol=1e-12)
                xx=(d['x'][query]-mean)/scale;scores=[]
                for head in au['heads']:
                    assert head['positive_count']==int(np.sum(r.state(y[train])==head['class_index']))
                    if head['status']=='fit':
                        support=np.array(head['support_vectors']);assert set(head['support_rows'])<=set(train)
                        np.testing.assert_allclose(support,(d['x'][head['support_rows']]-mean)/scale,rtol=1e-12,atol=1e-12)
                        squared=np.maximum(np.sum(xx*xx,axis=1)[:,None]+np.sum(support*support,axis=1)[None,:]-2*xx@support.T,0)
                        pred=np.exp(-head['gamma']*squared)@np.array(head['dual_coef'])[0]+head['intercept'][0]
                    else:pred=np.ones(len(query))
                    scores.append(pred);heads+=1
                scores=np.array(scores).T
                if pos<3:dev[[lookup[int(ix)] for ix in query]]=scores
                else:np.testing.assert_allclose(scores,store['test_rbf'],atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(dev,store['dev_rbf'],atol=1e-9,rtol=1e-9)
            risks={key:(old['dev_risk_'+key],old['test_risk_'+key]) for key in ['B','four']};risks['four_rbf']=(store['dev_rbf'],store['test_rbf'])
            for arm in a.ARMS:
                path=a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz';z=np.load(path);source_hashes[str(path)]=r.sha(path)
                sel=json.loads((folder/(arm+'_SELECTION.json')).read_text());candidates=sel['candidates'];chosen=sel['chosen']
                # Recompute the full threshold search on the smoke fold and one additional fold.
                if rep in [0,99]:
                    cand2,chosen2=h.decisions(z,y[z['dev_rows']],risks);assert r.safe(cand2)==candidates and r.safe(chosen2)==chosen
                expected={'uncapped_with_RBF':h.select([p for p in candidates if p['cap_quantile'] is None]),'development_selected_cap':h.select([p for p in candidates if p['cap_quantile'] is not None])}
                for cap in h.CAPS[1:]:expected['cap_'+str(cap)]=h.select([p for p in candidates if p['cap_quantile']==cap])
                assert r.safe(expected)==chosen
                ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
                records={v['policy']:v for v in selected+fixed if v['rep']==rep and v['arm']==arm}
                allpicks=[(p['risk_model']+'_cap_'+str(p['cap_quantile']),p) for p in candidates]+list(chosen.items())
                for name,pick in allpicks:
                    row=records[name];da=n.action_for(pick,z['dev_prediction'],ds,{key:v[0] for key,v in risks.items()});ta=n.action_for(pick,z['prediction'],qs,{key:v[1] for key,v in risks.items()})
                    if da is None:assert row['test'] is None and row['development'] is None
                    else:
                        assert r.safe(n.extended_metrics(z['dev_prediction'],y[z['dev_rows']],da))==row['development']
                        assert r.safe(n.extended_metrics(z['prediction'],y[z['test']],ta))==row['test']
                        assert row['development']['FN']==0 and row['development']['accuracy']>.8 and row['development']['reported_as_normal']>0 and row['development']['abnormal_released']>0
                        if pick['cap_quantile'] is not None:
                            norm=r.state(z['dev_prediction'])==1;dr=risks[pick['risk_model']][0]
                            cap=float(np.quantile(dr[norm,1],pick['cap_quantile'],method='linear'));assert cap==pick['cap_score']
                            assert not np.any(da&norm&(dr[:,1]>cap))
                    compared+=1
                    if name not in chosen and name!='B_cap_0.9':continue
                    if name in chosen:
                        if ta is not None:np.testing.assert_array_equal(store[arm+'_'+name],ta)
                        else:assert not store[arm+'_'+name].any()
                    # Infeasible action is stored empty for alignment only, not counted as success.
                    ta=np.zeros(len(z['test']),bool) if ta is None else ta;before=old[arm+'_normal_check_action'];truth=r.state(y[z['test']]);pc=r.state(z['prediction'])
                    original=before&(pc==1)&(truth!=1);after=ta&(pc==1)&(truth!=1);q=tally[(arm,name)]
                    q['old_FN']+=int(original.sum());q['blocked_old_FN']+=int(np.sum(original&~ta));q['remaining_old_FN']+=int(np.sum(original&ta));q['new_FN']+=int(np.sum(after&~original))
                    if original.any():q['old_FN_rounds'].append(rep)
                    if after.any():q['remaining_FN_rounds'].append(rep)
                    for ix in z['test'][after]:q['FN_records'][str(ix)]+=1
                    if row['test'] is not None:
                        q['reporting_rate_change'].append(float(ta.mean()-before.mean()));q['normal_reports_change'].append(int(np.sum(ta&(pc==1))-np.sum(before&(pc==1))))
                        q['newly_withheld_correct_normal']+=int(np.sum(before&~ta&(pc==truth)&(truth==1)))
                        q['newly_withheld_correct_abnormal']+=int(np.sum(before&~ta&(pc==truth)&(truth!=1)))
        for (arm,policy),q in tally.items():
            q['FN_records']=dict(q['FN_records']);paired.append(dict(budget=budget,arm=arm,policy=policy,**q))
        reported=selected+[v for v in fixed if v['policy']=='B_cap_0.9']
        for arm in a.ARMS:
            for policy in dict.fromkeys(v['policy'] for v in reported):
                rr=[v for v in reported if v['arm']==arm and v['policy']==policy]
                failures.append(dict(budget=budget,arm=arm,policy=policy,infeasible=[v['rep'] for v in rr if not v['feasible']],developer_joint_fail=[v['rep'] for v in rr if not v['development_joint_pass']],test_FN=[v['rep'] for v in rr if v['test'] and v['test']['FN']>0],test_rate_fail=[v['rep'] for v in rr if v['test'] and v['test']['reporting_rate']<=.75],test_accuracy_fail=[v['rep'] for v in rr if v['test'] and (v['test']['accuracy'] is None or v['test']['accuracy']<=.8)],test_no_normal=[v['rep'] for v in rr if v['test'] and v['test']['no_normal_reports']],test_undefined_FNR=[v['rep'] for v in rr if v['test'] and v['test']['FNR'] is None]))
        print('HIGH RISK AUDIT',budget,'PASS',flush=True)
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in h.protocol()['input_hashes'].items())
    r.write(h.O/'SOURCE_PREDICTION_HASHES.json',source_hashes);r.write(h.O/'PAIRED_FN_REPORTING.json',paired);r.write(h.O/'FAILURE_ROUNDS.json',failures)
    r.write(h.O/'FINAL_QA.json',dict(outcomes_recomputed=compared,RBF_heads_reconstructed=heads,complete_repeats_per_budget=100,development_hard_FN_zero_checked=True,group_isolation_checked=True,normal_caps_from_development_only=True,selection_from_development_only=True,no_empty_normal_or_abnormal_success=True,original_MA_predictions_unchanged=True,input_hashes_unchanged=True))
    report(paired,failures);print('HIGH RISK FINAL PASS',compared,heads,flush=True)

def report(paired,failures):
    labels={'uncapped_with_RBF':'增加四参数RBF风险候选，不限制风险尾部','cap_0.95':'开发normal风险最高5%暂缓','cap_0.9':'开发normal风险最高10%暂缓','cap_0.8':'开发normal风险最高20%暂缓','development_selected_cap':'开发侧在上述三个比例中选型','B_cap_0.9':'固定幅值风险＋最高10%暂缓（预设对照）'}
    text=['# 剩余高MA FN的原因与有限跟进对照','',
        '论文、图表、原始数据、MA映射和三分类边界均不变。继续使用同一批100次划分；54与60条分开，单源和双源分别报告。新实验只训练风险头并选择放行规则。',
        '', '## 44个FN是什么','',
        '44是重复事件数，来自4条记录。它们真实MA略高于70mm，但映射值落在normal上沿。下表只作错误诊断，不用于按记录指定阈值。',
        '', '|内部索引|重复事件|参考MA(mm)|预测MA范围(mm)|B风险在开发normal中的平均百分位|四参数近邻为normal的比例|','|---|---:|---:|---|---:|---:|']
    diag=json.loads((h.O/'diagnostic/FN_RECORD_SUMMARY.json').read_text())
    for q in diag:
        if q['arm']=='HZ_to_SA':text.append(f"|{q['index']}|{q['repeated_FN_events']}|{q['true_MA']:.2f}|{q['prediction_range'][0]:.2f}—{q['prediction_range'][1]:.2f}|{100*q['normal_percentile_B']:.1f}%|{100*q['nearest3_normal_fraction']:.1f}%|")
    text+=['', '原检查实际选中了幅值风险模型。它能把这些记录排到normal风险较高的位置，但旧选择目标在开发FN=0时继续最大化报告数，阈值仍允许部分高风险normal通过。四参数最近邻和线性风险分数没有直接把这4条记录与normal清楚分开。这是当前数据的描述性结果，不能解释成确定生理机制。',
        '', '## 本轮固定方案','',
        '- 保留幅值/四参数Logistic风险头；另加四参数RBF SVC风险头（C=1,gamma=0.25），专门检查非线性信息。全部训练、标准化、类别权重只用相应训练折。',
        '- 预先固定5%、10%、20%三个开发normal高风险尾部暂缓比例。阈值取开发折外分数分位数，不使用测试分数分布或标签。测试中实际暂缓比例可能不同。',
        '- 每个固定比例内，仅用开发侧选择风险模型及HGB/Logistic/Boundary门；要求开发FN=0、accuracy>80%、至少一条normal报告和真实异常报告，再最大化报告数。另保留一个开发侧从三个比例中选型的完整策略。',
        '- 所有固定比例都报告，不根据测试效果挑一个比例称为开发侧最优。没有符合条件的规则记NA；报告率不足75%的候选保留并标为未达联合目标。',
        '', '## 单源结果','', '均值±SD按指标有定义的轮次计算；完整分母及失败轮次在JSON/CSV中。三项：accuracy>80%、reporting rate>75%、放行异常中的FNR=0。',
        '', '|开发条数|方案|Accuracy(%)|Reporting rate(%)|FNR(%)|FPR(%)|零FN轮数|测试三项达标|开发/测试同时达标|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(q,k):
        v=q[k]
        return 'NA' if v is None else f"{100*v['mean']:.2f} ± {100*v['sd']:.2f}" if v['sd'] is not None else f"{100*v['mean']:.2f}"
    summaries={b:json.loads((h.O/f'dev{b}/SELECTED_SUMMARY.json').read_text()) for b in [54,60]}
    for budget in [54,60]:
        fixed=json.loads((h.O/f'dev{budget}/FIXED_SUMMARY.json').read_text())
        summaries[budget]+=[v for v in fixed if v['policy']=='B_cap_0.9']
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['arm']=='HZ_to_SA':text.append('|'+ '|'.join([str(budget),labels[q['policy']],*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']],f"{q['zero_FN']}/100",f"{q['test_joint']}/100",f"{q['both_joint']}/100"])+'|')
    text+=['','## 44个旧FN拦住多少，代价是什么','', '|方案|拦住旧FN|仍放行旧FN|新增FN|报告率平均变化(百分点)|正常报告平均变化(条)|','|---|---:|---:|---:|---:|---:|']
    for q in paired:
        if q['budget']==54 and q['arm']=='HZ_to_SA':text.append(f"|{labels[q['policy']]}|{q['blocked_old_FN']}|{q['remaining_old_FN']}|{q['new_FN']}|{100*np.mean(q['reporting_rate_change']):.2f}|{np.mean(q['normal_reports_change']):.2f}|")
    text+=['','相对于上一轮normal检查，不把暂缓的异常当作已正确分类；上表“新增FN”指此前未放行、本轮被放行的FN。MA映射本身没有改变。',
        '', '## 报告率底线与normal报告','', '|条数|方案|最低报告率(%)|报告率≤75%的次数|每轮normal报告范围|无可行规则次数|','|---|---|---:|---:|---|---:|']
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['arm']!='HZ_to_SA':continue
            fail=next(v for v in failures if v['budget']==budget and v['arm']==q['arm'] and v['policy']==q['policy']);v=q['reported_as_normal']
            text.append(f"|{budget}|{labels[q['policy']]}|{100*q['reporting_rate']['min']:.2f}|{len(fail['test_rate_fail'])}|{int(v['min'])}—{int(v['max'])}|{q['infeasible']}|")
    text+=['','## 双源匹配对照','', '|条数|方案|Accuracy(%)|Reporting rate(%)|FNR(%)|FPR(%)|零FN轮数|测试三项达标|','|---|---|---:|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        for q in summaries[budget]:
            if q['arm']=='HZ+HB_to_SA':text.append('|'+ '|'.join([str(budget),labels[q['policy']],*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']],f"{q['zero_FN']}/100",f"{q['test_joint']}/100"])+'|')
    text+=['','## 解释边界','',
        '固定10%或20%风险尾部暂缓的结果，不能替代开发侧自动选型策略的结果。若自动选型偏向更宽松的5%，其FN也须完整报告。任何固定比例在本测试集的较好成绩，仍需冻结后独立验证，不能视为未来零FN保证。',
        '', '## 文件','',
        '- dev54/、dev60/：各100次，SELECTED_ROUNDS.csv为五种选择策略，FIXED_ROUNDS.csv为三种风险模型×四种cap的完整对照。',
        '- diagnostic/：4条记录的错误范围、风险分数与开发近邻诊断。',
        '- PAIRED_FN_REPORTING.json：FN事件与报告率配对变化。',
        '- FAILURE_ROUNDS.json：逐策略的不可行、FN、accuracy、报告率失败轮次。',
        '- FINAL_QA.json：重算指标、分组隔离、RBF得分重建及原输入哈希检查。',
        '- 全部属于历史队列上的探索性跟进，未改论文和既有结果。']
    (h.O/'REPORT_CN.md').write_text('\n'.join(text)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
