"""Audit all saved follow-up results and document the observed failure chain."""
import support_followup as s
import run as r
import fn_candidates as f
import numpy as np,json,csv
from threadpoolctl import threadpool_limits

def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    d,plan,role,m=r.init();items=read(r.OUT/'private/SPLITS.json');n_arrays=0;n_policies=0
    for budget in [54,60]:
        for item in items:
            rep=item['rep'];dest=s.O/f'dev{budget}/private'/f'round_{rep:02d}';assert (dest/'COMPLETE.json').exists()
            sp=s.spec(item,budget);fixed=read(dest/'FIXED.json');candidates=read(dest/'CANDIDATE.json');selected=read(dest/'SELECTED.json')
            assert len(fixed)==96 and len(candidates)==18 and len(selected)==6
            for arm in s.ARMS:
                reference=(r.OUT/'private'/f'round_{rep:02d}'/(arm+'_reporting.npz')) if budget==54 else r.OUT/'full60_control/private'/f'round_{rep:02d}'/(arm+'.npz')
                old=np.load(reference);results=[]
                for mode in s.MODES:
                    z=np.load(dest/(arm+'_'+mode+'.npz'));n_arrays+=1
                    np.testing.assert_array_equal(z['test'],sp['test']);assert set(z['dev_rows'])==set(sp['pool'])
                    assert np.isfinite(z['prediction']).all() and np.isfinite(z['dev_prediction']).all()
                    x=dict(pred=z['prediction'],dp=z['dev_prediction'],dy=d['medical'][z['dev_rows'],2],scores={k:z['score_'+k] for k in f.RULES},ds={k:z['dev_score_'+k] for k in f.RULES},guards={k:z['guard_'+k] for k in s.GUARDS},dg={k:z['dev_guard_'+k] for k in s.GUARDS})
                    saved=read(dest/(arm+'_'+mode+'_selection.json'));picks=saved['picks'];results.append((x,picks))
                    for k in s.GUARDS:np.testing.assert_array_equal(x['guards'][k],s.guard_masks(d,sp['pool'],sp['test'],z['prediction'])[k])
                    if mode=='baseline':
                        np.testing.assert_allclose(z['prediction'],old['prediction'],atol=1e-9,rtol=0)
                        for rule in f.RULES:np.testing.assert_allclose(z['score_'+rule],old['score_'+rule],atol=1e-9,rtol=0)
                    for rule in f.RULES:
                        assert picks[rule]==s.choose(x,rule);n_policies+=1
                        t=saved['original']['thresholds'][rule].get('threshold');act=np.zeros(len(z['test']),bool) if t is None else np.isfinite(z['score_'+rule])&(z['score_'+rule]<=t)
                        for guard in s.GUARDS:
                            v=next(q for q in fixed if q['arm']==arm and q['mode']==mode and q['rule']==rule and q['guard']==guard)
                            assert v['test']==r.safe(r.metrics(z['prediction'],d['medical'][z['test'],2],act&x['guards'][guard]))
                for rule in f.RULES:
                    valid=[i for i,(_,p) in enumerate(results) if p[rule]['feasible']];chosen=next(q for q in selected if q['arm']==arm and q['rule']==rule)
                    if not valid:assert chosen['test'] is None and chosen['mode'] is None
                    else:
                        i=max(valid,key=lambda i:(results[i][1][rule]['reported'],results[i][1][rule]['accuracy'],-i));x,picks=results[i]
                        assert chosen['mode']==s.MODES[i]
                        assert chosen['test']==r.safe(r.metrics(x['pred'],d['medical'][sp['test'],2],s.action(x,rule,picks[rule])))
        ops=read(s.O/f'dev{budget}/OPERATING_ROUNDS.json');assert len(ops)==3600
        for q in ops:
            dest=s.O/f'dev{budget}/private'/f"round_{q['rep']:02d}";z=np.load(dest/(q['arm']+'_'+q['mode']+'.npz'));rule=q['rule']
            error=.15 if q['error_budget']==.15 else .2-1e-12
            pick=r.gates.rc.select_threshold(z['dev_score_'+rule],r.gates.fr.ma_class(z['dev_prediction']),r.gates.fr.ma_class(d['medical'][z['dev_rows'],2]),error)
            assert r.safe(pick)==q['development'];t=pick.get('threshold')
            act=np.zeros(len(z['test']),bool) if t is None else np.isfinite(z['score_'+rule])&(z['score_'+rule]<=t)
            assert q['test']==r.safe(r.metrics(z['prediction'],d['medical'][z['test'],2],act))
    inputs=read(r.OUT/'PROTOCOL.json')['input_hashes'];assert all(r.sha(r.current.ROOT/'data'/name)==h for name,h in inputs.items())
    r.write(s.O/'FINAL_QA.json',dict(two_budgets_each100=True,prediction_arrays_checked=n_arrays,development_selections_recomputed=n_policies,baseline_predictions_scores_reproduced=True,test_indices_unchanged=True,metrics_and_denominators_recomputed=True,operating_thresholds_recomputed_from_development_only=True,input_hashes_unchanged=True))
    # Failure decomposition is a diagnostic, never a candidate selection rule.
    item=items[58];f.draw(item,plan);f.configure('residual',4);sp=s.spec(item,60);a=np.array(sp['pool']);b=np.array(sp['test'])
    z=np.load(r.OUT/'full60_control/private/round_58/HZ_to_SA.npz');truth=d['medical'][b,2]
    bad=np.flatnonzero((truth<50)&(z['prediction']>=50)&(z['prediction']<=70))
    src=r.c.source_model('HZ_to_SA',900);prior=r.c.source_predict(src,d['x'][b,1:2]);records=[]
    for j in bad:
        row=int(b[j]);v=dict(index=row,record=str(d['ids'][row]),reference_MA=float(truth[j]),UTEG_MA=float(d['x'][row,1]),source_prediction=float(prior[j]),baseline_prediction=float(z['prediction'][j]),local_correction=float(z['prediction'][j]-prior[j]),HGB_score=float(z['score_HGB'][j]))
        for mode in s.MODES[1:]:v[mode]=float(np.load(s.O/f'dev60/private/round_58/HZ_to_SA_{mode}.npz')['prediction'][j])
        records.append(v)
    info=read(r.OUT/'full60_control/private/round_58/HZ_to_SA_selection.json')['original']['thresholds']['HGB']
    r.c.source_rows=lambda arm,h:np.asarray(plan['source_rows'].get(arm,[]),int);r.c._SOURCE.clear();full=s.original_predict('HZ_to_SA',900,a,b)[0]
    for j,v in zip(bad,records):v['diagnostic_full_source_prediction']=float(full[j])
    diagnosis=dict(rep=58,human_repeat=59,records=records,local_B_range=[float(d['x'][a,1].min()),float(d['x'][a,1].max())],development_low_records=[dict(record=str(d['ids'][i]),MA=float(d['medical'][i,2]),B=float(d['x'][i,1])) for i in a if d['medical'][i,2]<50],HGB_threshold=info['threshold'],HGB_development_released_FN=info['abnormal_false_normal'],cause='Source predictions were low. Local correction outside local feature support moved them to normal. HGB scores were below the original overall-error threshold. Full-source counterfactual retained all60 local records and is diagnostic only.')
    r.write(s.O/'private/FAILURE_DIAGNOSIS.json',diagnosis)
    report(diagnosis)
    print('SUPPORT QA PASS',n_arrays,n_policies,flush=True)

def report(diag):
    lines=['# 单源低MA漏判：原因与100次开发侧调整','',
    '论文、图表、输入数据及原100次结果未改。两套设置继续分别保存：54条开发记录；完整60条开发记录。来源与测试抽样名单相同。所有候选均保留，不按测试结果删除重复。',
    '', '## 1. 原因已定位','',
    '问题发生在第59次重复（rep=58）。两条记录的U-TEG幅值为1.87167、1.85310，均低于本地开发范围2.02214–2.91299。4条本地低MA开发记录的幅值集中在2.02806–2.05732；本地适应在更低幅值区域缺少直接约束。',
    '', '| 记录 | 参考MA | 外院模型预测 | 本地适应增加 | 原最终预测 | 限制残差外推后 | 端点约束后 |','|---|---:|---:|---:|---:|---:|---:|']
    for v in diag['records']:lines.append('| '+' | '.join([v['record'],*[f'{v[k]:.3f}' for k in ['reference_MA','source_prediction','local_correction','baseline_prediction','residual_clamp','anchored_tails']]])+' |')
    lines+=['','HGB对两条记录的风险分数约0.071和0.152，低于该轮阈值0.344，因此两条均被放行。门前映射先出错，门没有拦住；并非门把正确类别改错。原阈值约束开发侧总体三分类错误率≤15%，不是开发侧FN=0；该轮开发侧也放行了1条FN。',
    '保持60条本地开发记录不变、诊断性恢复全部来源训练记录后，两条最终预测约48.12和48.71 mm。说明来源抽样会改变适应后的尾部外推；这项诊断未用于指定保留某条来源记录。',
    '', '## 2. 两种针对性修正','',
    '- residual_clamp：外院预测仍使用当前实际幅值；本地残差校正超出训练幅值范围时，使用最近端点的校正量。训练范围内不变。',
    '- anchored_tails：低于训练下界时，最终预测不得高于下界预测；高于训练上界时，最终预测不得低于上界预测。原本未违反该条件的预测保持不变。',
    '- 另测试仅在训练支持范围、或训练侧低/高MA边界之间放行normal的规则；0、0.1、0.2倍训练IQR的缓冲均保留。范围和缓冲不使用测试标签。',
    '', '## 3. HGB固定候选结果','',
    '下面展示开发侧accuracy>80%的阈值设置。该设置只改开发侧总体错误预算，不等于开发侧FN被硬约束为零。原15%错误预算结果也完整保存。FNR分母为实际放行的真实异常数，空分母记NA；accuracy分母为已放行记录，reporting rate分母为每轮181条记录。',
    '', '| 开发数 | 流程 | 映射设置 | Accuracy % | Reporting rate % | FNR % | FPR % | 零FNR/100 | NA | 三项同时达标/100 |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    names={'HZ_to_SA':'单源','HZ+HB_to_SA':'双源'}
    def fmt(v,k):
        x=v.get(k)
        return 'NA' if not x or x['mean'] is None else f"{100*x['mean']:.2f} ± {100*x['sd']:.2f}"
    for budget in [54,60]:
        for q in read(s.O/f'dev{budget}/OPERATING_SUMMARY.json'):
            if q['rule']!='HGB' or q['error_budget']!=.2:continue
            lines.append('| '+' | '.join([str(budget),names[q['arm']],q['mode'],*[fmt(q,k) for k in ['accuracy','reporting_rate','FNR','FPR']],str(q['zero_FN_rounds']),str(q['FNR_undefined_rounds']),str(q['joint_test_pass'])])+' |')
    lines+=['', '完整60条设置：两个外推修正候选的单源HGB均在100次中得到有定义的FNR=0，且100次accuracy均>80%；98次同时保住reporting rate>75%。因此已解决这一轮低MA漏判，但尚未实现每次重复的三项联合要求。',
    '54条设置：单源两种修正仍有17次FN，另1次未报告任何记录；端点约束下双源仍有16次FN、1次未报告任何记录。剩余FN均来自高MA；残差裁剪对双源54条还新增了部分高MA漏判，不建议全流程统一替换。',
    '单源这17次高MA漏判恰好对应同一条高MA开发记录未被抽中的17次。该记录参考MA约72.17 mm、U-TEG幅值约2.467 V，位于较多normal记录的幅值区域。这是重复抽样中的关联诊断，不能据此强制保留该记录；它提示54条方案的高MA问题涉及一维幅值信息重叠和训练覆盖，区别于已修复的低MA外推。',
    '', '## 4. 严格开发侧FN=0选型的结果','',
    '另一个独立选型器要求开发侧FN=0、accuracy>80%、有异常放行，再在三种映射和五种支持条件间最大化报告率。该选择器没有稳定达到测试侧联合目标；不可将固定候选取得的测试成绩说成这个严格选择器的成绩。',
    '', '| 开发数 | 流程 | 规则 | 开发无可行设置 | 测试零FNR | 测试NA | 测试联合达标 |','|---|---|---|---:|---:|---:|---:|']
    for budget in [54,60]:
        for q in read(s.O/f'dev{budget}/SELECTED_SUMMARY.json'):
            lines.append('| '+' | '.join([str(budget),names[q['arm']],q['rule'],*[str(q[k]) for k in ['development_infeasible_rounds','zero_FN_rounds','FNR_undefined_rounds','joint_test_pass']]])+' |')
    lines+=['', '## 5. 对下一步的判断','',
    '完整60条的低MA问题有明确的外推原因和针对性修正，适合继续验证；54条剩余高MA问题不能用同一修复冒充解决。54条涉及开发集中少量高MA记录的覆盖及一维幅值重叠，下一步应检查辅助参数是否提供独立信息，或在这些区域明确暂缓。继续一味加大惩罚缺少当前证据支持。',
    '这仍是同一历史数据上的探索性重复抽样，未得到新的独立验证队列；100次零FN描述本协议观察结果，不是未来病例必然零漏判。',
    '', '## 文件','',
    '- `dev54/`和`dev60/`：各100次的所有预测、阈值、可行性、完整指标。',
    '- `FIXED_*`：固定映射与支持条件；`CANDIDATE_*`及`SELECTED_*`：严格开发侧零FN选型。',
    '- `OPERATING_*`：开发侧15%和严格低于20%错误预算。',
    '- `private/FAILURE_DIAGNOSIS.json`：失败记录及来源/适应分解。',
    '- `SMOKE.json`、`FINAL_QA.json`：基线复现、测试标签扰动不变、逐轮分母和开发选型复核。']
    (s.O/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Export compact summaries without any record-level identifiers.
    for budget in [54,60]:
        rr=read(s.O/f'dev{budget}/OPERATING_SUMMARY.json');rows=[]
        for q in rr:
            v={k:q[k] for k in ['arm','mode','rule','error_budget','rounds','zero_FN_rounds','FNR_undefined_rounds','joint_test_pass']}
            for k in ['accuracy','reporting_rate','FNR','FPR','MAE','R2']:
                for stat in ['mean','sd','min','max','n']:v[k+'_'+stat]=q[k][stat] if q[k] else None
            rows.append(v)
        with (s.O/f'dev{budget}'/'OPERATING_SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as h:
            w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
