import json,csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import calibrate as m
from scipy.special import expit
r=m.r;f=m.f;b=m.b;a=m.a;n=m.n;O=m.O

def same(x,y):
    if isinstance(x,dict):
        assert set(x)==set(y)
        for k in x:same(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for v,w in zip(x,y):same(v,w)
    elif isinstance(x,(float,int)) and not isinstance(x,bool):assert np.isclose(x,y,atol=1e-12,rtol=1e-12,equal_nan=True),(x,y)
    else:assert x==y,(x,y)

def main():
    d,*_=r.init();allrows=[];failures=[];changes=[];checked=0
    for budget in [54,60]:
        rows=json.loads((O/f'dev{budget}/ROUNDS.json').read_text());index={(x['rep'],x['arm'],x['policy']):x for x in rows};assert len(index)==1200
        old=[x for x in json.loads((m.c.O/f'dev{budget}/ROUNDS.json').read_text()) if x['policy']=='classwise_reserve_0']
        oldix={(x['rep'],x['arm']):x for x in old}
        for rep in range(100):
            folder=O/f'dev{budget}/private/round_{rep:02d}';policies=json.loads((folder/'POLICIES.json').read_text());saved=np.load(folder/'ACTIONS_SCORES.npz')
            audit=json.loads((n.O/f'dev{budget}/private/round_{rep:02d}/RISK_TRAINING.json').read_text())['B']
            for au in audit:
                for j,h in enumerate(au['heads']):
                    assert h['positive_count']==int((r.state(d['medical'][au['train'],2])==[0,2][j]).sum())
                assert not set(d['groups'][au['train']])&set(d['groups'][au['query']])
            for arm in a.ARMS:
                z,rr,base=b.load(budget,rep,arm);dp=z['dev_prediction'];ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES};dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                np.testing.assert_array_equal(saved['test'],z['test']);np.testing.assert_array_equal(saved['dev_rows'],z['dev_rows'])
                full=audit[3]
                for j,h in enumerate(full['heads']):
                    raw=expit((d['x'][z['test'],1]-full['mean'][0])/full['scale'][0]*h['coef'][0][0]+h['intercept'][0])
                    np.testing.assert_allclose(raw,rr['test_risk_B'][:,j],rtol=1e-10,atol=1e-12)
                for mode in m.MODES:
                    dr,qr=m.make_scores(d['x'],z['dev_rows'],z['test'],rr['dev_risk_B'],rr['test_risk_B'],audit,mode)
                    np.testing.assert_array_equal(saved['dev_'+mode],dr);np.testing.assert_array_equal(saved['test_'+mode],qr)
                    pick=m.choose(dp,dy,rr['dev_risk_B'],dr,ds,base['cap_score'])
                    for upper in [False,True]:
                        name=mode+('_upper' if upper else '');p=f.gap_policy(pick,dp,ds,'upper') if upper else pick;same(r.safe(p),policies[arm+'_'+name]);row=index[rep,arm,name]
                        act=m.action(p,z['prediction'],rr['test_risk_B'],qr,qs);da=m.action(p,dp,rr['dev_risk_B'],dr,ds)
                        if act is None:assert row['test'] is None and not saved[arm+'_'+name].any()
                        else:
                            np.testing.assert_array_equal(act,saved[arm+'_'+name]);same(r.safe(n.extended_metrics(dp,dy,da)),row['development']);same(r.safe(n.extended_metrics(z['prediction'],ty,act)),row['test'])
                            assert row['development']['FN']==0 and row['development']['accuracy']>.8
                            assert p['normal']['cap_score']==base['cap_score']
                            assert not np.any(act&(r.state(z['prediction'])==1)&(rr['test_risk_B'][:,1]>base['cap_score']))
                        checked+=1
                cur=index[rep,arm,'common_B'];prev=oldix[rep,arm]
                changes.append(dict(budget=budget,round=rep+1,arm=arm,delta={k:cur['test'][k]-prev['test'][k] for k in ['reported','correct','incorrect','FN','FP','reported_as_normal','reporting_rate','accuracy','FPR']}))
            saved.close()
            if rep%25==0:print('AUDIT',budget,rep+1,flush=True)
        baseline=[dict(x,policy='baseline') for x in old];combined=baseline+rows;out=O/f'combined/dev{budget}';summary=b.h.summarize(combined)
        r.write(out/'ROUNDS.json',combined);r.write(out/'SUMMARY.json',summary)
        flat=[]
        for x in combined:
            v={k:x[k] for k in ['budget','arm','policy','feasible','development_joint_pass','joint_test_pass']};v['round']=x['rep']+1
            for side in ['development','test']:
                for key in ['reported','correct','incorrect','accuracy','reporting_rate','FN','FNR','FP','FPR','reported_as_normal','normal_released','abnormal_released','low_recall','high_recall','low_retained_correct_rate','high_retained_correct_rate','MAE','RMSE','R2']:
                    v[side+'_'+key]=None if x[side] is None else x[side][key]
            flat.append(v)
        with (out/'ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as ff:
            writer=csv.DictWriter(ff,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        for q in summary:
            rr=[x for x in combined if x['arm']==q['arm'] and x['policy']==q['policy']]
            failures.append(dict(budget=budget,arm=q['arm'],policy=q['policy'],rate=[x['rep']+1 for x in rr if x['test'] and x['test']['reporting_rate']<=.75],accuracy=[x['rep']+1 for x in rr if x['test'] and x['test']['accuracy']<=.8],FN=[x['rep']+1 for x in rr if x['test'] and x['test']['FN']>0],infeasible=[x['rep']+1 for x in rr if not x['feasible']]))
        allrows+=combined
    hashes=json.loads((O/'SOURCE_HASHES.json').read_text());assert all(r.sha(Path(k))==v for k,v in hashes.items())
    previous=json.loads((f.O/'SOURCE_HASHES.json').read_text());assert all(v==previous[k] for k,v in hashes.items() if k in previous)
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in m.protocol()['input_hashes'].items())
    r.write(O/'FAILURES.json',failures);r.write(O/'PAIRED_CHANGES.json',changes)
    r.write(O/'FINAL_QA.json',dict(new_rows_recomputed=checked,complete_100_repeats=True,developer_selection_reproduced=True,training_groups_disjoint=True,training_class_counts_verified=True,stored_risk_probabilities_reproduced=True,original10_veto_unchanged=True,original_inputs_and_MA_predictions_unchanged=True,new_test_zero_FN=sum(x['test']['FNR']==0 for x in allrows if x['policy']!='baseline')))
    report(allrows,failures,changes)
    print('FINAL QA PASS',checked,flush=True)

def report(rows,failures,changes):
    lines=['# 风险分数尺度核查与开发侧校准对照','',
      '日期：2026-11-17。论文、图表、MA预测、原始输入、两套100次抽样均未修改。所有新候选完整报告；没有逐轮切换策略或删除失败轮次。',
      '', '## 结论','',
      '存在折间风险分数尺度差异，并且统一尺度后有可重复的增量。统一幅值标尺common_B这一套固定方案使60条单源与双源在开发和测试两侧的全部100轮均达到FNR=0、accuracy>80%、reporting rate>75%。54条单源97/100、双源96/100，仍未全部达标。',
      '', '## 查到了什么','',
      '风险头是B单特征、类别均衡的Logistic模型。开发侧将三个分别拟合的折外分数汇总，测试侧使用完整开发池训练的第四个模型。不同模型的标准化均值、尺度、斜率和截距均不同，因此一个统一的分数阈值不必对应统一的幅值界限。',
      '60条第86轮：三个开发折风险模型分别使用40、39、41条，最终模型使用60条。在同一B=2.39530 V处，高MA风险分数分别为0.37720、0.29367、0.38293，最终模型为0.32305。开发predicted-normal记录的741对幅值排序中，高MA风险排序有86对发生反转，来自不同折标尺。每个单独的B风险头本身仍为单调函数。',
      '这能证明标尺不一致，不代表开发/测试风险分布的全部差异都来自校准。病例组成、MA预测差异和异常门学习仍可能影响结果。',
      '', '## 本轮改动','',
      '1. 固定10%策略：保留原始高风险分数的开发90分位阈值以及其逐条拦截，不重算/放宽该上限。',
      '2. common_B：额外normal检查改用共同的物理标尺，低风险轴为-B、高风险轴为B。只统一风险排序尺度，不修改任何B数值或MA预测，也不将该分数称作临床概率。',
      '3. train_cdf：每折用其训练B分布转为分位数；最终查询只用完整开发池分布。',
      '4. prior_probability：依据对应训练折的异常比例修正均衡类别权重造成的先验偏移；此对照表现较差，完整保留，不采用。',
      '5. 额外normal界限仍由开发零FN和最大放行数确定；低/高MA门仍由开发accuracy>80%确定。分别保留异常门区间中点和上端两个固定控制。',
      '', '## common_B：一套固定方案在两套全部100轮的结果','',
      '|开发条数|配置|Accuracy% 均值±SD|报告率% 均值±SD|FPR% 均值±SD|零FN|测试三项达标|开发/测试同时达标|最低Accuracy|最低报告率|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        summary=json.loads((O/f'combined/dev{budget}/SUMMARY.json').read_text())
        for q in summary:
            if q['policy']!='common_B':continue
            fmt=lambda k:f"{q[k]['mean']*100:.2f} ± {q[k]['sd']*100:.2f}"
            lines.append(f"|{budget}|{q['arm']}|{fmt('accuracy')}|{fmt('reporting_rate')}|{fmt('FPR')}|{q['zero_FN']}/100|{q['test_joint']}/100|{q['both_joint']}/100|{q['accuracy']['min']*100:.2f}%|{q['reporting_rate']['min']*100:.2f}%|")
    lines+=['','## 60条双源第86轮','', '|方案|放行/181|正确|错误|normal报告|Accuracy|报告率|FNR|FPR|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['baseline','common_B','train_cdf','prior_probability']:
        q=next(x for x in rows if x['budget']==60 and x['rep']==85 and x['arm']=='HZ+HB_to_SA' and x['policy']==name)['test']
        lines.append(f"|{name}|{q['reported']}/181|{q['correct']}|{q['incorrect']}|{q['reported_as_normal']}|{q['accuracy']*100:.2f}%|{q['reporting_rate']*100:.2f}%|{q['FNR']*100:.2f}%|{q['FPR']*100:.2f}%|")
    lines+=['', 'common_B比此前多放行42条，其中22条正确、20条错误。报告率70.72%→93.92%，accuracy92.97%→82.94%，FPR8.26%→20.42%。因此这是符合既定条件的取舍，不是所有指标一起提高。normal额外放行后，开发accuracy约束允许异常侧放行更多记录；两个变化共同形成报告率提升。',
      '', '## 54条仍需解决的轮次','', '|配置|方案|报告率≤75%|Accuracy≤80%|FN>0|','|---|---|---|---|---|']
    for q in failures:
        if q['budget']==54 and q['policy'] in ['common_B','common_B_upper','train_cdf','train_cdf_upper']:
            lines.append(f"|{q['arm']}|{q['policy']}|{q['rate']}|{q['accuracy']}|{q['FN']}|")
    lines+=['', 'common_B单源第28轮accuracy恰为80%，按严格>80%标准仍判失败。train_cdf单源达到98/100，但双源96/100；common_B_upper双源97/100。这里分别报告固定方案，不把各方案的成功轮次拼成一个“100轮成功”。',
      '', '剩余失败的具体原因：',
      '- 第28轮两种配置均放行175条，正确140条、错误35条，accuracy恰为80%。低MA侧9条错误、高MA侧26条错误；开发accuracy为85.11%，因而开发选择允许异常侧全部放行，但测试accuracy不满足严格大于80%的条件。',
      '- 第38轮双源放行163条，其中33条错误，accuracy79.75%。开发accuracy80.85%本身就接近边界；高MA门的开发阈值在测试侧未拦住高MA预测。',
      '- 第76轮两种配置都放行134条，报告率74.03%。主要剩余损失来自高MA的Logistic门：单源暂缓35条，其中12条正确、23条错误；normal另外有8条被原10%上限拦截、4条被额外检查拦截。不能只放宽某轮阈值来补两条。',
      '- 第89轮仍同时受到normal附加检查和高MA门约束：单源额外暂缓25条normal预测，高MA门暂缓26条（9条正确）；双源对应23条与22条（8条正确）。风险标尺统一后仍有跨样本泛化差异。',
      '', '这些是评价完成后的诊断，不用于回头修改该轮规则。下一步应重点核对高MA错误门的折外训练分数与最终分数，并在开发侧考虑accuracy余量；不能保证继续调一个阈值即可同时修复所有轮次。',
      '', '## 全部对照','', '|条数|配置|方案|Accuracy均值|报告率均值|FPR均值|零FN|三项达标|','|---|---|---|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        for q in json.loads((O/f'combined/dev{budget}/SUMMARY.json').read_text()):
            lines.append(f"|{budget}|{q['arm']}|{q['policy']}|{q['accuracy']['mean']*100:.2f}%|{q['reporting_rate']['mean']*100:.2f}%|{q['FPR']['mean']*100:.2f}%|{q['zero_FN']}/100|{q['test_joint']}/100|")
    lines+=['','## 统计口径与证据范围','',
      'Accuracy分母为已放行数；报告率分母为每轮181条测试记录；FNR分母为已放行真实异常，分子为其中被判为normal者；FPR分母为已放行真实normal，分子为其中被判为异常者。暂缓不算识别正确，无异常放行不视为零FNR达标。',
      '0 FN是当前两套重复评价的观察结果，不是未来病例绝对零漏判保证。均值±SD来自重叠抽样，不是独立验证置信区间。现有队列已反复用于研究诊断；尽管本轮阈值/校准只由开发侧确定，整个研究仍有适应性偏倚。',
      '', '## 文件','',
      '- combined/dev54/ROUNDS.csv与combined/dev60/ROUNDS.csv：各1400行，基线+6种固定对照、2来源配置、100轮；开发和测试分列。',
      '- DIAGNOSTIC.json：每轮同幅值分数差、训练量、排序反转等诊断。',
      '- FAILURES.json与PAIRED_CHANGES.json：失败轮次与逐轮正确/错误增量；RESIDUAL_DIAGNOSTIC.json：common_B剩余失败的开发/测试阻拦分解。',
      '- private/round_*/：保存风险分数、选出的门和完整放行动作。',
      '- PROTOCOL.json、SMOKE.json、FINAL_QA.json：对照规则、测试及复算证据。']
    (O/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
