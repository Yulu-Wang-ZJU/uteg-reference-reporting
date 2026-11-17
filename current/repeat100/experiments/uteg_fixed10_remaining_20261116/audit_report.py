import json,csv
import numpy as np
from threadpoolctl import threadpool_limits
import run_followup as f
import stratified_normal as s
r=f.r;O=f.O

def same(a,b):
    if a is None or b is None:assert a is b;return
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True),(a,b)
    else:assert a==b,(a,b)

def main():
    d,*_=r.init();allrows=[];checks=0;diagnostics=[]
    for budget in [54,60]:
        rows=json.loads((O/f'dev{budget}/ROUNDS.json').read_text())+json.loads((s.O/f'dev{budget}/ROUNDS.json').read_text())
        old={(x['rep'],x['arm']):x for x in json.loads((f.c.O/f'dev{budget}/ROUNDS.json').read_text()) if x['policy']=='classwise_reserve_0'}
        idx={(x['rep'],x['arm'],x['policy']):x for x in rows}
        assert len(rows)==1600 and len(idx)==1600
        for rep in range(100):
            ps=json.loads((O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text());sp=json.loads((s.O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text())
            acts=np.load(O/f'dev{budget}/private/round_{rep:02d}/ACTIONS.npz');sa=np.load(s.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS.npz')
            for arm in f.a.ARMS:
                z,risk,base=f.b.load(budget,rep,arm);dp=z['dev_prediction'];ds={k:z['dev_score_'+k] for k in f.a.RULES};qs={k:z['score_'+k] for k in f.a.RULES};dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                parent=ps[arm+'_baseline'];planned=f.plans(parent,dp,dy,risk['dev_risk_B'],ds)
                for name in f.NAMES+['normal_midpoint','normal_midpoint_upper']:
                    row=idx[rep,arm,name];key=arm+'_'+name
                    if name.startswith('normal_midpoint'):
                        p=s.select_policy(dp,dy,risk['dev_risk_B'],ds,base['cap_score'],name.endswith('_upper'));same(r.safe(p),sp[key])
                        da=s.action(p,dp,risk['dev_risk_B'],ds);act=s.action(p,z['prediction'],risk['test_risk_B'],qs);stored=sa[key]
                    else:
                        p=planned[name];same(r.safe(p),ps[key]);da=f.actions(name,p,parent,dp,ds,dp,risk['dev_risk_B'],ds);act=f.actions(name,p,parent,dp,ds,z['prediction'],risk['test_risk_B'],qs);stored=acts[key]
                    if act is None:assert not stored.any();assert row['test'] is None
                    else:
                        np.testing.assert_array_equal(act,stored)
                        same(r.safe(f.n.extended_metrics(dp,dy,da)),row['development']);same(r.safe(f.n.extended_metrics(z['prediction'],ty,act)),row['test'])
                        assert row['development']['FN']==0
                    if name=='baseline':same(row['test'],old[rep,arm]['test']);same(row['development'],old[rep,arm]['development'])
                    checks+=1
                if not idx[rep,arm,'baseline']['joint_test_pass']:
                    sections={}
                    for label,pred,rr,ss,truth in [('development',dp,risk['dev_risk_B'],ds,dy),('test',z['prediction'],risk['test_risk_B'],qs,ty)]:
                        pc=r.state(pred);tc=r.state(truth);act=f.c.apply(parent,pred,rr,ss);normal=f.b.normal_action(parent['normal'],pred,rr,ss);cap=rr[:,1]<=parent['normal']['cap_score'];extra=(pc==1)&cap&~normal;ab=(pc!=1);blocked=ab&~act
                        sections[label]=dict(normal_released=int(normal.sum()),cap_blocked=int(((pc==1)&~cap).sum()),additional_normal_blocked=int(extra.sum()),additional_normal_blocked_truth_counts=np.bincount(tc[extra],minlength=3).tolist(),abnormal_blocked=int(blocked.sum()),abnormal_blocked_correct=int((blocked&(pc==tc)).sum()),abnormal_blocked_incorrect=int((blocked&(pc!=tc)).sum()))
                    diagnostics.append(dict(budget=budget,round=rep+1,arm=arm,sections=sections))
            acts.close();sa.close()
        out=O/f'combined/dev{budget}';r.write(out/'SUMMARY.json',f.b.h.summarize(rows));r.write(out/'ROUNDS.json',rows)
        flat=[]
        for row in rows:
            v={k:row[k] for k in ['budget','arm','policy','feasible','development_joint_pass','joint_test_pass']};v['round']=row['rep']+1
            for part in ['development','test']:
                for key in ['reported','correct','incorrect','accuracy','reporting_rate','FN','FNR','FP','FPR','reported_as_normal','abnormal_released','normal_released','MAE','RMSE','R2']:
                    v[part+'_'+key]=None if row[part] is None else row[part][key]
            flat.append(v)
        with (out/'ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as ff:
            w=csv.DictWriter(ff,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        allrows+=rows;print('AUDIT',budget,'PASS',flush=True)
    hashes=json.loads((O/'SOURCE_HASHES.json').read_text());assert all(r.sha(f.Path(k))==v for k,v in hashes.items())
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in f.b.protocol()['input_hashes'].items())
    r.write(O/'DIAGNOSTIC_FAILURES.json',diagnostics)
    failures=[]
    for budget in [54,60]:
        for arm in f.a.ARMS:
            for name in f.NAMES+['normal_midpoint','normal_midpoint_upper']:
                rr=[x for x in allrows if x['budget']==budget and x['arm']==arm and x['policy']==name]
                failures.append(dict(budget=budget,arm=arm,policy=name,rate=[x['rep']+1 for x in rr if x['test'] and x['test']['reporting_rate']<=.75],accuracy=[x['rep']+1 for x in rr if x['test'] and x['test']['accuracy']<=.8],FN=[x['rep']+1 for x in rr if x['test'] and x['test']['FN']>0],infeasible=[x['rep']+1 for x in rr if not x['feasible']]))
    r.write(O/'FAILURES.json',failures)
    r.write(O/'FINAL_QA.json',dict(recomputed_rows=checks,policies_reselected_from_development=True,baseline_reproduced=True,all_100_draws_retained=True,all_inputs_and_saved_prediction_risk_hashes_unchanged=True,developer_zero_FN_for_feasible=True,test_zero_FN_rows=sum(bool(x['test'] and x['test']['FNR']==0) for x in allrows)))
    report(allrows,failures)

def report(rows,failures):
    lines=['# 固定10%风险策略：剩余失败轮次追查','',
      '本轮保持原始输入、MA预测、B风险模型、临床类别边界和开发normal高风险90分位上限不变。论文及图表未修改。54条和60条各100轮，单源/双源各自完整保存。',
      '', '## 结论','',
      '没有找到在两套设置、两种来源配置中全部100轮同时达标的统一方案。异常门阈值区间上端对照解决了60条双源第82轮，联合达标由98/100提高到99/100；第86轮仍未达标。54条单源最大风险秩对照由93/100提高到94/100，但增加低报告率轮次，不能称为全面改善。',
      '', '## 固定对照','',
      '- baseline：上一轮classwise_reserve_0，逐条复现。',
      '- gap_lower/gap_upper：低、高MA门阈值分别取开发分数空区间的下/上端，开发放行记录完全不变，normal规则不变。',
      '- rank_mean/rank_median/rank_max：以开发预测类别内的经验分布转换HGB、Logistic、Boundary分数，分别取均值、中位数、最大值；开发侧选择低/高MA门，normal规则不变。',
      '- normal_midpoint及其upper对照：以固定60 mm分开normal低/高两段，分别选开发零FN风险界限，仍受原全局10%上限约束；没有形成有效增量。',
      '', '所有方案要求开发accuracy>80%、FN=0且normal和真实异常报告非空，再最大化放行数；报告率>75%单独验收，无强制补数。方案和失败轮次均保留，不依据测试结果逐轮切换。',
      '', '## 全部100轮结果','', '|开发条数|配置|方案|Accuracy% 均值±SD|报告率% 均值±SD|FPR% 均值±SD|零FN|三项达标|开发/测试同时达标|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for budget in [54,60]:
        summary=json.loads((O/f'combined/dev{budget}/SUMMARY.json').read_text())
        for q in summary:
            fmt=lambda k:f"{q[k]['mean']*100:.2f} ± {q[k]['sd']*100:.2f}"
            lines.append(f"|{budget}|{q['arm']}|{q['policy']}|{fmt('accuracy')}|{fmt('reporting_rate')}|{fmt('FPR')}|{q['zero_FN']}/100|{q['test_joint']}/100|{q['both_joint']}/100|")
    lines+=['','## 第82、86轮：60条双源','', '|轮次|方案|放行/181|正确|错误|Accuracy|报告率|FNR|','|---|---|---:|---:|---:|---:|---:|---:|']
    for rep in [81,85]:
        for name in ['baseline','gap_upper','rank_mean','rank_median','normal_midpoint']:
            x=next(x for x in rows if x['budget']==60 and x['arm']=='HZ+HB_to_SA' and x['rep']==rep and x['policy']==name);m=x['test']
            lines.append(f"|{rep+1}|{name}|{m['reported']}/181|{m['correct']}|{m['incorrect']}|{100*m['accuracy']:.2f}%|{100*m['reporting_rate']:.2f}%|{100*m['FNR']:.2f}%|")
    lines+=['',
      '第82轮上端对照新增11条正确报告、9条错误报告；报告率72.93%→83.98%，accuracy93.94%→88.82%。这是用更多FP换取更多放行，不是无代价提高。该规则统一应用全部100轮，未为第82轮单独设置阈值。',
      '', '第86轮现有normal检查放行100条，另有20条测试normal预测被额外风险检查暂缓，事后均正确。但开发对应额外暂缓的11条含1条真实异常，故不能凭测试标签将该检查删除。异常门又暂缓29条，其中9条正确、20条错误；全部放行异常虽然能提高测试报告率，但开发accuracy会降至77.78%，不符合预定条件。',
      '', '54条低报告率反复出现在第10、76、89轮。单源额外normal暂缓分别为23、8、34条；开发对应区域均有真实异常，开发零FN约束导致保守界限。accuracy不足则主要来自FP较多，合并风险分数能够拦截部分FP，同时损失正确报告。',
      '', '## 失败轮次（1起始）','', '|条数|配置|方案|报告率≤75%|Accuracy≤80%|FN>0|','|---|---|---|---|---|---|']
    for x in failures:
        if x['policy'] in ['baseline','gap_upper','rank_max','normal_midpoint']:
            lines.append(f"|{x['budget']}|{x['arm']}|{x['policy']}|{x['rate']}|{x['accuracy']}|{x['FN']}|")
    lines+=['','## 指标与解释','',
      'Accuracy=已放行三分类正确数/已放行数；报告率=已放行数/181；FNR=真实异常且放行为normal的数量/已放行真实异常数；FPR=真实normal且放行为异常的数量/已放行真实normal数。暂缓不算识别正确，分母为空不记0。',
      '100轮为重叠抽样，均值±SD是当前抽样波动，不是独立验证的95%置信区间。此前已反复分析该队列，本轮是探索性改进；开发选型不使用测试标签，但不应据此声称整个研究没有适应性偏倚或未来FNR必为0。',
      '', '## 下一步判断','',
      '60条双源可保留gap_upper为候选，但第86轮仍阻止“全部达标”结论。54条不宜直接替换为新方案，因为未出现同时稳定提高报告率和accuracy的控制。下一步若继续，应检查开发OOF风险分数与最终拟合风险分数的一致性/校准；继续单纯放宽异常门，会增加FP并拉低accuracy。',
      '', '## 文件','',
      '- combined/dev54/ROUNDS.csv、combined/dev60/ROUNDS.csv：各1600行，8个固定方案×2来源×100轮，开发及测试指标分列。',
      '- FAILURES.json：所有失败轮次；DIAGNOSTIC_FAILURES.json：原方案失败轮次的阻拦分解。',
      '- PROTOCOL.json、normal_midpoint/PROTOCOL.json：事先固定的本轮对照规则。',
      '- FINAL_QA.json：3200结果重算、开发选型重现、原方案复现、数据和预测哈希核对。',
      '- private目录：逐轮策略、原测试记录索引及放行动作，含全部失败轮次。']
    (O/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
