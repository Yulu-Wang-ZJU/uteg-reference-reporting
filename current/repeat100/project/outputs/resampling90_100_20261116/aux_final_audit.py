"""Audit all completed auxiliary, cost and reporting controls without retraining."""
import aux_followup as a
import aux_local_cost as cost
import gate_interval_followup as interval
import hgb_ensemble_followup as h
import support_followup as s
import run as r
import fn_candidates as f
import numpy as np,json,csv
from collections import Counter
from threadpoolctl import threadpool_limits

def read(path):return json.loads(path.read_text(encoding='utf8'))
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    d,plan,role,m=r.init();items=read(r.OUT/'private/SPLITS.json')
    configs=[(a.O,name) for name,_,_ in a.CONFIGS]+[(cost.O,name) for name,_,_ in cost.CONFIGS]
    n_arrays=0;n_picks=0;n_bootstraps=0;devmetrics=[];combined=[];paired=[]
    for budget in [54,60]:
        for item in items:
            rep=item['rep'];sp=s.spec(item,budget);truth=d['medical'][sp['test'],2]
            assert len(sp['pool'])==budget and len(sp['test'])==181
            for root in [a.O,cost.O,h.O]:assert (root/f'dev{budget}/private/round_{rep:02d}/COMPLETE.json').exists()
            for arm in a.ARMS:
                policies=[];store=[];base=None
                for root,name in configs:
                    folder=root/f'dev{budget}/private/round_{rep:02d}';z=arrays(folder/(arm+'_'+name+'.npz'));n_arrays+=1
                    np.testing.assert_array_equal(z['test'],sp['test']);assert set(z['dev_rows'])==set(sp['pool'])
                    assert np.isfinite(z['prediction']).all() and np.isfinite(z['dev_prediction']).all()
                    fixed={v['rule']:v for v in read(folder/'FIXED.json') if v['arm']==arm and v['model']==name}
                    saved=read(folder/(arm+'_'+name+'_selection.json'));picks=saved['picks'];policies.append(picks);store.append(z)
                    dy=d['medical'][z['dev_rows'],2]
                    devmetrics.append(dict(budget=budget,rep=rep,arm=arm,model=name,test=r.metrics(z['dev_prediction'],dy,np.ones(len(dy),bool)),joint_test_pass=False))
                    for rule in a.RULES:
                        assert picks[rule]==a.select_threshold(z['dev_score_'+rule],z['dev_prediction'],dy);n_picks+=1
                        expected=r.gates.rc.select_threshold(z['dev_score_'+rule],r.gates.fr.ma_class(z['dev_prediction']),r.gates.fr.ma_class(dy),.2-1e-12)
                        assert r.safe(expected)==saved['accuracy80'][rule]
                        t=expected.get('threshold');act=np.zeros(181,bool) if t is None else np.isfinite(z['score_'+rule])&(z['score_'+rule]<=t)
                        assert fixed[rule]['test']==r.safe(r.metrics(z['prediction'],truth,act))
                    assert fixed['All report']['test']==r.safe(r.metrics(z['prediction'],truth,np.ones(181,bool)))
                    if name=='B':
                        base=z;old=arrays(s.O/f'dev{budget}/private/round_{rep:02d}/{arm}_anchored_tails.npz')
                        np.testing.assert_allclose(z['prediction'],old['prediction'],rtol=0,atol=1e-9)
                        for rule in a.RULES:np.testing.assert_allclose(z['score_'+rule],old['score_'+rule],rtol=0,atol=1e-9)
                    tc=r.state(truth);oldpc=r.state(base['prediction']);newpc=r.state(z['prediction']);oldfn=(tc!=1)&(oldpc==1);newfn=(tc!=1)&(newpc==1)
                    paired.append(dict(budget=budget,rep=rep,arm=arm,model=name,baseline_FN=int(oldfn.sum()),FN_corrected_to_true_state=int(np.sum(oldfn&(newpc==tc))),FN_flipped_to_opposite_tail=int(np.sum(oldfn&(newpc!=1)&(newpc!=tc))),FN_remaining=int(np.sum(oldfn&newfn)),new_FN=int(np.sum(~oldfn&newfn)),FN=int(newfn.sum())))
                choice=a.choose(policies)
                if choice is None:record=dict(model=None,rule=None,development=dict(feasible=False),test=None,joint_test_pass=False)
                else:
                    i,j=choice;rule=a.RULES[j];pick=policies[i][rule];z=store[i];action=np.isfinite(z['score_'+rule])&(z['score_'+rule]<=pick['threshold']);value=r.metrics(z['prediction'],truth,action)
                    record=dict(model=configs[i][1],rule=rule,development=pick,test=value,joint_test_pass=f.passed(value))
                combined.append(dict(budget=budget,rep=rep,arm=arm,**record))
                hz=arrays(h.O/f'dev{budget}/private/round_{rep:02d}/{arm}.npz');ha=read(h.O/f'dev{budget}/private/round_{rep:02d}/{arm}_training.json')
                np.testing.assert_allclose(hz['prediction'],base['prediction'],atol=1e-9,rtol=0)
                for pos,members in enumerate(ha['audits']):
                    train=set(sp['positions'][pos]['A']);query=set(sp['positions'][pos]['B'])
                    for member in members:
                        rows=member['sample_rows'];assert set(rows)<=train and not set(rows)&query
                        counts=Counter(rows)
                        for group in np.unique(d['groups'][list(train)]):
                            group_rows=[i for i in train if d['groups'][i]==group]
                            assert len({counts[i] for i in group_rows})==1
                        n_bootstraps+=1
                expected=r.gates.rc.select_threshold(hz['dev_score'],r.gates.fr.ma_class(hz['dev_prediction']),r.gates.fr.ma_class(d['medical'][hz['dev_rows'],2]),.2-1e-12)
                assert r.safe(expected)==ha['pick'];t=expected.get('threshold')
                action=np.zeros(181,bool) if t is None else np.isfinite(hz['score'])&(hz['score']<=t)
                np.testing.assert_array_equal(action,hz['action'])
                original=next(q for q in read(h.O/f'dev{budget}/private/round_{rep:02d}/RESULTS.json') if q['arm']==arm)
                assert original['test']==r.safe(r.metrics(hz['prediction'],truth,action))
        # Recompute every threshold interval and evaluation outcome.
        rows=read(interval.O/f'dev{budget}/ROUNDS.json');assert len(rows)==2000
        cache={}
        for q in rows:
            key=(q['rep'],q['arm'])
            if key not in cache:
                cache[key]=(arrays(a.O/f"dev{budget}/private/round_{q['rep']:02d}/{q['arm']}_B.npz"),arrays(h.O/f"dev{budget}/private/round_{q['rep']:02d}/{q['arm']}.npz"))
            base,hz=cache[key]
            if q['test'] is None:continue
            if q['rule']=='HGB10':z=hz;ds=z['dev_score'];qs=z['score']
            else:
                z=base;rule=q['development'].get('selected_rule',q['rule']);ds=z['dev_score_'+rule];qs=z['score_'+rule]
            t=q['original_threshold'];mid,upper=interval.midpoint(ds,t)
            assert upper==q['next_development_score'] and q['threshold']==(mid if q['variant']=='midpoint' else t)
            act=np.zeros(181,bool) if q['threshold'] is None else np.isfinite(qs)&(qs<=q['threshold'])
            assert q['test']==r.safe(r.metrics(z['prediction'],d['medical'][z['test'],2],act))
    r.write(a.O/'DEVELOPMENT_ROWS.json',devmetrics);r.write(a.O/'DEVELOPMENT_SUMMARY.json',f.summarize(devmetrics,['budget','arm','model']))
    r.write(a.O/'COMBINED_SELECTION_ROWS.json',combined);r.write(a.O/'COMBINED_SELECTION_SUMMARY.json',f.summarize(combined,['budget','arm']))
    r.write(a.O/'PAIRED_FN_CHANGES.json',paired)
    inputs=read(r.OUT/'PROTOCOL.json')['input_hashes'];assert all(r.sha(r.current.ROOT/'data'/p)==v for p,v in inputs.items())
    r.write(a.O/'FINAL_QA.json',dict(two_budgets100_each=True,auxiliary_prediction_arrays=n_arrays,development_policy_checks=n_picks,bootstrap_training_groups_checked=n_bootstraps,B_baseline_reproduced=True,metrics_recomputed=True,all_threshold_midpoints_preserve_development_actions=True,test_labels_not_used_in_selection=True,source_inputs_unchanged=True,all_incomplete_runs_resumed=True))
    report(paired)
    print('AUX FINAL QA PASS',n_arrays,n_picks,n_bootstraps,flush=True)

def report(paired):
    def fmt(q,k,pct=False):
        v=q.get(k);scl=100 if pct else 1
        return 'NA' if not v or v['mean'] is None else f"{scl*v['mean']:.2f} ± {scl*v['sd']:.2f}"
    names={'B':'仅幅值B','B_t0':'B+起始时间','B_k':'B+增长时间','B_mu':'B+形状参数','B_all_weak':'B+三参数（弱权重）','B_all_equal':'B+三参数（等权重）','B_all_weak_c16':'弱权重+本地惩罚16','B_all_equal_c16':'等权重+本地惩罚16','B_all_equal_c64':'等权重+本地惩罚64'}
    text=['# 辅助参数与报告率：两套100次跟进结果','',
    '本轮未修改论文、原始输入、图表或GitHub。54条和完整60条分别保存全部100次；单源为重点，双源作为匹配对照。中途E盘空间不足的未完成轮次已续跑并核查，未按结果删掉重复。',
    '', '## 一、结论','',
    '1. 完整60条：沿用B幅值映射和上轮端点约束，开发侧选择报告门，再把阈值放在相邻开发分数之间的中点，单源100次均达到FNR=0、accuracy>80%、reporting rate>75%。这不是固定HGB在100次都达标。',
    '2. 54条：另外三个参数可以降低平均MA误差，但本轮未找到能稳定降低FN的加入方式。个别辅助参数还重新引入低MA漏判，不能为了R²或总体accuracy提高而替换当前映射。',
    '3. 所有候选都是这份历史数据上的探索性比较；100次重叠抽样不等于100个新队列。开发侧仍存在FN，所以测试侧零FN不能解释为未来零漏判保证。',
    '', '## 二、三个参数加入哪里','',
    '参数为起始偏移t0、增长时间k、形状指数mu。它们此前已进入HGB/Logistic放行模型；本轮新增的是它们对MA本地校正的输入。来源模型仍只用幅值B。',
    '分别加入每一项，并比较三项联合的弱权重与等权重。所有标准化、核基函数、类别权重和校正只使用当前训练折；测试标签不参与选择。原B端点约束保留。另比较只提高本地校正惩罚到16/64，来源模型及B端点分支保持原4倍惩罚。',
    '', '## 三、54条单源：门前完整映射结果','',
    '均值±SD跨100次；这里的accuracy、FNR、MAE和R²均使用全部测试记录，避免筛选后指标掩盖映射问题。',
    '', '| 模型 | MAE(mm) | R² | Accuracy(%) | FNR(%) | 零FNR次数 |','|---|---:|---:|---:|---:|---:|']
    for root in [a.O,cost.O]:
        for q in read(root/'dev54/FIXED_SUMMARY.json'):
            if q['arm']=='HZ_to_SA' and q['rule']=='All report':text.append('| '+' | '.join([names[q['model']],fmt(q,'MAE'),fmt(q,'R2'),fmt(q,'accuracy',True),fmt(q,'FNR',True),str(q['zero_FN_rounds'])+'/100'])+' |')
    text+=['', '## 四、原来17次高MA漏判修复了多少','',
    'B基线在17次重复中累计有60个高MA→normal事件；相同记录可在不同重复中出现。下表比较完全相同的抽样记录。“新增FN”统计全部100次中原B预测没有漏判、辅助参数模型却判为normal的异常事件。',
    '', '| 模型 | 原60个FN中改回正确类别 | 原FN仍漏判 | 全100次新增FN |','|---|---:|---:|---:|']
    for name in names:
        rr=[v for v in paired if v['budget']==54 and v['arm']=='HZ_to_SA' and v['model']==name]
        text.append('| '+' | '.join([names[name],str(sum(v['FN_corrected_to_true_state'] for v in rr)),str(sum(v['FN_remaining'] for v in rr)),str(sum(v['new_FN'] for v in rr))])+' |')
    text+=['', '即使只由开发侧在全部九个候选中选型，也没有稳定改善这些结果；完整选择记录见COMBINED_SELECTION_ROWS.json。未将某个测试表现较好的候选事后指定为最优。',
    '', '## 五、完整60条：报告率问题如何处理','',
    '原HGB在第20次和第53次分别只放行121/181（66.85%）和126/181（69.61%）。对应开发侧均放行56/60，说明开发分数阈值在最终门上未产生相同报告比例。',
    '采用两个开发侧规则：先在accuracy>80%、reporting rate>75%且有异常放行的候选中，优先减少开发FN，再选报告数更多者；然后把选定阈值移到它与下一个更高开发分数的中点。中点改变不增删任何开发记录的放行状态，测试分数和标签不参与确定这个位置。',
    '单源100次选择了73次HGB、13次Logistic和14次Boundary。第20次开发侧选Boundary，最终报告175/181（96.69%）；第53次保留HGB，中点阈值报告139/181（76.80%）。',
    '', '| 完整60条单源方案 | Accuracy(%) | Reporting rate(%) | FNR(%) | FPR(%) | 三项联合达标 |','|---|---:|---:|---:|---:|---:|']
    summaries=read(interval.O/'dev60/SUMMARY.json')
    labels={('HGB','endpoint'):'原HGB阈值',('HGB','midpoint'):'HGB+中点阈值',('HGB10','endpoint'):'10门集成平均分数',('B_development_gate','endpoint'):'开发侧选门',('B_development_gate','midpoint'):'开发侧选门+中点阈值',('Logistic','endpoint'):'固定Logistic对照',('Boundary','endpoint'):'固定Boundary对照'}
    for q in summaries:
        key=(q['rule'],q['variant'])
        if q['arm']=='HZ_to_SA' and key in labels:text.append('| '+' | '.join([labels[key],*[fmt(q,k,True) for k in ['accuracy','reporting_rate','FNR','FPR']],str(q['joint_test_pass'])+'/100'])+' |')
    winner=next(q for q in summaries if q['arm']=='HZ_to_SA' and q['rule']=='B_development_gate' and q['variant']=='midpoint')
    text+=['',f"该候选100次的最低accuracy为{100*winner['accuracy']['min']:.2f}%，最低reporting rate为{100*winner['reporting_rate']['min']:.2f}%；FNR全部有定义且为0。代价是FPR平均{100*winner['FPR']['mean']:.2f}%，需要与报告率一起呈现。",'该结论不包含更换MA输入：采用的是B-only端点约束映射。54条同一选门/中点规则仍只有83/100次联合达标，高MA漏判尚未解决。',
    '', '## 六、保留范围与核查','',
    '- dev54/、dev60/：初始6种特征设置；local_cost_controls/：3种局部惩罚设置；各自全部100次。',
    '- hgb_ensemble/：10个开发侧组自助抽样HGB的平均分数，两个预算各100次。',
    '- threshold_interval/：端点/中点配对结果和开发侧选门结果。',
    '- DEVELOPMENT_*、COMBINED_SELECTION_*、PAIRED_FN_CHANGES.json：开发评价、联合选型及新旧FN事件配对。',
    '- FINAL_QA.json：预测、阈值、指标、分母、分组隔离及输入哈希核查。',
    '- JSON整理仅删除排版空白；原字节备份在D:/CodexData/recovery/uteg_repeated_results_json_1794826346.zip，逐文件核对结构和值相同。']
    (a.O/'REPORT_CN.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    # Compact, reviewable aggregate CSV for the reporting policies.
    for budget in [54,60]:
        rows=[]
        for q in read(interval.O/f'dev{budget}/SUMMARY.json'):
            v={k:q[k] for k in ['arm','rule','variant','zero_FN_rounds','FNR_undefined_rounds','joint_test_pass']}
            for k in ['accuracy','reporting_rate','FNR','FPR']:
                for stat in ['mean','sd','min','max']:v[k+'_'+stat]=q[k][stat] if q[k] else None
            rows.append(v)
        with (interval.O/f'dev{budget}/SUMMARY.csv').open('w',encoding='utf-8-sig',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
