"""Shared reporting accounting and retained historical diagnostic reports.

The current suite calls replay/sequences/paired_primary. The aggregate, verify
and run entries below retain the earlier combined model-curve diagnostic.
See the package README before using those entries.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,r2_score
from threadpoolctl import threadpool_limits
from . import core as c
from .gates import clean_write
from code_submit.paper_b085_hospital_replay.equal_error import budget_mix,exact_mix

SCHEDULES={'5-10-15':[300,600,900],'10-15':[600,900],'fixed15':[900]}
RULES=['ALL_REPORT','BOUNDARY95','LR','HGB','TWO_SIDED','BOUNDARY_PAPER']

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def replay(pred,actions,times):
    first=np.full_like(pred[900],np.nan);when=np.full(pred[900].shape,-1,int);reported=np.zeros(pred[900].shape,bool)
    for h in times:
        add=actions[h]&np.isfinite(pred[h])&~reported
        first[add]=pred[h][add];when[add]=h;reported|=add
    assert np.array_equal(when>=0,reported)
    return first,when,reported

def sequences(root):
    out=root/'sequence';out.mkdir(exist_ok=True);d,*_=c.context();results={}
    for direction in c.TARGET:
        points={h:load(root/'primary'/f'{direction}_T{h}_B0.npz') for h in c.HORIZONS}
        gates={h:load(root/'gates'/f'{direction}_T{h}.npz') for h in c.HORIZONS}
        idx=points[900]['test'];y=d['y'][idx,2]
        assert all(np.array_equal(points[h]['test'],idx) for h in c.HORIZONS)
        result={};saved={};pp={h:points[h]['prediction'] for h in c.HORIZONS}
        for rule in RULES:
            ac={h:np.isfinite(pp[h]) if rule=='ALL_REPORT' else points[h]['accepted'] if rule=='BOUNDARY95' else gates[h][rule+'_action'] for h in c.HORIZONS}
            result[rule]={}
            for label,times in SCHEDULES.items():
                p,t,a=replay(pp,ac,times);m=c.summarize(p,y,a)
                m['by_time']={str(h):dict(reported=int(np.sum(t==h)),correct=int(np.sum((t==h)&(c.accounting.states(p)==c.accounting.states(y))))) for h in times}
                result[rule][label]=m
                saved[f'{rule}__{label}_prediction']=p;saved[f'{rule}__{label}_time']=t;saved[f'{rule}__{label}_action']=a
        np.savez_compressed(out/f'{direction}.npz',**saved,test=idx)
        results[direction]=result
    clean_write(out/'RESULTS.json',results)
    # all301 remains distinct; its accepted boundary95 scheme is the selected one.
    points={h:load(root/'all301'/f'T{h}.npz') for h in c.HORIZONS};local=points[900]['local'];yy=d['y'][local,2]
    result={};saved={}
    for j,direction in enumerate(c.MAIN):
        result[direction]={}
        for label,times in SCHEDULES.items():
            p,t,a=replay({h:points[h]['prediction'][j] for h in c.HORIZONS},{h:points[h]['accepted'][j] for h in c.HORIZONS},times)
            result[direction][label]=c.summarize(p,yy,a)
            saved[f'{direction}__{label}_prediction']=p;saved[f'{direction}__{label}_time']=t;saved[f'{direction}__{label}_action']=a
    np.savez_compressed(out/'all301.npz',**saved,local=local)
    clean_write(out/'ALL301_RESULTS.json',result)
    return results,result

def diagnostics(root):
    out=root/'reporting_diagnostics';out.mkdir(exist_ok=True);d,*_=c.context();comparisons={};policies={};test=None
    for direction in c.MAIN:
        g=load(root/'gates'/f'{direction}_T900.npz');p=g['prediction'];test=g['test'];y=d['y'][test,2];truth=c.accounting.states(y);pred=c.accounting.states(p)
        dev=c.read(root/'gates'/f'{direction}_T900.development.json')
        lr_actions=[];dual_actions=[];eligible=[]
        for si in range(20):
            candidates=np.flatnonzero((pred[si]==0)&np.isfinite(g['LR_score'][si])&np.isfinite(g['DUAL_score'][si]))
            n=int(np.floor(.2*len(candidates)));eligible.append(dict(seed=1000+si,common_candidates=len(candidates),selected=n))
            for key,dest in [('LR',lr_actions),('DUAL',dual_actions)]:
                a=np.zeros(len(y),bool)
                if n:
                    ordered=sorted(candidates,key=lambda i:(g[key+'_score'][si,i],str(d['ids'][test[i]])))
                    a[ordered[:n]]=True
                dest.append(a)
        la=np.array(lr_actions);da=np.array(dual_actions);ab=truth!=0
        comparisons[direction]=dict(scope='Common finite-score predicted-normal candidates; exact floor20% quota within each seed.',
            eligible=eligible,LR=dict(released=int(la.sum()),abnormal_as_normal=int(np.sum(la&ab))),
            DUAL=dict(released=int(da.sum()),abnormal_as_normal=int(np.sum(da&ab))),
            paired_error_fraction_difference=c.accounting.paired_ratio((da&ab).astype(int)-(la&ab),la,d['groups'][test]))
        # Policies are defined from DEVELOPMENT score quantiles only. Test labels
        # below assess attainable error/yield combinations retrospectively.
        configs=[('none',0,0),('all',0,0),('boundary95',0,0)]
        configs += [('LR',q,0) for q in np.linspace(0,1,21)]
        configs += [('DUAL',q,t) for q in np.linspace(0,1,11) for t in np.linspace(0,1,11)]
        actions=np.zeros((len(configs),20,len(y)),bool)
        for si in range(20):
            scores=dev[si]['scores']
            def threshold(key,q):
                v=np.array([np.nan if x is None else x for x in scores[key]],float);v=v[np.isfinite(v)]
                return float(np.quantile(v,q,method='higher')) if len(v) else -np.inf
            for k,(kind,q,t) in enumerate(configs):
                if kind=='none':continue
                if kind=='all':a=np.isfinite(p[si])
                elif kind=='boundary95':a=g['boundary95_action'][si]
                elif kind=='LR':a=(g['LR_score'][si]<=threshold('LR',q))&np.isfinite(g['LR_score'][si])
                else:a=((pred[si]==0)&(g['DUAL_score'][si]<=threshold('DUAL',q))&np.isfinite(g['DUAL_score'][si]))|((pred[si]!=0)&(g['LR_score'][si]<=threshold('LR',t))&np.isfinite(g['LR_score'][si]))
                actions[k,si]=a&np.isfinite(p[si])
        correct=actions&(pred==truth);errors=actions&(pred!=truth)
        ey=errors.mean((1,2));cy=correct.mean((1,2));budgets=(.005,.01,.02,.05,.075,.10)
        matches={str(b):dict(budget_mix=budget_mix(ey,cy,b),exact_mix=exact_mix(ey,cy,b)) for b in budgets}
        policies[direction]=dict(configs=configs,actions=actions,correct=correct,errors=errors,ey=ey,cy=cy,matches=matches)
        np.savez_compressed(out/f'{direction}_policy_arrays.npz',actions=actions,correct=correct,errors=errors,test=test)
    clean_write(out/'MATCHED20.json',comparisons)
    groups=d['groups'][test];uniq=np.unique(groups);rng=np.random.default_rng(20261102);cross={}
    # Same10% identity groups for every method; remaining groups assess selected
    # policy. Repeated partitions are exploratory, not untouched external data.
    for direction,m in policies.items():
        rows=[]
        for repeat in range(20):
            chosen=np.random.default_rng(20261102+repeat).permutation(uniq)[:max(1,int(np.ceil(.1*len(uniq))))]
            cal=np.isin(groups,chosen);held=~cal
            ey=m['errors'][:,:,cal].mean((1,2));cy=m['correct'][:,:,cal].mean((1,2))
            ids=np.flatnonzero(ey<=.10+1e-12);best=min(ids,key=lambda i:(-cy[i],ey[i],i))
            a=m['actions'][best,:,held];err=m['errors'][best,:,held];cor=m['correct'][best,:,held]
            rows.append(dict(repeat=repeat,calibration_records=int(cal.sum()),held_records=int(held.sum()),policy_index=int(best),
                correct_yield=float(cor.mean()),incorrect_yield=float(err.mean()),coverage=float(a.mean()),
                conditional_error=float(err.sum()/a.sum()) if a.any() else None))
        cross[direction]=rows
    paired={}
    target=policies['target_only_SA']
    for arm in c.MAIN[1:]:
        src=policies[arm];paired[arm]={}
        for b in (.005,.01,.02,.05,.075,.10):
            left=target['matches'][str(b)]['budget_mix'];right=src['matches'][str(b)]['budget_mix']
            if left is None or right is None:paired[arm][str(b)]=None;continue
            def mix(m,sel,key):return (1-sel['weight_j'])*m[key][sel['strategy_i']]+sel['weight_j']*m[key][sel['strategy_j']]
            diff=mix(src,right,'correct').astype(float)-mix(target,left,'correct')
            paired[arm][str(b)]=dict(correct_yield_difference=c.accounting.paired_ratio(diff,np.ones_like(diff),groups),
                target_actual_error=left['error_yield'],source_actual_error=right['error_yield'],
                intervals='Descriptive group bootstrap conditional on retrospective chosen mixtures; not selection-adjusted.')
    clean_write(out/'EXPLORATORY_ERROR_MATCHING.json',dict(test_label_aware=True,denominator='incorrect reports / full test cohort, not conditional error',
        policies={k:dict(configs=v['configs'],error_yield=v['ey'],correct_yield=v['cy'],matching=v['matches']) for k,v in policies.items()},paired=paired,
        limitation='Retrospective convex combinations are not validated deployment policies or error non-inferiority.'))
    clean_write(out/'EXTRA_LABEL_CROSS_CALIBRATION.json',dict(calibration_fraction=.10,error_yield_cap=.10,partitions=20,
        grouping='identity proxy; same subsets for all methods',results=cross,independent_validation=False))

def paired_primary(root):
    d,*_=c.context();result={}
    z=load(root/'primary/target_only_SA_T900_B0.npz');y=d['y'][z['test'],2];t=c.accounting.states(y);groups=d['groups'][z['test']]
    base=c.accounting.states(z['prediction']);ab=t!=0;normal=t==0
    for arm in c.MAIN[1:]:
        p=load(root/'primary'/f'{arm}_T900_B0.npz');pred=c.accounting.states(p['prediction'])
        result[arm]=dict(FNR_difference=c.accounting.paired_ratio(((pred==0)&ab).astype(int)-((base==0)&ab),np.broadcast_to(ab,base.shape),groups),
            FPR_difference=c.accounting.paired_ratio(((pred!=0)&normal).astype(int)-((base!=0)&normal),np.broadcast_to(normal,base.shape),groups),
            MAE_difference=c.accounting.paired_ratio(abs(p['prediction']-y)-abs(z['prediction']-y),np.ones_like(base),groups))
    clean_write(root/'PAIRED_PRIMARY_INTERVALS.json',result)

def table(head,rows):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def pct(x):return 'NA' if x is None else f'{x:.2%}'
def num(x):return 'NA' if x is None else f'{x:.3f}'

def aggregate():
    root=c.OUT;out=root/'summary';out.mkdir(exist_ok=True);d,*_=c.context();rows=[];reports=[]
    for path in sorted((root/'primary').glob('*.metrics.json')):
        z=c.read(path);m=z['all_report'];a=z['boundary95']
        rows.append(dict(protocol='paper',configuration=path.stem,**{k:m[k] for k in ['fnr','fn','fn_denominator','fpr','fp','agreement','mae','r2','low_recall','high_recall']},coverage=a['coverage'],post_gate_agreement=a['agreement'],post_gate_fnr=a['fnr']))
    for path in sorted((root/'all301').glob('*.metrics.json')):
        for arm,z in c.read(path).items():
            m=z['all_report'];a=z['boundary95'];rows.append(dict(protocol='all301',configuration=arm+'_'+path.stem,
                **{k:m[k] for k in ['fnr','fn','fn_denominator','fpr','fp','agreement','mae','r2','low_recall','high_recall']},coverage=a['coverage'],post_gate_agreement=a['agreement'],post_gate_fnr=a['fnr']))
    with (out/'PRIMARY_AND_BUDGET_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    auxiliary=[]
    for path in (root/'calibration').glob('*.npz'):
        z=load(path);ref=d['medical'][z['test']]
        for arm in ['raw','affine','joint','huber','complete']:
            p=z[arm];valid=np.isfinite(p).all(-1)
            yy=np.broadcast_to(ref,p.shape);errors=np.abs(p[valid]-yy[valid]).mean(0)
            auxiliary.append(dict(direction=path.stem,arm=arm,R_MAE=errors[0],K_MAE=errors[1],MA_MAE=errors[2],angle_MAE=errors[3],
                interpretation='R/K/angle are auxiliary four-input Ridge channels; complete changes MA only'))
    with (out/'AUXILIARY_CALIBRATION_ERRORS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(auxiliary[0]));w.writeheader();w.writerows(auxiliary)
    historical=[c.OLD/'historical_controls/MANIFEST.json',c.OLD/'summary/REPORT_ZH.md',c.OLD/'EVALUATION_COMPLETE.json']
    clean_write(out/'HISTORICAL_REUSE.json',dict(scope='Original acquired-signal fitting and prior R/K/angle evidence retained under their existing protocol; not attributed to the new MA-only branch.',
        files={str(p):c.sha(p) for p in historical},new_auxiliary_controls='AUXILIARY_CALIBRATION_ERRORS.csv'))
    main=[c.read(root/'primary'/f'{a}_T900_B0.metrics.json') for a in c.MAIN]
    anchor=c.read(root/'all301/T900.metrics.json')
    lines=['# 历史诊断记录：inward4实验及模型曲线','',
        '本报告汇总历史实验及单独的模型曲线诊断；当前交付流程见代码包README。该历史分析保留301条分组评价，并按原60条开发＋201条评价、20个种子完成对照。B-only RBF外院先验＋本地RBF修正固定不再调参，inward4作用于最终MA。主要报告门沿用已选定的边界距离95%规则；原论文LR、HGB、双侧规则另外作为对照。','',
        '医生确认新B的说明已记录。工作文件同时保留B由参考MA混合计算的来源，因此所有含新B的结果均为诊断，不能据此证明未知MA的独立预测能力。按四参数生成的信号称为模型重建曲线；它不是新增的逐点实测数据。这些记录描述当时的诊断，不作为当前交付版本说明。','',
        '## 1. 原301条评价（每条一次）','',
        table(['方案','FNR','FPR','三分类一致率','MAE','R²','门覆盖'],[[a,pct(anchor[a]['all_report']['fnr'])+f" ({anchor[a]['all_report']['fn']}/48)",pct(anchor[a]['all_report']['fpr']),pct(anchor[a]['all_report']['agreement']),num(anchor[a]['all_report']['mae']),num(anchor[a]['all_report']['r2']),pct(anchor[a]['boundary95']['coverage'])] for a in c.MAIN]),'',
        '## 2. 原论文60/201评价','',
        table(['方案','门前FNR','门前FPR','三分类一致率','MAE','R²','门覆盖','门后一致率','门后FNR'],[[a,pct(z['all_report']['fnr'])+f" ({z['all_report']['fn']}/{z['all_report']['fn_denominator']})",pct(z['all_report']['fpr']),pct(z['all_report']['agreement']),num(z['all_report']['mae']),num(z['all_report']['r2']),pct(z['boundary95']['coverage']),pct(z['boundary95']['agreement']),pct(z['boundary95']['fnr'])] for a,z in zip(c.MAIN,main)]),'',
        '201条含7低、170正常、24高MA。20个种子是同一批记录的重复流程，620个异常事件对应31条异常记录。固定RBF主预测没有随机参数搜索；最终预测可能在20个种子间相同，门阈值仍来自各自开发分折。0次观察漏判不能解释为总体FNR为0。','',
        '## 3. 历史模型曲线诊断：生成及5/10分钟重拟合','',
        '以下为单独保存的历史模型曲线诊断，不是当前实测输入流程。采用已冻结的广义指数模型：B×[1−exp(−((t−t0)+/k)^μ)]，扣除前10点中位数。保留其余三个参数，以更新B生成900点曲线。再仅向五起点约束拟合器提供前300或600点，不把完整参数作为初值。每个片段按自身幅值归一化再还原，模型的幅值线性关系允许按各训练折的新B精确缩放。','',
        '480条×两个前缀共960次拟合；1条湖北记录前5分钟近乎平坦，标记不可用并保留在报告分母，10分钟恢复可用。原质控失败记录继续暂缓。重建曲线拟合R²接近1，是对同一数学模型的回拟合结果，不是MA预测R²或仪器真实性验证。','',
        '重建曲线含参考MA信息，早期片段也继承该信息。这一组序贯结果不得用作真实早期临床收益证据。使用原实测前缀的另一组结果保存在父目录，用于区分影响来源。','',
        '## 4. 历史模型曲线诊断中的首次报告（原论文口径，主边界门）','']
    seq=c.read(root/'reconstructed/sequence/RESULTS.json');ss=c.read(root/'reconstructed/sequence/ALL301_RESULTS.json')
    lines += [table(['方案','时间方案','正确/错误/未报','覆盖','一致率','FNR','FPR'],[[a,s,f"{m['correct']}/{m['incorrect']}/{m['withheld']}",pct(m['coverage']),pct(m['agreement']),pct(m['fnr']),pct(m['fpr'])] for a in c.MAIN for s,m in seq[a]['BOUNDARY95'].items()]),'',
        '每个时间方案单独锁定首次报告；后续预测不会撤销已发出的错误。表中FNR/FPR是已报告异常/正常中的条件比例；另存完整类别分母及未报告去向。','',
        '## 5. 历史模型曲线诊断中的301条序贯评价','',table(['方案','时间方案','正确/错误/未报','覆盖','一致率','FNR'],[[a,s,f"{m['correct']}/{m['incorrect']}/{m['withheld']}",pct(m['coverage']),pct(m['agreement']),pct(m['fnr'])] for a in c.MAIN for s,m in ss[a].items()]),'',
        '## 6. 历史实验清单与比较范围','',
        '- 主映射：九迁移方向＋三个中心target-only，三个检查点，20个种子；301条另列。',
        '- 六种原source模型：Ridge、KRR、PLS、SVR、MLP、TabTransformer，九方向×20种子；均接相同本地B-only inward4修正。原source模型保留四输入和原损失，主模型是单B＋inward4；因此比较是source方案敏感性分析，不能归因于模型结构一个因素。',
        '- 校准：原始、独立仿射、联合线性、联合Huber、最终MA修正。联合控制所需R/K/角度由辅助四参数Ridge提供，不能称为新B-only模型的三项预测能力。',
        '- 标签预算：20/40/全开发池与target-only/中位数控制；40条只对原SAHZU预算适用。',
        '- 表示和损失：单B/三参数/四参数；普通/类别加权/inward4，在301条和原论文主方向分别计算。',
        '- 报告门：all-report、主边界95%、原论文选点的边界门、LR、HGB、双侧门。风险输入用物理四参数及MA/距离/类别，未建模R/K/角度输出槽固定0；三个风险模型共享同样输入，绝不填真实参考值。',
        '- 序贯：5–10–15、10–15、固定15；原前缀与重建前缀两套均计算。',
        '- 附录探索：共同可用的预测正常候选中20%配额、分组配对区间、开发侧固定政策网格的回顾性等错误预算诊断、额外10%测试组标签交叉校准。后两项明确使用测试标签，不属于验证过的部署规则或错误非劣证据。',
        '- 原始信号拟合及未改变的R/K/角度旧证据保留原出处；没有把旧证据重新归给新MA分支。','',
        '目录：`primary/`、`mapping/`、`calibration/`、`controls/`、`gates/`、`sequence/`、`reporting_diagnostics/`；重建前缀对应结果在`reconstructed/`，曲线与重拟合参数在`reconstructed_curves/`。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    clean_write(out/'RESULT_INDEX.json',dict(primary=rows,all301=anchor,paper_main=dict(zip(c.MAIN,main)),reconstructed_sequence=seq,reconstructed_all301_sequence=ss))

def verify():
    root=c.OUT;checks={};required=[]
    assert len(list((root/'mapping').glob('*.npz')))==54
    assert len(list((root/'calibration').glob('*.npz')))==9
    assert len(list((root/'primary').glob('*.npz')))==52
    assert len(list((root/'gates').glob('*.npz')))==36
    assert len(list((root/'reconstructed/gates').glob('*.npz')))==36
    checked=0;d,*_=c.context()
    for base in [root,root/'reconstructed']:
        for path in (base/'primary').glob('*.npz'):
            z=load(path);y=d['y'][z['test'],2];pred=z['prediction'];valid=np.isfinite(pred);truth=np.broadcast_to(y,pred.shape)
            # Source attestation and validation are independent metadata fields.
            # Read both retained schemas without inferring one flag from another.
            from .input_contract import read_legacy_source_flags
            read_legacy_source_flags(z)
            m=c.summarize(pred,y);cm=confusion_matrix(c.accounting.states(truth[valid]),c.accounting.states(pred[valid]),labels=[-1,0,1])
            assert m['confusion']==cm.tolist() and m['fn']==int(cm[0,1]+cm[2,1])
            assert np.isclose(m['r2'],r2_score(truth[valid],pred[valid]))
            checked+=1
        for direction in c.TARGET:
            path=base/'sequence'/f'{direction}.npz';z=load(path)
            for rule in RULES:
                for schedule in SCHEDULES:
                    when=z[f'{rule}__{schedule}_time'];act=z[f'{rule}__{schedule}_action']
                    assert np.array_equal(when>=0,act)
            required.append(path)
    with np.load(root/'all301/T900.npz') as z,np.load(c.selected.OUT/'private/PREDICTIONS.npz') as old:
        np.testing.assert_allclose(z['prediction'],old['predictions'],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(z['accepted'],old['accepted'])
    original=c.read(root/'PROTOCOL.json')['input_hashes'];assert all(c.sha(p)==h for p,h in original.items())
    checks.update(mapping_configurations=54,calibration_directions=9,paper_primary_and_budget_files=52,
        gate_direction_checkpoints=72,reconstruction_fits=960,primary_numerical_rescores=checked,
        original_input_hashes_unchanged=True,selected301_reproduced=True,first_report_lock_checked=True,
        manuscript_changed=False,oracle_input_present=True,independent_predictive_goal_achieved=False)
    # Do not imply every legacy Huber optimizer converged; finite outputs and
    # statuses are retained. Count any non-convergence explicitly.
    huber_fail=[]
    for folder in ['mapping','calibration']:
        for path in (root/folder).glob('*.audit.json'):
            for r in c.read(path):
                if not r['huber']['success']:huber_fail.append(dict(file=str(path),seed=r['seed'],status=r['huber']['status']))
    checks['legacy_huber_nonconvergence']=huber_fail
    clean_write(root/'summary/VERIFICATION.json',checks)
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='EVALUATION_COMPLETE.json' and p.suffix in ['.json','.md','.csv','.npz'] and 'legacy_runtime' not in str(p) and 'source_models' not in str(p)]
    clean_write(root/'EVALUATION_COMPLETE.json',dict(status='COMPLETE_WITH_EXPLICIT_PROTOCOL_CHANGES_AND_ORACLE_LINEAGE',checks=checks,
        artifacts={str(p):c.sha(p) for p in files},code_hashes={str(p):c.sha(p) for p in Path(__file__).parent.glob('*.py')}))

def run():
    for root in [c.OUT,c.OUT/'reconstructed']:
        sequences(root);diagnostics(root);paired_primary(root)
    aggregate();verify();print('ALL_EXPERIMENTS_COMPLETE',str(c.OUT/'summary/REPORT_ZH.md'),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
