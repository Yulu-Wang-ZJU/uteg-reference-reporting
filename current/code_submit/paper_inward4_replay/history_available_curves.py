"""History-aware capability and locked first-report comparisons, three levels.

All earlier predictions remain available. Cross-time score comparison is fitted
using development outcomes only; outer labels are read only for evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

from . import matched_checkpoint_curves as m

c,e,check=m.c,m.e,m.check
ROOT=m.s.suite.ROOT/'history_available_curves_20261103'
PRIVATE=ROOT/'private'
TIMES=[5,10,15]
COVERAGES=[.10,.20,.40]
AGREEMENTS=[.85,.90,.95]
COLORS=['#396B91','#C1843E','#4B866D']
MARKERS=['o','s','^']


def initialize():
    m.s.suite.initialize();PRIVATE.mkdir(parents=True,exist_ok=True)
    contract=dict(
        claim='Use all predictions available by each checkpoint; retain earlier options without choosing by held-out correctness.',
        coverage_levels=COVERAGES,agreement_levels=AGREEMENTS,
        capability='At each time choose the lowest development-calibrated error-score prediction available for each record, then rank and select the same k records. Previously selected records may change; this is not a replay of issued reports.',
        calibrator='Increasing isotonic error calibration fitted separately for each checkpoint, on eligible development OOF score/error pairs. Development risk estimates cross-fitted by existing development fold; query calibration fitted on the full development set. No fitted risk is a guaranteed probability.',
        tie='Calibrated risk plus1e-6 times the training-score empirical percentile preserves score ordering on isotonic plateaus. Prefer earlier prediction on equal composite score; record-identity hash breaks remaining batch rank ties.',
        downstream_input='Existing5min LR/worst-member score;10min worst boundary-distance/consensus/low-B veto;15min boundary distance. Existing upstream mappings unchanged.',
        interface_limit='Development score is from its OOF predictor, whereas query5/10 scores add model consensus; this inherited difference is evaluated, not asserted calibrated.',
        sequence='For each fixed agreement target, select stage cutoff on development cumulative reports and still-unreported records. Query reports remain locked; later stages may use an earlier saved prediction for a previously unreported record. Cumulative coverage cannot decrease.',
        threshold='Maximize cumulative development report count with cumulative agreement>=target, minimum max(10,ceil(.1*N)) cumulative support,5 identity-proxy groups,2 folds. No added zero-FN constraint; FNR reported.',
        actual_agreement='Observed agreement may vary despite increasing information. No test-label best-of-time selection, no cumulative maxima of observed accuracy, no silent replacement of issued errors.',
        protocol_paper='Original60/201,20seeds,4020 record-seed events.',
        protocol_all301='Frozen grouped5outer folds,3inner-partition repetitions,903 record-repeat events.',
        uncertainty='Pooled ratios only; repeated events are not independent patients. Internal exploratory analysis after prior data inspection.',
        figures='Two time-axis quantitative grids, three levels each; PNG/PDF/SVG plus source CSV.',
        provenance='U-TEG MA and curves are original measured values. The 5- and 10-minute inputs use the first 300 and 600 samples of each 900-sample recording.',
        B_from_original_data=True,manuscript_changed=False,
        software_reference='https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html')
    p=ROOT/'CONTRACT.json'
    if p.exists():assert c.read(p)==contract
    else:m.write_json(p,contract)


def calibration_predict(score,error,fit_mask,query):
    mask=fit_mask & np.isfinite(score)
    out=np.full(len(query),np.inf)
    finite=np.isfinite(query)
    if not mask.any():return out,dict(n=0)
    xs=score[mask];ys=error[mask].astype(float)
    if len(np.unique(xs))==1:
        risk=np.full(finite.sum(),ys.mean());knots=[float(xs[0])];values=[float(ys.mean())]
    else:
        model=IsotonicRegression(increasing=True,out_of_bounds='clip',y_min=0,y_max=1).fit(xs,ys)
        risk=model.predict(query[finite]);knots=model.X_thresholds_.tolist();values=model.y_thresholds_.tolist()
    percentile=np.searchsorted(np.sort(xs),query[finite],side='right')/len(xs)
    out[finite]=risk+1e-6*percentile
    return out,dict(n=len(xs),errors=int(ys.sum()),knots=knots,values=values)


def update_history(old,new_prediction,new_score,new_eligible,minute):
    prediction,score,source=old
    change=new_eligible & np.isfinite(new_score) & (new_score<score-1e-12)
    result=(np.where(change,new_prediction,prediction),np.where(change,new_score,score),np.where(change,minute,source))
    assert np.all(result[1]<=score)
    return result


def histories(dev,query):
    dh=(np.full(len(dev['rows']),np.nan),np.full(len(dev['rows']),np.inf),np.full(len(dev['rows']),-1,int))
    qh=(np.full(len(query['ma5']),np.nan),np.full(len(query['ma5']),np.inf),np.full(len(query['ma5']),-1,int))
    development={};heldout={};fits=[]
    for minute in TIMES:
        st=m.stage(dev,query,minute)
        err=(st['dev_pred']!=dev['truth']).astype(int)
        ds=np.full(len(err),np.inf)
        for fold in np.unique(dev['fold']):
            hold=dev['fold']==fold;fit=st['dev_eligible'] & ~hold
            assert not set(dev['groups'][fit]) & set(dev['groups'][hold])
            ds[hold],_=calibration_predict(st['dev_score'],err,fit,st['dev_score'][hold])
        qs,cal=calibration_predict(st['dev_score'],err,st['dev_eligible'],st['score'])
        dh=update_history(dh,dev['ma'+str(minute)],ds,st['dev_eligible'],minute)
        qh=update_history(qh,st['prediction'],qs,st['eligible'],minute)
        development[minute]=tuple(x.copy() for x in dh)
        heldout[minute]=tuple(x.copy() for x in qh)
        fits.append(dict(minute=minute,calibration=cal))
    return development,heldout,fits


def cumulative_cutoff(dev,prediction,score,prior_prediction,prior_action,target):
    eligible=~prior_action & np.isfinite(prediction) & np.isfinite(score)
    best=None;minimum=max(10,int(np.ceil(.1*len(score))))
    old_correct=int(np.sum(prior_action & (e.state(prior_prediction)==dev['truth'])))
    for cutoff in np.unique(score[eligible]):
        new=eligible & (score<=cutoff);union=prior_action|new;n=int(union.sum())
        if n<minimum or len(np.unique(dev['groups'][union]))<5 or len(np.unique(dev['fold'][union]))<2:continue
        good=old_correct+int(np.sum(new & (e.state(prediction)==dev['truth'])))
        if good/n+1e-12<target:continue
        value=dict(threshold=float(cutoff),cumulative_selected=n,cumulative_correct=good,new_selected=int(new.sum()),development_agreement=good/n)
        if best is None or value['new_selected']>best['new_selected']:best=value
    return best


def replay(dev,dh,qh,target):
    da=np.zeros(len(dev['rows']),bool);dp=np.full(len(da),np.nan)
    n=len(qh[5][0]);qa=np.zeros(n,bool);qp=np.full(n,np.nan)
    qt=np.full(n,-1,int);qs=np.full(n,-1,int)
    result={};selections=[]
    for minute in TIMES:
        d_pred,d_score,_=dh[minute];q_pred,q_score,q_source=qh[minute]
        fit=cumulative_cutoff(dev,d_pred,d_score,dp,da,target)
        dn=np.zeros(len(da),bool) if fit is None else ~da & np.isfinite(d_score) & (d_score<=fit['threshold'])
        qn=np.zeros(n,bool) if fit is None else ~qa & np.isfinite(q_score) & (q_score<=fit['threshold'])
        old=qa.copy();old_pred=qp.copy();old_time=qt.copy()
        dp[dn]=d_pred[dn];da|=dn
        qp[qn]=q_pred[qn];qt[qn]=minute;qs[qn]=q_source[qn];qa|=qn
        assert np.array_equal(qp[old],old_pred[old]) and np.array_equal(qt[old],old_time[old])
        assert not (old & ~qa).any()
        result[minute]=dict(prediction=qp.copy(),action=qa.copy(),release_time=qt.copy(),source_time=qs.copy())
        selections.append(dict(minute=minute,target=target,fit=fit))
    return result,selections


def smoke():
    old=(np.array([55.,np.nan]),np.array([.1,np.inf]),np.array([5,-1]))
    h=update_history(old,np.array([80.,65.]),np.array([.2,.05]),np.array([True,True]),10)
    np.testing.assert_array_equal(h[0],[55,65]);np.testing.assert_array_equal(h[2],[5,10])
    dev,q,idx=m.load_context('paper','HZ_to_SA',0,0)
    dh,qh,_=histories(dev,q)
    poisoned=dict(q,truth=np.full(len(idx),999),ma15=np.full(len(idx),1e6))
    dd,qq,_=histories(dev,poisoned)
    for minute in [5,10]:
        for a,b in zip(qh[minute],qq[minute]):np.testing.assert_array_equal(a,b)
    d,*_=c.context();saved=d['y'][idx].copy()
    try:
        d['y'][idx]=1e6
        dp,qp,_=histories(dev,q)
        for minute in TIMES:
            for a,b in zip(qh[minute],qp[minute]):np.testing.assert_array_equal(a,b)
        for target in AGREEMENTS:
            a,af=replay(dev,dh,qh,target);b,bf=replay(dev,dp,qp,target);assert af==bf
            for minute in TIMES:
                for key in a[minute]:np.testing.assert_array_equal(a[minute][key],b[minute][key])
    finally:d['y'][idx]=saved
    m.write_json(ROOT/'SMOKE.json',dict(old_prediction_available=True,lower_estimated_risk_only=True,
        no_future_15min_effect_on_5_or_10=True,no_outer_label_effect=True,calibration_groups_disjoint=True,
        first_report_prediction_locked=True,cumulative_coverage_monotone=True,upstream_dependency_unchanged=True))
    print('HISTORY_SMOKE_PASS',flush=True)


def evaluate():
    d,*_=c.context();rows=[];per=[];calibration=[];selection=[]
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            reps=20 if protocol=='paper' else 3
            test=np.asarray(c.paper_spec(arm,1000)['test'] if protocol=='paper' else c.context()[6])
            where={int(i):j for j,i in enumerate(test)};shape=(reps,len(test));truth=d['y'][test,2]
            hp={minute:np.full(shape,np.nan) for minute in TIMES}
            hs={minute:np.full(shape,np.inf) for minute in TIMES}
            ht={minute:np.full(shape,-1,int) for minute in TIMES}
            seq={(a,minute):dict(prediction=np.full(shape,np.nan),action=np.zeros(shape,bool),
                                release_time=np.full(shape,-1,int),source_time=np.full(shape,-1,int)) for a in AGREEMENTS for minute in TIMES}
            seen=np.zeros(shape,int)
            for rep in range(reps):
                for fi in range(1 if protocol=='paper' else 5):
                    dev,q,idx=m.load_context(protocol,arm,rep,fi);jj=[where[int(i)] for i in idx];seen[rep,jj]+=1
                    dh,qh,cal=histories(dev,q)
                    calibration.append(dict(protocol=protocol,arm=arm,repeat=rep,fold=fi,fits=cal))
                    for minute in TIMES:hp[minute][rep,jj],hs[minute][rep,jj],ht[minute][rep,jj]=qh[minute]
                    for target in AGREEMENTS:
                        result,fit=replay(dev,dh,qh,target)
                        selection.append(dict(protocol=protocol,arm=arm,repeat=rep,fold=fi,target=target,selections=fit))
                        for minute in TIMES:
                            for key in result[minute]:seq[target,minute][key][rep,jj]=result[minute][key]
            assert (seen==1).all()
            for mode,targets in [('coverage',COVERAGES),('agreement_target',AGREEMENTS)]:
                for target in targets:
                    previous=None;previous_risk=None
                    for minute in TIMES:
                        predictions=hp[minute] if mode=='coverage' else seq[target,minute]['prediction']
                        actions=[];valids=[];met=[];risk=[]
                        for rep in range(reps):
                            if mode=='coverage':
                                action,k=m.exact_quota(hs[minute][rep],np.isfinite(hs[minute][rep]),d['ids'][test],target)
                            else:action=seq[target,minute]['action'][rep];k=None
                            feasible=action is not None;valids.append(feasible)
                            rr=dict(protocol=protocol,arm=arm,mode=mode,target=target,minute=minute,repeat=rep,
                                    feasible=feasible,requested_count=k)
                            if feasible:
                                mm=m.metrics(predictions[rep],truth,action);rr.update(mm);actions.append(action)
                                source=ht[minute][rep] if mode=='coverage' else seq[target,minute]['source_time'][rep]
                                rr['prediction_source_counts']={str(t):int(np.sum(action & (source==t))) for t in TIMES}
                                if mode=='agreement_target':
                                    hit=mm['agreement'] is not None and mm['agreement']+1e-12>=target
                                    rr['actual_target_met']=hit;met.append(hit)
                                else:risk.append(float(hs[minute][rep][action].sum()))
                            per.append(rr)
                        row=dict(protocol=protocol,arm=arm,mode=mode,target=target,minute=minute,repeats=reps,
                                 feasible_all_repeats=all(valids),feasible_repeats=sum(valids))
                        if all(valids):
                            actions=np.asarray(actions)
                            row.update(m.metrics(predictions,truth,actions))
                            sources=ht[minute] if mode=='coverage' else seq[target,minute]['source_time']
                            row['prediction_source_counts']={str(t):int(np.sum(actions & (sources==t))) for t in TIMES}
                            row['using_earlier_prediction']=int(np.sum(actions & (sources<minute)))
                            if mode=='agreement_target':
                                row['actual_target_met']=row['agreement'] is not None and row['agreement']+1e-12>=target
                                row['target_met_repeats']=sum(met)
                                if previous is not None:assert not (previous & ~actions).any()
                                previous=actions.copy()
                            else:
                                assert np.all(actions.sum(1)==int(np.floor(target*len(test)+.5)))
                                if previous_risk is not None:assert np.all(np.asarray(risk)<=previous_risk+1e-10)
                                previous_risk=np.asarray(risk)
                                row['estimated_selected_risk_sum']=float(np.sum(risk))
                            np.savez_compressed(PRIVATE/f'{protocol}_{arm}_{mode}_{target:.2f}_{minute}.npz',
                                prediction=predictions,action=actions,source_time=sources,test=test,
                                **({} if mode=='coverage' else {'release_time':seq[target,minute]['release_time']}))
                        else:row.update(coverage=None,agreement=None,actual_target_met=None)
                        rows.append(row)
            print('HISTORY_COMPLETE',protocol,arm,flush=True)
    m.csv_write(ROOT/'SOURCE_DATA.csv',rows);m.csv_write(ROOT/'PER_REPEAT.csv',per)
    m.write_json(ROOT/'METRICS.json',rows)
    m.write_json(PRIVATE/'CALIBRATION.json',calibration);m.write_json(PRIVATE/'SELECTION.json',selection)
    m.write_json(ROOT/'INPUT_MANIFEST.json',m.INPUTS)
    return rows


def plot(rows):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    lookup={(r['protocol'],r['arm'],r['mode'],r['target'],r['minute']):r for r in rows}
    for mode,levels in [('coverage',COVERAGES),('agreement_target',AGREEMENTS)]:
        capability=mode=='coverage';fig,axs=plt.subplots(2,3,figsize=(13.4,7.8),sharex=True,sharey=True)
        for ri,protocol in enumerate(['paper','all301']):
            for ai,arm in enumerate(c.MAIN):
                ax=axs[ri,ai]
                for li,target in enumerate(levels):
                    rr=[lookup[protocol,arm,mode,target,minute] for minute in TIMES]
                    key='agreement' if capability else 'coverage'
                    values=[np.nan if r.get(key) is None else 100*r[key] for r in rr]
                    ax.plot(TIMES,values,color=COLORS[li],lw=1.7)
                    for minute,value,r in zip(TIMES,values,rr):
                        if np.isfinite(value):
                            fill=COLORS[li] if capability or r.get('actual_target_met') else 'white'
                            ax.scatter(minute,value,marker=MARKERS[li],s=35,facecolors=fill,edgecolors=COLORS[li],lw=1.2,zorder=3)
                ax.set_title(m.LABELS[arm]+' | '+('原60/201' if protocol=='paper' else '301条内部评价'),fontsize=10)
                ax.set(xlim=(4.3,15.7),ylim=(-3,105),xticks=TIMES,yticks=[0,20,40,60,80,100],xlabel='观测时间（分钟）')
                if ai==0:ax.set_ylabel('三分类一致率（%）' if capability else '累计覆盖率（%）')
                ax.tick_params(labelbottom=True);ax.grid(axis='y',color='#E3E7EA',lw=.6);ax.set_axisbelow(True)
        fig.suptitle('使用截至当前的全部预测：固定覆盖率下的一致率' if capability else '保留早期已报告结果：固定一致率目标下的累计覆盖率',fontsize=14,y=.985,fontweight='bold')
        handles=[Line2D([0],[0],color=COLORS[i],marker=MARKERS[i],lw=1.7,label=f'{100*q:g}%') for i,q in enumerate(levels)]
        fig.legend(handles=handles,title='固定覆盖率' if capability else '固定开发侧一致率目标',loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=10)
        fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.17,hspace=.38,wspace=.16)
        note=('10分钟可沿用5分钟预测，15分钟可沿用5/10分钟预测；每次重新选择相同数量，属于能力比较。'
              if capability else '首次报告锁定，后续只增加报告；空心点表示实际一致率未达目标或无报告，累计覆盖率不下降。')
        fig.text(.5,.105,note,ha='center',fontsize=8.5)
        fig.text(.5,.068,'上排201条×20种子；下排301条×3次重复。历史预测选择及阈值仅用开发侧信息；一致率不代替FNR。',ha='center',fontsize=8)
        fig.text(.5,.03,'基于实测U-TEG信号的回顾性探索。',ha='center',fontsize=7.8,color='#555555')
        name='agreement_vs_time_history_fixed_coverage' if capability else 'coverage_vs_time_locked_fixed_agreement'
        for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'{name}.{ext}',dpi=300,facecolor='white')
        plt.close(fig)


def report(rows):
    lines=['# 三档历史信息可用比较','',
        '第一张图是用户确认的能力比较：截至10分钟可以沿用5分钟预测，截至15分钟可以沿用5/10分钟预测，再选择固定数量。它允许更换评价子集，不代表撤回已发出的临床报告。',
        '第二张图是序贯回放：5分钟已报告的预测锁定；10和15分钟只处理剩余记录。新报告也可以采用保存的早期预测。',
        '', '## 如何选历史预测', '',
        '先把各时点不同单位的可信分数，在开发数据上转换为可比较的错误风险分数。采用单调回归，开发侧按已有三折交叉计算，外层应用时用全部开发记录拟合。相同风险以开发分数百分位区分，仍相同则保留较早预测。',
        '每条记录只在估计风险更低时采用较晚预测。固定覆盖率用相同数量的最低风险记录；固定一致率目标在开发侧选择累计报告的阈值。测试真值不参与选时点、选记录或选阈值。',
        '新增信息使旧方案仍然可选，但风险估计可能有误，所以不能保证留出数据实际一致率单调增加。本实现没有取测试一致率的累计最大值。',
        '原有开发OOF单模型分数与应用侧模型共识分数的接口差异仍存在；这里的风险校准是探索性比较方法，不是校准或安全保证。',
        '[单调回归实现说明](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)',
        '', '## 单源的完整数值','',
        '| 模式 | 评价 | 固定水平 | 分钟 | 一致率 | 覆盖率 | 报告/错误 | FNR | 使用早期预测 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def pct(x):return '未定义' if x is None else f'{100*x:.2f}%'
    for r in rows:
        if r['arm']!='HZ_to_SA':continue
        if not r['feasible_all_repeats']:
            lines.append(f"| {r['mode']} | {r['protocol']} | {100*r['target']:g}% | {r['minute']} | 不可达 | — | — | — | — |")
        else:
            lines.append(f"| {r['mode']} | {r['protocol']} | {100*r['target']:g}% | {r['minute']} | {pct(r['agreement'])} | {pct(r['coverage'])} | {r['reported']}/{r['incorrect']} | {r['fn']}/{r['fn_denominator']}（{pct(r['fnr'])}） | {r['using_earlier_prediction']} |")
    lines+=['', 'FNR分母为已报告的真实异常；未报告不能算正确检出。低高MA召回及以全部该类记录为分母的正确报告比例均见SOURCE_DATA.csv。重复事件不是独立患者。',
        'B为原始U-TEG数据。5分钟和10分钟分析使用相应时段的实测信号，完整U-TEG记录为15分钟。本分析为回顾性探索。',
        '没有修改原论文、原始输入或已保存的前几版结果。','']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main(stage):
    initialize()
    manuscript=c.WS/'outputs/uteg_nc_goal_20261030/manuscript'
    before={str(p):c.sha(p) for p in manuscript.rglob('*') if p.is_file()}
    if stage in ['smoke','run']:smoke()
    if stage=='run':rows=evaluate();plot(rows);report(rows)
    elif stage=='plot':rows=c.read(ROOT/'METRICS.json');plot(rows);report(rows)
    assert all(c.sha(Path(p))==h for p,h in before.items())
    m.write_json(ROOT/'QA.json',dict(manuscript_files_unchanged=len(before),three_levels_each=True,
        earlier_prediction_retained_as_option=True,exact_fixed_coverage=True,estimated_risk_nonincreasing=True,
        cumulative_coverage_monotone=True,reported_predictions_locked=True,heldout_agreement_not_forced_monotone=True,
        visual_qa='pending',acquired_signal_inputs=True))
    print('HISTORY_FINISHED',stage,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','plot']);args=ap.parse_args()
    with threadpool_limits(limits=1):main(args.stage)
