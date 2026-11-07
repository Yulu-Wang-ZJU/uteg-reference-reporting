"""Matched checkpoint diagnostics; no manuscript or active-gate replacement.

Reuse current scores and candidate restrictions. Exact quotas are label-blind
batch diagnostics. Agreement targets are fitted on development OOF predictions.
Early checkpoints use the corresponding measured U-TEG signals.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from . import ten_minute_consensus as t

c, s, e, check = t.c, t.s, t.e, t.check
ROOT = s.suite.ROOT / 'matched_checkpoint_curves_20261103'
PRIVATE = ROOT / 'private'
COVERAGES = [.01, .02, .05, .10, .20, .40, .60, .75, .90]
AGREEMENTS = [.80, .85, .90, .95, .975]
MINUTES = [5, 10, 15]
LABELS = dict(zip(c.MAIN, ['Target-only', '单源迁移', '双源迁移']))
COLORS = {5: '#647E91', 10: '#C18039', 15: '#3D7968'}
MARKERS = {5: 'o', 10: 's', 15: '^'}
INPUTS = {}


def read_npz(path):
    INPUTS[str(path)] = c.sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write_json(path, obj):
    t.gates.clean_write(path, obj)


def initialize():
    s.suite.initialize()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    contract = dict(
        claim='Compare checkpoint report yield under matched quotas and development agreement targets; do not impose improvement.',
        backend='Python matplotlib', archetype='quantitative grid',
        coverage_targets=COVERAGES, agreement_targets=AGREEMENTS,
        scope='Separate checkpoint snapshots, NOT cumulative first-report trajectories. Existing sequential policies unchanged.',
        scores={'5': 'Worst LR error score across final and three development models; development OOF LR for threshold fit.',
                '10': 'Negative minimum boundary distance across final and three development models; development OOF distance for threshold fit.',
                '15': 'Negative boundary distance of final MA; development OOF distance for threshold fit.'},
        hard_candidates={'5': 'Finite score and all three development models agree with final predicted state.',
                         '10': 'Same class consensus, plus existing training-low-B normal veto max(maximum,Q3+1.5IQR).',
                         '15': 'Finite current prediction and score; original boundary threshold is scanned, not a hard veto.'},
        interpretation='Comparison of the current checkpoint-specific score/candidate families, not an ablation of time alone or a claim that all ranks have the same risk calibration.',
        quota='k=floor(q*N+0.5) per evaluation repetition. Rank all hard-eligible records by score, ties by fixed SHA256 of record identity. For301, rank the assembled outer-heldout scores, without labels. No per-fold rounding.',
        infeasible='If any repetition lacks k hard-eligible records, pooled quota point is not plotted; per-repeat results and support counts retained. No averaging over a favorable subset.',
        quota_interpretation='Label-blind retrospective batch quota, not a validated deployable gate or guaranteed agreement.',
        agreement_fit='One shared development-OOF cutoff maximizes support with agreement>=q; at least max(10,ceil(.1*n)) records,5 proxy groups,2 folds. Ties accepted as a block. NO additional zero-FN constraint in this comparison; FNR is reported. Query consensus remains an additional application check, as in existing candidates.',
        additional_constraint_comparison='Preliminary version additionally requiring zero development FN retained separately in additional_zero_fn_constraint; not the primary matched-agreement comparison.',
        heldout='All agreement, FNR and class recalls computed only after actions frozen. Actual agreement is not forced to equal target. Open points mark failure to reach pooled target; report repetition-level attainment too.',
        protocol_paper='Original60 development/201 evaluation,20 seeds;4020 record-seed events.',
        protocol_all301='Frozen grouped5outer folds,3 inner-partition repeats;903 record-repeat events.',
        statistics='Pooled count ratios, no significance/CI claims; repetitions are not independent patients.',
        source_data='Aggregate and per-repeat CSV plus private threshold/decision records.',
        provenance='U-TEG MA and curves are original measured values. The 5- and 10-minute inputs use the first 300 and 600 samples of each 900-sample recording.',
        B_from_original_data=True, valid_UTEG_only_validation=False, manuscript_changed=False)
    path=ROOT/'CONTRACT.json'
    if path.exists() and c.read(path)!=contract:
        assert (ROOT/'additional_zero_fn_constraint/CONTRACT.json').exists()
    write_json(path, contract)


def load_context(protocol, arm, rep, fi):
    name = f'base_{protocol}_{arm}_{rep if protocol=="paper" else fi}' if protocol=='paper' or rep==0 else f'repeat301_{arm}_{rep}_{fi}'
    z = read_npz(s.PRIVATE/(name+'.npz'))
    dev={k[4:]:v for k,v in z.items() if k.startswith('dev_')}
    query={k[6:]:v for k,v in z.items() if k.startswith('query_')}
    idx=z['test']
    d,*_=c.context()
    assert not set(d['groups'][dev['rows']]) & set(d['groups'][idx])
    members = read_npz(t.PRIVATE/f'members_{protocol}_{arm}_{rep}_{fi}.npz')['ma']
    np.testing.assert_array_equal(e.state(members), query['member_pred10'])
    dev['B10']=c.feature(dev['rows'],idx,600)[dev['rows'],0]
    query['B10']=c.feature(dev['rows'],idx,600)[idx,0]
    query['member_ma10']=members
    # Complete missing15-minute OOF mapping scores using exactly the existing
    # development splits and frozen mapper; no parameter/feature selection.
    p=PRIVATE/f'oof15_{protocol}_{arm}_{rep}_{fi}.npz'
    if p.exists():
        a=read_npz(p)
        np.testing.assert_array_equal(a['rows'],dev['rows'])
        dev['ma15']=a['prediction']
    else:
        dev['ma15']=np.full(len(dev['rows']),np.nan)
        seen=np.zeros(len(dev['rows']),int)
        for pi,train in enumerate(t.parts(protocol,arm,rep,fi)):
            use=np.flatnonzero(dev['fold']==pi)
            hold=dev['rows'][use]
            assert set(train)|set(hold)==set(dev['rows'])
            assert not set(d['groups'][train]) & set(d['groups'][hold])
            assert not set(d['groups'][train]) & set(d['groups'][idx])
            dev['ma15'][use]=c.predict(arm,900,train,hold)[0]
            seen[use]+=1
        assert (seen==1).all() and np.isfinite(dev['ma15']).all()
        np.savez_compressed(p,rows=dev['rows'],prediction=dev['ma15'])
        INPUTS[str(p)]=c.sha(p)
    dev['pred15']=e.state(dev['ma15'])
    return dev,query,idx


def stage(dev, query, minute):
    pred=e.state(query['ma'+str(minute)])
    eligible=np.isfinite(query['ma'+str(minute)])
    dv=np.isfinite(dev['ma'+str(minute)])
    if minute==5:
        ds=dev['LR_5']
        qs=np.maximum(query['LR_5'],np.max(query['member_LR_5'],axis=0))
        eligible &= np.all(query['member_pred5']==pred,axis=0)
    elif minute==10:
        ds=t.distance(dev['ma10'])
        qs=np.maximum(t.distance(query['ma10']),np.max(t.distance(query['member_ma10']),axis=0))
        eligible &= np.all(query['member_pred10']==pred,axis=0)
        guard=t.guard_fit(dev,True)
        eligible &= t.guard_apply(guard,query)
        dv &= t.guard_apply(guard,dev)
    else:
        ds=t.distance(dev['ma15']);qs=t.distance(query['ma15'])
    return dict(dev_score=ds,dev_pred=dev['pred'+str(minute)],dev_eligible=dv&np.isfinite(ds),
                score=qs,eligible=eligible&np.isfinite(qs),prediction=query['ma'+str(minute)])


def fit_target(score,pred,truth,eligible,groups,folds,target,zero_fn=False):
    """Only development arrays; no query reference argument exists."""
    minimum=max(10,int(np.ceil(.1*len(pred))))
    best=None
    for threshold in np.unique(score[eligible & np.isfinite(score)]):
        action=eligible & np.isfinite(score) & (score<=threshold)
        n=int(action.sum())
        if n<minimum or len(np.unique(groups[action]))<5 or len(np.unique(folds[action]))<2:
            continue
        correct=int(np.sum(pred[action]==truth[action]))
        fn=int(np.sum(action & (truth!=0) & (pred==0)))
        if correct/n+1e-12<target or (zero_fn and fn):
            continue
        candidate=dict(threshold=float(threshold),selected=n,correct=correct,fn=fn,agreement=correct/n)
        if best is None or n>best['selected']:
            best=candidate
    return best


def apply_target(score,eligible,fit):
    return np.zeros(len(score),bool) if fit is None else eligible & np.isfinite(score) & (score<=fit['threshold'])


def exact_quota(score,eligible,ids,target):
    k=int(np.floor(target*len(score)+.5))
    available=np.flatnonzero(eligible & np.isfinite(score))
    if len(available)<k:
        return None,k
    tie=np.asarray([hashlib.sha256(('uteg-quota-v1|'+str(i)).encode()).hexdigest() for i in ids])
    order=available[np.lexsort((tie[available],score[available]))]
    action=np.zeros(len(score),bool);action[order[:k]]=True
    assert action.sum()==k
    return action,k


def metrics(pred,truth,action):
    m=check.score(pred,truth,action)
    cm=np.asarray(m['confusion'])
    for i,name in [(0,'low'),(2,'high')]:
        den=m['all_true_class_counts'][i]
        m[name+'_correct_yield_full_class']=None if not den else int(cm[i,i])/den
    return m


def smoke():
    # Equal scores use a deterministic identity-only order, not correctness.
    scores=np.ones(20);eligible=np.ones(20,bool);ids=np.arange(20)
    a,k=exact_quota(scores,eligible,ids,.2)
    b,_=exact_quota(scores[::-1],eligible[::-1],ids[::-1],.2)
    np.testing.assert_array_equal(a,b[::-1]);assert a.sum()==k==4
    assert exact_quota(scores,np.arange(20)<3,ids,.2)[0] is None
    # Tied thresholds cannot select only the correct members of a tied block.
    p=np.zeros(20,int);truth=p.copy();truth[-1]=1
    assert fit_target(scores,p,truth,eligible,ids,ids%3,.975) is None
    relaxed=fit_target(scores,p,truth,eligible,ids,ids%3,.9)
    assert relaxed['selected']==20 and relaxed['fn']==1
    assert fit_target(scores,p,truth,eligible,ids,ids%3,.9,zero_fn=True) is None
    dev,q,idx=load_context('paper','HZ_to_SA',0,0)
    d,*_=c.context();saved=d['y'][idx].copy()
    for minute in MINUTES:
        a=stage(dev,q,minute)
        for target in AGREEMENTS:
            fit=fit_target(a['dev_score'],a['dev_pred'],dev['truth'],a['dev_eligible'],dev['groups'],dev['fold'],target)
            action=apply_target(a['score'],a['eligible'],fit)
            try:
                d['y'][idx]=1e6
                poison=dict(q,truth=np.full(len(idx),999))
                b=stage(dev,poison,minute)
                ff=fit_target(b['dev_score'],b['dev_pred'],dev['truth'],b['dev_eligible'],dev['groups'],dev['fold'],target)
                assert fit==ff
                np.testing.assert_array_equal(action,apply_target(b['score'],b['eligible'],ff))
            finally:
                d['y'][idx]=saved
        for target in COVERAGES:
            mask,k=exact_quota(a['score'],a['eligible'],d['ids'][idx],target)
            if mask is not None:
                assert mask.sum()==k and not (mask & ~a['eligible']).any()
    write_json(ROOT/'SMOKE.json',dict(exact_quota=True,deterministic_label_blind_ties=True,
        unattainable_not_filled=True,development_ties_blocked=True,outer_label_poison_invariant=True,
        groups_disjoint=True,original_B_values_retained=True))
    print('MATCHED_SMOKE_PASS',flush=True)


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(t.gates.safe(v),ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,np.ndarray)) else v for k,v in row.items()})


def evaluate():
    assert (ROOT/'SMOKE.json').exists()
    d,*_=c.context();aggregate=[];per=[];fitlog=[];capacity=[]
    for protocol in ['paper','all301']:
        for arm in c.MAIN:
            runs=20 if protocol=='paper' else 3
            test=np.asarray(c.paper_spec(arm,1000)['test'] if protocol=='paper' else c.context()[6])
            where={int(i):j for j,i in enumerate(test)};truth=d['y'][test,2]
            pred={m:np.full((runs,len(test)),np.nan) for m in MINUTES}
            scores={m:np.full((runs,len(test)),np.inf) for m in MINUTES}
            eligible={m:np.zeros((runs,len(test)),bool) for m in MINUTES}
            target_actions={(m,q):np.zeros((runs,len(test)),bool) for m in MINUTES for q in AGREEMENTS}
            seen=np.zeros((runs,len(test)),int)
            for rep in range(runs):
                for fi in range(1 if protocol=='paper' else 5):
                    dev,query,idx=load_context(protocol,arm,rep,fi)
                    jj=[where[int(i)] for i in idx];seen[rep,jj]+=1
                    for m in MINUTES:
                        st=stage(dev,query,m)
                        pred[m][rep,jj]=st['prediction'];scores[m][rep,jj]=st['score'];eligible[m][rep,jj]=st['eligible']
                        for target in AGREEMENTS:
                            fit=fit_target(st['dev_score'],st['dev_pred'],dev['truth'],st['dev_eligible'],dev['groups'],dev['fold'],target)
                            target_actions[m,target][rep,jj]=apply_target(st['score'],st['eligible'],fit)
                            fitlog.append(dict(protocol=protocol,arm=arm,repeat=rep,fold=fi,minute=m,target=target,fit=fit))
            assert (seen==1).all()
            for m in MINUTES:
                capacity.append(dict(protocol=protocol,arm=arm,minute=m,per_repeat_hard_eligible=eligible[m].sum(1).tolist(),
                                     minimum_hard_eligible=int(eligible[m].sum(1).min()),records_per_repeat=len(test)))
                for mode,targets in [('coverage',COVERAGES),('agreement_target',AGREEMENTS)]:
                    previous=None
                    for target in targets:
                        masks=[];available=[];met=[]
                        for rep in range(runs):
                            if mode=='coverage':
                                action,k=exact_quota(scores[m][rep],eligible[m][rep],d['ids'][test],target)
                            else:
                                action=target_actions[m,target][rep];k=None
                            valid=action is not None;available.append(valid)
                            row=dict(protocol=protocol,arm=arm,minute=m,mode=mode,target=target,repeat=1000+rep if protocol=='paper' else rep+1,
                                     feasible=valid,requested_count=k,hard_eligible=int(eligible[m][rep].sum()))
                            if valid:
                                mm=metrics(pred[m][rep],truth,action);row.update(mm);masks.append(action)
                                hit=mm['agreement'] is not None and mm['agreement']+1e-12>=target
                                if mode=='agreement_target':row['actual_target_met']=hit;met.append(hit)
                            per.append(row)
                        row=dict(protocol=protocol,arm=arm,minute=m,mode=mode,target=target,repeats=runs,
                                 feasible_repeats=sum(available),feasible_all_repeats=all(available))
                        if all(available):
                            masks=np.asarray(masks)
                            mm=metrics(pred[m],truth,masks);row.update(mm)
                            if mode=='coverage':
                                expected=int(np.floor(target*len(test)+.5))
                                assert np.all(masks.sum(1)==expected)
                                row['requested_count_per_repeat']=expected
                                row['actual_target_met']=None
                            else:
                                row['actual_target_met']=mm['agreement'] is not None and mm['agreement']+1e-12>=target
                                row['target_met_repeats']=sum(met)
                                if previous is not None:assert not (masks & ~previous).any()
                                previous=masks.copy()
                            np.savez_compressed(PRIVATE/f'action_{protocol}_{arm}_{m}_{mode}_{target:.3f}.npz',action=masks)
                        else:
                            row.update(coverage=None,agreement=None,actual_target_met=None)
                        aggregate.append(row)
            np.savez_compressed(PRIVATE/f'predictions_{protocol}_{arm}.npz',test=test,
                                **{f'prediction_{m}':pred[m] for m in MINUTES},
                                **{f'score_{m}':scores[m] for m in MINUTES},
                                **{f'eligible_{m}':eligible[m] for m in MINUTES})
            print('MATCHED_COMPLETE',protocol,arm,flush=True)
    csv_write(ROOT/'SOURCE_DATA.csv',aggregate);csv_write(ROOT/'PER_REPEAT.csv',per)
    write_json(PRIVATE/'DEVELOPMENT_THRESHOLDS.json',fitlog)
    write_json(ROOT/'CANDIDATE_CAPACITY.json',capacity)
    write_json(ROOT/'METRICS.json',aggregate)
    write_json(ROOT/'INPUT_MANIFEST.json',INPUTS)
    return aggregate


def draw_panel(ax,rows,protocol,arm,mode,title):
    for minute in MINUTES:
        rr=[r for r in rows if r['protocol']==protocol and r['arm']==arm and r['mode']==mode and r['minute']==minute]
        x=np.array([100*r['target'] for r in rr])
        key='agreement' if mode=='coverage' else 'coverage'
        y=np.array([np.nan if r.get(key) is None else 100*r[key] for r in rr])
        color=COLORS[minute];marker=MARKERS[minute]
        ax.plot(x,y,color=color,lw=1.5,label=f'{minute}分钟')
        if mode=='coverage':
            ax.scatter(x,y,color=color,s=24,marker=marker,zorder=4)
            for xx,r in zip(x,rr):
                if not r['feasible_all_repeats']:
                    ax.plot(xx,-5-(MINUTES.index(minute)*3),marker='x',color=color,ms=4,clip_on=False)
        else:
            for xx,yy,r in zip(x,y,rr):
                ax.scatter(xx,yy,edgecolors=color,facecolors=color if r.get('actual_target_met') else 'white',s=36,marker=marker,zorder=4,linewidths=1.2)
    ax.set_title(title,fontsize=10,pad=9)
    ax.set_ylim((-13,105) if mode=='coverage' else (-4,105))
    ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',color='#E5E8EB',lw=.6);ax.set_axisbelow(True)
    if mode=='coverage':
        ax.set_xscale('log');ax.set_xlim(.8,110);ax.set_xticks([1,2,5,10,20,40,75])
        ax.set_xticklabels(['1','2','5','10','20','40','75'])
        ax.set_xlabel('共同覆盖率目标（%，对数刻度）');ax.set_ylabel('实际三分类一致率（%）')
    else:
        ax.set_xlim(78.5,99);ax.set_xticks([80,85,90,95,97.5]);ax.set_xticklabels(['80','85','90','95','97.5'])
        ax.set_xlabel('开发侧一致率目标（%）');ax.set_ylabel('留出评价覆盖率（%）')


def plot(rows):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False})
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'{name}.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for i,protocol in enumerate(['paper','all301']):
        label='原60条开发 / 201条评价 × 20种子' if protocol=='paper' else '301条内部评价 × 3次重复'
        for j,mode in enumerate(['coverage','agreement_target']):
            draw_panel(axs[i,j],rows,protocol,'HZ_to_SA',mode,label)
    fig.suptitle('单源：在相同报告数量或相同一致率目标下比较',fontsize=14,y=.985,fontweight='bold')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
    fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.18,hspace=.43,wspace=.24)
    fig.text(.5,.105,'左：同一评价重复严格匹配报告数；底部×表示至少一次重复候选不足。右：实心=汇总实际一致率达标，空心=未达标或无报告。',ha='center',fontsize=8)
    fig.text(.5,.068,'各时点单独比较，含门规则差异；不是累计首次报告曲线。一致率达标不代表FNR达标，异常指标另见数值表。',ha='center',fontsize=8)
    fig.text(.5,.031,'基于实测U-TEG信号的回顾性探索；重复事件不是独立患者。',ha='center',fontsize=7.5,color='#555555')
    save(fig,'single_source_matched_curves')
    for mode in ['coverage','agreement_target']:
        fig,axs=plt.subplots(2,3,figsize=(14,8))
        for i,protocol in enumerate(['paper','all301']):
            for j,arm in enumerate(c.MAIN):
                draw_panel(axs[i,j],rows,protocol,arm,mode,LABELS[arm]+' | '+('原60/201' if protocol=='paper' else '301条内部评价'))
        fig.suptitle('相同覆盖率下的一致率' if mode=='coverage' else '相同开发侧一致率目标下的覆盖率',fontsize=14,y=.985,fontweight='bold')
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3)
        fig.subplots_adjust(left=.065,right=.985,top=.86,bottom=.17,hspace=.43,wspace=.24)
        note='底部×：至少一次重复硬条件候选不足；该点不按部分重复计算。' if mode=='coverage' else '实心：汇总实际一致率达到目标；空心：未达到或无报告。实际一致率及逐次达标次数见数据表。'
        fig.text(.5,.09,note+' 上排201×20；下排301×3。',ha='center',fontsize=8)
        fig.text(.5,.05,'各时点独立快照；含分数/候选条件差异。一致率不代替FNR。当前输入版本的探索性回放，非独立临床验证。',ha='center',fontsize=8,color='#555555')
        save(fig,'fixed_'+mode+'_all_workflows')


def report(rows):
    lines=['# 相同覆盖率与相同开发侧一致率目标的比较','',
        '本次只新增比较分析，没有覆盖现有可信门、序贯流程或论文。5、10、15分钟为单独快照，不用后期结果替换早期已报告结果。',
        '', '## 如何保持一致', '',
        '- 固定覆盖率：同一重复按完整评价记录数取k=floor(qN+0.5)，仅在硬条件候选内按既有分数排序。分数并列按固定身份哈希排序，不读取真实标签。301条先拼齐5个外层留出，再统一取k，不逐折取整。',
        '- 候选不足：该重复不可达；若任一重复不可达，汇总曲线缺点并标×，不只平均成功重复。',
        '- 固定一致率目标：只在开发OOF选择阈值，统一最低样本/组/折支持，不额外强制开发侧零漏报。FNR如实单列。完整同分组一起纳入，不能按答案打破并列。留出数据的实际一致率可能不达目标，空心标记明确显示。',
        '- 为扫描多个目标，本次统一使用共享阈值选点；原5分钟是分预测类别选点、原15分钟是分位数门。因此本图是新的比较协议，不能把其中放行数当成原序贯规则的放行数，也不能认为每个比较点都满足原安全性要求。',
        '- 额外要求开发侧零漏报的初步版本保留在additional_zero_fn_constraint。该版本10分钟90%目标与此前对应候选逐条一致；它回答的是多加一项安全约束后的表现，不替代主图的一致率匹配。',
        '- 固定覆盖率是批量排序诊断，不代表这些点已满足可信门的一致率/FNR要求。固定开发目标也不等于固定测试实际一致率。',
        '- 5分钟沿用最差LR风险分数及三模型类别一致；10分钟沿用最差边界距离及类别一致/低MA幅值保护；15分钟扫描原边界距离。不同时间还含门规则差异，不能把全部变化归因于增加观测。',
        '- 开发OOF只有单个留出模型分数，应用侧5/10分钟还加模型共识检查；这是现有方案的接口差异，本次保留并通过留出结果检查，不称为校准保证。',
        '', '## 单源：固定覆盖率', '',
        '| 评价 | 目标覆盖率 | 5分钟一致率 | 10分钟一致率 | 15分钟一致率 |','|---|---:|---:|---:|---:|']
    def fmt(v):return '未定义' if v is None else f'{100*v:.2f}%'
    for protocol in ['paper','all301']:
        for target in COVERAGES:
            rr=[next(r for r in rows if r['protocol']==protocol and r['arm']=='HZ_to_SA' and r['mode']=='coverage' and r['minute']==m and r['target']==target) for m in MINUTES]
            cells=[fmt(r['agreement']) if r['feasible_all_repeats'] else f"不可达（{r['feasible_repeats']}/{r['repeats']}次可达）" for r in rr]
            lines.append('| '+('原60/201' if protocol=='paper' else '301内部')+f' | {target*100:g}% | '+' | '.join(cells)+' |')
    lines += ['', '## 单源：固定开发一致率目标', '', '| 评价 | 目标 | 分钟 | 实际一致率 | 覆盖率 | 放行/错误 | FNR | 逐次达标 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['arm']!='HZ_to_SA' or r['mode']!='agreement_target':continue
        lines.append(f"| {r['protocol']} | {100*r['target']:g}% | {r['minute']} | {fmt(r['agreement'])} | {fmt(r['coverage'])} | {r['reported']}/{r['incorrect']} | {r['fn']}/{r['fn_denominator']}（{fmt(r['fnr'])}） | {r['target_met_repeats']}/{r['repeats']} |")
    lines += ['', '## 分母与证据边界', '',
        'Coverage=报告数/全部评价事件；Agreement=正确报告/报告数。FNR=已报告真实异常中判正常的比例，分母为零时未定义。low_recall/high_recall同样以已报告该真实类别为分母；low_correct_yield_full_class/high_correct_yield_full_class以全部该真实类别为分母，暂缓不能冒充正确识别。',
        '原60/201为201条记录×20种子；301内部为301条×3内层划分重复。均不是相应数量的独立患者，没有基于事件独立性的显著性或置信区间宣称。',
        'B为原始U-TEG数据。5分钟和10分钟分析使用相应时段的实测信号，完整U-TEG记录为15分钟。本分析为回顾性探索。',
        '未按测试答案选阈值、删除错误、修改标签，未强制一致率随时间上升。所有预先列出的目标与三种流程均保留。', '']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main(mode):
    initialize()
    manuscript=c.WS/'outputs/uteg_nc_goal_20261030/manuscript'
    before={str(p):c.sha(p) for p in manuscript.rglob('*') if p.is_file()}
    if mode in ['smoke','run']:smoke()
    if mode=='run':
        rows=evaluate();plot(rows);report(rows)
    elif mode=='plot':
        rows=c.read(ROOT/'METRICS.json');plot(rows);report(rows)
    assert all(Path(p).is_file() and c.sha(p)==h for p,h in before.items())
    write_json(ROOT/'QA.json',dict(manuscript_files_unchanged=len(before),no_test_label_threshold_selection=True,
        exact_quota_within_each_repetition=True,pooled_infeasible_points_not_silently_averaged=True,
        snapshot_not_cumulative=True,upstream_provenance_retained=True,visual_qa='pending',
        exports=['PNG','PDF','SVG','CSV']))
    print('MATCHED_FINISHED',mode,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','plot']);args=ap.parse_args()
    with threadpool_limits(limits=1):main(args.stage)
