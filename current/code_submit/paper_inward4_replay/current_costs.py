"""Finite development-selected local cost comparison on current fixed B."""
from contextlib import contextmanager
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c
from . import current_version as current
from .analysis import table,pct,num
from code_submit.ma_target_optimization.metrics import assess,selection_key

OUT=current.ROOT/'cost_comparison'
COSTS=[4.,8.,16.]


@contextmanager
def local_cost(value):
    old=c.selected.COST
    try:
        c.selected.COST=float(value);yield
    finally:c.selected.COST=old


def evaluate(train,query,inner,arm):
    # Warm source priors at exactly4 before changing local solver cost.
    assert c.selected.COST==4.
    c.source_model(arm,900)
    values=[]
    for cost in COSTS:
        with local_cost(cost):values.append(c.evaluate(train,query,inner,arm,900))
    d,*_=c.context()
    metrics=[assess(d['y'][v['train'],2],v['inner']) for v in values]
    chosen=min(range(len(COSTS)),key=lambda j:(selection_key(metrics[j]),j))
    return values,chosen,metrics


def run():
    current.activate();d,_,_,_,_,folds,local=c.context();OUT.mkdir(exist_ok=True)
    assert not (OUT/'RESULTS.json').exists(),'Refuse outer-result-driven rerun'
    protocol=dict(costs=COSTS,source_cost_fixed=4,changes='Only local inward-error multiplier. Source priors/weights,alpha.01,inputs,groups and gate unchanged.',
        selection='Within each outer training set use innerOOF selection_key: retain agreement>80 when possible, then lowerFNR/FPR and class error;5% is preference, not mandatory veto.',
        main_preserved=True,data_sha256=c.read(current.ROOT/'DATA_FREEZE.json')['current_dataset_sha256'],
        B_from_original_data=True,manuscript_changed=False)
    c.write(OUT/'FROZEN_PROTOCOL.json',protocol)
    # Exact baseline and test-label isolation on the first fold before scoring.
    f=folds[0];first={}
    for arm in c.MAIN:
        first[arm]=evaluate(f['A'],f['B'],f['inner'],arm)
        old=d['y'].copy()
        try:
            d['y'][f['B']]=1e6;repeat=evaluate(f['A'],f['B'],f['inner'],arm)
        finally:d['y'][:]=old
        assert first[arm][1:]==repeat[1:]
        for one,two in zip(first[arm][0],repeat[0]):
            for key in ['prediction','accepted','inner']:np.testing.assert_array_equal(one[key],two[key])
            assert one['threshold']==two['threshold'] and one['audits']==two['audits']
    c.write(OUT/'SMOKE_VERIFIED.json',dict(test_label_poison_leaves_selection_and_predictions_unchanged=True,source_cost_fixed4=True))
    pos={int(v):i for i,v in enumerate(local)};pp=np.full((3,3,len(local)),np.nan);aa=np.zeros_like(pp,bool)
    selected=np.full((3,len(local)),np.nan);action=np.zeros_like(selected,bool);audit=[]
    for fi,f in enumerate(folds):
        ix=[pos[int(v)] for v in f['B']]
        for j,arm in enumerate(c.MAIN):
            values,chosen,metrics=first[arm] if fi==0 else evaluate(f['A'],f['B'],f['inner'],arm)
            for k,v in enumerate(values):pp[j,k,ix]=v['prediction'];aa[j,k,ix]=v['accepted']
            selected[j,ix]=values[chosen]['prediction'];action[j,ix]=values[chosen]['accepted']
            audit.append(dict(fold=fi,arm=arm,selected_cost=COSTS[chosen],inner_metrics=metrics))
    with np.load(current.ROOT/'all301/T900.npz') as z:
        np.testing.assert_array_equal(pp[:,0],z['prediction']);np.testing.assert_array_equal(aa[:,0],z['accepted'])
    np.savez_compressed(OUT/'ALL301_PREDICTIONS.npz',predictions=pp,accepted=aa,selected=selected,selected_action=action,local=local)
    c.write(OUT/'ALL301_SELECTION.json',audit)
    result={'all301':{},'paper':{}}
    for j,arm in enumerate(c.MAIN):
        result['all301'][arm]={str(int(cost)):dict(all_report=c.summarize(pp[j,k],d['y'][local,2]),gate=c.summarize(pp[j,k],d['y'][local,2],aa[j,k])) for k,cost in enumerate(COSTS)}
        result['all301'][arm]['inner_selected']=dict(all_report=c.summarize(selected[j],d['y'][local,2]),gate=c.summarize(selected[j],d['y'][local,2],action[j]))
    for arm in c.MAIN:
        preds=[];actions=[];picks=[];selected_p=[];selected_a=[]
        for seed in c.SEEDS:
            spec=c.paper_spec(arm,seed)
            values,chosen,metrics=evaluate(spec['pool'],spec['test'],spec['positions'][3]['inner'],arm)
            preds.append([v['prediction'] for v in values]);actions.append([v['accepted'] for v in values])
            selected_p.append(values[chosen]['prediction']);selected_a.append(values[chosen]['accepted']);picks.append(dict(seed=seed,cost=COSTS[chosen],inner_metrics=metrics))
        p=np.asarray(preds);a=np.asarray(actions);test=spec['test'];y=d['y'][test,2]
        with np.load(current.ROOT/'primary'/f'{arm}_T900_B0.npz') as z:
            np.testing.assert_array_equal(p[:,0],z['prediction']);np.testing.assert_array_equal(a[:,0],z['accepted'])
        np.savez_compressed(OUT/f'PAPER_{arm}.npz',predictions=p,accepted=a,selected=selected_p,selected_action=selected_a,test=test,seed=c.SEEDS)
        c.write(OUT/f'PAPER_{arm}_SELECTION.json',picks)
        result['paper'][arm]={str(int(cost)):dict(all_report=c.summarize(p[:,k],y),gate=c.summarize(p[:,k],y,a[:,k])) for k,cost in enumerate(COSTS)}
        result['paper'][arm]['inner_selected']=dict(all_report=c.summarize(selected_p,y),gate=c.summarize(selected_p,y,selected_a))
    c.write(OUT/'RESULTS.json',result)
    lines=['# 当前新版B：局部异常误差惩罚4/8/16有限对照','',
        '当前主结果保持不变。外院先验固定inward4，仅改变本地修正对低MA向上、高MA向下误差的惩罚。所有三种固定结果均展示；inner_selected只由开发侧OOF选择，没有按外层结果挑选成本系数。','']
    for protocol_name in result:
        lines += [f'## {protocol_name}','',table(['流程','成本','一致率','FNR','FPR','低/高召回','MAE','R²','门覆盖'],[[arm,k,pct(m['all_report']['agreement']),pct(m['all_report']['fnr']),pct(m['all_report']['fpr']),pct(m['all_report']['low_recall'])+'/'+pct(m['all_report']['high_recall']),num(m['all_report']['mae']),num(m['all_report']['r2']),pct(m['gate']['coverage'])] for arm,methods in result[protocol_name].items() for k,m in methods.items()]),'']
    lines += ['输入仍沿用当前新版B及其既有来源；这些结果不改变其证据类别。所有成本4输出已逐值复现当前主表。低/高MA误差和混淆矩阵见RESULTS.json。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(OUT/'VERIFICATION.json',dict(cost4_exact_primary_replay=True,current_dataset_unchanged=c.sha(current.ROOT/'data/DATA15_CURRENT.npz')==protocol['data_sha256'],
        selection_on_development_only=True,outer_winner_not_promoted=True,manuscript_changed=False))
    print('CURRENT_COST_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
