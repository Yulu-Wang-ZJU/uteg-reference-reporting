"""Summaries from every completed round; SD of repetitions, never a CI."""
import experiment as e
import numpy as np
import json,csv,argparse,hashlib
from pathlib import Path
from collections import defaultdict
O=e.O;S=O/'summary';S.mkdir(exist_ok=True)

def flatten(x,prefix=''):
    out={}
    for k,v in x.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'_'))
        elif not isinstance(v,list):out[key]=v
    return out

def csvwrite(path,rows):
    rows=[flatten(x) for x in rows];keys=list(dict.fromkeys(k for x in rows for k in x))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def summarize(rows,keys):
    bank=defaultdict(list)
    for x in rows:bank[tuple(x.get(k) for k in keys)].append(flatten(x.get('test',x)))
    ans=[]
    for group,rr in bank.items():
        one=dict(zip(keys,group));one['rounds']=len(rr)
        for name in dict.fromkeys(k for x in rr for k in x):
            if name in keys or name in ['rep','time','budget','feasible','selection_alpha']:continue
            values=[x.get(name) for x in rr];valid=[float(v) for v in values if v is not None and isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v)]
            if not valid:continue
            one[name]=dict(n=len(valid),mean=float(np.mean(valid)),sd=float(np.std(valid,ddof=1)) if len(valid)>1 else None,min=float(np.min(valid)),max=float(np.max(valid)))
        ans.append(one)
    return ans

def write(name,rows,keys):
    sm=summarize(rows,keys);e.r.write(S/(name+'_rounds.json'),rows);csvwrite(S/(name+'_rounds.csv'),rows);e.r.write(S/(name+'_summary.json'),sm);csvwrite(S/(name+'_summary.csv'),sm);return sm

def readrows(pattern,filename='RESULTS.json'):
    return [v for p in sorted(O.glob(pattern)) if (p/'COMPLETE.json').exists() for v in json.loads((p/filename).read_text())]

def sequence(d):
    rows=[]
    for budget in [54,60]:
        for rep in range(100):
            paths=[O/f'primary/dev{budget}/time{h}/round_{rep:02d}' for h in [300,600,900]]
            if not all((p/'COMPLETE.json').exists() for p in paths):continue
            for arm in e.ARMS:
                z=[np.load(p/(arm+'.npz')) for p in paths];test=z[0]['test'];truth=d['medical'][test,2]
                assert all(np.array_equal(test,x['test']) for x in z)
                for rule in ['common_B','HGB','Logistic','Boundary','B95']:
                    for mode in ['first_report_locked','update_then_15min_current']:
                        pred=np.zeros(len(test));act=np.zeros(len(test),bool)
                        for h,x in zip([300,600,900],z):
                            now=x['action_'+rule];fresh=now&~act if mode=='first_report_locked' else now
                            if mode=='update_then_15min_current' and h==900:act=now.copy();pred=x['prediction'].copy()
                            else:pred[fresh]=x['prediction'][fresh];act|=now
                            rows.append(dict(budget=budget,rep=rep,arm=arm,rule=rule,mode=mode,time=h,test=e.n.extended_metrics(pred,truth,act)))
                for x in z:x.close()
    return write('time_schedules',rows,['budget','arm','rule','mode','time'])

def audit(d):
    checked=0;frozen=0;dispositions=0
    for folder in O.glob('primary/dev*/time*/round_*'):
        if not (folder/'COMPLETE.json').exists():continue
        rows=json.loads((folder/'RESULTS.json').read_text());sp=json.loads((folder/'SPLIT.json').read_text());tr=np.asarray(sp['pool']);te=np.asarray(sp['test']);assert not set(d['groups'][tr])&set(d['groups'][te])
        for part in sp['positions']:
            assert not set(d['groups'][part['A']])&set(d['groups'][part['B']])
            for f in part['inner']:assert not set(d['groups'][f['T']])&set(d['groups'][f['V']])
        for arm in e.ARMS:
            with np.load(folder/(arm+'.npz')) as z:
                assert np.array_equal(z['test'],te);p=e.r.state(z['prediction']);y=e.r.state(d['medical'][te,2])
                for row in [v for v in rows if v['arm']==arm]:
                    a=z['action_'+row['rule'].replace(' ','_')];m=row['test'];fn=int(np.sum(a&(y!=1)&(p==1)));fp=int(np.sum(a&(y==1)&(p!=1)));correct=int(np.sum(a&(y==p)));reported=int(a.sum());an=int(np.sum(a&(y!=1)));nn=int(np.sum(a&(y==1)))
                    assert (fn,fp,correct,reported)==(m['FN'],m['FP'],m['correct'],m['reported'])
                    for key,value in [('FNR',fn/an if an else None),('FPR',fp/nn if nn else None),('accuracy',correct/reported if reported else None),('reporting_rate',reported/len(a))]:
                        assert (value is None and m[key] is None) or (value is not None and abs(value-m[key])<1e-12)
                    assert row['disposition']['correct_released']+row['disposition']['incorrect_released']==reported;checked+=1;dispositions+=1
                budget=rows[0]['budget'];rep=rows[0]['rep'];h=rows[0]['time']
                if h==900 and arm in e.a.ARMS:
                    with np.load(e.LOCK/f'private/dev{budget}/round_{rep:02d}/{arm}.npz') as zz:
                        np.testing.assert_array_equal(zz['test_prediction'],z['prediction']);np.testing.assert_array_equal(zz['test_action'],z['action_common_B']);frozen+=1
    original=json.loads((e.LOCK/'MANUSCRIPT_BASELINE_HASHES.json').read_text());assert all(e.r.sha(Path(p))==v for p,v in original.items())
    assert all(e.r.sha(e.r.current.ROOT/'data'/k)==v for k,v in e.cal.protocol()['input_hashes'].items())
    out=dict(primary_metric_rows_recomputed=checked,disposition_rows_checked=dispositions,locked_prediction_and_action_pairs_identical=frozen,source_inputs_unchanged=True,manuscript_and_delivery_hashes_unchanged=True,primary_train_test_and_inner_groups_disjoint=True)
    e.r.write(S/'VERIFICATION.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');args=ap.parse_args();d,*_=e.init()
    primary=readrows('primary/dev*/time*/round_*');table=readrows('table1/round_*');nine=readrows('nine/round_*');budgets=readrows('budgets/round_*')
    p=write('Table2_and_time',primary,['budget','time','arm','rule']);t=write('Table1',table,['budget','arm','panel','method'])
    f2=write('Fig2_all_report',[v for v in primary if v['time']==900 and v['rule']=='All report'],['budget','arm'])
    confusion=[dict(budget=v['budget'],arm=v['arm'],rep=v['rep'],**{f'true_{i}_pred_{j}':v['test']['confusion'][i][j] for i in range(3) for j in range(3)}) for v in primary if v['time']==900 and v['rule']=='All report']
    write('Fig2_confusion_counts',confusion,['budget','arm'])
    dispos=[dict(budget=v['budget'],arm=v['arm'],rule=v['rule'],rep=v['rep'],**v['disposition']) for v in primary if v['time']==900];write('Fig3_dispositions',dispos,['budget','arm','rule'])
    curves=readrows('primary/dev*/time*/round_*','CURVES.json');write('Fig3_score_curves',[v for v in curves if v['time']==900],['budget','arm','rule','nominal_rate'])
    write('nine_configurations',nine,['mode','arm','rule']);write('nine_mapping',readrows('nine/round_*','MAPPING.json'),['mode','arm','stage'])
    combined=readrows('nine_controls/round_*')+[dict(v,mode='sampled90' if v['budget']==54 else 'full_local') for v in table]
    write('nine_mapping_and_calibration_controls',combined,['mode','arm','panel','method'])
    write('internal_mapping_and_calibration_controls',readrows('internal_controls/round_*'),['arm','panel','method'])
    write('label_budgets',budgets,['budget','arm'])
    for stage in ['linear','EXP3','EXP4','internal301']:write(stage,readrows(stage+'/round_*'),['budget','arm','rule'])
    sequence(d)
    ex=[dict(budget=v['budget'],arm=v['arm'],round=v['rep']+1,accuracy=v['test']['accuracy'],reporting_rate=v['test']['reporting_rate'],FNR=v['test']['FNR'],FPR=v['test']['FPR']) for v in primary if v['time']==900 and v['rule']=='common_B' and (v['test']['accuracy'] is None or v['test']['accuracy']<=.8 or v['test']['reporting_rate']<=.75 or v['test']['FNR']!=0)]
    csvwrite(S/'all_reference_threshold_exceptions.csv',ex)
    # Pair within the exact repeat and budget, never compare unpaired draws.
    pairing={(v['budget'],v['rep'],v['arm'],v['rule']):v for v in primary if v['time']==900};dif=[]
    for budget in [54,60]:
        for rep in range(100):
            for arm in e.ARMS:
                for rule in ['common_B','HGB','Logistic','Boundary','B95']:
                    aa=pairing.get((budget,rep,arm,rule));bb=pairing.get((budget,rep,arm,'All report'))
                    if aa is None or bb is None:continue
                    vals={k:(aa['test'][k]-bb['test'][k] if aa['test'][k] is not None and bb['test'][k] is not None else None) for k in ['accuracy','reporting_rate','FNR','FPR','correct','incorrect','FN','FP']};dif.append(dict(budget=budget,rep=rep,arm=arm,rule=rule,**vals))
    write('gate_minus_allreport_paired',dif,['budget','arm','rule'])
    counts={k:len(list((O/k).glob('**/COMPLETE.json'))) for k in ['table1','primary','budgets','nine','linear','EXP3','EXP4','internal301','nine_controls','internal_controls']};e.r.write(S/'COMPLETION.json',counts)
    if args.audit:audit(d)
    fmt=lambda v,pct=False:f"{v['mean']*(100 if pct else 1):.2f} ± {v['sd']*(100 if pct else 1):.2f}"
    lines=['# 固定方案100次重复：完整结果整理','', '正文、图表、PDF、公开附件及GitHub均未修改。所有已完成轮次保留。均值±样本标准差（ddof=1），不是95%置信区间。54与60分开；每轮181条测试记录，重复事件不等于独立病例。','', '## 15分钟固定方案','', '|开发量|方案|轮次|accuracy (%)|reporting rate (%)|FNR (%)|FPR (%)|','|---|---|---:|---:|---:|---:|---:|']
    for v in p:
        if v['time']==900 and v['rule']=='common_B':lines.append('|'+ '|'.join([str(v['budget']),v['arm'],str(v['rounds'])]+[fmt(v[k],True) for k in ['accuracy','reporting_rate','FNR','FPR']])+'|')
    lines+=['','## 门前MA映射','', '|开发量|方案|MAE (mm)|R²|低MA召回 (%)|高MA召回 (%)|FNR (%)|FPR (%)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for v in f2:lines.append('|'+ '|'.join([str(v['budget']),v['arm'],fmt(v['MAE']),fmt(v['R2'])]+[fmt(v[k],True) for k in ['low_recall','high_recall','FNR','FPR']])+'|')
    lines+=['','## 正文Table 1所需结果','', '|开发量|来源|面板|方法|R MAE|K MAE|MA MAE|角度MAE|','|---|---|---|---|---:|---:|---:|---:|']
    for v in t:lines.append('|'+ '|'.join([str(v[k]) for k in ['budget','arm','panel','method']]+[fmt(v[k]) for k in ['R_MAE','K_MAE','MA_MAE','angle_MAE']])+'|')
    lines+=['','## 解释时必须区分的事项','', '- 固定方案是normal专门检查加低/高MA分别选门，不能把它的数字直接写到旧HGB行。','- 60条设置中门前FN已为零。不能将此设置的零FN归因于门拦截了原本不存在的FN。','- 54条设置需对照门前/门后FNR、错误拦截数量和正确暂缓数量；完整配对差见gate_minus_allreport_paired。','- FNR分母是已放行的真实异常；FPR分母是已放行的真实normal。低/高召回同时提供条件召回和相对全体该类的保留正确率，二者不能混用。','- 早期时间和其他目标医院不预设测试零FN。开发不具备可行规则时保留该轮，报告率为0，accuracy/FNR缺失，不算作零错误。','- 301扩展为其独立内部敏感性：每轮271/301条、5折外层和3折内层，与54/60开发及181/201评价不同；全部结果是内部重复评价。','- 线性与RBF采用同一输入、成本4、alpha=0.01及尾部处理。表征敏感性以EXP3/EXP4已保存幅值进入当前B-only分支，不冒充旧的三/四输入模型实验。','- Fig.3单分数曲线保留各基线排序，复合规则只给实际工作点，不根据测试标签制造其最优包络。','- 首次报告锁定与更新能力对照单独保存；更新对照在15分钟用当时规则，不能说它保持累计报告率单调。','', '## 当前完成计数','', '```json',json.dumps(counts,indent=2),'```','', '逐轮数据、全部规则、曲线、分类计数和未达参考阈值轮次见本目录CSV/JSON。']
    lines+=['','## 读取完整结果','', '|用途|summary下文件前缀|','|---|---|','|Table 1六模型、五校正|Table1|','|Table 2全部规则、三种学习流程、三个时间|Table2_and_time|','|Fig.2门前MA误差、R²、各类召回|Fig2_all_report|','|Fig.2三分类混淆矩阵计数与波动|Fig2_confusion_counts|','|Fig.3每轮错误/正确放行暂缓计数|Fig3_dispositions|','|Fig.3单分数匹配报告率曲线|Fig3_score_curves|','|新门减all-report的配对变化|gate_minus_allreport_paired|','|九配置、附加target-only及所有对照|nine_configurations / nine_mapping_and_calibration_controls|','|20/40/60本地预算、Ridge/RBF|label_budgets|','|线性、EXP3/EXP4敏感性|linear / EXP3 / EXP4|','|301内部重复评价及映射/校正对照|internal301 / internal_mapping_and_calibration_controls|','|首次锁定与可更新方案|time_schedules|','|原HGB能力图的重复性诊断|Fig4_HGB_capability|','|边界及早期增长描述性关联|boundary_associations / early_growth_associations|','', '每类均提供rounds逐轮记录及summary汇总；JSON保留嵌套定义，CSV便于筛选。Fig4_HGB_capability_pooled另外保留跨轮汇总计数的回顾性匹配点，不与逐轮均值混用。reference201保存预先指定seed1000、完整源记录/60开发/201测试的图示数据，没有从100轮中挑选好例子。']
    (O/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(counts))

if __name__=='__main__':main()
