"""Current fixed B file only; no fold-wise regeneration and no legacy fallback."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import core as c
from . import primary
from .input_contract import input_provenance
from .analysis import replay, SCHEDULES, table, pct, num

ROOT=c.WS/'outputs/current_ma_20261103'
POINTER=c.WS/'outputs/ACTIVE_EXPERIMENT_DATASET.json'





def activate():
    pointer=c.read(POINTER);path=Path(pointer['dataset_path'])
    assert path.resolve()==(ROOT/'data/DATA15_CURRENT.npz').resolve()
    assert c.sha(path)==pointer['sha256']
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    d['y']=d['medical']
    with np.load(ROOT/'data/CURRENT_SCALED_CURVES.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids'])
        prefix={int(h):z['prefix_parameters'][i].copy() for i,h in enumerate(z['horizons'])}
        qc={int(h):z['qc'][i]&d['input_qc'] for i,h in enumerate(z['horizons'])}
    prefix[900]=d['x'].copy();qc[900]=d['input_qc'].astype(bool)
    folds=c.read(ROOT/'data/FROZEN_SPLITS.json');local=np.sort(np.r_[folds[0]['A'],folds[0]['B']])
    plan=c.read(ROOT/'data/SOURCE_PLAN.json');role=c.read(ROOT/'data/ROLE_FREEZE.json')
    c._CONTEXT=(d,prefix,np.full(len(d['x']),np.nan),plan,role,folds,local)
    def features(train,query,h,representation='B'):
        return prefix[h][:,{'B':[1],'three':[0,1,2],'four':[0,1,2,3]}[representation]].copy()
    def source_feature(direction,h,representation='B'):
        return features([],[],h,representation)
    def source_rows(direction,h):
        rows=np.asarray(plan['source_rows'].get(direction,[]),int);return rows[qc[h][rows]]
    c.feature=features;c.source_feature=source_feature;c.source_rows=source_rows;c.qc_at=lambda h:qc[h]
    # Retained compatibility flag; activate() loads saved arrays above.
    # No curve is generated or rescaled by setting this flag.
    c.RECONSTRUCTED=True;c.OUT=ROOT;c._SOURCE.clear()
    return c._CONTEXT


def evaluate():
    d,_,_,_,_,folds,local=activate()
    freeze=c.read(ROOT/'DATA_FREEZE.json')
    assert all(c.sha(p)==h for p,h in freeze['files'].items())
    # Existing B values remain unchanged when reference labels are masked.
    a=np.asarray(folds[0]['A']);b=np.asarray(folds[0]['B']);saved=d['y'].copy()
    old={h:c.feature(a,b,h) for h in c.HORIZONS}
    try:
        d['y'][b]=1e6
        for h in c.HORIZONS:np.testing.assert_array_equal(old[h],c.feature(a,b,h))
    finally:d['y'][:]=saved
    c.write(ROOT/'PROTOCOL.json',dict(data_freeze=str(ROOT/'DATA_FREEZE.json'),
        model='Selected B-only RBF source prior + local RBF correction, alpha .01, inward4, inverseclass cap20.',
        gate='InnerOOF boundary-distance95th percentile; no test-label threshold tuning.',
        evaluation='Frozen301 5outer/3inner; separately original60development/201test,20seeds.',
        change='Stored B values read directly from the active input file and retained across splits.',
        input_provenance=input_provenance(),
        B_from_original_data=True,manuscript_changed=False))
    pos={int(v):i for i,v in enumerate(local)}
    for h in c.HORIZONS:
        path=ROOT/'all301'/f'T{h}.npz'
        if path.exists():continue
        pred=np.full((3,len(local)),np.nan);ac=np.zeros_like(pred,bool);audits=[];seen=np.zeros(len(local),int)
        for fi,f in enumerate(folds):
            ix=[pos[int(v)] for v in f['B']];seen[ix]+=1
            for j,arm in enumerate(c.MAIN):
                result=c.evaluate(f['A'],f['B'],f['inner'],arm,h)
                pred[j,ix]=result['prediction'];ac[j,ix]=result['accepted'];audits.append(dict(fold=fi,arm=arm,**result))
        assert (seen==1).all() and np.isfinite(pred).all()
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,prediction=pred,accepted=ac,local=local,reference=d['y'][local,2],directions=c.MAIN,
                            B_from_original_data=True)
        c.write(path.with_suffix('.audit.json'),audits)
        c.write(path.with_suffix('.metrics.json'),{arm:dict(all_report=c.summarize(pred[j],d['y'][local,2]),boundary95=c.summarize(pred[j],d['y'][local,2],ac[j])) for j,arm in enumerate(c.MAIN)})
        print('CURRENT_ALL301',h,flush=True)
    for arm in c.MAIN:
        for h in c.HORIZONS:primary.paper(arm,h)
    c.write(ROOT/'EVALUATION_VERIFIED.json',dict(input_hashes_unchanged=all(c.sha(p)==h for p,h in freeze['files'].items()),
        current_feature_builder_does_not_read_reference_labels=True,original_B_values_retained=True,
        all301_each_once=True,manuscript_changed=False))


def report():
    d,*_=activate();rows=[];seq={};names={'target_only_SA':'Target-only','HZ_to_SA':'单源','HZ+HB_to_SA':'双源'}
    for protocol in ['all301','paper']:
        seq[protocol]={}
        for j,arm in enumerate(c.MAIN):
            pp={};aa={}
            for h in c.HORIZONS:
                path=ROOT/('all301' if protocol=='all301' else 'primary')/(f'T{h}.npz' if protocol=='all301' else f'{arm}_T{h}_B0.npz')
                with np.load(path,allow_pickle=False) as z:
                    pp[h]=z['prediction'][j].copy() if protocol=='all301' else z['prediction'].copy()
                    aa[h]=z['accepted'][j].copy() if protocol=='all301' else z['accepted'].copy()
                    idx=z['local'] if protocol=='all301' else z['test']
                y=d['y'][idx,2];m=c.summarize(pp[h],y);g=c.summarize(pp[h],y,aa[h])
                rows.append(dict(protocol=protocol,arm=arm,minutes=h//60,all_report=m,gate=g))
            seq[protocol][arm]={}
            for label,times in SCHEDULES.items():
                p,t,a=replay(pp,aa,times);seq[protocol][arm][label]=c.summarize(p,y,a)
    c.write(ROOT/'MAIN_RESULTS.json',dict(checkpoints=rows,sequence=seq))
    with (ROOT/'MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['protocol','arm','minutes','agreement','fnr','fn','fn_denominator','fpr','fp','fp_denominator','low_recall','high_recall','mae','r2','coverage','gate_agreement','gate_fnr']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            m=r['all_report'];g=r['gate'];writer.writerow(dict(protocol=r['protocol'],arm=r['arm'],minutes=r['minutes'],
                **{k:m[k] for k in fields[3:14]},coverage=g['coverage'],gate_agreement=g['agreement'],gate_fnr=g['fnr']))
    lines=['# 当前新版B：主结果','',
        '唯一工作输入为data/DATA15_CURRENT.npz，逐值沿用已确认的各医院固定标尺新版B。原B、旧备份和上一轮旧幅值对照移到项目外归档。本报告没有重新生成逐折B，也没有把此前8.33%/6.25%的逐折标尺结果当作当前固定文件结果。','',
        '模型沿用B-only RBF外院先验＋本地RBF修正、inward4和相同标签预算。301条分组评价与60条开发＋201条评价分开。5/10分钟使用相应时段实测信号的拟合参数；完整记录为15分钟。','',
        '## 15分钟主结果','']
    for protocol in ['all301','paper']:
        lines += [f'### {protocol}','',table(['流程','三分类一致率','FNR','FPR','低/高召回','MAE','R²','门覆盖','门后一致率','门后FNR'],[[names[r['arm']],pct(r['all_report']['agreement']),pct(r['all_report']['fnr'])+f" ({r['all_report']['fn']}/{r['all_report']['fn_denominator']})",pct(r['all_report']['fpr']),pct(r['all_report']['low_recall'])+'/'+pct(r['all_report']['high_recall']),num(r['all_report']['mae']),num(r['all_report']['r2']),pct(r['gate']['coverage']),pct(r['gate']['agreement']),pct(r['gate']['fnr'])] for r in rows if r['protocol']==protocol and r['minutes']==15]),'']
    lines+=['301条为11低、253正常、37高，每条一次外层预测；paper为201条（7低、170正常、24高）×20个种子。重复种子事件不能当作新增患者。门前FNR分母为全部真实异常，门后为已报告的真实异常；未报告不计正确。','',
        '## 5/10分钟单时点评价','']
    for protocol in ['all301','paper']:
        lines += [f'### {protocol}','',table(['流程','分钟','三分类一致率','FNR','FPR','MAE','R²','门覆盖'],[[names[r['arm']],r['minutes'],pct(r['all_report']['agreement']),pct(r['all_report']['fnr']),pct(r['all_report']['fpr']),num(r['all_report']['mae']),num(r['all_report']['r2']),pct(r['gate']['coverage'])] for r in rows if r['protocol']==protocol and r['minutes']<15]),'']
    lines+=['## 首次报告锁定的序贯结果','']
    for protocol in ['all301','paper']:
        lines += [f'### {protocol}','',table(['流程','时间方案','正确/错误/未报','一致率','FNR','覆盖'],[[names[arm],label,f"{m['correct']}/{m['incorrect']}/{m['withheld']}",pct(m['agreement']),pct(m['fnr']),pct(m['coverage'])] for arm in c.MAIN for label,m in seq[protocol][arm].items()]),'']
    lines+=['## 版本与证据说明','',
        '当前B为经医生核实的原始数据，代码直接读取该输入文件。模型拟合、标准化、权重和门阈值仅使用当次训练侧。早期分析使用相应时段的实测信号。','',
        '完整分层误差、三分类/二分类矩阵、已报告分母和未报告去向见MAIN_RESULTS.json。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CURRENT_MAIN_REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['evaluate','report']);a=ap.parse_args()
    with threadpool_limits(limits=1):globals()[a.stage]()
