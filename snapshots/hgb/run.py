import os
os.environ['OMP_NUM_THREADS']='1'
import ast, json, hashlib, csv, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parent
P=ROOT.parents[1]
OLD=P/'outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
HGB=OLD.parent/'ma_tail_weight_gate_run_20261026_141553/ridge_ma13d_hgb_prepare_20261026_151041'
DATA=P/'outputs/uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
DIRS=['HZ_to_SA','HZ+HB_to_SA']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x): (ROOT/n).write_text(json.dumps(x,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())+'\n')
def extract(p,names,ns):
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns
ns={'np':np,'Any':Any,'Mapping':Mapping,'MA_CLASSES':(-1,0,1)}
SELECT=P/'outputs/uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py'
extract(SELECT,['summary','select_threshold'],ns)
extract(OLD/'run_ridge10_fullchain.py',['ma_class','ma_features'],ns)
def paths(d,s):
    return OLD/'models'/d/'target15'/f'SAHZU_CFULL_seed{s}.joblib',HGB/'models_hgb'/d/'target15'/f'SAHZU_HGB_seed{s}.joblib'
def score(m,f):
    if m['status']=='FALLBACK_CONSTANT': return np.full(len(f),m['constant_score'])
    p=m['pre'];z=np.clip((np.where(np.isfinite(f),f,np.asarray(p['median']))-p['mean'])/p['sd'],-10,10)
    return m['model'].predict_proba(z)[:,1]
def freeze():
    assert not (ROOT/'FROZEN.json').exists()
    hashes={str(SELECT):sha(SELECT),str(DATA):sha(DATA)};rows=[]
    for d in DIRS:
        for s in range(1000,1020):
            bp,hp=paths(d,s);b=joblib.load(bp);h=joblib.load(hp)
            assert h['base_object_sha256']==sha(bp)
            hashes.update({str(bp):sha(bp),str(hp):sha(hp)})
            gs=[h['gates'][str(i)] for i in range(3)]
            idx=np.concatenate([g['outer_index'] for g in gs]);assert len(np.unique(idx))==60 and set(idx)==set(b['pool'])
            assert not set(idx)&set(b['test'])
            pred=np.concatenate([g['outer_pred'] for g in gs]);y=np.concatenate([g['outer_truth'] for g in gs])
            sc=np.concatenate([g['candidate_within_score'] for g in gs])
            lr=np.concatenate([g['base_score'] for g in gs])
            selected=ns['select_threshold'](sc,pred,y,.15)
            old=ns['select_threshold'](lr,pred,y,.15)
            assert old['threshold']==b['gate_models'][3]['threshold']['threshold']
            assert h['gates']['3']['adapter']==b['gate_models'][3]['adapter']
            rows.append(dict(direction=d,seed=s,HGB=selected,LR=old))
    write('FROZEN.json',dict(rows=rows,inputs=hashes,new_fits=0,features=13,scope='Full error-score HGB baseline, not historical within-only hybrid',test_labels_read=False))
    print('FROZEN: 40 working points; original LR selector reproduced; 0 new fits')
def table(n,rows):
    with (ROOT/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluate():
    fr=json.loads((ROOT/'FROZEN.json').read_text());assert all(sha(p)==h for p,h in fr['inputs'].items())
    data=np.load(DATA,allow_pickle=False);out=[];records=[];seeds=[]
    for d in DIRS:
        z=np.load(OLD/'eval/predictions'/f'{d}.npz',allow_pickle=False)
        idx=z['test_global_index'];truth=ns['ma_class'](data['medical'][idx,2]);pred=z['final_ma_class'];acts=[];scores=[]
        assert np.array_equal(z['test_ids'],data['ids'][idx]) and len(idx)==201
        for k,s in enumerate(range(1000,1020)):
            bp,hp=paths(d,s);b=joblib.load(bp);h=joblib.load(hp);src=b['source_model']
            assert np.array_equal(np.sort(b['test']),np.sort(idx))
            f=ns['ma_features'](data['x'][idx],z['final_med4'][k],src['ym'],src['ys'])
            sc=score(h['gates']['3']['model'],f)
            sc=np.where(z['input_qc']&np.isfinite(f).all(1)&np.isin(pred[k],[-1,0,1]),sc,np.inf)
            r=next(x for x in fr['rows'] if x['direction']==d and x['seed']==s)
            t=r['HGB']['threshold'];a=np.zeros(len(idx),bool) if t is None else sc<=t
            lt=r['LR']['threshold'];la=np.zeros(len(idx),bool) if lt is None else z['lr_score'][k]<=lt
            assert np.array_equal(la,z['lr_action'][k].astype(bool))
            acts.append(a);scores.append(sc)
        acts=np.asarray(acts);scores=np.asarray(scores)
        np.savez_compressed(ROOT/f'{d}.npz',pred=pred,truth=truth,action=acts,score=scores,test_ids=z['test_ids'],groups=data['groups'][idx],seed=z['seed'])
        for name,aa in [('LR',z['lr_action'].astype(bool)),('HGB',acts)]:
            correct=aa&(pred==truth);low=aa&(pred==0)&(truth==-1);high=aa&(pred==0)&(truth==1)
            row=dict(direction=d,method=name,correct=int(correct.sum()),released=int(aa.sum()),total=int(aa.size),agreement=100*correct.sum()/aa.sum() if aa.any() else None,coverage=100*aa.mean(),low_within=int(low.sum()),low_support=int((truth==-1).sum()*20),high_within=int(high.sum()),high_support=int((truth==1).sum()*20),correct_low=int((aa&(pred==-1)&(truth==-1)).sum()),correct_high=int((aa&(pred==1)&(truth==1)).sum()),active_seeds=int(aa.any(1).sum()))
            out.append(row)
            for k,s in enumerate(range(1000,1020)):
                seeds.append(dict(direction=d,method=name,seed=s,released=int(aa[k].sum()),correct=int(correct[k].sum()),low_within=int(low[k].sum()),high_within=int(high[k].sum())))
                for j,rid in enumerate(z['test_ids']): records.append(dict(direction=d,method=name,seed=s,record_id=str(rid),truth=int(truth[j]),prediction=int(pred[k,j]),release=int(aa[k,j])))
    table('summary.csv',out);table('per_seed.csv',seeds);table('record_actions.csv',records)
    write('QA.json',dict(new_fits=0,LR_actions_exact=True,input_hashes_unchanged=all(sha(p)==h for p,h in fr['inputs'].items()),test_tuning=False))
    print(json.dumps(out,indent=2))
if __name__=='__main__': {'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
