"""Evaluate frozen matched workflows without fitting or selecting thresholds."""
from pathlib import Path
import csv, hashlib, importlib.util, json, subprocess
import numpy as np
import joblib

R=Path(__file__).resolve().parent; G=R.parent; O=G.parent
F=G/'focused_reporting_20261030'
SEQ=O/'uteg_current_ma_sequence_20261026/actions'
EXT=O/'uteg_ridge_dual_head_external_eval_20261026_191117/actions'
PRED=O/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/eval/predictions'
DIRS=['HZ_to_SA','HZ+HB_to_SA']; SEEDS=np.arange(1000,1020); HORIZONS=[300,600,900]
TARGET='Target-only Ridge'
SCHEDULES={'5':[300],'10':[600],'15':[900],'5-10':[300,600],'5-10-15':[300,600,900],'10-15':[600,900]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(name,obj):
    p=R/name;assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def table(name,rows):
    p=R/name;assert not p.exists()
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def replay(checkpoints,schedule):
    shape=checkpoints[schedule[0]][0].shape
    released=np.zeros(shape,bool);pred=np.full(shape,9);time=np.full(shape,-1)
    for h in schedule:
        a,p=checkpoints[h];new=a&~released
        pred[new]=p[new];time[new]=h;released|=a
    return released,pred,time

def main():
    assert not (R/'RESULTS.json').exists()
    freeze=json.loads((R/'FREEZE.json').read_text(encoding='utf-8'))
    assert freeze['test_evaluated'] is False
    assert all(sha(p)==v for p,v in {**freeze['inputs'],**freeze['models']}.items())
    evalinputs={str(p.resolve()):sha(p) for p in [Path(__file__),R/'PROTOCOL.md',R/'FREEZE.json',G/'analysis/round1.py',F/'run.py',F/'RESULTS.json',F/'per_seed.csv']}
    for d in DIRS:
        for p in [SEQ/f'T{h}'/f'{d}.npz' for h in [300,600]]+[EXT/f'{d}.npz',SEQ/f'{d}_first_report.npz',PRED/f'{d}.npz']:
            evalinputs[str(p.resolve())]=sha(p)
    write('EVALUATION_INPUT_HASHES.json',evalinputs)
    tr=load('target_sequence_train',R/'train_prefix.py')
    base=load('target_sequence_base',F/'run.py')
    original=load('target_sequence_original_analysis',G/'analysis/round1.py')
    d15=tr.load_checkpoint(900)
    idx=None;checkpoints={TARGET:{}};private={}
    for h in HORIZONS:
        data=tr.load_checkpoint(h)
        for key in ['ids','groups','medical']:assert np.array_equal(data[key],d15[key],equal_nan=True) if data[key].dtype.kind in 'fc' else np.array_equal(data[key],d15[key])
        pp=[];aa=[];numeric=[]
        for seed in SEEDS:
            path=(F if h==900 else R/f'T{h}')/f'model_{seed}.joblib'
            assert str(path.resolve()) in freeze['models'] or str(path) in freeze['models'] or str(path.resolve()) in freeze['inputs'] or str(path) in freeze['inputs']
            m=joblib.load(path);I=m['test']
            prior=joblib.load(F/f'model_{seed}.joblib')
            assert m['seed']==seed and np.array_equal(m['pool'],prior['pool'])
            if idx is None:idx=I.copy()
            assert np.array_equal(idx,I) and len(m['pool'])==60
            assert not set(data['groups'][m['pool']])&set(data['groups'][I])
            x=data['x'][I];assert np.isfinite(x).all()
            med=base.ridge.predict(m['predictor'],x);pc=base.ma_class(med[:,2])
            bs=base.score_ma_gate(m['error'],base.ma_features(x,med,m['ym'],m['ys']),pc)
            ds=base.head_scores(m['heads'],x,med)
            a=base.action(m['threshold'],pc,bs,ds,data['input_qc'][I])
            pp.append(pc);aa.append(a);numeric.append(med)
        checkpoints[TARGET][h]=(np.array(aa),np.array(pp))
        private[f'target_pred_T{h}']=np.array(pp);private[f'target_action_T{h}']=np.array(aa);private[f'target_med_T{h}']=np.array(numeric)
    groups=d15['groups'][idx];truth=base.ma_class(d15['medical'][idx,2]);assert len(idx)==201 and len(np.unique(groups))==190
    assert [(truth==v).sum() for v in [-1,0,1]]==[7,170,24]
    for d in DIRS:
        checkpoints[d]={}
        for h in [300,600]:
            with np.load(SEQ/f'T{h}'/f'{d}.npz',allow_pickle=False) as q:
                assert np.array_equal(q['ids'].astype(str),d15['ids'][idx].astype(str))
                assert np.array_equal(q['groups'],groups) and np.array_equal(q['seed'],SEEDS)
                assert np.array_equal(q['index'],idx)
                checkpoints[d][h]=(q['action'].astype(bool),q['pred'].copy())
        with np.load(EXT/f'{d}.npz',allow_pickle=False) as q:
            assert np.array_equal(q['test_ids'].astype(str),d15['ids'][idx].astype(str))
            with np.load(PRED/f'{d}.npz',allow_pickle=False) as p:
                assert np.array_equal(p['seed'],SEEDS) and np.array_equal(p['test_global_index'],idx)
                assert np.array_equal(p['test_ids'].astype(str),q['test_ids'].astype(str))
                assert np.array_equal(p['final_med4'][:,:,2],q['final_ma'])
                assert np.array_equal(base.ma_class(p['final_med4'][:,:,2]),q['final_ma_class'])
            assert q['unified_action'].shape==(20,201)
            checkpoints[d][900]=(q['unified_action'].astype(bool),q['final_ma_class'].copy())
    stat=original.bootstrap(groups);one=np.ones((20,201));summaries={};arrays={};seeds=[];dest=[];times=[];overview=[]
    oldseed=list(csv.DictReader((F/'per_seed.csv').open(encoding='utf-8-sig')))
    oldresults=json.loads((F/'RESULTS.json').read_text(encoding='utf-8'))
    for arm,cp in checkpoints.items():
        summaries[arm]={};arrays[arm]={}
        for policy,schedule in SCHEDULES.items():
            a,p,t=replay(cp,schedule);good=a&(p==truth);bad=a&~good
            ar=dict(correct=good,incorrect=bad,unreported=~a,coverage=a,low_to_within=a&(p==0)&(truth==-1),high_to_within=a&(p==0)&(truth==1),correct_low=a&(p==-1)&(truth==-1),correct_high=a&(p==1)&(truth==1))
            assert np.all(good.astype(int)+bad+(~a)==1)
            arrays[arm][policy]=ar
            summary={'counts':{k:int(v.sum()) for k,v in ar.items()},'yields':{k:stat(v,one) for k,v in ar.items()},'agreement':stat(good,a),'within_reports':int((p==0).sum()),'within_contamination':stat(ar['low_to_within']|ar['high_to_within'],p==0),'zero_report_seeds':int((a.sum(1)==0).sum())}
            summaries[arm][policy]=summary
            overview.append(dict(method=arm,schedule=policy,**summary['counts'],coverage_pct=float(a.mean()*100),agreement_pct=float(good.sum()/a.sum()*100) if a.any() else None,correct_yield_pct=float(good.mean()*100),incorrect_yield_pct=float(bad.mean()*100),zero_report_seeds=summary['zero_report_seeds']))
            for j,seed in enumerate(SEEDS):
                row=dict(method=arm,schedule=policy,seed=int(seed),**{k:int(v[j].sum()) for k,v in ar.items()},agreement=float(good[j].sum()/a[j].sum()) if a[j].any() else None);seeds.append(row)
                if arm==TARGET and policy=='15':
                    old=next(r for r in oldseed if r['method']==TARGET and int(r['seed'])==seed)
                    assert all(row[k]==int(old[k]) for k in ar)
            for true in [-1,0,1]:
                for reported in [-1,0,1,9]:
                    n=int(((truth==true)&(p==reported)).sum());N=int((truth==true).sum()*20)
                    dest.append(dict(method=arm,schedule=policy,true_state=true,report_state=reported,n=n,N=N,proportion=n/N))
            for h in [300,600,900,-1]:
                at=t==h
                times.append(dict(method=arm,schedule=policy,time_seconds=h,events=int(at.sum()),correct=int((at&good).sum()),incorrect=int((at&bad).sum()),N=4020))
            if policy=='5-10-15':
                private[arm+'_first_class']=p;private[arm+'_first_time']=t;private[arm+'_reported']=a
                if arm!=TARGET:
                    with np.load(SEQ/f'{arm}_first_report.npz',allow_pickle=False) as old:
                        assert np.array_equal(old['reported'],a) and np.array_equal(old['first_class'],p) and np.array_equal(old['first_time'],t)
            if policy=='15':assert summary['counts']==oldresults['arms'][arm]['counts']
    assert [[summaries[d]['5-10-15']['counts'][k] for k in ['correct','incorrect','unreported']] for d in DIRS]==[[925,140,2955],[737,67,3216]]
    comparisons={};contrastrows=[];seedcomparisons=[]
    for d in DIRS:
        comparisons[d]={}
        for policy in SCHEDULES:
            comparisons[d][policy]={}
            for k in arrays[d][policy]:
                diff=arrays[d][policy][k].astype(int)-arrays[TARGET][policy][k].astype(int)
                s=stat(diff,one);comparisons[d][policy][k]=s
                contrastrows.append(dict(comparison=d+' minus target-only',schedule=policy,outcome=k,delta_events=int(diff.sum()),delta_pp=s['estimate']*100,ci_low_pp=s['lo']*100,ci_high_pp=s['hi']*100))
            for j,seed in enumerate(SEEDS):
                seedcomparisons.append(dict(comparison=d+' minus target-only',schedule=policy,seed=int(seed),**{k:int(arrays[d][policy][k][j].sum()-arrays[TARGET][policy][k][j].sum()) for k in arrays[d][policy]}))
    within={}
    for arm in checkpoints:
        within[arm]={}
        for left,right in [('5-10-15','15'),('5-10-15','10-15'),('10-15','15'),('5-10-15','5')]:
            within[arm][left+' minus '+right]={k:stat(arrays[arm][left][k].astype(int)-arrays[arm][right][k].astype(int),one) for k in arrays[arm][left]}
    assert all(sha(p)==v for p,v in {**freeze['inputs'],**freeze['models'],**evalinputs}.items())
    protected=json.loads((R/'PROTECTED_BEFORE.json').read_text());assert all(sha(p)==h for p,h in protected['protected_files'].items())
    repo=O.parent/'releases/uteg_reference_reporting_20261030'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()==protected['public_repo_head']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=repo).decode()==protected['public_repo_status']
    (R/'private').mkdir(exist_ok=True)
    np.savez_compressed(R/'private/evaluation_arrays.npz',**private,seed=SEEDS,truth=truth,groups=groups,test_global_index=idx)
    table('summary.csv',overview);table('paired_contrasts.csv',contrastrows);table('per_seed.csv',seeds);table('paired_seeds.csv',seedcomparisons);table('class_destinations.csv',dest);table('first_report_times.csv',times)
    write('RESULTS.json',dict(records=201,groups=190,seeds=20,label_budget=60,events=4020,arms=summaries,source_minus_target=comparisons,schedule_contrasts=within))
    write('CHECKS.json',dict(status='PASS',target_900_all_seed_outcomes_reproduced=True,all_transfer_900_counts_reproduced=True,transfer_full_sequence_arrays_exact=True,original_input_hashes_unchanged=True,protected_deliverables_unchanged=True,repository_unchanged=True,new_thresholds_selected_on_test=False))
    print(json.dumps({a:summaries[a]['5-10-15'] for a in checkpoints},indent=2))

if __name__=='__main__':main()
