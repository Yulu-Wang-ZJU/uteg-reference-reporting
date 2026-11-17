"""Snapshot selected policy and all repetitions. No fitting or model search."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import sys,json,shutil,hashlib,csv,platform
from pathlib import Path
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_risk_calibration_20261117'))
import calibrate as m
import numpy as np
r=m.r;O=Path(__file__).parent;WS=Path(str(_repeat_root / 'project' / ''));MAN=WS/'outputs/uteg_nc_goal_20261030/manuscript'
def write(p,x):r.write(p,x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert not (O/'MANIFEST.json').exists(),'Locked package already exists: create a new version instead of overwriting.'
    papers={str(p):sha(p) for p in sorted(MAN.iterdir()) if p.is_file() and (p.suffix in ['.tex','.pdf'] or p.name=='Supplementary_Materials.zip')}
    write(O/'MANUSCRIPT_BASELINE_HASHES.json',papers)
    d,*_=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text());summaries=[];allreport=[];failures=[]
    write(O/'SELECTED_POLICY.json',dict(version='2026-11-17-common_B-locked',selected='common_B',not_selected=['common_B_upper','train_cdf','prior_probability'],decision='Author accepted current scheme; no further tuning. All repetitions retained.',
        scope='15-minute Huzhou-to-SAHZU and Huzhou+Hubei-to-SAHZU,54/60 development settings,100 matched subsampling repeats each.',
        upstream='Unchanged saved B-only anchored-tail MA predictions and cached gate/risk scores used in2026-11-16/17 experiments. This is not a new mapping fit.',
        normal='Original raw B logistic high-risk development90th-percentile veto remains exact. Additional normal bounds use -B and B on a shared physical scale, and the developer-selected normal gate. No B values or MA predictions are changed.',
        abnormal='Predicted-low and predicted-high gates selected separately from All/HGB/Logistic/Boundary on development data; midpoint thresholds, reserve0. Maximize developer report count with accuracy>80%,FN0,nonempty normal/true-abnormal reports.',
        threshold_selection='Only development predictions/reference labels/scores; normal empty-score-gap endpoints retain developer actions. Test labels used only for evaluation.',
        acceptance='Observed test FN0 is primary. Accuracy80% and rate75% remain reference flags, not grounds to exclude a round. No per-round switch of strategy.',
        sampling='54 samples90% of original60 development records;60 retains entire originalpool. Source Huzhou86/95 and Hubei71/79, test181/201,100 matched stratified group draws. No role crossing.',
        interpretation='Repeated subsampling mean±sampleSD, not independent validation or95%CI; no future zero-FNR guarantee.'))
    for budget in [54,60]:
        rows=[x for x in json.loads((m.O/f'dev{budget}/ROUNDS.json').read_text()) if x['policy']=='common_B'];assert len(rows)==200
        write(O/f'results/dev{budget}/ROUNDS.json',rows);summary=m.b.h.summarize(rows);write(O/f'results/dev{budget}/SUMMARY.json',summary)
        summaries += [dict(x,budget=budget) for x in summary];flat=[];pre=[]
        index={(x['rep'],x['arm']):x for x in rows}
        for rep in range(100):
            src=m.O/f'dev{budget}/private/round_{rep:02d}';pol=json.loads((src/'POLICIES.json').read_text());cached=np.load(src/'ACTIONS_SCORES.npz');folder=O/f'private/dev{budget}/round_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
            selected={arm:pol[arm+'_common_B'] for arm in m.a.ARMS};write(folder/'POLICIES.json',selected)
            sp=m.support.spec(items[rep],budget);write(folder/'SPLIT.json',dict(seed=items[rep]['seed'],source_rows=items[rep]['sources'],target=sp))
            shutil.copy2(m.n.O/f'dev{budget}/private/round_{rep:02d}/RISK_TRAINING.json',folder/'RISK_TRAINING.json')
            for arm in m.a.ARMS:
                z,rr,_=m.b.load(budget,rep,arm);values={};row=index[rep,arm]
                for prefix,side,ids,pp,risk,pref in [('dev_','development',z['dev_rows'],z['dev_prediction'],rr['dev_risk_B'],'dev_score_'),('test_','test',z['test'],z['prediction'],rr['test_risk_B'],'score_')]:
                    values[prefix+'ids']=ids;values[prefix+'groups']=d['groups'][ids].astype(str);values[prefix+'prediction']=pp;values[prefix+'truth']=d['medical'][ids,2];values[prefix+'B']=d['x'][ids,1];values[prefix+'raw_risk']=risk
                    scores={k:z[pref+k] for k in m.a.RULES};calibrated=np.c_[-d['x'][ids,1],d['x'][ids,1]]
                    action=m.action(selected[arm],pp,risk,calibrated,scores);values[prefix+'action']=action
                    if side=='test':np.testing.assert_array_equal(action,cached[arm+'_common_B'])
                    for k,v in scores.items():values[prefix+'score_'+k]=v
                np.savez_compressed(folder/(arm+'.npz'),**values)
                metric=r.metrics(z['prediction'],d['medical'][z['test'],2],np.ones(len(z['test']),bool));pre.append(dict(budget=budget,rep=rep,arm=arm,policy='all_report',feasible=True,development=None,development_joint_pass=False,test=metric,joint_test_pass=False,risk_model='none',cap_quantile=None))
                fail=[]
                if row['test']['accuracy']<=.8:fail.append('accuracy <=80%')
                if row['test']['reporting_rate']<=.75:fail.append('reporting rate <=75%')
                if fail:failures.append(dict(budget=budget,round=rep+1,arm=arm,reference_flags=fail,test=row['test']))
                record=dict(budget=budget,round=rep+1,arm=arm,policy='common_B',all_reference_thresholds_met=row['joint_test_pass'])
                for side in ['development','test']:
                    for k,v in row[side].items():
                        if not isinstance(v,(dict,list)):record[side+'_'+k]=v
                flat.append(record)
            cached.close()
        csvwrite(O/f'results/dev{budget}/ROUNDS.csv',flat);write(O/f'results/dev{budget}/ALL_REPORT_ROUNDS.json',pre);allreport+=pre
    write(O/'SUMMARY.json',summaries);write(O/'REFERENCE_THRESHOLD_EXCEPTIONS.json',failures)
    flat=[]
    for q in summaries:
        out={k:q[k] for k in ['budget','arm','rounds','zero_FN','test_joint','both_joint']}
        for key in ['accuracy','reporting_rate','FNR','FPR']:
            for stat in ['mean','sd','min','max']:out[key+'_'+stat+'_percent']=100*q[key][stat]
        flat.append(out)
    csvwrite(O/'SUMMARY.csv',flat)
    write(O/'ALL_REPORT_SUMMARY.json',[dict(budget=budget,arm=arm,**{key:dict(mean=float(np.mean([x['test'][key] for x in allreport if x['budget']==budget and x['arm']==arm])),sd=float(np.std([x['test'][key] for x in allreport if x['budget']==budget and x['arm']==arm],ddof=1))) for key in ['MAE','R2','accuracy','FNR','FPR']}) for budget in [54,60] for arm in m.a.ARMS])
    # Preserve executed source files and protocols. No full-training portability claim.
    files=set();virtual=[]
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path)
            if path.suffix=='.py' and (path.is_relative_to(WS) or path.is_relative_to(Path(str(_repeat_root / 'experiments' / '')))):
                if path.exists():files.add(path)
                else:virtual.append(str(path))
    files.update((r.OUT).glob('*.py'));files.update(m.O.glob('*.py'));files.update(m.f.O.glob('*.py'))
    code_index=[]
    for src in sorted(files):
        rel=Path('project')/src.relative_to(WS) if src.is_relative_to(WS) else Path('experiments')/src.relative_to(Path(str(_repeat_root / 'experiments' / '')))
        dst=O/'code_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);code_index.append(dict(original=str(src),snapshot=str(dst.relative_to(O)),sha256=sha(dst)))
    write(O/'CODE_INDEX.json',dict(files=code_index,dynamic_module_paths=virtual,note='Dynamic numerical runtime is configured by snapshotted current_suite/run activation; its nominal runtime.py path is not a missing disk source. Source snapshots preserve existing import paths; verify_locked.py is the standalone replay entry, not a full model-training runner.'))
    for name in ['PROTOCOL.json','FINAL_QA.json','SMOKE.json','SOURCE_HASHES.json','FAILURES.json','REPORT_CN.md','RESIDUAL_DIAGNOSTIC.json']:
        dst=O/'provenance'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(m.O/name,dst)
    for name in ['PROTOCOL.json','FINAL_QA.json']:
        shutil.copy2(r.OUT/name,O/'provenance'/('resampling_'+name))
    write(O/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    write(O/'PROVENANCE.json',dict(source_result_root=str(m.O),source_rows_hashes={str(m.O/f'dev{b}/ROUNDS.json'):sha(m.O/f'dev{b}/ROUNDS.json') for b in [54,60]},old_candidates='Retained in original experiment folders and provenance summaries; excluded from selected400 rows. Nothing deleted.',privacy='private/ contains record-level reference values and identifiers for local authorized verification only. Not uploaded to GitHub or included in public supplementary materials.'))
    assert all(sha(Path(k))==v for k,v in papers.items())
    print('PACKAGED',len(files),'code files;',len(failures),'reference exceptions; selected400 runs')
if __name__=='__main__':main()
