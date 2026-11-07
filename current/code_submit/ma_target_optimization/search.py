"""Resumable development search. This module cannot evaluate the test cohort."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs
from .models import fit_candidate
from .metrics import assess, selection_key

ROOT=Path(__file__).resolve().parents[2]


def candidates():
    grid=[]
    for features,alpha,cap in itertools.product(('linear','loglinear','logquadratic'),(.001,.01,.1,1.),(1.,5.,20.)):
        grid.append(dict(family='ridge',features=features,alpha=alpha,cap=cap))
    for features,alpha,strength,cost in itertools.product(('linear','logquadratic'),(.01,.1),(1.,4.),(1.,4.,16.)):
        grid.append(dict(family='boundary',features=features,alpha=alpha,cap=5.,boundary_strength=strength,abnormal_cost=cost))
    for features,alpha,gamma,cap in itertools.product(('linear','loglinear'),(.1,1.,10.),(.0625,.25,1.),(1.,5.,20.)):
        grid.append(dict(family='kernel',features=features,alpha=alpha,gamma=gamma,cap=cap))
    for arm,source_alpha,source_cap,cap,penalty in itertools.product(('HZ_to_SA','HZ+HB_to_SA'),(.1,10.),(1.,5.),(1.,5.,20.),(.1,10.)):
        grid.append(dict(family='transfer',mode='joint',arm=arm,source_alpha=source_alpha,source_cap=source_cap,cap=cap,adapter_penalty=penalty))
    for arm,source_cap,mode,alpha,cap in itertools.product(('HZ_to_SA','HZ+HB_to_SA'),(1.,5.),('ma_only','residual'),(.01,.1),(1.,5.,20.)):
        if mode=='residual' and cap!=1:
            continue
        grid.append(dict(family='transfer',mode=mode,arm=arm,source_alpha=10.,source_cap=source_cap,cap=cap,alpha=alpha,features='linear'))
    for source_cap,alpha,cap in itertools.product((1.,5.),(.01,.1),(1.,5.,20.)):
        grid.append(dict(family='separate_sources',arm='HZ+HB_to_SA',source_alpha=10.,source_cap=source_cap,features='linear',alpha=alpha,cap=cap))
    for c in grid:
        c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    assert len({c['id'] for c in grid})==len(grid)
    return grid


def training_view(data, rows):
    """Copy only source and permitted development arrays into the estimator."""
    rows=np.asarray(rows,int)
    allowed=set(data['allowed_training_indices'].tolist())
    if not set(rows.tolist())<=allowed or not np.isfinite(data['y'][rows]).all():
        raise ValueError('Illegal training rows')
    source_rows=data['source_rows']
    sources={k:(data['x'][v].copy(),data['y'][v].copy()) for k,v in source_rows.items()}
    center_rows={center:source_rows['HZ+HB_to_SA'][data['centers'][source_rows['HZ+HB_to_SA']]==center] for center in ('Huzhou','Hubei')}
    return dict(x=data['x'][rows].copy(),y=data['y'][rows].copy(),sources=sources,
                source_centers={k:(data['x'][v].copy(),data['y'][v].copy()) for k,v in center_rows.items()})


def development_predictions(data,partitions,candidate,source_cache=None):
    cache={} if source_cache is None else source_cache
    pool=next(iter(partitions.values()))['pool'];positions={int(v):i for i,v in enumerate(pool)}
    predictions=[]
    for seed,partition in partitions.items():
        pred=np.full(len(pool),np.nan)
        for fold in partition['folds']:
            A,B=fold['A'],fold['B']
            if set(data['groups'][A]) & set(data['groups'][B]):
                raise ValueError('Group overlap')
            fit=fit_candidate(candidate,training_view(data,A),cache)
            pred[[positions[int(v)] for v in B]]=fit.predict(data['x'][B])
        if not np.isfinite(pred).all():
            raise RuntimeError('OOF coverage is incomplete')
        predictions.append(pred)
    return np.asarray(predictions)


def make_summary(records,out):
    good=[r for r in records if r['status']=='complete']
    ordered=sorted(good,key=lambda r:(selection_key(r['metrics']),r['candidate']['id']))
    feasible=[r for r in ordered if r['metrics']['mapping_target_pass']]
    accuracy=[r for r in ordered if r['metrics']['agreement']>.8]
    low_fnr=sorted(good,key=lambda r:(r['metrics']['fnr'],-r['metrics']['agreement']))
    frontier=[]
    for a in good:
        m=a['metrics']
        if not any((b['metrics']['agreement']>=m['agreement'] and b['metrics']['fnr']<=m['fnr'] and
                    (b['metrics']['agreement']>m['agreement'] or b['metrics']['fnr']<m['fnr'])) for b in good):
            frontier.append(a)
    def compact(r):
        m=r['metrics']
        return {'candidate':r['candidate'],**{k:m[k] for k in ('agreement','fnr','fpr','low_recall','high_recall','low_mae','high_mae','macro_mae','mapping_target_pass')}}
    summary={'completed':len(good),'failed':len(records)-len(good),'feasible_count':len(feasible),
             'best_feasible':[compact(r) for r in feasible[:5]],'best_accuracy_qualified':[compact(r) for r in accuracy[:5]],
             'lowest_fnr':[compact(r) for r in low_fnr[:5]],'frontier':[compact(r) for r in frontier],
             'best_per_family':{},'score_scope':'development_selection_scores_not_unbiased_validation'}
    for family in sorted({r['candidate']['family'] for r in good}):
        summary['best_per_family'][family]=[compact(r) for r in ordered if r['candidate']['family']==family][:2]
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def run(out,seeds,grid=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=seeds
    data,partitions,input_hashes=load_development_inputs(ROOT,cfg)
    # The search has no legitimate need for test features either.
    for partition in partitions.values():
        data['x'][partition['test']]=np.nan
    grid=candidates() if grid is None else grid
    code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')}
    protocol={'version':1,'seeds':seeds,'candidates':grid,'input_hashes':input_hashes,'code_hashes':code_hashes,
              'mapping_targets':{'all_report_agreement_gt':.8,'abnormal_fnr_lt':.05,'fnr_lt_fpr':True},
              'gate_target':{'post_gate_coverage_gt':.75},'positive':'reference_MA_lt_50_or_gt_70',
              'fnr':'abnormal_predicted_within / all_reference_abnormal (all-report)',
              'fpr':'within_predicted_abnormal / all_reference_within (all-report)',
              'coverage':'after_trust_gate; not a substitute for all-report performance',
              'search':'predeclared development-only candidates; no gate fitted in this stage',
              'selection':'feasible first; otherwise lowest FNR among agreement>80%; report frontier',
              'no_test_features_or_labels':True,'paper_unchanged':True,'user_coverage_correction_confirmed':True}
    text=json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)
    protocol_path=out/'PROTOCOL.json'
    if protocol_path.exists() and protocol_path.read_text(encoding='utf-8')!=text:
        raise RuntimeError('Cannot resume under changed protocol/code/inputs')
    protocol_path.write_text(text,encoding='utf-8')
    result_path=out/'CANDIDATES.jsonl'
    records=[json.loads(line) for line in result_path.read_text(encoding='utf-8').splitlines()] if result_path.exists() else []
    done={r['candidate']['id'] for r in records};cache={}
    pool=next(iter(partitions.values()))['pool'];reference=data['y'][pool,2]
    for number,c in enumerate(grid):
        if c['id'] in done:
            continue
        start=time.time()
        try:
            prediction=development_predictions(data,partitions,c,cache)
            metrics=assess(np.broadcast_to(reference,prediction.shape),prediction)
            np.savez_compressed(out/'private'/f"{c['id']}.npz",prediction=prediction,reference=reference,pool=pool,seeds=seeds)
            row={'candidate':c,'status':'complete','metrics':metrics,'seconds':time.time()-start}
        except Exception as error:
            row={'candidate':c,'status':'failed','error':f'{type(error).__name__}: {error}','seconds':time.time()-start}
        with result_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        records.append(row)
        if (number+1)%10==0 or number+1==len(grid):
            summary=make_summary(records,out)
            print(json.dumps({'done':number+1,'total':len(grid),'failed':summary['failed'],'feasible':summary['feasible_count'],'latest_family':c['family']},ensure_ascii=False),flush=True)
    summary=make_summary(records,out)
    print(json.dumps({'status':'complete','completed':summary['completed'],'failed':summary['failed'],'feasible':summary['feasible_count']},ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1000,1005)))
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:
        raise ValueError('run-name must be one path component')
    run(ROOT/'outputs/uteg_mapping_target_runs'/args.run_name,args.seeds)
