"""C60 versus pre-existing C100 training support on identical C60 validation rows.

The extra 40 were frozen historically and are disjoint from the current test
cohort. This is a separate label-budget diagnostic, never mixed with C60 results.
"""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .search import ROOT,training_view,make_summary
from .models import fit_candidate
from .representation_search import fit_classmean
from .metrics import assess
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs,sha256


def load_extra(data,parts,cfg):
    path=ROOT/'outputs/uteg_d2_ci_native85_20261007_012322/ROLE_FREEZE.json'
    role=json.loads(path.read_text(encoding='utf-8-sig'));anchor=next(iter(parts.values()))
    if set(role['C60'])!=set(anchor['pool']) or set(role['test'])!=set(anchor['test']):raise ValueError('Historical role mismatch')
    extra=np.asarray(role['added'],int)
    if len(extra)!=40 or set(role['C100'])!=set(anchor['pool'])|set(extra):raise ValueError('Invalid C100 definition')
    source=data['source_rows']['HZ+HB_to_SA'];test=anchor['test']
    for other in [source,test]:
        if set(extra)&set(other) or set(data['groups'][extra])&set(data['groups'][other]):raise ValueError('Expanded training overlap')
    if not np.all(data['centers'][extra]=='SAHZU'):raise ValueError('Unexpected target center')
    # Read only the established extra-development references into the masked copy.
    with np.load(ROOT/cfg['data_path'],allow_pickle=False) as z:data['y'][extra]=z['medical'][extra]
    if not np.isfinite(data['y'][extra]).all():raise ValueError('Invalid extra-development labels')
    if np.isfinite(data['y'][test]).any():raise ValueError('Test labels became visible')
    data['allowed_training_indices']=np.unique(np.r_[data['allowed_training_indices'],extra])
    data['x'][test]=np.nan
    return extra,{str(path):sha256(path)}


def selected_candidates():
    out=[]
    # Pre-existing development choices only; no test-based choice.
    for run in ['goal_mapping_refine_v1','goal_hierarchical_screen_v1']:
        s=json.loads((ROOT/'outputs/uteg_mapping_target_runs'/run/'SUMMARY.json').read_text(encoding='utf-8'))
        if run=='goal_mapping_refine_v1':
            out.extend(r['candidate'] for r in s['best_accuracy_qualified'][:1]+s['lowest_fnr'][:1])
        else:out.extend(r['candidate'] for r in s['lowest_fnr'][:1])
    for alpha,cap in [(.01,1.),(.01,5.),(.01,20.),(.1,5.)]:
        out.append(dict(id=f'ridge_a{alpha}_w{cap}',family='ridge',features='linear',alpha=alpha,cap=cap))
    for learner,cost in [('logistic',1.),('logistic',4.),('logistic',16.),('svm',4.)]:
        out.append(dict(id=f'class_{learner}_{cost}',family='class_mean',learner=learner,C=1.,cost=cost))
    return out


def fit_any(c,data,A,cache):
    if c['family']=='class_mean':return fit_classmean(data['x'][A],data['y'][A,2],c)
    if c['family']=='hierarchical':
        from .hierarchical_search import fit
        return fit(training_view(data,A),c)
    return fit_candidate(c,training_view(data,A),cache)


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_learning_curve_v1';out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1020))
    data,parts,hashes=load_development_inputs(ROOT,cfg);extra,more=load_extra(data,parts,cfg);hashes.update(more)
    pool=next(iter(parts.values()))['pool'];pos={int(v):i for i,v in enumerate(pool)};candidates=selected_candidates()
    protocol={'candidates':candidates,'seeds':cfg['seeds'],'input_hashes':hashes,'code_hash':sha256(__file__),
              'evaluation_rows':'same 60 target development records and same held-out folds for both budgets',
              'C60':'train on A, validate on B','C100':'train on A plus up to 40 historical extra-development rows, validate on same B; exclude any groups shared with B',
              'extra_support':{'low':int(np.sum(data['y'][extra,2]<50)),'within':int(np.sum((data['y'][extra,2]>=50)&(data['y'][extra,2]<=70))),'high':int(np.sum(data['y'][extra,2]>70))},
              'test_access':'none; original 201 test features and labels masked',
              'scope':'budget diagnostic, no claim of equal budgets across C60 versus C100'}
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');rows=[];cache={}
    path=out/'CANDIDATES.jsonl'
    if path.exists():raise RuntimeError('Use completed existing result; do not overwrite')
    for number,c in enumerate(candidates):
        for budget in [60,100]:
            start=time.time();candidate={**c,'base_id':c['id'],'id':c['id']+f'_C{budget}','budget':budget}
            try:
                preds=[];extra_counts=[]
                for seed,part in parts.items():
                    p=np.full(len(pool),np.nan)
                    for fold in part['folds']:
                        A,B=fold['A'],fold['B'];add=extra[~np.isin(data['groups'][extra],data['groups'][B])] if budget==100 else np.array([],int)
                        train=np.r_[A,add];extra_counts.append(len(add))
                        if set(data['groups'][train])&set(data['groups'][B]):raise ValueError('Training/validation group overlap')
                        model=fit_any(c,data,train,cache);p[[pos[int(v)] for v in B]]=model.predict(data['x'][B])
                    preds.append(p)
                preds=np.asarray(preds);y=data['y'][pool,2];m=assess(np.broadcast_to(y,preds.shape),preds)
                np.savez_compressed(out/'private'/f"{candidate['id']}.npz",prediction=preds,reference=y,pool=pool,seeds=cfg['seeds'])
                row={'candidate':candidate,'status':'complete','metrics':m,'extra_training_count_range':[min(extra_counts),max(extra_counts)],'seconds':time.time()-start}
            except Exception as error:row={'candidate':candidate,'status':'failed','error':str(error)}
            rows.append(row)
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        s=make_summary(rows,out);print(json.dumps({'pairs':number+1,'total':len(candidates),'failed':s['failed'],'feasible':s['feasible_count']}),flush=True)
    return make_summary(rows,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
