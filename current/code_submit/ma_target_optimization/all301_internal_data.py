"""Separate, explicitly authorized301-record internal evaluation contract.

No historical partition or masked development cache is modified. Old test rows
change role only inside this new experiment, never retroactively in old results.
"""
import json
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from .search import ROOT
from . import frozen_moment_mapping as existing
from .full_curve_cost_sensitive import load_waveforms
from .metrics import states
from code_submit.ma_imbalance_mapping.data_contract import sha256,DEFAULT_DATA_PATH

BASE=ROOT/'outputs/uteg_mapping_target_runs'
OUT=BASE/'all301_nested_internal_20261102'
SEED=20261102


def load_inputs():
    auth=json.loads((OUT/'AUTHORIZATION.json').read_text(encoding='utf-8'))
    assert auth['user_answer']=='同意，另建全301条内部验证'
    data,parts,extra,oldblocks,cur,hashes=existing.load_inputs(1,larger=True)
    _,_,extra2,reps,mask,more=load_waveforms(1);hashes.update(more)
    np.testing.assert_array_equal(extra,extra2)
    oldmain=parts[1000]['pool'];oldtest=parts[1000]['test']
    local=np.sort(np.r_[oldmain,extra,oldtest]);source=data['source_rows']['HB_to_SA']
    assert len(local)==301 and len(np.unique(local))==301 and len(source)==79
    assert not set(data['groups'][local]).intersection(data['groups'][source])
    cache=BASE/'frozen_curve_holdout_201_20261102/private/TEST_FEATURES.npz'
    z=np.load(cache);np.testing.assert_array_equal(z['test'],oldtest)
    amp=oldblocks['amp6'][0].copy();amp[oldtest]=z['amp6']
    small=oldblocks['embedding_amp6'][0].copy();small[oldtest]=z['embedding_amp6']
    big=oldblocks['base_amp6'][0].copy();big[oldtest]=z['base_amp6']
    cur=cur.copy();cur[oldtest]=z['current18']
    smooth=reps['smooth900'].copy();smooth[oldtest]=z['smooth900']
    path=ROOT/DEFAULT_DATA_PATH
    with np.load(path,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],data['ids'])
        four=np.asarray(a['x'],float).copy();reference=np.asarray(a['medical'],float)
    assert four.shape[1]==4
    allowed=np.r_[source,local]
    yy=np.full_like(reference,np.nan);yy[allowed]=reference[allowed]
    assert np.isfinite(yy[allowed]).all()
    hidden=np.ones(len(yy),bool);hidden[allowed]=False
    for array in [four,amp,small,big,cur,smooth]:
        array[hidden]=np.nan;assert np.isfinite(array[allowed]).all()
    data=dict(data,x=four,y=yy,source_rows={'HB_to_SA':source},allowed_training_indices=allowed)
    blocks=dict(fitted4=[four],amp6=[amp],base_amp6=[big,amp],embedding_amp6=[small,amp])
    for p in [cache,path,OUT/'AUTHORIZATION.json',__file__]:hashes[str(p)]=sha256(p)
    metadata=dict(local_records=len(local),identity_proxy_groups=len(np.unique(data['groups'][local])),
        local_state_counts=np.bincount(states(yy[local,2]),minlength=3).tolist(),
        hb_records=len(source),hb_state_counts=np.bincount(states(yy[source,2]),minlength=3).tolist(),
        historical_role_counts=dict(main=len(oldmain),extra=len(extra),former_test=len(oldtest)),
        new_usage='All301 local records rotate through disjoint new fold roles; no independent external validation claim.')
    assert metadata['identity_proxy_groups']==287 and metadata['local_state_counts']==[11,253,37]
    return data,local,blocks,cur,smooth,hashes,metadata


def build_splits(local,y,groups):
    """Generate once; labels stratify folds only, never select a favorable split."""
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    result=[]
    for fi,(ta,va) in enumerate(outer.split(np.zeros(len(local)),states(y[local]),groups[local])):
        a,b=local[ta],local[va];inner=[]
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+1+fi)
        for it,iv in splitter.split(np.zeros(len(a)),states(y[a]),groups[a]):
            inner.append(dict(T=a[it].tolist(),V=a[iv].tolist()))
        result.append(dict(A=a.tolist(),B=b.tolist(),inner=inner))
    return result


def validate_splits(local,groups,source,folds):
    seen=[]
    assert len(folds)==5
    for f in folds:
        a,b=np.array(f['A']),np.array(f['B'])
        assert set(a)|set(b)==set(local) and not set(a)&set(b)
        assert not set(groups[a]).intersection(groups[b])
        assert not set(groups[source]).intersection(groups[b])
        seen.extend(b.tolist());iv=[]
        assert len(f['inner'])==3
        for inner in f['inner']:
            t,v=np.array(inner['T']),np.array(inner['V'])
            assert set(t)|set(v)==set(a) and not set(t)&set(v)
            assert not set(groups[t]).intersection(groups[v])
            assert not set(groups[t]).intersection(groups[b])
            assert not set(groups[v]).intersection(groups[b])
            assert not set(groups[source]).intersection(groups[v])
            iv.extend(v.tolist())
        assert sorted(iv)==sorted(a.tolist())
    assert sorted(seen)==sorted(local.tolist())


def prepare(verbose=True):
    data,local,blocks,cur,smooth,hashes,metadata=load_inputs()
    path=OUT/'private/FROZEN_SPLITS.json'
    if path.exists():folds=json.loads(path.read_text())
    else:
        folds=build_splits(local,data['y'][:,2],data['groups'])
        path.write_text(json.dumps(folds,indent=2),encoding='utf-8')
    validate_splits(local,data['groups'],data['source_rows']['HB_to_SA'],folds)
    overview=[]
    for fi,f in enumerate(folds):
        r=dict(fold=fi,inner=[])
        for name in ['A','B']:
            rows=np.array(f[name]);r[name]=dict(records=len(rows),groups=len(np.unique(data['groups'][rows])),
                classes=np.bincount(states(data['y'][rows,2]),minlength=3).tolist())
        for ff in f['inner']:
            r['inner'].append({name:dict(records=len(ff[name]),classes=np.bincount(states(data['y'][ff[name],2]),minlength=3).tolist()) for name in ['T','V']})
        overview.append(r)
    summary=dict(**metadata,sklearn_version=sklearn.__version__,split_seed=SEED,folds=overview,
        checks=dict(group_disjoint=True,each_local_record_evaluated_once=True,inner_folds_exclude_outer_queries=True),
        split_sha256=sha256(path),input_hashes=hashes)
    (OUT/'SPLIT_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    decision=BASE/'VALIDATION_ROLE_DECISION_20261102.json';d=json.loads(decision.read_text(encoding='utf-8'))
    d.update(scope_change_executed=True,new_roles_scope=str(OUT),original_data_roles_preserved=True)
    decision.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
    if verbose:print(json.dumps(dict(metadata=metadata,folds=overview,split_sha256=summary['split_sha256']),ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':prepare()
