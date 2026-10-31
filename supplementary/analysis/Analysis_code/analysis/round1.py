from pathlib import Path
import csv, hashlib, json
import numpy as np
import joblib
import os

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['UTEG_ARTIFACT_ROOT']).resolve()
OLD = OUT / 'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
SEQ = OUT / 'uteg_current_ma_sequence_20261026/actions'
EXT = OUT / 'uteg_ridge_dual_head_external_eval_20261026_191117/actions'
DATA = OUT / 'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
DIRECTIONS = ['HZ_to_SA', 'HZ+HB_to_SA']
SEEDS = np.arange(1000, 1020)
hashes = {}

def remember(p):
    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def npz(p):
    with np.load(remember(p), allow_pickle=False) as z:
        return dict(z)

def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def table(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def bootstrap(groups):
    ug = np.unique(groups)
    draw = np.random.default_rng(20261030).integers(0, len(ug), (5000, len(ug)))
    def stat(num, den):
        num = np.asarray(num, float)
        den = np.broadcast_to(den, num.shape)
        gn = np.array([num[:, groups == g].sum() for g in ug])
        gd = np.array([den[:, groups == g].sum() for g in ug])
        db = gd[draw].sum(1); valid = db > 0
        value = float(gn.sum()/gd.sum()) if gd.sum() else None
        ci = np.quantile(gn[draw].sum(1)[valid]/db[valid], [.025, .975]).tolist() if valid.any() else [None, None]
        return dict(estimate=value, lo=ci[0], hi=ci[1], valid_resamples=int(valid.sum()))
    return stat

def main():
    assert not (HERE/'RESULTS.json').exists(), 'Use a new round directory, never overwrite.'
    remember(HERE/'PROTOCOL.md'); remember(Path(__file__))
    D = npz(DATA)
    result = {}; destinations = []; times = []; seedrows = []; checks = []
    for direction in DIRECTIONS:
        z = npz(OLD/'eval/predictions'/f'{direction}.npz')
        idx = z['test_global_index']; groups = D['groups'][idx]
        truth = np.where(D['medical'][idx,2]<50,-1,np.where(D['medical'][idx,2]>70,1,0))
        assert len(idx)==201 and np.array_equal(z['seed'],SEEDS)
        assert np.array_equal(z['test_ids'],D['ids'][idx]) and z['input_qc'].all()
        constants = []
        for seed in SEEDS:
            p=OLD/'models'/direction/'target15'/f'SAHZU_CFULL_seed{seed}.joblib'
            model=joblib.load(remember(p)); pool=np.asarray(model['pool'])
            assert len(pool)==60 and np.array_equal(model['test'],idx)
            assert not set(D['groups'][pool]) & set(groups)
            assert not set(model['source_rows']) & set(pool.tolist()+idx.tolist())
            constants.append(float(np.median(D['medical'][pool,2])))
        stat=bootstrap(groups); one=np.ones((20,201))
        errors={'Target-pool median':abs(np.array(constants)[:,None]-D['medical'][idx,2]),
                'Source mapping':abs(z['raw_med4'][:,:,2]-D['medical'][idx,2]),
                'Complete adaptation':abs(z['final_med4'][:,:,2]-D['medical'][idx,2])}
        prior={name:stat(e,one) for name,e in errors.items()}
        prior['paired_median_minus_complete']=stat(errors['Target-pool median']-errors['Complete adaptation'],one)
        prior['constant_range_mm']=[min(constants),max(constants)]
        checkpoints={}
        for horizon in [300,600]:
            q=npz(SEQ/f'T{horizon}'/f'{direction}.npz')
            assert np.array_equal(q['ids'].astype(str),D['ids'][idx].astype(str))
            assert np.array_equal(q['groups'],groups) and np.array_equal(q['seed'],SEEDS)
            checkpoints[horizon]=(q['action'].astype(bool),q['pred'])
        q=npz(EXT/f'{direction}.npz')
        assert np.array_equal(q['test_ids'].astype(str),D['ids'][idx].astype(str))
        checkpoints[900]=(q['unified_action'].astype(bool),q['final_ma_class'])
        policies={}; arrays={}
        for name,schedule in [('5-10-15',[300,600,900]),('10-15',[600,900]),('15',[900])]:
            released=np.zeros((20,201),bool); pred=np.full((20,201),9); time=np.full((20,201),-1)
            for horizon in schedule:
                a,p=checkpoints[horizon]; new=a & ~released
                pred[new]=p[new]; time[new]=horizon; released |= a
            good=released & (pred==truth); bad=released & ~good
            arrays[name]=dict(correct=good,incorrect=bad,unreported=~released,coverage=released)
            policies[name]={key:stat(val,one) for key,val in arrays[name].items()}
            policies[name]['agreement']=stat(good,released)
            policies[name]['counts']={key:int(val.sum()) for key,val in arrays[name].items()}
            policies[name]['zero_report_seeds']=int((released.sum(1)==0).sum())
            assert np.all(good.astype(int)+bad+(~released)==1)
            if name=='5-10-15':
                original=npz(SEQ/f'{direction}_first_report.npz')
                assert np.array_equal(original['reported'],released)
                assert np.array_equal(original['first_class'],pred) and np.array_equal(original['first_time'],time)
                assert np.array_equal(original['truth'],truth)
            for label in [-1,0,1]:
                mask=np.broadcast_to(truth==label,one.shape); denom=int(mask.sum())
                for dest in [-1,0,1,9]:
                    count=int((mask & (pred==dest)).sum())
                    destinations.append(dict(direction=direction,policy=name,true_state=label,report_state=dest,n=count,N=denom,proportion=count/denom if denom else None))
            for horizon in [300,600,900,-1]:
                times.append(dict(direction=direction,policy=name,time_seconds=horizon,n=int((time==horizon).sum()),N=4020))
            for k,seed in enumerate(SEEDS):
                count=int(released[k].sum()); correct=int(good[k].sum())
                seedrows.append(dict(direction=direction,policy=name,seed=int(seed),correct=correct,incorrect=int(bad[k].sum()),unreported=int((~released[k]).sum()),coverage=count/201,agreement=correct/count if count else None))
        differences={}
        for a,b in [('5-10-15','10-15'),('10-15','15'),('5-10-15','15')]:
            differences[a+' minus '+b]={key:stat(arrays[a][key].astype(int)-arrays[b][key].astype(int),one) for key in arrays[a]}
        result[direction]=dict(records=201,groups=len(np.unique(groups)),seeds=20,prior=prior,policies=policies,paired_differences=differences)
        checks.append(dict(direction=direction,original_sequence_reproduced=True,pool_test_groups_disjoint=True,records=201,groups=len(np.unique(groups))))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    dump('RESULTS.json',result); dump('INPUT_HASHES.json',hashes)
    table('class_destinations.csv',destinations); table('first_report_times.csv',times); table('policy_seeds.csv',seedrows)
    dump('CHECKS.json',dict(checks=checks,input_hashes_unchanged=True,new_fits=0,new_thresholds=0,resamples=5000))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
