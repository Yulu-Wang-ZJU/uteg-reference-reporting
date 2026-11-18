"""SAHZU-only fixed60/244 extension, matched54/60, locked estimators/rules."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import os,sys,json,time,copy,argparse,csv
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_full90_target70_20261117'))
import full90 as f
import numpy as np
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent
e=f.e;r=f.r;c=f.c
ARMS=['target_only_SA','HZ_to_SA','HB_to_SA','HZ+HB_to_SA']
OLD=f.WS/'outputs/resampling90_100_20261116/private/SPLITS.json'

def write(p,x): r.write(p,x)
def init():
    d=f.init();c.OUT=O/'runtime';r.suite.ROOT=O/'runtime';return d
def separate(d,a,b):
    assert not set(a)&set(b)
    assert not set(d['groups'][a])&set(d['groups'][b])
def positions(d,pool,test,seed):
    out=[]
    for j,fold in enumerate(r.folds(pool,d,seed)):
        out.append(dict(A=fold['T'],B=fold['V'],inner=r.folds(fold['T'],d,seed+101+j)))
    out.append(dict(A=pool,B=test,inner=r.folds(pool,d,seed+201)))
    return out
def prepare(d):
    if (O/'private/SPLITS.json').exists():return json.loads((O/'private/SPLITS.json').read_text())
    proposal=json.loads((O/'SPLIT_AUDIT_AND_PROPOSAL.json').read_text())
    assert r.sha(r.current.ROOT/'data/DATA15_CURRENT.npz')==proposal['input_sha256']
    pool=np.array(proposal['candidate']['development']);test=np.array(proposal['candidate']['test'])
    separate(d,pool,test);assert len(pool)==60 and len(test)==244
    assert set(np.r_[pool,test])==set(np.flatnonzero(d['centers']=='SAHZU'))
    old=json.loads(OLD.read_text());items=[];checks=0
    for item in old:
        seed=item['seed'];rng=np.random.default_rng(seed+17000000)
        a=f.sample(pool,d,rng,.9);b=f.sample(test,d,rng,.9)
        assert len(a)==54 and len(b)==220
        assert np.bincount(r.state(d['medical'][a,2]),minlength=3).tolist()==[4,43,7]
        assert np.bincount(r.state(d['medical'][b,2]),minlength=3).tolist()==[7,187,26]
        for budget,train in [(54,a),(60,pool)]:
            sp=dict(pool=train,test=b,positions=positions(d,train,b,seed))
            new=dict(rep=item['rep'],seed=seed,budget=budget,sources=item['sources'],targets={'SAHZU':sp})
            assert new['sources']==old[item['rep']]['sources']
            separate(d,train,b)
            seen=[]
            for pos,part in enumerate(sp['positions']):
                aa=np.array(part['A']);bb=np.array(part['B']);separate(d,aa,bb)
                assert set(aa)<=set(train)
                if pos<3:seen.extend(bb.tolist());assert set(aa)|set(bb)==set(train)
                else:assert set(bb)==set(b)
                inner=[]
                for fold in part['inner']:
                    t=np.array(fold['T']);v=np.array(fold['V']);separate(d,t,v)
                    assert set(t)|set(v)==set(aa);separate(d,t,b);inner+=v.tolist();checks+=1
                assert sorted(inner)==sorted(aa)
            assert sorted(seen)==sorted(train)
            for arm in ARMS:
                src=f.source_rows(new,arm,900,d);separate(d,src,np.r_[train,b])
            items.append(r.safe(new))
    write(O/'AUTHORIZATION.json',dict(answer='最少调整成员，保持60／244和类别数量',scope='SAHZU only. User further requests minimum changes relative to manuscript full60; matched54 retained separately. No tuning.'))
    write(O/'private/SPLITS.json',items)
    write(O/'ROLE_CHANGE.json',proposal['candidate'])
    protected=f.frozen_files();protected[str(OLD)]=r.sha(OLD)
    write(O/'PROTECTED_FILES.json',protected)
    write(O/'PROTOCOL.json',dict(primary='full60/220; matched54/220 separately',repeats=100,
        role_counts={'development':[4,48,8],'test':[8,207,29]},repeat_counts={'dev54':[4,43,7],'dev60':[4,48,8],'test220':[7,187,26]},
        roles='Approved minimum three-for-three member exchange; fixed60/244 role pools across100 repetitions. Identity groups stay on one side.',
        sampling='Independent90% stratified group draws inside fixed pools, half-up rounding/largest remainder. Matched220 test records and original source draws across budgets/arms.',
        mapping='Unchanged B-input RBF anchored-tail MA mapping, asymmetric loss coefficient4, alpha0.01; existing class weighting. All models refitted.',
        reporting='Locked common_B selector from existing manuscript; normal risk90th-percentile cutoff, B bounds, class-specific gates. Thresholds fitted solely on each development subset, no test-driven tuning.',
        scope='15-minute MA mapping/reporting; four target-SAHZU arms only. Other-centre target results and all previous results untouched.',
        statistics='Mean and sampleSD across100 overlapping repeated samples; not independent cohorts or95%confidence interval. Undefined denominators counted explicitly.',
        no_future_zero_FN_guarantee=True))
    write(O/'SPLIT_AUDIT.json',dict(passed=True,items=len(items),nested_folds_checked=checks,group_disjoint=True,source_draws_identical=True,matched_budgets=True))
    return items
def check_counts(bundle):
    z,_,_,_,rows,_,actions=bundle
    for row in rows:
        m=row['test'];cm=np.asarray(m['confusion'])
        assert cm.sum()==m['reported'] and cm.trace()==m['correct']
        assert int(cm[0,1]+cm[2,1])==m['FN'] and int(cm[1,0]+cm[1,2])==m['FP']
        assert int(actions[row['rule']].sum())==m['reported']
def smoke(d,items):
    checks=[];start=time.time()
    for budget in [54,60]:
        item=next(i for i in items if i['rep']==0 and i['budget']==budget)
        for arm in ARMS:
            f.activate(item,d);left=f.bundle(d,item,arm,900);check_counts(left)
            allowed=np.r_[item['targets']['SAHZU']['pool'],c.source_rows(arm,900)]
            forbidden=np.setdiff1d(np.arange(len(d['x'])),allowed)
            saved=d['medical'].copy();savedy=d['y'].copy()
            try:
                d['medical'][forbidden]=12345;d['y'][forbidden]=12345
                f.activate(item,d);right=f.bundle(d,item,arm,900);f.compare_bundle(left,right)
            finally:d['medical'][:]=saved;d['y'][:]=savedy;c._SOURCE.clear()
            checks.append(dict(budget=budget,arm=arm,forbidden_labels_changed_without_affecting_predictions_scores_policy_actions=True))
            print('SMOKE',budget,arm,'PASS',round(time.time()-start,1),flush=True)
    write(O/'SMOKE.json',dict(passed=True,checks=checks,seconds=time.time()-start))
def run(d,item):
    folder=O/'private'/f'dev{item["budget"]}'/f'round_{item["rep"]:03d}'
    if (folder/'COMPLETE.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);rows=[];start=time.time()
    f.activate(item,d)
    for arm in ARMS:
        path=folder/(arm+'.json')
        if path.exists():rows+=json.loads(path.read_text())['rows'];continue
        bundle=f.bundle(d,item,arm,900);check_counts(bundle)
        z,dr,qr,audit,rr,picks,actions=bundle
        for row in rr:row['budget']=item['budget']
        np.savez_compressed(folder/(arm+'.npz'),**z,dev_risk=dr,test_risk=qr,reference=d['medical'][z['test'],2],**{'action_'+k.replace(' ','_'):v for k,v in actions.items()})
        write(path,dict(rows=rr,policies=picks,risk_training=audit));rows+=rr
    write(folder/'RESULTS.json',rows)
    write(folder/'COMPLETE.json',dict(rep=item['rep'],budget=item['budget'],models=4,seconds=time.time()-start))
    print('COMPLETE',item['rep']+1,item['budget'],round(time.time()-start,1),flush=True)
def aggregate():
    rows=[]
    for budget in [54,60]:
        for rep in range(100):
            folder=O/'private'/f'dev{budget}'/f'round_{rep:03d}'
            assert (folder/'COMPLETE.json').exists(),(budget,rep)
            rows+=json.loads((folder/'RESULTS.json').read_text())
    summary=[]
    for budget in [54,60]:
        for arm in ARMS:
            for rule in ['All report','common_B',*e.a.RULES,'B95']:
                rr=[v for v in rows if v['budget']==budget and v['arm']==arm and v['rule']==rule]
                assert len(rr)==100 and len({v['rep'] for v in rr})==100
                result=dict(budget=budget,arm=arm,rule=rule,repeats=100,feasible_rounds=sum(v['feasible'] for v in rr),FN_rounds=sum(v['test']['FN']>0 for v in rr),total_FN=sum(v['test']['FN'] for v in rr),
                    zero_report_rounds=sum(v['test']['reported']==0 for v in rr),zero_normal_report_rounds=sum(v['test']['reported_as_normal']==0 for v in rr))
                for key in ['accuracy','reporting_rate','FNR','FPR','MAE','RMSE','R2','low_recall','high_recall','low_MAE','high_MAE','low_retained_correct_rate','high_retained_correct_rate']:
                    a=np.array([v['test'][key] for v in rr if v['test'][key] is not None],float)
                    result[key+'_n']=len(a);result[key+'_mean']=float(a.mean()) if len(a) else None;result[key+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
                summary.append(result)
    write(O/'ALL_ROUNDS.json',rows);write(O/'SUMMARY.json',summary)
    with (O/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as s:
        w=csv.DictWriter(s,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    protected=json.loads((O/'PROTECTED_FILES.json').read_text());assert all(r.sha(Path(p))==h for p,h in protected.items())
    write(O/'FINAL_CHECKS.json',dict(passed=True,all100_retained_per_budget=True,workflow_evaluations=800,rule_results=len(rows),protected_files_unchanged=True))
    print(json.dumps([v for v in summary if v['rule']=='common_B'],indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','run','aggregate']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=='aggregate':aggregate()
        else:
            d=init();items=prepare(d)
            if args.stage=='prepare':print('PREPARED',len(items),flush=True)
            elif args.stage=='smoke':smoke(d,items)
            else:
                assert json.loads((O/'SMOKE.json').read_text())['passed']
                for item in items:
                    if args.start<=item['rep']<args.stop:run(d,item)
