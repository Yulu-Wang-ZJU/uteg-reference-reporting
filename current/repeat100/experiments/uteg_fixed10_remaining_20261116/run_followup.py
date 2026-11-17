"""Finite developer-only abnormal-gate controls; frozen normal masks and 10% cap."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import sys, json, copy, csv, argparse
from pathlib import Path
sys.path.insert(0,str(_repeat_root / 'project' / 'outputs/resampling90_100_20261116'))
import numpy as np
from threadpoolctl import threadpool_limits
import run as r
import aux_followup as a
import normal_hard_fn as n
import fixed10_gate_balance as b
import fixed10_classwise as c
import fixed10_gap_end as g
O=Path(__file__).parent
NAMES=['baseline','gap_lower','gap_upper','rank_mean','rank_median','rank_max']

def ranks(devpred,devscores,pred,scores,mode):
    out=np.zeros(len(pred)); pc=r.state(pred); dc=r.state(devpred)
    for cls in [0,2]:
        columns=[]
        for rule in a.RULES:
            reference=np.sort(devscores[rule][dc==cls]); vals=scores[rule][pc==cls]
            if len(reference):
                columns.append((np.searchsorted(reference,vals,side='left')+np.searchsorted(reference,vals,side='right'))/(2*len(reference)))
            else: columns.append(np.ones(len(vals)))
        values=np.array(columns)
        out[pc==cls]=getattr(np,mode)(values,axis=0)
    return out

def choose(pred,truth,score,normal):
    pc=r.state(pred);tc=r.state(truth);nn=int(normal.sum());banks=[]
    assert not np.any(normal&(tc!=1))
    for cls in [0,2]:
        vals=np.unique(score[pc==cls]); thresholds=[None]
        if len(vals):thresholds += [float(x) for x in vals[::-1]]+[float(np.nextafter(vals[0],-np.inf))]
        bank=[]
        for t in thresholds:
            mask=(pc==cls) if t is None else (pc==cls)&(score<=t)
            bank.append((t,int(mask.sum()),int(np.sum(mask&(pc==tc))),int(np.sum(mask&(tc!=1)))))
        banks.append(bank)
    best=None
    for lo in banks[0]:
        for hi in banks[1]:
            count=nn+lo[1]+hi[1];correct=nn+lo[2]+hi[2]
            if nn==0 or lo[3]+hi[3]==0 or correct<=.8*count:continue
            if best is None or (count,correct/count)>(best['reported'],best['accuracy']):
                best=dict(reported=count,accuracy=correct/count,low=lo[0],high=hi[0])
    return dict(feasible=False) if best is None else dict(best,feasible=True)

def rank_action(pick,pred,score,normal):
    if not pick['feasible']:return None
    act=normal.copy();pc=r.state(pred)
    for cls,name in [(0,'low'),(2,'high')]:
        act |= (pc==cls) & (True if pick[name] is None else score<=pick[name])
    return act

def gap_policy(parent,dp,ds,mode):
    p=copy.deepcopy(parent)
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:
        gate=p['abnormal'][key]
        if gate['rule']=='All':continue
        vals=ds[gate['rule']][r.state(dp)==cls]
        old=gate['threshold']
        if mode=='upper':gate['threshold'],_=g.edge(vals,old)
        else:gate['threshold']=float(np.max(vals[vals<=old]))
        np.testing.assert_array_equal(vals<=old,vals<=gate['threshold'])
    return p

def plans(parent,dp,dy,drisk,ds):
    normal=b.normal_action(parent['normal'],dp,drisk,ds)
    result={'baseline':parent}
    for mode in ['lower','upper']:result['gap_'+mode]=gap_policy(parent,dp,ds,mode)
    for mode in ['mean','median','max']:
        score=ranks(dp,ds,dp,ds,mode);result['rank_'+mode]=choose(dp,dy,score,normal)
    return result

def actions(name,pick,parent,dp,ds,pred,risk,scores):
    if name.startswith('rank_'):
        score=ranks(dp,ds,pred,scores,name.split('_')[1])
        return rank_action(pick,pred,score,b.normal_action(parent['normal'],pred,risk,scores))
    return c.apply(pick,pred,risk,scores)

def main(smoke=False):
    protocol=dict(parent_sha=r.sha(c.O/'PROTOCOL.json'),budgets=[54,60],repeats=100,arms=a.ARMS,candidates=NAMES,
        fixed='All MA predictions, normal release masks, B risk heads, original 90th-percentile developer cap, splits unchanged.',
        gap='Lower/upper endpoints of the empty developer abnormal-score intervals; preserve every developer release action.',
        rank='Within each predicted abnormal class, convert each HGB/Logistic/Boundary score to a midrank using developer predictions only; mean, median and maximum are separate fixed controls. Query ranks use the developer empirical distribution. No query/test distribution enters fitting.',
        selection='For each rank control select low/high thresholds separately from developer scores (plus All/None); require developer FN=0, accuracy>80%, nonempty normal and true-abnormal reports; maximize report count then accuracy. Rate>75% evaluated without forced release. No per-repeat test choice.',
        scope='Exploratory reuse of existing evaluation data; all candidates and all100 draws retained. No model retraining, input editing or manuscript changes.',input_hashes=b.protocol()['input_hashes'])
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(protocol)
    else:r.write(path,protocol)
    d,*_=r.init();outputs=[];hashes={};checks=0
    for budget in [54,60]:
        rows=[]
        for rep in ([0,9,27,81,85] if smoke else range(100)):
            saved={};policies={}
            for arm in a.ARMS:
                z,risk,_=b.load(budget,rep,arm);parent=json.loads((c.O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text())[arm+'_classwise_reserve_0']
                ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES};dp=z['dev_prediction'];dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                picks=plans(parent,dp,dy,risk['dev_risk_B'],ds)
                baseline=c.apply(parent,z['prediction'],risk['test_risk_B'],qs)
                olddev=c.apply(parent,dp,risk['dev_risk_B'],ds)
                saved['test']=z['test'];saved['dev_rows']=z['dev_rows']
                for name,pick in picks.items():
                    da=actions(name,pick,parent,dp,ds,dp,risk['dev_risk_B'],ds)
                    act=actions(name,pick,parent,dp,ds,z['prediction'],risk['test_risk_B'],qs)
                    dev=None if da is None else n.extended_metrics(dp,dy,da)
                    test=None if act is None else n.extended_metrics(z['prediction'],ty,act)
                    if act is not None:
                        norm=r.state(z['prediction'])==1
                        np.testing.assert_array_equal(act&norm,baseline&norm)
                        assert dev['FN']==0 and dev['accuracy']>.8
                        assert test['FN']==n.extended_metrics(z['prediction'],ty,baseline)['FN']
                        if name.startswith('gap_'):np.testing.assert_array_equal(da,olddev)
                        if name=='gap_upper':assert not np.any(baseline&~act)
                        if name=='gap_lower':assert not np.any(act&~baseline)
                    rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=pick['feasible'],development=dev,development_joint_pass=bool(dev and dev['success']),test=test,joint_test_pass=bool(test and test['success']),risk_model='B',cap_quantile=.9))
                    saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if act is None else act
                    policies[arm+'_'+name]=pick;checks+=1
                if smoke:
                    # Perturb every query score/prediction: developer-selected policies cannot change.
                    again=plans(parent,dp,dy,risk['dev_risk_B'],ds)
                    assert r.safe(again)==r.safe(picks)
                    for mode in ['mean','median','max']:
                        qa=ranks(dp,ds,z['prediction'],qs,mode)
                        order=np.arange(len(qa))[::-1]
                        qb=ranks(dp,ds,z['prediction'][order],{k:v[order] for k,v in qs.items()},mode)
                        np.testing.assert_array_equal(qa[order],qb)
                for source in [a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz',n.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS_RISKS.npz']:
                    hashes[str(source)]=r.sha(source)
            if not smoke:
                folder=O/f'dev{budget}/private/round_{rep:02d}';r.write(folder/'POLICIES.json',policies);np.savez_compressed(folder/'ACTIONS.npz',**saved)
            if rep%20==0:print('PROGRESS',budget,rep+1,flush=True)
        if not smoke:
            r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',b.h.summarize(rows))
            flat=[]
            for row in rows:
                v={k:row[k] for k in ['budget','rep','arm','policy','feasible','development_joint_pass','joint_test_pass']};v['round']=row['rep']+1
                for part in ['development','test']:
                    for key in ['reported','correct','incorrect','accuracy','reporting_rate','FN','FNR','FP','FPR','reported_as_normal','abnormal_released']:
                        v[part+'_'+key]=None if row[part] is None else row[part][key]
                flat.append(v)
            with (O/f'dev{budget}/ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        outputs+=rows
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in protocol['input_hashes'].items())
    assert all(r.sha(Path(k))==v for k,v in hashes.items())
    r.write(O/('SMOKE.json' if smoke else 'CHECKS.json'),dict(rows=checks,normal_actions_unchanged=True,developer_FN_zero=True,gap_developer_actions_identical=True,source_hashes_unchanged=True,all_inputs_unchanged=True,query_rank_permutation_check=smoke))
    if not smoke:r.write(O/'SOURCE_HASHES.json',hashes)
    print('COMPLETE',checks,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.smoke)
