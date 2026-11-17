"""Normal-only interval-end control; identical development decisions, fixed10% cap."""
import copy,json
import numpy as np
from threadpoolctl import threadpool_limits
import fixed10_gate_balance as b
import normal_hard_fn as n
import aux_followup as a
import run as r
O=b.O/'normal_gap_end'

def edge(values,threshold,cap=None):
    higher=np.unique(values[np.isfinite(values)&(values>threshold)])
    new=threshold if not len(higher) else float(np.nextafter(higher[0],-np.inf))
    if cap is not None:new=min(new,cap)
    assert new>=threshold
    np.testing.assert_array_equal(values<=new,values<=threshold)
    return new,dict(before=threshold,after=new,next_excluded=float(higher[0]) if len(higher) else None,cap=cap)

def adjust(pick,devpred,devrisk,devscores):
    p=copy.deepcopy(pick)
    if not p['feasible']:return p,{}
    norm=r.state(devpred)==1;npick=p['normal'];limits=npick['risk_thresholds'];audit={}
    limits['low'],audit['low']=edge(devrisk[norm,0],limits['low'])
    limits['high'],audit['high']=edge(devrisk[norm,1],limits['high'],npick['cap_score'])
    gate=npick['gate']
    if gate['rule']!='All':gate['threshold'],audit['gate']=edge(devscores[gate['rule']][norm],gate['threshold'])
    np.testing.assert_array_equal(b.apply(pick,devpred,devrisk,devscores),b.apply(p,devpred,devrisk,devscores))
    return p,audit

def main():
    protocol=dict(parent_protocol_hash=r.sha(b.O/'PROTOCOL.json'),scope='All100 repeats, budgets54/60, single/dual, independent fixed80/85 developer accuracy-floor controls.',
        rule='Only widen normal-release low/high risk and normal-gate cutoffs to the representable value immediately below the next excluded DEVELOPMENT predicted-normal score. If no excluded score, unchanged. High-risk cutoff never exceeds the original fixed90th-percentile cap. Abnormal gate and all fitted predictions/risk scores unchanged.',
        invariance='Every component and complete development release mask identical. Test labels and scores not used to determine cutoffs. Test release mask can only expand through normal predictions; check and report any FN, no selective deletion.',
        input_hashes=b.protocol()['input_hashes'])
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(protocol)
    else:r.write(path,protocol)
    d,*_=r.init();checks=0
    for budget in [54,60]:
        allrows=[]
        for rep in range(100):
            folder=O/f'dev{budget}/private/round_{rep:02d}';rows=[];saved={};audits={}
            for arm in a.ARMS:
                z,risk,base=b.load(budget,rep,arm);original=json.loads((b.O/f'dev{budget}/private/round_{rep:02d}/{arm}_POLICIES.json').read_text());ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
                saved['test']=z['test'];saved['dev_rows']=z['dev_rows'];y=d['medical'][z['test'],2]
                for floor in [80,85]:
                    key='separate_gates_'+str(floor);p,au=adjust(original[key],z['dev_prediction'],risk['dev_risk_B'],ds)
                    act=b.apply(p,z['prediction'],risk['test_risk_B'],qs);before=b.apply(original[key],z['prediction'],risk['test_risk_B'],qs)
                    metric=None
                    if act is not None:
                        assert not np.any(before&~act)
                        normal=r.state(z['prediction'])==1
                        np.testing.assert_array_equal(act&~normal,before&~normal)
                        assert not np.any(act&normal&(risk['test_risk_B'][:,1]>base['cap_score']))
                        metric=n.extended_metrics(z['prediction'],y,act)
                    name='normal_gap_end_'+str(floor);rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=p['feasible'],development=p.get('development'),development_joint_pass=bool(p['feasible'] and p['development']['success']),test=metric,joint_test_pass=bool(metric and metric['success']),risk_model='B',cap_quantile=.9))
                    saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if act is None else act;audits[arm+'_'+name]=dict(policy=p,adjustments=au);checks+=1
            r.write(folder/'POLICIES.json',audits);r.write(folder/'RESULTS.json',rows);np.savez_compressed(folder/'ACTIONS.npz',**saved);r.write(folder/'COMPLETE.json',dict(rep=rep,budget=budget));allrows+=rows
        r.write(O/f'dev{budget}/ROUNDS.json',allrows);r.write(O/f'dev{budget}/SUMMARY.json',b.h.summarize(allrows));print('GAP END',budget,'COMPLETE',flush=True)
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in protocol['input_hashes'].items())
    r.write(O/'CHECKS.json',dict(policies=checks,developer_actions_identical=True,abnormal_test_actions_identical=True,test_normal_actions_only_expand=True,fixed10_cap_unchanged=True,input_hashes_unchanged=True))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
