"""Fixed10% normal checks; distinct low/high gates and one-error developer reserve."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import run as r
import aux_followup as a
import normal_hard_fn as n
import fixed10_gate_balance as b
import fixed10_gap_end as g
O=b.O/'classwise'

def choose(pred,truth,scores,normal_mask,reserve):
    pc=r.state(pred);tc=r.state(truth);nn=int(normal_mask.sum())
    assert not np.any(normal_mask&(tc!=1))
    if nn==0:return dict(feasible=False,reason='No normal reports')
    banks=[]
    for cls in [0,2]:
        bank=[]
        for gate in b.gate_options(scores,pc==cls):
            action=(pc==cls)&b.gate_action(gate,scores,len(pred));count=int(action.sum());correct=int(np.sum(action&(pc==tc)));ab=int(np.sum(action&(tc!=1)))
            bank.append(dict(gate=gate,count=count,correct=correct,abnormal=ab))
        banks.append(bank)
    best=None
    for low in banks[0]:
        for high in banks[1]:
            count=nn+low['count']+high['count'];correct=nn+low['correct']+high['correct']
            if low['abnormal']+high['abnormal']==0 or correct<=.8*count+reserve:continue
            if best is None or (count,correct/count)>(best['reported'],best['accuracy']):best=dict(low_gate=low['gate'],high_gate=high['gate'],reported=count,accuracy=correct/count,developer_correct=correct)
    return dict(feasible=False,reserve=reserve,reason='No classwise gate meets developer constraints') if best is None else dict(best,feasible=True,reserve=reserve)

def apply(pick,pred,risk,scores):
    if not pick['feasible']:return None
    action=b.normal_action(pick['normal'],pred,risk,scores);pc=r.state(pred)
    action|=(pc==0)&b.gate_action(pick['abnormal']['low_gate'],scores,len(pred))
    action|=(pc==2)&b.gate_action(pick['abnormal']['high_gate'],scores,len(pred))
    return action

def main():
    proto=dict(parent_protocol_hash=r.sha(g.O/'PROTOCOL.json'),budgets=[54,60],repeats=100,arms=a.ARMS,reserves=[0,1],
        fixed='Keep normal_gap_end_80 normal rules exactly, including original90th-percentile cap and original MA/state predictions. Neither normal thresholds nor normal report actions are reselected.',
        low_high='Separate All,HGB,Logistic,Boundary gates for predicted low and high classes; thresholds from each developer predicted class, midpoint in developer score gaps. No test distribution enters thresholds.',
        selection='Developer FN=0 and nonempty normal/true-abnormal reports; developer_correct > 0.8*reported + reserve, where reserve=0 or1. Maximize reported count, then accuracy. No test-based choice between reserve values. All100 repeats retained; joint reporting>75% evaluated separately.',
        scope='Exploratory classwise control motivated by low/high error imbalance in previous diagnostics, not independent validation.',input_hashes=b.protocol()['input_hashes'])
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(proto)
    else:r.write(path,proto)
    d,*_=r.init()
    for budget in [54,60]:
        allrows=[]
        for rep in range(100):
            folder=O/f'dev{budget}/private/round_{rep:02d}';rows=[];policies={};saved={}
            parents=json.loads((g.O/f'dev{budget}/private/round_{rep:02d}/POLICIES.json').read_text())
            for arm in a.ARMS:
                z,risk,base=b.load(budget,rep,arm);parent=parents[arm+'_normal_gap_end_80']['policy'];normal=parent['normal'];ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES};dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                saved['test']=z['test'];saved['dev_rows']=z['dev_rows'];nm=b.normal_action(normal,z['dev_prediction'],risk['dev_risk_B'],ds)
                for reserve in [0,1]:
                    ab=choose(z['dev_prediction'],dy,ds,nm,reserve);p=dict(feasible=ab['feasible'],normal=normal,abnormal=ab)
                    act=apply(p,z['prediction'],risk['test_risk_B'],qs);da=apply(p,z['dev_prediction'],risk['dev_risk_B'],ds);dev=None if da is None else n.extended_metrics(z['dev_prediction'],dy,da);test=None if act is None else n.extended_metrics(z['prediction'],ty,act)
                    if act is not None:
                        np.testing.assert_array_equal(act&(r.state(z['prediction'])==1),b.normal_action(normal,z['prediction'],risk['test_risk_B'],qs))
                        assert dev['FN']==0 and dev['correct']>.8*dev['reported']+reserve and dev['reported_as_normal']>0 and dev['abnormal_released']>0
                    name='classwise_reserve_'+str(reserve);p['development']=dev;policies[arm+'_'+name]=p;saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if act is None else act
                    rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=p['feasible'],development=dev,development_joint_pass=bool(dev and dev['success']),test=test,joint_test_pass=bool(test and test['success']),risk_model='B',cap_quantile=.9))
            r.write(folder/'POLICIES.json',policies);r.write(folder/'RESULTS.json',rows);np.savez_compressed(folder/'ACTIONS.npz',**saved);r.write(folder/'COMPLETE.json',dict(rep=rep,budget=budget));allrows+=rows
        r.write(O/f'dev{budget}/ROUNDS.json',allrows);r.write(O/f'dev{budget}/SUMMARY.json',hsummary(allrows));print('CLASSWISE',budget,'COMPLETE',flush=True)
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in proto['input_hashes'].items())

def hsummary(rows):return b.h.summarize(rows)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
