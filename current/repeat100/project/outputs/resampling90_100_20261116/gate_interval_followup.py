"""Resolve a development threshold's empty score interval at its midpoint."""
import aux_followup as a
import hgb_ensemble_followup as h
import run as r
import fn_candidates as f
import support_followup as s
import numpy as np,json
from threadpoolctl import threadpool_limits
O=a.O/'threshold_interval'

def midpoint(score,threshold):
    if threshold is None:return None,None
    higher=np.unique(score[np.isfinite(score)&(score>threshold)])
    if not len(higher):return threshold,None
    upper=float(higher[0]);middle=threshold+(upper-threshold)/2
    if middle>=upper:middle=threshold
    np.testing.assert_array_equal(np.isfinite(score)&(score<=threshold),np.isfinite(score)&(score<=middle))
    return float(middle),upper

def main():
    d,plan,role,m=r.init()
    r.write(O/'PROTOCOL.json',dict(method='Threshold midpoint of the selected observed development score and the next strictly higher finite development score; no next score => unchanged.',invariance='All development release masks must be exactly unchanged. Test labels and test-score distribution never used to select threshold.',scope='Paired endpoint/midpoint comparison for all100 repeats, both budgets, original HGB/Logistic/Boundary,10-HGB ensemble and B-only development gate selector. B-anchored MA mapping fixed.'))
    for budget in [54,60]:
        rows=[]
        for rep in range(100):
            for arm in a.ARMS:
                z=np.load(a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz')
                picks={k:a.select_threshold(z['dev_score_'+k],z['dev_prediction'],d['medical'][z['dev_rows'],2]) for k in a.RULES}
                decision=a.choose([picks])
                entries=[]
                for rule in a.RULES:
                    pick=r.gates.rc.select_threshold(z['dev_score_'+rule],r.gates.fr.ma_class(z['dev_prediction']),r.gates.fr.ma_class(d['medical'][z['dev_rows'],2]),.2-1e-12)
                    entries.append((rule,z['prediction'],z['test'],z['dev_score_'+rule],z['score_'+rule],pick))
                hz=np.load(h.O/f'dev{budget}/private/round_{rep:02d}/{arm}.npz');hp=json.loads((h.O/f'dev{budget}/private/round_{rep:02d}/{arm}_training.json').read_text(encoding='utf8'))['pick']
                entries.append(('HGB10',hz['prediction'],hz['test'],hz['dev_score'],hz['score'],hp))
                if decision is not None:
                    rule=a.RULES[decision[1]];entries.append(('B_development_gate',z['prediction'],z['test'],z['dev_score_'+rule],z['score_'+rule],dict(picks[rule],selected_rule=rule)))
                else:
                    for variant in ['endpoint','midpoint']:rows.append(dict(rep=rep,budget=budget,arm=arm,rule='B_development_gate',variant=variant,development=dict(feasible=False),test=None,joint_test_pass=False))
                for name,pred,test,devscore,score,pick in entries:
                    t=pick.get('threshold');middle,upper=midpoint(devscore,t)
                    for variant,threshold in [('endpoint',t),('midpoint',middle)]:
                        action=np.zeros(len(test),bool) if threshold is None else np.isfinite(score)&(score<=threshold)
                        value=r.metrics(pred,d['medical'][test,2],action)
                        rows.append(dict(rep=rep,budget=budget,arm=arm,rule=name,variant=variant,original_threshold=t,next_development_score=upper,threshold=threshold,development=pick,test=value,joint_test_pass=f.passed(value)))
        r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',f.summarize(rows,['arm','rule','variant']))
    print('MIDPOINT COMPLETE',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
