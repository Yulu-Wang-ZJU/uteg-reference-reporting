"""Predeclared development-error budgets15% and strictly below20%; no retrain."""
import support_followup as s
import run as r
import fn_candidates as f
import numpy as np,json
from threadpoolctl import threadpool_limits
def main():
    d,plan,role,m=r.init()
    protocol=dict(error_budgets=[.15,.2-1e-12],selection='Use same nested-development predictions and existing max-reporting threshold selector. 20% minus epsilon enforces development accuracy>80%. No test reference used until outcomes.',models=s.MODES,budgets=[54,60],rules=f.RULES,guards='none',scope='Fixed operating-point sensitivity, separate from strict development-FN=0 selector. Does not claim development FN=0.')
    r.write(s.O/'OPERATING_PROTOCOL.json',protocol)
    for budget in [54,60]:
        rows=[]
        for rep in range(100):
            dest=s.O/f'dev{budget}'/'private'/f'round_{rep:02d}'
            for arm in s.ARMS:
                for mode in s.MODES:
                    z=np.load(dest/(arm+'_'+mode+'.npz'));dp=r.gates.fr.ma_class(z['dev_prediction']);dy=r.gates.fr.ma_class(d['medical'][z['dev_rows'],2])
                    for rule in f.RULES:
                        for error in [.15,.2-1e-12]:
                            pick=r.gates.rc.select_threshold(z['dev_score_'+rule],dp,dy,error);t=pick.get('threshold')
                            action=np.zeros(len(z['test']),bool) if t is None else np.isfinite(z['score_'+rule])&(z['score_'+rule]<=t)
                            value=r.metrics(z['prediction'],d['medical'][z['test'],2],action)
                            rows.append(dict(rep=rep,budget=budget,arm=arm,mode=mode,rule=rule,error_budget=round(error,2),development=pick,test=value,joint_test_pass=f.passed(value)))
        r.write(s.O/f'dev{budget}'/'OPERATING_ROUNDS.json',rows)
        r.write(s.O/f'dev{budget}'/'OPERATING_SUMMARY.json',f.summarize(rows,['arm','mode','rule','error_budget']))
    print('OPERATING COMPLETE',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
