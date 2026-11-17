"""Read-only decomposition of fixed10% risk policy failures."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[3]
import json
from pathlib import Path
import numpy as np
import run as r
import aux_followup as a
import normal_hard_fn as n
import high_risk_followup as h
O=Path(str(_repeat_root / 'experiments' / 'uteg_fixed10_gate_balance_20261116'))

def main():
    d,*_=r.init();out=[]
    for budget in [54,60]:
        for rep in range(100):
            old=np.load(n.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS_RISKS.npz')
            for arm in a.ARMS:
                z=np.load(a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz');ss=json.loads((h.O/f'dev{budget}/private/round_{rep:02d}/{arm}_SELECTION.json').read_text());p=next(v for v in ss['candidates'] if v['risk_model']=='B' and v['cap_quantile']==.9)
                normal=r.state(z['prediction'])==1;truth=d['medical'][z['test'],2];tc=r.state(truth);pc=r.state(z['prediction']);risk=old['test_risk_B'];scores={k:z['score_'+k] for k in a.RULES}
                release=n.action_for(p,z['prediction'],scores,{'B':risk})
                cap=~normal|(risk[:,1]<=p['cap_score']);safety=~normal|((risk[:,0]<=p['risk_thresholds']['low'])&(risk[:,1]<=p['risk_thresholds']['high']));gate=scores[p['rule']]<=p['threshold']
                group=lambda m:dict(n=int(m.sum()),normal_predictions=int(np.sum(m&normal)),abnormal_predictions=int(np.sum(m&~normal)),correct=int(np.sum(m&(tc==pc))),FP=int(np.sum(m&(tc==1)&~normal)),FN=int(np.sum(m&(tc!=1)&normal)))
                met=n.extended_metrics(z['prediction'],truth,release)
                out.append(dict(budget=budget,rep=rep,arm=arm,rule=p['rule'],test=met,development=p['development'],cap_only=group(cap),risk_limits_only=group(safety),withheld_by_gate_despite_risk_pass=group(safety&~gate),withheld_by_extra_risk_limits=group(cap&~safety),abnormal_reports_withheld=group(~normal&~release),normal_reports_withheld=group(normal&~release)))
    r.write(O/'DIAGNOSTIC_ALL_ROUNDS.json',out)
    for budget in [54,60]:
        for arm in a.ARMS:
            rr=[v for v in out if v['budget']==budget and v['arm']==arm];bad=[v for v in rr if not v['test']['success']]
            print(budget,arm,'failures',len(bad),'cap-only_FN',sum(v['cap_only']['FN'] for v in rr),flush=True)
            for v in bad:print(json.dumps(r.safe({k:v[k] for k in ['rep','rule','withheld_by_gate_despite_risk_pass','withheld_by_extra_risk_limits','abnormal_reports_withheld']})),flush=True)

if __name__=='__main__':main()
