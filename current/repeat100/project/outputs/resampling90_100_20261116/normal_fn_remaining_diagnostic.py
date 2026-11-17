"""Read-only outcome diagnosis; does not select or modify any policy."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[3]
import json,collections,csv
from pathlib import Path
import numpy as np
import run as r
import aux_followup as a
import normal_hard_fn as n
O=Path(str(_repeat_root / 'experiments' / 'uteg_high_risk_followup_20261116'))

def main():
    d,*_=r.init();y=d['medical'][:,2];out=[];records=collections.defaultdict(list)
    for budget in [54,60]:
        for rep in range(100):
            folder=n.O/f'dev{budget}/private/round_{rep:02d}';saved=np.load(folder/'ACTIONS_RISKS.npz')
            for arm in a.ARMS:
                z=np.load(a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz');sel=json.loads((folder/(arm+'_SELECTION.json')).read_text())
                pick=sel['policies']['normal_check'];act=saved[arm+'_normal_check_action'];t=z['test'];dev=z['dev_rows']
                fn=act&(r.state(z['prediction'])==1)&(r.state(y[t])!=1)
                if not fn.any():continue
                x=d['x'];mean=x[dev].mean(0);sd=np.maximum(x[dev].std(0),1e-9)
                for pos in np.flatnonzero(fn):
                    ix=int(t[pos]);dist=np.sqrt(np.mean(((x[dev]-x[ix])/sd)**2,axis=1));near=np.argsort(dist)[:5]
                    row=dict(budget=budget,rep=rep,arm=arm,index=ix,true_MA=float(y[ix]),predicted_MA=float(z['prediction'][pos]),parameters=x[ix],chosen_risk_model=pick['risk_model'],gate=pick['rule'],gate_score=float(z['score_'+pick['rule']][pos]),gate_threshold=pick['threshold'],nearest_development=[dict(index=int(dev[j]),MA=float(y[dev[j]]),distance=float(dist[j])) for j in near])
                    for name in n.FEATURES:
                        dr=saved['dev_risk_'+name];qr=saved['test_risk_'+name];c=next(v for v in sel['normal_check_candidates'] if v['risk_model']==name)
                        normal=r.state(y[dev])==1;high=r.state(y[dev])==2
                        row[name]=dict(low_score=float(qr[pos,0]),high_score=float(qr[pos,1]),normal_high_percentile=float(np.mean(dr[normal,1]<=qr[pos,1])),high_score_range=[float(dr[high,1].min()),float(dr[high,1].max())],candidate_thresholds=c.get('risk_thresholds'),candidate_feasible=c['feasible'])
                    out.append(row);records[(budget,arm,ix)].append(row)
    summary=[]
    for (budget,arm,ix),vv in records.items():
        summary.append(dict(budget=budget,arm=arm,index=ix,repeated_FN_events=len(vv),true_MA=vv[0]['true_MA'],parameters=vv[0]['parameters'],prediction_range=[min(v['predicted_MA'] for v in vv),max(v['predicted_MA'] for v in vv)],chosen_risk_models=dict(collections.Counter(v['chosen_risk_model'] for v in vv)),B_high_risk_range=[min(v['B']['high_score'] for v in vv),max(v['B']['high_score'] for v in vv)],four_high_risk_range=[min(v['four']['high_score'] for v in vv),max(v['four']['high_score'] for v in vv)],normal_percentile_B=float(np.mean([v['B']['normal_high_percentile'] for v in vv])),normal_percentile_four=float(np.mean([v['four']['normal_high_percentile'] for v in vv])),nearest3_normal_fraction=float(np.mean([q['MA']>=50 and q['MA']<=70 for v in vv for q in v['nearest_development'][:3]]))))
    r.write(O/'diagnostic/FN_EVENTS.json',out);r.write(O/'diagnostic/FN_RECORD_SUMMARY.json',summary)
    print(json.dumps(r.safe(summary),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
