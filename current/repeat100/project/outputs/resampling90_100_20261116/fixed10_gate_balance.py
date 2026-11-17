"""Keep fixed90th-percentile high-risk cap; separate normal/abnormal release gates."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[3]
import argparse,json,time,csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run as r
import aux_followup as a
import normal_hard_fn as n
import high_risk_followup as h
from gate_interval_followup import midpoint
O=Path(str(_repeat_root / 'experiments' / 'uteg_fixed10_gate_balance_20261116'))
GATES=['All',*a.RULES];FLOORS=[.8,.85]

def gate_action(pick,scores,count):
    if pick['rule']=='All':return np.ones(count,bool)
    return np.isfinite(scores[pick['rule']])&(scores[pick['rule']]<=pick['threshold'])

def gate_options(scores,subset):
    yield dict(rule='All',threshold=None,endpoint=None,next_score=None)
    for rule in a.RULES:
        vals=scores[rule][subset];vals=vals[np.isfinite(vals)]
        for t in np.unique(vals)[::-1]:
            middle,upper=midpoint(vals,float(t))
            yield dict(rule=rule,threshold=middle,endpoint=float(t),next_score=upper)

def best_normal(pred,truth,risk,scores):
    normal=r.state(pred)==1;abnormal_truth=r.state(truth)!=1
    masks,limits,cap=h.capped_profiles(pred,risk,.9);masks &= normal[None,:]
    best=None;count=0
    for gate in gate_options(scores,normal):
        active=masks&gate_action(gate,scores,len(pred))[None,:]
        sizes=active.sum(1);bad=np.any(active&abnormal_truth[None,:],axis=1);valid=(sizes>0)&~bad;count+=int(valid.sum())
        if not valid.any():continue
        candidates=np.flatnonzero(valid);j=int(candidates[np.argmax(sizes[candidates])]);size=int(sizes[j])
        # All admissible normal reports are correct under developer FN=0.
        # Maximal normal count cannot harm combined developer accuracy.
        if best is None or size>best['reported']:
            best=dict(gate=gate,risk_thresholds=limits[j],reported=size,cap_score=cap)
    if best is None:return dict(feasible=False,reason='No developer-zero-FN normal reports')
    return dict(best,feasible=True,candidate_count=count)

def normal_action(pick,pred,risk,scores):
    if not pick['feasible']:return np.zeros(len(pred),bool)
    normal=r.state(pred)==1;lim=pick['risk_thresholds']
    return normal&(risk[:,0]<=lim['low'])&(risk[:,1]<=lim['high'])&gate_action(pick['gate'],scores,len(pred))

def choose_abnormal(pred,truth,scores,normal_mask,floor):
    pc=r.state(pred);tc=r.state(truth);assert not np.any(normal_mask&(pc!=1))
    assert not np.any(normal_mask&(tc!=1))
    if not normal_mask.any():return dict(feasible=False,reason='No normal reports')
    best=None;count=0
    for gate in gate_options(scores,pc!=1):
        action=normal_mask|((pc!=1)&gate_action(gate,scores,len(pred)))
        nn=int(action.sum());cc=int(np.sum(action&(pc==tc)));ab=int(np.sum(action&(tc!=1)))
        if not ab or cc<=floor*nn:continue
        count+=1
        if best is None or (nn,cc/nn)>(best['reported'],best['accuracy']):
            best=dict(gate=gate,reported=nn,accuracy=cc/nn)
    if best is None:return dict(feasible=False,reason='No abnormal gate satisfies developer accuracy floor and nonempty abnormalities',accuracy_floor=floor)
    return dict(best,feasible=True,candidate_count=count,accuracy_floor=floor)

def apply(pick,pred,risk,scores):
    if not pick['feasible']:return None
    normal=normal_action(pick['normal'],pred,risk,scores)
    return normal|((r.state(pred)!=1)&gate_action(pick['abnormal']['gate'],scores,len(pred)))

def decisions(z,truth,drisk,baseline):
    dp=z['dev_prediction'];ds={k:z['dev_score_'+k] for k in a.RULES}
    fixed=dict(feasible=True,gate={k:baseline.get(k) for k in ['rule','threshold','endpoint','next_score']},risk_thresholds=baseline['risk_thresholds'],cap_score=baseline['cap_score'])
    retuned=best_normal(dp,truth,drisk,ds);normals={'locked_normal':fixed,'separate_gates':retuned};out={}
    for mode,npick in normals.items():
        mask=normal_action(npick,dp,drisk,ds)
        if npick['feasible']:assert not np.any(mask&(r.state(truth)!=1))
        for floor in FLOORS:
            apick=choose_abnormal(dp,truth,ds,mask,floor) if npick['feasible'] else dict(feasible=False,reason='No normal policy')
            policy=mode+'_'+str(int(round(floor*100)))
            pick=dict(feasible=bool(npick['feasible'] and apick['feasible']),normal=npick,abnormal=apick,accuracy_floor=floor,mode=mode)
            if pick['feasible']:
                action=apply(pick,dp,drisk,ds);pick['development']=n.extended_metrics(dp,truth,action)
                assert pick['development']['FN']==0 and pick['development']['accuracy']>floor
            out[policy]=pick
    return out

def protocol():
    return dict(budgets=[54,60],repeats=100,arms=a.ARMS,accuracy_floors=FLOORS,
        fixed='Original saved B-only anchored MA predictions, B logistic risk heads, split indices, normal high-risk cap exactly the original development90th percentile. No change to original MA, state boundaries, risk model, cap fraction or test labels.',
        policies='Reference B_cap_0.9; locked_normal_80/85 preserve original normal release actions exactly and independently select abnormal gate; separate_gates_80/85 reselect normal low/high risk limits and a normal-specific gate under the same90th-percentile cap, independently select abnormal gate.',
        normal_selection='Enumerate original bounded B risk profiles and All,HGB,Logistic,Boundary gates on developer predicted normals; require zero abnormal-to-normal releases and nonempty normal reports. Maximize correct developer normal reports; ties first listed policy. This independently maximizes any combined attainable accuracy/rate because zero-FN normal predictions are all correct.',
        abnormal_selection='All,HGB,Logistic,Boundary; thresholds from unique developer predicted-abnormal scores, midpoint in adjacent score gaps. Require overall developer accuracy strictly above fixed80% or85%, at least1 released true abnormal. Maximize total report count then accuracy. No feasible candidate remains NA. No chosen floor from test performance.',
        reporting='All4 alternatives applied to all100 draws; jointgoal FNR=0,accuracy>80%,reporting_rate>75%,normal and true abnormal release nonempty. Also report developer joint feasibility and conditional errors; no discarded rounds. Rate below75% remains failed, not filled by forced release.',
        scope='Exploratory follow-up to previously examined outcomes, no new independent validation. Test labels used only to report outcomes; threshold selection uses only developer fold-out predictions.',
        parent_protocol_hash=r.sha(h.O/'PROTOCOL.json'),split_hash=r.sha(r.OUT/'private/SPLITS.json'),input_hashes=n.protocol()['input_hashes'])

def load(budget,rep,arm):
    z=np.load(a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz');risk=np.load(n.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS_RISKS.npz')
    ss=json.loads((h.O/f'dev{budget}/private/round_{rep:02d}/{arm}_SELECTION.json').read_text());base=next(v for v in ss['candidates'] if v['risk_model']=='B' and v['cap_quantile']==.9)
    return z,risk,base

def smoke(d):
    z,risk,base=load(54,0,'HZ_to_SA');dy=d['medical'][z['dev_rows'],2];picked=decisions(z,dy,risk['dev_risk_B'],base)
    z2={k:z[k].copy() for k in z.files};z2['prediction']*=5
    for k in a.RULES:z2['score_'+k][:]=12345
    changed=decisions(z2,dy,risk['dev_risk_B'],base);assert r.safe(picked)==r.safe(changed)
    ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
    old=n.action_for(base,z['prediction'],qs,{'B':risk['test_risk_B']})
    for name,p in picked.items():
        action=apply(p,z['prediction'],risk['test_risk_B'],qs)
        if p['feasible']:
            if name.startswith('locked'):np.testing.assert_array_equal(action&(r.state(z['prediction'])==1),old&(r.state(z['prediction'])==1))
            assert not np.any(action&(r.state(z['prediction'])==1)&(risk['test_risk_B'][:,1]>base['cap_score']))
    # Small brute-force combination audit for normal-bank optimization.
    dp=np.array([55.,60.,65.,75.,40.,68.]);dy=np.array([45.,60.,65.,75.,40.,73.]);dr=np.array([[.8,.2],[.1,.1],[.2,.2],[.1,.8],[.8,.1],[.1,.9]])
    scores={k:np.arange(6)/10 for k in a.RULES};npick=best_normal(dp,dy,dr,scores);mask=normal_action(npick,dp,dr,scores)
    apick=choose_abnormal(dp,dy,scores,mask,.8);actual=int((mask|((r.state(dp)!=1)&gate_action(apick['gate'],scores,6))).sum())
    masks,_,_=h.capped_profiles(dp,dr,.9);best=-1
    for nm in masks:
        for ng in gate_options(scores,r.state(dp)==1):
            nx=nm&(r.state(dp)==1)&gate_action(ng,scores,6)
            if not nx.any() or np.any(nx&(r.state(dy)!=1)):continue
            for ag in gate_options(scores,r.state(dp)!=1):
                ax=nx|((r.state(dp)!=1)&gate_action(ag,scores,6));m=n.extended_metrics(dp,dy,ax)
                if m['FNR']==0 and m['accuracy']>.8:best=max(best,m['reported'])
    assert actual==best
    r.write(O/'SMOKE.json',dict(test_predictions_and_gate_scores_do_not_affect_selection=True,locked_normal_actions_identical=True,fixed90th_percentile_cap_respected=True,normal_abnormal_decomposition_matches_bruteforce=True))
    print('FIXED10 BALANCE SMOKE PASS',flush=True)

def run_one(d,budget,rep):
    folder=O/f'dev{budget}/private/round_{rep:02d}'
    if (folder/'COMPLETE.json').exists():return
    start=time.time();rows=[];saved={}
    for arm in a.ARMS:
        z,risk,base=load(budget,rep,arm);dy=d['medical'][z['dev_rows'],2];picks=decisions(z,dy,risk['dev_risk_B'],base)
        r.write(folder/(arm+'_POLICIES.json'),picks);qs={k:z['score_'+k] for k in a.RULES};testtruth=d['medical'][z['test'],2]
        saved['test']=z['test'];saved['dev_rows']=z['dev_rows']
        original=n.action_for(base,z['prediction'],qs,{'B':risk['test_risk_B']})
        basevalue=n.extended_metrics(z['prediction'],testtruth,original)
        rows.append(dict(budget=budget,rep=rep,arm=arm,policy='baseline_fixed10',feasible=True,development=base['development'],development_joint_pass=base['development']['success'],test=basevalue,joint_test_pass=basevalue['success'],risk_model='B',cap_quantile=.9))
        saved[arm+'_baseline_fixed10']=original
        for name,p in picks.items():
            action=apply(p,z['prediction'],risk['test_risk_B'],qs)
            if action is not None:
                normal=r.state(z['prediction'])==1;assert not np.any(action&normal&(risk['test_risk_B'][:,1]>base['cap_score']))
                if name.startswith('locked'):np.testing.assert_array_equal(action&normal,original&normal)
            met=None if action is None else n.extended_metrics(z['prediction'],testtruth,action)
            rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=p['feasible'],development=p.get('development'),development_joint_pass=bool(p['feasible'] and p['development']['success']),test=met,joint_test_pass=bool(met and met['success']),risk_model='B',cap_quantile=.9))
            saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if action is None else action
    r.write(folder/'RESULTS.json',rows);np.savez_compressed(folder/'ACTIONS.npz',**saved);r.write(folder/'COMPLETE.json',dict(budget=budget,rep=rep,seconds=time.time()-start))
    print('FIXED10 BALANCE',budget,rep,round(time.time()-start,2),flush=True)

def aggregate():
    for budget in [54,60]:
        rows=[v for p in sorted((O/f'dev{budget}/private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/'RESULTS.json').read_text())]
        r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',h.summarize(rows));flat=[]
        for v in rows:
            row={k:v[k] for k in ['budget','rep','arm','policy','feasible','development_joint_pass','joint_test_pass']}
            for side in ['development','test']:
                for key in ['accuracy','reporting_rate','FNR','FPR','FN','FP','reported_as_normal','reported_as_low','reported_as_high','predicted_normal_retention','abnormal_released','true_low_withheld','true_high_withheld']:
                    row[side+'_'+key]=(v[side] or {}).get(key)
            flat.append(row)
        if flat:
            with (O/f'dev{budget}/ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as file:
                wr=csv.DictWriter(file,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
    print('FIXED10 AGGREGATE PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args();proto=protocol();path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(proto)
    else:r.write(path,proto)
    if args.stage=='aggregate':aggregate();return
    d,*_=r.init()
    if args.stage=='smoke':smoke(d)
    else:
        assert (O/'SMOKE.json').exists()
        for rep in range(args.start,args.stop):run_one(d,args.budget,rep)
    assert all(r.sha(r.current.ROOT/'data'/key)==value for key,value in proto['input_hashes'].items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
