"""Fixed 60mm midpoint separates two developer zero-FN normal checks."""
import copy,json
import run_followup as f
import numpy as np
from threadpoolctl import threadpool_limits
r=f.r;b=f.b;a=f.a;n=f.n;c=f.c;g=f.g
O=f.O/'normal_midpoint'

def select(dp,dy,risk,cap):
    masks,limits,derived=b.h.capped_profiles(dp,risk,.9)
    assert derived==cap
    pc=r.state(dp);tc=r.state(dy);parts=[]
    for lower in [True,False]:
        band=(pc==1)&((dp<=60) if lower else (dp>60))
        active=masks&band[None,:];count=active.sum(axis=1)
        good=~np.any(active&(tc!=1)[None,:],axis=1)
        if not np.any(good&(count>0)):
            parts.append(dict(enabled=False,lower=lower));continue
        ids=np.flatnonzero(good&(count>0));idx=int(ids[np.argmax(count[ids])]);lim=copy.deepcopy(limits[idx])
        for column,key in enumerate(['low','high']):
            lim[key],_=g.edge(risk[band,column],lim[key],cap if key=='high' else None)
        part=dict(enabled=True,lower=lower,limits=lim);parts.append(part)
        observed=band&(risk[:,0]<=lim['low'])&(risk[:,1]<=lim['high'])
        np.testing.assert_array_equal(observed,active[idx])
    return dict(parts=parts,cap_score=cap,midpoint=60)

def normal_action(p,pred,risk):
    out=np.zeros(len(pred),bool);pc=r.state(pred)
    for part in p['parts']:
        if not part['enabled']:continue
        band=(pc==1)&((pred<=60) if part['lower'] else (pred>60));lim=part['limits']
        out|=band&(risk[:,0]<=lim['low'])&(risk[:,1]<=lim['high'])
    assert not np.any(out&(risk[:,1]>p['cap_score']))
    return out

def select_policy(dp,dy,risk,scores,cap,upper):
    normal=select(dp,dy,risk,cap);nm=normal_action(normal,dp,risk)
    abnormal=c.choose(dp,dy,scores,nm,0)
    pick=dict(feasible=abnormal['feasible'],normal=normal,abnormal=abnormal)
    if upper and pick['feasible']:pick=f.gap_policy(pick,dp,scores,'upper')
    return pick

def action(p,pred,risk,scores):
    if not p['feasible']:return None
    out=normal_action(p['normal'],pred,risk);pc=r.state(pred)
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:out|=(pc==cls)&b.gate_action(p['abnormal'][key],scores,len(pred))
    return out

def main(smoke=False):
    proto=dict(parent_sha=r.sha(f.O/'PROTOCOL.json'),budgets=[54,60],repeats=100,arms=a.ARMS,
        normal='Fixed midpoint60mm of clinical normal range50-70mm; two nonoverlapping predicted-normal bands. Original B low/high risk probabilities and original global developer90th-percentile high-risk cap unchanged. Within each band choose existing capped low/high risk rectangles maximizing developer releases with zeroFN. All-normal gate only; empty safe band is withheld, not treated as zero-risk success. Expand thresholds within empty developer band-specific score gaps without crossing the fixed global cap.',
        abnormal='Same developer classwise low/high selection as previous reserve0; fixed midpoint and upper-end abnormal threshold controls separately reported.',
        constraints='No test label/score passed into select_policy. No changing split/MA/risk model/cap/clinical boundaries. All100 bothbudgets andarms retained. No testwise strategy switching.',
        exploratory='This hypothesis follows observed failures on repeatedly explored data; not independent validation.',input_hashes=b.protocol()['input_hashes'])
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(proto)
    else:r.write(path,proto)
    d,*_=r.init();checked=0
    for budget in [54,60]:
        rows=[]
        for rep in ([9,27,81,85] if smoke else range(100)):
            policies={};saved={}
            for arm in a.ARMS:
                z,risk,base=b.load(budget,rep,arm);dp=z['dev_prediction'];ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
                dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                saved['test']=z['test'];saved['dev_rows']=z['dev_rows']
                for upper in [False,True]:
                    name='normal_midpoint'+('_upper' if upper else '')
                    p=select_policy(dp,dy,risk['dev_risk_B'],ds,base['cap_score'],upper)
                    da=action(p,dp,risk['dev_risk_B'],ds);act=action(p,z['prediction'],risk['test_risk_B'],qs)
                    dev=None if da is None else n.extended_metrics(dp,dy,da);test=None if act is None else n.extended_metrics(z['prediction'],ty,act)
                    if dev is not None:assert dev['FN']==0 and dev['accuracy']>.8 and dev['reported_as_normal']>0
                    if smoke:
                        assert r.safe(p)==r.safe(select_policy(dp,dy,risk['dev_risk_B'],ds,base['cap_score'],upper))
                        assert p['normal']['cap_score']==float(np.quantile(risk['dev_risk_B'][r.state(dp)==1,1],.9))
                    policies[arm+'_'+name]=p;saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if act is None else act
                    rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=p['feasible'],development=dev,development_joint_pass=bool(dev and dev['success']),test=test,joint_test_pass=bool(test and test['success']),risk_model='B',cap_quantile=.9));checked+=1
            if not smoke:
                folder=O/f'dev{budget}/private/round_{rep:02d}';r.write(folder/'POLICIES.json',policies);np.savez_compressed(folder/'ACTIONS.npz',**saved)
            if rep%20==0:print('MIDPOINT',budget,rep+1,flush=True)
        if not smoke:
            r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',b.h.summarize(rows))
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in proto['input_hashes'].items())
    r.write(O/('SMOKE.json' if smoke else 'CHECKS.json'),dict(rows=checked,cap_fixed=True,developer_FN_zero=True,inputs_unchanged=True))
    print('MIDPOINT COMPLETE',checked,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.smoke)
