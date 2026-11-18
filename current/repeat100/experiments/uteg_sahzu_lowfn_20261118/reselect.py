"""Symmetric low-risk guard with the original development80% abnormal selector."""
import study as s
import numpy as np
import json,csv
from threadpoolctl import threadpool_limits
O=s.O/'symmetric90_reselected';r=s.r;e=s.e

def fit(z,d,base):
    g=s.fit_guard(z,d);ds={k:z['dev_score_'+k] for k in e.a.RULES}
    drisk=np.c_[-d['x'][z['dev_rows'],1],d['x'][z['dev_rows'],1]]
    nm=e.cal.normal_action(base['normal'],z['dev_prediction'],z['dev_risk'],drisk,ds)
    nm=s.apply_guard('symmetric90',z['dev_prediction'],z['dev_risk'],nm,g)
    ab=e.cal.c.choose(z['dev_prediction'],d['medical'][z['dev_rows'],2],ds,nm,0)
    return dict(normal=base['normal'],low_guard=g['low_risk90'],abnormal=ab,feasible=bool(ab['feasible']))
def apply(z,d,pick,dev=False):
    prefix='dev_' if dev else '';ids=z['dev_rows'] if dev else z['test'];pred=z[prefix+'prediction'];raw=z['dev_risk'] if dev else z['test_risk']
    scores={k:z[prefix+'score_'+k] for k in e.a.RULES}
    if not pick['feasible']:return np.zeros(len(pred),bool)
    risk=np.c_[-d['x'][ids,1],d['x'][ids,1]]
    a=e.cal.normal_action(pick['normal'],pred,raw,risk,scores)&(raw[:,0]<=pick['low_guard'])
    pc=r.state(pred)
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:a|=(pc==cls)&e.cal.b.gate_action(pick['abnormal'][key],scores,len(pred))
    return a

with threadpool_limits(limits=1):
    d=s.q.init();rows=[];policies=[];fn=[];smoke=0
    r.write(O/'PROTOCOL.json',dict(change='Keep fixed symmetric90 low-risk normal veto; reselect abnormal gates with the unchanged existing developmentFN0/accuracy>80% selector.',
        reason='Normal-only veto can reduce development accuracy with old abnormal thresholds; maintain the original feasibility constraints after tightening normal reports.',
        no_search=True,all100_each_budget=True,normal_only_guard=True,model_predictions_unchanged=True))
    for budget in [60,54]:
        for rep in range(100):
            folder=s.P/'private'/f'dev{budget}'/f'round_{rep:03d}';saved={}
            for arm in s.q.ARMS:
                with np.load(folder/(arm+'.npz')) as archive:z={k:archive[k].copy() for k in archive.files}
                base=json.loads((folder/(arm+'.json')).read_text())['policies']['common_B'];pick=fit(z,d,base)
                if rep==0:
                    dd=dict(d,medical=d['medical'].copy());dd['medical'][np.setdiff1d(np.arange(len(d['x'])),z['dev_rows'])]=9999
                    assert r.safe(fit(z,dd,base))==r.safe(pick);smoke+=1
                a=apply(z,d,pick);da=apply(z,d,pick,True)
                dev=e.n.extended_metrics(z['dev_prediction'],d['medical'][z['dev_rows'],2],da);met=e.n.extended_metrics(z['prediction'],z['reference'],a)
                if pick['feasible']:assert dev['FN']==0 and dev['accuracy']>.8 and dev['reported_as_normal']>0 and dev['abnormal_released']>0
                rows.append(dict(budget=budget,rep=rep,arm=arm,feasible=pick['feasible'],development=dev,test=met))
                policies.append(dict(budget=budget,rep=rep,arm=arm,policy=pick));saved[arm+'_test']=z['test'];saved[arm+'_action']=a
                flag=a&(r.state(z['prediction'])==1)&(r.state(z['reference'])!=1)
                for j in np.flatnonzero(flag):fn.append(dict(budget=budget,rep=rep,arm=arm,record=str(d['ids'][z['test'][j]]),MA=float(z['reference'][j]),prediction=float(z['prediction'][j])))
            path=O/'private'/f'dev{budget}';path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/f'round_{rep:03d}.npz',**saved)
            if rep%25==0:print('RESELECT',budget,rep+1,flush=True)
    summary=[]
    for budget in [60,54]:
        for arm in s.q.ARMS:
            rr=[x for x in rows if x['budget']==budget and x['arm']==arm];assert len(rr)==100
            v=dict(budget=budget,arm=arm,feasible_rounds=sum(x['feasible'] for x in rr),FN_rounds=sum(x['test']['FN']>0 for x in rr),FN_events=sum(x['test']['FN'] for x in rr),
                zero_normal_rounds=sum(x['test']['reported_as_normal']==0 for x in rr),accuracy_gt80_rounds=sum(x['test']['accuracy'] is not None and x['test']['accuracy']>.8 for x in rr),reporting_gt80_rounds=sum(x['test']['reporting_rate']>.8 for x in rr))
            for k in ['accuracy','reporting_rate','FNR','FPR','MAE','R2']:
                aa=np.array([x['test'][k] for x in rr if x['test'][k] is not None]);v[k+'_mean']=float(aa.mean()) if len(aa) else None;v[k+'_sd']=float(aa.std(ddof=1)) if len(aa)>1 else None;v[k+'_n']=len(aa)
            summary.append(v)
    for name,vals in [('ROUNDS',rows),('POLICIES',policies),('FN_EVENTS',fn),('SUMMARY',summary)]:r.write(O/(name+'.json'),vals)
    with (O/'SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    r.write(O/'CHECKS.json',dict(passed=True,nondevelopment_label_poison_checks=smoke,feasible_development_FN0_accuracy80_verified=True,all100_retained=True))
    print('DONE',len(rows),flush=True)
