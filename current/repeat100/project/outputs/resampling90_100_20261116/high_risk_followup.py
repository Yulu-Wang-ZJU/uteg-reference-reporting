"""Finite high-risk normal-report controls, trained and selected on development only."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[3]
import argparse,json,time,csv
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import normal_hard_fn as n
import aux_followup as a
import support_followup as s
import run as r
from gate_interval_followup import midpoint

O=Path(str(_repeat_root / 'experiments' / 'uteg_high_risk_followup_20261116'))
FAMILIES=['B','four','four_rbf'];CAPS=[None,.95,.9,.8]

def fit_rbf(x,y,groups,train,query):
    train=np.asarray(train,int);query=np.asarray(query,int)
    assert not set(groups[train])&set(groups[query])
    scaler=StandardScaler().fit(x[train]);xt=scaler.transform(x[train]);xq=scaler.transform(x[query])
    values=[];audit=dict(train=train,query=query,mean=scaler.mean_,scale=scaler.scale_,heads=[])
    for label in [0,2]:
        target=(r.state(y[train])==label).astype(int)
        if np.unique(target).size<2:
            val=np.ones(len(query));head=dict(class_index=label,status='unestimable',positive_count=int(target.sum()))
        else:
            model=SVC(C=1.,gamma=.25,kernel='rbf',class_weight='balanced',tol=1e-6,max_iter=10000)
            model.fit(xt,target);assert model.fit_status_==0
            val=model.decision_function(xq);head=dict(class_index=label,status='fit',positive_count=int(target.sum()),support_rows=train[model.support_],support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,gamma=model._gamma)
        values.append(val);audit['heads'].append(head)
    return np.array(values).T,audit

def rbf_scores(d,sp,dev_rows):
    lookup={int(ix):i for i,ix in enumerate(dev_rows)};dev=np.full((len(dev_rows),2),np.nan);audits=[]
    for part in sp['positions'][:3]:
        v,au=fit_rbf(d['x'],d['medical'][:,2],d['groups'],part['A'],part['B'])
        dev[[lookup[int(ix)] for ix in part['B']]]=v;audits.append(au)
    query,au=fit_rbf(d['x'],d['medical'][:,2],d['groups'],sp['pool'],sp['test']);audits.append(au)
    assert np.isfinite(dev).all() and np.isfinite(query).all()
    return (dev,query),audits

def capped_profiles(pred,risk,quantile):
    masks,limits=n.profiles(pred,risk);normal=r.state(pred)==1
    if quantile is None:return masks,limits,None
    if not normal.any():return masks,limits,None
    cap=float(np.quantile(risk[normal,1],quantile,method='linear'))
    masks &= ~normal[None,:]|(risk[None,:,1]<=cap)
    limits=[None if p is None else dict(low=p['low'],high=min(p['high'],cap)) for p in limits]
    _,idx=np.unique(masks,axis=0,return_index=True);idx=np.sort(idx)
    return masks[idx],[limits[i] for i in idx],cap

def choose(scores,pred,truth,risk,quantile):
    pc=r.state(pred);tc=r.state(truth);masks,limits,cap=capped_profiles(pred,risk,quantile);found=[];counts={}
    for rule in a.RULES:
        score=scores[rule];order=np.argsort(score,kind='stable');ss=score[order]
        ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]&np.isfinite(ss));base=masks[:,order]&np.isfinite(ss)[None,:]
        cumul=lambda flag:np.cumsum(base&flag[order][None,:],axis=1)[:,ends]
        total=cumul(np.ones(len(pred),bool));correct=cumul(pc==tc);fn=cumul((pc==1)&(tc!=1));ab=cumul(tc!=1);normal=cumul(pc==1)
        valid=(total>0)&(correct>.8*total)&(fn==0)&(ab>0)&(normal>0);counts[rule]=int(valid.sum())
        if not valid.any():continue
        ii,jj=np.where(valid);rank=np.lexsort((jj,ii,-correct[ii,jj]/total[ii,jj],-total[ii,jj]));i,j=int(ii[rank[0]]),int(jj[rank[0]])
        endpoint=float(ss[ends[j]]);threshold,upper=midpoint(score,endpoint);act=masks[i]&np.isfinite(score)&(score<=threshold)
        met=n.extended_metrics(pred,truth,act)
        assert met['FN']==0 and met['accuracy']>.8 and met['abnormal_released']>0 and met['reported_as_normal']>0
        found.append(dict(feasible=True,rule=rule,threshold=threshold,endpoint=endpoint,next_score=upper,risk_thresholds=limits[i],development=met,cap_quantile=quantile,cap_score=cap))
    if not found:return dict(feasible=False,cap_quantile=quantile,cap_score=cap,reason='No zero-FN rule with accuracy>80%, normal reports and released true abnormalities',candidate_counts=counts)
    best=min(found,key=lambda v:(-v['development']['reported'],-v['development']['accuracy'],a.RULES.index(v['rule'])));best['candidate_counts']=counts
    return best

def select(candidates):
    valid=[v for v in candidates if v['feasible']]
    if not valid:return dict(feasible=False,reason='All prespecified candidates infeasible')
    return min(valid,key=lambda v:(-v['development']['reported'],-v['development']['accuracy'],FAMILIES.index(v['risk_model']),CAPS.index(v['cap_quantile'])))

def decisions(z,devtruth,risks):
    scores={k:z['dev_score_'+k] for k in a.RULES};candidates=[]
    for family in FAMILIES:
        for quantile in CAPS:
            pick=choose(scores,z['dev_prediction'],devtruth,risks[family][0],quantile);pick['risk_model']=family;candidates.append(pick)
    selected={'uncapped_with_RBF':select([p for p in candidates if p['cap_quantile'] is None])}
    for quantile in CAPS[1:]:selected['cap_'+str(quantile)]=select([p for p in candidates if p['cap_quantile']==quantile])
    selected['development_selected_cap']=select([p for p in candidates if p['cap_quantile'] is not None])
    return candidates,selected

def protocol():
    return dict(budgets=[54,60],repeats=100,arms=a.ARMS,risk_families=FAMILIES,cap_quantiles=CAPS,
        rationale='Previous released FN scores rank near the upper tail of normal development risks. Test a finite conservative normal-release cap, and nonlinear4-parameter discrimination. Diagnostic test outcomes motivate this exploratory family, not numeric per-record rules; no test label selects a threshold or candidate.',
        mapping='Reuse all saved B-only anchored MA predictions and nested gate scores unchanged. Class boundaries remain50/70. No relabelling; a risky predicted-normal report may only be withheld.',
        risk='B/four logistic risks exactly reuse previous crossfit scores. New4-parameter RBF SVC: train-only StandardScaler,C=1,gamma=.25,class_weight=balanced,tol1e-6; separate low and high heads. Same3 grouped development folds; final current pool fit. Decision scores are not claimed calibrated probabilities.',
        cap='Upper bound for normal-release high-risk threshold: fixed95th/90th/80th percentile of DEVELOPMENT predicted-normal high-risk scores. Thus top5/10/20% developer risk tail is withheld; actual test reporting loss is measured, not assumed. No cap is an extra reference. Fixed cap controls all reported, no test-selected winner.',
        selection='For each candidate require developer FN=0,accuracy>80%,normal release>0,true-abnormal release>0. Maximize reporting count thenaccuracy; ties prefer B,four,four_rbf and highercap. All infeasible are NA. reporting_rate>75% defines joint goal; best zero-FN candidates below75% are retained and labelled failed joint goal. Select risks within each fixedcap; final adaptive-cap policy chooses from three capped policies using developer outcomes only.',
        no_large_reporting_loss='Report global>75% target and paired reporting-rate changes versus prior normal-check. All normal release counts and abnormal withholding retained. Empty normal/abnormal reports never count as success.',
        source_protocol_hash=r.sha(n.O/'PROTOCOL.json'),splits_hash=r.sha(r.OUT/'private/SPLITS.json'),input_hashes=n.protocol()['input_hashes'])

def smoke(d,items):
    sp=s.spec(items[0],54);z=np.load(a.O/'dev54/private/round_00/HZ_to_SA_B.npz');old=np.load(n.O/'dev54/private/round_00/ACTIONS_RISKS.npz')
    risks={k:(old['dev_risk_'+k],old['test_risk_'+k]) for k in ['B','four']};risks['four_rbf'],audits=rbf_scores(d,sp,z['dev_rows']);picks,chosen=decisions(z,d['medical'][z['dev_rows'],2],risks)
    original=d['medical'][sp['test']].copy()
    try:
        d['medical'][sp['test']]=9999;again,au=rbf_scores(d,sp,z['dev_rows']);r2=dict(risks,four_rbf=again);p2,c2=decisions(z,d['medical'][z['dev_rows'],2],r2)
        np.testing.assert_array_equal(risks['four_rbf'][0],again[0]);np.testing.assert_array_equal(risks['four_rbf'][1],again[1]);assert r.safe(picks)==r.safe(p2) and r.safe(chosen)==r.safe(c2)
    finally:d['medical'][sp['test']]=original
    p=sp['positions'][0];y=d['medical'][:,2].copy();p1,au1=fit_rbf(d['x'],y,d['groups'],p['A'],p['B']);y[p['B']]=9999;p2,au2=fit_rbf(d['x'],y,d['groups'],p['A'],p['B']);np.testing.assert_array_equal(p1,p2);assert r.safe(au1)==r.safe(au2)
    for model in ['B','four']:
        reference=n.strict_pick({k:z['dev_score_'+k] for k in a.RULES},z['dev_prediction'],d['medical'][z['dev_rows'],2],risks[model][0])
        current=next(v for v in picks if v['risk_model']==model and v['cap_quantile'] is None)
        for key in reference:assert r.safe(reference[key])==r.safe(current[key])
    for cap in CAPS[1:]:
        masks,params,t=capped_profiles(z['dev_prediction'],risks['B'][0],cap);normal=r.state(z['dev_prediction'])==1
        assert not np.any(masks[:,normal]&(risks['B'][0][normal,1]>t)[None,:])
    r.write(O/'SMOKE.json',dict(test_label_poison_invariance=True,fold_label_poison_invariance=True,uncapped_controls_reproduced=True,normal_risk_cap_applied=True))
    print('HIGH RISK SMOKE PASS',flush=True)

def run_one(d,item,budget):
    folder=O/f'dev{budget}/private'/f"round_{item['rep']:02d}"
    if (folder/'COMPLETE.json').exists():return
    start=time.time();sp=s.spec(item,budget);y=d['medical'][:,2]
    old=np.load(n.O/f'dev{budget}/private/round_{item["rep"]:02d}/ACTIONS_RISKS.npz')
    risks={k:(old['dev_risk_'+k],old['test_risk_'+k]) for k in ['B','four']};risks['four_rbf'],audits=rbf_scores(d,sp,old['dev_rows'])
    r.write(folder/'RBF_TRAINING.json',audits);saved=dict(dev_rows=old['dev_rows'],test=old['test'],dev_rbf=risks['four_rbf'][0],test_rbf=risks['four_rbf'][1]);rows=[];fixed=[]
    for arm in a.ARMS:
        z=np.load(a.O/f'dev{budget}/private/round_{item["rep"]:02d}/{arm}_B.npz');np.testing.assert_array_equal(z['dev_rows'],old['dev_rows']);np.testing.assert_array_equal(z['test'],old['test'])
        candidates,chosen=decisions(z,y[z['dev_rows']],risks);r.write(folder/(arm+'_SELECTION.json'),dict(candidates=candidates,chosen=chosen))
        scores={k:z['score_'+k] for k in a.RULES};ds={k:z['dev_score_'+k] for k in a.RULES}
        def score(pick,policy):
            action=n.action_for(pick,z['prediction'],scores,{k:v[1] for k,v in risks.items()});devact=n.action_for(pick,z['dev_prediction'],ds,{k:v[0] for k,v in risks.items()})
            if devact is not None:
                assert r.safe(n.extended_metrics(z['dev_prediction'],y[z['dev_rows']],devact))==r.safe(pick['development'])
                assert pick['development']['FN']==0
            metric=None if action is None else n.extended_metrics(z['prediction'],y[z['test']],action)
            return dict(rep=item['rep'],budget=budget,arm=arm,policy=policy,feasible=pick['feasible'],rule=pick.get('rule'),risk_model=pick.get('risk_model'),cap_quantile=pick.get('cap_quantile'),development=pick.get('development'),development_joint_pass=bool(pick['feasible'] and pick['development']['success']),test=metric,joint_test_pass=bool(metric and metric['success'])),action
        for pick in candidates:
            name=pick['risk_model']+'_cap_'+str(pick['cap_quantile']);row,action=score(pick,name);fixed.append(row)
        for name,pick in chosen.items():
            row,action=score(pick,name);rows.append(row);saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if action is None else action
    r.write(folder/'SELECTED.json',rows);r.write(folder/'FIXED.json',fixed);np.savez_compressed(folder/'RISK_ACTIONS.npz',**saved)
    r.write(folder/'COMPLETE.json',dict(rep=item['rep'],budget=budget,seconds=time.time()-start));print('HIGH RISK',budget,item['rep'],round(time.time()-start,2),flush=True)

def summarize(rows):
    result=[]
    for arm in a.ARMS:
        for policy in dict.fromkeys(v['policy'] for v in rows):
            rr=[v for v in rows if v['arm']==arm and v['policy']==policy];valid=[v for v in rr if v['test'] is not None]
            q=dict(arm=arm,policy=policy,rounds=len(rr),feasible=len(valid),infeasible=len(rr)-len(valid),development_joint=sum(v['development_joint_pass'] for v in rr),test_joint=sum(v['joint_test_pass'] for v in rr),both_joint=sum(v['development_joint_pass'] and v['joint_test_pass'] for v in rr),zero_FN=sum(v['test']['FNR']==0 and not v['test']['no_normal_reports'] for v in valid),undefined_FNR=sum(v['test']['FNR'] is None for v in valid),no_normal=sum(v['test']['no_normal_reports'] for v in valid),risk_families={key:sum(v['risk_model']==key for v in rr) for key in [None,*FAMILIES]},cap_choices={str(key):sum(v['cap_quantile']==key for v in rr) for key in CAPS})
            for name in ['accuracy','reporting_rate','FNR','FPR','FN','FN_low','FN_high','MAE','R2','reported_as_normal','predicted_normal_retention','low_retained_correct_rate','high_retained_correct_rate']:
                vals=[v['test'][name] for v in valid if v['test'][name] is not None]
                q[name]=dict(mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,min=float(min(vals)),max=float(max(vals)),n=len(vals)) if vals else None
            result.append(q)
    return result

def aggregate():
    for budget in [54,60]:
        for kind in ['SELECTED','FIXED']:
            rows=[v for p in sorted((O/f'dev{budget}/private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/(kind+'.json')).read_text())]
            r.write(O/f'dev{budget}'/(kind+'_ROUNDS.json'),rows);r.write(O/f'dev{budget}'/(kind+'_SUMMARY.json'),summarize(rows))
            flat=[]
            for v in rows:
                row={k:v[k] for k in ['rep','budget','arm','policy','feasible','rule','risk_model','cap_quantile','development_joint_pass','joint_test_pass']}
                for side in ['development','test']:
                    for key in ['accuracy','reporting_rate','FNR','FPR','FN','FN_low','FN_high','reported_as_normal','predicted_normal_retention','abnormal_released','true_high_withheld']:
                        row[side+'_'+key]=(v[side] or {}).get(key)
                flat.append(row)
            if flat:
                with (O/f'dev{budget}'/(kind+'_ROUNDS.csv')).open('w',newline='',encoding='utf-8-sig') as file:
                    wr=csv.DictWriter(file,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
    print('HIGH RISK AGGREGATED',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    proto=protocol();path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==r.safe(proto)
    else:r.write(path,proto)
    if args.stage=='aggregate':aggregate();return
    d,*_=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text())
    if args.stage=='smoke':smoke(d,items)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[args.start:args.stop]:run_one(d,item,args.budget)
    assert all(r.sha(r.current.ROOT/'data'/key)==val for key,val in proto['input_hashes'].items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
