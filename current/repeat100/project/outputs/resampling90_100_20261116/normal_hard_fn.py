"""Three prespecified development-only policies; MA predictions remain unchanged."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[3]
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run as r
import aux_followup as a
import fn_candidates as f
import support_followup as s
from gate_interval_followup import midpoint

# Keep new artifacts off the nearly full project drive. No existing files removed.
O=Path(str(_repeat_root / 'experiments' / 'uteg_normal_hard_fn_20261116/required_normal_check'))
FEATURES={'B':[1], 'four':[0,1,2,3]}
POLICIES=['existing','hard_zero_FN','normal_check']

def extended_metrics(pred,truth,action):
    v=r.metrics(pred,truth,action);pc=r.state(pred);tc=r.state(truth)
    for k,name in enumerate(['low','normal','high']):
        denom=int(np.sum(pc==k));n=int(np.sum(action&(pc==k)))
        v['predicted_'+name+'_before_gate']=denom
        v['reported_as_'+name]=n
        v['predicted_'+name+'_retention']=n/denom if denom else None
        v['true_'+name+'_released']=int(np.sum(action&(tc==k)))
        v['true_'+name+'_withheld']=int(np.sum(~action&(tc==k)))
    v['FN_low']=int(np.sum(action&(pc==1)&(tc==0)))
    v['FN_high']=int(np.sum(action&(pc==1)&(tc==2)))
    v['withheld']=int(np.sum(~action))
    v['no_normal_reports']=v['reported_as_normal']==0
    v['success']=bool(f.passed(v) and not v['no_normal_reports'])
    return v

def risk_fit(x,y,groups,train,query,cols):
    train=np.asarray(train,int);query=np.asarray(query,int)
    assert not set(groups[train])&set(groups[query])
    sc=StandardScaler().fit(x[train][:,cols]);xt=sc.transform(x[train][:,cols]);xq=sc.transform(x[query][:,cols])
    scores=[];audit=dict(train=train,query=query,columns=cols,mean=sc.mean_,scale=sc.scale_,heads=[])
    for cls in [0,2]:
        target=(r.state(y[train])==cls).astype(int);count=int(target.sum())
        if count in [0,len(train)]:
            score=np.full(len(query),1.);info=dict(class_index=cls,positive_count=count,status='unestimable_conservative_score_one')
        else:
            model=LogisticRegression(C=.1,class_weight='balanced',solver='lbfgs',max_iter=1000,tol=1e-9)
            model.fit(xt,target);score=model.predict_proba(xq)[:,1]
            info=dict(class_index=cls,positive_count=count,status='fit',coef=model.coef_,intercept=model.intercept_,iterations=model.n_iter_)
            assert int(model.n_iter_[0])<1000
        scores.append(score);audit['heads'].append(info)
    return np.stack(scores,axis=1),audit

def risk_scores(x,y,groups,sp,dev_rows):
    lookup={int(v):i for i,v in enumerate(dev_rows)};result={};audits={}
    for name,cols in FEATURES.items():
        dp=np.full((len(dev_rows),2),np.nan);records=[]
        for part in sp['positions'][:3]:
            q,au=risk_fit(x,y,groups,part['A'],part['B'],cols)
            dp[[lookup[int(v)] for v in part['B']]]=q;records.append(au)
        qp,au=risk_fit(x,y,groups,sp['pool'],sp['test'],cols);records.append(au)
        assert np.isfinite(dp).all() and np.isfinite(qp).all()
        result[name]=(dp,qp);audits[name]=records
    return result,audits

def mid_grid(v):
    values=np.unique(v);return np.array([midpoint(v,float(t))[0] for t in values])

def profiles(pred,risks=None):
    normal=r.state(pred)==1;n=len(pred)
    if risks is None:return np.ones((1,n),bool),[None]
    lo=mid_grid(risks[normal,0]);hi=mid_grid(risks[normal,1])
    if not len(lo) or not len(hi):return np.ones((1,n),bool),[None]
    pairs=np.array([(l,h) for l in lo for h in hi])
    masks=~normal[None,:]|((risks[None,:,0]<=pairs[:,0,None])&(risks[None,:,1]<=pairs[:,1,None]))
    # Prefer the more permissive score thresholds when development actions tie.
    masks=masks[::-1];pairs=pairs[::-1]
    _,idx=np.unique(masks,axis=0,return_index=True);idx=np.sort(idx)
    return masks[idx],[dict(low=float(l),high=float(h)) for l,h in pairs[idx]]

def strict_pick(scores,pred,truth,risk=None):
    """No soft fallback: FN=0, accuracy>80%, both normal and abnormal releases."""
    pc=r.state(pred);tc=r.state(truth);masks,params=profiles(pred,risk)
    found=[];counts={}
    for rule in a.RULES:
        score=scores[rule];order=np.argsort(score,kind='stable');ss=score[order]
        ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]&np.isfinite(ss))
        base=masks[:,order]&np.isfinite(ss)[None,:]
        cumulative=lambda flag:np.cumsum(base&flag[order][None,:],axis=1)[:,ends]
        n=cumulative(np.ones(len(pred),bool));correct=cumulative(pc==tc)
        fn=cumulative((pc==1)&(tc!=1));ab=cumulative(tc!=1);normal=cumulative(pc==1)
        valid=(n>0)&(correct>.8*n)&(fn==0)&(ab>0)&(normal>0)
        counts[rule]=int(valid.sum())
        if not valid.any():continue
        ii,jj=np.where(valid);rank=np.lexsort((jj,ii,-correct[ii,jj]/n[ii,jj],-n[ii,jj]));i=int(ii[rank[0]]);j=int(jj[rank[0]])
        endpoint=float(ss[ends[j]]);threshold,upper=midpoint(score,endpoint)
        action=masks[i]&np.isfinite(score)&(score<=threshold)
        met=extended_metrics(pred,truth,action)
        assert met['FN']==0 and met['reported_as_normal']>0 and met['abnormal_released']>0 and met['accuracy']>.8
        found.append(dict(feasible=True,rule=rule,endpoint=endpoint,threshold=threshold,next_score=upper,risk_thresholds=params[i],development=met))
    if not found:return dict(feasible=False,reason='No rule satisfies zero FN, accuracy>80%, nonzero normal reports and released true abnormalities',candidate_counts=counts)
    pick=min(found,key=lambda p:(-p['development']['reported'],-p['development']['accuracy'],a.RULES.index(p['rule'])))
    pick['candidate_counts']=counts;return pick

def action_for(pick,pred,scores,risks):
    if not pick['feasible']:return None
    action=np.isfinite(scores[pick['rule']])&(scores[pick['rule']]<=pick['threshold'])
    limits=pick.get('risk_thresholds')
    if limits is not None:
        normal=r.state(pred)==1;z=risks[pick['risk_model']]
        action &= ~normal|((z[:,0]<=limits['low'])&(z[:,1]<=limits['high']))
    return action

def decide(z,truth,risks):
    pred=z['dev_prediction'];scores={k:z['dev_score_'+k] for k in a.RULES}
    pp={k:a.select_threshold(scores[k],pred,truth) for k in a.RULES};choice=a.choose([pp])
    if choice is None:baseline=dict(feasible=False,reason='Existing selector infeasible')
    else:
        rule=a.RULES[choice[1]];threshold,upper=midpoint(scores[rule],pp[rule]['threshold'])
        act=np.isfinite(scores[rule])&(scores[rule]<=threshold)
        baseline=dict(feasible=True,rule=rule,endpoint=pp[rule]['threshold'],threshold=threshold,next_score=upper,risk_thresholds=None,development=extended_metrics(pred,truth,act))
    hard=strict_pick(scores,pred,truth)
    candidates=[]
    for name,(dr,qr) in risks.items():candidates.append(dict(strict_pick(scores,pred,truth,dr),risk_model=name))
    valid=[(i,p) for i,p in enumerate(candidates) if p['feasible']]
    normal=min(valid,key=lambda ip:(-ip[1]['development']['reported'],-ip[1]['development']['accuracy'],ip[0]))[1] if valid else dict(feasible=False,reason='Neither B nor four-parameter normal checks produced an eligible rule')
    return dict(existing=baseline,hard_zero_FN=hard,normal_check=normal),candidates

def protocol():
    return dict(budgets=[54,60],repeats=100,arms=a.ARMS,policies=POLICIES,
        mapping='Reuse unchanged saved B-only anchored MA predictions and nested gate scores from aux_followup; no MA refit, relabelling or test-dependent replacement.',
        zero_FN='Hard alternatives require development FN=0, accuracy>80%, at least one predicted-normal release and one released true abnormal. Maximize reported count, then accuracy. Reporting rate>75% defines joint target separately; below-target candidates remain explicitly labelled.',
        normal_check='Only veto predicted-normal releases. Independent low-vs-rest and high-vs-rest logistic scores, C=0.1, inverse-class balanced weights, train-only StandardScaler. Compare B alone and all4 parameters. Policy3 always applies a check; policy2 is its no-extra-check control. No prediction labels are changed.',
        risk_training='Same3 grouped outer development folds; a risk score for a development record never uses its label/group. Final scores trained on current local pool only. Source data and test labels do not enter new heads. Scores are rankings, not claimed calibrated clinical probabilities.',
        thresholds='All unique observed development normal-risk scores, midpoint within adjacent normal-score gaps; both low/high thresholds searched jointly with each of HGB,Logistic,Boundary. Main gate threshold also midpoint. Exact developer action invariance checked. Ties prefer B then four; HGB,Logistic,Boundary. Finite prespecified search.',
        pilot='Parent directory retains smoke and3 pilot repeats of an optional-check selector. Before full evaluation policy3 was made an obligatory normal check to distinguish it from policy2; pilot is not pooled into the full100 results.',
        no_degeneracy='No-normal or no-true-abnormal release is not success; report normal release counts and fraction, abnormal withheld counts, plus total reporting rate. No feasible rule=>NA, never FN=0 by empty release.',
        scope='Exploratory reuse of100 overlapping draws, not independent clinical validation. Existing soft selector is a reference, not a zero-development-FN eligible policy.',
        splits_hash=r.sha(r.OUT/'private/SPLITS.json'),source_protocol_hash=r.sha(a.O/'PROTOCOL.json'),
        input_hashes=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes'])

def smoke(d,items):
    sp=s.spec(items[0],54);z=np.load(a.O/'dev54/private/round_00/HZ_to_SA_B.npz');y=d['medical'][:,2]
    risks,audit=risk_scores(d['x'],y,d['groups'],sp,z['dev_rows']);pick,candidates=decide(z,y[z['dev_rows']],risks)
    poisoned=y.copy();poisoned[sp['test']]=9999
    rr,au=risk_scores(d['x'],poisoned,d['groups'],sp,z['dev_rows']);p2,c2=decide(z,poisoned[z['dev_rows']],rr)
    assert r.safe(pick)==r.safe(p2) and r.safe(candidates)==r.safe(c2)
    for key in risks:
        np.testing.assert_allclose(risks[key][0],rr[key][0],atol=0,rtol=0);np.testing.assert_allclose(risks[key][1],rr[key][1],atol=0,rtol=0)
    part=sp['positions'][0];yp=y.copy();yp[part['B']]=9999
    first,au=risk_fit(d['x'],y,d['groups'],part['A'],part['B'],FEATURES['four']);second,au2=risk_fit(d['x'],yp,d['groups'],part['A'],part['B'],FEATURES['four'])
    np.testing.assert_array_equal(first,second);assert r.safe(au)==r.safe(au2)
    constant={k:np.zeros(3) for k in a.RULES}
    assert not strict_pick(constant,np.array([60.,60.,60.]),np.array([40.,60.,80.]))['feasible']
    assert not strict_pick(constant,np.array([40.,80.,80.]),np.array([40.,80.,80.]))['feasible']
    # Compare vectorized search with a brute-force audit on a small nontrivial example.
    pr=np.array([55.,60.,65.,75.,40.,68.]);tr=np.array([45.,60.,65.,75.,40.,73.]);risk=np.array([[.8,.2],[.1,.1],[.2,.2],[.1,.8],[.8,.1],[.1,.9]])
    sc={k:np.array([.1,.2,.3,.4,.5,.6]) for k in a.RULES};q=strict_pick(sc,pr,tr,risk);ms,_=profiles(pr,risk);best=-1
    for mask in ms:
        for t in sc['HGB']:
            met=extended_metrics(pr,tr,mask&(sc['HGB']<=t))
            if met['FNR']==0 and met['accuracy']>.8 and met['reported_as_normal']>0:best=max(best,met['reported'])
    assert q['development']['reported']==best
    r.write(O/'SMOKE.json',dict(test_label_poison_invariance=True,fold_label_poison_invariance=True,training_group_disjoint=True,no_normal_or_nonzero_FN_rejected=True,vectorized_search_matches_bruteforce=True))
    print('NORMAL CHECK SMOKE PASS',flush=True)

def run_one(d,item,budget):
    dest=O/f'dev{budget}/private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    start=time.time();sp=s.spec(item,budget);y=d['medical'][:,2];base=a.O/f'dev{budget}/private'/f"round_{item['rep']:02d}"
    z0=np.load(base/'HZ_to_SA_B.npz');risks,audits=risk_scores(d['x'],y,d['groups'],sp,z0['dev_rows'])
    r.write(dest/'RISK_TRAINING.json',audits);rows=[];saved=dict(dev_rows=z0['dev_rows'],test=z0['test'])
    for name,(dr,qr) in risks.items():saved['dev_risk_'+name]=dr;saved['test_risk_'+name]=qr
    for arm in a.ARMS:
        z=np.load(base/(arm+'_B.npz'));np.testing.assert_array_equal(z['dev_rows'],z0['dev_rows']);np.testing.assert_array_equal(z['test'],z0['test'])
        picks,candidates=decide(z,y[z['dev_rows']],risks)
        r.write(dest/(arm+'_SELECTION.json'),dict(policies=picks,normal_check_candidates=candidates))
        ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES}
        for policy,pick in picks.items():
            act=action_for(pick,z['prediction'],qs,{k:v[1] for k,v in risks.items()});devact=action_for(pick,z['dev_prediction'],ds,{k:v[0] for k,v in risks.items()})
            met=None if act is None else extended_metrics(z['prediction'],y[z['test']],act)
            if devact is not None:assert r.safe(extended_metrics(z['dev_prediction'],y[z['dev_rows']],devact))==r.safe(pick['development'])
            if policy!='existing' and pick['feasible']:assert pick['development']['FN']==0 and pick['development']['reported_as_normal']>0
            saved[arm+'_'+policy+'_action']=np.zeros(len(z['test']),bool) if act is None else act
            rows.append(dict(rep=item['rep'],budget=budget,arm=arm,policy=policy,feasible=pick['feasible'],development=pick.get('development'),development_joint_pass=bool(pick['feasible'] and pick['development']['success']),rule=pick.get('rule'),risk_model=pick.get('risk_model','none'),test=met,joint_test_pass=bool(met and met['success'])))
    np.savez_compressed(dest/'ACTIONS_RISKS.npz',**saved);r.write(dest/'RESULTS.json',rows);r.write(dest/'COMPLETE.json',dict(rep=item['rep'],budget=budget,seconds=time.time()-start))
    print('NORMAL CHECK',budget,item['rep'],round(time.time()-start,2),flush=True)

def aggregate():
    for budget in [54,60]:
        rows=[v for p in sorted((O/f'dev{budget}/private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/'RESULTS.json').read_text(encoding='utf8'))]
        r.write(O/f'dev{budget}/ROUNDS.json',rows);summary=[]
        for arm in a.ARMS:
            for policy in POLICIES:
                rr=[v for v in rows if v['arm']==arm and v['policy']==policy];valid=[v for v in rr if v['test'] is not None]
                q=dict(arm=arm,policy=policy,rounds=len(rr),feasible=len(valid),infeasible=len(rr)-len(valid),development_joint_pass=sum(v['development_joint_pass'] for v in rr),test_joint_pass=sum(v['joint_test_pass'] for v in rr),test_zero_FN=sum(v['test']['FNR']==0 and not v['test']['no_normal_reports'] for v in valid),test_no_normal=sum(v['test']['no_normal_reports'] for v in valid),test_FNR_undefined=sum(v['test']['FNR'] is None for v in valid),rules={k:sum(v['rule']==k for v in rr) for k in [None,*a.RULES]},risk_models={k:sum(v['risk_model']==k for v in rr) for k in [None,'none','B','four']})
                for key in ['accuracy','reporting_rate','FNR','FPR','MAE','R2','reported_as_normal','predicted_normal_retention','reported_as_low','reported_as_high','FN','FN_low','FN_high','FP','withheld','true_low_withheld','true_high_withheld','low_retained_correct_rate','high_retained_correct_rate']:
                    vals=[v['test'][key] for v in valid if v['test'][key] is not None]
                    q[key]=dict(mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,min=float(min(vals)),max=float(max(vals)),n=len(vals)) if vals else None
                summary.append(q)
        r.write(O/f'dev{budget}/SUMMARY.json',summary)
        flat=[]
        for v in rows:
            row={k:v[k] for k in ['rep','budget','arm','policy','feasible','development_joint_pass','rule','risk_model','joint_test_pass']}
            for side in ['development','test']:
                for k in ['FN','FNR','FPR','accuracy','reporting_rate','reported_as_normal','predicted_normal_retention','reported_as_low','reported_as_high','abnormal_released','true_low_withheld','true_high_withheld']:
                    row[side+'_'+k]=(v[side] or {}).get(k)
            flat.append(row)
        if flat:
            with (O/f'dev{budget}/ROUNDS.csv').open('w',encoding='utf-8-sig',newline='') as out:
                wr=csv.DictWriter(out,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
    print('NORMAL CHECK AGGREGATE',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    proto=protocol();path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==r.safe(proto)
    else:r.write(path,proto)
    if args.stage=='aggregate':aggregate();return
    d,*_=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    if args.stage=='smoke':smoke(d,items)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[args.start:args.stop]:run_one(d,item,args.budget)
    assert all(r.sha(r.current.ROOT/'data'/name)==value for name,value in proto['input_hashes'].items())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
