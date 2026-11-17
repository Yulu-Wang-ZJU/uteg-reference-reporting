"""Developer-only risk scale controls, original fixed10 cap remains an outer veto."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import sys,json,copy,csv,argparse
from pathlib import Path
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_fixed10_remaining_20261116'))
import run_followup as f
import support_followup as support
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
r=f.r;b=f.b;n=f.n;a=f.a;c=f.c;g=f.g
O=Path(__file__).parent
MODES=['common_B','train_cdf','prior_probability']

def transform(model,x,ids,raw,mode):
    ids=np.asarray(ids,int);train=np.asarray(model['train'],int)
    if mode=='common_B':return np.column_stack([-x[ids,1],x[ids,1]])
    if mode=='train_cdf':
        ref=np.sort(x[train,1]);vals=x[ids,1]
        rank=(np.searchsorted(ref,vals,'left')+np.searchsorted(ref,vals,'right'))/(2*len(ref))
        return np.column_stack([1-rank,rank])
    out=raw.copy()
    for j,head in enumerate(model['heads']):
        count=head['positive_count'];total=len(train)
        if head['status']!='fit':out[:,j]=1;continue
        # Balanced class weighting changes the fitted prior to 1/2.
        out[:,j]=expit(logit(np.clip(raw[:,j],1e-12,1-1e-12))+np.log(count/(total-count)))
    return out

def make_scores(x,dev_rows,test,rawdev,rawtest,audit,mode):
    dev=np.empty_like(rawdev);loc={int(v):i for i,v in enumerate(dev_rows)}
    for model in audit[:3]:
        ids=np.asarray(model['query'],int);jj=np.array([loc[int(v)] for v in ids])
        dev[jj]=transform(model,x,ids,rawdev[jj],mode)
    query=transform(audit[3],x,test,rawtest,mode)
    return dev,query

def choose_normal(pred,truth,raw,risk,scores,cap):
    pc=r.state(pred);tc=r.state(truth);normal=pc==1
    masks,limits=n.profiles(pred,risk)
    masks &= normal[None,:]&(raw[:,1]<=cap)[None,:]
    best=None
    for gate in b.gate_options(scores,normal):
        active=masks&b.gate_action(gate,scores,len(pred))[None,:]
        size=active.sum(1);valid=(size>0)&~np.any(active&(tc!=1)[None,:],axis=1)
        if not valid.any():continue
        ii=np.flatnonzero(valid);i=int(ii[np.argmax(size[ii])]);count=int(size[i])
        if best is None or count>best['reported']:
            best=dict(feasible=True,gate=gate,limits=limits[i],reported=count,cap_score=cap)
    if best is None:return dict(feasible=False,cap_score=cap)
    # Empty developer score gaps: preserve all developer normal decisions.
    before=normal_action(best,pred,raw,risk,scores)
    for j,key in enumerate(['low','high']):best['limits'][key],_=g.edge(risk[normal,j],best['limits'][key])
    if best['gate']['rule']!='All':
        rule=best['gate']['rule'];best['gate']['threshold'],_=g.edge(scores[rule][normal],best['gate']['threshold'])
    np.testing.assert_array_equal(before,normal_action(best,pred,raw,risk,scores))
    return best

def normal_action(p,pred,raw,risk,scores):
    if not p['feasible']:return np.zeros(len(pred),bool)
    lim=p['limits']
    out=(r.state(pred)==1)&(raw[:,1]<=p['cap_score'])&(risk[:,0]<=lim['low'])&(risk[:,1]<=lim['high'])&b.gate_action(p['gate'],scores,len(pred))
    return out

def choose(dp,dy,raw,risk,scores,cap):
    normal=choose_normal(dp,dy,raw,risk,scores,cap)
    nm=normal_action(normal,dp,raw,risk,scores)
    ab=c.choose(dp,dy,scores,nm,0)
    return dict(feasible=normal['feasible'] and ab['feasible'],normal=normal,abnormal=ab)

def action(p,pred,raw,risk,scores):
    if not p['feasible']:return None
    out=normal_action(p['normal'],pred,raw,risk,scores);pc=r.state(pred)
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:out|=(pc==cls)&b.gate_action(p['abnormal'][key],scores,len(pred))
    return out

def diag(x,dp,dr,qr,dev_rows,test,audit):
    grid=np.quantile(x[dev_rows,1],[.1,.25,.5,.75,.9]);curves=[]
    full=audit[-1];reconstructed=np.zeros_like(dr);fold_id=np.zeros(len(dev_rows),int);lookup={int(v):i for i,v in enumerate(dev_rows)}
    for pos,model in enumerate(audit):
        vv=[]
        for j,h in enumerate(model['heads']):
            if h['status']!='fit':vv.append(np.ones(len(grid)).tolist());continue
            coef=float(h['coef'][0][0]);inter=float(h['intercept'][0]);mean=model['mean'][0];scale=model['scale'][0]
            vv.append(expit((grid-mean)/scale*coef+inter).tolist())
            if pos<3:
                jj=[lookup[int(v)] for v in model['query']];reconstructed[jj,j]=expit((x[model['query'],1]-mean)/scale*coef+inter);fold_id[jj]=pos
        curves.append(vv)
    np.testing.assert_allclose(reconstructed,dr,atol=1e-12,rtol=1e-10)
    normal=r.state(dp)==1;inversions=[]
    for j in [0,1]:
        raw=(-x[dev_rows,1] if j==0 else x[dev_rows,1])[normal];scores=dr[normal,j]
        delta=raw[:,None]-raw[None,:];sd=scores[:,None]-scores[None,:];pairs=np.triu(np.ones(delta.shape,bool),1)&(delta!=0)
        inversions.append(dict(count=int(np.sum(pairs&(delta*sd<0))),pairs=int(pairs.sum())))
    return dict(grid_B=grid,head_scores_by_model=curves,training_sizes=[len(v['train']) for v in audit],developer_rank_inversions=inversions,
        max_same_B_high_score_range=float(np.max(np.ptp(np.asarray(curves)[:,1,:],axis=0))),
        score_quantiles_dev=np.quantile(dr,[.1,.5,.9],axis=0),score_quantiles_test=np.quantile(qr,[.1,.5,.9],axis=0),
        note='Same-B model comparison isolates scale variation; marginal dev/test score shifts also reflect case mix and do not by themselves establish miscalibration.')

def protocol():
    return dict(budgets=[54,60],repeats=100,arms=a.ARMS,modes=MODES,abnormal_endpoints=['midpoint','upper'],
        fixed='Exact old raw B high-risk90th-percentile cap and its per-record veto retained; original MA predictions, source splits and clinical class boundaries unchanged. Calibrations replace additional low/high risk bounds only, not the10% safety veto.',
        common_B='Single-feature fitted logistic scores are monotone functions of B with fold-specific intercept/slope. Use shared physical scores -B for low risk and B for high risk. No learned population scale, labels or query distribution enters this coordinate change. Check coefficient directions.',
        train_cdf='Convert B to within-training-sample mid-CDF (low=1-CDF,high=CDF), each development record uses its fold training reference; test uses full development reference.',
        prior_probability='Undo balanced-class prior using sigmoid(logit(raw probability)+log(n_positive/n_negative)); counts from the corresponding risk-training fold only. It is a prior correction, not a guarantee of calibrated probabilities.',
        selector='Existing low/high rectangle plus All/HGB/Logistic/Boundary normal gate; developer FN0, maximize normal reports; same developer>80% classwise abnormal selector. Two fixed abnormal endpoint controls, all100 each. No test-based choice.',
        limitations='Repeatedly explored evaluation data; exploratory. Test zeroFN assessed, never coerced. No manuscript edits.',input_hashes=b.protocol()['input_hashes'])

def run(smoke=False):
    p=O/'PROTOCOL.json';proto=protocol()
    if p.exists():assert json.loads(p.read_text())==r.safe(proto)
    else:r.write(p,proto)
    d,*_=r.init();all_diag=[];check_count=0;source_hashes={}
    for budget in [54,60]:
        rows=[]
        for rep in ([9,27,85] if smoke else range(100)):
            z0,rr0,_=b.load(budget,rep,a.ARMS[0]);auditpath=n.O/f'dev{budget}/private/round_{rep:02d}/RISK_TRAINING.json';audits=json.loads(auditpath.read_text())['B']
            for au in audits:
                assert not set(d['groups'][au['train']])&set(d['groups'][au['query']])
                assert not set(au['train'])&set(z0['test'])
                for j,h in enumerate(au['heads']):
                    assert h['status']=='fit' and ((h['coef'][0][0]<0) if j==0 else (h['coef'][0][0]>0))
            transformed={mode:make_scores(d['x'],z0['dev_rows'],z0['test'],rr0['dev_risk_B'],rr0['test_risk_B'],audits,mode) for mode in MODES}
            saved={'dev_rows':z0['dev_rows'],'test':z0['test']};policies={}
            for mode,(dr,qr) in transformed.items():saved['dev_'+mode]=dr;saved['test_'+mode]=qr
            for arm in a.ARMS:
                z,rr,base=b.load(budget,rep,arm);ds={k:z['dev_score_'+k] for k in a.RULES};qs={k:z['score_'+k] for k in a.RULES};dy=d['medical'][z['dev_rows'],2];ty=d['medical'][z['test'],2]
                assert base['cap_score']==float(np.quantile(rr['dev_risk_B'][r.state(z['dev_prediction'])==1,1],.9))
                all_diag.append(dict(budget=budget,rep=rep,arm=arm,**diag(d['x'],z['dev_prediction'],rr['dev_risk_B'],rr['test_risk_B'],z['dev_rows'],z['test'],audits)))
                for mode,(dr,qr) in transformed.items():
                    pick=choose(z['dev_prediction'],dy,rr['dev_risk_B'],dr,ds,base['cap_score'])
                    for upper in [False,True]:
                        name=mode+('_upper' if upper else '')
                        p=f.gap_policy(pick,z['dev_prediction'],ds,'upper') if upper else copy.deepcopy(pick)
                        da=action(p,z['dev_prediction'],rr['dev_risk_B'],dr,ds);act=action(p,z['prediction'],rr['test_risk_B'],qr,qs)
                        dev=None if da is None else n.extended_metrics(z['dev_prediction'],dy,da);test=None if act is None else n.extended_metrics(z['prediction'],ty,act)
                        if act is not None:
                            assert dev['FN']==0 and dev['accuracy']>.8 and dev['reported_as_normal']>0
                            assert not np.any(act&(r.state(z['prediction'])==1)&(rr['test_risk_B'][:,1]>base['cap_score']))
                        rows.append(dict(budget=budget,rep=rep,arm=arm,policy=name,feasible=p['feasible'],development=dev,development_joint_pass=bool(dev and dev['success']),test=test,joint_test_pass=bool(test and test['success']),risk_model='B',cap_quantile=.9))
                        saved[arm+'_'+name]=np.zeros(len(z['test']),bool) if act is None else act;policies[arm+'_'+name]=p;check_count+=1
                    if smoke:
                        # Reordering/changing query features cannot fit the developer transformation.
                        xx=d['x'].copy();xx[z0['test'],1]+=20
                        dd,_=make_scores(xx,z0['dev_rows'],z0['test'],rr0['dev_risk_B'],rr0['test_risk_B'],audits,mode)
                        np.testing.assert_array_equal(dr,dd)
                        assert r.safe(pick)==r.safe(choose(z['dev_prediction'],dy,rr['dev_risk_B'],dd,ds,base['cap_score']))
                for path in [a.O/f'dev{budget}/private/round_{rep:02d}/{arm}_B.npz',auditpath,n.O/f'dev{budget}/private/round_{rep:02d}/ACTIONS_RISKS.npz']:source_hashes[str(path)]=r.sha(path)
            if not smoke:
                folder=O/f'dev{budget}/private/round_{rep:02d}';r.write(folder/'POLICIES.json',policies);np.savez_compressed(folder/'ACTIONS_SCORES.npz',**saved)
            if rep%10==0:print('CALIBRATION',budget,rep+1,flush=True)
        if not smoke:
            r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',b.h.summarize(rows))
    assert all(r.sha(r.current.ROOT/'data'/k)==v for k,v in proto['input_hashes'].items())
    assert all(r.sha(Path(k))==v for k,v in source_hashes.items())
    r.write(O/('SMOKE.json' if smoke else 'CHECKS.json'),dict(rows=check_count,input_and_source_hashes_unchanged=True,training_groups_disjoint=True,original10_cap_preserved=True,developer_FN_zero=True))
    if not smoke:r.write(O/'DIAGNOSTIC.json',all_diag);r.write(O/'SOURCE_HASHES.json',source_hashes)
    print('DONE',check_count,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.smoke)
