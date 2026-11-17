"""Dual-source loss controls with development-only normal safety envelopes."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import sys,json,argparse,time,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(_repeat_root / 'experiments' / 'uteg_locked_expansion_20261117'))
import experiment as e
import aggregate as ag
O=Path(__file__).parent
MODELS=['rbf_uniform','rbf_balanced','linear_uniform']
LINEAR_SOURCE={}

def solve(z,target,w):
    n=len(target);pen=np.r_[np.repeat(.01,z.shape[1]-1),0.]
    return np.linalg.solve(z.T@(w[:,None]*z)/n+np.diag(pen)+1e-12*np.eye(z.shape[1]),z.T@(w*target)/n)

def predict(d,arm,tr,qu,model):
    tr=np.asarray(tr,int);qu=np.asarray(qu,int);src=e.r.c.source_rows(arm,900)
    assert not set(d['groups'][np.r_[tr,src]])&set(d['groups'][qu])
    x=d['x'][:,1:2];y=d['medical'][:,2];kind='uniform' if model.endswith('uniform') else 'balanced'
    low=x[tr,0].min();high=x[tr,0].max();clip=x.copy();clip[qu,0]=np.clip(clip[qu,0],low,high)
    if model=='linear_uniform':
        key=arm
        if key not in LINEAR_SOURCE:
            mean=x[src].mean(0);scale=np.maximum(x[src].std(0),1.e-6);zs=np.c_[(x[src]-mean)/scale,np.ones(len(src))]
            w=np.zeros(len(src))
            for center in np.unique(d['centers'][src]):
                mask=d['centers'][src]==center;w[mask]=1/mask.sum()
            w/=w.mean();coef=solve(zs,(y[src]-60)/10,w);LINEAR_SOURCE[key]=(mean,scale,coef)
        mean,scale,coef=LINEAR_SOURCE[key]
        source=lambda xx,sm=mean.copy(),ss=scale.copy(),sc=coef.copy():np.maximum(60+10*np.c_[(xx-sm)/ss,np.ones(len(xx))]@sc,0.)
        mu=x[tr].mean(0);sd=np.maximum(x[tr].std(0),1.e-6)
        z=np.c_[(x[tr]-mu)/sd,np.ones(len(tr))];q=np.c_[(x[qu]-mu)/sd,np.ones(len(qu))];qc=np.c_[(clip[qu]-mu)/sd,np.ones(len(qu))]
    else:
        source_model=e.r.c.source_model(arm,900,'B',kind)
        source=lambda xx:e.r.c.source_predict(source_model,xx)
        shadow=dict(d,y=np.full_like(d['y'],np.nan));shadow['y'][tr]=d['y'][tr]
        z,q,*_=e.r.c.selected.models.design(shadow,[x],tr,qu,'target_only')
        zc,qc,*_=e.r.c.selected.models.design(shadow,[clip],tr,qu,'target_only')
        np.testing.assert_allclose(z,zc,rtol=0,atol=1e-12)
    w=e.r.c.selected.models.weights(y[tr],0,kind)
    coef=solve(z,(y[tr]-source(x[tr]))/10,w)
    pred=source(x[qu])+10*q@coef;endpoint=source(clip[qu])+10*qc@coef
    below=x[qu,0]<low;above=x[qu,0]>high
    pred[below]=np.minimum(pred[below],endpoint[below]);pred[above]=np.maximum(pred[above],endpoint[above])
    return np.maximum(pred,0.)

def collect(d,arm,tr,te,model,seed):
    # Match the original archived development folds where supplied by the caller.
    parts=seed if isinstance(seed,list) else e.r.folds(tr,d,seed)
    loc={int(v):j for j,v in enumerate(tr)};dp=np.full(len(tr),np.nan)
    for f in parts:
        aa=np.asarray(f['T']);bb=np.asarray(f['V']);assert set(aa)|set(bb)==set(tr)
        pp=predict(d,arm,aa,bb,model);dp[[loc[int(v)] for v in bb]]=pp
    assert np.isfinite(dp).all()
    return dp,predict(d,arm,tr,te,model)

def choose(dp,dy,dr,old_dp,envelope,abnormal_rule):
    cls=e.r.state(dy);low=cls==0 if envelope=='tail_max' and np.any(cls==0) else np.ones(len(dy),bool)
    high=cls==2 if envelope=='tail_max' and np.any(cls==2) else np.ones(len(dy),bool)
    low_error=max(0.,float(np.max(dp[low]-dy[low])));high_error=max(0.,float(np.max(dy[high]-dp[high])))
    old_normal=e.r.state(old_dp)==1
    cap=float(np.quantile(dr[old_normal,1],.9)) if old_normal.any() else -np.inf
    p=dict(envelope=envelope,normal_lower=50+low_error,normal_upper=70-high_error,cap=cap,abnormal_rule=abnormal_rule)
    for label,bound,key in [(0,50,'low_margin'),(2,70,'high_margin')]:
        mask=e.r.state(dp)==label;distance=abs(dp-bound)
        if abnormal_rule=='all':p[key]=0.;continue
        choices=[float('inf')]+sorted(set(distance[mask].tolist()))
        best=None
        for t in choices:
            active=mask&(distance>=t);n=int(active.sum())
            if not n or np.mean(cls[active]==label)<.85:continue
            if best is None or n>best[0]:best=(n,t)
        p[key]=best[1] if best else float('inf')
    return p

def action(p,pred,risk):
    cls=e.r.state(pred)
    normal=(cls==1)&(pred>p['normal_lower'])&(pred<p['normal_upper'])&(risk[:,1]<=p['cap'])
    return normal|((cls==0)&(50-pred>=p['low_margin']))|((cls==2)&(pred-70>=p['high_margin']))

def case(d,arm,path,folds,seed,check):
    with np.load(path) as z:old={k:z[k] for k in z.files}
    tr=old['dev_rows'];te=old['test'];dy=d['medical'][tr,2];ty=d['medical'][te,2];result={};policies={};mapping={}
    result['original']=(old['prediction'],old['action_common_B'])
    for model in MODELS:
        dp,pr=collect(d,arm,tr,te,model,folds or seed)
        mapping[model]=dict(development_prediction=dp,prediction=pr,dev_rows=tr,test=te)
        for envelope in ['all_max','tail_max']:
            for ab in ['all','accuracy85']:
                name=model+'_'+envelope+'_'+ab;p=choose(dp,dy,old['dev_risk'],old['dev_prediction'],envelope,ab)
                da=action(p,dp,old['dev_risk']);a=action(p,pr,old['test_risk'])
                dm=e.n.extended_metrics(dp,dy,da);assert dm['FN']==0
                if check:
                    poison=dy.copy();assert p==choose(dp,poison,old['dev_risk'],old['dev_prediction'],envelope,ab)
                    np.testing.assert_array_equal(a[::-1],action(p,pr[::-1],old['test_risk'][::-1]))
                p['development']=dm;policies[name]=p;result[name]=(pr,a)
    return old,result,policies,mapping

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    root=O/('smoke_v2' if args.smoke else 'full_v2');root.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope='Dual-source only: SA54/60,HZ36/40,HB36/40 and separate301internal outer5,100 repetitions per setting. Target-only/single-source existing fair controls remain unchanged.',
        models=MODELS,source='Same per-repeat two-source draws, equal total centre weight, source model then local residual correction. Source and local squared-error regularization.01 fixed. RBF basis and endpoint guard match the old architecture. Uniform and class-balanced weights fixed, no hyperparameter search.',
        gate='Development OOF directional maximum residual envelope for normal reports: all records or true low/high tails. Original raw10%risk cap and original risk model scores retained exactly. Abnormal all-report and developerclassaccuracy>=.85 boundary-margin controls. Full continuous MA and state derived from same prediction.',
        splits='Archived development folds for each case. Current source sampling and outer evaluation sets unchanged; internal301=271records per repeat.',
        constraints='Developer FN0 is checked. Evaluation FN0,accuracy>.80,reporting>.75 and nonempty normal reports jointly assessed, not forced. All settings/candidates/rounds retained. Repeatedly reviewed evaluation data => exploratory internal evidence.',no_manuscript_edits=True)
    pp=O/'PROTOCOL.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:e.r.write(pp,protocol)
    d,plan,role,models,items=e.init();e.GLOBAL_D=d
    for rep in (range(2) if args.smoke else range(args.start,args.stop)):
        dest=root/f'round_{rep:02d}'
        if (dest/'COMPLETE.json').exists():continue
        dest.mkdir(exist_ok=True);start=time.time();e.activate(items[rep],plan,dest);LINEAR_SOURCE.clear();rows=[];policies=[]
        cases=[]
        for budget in [54,60]:
            base=e.O/f'primary/dev{budget}/time900/round_{rep:02d}';sp=json.loads((base/'SPLIT.json').read_text())
            folds=[dict(T=x['A'],V=x['B']) for x in sp['positions'][:3]]
            cases.append(('SA',str(budget),'HZ+HB_to_SA',base/'HZ+HB_to_SA.npz',folds))
        for setting in ['sampled90','full_local']:
            for target,arm in [('Huzhou','SA+HB_to_HZ'),('Hubei','SA+HZ_to_HB')]:
                sp=e.specification(items[rep],'sampled' if setting=='sampled90' else 'full',role,target)
                folds=[dict(T=x['A'],V=x['B']) for x in sp['positions'][:3]]
                cases.append((target,setting,arm,e.O/f'nine/round_{rep:02d}/{setting}_{arm}.npz',folds))
        for stage,setting,arm,path,folds in cases:
            old,results,ps,maps=case(d,arm,path,folds,items[rep]['seed'],rep==0)
            for method,(pred,act) in results.items():
                rows.append(dict(stage=stage,setting=setting,rep=rep,arm=arm,method=method,test=e.n.extended_metrics(pred,d['medical'][old['test'],2],act),all_report=e.n.extended_metrics(pred,d['medical'][old['test'],2],np.ones(len(pred),bool))))
            policies.append(dict(stage=stage,setting=setting,arm=arm,policies=ps))
            np.savez_compressed(dest/f'{stage}_{setting}.npz',test=old['test'],**{k+'_'+suffix:v[i] for k,v in results.items() for i,suffix in enumerate(['prediction','action'])})
            for model,z in maps.items():np.savez_compressed(dest/f'{stage}_{setting}_{model}_mapping.npz',**z)
        arm='HZ+HB_to_SA';base=e.O/f'internal301/round_{rep:02d}';splits=json.loads((base/'SPLITS.json').read_text());alltest=[];parts={}
        for fi,sp in enumerate(splits):
            folds=[dict(T=x['A'],V=x['B']) for x in sp['positions'][:3]]
            old,results,ps,maps=case(d,arm,base/f'fold{fi}_{arm}.npz',folds,items[rep]['seed']+fi,rep==0)
            alltest.extend(old['test']);policies.append(dict(stage='internal301',setting='271',arm=arm,fold=fi,policies=ps))
            for model,z in maps.items():np.savez_compressed(dest/f'internal_fold{fi}_{model}_mapping.npz',**z)
            for method,(pred,act) in results.items():
                if method not in parts:parts[method]=[[],[]]
                parts[method][0].extend(pred);parts[method][1].extend(act)
        te=np.array(alltest);assert len(set(te))==271
        for method,(pr,ac) in parts.items():
            pr=np.array(pr);ac=np.array(ac);rows.append(dict(stage='internal301',setting='271',rep=rep,arm=arm,method=method,test=e.n.extended_metrics(pr,d['medical'][te,2],ac),all_report=e.n.extended_metrics(pr,d['medical'][te,2],np.ones(len(te),bool))))
        np.savez_compressed(dest/'internal301.npz',test=te,**{k+'_'+suffix:np.array(v[i]) for k,v in parts.items() for i,suffix in enumerate(['prediction','action'])})
        e.r.write(dest/'RESULTS.json',rows);e.r.write(dest/'POLICIES.json',policies);e.r.write(dest/'COMPLETE.json',dict(seconds=time.time()-start))
        print('ROUND',rep+1,round(time.time()-start,1),flush=True)
    rows=[x for p in sorted(root.glob('round_*/COMPLETE.json')) for x in json.loads((p.parent/'RESULTS.json').read_text())];summ=ag.summarize(rows,['stage','setting','arm','method'])
    for s in summ:
        rr=[x for x in rows if all(x[k]==s[k] for k in ['stage','setting','arm','method'])]
        s['positive_FN_rounds']=sum(x['test']['FN']>0 for x in rr);s['FN_total']=sum(x['test']['FN'] for x in rr)
        s['joint_pass_rounds']=sum(x['test']['FN']==0 and (x['test']['accuracy'] or 0)>.8 and x['test']['reporting_rate']>.75 and x['test']['reported_as_normal']>0 and x['test']['abnormal_released']>0 for x in rr)
        s['no_normal_report_rounds']=sum(x['test']['reported_as_normal']==0 for x in rr)
    e.r.write(root/'ROUNDS.json',rows);ag.csvwrite(root/'ROUNDS.csv',rows);e.r.write(root/'SUMMARY.json',summ);ag.csvwrite(root/'SUMMARY.csv',summ)
    assert all(e.r.sha(e.r.current.ROOT/'data'/k)==v for k,v in e.cal.protocol()['input_hashes'].items())
    print('DONE',len(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
