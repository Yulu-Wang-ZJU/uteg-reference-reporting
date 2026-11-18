"""Reference244, fixed-threshold ablations and complete matched summaries."""
import complete as m
import numpy as np
import json,time,argparse,platform,importlib.util
from pathlib import Path
from threadpoolctl import threadpool_limits
r=m.r;e=m.e;c=m.c;O=m.O;ag=m.ag
def archive(b,h,rep,arm):return m.load(O/f'primary/dev{b}/time{h}/round_{rep:02d}/{arm}.npz')
def reference(d):
    dest=O/'reference244';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    tr=np.asarray(m.INDEX[0,60]['targets']['SAHZU']['pool']);te=np.setdiff1d(np.flatnonzero(d['centers']=='SAHZU'),tr)
    assert len(tr)==60 and len(te)==244;m.q.separate(d,tr,te)
    seed=1000;sp=dict(pool=tr,test=te,positions=m.q.positions(d,tr,te,seed));item=dict(rep=0,seed=seed,sources={ct:np.flatnonzero(d['centers']==ct).tolist() for ct in ['SAHZU','Huzhou','Hubei']},targets={'SAHZU':sp})
    m.q.f.activate(item,d);models=r.suite.numerical_models();rows=[];policies={};four=[]
    for arm in m.q.ARMS:
        z,au=m.bundle(d,item,arm,900);rr,pol,acts,cc=m.finish_bundle(d,z,900);pp,raw=m.ad.primary_four(d,models,arm,sp,z['prediction']);four.append(pp)
        rows.extend(dict(x,arm=arm) for x in rr);policies[arm]=pol
        np.savez_compressed(dest/(arm+'.npz'),**z,prediction_four=pp,truth_four=d['medical'][te],source_prediction_four=raw,**{'action_'+k.replace(' ','_'):v for k,v in acts.items()})
    # Prespecified range-spanning selection; no predicted errors used.
    y=d['medical'][te,2];targets=np.linspace(y.min(),y.max(),10);selected=[]
    for value in targets:
        idx=next(int(i) for i in np.lexsort((d['ids'][te].astype(str),abs(y-value))) if int(i) not in selected);selected.append(idx)
    selected=np.array(selected);chosen=te[selected];v=np.asarray(four)[:,selected,:];R,K,A=v[...,0],v[...,1],v[...,2]
    assert np.all(K>0) and np.all(A>20)
    beta=K/np.log((A-2)/(A-20));tau=R+beta*np.log1p(-2/A);t=np.arange(0,2701,5)/60
    assert np.all(tau>=0) and np.all(beta>0)
    curves=A[...,None]*(-np.expm1(-np.maximum(t-tau[...,None],0)/beta[...,None]))
    np.testing.assert_allclose(A*(-np.expm1(-(R-tau)/beta)),2,atol=1e-10)
    np.testing.assert_allclose(A*(-np.expm1(-(R+K-tau)/beta)),20,atol=1e-10)
    tegpath=r.current.ROOT/'figure2_mapping_20261103/private/PAIRED_SIGNALS.npz';teg=m.load(tegpath)
    np.testing.assert_array_equal(teg['ids'],d['ids'][:304])
    with np.load(r.current.ROOT/'data/CURRENT_SCALED_CURVES.npz') as z:
        ux=z['scaled_zeroed'][chosen].copy();um=z['observed_mask'][chosen].copy()
    np.savez_compressed(dest/'TEN_EXAMPLES.npz',rows=chosen,ids=d['ids'][chosen],time_minutes=t,predicted_curves=curves,predicted_readouts=v,reference_readouts=d['medical'][chosen],reference_teg=teg['teg'][chosen],reference_time=teg['teg_time'][chosen],reference_mask=teg['teg_mask'][chosen],uteg=ux,uteg_mask=um,arms=np.array(m.q.ARMS))
    B=d['x'][te,1];coef=np.polyfit(B,y,1);line=np.polyval(coef,B)
    r.write(dest/'B_MA_ASSOCIATION.json',dict(n=244,slope=coef[0],intercept=coef[1],R2=1-np.sum((y-line)**2)/np.sum((y-y.mean())**2),descriptive_not_predictive=True))
    r.write(dest/'EXAMPLE_SELECTION.json',dict(rule='Nearest distinct records to ten equally spaced reference-MA targets, ties byrecordID; no prediction-error selection.',rows=chosen,ids=d['ids'][chosen],targets=targets,reference_MA=y[selected],seed=1000,
        reconstruction='Existing delayed exponential from predictedR,K,MA; not a separately trained trajectory predictor. Stored measured traces remain unchanged; time grid0–45minutes applies to reconstruction.',paired_teg_source_sha256=r.sha(tegpath)))
    r.write(dest/'RESULTS.json',rows);r.write(dest/'POLICIES.json',policies);r.write(dest/'SPLIT.json',sp);r.write(dest/'COMPLETE.json',dict(seed=seed,source='All original source records',development=60,test=244,all_four_readouts=True))
    ag.write('Fig2_reference244',rows,['arm','rule']);print('REFERENCE244 COMPLETE',flush=True)
def ablations(d):
    rows=[];checks={'full':(),'without_low_risk':('low_risk',),'without_high_risk':('high_risk',),'without_B_bounds':('B_bounds',),'without_normal_score':('normal_score',),'without_both_risks':('low_risk','high_risk'),'without_normal_checks':('low_risk','high_risk','B_bounds','normal_score')}
    for b in [60,54]:
        for rep in range(100):
            pol=m.read(O/f'primary/dev{b}/time900/round_{rep:02d}/POLICIES.json')
            for arm in m.q.ARMS:
                z=archive(b,900,rep,arm);pick=pol[arm]['common_B'];saved={}
                for name,drop in checks.items():
                    mask=m.apply_policy(d,z,900,pick,drop=drop)
                    if not drop:np.testing.assert_array_equal(mask,z['action_common_B'])
                    assert np.all(mask|~z['action_common_B'])
                    rows.append(dict(budget=b,rep=rep,arm=arm,ablation=name,test=e.n.extended_metrics(z['prediction'],d['medical'][z['test'],2],mask)))
                    saved[name]=mask
                dest=O/f'ablation/dev{b}/round_{rep:02d}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(arm+'.npz'),test=z['test'],**saved)
    ag.write('gate_ablation',rows,['budget','arm','ablation']);print('ABLATION COMPLETE',len(rows),flush=True)
def capability(d):
    rows=[];pooled={};curves=[]
    for b in [60,54]:
        for rep in range(100):
            for arm in m.q.ARMS:
                z={h:archive(b,h,rep,arm) for h in [300,600,900]};ids=z[300]['test'];truth=r.state(d['medical'][ids,2]);n=len(ids)
                assert n==220
                for h in [300,600,900]:
                    np.testing.assert_array_equal(ids,z[h]['test']);p=z[h]['prediction'].copy();score=z[h]['score_HGB'].copy()
                    if h==600:
                        use=z[300]['action_HGB']&~z[600]['action_HGB'];p[use]=z[300]['prediction'][use]
                        score=np.where(use,m.cap.calibrated(z[300],z[300]['score_HGB'],d),m.cap.calibrated(z[600],z[600]['score_HGB'],d))
                    order=np.lexsort((d['ids'][ids].astype(str),score));pc=r.state(p)[order];tc=truth[order]
                    counts=np.cumsum(np.c_[np.ones(n),pc==tc,(pc==1)&(tc!=1),tc!=1,(pc!=1)&(tc==1),tc==1],axis=0)
                    key=b,arm,h;pooled[key]=pooled.get(key,np.zeros_like(counts))+counts
                    for rate in [.1,.2,.4]:rows.append(dict(budget=b,rep=rep,arm=arm,time=h,condition='fixed_reporting_rate',level=rate,status='integer_count',test=m.cap.metric(counts[int(np.floor(rate*n))-1],n)))
                    for acc in [.85,.90,.95]:
                        v,status=m.cap.matched(counts,acc);rows.append(dict(budget=b,rep=rep,arm=arm,time=h,condition='matched_observed_accuracy',level=acc,status=status,test=m.cap.metric(v,n)))
    ag.write('Fig4_HGB_capability',rows,['budget','arm','time','condition','level'])
    result=[]
    for (b,arm,h),counts in pooled.items():
        for rate in [.1,.2,.4]:result.append(dict(budget=b,arm=arm,time=h,condition='fixed_reporting_rate',level=rate,test=m.cap.metric(counts[int(np.floor(rate*220))-1],22000)))
        for acc in [.85,.90,.95]:
            v,status=m.cap.matched(counts,acc);result.append(dict(budget=b,arm=arm,time=h,condition='matched_observed_accuracy',level=acc,status=status,test=m.cap.metric(v,22000)))
        curves.extend(dict(budget=b,arm=arm,time=h,rank=i+1,**m.cap.metric(v,22000)) for i,v in enumerate(counts))
    r.write(ag.S/'Fig4_HGB_capability_pooled.json',result);ag.csvwrite(ag.S/'Fig4_HGB_rank_curves_pooled.csv',curves)
    r.write(O/'CAPABILITY_PROTOCOL.json',dict(same_original_HGB_logic=True,fixed_reporting_rates=[.1,.2,.4],counts=[22,44,88],fixed_accuracy=[.85,.9,.95],matching='Observed-accuracy retrospective interpolation, not a deployment rule; retain missing matches.',no_forced_zero_FN=True,no_monotonicity_enforcement=True))
    print('CAPABILITY COMPLETE',len(rows),flush=True)
def sequences(d):
    rows=[]
    for b in [60,54]:
        for rep in range(100):
            for arm in m.q.ARMS:
                z={h:archive(b,h,rep,arm) for h in [300,600,900]};truth=d['medical'][z[300]['test'],2]
                for rule in ['common_B','HGB','Logistic','Boundary','B95']:
                    for sched in [[300,600,900],[600,900],[900]]:
                        for mode in ['first_report_locked','update_then_15min_current']:
                            pred=np.zeros(len(truth));act=np.zeros(len(truth),bool)
                            for h in sched:
                                now=z[h]['action_'+rule];fresh=now&~act if mode=='first_report_locked' else now
                                if mode=='update_then_15min_current' and h==900:act=now.copy();pred=z[h]['prediction'].copy()
                                else:pred[fresh]=z[h]['prediction'][fresh];act|=now
                                rows.append(dict(budget=b,rep=rep,arm=arm,rule=rule,schedule='-'.join(str(t//60) for t in sched),mode=mode,time=h,test=e.n.extended_metrics(pred,truth,act)))
    ag.write('time_schedules',rows,['budget','arm','rule','schedule','mode','time']);print('SEQUENCES COMPLETE',len(rows),flush=True)
def summarize(d):
    table=[];primary=[];curves=[];linear=[];budgets=[];mapping=[];controls=[]
    for rep in range(100):
        table+=m.read(O/f'table1/round_{rep:02d}/RESULTS.json');budgets+=m.read(O/f'budgets/round_{rep:02d}/RESULTS.json')
        for b in [60,54]:
            for arm in m.q.ARMS[1:]:
                for method in r.mapping.METHODS:
                    z=m.load(O/f'table1/round_{rep:02d}/{arm}_{method}_dev{b}.npz');p=z['prediction'];truth=d['medical'][z['test'],2]
                    assert np.isfinite(p).all();controls.append(dict(budget=b,rep=rep,arm=arm,panel='mapping',method=method,test=e.n.extended_metrics(p[:,2],truth,np.ones(len(truth),bool))))
                z=m.load(O/f'table1/round_{rep:02d}/{arm}_calibration_dev{b}.npz');truth=d['medical'][z['test'],2]
                for key,p in z.items():
                    if key=='test':continue
                    assert np.isfinite(p).all();controls.append(dict(budget=b,rep=rep,arm=arm,panel='calibration',method=key,test=e.n.extended_metrics(p[:,2],truth,np.ones(len(truth),bool))))
        for b in [60,54]:
            for h in [300,600,900]:
                dest=O/f'primary/dev{b}/time{h}/round_{rep:02d}';assert (dest/'COMPLETE.json').exists();primary+=m.read(dest/'RESULTS.json');curves+=m.read(dest/'CURVES.json')
                for arm in m.q.ARMS:
                    z=archive(b,h,rep,arm);y=d['medical'][z['test'],2];pc=r.state(z['prediction']);tc=r.state(y);a=np.ones(len(y),bool)
                    met=e.n.extended_metrics(z['prediction'],y,a)
                    for cls,name in [(0,'low'),(1,'normal'),(2,'high')]:
                        sel=tc==cls;met[name+'_MAE']=float(np.mean(abs(z['prediction'][sel]-y[sel]))) if sel.any() else None;met[name+'_recall']=float(np.mean(pc[sel]==cls)) if sel.any() else None
                    mapping.append(dict(budget=b,rep=rep,arm=arm,time=h,test=met))
            linear+=m.read(O/f'linear/dev{b}/time900/round_{rep:02d}/RESULTS.json')
    assert len(table)==7200 and len(primary)==14400 and len(linear)==4800
    ag.write('Table1',table,['budget','arm','panel','method'])
    ag.write('Table1_MA_state_controls',controls,['budget','arm','panel','method'])
    ag.write('Table2', [x for x in primary if x['time']==900],['budget','arm','rule'])
    ag.write('early_reporting',primary,['budget','arm','time','rule'])
    ag.write('all_report_mapping',mapping,['budget','arm','time'])
    ag.write('Fig3_rank_curves',curves,['budget','arm','time','rule','nominal_rate'])
    ag.write('linear_control',linear,['budget','arm','rule'])
    ag.write('label_budgets',budgets,['budget','arm'])
    # Reassemble nine directions without recomputing the other target centres.
    nine=[]
    for rep in range(100):
        for x in m.read(m.OLD/f'nine/round_{rep:02d}/RESULTS.json'):
            if x['arm'].startswith('target_only') or x['arm'].endswith('_to_SA'):continue
            nine.append(dict(x,origin='unchanged_other_target',policy_version='previous_common_B'))
        for x in primary:
            if x['rep']==rep and x['time']==900 and x['arm'] in m.q.ARMS[1:]:
                nine.append(dict(x,mode='full_local' if x['budget']==60 else 'sampled90',development_n=x['budget'],origin='new_SAHZU_target',policy_version='symmetric90'))
    ag.write('nine_reporting',nine,['mode','arm','rule','origin','policy_version'])
    print('SUMMARIES COMPLETE',flush=True)
def audit(d):
    checked=0;frozen=0
    for b in [60,54]:
        for rep in range(100):
            for h in [300,600,900]:
                folder=O/f'primary/dev{b}/time{h}/round_{rep:02d}';rows=m.read(folder/'RESULTS.json');sp=m.INDEX[rep,b]['targets']['SAHZU']
                for arm in m.q.ARMS:
                    z=archive(b,h,rep,arm);np.testing.assert_array_equal(z['test'],sp['test']);assert set(z['dev_rows'])==set(sp['pool']);q=m.q;q.separate(d,z['dev_rows'],z['test'])
                    if h==900:
                        orig=m.load(m.P/f'private/dev{b}/round_{rep:03d}/{arm}.npz');lock=m.load(m.L/f'private/dev{b}/round_{rep:03d}.npz')
                        np.testing.assert_array_equal(orig['prediction'],z['prediction']);np.testing.assert_array_equal(lock[arm+'_action'],z['action_common_B']);frozen+=1
                    for row in [x for x in rows if x['arm']==arm]:
                        act=z['action_'+row['rule'].replace(' ','_')];actual=e.n.extended_metrics(z['prediction'],d['medical'][z['test'],2],act)
                        assert r.safe(actual)==r.safe(row['test']);checked+=1
    for p,sha in m.read(O/'PROTECTED.json').items():assert r.sha(Path(p))==sha,p
    # Source controls were reused only for exactly unchanged source draws.
    caches=[]
    for p in sorted((O/'table1').glob('round_*/*_source.joblib')):
        src=m.OLD/('nine_controls' if p.name.startswith('HB_to_SA') else 'table1')/p.parent.name/p.name
        assert r.sha(src)==r.sha(p);caches.append(dict(file=str(p.relative_to(O)),source=str(src),sha256=r.sha(p)))
    assert len(caches)==1800
    r.write(O/'SOURCE_CACHE_MANIFEST.json',caches)
    r.write(O/'FINAL_AUDIT.json',dict(passed=True,all_100_rounds_retained=True,primary_metric_rows_recomputed=checked,locked15min_predictions_actions_identical=frozen,source_model_caches_verified=len(caches),protected_files_unchanged=True,other_target_centres_not_recomputed=True))
    print('AUDIT PASS',checked,frozen,flush=True)
def runtime(d):
    rows=[]
    for b in [60,54]:
        item=m.INDEX[0,b]
        for arm in m.q.ARMS:
            for h in [300,600,900]:
                for repeat in range(5):
                    m.q.f.activate(item,d);t=time.perf_counter();z,_=m.bundle(d,item,arm,h);s=time.perf_counter();out=m.finish_bundle(d,z,h);u=time.perf_counter()
                    loops=100;start=time.perf_counter()
                    for _ in range(loops):m.apply_policy(d,z,h,out[1]['common_B'])
                    rows.append(dict(budget=b,arm=arm,time=h,repeat=repeat,fit_and_development_scores_s=s-t,rule_selection_s=u-s,gate_application_batch_s=(time.perf_counter()-start)/loops,batch_n=len(z['test'])))
    ag.write('runtime',rows,['budget','arm','time']);r.write(O/'RUNTIME_PROTOCOL.json',dict(platform=platform.platform(),processor=platform.processor(),threads=1,repeat_count=5,batch=220,scope='Compute-only: source/local fits plus nested gate development; selection; mask application to already computed predictions/scores. Not signal acquisition time or a clinical turnaround benchmark.'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reference','all','runtime']);args=ap.parse_args();d=m.setup()
    if args.stage=='reference':reference(d);return
    if args.stage=='runtime':runtime(d);return
    summarize(d);ablations(d);capability(d);sequences(d)
    import reference_and_associations as ref
    ref.O=O;ref.associations(d);audit(d)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
