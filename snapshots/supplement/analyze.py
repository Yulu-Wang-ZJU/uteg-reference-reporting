"""Appendix prediction export, then reference join and paired statistics. No fits."""
import sys,time,csv,ast,platform
from typing import Any,Mapping
import numpy as np,joblib
import batch as b
P=b.ROOT;O=b.OUT
EXT=O/'uteg_ridge_dual_head_external_eval_20261026_191117'
KRR=O/'uteg_signal_model_goal_20261026_000929/krr_fullchain_20261026_102726'
def cls(x):return np.where(np.isfinite(x),np.where(x<50,-1,np.where(x>70,1,0)),9).astype(np.int8)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(name,rows):
    path=P/'tables'/name;path.parent.mkdir(exist_ok=True)
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(name,**kw):
    p=P/'predictions'/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**kw)
def class_actions(pc,bs,ds,valid,th):
    a=np.zeros_like(pc,dtype=bool)
    if not th['all_withhold']:
        if th['normal'] is not None:a|=valid&(pc==0)&np.isfinite(ds)&(ds<=th['normal'])
        if th['tail'] is not None:a|=valid&(pc!=0)&np.isfinite(bs)&(bs<=th['tail'])
    return a
def infer(t,obj,x):
    src=obj['source_model'];raw=t.m.predict(src,x);ca=obj['cache'][obj['positions'][3]['base']]
    pure=t.m.calibrated(ca['cal'],raw);ad=obj['final_selection']['adapter']
    final=pure if ad=='H' else t.m.ans['adapt'](ca['adapters'][ad],x,raw,pure,src)
    return raw,pure,final
def freeze_margin():
    # Pure historical selector; trained/development labels only, before test score export.
    ns={'np':np,'Any':Any,'Mapping':Mapping,'MA_CLASSES':(-1,0,1)}
    path=O/'uteg_nine_direction_ma_20261025_195548/code/fit_ma_gates.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['summary','select_threshold']]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    D=b.data_at(900);out={}
    for d in b.TARGET:
        thresholds=[]
        for s in b.SEEDS:
            obj=joblib.load(b.old_path(d,s));scores=[];pred=[];truth=[]
            for code in [0,1,2]:
                p=obj['positions'][code];ad=obj['gate_models'][code]['adapter'];ma=p['vv'][ad][:,2];i=p['B']
                sc=-np.minimum(abs(ma-50),abs(ma-70))/10;sc[~D['input_qc'][i]]=np.nan
                scores.extend(sc);pred.extend(cls(ma));truth.extend(cls(D['medical'][i,2]))
            q=ns['select_threshold'](np.array(scores),np.array(pred),np.array(truth),.10 if b.TARGET[d]=='Huzhou' else .15)
            thresholds.append({'seed':s,'threshold':q['threshold'],'eligible':q['eligible'],'qualification':q['qualification']})
        out[d]=thresholds
    b.write(P/'MARGIN_DEVELOPMENT_FREEZE.json',{'selector_sha':b.sha(path),'directions':out,'test_access':False})

def score():
    assert (P/'ALL_MODELS_FROZEN.json').exists()
    plan=b.read(P/'PLAN.json');assert b.sha(b.__file__)==plan['runner_sha']
    for p,h in plan['immutable_hashes'].items():assert b.sha(p)==h,('immutable changed',p)
    freeze_margin();timings=[]
    for task in plan['tasks']:
        dest=P/'predictions'/(task['id']+'.npz')
        if dest.exists():continue
        r,runner,t,seq,c,sel=b.env(task['id'],task['cap']);D=b.data_at(task['horizon'])
        # No test labels are passed to any inference/decision function.
        D['medical'][:]=np.nan;D['ci'][:]=np.nan
        outputs=[];base=[];dual=[];acts=[];valids=[]
        for s in b.SEEDS:
            fr=b.read(r.ROOT/'frozen'/f'{s}.json');assert b.sha(fr['target_path'])==fr['target_sha']
            obj=joblib.load(fr['target_path']);idx=obj['test'];x=D['x'][idx]
            start=time.perf_counter();rr,hh,ff=infer(t,obj,x);elapsed=time.perf_counter()-start
            outputs.append((rr,hh,ff));pc=cls(ff[:,2]);valid=D['input_qc'][idx]&np.isfinite(ff).all(1);valids.append(valid)
            timings.append(dict(task=task['id'],seed=s,stage='batch_mapping_and_adaptation',records=len(idx),seconds=elapsed,seconds_per_record=elapsed/len(idx)))
            if task['arm'].startswith('prefix_'):
                dp=r.ROOT/'dual'/f'{task["direction"]}_{s}.joblib';assert b.sha(dp)==fr['dual_sha']
                heads=joblib.load(dp)['heads'][3];src=obj['source_model']
                f=runner.ma_features(x,ff,src['ym'],src['ys']);bs=runner.score_ma_gate(obj['gate_models'][3]['model'],f,pc)
                ds=np.maximum(seq.score_head(c,heads['low'],x[:,1],ff[:,2]),seq.score_head(c,heads['high'],x[:,1],ff[:,2]))
                base.append(bs);dual.append(ds);acts.append(class_actions(pc,bs,ds,valid,fr['threshold']))
        kw=dict(seed=b.SEEDS,ids=D['ids'][idx],groups=D['groups'][idx],index=idx,valid=np.array(valids),
                raw_med4=np.stack([v[0] for v in outputs]),pure_huber_med4=np.stack([v[1] for v in outputs]),final_med4=np.stack([v[2] for v in outputs]))
        if acts:kw.update(action=np.array(acts),base_score=np.array(base),dual_score=np.array(dual))
        save(task['id']+'.npz',**kw)
        print('SCORED',task['id'],flush=True)
    table('B4_inference_batch_timings.csv',timings)
    files=list((P/'predictions').glob('*.npz'));assert len(files)==len(plan['tasks'])
    b.write(P/'NEW_PREDICTIONS_FROZEN.json',{'files':{str(p):b.sha(p) for p in files},'reference_evaluation':False,'fit_units':0,'analysis_sha':b.sha(__file__)})

def metrics(a,p,y):
    y=np.broadcast_to(y,a.shape);good=a&(p==y);bad=a&(p!=y);n=a.size;k=int(a.sum())
    out={'events':n,'reported':k,'correct':int(good.sum()),'incorrect':int(bad.sum()),'withheld':int((~a).sum()),
         'agreement':float(good.sum()/k) if k else None,'coverage':k/n,'correct_yield':good.sum()/n,'incorrect_yield':bad.sum()/n,
         'within_reports':int((a&(p==0)).sum()),'normal_to_tail':int((a&(y==0)&(p!=0)).sum()),'low_high_cross':int((a&(y!=0)&(p==-y)).sum())}
    for lab,v in [('low',-1),('high',1)]:
        truth=y==v;badwithin=a&truth&(p==0)
        total=int(truth.sum());reported=int((a&truth).sum());normal=int((a&(p==0)).sum());nn=int(badwithin.sum())
        out.update({lab+'_support':total,lab+'_reported':reported,lab+'_to_within':nn,lab+'_correct':int((a&truth&(p==v)).sum()),lab+'_withheld':int((~a&truth).sum()),
                    lab+'_to_within_rate_all':nn/total if total else None,lab+'_to_within_rate_released':nn/reported if reported else None,lab+'_fraction_within_reports':nn/normal if normal else None})
    return out
def boot_ratio(num,den,groups):
    # 20 seeds stay together within each identity-proxy record/group cluster.
    if np.ndim(num)==2:num=np.sum(num,0)
    if np.ndim(den)==2:den=np.sum(den,0)
    unique=np.unique(groups);a=np.array([np.sum(num[groups==g]) for g in unique]);w=np.array([np.sum(den[groups==g]) for g in unique])
    ix=np.random.default_rng(20261027).integers(0,len(unique),(2000,len(unique)))
    dd=w[ix].sum(1);ok=dd>0
    if not ok.any():return [None,None]
    q=a[ix].sum(1)[ok]/dd[ok];return np.quantile(q,[.025,.975]).tolist()
def ratio_delta(an,ad,bn,bd,groups):
    arrays=[np.sum(v,0) if np.ndim(v)==2 else v for v in [an,ad,bn,bd]];ug=np.unique(groups)
    z=np.array([[v[groups==g].sum() for g in ug] for v in arrays]);ix=np.random.default_rng(20261027).integers(0,len(ug),(2000,len(ug)))
    sums=z[:,ix].sum(2);ok=(sums[1]>0)&(sums[3]>0)
    point=z[0].sum()/z[1].sum()-z[2].sum()/z[3].sum() if z[1].sum() and z[3].sum() else None
    ci=np.quantile(sums[0,ok]/sums[1,ok]-sums[2,ok]/sums[3,ok],[.025,.975]).tolist() if ok.any() else [None,None]
    return dict(estimate=point,ci_low=ci[0],ci_high=ci[1])
def with_ci(a,p,y,g):
    z=metrics(a,p,y);ones=np.ones_like(a);good=a&(p==y)
    for key,num,den in [('agreement',good,a),('coverage',a,ones),('correct_yield',good,ones),('incorrect_yield',a&~good,ones)]:
        q=boot_ratio(num,den,g);z[key+'_ci_low']=q[0];z[key+'_ci_high']=q[1]
    for key,v in [('low',-1),('high',1)]:
        truth=np.broadcast_to(y==v,a.shape);q=boot_ratio(a&truth&(p==0),truth,g)
        z[key+'_to_within_rate_all_ci_low']=q[0];z[key+'_to_within_rate_all_ci_high']=q[1]
    z['zero_report_seeds']=int((a.sum(1)==0).sum());z['records']=len(y);z['identity_proxy_groups']=len(np.unique(g));z['seeds']=len(a)
    return z

def old_predictions(root,d):
    z=load(root/'eval/predictions'/f'{d}.npz');D=b.data_at(900);idx=z['test_global_index']
    assert np.array_equal(z['test_ids'],D['ids'][idx])
    z.update(ids=z['test_ids'],index=idx,groups=D['groups'][idx],valid=np.broadcast_to(z['input_qc'],z['final_ma'].shape))
    return z

def summarize():
    frozen=b.read(P/'NEW_PREDICTIONS_FROZEN.json');assert all(b.sha(p)==h for p,h in frozen['files'].items())
    D=b.data_at(900);plan=b.read(P/'PLAN.json');rows=[];seedrows=[];adapt=[];flows=[];residual=[];budget=[]
    for d in b.TARGET:
        sets={'RIDGE-10':old_predictions(b.OLD,d),'KRR-1':old_predictions(KRR,d)}
        for name in ['PLS-2','SVR-1','MLP-8','TT-SMALL']:sets[name]=load(P/'predictions'/f'mapping_{name}__{d}.npz')
        sets['RIDGE-10-C20']=load(P/'predictions'/f'budget_C20__{d}.npz')
        baseline=sets['RIDGE-10'];idx=baseline['index'];ref=D['medical'][idx];y=cls(ref[:,2]);groups=D['groups'][idx]
        for name,z in sets.items():
            assert np.array_equal(z['ids'],baseline['ids']);assert np.array_equal(z['seed'],b.SEEDS)
            valid=np.broadcast_to(D['input_qc'][idx],z['final_med4'].shape[:2])
            assert np.isfinite(z['final_med4'][valid]).all()
            for stage in ['raw_med4','pure_huber_med4','final_med4']:
                med=z[stage];pc=cls(med[:,:,2]);row={'direction':d,'method':name,'stage':stage,'horizon':900,**metrics(valid,pc,y)}
                for j,param in enumerate(['R','K','MA','alpha']):
                    error=np.abs(med[:,:,j]-ref[None,:,j]);num=np.where(valid,error,0);q=boot_ratio(num,valid,groups)
                    row.update({param+'_MAE':float(num.sum()/valid.sum()),param+'_MAE_ci_low':q[0],param+'_MAE_ci_high':q[1]})
                    for s,err in zip(b.SEEDS,error):seedrows.append({'direction':d,'method':name,'stage':stage,'seed':s,'parameter':param,'MAE':float(err[valid[0]].mean())})
                rows.append(row)
            if name=='RIDGE-10':
                before=cls(z['raw_med4'][:,:,2]);after=cls(z['final_med4'][:,:,2]);rawerr=np.abs(z['raw_med4']-ref);finerr=np.abs(z['final_med4']-ref)
                for j,param in enumerate(['R','K','MA','alpha']):
                    q=ratio_delta(np.where(valid,rawerr[:,:,j],0),valid,np.where(valid,finerr[:,:,j],0),valid,groups)
                    adapt.append(dict(direction=d,parameter=param,contrast='raw_MAE_minus_adapted_MAE',**q))
                for v in [-1,0,1]:
                    for pre in [-1,0,1]:
                        for post in [-1,0,1]:flows.append(dict(direction=d,truth=v,before=pre,after=post,events=int((valid&(y==v)&(before==pre)&(after==post)).sum())))
                for j,i in enumerate(idx):
                    for k,s in enumerate(b.SEEDS):
                        residual.append(dict(direction=d,seed=s,record_id=str(D['ids'][i]),group=str(groups[j]),qc=bool(valid[k,j]),reference_MA=ref[j,2],raw_MA=z['raw_med4'][k,j,2],adapted_MA=z['final_med4'][k,j,2],residual_MA=z['final_med4'][k,j,2]-ref[j,2]))
        for n,z,stage in [(0,baseline,'raw_med4'),(20,sets['RIDGE-10-C20'],'final_med4'),(60 if b.TARGET[d]=='SAHZU' else 40,baseline,'final_med4')]:
            e=np.abs(z[stage]-ref);valid=baseline['valid'];p=cls(z[stage][:,:,2]);rr=dict(direction=d,calibration_n=n,**metrics(valid,p,y))
            for j,k in enumerate(['R','K','MA','alpha']):
                q=boot_ratio(e[:,:,j]*valid,valid,groups);rr.update({k+'_MAE':float(e[:,:,j][valid].mean()),k+'_ci_low':q[0],k+'_ci_high':q[1]})
            budget.append(rr)
    table('A3_mapping_comparison.csv',rows);table('B3_mapping_seed_values.csv',seedrows)
    table('A4_adaptation_paired.csv',adapt);table('A4_adaptation_class_flow.csv',flows);table('A3_MA_residual_records.csv',residual);table('B1_calibration_budget.csv',budget)
    gate_tables(D);temporal_tables(D);metadata(D);b.write(P/'STATISTICS_COMPLETE.json',{'status':'COMPLETE','tables':len(list((P/'tables').glob('*.csv'))),'new_fits':0})

def gate_tables(D):
    points=[];flows=[];seedrows=[];paired=[];risk=[];matched=[];margins=b.read(P/'MARGIN_DEVELOPMENT_FREEZE.json')['directions']
    for d in b.TARGET:
        z=old_predictions(b.OLD,d);u=load(EXT/'actions'/f'{d}.npz');assert np.array_equal(z['ids'],u['test_ids'])
        idx=z['index'];y=cls(D['medical'][idx,2]);g=D['groups'][idx];p=z['final_ma_class'];valid=z['valid'];ma=z['final_ma']
        score=-np.minimum(abs(ma-50),abs(ma-70))/10;am=np.zeros_like(valid)
        for j,q in enumerate(margins[d]):
            if q['threshold'] is not None:am[j]=valid[j]&(score[j]<=q['threshold'])
        actions={'All report':valid,'MA margin':am,'MA-ERROR-LR':z['lr_action'].astype(bool),'Dual protection':u['unified_action'].astype(bool)}
        for name,a in actions.items():
            rr=with_ci(a,p,y,g);points.append(dict(direction=d,method=name,horizon=900,**rr))
            for j,s in enumerate(b.SEEDS):seedrows.append(dict(direction=d,method=name,seed=s,**metrics(a[j:j+1],p[j:j+1],y)))
            for v in [-1,0,1]:
                for dest in [-1,0,1,9]:
                    mask=(~a) if dest==9 else (a&(p==dest));n=int((mask&(y==v)).sum());den=int((y==v).sum())*20
                    flows.append(dict(direction=d,method=name,truth=v,reported_class=dest,n=n,N=den,fraction=n/den if den else None))
        current=actions['Dual protection']
        for baseline_name in ['MA-ERROR-LR','All report']:
            base=actions[baseline_name]
            for metric,an,ad,bn,bd in [('coverage',current,np.ones_like(p),base,np.ones_like(p)),('agreement',current&(p==y),current,base&(p==y),base),('correct_abnormal_yield',current&(p==y)&(y!=0),np.ones_like(p),base&(p==y)&(y!=0),np.ones_like(p)),('abnormal_within_yield',current&(p==0)&(y!=0),np.ones_like(p),base&(p==0)&(y!=0),np.ones_like(p))]:
                paired.append(dict(direction=d,contrast='Dual minus '+baseline_name,metric=metric,**ratio_delta(an,ad,bn,bd,g)))
        # Descriptive equal-coverage risk curves, never operational thresholds.
        # Complete ties use stable record-ID ordering, frozen independently of truth.
        for population in ['all','predicted_within']:
            methods={'MA margin':score,'MA-ERROR-LR':z['lr_score']}
            if population=='predicted_within':methods['Dual within risk']=u['dual_score']
            for q in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
                aa={}
                common=valid.copy()
                if population=='predicted_within':common &= p==0
                for sc in methods.values():common &= np.isfinite(sc)
                for name,sc in methods.items():
                    a=np.zeros_like(valid)
                    for j in range(20):
                        ii=np.flatnonzero(common[j]);order=np.lexsort((z['ids'][ii].astype(str),sc[j,ii]));k=int(np.floor(q*len(ii)));a[j,ii[order[:k]]]=True
                    aa[name]=a;rr=metrics(a,p,y);risk.append(dict(direction=d,population=population,method=name,requested_fraction=q,eligible_events=int(common.sum()),**rr))
                for name in methods:
                    if name=='MA-ERROR-LR':continue
                    a=aa[name];base=aa['MA-ERROR-LR']
                    for key,an,bn,dena,denb in [('agreement',a&(p==y),base&(p==y),a,base),('abnormal_within_yield',a&(p==0)&(y!=0),base&(p==0)&(y!=0),np.ones_like(p),np.ones_like(p))]:
                        matched.append(dict(direction=d,population=population,method=name,requested_fraction=q,metric=key,**ratio_delta(an,dena,bn,denb,g)))
        save('gate_actions__'+d+'.npz',ids=z['ids'],groups=g,index=idx,pred=p,all_report=valid,margin=am,lr=actions['MA-ERROR-LR'],dual=current)
    table('A2_A5_gate_working_points.csv',points);table('A6_complete_class_destinations.csv',flows);table('B3_gate_seed_values.csv',seedrows)
    table('B2_gate_paired_differences.csv',paired);table('A5_risk_coverage.csv',risk);table('A5_matched_coverage.csv',matched)

def temporal_tables(D):
    rows=[];seeds=[];diff=[];firstrows=[]
    for d in b.TARGET:
        z=old_predictions(b.OLD,d);idx=z['index'];y=cls(D['medical'][idx,2]);g=D['groups'][idx];series=[]
        for h in [300,600]:
            if d in b.MISSING_PREFIX:
                p=load(P/'predictions'/f'prefix_{h}__{d}.npz');pred=cls(p['final_med4'][:,:,2]);a=p['action']
            else:
                p=load(b.SEQ/'actions'/f'T{h}'/f'{d}.npz');pred=p['pred'];a=p['action']
            assert np.array_equal(p['ids'],z['ids']);series.append((h,a,pred))
        u=load(EXT/'actions'/f'{d}.npz');series.append((900,u['unified_action'].astype(bool),u['final_ma_class']))
        ever=np.zeros_like(series[0][1]);fp=np.full_like(series[0][2],9);ft=np.full_like(fp,-1,dtype=int);prev_good=np.zeros_like(ever);prev_bad=np.zeros_like(ever)
        for h,a,pred in series:
            rows.append(dict(direction=d,mode='fixed_time',horizon=h,**with_ci(a,pred,y,g)))
            new=a&~ever;fp[new]=pred[new];ft[new]=h;ever|=a
            good=ever&(fp==y);bad=ever&(fp!=y)
            row=dict(direction=d,mode='cumulative_first_report',horizon=h,**with_ci(ever,fp,y,g),new_correct=int((good&~prev_good).sum()),new_incorrect=int((bad&~prev_bad).sum()))
            assert row['correct']+row['incorrect']+row['withheld']==row['events'];rows.append(row)
            for j,s in enumerate(b.SEEDS):seeds.append(dict(direction=d,seed=s,horizon=h,**metrics(ever[j:j+1],fp[j:j+1],y)))
            for key,an,bn,ad,bd in [('correct_yield',good,a&(pred==y),np.ones_like(a),np.ones_like(a)),('incorrect_yield',bad,a&(pred!=y),np.ones_like(a),np.ones_like(a)),('agreement',good,a&(pred==y),ever,a)]:
                diff.append(dict(direction=d,horizon=h,metric=key,contrast='sequential minus fixed',**ratio_delta(an,ad,bn,bd,g)))
            prev_good=good.copy();prev_bad=bad.copy()
        for h in [300,600,900,-1]:firstrows.append(dict(direction=d,first_report_seconds=h,n=int((ft==h).sum()),N=ft.size,proportion=float((ft==h).mean())))
        save('sequence__'+d+'.npz',ids=z['ids'],groups=g,index=idx,first_class=fp,first_time=ft,reported=ever)
    table('A7_pointwise_cumulative.csv',rows);table('B3_sequence_seed_values.csv',seeds);table('B2_sequence_fixed_paired.csv',diff);table('A7_first_report_times.csv',firstrows)

def metadata(D):
    role=b.read(b.BASE/'ROLE_FREEZE.json');rows=[];roles=[]
    for center in ['Huzhou','SAHZU','Hubei']:
        ix=np.flatnonzero(D['centers']==center);y=cls(D['medical'][ix,2]);q=D['input_qc'][ix]
        rows.append(dict(center=center,records=len(ix),identity_proxy_groups=len(np.unique(D['groups'][ix])),QC_pass=int(q.sum()),QC_fail=int((~q).sum()),low=int((y==-1).sum()),within=int((y==0).sum()),high=int((y==1).sum()),patient_identity_verified=False))
    for d in b.TARGET:
        obj=joblib.load(b.old_path(d,1000))
        for name,ix in [('source',obj['source_rows']),('calibration_full',obj['pool']),('external_test',obj['test'])]:
            y=cls(D['medical'][ix,2]);roles.append(dict(direction=d,role=name,records=len(ix),identity_proxy_groups=len(np.unique(D['groups'][ix])),low=int((y==-1).sum()),within=int((y==0).sum()),high=int((y==1).sum()),QC_pass=int(D['input_qc'][ix].sum())))
        sets=[set(D['groups'][obj[k]]) for k in ['source_rows','pool','test']]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2],('group leakage',d)
    table('A1_center_QC.csv',rows);table('A1_data_roles.csv',roles)
    b.write(P/'HARDWARE.json',dict(platform=platform.platform(),processor=platform.processor(),python=sys.version,numpy=np.__version__,logical_cpus=__import__('os').cpu_count(),threads_per_worker=1,concurrent_training_workers=3,device='CPU',timing_note='Saved per-task fitting includes bookkeeping; batch inference timing excludes acquisition and source fitting. Identity proxies are not verified patients.'))

if __name__=='__main__':
    {'score':score,'summarize':summarize}[sys.argv[1]]()
