"""Metric, grouping and reporting summaries for the current analyses."""
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c, current_suite as suite, mapping, analysis
from .gates import clean_write

ROOT=suite.ROOT
OUT=ROOT/'summary'
NAMES={'target_only_SA':'Target-only','HZ_to_SA':'单源','HZ+HB_to_SA':'双源'}
READOUTS=['R','K','MA','angle']
LABELS=['low','normal','high']
ROWS=[]
SCORES={}
CHECKS=[]


def states(x):
    x=np.asarray(x)
    return np.where(~np.isfinite(x),9,np.where(x<50,-1,np.where(x>70,1,0)))


def div(a,b):return float(a/b) if b else None


def score(pred,truth,action=None):
    """Recompute counts without using the production accounting functions."""
    p=np.asarray(pred,float);y=np.broadcast_to(truth,p.shape)
    assert np.isfinite(y).all()
    valid=np.isfinite(p);a=valid if action is None else np.asarray(action,bool)&valid
    pc=states(p);tc=states(y);cm=np.zeros((3,3),int)
    for i,ti in enumerate([-1,0,1]):
        for j,pj in enumerate([-1,0,1]):cm[i,j]=np.sum(a&(tc==ti)&(pc==pj))
    withheld=[int(np.sum(~a&(tc==i))) for i in [-1,0,1]]
    n=int(a.sum());correct=int(np.trace(cm));fn=int(cm[0,1]+cm[2,1]);fp=int(cm[1,0]+cm[1,2])
    ad=int(cm[0].sum()+cm[2].sum());nd=int(cm[1].sum())
    err=p[a]-y[a];den=np.sum((y[a]-y[a].mean())**2) if n else 0
    result=dict(events=p.size,reported=n,correct=correct,incorrect=n-correct,withheld=p.size-n,
        coverage=n/p.size,agreement=div(correct,n),fn=fn,fn_denominator=ad,fnr=div(fn,ad),
        fp=fp,fp_denominator=nd,fpr=div(fp,nd),low_recall=div(cm[0,0],cm[0].sum()),high_recall=div(cm[2,2],cm[2].sum()),
        confusion=cm.tolist(),binary_confusion=[[nd-fp,fp],[fn,ad-fn]],withheld_by_true_state=withheld,
        all_true_class_counts=[int(np.sum(tc==i)) for i in [-1,0,1]],
        correct_yield=correct/p.size,incorrect_yield=(n-correct)/p.size,
        mae=float(np.mean(abs(err))) if n else None,rmse=float(np.sqrt(np.mean(err**2))) if n else None,
        r2=float(1-np.sum(err**2)/den) if den else None)
    result['class_mae']={};result['class_errors']={}
    for name,k in zip(LABELS,[-1,0,1]):
        e=p[a&(tc==k)]-y[a&(tc==k)]
        result['class_mae'][name]=float(np.mean(abs(e))) if len(e) else None
        result['class_errors'][name]=dict(n=len(e),mae=float(np.mean(abs(e))) if len(e) else None,
            rmse=float(np.sqrt(np.mean(e**2))) if len(e) else None,
            bias=float(np.mean(e)) if len(e) else None,
            median_ae=float(np.median(abs(e))) if len(e) else None,
            p90_ae=float(np.quantile(abs(e),.9)) if len(e) else None)
    recalls=[div(cm[i,i],cm[i].sum()) for i in range(3)]
    result['balanced_accuracy']=float(np.mean([x for x in recalls if x is not None])) if n else None
    result['macro_f1']=float(np.mean([div(2*cm[i,i],cm[i].sum()+cm[:,i].sum()) or 0 for i in range(3)])) if n else None
    assert correct+(n-correct)+(p.size-n)==p.size
    assert np.array_equal(cm.sum(1)+withheld,result['all_true_class_counts'])
    return result


def compare(actual,stored,where):
    for key,v in actual.items():
        if key not in stored:continue
        w=stored[key]
        if isinstance(v,dict):compare(v,w,where+'/'+key)
        elif v is None:assert w is None,(where,key,w)
        else:np.testing.assert_allclose(v,w,rtol=1e-11,atol=1e-11,err_msg=where+'/'+key)


def add(protocol,section,arm,variant,pred,y,action=None,stored=None):
    key='|'.join([protocol,section,arm,variant]);assert key not in SCORES,key
    m=score(pred,y,action);compare(m,c.summarize(pred,y,action),key+'/production')
    if stored is not None:compare(m,stored,key+'/saved')
    SCORES[key]=m
    ROWS.append(dict(protocol=protocol,section=section,arm=arm,variant=variant,**m))
    return m


def readouts(p,y):
    p=np.asarray(p);y=np.broadcast_to(y,p.shape);ans={}
    for j,name in enumerate(READOUTS):
        ok=np.isfinite(p[...,j]);a=p[...,j][ok];b=y[...,j][ok];e=a-b;den=np.sum((b-b.mean())**2)
        ans[name]=dict(events=len(a),missing_events=ok.size-int(ok.sum()),mae=float(np.mean(abs(e))),
            rmse=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/den) if den else None)
    return ans


def inventory():
    expected={
        'primary':[f'{a}_T{h}_B0.npz' for a in c.TARGET for h in c.HORIZONS]+
            [f'{a}_T900_B{b}.npz' for a in c.TARGET for b in ([20,40] if c.TARGET[a]=='SAHZU' else [20])],
        'all301':[f'T{h}.npz' for h in c.HORIZONS],
        'mapping':[f'{a}_{m}.npz' for a in c.TARGET if not a.startswith('target_only') for m in mapping.METHODS],
        'calibration':[f'{a}.npz' for a in c.TARGET if not a.startswith('target_only')],
        'all301_mapping':[f'{m}.npz' for m in mapping.METHODS],
        'gates':[f'{a}_T{h}.npz' for a in c.TARGET for h in c.HORIZONS],
        'all301_gates':[f'{a}_T{h}.npz' for a in c.MAIN for h in c.HORIZONS],
        'controls':[f'{a}_{r}_{l}.npz' for a in c.MAIN+['all301'] for r,l in [('B','uniform'),('B','class'),('three','inward4'),('four','inward4')]],
        'all301_local_controls':['budget20.npz','budget40.npz','budget60.npz']+[f'calibration_{a}.npz' for a in c.MAIN[1:]],
        'sequence':[f'{a}.npz' for a in c.TARGET]+['all301.npz'],
        'cost_comparison':['ALL301_PREDICTIONS.npz']+[f'PAPER_{a}.npz' for a in c.MAIN]}
    for folder,names in expected.items():
        actual={p.name for p in (ROOT/folder).glob('*.npz')}
        assert set(names)<=actual,(folder,'missing',set(names)-actual)
    return {k:dict(expected_count=len(v),files=v) for k,v in expected.items()}


def split_audit(d,folds,local):
    def separate(a,b):assert not set(d['groups'][a])&set(d['groups'][b])
    seen=[];rows=[]
    for fi,f in enumerate(folds):
        a=np.asarray(f['A']);b=np.asarray(f['B']);separate(a,b);seen.extend(b)
        assert set(a)|set(b)==set(local)
        inner_seen=[]
        for s in f['inner']:
            separate(s['T'],s['V']);assert set(s['T'])|set(s['V'])==set(a);inner_seen.extend(s['V'])
        assert sorted(inner_seen)==sorted(a)
        counts=lambda v:[int(np.sum(states(d['y'][v,2])==k)) for k in [-1,0,1]]
        rows.append(dict(fold=fi,train=len(a),test=len(b),train_counts=counts(a),test_counts=counts(b)))
    assert sorted(seen)==sorted(local)
    for arm in c.TARGET:
        for budget in ([0,20,40] if c.TARGET[arm]=='SAHZU' else [0,20]):
            for seed in c.SEEDS:
                sp=c.paper_spec(arm,seed,budget);separate(sp['pool'],sp['test'])
                separate(c.source_rows(arm,900),sp['test'])
                for s in sp['positions'][3]['inner']:separate(s['T'],s['V'])
    clean_write(OUT/'SPLIT_COUNTS.json',dict(all301=rows,local_records=len(local),local_proxy_groups=len(set(d['groups'][local])),
        class_counts=[int(np.sum(states(d['y'][local,2])==k)) for k in [-1,0,1]],
        outer_each_once=True,inner_groups_disjoint=True,paper_pool_test_source_groups_disjoint=True))


def aggregate(d,local):
    multi={};nonconvergence=[]
    for path in sorted((ROOT/'primary').glob('*.npz')):
        z=analysis.load(path);saved=c.read(path.with_suffix('.metrics.json'));arm=saved['direction'];h=saved['horizon'];budget=saved['budget']
        assert np.array_equal(z['seed'],c.SEEDS)
        for si,seed in enumerate(c.SEEDS):
            sp=c.paper_spec(arm,seed,budget);np.testing.assert_array_equal(z['test'],sp['test'])
            np.testing.assert_array_equal(z['training'][si],sp['pool'][c.qc_at(h)[sp['pool']]])
        y=d['y'][z['test'],2]
        for key,ac in [('all_report',None),('boundary95',z['accepted'])]:
            add('paper','primary',arm,f'T{h}_B{budget}_{key}',z['prediction'],y,ac,saved[key])
    for h in c.HORIZONS:
        z=analysis.load(ROOT/'all301'/f'T{h}.npz');np.testing.assert_array_equal(z['local'],local)
        saved=c.read(ROOT/'all301'/f'T{h}.metrics.json')
        for j,arm in enumerate(c.MAIN):
            for key,ac in [('all_report',None),('boundary95',z['accepted'][j])]:
                add('all301','primary',arm,f'T{h}_{key}',z['prediction'][j],d['y'][local,2],ac,saved[arm][key])
    for folder in ['mapping','calibration']:
        for path in sorted((ROOT/folder).glob('*.npz')):
            z=analysis.load(path);saved=c.read(path.with_suffix('.metrics.json'));y=d['medical'][z['test']]
            for kind in ['raw','affine','joint','huber','complete']:
                key=f'{folder}/{path.stem}/{kind}';multi[key]=readouts(z[kind],y)
                add('paper',folder,path.stem,kind,z[kind][...,2],y[:,2],stored=saved[kind])
            for a in c.read(path.with_suffix('.audit.json')):
                if not a['huber']['success']:nonconvergence.append(dict(file=str(path),seed=a['seed'],status=a['huber']['status']))
    for path in sorted((ROOT/'all301_mapping').glob('*.npz')):
        z=analysis.load(path);saved=c.read(path.with_suffix('.metrics.json'))
        for j,arm in enumerate(c.MAIN):
            multi[f'all301_mapping/{path.stem}/{arm}']=readouts(z['prediction'][j],d['medical'][local])
            for key,ac in [('all_report',None),('gate',z['accepted'][j])]:
                add('all301','six_models',arm,path.stem+'_'+key,z['prediction'][j,:,2],d['y'][local,2],ac,saved[arm][key])
    for folder in ['controls','all301_local_controls']:
        for path in sorted((ROOT/folder).glob('*.npz')):
            z=analysis.load(path);saved=c.read(path.with_suffix('.metrics.json'))
            if path.stem.startswith('calibration_'):
                arm=path.stem.removeprefix('calibration_')
                for kind in ['raw','affine','joint','huber','complete']:
                    multi[f'all301_calibration/{arm}/{kind}']=readouts(z[kind],d['medical'][local])
                    add('all301','calibration',arm,kind,z[kind][:,2],d['y'][local,2],stored=saved[kind])
                for a in c.read(path.with_suffix('.audit.json')):
                    if not a['huber']['success']:nonconvergence.append(dict(file=str(path),fold=a['fold'],status=a['huber']['status']))
                continue
            is301='local' in z
            for j,arm in enumerate(c.MAIN if is301 else [next(a for a in c.MAIN if path.stem.startswith(a+'_'))]):
                p=z['prediction'][j] if is301 else z['prediction'];aa=z['accepted'][j] if is301 else z['accepted'];ss=saved[arm] if is301 else saved
                for key,ac in [('all_report',None),('gate',aa)]:
                    add('all301' if is301 else 'paper',folder,arm,path.stem+'_'+key,p,d['y'][z['local'] if is301 else z['test'],2],ac,ss[key])
    med=c.read(ROOT/'controls/MEDIAN_RESULTS.json')
    for key,stored in med.items():
        arm,budget=key.rsplit('_B',1);ps=[]
        for seed in c.SEEDS:
            sp=c.paper_spec(arm,seed,int(budget));ps.append(np.full(len(sp['test']),np.median(d['y'][sp['pool'],2])))
        add('paper','median',arm,'B'+budget,np.asarray(ps),d['y'][sp['test'],2],stored=stored)
    for protocol in ['paper','all301']:
        for arm in (c.TARGET if protocol=='paper' else c.MAIN):
            for h in c.HORIZONS:
                path=ROOT/('gates' if protocol=='paper' else 'all301_gates')/f'{arm}_T{h}.npz'
                z=analysis.load(path);saved=c.read(path.with_suffix('.metrics.json'));y=d['y'][z['test'] if protocol=='paper' else z['local'],2]
                for rule in (['LR','HGB','TWO_SIDED','BOUNDARY_PAPER'] if protocol=='paper' else ['LR','HGB','TWO_SIDED','BOUNDARY_PAPER','BOUNDARY95']):
                    aa=z[rule+'_action'] if protocol=='paper' else z[rule]
                    add(protocol,'gates',arm,f'T{h}_{rule}',z['prediction'],y,aa,saved[rule])
    costs=c.read(ROOT/'cost_comparison/RESULTS.json')
    for protocol in ['all301','paper']:
        for j,arm in enumerate(c.MAIN):
            z=analysis.load(ROOT/'cost_comparison'/('ALL301_PREDICTIONS.npz' if protocol=='all301' else f'PAPER_{arm}.npz'))
            y=d['y'][z['local'] if protocol=='all301' else z['test'],2]
            for k,cost in enumerate(['4','8','16','inner_selected']):
                if cost=='inner_selected':p=z['selected'][j] if protocol=='all301' else z['selected'];a=z['selected_action'][j] if protocol=='all301' else z['selected_action']
                else:p=z['predictions'][j,k] if protocol=='all301' else z['predictions'][:,k];a=z['accepted'][j,k] if protocol=='all301' else z['accepted'][:,k]
                for kind,ac in [('all_report',None),('gate',a)]:
                    add(protocol,'cost',arm,cost+'_'+kind,p,y,ac,costs[protocol][arm][cost][kind])
    clean_write(OUT/'FOUR_READOUT_ERRORS.json',multi)
    clean_write(OUT/'HUBER_OPTIMIZER_STATUS.json',dict(nonconvergence=nonconvergence,count=len(nonconvergence),
        handling='Finite fitted outputs retained and convergence failures disclosed; no silent method replacement.'))


def sequence_audit(d):
    differences=[]
    for protocol in ['paper','all301']:
        stored=c.read(ROOT/'sequence'/('RESULTS.json' if protocol=='paper' else 'ALL301_ALL_RULES.json'))
        for arm in (c.TARGET if protocol=='paper' else c.MAIN):
            points={h:analysis.load(ROOT/('gates' if protocol=='paper' else 'all301_gates')/f'{arm}_T{h}.npz') for h in c.HORIZONS}
            ids=points[900]['test'] if protocol=='paper' else points[900]['local'];y=d['y'][ids,2];saved={}
            existing=analysis.load(ROOT/'sequence'/f'{arm}.npz') if protocol=='paper' else None
            for rule in analysis.RULES:
                values={}
                pp={h:points[h]['prediction'] for h in c.HORIZONS}
                aa={h:np.isfinite(pp[h]) if rule=='ALL_REPORT' else points[h]['boundary95_action'] if protocol=='paper' and rule=='BOUNDARY95'
                    else points[h][rule+'_action'] if protocol=='paper' else points[h][rule] for h in c.HORIZONS}
                for name,times in analysis.SCHEDULES.items():
                    p,t,a=analysis.replay(pp,aa,times)
                    # Independent earliest-eligible checkpoint verification.
                    first=np.full(p.shape,-1,int)
                    for h in times:first=np.where((first<0)&aa[h]&np.isfinite(pp[h]),h,first)
                    np.testing.assert_array_equal(t,first)
                    for h in times:np.testing.assert_array_equal(p[t==h],pp[h][t==h])
                    m=add(protocol,'sequence',arm,rule+'_'+name,p,y,a,stored[arm][rule][name]);values[name]=m
                    by_time={str(h):dict(reported=int(np.sum(t==h)),correct=int(np.sum((t==h)&(states(p)==states(y))))) for h in times}
                    assert by_time==stored[arm][rule][name]['by_time']
                    for key,v in [('prediction',p),('time',t),('action',a)]:
                        label=f'{rule}__{name}_{key}';saved[label]=v
                        if existing is not None:np.testing.assert_array_equal(v,existing[label])
                for left,right in [('fixed15','10-15'),('fixed15','5-10-15'),('10-15','5-10-15')]:
                    differences.append(dict(protocol=protocol,arm=arm,rule=rule,baseline=left,alternative=right,
                        **{k:values[right][k]-values[left][k] for k in ['correct','incorrect','withheld','correct_yield','incorrect_yield','coverage']}))
            if protocol=='all301':np.savez_compressed(OUT/f'ALL301_SEQUENCE_{arm}.npz',**saved,local=ids)
    clean_write(OUT/'SEQUENCE_PAIRED_COUNTS.json',differences)


def matched_and_intervals(d,local):
    matched={}
    for arm in c.MAIN:
        z=analysis.load(ROOT/'gates'/f'{arm}_T900.npz');idx=z['test'];truth=states(d['y'][idx,2]);pc=states(z['prediction'])
        acts={k:np.zeros(pc.shape,bool) for k in ['LR','DUAL']};details=[]
        for si,seed in enumerate(c.SEEDS):
            eligible=np.flatnonzero((pc[si]==0)&np.isfinite(z['LR_score'][si])&np.isfinite(z['DUAL_score'][si]));n=int(np.floor(len(eligible)*.2))
            for key in acts:
                chosen=sorted(eligible,key=lambda i:(z[key+'_score'][si,i],str(d['ids'][idx[i]])))[:n];acts[key][si,chosen]=True
            details.append(dict(seed=seed,candidates=len(eligible),selected=n))
        ab=truth!=0;delta=(acts['DUAL']&ab).astype(float)-(acts['LR']&ab)
        matched[arm]=dict(scope='Common finite-score predicted-normal candidates; exact floor20% quota per seed; query labels used for scoring only.',
            rows=details,LR=dict(released=int(acts['LR'].sum()),abnormal_as_normal=int((acts['LR']&ab).sum())),
            DUAL=dict(released=int(acts['DUAL'].sum()),abnormal_as_normal=int((acts['DUAL']&ab).sum())),
            paired_difference=c.accounting.paired_ratio(delta,acts['LR'],d['groups'][idx]))
        np.savez_compressed(OUT/f'MATCHED20_{arm}.npz',**acts,test=idx)
    clean_write(OUT/'MATCHED20.json',matched)
    primary=analysis.load(ROOT/'all301/T900.npz');cost=analysis.load(ROOT/'cost_comparison/ALL301_PREDICTIONS.npz');y=d['y'][local,2];tc=states(y);group=d['groups'][local];intervals={}
    pairs=[(a+' vs target',primary['prediction'][j],primary['prediction'][0]) for j,a in enumerate(c.MAIN[1:],1)]
    pairs += [(a+' selected-cost vs baseline',cost['selected'][j],primary['prediction'][j]) for j,a in enumerate(c.MAIN)]
    for name,p,q in pairs:
        pc=states(p);qc=states(q)
        quantities=dict(agreement=((pc==tc).astype(float)-(qc==tc),np.ones(len(y))),
            fnr=(((tc!=0)&(pc==0)).astype(float)-((tc!=0)&(qc==0)),tc!=0),
            fpr=(((tc==0)&(pc!=0)).astype(float)-((tc==0)&(qc!=0)),tc==0),mae=(abs(p-y)-abs(q-y),np.ones(len(y))))
        intervals[name]={k:c.accounting.paired_ratio(v[None,:],den[None,:],group) for k,(v,den) in quantities.items()}
    clean_write(OUT/'ALL301_PAIRED_INTERVALS.json',dict(results=intervals,
        interpretation='Descriptive identity-proxy group bootstrap of fixed OOF predictions; no full refitting, no correction for research-history selection, no independent validation.'))


def select(protocol,section,arm,variant):return SCORES['|'.join([protocol,section,arm,variant])]


def metric_row(arm,m,gate=None):
    return [NAMES.get(arm,arm),analysis.pct(m['agreement']),analysis.pct(m['fnr'])+f" ({m['fn']}/{m['fn_denominator']})",
        analysis.pct(m['fpr']),analysis.pct(m['low_recall'])+'/'+analysis.pct(m['high_recall']),analysis.num(m['mae']),analysis.num(m['rmse']),analysis.num(m['r2']),analysis.pct((gate or m)['coverage'])]






