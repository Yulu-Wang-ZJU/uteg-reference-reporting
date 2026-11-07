"""Aggregate current-version evidence for manuscript figures and tables.

Public tables contain aggregate/per-seed statistics only. Record-level inputs,
predictions, identities, curves and group assignments never enter these tables.
"""
import hashlib,time,platform,json
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,analysis,gates,current_finalize as check

ROOT=suite.ROOT;OUT=ROOT/'manuscript_sync';AUX=OUT/'experiments/auxiliary';REP=OUT/'experiments/representation'
READOUTS=['R','K','MA','angle'];DIRECTIONS=[a for a in c.TARGET if not a.startswith('target_only')]
BOOKS={str(i):{} for i in range(1,9)};E={};WEIGHTS={}

def flat(m,prefix=''):
    out={}
    for k,v in m.items():
        key=prefix+('normal_reports' if k=='within_reports' else k)
        if isinstance(v,dict):out.update(flat(v,key+'_'))
        elif isinstance(v,(list,tuple,np.ndarray)):out[key]=json.dumps(gates.safe(v),separators=(',',':'))
        else:out[key]=v
    return out

def add(book,sheet,row):BOOKS[str(book)].setdefault(sheet,[]).append(flat(row))

def ratio(num,den,groups,reps=5000):
    n=np.atleast_2d(np.asarray(num,float));d=np.broadcast_to(den,n.shape).astype(float);groups=np.asarray(groups)
    valid=np.isfinite(n)&np.isfinite(d)
    n=np.where(valid,n,0.);d=np.where(valid,d,0.)
    names,inv=np.unique(groups,return_inverse=True);ng=len(names)
    ns=np.bincount(inv,weights=n.sum(0),minlength=ng);ds=np.bincount(inv,weights=d.sum(0),minlength=ng)
    key=(ng,reps)
    if key not in WEIGHTS:WEIGHTS[key]=np.random.default_rng(20261103).multinomial(ng,np.ones(ng)/ng,size=reps)
    w=WEIGHTS[key];bn=w@ns;bd=w@ds;ok=bd>0
    return dict(estimate=float(ns.sum()/ds.sum()) if ds.sum()>0 else None,low=float(np.quantile(bn[ok]/bd[ok],.025)) if ok.any() else None,
        high=float(np.quantile(bn[ok]/bd[ok],.975)) if ok.any() else None,valid=int(ok.sum()),resamples=reps)

def paired_ratio_difference(left_num,left_den,right_num,right_den,groups,reps=2000):
    """Paired group bootstrap with each policy's own released denominator."""
    groups=np.asarray(groups);names,inv=np.unique(groups,return_inverse=True);ng=len(names)
    parts=[]
    for num,den in [(left_num,left_den),(right_num,right_den)]:
        n=np.atleast_2d(np.asarray(num,float));d=np.broadcast_to(den,n.shape).astype(float)
        valid=np.isfinite(n)&np.isfinite(d);n=np.where(valid,n,0);d=np.where(valid,d,0)
        parts.append((np.bincount(inv,weights=n.sum(0),minlength=ng),np.bincount(inv,weights=d.sum(0),minlength=ng)))
    key=(ng,reps)
    if key not in WEIGHTS:WEIGHTS[key]=np.random.default_rng(20261103).multinomial(ng,np.ones(ng)/ng,size=reps)
    w=WEIGHTS[key];(ln,ld),(rn,rd)=parts;bl=w@ld;br=w@rd;ok=(bl>0)&(br>0)
    samples=(w@ln)[ok]/bl[ok]-(w@rn)[ok]/br[ok]
    return dict(estimate=float(ln.sum()/ld.sum()-rn.sum()/rd.sum()) if ld.sum()>0 and rd.sum()>0 else None,low=float(np.quantile(samples,.025)) if len(samples) else None,high=float(np.quantile(samples,.975)) if len(samples) else None,valid=int(ok.sum()),resamples=reps)

def summary(pred,y,action,groups,reps=2000):
    p=np.atleast_2d(pred);a=np.isfinite(p) if action is None else np.atleast_2d(action)&np.isfinite(p)
    t=c.accounting.states(y);q=c.accounting.states(p);good=a&(q==t);bad=a&~good;one=np.ones_like(p)
    m=check.score(p,y,a)
    m['intervals']={k:ratio(n,de,groups,reps) for k,n,de in [('agreement',good,a),('coverage',a,one),('correct_yield',good,one),('incorrect_yield',bad,one),('mae',np.where(a,abs(p-y),0),a)]}
    return m

def gate_action(z,rule):
    if rule=='ALL_REPORT':return np.isfinite(z['prediction'])
    if rule=='BOUNDARY95':return z['boundary95_action'] if 'boundary95_action' in z else z['BOUNDARY95_action']
    return z[rule+'_action']

def destinations(book,sheet,meta,p,y,a):
    p=np.atleast_2d(p);a=np.atleast_2d(a);q=np.where(a,c.accounting.states(p),9);t=c.accounting.states(y)
    for state in [-1,0,1]:
        den=int((t==state).sum()*len(p))
        for target in [-1,0,1,9]:add(book,sheet,dict(**meta,true_state=state,report_state=target,count=int(((q==target)&(t==state)).sum()),denominator=den))

def readouts(p,y,idx):
    p=np.asarray(p);v=check.readouts(p,y);d,*_=c.context()
    for j,name in enumerate(READOUTS):
        valid=np.isfinite(p[:,:,j]);v[name]['interval']=ratio(np.where(valid,abs(p[:,:,j]-y[:,j]),0),valid,d['groups'][idx])
    return v

def cohort():
    d,_,_,_,role,folds,local=c.context();E['cohort']=[]
    for center in ['SAHZU','Huzhou','Hubei']:
        idx=np.flatnonzero(d['centers']==center);qc=d['input_qc'][idx];t=c.accounting.states(d['y'][idx,2])
        row=dict(center=center,records=len(idx),qc_pass=int(qc.sum()),low=int((t==-1).sum()),normal=int((t==0).sum()),high=int((t==1).sum()))
        E['cohort'].append(row);add(1,'Cohort and QC',row)
        r=role['roles'][center]
        for name,ix in [('development',r['pool_full']),('evaluation',r['test'])]:
            tc=c.accounting.states(d['y'][ix,2]);add(1,'Data roles',dict(center=center,role=name,records=len(ix),low=int((tc==-1).sum()),normal=int((tc==0).sum()),high=int((tc==1).sum())))
    for row in c.read(ROOT/'summary/SPLIT_COUNTS.json')['all301']:add(8,'Frozen folds',row)

def predictions():
    d,*_=c.context();E['mapping']={};E['primary']={};E['calibration']={};E['budget']={}
    for arm in DIRECTIONS:
        z=analysis.load(AUX/f'{arm}_B0.npz');idx=z['test'];y=d['medical'][idx];E['primary'][arm]={}
        for phase in ['raw','huber','complete']:
            v=readouts(z[phase],y,idx);E['primary'][arm][phase]=v
            add(2,'Paired adaptation',dict(arm=arm,phase=phase,**v))
            for si,seed in enumerate(c.SEEDS):add(5,'All mapping seeds',dict(arm=arm,phase=phase,seed=seed,**check.readouts(z[phase][si],y)))
        delta=ratio(abs(z['complete'][:,:,2]-y[:,2])-abs(z['raw'][:,:,2]-y[:,2]),np.ones(z['raw'].shape[:2]),d['groups'][idx])
        E['primary'][arm]['paired_MAE']=delta;add(2,'Adaptation paired gains',dict(arm=arm,contrast='complete minus raw MAE',**delta))
        for before in [-1,0,1]:
            for after in [-1,0,1]:
                for truth in [-1,0,1]:add(2,'Class transitions',dict(arm=arm,truth=truth,raw=before,adapted=after,count=int(((c.accounting.states(z['raw'][:,:,2])==before)&(c.accounting.states(z['complete'][:,:,2])==after)&(c.accounting.states(y[:,2])==truth)).sum())))
        E['mapping'][arm]={}
        for method in suite.mapping.METHODS:
            s=analysis.load(AUX/f'six_{arm}_{method}.npz');v=readouts(s['complete'],y,idx);E['mapping'][arm][method]=v
            add(2,'Mapping baselines',dict(arm=arm,method=method,phase='complete',**v))
        q=analysis.load(ROOT/'calibration'/f'{arm}.npz');E['calibration'][arm]={}
        for kind in ['raw','affine','joint','huber','complete']:
            p=q[kind].copy()
            if kind=='complete':p=z['complete']
            v=readouts(p,y,idx);E['calibration'][arm][kind]=v;add(2,'Adaptation comparators',dict(arm=arm,method=kind,**v))
        E['budget'][arm]={}
        for b in ([20,40,60] if c.TARGET[arm]=='SAHZU' else [20,40]):
            zz=analysis.load(AUX/f'{arm}_B{0 if b==(60 if c.TARGET[arm]=="SAHZU" else 40) else b}.npz');v=readouts(zz['complete'],y,idx)
            E['budget'][arm][str(b)]=v;add(5,'Adaptation budgets',dict(arm=arm,budget=b,**v))
    E['ridge_budget']={}
    for b in [20,40,60]:
        z=analysis.load(AUX/f'target_ridge_B{b}.npz');idx=z['test'];y=d['medical'][idx];v=readouts(z['prediction'],y,idx);E['ridge_budget'][str(b)]=v;add(7,'Budget_4param_Aggregate',dict(arm='target_Ridge',budget=b,**v))
        for arm in c.MAIN[1:]:
            q=analysis.load(AUX/f'{arm}_B{0 if b==60 else b}.npz')
            for j,name in enumerate(READOUTS):add(7,'Budget_Paired',dict(arm=arm,budget=b,readout=name,contrast='target Ridge minus source',**ratio(abs(z['prediction'][:,:,j]-y[:,j])-abs(q['complete'][:,:,j]-y[:,j]),np.ones((20,len(idx))),d['groups'][idx])))
            add(7,'Budget_4param_Aggregate',dict(arm=arm,budget=b,**E['budget'][arm][str(b)]))
        for au in c.read((AUX/f'target_ridge_B{b}.npz').with_suffix('.audit.json')):add(7,'Budget_DevAlpha',dict(budget=b,seed=au['seed'],alpha=au['alpha'],inner_mae=au['inner_mae']))

def reports():
    d,*_=c.context();E['reporting']={};E['eligibility']=[];E['matched']={}
    for arm in c.TARGET:
        z=analysis.load(ROOT/'gates'/f'{arm}_T900.npz');idx=z['test'];y=d['y'][idx,2];gg=d['groups'][idx];E['reporting'][arm]={}
        for rule in analysis.RULES:
            a=gate_action(z,rule);m=summary(z['prediction'],y,a,gg);E['reporting'][arm][rule]=m
            add(3,'Gate working points',dict(arm=arm,rule=rule,**m));destinations(3,'Report destinations',dict(arm=arm,rule=rule),z['prediction'],y,a)
            for si,s in enumerate(c.SEEDS):add(5,'All reporting seeds',dict(arm=arm,rule=rule,seed=s,**check.score(z['prediction'][si],y,a[si])))
        if arm in DIRECTIONS:
            heads=c.read((ROOT/'gates'/f'{arm}_T900.npz').with_suffix('.support.json'));ths=c.read((ROOT/'gates'/f'{arm}_T900.npz').with_suffix('.thresholds.json'))
            row=dict(arm=arm,records=len(idx),low_records=int((y<50).sum()),normal_records=int(((y>=50)&(y<=70)).sum()),high_records=int((y>70).sum()),low_head=sum(h['low']['status']=='SUPPORTED' for h in heads),high_head=sum(h['high']['status']=='SUPPORTED' for h in heads),normal_qualified=sum(h['dual']['normal'] is not None for h in ths),tail_selected=sum(not h['dual']['all_withhold'] for h in ths),tail_numeric=sum(h['dual']['tail'] is not None for h in ths),reporting_seeds=int(np.sum(z['TWO_SIDED_action'].sum(1)>0)),released=int(z['TWO_SIDED_action'].sum()))
            E['eligibility'].append(row);add(7,'Center eligibility',row)
            add(1,'Nine-direction overview',dict(arm=arm,MAE=E['primary'][arm]['complete']['MA']['mae'],all_report=E['reporting'][arm]['ALL_REPORT'],primary=E['reporting'][arm]['BOUNDARY95']))
        common=(c.accounting.states(z['prediction'])==0)&np.isfinite(z['LR_score'])&np.isfinite(z['DUAL_score']);la=[];da=[]
        for i in range(20):
            ids=np.flatnonzero(common[i]);n=int(.2*len(ids))
            for k,store in [('LR',la),('DUAL',da)]:
                a=np.zeros(len(y),bool);order=sorted(ids,key=lambda j:(z[k+'_score'][i,j],str(d['ids'][idx[j]])));a[order[:n]]=True;store.append(a)
        la=np.array(la);da=np.array(da);ab=c.accounting.states(y)!=0
        m=dict(candidates=int(common.sum()),released=int(la.sum()),LR_errors=int((la&ab).sum()),DUAL_errors=int((da&ab).sum()),delta=ratio((da&ab).astype(int)-(la&ab),la,gg))
        E['matched'][arm]=m;add(3,'Matched coverage',dict(arm=arm,scope='common predicted-normal finite-score candidates, floor20percent per seed',**m))
    target=analysis.load(ROOT/'gates/target_only_SA_T900.npz');idx=target['test'];y=d['y'][idx,2];gg=d['groups'][idx];E['target_contrasts']={}
    for arm in c.MAIN[1:]:
        z=analysis.load(ROOT/'gates'/f'{arm}_T900.npz');left=gate_action(target,'BOUNDARY95');right=gate_action(z,'BOUNDARY95');t=c.accounting.states(y)
        v={k:ratio(rr.astype(int)-ll,np.ones_like(rr),gg) for k,ll,rr in [('correct',left&(c.accounting.states(target['prediction'])==t),right&(c.accounting.states(z['prediction'])==t)),('incorrect',left&(c.accounting.states(target['prediction'])!=t),right&(c.accounting.states(z['prediction'])!=t))]}
        E['target_contrasts'][arm]=v;add(7,'Paired contrasts',dict(arm=arm,rule='BOUNDARY95',**v))

def sequence():
    d,*_=c.context();E['sequence']={};E['cumulative']={};E['sequence_contrasts']={};E['internal']=[]
    partition=c.read(c.WS/'outputs/uteg_reporting_increment_20261030/subset_mask.json')
    for arm in c.TARGET:
        z=analysis.load(ROOT/'sequence'/f'{arm}.npz');idx=z['test'];y=d['y'][idx,2];gg=d['groups'][idx];t=c.accounting.states(y);E['sequence'][arm]={}
        for rule in analysis.RULES:
            E['sequence'][arm][rule]={}
            for schedule in analysis.SCHEDULES:
                prefix=f'{rule}__{schedule}';p=z[prefix+'_prediction'];a=z[prefix+'_action'];when=z[prefix+'_time'];m=summary(p,y,a,gg)
                E['sequence'][arm][rule][schedule]=m;add(4,'Pointwise and cumulative',dict(arm=arm,rule=rule,schedule=schedule,**m));destinations(6,'Timing destinations',dict(arm=arm,rule=rule,schedule=schedule),p,y,a)
                for h in analysis.SCHEDULES[schedule]:add(4,'First-report times',dict(arm=arm,rule=rule,schedule=schedule,minutes=h//60,reported=int((when==h).sum()),correct=int(((when==h)&(c.accounting.states(p)==t)).sum())))
                for si,s in enumerate(c.SEEDS):add(5,'All sequence seeds',dict(arm=arm,rule=rule,schedule=schedule,seed=s,**check.score(p[si],y,a[si])))
        E['cumulative'][arm]={};p=z['BOUNDARY95__5-10-15_prediction'];when=z['BOUNDARY95__5-10-15_time'];qc=np.isfinite(p);ok=c.accounting.states(p)==t
        for h in c.HORIZONS:
            a=qc&(when<=h)&(when>=0);E['cumulative'][arm][str(h)]=summary(p,y,a,gg,3000)
            add(4,'Cumulative checkpoints',dict(arm=arm,rule='BOUNDARY95',schedule='5-10-15',minutes=h//60,**E['cumulative'][arm][str(h)]))
        if arm in c.MAIN:
            E['sequence_contrasts'][arm]={}
            for name,base_schedule,base_h in [('continued','5-10-15',300),('versus_fixed15','fixed15',900),('versus10-15','10-15',900)]:
                bp=z[f'BOUNDARY95__{base_schedule}_prediction'];ba=z[f'BOUNDARY95__{base_schedule}_action']&(z[f'BOUNDARY95__{base_schedule}_time']<=base_h)
                vals={}
                for k,good in [('correct',True),('incorrect',False)]:
                    num=(qc&(ok==good)).astype(int)-(ba&((c.accounting.states(bp)==t)==good));vals[k]=ratio(num,np.ones_like(num),gg,3000 if name=='continued' else 5000)
                E['sequence_contrasts'][arm][name]=vals;add(5,'Waiting contrasts',dict(arm=arm,contrast=name,**vals))
            for schedule in analysis.SCHEDULES:
                p=z[f'BOUNDARY95__{schedule}_prediction'];a=z[f'BOUNDARY95__{schedule}_action']
                for label,mask in [('full201',np.ones(len(idx),bool)),('selected60',np.isin(idx,partition['selected_global_indices'])),('complement141',np.isin(idx,partition['complement_global_indices']))]:
                    m=summary(p[:,mask],y[mask],a[:,mask],gg[mask],5000);row=dict(arm=arm,schedule=schedule,partition=label,records=int(mask.sum()),**m);E['internal'].append(row);add(7,'Internal_Event',row)
    E['sequence_target_contrasts']={};base=analysis.load(ROOT/'sequence/target_only_SA.npz');idx=base['test'];t=c.accounting.states(d['y'][idx,2]);gg=d['groups'][idx]
    for arm in c.MAIN[1:]:
        z=analysis.load(ROOT/'sequence'/f'{arm}.npz');v={}
        for schedule in analysis.SCHEDULES:
            pre=f'BOUNDARY95__{schedule}';v[schedule]={}
            for k,good in [('correct',True),('incorrect',False)]:
                ba=base[pre+'_action']&((c.accounting.states(base[pre+'_prediction'])==t)==good);sa=z[pre+'_action']&((c.accounting.states(z[pre+'_prediction'])==t)==good)
                v[schedule][k]=ratio(sa.astype(int)-ba,np.ones_like(sa),gg)
            add(7,'Sequence_Contrasts',dict(arm=arm,schedule=schedule,**v[schedule]))
        E['sequence_target_contrasts'][arm]=v

def representation():
    d,*_=c.context();E['representation']={};E['signal']=c.read(OUT/'experiments/signal/SUMMARY.json')
    for rep in ['EXP3','EXP4']:
        for arm in DIRECTIONS:
            z=analysis.load(REP/f'{rep}_{arm}.npz');idx=z['test'];y=d['y'][idx,2];gg=d['groups'][idx];E['representation'].setdefault(arm,{})[rep]={}
            for rule in ['ALL_REPORT','BOUNDARY95','LR','TWO_SIDED']:
                a=np.isfinite(z['prediction']) if rule=='ALL_REPORT' else z[rule+'_action'];m=summary(z['prediction'],y,a,gg)
                E['representation'][arm][rep][rule]=m;add(5,'Representation reporting',dict(arm=arm,representation=rep,rule=rule,**m))
            companion=analysis.load(REP/f'{rep}_{arm}_readouts.npz')
            add(5,'Representation mapping',dict(arm=arm,representation=rep,**readouts(companion['prediction'],d['medical'][idx],idx)))
    for rep,v in E['signal'].items():add(5,'Representation signal',dict(representation=rep,**v))
    E['representation_curves']={}
    for arm in DIRECTIONS:
        zs={r:analysis.load(REP/f'{r}_{arm}.npz') for r in ['EXP3','EXP4']};idx=zs['EXP3']['test'];y=d['y'][idx,2];gg=d['groups'][idx];tc=c.accounting.states(y)
        common=np.isfinite(zs['EXP3']['LR_score'])&np.isfinite(zs['EXP4']['LR_score']);curves=[]
        for percent in range(5,101,5):
            q=percent/100
            vals={}
            for rep,z in zs.items():
                a=np.zeros(common.shape,bool)
                for s in range(20):
                    inds=np.flatnonzero(common[s]);n=percent*len(inds)//100;order=sorted(inds,key=lambda i:(z['LR_score'][s,i],str(d['ids'][idx[i]])));a[s,order[:n]]=True
                correct=a&(c.accounting.states(z['prediction'])==tc);vals[rep]=dict(agreement=ratio(correct,a,gg,2000),coverage=float(a.mean()),correct=correct,action=a)
                add(5,'Representation matched',dict(arm=arm,quota=float(q),representation=rep,agreement=vals[rep]['agreement'],coverage=vals[rep]['coverage']))
            diff=paired_ratio_difference(vals['EXP3']['correct'],vals['EXP3']['action'],vals['EXP4']['correct'],vals['EXP4']['action'],gg,2000)
            add(5,'Representation matched delta',dict(arm=arm,quota=float(q),**diff));curves.append(dict(quota=float(q),EXP3={k:v for k,v in vals['EXP3'].items() if k not in ['correct','action']},EXP4={k:v for k,v in vals['EXP4'].items() if k not in ['correct','action']},delta=diff))
        E['representation_curves'][arm]=curves

def extras():
    dev=OUT/'experiments/development';E['development']=c.read(dev/'SUMMARY.json');E['preserved']=c.read(dev/'PRESERVED_TEST.json');E['diagnostics']={k:c.read(OUT/f'experiments/diagnostics/{k}.json') for k in ['COMPLETE','D2','D3','FAMILY']}
    for row in E['development']:
        name=row['candidate'];sheet='Adaptation strengths' if name.startswith('strength') or name in ['base','raw'] else 'Nested side protection' if name in ['BASE','LOW','HIGH','BOTH','side_selected'] else 'PLS SVR reuse dev';add(5,sheet,row)
    for name,sheet in [('PRESERVED_TEST','Preserved channel external'),('PRESERVED_DEV','Preserved channel dev'),('HEADROOM','Oracle headroom dev')]:
        for row in c.read(dev/f'{name}.json'):
            # Filter the exported summary only; retain the source evidence.
            exported = ({k:v for k,v in row.items() if k != 'valid_UTEG_only_validation'}
                        if sheet == 'Preserved channel external' else row)
            add(5,sheet,exported)
    for row in c.read(OUT/'experiments/diagnostics/CAP_POINTS.json'):add(7,'Cap points',row)
    for kind in ['D2','D3']:
        obj=E['diagnostics'][kind]
        for arm,bs in obj['point'].items():
            for b,v in bs.items():add(7,'EE matching' if kind=='D2' else 'D3 aggregate',dict(arm=arm,budget=b,**v))
        for arm,bs in obj['contrasts'].items():
            for b,v in bs.items():add(7,'EE contrasts' if kind=='D2' else 'D3 contrasts',dict(arm=arm,budget=b,**v))
    for arm,vals in E['diagnostics']['D2']['points'].items():
        for i,(cc,ee) in enumerate(zip(vals['correct'],vals['error'])):add(7,'Policy points',dict(arm=arm,index=i,correct_yield=cc,error_yield=ee))
    E['all301']=c.read(ROOT/'summary/METRICS.json')['results']
    for key,m in E['all301'].items():
        if key.startswith('all301|'):add(8,'Internal301 outcomes',dict(configuration=key,**m))

def enrich():
    d,*_=c.context();E['early']={};E['risk_curves']={};E['target_prediction']={}
    for arm in c.TARGET:
        E['early'][arm]={}
        for h in c.HORIZONS:
            z=analysis.load(ROOT/'gates'/f'{arm}_T{h}.npz');idx=z['test'];y=d['y'][idx,2];gg=d['groups'][idx];E['early'][arm][str(h)]={}
            for rule in analysis.RULES:
                a=gate_action(z,rule);m=summary(z['prediction'],y,a,gg);E['early'][arm][str(h)][rule]=m
                add(4,'Checkpoint reports',dict(arm=arm,minutes=h//60,rule=rule,**m))
        z=analysis.load(ROOT/'gates'/f'{arm}_T900.npz');idx=z['test'];y=d['y'][idx,2];gg=d['groups'][idx];p=c.accounting.states(z['prediction']);t=c.accounting.states(y)
        curves=[]
        for key in ['LR','HGB','BOUNDARY']:
            scores=z[key+'_score'];valid=np.isfinite(scores)&np.isfinite(z['prediction'])
            for percent in range(5,101,5):
                q=percent/100
                a=np.zeros(valid.shape,bool)
                for si in range(20):
                    ids=np.flatnonzero(valid[si]);order=sorted(ids,key=lambda i:(scores[si,i],str(d['ids'][idx[i]])));a[si,order[:percent*len(ids)//100]]=True
                v=dict(rule=key,quota=float(q),coverage=float(a.mean()),agreement=ratio(a&(p==t),a,gg,2000));curves.append(v);add(3,'Risk coverage curves',dict(arm=arm,**v))
        E['risk_curves'][arm]=curves
    z=analysis.load(AUX/'target_ridge_B60.npz');idx=z['test'];y=d['medical'][idx];gg=d['groups'][idx]
    median=np.array([np.full(len(idx),np.median(d['y'][c.paper_spec('target_only_SA',s)['pool'],2])) for s in c.SEEDS])
    for name,p in [('Target median',median),('Target Ridge',z['prediction'][:,:,2])]:
        m=summary(p,y[:,2],None,gg,5000);E['target_prediction'][name]=m;add(6,'Target prediction',dict(method=name,**m))
        destinations(6,'Prediction destinations',dict(method=name),p,y[:,2],np.isfinite(p))
    for arm in c.MAIN:
        p=analysis.load(ROOT/'primary'/f'{arm}_T900_B0.npz')['prediction'];m=summary(p,y[:,2],None,gg,5000);E['target_prediction'][arm]=m;add(6,'Target prediction',dict(method=arm,**m))
        add(6,'Prediction contrasts',dict(method=arm,contrast='current MA minus target Ridge MAE',**ratio(abs(p-y[:,2])-abs(z['prediction'][:,:,2]-y[:,2]),np.ones_like(p),gg)))
    single=analysis.load(AUX/'HZ_to_SA_B0.npz');dual=analysis.load(AUX/'HZ+HB_to_SA_B0.npz');E['source_contrasts']={}
    for stage in ['raw','complete']:
        v=ratio(abs(dual[stage][:,:,2]-y[:,2])-abs(single[stage][:,:,2]-y[:,2]),np.ones((20,len(idx))),gg)
        E['source_contrasts'][stage]=v;add(5,'Source contrasts',dict(stage=stage,contrast='dual minus single MAE',**v))
    z=analysis.load(OUT/'experiments/development/PRESERVED_TEST.npz');idx=z['test'];y=d['y'][idx,2];t=c.accounting.states(y);gg=d['groups'][idx];E['preserved_intervals']={}
    for key,good in [('correct',True),('incorrect',False)]:
        a=z['PRESERVED']&((c.accounting.states(z['prediction'])==t)==good);b=z['BASE']&((c.accounting.states(z['base_prediction'])==t)==good)
        v=ratio(a.astype(int)-b,np.ones_like(a),gg,10000);E['preserved_intervals'][key]=v;add(5,'Preserved channel intervals',dict(metric=key,**v))
    for row in E['eligibility']:add(1,'Evaluation denominators',dict(arm=row['arm'],records=row['records'],seed_count=20,events=row['records']*20,low_events=row['low_records']*20,normal_events=row['normal_records']*20,high_events=row['high_records']*20))

def runtime():
    d,*_=c.context();arm='HZ+HB_to_SA';sp=c.paper_spec(arm,1000);a=sp['pool'];b=sp['test'][:1];x=c.feature(a,b,900);src=c.source_model(arm,900)
    shadow=dict(d,y=np.full_like(d['y'],np.nan));shadow['y'][a]=d['y'][a];z,q,y,_,_=c.selected.models.design(shadow,[x],a,b,'target_only')
    base=(c.source_predict(src,x[a])-60)/10;w=c.selected.models.weights(y,0,'inward4');coef,_=c.selected.solve_offset(z,(y-60)/10,w,base)
    # Rebuild the local kernel basis by the same numerical definition, then time
    # only forward evaluation; fitting and loading are excluded.
    zz,basis=c.basis_fit(x[a]);np.testing.assert_allclose(zz@zz.T,z@z.T,rtol=1e-7,atol=1e-7)
    model=joblib.load(ROOT/'gates/models'/f'{arm}_T900_1000.joblib');p=analysis.load(ROOT/'primary'/f'{arm}_T900_B0.npz');pred=p['prediction'][0,:1];xx,med,features=gates.fields(b,p['query_b'][0,:1],pred,a,900);pc=c.accounting.states(pred)
    funcs={'Source mapping':lambda:c.source_predict(src,x[b]),'Local kernel basis':lambda:c.basis_query(basis,x[b]),'Local correction product':lambda:q@coef,'LR score':lambda:gates.fr.score_ma_gate(model['lr'],features,pc),'Dual risk heads':lambda:gates.fr.head_scores(model['heads'],xx,med),'Boundary comparison':lambda:-np.minimum(abs(pred-50),abs(pred-70))<=p['threshold'][0]}
    times=[]
    for name,fn in funcs.items():
        fn();vs=[]
        for _ in range(100):
            t=time.perf_counter_ns();fn();vs.append((time.perf_counter_ns()-t)/1e6)
        v=dict(component=name,repetitions=100,median_ms=float(np.median(vs)),p25_ms=float(np.quantile(vs,.25)),p75_ms=float(np.quantile(vs,.75)),scope='one case, warm CPU, excludes fitting/loading/acquisition')
        times.append(v);add(5,'Dual-risk timing' if name=='Dual risk heads' else 'Component timing',v)
    E['runtime']=times
    for rep in ['EXP3','EXP4']:
        v=E['signal'][rep];add(5,'Task timing',dict(task=rep,records=len(d['ids']),median_seconds=v['median_seconds'],scope='per-record signal fit, not acquisition time'))

def run():
    suite.initialize();OUT.mkdir(exist_ok=True)
    for fn in [cohort,predictions,reports,sequence,representation,extras,enrich,runtime]:fn();print('AGGREGATED',fn.__name__,flush=True)
    import sys
    from . import manuscript_detail_tables
    manuscript_detail_tables.run(sys.modules[__name__])
    version=dict(version='newB_20261103',data_sha256=c.sha(ROOT/'data/DATA15_CURRENT.npz'),evaluation='Books1-7 original center roles; Book8 local301 grouped internal validation.',
        evidence='The participating hospitals supplied U-TEG MA values and voltage recordings, together with the corresponding instrument-exported TEG readouts. The 5/10-minute inputs use the first 300/600 positions of the 900-point grid. Supplementary Methods A1 and the code package INPUT_CONTRACT.json describe input preparation.',
        privacy='Aggregate and seed-level statistics only; no patient or record-level data.',units='R/K minutes, MA mm, angle degrees; rates 0-1 unless stated.')
    for number,sheets in BOOKS.items():
        sheets['README']=[dict(field=k,value=v) for k,v in version.items()]
        sheets['Definitions']=[dict(metric=k,definition=v) for k,v in {'agreement':'correct reports / released reports; low/normal/high states','coverage':'released / all retained events','FNR':'abnormal reference released as normal / released abnormal references; all-report includes all valid records','FPR':'normal reference released as abnormal / released normal references','unreported':'not counted as correct; reported separately','seeds':'repeated evaluation of the same records, not independent patients','low':'MA <50','normal':'50 <= MA <=70','high':'MA >70'}.items()]
    gates.clean_write(OUT/'EVIDENCE.json',E);gates.clean_write(OUT/'WORKBOOK_TABLES.json',BOOKS);gates.clean_write(OUT/'AGGREGATION_COMPLETE.json',dict(books=8,sheets={k:len(v) for k,v in BOOKS.items()},version=version))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
