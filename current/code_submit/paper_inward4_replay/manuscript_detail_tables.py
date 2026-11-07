"""Complete aggregate electronic-table detail; never export record identifiers."""
import numpy as np
from . import core as c,analysis,current_finalize as check,manuscript_diagnostics as dg

def run(r):
    d,*_=c.context();add=r.add;dest=r.destinations
    final_controls(r)
    # Model/stage and run-level controls retained in the original electronic tables.
    for arm in r.DIRECTIONS:
        for method in r.suite.mapping.METHODS:
            z=analysis.load(r.AUX/f'six_{arm}_{method}.npz');ix=z['test'];y=d['medical'][ix]
            for phase in ['raw','huber','complete']:
                if phase!='complete':add(2,'Mapping baselines',dict(arm=arm,method=method,phase=phase,**r.readouts(z[phase],y,ix)))
                for si,s in enumerate(c.SEEDS):add(5,'All mapping seeds',dict(arm=arm,method=method,phase=phase,seed=s,**check.readouts(z[phase][si],y)))
        z=analysis.load(r.ROOT/'calibration'/f'{arm}.npz');ix=z['test'];y=d['medical'][ix]
        for phase in ['raw','affine','joint','huber','complete']:
            pp=analysis.load(r.AUX/f'{arm}_B0.npz')['complete'] if phase=='complete' else z[phase]
            for si,s in enumerate(c.SEEDS):add(2,'Adaptation comparator seeds',dict(arm=arm,method=phase,seed=s,**check.readouts(pp[si],y)))
    for b in [20,40,60]:
        for arm in ['target_Ridge',*c.MAIN[1:]]:
            z=analysis.load(r.AUX/(f'target_ridge_B{b}.npz' if arm=='target_Ridge' else f'{arm}_B{0 if b==60 else b}.npz'));p=z['prediction'] if arm=='target_Ridge' else z['complete'];ix=z['test'];y=d['medical'][ix]
            dest(7,'Budget_ClassDest',dict(arm=arm,budget=b),p[:,:,2],y[:,2],np.isfinite(p[:,:,2]))
            for si,s in enumerate(c.SEEDS):add(7,'Budget_Seed',dict(arm=arm,budget=b,seed=s,**check.readouts(p[si],y)))
    for arm in c.MAIN:
        z=analysis.load(r.ROOT/'gates'/f'{arm}_T900.npz');ix=z['test'];y=d['y'][ix,2];a=z['boundary95_action']
        add(7,'Working points',dict(arm=arm,**check.score(z['prediction'],y,a)))
        dest(7,'Class destinations',dict(arm=arm),z['prediction'],y,a)
        for si,s in enumerate(c.SEEDS):add(7,'All seeds',dict(arm=arm,seed=s,**check.score(z['prediction'][si],y,a[si])))
        p=z['prediction'];t=c.accounting.states(y);correct=c.accounting.states(p)==t;valid=np.isfinite(p);gg=d['groups'][ix]
        for rule in analysis.RULES:
            action=r.gate_action(z,rule)
            for key,left,right in [('coverage',action,valid),('correct_yield',action&correct,valid&correct),('incorrect_yield',action&~correct,valid&~correct)]:
                v=r.ratio(left.astype(int)-right,np.ones_like(left),gg,2000)
                add(5,'Reporting contrasts',dict(arm=arm,rule=rule,metric=key,contrast='rule minus all report',**v))
            v=r.paired_ratio_difference(action&correct,action,valid&correct,valid,gg,2000)
            add(5,'Reporting contrasts',dict(arm=arm,rule=rule,metric='agreement',contrast='rule minus all report',**v))
    # Each true representation has its own fitted models and complete destinations.
    for arm in r.DIRECTIONS:
        pair={rep:analysis.load(r.REP/f'{rep}_{arm}.npz') for rep in ['EXP3','EXP4']};ix=pair['EXP3']['test'];y=d['y'][ix,2];gg=d['groups'][ix]
        for rule in analysis.RULES:
            vals={}
            for rep,z in pair.items():
                a=r.gate_action(z,rule);p=z['prediction'];vals[rep]=(p,a)
                dest(5,'Representation destinations',dict(arm=arm,representation=rep,rule=rule),p,y,a)
                for si,s in enumerate(c.SEEDS):add(5,'Representation seeds',dict(arm=arm,representation=rep,rule=rule,seed=s,**check.score(p[si],y,a[si])))
            p,a=vals['EXP3'];q,b=vals['EXP4'];t=c.accounting.states(y)
            v=r.paired_ratio_difference(a&(c.accounting.states(p)==t),a,b&(c.accounting.states(q)==t),b,gg)
            add(5,'Representation reporting delta',dict(arm=arm,rule=rule,contrast='EXP3 minus EXP4 agreement',**v))
        x=analysis.load(r.REP/f'EXP3_{arm}_readouts.npz');z=analysis.load(r.REP/f'EXP4_{arm}_readouts.npz')
        for j,name in enumerate(r.READOUTS):
            num=abs(x['prediction'][:,:,j]-d['medical'][ix,j])-abs(z['prediction'][:,:,j]-d['medical'][ix,j])
            add(5,'Representation mapping delta',dict(arm=arm,readout=name,**r.ratio(num,np.ones_like(num),gg)))
    # Fixed internal partitions, full timings and six gate controls.
    partition=c.read(c.WS/'outputs/uteg_reporting_increment_20261030/subset_mask.json')
    for arm in c.MAIN:
        z=analysis.load(r.ROOT/'sequence'/f'{arm}.npz');ix=z['test'];y=d['y'][ix,2]
        for schedule in analysis.SCHEDULES:
            prefix=f'BOUNDARY95__{schedule}';p=z[prefix+'_prediction'];a=z[prefix+'_action'];when=z[prefix+'_time']
            for label,mask in [('full201',np.ones(len(ix),bool)),('selected60',np.isin(ix,partition['selected_global_indices'])),('complement141',np.isin(ix,partition['complement_global_indices']))]:
                meta=dict(arm=arm,schedule=schedule,partition=label)
                dest(7,'Internal_ClassDest',meta,p[:,mask],y[mask],a[:,mask])
                for si,s in enumerate(c.SEEDS):add(7,'Internal_Seed',dict(**meta,seed=s,**check.score(p[si,mask],y[mask],a[si,mask])))
                for h in analysis.SCHEDULES[schedule]:add(7,'Internal_FirstReport',dict(**meta,minutes=h//60,reported=int((when[:,mask]==h).sum()),correct=int(((when[:,mask]==h)&(c.accounting.states(p[:,mask])==c.accounting.states(y[mask]))).sum())))
    for row in c.read(r.OUT/'experiments/forward/SUMMARY.json')['rows']:add(7,'Internal_Forward',row)
    add(7,'Internal_Scope',dict(scope='Original201 and fixed60/141 internal partitions; deterministic recipe recomputed, saved gates forward-scored; not independent validation.',contexts=180,input_lineage='original U-TEG B values; measured 5- and 10-minute signals; complete 15-minute recordings'))
    for row in r.E['cohort']:add(7,'Internal_QC',row)
    # Existing label-aware policy families: class and seed accounting, including mixtures.
    family=c.read(dg.OUT/'FAMILY.json');fold_counts=None
    for arm in c.MAIN:
        z=analysis.load(dg.OUT/f'private_{arm}.npz');ix=z['test'];y=d['y'][ix,2];truth=c.accounting.states(y);pc=z['pred_class'];gi=z['group_index'];names=sorted(set(d['groups'][ix].astype(str)));fold=dg.folds(names);nf=len(names)
        actions=z['actions'];cg=z['correct_group'];eg=z['error_group'];n=np.bincount(gi,minlength=nf)
        for f in range(5):
            cal=fold!=f;hold=fold==f;cn=20*n[cal].sum();hn=20*n[hold].sum();cc=cg[:,cal].sum(1)/cn;ev=eg[:,cal].sum(1)/cn;hc=cg[:,hold].sum(1)/hn;he=eg[:,hold].sum(1)/hn
            for budget in dg.BUDGETS:
                mix=dg.ee.budget_mix(ev,cc,budget);i,j,t=mix['strategy_i'],mix['strategy_j'],mix['weight_j'];a=(1-t)*actions[i]+t*actions[j];mask=hold[gi]
                add(7,'D3 folds',dict(arm=arm,fold=f,budget=budget,calibration_records=int(cn/20),heldout_records=int(hn/20),**mix,heldout_correct_yield=(1-t)*hc[i]+t*hc[j],heldout_error_yield=(1-t)*he[i]+t*he[j]))
                for tc in [-1,0,1]:
                    sel=mask&(truth==tc);den=int(sel.sum()*20)
                    for target in [-1,0,1,9]:
                        count=float((1-a[:,sel]).sum()) if target==9 else float((a[:,sel]*(pc[:,sel]==target)).sum())
                        add(7,'D3 class',dict(arm=arm,fold=f,budget=budget,true_state=tc,report_state=target,count=count,denominator=den))
        for budget,branches in r.E['diagnostics']['D2']['point'][arm].items():
            for kind,mix in branches.items():
                if mix is None:continue
                i,j,t=mix['strategy_i'],mix['strategy_j'],mix['weight_j'];a=(1-t)*actions[i]+t*actions[j]
                for si,s in enumerate(c.SEEDS):add(7,'EE seeds',dict(arm=arm,budget=float(budget),kind=kind,seed=s,correct=float((a[si]*(pc[si]==truth)).sum()),incorrect=float((a[si]*(pc[si]!=truth)).sum()),denominator=len(ix)))
                for tc in [-1,0,1]:
                    sel=truth==tc
                    for target in [-1,0,1,9]:
                        count=float((1-a[:,sel]).sum()) if target==9 else float((a[:,sel]*(pc[:,sel]==target)).sum())
                        add(7,'EE class',dict(arm=arm,budget=float(budget),kind=kind,true_state=tc,report_state=target,count=count,denominator=int(sel.sum()*20)))
    print('DETAIL_TABLES_COMPLETE',flush=True)

def final_controls(r):
    """Finish matched-ranking and internal representation tables from saved fits."""
    d,*_=c.context()
    for rep in ['EXP3','EXP4']:
        for arm in c.MAIN:
            z=analysis.load(r.REP/f'{rep}_all301_{arm}.npz');ix=z['local'];y=d['y'][ix,2]
            for rule in analysis.RULES:
                a=r.gate_action(z,rule)
                r.add(8,'Refitted representation',dict(arm=arm,representation=rep,rule=rule,**check.score(z['prediction'],y,a)))
                r.destinations(8,'Representation destinations',dict(arm=arm,representation=rep,rule=rule),z['prediction'][None,:],y,a[None,:])
    z=analysis.load(r.AUX/'target_ridge_all301.npz');ix=z['local'];y=d['medical'][ix]
    r.add(8,'Four-output target Ridge',dict(method='Four-input Ridge with inner alpha selection',**check.readouts(z['prediction'],y),**check.score(z['prediction'][:,2],y[:,2],np.ones(len(ix),bool))))
    r.destinations(8,'Ridge destinations',dict(method='Target Ridge'),z['prediction'][:,2][None,:],y[:,2],np.ones((1,len(ix)),bool))
    for arm in c.TARGET:
        z=analysis.load(r.ROOT/'gates'/f'{arm}_T900.npz');ix=z['test'];p=c.accounting.states(z['prediction']);t=c.accounting.states(d['y'][ix,2]);groups=d['groups'][ix]
        scores={k:z[k+'_score'] for k in ['LR','HGB','BOUNDARY']}
        scores['TWO_SIDED']=np.where(p==0,z['DUAL_score'],z['LR_score'])
        for other in ['HGB','BOUNDARY','TWO_SIDED']:
            common=np.isfinite(z['prediction'])&np.isfinite(scores['LR'])&np.isfinite(scores[other])
            actions=[]
            for key in ['LR',other]:
                a=np.zeros(common.shape,bool)
                for si in range(len(c.SEEDS)):
                    ids=np.flatnonzero(common[si]);ordered=sorted(ids,key=lambda j:(scores[key][si,j],str(d['ids'][ix[j]])))
                    a[si,ordered[:int(.2*len(ids))]]=True
                actions.append(a)
            a,b=actions
            v=r.paired_ratio_difference(b&(p==t),b,a&(p==t),a,groups,2000)
            r.add(3,'All-candidate matched20',dict(arm=arm,comparison=other+' minus LR',scope='Common finite scores, all predicted classes; floor20percent per seed; descriptive test ranking',candidates=int(common.sum()),selected_each=int(a.sum()),LR_correct=int((a&(p==t)).sum()),other_correct=int((b&(p==t)).sum()),**v))
