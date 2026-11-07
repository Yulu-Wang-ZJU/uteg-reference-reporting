"""Current-version exploratory D2/D3, with explicit label-aware selection."""
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,analysis,gates
from code_submit.paper_b085_hospital_replay import equal_error as ee

OUT=suite.ROOT/'manuscript_sync/experiments/diagnostics'
BUDGETS=[.005,.01,.02,.05,.075,.10]

def interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(low=float(np.quantile(a,.025)) if len(a) else None,high=float(np.quantile(a,.975)) if len(a) else None,n=len(a))

def freeze():
    path=OUT/'FAMILY.json'
    if path.exists():return c.read(path)
    candidates={};grid={0.,.05,.10,.15,.20}
    for arm in c.MAIN:
        dev=c.read(suite.ROOT/'gates'/f'{arm}_T900_B0.development.json') if (suite.ROOT/'gates'/f'{arm}_T900_B0.development.json').exists() else c.read(suite.ROOT/'gates'/f'{arm}_T900.development.json')
        candidates[arm]=[]
        for rows in dev:
            cs=ee.candidate_rows(rows['rows']);candidates[arm].append(cs)
            grid.update(round(q['error_rate'],14) for q in cs if q['error_rate']<=.20+1e-12)
    grid=sorted(grid);configs={};caprows=[]
    for arm in c.MAIN:
        configs[arm]=[dict(first_cap=None,last_cap=None,selections=[dict(normal=None,tail=None)]*20)]
        signatures={tuple((None,None) for _ in range(20)):0}
        for cap in grid:
            sels=[ee.select_at_cap(cs,cap) for cs in candidates[arm]]
            signature=tuple((r['normal'],r['tail']) for r in sels)
            if signature not in signatures:
                signatures[signature]=len(configs[arm]);configs[arm].append(dict(first_cap=cap,last_cap=cap,selections=sels))
            elif signatures[signature]>0:configs[arm][signatures[signature]]['last_cap']=cap
            if cap in [0.,.05,.10,.15,.20]:caprows.append(dict(arm=arm,cap=cap,index=signatures[signature]))
    result=dict(grid=grid,configs=configs,caprows=caprows,development_only_family=True,
        analysis='D2 uses all evaluation labels; D3 uses four folds of additional evaluation labels; neither is untouched validation.',
        bootstrap_D2=5000,bootstrap_D3=1000,B_from_original_data=True)
    gates.clean_write(path,result);return result

def arrays(family):
    d,*_=c.context();ma={};idx=None;caprows=[]
    for arm in c.MAIN:
        z=analysis.load(suite.ROOT/'gates'/f'{arm}_T900.npz');test=z['test']
        if idx is not None:np.testing.assert_array_equal(test,idx)
        idx=test;groups=d['groups'][idx].astype(str);names=sorted(set(groups));ng=len(names)
        gi=np.array([names.index(v) for v in groups]);truth=c.accounting.states(d['y'][idx,2]);pred=c.accounting.states(z['prediction'])
        acts=[]
        for cfg in family['configs'][arm]:
            act=np.zeros_like(pred,bool)
            for s,sel in enumerate(cfg['selections']):
                n,t=sel['normal'],sel['tail'];ds=z['DUAL_score'][s];bs=z['LR_score'][s]
                if n is not None:act[s]|=(pred[s]==0)&np.isfinite(ds)&(ds<=n)
                if t is not None:act[s]|=(pred[s]!=0)&np.isfinite(bs)&(bs<=t)
                act[s]&=np.isfinite(z['prediction'][s])
            acts.append(act)
        acts=np.array(acts);ok=acts&(pred[None,:,:]==truth[None,None,:]);err=acts&~ok
        cg=np.array([ok[:,:,gi==g].sum((1,2)) for g in range(ng)]).T
        eg=np.array([err[:,:,gi==g].sum((1,2)) for g in range(ng)]).T
        ma[arm]=dict(c=cg,e=eg,n=np.bincount(gi),groups=names)
        np.savez_compressed(OUT/f'private_{arm}.npz',actions=acts,pred_class=pred,test=idx,group_index=gi,correct_group=cg,error_group=eg)
        for row in family['caprows']:
            if row['arm']!=arm:continue
            j=row['index'];m=c.summarize(z['prediction'],d['y'][idx,2],acts[j]);caprows.append(dict(**row,**m))
    gates.clean_write(OUT/'CAP_POINTS.json',caprows)
    return ma

def vectors(m,w):
    n=20*np.dot(m['n'],w)
    return m['c']@w/n,m['e']@w/n,n

def matched(m,w):
    cc,eev,_=vectors(m,w)
    return {str(b):{k:func(eev,cc,b) for k,func in [('exact',ee.exact_mix),('at_most',ee.budget_mix),('deterministic',ee.deterministic_best)]} for b in BUDGETS}

def contrast(mt,ms,key):
    return ms[key]['correct_yield']-mt[key]['correct_yield'] if mt[key] is not None and ms[key] is not None else np.nan

def d2(ma):
    path=OUT/'D2.json'
    if path.exists():return
    ng=len(ma[c.MAIN[0]]['n']);point={a:matched(m,np.ones(ng)) for a,m in ma.items()}
    vals={a:{str(b):{k:[] for k in ['exact','at_most']} for b in BUDGETS} for a in c.MAIN[1:]}
    rng=np.random.default_rng(20261030)
    for i in range(5000):
        w=rng.multinomial(ng,np.ones(ng)/ng);res={a:matched(m,w) for a,m in ma.items()}
        for a in vals:
            for b in vals[a]:
                for k in vals[a][b]:vals[a][b][k].append(contrast(res[c.MAIN[0]][b],res[a][b],k))
        if (i+1)%1000==0:print('D2_BOOTSTRAP',i+1,flush=True)
    contrasts={a:{b:{k:dict(delta=contrast(point[c.MAIN[0]][b],point[a][b],k),**interval(v)) for k,v in fields.items()} for b,fields in budgets.items()} for a,budgets in vals.items()}
    points={a:dict(correct=vectors(m,np.ones(ng))[0],error=vectors(m,np.ones(ng))[1]) for a,m in ma.items()}
    gates.clean_write(path,dict(point=point,contrasts=contrasts,points=points,bootstraps=5000,selection_repeated_each_bootstrap=True))

def folds(names):
    order=sorted(names,key=lambda g:hashlib.sha256(('uteg-equal-error-20261030|'+g).encode()).hexdigest())
    return np.array([order.index(g)%5 for g in names])

def cross(ma,w,fold):
    result={a:{str(b):dict(correct=0.,error=0.,denominator=0.) for b in BUDGETS} for a in ma}
    for f in range(5):
        cal=w*(fold!=f);hold=w*(fold==f)
        for a,m in ma.items():
            if np.dot(m['n'],cal)==0 or np.dot(m['n'],hold)==0:continue
            cc,ev,_=vectors(m,cal);hc,he,den=vectors(m,hold)
            for b in BUDGETS:
                mix=ee.budget_mix(ev,cc,b);i,j,t=mix['strategy_i'],mix['strategy_j'],mix['weight_j'];v=result[a][str(b)]
                v['correct']+=((1-t)*hc[i]+t*hc[j])*den;v['error']+=((1-t)*he[i]+t*he[j])*den;v['denominator']+=den
    for a in result:
        for v in result[a].values():v.update(correct_yield=v['correct']/v['denominator'],error_yield=v['error']/v['denominator'])
    return result

def d3(ma):
    path=OUT/'D3.json'
    if path.exists():return
    first=ma[c.MAIN[0]];ng=len(first['n']);fold=folds(first['groups']);point=cross(ma,np.ones(ng),fold)
    vals={a:{str(b):dict(correct=[],error=[]) for b in BUDGETS} for a in c.MAIN[1:]}
    rng=np.random.default_rng(20261031)
    for i in range(1000):
        w=rng.multinomial(ng,np.ones(ng)/ng);res=cross(ma,w,fold)
        for a in vals:
            for b in vals[a]:
                for k in ['correct','error']:vals[a][b][k].append(res[a][b][k+'_yield']-res[c.MAIN[0]][b][k+'_yield'])
        if (i+1)%250==0:print('D3_BOOTSTRAP',i+1,flush=True)
    contrasts={a:{b:{k:dict(delta=point[a][b][k+'_yield']-point[c.MAIN[0]][b][k+'_yield'],**interval(v)) for k,v in fields.items()} for b,fields in bs.items()} for a,bs in vals.items()}
    gates.clean_write(path,dict(point=point,contrasts=contrasts,fold_group_counts=np.bincount(fold),bootstraps=1000,calibration_selection_repeated=True))

def smoke():
    e=np.array([0.,.1,.2]);p=np.array([0.,.6,.5]);m=ee.exact_mix(e,p,.15)
    assert abs(m['correct_yield']-.55)<1e-12
    assert abs(ee.budget_mix(e,p,.15)['correct_yield']-.6)<1e-12
    assert ee.exact_mix(e,p,.21) is None
    gates.clean_write(OUT/'SMOKE.json',dict(descending_frontier_preserved=True,unreachable_exact_not_extrapolated=True,all_withhold_always_feasible=True))

def run():
    suite.initialize();OUT.mkdir(parents=True,exist_ok=True);smoke();family=freeze();ma=arrays(family);d2(ma);d3(ma)
    gates.clean_write(OUT/'COMPLETE.json',dict(family_counts={a:len(v) for a,v in family['configs'].items()},grid_count=len(family['grid']),D2=5000,D3=1000,primary_predictions_and_actions_unchanged=True))
    print('EXPLORATORY_DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
