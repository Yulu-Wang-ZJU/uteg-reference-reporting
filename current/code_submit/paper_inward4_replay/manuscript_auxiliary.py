"""Four-readout companions and original nested target-Ridge controls.

Primary MA arrays are copied exactly. Auxiliary R/K/angle use joint Huber;
this companion is explicitly named, never silently substituted into sealed
MA-only experiment files.
"""
import argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,mapping,analysis
from .gates import clean_write

OUT=suite.ROOT/'manuscript_sync/experiments/auxiliary'
ALPHAS=[.1,1.,10.,100.]


def ridge_fit(d,train,alpha):
    x=d['x'][train];y=d['medical'][train];xm=x.mean(0);xs=np.maximum(x.std(0),1e-6)
    ym=y.mean(0);ys=np.maximum(y.std(0),1e-6);z=(x-xm)/xs
    coef=np.linalg.solve(z.T@z+alpha*np.eye(4),z.T@((y-ym)/ys))
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,coef=coef)


def ridge_predict(m,x):
    p=m['ym']+m['ys']*((x-m['xm'])/m['xs']@m['coef'])
    return np.column_stack([np.maximum(p[:,:3],0),np.clip(p[:,3],0,89.999999)])


def choose(d,train,inner):
    vals=[]
    for alpha in ALPHAS:
        err=[]
        for f in inner:
            a=np.asarray(f['T']);b=np.asarray(f['V']);p=ridge_predict(ridge_fit(d,a,alpha),d['x'][b])
            err.extend(abs(p[:,2]-d['medical'][b,2]))
        vals.append(float(np.mean(err)))
    j=min(range(len(ALPHAS)),key=lambda i:(vals[i],-ALPHAS[i]))
    return ALPHAS[j],vals


def run():
    suite.initialize();m=suite.numerical_models();d,_,_,_,_,folds,local=c.context();OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(primary_MA='Exact saved inward4 B-only primary, unchanged.',auxiliary_readouts='R/K/angle use joint Huber calibrated current-four-input Ridge source predictions.',
        comparison='Six-model complete4 companion uses each saved Huber output with its saved completeMA; sealed raw complete arrays remain unchanged.',
        direct_ridge='Train-only standardization; penalties .1,1,10,100 selected by original inner-fold MA error, ties larger; fixed pool/test roles.',
        budgets=[20,40,60],B_from_original_data=True,data_hash=c.sha(suite.ROOT/'data/DATA15_CURRENT.npz'))
    p=OUT/'PROTOCOL.json'
    if p.exists():assert c.read(p)==protocol
    else:clean_write(p,protocol)
    # Smoke: output transforms and selection cannot see poisoned test labels.
    sp=c.paper_spec('target_only_SA',1000);a=sp['pool'];b=sp['test'];inner=sp['positions'][3]['inner']
    pick=choose(d,a,inner);pred=ridge_predict(ridge_fit(d,a,pick[0]),d['x'][b]);old=d['medical'].copy()
    try:
        d['medical'][b]=1e6
        assert choose(d,a,inner)==pick
        np.testing.assert_array_equal(ridge_predict(ridge_fit(d,a,pick[0]),d['x'][b]),pred)
    finally:d['medical'][:]=old
    clean_write(OUT/'SMOKE.json',dict(test_label_poison_no_change=True,penalty_selection_train_only=True))
    for budget in [20,40,60]:
        path=OUT/f'target_ridge_B{budget}.npz'
        if path.exists():continue
        pp=[];choices=[]
        for seed in c.SEEDS:
            sp=c.paper_spec('target_only_SA',seed,0 if budget==60 else budget)
            alpha,errs=choose(d,sp['pool'],sp['positions'][3]['inner'])
            pp.append(ridge_predict(ridge_fit(d,sp['pool'],alpha),d['x'][sp['test']]))
            choices.append(dict(seed=seed,alpha=alpha,inner_mae=errs,train=sp['pool']))
        np.savez_compressed(path,prediction=pp,test=sp['test'],seed=c.SEEDS)
        clean_write(path.with_suffix('.audit.json'),choices)
        print('TARGET_RIDGE_BUDGET',budget,flush=True)
    path=OUT/'target_ridge_all301.npz'
    if not path.exists():
        pred=np.full((len(local),4),np.nan);where={int(v):i for i,v in enumerate(local)};audit=[]
        for fi,f in enumerate(folds):
            alpha,errs=choose(d,f['A'],f['inner']);ix=[where[i] for i in f['B']]
            pred[ix]=ridge_predict(ridge_fit(d,f['A'],alpha),d['x'][f['B']]);audit.append(dict(fold=fi,alpha=alpha,inner_mae=errs))
        np.savez_compressed(path,prediction=pred,local=local);clean_write(path.with_suffix('.audit.json'),audit)
    for arm in c.TARGET:
        if arm.startswith('target_only'):continue
        for budget in ([0,20,40] if c.TARGET[arm]=='SAHZU' else [0,20]):
            path=OUT/f'{arm}_B{budget}.npz'
            if path.exists():continue
            z=analysis.load(suite.ROOT/'primary'/f'{arm}_T900_B{budget}.npz');aux=mapping.source(arm,'RIDGE-10',1000)
            raw=[];huber=[];complete=[];audits=[]
            for si,seed in enumerate(c.SEEDS):
                sp=c.paper_spec(arm,seed,budget);a=sp['pool'];b=sp['test'];valid=c.qc_at(900)[b];qq=b[valid]
                pt=m.predict(aux,d['x'][a]);pq=m.predict(aux,d['x'][qq]);source=c.source_model(arm,900)
                pt[:,2]=c.source_predict(source,d['x'][a,1:2]);pq[:,2]=z['source_raw'][si,valid]
                cal=m.fit_cal(pt,d['medical'][a],aux['ym'],aux['ys']);assert cal['finite']
                hp=m.project(m.predict_cal(cal,pq));cp=hp.copy();cp[:,2]=z['prediction'][si,valid]
                for val,dest in [(pq,raw),(hp,huber),(cp,complete)]:
                    full=np.full((len(b),4),np.nan);full[valid]=val;dest.append(full)
                audits.append(dict(seed=seed,calibration=cal))
            np.testing.assert_array_equal(np.asarray(complete)[:,:,2],z['prediction'])
            np.savez_compressed(path,raw=raw,huber=huber,complete=complete,test=b,seed=c.SEEDS)
            clean_write(path.with_suffix('.audit.json'),audits)
            print('FOUR_READOUT_COMPANION',arm,budget,flush=True)
    for path in (suite.ROOT/'mapping').glob('*.npz'):
        dest=OUT/f'six_{path.stem}.npz'
        if dest.exists():continue
        z=analysis.load(path);p=z['huber'].copy();p[:,:,2]=z['complete'][:,:,2]
        np.savez_compressed(dest,raw=z['raw'],huber=z['huber'],complete=p,test=z['test'],seed=z['seed'])
    clean_write(OUT/'COMPLETE.json',dict(primary_readout_bundles=21,six_mapping_bundles=54,target_ridge_budgets=3,target_ridge_all301=1,
        primaryMA_exact_unchanged=True,auxiliary_protocol_explicit=True))
    print('AUXILIARY_COMPLETE',flush=True)

def representation_companions():
    from . import manuscript_representation as repmod
    for rep in ['EXP3','EXP4']:
        kind=repmod.activate(rep);m=suite.numerical_models();d,*_=c.context();x=c.feature([],[],900,kind)
        # EXP3's fourth slot is the constant mu=1. Standardization makes it
        # identically zero; the source Ridge is exactly a three-input fit.
        auxiliary_x=c.feature([],[],900,'four')
        for arm in c.TARGET:
            if arm.startswith('target_only'):continue
            path=repmod.OUT/f'{rep}_{arm}_readouts.npz'
            if path.exists():continue
            rows=c.source_rows(arm,900);labels=np.full_like(d['medical'],np.nan);labels[rows]=d['medical'][rows]
            src=m.fit_source(dict(d,x=auxiliary_x,medical=labels),rows,[],'RIDGE-10',1000);assert src['success']
            primary=c.source_model(arm,900,representation=kind);saved=analysis.load(repmod.OUT/f'{rep}_{arm}.npz');result=[]
            for si,seed in enumerate(c.SEEDS):
                sp=c.paper_spec(arm,seed);a=sp['pool'];b=sp['test'];valid=c.qc_at(900)[b];qq=b[valid]
                pt=m.predict(src,auxiliary_x[a]);pq=m.predict(src,auxiliary_x[qq]);pt[:,2]=c.source_predict(primary,x[a]);pq[:,2]=c.source_predict(primary,x[qq])
                cal=m.fit_cal(pt,d['medical'][a],src['ym'],src['ys']);assert cal['finite'];p=m.project(m.predict_cal(cal,pq));p[:,2]=saved['prediction'][si,valid]
                full=np.full((len(b),4),np.nan);full[valid]=p;result.append(full)
            np.savez_compressed(path,prediction=result,test=b,seed=c.SEEDS)
            print('REPRESENTATION_READOUTS',rep,arm,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',nargs='?',default='main',choices=['main','representation']);a=ap.parse_args()
    with threadpool_limits(limits=1):
        if a.stage=='main':run()
        else:representation_companions()
