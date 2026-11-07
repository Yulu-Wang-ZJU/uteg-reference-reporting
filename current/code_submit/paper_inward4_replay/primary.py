"""Full paper splits plus separate301 folds; new primary gate is boundary95."""
import argparse
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c

def initialize():
    c.OUT.mkdir(parents=True,exist_ok=True)
    d,prefix,peak,plan,role,folds,local=c.context()
    path=c.OUT/'PROTOCOL.json'
    hashes={str(p):c.sha(p) for p in [c.BASE/'DATA15.npz',c.BASE/'PREFIX_X.npz',c.BASE/'ROLE_FREEZE.json',c.SPLIT,
        c.selected.OUT/'private/PREDICTIONS.npz',c.selected.OUT/'FROZEN_PROTOCOL.json',c.selected.previous.RAW]}
    protocol=dict(status='FROZEN_BEFORE_NEW_EVALUATION',primary='B-only RBF source prior plus local RBF correction; final-MA inward4,alpha .01,class_inverse cap20',
        primary_gate='Original selected inner OOF95th-percentile boundary distance',
        data='Read original B values at 900 samples and saved coefficients fitted to measured 300/600-sample signals.',
        protocols={'all301':'Existing5outer/3inner grouped splits, all301 each once,15min exact selected baseline replay',
                   'paper':'Original roles and20seeds,60 target pool/201 test atSA; frozen external-center roles unchanged'},
        experiments=['primary3checkpoints9transferdirections+3target_only','six original source mappers with common inward4 local correction',
            'raw/independent-affine/joint-linear/Huber/full-MA adaptation','20/40/full local budgets+median',
            'B/3parameter/4parameter and uniform/class/inward4 controls','all-report/boundary95/LR/HGB/two-sided gates',
            '5-10-15/10-15/fixed15 first-report lock','matched20/risk-curves/retrospective-error-matching/extra-label-cross-calibration',
            'signal-fit and unchanged auxiliary readout historical audit'],
        source_rows=plan['source_rows'],input_hashes=hashes,B_from_original_data=True,valid_UTEG_only_validation=False,
        manuscript_changed=False,independent_predictive_goal_achieved=False,
        scope='New MA-specific branch; original R,K,angle and signal-fit evidence remain historical, explicitly not reattributed to the new B-only MA model.')
    if path.exists():assert c.read(path)==protocol
    else:c.write(path,protocol)
    return protocol

def paper(direction,h,budget=0):
    path=c.OUT/'primary'/f'{direction}_T{h}_B{budget}.npz'
    if path.exists():return
    d,*_=c.context();pp=[];aa=[];ii=[];ib=[];qb=[];thresholds=[];audits=[];source_raw=[];train_order=None
    for seed in c.SEEDS:
        spec=c.paper_spec(direction,seed,budget);p=spec['positions'][3]
        out=c.evaluate(spec['pool'],spec['test'],p['inner'],direction,h)
        if train_order is None:train_order=out['train']
        # Pools vary at C20/C40: do not collapse training row identities.
        pp.append(out['prediction']);aa.append(out['accepted']);ii.append(out['inner']);ib.append(out['inner_b']);qb.append(out['query_b']);thresholds.append(out['threshold'])
        audits.append(dict(seed=seed,train=out['train'].tolist(),fits=out['audits']))
        model=c.source_model(direction,h);x=c.feature(out['train'],out['query'],h);valid=c.qc_at(h)[out['query']]
        prior=np.full(len(out['query']),np.nan)
        if model is not None:prior[valid]=c.source_predict(model,x[out['query'][valid]])
        source_raw.append(prior)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,prediction=pp,accepted=aa,inner=ii,inner_b=ib,query_b=qb,threshold=thresholds,
        source_raw=source_raw,test=spec['test'],seed=c.SEEDS,training=np.array([a['train'] for a in audits]),
        B_from_original_data=np.asarray(c.original_B_at(h)),valid_UTEG_only_validation=np.asarray(not c.original_B_at(h)))
    c.write(path.with_suffix('.audit.json'),audits)
    p=np.array(pp);a=np.array(aa);y=d['y'][spec['test'],2]
    metrics=dict(all_report=c.summarize(p,y),boundary95=c.summarize(p,y,a),input_hash=c.sha(path),horizon=h,budget=budget,direction=direction)
    for key in ['all_report','boundary95']:
        metrics[key].update(B_from_original_data=c.original_B_at(h),valid_UTEG_only_validation=not c.original_B_at(h))
    c.write(path.with_suffix('.metrics.json'),metrics)
    print('PAPER_COMPLETE',direction,h,budget,flush=True)

def all301(h):
    path=c.OUT/'all301'/f'T{h}.npz'
    if path.exists():return
    d,_,_,_,_,folds,local=c.context();pos={int(v):i for i,v in enumerate(local)}
    pred=np.full((3,len(local)),np.nan);accepted=np.zeros((3,len(local)),bool);audits=[]
    for fi,f in enumerate(folds):
        arrays=[]
        for j,direction in enumerate(c.MAIN):
            out=c.evaluate(f['A'],f['B'],f['inner'],direction,h)
            ix=[pos[int(v)] for v in f['B']];pred[j,ix]=out['prediction'];accepted[j,ix]=out['accepted']
            arrays.append(out)
        audits.append(dict(fold=fi,arms=arrays))
    if h==900:
        with np.load(c.selected.OUT/'private/PREDICTIONS.npz',allow_pickle=False) as old:
            np.testing.assert_allclose(pred,old['predictions'],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(accepted,old['accepted'])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,prediction=pred,accepted=accepted,local=local,reference=d['y'][local,2],directions=c.MAIN)
    c.write(path.with_suffix('.audit.json'),audits)
    c.write(path.with_suffix('.metrics.json'),{arm:dict(all_report=c.summarize(pred[j],d['y'][local,2]),boundary95=c.summarize(pred[j],d['y'][local,2],accepted[j])) for j,arm in enumerate(c.MAIN)})
    print('ALL301_COMPLETE',h,flush=True)

def smoke():
    checks=c.selected.synthetic_tests();d,_,_,_,_,folds,local=c.context()
    out=c.evaluate(folds[0]['A'],folds[0]['B'],folds[0]['inner'],'HZ+HB_to_SA',900)
    with np.load(c.selected.OUT/'private/FOLD_0.npz',allow_pickle=False) as old:
        np.testing.assert_allclose(out['prediction'],old['prediction'][2],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(out['accepted'],old['accepted'][2])
    # Earlier features contain no reference-MA information and never use T900.
    a=np.array(folds[0]['A']);b=np.array(folds[0]['B']);original=d['y'].copy()
    for h in [300,600]:
        x=c.feature(a,b,h);d['y'][b]=1e6;xx=c.feature(a,b,h);d['y'][:]=original
        np.testing.assert_array_equal(x,xx)
    checks.update(selected_dual_first_fold_reproduced=True,early_features_label_invariant=True,
        original_roles_preserved=True,primary_config_frozen=True)
    c.write(c.OUT/'SMOKE_VERIFIED.json',checks);print('SMOKE_PASSED',flush=True)

def run(stage):
    initialize()
    if stage=='smoke':smoke();return
    assert (c.OUT/'SMOKE_VERIFIED.json').exists()
    start=time.time()
    if stage in ('main','all'):
        for h in c.HORIZONS:all301(h)
        for direction in c.MAIN:
            for h in c.HORIZONS:paper(direction,h)
    if stage in ('rest','all'):
        for direction in c.TARGET:
            for h in c.HORIZONS:paper(direction,h)
        for direction in c.TARGET:
            for budget in ([20,40] if c.TARGET[direction]=='SAHZU' else [20]):paper(direction,900,budget)
    c.write(c.OUT/f'PRIMARY_{stage.upper()}_COMPLETE.json',dict(seconds=time.time()-start,files=len(list((c.OUT/'primary').glob('*.npz')))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','main','rest','all']);args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.stage)
