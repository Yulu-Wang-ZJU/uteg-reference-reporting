"""Complete current-version experiments without consulting archived data."""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from . import core as c
from . import current_version as current
from . import primary, mapping, controls, gates, analysis
from .input_contract import input_provenance, check_protocol

ROOT=current.ROOT


def numerical_models():
    """Load historical numerical definitions only, with a new local runtime."""
    source=c.WS/'outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551/runtime.py'
    directory=ROOT/'numerical_runtime';directory.mkdir(exist_ok=True)
    text=source.read_text(encoding='utf-8').replace('OUT.parent /',f'Path({str(c.WS/"outputs")!r}) /')
    runtime=types.ModuleType('runtime');runtime.__file__=str(directory/'runtime.py');sys.modules['runtime']=runtime
    exec(compile(text,str(source),'exec'),runtime.__dict__)
    runtime.BASE=ROOT/'data'
    m=runtime.load('current_fixed_B_source_models',c.WS/'outputs/uteg_hill_exp_matched_20261012_153337/models.py')
    mapping._ENV=m
    return m


def initialize():
    current.activate();freeze=c.read(ROOT/'DATA_FREEZE.json')
    assert all(c.sha(p)==h for p,h in freeze['files'].items())
    manifest=dict(data_sha256=freeze['current_dataset_sha256'],
        source='Current fixed dataset and prefix parameters only; historical Python numerical definitions may be reused, never historical B/data/models.',
        required=['52paper_primary_budget_bundles','3all301_primary_bundles','54source_mapping_bundles',
            '9primary_calibration_bundles','all301_six_model_controls','16representation_loss_bundles','median_controls',
            '36paper_reporting_gate_bundles','3all301_reporting_gate_bundles','all_three_schedules',
            'matched20quota_and_paired_intervals','finite_local_cost_comparison','final_consistency_audit'],
        scopes=dict(paper='9source directions+3target-only,20seeds; original center-specific pool/test roles',
            all301='Frozen5outer/3inner local301; source arms HZ andHZ+HB; all301 only refers to the local cohort'),
        selection='Frozen historical hyperparameters and development-only gates. No outer-result winner promotion.',
        B_from_original_data=True,source_status=input_provenance()['source_status'],
        input_provenance=input_provenance(),manuscript_changed=False)
    p=ROOT/'FULL_SUITE_PROTOCOL.json'
    if p.exists():check_protocol(c.read(p),manifest)
    else:c.write(p,manifest)


def smoke():
    initialize();m=numerical_models();d,*_=c.context();direction='HZ_to_SA';rows=c.source_rows(direction,900)
    x=c.source_feature(direction,900,'four');yy=np.full_like(d['medical'],np.nan);yy[rows]=d['medical'][rows]
    data=dict(d,x=x,medical=yy,ci=np.where(np.isin(np.arange(len(x)),rows),d['ci'],np.nan))
    a=m.fit_source(data,rows,[],'RIDGE-10',1000)
    poison=dict(data,medical=data['medical'].copy(),ci=data['ci'].copy())
    forbidden=np.setdiff1d(np.arange(len(x)),rows);poison['medical'][forbidden]=1e6;poison['ci'][forbidden]=1e6
    b=m.fit_source(poison,rows,[],'RIDGE-10',1000)
    np.testing.assert_array_equal(m.predict(a,x),m.predict(b,x))
    source_audit=dict(rows=len(rows),excluded_labels_do_not_change_source_model=True,
        source_inputs_match_current_file=bool(np.array_equal(x,d['x'])))
    c.write(ROOT/'FULL_SUITE_SMOKE.json',source_audit);print('CURRENT_SUITE_SMOKE_PASSED',flush=True)


def primary_rest():
    initialize();assert (ROOT/'FULL_SUITE_SMOKE.json').exists()
    for direction in c.TARGET:
        for h in c.HORIZONS:primary.paper(direction,h)
        for budget in ([20,40] if c.TARGET[direction]=='SAHZU' else [20]):primary.paper(direction,900,budget)
    controls.run(fixed_input=True)
    c.write(ROOT/'PRIMARY_REST_COMPLETE.json',dict(primary_files=len(list((ROOT/'primary').glob('*.npz'))),controls=16))


def run_mapping(method):
    initialize();assert (ROOT/'FULL_SUITE_SMOKE.json').exists();numerical_models()
    p=ROOT/'MAPPING_PROTOCOL.json'
    protocol=dict(methods=mapping.METHODS,source_inputs='Four parameters read from the active fixed B file.',
        primary='B-only RBF inward4 separately reported',
        comparison='Original six source recipes and losses retained, common currentB-only local inward4 correction; not a pure architecture ablation.',
        calibration='MA source prior plus auxiliary R/K/angle from current-four-parameter Ridge for joint controls.',
        source_seeds='20seeds MLP/TT; deterministic source fits cached at1000',steps=500,
        data_sha256=c.read(ROOT/'DATA_FREEZE.json')['current_dataset_sha256'],B_from_original_data=True)
    if p.exists():assert c.read(p)==protocol
    else:c.write(p,protocol)
    mapping.run(method)
    all301_mapping(method)


def all301_mapping(method):
    """Six fixed four-output local regressors and matched source-prior controls."""
    m=mapping.env();d,_,_,_,_,folds,local=c.context();pos={int(v):i for i,v in enumerate(local)}
    methods=mapping.METHODS if method=='all' else [method]
    for method in methods:
        path=ROOT/'all301_mapping'/f'{method}.npz'
        if path.exists():continue
        pred=np.full((3,len(local),4),np.nan);accept=np.zeros((3,len(local)),bool);audits=[]
        for fi,f in enumerate(folds):
            a=np.asarray(f['A'],int);b=np.asarray(f['B'],int);ix=[pos[int(v)] for v in b]
            # Target-only counterpart: all four readouts predicted from identical
            # local labels; no hyperparameter search or auxiliary test values.
            def local_fit(train,query):
                labels=np.full_like(d['medical'],np.nan);labels[train]=d['medical'][train]
                dat=dict(d,medical=labels,ci=np.full_like(d['ci'],np.nan))
                dat['ci'][train]=d['ci'][train]
                model=m.fit_source(dat,train,[],method,1000+fi,steps=500 if method in mapping.METHODS[-2:] else None)
                assert model['success']
                return m.predict(model,d['x'][query])
            pred[0,ix]=local_fit(a,b);inner=np.full(len(a),np.nan);apos={int(v):i for i,v in enumerate(a)}
            for sub in f['inner']:
                q=np.asarray(sub['V'],int);p=local_fit(np.asarray(sub['T'],int),q)
                inner[[apos[int(v)] for v in q]]=p[:,2]
            accept[0,ix]=c.selected.models.boundary_gate(inner[None,:],pred[0,ix,2][None,:])[0][0]
            for j,arm in enumerate(c.MAIN[1:],1):
                src=mapping.source(arm,method,1000+fi)
                def prior(train,query,h):
                    x=c.feature(train,query,h,'four')
                    return m.predict(src,x[train])[:,2],m.predict(src,x[query])[:,2]
                out=c.evaluate(a,b,f['inner'],arm,900,prior_override={'predict_rows':prior})
                p=m.predict(src,d['x'][b]);p[:,2]=out['prediction'];pred[j,ix]=p;accept[j,ix]=out['accepted']
                audits.append(dict(fold=fi,arm=arm,source_seed=1000+fi,local_fits=out['audits']))
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path,prediction=pred,accepted=accept,local=local)
        c.write(path.with_suffix('.audit.json'),audits)
        c.write(path.with_suffix('.metrics.json'),{arm:dict(all_report=c.summarize(pred[j,:,2],d['y'][local,2]),gate=c.summarize(pred[j,:,2],d['y'][local,2],accept[j]),
            readout_mae=np.mean(abs(pred[j]-d['medical'][local]),axis=0),
            auxiliary_scope='Target-only jointly predicts4readouts; source arms updateMA only, other outputs source-only.') for j,arm in enumerate(c.MAIN)})
        print('ALL301_SIX_MODEL_COMPLETE',method,flush=True)


def subfolds(rows,seed):
    d,*_=c.context();rows=np.asarray(rows,int);y=c.accounting.states(d['y'][rows,2])
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    return [dict(T=rows[a].tolist(),V=rows[b].tolist()) for a,b in splitter.split(rows,y,d['groups'][rows])]


def all301_gates():
    d,_,_,_,_,folds,local=c.context();pos={int(v):i for i,v in enumerate(local)}
    for direction in c.MAIN:
        for h in c.HORIZONS:
            path=ROOT/'all301_gates'/f'{direction}_T{h}.npz'
            if path.exists():continue
            prediction=np.full(len(local),np.nan);actions={k:np.zeros(len(local),bool) for k in ['LR','HGB','TWO_SIDED','BOUNDARY_PAPER','BOUNDARY95']}
            audits=[]
            for fi,f in enumerate(folds):
                seed=1000+fi;dev_score={k:[] for k in ['LR','HGB','DUAL','BOUNDARY']};dev_pred=[];dev_true=[];rows=[]
                for pi,sub in enumerate(f['inner']):
                    inn=subfolds(sub['T'],22000+fi*3+pi)
                    out=c.evaluate(sub['T'],sub['V'],inn,direction,h)
                    ss,_=gates.fit_score(out,h,seed,direction,pi)
                    pc=gates.fr.ma_class(out['prediction']);t=gates.fr.ma_class(d['y'][out['query'],2])
                    for k in dev_score:dev_score[k].extend(ss[k])
                    dev_pred.extend(pc);dev_true.extend(t)
                    for k,i in enumerate(out['query']):rows.append(dict(record_id=str(d['ids'][i]),pred_class=int(pc[k]),truth_class=int(t[k]),
                        base_score=float(ss['LR'][k]) if np.isfinite(ss['LR'][k]) else None,dual_score=float(ss['DUAL'][k]) if np.isfinite(ss['DUAL'][k]) else None))
                out=c.evaluate(f['A'],f['B'],f['inner'],direction,h);ss,model=gates.fit_score(out,h,seed,direction,3)
                th={k:gates.rc.select_threshold(np.asarray(dev_score[k]),np.asarray(dev_pred),np.asarray(dev_true),.15) for k in ['LR','HGB','BOUNDARY']}
                dual=gates.fr.choose(rows);ix=[pos[int(v)] for v in f['B']];prediction[ix]=out['prediction'];valid=np.isfinite(out['prediction'])
                for key,score in [('LR','LR'),('HGB','HGB'),('BOUNDARY_PAPER','BOUNDARY')]:
                    v=th[score].get('threshold');actions[key][ix]=False if v is None else (ss[score]<=v)&np.isfinite(ss[score])&valid
                actions['TWO_SIDED'][ix]=gates.fr.action(dual,gates.fr.ma_class(out['prediction']),ss['LR'],ss['DUAL'],valid)
                actions['BOUNDARY95'][ix]=out['accepted'];audits.append(dict(fold=fi,thresholds=th,dual=dual,support=model['support']))
            with np.load(ROOT/'all301'/f'T{h}.npz') as z:
                np.testing.assert_array_equal(prediction,z['prediction'][c.MAIN.index(direction)])
                np.testing.assert_array_equal(actions['BOUNDARY95'],z['accepted'][c.MAIN.index(direction)])
            path.parent.mkdir(exist_ok=True);np.savez_compressed(path,prediction=prediction,**actions,local=local)
            gates.clean_write(path.with_suffix('.audit.json'),audits)
            gates.clean_write(path.with_suffix('.metrics.json'),{k:c.summarize(prediction,d['y'][local,2],a) for k,a in actions.items()})
            print('ALL301_REPORTING_COMPLETE',direction,h,flush=True)


def run_gates():
    initialize();assert len(list((ROOT/'primary').glob('*.npz')))==52
    gates.run();all301_gates()


def sequences():
    initialize();analysis.sequences(ROOT);analysis.paired_primary(ROOT)
    d,*_=c.context();result={}
    for direction in c.MAIN:
        points={h:analysis.load(ROOT/'all301_gates'/f'{direction}_T{h}.npz') for h in c.HORIZONS}
        y=d['y'][points[900]['local'],2];result[direction]={}
        for rule in ['ALL_REPORT','BOUNDARY95','LR','HGB','TWO_SIDED','BOUNDARY_PAPER']:
            actions={h:np.isfinite(points[h]['prediction']) if rule=='ALL_REPORT' else points[h][rule] for h in c.HORIZONS}
            result[direction][rule]={}
            for name,times in analysis.SCHEDULES.items():
                pp,tt,aa=analysis.replay({h:points[h]['prediction'] for h in c.HORIZONS},actions,times)
                mm=c.summarize(pp,y,aa);mm['by_time']={str(h):dict(reported=int(np.sum(tt==h)),correct=int(np.sum((tt==h)&(c.accounting.states(pp)==c.accounting.states(y))))) for h in times}
                result[direction][rule][name]=mm
    gates.clean_write(ROOT/'sequence/ALL301_ALL_RULES.json',result)
    print('CURRENT_FULL_SEQUENCES_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','primary','mapping','gates','sequence']);ap.add_argument('--method',default='all')
    a=ap.parse_args()
    with threadpool_limits(limits=1):
        if a.stage=='mapping':run_mapping(a.method)
        else:{'smoke':smoke,'primary':primary_rest,'gates':run_gates,'sequence':sequences}[a.stage]()
