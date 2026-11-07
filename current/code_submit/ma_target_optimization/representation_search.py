"""Development-only representation and direct-state diagnostic comparisons.

Class-mean predictions are explicitly coarse categorical regressors. They
cannot be described as precise continuous MA mappings if classification wins.
"""
import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from .search import ROOT, training_view, make_summary
from .models import Features, fit_candidate
from .metrics import assess, states
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def curve_features(y):
    y=np.asarray(y,float)
    if len(y)<30 or not np.isfinite(y).all():
        raise ValueError('Insufficient real observed signal')
    # Signals have already undergone their frozen center-specific preprocessing.
    end=float(np.median(y[-30:]));peak=float(np.quantile(y,.99))
    values=[]
    for t in (60,120,180,300,450,600,750,900):
        hi=min(t,len(y));lo=max(0,hi-30)
        values.append(float(np.median(y[lo:hi])))
    increments=np.diff(values)
    values+=increments.tolist()
    values += [end,peak,float(np.mean(y)),float(np.std(y)),float(np.std(np.diff(y))),len(y)/900.]
    return np.asarray(values)


def load_representations(data,partitions):
    base=ROOT/'outputs'; paths=[base/'uteg_cohort399_20261006_201800/private/data.npz',base/'uteg_platform_filter_pilot_20261006_124636/private/CUT_5.npz',base/'uteg_legacy_aligned_threecenter_20261022_155614/HUBEI_SIGNAL_INPUTS.npz']
    allowed=data['allowed_training_indices'];n=len(data['x']);new=np.full((n,21),np.nan)
    with np.load(paths[0],allow_pickle=False) as z:
        original=z['original_row_index'];assert len(original)==399
        assert np.array_equal(z['ids'],data['ids'][:399])
    with np.load(paths[1],allow_pickle=False) as z:
        old_signal=z['signal'];old_observed=z['observed']
    with np.load(paths[2],allow_pickle=False) as z:
        hb_signal=z['denoise_p5'];hb_observed=z['observed']
    for index in allowed:
        if index<399:
            row=int(original[index]);signal=old_signal[row];observed=old_observed[row]
        else:
            row=index-399;signal=hb_signal[row];observed=hb_observed[row]
        # No padding enters features; no measurements after 900 s are used.
        y=signal[:900][observed[:900]]
        new[index]=curve_features(y)
    old=data['x'].copy()
    test=next(iter(partitions.values()))['test'];old[test]=np.nan
    return {'parameters4':old,'curve21':new,'parameters_plus_curve':np.c_[old,new]}, {str(p):sha256(p) for p in paths}


@dataclass
class ClassMeanModel:
    feature_map: Features
    classifier: object
    means: np.ndarray

    def predict(self,x):
        label=self.classifier.predict(self.feature_map.transform(x)).astype(int)
        return self.means[label]


def fit_classmean(x,y,c):
    f=Features.fit(x,'linear');z=f.transform(x);label=states(y)
    w=np.where(label==1,1.,c['cost']);w=w/w.mean()
    if c['learner']=='logistic':
        model=LogisticRegression(C=c['C'],max_iter=3000,solver='lbfgs')
    elif c['learner']=='svm':
        model=SVC(C=c['C'],gamma='scale',kernel='rbf')
    elif c['learner']=='histogram':
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=c['leaf'],l2_regularization=1.,early_stopping=False,random_state=20261101)
    elif c['learner']=='forest':
        model=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=c['leaf'],max_features=.8,n_jobs=1,random_state=20261101)
    else:
        raise ValueError(c['learner'])
    model.fit(z,label,sample_weight=w)
    # Missing classes receive NaN means, but an estimator cannot predict a
    # class absent from its fit. No fake samples or test-derived priors.
    means=np.array([y[label==k].mean() if np.any(label==k) else np.nan for k in range(3)])
    return ClassMeanModel(f,model,means)


def candidate_grid():
    grid=[]
    for representation in ('parameters4','curve21','parameters_plus_curve'):
        for learner,C,cost in itertools.product(('logistic','svm'),(.1,1.,10.),(1.,4.,16.,64.)):
            grid.append(dict(family='class_mean',representation=representation,learner=learner,C=C,cost=cost))
        for learner,leaf,cost in itertools.product(('histogram','forest'),(3,8),(1.,4.,16.)):
            grid.append(dict(family='class_mean',representation=representation,learner=learner,leaf=leaf,cost=cost))
        if representation!='parameters4':
            for alpha,cap in itertools.product((.001,.01,.1,1.),(1.,5.,20.)):
                grid.append(dict(family='ridge',representation=representation,features='linear',alpha=alpha,cap=cap))
            for alpha,strength,cost in itertools.product((.01,.1),(1.,4.),(4.,16.)):
                grid.append(dict(family='boundary',representation=representation,features='linear',alpha=alpha,cap=5.,boundary_strength=strength,abnormal_cost=cost))
    for c in grid:
        c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    return grid


def run(out,seeds):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=seeds
    data,partitions,input_hashes=load_development_inputs(ROOT,cfg)
    matrices,additional=load_representations(data,partitions);input_hashes.update(additional)
    grid=candidate_grid();pool=next(iter(partitions.values()))['pool'];position={int(v):i for i,v in enumerate(pool)}
    protocol={'seeds':seeds,'candidates':grid,'input_hashes':input_hashes,'features':{'curve21':'8 observed prefix levels, 7 increments, end, peak, mean, std, diff_std, observed_fraction; no labels'},
              'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).resolve().parent.glob('*.py')},
              'scope':'development-only; test features left NaN; no test evaluation',
              'class_mean_scope':'direct classification followed by training-class mean; coarse regression diagnostic, not continuous mapping superiority',
              'targets':'all-report agreement>80%, abnormal FNR<5% and FNR<FPR; later trust-gate coverage>75%'}
    text=json.dumps(protocol,ensure_ascii=False,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:
        raise RuntimeError('Protocol changed')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8')
    output=out/'CANDIDATES.jsonl';records=[json.loads(v) for v in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    done={r['candidate']['id'] for r in records}
    for number,c in enumerate(grid):
        if c['id'] in done:
            continue
        start=time.time();data['x']=matrices[c['representation']]
        try:
            all_pred=[]
            for seed,part in partitions.items():
                pred=np.full(len(pool),np.nan)
                for fold in part['folds']:
                    A,B=fold['A'],fold['B']
                    if c['family']=='class_mean':
                        model=fit_classmean(data['x'][A],data['y'][A,2],c)
                    else:
                        model=fit_candidate(c,training_view(data,A),{})
                    pred[[position[int(v)] for v in B]]=model.predict(data['x'][B])
                all_pred.append(pred)
            all_pred=np.asarray(all_pred);y=data['y'][pool,2]
            m=assess(np.broadcast_to(y,all_pred.shape),all_pred)
            np.savez_compressed(out/'private'/f"{c['id']}.npz",prediction=all_pred,reference=y,pool=pool,seeds=seeds)
            row={'candidate':c,'status':'complete','metrics':m,'seconds':time.time()-start}
        except Exception as error:
            row={'candidate':c,'status':'failed','error':f'{type(error).__name__}: {error}','seconds':time.time()-start}
        with output.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        records.append(row)
        if (number+1)%10==0 or number+1==len(grid):
            s=make_summary(records,out);print(json.dumps({'done':number+1,'total':len(grid),'failed':s['failed'],'feasible':s['feasible_count'],'representation':c['representation']}),flush=True)
    return make_summary(records,out)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',required=True);args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:raise ValueError('Invalid run-name')
    with threadpool_limits(limits=1):
        run(ROOT/'outputs/uteg_mapping_target_runs'/args.run_name,list(range(1000,1005)))
