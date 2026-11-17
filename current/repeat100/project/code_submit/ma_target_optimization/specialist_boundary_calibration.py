"""Small train-selected score calibrations and conditional MA rescue heads.

The reference states stay at50/70mm. A fixed score offset is applied to the
selected backup's continuous output, or a class-conditional mean is fitted on
the actual training records. Query labels never fit either operation.
"""
import argparse
import itertools
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import frozen_moment_specialist as source
from .feature_combination_audit import detailed
from .metrics import states,selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

GRID=[dict(low_offset=l,high_offset=h,priority=p,numeric=n)
      for l,h,p,n in itertools.product([0.,1.,2.,4.],[0.,1.,2.,4.],
          ['low_first','high_first'],['shifted','conditional_mean'])]
GRID.append(dict(control='primary'))


def conditional_means(data,train):
    y=data['y'][train,2];state=states(y);answer=[];counts=[]
    for k in range(3):
        values=y[state==k];origin='target_training'
        if not len(values):
            sy=data['y'][data['source_rows']['HB_to_SA'],2]
            values=sy[states(sy)==k];origin='source_training_fallback'
        if not len(values):raise ValueError('No training support for conditional MA')
        answer.append(float(values.mean()));counts.append(dict(n=len(values),origin=origin))
    assert np.array_equal(states(answer),np.arange(3))
    return np.asarray(answer),counts


def predict(raw,c,means):
    if 'control' in c:return raw[:,0].copy()
    out=raw[:,0].copy();original_normal=states(out)==1;provider=np.zeros(len(out),int)
    def high():
        take=original_normal&(states(out)==1)&(raw[:,0]>70-c['high_offset'])
        out[take]=raw[take,0]+c['high_offset'];provider[take]=3
    def low():
        for j in [1,2]:
            take=original_normal&(states(out)==1)&(raw[:,j]<50+c['low_offset'])
            out[take]=np.maximum(raw[take,j]-c['low_offset'],0);provider[take]=j
    if c['priority']=='high_first':high();low()
    else:low();high()
    if c['numeric']=='conditional_mean':
        selected=provider!=0;out[selected]=means[states(out[selected])]
    elif c['numeric']!='shifted':raise ValueError(c['numeric'])
    assert np.array_equal(out[~original_normal],raw[~original_normal,0])
    return out


def evaluate(data,blocks,cur,smooth,fold,extra):
    a0,b=fold['A'],fold['B'];ad=extra[~np.isin(data['groups'][extra],data['groups'][b])]
    tr=np.r_[a0,ad];raw,model_audit=source.models(data,blocks,cur,smooth,tr,b)
    means,counts=conditional_means(data,tr)
    fixed=np.asarray([predict(raw,c,means) for c in GRID])
    inner=np.full((len(GRID),len(a0)),np.nan);pos={int(v):j for j,v in enumerate(a0)}
    for f in fold['inner']:
        t,v=f['T'],f['V'];add=ad[~np.isin(data['groups'][ad],data['groups'][v])];tt=np.r_[t,add]
        p,_=source.models(data,blocks,cur,smooth,tt,v)
        inner_mean,_=conditional_means(data,tt)
        inner[:,[pos[int(j)] for j in v]]=np.asarray([predict(p,c,inner_mean) for c in GRID])
    y=data['y'][a0,2];mm=[detailed(y,p) for p in inner]
    choice=dict(mse=int(np.argmin(np.mean((inner-y)**2,axis=1))),
        safety=min(range(len(GRID)),key=lambda j:(selection_key(mm[j]),j)))
    audit=dict(model_audit=model_audit,conditional_means=means.tolist(),conditional_counts=counts)
    return fixed,choice,audit,inner,raw


def boundary_diagnostic(raw,y):
    reference=np.broadcast_to(y,raw.shape[:2]);t=states(reference)
    p,_=source.apply_rule(raw.reshape(-1,3),source.RULES['moment_then_curve_low'])
    q=states(p.reshape(reference.shape));groups={
        'remaining_low_to_normal':(t==0)&(q==1),
        'remaining_high_to_normal':(t==2)&(q==1),
        'high_to_low':(t==2)&(q==0),
        'correct_normal':(t==1)&(q==1)}
    result={}
    for key,k in groups.items():
        result[key]=dict(events=int(k.sum()),primary_ma_quantiles=np.quantile(raw[:,:,0][k],[0,.25,.5,.75,1]).tolist() if k.any() else [],
            minimum_backup_ma_quantiles=np.quantile(np.min(raw[:,:,1:],axis=2)[k],[0,.25,.5,.75,1]).tolist() if k.any() else [])
    return dict(scope='Outer error description after all predictions fixed, not fitted calibration or threshold tuning.',groups=result)


def run(seeds,name,smoke=False):
    data,parts,extra,blocks,cur,hashes=source.moment.load_inputs(seeds)
    d2,p2,ex2,reps,mask,more=source.load_waveforms(seeds);hashes.update(more)
    assert np.array_equal(extra,ex2) and np.array_equal(parts[1000]['pool'],p2[1000]['pool'])
    smooth=reps['smooth900'];out=ROOT/'outputs/uteg_mapping_target_runs'/name
    (out/'private').mkdir(parents=True,exist_ok=True)
    protocol=dict(candidates=GRID,seeds=list(parts),scope='Same C100 60 query grouped development design; original201 masked.',
        models='Frozen3 specialist regressors from previous20-seed experiment; no encoder retraining.',
        calibration='Prespecified0/1/2/4mm output offsets: low backup score<50+offset returns score-offset; primary-normal score>70-offset returns score+offset. Priority low-first/high-first. Actual reference boundaries50/70 unchanged. Existing primary abnormal outputs preserved.',
        numeric='Shifted continuous output or fitted same-class target-training mean on rescued cases; source-training fallback only if class absent locally. Conditional mean is a separately evaluated predictive head, never selected using query truth. All numeric/class metrics use that same final output.',
        isolation='Each inner conditional mean uses its own inner TRAINING records, not outer A labels wholesale. Inner predictions select offsets/priority/numeric head; outer labels evaluate only. No case dropping or withholding.',
        extension='5seed screen extends a candidate if joint pass or agreement>80,FNR<15% and low recall>=75%, or lower MAE at no worse agreement/FNR than previous rescue; fixed candidates exploratory.',
        input_hashes=hashes,code_sha256=sha256(__file__),test_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    old=np.load(ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_specialist_20seed_20261102/private/PREDICTIONS.npz')
    pool=parts[1000]['pool'];assert np.array_equal(old['pool'],pool);pos={int(v):j for j,v in enumerate(pool)}
    fixed=np.full((len(GRID),seeds,len(pool)),np.nan);nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']};ledger=[];begin=time.time()
    control=next(j for j,c in enumerate(GRID) if c==dict(low_offset=0.,high_offset=0.,priority='low_first',numeric='shifted'))
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            p,choice,audit,inner,raw=evaluate(data,blocks,cur,smooth,fold,extra)
            ix=[pos[int(v)] for v in fold['B']]
            assert np.max(abs(raw-old['base'][si,ix]))<1e-7
            assert np.max(abs(p[control]-old['fixed'][3,si,ix]))<1e-7
            if smoke:
                poison=dict(data);poison['y']=data['y'].copy();poison['y'][fold['B']]=np.nan
                changed={k:[v.copy() for v in arrays] for k,arrays in blocks.items()}
                for arrays in changed.values():
                    for v in arrays:v[fold['B']]=1e4
                cc=cur.copy();cc[fold['B']]=1e4;ss=smooth.copy();ss[fold['B']]=1e4
                _,choice2,audit2,inner2,_=evaluate(poison,changed,cc,ss,fold,extra)
                assert choice2==choice and audit2==audit;np.testing.assert_array_equal(inner2,inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True,previous_models_and_rule_replayed=True),indent=2))
                print('SMOKE_COMPLETE',round(time.time()-begin,1),flush=True);return
            fixed[:,si,ix]=p
            for key,j in choice.items():nested[key][si,ix]=p[j]
            ledger.append(dict(seed=seed,fold=fi,choice={k:GRID[j] for k,j in choice.items()},audit=audit))
        print('CALIBRATION_SEED',seed,round(time.time()-begin,1),flush=True)
    yy=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    rows=[dict(candidate=c,metrics=detailed(yy,p)) for c,p in zip(GRID,fixed)]
    selected=[dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()]
    for row,p in zip(rows+selected,list(fixed)+list(nested.values())):
        a,b=states(yy),states(p)
        assert [[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]==row['metrics']['confusion']
        assert abs(row['metrics']['r2']-(1-np.sum((p-yy)**2)/np.sum((yy-yy.mean())**2)))<1e-12
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=fixed,reference=yy[0],pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    oldmetric=rows[control]['metrics']
    promising=[r for r in rows if r['metrics']['mapping_target_pass'] or
        (r['metrics']['agreement']>.8 and r['metrics']['fnr']<.15 and r['metrics']['low_recall']>=.75) or
        (r['metrics']['agreement']>=oldmetric['agreement'] and r['metrics']['fnr']<=oldmetric['fnr'] and r['metrics']['mae']<oldmetric['mae']-1e-6)]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows,nested=selected,promising=promising,
        diagnostic=boundary_diagnostic(old['base'][:seeds],yy[0]),checks=dict(previous_models_rule_replayed=True,input_hashes_unchanged=True),
        test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='specialist_boundary_calibration_screen_20261102');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output,a.smoke)
