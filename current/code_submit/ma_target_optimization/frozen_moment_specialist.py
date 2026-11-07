"""Prespecified abnormal-rescue cascades of complementary fixed regressors.

The primary model keeps every existing abnormal prediction. Backups may change
only a primary-normal prediction, using their own fixed50/70 boundaries. Every
record is reported; returned MA is the actual chosen model's continuous value.
"""
import argparse
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import frozen_moment_mapping as moment
from .full_curve_cost_sensitive import load_waveforms,kernel_design,solve
from .feature_combination_audit import assemble,all_linear_predictions,detailed
from .metrics import states,selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

# columns: primary current18, pretrained MOMENT, smooth-curve boundary model.
RULES={
 'primary':[],
 'moment_low':[(1,[0])],
 'curve_low':[(2,[0])],
 'moment_then_curve_low':[(1,[0]),(2,[0])],
 'curve_then_moment_low':[(2,[0]),(1,[0])],
 'moment_any':[(1,[0,2])],
 'both_low_then_moment_high':[(1,[0]),(2,[0]),(1,[2])],
 'moment_any_then_curve_low':[(1,[0,2]),(2,[0])],
 'moment_low_then_curve_any':[(1,[0]),(2,[0,2])],
}


def apply_rule(p,rule):
    out=p[:,0].copy();provider=np.zeros(len(p),int)
    for column,allowed in rule:
        take=(states(out)==1)&np.isin(states(p[:,column]),allowed)
        out[take]=p[take,column];provider[take]=column
    return out,provider


def models(data,blocks,cur,smooth,a,b):
    primary=all_linear_predictions(data,cur,a,b,'HB_dense',[dict(columns=list(range(18)))])[0]
    z,q,y,w,diag=moment.design(data,blocks['embedding_amp6'],a,b,'HB_dense','linear')
    coef,_=solve(z,(y-60)/10,w,'boundary',16.,alpha=.001)
    pretrained=np.maximum(60+10*q@coef,0)
    digest=hashlib.sha256(coef.tobytes()).hexdigest()
    tx,y,w,mean,scale,ns=assemble(data,smooth,a,'target_class');assert ns==0
    z,q,_=kernel_design(tx[:,:-1],(smooth[b]-mean)/scale)
    coef,_=solve(z,(y-60)/10,w,'boundary',64.)
    curve=np.maximum(60+10*q@coef,0)
    return np.c_[primary,pretrained,curve],dict(moment_coef_sha256=digest,
        curve_coef_sha256=hashlib.sha256(coef.tobytes()).hexdigest(),**diag)


def evaluate(data,blocks,cur,smooth,fold,extra):
    a0,b=fold['A'],fold['B'];ad=extra[~np.isin(data['groups'][extra],data['groups'][b])]
    p,audit=models(data,blocks,cur,smooth,np.r_[a0,ad],b)
    inner=np.full((len(a0),3),np.nan);pos={int(v):j for j,v in enumerate(a0)}
    for f in fold['inner']:
        t,v=f['T'],f['V'];add=ad[~np.isin(data['groups'][ad],data['groups'][v])]
        pp,_=models(data,blocks,cur,smooth,np.r_[t,add],v)
        inner[[pos[int(j)] for j in v]]=pp
    predictions=np.array([apply_rule(p,r)[0] for r in RULES.values()])
    inner_pred=np.array([apply_rule(inner,r)[0] for r in RULES.values()])
    y=data['y'][a0,2];mm=[detailed(y,v) for v in inner_pred]
    chosen=dict(mse=int(np.argmin(np.mean((inner_pred-y)**2,axis=1))),
        safety=min(range(len(RULES)),key=lambda j:(selection_key(mm[j]),j)))
    return predictions,chosen,audit,p,inner


def run(seeds,name,smoke=False):
    data,parts,extra,blocks,cur,hashes=moment.load_inputs(seeds)
    d2,p2,ex2,reps,mask,more=load_waveforms(seeds);hashes.update(more)
    assert np.array_equal(extra,ex2) and np.array_equal(parts[1000]['pool'],p2[1000]['pool'])
    smooth=reps['smooth900'];out=ROOT/'outputs/uteg_mapping_target_runs'/name
    (out/'private').mkdir(parents=True,exist_ok=True)
    protocol=dict(rules=RULES,seeds=list(parts),
        question='Can pretrained low-MA support complement two earlier models while retaining all existing abnormal predictions?',
        scope='Same60 grouped C100 development queries, extra40 training-only; no original201 features/labels.',
        models='Primary current18 HB DenseWeight; pretrained+amp6 HB DenseWeight linear Gram alpha.001 boundary16; smooth900 target class-weighted RBF alpha.01 boundary64. Settings fixed before cascade results; adaptive development experiment, not untouched validation.',
        rule='If current output remains normal50..70, accept backup only for its listed abnormal class. Actual continuous backup MA is returned, no clipping to category, dropping, or withheld records. Primary abnormal predictions are never overridden. All errors including opposite-tail count.',
        selection='Inner3fold predictions choose one of9 fixed rules; outer3fold evaluates. No query-label rule fitting or threshold changes.',
        extension='5seed screen extends if strict pass or agreement>80,FNR<37%,low recall>=25%,or R2>.35.',
        input_hashes=hashes,code_sha256=sha256(__file__),test_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=parts[1000]['pool'];pos={int(v):j for j,v in enumerate(pool)}
    old=np.load(ROOT/'outputs/uteg_mapping_target_runs/systematic_shift_audit_20261101/private/current300_HB_dense.npz')
    old_m=np.load(ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_mapping_screen_20261102/private/PREDICTIONS.npz')
    configs=json.loads((ROOT/'outputs/uteg_mapping_target_runs/frozen_moment_mapping_screen_20261102/RESULTS.json').read_text(encoding='utf-8'))['fixed']
    mi=next(i for i,r in enumerate(configs) if r['candidate']==dict(representation='embedding_amp6',recipe='HB_dense',basis='linear',alpha=.001,loss='boundary',cost=16.))
    curve_old=np.load(ROOT/'outputs/uteg_mapping_target_runs/full_curve_cost_sensitive_20261102/private/PREDICTIONS.npz')
    cc=json.loads((ROOT/'outputs/uteg_mapping_target_runs/full_curve_cost_sensitive_20261102/RESULTS.json').read_text())['fixed']
    ci=next(i for i,r in enumerate(cc) if r['candidate']==dict(representation='smooth900',recipe='target_class',kind='boundary',cost=64.))
    fixed=np.full((len(RULES),seeds,len(pool)),np.nan);basepred=np.full((seeds,len(pool),3),np.nan)
    nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']};ledger=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            p,choice,audit,raw,inner=evaluate(data,blocks,cur,smooth,fold,extra)
            ix=[pos[int(v)] for v in fold['B']]
            assert np.max(abs(raw[:,0]-old['none'][si,ix]))<1e-7
            assert np.max(abs(raw[:,2]-curve_old['fixed'][ci,si,ix]))<1e-7
            if si<5:assert np.max(abs(raw[:,1]-old_m['fixed'][mi,si,ix]))<1e-7
            if smoke:
                poison=dict(data);poison['y']=data['y'].copy();poison['y'][fold['B']]=np.nan
                changed={k:[v.copy() for v in arrays] for k,arrays in blocks.items()}
                for arrays in changed.values():
                    for v in arrays:v[fold['B']]=1e4
                c2=cur.copy();c2[fold['B']]=1e4;s2=smooth.copy();s2[fold['B']]=1e4
                _,ch2,au2,_,in2=evaluate(poison,changed,c2,s2,fold,extra)
                assert ch2==choice and au2==audit;np.testing.assert_array_equal(in2,inner)
                (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_poison_invariant=True,all3models_replayed=True),indent=2))
                print('SMOKE_COMPLETE',round(time.time()-begin,1),flush=True);return
            fixed[:,si,ix]=p;basepred[si,ix]=raw
            for key,j in choice.items():nested[key][si,ix]=p[j]
            ledger.append(dict(seed=seed,fold=fi,selection={k:list(RULES)[j] for k,j in choice.items()},audit=audit))
        print('RESCUE_SEED',seed,round(time.time()-begin,1),flush=True)
    yy=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    rows=[dict(rule=c,metrics=detailed(yy,p)) for c,p in zip(RULES,fixed)]
    selected=[dict(objective=k,metrics=detailed(yy,p)) for k,p in nested.items()]
    for row,p in zip(rows+selected,list(fixed)+list(nested.values())):
        a,b=states(yy),states(p)
        assert [[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]==row['metrics']['confusion']
        assert np.array_equal(p[states(basepred[:,:,0])!=1],basepred[:,:,0][states(basepred[:,:,0])!=1])
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=fixed,reference=yy[0],base=basepred,pool=pool,seeds=list(parts),**nested)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    promising=[r for r in rows if r['metrics']['mapping_target_pass'] or r['metrics']['r2']>.35 or
        (r['metrics']['agreement']>.8 and r['metrics']['fnr']<.37 and r['metrics']['low_recall']>=.25)]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=rows,nested=selected,promising=promising,
        checks=dict(all3models_replayed=True,primary_abnormal_predictions_preserved=True,input_hashes_unchanged=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='frozen_moment_specialist_screen_20261102');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output,a.smoke)
