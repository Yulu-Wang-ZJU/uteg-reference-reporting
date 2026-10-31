from __future__ import annotations
import ast, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import joblib
from scipy.optimize import minimize
from scipy.special import expit

ROOT=Path(__file__).resolve().parent
G=ROOT.parent
OUT=G.parent
OLD=OUT/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
DATA=OUT/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
CONTRACT=OUT/'uteg_ridge_dual_head_prepare_20261026_172952/dual_head_contract.py'
SELECTOR=OUT/'uteg_ridge_dual_head_unified_selector_20261026_190142/run_unified_selector.py'
EXT=OUT/'uteg_ridge_dual_head_external_eval_20261026_191117/actions'
DIRS=['HZ_to_SA','HZ+HB_to_SA']; SEEDS=list(range(1000,1020)); ALPHAS=[.1,1.,10.,100.]
MA_CLASSES=[-1,0,1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    path=ROOT/name
    assert not path.exists(),f'Never overwrite {path}'
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
def table(name,rows):
    assert not (ROOT/name).exists()
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

# Import only the six pure historical functions, without the runner's runtime.
names={'ma_class','ma_features','_preprocess_fit','_empty_gate_pre','_fit_ma_gate','score_ma_gate'}
tree=ast.parse((OLD/'run_ridge10_fullchain.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(OLD/'run_ridge10_fullchain.py'),'exec'))
c=load('focused_dual_contract',CONTRACT); sel=load('focused_original_selector',SELECTOR)
ridge=load('focused_ridge',G/'target_ridge/run.py')

def original(d,s): return OLD/'models'/d/'target15'/f'SAHZU_CFULL_seed{s}.joblib'
def choose(rows):
    values=[None]+sorted({r['dual_score'] for r in rows if r['pred_class']==0 and r['dual_score'] is not None})
    candidates=[dict(threshold=t,candidate_index=j,n=len(rows),release_record_ids=[r['record_id'] for r in rows if t is not None and r['pred_class']==0 and r['dual_score'] is not None and r['dual_score']<=t]) for j,t in enumerate(values)]
    n=sel.choose_normal(candidates,rows,{r['record_id']:r for r in rows});t=sel.choose_common_tail(rows,n)
    return dict(normal=n['selected']['threshold'] if n['selected'] else None,tail=t['selected']['threshold'] if t['selected'] else None,all_withhold=t['selected'] is None,normal_status=n['status'],tail_status=t['status'])

def head_scores(heads,x,med):
    values=[]
    for h in ['low','high']:
        m=heads[h]
        if m is None: values.append(np.full(len(x),np.inf));continue
        p=m['pre'];f=np.column_stack([x[:,1],med[:,2]])
        z=np.clip((np.where(np.isfinite(f),f,p['median'])-p['mean'])/p['sd'],-10,10)
        values.append(c.score_head(m['theta'],z))
    return np.maximum(*values)

def action(th,pred,base,dual,qc):
    out=np.zeros(len(pred),bool)
    if th['all_withhold']:return out
    if th['normal'] is not None:out|=(pred==0)&np.isfinite(dual)&(dual<=th['normal'])
    if th['tail'] is not None:out|=np.isin(pred,[-1,1])&np.isfinite(base)&(base<=th['tail'])
    return out&qc

def checks():
    rng=np.random.default_rng(381);x=rng.normal(size=(30,4));med=rng.normal(size=(30,4))*3+60
    a=ma_features(x,med,np.zeros(4),np.ones(4));b=ma_features(x,med,np.arange(4),np.arange(4)+1.)
    pa=_preprocess_fit(a);pb=_preprocess_fit(b)
    za=np.clip((a-pa['mean'])/pa['sd'],-10,10);zb=np.clip((b-pb['mean'])/pb['sd'],-10,10)
    assert np.allclose(za,zb,atol=1e-12)
    assert np.array_equal(ma_class([49.9,50,70,70.1,np.nan]),[-1,0,0,1,9])
    assert not action(dict(all_withhold=True),np.array([0]),np.array([0.]),np.array([0.]),np.array([True])).any()
    assert np.isinf(head_scores({'low':None,'high':None},x,med)).all()
    assert c.support_summary(np.array([0,0,0,0,0,0,0,1]))['status']=='UNSUPPORTED'
    return dict(normalization_max_difference=float(abs(za-zb).max()),synthetic_contract=True,selector=sel._synthetic_tests())

def train():
    assert not (ROOT/'FREEZE.json').exists()
    assert not list(ROOT.glob('model_*.joblib')),'Partial run requires explicit audit; do not restart'
    inputs={str(p):sha(p) for p in [DATA,CONTRACT,SELECTOR,OLD/'run_ridge10_fullchain.py',G/'target_ridge/run.py',ROOT/'PROTOCOL.md',Path(__file__)]}
    summaries=[];foldrows=[];models={};counts=dict(ridge=0,error=0,dual=0);supportrows=[]
    testchecks=checks()
    for seed in SEEDS:
        p=original(DIRS[0],seed);inputs[str(p)]=sha(p);o=joblib.load(p)
        other=original(DIRS[1],seed);inputs[str(other)]=sha(other);oo=joblib.load(other)
        assert o['pool']==oo['pool'] and o['test']==oo['test']
        for k in range(4):
            for f in ['A','B']:assert o['positions'][k][f]==oo['positions'][k][f]
            assert [(v['T'],v['V']) for v in o['positions'][k]['inner']]==[(v['T'],v['V']) for v in oo['positions'][k]['inner']]
        pool=np.array(o['pool']); lookup={int(g):i for i,g in enumerate(pool)}
        with np.load(DATA,allow_pickle=False) as z:
            # The training interface exposes reference outcomes only for the reserved pool.
            x=z['x'][pool].copy();y=z['medical'][pool].copy();ids=z['ids'][pool].copy();groups=z['groups'][pool].copy();qc=z['input_qc'][pool].copy()
            assert not set(groups)&set(z['groups'][o['test']])
        assert len(pool)==60 and np.isfinite(x).all() and np.isfinite(y).all()
        final=None;rows=[]
        for pos in range(4):
            part=o['positions'][pos];A=np.array([lookup[i] for i in part['A']]);B=np.array([lookup[i] for i in part['B']],int);loc={int(v):j for j,v in enumerate(A)}
            cross={a:np.full((len(A),4),np.nan) for a in ALPHAS};covered=[]
            for f in part['inner']:
                T=np.array([lookup[i] for i in f['T']]);V=np.array([lookup[i] for i in f['V']]);covered.extend(V)
                assert set(T)|set(V)==set(A) and not set(groups[T])&set(groups[V])
                for a in ALPHAS:
                    m=ridge.fit(x[T],y[T],a);counts['ridge']+=1
                    cross[a][[loc[int(v)] for v in V]]=ridge.predict(m,x[V])
            assert sorted(covered)==sorted(A)
            losses={a:float(abs(cross[a][:,2]-y[A,2]).mean()) for a in ALPHAS}
            best=min(ALPHAS,key=lambda a:(losses[a],-a)); med=cross[best]
            model=ridge.fit(x[A],y[A],best);counts['ridge']+=1
            pc=ma_class(med[:,2]);truth=ma_class(y[A,2]);ym=y[A].mean(0);ys=np.maximum(y[A].std(0),1e-6)
            gate=_fit_ma_gate(ma_features(x[A],med,ym,ys),pc,truth,f'target_only_{seed}_{pos}',seed);counts['error']+=1
            if not gate['success'] and not gate['fallback']:raise RuntimeError('error gate optimizer did not converge')
            heads={}
            for head in ['low','high']:
                mask,hy=c.build_head_mask(truth,pc,med[:,2],qc[A],head);support=c.support_summary(hy,ids[A][mask])
                supportrows.append(dict(seed=seed,position=pos,head=head,n=support['n'],class0=support['class0_n'],class1=support['class1_n'],status=support['status']))
                if support['status']!='SUPPORTED':heads[head]=None;continue
                z,pre=c.build_features(x[A,1][mask],med[:,2][mask],np.ones(mask.sum(),bool));w=c.class_balanced_weights(hy)
                opt=minimize(c.weighted_bce_objective,np.zeros(3),args=(z,hy,w,.1),jac=c.weighted_bce_gradient,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9,'gtol':1e-9});counts['dual']+=1
                assert opt.success and np.isfinite(opt.x).all()
                heads[head]=dict(theta=opt.x,pre=pre,support=support)
            foldrows.append(dict(seed=seed,position=pos,training_n=len(A),heldout_n=len(B),alpha=best,inner_MA_MAE=losses[best],error_fallback=bool(gate['fallback'])))
            if len(B):
                assert not set(groups[A])&set(groups[B]) and set(A)|set(B)==set(range(60))
                pb=ridge.predict(model,x[B]);cb=ma_class(pb[:,2]);bs=score_ma_gate(gate,ma_features(x[B],pb,ym,ys),cb);ds=head_scores(heads,x[B],pb)
                for j,i in enumerate(B): rows.append(dict(record_id=str(ids[i]),pred_class=int(cb[j]),truth_class=int(ma_class(y[[i],2])[0]),base_score=float(bs[j]) if qc[i] and np.isfinite(bs[j]) else None,dual_score=float(ds[j]) if qc[i] and cb[j]==0 and np.isfinite(ds[j]) else None))
            else:final=dict(predictor=model,error=gate,heads=heads,ym=ym,ys=ys,pool=pool,test=np.array(o['test']),seed=seed)
        assert len(rows)==60 and len({r['record_id'] for r in rows})==60
        final['threshold']=choose(rows)
        priorpath=G/'target_ridge'/f'HZ_to_SA_{seed}.joblib';inputs[str(priorpath)]=sha(priorpath);prior=joblib.load(priorpath)
        for key in ['xm','xs','ym','ys','coef','alpha']:assert np.allclose(final['predictor'][key],prior[key],atol=1e-12),key
        path=ROOT/f'model_{seed}.joblib';joblib.dump(final,path,compress=3);models[str(path)]=sha(path)
        summaries.append(dict(seed=seed,**final['threshold']))
        print('FROZEN',seed,final['threshold'],flush=True)
    assert counts['ridge']==1040 and counts['error']==80 and counts['dual']<=160
    assert all(sha(p)==h for p,h in inputs.items())
    table('development_folds.csv',foldrows);table('development_heads.csv',supportrows);table('development_working_points.csv',summaries)
    write('FREEZE.json',dict(inputs=inputs,models=models,fit_counts=counts,test_evaluated=False,checks=testchecks,previous_target_predictors_identical=True,shared_comparator=True))

def evaluate():
    assert not (ROOT/'RESULTS.json').exists()
    freeze=json.loads((ROOT/'FREEZE.json').read_text());assert all(sha(p)==h for p,h in {**freeze['inputs'],**freeze['models']}.items())
    D=dict(np.load(DATA,allow_pickle=False));pred=[];release=[];idx=None
    for s in SEEDS:
        m=joblib.load(ROOT/f'model_{s}.joblib');I=m['test']
        if idx is None:idx=I
        assert np.array_equal(I,idx)
        med=ridge.predict(m['predictor'],D['x'][I]);pc=ma_class(med[:,2]);bs=score_ma_gate(m['error'],ma_features(D['x'][I],med,m['ym'],m['ys']),pc);ds=head_scores(m['heads'],D['x'][I],med)
        pred.append(pc);release.append(action(m['threshold'],pc,bs,ds,D['input_qc'][I]))
    truth=ma_class(D['medical'][idx,2]);one=np.ones((20,len(idx)));arms={'Target-only Ridge':(np.array(pred),np.array(release))};evalhash={}
    for d in DIRS:
        path=EXT/f'{d}.npz';evalhash[str(path)]=sha(path);z=dict(np.load(path,allow_pickle=False))
        assert np.array_equal(z['test_ids'].astype(str),D['ids'][idx].astype(str)) and np.array_equal(z['seed'],SEEDS)
        arms[d]=(z['final_ma_class'],z['unified_action'].astype(bool))
    sys.path.insert(0,str(G/'analysis'));from round1 import bootstrap
    stat=bootstrap(D['groups'][idx]);summaries={};arrays={};seedrows=[];destinations=[]
    for name,(p,a) in arms.items():
        good=a&(p==truth);bad=a&~good;low=a&(p==0)&(truth==-1);high=a&(p==0)&(truth==1);within=a&(p==0)
        arr=dict(correct=good,incorrect=bad,unreported=~a,coverage=a,low_to_within=low,high_to_within=high,correct_low=a&(p==-1)&(truth==-1),correct_high=a&(p==1)&(truth==1))
        assert np.all(good.astype(int)+bad+(~a)==1);arrays[name]=arr
        summaries[name]=dict(counts={k:int(v.sum()) for k,v in arr.items()},yields={k:stat(v,one) for k,v in arr.items()},agreement=stat(good,a),within_contamination=stat(low|high,within),zero_report_seeds=int((a.sum(1)==0).sum()),within_reports=int(within.sum()),low_support=int((truth==-1).sum()*20),high_support=int((truth==1).sum()*20))
        for k,s in enumerate(SEEDS):
            n=int(a[k].sum());g=int(good[k].sum());seedrows.append(dict(method=name,seed=s,**{key:int(v[k].sum()) for key,v in arr.items()},agreement=g/n if n else None))
        for t in MA_CLASSES:
            for q in [-1,0,1,9]:
                mask=(truth==t);report=np.where(a,p,9);n=int(((report==q)&mask).sum());N=int(mask.sum()*20)
                destinations.append(dict(method=name,true_state=t,report_state=q,n=n,N=N,proportion=n/N if N else None))
    contrasts={};pairedseeds=[]
    for d in DIRS:
        contrasts[d]={k:stat(arrays['Target-only Ridge'][k].astype(int)-arrays[d][k].astype(int),one) for k in arrays[d]}
        for j,s in enumerate(SEEDS):pairedseeds.append(dict(comparison='Target-only minus '+d,seed=s,**{k:int(arrays['Target-only Ridge'][k][j].sum()-arrays[d][k][j].sum()) for k in arrays[d]}))
    assert all(sha(p)==h for p,h in {**freeze['inputs'],**freeze['models'],**evalhash}.items())
    # The frozen reference action counts must reproduce the manuscript.
    assert [summaries[d]['counts']['coverage'] for d in DIRS]==[536,664]
    assert [summaries[d]['counts']['correct'] for d in DIRS]==[485,620]
    table('per_seed.csv',seedrows);table('destinations.csv',destinations);table('paired_seeds.csv',pairedseeds)
    write('RESULTS.json',dict(records=len(idx),groups=len(np.unique(D['groups'][idx])),seeds=20,arms=summaries,target_only_minus_reference=contrasts,reference_actions_hashes=evalhash,checks=dict(inputs_unchanged=True,reference_counts_reproduced=True,new_thresholds_selected_on_test=False)))
    print(json.dumps({k:{'counts':v['counts'],'agreement':v['agreement'],'zero_report_seeds':v['zero_report_seeds']} for k,v in summaries.items()},indent=2))

if __name__=='__main__':
    {'check':lambda:print(json.dumps(checks(),default=str)), 'train':train,'evaluate':evaluate}[sys.argv[1]]()
