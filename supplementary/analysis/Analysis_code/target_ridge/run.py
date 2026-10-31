from pathlib import Path
import sys, json, hashlib, csv
import numpy as np
import joblib
import os

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ['UTEG_ARTIFACT_ROOT']).resolve()
OLD=OUT/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
DATA=OUT/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
DIRS=['HZ_to_SA','HZ+HB_to_SA']; SEEDS=list(range(1000,1020)); ALPHAS=[.1,1,10,100]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,o): (ROOT/n).write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def table(n,rows):
    with (ROOT/n).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def project(y):
    y=y.copy();y[...,:3]=np.maximum(y[...,:3],0);y[...,3]=np.clip(y[...,3],0,89.999999);return y
def fit(x,y,a):
    xm=x.mean(0);xs=np.maximum(x.std(0),1e-6);ym=y.mean(0);ys=np.maximum(y.std(0),1e-6)
    xx=(x-xm)/xs; yy=(y-ym)/ys
    c=np.linalg.solve(xx.T@xx+a*np.eye(x.shape[1]),xx.T@yy)
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,coef=c,alpha=a)
def predict(m,x): return project(m['ym']+m['ys']*((x-m['xm'])/m['xs']@m['coef']))
def path(d,s): return OLD/'models'/d/'target15'/f'SAHZU_CFULL_seed{s}.joblib'
def train():
    assert not (ROOT/'FREEZE.json').exists()
    D=dict(np.load(DATA,allow_pickle=False)); hashes={str(DATA):sha(DATA),str(ROOT/'PROTOCOL.md'):sha(ROOT/'PROTOCOL.md'),str(Path(__file__)):sha(__file__)}
    models={};rows=[]
    for d in DIRS:
        for s in SEEDS:
            p=path(d,s);hashes[str(p)]=sha(p);o=joblib.load(p);pool=np.array(o['pool']);test=np.array(o['test'])
            assert len(pool)==60 and len(test)==201 and not set(D['groups'][pool])&set(D['groups'][test])
            folds=[o['positions'][k] for k in range(3)]
            held=[i for f in folds for i in f['B']];assert sorted(held)==sorted(pool)
            scores=[]
            for a in ALPHAS:
                errors=[]
                for f in folds:
                    A=f['A']; B=f['B']
                    assert set(A)|set(B)==set(pool) and not set(D['groups'][A])&set(D['groups'][B])
                    m=fit(D['x'][A],D['medical'][A],a)
                    errors.extend(abs(predict(m,D['x'][B])[:,2]-D['medical'][B,2]))
                scores.append(float(np.mean(errors)))
                rows.append(dict(direction=d,seed=s,alpha=a,development_MA_MAE=scores[-1]))
            best=min(range(4),key=lambda i:(scores[i],-ALPHAS[i])); m=fit(D['x'][pool],D['medical'][pool],ALPHAS[best])
            m.update(pool=pool,test=test,seed=s,direction=d,development_scores=scores)
            dest=ROOT/f'{d}_{s}.joblib';joblib.dump(m,dest);models[str(dest)]=sha(dest)
    table('development_selection.csv',rows)
    write('FREEZE.json',dict(inputs=hashes,models=models,final_models=40,regression_solves=520,test_evaluated=False))
    print('Frozen 40 target-only models. No test-label evaluation performed.')

def evaluate():
    assert not (ROOT/'RESULTS.json').exists()
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    assert all(sha(p)==h for p,h in {**freeze['inputs'],**freeze['models']}.items())
    sys.path.insert(0,str(ROOT.parent/'analysis'));from round1 import bootstrap
    D=dict(np.load(DATA,allow_pickle=False));rows=[];destinations=[];summary={}
    for d in DIRS:
        z=dict(np.load(OLD/'eval/predictions'/f'{d}.npz',allow_pickle=False));idx=z['test_global_index'];Y=D['medical'][idx];truth=np.where(Y[:,2]<50,-1,np.where(Y[:,2]>70,1,0))
        assert np.array_equal(z['test_ids'],D['ids'][idx]) and np.array_equal(z['seed'],SEEDS)
        predictions=[];constants=[];alphas=[]
        for s in SEEDS:
            m=joblib.load(ROOT/f'{d}_{s}.joblib');assert np.array_equal(m['test'],idx)
            predictions.append(predict(m,D['x'][idx]));constants.append(np.median(D['medical'][m['pool']],axis=0));alphas.append(m['alpha'])
        pred=np.array(predictions);stat=bootstrap(D['groups'][idx]);one=np.ones((20,201))
        arms={'Target-only Ridge':pred,'Complete adaptation':z['final_med4'],'Target-pool medians':np.broadcast_to(np.array(constants)[:,None,:],pred.shape)}
        results={};errs={}
        for name,P in arms.items():
            E=abs(P-Y);errs[name]=E;pc=np.where(P[:,:,2]<50,-1,np.where(P[:,:,2]>70,1,0))
            results[name]={param:stat(E[:,:,j],one) for j,param in enumerate(['R','K','MA','alpha_R'])}
            results[name]['ungated_agreement']=stat(pc==truth,one)
            for true in [-1,0,1]:
                for report in [-1,0,1]:
                    mask=np.broadcast_to(truth==true,one.shape)
                    destinations.append(dict(direction=d,method=name,true_state=true,predicted_state=report,count=int((mask&(pc==report)).sum()),denominator=int(mask.sum())))
            for k,s in enumerate(SEEDS):
                rows.append(dict(direction=d,method=name,seed=s,alpha=alphas[k] if name=='Target-only Ridge' else None,**{param:float(E[k,:,j].mean()) for j,param in enumerate(['R','K','MA','alpha_R'])}))
        contrasts={}
        for a,b in [('Target-pool medians','Target-only Ridge'),('Target-only Ridge','Complete adaptation')]:
            contrasts[a+' minus '+b]={param:stat(errs[a][:,:,j]-errs[b][:,:,j],one) for j,param in enumerate(['R','K','MA','alpha_R'])}
        summary[d]=dict(records=201,groups=len(np.unique(D['groups'][idx])),seeds=20,selected_alphas=alphas,methods=results,paired_differences=contrasts)
    write('RESULTS.json',summary);table('per_seed.csv',rows);table('class_destinations.csv',destinations)
    write('CHECKS.json',dict(frozen_inputs_unchanged=all(sha(p)==h for p,h in {**freeze['inputs'],**freeze['models']}.items()),no_new_gate=True,test_tuning=False))
    print(json.dumps({d:{'MA':{n:v['MA'] for n,v in r['methods'].items()},'MA_contrasts':{n:v['MA'] for n,v in r['paired_differences'].items()},'alphas':r['selected_alphas']} for d,r in summary.items()},indent=2))

if __name__=='__main__': {'train':train,'evaluate':evaluate}[sys.argv[1]]()
