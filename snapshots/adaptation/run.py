from pathlib import Path
import json, hashlib, csv, time, sys
import numpy as np
import joblib

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
OLD=OUT/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
DATA=OUT/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
DIRS=['HZ_to_SA','HZ+HB_to_SA']; SEEDS=list(range(1000,1020))
ARMS=['No adaptation','Per-parameter affine','Joint linear','Current complete adaptation']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): (ROOT/p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def table(p,rows):
    with (ROOT/p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def project(x):
    y=np.array(x,float,copy=True); y[...,:3]=np.maximum(y[...,:3],0);y[...,3]=np.clip(y[...,3],0,89.999999);return y
def raw(m,x):return project(m['ym']+m['ys']*(((x-m['xm'])/m['xs'])@m['coef']+m['intercept']))
def modelpath(d,s):return OLD/'models'/d/'target15'/f'SAHZU_CFULL_seed{s}.joblib'
def fit(p,y,m,arm):
    scale=np.maximum(m['ys'],1e-6);v=(project(p)-m['ym'])/scale;u=(y-m['ym'])/scale
    vm=v.mean(0);um=u.mean(0);vc=v-vm;uc=u-um
    if arm=='affine':
        a=np.array([np.linalg.lstsq(vc[:,j,None],uc[:,j],rcond=None)[0][0] for j in range(4)])
        M=np.diag(a)
    else:M=np.linalg.solve(vc.T@vc+10*np.eye(4),vc.T@uc+10*np.eye(4))
    return dict(matrix=M,vbar=vm,ubar=um,mu=m['ym'],scale=scale)
def pred(m,p):return project(m['mu']+m['scale']*(m['ubar']+((project(p)-m['mu'])/m['scale']-m['vbar'])@m['matrix']))
def cls(x):return np.where(x<50,-1,np.where(x>70,1,0))
def train():
    assert not (ROOT/'FROZEN.json').exists()
    D=dict(np.load(DATA,allow_pickle=False)); hashes={str(DATA):sha(DATA)};counts=[];models={}
    common=None
    for d in DIRS:
        for s in SEEDS:
            path=modelpath(d,s);hashes[str(path)]=sha(path);o=joblib.load(path)
            A=o['positions'][3]['A'];test=o['test'];src=o['source_model']
            assert len(A)==60 and set(A)==set(o['pool']) and len(test)==201
            assert not set(A)&set(test) and not set(o['source_rows'])&set(A+test)
            assert not set(D['groups'][A])&set(D['groups'][test])
            if common is None:common=test
            assert test==common
            assert src['name']=='RIDGE-10' and src['success']
            p=raw(src,D['x'][A]);y=D['medical'][A];assert np.isfinite(p).all() and np.isfinite(y).all()
            for arm in ['affine','joint']:
                q=fit(p,y,src,arm);assert np.isfinite(q['matrix']).all()
                path=ROOT/f'{d}_{s}_{arm}.joblib';joblib.dump(q,path);models[path.name]=sha(path)
                counts.append(dict(direction=d,seed=s,arm=arm,records=len(A),regression_solves=4 if arm=='affine' else 1))
    assert len(counts)==80
    table('fit_ledger.csv',counts)
    write('FROZEN.json',dict(protocol_sha=sha(ROOT/'PROTOCOL.md'),script_sha=sha(__file__),inputs=hashes,models=models,procedures=80,regression_solves=200,test_evaluated=False))
    print('Frozen: 80 procedures / 200 regression solves',flush=True)
def evaluate():
    fr=json.loads((ROOT/'FROZEN.json').read_text());assert sha(__file__)==fr['script_sha']
    assert sha(ROOT/'PROTOCOL.md')==fr['protocol_sha']
    assert all(sha(p)==h for p,h in fr['inputs'].items())
    assert all(sha(ROOT/p)==h for p,h in fr['models'].items())
    D=dict(np.load(DATA,allow_pickle=False));summary=[];seedrows=[];deltas=[];records=[];checks=[]
    for d in DIRS:
        history=OLD/'eval/predictions'/f'{d}.npz';z=dict(np.load(history,allow_pickle=False))
        idx=z['test_global_index'];Y=D['medical'][idx];truth=cls(Y[:,2]);g=D['groups'][idx];valid=z['input_qc'].astype(bool)
        assert valid.all() and np.isfinite(Y).all() and np.array_equal(z['seed'],SEEDS)
        assert np.array_equal(z['test_ids'],D['ids'][idx])
        new=[];err=0
        for k,s in enumerate(SEEDS):
            o=joblib.load(modelpath(d,s));p=raw(o['source_model'],D['x'][idx]);err=max(err,float(abs(p-z['raw_med4'][k]).max()))
            # Reproduce cached Huber intermediate to check historical stage identity.
            ca=o['cache'][o['positions'][3]['base']]['cal']
            hp=project(ca['mu']+ca['scale']*(ca['ubar']+((p-ca['mu'])/ca['scale']-ca['vbar'])@ca['matrix']+ca['intercept']))
            assert np.allclose(hp,z['pure_huber_med4'][k],atol=1e-8)
            assert z['source_adapter'][k]==o['final_selection']['adapter']
            new.append([p,pred(joblib.load(ROOT/f'{d}_{s}_affine.joblib'),p),pred(joblib.load(ROOT/f'{d}_{s}_joint.joblib'),p),z['final_med4'][k]])
        assert err<1e-8
        P=np.array(new).transpose(1,0,2,3);assert np.isfinite(P).all()
        np.savez_compressed(ROOT/f'{d}_predictions.npz',predictions=P,arms=ARMS,seed=SEEDS,ids=z['test_ids'],groups=g,reference=Y)
        ug=np.unique(g); ix=np.random.default_rng(20261029).integers(0,len(ug),(5000,len(ug)))
        def stat(num,den):
            a=np.asarray(num,float);b=np.broadcast_to(den,a.shape).astype(float)
            aa=np.array([a[:,g==v].sum() for v in ug]);bb=np.array([b[:,g==v].sum() for v in ug]);ds=bb[ix].sum(1);ok=ds>0
            point=float(aa.sum()/bb.sum()) if bb.sum() else None
            ci=np.quantile(aa[ix].sum(1)[ok]/ds[ok],[.025,.975]).tolist() if ok.any() else [None,None]
            return point,*ci
        maes=[]
        for a,name in enumerate(ARMS):
            pc=cls(P[a,:,:,2]);E=abs(P[a]-Y);maes.append(E)
            nums={'agreement':pc==truth,'low_to_within':(truth==-1)&(pc==0),'high_to_within':(truth==1)&(pc==0),'normal_to_abnormal':(truth==0)&(pc!=0),'low_correct':(truth==-1)&(pc==-1),'high_correct':(truth==1)&(pc==1),'low_high_confusion':(truth!=0)&(pc==-truth)}
            denoms={key:np.broadcast_to(truth==(-1 if key.startswith('low_') else 1),pc.shape) if key in ['low_to_within','low_correct','high_to_within','high_correct'] else np.broadcast_to(truth==0,pc.shape) if key=='normal_to_abnormal' else np.broadcast_to(truth!=0,pc.shape) if key=='low_high_confusion' else np.ones(pc.shape,bool) for key in nums}
            row=dict(direction=d,arm=name,records=len(idx),groups=len(ug),seeds=20,events=pc.size)
            for j,key in enumerate(['R','K','MA','alpha_R']):
                v=stat(E[:,:,j],np.ones(pc.shape));row.update({key+'_MAE':v[0],key+'_lo':v[1],key+'_hi':v[2]})
            for key,num in nums.items():
                v=stat(num,denoms[key]);row.update({key:v[0],key+'_lo':v[1],key+'_hi':v[2],key+'_n':int(num.sum()),key+'_N':int(denoms[key].sum())})
            summary.append(row)
            for k,s in enumerate(SEEDS):
                sr=dict(direction=d,arm=name,seed=s,agreement=float((pc[k]==truth).mean()))
                for j,key in enumerate(['R','K','MA','alpha_R']):sr[key+'_MAE']=float(E[k,:,j].mean())
                seedrows.append(sr)
                for n,i in enumerate(idx):records.append(dict(direction=d,arm=name,seed=s,record_id=str(D['ids'][i]),group=str(g[n]),true_MA=float(Y[n,2]),pred_MA=float(P[a,k,n,2]),true_state=int(truth[n]),pred_state=int(pc[k,n])))
        for a in range(1,4):
            for j,key in enumerate(['R','K','MA','alpha_R']):
                v=stat(maes[0][:,:,j]-maes[a][:,:,j],np.ones((20,len(idx))))
                deltas.append(dict(direction=d,contrast='No adaptation minus '+ARMS[a],parameter=key,estimate=v[0],lo=v[1],hi=v[2]))
        checks.append(dict(direction=d,raw_reproduction_max_error=err,history_sha=sha(history),records=len(idx),groups=len(ug),low_records=int((truth==-1).sum()),within_records=int((truth==0).sum()),high_records=int((truth==1).sum())))
    table('summary.csv',summary);table('per_seed.csv',seedrows);table('paired_mae_gain.csv',deltas);table('record_actions.csv',records)
    write('QA.json',dict(checks=checks,input_hashes_unchanged=all(sha(p)==h for p,h in fr['inputs'].items()),fit_procedures=80,new_gate_fits=0,test_tuning=False))
    print(json.dumps([{k:r[k] for k in ['direction','arm','MA_MAE','agreement','low_to_within_n','low_to_within_N','high_to_within_n','high_to_within_N','normal_to_abnormal']} for r in summary],indent=2))
if __name__=='__main__':
    {'train':train,'evaluate':evaluate}[sys.argv[1]]()
