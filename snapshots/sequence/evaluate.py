"""Frozen-policy scoring and separate reference summary. Does not fit or select."""
from pathlib import Path
import json,csv
import numpy as np,joblib
import run as core
P=core.ROOT

def infer(r,runner,t,c,obj,heads,D,idx):
    src=obj['source_model'];x=D['x'][idx];raw=t.m.predict(src,x)
    ca=obj['cache'][obj['positions'][3]['base']];pure=t.m.calibrated(ca['cal'],raw);ad=obj['gate_models'][3]['adapter']
    final=pure if ad=='H' else t.m.ans['adapt'](ca['adapters'][ad],x,raw,pure,src)
    pc=runner.ma_class(final[:,2]);f=runner.ma_features(x,final,src['ym'],src['ys'])
    bs=runner.score_ma_gate(obj['gate_models'][3]['model'],f,pc)
    ds=np.maximum(core.score_head(c,heads['low'],x[:,1],final[:,2]),core.score_head(c,heads['high'],x[:,1],final[:,2]))
    valid=D['input_qc'][idx]&np.isfinite(final).all(1)&np.isin(pc,[-1,0,1])
    return final,pc,bs,ds,valid

def actions(pc,bs,ds,valid,th):
    a=np.zeros(len(pc),bool)
    if not th['all_withhold']:
        if th['normal'] is not None:a |= valid&(pc==0)&np.isfinite(ds)&(ds<=th['normal'])
        if th['tail'] is not None:a |= valid&(pc!=0)&np.isfinite(bs)&(bs<=th['tail'])
    return a

def verify900(r,runner,t,c):
    D=core.data_at(900)
    checkpoints={}
    for path in (core.OUT/'uteg_ridge_dual_head_runner_20261026_173951/models').glob('*.json'):
        wrapper=core.read(path);v=wrapper['payload'];checkpoints[v['slot_key']]=v
    entries=core.read(core.OUT/'uteg_ridge_dual_head_unified_selector_20261026_190142/THRESHOLDS_FREEZE.json')['entries']
    idx={(e['direction'],e['seed']):e for e in entries};cases=0
    for d in core.DIRECTIONS:
        z=np.load(core.EXT/'actions'/f'{d}.npz')
        for j,s in enumerate(core.SEEDS):
            obj=joblib.load(core.old_obj(d,s));heads={}
            for h in ['low','high']:
                v=checkpoints.get(f'{d}|s{s}|p3|{h}')
                if v is None:heads[h]=None
                else:
                    heads[h]={'theta':np.asarray(v['theta']),'pre':{k:np.asarray(x) for k,x in v['preprocess'].items()}}
            med,pc,bs,ds,valid=infer(r,runner,t,c,obj,heads,D,obj['test'])
            th=idx[(d,s)];tau={'normal':th['normal_threshold'],'tail':th['common_tail_threshold'],'all_withhold':th['common_tail_status']=='NO_FEASIBLE_COMMON_TAIL_THRESHOLD'}
            a=actions(pc,bs,ds,valid,tau)
            assert np.allclose(med[:,2],z['final_ma'][j],equal_nan=True)
            assert np.array_equal(pc,z['final_ma_class'][j]) and np.array_equal(a,z['unified_action'][j]),(d,s,'action mismatch')
            assert np.allclose(bs,z['lr_score'][j],equal_nan=True)
            cases+=1
    core.write(P/'T900_REPLAY_CHECK.json',{'status':'PASS','contexts':cases,'fit_units':0,'same_actions':True})

def score():
    freeze=core.read(P/'ALL_MODELS_FROZEN.json');assert freeze['count']==160
    assert all(core.sha(p)==h for p,h in freeze['models'].items())
    r,runner,t,c,sel=core.env(300);verify900(r,runner,t,c)
    for horizon in [300,600]:
        r,runner,t,c,sel=core.env(horizon);D=core.data_at(horizon)
        for d in core.DIRECTIONS:
            out=[]
            for s in core.SEEDS:
                fp=r.ROOT/'frozen'/f'{d}_{s}.json';fr=core.read(fp)
                op=r.ROOT/'models'/d/'target15'/f'{core.TARGET[d]}_CFULL_seed{s}.joblib';dp=r.ROOT/'dual'/f'{d}_{s}.joblib'
                assert core.sha(op)==fr['target_sha'] and core.sha(dp)==fr['dual_sha']
                obj=joblib.load(op);heads=joblib.load(dp)['heads'][3];ii=obj['test']
                med,pc,bs,ds,valid=infer(r,runner,t,c,obj,heads,D,ii)
                a=actions(pc,bs,ds,valid,fr['threshold']);out.append((med,pc,bs,ds,valid,a))
            dest=P/'actions'/f'T{horizon}';dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{d}.npz',seed=core.SEEDS,ids=D['ids'][ii],groups=D['groups'][ii],index=ii,medical=np.stack([o[0] for o in out]),pred=np.stack([o[1] for o in out]),base_score=np.stack([o[2] for o in out]),dual_score=np.stack([o[3] for o in out]),valid=np.stack([o[4] for o in out]),action=np.stack([o[5] for o in out]))
    files=list((P/'actions').rglob('*.npz'));assert len(files)==8
    core.write(P/'ACTIONS_FROZEN.json',{'sources':{str(p):core.sha(p) for p in files},'count':8,'fit_units':0,'test_reference_evaluation':False})

def metrics(a,p,y):
    good=a&(p==y[None,:]);bad=a&~good;total=a.size
    return {'total':total,'released':int(a.sum()),'correct':int(good.sum()),'wrong':int(bad.sum()),'unreported':int((~a).sum()),'agreement':float(good.sum()/a.sum()) if a.any() else None,'coverage':float(a.mean()),'correct_yield':float(good.mean()),'wrong_yield':float(bad.mean()),'low_to_within':int((a&(y[None,:]==-1)&(p==0)).sum()),'high_to_within':int((a&(y[None,:]==1)&(p==0)).sum()),'normal_to_tail':int((a&(y[None,:]==0)&(p!=0)).sum()),'low_high_cross':int((a&(y[None,:]!=0)&(p==-y[None,:])).sum())}

def interval(values,groups,adjust=1):
    unique=np.unique(groups);w=np.array([(groups==g).sum() for g in unique]);v=np.array([values[groups==g].sum() for g in unique])
    rng=np.random.default_rng(20261026);ix=rng.integers(0,len(unique),(20000,len(unique)))
    b=v[ix].sum(1)/w[ix].sum(1);return np.quantile(b,[.025/adjust,1-.025/adjust]).tolist()

def summarize():
    fr=core.read(P/'ACTIONS_FROZEN.json');assert all(core.sha(p)==h for p,h in fr['sources'].items())
    D=core.data_at(900);out={};rows=[]
    for d in core.DIRECTIONS:
        seq=[]
        for h in [300,600]:
            z=np.load(P/'actions'/f'T{h}'/f'{d}.npz');seq.append((z['action'],z['pred']))
        old=np.load(core.EXT/'actions'/f'{d}.npz');assert np.array_equal(z['ids'].astype(str),old['test_ids'].astype(str))
        seq.append((old['unified_action'].astype(bool),old['final_ma_class']))
        ii=z['index'];yy=D['medical'][ii,2];y=np.where(yy<50,-1,np.where(yy>70,1,0));groups=z['groups']
        ever=np.zeros_like(seq[0][0]);first=np.full_like(seq[0][1],9);firsttime=np.full_like(first,-1,dtype=int);cum={};point={};gg=[];bb=[]
        for h,(a,p) in zip([300,600,900],seq):
            point[str(h)]=metrics(a,p,y);new=a&~ever;first[new]=p[new];firsttime[new]=h;ever|=a;cum[str(h)]=metrics(ever,first,y)
            gg.append((ever&(first==y)).mean(0));bb.append((ever&(first!=y)).mean(0))
            for j,s in enumerate(core.SEEDS):
                row=metrics(ever[j:j+1],first[j:j+1],y);rows.append(dict(direction=d,seed=s,horizon=h,**row))
        delta={}
        for name,v in [('correct',gg[-1]-gg[0]),('wrong',bb[-1]-bb[0])]:delta[name]={'estimate':float(v.mean()),'ci95':interval(v,groups),'ci95_bonferroni8':interval(v,groups,8)}
        out[d]={'n_records':len(y),'n_groups':len(np.unique(groups)),'seeds':20,'pointwise':point,'cumulative':cum,'delta15_minus5':delta}
        np.savez_compressed(P/'actions'/f'{d}_first_report.npz',ids=z['ids'],groups=groups,truth=y,first_class=first,first_time=firsttime,reported=ever)
    core.write(P/'RESULTS.json',out)
    with (P/'SEED_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    core.write(P/'STATUS.json',{'stage':'EVALUATED','directions':4,'seeds':20,'horizons':[300,600,900],'test_tuning':False})
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    import sys
    {'score':score,'summarize':summarize}[sys.argv[1]]()
