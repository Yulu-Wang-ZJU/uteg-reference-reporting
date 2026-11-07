"""Nonlinear MA regression with source-center identities and balanced losses."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .models import weights
from .metrics import assess
from .search import ROOT,make_summary
from .representation_search import load_representations
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs,sha256


@dataclass
class CenterTree:
    model: object
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_code: int

    def predict(self,x):
        v=(x-self.target_mean)/self.target_scale
        return np.maximum(self.model.predict(np.c_[v,np.full(len(v),self.target_code)]),0)


def fit(data,A,c):
    source=data['source_rows'].get(c['arm'],np.array([],int))
    idx=np.r_[source,A];center_names=sorted(set(data['centers'][idx]));blocks=[];ys=[];ws=[]
    target_mean=target_scale=target_code=None
    for code,name in enumerate(center_names):
        rows=idx[data['centers'][idx]==name];x=data['x'][rows];y=data['y'][rows,2]
        mean=x.mean(0) if c['center_scale'] else np.zeros(x.shape[1]);scale=np.maximum(x.std(0),1e-6) if c['center_scale'] else np.ones(x.shape[1])
        blocks.append(np.c_[(x-mean)/scale,np.full(len(x),code)]);ys.append(y)
        mass=c['target_mass'] if name=='SAHZU' else 1.
        ws.append(weights(y,c['cap'])*mass/len(y))
        if name=='SAHZU':target_mean,target_scale,target_code=mean,scale,code
    x=np.vstack(blocks);y=np.concatenate(ys);w=np.concatenate(ws);w/=w.mean()
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Non-finite training values')
    estimator=HistGradientBoostingRegressor(loss=c['loss'],max_iter=120,max_leaf_nodes=7,min_samples_leaf=c['leaf'],l2_regularization=1.,learning_rate=.08,categorical_features=[False]*(x.shape[1]-1)+[True],early_stopping=False,random_state=20261101)
    estimator.fit(x,y,sample_weight=w)
    return CenterTree(estimator,target_mean,target_scale,target_code)


def grid():
    ans=[]
    for arm in ['target_only','HZ_to_SA','HZ+HB_to_SA']:
        for rep,scale,loss,leaf,mass,cap in itertools.product(['parameters4','parameters_plus_curve'],[False] if arm=='target_only' else [False,True],['squared_error','absolute_error'],[1,5],[1.] if arm=='target_only' else [4.,16.],[5.,20.]):
            c=dict(family='center_tree',arm=arm,representation=rep,center_scale=scale,loss=loss,leaf=leaf,target_mass=mass,cap=cap)
            c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12];ans.append(c)
    return ans


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_center_tree_v1';out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1005))
    data,parts,hashes=load_development_inputs(ROOT,cfg);matrices,more=load_representations(data,parts);hashes.update(more)
    pool=next(iter(parts.values()))['pool'];pos={int(v):i for i,v in enumerate(pool)};candidates=grid()
    protocol={'candidates':candidates,'seeds':cfg['seeds'],'input_hashes':hashes,'code_hash':sha256(__file__),
              'scope':'development-only nonlinear numeric MA mapping; original 60 target labels; no state-head substitution',
              'center_handling':'known center categorical feature, optional center-specific input scale fitted only on training rows',
              'weighting':'inverse-class frequencies capped within each center; source-center loss mass 1, target mass as specified',
              'test_access':'none; all representations leave test rows NaN'}
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:raise RuntimeError('Changed protocol')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');path=out/'CANDIDATES.jsonl'
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [];done={r['candidate']['id'] for r in records}
    for number,c in enumerate(candidates):
        if c['id'] in done:continue
        start=time.time();data['x']=matrices[c['representation']]
        try:
            preds=[]
            for seed,part in parts.items():
                p=np.full(len(pool),np.nan)
                for fold in part['folds']:
                    A,B=fold['A'],fold['B'];model=fit(data,A,c);p[[pos[int(v)] for v in B]]=model.predict(data['x'][B])
                preds.append(p)
            preds=np.asarray(preds);y=data['y'][pool,2];m=assess(np.broadcast_to(y,preds.shape),preds)
            np.savez_compressed(out/'private'/f"{c['id']}.npz",prediction=preds,reference=y,pool=pool,seeds=cfg['seeds'])
            row={'candidate':c,'status':'complete','metrics':m,'seconds':time.time()-start}
        except Exception as error:row={'candidate':c,'status':'failed','error':str(error)}
        records.append(row)
        with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if (number+1)%10==0 or number+1==len(candidates):
            s=make_summary(records,out);print(json.dumps({'done':number+1,'total':len(candidates),'failed':s['failed'],'feasible':s['feasible_count']}),flush=True)
    return make_summary(records,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
