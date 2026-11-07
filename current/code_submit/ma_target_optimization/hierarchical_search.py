"""Center-aware partial pooling; development-only target evaluation."""
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from .models import Features, weights
from .metrics import states, assess
from .search import ROOT, training_view, make_summary
from code_submit.ma_imbalance_mapping.data_contract import load_development_inputs, sha256


def objective(theta,z,y,w,penalty,boundary,cost):
    pred=60+10*z@theta;err=(pred-y)/10
    value=float(np.mean(w*err**2)+np.sum(penalty*theta**2))
    grad=z.T@(2*w*err)/len(y)+2*penalty*theta
    if boundary:
        logits=(pred[:,None]-[50.,70.])/2.
        target=np.c_[y>=50,y>70]
        bw=w*np.where(states(y)==1,1.,cost);bw/=bw.mean()
        value+=float(boundary*np.mean(bw[:,None]*(np.logaddexp(0,logits)-target*logits)))
        grad+=z.T@(boundary*bw*np.sum(expit(logits)-target,axis=1)*2.5)/len(y)
    return value,grad


@dataclass
class HierarchicalModel:
    transforms: list
    theta: np.ndarray

    def predict(self,x):
        v=np.c_[self.transforms[-1].transform(x),np.ones(len(x))]
        width=v.shape[1]
        return np.maximum(60+10*(v@self.theta[:width]+v@self.theta[-width:]),0)


def fit(train,c):
    center_names=['Huzhou']+(['Hubei'] if c['arm']=='HZ+HB_to_SA' else [])
    sets=[train['source_centers'][name] for name in center_names]+[(train['x'],train['y'])]
    transforms=[Features.fit(x,c['features']) for x,y in sets]
    width=sets[0][0].shape[1]+1;blocks=[];labels=[];importance=[]
    for center,((x,y),transform) in enumerate(zip(sets,transforms)):
        v=np.c_[transform.transform(x),np.ones(len(x))]
        z=np.zeros((len(x),width*(1+len(sets))));z[:,:width]=v;z[:,width*(center+1):width*(center+2)]=v
        blocks.append(z);labels.append(y[:,2])
        # Balance each source center equally; target gets its specified mass.
        importance.append(weights(y[:,2],c['cap'])/len(y)*(c['target_mass'] if center==len(sets)-1 else 1.))
    z=np.vstack(blocks);y=np.concatenate(labels);w=np.concatenate(importance);w/=w.mean()
    penalty=np.full(z.shape[1],c['deviation_penalty']);penalty[:width]=c['shared_penalty'];penalty[width-1]=0
    initial=np.linalg.solve(z.T@(w[:,None]*z)+len(y)*np.diag(penalty)+1e-10*np.eye(z.shape[1]),z.T@(w*(y-60)/10))
    if c['boundary_strength']:
        result=minimize(objective,initial,args=(z,y,w,penalty,c['boundary_strength'],c['abnormal_cost']),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-11,'gtol':1e-8,'maxls':50})
        if not result.success:raise RuntimeError(str(result.message))
        theta=result.x
    else:theta=initial
    return HierarchicalModel(transforms,theta)


def grid():
    result=[]
    for arm,features,dev,cap,mass in itertools.product(('HZ_to_SA','HZ+HB_to_SA'),('linear','loglinear'),(.01,.1,1.,10.),(5.,20.),(1.,4.)):
        result.append(dict(family='hierarchical',arm=arm,features=features,deviation_penalty=dev,shared_penalty=.01,cap=cap,target_mass=mass,boundary_strength=0,abnormal_cost=1.))
    for arm,features,dev,strength,cost in itertools.product(('HZ_to_SA','HZ+HB_to_SA'),('linear','loglinear'),(.1,1.),(1.,4.),(4.,16.)):
        result.append(dict(family='hierarchical',arm=arm,features=features,deviation_penalty=dev,shared_penalty=.01,cap=5.,target_mass=4.,boundary_strength=strength,abnormal_cost=cost))
    for c in result:c['id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    return result


def run():
    out=ROOT/'outputs/uteg_mapping_target_runs/goal_hierarchical_screen_v1';out.mkdir(parents=True,exist_ok=True);(out/'private').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'code_submit/ma_imbalance_mapping/config.json').read_text(encoding='utf-8'));cfg['seeds']=list(range(1000,1005))
    data,partitions,hashes=load_development_inputs(ROOT,cfg);pool=next(iter(partitions.values()))['pool'];pos={int(v):i for i,v in enumerate(pool)}
    data['x'][next(iter(partitions.values()))['test']]=np.nan
    candidates=grid();protocol={'candidates':candidates,'seeds':cfg['seeds'],'input_hashes':hashes,'code_hash':sha256(__file__),'target_budget':60,'scope':'development-only; source and target transforms fit on own training rows; no test selection',
                              'shared_relation':'global coefficients plus penalized center-specific deviations; same MA units and boundaries',
                              'targets':'all-report agreement>80%; FNR<5% and FNR<FPR; subsequent gate coverage>75%'}
    text=json.dumps(protocol,indent=2)
    if (out/'PROTOCOL.json').exists() and (out/'PROTOCOL.json').read_text(encoding='utf-8')!=text:raise RuntimeError('Protocol changed')
    (out/'PROTOCOL.json').write_text(text,encoding='utf-8');path=out/'CANDIDATES.jsonl'
    rows=[json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [];done={r['candidate']['id'] for r in rows}
    for number,c in enumerate(candidates):
        if c['id'] in done:continue
        start=time.time()
        try:
            predictions=[]
            for seed,part in partitions.items():
                pred=np.full(len(pool),np.nan)
                for fold in part['folds']:
                    A,B=fold['A'],fold['B'];model=fit(training_view(data,A),c);pred[[pos[int(v)] for v in B]]=model.predict(data['x'][B])
                predictions.append(pred)
            predictions=np.asarray(predictions);y=data['y'][pool,2];metrics=assess(np.broadcast_to(y,predictions.shape),predictions)
            np.savez_compressed(out/'private'/f"{c['id']}.npz",prediction=predictions,reference=y,pool=pool,seeds=cfg['seeds'])
            row={'candidate':c,'status':'complete','metrics':metrics,'seconds':time.time()-start}
        except Exception as error:row={'candidate':c,'status':'failed','error':str(error),'seconds':time.time()-start}
        with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row)
        if (number+1)%10==0 or number+1==len(candidates):
            s=make_summary(rows,out);print(json.dumps({'done':number+1,'total':len(candidates),'failed':s['failed'],'feasible':s['feasible_count']}),flush=True)
    return make_summary(rows,out)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
