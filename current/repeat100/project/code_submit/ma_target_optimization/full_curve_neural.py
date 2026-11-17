"""Small complete900-point CNN with explicit low/high inward-error costs.

Five-seed nested development screen. Original test remains unavailable. Checkpoint
and loss selection uses inner folds; training metrics diagnose memorization.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .full_curve_cost_sensitive import load_waveforms
from .density_reweight_mapping import label_weights
from .feature_combination_audit import detailed
from .metrics import selection_key
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

LOSSES = ['mse', 'weighted_mse', 'residual4', 'boundary16']
ARMS = ['target_only', 'HB_to_SA']
CHECKPOINTS = [50, 150, 300]


class DeterministicPool(nn.Module):
    def forward(self,x):
        n=x.shape[-1]
        return torch.stack([x[:,:,(i*n)//10:((i+1)*n+9)//10].mean(-1) for i in range(10)],dim=-1)


class WaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(2,8,9,stride=3,padding=4),nn.GELU(),
            nn.Conv1d(8,12,9,stride=3,padding=4),nn.GELU(),
            nn.Conv1d(12,16,5,stride=3,padding=2),nn.GELU(),DeterministicPool())
        self.head = nn.Sequential(nn.Linear(160,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,2))

    def forward(self, x, center):
        return self.head(self.encoder(x).flatten(1)).gather(1,center[:,None])[:,0]


def objective(pred, y, w, kind):
    residual = pred-y
    loss = residual.square()
    low, high = y < -1, y > 1
    if kind == 'residual4':
        inward = (low & (residual>0)) | (high & (residual<0))
        loss = loss*(1+3*inward)
    elif kind == 'boundary16':
        normal = ~low & ~high
        loss = loss+16*(torch.relu(pred+1.05).square()*low+torch.relu(1.05-pred).square()*high)
        loss = loss+(torch.relu(-1-pred).square()+torch.relu(pred-1).square())*normal
    elif kind not in ['mse','weighted_mse']:
        raise ValueError(kind)
    return (w*loss).mean()


def prepare(data, curves, mask, a, b, arm, loss):
    source = data['source_rows']['HB_to_SA'] if arm == 'HB_to_SA' else np.array([],int)
    tr = np.r_[source,a]
    assert not set(data['groups'][tr]).intersection(data['groups'][b])
    values = curves[tr,:900][mask[tr]]
    mean = float(values.mean())
    scale = max(float(values.std()),1e-6)
    # Shared physical scale across centers, learned only on training observations.
    # Padding is masked to zero after scaling; the second channel carries the mask.
    def transform(ix):
        v = np.where(mask[ix],(curves[ix,:900]-mean)/scale,0)
        return np.stack([v,mask[ix].astype(float)],axis=1)
    y = data['y'][tr,2]
    ns = len(source)
    w = np.ones(len(tr))
    groups = [np.arange(ns),np.arange(ns,len(tr))] if ns else [np.arange(len(tr))]
    for group in groups:
        w[group] = (label_weights(y[group],'class_inverse',20.)[0] if loss!='mse' else 1.)/len(group)
    w /= w.mean()
    center = np.r_[np.ones(ns,int),np.zeros(len(a),int)]
    info = dict(mean=mean,scale=scale,n_source=ns,n_target=len(a),
        class_counts=[int((y<50).sum()),int(((y>=50)&(y<=70)).sum()),int((y>70).sum())])
    return transform(tr),transform(b),(y-60)/10,w,center,tr,info


def train(data,curves,mask,a,b,arm,kind,seed,checkpoints=CHECKPOINTS,device='cuda'):
    tx,qx,y,w,center,tr,scaling = prepare(data,curves,mask,a,b,arm,kind)
    torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed_all(seed)
    model = WaveNet().to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    to = lambda v,dtype=torch.float32:torch.as_tensor(v,dtype=dtype,device=device)
    tx,qx,y,w,center = to(tx),to(qx),to(y),to(w),to(center,torch.long)
    qc = torch.zeros(len(b),dtype=torch.long,device=device)
    predictions = {}
    diagnostics = {}
    for epoch in range(1,max(checkpoints)+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        p = model(tx,center)
        loss = objective(p,y,w,kind)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite neural loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),10.)
        optim.step()
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                predictions[epoch] = np.maximum(60+10*model(qx,qc).cpu().numpy(),0)
                trainp = np.maximum(60+10*model(tx,center).cpu().numpy(),0)
            target = np.arange(len(tr)) >= len(tr)-len(a)
            diagnostics[epoch] = dict(loss=float(loss.detach().cpu()),
                training_target_metrics=detailed(data['y'][tr[target],2],trainp[target]))
    digest = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    return predictions,dict(scaling=scaling,checkpoints=diagnostics,parameter_sha256=digest,
                            parameter_count=sum(p.numel() for p in model.parameters()))


def run(seeds,name,smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data,parts,extra,reps,mask,hashes=load_waveforms(seeds)
    curves=reps['raw900']
    out=ROOT/'outputs/uteg_mapping_target_runs'/name
    (out/'private').mkdir(parents=True,exist_ok=True)
    checkpoints=[5] if smoke else CHECKPOINTS
    protocol=dict(scope='C100 same60 outer target queries, extra40 training-only, frozen3x3 grouped folds; original201 masked.',
        seeds=list(parts),arms=ARMS,losses=LOSSES,checkpoints=checkpoints,device=device,torch=torch.__version__,
        input='All900 baseline-corrected first-channel values plus observed-point mask. No temporal binning before network, no per-record peak scaling. Global training-only observed-voltage mean/SD shared across centers; missing source tail masked zero.',
        architecture='Conv2to8(k9,s3),Conv8to12(k9,s3),Conv12to16(k5,s3),GELU,10 adaptive-bin means implemented as deterministic slices/reductions,linear160to16,GELU,dropout.1,2 center-specific scalar MA heads. No4parameter input.',
        training='Fullbatch AdamW lr.003 decay.01 clip10; identical initial seed across losses/arms for each fold. Source/target equal base domain mass. Class inverse weights computed within each training domain, cap20; MSE control uniform.',
        loss_definitions='MSE, inverse-class MSE, inward-residual cost4, correct-side boundary hinge cost16. Normal outside-range hinge cost1. Continuous MA output; no class projection or withholding.',
        selection='Inner grouped folds choose arm/loss/epoch by MSE and safety ordering only; outer fixed candidates exploratory.',
        extension='Five-seed screen extends a promising configuration only if agreement>80,FNR<37%,low-recall>=25%, or R2>.35, or joint target passes. This is a development resource rule, not an external validation claim.',
        input_hashes=hashes,code_sha256=sha256(__file__),test_evaluated=False,manuscript_changed=False,
        previous_turn='Progress: complete900-point matched comparison established higher MA R2 than4parameters, but low recall0; full-curve directional kernel now provides independent cost-control comparison.')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=parts[1000]['pool'];pos={int(v):j for j,v in enumerate(pool)}
    grid=[dict(arm=a,loss=l,epoch=e) for a in ARMS for l in LOSSES for e in checkpoints]
    full=np.full((len(grid),seeds,len(pool)),np.nan)
    selected={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']}
    ledger=[];training=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            a0,b=fold['A'],fold['B'];extra0=extra[~np.isin(data['groups'][extra],data['groups'][b])]
            a=np.r_[a0,extra0];ix=[pos[int(v)] for v in b];ip={int(v):j for j,v in enumerate(a0)}
            inner=np.full((len(grid),len(a0)),np.nan)
            number=0
            for arm in ARMS:
                for kind in LOSSES:
                    key=f'{seed}_{fi}_{arm}_{kind}'
                    cache=out/f'private/{key}.npz';auditfile=out/f'private/{key}.json'
                    if cache.exists() and auditfile.exists():
                        z=np.load(cache);pp={e:z[f'p{e}'] for e in checkpoints};inp=z['inner'];info=json.loads(auditfile.read_text())
                    else:
                        pp,info=train(data,curves,mask,a,b,arm,kind,seed*100+fi,checkpoints,device)
                        inp=np.full((len(checkpoints),len(a0)),np.nan)
                        for ii,innerfold in enumerate(fold['inner']):
                            tr,va=innerfold['T'],innerfold['V'];add=extra0[~np.isin(data['groups'][extra0],data['groups'][va])];tr=np.r_[tr,add]
                            ipred,_=train(data,curves,mask,tr,va,arm,kind,seed*100+10+fi*3+ii,checkpoints,device)
                            for ei,e in enumerate(checkpoints):
                                inp[ei,[ip[int(v)] for v in va]]=ipred[e]
                        np.savez_compressed(cache,inner=inp,**{f'p{e}':p for e,p in pp.items()})
                        auditfile.write_text(json.dumps(info,indent=2),encoding='utf-8')
                    for ei,e in enumerate(checkpoints):
                        full[number,si,ix]=pp[e]
                        inner[number]=inp[ei]
                        training.append(dict(seed=seed,fold=fi,candidate=grid[number],metrics=info['checkpoints'][str(e) if str(e) in info['checkpoints'] else e]['training_target_metrics']))
                        number+=1
                    if smoke:
                        poison=dict(data);poison['y']=data['y'].copy();poison['y'][b]=np.nan
                        changed=curves.copy();changed[b,:900]+=100
                        _,info2=train(poison,changed,mask,a,b,arm,kind,seed*100+fi,checkpoints,device)
                        assert info2==info
                        (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True,finite_predictions=True,parameter_count=info['parameter_count']),indent=2))
                        print('SMOKE_COMPLETE',time.time()-begin,flush=True)
                        return
            assert np.isfinite(inner).all()
            y=data['y'][a0,2];mm=[detailed(y,p) for p in inner]
            best={'mse':int(np.argmin(np.mean((inner-y)**2,axis=1))),
                  'safety':min(range(len(grid)),key=lambda j:(selection_key(mm[j]),j))}
            for objective,j in best.items():
                selected[objective][si,ix]=full[j,si,ix]
                ledger.append(dict(seed=seed,fold=fi,objective=objective,candidate=grid[j],inner_metrics=mm[j]))
            print('CNN_FOLD',seed,fi,'seconds',round(time.time()-begin,1),flush=True)
    y=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    result=[dict(candidate=c,metrics=detailed(y,p)) for c,p in zip(grid,full)]
    nested=[dict(objective=k,metrics=detailed(y,p)) for k,p in selected.items()]
    for z,p in zip(result+nested,list(full)+list(selected.values())):
        a=np.where(y<50,0,np.where(y>70,2,1));b=np.where(p<50,0,np.where(p>70,2,1))
        assert [[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]==z['metrics']['confusion']
        assert abs(z['metrics']['r2']-(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)))<1e-12
    for path,digest in hashes.items():
        assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=full,reference=y[0],pool=pool,seeds=list(parts),**selected)
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    (out/'private/TRAINING_METRICS.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
    promising=[]
    for z in result:
        m=z['metrics']
        if m['mapping_target_pass'] or (m['agreement']>.8 and m['fnr']<.37 and m['low_recall']>=.25) or m['r2']>.35:
            promising.append(z)
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=result,nested=nested,promising=promising,
        checks=dict(input_hashes_unchanged=True,independent_metrics_verified=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--output',default='full_curve_neural_screen_20261102');parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    with threadpool_limits(limits=1):
        run(a.seeds,a.output,a.smoke)
