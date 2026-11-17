"""Complete-waveform shared encoder for MA-state and conditional-MA learning.

Conditional outputs belong to their predicted state by construction. Mixture
means are retained as controls; errors are never measured on hidden/clamped
alternative predictions. Query labels cannot fit transforms, experts or rules.
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
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .full_curve_cost_sensitive import load_waveforms
from .full_curve_neural import DeterministicPool,prepare
from .feature_combination_audit import detailed
from .metrics import selection_key,states
from .search import ROOT
from code_submit.ma_imbalance_mapping.data_contract import sha256

ARMS=['target_only','HB_to_SA']
LOSSES=['uniform','balanced','fn4']
EPOCHS=[50,150,300]
POLICIES=['mixture_mean','argmax','fn4','fn16','inner_threshold']


def expert_values(raw):
    # Strict abnormal endpoints prevent float32 sigmoid saturation from
    # reporting a low/high expert as exactly50/70 (the normal category).
    return torch.stack([49.999*torch.sigmoid(raw[:,0]),
                        50+20*torch.sigmoid(raw[:,1]),
                        70.001+10*F.softplus(raw[:,2])],dim=1)


class JointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(2,8,9,stride=3,padding=4),nn.GELU(),
            nn.Conv1d(8,12,9,stride=3,padding=4),nn.GELU(),
            nn.Conv1d(12,16,5,stride=3,padding=2),nn.GELU(),DeterministicPool())
        self.hidden=nn.Sequential(nn.Linear(160,16),nn.GELU(),nn.Dropout(.1))
        self.output=nn.Linear(16,12)
        with torch.no_grad():
            b=self.output.bias.view(2,6)
            b[:,:3]=0
            b[:,3]=float(np.log(45/(49.999-45)))
            b[:,4]=0
            b[:,5]=float(np.log(np.expm1((75-70.001)/10)))

    def forward(self,x,center):
        h=self.hidden(self.encoder(x).flatten(1))
        p=self.output(h).reshape(-1,2,6)
        p=p[torch.arange(len(p),device=p.device),center]
        return p[:,:3],expert_values(p[:,3:])


def objective(logits,experts,y,w,kind):
    label=torch.where(y<50,0,torch.where(y>70,2,1))
    ce=F.cross_entropy(logits,label,reduction='none')
    conditional=experts.gather(1,label[:,None])[:,0]
    regression=((conditional-y)/10).square()
    loss=ce+regression
    if kind=='fn4':
        logp=F.log_softmax(logits,dim=1)
        log_abnormal=torch.logsumexp(logp[:,[0,2]],dim=1)
        loss=loss-4*log_abnormal*(label!=1)
    elif kind not in ['uniform','balanced']:
        raise ValueError(kind)
    return torch.mean(w*loss)


def decode(prob,experts,policy,threshold=None):
    prob,experts=np.asarray(prob),np.asarray(experts)
    if policy=='mixture_mean':
        return np.sum(prob*experts,axis=1)
    if policy=='argmax':
        action=np.argmax(prob,axis=1)
    elif policy in ['fn4','fn16']:
        cost=4. if policy=='fn4' else 16.
        matrix=np.array([[0,cost,2],[1,0,1],[2,cost,0]],float)
        action=np.argmin(prob@matrix,axis=1)
    elif policy=='inner_threshold':
        if threshold is None:raise ValueError('Training-side threshold required')
        tail=np.where(prob[:,0]>=prob[:,2],0,2)
        action=np.where(1-prob[:,1]>=threshold,tail,1)
    else:
        raise ValueError(policy)
    p=experts[np.arange(len(prob)),action]
    if not np.array_equal(states(p),action):
        raise AssertionError('Conditional MA and selected state disagree')
    return p


def choose_threshold(y,prob,experts):
    score=1-prob[:,1]
    thresholds=np.r_[-np.inf,np.unique(score),np.inf]
    candidates=[detailed(y,decode(prob,experts,'inner_threshold',float(t))) for t in thresholds]
    j=min(range(len(thresholds)),key=lambda i:(selection_key(candidates[i]),i))
    return float(thresholds[j]),candidates[j]


def train(data,curves,mask,a,b,arm,kind,seed,epochs=EPOCHS,device='cuda'):
    tx,qx,yn,w,center,tr,scaling=prepare(data,curves,mask,a,b,arm,'mse' if kind=='uniform' else 'weighted_mse')
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=JointNet().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    to=lambda v,dtype=torch.float32:torch.as_tensor(v,dtype=dtype,device=device)
    tx,qx,y,w,ci=to(tx),to(qx),to(60+10*yn),to(w),to(center,torch.long)
    qc=torch.zeros(len(b),dtype=torch.long,device=device)
    probabilities={};experts={};diags={}
    for epoch in range(1,max(epochs)+1):
        model.train();optim.zero_grad(set_to_none=True)
        logits,conditional=model(tx,ci)
        loss=objective(logits,conditional,y,w,kind)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optim.step()
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                logits,qma=model(qx,qc);prob=F.softmax(logits,dim=1).cpu().numpy().astype(float)
                prob/=prob.sum(1,keepdims=True)
                probabilities[epoch]=prob;experts[epoch]=qma.cpu().numpy().astype(float)
                lg,tm=model(tx,ci);tp=F.softmax(lg,dim=1).cpu().numpy().astype(float);tp/=tp.sum(1,keepdims=True)
                tm=tm.cpu().numpy().astype(float)
            keep=np.arange(len(tr))>=len(tr)-len(a)
            diags[epoch]=dict(loss=float(loss.detach().cpu()),training_target_metrics={p:detailed(data['y'][a,2],decode(tp[keep],tm[keep],p)) for p in POLICIES[:-1]})
    digest=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    return probabilities,experts,dict(scaling=scaling,parameter_sha256=digest,parameter_count=sum(v.numel() for v in model.parameters()),checkpoints=diags)


def run(seeds,name,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data,parts,extra,reps,mask,hashes=load_waveforms(seeds);curves=reps['raw900']
    out=ROOT/'outputs/uteg_mapping_target_runs'/name;(out/'private').mkdir(parents=True,exist_ok=True)
    epochs=[5] if smoke else EPOCHS
    grid=[dict(arm=a,loss=l,epoch=e,policy=p) for a in ARMS for l in LOSSES for e in epochs for p in POLICIES]
    protocol=dict(scope='C100, same60 grouped target development queries, extra40 training-only; 3outer and3inner folds, original201 masked.',
        seeds=list(parts),candidates=grid,torch=torch.__version__,device=device,
        waveform='Complete900 baseline-corrected first-channel points and observation mask, no temporal binning before encoder. Shared training-only observed-voltage scaling; masked source tail.',
        model='Same small convolution encoder as prior900-point CNN, two center-specific6-output heads:3state logits and3conditional continuous MA experts. Shared features jointly learn class CE and true-state expert squared MA error/100.',
        expert_ranges='low=(49.999)*sigmoid, normal=50+20*sigmoid, high=70.001+10*softplus; initialize45/60/75mm. No queried label, post-hoc MA correction, dropping or selective reporting.',
        weighting='Uniform or training-domain class-inverse weights cap20; source/target equal total base mass. FN4 adds -4log(prob(low)+prob(high)) on truly abnormal TRAINING rows. Both source and target heads penalized.',
        policies='Mixture mean is numerical control. Argmax and cost4/16 choose a state and its modeled conditional MA. Costs true-low/high-to-normal=4/16, normal-to-abnormal=1, opposite-tail=2; scores are not claimed clinically calibrated probabilities. Inner-threshold policy chooses threshold and opposite-tail direction using only inner OOF predictions.',
        selection='Inner folds select arm/loss/epoch/policy by MSE or safety criteria; outer labels evaluate only. Ranking minimizes FNR then FPR within accuracy-qualified sets; never increases FPR solely for strict FNR<FPR. Acceptance flag retains the original strict inequalities. Fixed candidates exploratory.',
        extension='5seed screen expands if joint pass, or agreement>80,FNR<37%,low recall>=25%, or R2>.35. A positive result still requires further locked validation.',
        reference='https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',
        input_hashes=hashes,code_sha256=sha256(__file__),selection_code_sha256=sha256(ROOT/'code_submit/ma_target_optimization/metrics.py'),test_evaluated=False,manuscript_changed=False)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool=parts[1000]['pool'];pos={int(v):j for j,v in enumerate(pool)}
    full=np.full((len(grid),seeds,len(pool)),np.nan);nested={k:np.full((seeds,len(pool)),np.nan) for k in ['mse','safety']}
    base_count=len(ARMS)*len(LOSSES)*len(epochs)
    probs=np.full((base_count,seeds,len(pool),3),np.nan);conditional=np.full_like(probs,np.nan)
    ledger=[];threshold_ledger=[];begin=time.time()
    for si,(seed,part) in enumerate(parts.items()):
        for fi,fold in enumerate(part['folds']):
            a0,b=fold['A'],fold['B'];add=extra[~np.isin(data['groups'][extra],data['groups'][b])];a=np.r_[a0,add]
            ix=[pos[int(v)] for v in b];ip={int(v):j for j,v in enumerate(a0)}
            inner=np.full((len(grid),len(a0)),np.nan);number=0;base_number=0
            for arm in ARMS:
                for kind in LOSSES:
                    key=f'{seed}_{fi}_{arm}_{kind}';cache=out/f'private/{key}.npz';afile=out/f'private/{key}.json'
                    if cache.exists() and afile.exists():
                        z=np.load(cache);pp={e:z[f'p{e}'] for e in epochs};ee={e:z[f'e{e}'] for e in epochs};iprob=z['inner_prob'];iexpert=z['inner_expert'];info=json.loads(afile.read_text())
                    else:
                        pp,ee,info=train(data,curves,mask,a,b,arm,kind,seed*100+fi,epochs,device)
                        iprob=np.full((len(epochs),len(a0),3),np.nan);iexpert=np.full_like(iprob,np.nan)
                        for ii,f in enumerate(fold['inner']):
                            tr,va=f['T'],f['V'];extra1=add[~np.isin(data['groups'][add],data['groups'][va])];tr=np.r_[tr,extra1]
                            p0,e0,_=train(data,curves,mask,tr,va,arm,kind,seed*100+10+fi*3+ii,epochs,device)
                            for ei,e in enumerate(epochs):
                                ji=[ip[int(v)] for v in va];iprob[ei,ji]=p0[e];iexpert[ei,ji]=e0[e]
                        np.savez_compressed(cache,inner_prob=iprob,inner_expert=iexpert,**{f'p{e}':pp[e] for e in epochs},**{f'e{e}':ee[e] for e in epochs})
                        afile.write_text(json.dumps(info,indent=2),encoding='utf-8')
                    if smoke:
                        poison=dict(data);poison['y']=data['y'].copy();poison['y'][b]=np.nan
                        changed=curves.copy();changed[b,:900]+=100
                        _,_,d2=train(poison,changed,mask,a,b,arm,kind,seed*100+fi,epochs,device)
                        assert d2==info
                        (out/'SMOKE_VERIFIED.json').write_text(json.dumps(dict(query_inputs_labels_do_not_fit=True,parameter_count=info['parameter_count']),indent=2))
                        print('SMOKE_COMPLETE',time.time()-begin,flush=True);return
                    for ei,e in enumerate(epochs):
                        threshold,tm=choose_threshold(data['y'][a0,2],iprob[ei],iexpert[ei])
                        threshold_ledger.append(dict(seed=seed,fold=fi,arm=arm,loss=kind,epoch=e,threshold=threshold,inner_metrics=tm))
                        for policy in POLICIES:
                            full[number,si,ix]=decode(pp[e],ee[e],policy,threshold)
                            inner[number]=decode(iprob[ei],iexpert[ei],policy,threshold)
                            number+=1
                        probs[base_number,si,ix]=pp[e];conditional[base_number,si,ix]=ee[e];base_number+=1
            yy=data['y'][a0,2];mm=[detailed(yy,p) for p in inner]
            choices={'mse':int(np.argmin(np.mean((inner-yy)**2,axis=1))),'safety':min(range(len(grid)),key=lambda j:(selection_key(mm[j]),j))}
            for key,j in choices.items():
                nested[key][si,ix]=full[j,si,ix];ledger.append(dict(seed=seed,fold=fi,objective=key,candidate=grid[j],inner_metrics=mm[j]))
            print('JOINT_FOLD',seed,fi,'seconds',round(time.time()-begin,1),flush=True)
    reference=np.broadcast_to(data['y'][pool,2],(seeds,len(pool)))
    result=[dict(candidate=c,metrics=detailed(reference,p)) for c,p in zip(grid,full)]
    selected=[dict(objective=k,metrics=detailed(reference,p)) for k,p in nested.items()]
    for row,p in zip(result+selected,list(full)+list(nested.values())):
        a=states(reference);b=states(p)
        cm=[[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]
        assert cm==row['metrics']['confusion']
        assert abs(row['metrics']['r2']-(1-np.sum((p-reference)**2)/np.sum((reference-reference.mean())**2)))<1e-12
    for path,digest in hashes.items():assert sha256(path)==digest
    np.savez_compressed(out/'private/PREDICTIONS.npz',fixed=full,reference=reference[0],pool=pool,seeds=list(parts),**nested)
    np.savez_compressed(out/'private/STATE_SCORES.npz',probabilities=probs,conditional_ma=conditional,reference=reference[0],pool=pool,seeds=list(parts))
    (out/'private/SELECTIONS.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    (out/'private/THRESHOLDS.json').write_text(json.dumps(threshold_ledger,indent=2),encoding='utf-8')
    promising=[z for z in result if z['metrics']['mapping_target_pass'] or z['metrics']['r2']>.35 or (z['metrics']['agreement']>.8 and z['metrics']['fnr']<.37 and z['metrics']['low_recall']>=.25)]
    (out/'RESULTS.json').write_text(json.dumps(dict(fixed=result,nested=selected,promising=promising,
        checks=dict(independent_metrics_verified=True,input_hashes_unchanged=True),test_evaluated=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--output',default='full_curve_joint_state_screen_20261102');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.seeds,a.output,a.smoke)
