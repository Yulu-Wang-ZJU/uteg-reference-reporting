"""Local frozen MOMENT features from approved first-column UTEG observations.

All records are encoded independently; no reference label fits this transform.
Test rows remain NaN. An identically configured random encoder is a control.
"""
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .search import ROOT
from .full_curve_cost_sensitive import load_waveforms
from code_submit.ma_imbalance_mapping.data_contract import sha256

BASE=ROOT/'outputs/uteg_mapping_target_runs'
RUNTIME=BASE/'moment_runtime'
CHECKPOINT=BASE/'moment_checkpoint_small_20261102'
OUTPUT=BASE/'frozen_moment_features_20261102'
REVISION='411e288267f82cce86296dbe4d6c8bc533cc162f'


def views(curve,observed):
    curve,observed=np.asarray(curve,float),np.asarray(observed,bool)
    assert curve.shape==observed.shape==(900,)
    x=np.zeros((3,512),np.float32);mask=np.zeros((3,512),bool)
    # Global view includes all900 points in fixed non-overlapping mean bins.
    edges=np.linspace(0,900,513).astype(int)
    for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        if observed[a:b].all():x[0,j]=curve[a:b].mean();mask[0,j]=True
    # Native-rate views include all900 points with124-point overlap.
    for j,start in [(1,0),(2,388)]:
        mask[j]=observed[start:start+512]
        x[j]=np.where(mask[j],curve[start:start+512],0)
    return x,mask


def state_digest(model):
    digest=hashlib.sha256()
    for k,v in model.state_dict().items():digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(pretrained=True, checkpoint=CHECKPOINT):
    if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
    from momentfm import MOMENTPipeline
    from safetensors.torch import load_file
    torch.manual_seed(20261102)
    config=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    model=MOMENTPipeline(config,model_kwargs=dict(task_name='embedding',enable_gradient_checkpointing=False))
    if pretrained:model.load_state_dict(load_file(str(checkpoint/'model.safetensors')),strict=True)
    model.init();model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def encode(model,x,mask,batch_size=16):
    device=next(model.parameters()).device
    # Official masking treats a patch as observed only when all8 points exist.
    valid=mask.reshape(len(mask),64,8).all(-1).any(-1)
    result=np.zeros((len(x),model.config.d_model),float)
    indices=np.flatnonzero(valid)
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size]
            xx=torch.as_tensor(x[ix,None,:],device=device)
            mm=torch.as_tensor(mask[ix],device=device)
            e=model(x_enc=xx,input_mask=mm).embeddings
            result[ix]=e.cpu().numpy()
    assert np.isfinite(result).all()
    return result,valid


def run():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    data,parts,extra,reps,observed,hashes=load_waveforms(5)
    rows=data['allowed_training_indices'];curve=reps['raw900'][:,:900]
    x,mask=zip(*(views(curve[j],observed[j]) for j in rows))
    x,mask=np.asarray(x).reshape(-1,512),np.asarray(mask).reshape(-1,512)
    OUTPUT.mkdir(exist_ok=True);(OUTPUT/'private').mkdir(exist_ok=True)
    protocol=dict(model='AutonLab/MOMENT-1-small',revision=REVISION,package='momentfm0.1.4',
        source='https://huggingface.co/AutonLab/MOMENT-1-small',
        task='Frozen pretrained embedding, no UTEG optimization. Strict safetensors loading; reconstruction head removed using official init(). Identical architecture random initialization is separate control.',
        normalization='Official per-record RevIN, using observed points only. Raw amplitude/current18 features will be separate downstream blocks, because normalization removes much amplitude information.',
        views='Global900-to512 nonoverlapping mean bins, native1second0:512 and388:900 windows. All native900 points covered. Missing observations zero-storage with false mask; full8-point observed patches required by official encoder; all-missing window embedding0 plus observed fraction.',
        data='Approved274 source/local development records only; no original201 test features, labels or uploads. Encoding independent per record. Short source tail is never treated as measurement.',
        code_sha256=sha256(__file__),input_hashes=hashes,weights_sha256=sha256(CHECKPOINT/'model.safetensors'),
        runtime_hashes={str(p):sha256(p) for p in (RUNTIME/'momentfm').rglob('*.py')},
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','safetensors','huggingface_hub']},
        test_evaluated=False,manuscript_changed=False)
    (OUTPUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    arrays={};checks=[];begin=time.time()
    for name,pretrained in [('pretrained',True),('random',False)]:
        model=load_model(pretrained);before=state_digest(model)
        embedding,valid=encode(model,x,mask)
        replay,_=encode(model,x[:6],mask[:6],batch_size=1)
        batch_error=float(np.max(abs(replay-embedding[:6])))
        assert batch_error<1e-4
        changed=x[:6].copy();changed[~mask[:6]]=1e5
        padding,_=encode(model,changed,mask[:6],batch_size=1)
        assert np.max(abs(padding-replay))<1e-6
        assert state_digest(model)==before
        #3 fractional lengths retain missingness explicitly, with no fitting.
        local=np.c_[embedding.reshape(len(rows),-1),mask.mean(1).reshape(len(rows),3)]
        full=np.full((len(curve),local.shape[1]),np.nan);full[rows]=local
        assert np.isnan(full[parts[1000]['test']]).all();arrays[name]=full
        checks.append(dict(name=name,weights_unchanged=True,all_parameters_frozen=True,
            batch_invariance_max_error=batch_error,masked_padding_invariant=True,empty_windows=int((~valid).sum())))
        del model;torch.cuda.empty_cache()
        print('FEATURES',name,local.shape,'seconds',round(time.time()-begin,1),flush=True)
    for p,digest in hashes.items():assert sha256(p)==digest
    np.savez_compressed(OUTPUT/'private/FEATURES.npz',**arrays,allowed=rows,pool=parts[1000]['pool'])
    (OUTPUT/'VERIFICATION.json').write_text(json.dumps(dict(checks=checks,input_hashes_unchanged=True,
        data_rows=len(rows),reference_labels_not_used_for_encoder=True,test_encoded=False),indent=2),encoding='utf-8')
    print('COMPLETE',str(OUTPUT),flush=True)


if __name__=='__main__':run()
