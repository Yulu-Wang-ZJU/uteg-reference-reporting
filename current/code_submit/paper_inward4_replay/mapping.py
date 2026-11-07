"""Six original source families, same local B-only inward4 correction.

Original source recipes keep four inputs and their original losses. This is a
source-recipe sensitivity comparison, not a pure model-architecture ablation.
"""
import argparse
import gc
import importlib.util
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c
from code_submit.paper_b085_hospital_replay.adaptation_controls import fit_cal,pred_cal

METHODS=['RIDGE-10','KRR-1','PLS-2','SVR-1','MLP-8','TT-SMALL']
_ENV=None

def env():
    global _ENV
    if _ENV is None:
        path=c.WS/'outputs/uteg_supplement_completion_20261027_203140/batch.py'
        spec=importlib.util.spec_from_file_location('inward_paper_legacy_batch',path)
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        b.ROOT=c.OUT/'legacy_runtime'
        r,runner,target,*_=b.env('source_models',100000)
        _ENV=target.m
    return _ENV

def source(direction,method,seed):
    path=c.OUT/'source_models'/direction/f'{method}_{seed if method in METHODS[-2:] else 1000}.joblib'
    if path.exists():
        if path.stat().st_size<32:
            path.rename(path.with_suffix('.incomplete'))
        else:return joblib.load(path)
    m=env();d,*_=c.context();rows=c.source_rows(direction,900)
    x=c.source_feature(direction,900,'four')
    data=dict(d,x=x,medical=np.full_like(d['medical'],np.nan),ci=np.full_like(d['ci'],np.nan))
    data['medical'][rows]=d['medical'][rows];data['ci'][rows]=d['ci'][rows]
    model=m.fit_source(data,rows,[],method,seed if method in METHODS[-2:] else 1000,steps=500 if method in METHODS[-2:] else None)
    assert model['success'] and np.isfinite(m.predict(model,x[rows])).all()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(path)
    return model

def task(direction,method):
    path=c.OUT/'mapping'/f'{direction}_{method}.npz'
    if path.exists():return
    m=env();d,*_=c.context();raw=[];affine=[];joint=[];huber=[];complete=[];audits=[]
    for seed in c.SEEDS:
        model=source(direction,method,seed);spec=c.paper_spec(direction,seed);a=spec['pool'];b=spec['test'];valid=c.qc_at(900)[b];bb=b[valid]
        def predictor(train,query,h):
            x=c.feature(train,query,h,'four')
            return m.predict(model,x[train])[:,2],m.predict(model,x[query])[:,2]
        p,au,_=c.predict(direction,900,a,bb,prior_override={'predict_rows':predictor})
        x=c.feature(a,bb,900,'four');pt=m.predict(model,x[a]);pq=m.predict(model,x[bb]);ref=d['medical'][a]
        cal={k:fit_cal(pt,ref,model,k) for k in ['affine','joint']}
        hc=m.fit_cal(pt,ref,model['ym'],model['ys']);assert hc['finite']
        ph=m.predict_cal(hc,pq);ph=m.project(ph)
        def full(arr):
            v=np.full((len(b),4),np.nan);v[valid]=arr;return v
        raw.append(full(pq));affine.append(full(pred_cal(cal['affine'],pq)));joint.append(full(pred_cal(cal['joint'],pq)));huber.append(full(ph))
        cp=pq.copy();cp[:,2]=p;complete.append(full(cp));audits.append(dict(seed=seed,local_fit=au,huber=hc,source_success=model['success']))
        gc.collect()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,raw=raw,affine=affine,joint=joint,huber=huber,complete=complete,test=b,seed=c.SEEDS)
    c.write(path.with_suffix('.audit.json'),audits)
    c.write(path.with_suffix('.metrics.json'),{k:c.summarize(np.asarray(v)[:,:,2],d['y'][b,2]) for k,v in [('raw',raw),('affine',affine),('joint',joint),('huber',huber),('complete',complete)]})
    print('MAPPING_COMPLETE',direction,method,flush=True)

def primary_calibration(direction):
    path=c.OUT/'calibration'/f'{direction}.npz'
    if path.exists():return
    m=env();d,*_=c.context();src=c.source_model(direction,900);rows=c.source_rows(direction,900)
    arrays={k:[] for k in ['raw','affine','joint','huber','complete']};audits=[]
    for seed in c.SEEDS:
        aux=source(direction,'RIDGE-10',seed);spec=c.paper_spec(direction,seed);a=spec['pool'];b=spec['test'];bb=b[c.qc_at(900)[b]]
        x4=c.feature(a,bb,900,'four');x=c.feature(a,bb,900)
        pt=m.predict(aux,x4[a]);pq=m.predict(aux,x4[bb]);pt[:,2]=c.source_predict(src,x[a]);pq[:,2]=c.source_predict(src,x[bb])
        scale={'ym':d['medical'][rows].mean(0),'ys':np.maximum(d['medical'][rows].std(0),1e-6)}
        cal={k:fit_cal(pt,d['medical'][a],scale,k) for k in ['affine','joint']}
        hc=m.fit_cal(pt,d['medical'][a],scale['ym'],scale['ys']);assert hc['finite']
        cp=pq.copy();cp[:,2]=c.predict(direction,900,a,bb)[0]
        vals={'raw':pq,'affine':pred_cal(cal['affine'],pq),'joint':pred_cal(cal['joint'],pq),
            'huber':m.project(m.predict_cal(hc,pq)),'complete':cp}
        for k,v in vals.items():
            vv=np.full((len(b),4),np.nan);vv[c.qc_at(900)[b]]=v;arrays[k].append(vv)
        audits.append(dict(seed=seed,huber=hc))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays,test=b,seed=c.SEEDS)
    c.write(path.with_suffix('.audit.json'),audits)
    c.write(path.with_suffix('.metrics.json'),{k:c.summarize(np.array(v)[:,:,2],d['y'][b,2]) for k,v in arrays.items()})

def run(method):
    p=c.OUT/'MAPPING_PROTOCOL.json'
    if not p.exists():c.write(p,dict(methods=METHODS,source_inputs='four parameters with source-center normalized B at900',
        primary='B-only source inward4 remains separately reported',
        comparison='Original source recipes preserved, including their original training objectives; all receive the same B-only local final-MA inward4 correction. Representation/loss differences are explicit.',
        calibration='MA source prior plus auxiliary R/K/angle from four-input Ridge for primary joint controls; these auxiliary channels are not new B-only predictions.',
        source_seeds='20 seeds for MLP/TT; deterministic source fits reused at1000',steps=500,B_from_original_data=True))
    methods=METHODS if method=='all' else [method]
    for direction in [d for d in c.TARGET if not d.startswith('target_only')]:
        for name in methods:task(direction,name)
        if 'RIDGE-10' in methods:primary_calibration(direction)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',default='all',choices=METHODS+['all']);a=ap.parse_args()
    with threadpool_limits(limits=1):run(a.method)
