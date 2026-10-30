from pathlib import Path
import os,sys,json,hashlib,time,shutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=OUT.parent
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/py','JOBLIB_TEMP_FOLDER':'cache/joblib','NUMBA_CACHE_DIR':'cache/numba','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'cache/mpl','XDG_CACHE_HOME':'cache/xdg'}.items():
    p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
sys.dont_write_bytecode=True
import numpy as np,scipy
from scipy.signal import savgol_filter
PREV=OUT/'uteg_zero5_med4_20261006_113257'
AUDIT=OUT/'uteg_processing_harmonization_audit_20261006_121445'
CHECK=OUT/'uteg_plateau_simplification_20261006_123640'
Q={'CUT_017':.0017,'CUT_1':.01,'CUT_3':.03,'CUT_5':.05}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(n,x):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(v):
    v=np.array(v,float);v=v[np.isfinite(v)];return dict(n=len(v),quantiles=np.quantile(v,[0,.05,.25,.5,.75,.95,1])) if len(v) else dict(n=0)
def noise(y):
    d=np.diff(y,2);return float(1.4826*np.median(abs(d-np.median(d)))/np.sqrt(6))
def apply(y,q=None):
    y=np.asarray(y,float)[:900];assert len(y)>=11 and np.isfinite(y).all()
    base=float(np.median(y[:10]));zero=y-base;sm=savgol_filter(zero,11,2,mode='interp');offset=float(np.median(sm[:10]));sm=sm-offset;z=sm.copy();H=float(np.median(sm[-30:]));k=None
    if q is not None and H>0:
        for i in range(9,len(y)-9):
            if np.all(sm[i:i+10]>=q*H):k=i;z[:i]=0.;break
    return z,dict(baseline=base,filter_rezero=offset,amplitude=H,k=k,threshold=(q*H if q is not None else None),status='filter_only' if q is None else ('applied' if k is not None else 'no_valid_crossing'))
def main():
    manifest=read(AUDIT/'private/RAW_FILE_MANIFEST.json');archive=PREV/'private/input_archive.npz';upstream=[archive,CHECK/'private/HUZHOU_PLATFORM_JUMPS.json',AUDIT/'private/RAW_FILE_MANIFEST.json'];a=dict(np.load(archive));assert len(manifest)==401
    for row in manifest:assert sha(row['path'])==row['sha256']
    write('PROTOCOL_FREEZE.json',dict(time=time.time(),files={str(p):sha(p) for p in [ROOT/'PROTOCOL.md',Path(__file__)]},inputs={str(p):sha(p) for p in upstream},raw_files=manifest,configuration=dict(thresholds=Q,window=11,order=2,mode='interp',persistence=10,baseline=10,endpoint=30),numpy=np.__version__,scipy=scipy.__version__,clinical_labels_used=False,new_model_fits=0))
    orig=a['original'];zero=a['zeroed'];obs=a['observed'];variants={'FILTER_ONLY':zero.copy(),**{n:zero.copy() for n in Q}};records=[]
    for i in range(306):
        y=orig[i,obs[i]];before=zero[i,obs[i]];amp=max(float(np.median(before[-30:])),1e-12)
        for name,q in [('FILTER_ONLY',None),*Q.items()]:
            z,info=apply(y,q);variants[name][i,obs[i]]=z;k=info['k'];records.append(dict(index=i,id=str(a['ids'][i]),variant=name,n=len(y),**info,RMSE=float(np.sqrt(np.mean((z-before)**2))),normalized_RMSE=float(np.sqrt(np.mean((z-before)**2)))/amp,MAE=float(np.mean(abs(z-before))),noise_before=noise(before),noise_after=noise(z),endpoint_change=float(np.median(z[-30:])-np.median(before[-30:])),boundary_jump=float(z[k]-z[k-1]) if k is not None and k else None,boundary_jump_ratio=float(z[k]/info['amplitude']) if k is not None else None,negative_after_cut=int(np.sum(z[k:]<0)) if k is not None else None))
    private=ROOT/'private';private.mkdir(exist_ok=True)
    np.savez_compressed(private/'ORIGINAL_AND_ZEROED.npz',original=orig,zeroed=zero,time=a['time'],observed=obs,ids=a['ids'],groups=a['groups'])
    for name,v in variants.items():
        np.savez_compressed(private/f'{name}.npz',signal=v,observed=obs,time=a['time'],ids=a['ids'],groups=a['groups'],source_center=np.where(np.arange(401)<306,'SAHZU','Huzhou'),new_processing_applied=np.arange(401)<306)
        pad=v.copy()
        for i in range(401):
            n=int(sum(obs[i]));pad[i,n:]=pad[i,n-1]
        np.savez_compressed(private/f'{name}_DISPLAY_ONLY_PADDED.npz',signal=pad,observed=obs,padded=~obs,ids=a['ids'])
    sums={}
    for name in variants:
        rows=[z for z in records if z['variant']==name];sums[name]=dict(n=len(rows),no_crossing=sum(z['status']=='no_valid_crossing' for z in rows),**{k:quant([z[k] for z in rows if z[k] is not None]) for k in ['k','normalized_RMSE','noise_before','noise_after','boundary_jump_ratio','endpoint_change','negative_after_cut']})
    hz=read(CHECK/'private/HUZHOU_PLATFORM_JUMPS.json');valid=[z for z in hz if z['constant_prefix_points']>=10];rates=np.array([z['first_rise_over_endpoint_rise'] for z in valid]);bins=[0,.001,.0015,.002,.003,.01,np.inf];counts=np.histogram(rates,bins)[0]
    actual=orig[306:,:];starts=actual[:,0];rise=read(CHECK/'HUZHOU_PLATFORM_JUMP_SUMMARY.json')['constant_prefix_at_least10'];hzsummary=dict(n=94,definition='(first value above platform - platform)/(median last30 - platform)',percentage=100*np.quantile(rates,[0,.05,.25,.5,.75,.95,1]),bins_percent=['<0.1','0.1–<0.15','0.15–<0.2','0.2–<0.3','0.3–<1','>=1'],bin_counts=counts,raw_platform_exactly_zero=int(sum(starts==0)),raw_platform_n=95,raw_platform_values=quant(starts),one_percent_outlier_ids=[z['id'] for z in valid if z['first_rise_over_endpoint_rise']>=.01],endpoint_limitation='record end, not verified physiological plateau',original_rule_recovered=False)
    write('HUZHOU_RATIO_DISTRIBUTION.json',hzsummary);write('PREPROCESSING_SUMMARY.json',sums);write('private/RECORD_PREPROCESSING.json',records)
    for row in manifest:assert sha(row['path'])==row['sha256']
    for p,h in read(ROOT/'PROTOCOL_FREEZE.json')['inputs'].items():assert sha(p)==h
    write('PROCESSING_COMPLETE.json',dict(time=time.time(),SAHZU_records=306,Huzhou_reference_records=95,variants=5,clinical_labels_loaded=False,new_model_fits=0,new_curve_fits=0,raw_unchanged=True,bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),free_E=shutil.disk_usage(ROOT).free))
    print(json.dumps(clean(dict(huzhou=hzsummary,SAHZU=sums)),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
