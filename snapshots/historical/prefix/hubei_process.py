from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent;P=ROOT.parents[1]
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
for key in ['TEMP','TMP','TMPDIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX','JOBLIB_TEMP_FOLDER','NUMBA_CACHE_DIR','TORCH_HOME']:
    d=ROOT/'cache'/key;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.special import expit
import ast,json,hashlib,time,shutil
OLD=P/'outputs/uteg_hubei_curve_qc_fixedlength_20261021_170606'
FIT=P/'outputs/uteg_platform_param_compare_20261006_125428/curves.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(n,v):(ROOT/n).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')
def shape(q,t,fam):
    a,B,k,nu=q;z=np.maximum(t-a,0);v=np.zeros(len(t));ok=z>0;power=nu*(np.log(z[ok])-np.log(k))
    v[ok]=B*expit(power) if fam=='HILL' else B*(-np.expm1(-np.exp(np.minimum(power,700))))
    return v-np.median(v[:10])
node=next(n for n in ast.parse(FIT.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='fit_simple')
ns=dict(np=np,time=time,least_squares=least_squares)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(FIT),'exec'),ns)
def fit(y,fam):
    ns['shape']=lambda q,t,_:shape(q,t,fam)
    return ns['fit_simple'](y,'F4')
def metric(y,p):
    d=p-y;sse=float(d@d);sst=float(np.square(y-y.mean()).sum())
    return dict(n=len(y),MAE=float(np.mean(abs(d))),RMSE=float(np.sqrt(sse/len(y))),R2=1-sse/sst if sst>1e-20 else None)
def preprocess(y):
    n=len(y);win=np.lib.stride_tricks.sliding_window_view(np.pad(y,5,mode='edge'),11)
    med=np.median(win,axis=1);mad=np.median(abs(win-med[:,None]),axis=1)
    flagged=abs(y-med)>np.maximum(4.5*1.4826*mad,max(.01*float(np.ptp(y)),1e-8))
    replace=np.zeros(n,bool);starts=np.flatnonzero(flagged & ~np.r_[False,flagged[:-1]])
    for start in starts:
        end=start
        while end<n and flagged[end]:end+=1
        if end-start<=2:replace[start:end]=True
    despiked=y.copy();despiked[replace]=med[replace]
    smooth=savgol_filter(despiked,11,2,mode='interp');shift=float(np.median(smooth[:10]));den=smooth-shift
    H=float(np.median(den[-30:]));cut=None
    if H>max(1e-8,1e-6*float(np.ptp(den))):
        for k in range(9,n-9):
            if np.all(den[k:k+10]>=.05*H):cut=k;break
    p5=den.copy()
    if cut is not None:p5[:cut]=0
    diff=np.diff(y);j=int(np.argmin(diff));amp=max(float(np.ptp(y)),1e-8)
    sigma=1.4826*float(np.median(abs(diff-np.median(diff))))
    before=float(np.median(y[max(0,j-19):j+1]));after=float(np.median(y[j+1:min(n,j+21)]))
    broken=bool(-diff[j]>.2*amp and before-after>.3*amp and -diff[j]>10*max(sigma,1e-8))
    info=dict(n=n,despike_points=np.flatnonzero(replace),despike_candidate_points=np.flatnonzero(flagged),baseline_shift=shift,
      net_tail_median=H,p5_cut_index=cut,p5_cut_seconds=cut+1 if cut is not None else None,p5_fraction=cut/n if cut is not None else 0,
      max_negative_jump_seconds=j+2,max_negative_jump=float(diff[j]),jump_relative_to_range=float(-diff[j]/amp),
      suspected_discontinuity=broken,denoise_MAE_change=float(np.mean(abs(den-(y-shift)))),p5_MAE_change=float(np.mean(abs(p5-(y-shift)))))
    return den,p5,info

inp=np.load(OLD/'STANDARDIZED_INPUT.npz');ids=[str(v) for v in inp['ids']];source=inp['wave'][:,0];masks=inp['observed_mask'][:,0]
cohort=json.loads((OLD/'COHORT_81.json').read_text(encoding='utf-8'));oldfits=json.loads((OLD/'ALL_FIT_RESULTS.json').read_text(encoding='utf-8'))
previous_bad=[r['audit_id'] for r in oldfits if r['column']==1 and r['family']=='HILL' and r['observed']['R2']<.95]
assert len(previous_bad)==7 and len(ids)==81
processed=np.full((81,2,900),np.nan);meta=[]
for i,aid in enumerate(ids):
    n=int(masks[i].sum());y=source[i,:n];den,p5,m=preprocess(y);m['audit_id']=aid;m['previous_review']=aid in previous_bad;meta.append(m)
    for a,v in enumerate([den,p5]):processed[i,a]=np.pad(v,(0,900-n),mode='edge')
np.savez_compressed(ROOT/'CANDIDATE_SIGNALS.npz',processed=processed,original=source,observed_mask=masks,ids=np.array(ids))
write('PREPROCESSING_LOG.json',meta)
write('FREEZE.json',dict(protocol_sha256=sha(ROOT/'PROTOCOL.md'),code_sha256=sha(Path(__file__)),source_archive_sha256=sha(OLD/'STANDARDIZED_INPUT.npz'),fitter_sha256=sha(FIT),candidate_archive_sha256=sha(ROOT/'CANDIDATE_SIGNALS.npz'),clinical_labels_used=False,positions_cap=324))
# Controlled operations, no scientific fits or extra optimizer calls.
test=np.zeros(101);test[50]=5.;d,p5,m=preprocess(test);assert 50 in m['despike_points'];assert max(abs(d))<1e-8
flat=np.full(101,2.);d,p5,m=preprocess(flat);assert max(abs(d))<1e-8 and m['p5_cut_index'] is None
write('CONTROLLED_TESTS.json',dict(isolated_spike=True,constant_input=True,new_test_fits=0,numerical_guard_note='Initial constant-input test caught SG roundoff triggering 5% crossing; added amplitude floor before any scientific fit'))
out=[];begin=time.time()
for i,aid in enumerate(ids):
    assert shutil.disk_usage(ROOT).free>6*2**30
    n=meta[i]['n'];shift=meta[i]['baseline_shift'];raw=source[i,:n]-shift
    for a,prep in enumerate(['DENOISE','DENOISE_P5']):
        for fam in ['HILL','EXP']:
            row=dict(audit_id=aid,prep=prep,family=fam,q=None)
            try:
                f=fit(processed[i,a],fam);q=f['q'];pred=shape(q,np.arange(1,901.),fam)
                cut=meta[i]['p5_cut_index'] if prep=='DENOISE_P5' else 0
                k=cut if cut is not None else 0
                row.update(status='FIT_RETURNED',q=q,fit=f,processed_observed=metric(processed[i,a,:n],pred[:n]),original_observed=metric(raw,pred[:n]),
                  retained_original=metric(raw[k:],pred[k:n]),retained_from_index=k,full=metric(processed[i,a],pred))
                a0,B,k0,nu=q
                row['bounds']=dict(onset_zero=a0<1e-4,k_lower=abs(k0-1)<1e-4,k_upper=abs(k0-900)<1e-3,nu_lower=abs(nu-.2)<1e-4,nu_upper=abs(nu-12)<1e-4)
            except Exception as e:row.update(status='FIT_EXCEPTION',error=repr(e))
            out.append(row)
    if i%10==0 or i==80:print(f'{i+1}/81 records; {len(out)} positions; {time.time()-begin:.1f}s',flush=True)
write('ALL_FIT_RESULTS.json',out)
write('RESOURCE_LEDGER.json',dict(positions=len(out),actual_fit_calls=len(out),seconds=time.time()-begin,residual_calls=sum(s['calls'] for r in out if 'fit' in r for s in r['fit']['starts']),
  raw_hashes_unchanged=all(sha(Path(r['UTEG_csv']))==r['UTEG_csv_sha256'] for r in cohort),mapping_calibration_gate_fits=0,new_downstream_results=False))
