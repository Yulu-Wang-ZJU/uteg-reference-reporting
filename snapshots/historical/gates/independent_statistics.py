"""Independent target aggregates and 24 paired intervals. No original metrics imports."""
import runtime as r
import numpy as np, joblib

NBOOT=20000
SEED=20261004
METHODS=['TRUST-SELECTED','ERROR-LR','GP-RISK','DOS-GATE','CI-MARGIN']

def weights(labels):
    W=np.zeros((NBOOT,len(labels)),np.int16)
    for cl in (-1,0,1,9):
        ix=np.flatnonzero(labels==cl)
        if len(ix):
            rng=np.random.default_rng(SEED+(cl+2)*100003)
            W[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=NBOOT)
    return W

def areas(scores,errors,W):
    good=np.isfinite(scores)&np.isfinite(errors)
    s=scores[good];e=errors[good];w=W[:,good].astype(float)
    out=np.zeros(len(w)); total=w.sum(1)
    if not len(s): return np.full(len(w),np.nan)
    order=np.argsort(s,kind='stable');s=s[order];e=e[order];w=w[:,order]
    # Independent cumulative integration, one complete tie block at a time.
    boundaries=np.r_[0,1+np.flatnonzero(np.diff(s)!=0),len(s)]
    c=np.zeros(len(w));v=np.zeros(len(w));low=.1*total
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        mass=w[:,lo:hi].sum(1); errs=(w[:,lo:hi]*e[lo:hi]).sum(1)
        density=np.divide(errs,mass,out=np.zeros_like(errs),where=mass>0)
        left=np.maximum(c,low);right=c+mass
        active=(right>left)&(mass>0)&(left>0)
        out[active] += density[active]*(right[active]-left[active])+(v[active]-density[active]*c[active])*np.log(right[active]/left[active])
        c+=mass;v+=errs
    return np.divide(out,.9*total,out=np.full(len(w),np.nan),where=total>0)

def main():
    fr=r.read('GATE_SELECTION_FREEZE.json');assert len(fr['runs'])==40
    expected=r.read('EVALUATION_SUMMARY.json')
    checks=[];summaries={};compare=[];maxdiff=0.0
    for d in (1,2):
        runs=[joblib.load(r.ROOT/'targets'/f'D{d}_seed{s}.joblib') for s in range(1000,1020)]
        truth=runs[0]['ytrue']; known=np.isin(truth,[-1,0,1]); W=weights(truth)
        assert all(np.array_equal(z['indices'],runs[0]['indices']) and np.array_equal(z['ytrue'],truth) for z in runs)
        boot={};point={};summaries[str(d)]={}
        for name in runs[0]['methods']:
            if name=='RANDOM':continue
            aa=np.stack([z['methods'][name]['action'] for z in runs]).astype(float)
            yh=np.stack([z['yhat'] for z in runs]);ee=(yh!=truth).astype(float)
            abar=aa.mean(0);joint=(aa*ee).mean(0)
            n=abar[known].sum();risk=joint[known].sum()/n if n else np.nan
            coverage=abar.mean(); afn=np.mean(aa*(yh==0)*(truth!=0)*known,axis=0).sum()
            row=dict(coverage=coverage,risk=risk,accuracy=1-risk if np.isfinite(risk) else np.nan,
                     abnormal_false_normal_mean=afn,mean_early_correct=np.mean(aa*(yh==truth)*known,axis=0).sum())
            summaries[str(d)][name]=row
            old=expected['aggregate'][str(d)]['methods'][name]
            for field,ours in [('early_risk',risk),('abnormal_false_normal_mean',afn)]:
                theirs=old.get(field)
                delta=abs(ours-theirs) if theirs is not None and np.isfinite(ours) else 0 if theirs is None and not np.isfinite(ours) else np.inf
                maxdiff=max(maxdiff,delta);checks.append(dict(direction=d,method=name,field=field,difference=delta))
            if name not in METHODS:continue
            den=W@(abar*known);num=W@(joint*known)
            br=np.divide(num,den,out=np.full(NBOOT,np.nan),where=den>0)
            bc=(W@abar)/W.sum(1)
            ba=np.zeros(NBOOT); ap=[]
            for z in runs:
                score=np.asarray(z['methods'][name]['score'],float).copy()
                good=known&np.isin(z['yhat'],[-1,0,1])
                err=(z['yhat']!=truth).astype(float);err[~good]=np.nan
                ba+=areas(score,err,W)/20
                ap.append(areas(score,err,np.ones((1,len(truth))))[0])
            boot[name]=dict(risk=br,coverage=bc,aurc=ba)
            point[name]=dict(risk=risk,coverage=coverage,aurc=float(np.mean(ap)))
            for label,arr in [('risk',br),('coverage',bc),('aurc',ba)]:
                valid=np.isfinite(arr)
                summaries[str(d)][name][label+'_ci95']=np.quantile(arr[valid],[.025,.975]).tolist() if valid.any() else [None,None]
                summaries[str(d)][name][label+'_bootstrap_valid']=int(valid.sum())
        for b in METHODS[1:]:
            for metric in ('risk','coverage','aurc'):
                a='TRUST-SELECTED';gain=(boot[a][metric]-boot[b][metric]) if metric=='coverage' else (boot[b][metric]-boot[a][metric])
                pg=(point[a][metric]-point[b][metric]) if metric=='coverage' else(point[b][metric]-point[a][metric])
                finite=np.isfinite(gain)
                ci=np.quantile(gain[finite],[.025,.975]).tolist() if finite.any() else [None,None]
                bon=np.quantile(gain[finite],[.05/48,1-.05/48]).tolist() if finite.any() else [None,None]
                key=f'D{d}:{a}>{b}:{metric}';old=expected['comparisons'][key]
                diffs=[]
                for ours,theirs in zip(ci+bon,old['CI95']+old['CI_Bonferroni_24']):
                    diffs.append(0 if ours is None and theirs is None else np.inf if ours is None or theirs is None else abs(ours-theirs))
                if np.isfinite(pg) and old['point_gain_a_b'] is not None:diffs.append(abs(pg-old['point_gain_a_b']))
                mx=max(diffs);maxdiff=max(maxdiff,mx)
                compare.append(dict(key=key,point_gain=pg,CI95=ci,CI_Bonferroni_24=bon,valid=int(finite.sum()),max_difference=mx))
        print('Independent statistics complete D'+str(d),flush=True)
    out=dict(status='PASS' if maxdiff<1e-10 else 'FAIL',max_abs_difference=maxdiff,
             threshold=1e-10,n_boot=NBOOT,seed=SEED,fit_count=0,
             no_original_evaluation_imported=True,summaries=summaries,comparisons=compare,checks=checks)
    r.write('INDEPENDENT_STATISTICS_VERIFICATION.json',out)
    print(out['status'],maxdiff,flush=True)

if __name__=='__main__':main()
