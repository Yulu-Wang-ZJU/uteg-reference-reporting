from pathlib import Path
import csv,json
import numpy as np
import joblib
P=Path(__file__).resolve().parent
O=P.parent/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
D=dict(np.load(P.parent/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'))
rows=list(csv.DictReader((P/'summary.csv').open(encoding='utf-8-sig')))
checks=[]
for d in ['HZ_to_SA','HZ+HB_to_SA']:
    z=np.load(P/f'{d}_predictions.npz');v=z['predictions'];y=z['reference'];c=lambda a:np.digitize(a,[50,70],right=False)
    def category(a):return np.where(a<50,0,np.where(a<=70,1,2))
    yc=category(y[:,2]);pc=category(v[:,:,:,2])
    for j,arm in enumerate(z['arms']):
        r=next(r for r in rows if r['direction']==d and r['arm']==arm)
        assert abs(float(r['MA_MAE'])-np.mean(abs(v[j,:,:,2]-y[:,2])))<1e-12
        assert abs(float(r['agreement'])-np.mean(pc[j]==yc))<1e-12
        for label,k in [('low',0),('high',2)]:
            assert int(r[label+'_to_within_n'])==np.count_nonzero((yc==k)&(pc[j]==1))
    for seed in range(1000,1020):
        o=joblib.load(O/'models'/d/'target15'/f'SAHZU_CFULL_seed{seed}.joblib');s=o['source_model'];a=o['positions'][3]['A']
        p=s['ym']+s['ys']*(((D['x'][a]-s['xm'])/s['xs'])@s['coef']+s['intercept'])
        p[:,:3]=np.maximum(p[:,:3],0);p[:,3]=np.clip(p[:,3],0,89.999999)
        aff=joblib.load(P/f'{d}_{seed}_affine.joblib');lin=joblib.load(P/f'{d}_{seed}_joint.joblib')
        u=(D['medical'][a]-s['ym'])/s['ys'];w=(p-s['ym'])/s['ys'];wc=w-w.mean(0);uc=u-u.mean(0)
        # Independent augmented least squares formulation of identity-regularized regression.
        mm=np.linalg.lstsq(np.vstack([wc,np.sqrt(10)*np.eye(4)]),np.vstack([uc,np.sqrt(10)*np.eye(4)]),rcond=None)[0]
        assert np.allclose(mm,lin['matrix'],atol=1e-10)
        for k in range(4):
            coef=np.linalg.lstsq(np.column_stack([w[:,k],np.ones(len(w))]),u[:,k],rcond=None)[0]
            assert np.allclose(w[:,k]*coef[0]+coef[1],aff['ubar'][k]+(w[:,k]-aff['vbar'][k])*aff['matrix'][k,k])
    checks.append({'direction':d,'all_20_seed_models_independently_verified':True,'counts_and_MAE_verified':True})
(P/'INDEPENDENT_CHECK.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
