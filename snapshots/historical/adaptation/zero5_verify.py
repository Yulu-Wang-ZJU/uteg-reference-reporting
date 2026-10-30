"""Replay with independent NumPy network, curve formula, calibration and metric implementations."""
import runtime as r
import numpy as np,joblib,time
from scipy.special import softmax
r.extract(r.PREV/'verify_independent.py',['project','ci','label','cal','weights','parts','delta','ratio','stats','boot'],globals())

def source(m,X):
    s=m['state'];tokens=len(m['xm']);z=((X-m['xm'])/m['xs'])[:,:,None]*s['val'][None];p='enc.layers.0.'
    qkv=z@s[p+'self_attn.in_proj_weight'].T+s[p+'self_attn.in_proj_bias'];q,k,v=[a.reshape(len(X),tokens,2,6).transpose(0,2,1,3) for a in np.split(qkv,3,-1)]
    a=(softmax(q@k.swapaxes(-1,-2)/np.sqrt(6),-1)@v).transpose(0,2,1,3).reshape(len(X),tokens,12);z=z+a@s[p+'self_attn.out_proj.weight'].T+s[p+'self_attn.out_proj.bias']
    z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*s[p+'norm1.weight']+s[p+'norm1.bias'];a=np.maximum(z@s[p+'linear1.weight'].T+s[p+'linear1.bias'],0);z=z+a@s[p+'linear2.weight'].T+s[p+'linear2.bias'];z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*s[p+'norm2.weight']+s[p+'norm2.bias']
    return m['ym']+m['ys']*(z.mean(1)@s['out.weight'].T+s['out.bias'])

def shape(q,t):
    power=np.power(t/q[3],q[4]);h=q[0]*np.exp(-t/q[1])+q[2]*power/(1+power);a=np.sort(h[:10]);return h-(a[4]+a[5])/2

def main():
    data=dict(np.load(r.ROOT/'private/data.npz'));archive=dict(np.load(r.ROOT/'private/input_archive.npz'));splits=r.read('SPLIT_MANIFEST.json');tables=r.read('TARGET_METRICS.json');saved={d:dict(np.load(r.ROOT/f'private/D{d}_predictions.npz')) for d in [1,2]};sp=joblib.load(r.ROOT/'private/source_predictions.joblib');freeze=r.read('MODEL_FREEZE.json');mx={k:0. for k in ['curve_metrics','prediction','scales','weights','Huber_objective','metrics','ROC','interval']};nmap=ncal=0
    for name in ['PROTOCOL_FREEZE.json','INPUT_IDENTITY.json']:
        for p,h in r.read(name)['files'].items():assert r.sha(p)==h,p
    assert r.sha(r.ROOT/'private/data.npz')==r.read('FITTED_INPUT_FREEZE.json')['data_sha256'];assert freeze['time']<r.read('EVALUATION_COMPLETE.json')['time'];assert r.sha(r.ROOT/'evaluate.py')==r.read('EVALUATION_CODE_FREEZE.json')['sha'];curve_rows=[]
    for i in range(401):
        path=r.ROOT/f'curve_fits/{i:03d}.joblib';assert r.sha(path)==r.read('FITTED_INPUT_FREEZE.json')['fit_hashes'][path.name];f=joblib.load(path);q=data['x_zero5'][i];assert np.array_equal(q,f['q']);obs=archive['observed'][i];y=archive['zeroed'][i,obs];t=np.arange(1,len(y)+1.);pred=shape(q,t);sse=np.square(pred-y).sum();r2=1-sse/np.square(y-y.mean()).sum();mx['curve_metrics']=max(mx['curve_metrics'],abs(sse-f['SSE']),abs(r2-f['R2']),abs(np.median(pred[:10])))
        x6=archive['x_zero6'][i];power=(t/x6[4])**x6[5];oldpred=x6[0]+x6[1]*np.exp(-t/x6[2])+x6[3]*power/(1+power);oldmse=np.mean((oldpred-y)**2);curve_rows.append(dict(index=i,n=len(y),six_R2=1-oldmse/np.var(y),five_R2=r2,six_MSE=oldmse,five_MSE=sse/len(y),five_MAE=np.mean(abs(pred-y)),selected_converged=f['selected_converged'],boundaries=f['boundaries']))
        start=np.sort(archive['original'][i,:10]);offset=(start[4]+start[5])/2;assert abs(offset-archive['baseline'][i])<1e-12;assert np.allclose(archive['original'][i,obs]-offset,y,rtol=0,atol=1e-12);assert np.array_equal(archive['x_native'][i,1:],x6[1:]);assert np.all(q[[1,3]]>=1-1e-10) and np.all(q[[1,3]]<=900+1e-8) and .2-1e-10<=q[4]<=12+1e-8
    r.write('private/CURVE_FIT_INDEPENDENT.json',curve_rows)
    for key,run in splits.items():
        obj=joblib.load(r.ROOT/f'runs/{key}.joblib');assert r.sha(r.ROOT/f'runs/{key}.joblib')==freeze['runs'][key]['run_hash'];d=run['d'];si=run['seed']-1000;Q,C,src=run['test'],run['C'],run['source'];groups=data['groups']
        for a,b in [(src,C),(src,Q),(C,Q)]:assert not set(groups[a])&set(groups[b])
        for arm in ['RAW6','ZERO6','ZERO5']:
            X=data[{'RAW6':'x','ZERO6':'x_zero6','ZERO5':'x_zero5'}[arm]]
            for f,fold in enumerate(run['source_folds']+[dict(T=src,V=[])]):
                p=(r.PREV/f'models/{key}_CI-CLIN_map{f}.joblib') if arm=='RAW6' else r.ROOT/f'models/{key}_{arm}_map{f}.joblib';m=joblib.load(p);nmap+=1;T,V=fold['T'],fold['V'];assert np.array_equal(m['train'],T) and np.array_equal(m['validation'],V);assert not set(groups[T])&set(groups[V]);params=1680 if arm!='ZERO5' else 1656;active=1608 if arm!='ZERO5' else 1596;assert m['params']==params and m['active']==active
                for name,v in [('xm',X[T].mean(0)),('xs',np.maximum(X[T].std(0),1e-6)),('ym',data['medical'][T].mean(0)),('ys',np.maximum(data['medical'][T].std(0),1e-6))]:mx['scales']=max(mx['scales'],delta(m[name],v))
                mx['weights']=max(mx['weights'],delta(m['CI_weights'],weights(data['ci'][T])))
                if arm=='ZERO6':assert m['initial_hash']==joblib.load(r.PREV/f'models/{key}_CI-CLIN_map{f}.joblib')['initial_hash']
                if arm!='RAW6':assert m['code_hash']==r.sha(r.ROOT/'core.py')
                if V:
                    best=np.inf;chosen=None
                    for row in m['trace']:
                        if 'validation_MED4_MAE' in row and row['validation_MED4_MAE']<best-1e-10:best=row['validation_MED4_MAE'];chosen=row['step']
                    assert m['steps']==chosen;ind=project(source(m,X[V]));mx['prediction']=max(mx['prediction'],delta(ind,obj['fold_predictions'][arm][f]['pred']));loc={q:i for i,q in enumerate(src)};mx['prediction']=max(mx['prediction'],delta(ind,obj['OOF'][arm][[loc[q] for q in V]]))
                else:
                    raw=source(m,X[Q]);mx['prediction']=max(mx['prediction'],delta(project(raw),saved[d][arm+'-RAW'][si]),delta(project(source(m,X[src])),sp[key][arm]['train']));assert m['steps']==int(np.median(obj['stage_metrics'][arm]['inner_steps']))
            m=obj['models'][arm];lookup=dict(zip(C,source(m,X[C])));rawQ=source(m,X[Q])
            for a in obj['calibrators'][arm]:
                ncal+=1;A,B=a['A'],a['B'];mod=a['cal'];rawA=np.array([lookup[i] for i in A]);mx['prediction']=max(mx['prediction'],delta(cal(mod,rawA),a['train_med']));assert not set(groups[A])&set(groups[B]);assert not set(groups[A])&set(groups[Q])
                if B:mx['prediction']=max(mx['prediction'],delta(cal(mod,np.array([lookup[i] for i in B])),a['validation_med']))
                if a['code']==3:mx['prediction']=max(mx['prediction'],delta(cal(mod,rawQ),saved[d][arm+'-H'][si]))
                u=(data['medical'][A]-m['ym'])/m['ys'];v=(project(rawA)-m['ym'])/m['ys'];mx['scales']=max(mx['scales'],delta(u.mean(0),mod['ubar']),delta(v.mean(0),mod['vbar']));e=(v-v.mean(0))@mod['matrix']+mod['intercept']-(u-u.mean(0));value=np.where(abs(e)<=1.345,e*e,2*1.345*abs(e)-1.345**2).sum()+10*np.square(mod['matrix']-np.eye(4)).sum();mx['Huber_objective']=max(mx['Huber_objective'],abs(value-mod['objective']))
            if arm=='RAW6':
                old=np.load(r.PREV/f'private/D{d}_predictions.npz');mx['prediction']=max(mx['prediction'],delta(saved[d]['RAW6-RAW'][si],old['CI-CLIN-RAW'][si]),delta(saved[d]['RAW6-H'][si],old['CI-CLIN-H'][si]))
    for d in [1,2]:
        dn=f'D{d}';a=saved[d];Y=a['reference_med4'];J=a['reference'];y=label(J);W=np.load(r.PREV/f'private/{dn}_bootstrap.npz')['weights'];Q=a['indices'];groups=data['groups'][Q];bs={};rr=r.read('ROC_DIAGNOSTICS.json')[dn]
        for g in np.unique(groups):
            ids=np.flatnonzero(groups==g);assert np.all(W[:,ids]==W[:,ids[:1]])
        for name,row in tables[dn].items():
            st=stats(a[name],Y,J)
            for metric in ['accuracy','BA','normal_FP','CI_MAE','macro_abnormal_AUROC','MED4_MAE']:mx['metrics']=max(mx['metrics'],delta(st[metric],row[metric]))
            if name.endswith('-H'):bs[name]=boot(a[name],J,W)
            for k in [-1,1]:
                if not np.any(y==k):assert rr[name][str(k)]['AUROC'] is None;continue
                au=[]
                for si,j in enumerate(ci(a[name])):
                    p=j[y==k]*k;n=j[y==0]*k;au.append(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])));ths=np.r_[np.inf,np.unique(np.r_[p,n])[::-1]];fp=(n[None,:]>=ths[:,None]).mean(1);tp=(p[None,:]>=ths[:,None]).mean(1)
                    for q in [.05,.1,.2]:mx['ROC']=max(mx['ROC'],abs(float(np.interp(q,fp,tp))-rr[name][str(k)]['per_seed'][si]['TPR_at_FPR'][str(q)]))
                mx['ROC']=max(mx['ROC'],abs(np.mean(au)-rr[name][str(k)]['AUROC']))
        for row in r.read('PRIMARY_STATISTICS.json'):
            if row['direction']!=dn:continue
            sign=-1 if row['metric'] in ['normal_FP','CI_MAE'] else 1;values=sign*(bs[row['method']][row['metric']]-bs[row['control']][row['metric']]);values=values[np.isfinite(values)];assert len(values)==row['valid_bootstrap'];mx['interval']=max(mx['interval'],delta(np.quantile(values,[.025,.975]),row['CI95']),delta(np.quantile(values,[.05/60,1-.05/60]),row['Bonferroni']))
    ledger=[r.read(p) for p in (r.ROOT/'ledger').glob('*.json')];assert len(ledger)==1043 and all(z['state']=='finished' for z in ledger);assert sum(z['kind']=='mapping' for z in ledger)==320 and sum(z['kind']=='Huber' for z in ledger)==320 and sum(z['kind']=='curve_record' for z in ledger)==401
    limits={k:(1e-7 if k in ['curve_metrics','Huber_objective'] else 1e-8) for k in mx};assert all(mx[k]<limits[k] for k in mx),mx;r.write('VERIFICATION.json',dict(passed=True,time=time.time(),maximum_difference=mx,tolerances=limits,curve_records=401,source_models_replayed=nmap,calibrators_replayed=ncal,complete_runs=40,independent_NumPy_forward_and_metrics=True,independent_curve_formula=True,raw_training_exact_replay=True,fit_positions=1043,science_positions=1041,primary_family=30,verification_code_sha256=r.sha(__file__)));print('VERIFIED',mx,flush=True)

if __name__=='__main__':main()
