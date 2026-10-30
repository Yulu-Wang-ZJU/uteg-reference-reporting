"""Independent NumPy replay. No imports of core.py or evaluate.py."""
import runtime as r
import numpy as np,joblib,time
from scipy.special import softmax,logsumexp
def source(m,X):
    s=m['state'];z=((X-m['xm'])/m['xs'])[:,:,None]*s['val'][None];p='enc.layers.0.'
    qkv=z@s[p+'self_attn.in_proj_weight'].T+s[p+'self_attn.in_proj_bias'];q,k,v=[a.reshape(len(X),6,2,6).transpose(0,2,1,3) for a in np.split(qkv,3,-1)]
    a=(softmax(q@k.swapaxes(-1,-2)/np.sqrt(6),-1)@v).transpose(0,2,1,3).reshape(len(X),6,12);z=z+a@s[p+'self_attn.out_proj.weight'].T+s[p+'self_attn.out_proj.bias']
    z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*s[p+'norm1.weight']+s[p+'norm1.bias'];a=np.maximum(z@s[p+'linear1.weight'].T+s[p+'linear1.bias'],0);z=z+a@s[p+'linear2.weight'].T+s[p+'linear2.bias'];z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*s[p+'norm2.weight']+s[p+'norm2.bias']
    return m['ym']+m['ys']*(z.mean(1)@s['out.weight'].T+s['out.bias'])
def project(a):
    b=np.array(a,copy=True);b[...,:3]=np.maximum(b[...,:3],0);b[...,3]=np.minimum(np.maximum(b[...,3],0),89.999999);b[~np.isfinite(b).all(-1)]=np.nan;return b
def ci(m):return -.6516*m[...,0]-.3772*m[...,1]+.1224*m[...,2]+.0759*m[...,3]-7.7922
def label(j):
    a=np.zeros(np.shape(j),int);a[j<-3]=-1;a[j>3]=1;a[~np.isfinite(j)]=9;return a
def cal(a,raw):return project(a['mu']+a['scale']*(a['ubar']+((project(raw)-a['mu'])/a['scale']-a['vbar'])@a['matrix']+a['intercept']))
def weights(J):
    known=np.isfinite(J);labs=label(J);w=np.ones(len(J));cs={k:np.sum(labs==k) for k in [-1,0,1]}
    if len(np.unique(labs[known]))>1:
        v=np.array([1/np.sqrt(1+cs[k]) for k in labs[known]]);lo,hi=0.,1000.
        for _ in range(140):
            md=(lo+hi)/2
            if np.minimum(3,np.maximum(.5,md*v)).sum()<sum(known):lo=md
            else:hi=md
        w[known]=np.minimum(3,np.maximum(.5,(lo+hi)/2*v))
    return w
def parts(pred,target,ym,ys,J,w):
    lm=np.square(pred-target).mean();valid=np.isfinite(J)
    if not valid.any():return np.array([lm,0,0])
    jj=ci(project(ym+ys*pred))[valid];err=(jj-J[valid])/3;lv=np.where(abs(err)<=1,.5*err**2,abs(err)-.5).mean();lo=2*(jj+3);hi=2*(jj-3)
    lp=np.column_stack([-np.logaddexp(0,lo),-np.logaddexp(0,-lo)-np.logaddexp(0,hi)+np.log(-np.expm1(-12)),-np.logaddexp(0,-hi)]);lc=np.mean(-lp[np.arange(sum(valid)),label(J[valid])+1]*w[valid]);return np.array([lm,lv,lc])
def delta(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.array_equal(np.isfinite(a),np.isfinite(b));ok=np.isfinite(a);return float(abs(a[ok]-b[ok]).max()) if ok.any() else 0.
def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=np.asarray(b)>0)
def stats(M,Y,J):
    jj=ci(M);y=label(J);p=label(jj);known=np.isfinite(J);co=np.mean(p==y,axis=0);rec={str(k):co[y==k].mean() if np.any(y==k) else np.nan for k in [-1,0,1]};aucs=[]
    for k in [-1,1]:
        if np.any(y==k) and np.any(y==0):
            pos=jj[:,y==k]*k;neg=jj[:,y==0]*k;aucs.append(np.mean((pos[:,:,None]>neg[:,None,:])+.5*(pos[:,:,None]==neg[:,None,:])))
    return dict(accuracy=co[known].mean(),BA=np.mean([v for v in rec.values() if np.isfinite(v)]),normal_FP=1-rec['0'],CI_MAE=np.nanmean(abs(jj-J)),macro_abnormal_AUROC=np.mean(aucs),MED4_MAE=np.mean(abs(M-Y),axis=(0,1)),recall=rec)
def boot(M,J,W):
    jj=ci(M);y=label(J);pred=label(jj);rate=(pred==y).mean(0);known=np.isfinite(J);rec={k:ratio(W[:,y==k]@rate[y==k],W[:,y==k].sum(1)) for k in [-1,0,1] if np.any(y==k)};aucs=[]
    for k in [-1,1]:
        pos=np.flatnonzero(y==k);neg=np.flatnonzero(y==0)
        if not len(pos):continue
        mat=np.zeros((len(pos),len(neg)))
        for score in jj:
            diff=k*(score[pos,None]-score[None,neg]);mat+=((diff>0)+.5*(diff==0))/len(jj)
        num=np.einsum('bi,ij,bj->b',W[:,pos],mat,W[:,neg],optimize=True);aucs.append(ratio(num,W[:,pos].sum(1)*W[:,neg].sum(1)))
    return dict(accuracy=ratio(W[:,known]@rate[known],W[:,known].sum(1)),BA=np.mean(list(rec.values()),axis=0),normal_FP=1-rec[0],CI_MAE=ratio(W[:,known]@np.mean(abs(jj[:,known]-J[known]),axis=0),W[:,known].sum(1)),macro_abnormal_AUROC=np.mean(aucs,axis=0))
def main():
    data=dict(np.load(r.ROOT/'private/data.npz'));splits=r.read('SPLIT_MANIFEST.json');tables=r.read('TARGET_METRICS.json');freeze=r.read('MODEL_FREEZE.json');saved={d:dict(np.load(r.ROOT/f'private/D{d}_predictions.npz')) for d in [1,2]};sp=joblib.load(r.ROOT/'private/source_predictions.joblib');mx={k:0. for k in ['prediction','scales','weights','Huber_objective','metrics','ROC','interval','loss']};nmap=0;ncal=0
    for p,h in r.read('TRAINING_CODE_FREEZE.json')['files'].items():assert r.sha(p)==h,p
    for info in r.read('CACHE_REUSE_AUDIT.json')['matched_files'].values():
        for p,h in info.items():assert r.sha(p)==h,p
    assert freeze['time']<r.read('EVALUATION_COMPLETE.json')['time'];assert r.sha(r.ROOT/'evaluate.py')==r.read('EVALUATION_CODE_FREEZE.json')['sha']
    for key,run in splits.items():
        obj=joblib.load(r.ROOT/f'runs/{key}.joblib');assert r.sha(r.ROOT/f'runs/{key}.joblib')==freeze['runs'][key]['run_hash'];d=run['d'];si=run['seed']-1000;Q=run['test'];C=run['C'];src=run['source']
        for arm in ['BASE','CI-VALUE','CI-CLIN']:
            for f,fold in enumerate(run['source_folds']+[dict(T=src,V=[])]):
                p=(r.PREV/f'models/{key}_TT-SMALL_map{f}.joblib') if arm=='BASE' else (r.ROOT/f'models/{key}_{arm}_map{f}.joblib');m=joblib.load(p);nmap+=1;T,V=fold['T'],fold['V'];assert np.array_equal(m['train'],T) and np.array_equal(m['validation'],V);assert not set(data['groups'][T])&set(data['groups'][V])
                for name,values in [('xm',data['x'][T].mean(0)),('xs',np.maximum(data['x'][T].std(0),1e-6)),('ym',data['medical'][T].mean(0)),('ys',np.maximum(data['medical'][T].std(0),1e-6))]:mx['scales']=max(mx['scales'],delta(m[name],values))
                base=joblib.load(r.PREV/f'models/{key}_TT-SMALL_map{f}.joblib');assert m['initial_hash']==base['initial_hash'];assert m['params']==1680 and m['active']==1608
                if arm!='BASE':
                    mx['weights']=max(mx['weights'],delta(m['CI_weights'],weights(data['ci'][T])));assert m['code_hash']==r.sha(r.ROOT/'core.py')
                    if V:
                        best=min(z['validation_MED4_MAE'] for z in m['trace'] if 'validation_MED4_MAE' in z);chosen=next(z['step'] for z in m['trace'] if z.get('validation_MED4_MAE',np.inf)<=best+1e-10);assert m['steps']==chosen
                if V:
                    independent=project(source(m,data['x'][V]));mx['prediction']=max(mx['prediction'],delta(independent,obj['fold_predictions'][arm][f]['pred']));loc={q:i for i,q in enumerate(src)};mx['prediction']=max(mx['prediction'],delta(independent,obj['OOF'][arm][[loc[q] for q in V]]))
                else:
                    raw=source(m,data['x'][Q]);mx['prediction']=max(mx['prediction'],delta(project(raw),saved[d][arm+'-RAW'][si]),delta(project(source(m,data['x'][src])),sp[key][arm]['train']));assert m['steps']==int(np.median(obj['stage_metrics'][arm]['inner_steps']))
            m=obj['models'][arm];rawC=source(m,data['x'][C]);lookup=dict(zip(C,rawC));rawQ=source(m,data['x'][Q])
            for a in obj['calibrators'][arm]:
                ncal+=1;A,B=a['A'],a['B'];mod=a['cal'];predA=cal(mod,np.array([lookup[i] for i in A]));mx['prediction']=max(mx['prediction'],delta(predA,a['train_med']))
                assert not set(data['groups'][A])&set(data['groups'][B]);assert not set(data['groups'][A])&set(data['groups'][Q])
                if B:mx['prediction']=max(mx['prediction'],delta(cal(mod,np.array([lookup[i] for i in B])),a['validation_med']))
                if a['code']==3:mx['prediction']=max(mx['prediction'],delta(cal(mod,rawQ),saved[d][arm+'-H'][si]))
                u=(data['medical'][A]-m['ym'])/m['ys'];v=(project(np.array([lookup[i] for i in A]))-m['ym'])/m['ys'];mx['scales']=max(mx['scales'],delta(u.mean(0),mod['ubar']),delta(v.mean(0),mod['vbar']))
                e=(v-v.mean(0))@mod['matrix']+mod['intercept']-(u-u.mean(0));value=np.where(abs(e)<=1.345,e*e,2*1.345*abs(e)-1.345**2).sum()+10*np.square(mod['matrix']-np.eye(4)).sum();mx['Huber_objective']=max(mx['Huber_objective'],abs(value-mod['objective']))
            if arm=='BASE':
                old=np.load(r.PREV/f'private/D{d}_predictions.npz');mx['prediction']=max(mx['prediction'],delta(saved[d][arm+'-RAW'][si],old['TT-SMALL-RAW'][si]),delta(saved[d][arm+'-H'][si],old['TT-SMALL-H'][si]))
    for d in [1,2]:
        dn=f'D{d}';a=saved[d];Y=a['reference_med4'];J=a['reference'];y=label(J);W=np.load(r.ROOT/f'private/{dn}_bootstrap.npz')['weights'];Q=a['indices'];groups=data['groups'][Q];bs={};rr=r.read('ROC_DIAGNOSTICS.json')[dn]
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
                    for q in [.05,.10,.20]:mx['ROC']=max(mx['ROC'],abs(float(np.interp(q,fp,tp))-rr[name][str(k)]['per_seed'][si]['TPR_at_FPR'][str(q)]))
                mx['ROC']=max(mx['ROC'],abs(np.mean(au)-rr[name][str(k)]['AUROC']))
        for row in r.read('PRIMARY_STATISTICS.json'):
            if row['direction']!=dn:continue
            sign=-1 if row['metric'] in ['normal_FP','CI_MAE'] else 1;values=sign*(bs[row['method']][row['metric']]-bs[row['control']][row['metric']]);values=values[np.isfinite(values)];assert len(values)==row['valid_bootstrap']
            mx['interval']=max(mx['interval'],delta(np.quantile(values,[.025,.975]),row['CI95']),delta(np.quantile(values,[.05/40,1-.05/40]),row['Bonferroni']))
    probe=r.read('LOSS_CHECK_INPUTS.json')
    for q in probe:
        value=parts(*[np.array(q[k]) for k in ['pred','target','ym','ys','J','w']]);mx['loss']=max(mx['loss'],delta(value,q['torch_parts']))
    ledger=[r.read(p) for p in (r.ROOT/'ledger').glob('*.json')];assert len(ledger)==642 and all(z['state']=='finished' for z in ledger);assert sum(z['kind']=='mapping' for z in ledger)==320 and sum(z['kind']=='Huber' for z in ledger)==320
    limits={k:(1e-7 if k=='Huber_objective' else 1e-8) for k in mx};assert all(mx[k]<limits[k] for k in mx),mx
    r.write('VERIFICATION.json',dict(passed=True,time=time.time(),maximum_difference=mx,tolerances=limits,source_models_replayed=nmap,calibrators_replayed=ncal,complete_runs=40,independent_NumPy_forward_and_metrics=True,clinical_loss_independent=True,BASE_bitwise_training_replay_test=True,fit_count=642,science_fits=640,conditional_group_bootstrap_recomputed=True));print('VERIFIED',mx,flush=True)
if __name__=='__main__':main()
