"""Independent forward, selection, dependency and statistical replay. No core/evaluate imports."""
import runtime as r
import numpy as np,joblib,time,importlib.util
from scipy.special import softmax,expit
r.extract(r.ROOT.parent/'uteg_zero5_med4_20261006_113257/verify_independent.py',['source'],globals());source_tt=source
r.extract(r.CLIN/'verify_independent.py',['project','ci','label','cal','delta','ratio','stats','boot'],globals())
r.extract(r.GATE/'independent_statistics.py',['areas'],globals())
sp=importlib.util.spec_from_file_location('numpy_gate_verification',r.GATE/'independent_models.py');im=importlib.util.module_from_spec(sp);sp.loader.exec_module(im);im.N_FEATURES=13

def source(m,X):
    fam=m.get('family','TT');z=(X-m['xm'])/m['xs']
    if not m['success']:return np.full((len(X),4),np.nan)
    if fam=='TT':return source_tt(m,X)
    if fam in ['RIDGE','PLS']:yp=z@m['coef']+m['intercept']
    elif fam=='MLP':
        s=m['state'];h=z@s['0.weight'].T+s['0.bias'];yp=(h*expit(h))@s['2.weight'].T+s['2.bias']
    else:
        def kern(v):return np.exp(-np.sum((z[:,None,:]-v[None,:,:])**2,axis=-1)/4)
        if fam=='KRR':yp=kern(m['train_z'])@m['dual']
        else:yp=np.column_stack([kern(v['support'])@v['dual']+v['intercept'] for v in m['svm']])
    return m['ym']+m['ys']*yp

def calibrated(a,raw):return cal(a,raw) if a['success'] and a['finite'] else np.full_like(raw,np.nan)
def features(X,M,m):
    J=ci(M);y=label(J);onehot=np.eye(3)[np.clip(y+1,0,2)];onehot[y==9]=0
    return np.concatenate([X,(M-m['ym'])/m['ys'],J[:,None]/3,np.minimum(abs(J[:,None]+3),abs(J[:,None]-3))/3,onehot],axis=1)

def thres(score,yh,yt):
    known=yt!=9;valid=known&(yh!=9)&np.isfinite(score);err=(yh!=yt).astype(float);area=areas(score[valid],err[valid],np.ones((1,int(sum(valid)))))[0] if valid.any() else np.nan;opts=[]
    for t in sorted(set(score[valid])):
        a=known&(yh!=9)&np.isfinite(score)&(score<=t);n=int(sum(a));risk=float(err[a].mean());cov=n/sum(known);ab=known&(yt!=0);afn=float(sum(a&ab&(yh==0))/sum(ab)) if ab.any() else None
        if n>=5 and cov>=.2 and risk<=.05+1e-12:opts.append(dict(threshold=float(t),coverage=cov,risk=risk,n=n,abnormal_false_normal=afn))
    if not opts:return dict(eligible=False,threshold=None,coverage=0.,risk=None,n=0,aurc=area)
    best=sorted(opts,key=lambda z:(-z['coverage'],z['risk'],1 if z['abnormal_false_normal'] is None else z['abnormal_false_normal'],z['threshold']))[0];return dict(best,eligible=True,aurc=area)

def select(entries,yh,yt):
    rows={n:dict(thres(z['score'],yh,yt),config=n,family=z['family'],params=z['params'],order=z['order']) for n,z in entries.items()}
    def choose(vs):
        qualified=[v for v in vs if v['eligible']]
        if qualified:return sorted(qualified,key=lambda v:(-round(v['coverage'],12),round(v['risk'],12),round(v['abnormal_false_normal'] if v['abnormal_false_normal'] is not None else 1,12),v['aurc'] if np.isfinite(v['aurc']) else np.inf,v['params'],v['order']))[0]
        z=sorted(vs,key=lambda v:(v['aurc'] if np.isfinite(v['aurc']) else np.inf,v['params'],v['order']))[0];return dict(z,threshold=None,eligible=False,coverage=0.,risk=None,n=0)
    fam={f:choose([z for z in rows.values() if z['family']==f]) for f in dict.fromkeys(z['family'] for z in rows.values())};return dict(configurations=rows,families=fam,selected=choose(list(fam.values())))

def main():
    assert r.read('EVALUATION_COMPLETE.json')['runs']==40
    for fname in ['PROTOCOL_FREEZE.json','INPUT_IDENTITY.json']:
        for path,h in r.read(fname)['files'].items():assert r.sha(path)==h,path
    assert r.sha(r.ROOT/'PROMPT.md')==r.read('PROTOCOL_FREEZE.json')['prompt_hash'];assert r.sha(r.ROOT/'evaluate.py')==r.read('EVALUATION_CODE_FREEZE.json')['sha']
    data=dict(np.load(r.ROOT/'private/data.npz'));splits=r.read('SPLIT_MANIFEST.json');mf=r.read('MODEL_FREEZE.json');assert mf['time']<r.read('EVALUATION_COMPLETE.json')['time'];mx={k:0. for k in ['mapping','scales','calibration','Huber_objective','gate','features','selection','metrics','bootstrap','interval']};counts=dict(mapping=0,calibration=0,gate=0);saved={d:dict(np.load(r.ROOT/f'private/D{d}_predictions.npz')) for d in [1,2]};gpred={d:joblib.load(r.ROOT/f'private/D{d}_gate_predictions.joblib') for d in [1,2]}
    def check(k,a,b):mx[k]=max(mx[k],delta(a,b))
    for key,run in splits.items():
        assert r.sha(r.ROOT/f'runs/{key}.joblib')==mf['runs'][key]['run_hash'];obj=joblib.load(r.ROOT/f'runs/{key}.joblib');S,C,Q=run['source'],run['C'],run['test'];d=run['d'];si=run['seed']-1000;groups=data['groups'];X=data['x'];Y=data['medical'];J=data['ci'];raws={}
        for a,b in [(S,C),(S,Q),(C,Q)]:assert not set(groups[a])&set(groups[b])
        for name,folds in obj['fold_models'].items():
            predlookup={}
            for fold in folds:
                m=joblib.load(fold['path']);T,V=fold['T'],fold['V'];assert list(m['train'])==T and list(m['validation'])==V;assert not set(groups[T])&set(groups[V]);counts['mapping']+=1
                for field,arr in [('xm',X[T].mean(0)),('xs',np.maximum(X[T].std(0),1e-6)),('ym',Y[T].mean(0)),('ys',np.maximum(Y[T].std(0),1e-6))]:check('scales',m[field],arr)
                p=project(source(m,X[V]));check('mapping',p,fold['pred']);predlookup.update(zip(V,p))
            check('mapping',np.array([predlookup[i] for i in S]),obj['source_OOF'][name])
        sf={}
        for fam,item in obj['families'].items():
            candidates=[n for n in obj['source_OOF'] if n.split('-')[0]==fam];scores={}
            for n in candidates:
                st=stats(obj['source_OOF'][n][None],Y[S],J[S]);scores[n]=.5*st['CI_MAE']/3+.5*(1-st['BA']);check('selection',np.array(scores[n]),np.array(item['selection_scores'][n]['S']))
            chosen=next(n for n in candidates if scores[n]<=min(scores.values())+.001);assert chosen==item['config'];sf[fam]=scores[chosen];m=item['model'];assert list(m['train'])==S;counts['mapping']+=1;raws[fam]=dict(zip(C,source(m,X[C])));rawq=source(m,X[Q]);check('mapping',project(rawq),saved[d][fam+'-RAW'][si])
            for pos in obj['calibrators'][fam]:
                a=pos['cal'];A,B=pos['A'],pos['B'];pa=calibrated(a,np.array([raws[fam][i] for i in A]));check('calibration',pa,pos['train_med'])
                if pos['code']<3:check('calibration',calibrated(a,np.array([raws[fam][i] for i in B])),pos['validation_med'])
                else:check('calibration',calibrated(a,rawq),saved[d][fam+'-H'][si])
        chosen=next(f for f in sf if sf[f]<=min(sf.values())+.001);assert chosen==obj['source_selection'];check('mapping',saved[d]['MAPPING-SELECTED-H'][si],saved[d][chosen+'-H'][si])
        for item in obj['calibration_objects'].values():
            A,fam,a=item['A'],item['family'],item['cal'];m=obj['families'][fam]['model'];raw=np.array([raws[fam][i] for i in A]);v=(project(raw)-m['ym'])/m['ys'];u=(Y[A]-m['ym'])/m['ys'];check('scales',a['vbar'],v.mean(0));check('scales',a['ubar'],u.mean(0));e=(v-v.mean(0))@a['matrix']+a['intercept']-(u-u.mean(0));val=np.where(abs(e)<1.345,e*e,2*1.345*abs(e)-1.345**2).sum()+10*np.square(a['matrix']-np.eye(4)).sum();check('Huber_objective',np.array(val),np.array(a['objective']));assert not set(groups[A])&set(groups[Q]);counts['calibration']+=1
        for base,z in obj['gates'].items():
            fam=z['family'];m=obj['families'][fam]['model'];lookup={i:j for j,i in enumerate(C)}
            for code,ex in z['examples'].items():
                A=ex['A'];B=ex.get('B',[]);built={}
                for part in ex['nested']:
                    T,V=part['T'],part['V'];assert not set(groups[T])&set(groups[V]);assert not set(groups[T+V])&set(groups[B]);assert not set(T+V)&set(Q)
                    a=next(v['cal'] for v in obj['calibration_objects'].values() if v['family']==fam and v['A']==T);built.update(zip(V,calibrated(a,np.array([raws[fam][i] for i in V]))))
                pm=np.array([built[i] for i in A]);check('calibration',pm,ex['train_med']);F=features(X[A],pm,m);check('features',F,ex['F']);assert F.shape[1]==13
                if code<3:
                    a=obj['calibrators'][fam][code]['cal'];pm=calibrated(a,np.array([raws[fam][i] for i in B]));vf=features(X[B],pm,m);check('features',vf,ex['validation_F']);yh=label(ci(pm));ix=[lookup[i] for i in B]
                else:
                    pm=saved[d][fam+'-H'][si];vf=features(X[Q],pm,m);yh=label(ci(pm))
                for nm,gm in z['models'][code].items():
                    good=yh!=9;sc=np.full(len(yh),np.inf);sc[good]=im.predict_independent(gm,vf[good],yh[good]);counts['gate']+=1
                    if code<3:check('gate',sc,z['OOF_entries'][nm]['score'][ix])
                    else:
                        for gn,sl in {**z['selection']['families'],'TRUST-SELECTED':z['selection']['selected']}.items():
                            if sl['config']==nm:
                                out=gpred[d][base+'::'+gn];check('gate',sc,out['score'][si]);act=(good&np.isfinite(sc)&(sc<=sl['threshold'])) if sl['threshold'] is not None else np.zeros(len(sc),bool);assert np.array_equal(act,out['action'][si]);check('mapping',pm,out['med'][si])
            yy=label(J[C]);yh=label(ci(z['OOF_med']));ours=select(z['OOF_entries'],yh,yy)
            for section in ['configurations','families']:
                for name,rr in ours[section].items():
                    old=z['selection'][section][name];assert rr['eligible']==old['eligible'] and rr['config']==old['config'] and rr['threshold']==old['threshold'];check('selection',np.array(rr['aurc']),np.array(old['aurc']))
            assert ours['selected']['config']==z['selection']['selected']['config'] and ours['selected']['threshold']==z['selection']['selected']['threshold']
        print('REPLAY',key,flush=True) if si%5==0 else None
    maps=r.read('MAPPING_METRICS.json');gtable=r.read('GATE_METRICS.json');mbi=r.read('MAPPING_STATISTICS.json');gbi=r.read('GATE_STATISTICS.json')
    for d in [1,2]:
        dn=f'D{d}';a=saved[d];J=a['reference'];Y=a['reference_med4'];Q=a['indices'];y=label(J);known=y!=9;W=np.load(r.ROOT/f'private/{dn}_bootstrap.npz')['weights'];oldboot=joblib.load(r.ROOT/f'private/{dn}_bootstrap_metrics.joblib');bmaps={};bgates={}
        for group in np.unique(data['groups'][Q]):
            ix=np.flatnonzero(data['groups'][Q]==group);assert np.all(W[:,ix]==W[:,ix[:1]])
        for name,row in maps[dn].items():
            st=stats(a[name],Y,J)
            for metric in ['accuracy','BA','normal_FP','CI_MAE','MED4_MAE']:check('metrics',np.array(st[metric]),np.array(row[metric]))
            if name.endswith('-H'):
                bmaps[name]=boot(a[name],J,W)
                for k in ['accuracy','BA','CI_MAE']:check('bootstrap',bmaps[name][k],oldboot['mapping'][name][k])
        for name,z in gpred[d].items():
            p=label(ci(z['med']));aa=z['action'].mean(0);joint=(z['action']*(p!=y)).mean(0);risk=joint[known].sum()/aa[known].sum() if aa[known].sum() else np.nan;row=gtable[dn][name];check('metrics',np.array(risk),np.array(row['risk'] if row['risk'] is not None else np.nan));check('metrics',np.array(aa.mean()),np.array(row['coverage']));area=[]
            for si in range(20):
                err=(p[si]!=y).astype(float);err[~known|(p[si]==9)]=np.nan;area.append(areas(z['score'][si],err,np.ones((1,len(y))))[0])
            check('metrics',np.array(np.mean(area)),np.array(row['AURC']))
            if name not in oldboot['gates']:continue
            br=ratio(W[:,known]@joint[known],W[:,known]@aa[known]);bc=ratio(W@aa,W.sum(1));ba=np.zeros(len(W))
            for si in range(20):
                err=(p[si]!=y).astype(float);err[~known|(p[si]==9)]=np.nan
                for lo in range(0,len(W),1000):ba[lo:lo+1000]+=areas(z['score'][si],err,W[lo:lo+1000])/20
            bgates[name]=dict(risk=br,coverage=bc,AURC=ba)
            for k,v in bgates[name].items():check('bootstrap',v,oldboot['gates'][name][k])
            print('INDEPENDENT BOOT',dn,name,flush=True)
        for rows,bs,size in [(mbi,bmaps,36),(gbi,bgates,30)]:
            for row in rows:
                if row['direction']!=dn:continue
                metric=row['metric'];sign=1 if metric in ['accuracy','BA','coverage'] else -1;dd=sign*(bs[row['method']][metric]-bs[row['control']][metric]);valid=dd[np.isfinite(dd)];assert len(valid)==row['valid_bootstrap']
                if len(valid):check('interval',np.quantile(valid,[.025,.975]),np.array(row['CI95']));check('interval',np.quantile(valid,[.05/(2*size),1-.05/(2*size)]),np.array(row['Bonferroni']))
        old=np.load(r.PREV/f'private/D{d}_predictions.npz');check('mapping',a['TT-H'],old['P5_F4-H'])
    for row in r.read(r.ROOT.parent/'uteg_platform_filter_pilot_20261006_124636/PROTOCOL_FREEZE.json')['raw_files']:assert r.sha(row['path'])==row['sha256']
    ledger=[r.read(p) for p in (r.ROOT/'ledger').glob('*.json')];assert all(z['state']=='finished' for z in ledger);assert sum(z['units'] for z in ledger)<=7640;assert all(v<(1e-7 if k=='Huber_objective' else 1e-8) for k,v in mx.items()),mx
    r.write('VERIFICATION.json',dict(passed=True,time=time.time(),maximum_difference=mx,counts=counts,raw401_unchanged=True,group_isolation=True,patient_independence_confirmed=False,strict_nested_dependencies=True,source_and_C20_selection_replayed=True,independent_numpy_forward=True,all_primary_bootstrap_recomputed=True,code_hash=r.sha(__file__)));print('VERIFIED',mx,flush=True)
if __name__=='__main__':main()
