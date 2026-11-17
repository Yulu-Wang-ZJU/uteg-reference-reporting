"""Isolated 90% stratified group subsampling; current Table 1/2 estimators."""
from pathlib import Path
import os, sys, json, hashlib, argparse, time, csv, warnings
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from code_submit.paper_inward4_replay import core as c, current_version as current, current_suite as suite, mapping, gates
from code_submit.paper_b085_hospital_replay.adaptation_controls import fit_cal,pred_cal

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(safe(x),indent=2,ensure_ascii=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(y):return np.where(y<50,0,np.where(y>70,2,1))

def quota(ix,d):
    ct=np.bincount(state(d['medical'][ix,2]),minlength=3)
    n=int(np.floor(.9*len(ix)+.5));v=ct*n/len(ix);q=np.floor(v).astype(int)
    order=sorted(range(3),key=lambda k:(-(v[k]-q[k]),k))
    for k in order[:n-q.sum()]:q[k]+=1
    return q

def sample(ix,d,rng):
    ix=np.asarray(ix,int);q=quota(ix,d);g=d['groups'][ix];ys=state(d['medical'][:,2])
    units=[ix[g==a] for a in np.unique(g)]
    multi=[a for a in units if len(a)>1]
    singles=[np.array([a[0] for a in units if len(a)==1 and ys[a[0]]==k],int) for k in range(3)]
    for attempt in range(10000):
        chosen=[a for a in multi if rng.random()<.9]
        used=np.concatenate(chosen) if chosen else np.array([],int)
        rem=q-np.bincount(ys[used],minlength=3)
        if any(rem[k]<0 or rem[k]>len(singles[k]) for k in range(3)):continue
        selected=np.sort(np.r_[used,*[rng.choice(singles[k],int(rem[k]),replace=False) for k in range(3)]])
        assert len(selected)==len(set(selected)) and np.array_equal(np.bincount(ys[selected],minlength=3),q)
        assert not set(d['groups'][selected])&set(d['groups'][np.setdiff1d(ix,selected)])
        return selected
    raise RuntimeError('No feasible grouped draw for quotas '+str(q))

def folds(ix,d,seed):
    ix=np.asarray(ix,int)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The least populated class')
        f=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
        result=[dict(T=ix[a],V=ix[b]) for a,b in f.split(ix,state(d['medical'][ix,2]),d['groups'][ix])]
    for f in result:
        assert not set(d['groups'][f['T']])&set(d['groups'][f['V']])
    return result

def prepare(d,plan,role):
    p=OUT/'private/SPLITS.json'
    if p.exists():return json.loads(p.read_text(encoding='utf8'))
    allplans=[]
    for rep in range(100):
        seed=202611160+rep;rng=np.random.default_rng(seed)
        sources={ct:sample(np.flatnonzero(d['centers']==ct),d,rng) for ct in ['SAHZU','Huzhou','Hubei']}
        targets={}
        for ct,r in role['roles'].items():
            a=sample(r['pool_full'],d,rng);b=sample(r['test'],d,rng)
            assert not set(d['groups'][a])&set(d['groups'][b])
            outer=folds(a,d,seed);positions=[]
            for j,f in enumerate(outer):positions.append(dict(A=f['T'],B=f['V'],inner=folds(f['T'],d,seed+101+j)))
            positions.append(dict(A=a,B=b,inner=folds(a,d,seed+201)))
            targets[ct]=dict(pool=a,test=b,positions=positions)
        allplans.append(dict(rep=rep,seed=seed,sources=sources,targets=targets))
    write(p,allplans)
    counts=[]
    for ct in ['SAHZU','Huzhou','Hubei']:
        for purpose,ix in [('source',sources[ct]),('development',targets[ct]['pool']),('test',targets[ct]['test'])]:
            counts.append(dict(center=ct,role=purpose,n=len(ix),classes=np.bincount(state(d['medical'][ix,2]),minlength=3)))
    write(OUT/'SAMPLE_COUNTS.json',counts)
    return safe(allplans)

def scoring(out,d,seed,arm,pos):
    a=out['train'];b=out['query'];p=out['inner'];pq=out['prediction']
    _,_,f=gates.fields(a,out['inner_b'],p,a,900);_,_,fq=gates.fields(b,out['query_b'],pq,a,900)
    pc=gates.fr.ma_class(p);t=gates.fr.ma_class(d['medical'][a,2]);qc=gates.fr.ma_class(pq)
    lr=gates.fr._fit_ma_gate(f,pc,t,f'inward_{arm}_900_{seed}_{pos}',seed)
    hs=gates.rc.fit_hgb(f,pc,t,np.ones(len(a),bool),direction=arm,seed=seed,pos=pos)
    return dict(Logistic=gates.fr.score_ma_gate(lr,fq,qc),HGB=gates.rc.score_hgb(hs,fq,qc,np.ones(len(b),bool)),Boundary=gates.rc.margin_scores(pq)),dict(lr_fallback=lr.get('fallback',False),hgb_status=hs['status'])

def reporting(d,sp,arm,seed):
    dev={k:[] for k in ['Logistic','HGB','Boundary']};dp=[];dy=[];status=[]
    for pos,part in enumerate(sp['positions'][:3]):
        o=c.evaluate(part['A'],part['B'],part['inner'],arm,900)
        scores,st=scoring(o,d,seed,arm,pos);status.append(st)
        for k in dev:dev[k].extend(scores[k])
        dp.extend(gates.fr.ma_class(o['prediction']));dy.extend(gates.fr.ma_class(d['medical'][o['query'],2]))
    part=sp['positions'][3];o=c.evaluate(sp['pool'],sp['test'],part['inner'],arm,900)
    scores,st=scoring(o,d,seed,arm,3);status.append(st)
    actions={'All report':np.ones(len(sp['test']),bool)};thresholds={}
    for k in dev:
        fit=gates.rc.select_threshold(np.asarray(dev[k]),np.asarray(dp),np.asarray(dy),.15);t=fit.get('threshold')
        actions[k]=np.zeros(len(sp['test']),bool) if t is None else np.isfinite(scores[k])&(scores[k]<=t)
        thresholds[k]=fit
    return o,actions,dict(thresholds=thresholds,status=status),scores

def metrics(p,y,act):
    p=np.asarray(p);y=np.asarray(y);a=np.asarray(act,bool);t=state(y);q=state(p)
    ab=a&(t!=1);norm=a&(t==1);ok=a&(t==q)
    div=lambda n,de:float(n/de) if de else None
    fn=int(np.sum(ab&(q==1)));fp=int(np.sum(norm&(q!=1)))
    v=dict(n=len(y),reported=int(a.sum()),correct=int(ok.sum()),incorrect=int((a&~ok).sum()),
        accuracy=div(ok.sum(),a.sum()),reporting_rate=div(a.sum(),len(y)),FNR=div(fn,ab.sum()),FPR=div(fp,norm.sum()),FN=fn,FP=fp,abnormal_released=int(ab.sum()),normal_released=int(norm.sum()),
        MAE=float(abs(p[a]-y[a]).mean()) if a.any() else None,RMSE=float(np.sqrt(np.mean((p[a]-y[a])**2))) if a.any() else None,R2=float(r2_score(y[a],p[a])) if a.sum()>1 and np.ptp(y[a]) else None)
    for k,name in enumerate(['low','normal','high']):
        m=a&(t==k);v[name+'_recall']=div(np.sum(m&(q==k)),m.sum());v[name+'_correct']=int(np.sum(m&(q==k)))
        v[name+'_MAE']=float(abs(p[m]-y[m]).mean()) if m.any() else None
        v[name+'_total']=int((t==k).sum());v[name+'_retained_correct_rate']=div(np.sum(m&(q==k)),(t==k).sum())
    v['confusion']=[[int(np.sum(a&(t==i)&(q==j))) for j in range(3)] for i in range(3)]
    return v

def four_metrics(p,y):return {name+'_MAE':float(np.mean(abs(p[:,j]-y[:,j]))) for j,name in enumerate(['R','K','MA','angle'])}

def init():
    d,_,_,plan,role,*_=current.activate();suite.ROOT=OUT
    m=suite.numerical_models();return d,plan,role,m

def smoke(d,plan,role,m):
    arm='HZ+HB_to_SA';sp=c.paper_spec(arm,1000);o,acts,audit,scores=reporting(d,sp,arm,1000)
    old=np.load(current.ROOT/'gates'/f'{arm}_T900.npz',allow_pickle=False)
    np.testing.assert_allclose(o['prediction'],old['prediction'][0],atol=1e-9,rtol=0)
    for a,k in [('Boundary','BOUNDARY_PAPER'),('Logistic','LR'),('HGB','HGB')]:
        np.testing.assert_array_equal(acts[a],old[k+'_action'][0]);np.testing.assert_allclose(scores[a],old[{'Boundary':'BOUNDARY','Logistic':'LR','HGB':'HGB'}[a]+'_score'][0],atol=1e-9,rtol=0)
    original=d['medical'][sp['test']].copy()
    try:
        d['medical'][sp['test']]=123456.;d['y'][sp['test']]=123456.
        c._SOURCE.clear();op,ap,au,_=reporting(d,sp,arm,1000)
        np.testing.assert_allclose(o['prediction'],op['prediction'],rtol=0,atol=1e-9)
        for k in acts:np.testing.assert_array_equal(acts[k],ap[k])
        assert safe(audit['thresholds'])==safe(au['thresholds'])
    finally:d['medical'][sp['test']]=original;d['y'][sp['test']]=original;c._SOURCE.clear()
    write(OUT/'SMOKE.json',dict(original_dual_predictions_reproduced=True,three_gate_scores_and_actions_reproduced=True,test_label_poison_predictions_actions_thresholds_unchanged=True))
    print('SMOKE PASS',flush=True)

def run_one(item,d,original_plan,m):
    rep=item['rep'];seed=item['seed'];dest=OUT/'private'/f'round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    start=time.time();dest.mkdir(parents=True,exist_ok=True);c.OUT=dest;c._SOURCE.clear()
    def rows(arm,h):
        original=np.asarray(original_plan['source_rows'].get(arm,[]),int)
        selected=set(sum(item['sources'].values(),[]));return np.array([i for i in original if i in selected],int)
    c.source_rows=rows
    sp=item['targets']['SAHZU'];a=np.asarray(sp['pool']);b=np.asarray(sp['test']);y=d['medical'][b]
    table1=[];table2=[];model_audits=[];preds={};oldsaved={}
    for arm in ['target_only_SA','HZ_to_SA','HZ+HB_to_SA']:
        o,actions,ga,scores=reporting(d,sp,arm,seed)
        preds[arm]=o['prediction'];oldsaved[arm]=o
        for rule,ac in actions.items():table2.append(dict(rep=rep,arm=arm,rule=rule,**metrics(o['prediction'],y[:,2],ac)))
        np.savez_compressed(dest/(arm+'_reporting.npz'),test=b,training=a,prediction=o['prediction'],**{'action_'+k.replace(' ','_'):v for k,v in actions.items()},**{'score_'+k:v for k,v in scores.items()})
        write(dest/(arm+'_gates.json'),ga)
        if arm=='target_only_SA':continue
        r=rows(arm,900);masked=np.full_like(d['medical'],np.nan);masked[r]=d['medical'][r]
        data=dict(d,x=d['x'],medical=masked,ci=np.full_like(d['ci'],np.nan));data['ci'][r]=d['ci'][r]
        models={}
        for method in mapping.METHODS:
            model=m.fit_source(data,r,[],method,seed,steps=500 if method in mapping.METHODS[-2:] else None)
            assert model['success'];models[method]=model
            pt=m.predict(model,d['x'][a]);pq=m.predict(model,d['x'][b])
            hc=m.fit_cal(pt,d['medical'][a],model['ym'],model['ys']);assert hc['finite']
            cp=m.project(m.predict_cal(hc,pq))
            def prior(train,query,h):return m.predict(model,d['x'][train])[:,2],m.predict(model,d['x'][query])[:,2]
            cp[:,2]=c.predict(arm,900,a,b,prior_override={'predict_rows':prior})[0]
            table1.append(dict(rep=rep,arm=arm,panel='mapping',method=method,**four_metrics(cp,y)))
            np.savez_compressed(dest/(arm+'_'+method+'.npz'),test=b,prediction=cp)
        aux=models['RIDGE-10'];src=c.source_model(arm,900)
        pt=m.predict(aux,d['x'][a]);pq=m.predict(aux,d['x'][b]);pt[:,2]=c.source_predict(src,d['x'][a,1:2]);pq[:,2]=c.source_predict(src,d['x'][b,1:2])
        # Current Table 1 uses source-cohort scales for joint calibration controls;
        # hybrid companion uses the original Ridge scaling for R/K/angle.
        scale=dict(ym=d['medical'][r].mean(0),ys=np.maximum(d['medical'][r].std(0),1e-6))
        hc=m.fit_cal(pt,d['medical'][a],scale['ym'],scale['ys']);assert hc['finite']
        hc_aux=m.fit_cal(pt,d['medical'][a],aux['ym'],aux['ys']);assert hc_aux['finite']
        cp=m.project(m.predict_cal(hc_aux,pq));cp[:,2]=o['prediction']
        vals={'No adaptation':pq,'Independent affine':pred_cal(fit_cal(pt,d['medical'][a],scale,'affine'),pq),'Joint linear':pred_cal(fit_cal(pt,d['medical'][a],scale,'joint'),pq),'Joint Huber':m.project(m.predict_cal(hc,pq)),'Asymmetric correction':cp}
        table1.append(dict(rep=rep,arm=arm,panel='mapping',method='Hybrid regression',**four_metrics(cp,y)))
        for kind,p in vals.items():table1.append(dict(rep=rep,arm=arm,panel='calibration',method=kind,**four_metrics(p,y)))
        np.savez_compressed(dest/(arm+'_calibration.npz'),test=b,**{k.replace(' ','_'):p for k,p in vals.items()})
        model_audits.append(dict(arm=arm,source_rows=r,source_hash=hashlib.sha256(r.tobytes()).hexdigest(),development=a,test=b))
    write(dest/'Table1.json',table1);write(dest/'Table2.json',table2);write(dest/'AUDIT.json',model_audits)
    write(dest/'COMPLETE.json',dict(rep=rep,seconds=time.time()-start,seed=seed))
    print('ROUND COMPLETE',rep,round(time.time()-start,1),flush=True)

def aggregate():
    rows={1:[],2:[]}
    for p in sorted((OUT/'private').glob('round_*/COMPLETE.json')):
        for i in rows:rows[i]+=json.loads((p.parent/f'Table{i}.json').read_text(encoding='utf8'))
    summaries={}
    for i,rr in rows.items():
        groups={};keys=['arm','panel','method'] if i==1 else ['arm','rule']
        for r in rr:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        summaries[i]=[]
        for key,v in groups.items():
            record=dict(zip(keys,key));record['repeats']=len(v)
            for name in v[0]:
                if name in keys+['rep','confusion']:continue
                values=[r[name] for r in v if r[name] is not None]
                if not values:record[name]=dict(mean=None,sd=None,valid=0);continue
                record[name]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None,min=float(min(values)),max=float(max(values)),valid=len(values))
            if i==2:record['zero_FN_rounds']=sum(r['FN']==0 and r['abnormal_released']>0 for r in v)
            summaries[i].append(record)
        write(OUT/f'Table{i}_summary.json',summaries[i])
        if rr:
            with (OUT/f'Table{i}_rounds.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
    def fmt(x,percent=False):
        if x['mean'] is None:return 'NA'
        scale=100 if percent else 1;return f"{x['mean']*scale:.2f} ± {x['sd']*scale:.2f}" if x['sd'] is not None else f"{x['mean']*scale:.2f}"
    text=['# 90%分层抽样，100轮重新训练：首批结果','', '原论文和原结果未修改。均值±样本标准差描述本次抽样协议下的波动，不是95%置信区间。测试集每轮181条；开发池每轮54条。各轮测试集重叠，不是新的独立外部队列。', '', '| 流程 | 规则 | Accuracy (%) | Reporting rate (%) | FNR (%) | FPR (%) | FN=0轮次/有效轮次 |','|---|---|---:|---:|---:|---:|---:|']
    for r in summaries[2]:text.append('| '+' | '.join([r['arm'],r['rule'],*[fmt(r[k],True) for k in ['accuracy','reporting_rate','FNR','FPR']],f"{r['zero_FN_rounds']}/{r['FNR']['valid']}"])+' |')
    text+=['','| 流程 | 对照 | 方法 | R MAE | K MAE | MA MAE | Angle MAE |','|---|---|---|---:|---:|---:|---:|']
    for r in summaries[1]:text.append('| '+' | '.join([r['arm'],r['panel'],r['method'],*[fmt(r[k+'_MAE']) for k in ['R','K','MA','angle']]])+' |')
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print('AGGREGATED',len(rows[1]),len(rows[2]),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','run','aggregate']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    if args.stage=='aggregate':aggregate();return
    d,plan,role,m=init();plans=prepare(d,plan,role)
    proto=dict(fraction=.9,repetitions=100,master_seed=202611160,scope='First batch: focal Table1 and Table2 at15min plus target-only reporting control.',sampling='Preserve original source/development/test roles; class quotas by largest remainder, total round-half-up; grouped random subsampling; same draws across methods.',training='Unchanged source models, class weights, asymmetric MA loss, regularization, projection, Huber auxiliary correction and development error threshold .15; nested3fold gate training. All models refitted per round.',summary='Mean and sample SD across rounds; not a confidence interval or independent cohort.',input_hashes={p.name:sha(p) for p in (current.ROOT/'data').glob('*') if p.is_file()},remaining='Nine-direction expansion, 5/10min, budget and representation controls, separate301 internal evaluation require their own follow-up batches; original manuscript not changed.')
    fp=OUT/'PROTOCOL.json'
    if fp.exists():assert json.loads(fp.read_text(encoding='utf8'))==safe(proto)
    else:write(fp,proto)
    if args.stage=='smoke':smoke(d,plan,role,m)
    if args.stage=='run':
        assert (OUT/'SMOKE.json').exists()
        for item in plans[args.start:args.stop]:run_one(item,d,plan,m)
        aggregate()
    assert all(sha(current.ROOT/'data'/p)==h for p,h in proto['input_hashes'].items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
