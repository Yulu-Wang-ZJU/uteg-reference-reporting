"""Current-data replays of the appendix's development-only alternatives."""
import argparse
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,gates,analysis,mapping

OUT=suite.ROOT/'manuscript_sync/experiments/development'
ARMS=[a for a in c.TARGET if not a.startswith('target_only') and a not in c.MAIN]
METHODS=['RIDGE-10','PLS-2','SVR-1']
POLICIES=['BASE','LOW','HIGH','BOTH']

def metric(y,p):
    t=c.accounting.states(y);q=c.accounting.states(p);rec=[float((q[t==v]==v).mean()) if np.any(t==v) else np.nan for v in [-1,0,1]]
    return dict(low_recall=rec[0],high_recall=rec[2],normal_fpr=1-rec[1],tail_recall=float(np.nanmean([rec[0],rec[2]])),
        balanced_accuracy=float(np.nanmean(rec)),agreement=float((t==q).mean()),mae=float(abs(y-p).mean()))

def choose(pred,y,base,fpr=.1,tail=0.,last=False):
    m={k:metric(y,v) for k,v in pred.items()};b=m[base]
    good=[k for k,v in m.items() if v['agreement']>=b['agreement']-.02-1e-12 and v['normal_fpr']<=b['normal_fpr']+fpr+1e-12 and v['mae']<=b['mae']+1+1e-12 and all(not np.isfinite(b[z]) or v[z]>=b[z]-tail-1e-12 for z in ['low_recall','high_recall'])]
    order=list(pred);best=max(good,key=lambda k:(round(m[k]['balanced_accuracy'],12),(1 if last else -1)*order.index(k)))
    return best,m

def side(raw,final,name):
    p=final.copy();mask=(final>=50)&(final<=70)&(((raw<50) if name in ['LOW','BOTH'] else False)|((raw>70) if name in ['HIGH','BOTH'] else False))
    p[mask]=(raw[mask]+final[mask])/2;return p

def prior(arm,method):
    m=mapping.env();src=mapping.source(arm,method,1000)
    def f(train,query,h):
        x=c.feature(train,query,h,'four');return m.predict(src,x[train])[:,2],m.predict(src,x[query])[:,2]
    return dict(predict_rows=f)

def calculate(arm,seed):
    d,*_=c.context();sp=c.paper_spec(arm,seed);pool=np.asarray(sp['pool']);where={int(v):i for i,v in enumerate(pool)}
    result={k:np.full(len(pool),np.nan) for k in ['base','raw','strength0','strength05','strength1','strength_selected',*METHODS,'mapper_selected',*POLICIES,'side_selected']}
    choices=[];gateparts=[]
    src=c.source_model(arm,900);rawall=c.source_predict(src,c.feature([],[],900,'B'))
    for pos,f in enumerate(sp['positions'][:3]):
        out=c.evaluate(f['A'],f['B'],f['inner'],arm,900);a=out['train'];b=out['query'];ix=[where[int(v)] for v in b]
        p=out['prediction'];r=rawall[b];ri=rawall[a];pi=out['inner'];y=d['y'][a,2]
        strength={v:ri+v*(pi-ri) for v in [0.,.5,1.]};pick,sm=choose(strength,y,1.,.05,.05,True)
        result['base'][ix]=p;result['raw'][ix]=r
        for v,k in [(0.,'strength0'),(.5,'strength05'),(1.,'strength1')]:result[k][ix]=r+v*(p-r)
        result['strength_selected'][ix]=r+pick*(p-r)
        si={v:side(ri,pi,v) for v in POLICIES};ps,ms=choose(si,y,'BASE')
        for v in POLICIES:result[v][ix]=side(r,p,v)
        result['side_selected'][ix]=side(r,p,ps)
        mi={};mo={}
        for method in METHODS:
            q=c.evaluate(f['A'],f['B'],f['inner'],arm,900,prior_override=prior(arm,method));mi[method]=q['inner'];mo[method]=q['prediction'];result[method][ix]=q['prediction']
        pm,mm=choose(mi,y,'RIDGE-10');result['mapper_selected'][ix]=mo[pm]
        choices.append(dict(position=pos,strength=pick,side=ps,mapper=pm,strength_metrics=sm,side_metrics=ms,mapper_metrics=mm))
        if arm=='HB_to_SA':
            new=dict(out,prediction=side(r,p,ps),inner=si[ps]);ss,_=gates.fit_score(new,900,seed,arm,pos)
            gateparts.append(dict(rows=b,prediction=new['prediction'],**ss))
    assert all(np.isfinite(v).all() for v in result.values())
    return dict(**result,pool=pool),dict(seed=seed,choices=choices,gateparts=gateparts)

def run():
    suite.initialize();suite.numerical_models();OUT.mkdir(parents=True,exist_ok=True)
    gates.clean_write(OUT/'PROTOCOL.json',dict(arms=ARMS,strengths=[0,.5,1],side_policies=POLICIES,mapping_controls=METHODS,
        selection='Each outer development block selects from its inner predictions only; no held-out target-test labels.',
        changed_baseline='Current B-only inward4 source/local branch; mapper comparison holds local inward4 correction fixed.',
        final_candidate_promotion=False,B_from_original_data=True))
    for arm in ARMS:
        for seed in c.SEEDS:
            path=OUT/'private'/f'{arm}_{seed}.npz';path.parent.mkdir(exist_ok=True)
            if path.exists():continue
            v,au=calculate(arm,seed);np.savez_compressed(path,**v);gates.clean_write(path.with_suffix('.json'),au)
        print('DEVELOPMENT_REPLAY_COMPLETE',arm,flush=True)
    summary=[];seedrows=[];d,*_=c.context()
    for arm in ARMS:
        zs=[analysis.load(OUT/'private'/f'{arm}_{s}.npz') for s in c.SEEDS];ys=[d['y'][z['pool'],2] for z in zs]
        for k in zs[0]:
            if k=='pool':continue
            summary.append(dict(arm=arm,candidate=k,**metric(np.concatenate(ys),np.concatenate([z[k] for z in zs]))))
            for s,z,y in zip(c.SEEDS,zs,ys):seedrows.append(dict(arm=arm,seed=s,candidate=k,**metric(y,z[k])))
    gates.clean_write(OUT/'SUMMARY.json',summary);gates.clean_write(OUT/'SEEDS.json',seedrows)
    gates.clean_write(OUT/'COMPLETE.json',dict(directions=7,seeds=20,development_only=True,candidates=len(zs[0])-1))

def headroom():
    suite.initialize();OUT.mkdir(parents=True,exist_ok=True);result=[]
    for arm in ARMS:
        dev=c.read(suite.ROOT/'gates'/f'{arm}_T900.development.json')
        for item in dev:
            rows=item['rows'];p=np.array([r['pred_class'] for r in rows]);t=np.array([r['truth_class'] for r in rows]);b=np.array([r['base_score'] if r['base_score'] is not None else np.inf for r in rows]);du=np.array([r['dual_score'] if r['dual_score'] is not None else np.inf for r in rows])
            th=gates.fr.choose(rows);action=gates.fr.action(th,p,b,du,np.ones(len(p),bool));fixed=action&(p==0);tail=(p!=0)&np.isfinite(b)
            gc=int((tail&(p==t)).sum());bc=int((tail&(p!=t)).sum());fn=int(fixed.sum());fc=int((fixed&(p==t)).sum());oracle=(0,0)
            for wrong in range(bc+1):
                n=fn+gc+wrong;good=fc+gc
                if n>=5 and good/n>.75:oracle=(n,good)
            result.append(dict(arm=arm,seed=item['seed'],events=len(p),reported=int(action.sum()),correct=int((action&(p==t)).sum()),
                within_reports=fn,correct_abnormal_available=gc,correct_abnormal_reported=int((action&tail&(p==t)).sum()),oracle_reported=oracle[0],oracle_correct=oracle[1],true_low=int((t==-1).sum()),true_high=int((t==1).sum())))
    gates.clean_write(OUT/'HEADROOM.json',result)

def preserved():
    suite.initialize();d,*_=c.context();arm='HB_to_SA';devbase=c.read(suite.ROOT/'gates'/f'{arm}_T900.development.json');testbase=analysis.load(suite.ROOT/'gates'/f'{arm}_T900.npz')
    results=[];devresults=[];actions={k:[] for k in ['BASE','REMATCHED','PRESERVED']};preds=[];audits=[]
    for si,seed in enumerate(c.SEEDS):
        pth=OUT/'private'/f'{arm}_{seed}.npz';z=analysis.load(pth);au=c.read(pth.with_suffix('.json'));byid={str(d['ids'][i]):j for j,i in enumerate(z['pool'])}
        sc={str(d['ids'][i]):{k:part[k][j] for k in ['LR','DUAL']} for part in au['gateparts'] for j,i in enumerate(part['rows'])}
        old=devbase[si]['rows'];new=[];keep=[];oldth=gates.fr.choose(old)
        for r in old:
            rid=r['record_id'];pc=int(c.accounting.states([z['side_selected'][byid[rid]]])[0]);bs=sc[rid]['LR'];ds=sc[rid]['DUAL']
            new.append(dict(record_id=rid,pred_class=pc,truth_class=r['truth_class'],base_score=bs,dual_score=ds))
            keep.append(dict(new[-1],dual_score=r['dual_score'] if r['pred_class']==0 and pc==0 else None))
        rematch=gates.fr.choose(new);tau=oldth['normal'];ids=[r['record_id'] for r in keep if tau is not None and r['dual_score'] is not None and r['dual_score']<=tau]
        candidate=dict(threshold=tau,candidate_index=0,n=len(keep),release_record_ids=ids)
        norm=gates.fr.sel.choose_normal([candidate],keep,{r['record_id']:r for r in keep});tail=gates.fr.sel.choose_common_tail(keep,norm)
        preserved=dict(normal=norm['selected']['threshold'] if norm['selected'] else None,tail=tail['selected']['threshold'] if tail['selected'] else None,all_withhold=tail['selected'] is None)
        for name,rows,th in [('BASE',old,oldth),('REMATCHED',new,rematch),('PRESERVED',keep,preserved)]:
            p=np.array([r['pred_class'] for r in rows]);t=np.array([r['truth_class'] for r in rows]);bs=np.array([r['base_score'] if r['base_score'] is not None else np.inf for r in rows]);ds=np.array([r['dual_score'] if r['dual_score'] is not None else np.inf for r in rows]);ac=gates.fr.action(th,p,bs,ds,np.ones(len(p),bool))
            devresults.append(dict(arm=name,seed=seed,**c.accounting.classify(p,t,ac)))
        sp=c.paper_spec(arm,seed);out=c.evaluate(sp['pool'],sp['test'],sp['positions'][3]['inner'],arm,900);src=c.source_model(arm,900);x=c.feature([],[],900,'B');raw=c.source_predict(src,x)
        inside={v:side(raw[out['train']],out['inner'],v) for v in POLICIES};pick,_=choose(inside,d['y'][out['train'],2],'BASE')
        newout=dict(out,inner=inside[pick],prediction=side(raw[out['query']],out['prediction'],pick));ss,_=gates.fit_score(newout,900,seed,arm,3);pc=c.accounting.states(newout['prediction']);oldpc=c.accounting.states(testbase['prediction'][si]);dual=np.where((oldpc==0)&(pc==0),testbase['DUAL_score'][si],np.inf)
        actions['BASE'].append(testbase['TWO_SIDED_action'][si]);actions['REMATCHED'].append(gates.fr.action(rematch,pc,ss['LR'],ss['DUAL'],np.isfinite(newout['prediction'])))
        actions['PRESERVED'].append(gates.fr.action(preserved,pc,ss['LR'],dual,np.isfinite(newout['prediction'])));preds.append(newout['prediction']);audits.append(dict(seed=seed,policy=pick,rematched=rematch,preserved=preserved))
    np.savez_compressed(OUT/'PRESERVED_TEST.npz',prediction=preds,base_prediction=testbase['prediction'],test=out['query'],**actions)
    for name,act in actions.items():results.append(dict(arm=name,**c.summarize(testbase['prediction'] if name=='BASE' else np.array(preds),d['y'][out['query'],2],act)))
    gates.clean_write(OUT/'PRESERVED_TEST.json',results);gates.clean_write(OUT/'PRESERVED_DEV.json',devresults);gates.clean_write(OUT/'PRESERVED_AUDIT.json',audits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','headroom','preserved']);a=ap.parse_args()
    with threadpool_limits(limits=1):globals()[a.stage]()
