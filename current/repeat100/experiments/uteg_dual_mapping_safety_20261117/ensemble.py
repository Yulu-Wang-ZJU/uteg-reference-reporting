"""Combine development-safe report candidates; never query evaluation outcomes."""
import sys,json,importlib.util,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
O=Path(__file__).parent
spec=importlib.util.spec_from_file_location('dual_mapping',O/'run.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
e=m.e;ag=m.ag
NAMES=['original']+m.MODELS

def ensemble(pred,act,ranking):
    pred=np.asarray(pred);act=np.asarray(act);cls=e.r.state(pred);n=pred.shape[1]
    output=pred[0].copy();release=np.zeros(n,bool);selected=np.full(n,-1,int)
    normal=(act&(cls==1)).any(0);low=(act&(cls==0)).any(0);high=(act&(cls==2)).any(0)
    wanted=np.where(normal,1,np.where(low&~high,0,np.where(high&~low,2,-1)))
    for j in ranking:
        take=(selected<0)&act[j]&(cls[j]==wanted)&(wanted>=0)
        output[take]=pred[j,take];selected[take]=j;release[take]=True
    assert np.array_equal(e.r.state(output[release]),wanted[release])
    return output,release,selected

def run_case(d,base,mapping_files):
    with np.load(base) as z:old={k:z[k] for k in z.files}
    tr=old['dev_rows'];te=old['test'];dy=d['medical'][tr,2];dp=[old['dev_prediction']];pr=[old['prediction']]
    for p in mapping_files:
        with np.load(p) as z:
            np.testing.assert_array_equal(tr,z['dev_rows']);np.testing.assert_array_equal(te,z['test'])
            dp.append(z['development_prediction']);pr.append(z['prediction'])
    dp=np.array(dp);pr=np.array(pr);ranking=np.argsort(np.mean(abs(dp-dy),axis=1),kind='stable');out={};policies={};devs={}
    for ab in ['all','accuracy85']:
        da=[];qa=[];pp=[]
        for i in range(len(NAMES)):
            p=m.choose(dp[i],dy,old['dev_risk'],old['dev_prediction'],'all_max',ab)
            da.append(m.action(p,dp[i],old['dev_risk']));qa.append(m.action(p,pr[i],old['test_risk']));pp.append(p)
        devpred,devact,ds=ensemble(dp,da,ranking);pred,act,selected=ensemble(pr,qa,ranking)
        dm=e.n.extended_metrics(devpred,dy,devact);assert dm['FN']==0
        out[ab]=(pred,act,selected);devs[ab]=dm;policies[ab]=dict(component_rules=pp,development=dm)
    feasible=[k for k in out if devs[k]['FN']==0 and (devs[k]['accuracy'] or 0)>.8 and devs[k]['reported_as_normal']>0 and devs[k]['abnormal_released']>0]
    chosen=max(feasible,key=lambda k:(devs[k]['reported'],devs[k]['accuracy'],k=='accuracy85')) if feasible else None
    if chosen is None:out['development_selected']=(old['prediction'].copy(),np.zeros(len(te),bool),np.full(len(te),-1,int))
    else:out['development_selected']=out[chosen]
    policies['selection']=dict(chosen=chosen,feasible=bool(chosen),ranking=[NAMES[i] for i in ranking],developer_MAE=np.mean(abs(dp-dy),axis=1).tolist())
    return old,out,policies

def main():
    root=O/'ensemble';root.mkdir(exist_ok=True)
    protocol=dict(components=NAMES,
      normal='Each component must pass its own all-development-maximum OOF residual envelope and the unchanged raw10%risk veto. Any safe normal candidate is eligible.',
      abnormal='When no normal candidate is safe, release low or high only when eligible abnormal candidates do not conflict in direction. Two fixed controls: all abnormal predictions, or developer85%classaccuracy margins.',
      selection='Prefer eligible normal to abnormal. Among eligible same-state predictions, use lowest developerOOFMAE, with deterministic ties. Also compare developer-only choice between the two abnormal controls: FN0,accuracy>.8,nonempty normal/abnormal releases,then maximum reports. Infeasible retained,not treated as success.',
      chronology='Exploratory combination proposed after individual mapping controls were evaluated. No per-evaluation-record truth used to select a model or threshold; no new independent-validation claim.',repeats=100,paper_unchanged=True)
    e.r.write(root/'PROTOCOL.json',protocol);d,*_=e.init();rows=[]
    for rep in range(100):
        folder=O/f'full_v2/round_{rep:02d}';cases=[];pols=[]
        for b in [54,60]:cases.append(('SA',str(b),'HZ+HB_to_SA',e.O/f'primary/dev{b}/time900/round_{rep:02d}/HZ+HB_to_SA.npz',[folder/f'SA_{b}_{k}_mapping.npz' for k in m.MODELS]))
        for setting in ['sampled90','full_local']:
            for stage,arm in [('Huzhou','SA+HB_to_HZ'),('Hubei','SA+HZ_to_HB')]:cases.append((stage,setting,arm,e.O/f'nine/round_{rep:02d}/{setting}_{arm}.npz',[folder/f'{stage}_{setting}_{k}_mapping.npz' for k in m.MODELS]))
        for stage,setting,arm,base,maps in cases:
            old,out,pol=run_case(d,base,maps)
            for method,(pred,act,selected) in out.items():rows.append(dict(stage=stage,setting=setting,rep=rep,arm=arm,method=method,test=e.n.extended_metrics(pred,d['medical'][old['test'],2],act)))
            pols.append(dict(stage=stage,setting=setting,policies=pol))
            np.savez_compressed(root/f'{stage}_{setting}_{rep}.npz',test=old['test'],**{k+'_'+s:v[i] for k,v in out.items() for i,s in enumerate(['prediction','action','model'])})
        te=[];parts={}
        for fi in range(5):
            old,out,pol=run_case(d,e.O/f'internal301/round_{rep:02d}/fold{fi}_HZ+HB_to_SA.npz',[folder/f'internal_fold{fi}_{k}_mapping.npz' for k in m.MODELS]);te.extend(old['test']);pols.append(dict(stage='internal301',fold=fi,policies=pol))
            for k,v in out.items():
                if k not in parts:parts[k]=[[],[],[]]
                for i in range(3):parts[k][i].extend(v[i])
        te=np.array(te);assert len(set(te))==271
        for method,(pred,act,selected) in parts.items():rows.append(dict(stage='internal301',setting='271',rep=rep,arm='HZ+HB_to_SA',method=method,test=e.n.extended_metrics(np.array(pred),d['medical'][te,2],np.array(act))))
        np.savez_compressed(root/f'internal301_{rep}.npz',test=te,**{k+'_'+s:np.array(v[i]) for k,v in parts.items() for i,s in enumerate(['prediction','action','model'])})
        e.r.write(root/f'POLICIES_{rep:02d}.json',pols)
    summary=ag.summarize(rows,['stage','setting','arm','method'])
    for s in summary:
        rr=[x for x in rows if all(x[k]==s[k] for k in ['stage','setting','arm','method'])]
        s['positive_FN_rounds']=sum(x['test']['FN']>0 for x in rr);s['FN_total']=sum(x['test']['FN'] for x in rr)
        s['joint_pass_rounds']=sum(x['test']['FN']==0 and (x['test']['accuracy'] or 0)>.8 and x['test']['reporting_rate']>.75 and x['test']['reported_as_normal']>0 and x['test']['abnormal_released']>0 for x in rr)
        s['no_normal_report_rounds']=sum(x['test']['reported_as_normal']==0 for x in rr)
    e.r.write(root/'ROUNDS.json',rows);ag.csvwrite(root/'ROUNDS.csv',rows);e.r.write(root/'SUMMARY.json',summary);ag.csvwrite(root/'SUMMARY.csv',summary)
    print('DONE',len(rows))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
