"""B085 replay of calibration controls and local-label budgets.

The module writes only below outputs/paper_B085_hospital_replay_20261102/adaptation_controls.
All numerical outputs carry the same original U-TEG B values as the revised
DATA15 copy.  Historical source/model artifacts are read-only.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, sys, time
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'outputs/paper_B085_hospital_replay_20261102/adaptation_controls'
DATA=WORKSPACE/'outputs/uteg_B085_by_hospital_20261102/private/DATA15_B085_BY_HOSPITAL.npz'
TRANSFER=WORKSPACE/'outputs/paper_B085_hospital_replay_20261102/transfer'
HISTORY=WORKSPACE/'outputs/uteg_supplement_completion_20261027_203140'
SEEDS=list(range(1000,1020)); DIRS=['HZ_to_SA','HZ+HB_to_SA','SA+HB_to_HZ','SA+HZ_to_HB','HB_to_SA','SA_to_HZ','HB_to_HZ','SA_to_HB','HZ_to_HB']
TARGET={'HZ_to_SA':'SAHZU','HB_to_SA':'SAHZU','HZ+HB_to_SA':'SAHZU','SA_to_HZ':'Huzhou','HB_to_HZ':'Huzhou','SA+HB_to_HZ':'Huzhou','SA_to_HB':'Hubei','HZ_to_HB':'Hubei','SA+HZ_to_HB':'Hubei'}
ARMS=['raw','affine','joint','complete']

def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for q in iter(lambda:f.read(1<<20),b''):h.update(q)
    return h.hexdigest()
def write(name,x):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')
def cls(x): return np.where(x<50,-1,np.where(x>70,1,0))
def project(x):
    y=np.asarray(x,float).copy();y[...,:3]=np.maximum(y[...,:3],0);y[...,3]=np.clip(y[...,3],0,89.999999);return y
def fit_cal(p,y,src,arm):
    scale=np.maximum(src['ys'],1e-6);v=(project(p)-src['ym'])/scale;u=(y-src['ym'])/scale;vm=v.mean(0);um=u.mean(0);vc=v-vm;uc=u-um
    if arm=='affine': M=np.diag([np.linalg.lstsq(vc[:,j,None],uc[:,j],rcond=None)[0][0] for j in range(4)])
    else: M=np.linalg.solve(vc.T@vc+10*np.eye(4),vc.T@uc+10*np.eye(4))
    return dict(matrix=M,vbar=vm,ubar=um,mu=src['ym'],scale=scale)
def pred_cal(m,p):return project(m['mu']+m['scale']*(m['ubar']+((project(p)-m['mu'])/m['scale']-m['vbar'])@m['matrix']))

def load_batch():
    spec=importlib.util.spec_from_file_location('b085_controls_batch',HISTORY/'batch.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.ROOT=ROOT/'budget_runs'
    def data_at(h):
        assert h==900
        with np.load(DATA,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    b.data_at=data_at;return b
def full_obj_path(direction,seed):return TRANSFER/'runs'/('prefix_900__'+direction)/'models'/direction/'target15'/f'{TARGET[direction]}_CFULL_seed{seed}.joblib'
def source_wrapper(r,d,s,obj):
    p=r.ROOT/'source'/f'{d}_{s}.joblib';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():joblib.dump({'families':{'RIDGE':{'model':obj['source_model']}},'source_selection':'RIDGE'},p,compress=3)
    return p

def fourway():
    out=ROOT/'calibration_controls';out.mkdir(parents=True,exist_ok=True);D=dict(np.load(DATA,allow_pickle=False));rows=[];seedrows=[];deltas=[];allpred={};bb=load_batch();proto=dict(version=1,input=str(DATA),input_sha256=sha(DATA),B_from_original_data=True,directions=['HZ_to_SA','HZ+HB_to_SA'],seeds=SEEDS,arms=ARMS,old_default_untouched=True,bootstrap_replicates=5000,bootstrap_seed=20261030,calibration_records=60,test_labels_used_only_after_fit=True)
    for d in ['HZ_to_SA','HZ+HB_to_SA']:
        test=None;preds=[];Y=None
        for s in SEEDS:
            obj=joblib.load(full_obj_path(d,s));A=np.asarray(obj['positions'][3]['A']);idx=np.asarray(obj['test']);src=obj['source_model'];p=project(src['ym']+src['ys']*(((D['x'][A]-src['xm'])/src['xs'])@src['coef']+src['intercept']));y=D['medical'][A];assert np.isfinite(p).all() and np.isfinite(y).all();cal={a:fit_cal(p,y,src,a) for a in ['affine','joint']};raw=project(src['ym']+src['ys']*(((D['x'][idx]-src['xm'])/src['xs'])@src['coef']+src['intercept']))
            cache=obj['cache'][obj['positions'][3]['base']];h=joblib.loads if False else None
            # Reconstruct the complete adapter from the frozen target object.
            # Use an isolated batch environment only to load the legacy numerical module.
            r,runner,target,*_=bb.env('controls_'+d+'_'+str(s),16000);target._load_data=lambda data_path=None,mode='T900':dict(D);target._load_role=lambda: bb.read(bb.BASE/'ROLE_FREEZE.json')
            hub=target.m.calibrated(cache['cal'],raw);ad=obj['final_selection']['adapter'];complete=hub if ad=='H' else target.m.ans['adapt'](cache['adapters'][ad],D['x'][idx],raw,hub,src)
            arr=np.array([raw,pred_cal(cal['affine'],raw),pred_cal(cal['joint'],raw),complete]);preds.append(arr);Y=D['medical'][idx];test=idx
        P=np.asarray(preds);truth=cls(Y[:,2]);pc=cls(P[:,:,:,2]);ug=np.unique(D['groups'][test]);rng=np.random.default_rng(20261030);ix=rng.integers(0,len(ug),(5000,len(ug)))
        def stat(num,den):
            # Keep all 20 seeds together in each identity-group bootstrap draw.
            # The previous implementation took quantiles over 20*5000
            # seed-level ratios, which mixed independent seed summaries rather
            # than bootstrapping the pooled 20-seed event total.
            n=np.asarray(num,float);d0=np.broadcast_to(den,n.shape).astype(float)
            aa=np.array([n[:,D['groups'][test]==v].sum(1) for v in ug]).T
            bb=np.array([d0[:,D['groups'][test]==v].sum(1) for v in ug]).T
            sampled_num=aa[:,ix].sum(2).sum(0)
            sampled_den=bb[:,ix].sum(2).sum(0)
            ok=sampled_den>0
            point=float(aa.sum()/bb.sum()) if bb.sum() else None
            ci=np.quantile((sampled_num[ok]/sampled_den[ok]),[.025,.975]).tolist() if ok.any() else [None,None]
            return point,ci[0],ci[1]
        for ai,arm in enumerate(ARMS):
            E=abs(P[:,ai]-Y);agree=(pc[:,ai]==truth);ab=truth!=0;norm=truth==0
            sm=stat(E[:,:,2],np.ones(agree.shape));sa=stat(agree,np.ones(agree.shape));sl=stat((truth[None,:]==-1)&(pc[:,ai]==0),truth[None,:]==-1);sh=stat((truth[None,:]==1)&(pc[:,ai]==0),truth[None,:]==1);sn=stat((truth[None,:]==0)&(pc[:,ai]!=0),truth[None,:]==0)
            row=dict(direction=d,arm=arm,seeds=20,records=len(test),events=int(agree.size),MA_MAE=sm[0],MA_MAE_lo=sm[1],MA_MAE_hi=sm[2],agreement=sa[0],agreement_lo=sa[1],agreement_hi=sa[2],low_to_within=sl[0],low_to_within_lo=sl[1],low_to_within_hi=sl[2],high_to_within=sh[0],high_to_within_lo=sh[1],high_to_within_hi=sh[2],normal_to_abnormal=sn[0],normal_to_abnormal_lo=sn[1],normal_to_abnormal_hi=sn[2],low_correct=int(np.sum((truth[None,:]==-1)&(pc[:,ai]==-1))),high_correct=int(np.sum((truth[None,:]==1)&(pc[:,ai]==1))))
            rows.append(row)
            for k,s in enumerate(SEEDS):seedrows.append(dict(direction=d,arm=arm,seed=s,MA_MAE=float(E[k,:,2].mean()),agreement=float(agree[k].mean())))
        for ai,arm in enumerate(ARMS[1:],1):
            for j,param in enumerate(('R','K','MA','alpha')):
                raw_error=np.abs(P[:,0,:,j]-Y[None,:,j])
                adapted_error=np.abs(P[:,ai,:,j]-Y[None,:,j])
                gain=stat(raw_error-adapted_error,np.ones_like(raw_error))
                deltas.append(dict(direction=d,contrast='raw minus '+arm,parameter=param,estimate=gain[0],lo=gain[1],hi=gain[2],bootstrap_groups=int(len(ug)),bootstrap_reps=5000))
        allpred[d]=P;np.savez_compressed(out/(d+'_predictions.npz'),predictions=P,arms=ARMS,seed=SEEDS,index=test,ids=D['ids'][test],groups=D['groups'][test],reference=Y,B_from_original_data=np.asarray(True))
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(seedrows[0]));w.writeheader();w.writerows(seedrows)
    with (out/'paired_mae_gain.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
    write(Path('calibration_controls')/'PROTOCOL.json',proto);write(Path('calibration_controls')/'RESULTS.json',dict(rows=rows,paired_mae_gain=deltas,B_from_original_data=True,bootstrap_seed=20261030,bootstrap_replicates=5000))
    return rows

def fit_budget_task(b,task_id,d,budget):
    r,runner,target,*_=b.env(task_id,16000);D=b.data_at(900);role=b.read(b.BASE/'ROLE_FREEZE.json');out=[]
    if not (r.ROOT/'RUNNER_FREEZE.json').exists():
        r.write('RUNNER_FREEZE.json',{'runner_sha':b.sha(HISTORY/'batch.py'),'protocol_sha':b.sha(HISTORY/'batch.py')})
    if not (r.ROOT/'ROUND_START_LEDGER.json').exists():
        r.write('ROUND_START_LEDGER.json',{'input_sha256':b.sha(DATA),'task':task_id,'test_labels_used_for_fit':False})
    # For C20, b.env installs the frozen C20 specification.  For C40, use the
    # published pool/position restriction from the minimal-label-budget rules.
    rb_spec=importlib.util.spec_from_file_location('budget_rules',WORKSPACE/'outputs/uteg_minimal_label_budget_20261030/run_budget_curve.py');rb=importlib.util.module_from_spec(rb_spec);rb_spec.loader.exec_module(rb);rb.DATA_PATH=DATA
    for s in SEEDS:
        full=joblib.load(full_obj_path(d,s));srcpath=source_wrapper(r,d,s,full);rows=full['source_rows']
        if budget==40:
            pools=rb.get_budget_pools(D,d,s);pool=pools['pool40'];positions=rb.restricted_positions(full,pool);test=pools['test']
            target._spec=lambda role0,target0,seed0,budget0,positions=positions,pool=pool,test=test:{'target':target0,'arm':'B40','seed':int(seed0),'d':2 if target0=='SAHZU' else 1,'pool':list(pool),'test':list(test),'positions':positions}
        # budget20 is installed by b.env's task-name patch.
        protocol_sha=b.sha(ROOT/'PROTOCOL.md') if (ROOT/'PROTOCOL.md').exists() else b.sha(HISTORY/'batch.py')
        objpath=runner.fit_target_one(target,d,s,srcpath,role,D,rows,protocol_sha);obj=joblib.load(objpath);out.append(objpath)
    return out

def budget_run(budget,dirs):
    b=load_batch();base=ROOT/'budget_runs';base.mkdir(parents=True,exist_ok=True);(ROOT/'budget_results').mkdir(parents=True,exist_ok=True);allrows=[]
    for d in dirs:
        tid=f'budget_{budget}__{d}';fit_budget_task(b,tid,d,budget)
        r,runner,target,*_=b.env(tid,16000);D=b.data_at(900);role=b.read(b.BASE/'ROLE_FREEZE.json');pred=[];truth=None
        for s in SEEDS:
            # Locate output generated by runner; C20 and C40 have different arm labels but same target path.
            paths=list((r.ROOT/'models'/d/'target15').glob(f'{TARGET[d]}_*_seed{s}.joblib'));assert paths
            obj=joblib.load(paths[0]);idx=np.asarray(obj['test']);src=obj['source_model'];raw=project(src['ym']+src['ys']*(((D['x'][idx]-src['xm'])/src['xs'])@src['coef']+src['intercept']));cache=obj['cache'][obj['positions'][3]['base']];hub=target.m.calibrated(cache['cal'],raw);ad=obj['final_selection']['adapter'];fin=hub if ad=='H' else target.m.ans['adapt'](cache['adapters'][ad],D['x'][idx],raw,hub,src);pred.append(fin);truth=D['medical'][idx];
        P=np.asarray(pred);Y=truth;tc=cls(Y[:,2]);pc=cls(P[:,:,2]);row=dict(direction=d,budget=budget,seeds=20,records=len(idx),MA_MAE=float(np.mean(abs(P[:,:,2]-Y[None,:,2]))),agreement=float(np.mean(pc==tc[None,:])),low_recall=float(np.mean(pc[:,tc==-1]==-1)) if np.any(tc==-1) else None,high_recall=float(np.mean(pc[:,tc==1]==1)) if np.any(tc==1) else None,B_from_original_data=True);allrows.append(row);np.savez_compressed(ROOT/'budget_results'/f'B{budget}_{d}.npz',prediction=P,reference=Y,index=idx,seed=SEEDS)
    with (ROOT/'budget_results'/f'summary_B{budget}.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    return allrows

def manifest():
    """Write a compact index after calibration/budget runs have finished."""
    paths=[]
    for p in sorted((ROOT/'calibration_controls').glob('*')):
        if p.is_file(): paths.append(dict(path=str(p),sha256=sha(p)))
    for p in sorted((ROOT/'budget_results').glob('*')):
        if p.is_file(): paths.append(dict(path=str(p),sha256=sha(p)))
    write(Path('MANIFEST.json'),dict(version=1,input=str(DATA),input_sha256=sha(DATA),B_from_original_data=True,calibration='two SA directions, 4 arms, 20 seeds, 5000 identity-group bootstrap',budget20='nine directions, 20 seeds',budget40='HZ_to_SA and HZ+HB_to_SA, 20 seeds',budget60='reuse transfer T900 full objects; no new fit',files=paths))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['calibration','budget20','budget40','all','manifest']);a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if a.command in ('calibration','all'):fourway()
    if a.command in ('budget20','all'):budget_run(20,DIRS)
    if a.command in ('budget40','all'):budget_run(40,['HZ_to_SA','HZ+HB_to_SA'])
    if a.command=='manifest':manifest()
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
