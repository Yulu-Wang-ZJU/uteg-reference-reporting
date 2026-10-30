"""Bounded appendix-only completion. Every historical dependency is read-only."""
from pathlib import Path
import os, sys, json, hashlib, types, time, argparse, ast, subprocess
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np
import joblib
OLD = OUT/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
BASE = OUT/'uteg_legacy_aligned_threecenter_20261022_155614'
SEQ = OUT/'uteg_current_ma_sequence_20261026'
SEEDS = list(range(1000,1020))
TARGET = {'HZ_to_SA':'SAHZU','HB_to_SA':'SAHZU','HZ+HB_to_SA':'SAHZU',
          'SA_to_HZ':'Huzhou','HB_to_HZ':'Huzhou','SA+HB_to_HZ':'Huzhou',
          'SA_to_HB':'Hubei','HZ_to_HB':'Hubei','SA+HZ_to_HB':'Hubei'}
MISSING_PREFIX = ['HB_to_SA','SA_to_HZ','HB_to_HZ','SA_to_HB','HZ_to_HB']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);assert p.resolve().is_relative_to(ROOT)
    p.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,default=clean,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def data_at(h):
    with np.load(BASE/'DATA15.npz',allow_pickle=False) as z:D={k:z[k].copy() for k in z.files}
    if h!=900:
        with np.load(BASE/'PREFIX_X.npz') as p:D['x']=p['x'][list(p['horizon_seconds']).index(h)].copy()
        D['input_qc']=np.isfinite(D['x']).all(1)
    return D
def old_path(d,s):return OLD/'models'/d/'target15'/f'{TARGET[d]}_CFULL_seed{s}.joblib'
def module_source(name,path,source):
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    exec(compile(source,str(path),'exec'),mod.__dict__);return mod

def env(task,cap):
    rd=ROOT/'runs'/task;rd.mkdir(parents=True,exist_ok=True)
    text=(OLD/'runtime.py').read_text(encoding='utf-8').replace('OUT.parent /',f'Path({str(OUT)!r}) /')
    r=module_source('runtime',rd/'runtime.py',text)
    r.TOTAL_FIT_CAP=cap;r.SCIENCE_FIT_CAP=cap;r.REPAIR_FIT_CAP=cap
    text=(OLD/'run_ridge10_fullchain.py').read_text(encoding='utf-8')
    # A pregate mapping/sensitivity evaluation has no reason to fit an endpoint gate.
    if not task.startswith('prefix_'):
        start=text.index('            r.ledger(f"{key}_MA_GATE_{code}"')
        end=text.index('            outer_b = _outer_b_metadata',start)
        text=text[:start]+'''            gate = None
            med_B = _position_outer_prediction(p, adapter) if B else np.empty((0,4))
            pred_ma_B = med_B[:,2]
            pred_B = ma_class(pred_ma_B)
            truth_B = ma_class(D["medical"][B,2])
            score_B = np.full(len(B), np.nan)
'''+text[end:]
        text=text.replace('"source_family": "RIDGE"','"source_family": src.get("name", "RIDGE-10")')
    runner=module_source('appendix_runner',rd/'derived_runner.py',text)
    ts=(BASE/'train_target_legacy.py').read_text(encoding='utf-8')
    old='OLD_MODELS = r.ROOT.parent / "uteg_hill_exp_matched_20261012_153337" / "models.py"'
    assert old in ts
    ts=ts.replace(old,f'OLD_MODELS = Path({str(OUT/"uteg_hill_exp_matched_20261012_153337/models.py")!r})')
    target=module_source('appendix_target',rd/'derived_target.py',ts)
    if task.startswith('budget_'):
        original_spec=target._spec
        target._spec=lambda role,target_name,seed,budget:original_spec(role,target_name,seed,'20')
    seq=module_source('appendix_sequence',SEQ/'run.py',(SEQ/'run.py').read_text(encoding='utf-8'))
    contract=r.load('appendix_dual_contract',OUT/'uteg_ridge_dual_head_prepare_20261026_172952/dual_head_contract.py')
    selector=r.load('appendix_dual_selector',OUT/'uteg_ridge_dual_head_unified_selector_20261026_190142/run_unified_selector.py')
    return r,runner,target,seq,contract,selector

def make_plan():
    assert not (ROOT/'PLAN.json').exists(), 'Frozen plan already exists'
    r,runner,t,*_=env('preflight',1)
    role=read(BASE/'ROLE_FREEZE.json'); tasks=[];sources={}
    for d in TARGET:
        obj=joblib.load(old_path(d,1000));sources[d]=obj['source_rows']
        assert not set(obj['source_rows'])&set(obj['pool']+obj['test'])
        for s in SEEDS:
            o=joblib.load(old_path(d,s));spec=t._spec(role,TARGET[d],s,'FULL')
            assert o['source_rows']==sources[d] and o['test']==spec['test'] and o['pool']==spec['pool']
    arms=[('mapping_'+m,900,'FULL',list(TARGET),m) for m in ['PLS-2','SVR-1','MLP-8','TT-SMALL']]
    arms += [('budget_C20',900,'20',list(TARGET),'RIDGE-10')]
    arms += [('prefix_'+str(h),h,'FULL',MISSING_PREFIX,'RIDGE-10') for h in [300,600]]
    for arm,h,budget,ds,model in arms:
        # One task/direction; process-level isolation prevents runtime-module cross talk.
        for d in ds:
            bounds=[]
            for s in SEEDS:
                spec=t._spec(role,TARGET[d],s,budget)
                unique={tuple(sorted(p['A'])) for p in spec['positions']}
                unique|={tuple(sorted(f['T'])) for p in spec['positions'] for f in p['inner']}
                assert all(set(v)<=set(spec['pool']) for v in unique)
                n=len(unique)*(1 if spec['d']==1 else 5)+(0 if spec['d']==1 else 20)
                if arm.startswith('prefix_'):n+=12 # four endpoint LR and at most eight dual heads
                bounds.append(n)
            source_units=0 if arm.startswith('budget_') else (20 if model in ['MLP-8','TT-SMALL'] else 4 if model=='SVR-1' else 1)
            upper=sum(bounds)+source_units
            tasks.append(dict(id=arm+'__'+d,arm=arm,horizon=h,budget=budget,direction=d,model=model,
                              target_fit_upper=sum(bounds),source_fit_upper=source_units,science_upper=upper,
                              cap=upper+max(5,int(np.ceil(.1*upper)))))
    # Hash main text/figure artifacts and all direct frozen data/protocol dependencies.
    locked=[]
    for folder in ['uteg_results_figures_20261026','uteg_hubei_cumulative_external_20261027_184309']:
        locked += [p for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.svg','.pdf','.html','.csv','.json']]
    locked += [OUT/'uteg_manuscript_scaffold_20261008_185739/main.tex',BASE/'DATA15.npz',BASE/'PREFIX_X.npz',BASE/'ROLE_FREEZE.json']
    for dep in [OLD,SEQ,BASE,OUT/'uteg_hill_exp_matched_20261012_153337',OUT/'uteg_ridge_dual_head_prepare_20261026_172952',OUT/'uteg_ridge_dual_head_unified_selector_20261026_190142']:
        locked += list(dep.glob('*.py'))
    hashes={str(p):sha(p) for p in sorted(set(locked))}
    plan=dict(version=1,seeds=SEEDS,directions=TARGET,source_rows=sources,tasks=tasks,
              formal_fit_upper=sum(v['science_upper'] for v in tasks),total_cap=sum(v['cap'] for v in tasks),
              protocol_sha=sha(ROOT/'PROTOCOL.md'),runner_sha=sha(__file__),immutable_hashes=hashes,
              new_test_results_used_for_selection=False,created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    write(ROOT/'PLAN.json',plan)
    print(json.dumps({k:plan[k] for k in ['formal_fit_upper','total_cap']},indent=2),flush=True)

def train_task(task_id,limit=None):
    plan=read(ROOT/'PLAN.json');task=next(x for x in plan['tasks'] if x['id']==task_id)
    assert sha(ROOT/'PROTOCOL.md')==plan['protocol_sha'] and sha(__file__)==plan['runner_sha']
    r,runner,t,seq,c,sel=env(task_id,task['cap'])
    for f,v in [('RUNNER_FREEZE.json',{'runner_sha':plan['runner_sha'],'protocol_sha':plan['protocol_sha']}),('ROUND_START_LEDGER.json',{'appendix_plan_sha':sha(ROOT/'PLAN.json'),'task':task})]:
        if not (r.ROOT/f).exists():r.write(f,v)
    D=data_at(task['horizon']);role=read(BASE/'ROLE_FREEZE.json');d=task['direction'];rows=plan['source_rows'][d]
    n=0
    for s in SEEDS:
        done=r.ROOT/'frozen'/f'{s}.json'
        if done.exists():continue
        start=time.perf_counter();src_seed=s if task['model'] in ['MLP-8','TT-SMALL'] else 1000
        srcpath=r.ROOT/'source'/f'{src_seed}.joblib'
        if not srcpath.exists():
            if task['arm']=='budget_C20':
                model=joblib.load(old_path(d,s))['source_model']
            else:
                S=dict(D);S['medical']=D['medical'].copy();S['ci']=D['ci'].copy()
                excluded=np.ones(len(D['x']),bool);excluded[rows]=False
                S['medical'][excluded]=np.nan;S['ci'][excluded]=np.nan
                r.set_scope('SOURCE_'+str(src_seed));attempt=r.register_attempt('fit','source',units=4 if task['model']=='SVR-1' else 1)
                try:
                    model=t.m.fit_source(S,rows,[],task['model'],src_seed,steps=500 if task['model'] in ['MLP-8','TT-SMALL'] else None)
                    assert np.isfinite(t.m.predict(model,D['x'][rows])).all()
                    r.finish_attempt(attempt,'COMPLETE')
                except Exception as e:r.finish_attempt(attempt,'FAILED',error=str(e));raise
            srcpath.parent.mkdir(exist_ok=True);joblib.dump({'families':{'RIDGE':{'model':model}},'actual_model':task['model']},srcpath,compress=3)
        objpath=runner.fit_target_one(t,d,s,srcpath,role,D,rows,plan['protocol_sha']);obj=joblib.load(objpath)
        assert obj['test']==joblib.load(old_path(d,s))['test']
        for p in obj['positions'].values():
            assert set(p['A']+p['B'])<=set(obj['pool'])
            for f in p['inner']:assert not set(f['T'])&set(f['V'])
        freeze={'target_sha':sha(objpath),'target_path':str(objpath),'seed':s,'task':task,'fit_seconds':time.perf_counter()-start,'test_evaluated':False}
        if task['arm'].startswith('prefix_'):
            dual=seq.fit_dual(r,c,runner,obj,D,d,s)
            freeze['threshold']=seq.choose(sel,dual['outer'])
            freeze['dual_sha']=sha(r.ROOT/'dual'/f'{d}_{s}.joblib')
        write(done,freeze);n+=1
        write(r.ROOT/'STATUS.json',{'state':'TRAINING','seed':s,'completed':len(list((r.ROOT/'frozen').glob('*.json'))),'fits':r.current_totals(),'last_seconds':time.perf_counter()-start})
        print(task_id,s,round(time.perf_counter()-start,2),r.current_totals(),flush=True)
        if limit and n>=limit:return
    write(r.ROOT/'COMPLETE.json',{'state':'MODELS_FROZEN','count':20,'fits':r.current_totals(),'test_evaluated':False})

def launch(workers):
    plan=read(ROOT/'PLAN.json');pending=[t for t in plan['tasks'] if not (ROOT/'runs'/t['id']/'COMPLETE.json').exists()]
    # Representative task of each arm first: implementation defects surface before a wide run.
    arms=list(dict.fromkeys(x['arm'] for x in pending));pending.sort(key=lambda x:(list(y['id'] for y in pending if y['arm']==x['arm']).index(x['id']),arms.index(x['arm'])))
    active=[];failed=[];logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active)<workers and not failed:
            t=pending.pop(0);f=(logdir/(t['id']+'.log')).open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-B',str(__file__),'task','--id',t['id']],stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT))
            active.append((p,t,f))
        for p,t,f in active[:]:
            if p.poll() is not None:
                f.close();active.remove((p,t,f))
                if p.returncode:failed.append(t['id'])
        write(ROOT/'BATCH_STATUS.json',{'running':[{'id':t['id'],'pid':p.pid} for p,t,f in active],'remaining':len(pending),'failed':failed,'completed':len(list((ROOT/'runs').glob('*/COMPLETE.json'))),'time':time.strftime('%Y-%m-%dT%H:%M:%S')})
        if failed and not active:break
        time.sleep(2)
    if failed:raise RuntimeError('Batch stopped; no new tasks scheduled after failure: '+','.join(failed))
    assert len(list((ROOT/'runs').glob('*/COMPLETE.json')))==len(plan['tasks'])
    write(ROOT/'ALL_MODELS_FROZEN.json',{'count':len(plan['tasks']),'test_evaluated':False,'plan_sha':sha(ROOT/'PLAN.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plan','task','launch']);ap.add_argument('--id');ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if args.command=='plan':make_plan()
    elif args.command=='task':train_task(args.id,args.limit)
    else:launch(args.workers)
