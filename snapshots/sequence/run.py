"""Current-policy prefix validation. Historical numerical functions are read-only.
Training and evaluation are separate commands. No test-driven model selection.
"""
from pathlib import Path
import os,sys,json,types,hashlib,argparse,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
OLD=OUT/'uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
BASE=OUT/'uteg_legacy_aligned_threecenter_20261022_155614'
EXT=OUT/'uteg_ridge_dual_head_external_eval_20261026_191117'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np,joblib
from scipy.optimize import minimize
DIRECTIONS=['HZ_to_SA','HZ+HB_to_SA','SA+HB_to_HZ','SA+HZ_to_HB']
SEEDS=list(range(1000,1020))
TARGET={'HZ_to_SA':'SAHZU','HZ+HB_to_SA':'SAHZU','SA+HB_to_HZ':'Huzhou','SA+HZ_to_HB':'Hubei'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def conv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x))
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,default=conv,indent=2),encoding='utf-8');tmp.replace(p)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def env(h):
    # Transient runtime rebinding, never import a historical runtime for writing.
    rd=ROOT/f'T{h}';rd.mkdir(exist_ok=True)
    r=types.ModuleType('runtime');r.__file__=str(rd/'runtime.py');sys.modules['runtime']=r
    src=(OLD/'runtime.py').read_text(encoding='utf-8').replace('OUT.parent /','ROOT.parent.parent /')
    exec(compile(src,str(OLD/'runtime.py'),'exec'),r.__dict__)
    r.TOTAL_FIT_CAP=8000;r.SCIENCE_FIT_CAP=7800;r.REPAIR_FIT_CAP=200
    runner=r.load('prefix_ridge_runner_'+str(h),OLD/'run_ridge10_fullchain.py')
    target=r.load('prefix_target_'+str(h),BASE/'train_target_legacy.py') if False else None
    # Legacy wrapper's OLD_MODELS path assumes an outputs-level run root.
    target=types.ModuleType('prefix_target_'+str(h));target.__file__=str(BASE/'train_target_legacy.py')
    ts=(BASE/'train_target_legacy.py').read_text(encoding='utf-8')
    ts=ts.replace('OLD_MODELS = r.ROOT.parent / "uteg_hill_exp_matched_20261012_153337" / "models.py"',f'OLD_MODELS = Path({str(OUT / "uteg_hill_exp_matched_20261012_153337/models.py")!r})')
    exec(compile(ts,target.__file__,'exec'),target.__dict__)
    contract=r.load('prefix_dual_contract_'+str(h),OUT/'uteg_ridge_dual_head_prepare_20261026_172952/dual_head_contract.py')
    selector=r.load('prefix_selector_'+str(h),OUT/'uteg_ridge_dual_head_unified_selector_20261026_190142/run_unified_selector.py')
    return r,runner,target,contract,selector

def data_at(h):
    z=np.load(BASE/'DATA15.npz'); D={k:z[k].copy() for k in z.files}
    if h!=900:
        p=np.load(BASE/'PREFIX_X.npz');D['x']=p['x'][list(p['horizon_seconds']).index(h)].copy()
        # Prefix quality uses only prefix finiteness; no 15-minute test QC veto.
        D['input_qc']=np.isfinite(D['x']).all(1)
    return D

def old_obj(d,s):return OLD/'models'/d/'target15'/f'{TARGET[d]}_CFULL_seed{s}.joblib'
def preflight():
    D=data_at(900);role=read(BASE/'ROLE_FREEZE.json'); r,runner,t,c,sel=env(300)
    hashes={str(p):sha(p) for p in [BASE/'DATA15.npz',BASE/'PREFIX_X.npz',BASE/'ROLE_FREEZE.json',OLD/'run_ridge10_fullchain.py',Path(c.__file__),Path(sel.__file__)]}
    bounds={};rows={}
    for d in DIRECTIONS:
        objs=[joblib.load(old_obj(d,s)) for s in SEEDS]
        rows[d]=objs[0]['source_rows']
        assert all(o['source_rows']==rows[d] for o in objs)
        assert not set(rows[d])&set(objs[0]['pool']+objs[0]['test'])
        for s,o in zip(SEEDS,objs):
            spec=t._spec(role,TARGET[d],s,'FULL')
            assert spec['pool']==o['pool'] and spec['test']==o['test']
            for p in spec['positions']:
                assert p['A']==o['positions'][p['code']]['A'] and p['B']==o['positions'][p['code']]['B']
                for f in p['inner']: assert not set(f['T'])&set(f['V'])
        n=0
        for s in SEEDS:
            spec=t._spec(role,TARGET[d],s,'FULL')
            unique={tuple(sorted(p['A'])) for p in spec['positions']}
            unique|={tuple(sorted(f['T'])) for p in spec['positions'] for f in p['inner']}
            n+=len(unique)*(1 if spec['d']==1 else 5)+(0 if spec['d']==1 else 20)+4+8
        bounds[d]=n
    for h in [300,600]:
        P=data_at(h)
        assert np.isfinite(P['x']).all() and P['x'].shape==(480,4)
        assert not np.array_equal(P['x'],D['x'])
    assert not np.array_equal(data_at(300)['x'],data_at(600)['x'])
    # Single-source data identity: no mismatch hidden by global row indices.
    E=np.load(OUT/'uteg_hill_exp_matched_20261012_153337/EXP_DATA.npz')
    assert np.array_equal(E['ids'],D['ids'][:399]) and np.allclose(E['medical'],D['medical'][:399])
    plan={'directions':DIRECTIONS,'seeds':SEEDS,'horizons':[300,600,900],'source_rows':rows,
          'per_horizon_fit_upper':bounds,'new_source_fits':8,'new_fit_upper':2*sum(bounds.values())+8,
          'hard_cap':16000,'hashes':hashes,'test_selection':False,'time900_reuse':str(EXT),
          'prefix_input_audit':read(BASE/'PREFIX_X_IDENTITY.json')}
    assert plan['new_fit_upper']<15600
    write(ROOT/'PREFLIGHT.json',plan);print('PREFLIGHT PASS',plan['new_fit_upper'],flush=True)
    return plan

def score_head(c,model,b,ma):
    if model is None:return np.full(len(ma),np.inf)
    pp=model['pre'];xx=np.column_stack([b,ma]);xx=np.clip((np.where(np.isfinite(xx),xx,pp['median'])-pp['mean'])/pp['sd'],-10,10)
    return c.score_head(model['theta'],xx)

def fit_dual(r,c,runner,obj,D,d,s):
    path=r.ROOT/'dual'/f'{d}_{s}.joblib'
    if path.exists():return joblib.load(path)
    heads={};outer=[]
    for pos,p in obj['positions'].items():
        A=p['A'];B=p['B'];ad=obj['gate_models'][pos]['adapter'];med=p['cross'][ad];pc=runner.ma_class(med[:,2]);truth=runner.ma_class(D['medical'][A,2]); pair={}
        for head in ['low','high']:
            mask,y=c.build_head_mask(truth,pc,med[:,2],D['input_qc'][A],head)
            support=c.support_summary(y,D['ids'][A][mask])
            if support['status']!='SUPPORTED':pair[head]=None;continue
            zz,pp=c.build_features(D['x'][A,1][mask],med[:,2][mask],np.ones(mask.sum(),bool));w=c.class_balanced_weights(y)
            key=f'DUAL_{d}_{s}_{pos}_{head}';r.set_scope(key);attempt=r.register_attempt('fit','dual-head')
            try:
                opt=minimize(c.weighted_bce_objective,np.zeros(3),args=(zz,y,w,.1),jac=c.weighted_bce_gradient,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9,'gtol':1e-9})
                assert opt.success and np.isfinite(opt.x).all()
                pair[head]={'theta':opt.x,'pre':pp,'support':support};r.finish_attempt(attempt,'COMPLETE')
            except Exception as e:r.finish_attempt(attempt,'FAILED',error=str(e));raise
        heads[pos]=pair
        if B:
            ob=obj['gate_models'][pos]['outer_B_data'];medb=p['vv'][ad];score=np.maximum(score_head(c,pair['low'],D['x'][B,1],medb[:,2]),score_head(c,pair['high'],D['x'][B,1],medb[:,2]))
            f=runner.ma_features(D['x'][B],medb,obj['source_model']['ym'],obj['source_model']['ys']);pc=runner.ma_class(medb[:,2]);bs=runner.score_ma_gate(obj['gate_models'][pos]['model'],f,pc)
            for j,i in enumerate(B):
                valid=D['input_qc'][i] and np.isfinite(medb[j]).all()
                outer.append({'record_id':str(D['ids'][i]),'pred_class':int(pc[j]),'truth_class':int(runner.ma_class(D['medical'][[i],2])[0]),'dual_score':float(score[j]) if valid and pc[j]==0 and np.isfinite(score[j]) else None,'base_score':float(bs[j]) if valid and np.isfinite(bs[j]) else None})
    out={'heads':heads,'outer':outer};path.parent.mkdir(exist_ok=True);joblib.dump(out,path,compress=3);return out

def choose(sel,rows):
    vals=[None]+sorted({v['dual_score'] for v in rows if v['pred_class']==0 and v['dual_score'] is not None})
    cand=[{'threshold':v,'candidate_index':j,'n':len(rows),'release_record_ids':[x['record_id'] for x in rows if v is not None and x['pred_class']==0 and x['dual_score'] is not None and x['dual_score']<=v]} for j,v in enumerate(vals)]
    normal=sel.choose_normal(cand,rows,{x['record_id']:x for x in rows});tail=sel.choose_common_tail(rows,normal)
    n=normal['selected'];q=tail['selected']
    return {'normal':n['threshold'] if n else None,'tail':q['threshold'] if q else None,'all_withhold':q is None,'normal_status':normal['status'],'tail_status':tail['status']}

def train(limit=None):
    plan=read(ROOT/'PREFLIGHT.json')
    assert all(sha(p)==v for p,v in plan['hashes'].items())
    count=0
    for h in [300,600]:
        r,runner,t,c,sel=env(h);D=data_at(h);role=read(BASE/'ROLE_FREEZE.json')
        for f,v in [('RUNNER_FREEZE.json',{'runner_sha':sha(__file__),'protocol_sha':sha(ROOT/'PROTOCOL.txt')}),('ROUND_START_LEDGER.json',{'new_round':True})]:
            if not (r.ROOT/f).exists():r.write(f,v)
        for d in DIRECTIONS:
            srcpath=r.ROOT/'source'/f'{d}.joblib';rows=plan['source_rows'][d]
            if not srcpath.exists():
                # Exclude all other centers' labels even from the in-memory fit input.
                S=dict(D);S['medical']=D['medical'].copy();S['ci']=D['ci'].copy();excluded=np.ones(len(D['x']),bool);excluded[rows]=False;S['medical'][excluded]=np.nan;S['ci'][excluded]=np.nan
                r.set_scope('SOURCE_'+d);attempt=r.register_attempt('fit','source')
                model=t.m.fit_source(S,rows,[],'RIDGE-10',1000);runner._assert_ridge10_full_model(model,rows)
                srcpath.parent.mkdir(exist_ok=True);joblib.dump({'families':{'RIDGE':{'model':model}}},srcpath,compress=3);r.finish_attempt(attempt,'COMPLETE')
            for s in SEEDS:
                out=r.ROOT/'frozen'/f'{d}_{s}.json'
                if out.exists():continue
                start=time.time();objpath=runner.fit_target_one(t,d,s,srcpath,role,D,rows,sha(ROOT/'PROTOCOL.txt'));obj=joblib.load(objpath)
                dual=fit_dual(r,c,runner,obj,D,d,s);threshold=choose(sel,dual['outer'])
                write(out,{'direction':d,'seed':s,'horizon':h,'threshold':threshold,'target_sha':sha(objpath),'dual_sha':sha(r.ROOT/'dual'/f'{d}_{s}.joblib'),'test_evaluated':False})
                count+=1;write(ROOT/'STATUS.json',{'stage':'TRAINING','horizon':h,'direction':d,'seed':s,'new_contexts_this_run':count,'fit_totals_this_horizon':r.current_totals()})
                print('FROZEN',h,d,s,'seconds',round(time.time()-start,1),'fits',r.current_totals()['total'],flush=True)
                if limit and count>=limit:return
    frozen=list(ROOT.glob('T*/frozen/*.json'));assert len(frozen)==160
    write(ROOT/'ALL_MODELS_FROZEN.json',{'count':160,'models':{str(p):sha(p) for p in frozen},'test_evaluated':False,'protocol_sha':sha(ROOT/'PROTOCOL.txt')})

def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['preflight','train']);a.add_argument('--limit',type=int);args=a.parse_args()
    if args.command=='preflight':preflight()
    else:train(args.limit)
if __name__=='__main__':main()
