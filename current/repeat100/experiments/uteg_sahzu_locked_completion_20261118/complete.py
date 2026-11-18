"""Complete the locked SAHZU60/244 protocol; no manuscript edits or tuning."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import os,sys,json,time,argparse,importlib.util,copy
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
P=Path(str(_repeat_root / 'experiments' / 'uteg_sahzu60_test244_20261118'))
OLD=Path(str(_repeat_root / 'experiments' / 'uteg_locked_expansion_20261117'))
L=Path(str(_repeat_root / 'experiments' / 'uteg_sahzu_lowfn_20261118/symmetric90_reselected'))
O=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sahzu_extension_runner',P/'run.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
import numpy as np
import additional as ad
import extensions as ex
import aggregate as ag
import capability as cap
from threadpoolctl import threadpool_limits
e=q.e;r=q.r;c=q.c
ITEMS=json.loads((P/'private/SPLITS.json').read_text())
INDEX={(i['rep'],i['budget']):i for i in ITEMS}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def setup():
    (O/'runtime').mkdir(parents=True,exist_ok=True)
    d=q.init();c.OUT=O/'runtime';r.suite.ROOT=O/'runtime';e.O=O;ad.O=O;ex.O=O;ag.O=O;ag.S=O/'summary';ag.S.mkdir(parents=True,exist_ok=True)
    e.ARMS=q.ARMS;e.a.ARMS=q.ARMS[1:]
    def activate(item,plan,dest):
        dest.mkdir(parents=True,exist_ok=True);q.f.activate(item,d)
    e.activate=activate
    e.specification=lambda item,budget,role:INDEX[item['rep'],budget]['targets']['SAHZU']
    e.b.load=lambda budget,rep,arm:(load(P/f'private/dev{budget}/round_{rep:03d}/{arm}.npz'),None,None)
    return d
def protected():
    paths=list((q.f.WS/'outputs/uteg_nc_goal_20261030/manuscript').glob('*.tex'))
    paths+=list((q.f.WS/'outputs/uteg_nc_goal_20261030/manuscript').glob('*.pdf'))
    paths+=list((q.f.WS/'outputs/uteg_nc_goal_20261030/manuscript').glob('*.zip'))
    paths+=list((L.parent/'locked60_20261118').rglob('*.json'))
    paths +=[P/'private/SPLITS.json',r.current.ROOT/'data/DATA15_CURRENT.npz',r.current.ROOT/'data/CURRENT_SCALED_CURVES.npz']
    return {str(p):r.sha(p) for p in paths if p.is_file()}
def prepare(d):
    if not (O/'PROTECTED.json').exists():r.write(O/'PROTECTED.json',protected())
    old=read(q.OLD);checks=0
    for item in ITEMS:
        assert item['sources']==old[item['rep']]['sources']
        sp=item['targets']['SAHZU'];q.separate(d,sp['pool'],sp['test'])
        assert len(sp['pool'])==item['budget'] and len(sp['test'])==220
        for part in sp['positions']:
            q.separate(d,part['A'],part['B'])
            for fold in part['inner']:q.separate(d,fold['T'],fold['V']);checks+=1
    r.write(O/'PROTOCOL.json',dict(primary='Fixed approved60development/244test pool,220test per repeat',sensitivity='54 of the same60, matched sources/test, separate100 rounds',repeats=100,
        policy='Fixed symmetric90 low-risk normal veto, original high-risk/B-bound/normal gate plus original abnormal gate selector; no hyperparameter or cutoff search.',
        early='Each checkpoint uses its own saved prefix parameters for predictions, gates and amplitude bounds. No15-minute B in early gate.',
        controls='Same source fitted controls and unchanged source draws, local corrections refit on new roles. Fixed linear loss4/alpha.01; existing Ridge development selection retained.',
        ablation='Remove one final normal check with all remaining fitted thresholds and abnormal gates held fixed; no reselection.',
        statistics='100 overlapping repeated-sample estimates, mean and sampleSD, not independent cohorts or95%CI. Undefined denominators explicit.',
        unchanged='Other target centres,internal301,old54/60,manuscript,PDF,ZIP,GitHub',group_disjoint_checks=checks))
def fit_policy(d,z,h,base):
    dp=z['dev_prediction'];dr=z['dev_risk'];norm=r.state(dp)==1
    if not base.get('feasible') and not base.get('normal',{}).get('feasible'):
        return dict(feasible=False,reason='No feasible development normal branch')
    if not norm.any():return dict(feasible=False,reason='No development normal predictions')
    xx=c.feature([],[],h,'four')[z['dev_rows'],1];ds={k:z['dev_score_'+k] for k in e.a.RULES}
    guard=float(np.quantile(dr[norm,0],.9))
    nm=e.cal.normal_action(base['normal'],dp,dr,np.c_[-xx,xx],ds)&(dr[:,0]<=guard)
    ab=e.cal.c.choose(dp,d['medical'][z['dev_rows'],2],ds,nm,0)
    return dict(normal=base['normal'],low_guard=guard,abnormal=ab,feasible=bool(ab['feasible']))
def apply_policy(d,z,h,pick,dev=False,drop=()):
    pre='dev_' if dev else '';ids=z['dev_rows'] if dev else z['test'];pred=z[pre+'prediction'];raw=z['dev_risk'] if dev else z['test_risk'];ss={k:z[pre+'score_'+k] for k in e.a.RULES}
    if not pick['feasible']:return np.zeros(len(pred),bool)
    p=pick['normal'];bb=c.feature([],[],h,'four')[ids,1];pc=r.state(pred)
    act=(pc==1)
    if 'high_risk' not in drop:act &= raw[:,1]<=p['cap_score']
    if 'low_risk' not in drop:act &= raw[:,0]<=pick['low_guard']
    if 'B_bounds' not in drop:act &= (-bb<=p['limits']['low'])&(bb<=p['limits']['high'])
    if 'normal_score' not in drop:act &= e.cal.b.gate_action(p['gate'],ss,len(pred))
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:act|=(pc==cls)&e.cal.b.gate_action(pick['abnormal'][key],ss,len(pred))
    return act
def finish_bundle(d,z,h):
    rows,pol,acts,curves=e.evaluate_rules(d,z,z['dev_risk'],z['test_risk'],h)
    pick=fit_policy(d,z,h,pol['common_B']);pol['previous_common_B']=pol['common_B'];pol['common_B']=pick
    act=apply_policy(d,z,h,pick);da=apply_policy(d,z,h,pick,True);acts['common_B']=act
    pc=r.state(z['prediction']);tc=r.state(d['medical'][z['test'],2]);err=pc!=tc;fp=(pc!=1)&(tc==1);fn=(pc==1)&(tc!=1)
    for row in rows:
        if row['rule']=='common_B':
            row.update(feasible=pick['feasible'],test=e.n.extended_metrics(z['prediction'],d['medical'][z['test'],2],act),development=e.n.extended_metrics(z['dev_prediction'],d['medical'][z['dev_rows'],2],da),
                disposition=dict(before_correct=int((~err).sum()),before_incorrect=int(err.sum()),correct_released=int((act&~err).sum()),correct_withheld=int((~act&~err).sum()),incorrect_released=int((act&err).sum()),incorrect_withheld=int((~act&err).sum()),FP_withheld=int((~act&fp).sum()),FN_withheld=int((~act&fn).sum())))
    for row in rows:
        m=row['test'];cm=np.array(m['confusion']);assert cm.sum()==m['reported'] and cm.trace()==m['correct']
        assert cm[0,1]+cm[2,1]==m['FN'] and cm[1,0]+cm[1,2]==m['FP']
    return rows,pol,acts,curves
def bundle(d,item,arm,h):
    sp=item['targets']['SAHZU'];z=e.collect(d,sp,arm,item['seed'],h);dr,qr,audit=e.risks(d,sp,z['dev_rows'],h);z.update(dev_risk=dr,test_risk=qr)
    return z,audit
def smoke(d):
    checks=[]
    for h in [300,600]:
        item=INDEX[0,60];q.f.activate(item,d);z,_=bundle(d,item,'HZ+HB_to_SA',h);left=finish_bundle(d,z,h)
        # Poison all labels outside allowed local development and source training.
        allowed=np.r_[item['targets']['SAHZU']['pool'],c.source_rows('HZ+HB_to_SA',h)];bad=np.setdiff1d(np.arange(len(d['x'])),allowed)
        save=d['medical'].copy();savey=d['y'].copy()
        try:
            d['medical'][bad]=9999;d['y'][bad]=9999;q.f.activate(item,d);zz,_=bundle(d,item,'HZ+HB_to_SA',h);right=finish_bundle(d,zz,h)
            for k in z:np.testing.assert_allclose(z[k],zz[k],atol=1e-10,rtol=1e-10)
            assert r.safe(left[1])==r.safe(right[1])
            for k in left[2]:np.testing.assert_array_equal(left[2][k],right[2][k])
        finally:d['medical'][:]=save;d['y'][:]=savey;c._SOURCE.clear()
        # No full-record feature access is allowed in an early evaluation.
        original=c.feature
        def guarded(a,b,t,representation='B'):
            assert t==h,('future-feature access',t,h)
            return original(a,b,t,representation)
        try:
            c.feature=guarded;q.f.activate(item,d);zz,_=bundle(d,item,'HZ+HB_to_SA',h);rr=finish_bundle(d,zz,h)
            for k in z:np.testing.assert_allclose(z[k],zz[k],atol=1e-10,rtol=1e-10)
            assert r.safe(left[1])==r.safe(rr[1])
        finally:c.feature=original;c._SOURCE.clear()
        checks.append(dict(time=h,test_label_poison_pass=True,only_matching_checkpoint_features=True));print('SMOKE',h,'PASS',flush=True)
    for budget in [54,60]:
        item=INDEX[0,budget];q.f.activate(item,d)
        frozen=load(L/f'private/dev{budget}/round_000.npz')
        for arm in q.ARMS:
            z=load(P/f'private/dev{budget}/round_000/{arm}.npz');rr=finish_bundle(d,z,900);np.testing.assert_array_equal(rr[2]['common_B'],frozen[arm+'_action'])
    r.write(O/'SMOKE.json',dict(passed=True,checks=checks,locked15min_actions_equal=True))
def primary(d,item,h,linear=False):
    budget=item['budget'];rep=item['rep'];dest=O/('linear' if linear else 'primary')/f'dev{budget}/time{h}/round_{rep:02d}'
    if (dest/'COMPLETE.json').exists():return
    start=time.time();q.f.activate(item,d);ex.LINEAR_CACHE.clear();oldpredict=c.predict
    if linear:c.predict=ex.linear_predict
    try:
        rows=[];curves=[];policies={}
        for arm in q.ARMS:
            if h==900 and not linear:z=load(P/f'private/dev{budget}/round_{rep:03d}/{arm}.npz');audit={"reuse":"unchanged15min predictions"}
            else:z,audit=bundle(d,item,arm,h)
            rr,pol,acts,cc=finish_bundle(d,z,h)
            if h==900 and not linear:
                locked=load(L/f'private/dev{budget}/round_{rep:03d}.npz');np.testing.assert_array_equal(acts['common_B'],locked[arm+'_action'])
            zz={k:v for k,v in z.items() if not k.startswith('action_')};zz.update({'action_'+k.replace(' ','_'):v for k,v in acts.items()})
            dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(arm+'.npz'),**zz)
            r.write(dest/(arm+'_risk_training.json'),audit)
            rows.extend(dict(x,budget=budget,rep=rep,arm=arm,time=h) for x in rr);curves.extend(dict(x,budget=budget,rep=rep,arm=arm,time=h) for x in cc);policies[arm]=pol
        r.write(dest/'RESULTS.json',rows);r.write(dest/'CURVES.json',curves);r.write(dest/'POLICIES.json',policies);r.write(dest/'COMPLETE.json',dict(seconds=time.time()-start))
        print('LINEAR' if linear else 'PRIMARY',budget,h,rep+1,round(time.time()-start,1),flush=True)
    finally:c.predict=oldpredict
def table1(d,models,item):
    rep=item['rep'];dest=O/f'table1/round_{rep:02d}';dest.mkdir(parents=True,exist_ok=True)
    for arm in q.ARMS[1:]:
        parent=OLD/('nine_controls' if arm=='HB_to_SA' else 'table1')/f'round_{rep:02d}'
        for method in r.mapping.METHODS:
            src=parent/(arm+'_'+method+'_source.joblib');target=dest/src.name
            assert src.exists(),src
            if not target.exists():os.link(src,target)
    e.table1_round(d,None,None,models,item)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','primary','table1','budgets','linear']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    d=setup();prepare(d)
    if args.stage=='smoke':smoke(d);return
    assert read(O/'SMOKE.json')['passed']
    models=r.suite.numerical_models() if args.stage in ['table1','budgets'] else None
    role={'roles':{'SAHZU':{'pool_full':INDEX[0,60]['targets']['SAHZU']['pool']}}}
    for rep in range(args.start,args.stop):
        if args.stage=='table1':table1(d,models,INDEX[rep,60])
        elif args.stage=='budgets':ad.budgets_round(d,None,role,models,INDEX[rep,60])
        else:
            for budget in [60,54]:
                for h in ([900] if args.stage=='linear' else [900,300,600]):primary(d,INDEX[rep,budget],h,args.stage=='linear')
    for p,sha in read(O/'PROTECTED.json').items():assert r.sha(Path(p))==sha,p
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
