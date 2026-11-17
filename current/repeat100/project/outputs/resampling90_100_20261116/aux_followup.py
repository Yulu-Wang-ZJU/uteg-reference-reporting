"""Train-only auxiliary U-TEG features for local MA correction;100 paired draws."""
import run as r
import fn_candidates as f
import support_followup as s
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from compact_results import install
install(r)
O=r.OUT/'aux_followup'
CONFIGS=[('B',[],0.),('B_t0',[0],.25),('B_k',[2],.25),('B_mu',[3],.25),('B_all_weak',[0,2,3],.25),('B_all_equal',[0,2,3],1.)]
NAMES=[x[0] for x in CONFIGS]
ARMS=['HZ_to_SA','HZ+HB_to_SA'];RULES=['HGB','Logistic','Boundary']
MODE='B'
LOCAL_COSTS={}

def kernel_design(x,a,b,aux,weight):
    cols=[1]+aux;train=x[a][:,cols];query=x[b][:,cols]
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    factors=np.r_[1.,np.ones(len(aux))*np.sqrt(weight/len(aux))]
    zt=(train-mean)/scale*factors;zq=(query-mean)/scale*factors
    def kernel(u,v):return np.exp(-np.maximum(np.sum(u*u,1)[:,None]+np.sum(v*v,1)[None,:]-2*u@v.T,0))
    K=kernel(zt,zt);eig,vec=np.linalg.eigh((K+K.T)/2);keep=eig>max(float(eig.max())*1e-12,1e-12)
    root=np.sqrt(eig[keep]);V=vec[:,keep]
    z=np.c_[V*root,np.ones(len(a))];q=np.c_[kernel(zq,zt)@(V/root),np.ones(len(b))]
    return z,q,dict(columns=cols,aux_kernel_weight=weight,train_mean=mean,train_sd=scale,kernel_rank=int(keep.sum()))

def prediction(direction,h,train,query,representation='B',loss='inward4',prior_override=None):
    s.MODE='anchored_tails'
    anchored,audit,xb=s.prediction(direction,h,train,query,representation,loss,prior_override)
    if MODE=='B':return anchored,audit,xb
    assert representation=='B' and loss=='inward4' and prior_override is None
    d,*_=r.c.context();a=np.asarray(train,int);b=np.asarray(query,int)
    assert not set(d['groups'][a])&set(d['groups'][b])
    _,aux,weight=CONFIGS[NAMES.index(MODE)];x=r.c.feature(a,b,h,'four')
    z,q,extra=kernel_design(x,a,b,aux,weight);y=d['medical'][a,2]
    w=r.c.selected.models.weights(y,0,'inward4');src=r.c.source_model(direction,h,'B','inward4')
    base=(r.c.source_predict(src,xb[a])-60)/10;bq=(r.c.source_predict(src,xb[b])-60)/10
    previous_cost=r.c.selected.COST
    try:
        r.c.selected.COST=LOCAL_COSTS.get(MODE,4.)
        coef,info=r.c.selected.solve_offset(z,(y-60)/10,w,base)
    finally:r.c.selected.COST=previous_cost
    p=np.maximum(60+10*(bq+q@coef),0.)
    below=xb[b,0]<xb[a,0].min();above=xb[b,0]>xb[a,0].max()
    p[below]=np.minimum(p[below],anchored[below]);p[above]=np.maximum(p[above],anchored[above])
    assert np.isfinite(p).all()
    audit=dict(mode=MODE,**extra,solver=info,training_rows=a,coefficient_sha256=r.c.selected.ahash(coef),tail_safeguard='B-only anchored prediction bounds outside current training B range')
    return p,audit,xb

def configure(name):
    global MODE
    MODE=name;f.configure('residual',4);r.c.predict=prediction

def select_threshold(score,pred,truth):
    """Joint-development feasible policies first minimize FN count.
    Nonzero FN remains explicitly unqualified; it is not relabelled as success.
    """
    pc=r.state(pred);tc=r.state(truth);choices=[]
    for t in np.unique(score[np.isfinite(score)]):
        action=np.isfinite(score)&(score<=t);n=int(action.sum());ab=action&(tc!=1)
        if n/len(pred)<=.75 or not ab.any():continue
        acc=float(np.mean(pc[action]==tc[action]))
        if acc<=.8:continue
        fn=int(np.sum(ab&(pc==1)))
        choices.append(dict(threshold=float(t),reported=n,reporting_rate=n/len(pred),accuracy=acc,FN=fn,FNR=fn/int(ab.sum()),abnormal_released=int(ab.sum()),joint=fn==0))
    if not choices:return dict(feasible=False)
    q=min(choices,key=lambda x:(x['FN'],-x['reported'],-x['accuracy'],x['threshold']))
    return dict(q,feasible=True)

def choose(policies):
    valid=[(i,j,p[rule]) for i,p in enumerate(policies) for j,rule in enumerate(RULES) if p[rule]['feasible']]
    if not valid:return None
    return min(valid,key=lambda t:(t[2]['FN'],-t[2]['reported'],-t[2]['accuracy'],t[0],t[1]))[:2]

def apply(x,rule,pick):return np.isfinite(x['scores'][rule])&(x['scores'][rule]<=pick['threshold'])

def main_round(item,budget,d,plan):
    dest=O/f'dev{budget}/private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    start=time.time();dest.mkdir(parents=True,exist_ok=True);r.c.OUT=dest;f.draw(item,plan);sp=s.spec(item,budget)
    fixed=[];candidates=[];selected=[]
    for arm in ARMS:
        store=[];policies=[]
        for name,aux,weight in CONFIGS:
            configure(name);x=f.collect(d,sp,arm,item['seed']);picks={k:select_threshold(x['ds'][k],x['dp'],x['dy']) for k in RULES};policies.append(picks);store.append(x)
            original={k:r.gates.rc.select_threshold(x['ds'][k],r.gates.fr.ma_class(x['dp']),r.gates.fr.ma_class(x['dy']),.2-1e-12) for k in RULES}
            np.savez_compressed(dest/(arm+'_'+name+'.npz'),test=sp['test'],dev_rows=x['dev_rows'],prediction=x['pred'],dev_prediction=x['dp'],**{'score_'+k:v for k,v in x['scores'].items()},**{'dev_score_'+k:v for k,v in x['ds'].items()})
            r.write(dest/(arm+'_'+name+'_selection.json'),dict(picks=picks,accuracy80=original))
        decision=choose(policies)
        # Per-rule model selection and B-only gate selection are separate controls.
        decisions={}
        for rule in RULES:
            valid=[i for i,p in enumerate(policies) if p[rule]['feasible']]
            decisions[rule]=min(valid,key=lambda i:(policies[i][rule]['FN'],-policies[i][rule]['reported'],-policies[i][rule]['accuracy'],i)) if valid else None
        base_choice=choose([policies[0]])
        truth=d['medical'][sp['test'],2]
        for i,(name,_,_) in enumerate(CONFIGS):
            x=store[i];saved=json.loads((dest/(arm+'_'+name+'_selection.json')).read_text(encoding='utf8'))
            for rule in ['All report',*RULES]:
                t=saved['accuracy80'].get(rule,{}).get('threshold')
                action=np.ones(len(truth),bool) if rule=='All report' else (np.zeros(len(truth),bool) if t is None else np.isfinite(x['scores'][rule])&(x['scores'][rule]<=t))
                v=r.metrics(x['pred'],truth,action)
                fixed.append(dict(rep=item['rep'],budget=budget,arm=arm,model=name,rule=rule,test=v,joint_test_pass=f.passed(v)))
            for rule,pick in policies[i].items():
                v=r.metrics(x['pred'],truth,apply(x,rule,pick)) if pick['feasible'] else None
                candidates.append(dict(rep=item['rep'],budget=budget,arm=arm,model=name,rule=rule,development=pick,test=v,joint_test_pass=f.passed(v)))
        for policy,choice in [('joint',decision),*([('B_gate_only',base_choice)] if NAMES[0]=='B' else []),*[(rule,None if index is None else (index,RULES.index(rule))) for rule,index in decisions.items()]]:
            if choice is None:
                selected.append(dict(rep=item['rep'],budget=budget,arm=arm,policy=policy,model=None,rule=None,development=dict(feasible=False),test=None,joint_test_pass=False));continue
            i,j=choice;rule=RULES[j];pick=policies[i][rule];v=r.metrics(store[i]['pred'],truth,apply(store[i],rule,pick))
            selected.append(dict(rep=item['rep'],budget=budget,arm=arm,policy=policy,model=NAMES[i],rule=rule,development=pick,test=v,joint_test_pass=f.passed(v)))
    r.write(dest/'FIXED.json',fixed);r.write(dest/'CANDIDATE.json',candidates);r.write(dest/'SELECTED.json',selected);r.write(dest/'COMPLETE.json',dict(rep=item['rep'],budget=budget,seconds=time.time()-start))
    print('AUX',budget,item['rep'],round(time.time()-start,2),flush=True)

def summarize():
    for budget in [54,60]:
        for key in ['FIXED','CANDIDATE','SELECTED']:
            rows=[v for p in sorted((O/f'dev{budget}/private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/(key+'.json')).read_text(encoding='utf8'))]
            keys=['arm','policy'] if key=='SELECTED' else ['arm','model','rule']
            r.write(O/f'dev{budget}'/(key+'_ROUNDS.json'),rows);summary=f.summarize(rows,keys)
            for q in summary:
                rr=[x for x in rows if all(x[k]==q[k] for k in keys)]
                if key=='SELECTED':
                    q['models']={name:sum(x['model']==name for x in rr) for name in [None,*NAMES]}
                    q['rules']={name:sum(x['rule']==name for x in rr) for name in [None,*RULES]}
            r.write(O/f'dev{budget}'/(key+'_SUMMARY.json'),summary)

def smoke(d,plan,items):
    item=items[0];f.draw(item,plan);sp=s.spec(item,54);arm='HZ_to_SA';configure('B');x=f.collect(d,sp,arm,item['seed']);old=np.load(s.O/'dev54/private/round_00/HZ_to_SA_anchored_tails.npz')
    np.testing.assert_allclose(x['pred'],old['prediction'],atol=1e-9,rtol=0)
    for k in RULES:np.testing.assert_allclose(x['scores'][k],old['score_'+k],atol=1e-9,rtol=0)
    configure('B_all_weak');x=f.collect(d,sp,arm,item['seed']);picks={k:select_threshold(x['ds'][k],x['dp'],x['dy']) for k in RULES};truth=d['medical'][sp['test']].copy()
    try:
        d['medical'][sp['test']]=9999.;configure('B_all_weak');z=f.collect(d,sp,arm,item['seed'])
        np.testing.assert_allclose(x['pred'],z['pred'],atol=1e-9,rtol=0)
        for k in RULES:
            np.testing.assert_allclose(x['scores'][k],z['scores'][k],atol=1e-9,rtol=0);assert picks[k]==select_threshold(z['ds'][k],z['dp'],z['dy'])
    finally:d['medical'][sp['test']]=truth
    # Query feature changes cannot alter the training basis or fitted statistics.
    a=np.asarray(sp['pool']);b=np.asarray(sp['test']);xx=d['x'].copy();z1,q1,st1=kernel_design(xx,a,b,[0,2,3],.25);xx[b]*=7;z2,q2,st2=kernel_design(xx,a,b,[0,2,3],.25)
    np.testing.assert_allclose(z1,z2,atol=1e-12,rtol=0);assert r.safe(st1)==r.safe(st2)
    r.write(O/'SMOKE.json',dict(B_anchor_reproduces_previous=True,poison_test_labels_does_not_change_fits_scores_selection=True,query_features_do_not_fit_scaling_or_basis=True))
    print('AUX SMOKE PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);a=ap.parse_args()
    if a.stage=='aggregate':summarize();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    proto=dict(configs=CONFIGS,budgets=[54,60],repeats=100,arms=ARMS,split_sha256=r.sha(r.OUT/'private/SPLITS.json'),cost=4,alpha=.01,
        kernel='Source remains B-only. Local correction kernel exp(-squared standardized B distance - lambda*mean standardized auxiliary squared distance). Onset t0, growth time k and shape mu are added individually and jointly. Scaling and basis from current training rows only.',
        tails='Outside training B support, preserve directional bounds from previous B-only anchored prediction.',
        controls='Fixed accuracy80 threshold per rule, all-report mapping metrics, per-rule feature selection, joint feature/rule selection, B-only rule selection.',
        selection='Candidates must have development accuracy>80%, reporting rate>75%, at least1 released abnormal. Minimize development FN count, then maximize reporting rate and accuracy. Ties prefer simpler/earlier feature set then HGB,Logistic,Boundary. Nonzero development FN explicitly flagged unqualified; no feasible candidate is NA.',
        note='The existing HGB/Logistic gates already receive all4 measured U-TEG coefficients; this experiment adds auxiliary information to the MA mapping. Historical data repeated exploration, not independent clinical validation. Manuscript unchanged.')
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==r.safe(proto)
    else:r.write(path,proto)
    if a.stage=='smoke':smoke(d,plan,items)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[a.start:a.stop]:main_round(item,a.budget,d,plan)
    inputs=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes']
    assert all(r.sha(r.current.ROOT/'data'/p)==v for p,v in inputs.items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
