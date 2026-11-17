"""Ten development-only HGB bootstrap fits, mean score; B-anchored mapping fixed."""
import run as r
import fn_candidates as f
import support_followup as s
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from compact_results import install
install(r)
O=r.OUT/'aux_followup'/'hgb_ensemble';N=10

def scoring(out,d,seed,arm,pos):
    a=np.asarray(out['train']);b=np.asarray(out['query']);p=out['inner'];pq=out['prediction']
    _,_,features=r.gates.fields(a,out['inner_b'],p,a,900);_,_,query=r.gates.fields(b,out['query_b'],pq,a,900)
    pc=r.gates.fr.ma_class(p);truth=r.gates.fr.ma_class(d['medical'][a,2]);qc=r.gates.fr.ma_class(pq)
    error=pc!=truth;groups=d['groups'][a];unique=np.unique(groups)
    rng=np.random.default_rng(seed+5039*pos);scores=[];audit=[]
    # Resample whole groups within gate-training data. Neither evaluation rows
    # nor their labels enter bootstrap selection. Preserve record dependence.
    for member in range(N):
        sampled=rng.choice(unique,len(unique),replace=True)
        ix=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        h=r.gates.rc.fit_hgb(features[ix],pc[ix],truth[ix],np.ones(len(ix),bool),direction=arm,seed=seed+member,pos=pos)
        scores.append(r.gates.rc.score_hgb(h,query,qc,np.ones(len(b),bool)))
        audit.append(dict(member=member,n=len(ix),errors=int(error[ix].sum()),status=h['status'],sample_rows=a[ix]))
    return np.mean(scores,axis=0),audit

def compute(d,sp,arm,seed):
    ds=[];dp=[];dy=[];devrows=[];audits=[]
    for pos,part in enumerate(sp['positions'][:3]):
        out=r.c.evaluate(part['A'],part['B'],part['inner'],arm,900);score,au=scoring(out,d,seed,arm,pos)
        ds.extend(score);dp.extend(out['prediction']);dy.extend(d['medical'][out['query'],2]);devrows.extend(out['query']);audits.append(au)
    out=r.c.evaluate(sp['pool'],sp['test'],sp['positions'][3]['inner'],arm,900);score,au=scoring(out,d,seed,arm,3);audits.append(au)
    pick=r.gates.rc.select_threshold(np.asarray(ds),r.gates.fr.ma_class(np.asarray(dp)),r.gates.fr.ma_class(np.asarray(dy)),.2-1e-12);t=pick.get('threshold')
    action=np.zeros(len(sp['test']),bool) if t is None else np.isfinite(score)&(score<=t)
    return dict(prediction=out['prediction'],score=score,dev_score=np.array(ds),dev_prediction=np.array(dp),dev_rows=np.array(devrows),action=action,pick=pick,audits=audits)

def run_one(item,budget,d,plan):
    dest=O/f'dev{budget}/private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    start=time.time();dest.mkdir(parents=True,exist_ok=True);r.c.OUT=dest;f.draw(item,plan);s.configure('anchored_tails');sp=s.spec(item,budget);rows=[]
    for arm in s.ARMS:
        x=compute(d,sp,arm,item['seed']);v=r.metrics(x['prediction'],d['medical'][sp['test'],2],x['action'])
        rows.append(dict(rep=item['rep'],budget=budget,arm=arm,rule='HGB_bootstrap_mean10',development=x['pick'],test=v,joint_test_pass=f.passed(v)))
        np.savez_compressed(dest/(arm+'.npz'),test=sp['test'],**{k:v for k,v in x.items() if k not in ['pick','audits']})
        r.write(dest/(arm+'_training.json'),dict(pick=x['pick'],audits=x['audits']))
    r.write(dest/'RESULTS.json',rows);r.write(dest/'COMPLETE.json',dict(rep=item['rep'],seconds=time.time()-start));print('HGB10',budget,item['rep'],round(time.time()-start,2),flush=True)

def aggregate():
    for budget in [54,60]:
        rows=[v for p in sorted((O/f'dev{budget}/private').glob('round_*/COMPLETE.json')) for v in json.loads((p.parent/'RESULTS.json').read_text(encoding='utf8'))]
        r.write(O/f'dev{budget}/ROUNDS.json',rows);r.write(O/f'dev{budget}/SUMMARY.json',f.summarize(rows,['arm','rule']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=60);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    if args.stage=='aggregate':aggregate();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    protocol=dict(repeats=100,budgets=[54,60],members=N,split_sha256=r.sha(r.OUT/'private/SPLITS.json'),mapping='Previous B-only anchored_tails, unchanged.',bootstrap='Whole training groups sampled with replacement;10 fixed members; original HGB config; mean query score.',selection='Threshold on nested-development ensemble scores, accuracy>80% (error<20%); no evaluation-label or evaluation-reporting-rate adjustment.',scope='Separate exploratory sensitivity; not a guarantee of zero FN.')
    file=O/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text(encoding='utf8'))==protocol
    else:r.write(file,protocol)
    if args.stage=='smoke':
        item=items[19];f.draw(item,plan);s.configure('anchored_tails');sp=s.spec(item,60);x=compute(d,sp,'HZ_to_SA',item['seed']);truth=d['medical'][sp['test']].copy()
        try:
            d['medical'][sp['test']]=9999.;r.c._SOURCE.clear();z=compute(d,sp,'HZ_to_SA',item['seed'])
            for k in ['prediction','score','dev_score','action']:np.testing.assert_allclose(x[k],z[k],atol=1e-9,rtol=0)
            assert r.safe(x['pick'])==r.safe(z['pick'])
        finally:d['medical'][sp['test']]=truth
        r.write(O/'SMOKE.json',dict(test_label_poison_invariance=True,training_group_bootstrap=True));print('HGB10 SMOKE PASS',flush=True)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items[args.start:args.stop]:run_one(item,args.budget,d,plan)
    inp=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes']
    assert all(r.sha(r.current.ROOT/'data'/k)==h for k,h in inp.items())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
