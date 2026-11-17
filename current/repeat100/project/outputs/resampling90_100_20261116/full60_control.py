"""Training-budget control: retain all original60 local development records.
Source and test use exactly the 90% draws of the primary resampling protocol.
This control does not replace variation from subsampling local development.
"""
import run as r
import fn_candidates as f
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
O=r.OUT/'full60_control'

def run_one(item,d,plan,role):
    dest=O/'private'/f"round_{item['rep']:02d}"
    if (dest/'COMPLETE.json').exists():return
    start=time.time();dest.mkdir(parents=True,exist_ok=True);r.c.OUT=dest;f.draw(item,plan);f.configure('residual',4)
    a=np.asarray(role['roles']['SAHZU']['pool_full']);b=np.asarray(item['targets']['SAHZU']['test']);seed=item['seed']
    positions=[]
    for j,fold in enumerate(r.folds(a,d,seed)):
        positions.append(dict(A=fold['T'],B=fold['V'],inner=r.folds(fold['T'],d,seed+101+j)))
    positions.append(dict(A=a,B=b,inner=r.folds(a,d,seed+201)))
    sp=dict(pool=a,test=b,positions=positions);r.write(dest/'SPLIT.json',sp);fixed=[];strict=[]
    for arm in f.ARMS:
        x=f.collect(d,sp,arm,seed)
        picks={k:f.choose(x['ds'][k],x['dp'],x['dy']) for k in f.RULES}
        np.savez_compressed(dest/(arm+'.npz'),test=b,training=a,prediction=x['pred'],**{'action_'+k.replace(' ','_'):v for k,v in x['actions'].items()},**{'score_'+k:v for k,v in x['scores'].items()})
        r.write(dest/(arm+'_selection.json'),dict(picks=picks,original=x['audit']))
        for rule,action in x['actions'].items():
            v=r.metrics(x['pred'],d['medical'][b,2],action)
            fixed.append(dict(rep=item['rep'],arm=arm,rule=rule,test=v,joint_test_pass=f.passed(v)))
        for rule,pick in picks.items():
            v=r.metrics(x['pred'],d['medical'][b,2],f.apply(x,rule,pick)) if pick['feasible'] else None
            strict.append(dict(rep=item['rep'],arm=arm,rule=rule,development=pick,test=v,joint_test_pass=f.passed(v)))
    r.write(dest/'FIXED.json',fixed);r.write(dest/'STRICT.json',strict);r.write(dest/'COMPLETE.json',dict(rep=item['rep'],seconds=time.time()-start))
    print('FULL60',item['rep'],round(time.time()-start,1),flush=True)

def aggregate():
    for key in ['FIXED','STRICT']:
        rr=[x for p in sorted((O/'private').glob('round_*/COMPLETE.json')) for x in json.loads((p.parent/(key+'.json')).read_text(encoding='utf8'))]
        r.write(O/(key+'_ROUNDS.json'),rr);r.write(O/(key+'_SUMMARY.json'),f.summarize(rr,['arm','rule']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','aggregate']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=100);args=ap.parse_args()
    if args.stage=='aggregate':aggregate();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    proto=dict(splits_sha256=r.sha(r.OUT/'private/SPLITS.json'),repeats=100,source_fraction=.9,test_fraction=.9,local_development=60,cost=4,alpha=.01,
        purpose='Separate loss of development examples from source/test subsampling; retain all60, never selectively retain a favourable record.',
        folds='Recreate3outer/3inner development folds for each same repeat seed; all rules fit only development predictions.',
        scope='Distinct sensitivity control; no local-training-sample uncertainty, cannot substitute for the54-development protocol.',
        strict='Also show developer-selected zero-FN thresholds with0/1/2mm normal guard, independently from original15%error threshold.')
    path=O/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==proto
    else:r.write(path,proto)
    for item in items[args.start:args.stop]:run_one(item,d,plan,role)
    aggregate()
    inp=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes']
    assert all(r.sha(r.current.ROOT/'data'/p)==h for p,h in inp.items())

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
