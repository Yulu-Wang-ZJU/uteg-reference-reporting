"""Local-only asymmetric-cost sensitivity after auxiliary-feature comparison."""
import aux_followup as a
import run as r
import fn_candidates as f
import support_followup as s
import numpy as np,json,argparse
from threadpoolctl import threadpool_limits
O=a.O/'local_cost_controls'
CONFIGS=[('B_all_weak_c16',[0,2,3],.25),('B_all_equal_c16',[0,2,3],1.),('B_all_equal_c64',[0,2,3],1.)]

def setup():
    a.O=O;a.CONFIGS=CONFIGS;a.NAMES=[x[0] for x in CONFIGS];a.LOCAL_COSTS={a.NAMES[0]:16.,a.NAMES[1]:16.,a.NAMES[2]:64.}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','run','aggregate']);ap.add_argument('--budget',type=int,choices=[54,60],default=54);args=ap.parse_args();setup()
    if args.stage=='aggregate':a.summarize();return
    d,plan,role,m=r.init();items=json.loads((r.OUT/'private/SPLITS.json').read_text(encoding='utf8'))
    proto=dict(configs=CONFIGS,local_costs=a.LOCAL_COSTS,source_cost=4,anchor_cost=4,alpha=.01,repeats=100,budgets=[54,60],split_sha256=r.sha(r.OUT/'private/SPLITS.json'),reason='Targeted follow-up: auxiliary input improves average error but increases false-normal predictions. Test whether stronger local inward-error cost retains numeric gain while reducing FN.',selection='Same development-only joint selection as aux_followup; all results retained separately. No change to source model, B-only tail bounds, test labels or manuscript.')
    file=O/'PROTOCOL.json'
    if file.exists():assert json.loads(file.read_text(encoding='utf8'))==r.safe(proto)
    else:r.write(file,proto)
    if args.stage=='smoke':
        item=items[0];sp=s.spec(item,54);f.draw(item,plan);name='B_all_equal_c16';a.LOCAL_COSTS[name]=4.;a.configure(name);x=f.collect(d,sp,'HZ_to_SA',item['seed'])
        z=np.load(r.OUT/'aux_followup/dev54/private/round_00/HZ_to_SA_B_all_equal.npz');np.testing.assert_allclose(x['pred'],z['prediction'],atol=1e-9,rtol=0)
        a.LOCAL_COSTS[name]=16.;a.configure(name);x=f.collect(d,sp,'HZ_to_SA',item['seed']);picks={k:a.select_threshold(x['ds'][k],x['dp'],x['dy']) for k in a.RULES}
        truth=d['medical'][sp['test']].copy()
        try:
            d['medical'][sp['test']]=9999.;a.configure(name);v=f.collect(d,sp,'HZ_to_SA',item['seed'])
            np.testing.assert_allclose(x['pred'],v['pred'],atol=1e-9,rtol=0)
            for k in a.RULES:
                np.testing.assert_allclose(x['scores'][k],v['scores'][k],atol=1e-9,rtol=0);assert picks[k]==a.select_threshold(v['ds'][k],v['dp'],v['dy'])
        finally:d['medical'][sp['test']]=truth
        assert r.c.selected.COST==4
        r.write(O/'SMOKE.json',dict(cost4_reproduces_auxiliary_control=True,source_and_anchor_cost_restored4=True,test_label_poison_invariance=True));print('LOCAL COST SMOKE PASS',flush=True)
    else:
        assert (O/'SMOKE.json').exists()
        for item in items:a.main_round(item,args.budget,d,plan)
    inputs=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf8'))['input_hashes'];assert all(r.sha(r.current.ROOT/'data'/p)==h for p,h in inputs.items())
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
