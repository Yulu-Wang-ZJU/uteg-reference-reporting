"""Verify the reused numerical path against one saved original repeat, read-only."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('sahzu244_runner',Path(__file__).with_name('run.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

with threadpool_limits(limits=1):
    d=q.init();old=json.loads(q.OLD.read_text())[0]
    role=json.loads((q.r.current.ROOT/'data/ROLE_FREEZE.json').read_text())
    folder=q.e.O/'nine/round_00';policies=json.loads((folder/'POLICIES.json').read_text());checks=[]
    for budget,label in [(54,'sampled90'),(60,'full_local')]:
        if budget==54:sp=old['targets']['SAHZU']
        else:
            pool=np.array(role['roles']['SAHZU']['pool_full']);test=np.array(old['targets']['SAHZU']['test'])
            sp=dict(pool=pool,test=test,positions=q.positions(d,pool,test,old['seed']))
        item=dict(old,targets={'SAHZU':sp})
        for arm in q.ARMS:
            q.f.activate(item,d);bundle=q.f.bundle(d,item,arm,900)
            z,dr,qr,_,_,picks,actions=bundle
            key=label+'_'+arm
            with np.load(folder/(key+'.npz')) as saved:
                for field in z:np.testing.assert_allclose(z[field],saved[field],rtol=0,atol=1e-9)
                np.testing.assert_allclose(dr,saved['dev_risk'],rtol=0,atol=1e-9)
                np.testing.assert_allclose(qr,saved['test_risk'],rtol=0,atol=1e-9)
                np.testing.assert_array_equal(actions['common_B'],saved['action_common_B'])
            assert q.r.safe(picks['common_B'])==q.r.safe(policies[key]['common_B'])
            checks.append(dict(budget=budget,arm=arm,saved_predictions_scores_policy_actions_reproduced=True))
    q.write(q.O/'ORIGINAL_REPLAY_CHECK.json',dict(passed=True,repeat=0,checks=checks))
    print('ORIGINAL REPLAY PASS',len(checks))
