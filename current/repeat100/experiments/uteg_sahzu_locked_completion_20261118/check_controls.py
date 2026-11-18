"""Read/recompute checks of control isolation; no model selection."""
import complete as m
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
e=m.e;r=m.r;c=m.c;O=m.O
with threadpool_limits(limits=1):
    d=m.setup();models=r.suite.numerical_models();item=m.INDEX[0,60];m.q.f.activate(item,d)
    sp=item['targets']['SAHZU'];tr=np.array(sp['pool']);te=np.array(sp['test']);checks=[]
    for arm in m.q.ARMS[1:]:
        for method in r.mapping.METHODS:
            model=joblib.load(O/f'table1/round_00/{arm}_{method}_source.joblib')
            p,extra=e.correction(d,models,model,tr,te)
            # Poison every non-training label; only the development correction is refitted.
            dd=dict(d,medical=d['medical'].copy(),y=d['y'].copy());bad=np.setdiff1d(np.arange(len(d['x'])),tr);dd['medical'][bad]=9999;dd['y'][bad]=9999
            pp,_=e.correction(dd,models,model,tr,te);np.testing.assert_array_equal(p,pp)
            for budget in [54,60]:
                z=m.load(O/f'table1/round_00/{arm}_{method}_dev{budget}.npz');assert np.isfinite(z['prediction']).all();np.testing.assert_array_equal(z['test'],m.INDEX[0,budget]['targets']['SAHZU']['test'])
            checks.append(dict(arm=arm,method=method,nondevelopment_label_poison_pass=True))
    # All100 final Hybrid MA arrays must equal the locked primary mapping.
    count=0
    for rep in range(100):
        for budget in [54,60]:
            for arm in m.q.ARMS[1:]:
                z=m.load(O/f'table1/round_{rep:02d}/{arm}_calibration_dev{budget}.npz');base=m.load(m.P/f'private/dev{budget}/round_{rep:03d}/{arm}.npz')
                np.testing.assert_array_equal(z['test'],base['test']);np.testing.assert_array_equal(z['Asymmetric_correction'][:,2],base['prediction']);count+=1
    r.write(O/'CONTROL_AUDIT.json',dict(passed=True,poison_checks=checks,hybrid_primary_exact_matches=count,fixed_source_weights_reused=True))
    print('CONTROL AUDIT PASS',len(checks),count)
