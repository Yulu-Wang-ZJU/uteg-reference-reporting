"""Recompute fixed prediction/gate recipes and replay; no hyperparameter search.

Private arrays are checked in place. Only aggregate numerical differences leave
this check. Recomputing deterministic fitted coefficients is explicitly recorded.
"""
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from . import core as c,current_suite as suite,analysis,gates

OUT=suite.ROOT/'manuscript_sync/experiments/forward'

def run():
    suite.initialize();OUT.mkdir(parents=True,exist_ok=True);d,*_=c.context();rows=[]
    for arm in c.MAIN:
        for h in c.HORIZONS:
            z=analysis.load(suite.ROOT/'primary'/f'{arm}_T{h}_B0.npz')
            g=analysis.load(suite.ROOT/'gates'/f'{arm}_T{h}.npz')
            th=c.read(suite.ROOT/'gates'/f'{arm}_T{h}.thresholds.json')
            for si,seed in enumerate(c.SEEDS):
                path=OUT/f'{arm}_{h}_{seed}.json'
                if path.exists():rows.append(c.read(path));continue
                sp=c.paper_spec(arm,seed);o=c.evaluate(sp['pool'],sp['test'],sp['positions'][3]['inner'],arm,h)
                np.testing.assert_allclose(o['prediction'],z['prediction'][si],rtol=1e-10,atol=1e-9,equal_nan=True)
                np.testing.assert_allclose(o['inner'],z['inner'][si],rtol=1e-10,atol=1e-9,equal_nan=True)
                np.testing.assert_array_equal(o['accepted'],z['accepted'][si])
                model=joblib.load(suite.ROOT/'gates/models'/f'{arm}_T{h}_{seed}.joblib')
                x,med,f=gates.fields(o['query'],o['query_b'],o['prediction'],o['train'],h)
                pc=c.accounting.states(o['prediction']);valid=np.isfinite(o['prediction'])&c.qc_at(h)[o['query']]
                ss={'LR':gates.fr.score_ma_gate(model['lr'],f,pc),'HGB':gates.rc.score_hgb(model['hgb'],f,pc,valid),'DUAL':gates.fr.head_scores(model['heads'],x,med),'BOUNDARY':gates.rc.margin_scores(o['prediction'])}
                for k,v in ss.items():
                    v[~valid]=np.inf
                    np.testing.assert_allclose(v,g[k+'_score'][si],rtol=1e-10,atol=1e-9)
                for rule,key in [('LR','LR'),('HGB','HGB'),('BOUNDARY_PAPER','BOUNDARY')]:
                    t=th[si]['base'][key]['threshold'];a=np.zeros(len(pc),bool) if t is None else (ss[key]<=t)&np.isfinite(ss[key])&valid
                    np.testing.assert_array_equal(a,g[rule+'_action'][si])
                a=gates.fr.action(th[si]['dual'],pc,ss['LR'],ss['DUAL'],valid)
                np.testing.assert_array_equal(a,g['TWO_SIDED_action'][si])
                row=dict(arm=arm,minutes=h//60,seed=seed,records=len(pc),prediction_max_difference=float(np.nanmax(abs(o['prediction']-z['prediction'][si]))),actions_equal=True,scores_equal=True,recipe_refitted=True)
                gates.clean_write(path,row);rows.append(row)
            print('FORWARD_CHECK',arm,h,flush=True)
    gates.clean_write(OUT/'SUMMARY.json',dict(contexts=len(rows),rows=rows,recipe_refitted=True,raw_signal_refitted=False,hyperparameters_changed=False,input_lineage_unchanged=True))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
