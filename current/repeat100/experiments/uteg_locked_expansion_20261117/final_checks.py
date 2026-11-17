"""Scope, isolation and saved numerical-result checks, without changing paper/data."""
import experiment as e
import numpy as np
import json,joblib
from threadpoolctl import threadpool_limits

def main():
    d,plan,role,models,items=e.init();item=items[0];e.activate(item,plan,e.O/'checks');sp=e.specification(item,54,role);te=np.asarray(sp['test']);tr=np.asarray(sp['pool']);arm='HZ_to_SA'
    first=e.collect(d,sp,arm,item['seed'],900)
    model=joblib.load(e.O/'table1/round_00/HZ_to_SA_RIDGE-10_source.joblib');p,_=e.correction(d,models,model,tr,te)
    saved={k:d[k].copy() for k in ['y','medical','ci']};e.r.c._SOURCE.clear()
    try:
        for k in ['y','medical','ci']:d[k][te]=123456
        second=e.collect(d,sp,arm,item['seed'],900);q,_=e.correction(d,models,model,tr,te)
        for key in first:np.testing.assert_allclose(first[key],second[key],atol=1e-10,rtol=0)
        np.testing.assert_allclose(p,q,atol=1e-10,rtol=0)
    finally:
        for k,v in saved.items():d[k][:]=v
        e.r.c._SOURCE.clear()
    max_error=0.;checked=0;states=[]
    for folder in e.O.glob('table1/round_*'):
        if not (folder/'COMPLETE.json').exists():continue
        rows=json.loads((folder/'RESULTS.json').read_text())
        for row in rows:
            arm=row['arm'];budget=row['budget'];method=row['method'];panel=row['panel']
            file=folder/f'{arm}_{method}_dev{budget}.npz' if panel=='mapping' and method!='Hybrid regression' else folder/f'{arm}_calibration_dev{budget}.npz'
            with np.load(file) as z:
                key='prediction' if panel=='mapping' and method!='Hybrid regression' else ('Asymmetric_correction' if method=='Hybrid regression' else method.replace(' ','_'))
                pp=z[key];yy=d['medical'][z['test']];m=e.r.four_metrics(pp,yy)
                for k,v in m.items():max_error=max(max_error,abs(v-row[k]));assert abs(v-row[k])<1e-10
                states.append(dict(rep=row['rep'],budget=budget,arm=arm,panel=panel,method=method,test=e.r.metrics(pp[:,2],yy[:,2],np.ones(len(yy),bool))))
                checked+=1
    import aggregate
    aggregate.write('Table1_MA_states',states,['budget','arm','panel','method'])
    e.r.write(e.O/'checks/ISOLATION_AND_TABLE1.json',dict(test_y_medical_ci_poison_no_effect=True,cleared_source_cache=True,primary_prediction_scores_unchanged=True,fair_source_correction_unchanged=True,table1_rows_recomputed=checked,max_table1_difference=max_error))
    print('CHECKS PASS',checked)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
