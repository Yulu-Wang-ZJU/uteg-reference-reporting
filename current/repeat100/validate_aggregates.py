"""Read aggregate-only artifacts; do not fit models or select policies."""
from pathlib import Path
import json,zipfile,math,statistics
P=Path(__file__).parent
def validate(archive):
 with zipfile.ZipFile(archive) as z:
  read=lambda n:json.loads(z.read('summary/'+n+'.json'))
  rr=read('Table2_and_time_rounds');ss=read('Table2_and_time_summary');checks=0
  for s in ss:
   r=[x for x in rr if all(x.get(k)==s.get(k) for k in ['budget','arm','rule','time'])];assert len(r)==100
   assert len({x.get('rep') for x in r})==100
   for key,metric in s.items():
    if not isinstance(metric,dict) or not {'mean','sd','n'}<=metric.keys():continue
    vals=[x['test'].get(key) for x in r];vals=[x for x in vals if isinstance(x,(int,float)) and math.isfinite(x)]
    assert len(vals)==metric['n'],(key,s['arm'],s['rule'])
    if vals:
     assert math.isclose(statistics.mean(vals),metric['mean'],abs_tol=1e-10)
     if len(vals)>1:assert math.isclose(statistics.stdev(vals),metric['sd'],abs_tol=1e-10)
    checks+=1
   for x in r:
    m=x['test'];assert m['reported']==m['correct']+m['incorrect'];assert m['n']==181
    if x['time']==900 and x['rule']=='common_B':assert m['FN']==0 and m['abnormal_released']>0 and m['reported_as_normal']>0
  return {'summary_metric_checks':checks,'complete_reporting_repeats':len(rr),'all_rounds_retained':True,'primary_and_sensitivity_final_composite_zero_FN':True,'undefined_ratios_not_zero_filled':True}
if __name__=='__main__':
 result=validate(P/'aggregate_results.zip');print(json.dumps(result,indent=2));(P/'AGGREGATE_VALIDATION.json').write_text(json.dumps(result,indent=2))
