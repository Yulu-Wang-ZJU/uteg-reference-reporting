from pathlib import Path
import zipfile,json,numpy as np
with zipfile.ZipFile(Path(__file__).parent/'aggregate_results.zip') as z:
 rows=json.loads(z.read('summary/Table2_and_time_rounds.json'));summary=json.loads(z.read('summary/Table2_and_time_summary.json'))
 for r in summary:
  rr=[x for x in rows if all(x[k]==r[k] for k in ['budget','arm','time','rule'])]
  assert len(rr)==100
  for k in ['accuracy','reporting_rate','FNR','FPR']:
   v=[x['test'][k] for x in rr if x['test'][k] is not None];m=r[k]
   assert len(v)==m['n']
   if v:assert np.isclose(np.mean(v),m['mean']) and np.isclose(np.std(v,ddof=1),m['sd'])
  if r['budget']==60 and r['time']==900 and r['rule']=='common_B':
   assert all(x['test']['FN']==0 and x['test']['reported_as_normal']>0 for x in rr)
 print('All current reporting summaries match 100 saved repeats; full60 15-minute zero FN includes normal releases.')
