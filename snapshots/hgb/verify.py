import csv,json
from pathlib import Path
from collections import defaultdict
p=Path(__file__).resolve().parent
groups=defaultdict(list)
with (p/'record_actions.csv').open() as f:
    for r in csv.DictReader(f): groups[r['direction'],r['method']].append(r)
with (p/'summary.csv').open() as f: summary=list(csv.DictReader(f))
checks=[]
for s in summary:
    rows=groups[s['direction'],s['method']]
    assert len(rows)==4020 and len({r['record_id'] for r in rows})==201
    assert len({r['seed'] for r in rows})==20
    rel=[r for r in rows if r['release']=='1']
    counts=dict(released=len(rel),correct=sum(r['truth']==r['prediction'] for r in rel),
                low_within=sum(r['truth']=='-1' and r['prediction']=='0' for r in rel),
                high_within=sum(r['truth']=='1' and r['prediction']=='0' for r in rel),
                correct_low=sum(r['truth']==r['prediction']=='-1' for r in rel),
                correct_high=sum(r['truth']==r['prediction']=='1' for r in rel),
                low_support=sum(r['truth']=='-1' for r in rows),high_support=sum(r['truth']=='1' for r in rows))
    assert all(v==int(s[k]) for k,v in counts.items())
    assert abs(100*counts['correct']/len(rel)-float(s['agreement']))<1e-10
    assert abs(100*len(rel)/len(rows)-float(s['coverage']))<1e-10
    checks.append(dict(direction=s['direction'],method=s['method'],counts=counts))
(p/'INDEPENDENT_CHECK.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
print('PASS: all four rows independently reconstructed from record actions')
