"""Independent integer-count replay; no fitting, threshold selection or score editing."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent;OUT=P.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=read(P/'PREFLIGHT.json');result=read(P/'RESULTS.json')
assert all(sha(k)==v for k,v in pre['hashes'].items())
freeze=read(P/'ALL_MODELS_FROZEN.json');assert all(sha(k)==v for k,v in freeze['models'].items())
actfreeze=read(P/'ACTIONS_FROZEN.json');assert all(sha(k)==v for k,v in actfreeze['sources'].items())
fit={};status={}
for h in [300,600]:
    entries=[read(p) for p in (P/f'T{h}'/'ledger').glob('*.json')]
    fit[str(h)]=len(entries);status[str(h)]=dict(Counter(e['status'] for e in entries))
assert sum(fit.values())<=pre['new_fit_upper']<=pre['hard_cap']
D=np.load(OUT/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz')
report={};checks=0
for d,zr in result.items():
    seq=[np.load(P/'actions'/f'T{h}'/f'{d}.npz') for h in [300,600]]
    old=np.load(OUT/'uteg_ridge_dual_head_external_eval_20261026_191117/actions'/f'{d}.npz')
    assert np.array_equal(seq[0]['ids'],seq[1]['ids']) and np.array_equal(seq[0]['ids'],old['test_ids'])
    yy=D['medical'][seq[0]['index'],2];y=np.select([yy<50,yy>70],[-1,1],default=0)
    decisions=[x['action'] for x in seq]+[old['unified_action']]
    labels=[x['pred'] for x in seq]+[old['final_ma_class']]
    N=len(y);first=np.full((20,N),99);times=np.full((20,N),-1)
    for ti,h in enumerate([300,600,900]):
        for seed in range(20):
            for i in range(N):
                if times[seed,i]<0 and decisions[ti][seed,i]:first[seed,i]=labels[ti][seed,i];times[seed,i]=h
        cnt=Counter()
        for seed in range(20):
            for i in range(N):
                if times[seed,i]<0:cnt['unreported']+=1;continue
                cnt['released']+=1
                if first[seed,i]==y[i]:cnt['correct']+=1
                else:
                    cnt['wrong']+=1
                    if y[i]==-1 and first[seed,i]==0:cnt['low_to_within']+=1
                    if y[i]==1 and first[seed,i]==0:cnt['high_to_within']+=1
                    if y[i]==0:cnt['normal_to_tail']+=1
                    if y[i]*first[seed,i]==-1:cnt['low_high_cross']+=1
        expected=zr['cumulative'][str(h)]
        for k in ['released','unreported','correct','wrong','low_to_within','high_to_within','normal_to_tail','low_high_cross']:
            assert cnt[k]==expected[k],(d,h,k,cnt[k],expected[k]);checks+=1
        assert cnt['released']+cnt['unreported']==20*N
        assert cnt['wrong']==cnt['low_to_within']+cnt['high_to_within']+cnt['normal_to_tail']+cnt['low_high_cross']
    saved=np.load(P/'actions'/f'{d}_first_report.npz')
    assert np.array_equal(times,saved['first_time'])
    assert np.array_equal(first[times>=0],saved['first_class'][times>=0])
    report[d]={'records':N,'groups':len(set(seq[0]['groups'])),'low_records':int(sum(y==-1)),'high_records':int(sum(y==1)),
               'first_report_locked':True,'all_seed_counts_verified':True}
summary={'status':'PASS','independent_integer_checks':checks,'fit_units':fit,'fit_total':sum(fit.values()),'fit_status':status,
         'unchanged_historical_dependencies':True,'freeze_hashes_verified':True,'T900_replay':read(P/'T900_REPLAY_CHECK.json'),
         'directions':report,'test_threshold_tuning':False,'replay_fix':'T900 checkpoints use compact filenames and wrapper payloads; corrected lookup before new test scoring; no refit.'}
(P/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
