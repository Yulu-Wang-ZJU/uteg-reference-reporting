"""Fixed-policy component removal and inference-overhead audit; no model fitting."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import csv, hashlib, importlib.util, json, platform, time
from pathlib import Path
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).parent
SOURCE = Path(str(_repeat_root / 'experiments' / 'uteg_locked_commonB_20261117'))
spec = importlib.util.spec_from_file_location('locked_replay', SOURCE/'verify_locked.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
VARIANTS = ['full', 'without_risk_cap', 'without_amplitude_bounds', 'without_both_normal_checks', 'all_report']

def write(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def csvwrite(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def action(p, z, prefix, variant):
    pred = z[prefix+'prediction']; pc = v.state(pred)
    if variant == 'all_report': return np.ones(len(pred), bool)
    n = p['normal']; B = z[prefix+'B']; raw = z[prefix+'raw_risk']
    mask = (pc == 1) & v.gate(n['gate'], z, prefix)
    if variant not in ['without_risk_cap', 'without_both_normal_checks']:
        mask &= raw[:, 1] <= n['cap_score']
    if variant not in ['without_amplitude_bounds', 'without_both_normal_checks']:
        mask &= (-B <= n['limits']['low']) & (B <= n['limits']['high'])
    for cls, key in [(0, 'low_gate'), (2, 'high_gate')]:
        mask |= (pc == cls) & v.gate(p['abnormal'][key], z, prefix)
    return mask

def stats(x):
    x = np.array([a for a in x if a is not None], float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else 0.,
                min=float(x.min()), max=float(x.max()), n=len(x)) if len(x) else None

def timer(fn):
    for _ in range(20): fn()
    values = []
    for _ in range(100):
        t = time.perf_counter_ns(); fn(); values.append((time.perf_counter_ns()-t)/1e6)
    return values

def main():
    protocol = dict(scope='15-minute fixed-policy removal, HZ-to-SA and HZ+HB-to-SA, 54/60 development records, all 100 repeats',
      variants=VARIANTS, no_retraining=True, no_threshold_reselection=True,
      retained='All learned normal base-gate and abnormal-branch thresholds remain fixed in every removal variant.',
      FNR='Released abnormal predicted normal / all released reference-abnormal records; undefined denominators remain null.',
      FPR='Released normal predicted abnormal / all released reference-normal records.',
      uncertainty='Mean and sample SD across overlapping subsamples, not confidence intervals or independent trials.',
      timing='100 warm single-query calls after 20 warm-ups per component and selected case. Risk-head inference reconstructed from saved coefficients and checked against saved scores. Decision timing uses cached HGB/logistic/boundary scores; excludes those scorers, MA mapping, acquisition, loading and training. These are component overheads, not complete clinical latency.',
      source_policy_hash=v.sha(SOURCE/'SELECTED_POLICY.json'))
    write('PROTOCOL.json', protocol)
    rows=[]; paired=[]; times=[]; hashes={}; risk_maxerr=0.; action_checks=0
    for budget in [54,60]:
        for rep in range(100):
            folder=SOURCE/f'private/dev{budget}/round_{rep:02d}'
            policies=json.loads((folder/'POLICIES.json').read_text())
            audit=json.loads((folder/'RISK_TRAINING.json').read_text())['B'][-1]
            mean=np.array(audit['mean']); scale=np.array(audit['scale'])
            coef=np.array([h['coef'][0] for h in audit['heads']]); intercept=np.array([h['intercept'][0] for h in audit['heads']])
            def risk(B): return expit(((np.asarray(B).reshape(-1,1)-mean)/scale)@coef.T+intercept)
            for file in [folder/'POLICIES.json',folder/'RISK_TRAINING.json']:
                hashes[str(file)]=v.sha(file)
            for arm in ['HZ_to_SA','HZ+HB_to_SA']:
                path=folder/(arm+'.npz'); hashes[str(path)]=v.sha(path)
                with np.load(path) as f: z={k:f[k] for k in f.files}
                assert not set(z['dev_groups']) & set(z['test_groups'])
                p=policies[arm]; full=action(p,z,'test_','full')
                np.testing.assert_array_equal(full,z['test_action'])
                np.testing.assert_array_equal(action(p,z,'dev_','full'),z['dev_action']); action_checks+=2
                rebuilt=risk(z['test_B']); err=float(np.max(np.abs(rebuilt-z['test_raw_risk'])))
                risk_maxerr=max(err,risk_maxerr); np.testing.assert_allclose(rebuilt,z['test_raw_risk'],rtol=1e-10,atol=1e-12)
                pred=z['test_prediction']; truth=z['test_truth']; pc=v.state(pred); tc=v.state(truth)
                for variant in VARIANTS:
                    mask=action(p,z,'test_',variant); assert np.all(~full | mask)
                    m=v.metrics(pred,truth,mask)
                    rows.append(dict(budget=budget,rep=rep,arm=arm,variant=variant,**m,
                        correct_withheld=int((~mask&(pc==tc)).sum()),incorrect_withheld=int((~mask&(pc!=tc)).sum())))
                    added=mask&~full
                    paired.append(dict(budget=budget,rep=rep,arm=arm,variant=variant,
                        additional_reports=int(added.sum()),additional_correct=int((added&(pc==tc)).sum()),
                        additional_FN=int((added&(tc!=1)&(pc==1)).sum())))
                # First record of each predicted class: deterministic, independent of correctness.
                for cls in [0,1,2]:
                    ix=np.flatnonzero(pc==cls)
                    if not len(ix): continue
                    i=int(ix[0]); one={k:val[i:i+1] for k,val in z.items() if k.startswith('test_')}
                    fns={'two_risk_heads':lambda:risk(one['test_B']),
                         'fixed_branch_decision_cached_scores':lambda:action(p,one,'test_','full')}
                    for component,fn in fns.items():
                        values=timer(fn)
                        times.extend(dict(budget=budget,rep=rep,arm=arm,predicted_class=cls,component=component,call=j,ms=x) for j,x in enumerate(values))
    summary=[]
    for budget in [54,60]:
        for arm in ['HZ_to_SA','HZ+HB_to_SA']:
            for variant in VARIANTS:
                rr=[x for x in rows if (x['budget'],x['arm'],x['variant'])==(budget,arm,variant)]
                assert len(rr)==100 and {x['rep'] for x in rr}==set(range(100))
                summary.append(dict(budget=budget,arm=arm,variant=variant,rounds=100,
                    zero_FN_rounds=sum(x['FN']==0 for x in rr),FN_events=sum(x['FN'] for x in rr),
                    **{k:stats([x[k] for x in rr]) for k in ['accuracy','reporting_rate','FNR','FPR','reported','correct','incorrect','correct_withheld','incorrect_withheld']}))
    ts=[]
    for component in sorted({x['component'] for x in times}):
        vals=np.array([x['ms'] for x in times if x['component']==component])
        ts.append(dict(component=component,calls=len(vals),median_ms=float(np.median(vals)),p05_ms=float(np.quantile(vals,.05)),p95_ms=float(np.quantile(vals,.95)),mean_ms=float(vals.mean())))
    write('ROUNDS.json',rows); csvwrite('ROUNDS.csv',rows); csvwrite('PAIRED_COMPONENT_EFFECTS.csv',paired)
    write('SUMMARY.json',summary); csvwrite('TIMING_CALLS.csv',times)
    write('TIMING_SUMMARY.json',dict(environment=dict(platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),numpy=np.__version__),results=ts,scope=protocol['timing']))
    assert all(v.sha(Path(k))==h for k,h in hashes.items())
    papers=json.loads((SOURCE/'MANUSCRIPT_BASELINE_HASHES.json').read_text())
    assert all(v.sha(Path(k))==h for k,h in papers.items())
    write('INPUT_HASHES.json',hashes)
    write('CHECKS.json',dict(status='PASS',fixed_policy_replays=action_checks,all_100_repeats=True,
         risk_score_max_absolute_error=risk_maxerr,no_training_or_threshold_selection=True,
         input_files_unchanged=True,manuscript_files_unchanged=True,evaluations=len(rows)))
    lines=['# Fixed-policy component ablation','', 'No retraining or threshold reselection. All 100 repeats retained. Percentages are mean ± sample SD. FN totals are record–repeat events, not independent patients.','', '| Development | Arm | Component condition | Accuracy (%) | Reporting rate (%) | FNR (%) | FPR (%) | Zero-FN rounds | FN events |','|---|---|---|---|---|---|---|---|---|']
    for x in summary:
        vals=['{:.2f} ± {:.2f}'.format(100*x[k]['mean'],100*x[k]['sd']) for k in ['accuracy','reporting_rate','FNR','FPR']]
        lines.append('| '+ ' | '.join(map(str,[x['budget'],x['arm'],x['variant'],*vals,x['zero_FN_rounds'],x['FN_events']]))+' |')
    lines+=['','## Timing',protocol['timing'],'']
    for t in ts: lines.append(f"- {t['component']}: median {t['median_ms']:.6f} ms, 5th–95th percentile {t['p05_ms']:.6f}–{t['p95_ms']:.6f} ms ({t['calls']} calls).")
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='PASS',evaluations=len(rows),timing=ts)))

if __name__=='__main__': main()
