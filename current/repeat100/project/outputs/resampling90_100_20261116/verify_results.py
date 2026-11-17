"""Check saved outputs and selection, without retraining."""
import run as r
import fn_candidates as f
import json
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    d,plan,role,m=r.init();items=read(r.OUT/'private/SPLITS.json')
    assert len(items)==100 and items[:20]==read(r.ROOT/'outputs/resampling90_20_20261116/private/SPLITS.json')
    counts=dict(baseline_rounds=0,candidate_rounds=0,full60_rounds=0,candidate_predictions=0,selected_policies=0)
    for item in items:
        rep=item['rep'];base=r.OUT/'private'/f'round_{rep:02d}';folder=f.O/'private'/f'round_{rep:02d}';full=r.OUT/'full60_control/private'/f'round_{rep:02d}'
        assert all((p/'COMPLETE.json').exists() for p in [base,folder,full])
        counts['baseline_rounds']+=1;counts['candidate_rounds']+=1;counts['full60_rounds']+=1
        for ct,sp in item['targets'].items():
            assert set(sp['pool'])<=set(role['roles'][ct]['pool_full']) and set(sp['test'])<=set(role['roles'][ct]['test'])
            assert not set(d['groups'][sp['pool']])&set(d['groups'][sp['test']])
            for part in sp['positions']:
                assert not set(d['groups'][part['A']])&set(d['groups'][part['B']])
                assert set(sum((p['V'] for p in part['inner']),[]))==set(part['A'])
                for inner in part['inner']:
                    assert set(inner['T'])|set(inner['V'])==set(part['A'])
                    assert not set(d['groups'][inner['T']])&set(d['groups'][inner['V']])
        fixed=read(folder/'FIXED.json');selected=read(folder/'SELECTED.json');table2=read(base/'Table2.json')
        assert len(read(base/'Table1.json'))==24 and len(table2)==12 and len(fixed)==72
        sp=item['targets']['SAHZU'];y=d['medical'][sp['test'],2]
        for arm in f.ARMS:
            original=np.load(base/(arm+'_reporting.npz'))
            for rule in ['All report',*f.RULES]:
                expected=r.metrics(original['prediction'],y,original['action_'+rule.replace(' ','_')])
                actual=next(x for x in table2 if x['arm']==arm and x['rule']==rule)
                assert r.safe(expected)=={k:actual[k] for k in expected}
                a=next(x for x in fixed if x['arm']==arm and x['rule']==rule and x['kind']=='residual' and x['cost']==4)
                assert a['test']==r.safe(expected)
            picks=[];arrays=[]
            for kind,cost in f.CANDIDATES:
                name=f'{arm}_{kind}{cost}';z=np.load(folder/(name+'.npz'));arrays.append(z)
                np.testing.assert_array_equal(z['test'],sp['test']);assert np.isfinite(z['prediction']).all()
                assert set(z['dev_rows'])==set(sp['pool'])
                pick=read(folder/(name+'_selection.json'))['picks'];picks.append(pick)
                for rule in f.RULES:
                    assert pick[rule]==f.choose(z['dev_score_'+rule],z['dev_prediction'],d['medical'][z['dev_rows'],2])
                counts['candidate_predictions']+=1
                if kind=='residual' and cost==4:
                    np.testing.assert_allclose(z['prediction'],original['prediction'],atol=1e-9,rtol=0)
                    for rule in f.RULES:np.testing.assert_allclose(z['test_score_'+rule],original['score_'+rule],atol=1e-9,rtol=0)
            for rule in f.RULES:
                v=next(x for x in selected if x['arm']==arm and x['rule']==rule)
                valid=[i for i,p in enumerate(picks) if p[rule]['feasible']]
                if not valid:assert v['choice'] is None and v['test'] is None
                else:
                    i=max(valid,key=lambda i:(picks[i][rule]['reported'],picks[i][rule]['accuracy'],-i))
                    assert v['choice']==list(f.CANDIDATES[i])
                    z=arrays[i];q=dict(pred=z['prediction'],scores={rule:z['test_score_'+rule]})
                    assert v['test']==r.safe(r.metrics(z['prediction'],y,f.apply(q,rule,picks[i][rule])))
                counts['selected_policies']+=1
            z=np.load(full/(arm+'.npz'))
            np.testing.assert_array_equal(z['test'],sp['test']);assert set(z['training'])==set(role['roles']['SAHZU']['pool_full'])
            for v in read(full/'FIXED.json'):
                if v['arm']==arm:assert v['test']==r.safe(r.metrics(z['prediction'],y,z['action_'+v['rule'].replace(' ','_')]))
    proto=read(r.OUT/'PROTOCOL.json');assert all(r.sha(r.current.ROOT/'data'/p)==h for p,h in proto['input_hashes'].items())
    r.write(r.OUT/'FINAL_QA.json',dict(**counts,first20_preserved=True,same_source_test_draws=True,roles_groups_and_inner_folds_checked=True,original_cost4_reproduced_in_all100=True,selection_recomputed_from_development_only=True,test_metrics_and_denominators_checked=True,inputs_unchanged=True))
    print('QA PASS',counts,flush=True)

if __name__=='__main__':main()
