"""Standalone saved-prediction/action replay. Requires only Python and NumPy."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(x):return np.where(x<50,0,np.where(x>70,2,1))
def gate(p,z,prefix):
    return np.ones(len(z[prefix+'prediction']),bool) if p['rule']=='All' else np.isfinite(z[prefix+'score_'+p['rule']])&(z[prefix+'score_'+p['rule']]<=p['threshold'])
def replay(p,z,prefix):
    pred=z[prefix+'prediction'];pc=state(pred);B=z[prefix+'B'];raw=z[prefix+'raw_risk'];normal=p['normal'];lim=normal['limits']
    out=(pc==1)&(raw[:,1]<=normal['cap_score'])&(-B<=lim['low'])&(B<=lim['high'])&gate(normal['gate'],z,prefix)
    for cls,key in [(0,'low_gate'),(2,'high_gate')]:out|=(pc==cls)&gate(p['abnormal'][key],z,prefix)
    return out
def metrics(pred,truth,act):
    pc=state(pred);tc=state(truth);nr=int(act.sum());ab=int((act&(tc!=1)).sum());nn=int((act&(tc==1)).sum());correct=int((act&(pc==tc)).sum());fn=int((act&(tc!=1)&(pc==1)).sum());fp=int((act&(tc==1)&(pc!=1)).sum())
    div=lambda x,y:x/y if y else None
    return dict(n=len(pred),reported=nr,correct=correct,incorrect=nr-correct,accuracy=div(correct,nr),reporting_rate=nr/len(pred),FN=fn,FNR=div(fn,ab),FP=fp,FPR=div(fp,nn),abnormal_released=ab,normal_released=nn,reported_as_normal=int((act&(pc==1)).sum()))
def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text());assert all(sha(ROOT/k)==v for k,v in manifest['files'].items())
    count=0
    for budget in [54,60]:
        rows=json.loads((ROOT/f'results/dev{budget}/ROUNDS.json').read_text());assert len(rows)==200
        assert {(v['rep'],v['arm']) for v in rows}=={(i,a) for i in range(100) for a in ['HZ_to_SA','HZ+HB_to_SA']}
        for row in rows:
            folder=ROOT/f'private/dev{budget}/round_{row["rep"]:02d}';p=json.loads((folder/'POLICIES.json').read_text())[row['arm']]
            assert row['policy']=='common_B' and p['feasible']
            with np.load(folder/(row['arm']+'.npz')) as z:
                assert len(z['test_prediction'])==181 and len(z['dev_prediction'])==budget
                assert not set(z['dev_groups'])&set(z['test_groups'])
                cap=float(np.quantile(z['dev_raw_risk'][state(z['dev_prediction'])==1,1],.9));assert cap==p['normal']['cap_score']
                for prefix,side in [('dev_','development'),('test_','test')]:
                    act=replay(p,z,prefix);np.testing.assert_array_equal(act,z[prefix+'action'])
                    computed=metrics(z[prefix+'prediction'],z[prefix+'truth'],act)
                    for k,v in computed.items():
                        old=row[side][k]
                        if v is None:assert old is None
                        else:assert np.isclose(v,old,rtol=1e-12,atol=1e-12),(budget,row['rep'],row['arm'],side,k,v,old)
                    assert computed['FN']==0 and computed['abnormal_released']>0 and computed['reported_as_normal']>0
                    count+=1
        for summary in json.loads((ROOT/f'results/dev{budget}/SUMMARY.json').read_text()):
            rr=[x for x in rows if x['arm']==summary['arm']]
            for key in ['accuracy','reporting_rate','FNR','FPR']:
                vals=np.array([x['test'][key] for x in rr]);expected=summary[key]
                for stat,v in [('mean',vals.mean()),('sd',vals.std(ddof=1)),('min',vals.min()),('max',vals.max())]:assert np.isclose(v,expected[stat],atol=1e-12,rtol=1e-12)
    papers=json.loads((ROOT/'MANUSCRIPT_BASELINE_HASHES.json').read_text());assert all(sha(Path(k))==v for k,v in papers.items())
    report=dict(status='PASS',policy='common_B',replayed_evaluations=count,all_400_test_runs_zero_FN=True,original10_cap_verified=True,all_100_repeats_each_setting=True,summary_recomputed=True,manifest_verified=True,manuscript_files_unchanged=True,scope='Replay saved predictions and fixed policies; not a fresh training run.')
    (ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
