"""Lock the accepted60-record candidate and audit saved100-repeat evidence."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import json,hashlib,shutil,sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;F=O/'symmetric90_reselected';P=O.parent/'uteg_sahzu60_test244_20261118';L=O/'locked60_20261118'
WS=Path(str(_repeat_root / 'project' / ''))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
with np.load(WS/'outputs/current_ma_20261103/data/DATA15_CURRENT.npz') as z:d={k:z[k].copy() for k in z.files}
items=json.loads((P/'private/SPLITS.json').read_text());old=json.loads((WS/'outputs/resampling90_100_20261116/private/SPLITS.json').read_text())
rows=json.loads((F/'ROUNDS.json').read_text());policies=json.loads((F/'POLICIES.json').read_text());summ=json.loads((F/'SUMMARY.json').read_text())
arms=['target_only_SA','HZ_to_SA','HB_to_SA','HZ+HB_to_SA'];summary=[];counts=[];checks=0
def sep(a,b):
    assert not set(a)&set(b)
    assert not set(d['groups'][a])&set(d['groups'][b])
for budget in [60,54]:
    ii=[i for i in items if i['budget']==budget];assert len(ii)==100 and sorted(i['rep'] for i in ii)==list(range(100))
    for it in ii:
        sp=it['targets']['SAHZU'];train=np.array(sp['pool']);test=np.array(sp['test']);sep(train,test)
        assert len(train)==budget and len(test)==220
        assert it['sources']==old[it['rep']]['sources']
        observed=[]
        for j,pos in enumerate(sp['positions']):
            a=np.array(pos['A']);b=np.array(pos['B']);sep(a,b);sep(a,test)
            assert set(a)<=set(train)
            if j<3:observed+=b.tolist();assert set(a)|set(b)==set(train)
            else:assert set(a)==set(train) and set(b)==set(test)
            inner=[]
            for fold in pos['inner']:
                aa=np.array(fold['T']);bb=np.array(fold['V']);sep(aa,bb);assert set(aa)|set(bb)==set(a);inner+=bb.tolist();checks+=1
            assert sorted(inner)==sorted(a)
        assert sorted(observed)==sorted(train)
        with np.load(F/'private'/f'dev{budget}'/f'round_{it["rep"]:03d}.npz') as actions:
            for arm in arms:
                source=[] if arm.startswith('target_only') else sum([it['sources'][{'HZ':'Huzhou','HB':'Hubei'}[k]] for k in arm.split('_to_')[0].split('+')],[])
                sep(np.array(source,int),np.r_[train,test])
                folder=P/'private'/f'dev{budget}'/f'round_{it["rep"]:03d}'
                meta=json.loads((folder/(arm+'.json')).read_text())
                for ra in meta['risk_training']:
                    a=np.array(ra['train']);b=np.array(ra['query']);sep(a,b);assert set(a)<=set(train);sep(a,test)
                with np.load(folder/(arm+'.npz')) as z:
                    np.testing.assert_array_equal(z['test'],test);assert set(z['dev_rows'])==set(train)
                    np.testing.assert_array_equal(actions[arm+'_test'],test)
                    pick=next(x['policy'] for x in policies if (x['budget'],x['rep'],x['arm'])==(budget,it['rep'],arm))
                    dp=z['dev_prediction'];norm=(dp>=50)&(dp<=70)
                    assert pick['low_guard']==float(np.quantile(z['dev_risk'][norm,0],.9))
                    assert pick['normal']['cap_score']==float(np.quantile(z['dev_risk'][norm,1],.9))
                    a=actions[arm+'_action'];pc=np.where(z['prediction']<50,0,np.where(z['prediction']>70,2,1));tc=np.where(z['reference']<50,0,np.where(z['reference']>70,2,1))
                    cm=np.array([[int(np.sum(a&(tc==i)&(pc==j))) for j in range(3)] for i in range(3)])
                    rr=next(x for x in rows if (x['budget'],x['rep'],x['arm'])==(budget,it['rep'],arm));m=rr['test']
                    assert cm.tolist()==m['confusion'];assert cm.sum()==m['reported'];assert cm.trace()==m['correct'];assert int(cm[0,1]+cm[2,1])==m['FN'];assert int(cm[1,0]+cm[1,2])==m['FP']
                    assert int(cm[:,1].sum())==m['reported_as_normal'] and cm[:,1].sum()>0
                    assert cm[[0,2],:].sum()>0
                    assert not np.any(a&(pc==1)&~z['action_common_B'])
                    if budget==60:assert m['FN']==0
                    assert rr['feasible'] and rr['development']['FN']==0 and rr['development']['accuracy']>.8
                    counts.append(dict(budget=budget,rep=it['rep'],arm=arm,normal_reports=int(cm[:,1].sum()),correct_normal=int(cm[1,1]),true_normal_total=int((tc==1).sum()),true_abnormal_released=int(cm[[0,2],:].sum()),correct_low=int(cm[0,0]),correct_high=int(cm[2,2]),reported=int(cm.sum()),FN=m['FN']))
    for arm in arms:
        rr=[x for x in rows if x['budget']==budget and x['arm']==arm];v=next(x for x in summ if x['budget']==budget and x['arm']==arm);a=[x for x in counts if x['budget']==budget and x['arm']==arm]
        out=dict(budget=budget,arm=arm,repeats=100)
        for key in ['accuracy','reporting_rate','FNR','FPR']:
            vals=np.array([x['test'][key] for x in rr]);assert np.isfinite(vals).all()
            np.testing.assert_allclose(vals.mean(),v[key+'_mean'],atol=1e-14,rtol=0);np.testing.assert_allclose(vals.std(ddof=1),v[key+'_sd'],atol=1e-14,rtol=0)
            out[key+'_mean']=float(vals.mean());out[key+'_sd']=float(vals.std(ddof=1))
        for key in ['normal_reports','correct_normal','true_abnormal_released','correct_low','correct_high','reported']:
            vals=np.array([x[key] for x in a]);out[key]=dict(mean=float(vals.mean()),min=int(vals.min()),max=int(vals.max()))
        out['FN_events']=sum(x['FN'] for x in a);summary.append(out)
# Protect manuscript and all previous evidence; lock references plus selected actions.
protected=json.loads((O/'PROTECTED_HASHES.json').read_text());assert all(sha(Path(p))==h for p,h in protected.items())
L.mkdir(exist_ok=True)
write(L/'SELECTED_POLICY.json',dict(status='locked_at_user_request',date='2026-11-18',selected='symmetric90_with_original_development_abnormal_gate_reselection',
    scope='SAHZU target only, fixed approved60/244 roles,60development/220sampledtest,100repeats,15minutes,fourlearningconfigurations',
    changed='Append low-MA risk90th-percentile veto to existing normal checks; use unchanged abnormal gate selector to retain developmentFN0/accuracy>80 requirements.',
    fixed='No further parameter tuning. Mapping/loss/source sampling unchanged. Normal predictions are withheld, not relabelled.',
    author_answer='先固定这套60条的方案，不再调了',
    sensitivity54=str(F/'ROUNDS.json'),sensitivity54_status='Retained separately; not promoted to zeroFN primary evidence.',
    model_predictions=str(P/'private'),parameters=str(F/'POLICIES.json'),input_data_sha256=sha(WS/'outputs/current_ma_20261103/data/DATA15_CURRENT.npz'),
    paper_modified=False,no_external_writes=True))
write(L/'ROUNDS60.json',[x for x in rows if x['budget']==60]);write(L/'POLICIES60.json',[x for x in policies if x['budget']==60]);write(L/'SPLITS60.json',[x for x in items if x['budget']==60]);write(L/'NORMAL_REPORT_AUDIT.json',summary)
write(L/'AUDIT.json',dict(passed=True,nested_fold_checks=checks,workflow_count_checks=len(counts),original_source_draws_identical=True,groups_disjoint=True,all100_kept=True,
    all_primary_rounds_FN0=True,all_rounds_nonempty_normal_and_abnormal_reports=True,all_development_FN0_accuracy_gt80=True,
    metric_means_and_sampleSD_recomputed=True,existing_label_poison_smoke_checks=json.loads((F/'CHECKS.json').read_text()),protected_files_unchanged=True))
for src in (F/'private/dev60').glob('*.npz'):
    dst=L/'actions60'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
codepaths=[O/'study.py',O/'reselect.py',O/'audit_lock.py',P/'run.py']
for src in codepaths:
    dst=L/'code'/src.parent.name/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
manifest={str(p):sha(p) for p in L.rglob('*') if p.is_file() and p.name!='FILE_MANIFEST.json'}
for folder in [P/'private/dev60',F/'private/dev54']:
    manifest.update({str(p):sha(p) for p in folder.rglob('*') if p.is_file()})
manifest.update({str(F/n):sha(F/n) for n in ['ROUNDS.json','POLICIES.json','SUMMARY.json','PROTOCOL.json']})
write(L/'FILE_MANIFEST.json',manifest)
print(json.dumps([x for x in summary if x['budget']==60],ensure_ascii=False,indent=2))
