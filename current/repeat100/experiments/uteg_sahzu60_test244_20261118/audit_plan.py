"""Read-only split audit and proposed roles; no model fitting or active-data edits."""
from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
import hashlib
import json
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parent
BASE = Path(str(_repeat_root / 'project' / 'outputs/current_ma_20261103/data'))

def main():
    with np.load(BASE/'DATA15_CURRENT.npz') as z:
        d = {k:z[k].copy() for k in z.files}
    role = json.loads((BASE/'ROLE_FREEZE.json').read_text())
    pool = np.sort(role['roles']['SAHZU']['pool_full'])
    allrows = np.flatnonzero(d['centers']=='SAHZU')
    test = np.setdiff1d(allrows,pool)
    labels = np.where(d['medical'][:,2]<50,0,np.where(d['medical'][:,2]>70,2,1))
    counts = lambda ix: np.bincount(labels[ix],minlength=3).tolist()
    overlap = sorted(set(d['groups'][pool])&set(d['groups'][test]))
    entries = []
    for g in overlap:
        entries.append(dict(group=str(g),records=[dict(index=int(i),record=str(d['ids'][i]),
            role='development' if i in pool else 'test') for i in np.flatnonzero(d['groups']==g)]))
    # Candidate only: move each crossed group's development members to test,
    # replace with singleton records of the same MA class, in a fixed hash order.
    # No prediction, continuous MA value or outcome metric enters this choice.
    remove = [int(i) for i in pool if d['groups'][i] in overlap]
    _,inv,size = np.unique(d['groups'][allrows],return_inverse=True,return_counts=True)
    singleton = set(allrows[size[inv]==1].tolist())
    candidates = [int(i) for i in test if i in singleton]
    add = []
    for old in remove:
        available = [i for i in candidates if labels[i]==labels[old] and i not in add]
        available.sort(key=lambda i: hashlib.sha256(('sahzu60_test244_v1|'+str(d['ids'][i])).encode()).hexdigest())
        add.append(available[0])
    newpool = np.sort(np.r_[np.setdiff1d(pool,remove),add]).astype(int)
    newtest = np.setdiff1d(allrows,newpool)
    assert len(newpool)==60 and len(newtest)==244
    assert counts(newpool)==counts(pool) and counts(newtest)==counts(test)
    assert not set(d['groups'][newpool])&set(d['groups'][newtest])
    report = dict(status='pending_role_decision_no_models_run',
        original=dict(development_n=len(pool),test_n=len(test),development_classes=counts(pool),test_classes=counts(test)),
        intended_per_repeat=dict(development_n=54,test_n=220,development_classes=[4,43,7],test_classes=[7,187,26]),
        repeats=100,overlapping_groups=entries,
        source_sampling='Reuse original100 source draws: Huzhou86/95; Hubei71/79. Other-centre target results untouched.',
        estimator='Existing B-input RBF with locked common_B policy; no model/hyperparameter search.',
        candidate=dict(status='proposal_only_not_active',development=newpool.tolist(),test=newtest.tolist(),
            removed=[str(d['ids'][i]) for i in remove],added=[str(d['ids'][i]) for i in add],
            selection='Same MA classes; deterministic record-ID hash among singleton groups; no prediction-based choice.',
            group_disjoint=True),
        input_sha256=hashlib.sha256((BASE/'DATA15_CURRENT.npz').read_bytes()).hexdigest())
    (O/'SPLIT_AUDIT_AND_PROPOSAL.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','original','intended_per_repeat','overlapping_groups']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
