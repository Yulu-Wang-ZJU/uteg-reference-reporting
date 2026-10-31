"""Read-only independent-formula checks; no fitting or threshold selection."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import joblib
from scipy.special import expit
R=Path(__file__).resolve().parent;G=R.parent;O=G.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((R/'FREEZE.json').read_text());result=json.loads((R/'RESULTS.json').read_text())
assert all(sha(p)==h for p,h in {**f['inputs'],**f['models'],**result['reference_actions_hashes']}.items())
D=dict(np.load(O/'uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz',allow_pickle=False))
rows=list(csv.DictReader((R/'per_seed.csv').open(encoding='utf-8-sig')))
nr=0
for seed in range(1000,1020):
    m=joblib.load(R/f'model_{seed}.joblib');I=m['test'];x=D['x'][I];p=m['predictor']
    y=p['ym']+(((x-p['xm'])/p['xs'])@p['coef'])*p['ys'];y[:,:3]=np.maximum(y[:,:3],0);y[:,3]=np.clip(y[:,3],0,89.999999)
    pred=np.where(y[:,2]<50,-1,np.where(y[:,2]>70,1,0));truth=np.where(D['medical'][I,2]<50,-1,np.where(D['medical'][I,2]>70,1,0))
    features=np.column_stack([x,(y-m['ym'])/m['ys'],(y[:,2]-60)/10,np.minimum(abs(y[:,2]-50),abs(y[:,2]-70))/10,pred==-1,pred==0,pred==1])
    gate=m['error'];pre=gate['pre'];z=np.clip((features-pre['mean'])/pre['sd'],-10,10)
    risk=np.repeat(gate['constant_score'],len(I)) if gate['fallback'] else expit(z@gate['coef']+gate['intercept'])
    tails=[]
    for name in ['low','high']:
        head=m['heads'][name]
        if head is None:tails.append(np.full(len(I),np.inf));continue
        pre=head['pre'];z=np.clip((np.column_stack([x[:,1],y[:,2]])-pre['mean'])/pre['sd'],-10,10)
        tails.append(expit(z@head['theta'][:-1]+head['theta'][-1]))
    dual=np.maximum(*tails);t=m['threshold'];a=np.zeros(len(I),bool)
    if not t['all_withhold']:
        if t['normal'] is not None:a|=(pred==0)&(dual<=t['normal'])
        if t['tail'] is not None:a|=(pred!=0)&(risk<=t['tail'])
    a&=D['input_qc'][I];good=a&(pred==truth);bad=a&~good
    actual=dict(correct=int(good.sum()),incorrect=int(bad.sum()),unreported=int((~a).sum()),coverage=int(a.sum()),low_to_within=int((a&(truth==-1)&(pred==0)).sum()),high_to_within=int((a&(truth==1)&(pred==0)).sum()),correct_low=int((a&(truth==-1)&(pred==-1)).sum()),correct_high=int((a&(truth==1)&(pred==1)).sum()))
    expected=next(r for r in rows if r['method']=='Target-only Ridge' and int(r['seed'])==seed)
    assert all(v==int(expected[k]) for k,v in actual.items());nr+=1
for name,summary in result['arms'].items():
    rr=[r for r in rows if r['method']==name];assert len(rr)==20
    for k,n in summary['counts'].items():assert sum(int(r[k]) for r in rr)==n
    assert summary['counts']['correct']+summary['counts']['incorrect']+summary['counts']['unreported']==4020
    assert summary['zero_report_seeds']==sum(int(r['coverage'])==0 for r in rr)
dest=list(csv.DictReader((R/'destinations.csv').open(encoding='utf-8-sig')))
for name in result['arms']:
    for state,den in [(-1,140),(0,3400),(1,480)]:
        rr=[r for r in dest if r['method']==name and int(r['true_state'])==state]
        assert len(rr)==4 and sum(int(r['n']) for r in rr)==den and all(int(r['N'])==den for r in rr)
out=dict(status='PASS',read_only=True,new_fits=0,independent_formula_target_runs=nr,per_seed_and_class_counts=True,frozen_hashes_unchanged=True,scope='Numerical and implementation cross-check by implementation agent, not independent peer review')
path=R/'NUMERICAL_CHECK.json';assert not path.exists();path.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
