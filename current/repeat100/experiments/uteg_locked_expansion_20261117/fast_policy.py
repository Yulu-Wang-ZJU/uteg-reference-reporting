"""Exact cumulative-count implementation of the locked rectangle search."""
import copy,json
import numpy as np
import experiment as e

def choose_normal(pred,truth,raw,risk,scores,cap):
    normal=e.r.state(pred)==1;tc=e.r.state(truth);lo=e.n.mid_grid(risk[normal,0]);hi=e.n.mid_grid(risk[normal,1]);best=None
    if not len(lo) or not len(hi):return dict(feasible=False,cap_score=cap)
    li=np.searchsorted(lo,risk[:,0]);hj=np.searchsorted(hi,risk[:,1]);shape=(len(lo),len(hi));eligible=normal&(raw[:,1]<=cap)&(li<len(lo))&(hj<len(hi))
    for gate in e.b.gate_options(scores,normal):
        ok=eligible&e.b.gate_action(gate,scores,len(pred));a=np.zeros(shape,np.int32);bad=np.zeros(shape,np.int32)
        np.add.at(a,(li[ok],hj[ok]),1);b=ok&(tc!=1);np.add.at(bad,(li[b],hj[b]),1)
        sizes=a.cumsum(0).cumsum(1).ravel()[::-1];bad=bad.cumsum(0).cumsum(1).ravel()[::-1];valid=(sizes>0)&(bad==0)
        if not valid.any():continue
        ii=np.flatnonzero(valid);i=int(ii[np.argmax(sizes[ii])]);count=int(sizes[i]);flat=len(sizes)-1-i;l,h=np.unravel_index(flat,shape)
        if best is None or count>best['reported']:best=dict(feasible=True,gate=gate,limits=dict(low=float(lo[l]),high=float(hi[h])),reported=count,cap_score=cap)
    if best is None:return dict(feasible=False,cap_score=cap)
    before=e.cal.normal_action(best,pred,raw,risk,scores)
    for j,key in enumerate(['low','high']):best['limits'][key],_=e.cal.g.edge(risk[normal,j],best['limits'][key])
    if best['gate']['rule']!='All':
        rule=best['gate']['rule'];best['gate']['threshold'],_=e.cal.g.edge(scores[rule][normal],best['gate']['threshold'])
    np.testing.assert_array_equal(before,e.cal.normal_action(best,pred,raw,risk,scores));return best

def verify():
    d,*_=e.init();checked=0
    cases=[]
    for budget in [54,60]:
        for rep in range(100):
            folder=e.O/f'primary/dev{budget}/time900/round_{rep:02d}';pol=json.loads((folder/'POLICIES.json').read_text())
            for arm in e.ARMS:cases.append((folder/(arm+'.npz'),pol[arm]['common_B']))
    folder=e.O/'internal301/round_00';pol=json.loads((folder/'POLICIES.json').read_text())
    for fi in range(5):
        for arm in e.ARMS:cases.append((folder/f'fold{fi}_{arm}.npz',pol[f'{fi}_{arm}']['common_B']))
    for path,p in cases:
        with np.load(path) as z:
            dp=z['dev_prediction'];idx=z['dev_rows'];risk=np.c_[-d['x'][idx,1],d['x'][idx,1]];raw=z['dev_risk'];scores={k:z['dev_score_'+k] for k in e.a.RULES}
            q=choose_normal(dp,d['medical'][idx,2],raw,risk,scores,p['normal']['cap_score'])
            assert e.r.safe(q)==e.r.safe(p['normal']),(path,q,p['normal']);checked+=1
    e.r.write(e.O/'EXACT_SELECTOR_CHECK.json',dict(cases=checked,policies_exactly_identical=True,only_count_computation_changed=True));print('EXACT',checked)

if __name__=='__main__':verify()
