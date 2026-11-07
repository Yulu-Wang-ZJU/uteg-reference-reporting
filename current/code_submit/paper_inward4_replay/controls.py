"""Fixed representation/loss controls and median baseline on identical splits."""
import numpy as np
from threadpoolctl import threadpool_limits
from . import core as c

def run(fixed_input=False):
    d,_,_,_,_,folds,local=c.context();configs=[('B','uniform'),('B','class'),('three','inward4'),('four','inward4')]
    c.write(c.OUT/'CONTROLS_PROTOCOL.json',dict(configs=configs,selected=['B','inward4'],test_selection=False,
        paired_input=('Read identical B values from active fixed dataset; other3parameters unchanged; three omits nu' if fixed_input else 'Recompute B in each same training fold; other3parameters unchanged; three omits nu'),
        median='Training median MA only, no query input',B_from_original_data=True))
    for direction in c.MAIN:
        for rep,loss in configs:
            path=c.OUT/'controls'/f'{direction}_{rep}_{loss}.npz'
            if path.exists():continue
            pp=[];aa=[]
            for seed in c.SEEDS:
                spec=c.paper_spec(direction,seed)
                o=c.evaluate(spec['pool'],spec['test'],spec['positions'][3]['inner'],direction,900,rep,loss)
                pp.append(o['prediction']);aa.append(o['accepted'])
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=pp,accepted=aa,test=spec['test'],seed=c.SEEDS)
            c.write(path.with_suffix('.metrics.json'),dict(all_report=c.summarize(pp,d['y'][spec['test'],2]),gate=c.summarize(pp,d['y'][spec['test'],2],aa)))
            print('CONTROL_COMPLETE',direction,rep,loss,flush=True)
    for rep,loss in configs:
        path=c.OUT/'controls'/f'all301_{rep}_{loss}.npz'
        if path.exists():continue
        pred=np.full((3,len(local)),np.nan);acc=np.zeros_like(pred,bool);pos={int(v):i for i,v in enumerate(local)}
        for f in folds:
            for j,direction in enumerate(c.MAIN):
                o=c.evaluate(f['A'],f['B'],f['inner'],direction,900,rep,loss);ix=[pos[v] for v in f['B']]
                pred[j,ix]=o['prediction'];acc[j,ix]=o['accepted']
        np.savez_compressed(path,prediction=pred,accepted=acc,local=local)
        c.write(path.with_suffix('.metrics.json'),{arm:dict(all_report=c.summarize(pred[j],d['y'][local,2]),gate=c.summarize(pred[j],d['y'][local,2],acc[j])) for j,arm in enumerate(c.MAIN)})
    med={}
    for direction in c.TARGET:
        for budget in ([0,20,40] if c.TARGET[direction]=='SAHZU' else [0,20]):
            pp=[]
            for seed in c.SEEDS:
                sp=c.paper_spec(direction,seed,budget);pp.append(np.full(len(sp['test']),np.median(d['y'][sp['pool'],2])))
            med[f'{direction}_B{budget}']=c.summarize(pp,d['y'][sp['test'],2])
    c.write(c.OUT/'controls/MEDIAN_RESULTS.json',med)
    print('CONTROLS_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
