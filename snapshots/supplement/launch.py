"""Process scheduling only; does not change the frozen scientific runner."""
import subprocess,sys,time
import batch as b
def main():
    plan=b.read(b.ROOT/'PLAN.json');pending=[t for t in plan['tasks'] if not (b.ROOT/'runs'/t['id']/'COMPLETE.json').exists()]
    arms=list(dict.fromkeys(x['arm'] for x in pending))
    order={x['id']:(sum(y['arm']==x['arm'] for y in pending[:i]),arms.index(x['arm'])) for i,x in enumerate(pending)}
    pending.sort(key=lambda x:order[x['id']]);active=[];failed=[]
    logdir=b.ROOT/'logs';logdir.mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active)<3 and not failed:
            t=pending.pop(0);f=(logdir/(t['id']+'.log')).open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-B',str(b.ROOT/'batch.py'),'task','--id',t['id']],stdout=f,stderr=subprocess.STDOUT,cwd=str(b.ROOT))
            active.append((p,t,f))
        for p,t,f in active[:]:
            if p.poll() is not None:
                f.close();active.remove((p,t,f))
                if p.returncode:failed.append(t['id'])
        b.write(b.ROOT/'BATCH_STATUS.json',{'running':[{'id':t['id'],'pid':p.pid} for p,t,f in active],'remaining':len(pending),'failed':failed,'completed':len(list((b.ROOT/'runs').glob('*/COMPLETE.json'))),'time':time.strftime('%Y-%m-%dT%H:%M:%S')})
        if failed and not active:break
        time.sleep(2)
    if failed:raise RuntimeError('Stopped scheduling after failure: '+','.join(failed))
    assert len(list((b.ROOT/'runs').glob('*/COMPLETE.json')))==len(plan['tasks'])
    b.write(b.ROOT/'ALL_MODELS_FROZEN.json',{'count':len(plan['tasks']),'test_evaluated':False,'plan_sha':b.sha(b.ROOT/'PLAN.json')})
if __name__=='__main__':main()
