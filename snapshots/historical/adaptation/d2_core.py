import runtime as r
import numpy as np,torch,joblib,time,copy
from scipy.optimize import minimize
from scipy.special import expit,softmax
torch.set_default_dtype(torch.float64);torch.set_num_threads(1);FLOOR=1e-6
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.ZERO/'verify_independent.py',['source'],globals());source_tt=source
r.extract(r.OLDMAP/'verify.py',['source'],globals())
r.extract(r.GATE/'independent_statistics.py',['areas'],globals())
r.extract(r.TH/'core.py',['features','actions','threshold','select','ratio','risk_at','metrics','bootstrap','dump'],globals())
r.extract(r.CF/'core.py',['folds','calibrated'],globals())
scalar=dict(np=np,torch=torch,time=time,minimize=minimize)
r.extract(r.SC/'core.py',['logp','objective','fit','probabilities','score'],scalar)
rm=r.load('risk_new',r.GATE/'risk_models.py');rm.N_FEATURES=13
im=r.load('numpy_independent',r.GATE/'independent_models.py');im.N_FEATURES=13
CONFIGS=[dict(id=n,family='SCALAR',old=n) for n in ['SOURCE-CI-ORDINAL','HUBER-CI-ORDINAL']]
CONFIGS += [dict(id='LR-'+str(v),family='ERROR-LR',regularizer=v,old='ERROR-LR' if v==.1 else None) for v in [.1,1.]]
CONFIGS += [dict(id=f'GP-{l}-{n}',family='GP-RISK',length=l,noise=n,old=f'GP-RISK@{l:g}' if n==.05 else None) for l,n in [(.5,.05),(1.,.05),(2.,.05),(1.,.2)]]
CONFIGS += [dict(id=f'DOS-{w}',family='DOS-GATE',width=w,cost=.1,old='DOS-GATE' if w==8 else None) for w in [8,2,4]]

class SmallNet(torch.nn.Module):
 def __init__(self,out_dim,seed):
  super().__init__()
  with torch.random.fork_rng():
   torch.manual_seed(seed);self.fc1=torch.nn.Linear(13,WIDTH);self.out=torch.nn.Linear(WIDTH,out_dim)
 def forward(self,x):return self.out(torch.nn.functional.silu(self.fc1(x)))
WIDTH=8
rm._ClinicalNet=SmallNet
# The original GP function is changed only in the registered observation-noise constant.
gpnode=[n for n in r.ast.parse((r.GATE/'risk_models.py').read_text()).body if isinstance(n,r.ast.FunctionDef) and n.name=='_fit_gp'][0]
class Noise(r.ast.NodeTransformer):
 def visit_Constant(self,n):return r.ast.copy_location(r.ast.Name(id='REGISTERED_NOISE',ctx=r.ast.Load()),n) if isinstance(n.value,float) and n.value==.05 else n
exec(compile(r.ast.fix_missing_locations(r.ast.Module(body=[Noise().visit(gpnode)],type_ignores=[])),'registered_gp_noise','exec'),rm.__dict__)

def fit(cfg,F,yh,yt,seed,j=None):
 global WIDTH
 if cfg['family']=='SCALAR':return scalar['fit'](cfg['id'],j,yt)
 rm.REGULARIZER=cfg.get('regularizer',.1);rm.REGISTERED_NOISE=cfg.get('noise',.05);WIDTH=cfg.get('width',8)
 m=rm.fit_model(cfg,F,yh,yt,seed)
 if cfg['family']=='DOS-GATE':
  m['parameter_count_registered']=15*WIDTH+1
  if not m.get('fallback'):m['parameter_count_effective']=15*WIDTH+1
 m['registered_config']=cfg
 return m
def score(m,cfg,F,yh,rawj):
 global WIDTH
 out=np.full(len(yh),np.inf);ok=yh!=9
 if cfg['family']=='SCALAR':
  j=rawj if cfg['id'].startswith('SOURCE') else F[:,8]*3
  if ok.any():out[ok]=scalar['score'](m,j[ok],yh[ok])
 else:
  WIDTH=cfg.get('width',8)
  if ok.any():out[ok]=rm.predict_model(m,F[ok],yh[ok])
 return out
