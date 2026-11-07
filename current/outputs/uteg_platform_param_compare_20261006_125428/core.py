import runtime as r
import numpy as np,torch,joblib,time,math
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
CONFIGS={'TT-SMALL':(12,2,1,32)};ARMS=r.ARMS;FLOOR=1e-6
r.extract(r.OLD/'common.py',['project','ci','labels','fit_cal','predict_cal'],globals())
r.extract(r.CAP/'core.py',['TT','digest','calibrated','metrics','dump'],globals())
r.extract(r.PREV/'core.py',['weights','loss_parts'],globals())
r.extract(r.ZERO/'core.py',['fit','predict'],globals())
def inputs(data,arm):return data[arm]
def newnet(seed,dim=5):
    torch.manual_seed(seed);net=TT(CONFIGS['TT-SMALL'],seed);ix={5:[1,2,3,4,5],4:[2,3,4,5],3:[3,4,5]}[dim];net.val=nn.Parameter(net.val[ix].detach().clone());net.feat=nn.Parameter(net.feat[ix].detach().clone());return net
