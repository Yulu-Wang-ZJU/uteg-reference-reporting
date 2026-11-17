"""Signal shape and fitting helpers for the current U-TEG inputs."""
import ast,time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from . import core as c
FITTER=c.WS/'outputs/uteg_platform_param_compare_20261006_125428/curves.py'

def shape(q,t,family='F4'):
    t0,b,k,nu=np.asarray(q,float);z=np.maximum(np.asarray(t)-t0,0.);v=np.zeros(len(z));ok=z>0
    power=nu*(np.log(z[ok])-np.log(k))
    v[ok]=b*(-np.expm1(-np.exp(np.minimum(power,700.))))
    return v-np.median(v[:10])

def fitter():
    tree=ast.parse(FITTER.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_simple']
    ns=dict(np=np,time=time,least_squares=least_squares,expit=expit,shape=shape)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FITTER),'exec'),ns)
    return ns['fit_simple']

def activate():
    from .current_version import activate as current_activate
    return current_activate()
