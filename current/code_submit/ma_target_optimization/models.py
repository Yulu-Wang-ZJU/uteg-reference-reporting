"""Small mapping candidates. Every transform is fitted on its training rows.

The boundary-augmented objective is a study-specific proposal, not a CORAL
reproduction. It retains one continuous MA output and fixed 50/70 boundaries.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from code_submit.ma_imbalance_mapping.models import fit_mapping, fit_adapter, project_readouts
from .metrics import states


def weights(y, cap):
    labels=states(y)
    if cap == 1:
        return np.ones(len(labels))
    counts=np.bincount(labels,minlength=3)
    w=1/np.maximum(counts[labels],1)
    w=np.maximum(w,w.max()/cap)
    return w/w.mean()


@dataclass
class Features:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    polynomial: object = None
    post_mean: object = None
    post_scale: object = None

    @classmethod
    def fit(cls, x, kind):
        raw=np.log1p(np.maximum(x,0)) if kind.startswith('log') else np.asarray(x,float)
        mean=raw.mean(0); scale=np.maximum(raw.std(0),1e-6)
        obj=cls(kind,mean,scale)
        if kind.endswith('quadratic'):
            obj.polynomial=PolynomialFeatures(2,include_bias=False)
            z=obj.polynomial.fit_transform((raw-mean)/scale)
            obj.post_mean=z.mean(0);obj.post_scale=np.maximum(z.std(0),1e-6)
        return obj

    def transform(self,x):
        raw=np.log1p(np.maximum(x,0)) if self.kind.startswith('log') else np.asarray(x,float)
        z=(raw-self.mean)/self.scale
        if self.polynomial is not None:
            z=(self.polynomial.transform(z)-self.post_mean)/self.post_scale
        return z


def boundary_objective(theta,z,y,w,alpha,strength,abnormal_cost):
    # Output coordinates are fixed MA units: 60 + 10*ztheta.
    pred=60+10*(z@theta)
    residual=(pred-y)/10
    penalty=np.r_[np.ones(len(theta)-1),0.]
    value=float(np.mean(w*residual**2)+alpha*np.sum(penalty*theta**2))
    grad=z.T@(2*w*residual)/len(y)+2*alpha*penalty*theta
    if strength:
        logits=(pred[:,None]-np.array([50.,70.]))/2.
        target=np.column_stack((y>=50,y>70)).astype(float)
        bw=np.where(states(y)==1,1.,float(abnormal_cost));bw=bw/bw.mean()
        terms=np.logaddexp(0,logits)-target*logits
        value+=float(strength*np.mean(bw[:,None]*terms))
        grad+=z.T@(strength*bw*np.sum(expit(logits)-target,axis=1)*2.5)/len(y)
    return value,grad


@dataclass
class ContinuousModel:
    features: Features
    coef: np.ndarray
    diagnostics: dict

    def predict(self,x):
        z=self.features.transform(x)
        return np.maximum(60+10*(np.c_[z,np.ones(len(z))]@self.coef),0)


def fit_continuous(x,y,candidate):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Training arrays must be finite')
    f=Features.fit(x,candidate.get('features','linear'));z=f.transform(x);z=np.c_[z,np.ones(len(z))]
    w=weights(y,candidate.get('cap',1));alpha=candidate['alpha']
    target=(y-60)/10; penalty=np.diag(np.r_[np.ones(z.shape[1]-1),0.])
    initial=np.linalg.solve(z.T@(w[:,None]*z)+len(y)*alpha*penalty+1e-12*np.eye(z.shape[1]),z.T@(w*target))
    strength=candidate.get('boundary_strength',0.)
    if strength:
        result=minimize(boundary_objective,initial,args=(z,y,w,alpha,strength,candidate.get('abnormal_cost',1)),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-11,'gtol':1e-8,'maxls':50})
        if not result.success:
            raise RuntimeError(str(result.message))
        coef=result.x
    else:
        coef=initial
    return ContinuousModel(f,coef,{'n':len(y),'weights_min':float(w.min()),'weights_max':float(w.max()),'boundary_strength':strength})


@dataclass
class KernelModel:
    features: Features
    estimator: object

    def predict(self,x):
        return np.maximum(60+10*self.estimator.predict(self.features.transform(x)),0)


@dataclass
class TransferModel:
    source: object
    adapter: object
    mode: str

    def predict(self,x):
        if self.mode=='joint':
            return self.adapter.predict(x)[:,2]
        raw=project_readouts(self.source.predict(x))[:,2]
        if self.mode=='ma_only':
            return self.adapter.predict(raw[:,None])
        if self.mode=='residual':
            # The residual model uses 60 as its neutral output.
            return np.maximum(raw+self.adapter.predict(x)-60,0)
        raise ValueError(self.mode)


@dataclass
class SeparateSourceModel:
    sources: list
    adapter: object

    def predict(self,x):
        z=np.column_stack([s.predict(x)[:,2] for s in self.sources])
        return self.adapter.predict(z)


def fit_candidate(candidate, train, source_cache):
    """Train object contains only legal source rows and one target fold.

    No full dataset or test matrix is accepted by this fitting interface.
    """
    x,y=train['x'],train['y']
    family=candidate['family']
    if family in ('ridge','boundary'):
        return fit_continuous(x,y[:,2],candidate)
    if family=='kernel':
        f=Features.fit(x,candidate['features']);z=f.transform(x)
        est=KernelRidge(alpha=candidate['alpha'],kernel='rbf',gamma=candidate['gamma'])
        est.fit(z,(y[:,2]-60)/10,sample_weight=weights(y[:,2],candidate['cap']))
        return KernelModel(f,est)
    arm=candidate['arm']
    if family=='separate_sources':
        sources=[]
        for source_name in ('Huzhou','Hubei'):
            key=(source_name,candidate['source_alpha'],candidate['source_cap'])
            if key not in source_cache:
                sx,sy=train['source_centers'][source_name]
                source_cache[key]=fit_mapping(sx,sy,method='class_weighted' if candidate['source_cap']>1 else 'unweighted',parameter=1.,ridge_alpha=candidate['source_alpha'],weight_ratio_cap=candidate['source_cap'])
            sources.append(source_cache[key])
        z=np.column_stack([s.predict(x)[:,2] for s in sources])
        return SeparateSourceModel(sources,fit_continuous(z,y[:,2],candidate))
    if family!='transfer':
        raise ValueError(family)
    key=(arm,candidate['source_alpha'],candidate['source_cap'])
    if key not in source_cache:
        sx,sy=train['sources'][arm]
        source_cache[key]=fit_mapping(sx,sy,method='class_weighted' if candidate['source_cap']>1 else 'unweighted',parameter=1.,ridge_alpha=candidate['source_alpha'],weight_ratio_cap=candidate['source_cap'])
    source=source_cache[key];mode=candidate['mode']
    if mode=='joint':
        adapter=fit_adapter(source,x,y,method='class_weighted' if candidate['cap']>1 else 'unweighted',parameter=1.,weight_ratio_cap=candidate['cap'],penalty=candidate['adapter_penalty'])
    elif mode=='ma_only':
        raw=project_readouts(source.predict(x))[:,2]
        adapter=fit_continuous(raw[:,None],y[:,2],candidate)
    elif mode=='residual':
        raw=project_readouts(source.predict(x))[:,2]
        # Importance comes from the true MA category, not residual categories.
        # Residual candidates are deliberately unweighted controls in this round.
        if candidate['cap']!=1:
            raise ValueError('Residual control must use cap=1')
        adapter=fit_continuous(x,60+y[:,2]-raw,candidate)
    else:
        raise ValueError(mode)
    return TransferModel(source,adapter,mode)
