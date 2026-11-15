"""Frozen Stage A/B metric and paired-bootstrap routines."""
from itertools import combinations
import numpy as np

def metrics(y,p):
    """Rows represent bootstrap draws, columns represent participants."""
    e=p-y
    yc=y-y.mean(axis=1,keepdims=True)
    pc=p-p.mean(axis=1,keepdims=True)
    sst=(yc**2).sum(axis=1)
    var=(pc**2).sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        r2=np.where(sst>0,1-(e**2).sum(axis=1)/sst,np.nan)
        slope=np.where(np.ptp(p,axis=1)>1e-12,(pc*yc).sum(axis=1)/var,np.nan)
    return dict(MAE=np.abs(e).mean(axis=1),RMSE=np.sqrt((e**2).mean(axis=1)),R2=r2,
                bias=e.mean(axis=1),calibration_slope=slope,calibration_intercept=y.mean(axis=1)-slope*p.mean(axis=1),
                outside_0_30=((p<0)|(p>30)).mean(axis=1))

def summarize(y,pred,nboot,seed):
    pairs=list(combinations(pred,2))
    def calc(ix):
        arms={a:metrics(y[ix],p[ix]) for a,p in pred.items()}
        flat={a+'/'+k:v for a,ms in arms.items() for k,v in ms.items()}
        for a,b in pairs:
            for k in ('MAE','R2'):flat[a+'-'+b+'/delta_'+k]=arms[a][k]-arms[b][k]
        return flat
    point=calc(np.arange(len(y))[None,:])
    draws={key:[] for key in point}
    rng=np.random.Generator(np.random.PCG64(seed))
    for start in range(0,nboot,128):
        ix=rng.integers(0,len(y),size=(min(128,nboot-start),len(y)))
        for key,v in calc(ix).items():draws[key].append(v)
    answer={}
    for key,parts in draws.items():
        v=np.concatenate(parts); finite=v[np.isfinite(v)]; q=float(point[key][0])
        ci=np.quantile(finite,[.025,.975],method='linear').tolist() if len(finite)>=.95*nboot and np.isfinite(q) else None
        answer[key]=dict(estimate=q if np.isfinite(q) else None,interval=ci,valid=len(finite))
    return answer

