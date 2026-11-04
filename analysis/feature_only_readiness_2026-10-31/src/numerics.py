"""Frozen numerical primitives copied verbatim; new feature layouts are supplied explicitly."""
import math
import numpy as np
COUNTERS={"ridge_attempts":0,"ridge_completed":0}
def layout(c):
    return c["numeric"],c["categories"]
def numeric(v):
    if v in ['',None]: return np.nan
    x=float(v)
    if np.isinf(x): raise ValueError('Infinite input')
    return x

def prepare(rows,c):
    names,cats=layout(c);state={'numeric':{},'categories':cats}
    for name in names:
        v=np.array([numeric(r[name]) for r in rows]);a=v[np.isfinite(v)]
        med=float(np.median(a)) if len(a) else 0.0
        im=np.where(np.isfinite(v),v,med);mu=float(im.mean());sd=float(im.std())
        state['numeric'][name]={'median':med,'mean':mu,'sd':sd if sd>1e-12 else 1.0,'training_observed_n':int(len(a))}
    return state

def transform(rows,state):
    columns=[];labels=[]
    for name,s in state['numeric'].items():
        v=np.array([numeric(r[name]) for r in rows]);miss=~np.isfinite(v)
        columns += [(np.where(miss,s['median'],v)-s['mean'])/s['sd'],miss.astype(float)]
        labels += [name,name+'__missing']
    for name,levels in state['categories'].items():
        v=[r.get(name,'unknown') or 'unknown' for r in rows]
        if any(x not in levels for x in v):raise ValueError('Invalid categorical value: '+name)
        for level in levels:
            columns.append(np.array([x==level for x in v],dtype=float));labels.append(name+'=='+level)
    X=np.column_stack(columns)
    if not np.isfinite(X).all():raise ValueError('Nonfinite design')
    return X,labels

def ridge(X,y,alpha):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
    if len(X)!=len(y) or not len(y) or alpha<=0 or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Invalid ridge inputs')
    COUNTERS['ridge_attempts']+=1
    xm=X.mean(axis=0);ym=float(y.mean());z=X-xm
    coef=np.linalg.solve(z.T@z+len(y)*alpha*np.eye(X.shape[1]),z.T@(y-ym))
    intercept=ym-float(xm@coef)
    if not np.isfinite(coef).all() or not math.isfinite(intercept):raise ValueError('Nonfinite ridge solution')
    COUNTERS['ridge_completed']+=1
    return coef,intercept

def predict(X,model):
    p=X@model[0]+model[1]
    if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
    return p

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y;den=float(np.sum((y-y.mean())**2));v=float(np.sum((p-p.mean())**2))
    slope=float(np.sum((p-p.mean())*(y-y.mean()))/v) if np.ptp(p)>1e-12 and v>0 else None
    return {'n':len(y),'MSE':float(np.mean(e**2)),'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'R2':1-float(np.sum(e**2))/den if den>0 else None,'mean_prediction_error':float(e.mean()),'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'calibration_slope':slope,'outside_0_30_fraction':float(np.mean((p<0)|(p>30))),'clipped_MAE':float(np.mean(abs(np.clip(p,0,30)-y)))}

def interval(y,p,q,seed=20261030,replicates=5000):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    if y.ndim!=1 or len(y)<2 or p.shape!=y.shape or q.shape!=y.shape or not all(np.isfinite(x).all() for x in (y,p,q)):raise ValueError('Invalid paired score input')
    ae=np.abs(y-p)-np.abs(y-q);se=np.square(y-q)-np.square(y-p)
    sst=float(np.sum(np.square(y-y.mean())))
    ix=np.random.default_rng(seed).integers(0,len(y),size=(replicates,len(y)))
    yy=y[ix];den=np.sum(np.square(yy-yy.mean(1,keepdims=True)),axis=1)
    good=den>0;valid=int(good.sum());undefined=replicates-valid
    delta_r2=float(se.sum()/sst) if sst>0 else None
    rboots=se[ix].sum(1)[good]/den[good]
    rlo,rhi=(np.quantile(rboots,[.025,.975]).tolist() if valid>=math.ceil(.95*replicates) and sst>0 else [None,None])
    mlo,mhi=np.quantile(ae[ix].mean(1),[.025,.975])
    return {'delta_R2':delta_r2,'lower95_R2':rlo,'upper95_R2':rhi,
            'delta_MAE':float(ae.mean()),'lower95_MAE':float(mlo),'upper95_MAE':float(mhi),
            'replicates':replicates,'R2_valid_replicates':valid,'R2_undefined_replicates':undefined,
            'R2_interval_available':rlo is not None,'conditional_on_fixed_predictions':True,
            'R2_positive_favours_first':True,'MAE_negative_favours_first':True}
