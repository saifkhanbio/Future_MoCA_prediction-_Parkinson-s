"""Evaluation core shared by both launchers. No fitting or target-derived preprocessing."""
from itertools import combinations
import numpy as np

def require(test,reason):
    if not test: raise ValueError(reason)

def vector(x):
    a=np.asarray(x,dtype=float)
    require(np.all(np.isfinite(a)),"Non-finite numeric input")
    return a

def forecast_a(rows,states,arm,h,spec,complete):
    numeric=spec['active']['active_numeric'][arm]
    cats=spec['active']['active_categories'][arm]
    compact=spec['active']['active_numeric']['compact4']
    expected={name for k in numeric for name in (k,k+'__missing')}
    expected.update(k+'=='+level for k,levels in cats.items() for level in levels)
    predictions=[]
    for fold in range(5):
        state=states[f'{arm}_{h}m_fold{fold}']
        cols=state['columns']
        require(set(cols)==expected and len(cols)==len(expected),"Invalid active columns")
        vals={}
        for k in numeric:
            raw=[r['x'][k] for r in rows]
            missing=np.array([v is None for v in raw])
            require(not missing.any() or (not complete and k not in compact),"Required measure missing")
            q=state['preprocessing']['numeric'][k]
            median,mean,sd=vector([q['median'],q['mean'],q['sd']])
            require(sd>0,"Invalid source scale")
            vals[k]=(vector([median if v is None else v for v in raw])-mean)/sd
            vals[k+'__missing']=missing
        for k,levels in cats.items():
            require(state['preprocessing']['categories'][k]==levels,"Category binding mismatch")
            raw=[r['x'][k] for r in rows]
            require(all(v in levels for v in raw),"Unbound category")
            if complete or k in ('baseline_hvlt_form','baseline_sdmt_form'):
                require('unknown' not in raw,"Required category missing")
            for level in levels:vals[k+'=='+level]=np.array([v==level for v in raw])
        coefficients=vector(state['coef'])
        require(coefficients.shape==(len(cols),),"Coefficient dimension mismatch")
        predictions.append(vector(state['intercept'])+np.column_stack([vals[c] for c in cols])@coefficients)
    return vector(np.mean(predictions,axis=0))

def forecast_b(rows,state):
    fields=['baseline_moca','age_years','sex_male']
    require(state['features']==fields,"Reduced3 feature order mismatch")
    x=vector([[r['x'][k] for k in fields] for r in rows])
    require(np.all(x==np.floor(x)) and np.all((x[:,1]>=18)&(x[:,1]<=89)) and np.all(np.isin(x[:,2],[0,1])),"Invalid age or sex")
    mean,scale,coef=[vector(state[k]) for k in ('mean','scale','coefficient')]
    require(all(a.shape==(3,) for a in (mean,scale,coef)) and np.all(scale>0),"Invalid reduced model")
    return vector(state['intercept']+(x-mean)/scale@coef)

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

def evaluate(plan,spec,states,features,outcomes):
    stage=plan['stage']
    def index(rows):
        keys=[(r['h'],r['id']) for r in rows]
        require(len(set(keys))==len(keys) and all(h in plan['horizons'] for h,i in keys),"Duplicate identity or invalid horizon")
        return dict(zip(keys,rows))
    xx=index(features); yy=index(outcomes)
    require(xx.keys()==yy.keys(),"Outcome membership mismatch")
    expected={f'{arm}_{h}m_fold{i}' for arm in spec['active']['active_numeric'] for h in plan['horizons'] for i in range(5)} if stage=='A' else set(map(str,plan['horizons']))
    require(set(states)==expected,"Frozen model inventory mismatch")
    for r in features:
        vs=r['views'];require(plan['primary_view'] in vs and len(set(vs))==len(vs) and set(vs)<=plan['views'].keys(),"Invalid view membership")
        if stage=='A':
            require(('complete_all' not in vs or {'complete19','supported_all','supported19'}<=set(vs)) and ('supported_all' not in vs or 'supported19' in vs) and ('complete19' not in vs or 'supported19' in vs),"Invalid matched subset")
    answer={}
    for hi,h in enumerate(plan['horizons']):
        for view,settings in plan['views'].items():
            keys=sorted(k for k,r in xx.items() if k[0]==h and view in r['views'])
            require(len(keys)==settings['n'][hi],"Unexpected view size")
            rows=[xx[k] for k in keys];y=vector([yy[k]['y'] for k in keys])
            baseline=vector([r['x']['baseline_moca'] for r in rows])
            require(all(np.all((v>=0)&(v<=30)&(v==np.floor(v))) for v in [y,baseline]),"Invalid MoCA")
            pred={}
            for m in settings['models']:
                if m=='persistence':pred[m]=baseline
                elif m=='typical':pred[m]=baseline+spec['offset_'+stage][str(h)]
                elif stage=='B':pred[m]=forecast_b(rows,states[str(h)])
                else:pred[m]=forecast_a(rows,states,m,h,spec,view.startswith('complete') or view=='compact')
            seed=plan['seed_base']+100*settings['rank']+h
            answer[f'{view}/{h}']=dict(n=len(keys),seed=seed,metrics=summarize(y,pred,plan['bootstrap'],seed))
    return answer
