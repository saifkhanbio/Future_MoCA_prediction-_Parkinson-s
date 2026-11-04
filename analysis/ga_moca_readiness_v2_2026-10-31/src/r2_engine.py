"""Pure numerical/search engine. Search accepts outer-training data only."""
import math
import numpy as np
COUNTERS={'ridge_attempts':0,'ridge_completed':0}
def numeric(v):
    if v in ['',None]: return np.nan
    x=float(v)
    if np.isinf(x): raise ValueError('Infinite input')
    return x
def layout(c):
    core=c['features']['core'];opt=[r['name'] for r in c['features']['optional']]
    cats=dict(core['categories']);cats.update({k:v['levels'] for k,v in c['features']['nuisance_categories'].items()})
    return core['numeric']+opt,cats
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
def selected(mask,c):
    d=len(c['features']['optional'])
    if not isinstance(mask,(int,np.integer)) or mask<0 or mask>=2**d:raise ValueError('Invalid mask')
    return [r['name'] for r in c['features']['optional'] if mask & (1<<r['bit'])]
def column_indices(mask,c,labels):
    names=c['features']['core']['numeric']+selected(mask,c)
    wanted=set(names+[x+'__missing' for x in names])
    cats=set(c['features']['core']['categories'])
    for k,v in c['features']['nuisance_categories'].items():
        if set(v['trigger'])&set(names):cats.add(k)
    return np.array([i for i,l in enumerate(labels) if l in wanted or l.split('==')[0] in cats],dtype=int)
def burden(mask,c):
    return len({r['instrument'] for r in c['features']['optional'] if mask&(1<<r['bit'])})
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
class Fitness:
    def __init__(self,rows,y,inner,c):
        self.c=c;self.y=np.asarray(y,dtype=float);self.inner=np.asarray(inner,dtype=int)
        if len(rows)!=len(y) or len(inner)!=len(y) or set(self.inner)!={0,1,2}:raise ValueError('Invalid inner partition')
        self.variance=float(np.var(self.y))
        if not np.isfinite(self.variance) or self.variance<=0:raise ValueError('Undefined pooled training R2')
        self.parts=[];self.cache={};self.solves=0
        for k in range(3):
            train=np.flatnonzero(self.inner!=k);valid=np.flatnonzero(self.inner==k)
            tr=[rows[i] for i in train];va=[rows[i] for i in valid]
            state=prepare(tr,c);X,labels=transform(tr,state);V,labels2=transform(va,state)
            if labels!=labels2:raise ValueError('Design binding mismatch')
            self.parts.append((X,V,self.y[train],self.y[valid],labels))
    def __call__(self,mask):
        mask=int(mask);selected(mask,self.c)
        if mask in self.cache:return self.cache[mask]
        losses=[];fold_losses=[]
        for alpha in self.c['model']['lambdas']:
            fold=[];total=0.;n=0
            for X,V,y,z,labels in self.parts:
                ix=column_indices(mask,self.c,labels);p=predict(V[:,ix],ridge(X[:,ix],y,alpha));self.solves+=1
                err=np.square(z-p);fold.append(float(err.mean()));total+=float(err.sum());n+=len(z)
            losses.append(total/n);fold_losses.append(fold)
        best=min(losses);eligible=[i for i,v in enumerate(losses) if v<=best+1e-12]
        idx=max(eligible,key=lambda i:self.c['model']['lambdas'][i])
        out={'mask':mask,'fitness_schema':'mse_r2_v1','outcome_variance':self.variance,'pooled_sst':self.variance*len(self.y),'r2':float(1-losses[idx]/self.variance),'normalized_mse':float(losses[idx]/self.variance),'mse':float(losses[idx]),'burden':burden(mask,self.c),'scores':len(selected(mask,self.c)),'alpha':self.c['model']['lambdas'][idx],'fold_mse':fold_losses[idx],'lambda_grid_mse':losses,'lambda_grid_r2':(1-np.array(losses)/self.variance).tolist(),'all_fold_mse':fold_losses,'all_fold_sse':(np.array(fold_losses)*np.array([len(a[3]) for a in self.parts])).tolist(),'fold_sizes':[len(a[3]) for a in self.parts]}
        if not all(math.isfinite(x) for x in losses):raise ValueError('Nonfinite fitness')
        self.cache[mask]=out
        return out
def ranks_crowding(records):
    val=np.array([[r['mse'],r['burden']] for r in records],dtype=float);n=len(records)
    dom=np.all(val[:,None,:]<=val[None,:,:],axis=2)&np.any(val[:,None,:]<val[None,:,:],axis=2)
    remaining=dom.sum(axis=0);rank=np.full(n,-1,dtype=int);front=np.flatnonzero(remaining==0);k=0;crowd=np.zeros(n)
    while len(front):
        rank[front]=k
        for j in range(2):
            order=front[np.argsort(val[front,j],kind='stable')];span=val[order[-1],j]-val[order[0],j]
            if span>0:
                crowd[order[0]]=crowd[order[-1]]=np.inf
                if len(order)>2:crowd[order[1:-1]]+=(val[order[2:],j]-val[order[:-2],j])/span
        remaining-=dom[front].sum(axis=0);front=np.flatnonzero((remaining==0)&(rank<0));k+=1
    if np.any(rank<0):raise RuntimeError('Incomplete Pareto ranks')
    return rank,crowd
def mandatory(c):
    d=len(c['features']['optional'])
    return list(dict.fromkeys([0,c['controls']['enhanced_mask'],2**d-1]+[1<<i for i in range(d)]))
def choose_final(records):
    if not records:raise ValueError('Empty archive')
    minimum=min(r['mse'] for r in records)
    pick=min((r for r in records if r['mse']<=minimum+1e-12),key=lambda r:(r['scores'],r['burden'],r['mask']))
    return {'selected':pick,'anchor_mask':pick['mask'],'anchor_mse':pick['mse'],'heuristic_SE':0.0,'threshold':minimum+1e-12,'selection_rule':'maximum_inner_accuracy_then_feature_count'}
def pareto(records):
    out=[];previous=float('inf')
    for cost in sorted({r['burden'] for r in records}):
        rr=[r for r in records if r['burden']==cost];best=min(r['mse'] for r in rr)
        if best<previous:out.extend(r for r in rr if r['mse']==best)
        previous=min(previous,best)
    return sorted(out,key=lambda r:(r['burden'],r['mse'],r['mask']))
def search_ga(fitness,c,horizon,outer):
    plan=c['search'];d=len(c['features']['optional']);maxmask=2**d-1;size=plan['population'];history=[];visited=set()
    fixed=mandatory(c)
    if size<len(fixed):raise ValueError('Population smaller than mandatory mask set')
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed+10000*horizon+100*outer);pop=fixed.copy()
        while len(pop)<size:pop.append(int(rng.integers(0,maxmask+1)))
        for m in pop:fitness(m);visited.add(m)
        for generation in range(plan['offspring_batches']):
            records=[fitness(m) for m in pop];rank,crowd=ranks_crowding(records)
            def parent():
                a,b=rng.integers(0,len(pop),size=2)
                ka=(rank[a],-crowd[a]);kb=(rank[b],-crowd[b])
                return pop[a] if ka<kb else pop[b] if kb<ka else pop[int(rng.choice([a,b]))]
            children=[]
            for j in range(plan['offspring_each']):
                a,b=parent(),parent()
                if rng.random()<.9:
                    coin=int(rng.integers(0,maxmask+1));child=(a&coin)|(b&(~coin&maxmask))
                else:child=a
                mut=sum((1<<i) for i,x in enumerate(rng.random(d)) if x<1/d);child^=mut
                children.append(child);fitness(child);visited.add(child)
            union=pop+children;rr,cc=ranks_crowding([fitness(m) for m in union])
            order=sorted(range(len(union)),key=lambda i:(rr[i],-cc[i],union[i]))
            pop=[union[i] for i in order[:size]]
            history.append({'restart_seed':int(seed),'generation':generation+1,'unique_visited':len(visited),'population_unique':len(set(pop))})
    archive=[fitness(m) for m in sorted(visited)]
    return {'choice':choose_final(archive),'visited':sorted(visited),'pareto':pareto(archive),'history':history}
def search_random(fitness,c,horizon,outer,budget):
    rng=np.random.default_rng(20271030+10000*horizon+100*outer);seen=set(mandatory(c));d=len(c['features']['optional'])
    if budget<len(seen) or budget>2**d:raise ValueError('Invalid random-search budget')
    while len(seen)<budget:seen.add(int(rng.integers(0,2**d)))
    records=[fitness(m) for m in sorted(seen)]
    return {'choice':choose_final(records),'visited':sorted(seen),'pareto':pareto(records)}
def outer_task(train_rows,train_y,inner,test_rows,c,horizon,outer):
    """No held-out labels are accepted by this function."""
    f=Fitness(train_rows,train_y,inner,c);ga=search_ga(f,c,horizon,outer)
    random=search_random(f,c,horizon,outer,len(ga['visited']))
    masks={'core':c['controls']['core_mask'],'enhanced':c['controls']['enhanced_mask'],'full':c['controls']['full_mask'],'GA_selected':ga['choice']['selected']['mask'],'random_selected':random['choice']['selected']['mask']}
    state=prepare(train_rows,c);X,labels=transform(train_rows,state);V,_=transform(test_rows,state);predictions={};models={}
    for arm,mask in masks.items():
        fit=f(mask);ix=column_indices(mask,c,labels);model=ridge(X[:,ix],train_y,fit['alpha'])
        predictions[arm]=predict(V[:,ix],model)
        models[arm]={'mask':mask,'features':selected(mask,c),'burden':burden(mask,c),'alpha':fit['alpha'],'columns':[labels[i] for i in ix],'coef':model[0].tolist(),'intercept':model[1],'preprocessing':state}
    baseline=np.array([numeric(r['baseline_moca']) for r in test_rows]);trainbaseline=np.array([numeric(r['baseline_moca']) for r in train_rows])
    change=float(np.median(np.asarray(train_y)-trainbaseline))
    predictions['persistence']=baseline;predictions['typical_change']=baseline+change
    return predictions,{'GA':ga,'random':random,'models':models,'typical_change':change,'fitness_cache':list(f.cache.values()),'ridge_inner_solves':f.solves,'ridge_outer_solves':5,'unique_union':len(f.cache)}
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
