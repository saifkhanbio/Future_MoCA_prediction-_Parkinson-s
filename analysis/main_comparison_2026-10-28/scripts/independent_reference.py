"""Independent reference calculations; imports no production engine or runner."""
import math
import numpy as np
from scipy.linalg import lstsq

def as_number(x):
 try:
  v=float(x)
  return v if math.isfinite(v) else float('nan')
 except (TypeError,ValueError): return float('nan')

def clinical_state(rows,spec):
 out={'numeric':{},'order':list(spec['numeric']),'categories':spec['categories']}
 for name in spec['numeric']:
  a=[as_number(r[name]) for r in rows];good=sorted(x for x in a if math.isfinite(x));m=len(good)
  median=(good[(m-1)//2]+good[m//2])/2 if m else 0.
  filled=[x if math.isfinite(x) else median for x in a];avg=math.fsum(filled)/len(filled);sd=math.sqrt(math.fsum((x-avg)**2 for x in filled)/len(filled))
  out['numeric'][name]={'median':median,'mean':avg,'sd':sd if sd>1e-12 else 1.}
 return out

def clinical_design(rows,state):
 result=[];names=[]
 for k in state['order']: names += [k,k+'__missing']
 for k,levels in state['categories'].items(): names += [k+'='+v for v in levels]
 for r in rows:
  z=[]
  for k in state['order']:
   x=as_number(r[k]);missing=not math.isfinite(x);s=state['numeric'][k];z.extend([((s['median'] if missing else x)-s['mean'])/s['sd'],float(missing)])
  for k,levels in state['categories'].items():
   assert str(r.get(k,'unknown')) in levels
   z.extend([float(str(r.get(k,'unknown'))==v) for v in levels])
  result.append(z)
 return np.asarray(result),names

def rna_state(counts,library,panel,gate,omit=False):
 a=np.asarray(counts);v=np.log1p(a/np.asarray(library)[:,None]*1e6)/math.log(2);n=len(v)
 avg=np.array([math.fsum(x)/n for x in v.T]);sd=np.array([math.sqrt(math.fsum((x-m)**2)/n) for x,m in zip(v.T,avg)])
 need=max(3,int(math.ceil(.2*n)));active=np.array([sum(int(x)>=10 for x in col)>=need and s>1e-12 for col,s in zip(a.T,sd)])
 if omit:active[panel['NPIPB8_index']]=False
 scale=np.where(sd>1e-12,sd,1);z=(v-avg)/scale;z[:,~active]=0;paths=[]
 for members in panel['pathway_indices']:
  ix=[i for i in members if active[i]];q=np.array([math.fsum(row[ix])/len(ix) for row in z]) if len(ix)>=5 else np.zeros(n);mu=float(q.sum()/n);s=float(np.sqrt(np.dot(q-mu,q-mu)/n));ok=len(ix)>=5 and s>1e-12
  paths.append({'eligible_indices':ix,'coverage':len(ix),'active':bool(ok),'mean':mu if ok else 0.,'sd':s if ok else 1.})
 return {'mean':avg.tolist(),'sd':scale.tolist(),'active':active.tolist(),'minimum_training_samples_required':need,'gene5_indices':panel['gene5_indices'],'gene5_names':panel['gene5_names'],'pathway_names':panel['pathway_names'],'pathways':paths,'omit_npipb8':omit}

def rna_design(counts,library,state,arm):
 v=np.log1p(np.asarray(counts)/np.asarray(library)[:,None]*1e6)/math.log(2);z=(v-np.asarray(state['mean']))/np.asarray(state['sd']);z[:,np.logical_not(state['active'])]=0;parts=[];names=[]
 if arm in ['Genes5','Genes5_Hallmark11']:
  parts.append(z[:,state['gene5_indices']]);names += ['gene:'+x for x in state['gene5_names']]
 if arm in ['Hallmark11','Genes5_Hallmark11']:
  for p,n in zip(state['pathways'],state['pathway_names']):
   parts.append(((z[:,p['eligible_indices']].sum(axis=1)/p['coverage']-p['mean'])/p['sd'] if p['active'] else np.zeros(len(z)))[:,None]);names.append('pathway:'+n)
 return (np.hstack(parts) if parts else np.empty((len(z),0))),names

def augmented_svd_ridge(X,y,nclinical,lc,lr):
 X=np.asarray(X);n,p=X.shape;penalty=np.r_[np.repeat(lc,nclinical),np.repeat(lc if lr is None else lr,p-nclinical)]
 A=np.vstack([np.c_[np.ones(n),X]/np.sqrt(n),np.c_[np.zeros(p),np.diag(np.sqrt(penalty))]])
 b=np.r_[np.asarray(y)/np.sqrt(n),np.zeros(p)]
 beta=lstsq(A,b,lapack_driver='gelsd')[0]
 return beta[1:],float(beta[0])

def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);n=len(y);d=p-y;my=math.fsum(y)/n;mp=math.fsum(p)/n;sy=math.fsum((y-my)**2);sp=math.fsum((p-mp)**2)
 slope=math.fsum((y-my)*(p-mp))/sp if sp>0 and max(p)-min(p)>1e-12 else None
 return {'n':n,'MAE':math.fsum(abs(d))/n,'RMSE':math.sqrt(math.fsum(d*d)/n),'R2':1-math.fsum(d*d)/sy if sy>0 else None,'mean_prediction_error':math.fsum(d)/n,'calibration_slope':slope,'calibration_intercept':my-slope*mp if slope is not None else None,'fraction_predictions_outside_0_30':sum(x<0 or x>30 for x in p)/n,'clipped_MAE_0_30_descriptive':math.fsum(abs(np.clip(p,0,30)-y))/n}

def interval(y,left,right,reps=5000,seed=20261024):
 y=np.asarray(y);d=np.abs(np.asarray(left)-y)-np.abs(np.asarray(right)-y);rng=np.random.default_rng(seed);index=rng.integers(0,len(y),size=(reps,len(y)));dist=d[index].sum(axis=1)/len(y);lo,hi=np.quantile(dist,[.025,.975]);base=math.fsum(abs(np.asarray(right)-y))/len(y);delta=math.fsum(d)/len(y)
 return {'delta_MAE':delta,'conditional_lower95':float(lo),'conditional_upper95':float(hi),'relative_MAE_reduction':-delta/base if base>0 else None,'replicates':reps,'conditional_on_fixed_OOF':True}