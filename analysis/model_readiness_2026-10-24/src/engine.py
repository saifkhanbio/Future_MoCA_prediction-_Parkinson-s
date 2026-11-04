"""Training-only preprocessing and ridge primitives. Not fitted in readiness QA."""
import numpy as np,math
COUNTERS={'clinical_preparations':0,'RNA_preparations':0,'ridge_solves':0}
def numeric(v):
 try:
  x=float(v);return x if math.isfinite(x) else np.nan
 except (ValueError,TypeError):return np.nan
def prepare_clinical(rows,spec):
 COUNTERS['clinical_preparations']+=1;state={'numeric':{},'order':spec['numeric'],'categories':spec['categories']}
 for k in spec['numeric']:
  x=np.array([numeric(r[k]) for r in rows]);finite=x[np.isfinite(x)];median=float(np.median(finite)) if len(finite) else 0.;z=np.where(np.isfinite(x),x,median);sd=float(z.std())
  state['numeric'][k]={'median':median,'mean':float(z.mean()),'sd':sd if sd>1e-12 else 1.}
 return state
def transform_clinical(rows,state):
 cols=[];names=[]
 for k in state['order']:
  x=np.array([numeric(r[k]) for r in rows]);m=~np.isfinite(x);s=state['numeric'][k];cols.extend([(np.where(m,s['median'],x)-s['mean'])/s['sd'],m.astype(float)]);names.extend([k,k+'__missing'])
 for k,levels in state['categories'].items():
  vals=[str(r.get(k,'unknown')) for r in rows]
  if any(v not in levels for v in vals):raise ValueError('Categorical value outside locked dictionary')
  for lev in levels:cols.append(np.array([float(v==lev) for v in vals]));names.append(k+'='+lev)
 return np.column_stack(cols),names
def log_cpm(selected,library):
 x=np.asarray(selected);lib=np.asarray(library,float)
 if x.ndim!=2 or len(lib)!=len(x) or not np.isfinite(x).all() or (x<0).any() or not np.isfinite(lib).all() or (lib<=0).any():raise ValueError('Invalid count/library binding')
 return np.log2(1.+1e6*x/lib[:,None])
def prepare_rna(selected,library,panel,gate,omit_npipb8=False):
 COUNTERS['RNA_preparations']+=1;v=log_cpm(selected,library);mean=v.mean(axis=0);sd=v.std(axis=0);minimum=max(gate['minimum_training_samples'],math.ceil(gate['minimum_training_fraction']*len(v)));active=(np.sum(selected>=gate['minimum_raw_count'],axis=0)>=minimum)&(sd>gate['minimum_population_sd'])
 if omit_npipb8:active[panel['NPIPB8_index']]=False
 scale=np.where(sd>gate['minimum_population_sd'],sd,1.);z=(v-mean)/scale;z[:,~active]=0.;path=[]
 for ids in panel['pathway_indices']:
  use=[i for i in ids if active[i]];a=z[:,use].mean(axis=1) if len(use)>=5 else np.zeros(len(z));s=float(a.std());ok=len(use)>=5 and s>1e-12
  path.append({'eligible_indices':use,'coverage':len(use),'active':ok,'mean':float(a.mean()) if ok else 0.,'sd':s if ok else 1.})
 return {'mean':mean.tolist(),'sd':scale.tolist(),'active':active.tolist(),'minimum_training_samples_required':minimum,'gene5_indices':panel['gene5_indices'],'gene5_names':panel['gene5_names'],'pathway_names':panel['pathway_names'],'pathways':path,'omit_npipb8':omit_npipb8}
def transform_rna(selected,library,state,arm):
 v=log_cpm(selected,library);z=(v-np.array(state['mean']))/np.array(state['sd']);active=np.array(state['active'],bool);z[:,~active]=0.;cols=[];names=[]
 if arm in ['Genes5','Genes5_Hallmark11']:
  for i,n in zip(state['gene5_indices'],state['gene5_names']):cols.append(z[:,i]);names.append('gene:'+n)
 if arm in ['Hallmark11','Genes5_Hallmark11']:
  for n,s in zip(state['pathway_names'],state['pathways']):
   a=(z[:,s['eligible_indices']].mean(axis=1)-s['mean'])/s['sd'] if s['active'] else np.zeros(len(z));cols.append(a);names.append('pathway:'+n)
 if arm=='none':return np.empty((len(v),0)),[]
 if not cols:raise ValueError('Unknown RNA representation')
 return np.column_stack(cols),names
def solve(X,y,clinical_columns,lambda_c,lambda_r):
 COUNTERS['ridge_solves']+=1;X=np.asarray(X,float);y=np.asarray(y,float)
 if len(X)!=len(y) or len(y)==0 or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Invalid ridge data')
 penalty=np.full(X.shape[1],float(lambda_c));penalty[clinical_columns:]=float(lambda_r) if lambda_r is not None else float(lambda_c)
 if (penalty<=0).any():raise ValueError('Nonpositive penalty')
 xm=X.mean(axis=0);ym=float(y.mean());Z=X-xm;coef=np.linalg.solve(Z.T@Z+len(y)*np.diag(penalty),Z.T@(y-ym));intercept=float(ym-xm@coef)
 if not np.isfinite(coef).all() or not math.isfinite(intercept):raise ValueError('Nonfinite ridge solution')
 return coef,intercept
def metric(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if len(y)==0 or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid metric arrays')
 d=p-y;den=float(np.sum((y-y.mean())**2));pv=float(np.sum((p-p.mean())**2));slope=float(np.sum((p-p.mean())*(y-y.mean()))/pv) if pv>0 and float(np.ptp(p))>1e-12 else None
 return {'n':len(y),'MAE':float(np.abs(d).mean()),'RMSE':float(np.sqrt(np.mean(d*d))),'R2':float(1-np.sum(d*d)/den) if den>0 else None,'mean_prediction_error':float(d.mean()),'calibration_slope':slope,'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'fraction_predictions_outside_0_30':float(np.mean((p<0)|(p>30))),'clipped_MAE_0_30_descriptive':float(np.mean(np.abs(np.clip(p,0,30)-y)))}
def select_candidate(records):
 if not records:raise ValueError('Empty tuning candidates')
 losses=[x['pooled_MAE'] for x in records]
 if not np.isfinite(losses).all():raise ValueError('Invalid tuning loss')
 best=min(losses);ties=[x for x in records if x['pooled_MAE']<=best+1e-12]
 return min(ties,key=lambda x:(x['lambda_RNA'] is not None,-(x['lambda_RNA'] or 0),-x['lambda_clinical']))
def paired_interval(y,left,right,replicates,seed):
 y=np.asarray(y,float);a=np.asarray(left,float);b=np.asarray(right,float)
 if len(y)==0 or y.shape!=a.shape or a.shape!=b.shape or not np.isfinite(np.r_[y,a,b]).all():raise ValueError('Invalid paired inference')
 d=np.abs(a-y)-np.abs(b-y);rng=np.random.default_rng(seed);vals=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(replicates)]);lo,hi=np.percentile(vals,[2.5,97.5]);base=float(np.abs(b-y).mean())
 return {'delta_MAE':float(d.mean()),'conditional_lower95':float(lo),'conditional_upper95':float(hi),'relative_MAE_reduction':float(-d.mean()/base) if base>0 else None,'replicates':replicates,'conditional_on_fixed_OOF':True}
class RealBackend:
 def __init__(self,bundle):
  self.bundle=bundle;self.matrix=np.load(bundle['inputs']['rna_matrix'],mmap_mode='r');self.cache={};self.library={}
  for k,r in bundle['features'].items():self.library[k]=float(self.matrix[int(r['count_row'])].sum())
 def fit_predict(self,view,train_ids,target_ids,y_train,arm,lambda_c,lambda_r):
  b=self.bundle;key=(view['view_id'],tuple(train_ids));panel=b['panel'];spec=b['clinical_specs'][view['view_id']]
  def raw(ids):return self.matrix[np.array([int(b['features'][k]['count_row']) for k in ids])[:,None],np.array(panel['raw_columns'])[None,:]]
  if key not in self.cache:
   fr=[b['features'][k] for k in train_ids];cs=prepare_clinical(fr,spec);xc,cn=transform_clinical(fr,cs);rs=prepare_rna(raw(train_ids),[self.library[k] for k in train_ids],panel,b['plan']['RNA']['training_gene_gate'],view['omit_npipb8']);self.cache[key]=(cs,xc,cn,rs)
  cs,xc,cn,rs=self.cache[key];zc,_=transform_clinical([b['features'][k] for k in target_ids],cs)
  # RNA-off fits use clinical columns only; no hidden zero-penalty RNA block.
  if arm=='none' or lambda_r is None:xr=np.empty((len(train_ids),0));zr=np.empty((len(target_ids),0));rn=[]
  else:xr,rn=transform_rna(raw(train_ids),[self.library[k] for k in train_ids],rs,arm);zr,_=transform_rna(raw(target_ids),[self.library[k] for k in target_ids],rs,arm)
  coef,intercept=solve(np.column_stack([xc,xr]),y_train,xc.shape[1],lambda_c,lambda_r);pred=np.column_stack([zc,zr])@coef+intercept
  return pred,{'clinical_state':cs,'RNA_state':rs,'columns':cn+rn,'coef':coef.tolist(),'intercept':intercept,'lambda_clinical':lambda_c,'lambda_RNA':lambda_r,'train_ids':train_ids,'target_ids':target_ids,'RNA_off':lambda_r is None}