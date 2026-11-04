"""Clinical-only inference and metrics. No fit/tuning functions, RNA access or network."""
import math,numpy as np
def number(value):
 if value is None or str(value).strip()=='':return None
 try:x=float(value)
 except (ValueError,TypeError):raise ValueError('Malformed numeric feature')
 if not math.isfinite(x):raise ValueError('Nonfinite supplied numeric feature')
 return x

def validate_model(model,spec=None):
 s=model['state'];c=s['clinical_state']
 if model.get('qa_only') or model.get('arm')!='none' or s.get('RNA_off') is not True or s.get('lambda_RNA') is not None or model['selected'].get('lambda_RNA') is not None:raise ValueError('Not a frozen production clinical-only model')
 order=c['order'];cats=c['categories']
 if len(set(order))!=len(order) or set(order)!=set(c['numeric']):raise ValueError('Numeric schema mismatch')
 if spec and (order!=spec['numeric'] or cats!=spec['categories']):raise ValueError('Comparator specification mismatch')
 columns=[]
 for k in order:
  t=c['numeric'][k]
  if not all(isinstance(t[a],(int,float)) and math.isfinite(t[a]) for a in ['median','mean','sd']) or t['sd']<=0:raise ValueError('Invalid saved preprocessing')
  columns += [k,k+'__missing']
 for k,levels in cats.items():
  if not levels or len(set(levels))!=len(levels):raise ValueError('Invalid categorical dictionary')
  columns += [k+'='+v for v in levels]
 if columns!=s['columns'] or len(columns)!=len(s['coef']):raise ValueError('Saved coefficient/column mismatch')
 if not all(math.isfinite(float(x)) for x in s['coef']+[s['intercept']]):raise ValueError('Nonfinite saved parameter')
 return s,c

def design(rows,model,spec=None):
 s,c=validate_model(model,spec);z=[]
 for row in rows:
  vals=[]
  for k in c['order']:
   if k not in row:raise ValueError('Missing required feature column')
   x=number(row[k]);q=c['numeric'][k]
   vals.extend([((q['median'] if x is None else x)-q['mean'])/q['sd'],float(x is None)])
  for k,levels in c['categories'].items():
   if k not in row or str(row[k]) not in levels:raise ValueError('Categorical value outside locked dictionary')
   vals.extend(float(str(row[k])==v) for v in levels)
  z.append(vals)
 a=np.asarray(z,dtype=float).reshape(len(rows),len(s['columns']))
 if not np.isfinite(a).all():raise ValueError('Nonfinite design')
 return a

def predict_member(rows,model,spec=None):
 s,_=validate_model(model,spec);a=design(rows,model,spec)
 result=a @ np.asarray(s['coef'],float)+float(s['intercept'])
 if not np.isfinite(result).all():raise ValueError('Nonfinite prediction')
 return result

def predict_ensemble(rows,members,spec=None):
 if len(members)!=5 or {m['outer_fold'] for m in members}!=set(range(5)):raise ValueError('Missing or duplicate ensemble member')
 m=sorted(members,key=lambda x:x['outer_fold'])
 p=np.column_stack([predict_member(rows,x,spec) for x in m])
 return p.mean(axis=1),p

def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if y.ndim!=1 or y.shape!=p.shape or len(y)==0 or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid metric arrays')
 err=p-y;xc=p-p.mean();yc=y-y.mean();ss=float(yc@yc);vx=float(xc@xc)
 slope=float(xc@yc/vx) if np.ptp(p)>1e-12 and vx>0 else None
 return {'MAE':float(np.mean(np.abs(err))),'RMSE':float(np.sqrt(np.mean(err**2))),
 'R2':1-float(err@err)/ss if ss>0 else None,'mean_prediction_error':float(err.mean()),
 'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'calibration_slope':slope,
 'fraction_predictions_outside_0_30':float(np.mean((p<0)|(p>30))),
 'clipped_MAE_0_30_descriptive':float(np.mean(np.abs(np.clip(p,0,30)-y)))}

def bootstrap(y,predictions,seed,replicates=5000):
 names=['core','enhanced','persistence','source_typical_change']
 if set(predictions)!=set(names):raise ValueError('Prediction arm coverage mismatch')
 y=np.asarray(y,float);arrays={n:np.asarray(predictions[n],float) for n in names}
 for a in arrays.values():metrics(y,a)
 points={n:metrics(y,arrays[n]) for n in names}
 contrasts={'enhanced-minus-core':('enhanced','core'),'core-minus-persistence':('core','persistence'),
 'enhanced-minus-persistence':('enhanced','persistence'),'core-minus-source_typical_change':('core','source_typical_change'),
 'enhanced-minus-source_typical_change':('enhanced','source_typical_change')}
 keys=['MAE','RMSE','R2','mean_prediction_error','calibration_intercept','calibration_slope']
 draws={n:{k:[] for k in keys} for n in names};diffs={c:[] for c in contrasts}
 rng=np.random.default_rng(seed)
 for _ in range(replicates):
  ii=rng.integers(0,len(y),len(y));m={n:metrics(y[ii],a[ii]) for n,a in arrays.items()}
  for n in names:
   for k in keys:draws[n][k].append(m[n][k])
  for c,(a,b) in contrasts.items():diffs[c].append(m[a]['MAE']-m[b]['MAE'])
 def interval(values):
  good=[v for v in values if v is not None and math.isfinite(v)]
  return {'lower':float(np.percentile(good,2.5,method='linear')) if len(good)>=0.95*replicates else None,
   'upper':float(np.percentile(good,97.5,method='linear')) if len(good)>=0.95*replicates else None,
   'valid_replicates':len(good),'undefined_replicates':replicates-len(good)}
 return {'n':len(y),'replicates':replicates,'seed':seed,'models':{n:{'estimates':points[n],'intervals':{k:interval(v) for k,v in draws[n].items()}} for n in names},
 'contrasts':{c:{'delta_MAE':points[a]['MAE']-points[b]['MAE'],'interval':interval(diffs[c])} for c,(a,b) in contrasts.items()}}

def bind_outcomes(prediction_rows,outcome_rows):
 def keyed(rows):
  keys=[r['component_id'] for r in rows]
  if len(set(keys))!=len(keys):raise ValueError('Duplicate identity')
  return dict(zip(keys,rows))
 a=keyed(prediction_rows);b=keyed(outcome_rows)
 if set(a)!=set(b):raise ValueError('Outcome identity coverage mismatch')
 output=[]
 for key in sorted(a):
  r=a[key];o=b[key]
  if str(r['PATNO'])!=str(o['PATNO']) or int(r['horizon_months'])!=int(o['horizon_months']):raise ValueError('Outcome identity/horizon mismatch')
  if str(o['endpoint_QC_pass']).lower()!='true':raise ValueError('Unapproved outcome')
  y=number(o['education_adjusted_MoCA'])
  if y is None or not 0<=y<=30:raise ValueError('Invalid outcome value')
  output.append((r,y))
 return output
