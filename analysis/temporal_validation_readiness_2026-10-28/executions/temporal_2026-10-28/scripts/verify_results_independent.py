"""Independent endpoint, metric and bootstrap verification; no production inference/metric imports."""
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
import json,math,hashlib,csv
import numpy as np,pandas as pd
from scipy.stats import linregress
R=Path(__file__).resolve().parents[1];D=R.parents[1]
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f,delimiter='\t' if Path(p).suffix=='.tsv' else ','))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((D/'config/Temporal_Validation_Contract.json').read_text())
views=json.loads((D/'config/Evaluation_Views.json').read_text())
production=json.loads((R/'results/Results.json').read_text())
assert set(production)=={v['view_id'] for v in views}
pred=pd.DataFrame(json.loads((R/'restricted/Independent_Predictions.json').read_text()))
inputs=json.loads((D/'config/Source_Inputs.json').read_text())
raw=pd.read_csv(inputs['moca'],dtype=str,keep_default_na=False);assert raw.REC_ID.is_unique
raw['csv_row']=range(2,len(raw)+2);raw=raw.set_index('REC_ID')
features={r['PATNO']:r for r in read(D/'restricted/Target_Features_Resolved.tsv')}
itemmax=json.loads(Path(inputs['prior_cognitive_contract']).read_text())['cognition']['item_max']
outcomes={h:pd.DataFrame(read(D/f'restricted/Sealed_Outcomes_{h}m.tsv')).set_index('component_id') for h in [12,24,36]}
endpoint_checks=0
for h,df in outcomes.items():
 assert df.index.is_unique
 for cid,o in df.iterrows():
  r=raw.loc[o.source_REC_ID];b=features[o.PATNO]
  assert r.PATNO==o.PATNO and b['component_id']==cid and int(r.csv_row)==int(o.source_row)
  month,year=map(int,r.INFODT.split('/'));when=year*12+month;index=int(b['index_month'])
  assert when==int(o.assessment_month) and h-3<=when-index<=h+3
  values=[float(r[k]) for k in itemmax]
  assert all(math.isfinite(x) and x.is_integer() and 0<=x<=itemmax[k] for x,k in zip(values,itemmax))
  total=sum(values);education=float(b['education']);y=min(30,total+int(total<30 and education<=12))
  assert y==float(o.education_adjusted_MoCA) and total==float(o.raw_MoCA)
  if r.MCATOT:assert float(r.MCATOT) in [total,y]
  if r.MCAVFNUM:assert int(float(r.MCAVF))==int(float(r.MCAVFNUM)>=11)
  assert o.endpoint_QC_pass=='True';endpoint_checks+=1
names=['core','enhanced','persistence','source_typical_change']
quantkeys=['MAE','RMSE','R2','mean_prediction_error','calibration_intercept','calibration_slope']
metric_checks=0;max_diff=0.;compared_numbers=0
def close(expected,actual,path):
 global compared_numbers,max_diff
 if expected is None or actual is None:
  assert expected is None and actual is None,path;return
 assert math.isfinite(float(expected)) and math.isfinite(float(actual)),path
 diff=abs(float(expected)-float(actual));max_diff=max(max_diff,diff);compared_numbers+=1
 assert diff<=1e-8+1e-10*abs(float(expected)),path
def point(y,p):
 n=len(y);errors=[float(p[i]-y[i]) for i in range(n)];mae=math.fsum(abs(e) for e in errors)/n
 sse=math.fsum(e*e for e in errors);ym=math.fsum(map(float,y))/n;sst=math.fsum((float(v)-ym)**2 for v in y)
 if np.ptp(p)>1e-12:
  lr=linregress(p,y);slope=float(lr.slope);intercept=float(lr.intercept)
 else:slope=intercept=None
 return {'MAE':mae,'RMSE':math.sqrt(sse/n),'R2':1-sse/sst if sst>0 else None,
 'mean_prediction_error':math.fsum(errors)/n,'calibration_intercept':intercept,'calibration_slope':slope,
 'fraction_predictions_outside_0_30':sum(float(v)<0 or float(v)>30 for v in p)/n,
 'clipped_MAE_0_30_descriptive':math.fsum(abs(min(30,max(0,float(v)))-float(t)) for v,t in zip(p,y))/n}
def ci(values,B):
 good=values[np.isfinite(values)]
 bounds=np.quantile(good,[.025,.975],method='linear') if len(good)>=.95*B else [None,None]
 return {'lower':None if bounds[0] is None else float(bounds[0]),'upper':None if bounds[1] is None else float(bounds[1]),
 'valid_replicates':len(good),'undefined_replicates':B-len(good)}
answers={};details=[]
for v in views:
 name=v['view_id'];h=v['horizon_months'];result=production[name]
 if v['reuse_alias']:
  assert result['reuse_alias']==v['reuse_alias'] and result['result']==production[v['reuse_alias']]
  answers[name]={'reuse_alias':v['reuse_alias'],'result':answers[v['reuse_alias']]};continue
 members=read(D/v['member_manifest']);ids=sorted(r['component_id'] for r in members)
 pp=pred[(pred.horizon_months==h)&(pred.feature_variant==v['feature_variant'])].set_index('component_id')
 assert pp.index.is_unique and set(ids)<=set(pp.index)
 pp=pp.loc[ids];yy=outcomes[h].loc[ids]
 assert pp.PATNO.tolist()==yy.PATNO.tolist() and len(ids)==v['expected_n']==result['n']
 y=yy.education_adjusted_MoCA.astype(float).to_numpy();P={n:pp[n].to_numpy(float) for n in names}
 B=contract['metrics']['uncertainty']['replicates'];assert B==5000 and result['replicates']==B and result['seed']==v['bootstrap_seed']
 idx=np.random.default_rng(v['bootstrap_seed']).integers(0,len(y),size=(B,len(y)))
 Y=y[idx];ym=Y.mean(axis=1,keepdims=True);yc=Y-ym;sst=(yc**2).sum(axis=1)
 all_draws={};stats={}
 for arm,p in P.items():
  estimates=point(y,p);V=p[idx];E=V-Y;pm=V.mean(axis=1,keepdims=True);pc=V-pm;var=(pc**2).sum(axis=1)
  with np.errstate(invalid='ignore',divide='ignore'):
   r2=1-(E**2).sum(axis=1)/sst;r2[sst==0]=np.nan
   slope=(pc*yc).sum(axis=1)/var;slope[(np.ptp(V,axis=1)<=1e-12)|(var==0)]=np.nan
  draws={'MAE':abs(E).mean(axis=1),'RMSE':np.sqrt((E**2).mean(axis=1)),'R2':r2,
   'mean_prediction_error':E.mean(axis=1),'calibration_slope':slope,'calibration_intercept':ym[:,0]-slope*pm[:,0]}
  stats[arm]={'estimates':estimates,'intervals':{k:ci(a,B) for k,a in draws.items()}}
  all_draws[arm]=draws
  assert set(result['models'][arm]['estimates'])==set(estimates)
  assert set(result['models'][arm]['intervals'])==set(quantkeys)
  for k,value in estimates.items():close(value,result['models'][arm]['estimates'][k],name+'/'+arm+'/'+k);metric_checks+=1
  for k,interval in stats[arm]['intervals'].items():
   for field,val in interval.items():close(val,result['models'][arm]['intervals'][k][field],name+'/'+arm+'/'+k+'/'+field);metric_checks+=1
 contrasts={}
 for contrast in v['contrast_definitions']:
  a,b=contrast.split('-minus-');estimate=stats[a]['estimates']['MAE']-stats[b]['estimates']['MAE']
  interval=ci(all_draws[a]['MAE']-all_draws[b]['MAE'],B)
  contrasts[contrast]={'delta_MAE':estimate,'interval':interval}
  close(estimate,result['contrasts'][contrast]['delta_MAE'],name+'/'+contrast);metric_checks+=1
  for field,value in interval.items():close(value,result['contrasts'][contrast]['interval'][field],name+'/'+contrast+'/'+field);metric_checks+=1
 assert set(contrasts)==set(result['contrasts'])
 answers[name]={'n':len(ids),'replicates':B,'seed':v['bootstrap_seed'],'models':stats,'contrasts':contrasts}
 details.append({'view_id':name,'n':len(ids),'status':'PASS','paired_draws_verified':B})
with (R/'qa/Independent_Results.json').open('x') as f:json.dump(answers,f,indent=2,allow_nan=False)
with (R/'qa/View_Verification.json').open('x') as f:json.dump(details,f,indent=2)
summary={'status':'PASS','implementation':'Independent scalar/SciPy point metrics and vectorised paired bootstrap; no production launcher or metric imports.',
 'raw_endpoint_bindings_verified':endpoint_checks,'reported_views':len(views),'independently_recomputed_unique_views':len(details),
 'paired_replicates_per_view':5000,'total_paired_replicates_verified':len(details)*5000,
 'metric_fields_checked':metric_checks,'finite_numeric_comparisons':compared_numbers,'max_absolute_metric_difference':max_diff,
 'production_results_sha256':sha(R/'results/Results.json'),'production_prediction_sha256':sha(R/'restricted/predictions/Predictions.json'),
 'all_locked_results_verified':True,'models_fit':0,'paid_queries':0}
with (R/'qa/Independent_Result_Verification.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
# Only aggregate verified summaries, never participant records.
for h in [12,24,36]:
 r=production[f'main_{h}']
 print(json.dumps({'horizon':h,'n':r['n'],'models':{n:r['models'][n] for n in names},'contrasts':r['contrasts']}))
