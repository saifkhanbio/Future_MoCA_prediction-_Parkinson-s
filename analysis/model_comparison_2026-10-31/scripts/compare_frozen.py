"""Frozen-prediction comparison only. No model fitting or new participant predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime,csv,collections,itertools
import numpy as np
A=Path(__file__).resolve().parents[1];C=A.parent
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
def read(p):
 p=Path(p);sources[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)};return load(p)
def tab(name,rows):
 with (A/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]),delimiter='\t');w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def metrics(y,p):
 e=p-y;den=np.sum((y-y.mean())**2);v=np.sum((p-p.mean())**2);slope=float(np.sum((p-p.mean())*(y-y.mean()))/v) if v>1e-20 else None
 return {'n':len(y),'R2':float(1-np.sum(e**2)/den),'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),
 'MSE':float(np.mean(e**2)),'mean_prediction_error':float(e.mean()),'calibration_slope':slope,
 'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'outside_0_30_fraction':float(np.mean((p<0)|(p>30)))}
def main():
 plan=load(A/'config/Comparison_Plan.json');models=plan['models']
 paths={'main':C/'executions/main_2026-10-28','mae':C/'ga_expanded_hybrid_run_2026-10-30',
 'r2':C/'ga_r2_run_2026-10-31','fixed':C/'feature_only_run_2026-10-31','new':C/'ga_moca_run_v2_2026-10-31'}
 contract=read(C/'ga_moca_readiness_v2_2026-10-31/config/Analysis_Contract.json')
 old=read(C/'ga_r2_readiness_2026-10-30/config/Analysis_Contract.json')
 oldmae=read(C/'ga_readiness_2026-10-30/config/Analysis_Contract.json')
 selections={k:read(paths[k]/'results/Selections.json') for k in ['mae','r2','new']}
 original=read(paths['main']/'restricted/Predictions.json')
 source_metrics={k:read(paths[k]/('Metrics.json' if k=='main' else 'results/Metrics.json')) for k in paths}
 output=[];contrasts=[];matching=[];duplicates=[];feature_rows=[];stability=[]
 for h in [12,24,36]:
  data={k:read(p/'restricted'/f'OOF_{h}m.json') for k,p in paths.items() if k!='main'}
  ref=data['new'];ids=[r['PATNO'] for r in ref];y=np.array([r['y'] for r in ref])
  assert len(ids)==len(set(ids))==int(plan['n_expected'][str(h)]) and len({r['component_id'] for r in ref})==len(ids)
  pred={}
  for m in models:
   rows=[r for r in original if r['view_id']==f'main_{h}_{m["view"]}'] if m['source']=='main' else data[m['source']]
   lookup={r['PATNO']:r for r in rows};assert len(lookup)==len(rows)==len(ids) and set(lookup)==set(ids)
   aligned=[lookup[i] for i in ids]
   assert all(all(a[k]==b[k] for k in ['component_id','outer_fold','y']) for a,b in zip(ref,aligned))
   p=np.array([r[m['arm']] for r in aligned]);assert np.isfinite(p).all();pred[m['model_id']]=p
   values=metrics(y,p)
   sm=next(r for r in source_metrics[m['source']] if (r['view_id']==f'main_{h}_{m["view"]}' and r['model']==m['arm']) if m['source']=='main') if m['source']=='main' else next(r for r in source_metrics[m['source']] if r['horizon']==h and r['arm']==m['arm'])
   for field in ['n','R2','MAE','RMSE','mean_prediction_error','calibration_slope','calibration_intercept']:
    assert np.isclose(values[field],sm[field],atol=1e-11,rtol=1e-11),(h,m['model_id'],field)
   output.append({'horizon':h,'model_id':m['model_id'],'label':m['label'],**values})
   matching.append({'horizon':h,'model_id':m['model_id'],'n':len(ids),'same_participants':True,'same_components':True,'same_folds':True,'same_outcomes':True})
   if m['fixed_substantive_predictor_count'] is not None:
    for o in range(5):feature_rows.append({'horizon':h,'outer':o,'model_id':m['model_id'],'substantive_predictors':m['fixed_substantive_predictor_count'],'optional_features':None,'extra_instruments_above_mandatory_block':None,'selected_optional_features':[]})
   else:
    selected=sorted([r for r in selections[m['source']] if r['horizon']==h and r['arm']==m['arm']],key=lambda r:r['outer'])
    assert len(selected)==5;base=1 if m['source']=='new' else 10
    sets=[set(r['features']) for r in selected]
    for r in selected:feature_rows.append({'horizon':h,'outer':r['outer'],'model_id':m['model_id'],'substantive_predictors':base+len(r['features']),'optional_features':len(r['features']),'extra_instruments_above_mandatory_block':r['instruments'],'selected_optional_features':r['features']})
    stability.append({'horizon':h,'model_id':m['model_id'],'mean_pairwise_Jaccard':float(np.mean([len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(sets,2)])),'features_all_folds':sorted(set.intersection(*sets)),'unique_masks':len({r['mask'] for r in selected})})
  for alias,k,arm,target in [('full_MAE','mae','full','core_plus18'),('full_fixed_reference','fixed','full','core_plus18'),('latest_full','new','full','MoCA_plus18'),('GA_MAE_core','mae','core','clinical_core_mae'),('GA_MAE_enhanced','mae','enhanced','enhanced_clinical_mae')]:
   lookup={r['PATNO']:r[arm] for r in data[k]};p=np.array([lookup[i] for i in ids])
   diff=float(np.max(abs(p-pred[target])));assert diff<1e-11
   duplicates.append({'horizon':h,'alias':alias,'canonical_model':target,'max_prediction_difference':diff,'not_independent_replication':True})
  rng=np.random.default_rng(20261030+h);ix=rng.integers(0,len(y),size=(5000,len(y)));yb=y[ix];den=np.sum((yb-yb.mean(1)[:,None])**2,axis=1);assert np.all(den>0)
  sst=np.sum((y-y.mean())**2)
  for pair in plan['contrasts']:
   a,b=pair['a'],pair['b'];ad=abs(pred[a]-y)-abs(pred[b]-y);sd=(pred[b]-y)**2-(pred[a]-y)**2
   mae=ad[ix].mean(1);r2=sd[ix].sum(1)/den
   ci={'horizon':h,'a':a,'b':b,'delta_R2':float(sd.sum()/sst),'R2_lower95':float(np.quantile(r2,.025)),'R2_upper95':float(np.quantile(r2,.975)),
    'delta_MAE':float(ad.mean()),'MAE_lower95':float(np.quantile(mae,.025)),'MAE_upper95':float(np.quantile(mae,.975)),
    'n':len(y),'replicates':5000,'seed':20261030+h,'conditional_on_frozen_predictions':True,'post_hoc_unadjusted':True}
   contrasts.append(ci)
  np.savez_compressed(A/'restricted'/f'Matched_{h}m.npz',y=y,**pred)
 # Freeze exact existing compact recipe, sources and coefficients without creating any new predictions.
 compact=[]
 for h in [12,24,36]:
  for o in range(5):
   path=paths['new']/f'restricted/{h}m_outer{o}/Result.json';r=read(path)['models']['enhanced']
   assert r['mask']==7 and r['features']==['SDMT','HVLT_immediate','HVLT_delayed']
   compact.append({'horizon':h,'outer':o,'source_path':str(path),'source_sha256':sources[str(path)]['sha256'],'model_key':'models.enhanced','alpha':r['alpha'],'substantive_features':['baseline_moca']+r['features'],'design_columns':r['columns'],'model_payload_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
 save(A/'config/Existing_Compact_Model_Binding.json',{'models':compact,'family':'linear ridge','selection':'Fixed features; lambda chosen by inner pooled MSE from .001,.01,.1,1,10','preprocessing':'Existing training-only median imputation, numeric standardization, missing indicators and form categories; no clipping','form_controls':{k:v for k,v in contract['features']['nuisance_categories'].items() if k in ['baseline_sdmt_form','baseline_hvlt_form']},'required_harmonization':'Education-adjusted baseline and outcome MoCA; education/source score rules and test-form metadata remain necessary even though education is not a regressor. Cohort/timing eligibility remains required.','deployment_status':'No single final model or new ensemble defined; these are the existing five outer fits per horizon. A future deployment contract must specify prediction aggregation and evaluation population before scoring.'})
 temporal=read(C/'temporal_validation_readiness_2026-10-28/executions/temporal_2026-10-28/results/Results.json')
 temprows=[]
 for h in [12,24,36]:
  for arm,entry in temporal[f'main_{h}']['models'].items():
   temprows.append({'horizon':h,'model':arm,'n':temporal[f'main_{h}']['n'],**entry['estimates'],'validation_scope':'Earlier original clinical-only ensembles on separate later PPMI participants; not validation of any GA, compact4 or core-plus18 model'})
 # Check every consumed result file against its original manifest where available.
 checks=[]
 for key,base in paths.items():
  manifest=base/('Output_Manifest.json' if key=='main' else 'Verified_Artifact_Manifest.json' if key=='fixed' else 'Artifact_Manifest.json')
  x=read(manifest)['files'];registry={r['path']:r for r in x} if isinstance(x,list) else x
  n=0
  for path,info in list(sources.items()):
   p=Path(path)
   if p.is_relative_to(base) and p!=manifest:
    rel=str(p.relative_to(base))
    if rel in registry:
     assert registry[rel]['sha256']==info['sha256'] and registry[rel]['bytes']==info['bytes'];n+=1
  checks.append({'source':key,'consumed_files_matching_frozen_manifest':n})
 featuredefinitions={'core':old['features']['core'],'optional18':contract['features']['optional'],'original_enhanced_additions':['SDMT','HVLT_immediate','HVLT_delayed'],'compact4':['baseline_moca','SDMT','HVLT_immediate','HVLT_delayed'],
 'GA_differences':{'prior_MAE':oldmae['search']['final_choice'],'prior_R2':old['search']['final_choice'],'new_MoCA':contract['search']['final_choice']},
 'important':'Prior GA requires10 core variables and uses heuristic one-SE parsimony; new GA requires only MoCA and uses accuracy-first selection. This is not a clean ablation of the core variables. Core-plus18 and MoCA-plus18 include APOE and CBC-derived NLR; they are not exclusively clinical-assessment models.'}
 for name,rows in [('Metrics',output),('Contrasts',contrasts),('Feature_Counts',feature_rows),('GA_Stability',stability),('Temporal_Context',temprows)]:
  save(A/'results'/f'{name}.json',rows);tab(name+'.tsv',rows)
 save(A/'config/Feature_Definitions.json',featuredefinitions)
 save(A/'verification/Matching.json',{'status':'PASS','rows':matching,'duplicate_recipe_checks':duplicates,'frozen_manifest_checks':checks})
 save(A/'evidence/Source_Bindings.json',sources)
 save(A/'results/COMPARISON_COMPLETE.json',{'status':'COMPLETE_PENDING_INDEPENDENT_CHECK','metric_rows':len(output),'contrast_rows':len(contrasts),'participant_fits':0,'new_patient_predictions':0,'paid_queries':0})
 print(json.dumps({'matched_metric_rows':len(output),'paired_contrasts':len(contrasts),'all_matching_checks_passed':True,'fits':0,
 'key_36m_contrasts':[r for r in contrasts if r['horizon']==36 and (r['a'],r['b']) in [('cognitive4','core_plus18'),('MoCA_GA','cognitive4'),('MoCA_GA','prior_GA_R2'),('MoCA_GA','prior_GA_MAE')]]}),flush=True)
if __name__=='__main__':main()
