"""Frozen OOF comparison. No estimator fits, training imports, network calls or new patient predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,csv,itertools,datetime
import numpy as np
O=Path(__file__).resolve().parents[1];C=O.parent
D=C/'model_comparison_2026-10-31'
V=C/'ga_unrestricted_verification_v4_2026-11-01'
A=C/'ga_unrestricted_readiness_v4_10kgen_2026-11-01'
paths={'main':C/'executions/main_2026-10-28','mae':C/'ga_expanded_hybrid_run_2026-10-30','r2':C/'ga_r2_run_2026-10-31','fixed':C/'feature_only_run_2026-10-31','new':C/'ga_moca_run_v2_2026-10-31','v4':C/'ga_unrestricted_run_v4_10kgen_2026-11-01'}
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes();sources[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
def save(rel,x):(O/rel).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tab(name,rows):
 with (O/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]),delimiter='\t');w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def metrics(y,p):
 e=p-y;yc=y-y.mean();pc=p-p.mean();den=yc@yc;slope=float(pc@yc/(pc@pc)) if pc@pc>1e-20 else None
 return {'n':len(y),'R2':float(1-e@e/den),'MAE':float(np.abs(e).mean()),'MSE':float((e*e).mean()),'RMSE':float(np.sqrt((e*e).mean())),'mean_prediction_error':float(e.mean()),'calibration_slope':slope,'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'outside_0_30_fraction':float(((p<0)|(p>30)).mean())}
def main():
 plan=read(O/'config/Comparison_Plan.json');models=plan['models'];ids=[m['model_id'] for m in models]
 oldplan=read(D/'config/Comparison_Plan.json');oldmetrics=read(D/'results/Metrics.json')
 definitions=read(D/'config/Feature_Definitions.json');oldcounts=read(D/'results/Feature_Counts.json')
 vsummary=read(V/'evidence/Verification_Summary.json');vseal=read(V/'VERIFICATION_SEAL.json')
 assert vseal['status']=='PASS_WITH_DOCUMENTED_AUDIT_LIMITATION'
 assert sha(V/'Artifact_Manifest.json')==vseal['artifact_manifest_sha256']
 vr=read(paths['v4']/'results/Metrics.json');vc=read(paths['v4']/'results/Contrasts.json')
 selections=read(paths['v4']/'results/Selections.json');contract=read(A/'config/Analysis_Contract.json')
 groups={r['name']:r['instrument'] for r in contract['features']['optional']}
 core=definitions['core']['numeric']+list(definitions['core']['categories'])
 optional=[r['name'] for r in definitions['optional18']]
 fixed={'clinical_core_mae':core,'clinical_core_mse':core,'enhanced_clinical_mae':core+definitions['original_enhanced_additions'],'enhanced_clinical_mse':core+definitions['original_enhanced_additions'],'core_plus18':core+optional,'MoCA_plus18':['baseline_moca']+optional,'optional18_only':optional,'cognitive4':definitions['compact4'],'MoCA_ridge':['baseline_moca'],'persistence':['baseline_moca'],'typical_change':['baseline_moca']}
 native={'unrestricted28_GA_selected':'v4_unrestricted_GA','moca27_GA_selected':'v4_MoCA_GA','unrestricted28_random_selected':'v4_unrestricted_random','moca27_random_selected':'v4_MoCA_random','core_plus18':'core_plus18','MoCA_plus18':'MoCA_plus18','compact4':'cognitive4','prior_MoCA_GA':'MoCA_GA','persistence':'persistence','typical_change':'typical_change'}
 original=read(paths['main']/'restricted/Predictions.json')
 out=[];contrasts=[];matching=[];feature_rows=[];stability=[];frequencies=[];native_errors=[];metric_errors=[];identical=[]
 for h in [12,24,36]:
  data={k:read(p/'restricted'/f'OOF_{h}m.json') for k,p in paths.items() if k!='main'}
  reference=data['v4'];people=[r['PATNO'] for r in reference];y=np.array([r['y'] for r in reference]);pred={}
  assert len(people)==len(set(people))==len({r['component_id'] for r in reference})==plan['participants'][str(h)]
  for m in models:
   rows=[r for r in original if r['view_id']==f'main_{h}_{m["view"]}'] if m['source']=='main' else data[m['source']]
   lookup={r['PATNO']:r for r in rows};assert len(lookup)==len(rows)==len(people) and set(lookup)==set(people)
   aligned=[lookup[i] for i in people];assert all(all(a[k]==b[k] for k in ['component_id','outer_fold','y']) for a,b in zip(reference,aligned))
   p=np.array([r[m['arm']] for r in aligned],dtype=float);assert np.isfinite(p).all();pred[m['model_id']]=p
   val=metrics(y,p);out.append({'horizon':h,'model_id':m['model_id'],'label':m['label'],**val})
   frozen=next(r for r in vr if r['horizon']==h and r['arm']==m['arm']) if m['source']=='v4' else next(r for r in oldmetrics if r['horizon']==h and r['model_id']==m['model_id'])
   for field in ['R2','MAE','RMSE','mean_prediction_error','calibration_slope','calibration_intercept']:
    error=abs(val[field]-frozen[field]);assert error<1e-10;metric_errors.append(error)
   matching.append({'horizon':h,'model_id':m['model_id'],'n':len(people),'identities_components_folds_outcomes_match':True})
   sets=[]
   for fold in range(5):
    if m['model_id'] in fixed:features=fixed[m['model_id']];alpha=None
    elif m['source']=='v4':
     sel=next(r for r in selections if r['horizon']==h and r['outer']==fold and r['arm']==m['arm'])
     features=sel['features'];alpha=sel['alpha']
     assert len(features)==sel['feature_count'] and len({groups[f] for f in features})==sel['instruments']
    else:
     sel=next(r for r in oldcounts if r['horizon']==h and r['outer']==fold and r['model_id']==m['model_id'])
     base=['baseline_moca'] if m['model_id']=='MoCA_GA' else core
     features=base+sel['selected_optional_features'];alpha=None
     assert len(features)==sel['substantive_predictors']
    assert len(features)==len(set(features));sets.append(set(features))
    feature_rows.append({'horizon':h,'outer_fold':fold,'model_id':m['model_id'],'feature_count':len(features),'total_acquisition_groups':len({groups[f] for f in features}),'features':features,'acquisition_groups':sorted({groups[f] for f in features}),'alpha_if_v4':alpha})
   stability.append({'horizon':h,'model_id':m['model_id'],'unique_feature_sets':len({tuple(sorted(x)) for x in sets}),'mean_pairwise_Jaccard':float(np.mean([len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)])),'features_in_all5':sorted(set.intersection(*sets))})
   for f in groups:frequencies.append({'horizon':h,'model_id':m['model_id'],'feature':f,'selected_folds':sum(f in s for s in sets),'total_folds':5})
  for a,b in itertools.combinations(ids,2):
   diff=float(np.max(np.abs(pred[a]-pred[b])))
   if diff<1e-12:identical.append({'horizon':h,'a':a,'b':b,'max_prediction_difference':diff})
  rng=np.random.default_rng(20261031+h);ix=rng.integers(0,len(y),size=(5000,len(y)));yb=y[ix];den=((yb-yb.mean(1)[:,None])**2).sum(1);assert np.all(den>0)
  boot_r2={};boot_mae={}
  for model,p in pred.items():
   err=p-y;boot_r2[model]=1-(err*err)[ix].sum(1)/den;boot_mae[model]=np.abs(err)[ix].mean(1)
  metric_lookup={r['model_id']:r for r in out if r['horizon']==h}
  for a,b in itertools.combinations(ids,2):
   r2=boot_r2[a]-boot_r2[b];mae=boot_mae[a]-boot_mae[b]
   contrasts.append({'horizon':h,'a':a,'b':b,'delta_R2':metric_lookup[a]['R2']-metric_lookup[b]['R2'],'R2_lower95':float(np.quantile(r2,.025)),'R2_upper95':float(np.quantile(r2,.975)),'delta_MAE':metric_lookup[a]['MAE']-metric_lookup[b]['MAE'],'MAE_lower95':float(np.quantile(mae,.025)),'MAE_upper95':float(np.quantile(mae,.975)),'n':len(y),'replicates':5000,'seed':20261031+h,'post_hoc_unadjusted':True,'conditional_on_fixed_predictions':True})
  np.savez_compressed(O/'restricted'/f'Matched_{h}m.npz',PATNO=np.array(people),component_id=np.array([r['component_id'] for r in reference]),outer_fold=np.array([r['outer_fold'] for r in reference]),y=y,**pred)
 for x in vc:
  a,b=native[x['arm_a']],native[x['arm_b']]
  z=next(r for r in contrasts if r['horizon']==x['horizon'] and {r['a'],r['b']}=={a,b});sgn=1 if z['a']==a else -1
  expected={'delta_R2':sgn*z['delta_R2'],'delta_MAE':sgn*z['delta_MAE'],'lower95_R2':z['R2_lower95'] if sgn==1 else -z['R2_upper95'],'upper95_R2':z['R2_upper95'] if sgn==1 else -z['R2_lower95'],'lower95_MAE':z['MAE_lower95'] if sgn==1 else -z['MAE_upper95'],'upper95_MAE':z['MAE_upper95'] if sgn==1 else -z['MAE_lower95']}
  for field,value in expected.items():
   err=abs(x[field]-value);assert err<1e-10;native_errors.append(err)
 # Check consumed inputs against the sealed prior comparison, verification and training manifests.
 registries={}
 for base,manifest in [(D,D/'Artifact_Manifest.json'),(V,V/'Artifact_Manifest.json')]+[(p,p/('Output_Manifest.json' if k=='main' else 'Verified_Artifact_Manifest.json' if k=='fixed' else 'Artifact_Manifest.json')) for k,p in paths.items() if k!='v4']:
  mf=read(manifest)['files'];records=mf if isinstance(mf,dict) else {r['path']:r for r in mf}
  for rel,info in records.items():registries[str(base/rel)]=info
 registries.update(read(V/'restricted/Pre_Verification_Source_Manifest.json'))
 # Prior comparison binds all historical inputs and compact model payloads.
 registries.update(read(D/'evidence/Source_Bindings.json'))
 checks=[]
 for path,info in sources.items():
  if path in registries:
   frozen=registries[path];assert info['sha256']==frozen['sha256'] and info['bytes']==frozen['bytes'],path
   checks.append({'path':path,'matches_prior_seal':True})
 for name,rows in [('Metrics',out),('Contrasts',contrasts),('Feature_Sets',feature_rows),('Feature_Frequencies',frequencies),('Stability',stability)]:
  save(f'results/{name}.json',rows);tab(name+'.tsv',rows)
 ranking=[]
 for h in [12,24,36]:
  for metric,reverse in [('R2',True),('MAE',False)]:
   rr=sorted([r for r in out if r['horizon']==h],key=lambda x:x[metric],reverse=reverse)
   for rank,r in enumerate(rr,1):ranking.append({'horizon':h,'ranking_metric':metric,'rank_with_ties_ordered_for_display':rank,'model_id':r['model_id'],'value':r[metric],'descriptive_only':True})
 save('results/Rankings.json',ranking);tab('Rankings.tsv',ranking)
 save('config/Feature_Definitions.json',{'core':core,'optional18':optional,'compact4':fixed['cognitive4'],'acquisition_group_mapping':groups,'metadata_not_counted':'Education adjustment and form controls remain necessary. Counts concern substantive predictors and source groups, not encoded columns or measured minutes.','GA_recipe_caveat':definitions['GA_differences']})
 save('verification/Matching_And_Seals.json',{'status':'PASS','matches':matching,'sealed_inputs_checked':checks,'frozen_metric_checks':len(out),'max_metric_error':max(metric_errors),'v4_locked_contrasts_reproduced':len(vc),'max_native_contrast_error':max(native_errors),'identical_prediction_pairs':identical})
 save('evidence/V4_Verification_Context.json',vsummary)
 save('evidence/Source_Bindings.json',sources)
 save('results/COMPARISON_STATUS.json',{'status':'COMPUTED_PENDING_INDEPENDENT_CHECK','metric_rows':len(out),'contrast_rows':len(contrasts),'feature_set_rows':len(feature_rows),'training_fits':0,'new_patient_predictions':0,'paid_queries':0})
 print(json.dumps({'metric_rows':len(out),'contrast_rows':len(contrasts),'sealed_inputs_verified':len(checks),'max_metric_difference':max(metric_errors),'max_native_contrast_difference':max(native_errors),'frozen_results_unchanged':True}))
if __name__=='__main__':main()
