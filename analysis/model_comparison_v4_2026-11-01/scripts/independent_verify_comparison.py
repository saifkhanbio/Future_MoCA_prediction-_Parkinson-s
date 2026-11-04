"""Independent scalar-metric and integer-weight bootstrap verification. Does not import comparison/training code."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,math,hashlib,gzip,datetime
import numpy as np
O=Path(__file__).resolve().parents[1];C=O.parent
audit_bindings={}
def read(p):
 b=p.read_bytes();audit_bindings[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
plan=read(O/'config/Comparison_Plan.json');models=plan['models']
paths={'main':C/'executions/main_2026-10-28','mae':C/'ga_expanded_hybrid_run_2026-10-30','r2':C/'ga_r2_run_2026-10-31','fixed':C/'feature_only_run_2026-10-31','new':C/'ga_moca_run_v2_2026-10-31','v4':C/'ga_unrestricted_run_v4_10kgen_2026-11-01'}
original=read(paths['main']/'restricted/Predictions.json')
reported=read(O/'results/Metrics.json');pairs=read(O/'results/Contrasts.json')
errors=[];paired_errors=[];verified_metrics=0;verified_pairs=0
for h in [12,24,36]:
 rows={k:read(p/'restricted'/f'OOF_{h}m.json') for k,p in paths.items() if k!='main'}
 ref=rows['v4'];people=[r['PATNO'] for r in ref];actual=[float(r['y']) for r in ref];n=len(actual);mean=math.fsum(actual)/n
 centered=[y-mean for y in actual];sst=math.fsum(v*v for v in centered);predictions={};stat={}
 with np.load(O/'restricted'/f'Matched_{h}m.npz',allow_pickle=False) as archive:
  assert archive['PATNO'].tolist()==people
  assert archive['component_id'].tolist()==[r['component_id'] for r in ref]
  assert archive['outer_fold'].tolist()==[r['outer_fold'] for r in ref]
  assert archive['y'].tolist()==actual
  for m in models:
   source=[r for r in original if r['view_id']==f'main_{h}_{m["view"]}'] if m['source']=='main' else rows[m['source']]
   lookup={r['PATNO']:r for r in source}
   assert len(lookup)==n and set(lookup)==set(people)
   aligned=[lookup[person] for person in people]
   for a,b in zip(ref,aligned):
    assert (a['component_id'],a['outer_fold'],a['y'])==(b['component_id'],b['outer_fold'],b['y'])
   pp=[float(r[m['arm']]) for r in aligned];assert archive[m['model_id']].tolist()==pp
   residual=[p-y for p,y in zip(pp,actual)];mse=math.fsum(e*e for e in residual)/n
   pm=math.fsum(pp)/n;vp=math.fsum((p-pm)**2 for p in pp)
   slope=math.fsum((p-pm)*(y-mean) for p,y in zip(pp,actual))/vp
   val={'n':n,'R2':1-mse*n/sst,'MAE':math.fsum(abs(e) for e in residual)/n,'MSE':mse,'RMSE':math.sqrt(mse),'mean_prediction_error':math.fsum(residual)/n,'calibration_slope':slope,'calibration_intercept':mean-slope*pm,'outside_0_30_fraction':sum(p<0 or p>30 for p in pp)/n}
   target=next(r for r in reported if r['horizon']==h and r['model_id']==m['model_id'])
   for key,value in val.items():errors.append(abs(value-target[key]));assert errors[-1]<1e-10
   stat[m['model_id']]=val;predictions[m['model_id']]=np.array(pp);verified_metrics+=1
 rng=np.random.default_rng(20261031+h)
 weights=np.vstack([np.bincount(rng.integers(0,n,size=n),minlength=n) for _ in range(5000)])
 y=np.array(actual);sy=weights@y;den=weights@(y*y)-sy*sy/n;assert (den>0).all()
 r2boot={};maeboot={}
 for key,p in predictions.items():
  e=p-y;r2boot[key]=1-weights@(e*e)/den;maeboot[key]=(weights@np.abs(e))/n
 for row in [r for r in pairs if r['horizon']==h]:
  a,b=row['a'],row['b'];rs=r2boot[a]-r2boot[b];ms=maeboot[a]-maeboot[b]
  check={'delta_R2':stat[a]['R2']-stat[b]['R2'],'delta_MAE':stat[a]['MAE']-stat[b]['MAE'],'R2_lower95':float(np.percentile(rs,2.5)),'R2_upper95':float(np.percentile(rs,97.5)),'MAE_lower95':float(np.percentile(ms,2.5)),'MAE_upper95':float(np.percentile(ms,97.5))}
  for key,value in check.items():paired_errors.append(abs(row[key]-value));assert paired_errors[-1]<1e-10,(h,a,b,key)
  verified_pairs+=1
# Independently reconstruct each fold's substantive feature set from original selections.
definition=read(C/'model_comparison_2026-10-31/config/Feature_Definitions.json')
core=definition['core']['numeric']+list(definition['core']['categories']);optional=[r['name'] for r in definition['optional18']]
v4contract=read(C/'ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/Analysis_Contract.json')
groups={r['name']:r['instrument'] for r in v4contract['features']['optional']}
selected={key:read(paths[key]/'results/Selections.json') for key in ['mae','r2','new','v4']}
fixed={'clinical_core_mae':core,'clinical_core_mse':core,'enhanced_clinical_mae':core+definition['original_enhanced_additions'],'enhanced_clinical_mse':core+definition['original_enhanced_additions'],'core_plus18':core+optional,'MoCA_plus18':['baseline_moca']+optional,'optional18_only':optional,'cognitive4':definition['compact4'],'MoCA_ridge':['baseline_moca'],'persistence':['baseline_moca'],'typical_change':['baseline_moca']}
sets=read(O/'results/Feature_Sets.json')
for row in sets:
 model=next(m for m in models if m['model_id']==row['model_id'])
 if row['model_id'] in fixed:features=fixed[row['model_id']]
 else:
  r=next(r for r in selected[model['source']] if r['horizon']==row['horizon'] and r['outer']==row['outer_fold'] and r['arm']==model['arm'])
  features=r['features'] if model['source']=='v4' else (['baseline_moca'] if model['source']=='new' else core)+r['features']
 assert row['features']==features and row['feature_count']==len(features)
 assert row['total_acquisition_groups']==len({groups[f] for f in features})
for row in read(O/'results/Feature_Frequencies.json'):
 assert row['selected_folds']==sum(row['feature'] in s['features'] for s in sets if s['horizon']==row['horizon'] and s['model_id']==row['model_id'])
# Check all consumed inputs unchanged, and all prior Clinical_hallmark files by metadata.
bindings=read(O/'evidence/Source_Bindings.json');sourcechanges=[]
for path,info in bindings.items():
 p=Path(path)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256'] or p.stat().st_size!=info['bytes']:sourcechanges.append(path)
assert not sourcechanges
with gzip.open(O/'restricted/Preexisting_Metadata.json.gz','rt') as f:before=json.load(f)
changed=[];missing=[]
for rel,info in before.items():
 p=C/rel
 if not p.exists():missing.append(rel)
 elif p.stat().st_size!=info['bytes'] or p.stat().st_mtime_ns!=info['mtime_ns']:changed.append(rel)
assert not changed and not missing

prior_binding=read(C/'model_comparison_2026-10-31/evidence/Source_Bindings.json')
selection_binding_checks=0
for key in ['mae','r2','new']:
 p=paths[key]/'results/Selections.json'
 assert audit_bindings[str(p)]['sha256']==prior_binding[str(p)]['sha256'];selection_binding_checks+=1
(O/'verification/Independent_Source_Bindings.json').write_text(json.dumps(audit_bindings,indent=2)+'\n')

summary={'status':'PASS','method':'Independent re-read of source OOF files, scalar math.fsum metrics, integer-weight paired bootstrap, original selection reconstruction; no comparison/training-code imports.',
 'metric_rows_verified':verified_metrics,'paired_contrasts_verified':verified_pairs,'feature_sets_verified':len(sets),'historical_selection_seals_checked':selection_binding_checks,'max_metric_difference':max(errors),'max_contrast_difference':max(paired_errors),'source_files_rehashed':len(bindings),'preexisting_files_metadata_unchanged':len(before),'training_fits':0,'new_patient_predictions':0,'paid_queries':0,'v4_upstream_status':'PASS_WITH_DOCUMENTED_AUDIT_LIMITATION: full historical GA trajectories were not replayed; predictions and metrics verified.','verified_Riyadh':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat()}
(O/'verification/Independent_Verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
