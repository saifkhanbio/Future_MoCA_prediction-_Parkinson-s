"""Independent scalar/weighted-bootstrap verification; no production comparison imports."""
from pathlib import Path
import json,math,hashlib,gzip,datetime
import numpy as np
A=Path(__file__).resolve().parents[1];C=A.parent;P=C.parent
def read(p):return json.loads(p.read_text())
def close(a,b):
 assert abs(a-b)<=1e-9*max(1.,abs(a),abs(b)),(a,b)
def main():
 plan=read(A/'config/Comparison_Plan.json')
 folders={'main':'executions/main_2026-10-28','mae':'ga_expanded_hybrid_run_2026-10-30','r2':'ga_r2_run_2026-10-31','fixed':'feature_only_run_2026-10-31','new':'ga_moca_run_v2_2026-10-31'}
 metrics={(r['horizon'],r['model_id']):r for r in read(A/'results/Metrics.json')}
 intervals={(r['horizon'],r['a'],r['b']):r for r in read(A/'results/Contrasts.json')}
 checked=0;contrast_checked=0;maxdiff=0.
 for h in plan['horizons']:
  source={k:read(C/v/'restricted'/f'OOF_{h}m.json') for k,v in folders.items() if k!='main'}
  source['main']=read(C/folders['main']/'restricted/Predictions.json')
  ref=source['new'];keys=[r['PATNO'] for r in ref];y=[float(r['y']) for r in ref];n=len(y);ym=math.fsum(y)/n
  sst=math.fsum((v-ym)**2 for v in y);pred={}
  with np.load(A/'restricted'/f'Matched_{h}m.npz') as bundle:
   assert np.array_equal(bundle['y'],y)
   for model in plan['models']:
    rows=source[model['source']]
    if model['source']=='main':rows=[r for r in rows if r['view_id']==f'main_{h}_{model["view"]}']
    lookup={r['PATNO']:r for r in rows};assert set(lookup)==set(keys) and len(lookup)==len(rows)
    for r in ref:assert all(r[k]==lookup[r['PATNO']][k] for k in ['component_id','outer_fold','y'])
    values=[float(lookup[k][model['arm']]) for k in keys];name=model['model_id']
    assert np.array_equal(values,bundle[name]);pred[name]=np.array(values)
    errors=[p-v for p,v in zip(values,y)];sse=math.fsum(e*e for e in errors);pm=math.fsum(values)/n
    v=math.fsum((p-pm)**2 for p in values);slope=math.fsum((p-pm)*(t-ym) for p,t in zip(values,y))/v
    actual={'n':n,'R2':1-sse/sst,'MAE':math.fsum(abs(e) for e in errors)/n,'MSE':sse/n,'RMSE':math.sqrt(sse/n),
     'mean_prediction_error':math.fsum(errors)/n,'calibration_slope':slope,'calibration_intercept':ym-slope*pm,
     'outside_0_30_fraction':sum(p<0 or p>30 for p in values)/n}
    saved=metrics[(h,name)]
    for k,val in actual.items():close(val,saved[k]);maxdiff=max(maxdiff,abs(val-saved[k]))
    checked+=1
  # Generate the same seeded resamples one at a time, representing them as integer frequency weights.
  rng=np.random.default_rng(20261030+h);w=np.array([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(5000)],dtype=float)
  yy=np.array(y);means=(w@yy)/n;den=(w@(yy*yy))-n*means*means;assert (den>0).all()
  for pair in plan['contrasts']:
   a,b=pair['a'],pair['b'];ea=pred[a]-yy;eb=pred[b]-yy
   r2values=((w@(eb*eb))-(w@(ea*ea)))/den;maeval=((w@abs(ea))-(w@abs(eb)))/n
   rlo,rhi=np.percentile(r2values,[2.5,97.5]);mlo,mhi=np.percentile(maeval,[2.5,97.5])
   actual={'delta_R2':metrics[(h,a)]['R2']-metrics[(h,b)]['R2'],'R2_lower95':rlo,'R2_upper95':rhi,
    'delta_MAE':metrics[(h,a)]['MAE']-metrics[(h,b)]['MAE'],'MAE_lower95':mlo,'MAE_upper95':mhi}
   for k,v in actual.items():close(v,intervals[(h,a,b)][k]);maxdiff=max(maxdiff,abs(v-intervals[(h,a,b)][k]))
   contrast_checked+=1
 # Verify the recommended fixed recipe is actually present in all15 saved model objects.
 binding=read(A/'config/Existing_Compact_Model_Binding.json')
 for item in binding['models']:
  model=read(Path(item['source_path']))['models']['enhanced']
  assert hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['model_payload_sha256']
  expected=['baseline_moca','SDMT','HVLT_immediate','HVLT_delayed']
  numeric={c for c in model['columns'] if '__' not in c and '==' not in c}
  assert numeric==set(expected)
  categories={c.split('==')[0] for c in model['columns'] if '==' in c}
  assert categories=={'baseline_sdmt_form','baseline_hvlt_form'}
 for path,info in read(A/'evidence/Source_Bindings.json').items():
  p=Path(path);assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
 before=json.loads(gzip.decompress((A/'restricted/Preexisting_Metadata.json.gz').read_bytes()))
 for rel,expected in before.items():
  p=P/rel;assert p.is_file() and (p.stat().st_size,p.stat().st_mtime_ns)==tuple(expected),rel
 result={'status':'PASS','independent_method':'Re-read frozen predictions; scalar fsum metrics; integer-weight paired bootstrap, with no imports from the comparison or training code.',
 'metric_rows_verified':checked,'contrast_rows_verified':contrast_checked,'maximum_arithmetic_difference':maxdiff,
 'compact_model_payloads_verified':len(binding['models']),'source_files_unchanged':len(read(A/'evidence/Source_Bindings.json')),
 'preexisting_files_metadata_unchanged':len(before),'participant_fits':0,'new_patient_predictions':0,'paid_queries':0,'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (A/'verification/Independent_Verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
