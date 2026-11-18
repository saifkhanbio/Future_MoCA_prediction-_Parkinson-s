"""No-fit independent checks of saved synthetic outputs and integrity guards."""
import json,sys,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import launcher as L
D=L.D
plan=json.loads((L.CONTRACT/'Protocol.json').read_text())
f=pd.read_csv(D/'qa/synthetic_results/restricted/oof_predictions.csv')
stats=json.loads((D/'qa/synthetic_results/metrics.json').read_text())
maxpoint=0.;maxr2ci=0.
for (v,h),sub in f.groupby(['view','horizon']):
 sub=sub.sort_values('component');y=sub.observed.to_numpy()
 for a in plan['views'][v]['models']:
  p=sub[a].to_numpy();slope,intercept=np.polyfit(p,y,1)
  check=dict(MAE=mean_absolute_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),R2=r2_score(y,p),bias=np.mean(p-y),calibration_slope=slope,calibration_intercept=intercept,outside_0_30=np.mean((p<0)|(p>30)))
  for key,val in check.items():maxpoint=max(maxpoint,abs(val-stats[v+'/'+str(h)][a+'/'+key]['estimate']))
 rng=np.random.Generator(np.random.PCG64(plan['seed_base']+100*(list(plan['views']).index(v)+1)+h))
 ix=rng.integers(len(sub),size=(plan['bootstrap'],len(sub)));yy=y[ix];den=((yy-yy.mean(1)[:,None])**2).sum(1)
 for a,b in plan['paired_contrasts']:
  # Delta R2 (augmented minus clinical) equals (SSEclinical - SSEaugmented)/SST.
  delta=(((sub[b].to_numpy()[ix]-yy)**2).sum(1)-((sub[a].to_numpy()[ix]-yy)**2).sum(1))/den
  ci=np.quantile(delta,[.025,.975],method='linear')
  r=stats[v+'/'+str(h)][a+'-'+b+'/delta_R2']
  maxr2ci=max(maxr2ci,float(np.max(np.abs(ci-r['interval']))))
assert maxpoint<1e-10 and maxr2ci<1e-10
L.verify_sources()
seal=json.loads((L.CONTRACT/'LOCK.json').read_text())
upstream=json.loads((L.CONTRACT/'Source_SHA256.json').read_text())
# Pin the successful numerical suite and executable source without changing the contract.
codefiles=['launcher.py','numerical_core.py','paired_metrics.py','validate_synthetic.py','verify_final.py']
attestation=dict(status='PASS',contract_sha256=L.digest(L.CONTRACT/'LOCK.json'),code={p:L.digest(D/p) for p in codefiles},
 evidence={'qa/synthetic_validation.json':L.digest(D/'qa/synthetic_validation.json'),'qa/preflight_validation.json':L.digest(D/'qa/preflight_validation.json')},study_fits=0,study_execution_authorized=False)
(D/'VALIDATED_LAUNCHER.json').write_text(json.dumps(attestation,indent=2)+'\n')
L.verify_sources()
rejected=[]
original=L.digest
cases=[('contract content',L.CONTRACT/next(iter(seal['files']))),('upstream source',Path(next(iter(upstream)))),('contract seal',L.CONTRACT/'LOCK.json'),('launcher code',D/'numerical_core.py')]
for label,target in cases:
 def altered(p):return '0'*64 if Path(p)==target else original(p)
 with patch.object(L,'digest',side_effect=altered):
  try:L.verify_sources()
  except ValueError:rejected.append(label)
  else:raise AssertionError('Uncaught integrity fault: '+label)
r=subprocess.run([sys.executable,'-B',str(D/'launcher.py')],capture_output=True,text=True)
assert r.returncode==0
pre=json.loads(r.stdout);assert pre['study_fits']==0 and pre['status']=='PASS'
b=subprocess.run([sys.executable,'-B',str(D/'launcher.py'),'--mode','run-study'],capture_output=True,text=True)
assert b.returncode!=0 and 'Study fitting is not authorized' in b.stderr
assert not (D/'study_results').exists() and not (D/'STUDY_EXECUTION_AUTHORIZATION.json').exists()
L.verify_sources()
result=dict(status='PASS',independent_point_metric_max_difference=maxpoint,independent_paired_R2_interval_max_difference=maxr2ci,
 integrity_faults_rejected=rejected,contract_files_unchanged=len(seal['files']),upstream_files_unchanged=len(upstream),final_preflight=pre,
 study_execution_blocked=True,study_results_exist=False,study_fits=0,paid_queries=0)
(D/'qa/Final_Verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
