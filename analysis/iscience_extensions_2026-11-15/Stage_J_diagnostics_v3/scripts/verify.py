"""Independent reconstruction, numeric and bootstrap verification of Stage J."""
import os,csv,json,hashlib,statistics,itertools,math
import numpy as np
from pathlib import Path
J=Path(__file__).resolve().parents[1];P=J.parent;C=P.parents[1];AB=P/'Stage_AB_protocols_v1_2026-11-15';R=J/'results'
def load(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=load(J/'config/diagnostic_plan.json');lock=load(J/'config/LOCK.json');assert sha(J/'config/diagnostic_plan.json')==lock['plan']
# Reconstruct A from frozen coefficients by field name and B from independent raw-unit equations.
state={r['member_id']:r['state'] for r in load(C/'independent_validation_contract_2026-10-31/restricted/Frozen_Inference_States.json')}
eq={int(r['horizon_months']):r for r in rows(C/'Journal_of_Neurology/supplement/Table_S14_Frozen_Reduced_Model_Equations.csv') if r['recipe']=='reduced3'}
groups={}
for st in ('A','B'):
 targets={(r['h'],r['id']):r['y'] for r in load(AB/f'restricted/outcomes_{st}.json')}
 for row in load(AB/f'restricted/features_{st}.json'):
  h=row['h'];key=row['id'];x=row['x'];pred=[]
  if st=='B':
   p=float(eq[h]['intercept_raw'])+sum(x[k]*float(eq[h][k+'_coefficient_raw']) for k in ('baseline_moca','age_years','sex_male'))
  else:
   for j in range(5):
    m=state[f'compact4_{h}m_fold{j}'];terms=[m['intercept']]
    for k,w in zip(m['columns'],m['coef']):
     if '==' in k:
      n,v=k.split('==');z=int(x[n]==v)
     elif k.endswith('__missing'):z=int(x[k[:-9]] is None)
     else:
      t=m['preprocessing']['numeric'][k];z=((x[k] if x[k] is not None else t['median'])-t['mean'])/t['sd']
     terms.append(w*z)
    pred.append(math.fsum(terms))
   p=statistics.mean(pred)
  groups.setdefault((st,h),[]).append((key,x['baseline_moca'],targets[h,key],p))
for r in rows(P/'Stage_C_results_v1/restricted/oof_predictions.csv'):
 if r['view']=='expanded':groups.setdefault(('C',int(r['horizon'])),[]).append((r['component'],float(r['persistence']),float(r['observed']),float(r['compact4'])))
bands=rows(R/'baseline_bands.csv');assert len(bands)==24
freq=rows(R/'change_distribution.csv');cal=rows(R/'calibration.csv')
for s in rows(R/'followup_and_stability.csv'):
 st=s['stage'];h=int(s['h']);g=sorted(groups[st,h]);b,y,p=np.array([r[1:] for r in g]).T;d=y-b;n=len(d)
 np.testing.assert_allclose([n,n/int(s['frame']),np.mean(d),np.median(d),*np.quantile(d,[.25,.75]),abs(d).mean(),sum(d==0),sum(d<0),sum(d>0),sum(b==30),sum(y==30),sum(b>=29),sum(y>=29)], [float(s[k]) for k in 'n fraction mean_change median q25 q75 mean_abs_change unchanged lower higher baseline_30 followup_30 baseline_29_30 followup_29_30'.split()],atol=1e-10)
 assert {float(v['change']):int(v['n']) for v in freq if v['stage']==st and int(v['h'])==h}==dict(zip(*np.unique(d,return_counts=True)))
 slope=np.cov(p,y)[0,1]/np.var(p,ddof=1);cs={'bias':np.mean(p-y),'calibration_slope':slope,'calibration_intercept':np.mean(y)-slope*np.mean(p)}
 for v in cal:
  if v['stage']==st and int(v['h'])==h:assert abs(cs[v['metric']]-float(v['estimate']))<1e-8
 for v in bands:
  if v['stage']!=st or int(v['h'])!=h:continue
  lo=int(v['lower']);select=(b>=lo)&(b<=int(v['upper']));nr=sum(select);assert nr==int(v['n'])
  if nr<30:assert not v['delta'];continue
  err=(abs(p-y)-abs(d))[select];rng=np.random.Generator(np.random.PCG64(20261115+10000*('ABC'.index(st)+1)+100*h+lo))
  reps=np.concatenate([err[rng.integers(nr,size=(100,nr))].mean(1) for i in range(50)])
  np.testing.assert_allclose([float(v[k]) for k in ('model_mae','persistence_mae','delta','ci_low','ci_high')],[abs(p-y)[select].mean(),abs(d)[select].mean(),np.mean(err),*np.percentile(reps,[2.5,97.5])],atol=1e-10)
assert all(sha(P/k)==v for k,v in lock['sources'].items())
(J/'QA/Verification.json').write_text(json.dumps(dict(status='PASS',stages=3,horizon_cells=len(groups),band_rows=len(bands),inference_cells=sum(bool(r['delta']) for r in bands),fits=0,calibration_updates=0,hashes={p.name:sha(p) for p in R.glob('*.csv')}),indent=2))
print('Independent Stage J verification passed')
