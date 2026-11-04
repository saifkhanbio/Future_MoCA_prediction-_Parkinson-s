"""Independent scalar reconstruction of every production prediction; no launcher imports or target outcomes."""
import os,sys,math,json,csv,hashlib
from pathlib import Path
os.umask(0o077);sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];D=R.parents[1]
def deny(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and 'Sealed_Outcomes_' in str(args[0]):raise PermissionError('Prediction verification cannot open target outcomes')
sys.addaudithook(deny)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
lock=json.loads((R/'restricted/predictions/Prediction_Lock.json').read_text())
assert sha(R/'restricted/predictions/Predictions.json')==lock['prediction_sha256']
p=json.loads((R/'restricted/predictions/Predictions.json').read_text())
features={v:{r['component_id']:r for r in read(D/f'restricted/Target_Features_{v.title()}.tsv')} for v in ['resolved','strict']}
bindings=json.loads((D/'config/Frozen_Model_Binding.json').read_text())['models']
models={}
for b in bindings:
 assert sha(b['path'])==b['sha256']
 m=json.loads(Path(b['path']).read_text());s=m['state']
 assert m['arm']=='none' and not m['qa_only'] and s['RNA_off'] is True and s['lambda_RNA'] is None
 assert not set(features['resolved'])&set(s['train_ids']+s['target_ids'])
 models[b['horizon'],b['comparator'],b['outer_fold']]=s
cohorts={h:{r['component_id']:r for r in read(D/f'restricted/Validation_Cohort_{h}m.tsv')} for h in [12,24,36]}
expected={(k,h,v) for h,c in cohorts.items() for k in c for v in ['resolved','strict']}
keys=[(r['component_id'],r['horizon_months'],r['feature_variant']) for r in p]
assert len(keys)==len(set(keys))==lock['rows']==854 and set(keys)==expected
bench=json.loads((D/'config/Frozen_Benchmark_Binding.json').read_text())['offsets']
assert len(bench)==15
reconstructed=[];max_member=0.;max_ensemble=0.;member_checks=0;ensemble_checks=0
for r in p:
 cid=r['component_id'];h=r['horizon_months'];v=r['feature_variant'];row=features[v][cid]
 assert row['PATNO']==r['PATNO']==cohorts[h][cid]['PATNO']
 out={k:r[k] for k in ['PATNO','component_id','horizon_months','feature_variant']}
 for arm in ['core','enhanced']:
  values=[]
  for fold in range(5):
   s=models[h,arm,fold];st=s['clinical_state'];terms=[]
   # Reconstruct using coefficient names, rather than production column-building loops.
   for name,beta in zip(s['columns'],s['coef']):
    if name.endswith('__missing'):
     raw=row[name[:-9]];x=float(raw.strip()=='')
    elif '=' in name:
     key,level=name.split('=',1);assert row[key] in st['categories'][key];x=float(row[key]==level)
    else:
     raw=row[name];z=st['numeric'][name];value=z['median'] if raw.strip()=='' else float(raw)
     assert math.isfinite(value) and z['sd']>0
     x=(value-z['mean'])/z['sd']
    terms.append(float(beta)*x)
   y=math.fsum(terms)+float(s['intercept']);actual=r[arm+'_members'][fold]
   diff=abs(y-actual);max_member=max(max_member,diff);assert diff<=1e-10
   values.append(y);member_checks+=1
  y=math.fsum(values)/5;diff=abs(y-r[arm]);assert diff<=1e-10;max_ensemble=max(max_ensemble,diff)
  out[arm]=y;out[arm+'_members']=values;ensemble_checks+=1
 offsets=[x for x in bench if x['horizon']==h]
 assert {x['source_outer_fold'] for x in offsets}==set(range(5)) and all(x['weight']==.2 for x in offsets)
 out['persistence']=float(row['baseline_moca'])
 out['source_typical_change']=out['persistence']+math.fsum(x['already_frozen_offset'] for x in offsets)/5
 for key in ['persistence','source_typical_change']:assert abs(out[key]-r[key])<=1e-12
 reconstructed.append(out)
with (R/'restricted/Independent_Predictions.json').open('x') as f:json.dump(reconstructed,f,indent=2,allow_nan=False)
summary={'status':'PASS','implementation':'Independent scalar math.fsum evaluation by saved coefficient name; no production module imports',
 'production_rows_verified':len(p),'member_predictions_verified':member_checks,'ensemble_predictions_verified':ensemble_checks,
 'max_member_difference':max_member,'max_ensemble_difference':max_ensemble,
 'identity_and_horizon_coverage_verified':True,'benchmarks_verified':2*len(p),
 'outcome_files_read':0,'models_fit':0,'prediction_sha256':sha(R/'restricted/predictions/Predictions.json')}
with (R/'qa/Independent_Prediction_Verification.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
