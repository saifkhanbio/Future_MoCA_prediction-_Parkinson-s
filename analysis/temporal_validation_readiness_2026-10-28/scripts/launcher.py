"""Default is validation only. Target prediction/scoring needs a later authorisation receipt."""
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
import argparse,json,csv,hashlib,math
D=Path(__file__).resolve().parents[1];sys.path.insert(0,str(D/'src'))
from clinical_inference import validate_model,number,predict_ensemble,bootstrap,bind_outcomes
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def save(p,data):
 with Path(p).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
def authorize(receipt,action,lock):
 if not receipt:raise PermissionError('Target prediction/scoring requires a subsequent explicit user authorization receipt.')
 a=json.loads(Path(receipt).read_text())
 if a.get('contract_sha256')!=lock['contract_sha256'] or a.get('readiness_lock_sha256')!=sha(D/'READINESS_LOCK.json') or a.get('explicit_user_authorization') is not True or action not in a.get('allowed_actions',[]):raise PermissionError('Authorization is absent, mismatched or does not cover this action')
 return a
def output_dir(path):
 if not path:raise ValueError('Fresh output directory required')
 p=Path(path).resolve();allowed=(D/'executions').resolve()
 if allowed not in p.parents or p.exists():raise ValueError('Output must be a fresh child of this package executions directory')
 p.mkdir(parents=True,mode=0o700);return p
def load_bundle():
 lock=json.loads((D/'READINESS_LOCK.json').read_text())
 if lock['status']!='READY_NO_TARGET_SCORING':raise ValueError('Readiness lock absent')
 for x in lock['bound_files']:
  if sha(D/x['path'])!=x['sha256']:raise ValueError('Locked package file changed: '+x['path'])
 for x in json.loads((D/'evidence/Source_Manifest.json').read_text()):
  if sha(x['path'])!=x['sha256']:raise ValueError('Frozen source file changed')
 c=json.loads((D/'config/Temporal_Validation_Contract.json').read_text())
 if sha(D/'config/Temporal_Validation_Contract.json')!=lock['contract_sha256']:raise ValueError('Contract mismatch')
 for k in ['training_authorized','prediction_generation_authorized','scoring_authorized','paid_queries_authorized']:
  if c['execution_controls'][k] is not False:raise ValueError('Readiness authorizes no production action')
 feat={}
 for v in ['resolved','strict']:
  rr=read(D/f'restricted/Target_Features_{v.title()}.tsv')
  if len(rr)!=c['population']['unique_validation_participants'] or len({r['PATNO'] for r in rr})!=len(rr) or len({r['component_id'] for r in rr})!=len(rr):raise ValueError('Duplicate/missing target identity')
  expected={'PATNO','component_id','index_month'}|set(c['predictors']['enhanced']['numeric'])|set(c['predictors']['enhanced']['categories'])
  for r in rr:
   if set(r)!=expected:raise ValueError('Unexpected feature columns including possible future outcome')
   for k in c['predictors']['enhanced']['numeric']:number(r[k])
   for k,levels in c['predictors']['enhanced']['categories'].items():
    if r[k] not in levels:raise ValueError('Target category mismatch')
  feat[v]={r['PATNO']:r for r in rr}
 ids={r['component_id'] for r in feat['resolved'].values()}
 bound=json.loads((D/'config/Frozen_Model_Binding.json').read_text())['models'];models={}
 if len(bound)!=30:raise ValueError('Model coverage mismatch')
 for r in bound:
  if sha(r['path'])!=r['sha256']:raise ValueError('Model hash mismatch')
  m=json.loads(Path(r['path']).read_text());validate_model(m,c['predictors'][r['comparator']])
  if m['outer_fold']!=r['outer_fold'] or m['view_id']!=f"main_{r['horizon']}_{r['comparator']}" or r['weight']!=0.2:raise ValueError('Model identity/weight mismatch')
  if ids & set(m['state']['train_ids']+m['state']['target_ids']):raise ValueError('Development/target identity overlap')
  models.setdefault((r['horizon'],r['comparator']),[]).append(m)
 for h in [12,24,36]:
  for arm in ['core','enhanced']:
   group=models[h,arm]
   if len(group)!=5 or {m['outer_fold'] for m in group}!=set(range(5)):raise ValueError('Ensemble coverage mismatch')
 cohorts={h:read(D/f'restricted/Validation_Cohort_{h}m.tsv') for h in [12,24,36]}
 for h,rr in cohorts.items():
  if len(rr)!=c['population']['main_by_horizon'][str(h)] or len({r['component_id'] for r in rr})!=len(rr):raise ValueError('Horizon coverage mismatch')
  for r in rr:
   if r['PATNO'] not in feat['resolved'] or feat['resolved'][r['PATNO']]['component_id']!=r['component_id']:raise ValueError('Feature/cohort identity mismatch')
 views=json.loads((D/'config/Evaluation_Views.json').read_text())
 for v in views:
  rr=read(D/v['member_manifest'])
  if len(rr)!=v['expected_n'] or len({r['component_id'] for r in rr})!=len(rr):raise ValueError('View membership mismatch')
  expected={(r['PATNO'],r['component_id']) for r in cohorts[v['horizon_months']]}
  if not {(r['PATNO'],r['component_id']) for r in rr}<=expected:raise ValueError('View contains noncohort identity')
 return lock,c,feat,models,cohorts,views
def predict(bundle,out,receipt):
 lock,c,feat,models,cohorts,views=bundle;authorize(receipt,'predict',lock);out=output_dir(out)
 bench=json.loads((D/'config/Frozen_Benchmark_Binding.json').read_text())['offsets'];results=[]
 for h in [12,24,36]:
  bs=[x for x in bench if x['horizon']==h]
  if len(bs)!=5 or {x['source_outer_fold'] for x in bs}!=set(range(5)):raise ValueError('Benchmark coverage mismatch')
  offset=sum(x['weight']*x['already_frozen_offset'] for x in bs)
  for variant in ['resolved','strict']:
   rows=[feat[variant][r['PATNO']] for r in cohorts[h]]
   outputs={a:predict_ensemble(rows,models[h,a],c['predictors'][a]) for a in ['core','enhanced']}
   for i,r in enumerate(rows):
    base=number(r['baseline_moca'])
    results.append({'PATNO':r['PATNO'],'component_id':r['component_id'],'horizon_months':h,'feature_variant':variant,
    'core':float(outputs['core'][0][i]),'enhanced':float(outputs['enhanced'][0][i]),
    'core_members':outputs['core'][1][i].tolist(),'enhanced_members':outputs['enhanced'][1][i].tolist(),
    'persistence':base,'source_typical_change':base+offset})
 # No target outcome file is opened in this stage.
 save(out/'Predictions.json',results)
 save(out/'Prediction_Lock.json',{'contract_sha256':lock['contract_sha256'],'readiness_lock_sha256':sha(D/'READINESS_LOCK.json'),
  'rows':len(results),'prediction_sha256':sha(out/'Predictions.json'),'target_outcomes_opened':False,'models_fit':0})
 return {'status':'PREDICTIONS_SEALED_NOT_SCORED','rows':len(results),'output':str(out)}
def score(bundle,prediction_dir,out,receipt):
 lock,c,feat,models,cohorts,views=bundle;authorize(receipt,'score',lock)
 pd=Path(prediction_dir).resolve()
 if (D/'executions').resolve() not in pd.parents:raise ValueError('Prediction directory outside package')
 pl=json.loads((pd/'Prediction_Lock.json').read_text())
 if pl['contract_sha256']!=lock['contract_sha256'] or pl['readiness_lock_sha256']!=sha(D/'READINESS_LOCK.json') or sha(pd/'Predictions.json')!=pl['prediction_sha256']:raise ValueError('Prediction lock mismatch')
 rr=json.loads((pd/'Predictions.json').read_text());expected={(r['component_id'],h,v) for h,rows in cohorts.items() for r in rows for v in ['resolved','strict']}
 keys=[(r['component_id'],r['horizon_months'],r['feature_variant']) for r in rr]
 if len(keys)!=len(set(keys)) or set(keys)!=expected or len(rr)!=pl['rows']:raise ValueError('Prediction row coverage mismatch')
 for r in rr:
  for arm in ['core','enhanced']:
   members=r[arm+'_members']
   if len(members)!=5 or not all(math.isfinite(x) for x in members) or not math.isclose(r[arm],sum(members)/5,abs_tol=1e-10,rel_tol=0):raise ValueError('Ensemble result mismatch')
 out=output_dir(out);answers={}
 for view in views:
  vid=view['view_id'];h=view['horizon_months'];variant=view['feature_variant']
  if view['reuse_alias']:
   answers[vid]={'reuse_alias':view['reuse_alias'],'result':answers[view['reuse_alias']]};continue
  members=read(D/view['member_manifest']);who={r['component_id'] for r in members}
  pr=[r for r in rr if r['horizon_months']==h and r['feature_variant']==variant and r['component_id'] in who]
  outcomes=[r for r in read(D/f'restricted/Sealed_Outcomes_{h}m.tsv') if r['component_id'] in who]
  matched=bind_outcomes(pr,outcomes)
  if len(matched)!=view['expected_n']:raise ValueError('Scored cohort mismatch')
  y=[y for _,y in matched];pred={a:[r[a] for r,_ in matched] for a in ['core','enhanced','persistence','source_typical_change']}
  answers[vid]=bootstrap(y,pred,view['bootstrap_seed'],c['metrics']['uncertainty']['replicates'])
 save(out/'Results.json',answers);save(out/'Results_Provenance.json',{'prediction_lock':str(pd/'Prediction_Lock.json'),
 'prediction_lock_sha256':sha(pd/'Prediction_Lock.json'),'contract_sha256':lock['contract_sha256'],'models_fit':0})
 return {'status':'SCORED_REQUIRES_INDEPENDENT_VERIFICATION','views':len(answers),'output':str(out)}
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['validate','predict','score'],nargs='?',default='validate')
 a.add_argument('--authorization');a.add_argument('--output');a.add_argument('--predictions');args=a.parse_args()
 bundle=load_bundle()
 if args.mode=='validate':
  print(json.dumps({'status':'PASS_READINESS_ONLY','models':30,'participants':len(bundle[2]['resolved']),
  'views':len(bundle[5]),'models_fit':0,'target_predictions':0,'target_metrics':0}));return
 if args.mode=='predict':r=predict(bundle,args.output,args.authorization)
 else:
  if not args.predictions:raise ValueError('Scoring requires locked predictions')
  r=score(bundle,args.predictions,args.output,args.authorization)
 print(json.dumps(r))
if __name__=='__main__':main()
