"""Exercise production output separation and result handling using synthetic records only."""
import sys,os,copy,json,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
D=Path(__file__).resolve().parents[1];sys.path.insert(0,str(D/'scripts'));sys.path.insert(0,str(D/'src'))
import launcher
Q=D/'qa/synthetic_pipeline'
assert not Q.exists(),'Do not overwrite fixture outputs'
Q.mkdir()
for sub in ['config','restricted','executions']:(Q/sub).mkdir()
originalD=launcher.D;launcher.D=Q
c=json.loads((D/'config/Temporal_Validation_Contract.json').read_text())
# Synthetic execution receipt is scoped to a separate fake contract, never to the real package.
fake={'status':'SYNTHETIC_ONLY','contract_sha256':'SYNTHETIC_NOT_A_REAL_CONTRACT'}
(Q/'READINESS_LOCK.json').write_text(json.dumps(fake))
receipt=Q/'synthetic_receipt.json'
receipt.write_text(json.dumps({'contract_sha256':fake['contract_sha256'],'readiness_lock_sha256':launcher.sha(Q/'READINESS_LOCK.json'),'explicit_user_authorization':True,'allowed_actions':['predict','score'],'qa_only':True}))
c['metrics']['uncertainty']['replicates']=5000
features={};models={};cohorts={};views=[];memberlist=[]
for i in range(8):
 r={'PATNO':f'SYNTHETIC_{i}','component_id':f'SYNTHETIC_COMPONENT_{i}','index_month':24000}
 for k in c['predictors']['enhanced']['numeric']:r[k]=float(i+1)
 r['baseline_moca']=20.+i
 for k,levels in c['predictors']['enhanced']['categories'].items():r[k]=levels[i%len(levels)]
 memberlist.append(r)
features['resolved']={r['PATNO']:r for r in memberlist}
features['strict']=copy.deepcopy(features['resolved']);features['strict']['SYNTHETIC_0']['age_index']=''
for h in [12,24,36]:
 cohorts[h]=[{'PATNO':r['PATNO'],'component_id':r['component_id']} for r in memberlist]
 for arm in ['core','enhanced']:
  spec=c['predictors'][arm];col=[v for k in spec['numeric'] for v in [k,k+'__missing']]+[k+'='+level for k,levels in spec['categories'].items() for level in levels]
  group=[]
  for fold in range(5):
   coef=[0.]*len(col);coef[col.index('baseline_moca')]=.8
   if arm=='enhanced':coef[col.index('baseline_sdmt')]=.1
   group.append({'qa_only':False,'arm':'none','outer_fold':fold,'selected':{'lambda_RNA':None},
    'state':{'RNA_off':True,'lambda_RNA':None,'clinical_state':{'order':spec['numeric'],'categories':spec['categories'],
    'numeric':{k:{'median':2.,'mean':0.,'sd':1.} for k in spec['numeric']}},
    'columns':col,'coef':coef,'intercept':3.+fold*.1}})
  models[h,arm]=group
 def write_table(path,rows):
  import csv
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 write_table(Q/f'restricted/members_{h}.tsv',cohorts[h])
 write_table(Q/f'restricted/Sealed_Outcomes_{h}m.tsv',[{'PATNO':r['PATNO'],'component_id':r['component_id'],'horizon_months':h,'education_adjusted_MoCA':19+i+(h//12)%2,'endpoint_QC_pass':True} for i,r in enumerate(memberlist)])
 views.append({'view_id':f'main_{h}','horizon_months':h,'feature_variant':'resolved','member_manifest':f'restricted/members_{h}.tsv','expected_n':8,'bootstrap_seed':100+h,'reuse_alias':None})
views += [dict(views[0],view_id='strict_inputs_12',feature_variant='strict'),dict(views[-1],view_id='exclude_age_resolved_36',reuse_alias='main_36')]
(Q/'config/Frozen_Benchmark_Binding.json').write_text(json.dumps({'offsets':[{'horizon':h,'source_outer_fold':f,'weight':.2,'already_frozen_offset':-.4} for h in [12,24,36] for f in range(5)]}))
bundle=(fake,c,features,models,cohorts,views)
# Observe file opens: prediction must not read even synthetic outcome files.
import builtins
realopen=builtins.open;realioopen=__import__('io').open;stage={'predict':True}
def checked_open(file,*args,**kwargs):
 if stage['predict'] and 'Sealed_Outcomes_' in str(file):raise AssertionError('Prediction stage read outcomes')
 return realopen(file,*args,**kwargs)
def checked_io(file,*args,**kwargs):
 if stage['predict'] and 'Sealed_Outcomes_' in str(file):raise AssertionError('Prediction stage read outcomes')
 return realioopen(file,*args,**kwargs)
builtins.open=checked_open;__import__('io').open=checked_io
try:
 r=launcher.predict(bundle,str(Q/'executions/predictions'),str(receipt))
finally:
 builtins.open=realopen;__import__('io').open=realioopen;stage['predict']=False
assert r['rows']==48
prediction_file=Q/'executions/predictions/Predictions.json';before=launcher.sha(prediction_file)
r=launcher.score(bundle,str(Q/'executions/predictions'),str(Q/'executions/results'),str(receipt))
assert launcher.sha(prediction_file)==before
results=json.loads((Q/'executions/results/Results.json').read_text())
assert len(results)==5
assert results['exclude_age_resolved_36']['result']==results['main_36']
assert all(v['n']==8 for k,v in results.items() if 'reuse_alias' not in v)
rejected=0
for action in ['predict','score']:
 try:launcher.authorize(str(receipt),action,{'contract_sha256':'REAL_CONTRACT_MISMATCH'})
 except PermissionError:rejected+=1
 else:raise AssertionError('Synthetic receipt could authorize another contract')
# Reject a modified prediction file before creating result output.
rows=json.loads(prediction_file.read_text());rows[0]['enhanced']+=1
prediction_file.write_text(json.dumps(rows))
try:launcher.score(bundle,str(Q/'executions/predictions'),str(Q/'executions/must_not_exist'),str(receipt))
except ValueError:rejected+=1
else:raise AssertionError('Modified prediction file accepted')
assert not (Q/'executions/must_not_exist').exists()
# Preserve the mutation test as QA-only evidence; it cannot be mistaken for a valid production output.
(Q/'QA_ONLY.txt').write_text('Synthetic identities, artificial models and fake contract only. Prediction file deliberately mutated for corruption rejection test. Never use as study evidence or production authorization.')
launcher.D=originalD
out={'status':'PASS','synthetic_prediction_rows':48,'synthetic_result_views':5,'production_output_stages_exercised':True,
 'prediction_stage_cannot_read_outcomes':True,'locked_predictions_unchanged_by_scoring':True,'alias_handling_verified':True,
 'cross_contract_authorization_and_corruption_rejections':rejected,'target_participant_records_used':0,'models_fit':0}
(D/'qa/Pipeline_Fixture_Verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
