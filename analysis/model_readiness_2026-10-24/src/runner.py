"""Execution routing and result handling. Production invoked only behind launcher authorization."""
from pathlib import Path
import json,hashlib,datetime,os,traceback
import numpy as np
from engine import select_candidate,metric,paired_interval,COUNTERS

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def ids_hash(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def candidates(view,plan):
 c=plan['model']['clinical_lambdas'];r=plan['model']['RNA_lambdas']
 return [('none',a,None) for a in c]+[(arm,a,b) for arm in view['arms'] if arm!='none' for a in c for b in r]
def validate_prediction(pred,n):
 p=np.asarray(pred,float)
 if p.shape!=(n,) or not np.isfinite(p).all():raise ValueError('Backend returned invalid shape or nonfinite predictions')
 return p

def execute(bundle,output,selected,backend,qa_only=False):
 if qa_only and (not bundle.get('is_synthetic_QA') or any(not k.startswith('QA_') for k in bundle['features'])):raise ValueError('QA requires artificial identities and explicit synthetic bundle')
 output=Path(output)
 if output.exists():raise FileExistsError('Refuse to overwrite existing execution directory')
 if not selected or len(set(selected))!=len(selected):raise ValueError('Select unique nonempty view IDs')
 views=[v for v in bundle['views'] if v['view_id'] in selected]
 if len(views)!=len(selected):raise ValueError('Unknown selected view')
 output.mkdir(parents=True,exist_ok=False);private=output/'restricted';private.mkdir(mode=0o700);models=private/'models';models.mkdir(mode=0o700)
 prefix='QA_ONLY_' if qa_only else '';plan=bundle['plan'];spl=bundle['splits'];fit_attempts=0;fit_success=0;initial_solves=COUNTERS['ridge_solves'];predictions=[];tuning=[];logs=[];metrics=[];contrasts=[];allbyview={}
 write(output/'RUNNING.json',{'qa_only':qa_only,'selected_views':selected,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING_NOT_SUCCESS'})
 def call(view,train,target,y,arm,c,r,stage,o,i):
  nonlocal fit_attempts,fit_success
  if not train or not target or set(train)&set(target):raise ValueError('Invalid train-target partition')
  fit_attempts+=1;p,state=backend.fit_predict(view,train,target,np.asarray(y,float),arm,c,r);p=validate_prediction(p,len(target));fit_success+=1
  logs.append({'view_id':view['view_id'],'stage':stage,'outer_fold':o,'inner_fold':i,'arm':arm,'lambda_clinical':c,'lambda_RNA':r,'train_n':len(train),'target_n':len(target),'train_components_sha256':ids_hash(train),'target_components_sha256':ids_hash(target),'qa_only':qa_only})
  return p,state
 try:
  for view in views:
   name=view['view_id'];rows={r['component_id']:r for r in bundle['binding'] if r['view_id']==name};components=sorted(rows);pred={};pool={}
   if len(rows)!=view['expected_n']:raise ValueError('View membership count mismatch')
   for o in range(5):
    train=[k for k in components if spl[k]['outer_fold']!=o];test=[k for k in components if spl[k]['outer_fold']==o];ytrain=np.array([rows[k]['y'] for k in train]);typ=float(np.median([rows[k]['y']-rows[k]['baseline_moca'] for k in train]));loss={s:[] for s in candidates(view,plan)}
    for k in test:
     if k in pred:raise ValueError('Duplicate OOF assignment')
     pred[k]={'view_id':name,'component_id':k,'PATNO':rows[k]['PATNO'],'outer_fold':o,'y':rows[k]['y'],'persistence':rows[k]['baseline_moca'],'typical_change':rows[k]['baseline_moca']+typ,'qa_only':qa_only}
    for i in range(3):
     itr=[k for k in train if spl[k]['inner_for_outer_'+str(o)]!=i];iva=[k for k in train if spl[k]['inner_for_outer_'+str(o)]==i];truth=np.array([rows[k]['y'] for k in iva])
     for arm,c,r in candidates(view,plan):
      p,_=call(view,itr,iva,[rows[k]['y'] for k in itr],arm,c,r,'inner',o,i);loss[arm,c,r].extend(np.abs(p-truth).tolist())
    for arm in view['arms']:
     options=[]
     for (a,c,r),err in loss.items():
      if a=='none' or a==arm:
       if len(err)!=len(train):raise ValueError('Inner pooled validation coverage mismatch')
       options.append({'candidate_arm':a,'lambda_clinical':c,'lambda_RNA':r,'pooled_MAE':float(np.mean(err)),'validation_n':len(err)})
     chosen=select_candidate(options)
     for option in options:tuning.append({'view_id':name,'outer_fold':o,'requested_arm':arm,**option,'selected':option==chosen,'qa_only':qa_only})
     p,state=call(view,train,test,ytrain,arm,chosen['lambda_clinical'],chosen['lambda_RNA'],'outer_refit',o,-1)
     write(models/(prefix+name+'__'+arm+'__outer'+str(o)+'.json'),{'qa_only':qa_only,'view_id':name,'arm':arm,'outer_fold':o,'selected':chosen,'state':state})
     for k,value in zip(test,p):pred[k][arm]=float(value)
   ordered=[pred[k] for k in components]
   if len(ordered)!=view['expected_n'] or any(any(a not in r for a in view['arms']) for r in ordered):raise ValueError('Incomplete OOF predictions')
   for model in ['persistence','typical_change']+view['arms']:metrics.append({'view_id':name,'model':model,**metric([r['y'] for r in ordered],[r[model] for r in ordered]),'qa_only':qa_only})
   pairs=[(a,'none') for a in view['arms'] if a!='none']+[('none','persistence'),('none','typical_change')]
   for left,right in pairs:
    ci=paired_interval([r['y'] for r in ordered],[r[left] for r in ordered],[r[right] for r in ordered],plan['inference']['bootstrap_replicates'],plan['inference']['seed'])
    contrasts.append({'view_id':name,'left':left,'right':right,'primary':name=='main_36_enhanced' and left=='Genes5_Hallmark11' and right=='none',**ci,'qa_only':qa_only})
   predictions.extend(ordered);allbyview[name]=ordered
  # Core/enhanced comparisons are paired only when both view outputs exist.
  cross=[]
  for h in [12,24,36]:
   a=allbyview.get(f'main_{h}_core');b=allbyview.get(f'main_{h}_enhanced')
   if a is None or b is None:continue
   if [r['component_id'] for r in a]!=[r['component_id'] for r in b] or [r['y'] for r in a]!=[r['y'] for r in b]:raise ValueError('Core/enhanced pair mismatch')
   for arm in ['none','Genes5','Hallmark11','Genes5_Hallmark11']:cross.append({'horizon':h,'arm':arm,'left':'enhanced','right':'core',**paired_interval([r['y'] for r in a],[r[arm] for r in b],[r[arm] for r in a],plan['inference']['bootstrap_replicates'],plan['inference']['seed']),'qa_only':qa_only})
  expected=sum(v['planned_ridge_fits'] for v in views)
  if fit_success!=expected:raise ValueError('Fit schedule total mismatch')
  real_solves=COUNTERS['ridge_solves']-initial_solves
  if qa_only and real_solves!=0:raise RuntimeError('Real fitting occurred in QA')
  if not qa_only and real_solves!=expected:raise RuntimeError('Production ridge solve count mismatch')
  for path,obj in [(private/(prefix+'Predictions.json'),predictions),(private/(prefix+'Tuning.json'),tuning),(private/(prefix+'Fit_Log.json'),logs),(output/(prefix+'Metrics.json'),metrics),(output/(prefix+'Contrasts.json'),contrasts),(output/(prefix+'Core_Enhanced_Contrasts.json'),cross)]:write(path,obj)
  result={'status':'QA_ROUTING_COMPLETE_NOT_SCIENTIFIC_RESULTS' if qa_only else 'COMPLETE','qa_only':qa_only,'fit_attempts':fit_attempts,'successful_backend_calls':fit_success,'actual_ridge_solves':real_solves,'OOF_participant_view_rows':len(predictions),'planned_ridge_fits':expected,'selected_views':selected};write(output/(prefix+'Run_Counts.json'),result)
  write(output/'COMPLETE.json',result);(output/'RUNNING.json').unlink();return result
 except Exception as exc:
  write(output/'FAILED.json',{'status':'FAILED_NOT_COMPLETE','qa_only':qa_only,'exception_type':type(exc).__name__,'message':str(exc),'fit_attempts':fit_attempts,'successful_backend_calls':fit_success,'actual_ridge_solves':COUNTERS['ridge_solves']-initial_solves});raise