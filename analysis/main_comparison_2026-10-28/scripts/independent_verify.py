"""Independent audit of frozen production output. Does not import engine, runner or bundle."""
from pathlib import Path
import sys,os,json,csv,hashlib,math,collections,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import independent_reference as ref
C=Path(__file__).resolve().parents[1];P=C.parent;D=P/'model_readiness_2026-10-24';O=P/'executions/main_2026-10-28'
assert not (C/'qa/Independent_Verification.json').exists()
def read(p):return json.loads(p.read_text())
def table(p):
 with p.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ih(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def same(a,b,tol=1e-8):
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k],tol) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(same(x,y,tol) for x,y in zip(a,b))
 if a is None or b is None:return a is b
 if isinstance(a,(int,float)) and not isinstance(a,bool):return bool(np.isclose(a,b,atol=tol,rtol=tol))
 return a==b
checks=[]
def check(name,ok,detail=None):
 checks.append({'check':name,'passed':bool(ok),'detail':detail})
 if not ok:
  write(C/'qa/Independent_Verification_FAILED.json',{'passed':False,'checks':checks});raise AssertionError(name)
start=time.monotonic();plan=read(D/'config/Analysis_Contract.json');panel=read(D/'config/RNA_Panel.json');inputs=read(D/'config/Inputs.json');views=[v for v in read(D/'config/Views.json') if v['view_id'].startswith('main_')];names=[v['view_id'] for v in views]
check('Locked plan and code hashes remain exact',all(sha(D/x['path'])==x['sha256'] for x in read(D/'PLAN_LOCK.json')['files']))
check('External source input hashes unchanged',all(sha(Path(x['path']))==x['sha256'] for x in read(D/'evidence/Input_Manifest.json')))
counts=read(O/'Run_Counts.json');check('Complete production counts and scope',counts['status']=='COMPLETE' and counts['fit_attempts']==counts['successful_backend_calls']==counts['actual_ridge_solves']==5970 and counts['selected_views']==names and not counts['qa_only'] and (O/'COMPLETE.json').exists() and not (O/'RUNNING.json').exists() and not (O/'FAILED.json').exists())
# Snapshot production output before independent calculation; it is never rewritten.
production_before=[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()];write(C/'evidence/Production_Output_Before_Verification.json',production_before)
features={r['component_id']:r for r in table(D/'restricted/Clinical_Features.tsv')};splits={r['component_id']:r for r in table(D/'restricted/Grouped_Splits.tsv')};bound={(r['view_id'],r['component_id']):r for r in table(D/'restricted/View_Row_Binding.tsv') if r['view_id'] in names}
preds=read(O/'restricted/Predictions.json');logs=read(O/'restricted/Fit_Log.json');tuning=read(O/'restricted/Tuning.json');metrics=read(O/'Metrics.json');contrasts=read(O/'Contrasts.json');cross=read(O/'Core_Enhanced_Contrasts.json');sourceout={(r['PATNO'],int(r['horizon_months'])):r for r in table(Path(inputs['outcomes'])) if r['PD_matched_candidate']=='True'}
check('One held-out row for each expected component and view',len(preds)==1922 and len({(r['view_id'],r['component_id']) for r in preds})==1922 and {(r['view_id'],r['component_id']) for r in preds}==set(bound))
check('Every participant outcome and baseline value matches source binding',all(r['PATNO']==bound[r['view_id'],r['component_id']]['PATNO'] and r['y']==float(bound[r['view_id'],r['component_id']]['y'])==float(sourceout[r['PATNO'],int(bound[r['view_id'],r['component_id']]['horizon_months'])]['education_adjusted_MoCA']) and r['persistence']==float(bound[r['view_id'],r['component_id']]['baseline_moca']) and r['outer_fold']==int(splits[r['component_id']]['outer_fold']) and not r['qa_only'] for r in preds))
check('All 7688 model predictions finite',all(math.isfinite(r[a]) for r in preds for a in ['none','Genes5','Hallmark11','Genes5_Hallmark11']))
byview={n:sorted([r for r in preds if r['view_id']==n],key=lambda x:x['component_id']) for n in names}
# Compare each logged job to the immutable schedule; outer penalties are tuned and therefore checked separately.
schedule=[x for x in table(D/'restricted/Planned_Fit_Schedule.tsv') if x['view_id'] in names]
def key(x):return (x['view_id'],x['stage'],int(x['outer_fold']),int(x['inner_fold']),x['arm'],float(x['lambda_clinical']) if x['stage']=='inner' else None,(float(x['lambda_RNA']) if x['lambda_RNA'] not in [None,''] else None) if x['stage']=='inner' else None)
sm={key(r):r for r in schedule};lm={key(r):r for r in logs};check('All5970 scheduled jobs appear exactly once',len(sm)==len(lm)==len(logs)==len(schedule)==5970 and set(sm)==set(lm))
for k,l in lm.items():
 s=sm[k]
 assert all(int(s[f])==int(l[f]) for f in ['train_n','target_n']) and all(s[f]==l[f] for f in ['train_components_sha256','target_components_sha256']) and not l['qa_only']
check('Every production job matches frozen partition hashes and sizes',True)
check('Expected inner/outer split of fit counts',sum(x['stage']=='inner' for x in logs)==5850 and sum(x['stage']=='outer_refit' for x in logs)==120)
raw=np.load(inputs['rna_matrix'],mmap_mode='r');selected={k:raw[int(r['count_row']),panel['raw_columns']] for k,r in features.items()};library={k:float(raw[int(r['count_row'])].sum()) for k,r in features.items()}
max_loss=0.;max_prediction=0.;max_gradient=0.;reference_solves=0;verified_models=0;activity=[];verified_metrics=[];verified_contrasts=[];verified_cross=[];tuning_comparisons=0
for view in views:
 name=view['view_id'];rows=byview[name];ids=[r['component_id'] for r in rows];yby={r['component_id']:r['y'] for r in rows};pby={r['component_id']:r for r in rows};spec=plan['comparators'][view['clinical']]
 assert len(rows)==view['expected_n']
 for o in range(5):
  tr=[k for k in ids if int(splits[k]['outer_fold'])!=o];te=[k for k in ids if int(splits[k]['outer_fold'])==o]
  # Independently recompute ALL inner candidate losses using augmented SVD and separate preprocessing.
  errors={('none',lc,None):[] for lc in plan['model']['clinical_lambdas']}
  errors.update({(arm,lc,lr):[] for arm in view['arms'][1:] for lc in plan['model']['clinical_lambdas'] for lr in plan['model']['RNA_lambdas']})
  for i in range(3):
   a=[k for k in tr if int(splits[k]['inner_for_outer_'+str(o)])!=i];b=[k for k in tr if int(splits[k]['inner_for_outer_'+str(o)])==i]
   assert set(a).isdisjoint(b) and set(a+b)==set(tr) and set(a+b).isdisjoint(te)
   cs=ref.clinical_state([features[k] for k in a],spec);xa,_=ref.clinical_design([features[k] for k in a],cs);xb,_=ref.clinical_design([features[k] for k in b],cs)
   rs=ref.rna_state(np.array([selected[k] for k in a]),[library[k] for k in a],panel,plan['RNA']['training_gene_gate']);designs={'none':(xa,xb)}
   for arm in view['arms'][1:]:
    ra,_=ref.rna_design(np.array([selected[k] for k in a]),[library[k] for k in a],rs,arm);rb,_=ref.rna_design(np.array([selected[k] for k in b]),[library[k] for k in b],rs,arm);designs[arm]=(np.c_[xa,ra],np.c_[xb,rb])
   for (arm,lc,lr),loss in errors.items():
    A,B=designs[arm];coef,inter=ref.augmented_svd_ridge(A,[yby[k] for k in a],xa.shape[1],lc,lr);pr=B@coef+inter;reference_solves+=1;loss.extend(abs(pr-np.array([yby[k] for k in b])).tolist())
   for l in logs:
    if l['view_id']==name and l['outer_fold']==o and l['inner_fold']==i and l['stage']=='inner':assert ih(a)==l['train_components_sha256'] and ih(b)==l['target_components_sha256']
  independent_loss={k:math.fsum(e)/len(e) for k,e in errors.items()};assert all(len(e)==len(tr) for e in errors.values())
  # Derive outer preprocessing from the original count and clinical records, not saved states.
  cs=ref.clinical_state([features[k] for k in tr],spec);xc,cn=ref.clinical_design([features[k] for k in tr],cs);zc,_=ref.clinical_design([features[k] for k in te],cs)
  rs=ref.rna_state(np.array([selected[k] for k in tr]),[library[k] for k in tr],panel,plan['RNA']['training_gene_gate'])
  for arm in view['arms']:
   record=read(O/'restricted/models'/(name+'__'+arm+'__outer'+str(o)+'.json'));s=record['state'];assert not record['qa_only'] and s['train_ids']==tr and s['target_ids']==te
   assert same(s['clinical_state'],cs) and same(s['RNA_state'],rs),'Independent preprocessing mismatch'
   entries=[t for t in tuning if t['view_id']==name and t['outer_fold']==o and t['requested_arm']==arm]
   expected_options={k for k in errors if k[0] in ['none',arm]};assert len(entries)==len(expected_options) and {(t['candidate_arm'],t['lambda_clinical'],t['lambda_RNA']) for t in entries}==expected_options
   for t in entries:
    k=(t['candidate_arm'],t['lambda_clinical'],t['lambda_RNA']);err=abs(t['pooled_MAE']-independent_loss[k]);max_loss=max(max_loss,err);assert err<1e-8 and t['validation_n']==len(tr) and not t['qa_only'];tuning_comparisons+=1
   best=min(independent_loss[k] for k in expected_options);ties=[k for k in expected_options if independent_loss[k]<=best+1e-12];chosen=min(ties,key=lambda k:(k[2] is not None,-(k[2] or 0),-k[1]));picked=[t for t in entries if t['selected']];assert len(picked)==1
   assert (picked[0]['candidate_arm'],picked[0]['lambda_clinical'],picked[0]['lambda_RNA'])==chosen
   assert (record['selected']['candidate_arm'],s['lambda_clinical'],s['lambda_RNA'])==chosen and s['RNA_off']==(chosen[2] is None)
   if chosen[2] is None:X=xc;Z=zc;rn=[]
   else:
    xr,rn=ref.rna_design(np.array([selected[k] for k in tr]),[library[k] for k in tr],rs,arm);zr,_=ref.rna_design(np.array([selected[k] for k in te]),[library[k] for k in te],rs,arm);X=np.c_[xc,xr];Z=np.c_[zc,zr]
   assert s['columns']==cn+rn
   beta=np.asarray(s['coef']);inter=s['intercept'];want=Z@beta+inter;seen=np.array([pby[k][arm] for k in te]);err=float(np.max(abs(want-seen)));max_prediction=max(max_prediction,err);assert err<1e-8
   residual=X@beta+inter-np.array([yby[k] for k in tr]);pen=np.r_[np.repeat(s['lambda_clinical'],xc.shape[1]),np.repeat(s['lambda_RNA'],len(beta)-xc.shape[1])].astype(float);grad=X.T@residual/len(tr)+pen*beta;g=max(float(np.max(abs(grad))),abs(float(residual.mean())));max_gradient=max(max_gradient,g);assert g<1e-7
   l=lm[(name,'outer_refit',o,-1,arm,None,None)];assert l['lambda_clinical']==s['lambda_clinical'] and l['lambda_RNA']==s['lambda_RNA'] and ih(tr)==l['train_components_sha256'] and ih(te)==l['target_components_sha256']
   verified_models+=1
   activity.append({'view_id':name,'outer_fold':o,'arm':arm,'RNA_off':s['RNA_off'],'lambda_clinical':s['lambda_clinical'],'lambda_RNA':s['lambda_RNA'],'active_genes5':[n for ix,n in zip(panel['gene5_indices'],panel['gene5_names']) if rs['active'][ix]],'active_pathways':sum(x['active'] for x in rs['pathways']),'pathway_coverage':[x['coverage'] for x in rs['pathways']]})
  typ=float(np.median([yby[k]-float(bound[name,k]['baseline_moca']) for k in tr]));assert all(pby[k]['typical_change']==pby[k]['persistence']+typ for k in te)
 for model in ['persistence','typical_change']+view['arms']:
  independent=ref.metrics([r['y'] for r in rows],[r[model] for r in rows]);saved=next(x for x in metrics if x['view_id']==name and x['model']==model);assert all(same(saved[k],v) for k,v in independent.items());verified_metrics.append({'view_id':name,'model':model,**independent})
 for co in [x for x in contrasts if x['view_id']==name]:
  independent=ref.interval([r['y'] for r in rows],[r[co['left']] for r in rows],[r[co['right']] for r in rows]);assert all(same(co[k],v) for k,v in independent.items());verified_contrasts.append({'view_id':name,'left':co['left'],'right':co['right'],'primary':co['primary'],**independent})
 print('Verified numerical and participant bindings for '+name,flush=True)
check('All5850 inner candidate solutions independently recalculated by SVD',reference_solves==5850,{'max_pooled_MAE_difference':max_loss,'saved_candidate_records_checked':tuning_comparisons})
check('All120 outer states, selections and predictions independently verified',verified_models==120,{'max_prediction_difference':max_prediction,'max_optimality_gradient':max_gradient})
check('All36 model and benchmark metric rows independently reproduced',len(verified_metrics)==len(metrics)==36)
check('All30 within-view paired intervals independently reproduced',len(verified_contrasts)==len(contrasts)==30)
check('Single designated primary contrast retained',sum(bool(x['primary']) for x in contrasts)==1 and next(x for x in contrasts if x['primary'])['view_id']=='main_36_enhanced')
for co in cross:
 a=byview['main_'+str(co['horizon'])+'_core'];b=byview['main_'+str(co['horizon'])+'_enhanced'];assert [r['component_id'] for r in a]==[r['component_id'] for r in b] and [r['y'] for r in a]==[r['y'] for r in b]
 independent=ref.interval([r['y'] for r in a],[r[co['arm']] for r in b],[r[co['arm']] for r in a]);assert all(same(co[k],v) for k,v in independent.items());verified_cross.append({'horizon':co['horizon'],'arm':co['arm'],'left':'enhanced','right':'core',**independent})
check('All12 core/enhanced paired intervals independently reproduced',len(verified_cross)==len(cross)==12)
protected=read(C/'evidence/Protected_Before.json');check('All1590 protected prior artifacts unchanged',len(protected)==1590 and all(sha(Path(x['path']))==x['sha256'] for x in protected))
check('Production payload unchanged by verification',all(sha(O/x['path'])==x['sha256'] for x in production_before))
write(C/'qa/Recomputed_Metrics.json',verified_metrics);write(C/'qa/Recomputed_Contrasts.json',verified_contrasts);write(C/'qa/Recomputed_Core_Enhanced_Contrasts.json',verified_cross);write(C/'reports/RNA_Activity_and_Selections.json',activity)
result={'status':'INDEPENDENT_VERIFICATION_PASSED','passed':True,'checks':checks,'independent_reference_inner_solves':reference_solves,'production_ridge_fits':5970,'outer_models_verified_without_refitting':verified_models,'saved_tuning_records_checked':tuning_comparisons,'maximum_inner_MAE_difference':max_loss,'maximum_prediction_difference':max_prediction,'maximum_optimality_gradient':max_gradient,'independent_metric_rows':36,'independent_paired_contrasts':42,'protected_unchanged':len(protected),'wall_seconds':time.monotonic()-start,'paid_queries':0,'methods':'Independent augmented-design SVD for every inner fit; independent train-only preprocessing; all outer predictions reconstructed and positive-penalty optimality checked; separate scalar metrics and vectorized paired bootstrap. No engine/runner/bundle import. Production outputs never edited.'}
write(C/'qa/Independent_Verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['checks','methods']}),flush=True)