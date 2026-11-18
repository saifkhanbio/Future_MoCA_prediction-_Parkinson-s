# Build bindings and prospective contract only. Does not fit or score.
from pathlib import Path
import json,hashlib,os
import pandas as pd,numpy as np
from scan_qc import select_scans
os.umask(0o077)
D=Path(__file__).resolve().parent;P=D.parent;C=P/'DAT_SPECT_readiness_v1';G=P/'ga_unrestricted_readiness_v4_10kgen_2026-11-01';GR=P/'ga_unrestricted_run_v4_10kgen_2026-11-01';GV=P/'ga_unrestricted_verification_v4_2026-11-01'
assert not (D/'LOCK.json').exists()
sources={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);sources[str(p)]=sha(p);return p
def load(p):return json.loads(bind(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def tab(p,sep=','):return pd.read_csv(bind(p),sep=sep,dtype=str,keep_default_na=False)
oldlock=load(C/'LOCK.json');assert all(sha(C/p)==h for p,h in oldlock['files'].items())
B=load(G/'config/Input_Binding.json');A=load(G/'config/Analysis_Contract.json')
for path,record in B['files'].items():assert sha(path)==record['sha256']
prior=load(GV/'restricted/Pre_Verification_Source_Manifest.json')
for path in [GR/'results/Selections.json',GR/'RUN.json']:
 assert sha(path)==prior[str(path)]['sha256'];bind(path)
status=load(GV/'Verification_Status.json');assert status['discrepancies_affecting_reported_results']==0
x=tab(B['candidate_matrix'],'\t');assert x.PATNO.is_unique and x.component_id.is_unique
base=x[['PATNO','component_id','RNA_index_month']].rename(columns={'component_id':'component','RNA_index_month':'index_month'});base.index_month=base.index_month.astype(int)
F=P.parent/'strategy_redesign_v2_2026-10-20/download_resume_2026-10-20/restricted/ppmi/imaging/files'
raw=[]
for name in ['Xing_Core_Lab_-_DATSCAN_acquisition_metadata_20Oct2026.csv','Xing_Core_Lab_-_Quant_SBR_20Oct2026.csv','Xing_Core_Lab_-_Visual_Read_20Oct2026.csv']:
 t=tab(F/name);t['_row']=np.arange(len(t))+2;raw.append(t)
fields=load(C/'Protocol.json')['DAT_fields'];bind(C/'audit_and_bind.py')
audit,values,trace=select_scans(base,*raw,fields)
audit.to_csv(D/'restricted/Scan_Audit.csv',index=False);save(D/'restricted/Selected_Source_Rows.json',trace)
# Reconcile the reused algorithm against old results wherever the baseline is identical.
previous=tab(C/'restricted/Scan_Audit.csv');cross=audit.merge(previous,on='PATNO',suffixes=('_new','_old'))
same=cross[cross.index_month_new.astype(int).eq(cross.index_month_old.astype(int))]
for col in ['eligible','strict_preindex','reasons','flags','event','protocol']:
 assert same[col+'_new'].astype(str).eq(same[col+'_old'].astype(str)).all(),col
coh=[];allcoh=[];counts=[];folds=[];exclusions=[]
for h,path in B['cohorts'].items():
 z=tab(path,'\t').rename(columns={'component_id':'component','horizon_months':'horizon','outer_fold':'outer','outcome_source_row':'endpoint_row'})
 for col in ['horizon','outer','endpoint_row','elapsed_months']+[f'inner_for_outer_{j}' for j in range(5)]:z[col]=z[col].astype(int)
 z.y=z.y.astype(float);allcoh.append(z.copy())
 q=z.merge(audit[['component','eligible','index_month','scan_month','scan_minus_index','strict_preindex','reasons','protocol']],on='component',validate='one_to_one')
 q['endpoint_month']=q.index_month+q.elapsed_months
 counts.append(dict(horizon=int(h),original_ga_n=len(z),matched_n=int(q.eligible.sum()),strict_preindex_n=int(q.strict_preindex.sum())))
 for reason,n in q.loc[~q.eligible,'reasons'].str.split(';').explode().value_counts().items():exclusions.append(dict(horizon=int(h),reason=reason,n=int(n)))
 coh.append(q[q.eligible].copy())
m=pd.concat(coh);origin=pd.concat(allcoh);ids=sorted(set(m.component));xx=x.rename(columns={'component_id':'component'}).set_index('component').loc[ids]
for f in fields:xx[f]=[values[i][f] for i in ids]
xx.reset_index().to_csv(D/'restricted/Predictors.csv',index=False)
scols=['component','outer']+[f'inner_for_outer_{i}' for i in range(5)]
s=m[scols].drop_duplicates();assert s.component.is_unique;s.to_csv(D/'restricted/Grouped_Splits.csv',index=False)
mcols=['component','horizon','PATNO','index_month','endpoint_month','endpoint_row','scan_month','scan_minus_index','strict_preindex','protocol']
m[mcols].to_csv(D/'restricted/Membership.csv',index=False);m[['component','horizon','endpoint_row','y']].to_csv(D/'restricted/Endpoints.csv',index=False)
origin.to_csv(D/'restricted/Original_GA_Membership.csv',index=False)
reserved=tab(C/'restricted/Stage_A_Reserved.csv');assert not set(ids)&set(reserved.component);reserved.to_csv(D/'restricted/Stage_A_Reserved.csv',index=False)
selected=[r for r in load(GR/'results/Selections.json') if r['arm']=='unrestricted28_GA_selected'];assert len(selected)==15
names=[r['name'] for r in A['features']['optional']];assert len(names)==28
panels={}
def panel(ns):
 nums=[r['name'] for r in A['features']['optional'] if r['name'] in ns and r['type']=='numeric'];cats={k:v for k,v in A['features']['feature_categories'].items() if k in ns}
 cats.update({k:v['levels'] for k,v in A['features']['nuisance_categories'].items() if set(ns)&set(v['trigger'])})
 return dict(numeric=nums,categories=cats)
for r in selected:
 ns=[n for i,n in enumerate(names) if r['mask']&(1<<i)];assert ns==r['features']
 panels[str(r['horizon'])+'/'+str(r['outer'])]=dict(panel=panel(ns),features=ns,mask=r['mask'],historical_alpha=r['alpha'])
assert set(panels)=={str(h)+'/'+str(j) for h in [12,24,36] for j in range(5)}
# Selection depends only on each original outer-training cohort. Record those source identities.
selection_scope=[]
for h in [12,24,36]:
 for j in range(5):
  full=origin[origin.horizon.eq(h)];train=set(full.loc[full.outer.ne(j),'component']);test=set(full.loc[full.outer.eq(j),'component']);assert not train&test
  matched=m[m.horizon.eq(h)];assert set(matched.loc[matched.outer.eq(j),'component'])<=test
  selection_scope.append(dict(horizon=h,outer=j,original_training_components=sorted(train),original_test_components=sorted(test)))
save(D/'restricted/Selection_Training_Scope.json',selection_scope)
plan=dict(study_id='Clinical_hallmark_DAT_SPECT_GA28_v1',status='LOCKED_NO_STUDY_TRAINING',horizons=[12,24,36],primary_horizon=36,primary_contrast='GA_selected_DAT minus GA_selected, delta R2; higher favors DAT',key_secondary='same contrast delta MAE; lower favors DAT',
 arms=['GA_selected','GA_selected_DAT','full28','full28_DAT'],views={v:dict(n=[r['matched_n' if v=='month_compatible' else 'strict_preindex_n'] for r in counts]) for v in ['month_compatible','strict_preindex']},
 selection='Reuse the verified unrestricted28 GA-selected mask for the matching original horizon and outer fold. No global consensus mask, new GA search, selection using imaging or reuse across outer folds. Prior selection used broader original outer-training cohort, not only the imaging subset. Held-out identities were excluded from that original selection.',
 prior_selection_scope='Frozen legacy fold-specific selections; inner validation in this extension tunes ridge strength conditional on that selected mask. Do not describe this as a de novo nested GA search confined to the imaging subset.',
 frozen_fold_panels=panels,full28_panel=panel(names),feature_universe=names,DAT_fields=fields,
 penalties=A['model']['lambdas'],penalty_selection='minimum pooled inner MSE, tie within 1e-12 favors largest penalty; separately for each arm using identical matched rows and folds',
 ridge_objective=A['model']['objective'],preprocessing=A['preprocessing'],splits='Original GA identity grouped master, five outer and three inner; unchanged subsets across arms/horizons/views. NOT the later Stage C folds.',
 cohort='Original GA RNA-landmarked population, with imaging eligibility reaudited at original RNA_index_month; RNA is not a predictor.',
 scan_rules='Exact audited native Xing SC/BL selection and QC from DAT_SPECT_readiness_v1, re-evaluated at original GA landmark; same four CWM SBRs, no fallback or imaging imputation.',
 strict_timing='Same chosen scan; require earlier calendar month; no reselection.',endpoint='Exact original GA education-adjusted MoCA rows; elapsed months within horizon +/-3. No endpoint updates.',
 bootstrap=5000,seed_base=20261119,seed_rule='base + 100*view ordinal (1,2) + horizon',metrics=['MAE','RMSE','R2','bias','calibration_slope','calibration_intercept','outside_0_30'],paired_contrasts=[['GA_selected_DAT','GA_selected'],['full28_DAT','full28']],
 planned_fits=1920,fit_budget='3 horizons x 2 timing views x 4 fitted arms x 5 outer folds x (3 inner folds x 5 penalties + 1 outer refit)',
 uncertainty='95% paired percentile intervals conditional on fixed OOF predictions; PCG64, linear quantiles; require >=95% finite draws; secondary exploratory without multiplicity-adjusted significance claims.',
 scope='Exploratory internal incremental-value evaluation of previously exposed PPMI participants. Not external validation or untouched data. Four-score and enhanced imaging findings already observed; this is a new post-results hypothesis, locked before its own fitting.',
 horizon_48='Not included: no frozen 48-month GA selection exists. Reusing 36-month selections or launching a new 48-month search requires a separate contract.',training_authorized=False,paid_queries=False)
save(D/'Protocol.json',plan)
pd.DataFrame(counts).to_csv(D/'Cohort_Counts.csv',index=False);pd.DataFrame(exclusions).to_csv(D/'Exclusion_Reasons.csv',index=False)
for v in plan['views']:
 for h in plan['horizons']:
  q=m[m.horizon.eq(h)]
  if v=='strict_preindex':q=q[q.strict_preindex]
  for j in range(5):
   tr=q[q.outer.ne(j)];te=q[q.outer.eq(j)];assert len(tr)>0 and len(te)>0
   assert te[f'inner_for_outer_{j}'].eq(-1).all() and set(tr[f'inner_for_outer_{j}'])=={0,1,2}
   folds.append(dict(view=v,horizon=h,outer=j,train_n=len(tr),test_n=len(te),**{f'inner_{k}':int(tr[f'inner_for_outer_{j}'].eq(k).sum()) for k in range(3)}))
pd.DataFrame(folds).to_csv(D/'Fold_Counts.csv',index=False)
save(D/'Binding_Audit.json',dict(status='PASS',original_frame=len(x),matched_unique=len(ids),same_landmark_reconciled=len(same),changed_landmarks_in_shared_frame=len(cross)-len(same),original_GA_outer_folds_preserved=True,source_selections=15,fold_cells=len(folds),reserved_overlap=0,study_fits=0))
for path in [C/'Source_SHA256.json',G/'PLAN_LOCK.json',G/'LAUNCHER_LOCK.json',GV/'VERIFICATION_SEAL.json']:bind(path)
save(D/'Source_SHA256.json',sources);assert all(sha(p)==h for p,h in sources.items())
print(json.dumps(counts));print((D/'Binding_Audit.json').read_text())
