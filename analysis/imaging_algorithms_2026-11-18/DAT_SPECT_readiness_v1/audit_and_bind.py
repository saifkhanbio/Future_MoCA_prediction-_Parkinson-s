"""DAT-SPECT source/timing/QC audit and matched protocol assembly. Never fits."""
import os,json,hashlib,csv,math,collections,copy
from pathlib import Path
import pandas as pd,numpy as np
os.umask(0o077)
O=Path(__file__).resolve().parent;C=O.parent;P=C.parent
if (O/'LOCK.json').exists():raise SystemExit('Contract is locked; create a new amendment rather than overwrite')
R=O/'restricted';R.mkdir(exist_ok=True)
F=P/'strategy_redesign_v2_2026-10-20/download_resume_2026-10-20/restricted/ppmi/imaging/files'
S=C/'clinical_extensions/Stage_C_protocol_v1'
sources={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):sources[str(p)]=sha(p);return p
def read(p):
 d=pd.read_csv(bind(p),dtype=str,keep_default_na=False);d['_row']=np.arange(len(d))+2;return d
def num(v):
 try:x=float(v);return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def month(v):
 try:
  a,b=v.split('/');m,y=int(a),int(b)
  return y*12+m if len(a)==2 and len(b)==4 and 1<=m<=12 and 1900<=y<=2100 else None
 except:return None
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def frame(p,d):pd.DataFrame(d).to_csv(p,index=False)
ac=read(F/'Xing_Core_Lab_-_DATSCAN_acquisition_metadata_20Oct2026.csv')
qt=read(F/'Xing_Core_Lab_-_Quant_SBR_20Oct2026.csv')
vi=read(F/'Xing_Core_Lab_-_Visual_Read_20Oct2026.csv')
fields=['CAUDATE_L_REF_CWM','CAUDATE_R_REF_CWM','PUTAMEN_L_REF_CWM','PUTAMEN_R_REF_CWM']
for d in [ac,qt,vi]:d['_month']=d.DATSCAN_DATE.map(month)
ledger=read(C/'Fourth_year_extension_v1/audit/restricted/horizon_ledger.csv')
co=ledger[ledger.eligible.eq('True')&ledger.reserved.eq('False')].copy()
assert co.groupby('horizon').size().tolist()==[777,716,651,469]
original=json.loads(bind(S/'protocol.json').read_text())
x=read(S/'restricted/predictors.csv').drop(columns='_row').set_index('component')
sp=read(S/'restricted/grouped_splits.csv').drop(columns='_row').set_index('component')
reserve=set(read(S/'restricted/stage_A_reserved.csv').component)
assert len(reserve)==227 and not reserve&set(co.component)
base=co[['PATNO','component','index_month']].drop_duplicates()
assert base.PATNO.is_unique and base.component.is_unique
audit=[];values={};trace=[]
for b in base.itertuples():
 p=b.PATNO;ix=int(b.index_month);reasons=[];flags=[]
 aa=ac[ac.PATNO.eq(p)];qq=qt[qt.PATNO.eq(p)]
 # Select from union including acquisitions lacking quantitative output; no fallback after QC.
 aa=aa[aa.EVENT_ID.isin(['SC','BL'])];qq=qq[qq.EVENT_ID.isin(['SC','BL'])]
 invalid_timed=aa._month.isna().any() or qq._month.isna().any()
 cand=pd.concat([aa[['EVENT_ID','_month']],qq[['EVENT_ID','_month']]]).drop_duplicates()
 cand=cand[cand._month.between(ix-3,ix)].copy()
 chosen_month=None;event='';protocol='';a=ac.iloc[:0];q=qt.iloc[:0];visual='';val=[None]*4
 if invalid_timed:reasons.append('invalid_SC_BL_date')
 if cand.empty:reasons.append('no_preindex_scan_in_window')
 else:
  chosen_month=int(cand._month.max());event='BL' if cand[cand._month.eq(chosen_month)].EVENT_ID.eq('BL').any() else 'SC'
  a=aa[aa._month.eq(chosen_month)&aa.EVENT_ID.eq(event)]
  q=qq[qq._month.eq(chosen_month)&qq.EVENT_ID.eq(event)]
  prots=set(a.PROTOCOL)|set(q.PROTOCOL)
  if len(prots)!=1 or '' in prots:reasons.append('ambiguous_protocol_binding')
  else:protocol=next(iter(prots))
  if a.empty:reasons.append('no_acquisition_binding')
  else:
   # Same month has no scan UID. Conflicting acquisition signatures cannot safely be collapsed.
   signature=[k for k in ac.columns if k not in ['_row','_month','PATNO']]
   if len(a[signature].drop_duplicates())>1:reasons.append('multiple_conflicting_acquisitions')
   elif len(a)>1:flags.append('identical_acquisition_duplicates')
   if not a.DATSCAN_IMAGE_ACCEPTABLE.eq('Yes').all():reasons.append('acquisition_not_explicitly_acceptable')
   if not a.DATSCAN_LIGAND.isin(['','123I-DaTscan']).all():reasons.append('acquisition_tracer_conflict')
   if a.DATSCAN_LIGAND.eq('').any():flags.append('acquisition_tracer_missing_quant_corroboration_required')
   if not a.PREVIOUSLY_ACQUIRED.isin(['','No']).all():reasons.append('acquisition_prior_scan_conflict')
   if a.PREVIOUSLY_ACQUIRED.eq('').any():flags.append('acquisition_prior_flag_missing')
   for z in a.to_dict('records'):
    inj=z['DATSCAN_DATE_INJECTED']
    if inj and month(inj)!=chosen_month:reasons.append('injection_scan_month_discordance')
    if not inj:flags.append('injection_month_missing')
    dose=z['DATSCAN_TOTAL_DOSE_CALC']
    if dose and (num(dose) is None or num(dose)<=0 or z['DATSCAN_TOTAL_DOSE_UNIT']!='MBq'):reasons.append('dose_unit_or_value_review')
    elif not dose:flags.append('dose_missing')
  if q.empty:reasons.append('no_quantitative_binding')
  else:
   sig=['PROTOCOL','DATSCAN_LIGAND','PREVIOUSLY_ACQUIRED','DATSCAN_ANALYZED','DATSCAN_NOT_ANALYZED_REASON','DATSCAN_OTHER_SPECIFY']+fields
   if len(q[sig].drop_duplicates())>1:reasons.append('conflicting_quantitative_records')
   elif len(q)>1:flags.append('identical_quantitative_duplicates')
   if not q.DATSCAN_ANALYZED.eq('Yes').all():reasons.append('not_analyzed')
   if not q.DATSCAN_LIGAND.eq('123I-DaTscan').all():reasons.append('incompatible_tracer')
   if not q.PREVIOUSLY_ACQUIRED.eq('No').all():reasons.append('prior_acquisition_or_unknown')
   if not q.DATSCAN_NOT_ANALYZED_REASON.eq('').all() or not q.DATSCAN_OTHER_SPECIFY.eq('').all():reasons.append('quantitative_reason_or_comment_review')
   val=[num(q.iloc[0][k]) for k in fields]
   if any(v is None or v < -1 for v in val):reasons.append('missing_or_physically_invalid_SBR')
   if any(v is not None and v<0 for v in val):flags.append('negative_SBR_retained')
  if chosen_month==ix:flags.append('same_month_order_unknown')
  v=vi[vi.PATNO.eq(p)&vi._month.eq(chosen_month)&vi.PROTOCOL.eq(protocol)]
  visual=';'.join(sorted(set(v.DATSCAN_VISINTRP))) if len(v) else 'not_released'
  if len(set(v.DATSCAN_VISINTRP))>1:flags.append('visual_classification_discordance_not_QC_failure')
 row=dict(PATNO=p,component=b.component,index_month=ix,scan_month=chosen_month,
  scan_minus_index=None if chosen_month is None else chosen_month-ix,event=event,protocol=protocol,
  eligible=not reasons,strict_preindex=not reasons and chosen_month<ix,
  reasons=';'.join(sorted(set(reasons))),flags=';'.join(sorted(set(flags))),visual_classification=visual,
  acquisition_rows=';'.join(map(str,a._row)),quantitative_rows=';'.join(map(str,q._row)))
 audit.append(row)
 if not reasons:values[b.component]=dict(zip(fields,val))
 trace.append(dict(component=b.component,acquisitions=a.drop(columns='_month').to_dict('records'),quantitative=q.drop(columns='_month').to_dict('records')))
ad=pd.DataFrame(audit);frame(R/'Scan_Audit.csv',ad);dump(R/'Selected_Source_Rows.json',trace)
joined=co.merge(ad.drop(columns=['PATNO','index_month']),on='component',suffixes=('_clinical','_imaging'),validate='many_to_one')
accepted=joined[joined.eligible_imaging].copy()
frame(R/'Horizon_Audit.csv',joined)
counts=[];reason_counts=[];strata=[]
for h in ['12','24','36','48']:
 z=joined[joined.horizon.eq(h)]
 counts.append(dict(horizon=int(h),clinical_n=len(z),matched_n=int(z.eligible_imaging.sum()),strict_preindex_n=int(z.strict_preindex.sum()),same_month_n=int((z.eligible_imaging&z.scan_minus_index.eq(0)).sum())))
 for reason,n in z[~z.eligible_imaging].reasons.str.split(';').explode().value_counts().items():reason_counts.append(dict(horizon=int(h),reason=reason,n=int(n)))
 for k,d in z[z.eligible_imaging].groupby(['protocol','event','scan_minus_index']):strata.append(dict(horizon=int(h),protocol=k[0],event=k[1],scan_minus_index=int(k[2]),n=len(d)))
frame(O/'Cohort_Counts.csv',counts);frame(O/'Exclusion_Reasons.csv',reason_counts);frame(O/'Timing_Protocol_Strata.csv',strata)
ids=sorted(set(accepted.component));xx=x.loc[ids].copy()
for f in fields:xx[f]=[values[i][f] for i in ids]
frame(R/'Predictors.csv',xx.reset_index());frame(R/'Grouped_Splits.csv',sp.loc[ids].reset_index())
# Bind exactly the prior source endpoints; no new outcome definition.
m1=read(S/'restricted/membership.csv').drop(columns='_row')
e1=read(S/'restricted/endpoints.csv').drop(columns='_row')
Y=C/'Fourth_year_extension_v1/protocol/restricted'
membership=pd.concat([m1,read(Y/'membership.csv').drop(columns='_row')],ignore_index=True)
endpoints=pd.concat([e1,read(Y/'endpoints.csv').drop(columns='_row')],ignore_index=True)
keys=accepted[['component','horizon','strict_preindex','protocol','scan_month','scan_minus_index']]
bound=keys.merge(membership,on=['component','horizon'],validate='one_to_one')
ep=keys[['component','horizon']].merge(endpoints,on=['component','horizon'],validate='one_to_one')
assert len(ep)==len(keys) and bound.endpoint_row.tolist()==ep.endpoint_row.tolist()
frame(R/'Membership.csv',bound);frame(R/'Endpoints.csv',ep);frame(R/'Stage_A_Reserved.csv',[{'component':v} for v in sorted(reserve)])
panels={k:copy.deepcopy(original['panels'][k]) for k in ['compact4','enhanced','moca_only']}
for k in ['compact4','enhanced']:
 panels[k+'_DAT']=copy.deepcopy(panels[k]);panels[k+'_DAT']['numeric']+=fields
arms=['persistence','moca_only','compact4','compact4_DAT','enhanced','enhanced_DAT']
plan=dict(study_id='Clinical_hallmark_DAT_SPECT_v1',status='LOCKED_NO_TRAINING',horizons=[12,24,36,48],
 primary_horizon=48,primary_contrast='compact4_DAT minus compact4, raw MAE',key_secondary='enhanced_DAT minus enhanced, 48 months',
 interpretation='Exploratory internal incremental-value evaluation; no untouched or external validation claim',
 panels=panels,views={v:dict(n=[r['matched_n' if v=='month_compatible' else 'strict_preindex_n'] for r in counts],models=arms) for v in ['month_compatible','strict_preindex']},
 DAT_fields=fields,reference='cerebral white matter (CWM)',feature_source='single native Xing SBR export, no OCC or AMP duplicates',
 timing='Latest acquisition-or-quantitative SC/BL month in index-3 through index; BL priority for same-month tie; selection before QC; no fallback',
 strict_timing='Same selected scan; require scan_month < index_month, no reselection',
 same_month_scope='Month-compatible baseline information; within-month ordering cannot be certified.',
 preprocessing='Training-only numeric median (0 if all missing), mean/population-SD scaling (1 if <=1e-12), missing indicators and fixed categories; DAT observed for every participant',
 penalties=original['penalties'],ridge_objective=original['ridge_objective'],inner_selection=original['inner_selection'],
 splits='Exact inherited Stage C identity master: five outer, three inner; subset only; same assignments for every arm/horizon/view',
 bootstrap=5000,seed_base=20261118,seed_rule='base + 100 * view ordinal (1,2) + horizon',
 metrics=['MAE','RMSE','R2','bias','calibration_slope','calibration_intercept','outside_0_30'],
 paired_contrasts=[['compact4_DAT','compact4'],['enhanced_DAT','enhanced']],
 uncertainty='95% paired percentile participant intervals conditional on OOF predictions; no full-development resampling; exploratory, not confirmatory; secondary comparisons descriptive',
 planned_fits=3200,training_authorized=False,launcher_validated=False,
 restrictions=['No GA, feature selection, clipping, tuning-grid expansion or outcome-informed QC','No imputation of missing DAT to inflate sample size','Do not interpret visual normality as image rejection','No paid queries; prior artifacts read-only'])
dump(O/'Protocol.json',plan)
folds=[]
for view in plan['views']:
 for h in plan['horizons']:
  z=bound[bound.horizon.eq(str(h))]
  if view=='strict_preindex':z=z[z.strict_preindex]
  a=sp.loc[z.component].astype(int)
  assert len(a)==plan['views'][view]['n'][plan['horizons'].index(h)]
  for j in range(5):
   tr=a[a.outer.ne(j)];te=a[a.outer.eq(j)]
   assert len(tr)>0 and len(te)>0
   col='inner_for_outer_'+str(j)
   assert te[col].eq(-1).all() and set(tr[col])=={0,1,2}
   folds.append(dict(view=view,horizon=h,outer=j,n_train=len(tr),n_test=len(te),**{f'inner_{k}':int(tr[col].eq(k).sum()) for k in range(3)}))
frame(O/'Fold_Counts.csv',folds)
# Hash only methods and upstream files read; no modifications to any previous study.
for p in (P/'optional_modality_overlap_2026-10-04/methods').glob('*SPECT*txt'):bind(p)
for p in [C/'Fourth_year_extension_v1/verification/Verification.json',C/'Fourth_year_extension_v1/protocol/LOCK.json',S/'LOCK.json']:bind(p)
assert all(sha(p)==v for p,v in sources.items())
dump(O/'Source_SHA256.json',sources)
dump(O/'Preflight.json',dict(status='PASS',unique_participants=len(ids),fold_cells=len(folds),reserved_overlap=0,study_fits=0,endpoint_bindings='exact prior frozen rows',source_hashes_unchanged=True,launcher_validated=False))
print(json.dumps(counts));print('Unique matched participants',len(ids))
