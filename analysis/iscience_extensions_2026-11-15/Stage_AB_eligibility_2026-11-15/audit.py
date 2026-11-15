"""Local stage A/B eligibility audit. Invoke with the Clinical_hallmark folder as argument.
Uses QC and availability only. No prediction, coefficient loading or model fitting."""
import sys, csv, json, hashlib, math, re, collections, ast, os
from pathlib import Path
import pandas as pd
import numpy as np
os.umask(0o077)
C=Path(sys.argv[1]); P=C.parent; B=P/'strategy_redesign_v2_2026-10-20'
O=Path(__file__).resolve().parent; R=O/'restricted'; R.mkdir(exist_ok=True)
T=C/'temporal_validation_readiness_2026-10-28/restricted'
A=C/'comprehensive_feature_audit_2026-10-29'
raw=B/'source_audit_2026-10-20/raw/ppmi/core'
amp=raw.parent.parent/'amp_pdrd'
H=(12,24,36); hashes={}
def record(f):
 if str(f.relative_to(P)) not in hashes:
  h=hashlib.sha256()
  with f.open('rb') as k:
   for v in iter(lambda:k.read(1048576),b''):h.update(v)
  hashes[str(f.relative_to(P))]=h.hexdigest()
 return f
def read(f):return pd.read_csv(record(f),sep='\t' if f.suffix=='.tsv' else ',',dtype=str,keep_default_na=False,encoding='utf-8-sig')
def load(f):return json.loads(record(f).read_text(encoding='utf-8-sig'))
def output(name,rows):
 pd.DataFrame(rows).to_csv(O/name,index=False)
def number(x):
 try:v=float(x);return v if math.isfinite(v) else None
 except (TypeError,ValueError):return None
def integer(x,lo,hi):
 v=number(x);return v is not None and v==int(v) and lo<=v<=hi
def only(d):assert d.PATNO.is_unique;return d.set_index('PATNO',drop=False)
frame=only(read(C/'clinical_temporal_validation_contract_2026-10-28/restricted/Baseline_Resolved_570.tsv'))
ep=read(T/'Endpoint_Audit_Refreshed.tsv')[['PATNO','horizon_months','elapsed_months','endpoint_QC_pass','withdrawal_review_hold']]
assert len(ep)==3*len(frame) and not ep.duplicated(['PATNO','horizon_months']).any()
dev=read(C/'model_readiness_2026-10-24/restricted/Participant_Manifest.tsv')
enr=only(read(next(raw.glob('Participant_Status_*.csv'))))
defs=load(C/'independent_validation_contract_2026-10-31/config/Feature_Definitions.json')
groups=defs['active_numeric'];members={}; flow=[]; missing=[]; Strata=[]
for h in H:
 t=ep[ep.horizon_months.eq(str(h))].set_index('PATNO').loc[frame.index]
 later=frame.index_year.isin(['2023','2024','2025'])
 enrolled_age=enr.ENROLL_AGE.reindex(frame.index).map(number)
 age=pd.to_numeric(frame.age_index,errors='coerce')
 adult=age.between(18,110)|(age.isna() & enrolled_age.between(18,110))
 bas=later & frame.baseline_ready.eq('True') & adult
 for f,limit in [('baseline_moca',30),('baseline_sdmt',110),('baseline_hvlt_immediate',36),('baseline_hvlt_delayed',12)]:
  bas &= frame[f].map(lambda v:integer(v,0,limit))
 bas &=frame.baseline_hvlt_form.isin(list('123456')) & frame.baseline_sdmt_form.isin(['1','2']) & frame.enhanced_tests_complete.eq('True')
 timed=pd.to_numeric(t.elapsed_months,errors='coerce').between(h-3,h+3)
 good=bas & timed & t.endpoint_QC_pass.eq('True') & t.withdrawal_review_hold.eq('False')
 ids=set(frame.index[good]); old=set(read(T/f'Validation_Cohort_{h}m.tsv').PATNO)
 assert not ids.intersection(dev.PATNO) and not set(frame.loc[sorted(ids)].component_id).intersection(dev.component_id)
 assert frame.loc[sorted(ids)].component_id.is_unique
 members[h]=ids
 flow.append(dict(horizon=h,frame=len(frame),era=int(later.sum()),baseline_compact_adult=int(bas.sum()),with_timed_followup=int((bas & timed).sum()),with_endpoint_QC=int((bas & timed & t.endpoint_QC_pass.eq('True')).sum()),compact=len(ids),old_temporal=len(old),shared_with_prior_temporal=len(old&ids),previous_temporal_only=len(old-ids),new_compact_only=len(ids-old)))
base=frame.loc[sorted(set.union(*members.values()))].copy().reset_index(drop=True)
base['RNA_index_month']=base['index_month']  # adapter for the old helper; values are clinical index months
legacy=record(C/'ga_readiness_2026-10-30/scripts/build_candidates.py').read_text(encoding='utf-8-sig')
# Copy only pure helper definitions and the baseline feature construction block into this audit.
# No historical launcher or module top-level is executed. Stop before original fit/view bindings.
tree=ast.parse(legacy)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in {'num','month','fields_for','choose','calc'}:
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<baseline QC helper>','exec'))
exec(legacy[legacy.index('prior=load('):legacy.index('def fields_for')])
# Preselect study IDs when reading native baseline forms, preserving original source line numbers in order for choose().
construct=legacy[legacy.index('matrix=base.copy()'):legacy.index('# UPSIT')]
construct=construct.replace("df['_line']=np.arange(len(df))+2;", "df['_line']=np.arange(len(df))+2;df=df[df.PATNO.isin(base.PATNO)].copy();")
exec(construct)
mx=only(matrix);rawcols={k:pd.to_numeric(mx[k],errors='coerce').notna() for k in set(groups['core_plus18'])}
pres=pd.DataFrame(rawcols,index=mx.index)
assert len(D)==13 and set(groups['MoCA_plus18'])<=set(pres.columns)
for stats in flow:
 h=stats['horizon'];ids=sorted(members[h]); complete={}
 for model in ['MoCA_plus18','core_plus18']:
  obs=pres.loc[ids,groups[model]].all(axis=1)&mx.loc[ids,'jlo_form'].isin(['1','2'])
  if model=='core_plus18':obs &= mx.loc[ids,'motor_state'].isin(['untreated','off','on','treated_unknown'])
  complete[model]=set(obs.index[obs])
  stats[model+'_fully_observed']=int(obs.sum())
 for f in groups['core_plus18']:
  missing.append(dict(horizon=h,feature=f,eligible=len(ids),observed=int(pres.loc[ids,f].sum()),missing=int((~pres.loc[ids,f]).sum())))
 for year,sub in mx.loc[ids].groupby('index_year'):
  Strata.append(dict(horizon=h,year=year,participants=len(sub),**{k:int(pres.loc[sub.index,k].sum()) for k in groups['core_plus18']}))
 stats['both_large_complete']=len(complete['MoCA_plus18']&complete['core_plus18'])
output('A_counts.csv',flow);output('A_availability.csv',missing);output('A_availability_by_year.csv',Strata)
priv=[]
for h in H:
 for p in sorted(members[h]):
  priv.append(dict(PATNO=p,component=mx.loc[p,'component_id'],horizon=h,**{k:bool(v) for k,v in pres.loc[p].items()},JLO_form_observed=mx.loc[p,'jlo_form'] in ['1','2'],motor_state_observed=mx.loc[p,'motor_state'] in ['untreated','off','on','treated_unknown']))
output('restricted/A_membership.csv',priv)
print('A',flow,flush=True)

# Stage B. Native release fields are used only for membership, metadata and scoring-rule agreement.
def data(name):return load(amp/('2023_v4release_1027__'+name+'.json'))
people=data('amp_pd_participants'); parent={x['participant_id']:x['participant_id'] for x in people}
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(a,b):
 a,b=root(a),root(b);parent[max(a,b)]=min(a,b)
sources=collections.defaultdict(list)
for x in people:
 for k in [('guid',x['guid']),(x['study'],x['study_participant_id'])]:
  if k[1]:sources[k].append(x['participant_id'])
for ids in sources.values():
 for j in ids[1:]:union(ids[0],j)
for x in data('amp_pd_participant_wgs_duplicates'):union(x['participant_id'],x['duplicate_sample_id'])
def roots_for(study):return {root(x['participant_id']) for x in people if x['study']==study}
ppmi=roots_for('PPMI');lcc=roots_for('LCC')
inventory={x['participant_id']:x for x in people};pdb=[x for x in people if x['study']=='PDBP']
dup=collections.Counter(root(x['participant_id']) for x in pdb)
geno=load(next(amp.glob('*_WB_DWGS__wgs_WB_DWGS_samples.json')))
genokeys={(x['participant_id'],x['sample_id']) for x in geno}
qcs=load(next(amp.glob('*_WB_DWGS__wgs_qc_flags.json')))
bad={root(x['participant_id']) for x in qcs if (x['participant_id'],x['sample_id']) in genokeys and x['participant_id'] in parent and any(x.get(k)=='Yes' for k in ['CLINICAL_SEX_DISCORDANT','DUPLICATE_WITHIN_COHORT','FAILED_SUBSEQUENT_QC','NEUROX_DISCORDANT'])}
diagnoses=collections.defaultdict(set)
for x in data('amp_pd_case_control'):diagnoses[x['participant_id']].add(x['diagnosis_at_baseline'])
pdlabels={"Parkinson's Disease","Idiopathic PD"}
def by_visit(src):
 res=collections.defaultdict(list)
 for r in src:res[(r['participant_id'],number(r['visit_month']))].append(r)
 return res
moca=by_visit(data('MOCA'));dem=by_visit(data('Demographics'))
domain={'moca_visuospatial_executive_subscore':5,'moca_naming_subscore':3,'moca_attention_digits_subscore':2,'moca11_attention_vigilance':1,'moca12_attention_serial_7s':3,'moca_language_subscore':3,'moca_abstraction_subscore':2,'moca_delayed_recall_subscore':5,'moca_orientation_subscore':6}
def domain_ok(r):
 if not integer(r.get('moca_total_score'),0,30) or not integer(r.get('code_education_12years_complete'),0,1):return False
 if not all(integer(r.get(k),0,m) for k,m in domain.items()):return False
 return min(30,sum(number(r[k]) for k in domain)+number(r['code_education_12years_complete']))==number(r['moca_total_score'])
def high(r):return r.get('education_12years_complete')=='Greater than 12 years of education' and number(r.get('code_education_12years_complete'))==0
def valid_total(r):return integer(r.get('moca_total_score'),0,30)
ledger=read(C/'external_validation_search_2026-10-28/restricted/AMP_Cognitive_Availability_Ledger.tsv')
expected={h:set(ledger.loc[ledger.study.eq('PDBP') & ledger.baseline_PD.eq('True') & ledger.PPMI_linked.eq('False') & ledger[f'pair_{h}_domain_QC'].eq('True'),'participant_id']) for h in H}
bp=[]; bs=[]; cols=['prior_recorded_pair','prior_domain_pair','unique_nominal_domain_pair','adult_age_sex','higher_education_baseline','higher_education_both','identity_clear','strict_demographic_band','calendar_verified','full_items_verified']
for h in H:
 domids=set()
 for x in pdb:
  pid=x['participant_id']; comp=root(pid)
  if not diagnoses[pid]&pdlabels or comp in ppmi:continue
  ms=moca[pid,0];fs=moca[pid,h]; ds=dem[pid,0]
  recorded=any(map(valid_total,ms)) and any(map(valid_total,fs))
  orig=any(map(domain_ok,ms)) and any(map(domain_ok,fs))
  if orig:domids.add(pid)
  nominal=orig and len(ms)==len(fs)==1 and ms[0]['visit_name']=='M0' and fs[0]['visit_name']==f'M{h}'
  demo=nominal and len(ds)==1 and ds[0]['visit_name']=='M0' and integer(ds[0]['age_at_baseline'],18,89) and ds[0]['sex'] in ('Male','Female')
  ed=demo and high(ms[0]) and ds[0]['education_level_years'] in ('12-16 years','Greater than 16 years')
  both=ed and high(fs[0])
  # GUID is matched where populated. Empty GUID gives no positive cross-cohort check.
  guidok=bool(x['guid']) and all(not r.get('GUID') or str(r['GUID']).lower()==str(x['guid']).lower() for r in (ms[:1]+fs[:1]+ds[:1]))
  ready=bool(both and guidok and comp not in lcc and dup[comp]==1 and comp not in bad and diagnoses[pid]<=pdlabels)
  bp.append(dict(participant_id=pid,component=comp,horizon=h,**dict(zip(cols,[recorded,orig,nominal,demo,ed,both,ready,bool(ready and ds[0]['education_level_years']=='Greater than 16 years'),False,False])),identity_flag_released=comp in bad,shares_LCC_component=comp in lcc,duplicate_PDBP_component=dup[comp]>1,guid_check=guidok,ambiguous_or_low_baseline_education=demo and not ed,followup_education_conflict=ed and not both))
 assert domids==expected[h]
 sub=[r for r in bp if r['horizon']==h]
 bs.append(dict(horizon=h,baseline_PD_no_known_PPMI=len(sub),**{k:sum(bool(r[k]) for r in sub) for k in cols},wgs_flag_among_education_matched=sum(r['higher_education_both'] and r['identity_flag_released'] for r in sub),missing_or_discordant_guid_among_education_matched=sum(r['higher_education_both'] and not r['guid_check'] for r in sub),shares_lcc_among_education_matched=sum(r['higher_education_both'] and r['shares_LCC_component'] for r in sub)))
output('B_counts.csv',bs);output('restricted/B_membership.csv',bp)
assert [r['prior_recorded_pair'] for r in bs]==[414,332,272] and [r['prior_domain_pair'] for r in bs]==[228,193,155]
(O/'Source_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
print('B',bs,flush=True)

# Site/year coverage audit for optional missingness, using the union of eligible baselines.
# This lower-bound screen counts an instrument as documented when at least one relevant
# baseline in that released center/year has an observed value passing the frozen rules.
centerfile=next((B/'source_audit_2026-10-20/raw/ppmi/recovered_members').glob('Laboratory_Procedures_with_Elapsed_Times_*.csv'))
site=read(centerfile);site=site.loc[site.PATNO.isin(mx.index)&site.EVENT_ID.isin(['SC','BL'])].copy()
site['month_n']=site.INFODT.map(month);site['index_n']=site.PATNO.map(mx.index_month.astype(int))
site=site.loc[(site.index_n-site.month_n).between(0,3) & site.CNO.ne('')]
def sitebind(x):
 vals=set(x.CNO)
 return next(iter(vals)) if len(vals)==1 else None
mapping={i:sitebind(v) for i,v in site.groupby('PATNO')}
strata=collections.defaultdict(list)
for i in mx.index:
 if mapping.get(i):strata[mapping[i],mx.loc[i,'index_year']].append(i)
site_stats=[]
for h in H:
 dct={'horizon':h}
 for model,name in [('MoCA_plus18','19'),('core_plus18','28')]:
  cols=groups[model]; proven=set()
  for i in members[h]:
   narrow=bool(pres.loc[i,cols].all()) and mx.loc[i,'jlo_form'] in ('1','2') and (model!='core_plus18' or mx.loc[i,'motor_state'] in ('untreated','off','on','treated_unknown'))
   peers=strata.get((mapping.get(i),mx.loc[i,'index_year']),[])
   supported=bool(peers) and pres.loc[peers,cols].any(axis=0).all() and mx.loc[peers,'jlo_form'].isin(['1','2']).any() and (model!='core_plus18' or mx.loc[peers,'motor_state'].isin(['untreated','off','on','treated_unknown']).any())
   if narrow or supported:proven.add(i)
  dct[name+'_observed_or_stratum_supported']=len(proven)
  for r in priv:
   if r['horizon']==h:r[name+'_observed_or_stratum_supported']=r['PATNO'] in proven
 dct['mapped_center']=sum(bool(mapping.get(i)) for i in members[h]);site_stats.append(dct)
output('A_site_year_support.csv',site_stats);output('restricted/A_membership.csv',priv)
# Full-item availability and date metadata are independently bound rather than assumed.
clinical_fields=set(next(iter(moca.values()))[0]);items=[k for k in clinical_fields if re.match(r'moca\d\d_',k) and not k.startswith('moca14_')]
assert len(items)==26
assert not any(re.search(r'assessment_date|visit_date|elapsed_days',k,re.I) for k in clinical_fields)
for h in H:
 ids=[r['participant_id'] for r in bp if r['horizon']==h and r['prior_recorded_pair']]
 complete=sum(any(all(number(d.get(k)) is not None for k in items) for d in moca[i,0]) and any(all(number(d.get(k)) is not None for k in items) for d in moca[i,h]) for i in ids)
 assert complete==0
metadata=record(B/'pdbp_followup_metadata_search_2026-10-21/Metadata_Audit_Report.txt')
(O/'Source_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
print('Site/year',site_stats)
