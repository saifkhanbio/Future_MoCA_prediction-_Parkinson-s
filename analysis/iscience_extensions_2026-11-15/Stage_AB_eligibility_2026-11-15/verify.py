"""Verify membership sets and aggregate counts without generating predictions."""
from pathlib import Path
import sys,csv,json,hashlib,collections
C=Path(sys.argv[1]); N=C.parent; O=Path(__file__).resolve().parent
def rows(p,sep=','):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=sep))
def hsh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def ids(rs,key):return {r[key] for r in rs}
hs=(12,24,36)
v=json.loads((O/'Source_SHA256.json').read_text())
assert all(hsh(N/p)==d for p,d in v.items())
a=rows(O/'restricted/A_membership.csv'); b=rows(O/'restricted/B_membership.csv')
panel=json.loads((C/'independent_validation_contract_2026-10-31/config/Feature_Definitions.json').read_text())['active_numeric']
registry=C/'temporal_validation_readiness_2026-10-28/restricted'
q=(N/'strategy_redesign_v2_2026-10-20/source_audit_2026-10-20/raw/amp_pdrd')
def src(n):return json.loads((q/('2023_v4release_1027__'+n+'.json')).read_text())
demo={r['participant_id']:r for r in src('Demographics') if float(r['visit_month'])==0}
mo=collections.defaultdict(list)
for r in src('MOCA'):mo[r['participant_id'],int(float(r['visit_month']))].append(r)
old=rows(C/'external_validation_search_2026-10-28/restricted/AMP_Cognitive_Availability_Ledger.tsv','\t')
s1=rows(O/'A_counts.csv');s2=rows(O/'B_counts.csv')
for j,h in enumerate(hs):
 aa=[r for r in a if int(r['horizon'])==h]; bb=[r for r in b if int(r['horizon'])==h]
 assert ids(aa,'PATNO')==ids(rows(registry/f'Validation_Cohort_{h}m.tsv','\t'),'PATNO')
 assert len(ids(aa,'component'))==len(aa)
 for k,features in panel.items():
  nn=sum(all(r[z]=='True' for z in features) and (k=='compact4' or r['JLO_form_observed']=='True') and (k!='core_plus18' or r['motor_state_observed']=='True') for r in aa)
  assert nn==int(s1[j]['compact' if k=='compact4' else k+'_fully_observed'])
 prior=[r for r in old if r['study']=='PDBP' and r['baseline_PD']=='True' and r['PPMI_linked']=='False' and r[f'pair_{h}_domain_QC']=='True']
 expected=set()
 for p in ids(prior,'participant_id'):
  d=demo[p];x=mo[p,0];y=mo[p,h]
  if len(x)!=1 or len(y)!=1:continue
  if d['sex'] not in ('Male','Female') or not 18<=float(d['age_at_baseline'])<=89 or d['education_level_years'] not in ['12-16 years','Greater than 16 years']:continue
  if all(r['education_12years_complete']=='Greater than 12 years of education' and str(r['code_education_12years_complete'])=='0' for r in x+y):expected.add(p)
 got={r['participant_id'] for r in bb if r['higher_education_both']=='True'}
 assert got==expected
 retained=[r for r in bb if r['identity_clear']=='True']; assert len(ids(retained,'component'))==len(retained)==int(s2[j]['identity_clear'])
 assert expected-ids(retained,'participant_id')=={r['participant_id'] for r in bb if r['higher_education_both']=='True' and r['identity_flag_released']=='True'}
def coverage(rs,key):
 groups=[{r[key] for r in rs if int(r['horizon'])==h} for h in hs]
 return {'unique':len(set.union(*groups)),'all_three':len(set.intersection(*groups))}
result={'verification':'passed','inputs_with_unchanged_hashes':len(v),'A':coverage(a,'component'),'B_nominal':coverage([r for r in b if r['identity_clear']=='True'],'component'),'checks':['PPMI ID sets against amended horizon manifests','PPMI panel intersections from observed flags','PDBP education intersections from original availability ledger and native AMP labels','one identity component per retained row per horizon','identity exclusions accounted for','all bound audit input hashes rechecked'],'model_fits':0,'predictions':0,'cloud_queries':0}
(O/'Verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result)
