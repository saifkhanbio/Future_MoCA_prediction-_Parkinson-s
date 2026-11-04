"""Independent refresh validation with pandas; does not import the amendment builder."""
from pathlib import Path
import pandas as pd,json,math,re,hashlib
D=Path(__file__).resolve().parents[1];P=D.parent
T=P/'clinical_temporal_validation_contract_2026-10-28';A=P/'additional_ppmi_eligibility_prior_use_2026-10-28'
def frame(p):return pd.read_csv(p,sep='\t' if Path(p).suffix=='.tsv' else ',',dtype=str,keep_default_na=False)
def integer(s,upper):
 try:x=float(s);return math.isfinite(x) and x==int(x) and 0<=x<=upper
 except (TypeError,ValueError):return False
def month(s):
 if not re.fullmatch(r'[0-9]{2}/[0-9]{4}',s):return None
 m,y=map(int,s.split('/'))
 return y*12+m if 1<=m<=12 and 1900<=y<=2100 else None
def norm(s):
 try:
  x=float(s)
  if math.isfinite(x):return x
 except (TypeError,ValueError):pass
 return str(s or '').strip()
inputs=json.loads((D/'config/Source_Inputs.json').read_text());raw=frame(inputs['moca']);old=frame(inputs['moca_prior_version'])
assert raw.REC_ID.is_unique and old.REC_ID.is_unique
raw['month']=raw.INFODT.map(month);raw['csv_row']=range(2,len(raw)+2)
items=json.loads(Path(inputs['prior_cognitive_contract']).read_text())['cognition']['item_max']
fields=list(items)+['MCATOT','MCAVFNUM']
base=frame(A/'restricted/Baseline_Audit.tsv').set_index('PATNO')
resolved=frame(T/'restricted/Baseline_Resolved_570.tsv').set_index('PATNO')
audit=frame(D/'restricted/Endpoint_Audit_Refreshed.tsv').set_index(['PATNO','horizon_months'])
checks=0;sets={12:set(),24:set(),36:set()}
for p,b in base.iterrows():
 rows=raw[raw.PATNO==p];index=int(b.index_month)
 for h in [12,24,36]:
  a=audit.loc[(p,str(h))]
  rows_h=rows[(rows.month>=index+h-3)&(rows.month<=index+h+3)].copy()
  rows_h['distance']=abs(rows_h.month-index-h)
  rows_h=rows_h.sort_values(['distance','month','EVENT_ID','csv_row'])
  valid=False;expected=None;chosen=None
  if len(rows_h):
   first=rows_h.iloc[0];ties=rows_h[(rows_h.distance==first.distance)&(rows_h.month==first.month)&(rows_h.EVENT_ID==first.EVENT_ID)]
   payloads={tuple(norm(r[k]) for k in fields) for _,r in ties.iterrows()}
   if len(payloads)==1:
    chosen=first
    item_ok=all(integer(first[k],v) for k,v in items.items())
    if item_ok:
     total=sum(int(float(first[k])) for k in items);edu=b.education
     education_ok=integer(edu,40) or total==30
     if education_ok:
      expected=min(30,total+(1 if total<30 and integer(edu,40) and float(edu)<=12 else 0))
      total_ok=not first.MCATOT or integer(first.MCATOT,30) and float(first.MCATOT) in [total,expected]
      fluency_ok=not first.MCAVFNUM or integer(first.MCAVFNUM,10000) and int(float(first.MCAVF))==int(float(first.MCAVFNUM)>=11)
      valid=total_ok and fluency_ok
    if b.death_month and first.month>int(b.death_month):valid=False
    if b.death_conflict=='True' or b.invalid_death_date=='True':valid=False
    if b.earliest_withdrawal_month and first.month>int(b.earliest_withdrawal_month):valid=False
  eligible=valid and b.matched_baseline_ready=='True'
  assert (a.matched_eligible_optional_missing_allowed=='True')==eligible;checks+=1
  assert a.source_REC_ID==(chosen.REC_ID if chosen is not None else '');checks+=1
  if eligible:
   assert float(a.education_adjusted_MoCA)==expected;checks+=1;sets[h].add(p)
for variant,bm in [('Resolved',resolved),('Strict',base)]:
 f=frame(D/f'restricted/Target_Features_{variant}.tsv').set_index('PATNO')
 for p,r in f.iterrows():
  for k,v in r.items():assert norm(v)==norm(bm.loc[p,k]);checks+=1
eligible=set.union(*sets.values());common=set.intersection(*sets.values());assert len(eligible)==227 and len(common)==33
assert [len(sets[h]) for h in [12,24,36]]==[213,175,39]
f=frame(D/'restricted/Target_Features_Resolved.tsv')
assert set(f.PATNO)==eligible and f.component_id.is_unique
roles=frame(D/'restricted/Development_Validation_Roles.tsv')
assert roles.PATNO.is_unique and roles.component_id.is_unique
original_unchanged=0;gains=[]
newid=raw.set_index('REC_ID');oldrow={str(i+2):r for i,r in old.iterrows()}
for h in [12,24,36]:
 c=frame(D/f'restricted/Validation_Cohort_{h}m.tsv').set_index('PATNO')
 o=frame(D/f'restricted/Sealed_Outcomes_{h}m.tsv').set_index('PATNO')
 assert set(c.index)==set(o.index)==sets[h]
 for p,r in o.iterrows():
  source=newid.loc[r.source_REC_ID]
  assert source.PATNO==p and int(source.csv_row)==int(r.source_row)
  assert int(source.month)==int(r.assessment_month)
  assert float(r.education_adjusted_MoCA)==float(audit.loc[(p,str(h)),'education_adjusted_MoCA']);checks+=4
  assert c.loc[p,'component_id']==base.loc[p,'component_id'];checks+=1
 for r in frame(T/f'restricted/Sealed_Outcomes_{h}m.tsv').to_dict('records'):
  x=o.loc[r['PATNO']];s=oldrow[r['source_row']]
  assert s.REC_ID==x.source_REC_ID and norm(r['education_adjusted_MoCA'])==norm(x.education_adjusted_MoCA)
  assert tuple(norm(s[k]) for k in fields+['PATNO','INFODT','EVENT_ID'])==tuple(norm(newid.loc[s.REC_ID,k]) for k in fields+['PATNO','INFODT','EVENT_ID'])
  original_unchanged+=1;checks+=3
 for p in sets[h]:
  if p not in set(frame(T/f'restricted/Validation_Cohort_{h}m.tsv').PATNO):gains.append((p,h))
views=json.loads((D/'config/Evaluation_Views.json').read_text())
for v in views:
 h=v['horizon_months'];name=v['view_id'];expected=sets[h]
 if name.startswith('complete_core'):expected={p for p in expected if resolved.loc[p,'resolved_core_complete']=='True'}
 elif name.startswith('exclude_age_resolved'):expected={p for p in expected if resolved.loc[p,'age_resolution']!='preindex_birth_month_corroborated'}
 elif name.startswith('common_horizons'):expected=common
 members=frame(D/v['member_manifest'])
 assert set(members.PATNO)==expected and members.component_id.is_unique and len(members)==v['expected_n'];checks+=3
 if v['reuse_alias']:
  alias=next(x for x in views if x['view_id']==v['reuse_alias'])
  assert frame(D/alias['member_manifest']).equals(members);checks+=1
assert original_unchanged==420 and len(gains)==7
# Baseline score record bindings include shifted CSV row positions, using stable identifiers.
binding=frame(D/'restricted/Baseline_Record_Rebinding.tsv')
for r in binding.to_dict('records'):
 oldr=oldrow[r['old_row']];newr=newid.loc[r['REC_ID']]
 assert oldr.REC_ID==r['REC_ID'] and oldr.PATNO==r['PATNO']==newr.PATNO
 assert int(newr.csv_row)==int(r['new_row']);checks+=2
out={'status':'PASS','independent_implementation':'Pandas sort/selection and independent item-level scoring; no builder imports',
 'assertion_checks':checks,'candidate_horizon_pairs':len(audit),'horizon_counts':[len(sets[h]) for h in [12,24,36]],
 'unique_participants':len(eligible),'common_horizons':len(common),'original_endpoints_unchanged':original_unchanged,
 'new_followup_pairs':len(gains),'target_model_predictions':0,'target_model_metrics':0}
(D/'qa/Independent_Amendment_Verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
