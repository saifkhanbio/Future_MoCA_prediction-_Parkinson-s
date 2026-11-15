"""Build a new, local Stage C package from the verified eligibility audit. Never fit."""
import os,sys,json,csv,math,re,collections,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
os.umask(0o077)
O=Path(__file__).resolve().parent; C=O.parents[2]; P=C.parent
R=O/'restricted'; E=O.parent/'Stage_C_eligibility_v1'
if (O/'LOCK.json').exists():raise SystemExit('Package sealed')
R.mkdir(exist_ok=True)
B=P/'strategy_redesign_v2_2026-10-20'; A=C/'comprehensive_feature_audit_2026-10-29'
def digest(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as fp:
  for b in iter(lambda:fp.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
sources=json.loads((E/'input_SHA256.json').read_text())
def record(f):
 f=Path(f)
 if str(f) in sources:assert digest(f)==sources[str(f)]
 else:sources[str(f)]=digest(f)
 return f
def read(f):
 f=record(f);return pd.read_csv(f,sep='\t' if f.suffix=='.tsv' else ',',dtype=str,keep_default_na=False)
def load(f):return json.loads(record(f).read_text())
def dump(file,value):file.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,frame):frame.to_csv(path,index=False)
assert load(E/'verification.json')['verification']=='PASS'
record(E/'input_SHA256.json');record(E/'verify.py')
defs=load(C/'independent_validation_contract_2026-10-31/config/Feature_Definitions.json')
parent=load(C/'model_readiness_2026-10-24/config/Analysis_Contract.json')
record(O.parent/'Stage_AB_protocols_v1_2026-11-15/evaluation.py')
record(O.parent/'Stage_AB_protocols_v1_2026-11-15/Protocol_A.json')
groups=defs['active_numeric']; cats=defs['active_categories']
roster=read(O.parent/'Stage_AB_eligibility_2026-11-15/restricted/A_membership.csv')
reserve=set(roster.component);assert len(reserve)==227
ledger=read(E/'restricted/horizon_ledger.csv')
ledger=ledger[ledger.eligible.eq('True') & ~ledger.component.isin(reserve)].copy()
assert ledger.reserved.eq('False').all()
baseline=read(E/'restricted/baseline_features.csv');base=baseline[baseline.PATNO.isin(ledger.PATNO)].copy()
assert base.PATNO.is_unique and base.component_id.is_unique and len(base)==839
base['RNA_index_month']=base['index_month'].astype(int)
base['baseline_hvlt_form']=base.baseline_hvlt_form.map(lambda x:str(int(float(x))))
base['baseline_sdmt_form']=base.baseline_sdmt_form.map(lambda x:str(int(float(x))))
prior_inputs=load(C/'additional_ppmi_eligibility_prior_use_2026-10-28/config/Inputs.json')
raw=Path(prior_inputs['hvlt']).parent
# Reuse only the audited predictor-construction block. Its boundaries exclude the original output code.
frozen_code=record(E/'audit.py').read_text()
feature_block=frozen_code.split("prior=load(A/'evidence/Clinical_Feature_Definitions.json')",1)[1].split("mx=matrix.set_index",1)[0]
prior=load(A/'evidence/Clinical_Feature_Definitions.json')
input_hashes=sources
feature_block=feature_block.replace('sources={};provenance','feature_sources={};provenance').replace("sources[d['new_name']]","feature_sources[d['new_name']]")
exec(compile(feature_block,'<audited Stage C clinical predictors>','exec'))
sources=input_hashes
mx=matrix.set_index('PATNO',verify_integrity=True)
availability=read(E/'restricted/optional_feature_availability.csv').set_index('PATNO').loc[mx.index]
allnum=groups['core_plus18'];allcat=cats['core_plus18']
for col in allnum:
 v=pd.to_numeric(mx[col],errors='coerce');assert v.notna().equals(availability[col].eq('True'))
 mx[col]=v
for k,levels in allcat.items():
 mx[k]=mx[k].replace('','unknown')
 assert set(mx[k])<=set(levels)
assert mx.jlo_form.ne('unknown').equals(availability.jlo_observed.eq('True'))
assert mx.motor_state.ne('unknown').equals(availability.motor_state_observed.eq('True'))
complete=mx[allnum].notna().all(axis=1)&mx.jlo_form.ne('unknown')&mx.motor_state.ne('unknown')
x=mx[['component_id']+allnum+list(allcat)].rename(columns={'component_id':'component'})
table(R/'predictors.csv',x.sort_values('component'))
mo=read(C/'moca_refresh_audit_2026-10-28/raw/Montreal_Cognitive_Assessment__MoCA__28Oct2026.csv')
items=load(Path(prior_inputs['prior_cognitive_contract']))['cognition']['item_max']
entries=[]
for r in ledger.itertuples():
 pos=int(float(r.endpoint_row));v=mo.iloc[pos-2];h=int(r.horizon);ix=int(r.index_month);end=int(float(r.endpoint_month))
 assert v.PATNO==r.PATNO and sum([int(v.INFODT[3:])*12,int(v.INFODT[:2])])==end and abs(end-ix-h)<=3
 raw_total=sum(int(float(v[k])) for k in items);edu=mx.at[r.PATNO,'education']
 assert raw_total==30 or np.isfinite(edu)
 y=min(30,raw_total+int(edu<=12))
 assert not v.MCATOT or float(v.MCATOT) in (raw_total,y)
 entries.append(dict(component=r.component,horizon=h,index_month=ix,endpoint_month=end,endpoint_row=pos,complete28=int(complete[r.PATNO]),historical=int(max(ix,end)<2023*12+1),y=y))
rr=pd.DataFrame(entries).sort_values(['horizon','component'])
assert not rr.duplicated(['component','horizon']).any()
table(R/'membership.csv',rr.drop(columns='y'));table(R/'endpoints.csv',rr[['component','horizon','endpoint_row','y']])
table(R/'stage_A_reserved.csv',pd.DataFrame({'component':sorted(reserve)}))
# Across-horizon master: balance each baseline calendar year while balancing total allocation.
pop=dict(zip(base.component_id,base.index_year));ids=sorted(pop)
def foldmap(ids,n,salt):
 count=[0]*n; result={}
 def key(v):return hashlib.sha256(('StageC_v1|'+salt+'|'+v).encode()).hexdigest()
 for yr in sorted(set(pop[i] for i in ids)):
  local=[0]*n
  for i in sorted([i for i in ids if pop[i]==yr],key=key):
   j=min(range(n),key=lambda a:(local[a],count[a],key(i+'|'+str(a))))
   result[i]=j;count[j]+=1;local[j]+=1
 return result
outer=foldmap(ids,5,'outer');inner=[foldmap([i for i in ids if outer[i]!=j],3,'inner'+str(j)) for j in range(5)]
table(R/'grouped_splits.csv',pd.DataFrame([dict(component=i,outer=outer[i],**{'inner_for_outer_'+str(j):inner[j].get(i,-1) for j in range(5)}) for i in ids]))
def panel(nums,ct):return dict(numeric=nums,categories=ct)
spec={'compact4':panel(groups['compact4'],cats['compact4']),'moca_only':panel(['baseline_moca'],{}),'core':parent['comparators']['core'],'enhanced':panel(parent['comparators']['core']['numeric']+groups['compact4'][1:],{**parent['comparators']['core']['categories'],**cats['compact4']}),'full28':panel(allnum,allcat)}
arms=['compact4','persistence','moca_only','core','enhanced']
plan=dict(study='Stage_C_expanded_PPMI_v1',horizons=[12,24,36],primary_horizon=12,primary_contrast='expanded, compact4 minus persistence, paired raw MAE',panels=spec,views={'expanded':dict(n=[777,716,651],models=arms),'complete28':dict(n=[560,527,494],models=arms+['full28']),'historical':dict(n=[475,385,343],models=arms)},penalties=parent['model']['clinical_lambdas'],ridge_objective='mean squared error plus lambda times sum of squared coefficients; unpenalized intercept',inner_selection='minimum pooled inner validation MAE; within 1e-12 prefer largest penalty',splits='5 outer, 3 inner; single component master allocation stratified on clinical baseline year, reused across horizons and views',bootstrap=5000,seed_base=20261115,seed_rule='base + 100 * view ordinal (1,2,3) + horizon',study_fits_planned=3120,stage_A_reserved=227,validation_scope='grouped internal validation; no external or fresh temporal evaluation')
for j,(view,condition) in enumerate([('expanded',rr.horizon>0),('complete28',rr.complete28.eq(1)),('historical',rr.historical.eq(1))]):
 assert list(rr[condition].groupby('horizon').size())==plan['views'][view]['n']
dump(O/'protocol.json',plan)
assert all(digest(p)==v for p,v in sources.items())
dump(O/'source_hashes.json',sources)
