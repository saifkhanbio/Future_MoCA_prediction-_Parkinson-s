"""Build local evaluation inputs from the verified eligibility audit. No inference."""
import csv, hashlib, json, math, sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
C=ROOT.parents[2]; N=C.parent
AUD=ROOT.parent/'Stage_AB_eligibility_2026-11-15'
R=ROOT/'restricted'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=1)+'\n',encoding='utf8')
def rows(path,sep=','):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=sep))
def confirm(ok): 
 if not ok:raise ValueError('Input binding failed')
if (ROOT/'LOCK.json').exists():raise SystemExit('Protocol already sealed')
pinned={str(Path(k)).replace('\\','/'):v for k,v in load(AUD/'Source_SHA256.json').items()}
# On Linux, interpret Windows-authored manifest keys as relative portable paths.
for k,v in load(AUD/'Source_SHA256.json').items():confirm(digest(N/k.replace('\\','/'))==v)
auditcode=(AUD/'audit.py').read_text(encoding='utf8')
sys.argv=[str(ROOT/'bind_inputs.py'),str(C)]
# Use the exact audited QC construction, ending before its first output instruction.
ns={'__file__':str(ROOT/'bind_inputs.py')}
exec(compile(auditcode.split('for stats in flow:')[0],'<audited baseline reconstruction>','exec'),ns)
mx=ns['mx']; groups=ns['groups'];D=ns['defs'];auditrows=rows(AUD/'restricted/A_membership.csv')
Ac=C/'independent_validation_contract_2026-10-31'
states=load(Ac/'restricted/Frozen_Inference_States.json')
byid={r['member_id']:r['state'] for r in states}
bind=load(Ac/'config/Frozen_Model_Binding.json')['members']
confirm(len(states)==len(byid)==len(bind)==45)
for m in bind:
 state=byid[m['member_id']]
 confirm(hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==m['state_sha256'])
 confirm(state['columns']==m['columns'] and m['weight']==0.2)
Broot=C/'LCC_nominal_source_verification_2026-11-05'; bundle=Broot/'restricted/Source_Model_Package/Model_Bundle.json'
confirm(digest(bundle)==load(bundle.parent/'RESULT_SEAL.json')['files']['Model_Bundle.json']['sha256'])
confirm(load(Broot/'qa/Independent_Model_Verification.json')['all_pass'])
model=load(bundle)
save(R/'models_A.json',byid)
save(R/'models_B.json',{str(h):model['models'][f'{h}:reduced3'] for h in (12,24,36)})
paths=[Ac/'restricted/Frozen_Inference_States.json',Ac/'config/Frozen_Model_Binding.json',Ac/'config/Validation_Contract.json',Ac/'config/Feature_Definitions.json',Ac/'config/Frozen_Benchmark_Binding.json',Ac/'CONTRACT_LOCK.json',bundle,bundle.parent/'RESULT_SEAL.json',Broot/'qa/Independent_Model_Verification.json',AUD/'restricted/A_membership.csv',AUD/'restricted/B_membership.csv',AUD/'Source_SHA256.json',AUD/'Verification.json',AUD/'audit.py']
A=[];Y=[];B=[];Z=[]; nums=groups['core_plus18']; cats=D['active_categories']['core_plus18']
for h in (12,24,36):
 ep=C/f'temporal_validation_readiness_2026-10-28/restricted/Sealed_Outcomes_{h}m.tsv';paths.append(ep);ends={x['PATNO']:x for x in rows(ep,'\t')}
 for r in auditrows:
  if int(r['horizon'])!=h:continue
  i=r['PATNO']; comp=r['component']; f=mx.loc[i];e=ends[i]
  confirm(comp==f.component_id==e['component_id'] and int(e['horizon_months'])==h and h-3<=int(e['elapsed_months'])<=h+3 and e['endpoint_QC_pass']=='True')
  features={}
  for k in nums:
   v=ns['num'](f[k]);confirm((v is not None)==(r[k]=='True'));features[k]=v
  for k,levels in cats.items():
   value=str(f[k]);features[k]=value if value in levels else 'unknown'
  complete={k:all(r[n]=='True' for n in groups[k]) and r['JLO_form_observed']=='True' and (k!='core_plus18' or r['motor_state_observed']=='True') for k in ('MoCA_plus18','core_plus18')}
  views=['compact']
  for flag,view in [(complete['MoCA_plus18'],'complete19'),(complete['core_plus18'],'complete_all'),(r['19_observed_or_stratum_supported']=='True','supported19'),(r['28_observed_or_stratum_supported']=='True','supported_all')]:
   if flag:views.append(view)
  A.append(dict(id=comp,h=h,x=features,views=views));Y.append(dict(id=comp,h=h,y=int(float(e['education_adjusted_MoCA']))))
amp=N/'strategy_redesign_v2_2026-10-20/source_audit_2026-10-20/raw/amp_pdrd'
mfile=amp/'2023_v4release_1027__MOCA.json';df=amp/'2023_v4release_1027__Demographics.json'
from collections import defaultdict
vis=defaultdict(list);demo=defaultdict(list)
for r in load(mfile):
 if r['visit_name'] in ('M0','M12','M24','M36'):vis[r['participant_id'],r['visit_name']].append(r)
for r in load(df):
 if r['visit_name']=='M0':demo[r['participant_id']].append(r)
for r in rows(AUD/'restricted/B_membership.csv'):
 if r['identity_clear']!='True':continue
 i=r['participant_id']; h=int(r['horizon']);base,=vis[i,'M0'];end,=vis[i,f'M{h}'];d,=demo[i]
 confirm(float(base['visit_month'])==0 and float(end['visit_month'])==h and float(d['visit_month'])==0)
 confirm(d['sex'] in ('Male','Female') and float(d['age_at_baseline']).is_integer() and 18<=float(d['age_at_baseline'])<=89)
 for z in [base,end]:confirm(z['education_12years_complete']=='Greater than 12 years of education' and float(z['code_education_12years_complete'])==0)
 views=['nominal']+(['strict_education'] if r['strict_demographic_band']=='True' else [])
 B.append(dict(id=r['component'],h=h,views=views,x=dict(baseline_moca=int(float(base['moca_total_score'])),age_years=int(float(d['age_at_baseline'])),sex_male=int(d['sex']=='Male'))))
 Z.append(dict(id=r['component'],h=h,y=int(float(end['moca_total_score']))))
for stage,xx,yy in [('A',A,Y),('B',B,Z)]:
 key=lambda r:(r['h'],r['id'])
 confirm(len(set(map(key,xx)))==len(xx)==len(yy) and set(map(key,xx))==set(map(key,yy)))
 save(R/f'features_{stage}.json',sorted(xx,key=key));save(R/f'outcomes_{stage}.json',sorted(yy,key=key))
for file in paths:pinned[file.relative_to(N).as_posix()]=digest(file)
save(ROOT/'source_hashes.json',pinned)
save(ROOT/'model_spec.json',{'active':D,'offset_A':load(Ac/'config/Validation_Contract.json')['prediction']['source_offsets_by_horizon'],'offset_B':model['typical_change']})
print('Bindings complete',len(A),len(B),len(pinned))