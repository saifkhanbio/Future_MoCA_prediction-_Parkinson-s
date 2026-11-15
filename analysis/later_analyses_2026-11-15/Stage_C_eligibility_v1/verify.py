"""Independent set reconciliation and native-form checks; no model imports."""
from pathlib import Path
import csv, json, hashlib, collections
O=Path(__file__).resolve().parent
C=O.parents[2]
def read(p):
 p=Path(p)
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t' if p.suffix=='.tsv' else ','))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def month(v):
 m,y=map(int,v.split('/'));return 12*y+m
def yes(r,k):return r[k]=='True'
inp=json.loads((O/'input_SHA256.json').read_text());assert all(sha(p)==v for p,v in inp.items())
I=json.loads((C/'additional_ppmi_eligibility_prior_use_2026-10-28/config/Inputs.json').read_text())
risk={r['PATNO']:r for r in read(O/'restricted/baseline_features.csv')}
led=read(O/'restricted/horizon_ledger.csv')
old=read(C/'model_readiness_2026-10-24/restricted/View_Row_Binding.tsv')
orig={r['component_id'] for r in read(I['current_manifest'])}
a=read(O.parent/'Stage_AB_eligibility_2026-11-15/restricted/A_membership.csv')
reserved={r['component'] for r in a}
groups=json.loads((C/'independent_validation_contract_2026-10-31/config/Feature_Definitions.json').read_text())['active_numeric']
opt={r['PATNO']:r for r in read(O/'restricted/optional_feature_availability.csv')}
sources={k:read(I[k]) for k in ['sdmt','hvlt']}
sources['moca']=read(C/'moca_refresh_audit_2026-10-28/raw/Montreal_Cognitive_Assessment__MoCA__28Oct2026.csv')
items=json.loads(Path(I['prior_cognitive_contract']).read_text())['cognition']['item_max']
def check_moca(row,edu):
 scores=[int(float(row[k])) for k in items]
 assert all(0<=v<=cap for v,cap in zip(scores,items.values()))
 raw=sum(scores);assert raw==30 or edu!=''
 adjusted=min(30,raw+int(edu!='' and float(edu)<=12))
 assert not row['MCATOT'] or float(row['MCATOT']) in (raw,adjusted)
 assert not row['MCAVFNUM'] or (float(row['MCAVFNUM'])>=11)==bool(float(row['MCAVF']))
 return adjusted
sets={}; histsets={};extras=[]; stats=read(O/'eligibility_counts.csv')
for s in stats:
 h=int(s['horizon']);subset=[r for r in led if int(r['horizon'])==h]
 assert len({r['PATNO'] for r in subset})==len(subset)==len(risk)
 pool=set(); poolid=set(); hist=set(); allc=set()
 for r in subset:
  p=r['PATNO'];b=risk[p];assert yes(r,'reserved')==(r['component'] in reserved)
  q=not b['baseline_holds'] and yes(b,'enhanced_tests_complete') and r['endpoint_row'] and not r['endpoint_reasons']
  assert bool(q)==yes(r,'eligible')
  if not q:continue
  ix=int(r['index_month'])
  for key,ref in [('moca','baseline_moca_row'),('sdmt','sdmt_row'),('hvlt','hvlt_row'),('moca','endpoint_row')]:
   v=sources[key][int(float(r[ref]))-2];t=month(v['INFODT']);assert v['PATNO']==p
   if ref=='endpoint_row':
    assert abs(t-ix-h)<=3 and t==int(float(r['endpoint_month']))
    assert not b['death_month'] or t<=int(float(b['death_month']))
    assert not b['earliest_withdrawal_month'] or t<=int(float(b['earliest_withdrawal_month']))
   else:assert v['EVENT_ID'] in ('SC','BL') and 0<=ix-t<=3
   if key=='moca':
    y=check_moca(v,b['education'])
    if ref=='baseline_moca_row':assert y==float(b['baseline_moca'])
   if key=='sdmt':assert float(v['SDMTOTAL'])==float(b['baseline_sdmt']) and int(v['SDMTVRSN']) in (1,2)
   if key=='hvlt':assert sum(float(v[k]) for k in ('HVLTRT1','HVLTRT2','HVLTRT3'))==float(b['baseline_hvlt_immediate']) and float(v['HVLTRDLY'])==float(b['baseline_hvlt_delayed']) and 1<=int(v['HVLTVRSN'])<=6
  k=r['component'];assert k and k not in allc;allc.add(k)
  if k in reserved:continue
  pool.add(k);poolid.add(p)
  if ix<2023*12+1:hist.add(k)
 prior={r['component_id'] for r in old if r['view_id']==f'main_{h}_enhanced'}
 assert len(allc)==int(s['all_four_score_matched']) and len(allc&reserved)==int(s['reserved_stage_A_overlap'])
 for name,val in [('available_excluding_stage_A',len(pool)),('net_increase',len(pool)-len(prior)),('additional_outside_original_351',len(pool-orig)),('original_351_new_at_horizon',len((pool&orig)-prior)),('old_horizon_no_longer_eligible',len(prior-pool)),('pre2023_total',len(hist)),('pre2023_additional_outside_351',len(hist-orig))]:
  assert val==int(s[name])
 for k,fields in groups.items():
  n=sum(all(yes(opt[p],key) for key in fields) and (k=='compact4' or yes(opt[p],'jlo_observed')) and (k!='core_plus18' or yes(opt[p],'motor_state_observed')) for p in poolid)
  assert n==int(s[k+'_fully_observed'])
 sets[h]=pool;histsets[h]=hist
 extras.append(dict(horizon=h,stage_A_original_horizon=len({r['component'] for r in a if int(r['horizon'])==h}),new_index_overlap_reserved_any_horizon=len(allc&reserved)))
result={'verification':'PASS','unique_stage_A_reserved':len(reserved),'unique_stage_C_any_horizon':len(set.union(*sets.values())),'unique_additional_any_horizon':len(set.union(*sets.values())-orig),'all_three_horizons':len(set.intersection(*sets.values())),'unique_pre2023_any_horizon':len(set.union(*histsets.values())),'unique_pre2023_new':len(set.union(*histsets.values())-orig),'stage_A_reconciliation':extras,'verified_input_files':len(inp),'fit_count':0,'prediction_count':0}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
