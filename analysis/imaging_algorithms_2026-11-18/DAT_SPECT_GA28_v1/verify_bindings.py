"""Independent source-row, cohort, endpoint and fold audit; no model fitting."""
from pathlib import Path
import csv,json,hashlib,math,collections
O=Path(__file__).resolve().parent;C=O.parent;R=O/'restricted'
F=C.parent/'strategy_redesign_v2_2026-10-20/download_resume_2026-10-20/restricted/ppmi/imaging/files'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def mo(v):
 try:
  mm,yy=v.split('/')
  return 12*int(yy)+int(mm) if len(mm)==2 and len(yy)==4 and 1<=int(mm)<=12 and 1900<=int(yy)<=2100 else None
 except:return None
def number(v):
 try:
  x=float(v);return x if math.isfinite(x) else None
 except:return None
def group(rows):
 d=collections.defaultdict(list)
 for i,r in enumerate(rows,2):d[r['PATNO']].append(dict(r,line=i))
 return d
A=group(read(F/'Xing_Core_Lab_-_DATSCAN_acquisition_metadata_20Oct2026.csv'))
Q=group(read(F/'Xing_Core_Lab_-_Quant_SBR_20Oct2026.csv'))
ledger=read(R/'Scan_Audit.csv')
plan=json.loads((O/'Protocol.json').read_text());fields=plan['DAT_fields'];independent=set();selected={}
for row in ledger:
 p=row['PATNO'];ix=int(row['index_month']);a0=[r for r in A[p] if r['EVENT_ID'] in ('SC','BL')];q0=[r for r in Q[p] if r['EVENT_ID'] in ('SC','BL')]
 bad=any(mo(r['DATSCAN_DATE']) is None for r in a0+q0)
 dates=[mo(r['DATSCAN_DATE']) for r in a0+q0 if mo(r['DATSCAN_DATE']) is not None and ix-3<=mo(r['DATSCAN_DATE'])<=ix]
 if not dates:
  assert row['eligible']=='False';continue
 chosen=max(dates);ev='BL' if any(mo(r['DATSCAN_DATE'])==chosen and r['EVENT_ID']=='BL' for r in a0+q0) else 'SC'
 a=[r for r in a0 if mo(r['DATSCAN_DATE'])==chosen and r['EVENT_ID']==ev]
 q=[r for r in q0 if mo(r['DATSCAN_DATE'])==chosen and r['EVENT_ID']==ev]
 prots={r['PROTOCOL'] for r in a+q}
 bad |= len(prots)!=1 or '' in prots or not a or not q
 if a:
  cols=[k for k in a[0] if k not in ('PATNO','line')]
  bad |= len({tuple(r[k] for k in cols) for r in a})!=1
  for r in a:
   bad |= r['DATSCAN_IMAGE_ACCEPTABLE']!='Yes' or r['DATSCAN_LIGAND'] not in ('','123I-DaTscan') or r['PREVIOUSLY_ACQUIRED'] not in ('','No')
   bad |= bool(r['DATSCAN_DATE_INJECTED'] and mo(r['DATSCAN_DATE_INJECTED'])!=chosen)
   if r['DATSCAN_TOTAL_DOSE_CALC']:
    n=number(r['DATSCAN_TOTAL_DOSE_CALC'])
    bad |= n is None or n<=0 or r['DATSCAN_TOTAL_DOSE_UNIT']!='MBq'
 if q:
  cols=['PROTOCOL','DATSCAN_LIGAND','PREVIOUSLY_ACQUIRED','DATSCAN_ANALYZED','DATSCAN_NOT_ANALYZED_REASON','DATSCAN_OTHER_SPECIFY']+fields
  bad |= len({tuple(r[k] for k in cols) for r in q})!=1
  for r in q:
   bad |= r['DATSCAN_ANALYZED']!='Yes' or r['DATSCAN_LIGAND']!='123I-DaTscan' or r['PREVIOUSLY_ACQUIRED']!='No'
   bad |= bool(r['DATSCAN_NOT_ANALYZED_REASON'] or r['DATSCAN_OTHER_SPECIFY'])
   bad |= any(number(r[k]) is None or number(r[k]) < -1 for k in fields)
 assert (not bad)==(row['eligible']=='True'),'Independent QC membership differs'
 assert chosen==int(float(row['scan_month'])) and ev==row['event']
 assert row['strict_preindex']==str(not bad and chosen<ix)
 assert row['acquisition_rows']==';'.join(str(r['line']) for r in a)
 assert row['quantitative_rows']==';'.join(str(r['line']) for r in q)
 if not bad:
  independent.add(row['component']);selected[row['component']]=q[0]

def tsv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
b=json.loads((C/'ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/Input_Binding.json').read_text())
old={r['component_id']:r for r in tsv(b['candidate_matrix'])}
assert all(int(row['index_month'])==int(old[row['component']]['RNA_index_month']) and row['PATNO']==old[row['component']]['PATNO'] for row in ledger)
original={}
for h,path in b['cohorts'].items():
 for r in tsv(path):original[(r['component_id'],h)]=r
expected={k for k in original if k[0] in independent};expected_ids={k[0] for k in expected}
predictors={r['component']:r for r in read(R/'Predictors.csv')};assert set(predictors)==expected_ids
for p,r in predictors.items():
 for k,v in old[p].items():assert r['component' if k=='component_id' else k]==v,k
 assert all(number(r[k])==number(selected[p][k]) for k in fields)
splits={r['component']:r for r in read(R/'Grouped_Splits.csv')}
master={r['component_id']:r for r in tsv(b['master_splits'])}
for p,r in splits.items():
 assert r['outer']==master[p]['outer_fold']
 assert all(r[f'inner_for_outer_{j}']==master[p][f'inner_for_outer_{j}'] for j in range(5))
reserve={r['component'] for r in read(R/'Stage_A_Reserved.csv')};assert not expected_ids&reserve
endpoints={(r['component'],r['horizon']):r for r in read(R/'Endpoints.csv')};membership=read(R/'Membership.csv')
assert set(endpoints)==expected and len(membership)==len(expected)
scanby={r['component']:r for r in ledger}
for r in membership:
 k=r['component'],r['horizon'];z=original[k];a=scanby[k[0]]
 assert number(endpoints[k]['y'])==number(z['y']) and r['endpoint_row']==z['outcome_source_row']==endpoints[k]['endpoint_row']
 assert int(r['endpoint_month'])-int(r['index_month'])==int(z['elapsed_months'])
 assert r['strict_preindex']==a['strict_preindex'] and number(r['scan_month'])==number(a['scan_month'])
 assert abs(int(r['endpoint_month'])-int(r['index_month'])-int(r['horizon']))<=3
 assert splits[k[0]]['outer']==z['outer_fold']
scopes=json.loads((R/'Selection_Training_Scope.json').read_text())
for scope in scopes:
 h=str(scope['horizon']);j=scope['outer'];train={k[0] for k,z in original.items() if k[1]==h and int(z['outer_fold'])!=j};test={k[0] for k,z in original.items() if k[1]==h and int(z['outer_fold'])==j}
 assert set(scope['original_training_components'])==train and set(scope['original_test_components'])==test and not train&test
 matched_test={r['component'] for r in membership if r['horizon']==h and int(splits[r['component']]['outer'])==j};assert matched_test<=test and not matched_test&train
sels=json.loads((C/'ga_unrestricted_run_v4_10kgen_2026-11-01/results/Selections.json').read_text())
for z in sels:
 if z['arm']=='unrestricted28_GA_selected':
  target=plan['frozen_fold_panels'][str(z['horizon'])+'/'+str(z['outer'])];assert target['mask']==z['mask'] and target['features']==z['features'] and target['historical_alpha']==z['alpha']
counts=collections.Counter(r['horizon'] for r in membership);strict=collections.Counter(r['horizon'] for r in membership if r['strict_preindex']=='True')
assert [counts[str(h)] for h in plan['horizons']]==plan['views']['month_compatible']['n']
assert [strict[str(h)] for h in plan['horizons']]==plan['views']['strict_preindex']['n']
sources=json.loads((O/'Source_SHA256.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in sources.items())
result=dict(status='PASS',participants_audited=len(ledger),unique_imaging_eligible=len(independent),matched_unique=len(expected_ids),matched_counts=dict(counts),strict_counts=dict(strict),original_RNA_landmarks=True,source_QC_and_row_binding=True,clinical_values_unchanged=True,original_GA_splits_preserved=True,fold_specific_selection_scope_verified=True,endpoint_bindings_verified=True,frozen_sources_unchanged=True,study_fits=0,paid_queries=0)
(O/'qa/Independent_Binding_Verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
