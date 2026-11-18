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
predictors={r['component']:r for r in read(R/'Predictors.csv')}
assert set(predictors)==independent
old={r['component']:r for r in read(C/'clinical_extensions/Stage_C_protocol_v1/restricted/predictors.csv')}
for p,r in predictors.items():
 assert all(r[k]==v for k,v in old[p].items())
 assert all(number(r[k])==number(selected[p][k]) for k in fields)
splits={r['component']:r for r in read(R/'Grouped_Splits.csv')}
osplits={r['component']:r for r in read(C/'clinical_extensions/Stage_C_protocol_v1/restricted/grouped_splits.csv')}
assert all(splits[p]==osplits[p] for p in splits)
reserve={r['component'] for r in read(R/'Stage_A_Reserved.csv')};assert not independent&reserve
endpoints={(r['component'],r['horizon']):r for r in read(R/'Endpoints.csv')}
orig=[]
for src in [C/'clinical_extensions/Stage_C_protocol_v1/restricted/endpoints.csv',C/'Fourth_year_extension_v1/protocol/restricted/endpoints.csv']:orig+=read(src)
orig={(r['component'],r['horizon']):r for r in orig}
membership=read(R/'Membership.csv')
assert len(membership)==len(endpoints)
clinical=read(C/'Fourth_year_extension_v1/audit/restricted/horizon_ledger.csv')
expected={(r['component'],r['horizon']) for r in clinical if r['eligible']=='True' and r['reserved']=='False' and r['component'] in independent}
assert set(endpoints)==expected
for r in membership:
 k=(r['component'],r['horizon']);assert endpoints[k]==orig[k] and r['endpoint_row']==orig[k]['endpoint_row']
 assert abs(int(float(r['endpoint_month']))-int(r['index_month'])-int(r['horizon']))<=3
 assert -3<=int(float(r['scan_minus_index']))<=0
counts=collections.Counter(r['horizon'] for r in membership)
strict=collections.Counter(r['horizon'] for r in membership if r['strict_preindex']=='True')
assert [counts[str(h)] for h in plan['horizons']]==plan['views']['month_compatible']['n']
assert [strict[str(h)] for h in plan['horizons']]==plan['views']['strict_preindex']['n']
for view in plan['views']:
 for h in plan['horizons']:
  ids=[r['component'] for r in membership if int(r['horizon'])==h and (view=='month_compatible' or r['strict_preindex']=='True')]
  for j in range(5):
   test=[p for p in ids if int(splits[p]['outer'])==j];train=set(ids)-set(test)
   assert test and train
   assert all(int(splits[p]['inner_for_outer_'+str(j)])==-1 for p in test)
   assert {int(splits[p]['inner_for_outer_'+str(j)]) for p in train}=={0,1,2}
sources=json.loads((O/'Source_SHA256.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in sources.items())
result=dict(status='PASS',participants_audited=len(ledger),unique_eligible=len(independent),matched_counts=dict(counts),
 strict_counts=dict(strict),source_row_bindings_verified=True,exact_clinical_values_and_master_splits=True,
 frozen_endpoints_unchanged=True,all_source_hashes_unchanged=True,study_fits=0,paid_queries=0,
 production_launcher_validated=False)
(O/'Independent_Verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
