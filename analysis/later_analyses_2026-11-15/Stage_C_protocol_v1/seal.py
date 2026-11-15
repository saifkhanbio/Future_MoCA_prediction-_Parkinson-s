"""Reconcile grouped assignments to the source rosters, test the launcher, then seal."""
import csv,json,os
from pathlib import Path
from collections import Counter
from launcher import D,digest,inputs,validate,require
from validate_synthetic import exercise
os.umask(0o077)
if (D/'LOCK.json').exists():raise SystemExit('Already locked')
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
a=D.parent/'Stage_AB_eligibility_2026-11-15/restricted/A_membership.csv'
e=D.parent/'Stage_C_eligibility_v1/restricted/horizon_ledger.csv'
plan,x,m,y,s,reserve=inputs()
assert reserve=={r['component'] for r in rows(a)} and len(reserve)==227
expected={(r['component'],int(r['horizon'])) for r in rows(e) if r['eligible']=='True' and r['component'] not in reserve}
assert expected==set(m.index) and len(expected)==2144 and len(x)==839
folds=validate(plan,x,m,y,s,reserve)
for r in folds:
    assert r['test']>0 and min(r[k] for k in ['inner0','inner1','inner2'])>0
    assert r['train']==sum(r[k] for k in ['inner0','inner1','inner2'])
src=json.loads((D/'source_hashes.json').read_text())
assert all(digest(k)==v for k,v in src.items())
assert not (D.parent/'Stage_C_results_v1').exists()
synthetic=exercise()
with (D/'fold_counts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(folds[0]));w.writeheader();w.writerows(folds)
dump(D/'synthetic_validation.json',synthetic)
dump(D/'split_verification.json',dict(status='PASS',master_people=len(x),reserved_people=len(reserve),overlap=len(set(s.index)&reserve),eligible_person_horizons=len(expected),outer_counts={str(k):int(v) for k,v in s.outer.value_counts().sort_index().items()},view_horizon_outer_cells=len(folds),smallest_outer_test=min(r['test'] for r in folds),smallest_inner_validation=min(r[k] for r in folds for k in ['inner0','inner1','inner2']),sources_verified=len(src),study_fits=0,study_predictions=0))
covered=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='LOCK.json']
dump(D/'LOCK.json',dict(stage=plan['study'],status='LOCKED_SYNTHETIC_PASS',files={str(p.relative_to(D)):digest(p) for p in sorted(covered)}))
print(json.dumps(synthetic))
