"""Hash/schema/identity checks only; no model fitting."""
from support import *
import csv
def tabs(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def validate_plan():
    lock=json.loads((ROOT/'PLAN_LOCK.json').read_text())
    if lock['participant_training_authorized'] is not False:raise IntegrityError('Plan is not execution authorization')
    for rel,h in lock['sha256'].items():
        if file_sha(ROOT/rel)!=h:raise IntegrityError('Plan changed')
    b=json.loads((ROOT/'config/Input_Binding.json').read_text())
    for path,h in b['files'].items():
        if file_sha(path)!=h['sha256'] or Path(path).stat().st_size!=h['bytes']:raise IntegrityError('Input changed')
    parent=Path(b['parent_plan']);pl=json.loads(parent.read_text())
    for rel,h in pl['sha256'].items():
        if file_sha(parent.parent/rel)!=h:raise IntegrityError('Parent input plan changed')
    for infos in b['references'].values():
        for r in infos.values():
            if file_sha(r['path'])!=r['sha256'] or file_sha(r['manifest'])!=r['manifest_sha256']:raise IntegrityError('Frozen reference changed')
    return b
def validate_data():
    from legacy import references
    before=dict(frozen.COUNTERS);b=validate_plan()
    c=json.loads((ROOT/'config/Analysis_Contract.json').read_text())
    raw=tabs(b['candidate_matrix']);by={r['PATNO']:r for r in raw}
    if len(by)!=351 or len(raw)!=351 or len({r['component_id'] for r in raw})!=351:raise IntegrityError('Frame identity/count mismatch')
    if len(c['features']['optional'])!=28 or [r['bit'] for r in c['features']['optional']]!=list(range(28)):raise IntegrityError('Feature genome mismatch')
    names,cats=frozen.layout(c)
    for r in raw:
        for n in names:frozen.numeric(r[n])
        if any(r[k] not in v for k,v in cats.items()):raise IntegrityError('Invalid source category')
    master={r['component_id']:r for r in tabs(b['master_splits'])};views=tabs(b['view_binding'])
    cohorts={int(h):tabs(p) for h,p in b['cohorts'].items()};persistent={}
    for h,rows in cohorts.items():
        if len(rows)!={12:333,24:321,36:307}[h] or len({r['PATNO'] for r in rows})!=len(rows) or len({r['component_id'] for r in rows})!=len(rows):raise IntegrityError('Cohort mismatch')
        view={r['PATNO']:r for r in views if r['view_id']==f'main_{h}_enhanced'}
        if set(view)!={r['PATNO'] for r in rows}:raise IntegrityError('Historical membership changed')
        for r in rows:
            original={**view[r['PATNO']],**master[r['component_id']]}
            if any(str(original[k])!=str(v) for k,v in r.items()):raise IntegrityError('Historical row/outcome/split changed')
            if r['component_id']!=by[r['PATNO']]['component_id']:raise IntegrityError('Identity join mismatch')
            signature=tuple(r[k] for k in ['component_id','outer_fold']+[f'inner_for_outer_{i}' for i in range(5)])
            if r['PATNO'] in persistent and persistent[r['PATNO']]!=signature:raise IntegrityError('Cross-horizon grouped split changed')
            persistent[r['PATNO']]=signature
            if not np.isfinite(float(r['y'])) or not 0<=float(r['y'])<=30:raise IntegrityError('Invalid outcome')
        references(h,rows)
    from launch_hybrid import make_specs
    specs=make_specs(c,by,cohorts,False)
    if len(specs)!=30 or frozen.COUNTERS!=before:raise IntegrityError('Read-only context creation failed')
    return c,by,cohorts,{'status':'PASS','frame':351,'cohorts':{h:len(r) for h,r in cohorts.items()},'contexts':30,'participant_fits':0,'new_predictions':0,'grouped_bindings':961,'search_started':False}
