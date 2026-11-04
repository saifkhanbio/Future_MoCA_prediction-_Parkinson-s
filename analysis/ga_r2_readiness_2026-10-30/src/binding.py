"""Read-only participant binding; invokes no fit, selection or model scoring."""
from support import *
import importlib.util
def validate_plan():
    lock=json.loads((ROOT/'PLAN_LOCK.json').read_text())
    if lock.get('participant_training_authorized') is not False:raise IntegrityError('Readiness plan must not authorize training')
    for rel,h in lock['sha256'].items():
        if file_sha(ROOT/rel)!=h:raise IntegrityError('R2 plan hash mismatch')
    binding=json.loads((ROOT/'config/Input_Binding.json').read_text())
    if file_sha(FROZEN/'PLAN_LOCK.json')!=binding['parent_plan_lock_sha256']:raise IntegrityError('Parent plan changed')
    for path,h in binding['files'].items():
        if file_sha(path)!=h:raise IntegrityError('Input binding hash mismatch')
    return binding
def validate_data():
    binding=validate_plan()
    spec=importlib.util.spec_from_file_location('original_read_only_binding',FROZEN/'src/launch.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    before=dict(old.COUNTERS)
    parent,by,cohorts,report=old.validate()
    if old.COUNTERS!=before:raise IntegrityError('Binding validation unexpectedly fitted a model')
    c=json.loads((ROOT/'config/Analysis_Contract.json').read_text())
    for name in binding['unchanged_parent_fields']:
        if c[name]!=parent[name]:raise IntegrityError('Inherited cohort/feature rule changed')
    if c['model']['lambdas']!=parent['model']['lambdas']:raise IntegrityError('Penalty grid changed')
    for name in ('core_mask','enhanced_mask','full_mask'):
        if c['controls'][name]!=parent['controls'][name]:raise IntegrityError('Comparator mask changed')
    for h,cohort in cohorts.items():
        for outer in range(5):
            y=np.array([float(r['y']) for r in cohort if int(r['outer_fold'])!=outer])
            if not np.isfinite(y).all() or np.var(y)<=0:raise IntegrityError('Undefined pooled training R2')
    legacy=json.loads((ROOT/'config/Legacy_Comparison.json').read_text())
    for obj in list(legacy['OOF'].values())+[legacy['selections']]:
        if file_sha(obj['path'])!=obj['sha256']:raise IntegrityError('Frozen MAE reference changed')
    report.update(study_id=c['study_id'],pooled_variance_valid_in_contexts=15,plan_lock_sha256=file_sha(ROOT/'PLAN_LOCK.json'),legacy_reference_hashes_verified=True,participant_data_fits=0,new_outcome_scoring_performed=False)
    return c,by,cohorts,report
