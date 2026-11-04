"""Immutable-source, read-only cohort binding for mandatory-MoCA subset search."""
from support import *
import importlib.util
def validate_plan():
    lock=json.loads((ROOT/'PLAN_LOCK.json').read_text())
    if lock.get('participant_training_authorized') is not False:raise IntegrityError('Plan lock is not an execution authorization')
    for rel,h in lock['sha256'].items():
        if file_sha(ROOT/rel)!=h:raise IntegrityError('Plan changed')
    b=json.loads((ROOT/'config/Input_Binding.json').read_text())
    if file_sha(FROZEN/'PLAN_LOCK.json')!=b['parent_plan_lock_sha256']:raise IntegrityError('Parent plan changed')
    for path,h in b['files'].items():
        if file_sha(path)!=h:raise IntegrityError('Input changed')
    for info in b['reference_files'].values():
        if file_sha(info['path'])!=info['sha256']:raise IntegrityError('Frozen MoCA-plus18 changed')
    return b
def validate_data():
    b=validate_plan();fixed=Path(b['fixed_readiness']);saved_path=list(sys.path)
    try:
        sys.path.insert(0,str(fixed/'src'))
        spec=importlib.util.spec_from_file_location('fixed_feature_readonly_binding',fixed/'src/binding.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        before=dict(frozen.COUNTERS);fc,by,cohorts,refs,report=module.validate_real()
    finally:
        sys.path[:]=saved_path
    if frozen.COUNTERS!=before:raise IntegrityError('Validation fitted a model')
    c=json.loads((ROOT/'config/Analysis_Contract.json').read_text())
    if c['features']['core']!={'numeric':['baseline_moca'],'categories':{}}:raise IntegrityError('Mandatory predictor block changed')
    if c['features']['optional']!=fc['optional_feature_register'] or c['features']['nuisance_categories']!=fc['form_controls']:raise IntegrityError('Optional feature/form definition changed')
    for name in ['population','preprocessing','missingness','splits']:
        if c[name]!=fc[name]:raise IntegrityError('Cohort/preprocessing definition changed')
    if c['model']['lambdas']!=fc['model']['lambdas']:raise IntegrityError('Penalty grid changed')
    for cohort in cohorts.values():
        for outer in range(5):
            y=np.array([float(r['y']) for r in cohort if int(r['outer_fold'])!=outer])
            if not np.isfinite(y).all() or np.var(y)<=0:raise IntegrityError('Undefined training R2')
    report.update(study_id=c['study_id'],mandatory_predictors=['baseline_moca'],optional_features=18,participant_data_fits=0,search_started=False)
    return c,by,cohorts,report
