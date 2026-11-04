"""Read-only bindings for frozen OOF comparators; no refitting."""
from support import *
def references(h,cohort):
    b=json.loads((ROOT/'config/Input_Binding.json').read_text());out={};allrows={}
    for role,info in b['references'][str(h)].items():
        if file_sha(info['path'])!=info['sha256'] or file_sha(info['manifest'])!=info['manifest_sha256']:raise IntegrityError('Frozen comparator changed')
        source=json.loads(Path(info['path']).read_text());index={r['PATNO']:r for r in source}
        if len(index)!=len(source) or set(index)!={r['PATNO'] for r in cohort}:raise IntegrityError('Reference identity mismatch')
        ordered=[]
        for r in cohort:
            rr=index[r['PATNO']]
            if rr['component_id']!=r['component_id'] or int(rr['outer_fold'])!=int(r['outer_fold']) or float(rr['y'])!=float(r['y']):raise IntegrityError('Reference outcome/fold binding mismatch')
            ordered.append(rr)
        allrows[role]=ordered
        key={'compact4':'enhanced','core_plus18':'full','MoCA_plus18':'optional18_plus_moca'}[role]
        out[role]=np.array([float(r[key]) for r in ordered])
    out['prior_MoCA_GA']=np.array([float(r['GA_selected']) for r in allrows['compact4']])
    for key in ['persistence','typical_change']:out[key]=np.array([float(r[key]) for r in allrows['core_plus18']])
    if any(not np.isfinite(v).all() for v in out.values()):raise IntegrityError('Invalid frozen prediction')
    return out
