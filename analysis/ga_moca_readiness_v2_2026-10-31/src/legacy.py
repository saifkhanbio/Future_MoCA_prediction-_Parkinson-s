"""Evaluation-only binding to frozen18+MoCA predictions; no source models changed."""
from support import *
def reference(h):
    info=json.loads((ROOT/'config/Legacy_Comparison.json').read_text())['OOF'][str(h)]
    if file_sha(info['path'])!=info['sha256']:raise IntegrityError('Frozen reference changed')
    return json.loads(Path(info['path']).read_text())
def bind_frozen_full(h,cohort,reproduced):
    old=reference(h);mapping={r['PATNO']:r for r in old}
    if len(mapping)!=len(old) or set(mapping)!={r['PATNO'] for r in cohort}:raise IntegrityError('Frozen reference identities differ')
    values=[]
    for r in cohort:
        s=mapping[r['PATNO']]
        if s['component_id']!=r['component_id'] or int(s['outer_fold'])!=int(r['outer_fold']) or float(s['y'])!=float(r['y']):raise IntegrityError('Frozen reference outcome/fold mismatch')
        values.append(float(s['optional18_plus_moca']))
    values=np.asarray(values)
    if not np.allclose(values,reproduced,atol=1e-9,rtol=0):raise IntegrityError('MoCA-plus18 reproduction failed')
    return values
def compare_frozen(out):
    checks=[];maximum=0.
    for h in [12,24,36]:
        old={r['PATNO']:r for r in reference(h)};new=json.loads((out/f'restricted/OOF_{h}m.json').read_text())
        if set(old)!={r['PATNO'] for r in new}:raise IntegrityError('Reference coverage mismatch')
        if any(r['full']!=old[r['PATNO']]['optional18_plus_moca'] for r in new):raise IntegrityError('Frozen full predictions not reused exactly')
        for outer in range(5):
            r=json.loads((out/f'restricted/{h}m_outer{outer}/Result.json').read_text())
            diff=max(abs(float(p)-float(old[pid]['optional18_plus_moca'])) for pid,p in zip(r['test_ids'],r['models']['full']['predictions']))
            maximum=max(maximum,diff)
            if diff>1e-9:raise IntegrityError('Full refit compatibility failed')
        checks.append({'horizon':h,'n':len(new),'saved_reference_reused_exactly':True})
    write_json(out/'results/Frozen_Full_Parity.json',{'status':'PASS','checks':checks,'maximum_reproduction_difference':maximum,'source_model_modified':False,'compatibility_refits':15})
