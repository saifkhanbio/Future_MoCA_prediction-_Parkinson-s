"""Read-only data integrity and exact frozen reference binding; no numerical fitting."""
from pathlib import Path
import json,csv,math
from engine import require,sha,IntegrityError,numeric,specs,COUNTERS
ROOT=Path(__file__).resolve().parents[1]
def readtab(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def load_plan(root=ROOT):
    root=Path(root);lock=json.loads((root/'PLAN_LOCK.json').read_text())
    require(lock['participant_training_authorized'] is False,'Readiness must not authorize training')
    for rel,d in lock['sha256'].items():require(sha(root/rel)==d,'Locked plan hash mismatch')
    c=json.loads((root/'config/Analysis_Contract.json').read_text());b=json.loads((root/'config/Input_Binding.json').read_text())
    require(c['execution']['participant_training_authorized'] is False,'Readiness authorization changed')
    for name,entry in b['files'].items():
        p=Path(name);s=p.stat()
        require(s.st_size==entry['bytes'] and sha(p)==entry['sha256'],'Source binding changed')
    parent_path=Path(b['parent_plan']);parent=json.loads(parent_path.read_text())
    for rel,d in parent['sha256'].items():require(sha(parent_path.parent/rel)==d,'Parent readiness artifact changed')
    r=Path(b['reference_run']);seal=json.loads((r/'RESULTS_LOCK.json').read_text())
    require(seal['status']=='FROZEN_VERIFIED' and seal['artifact_manifest_sha256']==sha(r/'Artifact_Manifest.json'),'Reference seal changed')
    manifest=json.loads((r/'Artifact_Manifest.json').read_text())['files']
    for name,entry in b['files'].items():
        p=Path(name)
        if p.is_relative_to(r) and str(p.relative_to(r)) in manifest:
            require(manifest[str(p.relative_to(r))]['sha256']==entry['sha256'],'Reference not bound to original result seal')
    return c,b
def validate_dataset(c,by,cohorts,refs,expected_sizes,synthetic=False):
    require(len(by)==len({r['component_id'] for r in by.values()}),'Duplicate matrix components')
    hset={12,24,36};require(set(cohorts)==set(refs)==hset,'Horizon coverage changed')
    persistent={}
    required=set().union(*(set(a['numeric'])|set(a['categories']) for a in c['arms'].values()))
    for pid,r in by.items():
        require(pid==r['PATNO'],'Matrix key mismatch')
        for name in required:
            if name in c['arms']['optional18_only']['categories']:
                require(r[name] in c['arms']['optional18_only']['categories'][name],'Invalid locked category')
            else:numeric(r[name])
    for h,rows in cohorts.items():
        require(len(rows)==expected_sizes[h] and len({r['PATNO'] for r in rows})==len(rows) and len({r['component_id'] for r in rows})==len(rows),'Cohort count/identity mismatch')
        reference={r['PATNO']:r for r in refs[h]};require(len(reference)==len(refs[h])==len(rows),'Duplicate or missing reference rows')
        require(set(reference)=={r['PATNO'] for r in rows},'Reference participant mismatch')
        for row in rows:
            pid=row['PATNO'];require(pid in by and row['component_id']==by[pid]['component_id'],'Matrix join mismatch')
            require(float(row['baseline_moca'])==float(by[pid]['baseline_moca']),'Baseline MoCA binding failed')
            require(float(row['horizon_months'])==h and h-3<=float(row['elapsed_months'])<=h+3,'Outcome timing binding failed')
            require(math.isfinite(float(row['y'])) and 0<=float(row['y'])<=30,'Outcome range invalid')
            signature=tuple(row[k] for k in ['component_id','outer_fold']+[f'inner_for_outer_{j}' for j in range(5)])
            require(pid not in persistent or persistent[pid]==signature,'Cross-horizon grouping changed')
            persistent[pid]=signature
            rr=reference[pid]
            require(rr['component_id']==row['component_id'] and int(rr['outer_fold'])==int(row['outer_fold']) and float(rr['y'])==float(row['y']),'Frozen reference binding mismatch')
            for arm in ['full']+c['context_only_reference_arms']:require(math.isfinite(float(rr[arm])),'Nonfinite frozen prediction')
            require(float(rr['persistence'])==float(row['baseline_moca']),'Frozen persistence binding mismatch')
        for outer in range(5):
            train=[r for r in rows if int(r['outer_fold'])!=outer];test=[r for r in rows if int(r['outer_fold'])==outer]
            require(train and test,'Empty outer fold')
            require(set(int(r[f'inner_for_outer_{outer}']) for r in train)=={0,1,2},'Inner folds missing')
            require(all(int(r[f'inner_for_outer_{outer}'])==-1 for r in test),'Outer-test inner label leak')
            for inner in range(3):
                a={r['component_id'] for r in train if int(r[f'inner_for_outer_{outer}'])!=inner}
                v={r['component_id'] for r in train if int(r[f'inner_for_outer_{outer}'])==inner}
                require(a and v and not a&v,'Inner identity overlap')
    tasks=specs(c,by,cohorts,synthetic)
    return {'pass':True,'cohorts':{str(h):len(x) for h,x in cohorts.items()},'contexts':15,'new_arm_tasks':len(tasks),'participant_fits':0}
def validate_real(root=ROOT):
    before=dict(COUNTERS);c,b=load_plan(root);matrix=readtab(b['candidate_matrix']);by={r['PATNO']:r for r in matrix}
    require(len(by)==len(matrix)==351,'Candidate frame changed')
    cohorts={int(h):readtab(p) for h,p in b['cohorts'].items()};refs={int(h):json.loads(Path(p).read_text()) for h,p in b['OOF'].items()}
    report=validate_dataset(c,by,cohorts,refs,{12:333,24:321,36:307})
    master={r['component_id']:r for r in readtab(b['master_splits'])};views=readtab(b['view_binding'])
    for h,rows in cohorts.items():
        view={r['PATNO']:r for r in views if r['view_id']==f'main_{h}_enhanced'}
        require(set(view)=={r['PATNO'] for r in rows},'Historical outcome view membership changed')
        for row in rows:
            original={**view[row['PATNO']],**master[row['component_id']]}
            require(all(str(original[k])==str(v) for k,v in row.items()),'Historical row/outcome/group binding changed')
        for outer in range(5):
            rr=json.loads((Path(b['reference_run'])/f'restricted/{h}m_outer{outer}/Result.json').read_text())
            ix={pid:i for i,pid in enumerate(rr['test_ids'])}
            expected=[r for r in refs[h] if int(r['outer_fold'])==outer]
            require(set(ix)=={r['PATNO'] for r in expected},'Reference context coverage mismatch')
            for r in expected:
                j=ix[r['PATNO']]
                require(rr['test_components'][j]==r['component_id'] and float(rr['predictions']['full'][j])==float(r['full']),'Frozen full prediction mismatch')
    require(COUNTERS==before,'Read-only validation unexpectedly fitted data')
    report.update(source_bindings=len(b['files']),historical_row_bindings=961,frozen_full_predictions_bound=961,new_predictions=0,paid_queries=0)
    return c,by,cohorts,refs,report
