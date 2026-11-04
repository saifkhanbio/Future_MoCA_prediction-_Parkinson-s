"""Read-only validation, or separately authorized production execution."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,csv,hashlib,itertools,time,traceback,argparse
import numpy as np
from ga_engine import outer_task,metrics,interval,layout,numeric,COUNTERS
ROOT=Path(__file__).resolve().parents[1]
def readtab(path):
    with open(path,newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def writejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def validate(allow_draft=False):
    c=json.loads((ROOT/'config/Analysis_Contract.json').read_text());matrix=readtab(ROOT/'restricted/Candidate_Matrix.tsv')
    lockfile=ROOT/'PLAN_LOCK.json'
    if lockfile.exists():
        lock=json.loads(lockfile.read_text())
        for rel,digest in lock['sha256'].items():
            if sha(ROOT/rel)!=digest:raise RuntimeError('Locked artifact changed: '+rel)
    elif not allow_draft:raise RuntimeError('PLAN_LOCK missing')
    sources=json.loads((ROOT/'evidence/Source_Hashes.json').read_text())
    for path,meta in sources.items():
        if sha(path)!=meta['sha256']:raise RuntimeError('Source hash changed')
    by={r['PATNO']:r for r in matrix}
    if len(by)!=351 or len(by)!=len(matrix):raise RuntimeError('Frame/ID mismatch')
    if len({r['component_id'] for r in matrix})!=len(matrix):raise RuntimeError('Identity component is not one-to-one')
    numeric_names,cats=layout(c)
    for row in matrix:
        for name in numeric_names:numeric(row[name])
        for name,levels in cats.items():
            if row[name] not in levels:raise RuntimeError('Invalid locked category')
    if c['features']['held'].get('UPSIT') is None or any(r['name']=='UPSIT' for r in c['features']['optional']):raise RuntimeError('UPSIT hold violated')
    if [r['bit'] for r in c['features']['optional']]!=list(range(18)):raise RuntimeError('Candidate bits changed')
    if len(c['search']['seeds'])!=c['search']['restarts'] or c['search']['population']!=40 or c['search']['offspring_batches']!=25 or c['search']['offspring_each']!=40:raise RuntimeError('Schedule changed')
    cohorts={};summary=[]
    for h,n in [(12,333),(24,321),(36,307)]:
        cohort=readtab(ROOT/f'restricted/Cohort_{h}m.tsv');cohorts[h]=cohort
        if len(cohort)!=n or len({r['PATNO'] for r in cohort})!=n or len({r['component_id'] for r in cohort})!=n:raise RuntimeError('Cohort binding failed')
        for r in cohort:
            if r['PATNO'] not in by or r['component_id']!=by[r['PATNO']]['component_id']:raise RuntimeError('Identity join failed')
            if int(float(r['horizon_months']))!=h or not 0<=float(r['y'])<=30 or not h-3<=float(r['elapsed_months'])<=h+3:raise RuntimeError('Outcome binding failed')
            if float(r['baseline_moca'])!=float(by[r['PATNO']]['baseline_moca']):raise RuntimeError('Baseline binding failed')
        for outer in range(5):
            train=[r for r in cohort if int(r['outer_fold'])!=outer];test=[r for r in cohort if int(r['outer_fold'])==outer]
            if not train or not test:raise RuntimeError('Empty outer fold')
            if set(r['component_id'] for r in train)&set(r['component_id'] for r in test):raise RuntimeError('Outer identity leakage')
            if set(int(r[f'inner_for_outer_{outer}']) for r in train)!={0,1,2}:raise RuntimeError('Inner coverage failed')
            if any(int(r[f'inner_for_outer_{outer}'])!=-1 for r in test):raise RuntimeError('Held-out rows have inner assignments')
            for inner in range(3):
                a={r['component_id'] for r in train if int(r[f'inner_for_outer_{outer}'])!=inner}
                b={r['component_id'] for r in train if int(r[f'inner_for_outer_{outer}'])==inner}
                if not a or not b or a&b:raise RuntimeError('Inner identity leakage')
        summary.append({'horizon':h,'participants':n,'outer_folds':5,'inner_folds':3})
    return c,by,cohorts,{'pass':True,'locked':lockfile.exists(),'cohorts':summary,'source_hashes_verified':len(sources),'participant_data_fits':0}
def execute_dataset(c,by,cohorts,out,synthetic=False,execution_record=None):
    out=Path(out)
    if out.exists():raise RuntimeError('Output path already exists')
    out.mkdir(parents=True)
    for name in ['restricted','results']:(out/name).mkdir()
    writejson(out/'STATUS.json',{'status':'RUNNING','synthetic':synthetic})
    if execution_record is not None:writejson(out/'Execution_Context.json',execution_record)
    counter_start=dict(COUNTERS);started=time.perf_counter();total=0;allmetrics=[];allcontrasts=[];selections=[];totals=[]
    try:
        for h,cohort in sorted(cohorts.items()):
            if synthetic and not all(r['PATNO'].startswith('SYNTH_') for r in cohort):raise RuntimeError('Synthetic mode received non-synthetic ID')
            y=np.array([float(r['y']) for r in cohort]);pred={k:np.full(len(cohort),np.nan) for k in c['arms']+['persistence','typical_change']};coverage=np.zeros(len(cohort),int)
            for outer in range(5):
                tr=[i for i,r in enumerate(cohort) if int(r['outer_fold'])!=outer];te=[i for i,r in enumerate(cohort) if int(r['outer_fold'])==outer]
                train=[by[cohort[i]['PATNO']] for i in tr];test=[by[cohort[i]['PATNO']] for i in te]
                # Search/refit never receive y[te]. Only this orchestrator later evaluates predictions.
                ps,details=outer_task(train,y[tr],[int(cohort[i][f'inner_for_outer_{outer}']) for i in tr],test,c,h,outer)
                expected=15*details['unique_union']+5
                if details['ridge_inner_solves']+details['ridge_outer_solves']!=expected:raise RuntimeError('Fit count mismatch')
                if len(details['GA']['visited'])!=len(details['random']['visited']):raise RuntimeError('Unequal search budget')
                if len(details['GA']['visited'])>3120:raise RuntimeError('Search exceeded cap')
                for arm,p in ps.items():
                    if len(p)!=len(te) or not np.isfinite(p).all():raise RuntimeError('Prediction coverage failed')
                    pred[arm][te]=p
                coverage[te]+=1;total+=expected
                details['training_ID_sha256']=hashlib.sha256('\n'.join(cohort[i]['PATNO'] for i in tr).encode()).hexdigest()
                details['test_ID_sha256']=hashlib.sha256('\n'.join(cohort[i]['PATNO'] for i in te).encode()).hexdigest()
                writejson(out/f'restricted/Context_{h}m_outer{outer}.json',details)
                counts={'horizon':h,'outer_fold':outer,'GA_unique':len(details['GA']['visited']),'random_unique':len(details['random']['visited']),'union_unique':details['unique_union'],'ridge_solves':expected};totals.append(counts)
                for arm,m in details['models'].items():selections.append(dict(horizon=h,outer_fold=outer,arm=arm,mask=m['mask'],features=m['features'],instruments=m['burden'],scores=len(m['features']),alpha=m['alpha']))
                writejson(out/'results/Progress.json',{'completed_contexts':len(totals),'ridge_solves':total,'synthetic':synthetic})
            if not np.all(coverage==1) or any(not np.isfinite(p).all() for p in pred.values()):raise RuntimeError('Incomplete OOF coverage')
            fields=['PATNO','component_id','y','outer_fold']+list(pred)
            with (out/f'restricted/OOF_{h}m.tsv').open('w',newline='') as f:
                w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader()
                for i,r in enumerate(cohort):w.writerow(dict(PATNO=r['PATNO'],component_id=r['component_id'],y=float(y[i]),outer_fold=r['outer_fold'],**{k:float(p[i]) for k,p in pred.items()}))
            for arm,p in pred.items():allmetrics.append(dict(horizon=h,arm=arm,**metrics(y,p)))
            for a,b in itertools.combinations(pred,2):
                allcontrasts.append(dict(horizon=h,arm_a=a,arm_b=b,**interval(y,pred[a],pred[b],seed=20261030+h)))
            if h==36:writejson(out/'results/Primary_Exploratory_Contrast.json',dict(horizon=36,contrast='GA_selected minus enhanced',**interval(y,pred['GA_selected'],pred['enhanced'],seed=20261066)))
        if COUNTERS['ridge_completed']-counter_start['ridge_completed']!=total:raise RuntimeError('Global fit ledger mismatch')
        if total>c['budget']['total_ridge_solve_upper']:raise RuntimeError('Total fit ceiling exceeded')
        for name,obj in [('Metrics',allmetrics),('Contrasts',allcontrasts),('Selections',selections),('Fit_Counts',totals)]:writejson(out/f'results/{name}.json',obj)
        writejson(out/'STATUS.json',{'status':'COMPLETED_AWAITING_INDEPENDENT_VERIFICATION','synthetic':synthetic,'ridge_solves':total,'participant_data_fits':0 if synthetic else total,'wall_seconds':time.perf_counter()-started})
        entries={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        writejson(out/'Artifact_Manifest.json',{'sha256':entries})
        return {'contexts':len(totals),'ridge_solves':total,'synthetic':synthetic,'wall_seconds':time.perf_counter()-started}
    except BaseException as exc:
        writejson(out/'STATUS.json',{'status':'FAILED','synthetic':synthetic,'completed_context_ridge_solves':total,'all_ridge_attempts':COUNTERS['ridge_attempts']-counter_start['ridge_attempts'],'all_ridge_completed':COUNTERS['ridge_completed']-counter_start['ridge_completed'],'error_type':type(exc).__name__,'message':str(exc)})
        raise
def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['validate','run']);parser.add_argument('--output');parser.add_argument('--authorization');args=parser.parse_args()
    c,by,cohorts,report=validate()
    if args.command=='validate':print(json.dumps(report));return
    if not args.output or not args.authorization:raise RuntimeError('Run requires explicit authorization record and fresh output path')
    authorization=json.loads(Path(args.authorization).read_text())
    if authorization.get('plan_lock_sha256')!=sha(ROOT/'PLAN_LOCK.json') or authorization.get('participant_training_authorized') is not True or not authorization.get('user_instruction'):raise RuntimeError('Missing or mismatched authorization')
    out=Path(args.output).resolve();study=ROOT.parent.resolve()
    if not out.is_relative_to(study) or out==study or out.is_relative_to(ROOT):raise RuntimeError('Output must be a new separate folder inside Clinical_hallmark')
    if out.exists():raise RuntimeError('Output already exists')
    result=execute_dataset(c,by,cohorts,out,execution_record={'authorization':authorization,'plan_lock_sha256':sha(ROOT/'PLAN_LOCK.json'),'contract_sha256':sha(ROOT/'config/Analysis_Contract.json')})
    print(json.dumps(result))
if __name__=='__main__':main()
