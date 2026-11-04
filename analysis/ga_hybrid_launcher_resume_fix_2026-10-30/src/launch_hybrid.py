"""Guarded 12-CPU/one-GPU coordinator. Study execution requires a future matching authorization record."""
from support import *
from backend import gpu_ready
from search import context_task
import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import argparse,threading,csv,itertools,importlib.util
def project(row,c):
    names,cats=frozen.layout(c)
    return {k:row[k] for k in names+list(cats)+['PATNO','component_id']}
def make_specs(c,by,cohorts,synthetic):
    specs=[]
    for h,cohort in sorted(cohorts.items()):
        if len({r['PATNO'] for r in cohort})!=len(cohort) or len({r['component_id'] for r in cohort})!=len(cohort):raise IntegrityError('Duplicate evaluation identities')
        for r in cohort:
            if r['component_id']!=by[r['PATNO']]['component_id']:raise IntegrityError('Cohort identity join mismatch')
        for outer in range(5):
            tr=[r for r in cohort if int(r['outer_fold'])!=outer];te=[r for r in cohort if int(r['outer_fold'])==outer]
            if any(int(r[f'inner_for_outer_{outer}'])!=-1 for r in te):raise IntegrityError('Held-out inner assignment')
            spec={'context_id':f'{h}m_outer{outer}','horizon':h,'outer':outer,'train_rows':[project(by[r['PATNO']],c) for r in tr],'train_y':[float(r['y']) for r in tr],'inner':[int(r[f'inner_for_outer_{outer}']) for r in tr],'test_rows':[project(by[r['PATNO']],c) for r in te],'synthetic':synthetic}
            validate_spec(spec,c);specs.append(spec)
    return specs
def validate_results(results,specs):
    expected={s['context_id']:s for s in specs}
    if len(results)!=len(expected) or {r['context_id'] for r in results}!=set(expected) or any(r['status']!='COMPLETE' for r in results):raise IntegrityError('All-context completion barrier failed')
    expected_arms={'core','enhanced','full','GA_selected','random_selected','persistence','typical_change'}
    if any('exhaustive_selected' in r['predictions'] for r in results):expected_arms.add('exhaustive_selected')
    if any(set(r['predictions'])!=expected_arms for r in results):raise IntegrityError('Different model arms across contexts')
    for r in results:
        spec=expected[r['context_id']]
        if r['test_ids']!=[x['PATNO'] for x in spec['test_rows']] or r['test_components']!=[x['component_id'] for x in spec['test_rows']]:raise IntegrityError('Prediction identity binding failed')
        if any(len(v)!=len(spec['test_rows']) or not np.isfinite(v).all() for v in r['predictions'].values()):raise IntegrityError('Prediction shape/finite check failed')
        if r['binding']['context_digest']!=digest(spec):raise IntegrityError('Context output digest mismatch')
def score(results,specs,cohorts,out):
    validate_results(results,specs) # No scoring or metric files before this barrier.
    mapping={r['context_id']:r for r in results};metrics=[];contrasts=[];selections=[]
    for h,cohort in sorted(cohorts.items()):
        ids=[r['PATNO'] for r in cohort];where={v:i for i,v in enumerate(ids)};y=np.array([float(r['y']) for r in cohort])
        arms=['core','enhanced','full','GA_selected','random_selected']
        if 'exhaustive_selected' in mapping[f'{h}m_outer0']['predictions']:arms.append('exhaustive_selected')
        arms+=['persistence','typical_change']
        pred={a:np.full(len(ids),np.nan) for a in arms};coverage=np.zeros(len(ids),int)
        for outer in range(5):
            r=mapping[f'{h}m_outer{outer}']
            if set(r['predictions'])!=set(arms):raise IntegrityError('Different model arms across contexts')
            ix=[where[i] for i in r['test_ids']];coverage[ix]+=1
            for a in arms:pred[a][ix]=r['predictions'][a]
            for a,m in r['models'].items():selections.append({'horizon':h,'outer':outer,'arm':a,'mask':m['mask'],'alpha':m['alpha'],'instruments':m['burden'],'features':m['features']})
        if not np.all(coverage==1) or any(not np.isfinite(v).all() for v in pred.values()):raise IntegrityError('OOF coverage failed')
        rows=[]
        for i,r in enumerate(cohort):rows.append({'PATNO':r['PATNO'],'component_id':r['component_id'],'y':float(y[i]),'outer_fold':int(r['outer_fold']),**{a:float(p[i]) for a,p in pred.items()}})
        write_json(out/'restricted'/f'OOF_{h}m.json',rows)
        for a,p in pred.items():metrics.append({'horizon':h,'arm':a,**frozen.metrics(y,p)})
        for a,b in itertools.combinations(arms,2):contrasts.append({'horizon':h,'arm_a':a,'arm_b':b,**frozen.interval(y,pred[a],pred[b],seed=20261030+h)})
        if h==36:write_json(out/'results/Primary_Exploratory_Contrast.json',{'horizon':36,'contrast':'GA_selected minus enhanced',**frozen.interval(y,pred['GA_selected'],pred['enhanced'],seed=20261066)})
    write_json(out/'results/Metrics.json',metrics);write_json(out/'results/Contrasts.json',contrasts);write_json(out/'results/Selections.json',selections)
def execute(c,by,cohorts,p,out,synthetic,identity,resume=False,controls=None,execution_record=None):
    controls=controls or {}
    if controls and not synthetic:raise IntegrityError('Synthetic controls forbidden for participant execution')
    out=Path(out).resolve()
    if synthetic:
        if not out.is_relative_to(ROOT/'qa'):raise IntegrityError('Synthetic output must stay under launcher QA')
    elif out.parent!=STUDY:raise IntegrityError('Production output must be a fresh study sibling folder')
    specs=make_specs(c,by,cohorts,synthetic);validate_profile(p,c)
    signature={'implementation':identity,'contract':digest(c),'profile':digest(p),'contexts':{s['context_id']:digest(s) for s in specs},'evaluation':digest(cohorts),'synthetic':synthetic}
    elapsed=0.
    if resume:
        if not out.exists():raise IntegrityError('Resume directory absent')
        previous=json.loads((out/'RUN.json').read_text())
        if previous['binding']!=signature:raise IntegrityError('Run resume binding mismatch')
        if previous['status']=='COMPLETE':raise IntegrityError('Run already completed; verify rather than re-execute')
        elapsed=previous['active_elapsed_seconds']
        if previous['status']=='RUNNING':elapsed+=max(0,time.time()-previous['heartbeat_UTC_unix'])

    else:
        if out.exists():raise IntegrityError('Output exists; explicit resume required')
        out.mkdir();(out/'restricted').mkdir(mode=0o700);(out/'results').mkdir()
    lock=(out/'coordinator.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise IntegrityError('Another coordinator owns this run')
    if resume and (out/'STOP_REQUESTED.json').exists():(out/'STOP_REQUESTED.json').rename(out/f'STOP_HISTORY_{time.time_ns()}.json')
    start=time.monotonic();deadline=start+max(0,p['active_guard_seconds']-elapsed);stop_path=out/'STOP_REQUESTED.json';events=[]
    state={'status':'RUNNING','binding':signature,'active_elapsed_seconds':elapsed,'heartbeat_UTC_unix':time.time(),'synthetic':synthetic,'participant_data_fits':0 if synthetic else None}
    write_json(out/'RUN.json',state)
    if execution_record is not None and not resume:write_json(out/'restricted/Execution_Authorization.json',execution_record)
    done=threading.Event()
    def heartbeat():
        while not done.wait(1):
            state.update(active_elapsed_seconds=elapsed+time.monotonic()-start,heartbeat_UTC_unix=time.time());write_json(out/'RUN.json',state)
    hb=threading.Thread(target=heartbeat,daemon=True);hb.start()
    cp=None;gp=None;results=[];future_map={};failure=None;cpu_only=controls.get('force_cpu',False)
    try:
        if not cpu_only:
            gp=cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'))
            try:gpu_info=gp.submit(gpu_ready,controls.get('gpu_unavailable',False)).result()
            except (HardwareFailure,BrokenProcessPool) as ex:
                events.append({'type':'GPU_UNAVAILABLE','reason':str(ex),'fallback':'15_CPU'});gp.shutdown(wait=True,cancel_futures=True);gp=None;cpu_only=True
        cp=cf.ProcessPoolExecutor(max_workers=p['cpu_only_workers'] if cpu_only else p['cpu_workers'],mp_context=mp.get_context('spawn'))
        routing={}
        for s in specs:
            use_gpu=not cpu_only and s['outer']==4;pool=gp if use_gpu else cp;kind='gpu' if use_gpu else 'cpu';routing[s['context_id']]=kind
            faults=controls.get('context_faults',{}).get(s['context_id'],{})
            fut=pool.submit(context_task,s,c,p,out/'restricted'/s['context_id'],kind,identity,deadline,stop_path,faults)
            future_map[fut]=(s,kind)
        write_json(out/'results/Routing.json',{'CPU_workers':p['cpu_only_workers'] if cpu_only else p['cpu_workers'],'GPU_workers':0 if cpu_only else 1,'assignments':routing,'events':events})
        while future_map:
            ready,_=cf.wait(future_map,timeout=1,return_when=cf.FIRST_COMPLETED)
            for fut in ready:
                s,kind=future_map.pop(fut)
                try:r=fut.result()
                except BrokenProcessPool as ex:
                    if kind!='gpu':raise
                    if gp is not None:gp.shutdown(wait=True,cancel_futures=True);gp=None
                    events.append({'type':'GPU_PROCESS_LOSS','context':s['context_id'],'fallback':'CPU_resume_unchanged_checkpoint'})
                    faults=dict(controls.get('context_faults',{}).get(s['context_id'],{}));faults.pop('crash_after_batch',None)
                    retry=cp.submit(context_task,s,c,p,out/'restricted'/s['context_id'],'cpu',identity,deadline,stop_path,{**faults,**({'gpu_unavailable':True} if synthetic else {})})
                    future_map[retry]=(s,'cpu');continue
                results.append(r)
                if r['status']=='PAUSED' and not stop_path.exists():write_json(stop_path,{'reason':'Context paused; all-context scoring withheld'})
        if any(r['status']!='COMPLETE' for r in results):
            state['status']='PAUSED'
        else:
            validate_results(results,specs)
            write_json(out/'restricted/All_Context_Results.json',results)
            # Held-out outcomes are first consumed for metrics only after all contexts complete.
            score(results,specs,cohorts,out);state['status']='COMPLETE'
            state['logical_fits']=sum(r['logical_fits'] for r in results)
            if not p['exhaustive'] and state['logical_fits']>c['budget']['total_ridge_solve_upper']:raise IntegrityError('Locked logical fit ceiling exceeded')
            state['participant_data_fits']=0 if synthetic else state['logical_fits']
        write_json(out/'results/Execution_Events.json',events)
    except BaseException as ex:
        failure=ex;write_json(stop_path,{'reason':'Task failure; cancel other contexts','type':type(ex).__name__})
        for f in future_map:f.cancel()
        state.update(status='FAILED',error={'type':type(ex).__name__,'message':str(ex)})
        write_json(out/'results/Execution_Events.json',events)
    finally:
        if cp:cp.shutdown(wait=True,cancel_futures=True)
        if gp:gp.shutdown(wait=True,cancel_futures=True)
        done.set();hb.join()
        state.update(active_elapsed_seconds=elapsed+time.monotonic()-start,heartbeat_UTC_unix=time.time())
        write_json(out/'RUN.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if failure:raise failure
    return state
def validate_release(require_lock=True):
    identity=source_identity()
    lockpath=ROOT/'LAUNCHER_LOCK.json'
    if require_lock and not lockpath.exists():raise IntegrityError('Launcher readiness lock absent')
    if lockpath.exists():
        lock=json.loads(lockpath.read_text())
        for rel,h in lock['sha256'].items():
            if file_sha(ROOT/rel)!=h:raise IntegrityError('Launcher lock hash mismatch')
        if lock['implementation_identity']!=identity:raise IntegrityError('Implementation binding changed')
    return identity
def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['validate','run']);ap.add_argument('--profile',choices=['locked_v1','expanded_proposed'],default='locked_v1');ap.add_argument('--output');ap.add_argument('--authorization');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    identity=validate_release()
    if args.command=='validate':
        print(json.dumps({'status':'VALIDATED_READ_ONLY','implementation_identity':identity,'participant_fits':0,'search_started':False}));return
    if not args.authorization or not args.output:raise IntegrityError('Participant execution requires a separate explicit authorization record; none is supplied by this package')
    ppath=ROOT/f'config/{args.profile}.json';auth=json.loads(Path(args.authorization).read_text())
    required={'participant_training_authorized':True,'launcher_lock_sha256':file_sha(ROOT/'LAUNCHER_LOCK.json'),'parent_plan_lock_sha256':file_sha(FROZEN/'PLAN_LOCK.json'),'profile_sha256':file_sha(ppath),'profile_id':args.profile}
    if any(auth.get(k)!=v for k,v in required.items()) or not isinstance(auth.get('user_instruction'),str) or not auth['user_instruction'].strip():raise IntegrityError('Missing or mismatched execution authorization')
    out=Path(args.output).resolve()
    if out.parent!=STUDY or out==ROOT or out==FROZEN or (out.exists() and not args.resume):raise IntegrityError('Fresh separate study output required')
    # Import old read-only binding validator only after the authorization gate.
    spec=importlib.util.spec_from_file_location('frozen_launcher_binding',FROZEN/'src/launch.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    c,by,cohorts,report=old.validate()
    p=json.loads(ppath.read_text())
    result=execute(c,by,cohorts,p,out,False,identity,resume=args.resume,execution_record=auth)
    print(json.dumps(result))
if __name__=='__main__':main()
