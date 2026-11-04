"""Guarded 16-CPU/one-GPU coordinator. Study execution requires a future matching authorization record."""
from support import *
from backend import gpu_ready
from search import context_task
from binding import validate_plan,validate_data
from legacy import references
import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import argparse,threading,csv,itertools,importlib.util
def project(row,c):
    names,cats=frozen.layout(c)
    return {k:row[k] for k in names+list(cats)+['PATNO','component_id']}
def make_specs(c,by,cohorts,synthetic):
    specs=[]
    for arm in c['arms']:
        ca=frozen.for_arm(c,arm)
        for h,cohort in sorted(cohorts.items()):
            if len({r['PATNO'] for r in cohort})!=len(cohort) or len({r['component_id'] for r in cohort})!=len(cohort):raise IntegrityError('Duplicate evaluation identities')
            for r in cohort:
                if r['component_id']!=by[r['PATNO']]['component_id']:raise IntegrityError('Cohort identity join mismatch')
            for outer in range(5):
                tr=[r for r in cohort if int(r['outer_fold'])!=outer];te=[r for r in cohort if int(r['outer_fold'])==outer]
                if any(int(r[f'inner_for_outer_{outer}'])!=-1 for r in te):raise IntegrityError('Held-out inner assignment')
                spec={'context_id':f'{arm}_{h}m_outer{outer}','arm':arm,'horizon':h,'outer':outer,'train_rows':[project(by[r['PATNO']],ca) for r in tr],'train_y':[float(r['y']) for r in tr],'inner':[int(r[f'inner_for_outer_{outer}']) for r in tr],'test_rows':[project(by[r['PATNO']],ca) for r in te],'synthetic':synthetic}
                validate_spec(spec,ca);specs.append(spec)
    return specs
def validate_results(results,specs):
    expected={s['context_id']:s for s in specs}
    if len(results)!=len(expected) or {r['context_id'] for r in results}!=set(expected) or any(r['status']!='COMPLETE' for r in results):raise IntegrityError('All-context completion barrier failed')
    expected_arms={'GA_selected','random_selected'}
    if any('exhaustive_selected' in r['predictions'] for r in results):expected_arms.add('exhaustive_selected')
    if any(set(r['predictions'])!=expected_arms for r in results):raise IntegrityError('Different model arms across contexts')
    for r in results:
        spec=expected[r['context_id']]
        if r['test_ids']!=[x['PATNO'] for x in spec['test_rows']] or r['test_components']!=[x['component_id'] for x in spec['test_rows']]:raise IntegrityError('Prediction identity binding failed')
        if any(len(v)!=len(spec['test_rows']) or not np.isfinite(v).all() for v in r['predictions'].values()):raise IntegrityError('Prediction shape/finite check failed')
        if r['binding']['context_digest']!=digest(spec):raise IntegrityError('Context output digest mismatch')
def score(results,specs,cohorts,out):
    validate_results(results,specs)
    mapping={r['context_id']:r for r in results};metrics=[];contrasts=[];selections=[]
    for h,cohort in sorted(cohorts.items()):
        ids=[r['PATNO'] for r in cohort];where={v:i for i,v in enumerate(ids)};y=np.array([float(r['y']) for r in cohort]);pred={}
        for searcharm in ['unrestricted28','moca27']:
            local=['GA_selected','random_selected']
            if 'exhaustive_selected' in mapping[f'{searcharm}_{h}m_outer0']['predictions']:local.append('exhaustive_selected')
            for a in local:pred[searcharm+'_'+a]=np.full(len(ids),np.nan)
            coverage=np.zeros(len(ids),int)
            for outer in range(5):
                r=mapping[f'{searcharm}_{h}m_outer{outer}'];ix=[where[i] for i in r['test_ids']];coverage[ix]+=1
                for a in local:pred[searcharm+'_'+a][ix]=r['predictions'][a]
                for a,m in r['models'].items():selections.append({'horizon':h,'outer':outer,'arm':searcharm+'_'+a,'mask':m['mask'],'alpha':m['alpha'],'instruments':m['burden'],'feature_count':len(m['features']),'features':m['features']})
            if not np.all(coverage==1):raise IntegrityError('OOF coverage failed')
        if results[0]['synthetic']:
            # Synthetic reference controls are fixed fixture baselines, never real scores.
            base=np.array([float(r['baseline_moca']) for r in cohort])
            for arm in ['compact4','MoCA_plus18','core_plus18','prior_MoCA_GA','persistence','typical_change']:pred[arm]=base.copy()
        else:pred.update(references(h,cohort))
        if any(not np.isfinite(v).all() for v in pred.values()):raise IntegrityError('Nonfinite OOF output')
        rows=[{'PATNO':r['PATNO'],'component_id':r['component_id'],'y':float(y[i]),'outer_fold':int(r['outer_fold']),**{a:float(pp[i]) for a,pp in pred.items()}} for i,r in enumerate(cohort)]
        write_json(out/'restricted'/f'OOF_{h}m.json',rows)
        for a,pp in pred.items():metrics.append({'horizon':h,'arm':a,**frozen.metrics(y,pp)})
        for a,b in itertools.combinations(pred,2):contrasts.append({'horizon':h,'arm_a':a,'arm_b':b,**frozen.interval(y,pred[a],pred[b],seed=20261031+h)})
        if h==36:write_json(out/'results/Primary_Exploratory_Contrast.json',{'horizon':36,'contrast':'unrestricted28_GA_selected minus moca27_GA_selected','primary_metric':'delta_R2',**frozen.interval(y,pred['unrestricted28_GA_selected'],pred['moca27_GA_selected'],seed=20261031+h)})
    write_json(out/'results/Metrics.json',metrics);write_json(out/'results/Contrasts.json',contrasts);write_json(out/'results/Selections.json',selections)
def initial_elapsed(c,synthetic):
    if synthetic:return 0.
    prior=c['execution'].get('prior_active_seconds',0.)
    if isinstance(prior,bool) or not isinstance(prior,(int,float)) or not math.isfinite(prior) or prior<0:
        raise IntegrityError('Invalid inherited active time')
    return float(prior)
def execute(c,by,cohorts,p,out,synthetic,identity,resume=False,controls=None,execution_record=None):
    controls=controls or {}
    if not synthetic:authorize(c,p,identity,execution_record)
    if controls and not synthetic:raise IntegrityError('Synthetic controls forbidden for participant execution')
    if not synthetic:
        import shutil
        mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        if os.cpu_count()<20 or mem<24*1024**3 or shutil.disk_usage(STUDY).free<p['startup_disk_bytes']:raise IntegrityError('Locked startup resource gate failed')
    out=Path(out).resolve()
    if synthetic:
        if not out.is_relative_to(ROOT/'qa'):raise IntegrityError('Synthetic output must stay under launcher QA')
    elif out.parent!=STUDY:raise IntegrityError('Production output must be a fresh study sibling folder')
    specs=make_specs(c,by,cohorts,synthetic)
    for arm in c['arms']:validate_profile(p,frozen.for_arm(c,arm))
    signature={'implementation':identity,'contract':digest(c),'profile':digest(p),'contexts':{s['context_id']:digest(s) for s in specs},'evaluation':digest(cohorts),'synthetic':synthetic}
    elapsed=initial_elapsed(c,synthetic)
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
    state={'status':'RUNNING','binding':signature,'active_elapsed_seconds':elapsed,'heartbeat_UTC_unix':time.time(),'synthetic':synthetic,'participant_data_fits':0 if synthetic else None,'inherited_active_seconds':initial_elapsed(c,synthetic)}
    write_json(out/'RUN.json',state)
    if execution_record is not None and not resume:write_json(out/'restricted/Execution_Authorization.json',execution_record)
    done=threading.Event()
    def heartbeat():
        while not done.wait(1):
            if not synthetic and not stop_path.exists():
                import shutil
                mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                if mem<p['memory_available_pause_bytes'] or shutil.disk_usage(out).free<p['disk_free_pause_bytes']:
                    write_json(stop_path,{'reason':'Resource guard: checkpoint and withhold scoring'})
            state.update(active_elapsed_seconds=elapsed+time.monotonic()-start,heartbeat_UTC_unix=time.time());write_json(out/'RUN.json',state)
    hb=threading.Thread(target=heartbeat,daemon=True);hb.start()
    cp=None;gp=None;results=[];future_map={};failure=None;cpu_only=controls.get('force_cpu',False)
    try:
        if not cpu_only:
            gp=cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'))
            try:gpu_info=gp.submit(gpu_ready,controls.get('gpu_unavailable',False)).result()
            except (HardwareFailure,BrokenProcessPool) as ex:
                events.append({'type':'GPU_UNAVAILABLE','reason':str(ex),'fallback':'18_CPU'});gp.shutdown(wait=True,cancel_futures=True);gp=None;cpu_only=True
        cp=cf.ProcessPoolExecutor(max_workers=p['cpu_only_workers'] if cpu_only else p['cpu_workers'],mp_context=mp.get_context('spawn'))
        routing={};pending=list(specs);overrides={}
        worker_limit=p['cpu_only_workers'] if cpu_only else p['cpu_workers']
        def enqueue(spec,kind):
            pool=gp if kind=='gpu' else cp
            faults=overrides.get(spec['context_id'],controls.get('context_faults',{}).get(spec['context_id'],{}))
            fut=pool.submit(context_task,spec,frozen.for_arm(c,spec['arm']),p,out/'restricted'/spec['context_id'],kind,identity,deadline,stop_path,faults)
            future_map[fut]=(spec,kind)
            events.append({'type':'DISPATCH','context':spec['context_id'],'backend':kind,'retry':spec['context_id'] in routing})
            routing[spec['context_id']]=kind
        def dispatch():
            if time.monotonic()>=deadline and not stop_path.exists():
                write_json(stop_path,{'reason':'Cumulative active-time guard; pause without scoring'})
            if stop_path.exists():return
            # A shared queue keeps idle CPUs from waiting behind a fixed GPU backlog.
            if pending and gp is not None and not any(kind=='gpu' for _,kind in future_map.values()):
                enqueue(pending.pop(0),'gpu')
            free=worker_limit-sum(kind=='cpu' for _,kind in future_map.values())
            for _ in range(min(free,len(pending))):enqueue(pending.pop(0),'cpu')
            write_json(out/'results/Routing.json',{'CPU_workers':worker_limit,'GPU_workers':0 if gp is None else 1,
                       'scheduler':'shared_unstarted_context_queue','assignments':routing,'events':events})
        refit_phase=False
        while True:
            dispatch()
            while future_map:
                ready,_=cf.wait(future_map,timeout=1,return_when=cf.FIRST_COMPLETED)
                for fut in ready:
                    spec,kind=future_map.pop(fut)
                    try:r=fut.result()
                    except BrokenProcessPool:
                        if kind!='gpu':raise
                        if gp is not None:gp.shutdown(wait=True,cancel_futures=True);gp=None
                        events.append({'type':'GPU_PROCESS_LOSS','context':spec['context_id'],'fallback':'CPU_resume_unchanged_checkpoint'})
                        faults=dict(controls.get('context_faults',{}).get(spec['context_id'],{}));faults.pop('crash_after_batch',None)
                        overrides[spec['context_id']]=faults
                        pending.insert(0,spec)
                        continue
                    results.append(r)
                    if r['status'] in ('PAUSED','CONVERGENCE_HOLD') and not stop_path.exists():
                        write_json(stop_path,{'reason':'Context paused; all-context scoring withheld'})
                dispatch()
            if len(results)==len(specs) and all(r['status']=='SEARCH_READY' for r in results):
                if refit_phase:raise IntegrityError('Refit gate did not advance')
                expected={spec['context_id']:spec for spec in specs}
                for r in results:
                    if not r['convergence']['passed'] or r['binding']['context_digest']!=digest(expected[r['context_id']]):raise IntegrityError('Invalid convergence readiness record')
                write_json(out/'restricted/Convergence_Gate.json',{'status':'PASS','binding':signature,
                    'selection_digests':{r['context_id']:r['selection_digest'] for r in results},
                    'arm_contracts':{a:digest(frozen.for_arm(c,a)) for a in c['arms']}})
                refit_phase=True;pending=list(specs);results=[];continue
            break
        if pending or len(results)!=len(specs) or any(r['status']!='COMPLETE' for r in results):
            state['status']='CONVERGENCE_HOLD' if any(r['status']=='CONVERGENCE_HOLD' for r in results) else 'PAUSED'
        else:
            validate_results(results,specs)
            write_json(out/'restricted/All_Context_Results.json',results)
            # Held-out outcomes are first consumed for metrics only after all contexts complete.
            score(results,specs,cohorts,out)
            state['status']='COMPLETE'
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
    validate_plan()
    identity=source_identity()
    lockpath=ROOT/'LAUNCHER_LOCK.json'
    if require_lock and not lockpath.exists():raise IntegrityError('Launcher readiness lock absent')
    if lockpath.exists():
        lock=json.loads(lockpath.read_text())
        for rel,h in lock['sha256'].items():
            if file_sha(ROOT/rel)!=h:raise IntegrityError('Launcher lock hash mismatch')
        if lock['implementation_identity']!=identity:raise IntegrityError('Implementation binding changed')
    return identity
def authorize(c,p,identity,auth):
    if not isinstance(auth,dict) or auth.get('participant_training_authorized') is not True:raise IntegrityError('Separate explicit execution authorization required')
    if identity!=validate_release():raise IntegrityError('Implementation identity mismatch')
    if digest(c)!=digest(json.loads((ROOT/'config/Analysis_Contract.json').read_text())) or digest(p)!=digest(json.loads((ROOT/'config/r2_expanded_v1.json').read_text())):raise IntegrityError('Nonlocked participant contract/profile')
    required={'participant_training_authorized':True,'study_id':'Clinical_hallmark_GA_unrestricted_v4_200k_10kgen','plan_lock_sha256':file_sha(ROOT/'PLAN_LOCK.json'),'launcher_lock_sha256':file_sha(ROOT/'LAUNCHER_LOCK.json'),'parent_plan_lock_sha256':file_sha(FROZEN/'PLAN_LOCK.json'),'profile_sha256':file_sha(ROOT/'config/r2_expanded_v1.json'),'profile_id':'joint28_27_200k_10kgen_v4'}
    if any(auth.get(k)!=v for k,v in required.items()) or not isinstance(auth.get('user_instruction'),str) or not auth['user_instruction'].strip():raise IntegrityError('Missing or mismatched R2 execution authorization')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['validate','run']);ap.add_argument('--profile',choices=['r2_expanded_v1'],default='r2_expanded_v1');ap.add_argument('--output');ap.add_argument('--authorization');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    identity=validate_release()
    if args.command=='validate':
        c,by,cohorts,report=validate_data();p=json.loads((ROOT/'config/r2_expanded_v1.json').read_text())
        for arm in c['arms']:validate_profile(p,frozen.for_arm(c,arm))
        print(json.dumps({'status':'VALIDATED_READ_ONLY','implementation_identity':identity,'participant_fits':0,'search_started':False,'binding_report':report}));return
    if not args.authorization or not args.output:raise IntegrityError('Participant execution requires a separate explicit R2 authorization record')
    c=json.loads((ROOT/'config/Analysis_Contract.json').read_text());p=json.loads((ROOT/'config/r2_expanded_v1.json').read_text());auth=json.loads(Path(args.authorization).read_text())
    authorize(c,p,identity,auth)
    out=Path(args.output).resolve()
    if out.parent!=STUDY or out==ROOT or out==FROZEN or (out.exists() and not args.resume):raise IntegrityError('Fresh separate study output required')
    c,by,cohorts,report=validate_data()
    result=execute(c,by,cohorts,p,out,False,identity,resume=args.resume,execution_record=auth)
    print(json.dumps(result))
if __name__=='__main__':main()
