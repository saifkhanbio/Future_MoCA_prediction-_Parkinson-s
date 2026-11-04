"""Training-only bounded search and gated outer refits."""
from support import *
from backend import Fitness
from search_core import *
def context_task(spec,c,p,path,backend,binding_id,deadline=None,stop_file=None,faults=None):
    begun=time.time()
    validate_spec(spec,c);validate_profile(p,c);faults=faults or {}
    if faults.get('not_before'):
        while time.time()<faults['not_before']:time.sleep(.05)
    if faults and not spec['synthetic']:raise IntegrityError('Test fault passed to production')
    work_started=time.time()
    if not spec['synthetic']:
        from launch_hybrid import authorize
        parent=json.loads((ROOT/'config/Analysis_Contract.json').read_text())
        authpath=Path(path).parent/'Execution_Authorization.json'
        if not authpath.is_file():raise IntegrityError('Participant context lacks execution authorization')
        authorize(parent,p,binding_id,json.loads(authpath.read_text()))
        if digest(c)!=digest(frozen.for_arm(parent,spec['arm'])):raise IntegrityError('Nonlocked participant arm')

    binding={'schema':'joint28_r2_v1','context_digest':digest(spec),'contract_digest':digest(c),'profile_digest':digest(p),'implementation':binding_id}
    store=Store(path,binding)
    try:
        complete=store.get('context_status')=='COMPLETE'
        f=Fitness(spec,c,store,backend,p['batch_masks'],faults,initialize_device=not complete)
        if complete:
            result=store.get('result')
            if not result or result['binding']!=binding:raise IntegrityError('Completed result binding mismatch')
            if any(store.refit(arm)!=model for arm,model in result['models'].items()):raise IntegrityError('Completed refit mismatch')
            if store.ledger()['unique_masks']!=result['ledger']['unique_masks']:raise IntegrityError('Completed coverage mismatch')
            return result
        def stop(state=None):
            return (deadline is not None and time.monotonic()>=deadline) or (stop_file is not None and Path(stop_file).exists()) or (faults.get('pause_after_steps') is not None and (state or store.get('state',{})).get('steps',0)>=faults['pause_after_steps'])
        search_started=time.monotonic()
        s=run_search(f,store,p,spec,stop)
        s['timing']['search_seconds']=s['timing'].get('search_seconds',0.)+time.monotonic()-search_started
        store.set('state',s)
        ga=store.get('GA');random=store.get('random')
        if ga['visited_count']!=random['visited_count']:raise IntegrityError('Unequal logical search budget')
        if p.get('convergence',{}).get('enforce_gate',False):
            report=s['convergence'];selection_digest=store.get('selection_frozen_digest')
            ready={'status':'SEARCH_READY' if report['passed'] else 'CONVERGENCE_HOLD','context_id':spec['context_id'],
                   'convergence':report,'selection_digest':selection_digest,'binding':binding,'ledger':store.ledger(),'synthetic':spec['synthetic']}
            if not report['passed']:
                store.set('context_status','CONVERGENCE_HOLD');store.set('readiness',ready)
                write_json(Path(path)/'Convergence.json',ready);return ready
            gatefile=Path(path).parent/'Convergence_Gate.json'
            if not gatefile.exists():
                store.set('context_status','SEARCH_READY');store.set('readiness',ready)
                write_json(Path(path)/'Convergence.json',ready);return ready
            proof=json.loads(gatefile.read_text());sig=proof['binding']
            expected={f'{a}_{h}m_outer{o}' for a in c['arms'] for h in [12,24,36] for o in range(5)}
            if proof['status']!='PASS' or set(proof['selection_digests'])!=expected or proof['selection_digests'].get(spec['context_id'])!=selection_digest or sig['implementation']!=binding_id or sig['profile']!=digest(p) or sig['contexts'].get(spec['context_id'])!=digest(spec) or proof['arm_contracts'].get(spec['arm'])!=digest(c):
                raise IntegrityError('All-context convergence proof mismatch')
        masks={'GA_selected':ga['choice']['selected']['mask'],'random_selected':random['choice']['selected']['mask']}
        if p['exhaustive']:masks['exhaustive_selected']=store.get('exhaustive')['choice']['selected']['mask']
        state=frozen.prepare(spec['train_rows'],c);X,labels=frozen.transform(spec['train_rows'],state);V,labels2=frozen.transform(spec['test_rows'],state)
        if labels!=labels2:raise IntegrityError('Outer design binding mismatch')
        pred={};models={}
        for arm,m in masks.items():
            if stop(s):raise Paused('Refit paused at arm boundary')
            model=store.refit(arm)
            if model is None:
                fit=f.cache[m];ix=frozen.column_indices(m,c,labels)
                event=store.event('outer_refit',1)
                before=dict(frozen.COUNTERS)
                try:coef,intercept=frozen.ridge(X[:,ix],spec['train_y'],fit['alpha']);pp=frozen.predict(V[:,ix],(coef,intercept))
                except Exception as ex:
                    store.finish_event(event,'FAILED',frozen.COUNTERS['ridge_attempts']-before['ridge_attempts'],frozen.COUNTERS['ridge_completed']-before['ridge_completed'],str(ex));raise
                model={'inner_mse':fit['mse'],'inner_r2':fit['r2'],'mask':m,'alpha':fit['alpha'],'features':c['features']['core']['numeric']+list(c['features']['core']['categories'])+frozen.selected(m,c),'burden':frozen.burden(m,c),'coef':coef.tolist(),'intercept':intercept,'columns':[labels[i] for i in ix],'predictions':pp.tolist(),'preprocessing':state}
                store.put_refit(arm,model,event)
            pred[arm]=model['predictions'];models[arm]=model
        if any(len(v)!=len(spec['test_rows']) or not np.isfinite(v).all() for v in pred.values()):raise IntegrityError('Invalid final predictions')
        ledger=store.ledger();expected_masks=store.db.execute('SELECT COUNT(*) FROM (SELECT mask FROM visit_union GROUP BY mask)').fetchone()[0]
        if ledger['unique_masks']!=expected_masks or ledger['refits']!=len(masks):raise IntegrityError('Logical fit coverage mismatch')
        result={'status':'COMPLETE','context_id':spec['context_id'],'binding':binding,'GA':ga,'random':random,'exhaustive':store.get('exhaustive'),'restart_summaries':s['restart_summaries'],'predictions':pred,'models':models,'test_ids':[r['PATNO'] for r in spec['test_rows']],'test_components':[r['component_id'] for r in spec['test_rows']],'logical_fits':expected_masks*15+len(masks),'ledger':ledger,'fallbacks':store.get('fallbacks',[]),'final_backend':f.kind,'pid':os.getpid(),'arm':spec['arm'],'start_unix':begun,'work_start_unix':work_started,'end_unix':time.time(),'synthetic':spec['synthetic'],'convergence':s.get('convergence'),'timing':s['timing']}
        s['phase']='COMPLETE'
        with store.db:
            store.set('state',s,False);store.set('result',result,False);store.set('context_status','COMPLETE',False)
        write_json(Path(path)/'Result.json',result)
        return result
    except Paused as ex:
        store.set('context_status','PAUSED')
        return {'status':'PAUSED','context_id':spec['context_id'],'reason':str(ex),'ledger':store.ledger(),'synthetic':spec['synthetic']}
    except BaseException as ex:
        store.set('context_status','FAILED');store.set('error',{'type':type(ex).__name__,'message':str(ex)})
        raise
    finally:store.close()
