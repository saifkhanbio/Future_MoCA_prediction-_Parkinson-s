"""Resumable deterministic candidate search. Receives outer-training outcomes only."""
from support import *
from backend import Fitness
def record_choice(f,visited):
    records=[f.cache[m] for m in sorted(visited)]
    return {'choice':frozen.choose_final(records),'visited':sorted(visited),'pareto':frozen.pareto(records)}
def envelope(records,max_cost):
    return [min(r['mae'] for r in records if r['burden']<=b) for b in range(max_cost+1)]
def stagnated(history,rule,generation):
    if not rule or generation<rule['minimum_generation'] or len(history)<rule['window']+1:return False
    window=history[-rule['window']-1:]
    return max(a-b for a,b in zip(window[0]['envelope'],window[-1]['envelope']))<rule['minimum_progress'] and len({r['selected'] for r in window})==1
def rng_from(state,seed=None):
    rng=np.random.default_rng(seed)
    if state is not None:rng.bit_generator.state=state
    return rng
def children(pop,fitness,rng,d,count,visited,limit):
    records=[fitness.cache[m] for m in pop];rank,crowd=frozen.ranks_crowding(records);maxmask=2**d-1;accepted=[];seen=set(visited)
    def parent():
        a,b=rng.integers(0,len(pop),size=2);ka=(rank[a],-crowd[a]);kb=(rank[b],-crowd[b])
        return pop[a] if ka<kb else pop[b] if kb<ka else pop[int(rng.choice([a,b]))]
    for _ in range(count):
        if limit is not None and len(seen)>=limit:break
        a,b=parent(),parent()
        if rng.random()<.9:
            coin=int(rng.integers(0,maxmask+1));child=(a&coin)|(b&(~coin&maxmask))
        else:child=a
        child^=sum(1<<i for i,x in enumerate(rng.random(d)) if x<1/d)
        accepted.append(child);seen.add(child)
    return accepted
def run_search(f,store,p,spec,stop):
    d=len(f.c['features']['optional']);space=2**d;fixed=frozen.mandatory(f.c)
    s=store.get('state',{'phase':'GA','restart':0,'generation':0,'initialized':False,'population':[],'rng':None,'history':[],'restart_summaries':[],'steps':0})
    def checkpoint(algorithm=None,restart=0,masks=()):
        s['steps']+=1;store.checkpoint(s,algorithm,restart,masks)
    while s['phase'] not in ('REFIT','COMPLETE'):
        if stop(s):raise Paused('Context paused at checkpoint boundary')
        if s['phase']=='GA':
            ri=s['restart']
            if ri==len(p['seeds']):
                ga=record_choice(f,store.visited('GA'));store.set('GA',ga)
                s.update(phase='RANDOM',initialized=False,rng=None);checkpoint();continue
            seed=p['seeds'][ri]+10000*spec['horizon']+100*spec['outer']
            if not s['initialized']:
                rng=rng_from(None,seed);pop=list(fixed)
                while len(pop)<p['population']:pop.append(int(rng.integers(0,space)))
                if p['unique_limit'] is not None and len(set(pop))>p['unique_limit']:raise IntegrityError('Initial population exceeds restart budget')
                f.evaluate(pop,lambda:stop(s));visited=set(pop)
                rr=[f.cache[m] for m in sorted(visited)]
                s.update(initialized=True,population=pop,generation=0,rng=rng.bit_generator.state,history=[{'generation':0,'envelope':envelope(rr,max(frozen.burden(space-1,f.c),0)),'selected':frozen.choose_final(rr)['selected']['mask']}])
                checkpoint('GA',ri,pop);continue
            visited=store.visited('GA',ri)
            reason=None
            if p['unique_limit'] is not None and len(visited)>=p['unique_limit']:reason='unique_mask_limit'
            elif s['generation']>=p['generations']:reason='generation_limit'
            elif stagnated(s['history'],p.get('stagnation'),s['generation']):reason='stagnation'
            if reason:
                s['restart_summaries'].append({'restart':ri,'seed':p['seeds'][ri],'completed_generations':s['generation'],'unique_masks':len(visited),'reason':reason})
                s.update(restart=ri+1,generation=0,initialized=False,population=[],rng=None,history=[])
                checkpoint();continue
            rng=rng_from(s['rng']);offspring=children(s['population'],f,rng,d,p['offspring'],visited,p['unique_limit'])
            f.evaluate(offspring,lambda:stop(s));visited.update(offspring)
            union=s['population']+offspring;rr,cc=frozen.ranks_crowding([f.cache[m] for m in union])
            order=sorted(range(len(union)),key=lambda i:(rr[i],-cc[i],union[i]))
            s['population']=[union[i] for i in order[:p['population']]];s['rng']=rng.bit_generator.state
            if len(offspring)==p['offspring']:
                s['generation']+=1;records=[f.cache[m] for m in sorted(visited)]
                s['history'].append({'generation':s['generation'],'envelope':envelope(records,frozen.burden(space-1,f.c)),'selected':frozen.choose_final(records)['selected']['mask']})
                if p.get('stagnation'):s['history']=s['history'][-p['stagnation']['window']-1:]
            else:
                s['restart_summaries'].append({'restart':ri,'seed':p['seeds'][ri],'completed_generations':s['generation'],'partial_generation_offspring':len(offspring),'unique_masks':len(visited),'reason':'unique_mask_limit'})
                s.update(restart=ri+1,generation=0,initialized=False,population=[],rng=None,history=[])
            checkpoint('GA',ri,offspring);continue
        if s['phase']=='RANDOM':
            budget=len(store.visited('GA'))
            if not s['initialized']:
                f.evaluate(fixed,lambda:stop(s));rng=rng_from(None,20271030+10000*spec['horizon']+100*spec['outer'])
                s.update(initialized=True,rng=rng.bit_generator.state);checkpoint('RANDOM',0,fixed);continue
            seen=store.visited('RANDOM')
            if len(seen)==budget:
                store.set('random',record_choice(f,seen));store.set('selection_frozen_digest',digest({'GA':store.get('GA'),'random':store.get('random')}))
                s.update(phase='ENUM' if p['exhaustive'] else 'REFIT',cursor=0);checkpoint();continue
            if len(seen)>budget:raise IntegrityError('Random logical budget exceeded')
            rng=rng_from(s['rng']);added=[]
            while len(added)<p['batch_masks'] and len(seen)<budget:
                m=int(rng.integers(0,space))
                if m not in seen:seen.add(m);added.append(m)
            f.evaluate(added,lambda:stop(s));s['rng']=rng.bit_generator.state;checkpoint('RANDOM',0,added);continue
        if s['phase']=='ENUM':
            if digest({'GA':store.get('GA'),'random':store.get('random')})!=store.get('selection_frozen_digest'):raise IntegrityError('Selection archive changed during enumeration')
            if s['cursor']==space:
                if len(f.cache)!=space or set(f.cache)!=set(range(space)):raise IntegrityError('Incomplete exhaustive coverage')
                store.set('exhaustive',record_choice(f,range(space)));s['phase']='REFIT';checkpoint();continue
            end=min(space,s['cursor']+p['batch_masks'])
            f.evaluate(range(s['cursor'],end),lambda:stop(s));s['cursor']=end;checkpoint();continue
        raise IntegrityError('Unknown checkpoint phase')
    return s
def context_task(spec,c,p,path,backend,binding_id,deadline=None,stop_file=None,faults=None):
    validate_spec(spec,c);validate_profile(p,c);faults=faults or {}
    if faults and not spec['synthetic']:raise IntegrityError('Test fault passed to production')
    binding={'schema':1,'context_digest':digest(spec),'contract_digest':digest(c),'profile_digest':digest(p),'implementation':binding_id}
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
        s=run_search(f,store,p,spec,stop)
        ga=store.get('GA');random=store.get('random')
        if len(ga['visited'])!=len(random['visited']):raise IntegrityError('Unequal logical search budget')
        masks={'core':c['controls']['core_mask'],'enhanced':c['controls']['enhanced_mask'],'full':c['controls']['full_mask'],'GA_selected':ga['choice']['selected']['mask'],'random_selected':random['choice']['selected']['mask']}
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
                model={'mask':m,'alpha':fit['alpha'],'features':frozen.selected(m,c),'burden':frozen.burden(m,c),'coef':coef.tolist(),'intercept':intercept,'columns':[labels[i] for i in ix],'predictions':pp.tolist(),'preprocessing':state}
                store.put_refit(arm,model,event)
            pred[arm]=model['predictions'];models[arm]=model
        baseline=np.array([frozen.numeric(r['baseline_moca']) for r in spec['test_rows']])
        trainbase=np.array([frozen.numeric(r['baseline_moca']) for r in spec['train_rows']])
        pred['persistence']=baseline.tolist();pred['typical_change']=(baseline+np.median(np.asarray(spec['train_y'])-trainbase)).tolist()
        if any(len(v)!=len(spec['test_rows']) or not np.isfinite(v).all() for v in pred.values()):raise IntegrityError('Invalid final predictions')
        ledger=store.ledger();expected_masks=len(set(ga['visited'])|set(random['visited'])) if not p['exhaustive'] else 2**len(c['features']['optional'])
        if ledger['unique_masks']!=expected_masks or ledger['refits']!=len(masks):raise IntegrityError('Logical fit coverage mismatch')
        result={'status':'COMPLETE','context_id':spec['context_id'],'binding':binding,'GA':ga,'random':random,'exhaustive':store.get('exhaustive'),'restart_summaries':s['restart_summaries'],'predictions':pred,'models':models,'test_ids':[r['PATNO'] for r in spec['test_rows']],'test_components':[r['component_id'] for r in spec['test_rows']],'logical_fits':expected_masks*15+len(masks),'ledger':ledger,'fallbacks':store.get('fallbacks',[]),'final_backend':f.kind,'pid':os.getpid(),'synthetic':spec['synthetic']}
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
