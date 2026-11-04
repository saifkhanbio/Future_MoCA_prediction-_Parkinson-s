"""Bounded-memory search archives and prespecified convergence rules."""
from support import *
def new_archive(max_cost):return {'minimum':None,'ties':[],'points':[None]*(max_cost+1)}
def update_archive(a,records):
    for raw in records:
        r={k:raw[k] for k in ['mask','mse','normalized_mse','r2','burden','scores','alpha']}
        if a['minimum'] is None or r['mse']<a['minimum']:
            a['minimum']=r['mse'];a['ties']=[t for t in a['ties'] if t['mse']<=a['minimum']+1e-12]
        if r['mse']<=a['minimum']+1e-12:
            same=next((i for i,t in enumerate(a['ties']) if t['mse']==r['mse']),None)
            key=lambda x:(x['scores'],x['burden'],x['mask'])
            if same is None:a['ties'].append(r)
            elif key(r)<key(a['ties'][same]):a['ties'][same]=r
        old=a['points'][r['burden']]
        if old is None or (r['mse'],r['scores'],r['mask'])<(old['mse'],old['scores'],old['mask']):a['points'][r['burden']]=r
    return a
def archive_selected(a):return min(a['ties'],key=lambda r:(r['scores'],r['burden'],r['mask']))
def archive_envelope(a):
    best=None;out=[]
    for r in a['points']:
        if r is not None:best=r['normalized_mse'] if best is None else min(best,r['normalized_mse'])
        out.append(best)
    return out
def envelope(records,max_cost):return archive_envelope(update_archive(new_archive(max_cost),records))
def record_choice(f,store,algorithm,a,synthetic):
    selected=f.cache[archive_selected(a)['mask']];front=[];prev=float('inf')
    for r in a['points']:
        if r is not None:
            if r['mse']<prev:front.append(r)
            prev=min(prev,r['mse'])
    result={'choice':{'selected':selected,'anchor_mask':selected['mask'],'anchor_mse':selected['mse'],'heuristic_SE':0.,
         'threshold':a['minimum']+1e-12,'selection_rule':'maximum_inner_accuracy_then_total_feature_count'},
         'visited_count':len(store.visited(algorithm)),'visited_digest':store.union_digests[algorithm],
         'pareto':front,'envelope':archive_envelope(a),'pareto_ties':'one canonical mask per identical objective point'}
    if synthetic and result['visited_count']<=100000:result['visited']=list(store.visited(algorithm))
    return result
def stagnated(history,rule,generation,unique_masks=None):
    if not rule:return False
    windows=rule.get('consecutive_windows',1);width=rule['window'];total=windows*width
    if generation<rule['minimum_generation'] or len(history)<total+1:return False
    if (unique_masks or 0)<rule.get('minimum_unique',0):return False
    rows=history[-total-1:]
    if len({r['selected'] for r in rows})!=1:return False
    for i in range(windows):
        a,b=rows[i*width]['envelope'],rows[(i+1)*width]['envelope']
        if any((x is None)!=(y is None) for x,y in zip(a,b)):return False
        gains=[x-y for x,y in zip(a,b) if x is not None]
        if not gains or max(gains)>=rule['minimum_progress']:return False
    return True
def convergence_report(s,p):
    rule=p.get('convergence')
    if not rule:return {'passed':True,'enforced':False,'reason':'Synthetic transport test; no convergence claim'}
    summaries=s['restart_summaries'];n=rule['main_restarts'];tol=rule['tolerance']
    main=summaries[:n];confirm=summaries[n:]
    width=len(s['ga_archive']['points'])-1
    def merged(rows):
        a=new_archive(width)
        for row in rows:update_archive(a,[x for x in row['archive']['points'] if x]+row['archive']['ties'])
        return a
    a,b=merged(main),merged(confirm);g=s['ga_archive'];ga=archive_selected(g)
    def gap(x):
        xx,yy=archive_envelope(x),archive_envelope(g)
        if any((i is None)!=(j is None) for i,j in zip(xx,yy)):return None
        return max(abs(i-j) for i,j in zip(xx,yy) if i is not None)
    ma,co=gap(a),gap(b);rep=sum(archive_selected(x['archive'])['normalized_mse']<=ga['normalized_mse']+tol for x in confirm)
    random_gain=ga['normalized_mse']-archive_selected(s['random_archive'])['normalized_mse']
    checks={'scheduled_restarts':len(main)==n and len(confirm)==rule['confirmation_restarts'],
       'all_restarts_stabilized':not rule['require_all_stagnated'] or all(x['reason']=='stagnation' for x in summaries),
       'main_envelope_reproduced':ma is not None and ma<=tol,'confirmation_envelope_reproduced':co is not None and co<=tol,
       'confirmation_best_replicated':rep>=rule['minimum_replicates'],'random_challenge_passed':random_gain<=tol}
    return dict(passed=all(checks.values()),enforced=rule.get('enforce_gate',True),checks=checks,
       main_envelope_max_gap_R2=ma,confirmation_envelope_max_gap_R2=co,confirmation_replicates=rep,
       random_gain_R2=random_gain,stagnated_restarts=sum(x['reason']=='stagnation' for x in summaries),
       hard_limit_restarts=sum(x['reason']!='stagnation' for x in summaries),global_optimum_proven=False)
def rng_from(state,seed=None):
    rng=np.random.default_rng(seed)
    if state is not None:rng.bit_generator.state=state
    return rng
def children(pop,fitness,rng,d,count,visited,limit):
    records=[fitness.cache[m] for m in pop];rank,crowd=frozen.ranks_crowding(records);maxmask=2**d-1;accepted=[];novel=set();initial=len(visited)
    def parent():
        a,b=rng.integers(0,len(pop),size=2);ka=(rank[a],-crowd[a]);kb=(rank[b],-crowd[b])
        return pop[a] if ka<kb else pop[b] if kb<ka else pop[int(rng.choice([a,b]))]
    for _ in range(count):
        if limit is not None and initial+len(novel)>=limit:break
        a,b=parent(),parent()
        if rng.random()<.9:
            coin=int(rng.integers(0,maxmask+1));child=(a&coin)|(b&(~coin&maxmask))
        else:child=a
        child^=sum(1<<i for i,x in enumerate(rng.random(d)) if x<1/d)
        accepted.append(child)
        if child not in novel and child not in visited:novel.add(child)
    return accepted
def run_search(f,store,p,spec,stop):
    d=len(f.c['features']['optional']);space=2**d;fixed=frozen.mandatory(f.c);cost=frozen.burden(space-1,f.c)
    s=store.get('state',{'phase':'GA','restart':0,'generation':0,'initialized':False,'population':[],'rng':None,
       'history':[],'restart_summaries':[],'steps':0,'ga_archive':new_archive(cost),'random_archive':new_archive(cost)})
    s.setdefault('timing',{'fitness_seconds':0.,'ga_proposals':0,'random_new_masks':0,'random_draws':0})
    def evaluate(masks):
        started=time.monotonic();records=f.evaluate(masks,lambda:stop(s))
        s['timing']['fitness_seconds']+=time.monotonic()-started
        return records
    def checkpoint(algorithm=None,restart=0,masks=()):
        s['steps']+=1;store.checkpoint(s,algorithm,restart,masks)
    def finish_restart(reason,partial=None):
        row={'restart':s['restart'],'seed':p['seeds'][s['restart']],'completed_generations':s['generation'],
             'unique_masks':len(store.visited('GA',s['restart'])),'reason':reason,'archive':s['restart_archive']}
        if partial is not None:row['partial_generation_offspring']=partial
        s['restart_summaries'].append(row)
        s.update(restart=s['restart']+1,generation=0,initialized=False,population=[],rng=None,history=[])
    while s['phase'] not in ('REFIT','COMPLETE'):
        if stop(s):raise Paused('Context paused at checkpoint boundary')
        if s['phase']=='GA':
            ri=s['restart']
            if ri==len(p['seeds']):
                store.set('GA',record_choice(f,store,'GA',s['ga_archive'],spec['synthetic']))
                s.update(phase='RANDOM',initialized=False,rng=None);checkpoint();continue
            seed=p['seeds'][ri]+1000000*int(spec['arm']=='moca27')+10000*spec['horizon']+100*spec['outer']
            if not s['initialized']:
                rng=rng_from(None,seed);pop=list(fixed)
                while len(pop)<p['population']:pop.append(int(rng.integers(0,space)))
                if p['unique_limit'] is not None and len(set(pop))>p['unique_limit']:raise IntegrityError('Initial population exceeds restart budget')
                s['timing']['ga_proposals']+=len(pop);records=evaluate(pop);a=update_archive(new_archive(cost),records);update_archive(s['ga_archive'],records)
                s.update(initialized=True,population=pop,generation=0,rng=rng.bit_generator.state,restart_archive=a,
                    history=[{'generation':0,'envelope':archive_envelope(a),'selected':archive_selected(a)['mask']}])
                checkpoint('GA',ri,pop);continue
            visited=store.visited('GA',ri);reason=None
            # Stabilization is tested first at a shared boundary; caps never fabricate stabilization.
            if stagnated(s['history'],p.get('stagnation'),s['generation'],len(visited)):reason='stagnation'
            elif p['unique_limit'] is not None and len(visited)>=p['unique_limit']:reason='unique_mask_limit'
            elif s['generation']>=p['generations']:reason='generation_limit'
            if reason:finish_restart(reason);checkpoint();continue
            rng=rng_from(s['rng']);offspring=children(s['population'],f,rng,d,p['offspring'],visited,p['unique_limit'])
            s['timing']['ga_proposals']+=len(offspring);records=evaluate(offspring);update_archive(s['restart_archive'],records);update_archive(s['ga_archive'],records)
            union=s['population']+offspring;rank,crowd=frozen.ranks_crowding([f.cache[m] for m in union])
            order=sorted(range(len(union)),key=lambda i:(rank[i],-crowd[i],union[i]))
            s['population']=[union[i] for i in order[:p['population']]];s['rng']=rng.bit_generator.state
            if len(offspring)==p['offspring']:
                s['generation']+=1;a=s['restart_archive']
                s['history'].append({'generation':s['generation'],'envelope':archive_envelope(a),'selected':archive_selected(a)['mask']})
                if p.get('stagnation'):s['history']=s['history'][-p['stagnation'].get('consecutive_windows',1)*p['stagnation']['window']-1:]
            checkpoint('GA',ri,offspring)
            if len(offspring)<p['offspring']:finish_restart('unique_mask_limit',len(offspring));checkpoint()
            continue
        if s['phase']=='RANDOM':
            budget=len(store.visited('GA'))
            if not s['initialized']:
                s['timing']['random_new_masks']+=len(set(fixed));records=evaluate(fixed);update_archive(s['random_archive'],records)
                rng=rng_from(None,20271031+1000000*int(spec['arm']=='moca27')+10000*spec['horizon']+100*spec['outer'])
                s.update(initialized=True,rng=rng.bit_generator.state);checkpoint('RANDOM',0,fixed);continue
            seen=store.visited('RANDOM')
            if len(seen)==budget:
                store.set('random',record_choice(f,store,'RANDOM',s['random_archive'],spec['synthetic']))
                store.set('selection_frozen_digest',digest({'GA':store.get('GA'),'random':store.get('random')}))
                s['convergence']=convergence_report(s,p);s.update(phase='REFIT');checkpoint();continue
            if len(seen)>budget:raise IntegrityError('Random logical budget exceeded')
            rng=rng_from(s['rng']);added=[];new=set()
            while len(added)<p['batch_masks'] and len(seen)+len(new)<budget:
                m=int(rng.integers(0,space));s['timing']['random_draws']+=1
                if m not in new and m not in seen:new.add(m);added.append(m)
            s['timing']['random_new_masks']+=len(added);records=evaluate(added);update_archive(s['random_archive'],records)
            s['rng']=rng.bit_generator.state;checkpoint('RANDOM',0,added);continue
        raise IntegrityError('Unknown search phase; exhaustive enumeration is not implemented here')
    return s
