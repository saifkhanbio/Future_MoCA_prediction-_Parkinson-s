from common import *
import concurrent.futures as cf,multiprocessing as mp,time,traceback
FIELDS=['mask','mse','normalized_mse','r2','burden','scores','alpha']
def verify_context(task):
    arm,h,o=task;name=f'{arm}_{h}m_outer{o}';started=time.time()
    base,profile,b,by,cohorts=data();c,sp,tr,te=spec_for(base,by,cohorts,arm,h,o)
    path=R/'restricted'/name;res=read(path/'Result.json');ready=read(path/'Convergence.json')
    assert res['status']=='COMPLETE' and res['context_id']==name and res['synthetic'] is False
    binding={'schema':'joint28_r2_v1','context_digest':digest(sp),'contract_digest':digest(c),
        'profile_digest':digest(profile),'implementation':read(A/'LAUNCHER_LOCK.json')['implementation_identity']}
    assert res['binding']==ready['binding']==binding
    assert res['test_ids']==[r['PATNO'] for r in te] and res['test_components']==[r['component_id'] for r in te]
    assert read(R/'RUN.json')['binding']['contexts'][name]==digest(sp)
    wal=path/'checkpoint.sqlite-wal';assert not wal.exists() or wal.stat().st_size==0
    db=sqlite3.connect((path/'checkpoint.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON')
    assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
    meta={}
    for k,v,checksum in db.execute('SELECT k,v,h FROM meta'):
        assert hashlib.sha256(v.encode()).hexdigest()==checksum
        meta[k]=json.loads(v)
    assert meta['binding']==binding and meta['result']==res and meta['context_status']=='COMPLETE' and meta['state']['phase']=='COMPLETE'
    assert meta['readiness']==ready and meta['state']['restart_summaries']==res['restart_summaries']
    fits={}
    for a,v,checksum in db.execute('SELECT arm,v,h FROM refits'):
        assert hashlib.sha256(v.encode()).hexdigest()==checksum
        fits[a]=json.loads(v)
    assert fits==res['models'] and set(fits)=={'GA_selected','random_selected'}
    n=db.execute('SELECT COUNT(*) FROM fitness').fetchone()[0]
    masks=np.empty(n,np.int64);mse=np.empty(n);costs=np.empty(n,np.int16);counts=np.empty(n,np.int16);alphas=np.empty(n)
    y=np.array(sp['train_y']);inner=np.array(sp['inner']);variance=float(y.var());sizes=np.array([(inner==j).sum() for j in range(3)])
    # Existing snapshot is opened immutably and compared without altering production checkpoints.
    B=Path(read(E/'LATEST_RESUME.json')['process_record']).parents[1]
    oldpath=B/'restricted/Interrupted_Run_Snapshot/restricted'/name/'checkpoint.sqlite'
    olddb=sqlite3.connect(oldpath.as_uri()+'?mode=ro&immutable=1',uri=True) if oldpath.exists() else None
    oldcursor=iter(olddb.execute('SELECT mask,h FROM fitness ORDER BY mask')) if olddb else iter([])
    olditem=next(oldcursor,None);oldcount=0
    if olddb:
        oldstate=json.loads(olddb.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
        assert res['restart_summaries'][:len(oldstate['restart_summaries'])]==oldstate['restart_summaries']
    pos=0;max_pool=0.;max_sse=0.;cursor=db.execute('SELECT mask,v,h FROM fitness ORDER BY mask')
    while True:
        block=cursor.fetchmany(2048)
        if not block:break
        rr=[]
        for mask,packed,checksum in block:
            raw=zlib.decompress(packed) if isinstance(packed,bytes) else packed.encode()
            assert hashlib.sha256(raw).hexdigest()==checksum
            q=json.loads(raw);assert q['mask']==mask and 0<=mask<(1<<len(c['features']['optional']))
            if olditem is not None:
                assert olditem[0]>=mask or olditem[0] in masks[:pos] or False,'Missing resumed cache mask'
                if olditem[0]==mask:
                    assert olditem[1]==checksum;oldcount+=1;olditem=next(oldcursor,None)
            rr.append(q)
        ids=np.array([q['mask'] for q in rr]);grid=np.array([q['lambda_grid_mse'] for q in rr]);fm=np.array([q['all_fold_mse'] for q in rr]);fs=np.array([q['all_fold_sse'] for q in rr])
        assert grid.shape==(len(rr),5) and fm.shape==fs.shape==(len(rr),5,3)
        assert np.isfinite(grid).all() and np.isfinite(fm).all() and np.isfinite(fs).all() and (grid>=0).all() and (fs>=0).all()
        pooled=(fm*sizes).sum(2)/sizes.sum();max_pool=max(max_pool,float(abs(pooled-grid).max()))
        max_sse=max(max_sse,float(abs(fs-fm*sizes).max()));near(pooled,grid);near(fs,fm*sizes)
        near(np.array([q['lambda_grid_r2'] for q in rr]),1-grid/variance)
        eligible=grid<=grid.min(1)[:,None]+1e-12;best=np.max(np.where(eligible,np.arange(5),-1),axis=1)
        losses=grid[np.arange(len(rr)),best];acost,acount=burden_count(ids,c)
        for j,q in enumerate(rr):
            assert q['fitness_schema']=='joint28_r2_v1' and q['fold_sizes']==sizes.tolist()
            assert q['alpha']==c['model']['lambdas'][best[j]] and q['burden']==int(acost[j]) and q['scores']==int(acount[j])
            near([q['mse'],q['normalized_mse'],q['r2'],q['outcome_variance'],q['pooled_sst']],
                 [losses[j],losses[j]/variance,1-losses[j]/variance,variance,variance*len(y)])
            near(q['fold_mse'],fm[j,best[j]])
        end=pos+len(rr);masks[pos:end]=ids;mse[pos:end]=losses;costs[pos:end]=acost;counts[pos:end]=acount;alphas[pos:end]=[q['alpha'] for q in rr];pos=end
    assert pos==n and olditem is None and np.all(masks[1:]>masks[:-1])
    if olddb:assert oldcount==olddb.execute('SELECT COUNT(*) FROM fitness').fetchone()[0];olddb.close()
    def locations(ids):
        ids=np.asarray(ids,np.int64);ix=np.searchsorted(masks,ids)
        assert np.all(ix<n) and np.array_equal(masks[ix],ids)
        return ix
    def summary_record(i):
        return dict(mask=int(masks[i]),mse=float(mse[i]),normalized_mse=float(mse[i]/variance),r2=float(1-mse[i]/variance),
                    burden=int(costs[i]),scores=int(counts[i]),alpha=float(alphas[i]))
    maxcost=int(burden_count([(1<<len(c['features']['optional']))-1],c)[0][0])
    def archive(ids):
        ix=locations(ids);minimum=float(mse[ix].min());eligible=ix[mse[ix]<=minimum+1e-12]
        ties=[]
        for val in sorted(set(mse[eligible].tolist())):
            js=eligible[mse[eligible]==val];j=min(js,key=lambda i:(int(counts[i]),int(costs[i]),int(masks[i])));ties.append(summary_record(j))
        points=[None]*(maxcost+1)
        for bcost in np.unique(costs[ix]):
            js=ix[costs[ix]==bcost];j=min(js,key=lambda i:(float(mse[i]),int(counts[i]),int(masks[i])));points[int(bcost)]=summary_record(j)
        return {'minimum':minimum,'ties':ties,'points':points}
    def check_archive(got,want):
        assert [x['mask'] if x else None for x in got['points']]==[x['mask'] if x else None for x in want['points']]
        assert {x['mask'] for x in got['ties']}=={x['mask'] for x in want['ties']}
        near(got['minimum'],want['minimum'])
        for aa,bb in zip(got['points'],want['points']):
            if aa is not None:
                assert aa.keys()==bb.keys()
                for k in FIELDS:near(aa[k],bb[k])
    def env(arch):
        out=[];best=math.inf
        for q in arch['points']:
            if q:best=min(best,q['mse']/variance)
            out.append(None if math.isinf(best) else best)
        return out
    def chosen(arch):return min(arch['ties'],key=lambda q:(q['scores'],q['burden'],q['mask']))
    chain=hashlib.sha256(b'').hexdigest();restarts=[[] for _ in range(25)];random=[]
    for alg,ri,mask in db.execute('SELECT algorithm,restart,mask FROM visits ORDER BY rowid'):
        chain=hashlib.sha256(chain.encode()+enc([alg,ri,mask])).hexdigest()
        if alg=='GA':assert 0<=ri<25;restarts[ri].append(mask)
        else:assert alg=='RANDOM' and ri==0;random.append(mask)
    assert chain==meta['visits_digest']
    unions={};chains={}
    for alg,mask in db.execute('SELECT algorithm,mask FROM visit_union ORDER BY rowid'):
        unions.setdefault(alg,[]).append(mask)
        chains[alg]=hashlib.sha256(chains.get(alg,hashlib.sha256(b'').hexdigest()).encode()+enc(mask)).hexdigest()
    assert chains==meta['union_digests'] and set(unions)=={'GA','RANDOM'}
    ga=np.unique(np.concatenate([np.array(v,np.int64) for v in restarts]));random=np.array(random,np.int64)
    assert np.array_equal(ga,np.sort(unions['GA'])) and np.array_equal(np.sort(random),np.sort(unions['RANDOM']))
    assert len(ga)==len(random) and np.array_equal(np.union1d(ga,random),masks)
    fixed=set(mandatory(c));rebuilt=[]
    for i,(ids,ss) in enumerate(zip(restarts,res['restart_summaries'])):
        assert len(ids)==len(set(ids))==ss['unique_masks'] and fixed.issubset(ids)
        assert ss['restart']==i and ss['seed']==profile['seeds'][i] and ss['reason']=='stagnation'
        assert 400<=ss['completed_generations']<=10000 and 50000<=ss['unique_masks']<=200000
        a=archive(ids);check_archive(a,ss['archive']);rebuilt.append(a)
    assert len(res['restart_summaries'])==25 and meta['state']['history']==[]
    allga=archive(ga);allrandom=archive(random)
    check_archive(allga,meta['state']['ga_archive']);check_archive(allrandom,meta['state']['random_archive'])
    for key,ids,a,alg in [('GA',ga,allga,'GA'),('random',random,allrandom,'RANDOM')]:
        q=meta[key];assert q==res[key] and q['visited_count']==len(ids) and q['visited_digest']==chains[alg]
        pick=chosen(a);assert pick['mask']==q['choice']['selected']['mask'] and pick['alpha']==q['choice']['selected']['alpha']
        near(q['choice']['threshold'],a['minimum']+1e-12);assert q['envelope']==env(a)
        front=[];previous=math.inf
        for point in a['points']:
            if point and point['mse']<previous:front.append(point['mask'])
            if point:previous=min(previous,point['mse'])
        assert front==[x['mask'] for x in q['pareto']]
    # Replay the seeded random comparator exactly; GA evolutionary trajectories are not replayed.
    seed=20271031+1000000*int(arm=='moca27')+10000*h+100*o
    draws=meta['state']['timing']['random_draws'];rng=np.random.default_rng(seed)
    sampled=rng.integers(0,1<<len(c['features']['optional']),size=draws)
    assert np.array_equal(np.union1d(np.array(list(fixed),np.int64),sampled),np.sort(random))
    assert rng.bit_generator.state==meta['state']['rng']
    main=archive(np.unique(np.concatenate([np.array(v,np.int64) for v in restarts[:20]])))
    confirm=archive(np.unique(np.concatenate([np.array(v,np.int64) for v in restarts[20:]])))
    def gap(ar):
        x,z=env(ar),env(allga);assert [v is None for v in x]==[v is None for v in z]
        return max(abs(a-b) for a,b in zip(x,z) if a is not None)
    rep=sum(chosen(a)['normalized_mse']<=chosen(allga)['normalized_mse']+1e-4 for a in rebuilt[20:])
    gain=chosen(allga)['normalized_mse']-chosen(allrandom)['normalized_mse']
    conv=res['convergence'];assert conv==ready['convergence']==meta['state']['convergence']
    assert gap(main)<=1e-4 and gap(confirm)<=1e-4 and rep>=3 and gain<=1e-4
    near([gap(main),gap(confirm),gain],[conv['main_envelope_max_gap_R2'],conv['confirmation_envelope_max_gap_R2'],conv['random_gain_R2']])
    assert conv['passed'] and conv['enforced'] and all(conv['checks'].values()) and conv['confirmation_replicates']==rep and conv['global_optimum_proven'] is False
    gate=read(R/'restricted/Convergence_Gate.json');sd=digest({'GA':meta['GA'],'random':meta['random']})
    assert sd==meta['selection_frozen_digest']==ready['selection_digest']==gate['selection_digests'][name]
    assert gate['status']=='PASS' and gate['binding']==read(R/'RUN.json')['binding'] and gate['arm_contracts'][arm]==digest(c)
    events=[dict(zip(['id','kind','planned','status','attempted','completed','detail'],r)) for r in db.execute('SELECT * FROM events ORDER BY id')]
    ledger=res['ledger'];assert ledger['events']==events
    for x in events:
        assert x['status'] in ['COMPLETE','INTERRUPTED_UNKNOWN']
        if x['status']=='COMPLETE':assert x['planned']==x['attempted']==x['completed']
    inn=sum(x['completed'] or 0 for x in events if x['status']=='COMPLETE' and x['kind'].endswith('_fitness'))
    out=sum(x['completed'] or 0 for x in events if x['status']=='COMPLETE' and x['kind']=='outer_refit')
    uncertain=sum(x['planned'] for x in events if x['status']=='INTERRUPTED_UNKNOWN')
    assert inn==15*n and out==2 and res['logical_fits']==inn+out==ledger['known_validated_solves']
    assert ledger['unique_masks']==n and ledger['refits']==2 and ledger['unknown_interrupted_attempts_upper']==uncertain
    assert ledger['physical_attempts_upper']==ledger['physical_attempts_lower']+uncertain
    ready_events=ready['ledger']['events'];assert all(x['kind']!='outer_refit' for x in ready_events)
    assert events[:len(ready_events)]==ready_events and all(x['kind']=='outer_refit' for x in events[len(ready_events):])
    X,T,labels,prep=full_design(sp['train_rows'],sp['test_rows'],c);prediction_error=0.;residual_error=0.;inner_grid_error=0.
    reconstructed={};models=[]
    for label,model in fits.items():
        key='GA' if label=='GA_selected' else 'random';mask=model['mask'];ix=indices(mask,c,labels)
        assert model['features']==features(mask,c) and model['columns']==[labels[i] for i in ix]
        assert model['burden']==int(burden_count([mask],c)[0][0])
        assert model['preprocessing']['categories']==prep['categories']
        for field,vals in prep['numeric'].items():
            assert model['preprocessing']['numeric'][field].keys()==vals.keys()
            for k,val in vals.items():near(val,model['preprocessing']['numeric'][field][k],1e-12)
        beta=np.array(model['coef']);intercept=model['intercept'];xx=X[:,ix];tt=T[:,ix];p=tt@beta+intercept
        pe=float(np.max(abs(p-np.array(model['predictions']))));prediction_error=max(prediction_error,pe)
        near(p,model['predictions'],1e-9);near(p,res['predictions'][label],1e-9)
        z=xx-xx.mean(0);yc=y-y.mean();residual=z.T@(z@beta-yc)+len(y)*model['alpha']*beta
        re=float(np.max(abs(residual))) if len(residual) else 0.;residual_error=max(residual_error,re);assert re<1e-6
        near(intercept,y.mean()-xx.mean(0)@beta,1e-9)
        near([model['inner_mse'],model['inner_r2'],model['alpha']],[meta[key]['choice']['selected']['mse'],meta[key]['choice']['selected']['r2'],meta[key]['choice']['selected']['alpha']])
        # Independent SVD reconstruction of selected inner grids, not a new search or saved model.
        grid=np.zeros((5,3))
        for k in range(3):
            itr=np.flatnonzero(inner!=k);iva=np.flatnonzero(inner==k)
            xi,vi,lab,_=full_design([sp['train_rows'][j] for j in itr],[sp['train_rows'][j] for j in iva],c)
            kk=indices(mask,c,lab);xxi=xi[:,kk];vvi=vi[:,kk];xm=xxi.mean(0);ym=y[itr].mean()
            u,singular,vt=np.linalg.svd(xxi-xm,full_matrices=False)
            for j,alpha in enumerate(c['model']['lambdas']):
                coeff=vt.T@((singular/(singular**2+len(itr)*alpha))*(u.T@(y[itr]-ym)))
                pp=(vvi-xm)@coeff+ym;grid[j,k]=np.mean((pp-y[iva])**2)
        expected=np.array(meta[key]['choice']['selected']['all_fold_mse'])
        inner_grid_error=max(inner_grid_error,float(abs(grid-expected).max()));near(grid,expected,1e-9)
        reconstructed[label]=p.tolist()
        models.append({'context':name,'arm':label,'mask':mask,'features':model['features'],'alpha':model['alpha'],
                       'feature_count':len(model['features']),'instruments':model['burden']})
    db.close()
    write(V/'restricted'/f'{name}_Reconstructed.json',{'test_ids':res['test_ids'],'component_ids':res['test_components'],'predictions':reconstructed})
    report={'context':name,'status':'PASS','cache_records':n,'pre_resume_cache_records_preserved':oldcount,'logical_fits':res['logical_fits'],
      'unknown_interrupted_attempts_upper':uncertain,'GA_distinct_masks':len(ga),'random_distinct_masks':len(random),'random_draws_replayed':draws,
      'restarts_checked':25,'restart_generations_range':[min(x['completed_generations'] for x in res['restart_summaries']),max(x['completed_generations'] for x in res['restart_summaries'])],
      'restart_unique_masks_range':[min(x['unique_masks'] for x in res['restart_summaries']),max(x['unique_masks'] for x in res['restart_summaries'])],
      'max_pooled_fitness_error':max_pool,'max_fold_sse_error':max_sse,'max_prediction_error':prediction_error,'max_ridge_equation_residual':residual_error,
      'max_selected_inner_grid_error':inner_grid_error,'diagnostic_inner_linear_algebra_checks':30,'new_production_fits':0,'outer_models_verified':2,
      'reconstructed_prediction_count':sum(len(p) for p in reconstructed.values()),'models':models,'convergence_summary_and_envelopes':'PASS',
      'historical_GA_generations_replayed':False,'history_limitation':'Completed restarts clear their per-generation history; the 400-transition plateau is supported by locked runtime code and checksummed stop summaries, not independently replayed here.',
      'elapsed_seconds':time.time()-started}
    write(V/'evidence'/f'Context_{name}.json',report)
    return {k:report[k] for k in ['context','status','cache_records','elapsed_seconds']}
def main():
    tasks=[(a,h,o) for a in ['unrestricted28','moca27'] for h in [12,24,36] for o in range(5)]
    done=[];failures=[]
    with cf.ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(verify_context,t):t for t in tasks}
        for f in cf.as_completed(fs):
            try:r=f.result();done.append(r);print(json.dumps(r),flush=True)
            except BaseException as ex:
                failures.append({'context':str(fs[f]),'error':str(ex),'traceback':traceback.format_exc()})
                print(json.dumps({'context':str(fs[f]),'status':'FAILED','error':str(ex)}),flush=True)
    write(V/'evidence/Context_Verification_Summary.json',{'status':'PASS' if not failures and len(done)==30 else 'FAIL','passed':done,'failures':failures})
    if failures:raise RuntimeError('Context verification failed; see saved evidence')
if __name__=='__main__':main()
