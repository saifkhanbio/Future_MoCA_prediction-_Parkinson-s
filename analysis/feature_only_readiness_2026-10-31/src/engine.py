"""Fixed feature-block engine. Worker tasks never contain held-out outcomes."""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
import json,hashlib,math
import numpy as np
from numerics import prepare,transform,ridge,predict,metrics,interval,COUNTERS,numeric
class IntegrityError(RuntimeError):pass
def require(ok,message):
    if not ok:raise IntegrityError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def arm_config(contract,arm):
    require(arm in contract['arms'],'Unknown new arm')
    a=contract['arms'][arm]
    require(len(set(a['numeric']))==len(a['numeric']),'Duplicate numeric feature')
    return {'numeric':a['numeric'],'categories':a['categories']}
def project(raw,contract,arm):
    a=arm_config(contract,arm)
    return {k:raw[k] for k in ['PATNO','component_id']+a['numeric']+list(a['categories'])}
def validate_task(s,c):
    expected={'context_id','horizon','outer','arm','train_rows','train_y','inner','test_rows','synthetic'}
    require(set(s)==expected,'Unexpected worker fields; outer outcomes forbidden')
    require(s['arm'] in c['arms'] and s['horizon'] in [12,24,36] and s['outer'] in range(5),'Invalid task identity')
    require(s['context_id']==f"{s['horizon']}m_outer{s['outer']}_{s['arm']}",'Task identity mismatch')
    a=arm_config(c,s['arm']);columns=set(a['numeric'])|set(a['categories'])|{'PATNO','component_id'}
    tr,te=s['train_rows'],s['test_rows'];y=np.asarray(s['train_y'],dtype=float);inner=s['inner']
    require(len(tr)==len(y)==len(inner) and len(te)>0 and len(tr)>3,'Empty or inconsistent task')
    require(np.isfinite(y).all() and np.var(y)>0,'Invalid training outcome variance')
    require(set(inner)=={0,1,2} and all(type(k) is int for k in inner),'Invalid inner folds')
    for rows in [tr,te]:
        require(len({r['PATNO'] for r in rows})==len(rows) and len({r['component_id'] for r in rows})==len(rows),'Duplicate identities')
        for r in rows:
            require(set(r)==columns,'Predictor allowlist violation')
            if s['synthetic']:
                require(str(r['PATNO']).startswith('SYNTH_') and str(r['component_id']).startswith('SYNTH_'),'Real record in synthetic fixture')
            for n in a['numeric']:numeric(r[n])
            for n,levels in a['categories'].items():require(r[n] in levels,'Invalid form category')
    require(not ({r['PATNO'] for r in tr}&{r['PATNO'] for r in te}),'Outer participant overlap')
    require(not ({r['component_id'] for r in tr}&{r['component_id'] for r in te}),'Outer component overlap')
    for k in [0,1,2]:
        require(any(x==k for x in inner) and any(x!=k for x in inner),'Empty inner partition')
    return a
def select_alpha(losses,lambdas):
    require(len(losses)==len(lambdas) and np.isfinite(losses).all(),'Invalid penalty loss grid')
    best=min(losses)
    return max((i for i,v in enumerate(losses) if v<=best+1e-12),key=lambda i:lambdas[i])
def fit_task(s,c):
    a=validate_task(s,c);before=dict(COUNTERS);tr=s['train_rows'];te=s['test_rows'];y=np.asarray(s['train_y'],dtype=float);inner=np.asarray(s['inner'])
    parts=[];audit=[]
    for k in range(3):
        it=np.flatnonzero(inner!=k);iv=np.flatnonzero(inner==k)
        r=[tr[i] for i in it];v=[tr[i] for i in iv];state=prepare(r,a);X,lab=transform(r,state);V,lab2=transform(v,state)
        require(lab==lab2 and len(lab)==c['arms'][s['arm']]['design_columns'],'Design binding failed')
        parts.append((X,V,y[it],y[iv]))
        audit.append({'inner':k,'train_ids':[tr[i]['PATNO'] for i in it],'validation_ids':[tr[i]['PATNO'] for i in iv],'preprocessing':state})
    losses=[];sse_grid=[]
    for alpha in c['model']['lambdas']:
        ss=[]
        for X,V,u,v in parts:ss.append(float(np.sum((predict(V,ridge(X,u,alpha))-v)**2)))
        sse_grid.append(ss);losses.append(sum(ss)/len(y))
    idx=select_alpha(losses,c['model']['lambdas']);alpha=c['model']['lambdas'][idx]
    state=prepare(tr,a);X,labels=transform(tr,state);V,labels2=transform(te,state)
    require(labels==labels2,'Outer design mismatch')
    coef,intercept=ridge(X,y,alpha);p=predict(V,(coef,intercept))
    fits=COUNTERS['ridge_completed']-before['ridge_completed']
    require(fits==16 and COUNTERS['ridge_attempts']-before['ridge_attempts']==16,'Fit ledger mismatch')
    forbidden=set(c['mandatory_exclusion'][s['arm']])
    require(not any(l.split('__')[0].split('==')[0] in forbidden for l in labels),'Excluded core predictor reintroduced')
    return {'context_id':s['context_id'],'horizon':s['horizon'],'outer':s['outer'],'arm':s['arm'],'synthetic':s['synthetic'],
    'test_ids':[r['PATNO'] for r in te],'test_components':[r['component_id'] for r in te],'predictions':p.tolist(),
    'alpha':alpha,'lambda_grid_mse':losses,'lambda_grid_fold_sse':sse_grid,'inner_fold_sizes':[len(x[3]) for x in parts],
    'model':{'columns':labels,'coef':coef.tolist(),'intercept':intercept,'preprocessing':state},
    'inner_audit':audit,'task_sha256':digest(s),'logical_fits':fits}
def specs(c,by,cohorts,synthetic):
    out=[]
    for h,cohort in sorted(cohorts.items()):
        for outer in range(5):
            train=[r for r in cohort if int(r['outer_fold'])!=outer];test=[r for r in cohort if int(r['outer_fold'])==outer]
            for arm in c['arms']:
                s={'context_id':f'{h}m_outer{outer}_{arm}','horizon':h,'outer':outer,'arm':arm,
                'train_rows':[project(by[r['PATNO']],c,arm) for r in train],'test_rows':[project(by[r['PATNO']],c,arm) for r in test],
                'train_y':[float(r['y']) for r in train],'inner':[int(r[f'inner_for_outer_{outer}']) for r in train],'synthetic':synthetic}
                validate_task(s,c);out.append(s)
    return out
def assemble(c,cohorts,references,tasks,results):
    expected={s['context_id']:s for s in tasks}
    require(len(results)==len(expected) and len({r['context_id'] for r in results})==len(expected),'Missing or duplicate result contexts')
    values={}
    for r in results:
        require(r['context_id'] in expected,'Unexpected result context')
        s=expected[r['context_id']]
        require(r['task_sha256']==digest(s) and r['logical_fits']==16,'Altered task binding or fit count')
        for k in ['horizon','outer','arm','synthetic']:require(r[k]==s[k],'Result context mismatch')
        ids=[x['PATNO'] for x in s['test_rows']];groups=[x['component_id'] for x in s['test_rows']]
        require(r['test_ids']==ids and r['test_components']==groups,'Prediction identity/order mismatch')
        require(len(r['predictions'])==len(ids) and np.isfinite(r['predictions']).all(),'Invalid predictions')
        for pid,p in zip(ids,r['predictions']):
            key=(r['horizon'],pid,r['arm']);require(key not in values,'Duplicate OOF prediction');values[key]=float(p)
    oof={};metric_rows=[];contrast_rows=[]
    for h,rows in sorted(cohorts.items()):
        ref={r['PATNO']:r for r in references[h]};require(len(ref)==len(rows),'Reference coverage mismatch')
        arr=[]
        for row in rows:
            pid=row['PATNO'];rr=ref[pid]
            require(rr['component_id']==row['component_id'] and int(rr['outer_fold'])==int(row['outer_fold']) and float(rr['y'])==float(row['y']),'Reference identity/outcome mismatch')
            r={k:row[k] for k in ['PATNO','component_id']};r.update(y=float(row['y']),outer_fold=int(row['outer_fold']))
            for arm in c['arms']:r[arm]=values[(h,pid,arm)]
            for arm in ['full']+c['context_only_reference_arms']:r[arm]=float(rr[arm])
            arr.append(r)
        oof[h]=arr;y=np.array([r['y'] for r in arr])
        for arm in list(c['arms'])+['full']+c['context_only_reference_arms']:
            p=np.array([r[arm] for r in arr]);metric_rows.append({'horizon':h,'arm':arm,**metrics(y,p)})
        for pair in c['contrasts']:
            a,b=pair['first'],pair['reference'];p=np.array([r[a] for r in arr]);q=np.array([r[b] for r in arr])
            iv=interval(y,p,q,seed=20261031+h,replicates=c['inference']['bootstrap_replicates'])
            contrast_rows.append({'horizon':h,'n':len(y),'first':a,'reference':b,'role':'primary_exploratory' if h==36 and a=='optional18_only' and b=='full' else 'secondary_unadjusted',**iv})
    require(sum(r['logical_fits'] for r in results)==480,'Total fit budget mismatch')
    return oof,metric_rows,contrast_rows
def execute_dataset(c,by,cohorts,references,out,workers,synthetic,record):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp,traceback,time
    out=Path(out);require(not out.exists(),'Output already exists')
    require(type(workers) is int and 1<=workers<=12,'Worker count out of bounds')
    tasks=specs(c,by,cohorts,synthetic)
    require(len(tasks)==30,'Expected exactly30 new arm/fold/horizon tasks')
    out.mkdir(mode=0o700);(out/'restricted').mkdir(mode=0o700);(out/'results').mkdir()
    start=time.monotonic();write_json(out/'RUN.json',{'status':'RUNNING','synthetic':synthetic,'workers':workers,'record':record})
    results=[]
    try:
        if workers==1:
            for s in tasks:
                r=fit_task(s,c);results.append(r);write_json(out/'restricted'/f"{r['context_id']}.json",r)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
                fs=[ex.submit(fit_task,s,c) for s in tasks]
                for future in as_completed(fs):
                    r=future.result();results.append(r);write_json(out/'restricted'/f"{r['context_id']}.json",r)
        results.sort(key=lambda r:(r['horizon'],r['outer'],r['arm']))
        oof,mm,cc=assemble(c,cohorts,references,tasks,results)
        for h,rows in oof.items():write_json(out/'restricted'/f'OOF_{h}m.json',rows)
        write_json(out/'results/Metrics.json',mm);write_json(out/'results/Contrasts.json',cc)
        write_json(out/'results/Primary_Exploratory_Contrast.json',next(x for x in cc if x['role']=='primary_exploratory'))
        write_json(out/'results/Tuning.json',[{k:r[k] for k in ['context_id','horizon','outer','arm','alpha','lambda_grid_mse','inner_fold_sizes','logical_fits']} for r in results])
        receipt={'status':'COMPLETE_PENDING_INDEPENDENT_VERIFICATION','synthetic':synthetic,'new_ridge_fits':480,
                 'inner_fits':450,'outer_refits':30,'reference_refits':0,'task_count':30,'paid_queries':0,'seconds':time.monotonic()-start,'workers':workers}
        write_json(out/'RUN_COMPLETION.json',receipt)
        manifest={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
        write_json(out/'Artifact_Manifest.json',manifest);return receipt
    except BaseException as e:
        write_json(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_contexts':len(results),'automatic_retry':False})
        raise
