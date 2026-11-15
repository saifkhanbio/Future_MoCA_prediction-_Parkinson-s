"""Default is preflight. The study mode is reserved for a separately authorized run."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from paired_metrics import summarize
os.umask(0o077)
D=Path(__file__).resolve().parent

def require(condition,message):
    if not condition:raise ValueError(message)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()

def verify_lock():
    seal=json.loads((D/'LOCK.json').read_text())
    require(all(digest(D/k)==v for k,v in seal['files'].items()),'Package digest mismatch')
    sources=json.loads((D/'source_hashes.json').read_text())
    require(all(digest(k)==v for k,v in sources.items()),'Frozen source digest mismatch')

def validate(plan,x,m,y,s,reserve):
    require(x.index.is_unique and s.index.is_unique and m.index.is_unique and y.index.is_unique,'Duplicate participant or endpoint')
    require(set(x.index)==set(s.index)==set(m.index.get_level_values('component')),'Master roster mismatch')
    require(set(x.index).isdisjoint(reserve),'Stage A component in development')
    require(set(m.index)==set(y.index) and set(m.index.get_level_values('horizon'))==set(plan['horizons']),'Endpoint membership mismatch')
    require(m.endpoint_row.equals(y.endpoint_row.reindex(m.index)),'Wrong bound endpoint row')
    require(not m.endpoint_row.duplicated().any(),'Endpoint row reused')
    targets=y.y.to_numpy()
    require(np.all(np.isfinite(targets)&(targets>=0)&(targets<=30)&(targets==np.floor(targets))),'Invalid MoCA endpoint')
    for panel in plan['panels'].values():
        data=x[panel['numeric']].to_numpy(dtype=float)
        require(not np.isinf(data).any(),'Infinite predictor')
        for k,levels in panel['categories'].items():require(x[k].isin(levels).all(),'Unsupported category')
    compact=plan['panels']['compact4']
    require(x[compact['numeric']].notna().all().all() and x[list(compact['categories'])].ne('unknown').all().all(),'Compact fields unavailable')
    require(x.baseline_moca.between(0,30).all() and x.baseline_moca.mod(1).eq(0).all(),'Invalid baseline MoCA')
    full=plan['panels']['full28']; complete=x[full['numeric']].notna().all(axis=1)&x[list(full['categories'])].ne('unknown').all(axis=1)
    require(np.array_equal(m.complete28.to_numpy(),complete.reindex(m.index.get_level_values('component')).astype(int).to_numpy()),'Complete panel membership differs')
    require(np.array_equal(m.historical,(m[['index_month','endpoint_month']].max(axis=1)<2023*12+1).astype(int)),'Historical cutoff differs')
    require(np.all(np.abs(m.endpoint_month-m.index_month-m.index.get_level_values('horizon'))<=3),'Visit outside horizon')
    require(s.outer.isin(range(5)).all(),'Unexpected outer fold')
    folds=[]
    for j in range(5):
        col=s['inner_for_outer_'+str(j)]
        require(col[s.outer.eq(j)].eq(-1).all() and col[s.outer.ne(j)].isin(range(3)).all(),'Inner/outer leakage or invalid fold')
    for (v,h,ids) in blocks(plan,m):
        require(len(ids)==plan['views'][v]['n'][plan['horizons'].index(h)],'Unexpected view size')
        sub=s.loc[ids]
        for j in range(5):
            train=sub[sub.outer.ne(j)]
            counts=train['inner_for_outer_'+str(j)].value_counts()
            require(len(train)<len(sub) and set(counts.index)=={0,1,2},'Empty validation fold')
            folds.append(dict(view=v,horizon=h,outer=j,test=len(sub)-len(train),train=len(train),inner0=int(counts[0]),inner1=int(counts[1]),inner2=int(counts[2])))
    return folds

def blocks(plan,m):
    for v in plan['views']:
        for h in plan['horizons']:
            z=m.xs(h,level='horizon')
            if v!='expanded':z=z[z[v].eq(1)]
            yield v,h,sorted(z.index)

def design(x,train,test,panel):
    nums=panel['numeric'];a=x.loc[train,nums].to_numpy(dtype=float);b=x.loc[test,nums].to_numpy(dtype=float)
    am=np.isnan(a);bm=np.isnan(b)
    med=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0 for z in a.T])
    a=np.where(am,med,a);b=np.where(bm,med,b)
    mean=a.mean(0);sd=a.std(0);sd[sd<=1e-12]=1
    aa=[(a-mean)/sd,am];bb=[(b-mean)/sd,bm]
    for k,levels in panel['categories'].items():
        lv=np.array(levels)
        aa.append(x.loc[train,k].to_numpy()[:,None]==lv);bb.append(x.loc[test,k].to_numpy()[:,None]==lv)
    return np.column_stack(aa),np.column_stack(bb)

def ridge(a,b,y,penalty):
    mean=a.mean(0);d=a-mean
    coef=np.linalg.solve(d.T@d+np.eye(a.shape[1])*len(a)*penalty,d.T@(y-y.mean()))
    return y.mean()+(b-mean)@coef

def predict_fold(x,target,split,j,panel,grid):
    training=split.index[split.outer.ne(j)];test=split.index[split.outer.eq(j)]
    assignments=split.loc[training,'inner_for_outer_'+str(j)]
    errors=np.zeros(len(grid))
    for k in range(3):
        ti=training[assignments.ne(k)];vi=training[assignments.eq(k)]
        a,b=design(x,ti,vi,panel)
        for i,penalty in enumerate(grid):
            errors[i]+=np.abs(ridge(a,b,target.loc[ti].to_numpy(),penalty)-target.loc[vi].to_numpy()).sum()
    errors/=len(training)
    selected=max(z for z,e in zip(grid,errors) if e<=min(errors)+1e-12)
    a,b=design(x,training,test,panel)
    return test,ridge(a,b,target.loc[training].to_numpy(),selected),selected,errors.tolist()

def run(plan,x,m,y,s,reserve):
    validate(plan,x,m,y,s,reserve)
    aggregates={};out=[];tuning=[];fits=0
    for view,h,ids in blocks(plan,m):
        sp=s.loc[ids];target=y.xs(h,level='horizon').loc[ids,'y'];pred={}
        for arm in plan['views'][view]['models']:
            if arm=='persistence':
                pred[arm]=x.loc[ids,'baseline_moca'].to_numpy();continue
            oof=pd.Series(np.nan,index=ids)
            for j in range(5):
                ti,pr,pen,loss=predict_fold(x,target,sp,j,plan['panels'][arm],plan['penalties'])
                oof.loc[ti]=pr
                tuning.append(dict(view=view,horizon=h,arm=arm,outer=j,penalty=pen,inner_MAE=loss))
                fits+=3*len(plan['penalties'])+1
            require(np.isfinite(oof).all(),'Incomplete OOF predictions')
            pred[arm]=oof.to_numpy()
        for i,p in enumerate(ids):
            out.append(dict(component=p,view=view,horizon=h,outer=int(sp.at[p,'outer']),observed=float(target[p]),**{a:float(v[i]) for a,v in pred.items()}))
        seed=plan['seed_base']+100*(1+list(plan['views']).index(view))+h
        aggregates[f'{view}/{h}']=summarize(target.to_numpy(),pred,plan['bootstrap'],seed)
    require(fits==plan['study_fits_planned'],'Fit schedule mismatch')
    return out,aggregates,tuning

def inputs():
    r=D/'restricted'
    def read(n):return pd.read_csv(r/(n+'.csv'),dtype={'component':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str,'jlo_form':str})
    return (json.loads((D/'protocol.json').read_text()),read('predictors').set_index('component'),read('membership').set_index(['component','horizon']),read('endpoints').set_index(['component','horizon']),read('grouped_splits').set_index('component'),set(read('stage_A_reserved').component))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','synthetic','run-study'],default='preflight');opt=p.parse_args()
    verify_lock()
    if opt.mode=='synthetic':
        from validate_synthetic import exercise
        print(json.dumps(exercise()));raise SystemExit
    args=inputs()
    if opt.mode=='preflight':
        print(json.dumps({'status':'PASS','fold_cells':len(validate(*args)),'study_fits':0}));raise SystemExit
    # One atomic publication. Concurrent or repeated invocations cannot overwrite a completed result.
    root=D.parent/'Stage_C_results_v1'
    if root.exists():raise SystemExit('Result directory exists')
    with tempfile.TemporaryDirectory(dir=D.parent,prefix='.stage_c_run_') as tmp:
        stage=Path(tmp)/'deliverables';stage.mkdir();rr=stage/'restricted';rr.mkdir()
        out,stats,tune=run(*args)
        pd.DataFrame(out).to_csv(rr/'oof_predictions.csv',index=False)
        (stage/'metrics.json').write_text(json.dumps(stats,allow_nan=False)+'\n')
        (stage/'tuning.json').write_text(json.dumps(tune,allow_nan=False)+'\n')
        verify_lock()
        stage.rename(root)
