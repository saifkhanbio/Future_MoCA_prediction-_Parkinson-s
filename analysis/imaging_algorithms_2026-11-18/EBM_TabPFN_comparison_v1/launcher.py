"""Preparation release: real study execution is deliberately disabled."""
import os
from pathlib import Path
D=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_HOME=str(D/'cache/hf'),TABPFN_MODEL_CACHE_DIR=str(D/'models'),TABPFN_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',SCARF_NO_ANALYTICS='true')
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
sys.path.append(str(D/'source_snapshot'))
import importlib.util
spec=importlib.util.spec_from_file_location('source',D/'source_snapshot/launcher.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
from paired_metrics import summarize
P=json.loads((D/'Protocol.json').read_text())

def verify():
    source.verify_lock()
    if (D/'RuntimeVersions.json').exists():
        import importlib.metadata as metadata
        for name,version in json.loads((D/'RuntimeVersions.json').read_text()).items():
            if metadata.version(name)!=version:raise ValueError('Runtime version changed: '+name)
    if (D/'LOCK.json').exists():
        for f,h in json.loads((D/'LOCK.json').read_text())['files'].items():
            if source.digest(D/f)!=h:raise ValueError('Locked file changed: '+f)

def estimate(algorithm,config,a,b,y,seed):
    if algorithm=='ridge':return source.ridge(a,b,y,config)
    if algorithm=='EBM':
        from interpret.glassbox import ExplainableBoostingRegressor
        model=ExplainableBoostingRegressor(**P['EBM_fixed'],**config,random_state=seed)
    elif algorithm=='TabPFN':
        from tabpfn import TabPFNRegressor
        path=D/P['checkpoint']
        if not path.is_file():raise RuntimeError('Local locked checkpoint missing; no automatic download permitted')
        model=TabPFNRegressor(**P['TabPFN_fixed'],**config,model_path=path,random_state=seed)
    else:raise ValueError('Unknown model')
    model.fit(a,y)
    prediction=np.asarray(model.predict(b),dtype=float)
    if prediction.shape!=(len(b),) or not np.isfinite(prediction).all():raise ValueError('Invalid predictions')
    return prediction

def fold(x,target,split,j,panel,algorithm,grid,seed):
    train=split.index[split.outer.ne(j)];test=split.index[split.outer.eq(j)]
    inner=split.loc[train,'inner_for_outer_'+str(j)];loss=np.zeros(len(grid))
    for k in range(3):
        ti=train[inner.ne(k)];vi=train[inner.eq(k)]
        a,b=source.design(x,ti,vi,panel)
        for ci,c in enumerate(grid):
            pr=estimate(algorithm,c,a,b,target.loc[ti].to_numpy(),seed+k)
            loss[ci]+=np.abs(pr-target.loc[vi].to_numpy()).sum()
    loss/=len(train);selected=int(np.flatnonzero(loss<=loss.min()+1e-12)[0])
    a,b=source.design(x,train,test,panel)
    return test,estimate(algorithm,grid[selected],a,b,target.loc[train].to_numpy(),seed+9),selected,loss.tolist()

def evaluate(plan,x,m,y,s,reserve):
    """Shared numerical engine; CLI only invokes on generated synthetic inputs."""
    rows=[];tune=[];metrics={};fits=0
    for view,vp in plan['views'].items():
        for h in plan['horizons']:
            z=m.xs(h,level='horizon');z=z if view=='expanded' else z[z.complete28.eq(1)]
            ids=sorted(z.index);sp=s.loc[ids];target=y.xs(h,level='horizon').loc[ids,'y'];pred={'persistence':x.loc[ids,'baseline_moca'].to_numpy()}
            if set(ids)&reserve:raise ValueError('Reserved participant included')
            for panel in vp['panels']:
                for alg in plan['algorithms']:
                    arm=panel+'__'+alg;oof=pd.Series(np.nan,index=ids)
                    for j in range(5):
                        ix,pr,selected,loss=fold(x,target,sp,j,plan['panels'][panel],alg,plan['grids'][alg],plan['seed_base']+1000*h+10*j)
                        oof.loc[ix]=pr;fits+=3*len(plan['grids'][alg])+1
                        tune.append(dict(view=view,horizon=h,arm=arm,outer=j,selected=selected,inner_MAE=loss))
                    if not np.isfinite(oof).all():raise ValueError('Missing outer prediction')
                    pred[arm]=oof.to_numpy()
            for i,person in enumerate(ids):rows.append(dict(component=person,view=view,horizon=h,outer=int(sp.at[person,'outer']),observed=float(target[person]),**{a:float(v[i]) for a,v in pred.items()}))
            metrics[view+'/'+str(h)]=summarize(target.to_numpy(),pred,plan['bootstrap'],plan['seed_base']+h+100*(1+list(plan['views']).index(view)))
            if view=='complete28' and h==12:
                rng=np.random.default_rng(plan['seed_base']+112);draw=rng.integers(0,len(ids),size=(plan['bootstrap'],len(ids)))
                base=np.abs(pred['compact4__ridge']-target.to_numpy());adjusted={}
                for alg in ['EBM','TabPFN']:
                    diff=np.abs(pred['compact4__'+alg]-target.to_numpy())-base
                    adjusted[alg]={'delta_MAE':float(diff.mean()),'interval_97_5':np.quantile(diff[draw].mean(1),[.0125,.9875]).tolist()}
                metrics['co_primary_adjusted']=adjusted
    return rows,metrics,tune,fits

def preflight():
    verify();args=source.inputs();folds=source.validate(*args)
    assert P['panels']==args[0]['panels']
    for view,v in P['views'].items():assert v['n']==args[0]['views'][view]['n']
    return {'status':'PASS','study_fits':0,'inherited_fold_cells':len(folds),'matched_counts':{k:v['n'] for k,v in P['views'].items()},'scheduled_fit_calls':P['study_fits_planned'],'checkpoint_present':(D/P['checkpoint']).is_file()}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','synthetic','run-study'],default='preflight');opt=ap.parse_args()
    if opt.mode=='run-study':raise SystemExit('BLOCKED: study training is not authorized in this preparation release')
    if opt.mode=='preflight':print(json.dumps(preflight(),indent=2))
    else:
        verify()
        from validate_synthetic import exercise
        print(json.dumps(exercise(),indent=2))
