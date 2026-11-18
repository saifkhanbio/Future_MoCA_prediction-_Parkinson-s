"""Independent preprocessing, SVD ridge replay, tuning and metric verification."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import json,hashlib,itertools,sys,time
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
E=Path(__file__).resolve().parent;D=E.parent;R=E/'results';V=E/'verification';P=json.loads((D/'Protocol.json').read_text())
V.mkdir(exist_ok=True)
def load():
    path=D/'source_snapshot/restricted'
    def read(n):return pd.read_csv(path/(n+'.csv'),dtype={'component':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str,'jlo_form':str})
    return read('predictors').set_index('component'),read('membership').set_index(['component','horizon']),read('endpoints').set_index(['component','horizon']),read('grouped_splits').set_index('component'),set(read('stage_A_reserved').component)
def design(x,train,test,panel):
    names=panel['numeric'];aa=x.loc[train,names].to_numpy(float);bb=x.loc[test,names].to_numpy(float);ma=np.isnan(aa);mb=np.isnan(bb)
    med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in aa.T])
    aa=np.where(ma,med,aa);bb=np.where(mb,med,bb);mean=np.mean(aa,axis=0);sd=np.std(aa,axis=0);sd=np.where(sd<=1e-12,1.,sd)
    tr=[(aa-mean)/sd,ma.astype(float)];te=[(bb-mean)/sd,mb.astype(float)]
    for key,levels in panel['categories'].items():
        tr.append(np.column_stack([(x.loc[train,key]==lv).to_numpy() for lv in levels]));te.append(np.column_stack([(x.loc[test,key]==lv).to_numpy() for lv in levels]))
    return np.column_stack(tr),np.column_stack(te)
def svd(a,b,y,pen):
    mu=np.mean(a,axis=0);u,s,vt=np.linalg.svd(a-mu,full_matrices=False)
    coef=vt.T@((s/(s*s+len(a)*pen))*(u.T@(y-y.mean())))
    return y.mean()+(b-mu)@coef

def audit_checkpoint(file):
    x,m,y,s,reserve=load();z=json.loads(Path(file).read_text());t=z['task'];v,h,p,alg,j=[t[k] for k in ['view','horizon','panel','algorithm','outer']]
    eligible=m.xs(h,level='horizon');eligible=eligible if v=='expanded' else eligible[eligible.complete28==1];ids=sorted(eligible.index);assert not set(ids)&reserve
    sp=s.loc[ids];train=sp.index[sp.outer!=j];test=sp.index[sp.outer==j];assert list(test)==z['ids']
    target=y.xs(h,level='horizon')['y'];grid=P['grids'][alg];assert len(z['trace'])==3*len(grid)+1
    loss=np.zeros(len(grid));maxerr=0.;checks=0;position=0
    for k in range(3):
        ti=train[sp.loc[train,'inner_for_outer_'+str(j)]!=k];vi=train[sp.loc[train,'inner_for_outer_'+str(j)]==k]
        a,b=design(x,ti,vi,P['panels'][p]);yy=target.loc[ti].to_numpy()
        for ci,c in enumerate(grid):
            q=z['trace'][position];position+=1;assert q['config']==c and q['seed']==P['seed_base']+1000*h+10*j+k
            assert q['train_design_sha256']==hashlib.sha256(a.tobytes()).hexdigest();assert q['train_target_sha256']==hashlib.sha256(yy.tobytes()).hexdigest()
            pred=np.array(q['prediction']);assert len(pred)==len(vi) and np.isfinite(pred).all();loss[ci]+=np.abs(pred-target.loc[vi].to_numpy()).sum()
            if alg=='ridge':maxerr=max(maxerr,float(np.max(np.abs(pred-svd(a,b,yy,c)))));checks+=1
    loss/=len(train);assert np.allclose(loss,z['inner_MAE'],atol=1e-12,rtol=0);selected=int(np.flatnonzero(loss<=loss.min()+1e-12)[0]);assert selected==z['selected']
    a,b=design(x,train,test,P['panels'][p]);yy=target.loc[train].to_numpy();last=z['trace'][-1];seed=P['seed_base']+1000*h+10*j+9
    assert last['config']==grid[selected] and last['seed']==seed and last['prediction']==z['prediction'];assert last['train_design_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
    if alg=='ridge':pr=svd(a,b,yy,grid[selected]);checks+=1
    elif alg=='EBM':
        from interpret.glassbox import ExplainableBoostingRegressor
        model=ExplainableBoostingRegressor(**P['EBM_fixed'],**grid[selected],random_state=seed);model.fit(a,yy);pr=model.predict(b);checks+=1
    else:
        from tabpfn import TabPFNRegressor
        model=TabPFNRegressor(**P['TabPFN_fixed'],**grid[selected],model_path=D/P['checkpoint'],random_state=seed);model.fit(a,yy);pr=model.predict(b);checks+=1
    maxerr=max(maxerr,float(np.max(np.abs(pr-np.array(z['prediction'])))));assert maxerr<1e-8,(t,maxerr)
    return dict(task=t,fit_replays=checks,max_prediction_difference=maxerr,selected=selected)

def calc(y,p):
    error=p-y;ym=y.mean(1);pm=p.mean(1);yd=y-ym[:,None];pd=p-pm[:,None];ssy=np.einsum('ij,ij->i',yd,yd);ssp=np.einsum('ij,ij->i',pd,pd)
    with np.errstate(divide='ignore',invalid='ignore'):
        slope=np.where(np.ptp(p,axis=1)>1e-12,np.einsum('ij,ij->i',yd,pd)/ssp,np.nan)
        r2=np.where(ssy>0,1-np.einsum('ij,ij->i',error,error)/ssy,np.nan)
    return {'MAE':abs(error).mean(1),'RMSE':np.sqrt(np.mean(error**2,axis=1)),'R2':r2,'bias':error.mean(1),'calibration_slope':slope,'calibration_intercept':ym-slope*pm,'outside_0_30':((p<0)|(p>30)).mean(1)}
def verify_metrics():
    production=json.loads((R/'metrics.json').read_text());x,m,y,s,reserve=load();df=pd.read_csv(R/'restricted/oof_predictions.csv',dtype={'component':str});maxdiff=0.;count=0;table=[]
    def compare(a,b):
        nonlocal maxdiff
        if b is None:assert a is None;return
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maxdiff=max(maxdiff,err);assert err<1e-9,(a,b)
    for view,vp in P['views'].items():
        for h in P['horizons']:
            z=df[(df.view==view)&(df.horizon==h)].sort_values('component');ids=list(z.component);eligible=m.xs(h,level='horizon');eligible=eligible if view=='expanded' else eligible[eligible.complete28==1]
            assert ids==sorted(eligible.index) and len(ids)==vp['n'][P['horizons'].index(h)];assert np.array_equal(z.observed.to_numpy(),y.xs(h,level='horizon').loc[ids,'y']);assert np.array_equal(z.outer,s.loc[ids,'outer'])
            assert np.array_equal(z.persistence,x.loc[ids,'baseline_moca']);arms=['persistence']+[p+'__'+a for p in vp['panels'] for a in P['algorithms']]
            for panel in vp['panels']:
                for alg in P['algorithms']:
                    for j in range(5):
                        ck=json.loads((R/'restricted/checkpoints'/f'{view}_{h}_{panel}_{alg}_{j}.json').read_text());assert np.allclose(z.set_index('component').loc[ck['ids'],panel+'__'+alg],ck['prediction'],atol=1e-12,rtol=0)
            truth=z.observed.to_numpy();seed=P['seed_base']+h+100*(1+list(P['views']).index(view));rng=np.random.Generator(np.random.PCG64(seed));ix=rng.integers(0,len(z),size=(P['bootstrap'],len(z)));draw={};point={}
            for arm in arms:
                q=z[arm].to_numpy();point[arm]=calc(truth[None,:],q[None,:]);draw[arm]=calc(truth[ix],q[ix])
            flat={a+'/'+k:(point[a][k][0],v) for a in arms for k,v in draw[a].items()}
            for a,b in itertools.combinations(arms,2):
                for key in ['MAE','R2']:flat[a+'-'+b+'/delta_'+key]=(point[a][key][0]-point[b][key][0],draw[a][key]-draw[b][key])
            assert set(flat)==set(production[view+'/'+str(h)])
            for key,(value,vals) in flat.items():
                good=vals[np.isfinite(vals)];ci=np.quantile(good,[.025,.975]).tolist() if len(good)>=.95*P['bootstrap'] and np.isfinite(value) else None
                expected=production[view+'/'+str(h)][key];compare(float(value) if np.isfinite(value) else None,expected['estimate']);compare(ci,expected['interval']);assert len(good)==expected['valid'];count+=1
            for arm in arms:table.append(dict(view=view,horizon=h,n=len(z),model=arm,**{key:float(point[arm][key][0]) for key in point[arm]}))
            if view=='complete28' and h==12:
                for alg in ['EBM','TabPFN']:
                    diff=np.abs(z['compact4__'+alg].to_numpy()-truth)-np.abs(z.compact4__ridge.to_numpy()-truth);want=production['co_primary_adjusted'][alg];compare(float(diff.mean()),want['delta_MAE']);compare(np.quantile(diff[ix].mean(1),[.0125,.9875]),want['interval_97_5'])
    pd.DataFrame(table).to_csv(V/'Verified_Model_Performance.csv',index=False)
    return count,maxdiff

def main():
    files=sorted((R/'restricted/checkpoints').glob('*.json'));assert len(files)==225
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        results=[]
        for row in pool.map(audit_checkpoint,files):
            results.append(row)
            if len(results)%15==0:print('Verified checkpoint',len(results),flush=True)
    (V/'Prediction_Replays.json').write_text(json.dumps(results,indent=2));count,error=verify_metrics()
    import sys;sys.path.insert(0,str(D));import launcher;launcher.verify()
    report=dict(status='PASS',production_fit_calls=2475,independent_ridge_svd_fits=1200,independent_EBM_TabPFN_outer_refits=150,checkpoints_audited=225,metrics_and_intervals_verified=count,max_prediction_difference=max(r['max_prediction_difference'] for r in results),max_metric_difference=error,co_primary_intervals_verified=True,all_frozen_hashes_unchanged=True)
    (V/'Independent_Verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
