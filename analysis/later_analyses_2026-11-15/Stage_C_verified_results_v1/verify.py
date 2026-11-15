"""Independent Stage C replay. Reads frozen inputs and production outputs only."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.umask(0o077)
from pathlib import Path
import itertools,json,hashlib,csv,platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

V=Path(__file__).resolve().parent
O=V.parent/'Stage_C_results_v1'
D=V.parent/'Stage_C_protocol_v1'
E=V.parent/'Stage_C_eligibility_v1'
def load(path):return json.loads(path.read_text())
def read(path):return pd.read_csv(path,dtype={'component':str,'PATNO':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str,'jlo_form':str},float_precision='round_trip')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fp:
        for block in iter(lambda:fp.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
locks=load(D/'LOCK.json')['files'];sources=load(D/'source_hashes.json')
frozen={str(D/name):v for name,v in locks.items()};frozen.update(sources)
production={str(p):sha(p) for p in [O/'metrics.json',O/'tuning.json',O/'restricted/oof_predictions.csv']}
def integrity():
    assert all(sha(p)==v for p,v in {**frozen,**production}.items())
integrity()
plan=load(D/'protocol.json');results=load(O/'metrics.json');log=load(O/'tuning.json')
tuning={(r['view'],r['horizon'],r['arm'],r['outer']):r for r in log}
assert len(tuning)==len(log)==195
xr=read(D/'restricted/predictors.csv').set_index('component',verify_integrity=True)
splits=read(D/'restricted/grouped_splits.csv').set_index('component',verify_integrity=True)
end=read(D/'restricted/endpoints.csv').set_index(['horizon','component'],verify_integrity=True)
mem=read(D/'restricted/membership.csv').set_index(['horizon','component'],verify_integrity=True)
audit=read(E/'restricted/horizon_ledger.csv')
held=set(read(V.parent/'Stage_AB_eligibility_2026-11-15/restricted/A_membership.csv').component)
assert len(held)==227 and held==set(read(D/'restricted/stage_A_reserved.csv').component)
audit=audit[audit.eligible & ~audit.component.isin(held)].set_index(['horizon','component'],verify_integrity=True)
assert set(audit.index)==set(mem.index)==set(end.index) and len(mem)==2144
assert set(xr.index)==set(splits.index)==set(mem.index.get_level_values('component')) and len(xr)==839 and not set(xr.index)&held
mocas=read(next(Path(p) for p in sources if p.endswith('MoCA__28Oct2026.csv')))
items=load(next(Path(p) for p in sources if p.endswith('Assessment_Contract.json') and 'biomarker_value_contract' in p))['cognition']['item_max']
for key,row in audit.iterrows():
    h,person=key; ref=int(row.endpoint_row)
    assert end.at[key,'endpoint_row']==mem.at[key,'endpoint_row']==ref
    native=mocas.iloc[ref-2]; assert native.PATNO==row.PATNO
    month,year=map(int,native.INFODT.split('/'));date=year*12+month
    assert date==row.endpoint_month==mem.at[key,'endpoint_month'] and row.index_month==mem.at[key,'index_month'] and abs(date-row.index_month-h)<=3
    total=sum(int(native[k]) for k in items)
    assert all(0<=native[k]<=v for k,v in items.items())
    assert min(30,total+int(xr.at[person,'education']<=12))==end.at[key,'y']
prod=read(O/'restricted/oof_predictions.csv').set_index(['view','horizon','component'],verify_integrity=True)
assert set(results)=={f'{v}/{h}' for v in plan['views'] for h in plan['horizons']}
checked=[];fits=0;n_predictions=0;cells=0;maxerr=dict(prediction=0.,inner_loss=0.,metrics=0.)
def compare(a,b,kind,atol):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error=float(np.max(np.abs(a-b)));maxerr[kind]=max(maxerr[kind],error)
    assert error<atol,(kind,error,atol)
def matrices(train,test,panel):
    vals=xr.loc[train,panel['numeric']];vals2=xr.loc[test,panel['numeric']]
    median=vals.median().fillna(0);a=vals.fillna(median);b=vals2.fillna(median)
    sd=a.std(ddof=0);scale=sd.where(sd>1e-12,1.);avg=a.mean()
    left=[((a-avg)/scale).values,vals.isna().values]
    right=[((b-avg)/scale).values,vals2.isna().values]
    for feat,levels in panel['categories'].items():
        for cols,ids in [(left,train),(right,test)]:
            cols.append(pd.get_dummies(pd.Categorical(xr.loc[ids,feat],categories=levels)).to_numpy())
    return np.column_stack(left),np.column_stack(right)
def all_stats(y,forecast,indices):
    # Each row is one distinct participant bootstrap sample.
    obs=y[indices];out={};yc=obs-obs.mean(1)[:,None];var=(yc*yc).sum(1)
    for name,pred in forecast.items():
        p=pred[indices];err=p-obs;pc=p-p.mean(1)[:,None]
        with np.errstate(invalid='ignore',divide='ignore'):
            slope=np.where(np.ptp(p,axis=1)>1e-12,np.sum(pc*yc,axis=1)/np.sum(pc*pc,axis=1),np.nan)
            r2=np.where(var>0,1.-np.sum(err*err,axis=1)/var,np.nan)
        for k,v in {'MAE':np.mean(abs(err),axis=1),'RMSE':np.sqrt(np.mean(err**2,axis=1)),'R2':r2,'bias':np.mean(err,axis=1),'calibration_slope':slope,'calibration_intercept':obs.mean(1)-slope*p.mean(1),'outside_0_30':np.mean((p<0)|(p>30),axis=1)}.items():out[name+'/'+k]=v
    for a,b in itertools.combinations(forecast,2):
        for metric in ('MAE','R2'):out[a+'-'+b+'/delta_'+metric]=out[a+'/'+metric]-out[b+'/'+metric]
    return out
keys_seen=set()
for rank,(view,spec) in enumerate(plan['views'].items(),1):
    for ix,h in enumerate(plan['horizons']):
        m=mem.xs(h);ids=m.index
        panel28=plan['panels']['full28']
        if view=='complete28':
            z=xr.loc[ids]
            ids=z.index[z[panel28['numeric']].notna().all(axis=1)&z[list(panel28['categories'])].ne('unknown').all(axis=1)]
        if view=='historical':ids=m.index[(m.endpoint_month<2023*12+1)&(m.index_month<2023*12+1)]
        ids=sorted(ids);df=prod.xs((view,h)).sort_index()
        assert ids==list(df.index) and len(ids)==spec['n'][ix] and not set(ids)&held
        truth=end.xs(h).loc[ids,'y']
        assert (df.observed.to_numpy()==truth.to_numpy()).all() and (df.outer==splits.loc[ids,'outer']).all()
        for name in set(prod.columns)-{'outer','observed'}-set(spec['models']):assert df[name].isna().all()
        f={name:df[name].to_numpy() for name in spec['models']}
        assert np.array_equal(f['persistence'],xr.loc[ids,'baseline_moca'].to_numpy())
        sub=splits.loc[ids]
        for j in range(5):
            train=sub.index[sub.outer!=j];test=sub.index[sub.outer==j];inner=sub['inner_for_outer_'+str(j)]
            assert set(inner[train])=={0,1,2} and (inner[test]==-1).all() and set(train).isdisjoint(test)
            for arm in spec['models']:
                if arm=='persistence':continue
                rk=(view,h,arm,j);keys_seen.add(rk)
                panel=plan['panels'][arm];grid=plan['penalties'];loss=np.zeros(len(grid))
                for k in range(3):
                    tr=train[inner[train]!=k];va=train[inner[train]==k];a,b=matrices(tr,va,panel)
                    for i,pen in enumerate(grid):
                        model=Ridge(alpha=pen*len(tr),solver='svd').fit(a,truth.loc[tr])
                        loss[i]+=abs(model.predict(b)-truth.loc[va].to_numpy()).sum();fits+=1
                loss/=len(train);selected=max(p for p,e in zip(grid,loss) if e<=loss.min()+1e-12)
                saved=tuning[rk];assert selected==saved['penalty'];compare(loss,saved['inner_MAE'],'inner_loss',1e-9)
                a,b=matrices(train,test,panel);model=Ridge(alpha=selected*len(train),solver='svd').fit(a,truth.loc[train])
                compare(model.predict(b),df.loc[test,arm].to_numpy(),'prediction',1e-8);fits+=1
        n_predictions+=len(ids)*len(f)
        y=truth.to_numpy();point=all_stats(y,f,np.arange(len(y))[None,:]);nb=plan['bootstrap']
        rng=np.random.default_rng(plan['seed_base']+rank*100+h);draws={k:[] for k in point}
        for start in range(0,nb,80):
            indices=rng.integers(len(y),size=(min(80,nb-start),len(y)))
            for k,v in all_stats(y,f,indices).items():draws[k].extend(v)
        expected=results[f'{view}/{h}'];assert set(expected)==set(point)
        for k,val in point.items():
            saved=expected[k];v=np.array(draws[k]);valid=v[np.isfinite(v)]
            assert len(valid)==saved['valid'];est=float(val[0]);lo=hi=None
            if np.isfinite(est):compare(est,saved['estimate'],'metrics',1e-8)
            else:assert saved['estimate'] is None
            if len(valid)>=.95*nb and np.isfinite(est):
                lo,hi=np.percentile(valid,[2.5,97.5]);compare([lo,hi],saved['interval'],'metrics',1e-8)
            else:assert saved['interval'] is None
            checked.append(dict(view=view,horizon=h,n=len(y),measure=k,estimate=est,lower95=lo,upper95=hi,valid_bootstrap=len(valid)))
        cells+=1
assert fits==3120 and keys_seen==set(tuning) and len(prod)==sum(sum(x['n']) for x in plan['views'].values())
integrity()
pd.DataFrame(checked).to_csv(V/'verified_estimates.csv',index=False)
report=dict(status='PASS',protocol_sha256=sha(D/'LOCK.json'),production_sha256=production,production_fits=3120,independent_svd_ridge_fits=fits,outer_tuning_decisions=len(keys_seen),penalty_losses_checked=5*len(keys_seen),out_of_fold_forecasts_checked=n_predictions,view_person_horizon_rows=len(prod),unique_development_identities=len(xr),stage_A_reserved=len(held),stage_A_overlap=0,native_outcome_bindings=len(end),view_horizon_cells=cells,verified_estimate_and_interval_records=len(checked),resamples_per_cell=plan['bootstrap'],largest_absolute_discrepancies=maxerr,frozen_files_verified=len(frozen),python=platform.python_version(),sklearn=sklearn.__version__)
(V/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
