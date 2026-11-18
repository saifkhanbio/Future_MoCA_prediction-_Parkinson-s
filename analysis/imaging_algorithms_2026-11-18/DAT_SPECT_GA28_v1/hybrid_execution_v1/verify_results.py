"""Independent study verifier: reads frozen CSVs; separate preprocessing/SVD ridge and metrics.
Does not import production numerical_core, predict_fold or paired_metrics.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.umask(0o077)
from pathlib import Path
import json,hashlib,time,datetime
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
D=Path(__file__).resolve().parent.parent;C=D;R=D/'study_results';V=D/'study_verification'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_hashes():
 for name,root in [('LOCK.json',C),('Source_SHA256.json',None)]:
  j=json.loads((C/name).read_text());files=j['files'] if name=='LOCK.json' else j
  assert all(sha(root/p if root else p)==v for p,v in files.items())
 for manifest,root in [(D/'Artifact_SHA256.json',D),(R/'SHA256.json',R)]:
  assert all(sha(root/p)==h for p,h in json.loads(manifest.read_text()).items())
 seal=json.loads((D/'VALIDATED_LAUNCHER.json').read_text())
 assert seal['lock_sha256']==sha(C/'LOCK.json')
check_hashes();V.mkdir(exist_ok=False);(V/'restricted').mkdir();started=time.monotonic()
plan=json.loads((C/'Protocol.json').read_text())
def read(n):return pd.read_csv(C/'restricted'/f'{n}.csv',dtype={'component':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str})
x=read('Predictors').set_index('component');m=read('Membership');y=read('Endpoints').set_index(['component','horizon']);s=read('Grouped_Splits').set_index('component');reserved=set(read('Stage_A_Reserved').component)
pred=pd.read_csv(R/'restricted/predictions.csv',dtype={'component':str})
tuning=json.loads((R/'tuning.json').read_text());metrics=json.loads((R/'metrics.json').read_text());execution=json.loads((R/'execution.json').read_text())
assert execution['completed_fits']==1920 and execution['lock_sha256']==sha(C/'LOCK.json') and execution['kind']=='study'
assert not pred.duplicated(['component','view','horizon']).any()
assert set(pred.component).isdisjoint(reserved)
assert len(pred)==sum(sum(a['n']) for a in plan['views'].values())
assert len(tuning)==120
lookup={(t['view'],t['horizon'],t['arm'],t['outer']):t for t in tuning};assert len(lookup)==120
assert set(pred.view)==set(plan['views']) and set(pred.horizon)==set(plan['horizons'])
assert set(metrics)=={v+'/'+str(h) for v in plan['views'] for h in plan['horizons']}
# Independent column-by-column training-only design construction.
def transform(train,test,panel):
 A=[];B=[];MA=[];MB=[]
 for column in panel['numeric']:
  a=x.loc[train,column].to_numpy(float);b=x.loc[test,column].to_numpy(float)
  missing_a=np.isnan(a);missing_b=np.isnan(b)
  fill=float(np.median(a[~missing_a])) if np.any(~missing_a) else 0.
  a=np.where(missing_a,fill,a);b=np.where(missing_b,fill,b)
  center=np.mean(a);scale=np.sqrt(np.mean(np.square(a-center)))
  if scale<=1e-12:scale=1.
  A.append((a-center)/scale);B.append((b-center)/scale);MA.append(missing_a);MB.append(missing_b)
 A+=MA;B+=MB
 for col,levels in panel['categories'].items():
  for level in levels:A.append(x.loc[train,col].eq(level).to_numpy());B.append(x.loc[test,col].eq(level).to_numpy())
 return np.array(A,dtype=float).T,np.array(B,dtype=float).T
def oracle_panel(p,arm,h,j):
 import copy
 spec=copy.deepcopy(p['frozen_fold_panels'][str(h)+'/'+str(j)]['panel'] if arm.startswith('GA_selected') else p['full28_panel'])
 if arm.endswith('_DAT'):spec['numeric']+=p['DAT_fields']
 return spec

fits=0;maxpred=0.;maxloss=0.;foldchecks=[];independent_rows=[]
for view in plan['views']:
 for hi,h in enumerate(plan['horizons']):
  membership=m[m.horizon.eq(h)]
  if view=='strict_preindex':membership=membership[membership.scan_month.lt(membership.index_month)]
  ids=sorted(membership.component);assert len(ids)==plan['views'][view]['n'][hi]
  q=pred[pred.view.eq(view)&pred.horizon.eq(h)].set_index('component').sort_index()
  assert list(q.index)==ids
  yy=y.xs(h,level='horizon').loc[ids,'y'];mh=membership.set_index('component').loc[ids]
  np.testing.assert_array_equal(q.observed,yy)
  np.testing.assert_array_equal(q.endpoint_row,mh.endpoint_row)
  np.testing.assert_array_equal(q.endpoint_row,y.xs(h,level='horizon').loc[ids,'endpoint_row'])
  np.testing.assert_array_equal(q.outer,s.loc[ids,'outer'])
  fresh=pd.DataFrame(index=ids)
  for arm in plan['arms']:
   for outer in range(5):
    tr=[p for p in ids if s.at[p,'outer']!=outer];te=[p for p in ids if s.at[p,'outer']==outer]
    assert set(tr).isdisjoint(te)
    loss_lists={lam:[] for lam in plan['penalties']}
    for inner in range(3):
     it=[p for p in tr if s.at[p,'inner_for_outer_'+str(outer)]!=inner];iv=[p for p in tr if s.at[p,'inner_for_outer_'+str(outer)]==inner]
     assert set(it).isdisjoint(iv) and set(it+iv)==set(tr) and set(iv).isdisjoint(te)
     a,b=transform(it,iv,oracle_panel(plan,arm,h,outer))
     for lam in plan['penalties']:
      z=Ridge(alpha=len(it)*lam,solver='svd').fit(a,yy.loc[it]).predict(b);fits+=1
      loss_lists[lam].extend(np.square(z-yy.loc[iv].to_numpy()))
    losses=[float(np.mean(loss_lists[lam])) for lam in plan['penalties']]
    chosen=max(lam for lam,l in zip(plan['penalties'],losses) if l<=min(losses)+1e-12)
    production=lookup[(view,h,arm,outer)]
    assert chosen==production['penalty']
    assert production['panel']==oracle_panel(plan,arm,h,outer)
    assert production['clinical_mask']==(plan['frozen_fold_panels'][str(h)+'/'+str(outer)]['mask'] if arm.startswith('GA_selected') else 2**28-1)
    lossdiff=float(np.max(np.abs(np.array(losses)-production['inner_MSE'])));maxloss=max(maxloss,lossdiff)
    a,b=transform(tr,te,oracle_panel(plan,arm,h,outer))
    z=Ridge(alpha=len(tr)*chosen,solver='svd').fit(a,yy.loc[tr]).predict(b);fits+=1
    diff=float(np.max(np.abs(z-q.loc[te,arm])));maxpred=max(maxpred,diff);fresh.loc[te,arm]=z
    foldchecks.append(dict(view=view,horizon=h,arm=arm,outer=outer,train_n=len(tr),test_n=len(te),penalty=chosen,max_prediction_difference=diff,max_inner_MSE_difference=lossdiff))
  fresh['component']=ids;fresh['view']=view;fresh['horizon']=h;fresh['observed']=yy
  independent_rows.append(fresh.reset_index(drop=True))
  print(json.dumps({'verified_cell':view+'/'+str(h),'fits':fits}),flush=True)
assert fits==1920 and maxpred<1e-9 and maxloss<1e-10
pd.concat(independent_rows).to_csv(V/'restricted/independent_predictions.csv',index=False)
pd.DataFrame(foldchecks).to_csv(V/'Fold_Verification.csv',index=False)
# Verify every point estimate, interval endpoint and finite-bootstrap count independently.
maxpoint=0.;maxinterval=0.;statchecks=[]
for view in plan['views']:
 for h in plan['horizons']:
  q=pred[pred.view.eq(view)&pred.horizon.eq(h)].sort_values('component');actual=q.observed.to_numpy();n=len(q)
  rng=np.random.Generator(np.random.PCG64(plan['seed_base']+100*(list(plan['views']).index(view)+1)+h))
  ix=rng.integers(n,size=(plan['bootstrap'],n));drawy=actual[ix];ym=drawy.mean(1);yc=drawy-ym[:,None];sst=np.square(yc).sum(1)
  points={};draws={}
  for arm in plan['arms']:
   p=q[arm].to_numpy();e=p-actual
   slope,intercept=np.polyfit(p,actual,1) if np.ptp(p)>1e-12 else (np.nan,np.nan)
   vals=dict(MAE=mean_absolute_error(actual,p),RMSE=np.sqrt(mean_squared_error(actual,p)),R2=r2_score(actual,p) if np.ptp(actual)>0 else np.nan,bias=np.mean(e),calibration_slope=slope,calibration_intercept=intercept,outside_0_30=np.mean((p<0)|(p>30)))
   bp=p[ix];err=bp-drawy;pm=bp.mean(1);pc=bp-pm[:,None]
   with np.errstate(divide='ignore',invalid='ignore'):
    sl=np.where(np.ptp(bp,axis=1)>1e-12,np.sum(pc*yc,1)/np.sum(pc*pc,1),np.nan)
    rr=np.where(sst>0,1-np.sum(err*err,1)/sst,np.nan)
   ds=dict(MAE=np.mean(np.abs(err),1),RMSE=np.sqrt(np.mean(err*err,1)),R2=rr,bias=np.mean(err,1),calibration_slope=sl,calibration_intercept=ym-sl*pm,outside_0_30=np.mean((bp<0)|(bp>30),1))
   for k in vals:points[arm+'/'+k]=vals[k];draws[arm+'/'+k]=ds[k]
  for a,b in plan['paired_contrasts']:
   for metric in ['MAE','R2']:
    key=a+'-'+b+'/delta_'+metric;points[key]=points[a+'/'+metric]-points[b+'/'+metric];draws[key]=draws[a+'/'+metric]-draws[b+'/'+metric]
  frozen=metrics[view+'/'+str(h)];assert set(frozen)==set(points)
  for key,val in points.items():
   f=frozen[key];boot=draws[key];valid=boot[np.isfinite(boot)];assert len(valid)==f['valid']
   if np.isfinite(val):pdiff=abs(val-f['estimate'])
   else:assert f['estimate'] is None;pdiff=0.
   if np.isfinite(val) and len(valid)>=.95*plan['bootstrap']:
    interval=np.quantile(valid,[.025,.975],method='linear');cdiff=float(np.max(np.abs(interval-f['interval'])))
   else:assert f['interval'] is None;cdiff=0.
   maxpoint=max(maxpoint,pdiff);maxinterval=max(maxinterval,cdiff)
   statchecks.append(dict(view=view,horizon=h,statistic=key,point_difference=pdiff,max_interval_difference=cdiff,valid_draws=len(valid)))
assert maxpoint<1e-10 and maxinterval<1e-10
pd.DataFrame(statchecks).to_csv(V/'Metric_Verification.csv',index=False)
check_hashes()
result=dict(status='PASS',production_fits=execution['completed_fits'],independent_verification_fits=fits,selected_penalties_verified=len(foldchecks),metrics_and_intervals_verified=len(statchecks),max_prediction_difference=maxpred,max_inner_MSE_difference=maxloss,max_metric_point_difference=maxpoint,max_metric_interval_difference=maxinterval,prediction_rows=len(pred),matched_membership_endpoints_and_folds=True,reserved_identities_excluded=True,contract_and_source_hashes_unchanged=True,prior_artifacts_unchanged=True,production_outputs_unchanged=True,elapsed_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paid_queries=0)
(V/'Independent_Verification.json').write_text(json.dumps(result,indent=2)+'\n')
(V/'SHA256.json').write_text(json.dumps({str(p.relative_to(V)):sha(p) for p in V.rglob('*') if p.is_file()},indent=2)+'\n')
print(json.dumps(result,indent=2))
