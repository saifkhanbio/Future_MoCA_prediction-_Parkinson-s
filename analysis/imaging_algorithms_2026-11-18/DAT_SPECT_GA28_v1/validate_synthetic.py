"""Synthetic-only validation with separate SVD oracle; no study fitting."""
from copy import deepcopy
import json,subprocess,sys
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import launcher as L
from numerical_core import predict_fold,design,ridge

def fixture():
 p=json.loads((L.D/'Protocol.json').read_text());n=120;ids=['SYN%04d'%i for i in range(n)];rng=np.random.default_rng(20261119)
 names=p['full28_panel']['numeric'];x=pd.DataFrame(rng.normal(size=(n,len(names))),index=ids,columns=names)
 x.baseline_moca=rng.integers(16,31,n);x.age_index=rng.uniform(40,85,n);x.SDMT=rng.integers(12,75,n);x.HVLT_immediate=rng.integers(0,35,n);x.HVLT_delayed=rng.integers(0,13,n)
 x.education=16.;x.NLR=np.nan;x.loc[ids[::7],'part3']=np.nan
 for c,levels in p['full28_panel']['categories'].items():x[c]=rng.choice(levels,n)
 for c in p['DAT_fields']:x[c]=rng.uniform(-.2,2,n)
 s=pd.DataFrame({'outer':np.arange(n)%5},index=ids)
 for j in range(5):s[f'inner_for_outer_{j}']=np.where(s.outer.eq(j),-1,(np.arange(n)//5+j)%3)
 rows=[]
 for h in p['horizons']:
  for i,k in enumerate(ids):
   ix=24001+i%12;lag=0 if i%7==0 else -1-i%3
   y=np.clip(round(7+.5*x.at[k,'baseline_moca']+.025*x.at[k,'SDMT']+.2*x.at[k,'HVLT_delayed']+.4*x.at[k,p['DAT_fields'][0]]-.02*h+rng.normal(0,2)),0,30)
   rows.append(dict(component=k,horizon=h,index_month=ix,endpoint_month=ix+h,endpoint_row=h*10000+i,scan_month=ix+lag,scan_minus_index=lag,strict_preindex=lag<0,y=y))
 f=pd.DataFrame(rows).set_index(['component','horizon']);m=f.drop(columns='y');y=f[['y','endpoint_row']]
 for v in p['views']:p['views'][v]['n']=[120 if v=='month_compatible' else 102]*3
 return p,x,m,y,s,{'RESERVED'}
def independent_design(x,tr,te,p):
 A=[];B=[];MA=[];MB=[]
 for c in p['numeric']:
  a=x.loc[tr,c].to_numpy(float);b=x.loc[te,c].to_numpy(float);am=np.isnan(a);bm=np.isnan(b)
  med=np.median(a[~am]) if (~am).any() else 0.;a=np.where(am,med,a);b=np.where(bm,med,b);mu=a.mean();sd=np.sqrt(np.mean((a-mu)**2));sd=sd if sd>1e-12 else 1.
  A.append((a-mu)/sd);B.append((b-mu)/sd);MA.append(am);MB.append(bm)
 A+=MA;B+=MB
 for c,levels in p['categories'].items():
  for level in levels:A.append(x.loc[tr,c].eq(level).to_numpy());B.append(x.loc[te,c].eq(level).to_numpy())
 return np.array(A,float).T,np.array(B,float).T

def exercise():
 args=fixture();p,x,m,y,s,reserve=args;payload=L.evaluate(*args);rows,stats,tuning,run=payload;assert run['completed_fits']==1920 and len(tuning)==120
 frame=pd.DataFrame(rows);lookup={(r['view'],r['horizon'],r['arm'],r['outer']):r for r in tuning};maxpred=0.;maxloss=0.;fits=0
 for (v,h),q in frame.groupby(['view','horizon']):
  q=q.set_index('component').sort_index();ids=list(q.index);yy=y.xs(h,level='horizon').y
  assert q.endpoint_row.equals(m.xs(h,level='horizon').loc[ids,'endpoint_row']);np.testing.assert_array_equal(q.observed,yy.loc[ids])
  for arm in p['arms']:
   for j in range(5):
    tr=[k for k in ids if s.at[k,'outer']!=j];te=[k for k in ids if s.at[k,'outer']==j]
    # Build oracle panel directly from frozen specification, not launcher.panel.
    pan=deepcopy(p['frozen_fold_panels'][str(h)+'/'+str(j)]['panel'] if arm.startswith('GA_selected') else p['full28_panel'])
    if arm.endswith('_DAT'):pan['numeric']+=p['DAT_fields']
    assert pan==lookup[v,h,arm,j]['panel']
    errs={lam:[] for lam in p['penalties']}
    for inner in range(3):
     it=[k for k in tr if s.at[k,f'inner_for_outer_{j}']!=inner];iv=[k for k in tr if s.at[k,f'inner_for_outer_{j}']==inner];a,b=independent_design(x,it,iv,pan)
     for lam in p['penalties']:
      z=Ridge(alpha=len(it)*lam,solver='svd').fit(a,yy.loc[it]).predict(b);fits+=1;errs[lam].extend(np.square(z-yy.loc[iv].to_numpy()))
    losses=[float(np.mean(errs[lam])) for lam in p['penalties']];chosen=max(lam for lam,e in zip(p['penalties'],losses) if e<=min(losses)+1e-12)
    r=lookup[v,h,arm,j];assert chosen==r['penalty'];maxloss=max(maxloss,float(np.max(np.abs(np.array(losses)-r['inner_MSE']))))
    a,b=independent_design(x,tr,te,pan);z=Ridge(alpha=len(tr)*chosen,solver='svd').fit(a,yy.loc[tr]).predict(b);fits+=1;maxpred=max(maxpred,float(np.max(np.abs(z-q.loc[te,arm]))))
 assert fits==1920 and maxpred<1e-9 and maxloss<1e-10
 maxpoint=0.;maxci=0.;nmetrics=0
 for (v,h),q in frame.groupby(['view','horizon']):
  q=q.sort_values('component');yy=q.observed.to_numpy();rng=np.random.Generator(np.random.PCG64(p['seed_base']+100*(list(p['views']).index(v)+1)+h));ix=rng.integers(len(q),size=(p['bootstrap'],len(q)))
  by=yy[ix];ym=by.mean(1);yc=by-ym[:,None];sst=np.square(yc).sum(1);points={};draws={}
  for arm in p['arms']:
   z=q[arm].to_numpy();sl,it=np.polyfit(z,yy,1);point=dict(MAE=mean_absolute_error(yy,z),RMSE=np.sqrt(mean_squared_error(yy,z)),R2=r2_score(yy,z),bias=np.mean(z-yy),calibration_slope=sl,calibration_intercept=it,outside_0_30=np.mean((z<0)|(z>30)))
   bp=z[ix];e=bp-by;pm=bp.mean(1);pc=bp-pm[:,None];slope=np.sum(pc*yc,1)/np.sum(pc*pc,1)
   ds=dict(MAE=np.abs(e).mean(1),RMSE=np.sqrt(np.square(e).mean(1)),R2=1-np.square(e).sum(1)/sst,bias=e.mean(1),calibration_slope=slope,calibration_intercept=ym-slope*pm,outside_0_30=np.mean((bp<0)|(bp>30),1))
   for k in point:points[arm+'/'+k]=point[k];draws[arm+'/'+k]=ds[k]
  for a,b in p['paired_contrasts']:
   for k in ['MAE','R2']:key=a+'-'+b+'/delta_'+k;points[key]=points[a+'/'+k]-points[b+'/'+k];draws[key]=draws[a+'/'+k]-draws[b+'/'+k]
  assert set(points)==set(stats[v+'/'+str(h)])
  for k,z in points.items():
   r=stats[v+'/'+str(h)][k];assert r['valid']==5000;maxpoint=max(maxpoint,abs(z-r['estimate']));maxci=max(maxci,float(np.max(np.abs(np.quantile(draws[k],[.025,.975],method='linear')-r['interval']))));nmetrics+=1
 assert maxpoint<1e-10 and maxci<1e-10 and nmetrics==192
 # Held-out labels and test predictors do not influence training or selection.
 yy=y.xs(36,level='horizon').y;pan=L.panel(p,'GA_selected_DAT',36,0);ti,z,lam,loss=predict_fold(x,yy,s,0,pan,p['penalties']);mut=yy.copy();mut.loc[ti]=0
 _,z2,lam2,loss2=predict_fold(x,mut,s,0,pan,p['penalties']);assert np.array_equal(z,z2) and lam==lam2 and loss==loss2
 tr=list(s.index[s.outer.ne(0)]);xx=x.copy();xx.loc[ti,'age_index']=9999;a,_=design(x,tr,list(ti),pan);aa,_=design(xx,tr,list(ti),pan);assert np.array_equal(a,aa)
 constant=yy.copy();constant[:]=25;_,z,lam,_=predict_fold(x,constant,s,0,pan,p['penalties']);assert lam==10 and np.all(z==25)
 assert ridge(np.array([[0.],[1.]]),np.array([[100.]]),np.array([0.,30.]),.001)[0]>30
 zero=deepcopy(p);k='36/0';zero['frozen_fold_panels'][k]=dict(panel={'numeric':[],'categories':{}},features=[],mask=0,historical_alpha=10)
 L.validate(zero,x,m,y,s,reserve);_,z,_,_=predict_fold(x,yy,s,0,L.panel(zero,'GA_selected',36,0),p['penalties']);assert np.allclose(z,yy.loc[tr].mean())
 rejected=[]
 def reject(label,args):
  try:L.validate(*args)
  except (ValueError,KeyError):rejected.append(label);return
  raise AssertionError('Fault accepted: '+label)
 for label,which,change in [
 ('reserved identity',5,lambda a:{x.index[0]}),('duplicate endpoint',3,lambda a:pd.concat([a,a.iloc[:1]])),('missing endpoint',3,lambda a:a.iloc[1:])]:
  ar=list(args);ar[which]=change(ar[which]);reject(label,ar)
 for label,col,val in [('missing DAT',p['DAT_fields'][0],np.nan),('invalid DAT',p['DAT_fields'][0],-2),('invalid form','jlo_form','BAD')]:
  xx=x.copy();xx.iloc[0,xx.columns.get_loc(col)]=val;reject(label,(p,xx,m,y,s,reserve))
 for label,col,val in [('invalid outcome','y',31),('misbound outcome','endpoint_row',-99)]:
  z=y.copy();z.iloc[0,z.columns.get_loc(col)]=val;reject(label,(p,x,m,z,s,reserve))
 for label,col,val in [('strict mismatch','strict_preindex',True),('scan timing','scan_month',-1),('outcome window','endpoint_month',-1)]:
  z=m.copy();z.iloc[0,z.columns.get_loc(col)]=val;reject(label,(p,x,z,y,s,reserve))
 z=s.copy();z.iloc[0,z.columns.get_loc('inner_for_outer_0')]=1;reject('inner outer leakage',(p,x,m,y,z,reserve))
 for label,change in [('wrong count',lambda z:z['views']['strict_preindex']['n'].__setitem__(0,103)),('wrong feature mask',lambda z:z['frozen_fold_panels']['12/0'].__setitem__('mask',0)),('global panel substitution',lambda z:z['frozen_fold_panels']['12/0'].__setitem__('panel',z['full28_panel'])),('wrong model roster',lambda z:z['arms'].pop()),('wrong imaging block',lambda z:z['DAT_fields'].pop())]:
  z=deepcopy(p);change(z);reject(label,(z,x,m,y,s,reserve))
 root=L.D/'qa/synthetic_results';L.publish(root,payload,'synthetic');disk=pd.read_csv(root/'restricted/predictions.csv');assert disk.shape==frame.shape and list(disk.component)==list(frame.component);np.testing.assert_allclose(disk.select_dtypes('number'),frame.select_dtypes('number'),atol=1e-12,rtol=0)
 for label,dest,pay in [('overwrite result',root,payload),('out of scope',L.D.parent/'invalid_results',payload),('nonfinite output',L.D/'qa/invalid_result',deepcopy(payload))]:
  if label=='nonfinite output':pay[0][0]['GA_selected']=float('nan')
  try:L.publish(dest,pay,'synthetic')
  except ValueError:rejected.append(label)
  else:raise AssertionError(label)
 assert not (L.D/'qa/invalid_result').exists()
 b=subprocess.run([sys.executable,'-B',str(L.D/'launcher.py'),'--mode','run-study'],capture_output=True,text=True);assert b.returncode!=0 and 'Study training not authorized' in b.stderr;assert not (L.D/'study_results').exists()
 (L.D/'qa/study_guard.txt').write_text(b.stderr);L.integrity()
 return dict(status='PASS',synthetic_matrix_fits=1920,independent_SVD_matrix_fits=fits,selected_penalties_verified=120,metrics_and_intervals_verified=nmetrics,max_prediction_difference=maxpred,max_inner_MSE_difference=maxloss,max_metric_difference=maxpoint,max_interval_difference=maxci,synthetic_rows=len(rows),faults_rejected=rejected,heldout_label_and_preprocessing_checks=True,constant_all_missing_intercept_only_checks=True,study_execution_blocked=True,study_fits=0,paid_queries=0)
