"""Synthetic integration, independent numerical oracle and rejection tests. No study models."""
from copy import deepcopy
import json,subprocess,sys,hashlib
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from launcher import D,CONTRACT,validate,evaluate,design,predict_fold,ridge,publish,digest,verify_sources
from paired_metrics import metrics,summarize
def fixture():
 plan=json.loads((CONTRACT/'Protocol.json').read_text())
 rng=np.random.default_rng(18112026);n=120;ids=['SYN%04d'%i for i in range(n)]
 fields=plan['panels']['enhanced_DAT']['numeric']
 x=pd.DataFrame(rng.normal(size=(n,len(fields))),index=ids,columns=fields)
 x.baseline_moca=rng.integers(17,31,n);x.SDMT=rng.integers(20,80,n);x.HVLT_immediate=rng.integers(8,35,n);x.HVLT_delayed=rng.integers(0,13,n)
 for k in plan['DAT_fields']:x[k]=rng.uniform(-.1,2,n)
 for k,levels in plan['panels']['enhanced_DAT']['categories'].items():x[k]=rng.choice(levels[:-1],n)
 x.loc[ids[::7],'age_index']=np.nan;x.loc[ids[::9],'motor_state']='unknown'
 x['education']=16.;x['diagnosis_duration_years']=np.nan
 sp=pd.DataFrame({'outer':np.arange(n)%5},index=ids)
 for j in range(5):sp['inner_for_outer_'+str(j)]=np.where(sp.outer.eq(j),-1,(np.arange(n)//5+j)%3)
 rows=[]
 for h in plan['horizons']:
  for i,p in enumerate(ids):
   ix=2015*12+1+i%12;lag=0 if i%7==0 else -1-i%3
   yy=np.clip(round(.5*x.at[p,'baseline_moca']+.035*x.at[p,'SDMT']+.1*x.at[p,'HVLT_immediate']+.2*x.at[p,'HVLT_delayed']+.3*x.at[p,plan['DAT_fields'][0]]+5-h*.02+rng.normal(0,1.5)),0,30)
   rows.append(dict(component=p,horizon=h,index_month=ix,endpoint_month=ix+h,endpoint_row=10000*h+i,
    scan_month=ix+lag,scan_minus_index=lag,strict_preindex=lag<0,y=yy))
 data=pd.DataFrame(rows).set_index(['component','horizon']);m=data.drop(columns='y');y=data[['y','endpoint_row']]
 for v in plan['views']:plan['views'][v]['n']=[n if v=='month_compatible' else int(m.xs(h,level='horizon').strict_preindex.sum()) for h in plan['horizons']]
 return plan,x,m,y,sp,{'RESERVED_SYN'}
def oracle_design(x,tr,te,panel):
 aa=[];bb=[];ma=[];mb=[]
 for c in panel['numeric']:
  a=x.loc[tr,c].to_numpy(float);b=x.loc[te,c].to_numpy(float)
  am=np.isnan(a);bm=np.isnan(b)
  med=np.median(a[~am]) if (~am).any() else 0.
  a=np.where(am,med,a);b=np.where(bm,med,b);mu=np.mean(a);sd=np.sqrt(np.mean((a-mu)**2))
  if sd<=1e-12:sd=1.
  aa.append((a-mu)/sd);bb.append((b-mu)/sd);ma.append(am);mb.append(bm)
 ca=[];cb=[]
 for c,levels in panel['categories'].items():
  for v in levels:ca.append(x.loc[tr,c].eq(v).to_numpy());cb.append(x.loc[te,c].eq(v).to_numpy())
 return np.array(aa+ma+ca,dtype=float).T,np.array(bb+mb+cb,dtype=float).T
def exercise():
 args=fixture();plan,x,m,y,s,reserve=args
 payload=evaluate(*args);rows,stats,tuning,execution=payload
 assert execution['completed_fits']==3200 and len(tuning)==200 and len(stats)==8
 lookup={(r['view'],r['horizon'],r['arm'],r['outer']):r for r in tuning}
 frame=pd.DataFrame(rows);maxerr=0.;maxloss=0.;oraclefits=0
 for (view,h),sub in frame.groupby(['view','horizon']):
  sub=sub.set_index('component').sort_index();ids=list(sub.index);target=y.xs(h,level='horizon').y
  assert sub.endpoint_row.equals(m.xs(h,level='horizon').loc[ids,'endpoint_row'])
  assert np.array_equal(sub.observed,target.loc[ids]) and np.array_equal(sub.persistence,x.loc[ids,'baseline_moca'])
  for arm,panel in plan['panels'].items():
   for j in range(5):
    tr=[i for i in ids if s.at[i,'outer']!=j];te=[i for i in ids if s.at[i,'outer']==j]
    losses={p:[] for p in plan['penalties']}
    for k in range(3):
     it=[i for i in tr if s.at[i,'inner_for_outer_'+str(j)]!=k];iv=[i for i in tr if s.at[i,'inner_for_outer_'+str(j)]==k]
     a,b=oracle_design(x,it,iv,panel)
     for pen in plan['penalties']:
      z=Ridge(alpha=len(it)*pen,solver='svd').fit(a,target.loc[it]).predict(b)
      losses[pen].extend(np.abs(z-target.loc[iv].to_numpy()));oraclefits+=1
    loss=[float(np.mean(losses[p])) for p in plan['penalties']]
    best=max(p for p,e in zip(plan['penalties'],loss) if e<=min(loss)+1e-12)
    t=lookup[view,h,arm,j];assert best==t['penalty']
    maxloss=max(maxloss,float(np.max(np.abs(np.array(loss)-t['inner_MAE']))))
    a,b=oracle_design(x,tr,te,panel)
    z=Ridge(alpha=len(tr)*best,solver='svd').fit(a,target.loc[tr]).predict(b);oraclefits+=1
    maxerr=max(maxerr,float(np.max(np.abs(z-sub.loc[te,arm]))))
 assert maxerr<1e-9 and maxloss<1e-10 and oraclefits==3200
 # Independent paired bootstrap with the exact locked seed; direct per-person errors.
 maxci=0.
 for (view,h),sub in frame.groupby(['view','horizon']):
  sub=sub.sort_values('component');actual=sub.observed.to_numpy()
  rng=np.random.Generator(np.random.PCG64(plan['seed_base']+100*(list(plan['views']).index(view)+1)+h))
  ix=rng.integers(len(sub),size=(plan['bootstrap'],len(sub)))
  for a,b in plan['paired_contrasts']:
   ea=np.abs(sub[a].to_numpy()-actual);eb=np.abs(sub[b].to_numpy()-actual)
   d=ea-eb;key=a+'-'+b+'/delta_MAE';r=stats[view+'/'+str(h)][key]
   assert abs(r['estimate']-d.mean())<1e-12
   ci=np.quantile(d[ix].mean(1),[.025,.975],method='linear')
   maxci=max(maxci,float(np.max(np.abs(ci-r['interval']))))
 assert maxci<1e-10
 # Leakage: held-out labels do not alter the fold's tuning or predictions.
 target=y.xs(48,level='horizon').y;panel=plan['panels']['enhanced_DAT']
 ids,p,pen,loss=predict_fold(x,target,s,0,panel,plan['penalties'])
 mutated=target.copy();mutated.loc[ids]=0
 ids2,p2,pen2,loss2=predict_fold(x,mutated,s,0,panel,plan['penalties'])
 assert np.array_equal(p,p2) and pen==pen2 and loss==loss2
 tr=list(s.index[s.outer.ne(0)]);z=x.copy();z.loc[ids,'age_index']=99999
 a,b=design(x,tr,list(ids),panel);a2,_=design(z,tr,list(ids),panel);assert np.array_equal(a,a2)
 constant=target.copy();constant[:]=25
 _,p3,p3pen,_=predict_fold(x,constant,s,0,panel,plan['penalties'])
 assert p3pen==max(plan['penalties']) and np.all(p3==25)
 # Raw, unbounded predictions are not clipped.
 assert ridge(np.array([[0.],[1.]]),np.array([[100.]]),np.array([0.,30.]),.001)[0]>30
 special=metrics(np.array([[1.,1.]]),np.array([[2.,2.]]));assert np.isnan(special['R2'][0]) and np.isnan(special['calibration_slope'][0])
 special2=metrics(np.array([[1.,2.]]),np.array([[20.,20.]]));assert special2['R2'][0]<0
 undefined=summarize(np.array([1.,1.]),{'a':np.array([2.,2.])},128,1)
 assert undefined['a/R2']['estimate'] is None and undefined['a/R2']['interval'] is None
 rejected=[]
 def reject(label,xx=x,mm=m,yy=y,ss=s,rv=reserve,pl=plan):
  try:validate(pl,xx,mm,yy,ss,rv)
  except ValueError:rejected.append(label);return
  raise AssertionError('Fault accepted: '+label)
 reject('reserved participant',rv={x.index[0]})
 reject('duplicate endpoint',yy=pd.concat([y,y.iloc[:1]]))
 reject('missing endpoint',yy=y.iloc[1:])
 yy=y.copy();yy.iloc[0,yy.columns.get_loc('endpoint_row')]+=1;reject('wrong endpoint row',yy=yy)
 yy=y.copy();yy.iloc[0,yy.columns.get_loc('y')]=31;reject('invalid outcome',yy=yy)
 xx=x.copy();xx.iloc[0,xx.columns.get_loc(plan['DAT_fields'][0])]=np.nan;reject('missing DAT',xx=xx)
 xx=x.copy();xx.iloc[0,xx.columns.get_loc(plan['DAT_fields'][0])]=-2;reject('invalid SBR',xx=xx)
 xx=x.copy();xx.iloc[0,xx.columns.get_loc('baseline_sdmt_form')]='unknown';reject('unknown required form',xx=xx)
 ss=s.copy();ss.iloc[0,ss.columns.get_loc('inner_for_outer_0')]=1;reject('inner outer leakage',ss=ss)
 mm=m.copy();mm.iloc[0,mm.columns.get_loc('strict_preindex')]=True;reject('wrong strict cohort',mm=mm)
 mm=m.copy();mm.iloc[0,mm.columns.get_loc('scan_month')]+=1;reject('scan timing mismatch',mm=mm)
 mm=m.copy();mm.iloc[0,mm.columns.get_loc('endpoint_month')]+=4;reject('wrong outcome window',mm=mm)
 pl=deepcopy(plan);pl['views']['strict_preindex']['n'][0]+=1;reject('incorrect matched n',pl=pl)
 pl=deepcopy(plan);pl['panels']['compact4_DAT']['numeric']=pl['panels']['compact4_DAT']['numeric'][:-1];reject('altered imaging block',pl=pl)
 pl=deepcopy(plan);pl['views']['month_compatible']['models'].remove('enhanced');reject('unequal model roster',pl=pl)
 root=D/'qa'/'synthetic_results'
 publish(root,payload,'synthetic')
 disk=pd.read_csv(root/'restricted/oof_predictions.csv')
 assert disk.shape==frame.shape and list(disk.component)==list(frame.component)
 assert np.allclose(disk.select_dtypes('number'),frame.select_dtypes('number'),rtol=0,atol=1e-12)
 manifest=json.loads((root/'SHA256.json').read_text());assert all(digest(root/p)==v for p,v in manifest.items())
 try:publish(root,payload,'synthetic')
 except ValueError:rejected.append('overwrite existing result')
 else:raise AssertionError('Existing result overwritten')
 broken=deepcopy(payload);broken[0][0]['compact4']=float('nan')
 try:publish(D/'qa'/'failed_result',broken,'synthetic')
 except ValueError:rejected.append('nonfinite output atomic rejection')
 else:raise AssertionError('Invalid output accepted')
 assert not (D/'qa'/'failed_result').exists()
 try:publish(D.parent/'outside_result',payload,'synthetic')
 except ValueError:rejected.append('output outside analysis')
 else:raise AssertionError('Unsafe output accepted')
 blocked=subprocess.run([sys.executable,'-B',str(D/'launcher.py'),'--mode','run-study'],capture_output=True,text=True)
 assert blocked.returncode!=0 and 'Study fitting is not authorized' in blocked.stderr and not (D/'study_results').exists()
 (D/'qa/study_execution_guard.txt').write_text(blocked.stderr)
 verify_sources()
 return dict(status='PASS',synthetic_people=len(x),synthetic_nested_fits=3200,independent_SVD_fits=oraclefits,
 max_prediction_difference=maxerr,max_inner_loss_difference=maxloss,max_paired_MAE_interval_difference=maxci,
 bootstrap_per_view_horizon=plan['bootstrap'],synthetic_predictions=len(rows),faults_rejected=rejected,
 heldout_label_and_preprocessing_checks=True,constant_and_all_missing_columns=True,
 raw_prediction_and_undefined_metric_checks=True,atomic_output_and_roundtrip=True,study_execution_blocked=True,study_fits=0,paid_queries=0)
