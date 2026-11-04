"""Numerical validation using synthetic rows only. No target feature/outcome file may be read."""
import sys,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,math,copy,importlib.util,numpy as np
from scipy.stats import linregress
D=Path(__file__).resolve().parents[1];P=D.parent
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=str(args[0])
  if any(t in path for t in ['Target_Features_','Sealed_Outcomes_','Baseline_Resolved_','Endpoint_Audit_']):raise PermissionError('Numerical tests must not read target records')
sys.addaudithook(guard)
sys.path.insert(0,str(D/'src'));sys.path.insert(0,str(D/'scripts'))
import clinical_inference as ci
import launcher
q=P/'model_readiness_2026-10-24/src/engine.py'
spec=importlib.util.spec_from_file_location('frozen_engine',q);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
binding=json.loads((D/'config/Frozen_Model_Binding.json').read_text())['models']
contract=json.loads((D/'config/Temporal_Validation_Contract.json').read_text())
rng=np.random.default_rng(20261028);groups={};fixtures={};max_error=0.;count=0
def scalar_reference(row,model):
 s=model['state'];c=s['clinical_state'];terms=[]
 for name,beta in zip(s['columns'],s['coef']):
  if name.endswith('__missing'):
   val=float(row[name[:-9]] is None or row[name[:-9]]=='')
  elif '=' in name:
   key,level=name.split('=',1);val=float(str(row[key])==level)
  else:
   raw=row[name];st=c['numeric'][name];value=st['median'] if raw is None or raw=='' else float(raw)
   val=(value-st['mean'])/st['sd']
  terms.append(beta*val)
 return math.fsum(terms)+s['intercept']
for b in binding:
 m=json.loads(Path(b['path']).read_text());s=m['state'];c=s['clinical_state']
 rows=[]
 for i in range(64):
  row={k:(None if i==0 or (i+j)%7==0 else '' if (i+j)%17==0 else float(rng.normal(t['mean'],t['sd']))) for j,(k,t) in enumerate(c['numeric'].items())}
  row.update({k:levels[i%len(levels)] for k,levels in c['categories'].items()})
  rows.append(row)
 z=ci.design(rows,m,contract['predictors'][b['comparator']]);ref,names=engine.transform_clinical(rows,c)
 assert names==s['columns'];np.testing.assert_allclose(z,ref,atol=1e-12,rtol=0)
 actual=ci.predict_member(rows,m);expected=np.asarray([scalar_reference(r,m) for r in rows])
 err=float(np.max(np.abs(actual-expected)));max_error=max(max_error,err)
 np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
 groups.setdefault((b['horizon'],b['comparator']),[]).append(m)
 fixtures[b['comparator']]=rows;count+=len(rows)
for (h,arm),members in groups.items():
 rows=fixtures[arm];a,each=ci.predict_ensemble(rows,list(reversed(members)),contract['predictors'][arm])
 expected=np.asarray([math.fsum(scalar_reference(r,m) for m in members)/5 for r in rows])
 np.testing.assert_allclose(a,expected,atol=1e-10,rtol=0);assert each.shape==(64,5)
assert all(v==0 for v in engine.COUNTERS.values())
faults=[]
def reject(label,func):
 try:func()
 except (ValueError,PermissionError,KeyError):faults.append(label)
 else:raise AssertionError('Failed to reject '+label)
m=groups[12,'enhanced'][0];row=fixtures['enhanced'][1];key=m['state']['clinical_state']['order'][0]
for field,change in [
 ('RNA-enabled',lambda x:x['state'].update(RNA_off=False)),
 ('RNA-lambda',lambda x:x['state'].update(lambda_RNA=1)),
 ('invalid-sd',lambda x:x['state']['clinical_state']['numeric'][key].update(sd=0)),
 ('nonfinite-coef',lambda x:x['state']['coef'].__setitem__(0,float('nan'))),
 ('wrong-column-order',lambda x:x['state']['columns'].reverse()),
 ('QA-model',lambda x:x.update(qa_only=True))]:
 altered=copy.deepcopy(m);change(altered);reject(field,lambda x=altered:ci.predict_member([row],x))
for label,value in [('NaN',float('nan')),('infinity',float('inf')),('malformed','oops')]:
 altered=dict(row);altered[key]=value;reject(label,lambda r=altered:ci.predict_member([r],m))
altered=dict(row);del altered[key];reject('missing-feature-column',lambda:ci.predict_member([altered],m))
altered=dict(row);altered['motor_state']='invalid';reject('unknown-category',lambda:ci.predict_member([altered],m))
reject('missing-ensemble-member',lambda:ci.predict_ensemble([row],groups[12,'enhanced'][:4]))
dups=[groups[12,'enhanced'][0]]*5;reject('duplicate-ensemble-fold',lambda:ci.predict_ensemble([row],dups))
altered=copy.deepcopy(m);altered['state']['RNA_state']={'intentionally_unused':'poison'}
np.testing.assert_allclose(ci.predict_member([row],altered),ci.predict_member([row],m),atol=0,rtol=0)
# Independent metric reference: SciPy calibration, elementary scalar error calculations.
y=np.array([0.,5.,12.,18.,20.,23.,26.,28.,29.,30.,30.])
pred={'core':np.array([-2.,8.,14.,17.,23.,21.,27.,29.,32.,28.,31.]),
'enhanced':np.array([-1.,6.,13.,18.,21.,23.,26.,29.,30.,29.,31.]),
'persistence':np.array([3.,9.,15.,20.,21.,24.,28.,28.,29.,30.,30.]),
'source_typical_change':np.array([3.,9.,15.,20.,21.,24.,28.,28.,29.,30.,30.])-0.4}
for p in pred.values():
 r=ci.metrics(y,p);e=p-y;lr=linregress(p,y)
 expected={'MAE':sum(abs(float(x)) for x in e)/len(y),'RMSE':math.sqrt(sum(float(x)**2 for x in e)/len(y)),
 'R2':1-sum(float(x)**2 for x in e)/sum(float(t-y.mean())**2 for t in y),'mean_prediction_error':float(e.mean()),
 'calibration_intercept':lr.intercept,'calibration_slope':lr.slope,'fraction_predictions_outside_0_30':sum(x<0 or x>30 for x in p)/len(y),
 'clipped_MAE_0_30_descriptive':sum(abs(min(30,max(0,float(p[i])))-float(y[i])) for i in range(len(y)))/len(y)}
 for k,v in expected.items():assert math.isclose(r[k],v,rel_tol=0,abs_tol=1e-10)
B=5000;seed=123456;actual=ci.bootstrap(y,pred,seed,B)
idx=np.random.default_rng(seed).integers(0,len(y),size=(B,len(y)));Y=y[idx]
draws={}
for name,p in pred.items():
 V=p[idx];E=V-Y;yc=Y-Y.mean(axis=1,keepdims=True);pc=V-V.mean(axis=1,keepdims=True)
 sst=np.sum(yc**2,axis=1);var=np.sum(pc**2,axis=1)
 with np.errstate(invalid='ignore',divide='ignore'):
  slope=np.sum(pc*yc,axis=1)/var
  slope[(np.ptp(V,axis=1)<=1e-12)|(var==0)]=np.nan
  r2=1-np.sum(E**2,axis=1)/sst;r2[sst==0]=np.nan
 draws[name]={'MAE':np.mean(abs(E),axis=1),'RMSE':np.sqrt(np.mean(E**2,axis=1)),
 'R2':r2,'mean_prediction_error':E.mean(axis=1),'calibration_slope':slope,
 'calibration_intercept':Y.mean(axis=1)-slope*V.mean(axis=1)}
 for metric,values in draws[name].items():
  valid=values[np.isfinite(values)];a=actual['models'][name]['intervals'][metric]
  assert a['valid_replicates']==len(valid)
  if len(valid)>=B*.95:
   expected=np.quantile(valid,[.025,.975],method='linear');np.testing.assert_allclose([a['lower'],a['upper']],expected,atol=1e-10,rtol=0)
  else:assert a['lower'] is None and a['upper'] is None
for c,r in actual['contrasts'].items():
 a,b=c.split('-minus-');v=draws[a]['MAE']-draws[b]['MAE'];expected=np.quantile(v,[.025,.975],method='linear')
 np.testing.assert_allclose([r['interval']['lower'],r['interval']['upper']],expected,atol=1e-10,rtol=0)
assert ci.metrics([3,3],[2,4])['R2'] is None
assert ci.metrics([2,4],[3,3])['calibration_slope'] is None
assert ci.metrics([1,2],[100,200])['R2']<0
edge=ci.bootstrap([3.,3.],{k:[3.,3.] for k in pred},101,5000)
assert edge['models']['core']['intervals']['R2']['valid_replicates']==0
assert edge['models']['core']['intervals']['R2']['lower'] is None
assert edge['contrasts']['enhanced-minus-core']['interval']['lower']==0
reject('nonfinite-prediction',lambda:ci.metrics([1,2],[2,float('nan')]))
reject('unequal-metric-length',lambda:ci.metrics([1,2],[3]))
# Synthetic participant matching: permutation must not alter correct pairing.
pr=[{'component_id':'SYN_'+str(i),'PATNO':'SYN_'+str(i),'horizon_months':12} for i in range(3)]
out=[dict(r,education_adjusted_MoCA=20+i,endpoint_QC_pass=True) for i,r in enumerate(pr)]
assert ci.bind_outcomes(pr,out)==ci.bind_outcomes(pr,list(reversed(out)))
reject('duplicate-outcome',lambda:ci.bind_outcomes(pr,out+[out[0]]))
reject('missing-outcome',lambda:ci.bind_outcomes(pr,out[:2]))
bad=copy.deepcopy(out);bad[0]['PATNO']='SYN_WRONG';reject('wrong-participant-binding',lambda:ci.bind_outcomes(pr,bad))
bad=copy.deepcopy(out);bad[0]['horizon_months']=24;reject('wrong-horizon-binding',lambda:ci.bind_outcomes(pr,bad))
bad=copy.deepcopy(out);bad[0]['endpoint_QC_pass']=False;reject('failed-outcome-QC',lambda:ci.bind_outcomes(pr,bad))
reject('missing-execution-authorization',lambda:launcher.authorize(None,'predict',{'contract_sha256':'synthetic'}))
reject('output-outside-package',lambda:launcher.output_dir('/tmp/not-authorized-study-output'))
summary={'status':'PASS','fixture_type':'synthetic rows only; audit hook blocked access to temporal target features/outcomes',
 'frozen_models_checked':len(binding),'synthetic_member_prediction_checks':count,'five_member_ensembles_checked':len(groups),
 'max_scalar_reference_prediction_error':max_error,'source_transform_equivalence':True,'source_fit_counters':engine.COUNTERS,
 'paired_bootstrap_replicates_independently_verified':B,'constant_outcome_and_prediction_fixture_replicates':5000,
 'binding_and_rejection_checks':faults,'target_predictions_generated':0,'target_metrics_computed':0,'models_fit':0}
(D/'qa/Numerical_Verification.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
