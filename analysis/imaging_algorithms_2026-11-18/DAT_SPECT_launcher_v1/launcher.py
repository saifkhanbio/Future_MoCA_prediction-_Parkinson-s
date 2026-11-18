"""DAT-SPECT nested evaluation. Default preflight; study execution requires a separate authorization."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import json,hashlib,argparse,tempfile,fcntl
from pathlib import Path
import numpy as np,pandas as pd
from numerical_core import design,ridge,predict_fold
from paired_metrics import summarize
os.umask(0o077)
D=Path(__file__).resolve().parent
CONTRACT=D.parent/'DAT_SPECT_readiness_v1'
def require(c,msg):
 if not c:raise ValueError(msg)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_sources():
 seal=json.loads((CONTRACT/'LOCK.json').read_text())
 require(all(digest(CONTRACT/p)==v for p,v in seal['files'].items()),'Contract integrity mismatch')
 sources=json.loads((CONTRACT/'Source_SHA256.json').read_text())
 require(all(digest(p)==v for p,v in sources.items()),'Upstream source integrity mismatch')
 if (D/'VALIDATED_LAUNCHER.json').exists():
  sealed=json.loads((D/'VALIDATED_LAUNCHER.json').read_text())
  require(sealed.get('status')=='PASS' and sealed.get('contract_sha256')==digest(CONTRACT/'LOCK.json'),'Validation seal or contract binding mismatch')
  require(all(digest(D/p)==v for p,v in sealed['code'].items()),'Launcher integrity mismatch')
def inputs():
 r=CONTRACT/'restricted'
 def read(n):return pd.read_csv(r/(n+'.csv'),dtype={'component':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str,'jlo_form':str,'protocol':str})
 return (json.loads((CONTRACT/'Protocol.json').read_text()),read('Predictors').set_index('component'),
 read('Membership').set_index(['component','horizon']),read('Endpoints').set_index(['component','horizon']),
 read('Grouped_Splits').set_index('component'),set(read('Stage_A_Reserved').component))
def blocks(plan,m):
 for view in plan['views']:
  for h in plan['horizons']:
   q=m.xs(h,level='horizon')
   if view=='strict_preindex':q=q[q.strict_preindex]
   yield view,h,sorted(q.index)
def validate(plan,x,m,y,s,reserve):
 require(all(t.index.is_unique for t in [x,m,y,s]),'Duplicate identity or endpoint')
 require(set(x.index)==set(s.index)==set(m.index.get_level_values('component')),'Master membership mismatch')
 require(not set(x.index)&reserve,'Reserved identity')
 require(set(m.index)==set(y.index),'Outcome membership mismatch')
 require(set(m.index.get_level_values('horizon'))==set(plan['horizons']),'Wrong horizon')
 require(m.endpoint_row.equals(y.endpoint_row.reindex(m.index)),'Misbound endpoint row')
 require(not m.endpoint_row.duplicated().any(),'Endpoint row reused')
 yy=y.y.to_numpy()
 require(np.all(np.isfinite(yy)&(yy>=0)&(yy<=30)&(yy==np.floor(yy))),'Invalid outcome')
 for k,p in plan['panels'].items():
  require(not np.isinf(x[p['numeric']].to_numpy(dtype=float)).any(),'Infinite predictor')
  for col,levels in p['categories'].items():require(x[col].isin(levels).all(),'Invalid category')
 for k in ['compact4','enhanced']:
  require(plan['panels'][k+'_DAT']['numeric']==plan['panels'][k]['numeric']+plan['DAT_fields'],'Wrong DAT augmentation')
  require(plan['panels'][k+'_DAT']['categories']==plan['panels'][k]['categories'],'Category mismatch')
 comp=plan['panels']['compact4']
 require(x[comp['numeric']].notna().all().all(),'Missing mandatory cognitive score')
 require(x[list(comp['categories'])].ne('unknown').all().all(),'Unknown mandatory form')
 require(x.baseline_moca.between(0,30).all()&x.baseline_moca.mod(1).eq(0).all(),'Invalid baseline MoCA')
 dat=x[plan['DAT_fields']].to_numpy(dtype=float)
 require(np.all(np.isfinite(dat)&(dat>=-1)),'Missing or invalid DAT')
 require(m.strict_preindex.isin([True,False]).all(),'Invalid strict membership')
 require(np.array_equal(m.strict_preindex.to_numpy(),(m.scan_month<m.index_month).to_numpy()),'Incorrect strict membership')
 require(np.all(m.scan_month-m.index_month==m.scan_minus_index),'Inconsistent scan timing')
 require(m.scan_minus_index.between(-3,0).all(),'Scan outside baseline window')
 require(np.all(np.abs(m.endpoint_month-m.index_month-m.index.get_level_values('horizon'))<=3),'Outcome outside horizon')
 require(m.groupby(level='component').scan_month.nunique().eq(1).all(),'Scan changed across horizons')
 require(m.groupby(level='component').index_month.nunique().eq(1).all(),'Landmark changed across horizons')
 require(s.outer.isin(range(5)).all(),'Invalid outer fold')
 for j in range(5):
  z=s['inner_for_outer_'+str(j)]
  require(z[s.outer.eq(j)].eq(-1).all() and z[s.outer.ne(j)].isin(range(3)).all(),'Inner outer leakage')
 cells=[]
 for v,h,ids in blocks(plan,m):
  require(len(ids)==plan['views'][v]['n'][plan['horizons'].index(h)],'Wrong cohort count')
  require(plan['views'][v]['models']==['persistence','moca_only','compact4','compact4_DAT','enhanced','enhanced_DAT'],'Wrong matched arms')
  for j in range(5):
   z=s.loc[ids];tr=z[z.outer.ne(j)];te=z[z.outer.eq(j)]
   require(len(tr)>0 and len(te)>0 and set(tr['inner_for_outer_'+str(j)])=={0,1,2},'Empty fold')
   cells.append(dict(view=v,horizon=h,outer=j,train=len(tr),test=len(te)))
 require(len(cells)*5*(3*len(plan['penalties'])+1)==plan['planned_fits'],'Fit count mismatch')
 return cells
def evaluate(plan,x,m,y,s,reserve):
 validate(plan,x,m,y,s,reserve)
 output=[];tuning=[];stats={};fits=0
 for view,h,ids in blocks(plan,m):
  target=y.xs(h,level='horizon').loc[ids,'y'];sp=s.loc[ids];pred={}
  for arm in plan['views'][view]['models']:
   if arm=='persistence':pred[arm]=x.loc[ids,'baseline_moca'].to_numpy(dtype=float);continue
   pp=pd.Series(np.nan,index=ids)
   for j in range(5):
    ti,values,pen,loss=predict_fold(x,target,sp,j,plan['panels'][arm],plan['penalties'])
    pp.loc[ti]=values;fits+=3*len(plan['penalties'])+1
    tuning.append(dict(view=view,horizon=h,arm=arm,outer=j,penalty=pen,inner_MAE=loss))
   require(pp.notna().all(),'Incomplete predictions');pred[arm]=pp.to_numpy()
  for i,p in enumerate(ids):
   output.append(dict(component=p,view=view,horizon=h,outer=int(sp.at[p,'outer']),endpoint_row=int(m.at[(p,h),'endpoint_row']),
    observed=float(target.loc[p]),**{a:float(pr[i]) for a,pr in pred.items()}))
  seed=plan['seed_base']+100*(list(plan['views']).index(view)+1)+h
  full=summarize(target.to_numpy(),pred,plan['bootstrap'],seed)
  allowed={a+'/'+k for a in pred for k in plan['metrics']}
  for a,b in plan['paired_contrasts']:
   for metric in ['MAE','R2']:
    key=a+'-'+b+'/delta_'+metric;reverse=b+'-'+a+'/delta_'+metric
    if key not in full:
     r=full[reverse]
     full[key]=dict(estimate=-r['estimate'] if r['estimate'] is not None else None,
      interval=[-r['interval'][1],-r['interval'][0]] if r['interval'] is not None else None,valid=r['valid'])
    allowed.add(key)
  stats[view+'/'+str(h)]={k:full[k] for k in sorted(allowed)}
 require(fits==plan['planned_fits'],'Wrong completed fits')
 return output,stats,tuning,dict(completed_fits=fits,views_horizons=len(stats))
def publish(root,payload,kind):
 root=Path(root).resolve()
 require(root.parent==D.resolve() or root.parent==(D/'qa').resolve(),'Output outside launcher folder')
 root.parent.mkdir(exist_ok=True)
 # Nonblocking exclusive lock protects atomic publication and duplicate concurrent writers.
 with (root.parent/(root.name+'.publication.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(not root.exists(),'Result destination exists')
  with tempfile.TemporaryDirectory(dir=root.parent,prefix='.dat_publish_') as tmp:
   stage=Path(tmp)/'deliverables';stage.mkdir();(stage/'restricted').mkdir()
   rows,stats,tuning,runinfo=payload
   require(kind in ['synthetic','study'],'Unknown result kind')
   if kind=='synthetic':require(all(r['component'].startswith('SYN') for r in rows),'Study identity in synthetic output')
   require(all(np.isfinite(r[a]) for r in rows for a in ['observed','persistence','moca_only','compact4','compact4_DAT','enhanced','enhanced_DAT']),'Nonfinite result')
   pd.DataFrame(rows).to_csv(stage/'restricted/oof_predictions.csv',index=False)
   for n,data in [('metrics',stats),('tuning',tuning),('execution',dict(runinfo,kind=kind,contract_sha256=digest(CONTRACT/'LOCK.json')))]:
    (stage/(n+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
   manifest={str(p.relative_to(stage)):digest(p) for p in stage.rglob('*') if p.is_file()}
   (stage/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
   verify_sources()
   stage.rename(root)
 return root
def study_authorization():
 p=D/'STUDY_EXECUTION_AUTHORIZATION.json'
 require(p.exists(),'Study fitting is not authorized: separate execution instruction and authorization record required')
 a=json.loads(p.read_text())
 require(a.get('authorized') is True and a.get('contract_sha256')==digest(CONTRACT/'LOCK.json') and a.get('user_instruction'),'Invalid study execution authorization')
 require((D/'VALIDATED_LAUNCHER.json').exists(),'Launcher not sealed as validated')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','synthetic','run-study'],default='preflight');opt=p.parse_args()
 verify_sources()
 if opt.mode=='synthetic':
  from validate_synthetic import exercise
  print(json.dumps(exercise()));return
 if opt.mode=='run-study':
  study_authorization()
  require(not (D/'study_results').exists(),'Study result already exists; refusing refit')
 args=inputs()
 if opt.mode=='preflight':
  print(json.dumps(dict(status='PASS',fold_cells=len(validate(*args)),study_fits=0,planned_fits=args[0]['planned_fits'])));return
 payload=evaluate(*args);publish(D/'study_results',payload,'study')
 print(json.dumps(dict(status='COMPLETE',study_fits=payload[3]['completed_fits'])))
if __name__=='__main__':main()
