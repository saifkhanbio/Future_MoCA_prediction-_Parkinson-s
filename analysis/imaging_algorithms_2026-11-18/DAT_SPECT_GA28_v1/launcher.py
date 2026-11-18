"""Frozen GA-selected and full28 DAT comparison; default is read-only preflight."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.umask(0o077)
from pathlib import Path
import json,hashlib,argparse,copy,fcntl,tempfile
import numpy as np,pandas as pd
from numerical_core import predict_fold
from paired_metrics import summarize
D=Path(__file__).resolve().parent
ARMS=['GA_selected','GA_selected_DAT','full28','full28_DAT']
def need(c,msg):
 if not c:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def integrity():
 lock=json.loads((D/'LOCK.json').read_text());need(all(sha(D/p)==v for p,v in lock['files'].items()),'Locked file changed')
 sources=json.loads((D/'Source_SHA256.json').read_text());need(all(sha(p)==v for p,v in sources.items()),'Source changed')
 if (D/'VALIDATED_LAUNCHER.json').exists():
  z=json.loads((D/'VALIDATED_LAUNCHER.json').read_text());need(z['status']=='PASS' and z['lock_sha256']==sha(D/'LOCK.json'),'Validation seal mismatch')
def inputs():
 def read(n):return pd.read_csv(D/'restricted'/f'{n}.csv',dtype={'component':str,'baseline_sdmt_form':str,'baseline_hvlt_form':str,'jlo_form':str})
 return json.loads((D/'Protocol.json').read_text()),read('Predictors').set_index('component'),read('Membership').set_index(['component','horizon']),read('Endpoints').set_index(['component','horizon']),read('Grouped_Splits').set_index('component'),set(read('Stage_A_Reserved').component)
def panel(p,arm,h,j):
 need(arm in ARMS,'Unknown model')
 z=copy.deepcopy(p['frozen_fold_panels'][str(h)+'/'+str(j)]['panel'] if arm.startswith('GA_selected') else p['full28_panel'])
 if arm.endswith('_DAT'):z['numeric']+=p['DAT_fields']
 return z
def blocks(p,m):
 for v in p['views']:
  for h in p['horizons']:
   q=m.xs(h,level='horizon')
   if v=='strict_preindex':q=q[q.strict_preindex]
   yield v,h,sorted(q.index)
def validate(p,x,m,y,s,reserved):
 need(p['arms']==ARMS and p['horizons']==[12,24,36],'Wrong arms or horizons')
 need(p['penalties']==[.001,.01,.1,1.,10.] and p['bootstrap']==5000,'Wrong tuning or bootstrap contract')
 need(len(p['feature_universe'])==28 and len(set(p['feature_universe']))==28,'Invalid candidate universe')
 need(len(set(p['DAT_fields']))==4 and not set(p['DAT_fields'])&set(p['feature_universe']),'Invalid imaging block')
 need(set(p['frozen_fold_panels'])=={str(h)+'/'+str(j) for h in p['horizons'] for j in range(5)},'Missing fold-specific selection')
 full=p['full28_panel'];need(set(full['numeric'])|{'motor_state'}==set(p['feature_universe']),'Full28 panel mismatch')
 for k,r in p['frozen_fold_panels'].items():
  need(isinstance(r['mask'],int) and 0<=r['mask']<2**28,'Invalid mask')
  names=[n for i,n in enumerate(p['feature_universe']) if r['mask']&(1<<i)]
  need(names==r['features'],'Mask name mismatch')
  need(r['panel']['numeric']==[n for n in full['numeric'] if n in names],'Fold numeric panel mismatch')
  cats={}
  if 'motor_state' in names:cats['motor_state']=full['categories']['motor_state']
  for c,triggers in [('baseline_sdmt_form',['SDMT']),('baseline_hvlt_form',['HVLT_immediate','HVLT_delayed','HVLT_recognition_discrimination']),('jlo_form',['JLO'])]:
   if set(names)&set(triggers):cats[c]=full['categories'][c]
  need(cats==r['panel']['categories'],'Triggered form mismatch')
 need(all(t.index.is_unique for t in [x,m,y,s]),'Duplicate identity or endpoint')
 need(set(x.index)==set(s.index)==set(m.index.get_level_values('component')),'Master membership mismatch')
 need(not set(x.index)&reserved,'Reserved identity')
 need(set(m.index)==set(y.index),'Endpoint membership mismatch')
 need(m.endpoint_row.equals(y.endpoint_row.reindex(m.index)) and not m.endpoint_row.duplicated().any(),'Wrong endpoint binding')
 yy=y.y.to_numpy();need(np.all(np.isfinite(yy)&(yy>=0)&(yy<=30)&(yy==np.floor(yy))),'Invalid outcome')
 need(not np.isinf(x[full['numeric']].to_numpy(float)).any(),'Infinite clinical predictor')
 for c,levels in full['categories'].items():need(x[c].isin(levels).all(),'Invalid source category')
 need(x.baseline_moca.between(0,30).all(),'Invalid baseline MoCA')
 z=x[p['DAT_fields']].to_numpy(float);need(np.all(np.isfinite(z)&(z>=-1)),'Missing or invalid DAT')
 need(m.strict_preindex.isin([True,False]).all(),'Wrong strict type')
 need(np.array_equal(m.strict_preindex,m.scan_month<m.index_month),'Wrong strict membership')
 need(np.array_equal(m.scan_month-m.index_month,m.scan_minus_index),'Wrong scan timing')
 need(m.scan_minus_index.between(-3,0).all(),'Scan outside window')
 need(np.all(np.abs(m.endpoint_month-m.index_month-m.index.get_level_values('horizon'))<=3),'Outcome outside window')
 for c in ['index_month','scan_month']:need(m.groupby(level='component')[c].nunique().eq(1).all(),'Cross-horizon landmark mismatch')
 need(s.outer.isin(range(5)).all(),'Invalid outer fold')
 cells=[]
 for v,h,ids in blocks(p,m):
  need(len(ids)==p['views'][v]['n'][p['horizons'].index(h)],'Wrong matched count')
  for j in range(5):
   z=s.loc[ids];tr=z[z.outer.ne(j)];te=z[z.outer.eq(j)];col=f'inner_for_outer_{j}'
   need(len(te)>0 and len(tr)>0 and te[col].eq(-1).all() and set(tr[col])=={0,1,2},'Fold leakage or empty fold')
   cells.append(dict(view=v,horizon=h,outer=j,train_n=len(tr),test_n=len(te)))
 need(len(cells)*len(ARMS)*16==p['planned_fits']==1920,'Wrong fit budget')
 return cells
def evaluate(p,x,m,y,s,reserved):
 validate(p,x,m,y,s,reserved);rows=[];tuning=[];stats={};fits=0
 for v,h,ids in blocks(p,m):
  yy=y.xs(h,level='horizon').loc[ids,'y'];sp=s.loc[ids];pred={}
  for arm in ARMS:
   pr=pd.Series(np.nan,index=ids)
   for j in range(5):
    spec=panel(p,arm,h,j);ti,values,lam,loss=predict_fold(x,yy,sp,j,spec,p['penalties']);fits+=16;pr.loc[ti]=values
    tuning.append(dict(view=v,horizon=h,arm=arm,outer=j,penalty=lam,inner_MSE=loss,clinical_mask=p['frozen_fold_panels'][str(h)+'/'+str(j)]['mask'] if arm.startswith('GA_selected') else 2**28-1,panel=spec))
   need(pr.notna().all(),'Incomplete predictions');pred[arm]=pr.to_numpy()
  for i,k in enumerate(ids):rows.append(dict(component=k,view=v,horizon=h,outer=int(sp.at[k,'outer']),endpoint_row=int(m.at[(k,h),'endpoint_row']),observed=float(yy.loc[k]),**{a:float(pr[i]) for a,pr in pred.items()}))
  raw=summarize(yy.to_numpy(),pred,p['bootstrap'],p['seed_base']+100*(list(p['views']).index(v)+1)+h)
  out={a+'/'+k:raw[a+'/'+k] for a in ARMS for k in p['metrics']}
  for a,b in p['paired_contrasts']:
   for metric in ['MAE','R2']:
    key=a+'-'+b+'/delta_'+metric;other=raw[b+'-'+a+'/delta_'+metric]
    out[key]=dict(estimate=-other['estimate'] if other['estimate'] is not None else None,interval=[-other['interval'][1],-other['interval'][0]] if other['interval'] is not None else None,valid=other['valid'])
  stats[v+'/'+str(h)]=out
 need(fits==p['planned_fits'],'Wrong completed fits')
 return rows,stats,tuning,dict(completed_fits=fits,scope='Frozen GA masks, conditional matched-subset ridge retuning; no GA search')
def publish(root,payload,kind):
 root=Path(root).resolve();need(root.parent in [D.resolve(),(D/'qa').resolve()],'Output outside analysis')
 with (root.parent/(root.name+'.publication.lock')).open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);need(not root.exists(),'Existing result destination')
  with tempfile.TemporaryDirectory(dir=root.parent,prefix='.publish_') as tmp:
   stage=Path(tmp)/'result';stage.mkdir();(stage/'restricted').mkdir();rows,stats,tuning,run=payload
   need(kind in ['study','synthetic'],'Invalid kind')
   if kind=='synthetic':need(all(r['component'].startswith('SYN') for r in rows),'Study data in synthetic output')
   need(all(np.isfinite(r[a]) for r in rows for a in ARMS+['observed']),'Nonfinite output')
   pd.DataFrame(rows).to_csv(stage/'restricted/predictions.csv',index=False)
   for n,data in [('metrics',stats),('tuning',tuning),('execution',dict(run,kind=kind,lock_sha256=sha(D/'LOCK.json')))]:
    (stage/(n+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
   (stage/'SHA256.json').write_text(json.dumps({str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},indent=2)+'\n');integrity();stage.rename(root)
def authorize():
 a=D/'STUDY_EXECUTION_AUTHORIZATION.json';need(a.exists(),'Study training not authorized')
 j=json.loads(a.read_text());need(j.get('authorized') is True and j.get('lock_sha256')==sha(D/'LOCK.json') and j.get('user_instruction'),'Invalid authorization')
 need((D/'VALIDATED_LAUNCHER.json').exists(),'Launcher not validated');need(not (D/'study_results').exists(),'Study results already exist')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','synthetic','run-study'],default='preflight');o=ap.parse_args();integrity()
 if o.mode=='synthetic':
  from validate_synthetic import exercise
  print(json.dumps(exercise()));return
 if o.mode=='run-study':authorize()
 args=inputs();cells=validate(*args)
 if o.mode=='preflight':print(json.dumps(dict(status='PASS',fold_cells=len(cells),planned_fits=1920,study_fits=0)));return
 publish(D/'study_results',evaluate(*args),'study');print(json.dumps(dict(status='COMPLETE',fits=1920)))
if __name__=='__main__':main()
