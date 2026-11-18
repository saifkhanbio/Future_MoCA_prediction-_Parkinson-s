# Execution-only parallel wrapper; original analysis and code remain frozen.
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.umask(0o077)
import sys,json,time,copy,multiprocessing,fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
D=Path(__file__).resolve().parent.parent;E=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import launcher as L
from launcher import validate,panel,summarize,need,ARMS,blocks
from numerical_core import design,predict_fold

def cpu_job(job):
 key,x,yy,sp,j,spec,grid=job;start=time.monotonic();r=predict_fold(x,yy,sp,j,spec,grid)
 return key,dict(value=r,backend='CPU',pid=os.getpid(),fits=16,seconds=time.monotonic()-start)

def gpu_job(job):
 import torch
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 need(torch.cuda.is_available(),'CUDA unavailable; refusing silent routing change')
 key,x,yy,sp,j,spec,grid=job;start=time.monotonic();training=sp.index[sp.outer.ne(j)];test=sp.index[sp.outer.eq(j)];assign=sp.loc[training,f'inner_for_outer_{j}'];errors=np.zeros(len(grid))
 def batched(a,b,y,lams):
  A=torch.as_tensor(a,dtype=torch.float64,device='cuda');B=torch.as_tensor(b,dtype=torch.float64,device='cuda');Y=torch.as_tensor(y,dtype=torch.float64,device='cuda');mu=A.mean(0);ym=Y.mean();Z=A-mu
  gram=Z.T@Z;rhs=Z.T@(Y-ym);pen=torch.as_tensor(lams,dtype=torch.float64,device='cuda')
  matrices=gram[None,:,:]+len(a)*pen[:,None,None]*torch.eye(a.shape[1],dtype=torch.float64,device='cuda')[None,:,:]
  coef=torch.linalg.solve(matrices,rhs[None,:,None].expand(len(lams),-1,-1)).squeeze(-1)
  output=ym+(B-mu)@coef.T;torch.cuda.synchronize();return output.cpu().numpy()
 for k in range(3):
  ti=training[assign.ne(k)];vi=training[assign.eq(k)];a,b=design(x,ti,vi,spec)
  pr=batched(a,b,yy.loc[ti].to_numpy(),grid);errors+=np.square(pr-yy.loc[vi].to_numpy()[:,None]).sum(0)
 errors/=len(training);lam=max(z for z,e in zip(grid,errors) if e<=min(errors)+1e-12)
 a,b=design(x,training,test,spec);pr=batched(a,b,yy.loc[training].to_numpy(),[lam])[:,0]
 need(np.isfinite(pr).all(),'Nonfinite GPU predictions')
 return key,dict(value=(test,pr,lam,errors.tolist()),backend='CUDA_float64',pid=os.getpid(),fits=16,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())

def parallel(args):
 p,x,m,y,s,reserved=args;validate(*args);jobs=[]
 for v,h,ids in L.blocks(p,m):
  yy=y.xs(h,level='horizon').loc[ids,'y'];sp=s.loc[ids]
  for arm in ARMS:
   for j in range(5):jobs.append(((v,h,arm,j),x,yy,sp,j,panel(p,arm,h,j),p['penalties']))
 context=multiprocessing.get_context('spawn');results={};start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4,mp_context=context) as cpu,ProcessPoolExecutor(max_workers=1,mp_context=context) as gpu:
  futures=[(gpu if job[0][2].startswith('full28') else cpu).submit(gpu_job if job[0][2].startswith('full28') else cpu_job,job) for job in jobs]
  for future in futures:
   key,r=future.result();need(key not in results,'Duplicate result');results[key]=r
 need(len(results)==120 and sum(r['fits'] for r in results.values())==1920,'Incomplete hybrid fits')
 payload=assemble(*args,results)
 routing=[dict(view=k[0],horizon=k[1],arm=k[2],outer=k[3],**{n:z for n,z in r.items() if n!='value'}) for k,r in sorted(results.items())]
 payload[3].update(cpu_workers=4,gpu_workers=1,cpu_fits=sum(r['fits'] for r in routing if r['backend']=='CPU'),gpu_fits=sum(r['fits'] for r in routing if r['backend']=='CUDA_float64'),parallel_elapsed_seconds=time.monotonic()-start,dtype='float64')
 return payload,routing

def assemble(p,x,m,y,s,reserved,results):
 validate(p,x,m,y,s,reserved);rows=[];tuning=[];stats={};fits=0
 for v,h,ids in blocks(p,m):
  yy=y.xs(h,level='horizon').loc[ids,'y'];sp=s.loc[ids];pred={}
  for arm in ARMS:
   pr=pd.Series(np.nan,index=ids)
   for j in range(5):
    spec=panel(p,arm,h,j);ti,values,lam,loss=results[(v,h,arm,j)]["value"];fits+=16;pr.loc[ti]=values
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

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','run']);o=ap.parse_args();L.integrity()
 if o.mode=='validate':
  from validate_synthetic import fixture
  args=fixture();start=time.monotonic();reference=L.evaluate(*args);serial=time.monotonic()-start
  payload,routing=parallel(args)
  a=pd.DataFrame(reference[0]);b=pd.DataFrame(payload[0]);need(list(a.component)==list(b.component),'Row order changed')
  diff=float(np.max(np.abs(a[ARMS].to_numpy()-b[ARMS].to_numpy())));need(diff<1e-9,'Hybrid prediction mismatch')
  lossdiff=0.
  for u,v in zip(reference[2],payload[2]):
   need(u['penalty']==v['penalty'] and u['panel']==v['panel'],'Tuning mismatch');lossdiff=max(lossdiff,float(np.max(np.abs(np.array(u['inner_MSE'])-v['inner_MSE']))))
  metricdiff=0.
  for k,ms in reference[1].items():
   for name,u in ms.items():
    v=payload[1][k][name];need(u['valid']==v['valid'],'Bootstrap count mismatch')
    for field in ['estimate','interval']:
     if u[field] is None:need(v[field] is None,'Undefined metric mismatch')
     else:metricdiff=max(metricdiff,float(np.max(np.abs(np.asarray(u[field])-v[field]))))
  need(lossdiff<1e-10 and metricdiff<1e-9,'Hybrid numerical mismatch')
  L.publish(D/'qa/hybrid_synthetic_v1',payload,'synthetic');pd.DataFrame(routing).to_csv(E/'Synthetic_Routing.csv',index=False)
  result=dict(status='PASS',synthetic_serial_fits=1920,synthetic_hybrid_fits=1920,cpu_fits=payload[3]['cpu_fits'],gpu_fits=payload[3]['gpu_fits'],max_prediction_difference=diff,max_inner_MSE_difference=lossdiff,max_metric_and_interval_difference=metricdiff,serial_seconds=serial,hybrid_seconds=payload[3]['parallel_elapsed_seconds'],study_fits=0)
  (E/'Hybrid_Validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
 L.authorize();seal=json.loads((E/'EXECUTION_LOCK.json').read_text());need(seal['analysis_lock_sha256']==L.sha(D/'LOCK.json') and all(L.sha(E/p)==h for p,h in seal['files'].items()),'Hybrid execution seal mismatch')
 with (E/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);need(not (D/'study_results').exists(),'Existing study output');start=time.monotonic()
  payload,routing=parallel(L.inputs());L.publish(D/'study_results',payload,'study');pd.DataFrame(routing).to_csv(E/'Study_Routing.csv',index=False)
  result=dict(status='COMPLETE',**payload[3],total_seconds=time.monotonic()-start)
  (E/'Run_Record.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
