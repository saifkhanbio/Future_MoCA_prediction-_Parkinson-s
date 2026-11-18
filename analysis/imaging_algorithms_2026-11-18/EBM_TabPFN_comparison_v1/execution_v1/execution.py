import os,sys,json,time,hashlib,copy,fcntl
from pathlib import Path
E=Path(__file__).resolve().parent;D=E.parent
sys.path.insert(0,str(D))
import launcher as L
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
R=E/'results';C=R/'restricted/checkpoints'
def write(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));os.replace(tmp,p)
def gpu_estimate(config,a,b,y,seed):
    import torch
    from tabpfn import TabPFNRegressor
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    settings=dict(L.P['TabPFN_fixed']);settings.update(device='cuda',inference_precision=torch.float32)
    model=TabPFNRegressor(**settings,**config,model_path=D/L.P['checkpoint'],random_state=seed)
    model.fit(a,y);pr=np.asarray(model.predict(b),dtype=float)
    if not np.isfinite(pr).all():raise ValueError('Invalid GPU prediction')
    return pr

def ridge_gpu(a,b,y,penalty):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    aa=torch.as_tensor(a,device='cuda',dtype=torch.float64);bb=torch.as_tensor(b,device='cuda',dtype=torch.float64);yy=torch.as_tensor(y,device='cuda',dtype=torch.float64)
    mean=aa.mean(0);centered=aa-mean
    coef=torch.linalg.solve(centered.T@centered+torch.eye(aa.shape[1],device='cuda',dtype=torch.float64)*len(a)*penalty,centered.T@(yy-yy.mean()))
    return (yy.mean()+(bb-mean)@coef).cpu().numpy()
def validate_gpu():
    import torch
    L.verify();rng=np.random.default_rng(118);checks=[]
    for width in [17,39,72]:
        a=rng.normal(size=(90,width));a[:,-1]=1;b=rng.normal(size=(20,width));b[:,-1]=1;y=25+a[:,0]+rng.normal(size=90)
        for penalty in L.P['grids']['ridge']:
            cpu=L.source.ridge(a,b,y,penalty);gpu=ridge_gpu(a,b,y,penalty)
            error=float(np.max(np.abs(cpu-gpu)));assert error<=1e-9
            checks.append(dict(width=width,penalty=penalty,max_absolute_difference=error))
    write(E/'GPU_Validation.json',dict(status='PASS',algorithm='ridge FP64 only',tolerance_moca=1e-9,checks=checks,torch=torch.__version__,device=torch.cuda.get_device_name(),study_fits=0,TabPFN_GPU='REJECTED: synthetic agreement tolerance exceeded; locked CPU retained'))
    print(json.dumps(checks),flush=True)

def jobs():
    return [dict(view=v,horizon=h,panel=p,algorithm=a,outer=j) for v,vp in L.P['views'].items() for h in L.P['horizons'] for p in vp['panels'] for a in L.P['algorithms'] for j in range(5)]
def name(t):return '_'.join(str(t[k]) for k in ['view','horizon','panel','algorithm','outer'])
def worker(t):
    L.verify();_,x,m,y,s,reserve=L.source.inputs();v,h,p,alg,j=[t[k] for k in ['view','horizon','panel','algorithm','outer']]
    z=m.xs(h,level='horizon');z=z if v=='expanded' else z[z.complete28.eq(1)];ids=sorted(z.index)
    assert not set(ids)&reserve
    sp=s.loc[ids];target=y.xs(h,level='horizon').loc[ids,'y'];trace=[];original=L.estimate
    def recorded(a,c,aa,bb,yy,seed):
        pr=ridge_gpu(aa,bb,yy,c) if a=='ridge' else original(a,c,aa,bb,yy,seed)
        trace.append(dict(config=c,seed=seed,n_train=len(aa),n_predict=len(bb),train_design_sha256=hashlib.sha256(aa.tobytes()).hexdigest(),train_target_sha256=hashlib.sha256(yy.tobytes()).hexdigest(),prediction=pr.tolist()))
        return pr
    L.estimate=recorded;start=time.time()
    try:ix,pr,selected,loss=L.fold(x,target,sp,j,L.P['panels'][p],alg,L.P['grids'][alg],L.P['seed_base']+1000*h+10*j)
    finally:L.estimate=original
    answer=dict(task=t,ids=list(ix),prediction=pr.tolist(),selected=selected,inner_MAE=loss,trace=trace,fit_calls=len(trace),seconds=time.time()-start,device='cuda-float64' if alg=='ridge' else 'cpu')
    write(C/(name(t)+'.json'),answer)
    return name(t),len(trace)

def assemble():
    _,x,m,y,s,reserve=L.source.inputs();out=[];tuning=[];aggregate={};fits=0
    for v,vp in L.P['views'].items():
        for h in L.P['horizons']:
            z=m.xs(h,level='horizon');z=z if v=='expanded' else z[z.complete28.eq(1)];ids=sorted(z.index);truth=y.xs(h,level='horizon').loc[ids,'y'];pred={'persistence':x.loc[ids,'baseline_moca'].to_numpy()}
            for panel in vp['panels']:
                for alg in L.P['algorithms']:
                    arm=panel+'__'+alg;oof=pd.Series(np.nan,index=ids)
                    for j in range(5):
                        task=dict(view=v,horizon=h,panel=panel,algorithm=alg,outer=j);z=json.loads((C/(name(task)+'.json')).read_text());assert z['task']==task
                        assert z['ids']==list(s.loc[ids].index[s.loc[ids].outer.eq(j)])
                        oof.loc[z['ids']]=z['prediction'];fits+=z['fit_calls'];tuning.append({**task,'selected':z['selected'],'inner_MAE':z['inner_MAE']})
                    assert np.isfinite(oof).all();pred[arm]=oof.to_numpy()
            for i,person in enumerate(ids):out.append(dict(component=person,view=v,horizon=h,outer=int(s.at[person,'outer']),observed=float(truth[person]),**{a:float(z[i]) for a,z in pred.items()}))
            aggregate[v+'/'+str(h)]=L.summarize(truth.to_numpy(),pred,L.P['bootstrap'],L.P['seed_base']+h+100*(1+list(L.P['views']).index(v)))
            if v=='complete28' and h==12:
                rng=np.random.default_rng(L.P['seed_base']+112);draw=rng.integers(0,len(ids),size=(L.P['bootstrap'],len(ids)));base=np.abs(pred['compact4__ridge']-truth.to_numpy());adj={}
                for alg in ['EBM','TabPFN']:
                    diff=np.abs(pred['compact4__'+alg]-truth.to_numpy())-base;adj[alg]={'delta_MAE':float(diff.mean()),'interval_97_5':np.quantile(diff[draw].mean(1),[.0125,.9875]).tolist()}
                aggregate['co_primary_adjusted']=adj
    assert fits==2475
    pd.DataFrame(out).to_csv(R/'restricted/oof_predictions.csv',index=False);write(R/'metrics.json',aggregate);write(R/'tuning.json',tuning)
    L.verify();write(R/'STATUS.json',dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',fit_calls=fits,tasks=225))

def main():
    L.preflight();assert json.loads((E/'GPU_Validation.json').read_text())['status']=='PASS'
    for f,h in json.loads((E/'EXECUTION_LOCK.json').read_text())['files'].items():assert L.source.digest(E/f)==h
    R.mkdir(exist_ok=True);C.mkdir(parents=True,exist_ok=True)
    lock=(E/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (R/'metrics.json').exists():raise RuntimeError('Completed results already exist')
    todo=[t for t in jobs() if not (C/(name(t)+'.json')).exists()]
    start=time.time();completed=225-len(todo)
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as cpu,ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as gpu:
        fs={ (gpu if t['algorithm']=='ridge' else cpu).submit(worker,t):t for t in todo}
        for f in as_completed(fs):
            label,n=f.result();completed+=1
            write(R/'STATUS.json',dict(status='RUNNING',completed_tasks=completed,total_tasks=225,last_completed=label,elapsed_seconds=time.time()-start))
            print(completed,label,n,flush=True)
    assemble()
if __name__=='__main__':
    if sys.argv[1:] == ['validate-gpu']:validate_gpu()
    else:main()
