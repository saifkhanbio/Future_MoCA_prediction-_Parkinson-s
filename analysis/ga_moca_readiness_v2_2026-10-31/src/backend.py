"""Cached float64 fitness engine; one CUDA owner, atomic mask records and explicit hardware fallback."""
from support import *
_TORCH=None
def gpu_ready(force_unavailable=False):
    global _TORCH
    if force_unavailable:raise HardwareFailure('Synthetic CUDA-unavailable injection')
    if _TORCH is None:
        try:
            import torch
            if not torch.cuda.is_available():raise HardwareFailure('CUDA unavailable')
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.empty(1,dtype=torch.float64,device='cuda');torch.cuda.synchronize()
            _TORCH=torch
        except ImportError as ex:raise HardwareFailure('CUDA backend unavailable: '+str(ex)) from ex
        except HardwareFailure:raise
        except RuntimeError as ex:
            if is_hardware(ex):raise HardwareFailure(str(ex)) from ex
            raise
    return {'available':True,'device':_TORCH.cuda.get_device_name(0),'pid':os.getpid()}
def is_hardware(ex):
    text=str(ex).lower()
    return isinstance(ex,HardwareFailure) or any(s in text for s in ('cuda out of memory','cuda error: out of memory','cuda driver initialization failed','cuda error: initialization error','cuda error: device is busy or unavailable','no cuda gpus are available'))
class Fitness:
    def __init__(self,spec,c,store,backend='cpu',batch=256,faults=None,initialize_device=True):
        self.spec=spec;self.c=c;self.store=store;self.batch=batch;self.faults=faults or {}
        if self.faults and not spec['synthetic']:raise IntegrityError('Fault injection forbidden in production')
        self.kind=store.get('backend',backend);self.cache=store.records()
        if backend=='cpu' and self.kind=='gpu':self.fallback('Coordinator rerouted this context to CPU')
        reference=frozen.Fitness(spec['train_rows'],spec['train_y'],spec['inner'],c)
        self.parts=reference.parts;self.variance=reference.variance
        self.prepared=[]
        for X,V,y,z,labels in self.parts:
            xm=X.mean(0);ym=float(y.mean());Z=X-xm
            self.prepared.append((Z.T@Z,Z.T@(y-ym),V-xm,ym,z,len(y),labels))
        self.attempted=0;self.completed=0;self.batches=0
        if self.kind=='gpu' and initialize_device:
            try:gpu_ready(self.faults.get('gpu_unavailable',False))
            except HardwareFailure as ex:self.fallback(str(ex))
        self.store.set('backend',self.kind)
        for r in self.cache.values():self.validate_record(r)
    def fallback(self,reason):
        log=self.store.get('fallbacks',[]);log.append({'reason':reason,'from':self.kind,'to':'cpu','pid':os.getpid()})
        self.kind='cpu';self.store.set('backend','cpu');self.store.set('fallbacks',log)
    def validate_record(self,r):
        required={'mask','fitness_schema','mse','r2','normalized_mse','outcome_variance','pooled_sst','burden','scores','alpha','fold_mse','all_fold_mse','all_fold_sse','lambda_grid_mse','lambda_grid_r2','fold_sizes'}
        if set(r)!=required or r.get('fitness_schema')!='mse_r2_v1':raise IntegrityError('Wrong fitness schema; MAE caches cannot be reused')
        d=len(self.c['features']['optional']);m=r['mask']
        if not isinstance(m,int) or not 0<=m<2**d:raise IntegrityError('Bad cached mask')
        fold=np.asarray(r['all_fold_mse'],dtype=float);sse=np.asarray(r['all_fold_sse'],dtype=float);sizes=np.asarray(r['fold_sizes'])
        if fold.shape!=(5,3) or sse.shape!=(5,3) or not np.isfinite(fold).all() or not np.isfinite(sse).all() or (fold<0).any() or (sse<0).any() or not np.array_equal(sizes,[len(p[4]) for p in self.prepared]):raise IntegrityError('Invalid cached fold record')
        grid=np.asarray(r['lambda_grid_mse'],dtype=float);rr=np.asarray(r['lambda_grid_r2'],dtype=float)
        if grid.shape!=(5,) or rr.shape!=(5,) or not np.isfinite(grid).all() or not np.isfinite(rr).all():raise IntegrityError('Invalid squared-error grid')
        if not np.allclose(fold*sizes,sse,atol=1e-10,rtol=1e-12) or not np.allclose(sse.sum(1)/sizes.sum(),grid,atol=1e-12,rtol=1e-12):raise IntegrityError('Invalid squared-error pooling')
        j=max((i for i,v in enumerate(grid) if v<=min(grid)+1e-12),key=lambda i:self.c['model']['lambdas'][i])
        expected=[self.variance,self.variance*sum(sizes),float(grid[j]/self.variance),float(1-grid[j]/self.variance)]
        actual=[r['outcome_variance'],r['pooled_sst'],r['normalized_mse'],r['r2']]
        if not np.allclose(actual,expected,atol=1e-12,rtol=1e-12) or not np.allclose(rr,1-grid/self.variance,atol=1e-12,rtol=1e-12):raise IntegrityError('Invalid pooled R2 binding')
        if r['alpha']!=self.c['model']['lambdas'][j] or r['mse']!=grid[j] or r['burden']!=frozen.burden(m,self.c) or r['scores']!=len(frozen.selected(m,self.c)) or not np.array_equal(r['fold_mse'],fold[j]):raise IntegrityError('Cached MSE choice or mask binding mismatch')
    def pack(self,masks,sums):
        sizes=np.array([len(p[4]) for p in self.prepared]);out=[]
        if not np.isfinite(sums).all() or (sums<0).any():raise IntegrityError('Nonfinite or negative squared-error fitness')
        for i,m in enumerate(masks):
            grid=sums[i].sum(1)/sum(sizes);fold=sums[i]/sizes[None,:]
            j=max((j for j,v in enumerate(grid) if v<=min(grid)+1e-12),key=lambda j:self.c['model']['lambdas'][j])
            r={'mask':int(m),'fitness_schema':'mse_r2_v1','mse':float(grid[j]),'r2':float(1-grid[j]/self.variance),'normalized_mse':float(grid[j]/self.variance),'outcome_variance':self.variance,'pooled_sst':self.variance*int(sum(sizes)),'burden':frozen.burden(m,self.c),'scores':len(frozen.selected(m,self.c)),'alpha':self.c['model']['lambdas'][j],'fold_mse':fold[j].tolist(),'all_fold_mse':fold.tolist(),'all_fold_sse':sums[i].tolist(),'lambda_grid_mse':grid.tolist(),'lambda_grid_r2':(1-grid/self.variance).tolist(),'fold_sizes':sizes.tolist()}
            self.validate_record(r);out.append(r)
        return out
    def cpu(self,masks):
        sums=np.empty((len(masks),5,3))
        for i,m in enumerate(masks):
            ix=frozen.column_indices(m,self.c,self.prepared[0][-1])
            for k,(G,b,V,ym,z,n,labels) in enumerate(self.prepared):
                g=G[np.ix_(ix,ix)];rhs=b[ix];eye=np.eye(len(ix))
                for j,a in enumerate(self.c['model']['lambdas']):
                    self.attempted+=1;coef=np.linalg.solve(g+n*a*eye,rhs);p=V[:,ix]@coef+ym
                    if not np.isfinite(p).all():raise IntegrityError('Nonfinite CPU prediction')
                    self.completed+=1;sums[i,j,k]=np.square(z-p).sum()
        return self.pack(masks,sums)
    def gpu(self,masks):
        gpu_ready();torch=_TORCH
        from collections import defaultdict
        groups=defaultdict(list)
        for i,m in enumerate(masks):
            ix=frozen.column_indices(m,self.c,self.prepared[0][-1]);groups[len(ix)].append((i,ix))
        sums=torch.empty((len(masks),5,3),dtype=torch.float64,device='cuda')
        alpha=torch.tensor(self.c['model']['lambdas'],dtype=torch.float64,device='cuda')
        for k,(G,b,V,ym,z,n,labels) in enumerate(self.prepared):
            Gt=torch.as_tensor(G,dtype=torch.float64,device='cuda');bt=torch.as_tensor(b,dtype=torch.float64,device='cuda')
            Vt=torch.as_tensor(V,dtype=torch.float64,device='cuda');zt=torch.as_tensor(z,dtype=torch.float64,device='cuda')
            for width,entries in sorted(groups.items()):
                dest=torch.tensor([i for i,ix in entries],dtype=torch.long,device='cuda')
                ix=torch.tensor(np.stack([ix for i,ix in entries]),dtype=torch.long,device='cuda')
                A=Gt[ix[:,:,None],ix[:,None,:]][:,None,:,:]+n*alpha[None,:,None,None]*torch.eye(width,dtype=torch.float64,device='cuda')[None,None,:,:]
                rhs=bt[ix][:,None,:,None].expand(len(entries),5,width,1).contiguous()
                self.attempted+=len(entries)*5
                beta,info=torch.linalg.solve_ex(A,rhs,check_errors=False)
                p=torch.bmm(beta.squeeze(-1),Vt[:,ix].permute(1,2,0))+ym
                if not bool((info==0).all().item()) or not bool(torch.isfinite(p).all().item()):raise IntegrityError('Invalid GPU numerical solution')
                self.completed+=len(entries)*5;sums[dest,:,k]=torch.square(p-zt[None,None,:]).sum(-1)
                if self.faults.get('gpu_oom_after_solves') and self.completed>=self.faults['gpu_oom_after_solves']:raise HardwareFailure('Synthetic CUDA out of memory after partial batch')
        result=sums.cpu().numpy();torch.cuda.synchronize()
        return self.pack(masks,result)
    def evaluate(self,masks,stop=lambda:False):
        missing=list(dict.fromkeys(int(m) for m in masks if int(m) not in self.cache))
        for m in missing:frozen.selected(m,self.c)
        for start in range(0,len(missing),self.batch):
            if stop():raise Paused('Stop requested before fitness batch')
            chunk=missing[start:start+self.batch];self.attempted=0;self.completed=0
            event=self.store.event(self.kind+'_fitness',len(chunk)*15)
            try:
                records=self.gpu(chunk) if self.kind=='gpu' else self.cpu(chunk)
                if self.faults.get('nonfinite'):raise IntegrityError('Synthetic nonfinite result injection')
                if self.attempted!=15*len(chunk) or self.completed!=self.attempted:raise IntegrityError('Fitness solve ledger mismatch')
                self.store.finish_event(event,'COMPLETE',self.attempted,self.completed,records=records)
            except Exception as ex:
                self.store.finish_event(event,'FAILED_HARDWARE' if self.kind=='gpu' and is_hardware(ex) else 'FAILED',self.attempted,self.completed,str(ex))
                if self.kind=='gpu' and is_hardware(ex):
                    self.fallback(str(ex));self.evaluate(chunk,stop);records=[self.cache[m] for m in chunk]
                else:raise
            for r in records:self.cache[r['mask']]=r
            self.batches+=1
            if self.faults.get('crash_after_batch')==self.batches:os._exit(71)
        return [self.cache[int(m)] for m in masks]
    def __call__(self,mask):return self.evaluate([mask])[0]
