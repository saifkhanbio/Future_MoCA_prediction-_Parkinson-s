"""Data-free extension checks; never import or execute study launchers."""
from pathlib import Path
import ast
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'analysis/imaging_algorithms_2026-11-18'

def functions(path,names,extra=None):
    source=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    ns={'np':np,'pd':pd}
    if extra:ns.update(extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def main():
    rng=np.random.default_rng(4817)
    a=rng.normal(size=(24,5));a[:,-1]=a[:,0];b=rng.normal(size=(8,5));y=rng.normal(size=24)
    packages=['DAT_SPECT_launcher_v1/numerical_core.py','DAT_SPECT_GA28_v1/numerical_core.py','EBM_TabPFN_comparison_v1/source_snapshot/launcher.py']
    kernels=[functions(EXT/p,['design','ridge']) for p in packages]
    for penalty in [.001,.01,.1,1,10]:
        mu=a.mean(0);u,s,vt=np.linalg.svd(a-mu,full_matrices=False)
        coef=vt.T@((s/(s*s+len(a)*penalty))*(u.T@(y-y.mean())))
        expected=y.mean()+(b-mu)@coef
        for k in kernels:np.testing.assert_allclose(k['ridge'](a,b,y,penalty),expected,atol=1e-10,rtol=1e-10)
    x=pd.DataFrame({'x':[1,np.nan,3,1000],'empty':[np.nan]*4,'form':['a','b','a','b']},index=['SYN0','SYN1','SYN2','SYN3'])
    original=x.copy(deep=True);panel={'numeric':['x','empty'],'categories':{'form':['a','b']}}
    designs=[]
    for k in kernels:
        aa,bb=k['design'](x,x.index[:3],x.index[3:],panel)
        np.testing.assert_allclose(bb[0,0],(1000-2)/np.sqrt(2/3));assert bb[0,1]==0 and bb[0,3]==1
        pd.testing.assert_frame_equal(x,original);designs.append((aa,bb))
    for aa,bb in designs[1:]:np.testing.assert_array_equal(aa,designs[0][0]);np.testing.assert_array_equal(bb,designs[0][1])
    metric=functions(EXT/'EBM_TabPFN_comparison_v1/source_snapshot/paired_metrics.py',['metrics'])['metrics']
    z=metric(np.array([[1.,2.,4.,5.]]),np.array([[1.5,2.5,3.5,5.5]]))
    for name,value in [('MAE',.5),('RMSE',.5),('R2',.9)]:np.testing.assert_allclose(z[name],value)
    # Shared EBM/TabPFN orchestration is exercised with a stub estimator, not those backends.
    traces=[]
    def estimate(alg,config,aa,bb,yy,seed):
        traces.append((alg,config,seed,len(aa),len(bb)))
        return np.full(len(bb),yy.mean())
    src=SimpleNamespace(design=kernels[-1]['design'])
    fold=functions(EXT/'EBM_TabPFN_comparison_v1/launcher.py',['fold'],{'source':src,'estimate':estimate})['fold']
    ids=['SYN%03d'%i for i in range(30)];xx=pd.DataFrame({'x':rng.normal(size=30)},index=ids)
    sp=pd.DataFrame({'outer':np.arange(30)%5},index=ids)
    for j in range(5):sp['inner_for_outer_'+str(j)]=np.where(sp.outer==j,-1,(np.arange(30)//5+j)%3)
    target=pd.Series(20.,index=ids);spec={'numeric':['x'],'categories':{}}
    ix,pred,selected,loss=fold(xx,target,sp,0,spec,'synthetic',[10,1,.1],701)
    assert list(ix)==list(sp.index[sp.outer==0]) and selected==0 and len(traces)==10
    np.testing.assert_array_equal(pred,20.);np.testing.assert_array_equal(loss,0.)
    # Clinical imaging tunes MAE; GA28 imaging tunes MSE; preserve their distinct contracts.
    for rel in ['DAT_SPECT_launcher_v1/numerical_core.py','DAT_SPECT_GA28_v1/numerical_core.py']:
        k=functions(EXT/rel,['predict_fold'],kernels[0]);out=k['predict_fold'](xx,target,sp,0,spec,[.001,.01,.1,1,10])
        assert out[2]==10;np.testing.assert_array_equal(out[1],20.)
        # Verify non-tied MAE/MSE selection against independent SVD predictions.
        yy=pd.Series(20+3*xx.x.to_numpy()+rng.normal(size=len(xx)),index=ids)
        grid=[.001,.01,.1,1,10];loss=np.zeros(5);train=sp.index[sp.outer!=0]
        for inner in range(3):
            ti=train[sp.loc[train,'inner_for_outer_0']!=inner];vi=train[sp.loc[train,'inner_for_outer_0']==inner]
            aa,bb=kernels[0]['design'](xx,ti,vi,spec);mu=aa.mean(0);uu,ss,vv=np.linalg.svd(aa-mu,full_matrices=False);yt=yy.loc[ti].to_numpy()
            for n,pen in enumerate(grid):
                coef=vv.T@((ss/(ss*ss+len(aa)*pen))*(uu.T@(yt-yt.mean())))
                err=yt.mean()+(bb-mu)@coef-yy.loc[vi].to_numpy()
                loss[n]+=np.square(err).sum() if 'GA28' in rel else np.abs(err).sum()
        loss/=len(train);actual=k['predict_fold'](xx,yy,sp,0,spec,grid)
        np.testing.assert_allclose(actual[3],loss,atol=1e-10,rtol=1e-10)
        assert actual[2]==max(pen for pen,err in zip(grid,loss) if err<=loss.min()+1e-12)

    print('PASS: 15 ridge/SVD comparisons; three training-only designs; metric checks; synthetic fold/tie checks. No EBM/TabPFN backend fits, private data, checkpoints, study runs or file writes.')

if __name__=='__main__':main()
