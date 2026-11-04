from pathlib import Path
import sys,os,json,copy,math
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
C=Path(__file__).resolve().parents[1];D=C.parent/'model_readiness_2026-10-24';sys.path.insert(0,str(D/'src'))
import numpy as np
import engine
import independent_reference as ref
from bundle import verify_seal
verify_seal(D);assert not (C/'qa/Numerical_Checks.json').exists()
checks=[];largest=0.;pairs=0

def check(name,ok,detail=None):
 checks.append({'check':name,'passed':bool(ok),'detail':detail})
 if not ok:raise AssertionError(name)
def close(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=tol))
def states(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(states(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(states(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool):return close(a,b)
 return a==b
rng=np.random.default_rng(20261028)
for case,n,p in [('ordinary',40,9),('rank_deficient',25,12),('wide',9,15),('constant_outcome',30,8)]:
 X=rng.normal(size=(n,p))+np.arange(p);y=rng.normal(25,3,n)
 if case=='rank_deficient':X[:,2]=X[:,1];X[:,4]=1.;X[:,5]=0.
 if case=='constant_outcome':y[:]=26.
 ncl=p-3
 errors=[];gradient=[]
 for lc in [.001,.01,.1,1.,10.]:
  for lr in [.01,.1,1.,10.]:
   a,i=engine.solve(X,y,ncl,lc,lr);b,j=ref.augmented_svd_ridge(X,y,ncl,lc,lr);pairs+=1
   err=float(max(np.max(abs(a-b)),abs(i-j),np.max(abs(X@a+i-(X@b+j)))));errors.append(err);largest=max(largest,err)
   residual=X@a+i-y;pen=np.r_[np.repeat(lc,ncl),np.repeat(lr,p-ncl)];g=X.T@residual/n+pen*a;gradient.append(max(float(np.max(abs(g))),abs(float(residual.mean()))))
 check('Block-ridge SVD equivalence: '+case,max(errors)<1e-8,{'max_absolute_error':max(errors),'comparisons':20})
 check('Ridge first-order optimum: '+case,max(gradient)<1e-8,{'max_gradient':max(gradient)})
for lc in [.001,.01,.1,1,10]:
 X=rng.normal(size=(20,6));X[:,0]=1;y=rng.normal(size=20);a,i=engine.solve(X,y,6,lc,None);b,j=ref.augmented_svd_ridge(X,y,6,lc,None);pairs+=1
 check('Clinical-only SVD equivalence lambda='+str(lc),close(np.r_[a,i],np.r_[b,j]))
spec={'numeric':['observed','partial','all_missing','constant'],'categories':{'category':['a','b','unknown']}}
rows=[{'observed':x,'partial':v,'all_missing':'','constant':7,'category':'a' if x%2 else 'b'} for x,v in zip(range(1,9),['',2,3,'nan',5,6,float('inf'),8])]
state=engine.prepare_clinical(rows,spec);reference=ref.clinical_state(rows,spec);check('Clinical fitted state matches independent arithmetic',states(state,reference))
a,names=engine.transform_clinical(rows,state);b,n2=ref.clinical_design(rows,reference);check('Clinical training transform, all-missing and constant columns',close(a,b) and names==n2 and np.isfinite(a).all())
held=[{'observed':1e8,'partial':'','all_missing':999,'constant':8,'category':'unknown'}];before=copy.deepcopy(state);a,_=engine.transform_clinical(held,state);b,_=ref.clinical_design(held,reference);check('Held-out extremes cannot update clinical imputation/scaling',close(a,b) and state==before)
reverse=engine.prepare_clinical(list(reversed(rows)),spec);check('Clinical fitted state invariant to training row order',states(reverse,state))
counts=rng.poisson(50,size=(30,18));counts[:,4:7]=0;counts[:5,4]=10;counts[:6,5]=10;counts[:,6]=9;counts[:,7]=100;counts[:,8]=0
panel={'NPIPB8_index':9,'gene5_indices':[0,4,5,8,9],'gene5_names':['QA_g0','QA_g4','QA_g5','QA_g8','QA_g9'],'pathway_indices':[list(range(10)),list(range(10,14)),[4,6,7,8,10]],'pathway_names':['QA_valid','QA_too_small','QA_inactive']};gate={'minimum_raw_count':10,'minimum_training_fraction':.2,'minimum_training_samples':3,'minimum_population_sd':1e-12};lib=np.full(30,1e6)
for omit in [False,True]:
 state=engine.prepare_rna(counts,lib,panel,gate,omit);reference=ref.rna_state(counts,lib,panel,gate,omit);check('RNA train-only state agrees with independent construction; omit='+str(omit),states(state,reference))
 check('Expression frequency/count/variance boundaries; omit='+str(omit),not state['active'][4] and state['active'][5] and not state['active'][6] and not state['active'][7] and state['active'][9]!=omit)
 for arm in ['none','Genes5','Hallmark11','Genes5_Hallmark11']:
  a,names=engine.transform_rna(counts,lib,state,arm);b,n2=ref.rna_design(counts,lib,reference,arm);check('RNA fitted transform '+arm+' omit='+str(omit),close(a,b) and names==n2 and np.isfinite(a).all())
 before=copy.deepcopy(state);target=np.full((2,18),100000000);a,_=engine.transform_rna(target,[2e9,2e9],state,'Genes5_Hallmark11');b,_=ref.rna_design(target,[2e9,2e9],reference,'Genes5_Hallmark11');check('Held-out expression cannot activate/filter/refit RNA; omit='+str(omit),close(a,b) and state==before)
zero=engine.prepare_rna(np.zeros((30,18),int),lib,panel,gate);z,_=engine.transform_rna(np.ones((2,18),int),[1e6]*2,zero,'Genes5_Hallmark11');check('Entirely inactive RNA yields finite fixed zero columns',np.isfinite(z).all() and np.all(z==0) and z.shape==(2,8))
check('Library denominator uses all counts, not the selected panel',close(engine.log_cpm([[10,20]],[1000]),np.log2(1+np.array([[10000,20000]]))))
# Fit a tiny synthetic integrated pipeline and compare numerical solution; no participant data.
cs=engine.prepare_clinical(rows,spec);xc,_=engine.transform_clinical(rows,cs);rs=engine.prepare_rna(counts[:8],lib[:8],panel,gate);xr,_=engine.transform_rna(counts[:8],lib[:8],rs,'Genes5_Hallmark11');X=np.c_[xc,xr];y=np.arange(8.)+20
coef,intercept=engine.solve(X,y,xc.shape[1],.1,1.);beta,j=ref.augmented_svd_ridge(X,y,xc.shape[1],.1,1.);pairs+=1;check('Integrated fitted clinical-plus-RNA design matches independent SVD',close(np.r_[coef,intercept],np.r_[beta,j]))
result={'status':'NUMERICAL_QA_PASSED','passed':True,'checks':checks,'production_engine_synthetic_ridge_fits':engine.COUNTERS['ridge_solves'],'independent_synthetic_SVD_fits':pairs,'training_preprocessing_calls_on_artificial_data':dict(engine.COUNTERS),'largest_solver_difference':largest,'real_participant_fits':0,'production_started':False,'paid_queries':0,'reference':'Separate augmented-design SVD implementation and scalar preprocessing arithmetic; no production metric functions used by reference.'}
(C/'qa/Numerical_Checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['checks','reference']})[:-1]+',"checks_passed":'+str(len(checks))+'}')