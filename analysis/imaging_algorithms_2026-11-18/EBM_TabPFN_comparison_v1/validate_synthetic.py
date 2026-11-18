import json,copy,subprocess,sys
import numpy as np,pandas as pd
import launcher as L

def exercise():
    rng=np.random.default_rng(71);n=90;ids=['SYN%03d'%i for i in range(n)];p=copy.deepcopy(L.P);p['bootstrap']=64
    x=pd.DataFrame(rng.normal(size=(n,27)),index=ids,columns=p['panels']['full28']['numeric']);x['baseline_moca']=rng.integers(18,31,n)
    for k,levels in p['panels']['full28']['categories'].items():x[k]=rng.choice(levels[:-1],n)
    s=pd.DataFrame({'outer':np.arange(n)%5},index=ids)
    for j in range(5):s['inner_for_outer_'+str(j)]=np.where(s.outer==j,-1,(np.arange(n)//5+j)%3)
    rows=[]
    for h in p['horizons']:
        for i,person in enumerate(ids):rows.append(dict(component=person,horizon=h,index_month=24169,endpoint_month=24169+h,endpoint_row=h*1000+i,complete28=1,historical=1,y=np.clip(round(.7*x.at[person,'baseline_moca']+7+x.at[person,'SDMT']+rng.normal()),0,30)))
    z=pd.DataFrame(rows).set_index(['component','horizon']);m=z.drop(columns='y');y=z[['y','endpoint_row']]
    sp=json.loads((L.D/'source_snapshot/protocol.json').read_text())
    for v in sp['views']:sp['views'][v]['n']=[n]*3
    L.source.validate(sp,x,m,y,s,set())
    rejected=[]
    for name in ['reserved','endpoint','fold']:
        yy=y.copy();ss=s.copy();reserve=set()
        if name=='reserved':reserve={ids[0]}
        if name=='endpoint':yy.iloc[0,1]+=1
        if name=='fold':ss.loc[ids[0],'inner_for_outer_0']=0
        try:L.source.validate(sp,x,m,yy,ss,reserve)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    # Full topology exercised with a marked deterministic stub, never misreported as real model fits.
    actual=L.estimate;calls=[]
    def stub(alg,c,a,b,target,seed):
        calls.append(alg);return L.source.ridge(a,b,target,1.)
    L.estimate=stub
    out,metrics,tune,fitcalls=L.evaluate(p,x,m,y,s,set())
    assert fitcalls==2475 and len(calls)==2475 and len(tune)==225 and len(out)==540
    for view in p['views']:
        for h in p['horizons']:
            rr=[r for r in out if r['view']==view and r['horizon']==h]
            assert len(rr)==n and len({r['component'] for r in rr})==n
    # Independent metric arithmetic for matched predictions.
    rr=[r for r in out if r['view']=='complete28' and r['horizon']==12]
    calc=np.mean([abs(r['compact4__EBM']-r['observed'])-abs(r['compact4__ridge']-r['observed']) for r in rr])
    assert abs(calc-metrics['co_primary_adjusted']['EBM']['delta_MAE'])<1e-12
    L.estimate=actual
    target=y.xs(12,level='horizon').y;real={}
    for alg in p['algorithms']:
        grid=p['grids'][alg]
        ix,pr,sel,loss=L.fold(x,target,s,0,p['panels']['compact4'],alg,grid,711)
        changed=target.copy();changed.loc[ix]=0
        ix2,pr2,sel2,loss2=L.fold(x,changed,s,0,p['panels']['compact4'],alg,grid,711)
        assert sel==sel2 and np.allclose(pr,pr2,atol=1e-8,rtol=1e-8) and np.allclose(loss,loss2,atol=1e-8)
        real[alg]={'synthetic_fit_calls':2*(3*len(grid)+1),'finite_predictions':True,'heldout_label_invariance':True,'repeat_max_difference':float(np.max(np.abs(pr-pr2)))}
    extra=[]
    for panel in ['enhanced','full28']:
        tr=s.index[s.outer.ne(0)];te=s.index[s.outer.eq(0)]
        a,b=L.source.design(x,tr,te,p['panels'][panel])
        for alg in p['algorithms']:
            pred=L.estimate(alg,p['grids'][alg][0],a,b,target.loc[tr].to_numpy(),177)
            assert len(pred)==len(te) and np.isfinite(pred).all()
            extra.append({'panel':panel,'algorithm':alg,'encoded_columns':a.shape[1]})
    # Train-only preprocessing cannot change when heldout predictors change.
    train=s.index[s.outer.ne(0)];test=s.index[s.outer.eq(0)];xx=x.copy();xx.loc[test,'SDMT']=1e9
    a,b=L.source.design(x,train,test,p['panels']['compact4']);a2,_=L.source.design(xx,train,test,p['panels']['compact4']);assert np.array_equal(a,a2)
    from sklearn.linear_model import Ridge
    oracle=Ridge(alpha=len(train),solver='svd').fit(a,target.loc[train]).predict(b)
    diff=float(np.max(np.abs(oracle-L.source.ridge(a,b,target.loc[train].to_numpy(),1.))));assert diff<1e-9
    run=subprocess.run([sys.executable,str(L.D/'launcher.py'),'--mode','run-study'],capture_output=True,text=True);assert run.returncode!=0 and 'not authorized' in run.stderr
    report={'status':'PASS','study_fits':0,'fixture_people':n,'structural_stub_fit_calls':fitcalls,'real_backend_tests':real,'additional_panel_smoke_tests':extra,'faults_rejected':rejected,'ridge_svd_max_difference':diff,'preprocessing_train_only':True,'study_execution_blocked':True}
    (L.D/'Synthetic_Validation.json').write_text(json.dumps(report,indent=2));return report
