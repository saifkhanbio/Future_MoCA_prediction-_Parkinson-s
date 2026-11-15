"""Synthetic integration, numerical, leakage and rejection tests."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from launcher import D,run,validate,predict_fold,design,ridge,require

def exercise():
    plan=json.loads((D/'protocol.json').read_text()); plan['bootstrap']=128
    rng=np.random.default_rng(71);n=90
    ids=['SYN%03d'%i for i in range(n)]
    x=pd.DataFrame(rng.normal(size=(n,27)),index=ids,columns=plan['panels']['full28']['numeric'])
    s=pd.DataFrame({'outer':np.arange(n)%5},index=ids)
    for j in range(5):s['inner_for_outer_'+str(j)]=np.where(s.outer==j,-1,(np.arange(n)//5+j)%3)
    x['baseline_moca']=rng.integers(18,31,n)
    for k,levels in plan['panels']['full28']['categories'].items():x[k]=rng.choice(levels[:-1],n)
    x.loc[ids[::4],'age_index']=np.nan;x.loc[ids[::7],'jlo_form']='unknown'
    rows=[]
    for h in plan['horizons']:
        for i,p in enumerate(ids):
            ix=2014*12+1 if i%3 else 2024*12+1
            rows.append(dict(component=p,horizon=h,index_month=ix,endpoint_month=ix+h,endpoint_row=1000*h+i,complete28=int(i%4!=0 and i%7!=0),historical=int(i%3!=0),y=np.clip(round(.7*x.at[p,'baseline_moca']+7+x.at[p,'SDMT']+rng.normal()),0,30)))
    data=pd.DataFrame(rows).set_index(['component','horizon']);m=data.drop(columns='y');y=data[['y','endpoint_row']];reserve={'SYN_HELD_OUT'}
    for v in plan['views']:
        plan['views'][v]['n']=[n if v=='expanded' else int(m.xs(h,level='horizon')[v].sum()) for h in plan['horizons']]
    out,stats,tuning=run(plan,x,m,y,s,reserve)
    assert len(stats)==9 and len(tuning)==195 and len(out)==sum(sum(v['n']) for v in plan['views'].values())
    for k,v in stats.items():
        view,h=k.split('/');rows=[r for r in out if r['view']==view and r['horizon']==int(h)]
        delta=np.mean([abs(r['compact4']-r['observed'])-abs(r['persistence']-r['observed']) for r in rows])
        assert abs(delta-v['compact4-persistence/delta_MAE']['estimate'])<1e-11
        assert all(r['persistence']==x.at[r['component'],'baseline_moca'] for r in rows)
    target=y.xs(12,level='horizon').y; panel=plan['panels']['core'];grid=plan['penalties']
    test,p,alpha,loss=predict_fold(x,target,s,0,panel,grid)
    changed=target.copy();changed.loc[test]=0
    _,p2,alpha2,loss2=predict_fold(x,changed,s,0,panel,grid)
    assert alpha==alpha2 and np.array_equal(p,p2) and loss==loss2
    ti=s.index[s.outer!=0];z=x.copy();z.loc[test,'age_index']=99999
    a,b=design(x,ti,test,panel);a2,_=design(z,ti,test,panel);assert np.array_equal(a,a2)
    oracle=Ridge(alpha=len(ti)*alpha,solver='svd').fit(a,target.loc[ti]).predict(b)
    maxerr=float(np.max(np.abs(oracle-p)));assert maxerr<1e-9
    # Check exact ties and a train-all-missing predictor with observed test entries.
    tied=target.copy();tied[:]=25
    z.loc[ti,'age_index']=np.nan
    _,pc,alpha3,_=predict_fold(z,tied,s,0,panel,grid)
    assert alpha3==max(grid) and np.all(pc==25)
    rejects=[]
    def reject(label,xx=x,mm=m,yy=y,ss=s,rv=reserve):
        try:validate(plan,xx,mm,yy,ss,rv)
        except ValueError:rejects.append(label);return
        raise AssertionError('Fault accepted: '+label)
    reject('reserved identity',rv={ids[0]});reject('duplicated endpoint',yy=pd.concat([y,y.iloc[:1]]))
    yy=y.copy();yy.iloc[0,yy.columns.get_loc('endpoint_row')]+=1;reject('misbound outcome',yy=yy)
    ss=s.copy();ss.loc[test[0],'inner_for_outer_0']=1;reject('outer test in tuning',ss=ss)
    xx=x.copy();xx.iloc[0,xx.columns.get_loc('baseline_sdmt_form')]='unknown';reject('unsupported baseline form',xx=xx)
    mm=m.copy();mm.iloc[0,mm.columns.get_loc('complete28')]=1;reject('incorrect complete panel',mm=mm)
    mm=m.copy();mm.iloc[0,mm.columns.get_loc('historical')]=1;reject('calendar leakage',mm=mm)
    mm=m.copy();mm.iloc[0,mm.columns.get_loc('endpoint_month')]+=5;reject('visit out of window',mm=mm)
    return dict(status='PASS',fixture_people=n,bootstrap_draws_per_view=128,synthetic_ridge_fits=3120,study_fits=0,views_and_horizons=len(stats),faults_rejected=rejects,independent_sklearn_max_abs_error=maxerr,heldout_labels_do_not_change_tuning_or_prediction=True)
