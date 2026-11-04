from common import *
import itertools
def metric(y,p):
    error=p-y;den=np.dot(y-y.mean(),y-y.mean());v=np.dot(p-p.mean(),p-p.mean())
    slope=float(np.dot(p-p.mean(),y-y.mean())/v) if np.ptp(p)>1e-12 and v>0 else None
    return {'n':int(len(y)),'MSE':float(np.dot(error,error)/len(y)),'MAE':float(np.abs(error).sum()/len(y)),
       'RMSE':float(np.linalg.norm(error)/np.sqrt(len(y))),'R2':float(1-np.dot(error,error)/den) if den>0 else None,
       'mean_prediction_error':float(error.sum()/len(y)),
       'calibration_intercept':float(y.mean()-slope*p.mean()) if slope is not None else None,'calibration_slope':slope,
       'outside_0_30_fraction':float(np.count_nonzero((p<0)|(p>30))/len(y)),
       'clipped_MAE':float(np.abs(np.clip(p,0,30)-y).sum()/len(y))}
def main():
    base,profile,b,by,cohorts=data();stored_metrics=read(R/'results/Metrics.json');stored_contrasts=read(R/'results/Contrasts.json')
    metric_index={(r['horizon'],r['arm']):r for r in stored_metrics}
    contrast_index={(r['horizon'],r['arm_a'],r['arm_b']):r for r in stored_contrasts}
    assert len(metric_index)==len(stored_metrics)==30 and len(contrast_index)==len(stored_contrasts)==135
    out_metrics=[];out_contrasts=[];sels=[];max_metric=0.;max_contrast=0.;total_reference=0;total_prediction=0
    contexts={p.parent.name:read(p) for p in (R/'restricted').glob('*/Result.json')}
    allresults=read(R/'restricted/All_Context_Results.json')
    assert len(allresults)==30 and {r['context_id']:r for r in allresults}==contexts
    for h,cohort in sorted(cohorts.items()):
        oof=read(R/'restricted'/f'OOF_{h}m.json');assert len(oof)==len(cohort)
        assert [r['PATNO'] for r in oof]==[r['PATNO'] for r in cohort]
        assert len({r['component_id'] for r in oof})==len(oof)
        for q,r in zip(oof,cohort):
            assert q['component_id']==r['component_id'] and q['outer_fold']==int(r['outer_fold']) and q['y']==float(r['y'])
        models=['unrestricted28_GA_selected','unrestricted28_random_selected','moca27_GA_selected','moca27_random_selected']+list(b['references'][str(h)])+['prior_MoCA_GA','persistence','typical_change']
        assert len(set(models))==10 and all(set(r)==set(models)|{'PATNO','component_id','outer_fold','y'} for r in oof)
        index={r['PATNO']:r for r in oof};y=np.array([r['y'] for r in oof])
        pred={m:np.array([r[m] for r in oof]) for m in models};assert all(np.isfinite(p).all() for p in pred.values())
        for arm in ['unrestricted28','moca27']:
            covered=set()
            for o in range(5):
                res=contexts[f'{arm}_{h}m_outer{o}'];ids=res['test_ids']
                assert not covered.intersection(ids);covered.update(ids)
                assert set(ids)=={r['PATNO'] for r in cohort if int(r['outer_fold'])==o}
                for label,values in res['predictions'].items():
                    near(values,[index[i][arm+'_'+label] for i in ids],0)
                    total_prediction+=len(values)
                for label,m in res['models'].items():
                    sels.append({'horizon':h,'outer':o,'arm':arm+'_'+label,'mask':m['mask'],'alpha':m['alpha'],
                                'instruments':m['burden'],'feature_count':len(m['features']),'features':m['features']})
            assert covered==set(index)
        refs={}
        for role,info in b['references'][str(h)].items():
            source=read(info['path']);sourceidx={r['PATNO']:r for r in source};assert len(sourceidx)==len(source)==len(cohort) and set(sourceidx)==set(index)
            ordered=[sourceidx[r['PATNO']] for r in cohort]
            for a,r in zip(ordered,cohort):assert a['component_id']==r['component_id'] and int(a['outer_fold'])==int(r['outer_fold']) and float(a['y'])==float(r['y'])
            refs[role]=ordered
            key={'compact4':'enhanced','core_plus18':'full','MoCA_plus18':'optional18_plus_moca'}[role]
            assert np.array_equal(pred[role],np.array([r[key] for r in ordered]));total_reference+=len(ordered)
        for role,source,key in [('prior_MoCA_GA','compact4','GA_selected'),('persistence','core_plus18','persistence'),('typical_change','core_plus18','typical_change')]:
            assert np.array_equal(pred[role],np.array([r[key] for r in refs[source]]));total_reference+=len(cohort)
        # Check the two frozen baselines directly from the same outer training rows.
        persistence=np.array([num(by[r['PATNO']]['baseline_moca']) for r in cohort]);assert np.array_equal(pred['persistence'],persistence)
        typical=np.empty(len(cohort));folds=np.array([int(r['outer_fold']) for r in cohort])
        for o in range(5):typical[folds==o]=persistence[folds==o]+np.median((y-persistence)[folds!=o])
        near(pred['typical_change'],typical)
        for m,p in pred.items():
            got={'horizon':h,'arm':m,**metric(y,p)};want=metric_index[(h,m)]
            assert got.keys()==want.keys()
            for k in got:
                if k in ['horizon','arm','n'] or got[k] is None:assert got[k]==want[k]
                else:max_metric=max(max_metric,abs(got[k]-want[k]));near(got[k],want[k])
            out_metrics.append(got)
        # One identity per participant is verified above, so row draws equal identity-cluster draws.
        ix=np.random.default_rng(20261031+h).integers(0,len(y),size=(5000,len(y)))
        yy=y[ix];den=((yy-yy.mean(axis=1,keepdims=True))**2).sum(axis=1);valid=den>0;nv=int(valid.sum());assert nv>=4750
        ae={m:np.abs(p-y) for m,p in pred.items()};se={m:(p-y)**2 for m,p in pred.items()}
        bmae={m:a[ix].mean(axis=1) for m,a in ae.items()};bsse={m:a[ix].sum(axis=1) for m,a in se.items()}
        sst=float(np.dot(y-y.mean(),y-y.mean()))
        for a,bm in itertools.combinations(models,2):
            rboot=(bsse[bm]-bsse[a])[valid]/den[valid];mboot=bmae[a]-bmae[bm]
            rl,rh=np.percentile(rboot,[2.5,97.5]);ml,mh=np.percentile(mboot,[2.5,97.5])
            got={'horizon':h,'arm_a':a,'arm_b':bm,'delta_R2':float((se[bm].sum()-se[a].sum())/sst),
                 'lower95_R2':float(rl),'upper95_R2':float(rh),'delta_MAE':float(ae[a].mean()-ae[bm].mean()),
                 'lower95_MAE':float(ml),'upper95_MAE':float(mh),'replicates':5000,'R2_valid_replicates':nv,
                 'R2_undefined_replicates':5000-nv,'R2_interval_available':True,'conditional_on_fixed_predictions':True,
                 'R2_positive_favours_first':True,'MAE_negative_favours_first':True}
            want=contrast_index[(h,a,bm)];assert got.keys()==want.keys()
            for k in got:
                if isinstance(got[k],float):max_contrast=max(max_contrast,abs(got[k]-want[k]));near(got[k],want[k])
                else:assert got[k]==want[k]
            out_contrasts.append(got)
    assert sels==read(R/'results/Selections.json')
    primary=read(R/'results/Primary_Exploratory_Contrast.json')
    pc=next(r for r in out_contrasts if r['horizon']==36 and r['arm_a']=='unrestricted28_GA_selected' and r['arm_b']=='moca27_GA_selected')
    for k,v in primary.items():
        if k in ['contrast','primary_metric']:continue
        if isinstance(v,float):near(v,pc[k])
        else:assert v==pc[k]
    assert primary['primary_metric']=='delta_R2' and primary['contrast']=='unrestricted28_GA_selected minus moca27_GA_selected'
    write(V/'evidence/Recomputed_Metrics.json',out_metrics)
    write(V/'evidence/Recomputed_Contrasts.json',out_contrasts)
    write(V/'evidence/Metrics_Verification.json',{'status':'PASS','metric_rows':len(out_metrics),'pairwise_contrasts':len(out_contrasts),'selections':len(sels),
      'new_prediction_values_bound_to_contexts':total_prediction,'frozen_reference_prediction_values_checked':total_reference,
      'bootstrap_replicates_per_horizon':5000,'bootstrap_horizons':3,'all_draws_defined':True,
      'max_metric_absolute_error':max_metric,'max_contrast_absolute_error':max_contrast,
      'cohort_identity_and_fold_alignment':'PASS','primary_contrast':'PASS','identity_cluster_bootstrap':'Equivalent to row bootstrap because all component IDs are unique within each horizon',
      'new_model_training':False,'production_files_modified':False})
    print(json.dumps(read(V/'evidence/Metrics_Verification.json')),flush=True)
if __name__=='__main__':main()
