"""Verify locked PPMI/PDBP results from original source states and raw-unit equations.
The independent estimates and bootstrap use no launcher statistical implementation.
Production forecasting functions are imported solely for paired numerical parity checks.
"""
import csv, hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'Stage_AB_protocols_v1_2026-11-15'
C=ROOT.parents[2]
AUD=OUT.parent/'Stage_AB_eligibility_2026-11-15'
PIN='c19e441103b6752adc4fcf6c89dd79dc773c6538aaaeb5b8fb30f96d63835491'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,what):
    if not ok:raise ValueError(what)
def inventory():
    check(sha(ROOT/'LOCK.json')==PIN,'protocol digest')
    lock=read(ROOT/'LOCK.json')
    for group in ('public','restricted'):
        for f,want in lock[group].items():check(sha(ROOT/f)==want,'locked '+f)
    sources=read(ROOT/'source_hashes.json')
    for f,want in sources.items():check(sha(C.parent/f)==want,'source '+f)
    return len(lock['public']),len(lock['restricted']),len(sources)
def feature_forecast(x,arm,h):
    if arm=='persistence':return x['baseline_moca']
    if arm=='typical':return x['baseline_moca']+offset[h]
    if arm=='reduced3':
        return math.fsum([float(equations[h]['intercept_raw'])]+[x[k]*float(equations[h][k+'_coefficient_raw']) for k in ('baseline_moca','age_years','sex_male')])
    members=[]
    for fold in range(5):
        m=stateA[f'{arm}_{h}m_fold{fold}']; num=m['preprocessing']['numeric']; terms=[m['intercept']]
        for k,weight in zip(m['columns'],m['coef']):
            if '==' in k:
                field,val=k.split('=='); term=int(x[field]==val)
            elif k.endswith('__missing'):term=int(x[k[:-9]] is None)
            else:
                q=num[k]; raw=x[k] if x[k] is not None else q['median']
                term=(raw-q['mean'])/q['sd']
            terms.append(weight*term)
        members.append(math.fsum(terms))
    return math.fsum(members)/5

def reference(ix,y,preds):
    """Calculate by sampled paired errors, with direct differences of individual losses."""
    obs=y[ix]; dev=obs-obs.mean(1)[:,None]; sst=np.einsum('ij,ij->i',dev,dev)
    ans={}; ae={}; se={}
    for name,p in preds.items():
        vals=p[ix]; err=vals-obs; ae[name]=np.abs(err);se[name]=np.square(err)
        centered=vals-vals.mean(1)[:,None]; ss=np.einsum('ij,ij->i',centered,centered)
        with np.errstate(divide='ignore',invalid='ignore'):
            sl=np.where(np.ptp(vals,axis=1)>1e-12,np.einsum('ij,ij->i',dev,centered)/ss,np.nan)
            r2=np.where(sst>0,1-se[name].sum(1)/sst,np.nan)
        d=dict(MAE=ae[name].mean(1),RMSE=np.sqrt(se[name].mean(1)),R2=r2,bias=err.mean(1),
               calibration_intercept=obs.mean(1)-sl*vals.mean(1),calibration_slope=sl,outside_0_30=np.mean((vals<0)|(vals>30),1))
        ans.update({name+'/'+k:v for k,v in d.items()})
    for a,b in itertools.combinations(preds,2):
        prefix=a+'-'+b+'/'
        ans[prefix+'delta_MAE']=(ae[a]-ae[b]).mean(1)
        with np.errstate(divide='ignore',invalid='ignore'):
            ans[prefix+'delta_R2']=np.where(sst>0,(se[b]-se[a]).sum(1)/sst,np.nan)
    return ans
def quantile(ordered,q):
    location=(len(ordered)-1)*q;j=int(location);f=location-j
    return (1-f)*ordered[j]+f*ordered[min(j+1,len(ordered)-1)]

sys.path.insert(0,str(ROOT))
from evaluation import forecast_a,forecast_b
start=inventory()
spec=read(ROOT/'model_spec.json')
asrc=C/'independent_validation_contract_2026-10-31'
stateA={r['member_id']:r['state'] for r in read(asrc/'restricted/Frozen_Inference_States.json')}
equations={int(r['horizon_months']):r for r in rows(C/'manuscript_supplement/Table_S14_Frozen_Reduced_Model_Equations.csv') if r['recipe']=='reduced3'}
offA=read(asrc/'config/Frozen_Benchmark_Binding.json')['offsets']
stats={};export=[];maxerr=0.;maxpred=0.;source_outputs={}
for stage in ['A','B']:
    plan=read(ROOT/f'Protocol_{stage}.json');restricted=ROOT/'restricted'
    outpath=restricted/f'Study_results_{stage}.json';source_outputs[stage]=sha(outpath);saved=read(outpath)
    check(saved['protocol_lock']==PIN and saved['stage']==stage,'result lock')
    xx=read(restricted/f'features_{stage}.json'); yy=read(restricted/f'outcomes_{stage}.json')
    actual_states=read(restricted/f'models_{stage}.json')
    offset=({h:math.fsum(r['already_frozen_offset']*r['weight'] for r in offA if r['horizon']==h) for h in (12,24,36)} if stage=='A' else {h:float(r['PPMI_mean_change']) for h,r in equations.items()})
    outcomes={(r['h'],r['id']):r['y'] for r in yy};membership={(r['h'],r['id']) for r in xx}
    check(len(membership)==len(xx)==len(yy) and membership==set(outcomes),'outcome identity join')
    ledger=rows(AUD/f'restricted/{stage}_membership.csv')
    expected={(int(r['horizon']),r['component']) for r in ledger if stage=='A' or r['identity_clear']=='True'}
    check(expected==membership,'audited membership')
    stagecount=0;parities=0;nviews=0;statnames=set()
    for h in plan['horizons']:
        for view,v in plan['views'].items():
            name=f'{view}/{h}';statnames.add(name);raw=saved['results'][name];n=raw['n']
            records=sorted([r for r in xx if r['h']==h and view in r['views']],key=lambda r:r['id'])
            check(n==len(records)==v['n'][plan['horizons'].index(h)],'view membership')
            y=np.array([outcomes[h,r['id']] for r in records],dtype=float);pred={}
            for model in v['models']:
                p=np.array([feature_forecast(r['x'],model,h) for r in records])
                if model in ('typical','persistence'):
                    actual=np.array([r['x']['baseline_moca'] for r in records])+(spec['offset_'+stage][str(h)] if model=='typical' else 0)
                elif stage=='A':actual=forecast_a(records,actual_states,model,h,spec,view.startswith('complete') or view=='compact')
                else:actual=forecast_b(records,actual_states[str(h)])
                diff=float(np.max(np.abs(actual-p)));check(diff<1e-10,'individual prediction')
                maxpred=max(maxpred,diff);parities+=len(p);pred[model]=p
            seed=plan['seed_base']+100*v['rank']+h
            check(raw['seed']==seed,'bootstrap seed')
            point=reference(np.arange(n)[None,:],y,pred)
            check(set(point)==set(raw['metrics']),'metric inventory')
            rng=np.random.Generator(np.random.PCG64(seed)); reps=plan['bootstrap']
            dist={k:[] for k in point}
            for i in range(0,reps,117):
                ix=rng.integers(n,size=(min(117,reps-i),n)); vals=reference(ix,y,pred)
                for k,arr in vals.items():dist[k].append(arr)
            for k,arr in dist.items():
                obs=float(point[k][0]); s=raw['metrics'][k];a=np.concatenate(arr);a=np.sort(a[np.isfinite(a)])
                ci=[quantile(a,.025),quantile(a,.975)] if len(a)>=.95*reps and math.isfinite(obs) else None
                check(len(a)==s['valid'] and (ci is None)==(s['interval'] is None) and math.isfinite(obs)==(s['estimate'] is not None),'undefined/count '+k)
                diffs=[]
                if math.isfinite(obs):diffs.append(abs(obs-s['estimate']))
                if ci is not None:diffs.extend(abs(np.array(ci)-np.array(s['interval'])))
                err=max(diffs,default=0);check(err<1e-9,'metric/interval '+k)
                maxerr=max(maxerr,err);stagecount+=1
                export.append(dict(stage=stage,view=view,horizon=h,n=n,measure=k,estimate=s['estimate'],low95=(s['interval'] or [None,None])[0],high95=(s['interval'] or [None,None])[1],valid_draws=s['valid'],planned_draws=reps))
            nviews+=1
    check(statnames==set(saved['results']),'view inventory')
    stats[stage]=dict(views=nviews,participant_forecast_comparisons=parities,all_estimates_and_intervals_checked=stagecount)
check(start==inventory(),'input preservation')
with (OUT/'Verified_results.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
result=dict(status='PASS',lock_sha256=PIN,stage=stats,max_absolute_forecast_disagreement=maxpred,max_absolute_point_or_interval_disagreement=maxerr,result_sha256=source_outputs,
            frozen_files_verified=dict(public=start[0],restricted=start[1],sources=start[2]),verification='Separate arithmetic from original Stage A states and original-unit Table S14 equations. Replayed every paired resample using locked seeds. All finite counts, undefined results, points and interval bounds compared. Production comparison for participant forecasts uses hash-bound forecast functions, since launchers persist aggregate outputs.',new_fits=0,paid_queries=0)
(OUT/'Verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
