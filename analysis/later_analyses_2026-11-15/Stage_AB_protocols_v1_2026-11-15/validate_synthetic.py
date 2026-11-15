"""Independent closed-form checks against actual CLI synthetic outputs."""
import copy,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from fixtures import fixture
from evaluation import forecast_a,forecast_b,evaluate,summarize
from launch import digest,verify
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rejected(name,func):
    try:func()
    except (ValueError,KeyError):return name
    raise RuntimeError('Invalid fixture accepted: '+name)
def scalar_a(row,states,arm,h):
    members=[]
    for fold in range(5):
        s=states[f'{arm}_{h}m_fold{fold}'];terms=[]
        for col,w in zip(s['columns'],s['coef']):
            if '==' in col:
                k,level=col.split('==');v=int(row[k]==level)
            elif col.endswith('__missing'):v=int(row[col[:-9]] is None)
            else:
                q=s['preprocessing']['numeric'][col]
                v=((q['median'] if row[col] is None else row[col])-q['mean'])/q['sd']
            terms.append(v*w)
        members.append(s['intercept']+sum(terms))
    return sum(members)/5
def scalar_b(row,s):
    weights=[w/scale for w,scale in zip(s['coefficient'],s['scale'])]
    intercept=s['intercept']-sum(w*v for w,v in zip(weights,s['mean']))
    return intercept+sum(row[k]*w for k,w in zip(s['features'],weights))
def close(a,b):
    if not np.allclose(a,b,atol=2e-10,rtol=1e-12):raise AssertionError('Numerical disagreement')
def cli(stage,mode):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'launch.py'),'--stage',stage,'--mode',mode],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)
def validate():
    spec=read(ROOT/'model_spec.json');result={};guards=[]
    for stage in ['A','B']:
        plan=read(ROOT/f'Protocol_{stage}.json')
        models,x,y=fixture(plan,spec);expected={(r['h'],r['id']):r['y'] for r in y}
        synthetic=cli(stage,'synthetic')
        metrics_checked=0;intervals_checked=0
        for h in plan['horizons']:
            for view,settings in plan['views'].items():
                z=sorted((r for r in x if r['h']==h and view in r['views']),key=lambda r:r['id'])
                yy=np.array([expected[(h,r['id'])] for r in z])
                forecast={}
                for model in settings['models']:
                    if model in ('persistence','typical'):
                        pp=[r['x']['baseline_moca']+(spec['offset_'+stage][str(h)] if model=='typical' else 0) for r in z]
                    elif stage=='A':pp=[scalar_a(r['x'],models,model,h) for r in z]
                    else:pp=[scalar_b(r['x'],models[str(h)]) for r in z]
                    forecast[model]=np.array(pp);err=forecast[model]-yy
                    slope,intercept=np.polyfit(pp,yy,1)
                    vals={'MAE':np.mean(abs(err)),'RMSE':np.sqrt(np.mean(err**2)),'R2':1-sum(err**2)/sum((yy-yy.mean())**2),'bias':np.mean(err),'outside_0_30':np.mean((forecast[model]<0)|(forecast[model]>30)),'calibration_slope':slope,'calibration_intercept':intercept}
                    for k,v in vals.items():
                        close(v,synthetic[f'{view}/{h}']['metrics'][model+'/'+k]['estimate']);metrics_checked+=1
                first,ref=settings['models'][:2]
                d=np.abs(forecast[first]-yy)-np.abs(forecast[ref]-yy)
                seed=plan['seed_base']+settings['rank']*100+h
                ix=np.random.Generator(np.random.PCG64(seed)).integers(len(z),size=(plan['bootstrap'],len(z)))
                rec=synthetic[f'{view}/{h}']['metrics'][first+'-'+ref+'/delta_MAE']
                close(rec['estimate'],d.mean());close(rec['interval'],np.percentile(d[ix].mean(axis=1),[2.5,97.5]));intervals_checked+=1
        guards.append(rejected(stage+' duplicate identity',lambda:evaluate(plan,spec,models,x+[x[0]],y)))
        guards.append(rejected(stage+' missing outcome',lambda:evaluate(plan,spec,models,x,y[:-1])))
        guards.append(rejected(stage+' incomplete model inventory',lambda:evaluate(plan,spec,{},x,y)))
        guards.append(rejected(stage+' invalid score',lambda:evaluate(plan,spec,models,x,[dict(r,y=31) for r in y])))
        if stage=='A':
            item=copy.deepcopy(x[0]);item['x']['SDMT']=None
            guards.append(rejected('A required SDMT missing',lambda:forecast_a([item],models,'compact4',12,spec,True)))
            item=copy.deepcopy(x[0]);item['x']['baseline_sdmt_form']='other'
            guards.append(rejected('A unbound form',lambda:forecast_a([item],models,'compact4',12,spec,True)))
        else:
            item=copy.deepcopy(x[0]);item['x']['age_years']=90
            guards.append(rejected('B age outside source rules',lambda:forecast_b([item],models['12'])))
            bad=dict(models['12'],features=['age_years','baseline_moca','sex_male'])
            guards.append(rejected('B reordered predictors',lambda:forecast_b(x[:1],bad)))
        pre=cli(stage,'preflight')
        result[stage]=dict(synthetic_view_horizons=len(synthetic),independent_point_estimates=metrics_checked,
                           independently_reconstructed_paired_intervals=intervals_checked,bootstrap_replicates_per_view=plan['bootstrap'],preflight=pre)
    z=summarize(np.array([25.,25.]),{'constant':np.array([35.,35.])},100,7)
    assert z['constant/R2']['valid']==0 and z['constant/R2']['interval'] is None and z['constant/calibration_slope']['estimate'] is None and z['constant/outside_0_30']['estimate']==1
    with tempfile.TemporaryDirectory(dir=ROOT) as t:
        tmp=Path(t);(tmp/'x').write_text('original');hashes={'x':digest(tmp/'x')};(tmp/'x').write_text('modified')
        guards.append(rejected('modified binding',lambda:verify(hashes,tmp)))
    return dict(status='PASS',stage_results=result,rejected_invalid_cases=guards,edge_cases=['constant outcome','constant prediction','unclipped out-of-range prediction'],study_fits=0,study_predictions=0,paid_queries=0,lock_sha256=digest(ROOT/'LOCK.json'))
if __name__=='__main__':
    result=validate();(ROOT/'Synthetic_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
