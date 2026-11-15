"""Deterministic simulated rows and model parameters. No local participant files are read."""
import math
def fixture(plan,spec):
    stage=plan['stage']; records=[]; outcomes=[]; states={}
    active=spec['active']
    for hi,h in enumerate(plan['horizons']):
        if stage=='A':
            for ai,(arm,nums) in enumerate(active['active_numeric'].items()):
                cats=active['active_categories'][arm]
                for fold in range(5):
                    cols=[s for k in nums for s in [k,k+'__missing']]+[k+'=='+v for k,levels in cats.items() for v in levels]
                    if fold%2:cols.reverse()
                    prep={k:dict(mean=3+j+.3*fold,median=7+j+fold,sd=1.2+.1*j+.3*fold) for j,k in enumerate(nums)}
                    states[f'{arm}_{h}m_fold{fold}']=dict(columns=cols,coef=[(.07+ai*.01)*math.cos(j+fold) for j in range(len(cols))],intercept=23+hi+fold*.2,preprocessing=dict(numeric=prep,categories=cats))
        else:states[str(h)]=dict(features=['baseline_moca','age_years','sex_male'],mean=[25,61,.5],scale=[2,8,.5],coefficient=[1.3+hi*.1,-.4,.25],intercept=24.5+hi)
        n=plan['views'][plan['primary_view']]['n'][hi]
        for i in range(n):
            identity=f'SYN-{i:04d}'; views=[k for k,v in plan['views'].items() if i<v['n'][hi]]
            if stage=='A':
                x={k:float(2+(i+j)%10) for j,k in enumerate(active['active_numeric']['core_plus18'])}
                for k,levels in active['active_categories']['core_plus18'].items():x[k]=levels[i%(len(levels)-1)]
                if 'complete19' not in views:x['NLR']=None
                if 'complete_all' not in views:x['part3']=None
                if 'complete_all' not in views:x['motor_state']='unknown'
            else:x=dict(age_years=41+i%39,sex_male=i%2)
            x['baseline_moca']=18+i%13
            records.append(dict(h=h,id=identity,views=views,x=x));outcomes.append(dict(h=h,id=identity,y=18+(3*i+hi)%13))
    return states,records,outcomes[::-1]
