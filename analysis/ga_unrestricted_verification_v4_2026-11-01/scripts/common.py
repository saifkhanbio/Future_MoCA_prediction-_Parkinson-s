import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,sqlite3,zlib,copy,math,datetime,gzip
import numpy as np
V=Path(__file__).resolve().parents[1]; C=V.parent
A=C/'ga_unrestricted_readiness_v4_10kgen_2026-11-01'
E=C/'ga_unrestricted_execution_v4_10kgen_2026-11-01'
R=C/'ga_unrestricted_run_v4_10kgen_2026-11-01'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p)
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(enc(v)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def tab(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def near(a,b,tol=1e-10):
    assert np.allclose(a,b,atol=tol,rtol=1e-11,equal_nan=False),'Numerical agreement check failed'
def data():
    c=read(A/'config/Analysis_Contract.json');p=read(A/'config/r2_expanded_v1.json');binding=read(A/'config/Input_Binding.json')
    raw=tab(binding['candidate_matrix']);by={r['PATNO']:r for r in raw};cohorts={int(h):tab(path) for h,path in binding['cohorts'].items()}
    assert len(raw)==len(by)==351 and len({r['component_id'] for r in raw})==351
    master={r['component_id']:r for r in tab(binding['master_splits'])};views=tab(binding['view_binding']);seen={}
    for h,rows in cohorts.items():
        assert len(rows)=={12:333,24:321,36:307}[h] and len({r['PATNO'] for r in rows})==len(rows)==len({r['component_id'] for r in rows})
        view={r['PATNO']:r for r in views if r['view_id']==f'main_{h}_enhanced'}
        assert set(view)=={r['PATNO'] for r in rows}
        for r in rows:
            orig={**view[r['PATNO']],**master[r['component_id']]}
            assert all(str(orig[k])==str(v) for k,v in r.items())
            assert by[r['PATNO']]['component_id']==r['component_id'] and 0<=float(r['y'])<=30
            sig=tuple(r[k] for k in ['component_id','outer_fold']+[f'inner_for_outer_{i}' for i in range(5)])
            if r['PATNO'] in seen:assert sig==seen[r['PATNO']]
            seen[r['PATNO']]=sig
    return c,p,binding,by,cohorts
def arm_contract(base,arm):
    c=copy.deepcopy(base);c['active_arm']=arm
    if arm=='moca27':
        c['features']['optional']=[dict(x,bit=i) for i,x in enumerate(x for x in base['features']['optional'] if x['name']!='baseline_moca')]
        c['features']['core']={'numeric':['baseline_moca'],'categories':{}}
        c['features']['mandatory_instruments']=['MoCA']
    c['controls']['full_mask']=(1<<len(c['features']['optional']))-1
    return c
def layout(c):
    ns=c['features']['core']['numeric']+[x['name'] for x in c['features']['optional'] if x.get('type','numeric')=='numeric']
    cs=dict(c['features']['core']['categories'])
    cs.update({k:v for k,v in c['features'].get('feature_categories',{}).items() if k in {x['name'] for x in c['features']['optional']}})
    cs.update({k:v['levels'] for k,v in c['features']['nuisance_categories'].items()})
    return ns,cs
def num(x):return float(x) if x not in ['',None] else float('nan')
def spec_for(base,by,cohorts,arm,h,o):
    c=arm_contract(base,arm);rr=cohorts[h];tr=[r for r in rr if int(r['outer_fold'])!=o];te=[r for r in rr if int(r['outer_fold'])==o]
    assert all(int(r[f'inner_for_outer_{o}'])==-1 for r in te)
    assert not ({r['component_id'] for r in tr}&{r['component_id'] for r in te})
    ns,cs=layout(c);keys=ns+list(cs)+['PATNO','component_id']
    sp={'context_id':f'{arm}_{h}m_outer{o}','arm':arm,'horizon':h,'outer':o,'train_rows':[{k:by[r['PATNO']][k] for k in keys} for r in tr],
        'train_y':[float(r['y']) for r in tr],'inner':[int(r[f'inner_for_outer_{o}']) for r in tr],
        'test_rows':[{k:by[r['PATNO']][k] for k in keys} for r in te],'synthetic':False}
    assert set(sp['inner'])=={0,1,2}
    return c,sp,tr,te
def full_design(train,test,c):
    names,cats=layout(c);X=[];T=[];labels=[];prep={'numeric':{},'categories':cats}
    for name in names:
        x=np.array([num(r[name]) for r in train]);t=np.array([num(r[name]) for r in test])
        valid=x[np.isfinite(x)];med=float(np.median(valid)) if len(valid) else 0.
        xi=np.where(np.isfinite(x),x,med);ti=np.where(np.isfinite(t),t,med);mu=float(xi.mean());sd=float(xi.std());sd=sd if sd>1e-12 else 1.
        prep['numeric'][name]={'median':med,'mean':mu,'sd':sd,'training_observed_n':len(valid)}
        X.extend([(xi-mu)/sd,(~np.isfinite(x)).astype(float)]);T.extend([(ti-mu)/sd,(~np.isfinite(t)).astype(float)])
        labels.extend([name,name+'__missing'])
    for name,levels in cats.items():
        assert all((r.get(name) or 'unknown') in levels for r in train+test)
        for level in levels:
            X.append(np.array([(r.get(name) or 'unknown')==level for r in train],float))
            T.append(np.array([(r.get(name) or 'unknown')==level for r in test],float));labels.append(name+'=='+level)
    return np.column_stack(X),np.column_stack(T),labels,prep
def features(mask,c):
    return c['features']['core']['numeric']+list(c['features']['core']['categories'])+[x['name'] for x in c['features']['optional'] if mask&(1<<x['bit'])]
def indices(mask,c,labels):
    fs=features(mask,c);cat=set(c['features']['core']['categories'])|(set(fs)&set(c['features'].get('feature_categories',{})))
    cat|={k for k,v in c['features']['nuisance_categories'].items() if set(v['trigger'])&set(fs)}
    wanted=set(fs+[n+'__missing' for n in fs])
    return [i for i,n in enumerate(labels) if n in wanted or n.split('==')[0] in cat]
def burden_count(masks,c):
    masks=np.asarray(masks,dtype=np.int64);counts=np.full(len(masks),len(c['features']['core']['numeric'])+len(c['features']['core']['categories']),dtype=np.int16)
    groups={g:np.ones(len(masks),bool) for g in c['features'].get('mandatory_instruments',[])}
    for f in c['features']['optional']:
        z=(masks&(1<<f['bit']))!=0;counts+=z
        if f['instrument'] not in groups:groups[f['instrument']]=np.zeros(len(masks),bool)
        groups[f['instrument']]|=z
    costs=sum(groups.values()).astype(np.int16)
    return costs,counts
def mandatory(c):
    n=len(c['features']['optional']);full=(1<<n)-1;bits={f['name']:f['bit'] for f in c['features']['optional']}
    core=['age_index','sex_male','education','baseline_moca','diagnosis_duration_years','part2','part3','gds15','pd_medication','motor_state']
    def mask(fs):return sum(1<<bits[x] for x in fs if x in bits)
    compact=['baseline_moca','SDMT','HVLT_immediate','HVLT_delayed']
    broad=[x['name'] for x in c['features']['optional'] if x['name'] not in set(core)-{'baseline_moca'}]
    return list(dict.fromkeys([0,full,mask(core),mask(compact),mask(broad)]+[1<<i for i in range(n)]+[full^(1<<i) for i in range(n)]))
