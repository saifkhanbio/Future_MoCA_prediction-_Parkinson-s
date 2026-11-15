import os,sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
os.umask(0o077)
J=Path(__file__).resolve().parents[1];P=J.parent;AB=P/'Stage_AB_protocols_v1_2026-11-15'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=read(J/'config/LOCK.json');assert sha(J/'config/diagnostic_plan.json')==lock['plan'] and all(sha(P/k)==v for k,v in lock['sources'].items())
cfg=read(J/'config/diagnostic_plan.json');R=J/'results';sys.path.insert(0,str(AB))
from evaluation import forecast_a,forecast_b
spec=read(AB/'model_spec.json');frames=[]
for st in ('A','B'):
 r=AB/'restricted';xx=read(r/f'features_{st}.json');yy={(x['h'],x['id']):x['y'] for x in read(r/f'outcomes_{st}.json')};models=read(r/f'models_{st}.json')
 for h in cfg['horizons']:
  z=sorted([x for x in xx if x['h']==h and cfg['stages'][st] in x['views']],key=lambda x:x['id'])
  pred=forecast_a(z,models,'compact4',h,spec,False) if st=='A' else forecast_b(z,models[str(h)])
  for x,p in zip(z,pred):frames.append((st,h,x['id'],x['x']['baseline_moca'],yy[h,x['id']],p))
c=pd.read_csv(P/'Stage_C_results_v1/restricted/oof_predictions.csv')
for x in c[c.view=='expanded'].itertuples():frames.append(('C',x.horizon,x.component,x.persistence,x.observed,x.compact4))
df=pd.DataFrame(frames,columns=['stage','h','id','baseline','observed','predicted']).sort_values(['stage','h','id']);assert not df.duplicated(['stage','h','id']).any()
df.to_csv(J/'restricted/forecast_bindings.csv',index=False)
summary=[];freq=[];bands=[];cal=[]
ab=pd.read_csv(P/'Stage_AB_verified_results_v1/Verified_results.csv');cc=pd.read_csv(P/'Stage_C_verified_results_v1/verified_estimates.csv').rename(columns={'lower95':'low95','upper95':'high95'}).assign(stage='C')
ag=pd.concat([ab,cc])
for (stage,h),g in df.groupby(['stage','h']):
 b=g.baseline;y=g.observed;chg=y-b;pred=g.predicted;n=len(b)
 denom={'A':301,'B':887,'C':1175}[stage]
 summary.append(dict(stage=stage,h=h,n=n,frame=denom,fraction=n/denom,mean_change=chg.mean(),q25=chg.quantile(.25),median=chg.median(),q75=chg.quantile(.75),mean_abs_change=chg.abs().mean(),unchanged=int((chg==0).sum()),lower=int((chg<0).sum()),higher=int((chg>0).sum()),baseline_30=int((b==30).sum()),followup_30=int((y==30).sum()),baseline_29_30=int((b>=29).sum()),followup_29_30=int((y>=29).sum())))
 for k,v in chg.value_counts().sort_index().items():freq.append(dict(stage=stage,h=h,change=k,n=v,percent=100*v/n))
 arm='reduced3' if stage=='B' else 'compact4'
 for metric in ['bias','calibration_intercept','calibration_slope']:
  rec=ag[(ag.stage==stage)&(ag.horizon==h)&(ag.view==cfg['stages'][stage])&(ag.measure==arm+'/'+metric)].iloc[0];cal.append(dict(stage=stage,h=h,metric=metric,estimate=rec.estimate,lo=rec.low95,hi=rec.high95))
 if [stage,h] in cfg['omit_subdivision']:continue
 for lo,hi in cfg['bands']:
  sub=g[g.baseline.between(lo,hi)];nr=len(sub)
  row=dict(stage=stage,h=h,lower=lo,upper=hi,n=nr)
  if nr>=cfg['minimum_n']:
   me=(sub.predicted-sub.observed).abs();pe=(sub.baseline-sub.observed).abs();loss=(me-pe).to_numpy()
   rng=np.random.default_rng(20261115+10000*('ABC'.index(stage)+1)+100*h+lo)
   draws=loss[rng.integers(nr,size=(5000,nr))].mean(1);ci=np.quantile(draws,[.025,.975])
   row.update(model_mae=me.mean(),persistence_mae=pe.mean(),delta=loss.mean(),ci_low=ci[0],ci_high=ci[1])
  bands.append(row)
for k,s in [('followup_and_stability',summary),('change_distribution',freq),('baseline_bands',bands),('calibration',cal)]:pd.DataFrame(s).to_csv(R/(k+'.csv'),index=False)
