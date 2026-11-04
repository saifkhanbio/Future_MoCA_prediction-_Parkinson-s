"""Independent pandas-based re-derivation; does not import the production audit rules implementation."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,datetime,math
D=Path(__file__).resolve().parents[1];P=D.parent;B=P.parent/'strategy_redesign_v2_2026-10-20';I=json.loads((D/'config/Inputs.json').read_text());checks=[]
def check(name,ok,detail=None):checks.append({'check':name,'passed':bool(ok),'detail':detail})
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def df(p):return pd.read_csv(p,sep='\t' if Path(p).suffix=='.tsv' else ',',dtype=str,keep_default_na=False)
def months(s):
 z=pd.to_datetime(s.where(s.str.fullmatch(r'\d{2}/\d{4}')),format='%m/%Y',errors='coerce');return z.dt.year*12+z.dt.month
risk=pd.read_csv(D/'restricted/Baseline_Risk_Set.tsv',sep='\t',dtype={'PATNO':str});out=pd.read_csv(D/'restricted/Cognitive_Endpoint_Eligibility.tsv',sep='\t',dtype={'PATNO':str});det=pd.read_csv(D/'restricted/Detailed_Cognitive_Eligibility.tsv',sep='\t',dtype={'PATNO':str});resolved=pd.read_csv(D/'restricted/Comparator_Features_Resolved.tsv',sep='\t',dtype={'PATNO':str})
reg=df(I['rna_registry']);bl=reg[reg.visit.eq('BL')];qc=df(I['rna_qc']);check('Baseline sample, participant and identity uniqueness',len(bl)==410 and bl.PATNO.is_unique and bl['sample'].is_unique and risk.component_id.is_unique and risk.component_id.notna().all())
check('Baseline registry and risk set bind exactly',set(bl.PATNO)==set(risk.PATNO) and set(bl['sample'])==set(risk.sample_id))
frames={}
for name in ['moca','education','sdmt','hvlt','diagnosis','blood_collection','baseline_start']:
 z=df(I[name]);z['source_row']=np.arange(len(z))+2;z['month']=months(z['INFODT']);frames[name]=z[z.PATNO.isin(set(bl.PATNO))].copy()
items=load(I['prior_cognitive_contract'])['cognition']['item_max'];md=frames['moca'].set_index('source_row');num=md[list(items)].apply(pd.to_numeric,errors='coerce');valid=((num%1==0)&(num>=0)&num.le(pd.Series(items))).all(axis=1);raw=num.sum(axis=1,min_count=len(items));baseok=[];enhanced=[];epok=[];select_errors=0;score_errors=0
# Array-based item reconstruction with independently coded acceptance logic.
def qcscore(r,edu):
 if r is None:return False,None,None
 sr=int(r['source_row']);rawscore=raw.loc[sr];isvalid=bool(valid.loc[sr]);adj=None
 if isvalid:
  if pd.notna(edu) and 0<=edu<=40 and edu%1==0:adj=min(30,rawscore+int(edu<=12 and rawscore<30))
  elif rawscore==30:adj=30
  else:isvalid=False
  t=pd.to_numeric(r['MCATOT'],errors='coerce')
  if r['MCATOT']!='' and (pd.isna(t) or t%1!=0 or t not in [rawscore,adj]):isvalid=False
 w=pd.to_numeric(r['MCAVFNUM'],errors='coerce');f=pd.to_numeric(r['MCAVF'],errors='coerce')
 if r['MCAVFNUM']!='':
  if pd.isna(w) or w%1!=0 or not 0<=w<=10000:isvalid=False
  elif pd.notna(f) and f in [0,1] and f!=int(w>=11):isvalid=False
 return isvalid,rawscore if valid.loc[sr] else None,adj
selected={}
for rr in risk.itertuples(index=False):
 idx=rr.RNA_index_month;p=rr.PATNO
 z=frames['education'];z=z[z.PATNO.eq(p)&z.EVENT_ID.isin(['SC','BL'])&z.month.le(idx)].copy();z['bl_priority']=z.EVENT_ID.eq('BL').astype(int);z=z.sort_values(['month','bl_priority','source_row'],ascending=[False,False,True]);er=None if z.empty else z.iloc[0];edu=pd.to_numeric(er.EDUCYRS,errors='coerce') if er is not None else np.nan
 if pd.notna(edu) and (edu%1 or not 0<=edu<=40):edu=np.nan
 checkedu=(pd.isna(edu) and pd.isna(rr.education)) or edu==rr.education
 score_errors+=int(not checkedu)
 m=frames['moca'];m=m[m.PATNO.eq(p)]
 for h in [0,12,24,36]:
  z=m[m.month.between(idx-3,idx)&m.EVENT_ID.isin(['SC','BL'])].copy() if h==0 else m[m.month.between(idx+h-3,idx+h+3)].copy()
  z['distance']=(z.month-idx-h).abs();z['bl_priority']=z.EVENT_ID.map({'BL':0,'SC':1});sort=['distance','bl_priority','source_row'] if h==0 else ['distance','month','EVENT_ID','source_row'];z=z.sort_values(sort);r=None if z.empty else z.iloc[0];selected[p,h]=r
  erow=rr.baseline_moca_source_row if h==0 else out.loc[out.PATNO.eq(p)&out.horizon_months.eq(h),'source_row'].iloc[0]
  select_errors+=int(not((r is None and pd.isna(erow)) or (r is not None and r.source_row==erow)))
  good,rv,av=qcscore(r,edu)
  if h==0:
   basic=good and rr.native_preindex_PD_confirmed and rr.RNA_technical_ready and rr.RNA_native_collection_confirmed and abs(rr.reference_delta_months)<=1 and not rr.death_conflict and not rr.invalid_death_date and (pd.isna(rr.death_month) or idx<=rr.death_month)
   baseok.append((p,basic))
   saved=rr.baseline_moca
  else:
   saved=out.loc[out.PATNO.eq(p)&out.horizon_months.eq(h),'education_adjusted_MoCA'].iloc[0]
   if r is not None and pd.notna(rr.death_month) and r.month>rr.death_month:good=False
   good=good and not rr.death_conflict and not rr.invalid_death_date
   epok.append((p,h,good))
  score_errors+=int(not((av is None and pd.isna(saved)) or (av is not None and av==saved)))
 # Detailed baseline arithmetic and form coding independently reconstructed.
 vals=[]
 for kind,scorecols,form,maxform,maxscore in [('sdmt',['SDMTOTAL'],'SDMTVRSN',2,110),('hvlt',['HVLTRT1','HVLTRT2','HVLTRT3','HVLTRDLY'],'HVLTVRSN',6,12)]:
  z=frames[kind];z=z[z.PATNO.eq(p)&z.EVENT_ID.isin(['SC','BL'])&z.month.between(idx-3,idx)].copy();z['distance']=idx-z.month;z['bl_priority']=z.EVENT_ID.map({'BL':0,'SC':1});z=z.sort_values(['distance','bl_priority','source_row']);r=None if z.empty else z.iloc[0]
  if r is None:vals.append(False)
  else:
   v=pd.to_numeric(r[scorecols],errors='coerce');f=pd.to_numeric(r[form],errors='coerce');vals.append(bool(v.notna().all() and (v%1==0).all() and v.between(0,maxscore).all() and pd.notna(f) and f%1==0 and 1<=f<=maxform))
 enhanced.append((p,all(vals)))
check('Every MoCA baseline and follow-up row selected independently by date',select_errors==0,{'mismatches':select_errors})
check('Education and item-level MoCA scores independently reconstructed',score_errors==0,{'mismatches':score_errors})
b=pd.DataFrame(baseok,columns=['PATNO','baseline']).set_index('PATNO');h=pd.DataFrame(enhanced,columns=['PATNO','enhanced']).set_index('PATNO');v=pd.DataFrame(epok,columns=['PATNO','horizon_months','valid'])
check('Baseline eligibility independently recovered',np.array_equal(risk.PATNO.map(b.baseline),risk.diagnosis_supported_baseline_ready))
check('Enhanced baseline eligibility independently recovered',np.array_equal(risk.PATNO.map(h.enhanced),risk.enhanced_baseline_complete))
v['base']=v.PATNO.map(b.baseline);v['enh']=v.PATNO.map(h.enhanced);v['SWEDD']=v.PATNO.map(risk.set_index('PATNO').SWEDD_recruitment);v['matched']=v.valid&v.base&v.enh&~v.SWEDD
recovered=v.groupby('horizon_months').matched.sum().astype(int).to_dict();check('Matched eligibility independently recovers all horizons',recovered=={12:333,24:321,36:307},recovered)
check('Every endpoint QC decision agrees',np.array_equal(v.set_index(['PATNO','horizon_months']).loc[pd.MultiIndex.from_frame(out[['PATNO','horizon_months']]),'valid'],out.endpoint_QC_ready))
sets=[set(v.loc[v.horizon_months.eq(x)&v.matched,'PATNO']) for x in [12,24,36]];check('Shared follow-up cohort independently recovered',len(set.intersection(*sets))==280)
# Source diagnosis and actual RNA collection, not inherited labels.
diag=frames['diagnosis'];blood=frames['blood_collection'];diag_good=0;rna_good=0
for r in risk.itertuples(index=False):
 z=diag[diag.PATNO.eq(r.PATNO)&diag.EVENT_ID.isin(['SC','BL'])&diag.month.le(r.RNA_index_month)];codes=set(pd.to_numeric(z.PRIMDIAG,errors='coerce').dropna());diag_good+=codes=={1}
 z=blood[blood.PATNO.eq(r.PATNO)&blood.EVENT_ID.eq('BL')];draws=months(z.BLDDRDT);rna_good+=len(z)>0 and draws.eq(r.RNA_index_month).all() and z.BLDRNA.eq('1').all()
check('Native pre-index PD diagnosis confirmed independently',diag_good==410);check('Native collection month and RNA collected flag confirmed independently',rna_good==410)
mat=np.load(I['rna_matrix'],mmap_mode='r');q=qc.set_index('sample');vector_errors=0
for r in bl.itertuples(index=False):
 a=mat[int(r.count_row)];vector_errors+=int(not(np.issubdtype(a.dtype,np.integer) and (a>=0).all() and hashlib.sha256(a.tobytes()).hexdigest()==q.loc[r.sample,'count_vector_sha256']))
check('All 410 raw vectors independently match source QC hashes',vector_errors==0)
l=pd.read_csv(D/'restricted/Source_Row_Ledger.tsv',sep='\t');dyn=l[l.role.isin(['baseline_moca','baseline_sdmt','baseline_hvlt','part2','part3','gds','diagnosis_history'])&l.selected_source_row.notna()];check('No selected dynamic predictor after RNA landmark or outside prior three months',dyn.month_minus_index.between(-3,0).all())
check('All selected follow-up records respect calendar windows',(out.loc[out.source_row.notna(),'elapsed_months']-out.loc[out.source_row.notna(),'horizon_months']).abs().le(3).all())
check('Baseline shared across horizons and no repeated person within horizon',not out.duplicated(['PATNO','horizon_months']).any() and len(out)==1230)
# Static sex source resolution, raw source re-check.
smanifest=load(D/'evidence/Static_Sex_Input_Manifest.json');meta=df(smanifest[0]['path']);dem=df(I['demographics']);un=df(I['anchor_frame']);sx_errors=0
for r in resolved.itertuples(index=False):
 m=meta[meta.Sample.eq(r.sample_id)];dd=dem[dem.PATNO.eq(r.PATNO)&dem.PAG_NAME.eq('SCREEN')];uu=un[un.PATNO.eq(r.PATNO)&un.EVENT_ID.eq('BL')];expected=1. if len(m)==1 and m.GENDER.iloc[0]=='Male' else 0. if len(m)==1 and m.GENDER.iloc[0]=='Female' else None
 sx_errors+=int(not(expected is not None and set(pd.to_numeric(dd.SEX,errors='coerce').dropna())=={expected} and set(pd.to_numeric(uu.SEX,errors='coerce').dropna())=={expected} and r.sex_male==expected))
check('Resolved static sex independently agrees across all three sources',sx_errors==0)
# Preserve parent artifacts and all previously protected frozen results.
expected={str(P/r['path']):r['sha256'] for r in load(D/'evidence/Parent_Artifact_Manifest_Before.json')['files']}
J=B/'biomarker_value_journal_neurology_clinical_v2_2026-10-24'
for r in load(J/'qa/Preservation_Before.json'):expected[r['path']]=r['sha256']
for r in load(J/'Artifact_Manifest.json')['files']:expected[str(J/r['path'])]=r['sha256']
for manifest in [D/'evidence/Input_Manifest.json',D/'evidence/Additional_Input_Manifest.json',D/'evidence/Static_Sex_Input_Manifest.json',P/'evidence/Source_Manifest.json']:
 for r in load(manifest):expected[r['path']]=r['sha256']
changed=[p for p,hsh in expected.items() if not Path(p).is_file() or sha(p)!=hsh]
check('Previous study artifacts, frozen results and inputs unchanged',not changed,{'files_checked':len(expected),'changes':len(changed)})
result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':all(x['passed'] for x in checks),'checks':checks,'protected_files_checked':len(expected),'scope':'Separate pandas implementation for source selection and MoCA arithmetic; no reuse of production selection or scoring functions. Source preservation is scoped to recorded manifests.'}
(D/'qa/Independent_Verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':result['passed'],'checks':len(checks),'protected_files_checked':len(expected),'failures':[x for x in checks if not x['passed']]},indent=2));assert result['passed'],'Audit verification failed; review aggregate QA record'