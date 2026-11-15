"""Stage C eligibility only. Run under WSL with python -B."""
import sys,os,csv,json,collections,hashlib,math,re
from pathlib import Path
import pandas as pd
import numpy as np
os.umask(0o077)
O=Path(__file__).resolve().parent
R=O/'restricted'
R.mkdir(exist_ok=True)
C=O.parents[2];P=C.parent;B=P/'strategy_redesign_v2_2026-10-20'
A=C/'comprehensive_feature_audit_2026-10-29'
audit=C/'additional_ppmi_eligibility_prior_use_2026-10-28'
T=C/'clinical_temporal_validation_contract_2026-10-28'
sys.path[:0]=[str(audit/'scripts'),str(T/'scripts')]
import audit_common as ac
from audit_common import group,number,integer,unique,month,select,moca,itemscore,detail,ITEMS
from contract_common import resolve
from collections import Counter
hashes={}
def record(f):
 f=Path(f)
 if str(f) not in hashes:hashes[str(f)]=ac.sha(f)
 return f
def rows(f):
 f=record(f)
 with f.open(encoding='utf-8-sig',newline='') as fh:return [dict(r,_row=j) for j,r in enumerate(csv.DictReader(fh,delimiter='\t' if f.suffix=='.tsv' else ','),2)]
def put(path,rr):
 pd.DataFrame(rr).to_csv(O/path,index=False)
def read(f):
 f=record(f);return pd.read_csv(f,sep='\t' if f.suffix=='.tsv' else ',',dtype=str,keep_default_na=False)
def load(f):return json.loads(record(f).read_text())
I=load(audit/'config/Inputs.json')
record(audit/'scripts/audit_common.py');record(T/'scripts/contract_common.py');record(I['prior_cognitive_contract'])
I['moca']=str(C/'moca_refresh_audit_2026-10-28/raw/Montreal_Cognitive_Assessment__MoCA__28Oct2026.csv')
assert ac.sha(I['moca'])=='d9d4edbfb8744e9f81e8dadf177ef5ddbc5f44be352002232aeeb4879bb02cd0'
raw=Path(I['hvlt']).parent
orig={r['component_id']:r for r in rows(I['current_manifest'])}
frozen=rows(C/'model_readiness_2026-10-24/restricted/View_Row_Binding.tsv')
old={h:{r['component_id'] for r in frozen if r['view_id']==f'main_{h}_enhanced'} for h in (12,24,36)}
assert len(orig)==351 and [len(x) for x in old.values()]==[333,321,307]
reserved={r['component'] for r in rows(O.parent/'Stage_AB_eligibility_2026-11-15/restricted/A_membership.csv')}
assert len(reserved)==227 and not reserved & set(orig)
components=group(rows(I['identity_components']),'source_key')
comp={k:next(iter(v)) if len(v)==1 else None for k,rr in components.items() for v in [{r['component_id'] for r in rr}]}
compn=Counter(v for k,v in comp.items() if k.startswith('PPMI:'))
diag=rows(I['diagnosis']); ids={r['PATNO'] for r in diag if r['EVENT_ID'] in ('SC','BL') and number(r['PRIMDIAG'])==1}
keys=['moca','education','baseline_start','hvlt','sdmt','conclusion','age','demographics','diagnosis_history','part2','part3','gds','participant_status']
source={k:[r for r in rows(I[k]) if r['PATNO'] in ids] for k in keys}
source['diagnosis']=[r for r in diag if r['PATNO'] in ids]
by={k:group(v) for k,v in source.items()}
with record(I['anchor_frame']).open(encoding='utf-8-sig',newline='') as f:
 by['unified']=group([{k:r[k] for k in ['PATNO','EVENT_ID','visit_date','Death_Date','SEX','age_at_visit']}|{'_row':j} for j,r in enumerate(csv.DictReader(f,delimiter='\t'),2) if r['PATNO'] in ids])
resolve_src={'dx':by['diagnosis_history'],'demo':by['demographics'],'age':by['age'],'unified':by['unified']}
p2=[k for k in source['part2'][0] if k.startswith('NP2') and k!='NP2PTOT'];p3=[k for k in source['part3'][0] if k.startswith('NP3') and k!='NP3TOT'];gd=[k for k in source['gds'][0] if k.startswith('GDS')]
assert (len(p2),len(p3),len(gd))==(13,33,15)
reverse={'GDSSATIS','GDSGSPIR','GDSHAPPY','GDSALIVE','GDSENRGY'}
mfields=list(ITEMS)+['MCATOT','MCAVFNUM'];hfields=['HVLTRT1','HVLTRT2','HVLTRT3','HVLTRDLY','HVLTVRSN'];sfields=['SDMTOTAL','SDMTVRSN']
risk=[];ends=[];omitted=[]
def log(*args):pass
for p in sorted(ids):
 component=comp.get('PPMI:'+p);reasons=[];flags=[]
 bs,_,bst=unique([r for r in by['baseline_start'][p] if r['EVENT_ID']=='BL'],['INFODT'])
 ub=[r for r in by['unified'][p] if r['EVENT_ID']=='BL'];ur,_,ust=unique(ub,['visit_date'])
 native=month(bs['INFODT']) if bs else None; unified=month(ur['visit_date']) if ur else None
 ref=native if native is not None else unified;refsource='native_BL_start' if native is not None else 'unified_BL_reference'
 if ref is None or bst=='conflicting_records' or ust=='conflicting_records' or (native is not None and unified is not None and abs(native-unified)>1):
  omitted.append({'PATNO':p,'reason':'missing_or_conflicting_baseline_reference'});continue
 idx=ref
 if not component or compn[component]!=1:reasons.append('ambiguous_or_missing_identity')
 dg=[r for r in by['diagnosis'].get(p,[]) if r['EVENT_ID'] in ['SC','BL'] and month(r['INFODT']) is not None and month(r['INFODT'])<=idx];codes={number(r['PRIMDIAG']) for r in dg if number(r['PRIMDIAG']) is not None};native_pd=1 in codes and not(codes-{1})
 if not native_pd:reasons.append('native_preindex_PD_not_confirmed' if not codes else 'contrary_native_preindex_diagnosis')
 for r in dg:log(p,'native_preindex_diagnosis','diagnosis',r,[r['_row']],month(r['INFODT']),idx,'PD' if number(r['PRIMDIAG'])==1 else 'contrary_or_missing')
 er,erefs,em,est=select(by['education'].get(p,[]),idx,['EDUCYRS'],education=True);edu=number(er['EDUCYRS']) if er and integer(er.get('EDUCYRS'),0,40) else None;log(p,'education','education',er,erefs,em,idx,est)
 if er and er.get('EDUCYRS_EQUIV','').strip() and number(er['EDUCYRS_EQUIV'])!=edu:flags.append('education_equivalent_differs')
 mr,mrefs,mm,mst=select(by['moca'].get(p,[]),idx,mfields);mq=moca(mr,edu,mst);reasons+=['baseline_moca_'+v for v in mq['reasons']];log(p,'baseline_moca','moca',mr,mrefs,mm,idx,mst)
 deaths0={r['Death_Date'] for r in by['unified'].get(p,[]) if r.get('Death_Date','').strip()};deathv={month(s) for s in deaths0};invaliddeath=None in deathv;deathv.discard(None);death=min(deathv) if deathv else None;dc=len(deathv)>1
 if death is not None and idx>death:reasons.append('baseline_after_death_month')
 if dc:reasons.append('conflicting_death_dates')
 if invaliddeath:reasons.append('invalid_death_date')
 wd=by['conclusion'].get(p,[]);wdmonths=[month(r['WDDT']) for r in wd if month(r.get('WDDT')) is not None];earliestwd=min(wdmonths) if wdmonths else None;wddeath=any(r.get('WDDEATH')=='1' for r in wd)
 if earliestwd is not None and idx>earliestwd:reasons.append('baseline_after_withdrawal_requires_review')
 row={'PATNO':p,'component_id':component,'index_month':idx,'index_year':(idx-1)//12,'reference_month':ref,'reference_source':refsource,'index_minus_reference':None if ref is None else idx-ref,'native_preindex_PD':native_pd,'baseline_moca':mq['adjusted'],'baseline_moca_raw':mq['raw'],'baseline_moca_row':None if mr is None else mr['_row'],'baseline_moca_month':mm,'baseline_moca_QC_reasons':';'.join(mq['reasons']),'baseline_moca_flags':';'.join(mq['flags']),'education':edu,'education_row':None if er is None else er['_row'],'education_month':em,'death_month':death,'death_conflict':dc,'invalid_death_date':invaliddeath,'death_withdrawal_without_date':wddeath and death is None,'earliest_withdrawal_month':earliestwd}
 ar,arefs,ast=unique([r for r in by['age'].get(p,[]) if r['EVENT_ID']=='BL'],['AGE_AT_VISIT']);age=number(ar.get('AGE_AT_VISIT')) if ar else None;row['age_index']=age+(idx-ref)/12 if age is not None and 18<=age<=110 and ref is not None else None;log(p,'age_index','age',ar,arefs,ref,idx,ast)
 sx=[r for r in by['demographics'].get(p,[]) if r.get('PAG_NAME')=='SCREEN'];sr,srefs,sst=unique(sx,['SEX']);sex=number(sr['SEX']) if sr and integer(sr.get('SEX'),0,1) else None;us={number(r['SEX']) for r in ub if integer(r.get('SEX'),0,1)};sxconf=sex is not None and (len(us)>1 or len(us)==1 and sex!=next(iter(us)))
 row['sex_male']=None if sxconf else sex;row['sex_annotation_source']='native_SCREEN_static_corroborated_unified_BL' if sex is not None and us=={sex} else 'native_SCREEN_static_only' if sex is not None and not sxconf else 'missing_or_conflicting';row['sex_source_discordance']=sxconf;row['sex_native_preindex_record']=any(r['EVENT_ID'] in ['SC','BL'] and month(r['INFODT']) is not None and month(r['INFODT'])<=idx for r in sx);log(p,'sex_static','demographics',sr,srefs,None if sr is None else month(sr.get('INFODT')),idx,sst)
 chosen={}
 for name,fields,motor in [('diagnosis_history',['PDDXDT'],False),('part2',p2+['NP2PTOT'],False),('part3',p3+['NP3TOT','PDSTATE','PDTRTMNT','PDMEDYN','DBSYN'],True),('gds',gd,False)]:
  r,refs,mo,st=select(by[name].get(p,[]),idx,fields,motor=motor);chosen[name]=r;log(p,name,name,r,refs,mo,idx,st)
 dx=month(chosen['diagnosis_history']['PDDXDT']) if chosen['diagnosis_history'] else None;duration=(idx-dx)/12 if dx is not None else None;row['diagnosis_duration_years']=duration if duration is not None and 0<=duration<=80 else None;row['part2']=itemscore(chosen['part2'],p2,'NP2PTOT');row['part3']=itemscore(chosen['part3'],p3,'NP3TOT')
 gr=chosen['gds'];row['gds15']=sum(1-float(gr[k]) if k in reverse else float(gr[k]) for k in gd) if gr and all(integer(gr.get(k),0,1) for k in gd) else None
 motor=chosen['part3'];med=None;state='unknown';medsrc='missing';contradiction=False
 if motor:
  tr=number(motor.get('PDTRTMNT'));pdmed=number(motor.get('PDMEDYN'));dbs=number(motor.get('DBSYN'));rawstate=motor.get('PDSTATE','')
  if pdmed in [0,1]:med=pdmed;medsrc='explicit_PDMEDYN'
  elif tr==0 and dbs!=1:med=0.;medsrc='explicit_no_treatment'
  contradiction=(tr==0 and(pdmed==1 or dbs==1 or rawstate in ['ON','OFF']))
  if contradiction:med=None;medsrc='contradictory_fields'
  if rawstate in ['ON','OFF'] and not contradiction:state=rawstate.lower()
  elif tr==0 and not contradiction:state='untreated'
  elif tr==1:state='treated_unknown'
 row['pd_medication']=med;row['motor_state']=state;row['medication_source']=medsrc;row['medication_contradiction']=contradiction;row['DBS_present_selected']=bool(motor and number(motor.get('DBSYN'))==1)
 detailrows={}
 for name,fields,fk,maxf in [('sdmt',sfields,'SDMTVRSN',2),('hvlt',hfields,'HVLTVRSN',6)]:
  r,refs,mo,st=select(by[name].get(p,[]),idx,fields);detailrows[name]=r;log(p,'baseline_'+name,name,r,refs,mo,idx,st);row['baseline_'+name+'_form']=int(float(r[fk])) if r and integer(r.get(fk),1,maxf) else None
 row['baseline_sdmt']=detail(detailrows['sdmt'],'SDMT');row['baseline_hvlt_immediate']=detail(detailrows['hvlt'],'HVLT_immediate');row['baseline_hvlt_delayed']=detail(detailrows['hvlt'],'HVLT_delayed')
 enh=['baseline_sdmt','baseline_hvlt_immediate','baseline_hvlt_delayed','baseline_sdmt_form','baseline_hvlt_form'];core=['age_index','sex_male','education','baseline_moca','diagnosis_duration_years','part2','part3','gds15','pd_medication'];row['enhanced_tests_complete']=all(row[k] is not None for k in enh);missing=[k for k in core if row[k] is None]+(['motor_state'] if state=='unknown' else []);row['core_missing_fields']=';'.join(missing);row['core_complete']=not missing
 result,_=resolve(row,resolve_src,select(by['diagnosis_history'][p],idx,['PDDXDT'])[3]);row=result
 erows=by['participant_status'][p];enroll={number(v['ENROLL_AGE']) for v in erows};adult=row['age_index'] is not None and 18<=row['age_index']<=110
 if not adult and not (row['age_index'] is None and len(enroll)==1 and None not in enroll and 18<=next(iter(enroll))<=110):reasons.append('adult_age_unverified')
 row['baseline_holds']=';'.join(sorted(set(reasons)));row['baseline_flags']=';'.join(flags);row['baseline_ready']=not reasons;row['matched_baseline_ready']=not reasons and row['enhanced_tests_complete'];risk.append(row)
 for h in [12,24,36]:
  r,refs,mo,st=select(by['moca'].get(p,[]),idx,mfields,h=h);q=moca(r,edu,st);log(p,'MoCA'+str(h),'moca',r,refs,mo,idx,st)
  if death is not None and mo is not None and mo>death:q['reasons'].append('assessment_after_death_month')
  if death is not None and mo==death:q['flags'].append('assessment_same_death_month')
  if dc:q['reasons'].append('conflicting_death_dates')
  if invaliddeath:q['reasons'].append('invalid_death_date')
  wdreview=mo is not None and earliestwd is not None and mo>earliestwd
  if wdreview:q['flags'].append('assessment_after_withdrawal_requires_review')
  endpointready=not q['reasons'] and not wdreview
  compact=row['baseline_ready'] and row['enhanced_tests_complete'] and endpointready
  o=orig.get(component,{})
  ends.append(dict(PATNO=p,component=component,horizon=h,index_month=idx,index_year=row['index_year'],index_source=refsource,baseline_moca_event=mr['EVENT_ID'] if mr else '',baseline_moca_row=mr['_row'] if mr else None,sdmt_row=detailrows['sdmt']['_row'] if detailrows['sdmt'] else None,hvlt_row=detailrows['hvlt']['_row'] if detailrows['hvlt'] else None,endpoint_row=r['_row'] if r else None,endpoint_month=mo,elapsed=mo-idx if mo is not None else None,baseline_pass=row['baseline_ready'],baseline_holds=row['baseline_holds'],four_scores=row['enhanced_tests_complete'],core_complete=row['resolved_core_complete'],endpoint_QC=endpointready,endpoint_reasons=';'.join(q['reasons']+(['withdrawal_review'] if wdreview else [])),eligible=compact,reserved=component in reserved,original=component in orig,old_horizon=component in old[h],clinical_minus_RNA_month=idx-int(o['RNA_index_month']) if o else None,window_status_at_export='not_open' if idx+h-3>2026*12+10 else 'open_incomplete' if idx+h+3>2026*12+10 else 'elapsed'))
put('restricted/horizon_ledger.csv',ends);put('restricted/no_clinical_anchor.csv',omitted)
put('restricted/baseline_features.csv',risk)
H=(12,24,36)
flow=[];members={};calendar=[]; landmarks=[]
for h in H:
 z=[r for r in ends if r['horizon']==h];match=[r for r in z if r['eligible']];pool=[r for r in match if not r['reserved']]
 members[h]={r['PATNO'] for r in pool}
 poolcomp={r['component'] for r in pool};assert len(poolcomp)==len(pool)
 hist=[r for r in pool if r['index_year']<2023];hset={r['component'] for r in hist}
 s=dict(horizon=h,PD_screen_frame=len(ids),missing_or_conflicting_anchor=len(omitted),adult_identity_PD_baseline_MoCA_pass=sum(r['baseline_pass'] for r in z),adult_baseline_four_scores=sum(r['baseline_pass'] and r['four_scores'] for r in z),MoCA_pair_baseline_eligible=sum(r['baseline_pass'] and r['endpoint_QC'] for r in z),all_four_score_matched=len(match),reserved_stage_A_overlap=len(match)-len(pool),available_excluding_stage_A=len(pool),original_horizon=len(old[h]),retained_original_horizon=len(poolcomp&old[h]),old_horizon_no_longer_eligible=len(old[h]-poolcomp),original_351_new_at_horizon=len((poolcomp&set(orig))-old[h]),additional_outside_original_351=len(poolcomp-set(orig)),net_increase=len(pool)-len(old[h]),pre2023_total=len(hist),pre2023_additional_outside_351=len(hset-set(orig)),pre2023_net_increase=len(hist)-len(old[h]),resolved_core_complete=sum(r['core_complete'] for r in pool))
 flow.append(s)
 for yr in sorted({r['index_year'] for r in z}):
  sub=[r for r in z if r['index_year']==yr and not r['reserved'] and r['baseline_pass'] and r['four_scores']]
  calendar.append(dict(horizon=h,index_year=yr,baseline_ready=len(sub),eligible=sum(r['eligible'] for r in sub),**{key:sum(not r['eligible'] and r['window_status_at_export']==key for r in sub) for key in ['not_open','open_incomplete','elapsed']}))
 for comp_id,t in sorted(orig.items()):
  entry=next((r for r in z if r['component']==comp_id),{})
  landmarks.append(dict(component=comp_id,horizon=h,RNA_horizon=comp_id in old[h],clinical_horizon=comp_id in poolcomp,clinical_minus_RNA_month=entry.get('clinical_minus_RNA_month'),reason_if_excluded='' if comp_id in poolcomp else entry.get('baseline_holds') or ('detailed_scores_or_forms' if entry and not entry['four_scores'] else entry.get('endpoint_reasons')) or 'outside_native_PD_frame_or_no_anchor'))
put('eligibility_counts.csv',flow);put('calendar_and_maturity.csv',calendar);put('restricted/original_landmark_comparison.csv',landmarks)
print('Main counts',json.dumps(flow),flush=True)
groups=load(C/'independent_validation_contract_2026-10-31/config/Feature_Definitions.json')['active_numeric']
base=pd.DataFrame([r for r in risk if r['PATNO'] in set.union(*members.values())]).copy()
base['RNA_index_month']=base['index_month']  # old QC code's argument name for landmark
base['baseline_hvlt_form']=base['baseline_hvlt_form'].map(lambda x:str(int(x)))
prior=load(A/'evidence/Clinical_Feature_Definitions.json')
mapping={'semantic_animals':'Animal_fluency','working_memory_LNS':'LNS','visuospatial_JLO_raw':'JLO','hvlt_recognition_discrimination':'HVLT_recognition_discrimination','RBDSQ_full':'RBDSQ_full','SCOPA_GI':'SCOPA_GI','SCOPA_urinary':'SCOPA_urinary','SCOPA_cardiovascular':'SCOPA_cardiovascular','STAI_trait':'STAI_trait','apathy':'Apathy','cognitive_complaint':'Cognitive_complaint','orthostatic_systolic_drop':'Systolic_BP_drop','orthostatic_diastolic_drop':'Diastolic_BP_drop'}
D=[dict(d,new_name=mapping[d['key']]) for d in prior if d['key'] in mapping]
forward={3,4,6,7,9,12,13,14,17,18,22,24,25,28,29,31,32,35,37,38,40}
sources={};provenance=[];qc=collections.Counter();case_flags=collections.Counter()
def num(x):
 try:v=float(x);return v if math.isfinite(v) else None
 except (TypeError,ValueError):return None
def month(x):
 if not re.fullmatch(r'\d{2}/\d{4}',str(x)):return None
 m,y=map(int,x.split('/'));return y*12+m if 1<=m<=12 else None
def fields_for(d):
 f=list(d['fields']+d.get('extra',[]))
 if d['key']=='working_memory_LNS':f += [f'LNS{i}{j}' for i in range(1,8) for j in 'ABC']
 if d['key']=='visuospatial_JLO_raw':f += [f'BJLOT{i}' for i in range(1,31)]+['JLO_TOTCALC']
 if d['key']=='hvlt_recognition_discrimination':f+=['HVLTRT1','HVLTRT2','HVLTRT3','HVLTRDLY','HVLTVRSN']
 return list(dict.fromkeys(f))
def choose(df,pid,ix,fs):
 z=df[(df.PATNO==pid)&df.EVENT_ID.isin(['SC','BL'])].copy()
 z['_month']=z.INFODT.map(month);z=z[z._month.notna()]
 z=z[(ix-z._month).between(0,3)]
 if not len(z):return None,'no_timed_source',[]
 z=z[z._month.eq(z._month.max())]
 ev='BL' if z.EVENT_ID.eq('BL').any() else 'SC';z=z[z.EVENT_ID.eq(ev)]
 sig=z[fs].drop_duplicates()
 if len(sig)>1:return None,'conflicting_same_month_event',z['_line'].astype(int).tolist()
 r=z.sort_values('_line').iloc[0].to_dict();return r,'selected',z['_line'].astype(int).tolist()
def calc(r,d,b):
 f=d['fields'];vals=[num(r.get(k)) for k in f]
 if any(x is None for x in vals):return None,'missing_item',[]
 method=d['method'];flags=[]
 if method=='scopa':vals=[3 if x==9 and k in ['SCAU'+str(i) for i in range(8,14)] else x for k,x in zip(f,vals)]
 if any(x<d['lo'] or x>d['hi'] for x in vals):return None,'range_or_special_code',[]
 if method not in ['sysdrop','diadrop'] and any(x!=int(x) for x in vals):return None,'noninteger_score',[]
 if method=='single':v=vals[0]
 elif method=='scopa':v=sum(vals)
 elif method=='rbd':v=sum(vals[:12])+int(any(vals[12:]))
 elif method=='stai':v=sum(x if int(k.replace('STAIAD','')) in forward else 5-x for k,x in zip(f,vals))
 elif method=='hvlt_disc':
  if vals[1]+vals[2]>12:return None,'invalid_false_positive_count',[]
  trials=[num(r.get(k)) for k in ['HVLTRT1','HVLTRT2','HVLTRT3','HVLTRDLY']]
  if any(x is None for x in trials):return None,'HVLT_binding_missing',[]
  if r.get('HVLTVRSN')!=b['baseline_hvlt_form'] or sum(trials[:3])!=num(b['baseline_hvlt_immediate']) or trials[3]!=num(b['baseline_hvlt_delayed']):return None,'HVLT_not_same_frozen_assessment_values_form',[]
  v=vals[0]-vals[1]-vals[2]
 elif method in ['sysdrop','diadrop']:
  s,di,st,dt=vals
  if not(50<=s<=260 and 50<=st<=260 and 30<=di<=160 and 30<=dt<=160 and di<s and dt<st):return None,'BP_range_review',[]
  v=s-st if method=='sysdrop' else di-dt
  flags+=['postural_difference_not_OH_diagnosis','historical_standing_duration_not_individually_recorded','same_visit_medication_timing_not_certified']
 else:raise ValueError(method)
 if d['key']=='working_memory_LNS':
  total=0;stopped=False;tail=False
  for i in range(1,8):
   items=[num(r.get(f'LNS{i}{j}')) for j in 'ABC']
   if stopped:
    if any(x not in [0,None] for x in items):return None,'LNS_positive_after_discontinue',[]
    tail|=any(x is None for x in items);continue
   if any(x not in [0,1] for x in items):return None,'LNS_incomplete_before_discontinue',[]
   total+=sum(items)
   if sum(items)==0:stopped=True
  if total!=v:return None,'LNS_item_total_discordance',[]
  if tail:flags.append('LNS_unadministered_tail_zero_by_discontinue')
 if d['key']=='visuospatial_JLO_raw':
  form=r.get('BENTONVERSION')
  if form not in ['1','2']:return None,'JLO_unknown_form',[]
  active=range(1,31,2) if form=='1' else range(2,31,2)
  other=range(2,31,2) if form=='1' else range(1,31,2)
  act=[num(r.get(f'BJLOT{i}')) for i in active];oth=[num(r.get(f'BJLOT{i}')) for i in other]
  if any(x not in [0,1] for x in act):return None,'JLO_incomplete_active_form',[]
  if any(x is not None for x in oth):return None,'JLO_extra_form_items',[]
  if sum(act)!=v or num(r.get('JLO_TOTCALC'))!=2*v:return None,'JLO_item_total_discordance',[]
  if not 0<=v<=15:return None,'JLO_raw_outside_15',[]
 return float(v),'ready_observed',flags
matrix=base.copy()
for d in D:
 f=next(raw.glob(d['prefix']+'*.csv'));df=read(f);df['_line']=np.arange(len(df))+2;sources[d['new_name']]=str(f)
 vals=[];forms=[]
 for b in base.to_dict('records'):
  r,status,lines=choose(df,b['PATNO'],int(b['RNA_index_month']),fields_for(d))
  flags=[];value=None
  if r is not None:value,status,flags=calc(r,d,b)
  vals.append(value);forms.append(r.get('BENTONVERSION','unknown') if r and value is not None else 'unknown')
  qc[(d['new_name'],status)]+=1
  for k in flags:case_flags[(d['new_name'],k)]+=1
  provenance.append({'PATNO':b['PATNO'],'feature':d['new_name'],'status':status,'value':value,'flags':flags,'source':str(f),'source_lines':lines,'month':int(r['_month']) if r else None,'EVENT_ID':r['EVENT_ID'] if r else None,'raw_inputs':{k:r.get(k,'') for k in fields_for(d)} if r else None})
 matrix[d['new_name']]=vals
 if d['new_name']=='JLO':matrix['jlo_form']=forms
# Existing three enhancement values copied with no source/value change.
for n,k in [('SDMT','baseline_sdmt'),('HVLT_immediate','baseline_hvlt_immediate'),('HVLT_delayed','baseline_hvlt_delayed')]:matrix[n]=pd.to_numeric(matrix[k],errors='raise')
# Source consensus APOE is a retrospective germline trait; no dynamic measurement-date substitution.
gfile=next(raw.glob('iu_genetic_consensus*.csv'));g=read(gfile);gen=[]
for b in base.to_dict('records'):
 z=g[g.PATNO.eq(b['PATNO'])];valid=z[z.APOE.str.fullmatch(r'E[234]/E[234]')]
 v=None;status='APOE_missing_or_invalid';flags=[]
 if len(valid)==len(z) and len(valid) and len({tuple(sorted(x.split('/'))) for x in valid.APOE})==1:
  # Notes containing explicit identity uncertainty are held; no absence-as-noncarrier coding.
  risk=valid.NOTES.str.contains('mismatch|discord|discrep|identity|swap|mixed|duplicate|conflict|contamin|fingerprint',case=False,regex=True)
  if risk.any():status='APOE_identity_note_review'
  elif not valid.WGS.eq('X').all():status='APOE_platform_inventory_review'
  else:v=float(valid.APOE.iloc[0].count('E4'));status='ready_static_germline';flags=['retrospective_consensus_not_known_at_baseline','ancestry_portability_not_established','WGS_inventory_flag_not_per_locus_QC']
 gen.append(v);qc[('APOE_e4_dosage',status)]+=1
 provenance.append({'PATNO':b['PATNO'],'feature':'APOE_e4_dosage','status':status,'value':v,'flags':flags,'source':str(gfile),'genotypes':valid.APOE.tolist(),'WGS_inventory':valid.WGS.tolist(),'notes_held_for_identity':bool(len(valid) and valid.NOTES.str.contains('mismatch|discord|discrep|identity|swap|mixed|duplicate|conflict|contamin|fingerprint',case=False,regex=True).any())})
matrix['APOE_e4_dosage']=gen
# NLR: source-month/event selection before laboratory QC, no favourable fallback.
lf=next((B/'source_audit_2026-10-20/raw/ppmi/recovered_members').glob('Blood_Chemistry*.csv'));lab=read(lf);lab['_line']=np.arange(len(lab))+2;lab=lab[lab.LTSTNAME.isin(['Neutrophils','Lymphocytes'])&lab.EVENT_ID.isin(['SC','BL'])].copy();lab['_month']=lab.LCOLLDT.map(month)
allow_comments={'','Result verified by repeat analysis','Group ordered to complete visit requirements.'}
nv=[];nlraw=[];nlr_audit=[]
for b in base.to_dict('records'):
 ix=int(b['RNA_index_month']);z=lab[lab.PATNO.eq(b['PATNO'])&lab._month.notna()].copy();z=z[(ix-z._month).between(0,3)]
 value=None;status='NLR_no_timed_source';flags=[];draw_values=[]
 if len(z):
  z=z[z._month.eq(z._month.max())];ev='BL' if z.EVENT_ID.eq('BL').any() else 'SC';z=z[z.EVENT_ID.eq(ev)]
  status='ready_observed'
  for key,d in z.groupby(['EVENT_ID','LCOLLDT','COLLTM','LABCODE'],dropna=False,sort=True):
   pair={};bad=None
   for test in ['Neutrophils','Lymphocytes']:
    q=d[d.LTSTNAME.eq(test)]
    if not len(q):bad='NLR_incomplete_draw';break
    if q[['LSIRES','LSIUNIT']].drop_duplicates().shape[0]!=1:bad='NLR_conflicting_result_in_draw';break
    if not q.LTSTCOMM.isin(allow_comments).all():bad='NLR_specimen_comment_hold';break
    if not q.LRESFLG.isin(['','H','L']).all():bad='NLR_unknown_result_flag';break
    x=num(q.LSIRES.iloc[0])
    if x is None or x<=0 or q.LSIUNIT.iloc[0]!='GI/L':bad='NLR_nonpositive_nonnumeric_or_unit';break
    reports=[month(x) for x in q.LRPTDT]
    if any(m is None for m in reports):bad='NLR_missing_report_month';break
    if any(m>ix for m in reports):bad='NLR_report_after_landmark';break
    if any(m<int(q._month.iloc[0]) for m in reports):bad='NLR_report_precedes_collection';break
    pair[test]=x
    if q.LRESFLG.isin(['H','L']).any():flags.append('abnormal_range_flag_retained')
    if q.LTSTCOMM.eq('Result verified by repeat analysis').any():flags.append('repeat_verified_retained')
   if bad:status=bad;break
   draw_values.append(pair)
  if status=='ready_observed':
   signatures={(r['Neutrophils'],r['Lymphocytes']) for r in draw_values}
   if len(signatures)!=1:status='NLR_conflicting_latest_draws'
   else:
    neu,lym=next(iter(signatures));ratio=neu/lym;value=math.log(ratio);flags.append('natural_log_ratio_sample_local')
 nv.append(value);nlraw.append(math.exp(value) if value is not None else None);qc[('NLR',status)]+=1
 info={'PATNO':b['PATNO'],'feature':'NLR','status':status,'value':value,'ratio':math.exp(value) if value is not None else None,'flags':sorted(set(flags)),'source':str(lf),'source_lines':z['_line'].astype(int).tolist(),'source_month':int(z['_month'].iloc[0]) if len(z) else None,'source_records':z.drop(columns=['_month']).to_dict('records')}
 provenance.append(info);nlr_audit.append(info)
matrix['NLR']=nv;matrix['NLR_raw_ratio']=nlraw
mx=matrix.set_index('PATNO',verify_integrity=True)
availability=[]
for stats in flow:
 h=stats['horizon'];sub=mx.loc[sorted(members[h])]
 for tag,fs in groups.items():
  good=sub[fs].apply(pd.to_numeric,errors='coerce').notna().all(axis=1)
  if tag!='compact4':good &= sub.jlo_form.isin(['1','2'])
  if tag=='core_plus18':good &= sub.motor_state.isin(['untreated','off','on','treated_unknown'])
  stats[tag+'_fully_observed']=int(good.sum())
  stats[tag+'_fully_observed_pre2023']=int((good & sub.index_year.lt(2023)).sum())
  if tag=='compact4':assert good.all()
 for col in groups['core_plus18']:
  availability.append(dict(horizon=h,predictor=col,total=len(sub),observed=int(pd.to_numeric(sub[col],errors='coerce').notna().sum())))
put('eligibility_counts.csv',flow);put('predictor_availability.csv',availability)
put('restricted/optional_feature_availability.csv',mx[groups['core_plus18']].apply(pd.to_numeric,errors='coerce').notna().assign(jlo_observed=mx.jlo_form.isin(['1','2']),motor_state_observed=mx.motor_state.isin(['untreated','off','on','treated_unknown'])).reset_index().to_dict('records'))
assert all(ac.sha(p)==v for p,v in hashes.items())
(O/'input_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('Finished',flush=True)
