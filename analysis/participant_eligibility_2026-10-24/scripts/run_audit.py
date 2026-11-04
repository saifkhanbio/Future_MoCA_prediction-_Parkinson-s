"""RNA-defined participant/timing/cognitive eligibility audit. No fitting or associations."""
from audit_common import *
import numpy as np
for z in json.loads((D/'evidence/Input_Manifest.json').read_text()):assert sha(z['path'])==z['sha256'],z['name']
assert sha(D/'config/Audit_Rules.json')==json.loads((D/'config/Audit_Rules_Record.json').read_text())['sha256']
reg=read('rna_registry');bl=[r for r in reg if r['visit']=='BL'];ids={r['PATNO'] for r in bl};assert len(bl)==len(ids)==410
qc={r['sample']:r for r in read('rna_qc')};components=group(read('identity_components'),'source_key')
comp={k:next(iter(v)) if len(v)==1 else None for k,rr in components.items() for v in [{r['component_id'] for r in rr}]}
compn=collections.Counter(comp.get('PPMI:'+r['PATNO']) for r in bl)
keys=['blood_collection','moca','education','diagnosis','baseline_start','hvlt','sdmt','conclusion','age','demographics','diagnosis_history','part2','part3','gds','cell_covariates']
source={k:[r for r in read(k) if r['PATNO'] in ids] for k in keys};by={k:group(v) for k,v in source.items()}
u=[r for r in read('anchor_frame',keep=['PATNO','EVENT_ID','visit_date','Death_Date','PRIMDIAG','COHORT','subgroup','SEX']) if r['PATNO'] in ids];by['unified']=group(u)
p2=[k for k in source['part2'][0] if k.startswith('NP2') and k!='NP2PTOT'];p3=[k for k in source['part3'][0] if k.startswith('NP3') and k!='NP3TOT'];gd=[k for k in source['gds'][0] if k.startswith('GDS')];assert (len(p2),len(p3),len(gd))==(13,33,15)
reverse={'GDSSATIS','GDSGSPIR','GDSHAPPY','GDSALIVE','GDSENRGY'};mfields=list(ITEMS)+['MCATOT','MCAVFNUM'];hfields=['HVLTRT1','HVLTRT2','HVLTRT3','HVLTRDLY','HVLTVRSN'];sfields=['SDMTOTAL','SDMTVRSN']
counts=np.load(I['rna_matrix'],mmap_mode='r');ann=read('rna_gene_annotation');assert counts.shape==(730,58780) and len(ann)==58780 and len({x['Geneid'] for x in ann})==58780
ledger=[];risk=[];endpoints=[];details=[];timing=[];exposure=[]
def log(p,role,source_name,r,refs,m,idx,st):
 ledger.append({'PATNO':p,'role':role,'source':source_name,'source_rows':','.join(map(str,refs)),'selected_source_row':None if r is None else r['_row'],'event':'' if r is None else r.get('EVENT_ID',''),'assessment_month':m,'index_month':idx,'month_minus_index':None if m is None or idx is None else m-idx,'selection_status':st})
for rr in sorted(bl,key=lambda r:r['PATNO']):
 p=rr['PATNO'];idx=month(rr['collection_months']);q=qc[rr['sample']];reasons=[];flags=[];component=comp.get('PPMI:'+p);v=counts[int(rr['count_row'])]
 binding=(q['PATNO']==p and q['visit']=='BL' and q['count_row']==rr['count_row'] and q['collection_months']==rr['collection_months'] and hashlib.sha256(v.tobytes()).hexdigest()==q['count_vector_sha256'])
 rawpass=bool(np.issubdtype(v.dtype,np.integer) and np.all(v>=0) and v.sum()>0)
 technical=rr['release_QC']=='pass' and not truth(rr['pool']) and truth(q['technical_count_pass']) and truth(q['metadata_numeric_valid']) and truth(q['header_matches_specimen_tokens']) and binding and rawpass
 if not technical:reasons.append('RNA_release_technical_or_binding_hold')
 if not component or compn[component]!=1:reasons.append('missing_ambiguous_or_duplicate_identity_component')
 br,brefs,bst=unique([r for r in by['blood_collection'].get(p,[]) if r['EVENT_ID']=='BL'],['BLDDRDT','BLDRNA']);bm=month(br.get('BLDDRDT')) if br else None
 collection_ok=idx is not None and bm==idx and br is not None and br['BLDRNA']=='1'
 if not collection_ok:reasons.append('RNA_native_collection_not_confirmed')
 if br and month(br['INFODT'])!=bm:flags.append('collection_form_month_differs_from_draw')
 bs,bsrefs,bsst=unique([r for r in by['baseline_start'].get(p,[]) if r['EVENT_ID']=='BL'],['INFODT']);ref=month(bs['INFODT']) if bs else None;refsource='native_BL_start'
 ub=[r for r in by['unified'].get(p,[]) if r['EVENT_ID']=='BL'];ur,urefs,ust=unique(ub,['visit_date'])
 if bsst=='conflicting_records':reasons.append('baseline_reference_conflict')
 if ref is None and bsst!='conflicting_records':ref=month(ur['visit_date']) if ur else None;refsource='unified_BL_reference';flags.append('unified_baseline_reference')
 if idx is None or ref is None or abs(idx-ref)>1:reasons.append('RNA_baseline_reference_discordance')
 log(p,'RNA_collection','blood_collection',br,brefs,bm,idx,bst);log(p,'baseline_reference', 'baseline_start' if refsource=='native_BL_start' else 'anchor_frame',bs if refsource=='native_BL_start' else ur,bsrefs if refsource=='native_BL_start' else urefs,ref,idx,bsst if refsource=='native_BL_start' else ust)
 dg=[r for r in by['diagnosis'].get(p,[]) if r['EVENT_ID'] in ['SC','BL'] and month(r['INFODT']) is not None and idx is not None and month(r['INFODT'])<=idx]
 diagvals={number(r['PRIMDIAG']) for r in dg if number(r['PRIMDIAG']) is not None};native_pd=1 in diagvals and not (diagvals-{1})
 if not native_pd:reasons.append('native_preindex_PD_not_confirmed' if not diagvals else 'contrary_native_preindex_diagnosis')
 for r in dg:log(p,'native_preindex_diagnosis','diagnosis',r,[r['_row']],month(r['INFODT']),idx,'PD' if number(r['PRIMDIAG'])==1 else 'contrary_or_missing')
 er,erefs,em,est=select(by['education'].get(p,[]),idx,['EDUCYRS'],education=True);edu=number(er['EDUCYRS']) if er and integer(er['EDUCYRS'],0,40) else None
 log(p,'education','education',er,erefs,em,idx,est)
 if er and er.get('EDUCYRS_EQUIV','').strip() and number(er['EDUCYRS_EQUIV'])!=edu:flags.append('education_equivalent_differs')
 mr,mrefs,mm,mst=select(by['moca'].get(p,[]),idx,mfields);mq=moca(mr,edu,mst);reasons+=['baseline_moca_'+x for x in mq['reasons']];log(p,'baseline_moca','moca',mr,mrefs,mm,idx,mst)
 deathstrings={r['Death_Date'] for r in by['unified'].get(p,[]) if r.get('Death_Date','').strip()};deaths={month(s) for s in deathstrings};invalid_death=None in deaths;deaths.discard(None);death=min(deaths) if deaths else None;death_conflict=len(deaths)>1
 if death is not None and idx is not None and idx>death:reasons.append('baseline_after_death_month')
 if death_conflict:reasons.append('conflicting_death_dates')
 if invalid_death:reasons.append('unparseable_documented_death_date')
 wd=by['conclusion'].get(p,[]);wdmonths=[month(r['WDDT']) for r in wd if month(r['WDDT']) is not None];earliest_wd=min(wdmonths) if wdmonths else None
 wddeath=any(r.get('WDDEATH')=='1' for r in wd)
 row={'PATNO':p,'component_id':component,'sample_id':rr['sample'],'count_row':int(rr['count_row']),'recruitment_group':rr['recruitment_group'],'SWEDD_recruitment':rr['recruitment_group']=='SWEDD','RNA_index_month':idx,'RNA_technical_ready':technical,'RNA_native_collection_confirmed':collection_ok,'RNA_baseline_reference_month':ref,'reference_source':refsource,'reference_delta_months':None if idx is None or ref is None else idx-ref,'native_preindex_PD_confirmed':native_pd,'native_diagnosis_codes':'|'.join(str(int(x)) for x in sorted(diagvals)),'education':edu,'education_source_row':None if er is None else er['_row'],'education_month':em,'baseline_moca':mq['adjusted'],'baseline_moca_raw':mq['raw'],'baseline_moca_month':mm,'baseline_moca_source_row':None if mr is None else mr['_row'],'baseline_moca_flags':';'.join(mq['flags']),'baseline_moca_QC_reasons':';'.join(mq['reasons']),'death_month':death,'death_conflict':death_conflict,'invalid_death_date':invalid_death,'death_withdrawal_without_date':wddeath and death is None,'earliest_withdrawal_month':earliest_wd,'RIN':number(q['RIN']),'RIN_below6_review':truth(q['RIN_below6_review']),'technical_review_flag':truth(q['technical_review_flag']),'phase':q['phase_prefix'],'plate':q['Plate'],'library_total':int(v.sum()),'RNA_binding_pass':binding}
 ar,arefs,ast=unique([r for r in by['age'].get(p,[]) if r['EVENT_ID']=='BL'],['AGE_AT_VISIT']);age=number(ar['AGE_AT_VISIT']) if ar else None
 row['age_index']=age+(idx-ref)/12 if age is not None and 18<=age<=110 and idx is not None and ref is not None else None;log(p,'age_index','age',ar,arefs,ref,idx,ast)
 sx=[r for r in by['demographics'].get(p,[]) if r['EVENT_ID'] in ['SC','BL'] and month(r['INFODT']) is not None and idx is not None and month(r['INFODT'])<=idx]
 sr,srefs,sst=unique(sx,['SEX']);row['sex_male']=number(sr['SEX']) if sr and integer(sr['SEX'],0,1) else None;log(p,'sex_male','demographics',sr,srefs,None if sr is None else month(sr['INFODT']),idx,sst)
 us={number(r['SEX']) for r in ub if integer(r['SEX'],0,1)};row['unified_BL_sex_male']=next(iter(us)) if len(us)==1 else None;row['sex_source_discordance']=row['sex_male'] is not None and row['unified_BL_sex_male'] is not None and row['sex_male']!=row['unified_BL_sex_male']
 if row['sex_source_discordance']:row['sex_male']=None;flags.append('sex_source_discordance')
 chosen={}
 for name,fields,motor in [('diagnosis_history',['PDDXDT'],False),('part2',p2+['NP2PTOT'],False),('part3',p3+['NP3TOT','PDSTATE','PDTRTMNT','PDMEDYN','DBSYN'],True),('gds',gd,False)]:
  r,refs,mo,st=select(by[name].get(p,[]),idx,fields,motor=motor);chosen[name]=r;log(p,name,name,r,refs,mo,idx,st)
 dx=month(chosen['diagnosis_history']['PDDXDT']) if chosen['diagnosis_history'] else None;duration=(idx-dx)/12 if idx is not None and dx is not None else None
 row['diagnosis_duration_years']=duration if duration is not None and 0<=duration<=80 else None;row['part2']=itemscore(chosen['part2'],p2,'NP2PTOT');row['part3']=itemscore(chosen['part3'],p3,'NP3TOT')
 gr=chosen['gds'];row['gds15']=sum(1-float(gr[k]) if k in reverse else float(gr[k]) for k in gd) if gr and all(integer(gr.get(k),0,1) for k in gd) else None
 motor=chosen['part3'];med=None;state='unknown';medsrc='missing'
 if motor:
  tr=number(motor.get('PDTRTMNT'));pd=number(motor.get('PDMEDYN'));dbs=number(motor.get('DBSYN'));rawstate=motor.get('PDSTATE','')
  if pd in [0,1]:med=pd;medsrc='explicit_PDMEDYN'
  elif tr==0 and dbs!=1:med=0.;medsrc='explicit_no_treatment'
  contradiction=(tr==0 and (pd==1 or dbs==1 or rawstate in ['ON','OFF']))
  if contradiction:med=None;medsrc='contradictory_fields'
  if rawstate in ['ON','OFF'] and not contradiction:state=rawstate.lower()
  elif tr==0 and not contradiction:state='untreated'
  elif tr==1:state='treated_unknown'
 row['pd_medication']=med;row['motor_state']=state;row['medication_source']=medsrc
 base_detail={}
 for name,fields,forms,maxform in [('sdmt',sfields,'SDMTVRSN',2),('hvlt',hfields,'HVLTVRSN',6)]:
  r,refs,mo,st=select(by[name].get(p,[]),idx,fields);base_detail[name]=(r,refs,mo,st);log(p,'baseline_'+name,name,r,refs,mo,idx,st)
  row['baseline_'+name+'_form']=int(float(r[forms])) if r and integer(r.get(forms),1,maxform) else None
 row['baseline_sdmt']=detail(base_detail['sdmt'][0],'SDMT');row['baseline_hvlt_immediate']=detail(base_detail['hvlt'][0],'HVLT_immediate');row['baseline_hvlt_delayed']=detail(base_detail['hvlt'][0],'HVLT_delayed')
 enhanced=all(row[k] is not None for k in ['baseline_sdmt','baseline_hvlt_immediate','baseline_hvlt_delayed','baseline_sdmt_form','baseline_hvlt_form']);row['enhanced_baseline_complete']=enhanced
 corekeys=['age_index','sex_male','education','baseline_moca','diagnosis_duration_years','part2','part3','gds15','pd_medication'];row['core_complete']=all(row[k] is not None for k in corekeys) and state!='unknown'
 cb=by['cell_covariates'].get(p,[]);row['historical_CBC_record']=bool(cb);row['CBC_nonfuture_complete']=len(cb)==1 and truth(cb[0]['complete_wbc_and_five_percentages']) and month(cb[0]['rna_collection_month'])==idx and number(cb[0]['month_gap_lab_minus_rna']) is not None and number(cb[0]['month_gap_lab_minus_rna'])<=0
 row['CBC_lag_months']=number(cb[0]['month_gap_lab_minus_rna']) if len(cb)==1 else None
 row['baseline_hold_reasons']=';'.join(sorted(set(reasons)));row['baseline_flags']=';'.join(flags);row['diagnosis_supported_baseline_ready']=not reasons;row['PD_main_baseline_candidate']=not reasons and not row['SWEDD_recruitment'];row['PD_matched_baseline_candidate']=row['PD_main_baseline_candidate'] and enhanced
 risk.append(row)
 timing.append({'PATNO':p,'sample_id':rr['sample'],'count_row':int(rr['count_row']),'collection_registry':rr['collection_months'],'native_draw_month':bm,'native_form_month':None if br is None else month(br['INFODT']),'native_collection_rows':','.join(map(str,brefs)),'collection_confirmed':collection_ok,'baseline_reference_month':ref,'RNA_minus_reference':row['reference_delta_months'],'same_month_order_resolved':False,'raw_vector_hash_pass':binding})
 exposure.append({'PATNO':p,'component_id':component,'used_in_any_579_DE':truth(rr['used_in_any_579_DE']),'used_in_primary_558_DE':truth(rr['used_in_primary_558_DE']),'used_in_528_classifier':truth(rr['used_in_528_classifier']),'previous_clinical_outcome_exposure':'possible_multiple_prior_PPMI_analyses_not_untouched'})
 for h in [12,24,36]:
  r,refs,mo,st=select(by['moca'].get(p,[]),idx,mfields,h=h);cq=moca(r,edu,st);log(p,'MoCA'+str(h),'moca',r,refs,mo,idx,st)
  if death is not None and mo is not None and mo>death:cq['reasons'].append('assessment_after_death_month')
  if death is not None and mo==death:cq['flags'].append('assessment_same_death_month')
  if death_conflict:cq['reasons'].append('conflicting_death_dates')
  if invalid_death:cq['reasons'].append('unparseable_documented_death_date')
  if mo is not None and earliest_wd is not None and mo>earliest_wd:cq['flags'].append('assessment_after_withdrawal_record_review')
  good=not cq['reasons'];endpoint={'PATNO':p,'component_id':component,'horizon_months':h,'source_row':None if r is None else r['_row'],'source_rows':','.join(map(str,refs)),'EVENT_ID':None if r is None else r['EVENT_ID'],'month':mo,'RNA_index_month':idx,'elapsed_months':None if mo is None or idx is None else mo-idx,'raw_MoCA':cq['raw'],'education_adjusted_MoCA':cq['adjusted'],'selection_status':st,'endpoint_QC_ready':good,'reasons':';'.join(cq['reasons']),'flags':';'.join(cq['flags']),'diagnosis_supported_eligible':row['diagnosis_supported_baseline_ready'] and good,'PD_core_candidate':row['PD_main_baseline_candidate'] and good,'PD_matched_candidate':row['PD_matched_baseline_candidate'] and good,'SWEDD_recruitment':row['SWEDD_recruitment'],'CBC_matched_candidate':row['PD_matched_baseline_candidate'] and good and row['CBC_nonfuture_complete']};endpoints.append(endpoint)
  for kind,name,fields,fk,maxform in [('SDMT','sdmt',sfields,'SDMTVRSN',2),('HVLT_immediate','hvlt',hfields,'HVLTVRSN',6),('HVLT_delayed','hvlt',hfields,'HVLTVRSN',6)]:
   r,refs,mo,st=select(by[name].get(p,[]),idx,fields,h=h);value=detail(r,kind);bval=detail(base_detail[name][0],kind);f1=int(float(r[fk])) if r and integer(r.get(fk),1,maxform) else None;f0=row['baseline_'+name+'_form'];dead=death is not None and mo is not None and mo>death
   details.append({'PATNO':p,'horizon_months':h,'endpoint':kind,'source_row':None if r is None else r['_row'],'source_rows':','.join(map(str,refs)),'month':mo,'elapsed_months':None if mo is None or idx is None else mo-idx,'baseline_value':bval,'value':value,'baseline_form':f0,'followup_form':f1,'same_form':f0 is not None and f0==f1,'selection_status':st,'after_death_hold':dead,'PD_detail_candidate':row['PD_main_baseline_candidate'] and value is not None and bval is not None and not dead and not death_conflict and not invalid_death,'PD_matched_detail_candidate':row['PD_matched_baseline_candidate'] and value is not None and not dead and not death_conflict and not invalid_death})
   log(p,kind+str(h),name,r,refs,mo,idx,st)
for name,rows in [('Baseline_Risk_Set',risk),('Cognitive_Endpoint_Eligibility',endpoints),('Detailed_Cognitive_Eligibility',details),('Source_Row_Ledger',ledger),('RNA_Timing_Ledger',timing),('Prior_Exposure_Ledger',exposure)]:tab('restricted/'+name+'.tsv',rows)
byrisk={r['PATNO']:r for r in risk};flow=[]
for h in [12,24,36]:
 z=[r for r in endpoints if r['horizon_months']==h];e=[r for r in z if r['PD_matched_candidate']]
 flow.append({'horizon_months':h,'RNA_frame':410,'dated_followup_selected':sum(r['source_row'] is not None for r in z),'followup_QC_pass_frame':sum(r['endpoint_QC_ready'] for r in z),'diagnosis_supported_eligible_including_SWEDD':sum(r['diagnosis_supported_eligible'] for r in z),'PD_core_candidate_excluding_SWEDD':sum(r['PD_core_candidate'] for r in z),'PD_matched_candidate_excluding_SWEDD':len(e),'CBC_matched_candidate':sum(r['CBC_matched_candidate'] for r in z),'SWEDD_diagnosis_supported_eligible':sum(r['diagnosis_supported_eligible'] and r['SWEDD_recruitment'] for r in z),'RIN_below6_matched':sum(byrisk[r['PATNO']]['RIN_below6_review'] for r in e),'strict_earlier_baseline_MoCA_matched':sum(byrisk[r['PATNO']]['baseline_moca_month']<byrisk[r['PATNO']]['RNA_index_month'] for r in e)})
sets=[{r['PATNO'] for r in endpoints if r['horizon_months']==h and r['PD_matched_candidate']} for h in [12,24,36]]
summary={'study_id':'Clinical_hallmark','status':'eligibility_audit_complete_pending_independent_verification','baseline_RNA':410,'native_PD_confirmed':sum(r['native_preindex_PD_confirmed'] for r in risk),'technical_pass':sum(r['RNA_technical_ready'] for r in risk),'native_RNA_collection_confirmed':sum(r['RNA_native_collection_confirmed'] for r in risk),'diagnosis_supported_baseline_ready_including_SWEDD':sum(r['diagnosis_supported_baseline_ready'] for r in risk),'PD_main_baseline_candidates':sum(r['PD_main_baseline_candidate'] for r in risk),'PD_matched_baseline_candidates':sum(r['PD_matched_baseline_candidate'] for r in risk),'SWEDD_RNA_frame':sum(r['SWEDD_recruitment'] for r in risk),'matched_all_three_horizons':len(set.intersection(*sets)),'horizons':flow,'baseline_holds':dict(collections.Counter(x for r in risk for x in r['baseline_hold_reasons'].split(';') if x)),'endpoint_holds_within_PD_matched_baseline':{str(h):dict(collections.Counter(x for r in endpoints if r['horizon_months']==h and byrisk[r['PATNO']]['PD_matched_baseline_candidate'] for x in r['reasons'].split(';') if x)) for h in [12,24,36]},'source_timing_limit':'month precision only; end-of-month baseline prediction; no exact-day or barcode claim','models_fit':0,'paid_queries':0}
save('results/Audit_Summary.json',summary);tab('results/Cohort_Flow.tsv',flow)
for h in [12,24,36]:tab('restricted/Matched_MoCA_'+str(h)+'m_Candidates.tsv',[dict(r,**{k:v for k,v in byrisk[r['PATNO']].items() if k not in r}) for r in endpoints if r['horizon_months']==h and r['PD_matched_candidate']])
print(json.dumps(summary,indent=2))