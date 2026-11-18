import pandas as pd, numpy as np, math
def num(v):
 try:x=float(v);return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None

def month(v):
 try:
  a,b=v.split('/');m,y=int(a),int(b)
  return y*12+m if len(a)==2 and len(b)==4 and 1<=m<=12 and 1900<=y<=2100 else None
 except:return None

def select_scans(base, ac, qt, vi, fields):
    ac=ac.copy();qt=qt.copy();vi=vi.copy()
    for d in [ac,qt,vi]:d["_month"]=d.DATSCAN_DATE.map(month)
    audit=[];values={};trace=[]
    for b in base.itertuples():
     p=b.PATNO;ix=int(b.index_month);reasons=[];flags=[]
     aa=ac[ac.PATNO.eq(p)];qq=qt[qt.PATNO.eq(p)]
     # Select from union including acquisitions lacking quantitative output; no fallback after QC.
     aa=aa[aa.EVENT_ID.isin(['SC','BL'])];qq=qq[qq.EVENT_ID.isin(['SC','BL'])]
     invalid_timed=aa._month.isna().any() or qq._month.isna().any()
     cand=pd.concat([aa[['EVENT_ID','_month']],qq[['EVENT_ID','_month']]]).drop_duplicates()
     cand=cand[cand._month.between(ix-3,ix)].copy()
     chosen_month=None;event='';protocol='';a=ac.iloc[:0];q=qt.iloc[:0];visual='';val=[None]*4
     if invalid_timed:reasons.append('invalid_SC_BL_date')
     if cand.empty:reasons.append('no_preindex_scan_in_window')
     else:
      chosen_month=int(cand._month.max());event='BL' if cand[cand._month.eq(chosen_month)].EVENT_ID.eq('BL').any() else 'SC'
      a=aa[aa._month.eq(chosen_month)&aa.EVENT_ID.eq(event)]
      q=qq[qq._month.eq(chosen_month)&qq.EVENT_ID.eq(event)]
      prots=set(a.PROTOCOL)|set(q.PROTOCOL)
      if len(prots)!=1 or '' in prots:reasons.append('ambiguous_protocol_binding')
      else:protocol=next(iter(prots))
      if a.empty:reasons.append('no_acquisition_binding')
      else:
       # Same month has no scan UID. Conflicting acquisition signatures cannot safely be collapsed.
       signature=[k for k in ac.columns if k not in ['_row','_month','PATNO']]
       if len(a[signature].drop_duplicates())>1:reasons.append('multiple_conflicting_acquisitions')
       elif len(a)>1:flags.append('identical_acquisition_duplicates')
       if not a.DATSCAN_IMAGE_ACCEPTABLE.eq('Yes').all():reasons.append('acquisition_not_explicitly_acceptable')
       if not a.DATSCAN_LIGAND.isin(['','123I-DaTscan']).all():reasons.append('acquisition_tracer_conflict')
       if a.DATSCAN_LIGAND.eq('').any():flags.append('acquisition_tracer_missing_quant_corroboration_required')
       if not a.PREVIOUSLY_ACQUIRED.isin(['','No']).all():reasons.append('acquisition_prior_scan_conflict')
       if a.PREVIOUSLY_ACQUIRED.eq('').any():flags.append('acquisition_prior_flag_missing')
       for z in a.to_dict('records'):
        inj=z['DATSCAN_DATE_INJECTED']
        if inj and month(inj)!=chosen_month:reasons.append('injection_scan_month_discordance')
        if not inj:flags.append('injection_month_missing')
        dose=z['DATSCAN_TOTAL_DOSE_CALC']
        if dose and (num(dose) is None or num(dose)<=0 or z['DATSCAN_TOTAL_DOSE_UNIT']!='MBq'):reasons.append('dose_unit_or_value_review')
        elif not dose:flags.append('dose_missing')
      if q.empty:reasons.append('no_quantitative_binding')
      else:
       sig=['PROTOCOL','DATSCAN_LIGAND','PREVIOUSLY_ACQUIRED','DATSCAN_ANALYZED','DATSCAN_NOT_ANALYZED_REASON','DATSCAN_OTHER_SPECIFY']+fields
       if len(q[sig].drop_duplicates())>1:reasons.append('conflicting_quantitative_records')
       elif len(q)>1:flags.append('identical_quantitative_duplicates')
       if not q.DATSCAN_ANALYZED.eq('Yes').all():reasons.append('not_analyzed')
       if not q.DATSCAN_LIGAND.eq('123I-DaTscan').all():reasons.append('incompatible_tracer')
       if not q.PREVIOUSLY_ACQUIRED.eq('No').all():reasons.append('prior_acquisition_or_unknown')
       if not q.DATSCAN_NOT_ANALYZED_REASON.eq('').all() or not q.DATSCAN_OTHER_SPECIFY.eq('').all():reasons.append('quantitative_reason_or_comment_review')
       val=[num(q.iloc[0][k]) for k in fields]
       if any(v is None or v < -1 for v in val):reasons.append('missing_or_physically_invalid_SBR')
       if any(v is not None and v<0 for v in val):flags.append('negative_SBR_retained')
      if chosen_month==ix:flags.append('same_month_order_unknown')
      v=vi[vi.PATNO.eq(p)&vi._month.eq(chosen_month)&vi.PROTOCOL.eq(protocol)]
      visual=';'.join(sorted(set(v.DATSCAN_VISINTRP))) if len(v) else 'not_released'
      if len(set(v.DATSCAN_VISINTRP))>1:flags.append('visual_classification_discordance_not_QC_failure')
     row=dict(PATNO=p,component=b.component,index_month=ix,scan_month=chosen_month,
      scan_minus_index=None if chosen_month is None else chosen_month-ix,event=event,protocol=protocol,
      eligible=not reasons,strict_preindex=not reasons and chosen_month<ix,
      reasons=';'.join(sorted(set(reasons))),flags=';'.join(sorted(set(flags))),visual_classification=visual,
      acquisition_rows=';'.join(map(str,a._row)),quantitative_rows=';'.join(map(str,q._row)))
     audit.append(row)
     if not reasons:values[b.component]=dict(zip(fields,val))
     trace.append(dict(component=b.component,acquisitions=a.drop(columns='_month').to_dict('records'),quantitative=q.drop(columns='_month').to_dict('records')))
    return pd.DataFrame(audit),values,trace
