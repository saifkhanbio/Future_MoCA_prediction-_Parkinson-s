from pathlib import Path
import json,hashlib,math,os
import pandas as pd
os.umask(0o077)
D=Path(__file__).resolve().parents[1];P=D.parent
read=lambda p:json.loads(Path(p).read_text());I=read(D/'config/Inputs.json');plan=read(D/'config/Analysis_Contract.json');views=read(D/'config/Views.json')
assert not (D/'PLAN_LOCK.json').exists()
def save(n,o):(D/n).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def table(n,z):z.to_csv(D/n,sep='\t',index=False)
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def load(k):return pd.read_csv(I[k],sep='\t',dtype=str,keep_default_na=False)
risk=load('baseline_risk');risk=risk[risk.PD_matched_baseline_candidate.eq('True')].copy();assert len(risk)==351 and risk.component_id.is_unique and risk.PATNO.is_unique
splitinput=risk[['component_id','phase','recruitment_group']].sort_values('component_id').reset_index(drop=True);table('restricted/Split_Input_Metadata.tsv',splitinput)
def allocate(data,n,salt):
 result={};total=[0]*n
 for stratum,g in data.groupby(['phase','recruitment_group'],sort=True):
  local=[0]*n
  ids=sorted(g.component_id,key=lambda k:digest(salt+'|order|'+k))
  for k in ids:
   f=min(range(n),key=lambda x:(local[x],total[x],digest(salt+'|fold|'+k+'|'+str(x))))
   result[k]=f;local[f]+=1;total[f]+=1
 return result
seed=plan['splits']['seed_label'];outer=allocate(splitinput,5,seed+'|outer');spl=splitinput.copy();spl['outer_fold']=spl.component_id.map(outer)
for o in range(5):
 m=allocate(splitinput[splitinput.component_id.map(outer).ne(o)],3,seed+'|inner|'+str(o));spl['inner_for_outer_'+str(o)]=spl.component_id.map(m).fillna(-1).astype(int)
table('restricted/Grouped_Splits.tsv',spl)
features=load('clinical_features').set_index('PATNO').loc[risk.PATNO].reset_index();assert features.PATNO.is_unique
for k in ['baseline_sdmt_form','baseline_hvlt_form']:
 features[k]=features[k].map(lambda x:str(int(float(x))) if str(x).strip() and float(x).is_integer() else 'unknown')
features['motor_state']=features.motor_state.where(features.motor_state.isin(plan['comparators']['core']['categories']['motor_state']),'unknown')
ri=risk.set_index('PATNO');features['RIN']=features.PATNO.map(ri.RIN);features['log10_library_total']=features.PATNO.map(ri.library_total).astype(float).map(math.log10);features['RNA_phase']=features.PATNO.map(ri.phase)
table('restricted/Clinical_Features.tsv',features.sort_values('PATNO'))
table('restricted/Participant_Manifest.tsv',risk[['PATNO','component_id','sample_id','count_row','RNA_index_month','recruitment_group','phase','RIN','RIN_below6_review','technical_review_flag']].sort_values('component_id'))
out=load('outcomes');out=out[out.PD_matched_candidate.eq('True')].copy();out['horizon_months']=out.horizon_months.astype(int);allsets=[set(out.loc[out.horizon_months.eq(h),'PATNO']) for h in [12,24,36]];common=set.intersection(*allsets);assert len(common)==280
binding=[];split_summary=[]
for v in views:
 z=out[out.horizon_months.eq(v['horizon'])].copy()
 if v['population']=='common':z=z[z.PATNO.isin(common)]
 elif v['population']=='RINge6':z=z[z.PATNO.map(ri.RIN).astype(float).ge(6)]
 elif v['population']=='PDrecruitment':z=z[z.PATNO.map(ri.recruitment_group).eq('PD')]
 assert len(z)==v['expected_n'] and z.PATNO.is_unique
 for r in z.sort_values('component_id').to_dict('records'):
  f=features.set_index('PATNO').loc[r['PATNO']];binding.append({'view_id':v['view_id'],'PATNO':r['PATNO'],'component_id':r['component_id'],'sample_id':f.sample_id,'count_row':int(f.count_row),'horizon_months':v['horizon'],'outcome_source_row':int(r['source_row']),'elapsed_months':int(r['elapsed_months']),'y':float(r['education_adjusted_MoCA']),'baseline_moca':float(f.baseline_moca)})
 ss=spl[spl.component_id.isin(z.component_id)]
 for o in range(5):
  te=ss[ss.outer_fold.eq(o)];tr=ss[ss.outer_fold.ne(o)];vals=[int(tr['inner_for_outer_'+str(o)].eq(i).sum()) for i in range(3)]
  assert len(te)>0 and len(tr)>0 and min(vals)>0 and max(vals)<len(tr)
  split_summary.append({'view_id':v['view_id'],'outer_fold':o,'train':len(tr),'test':len(te),'inner_validation_counts':vals,'inner_min_training':len(tr)-max(vals)})
bd=pd.DataFrame(binding);table('restricted/View_Row_Binding.tsv',bd);save('results_stub.json',{'note':'No results generated. This marker records zero fits only.','real_ridge_fits':0,'synthetic_ridge_fits':0})
ann=load('rna_gene_annotation');gmap={g:i for i,g in enumerate(ann.Geneid)};genes=read(I['genes5']);members=read(I['pathway_members']);needed=sorted({r['gene_id'] for r in members}|{g['gene_id'] for g in genes},key=lambda g:gmap[g]);local={g:i for i,g in enumerate(needed)}
panel={'exact_gene_ids':needed,'raw_columns':[gmap[g] for g in needed],'gene5_indices':[local[g['gene_id']] for g in genes],'gene5_names':[g['reference_name'] for g in genes],'pathway_names':plan['RNA']['pathways'],'pathway_indices':[[local[r['gene_id']] for r in members if n in r['selected_pathways']] for n in plan['RNA']['pathways']],'NPIPB8_index':local['ENSG00000255524.7'],'raw_total_genes':58780,'raw_shape':[730,58780]}
save('config/RNA_Panel.json',panel);save('qa/Split_Design_Summary.json',split_summary)
# Expand every planned fit without touching expression or fitting preprocessing.
sp=spl.set_index('component_id');jobs=[]
for v in views:
 components=sorted(bd.loc[bd.view_id.eq(v['view_id']),'component_id'])
 for o in range(5):
  train=[k for k in components if sp.loc[k,'outer_fold']!=o];test=[k for k in components if sp.loc[k,'outer_fold']==o]
  for i in range(3):
   tr=[k for k in train if sp.loc[k,'inner_for_outer_'+str(o)]!=i];va=[k for k in train if sp.loc[k,'inner_for_outer_'+str(o)]==i]
   specs=[('none',c,None) for c in plan['model']['clinical_lambdas']]+[(a,c,b) for a in v['arms'] if a!='none' for c in plan['model']['clinical_lambdas'] for b in plan['model']['RNA_lambdas']]
   for a,c,b in specs:jobs.append({'job_id':f'{v["view_id"]}|o{o}|i{i}|{a}|c{c}|r{b}','view_id':v['view_id'],'stage':'inner','outer_fold':o,'inner_fold':i,'arm':a,'lambda_clinical':c,'lambda_RNA':b,'train_n':len(tr),'target_n':len(va),'train_components_sha256':digest('\n'.join(tr)),'target_components_sha256':digest('\n'.join(va))})
  for a in v['arms']:jobs.append({'job_id':f'{v["view_id"]}|o{o}|refit|{a}','view_id':v['view_id'],'stage':'outer_refit','outer_fold':o,'inner_fold':-1,'arm':a,'lambda_clinical':None,'lambda_RNA':None,'train_n':len(train),'target_n':len(test),'train_components_sha256':digest('\n'.join(train)),'target_components_sha256':digest('\n'.join(test))})
job=pd.DataFrame(jobs);assert len(job)==8665 and job.job_id.is_unique;table('restricted/Planned_Fit_Schedule.tsv',job)
save('qa/Preparation_Summary.json',{'baseline_components':len(spl),'outer_counts':{str(k):int(n) for k,n in spl.outer_fold.value_counts().sort_index().items()},'views':len(views),'participant_view_rows':len(bd),'selected_raw_gene_union':len(needed),'planned_fits':len(job),'planned_inner':int(job.stage.eq('inner').sum()),'planned_outer_refits':int(job.stage.eq('outer_refit').sum()),'smallest_inner_training_partition':min(x['inner_min_training'] for x in split_summary),'models_fit':0,'preprocessing_fitted':False})
print((D/'qa/Preparation_Summary.json').read_text())