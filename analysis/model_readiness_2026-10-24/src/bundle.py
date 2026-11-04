"""Read-only bindings and preflight. No estimator imports or preprocessing fits."""
from pathlib import Path
import json,csv,hashlib,copy,math
import numpy as np
D=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def table(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f,delimiter='\t'))
def verify_seal(package=D):
 lock=read(package/'PLAN_LOCK.json')
 for x in lock['files']:
  p=package/x['path']
  if not p.is_file() or sha(p)!=x['sha256']:raise ValueError('Locked package integrity failure: '+x['path'])
 return sha(package/'PLAN_LOCK.json')
def load_bundle(package=D,require_seal=True):
 package=Path(package)
 if require_seal:verify_seal(package)
 for x in read(package/'evidence/Input_Manifest.json'):
  if sha(x['path'])!=x['sha256']:raise ValueError('Source integrity failure: '+x['name'])
 plan=read(package/'config/Analysis_Contract.json');views=read(package/'config/Views.json');panel=read(package/'config/RNA_Panel.json');inputs=read(package/'config/Inputs.json')
 feat=table(package/'restricted/Clinical_Features.tsv');participants=table(package/'restricted/Participant_Manifest.tsv');splits=table(package/'restricted/Grouped_Splits.tsv');binding=table(package/'restricted/View_Row_Binding.tsv')
 if len(feat)!=351 or len({r['component_id'] for r in feat})!=351 or len({r['PATNO'] for r in feat})!=351:raise ValueError('Invalid baseline feature identity')
 features={r['component_id']:r for r in feat};sp={r['component_id']:{k:int(r[k]) for k in ['outer_fold']+['inner_for_outer_'+str(i) for i in range(5)]} for r in splits};pm={r['component_id']:r for r in participants}
 if set(features)!=set(sp) or set(pm)!=set(sp):raise ValueError('Split/feature/component mismatch')
 for k,s in sp.items():
  if s['outer_fold'] not in range(5):raise ValueError('Invalid outer label')
  for o in range(5):
   expected=[-1] if s['outer_fold']==o else [0,1,2]
   if s['inner_for_outer_'+str(o)] not in expected:raise ValueError('Inner/outer split contamination')
  if features[k]['sample_id']!=pm[k]['sample_id'] or features[k]['count_row']!=pm[k]['count_row']:raise ValueError('Feature RNA binding mismatch')
 source_out={(r['PATNO'],int(r['horizon_months'])):r for r in table(inputs['outcomes']) if r['PD_matched_candidate']=='True'}
 reg={r['sample']:r for r in table(inputs['rna_registry'])};qc={r['sample']:r for r in table(inputs['rna_qc'])}
 matrix=np.load(inputs['rna_matrix'],mmap_mode='r');ann=table(inputs['rna_gene_annotation'])
 if list(matrix.shape)!=panel['raw_shape'] or str(matrix.dtype)!='int64' or len(ann)!=58780:raise ValueError('RNA matrix shape/type/annotation mismatch')
 if any(ann[i]['Geneid']!=g for i,g in zip(panel['raw_columns'],panel['exact_gene_ids'])):raise ValueError('Versioned RNA columns mismatch')
 for k,f in features.items():
  r=reg[f['sample_id']];q=qc[f['sample_id']];v=matrix[int(f['count_row'])]
  if r['PATNO']!=f['PATNO'] or r['count_row']!=f['count_row'] or r['visit']!='BL' or np.any(v<0) or int(v.sum())<=0 or hashlib.sha256(v.tobytes()).hexdigest()!=q['count_vector_sha256']:raise ValueError('RNA row/content binding failure')
 for r in binding:
  for k in ['count_row','horizon_months','outcome_source_row','elapsed_months']:r[k]=int(r[k])
  for k in ['y','baseline_moca']:r[k]=float(r[k])
  f=features[r['component_id']];s=source_out.get((r['PATNO'],r['horizon_months']))
  if s is None or f['PATNO']!=r['PATNO'] or f['sample_id']!=r['sample_id'] or int(f['count_row'])!=r['count_row'] or int(s['source_row'])!=r['outcome_source_row'] or float(s['education_adjusted_MoCA'])!=r['y'] or float(f['baseline_moca'])!=r['baseline_moca']:raise ValueError('Outcome/feature/person binding failure')
  if not math.isfinite(r['y']) or not 0<=r['y']<=30 or abs(r['elapsed_months']-r['horizon_months'])>3:raise ValueError('Outcome range or timing invalid')
 for v in views:
  z=[r for r in binding if r['view_id']==v['view_id']]
  if len(z)!=v['expected_n'] or len({r['component_id'] for r in z})!=len(z):raise ValueError('View size/uniqueness failure')
  for o in range(5):
   tr=[r for r in z if sp[r['component_id']]['outer_fold']!=o];te=[r for r in z if sp[r['component_id']]['outer_fold']==o]
   if not tr or not te:raise ValueError('Empty outer partition')
   for i in range(3):
    iv=[r for r in tr if sp[r['component_id']]['inner_for_outer_'+str(o)]==i]
    if not iv or len(iv)==len(tr):raise ValueError('Empty inner partition')
 specs={}
 for v in views:
  s=copy.deepcopy(plan['comparators'][v['clinical']])
  if v['technical_covariates']:s['numeric']+=plan['technical_sensitivity']['numeric_additions'];s['categories'].update(plan['technical_sensitivity']['categorical_additions'])
  specs[v['view_id']]=s
 return {'package':str(package),'plan':plan,'views':views,'panel':panel,'inputs':inputs,'features':features,'participants':pm,'splits':sp,'binding':binding,'clinical_specs':specs}
def authorize(package,authorization,selected,output):
 package=Path(package);p=Path(output).resolve();parent=package.parent.resolve()/'executions'
 if not authorization:raise PermissionError('No execution authorization; current stage permits validation only')
 auth=read(authorization);sealhash=verify_seal(package)
 if auth.get('approved_by_user') is not True or auth.get('action')!='TRAIN_CLINICAL_HALLMARK' or auth.get('plan_lock_sha256')!=sealhash:raise PermissionError('Missing or mismatched explicit execution authorization')
 if not selected or len(set(selected))!=len(selected) or set(auth.get('view_ids',[]))!=set(selected):raise PermissionError('Authorization view selection mismatch')
 if p.parent!=parent or p.exists():raise PermissionError('Output must be a new direct child of Clinical_hallmark/executions')
 return p