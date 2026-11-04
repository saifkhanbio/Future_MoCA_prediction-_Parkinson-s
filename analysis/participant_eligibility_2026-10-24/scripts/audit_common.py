from pathlib import Path
import json,csv,math,re,hashlib,os,collections
os.umask(0o077)
D=Path(__file__).resolve().parents[1];P=D.parent
I=json.loads((D/'config/Inputs.json').read_text())
C=json.loads(Path(I['prior_cognitive_contract']).read_text());ITEMS=C['cognition']['item_max']
def number(v):
 try:
  n=float(v);return n if math.isfinite(n) else None
 except (ValueError,TypeError):return None
def integer(v,lo,hi):
 n=number(v);return n is not None and n.is_integer() and lo<=n<=hi
def month(v):
 if not isinstance(v,str) or not re.fullmatch(r'\d{2}/\d{4}',v):return None
 m,y=map(int,v.split('/'));return y*12+m if 1<=m<=12 and 1900<=y<=2100 else None
def truth(v):return str(v).lower()=='true'
def save(n,obj):(D/n).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def tab(n,rows):
 if not rows:return
 with (D/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows({k:('' if v is None else v) for k,v in r.items()} for r in rows)
def read(name,keep=None):
 p=Path(I[name]);out=[]
 with p.open(encoding='utf-8-sig',newline='') as f:
  for i,r in enumerate(csv.DictReader(f,delimiter='\t' if p.suffix=='.tsv' else ','),2):
   if keep is not None:r={k:r.get(k,'') for k in keep}
   r['_row']=i;out.append(r)
 return out
def group(rows,key='PATNO'):
 d=collections.defaultdict(list)
 for r in rows:d[r[key]].append(r)
 return d
def canon(v):
 n=number(v);return n if n is not None else str(v or '').strip()
def select(rows,idx,fields,h=None,motor=False,education=False):
 if idx is None:return None,[],None,'no_index'
 cand=[]
 for r in rows:
  dt=r.get('EXAMDT','') if motor and r.get('EXAMDT','').strip() else r.get('INFODT','')
  m=month(dt)
  if m is None:continue
  if h is None:
   if r.get('EVENT_ID') not in ['SC','BL'] or m>idx or (not education and m<idx-3):continue
   key=(idx-m,0 if r['EVENT_ID']=='BL' else 1,{'OFF':0,'ON':1}.get(r.get('PDSTATE'),2) if motor else 0)
  else:
   if not idx+h-3<=m<=idx+h+3:continue
   key=(abs(m-idx-h),m,r.get('EVENT_ID',''))
  cand.append((key,r['_row'],r,m))
 if not cand:return None,[],None,'no_timed_record'
 cand.sort(key=lambda x:(x[0],x[1]));key,_,r,m=cand[0];peers=[x[2] for x in cand if x[0]==key];refs=[x['_row'] for x in peers]
 if len({tuple(canon(x.get(k)) for k in fields) for x in peers})>1:return None,refs,m,'conflicting_selected_records'
 return r,refs,m,'selected_identical_duplicates' if len(peers)>1 else 'selected'
def unique(rows,fields):
 refs=[r['_row'] for r in rows]
 if not rows:return None,refs,'missing_record'
 if len({tuple(canon(r.get(k)) for k in fields) for r in rows})!=1:return None,refs,'conflicting_records'
 return rows[0],refs,'consistent'
def moca(r,edu,selection):
 reasons=[];flags=[];raw=None;adj=None
 if r is None:return {'raw':None,'adjusted':None,'reasons':[selection],'flags':[]}
 if not all(integer(r.get(k),0,hi) for k,hi in ITEMS.items()):reasons.append('missing_or_invalid_items')
 else:
  raw=sum(int(float(r[k])) for k in ITEMS)
  if not integer(edu,0,40) and raw!=30:reasons.append('missing_or_invalid_education')
  else:adj=min(30,raw+int(raw<30 and integer(edu,0,40) and float(edu)<=12))
  t=r.get('MCATOT','')
  if not t.strip():flags.append('reported_total_missing_reconstructed')
  elif not integer(t,0,30) or float(t) not in [raw,adj]:reasons.append('reported_total_discordance')
  else:flags.append('reported_raw' if float(t)==raw else 'reported_education_adjusted')
 w=r.get('MCAVFNUM','')
 if integer(w,0,10000):
  if integer(r.get('MCAVF'),0,1) and int(float(r['MCAVF']))!=int(float(w)>=11):reasons.append('fluency_words_score_discordance')
 elif not w.strip():flags.append('fluency_wordcount_missing')
 else:reasons.append('invalid_fluency_wordcount')
 if raw==30:flags.append('raw_ceiling')
 return {'raw':raw,'adjusted':adj,'reasons':reasons,'flags':flags}
def detail(r,kind):
 if r is None:return None
 keys=['SDMTOTAL'] if kind=='SDMT' else ['HVLTRT1','HVLTRT2','HVLTRT3'] if kind=='HVLT_immediate' else ['HVLTRDLY']
 hi=110 if kind=='SDMT' else 12
 return sum(float(r[k]) for k in keys) if all(integer(r.get(k),0,hi) for k in keys) else None
def itemscore(r,keys,total=None):
 if r is None or not all(integer(r.get(k),0,4) for k in keys):return None
 s=sum(int(float(r[k])) for k in keys)
 if total and r.get(total,'').strip() and (not integer(r[total],0,4*len(keys)) or float(r[total])!=s):return None
 return s
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()