"""Shared integrity, provenance and single-writer checkpoint support."""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,sqlite3,fcntl,time,math,zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
FROZEN=STUDY/'ga_readiness_2026-10-30'
sys.path.insert(0,str(ROOT/'src'))
import r2_engine as frozen
class IntegrityError(RuntimeError):pass
class HardwareFailure(RuntimeError):pass
class Paused(RuntimeError):pass
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def decode_fitness(v):
    return json.loads(zlib.decompress(v) if isinstance(v,bytes) else v)
def digest(x):return hashlib.sha256(encoded(x)).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:f.write(encoded(obj)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def source_identity():
    sources=json.loads((ROOT/'evidence/Source_Hashes.json').read_text())
    for p,h in sources.items():
        if file_sha(p)!=h:raise IntegrityError('Frozen source changed')
    return digest({'sources':sources,'implementation':{p.name:file_sha(p) for p in sorted((ROOT/'src').glob('*.py'))}})
def validate_spec(spec,c):
    keys={'context_id','arm','horizon','outer','train_rows','train_y','inner','test_rows','synthetic'}
    if set(spec)!=keys:raise IntegrityError('Unexpected context fields; held-out outcomes forbidden')
    if spec['context_id']!=f"{spec['arm']}_{spec['horizon']}m_outer{spec['outer']}":raise IntegrityError('Context identity mismatch')
    if spec['arm']!=c['active_arm']:raise IntegrityError('Search arm mismatch')
    tr,te=spec['train_rows'],spec['test_rows'];y=np.asarray(spec['train_y'],dtype=float);inner=np.asarray(spec['inner'])
    if np.var(y)<=0:raise IntegrityError('Undefined pooled training R2')
    if len(tr)!=len(y) or len(tr)!=len(inner) or not te or set(inner)!={0,1,2} or not np.isfinite(y).all():raise IntegrityError('Invalid training data or folds')
    names,cats=frozen.layout(c);allowed=set(names)|set(cats)|{'PATNO','component_id'}
    for rows in (tr,te):
        if any(set(row)!=allowed for row in rows):raise IntegrityError('Unexpected row fields or missing predictor')
        ids=[str(r['PATNO']) for r in rows];groups=[str(r['component_id']) for r in rows]
        if len(set(ids))!=len(rows) or len(set(groups))!=len(rows):raise IntegrityError('Nonunique identity')
        for row in rows:
            if spec['synthetic'] and not (str(row['PATNO']).startswith('SYNTH_') and str(row['component_id']).startswith('SYNTH_')):raise IntegrityError('Non-synthetic record in test mode')
            for n in names:frozen.numeric(row[n])
            if any(row[k] not in levels for k,levels in cats.items()):raise IntegrityError('Invalid category')
    if {r['PATNO'] for r in tr}&{r['PATNO'] for r in te} or {r['component_id'] for r in tr}&{r['component_id'] for r in te}:raise IntegrityError('Outer identity leakage')
    return True
def validate_profile(p,c):
    if p.get('fitness_schema')!='joint28_r2_v1':raise IntegrityError('Wrong objective profile')
    if p.get('stagnation') and p['stagnation'].get('units')!='pooled_inner_R2_gain':raise IntegrityError('Wrong stagnation units')
    d=len(c['features']['optional']);mandatory=len(frozen.mandatory(c))
    if not (0<d<=28 and p['population']>=mandatory and p['offspring']>0 and p['generations']>0):raise IntegrityError('Invalid search profile')
    if not p['seeds'] or len(set(p['seeds']))!=len(p['seeds']):raise IntegrityError('Invalid restart seeds')
    if p['unique_limit'] is not None and not mandatory<=p['unique_limit']<=2**d:raise IntegrityError('Invalid unique-mask budget')
    if p['exhaustive']:raise IntegrityError('Enumeration is separate oracle QA, not a launcher execution mode')
    if p['batch_masks']<=0 or p['active_guard_seconds']<=0:raise IntegrityError('Invalid resource limits')
    st=p.get('stagnation')
    if st:
        windows=st.get('consecutive_windows',1)
        if windows<1 or st['minimum_generation']<windows*st['window'] or st.get('minimum_unique',0)<0:raise IntegrityError('Invalid multiwindow convergence rule')
    gate=p.get('convergence')
    if gate:
        if gate['main_restarts']+gate['confirmation_restarts']!=len(p['seeds']) or not 1<=gate['minimum_replicates']<=gate['confirmation_restarts'] or gate['tolerance']<0:raise IntegrityError('Invalid confirmation schedule')
    if st and (st['window']<=0 or st['minimum_generation']<st['window'] or st['minimum_progress']<0):raise IntegrityError('Invalid stagnation rule')
class Store:
    def __init__(self,path,binding):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.path.chmod(0o700)
        if self.path.is_symlink():raise IntegrityError('Symlink context output')
        self.lock=(self.path/'writer.lock').open('a+')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.lock.close();raise IntegrityError('Context already has a writer')
        self.db=sqlite3.connect(self.path/'checkpoint.sqlite',timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.executescript('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT,h TEXT);CREATE TABLE IF NOT EXISTS fitness(mask INTEGER PRIMARY KEY,v TEXT,h TEXT);CREATE TABLE IF NOT EXISTS visits(algorithm TEXT,restart INTEGER,mask INTEGER,PRIMARY KEY(algorithm,restart,mask));CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,kind TEXT,planned INTEGER,status TEXT,attempted INTEGER,completed INTEGER,detail TEXT);CREATE TABLE IF NOT EXISTS refits(arm TEXT PRIMARY KEY,v TEXT,h TEXT);')
        self.db.executescript('CREATE TABLE IF NOT EXISTS visit_union(algorithm TEXT,mask INTEGER,PRIMARY KEY(algorithm,mask));CREATE INDEX IF NOT EXISTS visits_by_mask ON visits(algorithm,mask);CREATE INDEX IF NOT EXISTS union_by_mask ON visit_union(mask);')
        old=self.get('binding')
        if old is None:self.set('binding',binding)
        elif old!=binding:self.close();raise IntegrityError('Checkpoint binding mismatch')
        self.db.execute("UPDATE events SET status='INTERRUPTED_UNKNOWN' WHERE status='RUNNING'");self.db.commit()
        self._visit_counts={(a,r):n for a,r,n in self.db.execute('SELECT algorithm,restart,COUNT(*) FROM visits GROUP BY algorithm,restart')}
        self._visit_counts.update({(a,None):n for a,n in self.db.execute('SELECT algorithm,COUNT(*) FROM visit_union GROUP BY algorithm')})
        chain=hashlib.sha256(b'').hexdigest()
        for algorithm,restart,mask in self.db.execute('SELECT algorithm,restart,mask FROM visits ORDER BY rowid'):
            chain=hashlib.sha256(chain.encode()+encoded([algorithm,restart,mask])).hexdigest()
        if chain!=self.get('visits_digest',hashlib.sha256(b'').hexdigest()):
            self.close();raise IntegrityError('Visited-mask checkpoint checksum mismatch')
        self.visit_digest=chain
        self.union_digests={}
        for algorithm,mask in self.db.execute('SELECT algorithm,mask FROM visit_union ORDER BY rowid'):
            prior=self.union_digests.get(algorithm,hashlib.sha256(b'').hexdigest())
            self.union_digests[algorithm]=hashlib.sha256(prior.encode()+encoded(mask)).hexdigest()
        if self.union_digests!=self.get('union_digests',{}):
            self.close();raise IntegrityError('Visited union checksum mismatch')
        missing=self.db.execute('SELECT 1 FROM visits v LEFT JOIN visit_union u ON v.algorithm=u.algorithm AND v.mask=u.mask WHERE u.mask IS NULL LIMIT 1').fetchone()
        extra=self.db.execute('SELECT 1 FROM visit_union u WHERE NOT EXISTS(SELECT 1 FROM visits v WHERE v.algorithm=u.algorithm AND v.mask=u.mask) LIMIT 1').fetchone()
        if missing or extra:self.close();raise IntegrityError('Visited union coverage mismatch')
        self.ledger()
    def get(self,k,default=None):
        row=self.db.execute('SELECT v,h FROM meta WHERE k=?',(k,)).fetchone()
        if row is None:return default
        v=json.loads(row[0])
        if digest(v)!=row[1]:raise IntegrityError('Checkpoint state hash mismatch')
        return v
    def set(self,k,v,commit=True):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?,?)',(k,encoded(v).decode(),digest(v)))
        if commit:self.db.commit()
    def records(self):
        result={}
        for mask,text,h in self.db.execute('SELECT mask,v,h FROM fitness ORDER BY mask'):
            r=decode_fitness(text)
            if digest(r)!=h or r['mask']!=mask:raise IntegrityError('Fitness checksum mismatch')
            result[mask]=r
        return result
    def event(self,kind,planned):
        cur=self.db.execute("INSERT INTO events(kind,planned,status) VALUES(?,?,'RUNNING')",(kind,planned));self.db.commit()
        return cur.lastrowid
    def finish_event(self,event,status,attempted,completed,detail='',records=None):
        with self.db:
            if records:
                for r in records:self.db.execute('INSERT INTO fitness VALUES(?,?,?)',(r['mask'],zlib.compress(encoded(r),1),digest(r)))
            self.db.execute('UPDATE events SET status=?,attempted=?,completed=?,detail=? WHERE id=?',(status,attempted,completed,detail,event))
    def checkpoint(self,state,algorithm=None,restart=0,masks=()):
        with self.db:
            if algorithm:
                for m in masks:
                    row=[algorithm,restart,int(m)]
                    cur=self.db.execute('INSERT OR IGNORE INTO visits VALUES(?,?,?)',row)
                    if cur.rowcount:
                        self.visit_digest=hashlib.sha256(self.visit_digest.encode()+encoded(row)).hexdigest()
                        key=(algorithm,restart);self._visit_counts[key]=self._visit_counts.get(key,0)+1
                        if self.db.execute('INSERT OR IGNORE INTO visit_union VALUES(?,?)',(algorithm,int(m))).rowcount:
                            key=(algorithm,None);self._visit_counts[key]=self._visit_counts.get(key,0)+1
                            prior=self.union_digests.get(algorithm,hashlib.sha256(b'').hexdigest())
                            self.union_digests[algorithm]=hashlib.sha256(prior.encode()+encoded(int(m))).hexdigest()
                self.set('visits_digest',self.visit_digest,False)
                self.set('union_digests',self.union_digests,False)
            self.set('state',state,False)
    def visited(self,algorithm,restart=None):
        return VisitSet(self,algorithm,restart)
    def refit(self,arm):
        row=self.db.execute('SELECT v,h FROM refits WHERE arm=?',(arm,)).fetchone()
        if row is None:return None
        v=json.loads(row[0])
        if digest(v)!=row[1]:raise IntegrityError('Refit checksum mismatch')
        return v
    def put_refit(self,arm,v,event):
        with self.db:
            self.db.execute('INSERT INTO refits VALUES(?,?,?)',(arm,encoded(v).decode(),digest(v)))
            self.db.execute("UPDATE events SET status='COMPLETE',attempted=1,completed=1 WHERE id=?",(event,))
    def ledger(self):
        rows=[dict(zip(('id','kind','planned','status','attempted','completed','detail'),r)) for r in self.db.execute('SELECT * FROM events ORDER BY id')]
        valid_statuses={'RUNNING','COMPLETE','FAILED','FAILED_HARDWARE','INTERRUPTED_UNKNOWN'}
        for r in rows:
            if r['status'] not in valid_statuses or r['planned']<=0:raise IntegrityError('Invalid event ledger')
            if r['status'] not in ('RUNNING','INTERRUPTED_UNKNOWN'):
                if not 0<=r['completed']<=r['attempted']<=r['planned']:raise IntegrityError('Invalid physical fit counts')
                if r['status']=='COMPLETE' and r['completed']!=r['planned']:raise IntegrityError('Incomplete successful event')
        inner=sum(r['completed'] for r in rows if r['status']=='COMPLETE' and r['kind'].endswith('_fitness'))
        outer=sum(r['completed'] for r in rows if r['status']=='COMPLETE' and r['kind']=='outer_refit')
        if inner!=15*self.db.execute('SELECT COUNT(*) FROM fitness').fetchone()[0] or outer!=self.db.execute('SELECT COUNT(*) FROM refits').fetchone()[0]:raise IntegrityError('Committed fit ledger disagrees with cached coverage')
        uncertain=sum(r['planned'] for r in rows if r['status']=='INTERRUPTED_UNKNOWN')
        attempts=sum(r['attempted'] or 0 for r in rows)
        return {'unique_masks':self.db.execute('SELECT COUNT(*) FROM fitness').fetchone()[0],'refits':self.db.execute('SELECT COUNT(*) FROM refits').fetchone()[0],'known_physical_attempts':attempts,'known_validated_solves':sum(r['completed'] or 0 for r in rows),'unknown_interrupted_attempts_upper':uncertain,'physical_attempts_lower':attempts,'physical_attempts_upper':attempts+uncertain,'events':rows}
    def close(self):
        if hasattr(self,'db'):self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()

class VisitSet:
    def __init__(self,store,algorithm,restart=None):self.store=store;self.algorithm=algorithm;self.restart=restart
    def __len__(self):return self.store._visit_counts.get((self.algorithm,self.restart),0)
    def __contains__(self,m):
        if self.restart is None:
            return self.store.db.execute('SELECT 1 FROM visit_union WHERE algorithm=? AND mask=?',(self.algorithm,int(m))).fetchone() is not None
        return self.store.db.execute('SELECT 1 FROM visits WHERE algorithm=? AND restart=? AND mask=?',(self.algorithm,self.restart,int(m))).fetchone() is not None
    def __iter__(self):
        if self.restart is None:
            rows=self.store.db.execute('SELECT mask FROM visit_union WHERE algorithm=? ORDER BY mask',(self.algorithm,))
        else:rows=self.store.db.execute('SELECT mask FROM visits WHERE algorithm=? AND restart=? ORDER BY mask',(self.algorithm,self.restart))
        for (m,) in rows:yield m

class CompactCache:
    # Memory usage is bounded independently of the number of evaluated masks.
    def __init__(self,store,validator,limit=4096):
        from collections import OrderedDict
        self.store=store;self.lru=OrderedDict();self.limit=limit;self.meta=CacheMeta(self)
        for mask,text,h in store.db.execute('SELECT mask,v,h FROM fitness ORDER BY mask'):
            r=decode_fitness(text)
            if digest(r)!=h or r['mask']!=mask:raise IntegrityError('Fitness checksum mismatch')
            validator(r);self[mask]=r
    def __len__(self):return self.store.db.execute('SELECT COUNT(*) FROM fitness').fetchone()[0]
    def __iter__(self):
        for (m,) in self.store.db.execute('SELECT mask FROM fitness ORDER BY mask'):yield m
    def __contains__(self,m):
        return m in self.lru or self.store.db.execute('SELECT 1 FROM fitness WHERE mask=?',(int(m),)).fetchone() is not None
    def __setitem__(self,m,r):
        self.lru[m]=r;self.lru.move_to_end(m)
        if len(self.lru)>self.limit:self.lru.popitem(last=False)
    def __getitem__(self,m):
        if m not in self.lru:
            row=self.store.db.execute('SELECT v,h FROM fitness WHERE mask=?',(int(m),)).fetchone()
            if row is None:raise IntegrityError('Missing fitness record')
            r=decode_fitness(row[0])
            if digest(r)!=row[1]:raise IntegrityError('Fitness checksum mismatch')
            self[m]=r
        self.lru.move_to_end(m);return self.lru[m]
    def summary(self,m):
        r=self[m]
        return {k:r[k] for k in ['mask','mse','burden','scores','alpha','normalized_mse','r2']}
class CacheMeta:
    def __init__(self,cache):self.cache=cache
    def __len__(self):return len(self.cache)
    def __getitem__(self,m):
        r=self.cache[m];return tuple(r[k] for k in ['mse','burden','scores','alpha','normalized_mse','r2'])
    def items(self):
        for m in self.cache:yield m,self[m]
