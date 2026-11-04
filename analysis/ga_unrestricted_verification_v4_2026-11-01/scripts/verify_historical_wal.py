from common import *
import shutil,concurrent.futures as cf,time
B=Path(read(E/'LATEST_RESUME.json')['process_record']).parents[1]
snapshot=B/'restricted/Interrupted_Run_Snapshot'
original=read(B/'evidence/Pre_Resume_Run_Manifest.json')['files']
target=V/'restricted/Historical_WAL_Read';target.mkdir(exist_ok=False)
def verify(folder):
    name=folder.name;start=time.time();source=folder/'checkpoint.sqlite';wal=Path(str(source)+'-wal')
    # Check archived byte hashes, then read WAL-dependent databases only from private copies.
    copied=False
    assert sha(source)==original[str(source.relative_to(snapshot))]['sha256']
    if wal.exists():
        assert sha(wal)==original[str(wal.relative_to(snapshot))]['sha256']
    if wal.exists() and wal.stat().st_size:
        dest=target/name;dest.mkdir()
        shutil.copyfile(source,dest/source.name);shutil.copyfile(wal,dest/wal.name)
        assert sha(dest/source.name)==sha(source) and sha(dest/wal.name)==sha(wal)
        db=sqlite3.connect((dest/source.name).as_uri()+'?mode=ro',uri=True);copied=True
    else:db=sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
    assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
    state_row=db.execute("SELECT v,h FROM meta WHERE k='state'").fetchone()
    assert hashlib.sha256(state_row[0].encode()).hexdigest()==state_row[1]
    oldstate=json.loads(state_row[0])
    result=read(R/'restricted'/name/'Result.json')
    assert result['restart_summaries'][:len(oldstate['restart_summaries'])]==oldstate['restart_summaries']
    current=(R/'restricted'/name/'checkpoint.sqlite').as_uri()+'?mode=ro&immutable=1'
    db.execute('ATTACH DATABASE ? AS final',(current,))
    count=db.execute('SELECT COUNT(*) FROM main.fitness').fetchone()[0]
    mismatch=db.execute('SELECT COUNT(*) FROM main.fitness o LEFT JOIN final.fitness f ON o.mask=f.mask WHERE f.mask IS NULL OR f.h!=o.h').fetchone()[0]
    assert mismatch==0
    actual_old_binding=json.loads(db.execute("SELECT v FROM main.meta WHERE k='binding'").fetchone()[0])
    assert actual_old_binding==result['binding']
    expected=read(B/'evidence'/f'Checkpoint_{name}.json')['cached_fitness_records']
    assert count==expected
    initial=read(V/'evidence'/f'Context_{name}.json')['pre_resume_cache_records_preserved']
    db.close()
    # The untouched archived DB and WAL are rechecked after reading private copies.
    assert sha(source)==original[str(source.relative_to(snapshot))]['sha256']
    if wal.exists():assert sha(wal)==original[str(wal.relative_to(snapshot))]['sha256']
    rec={'context':name,'status':'PASS','committed_pre_resume_records':count,'initial_main_file_only_count':initial,
         'additional_WAL_visible_records':count-initial,'missing_or_changed_records':mismatch,'WAL_private_copy_used':copied,
         'original_snapshot_preserved':True,'seconds':time.time()-start}
    write(V/'evidence'/f'Historical_{name}.json',rec);return rec
def main():
    folders=sorted(p.parent for p in (snapshot/'restricted').glob('*/checkpoint.sqlite'))
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        rows=[]
        for f in cf.as_completed([pool.submit(verify,p) for p in folders]):
            r=f.result();rows.append(r);print(json.dumps({k:r[k] for k in ['context','status','additional_WAL_visible_records']}),flush=True)
    total=sum(r['committed_pre_resume_records'] for r in rows);assert total==read(B/'evidence/Checkpoint_Validation.json')['cached_fitness_records']==14109049
    out={'status':'PASS','contexts':len(rows),'committed_pre_resume_records_verified':total,
         'initial_main_file_only_records':sum(r['initial_main_file_only_count'] for r in rows),
         'additional_WAL_visible_records':sum(r['additional_WAL_visible_records'] for r in rows),
         'missing_or_changed_records':sum(r['missing_or_changed_records'] for r in rows),
         'archived_snapshots_preserved':True,'method':'Hash-verified archived snapshot; nonempty WAL databases copied into private verification workspace and read using SQLite read-only WAL semantics. Every committed prior mask/hash is compared to the immutable completed production database.',
         'no_production_changes':True,'contexts_detail':rows}
    write(V/'evidence/Historical_Resume_Preservation.json',out);print(json.dumps({k:v for k,v in out.items() if k!='contexts_detail'}),flush=True)
if __name__=='__main__':main()
