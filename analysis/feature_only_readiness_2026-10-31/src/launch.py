"""Validate only by default; participant execution requires a separately authorized, hash-bound run."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,argparse
from engine import require,sha,execute_dataset,IntegrityError
from binding import validate_real,ROOT
def validate_authorization(root,path):
    root=Path(root);require(path is not None,'Explicit participant training authorization is required')
    auth=json.loads(Path(path).read_text());lock=root/'LAUNCHER_LOCK.json'
    require(lock.is_file(),'Validated launcher lock is required')
    require(auth.get('participant_training_authorized') is True and auth.get('study_id')=='Clinical_hallmark_feature_only_v1' and isinstance(auth.get('user_instruction'),str) and len(auth['user_instruction'].strip())>0,'Invalid participant authorization')
    require(auth.get('plan_lock_sha256')==sha(root/'PLAN_LOCK.json') and auth.get('launcher_lock_sha256')==sha(lock),'Authorization does not bind these locks')
    payload=json.loads(lock.read_text())
    require(payload['status']=='VALIDATED_SYNTHETIC_ONLY_NOT_TRAINING_AUTHORIZATION','Launcher is not validated')
    for rel,h in payload['sha256'].items():require(sha(root/rel)==h,'Validated launcher file changed')
    return auth
def output_path(root,path):
    require(path is not None,'Fresh output folder is required')
    p=Path(path).absolute();study=Path(root).parent.resolve()
    require(not p.is_symlink() and p.parent.resolve()==study and p.name.startswith('feature_only_run_'),'Output must be a fresh direct feature_only_run_ sibling in Clinical_hallmark')
    require(not p.exists(),'Output already exists')
    return p
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['validate','run']);ap.add_argument('--output');ap.add_argument('--authorization');ap.add_argument('--workers',type=int,default=12);a=ap.parse_args(argv)
    if a.command=='validate':
        *_,report=validate_real();print(json.dumps(report));return
    auth=validate_authorization(ROOT,a.authorization);out=output_path(ROOT,a.output)
    require(1<=a.workers<=12,'Worker count out of bounds')
    c,by,cohorts,refs,report=validate_real()
    result=execute_dataset(c,by,cohorts,refs,out,a.workers,False,{'authorization':auth,'plan_sha256':sha(ROOT/'PLAN_LOCK.json'),'launcher_sha256':sha(ROOT/'LAUNCHER_LOCK.json'),'input_validation':report})
    print(json.dumps(result))
if __name__=='__main__':main()
