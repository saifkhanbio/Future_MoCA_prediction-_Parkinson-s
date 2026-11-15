"""CLI for Stage A temporal PPMI and Stage B nominal PDBP protocols."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from evaluation import evaluate,require
from fixtures import fixture
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(files,root):
    for name,expected in files.items():
        require(digest(root/name)==expected,'File checksum mismatch: '+name)
def run(stage,mode,authorization=None):
    lock=read(ROOT/'LOCK.json')
    verify(lock['public'],ROOT)
    plan=read(ROOT/f'Protocol_{stage}.json');spec=read(ROOT/'model_spec.json')
    if mode=='synthetic':
        models,x,y=fixture(plan,spec)
        require(all(r['id'].startswith('SYN-') for r in x+y),'Synthetic identity')
        return evaluate(plan,spec,models,x,y)
    verify(lock['restricted'],ROOT)
    verify(read(ROOT/'source_hashes.json'),ROOT.parents[2].parent)
    if mode=='preflight':return {'stage':stage,'checksums':'pass','predictions':0}
    require(mode=='study' and authorization==digest(ROOT/'LOCK.json'),'An explicit study run requires this protocol lock digest')
    r=ROOT/'restricted'; out=r/f'Study_results_{stage}.json'
    require(not out.exists(),'Study results already exist')
    result=evaluate(plan,spec,read(r/f'models_{stage}.json'),read(r/f'features_{stage}.json'),read(r/f'outcomes_{stage}.json'))
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=r,delete=False) as tmp:
        path=Path(tmp.name)
        json.dump({'stage':stage,'protocol_lock':authorization,'results':result},tmp,allow_nan=False)
    try:os.link(path,out)
    finally:path.unlink(missing_ok=True)
    return {'stage':stage,'result':out.name}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['A','B'])
    parser.add_argument('--mode',choices=['synthetic','preflight','study'],default='synthetic')
    parser.add_argument('--study-lock',help='Protocol digest for separately authorized local study execution')
    args=parser.parse_args()
    result=run(args.stage,args.mode,args.study_lock)
    print(json.dumps(result,allow_nan=False))
