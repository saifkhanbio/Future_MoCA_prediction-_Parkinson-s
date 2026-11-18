"""Scheduling-only resume: nine CPU task workers, unchanged estimators."""
import execution as X
from concurrent.futures import ProcessPoolExecutor as OriginalExecutor
import json,hashlib

def scheduled_executor(*args,**kwargs):
    if kwargs.get('max_workers')==3:kwargs['max_workers']=9
    return OriginalExecutor(*args,**kwargs)

if __name__=='__main__':
    for f,h in json.loads((X.E/'PARALLEL_LOCK.json').read_text())['files'].items():
        assert hashlib.sha256((X.E/f).read_bytes()).hexdigest()==h
    X.ProcessPoolExecutor=scheduled_executor
    X.main()
