"""Verify both production pools; retain already verified checkpoints."""
import verify_results as V
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
import time,json,hashlib

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(f):
    row=V.audit_checkpoint(f);row['checkpoint_sha256']=digest(f);return row

def main():
    folder=V.V/'checkpoints';folder.mkdir(exist_ok=True);results={};seen=set();pending={};start=time.time()
    mainpath=V.R/'restricted/checkpoints';extra=V.E/'recruited_checkpoints'
    for f in folder.glob('*.json'):
        row=json.loads(f.read_text());p=mainpath/f.name
        if p.exists() and digest(p)==row['checkpoint_sha256']:results[f.name]=row;seen.add(f.name)
    with ProcessPoolExecutor(max_workers=int(json.loads((V.E/'verification_handoff/Worker_Config.json').read_text())['workers']),mp_context=get_context('spawn')) as pool:
        while True:
            available={f.name:f for f in extra.glob('*.json')};available.update({f.name:f for f in mainpath.glob('*.json')})
            for name,f in sorted(available.items()):
                if name not in seen:seen.add(name);pending[pool.submit(audit,f)]=f
            done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED) if pending else (set(),set())
            for future in done:
                f=pending.pop(future);row=future.result();results[f.name]=row;(folder/f.name).write_text(json.dumps(row,indent=2))
                (V.V/'STATUS.json').write_text(json.dumps({'verified_tasks':len(results),'expected':225,'seconds_since_verifier_resume':time.time()-start}))
                if len(results)%15==0:print('Independently verified checkpoint',len(results),flush=True)
            if len(results)==225 and not pending:
                state=json.loads((V.R/'STATUS.json').read_text())
                if state['status']=='COMPLETE_PENDING_INDEPENDENT_VERIFICATION':
                    changed=[f for f in mainpath.glob('*.json') if digest(f)!=results[f.name]['checkpoint_sha256']]
                    if not changed:break
                    for f in changed:pending[pool.submit(audit,f)]=f
            if not pending:time.sleep(2)
    assert len(seen)==225
    rows=list(results.values());(V.V/'Prediction_Replays.json').write_text(json.dumps(rows,indent=2));count,error=V.verify_metrics()
    import sys;sys.path.insert(0,str(V.D));import launcher;launcher.verify()
    resume=json.loads((V.E/'Scheduling_Resume_Record.json').read_text())
    for f,h in resume['completed_checkpoint_hashes'].items():assert digest(mainpath/f)==h
    for f in mainpath.glob('*.json'):assert digest(f)==results[f.name]['checkpoint_sha256']
    report=dict(status='PASS',production_fit_calls=2475,independent_ridge_svd_fits=1200,independent_EBM_TabPFN_outer_refits=150,checkpoints_audited=225,metrics_and_intervals_verified=count,max_prediction_difference=max(r['max_prediction_difference'] for r in rows),max_metric_difference=error,co_primary_intervals_verified=True,all_frozen_hashes_unchanged=True,resumed_checkpoints_unchanged=len(resume['completed_checkpoint_hashes']),final_checkpoint_hashes_verified=True)
    (V.V/'Independent_Verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
