from pathlib import Path
import os,sys,json,subprocess,datetime,time,hashlib
os.umask(0o077)
C=Path(__file__).resolve().parents[1];P=C.parent;D=P/'model_readiness_2026-10-24';out=P/'executions/main_2026-10-28'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert read(C/'qa/Numerical_Checks.json')['passed'] and read(C/'qa/Backend_Integration.json')['passed']
auth=read(C/'config/Execution_Authorization.json');assert auth['plan_lock_sha256']==sha(D/'PLAN_LOCK.json')
assert not out.exists() and not (C/'evidence/Production_Started.json').exists()
pre=subprocess.run([sys.executable,'-B',str(D/'scripts/launcher.py'),'validate'],capture_output=True,text=True);assert pre.returncode==0,pre.stderr
write(C/'qa/Production_Preflight.json',json.loads(pre.stdout))
write(C/'evidence/Production_Started.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_hash':sha(D/'PLAN_LOCK.json'),'numerical_QA_hash':sha(C/'qa/Numerical_Checks.json'),'integration_QA_hash':sha(C/'qa/Backend_Integration.json'),'authorization_hash':sha(C/'config/Execution_Authorization.json'),'view_ids':auth['view_ids'],'expected_production_fits':5970,'output':str(out),'paid_queries':0})
cmd=[sys.executable,'-B','-u',str(D/'scripts/launcher.py'),'run','--views']+auth['view_ids']+['--authorization',str(C/'config/Execution_Authorization.json'),'--output',str(out)]
start=time.monotonic()
with (C/'restricted/Launcher_Stdout.log').open('w') as f,(C/'restricted/Launcher_Stderr.log').open('w') as e:
 job=subprocess.Popen(cmd,stdout=f,stderr=e);write(C/'evidence/Process.json',{'pid':job.pid,'parent_pid':os.getpid(),'status':'RUNNING','output':str(out)});print('Locked production run started: six main views, 5970 fits.',flush=True);code=job.wait()
record={'returncode':code,'wall_seconds':time.monotonic()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output':str(out),'paid_queries':0};write(C/'evidence/Production_Process_Result.json',record)
if code!=0:raise RuntimeError('Production launcher failed; inspect preserved restricted logs')
counts=read(out/'Run_Counts.json');assert counts['actual_ridge_solves']==5970 and counts['status']=='COMPLETE' and not counts['qa_only']
print(json.dumps({'process':record,'counts':counts}),flush=True)