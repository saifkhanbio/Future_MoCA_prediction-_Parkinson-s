"""Execute the frozen prediction stage; permit outcome-file hashing but prohibit text decoding."""
from pathlib import Path
import sys,runpy,contextlib,json,io,datetime,builtins
R=Path(__file__).resolve().parents[1];D=R.parents[1]
assert not (R/'restricted/predictions').exists()
events={'binary_outcome_hash_reads':0,'outcome_text_reads':0}
real_io_open=io.open;real_open=builtins.open
def check(file,mode):
 if 'Sealed_Outcomes_' in str(file):
  if 'b' not in mode:events['outcome_text_reads']+=1;raise PermissionError('No outcome decoding during prediction')
  events['binary_outcome_hash_reads']+=1
def guarded_io(file,mode='r',*args,**kwargs):
 check(file,mode);return real_io_open(file,mode,*args,**kwargs)
def guarded_open(file,mode='r',*args,**kwargs):
 check(file,mode);return real_open(file,mode,*args,**kwargs)
def network_guard(event,args):
 if event in ['socket.connect','socket.getaddrinfo']:raise PermissionError('No network during local evaluation')
sys.addaudithook(network_guard);io.open=guarded_io;builtins.open=guarded_open
sys.argv=[str(D/'scripts/launcher.py'),'predict','--authorization',str(R/'Execution_Authorization.json'),'--output',str(R/'restricted/predictions')]
buf=io.StringIO()
try:
 with contextlib.redirect_stdout(buf):runpy.run_path(str(D/'scripts/launcher.py'),run_name='__main__')
finally:io.open=real_io_open;builtins.open=real_open
(R/'logs/prediction_stdout.txt').write_text(buf.getvalue())
(R/'qa/Prediction_Phase_Audit.json').write_text(json.dumps(dict(events,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_network=True),indent=2))
(R/'qa/Preflight_Guard_Correction.json').write_text(json.dumps({'issue':'Initial wrapper used Python open audit-event mode, which normalizes rb to r; it mistakenly rejected binary integrity hashing before prediction output creation.','correction':'Inspect the actual io.open/builtins.open mode argument instead; permit binary integrity hashing, reject text outcome reads.','locked_launcher_changed':False,'predictions_generated_before_correction':0},indent=2))
print(buf.getvalue())
