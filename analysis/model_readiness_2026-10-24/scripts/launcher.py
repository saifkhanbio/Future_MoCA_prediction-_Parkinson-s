"""Default mode is read-only validation. Training requires a separate future authorization."""
import sys,os,argparse,json
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
D=Path(__file__).resolve().parents[1];sys.path.insert(0,str(D/'src'))
from bundle import load_bundle,authorize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','run'],nargs='?',default='validate');ap.add_argument('--views',nargs='+');ap.add_argument('--authorization');ap.add_argument('--output');args=ap.parse_args()
 if args.mode=='validate':
  b=load_bundle(D,require_seal=True);print(json.dumps({'status':'VALIDATED_NO_FITTING','views':len(b['views']),'baseline_components':len(b['features']),'planned_budget':b['plan']['budget'],'models_fit':0}));return
 if not args.output or not args.views:raise PermissionError('Run requires explicit output and selected views')
 out=authorize(D,args.authorization,args.views,args.output)
 b=load_bundle(D,require_seal=True)
 from engine import RealBackend
 from runner import execute
 result=execute(b,out,args.views,RealBackend(b),qa_only=False)
 print(json.dumps(result))
if __name__=='__main__':main()