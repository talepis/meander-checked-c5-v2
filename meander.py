"""Small public CLI; dispatches one computation in a fresh process."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 if not __debug__:raise RuntimeError('Python -O/-OO is unsupported')
 ap=argparse.ArgumentParser(description='Exact closed meander numbers, audited orders 0..34, CPU and RAM only.')
 ap.add_argument('n',type=int);ap.add_argument('--mode',choices=['exact','modular','independent'],default='exact')
 ap.add_argument('--modulus',type=int);ap.add_argument('--threads',type=int);ap.add_argument('--ram-gib',type=float)
 ap.add_argument('--reserve-gib',type=float);ap.add_argument('--proved-cap',action='store_true');ap.add_argument('--output',type=Path)
 a=ap.parse_args()
 if not 0<=a.n<=34:ap.error('Audited public domain: 0..34. Larger orders are excluded from this release.')
 if a.mode!='exact' and (a.n==0 or a.modulus is None):ap.error('Modular modes require n>=1 and --modulus')
 if a.mode=='exact' and a.modulus is not None:ap.error('An exact calculation has no modulus')
 if a.mode=='independent' and a.n>31:ap.error('Independent verifier is audited only through n=31')
 args=[sys.executable,str(ROOT/'src/common/run.py'),str(a.n),'--mode',a.mode]
 for k,v in vars(a).items():
  if k in ('n','mode') or v is None or v is False:continue
  flag='--'+k.replace('_','-');args.append(flag)
  if v is not True:args.append(str(v))
 return subprocess.run(args,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}).returncode

if __name__=='__main__':sys.exit(main())
