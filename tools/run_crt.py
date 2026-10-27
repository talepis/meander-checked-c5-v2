"""Sequential CRT reproduction using the audited public modular driver."""
from pathlib import Path
import argparse,subprocess,sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('n',type=int);ap.add_argument('--mode',choices=['modular','independent'],default='modular');ap.add_argument('--threads',type=int);ap.add_argument('--ram-gib',type=float);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 if not 1<=a.n<=(31 if a.mode=='independent' else 34):ap.error('Outside audited domain')
 target=a.output_dir.resolve();target.mkdir(parents=True,exist_ok=False)
 mods=[(1<<57)-1,(1<<57)-3,1927 if a.mode=='independent' else 64];records=[]
 for i,m in enumerate(mods):
  p=target/f'pass_{i}.json';cmd=[sys.executable,str(ROOT/'meander.py'),str(a.n),'--mode',a.mode,'--modulus',str(m),'--output',str(p)]
  if a.threads is not None:cmd+=['--threads',str(a.threads)]
  if a.ram_gib is not None:cmd+=['--ram-gib',str(a.ram_gib)]
  subprocess.run(cmd,check=True);r=json.loads(p.read_text());records.append(dict(filename=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),modulus_decimal=str(m),residue_decimal=r['residue_decimal']))
 x=0;product=1
 for r in records:
  m=int(r['modulus_decimal']);x+=product*(((int(r['residue_decimal'])-x)*pow(product,-1,m))%m);product*=m
 bound=(math.comb(2*a.n,a.n)//(a.n+1))**2
 if product<=bound or x>bound:raise ArithmeticError('Insufficient uniqueness or incompatible reconstruction')
 cert=dict(n=a.n,reconstructed_decimal=str(x),moduli_product_decimal=str(product),upper_bound_decimal=str(bound),passes=3,independent_of_C5_V2=a.mode=='independent',records=records,scope='Fresh reproduction. Checkers for historical publication records use their separate schemas.')
 (target/'CRT_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n');print(json.dumps(cert,indent=2))

if __name__=='__main__':main()
