"""Audit published records and CRT uniqueness without NumPy or Numba."""
from pathlib import Path
import hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[1]

def integer(x):
 if isinstance(x,bool):raise ValueError('Boolean is not a decimal integer')
 if isinstance(x,int):return x
 if isinstance(x,str) and re.fullmatch(r'0|[1-9][0-9]*',x):return int(x)
 raise ValueError('Expected lossless nonnegative decimal integer')

def check(c,root=ROOT):
 root=Path(root).resolve();n=integer(c['n'])
 if not 29<=n<=34:raise ValueError('Certificate order outside this publication')
 if c['independent_of_C5_V2'] is not (n<=31):raise ValueError('Incorrect independence scope')
 if len(c['records'])!=integer(c['passes']) or c['passes']!=(10 if n<=30 else 3):raise ValueError('Incomplete passes')
 x=0;product=1
 for r in c['records']:
  m=integer(r['modulus_decimal']);a=integer(r['residue_decimal'])
  if m<2 or a>=m or math.gcd(product,m)!=1:raise ValueError('Invalid CRT modulus/residue')
  path=(root/r['filename']).resolve()
  if path==root or root not in path.parents:raise ValueError('Record path escapes release root')
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Record SHA mismatch')
  actual=json.loads(raw)
  if integer(actual['n'])!=n or integer(actual.get('modulus_decimal',actual.get('modulus')))!=m or integer(actual.get('residue_decimal',actual.get('residue')))!=a:raise ValueError('Record content mismatch')
  if n<=31:
   if actual.get('worker_completed') is not True:raise ValueError('Incomplete independent worker')
   flag='independent_of_C5_V2' if n==31 else 'independent_of_projected_sources'
   if actual.get(flag) is not True:raise ValueError('Missing independent arithmetic declaration')
  else:
   if actual.get('status')!='PASS' or actual.get('variant')!='v2':raise ValueError('Incorrect V2 record')
   if actual['result'].get('extremal_hypothesis_required_for_correctness') is not False:raise ValueError('Unchecked V2 record')
  if n<=30:
   supervision=json.loads(path.with_name(path.stem+'.supervision.json').read_text())
   if supervision.get('accepted') is not True or supervision.get('exit_code')!=0:raise ValueError('Rejected historical worker')
  layers=actual.get('layers')
  if layers is not None and (len(layers)!=2*n-1 or [integer(s['t']) for s in layers]!=list(range(1,2*n))):raise ValueError('Incomplete stage history')
  x+=product*(((a-x)*pow(product,-1,m))%m);product*=m
 bound=(math.comb(2*n,n)//(n+1))**2
 if product<=bound or product!=integer(c['moduli_product_decimal']) or bound!=integer(c['upper_bound_decimal']) or x>bound or x!=integer(c['reconstructed_decimal']):raise ValueError('CRT uniqueness or reconstruction mismatch')
 return dict(n=n,Mn=str(x),passes=len(c['records']),independent_of_C5_V2=c['independent_of_C5_V2'])

def main():
 certificates=json.loads((ROOT/'evidence/CRT_CERTIFICATES.json').read_text())['certificates'];results=[check(c) for c in certificates]
 if [r['n'] for r in results]!=list(range(29,35)):raise ValueError('Missing certificate')
 raw=(ROOT/'data/b005315.txt').read_bytes();rows=raw.decode('ascii').strip().splitlines()
 if not raw.endswith(b'\n\n') or len(rows)!=35:raise ValueError('Invalid b-file')
 table={}
 for n,line in enumerate(rows):
  if not re.fullmatch(r'(0|[1-9][0-9]*) [1-9][0-9]*',line):raise ValueError('Invalid b-file line')
  k,v=map(int,line.split());table[k]=v
  if n!=k:raise ValueError('Nonconsecutive b-file')
 for line in (ROOT/'data/OEIS_0_28_REFERENCE.txt').read_text().splitlines():
  if line.strip():
   n,v=map(int,line.split())
   if table[n]!=v:raise ValueError('Old OEIS term changed')
 if any(table[r['n']]!=int(r['Mn']) for r in results):raise ValueError('New b-file term differs')
 print(json.dumps(dict(status='PUBLIC_CERTIFICATES_AND_TABLE_PASS',results=results,scope='Arithmetic uniqueness and record integrity conditional on correctly computed residues; no execution attestation or formal toolchain proof.'),indent=2))

if __name__=='__main__':main()
