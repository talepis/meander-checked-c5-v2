"""Small stdlib-only CRT arithmetic/integrity checker. Not a proof of execution."""
from pathlib import Path
import argparse, hashlib, json, math

def integer(x):
    if isinstance(x,bool):raise ValueError('Boolean is not an integer')
    if isinstance(x,int):return x
    if isinstance(x,str) and x.isascii() and x.isdecimal():return int(x)
    raise ValueError('Lossless decimal integer required')

def reconstruct(records):
    x=0;P=1
    for r in records:
        p=integer(r['modulus_decimal']);v=integer(r['residue_decimal'])
        if p<2 or not 0<=v<p or math.gcd(P,p)!=1:raise ValueError('Invalid, duplicate or noncoprime modulus/residue')
        x+=P*(((v-x)*pow(P,-1,p))%p);P*=p
    return x,P

def check(c,folder=None):
    n=integer(c['n']);candidate=integer(c['candidate_decimal'])
    if not 1<=n<=31:raise ValueError('Audited certificate domain is 1..31')
    B=(math.comb(2*n,n)//(n+1))**2
    if not 0<=candidate<=B:raise ValueError('Candidate outside Catalan-squared range')
    records=c['records']
    if not records:raise ValueError('No modular records')
    for r in records:
        if integer(r['n'])!=n or r.get('worker_completed') is not True:raise ValueError('Incomplete or wrong-order worker')
        if r.get('independent_of_C5_V2') is not True:raise ValueError('Missing independence declaration; source audit still required')
        p=integer(r['modulus_decimal']);bits=integer(r['storage_bits']);D=4+2*((n-1)//2)
        if bits not in (16,32,64) or D*(p-1)>=min(1<<bits,1<<63):raise ValueError('Unsafe native arithmetic range')
        if r['source_bundle_digest']!=c['source_bundle_digest']:raise ValueError('Inconsistent numerical source identity')
        if integer(r['max_pre_reduce_decimal'])>D*(p-1):raise ValueError('Observed accumulation exceeds bound')
        if folder is not None:
            name=r['record_filename']
            if Path(name).name!=name:raise ValueError('Unsafe record path')
            payload=(Path(folder)/name).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=r['record_sha256']:raise ValueError('Raw worker record hash mismatch')
            actual=json.loads(payload)
            for key in ('n','worker_completed','independent_of_C5_V2','modulus_decimal','residue_decimal','storage_bits','source_bundle_digest','max_pre_reduce_decimal'):
                if actual[key]!=r[key]:raise ValueError('Certificate disagrees with raw worker output')
    x,P=reconstruct(records)
    if P<=B:raise ValueError('Insufficient CRT product')
    if any(candidate%integer(r['modulus_decimal'])!=integer(r['residue_decimal']) for r in records):raise ValueError('Residue mismatch')
    if x!=candidate:raise ValueError('Reconstructed integer mismatch')
    for key,value in [('reconstructed_decimal',x),('moduli_product_decimal',P),('upper_bound_decimal',B)]:
        if integer(c[key])!=value:raise ValueError('Incorrect stored '+key)
    return dict(status='CRT_ARITHMETIC_AND_RECORD_INTEGRITY_PASS',n=n,
        candidate_decimal=str(candidate),reconstructed_decimal=str(x),
        moduli_product_decimal=str(P),upper_bound_decimal=str(B),passes=len(records),
        scope='Deterministic uniqueness conditional on correctly computed residues and stated combinatorial model. Raw records/hashes are not an attestation or a stand-alone proof of the modular enumerations.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path);args=parser.parse_args()
    print(json.dumps(check(json.loads(args.certificate.read_text(encoding='utf-8')),args.certificate.parent),indent=2))
