"""Verify release-file hashes, preserved numerical sources, and CRT records."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
 if manifest.get('format')!='meander-release-sha256-v1':raise ValueError('Wrong manifest format')
 seen=set()
 for r in manifest['files']:
  name=r['path'];p=(ROOT/name).resolve()
  if p==ROOT or ROOT not in p.parents or name in seen:raise ValueError('Bad manifest path')
  seen.add(name)
  if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Release file differs: '+name)
 source=json.loads((ROOT/'provenance/NUMERICAL_SOURCE_SHA256.json').read_text(encoding='utf-8'))
 for r in source:
  if not r['byte_identical'] or r['original_SHA256']!=r['public_SHA256'] or digest(ROOT/r['public_file'])!=r['original_SHA256']:raise ValueError('Numerical source differs')
 from verify_certificates import main as crt_check
 crt_check()
 print(json.dumps(dict(status='RELEASE_INTEGRITY_AND_CRT_PASS',files=len(seen),numerical_sources=len(source))),flush=True)
if __name__=='__main__':main()
