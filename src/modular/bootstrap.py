from pathlib import Path
from public_runtime import bootstrap as _load
W=Path(__file__).resolve().parent

def load(variant='checked',n=24,threads=6):
 return _load(W,variant,n,threads)
