"""Distribution-only bootstrap. Original numerical kernels are unchanged."""
from pathlib import Path
import os,sys,hashlib,shutil,subprocess,importlib.util,time,tempfile
ROOT=Path(__file__).resolve().parents[2]
FOLDERS=('global_quotient/src','m31','disk_tiles','benchmark_runtime','global_quotient','contraction_cycle','contraction_extended','ram_fanout','n27_stability')
MAP_SHA='36b69e1c416c2e6888718b39aed0e391559dd16c720e1b39907174b3f848850a'
SETTINGS={};TIMINGS={};ACTIVE=None

def materialize_map(destination):
 source=ROOT/'data/meander_closemaps_n25.pkl'
 if hashlib.sha256(source.read_bytes()).hexdigest()!=MAP_SHA:raise RuntimeError('Untrusted closing-map payload')
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists():
  if hashlib.sha256(destination.read_bytes()).hexdigest()!=MAP_SHA:raise RuntimeError('Closing-map copy differs')
 else:
  try:os.link(source,destination)
  except OSError:shutil.copyfile(source,destination)

def bootstrap(W,variant='checked',n=24,threads=6):
 global ACTIVE
 if not __debug__:raise RuntimeError('Python -O/-OO is unsupported')
 wide=n>=34 or os.environ.get('V2_TEST_WIDE')=='1'
 names=('checked','checked_wide') if W.name=='modular' else ('project','project_wide')
 root=W/names[int(wide)]
 if ACTIVE is not None:
  if ACTIVE[0]!=root:raise RuntimeError('Use a fresh process when changing source widths')
  return ACTIVE[1]
 start=time.perf_counter()
 os.environ.update(NUMBA_NUM_THREADS=str(threads),MEANDER_NUMBA_THREADS=str(threads),MEANDER_TERMINAL_BACKEND='cpu',NUMBA_THREADING_LAYER='omp',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_CACHE_DIR=str(Path(os.environ.get('MEANDER_CACHE_DIR',str(Path(tempfile.gettempdir())/'mc'/hashlib.sha256(str(ROOT).encode()).hexdigest()[:6])))))
 materialize_map(root/'work/global_quotient/src/baseline/base/meander_closemaps_n25.pkl')
 sys.path[:0]=[str(root/'work'/f) for f in FOLDERS]
 import allocation_guard as G
 if sys.platform.startswith('linux'):
  common=Path(__file__).resolve().parent
  lib=common/'linux_ram.so'
  if not lib.is_file():
   cc=shutil.which('cc') or shutil.which('gcc')
   if not cc:raise RuntimeError('A C compiler is required for the native Linux RAM guard; install build-essential or GCC.')
   compile_start=time.perf_counter();subprocess.run([cc,'-O3','-shared','-fPIC',str(common/'linux_ram.c'),'-o',str(lib)],check=True)
   TIMINGS['native_guard_build_seconds']=time.perf_counter()-compile_start
  import linux_guard
  linux_guard.install()
 elif os.name=='nt':
  import fast_memory
  G.check=fast_memory.check
 else:raise RuntimeError('This release is tested for Windows x64 and Linux x86-64 only')
 import bounded_storage as BS
 def no_file(block):raise MemoryError('Public RAM-only release: coefficient-file creation is forbidden')
 BS.Block._open_file=no_file
 import numba as nb,variants_next as V,configure_bugfirst as B,functional_engine as F
 nb.set_num_threads(threads)
 if V.configure('projected-family-sym')!=2:raise RuntimeError('Unexpected source configuration')
 B.load_maps(n)
 value=(F,V,BS,G,SETTINGS['pool_bytes']);ACTIVE=(root,value)
 TIMINGS['bootstrap_and_map_seconds']=time.perf_counter()-start
 return value
