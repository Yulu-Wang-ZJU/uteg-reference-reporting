"""Task-local environment. Import BEFORE scientific or historical modules."""
from pathlib import Path
import os, sys, json, hashlib, math, shutil
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
OLD = OUT / 'uteg_med4_early_decision_20261004_191744'
MED = OUT / 'uteg_med4_reconstruction_20261004_155035'
CI = OUT / 'teg_ci_r_alpha_20261003_212646'
DATA = OUT / 'uteg_individual_translation_20261002_142224/private/data.npz'
for key, sub in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/py','JOBLIB_TEMP_FOLDER':'cache/joblib','NUMBA_CACHE_DIR':'cache/numba','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'cache/mpl','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/hf','CUDA_CACHE_PATH':'cache/cuda'}.items():
    path = ROOT / sub
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x

def read(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p,x):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    if not p.resolve().is_relative_to(ROOT.resolve()): raise RuntimeError('Write outside new run')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)
    return p

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def guard():
    n = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    free = shutil.disk_usage(ROOT).free
    if n > 2*2**30 or free < 6*2**30: raise RuntimeError('Storage budget reached')
    return dict(bytes=n, free_bytes=free)
