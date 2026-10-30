from pathlib import Path
import os,sys,json,hashlib,ast,time,math,shutil
ROOT=Path(__file__).resolve().parent
ROLE=ROOT.parent/'uteg_trust_role_split_20261006_143842'
MAP=ROOT.parent/'uteg_fourparam_mapping_trust_20261006_132600'
GATE=ROOT.parent/'uteg_trust_gate_selection_20261004_205412'
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/py','JOBLIB_TEMP_FOLDER':'cache/joblib','NUMBA_CACHE_DIR':'cache/numba','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'cache/mpl','XDG_CACHE_HOME':'cache/xdg'}.items():
    p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def read(p):
    p=Path(p);return json.loads((p if p.is_absolute() else ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=ROOT/p;assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(p,names,ns):
    a=ast.parse(Path(p).read_text(encoding='utf-8-sig'));nodes=[n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
def guard():
    z=dict(bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),free=shutil.disk_usage(ROOT).free)
    assert z['bytes']<=200*2**20 and z['free']>=6*2**30;return z
