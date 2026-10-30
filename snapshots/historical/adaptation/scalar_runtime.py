from pathlib import Path
import os,sys,json,hashlib,ast,math,time,shutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
MAP=OUT/'uteg_fourparam_mapping_trust_20261006_132600';CF=OUT/'uteg_trust_crossfit40_20261006_170625'
TH=OUT/'uteg_trust_threshold90_20261006_161248';GATE=OUT/'uteg_trust_gate_selection_20261004_205412';ZERO=OUT/'uteg_zero5_med4_20261006_113257'
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/py','JOBLIB_TEMP_FOLDER':'cache/joblib','NUMBA_CACHE_DIR':'cache/numba','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'cache/mpl','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/hf'}.items():
    p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def write(n,x):
    p=ROOT/n;assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def extract(p,names,ns):
    nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
def guard():
    a=dict(bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),E_free=shutil.disk_usage(ROOT).free)
    assert a['bytes']<=100*2**20 and a['E_free']>=6*2**30;return a
def dump(n,x):
    import joblib
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(x,p,compress=3)
