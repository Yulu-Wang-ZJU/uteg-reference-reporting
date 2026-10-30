from pathlib import Path
import os,sys,json,hashlib,ast,math,time,shutil
ROOT=Path(__file__).resolve().parent;ZERO=ROOT.parent/'uteg_zero5_med4_20261006_113257';PREV=ROOT.parent/'uteg_med4_source_clinical_20261006_104632';CAP=ROOT.parent/'uteg_med4_capacity_20261006_001701';OLD=ROOT.parent/'uteg_med4_early_decision_20261004_191744';PILOT=ROOT.parent/'uteg_platform_filter_pilot_20261006_124636';AUDIT=ROOT.parent/'uteg_processing_harmonization_audit_20261006_121445'
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','PYTHONPYCACHEPREFIX':'cache/py','JOBLIB_TEMP_FOLDER':'cache/joblib','NUMBA_CACHE_DIR':'cache/numba','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'cache/mpl','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/hf'}.items():
    p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def read(p):
    p=Path(p);return json.loads((p if p.is_absolute() else ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=ROOT/p;assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(p,names,ns):
    nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
def seed(d,s,f):return int.from_bytes(hashlib.sha256(f'capacity_v1|{d}|{s}|{f}'.encode()).digest()[:8],'little')%2147483647
def ledger(kind,key,**kw):
    p=ROOT/f'ledger/{key}.json';assert not p.exists();assert len(list((ROOT/'ledger').glob('*.json')))<10000;write(p.relative_to(ROOT),dict(kind=kind,key=key,state='started',started=time.time(),**kw))
def finish(key,**kw):
    x=read(f'ledger/{key}.json');x.update(state='finished',finished=time.time(),**kw);write(f'ledger/{key}.json',x)
def guard():
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());free=shutil.disk_usage(ROOT).free
    if size>2*2**30 or free<6*2**30:raise RuntimeError('storage cap')
    return dict(bytes=size,free=free)
PREPS=['FILTER','P017','P1','P3','P5'];FAMILIES=['F5','F4','F3'];ARMS=['BASE5']+[p+'_'+f for p in PREPS for f in FAMILIES]
FILES=dict(zip(PREPS,['FILTER_ONLY','CUT_017','CUT_1','CUT_3','CUT_5']))
