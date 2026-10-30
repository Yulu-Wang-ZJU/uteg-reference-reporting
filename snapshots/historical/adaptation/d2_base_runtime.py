from pathlib import Path
import os,sys,json,hashlib,ast,math,time,shutil,importlib.util
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
COHORT=OUT/'uteg_cohort399_20261006_201800';MAP=OUT/'uteg_cohort399_revalidation_20261006_202700';PREV=OUT/'uteg_cohort399_gate_revalidation_20261006_203700'
GATE=OUT/'uteg_trust_gate_selection_20261004_205412';TH=OUT/'uteg_trust_threshold90_20261006_161248';CF=OUT/'uteg_trust_crossfit40_20261006_170625';SC=OUT/'uteg_scalar_ordinal_trust_20261006_181750'
OLD=OUT/'uteg_med4_early_decision_20261004_191744';ZERO=OUT/'uteg_zero5_med4_20261006_113257';OLDMAP=OUT/'uteg_fourparam_mapping_trust_20261006_132600'
sys.dont_write_bytecode=True
for k in ['TEMP','TMP','TMPDIR','PYTHONPYCACHEPREFIX','JOBLIB_TEMP_FOLDER','NUMBA_CACHE_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CACHE_HOME','HF_HOME']:
 p=ROOT/'cache'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
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
 p=ROOT/p;assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(p,names,ns):
 nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def guard():
 z=dict(bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),E_free=shutil.disk_usage(ROOT).free);assert z['bytes']<=2*2**30 and z['E_free']>=6*2**30;return z
def ledger(key,kind,**kw):
 rows=list((ROOT/'ledger').glob('*.json'));assert len(rows)<1600;assert not (ROOT/f'ledger/{key}.json').exists();write(f'ledger/{key}.json',dict(kind=kind,state='started',time=time.time(),**kw))
def finish(key,**kw):
 z=read(f'ledger/{key}.json');z.update(state='finished',**kw);write(f'ledger/{key}.json',z)
