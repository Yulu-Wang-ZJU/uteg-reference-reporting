from pathlib import Path
import os,sys,json,hashlib,ast,time,shutil,math,importlib.util

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
for k in ['TEMP','TMP','TMPDIR','PYTHONPYCACHEPREFIX','JOBLIB_TEMP_FOLDER','NUMBA_CACHE_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CACHE_HOME','HF_HOME']:
    p=ROOT/'cache'/k
    p.mkdir(parents=True,exist_ok=True)
    os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
sys.dont_write_bytecode=True

def clean(v):
    import numpy as np
    if isinstance(v,dict): return {str(k):clean(z) for k,z in v.items()}
    if isinstance(v,(list,tuple)): return [clean(z) for z in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,(np.integer,np.bool_)): return v.item()
    if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
    if isinstance(v,Path): return str(v)
    return v

def write(n,z):
    p=ROOT/n
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(z),ensure_ascii=False,indent=2),encoding='utf-8')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def extract(p,names,ns):
    ns.setdefault('__name__','definitions_only')
    nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names),(p,names,[n.name for n in nodes])
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns)

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

def seed(d,s,f):
    return int.from_bytes(hashlib.sha256(f'legacy_aligned_20261022|{d}|{s}|{f}'.encode()).digest()[:8],'little')%2147483647

def guard():
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    free=shutil.disk_usage(ROOT).free
    assert size<2*2**30 and free>6*2**30,(size,free)
    return dict(bytes=size,free=free,science_raw=_LEDGER_TOTALS['science'],
                science_effective=max(0,_LEDGER_TOTALS['science']-RECLASSIFIED_SCIENCE_UNITS),
                reclassified_science_units=RECLASSIFIED_SCIENCE_UNITS,
                total_raw=_LEDGER_TOTALS['total'],
                total_with_snapshot_only=_LEDGER_TOTALS['total']+SNAPSHOT_ONLY_REPAIR_UNITS,
                signal=_LEDGER_TOTALS['signal'],
                repair_accounted=RECLASSIFIED_SCIENCE_UNITS+SNAPSHOT_ONLY_REPAIR_UNITS)

SCIENCE_FIT_CAP=32000
TOTAL_FIT_CAP=32400
SIGNAL_FIT_CAP=1041
REPAIR_FIT_CAP=300
NONSCIENCE_KINDS={'test','repair','audit'}

def _reclassified_science_units():
    """Read the immutable accounting override without changing old ledgers."""
    p=ROOT/'BUDGET_RECLASSIFICATION.json'
    if not p.exists(): return 0
    try:
        z=json.loads(p.read_text(encoding='utf-8-sig'))
        if z.get('reclassify_kind')!='repair': raise ValueError('unsupported kind')
        return int(z.get('reclassify_units',0))
    except Exception as e:
        raise RuntimeError(f'invalid BUDGET_RECLASSIFICATION.json: {p}') from e

RECLASSIFIED_SCIENCE_UNITS=_reclassified_science_units()

def _snapshot_only_repair_units():
    p=ROOT/'BUDGET_RECLASSIFICATION.json'
    if not p.exists(): return 0
    z=json.loads(p.read_text(encoding='utf-8-sig'))
    return int(z.get('snapshot_only_repair_units',0))

SNAPSHOT_ONLY_REPAIR_UNITS=_snapshot_only_repair_units()

def _ledger_totals():
    totals={'science':0,'total':0,'signal':0}
    p=ROOT/'ledger'
    if p.exists():
        for f in p.glob('*.json'):
            try:
                z=json.loads(f.read_text(encoding='utf-8-sig'))
                u=int(z.get('units',0)); kind=str(z.get('kind',''))
                totals['total']+=u
                if kind not in NONSCIENCE_KINDS: totals['science']+=u
                if kind in {'signal','signal_fit'}: totals['signal']+=u
            except Exception:
                continue
    return totals

_LEDGER_TOTALS=_ledger_totals()

def ledger(key,kind,units=1,**kw):
    p=ROOT/'ledger'/f'{key}.json'
    assert not p.exists(),p
    u=int(units)
    nxt={k:_LEDGER_TOTALS[k] for k in _LEDGER_TOTALS}
    nxt['total']+=u
    if str(kind) not in NONSCIENCE_KINDS: nxt['science']+=u
    if str(kind) in {'signal','signal_fit'}: nxt['signal']+=u
    effective_science=max(0,nxt['science']-RECLASSIFIED_SCIENCE_UNITS)
    repair_units=RECLASSIFIED_SCIENCE_UNITS+SNAPSHOT_ONLY_REPAIR_UNITS+(nxt['total']-nxt['science'])
    assert effective_science<=SCIENCE_FIT_CAP,("science fit cap",dict(raw=nxt,effective=effective_science,reclassified=RECLASSIFIED_SCIENCE_UNITS))
    effective_total=nxt['total']+SNAPSHOT_ONLY_REPAIR_UNITS
    assert effective_total<=TOTAL_FIT_CAP,("total fit cap",dict(raw=nxt,effective=effective_total,snapshot_only=SNAPSHOT_ONLY_REPAIR_UNITS))
    assert nxt['signal']<=SIGNAL_FIT_CAP,("signal fit cap",nxt)
    assert repair_units<=REPAIR_FIT_CAP,("repair fit cap",dict(raw=nxt,repair=repair_units,reclassified=RECLASSIFIED_SCIENCE_UNITS))
    write(p.relative_to(ROOT),dict(kind=kind,units=u,**kw))
    _LEDGER_TOTALS.update(nxt)

CO=OUT/'uteg_cohort399_20261006_201800'
OLD=OUT/'uteg_med4_early_decision_20261004_191744'
CAP=OUT/'uteg_med4_capacity_20261006_001701'
CLIN=OUT/'uteg_med4_source_clinical_20261006_104632'
MAP=OUT/'uteg_fourparam_mapping_trust_20261006_132600'
SRC=OUT/'uteg_cohort399_revalidation_20261006_202700'
ZERO=OUT/'uteg_zero5_med4_20261006_113257'
PLAT=OUT/'uteg_platform_param_compare_20261006_125428'
SC=OUT/'uteg_scalar_ordinal_trust_20261006_181750'
TH=OUT/'uteg_trust_threshold90_20261006_161248'
GATE=OUT/'uteg_trust_gate_selection_20261004_205412'
D1=OUT/'uteg_cohort399_gate_revalidation_20261006_203700'
D2=OUT/'uteg_c60_new_trust_20261007_170322'
AD=OUT/'uteg_d2_ci_native85_20261007_012322'
TRI=OUT/'uteg_three_center_validation_20261021_174409'
CLIN3=OUT/'uteg_three_center_clinical_20261021_202117'
NATIVE=OUT/'uteg_hubei_records_20261021_185505/native_bridge'

