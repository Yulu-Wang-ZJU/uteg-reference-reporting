"""Package and verify aggregate-only current manuscript artifacts."""
from pathlib import Path
import json,hashlib,re,zipfile,shutil,ast,importlib.metadata,math
import numpy as np
from . import core as c,current_suite as suite

SYNC=suite.ROOT/'manuscript_sync'
MAN=c.WS/'outputs/uteg_nc_goal_20261030/manuscript'

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def check_workbooks():
    import openpyxl
    books=json.loads((SYNC/'WORKBOOK_TABLES.json').read_text());ledger=[];cells=0
    forbidden={'record_id','patient_id','patient_name','group_id','record_index','ids','reference_ma','true_ma','observed_signal'}
    for no,sheets in books.items():
        path=SYNC/f'deliverables/Supplementary_Data/Supplementary_Data_{no}.xlsx'
        wb=openpyxl.load_workbook(path,read_only=True,data_only=False)
        assert set(wb.sheetnames)==set(sheets),(no,wb.sheetnames)
        for name,records in sheets.items():
            fields=list(dict.fromkeys(k for row in records for k in row))
            assert not forbidden.intersection(fields),(no,name)
            if 'groups' in fields:assert all(isinstance(row.get('groups'),int) for row in records),(no,name,'groups must be aggregate counts')
            expected=[fields]+[[row.get(k) if row.get(k) is not None else 'NA' for k in fields] for row in records]
            sh=wb[name];actual_rows=list(sh.iter_rows())
            assert len(actual_rows)==len(expected) and all(len(row)==len(fields) for row in actual_rows),(no,name,len(actual_rows),len(expected))
            for ri,(actual,exp) in enumerate(zip(actual_rows,expected),1):
                for ci,(cell,v) in enumerate(zip(actual,exp),1):
                    assert cell.data_type not in ('f','e'),(no,name,ri,ci,cell.data_type)
                    got=cell.value
                    if isinstance(v,(float,int)) and not isinstance(v,bool):
                        assert isinstance(got,(float,int)) and math.isclose(got,v,rel_tol=1e-12,abs_tol=1e-12),(no,name,ri,ci,got,v)
                    else:assert got==v,(no,name,ri,ci,got,v)
                    cells+=1
            ledger.append(dict(workbook=int(no),sheet=name,rows=len(records),columns=len(fields),exact_values_verified=True))
        wb.close()
    audit=dict(workbooks=8,sheets=len(ledger),cells=cells,no_formulas_or_cell_errors=True,no_record_level_fields=True,tables=ledger)
    write(SYNC/'WORKBOOK_QA.json',audit)
    agg=json.loads((SYNC/'AGGREGATION_COMPLETE.json').read_text());agg['sheets']={k:len(v) for k,v in books.items()};write(SYNC/'AGGREGATION_COMPLETE.json',agg)
    print('WORKBOOK_QA',len(ledger),cells,flush=True)

def prepare_code():
    """Repack the current allowlisted snapshot without importing old bundles."""
    candidates = [SYNC / 'code_package' / 'CODE_MANIFEST.json',
                  c.WS / 'CODE_MANIFEST.json']
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError('Extract the current code snapshot before repacking.')
    base = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    names = manifest['package_files']
    if len(names) != len(set(names)) or 'CODE_MANIFEST.json' in names:
        raise ValueError('Invalid package allowlist.')
    payload = {}
    for name in names:
        rel = Path(name)
        src = (base / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not src.is_relative_to(base):
            raise ValueError(f'Unsafe package member: {name}')
        if not src.is_file():
            raise FileNotFoundError(src)
        payload[name] = src.read_bytes()
    for field in ['source_files', 'documentation_files', 'protocol_files']:
        if not set(manifest[field]).issubset(payload):
            raise ValueError(f'{field} contains a file outside the allowlist.')
        manifest[field] = {name: hashlib.sha256(payload[name]).hexdigest()
                           for name in manifest[field]}
    payload['CODE_MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False,
                                               indent=2) + '\n').encode('utf-8')
    MAN.mkdir(parents=True, exist_ok=True)
    dest = MAN / 'Current_analysis_code.zip'
    staged = dest.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(staged, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(payload):
                archive.writestr(name, payload[name])
        with zipfile.ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError('Code archive integrity check failed.')
        staged.replace(dest)
        manifest_path.write_bytes(payload['CODE_MANIFEST.json'])
    finally:
        if staged.exists():
            staged.unlink()
    print('CURRENT_CODE_PACKAGE_REPACKED', len(payload), flush=True)

def rescore():
    from . import current_finalize as check
    from .gates import clean_write
    suite.initialize();check.OUT.mkdir(exist_ok=True)
    d,_,_,_,_,folds,local=c.context()
    check.inventory();check.split_audit(d,folds,local);check.aggregate(d,local)
    check.sequence_audit(d);check.matched_and_intervals(d,local)
    clean_write(check.OUT/'METRICS.json',dict(results=check.SCORES,scope='Current fixed-input rescore; historical archive audit not rerun.'))
    print('CURRENT_RESULTS_RESCORED',len(check.SCORES),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='workbooks':check_workbooks()
    elif sys.argv[1]=='code':prepare_code()
    elif sys.argv[1]=='rescore':rescore()
