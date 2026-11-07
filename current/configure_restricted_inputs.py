from pathlib import Path
import json,hashlib,shutil,sys
root=Path(__file__).resolve().parent
from code_submit.paper_inward4_replay.input_contract import input_provenance
src=Path(sys.argv[1]).resolve()
expected=json.loads((root/'RESTRICTED_INPUT_HASHES.json').read_text())
for name,sha in expected.items():
 p=src/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
out=root/'outputs/current_ma_20261103';data=out/'data'
assert not (out/'DATA_FREEZE.json').exists(),'Already configured; do not overwrite.'
data.mkdir(parents=True,exist_ok=True)
for name in expected:
 if name=='SUBSET_MASK.json':
  dst=root/'outputs/uteg_reporting_increment_20261030/subset_mask.json';dst.parent.mkdir(parents=True,exist_ok=True)
 else:dst=data/name
 shutil.copy2(src/name,dst)
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
files={str(data/name):sha for name,sha in expected.items() if name not in ['SUBSET_MASK.json']}
write(out/'DATA_FREEZE.json',dict(current_dataset_sha256=expected['DATA15_CURRENT.npz'],files=files,B_from_original_data=True,source_status='author_confirmed_measured',input_provenance=input_provenance()))
write(root/'outputs/ACTIVE_EXPERIMENT_DATASET.json',dict(dataset_path=str(data/'DATA15_CURRENT.npz'),sha256=expected['DATA15_CURRENT.npz']))
(out/'manuscript_sync').mkdir(exist_ok=True)
for name in ['EVIDENCE.json','WORKBOOK_TABLES.json']:
 shutil.copy2(root/name,out/'manuscript_sync'/name)
print('Restricted inputs configured with exact hashes; no data values changed.')
