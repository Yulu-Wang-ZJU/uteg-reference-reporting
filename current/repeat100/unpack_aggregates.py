from pathlib import Path
import zipfile
root=Path(__file__).parent
out=root
with zipfile.ZipFile(root/'aggregate_results.zip') as z:
    for name in z.namelist():
        if name.startswith('summary/') and name.endswith('.json') and '..' not in Path(name).parts:
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
print(out/'summary')
