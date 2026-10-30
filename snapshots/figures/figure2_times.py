from pathlib import Path
import os,re,json
from lxml import etree as ET
import fitz
# Release redaction: the source used one workstation-specific Anaconda DLL
# path.  Resolve it from an explicit environment override or CONDA_PREFIX so
# the plotting transformation remains portable without exposing that path.
_dll_root = os.environ.get('UTEG_CAIROSVG_DLL_ROOT')
if not _dll_root and os.environ.get('CONDA_PREFIX'):
    _dll_root = str(Path(os.environ['CONDA_PREFIX']) / 'Library' / 'bin')
if _dll_root:
    _add_dll_directory = getattr(os, 'add_dll_directory', None)
    if _add_dll_directory is not None:
        _add_dll_directory(_dll_root)
    os.environ['PATH'] = _dll_root + os.pathsep + os.environ.get('PATH', '')
import cairosvg
B=Path(__file__).resolve().parents[1]
src=B/'figures/terminology_aligned_20261029/Figure2_translation.svg'
out=B/'figures/times_roman_20261029';out.mkdir(exist_ok=True)
s=src.read_text(encoding='utf-8')
t=re.sub(r'font-family:[^;"\n]+',"font-family: 'Times New Roman'",s)
ns={'s':'http://www.w3.org/2000/svg'}
a=ET.fromstring(s.encode());b=ET.fromstring(t.encode())
assert a.xpath('//s:text//text()',namespaces=ns)==b.xpath('//s:text//text()',namespaces=ns)
for tag in ['path','use','image','line','polygon','polyline','rect','circle','ellipse']:
    assert [ET.tostring(x) for x in a.xpath('//s:'+tag,namespaces=ns)]==[ET.tostring(x) for x in b.xpath('//s:'+tag,namespaces=ns)]
p=out/src.name;p.write_text(t,encoding='utf-8')
cairosvg.svg2pdf(url=str(p),write_to=str(p.with_suffix('.pdf')))
with fitz.open(p.with_suffix('.pdf')) as d:
    fonts=[f[3] for f in d[0].get_fonts()]
    assert all('TimesNewRoman' in f for f in fonts),fonts
    d[0].get_pixmap(matrix=fitz.Matrix(3,3)).save(p.with_suffix('.png'))
(out/'QA.json').write_text(json.dumps(dict(fonts=fonts,text_unchanged=True,geometry_unchanged=True),indent=2))
print(fonts)
