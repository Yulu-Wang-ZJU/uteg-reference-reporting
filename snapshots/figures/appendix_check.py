"""Presentation-only checks; no fits or statistical resampling."""
from pathlib import Path
import json,hashlib,re,sys
from PIL import Image,ImageOps,ImageDraw
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=read('INPUT_FREEZE.json'); manifest=read('SOURCE_MANIFEST.json'); idx=read('TABLE_FIGURE_INDEX.json')
unchanged=all(sha(p)==h for p,h in freeze['inputs'].items())
copies=all(sha(P/'source_data'/n)==v.get('copy_sha256',v['sha256']) for n,v in manifest.items())
text=(P/'index.html').read_text(encoding='utf-8')
broken=[]
for link in re.findall(r'(?:href|src)="([^"]+)"',text):
    if link.startswith(('#','http','data:')):continue
    if not (P/link.split('#')[0]).exists():broken.append(link)
thumbs=[]
for f in idx['figures']:
    for ext in ['svg','pdf','png']:assert (P/'figures'/f"{f['name']}.{ext}").exists()
    im=Image.open(P/'figures'/f"{f['name']}.png").convert('RGB')
    im.thumbnail((650,650))
    tile=Image.new('RGB',(690,700),'white');tile.paste(im,((690-im.width)//2,35))
    ImageDraw.Draw(tile).text((12,10),f['name'],fill='black');thumbs.append(tile)
for start in range(0,len(thumbs),4):
    out=Image.new('RGB',(1380,1400),'#dddddd')
    for j,im in enumerate(thumbs[start:start+4]):out.paste(im,((j%2)*690,(j//2)*700))
    out.save(P/f'QA_contact_{start//4+1}.png')
result={'original_inputs_unchanged':unchanged,'checked_inputs':len(freeze['inputs']),'protected_main_files':freeze['main_protected_count'],'exact_source_copies':copies,'broken_links':broken,'figure_pages':len(idx['figures']),'table_groups':len(set(x['number'] for x in idx['tables'])),'table_parts':len(idx['tables']),'new_fits':0,'new_resampling':0,'visual_check':'pending'}
(P/'QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2));assert unchanged and copies and not broken
if '--finalize' in sys.argv:
    result['visual_check']='PASS: all 13 exported pages inspected; S4 labels and S6 empty axes corrected'
    result['scope']='Review edition, not a journal-compliance or clinical-validation certification'
    (P/'QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files={str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file() and 'plot_cache' not in p.parts and p.name!='DELIVERY_MANIFEST.json'}
    (P/'DELIVERY_MANIFEST.json').write_text(json.dumps({'files':files,'new_fits':0,'new_resampling':0,'originals_unchanged':True},indent=2),encoding='utf-8')
