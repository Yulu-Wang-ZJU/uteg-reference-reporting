from pathlib import Path
import zipfile,io,json,shutil,posixpath
from lxml import etree as E
import win32com.client
W=Path(__file__).parent;F=Path(__file__).resolve().parent/'project/outputs/uteg_nc_goal_20261030/manuscript/figures/newB_20261103'
p=F/'Figure1_workflow_ma.pptx';backup=W/'before_figures'/p.name
backup.parent.mkdir(exist_ok=True,parents=True)
backup.parent.mkdir(exist_ok=True,parents=True)
if not backup.exists():shutil.copy2(p,backup)
with zipfile.ZipFile(p) as z:payload={n:z.read(n) for n in z.namelist()}
rows=json.loads((Path(__file__).resolve().parent/'results/summary/Fig4_HGB_capability_summary.json').read_text())
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for j,(cond,level,key) in enumerate([('fixed_reporting_rate',.4,'accuracy'),('matched_observed_accuracy',.9,'reporting_rate')],1):
 rr=sorted([r for r in rows if r['budget']==60 and r['arm']=='HZ+HB_to_SA' and r['condition']==cond and r['level']==level],key=lambda r:r['time']);v=[100*r[key]['mean'] for r in rr]
 path=f'ppt/charts/chart{j}.xml';root=E.fromstring(payload[path])
 for n,value in zip(root.findall('.//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v',ns),v):n.text=str(value)
 for n in root.findall('.//c:ser/c:tx/c:v',ns):n.text=key
 for n in root.findall('.//c:valAx/c:scaling/c:min',ns):n.set('val','0' if j==2 else '80')
 payload[path]=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
 rel=E.fromstring(payload[f'ppt/charts/_rels/chart{j}.xml.rels'])
 target=next(n.get('Target') for n in rel if n.get('Type','').endswith('/package'));emb=posixpath.normpath('ppt/charts/'+target)
 with zipfile.ZipFile(io.BytesIO(payload[emb])) as z:book={n:z.read(n) for n in z.namelist()}
 sheet='xl/worksheets/sheet1.xml';r=E.fromstring(book[sheet]);sn={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 for i,value in enumerate(v,2):
  cell=r.find(f'.//s:c[@r="B{i}"]/s:v',sn);assert cell is not None;cell.text=str(value)
 book[sheet]=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True);b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for n,data in book.items():z.writestr(n,data)
 payload[emb]=b.getvalue()
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in payload.items():z.writestr(n,data)
pdf=F/'Figure1_workflow_ma.pdf'
if not (W/'before_figures'/pdf.name).exists():shutil.copy2(pdf,W/'before_figures'/pdf.name)
app=win32com.client.Dispatch('PowerPoint.Application');pres=app.Presentations.Open(str(p),ReadOnly=True,Untitled=False,WithWindow=False)
try:pres.SaveAs(str(pdf),32)
finally:pres.Close()
print('Updated editable Figure 1 native charts and PDF')
