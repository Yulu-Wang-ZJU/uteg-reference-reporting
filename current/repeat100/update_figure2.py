from pathlib import Path
import types,json,shutil
import numpy as np
import matplotlib.pyplot as plt
W=Path(__file__).parent;WS=Path(__file__).resolve().parent/'project';ROOT=WS/'outputs/current_ma_20261103';F=WS/'outputs/uteg_nc_goal_20261030/manuscript/figures/newB_20261103';R=Path(__file__).resolve().parent/'experiments/uteg_sahzu_locked_completion_20261118/reference244'
source=Path(__file__).resolve().parent.parent/'code_submit/paper_inward4_replay/manuscript_figure2.py'
text=source.read_text(encoding='utf8').replace('Agreement','Accuracy').replace('U-TEG B (V)','U-TEG MA (V)').replace('Reference MA (mm)','TEG MA (mm)')
text=text.replace("for x,j,title in [(.085,0,'Target-only'),(.565,1,'Single-source and dual-source')]:", "for x,j,title in [(.325,0,'Target-only, single-source and dual-source')]:")
text=text.replace("    fig.text(.5,.048,'Predicted state',ha='center',fontsize=6.5)", '')
text=text.replace("        b.yaxis.label.set_fontsize(6.4)", "        b.yaxis.label.set_fontsize(6.4)\n        b.set_xlabel('Predicted state', fontsize=7, labelpad=2)")
mod=types.ModuleType('figure2_current');mod.__file__=str(source);exec(compile(text,str(source),'exec'),mod.__dict__)
mod.ROOT=ROOT;mod.WS=WS;mod.REVIEW=ROOT/'figure2_mapping_20261103';mod.OUT=F

mod.STYLE.update({'font.family':'Times New Roman','mathtext.fontset':'custom',
                  'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
                  'mathtext.bf':'Times New Roman:bold','mathtext.fallback':None})
_original_save=mod.save
def save_with_readable_type(fig,name):
 from matplotlib.text import Text
 for item in fig.findobj(match=Text):
  item.set_fontfamily('Times New Roman')
  if item.get_text():item.set_fontsize(max(6.5,item.get_fontsize()))
 return _original_save(fig,name)
mod.save=save_with_readable_type

d=mod.load(ROOT/'data/DATA15_CURRENT.npz');signals=mod.load(ROOT/'data/CURRENT_SCALED_CURVES.npz');teg=mod.paired_signals(d)
zz=[mod.load(R/(a+'.npz')) for a in mod.ARMS];rows=zz[0]['test'];y=zz[0]['truth_four'][:,2]
for z in zz:np.testing.assert_array_equal(z['test'],rows);np.testing.assert_array_equal(z['truth_four'][:,2],y)
pp=[z['prediction'][None,:] for z in zz];ms=[mod.metrics(y,p) for p in pp]
def record_metrics(y,pp):
 mm=[mod.metrics(y,p[0]) for p in pp];assert mm[0]['confusion']==mm[1]['confusion']==mm[2]['confusion'];return mm
mod.record_metrics=record_metrics
sel=R/'EXAMPLE_SELECTION.json';j=json.loads(sel.read_text());examples=np.asarray(j['rows'],dtype=int)
assert len(examples)==10 and set(examples)<=set(rows)
readouts=np.stack([z['prediction_four'] for z in zz]);indices=np.array([list(rows).index(int(i)) for i in examples]);t=np.arange(0,2701,5)/60;curves,_=mod.reconstruct_teg(readouts[:,indices,:],t)
B=d['x'][rows,1];coef=np.linalg.lstsq(np.c_[np.ones(len(B)),B],y,rcond=None)[0];fit=coef[0]+coef[1]*B;r2=1-((y-fit)**2).sum()/((y-y.mean())**2).sum()
(W/'before_figures').mkdir(exist_ok=True,parents=True)
for ext in ['pdf','svg','png']:
 p=F/f'Figure2_translation.{ext}'
 if p.exists() and not (W/'before_figures'/p.name).exists():shutil.copy2(p,W/'before_figures'/p.name)
with plt.rc_context(mod.STYLE):mod.main_figure(d,rows,y,pp,ms,teg,examples,B,coef,r2,t,curves,signals)
(W/'FIGURE2_REFERENCE_CHECK.json').write_text(json.dumps({'records':244,'metrics':dict(zip(mod.ARMS,ms)),'descriptive_R2':float(r2),'reference_range_selection':True,'prediction_source':'reference244, first prespecified seed 1000; full source and local development pools','models_refitted':False},indent=2))
print('Figure 2: fixed 244-record reference predictions and all four readouts updated')
