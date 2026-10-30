from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
M=Path(__file__).resolve().parents[1];S=M.parent/'uteg_current_ma_sequence_20261026'
O=M/'figures/main_pair_20261029';O.mkdir(exist_ok=True)
assert json.loads((S/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
results=json.loads((S/'RESULTS.json').read_text());panels=[];hashes={};ref=None
for j,d in enumerate(['HZ_to_SA','HZ+HB_to_SA']):
 p=S/'actions'/f'{d}_first_report.npz';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(p)
 if ref is None:ref={k:z[k] for k in ['ids','groups','truth']}
 else:
  for k in ref:assert np.array_equal(ref[k],z[k])
 assert z['reported'].shape==(20,201)
 good=np.stack([z['reported']&(z['first_time']<=t)&(z['first_class']==z['truth'][None,:]) for t in [300,600,900]])
 bad=np.stack([z['reported']&(z['first_time']<=t)&(z['first_class']!=z['truth'][None,:]) for t in [300,600,900]])
 for k,t in enumerate([300,600,900]):
  r=results[d]['cumulative'][str(t)];assert good[k].sum()==r['correct'] and bad[k].sum()==r['wrong']
 assert np.all(np.diff(good.sum((1,2)))>=0) and np.all(np.diff(bad.sum((1,2)))>=0)
 groups=np.unique(z['groups']);assert len(groups)==190
 gc=np.stack([good[:,:,z['groups']==g].sum((1,2)) for g in groups],axis=1)
 bc=np.stack([bad[:,:,z['groups']==g].sum((1,2)) for g in groups],axis=1)
 n=np.array([(z['groups']==g).sum()*20 for g in groups])
 rng=np.random.default_rng(20261027+j);ii=rng.integers(0,190,(3000,190));den=n[ii].sum(1)
 boot=np.stack([gc[:,ii].sum(2)/den,bc[:,ii].sum(2)/den],axis=1)*100
 # boot shape: time, outcome, bootstrap replicate.
 bands=np.quantile(boot,[.025,.975],axis=2);delta=np.quantile(boot[2]-boot[0],[.025,.975],axis=1)
 v=dict(direction=d,correct=good.sum((1,2)).tolist(),incorrect=bad.sum((1,2)).tolist(),bands=bands.tolist(),delta=delta.tolist(),N=4020)
 v['low_within']=int((z['reported']&(z['first_class']==0)&(z['truth'][None,:]==-1)).sum())
 v['high_within']=int((z['reported']&(z['first_class']==0)&(z['truth'][None,:]==1)).sum());panels.append(v)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
fig=plt.figure(figsize=(183/25.4,150/25.4));gs=fig.add_gridspec(3,2,left=.10,right=.97,bottom=.09,top=.85,wspace=.34,hspace=.75,height_ratios=[1.3,.85,.48])
C,W,G='#258777','#BC6358','#E2E6E9'
fig.legend(handles=[Line2D([],[],color=C,marker='o',label='Correct'),Line2D([],[],color=W,marker='s',label='Incorrect'),Patch(facecolor=G,label='Not reported')],loc='upper center',bbox_to_anchor=(.54,.99),ncol=3,frameon=False)
for j,v in enumerate(panels):
 fig.text(.10 if j==0 else .60,.90,['Huzhou → SAHZU','Huzhou + Hubei → SAHZU'][j],fontsize=10,weight='bold')
 ax=fig.add_subplot(gs[0,j]);ax.text(-.17,1.04,chr(97+j),transform=ax.transAxes,weight='bold',fontsize=11)
 for k,(key,col,m) in enumerate([('correct',C,'o'),('incorrect',W,'s')]):
  y=np.array(v[key])*100/4020;band=np.array(v['bands'])
  ax.plot([5,10,15],y,color=col,marker=m,ms=4,lw=1.3)
  ax.fill_between([5,10,15],band[0,:,k],band[1,:,k],color=col,alpha=.12,lw=0)
  ax.annotate(str(v[key][-1]),(15,y[-1]),xytext=(5,0),textcoords='offset points',color=col,va='center')
 ax.set(xlim=(4.5,17),ylim=(0,30),xticks=[5,10,15],yticks=[0,10,20,30],xlabel='Observation (min)',ylabel='Cumulative first reports (%)')
 ax=fig.add_subplot(gs[1,j]);ax.text(-.17,1.12,chr(99+j),transform=ax.transAxes,weight='bold',fontsize=11);ax.set_title('Additional reports: 15 vs 5 min',loc='left')
 for k,(key,col,m) in enumerate([('correct',C,'o'),('incorrect',W,'s')]):
  val=(v[key][-1]-v[key][0])*100/4020;lo,hi=np.array(v['delta'])[:,k]
  ax.errorbar(val,k,xerr=[[val-lo],[hi-val]],fmt=m,color=col,capsize=3,ms=4)
  ax.text(val,k+.27,f'+{val:.2f}',ha='center',color=col)
 ax.set(xlim=(0,25),ylim=(1.65,-.65),yticks=[0,1],yticklabels=['Correct','Incorrect'],xticks=[0,10,20],xlabel='Change (percentage points)');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
 ax=fig.add_subplot(gs[2,j]);ax.text(-.17,1.13,chr(101+j),transform=ax.transAxes,weight='bold',fontsize=11);ax.set_title('Outcomes at 15 min',loc='left')
 vals=np.array([v['correct'][-1],v['incorrect'][-1],4020-v['correct'][-1]-v['incorrect'][-1]])/4020*100;assert np.isclose(vals.sum(),100)
 left=0
 for val,col in zip(vals,[C,W,G]):ax.barh(0,val,left=left,height=.5,color=col,edgecolor='white',lw=.4);left+=val
 ax.text(100-vals[2]/2,0,f'{vals[2]:.2f}% unreported',ha='center',va='center',fontsize=8)
 ax.set(xlim=(0,100),ylim=(-.5,.5),yticks=[],xticks=[0,50,100],xlabel='All record–seed events (%)');ax.spines['left'].set_visible(False)
fig.canvas.draw();ren=fig.canvas.get_renderer();outside=[]
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
  box=t.get_window_extent(ren)
  if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:outside.append(t.get_text())
assert not outside,outside
for ext in ['pdf','svg','png']:fig.savefig(O/f'Figure4_main_pair.{ext}',dpi=350)
(O/'SOURCE_DATA.json').write_text(json.dumps(panels,indent=2))
(O/'QA.json').write_text(json.dumps(dict(source_hashes=hashes,same_ids_groups_truth=True,records=201,groups=190,seeds=20,first_report_locked=True,counts_match_existing_results=True,new_fits=0,off_canvas_text=outside,bootstrap=3000),indent=2))
print(json.dumps(panels,indent=2))
