"""Redraw frozen Figure 3 statistics; no prediction or interval recomputation."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
M=Path(__file__).resolve().parents[1]
S=M.parent/'uteg_results_figures_20261026/source_data/reporting_statistics.json'
O=M/'figures/reporting_redesign_20261029'
O.mkdir(exist_ok=True)
data=json.loads(S.read_text(encoding='utf-8-sig'))
font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'text.color':'#252525',
 'axes.labelcolor':'#252525','xtick.color':'#454545','ytick.color':'#454545'})
fig=plt.figure(figsize=(183/25.4,99/25.4))
a=fig.add_axes([.085,.22,.36,.59]);b=fig.add_axes([.64,.22,.335,.59])
colors=['#387CA5','#258777'];dirs=['HZ_to_SA','HZ+HB_to_SA']
rows=[]
for j,d in enumerate(dirs):
    old=data['baseline'][d];new=data['current'][d];col=colors[j]
    a.plot([100*old['coverage'],100*new['coverage']],[100*old['agreement'],100*new['agreement']],color=col,lw=1.1,alpha=.65,zorder=1)
    for method,s in [('Error-based gate',old),('Two-sided rule',new)]:
        filled=method=='Two-sided rule';marker='D' if filled else 'o'
        x,y=100*s['coverage'],100*s['agreement']
        xi=np.array(s['coverage_interval'])*100;yi=np.array(s['agreement_interval'])*100
        a.errorbar(x,y,xerr=[[x-xi[0]],[xi[1]-x]],yerr=[[y-yi[0]],[yi[1]-y]],fmt=marker,
          mfc=col if filled else 'white',mec=col,ecolor=col,elinewidth=.85,capsize=2,ms=5,zorder=3)
        rows.append({'direction':d,'rule':method,**s})
    for t,(tail,yy) in enumerate([('low',3.4-j*2.6),('high',2.3-j*2.6)]):
        # Slight vertical separation preserves the two marginal intervals.
        v0,v1=old[tail+'_risk']*100,new[tail+'_risk']*100
        b.plot([v0,v1],[yy+.09,yy-.09],color=col,lw=1.1,alpha=.65,zorder=1)
        for filled,s,y in [(False,old,yy+.09),(True,new,yy-.09)]:
            v=s[tail+'_risk']*100;lo,hi=np.array(s[tail+'_interval'])*100
            b.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='D' if filled else 'o',mfc=col if filled else 'white',
                mec=col,ecolor=col,elinewidth=.9,capsize=2,ms=5,zorder=3)
            b.annotate(f'{v:.1f}',(v,y),xytext=(0,9 if not filled else -13),textcoords='offset points',
                ha='center',va='center',fontsize=8,color=col)
a.set(xlim=(0,65),ylim=(78,100),xticks=[0,20,40,60],yticks=[80,85,90,95,100],xlabel='Report coverage (%)',ylabel='Agreement among reports (%)')
a.grid(axis='y',color='#EBEEF0',lw=.5);a.set_axisbelow(True)
b.set(xlim=(-2,80),ylim=(-.95,4.2),xticks=[0,20,40,60,80],yticks=[3.4,2.3,.8,-.3],
      yticklabels=['Low MA','High MA','Low MA','High MA'],xlabel='Abnormal released as within range (%)')
b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0,pad=7)
b.axhline(1.72,color='#DCE1E4',lw=.7)
for y,label,c in [(4.05,'HZ → SA',colors[0]),(1.35,'HZ + HB → SA',colors[1])]:
    b.text(0,y,label,color=c,fontsize=9,fontweight='bold',va='bottom')
fig.text(.022,.91,'a',weight='bold',fontsize=12);fig.text(.085,.91,'Reporting trade-off',fontsize=11)
fig.text(.565,.91,'b',weight='bold',fontsize=12);fig.text(.64,.91,'Abnormal-to-within reports',fontsize=11)
direction_handles=[Line2D([],[],color=c,lw=2,label=l) for c,l in zip(colors,['HZ → SA','HZ + HB → SA'])]
a.legend(handles=direction_handles,loc='lower left',frameon=False,fontsize=8,handlelength=1.6)
handles=[Line2D([],[],ls='',marker='o',mfc='white',mec='#454545',ms=5,label='Error-based gate'),
         Line2D([],[],ls='',marker='D',mfc='#454545',mec='#454545',ms=5,label='Two-sided rule')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.025),ncol=2,frameon=False,fontsize=9,columnspacing=3)
fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[]
for t in fig.findobj(matplotlib.text.Text):
    if t.get_visible() and t.get_text():
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:bad.append(t.get_text())
assert not bad,bad
for ext in ['png','pdf','svg']:fig.savefig(O/f'Figure3_reporting.{ext}',dpi=400)
(O/'source_data.json').write_text(json.dumps(rows,indent=2))
(O/'QA.json').write_text(json.dumps({'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),
 'data_recomputed':False,'intervals_unchanged':True,'off_canvas_text':bad,'size_mm':[183,99],
 'font':'Times New Roman','connections':'discrete paired working points; not threshold curves'},indent=2))
