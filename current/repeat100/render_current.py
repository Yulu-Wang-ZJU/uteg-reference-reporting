from pathlib import Path
import json, shutil, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).parent
WS=Path(__file__).resolve().parent/'project'
M=WS/'outputs/uteg_nc_goal_20261030/manuscript'
F=M/'figures/newB_20261103'
R=W/'results'
import zipfile
with zipfile.ZipFile(W/'aggregate_results.zip') as _z: _z.extractall(R)
def load(n):return json.loads((R/'summary'/f'{n}_summary.json').read_text())
def choose(rows,**kw):return [x for x in rows if all(x.get(k)==v for k,v in kw.items())]
def one(rows,**kw):
 a=choose(rows,**kw);assert len(a)==1,(kw,len(a));return a[0]
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
F.mkdir(exist_ok=True,parents=True)
(W/'before_figures').mkdir(exist_ok=True)
def save(fig,n):
 for ext in ['pdf','png','svg']:
  p=F/f'{n}.{ext}'
  if p.exists() and not (W/'before_figures'/p.name).exists():shutil.copy2(p,W/'before_figures'/p.name)
  fig.savefig(p,dpi=250,bbox_inches='tight')
 plt.close(fig)
arms=['HZ_to_SA','HZ+HB_to_SA'];titles=['Single source','Dual source']
colors={'All report':'#666666','Boundary':'#396F9D','Logistic':'#2A8D7A','HGB':'#BA7D28','common_B':'#9B528F'}
labels={'All report':'All','Boundary':'Boundary','Logistic':'Logistic','HGB':'HGB','common_B':'Class-specific\nreporting gate'}
rules=list(colors)
# Figure 3: all quantities summarize the same 100 repeats, budget=60.
cur=load('Fig3_score_curves');ops=load('Table2_and_time');disp=load('Fig3_dispositions')
dr=json.loads((R/'summary/Fig3_dispositions_rounds.json').read_text())
fig=plt.figure(figsize=(7.2,7.0));gs=fig.add_gridspec(3,2,height_ratios=[1.5,1,1],hspace=.65,wspace=.30)
for col,arm in enumerate(arms):
 ax=fig.add_subplot(gs[0,col])
 for rule in ['Boundary','Logistic','HGB']:
  rows=sorted(choose(cur,budget=60,arm=arm,rule=rule,time=900),key=lambda x:x['nominal_rate'])
  x=np.array([100*r['reporting_rate']['mean'] for r in rows]);y=np.array([100*r['accuracy']['mean'] for r in rows]);sd=np.array([100*r['accuracy']['sd'] for r in rows])
  ax.plot(x,y,label=labels[rule],color=colors[rule]);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(100,y+sd),color=colors[rule],alpha=.08)
 for rule in rules:
  r=one(ops,budget=60,arm=arm,time=900,rule=rule)
  ax.scatter(100*r['reporting_rate']['mean'],100*r['accuracy']['mean'],color=colors[rule],edgecolors='white',s=35,marker='D' if rule=='common_B' else 'o',zorder=4)
 ax.set(xlim=(45,102),ylim=(70,102),xlabel='Reporting rate (%)',ylabel='Accuracy (%)',title=titles[col]);ax.text(-.18,1.08,'AB'[col],transform=ax.transAxes,fontweight='bold',fontsize=11)
 if col==0:ax.legend(fontsize=7,loc='lower left')
 else:ax.text(.03,.05,'Points: Table 2 operating points\nDiamond: Class-specific reporting gate',transform=ax.transAxes,fontsize=7)
 for sub,key in enumerate(['FP_withheld','correct_released']):
  ax=fig.add_subplot(gs[sub+1,col]);ys=[];es=[]
  for rule in rules:
   if sub==0:r=one(disp,budget=60,arm=arm,rule=rule);ys.append(r[key]['mean']);es.append(r[key]['sd'])
   else:
    rr=choose(dr,budget=60,arm=arm,rule=rule);v=np.array([100*r['correct_released']/r['before_correct'] for r in rr]);ys.append(v.mean());es.append(v.std(ddof=1))
  if sub==0:
   for offset,metric,hatch in [(-.19,'FP_withheld',None),(.19,'FN_withheld','///')]:
    vv=[one(disp,budget=60,arm=arm,rule=r)[metric] for r in rules]
    xx=np.arange(5)+offset; yy=[r['mean'] for r in vv]; ee=[r['sd'] for r in vv]
    ax.bar(xx,yy,color=[colors[k] for k in rules],width=.36,hatch=hatch,edgecolor='white',label=metric[:2])
    ax.errorbar(xx,yy,yerr=ee,fmt='none',ecolor='#333333',capsize=1,lw=.6)
    for i,v in enumerate(yy):ax.text(xx[i],v+ee[i]+.8,f'{v:.1f}',ha='center',fontsize=6)
   ax.legend(fontsize=6,loc='upper right')
  else:
   ax.bar(range(5),ys,color=[colors[k] for k in rules],width=.68);ax.errorbar(range(5),ys,yerr=es,fmt='none',ecolor='#333333',capsize=2,lw=.7)
   for i,v in enumerate(ys):ax.text(i,v+es[i]+1.1,f'{v:.1f}',ha='center',fontsize=7)
  ticklabels=[labels[k]+('\n'+f"{100*one(ops,budget=60,arm=arm,time=900,rule=k)['reporting_rate']['mean']:.1f}%" if sub==1 else '') for k in rules]
  ax.set(xticks=range(5),xticklabels=ticklabels,ylabel='Incorrect predictions withheld (count)' if sub==0 else 'Correct predictions retained (%)',ylim=(0,38) if sub==0 else (0,112))
  ax.tick_params(axis='x',labelsize=7,rotation=25)
  if sub==0:ax.text(-.25,1.08,'CD'[col],transform=ax.transAxes,fontweight='bold',fontsize=11)
fig.subplots_adjust(left=.12,right=.99,bottom=.08,top=.96);save(fig,'Figure3_reporting')
# Figure 4: retain HGB capability comparison, never substitute the composite endpoint.
cap=load('Fig4_HGB_capability');fig,axes=plt.subplots(2,2,figsize=(7.2,5.5));fig.subplots_adjust(hspace=.55,wspace=.32,bottom=.14)
for j,(cond,metric,levels) in enumerate([('fixed_reporting_rate','accuracy',[.1,.2,.4]),('matched_observed_accuracy','reporting_rate',[.85,.9,.95]),('fixed_reporting_rate','FNR',[.1,.2,.4]),('matched_observed_accuracy','FNR',[.85,.9,.95])]):
 ax=axes.flat[j]
 for k,l in enumerate(levels):
  rows=sorted(choose(cap,budget=60,arm=arms[1],condition=cond,level=l),key=lambda r:r['time'])
  assert len(rows)==3,(cond,l)
  y=np.array([100*r[metric]['mean'] if r.get(metric,{}).get('n',0) else np.nan for r in rows]);sd=np.array([100*r[metric]['sd'] if r.get(metric,{}).get('n',0) else np.nan for r in rows])
  ax.errorbar([5,10,15],y,yerr=sd,color=['#3978A8','#B17636','#348976'][k],marker=['o','s','^'][k],capsize=2,lw=1.3,label=f'{100*l:g}% '+('reporting rate' if cond=='fixed_reporting_rate' else 'accuracy'))
 ax.set(xticks=[5,10,15],xlim=(4,16),ylim=(74,103) if j==0 else (-22,115),xlabel='Observation time (min)',ylabel={'accuracy':'Accuracy (%)','reporting_rate':'Reporting rate (%)','FNR':'FNR (%)'}[metric]);ax.legend(fontsize=6.7,loc='best');ax.text(-.18,1.07,'ABCD'[j],transform=ax.transAxes,fontweight='bold',fontsize=11)
save(fig,'Figure4_main_pair')
# Small figure-1 capability insets, ready for the editable master.
for metric,cond,level,name in [('accuracy','fixed_reporting_rate',.4,'Figure1_accuracy_inset'),('reporting_rate','matched_observed_accuracy',.9,'Figure1_rate_inset')]:
 rows=sorted(choose(cap,budget=60,arm=arms[1],condition=cond,level=level),key=lambda r:r['time']);fig,ax=plt.subplots(figsize=(2.1,1.25));ax.errorbar([5,10,15],[100*r[metric]['mean'] for r in rows],yerr=[100*r[metric]['sd'] for r in rows],color='#3978A8',marker='o',capsize=2,lw=1);ax.set(xticks=[5,10,15],ylim=(0,105),xlabel='Time (min)',ylabel='Accuracy (%)' if metric=='accuracy' else 'Reporting rate (%)');ax.tick_params(labelsize=7);save(fig,name)
# Nine-configuration outcome summary.
nine=load('nine_configurations');rr=[x for x in nine if x['mode']=='full_local' and x['rule']=='common_B' and not x['arm'].startswith('target')]
fig,axs=plt.subplots(1,3,figsize=(9,4),sharey=True)
for ax,key in zip(axs,['accuracy','reporting_rate','FNR']):
 ax.errorbar([100*r[key]['mean'] for r in rr],range(len(rr)),xerr=[100*r[key]['sd'] for r in rr],fmt='o',capsize=3);ax.set(yticks=range(len(rr)),yticklabels=[r['arm'].replace('_to_',' → ') for r in rr],xlabel=key.replace('_',' ')+' (%)',xlim=(-2,105));ax.invert_yaxis()
fig.tight_layout();save(fig,'S2_nine_direction_report')
# Stability and label budgets.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
rounds=json.loads((R/'summary/Table2_and_time_rounds.json').read_text())
for ax,b in zip(axs,[54,60]):
 for arm,title,c in zip(arms,titles,['#3978A8','#B17636']):
  rr=choose(rounds,budget=b,arm=arm,time=900,rule='common_B');ax.scatter([100*r['test']['reporting_rate'] for r in rr],[100*r['test']['accuracy'] for r in rr],s=12,alpha=.6,label=title,color=c)
 ax.set(xlabel='Reporting rate (%)',ylabel='Accuracy (%)',title='90% development sampling' if b==54 else 'Full development pool');ax.legend(fontsize=7)
fig.tight_layout();save(fig,'S9_seed_stability')
lb=load('label_budgets');fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),sharey=True)
for ax,arm,title in zip(axs,arms,titles):
 for a,l,c in [(arm,title,'#3978A8'),('target_only_Ridge','Target-only Ridge','#B17636'),('target_only_SA','Target-only RBF','#348976')]:
  rows=sorted(choose(lb,arm=a),key=lambda r:r['budget']);ax.errorbar([r['budget'] for r in rows],[r['MAE']['mean'] for r in rows],yerr=[r['MAE']['sd'] for r in rows],marker='o',capsize=3,label=l,color=c)
 ax.set(xlabel='Local development records',ylabel='MAE (mm)',xticks=[20,40,60],title=title);ax.legend(fontsize=7)
fig.tight_layout();save(fig,'S12_label_budget');shutil.copy2(F/'S12_label_budget.pdf',F/'S8_adaptation_budget.pdf')
print('Updated Figure 3, Figure 4, inset assets and S2/S8/S9/S12')
