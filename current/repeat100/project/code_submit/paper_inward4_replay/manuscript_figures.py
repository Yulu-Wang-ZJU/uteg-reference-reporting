"""Vector figures, all derived from current experiment artifacts."""
import json,string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import core as c,current_suite as suite,analysis,gates

ROOT=suite.ROOT;SYNC=ROOT/'manuscript_sync';OUT=SYNC/'deliverables/figures/newB_20261103'
COLORS=['#4477AA','#CC8866','#338877','#9977AA','#AA4455','#999933']
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
E=c.read(SYNC/'EVIDENCE.json');ARMS=[a for a in c.TARGET if not a.startswith('target_only')]
def short(a):return a.replace('target_only_SA','Target-only').replace('_to_',' → ').replace('SA','SA')
def panel(ax,label,title):ax.set_title(title,fontsize=8);ax.text(-.16,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=10)
def save(fig,name):
    fig.tight_layout(pad=1.3)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig);print('FIGURE_COMPLETE',name,flush=True)
def errbar(ax,x,v,color,label=None,marker='o',pct=1):
    y=v['estimate']
    if y is None:return
    ax.errorbar(x,pct*y,yerr=[[pct*(y-v['low'])],[pct*(v['high']-y)]],fmt=marker,color=color,capsize=2,ms=4,label=label)

def prediction():
    from .manuscript_figure2 import run
    run()

def historical_translation_grid():
    """Retained historical plotting recipe; no longer used as main Figure 2."""
    d,*_=c.context();fig,axes=plt.subplots(3,2,figsize=(6.8,7.8));dual=['SA+HB_to_HZ','SA+HZ_to_HB','HZ+HB_to_SA']
    for j,(name,unit) in enumerate(zip(['R','K','MA','angle'],['min','min','mm','degrees'])):
        ax=axes.flat[j];x=np.arange(3)
        for k,phase in enumerate(['raw','complete']):
            vals=[E['primary'][a][phase][name] for a in dual];ax.bar(x+(k-.5)*.32,[v['mae'] for v in vals],.30,color=COLORS[k],label=['Source only','Locally corrected'][k])
            for xx,v in zip(x+(k-.5)*.32,vals):errbar(ax,xx,v['interval'],'#333333',marker='none')
        ax.set_xticks(x,[short(a) for a in dual],rotation=15);ax.set_ylabel(f'{name} MAE ({unit})');panel(ax,string.ascii_lowercase[j],f'{name} prediction')
        if j==0:ax.legend(fontsize=7)
    for j,arm in enumerate(c.MAIN[1:],4):
        ax=axes.flat[j];z=analysis.load(SYNC/'experiments/auxiliary'/f'{arm}_B0.npz');y=d['y'][z['test'],2]
        for k,phase in enumerate(['raw','complete']):ax.scatter(y,np.nanmean(z[phase][:,:,2],0)-y,s=8,alpha=.45,color=COLORS[k],label=phase)
        ax.axhline(0,color='black',lw=.6);ax.axvspan(50,70,color='#dddddd',alpha=.3);ax.set(xlabel='Reference MA (mm)',ylabel='Predicted − reference (mm)');panel(ax,string.ascii_lowercase[j],short(arm))
    save(fig,'Figure2_translation')

def selection():
    from .manuscript_figure3 import run
    run()

def historical_selection():
    fig,axes=plt.subplots(1,2,figsize=(6.8,4.0));markers={'ALL_REPORT':'s','BOUNDARY95':'o','LR':'^','TWO_SIDED':'D'}
    for j,arm in enumerate(c.MAIN[1:]):
        for rule,marker in markers.items():
            m=E['reporting'][arm][rule];x=100*m['coverage'];v=m['intervals']['agreement'];errbar(axes[0],x,v,COLORS[j],label=short(arm)+' '+rule,marker=marker,pct=100)
        for k,rule in enumerate(['ALL_REPORT','BOUNDARY95','LR','TWO_SIDED']):
            m=E['reporting'][arm][rule];cm=np.array(m['confusion']);den=np.array(m['all_true_class_counts']);total=den[0]+den[2]
            values=[cm[0,0]+cm[2,2],cm[0,1]+cm[2,1],cm[0,2]+cm[2,0],m['withheld_by_true_state'][0]+m['withheld_by_true_state'][2]]
            bottom=0
            for val,color,label in zip(values,[COLORS[2],COLORS[1],COLORS[4],'#DDDDDD'],['Correct abnormal','As normal','Opposite abnormal','Unreported']):
                axes[1].bar(j*5+k,100*val/total,bottom=bottom,color=color,label=label if j==0 and k==0 else None);bottom+=100*val/total
    axes[0].set(xlabel='Report coverage (%)',ylabel='Agreement among reports (%)',xlim=(0,103),ylim=(40,102));axes[0].legend(fontsize=5.7,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
    axes[1].set_xticks([0,1,2,3,5,6,7,8],['All','B95','LR','Dual']*2,rotation=35);axes[1].set_ylabel('Outcomes of reference-abnormal events (%)');axes[1].set_xlabel('Single source                    Dual source');axes[1].set_ylim(0,125);axes[1].legend(fontsize=5.8,ncol=2,loc='upper center')
    panel(axes[0],'a','Frozen reporting rules');panel(axes[1],'b','Full abnormal-class denominator');save(fig,'Figure3_reporting')

def observation():
    from .manuscript_figure4 import run
    run()

def historical_observation():
    fig,axs=plt.subplots(3,2,figsize=(6.8,7.5))
    for col,arm in enumerate(c.MAIN[1:]):
        for k,key in enumerate(['correct_yield','incorrect_yield']):
            vals=[E['cumulative'][arm][str(h)]['intervals'][key] for h in c.HORIZONS];ax=axs[0,col]
            ax.plot([5,10,15],[v['estimate']*100 for v in vals],color=COLORS[k],marker='o',ms=3,label=['Correct','Incorrect'][k]);ax.fill_between([5,10,15],[v['low']*100 for v in vals],[v['high']*100 for v in vals],color=COLORS[k],alpha=.15)
        axs[0,col].set(xlabel='Observation checkpoint (min)',ylabel='Cumulative first-report yield (%)',xticks=[5,10,15],ylim=(0,100));axs[0,col].legend(fontsize=7);panel(axs[0,col],string.ascii_lowercase[col],short(arm))
        vals=E['sequence_contrasts'][arm]['continued']
        for k,key in enumerate(['correct','incorrect']):errbar(axs[1,col],k,vals[key],COLORS[k],pct=100)
        axs[1,col].set(xticks=[0,1],xticklabels=['Correct','Incorrect'],ylabel='Added after 5 min (percentage points)',ylim=(-.3,9));axs[1,col].axhline(0,color='#888888',lw=.5);panel(axs[1,col],string.ascii_lowercase[2+col],'Continue from 5 to 15 min')
        for i,schedule in enumerate(['5-10-15','10-15','fixed15']):
            m=E['sequence'][arm]['BOUNDARY95'][schedule];bottom=0
            for color,key in zip([COLORS[0],COLORS[1],'#DDDDDD'],['correct','incorrect','withheld']):
                height=100*m[key]/m['events'];axs[2,col].bar(i,height,bottom=bottom,color=color,width=.65,label=key.capitalize() if i==0 else None);bottom+=height
        axs[2,col].set(xticks=[0,1,2],xticklabels=['5–10–15','10–15','Fixed 15'],ylabel='Final outcome (%)',ylim=(0,112));axs[2,col].legend(fontsize=6,ncol=3,loc='upper center');panel(axs[2,col],string.ascii_lowercase[4+col],'First reports remain locked')
    save(fig,'Figure4_main_pair')

def supplement():
    d,*_=c.context()
    fig,ax=plt.subplots(1,2,figsize=(6.8,3));centers=E['cohort']
    for i,cl in enumerate(['low','normal','high']):ax[0].bar(np.arange(3),[r[cl] for r in centers],bottom=[sum(r[k] for k in ['low','normal','high'][:i]) for r in centers],color=COLORS[i],label=cl)
    ax[0].set_xticks(range(3),[r['center'] for r in centers]);ax[0].set_ylabel('Paired records');ax[0].legend(fontsize=7)
    for i,r in enumerate(centers):ax[1].hist(d['y'][d['centers']==r['center'],2],bins=np.arange(0,101,5),histtype='step',color=COLORS[i],label=r['center'])
    ax[1].set(xlabel='Reference MA (mm)',ylabel='Records');ax[1].axvline(50,color='grey',ls=':');ax[1].axvline(70,color='grey',ls=':');ax[1].legend(fontsize=7);save(fig,'S1_cohort')
    fig,axs=plt.subplots(3,3,figsize=(6.8,6.8))
    for i,arm in enumerate(ARMS):
        ax=axs.flat[i]
        for j,rule in enumerate(['ALL_REPORT','BOUNDARY95','LR','HGB','BOUNDARY_PAPER']):
            m=E['reporting'][arm][rule]
            if m['agreement'] is not None:ax.plot(m['coverage']*100,m['agreement']*100,'o',color=COLORS[j],ms=4,label=rule)
        ax.set(xlim=(0,102),ylim=(0,102),xlabel='Coverage (%)',ylabel='Agreement (%)');panel(ax,string.ascii_lowercase[i],short(arm))
    axs.flat[0].legend(fontsize=5);save(fig,'S2_nine_direction_report')
    for page in range(3):
        fig,axs=plt.subplots(3,4,figsize=(6.8,7.5))
        for row,arm in enumerate(ARMS[page*3:page*3+3]):
            for col,key in enumerate(['R','K','MA','angle']):
                ax=axs[row,col];methods=suite.mapping.METHODS
                ax.bar(range(6),[E['mapping'][arm][m][key]['mae'] for m in methods],color=COLORS)
                ax.set_xticks(range(6),['Rid','KRR','PLS','SVR','MLP','TT'],rotation=60,fontsize=6);ax.set_ylabel(key+' MAE');panel(ax,string.ascii_lowercase[row*4+col],short(arm))
        save(fig,f'S3_mapping_{page+1}')
    fig,axs=plt.subplots(2,2,figsize=(6.8,5.5))
    for j,key in enumerate(['R','K','MA','angle']):
        ax=axs.flat[j]
        for i,arm in enumerate(ARMS):ax.plot([0,1],[E['primary'][arm]['raw'][key]['mae'],E['primary'][arm]['complete'][key]['mae']],'-o',ms=3,label=short(arm))
        ax.set(xticks=[0,1],xticklabels=['Source only','Locally corrected'],ylabel=key+' MAE');panel(ax,string.ascii_lowercase[j],key)
    axs[1,1].legend(fontsize=5.5);save(fig,'S4_paired_adaptation')
    fig,axs=plt.subplots(3,3,figsize=(6.8,6.8))
    for i,arm in enumerate(ARMS):
        z=analysis.load(SYNC/'experiments/auxiliary'/f'{arm}_B0.npz');y=d['y'][z['test'],2];p=np.nanmean(z['complete'][:,:,2],0);ok=np.isfinite(p);ax=axs.flat[i];ax.scatter((y[ok]+p[ok])/2,p[ok]-y[ok],s=8,alpha=.55,color=COLORS[0]);ax.axhline(0,color='grey',lw=.5);ax.set(xlabel='Mean MA (mm)',ylabel='Residual (mm)');panel(ax,string.ascii_lowercase[i],short(arm))
    save(fig,'S5_MA_residuals')
    for page in range(3):
        fig,axs=plt.subplots(1,3,figsize=(6.8,2.7))
        for i,arm in enumerate(ARMS[page*3:page*3+3]):
            ax=axs[i]
            for j,key in enumerate(['LR','HGB','BOUNDARY']):
                vals=[v for v in E['risk_curves'][arm] if v['rule']==key];xx=[v['coverage']*100 for v in vals];yy=[(v['agreement']['estimate'] or 0)*100 for v in vals];ax.plot(xx,yy,color=COLORS[j],label=key)
            ax.set(xlabel='Coverage (%)',ylabel='Agreement (%)',xlim=(0,100),ylim=(0,100));panel(ax,string.ascii_lowercase[i],short(arm));ax.legend(fontsize=6)
        save(fig,f'S6_risk_coverage_{page+1}')
    fig,axs=plt.subplots(3,3,figsize=(6.8,6.8))
    for i,arm in enumerate(ARMS):
        ax=axs.flat[i]
        for j,key in enumerate(['correct_yield','incorrect_yield']):ax.plot([5,10,15],[E['cumulative'][arm][str(h)][key]*100 for h in c.HORIZONS],'-o',ms=3,color=COLORS[j])
        ax.set(xlabel='Checkpoint (min)',ylabel='Full-cohort yield (%)',ylim=(0,100),xticks=[5,10,15]);panel(ax,string.ascii_lowercase[i],short(arm))
    save(fig,'S7_nine_direction_sequence')
    fig,axs=plt.subplots(3,3,figsize=(6.8,6.8))
    for i,arm in enumerate(ARMS):
        ax=axs.flat[i];bs=sorted(map(int,E['budget'][arm]));ax.plot([0]+bs,[E['primary'][arm]['raw']['MA']['mae']]+[E['budget'][arm][str(b)]['MA']['mae'] for b in bs],'-o',ms=3,color=COLORS[0]);ax.set(xlabel='Local paired records',ylabel='MA MAE (mm)');panel(ax,string.ascii_lowercase[i],short(arm))
    save(fig,'S8_adaptation_budget')
    fig,axs=plt.subplots(1,2,figsize=(6.8,3))
    for i,arm in enumerate(c.MAIN):
        z=analysis.load(ROOT/'gates'/f'{arm}_T900.npz');idx=z['test'];y=d['y'][idx,2]
        axs[0].plot(c.SEEDS,[checkscore(z['prediction'][j],y,z['boundary95_action'][j],'coverage')*100 for j in range(20)],'-o',ms=2,label=short(arm),color=COLORS[i]);axs[1].plot(c.SEEDS,[checkscore(z['prediction'][j],y,z['boundary95_action'][j],'agreement')*100 for j in range(20)],'-o',ms=2,color=COLORS[i])
    axs[0].set(xlabel='Seed',ylabel='Coverage (%)');axs[1].set(xlabel='Seed',ylabel='Agreement (%)');axs[0].legend(fontsize=6);save(fig,'S9_seed_stability')

def checkscore(p,y,a,key):return c.summarize(p,y,a)[key]

def representation():
    d,*_=c.context();fits=[c.read(SYNC/f'experiments/signal/fit_{i}.json') for i in range(480)];z=analysis.load(SYNC/'experiments/signal/REFITS.npz');qc=z['qc']&d['input_qc'];fig,axs=plt.subplots(2,2,figsize=(6.8,5.5))
    axs[0,0].boxplot([z['EXP4'][qc&(d['centers']==ce),3] for ce in ['SAHZU','Huzhou','Hubei']],tick_labels=['SAHZU','Huzhou','Hubei'],showfliers=False);axs[0,0].axhline(1,color='grey',ls=':');axs[0,0].set_ylabel('Fitted shape exponent μ')
    xx=np.array([f['EXP3']['nrmse'] for f in fits])[qc];yy=np.array([f['EXP4']['nrmse'] for f in fits])[qc];axs[0,1].scatter(xx,yy,s=7,alpha=.4);lim=max(xx.max(),yy.max());axs[0,1].plot([0,lim],[0,lim],color='grey',ls=':');axs[0,1].set(xlabel='EXP3 NRMSE',ylabel='EXP4 NRMSE')
    for i,arm in enumerate(c.MAIN[1:]):
        ax=axs[1,i]
        for j,rep in enumerate(['EXP4','EXP3']):
            vals=E['representation_curves'][arm];xx=[v[rep]['coverage']*100 for v in vals];yy=[v[rep]['agreement']['estimate']*100 if v[rep]['agreement']['estimate'] is not None else np.nan for v in vals];ax.plot(xx,yy,color=COLORS[j],label=rep)
        ax.set(xlabel='Coverage (%)',ylabel='Agreement (%)',xlim=(0,100),ylim=(0,100));ax.legend(fontsize=7)
    for i,ax in enumerate(axs.flat):panel(ax,string.ascii_lowercase[i],['Shape by centre','Signal-fitting errors',short(c.MAIN[1]),short(c.MAIN[2])][i])
    save(fig,'S10_representation')

def diagnostics():
    from code_submit.paper_b085_hospital_replay.equal_error import upper_hull
    fig,axs=plt.subplots(1,2,figsize=(6.8,3));dd=E['diagnostics']['D2']
    for i,arm in enumerate(c.MAIN):
        vals=dd['points'][arm];x=np.array(vals['error']);y=np.array(vals['correct']);ii=upper_hull(x,y);axs[0].scatter(100*x,100*y,s=7,alpha=.45,color=COLORS[i]);axs[0].plot(100*x[ii],100*y[ii],color=COLORS[i],label=short(arm))
    axs[0].set(xlabel='Incorrect reports / all events (%)',ylabel='Correct reports / all events (%)');axs[0].legend(fontsize=6)
    for i,(key,marker) in enumerate([('exact','o'),('at_most','s')]):
        for b,vals in dd['contrasts']['HZ+HB_to_SA'].items():
            v=vals[key]
            if v['delta'] is not None:errbar(axs[1],float(b)*100+(-.08 if i==0 else .08),dict(estimate=v['delta'],low=v['low'],high=v['high']),COLORS[i],marker=marker,pct=100)
    axs[1].axhline(0,color='grey',lw=.5);axs[1].set(xlabel='Error-yield budget (%)',ylabel='Dual − target correct yield (pp)');panel(axs[0],'a','Retrospective policy frontier');panel(axs[1],'b','Exact and at-most matching');save(fig,'S11_equal_error')
    fig,axs=plt.subplots(1,2,figsize=(6.8,3))
    for i,arm in enumerate(c.MAIN[1:]):
        for j,source in enumerate([E['ridge_budget'],E['budget'][arm]]):
            vals=[source[str(b)]['MA'] for b in [20,40,60]];axs[i].plot([20,40,60],[v['mae'] for v in vals],'-o',color=COLORS[j],ms=3,label=['Target-only Ridge',short(arm)][j])
            for b,v in zip([20,40,60],vals):errbar(axs[i],b,v['interval'],COLORS[j])
        axs[i].set(xlabel='Local paired records',ylabel='MA MAE (mm)',xticks=[20,40,60]);axs[i].legend(fontsize=7);panel(axs[i],string.ascii_lowercase[i],short(arm))
    save(fig,'S12_label_budget')

def run():
    suite.initialize();OUT.mkdir(parents=True,exist_ok=True)
    gates.clean_write(SYNC/'FIGURE_CONTRACT.json',dict(backend='Matplotlib; existing editable workflow diagram retained with input-version labels corrected',
        claim='Parameter mapping, report selection, and early-release consequences are evaluated separately.',
        main2='Numerical mapping and local correction; auxiliary readouts explicitly distinct from MA branch.',
        main3='Fixed-time report availability and state errors, with all-report denominator retained.',
        main4='Early reports can persist as either correct or incorrect; compare cumulative and fixed-time outcomes.',
        supplementary='Nine-direction, model, budget, representation, seed and exploratory controls.',
        dimensions='173 mm wide; panel-specific heights',exports=['editable PDF','editable SVG','PNG preview'],
        intervals='Current group-bootstrap artifacts only; no old intervals reused',evidence='Retrospective evaluation of measured U-TEG signals.'))
    for fn in [prediction,selection,observation,supplement,representation]:fn()

if __name__=='__main__':run()
