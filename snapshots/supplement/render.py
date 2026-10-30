"""Appendix figures only. Existing manuscript figures are never opened for writing."""
import os,sys,json,html
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(P/'plot_cache')
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.titlesize':7,
       'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42}
plt.rcParams.update(STYLE)
METHODS=['RIDGE-10','KRR-1','PLS-2','SVR-1','MLP-8','TT-SMALL']
COLORS=['#416B8A','#8B739A','#6E967E','#CA9470','#7E8D98','#BA7992']
DIRS=['HZ_to_SA','HB_to_SA','HZ+HB_to_SA','SA_to_HZ','HB_to_HZ','SA+HB_to_HZ','SA_to_HB','HZ_to_HB','SA+HZ_to_HB']
SHORT=lambda d:d.replace('_to_',' → ').replace('+',' + ')
FIG=P/'figures';FIG.mkdir(exist_ok=True)
captions=[]
def get(n):return pd.read_csv(P/'tables'/n)
def export(fig,name,title,caption,sources):
    for ext in ['svg','pdf','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig);captions.append(dict(name=name,title=title,caption=caption,sources=sources))
def panel(ax,i,title):ax.set_title(title,loc='left',pad=9);ax.text(-.13,1.06,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=8)
def err(ax,x,lo,hi,y,color,marker='o'):
    if not np.isfinite(x):return
    ax.plot([lo,hi],[y,y],color=color,lw=.8);ax.plot(x,y,marker,color=color,ms=3)

def main():
    centers=get('A1_center_QC.csv');fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
    for ax,(_,r) in zip(axes,centers.iterrows()):
        ax.axis('off');name=r['center'];origin={'Huzhou':'95 paired records\n↓ no quarantine','SAHZU':'306 paired records\n↓ 2 quarantined','Hubei':'84 document pairs\n↓ 3 technical flags'}[name];text=f"{name}\n\n{origin}\n↓\nFrozen records: {r.records}\nProxy groups: {r.identity_proxy_groups}\n↓\nQC pass / fail: {r.QC_pass} / {r.QC_fail}\n\nLow / within / high MA\n{r.low} / {r.within} / {r.high}"
        ax.text(.5,.5,text,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=1',facecolor='#F1F4F6',edgecolor='#D2D8DE'))
    export(fig,'S1_cohort','Data and quality control','Two-source original audit: 401 records, two quarantined, 399 retained. Hubei document recovery: 84 pairs, three explicit technical flags, 81 retained for evaluation; 79 pass signal QC, two remain in reporting denominators and are withheld. Group IDs are identity proxies, not independently verified patient identities. Huzhou has no high-MA reference records; that subgroup is not evaluable. Detailed roles are in the accompanying tables.',['A1_center_QC.csv','A1_data_roles.csv'])
    gate=get('A2_A5_gate_working_points.csv');fig,axes=plt.subplots(1,4,figsize=(7.2,4.5),sharey=True)
    specs=[('agreement','Agreement (%)'),('coverage','Coverage (%)'),('low_to_within_rate_all','Low → within (%)'),('high_to_within_rate_all','High → within (%)')]
    for j,(key,title) in enumerate(specs):
        ax=axes[j];panel(ax,j,title)
        for i,d in enumerate(DIRS):
            for m,name in enumerate(['MA-ERROR-LR','Dual protection']):
                r=gate[(gate.direction==d)&(gate.method==name)].iloc[0]
                err(ax,100*r[key],100*r[key+'_ci_low'],100*r[key+'_ci_high'],8-i+(m-.5)*.22,['#A3ADB5','#416B8A'][m],['s','o'][m])
        ax.set(xlim=(0,100),xticks=[0,50,100]);ax.grid(axis='x',color='#EEEEEE',lw=.5)
    axes[0].set_yticks(range(9),[SHORT(d) for d in DIRS[::-1]])
    fig.legend([Line2D([],[],marker='s',color='#A3ADB5'),Line2D([],[],marker='o',color='#416B8A')],['Original error gate','Two-sided protection'],loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(wspace=.28,bottom=.12,left=.23)
    export(fig,'S2_nine_direction_report','Nine-direction report performance','Frozen 15-minute working points; all 20 seeds retained. Agreement is correct/released, coverage released/all test record-seed events. Low/high false-within denominators are all true low/high events, not only released cases. Whiskers: 95% identity-proxy-group cluster bootstrap (2,000 resamples), seeds retained together. Missing subgroup/zero-release estimates are NA, not zero. These are operating points, not evidence of superior ranking at equal coverage.',['A2_A5_gate_working_points.csv'])
    mapping=get('A3_mapping_comparison.csv')
    for page in range(3):
        fig,axes=plt.subplots(3,4,figsize=(7.2,6.3));ds=DIRS[page*3:page*3+3]
        for i,d in enumerate(ds):
            for j,param in enumerate(['R','K','MA','alpha']):
                ax=axes[i,j];panel(ax,4*i+j,SHORT(d)+' · '+param)
                for m,name in enumerate(METHODS):
                    for st,offset,color in [('raw_med4',-.14,'#BBC2C7'),('final_med4',.14,COLORS[m])]:
                        r=mapping[(mapping.direction==d)&(mapping.method==name)&(mapping.stage==st)].iloc[0]
                        err(ax,r[param+'_MAE'],r[param+'_MAE_ci_low'],r[param+'_MAE_ci_high'],5-m+offset,color)
                ax.set_yticks(range(6),METHODS[::-1] if j==0 else []);ax.set_xlim(left=0)
                ax.set_xlabel('MAE ('+{'R':'min','K':'min','MA':'mm','alpha':'deg'}[param]+')')
        fig.tight_layout(h_pad=2,w_pad=1.2)
        export(fig,f'S3_mapping_{page+1}','Mapping baselines: target '+['SA','HZ','HB'][page],'Grey: unadapted; coloured: frozen target-adapted predictions. Evaluated before reporting rejection on the same QC-eligible cases, splits and budgets. Every method has one fixed existing configuration, not a test-selected winner. 95% group-cluster bootstrap; source neural seeds vary, deterministic source fits reused. Detailed pure-Huber intermediate results remain in the table.',['A3_mapping_comparison.csv','B3_mapping_seed_values.csv'])
    a=get('A4_adaptation_paired.csv');fig,axes=plt.subplots(1,4,figsize=(7.2,4.2),sharey=True)
    for j,p in enumerate(['R','K','MA','alpha']):
        ax=axes[j];panel(ax,j,p);ax.axvline(0,color='#888888',ls='--',lw=.7)
        for i,d in enumerate(DIRS):
            r=a[(a.direction==d)&(a.parameter==p)].iloc[0];err(ax,r.estimate,r.ci_low,r.ci_high,8-i,'#258777')
        ax.set_xlabel('Raw − adapted MAE')
    axes[0].set_yticks(range(9),[SHORT(d) for d in DIRS[::-1]]);fig.subplots_adjust(left=.24,wspace=.35)
    export(fig,'S4_paired_adaptation','Paired adaptation effects','Positive differences favour adaptation, negative differences show harm. Same model, test cases, seed and budget; paired group-cluster bootstrap. This is not a contrast of adding a second source center. Class transitions, including correct-to-wrong and abnormal-to-within movements, are fully tabulated.',['A4_adaptation_paired.csv','A4_adaptation_class_flow.csv'])
    r=get('A3_MA_residual_records.csv');fig,axes=plt.subplots(3,3,figsize=(7.2,6.8))
    for i,d in enumerate(DIRS):
        ax=axes.flat[i];panel(ax,i,SHORT(d));q=r[(r.direction==d)&r.qc].groupby('record_id').agg(ref=('reference_MA','first'),err=('residual_MA','mean'))
        ax.scatter(q.ref,q.err,s=7,c='#416B8A',alpha=.65,rasterized=True);ax.axhline(0,c='#888888',lw=.7)
        for v in [50,70]:ax.axvline(v,c='#C7C7C7',lw=.7,ls='--')
        ax.set_xlabel('Reference MA (mm)');ax.set_ylabel('Predicted − reference MA (mm)')
    fig.tight_layout(h_pad=2,w_pad=1.5)
    export(fig,'S5_MA_residuals','MA residuals across reference states','Each dot is one test record; residuals averaged over all 20 seeds for visibility. Dots are not independent if their identity proxy is shared. Vertical lines show reference label boundaries (50,70 mm), horizontal line is zero residual. Full seed-level data remain available; this panel does not substitute for subgroup counts.',['A3_MA_residual_records.csv'])
    risk=get('A5_risk_coverage.csv')
    for page in range(3):
        fig,axes=plt.subplots(3,2,figsize=(7.2,6.3))
        for i,d in enumerate(DIRS[3*page:3*page+3]):
            for j,pop in enumerate(['all','predicted_within']):
                ax=axes[i,j];panel(ax,2*i+j,SHORT(d)+' · '+('All candidates' if j==0 else 'Within candidates'))
                q=risk[(risk.direction==d)&(risk.population==pop)]
                for k,name in enumerate(['MA margin','MA-ERROR-LR','Dual within risk']):
                    z=q[q.method==name];ax.plot(z.coverage*100,(1-z.agreement)*100,'o-',ms=2.5,color=['#A3ADB5','#8B739A','#416B8A'][k],label=name)
                ax.set(xlabel='Cohort coverage (%)',ylabel='Report error (%)',ylim=(0,100));ax.grid(color='#EEEEEE',lw=.5)
        axes[0,1].legend(frameon=False,fontsize=6);fig.tight_layout(h_pad=2)
        export(fig,f'S6_risk_coverage_{page+1}','Risk ranking: target '+['SA','HZ','HB'][page],'Descriptive equal-quota evaluation on common finite-score candidates, not threshold selection. Quotas fixed at 10–100% in each seed; ties resolved by record ID independently of labels. Within-candidate error is false-within risk among released within candidates. Dual-within and abnormal-gate scores are not pooled into an unjustified global ranking. Exact matched-coverage paired intervals are in the table.',['A5_risk_coverage.csv','A5_matched_coverage.csv'])
    temporal=get('A7_pointwise_cumulative.csv');fig,axes=plt.subplots(3,3,figsize=(7.2,6.5))
    maximum=max(5,100*temporal[temporal['mode']=='cumulative_first_report'].correct_yield.max()*1.15)
    for i,d in enumerate(DIRS):
        ax=axes.flat[i];panel(ax,i,SHORT(d));q=temporal[(temporal.direction==d)&(temporal['mode']=='cumulative_first_report')].sort_values('horizon')
        for key,c,marker in [('correct_yield','#258777','o'),('incorrect_yield','#BC6358','s')]:
            ax.plot(q.horizon/60,q[key]*100,marker+'-',c=c,ms=3,label=key.split('_')[0]);ax.fill_between(q.horizon/60,q[key+'_ci_low']*100,q[key+'_ci_high']*100,color=c,alpha=.12)
        ax.set(xlabel='Observation (min)',ylabel='First-report yield (%)',xticks=[5,10,15],ylim=(0,max(maximum,100*temporal.incorrect_yield.max()*1.15)))
    axes[0,0].legend(fontsize=6,frameon=False);fig.tight_layout(h_pad=2)
    export(fig,'S7_nine_direction_sequence','Continued observation: complete directions','Unified MA policy at all times; same cohort and full denominator. First report is locked; later predictions cannot erase earlier errors. Correct, incorrect and unreported sum to 100% (all counts in table). Shaded areas: pointwise descriptive 95% group-cluster bootstrap, not simultaneous confidence bands. Fixed-time comparisons are paired in B2. The Hubei B-assisted strategy is separately identified, not inserted into these curves.',['A7_pointwise_cumulative.csv','A7_first_report_times.csv','B2_sequence_fixed_paired.csv'])
    budget=get('B1_calibration_budget.csv');fig,axes=plt.subplots(3,3,figsize=(7.2,6.2))
    for i,d in enumerate(DIRS):
        ax=axes.flat[i];panel(ax,i,SHORT(d));q=budget[budget.direction==d].sort_values('calibration_n');ax.plot(q.calibration_n,q.MA_MAE,'o-',c='#416B8A',ms=3);ax.fill_between(q.calibration_n,q.MA_ci_low,q.MA_ci_high,color='#416B8A',alpha=.15)
        ax.set(xlabel='Adaptation records',ylabel='MA MAE (mm)',xticks=q.calibration_n,ylim=(0,None))
    fig.tight_layout(h_pad=2)
    export(fig,'S8_adaptation_budget','Label-budget sensitivity','Zero = same frozen source prediction without adaptation. C20 uses pre-existing frozen nested pools; full budgets are C40/C60/C40. Tests and source mappings fixed. 20 seeds, paired record identities; 95% group-cluster intervals. No gate is refitted for this parameter-level sensitivity comparison. Other medical parameters and MA classes are provided in the table.',['B1_calibration_budget.csv'])
    s=get('B3_gate_seed_values.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,4.5),sharey=True)
    for j,key in enumerate(['agreement','coverage']):
        ax=axes[j];panel(ax,j,key.title()+' (%)')
        for i,d in enumerate(DIRS):
            q=s[(s.direction==d)&(s.method=='Dual protection')].sort_values('seed');yy=8-i+np.linspace(-.16,.16,len(q));ax.scatter(q[key]*100,yy,s=8,c='#416B8A',alpha=.7)
        ax.set(xlim=(0,100),xticks=[0,25,50,75,100]);ax.grid(axis='x',color='#EEEEEE',lw=.5)
    axes[0].set_yticks(range(9),[SHORT(d) for d in DIRS[::-1]]);fig.subplots_adjust(left=.24,wspace=.25)
    export(fig,'S9_seed_stability','All-seed reporting stability','Every seed 1000–1019 is retained. Zero-report seeds appear at zero coverage and have undefined agreement (not drawn at 0%). This is seed variability, distinct from patient/record sampling uncertainty. Zero-report counts and all per-seed numbers are tabulated.',['B3_gate_seed_values.csv','A2_A5_gate_working_points.csv'])
    (P/'FIGURE_CAPTIONS.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2),encoding='utf-8')
    body=''.join('<section><h2>'+html.escape(x['name']+' · '+x['title'])+'</h2><img src="figures/'+x['name']+'.svg"><p>'+html.escape(x['caption'])+'</p><p>'+''.join('<a href="tables/'+n+'">'+n+'</a> ' for n in x['sources'])+'</p></section>' for x in captions)
    text='<!doctype html><meta charset="utf-8"><title>Appendix evidence package</title><style>body{font:16px Arial;max-width:1100px;margin:40px auto;color:#222}section{padding:30px 0;border-top:1px solid #ddd}img{width:100%}p{line-height:1.6}a{font-size:13px}</style><h1>附录：完整评价与可靠性</h1><p>正文内容、图表与数字未改动。全部九方向、20个seed；无测试选型。图中 SA=浙二，HZ=湖州，HB=湖北。范围内≠所有临床意义上的正常。</p>'+body
    (P/'index.html').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
