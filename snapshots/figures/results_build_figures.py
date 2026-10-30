"""Read-only analysis of frozen predictions; new figure exports only. No model fitting."""
from pathlib import Path
if __name__ == '__main__' and not globals().get('_CURRENT_ONLY', False):
    import runpy
    runpy.run_path(str(Path(__file__).with_name('unify_current.py')), run_name='__main__')
    raise SystemExit(0)
import json, csv, hashlib, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
SD = OUT / 'source_data'
for p in (FIG, SD): p.mkdir(exist_ok=True)
EXT = ROOT/'outputs/uteg_ridge_dual_head_external_eval_20261026_191117'
RIDGE = ROOT/'outputs/uteg_signal_model_goal_20261026_000929/ridge10_fullchain_20261026_122551'
DATA = ROOT/'outputs/uteg_legacy_aligned_threecenter_20261022_155614/DATA15.npz'
TIME = ROOT/'outputs/uteg_evidence_figures_20261025_013932/source_data'
MAIN = ['HZ_to_SA','HZ+HB_to_SA']
DUAL = ['SA+HB_to_HZ','HZ+HB_to_SA','SA+HZ_to_HB']
CENTERS = ['Huzhou','SAHZU','Hubei']
COLORS = ['#387CA5','#258777','#BA8551']
BLUE,TEAL,RED,ORANGE,GRAY,INK = '#387CA5','#258777','#B4505B','#C48A49','#DDE3E7','#243442'
SHORT = {'HZ_to_SA':'HZ → SA','HZ+HB_to_SA':'HZ + HB → SA', 'SA+HB_to_HZ':'SA + HB → HZ','SA+HZ_to_HB':'SA + HZ → HB','HB_to_SA':'HB → SA','SA_to_HZ':'SA → HZ','HB_to_HZ':'HB → HZ','SA_to_HB':'SA → HB','HZ_to_HB':'HZ → HB'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8.2,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':9})
MM=1/25.4
sources={}
def note(path):
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
def readj(p):
    note(p); return json.loads(p.read_text(encoding='utf-8-sig'))
def readnp(p):
    note(p)
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def writecsv(name,rows):
    with (SD/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def label(ax,letter,title):
    ax.text(-.033/ax.get_position().width,1.075,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    ax.set_title(title,loc='left',pad=9)
def clean(ax):
    ax.tick_params(length=3,width=.6)
    ax.set_axisbelow(True)
def save(fig,name):
    fig.canvas.draw()
    for ext in ['svg','pdf','png']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=400)
    fig.canvas.draw()
    # Text box boundary check. Tight bounding boxes can include annotations; inspect exported PNG too.
    renderer=fig.canvas.get_renderer()
    outside=[]
    excluded={id(t) for ax in fig.axes for t in ax.get_xticklabels()+ax.get_yticklabels()}
    for txt in fig.findobj(matplotlib.text.Text):
        if id(txt) in excluded: continue
        if txt.get_visible() and txt.get_text():
            b=txt.get_window_extent(renderer)
            if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.width+2 or b.y1 > fig.bbox.height+2:
                outside.append(txt.get_text())
    qa[name]={'off_canvas_text':outside,'size_mm':[round(v/MM,2) for v in fig.get_size_inches()]}
    plt.close(fig)

qa={}
dataset=readnp(DATA)
reference_ids=dataset['ids'].astype(str)
id_index={v:i for i,v in enumerate(reference_ids)}
results=readj(EXT/'REFERENCE_EVALUATION_RESULT.json')
actions={}; preds={}; refs={}; cls={}
for d in results['direction_metrics']:
    actions[d]=readnp(EXT/'actions'/f'{d}.npz')
    preds[d]=readnp(RIDGE/'eval/predictions'/f'{d}.npz')
    assert np.array_equal(preds[d]['seed'],np.arange(1000,1020))
    assert np.array_equal(actions[d]['test_ids'],preds[d]['test_ids'])
    assert np.allclose(actions[d]['final_ma'],preds[d]['final_ma'],equal_nan=True)
    ix=[id_index[x] for x in actions[d]['test_ids'].astype(str)]
    refs[d]=dataset['medical'][ix]
    ma=refs[d][:,2]; cls[d]=np.where(ma<50,-1,np.where(ma>70,1,0))

# Fixed, patient-cluster bootstrap: all seed predictions for a record travel together.
BOOT=5000
def rate_interval(numer,denom):
    numer=np.asarray(numer,float); denom=np.asarray(denom,float)
    rng=np.random.default_rng(20261026)
    ix=rng.integers(0,len(numer),size=(BOOT,len(numer)))
    bot=denom[ix].sum(axis=1); good=bot>0
    if not np.any(good): return [None,None]
    return np.quantile(numer[ix].sum(axis=1)[good]/bot[good],[.025,.975]).tolist()
def release(d,baseline=False):
    a=actions[d]; p=a['final_ma_class']; r=a['baseline_lr_action' if baseline else 'unified_action'].astype(bool)
    return r&a['input_qc'][None,:].astype(bool)&np.isin(p,[-1,0,1])
def mstats(d,baseline=False):
    r=release(d,baseline); p=actions[d]['final_ma_class']; y=cls[d][None,:]
    good=r&(p==y); low=r&(p==0)&(y==-1); high=r&(p==0)&(y==1)
    stats=dict(released=int(r.sum()),correct=int(good.sum()),total=int(r.size),agreement=float(good.sum()/r.sum()) if r.sum() else None,coverage=float(r.mean()),low_wrong=int(low.sum()),high_wrong=int(high.sum()),low_total=int((y==-1).sum()*20),high_total=int((y==1).sum()*20))
    stats['agreement_interval']=rate_interval(good.mean(0),r.mean(0)); stats['coverage_interval']=rate_interval(r.mean(0),np.ones(r.shape[1]))
    for c,k,event in [(-1,'low',low),(1,'high',high)]:
        support=(cls[d]==c); stats[k+'_risk']=float(event.sum()/stats[k+'_total']) if stats[k+'_total'] else None
        stats[k+'_interval']=rate_interval(event.mean(0)[support],np.ones(support.sum())) if support.any() else [None,None]
    return stats
stats={d:mstats(d) for d in actions}
base={d:mstats(d,True) for d in MAIN}
for d,s in stats.items():
    expected=results['direction_metrics'][d]['metrics']
    assert s['released']==expected['released'] and s['correct']==expected['correct']
    assert s['low_wrong']==expected['low_to_within'] and s['high_wrong']==expected['high_to_within']

# Figure 2: parameter translation before report selection.
fig=plt.figure(figsize=(183*MM,156*MM))
gs=fig.add_gridspec(2,4,left=.09,right=.98,bottom=.13,top=.88,wspace=.63,hspace=.75,height_ratios=[.85,1.3])
fig.text(.055,.975,'Figure 2 | Cross-center translation',fontsize=12,fontweight='bold',va='top')
units=['R MAE (min)','K MAE (min)','MA MAE (mm)',r'$\alpha_R$ MAE (deg)']
errrows=[]
for k in range(4):
    ax=fig.add_subplot(gs[0,k]);label(ax,chr(97+k),units[k])
    for i,d in enumerate(DUAL):
        p=preds[d]; truth=refs[d][:,k]
        mask=p['input_qc'].astype(bool)&np.isfinite(truth)&np.all(np.isfinite(p['raw_med4'][:,:,k]),axis=0)&np.all(np.isfinite(p['final_med4'][:,:,k]),axis=0)
        for stage,key,offset,color,marker in [('Mapped','raw_med4',-.10,'#9CAAB3','o'),('Adapted','final_med4',.10,COLORS[i],'D')]:
            errors=np.abs(p[key][:,mask,k]-truth[None,mask]).mean(0)
            val=float(errors.mean()); lo,hi=rate_interval(errors,np.ones(len(errors)))
            ax.errorbar(val,i+offset,xerr=[[val-lo],[hi-val]],fmt=marker,color=color,ms=3.6,capsize=2,lw=.75)
            errrows.append(dict(direction=d,readout=units[k],stage=stage,n=int(mask.sum()),mae=val,lower=lo,upper=hi))
        ev=errrows[-2:];ax.plot([v['mae'] for v in ev],[i-.1,i+.1],color='#C8D1D8',lw=.65,zorder=0)
    ax.set_yticks(range(3),CENTERS if k==0 else ['']*3);ax.set_ylim(2.4,-.4);ax.set_xlim(left=0);clean(ax)
fig.legend(handles=[Line2D([],[],marker='o',color='#9CAAB3',ls='',label='Mapped'),Line2D([],[],marker='D',color=BLUE,ls='',label='Adapted')],loc='upper right',bbox_to_anchor=(.98,.974),frameon=False,ncol=2,fontsize=8)
scatterrows=[]
for j,d in enumerate(MAIN):
    ax=fig.add_subplot(gs[1,j*2:j*2+2]);label(ax,chr(101+j),SHORT[d])
    xx=refs[d][:,2]; yy=np.mean(preds[d]['final_med4'][:,:,2],axis=0); raw=preds[d]['raw_med4'][:,:,2].mean(0)
    ax.axhspan(50,70,color='#EDF3F6',zorder=0);ax.axvspan(50,70,color='#F6F8FA',zorder=-1)
    for c,color,marker in [(-1,ORANGE,'v'),(0,'#8FA2AF','o'),(1,BLUE,'^')]:
        mask=cls[d]==c;ax.scatter(xx[mask],yy[mask],s=14 if c else 8,c=color,marker=marker,alpha=.85,edgecolors='white',linewidths=.25,zorder=3)
    lo=min(20,float(np.nanmin(xx))-2,float(np.nanmin(yy))-2);hi=max(90,float(np.nanmax(xx))+2,float(np.nanmax(yy))+2)
    ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color='#637784')
    for cut in [50,70]: ax.axhline(cut,color='#ACBAC5',lw=.5); ax.axvline(cut,color='#ACBAC5',lw=.5)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Reference MA (mm)',ylabel='Adapted MA (mm)');clean(ax)
    ax.text(.03,.97,'n = 201',transform=ax.transAxes,fontsize=9,va='top')
    for i,x in enumerate(xx):scatterrows.append(dict(direction=d,record_id=str(actions[d]['test_ids'][i]),reference_ma=float(x),mapped_mean=float(raw[i]),adapted_mean=float(yy[i]),reference_class=int(cls[d][i])))
fig.text(.055,.035,'20 seeds · 95% intervals · Full-cohort predictions',fontsize=8.5,color='#607482')
save(fig,'Figure2_translation')
writecsv('figure2_readout_errors.csv',errrows);writecsv('figure2_ma_scatter.csv',scatterrows)

# Figure 3: current main operating points, matched old gate, and class disposition.
fig=plt.figure(figsize=(183*MM,105*MM));gs=fig.add_gridspec(1,2,left=.10,right=.98,bottom=.24,top=.78,wspace=.72,width_ratios=[1,1.12])
fig.text(.055,.96,'Figure 3 | Selective reporting',fontsize=12,fontweight='bold',va='top')
ax=fig.add_subplot(gs[0,0]);label(ax,'a','Agreement and coverage')
for j,d in enumerate(MAIN):
    for s,marker,alpha in [(base[d],'o',.6),(stats[d],'D',1)]:
        ax.scatter(100*s['coverage'],100*s['agreement'],marker=marker,s=30,c=COLORS[j],alpha=alpha,zorder=4)
        if marker=='D':
            al,ah=s['agreement_interval'];cl,ch=s['coverage_interval']
            ax.errorbar(100*s['coverage'],100*s['agreement'],xerr=[[100*(s['coverage']-cl)],[100*(ch-s['coverage'])]],yerr=[[100*(s['agreement']-al)],[100*(ah-s['agreement'])]],color=COLORS[j],lw=.8,capsize=2,fmt='none')
    ax.annotate('',xy=(100*stats[d]['coverage'],100*stats[d]['agreement']),xytext=(100*base[d]['coverage'],100*base[d]['agreement']),arrowprops=dict(arrowstyle='->',lw=.9,color=COLORS[j]))
    ax.text(100*stats[d]['coverage']+1,100*stats[d]['agreement']+(1.4 if j else -4.6),f"{SHORT[d]}\n{100*stats[d]['agreement']:.2f}% / {100*stats[d]['coverage']:.2f}%",fontsize=8,color=COLORS[j])
ax.set(xlim=(0,65),ylim=(73,101),xlabel='Coverage (%)',ylabel='Agreement (%)');clean(ax)
ax.legend(handles=[Line2D([],[],marker='o',color=INK,ls='',label='Previous rule'),Line2D([],[],marker='D',color=INK,ls='',label='Current rule')],loc='lower left',frameon=False,fontsize=8)
ax=fig.add_subplot(gs[0,1]);label(ax,'b','Abnormal → within')
labels=[]
for j,d in enumerate(MAIN):
    for k,tail in enumerate(['low','high']):
        row=j*2+k;labels.append(f"{'HZ' if j==0 else 'HZ+HB'} → SA · {tail.capitalize()}")
        old=base[d][tail+'_risk']*100;new=stats[d][tail+'_risk']*100;lo,hi=stats[d][tail+'_interval']
        ax.plot([old,new],[row,row],color=COLORS[j],lw=1.2)
        ax.scatter(old,row,s=22,facecolors='white',edgecolors=COLORS[j],zorder=4)
        ax.errorbar(new,row,xerr=[[new-100*lo],[100*hi-new]],fmt='D',color=COLORS[j],ms=4,capsize=2,lw=.75)
        ax.text(63,row,f"{stats[d][tail+'_wrong']}/{stats[d][tail+'_total']}",va='center',fontsize=8.5)
ax.text(63,-.52,'Current n/N',fontsize=7.5,ha='left')
ax.set_yticks(range(4),labels);ax.set_ylim(3.7,-.7);ax.set_xlim(0,88);ax.set_xlabel('Within true MA class (%)');clean(ax)
flows=[]
for j,d in enumerate(MAIN):
    p=actions[d]['final_ma_class'];r=release(d);y=cls[d][None,:];values=[]
    for c in [-1,0,1]:
        m=(y==c);den=int(m.sum()*20)
        correct=int((r&m&(p==c)).sum());within=int((r&m&(p==0)&(c!=0)).sum());other=int((r&m&(p!=c)).sum())-within; withheld=den-correct-within-other
        values.append(np.array([correct,within,other,withheld])/den*100)
        flows.append(dict(direction=d,reference_class=c,unique_records=int(m.sum()),record_seed_denominator=den,correct_release=correct,abnormal_to_within=within,other_wrong=other,withheld=withheld))
fig.text(.055,.075,'20 seeds · Low: 7 × 20 = 140; high: 24 × 20 = 480\nCounts are record–seed events, not independent patients. 95% cluster intervals.',fontsize=8,color='#607482')
save(fig,'Figure3_reporting');writecsv('figure3_class_disposition.csv',flows)

# Figure 4: keep the historical MA time replay separate from current Ridge endpoints.
seq=readj(TIME/'FIRST_REPORT_SUMMARIES.json');intervals=readj(TIME/'PAIRED_INTERVALS.json')
fig=plt.figure(figsize=(183*MM,146*MM));gs=fig.add_gridspec(2,3,left=.10,right=.98,bottom=.19,top=.85,wspace=.4,hspace=.68,height_ratios=[1.1,.85])
fig.text(.055,.975,'Figure 4 | Continued observation: additional reports and their cost',fontsize=10.5,fontweight='bold',va='top')
fig.text(.055,.936,'Exploratory MA time replay · earlier frozen MA workflow, not the current Ridge/shared-threshold policy',fontsize=7.2,color='#8E5E2E')
timerows=[]
for j,c in enumerate(CENTERS):
    ax=fig.add_subplot(gs[0,j]);label(ax,chr(97+j),c)
    rows=[seq['centers'][c][t]['mean'] for t in ['T300','T600','T900']];den=rows[0]['known_n'];bot=np.zeros(3)
    for key,color in [('correct_n',TEAL),('wrong_n',RED),('unreported_n',GRAY)]:
        vals=np.array([m[key]/den*100 for m in rows]);ax.bar(range(3),vals,bottom=bot,color=color,width=.62,edgecolor='white',lw=.5)
        for i,v in enumerate(vals):
            if v>=4:ax.text(i,bot[i]+v/2,f'{v:.1f}',ha='center',va='center',fontsize=6.8,color='white' if key!='unreported_n' else INK)
        bot+=vals
    ax.set_xticks(range(3),['5','10','15']);ax.set_xlabel('Observation time (min)');ax.set_ylim(0,100);ax.set_ylabel('Full test cohort (%)' if j==0 else '')
    ax.text(.5,1.01,f'n={int(den)}; 20 seeds',transform=ax.transAxes,ha='center',fontsize=6.7,color='#607482');clean(ax)
    for t,m in zip([5,10,15],rows):timerows.append(dict(center=c,time_min=t,unique_records=int(den),mean_correct=m['correct_n'],mean_wrong=m['wrong_n'],mean_unreported=m['unreported_n'],low_to_within=m['low_to_within_n'],high_to_within=m['high_to_within_n']))
ax=fig.add_subplot(gs[1,:]);label(ax,'d','Additional reporting by 15 min versus stopping at 5 min')
for j,c in enumerate(CENTERS):
    for k,outcome in enumerate(['correct_report','wrong_report']):
        rec=next(v for v in intervals['intervals'] if v['center']==c and v['outcome']==outcome);v=rec['point_estimate']*100;lo,hi=np.array(rec['bonferroni_95_interval'])*100; y=j+(k-.5)*.24
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o' if k==0 else 's',color=TEAL if k==0 else RED,capsize=2,ms=4,lw=1)
        ax.text(hi+.5,y,f'{v:+.2f} pp',va='center',fontsize=7)
ax.axvline(0,color='#98A7B1',lw=.6,ls='--');ax.set_yticks(range(3),CENTERS);ax.set_ylim(2.5,-.5);ax.set_xlim(-1,28);ax.set_xlabel('Change in fraction of the full test cohort (percentage points)');clean(ax)
fig.legend(handles=[Patch(color=TEAL,label='Correct report'),Patch(color=RED,label='Wrong report'),Patch(color=GRAY,label='Not reported')],loc='lower center',bbox_to_anchor=(.53,.065),ncol=3,frameon=False)
fig.text(.055,.025,'First report is retained; later observations can release previously unreported records. d: saved paired, Bonferroni-adjusted intervals.\nCumulative counts increase by construction; this replay does not isolate a causal benefit of new measurements.',fontsize=6.6,color='#607482')
save(fig,'Figure4_observation');writecsv('figure4_first_reports.csv',timerows)

# Full nine-direction context; never omit low-performing rotations.
order=['HZ_to_SA','HB_to_SA','SA_to_HZ','HB_to_HZ','SA_to_HB','HZ_to_HB','HZ+HB_to_SA','SA+HB_to_HZ','SA+HZ_to_HB']
fig,axs=plt.subplots(1,4,figsize=(183*MM,130*MM),gridspec_kw={'left':.20,'right':.98,'bottom':.14,'top':.82,'wspace':.48})
fig.text(.04,.965,'Figure S1 | All nine transfer directions under one fixed reporting rule',fontsize=10.4,fontweight='bold',va='top')
fig.text(.04,.92,'Shared-threshold gate · Ridge-10 · 20 seeds · target adaptation budgets HZ 40 / SA 60 / HB 40',fontsize=7.2,color='#607482')
metrics=[('agreement','Agreement (%)','agreement_interval'),('coverage','Coverage (%)','coverage_interval'),('low_risk','Low → within (%)','low_interval'),('high_risk','High → within (%)','high_interval')]
for k,(key,title,ik) in enumerate(metrics):
    ax=axs[k];label(ax,chr(97+k),title)
    for i,d in enumerate(order):
        s=stats[d];color=TEAL if d in MAIN else '#8C9BA5';v=s[key]
        if v is None:ax.text(1,i,'NA',va='center',fontsize=7,color='#8C9BA5');continue
        lo,hi=s[ik];ax.errorbar(v*100,i,xerr=[[(v-lo)*100],[(hi-v)*100]],fmt='D' if d in MAIN else 'o',color=color,ms=3.5,capsize=2,lw=.8)
    ax.set_yticks(range(9),[SHORT[d] for d in order] if k==0 else ['']*9);ax.set_ylim(8.7,-.7);ax.set_xlim(0,105 if k==0 else 50);ax.axhline(5.5,color='#D4DDE3',lw=.6);clean(ax)
fig.text(.04,.04,'NA: no reports (agreement) or no reference high-MA cases (Huzhou). Intervals resample patients, retaining all seeds together.\nHighlighted directions were chosen for exploratory presentation after viewing results; they do not establish universal transfer.',fontsize=6.7,color='#607482')
save(fig,'FigureS1_nine_directions')
tab=[]
for d in order:
    s=stats[d];tab.append(dict(direction=d,**{k:v for k,v in s.items() if not isinstance(v,list)},agreement_ci=json.dumps(s['agreement_interval']),coverage_ci=json.dumps(s['coverage_interval'])))
writecsv('nine_direction_summary.csv',tab)

captions={
'Figure2_translation':('转译与适配','A–D：分别显示R、K、MA及冻结R法α的映射前后绝对误差，三行对应两源中心到各目标中心；横坐标越小越好。E–F：两个重点方向的全测试队列MA散点，横轴参考MA、纵轴适配MA；每病例展示20个种子的均值，不选最佳种子，也不只展示放行样本。阴影及50/70线解释三状态边界。误差棒为5000次病例整群重采样的95%区间，重复种子不是独立患者。湖北数值误差采用39/41输入QC合格记录；报告覆盖率仍保留41的完整分母。'),
'Figure3_reporting':('选择性报告及异常风险','A：同一预测器、校准和测试病例下，原规则与当前规则的一致率—覆盖率操作点；箭头不是连续阈值曲线。B：低/高MA报范围内的比例，空心圆为原规则，实心菱形及区间为当前规则，右侧标出当前20个seed的合计n/N。低MA分母为7例×20=140，高MA为24例×20=480；这不是新增独立患者。旧C、D与B重复，已删除；完整正确、误报和拒报计数保留在源数据表。两组是探索性重点方向，完整九组见S1。原规则为MA-ERROR-LR，当前规则为双侧风险判断及冻结共享异常阈值；模型信息仅在图注与方法记录中说明。'),
'Figure4_observation':('继续观察的收益和代价——独立时序分支','A–C：早期已冻结MA流程的5、10、15分钟首次报告累计构成，正确、错误及未报告相加100%。D：15分钟相对5分钟新增正确和错误报告，展示既有的多重比较调整配对区间。该图不是当前Ridge＋共享阈值流程的时间扩展，不能与Figure 3的15分钟数值串成同一轨迹。湖北没有放行，保留该事实。累计增长不等于预测质量必然改善，也未单独识别新观测的因果收益。'),
'FigureS1_nine_directions':('完整九方向背景','同一Ridge＋共享阈值流程的九方向一致率、覆盖率、低/高MA误放比例。零放行时一致率显示NA，湖州没有高MA参考样本时高MA风险显示NA。绿色仅强调两组正文候选，不删除其余方向。')}
plan='''Results实验图组方案（绘图初稿；未改正文、Figure 1或旧HTML）

总线：Figure 1既有测量—转译—适配—报告流程保持不变。Figure 2回答医学读出如何跨中心转译；Figure 3回答哪些预测能被选择性报告及其代价；Figure 4回答继续观察有无新增报告收益。四个Agent通过功能证据获得意义，不重复画框图。

范例与可迁移逻辑：
1. Bai et al., Nature Communications 15,462 (2024), Fig.2/5/6及完整Results。Fig.2定义层与接口，Fig.5展示动作/反馈，Fig.6a先展示成本—收益目标取舍，6b/c再解释设计变量。我们保留Figure 1接口图，Figure 3先给一致率—覆盖率操作点、再解释病例去向。不复制3D布局，不声称机器人或自主实验能力。
https://www.nature.com/articles/s41467-023-44599-9
2. Rapp et al., Nature Chemical Engineering 1,97–107 (2024), Fig.1b–f、3a、4a–f及完整Results。预测器、策略、真实测量和反馈行为逐层举证，运行轨迹保留失败/不确定。我们的Figure 4将正确报告与错误同时累计展示，末段用配对增量回答额外收益；不把累计单调性当性能单调改善。不把其四个独立SAMPLE副本等同于我们的四类功能角色。
https://www.nature.com/articles/s44286-023-00002-4
3. Ghetmiri et al., Nature Communications 15,314 (2024), Fig.4–7、8及完整Results。将模型拟合、预测、新样本验证和医学读出分开。我们Figure 2先给医学参数误差，再给全队列MA散点；不把筛选后的散点当无条件预测能力。该文是医学建模范例，并非Agent架构论文。
https://www.nature.com/articles/s41467-023-44231-w

版式：沿用现有Times New Roman；英文简短标注、小写粗体a/b/c、白底、双栏183mm、蓝/青灰主色，红橙专用于错误类型；SVG文字可编辑，PDF嵌入TrueType，PNG用于预览。不借用范例数据或复制其结论。

证据身份：Figure 2–3为当前固定Ridge-10及已有适配，Figure 3使用共享阈值门；完整20个seed1000–1019。Figure 4是更早的MA时序验证，必须保留独立分支标题；在没有匹配新流程的时序结果前，它是探索性图稿，不应宣称整个主流程闭环已全部验证。正文现存CI及旧结果未更新，等待作者批准写作时统一。

曲线NMAE：旧模型化曲线反解存在成功/截断限制且并非本次Ridge预测同身份，不复用旧数值制造新模型的曲线保真结论；本轮不重做反解。图2当前以四医学参数为主。

Baseline安排：Figure 3的原LR门与共享阈值门比较使用同一批连续预测。历史六映射/六可信门比较包含旧模型或CI终点，不能拼作当前MA的排行榜。本轮只保留可配对的控制，其他历史比较应置附录并标清分支，后续若需主文多模型图必须先证明协议匹配，不能靠改标签转换。

统计：所有新计算仅从保存预测复算/重采样，不重新拟合、选择模型、调阈值或增加实验。Figure 2/3/S1置信区间是5000次固定随机数病例整群bootstrap（每病例20个seed一起重采样）；Figure 4沿用保存的20000次病例配对、Bonferroni六比较区间。探索性选择两方向不掩盖完整九方向；两主方向同为SAHZU目标，不能声称两次独立目标中心验证。
'''
(OUT/'FIGURE_PLAN_ZH.txt').write_text(plan,encoding='utf-8')
(OUT/'CAPTIONS_ZH.txt').write_text('\n\n'.join(f'{k} — {v[0]}\n{v[1]}' for k,v in captions.items()),encoding='utf-8')
(SD/'reporting_statistics.json').write_text(json.dumps({'current':stats,'baseline':base},indent=2),encoding='utf-8')
(OUT/'PROVENANCE.json').write_text(json.dumps({'sources':sources,'fit_units':0,'model_changes':False,'threshold_changes':False,'seeds':list(range(1000,1020)),'bootstrap_patient_clusters':BOOT,'time_branch_separate':True},indent=2),encoding='utf-8')
(OUT/'QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
cards=''.join(f'<section><h2>{html.escape(k)} · {html.escape(v[0])}</h2><img src="figures/{k}.svg" alt="{k}"><p>{html.escape(v[1])}</p><p><a href="figures/{k}.pdf">PDF</a> · <a href="figures/{k}.svg">可编辑SVG</a> · <a href="figures/{k}.png">PNG</a></p></section>' for k,v in captions.items())
(OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>U-TEG Results实验图组</title><style>body{font:16px/1.7 system-ui;color:#243442;background:#f3f5f6;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:28px}section{background:white;border:1px solid #d3dde3;padding:24px;margin:24px 0;border-radius:8px}img{display:block;width:100%;height:auto}p{max-width:940px}a{color:#226e93}.note{border-left:4px solid #b17e45;background:#fff4e5;padding:14px}</style><main><h1>Results：从转译到报告，再到继续观察</h1><p>Figure 1保留。以下为Figure 2–4的新图稿和完整九方向补充图；未修改论文正文或原汇报页面。</p><p class="note">Figure 2–3：当前Ridge＋共享阈值证据。Figure 4：独立的既有MA时序分支，不能当作当前流程的时间扩展。全部采用20个种子，不挑最佳seed。</p><p><a href="FIGURE_PLAN_ZH.txt">图组思路与三篇范例借鉴依据</a> · <a href="CAPTIONS_ZH.txt">中文图注</a> · <a href="source_data/nine_direction_summary.csv">完整九方向数据</a></p>'+cards+'</main></html>',encoding='utf-8')
print(json.dumps({'qa':qa,'main':{d:stats[d] for d in MAIN},'translation':errrows},indent=2))
