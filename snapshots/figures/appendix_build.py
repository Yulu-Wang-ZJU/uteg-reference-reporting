"""Read-only assembly of appendix review edition; no scientific experiment calls."""
import os,sys,json,html,hashlib,shutil,re,types
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent
os.environ['MPLCONFIGDIR']=str(P/'plot_cache');sys.dont_write_bytecode=True
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OLD=O/'uteg_supplement_completion_20261027_203140'
GOAL=O/'uteg_appendix_goal_20261028'
EXT=O/'uteg_preserved_within_external_20261028'
DEV=O/'uteg_preserved_within_dev_20261027'
STRENGTH=O/'uteg_ma_strength_dev_20261027'
CROSS=O/'uteg_ma_crossboundary_half_20261027'
NEST=O/'uteg_ma_crossboundary_nested10_20261027'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
esc=lambda s:html.escape(str(s),quote=True)
def dump(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
SOURCES={};TABLES=[];FIGS=[]
PALETTE={'ridge':'#356E8D','krr':'#8B739A','pls':'#6E967E','svr':'#CA9470','mlp':'#7E8D98','tt':'#BA7992',
         'error':'#8596A6','dual':'#356E8D','margin':'#B9926C','correct':'#258777','incorrect':'#BC6358','withheld':'#A8AFB5'}
DIRZH={'HZ_to_SA':'湖州→浙二','HB_to_SA':'湖北→浙二','HZ+HB_to_SA':'湖州＋湖北→浙二',
       'SA_to_HZ':'浙二→湖州','HB_to_HZ':'湖北→湖州','SA+HB_to_HZ':'浙二＋湖北→湖州',
       'SA_to_HB':'浙二→湖北','HZ_to_HB':'湖州→湖北','SA+HZ_to_HB':'浙二＋湖州→湖北'}
def source(path,name):
    path=Path(path);dest=P/'source_data'/name;dest.parent.mkdir(exist_ok=True)
    shutil.copy2(path,dest);SOURCES[name]={'source':str(path),'sha256':sha(path),'copy_sha256':sha(dest)}
    assert SOURCES[name]['sha256']==SOURCES[name]['copy_sha256']
    return 'source_data/'+name
def get(name):return pd.read_csv(OLD/'tables'/name)

FIGMETA={
'S1_cohort':('S1','队列、配对与质量控制','三个中心的纳入、排除和QC路径。表中病例记录、身份代理分组和真实患者身份核实状态分别列示。湖北保留81条评价记录，其中79条通过信号QC、2条保留在报告分母并拒报。','S1'),
'S2_nine_direction_report':('S2','九方向可信报告工作点','15分钟，九方向、全部20个seed。点为合并计数比例，线为既有95%代理患者分组区间。范围内误放以全部真实低／高MA事件为分母；零报告和无类别支持标NA。工作点不等于同覆盖率排序优势。','S2、S5'),
'S4_paired_adaptation':('S4','适配的配对收益','同病例、同模型、同seed和既定预算，正值表示适配降低误差。此处不比较增加来源中心的作用；异常类别的有益与有害迁移见Table S4。','S3、S4'),
'S5_MA_residuals':('S5','MA残差与参考状态','纵轴为预测−参考MA，横轴为参考MA；每点代表一条QC合格记录的20seed平均残差。50/70mm虚线为参考标签边界。平均误差与异常区间偏差应分别解释。','S3'),
'S7_nine_direction_sequence':('S7','继续观察与首次报告','统一数值MA策略。首次报告锁定，不以晚期正确覆盖早期错误。上图展示累计正确／错误及逐点95%区间，下方状态带显示正确、错误和未报告合计100%。不同湖北B分支不插入本图。','S7、S9'),
'S8_adaptation_budget':('S8','适配样本量敏感性','已有0/C20/FULL预算点，来源模型和测试队列固定；FULL为湖州40、浙二60、湖北40条记录。0为无目标适配，不表示0例训练。未针对这些预算重训报告门。','S8'),
'S9_seed_stability':('S9','全部20个seed的稳定性','显示1000–1019全部seed。零报告seed覆盖率为0，一致率为NA，并列出零报告次数。点的分散体现seed波动，不等同患者抽样置信区间。','S11')}
for page,target in enumerate(['浙二','湖州','湖北'],1):
    FIGMETA[f'S3_mapping_{page}']=('S3',f'完整映射baseline：目标{target}（{page}/3）','六个固定配置；灰色为适配前，方法颜色为最终适配后。误差在拒报前、原QC合格记录上计算。均沿用既有95%代理患者分组区间；中间Huber结果见表，不按外测挑选方法。','S3')
    FIGMETA[f'S6_risk_coverage_{page}']=('S6',f'风险排序与覆盖率：目标{target}（{page}/3）','在相同有限分数候选上作固定配额评价；不是重新选择部署阈值。全体候选与预测范围内候选分开，后者错误即真实异常报范围内。两种风险分支不拼成未经定义的全局分数。空面板明确标为不可评价。','S5')

def render_figures():
    # Reuse established layout code in a new namespace rooted in this output folder.
    text=(OLD/'render.py').read_text(encoding='utf-8')
    text=text.replace("if __name__=='__main__':main()",'')
    text=text.replace("'#416B8A'",repr(PALETTE['dual'])).replace("'#A3ADB5'",repr(PALETTE['error']))
    text=text.replace("['#8596A6','#8B739A','#356E8D']",repr([PALETTE['margin'],PALETTE['error'],PALETTE['dual']]))
    # Fix names after substitutions and leave the scientific data reader read-only.
    env={'__file__':str(P/'render_reused.py'),'__name__':'appendix_style'}
    exec(compile(text,str(OLD/'render.py'),'exec'),env)
    env['get']=get
    env['COLORS']=[PALETTE[k] for k in ['ridge','krr','pls','svr','mlp','tt']]
    env['STYLE'].update({'font.size':8,'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7})
    plt.rcParams.update(env['STYLE'])
    def export(fig,name,title,caption,sources):
        num,zh,cap,tab=FIGMETA[name]
        if name=='S7_nine_direction_sequence':
            plt.close(fig);fig=sequence_figure(env['DIRS'])
        if name=='S4_paired_adaptation':
            for ax,unit in zip(fig.axes,['min','min','mm','deg']):ax.set_xlabel(f'Raw − adapted MAE\n({unit})')
        if name=='S2_nine_direction_report':
            # Preserve missing estimates visibly without drawing a false zero point.
            data=get('A2_A5_gate_working_points.csv')
            for j,key in enumerate(['agreement','coverage','low_to_within_rate_all','high_to_within_rate_all']):
                for i,d in enumerate(env['DIRS']):
                    q=data[(data.direction==d)&data.method.isin(['MA-ERROR-LR','Dual protection'])]
                    if q[key].isna().all():fig.axes[j].text(50,8-i,'NA',ha='center',va='center',fontsize=7,color='#777777')
        if name.startswith('S6_risk_coverage'):
            for ax in fig.axes:
                if not any(np.isfinite(np.asarray(l.get_ydata(),float)).any() for l in ax.lines):
                    ax.text(.5,.5,'Not evaluable',transform=ax.transAxes,ha='center',color='#777777')
                    ax.set_xlim(0,100);ax.set_xticks([0,50,100])
        if name=='S9_seed_stability':
            data=get('B3_gate_seed_values.csv')
            for i,d in enumerate(env['DIRS']):
                q=data[(data.direction==d)&(data.method=='Dual protection')]
                fig.axes[1].text(98,8-i,f"{int((q.coverage==0).sum())}/20 zero",ha='right',va='center',fontsize=6.5,color='#666666')
        for ext in ['svg','pdf','png']:
            fig.savefig(P/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        FIGS.append({'name':name,'number':num,'title':zh,'caption':cap,'tables':tab,
                     'evidence':'数据描述' if num=='S1' else '外部评价／描述性分析','sources':sources,'bootstrap_draws':None if num in ['S1','S9'] else 2000})
    env['export']=export
    env['main']()
    # Old builder creates an intermediate local index only; the final preview replaces it below.
    dump('FIGURE_CAPTIONS.json',FIGS)

def sequence_figure(dirs):
    q=get('A7_pointwise_cumulative.csv');q=q[q['mode']=='cumulative_first_report']
    fig=plt.figure(figsize=(7.2,8.3));outer=fig.add_gridspec(3,3,wspace=.42,hspace=.55)
    ymax=max(5,float(100*max(q.correct_yield.max(),q.incorrect_yield.max())*1.18))
    for i,d in enumerate(dirs):
        grid=outer[i//3,i%3].subgridspec(2,1,height_ratios=[3.8,1.05],hspace=.12)
        ax=fig.add_subplot(grid[0]);st=fig.add_subplot(grid[1]);z=q[q.direction==d].sort_values('horizon');x=z.horizon.to_numpy()/60
        ax.set_title(chr(97+i)+'  '+d.replace('_to_',' → ').replace('+',' + '),loc='left',fontsize=8,pad=9)
        for key,color,m in [('correct_yield',PALETTE['correct'],'o'),('incorrect_yield',PALETTE['incorrect'],'s')]:
            ax.plot(x,z[key]*100,m+'-',ms=3,lw=1,color=color)
            ax.fill_between(x,z[key+'_ci_low']*100,z[key+'_ci_high']*100,color=color,alpha=.13)
        ax.set(ylim=(0,ymax),xticks=[5,10,15],ylabel='First report (%)');ax.tick_params(labelbottom=False)
        un=z.withheld.to_numpy()/z.events.to_numpy();assert np.allclose(z.correct_yield+z.incorrect_yield+un,1)
        st.bar(x,z.correct_yield*100,width=2,color=PALETTE['correct'])
        st.bar(x,z.incorrect_yield*100,bottom=z.correct_yield*100,width=2,color=PALETTE['incorrect'])
        st.bar(x,un*100,bottom=(z.correct_yield+z.incorrect_yield)*100,width=2,color=PALETTE['withheld'])
        st.set(ylim=(0,100),xticks=[5,10,15],yticks=[0,100],xlabel='Observation (min)')
        st.tick_params(labelsize=6);st.set_ylabel('State (%)',fontsize=6)
    handles=[Line2D([],[],marker=m,color=PALETTE[k],label=l) for k,m,l in [('correct','o','Correct'),('incorrect','s','Incorrect'),('withheld','s','Unreported')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005),fontsize=8)
    fig.subplots_adjust(bottom=.09,top=.95,left=.09,right=.99)
    return fig

LABELS={'direction':'方向','method':'方法','stage':'阶段','arm':'分支','horizon':'时点（min）','center':'中心',
 'records':'记录数','identity_proxy_groups':'身份代理组','patient_identity_verified':'患者身份已核实','QC_pass':'QC通过','QC_fail':'QC未通过',
 'low':'低MA','within':'范围内','high':'高MA','role':'数据角色','events':'记录×seed事件','reported':'已报告','correct':'正确报告','incorrect':'错误报告','withheld':'未报告',
 'agreement':'一致率','coverage':'覆盖率','within_reports':'范围内报告','low_support':'真实低MA事件','high_support':'真实高MA事件',
 'low_to_within':'低MA→范围内','high_to_within':'高MA→范围内','low_correct':'正确低MA','high_correct':'正确高MA','correct_low':'正确低MA','correct_high':'正确高MA',
 'normal_to_tail':'范围内→异常','low_high_cross':'低高互判','zero_report_seeds':'零报告seed','R_MAE':'R MAE（min）','K_MAE':'K MAE（min）','MA_MAE':'MA MAE（mm）','alpha_MAE':'α MAE（°）',
 'truth':'参考类别','before':'适配前类别','after':'适配后类别','n':'n','N':'N','proportion':'比例','calibration_n':'适配记录数',
 'contrast':'比较','metric':'指标','estimate':'差值估计','ci_low':'95%下界','ci_high':'95%上界','parameter':'参数','mode':'策略',
 'first_report_seconds':'首次报告（min）','new_correct':'新增正确','new_incorrect':'新增错误','population':'候选群体',
 'requested_fraction':'固定配额','seed':'seed','unit':'单位','median':'中位数','p05':'P05','p95':'P95','measurement':'计时范围',
 'repeats':'次数','task':'任务','contexts':'上下文数','median_seconds':'中位耗时（s）','p05_seconds':'P05（s）','p95_seconds':'P95（s）','scope':'范围',
 'low_to_within_rate_all':'低MA误放／全部低MA','high_to_within_rate_all':'高MA误放／全部高MA','correct_yield':'正确报告／全部','incorrect_yield':'错误报告／全部',
 'report_class':'报告去向','true_class':'真实类别','destination':'去向','count':'计数'}
LABELS.update({'fraction':'比例','reported_class':'报告去向','oracle_coverage':'理想覆盖上限','correct_abnormal_available':'门前正确异常数','correct_abnormal_reported':'正确异常放行数','no_high_support_seeds':'无高MA支持seed','low_fraction_within_reports':'低MA／范围内报告','high_fraction_within_reports':'高MA／范围内报告'})
RATIOS={'agreement','coverage','proportion','fraction','requested_fraction','correct_yield','incorrect_yield','low_recall','high_recall','within_recall','normal_fpr','tail_recall','BA','oracle_coverage','score_max_coverage','low_to_within_rate_all','high_to_within_rate_all'}
def format_value(k,v,row):
    if pd.isna(v):return 'NA'
    if k=='direction':return DIRZH.get(str(v),str(v))
    if k=='center':return {'SAHZU':'浙二','Huzhou':'湖州','Hubei':'湖北'}.get(str(v),str(v))
    if k in ['horizon','first_report_seconds']:return str(int(v)//60) if float(v)>0 else '未报告'
    if isinstance(v,(bool,np.bool_)):return '是' if v else '否'
    if isinstance(v,(float,int,np.number)):
        if k in RATIOS or '_rate_' in k or '_fraction_' in k or k.endswith('_fraction') or (k.endswith(('_ci_low','_ci_high')) and k.rsplit('_ci_',1)[0] in RATIOS):return f'{100*v:.2f}%'
        if k in ['estimate','ci_low','ci_high'] and 'metric' in row and any(x in str(row['metric']) for x in ['agreement','coverage','yield','rate','proportion']):return f'{100*v:+.2f} pp'
        if float(v).is_integer():return str(int(v))
        return f'{v:.3f}'
    return str(v)
def table(num,part,title,path,cols=None,note='',evidence='外部评价',derived=None):
    df=pd.read_csv(path) if derived is None else derived
    filename=f'Table_{num}{part}.csv'
    if derived is None:link=source(path,filename)
    else:
        dest=P/'source_data'/filename;df.to_csv(dest,index=False,encoding='utf-8-sig');link='source_data/'+filename
        SOURCES[filename]={'derived_from_existing_results':str(path),'sha256':sha(dest),'no_new_experiment':True}
    columns=[c for c in (cols or list(df.columns)) if c in df.columns]
    header=''.join(f'<th>{esc(LABELS.get(c,c))}</th>' for c in columns)
    rows=''.join('<tr>'+''.join(f'<td>{esc(format_value(c,r[c],r))}</td>' for c in columns)+'</tr>' for _,r in df.iterrows())
    key='Table'+num+part;TABLES.append({'number':num,'part':part,'id':key,'title':title,'evidence':evidence,'rows':len(df),'columns_displayed':columns,'file':link,'note':note})
    return f'<details class="table" id="{key}"><summary><b>Table {num}{part}</b>　{esc(title)} <span class="count">{len(df)}行 · {esc(evidence)}</span></summary><p>{esc(note)} <a href="{link}" download>下载完整CSV</a></p><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div></details>'
def fig(name):
    f=next(x for x in FIGS if x['name']==name)
    return f'<figure id="{name}"><figcaption><span class="eyebrow">Figure {f["number"]}</span><h3>{esc(f["title"])}</h3></figcaption><a href="figures/{name}.svg" target="_blank"><img loading="lazy" src="figures/{name}.svg" alt="{esc(f["title"])}"></a><p class="caption">{esc(f["caption"])} 配套Table {esc(f["tables"])}。</p><div class="downloads"><a href="figures/{name}.png" download>PNG</a><a href="figures/{name}.svg" download>可编辑SVG</a><a href="figures/{name}.pdf" download>PDF</a></div></figure>'
def section(code,title,question,finding,limit,body,evidence='外部评价'):
    return f'<section id="{code}"><div class="sectionhead"><span class="sectionnum">{code}</span><div><span class="badge {"dev" if "开发" in evidence else ""}">{esc(evidence)}</span><h2>{esc(title)}</h2><p>{esc(question)}</p></div></div><div class="finding"><b>主要认识</b> {esc(finding)}</div>{body}<aside><b>解读边界</b> {esc(limit)}</aside></section>'

def build_html():
    t=OLD/'tables';parts=[]
    body=fig('S1_cohort')+table('S1','a','中心特征与QC',t/'A1_center_QC.csv',note='类别支持按固定参考MA标签统计；QC失败与病例排除不同。',evidence='数据描述')+table('S1','b','九方向数据角色',t/'A1_data_roles.csv',note='来源、适配和测试角色分别列示；共享目标队列不是独立重复验证。',evidence='数据描述')
    parts.append(section('A1','三中心数据、配对与质量控制','哪些记录参与研究，各自承担什么角色？','保留配对、QC及数据角色的完整追溯。','患者身份代理不等于已独立核实的患者ID；部分仪器与标本字段仍待作者核实。',body,'数据描述'))
    cols=['direction','agreement','coverage','reported','correct','low_to_within','low_support','high_to_within','high_support','low_correct','high_correct','zero_report_seeds','R_MAE','K_MAE','MA_MAE','alpha_MAE']
    parts.append(section('A2','九方向完整外部评价','正文两方向在完整跨中心结果中处于什么位置？','同一数值MA流程完整呈现九方向，保留低覆盖和零报告方向。','不存在九方向均有效的结论；湖州高MA无支持记NA。湖北→浙二新探索分支不替换此总表。',fig('S2_nine_direction_report')+table('S2','','九方向统一流程总表',t/'A2_nine_direction_summary.csv',cols,note='15分钟；20seed合并计数。区间与完整去向列在下载数据及Table S5/S6。')))
    mappingcols=['direction','method','stage','calibration_n','R_MAE','K_MAE','MA_MAE','alpha_MAE','agreement','low_correct','low_support','high_correct','high_support','normal_to_tail']
    body=''.join(fig('S3_mapping_'+str(i)) for i in [1,2,3])+fig('S5_MA_residuals')+table('S3','','完整映射baseline与参数误差',t/'A3_mapping_baselines_FULL.csv',mappingcols,note='raw_med4=适配前；pure_huber_med4=Huber后；final_med4=最终适配。全部在门控前、相同QC口径评价；下载数据保留四参数区间。')
    parts.append(section('A3','完整映射baseline与MA残差','已有小模型的转译性能如何，异常区间是否仍有偏差？','固定配置、匹配条件下比较六个方法；同时展示平均误差和MA状态相关残差。','不按测试成绩挑选使用方法，不预设Ridge每项第一；平均误差较小不等于异常保留。',body))
    body=fig('S4_paired_adaptation')+table('S4','a','适配前后配对误差差值',t/'A4_adaptation_paired.csv',note='差值为适配前MAE−适配后MAE，正值有利；R/K为min，MA为mm，α为度。')+table('S4','b','适配前后完整类别迁移',t/'A4_adaptation_class_flow.csv',note='truth/before/after类别编码：−1低MA、0范围内、1高MA；事件为记录×seed，不是患者数。')
    parts.append(section('A4','适配收益与类别迁移','平均误差改善时，哪些病例被纠正、哪些异常被压回范围内？','九方向MA平均误差均下降，异常状态的变化并非始终有利。','目标适配作用与增加来源中心的作用分别比较；参数绝对误差引用Table S3，不重复列另一份。',body))
    gatecols=['direction','method','agreement','coverage','correct','reported','within_reports','low_to_within','low_support','high_to_within','high_support','low_correct','high_correct','normal_to_tail','zero_report_seeds']
    body=''.join(fig('S6_risk_coverage_'+str(i)) for i in [1,2,3])+table('S5','a','可信门实际工作点',t/'A2_A5_gate_working_points.csv',gatecols,note='全部报告、MA边界距离、原风险门和双侧保护；相同上游与类别决策。区间保留在完整CSV。')+table('S5','b','同覆盖率的配对差值',t/'A5_matched_coverage.csv',note='相对原MA-ERROR-LR的配对差值；比例差显示为百分点pp。all与predicted_within候选集合分开，不能跨集合比较。')
    parts.append(section('A5','可信门baseline与覆盖率取舍','更高放行一致率来自更好排序，还是更严格拒绝？','实际部署工作点与同覆盖率比较并列，区分筛选强度与排序能力。','没有普遍优于所有baseline的结论；未通过开发验收的门仅在B2呈现开发负结果。',body))
    body=table('S6','','真实类别与全部报告去向',t/'A6_complete_class_destinations.csv',note='低／范围内／高三类分别到低／范围内／高／未报告。编码−1/0/1代表低／范围内／高。此表N为全部该真实类别事件；已放行该类和全部范围内报告两种条件风险另见Table S5a完整CSV的rate_released与fraction_within_reports列。NA不替换为0%。')
    parts.append(section('A6','异常去向与风险分母','异常究竟被识别、误放，还是仅被拒绝？','完整去向解释风险，拒报和正确异常报告不是同一个终点。','每个比例必须连同分母解释；不能用全部患者的大分母稀释某一异常类别的误放。',body))
    seqcols=['direction','mode','horizon','events','correct','incorrect','withheld','agreement','coverage','new_correct','new_incorrect','low_to_within','high_to_within']
    body=fig('S7_nine_direction_sequence')+table('S7','a','单时点与累计首次报告',t/'A7_pointwise_cumulative.csv',seqcols,note='pointwise=该时点单独报告；cumulative_first_report=首次报告锁定。正确＋错误＋未报告=完整分母；区间在完整CSV。')+table('S7','b','首次报告时间分布',t/'A7_first_report_times.csv',note='未报告独立列出，不作为15分钟成功报告。')
    parts.append(section('A7','继续观察的完整结果','再观察5分钟增加多少正确、错误报告，又留下多少未报告？','展示实际新增收益和等待代价，不把累计曲线上升本身当作策略优越。','离线回放不是临床实时部署；统一MA序贯与湖北B辅助策略保持独立身份。',body))
    body=fig('S8_adaptation_budget')+table('S8','','已有预算点的完整评价',t/'B1_calibration_budget.csv',['direction','calibration_n','R_MAE','K_MAE','MA_MAE','alpha_MAE','agreement','low_correct','low_support','high_correct','high_support','normal_to_tail'],note='0/C20/FULL，同来源模型、同测试集；比较参数与门前三态，不宣称对应的门控预算实验已完成。')
    parts.append(section('B1','适配样本量与资源需求','需要多少目标中心配对记录，已有预算点带来什么收益？','三个既有预算水平足以呈现适配收益的变化，不追加密集搜索。','不据三个点宣称最优预算；FULL实际为C40/C60/C40，以冻结清单为准。',body))
    body=table('S9','a','全部报告、原门与双侧保护',t/'B2_gate_paired_differences.csv',note='配对差值采用原表contrast方向；比例差为百分点。')
    body+=table('S9','b','序贯首次报告与固定时点',t/'B2_sequence_fixed_paired.csv',note='同队列、同分母；seq−fixed，新增正确与新增错误均保留，不能用后续纠正冲销首次错误。')
    body+=table('S9','c','单来源与双来源配置',t/'B2_source_configuration_paired.csv',note='双来源−单来源，不解释成已控制样本量的中心多样性因果效应。')
    body+=table('S9','d','湖北B辅助系统策略',t/'B2_Hubei_system_strategy.csv',note='不同分类和报告策略，属于系统组合对照，不替换A7统一MA策略。')
    body+=table('S9','e','湖北B策略相对统一MA的配对差值',t/'B2_Hubei_strategy_paired.csv',note='策略同时改变分类和报告时点，不称单组件消融。')
    overview=pd.DataFrame([
      ['三档MA适配强度','嵌套开发评价','7方向均未稳定兼顾两侧','未进入新门／外测','统一减弱适配不是通用方案'],
      ['跨边界分侧半量保护','开发筛查→嵌套评价','湖北→浙二通过上游筛查；湖州→湖北未稳定通过','湖北→浙二继续匹配原门','恢复异常同时增加范围内误报'],
      ['分侧保护后重新匹配原门','开发工作点诊断','正确异常增加，但覆盖率下降','该完整重匹配分支未外测','筛查池选点表现不是外测'],
      ['保留原范围内通道','开发选点＋冻结外测','覆盖及正确高MA增加；一致率下降，低MA略有损失','外测完成，单列探索性对照','异常误放没有减少'],
      ['原数值MA七方向可改善空间','开发侧oracle诊断','区分正确异常不足、排序不足、支持限制','不部署oracle、不反转分数','理想上界不是可达到性能保证'],
      ['已有PLS／SVR嵌套复用','嵌套开发评价','7方向均未获得稳定增量','未训练新门、未新增外测','低MA局部收益区间跨零'],
      ['既往门／信号候选','已有开发记录','低维LR、HGB、OOF对齐、校准差值等未形成稳定增量','保留机制解释，不逐轮凑图','部分使用MA/B分支，不能混作原数值MA基线']
    ],columns=['探索','证据层级','主要结果','验证终点','必要说明'])
    body+=table('S10','a','探索结果与验证边界',GOAL/'WRITING_READINESS_ZH.md',derived=overview,evidence='开发与外测分列',note='每项均注明验证层级；停止于开发的候选没有外测数字，不以空白暗示未展示的成功。')
    body+=table('S10','b','三档强度：开发侧完整结果',STRENGTH/'summary.csv',note='门控前，全部20seed；固定候选和内层选择分开，不能从本表反选测试最有利策略。',evidence='开发侧')
    body+=table('S10','c','分侧保护：嵌套开发评价',NEST/'summary.csv',note='10pp正常误报容忍在本轮运行前冻结；低、高MA和代价分别报告。',evidence='开发侧')
    body+=table('S10','d','保留范围内通道：开发选点诊断',DEV/'summary.csv',['arm','agreement','coverage','correct','reported','within_reports','low_to_within','low_support','high_to_within','high_support','correct_low','correct_high','normal_to_tail','zero_report_seeds'],note='选工作点与描述在同一开发池，非独立外测。BASE原流程；SELECTED完整重匹配；PRESERVED_WITHIN保留原范围内。',evidence='开发侧选点')
    body+=table('S10','e','保留范围内通道：冻结外部评价',EXT/'summary.csv',['arm','agreement','coverage','correct','reported','within_reports','low_to_within','low_support','high_to_within','high_support','correct_low','correct_high','normal_to_tail','zero_report_seeds'],note='湖北→浙二，201记录×20seed。低MA误放18/140、高MA21/480均未减少。区间为既有10,000次代理组bootstrap，见附带统计说明。',evidence='冻结外测 · 探索性')
    body+=table('S10','f','七方向门前供给与可信层空间',GOAL/'development_headroom_summary.csv',['direction','coverage','oracle_coverage','correct_abnormal_available','correct_abnormal_reported','within_reports','zero_report_seeds','no_high_support_seeds'],note='固定原范围内通道；oracle用真实正确性计算，只是有限样本诊断，绝不用于部署阈值。',evidence='开发侧oracle')
    body+=table('S10','g','既有PLS／SVR的嵌套复用',GOAL/'mapper_development_summary.csv',note='SELECTED仅在每折内层OOF选择、外层评价；固定方法行用于解释，不替代嵌套选择结果。没有方向通过冻结晋级条件。',evidence='嵌套开发侧')
    # Preserve the earlier fixed-side screen and the two decisive interval files, without extra logical tables.
    source(CROSS/'summary.csv','S10_fixed_side_screen.csv')
    source(EXT/'PAIRED_INTERVALS.json','S10_external_intervals.json')
    source(GOAL/'MAPPER_REUSE_RESULT.json','S10_nested_decisions.json')
    source(OLD/'HGB_DEVELOPMENT_NEGATIVE_REUSE.json','S10_HGB_development_negative.json')
    body+='<p class="minor">完整探索附件：<a href="source_data/S10_fixed_side_screen.csv">分侧固定策略筛查</a> · <a href="source_data/S10_external_intervals.json">外测配对区间</a> · <a href="source_data/S10_nested_decisions.json">嵌套晋级判断</a> · <a href="source_data/S10_HGB_development_negative.json">HGB开发负结果</a></p>'
    parts.append(section('B2','功能贡献与异常保护探索','哪些功能有作用，哪些最小改动没有获得稳定增量？','有收益与无收益结果按机制并列：湖北→浙二保留通道扩大覆盖，但没有消除异常误放。','探索分支与统一流程严格区分；跨模块变化称系统策略，开发选点数字不写成外测结果。',body,'功能对照＋开发／外测探索'))
    body=fig('S9_seed_stability')+table('S11','a','全部seed的可信报告',t/'B3_gate_seed_values.csv',gatecols+['seed'],note='保留所有1000–1019 seed；零报告一致率NA。病例/身份组数及聚合区间见Table S1/S2/S5。')
    body+=table('S11','b','全部seed的参数映射',t/'B3_mapping_seed_values.csv',note='全部seed及方法完整保留；神经模型seed变化与确定性源拟合复用见方法说明。')
    body+=table('S11','c','全部seed的时序报告',t/'B3_sequence_seed_values.csv',seqcols+['seed'],note='seed散布不替代患者代理组重采样区间；本表保留零报告和早期错误。')
    parts.append(section('B3','稳定性、支持数与不确定性','平均结果是否由少数运行或稀少类别支撑？','完整seed分布与类别支持共同限定结果可信度。','seed不是独立患者。原包2,000次与新增探索10,000次bootstrap分别标注，不改变旧区间以制造统一。',body))
    body=table('S12','a','当前计算阶段耗时',t/'B4_runtime_components.csv',note='同硬件warm微基准，不包括设备采集时间，不推断临床部署时延。',evidence='计算计时')
    body+=table('S12','b','双侧范围内风险判断耗时',t/'B4_dual_head_runtime.csv',note='只计双风险头，避免与包含它的全流程计时重复相加。',evidence='计算计时')
    body+=table('S12','c','离线训练与适配任务耗时',t/'B4_fit_context_runtime.csv',note='既有嵌套任务总耗时，不等同单病例在线推理。',evidence='历史任务计时')
    body+=table('S12','d','历史信号优化计时',t/'B4_archived_signal_optimizer_timings.csv',note='历史计时未在当前硬件统一重测，且不完整包含预处理；不称设备级真实时延。',evidence='历史计时')
    parts.append(section('B4','运行效率与复现','计算与配对标签需要哪些资源，如何复查？','分阶段给出已有计时、输入身份和复现入口，不为效率再跑模型。','真实患者身份、部分仪器／标本字段及前瞻性时序记录仍需作者核实。该版可用于审阅，不是临床安全或投稿合规认证。',body,'资源与复现'))
    nav='<a href="#readme">阅读口径</a><b>Ⅰ 数据与完整评价</b>'+''.join(f'<a href="#{c}">{c} {label}</a>' for c,label in [('A1','数据与QC'),('A2','九方向'),('A3','映射baseline'),('A4','适配收益'),('A5','可信门'),('A6','异常去向'),('A7','继续观察')])+'<b>Ⅱ 组件与可靠性</b>'+''.join(f'<a href="#{c}">{c} {label}</a>' for c,label in [('B1','样本量'),('B2','功能与探索'),('B3','稳定性'),('B4','效率与复现')])
    intro='''<header id="readme"><span class="eyebrow">U-TEG · SUPPLEMENTARY EVIDENCE · REVIEW EDITION</span><h1>附录：完整评价与可靠性</h1><p class="lead">两大块 · 11节 · 9组补充图（13页）· 12组补充表。只整理已有证据，不新增实验，不替换正文图表。</p><div class="legend"><span style="--c:#356E8D">双侧保护</span><span style="--c:#8596A6">原风险门</span><span style="--c:#258777">正确报告</span><span style="--c:#BC6358">错误报告</span><span style="--c:#A8AFB5">未报告</span></div><div class="stats"><h3>统一统计口径</h3><p>一致率＝正确／已报告；覆盖率＝已报告／完整评价分母。异常误放分别以全部真实低MA、高MA为分母；其他条件分母在表内注明。MA＜50／50–70／＞70；范围内不是“所有临床意义均正常”。</p><p>保留全部20个seed。记录数、身份代理组和记录×seed事件分别解释；零报告的一致率和无类别支持的风险均为NA。参数误差沿用原QC合格分母，报告评价保留QC失败并拒报。</p><p>区间均沿用已有描述性95%身份代理组bootstrap，重复seed随同组共同重采样：原附录2,000次，近期探索10,000次。这里统一定义与标注，不重算或改写旧区间。未作普适优越或临床安全保证。</p><p>图中SA＝浙二，HZ＝湖州，HB＝湖北。蓝色标签为外部评价；棕色标签为开发侧或探索阶段。Table S10显式分开两者。既有外部队列被反复使用，新增分支属于探索性外部评价。</p></div><p class="minor">使用方法：图可点开查看矢量原图；表格默认折叠，点击标题展开全部行，横向滚动查看。CSV保留完整列与原精度，网页仅格式化显示。<a href="TABLE_FIGURE_INDEX.json">图表索引</a> · <a href="REVIEW_NOTES.md">版本说明</a></p></header>'''
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f7;color:#23313c;font:16px/1.65 Arial,"Microsoft YaHei",sans-serif}nav{position:fixed;top:0;left:0;width:215px;height:100vh;overflow:auto;padding:28px 18px;background:#132d3d;color:#dbe5ec}nav a{display:block;color:#dbe5ec;text-decoration:none;padding:5px 8px;font-size:14px}nav b{display:block;font-size:12px;color:#91aabb;margin:22px 8px 8px}nav a:hover{background:#25485f;border-radius:5px}main{margin-left:215px;max-width:1320px;padding:40px 45px}header,section{background:white;border:1px solid #dde4e9;border-radius:12px;padding:34px;margin-bottom:28px;scroll-margin-top:18px}h1{font-size:35px;line-height:1.3;letter-spacing:-.5px;margin:10px 0 18px}h2{font-size:25px;line-height:1.4;margin:8px 0}h3{font-size:19px;margin:5px 0 16px}.eyebrow{font-size:12px;letter-spacing:1.4px;color:#517187;font-weight:bold}.lead{font-size:18px;color:#516473}.sectionhead{display:flex;gap:20px;align-items:flex-start}.sectionnum{font-size:30px;font-weight:bold;color:#356e8d;border-bottom:3px solid #356e8d}.sectionhead p{margin-top:5px;color:#61727f}.badge{font-size:12px;color:#356e8d;background:#eaf1f5;border-radius:20px;padding:4px 9px}.badge.dev{color:#8b612f;background:#f7efdf}.finding{padding:14px 18px;background:#edf4f3;border-left:3px solid #258777;margin:18px 0 28px}.stats{background:#f5f7f9;padding:20px 24px;border-radius:8px}.stats p{font-size:14px}figure{margin:28px 0;padding:20px 0;border-top:1px solid #e4e9ec}figure img{width:100%;height:auto;background:white;display:block}.caption{font-size:14px;color:#536570}.downloads a,.minor a{color:#356e8d;margin-right:10px}.downloads{font-size:13px}.table{margin:14px 0;border:1px solid #dce4e9;border-radius:7px;background:#fafcfd}.table summary{cursor:pointer;padding:14px 16px;font-size:15px}.table p{padding:0 16px;font-size:13px;color:#5f6f7b}.count{color:#7c8a93;font-size:12px}.scroll{overflow:auto;max-height:560px;border-top:1px solid #e1e7eb}table{border-collapse:collapse;min-width:100%;font-size:12px;line-height:1.5}th,td{padding:9px 12px;border-bottom:1px solid #e7ecef;text-align:right;white-space:nowrap}th{background:#eaf0f4;position:sticky;top:0;color:#2d4859;z-index:1}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f4f7f9}aside{background:#f8f5ee;border-left:3px solid #b9926c;padding:16px 20px;margin-top:24px;font-size:14px}.minor{font-size:13px;color:#697b88}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}.legend span{font-size:13px}.legend span:before{content:"";display:inline-block;width:11px;height:11px;background:var(--c);border-radius:2px;margin-right:6px}.blocktitle{font-size:22px;padding:12px 8px;color:#356e8d}.footer{font-size:13px;padding:20px;color:#6b7c87}@media(max-width:900px){nav{position:static;width:auto;height:auto;display:flex;flex-wrap:wrap;padding:12px}nav b{display:none}main{margin:0;padding:16px}header,section{padding:20px}h1{font-size:28px}}@media print{nav,.downloads{display:none}main{margin:0;padding:0}body{background:white}header,section{border:0;padding:12px;break-before:auto}figure{break-inside:avoid}.scroll{max-height:none;overflow:visible}details:not([open]){display:block}}'''
    body='<h2 class="blocktitle">Ⅰ　数据与完整评价</h2>'+''.join(parts[:7])+'<h2 class="blocktitle">Ⅱ　组件贡献、敏感性与可靠性</h2>'+''.join(parts[7:])
    (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>U-TEG 附录审阅版</title><style>'+css+'</style></head><body><nav><h3>附录审阅版</h3>'+nav+'</nav><main>'+intro+body+'<p class="footer">2026-10-28 · 已有证据整理版 · 正文与既有结果保持不变</p></main></body></html>',encoding='utf-8')
    dump('TABLE_FIGURE_INDEX.json',{'sections':['A1','A2','A3','A4','A5','A6','A7','B1','B2','B3','B4'],'figures':FIGS,'tables':TABLES,'palette':PALETTE})

def main():
    assert not (P/'DELIVERY_MANIFEST.json').exists(),'Do not overwrite a finalized edition'
    for name in ['source_data','figures']: (P/name).mkdir(exist_ok=True)
    protected=read(EXT/'FREEZE.json')['protected'];oldmanifest=read(OLD/'DELIVERY_MANIFEST.json')['files']
    inputs={**oldmanifest,**protected}
    extra=[DEV/'summary.csv',EXT/'summary.csv',EXT/'PAIRED_INTERVALS.json',GOAL/'mapper_development_summary.csv',GOAL/'development_headroom_summary.csv',GOAL/'MAPPER_REUSE_RESULT.json',STRENGTH/'summary.csv',NEST/'summary.csv',CROSS/'summary.csv']
    inputs.update({str(p):sha(p) for p in extra})
    assert all(sha(p)==h for p,h in inputs.items())
    dump('INPUT_FREEZE.json',{'inputs':inputs,'main_protected_count':len(protected),'new_fits':0,'new_statistics':0,'purpose':'presentation only'})
    render_figures();build_html()
    assert all(sha(p)==h for p,h in inputs.items())
    dump('SOURCE_MANIFEST.json',SOURCES)
    print('ASSEMBLED',len(FIGS),'figure pages;',len(set(t['number'] for t in TABLES)),'table groups;',len(TABLES),'table parts')
if __name__=='__main__':main()
