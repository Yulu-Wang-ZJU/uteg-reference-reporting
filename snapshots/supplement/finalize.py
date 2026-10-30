"""Independent count verification, resource measurements and evidence inventory."""
import sys,time,platform,ast
from pathlib import Path
import numpy as np,pandas as pd,joblib
import batch as b
P=b.ROOT;O=b.OUT
def csv(name,rows):
    pd.DataFrame(rows).to_csv(P/'tables'/name,index=False,encoding='utf-8-sig')
def zload(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def truth_ma(x):return np.select([x<50,x>70],[-1,1],default=0)
def audit():
    plan=b.read(P/'PLAN.json');frozen=b.read(P/'NEW_PREDICTIONS_FROZEN.json')
    for path,h in plan['immutable_hashes'].items():assert b.sha(path)==h,('changed historical artifact',path)
    for path,h in frozen['files'].items():assert b.sha(path)==h
    D=b.data_at(900);gates=pd.read_csv(P/'tables/A2_A5_gate_working_points.csv');mapping=pd.read_csv(P/'tables/A3_mapping_comparison.csv');times=pd.read_csv(P/'tables/A7_pointwise_cumulative.csv')
    checked=0
    for d in b.TARGET:
        zz=zload(b.OLD/'eval/predictions'/f'{d}.npz');idx=zz['test_global_index'];yy=truth_ma(D['medical'][idx,2]);act=zload(P/'predictions'/f'gate_actions__{d}.npz')
        for name,key in [('All report','all_report'),('MA margin','margin'),('MA-ERROR-LR','lr'),('Dual protection','dual')]:
            a=act[key];pred=act['pred'];row=gates[(gates.direction==d)&(gates.method==name)].iloc[0]
            good=bad=low=high=0
            for s in range(20):
                for j in range(len(idx)):
                    if a[s,j]:
                        if pred[s,j]==yy[j]:good+=1
                        else:bad+=1
                        if pred[s,j]==0:
                            low+=int(yy[j]==-1);high+=int(yy[j]==1)
            assert good==row.correct and bad==row.incorrect and low==row.low_to_within and high==row.high_to_within
            assert row.reported==good+bad and row.events==len(idx)*20
            checked+=1
        s=zload(P/'predictions'/f'sequence__{d}.npz')
        for h in [300,600,900]:
            a=(s['first_time']>0)&(s['first_time']<=h);pred=s['first_class'];good=int(np.count_nonzero(a&(pred==yy)))
            rr=times[(times.direction==d)&(times['mode']=='cumulative_first_report')&(times.horizon==h)].iloc[0]
            assert rr.correct==good and rr.reported==np.count_nonzero(a) and rr.withheld==a.size-np.count_nonzero(a)
            checked+=1
        for name in ['PLS-2','SVR-1','MLP-8','TT-SMALL','RIDGE-10-C20']:
            arm='budget_C20' if name=='RIDGE-10-C20' else 'mapping_'+name
            z=zload(P/'predictions'/f'{arm}__{d}.npz');q=D['input_qc'][idx]
            for stage in ['raw_med4','pure_huber_med4','final_med4']:
                rr=mapping[(mapping.direction==d)&(mapping.method==name)&(mapping.stage==stage)].iloc[0]
                for j,p in enumerate(['R','K','MA','alpha']):
                    errors=[abs(float(z[stage][s,i,j])-float(D['medical'][idx[i],j])) for s in range(20) for i in range(len(idx)) if q[i]]
                    assert np.isclose(np.mean(errors),rr[p+'_MAE'],rtol=1e-10)
                    checked+=1
    ledger=[];fits=0
    for task in plan['tasks']:
        root=P/'runs'/task['id'];complete=b.read(root/'COMPLETE.json');assert complete['count']==20
        total=0;failed=0
        for fp in (root/'ledger').glob('*.json'):
            r=b.read(fp);total+=r['units'];failed+=r['status']=='FAILED';assert r['status']!='RUNNING',fp
        assert total<=task['cap'];fits+=total
        ledger.append(dict(task=task['id'],actual_fit_units=total,formal_upper=task['science_upper'],cap=task['cap'],failed_fit_attempts=failed))
    csv('FIT_LEDGER_SUMMARY.csv',ledger)
    b.write(P/'INDEPENDENT_VERIFICATION.json',{'status':'PASS','independently_checked_metric_cells':checked,'fitted_tasks':len(ledger),'actual_fit_units':fits,'formal_fit_upper':plan['formal_fit_upper'],'all_20_seeds':True,'immutable_files_checked':len(plan['immutable_hashes']),'old_main_content_unchanged':True,'new_test_model_selection':False,'note':'Counting and error arithmetic implemented separately from analyze.py. Bootstrap code not independently reimplemented.'})

def resources():
    # Warm, single-record timing on fixed saved models; no fitting or threshold changes.
    r,runner,t,seq,c,sel=b.env('efficiency_readonly',1);D=b.data_at(900);rows=[]
    for d in ['HZ+HB_to_SA','SA+HB_to_HZ','SA+HZ_to_HB']:
        obj=joblib.load(b.old_path(d,1000));i=obj['pool'][0];x=D['x'][[i]];src=obj['source_model'];ca=obj['cache'][obj['positions'][3]['base']];ad=obj['final_selection']['adapter'];gate=obj['gate_models'][3]['model']
        raw=t.m.predict(src,x);pure=t.m.calibrated(ca['cal'],raw)
        def adapt():
            h=t.m.calibrated(ca['cal'],raw)
            return h if ad=='H' else t.m.ans['adapt'](ca['adapters'][ad],x,raw,h,src)
        med=adapt();pc=runner.ma_class(med[:,2]);f=runner.ma_features(x,med,src['ym'],src['ys'])
        funcs={'mapping_only':lambda:t.m.predict(src,x),'adaptation_only':adapt,'LR_risk_score_only':lambda:runner.score_ma_gate(gate,f,pc),'decision_comparison_only':lambda:np.asarray([.3])<=.5}
        for name,fn in funcs.items():
            for _ in range(10):fn()
            samples=[]
            for _ in range(100):
                start=time.perf_counter_ns();fn();samples.append((time.perf_counter_ns()-start)/1e6)
            rows.append(dict(direction=d,stage=name,repeats=100,unit='milliseconds',median=float(np.median(samples)),p05=float(np.quantile(samples,.05)),p95=float(np.quantile(samples,.95)),measurement='warm CPU; one calibration record; excludes loading and acquisition'))
    # Reuse archived optimizer timings instead of inventing signal-fit latency.
    hist=[]
    for p in (b.BASE/'prefix_signal_fits').glob('*.joblib'):
        o=joblib.load(p)
        if 'starts' in o and all('seconds' in v for v in o['starts']):
            hist.append(dict(center=o.get('center'),horizon=o.get('horizon'),seconds=sum(v['seconds'] for v in o['starts']),stages=len(o['starts'])))
    csv('B4_runtime_components.csv',rows);csv('B4_archived_signal_optimizer_timings.csv',hist)
    training=[]
    plan=b.read(P/'PLAN.json')
    for task in plan['tasks']:
        values=[b.read(p)['fit_seconds'] for p in (P/'runs'/task['id']/'frozen').glob('*.json')]
        training.append(dict(task=task['id'],contexts=len(values),median_seconds=float(np.median(values)),p05_seconds=float(np.quantile(values,.05)),p95_seconds=float(np.quantile(values,.95)),scope='source when uncached + nested adaptation; prefix includes LR fitting but excludes later dual-head fitting; includes serialization; not additive to component microbenchmarks'))
    csv('B4_fit_context_runtime.csv',training)

def inventory():
    result=b.read(P/'INDEPENDENT_VERIFICATION.json');plan=b.read(P/'PLAN.json')
    items=[
      ('A1','数据、配对与QC','S1；A1中心/QC、角色表','三中心；原始配对规则和冻结划分','A1_center_QC.csv; A1_data_roles.csv','B 已重算','真实患者身份、部分仪器型号/标本字段仍未独立核实；不构成新增拟合缺口','正式投稿前作者核对来源元数据，不推测数值','0'),
      ('A2','九方向完整评价','S2；九方向总表','9方向，900秒，20seed，C40/C60/C40','A2_nine_direction_summary.csv','A/B 已复用并重算','无计算缺口；湖州高MA分母0须NA','保留全部方向及类别支持','0'),
      ('A3','映射baseline与参数误差','S3三页；S5残差；完整映射表','6方法×9方向×20seed×适配前/Huber/最终','A3_mapping_comparison.csv; A3_MA_residual_records.csv','A/B/C 原缺口已补齐','旧CI终点或不同输入的比较不混入；本轮固定配置并非超参数搜索优胜者','无需再扩候选','PLS/SVR/MLP/TT新拟合；Ridge/KRR复用'),
      ('A4','同病例适配作用','S4；类别流向表','当前Ridge九方向同测试同预算','A4_adaptation_paired.csv; A4_adaptation_class_flow.csv','B 已重算','无计算缺口；收益可以为负','与A3误差互引','0'),
      ('A5','完整可信筛选比较','S2、S6三页；完整工作点表','全部报告、MA-margin、MA-error-LR、双侧保护；同上游','A2_A5_gate_working_points.csv; A5_risk_coverage.csv; A5_matched_coverage.csv','A/B 已重算','HGB开发验收失败仅作负结果；双侧分支异质分数不强行拼成全局风险曲线','不需要为了方法数量补跑已失败门','0'),
      ('A6','异常去向及风险分母','完整3×4去向表，与A5同套输出','全部9方向、低/范围内/高/未报告','A6_complete_class_destinations.csv','B 已重算','零分母NA，拒报不是正确异常；跨seed事件不是患者','无需新增图凑数','0'),
      ('A7','继续观察完整结果','S7；单点/累计/时间表','9方向×5/10/15分钟×20seed；HB B策略另列','A7_pointwise_cumulative.csv; A7_first_report_times.csv','A/B/C 原缺口已补齐','统一MA策略与湖北B探索分开；不能声称临床实时部署','无需新增时点','五个缺方向补5/10分钟；其余复用'),
      ('B1','适配预算敏感性','S8；四参数预算表','九方向0/C20/FULL，固定测试及来源映射','B1_calibration_budget.csv','A/B/C 原缺口已补齐','0是原始预测；C20由原冻结清单取得；不追加C30等密集扫参','三个有解释性的预算点即可','只补C20适配，不拟合新报告门'),
      ('B2','功能性贡献','配对差值表；与A4/A5/A7互引','适配、全部报告/门控、原门/双侧、固定时点/首次报告；B策略单列','B2_gate_paired_differences.csv; B2_sequence_fixed_paired.csv; B2_Hubei_system_strategy.csv','B 已重算','B策略同时改分类和报告时点，不归因单模块；不删除必要Agent','不做专家数量或机械Agent删除消融','0'),
      ('B3','稳定性和不确定性','S9；20seed分布、零报告、配对区间','全部既定seed；身份代理group整群区间','B3_gate_seed_values.csv; B3_sequence_seed_values.csv; B3_mapping_seed_values.csv','B 已重算','group不是已核实真实患者；稀少异常的区间不能证明安全','保留失败/零报告seed','0'),
      ('B4','效率与复现','分阶段资源表、代码及账本','同一CPU配置；100次warm推理；本轮训练及历史信号优化计时','B4_runtime_components.csv; B4_fit_context_runtime.csv; B4_archived_signal_optimizer_timings.csv','A/B 已完成','历史信号优化计时非当前硬件统一重测，且不含预处理；实验室真实工作流时延未测','准确限定计算阶段；不将5–15分钟采集计作计算耗时','0新增效率拟合'),
    ]
    fields=['条目','科学问题','拟用图表','方向时间方法预算','结果文件','状态','缺口或边界','最小补齐建议','新增拟合或统计']
    csv('APPENDIX_EVIDENCE_INVENTORY.csv',[dict(zip(fields,row)) for row in items])
    # Nine-direction summary is not a second independent result source.
    gate=pd.read_csv(P/'tables/A2_A5_gate_working_points.csv');a=gate[gate.method=='Dual protection'].copy()
    mapping=pd.read_csv(P/'tables/A3_mapping_comparison.csv');q=mapping[(mapping.method=='RIDGE-10')&(mapping.stage=='final_med4')][['direction','R_MAE','K_MAE','MA_MAE','alpha_MAE']]
    a=a.merge(q,on='direction',validate='one_to_one');a.to_csv(P/'tables/A2_nine_direction_summary.csv',index=False,encoding='utf-8-sig')
    old=O/'uteg_hubei_cumulative_external_20261027_184309';hb=b.read(old/'RESULT.json')
    csv('B2_Hubei_system_strategy.csv',[dict(strategy='separate HB B-assisted exploratory branch',**v) for v in hb['cumulative']])
    report=f'''# 附录补齐结果\n\n正文、现有四张主图和正文表格的内容与数字未改动。\n\n## 本轮完成\n\n- {len(plan['tasks'])}个固定任务，全部20个seed完成；新增实际拟合{result['actual_fit_units']:,}次，低于正式上界{plan['formal_fit_upper']:,}次。没有依据新增测试结果调参或选型。\n- 映射比较：Ridge-10、KRR-1复用；PLS-2、SVR-1、MLP-8、TT-SMALL补齐，九方向统一输入/划分/适配规则。\n- 适配敏感性：九方向原始预测、冻结C20、既定FULL预算，同一测试队列。\n- 时序：补齐其余五方向5/10分钟；九方向5/10/15分钟及首次报告回放全部齐备。\n- 可信门：四项匹配策略的工作点、风险覆盖、同覆盖率比较及完整异常去向；无新增门模型。\n- 全部seed、配对group区间、零报告情况、计时、独立计数/误差复算已交付。\n\n## 两大块组织\n\n第一块：A1数据与QC；A2九方向总览；A3映射；A4适配配对作用；A5门控；A6异常去向；A7继续观察。\n\n第二块：B1样本量；B2功能对照；B3稳定性；B4效率和复现。相同结果通过表格互引，不重复拟合或复制表格。\n\n## 必须保留的边界\n\n1. 湖州没有高MA参考样本，高MA风险不可评价；不是零风险。\n2. 浙二两正文方向共用同一测试队列，不是两套独立队列；九方向及全部seed完整保留。\n3. 分组是已有身份代理，患者身份尚未独立核验；不能将20seed事件称作独立患者。\n4. 附录统一MA时序与湖北B辅助探索策略分开；正文图4不被本轮替换。\n5. 原适配器保留CI辅助选择。评价用MA并不意味着训练全过程没有CI。\n6. 正式实验补齐不等于获得更优结果；任何负向变化、未报告及异常误放都保留。\n7. 本轮外测是既有且多次探索过的队列，不称全新未触碰的前瞻性验证。\n8. 95%区间是描述性group bootstrap，未作多重检验显著性声明；稀少/零错误的bootstrap区间不是临床安全保证。\n\n## 尚不能靠跑实验补齐的资料\n\n真实患者身份、部分仪器型号和标本字段须作者核实。效率表区分当前warm计算微基准、嵌套训练任务总耗时与历史信号优化计时；不虚构临床部署效率。其余已冻结计算任务没有待补方向/时点/seed。\n\n入口：index.html。详细对应关系：tables/APPENDIX_EVIDENCE_INVENTORY.csv。\n'''
    (P/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    (P/'REPRODUCE.md').write_text('''# Reproduction\n\nUse the existing Anaconda Python environment with NumPy, SciPy, sklearn, Torch and joblib. Preserve historical dependencies listed in PLAN.json.\n\n1. PLAN.json / PROTOCOL.md freeze the candidate matrix; do not regenerate them.\n2. `python -B launch.py` resumes only missing model contexts; no evaluation-dependent branches.\n3. `python -B analyze.py score` exports predictions and frozen actions before reference joins.\n4. `python -B analyze.py summarize` creates statistical tables.\n5. `python -B finalize.py` verifies and inventories.\n6. `python -B render.py` makes SVG/PDF/PNG appendix figures.\n\nNo patient data are uploaded. Record-level tables are local private source data, not automatically public-shareable. Copying this folder alone does not grant permission to distribute raw data.\n''',encoding='utf-8')
if __name__=='__main__':
    audit();resources();inventory();print('INDEPENDENT VERIFICATION AND INVENTORY COMPLETE')
