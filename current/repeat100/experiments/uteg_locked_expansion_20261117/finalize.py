"""Inventory and seal this local experiment output, without public delivery writes."""
import experiment as e
import aggregate as ag
import json,shutil,sys,platform,hashlib
import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits
O=e.O

def main():
    expected={k:100 for k in ['table1','budgets','nine','linear','EXP3','EXP4','internal301','nine_controls','internal_controls','representation_mapping']};expected['primary']=600
    actual={k:len(list((O/k).glob('**/COMPLETE.json'))) for k in expected};assert actual==expected,(actual,expected)
    for stem,required in [('Table1',4800),('Table2_and_time',10800),('nine_mapping_and_calibration_controls',21600),('nine_configurations',14400),('internal_mapping_and_calibration_controls',2200),('internal301',1800)]:
        rows=json.loads((ag.S/(stem+'_rounds.json')).read_text());assert len(rows)==required,(stem,len(rows),required)
        summary=json.loads((ag.S/(stem+'_summary.json')).read_text());assert all(v['rounds']==100 for v in summary),stem
    d,plan,role,models,items=e.init();checked=0
    for rep in range(100):
        folder=O/f'internal301/round_{rep:02d}';splits=json.loads((folder/'SPLITS.json').read_text());alltest=np.concatenate([s['test'] for s in splits]);assert len(alltest)==271 and len(set(alltest))==271
        for sp in splits:
            assert not set(d['groups'][sp['pool']])&set(d['groups'][sp['test']])
        rows=json.loads((folder/'RESULTS.json').read_text())
        for arm in e.ARMS:
            with np.load(folder/(arm+'.npz')) as z:
                assert set(z['test'])==set(alltest);p=e.r.state(z['prediction']);t=e.r.state(d['medical'][z['test'],2])
                for row in [v for v in rows if v['arm']==arm]:
                    act=z['action_'+row['rule'].replace(' ','_')];m=row['test'];assert int(np.sum(act&(t!=1)&(p==1)))==m['FN'];assert int(np.sum(act&(t==1)&(p!=1)))==m['FP'];assert int(np.sum(act&(t==p)))==m['correct'];checked+=1
    # Descriptive amplitude correlation, separate from predictive R2.
    corr=[]
    for item in items:
        ix=np.asarray(item['targets']['SAHZU']['test']);rr=float(np.corrcoef(d['x'][ix,1],d['medical'][ix,2])[0,1]);corr.append(dict(rep=item['rep'],n=len(ix),Pearson_r=rr,descriptive_R_squared=rr*rr))
    ag.write('Fig2C_descriptive_correlation',corr,[])
    snapshot=O/'code_snapshot';snapshot.mkdir(exist_ok=True)
    shutil.copytree(e.LOCK/'code_snapshot',snapshot/'locked_dependencies',dirs_exist_ok=True)
    for p in O.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    runtime=dict(python=sys.version,platform=platform.platform())
    for name in ['numpy','scipy','sklearn','torch']:
        module=__import__(name);runtime[name]=module.__version__
    e.r.write(O/'RUNTIME.json',runtime)
    e.r.write(O/'FINAL_CHECKS.json',dict(expected=expected,completed=actual,all_repetitions_retained=True,internal_metric_rows_checked=checked,original_inputs_and_manuscript_check=json.loads((ag.S/'VERIFICATION.json').read_text()),source_and_final_correction_isolation=json.loads((O/'checks/ISOLATION_AND_TABLE1.json').read_text()),selector_equivalence=json.loads((O/'EXACT_SELECTOR_CHECK.json').read_text()),no_paper_or_public_delivery_changes=True))
    text='''# 本轮实验完成清单

方案保持为已确认的 common_B：normal 专门检查、原10%高风险上限、低/高MA分别选门；不重新调成本、惩罚或阈值搜索空间。对照原定义保留；各自的阈值仍只由开发侧生成。

|对应内容|本轮状态|查看位置|
|---|---|---|
|Table 1：六源模型、Hybrid regression、五校正、四读数|54/60各100轮，齐|summary/Table1_*|
|Table 2：三种学习流程、全部放行基线及固定方案|54/60各100轮，齐|summary/Table2_and_time_*|
|Fig.2 E/F：门前误差、R²、低高召回、3×3矩阵|54/60各100轮，齐|summary/Fig2_*|
|Fig.2 A–D：实测、描述性关系、固定201条展示所需预测|原实测不变；另存完整源/60开发/201测试、seed1000的固定参考结果；未挑轮次/样本|reference201、summary/Fig2C_*|
|Fig.3：单分数曲线、实际门工作点、正确/错误放行暂缓数量|54/60各100轮，齐|summary/Fig3_*、gate_minus_allreport_paired_*|
|九配置及三个target-only|90%本地池/完整本地池各100轮；浙二54/60，其余36/40|summary/nine_configurations_*|
|九配置六模型和五校正|两个本地预算各100轮，齐|summary/nine_mapping_and_calibration_controls_*|
|5/10/15分钟|54/60各100轮；每时点规则固定后由开发侧拟合|primary、summary/time_schedules_*|
|Fig.4原HGB能力比较|三种固定报告率、三种匹配观察accuracy，各100轮；与新复合门实际工作点分开|summary/Fig4_HGB_*|
|20/40/60标签预算、target-only Ridge/RBF|各100轮，齐|summary/label_budgets_*|
|仅幅值的线性/RBF|两套预算各100轮；成本4、惩罚0.01固定|summary/linear_*及Fig2_all_report_*|
|EXP3/EXP4|两套预算各100轮，当前幅值分支和其他三读数均已汇总|summary/EXP3_*、EXP4_*、representation_four_readouts_*|
|301内部评价|每轮抽271条，5折外层/3折内层，100轮；单独报告|summary/internal301_*|
|301内部的六映射与五校正|相同100轮和外层划分，齐|summary/internal_mapping_and_calibration_controls_*|
|边界误判与5分钟增长关联|两套各100轮的描述性汇总；不重新选规则|summary/boundary_associations_*、early_growth_associations_*|

## 不应机械再跑100次的内容

- 医院样本组成、伦理/QC、采血条件、仪器信息和十例实测曲线属于描述性材料。
- 历史参数搜索、测试标签参与的探索性误差匹配、不再采用的策略不纳入本次固定方案；不把历史搜索当作新方案的验证。
- 运行时间需要单独的计时口径，不能把这100次抽样运行耗时当临床延迟；旧报告的100次warm-CPU计时也不是本次抽样SD。
- 301条的扩展100次结果此处只评价15分钟及其映射/校正对照。既有301早期时序、参数搜索仍是单独历史内部分析，不宣称已经转成100次；正文Fig.1/4的54/60时序与能力比较已齐。若以后保留这些历史内部分析，应保留原协议和标签，不混成当前结果。

## 更新稿件时不能直接替换的地方

1. 新复合规则不能改名填进HGB旧行。三种单分数曲线与复合规则实际工作点应区分。
2. 54/60是开发规模；181/201是本轮抽样测试/原测试队列；271/301是另一套内部敏感性。三种分母不可混用。
3. 六源模型保留原训练目标和四输入，局部MA校正及尾部处理统一；这是完整源模型方案比较，不是排除输入/损失差异的纯架构消融。
4. 原EXP3/EXP4多输入预测器与此处当前幅值主分支的敏感性定义不同。现在MA均从对应拟合幅值进入固定映射；其他读数从对应EXP3/4表示映射。不能只换旧表数字而保留旧预测器说明。
5. 本地预算20/40使用开发类别分层抽样，均在原60池内，两个子集不强制嵌套。不能沿用旧“哈希扩展的嵌套20/40”句子。
6. 均值±SD反映重叠样本上的重复抽样波动，不是95%CI或新增独立证据。无定义的FNR保留为空，不计作0。
7. 15分钟浙二三流程的两套100轮均观察到零FN。湖州目标、早期时点和301内部结果并非全部零FN，完整保留。

正文、附录、正式图表、PDF、补充材料ZIP、老师包和GitHub均未修改。此次只交付本地研究结果和后续同步依据。
'''
    (O/'COMPLETION_CHECKLIST.md').write_text(text,encoding='utf8')
    # The whole local folder includes private predictions; it is not a public repository upload.
    readme='''# 固定方案实验结果

先读 RESULTS.md，再读 COMPLETION_CHECKLIST.md。summary/ 提供逐轮与均值/样本SD/min/max/有效轮次数CSV及JSON；原始逐轮预测、阈值、划分和模型分别在各实验文件夹。

运行入口：experiment.py（主表/放行），additional.py（九配置/预算），extensions.py（线性/表征/内部），nine_controls.py及internal_controls.py（完整对照），representation_mapping.py，capability.py，reference_and_associations.py。aggregate.py仅汇总；final_checks.py检查隔离和主表数值；finalize.py封装本地清单。各入口读取已有固定数据和划分。已完成轮次按COMPLETE.json恢复，无需重复拟合。

code_snapshot/ 保存运行代码及原固定方案依赖。PROTOCOL与各分项PROTOCOL说明统计口径。FINAL_CHECKS.json记录验收，FILE_MANIFEST.json记录文件哈希。输入数据、原固定包以及论文交付均保持原样。
'''
    (O/'README.md').write_text(readme,encoding='utf8')
    manifest={str(p.relative_to(O)):dict(size=p.stat().st_size,sha256=e.r.sha(p)) for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='FILE_MANIFEST.json'}
    e.r.write(O/'FILE_MANIFEST.json',manifest);print('FINAL COMPLETE',len(manifest),'files')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
