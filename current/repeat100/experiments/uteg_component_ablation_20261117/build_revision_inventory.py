from pathlib import Path as _RepeatPath
_repeat_root = _RepeatPath(__file__).resolve().parents[2]
from pathlib import Path
import hashlib,json,re
P=Path(str(_repeat_root / 'project' / 'outputs/uteg_nc_goal_20261030/manuscript'))
R=Path(str(_repeat_root / 'experiments' / 'uteg_locked_expansion_20261117/summary'))
O=Path(__file__).parent
names=['main.tex','supplement_current.tex','supplement_internal301.tex','supplement_minimal_additions.tex','supplement_reporting_associations.tex','supplement_files.tex']
src={n:(P/n).read_text(encoding='utf-8').splitlines() for n in names}
hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
items=[]
def add(title,file,line,after,why,end=None):
 items.append(dict(title=title,file=file,line=line,before='\n'.join(src[file][line-1:(end or line)]),after=after,why=why))
def main(title,line,after,why,end=None):add(title,'main.tex',line,after,why,end)
def si(title,line,after,why,end=None):add(title,'supplement_current.tex',line,after,why,end)
def load(n):return json.loads((R/n).read_text())
def f(x,k,pct=False,d=2):
 s=100 if pct else 1;return f"{x[k]['mean']*s:.{d}f} ± {x[k]['sd']*s:.{d}f}"

main('摘要：以实际放行结果替换旧固定配额结果',91,
 '保留开场、方法及结尾，只替换结果句为：\n\nAcross nine transfer configurations, mean MA-state accuracy ranged from 84.1% to 94.7%, with mean reporting rates of 93.1–98.1% over 100 subsampling repeats using the full local development pools. In the dual-source SAHZU configuration, we observed no abnormal-to-normal releases in any repeat, with 84.24 ± 2.65% accuracy and a reporting rate of 93.08 ± 4.03%.',
 '九配置范围来自同一common_B规则、full_local设置，不混入扩展ensemble。零FN限定双源浙二；不能声称九配置全零FN。范围是九个配置均值的范围，不是置信区间。若摘要太长，删除双源句中的±SD，将波动留Table 2；此处先给完整数字。')
main('引言末段：更换核心结果句',107,
 '将“At 15 minutes and a reporting rate ... after ranking predictions for reporting.”替换为：\n\nAcross 100 subsampling repeats, the dual-source workflow reported 93.08% of test records on average at 15 minutes, with 84.24% accuracy and no abnormal-to-normal releases. We repeated the evaluation with 54 and 60 local development records and report these settings separately.\n\n其余句子保留。',
 '维持原来的问题→框架→结果顺序；引言不重复摘要中的整套SD。')
main('Results第一节：加入复合规则的一句话功能解释',116,
 '在第一句后增加：\n\nWe also evaluated a composite rule that checks predicted-normal states for abnormal risk and selects release thresholds separately for predicted-low and predicted-high states.\n\n其余关于能力比较和首次报告的区分保留。',
 'HGB仍为一个放行baseline；新复合规则属于Reporting Agent中的另一条规则。')
main('Results第一节：20-seed改为100轮实际抽样',118,
 '保留中心、九配置和原始角色池说明，将从“The test cohort contains ...”起的统计说明替换为：\n\nThe original test pool contained 7 low-, 170 normal- and 24 high-MA records. In each of 100 repeats, we sampled 181 test records, comprising 6 low, 153 normal and 22 high MA, while preserving the predefined groups. The main analysis used all 60 development records; a separate analysis used 54. We also sampled the source records within their original roles. Supplementary Methods give the complete nine-configuration sampling scheme.',
 '分清原始池201与每轮181；主文建议保留60为主，54为敏感性，延续原稿预算。')
main('Fig.1图注：保留HGB能力插图，更新统计口径',123,
 '将“Reporting rate uses all 4,020 ... (201 records, 20 development partitions).”替换为：\n\nWe summarize the HGB capability comparisons over 100 repeats, each using 60 development records and 181 sampled test records. These matched comparisons differ from the composite rule’s development-selected operating points.\n\n插图数值随Fig.4更新，其余图注保留。',
 '不能让框架插图暗示它就是新复合门实际工作点。')
main('映射节开头：原池与抽样评价对应',129,
 '将第二句替换为：\n\nWe evaluated single-source mapping from Huzhou and dual-source mapping from Huzhou and Hubei on the same sampled SAHZU test records in each repeat.\n\n其余保留。','避免读者把Table 1的均值±SD误认为201条固定预测bootstrap。')
main('映射节：主模型与六baseline数字更新',131,
 '将MA结果改为“2.25 ± 0.04 and 2.35 ± 0.04 mm”；六模型范围改为“2.37–4.44 mm for single-source transfer and 3.70–4.99 mm for dual-source transfer”。其余算法名称、文献和完整模型比较说明保留。',
 '这些是60开发设置的100轮均值及样本SD；范围按各baseline均值计算。')
main('校正结果段：更新误差及波动',135,
 'Local correction reduced MA error in both source configurations (Table 1b). For single-source transfer, MA error fell from 9.74 ± 0.07 mm without correction to 2.25 ± 0.04 mm with asymmetric correction. The corresponding dual-source values were 5.42 ± 0.11 and 2.35 ± 0.04 mm. Independent affine calibration achieved MA errors of 1.71 ± 0.05 and 2.24 ± 0.06 mm. We report the complete nine-configuration comparisons in the Supplementary Information.',
 '原逻辑保留，不再沿用旧固定预测的区间。')
main('Target-only对照：新版不能再写FNR为9.68%',138,
 'We compared a target-only RBF model with source transfer under the same asymmetric MA loss. Across 100 repeats with 60 development records, MA errors were 2.25 ± 0.04, 2.25 ± 0.04 and 2.35 ± 0.04 mm for target-only, single-source and dual-source learning, respectively. Corresponding predictive R² values were 0.817 ± 0.007, 0.818 ± 0.008 and 0.807 ± 0.008. Three-state accuracy was 82.21 ± 1.02%, 84.24 ± 0.94% and 84.19 ± 0.98%. None of the three workflows produced abnormal-to-normal predictions in these repeats; FPR was 21.05 ± 1.21%, 18.64 ± 1.11% and 18.70 ± 1.16%, respectively. Figure 2 illustrates predictions on the complete 201-record reference cohort.',
 '此处报告门前结果，与Table 2门后结果区分。Fig.2固定201示例和100轮摘要不混用。')
main('标签预算和Ridge段：更新已核实误差',140,
 '将“1.81 mm MA error and 92.96% accuracy ... with 22.58% FNR”替换为“1.80 ± 0.05 mm MA error at 60 records”。其余预算结论由新的20/40/60表对应支撑，删除对旧paired intervals的引用，改指100轮逐次配对差值摘要。',
 '此处不必再堆Ridge分类指标；完整指标保留补充表，不能沿用旧数字。')
main('九配置映射总结：去掉未经新版核算的旧范围',142,
 'We report MA errors before and after local correction for all nine transfer configurations in Supplementary Tables S3 and S4. These comparisons use matched source samples, local development records and test samples within each repeat.',
 '旧1.16–9.81 mm范围不直接继承；新范围须从逐轮配对结果汇总，不能从不匹配的均值拼接。')
main('Fig.2图注与展示：保留固定201示例，删除20次预测相同的推广',149,
 '用以下文字替换E/F统计说明：\n\n(E,F) Predictions and state assignments for the complete 201-record SAHZU reference cohort, using 60 development records and all designated source records. Each test record is shown once. The single- and dual-source models assigned the same states in this reference evaluation and share a matrix. Repeated-subsampling results are reported separately as mean ± SD.\n\nE中fixed-reference数值：target-only MAE 2.255、R² 0.817、accuracy 82.09%、FNR 0%、FPR 21.18%；单源2.257、0.818、84.08%、0%、18.82%；双源2.353、0.806、84.08%、0%、18.82%。',
 '这是已保存reference201结果；不挑最好轮次。A–D示例不为了增加统计误差条而重复画100次。')
main('Table 1表注和全部单元格',154,
 '表注首句替换为：\n\nValues are mean ± SD over 100 matched subsampling repeats at 15 minutes, before report selection. Each repeat used 60 local development records, 181 sampled SAHZU test records and sampled source records. The separate 54-record development analysis is provided in the Supplementary Information.\n\n算法功能说明保留。24行的逐单元格对照见本清单末尾“Table 1完整替换值”。',
 '100轮不是100个独立队列，±为SD；Hybrid和Asymmetric重复行保持一致。')
main('Reporting开头：不再以旧79.60%为主体',200,
 'A prediction with a small MA error can still fall on the wrong side of a state boundary. We therefore kept MA predictions fixed within each repeat and compared how reporting rules withheld errors and retained correct reports (Fig. 3). We evaluated boundary distance, logistic regression and HGB alongside the composite rule. Table 2 reports their development-selected operating points; Figure 3a,b compares the three scalar scores at matched report counts.',
 '保留原来的自然过渡，明确曲线和实际规则两种证据。')
main('Reporting异常识别：从固定7/24改成60与54的差异',202,
 'With 60 development records, both source workflows correctly classified all sampled low- and high-MA records before gating in every repeat. With 54 records, abnormal-to-normal predictions occurred in 17 of 100 repeats for each source workflow. The composite rule withheld these false-normal predictions, leaving no released FN in either development setting. We calculated FNR and FPR among released records and report withheld outcomes separately.',
 '这是复核新增规则最清楚的论证职责；不要把60门前已零FN写成门的独立功劳。边界关联挪到附录并在Discussion呼应一次。')
main('Reporting核心结果：完整规则与HGB取舍',204,
 'With the composite rule and 60 development records, single-source and dual-source accuracy was 84.21 ± 2.63% and 84.24 ± 2.65%, at reporting rates of 93.30 ± 3.56% and 93.08 ± 4.03%, respectively. We observed no released FN in any repeat. HGB gave higher accuracy and lower FPR, whereas the composite rule retained more reports, including more correctly classified low- and high-MA cases (Table 2). In the dual-source workflow, HGB achieved 93.18 ± 4.63% accuracy at an 84.95 ± 8.71% reporting rate, with an FPR of 7.92 ± 5.49%.',
 '不声称复合规则所有指标优于HGB；主优势是FN复核及报告可用性。')
main('Reporting收尾：补消融并连接Fig.4',207,
 'Removing the normal high-risk check reintroduced FN in 17 of 100 repeats in each 54-record source setting; the complete rule produced none. Removing the amplitude bounds alone did not increase FN in these evaluations (Supplementary Table, component ablation). We retain HGB as a scalar-score control for the matched reporting comparisons below, separately from the composite rule’s operating points.',
 '删除这段中旧B95结论或移到其附录段，不把两套规则穿插当主结论。')
main('Fig.3图注：新规则加点不伪造连续曲线',212,
 '保留A/B的Boundary、Logistic、HGB三条曲线；新增复合规则实际工作点，不为其拼一个未经定义的标量排名。C/D增加复合规则柱。图注统计部分替换为：\n\nWe summarize 100 repeats with 60 development records and 181 test records per repeat. Upper bars show the mean number of FP predictions withheld per repeat; error bars show SD. Lower bars show the mean fraction of correct predictions retained. The composite rule is shown at its development-selected operating point. The 54-record analysis, including withheld FN, is reported separately.\n\n删除640、3,380和20 seeds；按新逐轮分母生成图。',
 '60条门前全是FP，所以正文上排仍可画FP；54条不能继续称所有错误都是FP。')
main('Table 2：增加复合规则，换均值±SD及分母',218,
 '表注统计部分替换为：\n\nValues are mean ± SD over 100 matched repeats with 60 development records and 181 test records per repeat. Each test subset contains 6 low-, 153 normal- and 22 high-MA records. We compute accuracy, reporting rate, FNR and FPR within each repeat before summarizing them. Class-specific counts are mean counts per repeat. Undefined ratios remain missing and their valid-repeat counts are reported in the supplementary data.\n\n保留All、Boundary、Logistic、HGB，新增Composite rule。Low→normal与High→normal列按每轮计数，分母改为6与22；不再使用140/480。完整替换值见末尾。',
 '将mean±SD用于每轮指标，而不是把18,100事件当独立样本。')
main('时序节首段：HGB能力与新规则末次报告分开',242,
 '保留原段的前两句与具体10分钟选择逻辑；将Table 2引用说明改为：\n\nThese HGB capability comparisons use matched report counts or observed accuracy, whereas Table 2 compares development-selected operating points. We separately evaluated updated reports with the composite rule at the final checkpoint and first-report schedules that retained early decisions.\n\n保留Supplementary Data 4/8分开引用。',
 '15分钟当前状态零FN不等于整个首次报告历史从未发生FN。')
main('Fig.4固定报告率段：统一用逐轮统计',244,
 'At a nominal reporting rate of 40%, accuracy increased from 84.71 ± 2.94% at 5 minutes to 89.33 ± 4.02% at 10 minutes and 99.99 ± 0.14% at 15 minutes. We selected 72 of 181 records in each repeat. FNR used only selected abnormal references; the valid-repeat counts and class denominators are given in Supplementary Data 4.',
 '15分钟此配额只有5/100轮包含被选中的异常，不能将其FNR写作100轮都验证为0。')
main('Fig.4固定accuracy段：不能混用pooled与逐轮均值',246,
 'At 90% matched observed accuracy, mean reporting rate was 21.71 ± 28.04% at 5 minutes, 59.20 ± 33.10% at 10 minutes and 90.53 ± 1.87% at 15 minutes. The matched point was defined in 94, 100 and 100 repeats, respectively. We retained all repeats in the supplementary results and marked unattainable points as undefined. Matching overall accuracy did not directly constrain FNR among abnormal references.\n\n末尾5分钟增长关联的自然解释保留。',
 '这是建议采用的逐轮统计口径。另存pooled路径在10分钟90%点是1.31%，不能拿它与逐轮SD混合。5分钟6个未定义轮次必须保留状态。')
main('Fig.4图注：100轮、数量和误差条',251,
 '将60/201、20次、4,020事件替换为60开发、每轮181测试、100轮。10/20/40%每轮对应18/36/72条。将最后两句改为：\n\nPoints and error bars show the mean and SD of repeat-level metrics. For matched observed accuracy, we interpolate within each repeat’s score-ranked curve. We retain unattainable points as undefined and report the number of evaluable repeats. FNR is undefined when no abnormal reference is selected.\n\n其他HGB与15分钟预测、分母说明保留。',
 '需要图线、正文、Data 4使用同一口径，不能只换图注不换曲线。')
main('Discussion第1段：扩展为风险控制，不只提高accuracy',260,
 'For early assessment of clot strength, clinicians need to know which predicted MA states have enough support to report. We treated reporting as a separate task from estimating MA, allowing us to check abnormal risk without changing the underlying MA predictions. With later observations, we could revisit that decision as more signal became available.',
 '完整门在60条下并非总提高accuracy；不沿用“we improved”泛指每种规则。')
main('Discussion第2段：压缩旧B95比较',262,
 '保留首三句的跨中心校正与target-only比较；将B95到first-report两句替换为：\n\nThe matched comparisons show that source transfer and direct local learning can use the same reporting pathway. We therefore assess their reporting outcomes alongside their MA errors.\n\n删除最后一句与新增句的重复。',
 '旧B95探索不是新复合规则的主证据，具体结果留附录。')
main('Discussion第3段：说明映射和门各自做什么',264,
 'Across these learning configurations, we sought to avoid abnormal-to-normal reports as well as reduce numerical error. We used an asymmetric loss to penalize errors that moved abnormal references toward the normal range. With 60 development records, both source workflows already had no FN before gating. The 54-record analysis exposed false-normal predictions that the additional normal-risk check withheld. We therefore evaluate the mapping and release rule separately, alongside the correct abnormal reports retained.',
 '保留“方向损失”解释，新增数据支持的分工，不把门和映射作用混为一谈。')
main('Discussion第4段：取舍解释而非重复百分比',267,
 'HGB withheld more erroneous predictions, whereas the composite rule retained more reports, including correct abnormal reports. We chose the latter to check normal predictions for abnormal risk while preserving report availability. The component analysis supports the high-risk check in the 54-record setting; the amplitude bounds did not show an additional FN reduction in these repeats. Within the Reporting Agent, these rules provide different trade-offs between error interception and retained reports.',
 '不声称全面最优或所有组件必要；异常类报告可用性是具体价值。原selective prediction文献留对应句。')
main('Discussion验证段：补一句重复抽样的证据范围',274,
 '在第一句后增加：\n\nThe 100 subsampling repeats quantify sensitivity to the sampled records; they do not provide 100 independent validation cohorts.\n\n其余前瞻验证、临床时间和用血量测量保持。',
 '范围只集中解释一次，不在每段加防御性尾句。')
main('Methods交叉中心：补抽样设计',299,
 '在现有交叉拟合流程后增加：\n\nFor each repeat, we sampled records without replacement within the original source, development and test roles, preserving MA strata and predefined groups. For SAHZU, we sampled 181 of 201 test records and used either all 60 development records or a sampled set of 54. Source samples comprised 86 of 95 Huzhou records and 71 of 79 Hubei records. Comparators shared the same sampled records within each repeat.',
 '各轮之间允许重复选中同条记录；轮内不放回。跨中心剩余数量放附录逐配置列。')
main('Methods：替换20-seed稳定性说明',301,
 'We retained all 100 repeats for each setting. The 54- and 60-record development analyses used separate summaries. For the expanded local dataset, we sampled 271 of 301 records in each repeat and used five outer and three inner folds, preserving predefined groups. Each sampled record was evaluated once per repeat using a model trained without its outer fold. This internal evaluation remained separate from the original development/test roles.',
 '不能把301×100写成301人每人仅一次，也不能把旧“相同预测”说明移到新抽样评价。')
main('Methods Ridge：删除旧20seed/固定201用语',352,
 '保持Ridge原理、惩罚候选和内层选择不动；最后三句替换为：\n\nWe fitted the selected model on the development records for that repeat and evaluated the same sampled test records as the transfer models. The two source configurations shared this target-only comparator. We retained all repeats and calculated differences using matched repeat assignments.',
 '不把旧5,000次区间沿用为新100轮SD。')
main('Methods标签预算：nested hash改为实际分层抽样',354,
 'For the local-label budget comparison, we sampled 20, 40 or 60 records from the original 60-record development pool, stratifying by MA state and preserving predefined groups. The smaller subsets were not required to be nested. Transfer and target-only models used the same records at each budget and repeat. Ridge standardization and penalty selection used only development folds. We evaluated readout errors and ungated MA states separately from reporting.',
 '这是实验协议真实变化，不是语言润色；保留旧nested描述会与新代码不一致。')
main('Methods reporting controls：三个baseline保持原理，补一句',362,
 '保留原文，在段末增加：\n\nWe also evaluated the composite rule described below using the same predictions.',
 '三种baseline不改名、不改变原有阈值约束；新方法不能套用它们的15%错误阈值描述。')
main('Methods：新增复合规则',379,
 '原段作为三种单分数baseline阈值方法保留，随后新增：\n\nFor the composite rule, we assessed predicted-normal, predicted-low and predicted-high states separately. Before releasing a normal prediction, we required it to pass a development-fitted abnormal-risk check, bounds on U-TEG MA and the selected normal-state gate. We set the high-risk cutoff at the 90th percentile of out-of-fold scores for development predictions classified as normal. We selected low- and high-state gates separately, maximizing development reporting rate while requiring no FN and more than 80% accuracy among released development predictions. Supplementary Methods specify the candidate thresholds and component analysis.',
 '详细候选、严格不等式、缺失处理、同分规则和参数保存放附录，并按实际代码逐条落实；不写“guarantees zero FN”。')
main('Methods首次报告：加入可更新方案，不删原对照',383,
 '原首次报告对照保留并标明B95/对应规则；新增：\n\nIn the separate updating analysis, earlier predictions remained candidates at 10 minutes, while the final decision used the current 15-minute prediction and its composite rule. We reported these outcomes separately from schedules that locked the first released state.',
 '不能将最终当前报告FN0误写为所有早期已发布报告都无FN。')
main('Methods Fig.4：改为逐轮匹配统计',386,
 '将“For matched observed accuracy, we pooled counts ...”及其后汇总方式替换为：\n\nFor matched observed accuracy, we interpolated counts along each repeat’s ranked curve to identify the largest attainable reporting rate at 85%, 90% or 95% accuracy. We then summarized repeat-level reporting rate and FNR, recording unattainable accuracy points and zero abnormal-reference denominators as undefined.\n\n其他score训练、时间选择、ties规则保持。',
 '与建议新版Fig.4均值±SD保持一致；pooled曲线作为另一张附录数据保留，不能混标。')
main('统计：Fig.3误差条不再沿用旧bootstrap',396,
 '保留风险排名、quota与ties定义；将bootstrap说明和下排统计替换为：\n\nFor each repeat, we calculated the withheld FP count and the fraction of correct predictions retained. We summarize these quantities as mean ± SD. The separate 54-record analysis also reports withheld FN. Table 2 uses the development-selected thresholds.',
 '上排按用户要求保留具体数量，使用每轮数量的均值±SD；不改成百分比。')
main('统计：替换汇总公式和分母',398,
 'For repeat r, let N_r denote the number of test records, a_ir indicate release and e_ir indicate an incorrect three-state report. We calculated reporting rate as sum_i(a_ir)/N_r and accuracy as sum_i[a_ir(1-e_ir)]/sum_i(a_ir). We then report the arithmetic mean and sample SD of each metric across the 100 repeats.',
 '此项替换398–404的定义、公式和20次加权汇总语句；正式写入时转为LaTeX公式。',404)
main('统计：新SD与旧CI明确分开',407,
 'For the repeated-subsampling analyses, we report mean ± sample SD and retain all 100 repeat-level results. Paired comparisons use the same sampled records and development assignments. Undefined ratios remain missing, with valid-repeat counts reported. Any retained bootstrap intervals from the earlier fixed-cohort analyses are labelled with their original protocol. Repeated samples overlap and are not treated as independent cohorts. Numerical computation used SciPy and scikit-learn.',
 '保留原文软件引用；附录列历史区间对应分析，不把SD冒充95%CI。')
si('附录开头：整体口径',3,
 'The updated comparisons retain all 100 subsampling repeats. We report the full and 90% local-development settings separately. Earlier fixed-cohort analyses retain their original protocol labels. Part B3 defines the summaries and denominators.',
 '旧20seed不能作为整份附录总说明。')
si('附录九配置：范围与摘要一致',77,
 'Using the composite rule and the full local development pools, mean accuracy ranged from 84.13% to 94.66% and mean reporting rate from 93.08% to 98.08% across the nine transfer configurations. These are development-selected operating points summarized over 100 repeats, rather than fixed reporting quotas. We provide the sampled-development results separately.',
 '84.1–94.7、93.1–98.1是摘要取一位小数版本。')
si('附录九配置：区分统一规则与扩展复核',79,
 'The complete nine-configuration comparison applies the same composite-rule definition to every configuration. We separately report the additional dual-source review procedure evaluated for the other target centres and the internal cohort. These extension results are not substituted into the nine-configuration summary.',
 '统一common_B在湖州仍有FN；后续ensemble零FN是不同流程。不能挑各中心最佳方法拼一个九配置零FN表。扩展实现含候选预测选择，不能只称“原预测不变的门”。')
si('附录S2和图S2：汇总方式',84,
 '15 minutes, 100 repeats for each rule and development setting. We compute outcomes within each sampled test subset and report mean ± SD. Class counts and valid denominators are retained for every repeat. The original test pools contain 201, 55 and 39 records; sampled test counts are recorded separately.',
 '图S2第95行对应改成mean±SD，不能继续写20次pooled/group interval。')
si('附录映射段：同步Table 1',104,
 '按Table 1新版数值同步：单源六baseline均值范围2.37–4.44 mm，双源3.70–4.99 mm；Hybrid为2.25 ± 0.04和2.35 ± 0.04 mm。原模型定义和四读数说明保留。',
 'S3、S4和主表共用同一汇总来源。')
si('附录残差图：删除平均20预测说明',240,
 'MA residuals after local correction. We summarize residuals separately for each repeated-subsampling configuration. Record-level displays identify the number of repeats in which each record was evaluated; repeat-level errors are reported independently.',
 '若残差图保留固定完整队列则直接标fixed reference evaluation，不与100轮图混合。不得无声平均预测后计算评价指标。')
si('附录A5：新门定义和baseline数值',249,
 'We compared all-report, boundary distance, logistic regression, HGB and the composite rule on identical predictions within each repeat. The three scalar-score controls used the common development-error criterion. The composite rule instead applied state-specific release checks, as specified below. Table 2 gives the full-development results; the 54-record analysis and all repeat-level outcomes are provided in Supplementary Data 3.',
 '在此补normal risk heads、训练参数、10%上限、物理幅值上下界、分状态候选门和择优顺序，直接对照保存policy；旧HGB数字由新表替代。')
si('附录B95说明：保留定义，更新结果与分母',251,
 '保留前两句B95定义。结果改用100轮B95行：单源accuracy 86.07 ± 1.94%、reporting rate 93.49 ± 2.80%、FPR 16.68 ± 2.27%；双源85.17 ± 1.56%、94.05 ± 2.40%、17.74 ± 1.82%；FNR均为0。低高保留计数改为每轮均值，不再写140/480。',
 'B95仍是补充baseline，不恢复到正文主规则。')
si('附录A6：清除640/3380等旧固定分母',343,
 'For each repeat, we counted correct and incorrect predictions before gating and their released or withheld outcomes after gating. Correct-report retention uses all correct predictions in that repeat. Low- and high-state retention uses all sampled records in the corresponding reference class, including withheld records. We summarize counts and proportions across repeats and retain their numerators and denominators.',
 '对应Fig.3、Data 3，删除已替换的93.40/53.91/78.57/73.54。')
si('附录A7：更新时序结果但保留锁定分析的身份',360,
 '用100轮各规则的first_report_locked结果替换旧3,078/640、3,174/846等计数；正文新复合规则的update_then_15min_current单独成表。新增说明：\n\nThe locked schedule retains errors from early releases. In the updating analysis, the final result is reassessed using the 15-minute prediction and rule. We therefore report final-checkpoint FN separately from errors among all first reports.',
 '第368、378、411、420行及该节表格、Data 4按相同区分同步；不能把更新后的零FN覆盖旧早期错误。')
si('附录A7能力比较：同步逐轮匹配及有效轮次',401,
 '保留HGB时间选择、isotonic和ties定义；20/40/80改18/36/72。固定accuracy改为每轮插值后汇总mean±SD；另列pooled路径。Data 4分别记录valid repeat counts、未达到匹配accuracy的轮次、FNR无异常分母的轮次。',
 '第405行18点/603prefix/20分区等固定数量也由新数据清单生成，不能原样保留。')
si('附录预算：40条最优结论不再成立',473,
 'We used matched source and test samples within each repeat while varying the local reference budget. Single-source MA errors at 20, 40 and 60 records were 2.11 ± 0.39, 2.30 ± 0.34 and 2.25 ± 0.04 mm; dual-source errors were 2.17 ± 0.46, 2.40 ± 0.38 and 2.35 ± 0.04 mm. We report the corresponding target-only controls and four-readout results separately.',
 '删掉旧“40 records lowest”结论，新抽样结果不支持；不补主观解释。')
si('附录B2：功能比较和旧时序差值',501,
 '将此处及第504、508行的固定201/20次差值和区间，分别替换为匹配100轮的映射差、门前门后差、时序差。保留“source composition and sample size jointly change”解释。先按同轮相减再汇总mean±SD，不由两组SD相减得到差值波动。',
 '三项均已有逐轮数据，属于重新汇总，不重新训练；完整数表随S9a–c更新。')
si('附录EXP3/EXP4：区分当前幅值分支',511,
 'We fitted EXP3 and EXP4 to the observed signals under matched assignments. In the current comparison, MA prediction uses the amplitude fitted under each representation, while companion models use the corresponding coefficients to predict R, K and angle. We summarize mapping and reporting outcomes across 100 repeats. Signal-fit residuals are reported separately from downstream prediction errors.',
 '旧89.05/86.57等来自不同下游定义，不能只替换数字保留旧实验名称。')
si('附录新增：固定规则组件消融',518,
 '新增短段和一张小表，不替换S9a标题：\n\nWe removed the normal high-risk check, the amplitude bounds or both while keeping all other fitted components and thresholds unchanged. With 54 development records, removing the high-risk check produced FN in 17 of 100 repeats for each source workflow, whereas the complete rule produced none. Removing the amplitude bounds alone did not increase FN in these evaluations. We report the associated changes in accuracy and reporting rate.\n\n完整20行消融结果见本轮RESULTS.md。',
 '表中保留两预算、两source、五条件；fixed-policy removal不等于简化模型重新优化比较。')
si('附录B3：稳定性与不确定性',609,
 'We summarize variation across 100 matched subsampling repeats. The sampled source and test records vary between repeats; the full local-development setting retains the original development pool. We report mean ± sample SD and retain every repeat, including undefined ratios. Earlier fixed-cohort bootstrap analyses retain their original labels and are not combined with these summaries.',
 '第611–627行旧bootstrap表只服务保留的历史分析；S11a–c与图S9改当前100轮，不将新SD描述成置信区间。')
si('附录耗时：新增组件，不改旧计时事实',675,
 '保留原组件计时及范围说明，在段后增加：\n\nWe additionally timed the two abnormal-risk heads and the fixed composite-rule decision using one query record. After 20 warm-up calls, we recorded 100 calls for each selected case. Median times were 0.0044 ms for the risk heads and 0.0176 ms for the decision step. The latter used previously computed gate scores. These measurements exclude signal acquisition, fitting, MA prediction, loading and training.\n\nS12a新增两行；第5–95百分位分别为0.0040–0.0055、0.0156–0.0238 ms，完整调用值进Data 5。',
 '不把组件时间相加冒充端到端临床延迟；100计时重复与100抽样重复是两种单位。')
add('附录C1：预算设计同步','supplement_minimal_additions.tex',9,
 'We compared transfer and target-only models at 20-, 40- and 60-record local budgets using matched sampled source and test records over 100 repeats. We sampled the local subsets within the original development pool, stratifying by MA state and preserving groups. The subsets were not required to be nested. We report mean ± SD and matched repeat-level differences.',
 '对应第13、22、31行、预算图和表全部同步；原hash nested及bootstrap口径不再用于新结果。')
add('附录C2：历史60/141分区分析保留原身份','supplement_minimal_additions.tex',74,
 '保留原60/141条分区及其数值，在本节开头加：\n\nThis supplementary partition analysis uses the original fixed 201-record test cohort and its original reporting protocol. It is reported separately from the 100-repeat subsampling evaluation.',
 '没有新100轮对应的历史分析不能假装已替换；其标题和引用不承担新主规则的结果。')
add('附录D：301内部与扩展规则分开','supplement_internal301.tex',5,
 'We combined the original local development and test records with 40 additional records, giving 301 records with 11 low, 253 normal and 37 high MA. For the repeated internal evaluation, we sampled 271 records per repeat and used five outer and three inner folds, preserving groups. Each sampled record was evaluated once per repeat. We report these results separately from the centre-specific development/test evaluation. The additional review procedure is identified separately from the composite-rule baseline.',
 '扩展ensemble/all的78.78±2.21% accuracy、72.70±1.72%报告率、100轮零FN不能填到common_B或HGB旧行；先写清候选预测与复核流程。原301固定散点可单独作为固定参考图。')
add('附录S13表注：门前与门后连续误差不混','supplement_internal301.tex',21,
 'Repeated internal evaluation at 15 minutes. We report mean ± SD over 100 repeats of 271 sampled records. Rows distinguish all-report, scalar-score controls, the composite rule and the additional review procedure. Continuous mapping errors use all evaluated records unless explicitly labelled as selected-report errors. FNR and FPR use released records of the corresponding reference class; correct-class retention includes withheld records in its denominator.',
 '每行方法身份、训练量和分母同时标清；历史301固定结果不能直接与271抽样均值混排不标注。')
add('边界关联短分析：更新计数方式','supplement_reporting_associations.tex',2,
 'We measured the distance from reference MA to the nearer 50- or 70-mm boundary for each test record. Across repeats with 60 development records, 90.41 ± 2.12% of dual-source prediction errors occurred within 3 mm of a reference boundary. We report the corresponding single-source and 54-record results separately.',
 '不再用旧29/32代表100轮；同轮比例先算后平均。')
add('早期增长分析：保留关联语气和HGB身份','supplement_reporting_associations.tex',5,
 '保留增长指标计算定义和关联非机制解释；HGB结果更新为60开发双源：低增长组三分类错误率10.96±1.58%、暂缓28.08±27.39%；高增长组32.63±3.60%、暂缓47.12±23.01%。补一句：\n\nWe report HGB and composite-rule withholding separately, using the same repeat-level growth groups.',
 'S6b列名分别为Prediction errors、HGB withheld、Composite-rule withheld，不能把HGB关联当复合门机制。历史301关联保留原分析标签。')
add('附件索引：八份数据、代码、视频保留并按新协议标识','supplement_files.tex',6,
 'Supplementary_Materials.zip contains the supplement, eight workbooks, the analysis code archive and Supplementary Video 1. Updated worksheets distinguish the 54- and 60-record development settings, the nine-configuration comparisons and the repeated internal evaluation. Every repeated-analysis worksheet records the repeat identifier, method, numerator, denominator and metric definition. Retained fixed-cohort analyses carry separate protocol labels.',
 'Data 1九配置；2映射校正；3门/消融；4时间；5逐轮/计时/表示；6target-only；7预算与旧分区；8内部/扩展。详细sheet名称需在真正打包时按生成文件确定，不提前编造链接。')
main('Code availability：待发布后锁定新版本',413,
 '保留仓库地址。待本轮代码、规则、100轮汇总和新组件审计实际同步后，将release指向真实新tag/commit，并同步Current_analysis_code.zip。当前不填写尚未发布的日期或哈希。',
 '本次仅给修改建议，没有推送或生成新release；不能把旧release描述成含新实验。')

header='''# 全文逐项微调清单：修改前／建议修改后

本清单为建议稿，正文、附录、图表、PDF、ZIP、GitHub均未改动。所有“修改前”按2026-11-17读取的本地TeX原文摘取。行号指TeX源码，不是PDF行号。

建议主文继续以完整60条开发池为主，54条作为敏感性分析；两套各100轮，所有轮次保留。固定201条参考展示与每轮181条抽样评价分开。此安排延续原稿重点，不表示两套结果可混并。

必须明确的结果边界：common_B是主文已锁定复合规则；后续ensemble是另一种扩展复核流程，不能用不同流程的最优数字拼出“同一方案九配置全零FN”。主文两source的15分钟common_B均100轮零FN；湖州common_B并非如此。固定40%报告率的HGB能力比较在15分钟仅5轮有选中的异常，其他轮FNR未定义。不能写成这些轮次均证实零FN。

下面每项给出原文和建议后文。标为“局部替换”的项目，其余句子不动；标为新增的项目保留原段。数值表逐单元格对照在末尾。
'''
out=[header]
for i,x in enumerate(items,1):
 out += [f"## {i:02d}. {x['title']}",f"位置：[{x['file']}:{x['line']}]({(P/x['file']).as_posix()}:{x['line']})",'**修改前（现文件原文）**','```tex',x['before'],'```','**建议修改后**',x['after'],'**原因／范围**',x['why'],'']
out+=['## Table 1完整替换值','顺序为R MAE、K MAE、MA MAE、angle MAE。旧值来自当前TeX；新值是60开发设置100轮均值±SD。54条结果独立进入附录。','|配置/分组/方法|修改前四值|修改后四值|','|---|---|---|']
mapping={'Ridge':'RIDGE-10','Kernel Ridge':'KRR-1','Partial least squares':'PLS-2','Support-vector regression':'SVR-1','Multilayer perceptron':'MLP-8','TabTransformer':'TT-SMALL'}
t1=load('Table1_summary.json');arm='';panel='mapping'
for l in src['main.tex'][159:192]:
 if 'b. Calibration' in l:panel='calibration'
 if 'Huzhou +' in l:arm='HZ+HB_to_SA'
 elif 'Huzhou $' in l:arm='HZ_to_SA'
 if '&' in l and not l.startswith('\\'):
  cols=[v.strip().replace('\\\\','') for v in l.split('&')];method=cols[0]
  q=next(x for x in t1 if x['budget']==60 and x['arm']==arm and x['panel']==panel and x['method']==mapping.get(method,method))
  out.append('|'+arm+'/'+panel+'/'+method+'|'+'; '.join(cols[1:])+'|'+'; '.join(f(q,k) for k in ['R_MAE','K_MAE','MA_MAE','angle_MAE'])+'|')
out+=['','## Table 2完整替换值','前三类比率为accuracy、reporting rate、FNR；第四列为FPR。Correct low/high为每轮计数均值±SD，分母每轮6和22。低/高→normal在这套60条设置各轮均为0。','|配置/规则|旧accuracy / rate / FNR / FPR|新accuracy / rate / FNR / FPR (%)|新correct low / high|','|---|---|---|---|']
t2=load('Table2_and_time_summary.json');old={};arm=''
for l in src['main.tex'][223:234]:
 if 'Huzhou +' in l:arm='HZ+HB_to_SA'
 elif 'Huzhou $' in l:arm='HZ_to_SA'
 if '&' in l and not l.startswith('\\'):
  c=[v.strip().replace('\\\\','') for v in l.split('&')];old[(arm,c[0])]=' / '.join(c[1:5])
for arm in ['HZ_to_SA','HZ+HB_to_SA']:
 for rule in ['All report','Boundary','Logistic','HGB','common_B']:
  q=next(x for x in t2 if x['budget']==60 and x['time']==900 and x['arm']==arm and x['rule']==rule)
  label={'Boundary':'Boundary distance','common_B':'Composite rule'}.get(rule,rule)
  out.append('|'+arm+'/'+label+'|'+old.get((arm,label),'新增')+'|'+' / '.join(f(q,k,True) for k in ['accuracy','reporting_rate','FNR','FPR'])+'|'+' / '.join(f(q,k) for k in ['low_correct','high_correct'])+'|')
out+=['','## 补充图表和电子数据的逐项覆盖','以下逐项更换数值或统计标签，不代表所有历史分析都重新运行。未重做的保留原协议，不把20次机械改为100次。','|项目|修改前|建议修改后与来源|','|---|---|---|']
rows=[
('S1/样本和伦理/QC','478条、三中心及原角色池','保持不变；补一张抽样角色表说明每轮数量，不能把原样本数改成抽样数'),
('S2/九配置报告','20seed pooled/B95','nine_configurations_*：两本地预算、各规则、100轮；另列extension，不能混为同一规则'),
('S3/映射','旧六模型点值','nine_mapping_and_calibration_controls_*及Table1_*：四读数均值±SD'),
('S4/校正与paired changes','旧固定预测bootstrap','nine_mapping_and_calibration_controls_*：每轮同配置配对；原CI若保留明确固定队列'),
('S5/残差图','平均20次预测','指定固定参考展示或逐轮残差分布；不将平均预测重新计算当100轮指标'),
('S5a/门工作点','旧HGB、LR、Boundary结果','Table2_and_time_*及nine_configurations_*，增加Composite，显示有效分母'),
('S5b、S6风险排名图','旧匹配配额/曲线','Fig3_score_curves_*与实际工作点分开；新复合规则没有伪造标量曲线'),
('S6/完整去向','固定140/480等','逐轮低/正常/高→低/正常/高/暂缓，所有轮次完整'),
('S6b/边界与增长关联','29/32及旧20次HGB计数','boundary_associations_*、early_growth_associations_*，按门前错误与具体门暂缓分列'),
('S7a,b/时序','旧累计4,020事件','time_schedules_*：locked与update明确分开，按每轮分母汇总'),
('S7九方向首报图','旧B95全9时序','若保留旧图，标原固定队列B95分析；不能冒充新复合门100轮全9时序'),
('S8/预算','旧40条最优','label_budgets_*：20/40/60的100轮结果；修改非nested设计'),
('S9a–c/功能和配对','旧事件数及95%区间','gate_minus_allreport_paired_*、time_schedules_*、当前mapping配对摘要；单独新增component ablation表'),
('S10b–g/历史备选方法','20seed开发搜索','保留历史探索标签，不填成新主规则，正文不据此主张新效果'),
('S11a–c/逐轮稳定性','20seed','完整100 repeat记录，54/60分表，空分母留NA'),
('S12a/组件时间','5行旧组件','保留旧范围并新增风险头/复合判定两行；不把重采样时间作临床耗时'),
('S12c,d/信号拟合耗时','实际记录耗时','原拟合未改则保留；与新增组件计时分开'),
('S13/表示实验','旧多输入下游EXP3/4','EXP3_*、EXP4_*、representation_four_readouts_*；明确MA当前只用幅值'),
('C1/预算图表','nested20/40/60和旧CI','新分层预算100轮mean±SD；逐轮配对差另列'),
('C2/60和141子集','固定201队列历史首报','保留原结果并标独立协议，不当成301验证'),
('D/内部图表','固定301一次OOF','271/301抽样×100为新统计；完整301散点可作为固定参考，扩展ensemble行单列'),
('Data 1–8及代码包','旧20seed、旧release说明','随对应图表一起同步：新增逐轮ID/预算/方法/分母/SD；不删失败轮次；视频保留')]
out += ['|'+'|'.join(x)+'|' for x in rows]
out += ['','## 不改的内容','标题、临床背景、U-TEG原理、医院/伦理、样本来源、信号时间窗口、两条映射分支、固定损失与正则参数、已有自然表达、作者和基金信息不因本轮统计补充而改。研究主线仍为U-TEG转译→MA状态判断→是否及何时报告。',
 '新增或替换的所有数字应由本清单所指汇总文件自动填入；不能选择部分轮次、跨规则拼最佳结果，或把未定义FNR当成0。',
 f'本清单共{len(items)}项文字/图注/方法位置，另含Table 1的24行、Table 2的10行及22项补充图表/附件覆盖。']
target=O/'全文微调_修改前后逐项清单.md'
content=re.sub(r'(?m)(^\|.*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(out))
target.write_text(content,encoding='utf-8')
assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
(O/'REVISION_INVENTORY_CHECK.json').write_text(json.dumps(dict(items=len(items),source_hashes=hashes,manuscript_unchanged=True),indent=2),encoding='utf-8')
print(target);print('items',len(items))
