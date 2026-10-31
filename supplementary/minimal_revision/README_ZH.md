# Minimal revision code bundle

这是本轮最小修改的公开代码包。它不是一个可以脱离项目目录直接运行的 clean-checkout，也不包含临床原始数据、患者 ID、split mask、逐记录预测、NPZ 或 joblib 模型对象。

## 包含内容

- `run_budget_curve.py`：20/40/60 条目标中心配对标签的预测层预算分析。它读取已冻结的 C20/C60 迁移工件，并在运行时重新拟合 C40 target-side adapter 和每个预算的 target-only Ridge；随后读取固定测试队列进行评估。它不拟合或评分 reporting gate。
- `internal_pipeline/replay_internal_partition.py`：读取冻结的报告动作序列，对 full201、selected60 和 complement141 做聚类/事件级汇总重放；不训练模型，不改变阈值。
- `internal_pipeline/forward_recheck.py`：从已处理的前缀特征和冻结模型对象重放到首报动作，并与存档字段逐元素核对；不重复原始信号拟合或传感器采集，不训练模型，不选择新阈值。
- `requirements_public.txt`：脚本所需的常用 Python 依赖。

## 运行前提

脚本依赖原项目的相对目录和受限输入。需要把脚本放回与本轮输出相同的项目相对位置，或同步修改脚本开头的路径常量。受限输入包括 `DATA15.npz`、冻结的模型/适配器 joblib、预测 NPZ、报告动作存档、阈值冻结文件和现有 hash split 文件；这些文件没有随包发布，也不能用本包单独推断出结果。

预算脚本的运行会产生新的拟合工件，尤其是 C40 target-side adapter 与 target-only Ridge；请在项目副本中运行，不要把输出写回只读的历史冻结目录。内部重放和前向核验脚本只应使用只读冻结输入，并把汇总写到新的输出目录。

## 可复现边界

本包支持在作者具有相同受限输入、目录和运行环境时重跑相应分析。公开 workbook 中的汇总值不依赖把上述受限数据公开；workbook 本身不能重建患者级数据、bootstrap 分组或模型对象。内部 60/141 分区是此前已查看测试队列内的 robustness check，不是独立验证集。

## 基本检查

建议使用 Python 3.10+，先执行语法检查：

```text
python -m py_compile run_budget_curve.py internal_pipeline/replay_internal_partition.py internal_pipeline/forward_recheck.py
```

本代码打包/资产整合步骤仅做了静态 AST/语法检查、公开汇总一致性检查和 ZIP 成员审计，没有在打包阶段重新训练模型或新增实验。需要区分的是：本轮预算分析此前实际完成了 120 个 target-only 最终拟合和 40 个 C40 target-side adapter 最终拟合，并复用了 C20/C60 transfer 结果；前向复核阶段没有进行拟合。
