# Ridge-10 fullchain protocol v1

## 1. 范围

本轮唯一 mapper 候选为 Ridge-10；不新增信号、不改变 B 分类、不引入 GP/HGB/B-only
门。旧 B/WITHIN、KRR 和原 MA 结果仅作对照背景，不能复用其 target 系数或门对象。

## 2. 数据与角色

主数据为 legacy aligned three-centre `DATA15.npz`、`ROLE_FREEZE.json` 和九方向
`NESTED_SPLIT_MANIFEST.json`。source role 由方向确定：三条 dual 使用相应双源，六条
single 使用 D1/D2/Hubei source role；Hubei 仅 QC-good 79 行。被遗漏第三中心不得进入
source training、scaler、adapter 或 target fit。EXP D1/D2 的本地 source row identity
必须与 preflight 绑定，不能按目录名推断。

## 3. Ridge-10 身份

Ridge-10 必须来自已审计 `models.py`/`fit_source` 实现：`family=RIDGE`、`name=RIDGE-10`、
`config.alpha=10`，四输出，字段含 `xm/xs/ym/ys/coef/intercept`。full source 对象的训练行
必须精确等于该 source role 的全集，validation 必须为空；这不是“训练行为空”。Scaler、
四输出缩放和模型系数必须随对象保存。
已有 source full 只有在逐对象 role、rows、seed、scaler、config、coef digest 均通过时才
可复用；否则按 source full fit 计账。不能复用 KRR full object。

## 4. Target chain

每个 direction×seed 重新执行原合法 nested calibration/adapter/gate。原 CI 辅助 adapter
选择保留且单独记录；d=2 仍保留 ERROR-LR 与 deterministic MARGIN 两个选择头，d=1 不
暗中新增 ordinal fit。最终主终点是四参数 MA 与 MA-ERROR-LR。

MA gate 的 outer-B max-risk 固定为 Huzhou=.10、SAHZU/Hubei=.15，表示经验 release
错误率上限，不是 score threshold。所有 outer-B scores、pred/truth class、record IDs
先保存；之后通过 AST 抽取的旧 `fit_ma_gates.py` `summary/select_threshold` 枚举 finite
unique scores 并冻结 threshold。position 3 只用 positions 0--2 合并 outer-B pool 选择一次。
无合格 threshold 时 all-withhold。测试只应用冻结 threshold。

## 5. 预算与 ledger

target Huber=2520、adapter=6960、adapter-selection LR=2400、MA-ERROR-LR=720，
合计 12,600 science units。source full 最多 120；repair/check 最多 280；total≤13,000。
每个实际 fit/失败/修复在数值调用前持久登记，失败不伪装为 completed，重试用新 attempt key。
预检和 hash 检查不得偷偷消耗 science fit。

## 6. 隔离与验收

pilot 只用于开发契约检查，不读测试标签；完整训练后冻结所有对象、threshold、scores、
代码和身份 hash，经独立 preflight 后才可一次性评价测试。九方向全部保留；目标要求
agreement≥80%、coverage≥20%、真实 low/high→released-within 分尾误放=0、两尾和正常
支持非空且按 distinct record 计数。种子不是患者独立样本；区间按原 record/patient
cluster bootstrap。任何测试结果不得反向选择 mapper、seed、方向或 threshold。
