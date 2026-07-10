# Runge 全候选三阶 TM 穷举设计

## 目标

对每个报告预测尺度 $H$ 的全部 `102660` 条跨目标二源超边计算与当前 `estimate_mutual_information_transport_map(..., degree=3)` 数值等价的 EI 与二阶增量，消除离散 top-1000 初筛造成的排序偏差。

## 实验契约

唯一改变因素是候选覆盖范围：从离散初筛的 1000 条扩展为全部候选。MLP+Ridge rollout、4096 个最大熵干预样本、随机种子、source mode、目标定义、三阶 polynomial triangular TM、非负单项 EI 截断和 $\Delta_{2,\mathrm{TM}}=EI_{ij}-EI_i-EI_j$ 均保持不变。

## 计算结构

当前三阶 TM 在同一批样本上拟合并评估。批量实现严格保持联合列顺序 `[target, source_a, source_b]`、源边缘顺序 `[source_a, source_b]` 和 `source_a < source_b`。同时保持 degree-3 指数顺序、预测量按 `ddof=1` 标准化（低方差替换为 1）、`ridge=1e-6` 且截距不惩罚、残差 `std(ddof=1)`、`min_scale=1e-8`、原始 MI 及单项 EI 非负截断顺序。

批量实现复用单源 EI、源对边缘密度，以及同一 $(H,target,source_a)$ 的多项式设计矩阵与 Gram 分解。第二源响应按矩阵批量求解，并从残差 SSE 和残差尺度计算平均 log density；即使触发 `min_scale` 也不以“尺度必然等于经验残差标准差”为前提。

结果按 `H × target` 分块保存为压缩 NPZ。每个分块包含 schema/version，以及 source samples、rollout predictions、模型缓存、blend、$H$、target、seed、source mode、sample count、degree、ridge、min_scale 和候选顺序的指纹。恢复时核对指纹、数组形状、预期索引、唯一性、有限值和候选数；任何不一致都重算。临时 NPZ 写在目标目录，显式关闭并同步后用 `os.replace` 原子替换，避免 `np.savez_compressed` 自动扩展名造成误判。

source samples 由旧实验的确定性输入构造器重建，并要求生成的 rollout 与 `rollout_predictions_H060_n4096.npy` 逐数组哈希一致；由于旧 top-1000 未单独保存 source artifact，不能声称对旧 source 数组进行了独立哈希核验。重建后的 4096 个 source samples 另存为只读基准，后续恢复必须同时匹配该基准。模型缓存、配置哈希和 `ridge_weight=0.37`、`mlp_weight=0.63` 也写入 manifest 并在排名比较前断言一致。

## 正确性门槛

随机数据、近常数/min-scale 数据与真实 rollout 候选均须和当前逐条 degree-3 TM 对照。比较原始 MI、截断后单源 EI、联合 EI 与 $\Delta_2$，最大绝对误差须不高于 `1e-8`；候选总数须为每个 $H$ `102660`。若数值门槛失败，不启动全部尺度。

## 执行顺序

先运行单元测试，再运行 $H=1$ 小规模 smoke benchmark，随后完成 $H=1$ 全候选。启动其他尺度前必须以 fail-closed 方式同时通过：误差 `<=1e-8`、候选数 `102660`、输入/模型指纹匹配、无非有限值、完成运行时间与峰值内存记录、完成与原 top-1000 的排名变化诊断。任一项失败即停止。长任务进度记录在 `docs/log/live_status.md`，运行日志写入 `docs/log/logs/`。
