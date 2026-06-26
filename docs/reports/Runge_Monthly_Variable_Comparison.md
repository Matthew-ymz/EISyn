# Runge 月尺度多变量地理对照报告

## 实验目的

本实验用于回答：在不同气候变量上使用同一套 monthly Runge MLP-TM-EI path-effect 流程时，Top gateway、ACS 入向强度和 Top mediator 的地理中心是否大体一致，或是否呈现明显变量依赖的空间差异。

这里不把不同数据集的 component ID 当成同一物理模态，也不严格量化跨变量排名一致性。比较重点是地球图上的空间分布。

## 数据与覆盖

| 数据集 | 输入文件 | 变量 | 时间范围 | 有效格点 |
|---|---|---|---|---:|
| SLP | `data/ncep_reanalysis_slp/monthly/slp.mon.mean.nc` | `slp` | 1948-01 到 2026-02 | 10512 / 10512 |
| 2m air temperature | `data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc` | `air` | 1948-01 到 2026-02 | 18048 / 18048 |
| 1000hPa air temperature | `data/ncep_reanalysis_runge_validation/air.1000hPa.mon.mean.nc` | `air`, `level=1000` | 1948-01 到 2026-02 | 10512 / 10512 |
| SST | `data/noaa_ersst_v5/sst.mnmean.1948_2026.nc` | `sst` | 1948-01 到 2026-02 | 10988 / 16020 |

SST 保留 ocean-only valid mask，陆地无效格点在 component maps 中保持为 `NaN`。

## 统一流程

每个数据集都使用同一套预处理和下游分析：

1. 按时间排序并对齐 monthly samples。
2. 对每个格点按 calendar month 去除月气候态。
3. 对每个 calendar month 的 anomaly 按该月标准差标准化，零方差或无效标准差安全处理。
4. 对标准化后的 anomaly 时间序列做线性去趋势。
5. 在有效空间格点上做纬度面积加权 PCA 和 Varimax，提取 60 个 component。
6. 使用最近 4 个月 component state 预测未来 1 个月 component state。
7. 对每个数据集运行 MLP-TM-EI path-effect gateway/mediator 流程。

固定参数为 `n_components=60`、`lag=4`、`horizon=1`、`source_mode=latest`、`ei_estimator=tm`、`gateway_mode=path_effect`。

## 地理对照图

![Monthly Runge MLP-TM-EI gateway, ACS, and mediator centers](../../fig/runge_monthly_variable_comparison/gateway_mediator_centers.png)

图中每一行是一个变量，每一列分别是：

- `ACE`：component 作为源头的平均出向 total effect，用于看 Top gateway。
- `ACS`：component 作为接收端的平均入向 total effect，用于补充查看 “为什么 ACS 也需要看”。
- `AMCE`：component 作为中介时的平均 mediated effect，用于看 Top mediator。

点的位置是对应 component 的 sign-normalized loading map 的地理中心。点大小和颜色表示该列对应分数大小。component 标签只在各自数据集内部有效。

## 主要观察

SLP 的 ACE、ACS 和 AMCE 都出现了较强的高纬和南半球海洋信号，并且 `C41`、`C12`、`C23` 在不同列之间反复出现。SLP 内部的 gateway 与 mediator 地理分布相对更接近，但 ACS 图仍显示入向强度可把部分中心换到不同洋盆。

2m 气温的 ACE 更集中在南半球海洋、印度洋到西太平洋一带，ACS 则明显转向美洲、南大洋和大西洋附近。AMCE 与 ACE 有重合的 `C7`、`C8`，但也引入 `C39`、`C59`、`C11`，说明 mediator 排名不是 gateway 排名的简单复刻。

1000hPa 气温的 ACE 主要落在太平洋、非洲-印度洋和热带附近，ACS/AMCE 更偏向南半球和西太平洋/高纬区域。该变量与 2m 气温有部分南半球结构相似性，但 Top component 的中心分布并不完全一致。

SST 的 Top 点全部落在海洋有效区域，ACE、ACS、AMCE 都偏向太平洋及南半球海洋边界附近。它与三个大气变量相比差异最大，说明海温场在同一 EI path-effect 流程下输出的是更海盆化的地理结构。

总体上，四个数据集不是简单给出同一组地理中心。2m 气温和 1000hPa 气温有部分相似的南半球/洋盆分布，SLP 与 SST 则更有各自变量特征。ACE 和 ACS 也不能互相替代：ACS 明确展示了入向 total effect 强的 component，部分数据集上与 ACE 的 Top 5 空间位置不同。

## Top 组件摘要

| 数据集 | Top ACE gateway | Top ACS incoming | Top AMCE mediator |
|---|---|---|---|
| SLP | `component_12` | `component_41` | `component_41` |
| 2m air temperature | `component_07` | `component_08` | `component_08` |
| 1000hPa air temperature | `component_04` | `component_13` | `component_13` |
| SST | `component_01` | `component_10` | `component_27` |

这些 component ID 均为 dataset-local ID，不应跨数据集直接对应。

## 预测误差诊断

![Monthly Runge prediction error comparison](../../fig/runge_monthly_variable_comparison/prediction_error.png)

这里把 test RMSE 除以同一数据集 test 目标的 zero-predictor RMSE，得到相对 RMSE；数值越低越好。右图是 test correlation。`Selected MLP/blend` 是当前下游 EI 实际使用的验证集选择模型，可能是 MLP、ridge 或二者 blend。

| 数据集 | Selected relative RMSE | Selected corr | MLP blend weight | Selected 优于 persistence 的 component 数 | 最优整体 relative RMSE |
|---|---:|---:|---:|---:|---|
| SLP | 0.963 | 0.273 | 0.00 | 60 / 60 | Selected MLP/blend, 0.963 |
| 2m air temperature | 0.869 | 0.498 | 0.32 | 38 / 60 | Selected MLP/blend, 0.869 |
| 1000hPa air temperature | 0.896 | 0.447 | 0.10 | 51 / 60 | Selected MLP/blend, 0.896 |
| SST | 0.724 | 0.690 | 1.00 | 1 / 60 | Persistence, 0.599 |

主要诊断：

- SLP、2m 气温和 1000hPa 气温的 selected model 在 RMSE 上优于 zero predictor，也总体优于 persistence；但 SLP 的相关性仍低，说明可预测信号较弱。
- SLP 的 selected model 实际是 0% MLP + 100% ridge，1000hPa 也只有 10% MLP 权重。这说明当前非线性 MLP 对这些月尺度 component 的增益很有限。
- 2m 气温的 MLP 权重为 32%，有一定非线性增益，但 improvement 仍不大。
- SST 是最明显的问题：selected model 是 100% MLP，但 persistence baseline 的整体相对 RMSE 更低，并且 persistence 在 59 / 60 个 component 上优于 selected model。SST 的下游 EI 结果因此需要谨慎解释。

算法上仍有改进空间，优先级如下：

1. 把 persistence/AR skip 纳入验证集候选，而不是只在 MLP 和 ridge 之间 blend。SST 说明简单持久性已经是强基线。
2. 对不同变量使用变量自适应的 `lag` 和 `horizon`。SST 低频记忆强，`lag=4` 个月和 horizon 1 个月下 persistence 很强；大气变量则可能需要不同滞后窗口。
3. 先做 AR、VAR、ridge、MLP residual 的分层对照。当前结果显示很多收益来自线性/自回归结构，MLP 应该学习 residual，而不是替代强线性基线。
4. 对 EI/path-effect 只使用预测验证充分的模型，或在报告中同时给出 prediction skill mask，避免把低 skill component 的 EI 排名解释得过重。
5. 对 SST 单独测试 persistence-aware model 后再重跑 EI，否则当前 SST gateway/mediator 地理分布可能部分反映模型误差结构。

## SLP 月尺度 lag 对齐诊断

如果按 Runge 原文的物理解释，大气依赖通常在一个月内衰减，那么把 monthly SLP 的 `lag=4` 理解成 4 个月确实偏长。直接把 monthly SLP 改成 `lag=1` 后，预测误差略有改善，ACE 的 top 地理中心也更接近 weekly SLP；但 ACS 并没有同步对齐。

进一步测试了两类操作调整：

1. `lag=1/2/3/4`，但 monthly SLP 仍重新 PCA/Varimax。
2. monthly SLP 不重新定义 component，而是投影到 weekly SLP 的 60 个 Varimax loading 上，再用 `lag=1 month` 训练 MLP-TM-EI。

对齐结果保存于 `results/runge_monthly_slp_lag_sensitivity/alignment_summary.md`。表中距离表示 monthly top-5 center 到 weekly top-5 center 的平均最近球面距离，越小越接近。

| run | ACE mean km | ACS mean km | AMCE mean km | test RMSE | test corr |
|---|---:|---:|---:|---:|---:|
| monthly lag=4, refit PCA | 5617 | 5473 | 6054 | 0.947 | 0.273 |
| monthly lag=1, refit PCA | 3321 | 6342 | 5171 | 0.943 | 0.284 |
| monthly lag=1, weekly basis, source-topk | 936 | 6936 | 4314 | 0.928 | 0.237 |
| monthly lag=1, weekly basis, global-quantile | 1937 | 5284 | 5383 | 0.928 | 0.237 |
| monthly lag=1, weekly basis, dense graph | 936 | 5792 | 4592 | 0.928 | 0.237 |
| monthly lag=1, weekly basis, bidirectional-topk | 1937 | 6931 | 3999 | 0.928 | 0.237 |

结论：

- 从物理时间尺度看，monthly SLP 不应默认沿用 weekly 的 `lag=4`；`lag=1 month` 更合理。
- 从地理对齐看，单纯 `lag=1` 只能改善 ACE，不能同时改善 ACS/AMCE。
- 如果目标是接近 weekly SLP 的 component 语义，最有效的操作是固定 weekly Varimax basis，再把 monthly anomaly 投影过去。这样 ACE 几乎对齐，AMCE 也有改善。
- ACS 仍然不稳定。`source_topk` 偏向 outgoing ACE；`global_quantile` 和 dense graph 可以稍微改善 ACS，但会牺牲 AMCE。新增的 `bidirectional_topk` 没有解决 ACS。

因此当前推荐的 monthly SLP 对齐口径是：`weekly basis + lag=1 month + source_topk`，用于对齐 ACE/AMCE；ACS 需要作为低可信、时间尺度敏感的读数单独报告，不应强行解释为已经和 weekly 结果一致。

## 输出位置

- 汇总图：`fig/runge_monthly_variable_comparison/gateway_mediator_centers.png`
- SVG 图：`fig/runge_monthly_variable_comparison/gateway_mediator_centers.svg`
- 预测误差图：`fig/runge_monthly_variable_comparison/prediction_error.png`
- 预测误差摘要：`results/runge_monthly_variable_comparison/prediction_error_summary.md`
- SLP lag 对齐摘要：`results/runge_monthly_slp_lag_sensitivity/alignment_summary.md`
- 总结文件：`results/runge_monthly_variable_comparison/summary.md`
- 每个数据集的 component scores、component maps、manifest 和 gateway/mediator 输出：`results/runge_monthly_variable_comparison/<dataset>/`

## 解释边界

本图显示的是每个 Top component loading map 的中心点，不是完整 loading map 本身。因此它适合快速比较 “中心落在哪些区域”，但不能替代对完整空间模态的逐图检查。

不同变量的 EI/ACE/ACS/AMCE 数值不建议直接当作物理量做跨变量强弱比较；这里主要比较地理分布和同一变量内部的 Top 结构。

经度接近日期变更线的点在作图时做了轻微边界内移，避免标签和椭圆边界裁切；真实经纬度仍保存在 `component_maps.npz` 和下游中心计算结果中。
