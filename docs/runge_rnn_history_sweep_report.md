# Runge RNN 历史长度遍历实验最终报告

## 摘要

本实验在真实 Runge weekly 60 维 component score 上放开固定 `lag=4` 限制，使用 GRU/RNN 类模型做 `1,2,4,8` 周多步预测。调参只使用 train/validation；test split 只用于最终 held-out 对比和 paired circular block bootstrap。

最终 validation 选择的配置为：

- `history=2`
- `hidden_dim=128`
- `dropout=0.0`
- `weight_decay=1e-5`
- `seed=42`
- `rnn_objective=rollout_multistep`
- `RNN/Ridge` horizon-wise validation blend

核心结论：**更长历史没有带来更好泛化；最佳历史长度反而是 2 周。** 相比同框固定 `history=4` 候选，最佳 `history=2` 配置的 validation average RMSE 降低 `0.00156-0.00166`，test average RMSE 降低 `0.00078-0.00105`。收益仍主要集中在 `h=4` 和 `h=8`，短 horizon 基本退回 Ridge。

## 搜索结果

共完成 `118` 个候选：

- 初筛：`history=[1,2,4,8,12,16,24,32,52]`
- refinement：对 validation 前 3 的 `history=2,4,1` 搜索 `hidden_dim=[64,128,192,256]`、`dropout=[0,0.1,0.2]`、`weight_decay=[1e-5,1e-4,1e-3]`
- seed 复核：对前 2 个 validation 配置补跑 `seed=43,44`

按每个 history 的最佳 validation average RMSE 汇总：

| history | best val avg RMSE | best test avg RMSE |
|---:|---:|---:|
| 2 | 0.758156 | 0.765274 |
| 4 | 0.759719 | 0.766053 |
| 1 | 0.762180 | 0.769253 |
| 12 | 0.767089 | 0.770073 |
| 8 | 0.768252 | 0.771682 |
| 16 | 0.772437 | 0.774087 |
| 24 | 0.785233 | 0.783849 |
| 32 | 0.795887 | 0.792731 |
| 52 | 0.814634 | 0.810308 |

更长 history 明显恶化，说明当前 weekly component 预测任务中，RNN 没有从长历史窗口中获得稳定收益，反而增加了泛化压力。

## 最终 Test 对比

最终 held-out test metrics 如下，`BestBaseline` 是同一 horizon 上 `MLP` 与 `TunedRidge` 的较优者。

| horizon | RNN RMSE | MLP RMSE | TunedRidge RMSE | BestBaseline RMSE | RNN improvement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.709416 | 0.709445 | 0.709416 | 0.709416 | 0.000000 |
| 2 | 0.774509 | 0.774607 | 0.774509 | 0.774509 | 0.000000 |
| 4 | 0.787236 | 0.788960 | 0.788826 | 0.788826 | 0.001590 |
| 8 | 0.789933 | 0.795105 | 0.794919 | 0.794919 | 0.004986 |

Bootstrap 检验使用 paired circular block bootstrap，`block_size=26`，`reps=5000`：

| horizon | best baseline | RMSE improvement | 95% CI | p(improvement <= 0) |
|---:|---|---:|---|---:|
| 1 | TunedRidge | 0.000000 | [0.000000, 0.000000] | 1.000 |
| 2 | TunedRidge | 0.000000 | [0.000000, 0.000000] | 1.000 |
| 4 | TunedRidge | 0.001590 | [0.000343, 0.003249] | 0.003 |
| 8 | TunedRidge | 0.004986 | [0.002053, 0.008664] | 0.000 |

结论应表述为：`history=2` 的 validation-selected GRU forecast system 在 `h=4` 和 `h=8` 上显著优于最强 baseline；在 `h=1` 和 `h=2` 上最终预测退化为/等同于 tuned Ridge，不能声称 RNN 有额外收益。

## 60 维变量误差分布

为了检查 60 个 component 是否存在明显预测难度差异，进一步统计了最终 RNN 系统在 test split 上的 component-level RMSE。结果显示变量间误差差异较明显：

- 按 `h=1,2,4,8` 平均后的 component RMSE 范围为 `0.676070-0.869315`。
- 中位数为 `0.762644`，10%-90% 分位为 `0.714656-0.809245`。
- 最难预测的分量包括 `component_11`、`component_30`、`component_35`、`component_42`、`component_10`。
- 最容易预测的分量包括 `component_36`、`component_01`、`component_43`、`component_44`、`component_49`。
- horizon 越长，分布整体上移：median RMSE 从 h=1 的 `0.713166` 增至 h=8 的 `0.785428`。

![Component-level RMSE distribution](../fig/runge/rnn_history_sweep/component_rmse_distribution.png)

配套数据：

- `results/runge_rnn_history_sweep/component_rmse_summary.csv`
- `results/runge_rnn_history_sweep/component_rmse_distribution_by_horizon.csv`
- `results/runge_rnn_history_sweep/component_rmse_distribution_summary.json`

## 与固定 `history=4` 对比

同一 sweep 内的固定历史候选：

- 原固定配置近似候选：`history_04_h192_do0p0000_wd1em04_seed42`
  - validation average RMSE: `0.759818`
  - test average RMSE: `0.766328`
- `history=4` 内最佳候选：`history_04_h64_do0p0000_wd1em05_seed42`
  - validation average RMSE: `0.759719`
  - test average RMSE: `0.766053`
- 最终 `history=2` 候选：
  - validation average RMSE: `0.758156`
  - test average RMSE: `0.765274`

因此，放开 history 后的收益是小幅但方向一致的：最佳 `history=2` 相对固定 `history=4` 的 validation/test 平均 RMSE 都更低。但这个收益主要来自避免过长输入窗口，而不是发现了更长的可用记忆。

## 产物

- 候选排行榜：`results/runge_rnn_history_sweep/leaderboard.csv`
- 最终 test 指标：`results/runge_rnn_history_sweep/final_test_metrics.csv`
- 显著性检验：`results/runge_rnn_history_sweep/final_prediction_significance.json`
- sweep 图：`fig/runge/rnn_history_sweep/history_sweep_rmse.png`
- 最终多步 RMSE 图：`fig/runge/rnn_history_sweep/final_multistep_rmse.png`
- component 误差分布图：`fig/runge/rnn_history_sweep/component_rmse_distribution.png`
- 逐候选日志：`docs/log/runge_rnn_history_sweep_candidates.csv`
- 运行日志：`docs/log/runge_rnn_history_sweep_run.log`

三张 PNG 已目视检查，legend 均在轴外或空白区域，没有覆盖数据。
