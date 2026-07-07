# UniCM Modeformer 全历史 mode-pair Syn 分析

## 结论

本轮使用与 full-history overall EI 图完全相同的干预集合和 checkpoint 输出缓存，计算 source mode pair 到目标 mode lead 输出的 Gaussian log-det Syn。结果是 mode-level 二源筛查读数，不重新运行 UniCM forward。

- checkpoint seeds: `1, 2, 3`
- intervention samples: `8192`
- intervention support: all 12 historical months x 11 mode dimensions sampled independently from `[-4, 4]`
- sampling seed: `20260619`
- start month: `0`
- bootstrap repeats: `0`
- source modes: `nino, NPMM, SPMM, IOB, IOD, SIOD, TNA, nino12, nino3, nino4, WWV`
- target modes: `nino, IOD`
- reused prediction cache: `results/unicm_overall_ei_cpu_bound4_n8192/cache`

## 估计口径

每个 source mode 使用该 mode 的 12 个月历史向量作为一个多维源；target 是对应 lead 的单个目标 mode 输出。对每个 `(left, right, target, lead, checkpoint seed)` 估计 `I(left; target)`、`I(right; target)` 和 `I(left,right; target)`，Syn 定义为 `joint EI - left EI - right EI`。由于其他历史 mode 同步来自同一 full-history maximum-entropy intervention ensemble，但不进入 source 集合，读数是在这些 nuisance intervention variables 上边缘化后的 mode-pair Syn。

## Top source pairs

### nino

| Rank | Source pair | mean Syn 1..24 | joint EI | left EI | right EI |
|---:|---|---:|---:|---:|---:|
| 1 | nino + nino3 | 0.005216 | 0.494427 | 0.473612 | 0.015599 |
| 2 | nino + nino4 | 0.005194 | 0.489874 | 0.473612 | 0.011068 |
| 3 | nino + SPMM | 0.004559 | 0.490705 | 0.473612 | 0.012534 |
| 4 | nino + IOD | 0.004278 | 0.491251 | 0.473612 | 0.013361 |
| 5 | nino + NPMM | 0.002686 | 0.487969 | 0.473612 | 0.011671 |
| 6 | nino + nino12 | 0.002589 | 0.491968 | 0.473612 | 0.015768 |
| 7 | nino + WWV | 0.001728 | 0.480132 | 0.473612 | 0.004792 |
| 8 | nino + TNA | 0.001499 | 0.480917 | 0.473612 | 0.005806 |
| 9 | nino12 + nino3 | 0.001359 | 0.032726 | 0.015768 | 0.015599 |
| 10 | nino + IOB | 0.001179 | 0.480909 | 0.473612 | 0.006119 |

### IOD

| Rank | Source pair | mean Syn 1..24 | joint EI | left EI | right EI |
|---:|---|---:|---:|---:|---:|
| 1 | IOD + SIOD | 0.012107 | 0.350317 | 0.320329 | 0.017881 |
| 2 | nino + IOD | 0.007147 | 0.347583 | 0.020106 | 0.320329 |
| 3 | IOD + nino4 | 0.005648 | 0.345515 | 0.320329 | 0.019538 |
| 4 | NPMM + IOD | 0.005263 | 0.336838 | 0.011246 | 0.320329 |
| 5 | SPMM + IOD | 0.004950 | 0.334724 | 0.009445 | 0.320329 |
| 6 | IOB + IOD | 0.004779 | 0.333782 | 0.008674 | 0.320329 |
| 7 | IOD + TNA | 0.003301 | 0.331301 | 0.320329 | 0.007671 |
| 8 | IOD + WWV | 0.003011 | 0.329901 | 0.320329 | 0.006562 |
| 9 | IOD + nino3 | 0.002660 | 0.343585 | 0.320329 | 0.020596 |
| 10 | IOD + nino12 | 0.002530 | 0.333587 | 0.320329 | 0.010729 |

![Top mode-pair Syn curves](../../results/unicm_full_history_pair_syn_tm_degree1_n8192/fig/full_history_mode_pair_syn_top.png)

*图 1. 每个 target 按 1..24 lead 平均 Syn 排名前五的 source-mode pair 曲线；曲线为 checkpoint seed 均值。*

## 图表与数据

- raw rows: `results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_rows.jsonl`
- pair summary: `results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_summary.csv`
- lead summary: `results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_lead_summary.csv`
- top pairs: `results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_top_pairs.csv`
- figure: `results/unicm_full_history_pair_syn_tm_degree1_n8192/fig/full_history_mode_pair_syn_top.png`

## 解释边界

- 后端与 overall EI 图一致，使用 Gaussian log-det MI；这适合快速筛查，不等同于 transport-map PEID 的最终非线性分解。
- Syn 可以为负，表示 pair 的联合读数低于两个单源读数之和；这里不做非负截断。
- 结果只对应 frozen UniCM Modeformer learned mechanism，不是 reanalysis 预测技能评估。
