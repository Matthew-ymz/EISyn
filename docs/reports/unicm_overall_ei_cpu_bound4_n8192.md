# UniCM Modeformer 全历史最大熵整体 EI seed 鲁棒性分析

## 结论

全历史最大熵采样下，整体 EI lead 曲线未通过 seed 鲁棒性标准。

- checkpoint seeds: `1, 2, 3`
- intervention samples: `8192`
- intervention support: all 12 historical months x 11 mode dimensions sampled independently from `[-4, 4]`
- sampling seed: `20260619`
- bootstrap repeats: `200`
- start month: `0`
- target-mean overall EI 1..24: `0.617162` bits
- 主窗口：lead `1..24`；climate-relevant 补充窗口：lead `6..18`
- 通过标准：seed-pair Pearson >= `0.80`，Spearman >= `0.75`，top-3 EI lead overlap >= `2`。

## 干预口径

这里把 UniCM mode 分支的完整历史输入视为机制输入。每个样本同时采样 12 个历史月份和 11 个 mode 维度，形成 `(B, 12, 11)` 的 bounded uniform 最大熵输入；该历史张量写入模型 encoder 的 12 个月历史段，未来 24 个月仍由 decoder 在 `train=False` 下自回归生成。

整体 EI 读数使用 flattened full-history source，即 132 维历史 mode 输入，对每个目标 mode 和 lead 分别估计 `I(history_{1:12,1:11}; target_lead)`。高维整体读数采用 Gaussian log-det MI 作为快速筛查口径；它用于检查绝对量级和 seed 稳定性，不等同于二源 PEID/Syn 分解。

## Overall EI target 排名

| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min | status |
|---|---:|---:|---:|---:|---:|---|
| nino | 0.617162 | 0.395603 | 0.950 | 0.482 | 3 | 不稳定 |

![Full-history overall EI seed overlay](../../results/unicm_overall_ei_cpu_bound4_n8192/fig/overall_ei_seed_overlay.png)

*图 1. Full-history overall EI lead curves under the selected bounded maximum-entropy intervention. Each panel is one target mode and each curve is one checkpoint seed; stable targets should show both similar curve shape and similar lead ordering across seeds.*

## 不稳定 target

- nino: Pearson min `0.950`, Spearman min `0.482`, top-3 overlap min `3`。

## 图表与数据

- 逐 seed / target / lead 原始结果：`results/unicm_overall_ei_cpu_bound4_n8192/overall_ei_rows.jsonl`
- target 鲁棒性汇总：`results/unicm_overall_ei_cpu_bound4_n8192/overall_ei_seed_robustness_summary.csv`
- lead-level seed mean/std：`results/unicm_overall_ei_cpu_bound4_n8192/overall_ei_seed_lead_summary.csv`
- 图：`results/unicm_overall_ei_cpu_bound4_n8192/fig/overall_ei_seed_overlay.png`

## 解释边界

- 本报告只分析 frozen UniCM checkpoint 的 Modeformer learned mechanism，不使用 reanalysis 数据做预测复现。
- 这里的整体 EI 是全历史输入到单目标 lead 输出的 Gaussian log-det 读数，用于量级和 seed 稳定性筛查。
- 若整体 EI 量级和 seed 稳定性足够，再继续对指定 source pair 做二源 PEID/Syn 分解更合理。
