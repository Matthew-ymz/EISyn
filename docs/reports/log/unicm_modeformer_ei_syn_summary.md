# UniCM ENSO target EI/Syn 证据报告

## 结论

原文命题认为，在强厄尔尼诺事件发生前，NPMM 和 TNA 与 ENSO 的相互作用增强。这里仅保留 ENSO target：以 UniCM Modeformer 的 full-history maximum-entropy 机制读数来看，8192 样本后的主信号更集中在 ENSO 自身历史与区域 ENSO 指数/太平洋模态的组合上。NPMM/TNA 仍有可见读数，但不再是最强协同候选；`NPMM + TNA` 两个外部 mode 单独构成强二源协同驱动的说法缺乏支持。

本轮使用 `8192` 个最大熵干预样本和 `200` 次 bootstrap；checkpoint seeds、干预范围与 `4096`-sample baseline 保持一致。当前 top-5 为 `ENSO + nino3、ENSO + nino4、ENSO + SPMM、ENSO + IOD、ENSO + NPMM`。具体到远程模态，`ENSO + NPMM` 的平均 Syn 为 `0.002686` bits（rank `5`），`ENSO + TNA` 为 `0.001499` bits（rank `8`）；`NPMM + TNA` 直接到 ENSO 的平均 Syn 为 `-0.000139` bits。由此得到的新 insight 是：NPMM/TNA 对 ENSO 的贡献不宜表述为两个远程 mode 本身的强协同，而应降级为弱到中等的背景态调制信号；当前最稳妥的主结论是 ENSO 自身历史与赤道太平洋区域结构共同控制了主要 Syn 增益。

## EI 证据：ENSO target 主要携带短中期记忆

| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min |
|---|---:|---:|---:|---:|---:|
| ENSO | 0.617162 | 0.395603 | 0.950 | 0.482 | 3 |

![ENSO overall EI](assets/unicm_enso_overall_ei_seed_overlay.png)

*图 1. ENSO target 的 full-history overall EI lead 曲线（最大熵样本 `n=8192`）。彩色细线为 checkpoint seed，黑线为 seed mean，阴影为 seed standard deviation。*

这个趋势说明，UniCM learned mechanism 对 ENSO target 的有效信息主要集中在短中期。ENSO 在 lead 1 到 6 个月的 EI 明显高于后期，符合 ENSO 预测中短期记忆强、长期不确定性上升的物理直觉。

## 单源 EI：NPMM 是更稳定的远程信号，TNA 更像弱调制项

| Target | self EI | strongest non-self sources | NPMM EI | TNA EI |
|---|---:|---|---:|---:|
| ENSO | 0.473612 | nino12 0.015768; nino3 0.015599; IOD 0.013361; SPMM 0.012534 | 0.011671 | 0.005806 |

![ENSO source EI lead curves](assets/unicm_enso_source_ei_rankings.png)

*图 2. ENSO target 的单源 EI lead 曲线。左图单独显示 ENSO self source；右图显示按 lead 平均 EI 选出的非自身 Top-5，并保留 NPMM/TNA。实线和浅色带分别为 checkpoint seed mean 和 standard deviation。*

单源 EI 显示，ENSO 自身区域指数仍是主要信息来源。排除自身后，NPMM 仍处于前列，说明北太平洋经向模态在 UniCM 中携带 ENSO 输出的远程信息。TNA 的单源 EI 较小，说明它不是最强的独立 ENSO source；它更可能通过与 ENSO 背景态或其他太平洋 mode 的组合产生可见影响。

## Syn 证据：主要增益来自 ENSO 自身历史与区域 ENSO 结构

这张表的读法是：先用 `mean Syn 1..24` 和 `rank` 找候选 pair，再用区间与 seed 稳定性判断这个候选能不能信。

- `Target`：被解释的目标变量。这里是 ENSO，也就是看 source pair 对未来 ENSO 的影响。
- `Source pair`：两个输入源的组合，例如 `ENSO + SPMM`。它回答这两个源一起看时，是否比各自单独看多提供了额外信息。
- `rank`：按 `mean Syn 1..24` 从大到小排序。rank 越靠前，平均 Syn 越大；但 rank 只看均值，不等于显著或稳定。
- `mean Syn 1..24`：1 到 24 个月 lead 上的平均 Syn。越大表示二源组合的额外协同增益越强；接近 0 表示几乎没有协同；负值表示联合读数低于两个单源读数之和。
- `Syn seed SD`：3 个 checkpoint seed 的 pair-level Syn 标准差。越大表示不同模型 seed 之间越不稳定。
- `95% CI`：基于 3 个 seed 的 Syn 均值 95% 置信区间。区间跨过 0 时，不能稳妥地说该 pair 一定有正协同。这里 `n=3` 很小，所以 CI 只作为 sanity check。
- `seed rank range`：每个 seed 单独排序后，该 pair 的 rank 范围。范围越窄，排名越稳定；范围很宽说明 rank 对 checkpoint seed 敏感。
- `joint EI 1..24`：两个 source 合起来对 target 的平均 EI，也就是联合输入总共携带多少目标信息。
- `left EI 1..24`：source pair 左边那个源单独对 target 的平均 EI。
- `right EI 1..24`：source pair 右边那个源单独对 target 的平均 EI。

Syn 的计算关系是 `Syn = joint EI - left EI - right EI`。因此，Syn 是几个 EI 项相减后的剩余量；即使 EI 本身较大，Syn 也可能很小，并且会更容易受 checkpoint seed 差异影响。

| Target | Source pair | rank | mean Syn 1..24 | Syn seed SD | 95% CI | seed rank range | joint EI 1..24 | left EI 1..24 | right EI 1..24 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| ENSO | ENSO + nino3 | 1 | 0.005216 | 0.000672 | [0.003545, 0.006886] | 1-3 | 0.494427 | 0.473612 | 0.015599 |
| ENSO | ENSO + nino4 | 2 | 0.005194 | 0.002359 | [-0.000666, 0.011054] | 1-4 | 0.489874 | 0.473612 | 0.011068 |
| ENSO | ENSO + SPMM | 3 | 0.004559 | 0.002518 | [-0.001697, 0.010815] | 2-5 | 0.490705 | 0.473612 | 0.012534 |
| ENSO | ENSO + IOD | 4 | 0.004278 | 0.004353 | [-0.006535, 0.015091] | 1-19 | 0.491251 | 0.473612 | 0.013361 |
| ENSO | ENSO + NPMM | 5 | 0.002686 | 0.002452 | [-0.003404, 0.008777] | 4-9 | 0.487969 | 0.473612 | 0.011671 |
| ENSO | ENSO + nino12 | 6 | 0.002589 | 0.001873 | [-0.002064, 0.007241] | 3-11 | 0.491968 | 0.473612 | 0.015768 |
| ENSO | ENSO + WWV | 7 | 0.001728 | 0.001392 | [-0.001730, 0.005186] | 5-23 | 0.480132 | 0.473612 | 0.004792 |
| ENSO | ENSO + TNA | 8 | 0.001499 | 0.000294 | [0.000768, 0.002230] | 7-9 | 0.480917 | 0.473612 | 0.005806 |
| ENSO | nino12 + nino3 | 9 | 0.001359 | 0.001173 | [-0.001553, 0.004272] | 4-25 | 0.032726 | 0.015768 | 0.015599 |
| ENSO | ENSO + IOB | 10 | 0.001179 | 0.000762 | [-0.000713, 0.003071] | 6-15 | 0.480909 | 0.473612 | 0.006119 |
| ENSO | ENSO + SIOD | 11 | 0.001091 | 0.000776 | [-0.000838, 0.003020] | 7-18 | 0.479104 | 0.473612 | 0.004401 |
| ENSO | nino3 + nino4 | 12 | 0.000686 | 0.000655 | [-0.000942, 0.002314] | 8-33 | 0.027353 | 0.015599 | 0.011068 |
| ENSO | NPMM + nino3 | 14 | 0.000456 | 0.000666 | [-0.001198, 0.002110] | 10-43 | 0.027726 | 0.011671 | 0.015599 |
| ENSO | TNA + nino3 | 49 | -0.000034 | 0.000112 | [-0.000312, 0.000243] | 28-52 | 0.021371 | 0.005806 | 0.015599 |
| ENSO | NPMM + TNA | 55 | -0.000139 | 0.000141 | [-0.000488, 0.000210] | 44-55 | 0.017338 | 0.011671 | 0.005806 |

![ENSO mode-pair Syn leads](assets/unicm_enso_mode_pair_syn_leads.png)

*图 3. ENSO target 的 mode-pair Syn lead 曲线（最大熵样本 `n=8192`）。实线为每个 lead 的 seed mean；同色浅虚线为该 pair 在 lead 1..24 上的平均 Syn，对应上表 `mean Syn 1..24`。黑色虚线为 `NPMM + TNA` 直接二源 Syn。为突出 lead 结构，本图不绘制 checkpoint seed standard deviation。*

对 ENSO target，当前 top-5 是 `ENSO + nino3、ENSO + nino4、ENSO + SPMM、ENSO + IOD、ENSO + NPMM`。这说明 8192 样本下主导的不是单个远程模态，而是 ENSO 自身历史与赤道太平洋区域结构、南太平洋模态及印度洋背景态共同形成的协同增益。NPMM 仍保留一定调制信号，TNA 则更弱；`NPMM + TNA` 直接二源 Syn 接近零，不能作为强协同驱动的主证据。

用地球科学的话说，图 3 的意思很朴素：模型不是只看“ENSO 现在有多强”，还在看“暖异常更偏东、偏中太平洋，还是和其他海盆背景态一起出现”。前 1 到 7 个月，`ENSO + nino3` 和 `ENSO + nino4` 的 Syn 明显更高，说明 ENSO 的短期未来演变对赤道太平洋东西向 SST 结构很敏感；同样强度的 ENSO，如果空间型态不同，后面几个月的增长、衰减和位相演变也可能不同。这个解释和 ENSO diversity 文献是一致的：Trenberth and Stepaniak [1] 早就指出，单一 ENSO 指数不足以描述事件演变，需要额外刻画中东太平洋 SST 梯度；Capotondi et al. [2] 也把事件间差异总结为 ENSO 的振幅、空间型态、生命周期和触发机制差异。Ren and Jin [3] 进一步用 Niño3/Niño4 组合区分两类 ENSO，Kao and Yu [4] 与 Ashok et al. [5] 则分别从 EP/CP ENSO 和 ENSO Modoki 的角度说明，中太平洋型和东太平洋型事件不能简单当作同一种 ENSO 强度的线性放大。因此，这里的 `nino3` 和 `nino4` 更适合被解释为 ENSO 内部空间型态的调制因子，而不是 ENSO 之外的独立强迫源。曲线在 9 到 12 个月后整体贴近零，说明这种额外协同信息主要集中在短中期；到更长 lead，模型已经很难从这些二源组合里读出稳定的增量。

## Rank 与趋势可信度

- Top-5 pair 中 `5/5` 个在 3 个 checkpoint seed 上均为正；这支持 rank 的方向性，但样本数只有 `n=3`。
- 未校正 one-sample t test（跨 3 个 seed 的 pair-mean Syn vs 0）达到 `p<0.05` 的 selected pair: `ENSO + nino3, ENSO + TNA`；Benjamini-Hochberg 校正后达到 `q<0.05` 的 selected pair: `none`。
- `ENSO + nino3` 是当前 rank 1，平均 Syn 为 `0.005216`，95% CI 为 `[0.003545, 0.006886]`，seed rank range 为 `1-3`，正值 seed 数为 `3/3`。
- `ENSO + SPMM` 当前为 rank `3`，平均 Syn 为 `0.004559`，95% CI 为 `[-0.001697, 0.010815]`；它仍是重要候选，但不再适合作为唯一主导机制来表述。
- `ENSO + NPMM` 与 `ENSO + TNA` 分别为 rank `5` 和 rank `8`；二者方向均为正，但平均 Syn 较 top pair 明显更小，更适合解释为 ENSO 背景态上的弱到中等调制信号。
- `NPMM + TNA` 直接二源 Syn 的均值为 `-0.000139`，95% CI 为 `[-0.000488, 0.000210]`，且 seed rank range 为 `44-55`；这不足以支持强直接协同。

因为只有 3 个 checkpoint seed，显著性检验的自由度只有 2；这里的 p 值和 CI 只能作为 sanity check，不能替代更多 checkpoint 或更高样本的复核。rank 可信度更适合用三件事一起判断：跨 seed 是否同号、seed rank range 是否窄、lead 曲线是否在关键窗口保持同方向。

## 4096→8192 样本收敛

图 3 不再绘制误差棒；下表中的 `checkpoint SD` 仍定义为 3 个 checkpoint 的 lead-wise 标准差。`bootstrap SD` 单独估计固定 checkpoint 下最大熵采样的 Monte Carlo 波动，不与 checkpoint 间差异混用。

| Source pair | mean Syn 4096 | mean Syn 8192 | rank 4096→8192 | mean checkpoint SD 4096 | mean checkpoint SD 8192 | SD ratio | mean bootstrap SD 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENSO + SPMM | 0.005165 | 0.004559 | 2→3 | 0.004035 | 0.003349 | 0.830 | 0.001157 |
| ENSO + nino3 | 0.006120 | 0.005216 | 1→1 | 0.004711 | 0.003186 | 0.676 | 0.001235 |
| ENSO + NPMM | 0.003126 | 0.002686 | 6→5 | 0.003232 | 0.002675 | 0.828 | 0.001016 |
| ENSO + TNA | 0.001076 | 0.001499 | 10→8 | 0.000843 | 0.001332 | 1.581 | 0.000877 |
| NPMM + TNA | -0.000046 | -0.000139 | 45→55 | 0.000143 | 0.000186 | 1.302 | 0.000216 |

固定比较的 5 个 pair 中，`3/5` 个 pair 的平均 lead-wise checkpoint SD 在 8192 样本下下降。该比例只说明采样收敛减少了部分估计噪声；剩余 checkpoint SD 仍包含真实的 checkpoint 机制差异。

## 解释边界

本报告只分析 frozen UniCM Modeformer learned mechanism，不是 reanalysis 事件复现实验，也不是 1983 或 1997 个例归因。这里的 EI/Syn 使用 Gaussian log-det 估计，适合作为 full-history 机制筛查；若要把结论推进到最终 PEID 或事件级归因，需要进一步做非线性 transport-map PEID、高样本复核，或按厄尔尼诺事件窗口构造条件化干预。

## 参考文献

[1] Trenberth, K. E., & Stepaniak, D. P. (2001). Indices of El Niño Evolution. *Journal of Climate*, 14(8), 1697-1701. https://doi.org/10.1175/1520-0442(2001)014%3C1697:LIOENO%3E2.0.CO;2

[2] Capotondi, A., Wittenberg, A. T., Newman, M., Di Lorenzo, E., Yu, J.-Y., Braconnot, P., Cole, J., Dewitte, B., Giese, B., Guilyardi, E., Jin, F.-F., Karnauskas, K., Kirtman, B., Lee, T., Schneider, N., Xue, Y., & Yeh, S.-W. (2015). Understanding ENSO Diversity. *Bulletin of the American Meteorological Society*, 96(6), 921-938. https://doi.org/10.1175/BAMS-D-13-00117.1

[3] Ren, H.-L., & Jin, F.-F. (2011). Niño indices for two types of ENSO. *Geophysical Research Letters*, 38, L04704. https://doi.org/10.1029/2010GL046031

[4] Kao, H.-Y., & Yu, J.-Y. (2009). Contrasting Eastern-Pacific and Central-Pacific Types of ENSO. *Journal of Climate*, 22(3), 615-632. https://doi.org/10.1175/2008JCLI2309.1

[5] Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *Journal of Geophysical Research: Oceans*, 112, C11007. https://doi.org/10.1029/2006JC003798
