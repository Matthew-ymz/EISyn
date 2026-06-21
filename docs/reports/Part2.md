# Part 2: UniCM ENSO EI/Syn 机制证据

## 结论

本部分把 UniCM Modeformer 的全历史最大熵 EI、二源 Syn 和三源 interaction 结果合并为一个口径一致的 ENSO 机制报告。当前最稳妥的结论是：未来 ENSO 的主要机制读数来自 ENSO 自身历史与赤道太平洋空间结构的共同约束，而不是来自 `NPMM + TNA` 两个远程模态的直接强协同驱动。

在 `8192` 个最大熵干预样本、checkpoint seeds `1,2,3` 和 `200` 次 bootstrap 下，`ENSO + nino3` 是二源 Syn 的 rank 1，平均 Syn 为 `0.005216` bits；`ENSO + nino4`、`ENSO + SPMM`、`ENSO + IOD` 和 `ENSO + NPMM` 也位于前列。`ENSO + NPMM` 为 rank `5`，平均 Syn 为 `0.002686` bits；`ENSO + TNA` 为 rank `8`，平均 Syn 为 `0.001499` bits。相比之下，`NPMM + TNA` 直接到 ENSO 的平均 Syn 为 `-0.000139` bits，不能支持“两个远程模态直接联合驱动 ENSO”的强表述。

所以更合适的地球科学说法是：NPMM/TNA 可以保留为 ENSO 背景态上的弱到中等调制信号；`nino3`、`nino4`、`nino12` 则更像 ENSO 内部空间型态和东西向 SST 结构的调制因子。

## 实验口径

这里分析的是 frozen UniCM Modeformer learned mechanism，不是 reanalysis 预测技能评估，也不是单个历史事件归因。每个干预样本同时采样 12 个历史月份和 11 个 UniCM mode 维度，形成 `(B, 12, 11)` 的 bounded uniform 最大熵输入，历史张量写入 Modeformer encoder 的 12 个月历史段，未来 24 个月由 decoder 在 `train=False` 下自回归生成。

核心配置如下：

| Item | Value |
|---|---|
| checkpoint seeds | `1, 2, 3` |
| current intervention samples | `8192` |
| pair convergence baseline | `4096` |
| triple convergence baseline | `1024` |
| intervention support | all 12 historical months x 11 mode dimensions sampled independently from `[-4, 4]` |
| sampling seed | `20260619` |
| bootstrap repeats | `200` |
| target mode | ENSO |
| source modes | ENSO, NPMM, SPMM, IOB, IOD, SIOD, TNA, nino12, nino3, nino4, WWV |

整体 EI 使用 flattened full-history source，即 132 维历史 mode 输入，对每个 lead 的 ENSO 输出估计 `I(history; target_lead)`。先定义二源 Syn：

```math
\mathrm{Syn}_{ij}=EI_{ij}-EI_i-EI_j.
```

再用布尔子集格 Möbius 反演定义三源 interaction：

```math
\Delta_{ijk}=EI_{ijk}-EI_{ij}-EI_{ik}-EI_{jk}+EI_i+EI_j+EI_k.
```

所有这些读数都使用 Gaussian log-det 估计，适合作为 full-history 机制筛查；它们不等同于最终的非线性 transport-map PEID 分解。

## Mode 地理含义

![UniCM mode geography](assets/unicm_mode_geography.png)

*图 1. UniCM mode 输入的地理区域。ENSO 相关指数来自赤道太平洋不同经向区段；NPMM、SPMM 和 TNA 提供太平洋经向模态与热带北大西洋背景；IOD/SIOD/IOB 表示印度洋盆地和偶极型 SST 结构。*

这张图是解释后续 EI/Syn 的基础。`nino3`、`nino4` 和 `nino12` 不是 ENSO 之外的独立外部强迫，而是赤道太平洋内部空间结构的不同读数。因此当 `ENSO + nino3` 或 `ENSO + nino4` 出现高 Syn 时，更自然的解释是 ENSO 的当前强度需要和东西向 SST 型态一起读，才能判断未来几个月的演变。

## Overall EI: ENSO 信息主要集中在短中期

| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min |
|---|---:|---:|---:|---:|---:|
| ENSO | 0.617162 | 0.395603 | 0.950 | 0.482 | 3 |

![ENSO overall EI](assets/unicm_enso_overall_ei_seed_overlay.png)

*图 2. ENSO target 的 full-history overall EI lead 曲线。彩色细线为 checkpoint seed，黑线为 seed mean，阴影为 seed standard deviation。*

这张图说明，UniCM learned mechanism 对 ENSO 的有效信息主要集中在 lead 1 到 6 个月。短 lead 的 EI 明显高于后期，符合 ENSO 预测中短期记忆强、长期不确定性上升的物理直觉。三个 checkpoint 的曲线形状相近，Pearson min 达到 `0.950`；但 Spearman min 只有 `0.482`，说明不同 checkpoint 对具体 lead 排序仍不够稳定。因此 overall EI 可以支持“短中期记忆强”的方向性判断，但不能把每个 lead 的细粒度排序解释得太重。

## 单源 EI: NPMM 可见，TNA 较弱

| Target | self EI | strongest non-self sources | NPMM EI | TNA EI |
|---|---:|---|---:|---:|
| ENSO | 0.473612 | nino12 0.015768; nino3 0.015599; IOD 0.013361; SPMM 0.012534 | 0.011671 | 0.005806 |

![ENSO source EI lead curves](assets/unicm_enso_source_ei_rankings.png)

*图 3. ENSO target 的单源 EI lead 曲线。左图单独显示 ENSO self source；右图显示按 24 个月平均 EI 选出的非自身 Top-5，并保留 NPMM/TNA。实线和浅色带分别为 checkpoint seed mean 和 standard deviation。*

单源 EI 曲线显示，ENSO 自身历史在短 lead 占绝对主导，但随后快速衰减。排除自身后，`nino3`、`nino12` 和 IOD 的 EI 随 lead 增长并在较长 lead 位于前列，NPMM 则在中期达到较高水平后回落；这些长 lead 曲线的 checkpoint 波动也明显扩大，因此不宜过度解释精细排序。TNA 的曲线始终较低，更稳妥的说法是，它可能只在 ENSO 背景态或其他太平洋/印度洋模态共同存在时提供弱增量。

## 二源 Syn: 主增益来自 ENSO 历史和区域 ENSO 结构

| Target | Source pair | rank | mean Syn 1..24 | Syn seed SD | 95% CI | seed rank range | joint EI 1..24 | left EI 1..24 | right EI 1..24 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| ENSO | ENSO + nino3 | 1 | 0.005216 | 0.000672 | [0.003545, 0.006886] | 1-3 | 0.494427 | 0.473612 | 0.015599 |
| ENSO | ENSO + nino4 | 2 | 0.005194 | 0.002359 | [-0.000666, 0.011054] | 1-4 | 0.489874 | 0.473612 | 0.011068 |
| ENSO | ENSO + SPMM | 3 | 0.004559 | 0.002518 | [-0.001697, 0.010815] | 2-5 | 0.490705 | 0.473612 | 0.012534 |
| ENSO | ENSO + IOD | 4 | 0.004278 | 0.004353 | [-0.006535, 0.015091] | 1-19 | 0.491251 | 0.473612 | 0.013361 |
| ENSO | ENSO + NPMM | 5 | 0.002686 | 0.002452 | [-0.003404, 0.008777] | 4-9 | 0.487969 | 0.473612 | 0.011671 |
| ENSO | ENSO + nino12 | 6 | 0.002589 | 0.001873 | [-0.002064, 0.007241] | 3-11 | 0.491968 | 0.473612 | 0.015768 |
| ENSO | ENSO + TNA | 8 | 0.001499 | 0.000294 | [0.000768, 0.002230] | 7-9 | 0.480917 | 0.473612 | 0.005806 |
| ENSO | NPMM + TNA | 55 | -0.000139 | 0.000141 | [-0.000488, 0.000210] | 44-55 | 0.017338 | 0.011671 | 0.005806 |

![ENSO mode-pair Syn leads](assets/unicm_enso_mode_pair_syn_leads.png)

*图 4. ENSO target 的 mode-pair Syn lead 曲线。实线为每个 lead 的 seed mean；同色浅虚线为该 pair 在 lead 1..24 上的平均 Syn；黑色虚线为 `NPMM + TNA` 直接二源 Syn。*

这张图的核心信息很直接：模型不是只看“ENSO 现在有多强”，还在看“暖异常更偏东、偏中太平洋，还是和其他海盆背景态一起出现”。前 1 到 7 个月，`ENSO + nino3` 和 `ENSO + nino4` 的 Syn 明显更高，说明 ENSO 的短期未来演变对赤道太平洋东西向 SST 结构很敏感。同样强度的 ENSO，如果空间型态不同，后续几个月的增长、衰减和位相演变也可能不同。

这个解释和 ENSO diversity 文献一致。Trenberth and Stepaniak [1] 指出，单一 ENSO 指数不足以描述事件演变，需要额外刻画中东太平洋 SST 梯度；Capotondi et al. [2] 把事件间差异总结为 ENSO 的振幅、空间型态、生命周期和触发机制差异；Ren and Jin [3] 进一步用 Niño3/Niño4 组合区分两类 ENSO。Kao and Yu [4] 与 Ashok et al. [5] 则分别从 EP/CP ENSO 和 ENSO Modoki 角度说明，中太平洋型和东太平洋型事件不能简单当作同一种 ENSO 强度的线性放大。

因此，`nino3` 和 `nino4` 更适合被解释为 ENSO 内部空间型态的调制因子，而不是 ENSO 之外的独立强迫源。曲线在 9 到 12 个月后整体贴近零，说明这种额外协同信息主要集中在短中期；到更长 lead，模型已经很难从这些二源组合里读出稳定的增量。

## 可信度和样本收敛

Top-5 pair 中 `5/5` 个在 3 个 checkpoint seed 上均为正，这支持 rank 的方向性；但 checkpoint 数只有 `n=3`，显著性检验和 CI 只能作为 sanity check。未校正 one-sample t test 中，`ENSO + nino3` 和 `ENSO + TNA` 达到 `p<0.05`；Benjamini-Hochberg 校正后没有 selected pair 达到 `q<0.05`。所以表述应以“候选机制”和“调制信号”为主，不应写成已确认的物理因果通道。

| Source pair | mean Syn 4096 | mean Syn 8192 | rank 4096→8192 | checkpoint SD 4096 | checkpoint SD 8192 | SD ratio | bootstrap SD 4096 | bootstrap SD 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ENSO + SPMM | 0.005165 | 0.004559 | 2→3 | 0.004035 | 0.003349 | 0.830 | 0.001797 | 0.001157 |
| ENSO + nino3 | 0.006120 | 0.005216 | 1→1 | 0.004711 | 0.003186 | 0.676 | 0.001988 | 0.001235 |
| ENSO + NPMM | 0.003126 | 0.002686 | 6→5 | 0.003232 | 0.002675 | 0.828 | 0.001636 | 0.001016 |
| ENSO + TNA | 0.001076 | 0.001499 | 10→8 | 0.000843 | 0.001332 | 1.581 | 0.001365 | 0.000877 |
| NPMM + TNA | -0.000046 | -0.000139 | 45→55 | 0.000143 | 0.000186 | 1.302 | 0.000400 | 0.000216 |

全部 55 个 pair 的 4096→8192 排名 Spearman 为 `0.878`，top-5 重合 `4/5`。固定比较的 5 个 pair 中，`3/5` 个 pair 的平均 lead-wise checkpoint SD 下降；同时这 5 个 pair 的 bootstrap SD 均下降。二源主排序总体稳定，但 checkpoint 差异并不会随干预样本数增加而必然单调下降。

## 三源 interaction: 高阶增量仍依赖 ENSO 背景态

三源 interaction 已在同一组 8192-sample cache 上重算，覆盖 11 个 source mode 的全部 165 个无序三元组和 24 个 leads。

| Rank | Source triple | mean delta3 | seed SD | seed rank range | positive seeds | joint EI |
|---:|---|---:|---:|---|---:|---:|
| 1 | ENSO + nino3 + nino4 | 0.000282 | 0.000185 | 1-10 | 3/3 | 0.511657 |
| 2 | ENSO + nino12 + nino4 | 0.000235 | 0.000111 | 4-9 | 3/3 | 0.508868 |
| 3 | ENSO + SPMM + nino12 | 0.000225 | 0.000140 | 2-8 | 3/3 | 0.509713 |
| 4 | ENSO + nino12 + nino3 | 0.000203 | 0.000177 | 1-63 | 3/3 | 0.514345 |
| 5 | ENSO + NPMM + nino4 | 0.000195 | 0.000165 | 3-14 | 3/3 | 0.504753 |

![Top UniCM third-order EI interactions](assets/unicm_enso_mode_triple_interaction_leads.png)

*图 5. 平均三阶 interaction 排名前五的 lead 曲线。点线和误差棒分别为三个 checkpoint seed 的均值与标准差；同色虚线为该三元组在全部 lead 和 seed 上的平均值。*

8192 结果的 top-10 仍有 `10/10` 个三元组包含 ENSO 自身历史，因此“高阶增量依赖 ENSO 背景态”的方向未变。但是三源细排名并未收敛：1024→8192 的 165 项 rank Spearman 仅为 `0.016`，top-5 只重合 `1/5`，top-10 只重合 `2/10`；原 1024 rank 1 的 `ENSO + TNA + nino4` 在 8192 下为 rank `8`。因此三源结果只能支持背景态层面的弱结论，不能支持具体 top triple 的稳定机制排序。

## 数据与归档

当前主报告引用的核心图保留在 `docs/reports/assets/`。旧的 UniCM 单项报告已移入 `docs/reports/log/`，用于保留中间读数和早期解释上下文。

机器可读结果保留在：

- `results/unicm_overall_ei_cpu_bound4_n4096/`
- `results/unicm_full_history_mode_pair_syn_cpu_bound4_n4096/`
- `results/unicm_full_history_mode_triple_syn_cpu_bound4_n1024/`
- `results/unicm_overall_ei_cpu_bound4_n8192/`
- `results/unicm_full_history_mode_pair_syn_cpu_bound4_n8192/`
- `results/unicm_full_history_mode_triple_syn_cpu_bound4_n8192/`

## 解释边界

本文只分析 frozen UniCM Modeformer learned mechanism。EI/Syn 结果是机制筛查，不是预测技能评分、reanalysis 事件复现，也不是 1983 或 1997 个例归因。若要把结论推进到最终 PEID 或事件级归因，需要进一步做非线性 transport-map PEID、高样本复核，或按厄尔尼诺事件窗口构造条件化干预。

## 参考文献

[1] Trenberth, K. E., & Stepaniak, D. P. (2001). Indices of El Niño Evolution. *Journal of Climate*, 14(8), 1697-1701. https://doi.org/10.1175/1520-0442(2001)014%3C1697:LIOENO%3E2.0.CO;2

[2] Capotondi, A., Wittenberg, A. T., Newman, M., Di Lorenzo, E., Yu, J.-Y., Braconnot, P., Cole, J., Dewitte, B., Giese, B., Guilyardi, E., Jin, F.-F., Karnauskas, K., Kirtman, B., Lee, T., Schneider, N., Xue, Y., & Yeh, S.-W. (2015). Understanding ENSO Diversity. *Bulletin of the American Meteorological Society*, 96(6), 921-938. https://doi.org/10.1175/BAMS-D-13-00117.1

[3] Ren, H.-L., & Jin, F.-F. (2011). Niño indices for two types of ENSO. *Geophysical Research Letters*, 38, L04704. https://doi.org/10.1029/2010GL046031

[4] Kao, H.-Y., & Yu, J.-Y. (2009). Contrasting Eastern-Pacific and Central-Pacific Types of ENSO. *Journal of Climate*, 22(3), 615-632. https://doi.org/10.1175/2008JCLI2309.1

[5] Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *Journal of Geophysical Research: Oceans*, 112, C11007. https://doi.org/10.1029/2006JC003798
