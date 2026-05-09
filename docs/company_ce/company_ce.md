# 企业财务数据动力学中的 PEID 因果超图实验

本文整理公司年度面板数据上的 PEID 因果超图实验。它把原先分散在项目背景、数据处理、单尺度 PEID 计划、鲁棒性检验和文献解释中的内容合并为一个统一说明：先交代研究问题，再说明公司面板如何转成离散跨期状态，随后解释 EI/PEID 指标、当前主结果、资产-负债协同机制、鲁棒性结论和后续工作。

## 研究问题与项目定位

企业增长文献的入口问题是：企业增长是否完全随机，还是存在弱但非零的 persistence、极端状态回跳、反转或更一般的非随机转移结构。传统 MCM/QTPM 路线通常把单个增长变量离散为分位状态，构造增长状态转移矩阵，再和随机转移矩阵比较。这个项目把该入口扩展到多变量财务动力学：如果只看单变量增长矩阵，就无法表达资产、收入、员工、负债、现金、利润和成本之间的联合状态；如果构建多变量离散转移机制，就可以进一步估计变量之间的有效信息流和多源协同。

本文的长期研究问题仍然是企业财务系统是否存在更有效的宏观动力学表示，即某种粗粒化后的财务状态是否比原始微观变量层有更高的 dynamic effectiveness。但当前实验是更窄的一步：固定在单尺度公司变量上，不做粗粒化搜索，也不声称发现了严格意义上的真实因果涌现。它回答的是：在公司年度增长状态中，哪些单变量和变量组合对下一期状态具有 PEID/EI 意义下的候选机制信息结构。

这种定位很重要。真实企业数据是 observational panel，不是随机干预实验。当前图中的“因果”采用 EI/PEID 框架中的最大熵干预语义，表示在估计的离散转移机制下，源变量状态对下一期目标状态的机制性信息强度。它不等同于观察相关，但也不是标准计量经济学中的后门识别或政策干预效应。因此后续表述应使用“PEID 有效信息结构”“候选跨期机制信息”或“协同信息结构”，避免写成已经识别出无争议的经营干预因果。

## 数据来源与变量定义

当前主实验使用输入文件：

```text
data/inf_compustat_anual_US_filter_feas.csv
```

主变量为：

```text
inf_at, inf_revt, emp, inf_lt, inf_ch, inf_ni, inf_cogs
```

含义如下：

| 变量 | 含义 | 当前增长率变换 |
| --- | --- | --- |
| `inf_at` | 通胀调整总资产 | `log_growth` |
| `inf_revt` | 通胀调整营业收入 | `log_growth` |
| `emp` | 员工数 | `log_growth` |
| `inf_lt` | 通胀调整总负债 | `log_growth` |
| `inf_ch` | 通胀调整现金 | `log_growth` |
| `inf_ni` | 通胀调整净利润 | `signed_relative_change` |
| `inf_cogs` | 通胀调整营业成本 | `log_growth` |

`emp` 保留原始列，因为数据中没有 `inf_emp`。除 `inf_ni` 外，其余变量正值比例足够高，因此使用对数增长率。`inf_ni` 存在大量负值或非正值，不适合取对数，所以使用 signed relative change：

```text
(value_t - value_{t-1}) / (abs(value_{t-1}) + 1)
```

并使用 `WINSOR_LOWER = 0.01`、`WINSOR_UPPER = 0.99` 做轻度缩尾。

当前主结果配置为：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `BINS` | 5 | 每个变量增长率离散为 5 档 |
| `MAX_SOURCE_ORDER` | 2 | 只估计二元来源集合协同超边 |
| `ALPHA` | 0.5 | 条件转移概率平滑强度 |
| `MIN_SOURCE_COUNT` | 20 | 单个源状态最低样本支持 |
| `MIN_TOTAL_COUNT` | 100 | 完整转移样本最低数量 |
| `NULL_REPS` | 20 | 每类零分布置换重复次数 |
| `TOP_K` | 12 | 摘要图展示边数 |

## 从年度面板到离散转移样本

当前 PEID 估计不是直接在原始面板水平值上做的，而是先把公司年度面板转换为增长率，再离散化为状态，最后构造同一公司从 `t` 到 `t+1` 的跨期状态转移。

处理流程如下：

```text
原始 Compustat 年度面板
  -> 选择 gvkey, fyear 和实验变量
  -> 可选年份过滤
  -> 单变量缺失/非有限值过滤
  -> 根据正值比例选择 log_growth 或 signed_relative_change
  -> 只保留同一公司连续年份增长率
  -> 多变量增长率按 gvkey, mid_year 取交集
  -> 构造同一公司 t -> t+1 的增长率转移对
  -> 每个变量按 src+tgt 合并分布做五分位离散化
  -> 得到离散状态转移表
  -> 估计 pairwise EI 和二元 PEID synergy hyperedges
  -> 运行 target_shuffle 和 firm_time_shuffle 零分布
```

连续年度约束是关键。对每个公司，只有当 `fyear_t - fyear_{t-1} == 1` 时，增长率才被保留。增长率时间点记为 `mid_year`，例如 `mid_year = 2001` 表示从 2000 年到 2001 年的变化。

多变量样本采用内连接：只有某公司在某个 `mid_year` 上所有实验变量都具备有效增长率时，该公司-年份才进入完整增长率面板。因此最终样本不是每个变量各自最大样本，而是所有变量同时可用的交集。当前主结果审计表显示：

| 指标 | 当前值 |
| --- | ---: |
| 完整多变量增长率行数 | 286,049 |
| 完整跨期转移样本数 | 253,067 |
| 完整跨期转移公司数 | 24,394 |

离散化时，每个变量把来源端增长率和目标端增长率合并到一起生成同一组分位边界，再把 `_src` 和 `_tgt` 映射到状态 `1, 2, ..., 5`。这样可以保证同一变量在 `t` 端和 `t+1` 端使用同一标尺。当前完整离散转移表的每一行对应一条公司-年份转移样本，列结构为：

```text
gvkey, mid_year,
inf_at_src, inf_at_tgt,
inf_revt_src, inf_revt_tgt,
...
inf_cogs_src, inf_cogs_tgt
```

## PEID/EI 指标与零分布检验

成对 EI 边表示单个来源变量在时间 `t` 对单个目标变量在时间 `t+1` 的有效信息：

```text
w_{i -> j} = EI(X_t^i -> X_{t+1}^j)
```

估计时会按来源状态和目标状态计数，删除样本支持低于 `MIN_SOURCE_COUNT` 的来源状态，并对条件计数加上 `ALPHA` 平滑。有效信息可写成：

```text
EI = H(mean conditional target distribution)
     - mean H(conditional target distribution for each source state)
```

当前 7 个变量产生 `7 * 7 = 49` 条成对有向边。

PEID 协同超边表示多个来源变量联合起来对目标状态提供、且不能由单源投影解释的额外源侧协同。当前只枚举二元来源集合：

```text
{source_1, source_2}_t -> target_{t+1}
```

7 个变量对应 `C(7, 2) * 7 = 147` 条二元协同超边。当前实现不再用简单的“联合 EI 减去独立重估单源 EI 之和”来估计协同，因为那会把不同源集合下的支持过滤和平滑口径混在一起。当前使用源侧最大熵联合干预下的条件总相关口径：

```text
synergy_raw
= E_{target state} [
    TC(source_1, source_2 | target state)
  ]
```

该口径对应 PEID/EID 理论中的源侧协同非负性，因此 `synergy_raw` 本身应为非负。输出中保留 `synergy = synergy_raw`，主要用于兼容既有绘图和结果读取代码。

零分布检验使用两类置换：

1. `target_shuffle`：在全样本范围内打乱目标变量的下一期状态，破坏源-目标动态对应。
2. `firm_time_shuffle`：在每个公司内部打乱目标状态，保留部分公司内部状态分布但破坏时间顺序。

经验 p 值计算为：

```text
p_value = (1 + count(null_value >= observed_value)) / (n_null + 1)
```

当前两类 null 各重复 20 次，所以每条边有 40 个 null 样本，最小可见经验 p 值为 `1 / 41 = 0.02439`。这意味着当前 p 值适合排序和初步筛选，不应作为最终显著性证据。正式统计证据应把 `NULL_REPS` 提高到 100 或更多。

## 主实验结果

当前结果显示，公司变量的跨期信息结构主要集中在两个层面。

第一类是成对 EI 边。Top 成对边集中在：

- 自身状态延续：`inf_ni -> inf_ni`、`inf_revt -> inf_revt`、`inf_at -> inf_at`、`inf_cogs -> inf_cogs`、`emp -> emp`。
- 企业规模和经营规模链条：`inf_at -> inf_revt`、`emp -> inf_revt`、`inf_lt -> inf_revt`。
- 收入成本联动：`inf_at -> inf_cogs`、`emp -> inf_cogs`、`inf_revt -> inf_cogs`。
- 规模到组织扩张：`inf_at -> emp`。

第二类是 PEID 协同超边。Top 协同主要集中在：

- 资产负债组合：`{inf_at, inf_lt}` 对 `inf_at`、`inf_lt`、`inf_ni`、`inf_revt`、`emp` 都有较高协同。
- 收入成本组合：`{inf_revt, inf_cogs}` 对 `inf_cogs`、`inf_revt`、`inf_at` 有较高协同。
- 资产或经营变量与利润组合：`{inf_at, inf_ni}`、`{inf_revt, inf_ni}`、`{emp, inf_ni}` 对 `inf_ni` 的协同较强。

一个稳妥的论文式概括是：

> 在离散增长状态的公司年度面板中，跨期信息主要沿着“盈利状态惯性”“资产/员工/负债到收入与成本的经营规模链条”传播；多源协同则集中在资产负债表结构与损益表结构的组合上，尤其是资产-负债组合、收入-成本组合以及利润与规模变量的联合状态。

Top-K 图只是摘要视图，不代表原生结果被截断。正式解释应回到全量边表、全量协同表和完整 heatmap，而不是只解释图中显示的前 12 条边。

## 资产-负债协同的机制解释

当前最突出的二元协同超边是：

```text
{inf_at, inf_lt}_t => inf_at_{t+1}
```

其中 `inf_at` 是通胀调整总资产增长状态，`inf_lt` 是通胀调整总负债增长状态，目标 `inf_at` 是下一期通胀调整总资产增长状态。对应数值为：

| 指标 | 数值 |
| --- | ---: |
| `joint_ei` | 0.148899 |
| `single_ei_sum` | 0.120638 |
| `synergy_raw` | 0.028261 |
| `null_mean` | 0.006537 |
| `z_score` | 3.487640 |
| `p_value` | 0.02439 |

这说明，当前总资产增长状态与当前总负债增长状态的联合来源，比单独观察两者所能解释的部分多出约 `0.028261` bit 的源侧协同信息。稳妥表述是：资产增长与负债增长的联合状态，对下一期资产增长状态具有显著的 PEID 互补信息。

经济含义上，总资产给出企业规模和资产扩张惯性，总负债给出这种规模变化背后的融资结构与杠杆约束。单看总资产增长，会把债务融资扩张、权益融资扩张、资产重估和资产结构调整等不同机制混在一起；单看总负债增长，也无法判断新增负债是否真正转化为资产扩张，还是反映融资压力、短期债务滚动或亏损补洞。二者合在一起后，才更清楚地区分企业处于哪一种资产负债表状态：

- `inf_at` 高、`inf_lt` 高：更像债务融资支撑的资产扩张，下一期资产继续高增长的概率更高。
- `inf_at` 高、`inf_lt` 低：可能来自权益融资、留存收益、资产重估或资产结构调整，其持续性不同。
- `inf_at` 低、`inf_lt` 高：可能反映融资压力、负债滚动或亏损补洞，未必对应健康扩张。
- `inf_at` 低、`inf_lt` 低：更接近收缩、去杠杆或低增长状态。

资产负债表恒等式强化了这一解释：

```text
资产 = 负债 + 所有者权益
```

总资产与总负债天然不独立。它们的联合变化隐含杠杆率、融资结构、扩张质量和资产负债匹配等单变量看不到的信息。PEID 的 `synergy_raw` 正是在记录这类只有联合状态才显现的目标相关条件耦合。

离散状态也支持这一解释。当前 `inf_at` 第 5 档对应 log growth `> 0.176`，`inf_lt` 第 5 档对应 log growth `> 0.238`；第 1 档分别是 `inf_at <= -0.100617` 与 `inf_lt <= -0.130919`。在离散转移表中，未来 `inf_at_tgt = 5` 时，当前 `{inf_at_src = 5, inf_lt_src = 5}` 是占比最高的联合来源状态；未来 `inf_at_tgt = 1` 时，当前 `{inf_at_src = 1, inf_lt_src = 1}` 是占比最高的联合来源状态。也就是说，未来总资产增长不只依赖某个单变量高低，而明显依赖资产增长与负债增长的联合格局。

这条超边不能写成“总资产和总负债共同造成未来总资产增长”。它可能同时包含真实融资机制、会计结构约束、规模效应和分位离散化带来的机械共动。更好的写法是：当前总资产增长与总负债增长的联合状态，对下一期总资产增长状态具有显著互补预测信息或 PEID 有效信息。

## 鲁棒性检验

鲁棒性扫描来自：

```text
results/company_ce/csv/peid_sensitivity_main/
```

该扫描使用同一组主实验变量：

```text
inf_at, inf_revt, emp, inf_lt, inf_ch, inf_ni, inf_cogs
```

本轮鲁棒性扫描的目的不是重新做正式 p 值检验，而是检查结果结构是否随参数扰动改变。因此扫描中 `null_reps=0`，主要应看不同参数下重新估计出的因果结构是否仍然相似，而不是先读统计摘要表。当前最清楚的呈现方式是两张热图矩阵：

```text
fig/company_ce/peid_sensitivity_main/peid_robustness_pairwise_heatmap_grid.png
fig/company_ce/peid_sensitivity_main/peid_robustness_synergy_heatmap_grid.png
```

第一张图把每个参数设置下的完整 7x7 pairwise EI 矩阵画出来。纵轴是来源变量，横轴是下一期目标变量，所有 panel 使用统一色标。直接观察可以看到，最深的格子始终集中在 `inf_ni -> inf_ni`、`inf_revt -> inf_revt`、`inf_at -> inf_at`、`inf_cogs -> inf_cogs`、`emp -> emp` 这些自身延续边，以及 `inf_at`、`emp`、`inf_lt` 到收入/成本状态的经营规模链条。换言之，改变分箱、平滑、支持阈值、缩尾或年份窗口后，pairwise 因果热图的亮区位置基本不变。

第二张图把各参数设置下的 Top 协同来源集合按行对齐、目标变量按列展开。它显示协同亮区仍然集中在资产负债表组合和损益表组合上，尤其是 `{inf_at, inf_lt}`、`{inf_revt, inf_cogs}` 以及利润变量与规模变量的组合。不同参数会改变颜色深浅，也会让少量边在 Top 集合边界附近进出，但主结构族没有发生实质变化。

本轮所有扫描中，pairwise EI 和 `synergy_raw` 都保持非负，且 `synergy == synergy_raw`。这和当前代码采用的 EI 与源侧条件总相关口径一致。

离散档数是最需要认真解释的扰动。`BINS=3` 时 EI 和 synergy 数值整体变小，因为状态空间更粗，目标状态可区分性降低；`BINS=6` 时数值整体略变大，因为状态空间更细，条件分布差异更容易保留下来。这不是结论冲突，而是离散化分辨率改变后的自然结果。解释时应强调边族和相对排序稳定，避免把某个单一分箱下的 bit 数值当作不可变结构常数。

样本支持阈值、平滑强度和 winsor 缩尾几乎不影响结果。`MIN_SOURCE_COUNT=10,20,50` 完全一致，说明当前 253,067 条完整转移样本足以支撑 5 个单源状态和 25 个二源联合状态；`ALPHA=0.1,0.5,1.0` 几乎完全一致，说明排序不是 Laplace/Dirichlet 平滑参数造成的伪影；winsor 扰动没有改变结果，说明当前主结论不是由净利润极端增长值驱动。

年份窗口检验显示结构稳定但时期强度有差异。早期和晚期 panel 中，主要亮区仍然落在同一批变量组合上，说明早晚时期没有推翻主结构；早期样本中少数成对边的深浅和局部排序更容易变化，晚期样本更接近 full baseline。相较之下，协同热图中的资产负债表组合与损益表组合更稳定，更适合作为“资产负债表结构与损益表结构联合承载下一期状态信息”的证据。

## 相关文献与解释边界

Hoel, Albantakis, and Tononi 的 causal emergence 工作把 effective information 用作描述系统状态转移机制信息的量，支持用 EI 表述“源状态对下一期目标状态的机制信息”，但也提醒我们这不是标准计量经济学意义的干预估计。可引用：

- Hoel, E. P., Albantakis, L., and Tononi, G. (2013). *Quantifying causal emergence shows that macro can beat micro*. PNAS. DOI: https://doi.org/10.1073/pnas.1314922110
- Hoel, E. P. (2017). *When the Map Is Better Than the Territory*. Entropy. https://www.mdpi.com/1099-4300/19/5/188

Williams and Beer 的 partial information decomposition 提供了把多源信息分解为冗余、特有和协同成分的基础思想。虽然当前实验用的是 PEID/EID 的源侧分解而不是完整 PID，但“多源协同信息”这一解释方向与 PID 文献一致。可引用：

- Williams, P. L., and Beer, R. D. (2010). *Nonnegative Decomposition of Multivariate Information*. arXiv: https://arxiv.org/abs/1004.2515

本地 Zotero 中还有与当前项目最直接相关的 PEID 文献：

- Yang, M., Wang, S., and Zhang, J. (2026). *Partial Effective Information Decomposition for Synergistic Causality*. Zotero item key: `MYATYWAJ`.

这篇最适合用来支撑“源侧协同非负性”和“协同不是简单联合 EI 减去独立单源 EI”的方法定义。由于它目前在本地 Zotero 中是 preprint 条目，正式写作前需要确认是否已有公开版本、DOI 或 arXiv 链接。

企业增长和公司金融文献可支撑当前变量组的经济解释。Penrose 的企业增长理论强调企业增长受内部资源、管理能力和生产机会约束；Teece, Pisano, and Shuen 的 dynamic capabilities 文献强调企业资源、能力和组织过程共同塑造未来绩效；Coad 和 Rao 以及 Coad 关于企业增长的研究支持把销售、就业、生产率等企业变量看作动态共同演化的面板过程。可引用：

- Penrose, E. (1959/1995). *The Theory of the Growth of the Firm*. Oxford University Press. https://academic.oup.com/book/25306
- Teece, D. J., Pisano, G., and Shuen, A. (1997). *Dynamic Capabilities and Strategic Management*. Strategic Management Journal. DOI: https://doi.org/10.1002/(SICI)1097-0266(199708)18:7%3C509::AID-SMJ882%3E3.0.CO;2-Z
- Coad, A., and Rao, R. (2008). *Innovation and firm growth in high-tech sectors: A quantile regression approach*. Research Policy. https://www.sciencedirect.com/science/article/pii/S0048733308000152
- Coad, A. (2010). *Exploring the processes of firm growth: evidence from a vector auto-regression*. Industrial and Corporate Change. https://academic.oup.com/icc/article-pdf/19/6/1677/2156142/dtq018.pdf

财务变量之间的经济含义还可由公司金融文献补充。Cooper, Gulen, and Schill 的资产增长研究支持把 `inf_at` 视作具有跨期预测含义的状态变量；Fazzari, Hubbard, and Petersen 关于融资约束与企业投资的研究说明现金流、融资条件、投资和资产扩张之间存在系统关系。可引用：

- Cooper, M. J., Gulen, H., and Schill, M. J. (2008). *Asset Growth and the Cross-Section of Stock Returns*. Journal of Finance. https://ideas.repec.org/a/bla/jfinan/v63y2008i4p1609-1651.html
- Fazzari, S. M., Hubbard, R. G., and Petersen, B. C. (1988). *Financing Constraints and Corporate Investment*. Brookings Papers on Economic Activity. https://www.brookings.edu/articles/financing-constraints-and-corporate-investment/

这些文献支持的是“当前变量组的经济解释合理性”，不是直接证明每条 PEID 边都是因果关系。更准确的写法是：既有公司金融和企业增长文献说明这些变量共同刻画企业经营规模、融资状态、投资能力和盈利状态；当前实验进一步用离散转移机制的 EI/PEID 指标，给出这些变量在跨期状态转移中的候选机制信息结构。

## 当前结论与后续工作

当前最稳妥的结论是：公司年度增长状态中存在由资产负债表结构和损益表结构共同承载的跨期机制信息。单变量边反映经营规模与盈利状态的状态延续，多变量超边反映资产、负债、收入、成本和利润之间的联合约束。鲁棒性检验显示，这些结构不是由平滑参数、低支持状态、净利润极端值或单一时期样本驱动的；离散档数会改变 EI 和 synergy 的绝对 bit 数值，但不会改变主要结构族。

当前结果的解释边界也必须保留：

- 不要说这些边已经证明了严格因果干预效应。
- 不要把 `p_value` 当作最终显著性证据，因为主实验 `NULL_REPS = 20`，鲁棒性扫描 `null_reps=0`。
- 不要过度解释 `BINS = 5` 下某个 bit 数值的精确大小。
- 不要只讲 Top-K 图；正式解释应同时引用全量边表和鲁棒性表。

后续工作可以沿三条路线推进。

第一，提高零分布重复次数，至少把正式主实验的 `NULL_REPS` 提高到 100 或更多，使 p 值更适合写入论文。

第二，加入条件化设计，按行业、年份、规模、国家或其他可用环境变量构造条件转移机制，检查条件内 EI 与条件聚合 EI 的差异，避免把行业结构、时间冲击和规模差异误当成系统内在动力学。

第三，推进宏观状态比较。当前单尺度 PEID 结果已经提示，资产负债表联合状态和损益表联合状态承载了重要协同信息。因此后续构造宏观财务 regime 时，不应只按单个变量高低划分，而应显式吸收资产-负债、收入-成本、利润-规模变量等联合结构。只有在进一步证明某种宏观财务状态具有更高 EI、更清晰转移结构和稳健预测表现后，才适合讨论企业财务系统中的 effective-information emergence 或 conditional causal emergence。
