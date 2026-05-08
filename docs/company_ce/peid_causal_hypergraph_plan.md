# 单尺度企业变量 PEID 因果超图分析说明

## 研究目标

本分析固定在企业财务变量这一单一尺度上，不讨论因果涌现、不做粗粒化搜索，也不使用机器学习。目标是在离散变量的企业面板动力学中估计变量之间的机制性信息流，并将结果呈现为两类结构：

1. **Pairwise EI 因果图**：单个源变量在时间 `t` 对单个目标变量在时间 `t+1` 的有效信息。
2. **PEID 协同超图**：多个源变量联合起来对目标变量产生、且不能由单源 EI 简单解释的额外信息。

这里的“因果”采用项目现有 EI/PEID 框架中的最大熵干预口径：它表示在估计的离散转移机制下，源变量状态对下一期目标状态的机制性信息强度，不等同于观察相关，也不声称已经识别出严格实验干预因果。

## 输入数据与变量

默认输入为：

```text
data/inf_compustat_anual_US_filter_feas.csv
```

默认第一版变量为：

```text
at, revt, emp, dltt, lt, ch
```

这些变量覆盖资产、收入、员工、长期债务、总负债和现金。第一版建议控制在 6-8 个变量内，因为多变量联合状态空间会随变量数和离散档数快速膨胀。高维变量集可以作为后续扩展。

## 离散动力学构造

脚本首先为每个变量构造连续年份增长状态：

- 若变量绝大多数为正值，则使用 `log(value_t) - log(value_{t-1})`。
- 若变量包含较多非正值，则使用 `(value_t - value_{t-1}) / (abs(value_{t-1}) + 1)`，并做 winsorize 截尾。

随后只保留企业连续年份样本，构造：

```text
X_t     = 企业在 mid_year 的离散变量状态
X_{t+1} = 同一企业下一年 mid_year 的离散变量状态
```

离散化默认使用 3 档分位数。5 档可用于稳健性检查。10 档只建议用于单变量 QTPM 基线，不建议直接用于多变量 PEID 主分析。

## 指标定义

对源变量 `i` 和目标变量 `j`，pairwise EI 边权为：

```text
w_{i -> j} = EI(X_t^i -> X_{t+1}^j)
```

对源变量组 `A` 和目标变量 `j`，协同超边权为：

```text
Syn(A => j) = EI(X_t^A -> X_{t+1}^j) - sum_{i in A} EI(X_t^i -> X_{t+1}^j)
```

经验数据中 `synergy_raw` 可能为负，这通常表示单源项之和超过联合项，可能来自冗余、有限样本或支持集估计口径差异。因此输出表同时保留：

- `synergy_raw`：原始差值。
- `synergy`：用于画协同超图的正部，即 `max(0, synergy_raw)`。

## Null 检验

脚本提供两类 null：

1. `target_shuffle`：打乱目标变量的下一期状态，破坏源-目标动态对应。
2. `firm_time_shuffle`：在企业内部打乱目标状态，保留部分企业内分布但破坏时间顺序。

输出中包含 `null_mean`、`null_std`、`z_score` 和经验 `p_value`。当 `null_reps` 很小时，`p_value` 只能用于 smoke test，不应用作正式统计结论。正式图建议至少使用 100 次以上 null。

## 输出文件

核心分析脚本：

```text
scripts/company_ce/peid_causal_hypergraph.py
```

可调参 notebook：

```text
exp/company_ce/peid_causal_hypergraph.ipynb
```

默认结果表目录：

```text
results/company_ce/csv/peid/
```

主要表格包括：

- `peid_variable_audit.csv`：变量覆盖、变换方式和样本量。
- `peid_discretization_edges.csv`：每个变量的离散边界。
- `peid_discrete_transition_states.csv`：离散后的企业-year 转移样本。
- `peid_pairwise_edges.csv`：pairwise EI 有向边。
- `peid_synergy_hyperedges.csv`：PEID 协同超边。
- `peid_pairwise_null_samples.csv`：pairwise EI 的 null 样本。
- `peid_synergy_null_samples.csv`：synergy 的 null 样本。
- `peid_period_stability.csv`：早晚时期 pairwise EI 稳定性。
- `peid_top_edges_for_figures.csv`：图中使用的 top-k 边和超边。

默认结果图目录：

```text
fig/company_ce/peid/
```

主要图包括：

- `peid_pairwise_ei_heatmap.png`：变量之间 pairwise EI 热力图。
- `peid_top_pairwise_graph.png`：top-k 普通有向因果图。
- `peid_top_synergy_hypergraph.png`：top-k 协同超图。
- `peid_null_comparison.png`：真实值与 null 均值对比。
- `peid_period_stability_heatmap.png`：早晚时期 pairwise EI 稳定性。

## Notebook 参数

notebook 顶部参数区集中暴露：

- `VARIABLES`：变量列表。
- `BINS`：离散档数。
- `MAX_SOURCE_ORDER`：协同源变量阶数，第一版建议 `2`。
- `ALPHA`：Dirichlet/Laplace 平滑强度。
- `MIN_SOURCE_COUNT`：每个源状态支持的最小样本数。
- `NULL_REPS`：null 重复次数。
- `TOP_K`：图中展示的边数。
- `YEAR_START`, `YEAR_END`：可选年份过滤。
- `RUN_ANALYSIS`：是否重新计算。若设为 `False`，notebook 只读取缓存并重画图。

## 推荐复现实验

1. smoke test：`VARIABLES = ["at", "revt", "emp"]`，`BINS = 3`，`NULL_REPS = 2`。
2. 第一版主结果：`VARIABLES = ["at", "revt", "emp", "dltt", "lt", "ch"]`，`BINS = 3`，`NULL_REPS >= 20`。
3. 稳健性：把 `BINS` 改为 5；或按 `YEAR_START/YEAR_END` 拆分时间窗口。
4. 扩展实验：加入 `ni`, `xopr`, `cogs`, `teq` 等变量，但需要同步提高样本阈值并检查状态支持。

## 结论边界

- 该分析适合回答“哪些企业财务变量在离散动力学中对其他变量有机制性信息流”和“哪些变量组合具有协同影响”。
- 不适合直接回答“宏观状态是否优于微观变量”，因为本方案刻意固定在单一尺度。
- 不适合直接声称严格干预因果；若要靠近因果识别，需要进一步加入准实验、政策冲击或更强的条件化设计。
