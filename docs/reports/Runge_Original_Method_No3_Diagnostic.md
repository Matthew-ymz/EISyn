# Runge 原文方法 No.3 排名异常诊断

## 结论

最初地球图里 No.3 在 Runge 原文方法面板中排名偏高，不是 1948-2026 新数据造成的。旧的 1948-2011 复现结果里 No.3 已经很高：

| 数据 | No.3 ACE rank | No.3 ACS rank | ACE | ACS |
|---|---:|---:|---:|---:|
| 本地旧复现 1948-2011 | 5 | 2 | 0.048088 | 0.045281 |
| 本地新复现 1948-2026 | 5 | 3 | 0.045813 | 0.039972 |

因此，No.3 偏高主要来自本地 Runge 2015 复现口径，而不是新数据集本身。

进一步检查后，已经找到一个明确实现差异：此前脚本把 `run_pcmci` 的最终 MCI `p_matrix` 当作 parent set，再进入稀疏线性回归；但原文 Supplementary Table 2/3 对应的是 PC step parent set。改成 `run_pc_stable` 的 parents 后，在 1948-2026 新数据上 No.3 明显下降：

| 口径 | No.3 ACE rank | No.3 ACS rank | ACE | ACS |
|---|---:|---:|---:|---:|
| 修正前：MCI p-matrix threshold | 5 | 3 | 0.045813 | 0.039972 |
| 修正后：PC-stable parents | 12 | 13 | 0.032422 | 0.027424 |

所以当前最主要问题不是“不能按原文算法做”，而是本地实现之前没有严格使用原文的 parent-selection 对象。这个差异已在 `scripts/reproduce_runge2015_gateways.py` 中修正，并另存了修正版结果 `results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways_pcstable_corrected/`。

## 已确认的不一致

### 1. 旧复现年份不是原文年份

原文写的是 1948-2012，并有 3,339 个 weekly samples。当前仓库旧复现 manifest 实际记录为：

```text
start_year = 1948
end_year = 2011
```

本地 `data/ncep_reanalysis_slp/daily/slp.2012.nc` 是存在的，所以旧复现少用了 2012 年。这是一个确定的不一致。

### 2. No.3 的 parent / time-graph links 与原文补充表差很多

原文补充材料里 No.3 的 parents 是：

```text
P3 = (3t-1, 37t-1, 24t-1, 4t-4, 15t-2, 34t-1)
```

而修正前本地旧复现的 No.3 入边集合是 25 条，只和原文 P3 重合 3 条：

```text
overlap: (3,t-1), (34,t-1), (37,t-1)
missing from local: (4,t-4), (15,t-2), (24,t-1)
extra local links: 22 条
```

修正前本地新复现 1948-2026 的 No.3 入边集合是 22 条，和原文 P3 重合 4 条：

```text
overlap: (3,t-1), (4,t-4), (24,t-1), (34,t-1)
missing from local: (15,t-2), (37,t-1)
extra local links: 18 条
```

这说明 No.3 偏高的直接来源是本地 causal reconstruction / threshold 后的 No.3 连接比原文密得多，而不是绘图或 ACE/ACS 汇总公式。修正为 PC-stable parents 后，新数据 No.3 不再异常靠前，但 parent set 仍未和原文补充表完全一致。

### 3. 完整 component 编号对齐并没有官方表

仓库已有说明也指出：本地 orthomax 排序只对少数论文讨论的模态做了视觉校准，当前映射只包含：

```text
7 <-> 18
8 <-> 26
21 <-> 48
```

也就是说，No.3 是否真对应原文 Fig. 2 / Fig. 4 的 No.3 尚未被完整校准。原文只给图，不给 machine-readable 的 60-component loading table，因此完整编号对齐本身是不确定的。

## 当前最可能的根因

修正前 No.3 排名偏高主要由两类不一致叠加造成：

1. **causal graph 不一致**：本地实现误用了最终 MCI `p_matrix`，而不是 PC-stable parent set，导致 time-series graph 与原文补充表不匹配，No.3 多了大量入边/出边，抬高了 ACS/ACE。
2. **component label 不完全对齐**：本地只校准了少数 paper-discussed components，No.3 可能不是原文图中的同一个空间模态。

年份少用 2012 年是确定问题，但从旧复现和新复现 No.3 都高来看，它大概率不是唯一原因。

## 下一步

要进一步逼近原文 No.3，应按以下顺序做：

1. 先重跑 1948-2012，不是 1948-2011，确认年份修正后的 No.3 是否仍高。
2. 保存 PC-stable parent candidates，不只保存 threshold 后 `causal_edges.csv`，逐节点对比 Supplementary Table 2/3。
3. 对 No.3 的空间 loading 与原文 Supplementary Fig. 1/2 做视觉/相关性校准，确认编号是否错配。
4. 如果要画“原文一致”版本，应优先使用与补充表一致的 parent set / time graph，而不是修正前的 No.3 dense graph。
