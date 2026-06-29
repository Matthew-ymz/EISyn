# Runge SLP 1948-2026 Scheme A 结果

本文合并原来的多步 EI 路径设计说明和 1948-2026 SLP 结果展示。当前主结果采用方案 A：对每一条非自环边估计完整二阶背景源集合，用平均条件 EI 替换直接边权，再在 signed dense 图上计算有限长度 path score。本文不使用任何额外预定义模态加权后处理。

## 数据与处理口径

原始数据为 NCEP/NCAR daily sea-level pressure，文件范围从 `data/ncep_reanalysis_slp/daily/slp.1948.nc` 到 `slp.2026.nc`。预处理沿用 Runge 复现流程：

- 删除所有 2 月 29 日，把一年固定为 365 个 calendar day；
- 对每个格点按 calendar day 计算多年均值和标准差，得到标准化日异常；
- 对每个格点沿时间轴去除线性趋势；
- 使用纬度面积权重；
- 在月均场上拟合 60 个 PCA/Varimax component；
- 将 Varimax loading 投影回 daily SLP 异常场，再按连续 7 天窗口聚合为 weekly component scores。

本轮组件阶段的有效范围是 daily scores 从 `1948-01-01` 到 `2026-03-17`，weekly scores 到 `2026-03-11`。结果包含 `28546` 个 daily samples、`4078` 个 weekly samples、60 个 component。MLP-TM-EI 阶段用过去 4 周状态预测下一周 component state，transport-map EI 的干预样本数为 `4096`。

## 方案 A：完整二阶平均条件 EI

令 $n=60$。单源 pairwise MLP-TM-EI 记为

$$
E^{(0)}_{ij}=EI(X_i\to X_j),
\qquad i,j\in\{1,\ldots,n\}.
$$

考虑二阶协同时，对任意源 $i$、目标 $j$ 和背景源 $r\ne i,j$，定义条件 EI 增量：

$$
E_{i\to j\mid r}
=
EI(X_{\{i,r\}}\to X_j)-EI(X_r\to X_j).
$$

当前结果对每条非自环边使用完整背景源集合

$$
\mathcal{R}_{ij}=\{r:r\ne i,\ r\ne j\},
\qquad |\mathcal{R}_{ij}|=58,
$$

并取平均：

$$
\bar E^{(2)}_{ij}
=
\frac{1}{58}
\sum_{r\in\mathcal{R}_{ij}}
\left[
EI(X_{\{i,r\}}\to X_j)-EI(X_r\to X_j)
\right],
\qquad i\ne j.
$$

方案 A 的直接边矩阵为

$$
\mathbf{A}^{(2)}=\bar{\mathbf{E}}^{(2)},
\qquad
A^{(2)}_{ii}=0.
$$

这里不做非负性截断，不做 top-k 稀疏化，也不引入谱缩放参数。路径矩阵使用有限长度 walk-sum：

$$
\mathbf{T}^{(2)}_L
=
\sum_{\ell=1}^{L}\left(\mathbf{A}^{(2)}\right)^\ell,
\qquad L=60.
$$

三个节点分数为

$$
\mathrm{ACE}^{(2)}(i)
=
\frac{1}{n-1}\sum_{j\ne i}T^{(2)}_{L,ij},
$$

$$
\mathrm{ACS}^{(2)}(i)
=
\frac{1}{n-1}\sum_{j\ne i}T^{(2)}_{L,ji},
$$

$$
\mathrm{AMCE}^{(2)}(m)
=
\frac{1}{(n-1)(n-2)}
\sum_{\substack{s\ne m,\ t\ne m\\s\ne t}}
A^{(2)}_{sm}T^{(2)}_{L,mt}.
$$

ACE 衡量 component 作为外向影响源的强度，ACS 衡量 component 作为汇入目标的强度，AMCE 衡量 component 位于 source-to-target 中介位置的强度。这三个分数都是 signed dense 条件 EI 图上的 graph-walk path score，不是严格的信息论多步 EI。

## 对照：不考虑二阶协同

对照组使用同一套 pairwise MLP-TM-EI 矩阵 $\mathbf{E}^{(0)}$ 作为直接边，在相同的 signed dense 有限长度 walk-sum 上计算 ACE、ACS 和 AMCE。该对照不使用二阶 joint EI，因此表示“没有考虑二阶协同”的计算结果。为了让对照只反映二阶项差异，对照组同样不做 top-k 稀疏化、不做谱缩放。

## 主结果图

![Pairwise dense vs full Scheme A](../../results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2/pairwise_dense_vs_scheme_a_full_order2_map.png)

*图 1. 1948-2026 daily SLP 数据上的无二阶协同 pairwise dense path 与完整二阶 Scheme A 对比。a-b 为 pairwise dense 对照；c-d 为完整 Scheme A。a、c 展示 ACE/ACS：外圈颜色为 ACE，内圈颜色为 ACS。b、d 展示 AMCE。所有 60 个 component 都被绘制；标注节点为各自方法中 ACE、ACS 和 AMCE top 节点并集。*

## 完整性检查

本轮完整估计覆盖所有二阶背景项：

- 二阶 joint EI 记录：`102660`
- 期望二阶 joint EI 记录：`102660`
- 条件 EI 非自环边：`3540`
- 期望条件 EI 非自环边：`3540`
- 每条非自环边的背景源数：`58`
- 未覆盖条件边：`0`

完整 Scheme A 的直接边统计：

- 正条件边：`3540`
- 负条件边：`0`
- $\rho(\mathbf{A}^{(2)})=0.3387$
- 最大绝对直接边权：`0.1118`

无二阶协同 pairwise dense 对照的直接边统计：

- 正边：`3540`
- 负边：`0`
- $\rho(\mathbf{E}^{(0)})=0.3057$
- 最大绝对直接边权：`0.1108`

## 排名读数

完整 Scheme A 的 top 节点如下：

| Rank | ACE | ACS | AMCE |
|---:|---|---|---|
| 1 | component_01 (0.02997) | component_03 (0.01106) | component_01 (1.458e-04) |
| 2 | component_02 (0.01892) | component_38 (0.01099) | component_02 (1.214e-04) |
| 3 | component_17 (0.01369) | component_57 (0.01096) | component_03 (8.597e-05) |
| 4 | component_04 (0.01297) | component_36 (0.01093) | component_17 (8.407e-05) |
| 5 | component_27 (0.01254) | component_29 (0.01065) | component_27 (8.392e-05) |
| 6 | component_16 (0.01201) | component_41 (0.01059) | component_36 (7.645e-05) |
| 7 | component_10 (0.01201) | component_40 (0.01049) | component_16 (7.194e-05) |
| 8 | component_03 (0.01168) | component_26 (0.01040) | component_04 (6.897e-05) |
| 9 | component_05 (0.01160) | component_59 (0.01024) | component_10 (6.057e-05) |
| 10 | component_19 (0.01146) | component_55 (0.01021) | component_31 (5.716e-05) |

无二阶协同 pairwise dense 对照的 top 节点如下：

| Rank | ACE | ACS | AMCE |
|---:|---|---|---|
| 1 | component_01 (0.02766) | component_03 (0.009707) | component_01 (1.194e-04) |
| 2 | component_02 (0.01720) | component_38 (0.009668) | component_02 (1.011e-04) |
| 3 | component_17 (0.01223) | component_57 (0.009628) | component_03 (6.984e-05) |
| 4 | component_04 (0.01157) | component_36 (0.009590) | component_17 (6.833e-05) |
| 5 | component_27 (0.01113) | component_29 (0.009339) | component_27 (6.820e-05) |
| 6 | component_10 (0.01063) | component_41 (0.009270) | component_36 (6.136e-05) |
| 7 | component_16 (0.01059) | component_40 (0.009159) | component_16 (5.745e-05) |
| 8 | component_03 (0.01031) | component_26 (0.009097) | component_04 (5.543e-05) |
| 9 | component_05 (0.01026) | component_59 (0.008991) | component_10 (4.785e-05) |
| 10 | component_19 (0.01009) | component_55 (0.008911) | component_40 (4.487e-05) |

完整 Scheme A 与 pairwise dense 对照的空间排序基本一致，但二阶平均条件 EI 系统性抬高了路径分数尺度：ACE/ACS 色标上限从 `0.02766` 增至 `0.02997`，AMCE 绝对色标上限从 `1.194e-04` 增至 `1.458e-04`。AMCE top-10 中，完整 Scheme A 将 component_31 纳入前十，而 pairwise dense 对照的第十位是 component_40。

## 敏感性：$L=1$

把 path length 从 $L=60$ 改为 $L=1$ 时，$\mathbf{T}_L$ 退化为直接边矩阵本身：

$$
\mathbf{T}_1=\mathbf{A}.
$$

因此该结果只检验直接边层面的二阶平均条件 EI 差异，不包含更长 walk 的累积传播。

![Pairwise dense vs full Scheme A, L=1](../../results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2_l01/pairwise_dense_vs_scheme_a_full_order2_map.png)

*图 2. $L=1$ 时的无二阶协同 pairwise dense path 与完整二阶 Scheme A 对比。二阶 joint EI 覆盖、条件边数量和每边背景源数与图 1 相同，仅 path length 改为 1。*

$L=1$ 的完整性检查仍为：

- 二阶 joint EI 记录：`102660 / 102660`
- 条件 EI 非自环边：`3540 / 3540`
- 每条非自环边的背景源数：`58`
- 未覆盖条件边：`0`

$L=1$ 时完整 Scheme A 的 top 节点如下：

| Rank | ACE | ACS | AMCE |
|---:|---|---|---|
| 1 | component_01 (0.02010) | component_38 (0.007412) | component_01 (9.779e-05) |
| 2 | component_02 (0.01281) | component_03 (0.007358) | component_02 (8.233e-05) |
| 3 | component_04 (0.008976) | component_36 (0.007354) | component_03 (5.730e-05) |
| 4 | component_17 (0.008630) | component_57 (0.007323) | component_27 (5.487e-05) |
| 5 | component_27 (0.008205) | component_41 (0.007123) | component_17 (5.300e-05) |
| 6 | component_10 (0.008047) | component_29 (0.007110) | component_36 (4.794e-05) |
| 7 | component_16 (0.007923) | component_40 (0.007038) | component_04 (4.777e-05) |
| 8 | component_03 (0.007787) | component_26 (0.006932) | component_16 (4.743e-05) |
| 9 | component_05 (0.007708) | component_55 (0.006849) | component_10 (4.064e-05) |
| 10 | component_19 (0.007653) | component_59 (0.006827) | component_40 (3.850e-05) |

$L=1$ 的 pairwise dense 对照 top 节点如下：

| Rank | ACE | ACS | AMCE |
|---:|---|---|---|
| 1 | component_01 (0.01945) | component_38 (0.006848) | component_01 (8.400e-05) |
| 2 | component_02 (0.01222) | component_03 (0.006777) | component_02 (7.196e-05) |
| 3 | component_04 (0.008414) | component_36 (0.006776) | component_03 (4.885e-05) |
| 4 | component_17 (0.008063) | component_57 (0.006751) | component_27 (4.672e-05) |
| 5 | component_27 (0.007633) | component_41 (0.006546) | component_17 (4.505e-05) |
| 6 | component_10 (0.007479) | component_29 (0.006542) | component_04 (4.035e-05) |
| 7 | component_16 (0.007333) | component_40 (0.006455) | component_36 (4.019e-05) |
| 8 | component_03 (0.007208) | component_26 (0.006364) | component_16 (3.973e-05) |
| 9 | component_05 (0.007156) | component_59 (0.006290) | component_10 (3.369e-05) |
| 10 | component_19 (0.007073) | component_55 (0.006274) | component_40 (3.191e-05) |

$L=1$ 的结论与 $L=60$ 一致：完整二阶 Scheme A 相比 pairwise dense 对照主要抬高分数尺度，而不是改变主导空间排序。ACE/ACS 色标上限从 `0.01945` 增至 `0.02010`，AMCE 绝对色标上限从 `8.400e-05` 增至 `9.779e-05`。

## 原始 Hyper 指标复算

按照原始 hyper 指标，节点分数不再用 Scheme A 条件边替换 pairwise 直接边，而是在一阶 EI 指标上加二阶源成员贡献。对二阶源集合 $K=\{a,b\}$ 和目标 $j$，本轮使用旧 hypergraph 代码的二阶 Möbius 项：

$$
\Delta_2(K\to j)
=
EI(X_K\to X_j)-\sum_{s\in K}EI(X_s\to X_j),
$$

并按截图中的绝对值项进入节点分数：

$$
Syn^{EID}_{K\to j}=|\Delta_2(K\to j)|.
$$

源侧和目标侧分数为

$$
Hyper\text{-}ACE(i)
=
\frac{1}{n-1}
\left[
\sum_j |EI_{i\to j}|
+
\sum_{\substack{(K,j):i\in K\\|K|=2}}
\frac{|Syn^{EID}_{K\to j}|}{|K|}
\right],
$$

$$
Hyper\text{-}ACS(i)
=
\frac{1}{n-1}
\left[
\sum_s |EI_{s\to i}|
+
\sum_{\substack{(K,j):j=i\\|K|=2}}
|Syn^{EID}_{K\to i}|
\right].
$$

Hyper-AMCE 与前两个指标保持同一结构，由 pairwise path AMCE 和二阶源成员贡献两部分构成：

$$
Hyper\text{-}AMCE(m)
=
AMCE(m)
+
\frac{1}{n-1}
\sum_{\substack{(K,j):m\in K\\|K|=2}}
\frac{|Syn^{EID}_{K\to j}|}{|K|}.
$$

本轮复算复用最新 1948-2026 数据、最新训练好的 MLP 和全量二阶 joint EI 表。由于全量 `102660` 个二阶项没有对应 null/bootstrap 分布，本节先不使用截图中的 $z_{K\to j}\ge2$ 显著性门控；所有全量二阶项都进入 `|Syn|` 汇总。

![Original hyper metric full order-2](../../results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/original_hyper_metric_full_order2/original_hyper_metric_full_order2_map.png)

*图 3. 原始 hyper 指标在最新 1948-2026 MLP-TM-EI 输出上的复算。a 为 pairwise EI baseline 的 ACE/ACS；b 为 pairwise path AMCE；c 为原始 `EI + Syn` 的 Hyper-ACE/Hyper-ACS；d 为 `AMCE + Syn` 的 Hyper-AMCE。*

完整性与 Syn 汇总：

- 二阶 hyperedge 数：`102660 / 102660`
- $Syn$ 定义：`abs(delta2)`，其中 `delta2 = EI({a,b}->j)-EI(a->j)-EI(b->j)`
- 显著性门控：未使用，记为 `none_full_scan_no_null_z`
- signed $\Delta_2$ 正值：`99818`
- signed $\Delta_2$ 负值：`2842`
- $\Delta_2$ 范围：`[-0.001024, 0.008207]`

原始 Hyper-ACE/ACS top 节点如下：

| Rank | Hyper-ACE | Hyper-ACS |
|---:|---|---|
| 1 | component_01 (0.03843) | component_03 (0.02394) |
| 2 | component_02 (0.02945) | component_27 (0.02272) |
| 3 | component_04 (0.02496) | component_16 (0.02202) |
| 4 | component_16 (0.02467) | component_17 (0.02189) |
| 5 | component_17 (0.02462) | component_02 (0.02172) |
| 6 | component_27 (0.02455) | component_10 (0.02047) |
| 7 | component_03 (0.02414) | component_01 (0.02028) |
| 8 | component_10 (0.02413) | component_04 (0.02020) |
| 9 | component_19 (0.02409) | component_19 (0.01970) |
| 10 | component_05 (0.02336) | component_05 (0.01952) |

原始 Hyper-AMCE top 节点如下：

| Rank | Hyper-AMCE | Pairwise AMCE | Syn contribution |
|---:|---|---:|---:|
| 1 | component_01 (0.01899) | 8.228e-06 | 0.01898 |
| 2 | component_16 (0.01734) | 6.011e-06 | 0.01734 |
| 3 | component_02 (0.01723) | 5.680e-06 | 0.01723 |
| 4 | component_36 (0.01707) | 5.656e-06 | 0.01706 |
| 5 | component_19 (0.01702) | 3.337e-06 | 0.01702 |
| 6 | component_06 (0.01700) | 3.271e-06 | 0.01700 |
| 7 | component_03 (0.01694) | 6.415e-06 | 0.01693 |
| 8 | component_27 (0.01692) | 5.487e-06 | 0.01691 |
| 9 | component_24 (0.01682) | 3.174e-06 | 0.01682 |
| 10 | component_26 (0.01674) | 5.227e-06 | 0.01674 |

这版原始 hyper 指标与 Scheme A 的差异很大。原因是全量无门控的 `|Syn|` 项累计后量级约为 `1e-2`，而 pairwise path AMCE 约为 `1e-6`，所以 Hyper-AMCE 几乎完全由二阶源成员贡献主导。若要复现截图中的显著性筛选口径，需要为全量二阶项补充 null/bootstrap 并施加 $z_{K\to j}\ge2$ 门控。

## 原始 Hyper 指标：补回显著性门控

为对齐旧截图口径，本节使用最新 1948-2026 数据和最新 MLP，但恢复旧 hypergraph 的候选二阶项、block-shift null 和 $|z|\ge2$ 门控。该版本不是全量 `102660` 二阶扫描，而是先从 pairwise EI 中选候选二源项，再对候选项做 null 检验：

- 候选二阶项：`2764`
- null reps：`20`
- 门控：`|z| >= 2`
- 通过门控的二阶项：`455`
- 候选项 `abs(delta2)` 总和：`2.05368`
- 通过门控后 `abs(delta2)` 总和：`0.797970`

![Original hyper metric with z gate](../../results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/original_hyper_metric_z2_null20/original_hyper_metric_full_order2_map.png)

*图 4. 补回 $|z|\ge2$ 显著性门控后的原始 hyper 指标。a-b 为 pairwise path baseline；c-d 为 `EI + Syn` 和 `AMCE + Syn`。*

同时保留一张旧截图同布局图，便于直接横向比较：

![PEID z2 null20 old layout](../../results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/peid_hypergraph_z2_null20/pairwise_vs_peid_z2_null20_map.png)

门控后，二阶项不再像无门控全量版那样支配整张图。Hyper-AMCE 仍由二阶源成员贡献主导，但量级从无门控全量版的 `0.01899` 峰值降到 `0.001724` 峰值。Hyper-ACE 的 top 节点也更接近 pairwise baseline，只是在 source-membership 强节点上有局部抬升。

门控版 `AMCE + Syn` 的 top 节点如下：

| Rank | Hyper-AMCE | Pairwise AMCE | Syn contribution |
|---:|---|---:|---:|
| 1 | component_01 (0.001724) | 8.228e-06 | 0.001716 |
| 2 | component_02 (0.000917) | 5.680e-06 | 0.000911 |
| 3 | component_19 (0.000882) | 3.337e-06 | 0.000879 |
| 4 | component_16 (0.000821) | 6.011e-06 | 0.000815 |
| 5 | component_04 (0.000812) | 4.908e-06 | 0.000807 |
| 6 | component_31 (0.000773) | 2.601e-06 | 0.000771 |
| 7 | component_12 (0.000736) | 3.002e-06 | 0.000733 |
| 8 | component_10 (0.000702) | 5.942e-07 | 0.000701 |
| 9 | component_03 (0.000698) | 6.415e-06 | 0.000691 |
| 10 | component_13 (0.000680) | 2.490e-06 | 0.000678 |

## 结果文件

| 内容 | 文件 |
|---|---|
| 完整 Scheme A 结果目录 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/` |
| 对比主图 PNG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2/pairwise_dense_vs_scheme_a_full_order2_map.png` |
| 对比主图 SVG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2/pairwise_dense_vs_scheme_a_full_order2_map.svg` |
| 对比主图 PDF | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2/pairwise_dense_vs_scheme_a_full_order2_map.pdf` |
| manifest | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/manifest.json` |
| 完整二阶 joint EI 表 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/joint_order2_full.csv` |
| 条件 EI 矩阵 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/conditioned_ei_matrix.csv` |
| 背景项计数矩阵 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/conditioned_background_counts.csv` |
| 完整 Scheme A ACE/ACS 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/scheme_a_full_order2_gateway_scores.csv` |
| 完整 Scheme A AMCE 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/scheme_a_full_order2_mediator_scores.csv` |
| 无二阶协同 ACE/ACS 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/pairwise_dense_gateway_scores.csv` |
| 无二阶协同 AMCE 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2/pairwise_dense_mediator_scores.csv` |
| $L=1$ 结果目录 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2_l01/` |
| $L=1$ 对比主图 PNG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/scheme_a_full_order2_l01/pairwise_dense_vs_scheme_a_full_order2_map.png` |
| $L=1$ manifest | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/scheme_a_full_order2_l01/manifest.json` |
| 原始 Hyper 指标结果目录 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/original_hyper_metric_full_order2/` |
| 原始 Hyper 指标主图 PNG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/original_hyper_metric_full_order2/original_hyper_metric_full_order2_map.png` |
| 原始 Hyper 指标 manifest | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/original_hyper_metric_full_order2/manifest.json` |
| 原始 Hyper-ACE/ACS 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/original_hyper_metric_full_order2/original_hyper_gateway_scores.csv` |
| 原始 Hyper-AMCE 排名 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/original_hyper_metric_full_order2/original_hyper_mediator_scores.csv` |
| 门控版 PEID hypergraph 结果目录 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/peid_hypergraph_z2_null20/` |
| 门控版旧布局主图 PNG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/peid_hypergraph_z2_null20/pairwise_vs_peid_z2_null20_map.png` |
| 门控版 `AMCE + Syn` 结果目录 | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/original_hyper_metric_z2_null20/` |
| 门控版 `AMCE + Syn` 主图 PNG | `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/fig/runge/original_hyper_metric_z2_null20/original_hyper_metric_full_order2_map.png` |
