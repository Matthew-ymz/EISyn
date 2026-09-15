# HCP 57 人 PED 与 O-information 复现报告

## 摘要

本报告在同一批 57 名 HCP S1200 被试上，比较 REST 与七个任务态的全部 35 个 Yeo7 三网络组合。PED 与 O-information 均按目标论文公开代码的估计口径实现。

核心结果如下。

1. **PED 不估计连续空间互信息。** 作者先对每个时间序列作 z-score，再以 0 为阈值二值化；随后用离散 shared-exclusion（Hsx/SxPID）对三元联合熵进行 Partial Entropy Decomposition。它不是 Transport Map、KNN 或 KDE。O-information 则使用 HOI 默认的 Gaussian-copula 熵估计与有限样本偏差校正。
2. **REST 的 PED 冗余最高，但任务态的 PED 协同普遍略高。** 35 个组合等权平均后，REST 的 Red 为 0.443 bits、Syn 为 0.361 bits；任务态 Red 为 0.297--0.405 bits，Syn 为 0.382--0.405 bits。
3. **PED 的冗余/协同平衡随状态明显重组。** REST 的 35/35 个群体均值组合为 Red > Syn；EMOTION 为 35/35 个 Syn > Red。其余状态中，协同占优组合数为 GAMBLING 29、LANGUAGE 30、MOTOR 33、RELATIONAL 18、SOCIAL 15、WM 32。
4. **组合定位有清晰分工。** PED 冗余高值主要沿 DAN--VAN--FPN/Visual 骨架分布；PED 协同高值更多涉及 Limbic，并随状态换接 Visual、Somatomotor、DAN、FPN 或 DMN。
5. **O-information 在群体层面仍全部为正。** 八状态的 35 个组合均没有负的跨被试均值，因此 O-information 支持“冗余占优程度变化”，不支持稳定的群体 O-Syn 三元组。

![PED 与 O-information 状态概览](../../results/hcp_57_ped_oinfo_replication/state_overview.png)

**图 1｜状态层面的三元高阶信息。** 每名被试先对 35 个三元组取平均。a，PED Red；b，PED Syn；c，有符号 O-information。箱线图统计单位为被试，圆点为 57 人均值。O-information 的 0 线区分 redundancy-dominated 与 synergy-dominated。

## 1. 目标论文的估计方法

### 1.1 PED：离散 shared-exclusion，而非连续互信息估计

目标论文公开代码的 PED 流程为：

```python
z_data = zscore(data_sub, axis=1)
res = compute_PED(discretize(z_data), norm=False)
```

其中 `discretize` 将正值映射为 1、负值映射为 0。对于每个三元组 $(X_1,X_2,X_3)$，程序由八种二元联合状态的经验频率构造离散概率质量函数，不加伪计数；再把联合状态本身作为确定性目标，调用 SxPID 的 informative shared-exclusion redundancy function。对每个冗余格节点 $\alpha$ 与实现 $\boldsymbol{x}$，其局部 informative redundancy 为

$$
i_{\cap}^{+}(\boldsymbol{x};\alpha)
=-\log_2 P\!\left(\bigcup_{A\in\alpha}\{\boldsymbol{X}_A=\boldsymbol{x}_A\}\right).
$$

先按经验状态概率求期望，再在 18 节点三变量冗余格上作 Möbius 反演，得到非负的 informative PED 原子。论文代码定义：

$$
\mathrm{PED\ Red}=H_{\partial}(\{1\}\{2\}\{3\}),
$$

而 PED Syn 是以下七个高阶原子的和：

$$
\begin{aligned}
\mathrm{PED\ Syn}={}&H_{\partial}(\{3\}\{12\})
+H_{\partial}(\{2\}\{13\})
+H_{\partial}(\{1\}\{23\})\\
&+H_{\partial}(\{12\}\{13\}\{23\})
+H_{\partial}(\{13\}\{23\})
+H_{\partial}(\{12\}\{23\})
+H_{\partial}(\{12\}\{13\}).
\end{aligned}
$$

公开代码设置 `norm=False`，所以结果保留原生 bits，不除以联合熵。本复现嵌入作者发布的三变量 18 节点反演矩阵；在随机二元分布上与原 SxPID 输出逐原子核对，最大绝对误差为 $2.78\times10^{-16}$ bits。

这一方法的稳健性来自秩序较低的离散频率估计和幅度无关的符号编码，但代价也很明确：二值化会丢弃幅度信息，阈值附近样本可能翻转；有限样本偏差与时间窗长度仍然存在。它不是“连续空间互信息的稳健估计器”。

### 1.2 O-information：Gaussian-copula entropy

目标论文调用 HOI `Oinfo.fit(minsize=3, maxsize=3)`，未覆写默认估计器；对应 `method="gc"`。每个连续变量先经经验秩映射到标准高斯边缘，再通过协方差 Cholesky/log-determinant 估计多元高斯熵，并应用有限样本偏差校正。三变量 O-information 为

$$
\Omega(X_1,X_2,X_3)
=\sum_{i=1}^{3}H(X_i)
-\sum_{1\le i<j\le3}H(X_i,X_j)
+H(X_1,X_2,X_3).
$$

$\Omega>0$ 表示冗余占优，$\Omega<0$ 表示协同占优。O-information 只给出净平衡，不把冗余与协同分别分解为非负原子。

## 2. 数据与受控实验口径

- 被试：57 人，REST 与七任务完整配对。
- 状态：REST、EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL、WM。
- 变量：Schaefer-1000 分区先在 Yeo7 网络内提取 PC1，得到 Visual（V）、Somatomotor（SM）、Dorsal attention（DAN）、Salience/ventral attention（VAN）、Limbic（Lim）、Control（FPN）和 Default（DMN）。
- 组合：固定全部 $\binom{7}{3}=35$ 个无序三元组。
- 主分析：每个状态使用完整序列；长度依次为 1200、176、253、316、284、232、274、405。
- 长度敏感性：所有状态统一使用前 176 点，并在该窗口内独立中心化、二值化或 copula 变换。
- 时间稳定性：每个完整序列的前半与后半独立重估。
- 群体统计：5,000 次被试 bootstrap 置信区间；任务--REST 使用双侧配对 Wilcoxon，并在七任务内作 BH 校正。
- 组合稳定性：留一被试重算最高组合。
- PED 数值审计：非负容差为 $10^{-10}$ bits。四种窗口共检查 1,149,120 个 partial atoms；599 个负值全部在容差内，最小值为 $-7.77\times10^{-16}$ bits，显著违规为 0。未使用静默截断。

这属于“同算法、同队列上的网络级复现”，不是目标论文数据表的逐项复制。目标论文使用 100 名 HCP unrelated subjects 和 116 个皮层/皮层下区域；本报告使用现有 57 人数据与 Yeo7 PC1，因此结论粒度是七网络组合。

## 3. 状态层面结果

| 状态 | PED Red，均值 [95% CI] | PED Syn，均值 [95% CI] | 有符号 O-information，均值 [95% CI] |
| --- | ---: | ---: | ---: |
| REST | 0.443 [0.403, 0.489] | 0.361 [0.344, 0.376] | 0.311 [0.238, 0.388] |
| EMOTION | 0.297 [0.283, 0.311] | 0.405 [0.400, 0.409] | 0.096 [0.065, 0.131] |
| GAMBLING | 0.350 [0.333, 0.368] | 0.392 [0.384, 0.398] | 0.187 [0.140, 0.252] |
| LANGUAGE | 0.342 [0.330, 0.353] | 0.399 [0.396, 0.402] | 0.137 [0.110, 0.169] |
| MOTOR | 0.302 [0.287, 0.319] | 0.387 [0.381, 0.393] | 0.133 [0.102, 0.167] |
| RELATIONAL | 0.398 [0.379, 0.418] | 0.382 [0.376, 0.387] | 0.217 [0.178, 0.258] |
| SOCIAL | 0.405 [0.382, 0.428] | 0.381 [0.375, 0.387] | 0.252 [0.207, 0.297] |
| WM | 0.326 [0.309, 0.347] | 0.399 [0.392, 0.405] | 0.156 [0.118, 0.200] |

相对 REST，EMOTION、GAMBLING、LANGUAGE、MOTOR 和 WM 的 PED Red 显著降低，BH $q\le5.12\times10^{-4}$；RELATIONAL 与 SOCIAL 未通过校正。PED Syn 则在 EMOTION、GAMBLING、LANGUAGE、MOTOR 和 WM 显著升高，BH $q\le0.0133$；RELATIONAL 与 SOCIAL 未通过校正。O-information 的显著变化方向与 PED Red 一致：上述五个任务均低于 REST，RELATIONAL 与 SOCIAL 不显著。

统一到 176 点后，REST 的 PED Red 从 0.443 降到 0.333 bits，PED Syn 从 0.361 升到 0.397 bits，说明 REST 的绝对 Red/Syn 平衡对窗口长度尤其敏感。其他任务因本身更接近 176 点，变化较小。

## 4. 协同和冗余主要分布在哪些组合

![PED Red、PED Syn 与 O-information 的主要组合](../../results/hcp_57_ped_oinfo_replication/top_triplet_heatmaps.png)

**图 2｜跨状态高值三元组。** a，PED Red；b，PED Syn；c，有符号 O-information。PED 面板按跨状态平均值选择前 12 个组合；O-information 面板按绝对群体均值选择前 12 个组合。三个面板使用各自的绝对 bits 色标。

| 状态 | PED Red 第一组合 | PED Syn 第一组合 | O-information 第一冗余组合 | PED 群体平衡（Red / Syn 占优组合数） |
| --- | --- | --- | --- | ---: |
| REST | Lim+FPN+DMN，0.506 | V+VAN+Lim，0.376 | SM+DAN+VAN，0.513 | 35 / 0 |
| EMOTION | V+DAN+FPN，0.368 | V+VAN+Lim，0.412 | SM+DAN+VAN，0.234 | 0 / 35 |
| GAMBLING | V+DAN+FPN，0.486 | V+Lim+FPN，0.404 | DAN+VAN+FPN，0.510 | 6 / 29 |
| LANGUAGE | DAN+VAN+FPN，0.543 | V+DAN+Lim，0.415 | DAN+VAN+FPN，0.702 | 5 / 30 |
| MOTOR | DAN+VAN+FPN，0.419 | DAN+VAN+DMN，0.403 | DAN+VAN+FPN，0.433 | 2 / 33 |
| RELATIONAL | V+DAN+FPN，0.603 | SM+Lim+FPN，0.401 | V+DAN+FPN，0.721 | 17 / 18 |
| SOCIAL | V+DAN+VAN，0.542 | SM+DAN+Lim，0.403 | V+DAN+VAN，0.513 | 20 / 15 |
| WM | V+DAN+FPN，0.461 | SM+Lim+FPN，0.408 | DAN+VAN+FPN，0.460 | 3 / 32 |

### 4.1 PED 冗余骨架

任务态冗余的核心是 DAN 与 FPN/VAN/Visual 的组合：`V+DAN+FPN` 在 EMOTION、GAMBLING、RELATIONAL、WM 排名第一；`DAN+VAN+FPN` 在 LANGUAGE、MOTOR 第一；SOCIAL 为 `V+DAN+VAN`。REST 不同，顶部转为 `Lim+FPN+DMN`，但其留一被试第一名稳定度只有 38/57，第二候选 `VAN+FPN+DMN` 为 19/57，表明 REST 顶部两个组合接近。

除 REST 外，PED Red 的状态第一组合基本稳定：六个状态为 57/57 次留一仍第一，WM 为 56/57。

### 4.2 PED 协同组合

协同最高组合更频繁包含 Limbic：八个状态的第一组合中有七个包含 Lim，唯一例外是 MOTOR 的 `DAN+VAN+DMN`。其状态性表现为：

- REST 与 EMOTION：`V+VAN+Lim`；
- GAMBLING：`V+Lim+FPN`；
- LANGUAGE：`V+DAN+Lim`；
- RELATIONAL 与 WM：`SM+Lim+FPN`；
- SOCIAL：`SM+DAN+Lim`。

绝对 Syn 值的组合间跨度比 Red 小，顶部排序应谨慎解释。留一被试下，REST、GAMBLING、LANGUAGE、RELATIONAL、SOCIAL 的第一组合至少稳定 56/57 次；MOTOR 为 53/57，WM 为 54/57；EMOTION 最不确定，第一组合 40/57，`V+Lim+FPN` 为 15/57。

### 4.3 PED 与 O-information 的关系

PED Red 与有符号 O-information 的群体组合排序高度一致，各状态 Spearman $\rho=0.722$--0.960。用 PED Red − Syn 构成的净平衡与 O-information 比较，$\rho=0.722$--0.954。这说明两者在“哪些组合更冗余”上收敛，但不应把 O-information 数值当作 PED Red − Syn：两者使用不同分解定义和不同数据表示。

所有状态的 35 个 O-information 群体均值均为正。最接近 0 的是 LANGUAGE 的 `V+Lim+FPN`，均值仍为 0.0063 bits；因此个别被试的负值不能升级为稳定群体 O-Syn 结论。

## 5. 稳定性与鲁棒性

![长度与分半稳健性](../../results/hcp_57_ped_oinfo_replication/robustness.png)

**图 3｜窗口长度与时间分半稳定性。** 每个点表示一个状态中的一个三元组群体均值。上排比较完整序列与统一前 176 点；下排比较前半与后半。虚线为恒等线，状态图例置于数据区外。

| 指标 | 全长 vs 176 点：全部被试值 $\rho$ | 前半 vs 后半：全部被试值 $\rho$ | 群体组合排序：全长 vs 176 点 | 群体组合排序：前半 vs 后半 |
| --- | ---: | ---: | ---: | ---: |
| PED Red | 0.825 | 0.637 | 0.749--1.000 | 0.869--0.980 |
| PED Syn | 0.768 | 0.419 | 0.458--1.000 | 0.688--0.939 |
| O-information | 0.889 | 0.668 | 0.917--1.000 | 0.834--0.984 |

稳定性结论分两层：

- **群体组合排序较稳。** 除 REST 的 PED Syn 外，多数状态在统一长度和分半后仍保持较高的组合秩相关。
- **个体 PED Syn 较敏感。** 前后半的全部被试值相关只有 0.419；二值阈值、较短窗口和小幅 Syn 差异都会放大个体排序变化。
- **REST 的长度效应最明显。** 1200 点 REST 与 176 点任务直接比较，会同时改变离散频率偏差和采样稳定度；因此主报告保留全长复现，但状态差异必须结合统一长度结果解释。

稳健性措施包括：固定相同的 57 人与 35 个组合；每个敏感性窗口独立预处理；显式记录所有 partial atom 的数值非负审计；报告 bootstrap CI、配对检验与 BH 校正；并用留一被试与时间分半检查组合排序。没有用裁剪、伪计数或调参来美化 PED 结果。

## 6. 结论与边界

在当前 57 人 Yeo7 表征下，最稳健的结构结论是：**REST 更偏 PED 冗余；多数任务态更偏 PED 协同。任务冗余主要组织在 DAN--VAN--FPN/Visual 骨架上，而任务协同的顶部组合高度集中于 Limbic 与其他感觉、注意、控制或默认网络的跨系统耦合。**

同时保留三项边界：

1. PED 的二值化提高了对单调缩放与幅度异常值的耐受性，但损失幅度信息，且阈值附近不稳定。
2. PED Red 与 Syn 都是非负结构量；“协同占优”在本报告中仅指 Syn > Red，不等于 O-information 为负。
3. 35 个三元组高度重叠，本报告的组合排名是描述性定位。若要宣称某个组合具有任务特异机制，还需要逐组合配对置换、多重比较，以及独立 run 或独立队列复现。

## 7. 可复现产物

- 数值缓存：`results/hcp_57_ped_oinfo_replication/metrics.npz`
- 统计摘要：`results/hcp_57_ped_oinfo_replication/summary.json`
- 状态图：`results/hcp_57_ped_oinfo_replication/state_overview.png`
- 组合热图：`results/hcp_57_ped_oinfo_replication/top_triplet_heatmaps.png`
- 稳健性图：`results/hcp_57_ped_oinfo_replication/robustness.png`
- 计算入口：`scripts/analyze_hcp_ped_oinfo_57.py`
- 方法测试：`tests/test_hcp_ped_oinfo_57.py`

## 参考文献与实现依据

1. Santoro A, Neri M, Poetto S, et al. Charting higher-order models of brain function beyond pairwise interactions. *Nature Communications*. 2026;17:9207. [doi:10.1038/s41467-026-75959-w](https://doi.org/10.1038/s41467-026-75959-w)
2. Varley TF, Pope M, Puxeddu MG, Faskowitz J, Sporns O. Partial entropy decomposition reveals higher-order information structures in human brain activity. *PNAS*. 2023;120(30):e2300888120. [doi:10.1073/pnas.2300888120](https://doi.org/10.1073/pnas.2300888120)
3. Santoro et al. analysis code: [nplresearch/HOI_lenses_analysis](https://github.com/nplresearch/HOI_lenses_analysis)
