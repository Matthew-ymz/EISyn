# Broadcast redundancy

本文记录一个用于 PEID `1 -> n` 场景的目标侧冗余指标。核心目标是回答：同一个源侧干预变量的信息，是被多个目标变量重复承载，还是必须把多个目标变量合起来才能读出。

## 1. 设定

设源侧只有一个源块，记为 $S$。目标侧为向量变量 $\mathbf{Y}$，并给定一个目标划分

$$
\mathcal{Q}=\{B_1,\dots,B_m\},
\qquad
\mathbf{Y}=(\mathbf{Y}_{B_1},\dots,\mathbf{Y}_{B_m}).
\tag{1}
$$

所有互信息都在 PEID 的源侧最大熵干预分布下计算：

$$
q^{\max}(s)\,p(\mathbf{y}\mid s).
\tag{2}
$$

因此这里的 $EI(S\to \mathbf{Y}_{B_r})$ 不是观测相关，而是同一干预通道下的有效信息。

## 2. 定义

定义源 $S$ 到目标划分 $\mathcal{Q}$ 的 broadcast redundancy 为

$$
BR_{\mathcal{Q}}(S\to \mathbf{Y})
:=
\sum_{r=1}^{m} EI(S\to \mathbf{Y}_{B_r})
-
EI(S\to \mathbf{Y}).
\tag{3}
$$

其中

$$
EI(S\to \mathbf{Y}_{B_r})
=I_{q^{\max}}(S;\mathbf{Y}_{B_r}),
\qquad
EI(S\to \mathbf{Y})
=I_{q^{\max}}(S;\mathbf{Y}).
\tag{4}
$$

式 (3) 的符号约定是：正值表示冗余，负值表示互补。

用熵展开可得

$$
BR_{\mathcal{Q}}(S\to \mathbf{Y})
=
TC(\mathbf{Y}_{B_1},\dots,\mathbf{Y}_{B_m})
-
TC(\mathbf{Y}_{B_1},\dots,\mathbf{Y}_{B_m}\mid S).
\tag{5}
$$

因此，broadcast redundancy 衡量的是：知道源 $S$ 以后，目标块之间的总相关是否下降。

## 3. 解释

- $BR_{\mathcal{Q}}>0$：多个目标块重复承载了来自同一个源的信息。典型例子是广播或复制：$Y_1=S, Y_2=S$，此时两个单目标 EI 相加会重复计算同一份源信息。
- $BR_{\mathcal{Q}}=0$：目标块对源的读出近似可加。例如源 $S=(A,B)$，目标分别复制两个独立分量 $Y_1=A, Y_2=B$，则两个目标没有重复承载同一份信息。
- $BR_{\mathcal{Q}}<0$：目标块互补编码源。典型例子是 $Y_1=N, Y_2=S\oplus N$，其中 $N$ 是独立噪声；单看任一目标读不出 $S$，联合目标才能读出 $S$。

如果需要非负读数，建议只在展示层拆成两项：

$$
BR_{\mathcal{Q}}^{+}=\max\{BR_{\mathcal{Q}},0\},
\qquad
BC_{\mathcal{Q}}^{+}=\max\{-BR_{\mathcal{Q}},0\}.
\tag{6}
$$

其中 $BC_{\mathcal{Q}}^{+}$ 可称为 broadcast complementarity。不要把式 (6) 解释成完整 PID 原子；它只是式 (3) 的正负部分。

## 4. 与 PEID 源侧协同和 gateway phi 的关系

当前 PEID 主定义处理的是源侧划分：

$$
Syn^{EID}_{\mathcal{P}}(X_A\to T)
=
EI(X_A\to T)-\sum_{M\in\mathcal{P}}EI(M\to T).
\tag{7}
$$

在源侧最大熵独立干预下，式 (7) 可写成条件 total correlation，因此是非负的源侧非加性量。

Broadcast redundancy 是式 (7) 的目标侧对偶问题，但符号方向相反：它比较“各目标块分别读源”的信息总和与“联合目标读源”的信息量。源侧最大熵干预会消掉源侧观测相关，但不会消掉目标侧由同一个源诱导的重复编码。因此 `1 -> n` 需要单独报告式 (3)。

对某一个候选变量 $T$，可以把整个系统状态 $\mathbf{x}=(X_1,\dots,X_d)$ 作为源侧集合，定义变量级 gateway phi 为

$$
\phi^{EID}(T)
:=
EI(\mathbf{x}\to T)
-
\sum_{i=1}^{d} EI(X_i\to T).
\tag{8}
$$

这里所有项仍在同一组最大熵独立干预样本下估计。等价地，若 $\mathcal{P}_1=\{\{X_1\},\dots,\{X_d\}\}$ 是单源划分，则

$$
\phi^{EID}(T)=Syn^{EID}_{\mathcal{P}_1}(\mathbf{x}\to T).
\tag{9}
$$

式 (8) 和 broadcast redundancy 形成对照：$BR_{\mathcal{Q}}$ 固定一个源，检查它的信息是否被多个目标重复承载；$\phi^{EID}(T)$ 固定一个目标变量，检查整个系统对它的联合因果约束是否超过所有单源约束之和。因此 $\phi^{EID}(T)>0$ 表示 $T$ 的机制输入存在不可由单边 EI 相加解释的联合结构，可作为识别 causal gateway 的变量级读数。若目标是排名 gateway，建议同时报告 $EI(\mathbf{x}\to T)$、$\sum_i EI(X_i\to T)$ 和 $\phi^{EID}(T)$，避免把低联合 EI 下的估计噪声解释成结构性 gateway。

### 4.1 Runge 1948-2026 MLP 结果中的重叠性

在 Runge 1948-2026 daily SLP 的缓存 MLP transition model 上，我们用同一批最大熵干预预测样本和同一 Gaussian log-det MI 估计器，分别计算了 $\phi^{EID}(T)$ 与 $BR(S)$。为避免短期 persistence 主导结果，单源求和时默认排除自环项 $X_i(t)\to X_i(t+1)$。

![Runge gateway phi and broadcast redundancy](../../fig/runge_gateway_phi_broadcast_redundancy_map.png)

这张图显示两者确实有很强的经验重叠。原因是二者都在比较“联合 EI”和“单源/单目标 EI 之和”，只是固定的方向不同：

- $\phi^{EID}(T)$ 固定目标 $T$，问整个系统是否以联合方式约束这个目标。
- $BR(S)$ 固定源 $S$，问这个源的信息是否被多个目标重复承载。

当同一组 component 既作为源又作为目标，并且用同一 MLP 通道估计 EI 时，强联合依赖的区域会同时影响两张图。因此它们不是完全独立的两个发现工具，而是同一个源-目标 EI 矩阵的两个投影。实际解释时不要把两张图当成两份独立证据相加。

但二者也不是同一个指标。$\phi^{EID}(T)$ 是目标侧 gateway 候选读数；Runge 结果中它全部为正，说明许多目标节点的联合全系统输入 EI 高于跨节点单源 EI 之和。$BR(S)$ 是源侧 broadcaster 候选读数；Runge 结果中它全部为负，说明当前 MLP 没有显示一源多目标广播复制结构，而是更接近目标侧互补编码。因此，这组结果支持用 $\phi^{EID}(T)$ 辅助筛选 causal gateway，但不支持把 $BR(S)$ 解释为已经识别出正冗余 broadcaster。若仍要排序 broadcaster，只能报告“最不互补”的源节点，而不能称为强 broadcast redundancy。

## 5. 与已有文献的关系

文献搜索结论是：`broadcast redundancy` 这个名字没有成为通用术语，但式 (3) 的 Shannon 形式已经存在。最近关于高阶信息指标的论文把

$$
RSI(\mathbf{X};Y)
=
\sum_j I(X_j;Y)-I(\mathbf{X};Y)
\tag{10}
$$

称为 redundancy-synergy index。若令 $\mathbf{X}$ 为目标块集合 $(\mathbf{Y}_{B_1},\dots,\mathbf{Y}_{B_m})$，令 $Y=S$，则式 (10) 与本文的 $BR_{\mathcal{Q}}$ 相同。本文的新意不在 Shannon 代数本身，而在于把它放到 PEID 的最大熵干预通道中，用来解释目标侧的一源多目标广播冗余。

已有应用主要来自神经编码和高阶依赖分析：

- Gat 和 Tishby 早期讨论了行为猴神经元之间的 synergy 与 redundancy。
- Chechik 等提出 group redundancy measures，并用于比较 inferior colliculus 与 primary auditory cortex 中的群体神经编码冗余。
- Schneidman 等在 population code 中使用了符号相反的 whole-minus-sum 读数 $I(R_1,R_2;S)-I(R_1;S)-I(R_2;S)$；这和二目标 broadcast redundancy 只差一个负号。
- Timme 等综述并比较了 redundancy-synergy index、interaction information、PID 等多变量信息指标，并展示了神经 spiking 数据应用。
- Rosas、Mediano 等的 O-information 路线提供了无向的 redundancy/synergy dominance 读数；后续工作明确比较了有向 RSI 与无向 O-information。

因此，如果论文里引入 `broadcast redundancy`，建议表述为：它是 **RSI 在 PEID 目标侧一源多目标通道上的定向、干预式特化**。

## 6. 计算建议

离散机制上，直接复用 PEID 的 EI 计算：

1. 选定源块 $S$ 和目标划分 $\mathcal{Q}$。
2. 在 $S$ 上采样 $q^{\max}(s)$，通过机制得到联合目标样本 $\mathbf{Y}$。
3. 对每个目标块估计 $EI(S\to \mathbf{Y}_{B_r})$。
4. 对联合目标估计 $EI(S\to \mathbf{Y})$。
5. 用式 (3) 得到 $BR_{\mathcal{Q}}$，并可用式 (6) 展示正负部分。

连续机制上，应让所有 MI 项共享同一批干预样本和同一类估计器，例如同一套 Gaussian log-det 或 transport-map 密度估计流程。由于式 (3) 是多个 MI 的差，估计偏差会直接影响符号；正式实验应配套 bootstrap、permutation null 或跨 seed 稳定性检查。

计算 $\phi^{EID}(T)$ 时，把目标变量 $T$ 固定，复用同一批全系统干预样本估计 $EI(\mathbf{x}\to T)$ 和所有 $EI(X_i\to T)$。用于 gateway 排名时，优先比较通过 null test 或跨 seed 稳定的正 $\phi^{EID}$，再用联合 EI 过滤弱信号变量。

## 7. 命名建议

推荐字段名：

- `broadcast_redundancy`：式 (3) 的 signed net value。
- `broadcast_redundancy_pos`：式 (6) 的正部分。
- `broadcast_complementarity_pos`：式 (6) 的负部分取正。
- `broadcast_redundancy_ratio`：可选归一化，定义为 $BR_{\mathcal{Q}} / EI(S\to \mathbf{Y})$，仅在联合 EI 明显大于零时报告。
- `gateway_phi_eid`：式 (8) 的变量级 signed net value。
- `gateway_phi_eid_pos`：$\max\{\phi^{EID}(T),0\}$，用于 causal gateway 排名。

## 参考文献

- Yang, M., Wang, S., & Zhang, J. (2026). *Partial Effective Information Decomposition for Synergistic Causality*. arXiv:2605.03267. https://arxiv.org/abs/2605.03267
- Gat, I., & Tishby, N. (1998). *Synergy and Redundancy among Brain Cells of Behaving Monkeys*. NeurIPS 11. https://proceedings.neurips.cc/paper/1998/hash/7a6a74cbe87bc60030a4bd041dd47b78-Abstract.html
- Chechik, G., Globerson, A., Anderson, M. J., Young, E. D., Nelken, I., & Tishby, N. (2002). *Group Redundancy Measures Reveal Redundancy Reduction in the Auditory Pathway*. NeurIPS 14. https://papers.neurips.cc/paper/2021-group-redundancy-measures-reveal-redundancy-reduction-in-the-auditory-pathway
- Schneidman, E., Bialek, W., & Berry, M. J. (2003). *Synergy, Redundancy, and Independence in Population Codes*. Journal of Neuroscience, 23(37), 11539-11553. https://doi.org/10.1523/JNEUROSCI.23-37-11539.2003
- Timme, N., Alford, W., Flecker, B., & Beggs, J. M. (2014). *Synergy, redundancy, and multivariate information measures: an experimentalist's perspective*. Journal of Computational Neuroscience, 36, 119-140. https://doi.org/10.1007/s10827-013-0458-4
- Rosas, F. E., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). *Quantifying high-order interdependencies via multivariate extensions of the mutual information*. Physical Review E, 100, 032305. https://doi.org/10.1103/PhysRevE.100.032305
- Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023). *Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex*. Communications Biology, 6, 451. https://doi.org/10.1038/s42003-023-04843-w
- Rosas, F. E., Mediano, P. A. M., & Gastpar, M. (2024). *Characterising Directed and Undirected Metrics of High-Order Interdependence*. arXiv:2404.07140. https://arxiv.org/abs/2404.07140
- Neri, M., Vinchhi, D., Ferreyra, C., Robiglio, T., Ates, O., Ontivero-Ortega, M., Brovelli, A., Marinazzo, D., & Combrisson, E. (2024). *HOI: A Python toolbox for high-performance estimation of Higher-Order Interactions from multivariate data*. Journal of Open Source Software, 9(103), 7360. https://doi.org/10.21105/joss.07360
