# HCP Lausanne-83 PhiEID 综合报告

## 目录

1. [综合结论](#1-综合结论)
2. [研究逻辑和证据层级](#2-研究逻辑和证据层级)
3. [DMF 中的 PhiEID 临界增强](#3-dmf-中的-phieid-临界增强)
4. [HCP REST1 Lausanne-83 PhiEID pilot](#4-hcp-rest1-lausanne-83-phieid-pilot)
5. [REST1 vs Working Memory PhiEID 对照](#5-rest1-vs-working-memory-phieid-对照)
6. [方法口径](#6-方法口径)
7. [解释边界和下一步](#7-解释边界和下一步)
8. [产物索引](#8-产物索引)

## 1. 综合结论

三组结果共同支持一个克制的主线：$\Phi^{EID}$ 可以作为跨脑区、跨网络高阶动态整合的筛查指标，但当前证据最适合支持“候选机制和网络家族”，不适合写成单一精确 atom 或采样协议无关的绝对相变常数。

在 DMF 模型中，83 区 whole-state uniform 干预口径的 signed $\Phi^{EID}$ 稳定把增强定位在 $G\approx1.7\text{-}1.9$ 的 firing-rate 快速转变带。模块级 greedy hierarchy 进一步显示，这个增强不是单个 pair 主导，而是从 `DMN+Sub`、`DMN+FPN+Sub` 到更大跨模块集合的嵌套联合读出。

在真实 HCP-YA REST1 数据中，10 个 subject、2 个 run 的 Lausanne-83 ROI 动力学显示 observed whole-state PhiEID 在 20 / 20 个 subject-run 中均高于 ROI-wise circular-shift null。REST1 的稳定分解读法不是 exact greedy atom，而是 top atom family 和 module participation：Visual、VAN、FPN 和 DMN 的 participation 显著高于 null，且 LR/RL 稳定性较高。

REST1 与 Working Memory 的对照显示，WM 任务态的整体 PhiEID 更高，并在扣除 null 后仍保留更强的跨 ROI / 跨网络高阶结构。WM 不是换成一套完全不同的网络家族，而是在 REST 已出现的 DMN、Visual、VAN 和 FPN 结构上进一步增强，并扩展到 Lim 和 Sub 等更大范围的多网络组合。

最稳妥的一句话表述是：在 Lausanne-83 ROI 动力学中，$\Phi^{EID}$ 捕捉到高于 circular-shift null 的 high-order cross-network transition structure；REST 中该结构主要落在 Visual、VAN、FPN 和 DMN，WM 进一步增强 Visual、FPN、DMN 以及更大范围多网络 atom，但 exact greedy atom 仍应作为候选示例而不是稳定机制本身。

## 2. 研究逻辑和证据层级

这三份报告对应三个层级。

第一层是 DMF 方法验证。它回答：当一个可控全脑动力学系统进入 firing-rate 快速转变区时，$\Phi^{EID}$ 是否能识别临界增强；如果能，增强来自简单 pair 还是跨模块层级联合读出。

第二层是 HCP REST1 pilot。它回答：在真实 resting-state fMRI ROI transition 中，whole-state PhiEID 是否高于保留单 ROI 时间结构、破坏跨 ROI 同步关系的 circular-shift null；分解层面哪些网络家族最稳定。

第三层是 REST1 vs Working Memory 对照。它回答：任务态是否只是在整体动态信息上升，还是在扣除 null 后仍增强了跨网络 high-order PhiEID；增强是否改变了模块家族。

因此，证据从“可控模型中的机制可识别性”推进到“真实 REST 数据中的高于 null 信号”，再推进到“任务态相对静息态的网络家族变化”。这条链条比单独报告某一个 greedy atom 更稳。

## 3. DMF 中的 PhiEID 临界增强

### 3.1 数据和目标

DMF 部分参考 Mediano et al. (2025) Fig. 6 的问题设定：扫描全局耦合强度 $G$，观察系统从低 firing-rate 状态进入高 firing-rate 状态时，信息分解指标是否在转变附近出现峰值。

本报告不是原文数值的严格复现。原文使用基于 DTI 的结构连接矩阵，并报告 $\Phi^R$；这里使用 F-TRACT atlas 的 Lausanne2008-33 子包中第一个 `count` 矩阵作为代理耦合矩阵，保留 83 个 Lausanne 脑区，按最大值归一化后乘以 0.2，得到代理耦合矩阵 $\mathbf{C}$。主指标是 83 区 whole-state uniform 干预下的 signed $\Phi^{EID}$。

扫描范围为

$$
G=1.1,\ldots,3.0.
$$

临界区先由 firing-rate 曲线定义，而不是先由 $\Phi^{EID}$ 定义。平均放电率从 $G=1.6$ 的 $4.727\ \mathrm{Hz}$ 上升到 $G=1.9$ 的 $8.195\ \mathrm{Hz}$；离散斜率在 $G=1.8$ 附近最大，$G=1.8$ 与 $G=1.9$ 的斜率几乎相同。因此这里把 $G=1.7\text{-}1.9$ 作为快速转变带，而不是把某一个网格点写成精确相变常数。

### 3.2 Whole-state PhiEID 主结果

主结果使用 83 区 whole-state uniform 干预口径。source 是 83 个 singleton region 的当前状态 $\{s_E^i(t)\}_{i=1}^{83}$，target 是下一步 whole-system 83D 状态 $\mathbf{s}_E(t+\tau)$。每个 $G$ 和 seed 下，都用当前 trace 的 source 均值与尺度把方差匹配的 uniform 最大熵干预映射回物理 $s_E$ 空间，再用 DMF 方程推进一步。最后把 source 和 target 标准化，使用 Gaussian block conditional total correlation 读取 signed raw $\Phi^{EID}$，不对负值做非负截断。

具体计算为

$$
\Phi^{EID}
= EI_{\mathrm{do}}(\{s_E^i(t)\}_{i=1}^{83};\mathbf{s}_E(t+\tau))
-\sum_{i=1}^{83}EI_{\mathrm{do}}(s_E^i(t);\mathbf{s}_E(t+\tau)).
$$

![DMF 83 区 whole-state PhiEID 主结果](../../fig/dmf_83_region_oracle_phi_eid_main_g11.png)

*图 1. 83 区 Kuramoto-aligned whole-state $\Phi^{EID}$ 主结果。灰色带为 $G=1.7\text{-}1.9$。$G=1.0$ 不参与主结果峰值判定，只作为边界点审计。*

| 证据 | 结果 | 读法 |
|---|---:|---|
| Firing-rate 快速上升段 | $G=1.7\text{-}1.9$ | 先用动力学曲线定义候选临界带 |
| 最大 firing-rate 离散斜率 | $G=1.8$，约 $13.060\ \mathrm{Hz}/G$ | 与 $G=1.9$ 的 $13.027$ 很接近，不写成精确单点 |
| 83 区 whole-state uniform 复验 | 8/8 seeds 峰值位于 $G=1.7$ | 主结果口径，稳定命中临界带 |
| Long-trace continuation 复验 | 8/8 seeds 峰值位于 $G=1.7$ | 支持 continuation 协议下的鲁棒识别 |
| Independent restart 复验 | 0/8 seeds 峰值落入 $G=1.7\text{-}1.9$ | 说明结论依赖采样协议 |

这条证据链的关键顺序是：先从 firing rate 确定转变带，再看 $\Phi^{EID}$ 是否在同一区域出现内部峰。结果显示，uniform 干预下 8 个 seed 的 whole-state $\Phi^{EID}$ 全部在 $G=1.7$ 达峰，clip fraction 为 0，说明峰值不是由边界裁剪制造的。

Long-trace continuation 进一步检查了短 trace 的不稳定性。把模拟长度从 `t_total=0.55` 增加到 `1.05` 后，8/8 条曲线的全局峰值都在 $G=1.7$，均值曲线也在 $G=1.7$ 达到最大值，$G=1.8$ 次高。短 trace 下少数低有效样本点会产生尖峰；过滤 `sample_count<300` 后，原短 trace continuation 结果同样有 8/8 个 seed 的峰值回到 $G=1.8$。

### 3.3 边界点和尺度敏感性

$G=1.0$ 的高 $\Phi$ 值不应解释为另一个相变，也不应简单写成算法缺陷。EI 衡量的是“当前干预状态能多可区分地预测未来状态”。在低耦合、低放电率边界，DMF 状态变化很小，$s_E(t)$ 到 $s_E(t+\tau)$ 接近自保持映射，残差小，因此 whole EI 和由它构成的 $\Phi^{EID}$ 都可能偏高。

判断临界点时需要同时满足两个条件：峰值位于扫描内部，并且与 firing-rate 快速上升区一致。$G=1.0$ 是扫描左边界，没有对应动力学转变，所以只保留为 boundary audit，峰值识别排除它。

![DMF PhiEID robustness](../../fig/dmf_phi_eid_robustness_longtrace.png)

*图 2. Long-trace continuation 鲁棒性验证。8/8 个 seed 的全局峰值都在 $G=1.7$，top-2 与 top-3 也全部命中 $G=1.7\text{-}1.9$。*

主结果使用“每个 $G$ 下独立做 source-scale matching 和 target z-scoring”的口径。这个处理比较的是同一 $G$ 附近的机制结构，而不是让跨 $G$ 的物理尺度变化直接支配 entropy 和 EI。

如果取消逐 $G$ 标准化，改为每个 seed 在整个 $G$ sweep 上共用一套 source 标准化参数，并在物理 $s_E$ 单位下计算 target entropy，那么 uniform 干预的峰值不再落在临界带。8/8 个 seed 都错过 $G=1.7\text{-}1.9$，median peak 移动到 $G=2.3$；均值曲线的最高点在 $G=2.1$，且 $G=2.0\text{-}2.5$ 一带整体偏高。

![DMF 83 区 no-per-G standardization PhiEID 对照](../../fig/dmf_83_region_oracle_no_g_standardization.png)

*图 3. 取消逐 $G$ 标准化后的 83 区 whole-state $\Phi^{EID}$ 对照。该图是尺度敏感性审计，不作为主结果口径。*

### 3.4 模块级层级分解

临界区识别回答“什么时候”全脑不可约信息最高；模块级分解回答“它由哪些源共同贡献”。直接在 83 个脑区上枚举所有二分不可行，因此这里先把 83 个 Lausanne 区域粗略映射到 7 个显示模块：

$$
\{\mathrm{DMN},\mathrm{Som},\mathrm{Vis},\mathrm{VAN},\mathrm{FPN},\mathrm{Lim},\mathrm{Sub}\}.
$$

对任意模块集合 $C$，定义模块级协同残差

$$
\Phi(C;Y)
= EI_{\mathrm{do}}(\mathbf{x}_C;Y)
-\sum_{i\in C}EI_{\mathrm{do}}(\mathbf{x}_i;Y),
$$

其中 $Y$ 是全脑下一时刻状态，$\mathbf{x}_i$ 是第 $i$ 个模块内所有当前脑区状态。

贪婪二分从当前模块块 $C$ 开始，枚举所有非平凡二分 $C=L\cup R$，选择 $\Phi(L;Y)+\Phi(R;Y)$ 最大的二分，并把父块还不能被两个子块解释的非负差值记为当前层残差：

$$
\gamma(C\rightarrow L,R;Y)
=\Phi(C;Y)-\Phi(L;Y)-\Phi(R;Y).
$$

这个 $\gamma$ 是 greedy hierarchy 下的 residual atom，不是严格的 Möbius 纯阶原子。它回答的是“沿这条贪婪二分树，哪些模块集合仍需要被联合读取”。

![DMF 模块级 PhiEID 层级贪婪分解](../../fig/dmf_phi_eid_greedy_decomposition.png)

*图 4. DMF 模块级 $\Phi^{EID}$ 层级贪婪分解。*

模块级分解得到三点结论。

第一，模块级 $\Phi^{EID}$ 的峰值仍落在临界区附近。模块级 TM 复验中，uniform 干预为 8/8 seeds 峰值命中 $G=1.7\text{-}1.9$，clip fraction 为 0。这说明模块级结果可以作为 source-side 机制审计，但它和 83 区 whole-state 主结果不是同一数值口径，不能直接比较绝对值。

第二，临界区 residual 不由某一个二阶 pair 独占。跨所有 $G$ 汇总后，order 2 到 order 7 都有正贡献，累积量分别约为 4.886、7.211、9.270、10.008、9.940 和 10.086。

第三，峰值 $G=1.8$ 处形成一条可读的嵌套链。

| Atom | Order | Depth | Residual |
|---|---:|---:|---:|
| `DMN+Vis+VAN+FPN+Lim+Sub` | 6 | 1 | 1.502 |
| `DMN+Som+Vis+VAN+FPN+Lim+Sub` | 7 | 0 | 1.486 |
| `DMN+Vis+FPN+Lim+Sub` | 5 | 2 | 1.359 |
| `DMN+FPN+Lim+Sub` | 4 | 3 | 1.324 |
| `DMN+FPN+Sub` | 3 | 4 | 0.970 |
| `DMN+Sub` | 2 | 5 | 0.667 |

这条链说明，临界增强更像一个层级联合读出过程：局部组合有贡献，但最高 residual 仍需要跨多个功能系统一起读。

83 区预算受限局部二分用来检查“如果不先压缩成模块，脑区级分解会不会自然给出清楚的小组合”。答案是否定的。无额外约束时，局部搜索会退化成 single-region vs rest；加入 `min-split-size=5` 后，峰值仍在 $G=1.8$，但 top residual 变成粗块二分。这支持模块合并的必要性：83 区组合空间不会自然压缩成少数稳定、可命名的 pair。

## 4. HCP REST1 Lausanne-83 PhiEID pilot

### 4.1 数据和运行状态

HCP pilot 使用 10 个 subject 的 `REST1_LR` 和 `REST1_RL` resting-state fMRI，提取 Lausanne/Desikan-83 ROI 时间序列，用 Ridge 一步 transition model 计算 Gaussian log-det whole-state PhiEID，并和每个 ROI 独立 circular-shift null 对比。当前结果仍是 pilot，不是完整 HCP S1200 群体推断。

| 项目 | 设置 |
|---|---|
| HCP release | 2017 S1200 |
| Subjects | 100307, 103414, 105115, 110411, 111312, 113619, 115320, 117122, 118528, 118730 |
| Runs | REST1_LR, REST1_RL |
| Subject-run 数 | 20 |
| ROI | Lausanne/Desikan-83 |
| Time points | 1200 per subject-run |
| Null repetitions | 100 per subject-run |
| Main model | Ridge one-step transition |
| Main estimator | Gaussian log-det PhiEID screening |

所有 subject-run 都成功提取出 `1200 x 83` ROI time series，且 `synthetic = false`。

### 4.2 Whole-state PhiEID

当前 10-subject、2-run pilot 显示：真实 HCP ROI 动力学的 raw PhiEID 在每个 subject-run 上都高于 circular-shift null。

| 指标 | 数值 |
|---|---:|
| Subject-run 数 | 20 |
| 平均 observed raw PhiEID | 12.530882 bits |
| 平均 null raw PhiEID | 6.667682 bits |
| 平均 observed - null | 5.863200 bits |
| observed - null 范围 | 2.878884 到 14.772958 bits |
| empirical p-value <= 0.01 | 20 / 20 |
| median empirical p-value | 0.009901 |
| Ridge validation correlation 均值 | 0.840262 |
| Ridge RMSE / persistence RMSE 均值 | 1.035518 |
| Ridge 优于 persistence 的 subject-run 数 | 10 / 20 |
| 默认纯 MLP validation correlation 均值 | 0.774280 |
| 默认纯 MLP RMSE / persistence RMSE 均值 | 1.208291 |
| 默认纯 MLP 优于 Ridge 的 subject-run 数 | 0 / 20 |
| 默认纯 MLP 优于 persistence 的 subject-run 数 | 1 / 20 |

按 run 分开看：

| Run | Observed mean | Null mean | Difference mean | p <= 0.01 |
|---|---:|---:|---:|---:|
| REST1_LR | 13.624457 | 6.877152 | 6.747305 | 10 / 10 |
| REST1_RL | 11.437308 | 6.458213 | 4.979095 | 10 / 10 |

LR/RL 的 subject-level `observed - null` 相关为 `r = 0.779166`。这说明 whole-state PhiEID 的方向不依赖单个 phase-encoding direction；同一 subject 在 LR 和 RL 中的强弱排序也有较高一致性。

![HCP Lausanne-83 PhiEID null comparison](../../fig/hcp_lausanne_phi_eid_null_comparison.png)

*图 5. 每个 subject-run 的 observed raw PhiEID 和 null 均值。所有 observed 值都高于各自 100 个 null 的全部取值，因此每个 subject-run 的经验 p-value 都是当前 null 数下的最小值 `1 / 101 = 0.009901`。*

需要同时注意一个限制：Ridge validation correlation 较高，但只有 10 / 20 个 subject-run 的 RMSE 优于 persistence baseline。把一步预测器换成默认纯 MLP 后，MLP 没有改善 RMSE：平均 RMSE 从 Ridge 的 `0.725452` 升到 `0.843234`，20 / 20 个 subject-run 都未优于 Ridge，且只有 1 / 20 个 subject-run 优于 persistence。因此当前结果支持“真实数据中存在高于 circular-shift null 的高阶整合信号候选”，但不支持把当前 pilot 的拟合器直接从 Ridge 换成默认纯 MLP。

| 模型 | Mean RMSE | Mean RMSE / persistence | Mean validation corr | 优于 Ridge | 优于 persistence |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.725452 | 1.035518 | 0.840262 | - | 10 / 20 |
| Pure MLP | 0.843234 | 1.208291 | 0.774280 | 0 / 20 | 1 / 20 |

### 4.3 Phi 分解鲁棒性

这次 pilot 对每个 null time series 不只计算 whole PhiEID，也重复计算 module greedy decomposition 和 ROI leave-one-out burden。这样可以检验：观察到的模块 atom 和 ROI burden 是否也高于 null，而不只是 whole-state PhiEID 高于 null。

结论分三层。

1. Whole-state PhiEID 很稳：20 / 20 个 subject-run 都显著高于 null，LR/RL 相关也较高。
2. Module participation 是最稳的分解读法：Visual、VAN、FPN、DMN、Som 和 Lim 的 participation 都高于 null，其中 Visual、VAN、FPN、DMN 是最强四个模块；module participation 的 LR/RL 平均相关为 `r = 0.926870`。
3. Exact greedy atom 只能作为候选示例：多数 top module atoms 高于 null，FDR q = 0.010801；top-5 atom overlap 为 `0.700000`。但 module atom value vector 的 LR/RL 平均相关只有 `r = 0.077896`，说明 exact greedy atom 数值结构不够稳定。

![HCP Lausanne-83 PhiEID robustness](../../fig/hcp_lausanne_phi_eid_robustness.png)

### 4.4 Module atoms, participation 和 ROI burden

Top module atoms vs null 如下。

| Module atom | Observed mean | Null mean | Difference | Empirical p | FDR q |
|---|---:|---:|---:|---:|---:|
| DMN + Vis + VAN + FPN | 1.528929 | 0.002861 | 1.526068 | 0.009901 | 0.010801 |
| DMN + Som + Vis + VAN + FPN + Lim + Sub | 2.148013 | 1.120604 | 1.027408 | 0.009901 | 0.010801 |
| DMN + Vis + FPN | 1.003373 | 0.057502 | 0.945871 | 0.009901 | 0.010801 |
| DMN + Vis + VAN + FPN + Lim | 0.907978 | 0.022205 | 0.885773 | 0.009901 | 0.010801 |
| DMN + Som + Vis + VAN + FPN + Lim | 0.888630 | 0.009292 | 0.879338 | 0.009901 | 0.010801 |
| DMN + FPN | 0.647563 | 0.221752 | 0.425811 | 0.009901 | 0.010801 |
| DMN + Som + Vis + VAN + FPN | 0.418866 | 0.000000 | 0.418866 | 0.009901 | 0.010801 |
| DMN + Vis + VAN + FPN + Lim + Sub | 0.969511 | 0.669098 | 0.300413 | 0.009901 | 0.010801 |
| DMN + Vis + VAN + FPN + Sub | 0.308474 | 0.028310 | 0.280164 | 0.009901 | 0.010801 |
| DMN + Vis | 0.247279 | 0.018912 | 0.228368 | 0.009901 | 0.010801 |

模块层面的稳定读法是：高于 null 的 atom 反复包含 DMN、FPN、VAN 和 Visual networks，说明 high-order residual 主要是跨网络结构，而不是单一 ROI 或单一网络内部效应。

Module participation 把每个 atom 的 value 加到它包含的所有模块上。这个指标不要求 LR/RL 在 exact atom label 上完全一致，只检验“哪些脑网络反复参与高阶 residual”。

| Module | Observed mean | Null mean | Difference | Empirical p | FDR q |
|---|---:|---:|---:|---:|---:|
| Vis | 9.360290 | 3.597209 | 5.763081 | 0.009901 | 0.013201 |
| VAN | 7.707466 | 1.981929 | 5.725537 | 0.009901 | 0.013201 |
| FPN | 9.999290 | 5.090643 | 4.908646 | 0.009901 | 0.013201 |
| DMN | 10.390576 | 5.690235 | 4.700342 | 0.009901 | 0.013201 |
| Som | 3.711921 | 1.703675 | 2.008246 | 0.009901 | 0.013201 |
| Lim | 5.623079 | 4.506312 | 1.116766 | 0.009901 | 0.013201 |
| DAN | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| Sub | 3.919978 | 4.995912 | -1.075934 | 1.000000 | 1.000000 |

最强、也最容易解释的模块是 Visual、VAN、FPN 和 DMN。Visual participation 提示高阶 transition structure 包含感知输入相关的低层状态；VAN 提示注意重定向或显著性相关网络参与；FPN 指向前额顶叶控制系统；DMN 说明 resting-state 内在网络不是孤立背景，而是和视觉、注意、控制网络一起进入高阶整合结构。

ROI burden 的稳定读法是：候选贡献集中在 inferior/superior parietal、rostral/superior frontal、lateral occipital、middle temporal 和部分 sensorimotor 区域，主要覆盖 VAN、FPN、Visual 和 DMN/Som 节点。由于 ROI burden 是 leave-one-out score，它仍然不是精确的 83D high-order atom。

| ROI | Module | Observed mean | Null mean | Difference | Empirical p | FDR q |
|---|---|---:|---:|---:|---:|---:|
| ctx-rh-inferiorparietal | VAN | 0.558481 | 0.170939 | 0.387543 | 0.009901 | 0.009901 |
| ctx-lh-inferiorparietal | VAN | 0.546062 | 0.165618 | 0.380443 | 0.009901 | 0.009901 |
| ctx-lh-lateraloccipital | Vis | 0.529049 | 0.165914 | 0.363136 | 0.009901 | 0.009901 |
| ctx-rh-rostralmiddlefrontal | FPN | 0.525255 | 0.180825 | 0.344430 | 0.009901 | 0.009901 |
| ctx-rh-lateraloccipital | Vis | 0.497892 | 0.162754 | 0.335138 | 0.009901 | 0.009901 |
| ctx-lh-superiorfrontal | FPN | 0.500848 | 0.186228 | 0.314620 | 0.009901 | 0.009901 |
| ctx-rh-superiorparietal | VAN | 0.482703 | 0.169515 | 0.313188 | 0.009901 | 0.009901 |
| ctx-lh-rostralmiddlefrontal | FPN | 0.494682 | 0.183535 | 0.311147 | 0.009901 | 0.009901 |
| ctx-lh-superiorparietal | VAN | 0.475582 | 0.166095 | 0.309487 | 0.009901 | 0.009901 |
| ctx-rh-superiorfrontal | FPN | 0.495426 | 0.187717 | 0.307710 | 0.009901 | 0.009901 |
| ctx-lh-supramarginal | VAN | 0.473689 | 0.174533 | 0.299156 | 0.009901 | 0.009901 |
| ctx-lh-middletemporal | DMN | 0.452318 | 0.173729 | 0.278589 | 0.009901 | 0.009901 |
| ctx-lh-postcentral | Som | 0.441847 | 0.170460 | 0.271387 | 0.009901 | 0.009901 |
| ctx-lh-precuneus | Vis | 0.434895 | 0.167127 | 0.267768 | 0.009901 | 0.009901 |
| ctx-rh-middletemporal | DMN | 0.433478 | 0.169790 | 0.263688 | 0.009901 | 0.009901 |

## 5. REST1 vs Working Memory PhiEID 对照

REST1 与 Working Memory 对照使用同一 Lausanne-83 ROI pipeline、Ridge one-step transition model、circular-shift null 和 greedy module atom decomposition。

中文结论是：Working Memory 任务态下，整体 PhiEID 明显高于静息态，但 module atom 的核心网络家族并没有完全换一套。更准确的读法是，WM 在 REST 已经出现的 DMN、Visual、VAN 和 FPN 跨网络结构上进一步增强，并把部分 high-order atom 扩展到 Lim 和 Sub 等更大范围的多网络组合。

### 5.1 Whole-state PhiEID

| Condition | Observed mean | Null mean | Difference mean | Median empirical p | LR/RL r | Top-5 atom overlap |
|---|---:|---:|---:|---:|---:|---:|
| REST1 | 12.530882 | 6.667682 | 5.863200 | 0.009901 | 0.779166 | 0.700000 |
| Working Memory | 24.919033 | 17.976072 | 6.942961 | 0.009901 | 0.439262 | 0.680000 |

从 whole-state 结果看，REST1 的 observed raw PhiEID 均值为 `12.530882`，WM 为 `24.919033`。不过更应该看 observed - null：REST1 为 `5.863200`，WM 为 `6.942961`。这说明 WM 任务下不仅整体动态信息更大，在扣除 ROI-wise circular-shift null 后，仍保留更强的跨 ROI / 跨网络高阶结构。

### 5.2 Module atom distribution

Top-10 atom Jaccard overlap between REST1 and Working Memory is `0.538462`.

| Module atom | REST rank | WM rank | REST difference | WM difference | REST FDR q | WM FDR q |
|---|---:|---:|---:|---:|---:|---:|
| DMN+Vis+VAN+FPN+Lim+Sub | 8 | 1 | 0.300413 | 1.816711 | 0.010801 | 0.016973 |
| DMN+Vis+FPN | 3 | 2 | 0.945871 | 1.716589 | 0.010801 | 0.016973 |
| DMN+Vis+VAN+FPN+Sub | 9 | 3 | 0.280164 | 1.625304 | 0.010801 | 0.016973 |
| DMN+Vis+VAN+FPN | 1 | 5 | 1.526068 | 0.748855 | 0.010801 | 0.016973 |
| DMN+Som+Vis+VAN+FPN+Lim+Sub | 2 | 4 | 1.027408 | 1.336998 | 0.010801 | 0.016973 |
| DMN+Vis+VAN+FPN+Lim | 4 |  | 0.885773 | 0.000000 | 0.010801 |  |
| DMN+Som+Vis+VAN+FPN+Lim | 5 |  | 0.879338 | 0.000000 | 0.010801 |  |
| DMN+Vis+FPN+Lim | 11 | 6 | 0.169639 | 0.596508 | 0.010801 | 0.016973 |
| DMN+Vis | 10 | 7 | 0.228368 | 0.427115 | 0.010801 | 0.016973 |
| DMN+FPN | 6 | 8 | 0.425811 | 0.297159 | 0.010801 | 0.029703 |
| DMN+Som+Vis+VAN+FPN | 7 |  | 0.418866 | 0.000000 | 0.010801 |  |
| DMN+FPN+Sub |  | 11 | 0.000000 | -0.769192 |  | 1.000000 |

REST1 最强 atom 是 `DMN+Vis+VAN+FPN`；WM 最强 atom 是 `DMN+Vis+VAN+FPN+Lim+Sub`。因此，WM 更像是在原来的 DMN-Visual-attention-control 结构上，把任务相关的高阶动态整合扩展到更大范围，而不是产生一个完全无关的新结构。

### 5.3 Module participation

| Module | REST difference | WM difference | WM - REST | REST FDR q | WM FDR q |
|---|---:|---:|---:|---:|---:|
| DMN | 4.700342 | 5.705215 | 1.004873 | 0.013201 | 0.011315 |
| Som | 2.008246 | 1.174670 | -0.833576 | 0.013201 | 0.011315 |
| Vis | 5.763081 | 6.884611 | 1.121530 | 0.013201 | 0.011315 |
| VAN | 5.725537 | 5.669473 | -0.056064 | 0.013201 | 0.011315 |
| DAN | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| FPN | 4.908646 | 5.769435 | 0.860788 | 0.013201 | 0.011315 |
| Lim | 1.116766 | 2.027885 | 0.911119 | 0.013201 | 0.011315 |
| Sub | -1.075934 | 1.799319 | 2.875253 | 1.000000 | 0.011315 |

WM 中 Visual、FPN、DMN、Lim 和 Sub 的 participation 增强。Visual 和 FPN 增强符合视觉刺激输入、规则保持和执行控制；VAN 仍然较高，符合任务中注意重定向和显著性检测；DMN 没有消失也不意外，因为复杂认知任务中 DMN 可以和控制网络动态耦合，而不只是简单地“任务时关闭”。Lim 和 Sub 在 WM 中的增强应谨慎解释为更广泛状态调节、皮层下参与或任务态全局动力学的一部分，目前还不能写成具体边缘系统或皮层下机制。

Som participation 在 WM 中相对 REST 下降：REST difference 为 `2.008246`，WM difference 为 `1.174670`，WM - REST 为 `-0.833576`。这个下降不表示体感运动系统“不参与任务”，而是表示在这个 PhiEID 分解口径下，WM 的 high-order cross-network residual 更集中在视觉-控制-注意-内在/调节网络组合上。

### 5.4 Atom order distribution

| Atom order | REST observed mean | WM observed mean |
|---:|---:|---:|
| 2 | 0.894842 | 1.637446 |
| 3 | 1.003373 | 2.503890 |
| 4 | 1.792642 | 3.385153 |
| 5 | 1.948835 | 3.857060 |
| 6 | 1.858141 | 4.551771 |
| 7 | 2.148013 | 4.946505 |

Treat differences as strongest when the corresponding atom or module is above null after FDR correction and has meaningful LR/RL stability in both conditions. Exact greedy atom labels remain less stable than module participation, so the safest contrast is the network-family shift rather than a single exact atom.

## 6. 方法口径

### 6.1 HCP Gaussian log-det PhiEID

令 $\mathbf{x}_t$ 表示 83 维 ROI 状态。动力学模型拟合一步预测：

$$
\mathbf{x}_t \rightarrow \mathbf{x}_{t+1}.
$$

whole-state PhiEID 使用 Gaussian log-det screening：

$$
\Phi^{EID}
= EI(\mathbf{x}_t;\mathbf{x}_{t+1})
- \sum_i EI(x_t^i;\mathbf{x}_{t+1}).
$$

当前 full 83D 主结果使用 Gaussian log-det，是轻量筛查。full-dimensional TM 对这个 pilot 过重；TM 更适合放在低维模块级复核中。

### 6.2 Circular-shift null

null 使用每个 ROI 独立 circular shift。具体做法是：对每个 ROI 的完整时间序列分别随机平移一个不同的时间偏移量，并在序列末尾循环接回开头。因此，每个 ROI 自身的时间结构不会变，例如均值、方差、频谱和自相关模式基本保留；但不同 ROI 在同一时间点上的对齐关系被打乱。

这个 null 问的是：如果每个脑区都有同样的自身慢波和自相关，但跨脑区同步关系是随机错位的，那么 PhiEID 会有多高？它适合当前问题，因为我们关心的是跨 ROI 的同步、协同和 high-order integration 是否贡献了额外 PhiEID，而不是单个 ROI 自身的时间平滑性或自相关是否足以产生高 PhiEID。

经验 p-value 为

$$
p
= \frac{1 + \#\{\Phi_{\mathrm{null}} \ge \Phi_{\mathrm{obs}}\}}
{1 + N_{\mathrm{null}}}.
$$

表中的 FDR q-value 是对多个并行检验做 false discovery rate 校正后的显著性指标。本文中的 `FDR q` 可以读作经过多重比较校正后的 p-value。

## 7. 解释边界和下一步

当前结果已经形成了比较清楚的证据链，但仍有边界。

- HCP REST1 pilot 样本量只有 10 个 subject，虽然每个 subject 有 LR/RL 两个 run。
- 每个 subject-run 只有 100 个 null，因此最小可达 p-value 是 0.009901；后续若要更强统计分辨率，可以提高到 500 或 1000。
- Ridge 只有 10 / 20 个 subject-run 在 RMSE 上优于 persistence；默认纯 MLP 进一步降到 1 / 20，且 0 / 20 优于 Ridge，说明 transition model 仍需更系统的模型选择。
- 当前没有做 REST2，也没有做 motion、tSNR、mean FC 等混杂控制。
- Module participation 相比 null 显著且 LR/RL 稳定，但 exact greedy atom value vector 相关很低；模块结论应解释为跨网络 atom family / participation，而不是精确 partition。
- ROI burden 是 leave-one-out candidate score，不是 exhaustive high-order decomposition。
- DMF 临界增强结论依赖 continuation 采样和逐 $G$ 标准化口径；independent restart 与 raw-scale 敏感性分析不支持把峰值写成采样协议无关的绝对相变指标。

下一步建议：

1. 增加 `REST2_LR` 和 `REST2_RL`，做更完整的 test-retest。
2. 把 null repetitions 提高到 500 或更多，获得更细的 empirical p-value。
3. 加入 Ridge regularization sweep、MLP 容量/正则 sweep、GRU 或其他 transition model，对预测质量进行系统鲁棒性验证。
4. 报告 motion 与 PhiEID 的相关性，避免把运动混杂解释成整合信息。
5. 在低维模块空间做 TM-based EI 复核。
6. 对 DMF 模块级 greedy 分解做 bootstrap，检查 `DMN+FPN+Sub`、`DMN+Sub` 等嵌套 atom 是否稳定。
7. 如果后续扩展脑区级分解，需要沿用 uniform 干预口径，并先做独立鲁棒性验证。

## 8. 产物索引

### 8.1 DMF 产物

| 文件 | 含义 |
|---|---|
| `fig/dmf_83_region_oracle_phi_eid_main_g11.{png,svg,pdf}` | DMF 83 区 whole-state 主结果曲线图 |
| `fig/dmf_83_region_oracle_no_g_standardization.{png,svg,pdf}` | 取消逐 $G$ 标准化后的 83 区 whole-state $\Phi^{EID}$ 对照 |
| `fig/dmf_83_region_oracle_no_g_standardization_detdeg.{png,svg,pdf}` | 取消逐 $G$ 标准化后的 determinism / degeneracy 分解 |
| `fig/dmf_phi_eid_greedy_decomposition.{png,svg,pdf,npz}` | 模块级 greedy 分解 |
| `fig/dmf_phi_eid_region_local_split_min5_decomposition.{png,npz}` | 83 区局部二分搜索 |
| `fig/part2_dmf_phi_comparison.{png,svg,pdf}` | Uniform 干预口径下的 DMF 综合复现图 |
| `fig/dmf_phi_eid_robustness_longtrace.{png,svg,pdf}` | $\Phi^{EID}$ 多 seed 长 trace 鲁棒性验证图 |
| `fig/dmf_83_region_oracle_phi_eid_robustness.{png,svg,pdf}` | 83 区 Kuramoto-aligned whole-state $\Phi^{EID}$ 完整扫描图 |
| `fig/dmf_module_tm_phi_eid_robustness.{png,svg,pdf}` | 模块级 TM-$\Phi^{EID}$ uniform 干预复验图 |
| `results/dmf_83_region_oracle_phi_eid/` | 83 区 whole-state 复验缓存与 summary |
| `results/dmf_phi_eid_robustness_longtrace/` | 长 trace continuation 验证缓存、曲线表和峰值汇总 |
| `results/dmf_83_region_oracle_no_g_standardization/` | 取消逐 $G$ 标准化的 83 区 whole-state 对照缓存与 summary |
| `results/dmf_phi_eid_robustness/` | 短 trace continuation / independent restart 诊断缓存 |
| `results/dmf_module_tm_phi_eid/` | 模块级 TM-$\Phi^{EID}$ 复验缓存与 summary |

### 8.2 HCP 产物

| 文件 | 含义 |
|---|---|
| `results/hcp_lausanne_phi_eid_pilot/summary.json` | HCP REST1 pilot summary |
| `results/hcp_lausanne_phi_eid_pilot/robustness_summary.json` | HCP REST1 分解鲁棒性 summary |
| `results/hcp_lausanne_phi_eid_pilot/roi_timeseries/*.npz` | Lausanne-83 ROI time series cache |
| `fig/hcp_lausanne_phi_eid_null_comparison.{png,svg,pdf}` | REST1 observed vs null 对照图 |
| `fig/hcp_lausanne_phi_eid_decomposition.{png,svg,pdf}` | REST1 decomposition 图 |
| `fig/hcp_lausanne_phi_eid_robustness.{png,svg,pdf}` | REST1 robustness 图 |
| `docs/log/hcp_lausanne_phi_eid_pilot.md` | REST1 pilot 日志 |
| `docs/log/hcp_lausanne_phi_eid_robustness.md` | REST1 robustness 日志 |
| `docs/log/hcp_lausanne_ridge_mlp_rmse_comparison.md` | Ridge vs pure MLP RMSE 对照日志 |
| `results/hcp_lausanne_ridge_mlp_rmse_comparison.json` | Ridge vs pure MLP RMSE 对照结果 |

