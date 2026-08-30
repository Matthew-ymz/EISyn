# 预测尺度依赖的高阶气候可预测性：观测场协同超边与 UniCM 层级有效信息

## 目录

- [1. 科学问题与主要发现](#1-科学问题与主要发现)
- [2. SLP 实验：协同超边如何随预测尺度重组](#2-slp-实验协同超边如何随预测尺度重组)
  - [2.1 实验设计与 Runge 基准](#21-实验设计与-runge-基准)
  - [2.2 从分散短期联系到可复现的中长期源组合](#22-从分散短期联系到可复现的中长期源组合)
  - [2.3 地球科学含义与可检验假设](#23-地球科学含义与可检验假设)
  - [2.4 尺度凝聚与地理扩张的综合图](#24-尺度凝聚与地理扩张的综合图)
  - [2.5 北极相关超边的尺度变化与角色翻转](#25-北极相关超边的尺度变化与角色翻转)
  - [2.6 前五主成分的精确 Shapley 协同归因](#26-前五主成分的精确-shapley-协同归因)
- [3. UniCM 实验：冻结 Transformer 中的层级有效信息](#3-unicm-实验冻结-transformer-中的层级有效信息)
  - [3.1 可解释性分析口径](#31-可解释性分析口径)
  - [3.2 中期增强的系统级整合有效信息](#32-中期增强的系统级整合有效信息)
  - [3.3 ENSO 空间型态与 IOD 背景构成主导层级](#33-enso-空间型态与-iod-背景构成主导层级)
  - [3.4 路径无关固定模块验证](#34-路径无关固定模块验证)
  - [3.5 目标模态分解：IOD 是中期联合读出的主要接收端](#35-目标模态分解iod-是中期联合读出的主要接收端)
  - [3.6 十一模态的精确 Shapley 协同归因](#36-十一模态的精确-shapley-协同归因)
  - [3.7 Syn 引导的输出校准](#37-从机制读数到预测改进syn-引导的输出校准)
- [4. 综合讨论与解释边界](#4-综合讨论与解释边界)
- [5. 图表与数据索引](#5-图表与数据索引)
- [6. 参考文献](#6-参考文献)
- [附录 A. Runge 节点级指标对照](#附录-a-runge-节点级指标对照)
- [附录 B. 补充数值结果](#附录-b-补充数值结果)
- [附录 C. Runge 估计器阶数稳健性](#附录-c-runge-估计器阶数稳健性)

## 1. 科学问题与主要发现

本文用两个彼此独立的实验检验同一科学问题：气候系统的可预测信息是否依赖多个空间模态的联合状态，以及这种高阶依赖如何随预测尺度变化。第一个实验直接分析 1948—2026 年全球海平面气压（SLP）分量，在 Runge 等人提出的因果网络基准上识别二源协同超边；第二个实验不再从观测场重新拟合动力模型，而是把已经训练完成的 UniCM Transformer 视为冻结的气候转移机制，对其进行最大熵干预和有效信息分解。

两组证据分别回答“观测场中出现了什么尺度依赖结构”和“神经气候模型依靠什么联合信息进行预测”。SLP 实验发现，短期强超边较分散，而中长期结果逐渐集中到以 `No.0/No.1` 为核心的少数源组合；该源对先快速建立近全球目标通道，随后通过招募更多目标区域继续扩展并趋于饱和。不同超边同时呈现早期峰值、中期峰值、长期平台和长期增强四类演化。北极特例进一步显示，北极分量 `No.3` 在短期主要作为联合源参与向外读出，中期经历组合重排，长期头部质量收缩并转为主要作为联合预测目标。前五主成分的精确 Shapley 分解给出一致的整体视角：`No.3` 的协同份额由 $H=1$ 的 `31.9%` 降至 $H=60$ 的 `9.4%`，而 `No.0 + No.1` 的合计份额升至 `69.3%`。UniCM 实验发现，系统级整合有效信息增量 $\Xi$ 在 lead 7—10 形成中期增强，主要涉及 ENSO 空间型态与 IOD 背景。十一模态精确 Shapley 分解进一步表明，`ENSO + IOD + nino12 + nino3 + nino4` 五模态核心的协同份额由 lead 1 的 `46.8%` 升至 lead 8 的 `74.1%`，随后在 lead 24 回落至 `53.9%`。进一步把 target-specific Syn 用作输出校准的正则化先验后，在 2009—2016 年 96 个留出起报月份上，测试 nRMSE 由冻结模型的 `1.087` 降至 `0.981`；相对容量匹配的均匀 ridge，Syn 结构本身贡献 `0.0262` 的额外改善（相对改善 `2.61%`）。

这两项实验使用不同数据对象和不同机制载体，不能相互替代，也不构成同一动力方程下的闭环验证。它们共同支持的窄结论是：气候可预测信息不仅存在于单模态记忆或成对联系中，还存在于依赖预测窗口的联合状态中。

## 2. SLP 实验：协同超边如何随预测尺度重组

### 2.1 实验设计与 Runge 基准

为与 Runge 等人（2015）的状态维数保持一致，本文在 1948—2026 年扩展 SLP 样本上预先固定保留 60 个 Varimax 旋转主成分 [R1]。这里的 60 是为保证基准可比性而预先设定的截断维数；数据预处理、周尺度聚合和后续因果网络构造均沿用同一套固定流程。

在这一 60 分量基底下，与静态节点排序不同，这里把两个源分量到一个目标分量的非加性有效信息增量作为二源协同超边：

$$
\Delta_{2,\mathrm{TM}}(i,j\rightarrow k)
=EI_{\mathrm{TM}}(\{i,j\}\rightarrow k)
-EI_{\mathrm{TM}}(i\rightarrow k)
-EI_{\mathrm{TM}}(j\rightarrow k).
$$

多步读出器、最大熵干预、三阶 transport-map（TM）估计和候选构造见 [Method.md 第 4 节](Method.md#4-runge-多尺度二源超边估计)。每个预测尺度 $H$ 均穷举 `102660` 个跨目标二源候选，从而避免先用离散互信息 shortlist 再做 TM 排名造成的覆盖偏差。正文关注三个层次：具体超边位于哪些区域、相同超边能否跨尺度复现、以及强度随 $H$ 呈现何种连续型态。静态 ACE/ACS 节点指标只作为补充对照保留在附录 A。

### 2.2 从分散短期联系到可复现的中长期源组合

图 1a—c 首先给出空间结构的变化。$H=1$ 时，前三条超边分别为 `No.0 + No.3 → No.37`、`No.0 + No.11 → No.35` 和 `No.1 + No.5 → No.17`，强联系分散在不同源—目标组合之间。到 $H=10$，前三条超边转为 `No.0 + No.1 → No.28`、`No.0 + No.1 → No.32` 和 `No.0 + No.6 → No.32`。在 $H=60$，前十几乎全部围绕 `No.0 + No.1` 展开。这一变化说明，中长期结构不是短期联系的等比例衰减，而是源组合本身发生了重组。

这种重组不是只在前十阈值下出现。图 1d 采用一个直接的判断标准：在给定预测尺度 $H$ 和 top-$K$ 口径下，若源对 $(i,j)$ 至少有一条正协同超边进入全局前 $K$，即存在目标 $k$ 使该超边排名不超过 $K$ 且 $\Delta_{2,\mathrm{TM}}(i,j\rightarrow k;H)>0$，就把该源对记为“有效”。有效源对数就是满足这一条件的不同源对数量，不再使用熵或加权换算。

图中同时报告 $K=50,100,200,500$，用来检查结论是否依赖单一排名阈值。以 top-200 为例，有效源对数从 $H=1$ 的 `173` 降至 $H=10$ 的 `116`、$H=20$ 的 `86` 和 $H=60$ 的 `44`；四种 $K$ 下均呈总体下降，说明强协同超边逐渐由更少的源对贡献。这个“有效”只表示进入当前排名阈值，不等同于通过统计显著性检验。图 1e 进一步显示，`No.0 + No.1` 在 top-200 协同质量中的份额由 $H=1$ 的 `0.6%` 增至 $H=10$ 的 `11.4%`、$H=20$ 的 `18.5%` 和 $H=60$ 的 `25.3%`；从 $H=20$ 开始，全局前十均使用该源对。这里的结论是头部结构发生凝聚，而不是全部弱超边都收敛到同一源对。

连续强度曲线揭示了静态网络无法表达的时间结构（图 1g）。包含北极分量 `No.3` 的 `No.0 + No.3 → No.37` 在 $H=1$ 以 `0.008207` bits 位列全局第一，随后快速回落，到 $H=60$ 仅为 `0.000819` bits，构成短期北极源爆发的直接对照。`No.0 + No.6 → No.32` 在 $H=4$ 达到早期峰值，随后持续下降；`No.0 + No.1 → No.28` 在 $H=7$ 达到中期峰值；`No.0 + No.1 → No.50` 从中期开始维持约 `0.012—0.014` bits 的长期平台；`No.0 + No.1 → No.46` 则从短期低值逐步增强，在 $H=60$ 达到 `0.018027` bits。因此，$H$ 不是统一的衰减参数，而是区分快速调整、阶段性耦合、持续背景态和累积传播候选的重要坐标。图 1g 的纵轴按 Method 式（M.10）统一记为 $Syn^{\mathrm{EID}}$；“TM estimate”说明数值由 transport map 估计，数据缓存中的 `delta2_tm` 只是实现字段名。

附录 C 进一步固定全部干预样本、冻结 MLP rollout、候选全集和后处理，只把估计器从 Gaussian/affine TM 依次替换为二、三、四阶 TM。三个代表尺度的第一名超边在四种估计器下完全一致；第一名强度的跨估计器范围在 $H=1,10,60$ 分别只有 `0.008136—0.008558`、`0.017525—0.018130` 和 `0.018010—0.018320` bits。四条代表曲线相对三阶 TM 的 Pearson 相关均高于 `0.9988`，早期峰值、中期峰值和长期增强的峰位完全保持。由此，正文关于强超边量级和时间型态的结论不依赖三阶 TM；但十万级候选的全排序 Spearman 仅为 `0.281—0.713`，说明弱超边的精细次序仍对估计器敏感。

### 2.3 地球科学含义与可检验假设

SLP 实验把遥相关的比较单位从单条边扩展为“源组合—目标—预测窗口”三元组。同一个区域可以在短期不重要，却在另一个背景源共同存在时成为中长期信息通道；同一源组合也可以对不同目标表现为峰值、平台或持续增强。这种表示更适合描述依赖背景态的大气桥、Rossby 波列和海盆间耦合，而不是把它们压缩为固定的成对连接 [R2-R8]。

`No.0 + No.1` 的目标集合给出了比全局前十平均距离更明确的空间证据（图 1f）。固定全局 top-200 口径后，该源对在 $H=5,10,20,60$ 分别连接 `3/13/20/27` 个目标。每个目标用其分量载荷主中心定位，“最大目标跨度”定义为同一尺度全部目标中心两两球面大圆距离的最大值。该跨度从 $H=5$ 的 `13.91 × 10^3 km` 快速增至 $H=6$ 的 `18.18 × 10^3 km`，在 $H=15$ 接近 `19.92 × 10^3 km` 后基本饱和；与此同时，目标数仍从 $H=15$ 的 `16` 增至 $H=60$ 的 `27`。因此，更准确的空间图景是先快速建立近全球尺度的目标范围，再在既有最大跨度内继续招募和加密目标，而不是传播距离持续线性增长。

图 1 给出四个可检验假设。第一，若早期峰值来自快速大气调整，它应表现出更强的季节依赖，并在 block-bootstrap 中保持邻近尺度的一致性。第二，若中期峰值来自多区域相位关系向目标响应的转化，物理校准后的源区应同时满足稳定的空间载荷和时间先后关系。第三，若 `No.0 + No.1` 的目标扩张对应真实的跨区域传播，则“最大跨度先饱和、目标数后增长”的两阶段型态应在不同 top-$K$、目标中心定义和替代动力模型下保持。第四，长期平台与长期增强应对起始月份、推演误差和替代动力模型表现出不同敏感性。在完成这些检验前，本文只把超边称为候选机制，不将其等同于已确认的物理因果通道。

### 2.4 尺度凝聚与地理扩张的综合图

![SLP 协同超边的跨尺度重组](../../fig/earth_slp_hyperedge_dynamics.png)

*图 1. SLP 协同超边由分散的短期结构凝聚为具有广泛目标覆盖的中长期骨架。a—c，$H=1$、$H=10$ 和 $H=60$ 的全局 TM 前十超边；蓝色节点为源，绿色节点为仅作为目标出现的分量，紫色汇合点及箭头表示二源协同读出，线宽随 $Syn^{\mathrm{EID}}$ 增大。d，top-50、100、200 和 500 中至少贡献一条正协同超边的不同源对数量；“有效”表示进入相应 top-$K$，不是统计显著性判定，四种口径均显示源对数量总体下降。e，top-200 协同质量的源对组成，突出 `No.0 + No.1` 的增长；灰色为其余源对。f，`No.0 + No.1` 在 top-200 中的最大目标跨度和目标数；跨度为全部目标分量主中心之间的最大球面大圆距离，单位为 km。g，五条代表超边的 TM 估计：新增的 `No.0 + No.3 → No.37` 展示北极相关短期爆发，其余四条分别呈现早期峰值、中期峰值、长期平台和长期增强。纵轴符号与 Method 式（M.10）一致。所有尺度均来自完整候选集，而非离散 shortlist。*

### 2.5 北极相关超边的尺度变化与角色翻转

这里把“北极相关”操作化为：至少一个源或目标分量的载荷主中心位于北极圈（$66.5^\circ\mathrm{N}$）以北。在当前 60 分量基底中，只有 `No.3` 满足该条件，其主中心位于 $72.63^\circ\mathrm{E},69.20^\circ\mathrm{N}$；把阈值放宽到 $60^\circ\mathrm{N}$ 仍只选中同一分量。因而，下面把包含 `No.3` 的候选分成两类：`No.3` 属于二源集合时记为“北极作为源”，`No.3` 是读出目标时记为“北极作为目标”。所有 horizon 共享同一 `102660` 个候选、三阶 TM、干预样本和排名规则，唯一变化是 forecast horizon。

图 2b 显示北极相关头部质量不是单调衰减，而是“短期爆发—快速回落—中期再进入—长期收缩”。在最严格的 top-50 中，北极相关质量份额从 $H=1$ 的 `13.8%` 降至 $H=60$ 的 `1.5%`；top-100、200 和 500 的相应变化分别为 `9.5% → 0.9%`、`7.1% → 1.0%` 和 `6.7% → 2.0%`，因此长期衰减不依赖单一 $K$。中期局部回升则依赖头部宽度：top-200 在 $H=15$ 达到 `7.8%`，top-500 在 $H=9$ 达到 `7.0%`，而 top-50 在 $H=15,20,30$ 三个已评估点均没有北极相关超边。这意味着中期北极信号主要位于头部的第二梯队，不能表述为始终占据最强前几十条。

角色分解给出更稳定的方向变化（图 2c）。在 top-200 中，北极作为源的质量份额在 $H=1$ 为 `5.1%`，$H=15$ 为 `5.8%`，到 $H=30$ 降至 `2.8%`；北极作为目标同期为 `2.0%`、`2.1%` 和 `2.7%`，在 $H=30$ 已与源角色近乎平衡。到 $H=60$，北极源超边从 top-200 中完全消失，仍保留的两条北极相关超边均以 `No.3` 为目标，合计占 `1.0%`。由于完整候选集中单个节点作为源的组合机会是作为目标的两倍，图中不把源、目标绝对数量直接解释为方向因果；可解释的是同一候选空间内随 $H$ 出现的配对角色转移。

最强北极相关超边的身份进一步划分出三个阶段（图 2d）。$H=1$ 的 `No.0 + No.3 → No.37` 是全局第一，强度为 `0.008207` bits，代表孤立的短期北极源爆发。$H=5$—20 期间，领先身份在源和目标之间多次切换；其中 `No.18 + No.19 → No.3` 在 7 个已评估 horizon 进入 top-200，而 `No.0 + No.3 → No.48` 在 $H=9$—40 的 6 个已评估 horizon 均进入 top-200，说明中期不是单一路径延寿，而是入北极与出北极组合并存。到 $H=40,50,60$，领先项稳定为 `No.0 + No.13 → No.3`，全局排名依次为 `46/25/26`，强度为 `0.004700/0.005263/0.005109` bits。因而，长期变化更接近“头部参与度下降但剩余入北极通道变得稳定”，而不是所有北极协同同时消失。

![北极相关 SLP 二源协同超边的尺度变化](../../fig/earth_slp_arctic_hyperedge_horizon.png)

*图 2. 北极相关二源协同超边随 forecast horizon 由短期源爆发转为长期弱而稳定的目标读出。a，`No.3` 的符号统一载荷场、载荷主中心与北极圈阈值。b，top-50、100、200 和 500 中所有包含 `No.3` 的正协同质量份额。c，top-200 质量按“北极作为源”和“北极作为目标”分解。d，每个 horizon 最强北极相关超边的全局排名；排名越小越强，点面积随 $\Delta_{2,\mathrm{TM}}$ 增大，颜色区分 `No.3` 的角色。top-$K$ 表示描述性排名阈值，不是显著性筛选。所消费的 top-500 内估计采用 $10^{-10}$ bits 的非负数值容差；最小值为 `0.002397` bits，没有容差内归零值或显著非负性违例。当前结果尚无 block-bootstrap 不确定性，物理解释应限于候选机制。*

### 2.6 前五主成分的精确 Shapley 协同归因

二元超边回答“哪两个源共同指向哪个目标”，但不能给每个 SLP 主成分一个跨全部目标汇总的协同百分比。为与脑区实验的归因口径对齐，这里把前五个主成分 `No.0`—`No.4` 作为五个玩家，把每个 horizon 的全部 60 维未来 SLP 状态 $\mathbf{Y}_H$ 固定为共同 target。对玩家集合 $S$，先计算

$$
F_H(S)=EI_{\mathrm{aff}}\!\left(\mathbf{X}_{S}\rightarrow\mathbf{Y}_H\right),
$$

再定义去除单分量可加贡献后的联盟价值

$$
v_H(S)=F_H(S)-\sum_{i\in S}F_H(\{i\}).
$$

五玩家只有 $2^5=32$ 个联盟，因而无需 Monte Carlo 排列近似，可以精确计算 Shapley 值 $\phi_i(H)$。图中的百分比为 $100\phi_i(H)/v_H(N)$，在每个 horizon 内严格加和为 `100%`。这里分解的是前五分量对**全 60 维未来场的整体非加性读出**，不是图 1—2 中某条二源—单目标超边的再次拆分，因此两种百分比的分母不同。

计算沿用同一批 `4096` 个最大熵干预和冻结的 60 步 MLP rollout。为保持轻量并避免为每个联盟重新推演模型，在每个 horizon 拟合一次 affine degree-1 TM 的线性—高斯等价读出

$$
\mathbf{Y}_H=\mathbf{B}_H^{\mathsf T}\mathbf{X}+\boldsymbol{\epsilon}_H,
\qquad
\operatorname{Cov}(\mathbf{X})=\mathbf{I}_{60},
$$

随后用 log-determinant 解析计算全部联盟 EI。实际干预仍是各分量分位数支撑上的独立 bounded-uniform 采样；上式只是 affine TM 在逐坐标标准化后的高斯协方差等价表示，不把干预分布改成真实高斯。干预设计中的 60 个源坐标按构造相互独立；估计时显式使用单位对角协方差，而不把有限样本中偶然出现的微小源相关当成协同或负协同。主结果的残差协方差 ridge 为 $10^{-6}$，并以 $10^{-8}$ 和 $10^{-4}$ 做敏感性复核。

五分量博弈显示明显的尺度换挡（图 3a—b）。$H=1$ 的总协同为 `0.1807` bits，`No.3` 以 `31.9%` 居首，`No.4` 和 `No.1` 分别占 `26.2%` 与 `24.8%`；这与北极分量在短期强超边中的突出位置一致。到 $H=10$，总协同降至 `0.0548` bits，最大份额转为 `No.4` 的 `33.1%`，五个分量的构成仍持续重排。$H=20$ 以后结构趋于稳定并向 `No.0/No.1` 凝聚；$H=60$ 时总协同为 `0.0360` bits，`No.1` 和 `No.0` 分别占 `44.3%` 与 `25.0%`，合计 `69.3%`，而 `No.3` 降至 `9.4%`。因此，前五主成分的整体归因与二元超边的结论相互支持：短期由北极相关和其他分量共同承担，长期则由 `No.0/No.1` 主导，但它没有把这种统计归因提升为物理方向因果。

为检查“只看前五个分量”是否把其余 55 维背景误归给前五者，另做一个六玩家对照：保留五个单分量玩家，并把 `No.5`—`No.59` 合并为一个 `Others` 玩家，精确枚举 $2^6=64$ 个联盟（图 3c—d）。`Others` 在 $H=1,10,60$ 分别获得 `47.0%/48.1%/44.5%` 的跨块协同份额，说明前五分量并不是封闭系统；但其份额长期只缓慢下降，而前五内部仍由 `No.0/No.1` 接管，故核心尺度换挡不是简单的遗漏变量假象。`Others` 是一个 55 维块，其 Shapley 值衡量该块与五个显式玩家之间的跨块交互，不能解释为任一单独后续主成分的百分比。

![SLP 前五主成分的精确 Shapley 协同归因](../../fig/earth_slp_pc05_shapley.png)

*图 3. 前五个 SLP 主成分对全 60 维未来场协同读出的精确 Shapley 分解。a，五玩家博弈的百分比构成。b，对应的绝对 Shapley 贡献。c，加入 `Others = No.5—No.59` 块后的六玩家百分比。d，两种博弈的 grand-coalition interaction；六玩家值较大主要反映前五分量与其余背景场之间的跨块非加性读出。全部 horizon 使用同一批 `4096` 个独立最大熵干预、冻结 rollout 和 affine TM。$10^{-8}$ bits 非负容差下，最小联盟 interaction 为 `1.29 × 10^{-6}` bits，无容差内归零或显著违例；最大 Shapley 闭合误差为 `4.44 × 10^{-16}` bits。协方差 ridge 从 $10^{-8}$ 扫到 $10^{-4}$ 时，全部 horizon 的五玩家份额最大变化范围为 `4.10` 个百分点，在 $H=1,10,60$ 三个关键尺度内不超过 `2.02` 个百分点。完整 60 维 target 下，二阶 TM 最末级需 `2145` 个基函数、仅有 `1.91` 个样本/基函数，三阶需 `47905` 个基函数并超过样本数；为避免不稳健高阶拟合，未把 target 改成不同科学问题的标量进行替代验证。*

### 2.7 60-PC 层级树：跨尺度保持的核心—外围骨架

下面三棵树把同一 60-PC 系统在 $H=1,10,60$ 的完整贪婪二分过程直接展开。每个内部节点是仍被联合读取的 PC 联盟，节点颜色和标签表示该层的局部 Syn；末端是单个 PC。三个 horizon 采用相同变量、干预支持、TM 估计器和候选划分规则，只有预测时距改变，因此可以直接比较树形重组。

**$H=1$：**

![Earth SLP 60-PC Xi hierarchy at H=1](../../fig/earth_slp_pc60_xi_hierarchy_H001.png)

**$H=10$：**

![Earth SLP 60-PC Xi hierarchy at H=10](../../fig/earth_slp_pc60_xi_hierarchy_H010.png)

**$H=60$：**

![Earth SLP 60-PC Xi hierarchy at H=60](../../fig/earth_slp_pc60_xi_hierarchy_H060.png)

*层级树补充图 E1｜SLP 60-PC 在三个预测尺度上的 $\Xi$ 分解树。$H=1$ 和 $H=10$ 的主干均覆盖 59 个内部划分中的 58 个，主干比例为 `98.3%`；归一化 Colless 不平衡度分别为 `0.9977` 和 `0.9982`。$H=60$ 的 59 次内部划分全部位于同一主干，不平衡度为 `1.0000`。系统 $\Xi$ 从 $H=1$ 的 `11.189` bits 降到 $H=10$ 的 `7.763` bits，并在 $H=60$ 降到 `1.402` bits。*

这些树确实没有呈现“几个规模相近的大分支彼此分开”的平衡模块结构；但这不等于系统没有结构。它们更明确地支持一种**嵌套核心—外围组织**：每一步主要剥离一个外围 PC，剩余大联盟继续作为主干，直到最深层核心。核心成员还随尺度改变：$H=1$ 的最深核心为 `No.1 + No.3`，$H=10$ 为 `No.6 + No.7`，$H=60$ 为 `No.0 + No.1`。所以稳定的是“主干式组织形状”，不是一组跨尺度固定的模块或固定核心。

这一结论仍受表示和搜索口径限制。PC 是全局空间基，不是预定义地理区块；大系统树使用可扩展候选划分而非穷举全部二分。因此，图 E1 可以否定“当前 PC 表示下存在明显平衡分块”的直观读法，却不能证明原始气候场在所有表示下都没有模块。$10^{-8}$ bits 非负容差下，$H=60$ 有一个候选 pair 落在容差内负区间并按数值零处理；最终采用的层级节点全部非负，没有显著违例。

## 3. UniCM 实验：冻结 Transformer 中的层级有效信息

### 3.1 可解释性分析口径

第二个实验分析已经训练完成的 UniCM Transformer。模型参数保持冻结，输入为 11 个气候模态各自 12 个月的历史，目标为未来 1—24 个月的全模态状态。ENSO、nino12、nino3 和 nino4 描述赤道太平洋强度及东西向空间型态；NPMM、SPMM 和 TNA 提供太平洋经向与热带北大西洋背景；IOD、SIOD 和 IOB 描述印度洋盆地和偶极结构；WWV 表示赤道太平洋暖水体积。

在最大熵干预分布下，系统级整合有效信息增量定义为全部模态历史的整体 EI 减去各单模态 EI 之和：

$$
\Xi
=EI(\mathbf{X}_{1:11}\rightarrow\mathbf{Y})
-\sum_{m=1}^{11}EI(X_m\rightarrow\mathbf{Y}).
$$

$\Xi$ 衡量冻结模型中无法由单个模态信息相加解释的联合读出。随后沿贪婪层级树将 $\Xi$ 分解为模块原子 $\xi_C$。该分解用于定位哪些模态集合仍需被联合读取，但依赖当前贪婪路径和数值容差，不代表唯一的高阶 PID 分解。干预口径、Gaussian log-det 估计和层级闭合关系见 [Method.md 第 5—7 节](Method.md)。

![UniCM 的系统级整合有效信息及层级分解](../../fig/earth_unicm_hierarchical_ei.png)

*图 4. UniCM 中的联合信息结构及其预测用途。a，系统级联合增量 $\Xi$ 在 lead 8 达到 `0.184 ± 0.042` bits。b，$\Xi$ 的层级阶数构成；黑线为原子总和。c，lead 8 的主要原子及其源模态成员。d，各预测 target 的 $\Xi_j$；IOD 在 lead 7—10 最突出，但这些 target 项不能相加为系统总量。e，冻结预测及三种输出校准的 ORAS5 测试 nRMSE；四种方法共享相同测试样本，Syn prior 与 uniform ridge 还共享相同的 44 个输入特征和参数量。f，在每个 target—lead 内随机打乱 Syn 权重与源模态的对应关系；200 个重新调参的随机对照均未达到真实 Syn，`P=0.005`。*

### 3.2 中期增强的系统级整合有效信息

整体 EI 和单模态 EI 之和都随预测期增长而下降，但两者的差值并不单调。图 4a 中，$\Xi$ 在 lead 1—5 约为 `0.05—0.07` bits，随后在 lead 7—10 明显增强，并在 lead 8 达到 `0.183958 ± 0.042136` bits；lead 11—24 仍维持约 `0.09—0.15` bits。换言之，模型在短期可以较多依靠各模态自身记忆，而在中期更依赖多个模态的联合状态。

这一结论不等同于单个 lead 的精细排序已经稳定。ENSO 和 IOD 的整体 EI 曲线在 checkpoint 之间保持相似总体形状，但 lead 排序未通过全部 seed 鲁棒性标准。图 4a 因而支持“中期联合增量增强”这一尺度级结论，不支持把相邻月份的微小差异解释为确定的物理跃迁。

### 3.3 ENSO 空间型态与 IOD 背景构成主导层级

层级分解在数值精度内与总 $\Xi$ 闭合，逐 seed/lead 的最大偏差约为 `7.9 × 10^{-5}` bits。图 4b 表明，lead 8 的主要质量来自二至五阶原子，分别约占总量的 `21%`、`22%`、`20%` 和 `18%`；六阶以上主要表现为较小的跨块残差。这说明中期联合增量不是由单一超高阶项垄断，而是由多个低至中阶模块共同构成。

图 4c 进一步定位了这些模块。lead 8 的最大原子为 `ENSO + IOD + nino12 + nino3 + nino4`，贡献 `0.032661` bits，占总 $\Xi$ 的 `17.8%`。排名靠前的其他原子包括 `ENSO + nino12 + nino3`、`ENSO + IOD + nino12 + nino3`、`ENSO + IOD + nino3` 以及 `nino12 + nino3`。共同结构不是“ENSO 加任意外部模态”，而是 ENSO 当前强度、赤道太平洋东西向型态和 IOD 背景的嵌套联合读出。

这一解释与 ENSO diversity 文献一致：单一 ENSO 指数不足以描述事件的空间型态、生命周期和演变路径 [1-5]。在 UniCM 中，nino3、nino4 和 nino12 更适合解释为 ENSO 内部空间结构的不同读数，而非 ENSO 之外的独立强迫。IOD 的出现则表明，印度洋背景可以参与调制模型对 ENSO 中期演变的读取。这里的“参与”指冻结模型中的信息依赖，不自动等同于已识别的真实动力因果方向。

### 3.4 路径无关固定模块验证

自由 greedy 层级只能给出当前 EI 表和拆分准则下的一条分解路径。为检查 ENSO—IOD 结构是否只是路径选择的产物，进一步固定四个嵌套源集合，并直接计算

$$
\Xi_S
=EI(S\rightarrow\mathbf{Y})
-\sum_{m\in S}EI(X_m\rightarrow\mathbf{Y}),
$$

其中 $\mathbf{Y}$ 始终为同一 lead 的全部 11 个未来模态；checkpoint、8,192 个最大熵干预样本、bound 4、预测缓存和 affine degree-1 TM 均保持配对一致，唯一变化是固定源集合。集合依次为 `nino12 + nino3`、`ENSO + nino12 + nino3`、再加入 `nino4`，最后加入 `IOD`。该量是集合自身相对于单模态之和的总联合增量，不是自由 greedy 路径中的局部残差原子。

四个固定集合的跨 checkpoint 均值曲线都在 lead 8 达峰，各 checkpoint 的峰位均落在 lead 7—10。lead 8 时，四个集合的 $\Xi_S$ 依次为 `0.0308 ± 0.0073`、`0.0627 ± 0.0191`、`0.0853 ± 0.0382` 和 `0.1490 ± 0.0383` bits；最后的五模态集合相对于同一 seed 的全系统 $\Xi$ 平均达到 `80.6%`。这些嵌套集合不是互斥原子，因此该比例只表示固定集合相对于全系统联合增量的定位程度，不能跨集合相加。

相邻集合提供了更直接的配对比较。在 lead 7—10 窗口，加入综合 ENSO 指数、nino4 和 IOD 后，$\Xi_S$ 分别增加 `0.0282 ± 0.0179`、`0.0228 ± 0.0187` 和 `0.0525 ± 0.0282` bits；三个新增步骤在全部 `3 checkpoint × 24 lead` 配对单元中均保持正值。IOD 的增量最大且集中在中期，说明冻结模型并非只读取多个 ENSO 指数之间的内部空间差异，还把印度洋背景作为印太联合状态的一部分。更谨慎的地球科学解释是：该结果提出了“IOD 背景对 ENSO 空间型态形成状态门控”的可检验假设；它与延迟的跨海盆耦合相容，但不能单独识别印度洋到太平洋的真实动力方向。

![UniCM 路径无关固定模块协同](../../fig/unicm_fixed_module_xi_leads.png)

*图 5. 路径无关固定集合确认 UniCM 的中期联合读出主要集中在 ENSO 空间型态与 IOD 背景。a，四个固定集合的 $\Xi_S$；粗线为 checkpoint 均值，阴影为标准差，细线为三个 checkpoint。b，固定集合 $\Xi_S$ 相对于同一 seed 全系统 $\Xi$ 的比值后再跨 seed 平均；不同集合相互嵌套，比例不可相加。c，沿嵌套序列依次加入 ENSO、nino4 和 IOD 的配对 $\Delta\Xi_S$；IOD 的最大增量集中在 lead 8—11。d，lead 7—10 的逐 checkpoint 均值和跨 checkpoint 均值横线。全部条件使用相同干预样本、预测缓存、联合 target 和 affine degree-1 TM，且保留有符号结果。*

### 3.5 目标模态分解：IOD 是中期联合读出的主要接收端

联合全模态 target 可以定位输入侧的协同模块，却不能判断联合信息最终主要影响哪个预测量。为此，在 checkpoint、8,192 个最大熵干预样本、12 个月输入历史、bound 4 和 affine degree-1 TM 全部配对固定的条件下，只把 target 从 11 维联合输出依次改为单个未来模态 $Y_j$：

$$
\Xi_j
=EI(\mathbf{X}_{1:11}\rightarrow Y_j)
-\sum_{m=1}^{11}EI(X_m\rightarrow Y_j).
$$

全部 `3 checkpoint × 11 target × 24 lead` 的 $\Xi_j$ 均为正，但强度和时间型态高度不均匀（图 4d）。IOD 是唯一在系统中期峰值窗口显著突出的 target：lead 8 达到 `0.1873 ± 0.0673` bits，lead 7—10 平均为 `0.1635 ± 0.0637` bits。同期第二至第四位依次为 SIOD（`0.0488 ± 0.0316` bits）、nino3（`0.0461 ± 0.0227` bits）和 nino12（`0.0377 ± 0.0069` bits）；综合 ENSO 为 `0.0346 ± 0.0040` bits，WWV 仅为 `0.0134 ± 0.0118` bits。与此相对，nino4 和综合 ENSO 的各自峰值出现在 lead 1，nino3 和 nino12 的峰值出现在 lead 3，说明太平洋 ENSO 空间型态的强联合读出偏早，而 lead 7—10 的系统增强主要落到未来 IOD。

IOD 的特殊性不是“它最容易预测”，而是“它的中期预测最需要联合状态”。lead 8 时，未来 IOD 的整体 EI 为 `0.7168` bits，11 个单模态 EI 之和为 `0.5295` bits，剩余联合增量为 `0.1873` bits；按 checkpoint 分别计算比例后，$\Xi_{\mathrm{IOD}}/EI(\mathbf{X}_{1:11}\rightarrow Y_{\mathrm{IOD}})$ 平均为 `25.7%`。历史 IOD 仍是最大的单源读出，平均贡献 `0.3174` bits，说明模型并没有丢掉 IOD 自身记忆；但仅靠包括 IOD 在内的各单源信息相加，仍不能恢复整体预测。对照最清楚的是 WWV：lead 8 的整体 EI 更高，达到 `1.0212` bits，但其中 `0.9938` bits 已可由 WWV 历史单独读取，$\Xi_{\mathrm{WWV}}$ 只有 `0.0122` bits，约占整体 EI 的 `1.2%`。因此，热图显示的不是一般预测能力，而是每个未来模态对跨模态组合的额外依赖。

这一结构与 IOD 的物理定义和季节性相容。IOD 本身是西、东印度洋海温异常之差，其演变同时涉及印度洋盆地背景、东西向梯度以及与 ENSO—Walker 环流相关的跨海盆状态 [R4, R6, R7]；这些量只有联合读取时才可能形成稳定的未来偶极信号。更重要的是，当前干预缓存固定 `start_month=0`，其月序列与数据加载器一致，从一月开始循环；因而 lead 7—10 对应七月至十月，恰好覆盖 IOD 通常发展并接近成熟的季节。模型的月份嵌入可能在这个窗口把 ENSO 空间型态和印度洋背景组合成更强的 IOD 读出。由此，更准确的假设不是“IOD 单向门控 ENSO”，而是“在 IOD 季节性发展窗口，UniCM 把印太联合状态集中投影到未来 IOD”。

这里仍有两个边界。第一，固定起报月份使预测 lead 与目标季节混合，必须把 `start_month` 循环平移 12 次，才能区分真正的 7—10 个月延迟机制和七月至十月的季节锁相。第二，图 4d 使用全部 11 个历史模态作为源，尚不能证明 IOD 峰值完全由图 5 的五模态核心产生；仍需固定 `ENSO + IOD + nino12 + nino3 + nino4` 并计算 $\Xi_{S\rightarrow j}$。因此，当前结果把 IOD 定位为冻结模型中的主要联合接收端，但不把它等同于已识别的真实双向因果枢纽。

### 3.6 十一模态的精确 Shapley 协同归因

自由 greedy 层级给出哪些模态集合形成主要原子，但不能为全部 11 个模态提供路径无关、严格加和为 `100%` 的系统归因。为此，这里把每个模态的完整 12 个月历史作为一个玩家，把同一 lead 的全部 11 个未来模态固定为共同 target。对模态集合 $S$ 定义

$$
F_\ell(S)
=EI_{\mathrm{aff}}\!\left(\mathbf{X}_S\rightarrow\mathbf{Y}_\ell\right),
\qquad
v_\ell(S)
=F_\ell(S)-\sum_{m\in S}F_\ell(\{m\}).
$$

11 个玩家共有 $2^{11}=2048$ 个联盟，因此可以逐 checkpoint、逐 lead 穷举全部联盟并精确计算 Shapley 值 $\phi_m(\ell)$，不需要排列采样近似。百分比先在每个 checkpoint 内按 $100\phi_m(\ell)/v_\ell(N)$ 归一化，再跨三个 checkpoint 取均值；因而每个 lead 的均值构成仍严格加和为 `100%`。

计算复用同一批 `8192` 个 bound 4 独立最大熵干预、冻结预测缓存和 `start_month=0`。每个玩家包含该模态的 12 个历史坐标，target 始终是 11 维联合未来状态。与 SLP 的精确 Shapley 分解一致，affine degree-1 TM 的等价模型在逐坐标标准化后显式使用已知独立干预的单位源协方差，避免把 132 维有限样本中偶然出现的微小源相关累计成协同。这个先验与干预生成过程一致，但与图 4a 使用经验源协方差的系统 $\Xi$ 不是完全相同的数值口径；因此图 6 的绝对 $v(N)$ 用于显示本分解内部的尺度变化，不与图 4a 的数值逐点等同。

图 6a—b 显示，短期构成较分散，中期明显凝聚到 ENSO 空间型态与 IOD 背景。lead 1 的最大平均单模态份额只有 `12.2%`，而且三个 checkpoint 的首位模态并不一致。到 lead 7，nino3 在三个 checkpoint 中均居首，平均份额为 `19.8%`；lead 8 时 nino3 和 IOD 分别占 `18.5%` 和 `16.7%`。把 `ENSO + IOD + nino12 + nino3 + nino4` 作为图 5 已定位的五模态核心，其合计份额由 lead 1 的 `46.8 ± 8.6%` 升至 lead 7 的 `70.1 ± 5.1%`，并在 lead 8 达到 `74.1 ± 4.9%`。这说明中期系统增量不只是总量增强，而且其模态归因同时向印太核心收缩。

中期内部仍发生角色交接。跨 checkpoint 均值中，lead 9—12 的最大份额依次转为 IOD 的 `17.4%`、`18.0%`、`16.9%` 和 `16.2%`；但单个 checkpoint 的首位排序并不全部一致，因此更稳妥的结论是 nino3 与 IOD 在这一窗口共同突出，而不是 IOD 在每个模型中都确定领先。lead 13—23 的均值首位多数回到 nino3，但优势逐渐缩小；到 lead 24，五模态核心合计份额回落为 `53.9 ± 11.4%`，IOD 与 nino3 的平均份额仅为 `11.3%` 和 `11.2%`。长期变化因而是协同重新分散，而不是由某个单模态永久接管。

绝对贡献给出相同的尺度背景（图 6c—d）。独立先验口径下，grand-coalition interaction 从 lead 1 的 `0.1760 ± 0.0496` bits 升至 lead 8 的 `0.2755 ± 0.0440` bits，随后降至 lead 24 的 `0.1327 ± 0.0249` bits。全部 `3 checkpoint × 24 lead × 2036` 个二阶及以上联盟中，多模态 interaction 的最小值为 `0.000281` bits；在 $10^{-8}$ bits 非负容差下没有容差内负值或显著违例。最大 Shapley 闭合误差为 `1.11 × 10^{-16}` bits。协方差 ridge 从 $10^{-8}$ 扫到 $10^{-4}$ 时，任一模态、任一 lead 的跨 checkpoint 平均份额最大只变化 `0.0094` 个百分点，说明百分比趋势不由 ridge 选择驱动。

![UniCM 十一模态的精确 Shapley 协同归因](../../fig/earth_unicm_11mode_shapley.png)

*图 6. UniCM 十一模态对全模态未来状态整合增量的精确 Shapley 分解。a，三 checkpoint 平均的百分比构成。b，同一百分比的模态—lead 热图；白点标出每个 lead 的均值首位，首位不代表跨 checkpoint 排名一致。c，各模态的平均绝对 Shapley 贡献。d，grand-coalition interaction；灰线为三个 checkpoint，黑线为均值，阴影为标准差。全部条件共享 `8192` 个独立最大熵干预、冻结预测缓存、11 维联合 target 和 affine degree-1 TM，唯一变化是 forecast lead。百分比在 checkpoint 内归一化后再平均。*

### 3.6.1 十一模态的显式分解树

图 E2 把 lead 8 的三个冻结 checkpoint 分别画成树，而不是先平均拓扑。11 个模态足以穷举每个节点的全部无序二分，因此这里没有 SLP 60-PC 大树的候选搜索近似。节点标出该联盟的局部 Syn，三个面板共享颜色尺度。

![Earth UniCM 11-mode Xi hierarchy at lead 8](../../fig/earth_unicm_11mode_xi_hierarchy_lead08.png)

*层级树补充图 E2｜UniCM lead 8 的精确十一模态 $\Xi$ 分解树。三个 checkpoint 的系统 $\Xi$ 分别为 `0.210`、`0.207` 和 `0.135` bits。每棵树都有 9 个内部划分、主干比例 `100%`、归一化 Colless 不平衡度 `1.00`；三棵树均在五模态层收敛到 `{nino, IOD, nino12, nino3, nino4}`，但最深二模态核心分别为 `nino + IOD`、`nino + nino3` 和 `nino12 + nino3`。*

UniCM 与 SLP 的共同点不是“没有模块”，而是都缺少平衡、互不重叠的大分支，并由逐层剥离形成单一主干。不同点在于，UniCM 的五模态印太核心跨三个 checkpoint 完全一致，说明中层核心比最深二元核心更稳定；最深配对仍随 checkpoint 改变。由于这里逐节点穷举全部二分，链形不能归因于候选划分不足。它支持的是**一个稳定中层核心外加不稳定内部排序**，而不是 11 个模态毫无组织或存在若干彼此独立的固定模块。三个 checkpoint 的原子闭合误差均为 0，$10^{-4}$ bits 的既有数值容差下没有负原子或容差内归零值。

### 3.7 从机制读数到预测改进：Syn 引导的输出校准

图 4e—f 回答的是一个比“哪些模态具有高 Syn”更实际的问题：**冻结 Modeformer 已经给出预测后，Syn 能否帮助一个小型输出校准器更可靠地修正预测值？** 校准发生在 Transformer 之后，不改变 Modeformer 的参数，也不重新训练其动力过程。它只使用 ORAS5 的一段历史资料，学习如何把冻结预测映射到更合适的均值和振幅。

#### 校准器读取什么

三个发布 checkpoint 的预测先做等权平均。随后针对每个 target $j$ 和预测 lead $\ell$ 单独拟合一个线性校准器，共有 $11\times24=264$ 个校准单元。每个单元读取 44 个标准化特征：

- 当前 lead 下 11 个模态的冻结预测；
- 11 个模态在 12 个月输入历史中的最后一个值；
- 11 个历史均值；
- 11 个历史线性趋势。

下标的含义如下：

| 符号 | 含义 |
|---|---|
| $t$ | 一个起报时间样本 |
| $j$ | 要校准的目标模态，$j=1,\ldots,11$ |
| $\ell$ | 预测 lead，$\ell=1,\ldots,24$ |
| $m$ | 作为校准输入的源模态，$m=1,\ldots,11$ |
| $\widehat y^{\,F}_{t,m,\ell}$ | 冻结 Modeformer 对模态 $m$、lead $\ell$ 的等权集成预测 |
| $h^{\mathrm{last}}_{t,m}$、$h^{\mathrm{mean}}_{t,m}$、$h^{\mathrm{trend}}_{t,m}$ | 模态 $m$ 的 12 个月历史末值、均值和线性趋势 |

所有输入特征先用拟合段的均值和标准差进行标准化。波浪号表示标准化后的量，例如

$$
\widetilde{x}
=\frac{x-\mu^{\mathrm{fit}}_x}{s^{\mathrm{fit}}_x}.
$$

因此，一个起报样本 $t$ 在 lead $\ell$ 下的 44 维输入向量为

$$
\mathbf{z}_{t,\ell}
=
\left[
\widetilde{\widehat{\mathbf{y}}}^{\,F}_{t,\ell},
\widetilde{\mathbf{h}}^{\mathrm{last}}_t,
\widetilde{\mathbf{h}}^{\mathrm{mean}}_t,
\widetilde{\mathbf{h}}^{\mathrm{trend}}_t
\right]
\in\mathbb{R}^{44}.
$$

这里四个粗体向量都各含 11 个模态。校准输出写为

$$
\widehat{y}^{\,\mathrm{cal}}_{t,j,\ell}
=\beta_{0,j,\ell}
+\mathbf{z}_{t,\ell}^{\mathsf T}\boldsymbol{\beta}_{j,\ell}.
$$

其中，$\beta_{0,j,\ell}$ 是 target $j$、lead $\ell$ 的截距；$\boldsymbol{\beta}_{j,\ell}\in\mathbb{R}^{44}$ 是该校准单元拟合出的 44 个权重。把属于源模态 $m$ 的四个权重记为

$$
\boldsymbol{\beta}_{j,\ell,m}
=
\left[
\beta^{F}_{j,\ell,m},
\beta^{\mathrm{last}}_{j,\ell,m},
\beta^{\mathrm{mean}}_{j,\ell,m},
\beta^{\mathrm{trend}}_{j,\ell,m}
\right]^{\mathsf T},
$$

则上面的点积可以直接展开为

$$
\widehat{y}^{\,\mathrm{cal}}_{t,j,\ell}
=\beta_{0,j,\ell}
+\sum_{m=1}^{11}
\left(
\beta^{F}_{j,\ell,m}\widetilde{\widehat y}^{\,F}_{t,m,\ell}
+\beta^{\mathrm{last}}_{j,\ell,m}\widetilde h^{\mathrm{last}}_{t,m}
+\beta^{\mathrm{mean}}_{j,\ell,m}\widetilde h^{\mathrm{mean}}_{t,m}
+\beta^{\mathrm{trend}}_{j,\ell,m}\widetilde h^{\mathrm{trend}}_{t,m}
\right).
$$

这就是拟合出的 $\boldsymbol{\beta}$ 最终做的事情：在验证或测试样本到来时，把每个固定权重乘到对应的**标准化特征**上，再把 44 项与截距相加，得到 target $j$ 在 lead $\ell$ 的校准预测。

$\boldsymbol{\beta}$ 不是“11 个模态各有一个统一缩放系数”。每个 target—lead 都有自己的一组 44 个权重：

- $\beta^{F}_{j,\ell,j}$ 乘在目标模态自己的冻结预测上，最接近通常所说的缩放系数；
- $\beta^{F}_{j,\ell,m}$（$m\neq j$）把其他模态的冻结预测作为跨模态线性修正；
- 其余三个 $\beta$ 分别决定各模态的历史末值、均值和趋势应怎样修正最终输出；
- 因为输入已经标准化，$\beta$ 表示“特征变化一个拟合段标准差时，校准输出改变多少”，不是直接作用于原始物理量的裸乘数。

只有 univariate 基线近似于简单的

$$
\widehat y^{\,\mathrm{cal}}
=\beta_0+\beta^F\widetilde{\widehat y}^{\,F},
$$

即只对目标自身的冻结预测做截距与斜率修正。Syn prior 使用的是完整的 44 项加权和。这里没有改变 Modeformer 的预测路径；校准器只是根据冻结预测和最近一年的背景状态，对最终数值做一次轻量修正。

#### Syn 在校准的哪个位置起作用

对每个 target—lead 单元，把源模态 $m$ 参与的所有二源 Syn 相加，得到该模态的 Syn 重要性

$$
c_{j,\ell,m}
=\sum_{r\neq m}\operatorname{Syn}_{j,\ell}(m,r).
$$

这里 $r$ 是与源模态 $m$ 组成 Syn 对的另一个源模态。Syn 按定义非负；有限样本估计中接近零的小负数按预先声明的数值容差处理。记原始估计为 $\widehat{\operatorname{Syn}}$，本实验固定 $\delta_{\mathrm{Syn}}=0.002$ bit，并使用

$$
\operatorname{Syn}^{*}=
\begin{cases}
\widehat{\operatorname{Syn}}, & \widehat{\operatorname{Syn}}\ge 0,\\
0, & -\delta_{\mathrm{Syn}}\le \widehat{\operatorname{Syn}}<0,\\
\text{报错并停止}, & \widehat{\operatorname{Syn}}<-\delta_{\mathrm{Syn}}.
\end{cases}
$$

因此，容差内的小负估计只作为数值零，不解释为负协同；显著越过容差的负值则不会进入校准。当前缓存的最小原始值为 `-0.001426` bit，全部 `4,009` 个负估计都位于容差内。上式的 $c_{j,\ell,m}$ 实际对 $\operatorname{Syn}^{*}$ 求和；$c_{j,\ell,m}$ 越大，表示在预测 target $j$ 的 lead $\ell$ 时，模态 $m$ 参与的二源 Syn 总强度越大。校准器通过下面的广义 ridge 目标拟合截距和 44 个 $\beta$：

$$
\left(\widehat{\beta}_{0,j,\ell},
\widehat{\boldsymbol{\beta}}_{j,\ell}\right)
=\arg\min_{\beta_{0,j,\ell},\boldsymbol{\beta}_{j,\ell}}
\left[
\sum_{t\in\mathcal{D}_{\mathrm{fit}}}
\left(
y_{t,j,\ell}
-\beta_{0,j,\ell}
-\mathbf{z}_{t,\ell}^{\mathsf T}\boldsymbol{\beta}_{j,\ell}
\right)^2
+\alpha\sum_{m=1}^{11}
w_{j,\ell,m}
\left\lVert\boldsymbol{\beta}_{j,\ell,m}\right\rVert_2^2
\right],
$$

其中

$$
w_{j,\ell,m}
\propto
\left(c_{j,\ell,m}+\epsilon\right)^{-\gamma},
\qquad
\frac{1}{11}\sum_{m=1}^{11}w_{j,\ell,m}=1.
$$

公式中各量的作用是：

| 符号 | 含义及作用 |
|---|---|
| $y_{t,j,\ell}$ | ORAS5 中真实的 target $j$、lead $\ell$ 目标值 |
| $\mathcal{D}_{\mathrm{fit}}$ | 用于拟合 $\beta$ 的起报样本集合 |
| $\widehat{\beta}_{0,j,\ell}$ | 拟合后的截距，主要承担目标均值修正 |
| $\widehat{\boldsymbol{\beta}}_{j,\ell}$ | 拟合后的 44 个预测权重，最终直接用于计算校准输出 |
| $c_{j,\ell,m}$ | 源模态 $m$ 的 Syn 重要性 |
| $w_{j,\ell,m}$ | 模态 $m$ 的 ridge 惩罚权重；它不直接乘到预测值上 |
| $\alpha$ | 所有系数的总体收缩强度 |
| $\gamma$ | Syn 对不同模态惩罚强弱的区分程度 |
| $\epsilon$ | 防止 $c_{j,\ell,m}=0$ 时权重发散的小量 |

其中 $\bar c_{j,\ell}$ 是 11 个源模态 Syn 重要性的均值。实现中取 $\epsilon=0.05\bar c_{j,\ell}$，避免零 Syn 产生无限惩罚，并把 11 个权重归一化到均值为 1。因此，Syn 较高的源模态具有较小的 $w_{j,\ell,m}$，其四个 $\beta$ 受到较少收缩；Syn 较低的模态受到更多收缩。$\alpha$ 控制整体正则化强度，$\gamma$ 控制不同模态之间的区分程度。若 $\gamma=0$，所有 $w_{j,\ell,m}$ 都相同，模型就退化为 uniform ridge。实验只在验证段选择 $\alpha$ 和 $\gamma$，最终选择为 $\alpha=10000$、$\gamma=2$；测试段从未参与选参。

需要特别区分 $w$ 和 $\beta$：**Syn 生成的 $w$ 只在拟合阶段决定各组 $\beta$ 应该被压缩多少；真正生成校准预测的是拟合完成后的 $\widehat{\beta}_{0,j,\ell}$ 和 $\widehat{\boldsymbol{\beta}}_{j,\ell}$。** 到验证和测试阶段，$w$ 不再直接乘到预测值上。

#### 数据怎样划分，指标怎样计算

校准使用 ORAS5 1980—2014 月资料。按起报时间顺序划分为 253 个拟合样本、36 个验证样本和 48 个测试样本：拟合段截至 2001 年 12 月，验证段为 2004—2006 年，测试段为 2009—2012 年。相邻数据段之间保留与 24 个月预测窗匹配的空档，避免不同数据段共享未来目标月份。

主指标是 nRMSE。先在每个 target—lead 单元内计算测试 RMSE，再除以该模态在拟合段的标准差，最后对全部 264 个单元等权平均。因此，图 4e 中数值越低越好，每个模态和每个 lead 对总指标具有相同权重。

#### 图 4e 的四个点怎样比较

四个方法使用完全相同的测试样本：

- **Frozen**：三个 checkpoint 的等权平均预测，不做校准。
- **Univariate**：每个 target—lead 只使用该 target 自己的冻结预测，学习一个截距和斜率。
- **Uniform ridge**：使用全部 44 个特征，但所有源模态采用相同正则化强度。
- **Syn prior**：与 uniform ridge 使用相同的 44 个特征、相同参数量和相同训练样本，唯一变化是正则化强度按 Syn 分配。

Frozen 的 nRMSE 为 `1.087`，univariate 降至 `0.998`。这说明原始预测中存在可以由简单线性映射修正的均值或振幅偏差；这种降幅主要是数值校准，不等于模型学到了新的气候动力。

Syn prior 的 nRMSE 最低，为 `0.981`。从 Frozen 到 Syn prior 的总降幅为 `0.106`，但这个数同时包含普通校准和 Syn 结构两部分，不能全部归因于 Syn。公平的 Syn 对照是 uniform ridge：两者只有正则化权重不同。Uniform ridge 的 nRMSE 为 `1.007`，因此 Syn-specific 改善为 `0.0262`，相对改善 `2.61%`；12 个月 block-bootstrap 的 95% 区间为 `[0.0005, 0.0514]`。三个独立 checkpoint 上的改善均为正。

与 univariate 相比，Syn prior 的绝对改善为 `0.0167`，95% 区间为 `[0.0061, 0.0303]`。因此，当前最强证据是：**在相同的多模态特征和模型容量下，按 Syn 分配正则化比均匀分配更有效，且在总体平均上也优于单变量校准。**

#### 图 4f 为什么需要随机对照

Syn prior 与 uniform ridge 的比较只能说明 Syn 加权的非均匀正则化优于均匀正则化，不能判断正确的模态对应关系是否优于任意一种非均匀分配。图 4f 因而在每个 target—lead 单元内保留 11 个 Syn 重要性数值，只随机打乱它们对应的源模态标签。这样既保留了权重分布，又破坏了“哪个模态应当少收缩”的结构。随机对照数量在完整运行前固定为 200；每个对照都使用与真实 Syn 相同的特征、参数量、数据划分和超参数搜索预算，并在验证段重新选择 $\alpha$ 与 $\gamma$。

真实 Syn 相对 uniform ridge 的 nRMSE 改善为 `0.0262`；200 个独立打乱并重新调参的随机先验中，没有一个达到真实 Syn。有限随机检验因此为

$$
P=\frac{0+1}{200+1}=0.00498.
$$

这说明图 4e 的额外改善不是“随便给不同模态不同惩罚”就能得到，而依赖 Syn 权重是否分配给正确的源模态。它把图 4a—d 的机制读数推进了一步：Syn 不仅可用于解释冻结模型，还可作为有限样本校准的结构先验。

这一结论仍限定在平均 nRMSE。Syn prior 的平均 ACC 为 `0.274`，略低于 uniform ridge 的 `0.281`；不同 target 的收益也不完全一致。因此，图 4e—f 支持的是“Syn 引导的正则化改善全模态平均幅值误差”，而不是“每个高 Syn 模态都会预测得更准”，也不是已证明的业务预测增益。

## 4. 综合讨论与解释边界

图 1—6 形成一条由观测场到训练模型、再到预测用途的递进证据链。图 1 表明，全球 SLP 中的二源协同超边具有明确的预测尺度结构；图 2 把这一结果收窄到北极分量，显示其头部参与度先重组后收缩，并在长期由源角色转向目标角色。图 3 从超边转向分量级归因，确认前五分量的整体协同份额由短期 `No.3` 突出转为长期 `No.0/No.1` 凝聚，同时揭示其余 55 维背景约承担一半跨块协同。图 4a—d 表明，训练完成的气候 Transformer 在中期更依赖多模态联合读出，并可定位到 ENSO—IOD 嵌套源模块和未来 IOD 接收端；图 4e—f 进一步表明，这些机制读数可以作为输出校准的结构先验，而不是只停留在事后解释。图 5 确认五模态源核心不依赖自由 greedy 路径，图 6 则用精确 Shapley 归因显示该核心的系统份额在 lead 7—10 同步集中、随后重新分散。

这种呼应不应被写成两个实验已经相互验证。SLP 实验分析的是重新拟合的 60 个压力场分量，UniCM 实验分析的是预定义海气模态上的冻结神经网络；两者的变量、时间单位、动力载体和估计维度均不同。更稳妥的结论是，两种独立设置都显示：高阶可预测信息具有尺度选择性，并且需要以“联合状态”而非静态节点重要性来描述。

主要解释边界如下：

- SLP 的 60 个 Varimax 分量是在 1948—2026 扩展样本上重新拟合得到的，编号不是官方固定标签；在完成载荷物理校准前，不能把未校准节点直接命名为确定气候过程。
- SLP 全量穷举解决了 shortlist 覆盖偏差，但尚未进行 block-bootstrap 显著性筛选、季节分层和替代推演模型验证。
- 图 1g 的四条超边是事后选取的代表型态，用于说明时间响应的异质性，不代表全部候选的总体分布。
- 图 2 的北极定义依据分量载荷主中心，只在 $60^\circ\mathrm{N}$ 与北极圈两个中心纬度阈值下得到同一 `No.3`；它没有检验载荷面积阈值、季节依赖或独立分量基底，不能把角色翻转直接解释为真实北极因果方向反转。
- UniCM 的机制分析针对 frozen checkpoint，不做单个历史事件归因。
- Syn 正则化实验表明最大熵机制读数可以作为有限样本校准的结构先验。主结果已覆盖 2009—2016 年的 96 个月度起报日期，但仍只来自一套再分析产品和 affine degree-1 TM，且扩展时段跨越了数据流变化。在独立资料与更高阶估计上复核前，不应把 `2.61%` 的相对 nRMSE 改善外推为稳定业务收益。
- UniCM 的高维 EI、$\Xi$ 和二源 Syn 使用 affine/Gaussian degree-1 TM 等价的 log-det 估计；它适合机制筛查，但不等同于高阶 transport-map PEID 的最终非线性分解。
- 贪婪 $\Xi$ 分解依赖层级路径和数值容差；原子集合不是唯一的高阶 PID 表示。
- 图 3 的 Shapley 百分比依赖 affine TM、独立高斯干预先验和当前 Varimax 基底；它衡量冻结 rollout 的统计归因，不具有旋转不变性，也不能直接解释为某个地理区的物理贡献。完整 60 维 target 的二、三阶 TM 当前受样本—基函数比例限制。
- 固定模块集合由同一批 checkpoint 的 greedy 结果提出，因此图 5 排除了“必须进入自由路径才有信号”，但不是独立 checkpoint 或观测资料上的外部验证；其绝对量仍需 degree-2/3 TM 和干预支撑敏感性复核。
- 图 4d 的标量 target $\Xi_j$ 与联合 target $\Xi$ 使用不同的 readout 维度；$\sum_j\Xi_j$ 不等于联合 target $\Xi$，因此 IOD 的标量值不能解释为系统总量的占比。当前固定起报月份还混合了 lead 与目标季节，其接收端定位需通过 12 个 `start_month` 和固定五模态源集合的 $\Xi_{S\rightarrow j}$ 共同验证。
- 图 6 的 Shapley 百分比依赖 affine TM、已知独立干预先验和固定 `start_month=0`。它与图 4a 的经验源协方差 $\Xi$ 共享冻结预测，但绝对量口径不完全相同；相邻 lead 的单模态首位还存在 checkpoint 不一致，因此当前最稳健的是五模态核心在中期集中、长期回落的块级趋势。

## 5. 图表与数据索引

- 论文主图的可复现脚本：`scripts/plot_earth_system_main_figures.py`
- 图 1 PNG/SVG/PDF：`fig/earth_slp_hyperedge_dynamics.{png,svg,pdf}`
- 图 1 的 `No.0 + No.1` 目标跨度与数量摘要：`fig/earth_slp_hyperedge_dynamics_summary.json`
- 图 2 的分析脚本：`scripts/analyze_runge_arctic_hyperedge_horizon.py`
- 图 2 PNG/SVG/PDF 与摘要：`fig/earth_slp_arctic_hyperedge_horizon.{png,svg,pdf}`、`fig/earth_slp_arctic_hyperedge_horizon_summary.json`
- 图 3 的分析脚本：`scripts/analyze_runge_slp_pc05_shapley.py`
- 图 3 PNG/SVG/PDF 与摘要：`fig/earth_slp_pc05_shapley.{png,svg,pdf}`、`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/slp_pc05_shapley_affine/summary.json`
- 图 4 PNG/SVG/PDF：`fig/earth_unicm_hierarchical_ei.{png,svg,pdf}`；f—g 的预测校准数据来自 `results/unicm_synergy_regularized_forecast_extended_1980_2018/summary.json`
- 图 4f—g 的 target-specific Syn、校准脚本与报告：`results/unicm_target_pair_syn_tm_degree1_signed_n8192/target_pair_syn_summary.csv`、`scripts/run_unicm_synergy_regularized_calibration.py`、`results/unicm_synergy_regularized_forecast_extended_1980_2018/comparison_report.md`
- Runge 周尺度分量输入：`results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv`
- Runge 全候选三阶 TM 结果：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_exhaustive`
- Runge 代表超边强制 TM 趋势：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.csv`
- Runge 多估计器对照脚本：`scripts/compare_runge_tm_estimators.py`
- Runge 多估计器汇总：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_estimator_comparison`
- Runge 多估计器图：`fig/runge_tm_estimator_comparison.{png,svg,pdf}`
- UniCM 系统级 $\Xi$ 逐 seed 结果：`results/unicm_all_mode_target_phi_eid_cpu_bound4_n8192/all_mode_target_phi_eid_rows.csv`
- UniCM 贪婪层级原子：`results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_greedy_atoms.csv`
- UniCM 按阶数汇总：`results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_greedy_order_summary.csv`
- UniCM lead-8 主导原子：`results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_lead8_top_atoms.csv`
- UniCM 路径无关固定模块脚本：`scripts/plot_unicm_fixed_module_xi.py`
- 图 5 UniCM 路径无关固定模块图：`fig/unicm_fixed_module_xi_leads.{png,svg,pdf}`
- UniCM 路径无关固定模块结果与实验合同：`results/unicm_fixed_module_xi_tm_degree1_signed_n8192/`
- UniCM 目标模态分解脚本：`scripts/plot_unicm_target_resolved_xi.py`
- UniCM 目标模态分解独立诊断图：`fig/unicm_target_resolved_xi.{png,svg,pdf}`
- UniCM 目标模态分解结果与实验合同：`results/unicm_target_resolved_xi_tm_degree1_signed_n8192/`
- 图 6 的精确 Shapley 脚本：`scripts/analyze_unicm_11mode_shapley.py`
- 图 6 PNG/SVG/PDF 与摘要：`fig/earth_unicm_11mode_shapley.{png,svg,pdf}`、`results/unicm_11mode_shapley_affine/summary.json`

## 6. 参考文献

[1] Trenberth, K. E., & Stepaniak, D. P. (2001). Indices of El Niño Evolution. *Journal of Climate*, 14(8), 1697-1701. https://doi.org/10.1175/1520-0442(2001)014%3C1697:LIOENO%3E2.0.CO;2

[2] Capotondi, A., Wittenberg, A. T., Newman, M., Di Lorenzo, E., Yu, J.-Y., Braconnot, P., Cole, J., Dewitte, B., Giese, B., Guilyardi, E., Jin, F.-F., Karnauskas, K., Kirtman, B., Lee, T., Schneider, N., Xue, Y., & Yeh, S.-W. (2015). Understanding ENSO Diversity. *Bulletin of the American Meteorological Society*, 96(6), 921-938. https://doi.org/10.1175/BAMS-D-13-00117.1

[3] Ren, H.-L., & Jin, F.-F. (2011). Niño indices for two types of ENSO. *Geophysical Research Letters*, 38, L04704. https://doi.org/10.1029/2010GL046031

[4] Kao, H.-Y., & Yu, J.-Y. (2009). Contrasting Eastern-Pacific and Central-Pacific Types of ENSO. *Journal of Climate*, 22(3), 615-632. https://doi.org/10.1175/2008JCLI2309.1

[5] Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *Journal of Geophysical Research: Oceans*, 112, C11007. https://doi.org/10.1029/2006JC003798

[R1] Runge, J., Petoukhov, V., Donges, J. F., Hlinka, J., Jajcay, N., Vejmelka, M., Hartman, D., Marwan, N., Palus, M., & Kurths, J. (2015). Identifying causal gateways and mediators in complex spatio-temporal systems. *Nature Communications*, 6, 8502. https://doi.org/10.1038/ncomms9502

[R2] Bjerknes, J. (1969). Atmospheric teleconnections from the equatorial Pacific. *Monthly Weather Review*, 97(3), 163-172. https://doi.org/10.1175/1520-0493(1969)097%3C0163:ATFTEP%3E2.3.CO;2

[R3] Hoskins, B. J., & Karoly, D. J. (1981). The steady linear response of a spherical atmosphere to thermal and orographic forcing. *Journal of the Atmospheric Sciences*, 38(6), 1179-1196. https://doi.org/10.1175/1520-0469(1981)038%3C1179:TSLROA%3E2.0.CO;2

[R4] Alexander, M. A., Bladé, I., Newman, M., Lanzante, J. R., Lau, N.-C., & Scott, J. D. (2002). The atmospheric bridge: The influence of ENSO teleconnections on air-sea interaction over the global oceans. *Journal of Climate*, 15(16), 2205-2231. https://doi.org/10.1175/1520-0442(2002)015%3C2205:TABTIO%3E2.0.CO;2

[R5] Neale, R., & Slingo, J. (2003). The Maritime Continent and its role in the global climate: A GCM study. *Journal of Climate*, 16(5), 834-848. https://doi.org/10.1175/1520-0442(2003)016%3C0834:TMCAIR%3E2.0.CO;2

[R6] Ashok, K., Guan, Z., & Yamagata, T. (2001). Impact of the Indian Ocean Dipole on the relationship between the Indian monsoon rainfall and ENSO. *Geophysical Research Letters*, 28(23), 4499-4502. https://doi.org/10.1029/2001GL013294

[R7] Saji, N. H., Goswami, B. N., Vinayachandran, P. N., & Yamagata, T. (1999). A dipole mode in the tropical Indian Ocean. *Nature*, 401, 360-363. https://doi.org/10.1038/43854

[R8] Stuecker, M. F., Timmermann, A., Jin, F.-F., McGregor, S., & Ren, H.-L. (2013). A combination mode of the annual cycle and the El Niño/Southern Oscillation. *Nature Geoscience*, 6, 540-544. https://doi.org/10.1038/ngeo1826

## 附录 A. Runge 节点级指标对照

这一历史对照沿用 Runge 等人 [R1] 的节点级提问：哪些 SLP 分量更接近向外传播扰动的 source hub，哪些分量更容易接收其他区域的影响。对 Runge 的 PC-stable 复现，先对每一对节点取四周内最大绝对因果效应 $C_{i\rightarrow j}^{\max}$，再定义

$$
ACE_i=\frac{1}{N-1}\sum_{j\ne i}C_{i\rightarrow j}^{\max},
\qquad
ACS_j=\frac{1}{N-1}\sum_{i\ne j}C_{i\rightarrow j}^{\max}.
$$

Ridge+PEID 对照把同一节点的一阶 EI 与显著二阶交互压缩为一个静态合成分数。记 $e_{i\rightarrow k}$ 为一阶 EI，$\Delta^{(2)}_{ij\rightarrow k}$ 为历史流程输出的二阶 Möbius 交互，$\mathcal H_i^{\mathrm{out}}$ 和 $\mathcal H_k^{\mathrm{in}}$ 分别为通过 $|z|\ge2$ 门控、且包含源 $i$ 或指向目标 $k$ 的二阶候选集合，则

$$
\begin{aligned}
HACE_i
&=\frac{1}{N-1}\sum_{k\ne i}|e_{i\rightarrow k}|
+\frac{1}{|\mathcal H_i^{\mathrm{out}}|}
\sum_{(i,j)\rightarrow k\in\mathcal H_i^{\mathrm{out}}}
\frac{|\Delta^{(2)}_{ij\rightarrow k}|}{2},\\
HACS_k
&=\frac{1}{N-1}\sum_{i\ne k}|e_{i\rightarrow k}|
+\frac{1}{|\mathcal H_k^{\mathrm{in}}|}
\sum_{(i,j)\rightarrow k\in\mathcal H_k^{\mathrm{in}}}
|\Delta^{(2)}_{ij\rightarrow k}|.
\end{aligned}
$$

若相应二阶集合为空，第二项记为零。除以 2 是把一条二源交互等分给两个源节点；按每个节点实际通过门控的二阶项数取均值，避免候选数量本身直接决定节点分数。该历史实验使用 1948—2026 年周尺度 SLP、60 个 Varimax 分量、4 周输入滞后、一步预测、4,096 个最大熵干预样本和固定随机种子。二阶候选由前 14 个源与每个源的前 10 个目标构造，共得到 `1,638` 项；26 周 block null、20 次重复下有 `287` 项满足 $|z|\ge2$。

![Runge 节点级 ACE/ACS 与 Ridge+PEID 对照](../../fig/runge_node_ace_acs_comparison_1948_2026.png)

*图 A1. Runge 节点级指标与一、二阶合成指标的空间对照。a，修正父节点选择后的 Runge 2015 PC-stable ACE/ACS；b，Ridge+PEID 的一阶 EI 与显著二阶交互合成结果。外圈表示 source-side ACE/HACE，内圆表示 target-side ACS/HACS。两个面板估计对象和数值尺度不同，分别使用色标；b 中最大 HACE 超过稳健色标上限，以色标右端箭头标记。该图只比较节点排序和空间分布，不比较绝对数值。*

修正后的 Runge 复现中，ACE top-5 为 `No.1/0/16/8/26`，ACS top-5 为 `No.0/1/26/4/11`。一、二阶合成结果中，HACE top-5 为 `No.0/1/3/9/4`，HACS top-5 为 `No.10/3/26/0/1`。两种口径的源侧排序仍有较强整体一致性：60 个节点的 ACE–HACE Spearman 相关为 `0.772`，top-5 共同包含 `No.0/1`；目标侧一致性明显较弱，ACS–HACS Spearman 相关为 `0.389`，top-5 共同包含 `No.0/1/26`。因此，高阶合成并未整体推翻 Runge 的主要源节点结构，但显著改变了接收端 hub 的优先级。

`No.3` 最能说明两种问题设定的差异。在修正后的 Runge 图中，它的 ACE/ACS 只排第 `12/13`；在合成图中则升至 HACE/HACS 第 `3/2`。这不是“Runge 结论被否定”，而是说明一个节点可以在成对路径平均效应中并不突出，却频繁参与强二源联合读出。反过来，Runge 的 ACE/ACS 沿多步线性因果路径聚合，而合成分数只汇总直接的一阶读出和二阶源组合，两者不能互换为同一个因果中心性定义。

还需保留一个历史口径限制：二阶缓存把 $\Delta^{(2)}$ 保存为有符号 Möbius 交互，门控后的 287 项中有 13 项为负，最小值为 `-6.65\times10^{-4}` bits，且其 $|z|$ 超过 2。当前项目把 PEID Syn 定义为非负，因此这些负值不能解释为“负 Syn”，也不能通过静默截断修正。本附录仅按当时明确记录的绝对交互合成规则复现节点排序，并把它定位为历史诊断；正文的非负 Syn 结论不依赖该结果。若未来要把节点合成分数升级为正式 PEID 指标，需要使用满足非负约束的估计与容差审计重新计算。

## 附录 B. 补充数值结果

### B.1 Runge 三个代表尺度的前五超边

| 预测尺度 $H$ | 排名 | 超边 | $\Delta_{2,\mathrm{TM}}$ | 联合 EI | 单源 EI 之和 |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | `0+3→37` | 0.008207 | 0.161648 | 0.153441 |
| 1 | 2 | `0+11→35` | 0.006698 | 0.117270 | 0.110573 |
| 1 | 3 | `1+5→17` | 0.005681 | 0.147893 | 0.142213 |
| 1 | 4 | `0+12→37` | 0.005568 | 0.137311 | 0.131743 |
| 1 | 5 | `15+48→2` | 0.005274 | 0.100081 | 0.094807 |
| 10 | 1 | `0+1→28` | 0.017747 | 0.228734 | 0.210987 |
| 10 | 2 | `0+1→32` | 0.012679 | 0.206514 | 0.193835 |
| 10 | 3 | `0+6→32` | 0.010952 | 0.184992 | 0.174040 |
| 10 | 4 | `0+1→50` | 0.010754 | 0.180583 | 0.169829 |
| 10 | 5 | `0+1→55` | 0.010648 | 0.178373 | 0.167724 |
| 60 | 1 | `0+1→46` | 0.018027 | 0.231307 | 0.213280 |
| 60 | 2 | `0+1→30` | 0.014308 | 0.221244 | 0.206936 |
| 60 | 3 | `0+1→50` | 0.013515 | 0.200558 | 0.187043 |
| 60 | 4 | `0+1→41` | 0.012916 | 0.195218 | 0.182302 |
| 60 | 5 | `0+1→34` | 0.012818 | 0.195943 | 0.183124 |

### B.2 UniCM 低阶辅助证据

ENSO 自身历史在短 lead 占主导，排除自身后，nino3、nino12、IOD 和 NPMM 在中长期提供较小补充。二源 Syn 的平均量级约为 `10^{-3}—10^{-2}` bits，显著低于单模态 self EI，因此它更适合作为“联合读出相对于单源信息的额外增量”，而不应与模态自身记忆直接比较。

ENSO 目标中，`ENSO + nino3` 的平均 Syn 为 `0.005216` bits，`ENSO + nino4` 为 `0.005194` bits，`ENSO + IOD` 为 `0.004278` bits。IOD 目标中，`IOD + SIOD` 的平均 Syn 为 `0.012107` bits，`ENSO + IOD` 为 `0.007147` bits，`IOD + nino4` 为 `0.005648` bits。多数曲线在 lead 15 后趋近于零，且部分组合的 seed 标准差接近均值，因此这些结果只用于支持空间型态和背景态的解释，不用于建立稳定的二源排名。

## 附录 C. Runge 估计器阶数稳健性

该对照只改变连续 EI 估计器：degree 1 是 Gaussian/affine TM，即只保留协方差与线性条件均值；degree 2—4 依次加入二、三、四阶多项式条件结构。四个条件共享同一组 4,096 个最大熵干预样本、同一冻结 MLP ensemble rollout、同一 source/target、同一预测尺度、同一 `102660` 个候选全集，以及“各 EI 先截断到非负，再计算联合 EI 减单源 EI 之和”的后处理。全候选比较在 $H=1,10,60$ 上执行；四条正文代表超边则在全部 16 个尺度上配对重估。

![Runge 不同 TM 阶数的配对稳健性](../../fig/runge_tm_estimator_comparison.png)

*图 C1. Runge 强超边的量级与时间型态对估计器阶数稳健，但弱候选的细粒度排序更敏感。a—d，正文四条代表超边在 Gaussian/affine TM 与二至四阶 TM 下的配对强度曲线；e，全部 `102660` 个候选相对正文三阶 TM 的 Spearman 排序相关；f，各估计器前十与三阶 TM 前十的集合重合率。全部条件使用相同干预和 rollout，估计器是唯一处理因素。*

三个尺度的第一名身份完全不变：$H=1$ 均为 `0+3→37`，$H=10$ 均为 `0+1→28`，$H=60$ 均为 `0+1→46`。以三阶 TM 为基准，第一名强度的最大相对跨度从短期的 `5.14%` 降到中期的 `3.40%` 和长期的 `1.72%`。前十集合也较稳定：Gaussian、二阶和四阶 TM 相对三阶 TM 的重合率在 $H=1$ 为 `0.9/0.8/0.6`，在 $H=10$ 为 `0.9/0.9/1.0`，在 $H=60$ 为 `1.0/1.0/0.9`。

代表曲线提供了比单点排名更强的稳定性证据。相对三阶 TM，Gaussian、二阶和四阶 TM 在 64 个“超边 × 尺度”单元上的 Pearson 相关分别为 `0.99889`、`0.99899` 和 `0.99889`，配对绝对差中位数分别为 `0.000264`、`0.000187` 和 `0.000382` bits。`0+6→32` 的早期峰值在四种估计器下均位于 $H=4$，`0+1→28` 的中期峰值均位于 $H=7$，`0+1→46` 的长期增强均在 $H=60$ 达到最大。长期平台 `0+1→50` 的数值形状保持，但 Gaussian/二阶 TM 的离散最大值位于 $H=50$，三/四阶位于 $H=60$；由于 $H=20—60$ 的差值很小，这应解释为平台内部的轻微峰位漂移，而不是趋势反转。

需要区分强信号稳定性与全排序稳定性。Gaussian、二阶和四阶 TM 相对三阶 TM 的全候选 Spearman 在三个尺度分别落在 `0.281—0.342`、`0.593—0.630` 和 `0.697—0.713`；与此同时，各阶数判断为正值的候选比例从 Gaussian 的约 `0.56` 增至四阶 TM 的约 `1.00`。这说明增加多项式阶数会系统性抬升大量接近零的弱候选，并改变它们的次序。因而正文可以稳健陈述第一梯队超边及其时间型态，但不应把全体弱超边的精细排名或“正值候选比例”解释为稳定的物理结构；后者仍需 block-bootstrap、独立样本和 estimator-specific null 校准。

后续可在其余设置不变的条件下改变保留主成分数量，以检验主要超边结构对 PCA 截断维数的敏感性。
