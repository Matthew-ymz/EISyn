# 脑科学实验：83 ROI 临界识别与 HCP Yeo7 Phi 分解

## 结论

本报告保留三个互补结果块。

1. **83 ROI 临界相变识别。**在 83 区 DMF 的近似复现中，经验滞后分布上的 pairwise $\Phi^R$ 在 $G=1.8$ 达峰（$0.02168$），与平均发放率快速上升区重合，复现了原文的散点式相变识别关系。作为互补的机制分析，无裁剪 8-seed 最大熵干预在相近区域识别到 $\Phi^{EID}$ 宽峰：$G=1.7$ 的均值为 $12.384\pm0.041$ bits，$G=1.6\text{-}1.8$ 的临界窗显著高于前、后窗口。Yeo-7 跟进分解显示临界窗中 68.7% 的 $\Phi^{EID}$ 来自跨 ROI 协同，31.3% 来自单个 ROI 内的兴奋—抑制协同。
2. **HCP500/1000 PCA–Yeo7 Phi 分解。**在相同 30 名 HCP REST1_LR 被试中，Schaefer-500（$p=8,\alpha=10$）与重新验证的 Schaefer-1000（$p=5,\alpha=1$）均在 30/30 名被试中高于独立 PC1 circular-shift null。两种粒度的全七网络核都常进入 top-3，但不高于 matched null；相对地，缺少 Limbic 的六网络核均高于 matched null cohort（500：17/30 对 8.65/30；1000：12/30 对 6.35/30；各 20-null 未校正 $p=0.047619$）。
3. **HCP500 任务态迁移。**冻结静息态 Schaefer-500 的 $p=8,\alpha=10$，但对每个任务和被试重新拟合 PCA、标准化、Ridge 系数与残差协方差后，29 名共同被试的 REST raw history-source $\Phi^{EID}$ 均值为 6.196 bits，高于七个任务的全部群体均值；七项 task-minus-REST 配对差在 Holm 校正后均显著，八条件 Friedman $p=1.50\times10^{-8}$。REST 在 20/29 人中为八条件最大值，因此该结论是群体层面的条件排序，而非每名被试都成立。把 REST 严格截成与各任务相同的长度并在每个窗口内重新拟合后，REST 的跨被试方差只在 EMOTION 与 MOTOR 上得到区间整体高于 1 的稳定证据，说明原始高方差不能统一归因于单一长度因素。WM 内部的 20-null 分析进一步显示 30/30 人高于自身 null mean；缺 Limbic 的六网络核在 WM 中为 26/30，并由共同被试静息态的 16/29 增至 WM 的 25/29（Holm $p=0.023438$）。

这些实验分别回答不同问题：DMF 实验检验 $\Phi^{EID}$ 是否能定位可控模型中的临界动力学带；HCP 静息态实验检验降维后的真实网络动力学中是否存在高于同步破坏 null 的跨网络高阶结构；WM 实验检验同一方法能否迁移到任务态并识别协同核频率变化。它们不构成对特定脑机制、因果方向或精确稀疏 atom 的证明。

## 目录

1. [**83 ROI DMF：临界相变识别**](#dmf-critical)
   1. [全状态最大熵干预与临界判据](#dmf-intervention)
   2. [临界窗中的 Φ^EID 峰与 EI 形状对齐](#dmf-phi)
   3. [Φ^R 的相变复现与经验分布边界](#dmf-phi-r)
   4. [临界窗 Φ^EID 的 ROI—结构模块层级分解](#dmf-hierarchy)
   5. [Yeo-7 功能网络层级分解](#dmf-yeo7)
   6. [确定性 / 简并性分解与变化率](#dmf-detdeg)
   7. [解释边界](#dmf-limits)
2. [**HCP500 PCA–Yeo7：Phi 分解**](#hcp500)
   1. [数据、降维与动力学表征](#hcp500-data)
   2. [History-source Φ^EID 与 circular-shift null](#hcp500-phi)
   3. [Yeo7 模块历史分解](#hcp500-module)
3. [**HCP1000 PCA–Yeo7：Phi 分解**](#hcp1000)
   1. [数据、降维与模型选择](#hcp1000-data)
   2. [History-source Φ^EID 与 circular-shift null](#hcp1000-phi)
   3. [Yeo7 模块历史分解与 500 对照](#hcp1000-module)
4. [**HCP500 WM_LR：任务态 Phi 与协同核**](#hcp-wm)
   1. [整体 Phi 与 circular-shift null](#hcp-wm-phi)
   2. [协同核分布及静息态对照](#hcp-wm-module)
   3. [七任务 raw Phi 排序](#hcp-all-tasks)
5. [**讨论：解释边界与可复现性**](#discussion)
   1. [结论的适用范围](#discussion-scope)
   2. [结果与图形产物](#discussion-artifacts)
6. [**附录 A：Kuramoto 振子数与 whole-state Φ^EID 曲线形状**](#appendix-a)
   1. [临界峰的 EI 与 effectiveness 机制](#appendix-a-1)
   2. [时间窗、相变前检测与系统规模边界](#appendix-a-2)
7. [**附录 B：83 ROI DMF 动力学方程**](#appendix-b)
8. [**附录 C：Φ^R 的相变近似复现**](#appendix-c)
9. [**附录 D：83 ROI 对平均 Φ^EID 曲线**](#appendix-d)

<a id="dmf-critical"></a>

## 1. 83 ROI DMF：临界相变识别

<a id="dmf-intervention"></a>

### 1.1 全状态最大熵干预与临界判据

该实验使用 83 区 DMF、原始结构连接、直接长程兴奋耦合和 JFIC。每个试验同时干预全部兴奋性与抑制性门控变量，并将完整 166 维未来状态作为 target：

$$
\mathbf{s}_E,\mathbf{s}_I\overset{\mathrm{ind}}{\sim}U(0.30,0.70)^{83}.
$$

该分布是声明的生理支持 $[0.30,0.70]^{166}$ 内的最大熵分布；初态和 300 个 Euler 步的演化均不裁剪。主确认在 8 个独立种子、2048 个干预样本下评估 $G=1.0,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.2,3.0$。Gaussian block conditional-total-correlation 用于 166 维连续 EI；高维 transport map 在此样本规模下计算代价过高。

临界位置以可比 Kuramoto 全状态参照的理论 $K_c=1.5958$ 定义，而不再以旧的放电率阴影区替代。主判据是 $\Phi^{EID}$ 在该临界点附近形成内部峰/平台，并在高耦合端回落。

<a id="dmf-phi"></a>

### 1.2 临界窗中的 $\Phi^{EID}$ 峰与 EI 形状对齐

$$
\Phi^{EID}=EI_{\mathrm{do}}(\mathbf{s}_t;\mathbf{s}_{t+\tau})-
\sum_{i=1}^{166}EI_{\mathrm{do}}(s_{t,i};\mathbf{s}_{t+\tau}).
$$

$\Phi^{EID}$ 在平均发放率曲线开始快速抬升的 $G=1.6\text{-}1.8$ 形成宽峰（图 1A 右轴），最高均值位于 $G=1.7$：$12.384\pm0.041$ bits。它相对 $G=1.6$ 高 $0.106$ bits（配对单侧 $p=0.039$，7/8 个种子为正）；临界窗均值同时高于前、后窗口（两项 $p<1.2\times10^{-10}$，均为 8/8 个种子同向）。到 $G=3.0$，$\Phi^{EID}$ 降至 $8.144\pm0.022$ bits。平均发放率（图 1A 左轴）在这里是与参考文献 Fig. 6 对齐的序参量代理；它给出动力学转折的外部参照，$\Phi^{EID}$ 则检验该转折附近是否出现联合机制相对各区域机制之和的额外优势。

<a id="dmf-phi-r"></a>

### 1.3 临界窗 $\Phi^{EID}$ 的 ROI—结构模块层级分解

为定位临界窗中的协同来源，这里严格复用主确认的 $G\in\{1.6,1.7,1.8\}$、8 个 seed、每个条件 2048 个独立 $U(0.30,0.70)^{166}$ 干预样本、300 步未来、无裁剪和同一 Gaussian conditional-total-correlation 估计器。每个 ROI 的兴奋性与抑制性门控变量先绑定为一个二元 source block；随后根据经验结构连接代理矩阵进行固定 seed 的加权 Louvain 划分，得到 4 个非平凡结构模块和 2 个零结构强度的单节点模块。该分区来自邻接矩阵本身，不使用脑区名称或预设功能网络标签。

相变扫描使用的是全系统的 $\Phi^{EID}$，而不是任一 ROI 指标。对每个耦合强度 $G$，源变量是独立干预的当前 E/I 全状态

$$
\mathbf{s}_t=
(E_{1,t},I_{1,t},\ldots,E_{83,t},I_{83,t})^\mathsf{T},
$$

目标变量是同一 83 个 ROI 在 300 步后的完整 E/I 状态

$$
\mathbf{y}_{t+\tau}=
(E_{1,t+\tau},I_{1,t+\tau},\ldots,E_{83,t+\tau},I_{83,t+\tau})^\mathsf{T}.
$$

由于 166 个源变量在干预分布下彼此独立，主实验的系统级量为

$$
\begin{aligned}
\Phi^{EID}_{\mathrm{system}}(G)
&= I(\mathbf{s}_t;\mathbf{y}_{t+\tau})
-\sum_{r=1}^{83}\left[
I(E_{r,t};\mathbf{y}_{t+\tau})
+I(I_{r,t};\mathbf{y}_{t+\tau})
\right] \\
&= \sum_{r=1}^{83}\left[
H(E_{r,t}\mid\mathbf{y}_{t+\tau})
+H(I_{r,t}\mid\mathbf{y}_{t+\tau})
\right]
-H(\mathbf{s}_t\mid\mathbf{y}_{t+\tau}).
\end{aligned}
$$

即，它衡量所有当前 E/I 源变量对完整未来 E/I 状态的联合效应中，不能分配给任意单个源变量的部分。用该曲线识别临界窗：当前直接干预条件下，跨 seed 平均的 $\Phi^{EID}_{\mathrm{system}}(G)$ 在 $G=1.7$ 达到 $12.384$ bits 的峰值，因此后续 ROI 分析只在其相邻临界窗 $G\in\{1.6,1.7,1.8\}$ 内进行。

PEID 的层级可加性允许把最细 166-source $\Phi^{EID}$ 精确写成

$$
\Phi^{EID}_{166}
=\sum_{i=1}^{83}\Phi^{EID}_{\mathrm{within\ ROI}_i}
+\sum_{m=1}^{M}\Phi^{EID}_{\mathrm{within\ module}_m}
+\Phi^{EID}_{\mathrm{between\ modules}}.
$$

第一项是每个 ROI 内 $s_E$ 与 $s_I$ 的条件协同；第二项是同一结构模块内不同 ROI 之间的条件协同；第三项是结构模块之间的条件协同。复算的 $\Phi^{EID}$ 与主确认最大绝对误差为 $1.8\times10^{-12}$ bits，层级恒等式最大误差为 $1.8\times10^{-13}$ bits。

令 $\mathbf{s}_{r,t}=(E_{r,t},I_{r,t})^\mathsf{T}$ 为 ROI $r$ 的二元 E/I 源块，$\mathbf{s}_{-r,t}$ 为其余 82 个 ROI 的 164 个 E/I 源变量。先按 ROI 块而不是单个 E/I 标量分区，则有

$$
\Phi^{EID}_{\mathrm{system}}
=\sum_{r=1}^{83}\underbrace{I(E_{r,t};I_{r,t}\mid\mathbf{y}_{t+\tau})}_{\phi^{EID}_{r,E/I}}
+\underbrace{\left[
\sum_{r=1}^{83}H(\mathbf{s}_{r,t}\mid\mathbf{y}_{t+\tau})
-H(\mathbf{s}_t\mid\mathbf{y}_{t+\tau})
\right]}_{\Phi^{EID}_{\mathrm{between\ ROI}}}.
$$

图 1D 的左轴检验单个 ROI 的局部 E/I 项

$$
\phi^{EID}_{r,E/I}
= I(E_{r,t}; I_{r,t}\mid\mathbf{y}_{t+\tau}).
$$

这里只有两个源变量，因此条件 total correlation 正好等于条件互信息。它刻画的是：在控制整个未来 E/I 全状态后，该 ROI 的 $E/I$ 对仍保留多少不能分开解释的联合结构。它不是全脑 $\Phi^{EID}$，也不含该 ROI 与其他 ROI 的跨区项。

相对地，ROI 与其余全脑的跨区指标是

$$
\phi^{EID}_{r,\mathrm{cross}}
=I(\mathbf{s}_{r,t};\mathbf{s}_{-r,t}\mid\mathbf{y}_{t+\tau})
=\Phi^{EID}_{\mathrm{between\ ROI}}(\mathcal{R})
-\Phi^{EID}_{\mathrm{between\ ROI}}(\mathcal{R}\setminus\{r\}),
$$

其中 $\mathcal{R}=\{1,\ldots,83\}$。该量表示移除 ROI $r$ 后，ROI 块间部分下降多少；它是系统级 $\Phi^{EID}$ 的跨区部分的留一归因，不是另一次全系统相变扫描，也不能在 ROI 上相加，因为同一跨区联合结构可被多个 ROI 同时计入。局部项和跨区项在 8 个 seed、3 个临界 $G$ 条件上平均，共 24 个条件。

跨 83 个 ROI，局部 E/I 条件耦合与加权结构强度呈强负秩相关（图 1D 左轴，Spearman $\rho=-0.753$，双侧 $p=2.09\times10^{-16}$）；ROI 与其余全脑的跨区条件耦合则呈近乎单调的正相关（图 1D 右轴，$\rho=0.988$，双侧 $p=3.76\times10^{-68}$）。两组散点因此给出同一结构梯度下的互补模式：结构连接更强的 ROI 具有较弱的局部 E/I 联合结构，却具有更强的、与全脑其余 ROI 的联合结构。相关系数及显著性只在正文报告，不叠加到图内。

从脑科学角度，这一模式与“局部回路—分布式整合”的模型内分工一致。结构强的 ROI 接收和传播更多长程结构输入，其状态变化更受网络其余部分约束，因此在给定完整未来状态后，剩余的局部 E/I 联合结构较小；相反，它们的 E/I 块与全脑其他 ROI 的共同结构更大。结构较弱的 ROI 则呈现相反的相对模式，即更突出的局部 E/I 项和较弱的跨区项。

这不是严格的守恒或因果 trade-off：$\phi^{EID}_{r,\mathrm{cross}}$ 是可重叠的留一归因，跨 ROI 项不能对 ROI 简单求和；两种相关也都基于同一张结构网络。$E/I$ 是 DMF 门控状态而非体内直接记录的细胞群活动，且结构连接来自当前的代理矩阵。因此图 1 支持模型中“局部 E/I 回路与全脑整合的相对重心随结构强度改变”的解释，但不能证明结构强度在生物学上因果地改变 E/I 平衡或跨区信息传递。

<a id="dmf-yeo7"></a>

### 1.4 Yeo-7 功能网络层级分解

为直接检验结论是否依赖 Louvain 社区，这里只替换中尺度分组：68 个皮层 Lausanne/Desikan ROI 根据 fsaverage5 表面顶点与 Yeo2011 七网络注释的最大重叠归入 Visual、Somatomotor、Dorsal attention、Salience/ventral attention、Limbic、Frontoparietal control 或 Default mode。Yeo-7 不定义皮层下网络，因此 14 个皮层下 ROI 与 Brain-Stem 明确保留为一个 `Non-cortical` 组，而不被强行赋予 Yeo 标签。$G$、8 个 seed、每个条件的 2048 个干预样本、动力学噪声、300 步未来状态和估计器均与结构模块分解逐条件配对。

全系统分解为

$$
\Phi^{EID}_{166}
=\sum_{i=1}^{83}\Phi^{EID}_{\mathrm{within\ ROI}_i}
+\sum_{k=1}^{8}\Phi^{EID}_{\mathrm{cross\ ROI,within\ group}_k}
+\Phi^{EID}_{\mathrm{between\ groups}}.
$$

第二项只包含同一 Yeo 网络（或 Non-cortical 组）内不同 ROI 的协同；第三项包含不同功能组之间的协同。层级恒等式的最大误差为 $1.95\times10^{-13}$ bits。

临界窗平均总量为 $12.337$ bits。ROI 内成分为 $3.865$ bits（图 1E；31.3%），跨 ROI 成分为 $8.472$ bits（68.7%）；24/24 个 seed–$G$ 条件均为跨 ROI 大于 ROI 内。因此对“整个 $\Phi^{EID}$ 更多发生在脑区内还是跨脑区”的回答是：**当前 DMF 协议下主要是跨脑区协同，约为 ROI 内贡献的 2.19 倍。**随着 $G$ 从 1.6 增至 1.8，跨 ROI 比例进一步由 67.3% 升至 70.0%，而 ROI 内比例由 32.7% 降至 30.0%。

跨 ROI 成分中，同一功能组内部为 $1.246$ bits，占总量 10.1%、占跨 ROI 成分 14.7%；不同功能组之间为 $7.226$ bits，占总量 58.6%、占跨 ROI 成分 85.3%。后者在 24/24 个条件中都高于前者。图 1F 显示组内跨 ROI 贡献最高的是 Default mode（0.389 bits）与 Somatomotor（0.385 bits），但 Default 含 18 个 ROI、Somatomotor 含 11 个 ROI，而 Dorsal attention 和 Control 在本粗粒度映射中只有 2 和 3 个 ROI，因此这些绝对值同时受网络规模影响。

为隔离输入分布的作用，图 1C 将 WMS 与图 1A 的 $\Phi^{EID}$ 严格对齐：两者均使用完整 166 维 E/I source、完整 166 维未来 target、300-step horizon、2048 个样本、seeds 3–10、direct coupling、无状态裁剪、相同 target-noise seed 日程及 Gaussian 回归估计器；唯一处理因素是 source 来自自然稳态分布还是独立 $U(0.3,0.7)^{166}$ 干预分布。WMS 的展示扫描进一步加密为 $G=1.0,1.1,\ldots,3.0$，原 $\Phi^{EID}$ 的 10 个 $G$ 点均包含在该网格中。具体而言，两条曲线使用相同的 common-target whole-minus-sum 形式

$$
\Phi_p=I_p(\mathbf{s}_t;\mathbf{y}_{t+300})
-\sum_{i=1}^{166}I_p(s_{t,i};\mathbf{y}_{t+300}),
$$

其中 observational WMS 取 $p=p_{\mathrm{obs}}$ 并保留自然 source 协方差，$\Phi^{EID}$ 取 $p=p_{\mathrm{do}}=\prod_i p_i$。自然分布下的 WMS 在 168/168 个 seed–$G$ 条件中均为负，并在 $G=1.6$ 达到最深的 seed 均值 $-328.118\pm3.940$ bits，表明自然输入相关带来的冗余在临界窗附近最强。自然 source 协方差的中位条件数为 $5.37\times10^5$，因此绝对量级仍受 Gaussian 协方差正则化影响；这里优先解释跨 seed 稳定的曲线形状。

![DMF 临界窗多尺度汇总](../../fig/dmf_roi_yeo7_critical_summary_wms.png)

*图 1. DMF 临界窗的多尺度汇总。A：左轴为自发 DMF 轨迹的全脑平均兴奋性发放率，右轴为最大熵初态干预得到的全系统 $\Phi^{EID}$，为突出临界窗内的曲线形状，右轴显示范围固定为 8–13 bits。B：整体 EI 与 166 个单变量 E/I source EI 之和；后者不是 83 个 ROI-block EI 之和。C：平均发放率（左轴）与对齐后的全系统 observational $\Phi^{WMS}$（右轴）。WMS 与 A 中 $\Phi^{EID}$ 使用相同的完整 166 维 E/I source 和 target、300-step horizon、2048 样本、8 个种子、动力学、噪声日程、标准化与 Gaussian 估计器；仅 source 分布从独立 $U(0.3,0.7)^{166}$ 改为自然稳态分布。WMS 从 $G=1.0$ 到 $3.0$ 按 0.1 步长采样；曲线为 8 个种子的均值，阴影为标准误。A、C 的数值量级不同，不共用纵轴。A–C 的竖虚线为 Kuramoto 理论临界点 $K_c=1.5958$，灰带表示 DMF 临界平台 $G=1.6\text{-}1.8$。D：加权结构连接强度与 ROI 内 E/I 条件耦合（左轴）及跨 ROI 条件耦合（右轴）；每个点代表一个 ROI，两项均在 24 个临界窗条件上平均，相关统计见正文。E：24 个 seed–$G$ 条件平均的 ROI 内与跨 ROI $\Phi^{EID}$ 比例，误差线为条件均值的标准误。F：各功能组内部的跨 ROI 成分；括号给出 ROI 数量。组大小不同，因此 F 只描述总贡献，不能直接作为单位 ROI 的网络效应比较。G：68 个皮层 ROI 的跨 ROI 条件耦合四视角分布；15 个非皮层 ROI 仍参与 83 ROI 统计，但不投影到皮层表面。Yeo-7 仅覆盖皮层，Non-cortical 单列。原图 4A 的逐 $G$ 堆叠分解已删除。*

与 Louvain 分解相比，Yeo/Non-cortical 分组把更多贡献分到组间项：Louvain 模块间为 $4.283$ bits，而这里为 $7.226$ bits。这不能单独证明 Yeo 边界与动力学“更不一致”，因为 Yeo/Non-cortical 有 8 组、Louvain 有 6 组，分区更细时，PEID 的层级加性会自动把一部分原来的组内项移到组间项。严格比较两种分区是否具有额外解释力，还需要保持组数与组大小的 matched random-partition null。当前可直接支持的结论仅是：**无论采用结构 Louvain 还是预定义 Yeo-7，临界 $\Phi^{EID}$ 都不是单个 ROI 内部现象；Yeo 分组下尤其以跨功能组协同为主。**

<a id="dmf-detdeg"></a>

### 1.5 确定性 / 简并性分解与变化率

令 $H_0$ 为本 sweep 的固定目标参考熵，则

$$
D_{\mathrm{whole}}=H_0-H(\mathbf{T}\mid\mathbf{S}),\qquad
G_{\mathrm{whole}}=H_0-H(\mathbf{T}),
$$

$$
D_{\Sigma}=166H_0-\sum_{i=1}^{166}H(\mathbf{T}\mid S_i),\qquad
G_{\Sigma}=166G_{\mathrm{whole}}.
$$

![DMF 原始确定性 / 简并性曲线](../../fig/dmf_fullstate_maxent_detdeg_integrated_raw.png)

*图 2. 原始 bits 曲线。左轴为整体项，右轴为区域项之和；双轴仅保留真实量级，不能据此比较变化速度。*

![DMF 归一化确定性 / 简并性与变化率](../../fig/dmf_fullstate_maxent_detdeg_integrated_rate.png)

*图 3. 左图将每个种子内的四项缩放到各自 sweep 的 $[0,1]$ 范围；右图为相对于 $G$ 的变化率。因此该图比较的是形状与速度，而非 bits 大小。*

整体确定性在临界前接近平台、随后较平缓地下降；区域确定性和与两项简并性则在临界窗前快速上升，并在 $G\approx1.7$ 后翻转为下降。整体与区域简并性归一化后完全重合，因为 $G_{\Sigma}=166G_{\mathrm{whole}}$ 是定义上的比例关系，并不表示两种独立速度。简并性在临界窗的峰表示目标熵相对 $H_0$ 最低；高耦合端它回落，表示目标熵回到参考水平。区域确定性和的高耦合回落说明单一区域对未来完整状态的预测性变弱，而完整 source 的联合预测优势仍保留到临界窗附近。这一相对差异产生 $\Phi^{EID}$ 的临界峰。

<a id="dmf-limits"></a>

### 1.6 解释边界

$U(0,1)^{166}$ 是物理立方体上的绝对最大熵初态，但在该阴性对照中 $\Phi^{EID}$ 随 $G$ 下降，不能识别临界峰。因此本节的结论依赖于明确声明的 $[0.30,0.70]$ 生理支持，而不适用于无条件的 $[0,1]^{166}$ 干预。

该实验识别的是此 DMF、该代理结构连接、该时间窗和该连续高斯估计口径下的临界样动力学窗口，不对应人体大脑的固定耦合常数。确定性/简并性曲线提供与峰相容的熵分解，而不是同步、饱和或因果机制的直接测量。

<a id="hcp500"></a>

## 2. HCP500 PCA–Yeo7：Phi 分解

<a id="hcp500-data"></a>

### 2.1 数据、降维与动力学表征

数据为 30 名 HCP S1200 被试的 `REST1_LR` Schaefer-500 BOLD 时序，每名被试包含 $1200\times500$ 个时间点与皮层 parcel。500 个 parcel 按 Yeo7 标签分为 Vis、SomMot、DorsAttn、SalVentAttn、Limbic、Cont 和 Default；每个网络仅保留训练段拟合的一维 PCA（PC1），形成 7 维网络状态 $\mathbf{x}_t$。

Phi 实验使用 `sub-100206` 的时间验证选出的八阶 $\Delta$-Ridge，$p=8$、$\alpha=10$。模型以当前与过去七个时刻的网络状态为 source：

$$
\mathbf{h}_t=
\left[\mathbf{x}_t^\top,\mathbf{x}_{t-1}^\top,\ldots,\mathbf{x}_{t-7}^\top\right]^\top
\in\mathbb{R}^{56},
\qquad
\Delta\mathbf{x}_{t+1}=\mathbf{x}_{t+1}-\mathbf{x}_t.
$$

PCA、标准化器和 Ridge 均仅以每名被试的前 900 个时间点拟合；后 300 点不参与本 Phi 计算、参数选择或 null 重拟合。这里的目标是比较固定表征与固定模型下的 observed 和 null，而不是把该单被试模型推广为全体被试的预测最优模型。

<a id="hcp500-phi"></a>

### 2.2 History-source $\Phi^{EID}$ 与 circular-shift null

本实验的量是 56 维历史 source 到下一时刻 7 维状态的量，而不是 7 维一阶 whole-state $\Phi^{EID}$：

$$
\Phi^{EID}_{\mathrm{hist}\to\mathrm{next}}
=EI\!\left(\mathbf{h}_t;\mathbf{x}_{t+1}\right)
-\sum_{j=1}^{56}EI\!\left(h_{t,j};\mathbf{x}_{t+1}\right).
$$

估计使用 Gaussian log-det 口径，而非 TM。这样可在 56 维 history source、30 名被试、重复 null 重拟合和贪婪分解下保持可计算性；代价是结果依赖 Gaussian 近似，不能直接外推到非高斯的精确干预信息量。

null 对 7 条 PC1 时序分别施加独立、非零的 circular shift，并在相同的 $p$、$\alpha$ 与 900 点训练预算下重新拟合模型。它保留每条网络 PC1 的边际取值与自相关结构，但破坏网络间的时间对齐，检验观测到的跨网络结构是否超出网络内时间结构本身。

在 30 被试、每人 20-null 的扩展中，observed $\Phi^{EID}$ 的均值与中位数为 6.188481 与 6.068454 bits；observed-minus-null-mean 的均值与中位数为 1.984600 与 2.051671 bits，范围为 0.492521–4.096287 bits。30/30 名被试的 observed 都高于各自 null 均值，且未校正经验 $p<0.05$；由于仅有 20 个 null，每个被试的最小 p 值分辨率为 $1/21=0.047619$。

![30 被试 Yeo7-PC1 observed-minus-null PhiEID](../../results/hcp_schaefer500_yeo7_pc1_phi_null_all/observed_minus_null.png)

该结果支持在这一固定的 reduced-state、history-source 定义与 null 下，跨网络时间对齐带来额外的高阶结构；它不排除低频漂移、运动、生理噪声、PCA 表征选择或 Gaussian 近似造成的影响。

<a id="hcp500-module"></a>

### 2.3 Yeo7 模块历史分解

在全部 30 名被试上进行贪婪分解。为避免将同一网络的八阶历史误作 8 个独立脑区，属于同一个 Yeo7 网络的全部 8 个滞后 PC1 值被绑定为一个不可拆模块原子；候选空间因此是 7 个网络模块，而不是 56 个逐滞后变量。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | 固定集合未校正 $p<0.05$ |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **20 / 30** | 1.314357 bits | 20 / 20 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **17 / 30** | 1.227709 bits | 17 / 17 |
| SomMot + DorsAttn + SalVentAttn + Cont + Default | 9 / 30 | 0.956657 bits | 8 / 9 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.086753 bits | 7 / 8 |

![30 被试模块级 greedy PhiEID 核](../../results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png)

全 7 网络核最常出现（20/30），但它在 matched null cohort 中反而更常出现（均值 26.65/30；经验 $p=1$），因此不能将其读为真实数据特异的协同核。缺少 Limbic 的六网络广域核在真实数据中为 17/30，而 matched null cohort 的频率为 $8.65/30$（最大 12/30；经验 $p=1/21=0.047619$）；两个较小的候选核也分别为 9/30 对 $1.40/30$、8/30 对 $2.45/30$，同为该分辨率下的未校正 $p=1/21$。这支持真实静息态中若干非全网络模块核的出现频率和贡献高于该 circular-shift null；但 20 个 null 的 p 值分辨率有限，且统计未校正跨模块集合与 greedy 选择，因此不构成唯一生物学 atom 的确证。

<a id="hcp1000"></a>

## 3. HCP1000 PCA–Yeo7：Phi 分解

<a id="hcp1000-data"></a>

### 3.1 数据、降维与模型选择

同一 30 名 `REST1_LR` 被试的 `Schaefer1000` 矩阵为 $1200\times1000$。1000 个 parcel 按同一 Yeo7 顺序分为 Vis 162、SomMot 194、DorsAttn 122、SalVentAttn 121、Limbic 60、Cont 129 与 Default 212 个 parcel；每名被试的各网络 PC1 均只以前 900 点拟合并投影完整时序。

在 `sub-100206` 的训练段内以 600/700/800 三个时间验证折，从 $p\in\{1,2,3,5,8\}$ 与既有 Ridge $\alpha$ 网格选择模型。最优冻结配置为五阶 $\Delta$-Ridge，$p=5$、$\alpha=1$，平均 validation skill ratio 为 0.794433；因此 source 是 35 维网络历史，target 为下一时刻 7D 网络状态。后 300 点未参与 PC1、模型或参数选择。

<a id="hcp1000-phi"></a>

### 3.2 History-source $\Phi^{EID}$ 与 circular-shift null

对每名被试固定上述表征与模型，并以每条 PC1 独立、非零 circular shift 后重拟合同一模型生成 20 个 null。1000-parcel observed $\Phi^{EID}$ 的均值/中位数为 7.783676/7.734082 bits；observed-minus-null-mean 的均值/中位数为 2.997670/3.032261 bits，范围为 0.814450–6.085586 bits。30/30 名被试的 observed 均高于其 null 均值，未校正经验 p 均小于 0.05（最小分辨率 $1/21=0.047619$）。

![30 被试 Schaefer1000 Yeo7-PC1 observed-minus-null PhiEID](../../results/hcp_schaefer1000_yeo7_pc1_phi_null_all/observed_minus_null.png)

<a id="hcp1000-module"></a>

### 3.3 Yeo7 模块历史分解与 500 对照

分解中，同一网络的全部五个 PC1 历史滞后绑定为一个不可拆模块原子；每个 observed 与 null 都完整运行 greedy top-3。1000-parcel 的常见核如下。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | matched-null 频率均值；经验 p |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **21 / 30** | 1.637465 bits | 21.80 / 30；0.761905 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **12 / 30** | 1.575094 bits | 6.35 / 30；0.047619 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.500212 bits | 2.40 / 30；0.047619 |
| Vis + SomMot + DorsAttn + Cont + Default | 7 / 30 | 1.579938 bits | 2.90 / 30；0.047619 |

![30 被试 Schaefer1000 模块级 greedy PhiEID 核](../../results/hcp_schaefer1000_yeo7_module_phi_decomposition/top_core_consistency.png)

| 描述性比较 | Schaefer-500 | Schaefer-1000 |
|---|---:|---:|
| observed $\Phi^{EID}$ 均值（bits） | 6.188481 | 7.783676 |
| observed − null 均值（bits） | 1.984600 | 2.997670 |
| observed 高于 null mean | 30 / 30 | 30 / 30 |
| 全七网络核 top-3 频率 | 20 / 30 | 21 / 30 |
| 缺 Limbic 六网络核 top-3 频率 | 17 / 30 | 12 / 30 |

两种粒度都复现了跨网络时间对齐高于 circular-shift null 的方向性证据，并都将缺 Limbic 的六网络广域核识别为高于 matched-null 频率的候选结构。绝对 bits、最优滞后阶数和 atom 频率受分区粒度、PC1 表征与单被试调参影响；上表仅作描述性对照，不能当作空间粒度的正式统计检验，也不将 greedy 核解释为唯一生物学 atom。

<a id="hcp-wm"></a>

## 4. HCP500 WM_LR：任务态 Phi 与协同核

<a id="hcp-wm-phi"></a>


### 4.1 协同核分布及静息态对照

同一网络的 8 个历史滞后仍绑定为一个不可拆模块；每个 observed 和 null 均完整运行 greedy top-3。WM 中最常见的四个核如下。

| WM 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | WM matched-null 频率均值；经验 p |
|---|---:|---:|---:|
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **26 / 30** | 0.987442 bits | 18.55 / 30；0.047619 |
| 全部 7 个 Yeo 网络 | 14 / 30 | 1.045962 bits | 15.85 / 30；0.857143 |
| SomMot + DorsAttn + SalVentAttn + Cont + Default | 10 / 30 | 0.911514 bits | 1.30 / 30；0.047619 |
| Vis + SomMot + DorsAttn + Cont + Default | 7 / 30 | 0.945526 bits | 4.75 / 30；0.095238 |

![WM Yeo7 协同核分布及静息态频率对照](../../results/hcp_schaefer500_wm_yeo7_phi/wm_core_distribution.png)

缺 Limbic 的六网络核是 WM 最稳定的候选结构。它在 matched null 中的平均频率为 18.55/30、最大值为 24/30，而 observed 为 26/30（经验 $p=0.047619$）。在 29 名共同被试中，该核从静息态的 16/29 增至 WM 的 25/29：15 人两种状态均进入 top-3，10 人仅 WM 出现，1 人仅静息态出现，exact McNemar $p=0.011719$；对“缺 Limbic 六网络核”和“全七网络核”两项重点对照作 Holm 校正后 $p=0.023438$。相对地，全七网络核由静息态的 20/29 降至 WM 的 13/29，但差异未达到显著（McNemar 与 Holm $p=0.092285$），且 WM observed 频率不高于 matched null。

因此，REST–WM 的横向幅度比较以 observed raw $\Phi^{EID}$ 为准；在此基础上，greedy 核分布还显示 WM 更集中于缺 Limbic 的广域网络组合。该结果表明 WM 条件下 Vis、SomMot、DorsAttn、SalVentAttn、Cont 与 Default 的历史联合结构更频繁地成为 top 核；它不证明 Limbic 在工作记忆中不参与，也不把该六网络集合解释为唯一生物学 atom。20-null 的分辨率、不同拟合长度、任务共同驱动和未提供的运动/生理混杂仍限制机制解释。

<a id="hcp-all-tasks"></a>

### 4.3 七任务 raw Phi 排序

为检验静息态 raw $\Phi^{EID}$ 较高的现象是否只出现在 WM，这里将同一 Schaefer-500 协议扩展到 EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL 和 WM。跨任务固定 Yeo7-PC1 表征形式、八阶 $\Delta$-Ridge、$p=8$、$\alpha=10$、56 维历史 source、7 维 target 和 Gaussian log-det 估计器；但每个“任务 $\times$ 被试”都使用自己的前 75% 时间点重新拟合 PCA、标准化、Ridge 系数、截距和残差协方差。所谓“固定模型”因此指结构与超参数固定，不是跨任务复用同一组系数。本轮只比较 observed raw $\Phi^{EID}$，不生成新的 circular-shift null。

| 条件 | 29 名共同被试 raw $\Phi^{EID}$ 均值 | 中位数 |
|---|---:|---:|
| REST | **6.195996** | **6.126889** |
| EMOTION | 4.456909 | 4.445611 |
| GAMBLING | 4.717801 | 4.601157 |
| LANGUAGE | 4.556690 | 4.334584 |
| MOTOR | 5.170680 | 5.074332 |
| RELATIONAL | 4.755165 | 4.565034 |
| SOCIAL | 4.664090 | 4.681849 |
| WM | 4.940956 | 4.543372 |

![静息态与七任务 raw Phi 比较](../../results/hcp_schaefer500_all_tasks_phi/rest_all_tasks_raw_phi.png)

图 a 展示八个条件的跨被试 raw $\Phi^{EID}$ 分布；图 b 单独展开 29 名共同被试的 REST–WM 配对关系，每条线连接同一名被试；图 c 统计每个条件在多少名被试中取得八条件最大值。

任务态分析使用 30 名被试的 `WM_LR` `Schaefer500_taskRetained` 时序，每名被试为 $405\times500$。为保持方法迁移而不在任务态重新择优，模型冻结静息态 Schaefer-500 的八阶 $\Delta$-Ridge（$p=8,\alpha=10$）与 Gaussian log-det 估计器。每名被试的 Yeo7 PC1 仅以前 304 点拟合；source 仍为 56 维网络历史，target 仍为下一时刻 7 维网络状态。每个 observed 配置配对 20 个独立 PC1 circular-shift null，并对每个 null 重新拟合模型。

30 名被试的 observed $\Phi^{EID}$ 均值/中位数为 4.939700/4.557162 bits；observed-minus-null-mean 的均值/中位数为 1.781650/1.581241 bits。30/30 名被试均高于自身 null mean，22/30 的 observed 高于全部 20 个 null，因此达到当前分辨率下的最小经验 $p=1/21=0.047619$。受试者级 $\Delta\Phi$ 的均值 95% bootstrap CI 为 $[1.344991,2.423814]$ bits，paired sign-flip $p=5.0\times10^{-6}$。这支持 WM 中的跨网络时间对齐高于保留单网络自相关、但破坏网络间对齐的 null。

八条件重复测量的 Friedman 检验为 $\chi^2=49.919540$、$p=1.50\times10^{-8}$。REST 与每个任务的受试者内差值均为负，即任务 raw Phi 低于 REST；七项 paired sign-flip 的 Holm 校正 $p$ 从 $3.50\times10^{-5}$ 到 $8.45\times10^{-4}$，全部低于 0.05。与 REST 最接近的是 MOTOR，task-minus-REST 均值差仍为 $-1.025316$ bits（95% bootstrap CI $[-1.541913,-0.491539]$，Holm $p=0.000845$）；差距最大的是 EMOTION，为 $-1.739087$ bits（95% CI $[-2.296265,-1.216161]$，Holm $p=3.50\times10^{-5}$）。因此在预先使用的 $p=8,\alpha=10$ 配置下，**REST 的群体平均 $\Phi^{EID}$ 显著高于全部七个任务态。**第 4.4 节进一步检验该排序对超参数的依赖，结论不能外推为与 $p,\alpha$ 无关的普遍规律。

该排序不是逐人定律。REST 在 20/29 名共同被试中是八条件最大值；其余 9 人的最大值分别为 MOTOR 3 人、SOCIAL 2 人、WM 2 人、GAMBLING 1 人和 RELATIONAL 1 人。REST 的降序中位排名为第 1，但平均排名为 2.17。质量规则只标记 `sub-103515/WM` 的极端早期 PC1 瞬变，主分析仍保留该点。由于本轮未计算各任务 null，结论只针对 raw estimator 输出，不能进一步断言 REST 相对于各任务自身时间结构具有更高的 null 校正协同。

<a id="hcp-length-matched-variance"></a>

#### 4.3.1 REST–任务长度匹配方差检验

为检验 REST 的跨被试离散度是否只是由 1200 点长序列造成，这里对每个任务长度分别从 REST1_LR 取 12 个覆盖完整 run 的等距窗口：EMOTION 176 点、GAMBLING 253 点、LANGUAGE 316 点、MOTOR 284 点、RELATIONAL 232 点、SOCIAL 274 点和 WM 405 点。每个窗口都独立使用前 75% 时间点重新拟合 Yeo7 PC1、标准化、$p=8,\alpha=10$ 的 $\Delta$-Ridge 和残差协方差；任务态复用第 4.3 节相同估计器的 raw $\Phi^{EID}$。因此唯一主动改变的分析因素是 REST 序列长度，不使用 null 模型。

![REST 与七任务的长度匹配 raw Phi 方差](../../results/hcp_schaefer500_length_matched_variance/length_matched_variance.png)

七个小图分别给出 REST 与一个任务的两两配对分布。每个蓝点是该被试 12 个等长 REST 窗口 raw $\Phi^{EID}$ 的均值，橙点是同一被试的任务值，灰线连接同一被试；箱线图和面板内 SD 直接显示两组跨被试离散度。所有小图共享同一纵轴，因此可与第 4.3 节的原始分布图直接比较。图用于展示稳定的被试级 REST 值；下表的正式方差比仍对 12 个窗口位置分别计算跨被试方差后取平均，避免只看窗口均值掩盖时间位置变化。

| 任务 | 长度 | 任务 SD | 图中 REST SD | 单窗口 REST SD 范围 | 平均 REST/task 方差比（95% CI） | REST 方差较大的窗口 |
|---|---:|---:|---:|---:|---:|---:|
| EMOTION | 176 | 0.444 | 0.870 | 0.630--1.302 | **5.781**（3.288--11.040） | 12/12 |
| GAMBLING | 253 | 0.944 | 0.945 | 0.762--1.378 | 1.397（0.650--5.870） | 9/12 |
| LANGUAGE | 316 | 0.840 | 0.991 | 0.844--1.413 | 1.812（0.977--4.878） | 12/12 |
| MOTOR | 284 | 0.724 | 0.968 | 0.828--1.434 | **2.445**（1.561--4.200） | 12/12 |
| RELATIONAL | 232 | 0.978 | 0.925 | 0.727--1.297 | 1.252（0.658--3.297） | 8/12 |
| SOCIAL | 274 | 0.861 | 0.961 | 0.814--1.425 | 1.719（0.758--4.554） | 10/12 |
| WM | 405 | 1.702 | 1.033 | 0.892--1.458 | 0.460（0.193--3.024） | 0/12 |

长度匹配后，REST 的平均窗口方差在 EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL 和 SOCIAL 中仍大于任务，但只有 EMOTION 与 MOTOR 的 95% bootstrap CI 整体高于 1；其中 EMOTION、LANGUAGE 和 MOTOR 的 12/12 个窗口方向一致。WM 的普通方差反向主要受预先标记的 `sub-103515` 影响：排除该被试后，WM SD 从 1.702 降至 0.976，REST/task 平均窗口方差比由 0.460 升至 1.332，但 CI 仍跨 1（0.645--4.123）。IQR 与 MAD 的点估计在七个任务中均大于 1，进一步说明 WM 的普通方差反向并不代表其主体分布必然比 REST 更宽。

因此，**REST 高方差不是纯粹的全长序列效应，但也不是对七任务都同样稳定的状态规律。**长度匹配明显降低了 REST 的标准差：全长 REST 为 1.643 bits，而等长窗口依任务和位置为 0.630--1.458 bits；剩余差异最稳定地出现在 EMOTION 与 MOTOR。12 个窗口相互重叠，只作为 REST 时间位置的重复测量，不能当作独立被试；当前结果仍受单个 REST1_LR run、Gaussian log-det 估计器、任务事件结构和未回归的运动/生理混杂限制。

<a id="hcp-hyperparameter-robustness"></a>

### 4.4 REST–任务差异的 $p$–$\alpha$ 鲁棒性

原始 $p=8,\alpha=10$ 来自静息态模型选择，可能给 REST 带来选择优势。为检验这一点，这里预先固定 $p\in\{1,2,3,5,8\}$ 与 $\alpha\in\{0.1,1,10,100,1000\}$ 的 25 点网格。每个网格点把同一组超参数同时用于 REST 和七种任务态；29 名共同被试的每个状态仍独立重新拟合 PCA、缩放、Ridge 系数、截距和残差协方差。所有序列使用各自前 75%，只计算 raw $\Phi^{EID}$。每个网格点分别完成七项双侧 Monte Carlo paired sign-flip 检验（200,000 次），再在七任务内作 Holm 校正。

![REST 与七任务 raw Phi 的超参数鲁棒性](../../results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_robustness_overview.png)

图 a 给出每个网格点七项 REST-minus-task 群体均值差中的最小值；正值表示 REST 均值高于所有任务，负值表示至少一个任务反超。图 b 给出七项对比中 Holm 校正后显著的数量。REST 只在 **15/25** 个网格点为八条件群体均值最高状态，只有 **7/25** 个网格点满足“REST 高于全部任务且七项均显著”。这 7 个点为 $(p,\alpha)=(2,0.1),(2,1),(2,10),(3,1),(3,10),(5,10),(8,10)$。原始 $(8,10)$ 的 232 个“被试 $\times$ 状态”数值与第 4.3 节逐点完全一致，排除了结果变化来自实现路径差异。

![各任务 REST-minus-task raw Phi 超参数边际](../../results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_task_margins.png)

稳定区主要位于中等历史阶数和中等正则强度：$\alpha=10$ 时，$p=2,3,5,8$ 的七项对比均显著；但 $p=1$ 的所有 $\alpha$ 均为 0/7 显著。在过强正则 $\alpha=1000$ 下，五个 $p$ 均不再满足 REST 群体均值最高，且最多只有 1/7 项显著。弱正则与高阶历史的组合还会反转方向：$p=8,\alpha=0.1$ 时，REST 相对 EMOTION 和 RELATIONAL 分别低 2.789932 和 2.635 bits，两项任务高于 REST 的 Holm 校正 $p$ 分别为 0.003540 和 0.003150；其余任务的方向或显著性也不统一。

因此，最窄且可靠的结论是：**REST 高于七任务的 raw $\Phi^{EID}$ 在包含原配置的中等正则参数带内可复现，但对完整 25 点 $p$–$\alpha$ 网格并不鲁棒。**$p$ 同时改变 source 维数 $7p$，所以应在同一个 $(p,\alpha)$ 内解释 REST–任务差值，不能把不同 $p$ 的绝对 raw Phi 当作同维度量直接比较。上述 Holm 校正只针对每个网格点内部的七项任务对比，并未把 25 个网格点作为发现性假设族再次校正；因此 7/25 是敏感性描述，不是 175 项独立发现。极端超参数是否应被视为合理模型还需要独立、条件平衡的预测验证规则；在完成该步骤之前，不能从本扫描中事后挑选最支持 REST 或任务态的参数作为主结论。

#### 4.4.1 留出预测误差解释弱正则反转

为判断左下角弱正则、高阶历史的反转是否来自拟合失真，这里保持同一 25 点网格和每个“状态 $\times$ 被试”的前 75% 训练段，严格用最后 25% 时间点计算一步留出误差。主指标 delta-NRMSE 先用训练段的每个网络 delta 标准差归一化，再跨网络、时间点和被试汇总；同时报告 test-minus-train 泛化间隙，以及相对“$\mathbf{x}_{t+1}=\mathbf{x}_t$”持久性基线的技能值。技能为正表示优于持久性预测，为负表示模型在留出段反而更差。

![REST 与七任务的预测误差超参数诊断](../../results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_overview.png)

图 a 显示八状态等权平均的留出 delta-NRMSE，图 b 显示泛化间隙，图 c 单独展开 $p=8$ 时各状态随 $\alpha$ 的误差。结果支持过拟合解释：$p=8,\alpha=0.1$ 的训练误差仅为 0.6538，但留出误差升至 0.9612，泛化间隙达到 0.3074，持久性技能为 $-0.0406$；其平均 Ridge 系数 Frobenius 范数为 7.735。改用 $\alpha=10$ 后，留出误差降至 0.8726、泛化间隙降至 0.1685、持久性技能升至 0.2049，系数范数缩至 2.935。$p=8$ 的整体最低留出误差出现在 $\alpha=100$，为 0.8590，说明 $\alpha=10$ 位于高泛化区，但不是该阶数下的唯一或总体最优正则值。

![各状态的留出预测误差](../../results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_by_condition.png)

发生显著 raw $\Phi^{EID}$ 反转的两个任务同时给出最直接的失败证据。在 $p=8,\alpha=0.1$ 下，EMOTION 的训练/留出误差为 0.5487/1.0628，泛化间隙 0.5141，持久性技能 $-0.3454$；RELATIONAL 为 0.6176/1.0062，泛化间隙 0.3887，持久性技能 $-0.1106$。也就是说，弱正则模型在训练段把这两个较短任务拟合得异常好，但在未参与拟合的时间段已经不具备可靠预测能力。固定 $p=8$ 后，在全部 35 个“超参数 $\times$ 任务”对比中，REST-minus-task Phi 边际与 task-minus-REST 留出误差差的 Spearman $\rho=-0.592$（$p=1.79\times10^{-4}$），与泛化间隙差的 $\rho=-0.480$（$p=0.00351$）：任务相对 REST 过拟合越明显，Phi 排序越倾向任务反超。

这种反转的可能机制是：$p=8$ 提供 56 维历史 source，而任务态用于回归的训练样本仅约 124--296 行；$\alpha=0.1$ 允许更大的回归系数，并在训练段压低残差协方差。Gaussian log-det $\Phi^{EID}$ 同时依赖转移系数和训练残差协方差，因此会把这种训练内的高自由度拟合转化为偏大的 raw Phi。该证据说明预测失真是反转的重要来源，但不是数学上的唯一原因，因为 $\Phi^{EID}$ 不是预测误差的单调函数，且并非所有预测较差的点都发生显著方向反转。

从共享超参数选择看，$\alpha=10$ 在 40 个“八状态 $\times$ 五个 $p$”组合中的 29 个（72.5%）达到最低留出误差；八状态总体平均下，$p=1,2,3,5$ 的最优 $\alpha$ 都是 10，$p=8$ 则是 100。若在整个 25 点网格中按八状态与29名被试等权的留出误差选择一组共享参数，最优点是 $(p,\alpha)=(5,10)$，delta-NRMSE 为 0.8552；在这个不依据 REST–任务 Phi 方向选出的配置上，REST 相对最接近任务仍高 1.4243 bits，七项对比全部 Holm 显著。因此，更公平的预测型选参仍支持“REST 高于七任务”，但应把主配置更新为或至少敏感性报告 $(5,10)$，并在独立被试或独立 run 上验证以避免本次网格选择本身的乐观偏差。

<a id="hcp-all-task-cores"></a>

### 4.5 共享最优模型下的七任务协同核

为定位各任务的主要 Yeo7 协同组合，这里冻结不依据 Phi 方向选择的共享预测最优模型 $(p,\alpha)=(5,10)$。七任务使用相同30名被试；每个“任务 $\times$ 被试”仍在自己的前75%时间点上重新拟合 PCA、标准化、$\Delta$-Ridge 与残差协方差。同一网络的5个历史滞后绑定为不可拆模块，每名被试保留 greedy atom 贡献最高的三个核。每个 observed 配置另配20个独立 PC1 circular-shift null，并对每个 null 重新拟合模型和完整重跑 greedy top-3。

PEID 的层级加性保证沿模块细化路径的 atom 贡献可加（Zotero `MYATYWAJ`，全文）；本实验数值上 atom 和与模块整体 Phi 的最大误差为 $1.78\times10^{-15}$。但 greedy top-3 仍是为控制组合爆炸而选择的候选核，不等于对所有可能分解路径的唯一 exhaustive 解。核心“重要性”首先由跨被试进入 top-3 的频率定义，进入 top-3 时的平均 atom 贡献作为次级强度指标。

![七任务 Yeo7 协同核分布](../../results/hcp_schaefer500_all_tasks_core_decomposition/core_task_landscape.png)

图 a 的数字是30名被试中该核进入 top-3 的人数，星号表示 observed 频率高于全部20个 matched-null cohort，即达到当前分辨率的最小未校正 $p=1/21$；图 b 给出该核进入 top-3 时的平均 atom 贡献。七任务的每任务前三名只涉及三个组合：缺 Limbic 的六网络核、全七网络核，以及同时缺 Vis 和 Limbic 的五网络核。

| 任务 | 缺 Limbic 六网络核 | 全七网络核 | 缺 Vis/Limbic 五网络核 |
|---|---:|---:|---:|
| EMOTION | **22/30；0.638** | 14/30；0.648 | 11/30；0.640 |
| GAMBLING | **27/30；0.753** | 7/30；0.633 | 11/30；0.816 |
| LANGUAGE | **21/30；0.650** | 13/30；0.652 | 15/30；0.684 |
| MOTOR | 20/30；0.835 | **21/30；0.761** | 13/30；0.796 |
| RELATIONAL | **25/30；0.730** | 12/30；0.638 | 11/30；0.745 |
| SOCIAL | **22/30；0.693** | 14/30；0.722 | 14/30；0.811 |
| WM | **25/30；0.748** | 7/30；0.957 | 12/30；0.734 |

表中每格依次为 top-3 频率和 top 时平均贡献（bits）。最稳定的组合是

$$
\{\mathrm{Vis,SomMot,DorsAttn,SalVentAttn,Cont,Default}\},
$$

即缺 Limbic 的六网络核。除 MOTOR 中以20/30略低于全七网络核的21/30外，它在其余六个任务均排名第一；七任务频率范围为20--27/30，总计162次被试-任务 top-3。同一被试内也具有跨任务稳定性：每名被试平均在5.40/7个任务中出现该核，8/30名在全部七任务中出现，15/30名至少六个任务出现，27/30名至少四个任务出现。对应 null 频率均值依次为12.80、14.75、15.05、9.55、14.00、15.45和14.90/30，七个任务均达到未校正 $p=1/21$。因此，该核不是 WM 独有，而是当前共享最优模型下跨任务最稳定的广域协同骨架。

缺 Vis 和 Limbic 的五网络核在七任务中为11--15/30，虽然只有 LANGUAGE 达到预设的15/30高频阈值，但七任务的 observed 频率也都高于全部20个 null cohort；其 null 均值仅为1.25--2.10/30。这说明 SomMot、DorsAttn、SalVentAttn、Cont 与 Default 构成一个重复出现的次级核心。相反，全七网络核虽在 MOTOR 达到21/30，但其 null 均值为21.60/30、$p=0.810$；其余任务的 null 排名 $p$ 也为0.905--1.000。故“全网络参与”很容易由保留单网络时间结构的 null 产生，不能仅凭 observed 高频解释为任务特异协同。

![七任务 Yeo7 网络参与率](../../results/hcp_schaefer500_all_tasks_core_decomposition/network_participation.png)

网络参与率进一步解释了组合结构。SomMot、DorsAttn、SalVentAttn、Cont 和 Default 在各任务的被试级 top-3 atoms 中分别出现于87%--93%、78%--92%、79%--97%、82%--98%和80%--91%；Vis 为56%--69%。Limbic 只有9%--31%，在 GAMBLING 和 WM 最低（9%和11%），MOTOR 最高（31%）。这一差异与缺 Limbic 六网络核的跨任务主导一致，但不证明 Limbic 在任务执行中不参与；它只说明在当前模块级 greedy 分解中，Limbic 较少成为高贡献协同核的必要成员。

综合而言，各任务最重要的核不是七套完全不同的组合，而是一个共享的“缺 Limbic 六网络”主骨架，加上“缺 Vis/Limbic 五网络”次级骨架；任务差异主要体现在这两个骨架与全七网络核的相对频率和贡献强度。20-null 只有 $1/21$ 的最小分辨率，且星号未跨任务、跨核心校正，因此它们是探索性超-null证据。唯一质量标记仍为 `sub-103515/WM`，主分析保留该被试。

<a id="hcp-network-attribution"></a>

### 4.6 REST–MOTOR–WM 网络级 EI、Phi 归因与协同核

为在同一张图中区分“单网络携带的信息”“网络内部历史整合”和“跨网络协同”，这里使用 REST、MOTOR 与 WM 的 29 名共同被试，并冻结跨状态共享预测最优模型 $(p,\alpha)=(5,10)$。三个条件分别以前 75% 时间点拟合 Yeo7-PC1、标准化、$\Delta$-Ridge 与残差协方差；REST、MOTOR 和 WM 的拟合长度分别为 900、213 和 304 点。比较保持网络数、五阶历史 source、7 维 target 与 Gaussian log-det 估计器一致，但条件和有效样本长度仍同时变化。

令网络集合为 $\mathcal{N}=\{1,\ldots,m\}$，其中 $m=7$；目标为 $\mathbf{y}_t=\mathbf{x}_{t+1}$。网络 $i$ 的五阶历史向量为

$$
\mathbf{h}^{(i)}_t=
\left(x_{t,i},x_{t-1,i},\ldots,x_{t-4,i}\right)^\top,
$$

对任意 $S\subseteq\mathcal{N}$，记 $\mathbf{h}^{(S)}_t$ 为按固定网络顺序拼接 $\{\mathbf{h}^{(i)}_t:i\in S\}$ 得到的历史向量，并定义联盟 EI

$$
F(S)=EI\!\left(\mathbf{h}^{(S)}_t;\mathbf{y}_t\right),
\qquad F(\varnothing)=0.
$$

网络 $i$ 的模块 EI 为 $EI_i=F(\{i\})$。网络内部历史整合定义为

$$
\Phi_i^{\mathrm{within}}
=EI_i-\sum_{\ell=0}^{4}EI\!\left(x_{t-\ell,i};\mathbf{y}_t\right).
$$

因此，完整 35 维历史 source 的 raw $\Phi^{EID}$ 可以严格拆成网络内部项与跨网络项：

$$
\begin{aligned}
\Phi^{EID}_{\mathrm{raw}}
&=F(\mathcal{N})-
  \sum_{i\in\mathcal{N}}\sum_{\ell=0}^{4}
  EI\!\left(x_{t-\ell,i};\mathbf{y}_t\right)\\
&=\underbrace{F(\mathcal{N})-
  \sum_{i\in\mathcal{N}}F(\{i\})}_{\Phi^{EID}_{\mathrm{cross}}}
  +\sum_{i\in\mathcal{N}}\Phi_i^{\mathrm{within}}.
\end{aligned}
$$

#### 精确 Shapley 网络归因

对任意网络联盟 $S\subseteq\mathcal{N}$，定义跨网络协同联盟价值

$$
v(S)=F(S)-\sum_{j\in S}F(\{j\}),
\qquad v(\varnothing)=v(\{i\})=0.
$$

于是 $v(\mathcal{N})=\Phi^{EID}_{\mathrm{cross}}$。网络 $i$ 的精确 Shapley 归因为

$$
\psi_i=
\sum_{S\subseteq\mathcal{N}\setminus\{i\}}
\frac{|S|!\,(m-|S|-1)!}{m!}
\left[v(S\cup\{i\})-v(S)\right].
$$

其边际项可进一步展开为

$$
v(S\cup\{i\})-v(S)
=F(S\cup\{i\})-F(S)-F(\{i\}),
$$

即网络 $i$ 加入已有联盟 $S$ 后产生的联合 EI 增量，扣除网络 $i$ 单独已经具有的模块 EI。等价地，若 $\pi$ 遍历 $m!$ 个网络加入顺序，$P_i^\pi$ 表示顺序 $\pi$ 中位于 $i$ 之前的网络集合，则

$$
\psi_i=\frac{1}{m!}\sum_{\pi}
\left[v(P_i^\pi\cup\{i\})-v(P_i^\pi)\right].
$$

因此 Shapley 不是挑选某一个协同组合，而是对网络 $i$ 在所有可能联盟背景和加入顺序下的边际跨网络协同取加权平均。由 Shapley 的效率性质，

$$
\sum_{i\in\mathcal{N}}\psi_i
=v(\mathcal{N})-v(\varnothing)
=\Phi^{EID}_{\mathrm{cross}}.
$$

图中节点颜色使用的网络总 $\Phi$ 归因为

$$
c_i=\Phi_i^{\mathrm{within}}+\psi_i,
\qquad
\sum_{i\in\mathcal{N}}c_i=\Phi^{EID}_{\mathrm{raw}}.
$$

在给定联盟价值函数 $v$ 后，Shapley 是同时满足效率、对称性、虚玩家和可加性四项公理的唯一节点级分摊。这里的“公平”是数学公理意义上的归因约定，不等于证明 $\psi_i$ 是网络 $i$ 固有或独立产生的生物学量；一般的非单调联盟价值也不保证每个 $\psi_i$ 必为非负。

#### 与 greedy $\Phi$ 层级分析的严格区别

greedy 层级分析使用同一个集合函数 $\phi(S)=v(S)$，但目标不是把 $v(\mathcal{N})$ 分给单个网络，而是保留多网络组合。对 $|S|\geq2$，令 $\mathcal{B}(S)$ 为 $S$ 的全部无序非平凡二分 $(A,B)$，并定义该次细化留下的层级残差

$$
r(S;A,B)=\phi(S)-\phi(A)-\phi(B).
$$

当前 `nonnegative_tolerant` 实现只接受 $r(S;A,B)\geq-\tau$ 的候选，并选择

$$
(A^*,B^*)
=\underset{(A,B)\in\mathcal{B}(S),\ r(S;A,B)\geq-\tau}{\arg\max}
\left\{\phi(A)+\phi(B)\right\},
$$

即在容差内选择能被两个子块解释得最多、因而剩余高阶残差最小的二分。若不存在可用二分，或最佳子块捕获量不超过阈值 $\varepsilon$，则 $S$ 成为终端 atom，贡献为 $a(S)=\phi(S)$；否则记录当前组合的残差 atom

$$
a(S)=r(S;A^*,B^*)
$$

并分别递归细化 $A^*$ 与 $B^*$。对于得到的二叉细化树 $\mathcal{T}$，未截断的理论恒等式为

$$
\phi(\mathcal{N})=
\sum_{u\in\operatorname{Int}(\mathcal{T})}
r(S_u;A_u,B_u)
+\sum_{q\in\operatorname{Leaf}(\mathcal{T})}\phi(S_q).
$$

实现中只报告大于 $\varepsilon$ 的非负 atom，并容忍 $[-\tau,0)$ 的数值残差；本实验中所有已保留 atom 的和仍在数值精度内等于 $\Phi^{EID}_{\mathrm{cross}}$。该层级加性对应 PEID 的分区细化恒等式（Zotero `MYATYWAJ`，全文），但 greedy 只选择一棵二叉细化树，不枚举所有可能的层级树，所以结果可能受最优二分并列、阈值和层级策略影响。

| 比较问题 | 精确 Shapley | Greedy $\Phi$ 层级分析 |
|---|---|---|
| 分解对象 | 同一个跨网络价值 $v(S)$ | 同一个跨网络价值 $\phi(S)=v(S)$ |
| 输出单位 | 单网络归因 $\psi_i$ | 多网络集合 atom $a(S)$ |
| 计算逻辑 | 平均全部联盟背景或全部加入顺序 | 选择一棵逐次最优二分树 |
| 守恒关系 | $\sum_i\psi_i=v(\mathcal{N})$ | 树上残差与终端项之和为 $v(\mathcal{N})$ |
| 是否保留组合身份 | 否；联合协同被分摊给成员 | 是；atom 保留其网络集合 $S$ |
| 唯一性 | 给定 $v$ 和四项公理后唯一 | 依赖二分选择、并列规则与容差 |
| 本图用途 | 节点颜色：哪个网络平均分得多少 $\Phi$ | 多边形与图 d：哪些网络共同构成高贡献核 |

一个最小例子能直接说明差别。若三个网络只有纯三元协同，即 $v(\{A,B,C\})=1$ bit，而所有真子集的 $v$ 都为 0，则 Shapley 输出 $\psi_A=\psi_B=\psi_C=1/3$ bit；greedy 层级分析则保留一个 $\{A,B,C\}$ 终端 atom，贡献为 1 bit。前者适合画三个节点各自分得多少，后者保留“这 1 bit 必须由三者联合出现”的组合身份。

因此两者不是竞争估计器，而是对同一个 $\Phi^{EID}_{\mathrm{cross}}$ 回答两个不同问题：Shapley 回答“如何把总协同归因到各网络”，greedy 层级分析回答“总协同主要以哪些不可再由所选子块解释的多网络组合出现”。某网络 Shapley 值较低，不等于它不会进入高阶协同核；某网络未进入最常见 greedy 核，也不等于其 Shapley 归因为零。两种方法都复用全部 $2^7=128$ 个联盟的 EI 表；Shapley 对这张表做精确节点归因，greedy 则避免枚举所有可能的层级树。

上述 Shapley 守恒恒等式与 greedy 跨网络 atom 加性在 87 个“被试 $\times$ 条件”拟合中的最大数值误差为 $2.66\times10^{-15}$ bits。由于 Shapley 不保留多网络组合身份，图中仍以多边形和协同核矩阵单独显示 greedy 识别的组合结构。

![REST、MOTOR 与 WM 的 Yeo7 网络 EI、Phi 归因及协同核](../../results/hcp_schaefer500_yeo7_network_attribution/rest_motor_wm_network_attribution.png)

图 a 的节点大小编码模块 EI，节点颜色编码总 $\Phi$ 归因；多边形标出各条件最常进入被试 top-3 的协同核，其颜色编码该核进入 top-3 时的平均 greedy atom 贡献。图 b、c 分别给出模块 EI 和总 $\Phi$ 归因的跨被试均值与 bootstrap 95% CI；图 d 以数字表示协同核进入 top-3 的人数，以颜色表示该核的平均 atom 贡献，并与图 a 的多边形共享同一 atom 色标。REST、MOTOR 与 WM 的多边形对应值分别为 1.077、0.754 和 0.744 bits。

REST、MOTOR 与 WM 的 raw $\Phi^{EID}$ 均值分别为 5.677、4.252 和 4.123 bits，其中跨网络 $\Phi$ 分别为 5.013、3.937 和 3.797 bits，约占各自 raw $\Phi$ 的 88%、93% 和 92%；其余部分来自单个网络内部五阶历史的整合。REST 的总归因以 SomMot（0.988 bits）、DorsAttn（0.938 bits）和 Default（0.864 bits）最高；MOTOR 以 SomMot（0.701 bits）、SalVentAttn（0.692 bits）和 Cont（0.686 bits）最高；WM 以 Cont（0.706 bits）最高，DorsAttn 与 Default 均约为 0.669 bits。Limbic 在三条件中的平均归因均最低，但不为零，因此这些结果不支持将 Limbic 解释为不参与系统动力学。

协同核进一步区分了相近的节点级归因：REST 与 MOTOR 最常见的是全七网络核，分别为 15/29 和 20/29；WM 最常见的是缺 Limbic 的六网络核，为 24/29。说明 WM 的组合结构更集中于缺 Limbic 的广域核，而 MOTOR 虽也频繁出现该核（19/29），全七网络核仍略高（20/29）。本分析不使用 null 作横向条件基线，也未对网络级条件差作发现性显著性检验。`sub-103515/WM` 仍触发既定质量规则并保留；不同拟合长度、任务共同驱动及未回归的运动或生理混杂继续限制纯任务效应解释。

<a id="discussion"></a>

## 5. 讨论：解释边界与可复现性

<a id="discussion-scope"></a>

### 5.1 结论的适用范围

- DMF 结果支持 $\Phi^{EID}$ 在该代理结构连接、无裁剪全状态干预与 $[0.30,0.70]^{166}$ 生理支持下定位 Kuramoto 对齐的临界窗；它不等同于人体大脑存在同一精确耦合常数，也不外推到绝对最大熵 $[0,1]^{166}$ 干预。
- HCP 静息态结果来自 REST1_LR；任务态 raw Phi 已覆盖七种 `taskRetained` LR 任务。主比较固定 7 个 PC1 网络状态与 $p=8,\alpha=10$，PCA、缩放、Ridge 系数和残差协方差按任务和被试重新拟合；25 点敏感性扫描表明 REST 高于全部任务的结论只在部分中等正则配置成立。留出误差诊断将弱正则、高阶历史下的方向反转定位为明显过拟合，并给出跨状态共享预测最优点 $(p,\alpha)=(5,10)$。长度匹配检验进一步表明 REST 高方差在 EMOTION 与 MOTOR 上最稳定，而不是七任务统一规律。该共享最优模型的七任务20-null模块分解显示缺 Limbic 六网络核为跨任务主骨架；REST–MOTOR–WM 的精确 Shapley 分析进一步将 raw Phi 分为网络内历史整合和跨网络协同归因。尚未检验 RL run、独立被试上的嵌套选参、任务事件分段、去趋势、运动或生理混杂回归、GSR 或其他 null 构造。
- WM 与既有静息态的主幅度比较分别使用 304 和 900 个拟合时间点，只比较 raw Phi，因此均值差仍包含有效样本长度差异。独立的 12 窗口长度匹配分析只检验跨被试方差；WM 的 `sub-103515` 具有极端早期 PC1 瞬变，普通方差比对其高度敏感。
- HCP 的全体被试 Phi 结果使用 20 个 null，p 值分辨率有限，且未校正跨被试、跨模块集合和 greedy 选择造成的多重比较。
- 贪婪 atom 用于描述候选协同结构；它依赖分解顺序与候选空间，不是 exhaustive 的唯一高阶分解。

<a id="discussion-artifacts"></a>

### 5.2 结果与图形产物

| 实验 | 关键图与结果 |
|---|---|
| 83 ROI 临界识别 | `fig/dmf_phi_r_phase_reproduction.{png,svg,pdf}`、`fig/dmf_fullstate_maxent_critical_confirmation.{png,svg,pdf}`、`fig/dmf_roi_yeo7_critical_summary_wms.png`、`fig/dmf_kuramoto_fullstate_shape_alignment.{png,svg,pdf}`、`fig/dmf_fullstate_maxent_detdeg_integrated_raw.{png,svg,pdf}`、`fig/dmf_fullstate_maxent_detdeg_integrated_rate.{png,svg,pdf}`、`fig/dmf_pairwise_phi_eid_mean_curve.{png,svg,pdf}`、`results/dmf_83_whole_system_wms/aligned_observational_tau300_n2048_seeds3_10_dense_g01.npz`、`results/dmf_fullstate_uniform_support/confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz`、`results/dmf_phi_eid_hierarchical_topology/critical_hierarchy.npz`、`results/dmf_phi_eid_yeo7_hierarchy/critical_yeo7_hierarchy.npz`、`results/dmf_pairwise_phi_eid_confirmation/support030_070_tau400_n2048_seeds3_10.npz` |
| HCP500 Yeo7-PCA Phi/null | `results/hcp_schaefer500_yeo7_pc1_phi_null/summary.json`、`results/hcp_schaefer500_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP500 Yeo7 模块分解 | `results/hcp_schaefer500_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png` |
| HCP1000 Yeo7-PCA Phi/null | `results/hcp_schaefer1000_yeo7_ridge_selection/summary.json`、`results/hcp_schaefer1000_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP1000 Yeo7 模块分解 | `results/hcp_schaefer1000_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer1000_yeo7_module_phi_decomposition/top_core_consistency.png` |
| HCP500 WM_LR Phi 与协同核 | `results/hcp_schaefer500_wm_yeo7_phi/summary.json`、`results/hcp_schaefer500_wm_yeo7_phi/report.md`、`results/hcp_schaefer500_wm_yeo7_phi/wm_rest_phi_comparison.{png,svg,pdf}`、`results/hcp_schaefer500_wm_yeo7_phi/wm_core_distribution.{png,svg,pdf}` |
| HCP500 静息态与七任务 raw Phi | `results/hcp_schaefer500_all_tasks_phi/summary.json`、`results/hcp_schaefer500_all_tasks_phi/report.md`、`results/hcp_schaefer500_all_tasks_phi/rest_all_tasks_raw_phi.{png,svg,pdf}` |
| HCP500 REST–七任务长度匹配方差 | `results/hcp_schaefer500_length_matched_variance/summary.json`、`results/hcp_schaefer500_length_matched_variance/report.md`、`results/hcp_schaefer500_length_matched_variance/experiment_contract.json`、`results/hcp_schaefer500_length_matched_variance/rest_window_phi.npz`、`results/hcp_schaefer500_length_matched_variance/length_matched_variance.{png,svg,pdf}` |
| HCP500 REST–七任务 $p$–$\alpha$ 鲁棒性 | `results/hcp_schaefer500_phi_hyperparameter_robustness/summary.json`、`results/hcp_schaefer500_phi_hyperparameter_robustness/report.md`、`results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_robustness_overview.{png,svg,pdf}`、`results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_task_margins.{png,svg,pdf}` |
| HCP500 REST–七任务预测误差诊断 | `results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_summary.json`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_report.md`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_overview.{png,svg,pdf}`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_by_condition.{png,svg,pdf}` |
| HCP500 七任务共享最优模型协同核 | `results/hcp_schaefer500_all_tasks_core_decomposition/summary.json`、`results/hcp_schaefer500_all_tasks_core_decomposition/report.md`、`results/hcp_schaefer500_all_tasks_core_decomposition/core_task_landscape.{png,svg,pdf}`、`results/hcp_schaefer500_all_tasks_core_decomposition/network_participation.{png,svg,pdf}` |
| HCP500 REST–MOTOR–WM 网络归因 | `results/hcp_schaefer500_yeo7_network_attribution/summary.json`、`results/hcp_schaefer500_yeo7_network_attribution/report.md`、`results/hcp_schaefer500_yeo7_network_attribution/rest_motor_wm_network_attribution.{png,svg,pdf}` |

<a id="appendix-a"></a>

## 附录 A：Kuramoto 振子数与 whole-state $\Phi^{EID}$ 曲线形状

为避免把方程差异误读成振子数效应，这里重新使用同一个经典全局耦合 Kuramoto 方程，只改变振子数：

$$
\dot{\theta}_i
=\omega_i+\frac{K}{N}\sum_{j=1}^{N}\sin(\theta_j-\theta_i).
$$

除振子数外，两组实验使用同一协议：频率 $\omega_i$ 从零均值 Gaussian 抽样，随后对每个 seed 去均值并重缩放到 `sigma=1`；`N=2` 时这个协议退化为一对符号相反、标准差为 1 的频率。source 是全部振子的当前相位特征，target 是全部振子的未来相位状态，而不是整体速度；两组都直接计算与 Part2 大脑动力学 $\Phi^{EID}$ 相同的源侧 whole-minus-sum 结构：

$$
\Phi^{EID}
=
EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})
-\sum_{i=1}^{N} EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})
.
$$

其中 $\mathbf{s}_t^i=(\cos\theta_i(t),\sin\theta_i(t))$ 是第 $i$ 个振子的相位特征，$\mathbf{y}_{t+\tau}=\{(\cos\theta_i(t+\tau),\sin\theta_i(t+\tau))\}_{i=1}^{N}$ 是系统整体未来相位状态。$\Phi^{EID}$ 由该定义保证非负，因此无需引入人工非负截断。

下文首先以 `N=64` Oracle 结果解释临界峰的机制；振子数对照作为系统规模边界证据，统一放在附录末尾。

<a id="appendix-a-1"></a>

### A.1 临界峰的 EI 与 effectiveness 机制

为了检查这个峰值来自哪一项，进一步把 `N=64` Oracle whole-state 结果分解为联合 EI 与单独 EI 之和：

![Large-N Kuramoto EI decomposition](../../fig/classic_network_dynamics_benchmark/large_kuramoto_n64_ei_decomposition.png)

分解结果显示，$EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})$ 和 $\sum_i EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})$ 都随 $K$ 增大而整体下降。这不是反常现象，因为这里的 EI 衡量的是最大熵相位干预下，当前相位状态有多少可区分信息保留到未来 whole-state target 中。`K=0` 时，每个振子近似独立转动，当前相位到未来相位接近一一映射，所以联合 EI 和单独 EI 之和都很高，并且二者几乎相等，$\Phi^{EID}\approx0$。

随着 $K$ 增大，同步吸引会压缩相位差自由度，许多不同初始相位会被映射到更相似的未来状态，因此总的可区分信息下降。临界前沿附近，单个振子对未来全系统状态的解释力下降得更快，而联合状态仍保留对集体相位关系的解释力，所以两项差值扩大，$\Phi^{EID}$ 在 `K≈1.7` 达峰。到强同步区后，系统接近低维同步流形，联合 EI 本身也明显降低，差值随之回落。换言之，临界峰不是因为总 EI 最大，而是因为整体相对于部分之和的不可分解优势最大。

同一组 `N=64` Oracle 结果还可以按 effectiveness 的 determinism/degeneracy 口径拆开。这里固定参考熵 $H_0$ 为本 sweep 中最大的 Gaussian target entropy，并定义

$$
Det(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}\mid \mathcal{S}),\qquad
Deg(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}).
$$

其中 $\mathcal{S}$ 可以是全部振子的联合 source，也可以是某个单振子 source；$\mathbf{Y}$ 是 whole-state future target。为避免将四个高度相关的曲线拆散，左图把 whole-source determinism 与 degeneracy 合并到同一**线性**轴，从而突出 determinism 的低谷；右图在单一对数轴上并列 singleton-sum 的两项，保留其跨数量级的共同膨胀与接近。两图的同一条竖虚线标出 whole-source determinism 的最小点，便于把这两个尺度上的变化对齐。

![Large-N Kuramoto determinism and degeneracy decomposition](../../fig/classic_network_dynamics_benchmark/n64_detdeg/large_kuramoto_oracle_nsource_whole_state_phi_sweep_determinism_degeneracy.png)

这个分解补足了临界峰的解释。whole-state determinism 从 `K=0` 的约 `1110.05` bits 下降，在 `K=2.0` 附近降到约 `475.95` bits，随后强同步区又回升到 `K=4.0` 的约 `1078.10` bits；whole-state degeneracy 则从近零单调升高到 `K=4.0` 的约 `1044.37` bits。也就是说，强耦合同步并不是简单地让整体映射“更确定”；它同时把许多微观相位状态折叠到相似的未来同步状态，导致 degeneracy 急剧增加。EI 是二者的差，因此强同步区即便 determinism 回升，也会被更大的 degeneracy 抵消。

右图显示了为什么 $\Phi^{EID}$ 在临界附近最大。单振子口径的 degeneracy 被对每个 source 重复计算，随 $K$ 增大从 `K=1.0` 的约 `696.91` bits 快速升到 `K=4.0` 的约 `66839.89` bits；singleton-sum determinism 也在强同步区急剧放大，到 `K=4.0` 约 `66860.03` bits。两者都变大且彼此接近，说明单个振子在高同步区会获得大量共享的、重复的 whole-state 预测信息，但这些信息主要是同一个同步流形的冗余读出。临界附近则不同：联合状态仍能保留相位关系和集体模式，而单振子解释已经开始失效，所以 whole-minus-sum 差值在 `K≈1.7` 达到约 `279.63` bits。

<a id="appendix-a-2"></a>

### A.2 时间窗、相变前检测与系统规模边界

#### A.2.1 时间窗鲁棒性：避免强同步后，短窗不复现临界内部峰

基准 whole-state 曲线的 `tau=4` 结果保留为主对照。为检验其峰值是否只是高 $K$ 同步饱和造成的，新增一个严格配对的 multi-horizon Oracle sweep：对每个 seed，频率向量、均匀相位 intervention states 和 natural readout states 都固定并复用于全部 $(K,\tau)$ 条件；只改变统一的预测时间窗 $\tau\in\{0.5,0.75,1,1.5,2,4\}$，而不允许 $\tau$ 随 $K$ 自适应变化。所有条件仍使用 `N=64`、3 个 seeds、whole-state future phase target 与同一 N-source transport-map estimator。

![Paired large-N Kuramoto horizon sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep.png)

图 A 以未来 target 的 raw global order 的 $99\%$ 分位数 $R_{0.99}$ 审计强同步。预先设定 guard 为：对所有 $K$ 都要求 $R_{0.99}<0.8$。`tau=0.5` 在最强耦合 `K=4` 仍只有 $R_{0.99}=0.583$，完全通过；`tau=0.75` 为 $0.746$，也通过（仅约 $0.37\%$ target samples 的 $R\ge0.8$）。从 `tau=1` 起该 guard 开始失效：`tau=1` 仅 `K=4` 失败，`tau=1.5` 在 `K=3.2,4` 失败，`tau=2` 在 `K\ge2.6` 失败，而 `tau=4` 在 `K\ge2.2` 失败。

关键结果在图 B：**通过 guard 的两个短窗并没有给出与原图相同的临界内部峰。** `tau=0.5` 的 $\Phi^{EID}$ 从 `K=0` 的约 $0$ bits 持续升至 `K=4` 的 $229.69$ bits；`tau=0.75` 同样在 `K=4` 最大，为 $262.20$ bits。因此，在目标尚未进入强同步区的有限短时间内，耦合增强主要表现为 whole-state 联合可预测性的持续增强，而非在 $K_c\approx1.596$ 附近形成回落前的峰。随着时间窗变长，最大值才逐步向低 $K$ 移动：`tau=1` 的峰在 `K=4`（$279.54$ bits），`tau=1.5` 在 `K=3.2`（$281.00$ bits），`tau=2` 在 `K=2.6`（$280.27$ bits），配对的 `tau=4` 在 `K=1.8`（$278.92$ bits），与原 `tau=4` 图中 `K\approx1.7` 的峰一致到扫描分辨率。

因此，原始临界前沿峰的正确表述应收紧为：它是**中等有限观测时间（此处约 $\tau=4$）下**，在高 $K$ 同步吸引已压缩 whole-state 信息后出现的 whole-minus-sum 优势峰；它不是对所有预测时间窗都成立的、时间尺度无关的临界指标。短窗结果同时排除了一个较弱的替代解释：该峰并非仅由高 $K$ target 已完全同步所产生，因为在明确未强同步的 `tau=0.5,0.75` 条件下，曲线反而没有内部峰。

#### A.2.2 更长时间窗：峰位穿过而非收敛于理论 $K_c$

为直接检验“继续增大 $\tau$ 后，峰是否会停在临界相变点”的假设，保持同一配对 protocol、`N=64`、3 个 seeds 和 full-sample TM estimator，将时间窗扩展为 $\tau\in\{4,6,8,10,12\}$。扫描在转变区加密到 $K=0.8,0.9,\ldots,2.6$，并保留 $K=0,0.4,3.2,4.0$ 锚点，以区分内部峰和扫描端点峰。

![Long-horizon paired large-N Kuramoto sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_long_horizon_refined.png)

结果不支持单调收敛后固定在理论 $K_c=1.596$ 的解释。随着 $\tau$ 从 4 增至 12，$\Phi^{EID}$ 的内部峰位依次为 $K_{\rm peak}=1.8,1.6,1.5,1.4,1.3$（峰值分别为 $278.92,274.52,272.83,271.65,269.61$ bits）。因此，`tau=6` 的 $K_{\rm peak}=1.6$ 只是在当前 $0.1$ 网格上恰好贴近 $K_c$；继续增加时间窗后，峰越过 $K_c$ 并持续移向更低的 $K$，而非在 $K_c$ 停留。所有这些峰都是加密区内部点，且其 $R_{0.99}$ 仅为 $0.644,0.561,0.523,0.492,0.492$，strong fraction 均为零；故该左移不是由峰落在高 $K$ 强同步 guard 失效区造成的。

更稳妥的结论是：$K_{\rm peak}(\tau)$ 是有限时间有效信息的时间尺度依赖 crossover，可能在某一中等时间窗掠过临界区，但不能把 $\tau\to\infty$ 的峰位等同于静态 Kuramoto 临界点。长窗极限还可能受相位混合和吸引子压缩控制；若要定义渐近临界指标，需要另行研究固定有限尺寸下的长时间衰减、再做 $N\to\infty$ 的有限尺寸标度，而不能从当前峰位外推。

#### A.2.3 相变前检测：共同早期弛豫窗中的 $\Phi(\tau)$ 谱

前述长窗峰位不能直接用作预警器。为检验能否在 future target 尚未同步时识别系统的**最终动力学区间**，对全部 $K\in[0,4]$ 保留同一短时间窗，而不是为高 $K$ 自适应延长或截短 horizon。已有的 `tau=0.5,0.75` 结果与新增的 $\tau\in\{0.1,0.2,0.3,0.4,0.6\}$ 配对合并，得到共同谱 $\tau\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.75\}$。所有 $(K,\tau)$ 条件都满足 $R_{0.99}<0.8$；即使在 $K=4$、$\tau=0.75$，$R_{0.99}\approx0.75$，因此该谱只观测初始相位分布向同步吸引子弛豫的早期，而没有把已同步 target 当作特征。

![Pre-transition Kuramoto Phi-tau phase detection](../../fig/classic_network_dynamics_benchmark/large_kuramoto_pretransition_phi_tau_phase_detection.png)

图 B 显示：超临界 $K>K_c$ 条件在整个共同早期窗内已有更陡、更高的 whole-state $\Phi^{EID}(\tau)$ 谱，而此时图 A 证明其 target 尚未发生强同步。以已知的 $K_c=1.596$ 作为模拟中的超临界参考标签，只输入 7 个早期 $\Phi(\tau)$ 值，使用 leave-one-$K$-out（完整留出该 $K$ 的 3 个 seed）逻辑回归，得到超临界识别 AUROC 为 $0.983$。将每一条谱除以自身最大值、仅保留形状后，AUROC 仍为 $0.972$；因此区分力不只是 $\Phi$ 的整体幅度，时间尺度上的增长形状也携带信息。图 C 展示了留出 $K$ 后的预测概率。

##### A.2.3.1 识别算法与 AUROC 的计算

这个实验不是在单条真实轨迹上拟合未来标签，而是一个受控的 Oracle 可辨识性检验。数据单位是一个固定耦合和随机 seed 的组合 $(K,s)$。共有 17 个 $K$ 值、3 个 seed，因此有 $17\times3=51$ 个样本。对每个样本，先从同一 seed 的均匀初始相位 intervention support 出发，分别积分到 7 个早期 horizon，并计算 whole-state N-source 指标。输入特征向量为

$$
\mathbf{x}_{K,s}=
\left[
\Phi^{EID}_{K,s}(0.1),
\Phi^{EID}_{K,s}(0.2),
\Phi^{EID}_{K,s}(0.3),
\Phi^{EID}_{K,s}(0.4),
\Phi^{EID}_{K,s}(0.5),
\Phi^{EID}_{K,s}(0.6),
\Phi^{EID}_{K,s}(0.75)
\right].
$$

这里的每个 $\Phi^{EID}_{K,s}(\tau)$ 都是同一 whole-state 目标和同一 N-source transport-map estimator 下的

$$
\Phi^{EID}=EI_{\mathrm{do}}(\mathbf{S};\mathbf{Y}_{\tau})
-\sum_{i=1}^{64}EI_{\mathrm{do}}(\mathbf{s}_i;\mathbf{Y}_{\tau}),
$$

其中 $\mathbf{S}$ 是 64 个振子的联合当前相位特征，$\mathbf{s}_i$ 是第 $i$ 个振子的二维相位特征，$\mathbf{Y}_{\tau}$ 是 $\tau$ 后的 128 维 whole-state phase target。保留一个特征前，先审计自然 readout target 的 $R_{0.99}$；只有本实验中全部 $51$ 个样本都满足 $R_{0.99}<0.8$ 的共同 horizon 才进入上式。故模型没有看到已经强同步的 future target。

二分类标签不由 $\Phi$、早期 $R$ 或长时间 $R$ 阈值产生，而是由生成模型中已知的理论边界独立给出：

$$
y_K=\mathbb{I}(K>K_c),\qquad K_c=1.595769\ldots .
$$

这样标签表示“若继续演化，该参数属于超临界动力学区间”，而不是声称有限 $N$ 系统在一个任意 order 阈值处发生严格相变。每个样本另计算到 $\tau=20$ 的 raw order，作为连续审计量，但它不参与标签和分类器训练。

评估采用真正的 leave-one-$K$-out（LOKO）流程。对每个待测耦合 $K_*$：

1. 从训练集删除 $K_*$ 的全部 3 个 seed，只用其余 $16\times3=48$ 个样本。
2. 仅在这 48 个训练样本上，对每个特征维度计算均值 $\mu_j^{\mathrm{train}}$ 和标准差 $\sigma_j^{\mathrm{train}}$，并做标准化：

   $$
   \widetilde{x}_{ij}=\frac{x_{ij}-\mu_j^{\mathrm{train}}}{\sigma_j^{\mathrm{train}}}.
   $$

3. 在标准化后的训练集拟合固定正则强度 $C=1$ 的 logistic regression：

   $$
   \widehat p_{K,s}=\sigma\left(b+\mathbf{w}^{\mathsf T}\widetilde{\mathbf{x}}_{K,s}\right),
   \qquad
   \sigma(z)=\frac{1}{1+e^{-z}}.
   $$

4. 用该模型预测被完整留出的 3 个 $(K_*,s)$ 样本；遍历 17 个 $K_*$ 后，得到 51 个没有使用自身 $K$ 训练过的预测概率 $\widehat p_{K,s}$。

AUROC 不取某一个分类阈值，而检验这些概率是否把超临界样本整体排在次临界样本之前。令 $\mathcal{P}$ 是 24 个正类样本（8 个超临界 $K$、每个 3 个 seed），$\mathcal{N}$ 是 27 个负类样本（9 个次临界 $K$、每个 3 个 seed），则

$$
\operatorname{AUROC}
=
\frac{1}{|\mathcal{P}|\,|\mathcal{N}|}
\sum_{p\in\mathcal{P}}\sum_{n\in\mathcal{N}}
\left[
\mathbb{I}(\widehat p_p>\widehat p_n)
+\frac{1}{2}\mathbb{I}(\widehat p_p=\widehat p_n)
\right].
$$

本结果的 raw-spectrum AUROC 为 $0.9830247$，即 648 个正负样本对中有 637 对被正确排序（无并列时为 $637/648$）。shape-only 版本先将每个样本的谱除以该谱的最大值，再重复完全相同的 LOKO 流程；其 AUROC 为 $0.9722222$，即 630/648 对正确排序。后者是“谱形仍可分”的证据，而不是额外使用了 $K$、order parameter 或未来同步状态。

这里应把次临界状态称为**去相干／次临界动力学**，而不是默认称为“混沌相”：经典全局耦合 Kuramoto 的 $K<K_c$ 解一般可以是非同步的准周期运动，但不由本实验自动证明为严格混沌。有限 $N$ 下同步是 crossover，长时间 raw order 因而保留为连续审计量而未被任意阈值二分。当前结果的含义是：在这个已知方程、已知 $K_c$ 的 Oracle setting 中，早期 $\Phi(\tau)$ 谱可以预报未来进入超临界区；要转化为真实观测数据的预警器，仍需在未知参数、噪声、部分观测和独立时变轨迹上重新校准。

#### A.2.4 系统规模边界：只有大系统提供临界峰参照

![Kuramoto oscillator-count appendix](../../fig/part1_kuramoto_size_phi_eid_appendix.png)

该对照只展示 Oracle $\Phi^{EID}$ 与 corrected order，不再混入学习模型读出。**小 $N=2$ classic Kuramoto。** 在相同 whole-state 口径下，Oracle $\Phi^{EID}$ 没有形成清楚的内部临界峰；它在当前扫描范围内主要随强耦合增强，到 `K=4.0` 约为 `0.96` bits。`N=2` 的 corrected order 也不是热力学意义下的相变曲线，而是有限二振子锁相读数。

**大 $N=64$ classic Kuramoto。** 在完全相同的方程、source partition、whole-state target 和 $\Phi^{EID}$ 公式下，corrected global order 从低 $K$ 的近零状态进入高 $K$ 同步饱和区。理论临界耦合为 $K_c\approx1.596$；有限时间读出下最大斜率出现在 `K=2.2`。对应 Oracle N-source $\Phi^{EID}$ 从 `K=0` 的 `0` bits 升高，在 `K=1.7` 达峰，约 `279.63` bits；随后进入强同步区后明显回落，`K=4.0` 约 `13.58` bits。

这个边界对照说明，在方程形式、source/target 和 EI 分解公式都固定后，是否出现临界内部峰主要取决于系统规模。`N=2` 没有经典 Kuramoto 的热力学同步相变，所以不能期待它给出与大系统相同的 $\Phi^{EID}$ 峰；`N=64` 才提供清晰的 order-parameter 转变参照。

因此，Kuramoto 临界相变实验的核心证据链是三步：order parameter 给出同步转变区，whole-state $\Phi^{EID}$ 在转变前沿形成峰值，determinism/degeneracy 分解说明该峰来自“联合相位构型仍可区分、单振子读出快速冗余化”的差异，而不是来自总 EI、determinism 或 degeneracy 任一单项的简单最大化。

<a id="appendix-b"></a>

## 附录 B：83 ROI DMF 动力学方程

本附录给出第 1 节主确认实际积分的 83 ROI dynamic mean-field（DMF）方程，而非另行定义一个模型。令 $N=83$，兴奋性和抑制性 NMDA 门控状态分别为 $\mathbf{s}_E(t),\mathbf{s}_I(t)\in\mathbb{R}^{N}$；$\mathbf{C}\in\mathbb{R}^{N\times N}$ 为原始结构连接矩阵（行表示受体 ROI，列表示源 ROI），$\mathbf{j}^{\mathrm{FIC}}(G)\in\mathbb{R}^{N}$ 为该耦合值对应的 JFIC 向量。除矩阵乘法外，下式中的向量乘积、除法和函数均逐元素执行，$\mathbf{1}$ 为全 1 向量。

### B.1 局部电流与输入输出函数

兴奋性、抑制性群体的输入电流为

$$
\begin{aligned}
\mathbf{I}_E
&=w_E I_0\mathbf{1}
+w_+J_{\mathrm{NMDA}}\mathbf{s}_E
+GJ_{\mathrm{NMDA}}\mathbf{C}\mathbf{s}_E
-\mathbf{j}^{\mathrm{FIC}}(G)\odot\mathbf{s}_I,\\
\mathbf{I}_I
&=w_I I_0\mathbf{1}
+J_{\mathrm{NMDA}}\mathbf{s}_E
-\mathbf{s}_I.
\end{aligned}
$$

因此，主实验使用的是**直接**长程兴奋输入 $\mathbf{C}\mathbf{s}_E$，而不是扩散型 $\mathbf{C}\mathbf{s}_E-\operatorname{diag}(\mathbf{C}\mathbf{1})\mathbf{s}_E$；连接矩阵也没有做行归一化。兴奋性与抑制性放电率写为

$$
\mathbf{r}_E=f_{a_E,b_E,d_E}(\mathbf{I}_E),\qquad
\mathbf{r}_I=f_{a_I,b_I,d_I}(\mathbf{I}_I),
$$

其中

$$
f_{a,b,d}(x)=
\frac{a(x-b)}{1-\exp\!\left[-d\,a(x-b)\right]},
\qquad
f_{a,b,d}(b)=\frac{1}{d},
$$

后一个值是分母数值接近零时采用的连续极限。参数为

| 参数 | 数值 |
|---|---:|
| $w_E,w_I,I_0,w_+,J_{\mathrm{NMDA}}$ | $1.0,\ 0.7,\ 0.382,\ 1.4,\ 0.15$ |
| $a_E,b_E,d_E$ | $310,\ 0.403,\ 0.16$ |
| $a_I,b_I,d_I$ | $615,\ 0.288,\ 0.087$ |
| $\tau_E,\tau_I,\gamma_E,\sigma$ | $0.100,\ 0.010,\ 0.641,\ 0.01$ |

### B.2 随机 Euler--Maruyama 更新

第 $k$ 个积分步以 $\Delta t=0.001$ 更新为

$$
\begin{aligned}
\mathbf{s}_E^{k+1}
&=\mathbf{s}_E^k
+\Delta t\left[-\frac{\mathbf{s}_E^k}{\tau_E}
+(\mathbf{1}-\mathbf{s}_E^k)\odot\gamma_E\mathbf{r}_E^k\right]
+\sigma\sqrt{\Delta t}\,\boldsymbol{\xi}_E^k,\\
\mathbf{s}_I^{k+1}
&=\mathbf{s}_I^k
+\Delta t\left[-\frac{\mathbf{s}_I^k}{\tau_I}+\mathbf{r}_I^k\right]
+\sigma\sqrt{\Delta t}\,\boldsymbol{\xi}_I^k,
\end{aligned}
$$

其中 $\boldsymbol{\xi}_E^k,\boldsymbol{\xi}_I^k\overset{\mathrm{iid}}{\sim}\mathcal{N}(\mathbf{0},\mathbf{I})$，并在群体、ROI 与时间步之间独立。第 1 节的 EI 干预从

$$
\mathbf{s}_E^0,\mathbf{s}_I^0\overset{\mathrm{ind}}{\sim}U(0.30,0.70)^{83}
$$

开始，随后按上式积分 300 步；完整的 $[\mathbf{s}_E^{300},\mathbf{s}_I^{300}]\in\mathbb{R}^{166}$ 是 target。主确认设置 `state_boundary=none`，因此更新后**不**将任何状态裁剪回 $[0,1]$。这与初始干预支持 $[0.30,0.70]$ 的选择是两件事：前者规定最大熵外生初态，后者规定后续随机动力学不施加硬边界。

<a id="appendix-c"></a>

## 附录 C：$\Phi^R$ 的相变近似复现

![DMF 中 $\Phi^R$ 的相变近似复现](../../fig/dmf_phi_r_phase_reproduction.png)

*图 4. 参考文献 Fig. 6 中 $\Phi^R$ 相变检测的散点式近似复现。A：平均发放率；B：对每个脑区对的滞后样本计算 $\Phi^R$ 后的跨 pair 均值。每个点是一个耦合强度下的读数，实线仅连接相邻点以帮助辨认曲线形状。灰带标出发放率快速上升的 $G=1.7\text{-}1.9$ 区间，竖虚线为本扫描中平均发放率变化率最大的 $G=1.8$。该近似复现使用 Lausanne-33 count 派生的 83 区结构连接，不是原文未公开的 HCP Lausanne-83 精确矩阵；因此它验证曲线现象而不是对原图的数值复制。*

在 $G>1.0$ 的分析区间，$\Phi^R$ 在 $G=1.8$ 达到 $0.02168$，与平均发放率开始快速上升的位置重合。因此，该近似复现能够重现“$\Phi^R$ 在动力学转折附近达峰”的定性结果。该读数来自自发轨迹的经验滞后联合分布，并在脑区对之间取均值；它描述的是相变附近的 pairwise information dynamics，而不是独立干预下的系统级机制量。

由于 $\Phi^R$ 的概率口径由经验采样分布决定，改变时间窗、重采样权重或状态筛选规则时，曲线和峰位可能改变。加之精确 HCP Lausanne-83 结构连接未公开，这一结果应表述为对原文散点形状和峰位关系的近似复现，而不应表述为对其绝对数值或人体脑临界耦合常数的精确重现。

#### $\Phi^{EID}$ 相对 $\Phi^R$ 突出的信息

参考文献使用经验联合分布 $p(\mathbf{X}_t,\mathbf{X}_{t+1})$，对所有脑区对计算并平均 $\Phi^R$；原文将它解释为 synergy 与 transfer 的非负总和，并在平均发放率于 $G\approx2$ 快速上升处观察到峰值（Mediano et al., 2025，Zotero `26Q48H8Y`，全文）。当前 $\Phi^{EID}$ 实验回答的是相邻但不同的问题：在预先声明的最大熵干预下，完整 166 维当前状态对完整未来状态产生了多少不能由单一状态变量 EI 相加解释的联合影响（PEID，Zotero `MYATYWAJ`，全文）。

| 比较维度 | 文献中的 $\Phi^R$ | 当前 $\Phi^{EID}$ | $\Phi^{EID}$ 的实际增益 |
|---|---|---|---|
| 概率口径 | 自发轨迹的经验状态分布 | 声明支持集上的独立最大熵干预 | 更接近对固定动力学机制的探测，减少驻留频率和源相关结构对数值的直接影响 |
| 系统范围 | 每个脑区对的二元 $\Phi$，再跨 pair 平均 | 全部 166 个门控变量到完整未来状态的 whole-minus-sum | 直接检验全系统高阶联合效应，不必把全脑结论建立在 pair 平均上 |
| 可定位性 | 在完整 $\Phi$ID 中可区分多种源—靶信息流，但本文 Fig. 6 只报告 pairwise $\Phi^R$ 汇总 | 可按 ROI、模块或层级分区继续拆分 source-side synergy | 可把临界峰追溯到候选模块或高阶 source 集合 |
| 非负性 | 修正双重冗余后非负 | 在因子化干预的离散理论中等于条件 total correlation，因而非负 | **不是相对 $\Phi^R$ 的独有优势**；两者都解决了 $\Phi^{WMS}<0$ 的解释问题 |

因此，最稳妥的贡献表述不是“$\Phi^{EID}$ 比 $\Phi^R$ 更准确”，而是：**$\Phi^R$ 给出经验脑活动在相变附近的 pairwise synergy-plus-transfer 峰，$\Phi^{EID}$ 则给出在统一干预基准下的全系统机制协同峰。**后者的优势是机制归一化、全系统高阶性和后续层级分解；代价是结果依赖干预支持、动力学模型与高维估计器。与此同时，$\Phi^{EID}$ 只聚合 source-side synergy，不能替代完整 $\Phi$ID 对 Syn$\rightarrow$Syn、Syn$\rightarrow$Un 等靶侧编码模式的区分。

<a id="appendix-d"></a>

## 附录 D：83 ROI 对平均 $\Phi^{EID}$ 曲线

该曲线已并入图 1C：所有 3403 个无序 ROI 对的 $\Phi^{EID}$ 在每个 seed 内取均值后，再跨 8 个 seed 平均；阴影为 seed 均值的标准误。

令 ROI $r$ 的当前 E/I 状态为 $\mathbf{x}_{r,t}=(s_{E,r,t},s_{I,r,t})^\mathsf{T}$。对每个无序 ROI 对 $r<q$，source 与 target 都只保留该对的两个 ROI：$\mathbf{x}_{rq,t}=(\mathbf{x}_{r,t},\mathbf{x}_{q,t})$，$\mathbf{y}_{rq,t+\tau}=(\mathbf{x}_{r,t+\tau},\mathbf{x}_{q,t+\tau})$。该 ROI 对的 source-side 指标定义为

$$
\phi^{EID}_{rq}
=EI_{\mathrm{do}}(\mathbf{x}_{rq,t};\mathbf{y}_{rq,t+\tau})
-EI_{\mathrm{do}}(\mathbf{x}_{r,t};\mathbf{y}_{rq,t+\tau})
-EI_{\mathrm{do}}(\mathbf{x}_{q,t};\mathbf{y}_{rq,t+\tau}).
$$

曲线量是 83 个 ROI 的全部 $\binom{83}{2}=3403$ 个无序对的平均：

$$
\overline{\Phi}^{EID}_{\mathrm{pair}}(G)
=\frac{1}{3403}\sum_{r<q}\phi^{EID}_{rq}(G).
$$

每个条件使用独立 $U(0.30,0.70)^{166}$ 初态、2048 个干预样本、400 个 Euler 步、直接长程耦合、无状态裁剪、同一 Gaussian EI 估计器和与主扫描一致的 JFIC 日程；seed 为 3--10。图 1C 中的 pair-average 从低耦合端上升，在 $G=1.50\text{-}1.55$ 附近达到局部最大值后向高耦合端回落。跨 seed 峰位为 $G=1.531\pm0.009$。在完全匹配的 400 步全系统对照中，pair-average 的半峰全宽为 $0.442\pm0.011$，全系统为 $0.469\pm0.006$，配对差为 $-0.027\pm0.012$。因此“更窄”有数值支持，但幅度有限，不能描述成数量级上的尖锐化。

这一轻度尖锐化首先是**尺度选择**，而不是 pairwise 指标天然更准确。每个 pair 指标只保留两个 ROI 的 4 维 source--target 子系统，并减去两个单 ROI 对同一未来 pair 的 EI；它优先读出“两个 ROI 必须联合出现”才能提供的局部增量。低耦合时跨区联合效应尚弱；进入转折附近时，许多 ROI 对获得额外联合预测力；继续增大 $G$ 后，单 ROI 已能携带更多共同模态预测信息，pair 联合项相对两个单 ROI 基线的增量随之下降。与此一致，正 pair 比例从 $G=1.55$ 的 0.742 降至 $G=2.0$ 的 0.679，并在 $G=3.0$ 降至 0.563。

第二，3403 对取均值会压低个别 pair 的波动，使共同的局部耦合窗口更突出；但逐 pair 峰位本身分散，不能把均值曲线的尖峰解释成“所有 pair 在同一点同步达峰”。第三，全系统 $\Phi^{EID}$ 保留任意多 ROI 的高阶联合项，这些不同尺度的贡献可以覆盖更宽的 $G$ 区间，从而形成相对宽的平台。该解释与本地文献的边界一致：pairwise $\Phi$ID 属于局部信息动力学，不能替代 whole-system 分析（Menesse et al., 2024，Zotero `74HJ34NV`，全文）；高阶 synergy 则是整体提供、但较小预测子集合不能单独提供的信息（Mediano et al., 2022，Zotero `B9JQZSRU`，全文）。

图 1C 与图 1A 右轴的完整 $\Phi^{EID}$ 不是同一个量的不同可视化。图 1A 右轴的 source 是全部 166 个 E/I 标量，target 是完整未来 166 维状态，且从联合 EI 中扣除每一个单标量源的 EI；因此它保留了任意多 ROI 的联合项。图 1C 则只看每次两个 ROI 的 4 维 source--target 子系统，并以一个完整 ROI 的二元 E/I 块作为单独 source 基线。它既不构成图 1A 右轴的加和分解，也不能捕获需要三个或更多 ROI 才出现的协同；两者的 bits 绝对值不应直接比较。

<a id="dmf-hierarchy"></a>
