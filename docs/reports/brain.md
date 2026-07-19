# 脑科学实验：83 ROI 临界识别与 HCP Yeo7 $\Phi$/$\Xi$ 分解

## 结论

本报告保留三个互补结果块。

1. **83 ROI 临界相变识别。**在 83 区 DMF 的近似复现中，经验滞后分布上的 pairwise $\Phi^R$ 在 $G=1.8$ 达峰（$0.02168$），与平均发放率快速上升区重合，复现了原文的散点式相变识别关系。作为互补的机制分析，无裁剪 8-seed 最大熵干预在相近区域识别到 $\Phi^{EID}$ 宽峰：$G=1.7$ 的均值为 $12.384\pm0.041$ bits，$G=1.6\text{-}1.8$ 的临界窗显著高于前、后窗口。Yeo-7 跟进分解显示临界窗中 68.7% 的 $\Phi^{EID}$ 来自跨 ROI 协同，31.3% 来自单个 ROI 内的兴奋—抑制协同。
2. **HCP500/1000 PCA–Yeo7 Phi 分解。**在相同 30 名 HCP REST1_LR 被试中，Schaefer-500（$p=8,\alpha=10$）与重新验证的 Schaefer-1000（$p=5,\alpha=1$）均在 30/30 名被试中高于独立 PC1 circular-shift null。两种粒度的全七网络核都常进入 top-3，但不高于 matched null；相对地，缺少 Limbic 的六网络核均高于 matched null cohort（500：17/30 对 8.65/30；1000：12/30 对 6.35/30；各 20-null 未校正 $p=0.047619$）。
3. **HCP500 任务诱发表征与 $\Xi$ 层级分解。**任务态先在 `taskRetained-taskRegressed` 上拟合各 Yeo7 网络的 PC1，再用同一载荷投影 `taskRetained`；REST 则在自身时序上拟合并投影。最终共享配置为一维网络状态、三阶历史与 Ridge $\alpha=1$，即 21 维 source 预测 7 维下一时刻 target。29 名共同被试中，REST 的 system-level $\Xi$ 均值为 7.040 bits，高于七任务的 4.301--5.537 bits；七项配对 Wilcoxon 检验经 BH 校正后均显著。除总量外，七网络归因份额在仅比较任务态时 7/7 个网络均有显著状态效应；网络分布的 21 个任务两两对比中 18 个显著，greedy 层级 atom 分布中 11 个显著。该结果说明 REST 与任务态、以及多数任务对之间不仅整体协同强度不同，协同在单网络内部与跨网络组合之间的分配也不同。

这些实验分别回答不同问题：DMF 实验检验 $\Phi^{EID}$ 是否能定位可控模型中的临界动力学带；HCP 静息态实验检验降维后的真实网络动力学中是否存在高于同步破坏 null 的跨网络高阶结构；任务态实验检验以任务诱发 PCA 选择观测方向后，system-level $\Xi$ 及其网络和层级组合归因是否随状态改变。它们不构成对特定脑机制、因果方向或唯一稀疏 atom 的证明。

## 目录

1. [**83 ROI DMF：临界相变识别**](#dmf-critical)
   1. [全状态最大熵干预与临界判据](#dmf-intervention)
   2. [临界窗中的 Φ^EID 峰与 EI 形状对齐](#dmf-phi)
   3. [Φ^R 的相变复现与经验分布边界](#dmf-phi-r)
   4. [临界窗 Φ^EID 的 ROI—结构模块层级分解](#dmf-hierarchy)
   5. [Yeo-7 功能网络层级分解](#dmf-yeo7)
   6. [确定性 / 简并性分解与变化率](#dmf-detdeg)
   7. [解释边界](#dmf-limits)
2. [**HCP500 任务态 $\Phi$、$\Xi$ 与脑区分布**](#hcp-wm)
   1. [协同核分布及静息态对照](#hcp-wm-phi)
   2. [七任务 raw Phi 排序（历史表征参照）](#hcp-all-tasks)
   3. [七任务态的 Schaefer-500 任务特异脑区分布](#hcp-task-specific-regions)
3. [**讨论：解释边界与可复现性**](#discussion)
   1. [结论的适用范围](#discussion-scope)
   2. [结果与图形产物](#discussion-artifacts)
4. [**附录 A：Kuramoto 振子数与 whole-state Φ^EID 曲线形状**](#appendix-a)
   1. [临界峰的 EI 与 effectiveness 机制](#appendix-a-1)
   2. [时间窗、相变前检测与系统规模边界](#appendix-a-2)
5. [**附录 B：83 ROI DMF 动力学方程**](#appendix-b)
6. [**附录 C：Φ^R 的相变近似复现**](#appendix-c)
7. [**附录 D：83 ROI 对平均 Φ^EID 曲线**](#appendix-d)
8. [**附录 E：拟合模型参数鲁棒性**](#appendix-e)
    1. [REST–任务差异的 p–α 鲁棒性](#appendix-e-1)
    2. [留出预测误差解释弱正则反转](#appendix-e-2)
9. [**附录 F：HCP500 PCA–Yeo7 Phi–null 分解**](#appendix-f)
   1. [数据、降维与动力学表征](#appendix-f-1)
   2. [History-source Φ^EID 与 circular-shift null](#appendix-f-2)
   3. [Yeo7 模块历史分解](#appendix-f-3)
10. [**附录 G：HCP1000 PCA–Yeo7 Phi–null 分解**](#appendix-g)
    1. [数据、降维与模型选择](#appendix-g-1)
    2. [History-source Φ^EID 与 circular-shift null](#appendix-g-2)
    3. [Yeo7 模块历史分解与 500 对照](#appendix-g-3)

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


<a id="hcp-wm"></a>

## 2. HCP500 任务态 $\Phi$、$\Xi$ 与脑区分布

<a id="hcp-wm-phi"></a>


### 2.1 协同核分布及静息态对照

静息态 Schaefer-500 的表征、模型选择、Phi–null 对照和模块分解见附录 F；Schaefer-1000 的对应复现见附录 G。本节只保留任务态迁移及其与静息态基线的对照。

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

### 2.2 七任务 raw Phi 排序（历史表征参照）

作为原始表征参照，为检验静息态 raw $\Phi^{EID}$ 较高的现象是否只出现在 WM，这里将同一 Schaefer-500 协议扩展到 EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL 和 WM。跨任务固定 Yeo7-PC1 表征形式、八阶 $\Delta$-Ridge、$p=8$、$\alpha=10$、56 维历史 source、7 维 target 和 Gaussian log-det 估计器；但每个“任务 $\times$ 被试”都使用自己的前 75% 时间点重新拟合 PCA、标准化、Ridge 系数、截距和残差协方差。所谓“固定模型”因此指结构与超参数固定，不是跨任务复用同一组系数。本轮只比较 observed raw $\Phi^{EID}$，不生成新的 circular-shift null。第 2.3 节的任务诱发 PCA–$\Xi$ 分解是当前任务空间归因主结果。

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

八条件重复测量的 Friedman 检验为 $\chi^2=49.919540$、$p=1.50\times10^{-8}$。REST 与每个任务的受试者内差值均为负，即任务 raw Phi 低于 REST；七项 paired sign-flip 的 Holm 校正 $p$ 从 $3.50\times10^{-5}$ 到 $8.45\times10^{-4}$，全部低于 0.05。与 REST 最接近的是 MOTOR，task-minus-REST 均值差仍为 $-1.025316$ bits（95% bootstrap CI $[-1.541913,-0.491539]$，Holm $p=0.000845$）；差距最大的是 EMOTION，为 $-1.739087$ bits（95% CI $[-2.296265,-1.216161]$，Holm $p=3.50\times10^{-5}$）。因此在预先使用的 $p=8,\alpha=10$ 配置下，**REST 的群体平均 $\Phi^{EID}$ 显著高于全部七个任务态。**附录 E 进一步检验该排序对超参数的依赖，结论不能外推为与 $p,\alpha$ 无关的普遍规律。

该排序不是逐人定律。REST 在 20/29 名共同被试中是八条件最大值；其余 9 人的最大值分别为 MOTOR 3 人、SOCIAL 2 人、WM 2 人、GAMBLING 1 人和 RELATIONAL 1 人。REST 的降序中位排名为第 1，但平均排名为 2.17。质量规则只标记 `sub-103515/WM` 的极端早期 PC1 瞬变，主分析仍保留该点。由于本轮未计算各任务 null，结论只针对 raw estimator 输出，不能进一步断言 REST 相对于各任务自身时间结构具有更高的 null 校正协同。

<a id="hcp-length-matched-variance"></a>

#### 2.2.1 REST–任务长度匹配方差检验

为检验 REST 的跨被试离散度是否只是由 1200 点长序列造成，这里对每个任务长度分别从 REST1_LR 取 12 个覆盖完整 run 的等距窗口：EMOTION 176 点、GAMBLING 253 点、LANGUAGE 316 点、MOTOR 284 点、RELATIONAL 232 点、SOCIAL 274 点和 WM 405 点。每个窗口都独立使用前 75% 时间点重新拟合 Yeo7 PC1、标准化、$p=8,\alpha=10$ 的 $\Delta$-Ridge 和残差协方差；任务态复用第 2.2 节相同估计器的 raw $\Phi^{EID}$。因此唯一主动改变的分析因素是 REST 序列长度，不使用 null 模型。

![REST 与七任务的长度匹配 raw Phi 方差](../../results/hcp_schaefer500_length_matched_variance/length_matched_variance.png)

七个小图分别给出 REST 与一个任务的两两配对分布。每个蓝点是该被试 12 个等长 REST 窗口 raw $\Phi^{EID}$ 的均值，橙点是同一被试的任务值，灰线连接同一被试；箱线图和面板内 SD 直接显示两组跨被试离散度。所有小图共享同一纵轴，因此可与第 2.2 节的原始分布图直接比较。图用于展示稳定的被试级 REST 值；下表的正式方差比仍对 12 个窗口位置分别计算跨被试方差后取平均，避免只看窗口均值掩盖时间位置变化。

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

<a id="hcp-task-specific-regions"></a>

### 2.3 七任务态的 Schaefer-500 任务特异脑区分布

前述 EI/$\Phi$ 分析回答的是网络历史状态如何联合预测未来，而不是经典任务定位问题。它把 500 个 parcel 压缩成七个网络 PC1，逐状态标准化后拟合整段平稳一步动力学；这个流程主动移除了均值和整体幅度，忽略任务事件时序，也平均掉同一 Yeo7 网络内部的 parcel 异质性，因此不直接用于解释任务态之间的脑区分布差异。

为直接回答“不同任务主要落在哪些脑区”，这里改用数据中已有的成对 `Schaefer500_taskRetained` 与 `Schaefer500_taskRegressed` 时序，并保留全部 500 个 cortical parcels。该分析覆盖具有全部七个 LR 任务的 30 名被试；EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL 和 WM 分别有 176、253、316、284、232、274 和 405 个时间点。

#### Parcel 级任务诱发方差比例

对被试 $s$、任务 $c$、parcel $i$，先分别去除两条时序的时间均值：

$$
\begin{aligned}
r_{sci}(t)
&=x^{\mathrm{retained}}_{sci}(t)
-\overline{x}^{\mathrm{retained}}_{sci},\\
e_{sci}(t)
&=x^{\mathrm{regressed}}_{sci}(t)
-\overline{x}^{\mathrm{regressed}}_{sci}.
\end{aligned}
$$

两者的差

$$
u_{sci}(t)=r_{sci}(t)-e_{sci}(t)
$$

是被任务回归移除的拟合任务成分。定义 parcel 级任务诱发方差比例（task-evoked variance fraction, TEVF）

$$
f_{sci}
=\frac{\sum_t u_{sci}(t)^2}
{\sum_t r_{sci}(t)^2}.
$$

若 `taskRegressed` 是普通最小二乘 GLM 的残差，则 $e_{sci}$ 与 $u_{sci}$ 正交，因而

$$
\sum_t r_{sci}(t)^2
=\sum_t e_{sci}(t)^2+\sum_tu_{sci}(t)^2,
$$

并有 $0\leq f_{sci}\leq1$。数据验证与这一结构一致：全部 210 个“被试 $\times$ 任务”拟合中，parcel 级 residual--task component 绝对相关的跨拟合中位数为 $4.63\times10^{-9}$，最大值为 $2.43\times10^{-7}$；未截断比例范围为 0.0003--0.8905，重构误差为 0。因此这里不是从两个任意时序作差，而是直接恢复 task GLM 所解释的 parcel 级时间能量。

七任务跨被试和 parcel 的平均 TEVF 依次为 13.74%、14.22%、17.25%、30.39%、18.60%、21.26% 和 28.62%。MOTOR 与 WM 的整体任务解释比例最高，但整体强度仍不等于空间特异性。

#### 消除整体幅度后的任务特异空间分布

为避免把某任务整体更强误读为某些脑区更特异，对每个被试的每张 TEVF 图按其 500-parcel 均值归一化：

$$
q_{sci}=\frac{f_{sci}}
{500^{-1}\sum_{j=1}^{500}f_{scj}},
\qquad
\frac{1}{500}\sum_{i=1}^{500}q_{sci}=1.
$$

任务 $c$ 相对其余六任务的 parcel 特异性定义为

$$
d_{sci}
=q_{sci}
-\frac{1}{6}\sum_{c'\neq c}q_{sc'i}.
$$

该对比同时满足

$$
\sum_{i=1}^{500}d_{sci}=0,
\qquad
\sum_{c=1}^{7}d_{sci}=0,
$$

所以只保留相对空间形状。实现中的最大数值守恒误差为 $1.15\times10^{-14}$。网络层结果只是对同一 $f$ 或 $d$ 在 Yeo7 parcel 集内取均值，没有再次拟合 PCA 或动力学模型。

![七任务 Schaefer-500 任务诱发方差与任务特异空间分布](../../results/hcp_schaefer500_task_specific_regions/task_evoked_region_profiles.png)

图 a 显示绝对 TEVF：MOTOR 和 WM 整体较强，但各任务仍有不同 parcel 条带。图 c 去除每张图的整体幅度并减去其余任务后，任务分布明显分离。网络均值对比（图 d）显示：EMOTION 主要富集于 Vis（$+0.424$）；LANGUAGE 富集于 Cont（$+0.418$）和 Default（$+0.319$），同时相对缺失 Vis（$-0.876$）；MOTOR 富集于 SomMot（$+0.462$）、SalVentAttn（$+0.220$）和 Limbic（$+0.174$）；RELATIONAL 富集于 Vis（$+0.603$）和 DorsAttn（$+0.219$）；SOCIAL 最突出的是 DorsAttn（$+0.729$）。GAMBLING 和 WM 的七网络均值差异较弱，说明它们的可辨识信息更多位于网络内部 parcel 模式，而不是粗粒度网络总量。

每个任务的最高特异 parcel 进一步定位了网络内部结构：EMOTION 为 `RH_Vis_4`，GAMBLING 为 `RH_Vis_21`，LANGUAGE 为 `LH_Default_Temp_5`，MOTOR 为 `LH_SomMot_19`，RELATIONAL 为 `RH_Vis_21`，SOCIAL 为 `RH_Vis_23`，WM 为 `RH_Vis_5`。完整 top-10 排名及数值见该实验的 `summary.json` 与 `report.md`。这些 Schaefer 名称是 atlas parcel 标识，不应在缺少独立定位或统计阈值时进一步替换成更具体的功能解剖标签。

#### REST 与七任务的共同方差口径

REST 没有 `taskRetained`/`taskRegressed` 配对，因此不存在“被 task GLM 解释的方差”，不能在上一张 TEVF 图中把 REST 人为设为零。为将 REST 与七任务置于同一可计算口径，这里另用每个状态的 `taskRetained` 或 REST 原始时序计算 parcel 时间方差

$$
v_{sci}
=\frac{1}{T_c-1}
\sum_{t=1}^{T_c}
\left(x_{sci}(t)-\overline{x}_{sci}\right)^2,
$$

再在每个被试、每个状态内除以 500 个 parcel 的平均方差：

$$
q^{\mathrm{var}}_{sci}
=\frac{v_{sci}}
{500^{-1}\sum_{j=1}^{500}v_{scj}},
\qquad
\frac{1}{500}\sum_{i=1}^{500}q^{\mathrm{var}}_{sci}=1.
$$

因此该量只比较一个 run 内方差如何分布到不同 parcel，不比较 REST 与任务的绝对 BOLD 方差大小，也不受全局乘性尺度影响。八状态的状态特异对比为

$$
d^{\mathrm{var}}_{sci}
=q^{\mathrm{var}}_{sci}
-\frac{1}{7}\sum_{c'\neq c}q^{\mathrm{var}}_{sc'i}.
$$

REST 与全部七任务共有 29 名被试。图中白色或黑色横线将 REST 与任务分开；四个 panel 与上一图保持相同的 parcel、Yeo7 和状态减其余状态结构，但颜色不再表示 TEVF。

![REST 与七任务的 Schaefer-500 时间方差空间分布](../../results/hcp_schaefer500_task_specific_regions/rest_all_tasks_variance_profiles.png)

REST 的 Yeo7 方差富集以 Limbic（1.38）、Vis（1.26）、Default（1.22）较高；相对其余七状态，REST 最突出的是 Limbic（$+0.20$）和 Default（$+0.07$），相对较低的是 DorsAttn（$-0.13$）和 Vis（$-0.10$）。任务对比仍保留预期结构，例如 LANGUAGE 相对富集 Cont（$+0.20$），MOTOR 相对富集 Limbic（$+0.25$）和 SomMot（$+0.12$），RELATIONAL 相对富集 Vis（$+0.31$），SOCIAL 相对富集 DorsAttn（$+0.24$），WM 相对富集 Vis（$+0.21$）。

对同一方差富集图执行八状态 LOSO 最近质心分类，500-parcel 准确率为 58.6%（136/232；chance 12.5%；2,000 次被试内置换 $p=0.0005$），Yeo7 网络均值准确率为 37.9%（88/232，$p=0.0005$）。REST 的召回率为 44.8%（13/29）。该结果低于七任务 TEVF 的 90.0%，符合预期：TEVF 有针对性地隔离了 task GLM 成分，而共同方差口径同时包含自发活动、任务活动和残余混杂；它的用途是为 REST 提供不造假的空间参考，而不是替代 TEVF。

#### 任务诱发 PCA–$\Xi$ 网络归因与层级分解

为检验任务诱发方向是否改变下一时刻整合信息的空间构成，这里在每个 Yeo7 网络内先提取任务诱发 PCA 方向。对任务态，PCA 只在前 75% 时间点的

$$
\mathbf{U}_{sc}
=\mathbf{X}^{\mathrm{retained}}_{sc}
-\mathbf{X}^{\mathrm{regressed}}_{sc}
$$

上拟合，再用所得载荷矩阵投影原始 $\mathbf{X}^{\mathrm{retained}}_{sc}$；因此降维方向由 task GLM 移除的成分确定，而后续动力学仍使用保留完整任务信号的时序。REST 没有对应的任务回归版本，故在自身前 75% 时序上拟合 PCA，并投影自身。最终每个网络保留第一个主成分（$k=1$），七个网络形成 $\mathbf{x}_t\in\mathbb{R}^{7}$。任务态 PC1 的平均累计解释方差为 67.35%，REST 为 44.53%。

动力学模型使用三阶网络历史（$p=3$）预测下一时刻七网络状态：

$$
\mathbf{h}_t
=\left[
\mathbf{x}_t^\top,
\mathbf{x}_{t-1}^\top,
\mathbf{x}_{t-2}^\top
\right]^\top
\in\mathbb{R}^{21},
\qquad
\mathbf{x}_{t+1}
=\mathbf{A}\mathbf{h}_t+\mathbf{b}+\boldsymbol{\varepsilon}_t.
$$

每名被试、每个状态分别在前 75% 时间段拟合线性 $\Delta$-Ridge，正则强度为 $\alpha=1$；后 25% 只用于预测诊断。连续 EI 使用线性高斯 affine-TM 的协方差 log-det 形式，单位为 bits。该估计器是仿射 transport map 在高斯情形下的闭式实现；标准化 source 的干预协方差固定为单位阵，即各 source 维使用相互独立的单位高斯干预。最终 232 个“被试 $\times$ 状态”模型的平均 held-out RMSE/持久性基线比为 0.907，207/232 个模型优于持久性基线。

将 21 个“网络 $\times$ 滞后”标量视为最细 source，本文把其相对完整 source 的总协同记为 system-level $\Xi$：

$$
\Xi_{sc}
=EI(\mathbf{h}_t;\mathbf{x}_{t+1})
-\sum_{j=1}^{21}EI(h_{t,j};\mathbf{x}_{t+1}).
$$

该量就是 PEID 在 singleton source partition 下的 system-level synergy；这里使用 $\Xi$，是为了与第 2.2 节固定七维 PC1 表征下的历史 raw $\Phi^{EID}$ 结果区分。三个滞后在每个 Yeo7 网络内绑定为一个模块 $M_g$。网络内协同为

$$
\Xi_g^{\mathrm{within}}
=EI(M_g;\mathbf{x}_{t+1})
-\sum_{j\in M_g}EI(h_{t,j};\mathbf{x}_{t+1}),
$$

跨网络协同为

$$
\Xi^{\mathrm{cross}}
=EI(\mathbf{h}_t;\mathbf{x}_{t+1})
-\sum_{g=1}^{7}EI(M_g;\mathbf{x}_{t+1}).
$$

跨网络部分用精确 Shapley 值分配到七个网络，网络 $g$ 的守恒归因为

$$
C_g
=\Xi_g^{\mathrm{within}}
+\operatorname{Shapley}_g(\Xi^{\mathrm{cross}}),
\qquad
\sum_{g=1}^{7}C_g=\Xi.
$$

具体地，令 $\mathcal{N}$ 为七个 Yeo 网络的集合，$M_S$ 为联盟 $S\subseteq\mathcal{N}$ 所包含的全部网络历史变量。联盟价值定义为该联盟内部不可由单网络 EI 相加解释的跨网络协同：

$$
v(S)
=EI(M_S;\mathbf{x}_{t+1})
-\sum_{h\in S}EI(M_h;\mathbf{x}_{t+1}),
\qquad
v(\varnothing)=0.
$$

网络 $g$ 的精确 Shapley 值遍历其余六网络形成的全部 $2^6=64$ 个联盟：

$$
\operatorname{Shapley}_g
=\sum_{S\subseteq\mathcal{N}\setminus\{g\}}
\frac{|S|!\,(6-|S|)!}{7!}
\left[v(S\cup\{g\})-v(S)\right].
$$

方括号是把网络 $g$ 加入联盟 $S$ 后新增的跨网络协同；阶乘系数等价于在七网络所有加入顺序中，$S$ 恰好排在 $g$ 前面的概率。该定义满足效率性质

$$
\sum_{g\in\mathcal{N}}\operatorname{Shapley}_g
=v(\mathcal{N})
=\Xi^{\mathrm{cross}}.
$$

以 `sub-100206` 的 LANGUAGE--Control 为数值示例。若先行联盟为 $S=\{\mathrm{Vis},\mathrm{SomMot}\}$，则

$$
\begin{aligned}
v(S)&=0.171757,\\
v(S\cup\{\mathrm{Control}\})&=0.522372,\\
\Delta_{S,\mathrm{Control}}
&=0.522372-0.171757
=0.350615\ \text{bits}.
\end{aligned}
$$

此时 $|S|=2$，单个联盟的权重为 $2!4!/7!=1/105$，所以这一项对 Control Shapley 值的贡献为

$$
\frac{1}{105}\times0.350615
=0.003339\ \text{bits}.
$$

其余 63 个联盟完全同样计算。按先行联盟大小 $m=|S|$ 汇总如下；同一 $m$ 下共有 $\binom{6}{m}$ 个联盟，表中最后一列已经把该组全部联盟的加权贡献相加。

| $m$ | 联盟数 | 每个联盟权重 | 平均边际贡献（bits） | 该组 Shapley 贡献（bits） |
|---:|---:|---:|---:|---:|
| 0 | 1 | $1/7$ | 0.000000 | 0.000000 |
| 1 | 6 | $1/42$ | 0.207695 | 0.029671 |
| 2 | 15 | $1/105$ | 0.436730 | 0.062390 |
| 3 | 20 | $1/140$ | 0.679959 | 0.097137 |
| 4 | 15 | $1/105$ | 0.931481 | 0.133069 |
| 5 | 6 | $1/42$ | 1.190078 | 0.170011 |
| 6 | 1 | $1/7$ | 1.462705 | 0.208958 |

七组相加得到

$$
\operatorname{Shapley}_{\mathrm{Control}}
=0.701236\ \text{bits}.
$$

该被试的 Control 网络内协同为 $\Xi_{\mathrm{Control}}^{\mathrm{within}}=0.243910$ bits，所以最终 Control 守恒归因为 $C_{\mathrm{Control}}=0.243910+0.701236=0.945146$ bits。再除以该被试 LANGUAGE 的 system-level $\Xi=4.699821$ bits，得到 Control 份额 20.11%。图 a 的 20.5% 是对 29 名被试分别完成上述计算和归一化后，再对 29 个 Control 份额取算术平均的结果。

主图先在每名被试 $s$、每个状态 $c$ 内计算网络份额

$$
P_{scg}=\frac{C_{scg}}{\Xi_{sc}},
\qquad
\sum_{g=1}^{7}P_{scg}=1,
$$

再对 29 名被试取平均：

$$
\overline{P}_{cg}
=\frac{1}{29}\sum_{s=1}^{29}P_{scg},
\qquad
\sum_{g=1}^{7}100\overline{P}_{cg}=100\%.
$$

因此图 a 的每一列都是该状态 system-level $\Xi$ 在七个网络间的完整组成，精确数值相加严格等于 100%，不是七个彼此独立的效应量。图内数字只保留一位小数，显示值相加会因四舍五入得到 99.9%--100.1%；未舍入数组的最大单被试闭合误差仅为 $6.66\times10^{-16}$。这种归一化比较的是“总 $\Xi$ 如何分配”，不会把 REST 的整体幅度优势重复计入空间构成。跨网络 $\Xi^{\mathrm{cross}}$ 另用相同七网络顺序执行允许小数值误差、只保留正贡献的 greedy 层级分解；每个 atom 除以本 run 的 $\Xi^{\mathrm{cross}}$ 后比较组合份额。最大 $\Xi$ 恒等式误差为 $3.55\times10^{-15}$ bits，网络和 atom 份额的闭合误差均低于 $10^{-15}$。

![REST 与七任务的整体 Phi / system-level Xi 散点箱图](../../results/hcp_schaefer500_task_evoked_xi_tuning/final/overall_phi_rest_task_scatter_box.png)

| 状态 | system-level $\Xi$ 均值（bits） | REST 减该任务（bits） | BH $q$ |
|---|---:|---:|---:|
| REST | **7.0401** | -- | -- |
| EMOTION | 4.3006 | +2.7396 | $1.67\times10^{-5}$ |
| GAMBLING | 4.3414 | +2.6988 | $5.58\times10^{-5}$ |
| LANGUAGE | 4.6574 | +2.3827 | $9.65\times10^{-5}$ |
| MOTOR | 5.5130 | +1.5271 | 0.00378 |
| RELATIONAL | 5.0594 | +1.9807 | 0.000936 |
| SOCIAL | 5.5371 | +1.5030 | 0.00922 |
| WM | 4.9271 | +2.1131 | 0.000341 |

散点箱图中的每个点是一名共同被试，白色菱形为均值。七项 REST--任务双侧配对 Wilcoxon 检验在七项内作 BH 校正后均显著，因此当前任务诱发 PCA 表征仍保持 **REST 的整体 $\Xi$ 显著高于全部任务态**。该结论是群体配对结果，并不要求每名被试都满足同一排序；REST 高于各任务的被试比例为 69.0%--82.8%。

![REST 与七任务的 Xi 网络份额和层级 atom 分布](../../results/hcp_schaefer500_task_evoked_xi_tuning/final/selected_xi_state_distributions.png)

图 a 直接显示七网络占 system-level $\Xi$ 的平均组成份额，而不是“该状态减其余状态”的特异性，也不是七网络各自的绝对 $\Xi$。读图时应在同一列内比较颜色和数字：某网络份额升高，必然由同列其他网络份额降低来平衡。REST 以 SomMot（17.5%）、DorsAttn（16.9%）和 Default（16.1%）较高；LANGUAGE 的 Control 达 20.5%；RELATIONAL 与 SOCIAL 的 DorsAttn 分别为 20.5% 和 21.4%；MOTOR 的 SomMot 为 17.1%。仅比较七任务时，七个网络的重复测量 Friedman 状态效应均经 BH 校正显著（最大 $q=0.0248$）。网络份额分布的 28 个八状态两两对比中 25 个显著；去除 REST 后，21 个任务对中仍有 18 个显著。未显著的任务对仅为 GAMBLING--WM、RELATIONAL--SOCIAL 和 RELATIONAL--WM。

图 c 展示 120 个候选组合中跨状态平均份额最高的 12 个 greedy atom。REST 的全七网络 atom 为 17.0%，而多数任务的最大项是缺少 Limbic 的六网络组合：EMOTION 18.9%、GAMBLING 20.8%、LANGUAGE 17.3%、RELATIONAL 18.5%、SOCIAL 20.5% 和 WM 19.0%；MOTOR 的全七网络与缺 Limbic 六网络分别为 17.0% 和 16.5%。atom 分布的 28 个八状态对比中 17 个显著，21 个纯任务对中 11 个显著。由此可见，状态差异同时出现在单网络份额和跨网络组合份额上，但网络级证据比具体 greedy atom 更稳定。

参数选择没有使用 LOSO，也没有直接挑选使图面差异最大的配置。候选先在固定 8 名被试上筛选，再在未参与筛选的 21 名被试上确认，最后用 29 名共同被试汇总。相对基线 $(k,p,\alpha)=(1,5,10)$，最终 $(1,3,1)$ 将网络和 atom 的 between/within total-variation ratio 分别从 0.493 提高到 0.788、从 0.496 提高到 0.611；显著八状态对分别从 18/28 增至 25/28、从 9/28 增至 17/28。更弱正则 $\alpha=0.3$ 虽进一步增大分离度，却不能保持 REST--SOCIAL 显著，故未作为主配置。

解释上，$C_g$ 将网络内部的跨滞后协同与跨网络 Shapley 份额合并，因此“某网络份额较高”不等同于该网络局部激活更强，也不是对该网络的因果必要性证明。任务 PCA 载荷来自 retained--regressed 差值，但动力学投影对象仍是 retained；结果因而反映沿任务诱发方向观察到的完整任务态动力学，而不是 task-evoked 成分本身的 $\Xi$。REST 的 PCA 基底来自自身时序，与任务态的任务诱发基底具有不同的构造语义，因此 REST--任务的绝对差还包含表征选择差异。greedy atom 具有路径依赖性，显著组合只表示当前固定候选空间和算法下可重复的层级归因，不是唯一真实的脑网络层级。

#### TEVF 留一被试空间图识别验证（补充）

作为前述 TEVF parcel 图的补充，仅展示组均值热图仍可能夸大差异，因此用 leave-one-subject-out（LOSO）最近质心分类检验单个新被试。对每张 $q_{sc\cdot}$ 图先在 parcel 维去均值并作 $L_2$ 归一化；每一折只用其余 29 名被试分别形成七个任务质心，再按余弦相似度预测留出被试的七张图。显著性检验在每名被试内部独立置换七个任务标签，并对每次置换完整重跑 LOSO，共 2,000 次。该置换保持了每名被试的七张原始空间图及被试内依赖结构。LOSO 未参与任务诱发 PCA–$\Xi$ 的参数选择或主结论。

![七任务空间图的留一被试可辨识性](../../results/hcp_schaefer500_task_specific_regions/task_map_discriminability.png)

500-parcel 空间图的七分类准确率为 90.0%（189/210；chance 14.3%；置换 $p=(0+1)/(2000+1)=0.0005$）。各任务召回率为 EMOTION 86.7%、GAMBLING 60.0%、LANGUAGE 96.7%、MOTOR 100%、RELATIONAL 93.3%、SOCIAL 100% 和 WM 93.3%；主要混淆是 GAMBLING 被判为 RELATIONAL（9/30）。将完全相同的 TEVF 图先压缩为七个 Yeo7 网络均值，准确率降至 68.1%（143/210，$p=0.0005$），其中 GAMBLING 和 WM 分别只有 20.0% 和 43.3%。这构成独立于可视化尺度的证据：七任务确有可跨被试泛化的空间差异，而且相当一部分差异存在于 Yeo7 网络内部。

TEVF 与 $d$ 不是 EI、$\Phi$ 或 $\Xi$ 的替代估计器。它们回答“任务设计解释了哪些 parcel 的时间变异”，而任务诱发 PCA–$\Xi$ 回答“沿任务诱发方向观察时，历史状态的联合动力学信息如何分配到网络自身与跨网络组合”。本文以 $\Xi$ 网络和层级分解作为任务动力学归因的主结果，以 TEVF/$d$ 作为 parcel 级任务定位的互补证据；不能把 TEVF 称为 $\Xi$ 节点归因，也不能从当前观察性对比推出某脑区对行为的因果必要性。当前 Schaefer-500 只覆盖皮层，任务相关的杏仁核、纹状体、小脑等皮层下或非皮层结构仍未进入该图。

<a id="discussion"></a>

## 3. 讨论：解释边界与可复现性

<a id="discussion-scope"></a>

### 3.1 结论的适用范围

- DMF 结果支持 $\Phi^{EID}$ 在该代理结构连接、无裁剪全状态干预与 $[0.30,0.70]^{166}$ 生理支持下定位 Kuramoto 对齐的临界窗；它不等同于人体大脑存在同一精确耦合常数，也不外推到绝对最大熵 $[0,1]^{166}$ 干预。
- HCP 静息态结果来自 REST1_LR；任务态主 $\Xi$ 分析覆盖七种 `taskRetained` LR 任务。任务 PCA 在 retained--regressed 差值上拟合，再投影 retained；最终共享参数为 $(k,p,\alpha)=(1,3,1)$。该参数先在 8 名被试上筛选，并在未参与筛选的 21 名被试上确认；29 名完整汇总保持 REST 的 system-level $\Xi$ 显著高于全部任务，同时在七网络份额和 greedy atom 份额上检出多数任务对差异。相对地，第 2.2 节的 $(p,\alpha)=(8,10)$ raw $\Phi^{EID}$ 与附录 E 的 25 点扫描保留为历史表征和参数敏感性参照，不再作为任务空间归因的主结果。长度匹配检验表明 REST 高方差只在 EMOTION 与 MOTOR 上最稳定。30 名被试的 Schaefer-500 TEVF 仍直接描述 task GLM 移除的 parcel 级时间能量，而新 $\Xi$ 分解描述沿任务诱发 PCA 方向观察到的完整 retained 动力学；二者不能互换。尚未检验 RL run、独立 cohort、任务事件或条件子类型分层、去趋势、运动或生理混杂回归、GSR、皮层下结构或其他 null 构造。
- WM 与既有静息态的主幅度比较分别使用 304 和 900 个拟合时间点，只比较 raw Phi，因此均值差仍包含有效样本长度差异。独立的 12 窗口长度匹配分析只检验跨被试方差；WM 的 `sub-103515` 具有极端早期 PC1 瞬变，普通方差比对其高度敏感。
- HCP 的全体被试 Phi 结果使用 20 个 null，p 值分辨率有限，且未校正跨被试、跨模块集合和 greedy 选择造成的多重比较。
- 贪婪 atom 用于描述候选协同结构；它依赖分解顺序与候选空间，不是 exhaustive 的唯一高阶分解。

<a id="discussion-artifacts"></a>

### 3.2 结果与图形产物

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
| HCP500 REST 与七任务特异脑区分布 | `results/hcp_schaefer500_task_specific_regions/summary.json`、`results/hcp_schaefer500_task_specific_regions/report.md`、`results/hcp_schaefer500_task_specific_regions/task_evoked_region_maps.npz`、`results/hcp_schaefer500_task_specific_regions/task_evoked_region_profiles.{png,svg,pdf}`、`results/hcp_schaefer500_task_specific_regions/rest_all_tasks_variance_profiles.{png,svg,pdf}`、`results/hcp_schaefer500_task_specific_regions/task_map_discriminability.{png,svg,pdf}` |
| HCP500 任务诱发 PCA–$\Xi$ 网络与层级分解 | `results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/summary.json`、`results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/report.md`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/overall_phi_rest_task_scatter_box.{png,svg,pdf}`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/selected_xi_state_distributions.{png,svg,pdf}`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/parameter_tuning_comparison.{png,svg,pdf}` |

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

<a id="appendix-e"></a>
<a id="hcp-hyperparameter-robustness"></a>

## 附录 E：拟合模型参数鲁棒性

![HCP500 超参数鲁棒性与预测误差诊断](../../fig/brain_hcp500_robustness_plate.png)

*图 E1. HCP500 的超参数鲁棒性与预测诊断。A：25 点 $p$–$\alpha$ 网格上，七项 REST-minus-task 均值差的最小值（a）及 Holm 校正后显著的对比数（b）。B：七个任务各自的 REST-minus-task raw $\Phi^{EID}$ 边际。C：八状态等权平均的留出 delta-NRMSE（a）、test-minus-train 泛化间隙（b），以及 $p=8$ 时各状态随 $\alpha$ 的留出误差（c）。D：八个状态各自的留出 delta-NRMSE。A–B 回答 Phi 排序是否依赖超参数；C–D 检验弱正则反转是否伴随预测失真。*

<a id="appendix-e-1"></a>

### E.1 REST–任务差异的 $p$–$\alpha$ 鲁棒性

原始 $p=8,\alpha=10$ 来自静息态模型选择，可能给 REST 带来选择优势。为检验这一点，这里预先固定 $p\in\{1,2,3,5,8\}$ 与 $\alpha\in\{0.1,1,10,100,1000\}$ 的 25 点网格。每个网格点把同一组超参数同时用于 REST 和七种任务态；29 名共同被试的每个状态仍独立重新拟合 PCA、缩放、Ridge 系数、截距和残差协方差。所有序列使用各自前 75%，只计算 raw $\Phi^{EID}$。每个网格点分别完成七项双侧 Monte Carlo paired sign-flip 检验（200,000 次），再在七任务内作 Holm 校正。

图 E1A-a 给出每个网格点七项 REST-minus-task 群体均值差中的最小值；正值表示 REST 均值高于所有任务，负值表示至少一个任务反超。图 E1A-b 给出七项对比中 Holm 校正后显著的数量。REST 只在 **15/25** 个网格点为八条件群体均值最高状态，只有 **7/25** 个网格点满足“REST 高于全部任务且七项均显著”。这 7 个点为 $(p,\alpha)=(2,0.1),(2,1),(2,10),(3,1),(3,10),(5,10),(8,10)$。原始 $(8,10)$ 的 232 个“被试 $\times$ 状态”数值与第 2.2 节逐点完全一致，排除了结果变化来自实现路径差异。七个任务各自的边际见图 E1B。

稳定区主要位于中等历史阶数和中等正则强度：$\alpha=10$ 时，$p=2,3,5,8$ 的七项对比均显著；但 $p=1$ 的所有 $\alpha$ 均为 0/7 显著。在过强正则 $\alpha=1000$ 下，五个 $p$ 均不再满足 REST 群体均值最高，且最多只有 1/7 项显著。弱正则与高阶历史的组合还会反转方向：$p=8,\alpha=0.1$ 时，REST 相对 EMOTION 和 RELATIONAL 分别低 2.789932 和 2.635 bits，两项任务高于 REST 的 Holm 校正 $p$ 分别为 0.003540 和 0.003150；其余任务的方向或显著性也不统一。

因此，最窄且可靠的结论是：**REST 高于七任务的 raw $\Phi^{EID}$ 在包含原配置的中等正则参数带内可复现，但对完整 25 点 $p$–$\alpha$ 网格并不鲁棒。**$p$ 同时改变 source 维数 $7p$，所以应在同一个 $(p,\alpha)$ 内解释 REST–任务差值，不能把不同 $p$ 的绝对 raw Phi 当作同维度量直接比较。上述 Holm 校正只针对每个网格点内部的七项任务对比，并未把 25 个网格点作为发现性假设族再次校正；因此 7/25 是敏感性描述，不是 175 项独立发现。极端超参数是否应被视为合理模型还需要独立、条件平衡的预测验证规则；在完成该步骤之前，不能从本扫描中事后挑选最支持 REST 或任务态的参数作为主结论。

<a id="appendix-e-2"></a>

### E.2 留出预测误差解释弱正则反转

为判断左下角弱正则、高阶历史的反转是否来自拟合失真，这里保持同一 25 点网格和每个“状态 $\times$ 被试”的前 75% 训练段，严格用最后 25% 时间点计算一步留出误差。主指标 delta-NRMSE 先用训练段的每个网络 delta 标准差归一化，再跨网络、时间点和被试汇总；同时报告 test-minus-train 泛化间隙，以及相对“$\mathbf{x}_{t+1}=\mathbf{x}_t$”持久性基线的技能值。技能为正表示优于持久性预测，为负表示模型在留出段反而更差。

图 E1C-a 显示八状态等权平均的留出 delta-NRMSE，图 E1C-b 显示泛化间隙，图 E1C-c 单独展开 $p=8$ 时各状态随 $\alpha$ 的误差；各状态的完整误差网格见图 E1D。结果支持过拟合解释：$p=8,\alpha=0.1$ 的训练误差仅为 0.6538，但留出误差升至 0.9612，泛化间隙达到 0.3074，持久性技能为 $-0.0406$；其平均 Ridge 系数 Frobenius 范数为 7.735。改用 $\alpha=10$ 后，留出误差降至 0.8726、泛化间隙降至 0.1685、持久性技能升至 0.2049，系数范数缩至 2.935。$p=8$ 的整体最低留出误差出现在 $\alpha=100$，为 0.8590，说明 $\alpha=10$ 位于高泛化区，但不是该阶数下的唯一或总体最优正则值。

发生显著 raw $\Phi^{EID}$ 反转的两个任务同时给出最直接的失败证据。在 $p=8,\alpha=0.1$ 下，EMOTION 的训练/留出误差为 0.5487/1.0628，泛化间隙 0.5141，持久性技能 $-0.3454$；RELATIONAL 为 0.6176/1.0062，泛化间隙 0.3887，持久性技能 $-0.1106$。也就是说，弱正则模型在训练段把这两个较短任务拟合得异常好，但在未参与拟合的时间段已经不具备可靠预测能力。固定 $p=8$ 后，在全部 35 个“超参数 $\times$ 任务”对比中，REST-minus-task Phi 边际与 task-minus-REST 留出误差差的 Spearman $\rho=-0.592$（$p=1.79\times10^{-4}$），与泛化间隙差的 $\rho=-0.480$（$p=0.00351$）：任务相对 REST 过拟合越明显，Phi 排序越倾向任务反超。

这种反转的可能机制是：$p=8$ 提供 56 维历史 source，而任务态用于回归的训练样本仅约 124--296 行；$\alpha=0.1$ 允许更大的回归系数，并在训练段压低残差协方差。Gaussian log-det $\Phi^{EID}$ 同时依赖转移系数和训练残差协方差，因此会把这种训练内的高自由度拟合转化为偏大的 raw Phi。该证据说明预测失真是反转的重要来源，但不是数学上的唯一原因，因为 $\Phi^{EID}$ 不是预测误差的单调函数，且并非所有预测较差的点都发生显著方向反转。

从共享超参数选择看，$\alpha=10$ 在 40 个“八状态 $\times$ 五个 $p$”组合中的 29 个（72.5%）达到最低留出误差；八状态总体平均下，$p=1,2,3,5$ 的最优 $\alpha$ 都是 10，$p=8$ 则是 100。若在整个 25 点网格中按八状态与29名被试等权的留出误差选择一组共享参数，最优点是 $(p,\alpha)=(5,10)$，delta-NRMSE 为 0.8552；在这个不依据 REST–任务 Phi 方向选出的配置上，REST 相对最接近任务仍高 1.4243 bits，七项对比全部 Holm 显著。因此，对本附录所检验的“各状态分别在自身 retained 时序上拟合 PC1”的原始表征，$(5,10)$ 是更公平的预测型参照。第 2.3 节的主配置 $(k,p,\alpha)=(1,3,1)$ 使用不同的任务诱发 PCA 构造，并经过 8 人筛选、21 人独立确认，二者不应视为同一参数搜索中的冲突最优点。

<a id="appendix-f"></a>
<a id="hcp500"></a>

## 附录 F：HCP500 PCA–Yeo7 Phi–null 分解

<a id="appendix-f-1"></a>
<a id="hcp500-data"></a>

### F.1 数据、降维与动力学表征

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

<a id="appendix-f-2"></a>
<a id="hcp500-phi"></a>

### F.2 History-source $\Phi^{EID}$ 与 circular-shift null

本实验的量是 56 维历史 source 到下一时刻 7 维状态的量，而不是 7 维一阶 whole-state $\Phi^{EID}$：

$$
\Phi^{EID}_{\mathrm{hist}\to\mathrm{next}}
=EI\!\left(\mathbf{h}_t;\mathbf{x}_{t+1}\right)
-\sum_{j=1}^{56}EI\!\left(h_{t,j};\mathbf{x}_{t+1}\right).
$$

估计使用 Gaussian log-det 口径，而非 TM。这样可在 56 维 history source、30 名被试、重复 null 重拟合和贪婪分解下保持可计算性；代价是结果依赖 Gaussian 近似，不能直接外推到非高斯的精确干预信息量。

null 对 7 条 PC1 时序分别施加独立、非零的 circular shift，并在相同的 $p$、$\alpha$ 与 900 点训练预算下重新拟合模型。它保留每条网络 PC1 的边际取值与自相关结构，但破坏网络间的时间对齐，检验观测到的跨网络结构是否超出网络内时间结构本身。

在 30 被试、每人 20-null 的扩展中，observed $\Phi^{EID}$ 的均值与中位数为 6.188481 与 6.068454 bits；observed-minus-null-mean 的均值与中位数为 1.984600 与 2.051671 bits，范围为 0.492521–4.096287 bits。30/30 名被试的 observed 都高于各自 null 均值，且未校正经验 $p<0.05$；由于仅有 20 个 null，每个被试的最小 p 值分辨率为 $1/21=0.047619$。

该分布见图 F1A。结果支持在这一固定的 reduced-state、history-source 定义与 null 下，跨网络时间对齐带来额外的高阶结构；它不排除低频漂移、运动、生理噪声、PCA 表征选择或 Gaussian 近似造成的影响。

<a id="appendix-f-3"></a>
<a id="hcp500-module"></a>

### F.3 Yeo7 模块历史分解

在全部 30 名被试上进行贪婪分解。为避免将同一网络的八阶历史误作 8 个独立脑区，属于同一个 Yeo7 网络的全部 8 个滞后 PC1 值被绑定为一个不可拆模块原子；候选空间因此是 7 个网络模块，而不是 56 个逐滞后变量。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | 固定集合未校正 $p<0.05$ |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **20 / 30** | 1.314357 bits | 20 / 20 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **17 / 30** | 1.227709 bits | 17 / 17 |
| SomMot + DorsAttn + SalVentAttn + Cont + Default | 9 / 30 | 0.956657 bits | 8 / 9 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.086753 bits | 7 / 8 |

![HCP500 Yeo7 Phi-null 与模块协同核汇总](../../fig/brain_hcp500_yeo7_phi_null_summary.png)

*图 F1. HCP500 PCA–Yeo7 Phi–null 分解。A：30 名被试的 observed-minus-null-mean $\Phi^{EID}$；绿色表示当前 20-null 分辨率下的经验 $p<0.05$。B：每名被试 greedy top-3 中出现的模块核及其原子贡献；空白表示该核未进入对应被试的 top-3。*

全 7 网络核最常出现（20/30），但它在 matched null cohort 中反而更常出现（均值 26.65/30；经验 $p=1$），因此不能将其读为真实数据特异的协同核。缺少 Limbic 的六网络广域核在真实数据中为 17/30，而 matched null cohort 的频率为 $8.65/30$（最大 12/30；经验 $p=1/21=0.047619$）；两个较小的候选核也分别为 9/30 对 $1.40/30$、8/30 对 $2.45/30$，同为该分辨率下的未校正 $p=1/21$。这支持真实静息态中若干非全网络模块核的出现频率和贡献高于该 circular-shift null；但 20 个 null 的 p 值分辨率有限，且统计未校正跨模块集合与 greedy 选择，因此不构成唯一生物学 atom 的确证。

<a id="appendix-g"></a>
<a id="hcp1000"></a>

## 附录 G：HCP1000 PCA–Yeo7 Phi–null 分解

<a id="appendix-g-1"></a>
<a id="hcp1000-data"></a>

### G.1 数据、降维与模型选择

同一 30 名 `REST1_LR` 被试的 `Schaefer1000` 矩阵为 $1200\times1000$。1000 个 parcel 按同一 Yeo7 顺序分为 Vis 162、SomMot 194、DorsAttn 122、SalVentAttn 121、Limbic 60、Cont 129 与 Default 212 个 parcel；每名被试的各网络 PC1 均只以前 900 点拟合并投影完整时序。

在 `sub-100206` 的训练段内以 600/700/800 三个时间验证折，从 $p\in\{1,2,3,5,8\}$ 与既有 Ridge $\alpha$ 网格选择模型。最优冻结配置为五阶 $\Delta$-Ridge，$p=5$、$\alpha=1$，平均 validation skill ratio 为 0.794433；因此 source 是 35 维网络历史，target 为下一时刻 7D 网络状态。后 300 点未参与 PC1、模型或参数选择。

<a id="appendix-g-2"></a>
<a id="hcp1000-phi"></a>

### G.2 History-source $\Phi^{EID}$ 与 circular-shift null

对每名被试固定上述表征与模型，并以每条 PC1 独立、非零 circular shift 后重拟合同一模型生成 20 个 null。1000-parcel observed $\Phi^{EID}$ 的均值/中位数为 7.783676/7.734082 bits；observed-minus-null-mean 的均值/中位数为 2.997670/3.032261 bits，范围为 0.814450–6.085586 bits。30/30 名被试的 observed 均高于其 null 均值，未校正经验 p 均小于 0.05（最小分辨率 $1/21=0.047619$）。

该分布汇总于图 G1A。

<a id="appendix-g-3"></a>
<a id="hcp1000-module"></a>

### G.3 Yeo7 模块历史分解与 500 对照

分解中，同一网络的全部五个 PC1 历史滞后绑定为一个不可拆模块原子；每个 observed 与 null 都完整运行 greedy top-3。1000-parcel 的常见核如下。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | matched-null 频率均值；经验 p |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **21 / 30** | 1.637465 bits | 21.80 / 30；0.761905 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **12 / 30** | 1.575094 bits | 6.35 / 30；0.047619 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.500212 bits | 2.40 / 30；0.047619 |
| Vis + SomMot + DorsAttn + Cont + Default | 7 / 30 | 1.579938 bits | 2.90 / 30；0.047619 |

![HCP1000 Yeo7 Phi-null 与模块协同核汇总](../../fig/brain_hcp1000_yeo7_phi_null_summary.png)

*图 G1. HCP1000 PCA–Yeo7 Phi–null 分解。A：30 名被试的 observed-minus-null-mean $\Phi^{EID}$；绿色表示当前 20-null 分辨率下的经验 $p<0.05$。B：每名被试 greedy top-3 中出现的模块核及其原子贡献；空白表示该核未进入对应被试的 top-3。*

| 描述性比较 | Schaefer-500 | Schaefer-1000 |
|---|---:|---:|
| observed $\Phi^{EID}$ 均值（bits） | 6.188481 | 7.783676 |
| observed − null 均值（bits） | 1.984600 | 2.997670 |
| observed 高于 null mean | 30 / 30 | 30 / 30 |
| 全七网络核 top-3 频率 | 20 / 30 | 21 / 30 |
| 缺 Limbic 六网络核 top-3 频率 | 17 / 30 | 12 / 30 |

两种粒度都复现了跨网络时间对齐高于 circular-shift null 的方向性证据，并都将缺 Limbic 的六网络广域核识别为高于 matched-null 频率的候选结构。绝对 bits、最优滞后阶数和 atom 频率受分区粒度、PC1 表征与单被试调参影响；上表仅作描述性对照，不能当作空间粒度的正式统计检验，也不将 greedy 核解释为唯一生物学 atom。
<a id="dmf-hierarchy"></a>
