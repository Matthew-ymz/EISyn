# 脑科学实验：Schaefer100 DMF 临界识别与 HCP Yeo7 $\Xi$ 分解

## 结论

本报告保留三个互补结果块。

1. **Schaefer100 DMF 跨连接组复现。**将原 83 ROI 的有向 F-TRACT 代理矩阵替换为 93 名受试者的 100×100 对称结构连接群体均值后，$\Xi$ 从 $G=0$ 的 8.452 bits 上升，在 $G=1.3$ 达到 $26.131\pm0.163$ bits（均值 ± 跨 seed SD）的峰值，随后下降；8/8 个 seed 的峰位一致。平均发放率的最大离散导数位于 $G=1.5$。以 $\Xi$ 峰为中心重新选定的候选临界窗 $G\in\{1.2,1.3,1.4\}$ 中，88.00% 的 $\Xi$ 来自跨 ROI 协同，其中网络间分量占总量的 66.27%。
2. **HCP500/1000 PCA–Yeo7 $\Xi$ 分解。**在相同 30 名 HCP REST1_LR 被试中，Schaefer-500（$p=8,\alpha=10$）与重新验证的 Schaefer-1000（$p=5,\alpha=1$）均在 30/30 名被试中高于独立 PC1 circular-shift null。两种粒度的全七网络核都常进入 top-3，但不高于 matched null；相对地，缺少 Limbic 的六网络核均高于 matched null cohort（500：17/30 对 8.65/30；1000：12/30 对 6.35/30；各 20-null 未校正 $p=0.047619$）。
3. **HCP1000 任务诱发表征、$\Xi$ 层级分解与认知画像。**正文以 Schaefer-1000 为主要空间粒度：任务态先在 `taskRetained-taskRegressed` 上拟合各 Yeo7 网络的 PC1，再用同一载荷投影 `taskRetained`；REST 则在自身时序上拟合并投影。共享配置为一维网络状态、三阶历史与 Ridge $\alpha=1$，即 21 维 source 预测 7 维下一时刻 target。29 名共同被试中，REST 的 system-level $\Xi$ 均值为 6.985 bits，高于七任务的 4.288--5.686 bits；七项配对 Wilcoxon 检验经 BH 校正后均显著。七网络归因份额在仅比较任务态时 7/7 个网络均有显著状态效应，greedy 层级分解进一步描述了跨网络 $\Xi$ 在主要网络组合间的分配。合并主图同时展示一般认知与 LANGUAGE、MOTOR 全七网络 atom 的任务依赖关系，LANGUAGE 内 SomMot+Limbic+Cont 与 Story、Math ACC 的方向分化，以及 WM 中 Cont+Default 固定组合与 0-back 表现的负相关。Schaefer-500 的旧主图、领域认知探索及跨粒度对照移至附录 K。

这些实验分别回答不同问题：DMF 实验检验 $\Xi$ 是否能定位可控模型中的临界动力学带；HCP 静息态实验检验降维后的真实网络动力学中是否存在高于同步破坏 null 的跨网络高阶结构；任务态实验检验以任务诱发 PCA 选择观测方向后，system-level $\Xi$ 及其网络和层级组合归因是否随状态改变。它们不构成对特定脑机制、因果方向或唯一稀疏 atom 的证明。

## 目录

1. [**Schaefer100 DMF：跨连接组临界复现**](#dmf-critical)
   1. [数据来源、完整性与 ROI 顺序](#dmf-data)
   2. [结构连接聚合与尺度](#dmf-sc)
   3. [DMF 与分析契约](#dmf-intervention)
   4. [全局扫描](#dmf-phi)
   5. [临界窗层级分解](#dmf-hierarchy)
   6. [结构关联与稳定性](#dmf-topology)
   7. [综合结论与文献对照](#dmf-insights)
   8. [解释边界](#dmf-limits)
   9. [参考文献](#dmf-references)
2. [**HCP 任务态 $\Xi$、认知画像与脑区分布**](#hcp-wm)
   1. [主要结果：Schaefer-1000 任务诱发 PCA–$\Xi$ 网络归因与层级分解](#hcp-task-evoked-xi-main)
   2. [协同核分布及静息态对照](#hcp-wm-phi)
   3. [七任务 raw $\Xi$ 历史表征参照](#hcp-all-tasks)
   4. [七任务态的 Schaefer-500 任务特异脑区分布](#hcp-task-specific-regions)
3. [**讨论：解释边界与可复现性**](#discussion)
   1. [结论的适用范围](#discussion-scope)
   2. [结果与图形产物](#discussion-artifacts)
4. [**附录 A：Kuramoto 振子数与 whole-state $\Xi$ 曲线形状**](#appendix-a)
   1. [临界峰的 EI 与 effectiveness 机制](#appendix-a-1)
   2. [时间窗、相变前检测与系统规模边界](#appendix-a-2)
5. [**附录 B：Schaefer100 DMF 动力学方程**](#appendix-b)
6. [**附录 C：83 ROI 与 100 ROI 受控比较**](#appendix-c)
7. [**附录 D：Schaefer100 可复现文件**](#appendix-d)
8. [**附录 E：拟合模型参数鲁棒性**](#appendix-e)
    1. [REST–任务差异的 p–α 鲁棒性](#appendix-e-1)
    2. [留出预测误差解释弱正则反转](#appendix-e-2)
9. [**附录 F：HCP500 PCA–Yeo7 $\Xi$–null 分解**](#appendix-f)
   1. [数据、降维与动力学表征](#appendix-f-1)
   2. [History-source $\Xi$ 与 circular-shift null](#appendix-f-2)
   3. [Yeo7 模块历史分解](#appendix-f-3)
10. [**附录 G：HCP1000 PCA–Yeo7 $\Xi$–null 分解**](#appendix-g)
    1. [数据、降维与模型选择](#appendix-g-1)
    2. [History-source $\Xi$ 与 circular-shift null](#appendix-g-2)
    3. [Yeo7 模块历史分解与 500 对照](#appendix-g-3)
11. [**附录 H：HCP500 WM 协同核及静息态对照**](#appendix-h)
    1. [WM 表征、模型与 circular-shift null](#appendix-h-1)
    2. [协同核分布及静息态对照](#appendix-h-2)
12. [**附录 I：HCP500 七任务 raw $\Xi$ 与长度匹配方差**](#appendix-i)
    1. [七任务 raw $\Xi$ 排序](#appendix-i-1)
    2. [REST–任务长度匹配方差检验](#appendix-i-2)
13. [**附录 J：REST 与七任务的共同方差空间参照**](#appendix-j)
14. [**附录 K：HCP500 个体认知画像与候选筛选诊断**](#appendix-k)

<a id="dmf-critical"></a>

## 1. Schaefer100 DMF：跨连接组临界复现

93 名受试者的 100×100 结构连接（SC）矩阵群体均值，并沿用之前文献相同的 DMF、干预采样和 Gaussian 条件协方差估计流程，重新完成全局耦合扫描、observational WMS、ROI/结构模块分解和 Yeo-7 网络分解。

<a id="dmf-data"></a>

<!-- ### 1.1 数据来源、完整性与 ROI 顺序 -->

<!-- 原始 `average.zip` 已原样解压至 `data/neuromodulator_receptor_sc_100/`。本实验用 `CON_SC_1mio/sub-*.csv` 中的 93 个无表头 100×100 SC 矩阵构造群体均值；`average.csv` 是 100 ROI × 19 个受体/转运体指标，仅用于核对数据包组成，没有进入 DMF。ROI 名称、Yeo-7 分组、RAS 质心和 fsaverage5 表面注释采用 [CBIG Schaefer2018 官方发布](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal)，图谱方法见 [Schaefer et al.（2018）](https://doi.org/10.1093/cercor/bhx179)。

93 个 SC 文件均为有限、非负、严格对称且主对角线为 0 的 100×100 矩阵。个体矩阵零元素比例为 30.26%–80.66%（中位数 56.64%）；相对群体均值的上三角边相关为 0.589–0.971（中位数 0.944）；谱半径为 0.344–1.297（中位数 0.799）。每个文件的 SHA-256 记录在 `results/dmf_schaefer100/preparation_summary.json`。 -->

<!-- 压缩包没有提供 ROI 标签、图谱名称、SC 单位、扩散 MRI/纤维追踪预处理说明或来源文献。为审计 ROI 顺序，本实验将群体均值与仓库内顺序已知的 Schaefer100 HCP 共识 SC 比较：原顺序上三角边相关为 0.44956；2,000 次标签置换的零分布均值为 0.000059、标准差为 0.01407，单侧 $p=0.000500$。这强烈支持两者顺序一致，但不能替代上游标签，因此标签状态仍记为 **inferred（推断）**。 -->

<a id="dmf-sc"></a>

### 1.2 结构连接聚合与尺度

93 个个体矩阵按边做算术平均：

$$
\overline{\mathbf{C}}=\frac{1}{93}\sum_{m=1}^{93}\mathbf{C}^{(m)}. \tag{1}
$$

式（1）没有阈值化、二值化、行归一化或谱半径归一化。新 SC 的谱半径为 0.70113、ROI 平均加权强度为 0.38498、非对角密度为 96.18%、最大权重为 0.31974；旧 83 ROI 矩阵的对应值为 0.34327、0.19614、51.09% 和 0.10629。新 SC 的谱半径约为旧矩阵的 2.04 倍，因此跨连接组比较同时报告原始 $G$ 和有效耦合 $G\rho(\mathbf{C})$，不把两组的原始 $G$ 当作相同物理尺度。

<a id="dmf-intervention"></a>

### 1.3 DMF 与分析契约

除连接矩阵和 ROI 数外，实验沿用旧分析的 DMF 方程、参数和统计口径。$G$ 从 0.0 到 3.0 按 0.1 扫描；JFIC 在 $G=1.0$ 校准后复用于整段扫描；随机种子为 3–10。每个条件独立采样 2,048 个 200 维干预 source：

$$
\mathbf{s}_E,\mathbf{s}_I\overset{\mathrm{ind}}{\sim}U(0.30,0.70)^{100},
$$

并将 300 个积分步后的完整 200 维 E/I 状态作为 target。状态不裁剪，正式运行的最大裁剪率为 0%。JFIC 最终收敛，最大绝对发放率误差为 0.0461 Hz；31/31 个 $G$ 条件均检测到稳定状态。

这里需要区分两个参照：平均发放率最大离散导数位于 $G=1.5$，而全局扫描得到的 $\Xi$ 峰值点是 $G=1.3$。由于图 1 下排 D–H 要回答的是“$\Xi$ 峰附近的协同如何分布”，旧窗口 $G\in\{1.4,1.5,1.6\}$ 全部落在峰后，不能覆盖峰前—峰值—峰后的局部形状。因此本版按当前 0.1 网格，以 8/8 个 seed 一致的峰位 $G=1.3$ 为中心，重新选定对称三点窗口 $G\in\{1.2,1.3,1.4\}$。这是依据完整扫描结果作出的峰值邻近窗，不是独立预注册的临界性检验；$G=1.5$ 仍保留为发放率转折参照。

系统级指标为

$$
\Xi=EI_{\mathrm{do}}(\mathbf{s}_t;\mathbf{y}_{t+300})-
\sum_{i=1}^{200}EI_{\mathrm{do}}(s_{t,i};\mathbf{y}_{t+300}). \tag{2}
$$

为保持与旧 83 ROI 实验的估计口径一致，本实验继续使用 ridge 为 $10^{-6}$ 的 Gaussian 条件协方差估计。200 维 source、200 维 target、21×8 个条件和每条件 2,048 个样本使逐条件 TM 全量扫描的计算代价过高。Gaussian 估计只刻画二阶依赖，可能遗漏高阶非线性结构，并对协方差正则化敏感；因此主要解释曲线形状、配对峰位和层级比例，而不把绝对 bits 直接外推到其他估计器。

<a id="dmf-phi"></a>

### 1.4 全局扫描

平均发放率从 $G=0$ 的 1.952 Hz 单调上升到 $G=3.0$ 的 43.123 Hz，最大离散导数位于 $G=1.5$。$\Xi$ 从 $G=0$ 的 $8.452\pm0.102$ bits 上升，在 $G=1.3$ 达到 $26.131\pm0.163$ bits 的峰值，随后降至 $G=3.0$ 的 $6.302\pm0.038$ bits；这里及图 1 A–C 的误差均为跨 8 个 seed 的 SD。峰前增幅为 17.679 bits，即 209.17%。8/8 个 seed 的峰均位于 $G=1.3$。重新选定窗口内 $G=1.2,1.3,1.4$ 的均值依次为 25.617、26.131 和 25.887 bits，覆盖峰前上升、峰值和峰后回落。

图中的误差条带改用跨 seed SD，以直接呈现随机 seed 波动。在 $G=1.3$，$\Xi$ 的 SD 为 0.163 bits，仅占均值的 0.62%；whole EI 与单变量 EI 之和的 SD 分别为 0.327 和 0.224 bits，对应变异系数为 0.15% 和 0.11%。因此 seed 波动相对于曲线幅度很小，不影响峰位判断。

observational $\Phi^{WMS}$ 从 $G=0$ 的 −21.433 bits 下降，在 $G=1.3$ 达到最负均值 −502.591 bits，并在 $G=1.4$ 跳回 −189.520 bits，呈现与发放率转折对齐的明显折点。其绝对值受 Gaussian 条件协方差和 ridge 影响，应优先解释形状。

![Schaefer100 DMF 多尺度汇总](../../fig/dmf_schaefer100/dmf_schaefer100_summary_full.png)

*图 1｜Schaefer100 DMF 多尺度汇总。A：平均发放率与全系统 $\Xi$；B：whole EI 与单变量 EI 之和；C：平均发放率与 observational $\Phi^{WMS}$。A–C 展示完整 $G=0$–3 扫描；曲线为 8 个 seed 的均值，阴影为跨 seed 标准差（SD）。D–H 基于以 $\Xi$ 峰 $G=1.3$ 为中心重新选定的候选临界窗 $G\in\{1.2,1.3,1.4\}$，先在每个 seed–$G$ 条件中独立分解，再对 8 个 seed × 3 个 $G$ 的 24 个条件汇总。D：各 ROI 的局部 E/I（ROI 内）耦合与跨 ROI leverage 在 24 个条件上的均值；E：先在每个 seed–$G$ 条件内计算 ROI 内/跨 ROI 比例，再在每个 seed 内平均三个 $G$；柱高为 8 个 seed 的均值，误差线为跨 seed SD。F：Yeo-7 网络内跨 ROI 分量；G：网络间 $\Xi$ 精确 Shapley 归因；F–G 同样先在每个 seed 内平均三个 $G$，再显示 8 个 seed 的均值与 SD，标签斜杠后的数字为 ROI 数，且采用相同网络顺序和颜色。H：每个 ROI 的跨 ROI leverage 在 24 个条件上的均值，按官方 Schaefer2018-100/fsaverage5 parcel 边界直接映射到左、右半球的外侧面和内侧面四视角，四个视角共用同一色标；灰色为图谱背景/内侧壁，不做质心插值。*

<a id="dmf-hierarchy"></a>

### 1.5 峰值邻近窗层级分解

本节所有 D–H 结果都采用相同汇总流程。对每个 seed $s\in\{3,\ldots,10\}$ 和每个 $G\in\{1.2,1.3,1.4\}$，先独立生成干预样本、演化目标状态、估计条件协方差并完成层级分解；随后对所得 24 个 seed–$G$ 条件做等权汇总。因此正文中的 bits 是 24 个条件的算术平均，不是某个特定 $G$ 的取值。图 E 的比例仍先在每个条件内计算，而不是用平均分量除以平均总量。为单独显示随机 seed 波动，图 E–G 的误差线先在每个 seed 内平均三个 $G$，再计算 8 个 seed 之间的 SD；三个 $G$ 的位置差异不进入误差线。该 SD 描述模拟随机性，不应解释为人群统计不确定性。

在每个 seed–$G$ 条件内，系统量首先按 ROI 块分解为

$$
\Xi=\Xi_{\mathrm{within\ ROI}}+\Xi_{\mathrm{cross\ ROI}}. \tag{3}
$$

这一步不是按 ROI 数量或结构连接强度人为分摊，而是把同一个 whole-system EI advantage 按不同 source 粒度逐级展开。令 $\mathbf{x}=\mathbf{s}_t\in\mathbb{R}^{200}$ 为干预 source，$\mathbf{y}=\mathbf{y}_{t+300}\in\mathbb{R}^{200}$ 为未来 target。对任意 source 索引集合 $\mathcal{A}$，统一定义该变量块关于同一未来全系统状态的有效信息为

$$
\operatorname{EI}(\mathcal{A}\to\mathbf{y})
\equiv I(\mathbf{x}_{\mathcal{A}};\mathbf{y}),
$$

其中所有 EI 都在同一个干预分布、同一个演化时间和同一个 target $\mathbf{y}$ 下计算。全系统量就是联合干预的 EI 超出 200 个单变量 EI 之和的部分：

$$
\Xi
=\operatorname{EI}(\{1,\ldots,200\}\to\mathbf{y})
-\sum_{j=1}^{200}\operatorname{EI}(\{j\}\to\mathbf{y})
=I(\mathbf{x};\mathbf{y})-\sum_{j=1}^{200}I(x_j;\mathbf{y}).
$$

对第 $r$ 个 ROI，记

$$
\mathcal{B}_r=\{r,100+r\},
$$

其中两个索引分别对应该 ROI 的 E 和 I 状态。先比较每个 ROI 的 E/I 联合 EI 与两个单变量 EI，可得 ROI 内分量：

$$
\Xi_{\mathrm{within\ ROI}}
=\sum_{r=1}^{100}
\left[
\operatorname{EI}(\mathcal{B}_r\to\mathbf{y})
-\operatorname{EI}(\{r\}\to\mathbf{y})
-\operatorname{EI}(\{100+r\}\to\mathbf{y})
\right],
$$

即

$$
\Xi_{\mathrm{within\ ROI}}
=\sum_{r=1}^{100}
\left[
I(\mathbf{x}_{\mathcal{B}_r};\mathbf{y})
-I(x_r;\mathbf{y})
-I(x_{100+r};\mathbf{y})
\right].
$$

这里每个方括号回答的是：同时干预并读取某个 ROI 的 E/I 状态，相比把 E 和 I 各自单独作为 source，额外提供了多少关于未来全系统状态的信息。

接着把每个 ROI 的 E/I 二元组视为一个 source 块。全脑联合 EI 超出 100 个 ROI 块 EI 之和的部分定义为跨 ROI 分量：

$$
\Xi_{\mathrm{cross\ ROI}}
=\operatorname{EI}(\{1,\ldots,200\}\to\mathbf{y})
-\sum_{r=1}^{100}\operatorname{EI}(\mathcal{B}_r\to\mathbf{y}),
$$

即

$$
\Xi_{\mathrm{cross\ ROI}}
=I(\mathbf{x};\mathbf{y})
-\sum_{r=1}^{100}I(\mathbf{x}_{\mathcal{B}_r};\mathbf{y}).
$$

两式相加时，每个 ROI 块的 $\operatorname{EI}(\mathcal{B}_r\to\mathbf{y})$ 一正一负严格抵消，只剩下 whole-system EI 减去 200 个单变量 EI，因而恢复式（3）。换言之，先把 E 和 I 合并为 ROI 所增加的 EI 归入 ROI 内；再把 100 个 ROI 合并为全脑所增加的 EI 归入跨 ROI。

在当前实验中，各 source 标量由相互独立的 Gaussian 干预生成，因此上述逐级合并产生的 EI 增量非负（有限样本估计误差除外），并与代码中的 Gaussian 估计式严格等价。实现层面，各项由同一批 2,048 对 source–target 样本和同一个拟合模型估计，并加入 $10^{-6}$ ridge；没有为 ROI 内和跨 ROI 改变 target 或重新定义信息量。

ROI 内分量为 3.105 bits（12.00%），跨 ROI 分量为 22.773 bits（88.00%），合计 25.878 bits。式（3）的最大数值闭合误差为 $1.6\times10^{-13}$ bits，且 24/24 个条件均为跨 ROI 大于 ROI 内。

按图 E 的 seed-blocked 汇总，ROI 内与跨 ROI 比例的跨 seed SD 均为 0.20 个百分点，说明 88.00% 的跨 ROI 优势不是由个别 seed 推高。

进一步按 Yeo-7 分组：

$$
\Xi_{\mathrm{cross\ ROI}}=
\Xi_{\mathrm{within\ network}}+
\Xi_{\mathrm{between\ networks}}. \tag{4}
$$

第二步完全沿用相同的 EI 增量逻辑，只是把 100 个 ROI 块继续合并成 7 个 Yeo 网络块。令 $\mathcal{R}_k$ 为网络 $k$ 所含的 ROI 集合，并令

$$
\mathcal{C}_k=\bigcup_{r\in\mathcal{R}_k}\mathcal{B}_r
$$

为该网络包含的全部 E/I 标量索引。式（4）中的“网络内”是七个网络各自的跨 ROI EI 增量之和，不包含前一步已经归入 $\Xi_{\mathrm{within\ ROI}}$ 的 ROI 内 E/I 分量：

$$
\Xi_{\mathrm{within\ network}}
=\sum_{k=1}^{7}
\left[
\operatorname{EI}(\mathcal{C}_k\to\mathbf{y})
-\sum_{r\in\mathcal{R}_k}\operatorname{EI}(\mathcal{B}_r\to\mathbf{y})
\right],
$$

即

$$
\Xi_{\mathrm{within\ network}}
=\sum_{k=1}^{7}
\left[
I(\mathbf{x}_{\mathcal{C}_k};\mathbf{y})
-\sum_{r\in\mathcal{R}_k}I(\mathbf{x}_{\mathcal{B}_r};\mathbf{y})
\right].
$$

每个方括号比较“整个网络作为联合 source”与“该网络中的 ROI 分别作为 source”，因此只保留同一网络内跨 ROI 联合干预所增加的 EI。

最后，全脑联合 EI 超出七个网络块 EI 之和的部分定义为网络间分量：

$$
\Xi_{\mathrm{between\ networks}}
=\operatorname{EI}(\{1,\ldots,200\}\to\mathbf{y})
-\sum_{k=1}^{7}\operatorname{EI}(\mathcal{C}_k\to\mathbf{y}),
$$

即

$$
\Xi_{\mathrm{between\ networks}}
=I(\mathbf{x};\mathbf{y})
-\sum_{k=1}^{7}I(\mathbf{x}_{\mathcal{C}_k};\mathbf{y}).
$$

两式相加时，七个网络块的 $\operatorname{EI}(\mathcal{C}_k\to\mathbf{y})$ 严格抵消，剩下的正是 whole-system EI 减去 100 个 ROI 块 EI，即 $\Xi_{\mathrm{cross\ ROI}}$。整个层级因而可以统一写成

$$
\sum_{j=1}^{200} I(x_j;\mathbf{y})
\xrightarrow{\ +\Xi_{\mathrm{within\ ROI}}\ }
\sum_{r=1}^{100} I(\mathbf{x}_{\mathcal{B}_r};\mathbf{y})
\xrightarrow{\ +\Xi_{\mathrm{within\ network}}\ }
\sum_{k=1}^{7} I(\mathbf{x}_{\mathcal{C}_k};\mathbf{y})
\xrightarrow{\ +\Xi_{\mathrm{between\ networks}}\ }
I(\mathbf{x};\mathbf{y}),
$$

其中每支箭头表示把更细的 source 块合并后新增的 EI，三支箭头依次对应 ROI 内、网络内跨 ROI 和网络间分量。图 G 后续使用 Shapley 值，只是把最后的网络间 EI 增量对称归因给七个网络；Shapley 不参与式（3）和式（4）本身的数值闭合。

网络内跨 ROI 分量为 5.622 bits，占总 $\Xi$ 的 21.73%；网络间分量为 17.151 bits，占总量的 66.27%。24/24 个条件均为网络间大于网络内。网络内分量以 Visual（2.288 bits）和 Somatomotor（1.677 bits）最高，其后为 Dorsal attention（0.610）、Default mode（0.577）、Salience/ventral attention（0.299）、Frontoparietal control（0.153）和 Limbic（0.018）。这些绝对量同时受网络所含 ROI 数影响，不能直接解释为单位 ROI 效应。

为把网络间整合与图 F 的网络内分量对照，图 G 对七个 Yeo 网络的全部 $2^7=128$ 个联盟进行精确 Shapley 归因。对任意网络子集 $S$，联盟价值定义为该子集的联合 EI 超出其成员网络 EI 之和的部分，单网络和空集价值为 0。网络 $i$ 的份额为

$$
\psi_i=\sum_{S\subseteq\mathcal{N}\setminus\{i\}}\frac{|S|!(7-|S|-1)!}{7!}\left[v(S\cup\{i\})-v(S)\right]. \tag{5}
$$

按 Shapley 效率性质，$\sum_i\psi_i=\Xi_{\mathrm{between\ networks}}$；代码在每个 seed–$G$ 条件上检验该闭合关系。这里的 $\psi_i$ 是多变量网络间整合的对称归因，不是成对网络边，也不是唯一的生物学因果归属。主图报告守恒的绝对 bits，网络规模诊断另按每 ROI 及每个跨网络连接机会数归一化记录在结果摘要中。

峰值邻近窗平均归因以 Default mode（3.541 bits）、Salience/ventral attention（3.416）和 Dorsal attention（2.909）最高，其后为 Frontoparietal control（2.788）、Somatomotor（2.587）、Visual（1.539）和 Limbic（0.371）。七项之和为 17.151 bits，与式（4）的网络间分量一致，最大逐条件闭合误差为 $3.6\times10^{-15}$ bits。按每 ROI 或每个跨网络连接机会数归一化后，前三名均为 Salience/ventral attention、Frontoparietal control 和 Dorsal attention，说明主要排序并非仅由网络规模造成。图 F 与图 G 因而形成明确对照：Visual 更突出网络内部模块化耦合，而显著性、默认、注意和控制相关网络承担更多跨网络整合归因。

<a id="dmf-topology"></a>

### 1.6 结构关联与稳定性

跨 ROI leverage 与加权结构强度呈正 Spearman 相关（$\rho=0.971$，$p=2.02\times10^{-62}$），ROI 内耦合与结构强度呈负相关（$\rho=-0.984$，$p=3.09\times10^{-75}$），ROI involvement 与结构强度呈正相关（$\rho=0.969$，$p=5.14\times10^{-61}$）。24 个条件的 ROI involvement 排名两两 Spearman 相关中位数为 0.984，最小值为 0.936，说明空间排序不由单一 seed 或单一 $G$ 驱动。

involvement 和 leverage 是留一块条件总相关下降量。它们是非负敏感性分数，但彼此重叠，不是互斥且可相加的信息原子。

图 H 进一步显示，跨 ROI leverage 不是集中在单一功能系统，而是在双侧形成多个空间锚点。最高值包括 SomMot parcel、枕叶视觉 parcel、Salience/ventral attention 的内侧及额岛 parcel，以及楔前叶/后扣带和部分扣带–控制 parcel。楔前叶/后扣带高值与结构连接研究中的 posterior medial structural core 相互印证：Hagmann et al.（2008）将后内侧和顶叶皮层识别为高 degree、strength 和 betweenness 的结构核心，van den Heuvel 与 Sporns（2011）也把双侧楔前叶列入富集俱乐部枢纽。额岛、扣带和控制区高值则与 connector-hub 文献中“跨模块连接支持整合，同时维持模块化”的机制相容（Bertolero et al., 2018）。不过，本实验中 leverage 与结构强度的相关达到 $\rho=0.971$，因此图 H 首先是当前 SC 骨架在干预动力学中的空间投影，不能把这种对应视为独立于结构强度的新验证。

更值得注意的是，图 H 同时突出视觉和躯体运动等单模态区域，以及楔前叶/后扣带、显著性和控制相关区域，并不沿“单模态到跨模态”的主梯度单调升高（Margulies et al., 2016）。这对“高协同只位于高阶联合皮层”的简单解释构成修正，却与 Varley et al.（2023）的结果相容：高阶协同子系统遍布皮层，其较稳定的参与热点包括枕极、楔前叶和扣带区域，而且高协同组合往往跨越经典功能网络。结合图 F–G，可得到一个新的层级区分：视觉和 SomMot parcel 可以因结构嵌入而具有较高的跨 ROI leverage，但这不等于其所属网络承担最多的跨网络归因；后者仍以 Default、Salience/ventral attention、Dorsal attention 和 Frontoparietal control 更突出。因此，图 H 定位的是“移除某个 ROI 会使跨区条件总相关下降多少”，而不是认知层级、网络间 Shapley 份额或局部协同的直接脑图。

<a id="dmf-insights"></a>

### 1.7 综合结论与文献对照

**第一，$\Xi$ 峰反映的是“联合干预优势”最大，而不是系统信息总量最大。**在 $G=1.3$，whole EI 已由 $G=0$ 的 219.636 bits 降至 176.926 bits，单变量 EI 之和则由 211.184 bits 更快降至 150.795 bits；两者差值因而达到峰值。这说明耦合首先削弱单个变量独立解释未来全系统状态的能力，同时暂时保留联合状态中的关系信息。该解释与 causal emergence 将有效信息写成 determinism 与 degeneracy 权衡的思路一致，也与“信息转换”框架中局部信息转化为高阶协同的概念相容（Hoel et al., 2013；Varley & Hoel, 2022）。本实验新增的是：在具有结构连接约束的 DMF 中，这种联合优势沿耦合参数形成可重复的非单调峰。

**第二，$\Xi$ 峰可以作为动力学转变的邻近标志，但当前数据不足以把它称为严格临界点。**8/8 个 seed 的 $\Xi$ 峰均位于 $G=1.3$，平均发放率最大离散导数则位于 $G=1.5$；二者相邻而不重合。whole-brain 模型常在稳定性边界或亚稳态附近产生丰富动力学，临界性文献也强调应同时检查 susceptibility、长程相关、尺度不变性或 Jacobian 稳定性，而不能只凭单条峰形判定（Deco et al., 2011；Breakspear, 2017；Cocchi et al., 2017）。因此本文使用“转变前沿”“临界窗”或“临界性候选带”，不把 $G=1.3$ 宣称为已证明的相变点。

**第三，峰值邻近窗中的主要信息结构跨越功能网络边界。**跨 ROI 分量占总 $\Xi$ 的 88.00%，其中网络间分量占总量的 66.27%，且 24/24 个 seed–$G$ 条件中网络间分量均高于网络内分量。该结果与人脑高阶信息研究的总体方向一致：Luppi et al.（2022）发现协同信息更集中于跨模态、整合性皮层，Varley et al.（2023）发现高协同子系统通常跨越多个经典功能网络。区别在于，已有工作主要分析观察性 fMRI 中的统计协同；这里的 $\Xi$ 来自最大熵干预下的未来状态可区分性，并具有 ROI→网络→网络间的精确闭合分解。因此它提供的是结构约束动力学中的干预式协同证据，而不是对既有 O-information 或 PID 空间图的重复计算。

**第四，Visual 的高值只发生在网络内部，不表示它是最高级的全脑整合系统。**Visual 在图 F 的网络内跨 ROI 分量最高（2.288 bits），但在图 G 的网络间 Shapley 归因仅为 1.539 bits，明显低于 Default mode、Salience/ventral attention 和 Dorsal attention。视觉皮层具有高密度、拓扑规则且强同模块的局部连接，在以绝对 bits 汇总多个 parcel 的指标中容易形成较大的内部联合量；这更接近稳定的专门化模块，而非跨系统广播。真正与跨网络整合相关的是 Default mode（3.541 bits）和 Salience/ventral attention（3.416 bits）等网络。后者与显著性网络切换和控制模型相符（Menon & Uddin, 2010），前者及注意/控制网络的贡献也与 connector hub、rich-club 和动态整合研究相容（van den Heuvel & Sporns, 2011；Shine et al., 2016；Bertolero et al., 2018）。所以最有信息量的结论不是“Visual 最高”，而是 **Visual 呈现高网络内、低网络间的模块化特征；Default/Salience 呈现较低网络内、较高网络间的整合特征**。

**第五，结构中心性可能决定协同从局部向全局重新分配的空间通道。**结构强度与跨 ROI leverage 呈强正相关（$\rho=0.971$），与 ROI 内耦合呈强负相关（$\rho=-0.984$）。这比“强连接脑区拥有更多信息”的一般陈述更具体：结构嵌入越强，归因越从局部 E/I 闭环转向跨区联合预测。该模式与 rich-club/connector-hub 文献关于结构骨架支持全局通信的结果一致，但这里仍是相关证据，不能据此断言结构强度单独造成了 $\Xi$ 重分配。最直接的后续检验应是保持权重分布不变的 degree/strength-preserving rewiring，或对高强度节点实施匹配的虚拟 lesion。

**第六，相似的 $\Xi$ 峰形可以由不同的信息机制产生。**DMF 在发放率转折后表现为 whole-source 与 singleton-sum 的 determinism、degeneracy 四项共同下降；Kuramoto 对照则在强同步端出现 degeneracy 和 singleton 重复读出的持续增长。两者都可形成“先升后降”的 $\Xi$，但峰后动力学含义不同。这说明峰形本身不是机制指纹；将 determinism 与 degeneracy 分开呈现是必要的附录验证，也构成本实验相较只报告单一整合曲线的一个方法学增量。

**第七，83 ROI 与 100 ROI 的一致峰形支持跨连接组复现，但不是节点数消融。**在以有效耦合 $G\rho(\mathbf{C})$ 和每个 scalar source 的 $\Xi$ 对齐后，新 100 ROI 矩阵的峰位在 8/8 个 seed 中都高于旧矩阵，跨 ROI 份额也由 68.67% 增至 88.00%。然而两套连接组同时改变了节点数、密度、方向性、权重分布与谱尺度，因此这些差异只能表述为跨连接组的鲁棒性与描述性变化，不能归因于“100 ROI 更好”或单一拓扑因素。

**第八，受体数据为下一阶段提供了有文献依据但尚未使用的异质性层。**数据包中的 `average.csv` 含 19 个受体/转运体指标，但没有进入当前 DMF。Hansen et al.（2022）表明受体/转运体空间分布与结构连接、功能连接及神经动力学相关；这支持未来把区域受体谱映射为局部增益、时间常数或 E/I 参数的受控实验。当前结果不能被解释为受体梯度导致的网络差异，除非完成“仅改变受体调制项、其余参数固定”的比较并使用空间自相关保持 null。

综合而言，本实验最值得强调的新启发是：**临界邻近的全脑协同并非简单地“信息更多”，而是信息从单节点可读出形式转向跨 ROI、尤其跨功能网络的联合可读出形式；同时，专门化模块与整合网络在网络内和网络间归因上呈现可解释的双重分离。**这一结论把动力学扫描、干预式有效信息和多尺度脑网络归因连接在同一条可审计证据链上。

下一步最优先的三个验证是：（1）在 $G=1.1$–1.7 之间加密扫描，并同时计算 susceptibility、metastability 与 Jacobian 稳定性；（2）对 93 个个体 SC 分别复现峰位和网络归因，以区分群体均值效应与个体差异；（3）在保持 SC、干预和估计器不变时，仅加入受体驱动的局部参数异质性，并以空间自相关保持 null 检验增量解释力。

<a id="dmf-limits"></a>

### 1.8 解释边界

当前证据支持：Schaefer100 群体 SC 驱动的 DMF 在发放率转折附近出现 $\Xi$ 峰，并表现出更强的跨 ROI、尤其跨功能网络整合。

当前证据不支持将新矩阵断言为某一特定队列、流线数或单位，将 83/100 差异解释为纯粹的 ROI 数量效应，把模拟样本解释为真实受试者脑活动，或把 Gaussian EI 的绝对 bits 直接等同于 TM 等非线性估计器结果。在取得上游标签文件前，ROI 顺序也只能维持“高度一致的推断”状态。

此外，图 1 的 24 个条件来自 8 个模拟 seed 与 3 个固定耦合值，不是 24 名独立受试者；图 C 的 $\Phi^{WMS}$ 与 $\Xi$ 来自同一模拟和同类 Gaussian 估计流程，不能当作独立外部验证；Shapley 结果依赖当前联盟价值定义，且绝对网络贡献仍受网络规模影响。当前研究也尚未检验个体 SC、方向性连接、其他图谱分辨率、不同干预分布与预测时距，因而应把结论限定为当前模型和分析契约下的机制性发现。

<a id="dmf-references"></a>

### 1.9 参考文献

1. Hoel EP, Albantakis L, Tononi G. Quantifying causal emergence shows that macro can beat micro. *PNAS*. 2013;110:19790–19795. [doi:10.1073/pnas.1314922110](https://doi.org/10.1073/pnas.1314922110)
2. Varley TF, Hoel EP. Emergence as the conversion of information: a unifying theory. *Philosophical Transactions of the Royal Society A*. 2022;380:20210150. [doi:10.1098/rsta.2021.0150](https://doi.org/10.1098/rsta.2021.0150)
3. Yang M, et al. Partial Effective Information Decomposition for Synergistic Causality. *arXiv*. 2026. [doi:10.48550/arXiv.2605.03267](https://doi.org/10.48550/arXiv.2605.03267)
4. Deco G, Jirsa VK, McIntosh AR. Emerging concepts for the dynamical organization of resting-state activity in the brain. *Nature Reviews Neuroscience*. 2011;12:43–56. [doi:10.1038/nrn2961](https://doi.org/10.1038/nrn2961)
5. Breakspear M. Dynamic models of large-scale brain activity. *Nature Neuroscience*. 2017;20:340–352. [doi:10.1038/nn.4497](https://doi.org/10.1038/nn.4497)
6. Cocchi L, Gollo LL, Zalesky A, Breakspear M. Criticality in the brain: a synthesis of neurobiology, models and cognition. *Progress in Neurobiology*. 2017;158:132–152. [doi:10.1016/j.pneurobio.2017.07.002](https://doi.org/10.1016/j.pneurobio.2017.07.002)
7. Luppi AI, Mediano PAM, Rosas FE, et al. A synergistic core for human brain evolution and cognition. *Nature Neuroscience*. 2022;25:771–782. [doi:10.1038/s41593-022-01070-0](https://doi.org/10.1038/s41593-022-01070-0)
8. Varley TF, Pope M, Faskowitz J, Sporns O. Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex. *Communications Biology*. 2023;6:451. [doi:10.1038/s42003-023-04843-w](https://doi.org/10.1038/s42003-023-04843-w)
9. Luppi AI, Rosas FE, Mediano PAM, Menon DK, Stamatakis EA. Information decomposition and the informational architecture of the brain. *Trends in Cognitive Sciences*. 2024;28:352–368. [doi:10.1016/j.tics.2023.11.005](https://doi.org/10.1016/j.tics.2023.11.005)
10. Menon V, Uddin LQ. Saliency, switching, attention and control: a network model of insula function. *Brain Structure and Function*. 2010;214:655–667. [doi:10.1007/s00429-010-0262-0](https://doi.org/10.1007/s00429-010-0262-0)
11. van den Heuvel MP, Sporns O. Rich-club organization of the human connectome. *Journal of Neuroscience*. 2011;31:15775–15786. [doi:10.1523/JNEUROSCI.3539-11.2011](https://doi.org/10.1523/JNEUROSCI.3539-11.2011)
12. Shine JM, Bissett PG, Bell PT, et al. The dynamics of functional brain networks: integrated network states during cognitive task performance. *Neuron*. 2016;92:544–554. [doi:10.1016/j.neuron.2016.09.018](https://doi.org/10.1016/j.neuron.2016.09.018)
13. Bertolero MA, Yeo BTT, Bassett DS, D'Esposito M. A mechanistic model of connector hubs, modularity and cognition. *Nature Human Behaviour*. 2018;2:765–777. [doi:10.1038/s41562-018-0420-6](https://doi.org/10.1038/s41562-018-0420-6)
14. Hansen JY, Shafiei G, Markello RD, et al. Mapping neurotransmitter systems to the structural and functional organization of the human neocortex. *Nature Neuroscience*. 2022;25:1569–1581. [doi:10.1038/s41593-022-01186-3](https://doi.org/10.1038/s41593-022-01186-3)
15. Hagmann P, Cammoun L, Gigandet X, et al. Mapping the structural core of human cerebral cortex. *PLoS Biology*. 2008;6:e159. [doi:10.1371/journal.pbio.0060159](https://doi.org/10.1371/journal.pbio.0060159)
16. Margulies DS, Ghosh SS, Goulas A, et al. Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS*. 2016;113:12574–12579. [doi:10.1073/pnas.1608282113](https://doi.org/10.1073/pnas.1608282113)


<a id="hcp-wm"></a>

## 2. HCP 任务态 $\Xi$、认知画像与脑区分布

<a id="hcp-task-evoked-xi-main"></a>

### 2.1 主要结果：Schaefer-1000 任务诱发 PCA–$\Xi$、层级分解与认知画像

本实验先在每个 Yeo7 网络内提取任务诱发 PCA 方向。对任务态，PCA 只在前 75% 时间点的

$$
\mathbf{U}_{sc}
=\mathbf{X}^{\mathrm{retained}}_{sc}
-\mathbf{X}^{\mathrm{regressed}}_{sc}
$$

上拟合，再用所得载荷投影原始 $\mathbf{X}^{\mathrm{retained}}_{sc}$。因此任务态同时读取 retained 和 regressed 两组数据，但不是将两者拼接后共同做 PCA：`retained - regressed` 只负责拟合 PCA 方向，完整 retained 时序负责生成后续动力学状态。task GLM 移除的成分由此决定降维方向，而动力学仍保留完整任务信号。REST 没有任务回归版本，故在自身前 75% 时序上拟合并投影 PCA。正文使用 Schaefer-1000；最终每个网络保留第一主成分（$k=1$），形成七维网络状态 $\mathbf{x}_t$。任务态 PC1 的平均累计解释方差为 64.74%，REST 为 37.77%。

共享动力学配置为 $(k,p,\alpha)=(1,3,1)$。三阶网络历史

$$
\mathbf{h}_t
=\left[
\mathbf{x}_t^\top,
\mathbf{x}_{t-1}^\top,
\mathbf{x}_{t-2}^\top
\right]^\top
\in\mathbb{R}^{21}
$$

用于预测下一时刻七网络状态，模型为 $\mathbf{x}_{t+1}=\mathbf{A}\mathbf{h}_t+\mathbf{b}+\boldsymbol{\varepsilon}_t$。每名被试、每个状态分别在前 75% 时间段拟合线性 $\Delta$-Ridge，后 25% 只用于预测诊断。连续 EI 使用线性高斯 affine-TM 的协方差 log-det 闭式，标准化 source 的干预协方差固定为单位阵，结果单位为 bits。

将 21 个“网络 $\times$ 滞后”标量视为最细 source，其相对完整历史 source 的总协同定义为 system-level $\Xi$：

$$
\Xi_{sc}
=EI(\mathbf{h}_t;\mathbf{x}_{t+1})
-\sum_{j=1}^{21}EI(h_{t,j};\mathbf{x}_{t+1}).
$$

该量对应 PEID 在 singleton source partition 下的 system-level synergy。本文统一使用 $\Xi$；“system-level $\Xi$”特指这里的 21 个网络–滞后 singleton source partition，而附录 I.1 的“raw $\Xi$”使用固定七维 PC1 历史表征。两者符号相同，但 source 构造和可比范围不同。Schaefer-1000 的 232 个“被试 $\times$ 状态”模型平均 held-out RMSE/持久性基线比为 0.903，其中 212/232 个模型优于持久性基线。

合并主图 a 汇总整体幅度。29 名共同被试中，REST 的 system-level $\Xi$ 均值为 6.985 bits，七任务为 4.288--5.686 bits。REST 与每个任务的双侧配对 Wilcoxon 检验在七项内作 BH 校正后均显著，最小均值差仍为 REST--SOCIAL 的 1.299 bits（$q=0.0168$）。因此，当前表征首先保留了 **REST 整体 $\Xi$ 显著高于全部任务态** 的幅度结论。

为解释整体 $\Xi$ 如何分配，三个滞后在每个 Yeo7 网络内绑定为模块 $M_g$。网络内协同与跨网络协同分别定义为

$$
\Xi_g^{\mathrm{within}}
=EI(M_g;\mathbf{x}_{t+1})
-\sum_{j\in M_g}EI(h_{t,j};\mathbf{x}_{t+1}),
$$

$$
\Xi^{\mathrm{cross}}
=EI(\mathbf{h}_t;\mathbf{x}_{t+1})
-\sum_{g=1}^{7}EI(M_g;\mathbf{x}_{t+1}).
$$

跨网络部分用精确 Shapley 值分配。令 $\mathcal{N}$ 为七网络集合，联盟价值为

$$
v(S)
=EI(M_S;\mathbf{x}_{t+1})
-\sum_{h\in S}EI(M_h;\mathbf{x}_{t+1}),
\qquad v(\varnothing)=0.
$$

网络 $g$ 的 Shapley 值遍历其余六网络形成的全部 $2^6=64$ 个联盟：

$$
\operatorname{Shapley}_g
=\sum_{S\subseteq\mathcal{N}\setminus\{g\}}
\frac{|S|!\,(6-|S|)!}{7!}
\left[v(S\cup\{g\})-v(S)\right].
$$

它满足 $\sum_g\operatorname{Shapley}_g=\Xi^{\mathrm{cross}}$。网络 $g$ 的守恒归因为

$$
C_g
=\Xi_g^{\mathrm{within}}
+\operatorname{Shapley}_g,
\qquad
\sum_{g=1}^{7}C_g=\Xi.
$$

例如，`sub-100206` 的 Schaefer-1000 LANGUAGE system-level $\Xi$ 为 4.690 bits，Control 的守恒归因为 0.721 bits，占 15.37%。主图 c 对每名被试先计算 $P_{scg}=C_{scg}/\Xi_{sc}$，再对 29 名被试取平均，所以每列严格合计 100%。

![HCP Schaefer-1000 任务态 Xi、认知画像、层级分解与任务表现](../../results/hcp_schaefer1000_task_evoked_xi_replication/final/task_evoked_xi_main_combined.png)

*图 2｜Schaefer-1000 主结果。a：REST 与七任务的 system-level $\Xi$；b：主要 greedy atom 的绝对贡献；c：system-level $\Xi$ 的七网络平均组成份额；d：同一 29 名被试的四个冻结认知因子；e--f：一般认知分别与 LANGUAGE、MOTOR 全七网络 atom 绝对贡献的逐被试关系；g--h：LANGUAGE 中 SomMot+Limbic+Cont 固定组合总协同分别与 Story、Math ACC 的关系；i：WM 中 Cont+Default 固定组合总协同与 0-back ACC 的关系，并在图内同时标出 2-back 相关和两条件差异检验。d 中每个认知因子在 29 人内部单独标准化；g 和 i 为显示重叠而对横坐标加入固定轻微抖动，统计仍使用原始分数。全部虚线只作线性视觉引导，统计量均为 Spearman 秩相关。*

图 2a--c 显示任务不只改变整体协同强度，也重组协同的内部结构。REST 的 system-level $\Xi$ 高于全部七项任务，与无外部任务约束时自发动力学保留更大范围的全系统联合可预测性相容。进入任务后，整体 $\Xi$ 普遍下降，但七个网络并非同比例收缩：REST 中 SomMot、DorsAttn 和 Default 的平均份额较高，LANGUAGE 的 Control 份额升至 19.4%，RELATIONAL 与 SOCIAL 的 DorsAttn 份额分别达到 20.2% 和 21.3%。仅比较七任务，7/7 个网络仍有经 BH 校正的状态效应。因此，任务不仅压缩协同总量，也把剩余协同重新分配到与当前计算需求相匹配的网络。

层级 atom 进一步显示，这种重组发生在网络组合层面。REST 的全七网络 atom 为 1.047 bits；多数任务的主要高阶组合集中在缺少 Limbic 的六网络核，绝对贡献为 0.702--1.096 bits。MOTOR 同时保留全七网络与缺 Limbic 六网络两个较大的高层组合。任务态因此不是简单关闭全脑整合，而是把广泛协同约束到少数可重复出现的高阶组合中。

图 2e--f 给出一般认知的任务依赖关系。LANGUAGE 全七网络 atom 与一般认知正相关（$\rho=+0.578$，逐项置换 $p=0.00120$，两项 Holm $p=0.00240$）；MOTOR 保持负方向，但单项证据较弱（$\rho=-0.306$，逐项置换及 Holm $p=0.107$）。两个相关之差为 $0.884$，双侧置换 $p=0.000400$。这个结果不表示认知越高，全脑协同越多或越少，而是说明认知优势可能表现为按任务需求重新配置高阶协同：复杂语言加工偏向更广泛的联合组织，重复运动执行较少依赖最高层级的全系统绑定。这一解释与任务依赖的整合--分离和高效重配置文献相容（[Shine et al., 2016](https://doi.org/10.1016/j.neuron.2016.09.018)；[Schultz and Cole, 2016](https://doi.org/10.1523/JNEUROSCI.0358-16.2016)），但功能连接与 PEID atom 并非同一测量。

Schaefer-500 与 Schaefer-1000 的群体状态结果高度一致：56 个“状态 $\times$ 网络”平均归因份额的跨粒度相关为 0.955，平均绝对差为 0.597 个百分点，每个状态的 top-3 greedy atom 平均共享 2.50/3 个。正文因此以 1000 分区为主，500 分区旧主图、领域认知探索和完整跨粒度说明统一移至附录 K。

#### LANGUAGE 内 Story--Math 分化：SomMot+Limbic+Cont 固定组合

HCP 的 LANGUAGE run 并不是单一语言条件，而是在同一次扫描中交替呈现 Story 和 Math。Story 要求被试听取 5--9 句的伊索寓言改编故事，再从两个选项中判断故事主题；Math 同样使用听觉呈现和二选一按键反应，但要求连续完成加减运算，而且题目难度会按被试表现自适应。这一设计有意用 Math 控制听觉输入、持续加工和反应选择，同时突出 Story 的叙事意义提取，因此 Story--Math 也是该范式最初用于定位前颞叶语义加工的核心对照（[Binder et al., 2011](https://doi.org/10.1016/j.neuroimage.2010.09.048)；[Barch et al., 2013](https://doi.org/10.1016/j.neuroimage.2013.05.033)）。

这里检验的不是某个网络的平均激活，也不是两两功能连接。令

$$
S=\{\mathrm{SomMot},\mathrm{Limbic},\mathrm{Cont}\},
$$

且 $M_g$ 表示网络 $g$ 的三个滞后 source 绑定后形成的模块，则固定组合总协同为

$$
\Xi_S
=EI(M_S;\mathbf{x}_{t+1})
-\sum_{g\in S}EI(M_g;\mathbf{x}_{t+1}).
$$

$\Xi_S$ 测量三个网络的联合历史对下一时刻七网络状态所提供的、不能由三个网络分别预测后简单相加得到的信息。它与前文 greedy atom 的路径残差不同：本节对 29 名被试都直接读取同一个预先指定组合，因此没有“某组合只在少数被试路径中出现”的稀疏覆盖问题。

| Story $\rho$（逐项置换 $p$） | Math $\rho$（逐项置换 $p$） | $\Delta\rho=\rho_{\mathrm{Story}}-\rho_{\mathrm{Math}}$ | Williams 原始 $p$ | 三候选 Holm $p$ | $\Delta\rho$ bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| $+0.318$（0.0927） | $-0.222$（0.2442） | $+0.539$ | 0.0298 | 0.0895 | $[+0.055,+0.981]$ |

图 2g--h 直接给出 Schaefer-1000 的两个脑--行为散点。结果支持一个稳定的**方向模式**，而不是两个分别都显著的单项相关：SomMot+Limbic+Cont 与 Story ACC 为正，与 Math ACC 为负，两个相关之差 $\Delta\rho=0.539$；其原始差异检验和 bootstrap 区间支持正差异，但三个预指定组合内 Holm $p=0.0895$，未达到严格复现阈值。因此最准确的表述是：**Story 正、Math 负是优先验证的配置模式，而不是两个已经确认的单项脑--行为效应。**Story 只有 4 个不同取值且中位数为 100%，图 2g 中的大量重叠点和天花板效应必须与相关系数一起解读。

这个方向模式不是由单个极端被试造成的。逐一剔除任一被试后，Story $\rho$ 始终为 $[+0.236,+0.420]$、Math $\rho$ 为 $[-0.286,-0.150]$，$\Delta\rho$ 为 $[+0.429,+0.657]$。Story 与 Math ACC 本身只有很弱的正相关（$\rho=+0.136$，$p=0.481$），所以脑指标的相反方向不能简化为“Story 做得好的人必然 Math 做得差”。作为事后敏感性检查，在秩空间控制另一项 ACC 后，Story 方向仍为正（partial $\rho=+0.360$），Math 方向仍为负（partial $\rho=-0.282$）；这些 partial 结果未预先注册，也未作多重比较校正。与 Schaefer-500 的效应量、逐一剔除范围和跨分区个体一致性统一列于附录 K。

**为什么 Story 是正相关？**一个连贯故事要求被试把连续听觉语音转成语义单元，跨多句话维持角色、事件和因果关系，再把当前内容与已有概念知识结合，最后根据整体主题作出选择。SomMot 在这里不应窄化为“做按键动作”：听觉语言加工包含语音--感觉运动映射，运动和体感皮层也可携带与语音知觉相关的信息（[Hickok et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21315253/)）。Cont 则可作为随任务目标改变连接模式的灵活枢纽，维持当前问题、选择相关语义并协调分布式网络（[Cole et al., 2013](https://doi.org/10.1038/nn.3470)）；叙事理解研究也观察到额下回、额顶控制区与颞叶、扣带和感觉运动区域之间随语境改变的耦合（[Smirnov et al., 2014](https://doi.org/10.1016/j.neuropsychologia.2014.09.007)）。

Limbic 在本结果中尤其需要准确命名。Yeo7 的 Limbic 主要覆盖眶额和前颞皮层，它不是“情绪脑”的同义词，更不能直接等同于杏仁核或海马。前颞叶是原 Story--Math 范式希望增强的语义整合区域；近期多回波 fMRI 证据还提示，传统 Yeo Limbic 的大部分前颞和眶额区域可能更适合看作扩展 Default 系统的一部分，并指出这些区域在常规单回波 BOLD 中信号质量较差（[Girn et al., 2024](https://doi.org/10.1162/netn_a_00385)）。因此，当前正相关最通顺的神经解释是：**Story 表现较好的人，在听觉/感觉运动表征、前颞--眶额的概念与情境表征、以及目标导向控制之间形成了更强的联合时间约束；三者共同提供的信息超过各自信息的简单相加。**这与“叙事理解需要跨句意义整合和控制系统协调”相容，但不能把协同值进一步定位到某个具体前颞亚区或某一种语义过程。

**为什么 Math 是负相关？**负相关并不表示 Math 表现好的人“控制网络更弱”，更不表示他们的全脑协同更低。它只表示在 LANGUAGE 动力学中，SomMot、Limbic 与 Cont 这一个固定三网络组合的额外联合信息较少。连续心算更直接依赖双侧顶内沟、上/下顶叶以及额叶--扣带控制系统；运算复杂度增加时，顶内沟、额下回和扣带的参与也会上升（[Kong et al., 2005](https://doi.org/10.1016/j.cogbrainres.2004.09.011)）。这些关键算术通路并不完整包含在当前三网络组合中，尤其没有显式纳入 DorsAttn。由此，一个合理但仍待验证的机制是：Math 表现较好的人把动力学更集中到数值--顶叶和任务控制通路，较少需要把前颞/眶额情境表征与感觉运动、控制系统长期绑定；这可表现为当前组合协同较低，却不意味着计算所需的其他组合协同也较低。

Story 正、Math 负因而可以组成一个连贯的**任务配置故事**：同样的听觉输入和按键输出之上，Story 更依赖把语音序列、概念情境和目标选择绑定成整体叙事模型，Math 更依赖对数字和运算步骤进行受控、相对专门化的串行操作。SomMot+Limbic+Cont 的高协同可能标记前一种跨系统整合配置，帮助 Story；对后一种计算配置，它可能不是必要资源，甚至可能反映较不经济的广泛绑定。这里的重点不是“协同越多越好”，而是**同一高阶组合是否匹配当前任务的计算结构**。

这个故事目前足够通顺，值得作为优先候选加强验证，但还不能写成已证实的神经机制。第一，1000 分区只更换空间粒度，仍是同一 29 人，不是独立队列复现。第二，Story ACC 只有 4 个取值、范围为 75%--100%、中位数为 100%，明显的天花板和秩并列限制了效应分辨率；Math 有 19 个取值且采用自适应难度，两种 ACC 的测量尺度也不完全对称。第三，候选来自同一批被试的 500 分区探索，1000 分区检验只能提供跨图谱稳健性。第四，当前模型没有控制头动、信号质量、人口学变量和 HCP 家系结构，Yeo Limbic 还受前颞/眶额信号质量与网络边界不确定性影响。下一步最有价值的确认实验应在看新数据前固定 SomMot+Limbic+Cont、Story 正/Math 负和 $\Delta\rho>0$ 三个方向，优先使用另一 run 或独立 cohort，并采用家系交换块置换、运动与信号质量协变量；同时加入 DorsAttn/顶叶算术组合，直接检验“Story 的跨系统整合”与“Math 的顶叶专门化”是否形成可区分的双重模式。

#### WM 内 0-back--2-back 分化：Cont+Default 固定组合

HCP 的 WM run 同时包含 0-back 和 2-back。0-back 要求被试持续监测当前刺激并识别预先指定的固定目标，主要负荷是警觉、目标匹配和规则维持；2-back 则要求持续保存并更新最近两个刺激的顺序，额外增加了工作记忆更新和干扰抑制。为了判断先前 WM 总准确率相关究竟更接近一般任务控制还是高负荷工作记忆，这里不再搜索新组合，而是固定 Cont+Default，并分别关联两项条件准确率。被试 `104012` 缺少 2-back 分数，因此条件比较使用共同的 28 人。

| 空间粒度 | 2-back $\rho$（两条件 max-T $p$） | 0-back $\rho$（两条件 max-T $p$） | $\Delta\rho=\rho_{\mathrm{2back}}-\rho_{\mathrm{0back}}$ | 配对置换 $p$ |
|---|---:|---:|---:|---:|
| Schaefer-1000 | $-0.131$（0.7402） | **$-0.567$（0.00421）** | $+0.436$ | **0.00864** |
| Schaefer-500 | $-0.248$（0.3491） | **$-0.494$（0.0156）** | $+0.246$ | 0.1351 |

图 2i 展示正文 Schaefer-1000 结果：0-back 表现越高，WM 状态下 Cont+Default 固定组合总协同越低。该负相关通过针对 0-back 和 2-back 两项检验的 max-T 校正，而且两条相关的配对差异也显著。Schaefer-500 保留相同方向并独立达到 0-back 的两条件校正阈值，但条件差异未显著。因此，跨粒度最稳定的发现是 **0-back 与 Cont+Default 协同负相关**；“0-back 明显强于 2-back”的条件交互在 Schaefer-1000 中成立，在 500 分区中只有同方向支持。

年龄和性别不能解释 Schaefer-1000 的主模式：校正后 0-back 为 $\rho=-0.597$、max-T $p=0.00138$，2-back 为 $\rho=-0.163$、max-T $p=0.633$，条件差异为 $0.434$、配对置换 $p=0.00536$。反过来，在秩空间进一步用 0-back、年龄和性别解释 2-back 后，Cont+Default 与剩余 2-back 差异只有 partial $\rho=+0.129$、置换 $p=0.514$。这说明当前脑指标关联的主要不是 2-back 特有的更新能力，而更接近两种 WM 条件共享、但由 0-back 更纯粹测量的持续注意、固定目标识别和规则保持。

从神经功能上看，Control 支持任务规则和目标导向调节，Default 更常参与内部思维和与当前外部目标无关的加工。较好的 0-back 表现伴随较低的 Cont+Default 额外联合预测信息，与两个系统在简单外部目标监测中保持更清晰的功能分工、减少不必要的跨系统绑定相容。这里的“较低协同”并不等于 Default 活动更低，也不能直接证明两网络之间的功能连接减弱或存在抑制；它只表示两网络历史联合后超出各自信息简单相加的部分较少。因此，最准确的解释是 **高效注意和目标匹配可能依赖 Control 与 Default 的较低高阶联合约束**，而不是“整体协同越低，认知越好”。

这一结果仍有明确边界。当前 Cont+Default 指标由完整 WM run 估计，0-back 和 2-back 只在行为端分开；所以本分析证明的是同一 WM 脑指标对两类行为分数具有不同关联，尚未证明两个 block 内部的脑网络动力学本身不同。真正的条件神经验证需要取得每名被试的 0-back/2-back 事件时序，分别重算两套 Cont+Default 指标，并检验“2-back 脑指标--2-back 表现”和“0-back 脑指标--0-back 表现”的配对交互。

#### Schaefer-500 探索与个体诊断（附录 K）

Schaefer-500 的逐被试热图、领域认知筛选、指定组合补充、全 120 组合扫描和扩展候选均移至附录 K。它们保留候选生成、个体一致性和多重比较审计，但不再作为正文主结果。

<a id="hcp-wm-phi"></a>


### 2.2 协同核分布及静息态对照

WM 的模型迁移、circular-shift null、协同核分布及其与静息态的配对比较移至附录 H。静息态 Schaefer-500 的表征、模型选择与模块分解见附录 F；Schaefer-1000 的对应复现见附录 G。

<a id="hcp-all-tasks"></a>

### 2.3 七任务 raw $\Xi$ 历史表征参照

固定 Yeo7-PC1 表征下的七任务 raw $\Xi$ 排序、受试者内比较及 REST–任务长度匹配方差检验移至附录 I。第 2.1 节的任务诱发 PCA–$\Xi$ 分解是当前任务空间归因主结果；附录 I 仅保留历史表征与参数敏感性参照。

<a id="hcp-task-specific-regions"></a>

### 2.4 七任务态的 Schaefer-500 任务特异脑区分布

前述 EI/$\Xi$ 分析回答的是网络历史状态如何联合预测未来，而不是经典任务定位问题。它把 500 个 parcel 压缩成七个网络 PC1，逐状态标准化后拟合整段平稳一步动力学；这个流程主动移除了均值和整体幅度，忽略任务事件时序，也平均掉同一 Yeo7 网络内部的 parcel 异质性，因此不直接用于解释任务态之间的脑区分布差异。

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


REST 没有 `taskRetained`/`taskRegressed` 配对，因此不存在“被 task GLM 解释的方差”，不能与七任务直接共用 TEVF 定义。为避免把 REST 人为设为零，附录 J 另以所有状态都可计算的 parcel 时间方差构造补充空间参照。该共同方差口径包含自发活动、任务活动和残余混杂，只用于说明 REST 与任务的相对空间分布，不替代本节以 task GLM 成分定义的任务态 TEVF 主结果。

<a id="discussion"></a>

## 3. 讨论：解释边界与可复现性

<a id="discussion-scope"></a>

### 3.1 结论的适用范围

- DMF 结果支持 Schaefer100 群体平均对称 SC 在无裁剪、$[0.30,0.70]^{200}$ 全状态干预下，于发放率转折附近形成 $\Xi$ 峰，并以跨 ROI、尤其跨网络分量为主。数据包缺少上游标签、SC 单位和纤维追踪元数据，ROI 顺序仍是经矩阵相关审计支持的推断；83/100 差异也不是节点数或拓扑的单因素效应。
- HCP 静息态结果来自 REST1_LR；任务态主 $\Xi$ 分析覆盖七种 `taskRetained` LR 任务。正文以 Schaefer-1000 为主，Schaefer-500 的原合并图和探索性认知筛选保留在附录 K。任务 PCA 在 retained--regressed 差值上拟合，再投影 retained；最终共享参数为 $(k,p,\alpha)=(1,3,1)$。该参数先在 8 名被试上筛选，并在未参与筛选的 21 名被试上确认；29 名完整汇总保持 REST 的 system-level $\Xi$ 显著高于全部任务，同时显示七网络份额具有任务状态效应，并给出主要 greedy atom 的描述性组成。相对地，附录 I 的 $(p,\alpha)=(8,10)$ raw $\Xi$ 与附录 E 的 25 点扫描保留为历史表征和参数敏感性参照，不再作为任务空间归因的主结果。长度匹配检验表明 REST 高方差只在 EMOTION 与 MOTOR 上最稳定。30 名被试的 Schaefer-500 TEVF 仍直接描述 task GLM 移除的 parcel 级时间能量，而新 $\Xi$ 分解描述沿任务诱发 PCA 方向观察到的完整 retained 动力学；二者不能互换。WM 的 0-back/2-back 已在行为端分层，但脑指标仍由完整 WM run 估计；尚未检验 block 特异脑动力学、RL run、独立 cohort、去趋势、运动或生理混杂回归、GSR、皮层下结构或其他 null 构造。
- WM 与既有静息态的主幅度比较分别使用 304 和 900 个拟合时间点，只比较 raw Xi，因此均值差仍包含有效样本长度差异。独立的 12 窗口长度匹配分析只检验跨被试方差；WM 的 `sub-103515` 具有极端早期 PC1 瞬变，普通方差比对其高度敏感。
- HCP 的全体被试 Xi 结果使用 20 个 null，p 值分辨率有限，且未校正跨被试、跨模块集合和 greedy 选择造成的多重比较。
- 三项领域认知的全组合扫描在同一 29 人中完成搜索和效应估计。虽然晶体认知、记忆和加工速度各自的最强候选均覆盖 29/29 人，并同时达到原始双侧和逐项置换 $p<0.05$，但每项认知 2,872 个特征内均没有 BH $q<0.05$ 或 maxT $p<0.05$ 的结果，固定拆分也未显著确认。它们只用于提出候选；确认性检验需要预先固定少数假设、保持 HCP 家系交换结构并使用独立数据。
- 贪婪 atom 用于描述候选协同结构；它依赖分解顺序与候选空间，不是 exhaustive 的唯一高阶分解。

<a id="discussion-artifacts"></a>

### 3.2 结果与图形产物

| 实验 | 关键图与结果 |
|---|---|
| Schaefer100 DMF 临界复现 | `fig/dmf_schaefer100/dmf_schaefer100_summary_full.{png,svg,pdf}`、`fig/dmf_schaefer100/dmf_83_vs_100_comparison.{png,svg,pdf}`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_appendix_raw.{png,svg,pdf}`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_kuramoto_shape.{png,svg,pdf}`、`results/dmf_schaefer100/preparation_summary.json`、`results/dmf_schaefer100/group_mean_native.npz`、`results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz`、`results/dmf_schaefer100/full/main_confirmation.npz`、`results/dmf_schaefer100/full/observational_wms.npz`、`results/dmf_schaefer100/full/critical_topology.npz`、`results/dmf_schaefer100/full/critical_yeo7.npz`、`results/dmf_schaefer100/full/detdeg_appendix_summary.json` |
| HCP500 Yeo7-PCA Xi/null | `results/hcp_schaefer500_yeo7_pc1_phi_null/summary.json`、`results/hcp_schaefer500_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP500 Yeo7 模块分解 | `results/hcp_schaefer500_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png` |
| HCP1000 Yeo7-PCA Xi/null | `results/hcp_schaefer1000_yeo7_ridge_selection/summary.json`、`results/hcp_schaefer1000_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP1000 Yeo7 模块分解 | `results/hcp_schaefer1000_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer1000_yeo7_module_phi_decomposition/top_core_consistency.png` |
| HCP500 WM_LR Xi 与协同核 | `results/hcp_schaefer500_wm_yeo7_phi/summary.json`、`results/hcp_schaefer500_wm_yeo7_phi/report.md`、`results/hcp_schaefer500_wm_yeo7_phi/wm_rest_phi_comparison.{png,svg,pdf}`、`results/hcp_schaefer500_wm_yeo7_phi/wm_core_distribution.{png,svg,pdf}` |
| HCP500 静息态与七任务 raw Xi | `results/hcp_schaefer500_all_tasks_phi/summary.json`、`results/hcp_schaefer500_all_tasks_phi/report.md`、`results/hcp_schaefer500_all_tasks_phi/rest_all_tasks_raw_phi.{png,svg,pdf}` |
| HCP500 REST–七任务长度匹配方差 | `results/hcp_schaefer500_length_matched_variance/summary.json`、`results/hcp_schaefer500_length_matched_variance/report.md`、`results/hcp_schaefer500_length_matched_variance/experiment_contract.json`、`results/hcp_schaefer500_length_matched_variance/rest_window_phi.npz`、`results/hcp_schaefer500_length_matched_variance/length_matched_variance.{png,svg,pdf}` |
| HCP500 REST–七任务 $p$–$\alpha$ 鲁棒性 | `results/hcp_schaefer500_phi_hyperparameter_robustness/summary.json`、`results/hcp_schaefer500_phi_hyperparameter_robustness/report.md`、`results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_robustness_overview.{png,svg,pdf}`、`results/hcp_schaefer500_phi_hyperparameter_robustness/hyperparameter_task_margins.{png,svg,pdf}` |
| HCP500 REST–七任务预测误差诊断 | `results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_summary.json`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_report.md`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_overview.{png,svg,pdf}`、`results/hcp_schaefer500_phi_hyperparameter_robustness/prediction_error_by_condition.{png,svg,pdf}` |
| HCP500 REST 与七任务特异脑区分布 | `results/hcp_schaefer500_task_specific_regions/summary.json`、`results/hcp_schaefer500_task_specific_regions/report.md`、`results/hcp_schaefer500_task_specific_regions/task_evoked_region_maps.npz`、`results/hcp_schaefer500_task_specific_regions/task_evoked_region_profiles.{png,svg,pdf}`、`results/hcp_schaefer500_task_specific_regions/rest_all_tasks_variance_profiles.{png,svg,pdf}`、`results/hcp_schaefer500_task_specific_regions/task_map_discriminability.{png,svg,pdf}` |
| HCP500 任务诱发 PCA–$\Xi$ 网络与层级分解 | `results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/summary.json`、`results/hcp_schaefer500_task_evoked_xi_tuning/full/k1_p3_a1/arrays.npz`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/report.md`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/task_evoked_xi_main_combined.{png,svg,pdf}`、`results/hcp_schaefer500_task_evoked_xi_tuning/final/parameter_tuning_comparison.{png,svg,pdf}` |
| HCP1000 任务诱发 PCA–$\Xi$、一般认知与任务表现 | `results/hcp_schaefer1000_task_evoked_xi_replication/summary.json`、`results/hcp_schaefer1000_task_evoked_xi_replication/arrays.npz`、`results/hcp_schaefer1000_task_evoked_xi_replication/final/task_evoked_xi_main_combined.{png,svg,pdf}`、`results/hcp_language_story_math_candidates_schaefer1000_replication/summary.json`、`results/hcp_wm_back_condition_correlations/summary.json` |
| HCP500 领域认知全组合定向 Greedy 扫描 | `results/hcp_cognition_exhaustive_targeted_greedy/experiment_contract.md`、`results/hcp_cognition_exhaustive_targeted_greedy/summary.json`、`results/hcp_cognition_exhaustive_targeted_greedy/all_associations.jsonl`、`results/hcp_cognition_exhaustive_targeted_greedy/exhaustive_top_candidates_scatter.{png,svg,pdf}`、`results/hcp_cognition_exhaustive_targeted_greedy/exhaustive_search_landscape.{png,svg,pdf}`、`results/hcp_cognition_exhaustive_targeted_greedy/extended_interpretable_candidates.json`、`results/hcp_cognition_exhaustive_targeted_greedy/extended_task_aligned_correlations.{png,svg,pdf}` |

<a id="appendix-a"></a>

## 附录 A：Kuramoto 振子数与 whole-state $\Xi$ 曲线形状

为避免把方程差异误读成振子数效应，这里重新使用同一个经典全局耦合 Kuramoto 方程，只改变振子数：

$$
\dot{\theta}_i
=\omega_i+\frac{K}{N}\sum_{j=1}^{N}\sin(\theta_j-\theta_i).
$$

除振子数外，两组实验使用同一协议：频率 $\omega_i$ 从零均值 Gaussian 抽样，随后对每个 seed 去均值并重缩放到 `sigma=1`；`N=2` 时这个协议退化为一对符号相反、标准差为 1 的频率。source 是全部振子的当前相位特征，target 是全部振子的未来相位状态，而不是整体速度；两组都直接计算与 Part2 大脑动力学 $\Xi$ 相同的源侧 whole-minus-sum 结构：

$$
\Xi
=
EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})
-\sum_{i=1}^{N} EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})
.
$$

其中 $\mathbf{s}_t^i=(\cos\theta_i(t),\sin\theta_i(t))$ 是第 $i$ 个振子的相位特征，$\mathbf{y}_{t+\tau}=\{(\cos\theta_i(t+\tau),\sin\theta_i(t+\tau))\}_{i=1}^{N}$ 是系统整体未来相位状态。$\Xi$ 由该定义保证非负，因此无需引入人工非负截断。

下文首先以 `N=64` Oracle 结果解释临界峰的机制；振子数对照作为系统规模边界证据，统一放在附录末尾。

<a id="appendix-a-1"></a>

### A.1 临界峰的 EI 与 effectiveness 机制

为了检查这个峰值来自哪一项，进一步把 `N=64` Oracle whole-state 结果分解为联合 EI 与单独 EI 之和：

![Large-N Kuramoto EI decomposition](../../fig/classic_network_dynamics_benchmark/large_kuramoto_n64_ei_decomposition.png)

分解结果显示，$EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})$ 和 $\sum_i EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})$ 都随 $K$ 增大而整体下降。这不是反常现象，因为这里的 EI 衡量的是最大熵相位干预下，当前相位状态有多少可区分信息保留到未来 whole-state target 中。`K=0` 时，每个振子近似独立转动，当前相位到未来相位接近一一映射，所以联合 EI 和单独 EI 之和都很高，并且二者几乎相等，$\Xi\approx0$。

随着 $K$ 增大，同步吸引会压缩相位差自由度，许多不同初始相位会被映射到更相似的未来状态，因此总的可区分信息下降。临界前沿附近，单个振子对未来全系统状态的解释力下降得更快，而联合状态仍保留对集体相位关系的解释力，所以两项差值扩大，$\Xi$ 在 `K≈1.7` 达峰。到强同步区后，系统接近低维同步流形，联合 EI 本身也明显降低，差值随之回落。换言之，临界峰不是因为总 EI 最大，而是因为整体相对于部分之和的不可分解优势最大。

同一组 `N=64` Oracle 结果还可以按 effectiveness 的 determinism/degeneracy 口径拆开。这里固定参考熵 $H_0$ 为本 sweep 中最大的 Gaussian target entropy，并定义

$$
Det(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}\mid \mathcal{S}),\qquad
Deg(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}).
$$

其中 $\mathcal{S}$ 可以是全部振子的联合 source，也可以是某个单振子 source；$\mathbf{Y}$ 是 whole-state future target。为避免将四个高度相关的曲线拆散，左图把 whole-source determinism 与 degeneracy 合并到同一**线性**轴，从而突出 determinism 的低谷；右图在单一对数轴上并列 singleton-sum 的两项，保留其跨数量级的共同膨胀与接近。两图的同一条竖虚线标出 whole-source determinism 的最小点，便于把这两个尺度上的变化对齐。

![Large-N Kuramoto determinism and degeneracy decomposition](../../fig/classic_network_dynamics_benchmark/n64_detdeg/large_kuramoto_oracle_nsource_whole_state_phi_sweep_determinism_degeneracy.png)

这个分解补足了临界峰的解释。whole-state determinism 从 `K=0` 的约 `1110.05` bits 下降，在 `K=2.0` 附近降到约 `475.95` bits，随后强同步区又回升到 `K=4.0` 的约 `1078.10` bits；whole-state degeneracy 则从近零单调升高到 `K=4.0` 的约 `1044.37` bits。也就是说，强耦合同步并不是简单地让整体映射“更确定”；它同时把许多微观相位状态折叠到相似的未来同步状态，导致 degeneracy 急剧增加。EI 是二者的差，因此强同步区即便 determinism 回升，也会被更大的 degeneracy 抵消。

右图显示了为什么 $\Xi$ 在临界附近最大。单振子口径的 degeneracy 被对每个 source 重复计算，随 $K$ 增大从 `K=1.0` 的约 `696.91` bits 快速升到 `K=4.0` 的约 `66839.89` bits；singleton-sum determinism 也在强同步区急剧放大，到 `K=4.0` 约 `66860.03` bits。两者都变大且彼此接近，说明单个振子在高同步区会获得大量共享的、重复的 whole-state 预测信息，但这些信息主要是同一个同步流形的冗余读出。临界附近则不同：联合状态仍能保留相位关系和集体模式，而单振子解释已经开始失效，所以 whole-minus-sum 差值在 `K≈1.7` 达到约 `279.63` bits。

#### A.1.1 Schaefer100 DMF 的 determinism/degeneracy 附录验证

为检验 Schaefer100 DMF 的 $\Xi$ 峰是否来自与 Kuramoto 相似的 effectiveness 机制，这里复用正文全扫描的 8 个 seed、$G=0$–3、独立 $U(0.30,0.70)^{200}$ 全 E/I source、300 步 whole-state target 和 Gaussian log-determinant EI，不重新运行动力学。令 $N_s=200$ 为 scalar source 数，并固定 $H_0$ 为整个 DMF sweep 中最大的 target entropy，本实验得到 $H_0=370.346$ bits。为使 $EI=Det-Deg$ 在数值上严格闭合，实际计算采用

$$
\begin{aligned}
Deg_{\mathrm{whole}}(G)&=H_0-H(\mathbf{Y}_G),\\
Det_{\mathrm{whole}}(G)&=EI_{\mathrm{whole}}(G)+Deg_{\mathrm{whole}}(G),\\
Deg_{\Sigma}(G)&=N_s\,Deg_{\mathrm{whole}}(G),\\
Det_{\Sigma}(G)&=\sum_{i=1}^{N_s}EI_i(G)+Deg_{\Sigma}(G).
\end{aligned}
$$

这些式子与 $Det(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}\mid\mathcal{S})$ 及 $Deg(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y})$ 的定义代数等价。$Deg_{\Sigma}$ 对同一个 whole-state target entropy 重复计算 $N_s$ 次，因此其数万 bits 量级是 singleton 求和口径的结果，不能理解为系统额外产生了同等规模的独立简并性。

![Schaefer100 DMF determinism and degeneracy](../../fig/dmf_schaefer100/dmf_schaefer100_detdeg_appendix_raw.png)

*图 A1｜Schaefer100 DMF 的固定参考熵分解。A–D 分别为 whole-source determinism、whole-source degeneracy、singleton-sum determinism 和 singleton-sum degeneracy。曲线及阴影为 8 个 seed 的均值与 SEM；紫色虚线标出 $\Xi$ 峰值 $G=1.3$，黑色点线标出平均发放率最大变化点 $G=1.5$。四个面板使用各自的原始 bits 纵轴。*

四个分量都没有复现 Kuramoto 强耦合端的增长。DMF whole determinism 在 $G=1.2$ 达到 368.363 bits，whole degeneracy 在 $G=1.3$ 达到 182.055 bits；singleton-sum determinism 和 degeneracy 也都在 $G=1.3$ 达到最大值，分别为 36561.840 和 36411.045 bits。此后四项共同下降，到 $G=3.0$ 分别只剩 49.013、0.266、95.743 和 53.298 bits。从发放率转折点 $G=1.5$ 到 $G=3.0$，四项均在 8/8 个 seed 中下降。因此当前 DMF 的 $\Xi$ 回落发生在整体和 singleton 可预测结构同时塌缩的背景下，而不是 Kuramoto 中 degeneracy 与 singleton-sum 分量在强同步区继续膨胀的机制。

为了只比较形状，下一图先在每个模型、每个 seed、每个分量内部做 $[0,1]$ 范围归一化，再将横轴分别写成 $G/1.5$ 与 $K/K_c$。竖线 1 表示各自转变参照；比较范围固定为共同覆盖的相对耦合 0–2。这个归一化移除了绝对 bits、source 数和耦合单位差异，但没有消除模型方程、状态空间、预测时间窗及估计器差异。

![DMF and Kuramoto determinism-degeneracy shape comparison](../../fig/dmf_schaefer100/dmf_schaefer100_detdeg_kuramoto_shape.png)

*图 A2｜Schaefer100 DMF 与 $N=64$ Kuramoto determinism/degeneracy 曲线形状对照。DMF 使用 8 个 seed 的 Gaussian EI；Kuramoto 使用 2 个 seed 的 Oracle transport-map EI。每条曲线先在各自 seed 和分量内做范围归一化，再显示均值与 SEM。该图是探索性的跨模型形状比较，不是单因素受控因果对照。*

形状对照同样不支持“二者变化规律相似”。在相对耦合 0–2 上，DMF 与 Kuramoto 的描述性 Pearson 形状相关分别为 whole determinism $r=-0.008$、whole degeneracy $r=-0.785$、singleton-sum determinism $r=-0.793$、singleton-sum degeneracy $r=-0.785$。更直接地，从相对耦合 1 增至 2，DMF 四项在 8/8 个 seed 中全部下降，而 Kuramoto 四项在 2/2 个 seed 中全部上升。相关系数受插值网格与归一化方式影响，不作为显著性检验；稳定结论仅是高耦合分支方向相反。

因此，两套实验的共同点限于 whole-minus-sum $\Xi$ 都可在动力学转变附近形成峰；分解后的机制并不相同。Kuramoto 的峰伴随强同步端 degeneracy 和 singleton 重复读出的持续增长，当前 Schaefer100 DMF 则表现为四个分量在转折后共同衰减。这个阴性结果说明不能仅凭 $\Xi$ 峰形就把 DMF 的高耦合状态解释成 Kuramoto 式同步压缩。

<a id="appendix-a-2"></a>

### A.2 时间窗、相变前检测与系统规模边界

#### A.2.1 时间窗鲁棒性：避免强同步后，短窗不复现临界内部峰

基准 whole-state 曲线的 `tau=4` 结果保留为主对照。为检验其峰值是否只是高 $K$ 同步饱和造成的，新增一个严格配对的 multi-horizon Oracle sweep：对每个 seed，频率向量、均匀相位 intervention states 和 natural readout states 都固定并复用于全部 $(K,\tau)$ 条件；只改变统一的预测时间窗 $\tau\in\{0.5,0.75,1,1.5,2,4\}$，而不允许 $\tau$ 随 $K$ 自适应变化。所有条件仍使用 `N=64`、3 个 seeds、whole-state future phase target 与同一 N-source transport-map estimator。

![Paired large-N Kuramoto horizon sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep.png)

图 A 以未来 target 的 raw global order 的 $99\%$ 分位数 $R_{0.99}$ 审计强同步。预先设定 guard 为：对所有 $K$ 都要求 $R_{0.99}<0.8$。`tau=0.5` 在最强耦合 `K=4` 仍只有 $R_{0.99}=0.583$，完全通过；`tau=0.75` 为 $0.746$，也通过（仅约 $0.37\%$ target samples 的 $R\ge0.8$）。从 `tau=1` 起该 guard 开始失效：`tau=1` 仅 `K=4` 失败，`tau=1.5` 在 `K=3.2,4` 失败，`tau=2` 在 `K\ge2.6` 失败，而 `tau=4` 在 `K\ge2.2` 失败。

关键结果在图 B：**通过 guard 的两个短窗并没有给出与原图相同的临界内部峰。** `tau=0.5` 的 $\Xi$ 从 `K=0` 的约 $0$ bits 持续升至 `K=4` 的 $229.69$ bits；`tau=0.75` 同样在 `K=4` 最大，为 $262.20$ bits。因此，在目标尚未进入强同步区的有限短时间内，耦合增强主要表现为 whole-state 联合可预测性的持续增强，而非在 $K_c\approx1.596$ 附近形成回落前的峰。随着时间窗变长，最大值才逐步向低 $K$ 移动：`tau=1` 的峰在 `K=4`（$279.54$ bits），`tau=1.5` 在 `K=3.2`（$281.00$ bits），`tau=2` 在 `K=2.6`（$280.27$ bits），配对的 `tau=4` 在 `K=1.8`（$278.92$ bits），与原 `tau=4` 图中 `K\approx1.7` 的峰一致到扫描分辨率。

因此，原始临界前沿峰的正确表述应收紧为：它是**中等有限观测时间（此处约 $\tau=4$）下**，在高 $K$ 同步吸引已压缩 whole-state 信息后出现的 whole-minus-sum 优势峰；它不是对所有预测时间窗都成立的、时间尺度无关的临界指标。短窗结果同时排除了一个较弱的替代解释：该峰并非仅由高 $K$ target 已完全同步所产生，因为在明确未强同步的 `tau=0.5,0.75` 条件下，曲线反而没有内部峰。

#### A.2.2 更长时间窗：峰位穿过而非收敛于理论 $K_c$

为直接检验“继续增大 $\tau$ 后，峰是否会停在临界相变点”的假设，保持同一配对 protocol、`N=64`、3 个 seeds 和 full-sample TM estimator，将时间窗扩展为 $\tau\in\{4,6,8,10,12\}$。扫描在转变区加密到 $K=0.8,0.9,\ldots,2.6$，并保留 $K=0,0.4,3.2,4.0$ 锚点，以区分内部峰和扫描端点峰。

![Long-horizon paired large-N Kuramoto sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_long_horizon_refined.png)

结果不支持单调收敛后固定在理论 $K_c=1.596$ 的解释。随着 $\tau$ 从 4 增至 12，$\Xi$ 的内部峰位依次为 $K_{\rm peak}=1.8,1.6,1.5,1.4,1.3$（峰值分别为 $278.92,274.52,272.83,271.65,269.61$ bits）。因此，`tau=6` 的 $K_{\rm peak}=1.6$ 只是在当前 $0.1$ 网格上恰好贴近 $K_c$；继续增加时间窗后，峰越过 $K_c$ 并持续移向更低的 $K$，而非在 $K_c$ 停留。所有这些峰都是加密区内部点，且其 $R_{0.99}$ 仅为 $0.644,0.561,0.523,0.492,0.492$，strong fraction 均为零；故该左移不是由峰落在高 $K$ 强同步 guard 失效区造成的。

更稳妥的结论是：$K_{\rm peak}(\tau)$ 是有限时间有效信息的时间尺度依赖 crossover，可能在某一中等时间窗掠过临界区，但不能把 $\tau\to\infty$ 的峰位等同于静态 Kuramoto 临界点。长窗极限还可能受相位混合和吸引子压缩控制；若要定义渐近临界指标，需要另行研究固定有限尺寸下的长时间衰减、再做 $N\to\infty$ 的有限尺寸标度，而不能从当前峰位外推。

#### A.2.3 相变前检测：共同早期弛豫窗中的 $\Xi(\tau)$ 谱

前述长窗峰位不能直接用作预警器。为检验能否在 future target 尚未同步时识别系统的**最终动力学区间**，对全部 $K\in[0,4]$ 保留同一短时间窗，而不是为高 $K$ 自适应延长或截短 horizon。已有的 `tau=0.5,0.75` 结果与新增的 $\tau\in\{0.1,0.2,0.3,0.4,0.6\}$ 配对合并，得到共同谱 $\tau\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.75\}$。所有 $(K,\tau)$ 条件都满足 $R_{0.99}<0.8$；即使在 $K=4$、$\tau=0.75$，$R_{0.99}\approx0.75$，因此该谱只观测初始相位分布向同步吸引子弛豫的早期，而没有把已同步 target 当作特征。

![Pre-transition Kuramoto Xi-tau phase detection](../../fig/classic_network_dynamics_benchmark/large_kuramoto_pretransition_phi_tau_phase_detection.png)

图 B 显示：超临界 $K>K_c$ 条件在整个共同早期窗内已有更陡、更高的 whole-state $\Xi(\tau)$ 谱，而此时图 A 证明其 target 尚未发生强同步。以已知的 $K_c=1.596$ 作为模拟中的超临界参考标签，只输入 7 个早期 $\Xi(\tau)$ 值，使用 leave-one-$K$-out（完整留出该 $K$ 的 3 个 seed）逻辑回归，得到超临界识别 AUROC 为 $0.983$。将每一条谱除以自身最大值、仅保留形状后，AUROC 仍为 $0.972$；因此区分力不只是 $\Xi$ 的整体幅度，时间尺度上的增长形状也携带信息。图 C 展示了留出 $K$ 后的预测概率。

##### A.2.3.1 识别算法与 AUROC 的计算

这个实验不是在单条真实轨迹上拟合未来标签，而是一个受控的 Oracle 可辨识性检验。数据单位是一个固定耦合和随机 seed 的组合 $(K,s)$。共有 17 个 $K$ 值、3 个 seed，因此有 $17\times3=51$ 个样本。对每个样本，先从同一 seed 的均匀初始相位 intervention support 出发，分别积分到 7 个早期 horizon，并计算 whole-state N-source 指标。输入特征向量为

$$
\mathbf{x}_{K,s}=
\left[
\Xi_{K,s}(0.1),
\Xi_{K,s}(0.2),
\Xi_{K,s}(0.3),
\Xi_{K,s}(0.4),
\Xi_{K,s}(0.5),
\Xi_{K,s}(0.6),
\Xi_{K,s}(0.75)
\right].
$$

这里的每个 $\Xi_{K,s}(\tau)$ 都是同一 whole-state 目标和同一 N-source transport-map estimator 下的

$$
\Xi=EI_{\mathrm{do}}(\mathbf{S};\mathbf{Y}_{\tau})
-\sum_{i=1}^{64}EI_{\mathrm{do}}(\mathbf{s}_i;\mathbf{Y}_{\tau}),
$$

其中 $\mathbf{S}$ 是 64 个振子的联合当前相位特征，$\mathbf{s}_i$ 是第 $i$ 个振子的二维相位特征，$\mathbf{Y}_{\tau}$ 是 $\tau$ 后的 128 维 whole-state phase target。保留一个特征前，先审计自然 readout target 的 $R_{0.99}$；只有本实验中全部 $51$ 个样本都满足 $R_{0.99}<0.8$ 的共同 horizon 才进入上式。故模型没有看到已经强同步的 future target。

二分类标签不由 $\Xi$、早期 $R$ 或长时间 $R$ 阈值产生，而是由生成模型中已知的理论边界独立给出：

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

这里应把次临界状态称为**去相干／次临界动力学**，而不是默认称为“混沌相”：经典全局耦合 Kuramoto 的 $K<K_c$ 解一般可以是非同步的准周期运动，但不由本实验自动证明为严格混沌。有限 $N$ 下同步是 crossover，长时间 raw order 因而保留为连续审计量而未被任意阈值二分。当前结果的含义是：在这个已知方程、已知 $K_c$ 的 Oracle setting 中，早期 $\Xi(\tau)$ 谱可以预报未来进入超临界区；要转化为真实观测数据的预警器，仍需在未知参数、噪声、部分观测和独立时变轨迹上重新校准。

#### A.2.4 系统规模边界：只有大系统提供临界峰参照

![Kuramoto oscillator-count appendix](../../fig/part1_kuramoto_size_phi_eid_appendix.png)

该对照只展示 Oracle $\Xi$ 与 corrected order，不再混入学习模型读出。**小 $N=2$ classic Kuramoto。** 在相同 whole-state 口径下，Oracle $\Xi$ 没有形成清楚的内部临界峰；它在当前扫描范围内主要随强耦合增强，到 `K=4.0` 约为 `0.96` bits。`N=2` 的 corrected order 也不是热力学意义下的相变曲线，而是有限二振子锁相读数。

**大 $N=64$ classic Kuramoto。** 在完全相同的方程、source partition、whole-state target 和 $\Xi$ 公式下，corrected global order 从低 $K$ 的近零状态进入高 $K$ 同步饱和区。理论临界耦合为 $K_c\approx1.596$；有限时间读出下最大斜率出现在 `K=2.2`。对应 Oracle N-source $\Xi$ 从 `K=0` 的 `0` bits 升高，在 `K=1.7` 达峰，约 `279.63` bits；随后进入强同步区后明显回落，`K=4.0` 约 `13.58` bits。

这个边界对照说明，在方程形式、source/target 和 EI 分解公式都固定后，是否出现临界内部峰主要取决于系统规模。`N=2` 没有经典 Kuramoto 的热力学同步相变，所以不能期待它给出与大系统相同的 $\Xi$ 峰；`N=64` 才提供清晰的 order-parameter 转变参照。

因此，Kuramoto 临界相变实验的核心证据链是三步：order parameter 给出同步转变区，whole-state $\Xi$ 在转变前沿形成峰值，determinism/degeneracy 分解说明该峰来自“联合相位构型仍可区分、单振子读出快速冗余化”的差异，而不是来自总 EI、determinism 或 degeneracy 任一单项的简单最大化。

<a id="appendix-b"></a>

## 附录 B：Schaefer100 DMF 动力学方程

本附录给出第 1 节实际积分的 Schaefer100 dynamic mean-field（DMF）方程。令 $N=100$，兴奋性和抑制性 NMDA 门控状态分别为 $\mathbf{s}_E(t),\mathbf{s}_I(t)\in\mathbb{R}^{N}$；$\overline{\mathbf{C}}\in\mathbb{R}^{N\times N}$ 为式（1）的群体平均对称结构连接矩阵，$\mathbf{j}^{\mathrm{FIC}}(G)\in\mathbb{R}^{N}$ 为该耦合值对应的 JFIC 向量。除矩阵乘法外，下式中的向量乘积、除法和函数均逐元素执行，$\mathbf{1}$ 为全 1 向量。

### B.1 局部电流与输入输出函数

兴奋性、抑制性群体的输入电流为

$$
\begin{aligned}
\mathbf{I}_E
&=w_E I_0\mathbf{1}
+w_+J_{\mathrm{NMDA}}\mathbf{s}_E
+GJ_{\mathrm{NMDA}}\overline{\mathbf{C}}\mathbf{s}_E
-\mathbf{j}^{\mathrm{FIC}}(G)\odot\mathbf{s}_I,\\
\mathbf{I}_I
&=w_I I_0\mathbf{1}
+J_{\mathrm{NMDA}}\mathbf{s}_E
-\mathbf{s}_I.
\end{aligned}
$$

因此，主实验使用的是**直接**长程兴奋输入 $\overline{\mathbf{C}}\mathbf{s}_E$，而不是扩散型 $\overline{\mathbf{C}}\mathbf{s}_E-\operatorname{diag}(\overline{\mathbf{C}}\mathbf{1})\mathbf{s}_E$；连接矩阵也没有做行归一化。兴奋性与抑制性放电率写为

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
\mathbf{s}_E^0,\mathbf{s}_I^0\overset{\mathrm{ind}}{\sim}U(0.30,0.70)^{100}
$$

开始，随后按上式积分 300 步；完整的 $[\mathbf{s}_E^{300},\mathbf{s}_I^{300}]\in\mathbb{R}^{200}$ 是 target。主确认设置 `state_boundary=none`，因此更新后**不**将任何状态裁剪回 $[0,1]$。初始干预支持规定最大熵外生初态，`state_boundary` 则规定后续随机动力学不施加硬边界，两者不是同一设定。

<a id="appendix-c"></a>

## 附录 C：83 ROI 与 100 ROI 受控比较

图 2 使用有效耦合 $G\rho(\mathbf{C})$ 和每个 scalar source 的 $\Xi$，以降低连接尺度与系统维度差异的直接影响。结果汇总如下：

| 指标 | 83 ROI | 100 ROI |
|---|---:|---:|
| seed 峰位的平均有效耦合 | 约 0.614 | 0.911 |
| 每 source 的峰值 $\Xi$ | 约 0.074 bits | 0.131 bits |
| 峰值邻近窗跨 ROI 占比 | 68.67% | 88.00% |

这一比较只支持跨连接组的定性复现。新矩阵是对称、稠密的群体平均 SC；旧矩阵是稀疏、有向的 F-TRACT `count` 代理。两个系统分别有 200 和 166 个 scalar source，且谱半径、强度分布和模块结构不同。新数据的上游预处理及单位也未知。因此，图 2 中的峰位、单位 source 强度和层级比例差异不能单独归因于脑区数量、方向性、密度或某一种拓扑性质。

<a id="appendix-d"></a>

## 附录 D：Schaefer100 可复现文件

| 类型 | 路径 |
|---|---|
| 数据审计 | `results/dmf_schaefer100/preparation_summary.json` |
| 群体均值与标签 | `results/dmf_schaefer100/group_mean_native.npz`、`results/dmf_schaefer100/schaefer100_labels.txt` |
| 主扫描 | `results/dmf_schaefer100/full/main_confirmation.npz` |
| WMS | `results/dmf_schaefer100/full/observational_wms.npz` |
| 拓扑/层级分解 | `results/dmf_schaefer100/full/critical_topology.npz`、`results/dmf_schaefer100/full/topology_summary.json` |
| Yeo-7 分解 | `results/dmf_schaefer100/full/critical_yeo7.npz`、`results/dmf_schaefer100/full/yeo7_summary.json` |
| 汇总图 | `fig/dmf_schaefer100/dmf_schaefer100_summary_full.{png,svg,pdf}` |
| Determinism/degeneracy 附录验证 | `results/dmf_schaefer100/full/detdeg_appendix_summary.json`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_appendix_raw.{png,svg,pdf}`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_kuramoto_shape.{png,svg,pdf}` |
| 83/100 对照图 | `fig/dmf_schaefer100/dmf_83_vs_100_comparison.{png,svg,pdf}` |
| 实验契约与进度 | `docs/log/dmf_schaefer100_experiment_contract.md`、`docs/log/dmf_schaefer100_progress.json` |

完整流程可复现为：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_dmf_schaefer100_pipeline.py --mode full

python scripts/plot_dmf_schaefer100_detdeg_appendix.py
```

单线程 BLAS 用于避免小型线性代数在多线程调度下显著变慢，不改变统计定义或随机种子。

<a id="appendix-e"></a>
<a id="hcp-hyperparameter-robustness"></a>

## 附录 E：拟合模型参数鲁棒性

![HCP500 超参数鲁棒性与预测误差诊断](../../fig/brain_hcp500_robustness_plate.png)

*图 E1. HCP500 的超参数鲁棒性与预测诊断。A：25 点 $p$–$\alpha$ 网格上，七项 REST-minus-task 均值差的最小值（a）及 Holm 校正后显著的对比数（b）。B：七个任务各自的 REST-minus-task raw $\Xi$ 边际。C：八状态等权平均的留出 delta-NRMSE（a）、test-minus-train 泛化间隙（b），以及 $p=8$ 时各状态随 $\alpha$ 的留出误差（c）。D：八个状态各自的留出 delta-NRMSE。A–B 回答 Xi 排序是否依赖超参数；C–D 检验弱正则反转是否伴随预测失真。*

<a id="appendix-e-1"></a>

### E.1 REST–任务差异的 $p$–$\alpha$ 鲁棒性

原始 $p=8,\alpha=10$ 来自静息态模型选择，可能给 REST 带来选择优势。为检验这一点，这里预先固定 $p\in\{1,2,3,5,8\}$ 与 $\alpha\in\{0.1,1,10,100,1000\}$ 的 25 点网格。每个网格点把同一组超参数同时用于 REST 和七种任务态；29 名共同被试的每个状态仍独立重新拟合 PCA、缩放、Ridge 系数、截距和残差协方差。所有序列使用各自前 75%，只计算 raw $\Xi$。每个网格点分别完成七项双侧 Monte Carlo paired sign-flip 检验（200,000 次），再在七任务内作 Holm 校正。

图 E1A-a 给出每个网格点七项 REST-minus-task 群体均值差中的最小值；正值表示 REST 均值高于所有任务，负值表示至少一个任务反超。图 E1A-b 给出七项对比中 Holm 校正后显著的数量。REST 只在 **15/25** 个网格点为八条件群体均值最高状态，只有 **7/25** 个网格点满足“REST 高于全部任务且七项均显著”。这 7 个点为 $(p,\alpha)=(2,0.1),(2,1),(2,10),(3,1),(3,10),(5,10),(8,10)$。原始 $(8,10)$ 的 232 个“被试 $\times$ 状态”数值与附录 I.1 逐点完全一致，排除了结果变化来自实现路径差异。七个任务各自的边际见图 E1B。

稳定区主要位于中等历史阶数和中等正则强度：$\alpha=10$ 时，$p=2,3,5,8$ 的七项对比均显著；但 $p=1$ 的所有 $\alpha$ 均为 0/7 显著。在过强正则 $\alpha=1000$ 下，五个 $p$ 均不再满足 REST 群体均值最高，且最多只有 1/7 项显著。弱正则与高阶历史的组合还会反转方向：$p=8,\alpha=0.1$ 时，REST 相对 EMOTION 和 RELATIONAL 分别低 2.789932 和 2.635 bits，两项任务高于 REST 的 Holm 校正 $p$ 分别为 0.003540 和 0.003150；其余任务的方向或显著性也不统一。

因此，最窄且可靠的结论是：**REST 高于七任务的 raw $\Xi$ 在包含原配置的中等正则参数带内可复现，但对完整 25 点 $p$–$\alpha$ 网格并不鲁棒。**$p$ 同时改变 source 维数 $7p$，所以应在同一个 $(p,\alpha)$ 内解释 REST–任务差值，不能把不同 $p$ 的绝对 raw Xi 当作同维度量直接比较。上述 Holm 校正只针对每个网格点内部的七项任务对比，并未把 25 个网格点作为发现性假设族再次校正；因此 7/25 是敏感性描述，不是 175 项独立发现。极端超参数是否应被视为合理模型还需要独立、条件平衡的预测验证规则；在完成该步骤之前，不能从本扫描中事后挑选最支持 REST 或任务态的参数作为主结论。

<a id="appendix-e-2"></a>

### E.2 留出预测误差解释弱正则反转

为判断左下角弱正则、高阶历史的反转是否来自拟合失真，这里保持同一 25 点网格和每个“状态 $\times$ 被试”的前 75% 训练段，严格用最后 25% 时间点计算一步留出误差。主指标 delta-NRMSE 先用训练段的每个网络 delta 标准差归一化，再跨网络、时间点和被试汇总；同时报告 test-minus-train 泛化间隙，以及相对“$\mathbf{x}_{t+1}=\mathbf{x}_t$”持久性基线的技能值。技能为正表示优于持久性预测，为负表示模型在留出段反而更差。

图 E1C-a 显示八状态等权平均的留出 delta-NRMSE，图 E1C-b 显示泛化间隙，图 E1C-c 单独展开 $p=8$ 时各状态随 $\alpha$ 的误差；各状态的完整误差网格见图 E1D。结果支持过拟合解释：$p=8,\alpha=0.1$ 的训练误差仅为 0.6538，但留出误差升至 0.9612，泛化间隙达到 0.3074，持久性技能为 $-0.0406$；其平均 Ridge 系数 Frobenius 范数为 7.735。改用 $\alpha=10$ 后，留出误差降至 0.8726、泛化间隙降至 0.1685、持久性技能升至 0.2049，系数范数缩至 2.935。$p=8$ 的整体最低留出误差出现在 $\alpha=100$，为 0.8590，说明 $\alpha=10$ 位于高泛化区，但不是该阶数下的唯一或总体最优正则值。

发生显著 raw $\Xi$ 反转的两个任务同时给出最直接的失败证据。在 $p=8,\alpha=0.1$ 下，EMOTION 的训练/留出误差为 0.5487/1.0628，泛化间隙 0.5141，持久性技能 $-0.3454$；RELATIONAL 为 0.6176/1.0062，泛化间隙 0.3887，持久性技能 $-0.1106$。也就是说，弱正则模型在训练段把这两个较短任务拟合得异常好，但在未参与拟合的时间段已经不具备可靠预测能力。固定 $p=8$ 后，在全部 35 个“超参数 $\times$ 任务”对比中，REST-minus-task Xi 边际与 task-minus-REST 留出误差差的 Spearman $\rho=-0.592$（$p=1.79\times10^{-4}$），与泛化间隙差的 $\rho=-0.480$（$p=0.00351$）：任务相对 REST 过拟合越明显，Xi 排序越倾向任务反超。

这种反转的可能机制是：$p=8$ 提供 56 维历史 source，而任务态用于回归的训练样本仅约 124--296 行；$\alpha=0.1$ 允许更大的回归系数，并在训练段压低残差协方差。Gaussian log-det $\Xi$ 同时依赖转移系数和训练残差协方差，因此会把这种训练内的高自由度拟合转化为偏大的 raw Xi。该证据说明预测失真是反转的重要来源，但不是数学上的唯一原因，因为 $\Xi$ 不是预测误差的单调函数，且并非所有预测较差的点都发生显著方向反转。

从共享超参数选择看，$\alpha=10$ 在 40 个“八状态 $\times$ 五个 $p$”组合中的 29 个（72.5%）达到最低留出误差；八状态总体平均下，$p=1,2,3,5$ 的最优 $\alpha$ 都是 10，$p=8$ 则是 100。若在整个 25 点网格中按八状态与29名被试等权的留出误差选择一组共享参数，最优点是 $(p,\alpha)=(5,10)$，delta-NRMSE 为 0.8552；在这个不依据 REST–任务 Xi 方向选出的配置上，REST 相对最接近任务仍高 1.4243 bits，七项对比全部 Holm 显著。因此，对本附录所检验的“各状态分别在自身 retained 时序上拟合 PC1”的原始表征，$(5,10)$ 是更公平的预测型参照。第 2.1 节的主配置 $(k,p,\alpha)=(1,3,1)$ 使用不同的任务诱发 PCA 构造，并经过 8 人筛选、21 人独立确认，二者不应视为同一参数搜索中的冲突最优点。

<a id="appendix-f"></a>
<a id="hcp500"></a>

## 附录 F：HCP500 PCA–Yeo7 $\Xi$–null 分解

<a id="appendix-f-1"></a>
<a id="hcp500-data"></a>

### F.1 数据、降维与动力学表征

数据为 30 名 HCP S1200 被试的 `REST1_LR` Schaefer-500 BOLD 时序，每名被试包含 $1200\times500$ 个时间点与皮层 parcel。500 个 parcel 按 Yeo7 标签分为 Vis、SomMot、DorsAttn、SalVentAttn、Limbic、Cont 和 Default；每个网络仅保留训练段拟合的一维 PCA（PC1），形成 7 维网络状态 $\mathbf{x}_t$。

Xi 实验使用 `sub-100206` 的时间验证选出的八阶 $\Delta$-Ridge，$p=8$、$\alpha=10$。模型以当前与过去七个时刻的网络状态为 source：

$$
\mathbf{h}_t=
\left[\mathbf{x}_t^\top,\mathbf{x}_{t-1}^\top,\ldots,\mathbf{x}_{t-7}^\top\right]^\top
\in\mathbb{R}^{56},
\qquad
\Delta\mathbf{x}_{t+1}=\mathbf{x}_{t+1}-\mathbf{x}_t.
$$

PCA、标准化器和 Ridge 均仅以每名被试的前 900 个时间点拟合；后 300 点不参与本 Xi 计算、参数选择或 null 重拟合。这里的目标是比较固定表征与固定模型下的 observed 和 null，而不是把该单被试模型推广为全体被试的预测最优模型。

<a id="appendix-f-2"></a>
<a id="hcp500-phi"></a>

### F.2 History-source $\Xi$ 与 circular-shift null

本实验的量是 56 维历史 source 到下一时刻 7 维状态的量，而不是 7 维一阶 whole-state $\Xi$：

$$
\Xi_{\mathrm{hist}\to\mathrm{next}}
=EI\!\left(\mathbf{h}_t;\mathbf{x}_{t+1}\right)
-\sum_{j=1}^{56}EI\!\left(h_{t,j};\mathbf{x}_{t+1}\right).
$$

估计使用 Gaussian log-det 口径，而非 TM。这样可在 56 维 history source、30 名被试、重复 null 重拟合和贪婪分解下保持可计算性；代价是结果依赖 Gaussian 近似，不能直接外推到非高斯的精确干预信息量。

null 对 7 条 PC1 时序分别施加独立、非零的 circular shift，并在相同的 $p$、$\alpha$ 与 900 点训练预算下重新拟合模型。它保留每条网络 PC1 的边际取值与自相关结构，但破坏网络间的时间对齐，检验观测到的跨网络结构是否超出网络内时间结构本身。

在 30 被试、每人 20-null 的扩展中，observed $\Xi$ 的均值与中位数为 6.188481 与 6.068454 bits；observed-minus-null-mean 的均值与中位数为 1.984600 与 2.051671 bits，范围为 0.492521–4.096287 bits。30/30 名被试的 observed 都高于各自 null 均值，且未校正经验 $p<0.05$；由于仅有 20 个 null，每个被试的最小 p 值分辨率为 $1/21=0.047619$。

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

![HCP500 Yeo7 Xi-null 与模块协同核汇总](../../fig/brain_hcp500_yeo7_phi_null_summary.png)

*图 F1. HCP500 PCA–Yeo7 Xi–null 分解。A：30 名被试的 observed-minus-null-mean $\Xi$；绿色表示当前 20-null 分辨率下的经验 $p<0.05$。B：每名被试 greedy top-3 中出现的模块核及其原子贡献；空白表示该核未进入对应被试的 top-3。*

全 7 网络核最常出现（20/30），但它在 matched null cohort 中反而更常出现（均值 26.65/30；经验 $p=1$），因此不能将其读为真实数据特异的协同核。缺少 Limbic 的六网络广域核在真实数据中为 17/30，而 matched null cohort 的频率为 $8.65/30$（最大 12/30；经验 $p=1/21=0.047619$）；两个较小的候选核也分别为 9/30 对 $1.40/30$、8/30 对 $2.45/30$，同为该分辨率下的未校正 $p=1/21$。这支持真实静息态中若干非全网络模块核的出现频率和贡献高于该 circular-shift null；但 20 个 null 的 p 值分辨率有限，且统计未校正跨模块集合与 greedy 选择，因此不构成唯一生物学 atom 的确证。

<a id="appendix-g"></a>
<a id="hcp1000"></a>

## 附录 G：HCP1000 PCA–Yeo7 $\Xi$–null 分解

<a id="appendix-g-1"></a>
<a id="hcp1000-data"></a>

### G.1 数据、降维与模型选择

同一 30 名 `REST1_LR` 被试的 `Schaefer1000` 矩阵为 $1200\times1000$。1000 个 parcel 按同一 Yeo7 顺序分为 Vis 162、SomMot 194、DorsAttn 122、SalVentAttn 121、Limbic 60、Cont 129 与 Default 212 个 parcel；每名被试的各网络 PC1 均只以前 900 点拟合并投影完整时序。

在 `sub-100206` 的训练段内以 600/700/800 三个时间验证折，从 $p\in\{1,2,3,5,8\}$ 与既有 Ridge $\alpha$ 网格选择模型。最优冻结配置为五阶 $\Delta$-Ridge，$p=5$、$\alpha=1$，平均 validation skill ratio 为 0.794433；因此 source 是 35 维网络历史，target 为下一时刻 7D 网络状态。后 300 点未参与 PC1、模型或参数选择。

<a id="appendix-g-2"></a>
<a id="hcp1000-phi"></a>

### G.2 History-source $\Xi$ 与 circular-shift null

对每名被试固定上述表征与模型，并以每条 PC1 独立、非零 circular shift 后重拟合同一模型生成 20 个 null。1000-parcel observed $\Xi$ 的均值/中位数为 7.783676/7.734082 bits；observed-minus-null-mean 的均值/中位数为 2.997670/3.032261 bits，范围为 0.814450–6.085586 bits。30/30 名被试的 observed 均高于其 null 均值，未校正经验 p 均小于 0.05（最小分辨率 $1/21=0.047619$）。

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

![HCP1000 Yeo7 Xi-null 与模块协同核汇总](../../fig/brain_hcp1000_yeo7_phi_null_summary.png)

*图 G1. HCP1000 PCA–Yeo7 Xi–null 分解。A：30 名被试的 observed-minus-null-mean $\Xi$；绿色表示当前 20-null 分辨率下的经验 $p<0.05$。B：每名被试 greedy top-3 中出现的模块核及其原子贡献；空白表示该核未进入对应被试的 top-3。*

| 描述性比较 | Schaefer-500 | Schaefer-1000 |
|---|---:|---:|
| observed $\Xi$ 均值（bits） | 6.188481 | 7.783676 |
| observed − null 均值（bits） | 1.984600 | 2.997670 |
| observed 高于 null mean | 30 / 30 | 30 / 30 |
| 全七网络核 top-3 频率 | 20 / 30 | 21 / 30 |
| 缺 Limbic 六网络核 top-3 频率 | 17 / 30 | 12 / 30 |

两种粒度都复现了跨网络时间对齐高于 circular-shift null 的方向性证据，并都将缺 Limbic 的六网络广域核识别为高于 matched-null 频率的候选结构。绝对 bits、最优滞后阶数和 atom 频率受分区粒度、PC1 表征与单被试调参影响；上表仅作描述性对照，不能当作空间粒度的正式统计检验，也不将 greedy 核解释为唯一生物学 atom。

<a id="appendix-h"></a>

## 附录 H：HCP500 WM 协同核及静息态对照

<a id="appendix-h-1"></a>

### H.1 WM 表征、模型与 circular-shift null

任务态分析使用 30 名被试的 `WM_LR` `Schaefer500_taskRetained` 时序，每名被试为 $405\times500$。为保持方法迁移而不在任务态重新择优，模型冻结静息态 Schaefer-500 的八阶 $\Delta$-Ridge（$p=8,\alpha=10$）与 Gaussian log-det 估计器。每名被试的 Yeo7 PC1 仅以前 304 点拟合；source 仍为 56 维网络历史，target 仍为下一时刻 7 维网络状态。每个 observed 配置配对 20 个独立 PC1 circular-shift null，并对每个 null 重新拟合模型。

30 名被试的 observed $\Xi$ 均值/中位数为 4.939700/4.557162 bits；observed-minus-null-mean 的均值/中位数为 1.781650/1.581241 bits。30/30 名被试均高于自身 null mean，22/30 的 observed 高于全部 20 个 null，因此达到当前分辨率下的最小经验 $p=1/21=0.047619$。受试者级 $\Delta\Xi$ 的均值 95% bootstrap CI 为 $[1.344991,2.423814]$ bits，paired sign-flip $p=5.0\times10^{-6}$。这支持 WM 中的跨网络时间对齐高于保留单网络自相关、但破坏网络间对齐的 null。

<a id="appendix-h-2"></a>

### H.2 协同核分布及静息态对照

同一网络的 8 个历史滞后仍绑定为一个不可拆模块；每个 observed 和 null 均完整运行 greedy top-3。WM 中最常见的四个核如下。

| WM 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | WM matched-null 频率均值；经验 p |
|---|---:|---:|---:|
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **26 / 30** | 0.987442 bits | 18.55 / 30；0.047619 |
| 全部 7 个 Yeo 网络 | 14 / 30 | 1.045962 bits | 15.85 / 30；0.857143 |
| SomMot + DorsAttn + SalVentAttn + Cont + Default | 10 / 30 | 0.911514 bits | 1.30 / 30；0.047619 |
| Vis + SomMot + DorsAttn + Cont + Default | 7 / 30 | 0.945526 bits | 4.75 / 30；0.095238 |

![WM Yeo7 协同核分布及静息态频率对照](../../results/hcp_schaefer500_wm_yeo7_phi/wm_core_distribution.png)

*图 H1. WM Yeo7 协同核分布及其 observed、matched-null 与 REST 频率对照。*

缺 Limbic 的六网络核是 WM 最稳定的候选结构。它在 matched null 中的平均频率为 18.55/30、最大值为 24/30，而 observed 为 26/30（经验 $p=0.047619$）。在 29 名共同被试中，该核从静息态的 16/29 增至 WM 的 25/29：15 人两种状态均进入 top-3，10 人仅 WM 出现，1 人仅静息态出现，exact McNemar $p=0.011719$；对“缺 Limbic 六网络核”和“全七网络核”两项重点对照作 Holm 校正后 $p=0.023438$。相对地，全七网络核由静息态的 20/29 降至 WM 的 13/29，但差异未达到显著（McNemar 与 Holm $p=0.092285$），且 WM observed 频率不高于 matched null。

因此，REST–WM 的横向幅度比较以 observed raw $\Xi$ 为准；在此基础上，greedy 核分布还显示 WM 更集中于缺 Limbic 的广域网络组合。该结果表明 WM 条件下 Vis、SomMot、DorsAttn、SalVentAttn、Cont 与 Default 的历史联合结构更频繁地成为 top 核；它不证明 Limbic 在工作记忆中不参与，也不把该六网络集合解释为唯一生物学 atom。20-null 的分辨率、不同拟合长度、任务共同驱动和未提供的运动/生理混杂仍限制机制解释。

<a id="appendix-i"></a>

## 附录 I：HCP500 七任务 raw $\Xi$ 与长度匹配方差

<a id="appendix-i-1"></a>

### I.1 七任务 raw $\Xi$ 排序

作为原始表征参照，为检验静息态 raw $\Xi$ 较高的现象是否只出现在 WM，这里将同一 Schaefer-500 协议扩展到 EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL 和 WM。跨任务固定 Yeo7-PC1 表征形式、八阶 $\Delta$-Ridge、$p=8$、$\alpha=10$、56 维历史 source、7 维 target 和 Gaussian log-det 估计器；但每个“任务 $\times$ 被试”都使用自己的前 75% 时间点重新拟合 PCA、标准化、Ridge 系数、截距和残差协方差。所谓“固定模型”因此指结构与超参数固定，不是跨任务复用同一组系数。本轮只比较 observed raw $\Xi$，不生成新的 circular-shift null。第 2.1 节的任务诱发 PCA–$\Xi$ 分解是当前任务空间归因主结果。

| 条件 | 29 名共同被试 raw $\Xi$ 均值 | 中位数 |
|---|---:|---:|
| REST | **6.195996** | **6.126889** |
| EMOTION | 4.456909 | 4.445611 |
| GAMBLING | 4.717801 | 4.601157 |
| LANGUAGE | 4.556690 | 4.334584 |
| MOTOR | 5.170680 | 5.074332 |
| RELATIONAL | 4.755165 | 4.565034 |
| SOCIAL | 4.664090 | 4.681849 |
| WM | 4.940956 | 4.543372 |

![静息态与七任务 raw Xi 比较](../../results/hcp_schaefer500_all_tasks_phi/rest_all_tasks_raw_phi.png)

*图 I1. 固定 Yeo7-PC1 历史表征下静息态与七任务的 raw $\Xi$ 比较。a：八个条件的跨被试分布；b：29 名共同被试的 REST–WM 配对关系；c：各条件取得八条件最大值的被试数。*

八条件重复测量的 Friedman 检验为 $\chi^2=49.919540$、$p=1.50\times10^{-8}$。REST 与每个任务的受试者内差值均为负，即任务 raw Xi 低于 REST；七项 paired sign-flip 的 Holm 校正 $p$ 从 $3.50\times10^{-5}$ 到 $8.45\times10^{-4}$，全部低于 0.05。与 REST 最接近的是 MOTOR，task-minus-REST 均值差仍为 $-1.025316$ bits（95% bootstrap CI $[-1.541913,-0.491539]$，Holm $p=0.000845$）；差距最大的是 EMOTION，为 $-1.739087$ bits（95% CI $[-2.296265,-1.216161]$，Holm $p=3.50\times10^{-5}$）。因此在预先使用的 $p=8,\alpha=10$ 配置下，**REST 的群体平均 $\Xi$ 显著高于全部七个任务态。**附录 E 进一步检验该排序对超参数的依赖，结论不能外推为与 $p,\alpha$ 无关的普遍规律。

该排序不是逐人定律。REST 在 20/29 名共同被试中是八条件最大值；其余 9 人的最大值分别为 MOTOR 3 人、SOCIAL 2 人、WM 2 人、GAMBLING 1 人和 RELATIONAL 1 人。REST 的降序中位排名为第 1，但平均排名为 2.17。质量规则只标记 `sub-103515/WM` 的极端早期 PC1 瞬变，主分析仍保留该点。由于本轮未计算各任务 null，结论只针对 raw estimator 输出，不能进一步断言 REST 相对于各任务自身时间结构具有更高的 null 校正协同。

<a id="appendix-i-2"></a>
<a id="hcp-length-matched-variance"></a>

### I.2 REST–任务长度匹配方差检验

为检验 REST 的跨被试离散度是否只是由 1200 点长序列造成，这里对每个任务长度分别从 REST1_LR 取 12 个覆盖完整 run 的等距窗口：EMOTION 176 点、GAMBLING 253 点、LANGUAGE 316 点、MOTOR 284 点、RELATIONAL 232 点、SOCIAL 274 点和 WM 405 点。每个窗口都独立使用前 75% 时间点重新拟合 Yeo7 PC1、标准化、$p=8,\alpha=10$ 的 $\Delta$-Ridge 和残差协方差；任务态复用附录 H.1 的相同估计器计算 raw $\Xi$。因此唯一主动改变的分析因素是 REST 序列长度，不使用 null 模型。

![REST 与七任务的长度匹配 raw Xi 方差](../../results/hcp_schaefer500_length_matched_variance/length_matched_variance.png)

*图 I2. REST 与七任务的长度匹配 raw $\Xi$ 方差比较。每个蓝点是同一被试 12 个等长 REST 窗口的均值，橙点为任务值，灰线连接同一被试。*

七个小图分别给出 REST 与一个任务的两两配对分布。箱线图和面板内 SD 直接显示两组跨被试离散度，所有小图共享同一纵轴。图用于展示稳定的被试级 REST 值；下表的正式方差比仍对 12 个窗口位置分别计算跨被试方差后取平均，避免只看窗口均值掩盖时间位置变化。

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

<a id="appendix-j"></a>

## 附录 J：REST 与七任务的共同方差空间参照

REST 没有 `taskRetained`/`taskRegressed` 配对，因此不存在“被 task GLM 解释的方差”，不能在任务态 TEVF 图中把 REST 人为设为零。为将 REST 与七任务置于同一可计算口径，这里另用每个状态的 `taskRetained` 或 REST 原始时序计算 parcel 时间方差

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

该量只比较一个 run 内方差如何分布到不同 parcel，不比较 REST 与任务的绝对 BOLD 方差大小，也不受全局乘性尺度影响。八状态的状态特异对比定义为

$$
d^{\mathrm{var}}_{sci}
=q^{\mathrm{var}}_{sci}
-\frac{1}{7}\sum_{c'\neq c}q^{\mathrm{var}}_{sc'i}.
$$

REST 与全部七任务共有 29 名被试。图中白色或黑色横线将 REST 与任务分开；四个 panel 与任务态 TEVF 图保持相同的 parcel、Yeo7 和状态减其余状态结构，但颜色表示 parcel 时间方差富集，而不是 TEVF。

![REST 与七任务的 Schaefer-500 时间方差空间分布](../../results/hcp_schaefer500_task_specific_regions/rest_all_tasks_variance_profiles.png)

*图 J1. REST 与七任务的 Schaefer-500 parcel 时间方差空间参照。a、b：parcel 与 Yeo7 网络的状态内方差富集；c、d：各状态相对其余七状态的富集差。*

REST 的 Yeo7 方差富集以 Limbic（1.38）、Vis（1.26）和 Default（1.22）较高；相对其余七状态，REST 最突出的是 Limbic（$+0.20$）和 Default（$+0.07$），相对较低的是 DorsAttn（$-0.13$）和 Vis（$-0.10$）。任务对比仍保留预期结构，例如 LANGUAGE 相对富集 Cont（$+0.20$），MOTOR 相对富集 Limbic（$+0.25$）和 SomMot（$+0.12$），RELATIONAL 相对富集 Vis（$+0.31$），SOCIAL 相对富集 DorsAttn（$+0.24$），WM 相对富集 Vis（$+0.21$）。

对同一方差富集图执行八状态 LOSO 最近质心分类，500-parcel 准确率为 58.6%（136/232；chance 12.5%；2,000 次被试内置换 $p=0.0005$），Yeo7 网络均值准确率为 37.9%（88/232，$p=0.0005$），REST 的召回率为 44.8%（13/29）。该结果低于七任务 TEVF 的 90.0%，符合两种口径的差异：TEVF 有针对性地隔离 task GLM 成分，而共同方差同时包含自发活动、任务活动和残余混杂。因此，图 J1 只为 REST 提供统一但更宽泛的空间参照，不替代正文第 2.4 节的任务态 TEVF 主结果。

<a id="appendix-k"></a>

## 附录 K：HCP500 个体认知画像与候选筛选诊断

本附录集中保留 Schaefer-500 的原合并主图、个体异质性、探索性筛选和数值一致性检查。正文图 2 改以 Schaefer-1000 为主，并用直接的 Story、Math 任务表现替换旧图 g--i；以下材料用于说明 500 分区结论如何产生、哪些部分跨粒度保留，以及探索性候选受到的多重比较限制。

![Schaefer-500 REST 与七任务的 system-level Xi、网络份额、层级 atom 和认知画像](../../results/hcp_schaefer500_task_evoked_xi_tuning/final/task_evoked_xi_main_combined.png)

*图 K0｜Schaefer-500 原合并主图。a：REST 与七任务的 system-level $\Xi$；b：主要 greedy atom 的绝对贡献；c：system-level $\Xi$ 的七网络平均组成份额；d：四个冻结认知因子；e--f：一般认知分别与 LANGUAGE、MOTOR 全七网络 atom 的关系；g--i：从全组合扫描提出的晶体认知、记忆和加工速度探索候选。该图现在只作为 1000 分区正文主图的补充和候选来源记录。*

Schaefer-500 使用与正文相同的一维网络状态、三阶历史和 Ridge $\alpha=1$；任务态 PC1 平均累计解释方差为 67.35%，REST 为 44.53%。232 个模型的平均 held-out RMSE/持久性基线比为 0.907，其中 207/232 个优于持久性基线。REST system-level $\Xi$ 均值为 7.040 bits，七任务为 4.301--5.537 bits，七项 REST--任务配对检验经 BH 校正后均显著。REST 中 SomMot、DorsAttn 和 Default 的平均份额较高，LANGUAGE 的 Control 份额为 20.5%，RELATIONAL 与 SOCIAL 的 DorsAttn 份额分别为 20.5% 和 21.4%；仅比较七任务时 7/7 个网络均有显著状态效应。REST 的全七网络 atom 为 1.113 bits，多数任务的主要高阶组合集中在缺少 Limbic 的六网络核，贡献为 0.692--1.023 bits。

旧图 K0e--f 中，LANGUAGE 全七网络 atom 与一般认知正相关（$\rho=+0.518$，原始 $p=0.00402$），MOTOR 为负相关（$\rho=-0.400$，原始 $p=0.03133$）。换到 Schaefer-1000 后，LANGUAGE 增至 $+0.578$，MOTOR 减弱为 $-0.306$；两个相关之差仍为 $0.884$，置换 $p=0.000400$。因此群体状态、网络重分配和主要高阶组合属于强跨粒度结果，LANGUAGE--MOTOR 的一般认知方向差异属于中等跨粒度结果。

SomMot+Limbic+Cont 的 Story--Math 模式也由 Schaefer-500 提出。500 分区中 Story $\rho=+0.406$、Math $\rho=-0.224$，$\Delta\rho=+0.630$，Williams 原始 $p=0.0093$、三候选 Holm $p=0.0280$、bootstrap 95% CI 为 $[+0.192,+1.012]$。1000 分区保留 85.6% 的差异效应量；同一人的固定组合协同在两种分区间高度相关（$\rho=+0.849$，$p=5.97\times10^{-9}$）。这个很小的 $p$ 说明同一批被试的个体排序对 parcel 粒度稳定，不是独立队列复现，也不等于 Story 或 Math 脑--行为相关本身具有同样强的显著性。

旧图 K0g--i 的搜索空间校正结果集中如下。BH 在每项认知的 2,872 个候选特征内计算，maxT 使用同一搜索空间；三项候选均未达到校正后阈值。

| 认知候选 | BH $q$ | maxT $p$ |
|---|---:|---:|
| 晶体认知--EMOTION：DorsAttn+Limbic | 0.8545 | 0.5525 |
| 记忆--SOCIAL：SalVentAttn+Limbic+Default | 0.3092 | 0.1256 |
| 加工速度--RELATIONAL：Vis+Limbic+Cont | 0.9987 | 0.6756 |

![按一般认知排序的逐被试网络归因](../../results/hcp_cognition_individual_xi_profiles/g_ranked_network_attribution.png)

*图 K1. 按一般认知从高到低排列的网络归因。a：四个认知因子，每列在 29 人内部标准化；b：同一被试在 REST 和七任务中的七网络 $C_g/\Xi$ 份额。全部被试共用 0%--32% 色标。*

![按一般认知排序的逐被试层级 atom 贡献](../../results/hcp_cognition_individual_xi_profiles/g_ranked_atom_contributions.png)

*图 K2. 按一般认知从高到低排列的层级 atom 绝对贡献。a 与图 K1 相同；b：同一被试在八状态中的固定 12 个 greedy atom。颜色在全部被试共同的第 99.5 百分位 1.790 bits 处截断，逐被试页的单元格数字保留实际值。*

其余认知因子排序图采用与图 K1--K2 完全相同的被试、列顺序和色标，只改变被试行顺序：

| 排序因子 | 网络归因 | 层级 atom 贡献 |
|---|---|---|
| 一般认知 | [查看网络图](../../results/hcp_cognition_individual_xi_profiles/g_ranked_network_attribution.png) | [查看 atom 图](../../results/hcp_cognition_individual_xi_profiles/g_ranked_atom_contributions.png) |
| 晶体认知 | [查看网络图](../../results/hcp_cognition_individual_xi_profiles/cry_ranked_network_attribution.png) | [查看 atom 图](../../results/hcp_cognition_individual_xi_profiles/cry_ranked_atom_contributions.png) |
| 记忆 | [查看网络图](../../results/hcp_cognition_individual_xi_profiles/mem_ranked_network_attribution.png) | [查看 atom 图](../../results/hcp_cognition_individual_xi_profiles/mem_ranked_atom_contributions.png) |
| 加工速度 | [查看网络图](../../results/hcp_cognition_individual_xi_profiles/spd_ranked_network_attribution.png) | [查看 atom 图](../../results/hcp_cognition_individual_xi_profiles/spd_ranked_atom_contributions.png) |

![网络归因和层级 atom 与认知因子的描述性相关](../../results/hcp_cognition_individual_xi_profiles/descriptive_brain_cognition_spearman.png)

*图 K3. 脑信息特征与认知因子的描述性 Spearman 相关。a：四个认知因子与 $8\times7=56$ 个状态--网络份额；b：四个认知因子与固定 12 个 atom 在八状态中的绝对贡献。仅标注 $|\rho|\geq0.40$ 的单元；未调整人口学、头动、信号质量或家系结构。*

![四项认知因子的原始 P 值 top 状态--脑网络组合](../../results/hcp_cognition_atom_exploration/top_atom_associations.png)

*图 K4. 四项认知因子的探索性状态--atom 关联。每个面板只显示同时满足原始双侧 $p<0.05$ 和正贡献人数 $n+\geq5$ 的候选，并按 $|\rho|$ 排序；该图不使用校正后的 $q$ 值筛选。*

![四项认知第一名候选的逐被试散点](../../results/hcp_cognition_atom_exploration/top_atom_scatter.png)

*图 K5. 每项认知第一名候选的个体散点。橙色表示 atom 在该被试中具有正贡献，灰色零值表示该组合未进入其正贡献 greedy 路径。除一般认知--LANGUAGE 全七网络组合覆盖 29/29 人外，其余候选仅覆盖 5--9 人。*

![指定组合起步与固定组合总协同的 29 人认知相关](../../results/hcp_cognition_targeted_greedy_followup/targeted_greedy_cognition_scatter.png)

*图 K6. 指定候选组合起步后的 29 人脑--认知关系。a--c：指定组合首步 greedy 残差；d--f：相同组合对全七网络 target 的固定组合总协同。两类被试采用相同重算口径，虚线仅作线性视觉引导。*

![指定组合起步实验的精确数值重合审计](../../results/hcp_cognition_targeted_greedy_followup/targeted_greedy_exact_overlap_audit.png)

*图 K7. 代码一致性审计。a：旧缓存的自由 greedy atom 与本次重跑自由 greedy；b：本次重跑自由 greedy 与从同一候选组合单独启动得到的首步残差。两个比较的最大绝对差均为 0 bits。*

![三项领域认知的全组合相关搜索景观](../../results/hcp_cognition_exhaustive_targeted_greedy/exhaustive_search_landscape.png)

*图 K8. 三项认知各 2,872 个候选特征的相关搜索景观。横轴为 Spearman $\rho$，纵轴为 $-\log_{10}(\mathrm{raw}\ p)$；黑星标出逐项置换 $p$ 最小的全样本候选。该图强调旧图 K0g--i 来自大范围数据驱动筛选，而不是三个预先指定的单项检验。*
