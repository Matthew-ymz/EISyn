# 脑科学实验：Schaefer100 DMF 耦合转变与 HCP Yeo7 $\Xi$ 分解

## 结论

1. **Schaefer100 DMF 结果保持不变。** 群体平均结构连接上的 $\Xi$ 与 pairwise BOLD-like $\Phi^R$ 将耦合转变定位在相邻粗网格，但峰位依赖预测时距，因此不把单一峰解释为时间尺度无关的严格临界点。峰值邻近窗的 $\Xi$ 以跨 ROI、尤其跨网络分量为主；结构 null 说明经验边布局主要改变网络间份额，而非最大化总 $\Xi$。
2. **57 人 Schaefer-1000 主结果复现原结论。** 在统一的任务诱发 PCA、三阶历史和 Ridge $\alpha=1$ 下，REST 的 system-level $\Xi$ 均值为 7.122 bits，高于七任务的 4.633--6.243 bits；七项配对 Wilcoxon 检验均经 BH 校正显著。相较原 29 人，八状态均值排序完全一致，网络份额矩阵相关为 0.989，greedy atom 矩阵相关为 0.878，说明幅度排序与内部组成均稳定。
3. **57 人方法学验证限定了结论边界。** REST circular-shift null 中 56/57 人高于 null 均值；缺 Limbic 的六网络核高于 matched-null 频率。25 点参数网格中只有 12 点保持 REST 群体均值最高、7 点七项对比全部显著，表明结论依赖合理正则化，而非对任意超参数成立。按留出误差选择的最优共享点为 $(p,\alpha)=(5,10)$，且仍保留七项显著 REST 优势。Schaefer-1000 TEVF 的七任务 parcel 图可在留一被试分类中达到 90.2%，说明任务空间模式具有稳定的个体外可辨识性。

本文不再报告 Schaefer-500 HCP 实验，也移除了原 29/30 人的探索性认知关联；HCP 正文与验证统一使用同一批 57 名 Schaefer-1000 被试。

## 目录

1. [**Schaefer100 DMF：跨连接组耦合转变**](#dmf-main)
2. [**HCP Schaefer-1000：57 人任务态 $\Xi$ 与脑区分布**](#hcp-main)
3. [**讨论：解释边界与可复现性**](#discussion)
4. [**附录 A：DMF 补充诊断、EI 分量与 Kuramoto 对照**](#appendix-a)
5. [**附录 B：Schaefer100 DMF 动力学方程**](#appendix-b)
6. [**附录 C：83 ROI 与 100 ROI 受控比较**](#appendix-c)
7. [**附录 D：Schaefer100 可复现文件**](#appendix-d)
8. [**附录 E：HCP Schaefer-1000 的 57 人验证**](#appendix-e)

<a id="dmf-main"></a>

## 1. Schaefer100 DMF：跨连接组耦合转变

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

这里需要区分两个参照：平均发放率最大离散导数位于 $G=1.5$，而全局扫描得到的 $\Xi$ 峰值点是 $G=1.3$。由于图 1 C–G 要回答的是“$\Xi$ 峰附近的协同如何分布”，旧窗口 $G\in\{1.4,1.5,1.6\}$ 全部落在峰后，不能覆盖峰前—峰值—峰后的局部形状。因此本版按当前 0.1 网格，以 8/8 个 seed 一致的峰位 $G=1.3$ 为中心，重新选定对称三点窗口 $G\in\{1.2,1.3,1.4\}$。这是依据完整扫描结果作出的峰值邻近窗，不是独立预注册的临界性检验；$G=1.5$ 仍保留为发放率转折参照。

系统级指标为

$$
\Xi=EI_{\mathrm{do}}(\mathbf{s}_t;\mathbf{y}_{t+300})-
\sum_{i=1}^{200}EI_{\mathrm{do}}(s_{t,i};\mathbf{y}_{t+300}). \tag{2}
$$

为保持与旧 83 ROI 实验的估计口径一致，本实验继续使用 ridge 为 $10^{-6}$ 的 Gaussian 条件协方差估计。200 维 source、200 维 target、21×8 个条件和每条件 2,048 个样本使逐条件 TM 全量扫描的计算代价过高。Gaussian 估计只刻画二阶依赖，可能遗漏高阶非线性结构，并对协方差正则化敏感；因此主要解释曲线形状、配对峰位和层级比例，而不把绝对 bits 直接外推到其他估计器。

为与 Mediano、Rosas、Luppi 等人在 DMF 上使用的 integrated information decomposition 指标对照，另从每个自然稳态模拟的区域兴奋性发放率生成 Balloon–Windkessel BOLD-like 序列，并对全部 $\binom{100}{2}=4{,}950$ 个无序 ROI 对计算 Gaussian minimum-mutual-information（MMI）$\Phi^R$。对任一 ROI 对 $(X,Y)$ 和一个 BOLD 积分样本的滞后，定义

$$
\Phi^R_{XY}
=I(X_t,Y_t;X_{t+1},Y_{t+1})
-I(X_t;X_{t+1})-I(Y_t;Y_{t+1})
+\min_{i,j\in\{X,Y\}}I(i_t;j_{t+1}).
$$

每个 seed–$G$ 条件先对 ROI 对取平均，再跨 8 个 seed 汇总。该扫描使用与主实验相同的经验 SC、固定 JFIC、$G=0.0$–3.0 网格、seed 3–10、$\Delta t=0.001$ s 和噪声幅度 0.01；自然轨迹总长 1.5 s，最短 burn-in 为 0.3 s。$\Phi^R$ 的非负容差预先设为 $10^{-10}$ bits：仅容差内负值可记为数值零，低于阈值则中止。正式扫描的最小逐对原始值为 0.00104 bits，没有负值或归零。这里的 $\Phi^R$ 是 pairwise、observational、BOLD-like、one-step 指标，而式（2）的 $\Xi$ 是 full-state、interventional、300-step 指标；二者只比较随 $G$ 的峰形与峰位，不比较绝对 bits。

两个补充实验保持结构连接、JFIC、噪声、干预支持、样本量、seed 和估计器不变。动力学诊断实验只将 $G=1.10$–1.70 加密到 0.02，并增加 rate susceptibility、神经发放率相位 metastability 和确定性固定点 Jacobian 最大实部。预测时距实验只改变 target horizon 为 50、100、200、300、400、500 steps，并在同一 seed 和 $G$ 内复用配对 source 与噪声前缀。两项实验均预先声明 $\Xi$ 非负容差为 $10^{-8}$ bits；容差内负值可视为数值零，低于该阈值则中止。正式结果中最小值分别为 22.728 和 1.531 bits，没有负值、容差内归零或状态边界越界。

<a id="dmf-phi"></a>

### 1.4 全局扫描

平均发放率从 $G=0$ 的 1.952 Hz 单调上升到 $G=3.0$ 的 43.123 Hz，最大离散导数位于 $G=1.5$。$\Xi$ 从 $G=0$ 的 $8.452\pm0.102$ bits 上升，在 $G=1.3$ 达到 $26.131\pm0.163$ bits 的峰值，随后降至 $G=3.0$ 的 $6.302\pm0.038$ bits；图 1 A 的误差均为跨 8 个 seed 的 SD。峰前增幅为 17.679 bits，即 209.17%。8/8 个 seed 的峰均位于 $G=1.3$。重新选定窗口内 $G=1.2,1.3,1.4$ 的均值依次为 25.617、26.131 和 25.887 bits，覆盖峰前上升、峰值和峰后回落。

图中的误差条带改用跨 seed SD，以直接呈现随机 seed 波动。在 $G=1.3$，$\Xi$ 的 SD 为 0.163 bits，仅占均值的 0.62%；whole EI 与单变量 EI 之和的 SD 分别为 0.327 和 0.224 bits，对应变异系数为 0.15% 和 0.11%。因此 seed 波动相对于曲线幅度很小，不影响峰位判断。

observational $\Phi^{WMS}$ 从 $G=0$ 的 −21.433 bits 下降，在 $G=1.3$ 达到最负均值 −502.591 bits，并在 $G=1.4$ 跳回 −189.520 bits，呈现与发放率转折对齐的明显折点。其绝对值受 Gaussian 条件协方差和 ridge 影响，应优先解释形状。

pairwise BOLD-like $\Phi^R$ 从 $G=0$ 的 $3.631\pm0.159$ bits 上升，在 $G=1.2$ 达到 $6.906\pm0.095$ bits，随后回落；4/8 个 seed 的峰位为 $G=1.2$，另 4/8 个为 $G=1.3$。在 $G=1.3$，均值仍为 6.324 bits，但 SD 增至 1.249 bits，来自两个 seed 的提前回落。因此，$\Phi^R$ 与 $\Xi$ 支持的是 $G=1.2$–1.3 的共同峰带，而不是完全重合的单点定位；两者都早于发放率最大变化点 $G=1.5$。由于 $\Phi^R$ 与 $\Xi$ 共享同一 DMF 和 SC，且 observable、source/target 维度与预测时距不同，这一对齐属于同模型内的交叉指标一致性，不是独立数据复现。

![Schaefer100 DMF 多尺度汇总](../../fig/dmf_schaefer100/dmf_schaefer100_summary_full.png)

*图 1｜Schaefer100 DMF 多尺度汇总。A：上部为平均发放率与 full-state interventional $\Xi$，下部左轴为 full-state observational $\Phi^{WMS}$，右轴为全 4,950 个 ROI 对平均的 BOLD-like Gaussian-MMI $\Phi^R$；曲线为 8 个 seed 的均值，阴影为跨 seed SD。橙色点划线、紫色虚线和灰色点线依次标出粗网格 $\Phi^R$ 峰 $G=1.2$、$\Xi$ 峰 $G=1.3$ 与发放率最大变化点 $G=1.5$。三种信息量的 observable、维度与预测时距不同，双轴只用于比较峰形和峰位。B：$G\times$ target horizon 的跨 seed 平均 $\Xi$ 热图，白线为各时距的平均峰位；不同 horizon 改变 target 分布，颜色的绝对 bits 不宜直接作跨行强度比较。C–G 基于峰值邻近窗 $G\in\{1.2,1.3,1.4\}$，先在每个 seed–$G$ 条件中独立分解，再对 8 个 seed × 3 个 $G$ 的 24 个条件汇总。C：各 ROI 的局部 E/I（ROI 内）耦合与跨 ROI leverage；D：ROI 内/跨 ROI 比例，柱高为 8 个 seed 的均值，误差线为跨 seed SD；E：Yeo-7 网络内跨 ROI 分量；F：网络间 $\Xi$ 精确 Shapley 归因；E–F 标签斜杠后的数字为 ROI 数，并采用相同网络顺序和颜色；G：每个 ROI 的跨 ROI leverage 在 24 个条件上的均值，映射到双侧外侧面与内侧面。EI 分量及其余补充动力学图见附录 A。*

<a id="dmf-horizon"></a>

### 1.5 预测时间尺度

图 1B 只改变 target horizon 为 50、100、200、300、400 和 500 steps；同一 seed 与 $G$ 下复用 source 与噪声前缀。热图显示 $\Xi(G)$ 景观随预测时距发生明显变化，因此单一 300-step 峰位不能被解释为时间尺度无关的系统常数。由于 horizon 同时改变未来状态的可预测信息量，正文不进一步比较各行绝对 bits，也不依据峰位折线提出单调迁移机制；逐 seed 峰位轨迹、代表性曲线和加密动力学诊断统一保留在附录 A。

<a id="dmf-hierarchy"></a>

### 1.6 峰值邻近窗层级分解

本节所有图 1 C–G 结果都采用相同汇总流程。对每个 seed $s\in\{3,\ldots,10\}$ 和每个 $G\in\{1.2,1.3,1.4\}$，先独立生成干预样本、演化目标状态、估计条件协方差并完成层级分解；随后对所得 24 个 seed–$G$ 条件做等权汇总。因此正文中的 bits 是 24 个条件的算术平均，不是某个特定 $G$ 的取值。图 1D 的比例仍先在每个条件内计算，而不是用平均分量除以平均总量。为单独显示随机 seed 波动，图 1D–F 的误差线先在每个 seed 内平均三个 $G$，再计算 8 个 seed 之间的 SD；三个 $G$ 的位置差异不进入误差线。该 SD 描述模拟随机性，不应解释为人群统计不确定性。

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

按图 1D 的 seed-blocked 汇总，ROI 内与跨 ROI 比例的跨 seed SD 均为 0.20 个百分点，说明 88.00% 的跨 ROI 优势不是由个别 seed 推高。

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

其中每支箭头表示把更细的 source 块合并后新增的 EI，三支箭头依次对应 ROI 内、网络内跨 ROI 和网络间分量。图 1F 后续使用 Shapley 值，只是把最后的网络间 EI 增量对称归因给七个网络；Shapley 不参与式（3）和式（4）本身的数值闭合。

网络内跨 ROI 分量为 5.622 bits，占总 $\Xi$ 的 21.73%；网络间分量为 17.151 bits，占总量的 66.27%。24/24 个条件均为网络间大于网络内。网络内分量以 Visual（2.288 bits）和 Somatomotor（1.677 bits）最高，其后为 Dorsal attention（0.610）、Default mode（0.577）、Salience/ventral attention（0.299）、Frontoparietal control（0.153）和 Limbic（0.018）。这些绝对量同时受网络所含 ROI 数影响，不能直接解释为单位 ROI 效应。

为把网络间整合与图 1E 的网络内分量对照，图 1F 对七个 Yeo 网络的全部 $2^7=128$ 个联盟进行精确 Shapley 归因。对任意网络子集 $S$，联盟价值定义为该子集的联合 EI 超出其成员网络 EI 之和的部分，单网络和空集价值为 0。网络 $i$ 的份额为

$$
\psi_i=\sum_{S\subseteq\mathcal{N}\setminus\{i\}}\frac{|S|!(7-|S|-1)!}{7!}\left[v(S\cup\{i\})-v(S)\right]. \tag{5}
$$

按 Shapley 效率性质，$\sum_i\psi_i=\Xi_{\mathrm{between\ networks}}$；代码在每个 seed–$G$ 条件上检验该闭合关系。这里的 $\psi_i$ 是多变量网络间整合的对称归因，不是成对网络边，也不是唯一的生物学因果归属。主图报告守恒的绝对 bits，网络规模诊断另按每 ROI 及每个跨网络连接机会数归一化记录在结果摘要中。

峰值邻近窗平均归因以 Default mode（3.541 bits）、Salience/ventral attention（3.416）和 Dorsal attention（2.909）最高，其后为 Frontoparietal control（2.788）、Somatomotor（2.587）、Visual（1.539）和 Limbic（0.371）。七项之和为 17.151 bits，与式（4）的网络间分量一致，最大逐条件闭合误差为 $3.6\times10^{-15}$ bits。按每 ROI 或每个跨网络连接机会数归一化后，前三名均为 Salience/ventral attention、Frontoparietal control 和 Dorsal attention，说明主要排序并非仅由网络规模造成。图 1E 与图 1F 因而形成明确对照：Visual 更突出网络内部模块化耦合，而显著性、默认、注意和控制相关网络承担更多跨网络整合归因。

<a id="dmf-topology"></a>

### 1.7 结构关联与稳定性

跨 ROI leverage 与加权结构强度呈正 Spearman 相关（$\rho=0.971$，$p=2.02\times10^{-62}$），ROI 内耦合与结构强度呈负相关（$\rho=-0.984$，$p=3.09\times10^{-75}$），ROI involvement 与结构强度呈正相关（$\rho=0.969$，$p=5.14\times10^{-61}$）。24 个条件的 ROI involvement 排名两两 Spearman 相关中位数为 0.984，最小值为 0.936，说明空间排序不由单一 seed 或单一 $G$ 驱动。

involvement 和 leverage 是留一块条件总相关下降量。它们是非负敏感性分数，但彼此重叠，不是互斥且可相加的信息原子。

图 1G 进一步显示，跨 ROI leverage 不是集中在单一功能系统，而是在双侧形成多个空间锚点。最高值包括 SomMot parcel、枕叶视觉 parcel、Salience/ventral attention 的内侧及额岛 parcel，以及楔前叶/后扣带和部分扣带–控制 parcel。楔前叶/后扣带高值与结构连接研究中的 posterior medial structural core 相互印证：Hagmann et al.（2008）将后内侧和顶叶皮层识别为高 degree、strength 和 betweenness 的结构核心，van den Heuvel 与 Sporns（2011）也把双侧楔前叶列入富集俱乐部枢纽。额岛、扣带和控制区高值则与 connector-hub 文献中“跨模块连接支持整合，同时维持模块化”的机制相容（Bertolero et al., 2018）。不过，本实验中 leverage 与结构强度的相关达到 $\rho=0.971$，因此图 1G 首先是当前 SC 骨架在干预动力学中的空间投影，不能把这种对应视为独立于结构强度的新验证。

更值得注意的是，图 1G 同时突出视觉和躯体运动等单模态区域，以及楔前叶/后扣带、显著性和控制相关区域，并不沿“单模态到跨模态”的主梯度单调升高（Margulies et al., 2016）。这对“高协同只位于高阶联合皮层”的简单解释构成修正，却与 Varley et al.（2023）的结果相容：高阶协同子系统遍布皮层，其较稳定的参与热点包括枕极、楔前叶和扣带区域，而且高协同组合往往跨越经典功能网络。结合图 1E–F，可得到一个新的层级区分：视觉和 SomMot parcel 可以因结构嵌入而具有较高的跨 ROI leverage，但这不等于其所属网络承担最多的跨网络归因；后者仍以 Default、Salience/ventral attention、Dorsal attention 和 Frontoparietal control 更突出。因此，图 1G 定位的是“移除某个 ROI 会使跨区条件总相关下降多少”，而不是认知层级、网络间 Shapley 份额或局部协同的直接脑图。

<a id="dmf-structural-nulls"></a>

### 1.8 结构保持 null：经验 SC 的增量

结构强度与 leverage 的相关不能回答经验连接布局是否具有超出低阶结构约束的作用。为此，本实验对经验 SC 和三类单一 null 实现分别重新校准 JFIC，并在完全相同的 $G=0.0$--$3.0$ 网格、8 个配对 seed、干预分布、300-step horizon 和 Gaussian 估计器下完成全扫描。权重置乱 null 保留全局权重多重集；degree/strength null 精确保留每个节点的二值 degree 和加权 strength，同时改变 79.9% 的缺失边位置；Yeo-block null 则在每个无序网络块对内独立置乱，精确保留全部块对的权重多重集与总权重。所有构造均保持对称、非负、零对角线及总体密度。正式扫描没有状态越界或低于 $-10^{-8}$ bits 的 Syn 非负性违反，层级分解闭合误差不超过 $2.1\times10^{-13}$ bits。

![经验 SC 与结构保持 null 的 DMF 对照](../../fig/dmf_schaefer100/dmf_schaefer100_structural_nulls.png)

*图 S1｜经验 SC 与三类结构 null。a：全耦合扫描的 $\Xi$ 均值曲线，阴影为跨 8 个配对 seed 的 SD；b：逐 seed 峰值 $\Xi$ 与峰位 $G$ 的均值和 SD；c：以各矩阵自身平均峰位为中心的三点窗口中，跨 ROI 与网络间分量占总 $\Xi$ 的比例；d：各 null 精确保留的结构约束。误差条只描述模拟 seed 波动；每类 null 仅生成一张图，因此不构成跨图实现的 null 分布或显著性检验。*

结果首先否定了较强的结构最优性假设。经验 SC 的峰值 $\Xi$ 为 26.131 bits，而权重置乱、degree/strength 和 Yeo-block null 分别为 32.688、32.278 和 31.826 bits。经验减 null 的配对差依次为 $-6.558\pm0.170$、$-6.147\pm0.098$ 和 $-5.695\pm0.072$ bits，三组均为 0/8 个 seed 中经验值更高。跨 ROI 份额也从经验 SC 的 88.00% 上升至 92.00%、90.77% 和 90.87%。因此，经验解剖结构并不最大化整体联合干预优势或跨 ROI 协同；更随机的边布局反而产生更高且更广泛的 $\Xi$。

网络间份额给出一个更窄的正归因。经验 SC 为 66.27%，低于权重置乱的 79.55% 和 degree/strength null 的 75.76%，却高于 Yeo-block null 的 64.46%。在精确保留每个 Yeo 块对的权重分布和总量后，经验具体边布局仍提高 1.81 个百分点，并在 8/8 个配对 seed 中同向。这表明经验网络块之间“有多少连接权重”并不能完全解释网络间归因，块内具体边位置还贡献了稳定但较小的增量。不过该结论只比较了一张经验图与每类一张 null 图，8 个 seed 不是 8 张独立结构图，因此目前只能称为探索性结构归因，不能报告拓扑 null 的显著性。

原始峰位变化较小：经验 SC 的逐 seed 平均峰位为 $G=1.300$，三类 null 分别为 1.350、1.263 和 1.313。由于谱半径同时从经验值 0.701 降至 0.517--0.560，有效峰耦合 $G\rho(\mathbf{C})$ 的经验值 0.911 反而高于三类 null 的 0.698、0.678 和 0.735。谱半径是置乱后的下游拓扑性质，因此原始 $G$ 与有效耦合必须并列解释：峰在原始扫描轴上近似保留，不等于网络的有效耦合尺度不变。

<a id="dmf-insights"></a>

### 1.9 综合结论与文献对照

**第一，$\Xi$ 峰反映的是“联合干预优势”最大，而不是系统信息总量最大。**在 $G=1.3$，whole EI 已由 $G=0$ 的 219.636 bits 降至 176.926 bits，单变量 EI 之和则由 211.184 bits 更快降至 150.795 bits；两者差值因而达到峰值。这说明耦合首先削弱单个变量独立解释未来全系统状态的能力，同时暂时保留联合状态中的关系信息。该解释与 causal emergence 将有效信息写成 determinism 与 degeneracy 权衡的思路一致，也与“信息转换”框架中局部信息转化为高阶协同的概念相容（Hoel et al., 2013；Varley & Hoel, 2022）。本实验新增的是：在具有结构连接约束的 DMF 中，这种联合优势沿耦合参数形成可重复的非单调峰。

**第二，$\Xi$ 景观依赖预测时间尺度，当前证据不足以把单一峰位解释为严格临界点。**图 1B 显示，target horizon 改变时，$\Xi(G)$ 的幅度和峰值位置都会变化。whole-brain 模型确实常在稳定性边界或亚稳态附近产生丰富动力学，但临界性判定要求多个独立诊断和尺度检验共同支持（Deco et al., 2011；Breakspear, 2017；Cocchi et al., 2017）。附录中的加密扫描、susceptibility、metastability 与 Jacobian 尚未形成一致定位，因此只作为探索性诊断，不承担正文主结论。

**第三，峰值邻近窗中的主要信息结构跨越功能网络边界。**跨 ROI 分量占总 $\Xi$ 的 88.00%，其中网络间分量占总量的 66.27%，且 24/24 个 seed–$G$ 条件中网络间分量均高于网络内分量。该结果与人脑高阶信息研究的总体方向一致：Luppi et al.（2022）发现协同信息更集中于跨模态、整合性皮层，Varley et al.（2023）发现高协同子系统通常跨越多个经典功能网络。区别在于，已有工作主要分析观察性 fMRI 中的统计协同；这里的 $\Xi$ 来自最大熵干预下的未来状态可区分性，并具有 ROI→网络→网络间的精确闭合分解。因此它提供的是结构约束动力学中的干预式协同证据，而不是对既有 O-information 或 PID 空间图的重复计算。

**第四，Visual 的高值只发生在网络内部，不表示它是最高级的全脑整合系统。**Visual 在图 1E 的网络内跨 ROI 分量最高（2.288 bits），但在图 1F 的网络间 Shapley 归因仅为 1.539 bits，明显低于 Default mode、Salience/ventral attention 和 Dorsal attention。视觉皮层具有高密度、拓扑规则且强同模块的局部连接，在以绝对 bits 汇总多个 parcel 的指标中容易形成较大的内部联合量；这更接近稳定的专门化模块，而非跨系统广播。真正与跨网络整合相关的是 Default mode（3.541 bits）和 Salience/ventral attention（3.416 bits）等网络。后者与显著性网络切换和控制模型相符（Menon & Uddin, 2010），前者及注意/控制网络的贡献也与 connector hub、rich-club 和动态整合研究相容（van den Heuvel & Sporns, 2011；Shine et al., 2016；Bertolero et al., 2018）。所以最有信息量的结论不是“Visual 最高”，而是 **Visual 呈现高网络内、低网络间的模块化特征；Default/Salience 呈现较低网络内、较高网络间的整合特征**。

**第五，结构强度解释空间 leverage，但经验边布局并不最大化整体 $\Xi$。**结构强度与跨 ROI leverage 呈强正相关（$\rho=0.971$），与 ROI 内耦合呈强负相关（$\rho=-0.984$），说明结构嵌入越强，归因越从局部 E/I 闭环转向跨区联合预测。然而三类结构 null 的峰值 $\Xi$ 和跨 ROI 份额都高于经验 SC，权重置乱及 degree/strength null 的网络间份额也更高。这排除了“真实强连接节点或经验拓扑经过组织以最大化整体协同”的简单解释。仍然成立的正结果更窄：在精确保留 Yeo 网络块对权重后，经验边位置使网络间份额提高 1.81 个百分点。结构中心性相关因而描述了当前骨架上的空间映射，而结构 null 进一步把可归因增量限定在块对总量之外的具体边布局。

**第六，相似的 $\Xi$ 峰形可以由不同的信息机制产生。**DMF 在发放率转折后表现为 whole-source 与 singleton-sum 的 determinism、degeneracy 四项共同下降；Kuramoto 对照则在强同步端出现 degeneracy 和 singleton 重复读出的持续增长。两者都可形成“先升后降”的 $\Xi$，但峰后动力学含义不同。这说明峰形本身不是机制指纹；将 determinism 与 degeneracy 分开呈现是必要的附录验证，也构成本实验相较只报告单一整合曲线的一个方法学增量。

**第七，83 ROI 与 100 ROI 的一致峰形支持跨连接组复现，但不是节点数消融。**在以有效耦合 $G\rho(\mathbf{C})$ 和每个 scalar source 的 $\Xi$ 对齐后，新 100 ROI 矩阵的峰位在 8/8 个 seed 中都高于旧矩阵，跨 ROI 份额也由 68.67% 增至 88.00%。然而两套连接组同时改变了节点数、密度、方向性、权重分布与谱尺度，因此这些差异只能表述为跨连接组的鲁棒性与描述性变化，不能归因于“100 ROI 更好”或单一拓扑因素。

**第八，受体数据为下一阶段提供了有文献依据但尚未使用的异质性层。**数据包中的 `average.csv` 含 19 个受体/转运体指标，但没有进入当前 DMF。Hansen et al.（2022）表明受体/转运体空间分布与结构连接、功能连接及神经动力学相关；这支持未来把区域受体谱映射为局部增益、时间常数或 E/I 参数的受控实验。当前结果不能被解释为受体梯度导致的网络差异，除非完成“仅改变受体调制项、其余参数固定”的比较并使用空间自相关保持 null。

综合而言，本实验最值得强调的新启发是：**$\Xi$ 峰附近的全脑协同并非简单地“信息更多”，而是有限预测时距下的信息从单节点可读出形式转向跨 ROI、尤其跨功能网络的联合可读出形式；经验 SC 并不最大化这种整体协同，却在网络块权重固定后保留了较小而稳定的跨网络组织增量。**这一结论把动力学扫描、干预式有效信息、多尺度归因和结构保持对照连接在同一条可审计证据链上，但不把信息峰等同于严格临界点，也不把单一 null 实现当作总体显著性证据。

下一步最优先的三个验证是：（1）为每类结构约束生成至少 10 个独立 null 实现，构造峰值 $\Xi$、峰位和层级份额的拓扑零分布；（2）对 93 个个体 SC 分别复现峰位和网络归因，以区分群体均值效应与个体差异；（3）在保持 SC、干预和估计器不变时，仅加入受体驱动的局部参数异质性，并以空间自相关保持 null 检验增量解释力。

<a id="dmf-limits"></a>

### 1.10 解释边界

当前证据支持：Schaefer100 群体 SC 驱动的 DMF 出现可重复但预测时距依赖的 $\Xi$ 峰，并在峰值邻近窗表现出更强的跨 ROI、尤其跨功能网络整合。

当前证据不支持将新矩阵断言为某一特定队列、流线数或单位，将 83/100 差异解释为纯粹的 ROI 数量效应，把模拟样本解释为真实受试者脑活动，或把 Gaussian EI 的绝对 bits 直接等同于 TM 等非线性估计器结果。在取得上游标签文件前，ROI 顺序也只能维持“高度一致的推断”状态。

此外，图 1 的 24 个条件来自 8 个模拟 seed 与 3 个固定耦合值，不是 24 名独立受试者；图 1A 的 $\Phi^{WMS}$、$\Phi^R$ 与 $\Xi$ 共享同一 DMF 和经验 SC，不能当作独立外部验证。尤其 $\Phi^R$ 使用 1 ms BOLD-like 积分样本的一步滞后，尚未检验下采样间隔或 hemodynamic 参数敏感性；它与 300-step 干预式 $\Xi$ 的峰位相近，不表示二者测量同一个数学对象。Shapley 结果依赖当前联盟价值定义，且绝对网络贡献仍受网络规模影响。结构归因实验中每类 null 也只有一个图实现，8 个配对 seed 只衡量同一结构下的模拟随机性，不能替代跨 null 图的零分布。预测时距实验虽覆盖 50–500 steps，仍没有建立连续时间或渐近极限。当前研究也尚未检验个体 SC、方向性连接、其他图谱分辨率和不同干预分布，因而应把结论限定为当前模型和分析契约下的机制性发现。

<a id="dmf-references"></a>

### 1.11 参考文献

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
17. Mediano PAM, Rosas FE, Luppi AI, et al. Toward a unified taxonomy of information dynamics via Integrated Information Decomposition. *PNAS*. 2025;122:e2423297122. [doi:10.1073/pnas.2423297122](https://doi.org/10.1073/pnas.2423297122)


<a id="hcp-wm"></a>

<a id="hcp-main"></a>

## 2. HCP Schaefer-1000：57 人任务态 $\Xi$ 与脑区分布

### 2.1 数据、表征与分析契约

本节统一使用 57 名同时具有 `REST1_LR` 和七项 LR 任务的 HCP S1200 被试。每条时序均采用 Schaefer-1000 皮层分区，并按 Yeo7 标签归入 Visual、Somatomotor、Dorsal attention、Salience/ventral attention、Limbic、Control 与 Default 七个网络。不再混入 Schaefer-500 结果，也不保留原 29/30 人的探索性认知关联。

对任务态，网络内 PCA 只在前 75% 时间点的任务成分

$$
\mathbf{U}_{sc}
=\mathbf{X}^{\mathrm{retained}}_{sc}
-\mathbf{X}^{\mathrm{regressed}}_{sc}
$$

上拟合，再以同一载荷投影完整 `taskRetained` 时序。REST 没有 task GLM 配对，因此在自身前 75% 时序上拟合并投影。每个网络保留第一主成分，形成七维状态向量 $\mathbf{x}_t$。主分析固定 $(k,p,\alpha)=(1,3,1)$，以三阶历史

$$
\mathbf{h}_t=
\left[\mathbf{x}_t^\top,\mathbf{x}_{t-1}^\top,\mathbf{x}_{t-2}^\top\right]^\top
\in\mathbb{R}^{21}
$$

预测下一时刻七网络状态。每个“被试 $\times$ 状态”分别拟合线性 $\Delta$-Ridge；前 75% 用于估计，后 25% 只作预测诊断。连续 EI 使用线性高斯 affine-TM 的协方差 log-det 闭式，单位为 bits。

将 21 个网络–滞后标量作为 singleton sources，system-level synergy 定义为

$$
\Xi_{sc}
=EI(\mathbf{h}_t;\mathbf{x}_{t+1})
-\sum_{j=1}^{21}EI(h_{t,j};\mathbf{x}_{t+1}).
$$

网络归因将同一网络的三个滞后绑定为模块 $M_g$。网络内协同为

$$
\Xi_g^{\mathrm{within}}
=EI(M_g;\mathbf{x}_{t+1})
-\sum_{j\in M_g}EI(h_{t,j};\mathbf{x}_{t+1}),
$$

跨网络部分则通过七模块联盟价值的精确 Shapley 分配。最终守恒归因 $C_g$ 满足 $\sum_g C_g=\Xi$；图中网络份额先在每名被试内计算 $C_g/\Xi$，再作群体平均，所以每个状态列严格合计 100%。数值闭合误差不超过 $6.7\times10^{-16}$。

### 2.2 57 人主结果：A–C

![HCP Schaefer-1000 57 人任务态 Xi、层级分解与网络份额](../../results/hcp_schaefer1000_task_evoked_xi_57/final/hcp_schaefer1000_panels_a_c_57.png)

*图 2｜57 人 Schaefer-1000 主结果。a：REST 与七任务的 system-level $\Xi$；白色菱形为均值，显著性为相对 REST 的双侧配对 Wilcoxon，并在七项任务内作 BH 校正。b：57 人群体平均的主要 greedy hierarchy atom 绝对贡献。c：守恒网络归因占 system-level $\Xi$ 的平均份额，每列合计 100%。*

REST 的 system-level $\Xi$ 均值为 7.122 bits；EMOTION、GAMBLING、LANGUAGE、MOTOR、RELATIONAL、SOCIAL 与 WM 依次为 4.633、4.785、5.150、5.568、5.357、6.243 与 5.323 bits。七项 REST–任务均值差均为正，范围为 0.879--2.489 bits；全部七项配对检验经 BH 校正后显著，其中最弱的 SOCIAL 对比仍有 $q=0.0304$。排除噪声协方差条件数最大的 `sub-800941` 后，七项方向和显著性均不变。因此，**REST 整体 $\Xi$ 高于全部任务态**在完整 57 人中成立，但它是群体结论，不是逐人定律：REST 高于对应任务的被试比例为 61.4%--86.0%。

相较原 29 人，八状态均值排序的 Spearman 相关为 1.000；完整 greedy atom 矩阵相关为 0.878，原 top-12 与新 top-12 重合 11 项；七网络份额矩阵相关为 0.989，平均绝对变化仅 0.331 个百分点。任务态仍主要由缺 Limbic 的广泛六网络组合及相邻高阶组合承担，LANGUAGE 的 Control 份额为 19.2%，RELATIONAL 和 SOCIAL 的 DorsAttn 份额分别为 19.5% 与 20.5%。只比较七任务时，7/7 个网络均保留经 BH 校正的状态效应。由此可见，新样本没有改变“任务压低整体 $\Xi$，同时重分配剩余协同”的核心结论。

456 个“被试 $\times$ 状态”模型中，396 个留出预测优于持久性基线；held-out RMSE/持久性基线比的均值为 0.919。该诊断说明主结果并非普遍依赖失效的一步预测器，但不把预测误差最低等同于 $\Xi$ 估计最无偏。

所有 $\Xi$/Phi 数值的非负容差预先设为 $10^{-10}$ bits：仅 $[-10^{-10},0)$ 可视为数值零，低于该阈值必须中止。主 system-level $\Xi$、跨网络 $\Xi$、REST observed/null、模块 atom 与鲁棒性网格共检查 17,271 个值，没有容差内负值或显著非负性违反；完整计数见 `nonnegativity_audit.json`。

### 2.3 57 人验证总览

![HCP Schaefer-1000 57 人 null、模块、鲁棒性、预测与 TEVF 总览](../../results/hcp_schaefer1000_57_validation_suite/final/hcp_schaefer1000_validation_overview_57.png)

*图 3｜57 人 Schaefer-1000 验证。a：REST observed $\Xi$ 减 20 次独立 PC1 circular-shift null 均值。b：greedy top-3 模块核的 observed 频率与 matched-null 频率。c：25 点 $(p,\alpha)$ 网格中，每点七项 REST–任务均值差的最小值。d：同一网格的留出 delta-NRMSE。e：七任务平均 TEVF。f：TEVF 七任务图及 REST+七任务共同方差图的 LOSO 分类准确率；黑横线为 chance。*

REST null 使用 $(p,\alpha)=(5,1)$，每名被试生成 20 次独立、非零 circular shift，并完整重拟合模型。57 人中 56 人的 observed 高于自身 null 均值，56 人达到当前分辨率下的经验 $p<0.05$；observed-minus-null-mean 的群体均值为 3.112 bits。该结果支持估计量读取到跨网络时间对齐结构，而非仅由每条 PC1 的单变量自相关产生。

模块 matched-null 将同一 Yeo7 网络的五个滞后绑定为一个不可拆原子。全七网络核进入 42/57 人的 top-3，但 matched-null 均值已达 39.2/57，频率检验不显著（经验 $p=0.238$）。缺 Limbic 的六网络核进入 22/57 人，高于 matched-null 的 14.75/57（经验 $p=0.0476）；同时缺 Limbic 与 Control 的五网络核为 19/57，对应 null 均值 4.55/57（$p=0.0476$）。因此可重复的结论是若干特定广泛组合超过 matched null，而不是“原子越大越特殊”。20 次 null 的分辨率有限，这些 $p$ 值均为未校正的验证性描述。

超参数验证在 $p\in\{1,2,3,5,8\}$ 与 $\alpha\in\{0.1,1,10,100,1000\}$ 的 25 点网格上，对 REST 和七任务使用同一组参数。REST 只在 12/25 点保持群体均值最高，7/25 点的七项 REST–任务对比全部经任务内 Holm 校正显著。弱正则、高阶历史可发生方向反转，最弱点 $(8,0.1)$ 的最小边际为 $-3.395$ bits。以八状态和 57 人等权的留出 delta-NRMSE 选择共享参数，最优点为 $(5,10)$（NRMSE 0.835）；该点最小 REST–任务边际仍为 1.179 bits，七项对比全部显著。固定 $p=8$ 时，Phi 边际与 task-minus-REST 留出误差差呈负相关（$\rho=-0.522,\ p=0.00130$），与弱正则下的过拟合解释一致。故正文主配置的方向得到支持，但不能外推为任意模型自由度下的普遍定律。

### 2.4 Schaefer-1000 任务诱发脑区分布

网络 $\Xi$ 回答联合历史如何预测未来；它不等同于经典任务定位。为保留 1000 个 parcel 的空间异质性，另从成对 `taskRetained` 与 `taskRegressed` 时序恢复 task GLM 成分。对被试 $s$、任务 $c$、parcel $i$，分别去除时间均值后令 $u_{sci}(t)=r_{sci}(t)-e_{sci}(t)$，并定义 task-evoked variance fraction：

$$
f_{sci}
=\frac{\sum_t u_{sci}(t)^2}{\sum_t r_{sci}(t)^2}.
$$

在 OLS task regression 下，$e$ 与 $u$ 正交，因此 $f$ 是 parcel 级任务解释比例。七任务跨被试与 parcel 的平均 TEVF 依次为 12.96%、13.78%、15.02%、28.45%、17.21%、19.60% 和 26.79%，以 MOTOR 和 WM 最高。

![Schaefer-1000 七任务 TEVF 与任务特异空间富集](../../results/hcp_schaefer1000_57_validation_suite/tevf/task_evoked_region_profiles.png)

*图 4｜57 人 Schaefer-1000 任务诱发空间图。a、b：parcel 与 Yeo7 网络的 TEVF；c、d：每项任务相对其余六任务的空间富集差。空间富集先将每张 parcel 图除以自身 parcel 均值，因此反映形状而非整体幅度。*

留一被试最近质心分类对 399 张个体任务图逐张预测任务标签。Schaefer-1000 parcel 图准确率为 90.2%，Yeo7 网络均值图为 65.4%；两者在 2,000 次被试内标签置换下均为 $p=0.00050$。这说明任务特异模式能推广到未参与质心估计的被试，并且 parcel 内细粒度结构提供了显著的额外辨识信息。

![Schaefer-1000 七任务空间图可辨识性](../../results/hcp_schaefer1000_57_validation_suite/tevf/task_map_discriminability.png)

*图 5｜任务空间图的群体质心相关与 LOSO 混淆矩阵。a：七任务群体质心的空间相关。b：1000-parcel 图的行归一化混淆矩阵。c：Yeo7 网络均值图的混淆矩阵。*

REST 没有 task GLM，不能定义 TEVF。为作共同口径比较，对 REST 和七任务分别计算每个 parcel 的时间方差，再除以该 run 的 1000-parcel 平均方差。八状态 parcel 图的 LOSO 准确率为 64.9%，Yeo7 网络均值为 34.6%，两项置换检验均为 $p=0.00050$。该指标混合自发活动、任务活动和残余混杂，只用于空间参照，不替代 TEVF 或 $\Xi$。

![REST 与七任务的 Schaefer-1000 时间方差空间参照](../../results/hcp_schaefer1000_57_validation_suite/tevf/rest_all_tasks_variance_profiles.png)

*图 6｜REST 与七任务的共同时间方差空间参照。a、b：parcel 与 Yeo7 网络的状态内方差富集；c、d：每个状态相对其余七状态的富集差。*

<a id="discussion"></a>

## 3. 讨论：解释边界与可复现性

57 人结果确认了主配置下的三层证据链：REST 的 system-level $\Xi$ 整体更高；任务态对剩余 $\Xi$ 的网络份额和高阶组合进行重分配；任务 GLM 成分在 Schaefer-1000 parcel 空间形成可推广到留出被试的任务特异模式。三者分别描述整体联合可预测性、网络级守恒归因和经典任务空间分布，不能互相替代。

结论有四个边界。第一，主 $\Xi$ 结果仅覆盖 REST1_LR 与七项 LR 任务，未检验 RL run、家系结构、GSR、运动与生理混杂或皮层下结构。第二，25 点网格显示方向对模型自由度与正则化敏感；弱正则的高阶模型会反转，因而不声称参数无关。第三，20 次 null 的经验 $p$ 最小为 $1/21$，模块频率检验未作跨候选多重比较；greedy atom 只用于描述候选结构，不是唯一高阶分解。第四，TEVF 依赖 `taskRetained`/`taskRegressed` 的 GLM 配对，而共同方差图不是任务解释比例。

所有 57 人 HCP 结果由冻结契约 `results/hcp_schaefer1000_57_validation_suite/experiment_contract.json` 管理。长计算保留 PCA 缓存、逐被试或逐网格 checkpoint 与 `live_progress.json`；图形同时输出 PNG、SVG 和 PDF。主要产物如下：

| 内容 | 产物 |
|---|---|
| 主图 A–C | `results/hcp_schaefer1000_task_evoked_xi_57/final/hcp_schaefer1000_panels_a_c_57.{png,svg,pdf}` |
| REST null | `results/hcp_schaefer1000_57_validation_suite/null/{summary.json,observed_minus_null.png}` |
| 模块 matched-null | `results/hcp_schaefer1000_57_validation_suite/module/{summary.json,top_core_consistency.png}` |
| 参数鲁棒性 | `results/hcp_schaefer1000_57_validation_suite/robustness/{summary.json,hyperparameter_robustness_overview.png,hyperparameter_task_margins.png}` |
| 留出预测误差 | `results/hcp_schaefer1000_57_validation_suite/prediction/{prediction_error_summary.json,prediction_error_overview.png,prediction_error_by_condition.png}` |
| TEVF 与共同方差 | `results/hcp_schaefer1000_57_validation_suite/tevf/{summary.json,task_evoked_region_maps.npz,task_evoked_region_profiles.png,task_map_discriminability.png,rest_all_tasks_variance_profiles.png}` |
| 验证总览 | `results/hcp_schaefer1000_57_validation_suite/final/hcp_schaefer1000_validation_overview_57.{png,svg,pdf}` |
| 非负性审计 | `results/hcp_schaefer1000_57_validation_suite/nonnegativity_audit.json` |

<a id="appendix-a"></a>

## 附录 A：DMF 补充诊断、EI 分量与 Kuramoto 对照

正文图 1 将 observational $\Phi^{WMS}$、pairwise BOLD-like $\Phi^R$ 与发放率、$\Xi$ 合并为一个动力学面板。为保留系统量的代数来源，原 whole EI 与 singleton EI 之和单独列于此处。

![Schaefer100 DMF whole EI 与 singleton EI 之和](../../fig/dmf_schaefer100/dmf_schaefer100_ei_components_appendix.png)

*图 A0｜Schaefer100 DMF 的 whole EI 与 200 个 singleton EI 之和。曲线为 8 个 seed 的均值，阴影为跨 seed SD；二者之差为正文的全系统 $\Xi$。*

<a id="appendix-a-1"></a>

### A.1 Schaefer100 DMF 补充动力学结果

以下结果保留用于审计 $\Xi$ 峰与其他动力学量的关系，但目前不承担正文主结论。加密扫描只将 $G=1.10$–1.70 的步长改为 0.02；rate susceptibility 定义为 $100\,\mathrm{Var}_t(\overline{r_E})$，metastability 是带通后的区域兴奋性发放率相位所构造 Kuramoto order parameter 的时间标准差，Jacobian 最大实部则在延续得到的确定性固定点上计算。

![Schaefer100 DMF 加密耦合与动力学诊断](../../fig/dmf_schaefer100/dmf_schaefer100_critical_diagnostics.png)

*图 A1｜$\Xi$ 峰与补充动力学诊断的加密对照。A：全系统 $\Xi$；B：rate susceptibility；C：神经发放率相位 metastability；D：固定点 Jacobian 最大实部。A–C 的曲线与阴影为 8 个 seed 的均值与 SD，D 为确定性固定点诊断；紫色竖线标出 $\Xi$ 峰位中位数 $G=1.32$。*

$\Xi$ 的峰在 8/8 个 seed 中位于 $G=1.32$，但 susceptibility 与 metastability 在扫描右边界仍在上升，Jacobian 最大实部最接近零的位置也没有与其重合。这些指标没有形成共同峰位，且两个随机动力学指标的最大值触及扫描边界，因此当前结果不足以支持明确的临界定位机制。

正文图 1B 已保留多预测时距实验中最直接的 $G\times$ horizon 景观。下面仅补充逐 seed 峰位轨迹和四条代表性曲线。

![Schaefer100 DMF 多预测时距补充诊断](../../fig/dmf_schaefer100/dmf_schaefer100_multihorizon_appendix.png)

*图 A2｜预测时距的补充诊断。A：8 个 seed 的峰位轨迹及均值 ± SD；B：50、100、300 和 500 steps 下的 $\Xi(G)$ 曲线。*

峰位轨迹在短时距内存在折返，并未表现为稳定的单调迁移；不同 horizon 下的绝对 $\Xi$ 幅度还受到未来信息衰减影响。因此这两个面板只说明结果具有时间尺度敏感性，不据此提出更具体的峰位迁移机制。

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

<a id="appendix-a-2"></a>

### A.2 临界峰的 EI 与 effectiveness 机制

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

#### A.2.1 Schaefer100 DMF 的 determinism/degeneracy 附录验证

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

*图 A3｜Schaefer100 DMF 的固定参考熵分解。A–D 分别为 whole-source determinism、whole-source degeneracy、singleton-sum determinism 和 singleton-sum degeneracy。曲线及阴影为 8 个 seed 的均值与 SEM；紫色虚线标出 $\Xi$ 峰值 $G=1.3$，黑色点线标出平均发放率最大变化点 $G=1.5$。四个面板使用各自的原始 bits 纵轴。*

四个分量都没有复现 Kuramoto 强耦合端的增长。DMF whole determinism 在 $G=1.2$ 达到 368.363 bits，whole degeneracy 在 $G=1.3$ 达到 182.055 bits；singleton-sum determinism 和 degeneracy 也都在 $G=1.3$ 达到最大值，分别为 36561.840 和 36411.045 bits。此后四项共同下降，到 $G=3.0$ 分别只剩 49.013、0.266、95.743 和 53.298 bits。从发放率转折点 $G=1.5$ 到 $G=3.0$，四项均在 8/8 个 seed 中下降。因此当前 DMF 的 $\Xi$ 回落发生在整体和 singleton 可预测结构同时塌缩的背景下，而不是 Kuramoto 中 degeneracy 与 singleton-sum 分量在强同步区继续膨胀的机制。

为了只比较形状，下一图先在每个模型、每个 seed、每个分量内部做 $[0,1]$ 范围归一化，再将横轴分别写成 $G/1.5$ 与 $K/K_c$。竖线 1 表示各自转变参照；比较范围固定为共同覆盖的相对耦合 0–2。这个归一化移除了绝对 bits、source 数和耦合单位差异，但没有消除模型方程、状态空间、预测时间窗及估计器差异。

![DMF and Kuramoto determinism-degeneracy shape comparison](../../fig/dmf_schaefer100/dmf_schaefer100_detdeg_kuramoto_shape.png)

*图 A4｜Schaefer100 DMF 与 $N=64$ Kuramoto determinism/degeneracy 曲线形状对照。DMF 使用 8 个 seed 的 Gaussian EI；Kuramoto 使用 2 个 seed 的 Oracle transport-map EI。每条曲线先在各自 seed 和分量内做范围归一化，再显示均值与 SEM。该图是探索性的跨模型形状比较，不是单因素受控因果对照。*

形状对照同样不支持“二者变化规律相似”。在相对耦合 0–2 上，DMF 与 Kuramoto 的描述性 Pearson 形状相关分别为 whole determinism $r=-0.008$、whole degeneracy $r=-0.785$、singleton-sum determinism $r=-0.793$、singleton-sum degeneracy $r=-0.785$。更直接地，从相对耦合 1 增至 2，DMF 四项在 8/8 个 seed 中全部下降，而 Kuramoto 四项在 2/2 个 seed 中全部上升。相关系数受插值网格与归一化方式影响，不作为显著性检验；稳定结论仅是高耦合分支方向相反。

因此，两套实验的共同点限于 whole-minus-sum $\Xi$ 都可在动力学转变附近形成峰；分解后的机制并不相同。Kuramoto 的峰伴随强同步端 degeneracy 和 singleton 重复读出的持续增长，当前 Schaefer100 DMF 则表现为四个分量在转折后共同衰减。这个阴性结果说明不能仅凭 $\Xi$ 峰形就把 DMF 的高耦合状态解释成 Kuramoto 式同步压缩。

<a id="appendix-a-3"></a>

### A.3 时间窗、相变前检测与系统规模边界

#### A.3.1 时间窗鲁棒性：避免强同步后，短窗不复现临界内部峰

基准 whole-state 曲线的 `tau=4` 结果保留为主对照。为检验其峰值是否只是高 $K$ 同步饱和造成的，新增一个严格配对的 multi-horizon Oracle sweep：对每个 seed，频率向量、均匀相位 intervention states 和 natural readout states 都固定并复用于全部 $(K,\tau)$ 条件；只改变统一的预测时间窗 $\tau\in\{0.5,0.75,1,1.5,2,4\}$，而不允许 $\tau$ 随 $K$ 自适应变化。所有条件仍使用 `N=64`、3 个 seeds、whole-state future phase target 与同一 N-source transport-map estimator。

![Paired large-N Kuramoto horizon sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep.png)

图 A 以未来 target 的 raw global order 的 $99\%$ 分位数 $R_{0.99}$ 审计强同步。预先设定 guard 为：对所有 $K$ 都要求 $R_{0.99}<0.8$。`tau=0.5` 在最强耦合 `K=4` 仍只有 $R_{0.99}=0.583$，完全通过；`tau=0.75` 为 $0.746$，也通过（仅约 $0.37\%$ target samples 的 $R\ge0.8$）。从 `tau=1` 起该 guard 开始失效：`tau=1` 仅 `K=4` 失败，`tau=1.5` 在 `K=3.2,4` 失败，`tau=2` 在 `K\ge2.6` 失败，而 `tau=4` 在 `K\ge2.2` 失败。

关键结果在图 B：**通过 guard 的两个短窗并没有给出与原图相同的临界内部峰。** `tau=0.5` 的 $\Xi$ 从 `K=0` 的约 $0$ bits 持续升至 `K=4` 的 $229.69$ bits；`tau=0.75` 同样在 `K=4` 最大，为 $262.20$ bits。因此，在目标尚未进入强同步区的有限短时间内，耦合增强主要表现为 whole-state 联合可预测性的持续增强，而非在 $K_c\approx1.596$ 附近形成回落前的峰。随着时间窗变长，最大值才逐步向低 $K$ 移动：`tau=1` 的峰在 `K=4`（$279.54$ bits），`tau=1.5` 在 `K=3.2`（$281.00$ bits），`tau=2` 在 `K=2.6`（$280.27$ bits），配对的 `tau=4` 在 `K=1.8`（$278.92$ bits），与原 `tau=4` 图中 `K\approx1.7` 的峰一致到扫描分辨率。

因此，原始临界前沿峰的正确表述应收紧为：它是**中等有限观测时间（此处约 $\tau=4$）下**，在高 $K$ 同步吸引已压缩 whole-state 信息后出现的 whole-minus-sum 优势峰；它不是对所有预测时间窗都成立的、时间尺度无关的临界指标。短窗结果同时排除了一个较弱的替代解释：该峰并非仅由高 $K$ target 已完全同步所产生，因为在明确未强同步的 `tau=0.5,0.75` 条件下，曲线反而没有内部峰。

#### A.3.2 更长时间窗：峰位穿过而非收敛于理论 $K_c$

为直接检验“继续增大 $\tau$ 后，峰是否会停在临界相变点”的假设，保持同一配对 protocol、`N=64`、3 个 seeds 和 full-sample TM estimator，将时间窗扩展为 $\tau\in\{4,6,8,10,12\}$。扫描在转变区加密到 $K=0.8,0.9,\ldots,2.6$，并保留 $K=0,0.4,3.2,4.0$ 锚点，以区分内部峰和扫描端点峰。

![Long-horizon paired large-N Kuramoto sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_long_horizon_refined.png)

结果不支持单调收敛后固定在理论 $K_c=1.596$ 的解释。随着 $\tau$ 从 4 增至 12，$\Xi$ 的内部峰位依次为 $K_{\rm peak}=1.8,1.6,1.5,1.4,1.3$（峰值分别为 $278.92,274.52,272.83,271.65,269.61$ bits）。因此，`tau=6` 的 $K_{\rm peak}=1.6$ 只是在当前 $0.1$ 网格上恰好贴近 $K_c$；继续增加时间窗后，峰越过 $K_c$ 并持续移向更低的 $K$，而非在 $K_c$ 停留。所有这些峰都是加密区内部点，且其 $R_{0.99}$ 仅为 $0.644,0.561,0.523,0.492,0.492$，strong fraction 均为零；故该左移不是由峰落在高 $K$ 强同步 guard 失效区造成的。

更稳妥的结论是：$K_{\rm peak}(\tau)$ 是有限时间有效信息的时间尺度依赖 crossover，可能在某一中等时间窗掠过临界区，但不能把 $\tau\to\infty$ 的峰位等同于静态 Kuramoto 临界点。长窗极限还可能受相位混合和吸引子压缩控制；若要定义渐近临界指标，需要另行研究固定有限尺寸下的长时间衰减、再做 $N\to\infty$ 的有限尺寸标度，而不能从当前峰位外推。

#### A.3.3 相变前检测：共同早期弛豫窗中的 $\Xi(\tau)$ 谱

前述长窗峰位不能直接用作预警器。为检验能否在 future target 尚未同步时识别系统的**最终动力学区间**，对全部 $K\in[0,4]$ 保留同一短时间窗，而不是为高 $K$ 自适应延长或截短 horizon。已有的 `tau=0.5,0.75` 结果与新增的 $\tau\in\{0.1,0.2,0.3,0.4,0.6\}$ 配对合并，得到共同谱 $\tau\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.75\}$。所有 $(K,\tau)$ 条件都满足 $R_{0.99}<0.8$；即使在 $K=4$、$\tau=0.75$，$R_{0.99}\approx0.75$，因此该谱只观测初始相位分布向同步吸引子弛豫的早期，而没有把已同步 target 当作特征。

![Pre-transition Kuramoto Xi-tau phase detection](../../fig/classic_network_dynamics_benchmark/large_kuramoto_pretransition_phi_tau_phase_detection.png)

图 B 显示：超临界 $K>K_c$ 条件在整个共同早期窗内已有更陡、更高的 whole-state $\Xi(\tau)$ 谱，而此时图 A 证明其 target 尚未发生强同步。以已知的 $K_c=1.596$ 作为模拟中的超临界参考标签，只输入 7 个早期 $\Xi(\tau)$ 值，使用 leave-one-$K$-out（完整留出该 $K$ 的 3 个 seed）逻辑回归，得到超临界识别 AUROC 为 $0.983$。将每一条谱除以自身最大值、仅保留形状后，AUROC 仍为 $0.972$；因此区分力不只是 $\Xi$ 的整体幅度，时间尺度上的增长形状也携带信息。图 C 展示了留出 $K$ 后的预测概率。

##### A.3.3.1 识别算法与 AUROC 的计算

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

#### A.3.4 系统规模边界：只有大系统提供临界峰参照

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
| Pairwise BOLD-like $\Phi^R$ | `results/dmf_schaefer100/full/observational_phi_r.npz`、`results/dmf_schaefer100/full/observational_phi_r_summary.json` |
| 拓扑/层级分解 | `results/dmf_schaefer100/full/critical_topology.npz`、`results/dmf_schaefer100/full/topology_summary.json` |
| Yeo-7 分解 | `results/dmf_schaefer100/full/critical_yeo7.npz`、`results/dmf_schaefer100/full/yeo7_summary.json` |
| 汇总图 | `fig/dmf_schaefer100/dmf_schaefer100_summary_full.{png,svg,pdf}` |
| 加密耦合动力学诊断 | `results/dmf_schaefer100/critical_diagnostics/full/results.npz`、`results/dmf_schaefer100/critical_diagnostics/full/summary.json`、`fig/dmf_schaefer100/dmf_schaefer100_critical_diagnostics.{png,svg,pdf}` |
| 多预测时距扫描 | `results/dmf_schaefer100/multihorizon/full/results.npz`、`results/dmf_schaefer100/multihorizon/full/summary.json`、`fig/dmf_schaefer100/dmf_schaefer100_multihorizon_appendix.{png,svg,pdf}` |
| EI 分量附录图 | `fig/dmf_schaefer100/dmf_schaefer100_ei_components_appendix.{png,svg,pdf}` |
| Determinism/degeneracy 附录验证 | `results/dmf_schaefer100/full/detdeg_appendix_summary.json`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_appendix_raw.{png,svg,pdf}`、`fig/dmf_schaefer100/dmf_schaefer100_detdeg_kuramoto_shape.{png,svg,pdf}` |
| 结构保持 null 归因 | `results/dmf_schaefer100/structural_nulls/full/summary.json`、`results/dmf_schaefer100/structural_nulls/full/null_connectomes.npz`、`results/dmf_schaefer100/structural_nulls/full/{weight_shuffle,degree_strength,yeo_block}/main_confirmation.npz`、`fig/dmf_schaefer100/dmf_schaefer100_structural_nulls.{png,svg,pdf}` |
| 83/100 对照图 | `fig/dmf_schaefer100/dmf_83_vs_100_comparison.{png,svg,pdf}` |
| 实验契约与进度 | `docs/log/dmf_schaefer100_experiment_contract.md`、`docs/log/dmf_schaefer100_progress.json`、`docs/log/dmf_schaefer100_phi_r_contract.md`、`docs/log/dmf_schaefer100_phi_r_progress.json`、`docs/log/dmf_schaefer100_structural_null_contract.md`、`docs/log/dmf_schaefer100_structural_nulls_progress.json` |

完整流程可复现为：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_dmf_schaefer100_pipeline.py --mode full

python scripts/plot_dmf_schaefer100_detdeg_appendix.py

python scripts/run_dmf_schaefer100_critical_horizon.py --experiment critical --mode full

python scripts/run_dmf_schaefer100_critical_horizon.py --experiment horizon --mode full

python scripts/plot_dmf_schaefer100_critical_horizon.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u scripts/run_dmf_schaefer100_structural_nulls.py --mode full

python scripts/plot_dmf_schaefer100_structural_nulls.py
```

单线程 BLAS 用于避免小型线性代数在多线程调度下显著变慢，不改变统计定义或随机种子。

<a id="appendix-e"></a>

## 附录 E：HCP Schaefer-1000 的 57 人验证

### E.1 主结果相对原 29 人的变化

| 状态 | 29 人均值（bits） | 57 人均值（bits） | 57 人 REST−任务均值差 | BH $q$ |
|---|---:|---:|---:|---:|
| REST | 6.985 | 7.122 | — | — |
| EMOTION | 4.288 | 4.633 | 2.489 | $1.59\times10^{-7}$ |
| GAMBLING | 4.538 | 4.785 | 2.337 | $2.57\times10^{-7}$ |
| LANGUAGE | 4.702 | 5.150 | 1.972 | $4.97\times10^{-6}$ |
| MOTOR | 5.580 | 5.568 | 1.554 | $3.47\times10^{-5}$ |
| RELATIONAL | 5.160 | 5.357 | 1.765 | $4.97\times10^{-6}$ |
| SOCIAL | 5.686 | 6.243 | 0.879 | 0.0304 |
| WM | 5.028 | 5.323 | 1.799 | $4.97\times10^{-6}$ |

八状态均值排序完全不变。SOCIAL 的差距缩小且仍是最弱对比，但方向与校正后显著性保留。网络份额的最大绝对变化为 0.955 个百分点，说明主要网络重分配结论不依赖原 29 人样本。

### E.2 REST circular-shift null

null 在每名被试的七条 PC1 上分别施加独立、非零 circular shift，保留各网络自身的边际分布和自相关，同时破坏网络间同步；每个 null 都重新拟合 PCA 后的动力学模型。固定 $(p,\alpha)=(5,1)$，每人 20 次 null。

| 指标 | 57 人结果 |
|---|---:|
| observed $\Xi$ 均值 / 中位数 | 7.986 / 7.731 bits |
| observed−null-mean 均值 / 中位数 | 3.112 / 2.769 bits |
| observed 高于 null mean | 56/57 |
| 未校正经验 $p<0.05$ | 56/57 |

### E.3 模块核 matched-null

对 observed 和每个 null 都完整重跑 greedy top-3，随后在相同 replicate index 上跨 57 人汇总频率。这样检验的是“该模块核进入 top-3 的群体频率”，而不是对 observed 选出的 atom 值事后套用固定集合 null。

| 模块核 | observed 频率 | null 均值 | null 最大值 | 经验 $p$ |
|---|---:|---:|---:|---:|
| 全七网络 | 42 | 39.20 | 44 | 0.2381 |
| 缺 Limbic | 22 | 14.75 | 19 | 0.0476 |
| 缺 Limbic、Control | 19 | 4.55 | 9 | 0.0476 |
| 缺 SalVentAttn、Limbic | 15 | 6.20 | 11 | 0.0476 |

全七网络核的高频主要由组合规模本身解释；缺 Limbic 及更具体的广泛组合才显示超过 matched-null 的频率。由于只有 20 个 null 且候选不止一项，本表不作确认性多重比较声明。

### E.4 参数鲁棒性与预测误差

| $p$ | 最低留出误差对应 $\alpha$ | delta-NRMSE |
|---:|---:|---:|
| 1 | 10 | 0.8943 |
| 2 | 10 | 0.8608 |
| 3 | 10 | 0.8445 |
| 5 | 10 | **0.8350** |
| 8 | 100 | 0.8402 |

40 个“状态 $\times$ 阶数”单元中，$\alpha=10$ 在 28 个单元达到最低留出误差。网格层面，整体留出 NRMSE 与七项对比最小 Phi 边际的 Spearman 相关为 $-0.813$（$p=7.83\times10^{-7}$）：预测更差的网格通常也更不支持稳定的 REST 优势。该关系是模型诊断，不是预测误差对 Phi 偏差的因果证明。

### E.5 TEVF 与共同方差的分类结果

| 空间图 | 特征 | LOSO 准确率 | chance | 置换 $p$ |
|---|---|---:|---:|---:|
| 七任务 TEVF | 1000 parcels | 90.2% | 14.3% | 0.00050 |
| 七任务 TEVF | Yeo7 means | 65.4% | 14.3% | 0.00050 |
| REST+七任务方差富集 | 1000 parcels | 64.9% | 12.5% | 0.00050 |
| REST+七任务方差富集 | Yeo7 means | 34.6% | 12.5% | 0.00050 |

置换在每名被试内部独立打乱状态标签，并对每次置换完整重跑 LOSO；因此保持每名被试的整组空间图与跨被试结构。parcel 特征明显优于七网络均值，说明任务可辨识信息很大一部分位于 Yeo7 网络内部的细粒度空间模式。
