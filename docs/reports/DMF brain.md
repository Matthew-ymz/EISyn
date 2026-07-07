# Part 2: DMF 中的 $\Phi^{EID}$ 临界增强

这份报告回答两个问题。

第一，DMF 全脑动力学从低耦合进入高耦合时，$\Phi^{EID}$ 能不能识别出 firing-rate 快速变化的临界转变区。第二，如果临界区的 $\Phi^{EID}$ 确实升高，这个不可约信息来自少数简单 pair，还是来自跨模块的层级联合读出。

当前证据支持一个克制结论：在 continuation 扫描协议下，83 区 whole-state uniform 干预口径的 $\Phi^{EID}$ 稳定把临界增强定位在 $G\approx1.7\text{-}1.9$；模块级贪婪分解显示，这个增强不是单个网络对贡献，而是从 `DMN+Sub`、`DMN+FPN+Sub` 到更大跨模块集合的嵌套残差。这个结论不应写成采样协议无关的绝对相变常数，因为 independent restart 和取消逐 $G$ 标准化的敏感性分析会把峰值推向高耦合区。

![DMF 83 区 whole-state PhiEID 主结果](../../fig/dmf_83_region_oracle_phi_eid_main_g11.png)

*图 1. 83 区 Kuramoto-aligned whole-state $\Phi^{EID}$ 主结果。A：从 $G=1.1$ 开始展示的平均放电率；B：从 $G=1.1$ 开始展示的 uniform 干预 signed $\Phi^{EID}$。灰色带为 $G=1.7\text{-}1.9$。$G=1.0$ 不参与主结果峰值判定，只在附录 A 作为边界点审计。*

## 1. 数据从哪里来

实验近似复现 Mediano et al. (2025) Fig. 6 的全脑 Dynamic Mean Field (DMF) 设置。全局耦合参数扫描为

$$
G=1.0,1.1,\ldots,3.0.
$$

每个 $G$ 点模拟 83 个脑区的兴奋性门控变量 $s_E(t)$ 和抑制性门控变量 $s_I(t)$，并记录平均 firing rate。临界区不是先从 $\Phi^{EID}$ 定义出来的，而是先从 firing-rate 曲线看出来：平均放电率从 $G=1.6$ 的 $4.727\ \mathrm{Hz}$ 上升到 $G=1.9$ 的 $8.195\ \mathrm{Hz}$，离散斜率在 $G=1.8$ 附近最大，$G=1.8$ 与 $G=1.9$ 的斜率几乎相同。因此这里把 $G=1.7\text{-}1.9$ 作为快速转变带，而不是把某一个网格点写成精确相变常数。

耦合矩阵来自 F-TRACT `Lausanne2008-33.zip` 的第一个 `count` 矩阵。预处理步骤是：移除 `Unknown` 区域，保留 83 个 Lausanne 脑区，按最大值归一化，再乘以 0.2，得到代理耦合矩阵 $\mathbf{C}$。这个矩阵来自 SEEG 直接电刺激和 CCEP 的观测计数，不是 HCP diffusion-MRI 结构连接矩阵。因此，本报告适合支持方法验证、临界区识别和区域/模块分解原型，不能写成对原论文数值的严格复现。

## 2. 主结果一：临界相变的识别

主结果使用 83 区 whole-state uniform 干预口径。source 是 83 个 singleton region 的当前状态 $\{s_E^i(t)\}_{i=1}^{83}$，target 是下一步 whole-system 83D 状态 $\mathbf{s}_E(t+\tau)$。每个 $G$ 和 seed 下，都用当前 trace 的 source 均值与尺度把方差匹配的 uniform 最大熵干预映射回物理 $s_E$ 空间，再用 DMF 方程推进一步。最后把 source 和 target 标准化，使用 Gaussian block conditional total correlation 读取 signed raw $\Phi^{EID}$，不对负值做非负截断。

具体计算的是

$$
\Phi^{EID}
= EI_{\mathrm{do}}(\{s_E^i(t)\}_{i=1}^{83};\mathbf{s}_E(t+\tau))
-\sum_{i=1}^{83}EI_{\mathrm{do}}(s_E^i(t);\mathbf{s}_E(t+\tau)).
$$

这里的直觉很简单。低 $G$ 时，各脑区近似独立，联合读取 83 个源比逐个读取单源多不出太多不可约信息。高 $G$ 时，系统更接近同步或共同饱和，很多脑区携带相似信息，联合信息更多变成冗余。中间的转变区同时满足两个条件：动力学 regime 正在改变，单个脑区又不足以解释下一步全局状态，所以 whole-state 的不可约信息出现内部峰。

主结果如下。

| 证据 | 结果 | 读法 |
|---|---:|---|
| Firing-rate 快速上升段 | $G=1.7\text{-}1.9$ | 先用动力学曲线定义候选临界带 |
| 最大 firing-rate 离散斜率 | $G=1.8$，约 $13.060\ \mathrm{Hz}/G$ | 与 $G=1.9$ 的 $13.027$ 很接近，不写成精确单点 |
| 83 区 whole-state uniform 复验 | 8/8 seeds 峰值位于 $G=1.7$ | 主结果口径，稳定命中临界带 |
| Long-trace continuation 复验 | 8/8 seeds 峰值位于 $G=1.7$ | 支持原 continuation 协议下的鲁棒识别 |
| Independent restart 复验 | 0/8 seeds 峰值落入 $G=1.7\text{-}1.9$ | 说明结论依赖采样协议 |

这条证据链的关键点是顺序：先从 firing rate 确定转变带，再看 $\Phi^{EID}$ 是否在同一区域出现内部峰。结果显示，uniform 干预下 8 个 seed 的 whole-state $\Phi^{EID}$ 全部在 $G=1.7$ 达峰，clip fraction 为 0，说明峰值不是由边界裁剪制造的。其他干预设定只保留为缓存审计，主文不展开比较。

Long-trace continuation 进一步检查了短 trace 的不稳定性。把模拟长度从 `t_total=0.55` 增加到 `1.05` 后，8/8 条曲线的全局峰值都在 $G=1.7$，均值曲线也在 $G=1.7$ 达到最大值，$G=1.8$ 次高。短 trace 下少数低有效样本点会产生尖峰；过滤 `sample_count<300` 后，原短 trace continuation 结果同样有 8/8 个 seed 的峰值回到 $G=1.8$。

因此，稳妥表述是：在原 continuation 扫描协议且样本数足够时，83 区 whole-state $\Phi^{EID}$ 可以稳定识别 $G\approx1.7\text{-}1.9$ 的临界增强。不能更进一步写成“$\Phi^{EID}$ 是所有采样协议下的绝对相变指标”。Independent restart 敏感性分析中，8/8 个 seed 的峰值都落在 $G=2.7\text{-}3.0$，说明重启初值会改变轨迹分布，也会改变 $\Phi^{EID}$ 的峰值位置。

## 3. 尺度口径为什么重要

主结果使用“每个 $G$ 下独立做 source-scale matching 和 target z-scoring”的口径。这个处理有一个明确目的：比较的是同一 $G$ 附近的机制结构，而不是让跨 $G$ 的物理尺度变化直接支配 entropy 和 EI。

如果取消逐 $G$ 标准化，改为每个 seed 在整个 $G$ sweep 上共用一套 source 标准化参数，并在物理 $s_E$ 单位下计算 target entropy，那么 uniform 干预的峰值不再落在临界带。8/8 个 seed 都错过 $G=1.7\text{-}1.9$，median peak 移动到 $G=2.3$；均值曲线的最高点在 $G=2.1$，且 $G=2.0\text{-}2.5$ 一带整体偏高。

![DMF 83 区 no-per-G standardization PhiEID 对照](../../fig/dmf_83_region_oracle_no_g_standardization.png)

*图 2. 取消逐 $G$ 标准化后的 83 区 whole-state $\Phi^{EID}$ 对照。A：平均放电率；B：uniform 干预下的 signed $\Phi^{EID}$；C：8 个 seed 的峰值位置。灰色带仍为 $G=1.7\text{-}1.9$。该图是尺度敏感性审计，不作为主结果口径。*

对应的 determinism / degeneracy 分解也不能得到和 Kuramoto 一样的解释。Kuramoto 中，强同步区的 degeneracy 会急剧升高并抵消 determinism，因此 $\Phi^{EID}$ 在临界前沿形成峰值；而这个 DMF raw-scale 对照中，whole determinism 在临界区后继续升高，并在 $G\approx2.3$ 附近最高，whole degeneracy 只有小幅波动。也就是说，取消逐 $G$ 标准化后，曲线混入了跨 $G$ 的物理尺度和饱和程度变化，不能再把 $\Phi^{EID}$ 峰解释为临界区的机制协同增强。

![DMF 83 区 no-per-G standardization determinism and degeneracy](../../fig/dmf_83_region_oracle_no_g_standardization_detdeg.png)

*图 3. 取消逐 $G$ 标准化后的 determinism / degeneracy 分解。物理 $s_E$ 单位下 Gaussian differential entropy 的参考熵 $H_0$ 可为负；这里关注跨 $G$ 曲线形状，而不是绝对零点。*

## 4. 主结果二：贪婪层级分解

临界区识别告诉我们“什么时候”全脑不可约信息最高，但还没有回答“它由哪些源共同贡献”。直接在 83 个脑区上枚举所有二分不可行：根节点需要评估约 $2^{82}$ 个二分。因此报告用了两层分解。

第一层是模块级分解。先把 83 个 Lausanne 区域粗略映射到 7 个显示模块：

$$
\{\mathrm{DMN},\mathrm{Som},\mathrm{Vis},\mathrm{VAN},\mathrm{FPN},\mathrm{Lim},\mathrm{Sub}\}.
$$

对任意模块集合 $C$，定义模块级协同残差

$$
\Phi(C;Y)
= EI_{\mathrm{do}}(\mathbf{x}_C;Y)
-\sum_{i\in C}EI_{\mathrm{do}}(\mathbf{x}_i;Y),
$$

其中 $Y$ 是全脑下一时刻状态，$\mathbf{x}_i$ 是第 $i$ 个模块内所有当前脑区状态。计算使用标准化线性 Gaussian 转移；未选择的源维度视为独立最大熵干预下的边缘化背景。

贪婪二分的步骤是：

1. 从当前模块块 $C$ 开始，枚举所有非平凡二分 $C=L\cup R$。
2. 对每个二分，计算两个子块能解释的协同量 $\Phi(L;Y)+\Phi(R;Y)$。
3. 选择解释量最大的二分。
4. 把父块还不能被两个子块解释的非负差值记为当前层残差：

$$
\gamma(C\rightarrow L,R;Y)
=\Phi(C;Y)-\Phi(L;Y)-\Phi(R;Y).
$$

这个 $\gamma$ 是 greedy hierarchy 下的 residual atom，不是严格的 Möbius 纯阶原子。也就是说，它回答的是“沿这条贪婪二分树，哪些模块集合仍需要被联合读取”，而不是给出唯一的高阶信息分解。

![DMF 模块级 PhiEID 层级贪婪分解](../../fig/dmf_phi_eid_greedy_decomposition.png)

*图 4. DMF 模块级 $\Phi^{EID}$ 层级贪婪分解。A：模块级 $\Phi^{EID}$ 曲线；B：greedy residual 按模块阶数汇总后的比例；C：跨全部 $G$ 累积贡献最高的模块组合。*

模块级分解得到三点结论。

第一，模块级 $\Phi^{EID}$ 的峰值仍落在临界区附近。附录 TM 复验中，uniform 干预为 8/8 seeds 峰值命中 $G=1.7\text{-}1.9$，clip fraction 为 0。这说明模块级结果可以作为 source-side 机制审计，但它和 83 区 whole-state 主结果不是同一数值口径，不能直接比较绝对值。

第二，临界区 residual 不由某一个二阶 pair 独占。跨所有 $G$ 汇总后，order 2 到 order 7 都有正贡献，累积量分别约为 4.886、7.211、9.270、10.008、9.940 和 10.086。最高累积 residual 是根层 `DMN+Som+Vis+VAN+FPN+Lim+Sub`，其次是若干包含 `DMN`、`FPN`、`Lim`、`Sub` 的高阶组合。

第三，峰值 $G=1.8$ 处形成一条可读的嵌套链。前几个 residual atom 为：

| Atom | Order | Depth | Residual |
|---|---:|---:|---:|
| `DMN+Vis+VAN+FPN+Lim+Sub` | 6 | 1 | 1.502 |
| `DMN+Som+Vis+VAN+FPN+Lim+Sub` | 7 | 0 | 1.486 |
| `DMN+Vis+FPN+Lim+Sub` | 5 | 2 | 1.359 |
| `DMN+FPN+Lim+Sub` | 4 | 3 | 1.324 |
| `DMN+FPN+Sub` | 3 | 4 | 0.970 |
| `DMN+Sub` | 2 | 5 | 0.667 |

这条链的解释是：临界区的不可约信息不是“某两个模块发生了相互作用”这么简单。`DMN+Sub` 是最小的清晰组合之一，但它嵌在 `DMN+FPN+Sub`、`DMN+FPN+Lim+Sub` 以及更大跨模块集合里。换句话说，临界增强更像一个层级联合读出过程：局部组合有贡献，但最高的 residual 仍需要跨多个功能系统一起读。

第二层是 83 区预算受限局部二分，用来检查“如果不先压缩成模块，脑区级分解会不会自然给出清楚的小组合”。答案是否定的。无额外约束时，局部搜索会退化成 single-region vs rest：峰值 $G=1.8$ 处，前四层依次剥离 `Brain-Stem`、`Left-Accumbens-area`、`Left-Caudate`、`Left-Pallidum`。这说明当前目标函数偏向先剥离单点。

加入 `min-split-size=5` 后，峰值仍在 $G=1.8$，但 top residual 变成粗块二分。

![DMF 83-region PhiEID budgeted local split decomposition](../../fig/dmf_phi_eid_region_local_split_min5_decomposition.png)

*图 5. 83 区 $\Phi^{EID}$ 的预算受限局部二分搜索。A：83 区 singleton partition 的 $\Phi^{EID}$ 曲线；B：峰值点上预算搜索找到的前几层 split residual；C：top split residual。*

在 `split-search-budget=500`、`max-depth=3`、`min-split-size=5` 下，峰值 $G=1.8$ 的前四个 split residual 为：

| Depth | Split size | Residual |
|---:|---:|---:|
| 0 | 69 / 14 | 2.092 |
| 1 | 64 / 5 | 0.691 |
| 2 | 59 / 5 | 0.599 |
| 1 | 5 / 9 | 0.089 |

这个诊断支持模块合并的必要性：83 区组合空间不会自然压缩成少数稳定、可命名的 pair。如果不加入可解释先验，分解结果会偏向剥离链或难读的大块二分。因此，模块级 greedy hierarchy 是当前更适合讲机制故事的层级；83 区局部搜索更适合作为“为什么需要先做模块归纳”的审计。

## 5. 当前论文表述

最稳妥的表述是：

> 在 F-TRACT count-derived Lausanne-83 代理耦合矩阵上，DMF continuation 扫描显示平均 firing rate 在 $G\approx1.7\text{-}1.9$ 进入快速上升段。对齐 Kuramoto whole-state 口径后，uniform 干预下的 83 区 signed $\Phi^{EID}$ 在 8/8 个 seed 中于 $G=1.7$ 达到内部峰值；long-trace continuation 复验支持这一临界区识别。模块级 greedy hierarchy 进一步显示，临界区 $\Phi^{EID}$ 不是由单一 pair 主导，而是由包含 `DMN`、`FPN`、`Lim`、`Sub` 的嵌套跨模块联合读出贡献。该结论依赖 continuation 采样和逐 $G$ 标准化口径；independent restart 与 raw-scale 敏感性分析不支持把峰值写成采样协议无关的绝对相变指标。

## 6. 文件索引

| 文件 | 含义 |
|---|---|
| `fig/dmf_83_region_oracle_phi_eid_main_g11.{png,svg,pdf}` | 图 1，主结果曲线图；从 $G=1.1$ 开始展示 |
| `fig/dmf_83_region_oracle_no_g_standardization.{png,svg,pdf}` | 图 2，取消逐 $G$ 标准化后的 83 区 whole-state $\Phi^{EID}$ 对照 |
| `fig/dmf_83_region_oracle_no_g_standardization_detdeg.{png,svg,pdf}` | 图 3，取消逐 $G$ 标准化后的 determinism / degeneracy 分解 |
| `fig/dmf_phi_eid_greedy_decomposition.{png,svg,pdf,npz}` | 图 4 与数值缓存，模块级 greedy 分解 |
| `fig/dmf_phi_eid_region_local_split_min5_decomposition.{png,npz}` | 图 5 与数值缓存，83 区局部二分搜索 |
| `fig/part2_dmf_phi_comparison.{png,svg,pdf}` | 附录图 A1，uniform 干预口径下的 DMF 综合复现图 |
| `fig/dmf_phi_eid_robustness_longtrace.{png,svg,pdf}` | 附录图 A2，$\Phi^{EID}$ 多 seed 长 trace 鲁棒性验证图 |
| `fig/dmf_83_region_oracle_phi_eid_robustness.{png,svg,pdf}` | 附录图 A3，83 区 Kuramoto-aligned whole-state $\Phi^{EID}$ 完整扫描图 |
| `fig/dmf_module_tm_phi_eid_robustness.{png,svg,pdf}` | 附录图 B1，模块级 TM-$\Phi^{EID}$ uniform 干预复验图 |
| `results/dmf_83_region_oracle_phi_eid/` | 83 区 whole-state 复验缓存与 summary |
| `results/dmf_phi_eid_robustness_longtrace/` | 长 trace continuation 验证缓存、曲线表和峰值汇总 |
| `results/dmf_83_region_oracle_no_g_standardization/` | 取消逐 $G$ 标准化的 83 区 whole-state 对照缓存与 summary |
| `results/dmf_phi_eid_robustness/` | 短 trace continuation / independent restart 诊断缓存 |
| `results/dmf_module_tm_phi_eid/` | 模块级 TM-$\Phi^{EID}$ 复验缓存与 summary |
| `docs/log/dmf_83_region_oracle_phi_eid_robustness.md` | 83 区 Kuramoto-aligned whole-state 复验报告 |
| `docs/log/dmf_phi_eid_robustness_longtrace.md` | 长 trace 鲁棒性验证报告 |
| `docs/log/dmf_83_region_oracle_no_g_standardization.md` | 取消逐 $G$ 标准化的尺度敏感性审计报告 |
| `docs/log/dmf_module_tm_phi_eid_robustness.md` | 模块级 TM-$\Phi^{EID}$ 复验报告 |
| `scripts/validate_dmf_83_region_oracle_phi_eid.py` | 83 区 Kuramoto-aligned whole-state 复验脚本 |
| `scripts/validate_dmf_phi_eid_robustness.py` | 重跑 $\Phi^{EID}$ 多 seed 鲁棒性验证 |
| `scripts/validate_dmf_83_region_oracle_no_g_standardization.py` | 取消逐 $G$ 标准化的 83 区 whole-state 对照脚本 |
| `scripts/validate_dmf_module_tm_phi_eid.py` | 模块级真实 DMF 干预 + TM MI 复验脚本 |
| `scripts/plot_dmf_83_region_oracle_phi_eid_main_g11.py` | 从缓存生成 $G\geq1.1$ 的 83 区 whole-state 主结果图 |
| `scripts/plot_dmf_phi_eid_greedy_decomposition.py` | 重拟合 DMF lagged transition 并计算模块级 greedy 分解 |
| `scripts/plot_dmf_phi_eid_region_peel_decomposition.py` | 83 区 single-peel / budgeted local-split 分解脚本 |

## 7. 后续验证

正式论文图还需要四步增强：

1. 在 $G=1.7$、$G=1.8$ 和非临界对照点保存 transition matrix、残差协方差、标准化参数和 label，减少重复随机模拟。
2. 对模块级 greedy 分解做 bootstrap，检查 `DMN+FPN+Sub`、`DMN+Sub` 等嵌套 atom 是否稳定。
3. 明确论文中的 DMF 扫描协议：主结论使用 continuation；independent restart 只作为敏感性分析。
4. 如果后续扩展脑区级分解，需要沿用 uniform 干预口径，并先做独立鲁棒性验证。

## 附录 A. 边界点和鲁棒性审计

![DMF uniform 干预口径下的临界区 PhiEID 复现](../../fig/part2_dmf_phi_comparison.png)

*附录图 A1. Uniform 干预口径下的 DMF 综合复现。A：平均放电率；B：模块级 TM-$\Phi^{EID}$；C：83 区 Kuramoto-aligned whole-state signed $\Phi^{EID}$；D：两种口径在 8 个 seed 上识别出的峰值位置 $G^*$。灰色带为 $G=1.7\text{-}1.9$。该图用于附录审计，不作为主结果图。*

$G=1.0$ 的高 $\Phi$ 值不应解释为另一个相变，也不应简单写成算法缺陷。EI 衡量的是“当前干预状态能多可区分地预测未来状态”。在低耦合、低放电率边界，DMF 状态变化很小，$s_E(t)$ 到 $s_E(t+\tau)$ 接近自保持映射，残差小，因此 whole EI 和由它构成的 $\Phi^{EID}$ 都可能偏高。

换句话说，算法读到的是边界处的短时确定性，而不是临界敏感性。判断临界点时需要同时满足两个条件：峰值位于扫描内部，并且与 firing-rate 快速上升区一致。$G=1.0$ 是扫描左边界，没有对应动力学转变，所以只保留为 boundary audit，峰值识别排除它。

![DMF PhiEID robustness](../../fig/dmf_phi_eid_robustness_longtrace.png)

*附录图 A2. Long-trace continuation 鲁棒性验证。8/8 个 seed 的全局峰值都在 $G=1.7$，top-2 与 top-3 也全部命中 $G=1.7\text{-}1.9$。*

![DMF 83 区 whole-state PhiEID 鲁棒性复验](../../fig/dmf_83_region_oracle_phi_eid_robustness.png)

*附录图 A3. 83 区 Kuramoto-aligned whole-state $\Phi^{EID}$ 完整扫描复验。A：平均放电率；B：uniform 干预下的 83 区 signed $\Phi^{EID}$；C：uniform 干预在 8 个 seed 上识别出的峰值位置 $G^*$。该图包括 $G=1.0$ 边界点，因此只放在附录。*

## 附录 B. 模块级 TM 复验

模块级 TM 复验用于说明模块级口径也把峰值定位在临界区附近，但它不进入主结果。原因是它和 83 区 whole-state 口径不同：模块级结果把 83 个脑区压缩为 7 个粗模块，source 维度、干预映射和估计器都不同。因此，模块级结果只支持“临界区附近存在可解释的模块联合读出”这一机制审计，不直接改变主结果的证据层级。

模块级 TM 复验先把 83 个脑区压缩成 7 个非空粗模块的平均兴奋性门控变量 $s_E$。对每个模块维度施加独立方差匹配 uniform 干预

$$
U_i\sim \mathrm{Unif}[-\sqrt{3},\sqrt{3}],
$$

再映射回对应模块的物理 $s_E$ 均值，保留模块内 residual pattern 和背景 $s_I$，用真实 DMF 方程推进一步，最后用 transport-map MI 估计

$$
\Phi^{EID}
= I_{\mathrm{do}}(\mathbf{x}_t;\mathbf{x}_{t+1})
-\sum_i I_{\mathrm{do}}(x_t^i;\mathbf{x}_{t+1}).
$$

为避免只依赖线性 Gaussian 近似，模块级 TM 复验从同一条 long-trace continuation 中抽取背景 $s_I$ 与模块内 residual pattern，施加方差匹配 uniform 干预，再用真实 DMF 方程推进一步，并用 transport-map MI 估计 whole EI 和 singleton EI。`sample_count=4096` 时，clip fraction 为 0，说明峰值不是由边界裁剪制造的。8/8 个 seed 的峰值都落在 $G=1.7\text{-}1.9$。

![DMF 模块级 TM-PhiEID 鲁棒性复验](../../fig/dmf_module_tm_phi_eid_robustness.png)

*附录图 B1. 模块级 TM-$\Phi^{EID}$ 复验。该图用于说明模块级口径也把峰值定位在临界区附近，但不作为主结果图。*

## 附录 C. 83 区 single-peel 汇总

在 83 区 single-peel 分解中，$G=1.7$ aligned run 的 top-40 区域累计解释约 $59.9\%$ 的 peel residual。按粗模块汇总，`Sub` 的累计 residual 最高，约 $3.909$ bits，占总 peel residual 的 $17.97\%$；按模块内平均 residual，`DMN` 最高，约 $0.334$ bits / region，高于 `VAN` 的 $0.332$ 和 `Lim` 的 $0.327$。因此更稳妥的两层表述是：`Sub` 是总量最大的临界协同承载模块，`DMN` 的单区平均协同性质最强。

| Module | Top-40 residual sum (bits) | Mean residual (bits / region) | Share of total peel residual | Regions in top-40 |
|---|---:|---:|---:|---:|
| `Sub` | 3.909 | 0.326 | 17.97% | 12 |
| `Lim` | 2.614 | 0.327 | 12.02% | 8 |
| `Vis` | 1.623 | 0.325 | 7.46% | 5 |
| `DMN` | 1.335 | 0.334 | 6.14% | 4 |
| `VAN` | 1.327 | 0.332 | 6.10% | 4 |
| `Som` | 1.273 | 0.318 | 5.85% | 4 |
| `FPN` | 0.941 | 0.314 | 4.33% | 3 |
