# UniCM：O-information、WMS 与 PED synergy 能否加入 prior 对照

**优先考虑多源 target-augmented O-Shapley，以及双未来输出的 ΦWMS pair prior。PED-Hsx synergy 可以作为离散结构先验对照，但需要单独声明编码和指标语义。** 单目标、全阶 WMS 与当前 Ξ 的联盟函数相同，不能算作一个新的竞争定义。

本次完成定义核对、已有缓存检查和小型解析例子，没有新增这些候选的 UniCM 预测校准结果。当前图 A 已恢复保留方法分别选参，B、C 保留统一 α=30000、γ=3；两项已排除方法继续排除。

后续实测更新：用户已指定加入 O-Shapley 和双输出 ΦWMS，两项预测校准已完成，见[当前比较报告](unicm_information_prior_comparison.md)。下文保留此前考察时的定义与核对记录；PED仍未加入实测。

## 1. 已读取的证据

本地 Zotero API 可用。先检索并读取 PEID 条目，再核对相关论文全文；库内证据为：

| 本地 item key | 论文 | 本次证据 |
|---|---|---|
| MYATYWAJ | Partial Effective Information Decomposition for Synergistic Causality | PDF 全文，26/26 页；独立干预、EI 差值、条件总相关 |
| 966HXZNJ | Gradients of O-information highlight synergy and redundancy in physiological applications | PDF 全文，8/8 页；O-information 的增量归因 |
| WVS8LLKY | Partial entropy decomposition reveals higher-order information structures in human brain activity | 实质全文；Hsx、协同原子、离散化 |
| 3IYZCCSM | The Partial Entropy Decomposition: Decomposing multivariate entropy and mutual information via pointwise common surprisal | PDF 全文，31/31 页；Hcs 与部分熵的符号 |
| X34436WI | Measuring Integrated Information: Comparison of Candidate Measures in Theory and Simulation | PDF 全文，30/30 页；动态 whole-minus-sum 的分区定义 |

外部补充读取 [Rosas、Mediano、Gastpar（2024）](https://arxiv.org/html/2404.07140v1)的 RSI/O-information 关系，以及 [Stramaglia 等的 dynamic O-information](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.595736/full)。PED 的公开原文为 [Varley 等（2023）](https://doi.org/10.1073/pnas.2300888120)，基础 PED 定义见 [Ince（2017）](https://arxiv.org/abs/1702.01591)。部分 PMC 页面后续请求遇到访问检查，核对使用已取得的本地全文和作者公开页面。

## 2. WMS：单目标形式重复，动态双输出形式可以比较

令 $\mathbf h_m$ 表示源模态 $m$ 的 12 月完整历史，$Y_{j,\ell}$ 为冻结模型对目标 $j$、lead $\ell$ 的标量读出。信息量均在当前同一干预 channel 下估计，单位 bit。

单目标 whole-minus-sum 为

$$
W_{j,\ell}(S)=I(\mathbf h_S;Y_{j,\ell})-\sum_{m\in S}I(\mathbf h_m;Y_{j,\ell}).
\tag{2.1}
$$

式（2.1）与当前 Ξ-Shapley 的联盟函数逐项相同。在相同联盟、读出、干预分布与 EI 估计器下，对它做全阶 Shapley 会得到相同 attribution，继而得到相同惩罚和校准预测。这是定义等价，不需要另跑一轮证明优劣。如果只计算两源 WMS 再 half-edge 分配，改变的是联盟阶数截断；它属于低阶控制，不构成新的全阶指标。

经典动态 ΦWMS 则比较联合过去—联合未来信息与对应分块的过去—未来信息。接入当前 pair prior 时可以定义

$$
q^{W}_{ab,\ell}
=I(\mathbf h_a,\mathbf h_b;Y_{a,\ell},Y_{b,\ell})
-I(\mathbf h_a;Y_{a,\ell})-I(\mathbf h_b;Y_{b,\ell}).
\tag{2.2}
$$

它与当前 ΦR 缓存满足

$$
q^{W}_{ab,\ell}=q^{R}_{ab,\ell}
-\min_{r,s\in\{a,b\}} I(\mathbf h_r;Y_{s,\ell}).
\tag{2.3}
$$

式（2.3）可直接用缓存的 `phi_r_edges` 和 `singleton_ei` 求出，不需要重新调用 Transformer。按原接口将全部 55 条边各分一半给两个源，得到 lead-specific prior，再广播到各 target。这里采用每个源对的指定分区，没有执行全系统 minimum-information-partition 搜索或新增分区归一化。

**缓存核对结果：** 3 checkpoint × 24 lead × 55 对，共 3960 个式（2.2）的值，范围为 [0.001047542, 0.284084160] bit；声明容差 1e-8 bit，容差内归零数为 0，显著负值数为 0。ΦR 减去的修正项平均为 0.002544858 bit。因此候选确实不同于 ΦR，但预测表现仍需拟合后判断，不能由信息量大小推断。

一般观测分布的 ΦWMS 可以为负；当前独立源条件下，联合读出 MI 不小于两源分别对联合读出的 MI 之和，后者又不小于两个对应单目标 MI 之和，因此式（2.2）非负。若以后改成观测版本，应另行声明有符号 prior 的权重映射，不能沿用当前非负权重接口并静默截断。

## 3. O-information：推荐多源、包含目标的联盟函数

标准 O-information 是冗余与协同的净平衡，正值偏冗余、负值偏协同。它原本不区分源和目标。[Rosas 等（2019）](https://doi.org/10.1103/PhysRevE.100.032305)给出原始定义；多源定向指标与 O-information 的联系见上述 2024 年原文。

当前源块由独立干预生成，所以只对输入历史计算 $\Omega(\mathbf h_S)$ 会恒为零，不能产生预测 prior。对已有 channel 更合适的接入是“加入目标后的 O-information 增量”的相反数：

$$
v^{O}_{j,\ell}(S)
=-\left[\Omega(\mathbf h_S,Y_{j,\ell})-\Omega(\mathbf h_S)\right]
=(k-1)I(\mathbf h_S;Y_{j,\ell})
-\sum_{m\in S}I(\mathbf h_{S\setminus\{m\}};Y_{j,\ell}),
\quad k=|S|.
\tag{3.1}
$$

空联盟设为 0，单源联盟也为 0。这里每个模态的完整历史作为一个变量块，$k$ 按源块数量计算，而不是把全部历史标量重新视作独立玩家。对式（3.1）的全部 2048 个联盟做精确 Shapley，即可生成 target-and-lead-specific attribution，再接入原校准器。这是本实验构造的 **target-augmented O-Shapley 接口**，并非声称 O-information 原论文已定义了同样的校准方法。

**命题：** 对固定目标、固定全局联合分布及相互独立的源块，式（3.1）等于条件 dual total correlation：

$$
v^O(S)=\operatorname{DTC}(\mathbf h_S\mid Y)
-\operatorname{DTC}(\mathbf h_S)
=\operatorname{DTC}(\mathbf h_S\mid Y)\geq0.
\tag{3.2}
$$

式（3.2）由 DTC 的熵展开直接得到。对集合 $S$ 新增源 $r$，条件 DTC 的增量为

$$
\operatorname{DTC}(\mathbf h_{S\cup\{r\}}\mid Y)
-\operatorname{DTC}(\mathbf h_S\mid Y)
=\sum_{m\in S} I(\mathbf h_m;\mathbf h_r\mid
\mathbf h_{S\setminus\{m\}},Y)\geq0.
\tag{3.3}
$$

因此在当前独立源设定下，这个联盟函数单调，精确 Shapley 也非负，可接入现有非负惩罚接口。数值实现仍须声明 1e-8 bit 容差并审计，超阈值失败；一般有符号 O-information 不具有这些无条件结论。

当 $k=2$ 时，式（3.1）恰好等于式（2.1）的二源 WMS；因此不能把逐对 O-information 当作新的二源定义。$k\geq3$ 时，两者一般不同。小型精确检查使用三独立 fair bits、目标为三元 XOR：全阶 WMS 为 1 bit，式（3.1）为 2 bit，O 增量恒等式在全部非空联盟上的误差为 0。这也说明 O 分数不受目标 MI 的同一上界约束，不能解释为 PEID 总整合有效信息本身。

**计算可行性：** 式（3.1）只用联盟 MI，可沿用现有 affine degree-1 TM 和同一干预预测。无需离散化或换模型；归因预算仍是 2048 个联盟。原信息缓存没有保存全部 coalition EI，因此需要复用预测缓存重建联盟表，但不需要再次运行冻结 Transformer。

原论文的 **dynamic O-information** 还条件化目标历史。若采用该版本，应令目标自身历史作为条件、外部模态作为玩家，写清自预测 prior 的处理；不能直接把式（3.1）称为标准 dynamic O-information。这会改变自历史接口，适合作为另外明确命名的条件版本。

## 4. PED synergy：可作结构 prior，先确定具体 PED

PED 分解的是联合熵；PID/PEID 分解的是源关于目标的信息。将二者都称作 synergy 不意味着数值含义相同。PED 还需指定 shared-entropy 函数和哪些原子计入 synergy。[Varley 等（2023）](https://doi.org/10.1073/pnas.2300888120)使用离散 Hsx，并求非负 informative 原子；[Ince（2017）](https://arxiv.org/abs/1702.01591)的 Hcs 版本允许负的部分熵，不能混用两者的非负结论。

仓库已经有 [HCP 的三变量 PED-Hsx 实现](../../scripts/analyze_hcp_ped_oinfo_57.py)。它将每个标量序列二值化，在 18 节点熵冗余格反演，汇总作者采用的七个协同原子。一个可行的 prior 接口是：对每个源对和共同未来目标的离散变量三元组计算同一七原子和，再 half-edge 分给源。

不过完整源历史是 12 维向量。直接逐时间点二值化并将整个历史编码成一个类别，每个源就有最多 4096 个状态，两源加二值目标最多有 33554432 个联合状态；16384 个样本会非常稀疏。已有三标量二值实现不能原样处理这些历史块。若改用历史末值或固定的一维投影，则会丢掉部分历史，应明确这是压缩表示上的 prior。要比较指标本身，应在同一压缩、离散表示上同时重算其他 prior；只给 PED 换表示的结果属于整体方法对照。

**语义检查：** 对全部八种等概率二值三元状态，即三个相互独立变量，现有 Hsx 实现给出联合熵 3 bit、七原子 PED synergy=0.419505871 bit，最小原子=0.018251705 bit；而两源关于独立目标的 MI 为 0。由此可见，这个 PED synergy 不专门度量目标的预测协同。做 prior 对照是有意义的，但不能把它与 Ξ 按同一信息量语义解释。建议同时查看对目标打乱后的 prior 是否仍保留大部分结构，报告原始与目标打乱值；不预先相减或截断改变指标定义。

Hsx 需要离散概率事件，普通连续 MI 的 TM log-determinant 不能直接给出它的 shared-exclusion 原子。这里改用离散频率估计的理由是指标定义要求离散事件；代价是离散化损失和频数偏差。完整 11 源加目标的 PED 冗余格增长远快于 2048 个联盟，当前实用方案是局部三元 prior，不能称为全阶 PED。

## 5. 实验建议

| 候选 | 是否形成新的对照 | 接入路径 | 当前判断 |
|---|---|---|---|
| 单目标全阶 WMS-Shapley | 否，与 Ξ 相同 | 同一联盟函数 | 记录等价关系，不重复计数 |
| 双输出 ΦWMS pair prior | 是，但与 ΦR 密切相关 | 现有边缓存 → half-edge → lead-specific prior | 可直接拟合，先做 |
| 多源 target-augmented O-Shapley | 是，超过二源时区别出现 | TM coalition MI → O 增量 → 全阶 Shapley | 最适合作为新增的同口径对照 |
| PED-Hsx synergy pair prior | 是，但表示与语义需声明 | 固定离散三元组 → 七原子和 → half-edge | 适合补充结构 prior，先验证编码 |

后续新增对照应保留各自 validation 选参结果，并另给 α=30000、γ=3 的统一参数结果；与原方法使用相同 fit/test、44 维校准容量、共享 bootstrap。O-Shapley 和 ΦWMS 可以复用当前完整历史 channel。PED 应单独记录压缩、离散化、熵原子选择、源状态覆盖率及目标打乱诊断，避免同时改变多个因素后把误差变化归因于指标定义。

本次只做可行性考察；上表没有新增预测分数，也没有将候选加入当前排序。
