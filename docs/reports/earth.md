# 预测尺度依赖的高阶气候可预测性：观测场协同超边与 UniCM 层级有效信息

## 目录

- [1. 科学问题与主要发现](#1-科学问题与主要发现)
- [2. SLP 实验：协同超边如何随预测尺度重组](#2-slp-实验协同超边如何随预测尺度重组)
  - [2.1 实验设计与 Runge 基准](#21-实验设计与-runge-基准)
  - [2.2 从分散短期联系到可复现的中长期源组合](#22-从分散短期联系到可复现的中长期源组合)
  - [2.3 地球科学含义与可检验假设](#23-地球科学含义与可检验假设)
  - [2.4 尺度凝聚与地理扩张的综合图](#24-尺度凝聚与地理扩张的综合图)
- [3. UniCM 实验：冻结 Transformer 中的层级有效信息](#3-unicm-实验冻结-transformer-中的层级有效信息)
  - [3.1 可解释性分析口径](#31-可解释性分析口径)
  - [3.2 中期增强的系统级整合有效信息](#32-中期增强的系统级整合有效信息)
  - [3.3 ENSO 空间型态与 IOD 背景构成主导层级](#33-enso-空间型态与-iod-背景构成主导层级)
- [4. 综合讨论与解释边界](#4-综合讨论与解释边界)
- [5. 图表与数据索引](#5-图表与数据索引)
- [6. 参考文献](#6-参考文献)
- [附录 A. Runge 节点级指标对照](#附录-a-runge-节点级指标对照)
- [附录 B. 补充数值结果](#附录-b-补充数值结果)

## 1. 科学问题与主要发现

本文用两个彼此独立的实验检验同一科学问题：气候系统的可预测信息是否依赖多个空间模态的联合状态，以及这种高阶依赖如何随预测尺度变化。第一个实验直接分析 1948—2026 年全球海平面气压（SLP）分量，在 Runge 等人提出的因果网络基准上识别二源协同超边；第二个实验不再从观测场重新拟合动力模型，而是把已经训练完成的 UniCM Transformer 视为冻结的气候转移机制，对其进行最大熵干预和有效信息分解。

两组证据分别回答“观测场中出现了什么尺度依赖结构”和“神经气候模型依靠什么联合信息进行预测”。SLP 实验发现，短期强超边较分散，而中长期结果逐渐集中到以 `No.0/No.1` 为核心的少数源组合；该源对先快速建立近全球目标通道，随后通过招募更多目标区域继续扩展并趋于饱和。不同超边同时呈现早期峰值、中期峰值、长期平台和长期增强四类演化。UniCM 实验发现，系统级整合有效信息增量 $\Xi$ 并非在最短预测期最大，而是在 lead 8 达峰；层级分解进一步把这一中期增量定位到 ENSO 空间型态与 IOD 背景构成的二至五阶嵌套模块。

这两项实验使用不同数据对象和不同机制载体，不能相互替代，也不构成同一动力方程下的闭环验证。它们共同支持的窄结论是：气候可预测信息不仅存在于单模态记忆或成对联系中，还存在于依赖预测窗口的联合状态中。

## 2. SLP 实验：协同超边如何随预测尺度重组

### 2.1 实验设计与 Runge 基准

Runge 实验以全球 SLP 场的 60 个 Varimax 分量为节点。与静态节点排序不同，这里把两个源分量到一个目标分量的非加性有效信息增量作为二源协同超边：

$$
\Delta_{2,\mathrm{TM}}(i,j\rightarrow k)
=EI_{\mathrm{TM}}(\{i,j\}\rightarrow k)
-EI_{\mathrm{TM}}(i\rightarrow k)
-EI_{\mathrm{TM}}(j\rightarrow k).
$$

多步读出器、最大熵干预、三阶 transport-map（TM）估计和候选构造见 [Method.md 第 4 节](Method.md#4-runge-多尺度二源超边估计)。每个预测尺度 $H$ 均穷举 `102660` 个跨目标二源候选，从而避免先用离散互信息 shortlist 再做 TM 排名造成的覆盖偏差。正文关注三个层次：具体超边位于哪些区域、相同超边能否跨尺度复现、以及强度随 $H$ 呈现何种连续型态。静态 ACE/ACS 节点指标只作为补充对照保留在附录 A。

### 2.2 从分散短期联系到可复现的中长期源组合

图 1a—c 首先给出空间结构的变化。$H=1$ 时，前三条超边分别为 `No.0 + No.3 → No.37`、`No.0 + No.11 → No.35` 和 `No.1 + No.5 → No.17`，强联系分散在不同源—目标组合之间。到 $H=10$，前三条超边转为 `No.0 + No.1 → No.28`、`No.0 + No.1 → No.32` 和 `No.0 + No.6 → No.32`。在 $H=60$，前十几乎全部围绕 `No.0 + No.1` 展开。这一变化说明，中长期结构不是短期联系的等比例衰减，而是源组合本身发生了重组。

这种重组不是只在前十阈值下出现。图 1d 的“有效源对数”不是通过显著性检验的源对数量，也不对单个源对作“有效/无效”的二元判定；它是协同质量分布的熵等效数。对每个预测尺度 $H$，先按 $\Delta_{2,\mathrm{TM}}$ 从高到低取前 $K$ 条“源对 $\rightarrow$ 目标”超边。源对 $(i,j)$ 的正协同质量定义为

$$
w_{ij}^{(K)}(H)
=\sum_{k:\,(i,j\rightarrow k)\in\operatorname{TopK}(H)}
\max\!\left\{\Delta_{2,\mathrm{TM}}(i,j\rightarrow k;H),0\right\},
$$

即同一源对在 top-$K$ 内对不同目标的正协同增量之和；负值按零计。令

$$
p_{ij}^{(K)}(H)
=\frac{w_{ij}^{(K)}(H)}
{\sum_{a<b}w_{ab}^{(K)}(H)},
\qquad
N_{\mathrm{eff}}^{(K)}(H)
=\exp\!\left[
-\sum_{i<j:\,p_{ij}^{(K)}(H)>0}
p_{ij}^{(K)}(H)\log p_{ij}^{(K)}(H)
\right].
$$

若 top-$K$ 的正协同总质量为零，则该指标不定义。清晰的判读标准是：在固定 $K$ 下，若正协同质量由 $m$ 个源对完全均分，则 $N_{\mathrm{eff}}=m$；质量越集中在少数源对上，$N_{\mathrm{eff}}$ 越接近 1。因此，$N_{\mathrm{eff}}$ 随 $H$ 增大的总体下降表示头部协同质量发生凝聚，不能解释为其余源对已被统计检验判为无效。图中同时报告 $K=50,100,200,500$；只有总体下降方向在这些口径下保持一致，才把它视为不依赖单一 top-$K$ 截断的凝聚证据。以 top-200 为例，有效源对数从 $H=1$ 的 `159.0` 降至 $H=10$ 的 `75.8`、$H=20$ 的 `45.6` 和 $H=60$ 的 `18.3`。图 1e 进一步显示，`No.0 + No.1` 在 top-200 协同质量中的份额由 $H=1$ 的 `0.6%` 增至 $H=10$ 的 `11.4%`、$H=20$ 的 `18.5%` 和 $H=60$ 的 `25.3%`；从 $H=20$ 开始，全局前十均使用该源对。这里的结论是头部结构发生凝聚，而不是全部弱超边都收敛到同一源对。

连续强度曲线揭示了静态网络无法表达的时间结构（图 1g）。`No.0 + No.6 → No.32` 在 $H=4$ 达到早期峰值，随后持续下降；`No.0 + No.1 → No.28` 在 $H=7$ 达到中期峰值；`No.0 + No.1 → No.50` 从中期开始维持约 `0.012—0.014` bits 的长期平台；`No.0 + No.1 → No.46` 则从短期低值逐步增强，在 $H=60$ 达到 `0.018027` bits。因此，$H$ 不是统一的衰减参数，而是区分快速调整、阶段性耦合、持续背景态和累积传播候选的重要坐标。

附录 C 进一步固定全部干预样本、冻结 MLP rollout、候选全集和后处理，只把估计器从 Gaussian/affine TM 依次替换为二、三、四阶 TM。三个代表尺度的第一名超边在四种估计器下完全一致；第一名强度的跨估计器范围在 $H=1,10,60$ 分别只有 `0.008136—0.008558`、`0.017525—0.018130` 和 `0.018010—0.018320` bits。四条代表曲线相对三阶 TM 的 Pearson 相关均高于 `0.9988`，早期峰值、中期峰值和长期增强的峰位完全保持。由此，正文关于强超边量级和时间型态的结论不依赖三阶 TM；但十万级候选的全排序 Spearman 仅为 `0.281—0.713`，说明弱超边的精细次序仍对估计器敏感。

### 2.3 地球科学含义与可检验假设

SLP 实验把遥相关的比较单位从单条边扩展为“源组合—目标—预测窗口”三元组。同一个区域可以在短期不重要，却在另一个背景源共同存在时成为中长期信息通道；同一源组合也可以对不同目标表现为峰值、平台或持续增强。这种表示更适合描述依赖背景态的大气桥、Rossby 波列和海盆间耦合，而不是把它们压缩为固定的成对连接 [R2-R8]。

`No.0 + No.1` 的目标集合给出了比全局前十平均距离更明确的空间证据（图 1f）。固定全局 top-200 口径后，该源对在 $H=5,10,20,60$ 分别连接 `3/13/20/27` 个目标。每个目标用其分量载荷主中心定位，“最大目标跨度”定义为同一尺度全部目标中心两两球面大圆距离的最大值。该跨度从 $H=5$ 的 `13.91 × 10^3 km` 快速增至 $H=6$ 的 `18.18 × 10^3 km`，在 $H=15$ 接近 `19.92 × 10^3 km` 后基本饱和；与此同时，目标数仍从 $H=15$ 的 `16` 增至 $H=60$ 的 `27`。因此，更准确的空间图景是先快速建立近全球尺度的目标范围，再在既有最大跨度内继续招募和加密目标，而不是传播距离持续线性增长。

图 1 给出四个可检验假设。第一，若早期峰值来自快速大气调整，它应表现出更强的季节依赖，并在 block-bootstrap 中保持邻近尺度的一致性。第二，若中期峰值来自多区域相位关系向目标响应的转化，物理校准后的源区应同时满足稳定的空间载荷和时间先后关系。第三，若 `No.0 + No.1` 的目标扩张对应真实的跨区域传播，则“最大跨度先饱和、目标数后增长”的两阶段型态应在不同 top-$K$、目标中心定义和替代动力模型下保持。第四，长期平台与长期增强应对起始月份、推演误差和替代动力模型表现出不同敏感性。在完成这些检验前，本文只把超边称为候选机制，不将其等同于已确认的物理因果通道。

### 2.4 尺度凝聚与地理扩张的综合图

![SLP 协同超边的跨尺度重组](../../fig/earth_slp_hyperedge_dynamics.png)

*图 1. SLP 协同超边由分散的短期结构凝聚为具有广泛目标覆盖的中长期骨架。a—c，$H=1$、$H=10$ 和 $H=60$ 的全局 TM 前十超边；蓝色节点为源，绿色节点为仅作为目标出现的分量，紫色汇合点及箭头表示二源协同读出，线宽随 $\Delta_{2,\mathrm{TM}}$ 增大。d，top-50、100、200 和 500 中正协同质量分布的指数 Shannon 熵，即熵等效源对数；该量衡量头部质量由多少个等权源对构成，不是显著源对计数，四种口径均显示源对多样性下降。e，top-200 协同质量的源对组成，突出 `No.0 + No.1` 的增长；灰色为其余源对。f，`No.0 + No.1` 在 top-200 中的最大目标跨度和目标数；跨度为全部目标分量主中心之间的最大球面大圆距离，单位为 km。g，四条代表超边的强制 TM 重估，分别呈现早期峰值、中期峰值、长期平台和长期增强。所有尺度均来自完整候选集，而非离散 shortlist。*

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

*图 2. 冻结 UniCM Transformer 的系统级联合读出在中期增强，并可定位到 ENSO 空间型态与 IOD 背景构成的层级模块。a，11 个输入模态的地理区域及分析流程：12 个月模态历史输入冻结的 UniCM，最大熵干预后的 EI 与 $\Xi$ 再进入贪婪层级分解。b，$\Xi$ 的 checkpoint seed 均值和标准差，在 lead 8 达到 `0.184 ± 0.042` bits。c，按阶数汇总的 $\xi_C$；二至五阶单独显示，六至十一阶合并为高阶残差，黑线为原子总和，橙色虚线标记 lead 8。d，lead 8 的前八个原子及其源模态成员；误差线为 checkpoint seed 标准差，条形旁数字为该原子占总 $\Xi$ 的比例。*

### 3.2 中期增强的系统级整合有效信息

整体 EI 和单模态 EI 之和都随预测期增长而下降，但两者的差值并不单调。图 2b 中，$\Xi$ 在 lead 1—5 约为 `0.05—0.07` bits，随后在 lead 7—10 明显增强，并在 lead 8 达到 `0.183958 ± 0.042136` bits；lead 11—24 仍维持约 `0.09—0.15` bits。换言之，模型在短期可以较多依靠各模态自身记忆，而在中期更依赖多个模态的联合状态。

这一结论不等同于单个 lead 的精细排序已经稳定。ENSO 和 IOD 的整体 EI 曲线在 checkpoint 之间保持相似总体形状，但 lead 排序未通过全部 seed 鲁棒性标准。图 2b 因而支持“中期联合增量增强”这一尺度级结论，不支持把相邻月份的微小差异解释为确定的物理跃迁。

### 3.3 ENSO 空间型态与 IOD 背景构成主导层级

层级分解在数值精度内与总 $\Xi$ 闭合，逐 seed/lead 的最大偏差约为 `7.9 × 10^{-5}` bits。图 2c 表明，lead 8 的主要质量来自二至五阶原子，分别约占总量的 `21%`、`22%`、`20%` 和 `18%`；六阶以上主要表现为较小的跨块残差。这说明中期联合增量不是由单一超高阶项垄断，而是由多个低至中阶模块共同构成。

图 2d 进一步定位了这些模块。lead 8 的最大原子为 `ENSO + IOD + nino12 + nino3 + nino4`，贡献 `0.032661` bits，占总 $\Xi$ 的 `17.8%`。排名靠前的其他原子包括 `ENSO + nino12 + nino3`、`ENSO + IOD + nino12 + nino3`、`ENSO + IOD + nino3` 以及 `nino12 + nino3`。共同结构不是“ENSO 加任意外部模态”，而是 ENSO 当前强度、赤道太平洋东西向型态和 IOD 背景的嵌套联合读出。

这一解释与 ENSO diversity 文献一致：单一 ENSO 指数不足以描述事件的空间型态、生命周期和演变路径 [1-5]。在 UniCM 中，nino3、nino4 和 nino12 更适合解释为 ENSO 内部空间结构的不同读数，而非 ENSO 之外的独立强迫。IOD 的出现则表明，印度洋背景可以参与调制模型对 ENSO 中期演变的读取。这里的“参与”指冻结模型中的信息依赖，不自动等同于已识别的真实动力因果方向。

## 4. 综合讨论与解释边界

两张主图形成一条由观测场到训练模型的递进证据链。图 1 表明，全球 SLP 中的二源协同超边具有明确的预测尺度结构：短期联系分散，中长期逐渐收敛到少数可复现源组合；主导源对同时招募更多、地理覆盖更广的目标，并呈现多种连续时间型态。图 2 表明，一个已经训练完成的气候 Transformer 也在中期更依赖多模态联合读出，而且这种联合信息可以分解到具有明确气候含义的 ENSO—IOD 嵌套模块。

这种呼应不应被写成两个实验已经相互验证。SLP 实验分析的是重新拟合的 60 个压力场分量，UniCM 实验分析的是预定义海气模态上的冻结神经网络；两者的变量、时间单位、动力载体和估计维度均不同。更稳妥的结论是，两种独立设置都显示：高阶可预测信息具有尺度选择性，并且需要以“联合状态”而非静态节点重要性来描述。

主要解释边界如下：

- SLP 的 60 个 Varimax 分量是在 1948—2026 扩展样本上重新拟合得到的，编号不是官方固定标签；在完成载荷物理校准前，不能把未校准节点直接命名为确定气候过程。
- SLP 全量穷举解决了 shortlist 覆盖偏差，但尚未进行 block-bootstrap 显著性筛选、季节分层和替代推演模型验证。
- 图 1g 的四条超边是事后选取的代表型态，用于说明时间响应的异质性，不代表全部候选的总体分布。
- UniCM 分析只针对 frozen checkpoint 的 learned mechanism，不使用 reanalysis 数据重新验证预测，也不做单个历史事件归因。
- UniCM 的高维 EI、$\Xi$ 和二源 Syn 使用 Gaussian log-det 估计；它适合机制筛查，但不等同于 transport-map PEID 的最终非线性分解。
- 贪婪 $\Xi$ 分解依赖层级路径和数值容差；原子集合不是唯一的高阶 PID 表示。

## 5. 图表与数据索引

- 两张论文主图的可复现脚本：`scripts/plot_earth_system_main_figures.py`
- 图 1 PNG/SVG/PDF：`fig/earth_slp_hyperedge_dynamics.{png,svg,pdf}`
- 图 1 的 `No.0 + No.1` 目标跨度与数量摘要：`fig/earth_slp_hyperedge_dynamics_summary.json`
- 图 2 PNG/SVG/PDF：`fig/earth_unicm_hierarchical_ei.{png,svg,pdf}`
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

节点级 ACE/ACS 与 Ridge+PEID 结果仅作为不同估计口径的补充诊断。它们把与节点相连的一阶边和显著二阶项压缩成静态分数，只能回答哪个 component 更接近 source 或 target hub，不能表达“源组合—目标—预测窗口”的尺度结构，因此不承担正文结论。

![Runge 节点级 ACE/ACS 与 Ridge+PEID 对照](../../fig/runge_ridge_peid_order1_vs_order2_ace_acs_1948_2026.png)

*图 A1. 节点级指标对照。a 为修正后的 Runge 2015 PC-stable ACE/ACS；b 为 Ridge+PEID 一阶 EI；c 为一阶 EI 加显著二阶协同。不同面板的估计对象和尺度并不等价，不宜比较绝对数值或把节点排名作为正文的主要证据。*

修正后的 Runge 方法中，ACE top-3 为 `No.1/0/16`，ACS top-3 为 `No.0/1/26`；Ridge+PEID 中，ACE top-5 为 `No.0/1/3/9/4`，ACS top-5 为 `No.10/3/26/0/1`。父节点口径修正后，`No.3` 在扩展样本中的排名由 ACE 第 5、ACS 第 3 降至 ACE 第 12、ACS 第 13，说明静态排名对因果图构建口径敏感。

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
