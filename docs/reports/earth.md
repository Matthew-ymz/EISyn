# 预测尺度依赖的高阶气候机制：Runge SLP 超边与 UniCM 系统级 \(\Phi^{\mathrm{EID}}\)

## 目录

- [1. 研究问题与主要发现](#1-研究问题与主要发现)
- [2. 方法与实验设计](#2-方法与实验设计)
  - [2.1 Runge SLP 数据与节点级基线](#21-runge-slp-数据与节点级基线)
  - [2.2 高阶信息指标](#22-高阶信息指标)
- [3. Runge：随预测尺度演化的高阶遥相关](#3-runge随预测尺度演化的高阶遥相关)
  - [3.1 ACE/ACS 节点基线](#31-aceacs-节点基线)
  - [3.2 短、中、长期超边截面](#32-短中长期超边截面)
  - [3.3 跨尺度复现与典型演化型](#33-跨尺度复现与典型演化型)
  - [3.4 地球科学启示](#34-地球科学启示)
  - [3.5 可检验的机制假设](#35-可检验的机制假设)
- [4. UniCM：系统级 PhiEID 的计算与分解](#4-unicm系统级-phieid-的计算与分解)
  - [4.1 干预口径与模态地理含义](#41-干预口径与模态地理含义)
    - [模态的地理含义](#模态的地理含义)
  - [4.2 EI 与单模态记忆基线](#42-ei-与单模态记忆基线)
    - [整体 EI：ENSO 与 IOD 的全历史读数](#整体-eienso-与-iod-的全历史读数)
    - [单源 EI](#单源-ei)
    - [全模态自身 EI：不同模态具有不同记忆尺度](#全模态自身-ei不同模态具有不同记忆尺度)
    - [ENSO 目标的二源 Syn：空间型态提供辅助证据](#enso-目标的二源-syn空间型态提供辅助证据)
  - [4.3 系统级 PhiEID 的中期增强](#43-系统级-phieid-的中期增强)
  - [4.4 PhiEID 的层级贪婪分解](#44-phieid-的层级贪婪分解)
    - [分解定义与闭合关系](#分解定义与闭合关系)
  - [4.5 二源 Syn 的辅助证据](#45-二源-syn-的辅助证据)
    - [IOD 目标：自身记忆与印度洋/ENSO 背景共同调制](#iod-目标自身记忆与印度洋enso-背景共同调制)
- [5. 综合讨论与解释边界](#5-综合讨论与解释边界)
  - [5.1 解释边界](#51-解释边界)
- [6. 图表与数据索引](#6-图表与数据索引)
- [7. 参考文献](#7-参考文献)

## 1. 研究问题与主要发现

本文围绕同一科学问题组织两组互补证据：气候系统的可预测信息是否依赖多个空间模态的联合状态，以及这种高阶依赖如何随预测尺度变化。Runge SLP 实验从全球海平面气压分量出发，识别不同预测尺度 \(H\) 下的二源超边；UniCM 实验则计算全部气候模态历史对未来系统状态的系统级联合增量 \(\Phi^{\mathrm{EID}}\)，并以贪婪层级分解定位其主要来源。ACE/ACS、整体 EI、单源 EI 和二源 Syn 均作为支撑主结论的基线，而非叙述终点。

结果给出两点相互呼应的发现。第一，Runge 超边并非随 \(H\) 等比例衰减：早期峰值、中期峰值、长期平台和长期增强四类尺度响应同时存在；中长期反复出现的超边更多围绕 `No.0/No.1` 展开，且空间跨度呈弱增大趋势。这为地球科学提供了一个区别于静态遥相关网络的新视角：同一组区域可能只在特定预报窗口形成不可加的联合影响。第二，UniCM 的系统级 \(\Phi^{\mathrm{EID}}\) 在 lead 8 达峰，且主要由 ENSO 空间型态与 IOD 背景构成的二至五阶嵌套模块解释；这表明中期可预测性不仅取决于单个指数的记忆，还取决于多个海盆状态能否被联合读取。

这些结论均是模型和当前估计口径下的机制证据，不等同于历史事件归因或已验证的物理因果链。尤其是 Runge 分量尚未全部完成物理命名，UniCM 的高维信息量采用 Gaussian log-det 筛查；因此本文明确区分观测结果、物理解释与待检验假设。

## 2. 方法与实验设计

### 2.1 Runge SLP 数据与节点级基线

这一组实验只展示 ACE 和 ACS，不再展示 AMCE。实验在同一套 1948—2026 年 NCEP 海平面气压（SLP）周尺度 Varimax 分量上，对比 Runge 等人 [R1] 的线性因果网关/易感性算法与 Ridge+PEID 的 Hyper-ACE/Hyper-ACS。模型以 60 维分量最近 4 周的状态预测下一周状态。

数据预处理删除 2 月 29 日，并依次去除 365 日历日气候均值、逐日历日标准差和线性趋势。随后将逐日场聚合为月场，在月场上重新拟合 60 个 Varimax 旋转 PCA 空间权重，再把这些权重投影回预处理后的逐日场，并以连续 7 日均值得到周尺度分量。该“月场拟合—逐日投影—周尺度聚合”过程对应 Runge 等人 [R1] 的降维操作。Ridge 读出模型在同一组周尺度分量上重新训练，共得到 `4074` 个有效滞后样本，正则参数为 `alpha=1000`。

Runge 面板先用 PC-stable 筛选父节点，再以线性结构方程模型估计跨滞后因果效应。早期复现曾把最终 MCI 的 `p_matrix` 误作父节点集合；修正为 PC-stable 父节点后，`No.3` 在扩展样本中的排名由 ACE 第 5、ACS 第 3 降至 ACE 第 12、ACS 第 13。这个变化说明，节点排名对因果图构建口径敏感，也构成后续解释的重要限制。

### 2.2 高阶信息指标

记矩阵 \(\mathbf{C}\) 的元素 \(C_{ij}\) 为源分量 \(i\) 到目标分量 \(j\) 的跨滞后最大绝对因果效应，则 Runge 原文口径下

$$
\mathrm{ACE}_{\mathrm{Runge}}(i)=\frac{1}{n-1}\sum_{j\ne i}C_{ij},
\qquad
\mathrm{ACS}_{\mathrm{Runge}}(i)=\frac{1}{n-1}\sum_{j\ne i}C_{ji},
\qquad n=60 . \tag{2.1}
$$

Ridge+PEID 面板使用同一套 1948-2026 component scores。PEID 候选设置为旧口径：`intervention_samples=4096`、`candidate_top_sources=14`、`candidate_target_topk=10`、`order_max=2`、`null_reps=20`、显著门槛 \(|z|\ge2\)。记 \(EI_{i\to j}\) 为一阶有效信息，\(Syn_{K\Rightarrow j}^{\mathrm{EID}}\) 为二源集合 \(K\) 对目标 \(j\) 的 EID 协同项：

$$
Syn_{K\Rightarrow j}^{\mathrm{EID}}
=
EI\bigl(\mathbf{x}_t^K\to X_{t+1}^{(j)}\bigr)
-\sum_{a\in K}EI\bigl(X_t^{(a)}\to X_{t+1}^{(j)}\bigr).
\tag{2.2}
$$

图中的 Hyper-ACE 和 Hyper-ACS 保留一阶 EI 基线，并只加入满足 \(|z_{K\Rightarrow j}|\ge2\) 的二阶协同项。由于二阶超边经过显著性筛选，二阶项不再和一阶边共用 \(n-1\) 作分母，而是按每个节点实际关联的显著二阶超边数量求平均。令

$$
\mathcal{H}^{\mathrm{out}}_2(i)
=
\{(K,j):\, i\in K,\ |K|=2,\ |z_{K\Rightarrow j}|\ge2\},
\qquad
\mathcal{H}^{\mathrm{in}}_2(i)
=
\{(K,j):\, j=i,\ |K|=2,\ |z_{K\Rightarrow i}|\ge2\}.
\tag{2.3}
$$

$$
\mathrm{Hyper\text{-}ACE}(i)=
\frac{1}{n-1}\sum_j|EI_{i\to j}|
+
\begin{cases}
\displaystyle
\frac{1}{|\mathcal{H}^{\mathrm{out}}_2(i)|}
\sum_{(K,j)\in\mathcal{H}^{\mathrm{out}}_2(i)}
\frac{|Syn_{K\Rightarrow j}^{\mathrm{EID}}|}{|K|},
& |\mathcal{H}^{\mathrm{out}}_2(i)|>0,\\
0,& |\mathcal{H}^{\mathrm{out}}_2(i)|=0,
\end{cases} \tag{2.4}
$$

$$
\mathrm{Hyper\text{-}ACS}(i)=
\frac{1}{n-1}\sum_s|EI_{s\to i}|
+
\begin{cases}
\displaystyle
\frac{1}{|\mathcal{H}^{\mathrm{in}}_2(i)|}
\sum_{(K,j)\in\mathcal{H}^{\mathrm{in}}_2(i)}
|Syn_{K\Rightarrow i}^{\mathrm{EID}}|,
& |\mathcal{H}^{\mathrm{in}}_2(i)|>0,\\
0,& |\mathcal{H}^{\mathrm{in}}_2(i)|=0.
\end{cases} \tag{2.5}
$$

式（2.1）给出 Runge 的节点级基线，式（2.2）定义二源 EID 协同，式（2.3）—（2.5）则把显著二阶超边聚合到 Hyper-ACE/Hyper-ACS。两个 Hyper 指标只汇总一步预测中的直接一阶边和显著二阶超边，不计算超边影响节点后再沿因果图传播的高阶路径中心性。当前读数包含 `1638` 条二阶候选，其中 `287` 条通过 \(|z|\ge2\) 门槛。因此，二阶项修正的是显著局部超边的平均强度，而不是全部潜在超边的总体强度。

## 3. Runge：随预测尺度演化的高阶遥相关

### 3.1 ACE/ACS 节点基线

![Runge ACE/ACS 与 Ridge+PEID 节点基线](../../fig/runge_ridge_peid_order1_vs_order2_ace_acs_1948_2026.png)

*图 1. 节点级基线表明，一阶因果枢纽与显著二阶超边修正共同突出 `No.0/No.1`，但两种估计口径的排名并不等价。a 为修正后的 Runge 2015 PC-stable ACE/ACS；b 为 Ridge+PEID 一阶 EI；c 为一阶 EI 加显著二阶协同。外圈表示 ACE 或 Hyper-ACE，内圈表示 ACS 或 Hyper-ACS；a 使用线性 SEM 尺度，b/c 使用共同截断色标。加入二阶项后，ACE 前五为 `No.0/1/3/9/4`，ACS 前五为 `No.10/3/26/0/1`。*

修正后的 Runge 方法 ACE top-3 是 `No.1/0/16`，ACS top-3 是 `No.0/1/26`；Ridge+PEID 的 ACE top-5 是 `No.0/1/3/9/4`，ACS top-5 是 `No.10/3/26/0/1`。需要保留两个限制：第一，修正后的 PC-stable graph 仍不等于原文 Fig. 4 的逐项复刻；第二，60 个 Varimax component 的编号不是官方固定标签，当前只对少数论文讨论节点做了视觉校准，因此不能把低排名或未校准节点直接命名为确定气候过程。

上面的 ACE/ACS 地图把每个节点相关的一阶边和显著二阶超边都压缩成节点分数，因此只能回答“哪个 component 更像 source 或 target hub”。为了检查二阶项本身落在什么地理关系上，下面把视角从节点分数退回到具体超边。

### 3.2 短、中、长期超边截面

多步推演继续使用同一套 1948—2026 年周尺度分量和 `4074` 个有效滞后样本。验证集选择的读出器将 MLP 与 Ridge 转移模型加权组合，其中 `ridge_alpha=1000`，Ridge 和 MLP 的权重分别为 `0.37` 与 `0.63`。输入文件、结果清单和复现目录统一列于第 6 章，避免将内部路径混入结果叙事。

初步实验曾先用离散化 MI 筛选每个 \(H\) 的前 1000 条候选，再逐条重算三阶 TM MI。由于离散排序与 TM 排序差异较大，最终实验改为对每个报告尺度的全部 `102660` 条跨目标二源候选进行三阶 TM 穷举。批量实现严格复现原逐条估计器，并复用单源 EI、源对边缘密度和多项式设计矩阵；16 个尺度的 legacy 抽样最大误差均低于 \(10^{-14}\)。每条候选的非加性增量定义为

$$
\Delta_{2,\mathrm{TM}}^{[h]}(i,j\Rightarrow t)
=EI_{\{i,j\}\to t}^{[h]}-EI_{i\to t}^{[h]}-EI_{j\to t}^{[h]} . \tag{3.1}
$$

图 2—4 展示全量 TM 穷举在 \(H=1\)、\(H=10\) 和 \(H=60\) 的代表性横截面。每张图均从该尺度全部 `102660` 条候选中取全局前十；两个源节点汇入紫色中介点，再由中介点指向目标，浅灰点提供其余分量的空间参照。离散 shortlist 只在图 5 中作为覆盖偏差的诊断基准。

![H=1 的全量 TM 二阶超边](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H001_tm_exhaustive.png)

*图 2. 一周尺度的全局 TM 前十较为分散，尚未形成单一主导的源组合。全部 `102660` 条候选中的前三为 `No.0 + No.3 -> No.37`、`No.0 + No.11 -> No.35` 和 `No.1 + No.5 -> No.17`；排序和线宽使用式（3.1）。*

图 2 表明，短期联合信号在全量候选中仍较分散，因此“一周最强候选”不能直接等同于稳健物理遥相关。现有结果尚无 block-bootstrap 显著性检验，不能据此量化排名不确定性。

![H=10 的全量 TM 二阶超边](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H010_tm_exhaustive.png)

*图 3. 十周尺度的全局 TM 前十开始集中于 `No.0/No.1` 及其邻近组合。前三为 `No.0 + No.1 -> No.28`、`No.0 + No.1 -> No.32` 和 `No.0 + No.6 -> No.32`；全量排序见表 1。*

图 3 将 ACE/ACS 枢纽与具体二源组合联系起来，表明节点重要性可能来自条件于另一源区的联合影响。由于分量编号尚未全部物理校准，不能据此直接命名确定的大气过程。

![H=60 的全量 TM 二阶超边](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H060_tm_exhaustive.png)

*图 4. 六十周尺度的全局 TM 前十几乎全部围绕 `No.0 + No.1` 展开。前三为 `No.0 + No.1 -> No.46`、`No.0 + No.1 -> No.30` 和 `No.0 + No.1 -> No.50`。*

图 4 与慢耦合或跨区域传播累积的解释相容，但尚不能确认具体物理通道。全量穷举解决了候选覆盖问题，却没有替代 block-bootstrap、季节分层和物理校准。

表 1 汇总图 2—4 的全量三阶 TM 读数：\(H=1\) 的全局前三彼此分散，而 \(H=10\) 和 \(H=60\) 的强边逐渐收敛到 `No.0/No.1` 组合及其目标集合。

| 预测尺度 \(H\) | 全量 TM 排名 | 超边 | \(\Delta_{2,\mathrm{TM}}\) | 联合 EI | 单源 EI 之和 |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | `0+3->37` | 0.008207 | 0.161648 | 0.153441 |
| 1 | 2 | `0+11->35` | 0.006698 | 0.117270 | 0.110573 |
| 1 | 3 | `1+5->17` | 0.005681 | 0.147893 | 0.142213 |
| 1 | 4 | `0+12->37` | 0.005568 | 0.137311 | 0.131743 |
| 1 | 5 | `15+48->2` | 0.005274 | 0.100081 | 0.094807 |
| 10 | 1 | `0+1->28` | 0.017747 | 0.228734 | 0.210987 |
| 10 | 2 | `0+1->32` | 0.012679 | 0.206514 | 0.193835 |
| 10 | 3 | `0+6->32` | 0.010952 | 0.184992 | 0.174040 |
| 10 | 4 | `0+1->50` | 0.010754 | 0.180583 | 0.169829 |
| 10 | 5 | `0+1->55` | 0.010648 | 0.178373 | 0.167724 |
| 60 | 1 | `0+1->46` | 0.018027 | 0.231307 | 0.213280 |
| 60 | 2 | `0+1->30` | 0.014308 | 0.221244 | 0.206936 |
| 60 | 3 | `0+1->50` | 0.013515 | 0.200558 | 0.187043 |
| 60 | 4 | `0+1->41` | 0.012916 | 0.195218 | 0.182302 |
| 60 | 5 | `0+1->34` | 0.012818 | 0.195943 | 0.183124 |

全量结果进一步削弱了离散首位候选的解释价值。\(H=1\) 的离散第 1 候选在全量 TM 中仅排第 `16165`；16 个尺度中，离散第 1 候选只有 \(H=7\) 和 \(H=60\) 同时是全量 TM 第 1。相反，\(H=10\) 和 \(H=60\) 的全局强边更集中于 `No.0/No.1`，与节点级 ACE/ACS 枢纽证据相互呼应。

### 3.3 跨尺度复现与典型演化型

全量穷举覆盖 \(H=1,2,\ldots,10,15,20,30,40,50,60\)。图 5a 检验离散 shortlist 对全局 TM 前十的覆盖率；图 5b 给出离散第 1 候选在全量 TM 中的实际排名；图 5c 比较 shortlist 与穷举能够找到的最大 \(\Delta_{2,\mathrm{TM}}\)；图 5d 汇总跨尺度反复进入全量前十的超边。

![全候选三阶 TM 与离散 shortlist 的跨尺度比较](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive_summary.png)

*图 5. 离散 top-1000 无法稳定覆盖全局 TM 强超边，而全量排名在中长期集中到少数 `No.0/No.1` 组合。a，每个尺度的全量 TM top-10 中有多少条曾进入离散 top-1000；b，离散第 1 候选在全量 TM 中的排名，虚线为 top-1000 边界；c，全量穷举与 shortlist 内可见的最大 \(\Delta_{2,\mathrm{TM}}\)；d，跨尺度反复进入全量 top-10 的超边及其尺度内排名。所有尺度均包含 `102660` 条候选，legacy 抽样误差低于 \(10^{-14}\)。*

离散 shortlist 对全量 top-10 的覆盖仅为 `1—5/10`，其中 9 个尺度漏掉 `7—9` 条；离散第 1 候选在多数尺度的全量 TM 排名低于 `1000`。这说明离散 MI 可用于粗略计算预算分配，却不适合作为最终 TM 排名的硬筛选器。shortlist 在若干尺度仍能找到接近全局最大值的候选，但这种“最大值接近”不能保证候选集合或跨尺度复现结构正确。

全量前十的跨尺度结构比原 shortlist 结果更集中。`No.0 + No.1 -> No.50` 在 `11` 个尺度进入前十；`No.0 + No.1 -> No.28`、`No.0 + No.6 -> No.32` 和 `No.0 + No.1 -> No.32` 均出现 `8` 次；`No.0 + No.1 -> No.41` 出现 `6` 次。短尺度 \(H=1,2\) 仍较分散，而从 \(H=7\) 开始，围绕 `No.0/No.1` 的组合逐步主导。这把节点级 source/target hub 证据推进为具有预测窗口的候选机制，但前十复现次数仍受排名阈值影响，不能解释为超边存在概率。

图 6 从全量结果中选取四条代表边，进一步展示它们的连续尺度型态：

![代表性超边的强制 TM 尺度趋势](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.png)

*图 6. 四条代表超边呈现早期峰值、中期峰值、长期平台和长期增强四种尺度响应。每个点均直接重算 \(EI_i\)、\(EI_j\) 和 \(EI_{ij}\)，不再受离散前 1000 初筛缺失点影响。*

图 6 说明 \(H\) 不是统一衰减参数，而是区分候选动力过程的重要坐标。其限制在于四条曲线是事后选取的代表边，只验证指定超边，不代表全部候选的总体分布。

强制 TM 后，原先由缺失点造成的断线消失，趋势也更清楚。`No.0 + No.6 -> No.32` 是早期峰值型：从 \(H=1,2\) 的 `0.000258/0.001482` 升到 \(H=4\) 的 `0.018717`，之后降到 \(H=15,30,60\) 的 `0.006064/0.003694/0.002987`。`No.0 + No.1 -> No.28` 是中期峰值型：\(H=7\) 达到 `0.020379`，随后降到 \(H=20,40,60\) 的 `0.013970/0.009142/0.007221`。`No.0 + No.1 -> No.50` 则是较平滑的长尺度平台型：从 \(H=5\) 的 `0.011576` 到 \(H=15,30,60\) 的 `0.012524/0.013154/0.013515`，变化幅度小但持续为正。`No.0 + No.1 -> No.46` 最接近长尺度增强型：短期 \(H=1..10\) 基本低于 `0.0025`，到 \(H=20,30,40,50,60\) 依次为 `0.004079/0.009712/0.014776/0.017874/0.018027`。

### 3.4 地球科学启示

按全量 TM top-10 重新统计后，短期 \(H\le5\) 的源—目标平均距离中位数为 `9.53e3 km`，长期 \(H\ge20\) 为 `10.36e3 km`；最远源—目标距离中位数从 `13.88e3 km` 增至 `15.16e3 km`，三节点最大跨度中位数从 `14.26e3 km` 增至 `15.16e3 km`。这继续支持“较长 horizon 更偏向大尺度、远程组合”的弱趋势，但并非严格单调，也不意味着短期候选都是局地联系。中期 \(6\le H\le15\) 与长期的源—源距离中位数均为 `12.86e3 km`，主要与 `No.0/No.1` 组合反复出现有关。

从地球科学角度看，超边随 \(H\) 的分化提供了三点新认识。其一，遥相关不应只被描述为固定的成对连接；二源组合的非加性贡献可以在特定时间窗形成或消失。其二，中长期反复出现的 `No.0/No.1` 组合把节点级枢纽证据推进到候选机制层面，提示关键区域的作用可能依赖共同背景态。其三，长期空间跨度的弱增加与大尺度传播或慢耦合过程相容，但现有距离统计并不单调，不能据此断言传播方向。下一步只有在完成分量物理校准、季节分层、PEID null 与 block-bootstrap 后，才能把这些候选与经典大气桥、Rossby 波列或海盆间耦合机制 [R2-R8] 逐一对应。

这种尺度依赖性还提示，遥相关机制的比较单位应从单条边扩展为“源组合—目标—预测窗口”三元组，从而避免把不同时间尺度的过程混为同一连接。

### 3.5 可检验的机制假设

不同尺度的超边排序把“某个区域是否重要”推进为“某组区域在什么时间窗内共同重要”。这一变化可导出三个不超出现有证据的检验假设。第一，若早期峰值型超边主要反映快速大气调整，那么它们应在季节分层后表现出更强的季节依赖，并对周尺度状态扰动更敏感；反之，若峰值主要来自候选筛查误差，则其排名在 block-bootstrap 中不会稳定。第二，中期峰值型超边在 \(H=4\) 至 \(H=10\) 集中增强，可能对应多个分量的相位关系逐步转化为目标区响应。该假设要求物理校准后的源区同时满足明确的空间载荷结构和稳定的时间先后关系，不能只凭超边强度成立。

第三，长期平台型与长期增强型超边提供了区分“持续背景态”和“累积传播”的线索。`No.0 + No.1 -> No.50` 在长尺度保持近似平台，而 `No.0 + No.1 -> No.46` 从短期低值逐步增强；如果二者对应不同动力过程，那么前者应对起始月份更稳定，后者则应对推演误差累积和模型结构更敏感。当前结果尚未检验这些预期，因此它们只能作为后续实验设计，而不能写成已确认的海气通道。

这些假设也说明为什么单一静态网络不足以概括地球系统中的高阶联系。静态边会把短期峰值和长期增强压缩为一个平均强度，从而掩盖机制出现的时间窗口。以 \(H\) 为坐标的超边谱则允许分别检验季节选择性、跨尺度复现、空间跨度和估计器稳健性。只有当一条候选同时通过物理命名、跨 seed/季节复现、TM/PEID null 与 block-bootstrap，并在替代推演模型中保持相近的尺度型态时，才有足够证据进一步讨论大气桥、Rossby 波列或海盆间耦合 [R2-R8]。

进一步说，尺度型态本身可以作为机制筛选条件，而不只是结果展示方式。若一条超边仅在单个 \(H\) 进入前十、在相邻尺度迅速消失，则应优先检验估计方差和排名阈值；若同一源组合在连续尺度保持且强度平滑变化，则更适合进入物理诊断。这个判据并不要求所有真实过程都呈平滑曲线，而是要求强物理解释同时得到邻近尺度证据支持。现有结果中，`No.0 + No.1 -> No.28` 的中期连续出现和 `No.0 + No.1 -> No.46` 的长期增强，比 \(H=1\) 的孤立最高边更符合这一筛选逻辑。

超边也改变了背景态的表达方式。在成对网络中，第二个源通常只能作为额外边或控制变量出现；在二源超边中，两个源共同构成同一次干预的条件，可以直接检验“一个区域的作用是否依赖另一区域的状态”。这对海盆间耦合尤其重要，因为相同局地异常可能在不同远程背景下产生不同后续响应。不过，当前二源增量仍是预测模型读出上的统计量，只有结合载荷空间结构、时间先后和独立资料验证，才能把条件依赖提升为具体动力机制。

因此，Runge 实验的主要贡献不是给出一张新的静态遥相关图，而是把候选高阶联系组织成可比较、可证伪的尺度谱。该尺度谱为后续物理诊断明确了应优先验证的时间窗口和区域组合。

## 4. UniCM：系统级 PhiEID 的计算与分解

### 4.1 干预口径与模态地理含义

这里分析的是 frozen UniCM Modeformer learned mechanism，不是 reanalysis 预测技能评估，也不是单个历史事件归因。每个干预样本同时采样 12 个历史月份和 11 个 UniCM mode 维度，形成 `(B, 12, 11)` 的 bounded uniform 最大熵输入，历史张量写入 Modeformer encoder 的 12 个月历史段，未来 24 个月由 decoder 自回归生成。

核心配置如下：

| 项目 | 取值 |
|---|---|
| 检查点随机种子 | `1, 2, 3` |
| 干预样本数 | `8192` |
| 干预支持集 | 12 个历史月份 × 11 个模态维度，各维独立采样于 `[-4, 4]` |
| 采样随机种子 | `20260619` |
| 起始月份 | `0` |
| bootstrap 重复数 | ENSO 汇总为 `200`；IOD 源对曲线仅报告 seed 均值 |
| 目标模态 | 图 8、10、12 为 ENSO；图 11、13—15 为全部模态；图 9、16—17 包含 IOD |
| 源模态 | ENSO/nino、NPMM、SPMM、IOB、IOD、SIOD、TNA、nino12、nino3、nino4、WWV |

整体 EI 使用 flattened full-history source，即 132 维历史 mode 输入，对每个 lead 的目标 mode 输出估计 `EI(history; target_lead)`。高维整体读数采用 Gaussian log-det MI 作为快速筛查口径；它用于检查绝对量级和 seed 稳定性，不等同于最终的非线性 transport-map PEID 分解。本文保留二源 Syn 读数：

$$
\mathrm{Syn}_{ij}=EI_{ij}-EI_i-EI_j. \tag{4.1}
$$

式（4.1）中，`EI_i` 和 `EI_j` 是两个源模态的 12 个月历史分别到同一目标 lead 的单源 EI，`EI_{ij}` 是二者联合历史到该目标的 EI。所有读数都使用 Gaussian log-det 估计，适合作为全历史机制筛查；它们不等同于最终的非线性 transport-map PEID 分解。

#### 模态的地理含义

![UniCM 模态的地理区域](../../fig/unicm_mode_geography.png)

*图 7. 模态定义表明，后续高阶项同时刻画 ENSO 内部空间型态与跨海盆背景。ENSO 相关指数来自赤道太平洋不同经向区段；NPMM、SPMM 和 TNA 表征太平洋经向模态与热带北大西洋背景；IOD、SIOD 和 IOB 表征印度洋盆地及偶极型 SST 结构。*

这张图是解释后续 EI/Syn 的基础。`nino3`、`nino4` 和 `nino12` 不是 ENSO 之外的独立外部强迫，而是赤道太平洋内部空间结构的不同读数。因此当 `ENSO + nino3` 或 `ENSO + nino4` 出现高 Syn 时，更自然的解释是 ENSO 的当前强度需要和东西向 SST 型态一起读，才能判断未来几个月的演变。

图 7 只给出指数对应的区域范围，不包含模态载荷的时变结构或区域内空间异质性，因此它用于约束物理解释，而不能单独证明模态之间存在动力联系。

### 4.2 EI 与单模态记忆基线

#### 整体 EI：ENSO 与 IOD 的全历史读数

全历史整体 EI 的主窗口为 lead `1..24`，气候相关补充窗口为 lead `6..18`。跨 seed 鲁棒性标准为：seed 两两 Pearson 相关系数不低于 `0.80`，Spearman 相关系数不低于 `0.75`，EI 最高的三个 lead 至少重合 `2` 个。按此标准，ENSO/nino 和 IOD 的曲线形状具有一定一致性，但 lead 排序未通过鲁棒性检验。

| 目标 | 平均 EI（1—24） | 平均 EI（6—18） | 最小 Pearson | 最小 Spearman | 前三 lead 最小重合数 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| ENSO/nino | 0.617162 | 0.395603 | 0.950 | 0.482 | 3 | 不稳定 |
| IOD | 0.535641 | 0.467182 | 0.854 | 0.245 | 2 | 不稳定 |

![ENSO 整体 EI 的预测期曲线](../../fig/unicm_enso_overall_ei_seed_overlay.png)

*图 8. ENSO 的整体 EI 主要集中在前 1—6 个月，但具体 lead 排序未通过全部 seed 鲁棒性标准。彩色细线为各 checkpoint seed，黑线为 seed 均值，阴影为 seed 标准差。*

![各目标模态的全历史整体 EI](../../results/unicm_overall_ei_tm_degree1_n8192/fig/overall_ei_seed_overlay.png)

*图 9. 各目标的曲线形状具有一定 seed 一致性，但 lead 细粒度排序普遍比总体衰减趋势更不稳定。图中每个面板对应一个目标模态，每条曲线对应一个 checkpoint seed。*

图 8—9 支持“全历史输入含有短中期机制信息”，其意义是为后续 \(\Phi^{\mathrm{EID}}\) 提供总信息量参照；但 ENSO 与 IOD 均未通过 lead 排序鲁棒性标准，不能过度解释单月峰谷。

图 8—9 表明，UniCM 学到的 ENSO/nino 有效信息主要集中在 lead 1—6 个月。短期 EI 明显高于后期，符合 ENSO 短期记忆较强、长期不确定性上升的物理直觉。三个 checkpoint 的曲线形状相近，最小 Pearson 相关系数为 `0.950`；但最小 Spearman 相关系数只有 `0.482`，说明具体 lead 排序仍不稳定。IOD 在 `6..18` 窗口的平均 EI 更接近全窗口均值，但最小 Spearman 相关系数仅为 `0.245`。因此，整体 EI 只支持“全历史输入含有可读出的短中期机制信息”这一方向性判断，不能支撑单个 lead 的精细排序。

#### 单源 EI

| 目标 | 自身 EI | 最强非自身源 | NPMM EI | TNA EI |
|---|---:|---|---:|---:|
| ENSO | 0.473612 | nino12 0.015768; nino3 0.015599; IOD 0.013361; SPMM 0.012534 | 0.011671 | 0.005806 |

![ENSO 单源 EI 的预测期曲线](../../fig/unicm_enso_source_ei_rankings.png)

*图 10. ENSO 自身历史主导短期信息，非自身源仅在中长期提供较小补充。左图为 ENSO 自身源，右图为按 24 个月平均 EI 选出的前五个非自身源并保留 NPMM/TNA；实线和浅色带分别为 seed 均值和标准差。*

单源 EI 曲线显示，ENSO 自身历史在短 lead 占绝对主导，但随后快速衰减。排除自身后，`nino3`、`nino12` 和 IOD 的 EI 随 lead 增长并在较长 lead 位于前列，NPMM 则在中期达到较高水平后回落；这些长 lead 曲线的 checkpoint 波动也明显扩大，因此不宜过度解释精细排序。TNA 的曲线始终较低，更稳妥的说法是，它可能只在 ENSO 背景态或其他太平洋/印度洋模态共同存在时提供弱增量。

#### 全模态自身 EI：不同模态具有不同记忆尺度

为了和二源 Syn 的量级作对照，这里进一步把每个 mode 都作为 target，并只输入该 mode 自己的 12 个月历史，计算 self EI 随 lead 的变化。也就是说，每条曲线都对应 `source = target`，没有引入其他 mode 的历史。

![全部模态的自身 EI 曲线](../../fig/unicm_all_modes_self_ei_leads.png)

*图 11. 不同模态的自身记忆衰减显著不同，且其量级远高于二源 Syn。实线为 11 个模态的 checkpoint seed 均值，浅色带为标准差；纵轴为该模态 12 个月历史到未来状态的 EI。*

这张图说明，self EI 的绝对量级显著大于前面的二源 Syn：多数 mode 在 lead 1 都有约 `1.6-2.2` bits 的自身历史信息，而二源 Syn 通常只有 `10^{-3}` 到 `10^{-2}` bits。`NPMM`、`IOB`、`WWV`、`SPMM`、`TNA` 和 `SIOD` 的 self EI 衰减较慢，lead 12 仍约 `0.89-0.99` bits；相反，ENSO 相关的 `nino`、`nino3`、`nino4`、`nino12` 以及 `IOD` 在前 6 到 10 个月后快速下降，lead 24 基本接近 `0.05-0.08` bits。

因此，self EI 主要读到的是各模态状态本身的持久性和自回归记忆，不应直接拿它和二源 Syn 当作同一层面的机制强度比较。二源 Syn 更像是在“已经有各自单源信息之后，两个历史变量联合读数还能额外提供多少目标信息”；它小很多是预期内的结果，也解释了为什么在分析协同项时需要单独画 Syn 曲线，而不能只看总 EI 或 self EI。

#### ENSO 目标的二源 Syn：空间型态提供辅助证据

| 目标 | 源对 | 排名 | 平均 Syn（1—24） | Syn seed 标准差 | 95% CI | seed 排名范围 | 联合 EI | 左源 EI | 右源 EI |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| ENSO | ENSO + nino3 | 1 | 0.005216 | 0.000672 | [0.003545, 0.006886] | 1-3 | 0.494427 | 0.473612 | 0.015599 |
| ENSO | ENSO + nino4 | 2 | 0.005194 | 0.002359 | [-0.000666, 0.011054] | 1-4 | 0.489874 | 0.473612 | 0.011068 |
| ENSO | ENSO + SPMM | 3 | 0.004559 | 0.002518 | [-0.001697, 0.010815] | 2-5 | 0.490705 | 0.473612 | 0.012534 |
| ENSO | ENSO + IOD | 4 | 0.004278 | 0.004353 | [-0.006535, 0.015091] | 1-19 | 0.491251 | 0.473612 | 0.013361 |
| ENSO | ENSO + NPMM | 5 | 0.002686 | 0.002452 | [-0.003404, 0.008777] | 4-9 | 0.487969 | 0.473612 | 0.011671 |
| ENSO | ENSO + nino12 | 6 | 0.002589 | 0.001873 | [-0.002064, 0.007241] | 3-11 | 0.491968 | 0.473612 | 0.015768 |
| ENSO | ENSO + WWV | 7 | 0.001728 | - | - | - | 0.480132 | 0.473612 | 0.004792 |
| ENSO | ENSO + TNA | 8 | 0.001499 | 0.000294 | [0.000768, 0.002230] | 7-9 | 0.480917 | 0.473612 | 0.005806 |
| ENSO | nino12 + nino3 | 9 | 0.001359 | - | - | - | 0.032726 | 0.015768 | 0.015599 |
| ENSO | ENSO + IOB | 10 | 0.001179 | - | - | - | 0.480909 | 0.473612 | 0.006119 |
| ENSO | NPMM + TNA | 55 | -0.000139 | 0.000141 | [-0.000488, 0.000210] | 44-55 | 0.017338 | 0.011671 | 0.005806 |

![ENSO 目标的二源 Syn 曲线](../../fig/unicm_enso_mode_pair_syn_leads.png)

*图 12. ENSO 的短中期协同主要来自 ENSO 强度与 nino3/nino4 空间型态的联合读出。实线为各 lead 的 seed 均值，同色浅虚线为该源对在 lead 1—24 上的平均 Syn。*

这张图的核心信息很直接：模型不是只看“ENSO 现在有多强”，还在看“暖异常更偏东、偏中太平洋，还是和其他海盆背景态一起出现”。前 1 到 7 个月，`ENSO + nino3` 和 `ENSO + nino4` 的 Syn 明显更高，说明 ENSO 的短期未来演变对赤道太平洋东西向 SST 结构很敏感。同样强度的 ENSO，如果空间型态不同，后续几个月的增长、衰减和位相演变也可能不同。

这个解释和 ENSO diversity 文献一致。Trenberth and Stepaniak [1] 指出，单一 ENSO 指数不足以描述事件演变，需要额外刻画中东太平洋 SST 梯度；Capotondi et al. [2] 把事件间差异总结为 ENSO 的振幅、空间型态、生命周期和触发机制差异；Ren and Jin [3] 进一步用 Niño3/Niño4 组合区分两类 ENSO。Kao and Yu [4] 与 Ashok et al. [5] 则分别从 EP/CP ENSO 和 ENSO Modoki 角度说明，中太平洋型和东太平洋型事件不能简单当作同一种 ENSO 强度的线性放大。

因此，`nino3` 和 `nino4` 更适合被解释为 ENSO 内部空间型态的调制因子，而不是 ENSO 之外的独立强迫源。曲线在 9 到 12 个月后整体贴近零，说明这种额外协同信息主要集中在短中期；到更长 lead，模型已经很难从这些二源组合里读出稳定的增量。

### 4.3 系统级 PhiEID 的中期增强

以全部 11 个模态的未来状态为共同目标，计算系统级联合增量

$$
\Phi^{\mathrm{EID}}_{\ell}
= I(\mathbf{X}^{1:12}_{1:11};\mathbf{y}_{\ell}^{\mathrm{all}})
- \sum_{m=1}^{11} I(\mathbf{X}^{1:12}_{m};\mathbf{y}_{\ell}^{\mathrm{all}}). \tag{4.2}
$$

式（4.2）的源划分由 11 个单模态块组成，每个块包含该模态 12 个月的历史。实现不对单项 EI、\(\Phi^{\mathrm{EID}}\) 或分解残差施加非负截断；在当前结果中它们均保持理论预期的非负性。若未来运行出现负值，将作为估计或数值诊断直接报告，而不会改写为零。这一高维读数仍采用 Gaussian log-det 筛查，不等同于最终的非线性 transport-map PEID。

![全模态目标的系统级 PhiEID 曲线](../../fig/unicm_all_mode_target_phi_eid_leads.png)

*图 13. 系统级 \(\Phi^{\mathrm{EID}}\) 在 lead 8 达峰，而非在最短预测期最大。上图为 checkpoint seed 均值和标准差，下图比较整体 EI 与单模态 EI 之和；所有曲线均为未截断的 signed Gaussian log-det 读数。*

整体 EI 与单模态 EI 之和都随 lead 增长而下降，但两者差值并不单调。\(\Phi^{\mathrm{EID}}\) 在 lead 1—5 约为 `0.05-0.07` bits，随后在 lead 7—10 增强，并在 lead 8 达到 `0.183958 ± 0.042136` bits；lead 11—24 维持在约 `0.09-0.15` bits。系统级联合增量因而不是短期最大，而是在中期更明显。

这个结果和上面的二源 Syn 曲线一致：单源或单 pair 对整体未来状态的解释在短 lead 已经很强，但不可约的多模态联合增量主要出现在 6 到 10 个月附近。完整逐 seed 表见 `results/unicm_all_mode_target_phi_eid_cpu_bound4_n8192/all_mode_target_phi_eid_rows.csv`。

### 4.4 PhiEID 的层级贪婪分解

进一步对每个 lead 的全模态 \(\Phi^{\mathrm{EID}}\) 进行层级可加性分解。该分解不声称恢复唯一的高阶 PID 原子，而是回答一个更可读的问题：从全部模态出发，每一步尽量拆出可由两个子模块解释的部分后，还有哪些模态集合必须联合读取？

#### 分解定义与闭合关系

设全集为 $S=\{1,\ldots,11\}$，每个元素对应一个 UniCM 模态。对任意非空集合 $C\subseteq S$，令 $\mathbf{x}_C$ 表示集合内所有模态的 12 个月历史，$\mathbf{y}_{\ell}^{\mathrm{all}}$ 表示 lead $\ell$ 的全模态目标。集合的原始联合增量定义为

$$
\widetilde{\Phi}^{\mathrm{EID}}(C;\mathbf{y}_{\ell}^{\mathrm{all}})
= EI(\mathbf{x}_C;\mathbf{y}_{\ell}^{\mathrm{all}})
-\sum_{i\in C}EI(\mathbf{x}_i;\mathbf{y}_{\ell}^{\mathrm{all}}). \tag{4.3}
$$

式（4.3）先计算集合联合历史的 EI，再减去各单模态 EI 之和。正值表示联合读出包含单模态相加无法解释的信息；若出现负值，保留其 signed 数值作为估计或数值诊断。

现在把当前节点 $C$ 拆成两个互不重叠、并且并起来等于 $C$ 的子块：

$$
L\cap R=\varnothing,\qquad L\cup R=C,\qquad L\neq\varnothing,\qquad R\neq\varnothing. \tag{4.4}
$$

对每个候选二分 $(L,R)$，先计算两个子块已经能解释的协同量：

$$
B(L,R;\mathbf{y}_{\ell}^{\mathrm{all}})
=\widetilde{\Phi}^{\mathrm{EID}}(L;\mathbf{y}_{\ell}^{\mathrm{all}})
+\widetilde{\Phi}^{\mathrm{EID}}(R;\mathbf{y}_{\ell}^{\mathrm{all}}). \tag{4.5}
$$

对每个二分先计算原始残差 \(r(C;L,R)=\widetilde{\Phi}^{\mathrm{EID}}(C)-B(L,R)\)。实现只保留满足 \(r(C;L,R)\ge-\tau\) 的可容许二分，其中 `split_tolerance` \(\tau=10^{-4}\)。在可容许集合 \(\mathcal{A}_{\tau}(C)\) 中，贪婪步骤选择 \(B\) 最大的二分；若 \(B\) 近似相同，则选择残差更小者：

$$
(L^\star,R^\star)
=\underset{(L,R)\in\mathcal{A}_{\tau}(C)}{\arg\max}
\left[
\widetilde{\Phi}^{\mathrm{EID}}(L;\mathbf{y}_{\ell}^{\mathrm{all}})
+\widetilde{\Phi}^{\mathrm{EID}}(R;\mathbf{y}_{\ell}^{\mathrm{all}})
\right]. \tag{4.6}
$$

选定二分后，仅当残差超过 `eps` \(\varepsilon=10^{-5}\) 时，才把父块不能由两个子块解释的部分记录为残差原子：

$$
\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}})
=
\widetilde{\Phi}^{\mathrm{EID}}(C;\mathbf{y}_{\ell}^{\mathrm{all}})
-\widetilde{\Phi}^{\mathrm{EID}}(L^\star;\mathbf{y}_{\ell}^{\mathrm{all}})
-\widetilde{\Phi}^{\mathrm{EID}}(R^\star;\mathbf{y}_{\ell}^{\mathrm{all}}).
\tag{4.7}
$$

因为 $L^\star$ 和 $R^\star$ 正好二分 $C$，单源项会相互抵消，所以上式也可以写成更直接的 EI 差：

$$
\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}})
=
EI(\mathbf{x}_C;\mathbf{y}_{\ell}^{\mathrm{all}})
-EI(\mathbf{x}_{L^\star};\mathbf{y}_{\ell}^{\mathrm{all}})
-EI(\mathbf{x}_{R^\star};\mathbf{y}_{\ell}^{\mathrm{all}}).
\tag{4.8}
$$

如果 \(\gamma_C>\varepsilon\)，它表示分别联合读取两个子块后仍有信息只能由整个 \(C\) 读出。算法随后递归处理 \(L^\star\) 和 \(R^\star\)。当子块为 singleton、原始联合增量不超过 \(\varepsilon\)，或不存在捕获量超过 \(\varepsilon\) 的可容许二分时递归终止。最后一种情况没有对应的 \((L^\star,R^\star)\)，因此单独定义 terminal 原子

$$
\eta_C(\mathbf{y}_{\ell}^{\mathrm{all}})
=\widetilde{\Phi}^{\mathrm{EID}}(C;\mathbf{y}_{\ell}^{\mathrm{all}}). \tag{4.9}
$$

因此，对根节点 $S$，算法得到一棵二分树、split-residual 原子集合 \(\mathcal{R}_{\ell}\) 和 terminal 原子集合 \(\mathcal{U}_{\ell}\)。在分解容差内，两类原子之和闭合到报告的系统级联合增量：

$$
\Phi^{\mathrm{EID}}(S;\mathbf{y}_{\ell}^{\mathrm{all}})
\simeq
\sum_{C\in\mathcal{R}_{\ell}}\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}})
+\sum_{C\in\mathcal{U}_{\ell}}\eta_C(\mathbf{y}_{\ell}^{\mathrm{all}}). \tag{4.10}
$$

图中的阶数为 \(|C|\)：二阶表示源对残差，五阶表示五个模态必须一起读出的残差，`all 11 modes` 则表示根节点未被两个子块完全解释的全局残差。

式（4.3）定义原始集合联合增量，式（4.4）—（4.6）在容差约束下选择最能解释父块协同的二分，式（4.7）—（4.8）给出 signed split-residual 原子，式（4.9）定义无可用正分解时的 terminal 原子，式（4.10）检验两类原子对总量的数值闭合。该输出是贪婪层级下的 signed 残差分布，不是严格的 Möbius 纯阶原子；其结果依赖二分路径、\(\tau\) 和 \(\varepsilon\)，应解释为“沿当前贪婪树仍需联合读取的模态集合”，而不是唯一的高阶信息分解。

![UniCM 全模态 PhiEID 的贪婪层级分解](../../fig/unicm_phi_eid_greedy_decomposition.png)

*图 14. 贪婪原子在数值精度内与总 \(\Phi^{\mathrm{EID}}\) 闭合，主要贡献集中于 ENSO 空间型态及 IOD 背景的嵌套模块。左图为不同阶数原子的堆叠分布，黑线为原子之和；右图为按全部 seed 和 lead 平均贡献排序的主要模块。该分解依赖贪婪路径与数值容差，不代表唯一的高阶原子。*

分解结果在 `split_tolerance` 范围内与上一节的总 \(\Phi^{\mathrm{EID}}\) 闭合；逐 seed/lead 的最大偏差约为 `7.9e-05` bits。峰值仍在 lead 8，`phi_atom_sum_mean=0.183958` bits。按全部 `3 seeds × 24 leads`、缺失视为 0 的平均贡献排序，最强模块是：

| 排名 | 贪婪模块 | 阶数 | 平均原子量（bits） | 最大原子量（bits） | 非零次数 |
|---:|---|---:|---:|---:|---:|
| 1 | ENSO + IOD + nino12 + nino3 + nino4 | 5 | 0.009840 | 0.041831 | 34/72 |
| 2 | ENSO + nino12 + nino3 + nino4 | 4 | 0.008738 | 0.050393 | 26/72 |
| 3 | all 11 modes | 11 | 0.006128 | 0.013373 | 71/72 |
| 4 | ENSO + nino3 + nino4 | 3 | 0.005791 | 0.049302 | 18/72 |
| 5 | ENSO + nino12 + nino3 | 3 | 0.005745 | 0.049969 | 21/72 |
| 6 | nino12 + nino3 | 2 | 0.005337 | 0.038964 | 26/72 |

把 peak lead 8 单独展开后，可以更清楚地看到分布集中度：

![UniCM lead 8 的 PhiEID 原子分布](../../fig/unicm_phi_eid_lead8_distribution.png)

*图 15. Lead 8 的 \(\Phi^{\mathrm{EID}}\) 由二至五阶模块主导，其中 `ENSO + IOD + nino12 + nino3 + nino4` 是最大单个原子。a 为按 seed 均值排序的前 12 个原子；b 为模态成员矩阵；c 为不同阶数的原子质量；d 为总量与主要原子摘要。该图只展开峰值 lead，不能代表模块在全部预测期的稳定排序。*

在 lead 8，最大原子 `ENSO + IOD + nino12 + nino3 + nino4` 贡献 `0.032661` bits，占总 \(\Phi^{\mathrm{EID}}\) 的 `17.8%`。前 12 个原子合计覆盖 `87.6%` 的 lead-8 质量。按阶数看，二至五阶是主贡献区间，分别约占 `21%`、`22%`、`20%` 和 `18%`；六阶以上主要是较小的跨块残差。

这个结果说明，系统级 \(\Phi^{\mathrm{EID}}\) 的主要可解释层级集中在 ENSO 空间型态及 IOD 背景的嵌套组合上，而不是平均分散到全部模态。`all 11 modes` 残差几乎在每个 seed/lead 都存在，但量级较小，表示仍有弱的全局跨块残差。完整原子表见 `results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_greedy_atoms.csv`。



### 4.5 二源 Syn 的辅助证据

#### IOD 目标：自身记忆与印度洋/ENSO 背景共同调制

作为对照，这里将目标由 ENSO 改为 IOD，其余全历史最大熵干预口径保持一致：`8192` 个样本、checkpoint seeds `1, 2, 3`、lead `1..24`、干预范围 `[-4, 4]`，源变量仍为 11 个 UniCM 模态。图中展示 IOD 目标平均 Syn 排名前 12 的源对，并保留同一组固定对照源对。

| 排名 | 源对 | 平均 Syn（1—24） | seed 标准差 | 正值 seed 数 | 联合 EI | 左源 EI | 右源 EI |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | IOD + SIOD | 0.012107 | 0.009310 | 3/3 | 0.350317 | 0.320329 | 0.017881 |
| 2 | ENSO + IOD | 0.007147 | 0.002376 | 3/3 | 0.347583 | 0.020106 | 0.320329 |
| 3 | IOD + nino4 | 0.005648 | 0.002882 | 3/3 | 0.345515 | 0.320329 | 0.019538 |
| 4 | NPMM + IOD | 0.005263 | 0.004317 | 3/3 | 0.336838 | 0.011246 | 0.320329 |
| 5 | SPMM + IOD | 0.004950 | 0.004075 | 3/3 | 0.334724 | 0.009445 | 0.320329 |
| 6 | IOB + IOD | 0.004779 | 0.003570 | 3/3 | 0.333782 | 0.008674 | 0.320329 |
| 7 | IOD + TNA | 0.003301 | 0.002480 | 3/3 | 0.331301 | 0.320329 | 0.007671 |
| 8 | IOD + WWV | 0.003011 | 0.001956 | 3/3 | 0.329901 | 0.320329 | 0.006562 |
| 9 | IOD + nino3 | 0.002660 | - | - | 0.343585 | 0.320329 | 0.020596 |
| 10 | IOD + nino12 | 0.002530 | - | - | 0.333587 | 0.320329 | 0.010729 |

![IOD 目标的二源 Syn 曲线](../../fig/unicm_iod_mode_pair_syn_leads.png)

*图 16. IOD 的二源协同以自身记忆和印度洋/ENSO 背景的联合调制为主，并在 lead 15 后趋近于零。实线为各 lead 的 seed 均值，同色浅虚线为该源对在 lead 1—24 上的平均 Syn。*

IOD 结果的主信号与 ENSO target 不同：排名靠前的 pair 大多包含 IOD 自身历史，说明 IOD 未来状态的主要可预测部分仍由自身 12 个月历史提供；但 `IOD + SIOD`、`ENSO + IOD`、`IOD + nino4`、`NPMM + IOD` 等组合有正的额外二源增益。`IOD + SIOD` 在 lead 1 达峰，`ENSO + IOD` 和 `IOD + nino4` 在 lead 8 附近更强，说明印度洋内部结构和 ENSO/太平洋背景态主要影响短中期 IOD 演变。到 lead 15 后多数曲线贴近 0，不能支持长期稳定二源协同。

需要注意，`IOD + SIOD` 的 seed SD 仍接近均值，说明具体 rank 不宜过度解释。这里更稳妥的结论是：在当前 UniCM learned mechanism 中，IOD target 的二阶协同主要表现为 IOD 自身记忆与印度洋/ENSO 背景态的条件调制，而不是单个外部 mode 的独立强迫。

![各目标模态的主要二源 Syn 曲线](../../results/unicm_full_history_pair_syn_tm_degree1_n8192/fig/full_history_mode_pair_syn_top.png)

*图 17. 不同目标模态的主要二源组合各不相同，说明协同结构具有目标依赖性。每个目标显示按 lead 1—24 平均 Syn 排名前五的源模态对，曲线为 checkpoint seed 均值。该汇总适合比较总体结构，但会掩盖 seed 方差和单个 lead 的不稳定性。*

## 5. 综合讨论与解释边界

Runge 与 UniCM 给出尺度互补的高阶证据。Runge 结果表明，二源超边会随 \(H\) 在不同区域组合之间迁移；UniCM 结果表明，系统级 \(\Phi^{\mathrm{EID}}\) 在中期增强，并可追溯到 ENSO 空间型态与 IOD 背景的嵌套模块。两者共同支持一个窄而可检验的结论：气候可预测信息不仅存在于单模态记忆或成对联系中，还存在于依赖预测尺度的联合状态中。Runge 超边适合提出空间遥相关候选，UniCM 的 \(\Phi^{\mathrm{EID}}\) 分解适合定位多模态联合读出的层级；二者不能互相替代，也尚未构成同一动力方程下的闭环验证。

### 5.1 解释边界

- 本文只分析 frozen UniCM checkpoint 的 Modeformer learned mechanism，不使用 reanalysis 数据做预测复现，也不做单个历史事件归因。
- UniCM 的 overall EI 与 mode-pair Syn 使用 Gaussian log-det MI；这适合快速筛查，不等同于 transport-map PEID 的最终非线性分解。
- Syn 可以为负，表示 pair 的联合读数低于两个单源读数之和；本文所有 EI、\(\Phi^{\mathrm{EID}}\)、Syn 与分解残差均不做非负截断。
- Overall EI 的 ENSO/nino 与 IOD target 均未通过 lead 排序的 seed 鲁棒性标准；因此应解释稳定方向和量级，不应解释单个 lead 的精细排序。
- Runge SLP 面板中的 PC-stable graph 仍不是原文 Fig. 4 的逐项复刻；当前 60 个 Varimax component 是在 1948—2026 扩展样本上重新拟合得到的，编号也不是官方固定标签，不能把未校准节点直接命名为确定气候过程。
- 图 2—5、表 1 和跨尺度复现结论均来自每个 \(H\) 的全部 `102660` 条三阶 TM 候选；离散前 1000 候选只在图 5 中用于诊断初筛覆盖偏差。全量穷举解决了覆盖偏差，但尚未进行 block-bootstrap 显著性筛选。
- 四条代表超边的强制 TM 趋势已包含在全量候选中，但仍是事后选择的机制示例；地理距离只按分量空间中心计算，是空间跨度诊断，不等同于完整 loading footprint 的物理距离。

## 6. 图表与数据索引

- Overall EI 逐 seed / target / lead 原始结果：`results/unicm_overall_ei_tm_degree1_n8192/overall_ei_rows.jsonl`
- Overall EI target 鲁棒性汇总：`results/unicm_overall_ei_tm_degree1_n8192/overall_ei_seed_robustness_summary.csv`
- Overall EI lead-level seed mean/std：`results/unicm_overall_ei_tm_degree1_n8192/overall_ei_seed_lead_summary.csv`
- Overall EI 图：`results/unicm_overall_ei_tm_degree1_n8192/fig/overall_ei_seed_overlay.png`
- Full-history mode-pair Syn raw rows：`results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_rows.jsonl`
- Full-history mode-pair Syn pair summary：`results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_summary.csv`
- Full-history mode-pair Syn lead summary：`results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_lead_summary.csv`
- Full-history mode-pair Syn top pairs：`results/unicm_full_history_pair_syn_tm_degree1_n8192/full_history_mode_pair_syn_top_pairs.csv`
- Full-history mode-pair Syn 图：`results/unicm_full_history_pair_syn_tm_degree1_n8192/fig/full_history_mode_pair_syn_top.png`
- All-mode target pair Syn 完整 lead 表：`results/unicm_all_mode_target_pair_syn_cpu_bound4_n8192/all_mode_target_pair_syn_lead_summary.csv`
- All-mode target PhiEID 逐 seed 表：`results/unicm_all_mode_target_phi_eid_cpu_bound4_n8192/all_mode_target_phi_eid_rows.csv`
- Greedy PhiEID atom 表：`results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_greedy_atoms.csv`
- Runge Ridge+PEID 一阶/二阶 ACE/ACS 图：`fig/runge_ridge_peid_order1_vs_order2_ace_acs_1948_2026.png`
- Runge 周尺度分量输入：`results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv`（`component_scores_hash=2cd78d429fc66b30`）
- Runge 多步推演上游清单：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/pairwise_mlp_tm_ei_path_effects/manifest.json`
- Runge 全量三阶 TM \(H=1\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H001_tm_exhaustive.png`
- Runge 全量三阶 TM \(H=1\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H001_tm_exhaustive.csv`
- Runge 全量三阶 TM \(H=10\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H010_tm_exhaustive.png`
- Runge 全量三阶 TM \(H=10\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H010_tm_exhaustive.csv`
- Runge 全量三阶 TM \(H=60\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H060_tm_exhaustive.png`
- Runge 全量三阶 TM \(H=60\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H060_tm_exhaustive.csv`
- Runge 全量三阶 TM 结果目录：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_exhaustive`
- Runge 全候选三阶 TM 分块、排名与门禁结果：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_exhaustive`
- Runge 全候选与离散 shortlist 汇总图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive_summary.png`
- Runge 全候选汇总数据：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive_summary_summary.json`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) top10 距离表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_trends_top10_distances.csv`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) 距离汇总表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_trends_distance_summary.csv`
- Runge 多步 MLP+Ridge 代表超边强制 TM 趋势图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.png`
- Runge 多步 MLP+Ridge 代表超边强制 TM 趋势表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.csv`
- Runge 多步 MLP+Ridge 代表超边强制 TM 结果目录：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_forced_edges`

## 7. 参考文献

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
