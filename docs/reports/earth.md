# Runge 与 UniCM 时空因果机制证据

## 目录

- [Runge SLP：ACE/ACS 的原文口径与 Ridge+PEID 对齐](#runge-slpaceacs-的原文口径与-ridgepeid-对齐)
- [UniCM 实验口径](#unicm-实验口径)
- [Mode 地理含义](#mode-地理含义)
- [Overall EI：ENSO 与 IOD 的全历史读数](#overall-eienso-与-iod-的全历史读数)
- [单源 EI](#单源-ei)
- [All-mode self EI：不同模态的自身记忆尺度不同](#all-mode-self-ei不同模态的自身记忆尺度不同)
- [ENSO target 二源 Syn：空间型态提供短中期协同](#enso-target-二源-syn空间型态提供短中期协同)
- [All-mode target 二源 Syn：整体未来状态的协同主要来自 ENSO 空间结构](#all-mode-target-二源-syn整体未来状态的协同主要来自-enso-空间结构)
- [All-mode target PhiEID：系统级联合增量在中期增强](#all-mode-target-phieid系统级联合增量在中期增强)
- [All-mode target PhiEID 的层级贪婪分解](#all-mode-target-phieid-的层级贪婪分解)
- [IOD target 二源 Syn：自身记忆与印度洋/ENSO 背景共同调制](#iod-target-二源-syn自身记忆与印度洋enso-背景共同调制)
- [图表与数据索引](#图表与数据索引)
- [解释边界](#解释边界)
- [参考文献](#参考文献)

## Runge SLP：ACE/ACS 的原文口径与 Ridge+PEID 对齐

这一组实验只展示 ACE 和 ACS，不再展示 AMCE。目的很窄：在同一套 1948-2026 NCEP SLP 周尺度 Varimax 分量上，对比 Runge 等人 [R1] 的线性 causal gateway / susceptibility 算法，与 Ridge+PEID 的 Hyper-ACE / Hyper-ACS 读数。输入为 60 维 component，使用最近 4 周状态预测下一周状态。

数据处理按当前实际 pipeline 执行：读取 1948-2026 年 NCEP daily SLP，删除 2 月 29 日；在每个格点上按 365-day calendar day 减多年均值并除以多年标准差，再沿时间轴做线性去趋势。随后先把标准化、去趋势后的 daily fields 聚合为 monthly fields，在 monthly fields 上重新拟合 60 个 Varimax-rotated PCA component weights；再把这组空间权重投影回同样预处理过的 daily fields，得到 daily component scores，最后按连续 7 天均值聚合为 weekly component scores。这个 monthly-fit、daily-projection、weekly-aggregation 流程对应 Runge 等人 [R1] Methods 的降维操作；其中删除 2 月 29 日和 365-day calendar-day 标准化是本仓库对“去除季节均值和季节方差”的具体实现。Ridge 读出模型在这套 1948-2026 weekly scores 上重训；有效 lagged samples 为 `4074`，正则为 `alpha=1000`。

Runge 面板使用原文算法的核心步骤：先用 PC-stable parent selection 得到 sparse causal graph，再用线性 SEM 估计跨 lag causal effect。此前本地复现误把 `run_pcmci` 的最终 MCI `p_matrix` 当成 parent set，导致 No.3 排名异常偏高；这里已改为 `run_pc_stable` parents，再做稀疏线性回归和 link-density threshold。修正后，1948-2026 新数据上 No.3 从 ACE 第 5、ACS 第 3 降为 ACE 第 12、ACS 第 13。

记 \(C_{ij}\) 为源分量 \(i\) 到目标分量 \(j\) 的跨 lag 最大绝对 causal effect，则 Runge 原文口径下

$$
\mathrm{ACE}_{\mathrm{Runge}}(i)=\frac{1}{n-1}\sum_{j\ne i}C_{ij},
\qquad
\mathrm{ACS}_{\mathrm{Runge}}(i)=\frac{1}{n-1}\sum_{j\ne i}C_{ji},
\qquad n=60 .
$$

Ridge+PEID 面板使用同一套 1948-2026 component scores。PEID 候选设置为旧口径：`intervention_samples=4096`、`candidate_top_sources=14`、`candidate_target_topk=10`、`order_max=2`、`null_reps=20`、显著门槛 \(|z|\ge2\)。记 \(EI_{i\to j}\) 为一阶有效信息，\(Syn_{K\Rightarrow j}^{\mathrm{EID}}\) 为二源集合 \(K\) 对目标 \(j\) 的 EID 协同项：

$$
Syn_{K\Rightarrow j}^{\mathrm{EID}}
=
EI\bigl(X_t^K\to X_{t+1}^{(j)}\bigr)
-\sum_{a\in K}EI\bigl(X_t^{(a)}\to X_{t+1}^{(j)}\bigr).
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
\end{cases}
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
\end{cases}
$$

这两个 Hyper 指标是一步预测读出上的直接一阶边和显著二阶超边聚合，不计算“二阶超边影响一个节点后再沿 causal graph 多步传播”的高阶路径中心性。当前图使用的读数共有 `1638` 条二阶候选，其中 `287` 条通过 \(|z|\ge2\) 门槛。在线性读出上，二阶项主要修正通过显著性筛选留下的局部超边平均强度，而不是把未显著的潜在超边也计入分母。

![Runge reproduction and Ridge+PEID ACE/ACS](../../fig/runge_ridge_peid_order1_vs_order2_ace_acs_1948_2026.png)

*图 1. 同一套 1948-2026 SLP component 上，Runge 原文方法复现与 Ridge+PEID Hyper-ACE/Hyper-ACS 的对比。a 为修正后的 Runge 2015 PC-stable ACE/ACS 复现；b 为 Ridge+PEID 一阶 EI 读数；c 为 Ridge+PEID 一阶 EI 加显著二阶协同后的读数。外圈表示 ACE 或 Hyper-ACE，内圈表示 ACS 或 Hyper-ACS。a 面板使用 Runge 线性 SEM 尺度，b/c 面板使用共同截断色标。Ridge+PEID 加二阶后的 ACE top-5 是 `No.0/1/3/9/4`，ACS top-5 是 `No.10/3/26/0/1`。*

修正后的 Runge 方法 ACE top-3 是 `No.1/0/16`，ACS top-3 是 `No.0/1/26`；Ridge+PEID 的 ACE top-5 是 `No.0/1/3/9/4`，ACS top-5 是 `No.10/3/26/0/1`。需要保留两个限制：第一，修正后的 PC-stable graph 仍不等于原文 Fig. 4 的逐项复刻；第二，60 个 Varimax component 的编号不是官方固定标签，当前只对少数论文讨论节点做了视觉校准，因此不能把低排名或未校准节点直接命名为确定气候过程。

上面的 ACE/ACS 地图把每个节点相关的一阶边和显著二阶超边都压缩成节点分数，因此只能回答“哪个 component 更像 source 或 target hub”。为了检查二阶项本身落在什么地理关系上，下面把视角从节点分数退回到具体超边。

这一步也改成同一套 1948-2026 新数据：多步 rollout 的上游 manifest 为 `results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/pairwise_mlp_tm_ei_path_effects/manifest.json`，输入 scores 为 `results/runge_slp_daily_1948_2026_20260628/results/runge/2015_gateways/component_weekly_scores.csv`，`component_scores_hash=2cd78d429fc66b30`，有效 lagged samples 为 `4074`。该 rollout 使用验证集选择的 MLP+Ridge blend，其中 Ridge transition 在新数据 lagged split 上直接拟合，`ridge_alpha=1000`，`ridge_weight=0.37`，`mlp_weight=0.63`。

由于对所有 \(H=1,2,\ldots,10,15,20,30,40,50,60\) 的 `102660` 条跨目标二阶候选逐一做 transport-map MI 会很慢，这里采用两步口径：先用脚本内置的离散化 MI 对每个 horizon 做全局候选筛查，再对每个 horizon 的离散 top-1000 候选逐条重算 TM MI，并按 TM 二阶增量重新排序。每个候选都估计单源 \(EI_{i\to t}^{[h]}\)、\(EI_{j\to t}^{[h]}\) 和二源联合 \(EI_{\{i,j\}\to t}^{[h]}\)，再把二阶增量写成

$$
\Delta_{2,\mathrm{TM}}^{[h]}(i,j\Rightarrow t)
=EI_{\{i,j\}\to t}^{[h]}-EI_{i\to t}^{[h]}-EI_{j\to t}^{[h]} .
$$

下面不再只看 `No.0` 的局部 incident 超边，而是直接比较 \(H=1\)、\(H=10\) 和 \(H=60\) 三个预测截面的 TM 重估 top10 候选。每条超边由两个 source 节点汇入一个紫色 hub，再由 hub 指向 target 节点。图中只显示 top10 涉及的节点；其他 component 以浅灰点作为空间参照。这组图回答的是一个局部但更具体的问题：在给定预测尺度上，哪些二源组合在 TM 读数下提供了超出两个单源相加的额外读数？

![Top-10 H1 TM-reestimated second-order candidates](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H001_tm_reranked.png)

*补充图 1a. 新 1948-2026 SLP 数据上，多步 MLP+Ridge rollout 与 TM MI 重估下，\(H=1\) 的 top10 二阶候选。候选池为离散 MI 全局 top-1000，图中排序和线宽使用 \(\Delta_{2,\mathrm{TM}}\)。最高三条为 `No.0 + No.12 -> No.37`、`No.23 + No.21 -> No.26` 和 `No.9 + No.39 -> No.49`。*

![Top-10 H10 TM-reestimated second-order candidates](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H010_tm_reranked.png)

*补充图 1b. 同一口径下，\(H=10\) 的 top10 TM 重估二阶候选。最高三条为 `No.0 + No.1 -> No.28`、`No.0 + No.6 -> No.32` 和 `No.4 + No.9 -> No.7`。*

![Top-10 H60 TM-reestimated second-order candidates](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H060_tm_reranked.png)

*补充图 1c. 同一口径下，\(H=60\) 的 top10 TM 重估二阶候选。最高三条为 `No.0 + No.1 -> No.46`、`No.0 + No.1 -> No.50` 和 `No.1 + No.26 -> No.11`。*

TM 重估后，三组 top 候选的量级和离散 MI 图明显不同。下表中 `source EI sum` 是两个单源 EI 之和，`discrete shortlist rank` 是该边在原离散 MI 候选池中的名次；真正用于排序的是 \(\Delta_{2,\mathrm{TM}}\)。

| H | TM rank | Edge | \(\Delta_{2,\mathrm{TM}}\) | joint EI | source EI sum | discrete shortlist rank |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | `0+12->37` | 0.005568 | 0.137311 | 0.131743 | 325 |
| 1 | 2 | `23+21->26` | 0.003387 | 0.046392 | 0.043005 | 363 |
| 1 | 3 | `9+39->49` | 0.003337 | 0.047612 | 0.044274 | 489 |
| 1 | 4 | `13+7->39` | 0.003087 | 0.074715 | 0.071627 | 703 |
| 1 | 5 | `35+43->45` | 0.003054 | 0.004320 | 0.001267 | 101 |
| 10 | 1 | `0+1->28` | 0.017747 | 0.228734 | 0.210987 | 48 |
| 10 | 2 | `0+6->32` | 0.010952 | 0.184992 | 0.174040 | 247 |
| 10 | 3 | `4+9->7` | 0.010586 | 0.146991 | 0.136405 | 173 |
| 10 | 4 | `1+26->55` | 0.007335 | 0.124876 | 0.117541 | 380 |
| 10 | 5 | `14+45->23` | 0.006178 | 0.119884 | 0.113705 | 243 |
| 60 | 1 | `0+1->46` | 0.018027 | 0.231307 | 0.213280 | 1 |
| 60 | 2 | `0+1->50` | 0.013515 | 0.200558 | 0.187043 | 602 |
| 60 | 3 | `1+26->11` | 0.004039 | 0.094403 | 0.090364 | 808 |
| 60 | 4 | `14+45->5` | 0.003329 | 0.084115 | 0.080786 | 7 |
| 60 | 5 | `6+47->43` | 0.003107 | 0.035712 | 0.032605 | 127 |

这个 TM 重估结果进一步削弱了 \(H=1\) 离散 top1 `No.27 + No.58 -> No.11` 的解释价值。该边在离散 MI 下排名第 1，但 TM 重估后 \(\Delta_{2,\mathrm{TM}}=0.000968\)，在 \(H=1\) 离散 top-1000 候选内只排第 `230`。因此它不应作为一周尺度的物理遥相关候选。相反，\(H=10\) 和 \(H=60\) 的 TM top 候选开始更集中地包含 `No.0/No.1` 这类前面 ACE/ACS 中已经突出的节点；其中 \(H=60\) 的 `No.0 + No.1 -> No.46` 同时也是离散筛查第 1，TM 增量为 `0.018027`，是三张图里最强的二源非加性候选。

为了避免只解释三个截面，下面把同一 TM 重估规则扩展到 \(H=1,2,\ldots,10,15,20,30,40,50,60\)。上方面板统计每个 horizon 的 TM top10 里有多少边属于 `No.0+1`、`No.0/1 source` 或其他来源族，并叠加该 horizon 的最大 \(\Delta_{2,\mathrm{TM}}\)。下方面板列出跨 horizon 反复出现或在某个 horizon 特别强的 TM top10 候选，格子里的数字是该 horizon 内排名。

![Top-10 TM-reestimated second-order candidates across horizons](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.png)

*补充图 2. 新 1948-2026 SLP 数据上，不同 \(H\) 的 TM 重估 top10 正二阶候选汇总。候选池仍来自每个 horizon 的离散 MI top-1000，但图中排序、颜色和折线均使用 \(\Delta_{2,\mathrm{TM}}\)。最反复出现的候选包括 `No.0 + No.1 -> No.28`、`No.0 + No.6 -> No.32`、`No.0 + No.1 -> No.50` 和 `No.0 + No.1 -> No.46`。*

跨 horizon 的 TM 重估图比离散图更集中：`No.0 + No.1 -> No.28` 在 \(H=6,7,8,9,10\) 出现 `5` 次，最大 \(\Delta_{2,\mathrm{TM}}=0.020379\)；`No.0 + No.6 -> No.32` 也出现 `5` 次，覆盖 \(H=4,5,6,10,15\)，最大值 `0.018717`；`No.0 + No.1 -> No.50` 出现 `5` 次，覆盖 \(H=5,15,30,50,60\)，最大值 `0.013515`；`No.0 + No.1 -> No.46` 出现 `4` 次，集中在 \(H=30,40,50,60\)，最大值 `0.018027`。因此，短尺度 \(H=1\) 的强候选仍应谨慎解释，但中长期 horizon 上围绕 `No.0/No.1` 的组合已经和节点级 ACE/ACS 证据相互呼应。更稳妥的解释是：前面的显著 Ridge+PEID ACE/ACS 地图提供节点级 source/target hub 证据，而这里的 lead-resolved 超边图提供候选机制的 horizon 扫描；只有跨多个 \(H\) 反复出现，并且通过 TM/PEID null、季节分层和 block-bootstrap 检验的边，才值得写成物理遥相关链。

为了排除离散 top-1000 初筛造成的“缺失点”问题，我又对这四条代表边在全部报告 horizon 上强制重算 TM，不再要求它们进入该 horizon 的离散候选池：

![Forced TM hyperedge horizon trends](../../fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.png)

*补充图 3. 四条代表二阶候选的强制 TM horizon 趋势。每个点都直接重算 \(EI_i\)、\(EI_j\) 和 \(EI_{ij}\)，因此曲线不再受离散 top-1000 初筛缺失点影响。横轴为报告中使用的 \(H=1,2,\ldots,10,15,20,30,40,50,60\)。*

强制 TM 后，原先由缺失点造成的断线消失，趋势也更清楚。`No.0 + No.6 -> No.32` 是早期峰值型：从 \(H=1,2\) 的 `0.000258/0.001482` 升到 \(H=4\) 的 `0.018717`，之后降到 \(H=15,30,60\) 的 `0.006064/0.003694/0.002987`。`No.0 + No.1 -> No.28` 是中期峰值型：\(H=7\) 达到 `0.020379`，随后降到 \(H=20,40,60\) 的 `0.013970/0.009142/0.007221`。`No.0 + No.1 -> No.50` 则是较平滑的长尺度平台型：从 \(H=5\) 的 `0.011576` 到 \(H=15,30,60\) 的 `0.012524/0.013154/0.013515`，变化幅度小但持续为正。`No.0 + No.1 -> No.46` 最接近长尺度增强型：短期 \(H=1..10\) 基本低于 `0.0025`，到 \(H=20,30,40,50,60\) 依次为 `0.004079/0.009712/0.014776/0.017874/0.018027`。

地理距离的证据仍要写得谨慎。按 TM top10 统计，短期 \(H\le5\) 的 source-target 平均距离中位数为 `9.37e3 km`，长期 \(H\ge20\) 为 `10.15e3 km`；最远 source-target 距离中位数从 `12.55e3 km` 增到 `13.15e3 km`，三节点最大跨度中位数从 `13.11e3 km` 增到 `13.93e3 km`。这支持“较长 horizon 更偏向大尺度、远程组合”的弱趋势，但不是严格单调，也不意味着短 horizon 的所有候选都是局地相邻；例如中期 \(6\le H\le15\) 由于 `No.0/No.1` 组合反复出现，source-source 距离中位数反而最高，为 `12.75e3 km`。

## UniCM 实验口径

这里分析的是 frozen UniCM Modeformer learned mechanism，不是 reanalysis 预测技能评估，也不是单个历史事件归因。每个干预样本同时采样 12 个历史月份和 11 个 UniCM mode 维度，形成 `(B, 12, 11)` 的 bounded uniform 最大熵输入，历史张量写入 Modeformer encoder 的 12 个月历史段，未来 24 个月由 decoder 自回归生成。

核心配置如下：

| Item | Value |
|---|---|
| checkpoint seeds | `1, 2, 3` |
| current intervention samples | `8192` |
| intervention support | all 12 historical months x 11 mode dimensions sampled independently from `[-4, 4]` |
| sampling seed | `20260619` |
| start month | `0` |
| bootstrap repeats | ENSO summary: `200`; IOD pair curve: seed mean only |
| target mode | 图 3、图 5 和图 7 为 ENSO；图 6、图 8-11 为 all modes；图 4、图 12-13 包含 IOD |
| source modes | ENSO/nino, NPMM, SPMM, IOB, IOD, SIOD, TNA, nino12, nino3, nino4, WWV |

整体 EI 使用 flattened full-history source，即 132 维历史 mode 输入，对每个 lead 的目标 mode 输出估计 `EI(history; target_lead)`。高维整体读数采用 Gaussian log-det MI 作为快速筛查口径；它用于检查绝对量级和 seed 稳定性，不等同于最终的非线性 transport-map PEID 分解。本文保留二源 Syn 读数：

$$
\mathrm{Syn}_{ij}=EI_{ij}-EI_i-EI_j.
$$

其中 `EI_i` 和 `EI_j` 是两个 source mode 的 12 个月历史分别到同一目标 lead 输出的单源 EI，`EI_{ij}` 是二者联合 source 到同一目标的 EI。所有这些读数都使用 Gaussian log-det 估计，适合作为 full-history 机制筛查；它们不等同于最终的非线性 transport-map PEID 分解。

## Mode 地理含义

![UniCM mode geography](../../fig/unicm_mode_geography.png)

*图 2. UniCM mode 输入的地理区域。ENSO 相关指数来自赤道太平洋不同经向区段；NPMM、SPMM 和 TNA 提供太平洋经向模态与热带北大西洋背景；IOD/SIOD/IOB 表示印度洋盆地和偶极型 SST 结构。*

这张图是解释后续 EI/Syn 的基础。`nino3`、`nino4` 和 `nino12` 不是 ENSO 之外的独立外部强迫，而是赤道太平洋内部空间结构的不同读数。因此当 `ENSO + nino3` 或 `ENSO + nino4` 出现高 Syn 时，更自然的解释是 ENSO 的当前强度需要和东西向 SST 型态一起读，才能判断未来几个月的演变。

## Overall EI：ENSO 与 IOD 的全历史读数

全历史整体 EI 的主窗口为 lead `1..24`，climate-relevant 补充窗口为 lead `6..18`。seed 鲁棒性通过标准为：seed-pair Pearson >= `0.80`，Spearman >= `0.75`，top-3 EI lead overlap >= `2`。按这个标准，ENSO/nino 和 IOD 的曲线形状有一定一致性，但 lead 排序未通过鲁棒性标准。

| Target | mean EI 1..24 | mean EI 6..18 | Pearson min | Spearman min | top-3 overlap min | status |
|---|---:|---:|---:|---:|---:|---|
| ENSO/nino | 0.617162 | 0.395603 | 0.950 | 0.482 | 3 | 不稳定 |
| IOD | 0.535641 | 0.467182 | 0.854 | 0.245 | 2 | 不稳定 |

![ENSO overall EI](../../fig/unicm_enso_overall_ei_seed_overlay.png)

*图 3. ENSO target 的 full-history overall EI lead 曲线。彩色细线为 checkpoint seed，黑线为 seed mean，阴影为 seed standard deviation。*

![Full-history overall EI seed overlay](../../results/unicm_overall_ei_tm_degree1_n8192/fig/overall_ei_seed_overlay.png)

*图 4. Full-history overall EI lead curves under the selected bounded maximum-entropy intervention. Each panel is one target mode and each curve is one checkpoint seed; stable targets should show both similar curve shape and similar lead ordering across seeds.*

这两张图说明，UniCM learned mechanism 对 ENSO/nino 的有效信息主要集中在 lead 1 到 6 个月。短 lead 的 EI 明显高于后期，符合 ENSO 预测中短期记忆强、长期不确定性上升的物理直觉。三个 checkpoint 的曲线形状相近，Pearson min 达到 `0.950`；但 Spearman min 只有 `0.482`，说明不同 checkpoint 对具体 lead 排序仍不够稳定。IOD 的 mean EI 在 `6..18` 窗口更接近总体均值，但 Spearman min 只有 `0.245`。因此 overall EI 可以支持“全历史输入含有可读出的短中期机制信息”的方向性判断，但不能把每个 lead 的细粒度排序解释得太重。

## 单源 EI

| Target | self EI | strongest non-self sources | NPMM EI | TNA EI |
|---|---:|---|---:|---:|
| ENSO | 0.473612 | nino12 0.015768; nino3 0.015599; IOD 0.013361; SPMM 0.012534 | 0.011671 | 0.005806 |

![ENSO source EI lead curves](../../fig/unicm_enso_source_ei_rankings.png)

*图 5. ENSO target 的单源 EI lead 曲线。左图单独显示 ENSO self source；右图显示按 24 个月平均 EI 选出的非自身 Top-5，并保留 NPMM/TNA。实线和浅色带分别为 checkpoint seed mean 和 standard deviation。*

单源 EI 曲线显示，ENSO 自身历史在短 lead 占绝对主导，但随后快速衰减。排除自身后，`nino3`、`nino12` 和 IOD 的 EI 随 lead 增长并在较长 lead 位于前列，NPMM 则在中期达到较高水平后回落；这些长 lead 曲线的 checkpoint 波动也明显扩大，因此不宜过度解释精细排序。TNA 的曲线始终较低，更稳妥的说法是，它可能只在 ENSO 背景态或其他太平洋/印度洋模态共同存在时提供弱增量。

## All-mode self EI: 不同模态的自身记忆尺度不同

为了和二源 Syn 的量级作对照，这里进一步把每个 mode 都作为 target，并只输入该 mode 自己的 12 个月历史，计算 self EI 随 lead 的变化。也就是说，每条曲线都对应 `source = target`，没有引入其他 mode 的历史。

![All-mode self EI lead curves](../../fig/unicm_all_modes_self_ei_leads.png)

*图 6. UniCM 11 个 mode 的 self EI lead 曲线。实线为 checkpoint seed mean，浅色带为 seed standard deviation；横轴为 target lead，纵轴为该 mode 自身 12 个月历史到未来状态的 EI。*

这张图说明，self EI 的绝对量级显著大于前面的二源 Syn：多数 mode 在 lead 1 都有约 `1.6-2.2` bits 的自身历史信息，而二源 Syn 通常只有 `10^{-3}` 到 `10^{-2}` bits。`NPMM`、`IOB`、`WWV`、`SPMM`、`TNA` 和 `SIOD` 的 self EI 衰减较慢，lead 12 仍约 `0.89-0.99` bits；相反，ENSO 相关的 `nino`、`nino3`、`nino4`、`nino12` 以及 `IOD` 在前 6 到 10 个月后快速下降，lead 24 基本接近 `0.05-0.08` bits。

因此，self EI 主要读到的是各模态状态本身的持久性和自回归记忆，不应直接拿它和二源 Syn 当作同一层面的机制强度比较。二源 Syn 更像是在“已经有各自单源信息之后，两个历史变量联合读数还能额外提供多少目标信息”；它小很多是预期内的结果，也解释了为什么在分析协同项时需要单独画 Syn 曲线，而不能只看总 EI 或 self EI。

## ENSO target 二源 Syn：空间型态提供短中期协同

| Target | Source pair | rank | mean Syn 1..24 | Syn seed SD | 95% CI | seed rank range | joint EI 1..24 | left EI 1..24 | right EI 1..24 |
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

![ENSO mode-pair Syn leads](../../fig/unicm_enso_mode_pair_syn_leads.png)

*图 7. ENSO target 的 mode-pair Syn lead 曲线。实线为每个 lead 的 seed mean；同色浅虚线为该 pair 在 lead 1..24 上的平均 Syn.*

这张图的核心信息很直接：模型不是只看“ENSO 现在有多强”，还在看“暖异常更偏东、偏中太平洋，还是和其他海盆背景态一起出现”。前 1 到 7 个月，`ENSO + nino3` 和 `ENSO + nino4` 的 Syn 明显更高，说明 ENSO 的短期未来演变对赤道太平洋东西向 SST 结构很敏感。同样强度的 ENSO，如果空间型态不同，后续几个月的增长、衰减和位相演变也可能不同。

这个解释和 ENSO diversity 文献一致。Trenberth and Stepaniak [1] 指出，单一 ENSO 指数不足以描述事件演变，需要额外刻画中东太平洋 SST 梯度；Capotondi et al. [2] 把事件间差异总结为 ENSO 的振幅、空间型态、生命周期和触发机制差异；Ren and Jin [3] 进一步用 Niño3/Niño4 组合区分两类 ENSO。Kao and Yu [4] 与 Ashok et al. [5] 则分别从 EP/CP ENSO 和 ENSO Modoki 角度说明，中太平洋型和东太平洋型事件不能简单当作同一种 ENSO 强度的线性放大。

因此，`nino3` 和 `nino4` 更适合被解释为 ENSO 内部空间型态的调制因子，而不是 ENSO 之外的独立强迫源。曲线在 9 到 12 个月后整体贴近零，说明这种额外协同信息主要集中在短中期；到更长 lead，模型已经很难从这些二源组合里读出稳定的增量。

## All-mode target 二源 Syn: 整体未来状态的协同主要来自 ENSO 空间结构

这里把 target 从单个 `ENSO` mode 改成同一 lead 上 11 个未来 mode 的整体向量：

$$
Y_{\ell}^{\mathrm{all}} =
(\mathrm{ENSO}_{t+\ell}, \mathrm{NPMM}_{t+\ell}, \ldots, \mathrm{WWV}_{t+\ell}) .
$$

source 仍然是两个 mode 各自 12 个月历史，二源 Syn 定义为
`EI(source_i, source_j; Y_all) - EI(source_i; Y_all) - EI(source_j; Y_all)`。计算复用 `8192` 个 full-history 最大熵样本和 checkpoint seeds `1,2,3` 的已有预测缓存，没有重新执行 UniCM forward。

| Rank | Source pair | mean Syn 1..24 | seed SD | + seeds | seed rank range | joint EI |
|---:|---|---:|---:|---:|---|---:|
| 1 | nino12 + nino3 | 0.010650 | 0.000897 | 3/3 | 1-5 | 1.127968 |
| 2 | ENSO + nino3 | 0.008485 | 0.005602 | 3/3 | 1-9 | 1.161277 |
| 3 | nino3 + nino4 | 0.008248 | 0.004693 | 3/3 | 2-6 | 1.180777 |
| 4 | ENSO + IOD | 0.006689 | 0.000496 | 3/3 | 2-8 | 1.058004 |
| 5 | ENSO + nino12 | 0.006082 | 0.004256 | 3/3 | 4-10 | 1.148809 |
| 6 | IOD + nino3 | 0.006003 | 0.003825 | 3/3 | 5-10 | 1.031908 |
| 7 | ENSO + nino4 | 0.005872 | 0.006307 | 3/3 | 3-32 | 1.203812 |
| 8 | IOD + nino4 | 0.004638 | 0.000265 | 3/3 | 3-11 | 1.075691 |

![All-mode target mode-pair Syn leads](../../fig/unicm_all_mode_target_mode_pair_syn_leads.png)

*图 8. All-mode target 的二源 mode-pair Syn lead 曲线。彩色曲线为按 mean Syn 排名前 12 的 source pair，灰色细线为其余 source pair；浅色带为 checkpoint seed standard deviation，水平点线为该 pair 在 lead `1..24` 上的平均 Syn。*

这个 all-mode target 口径把未来整体气候 mode 状态作为一个多变量读出，因此不再只问“哪些 source pair 额外解释 ENSO”，而是问“哪些历史 pair 对 UniCM 未来整体状态有额外联合读数”。排名最高的组合仍集中在赤道太平洋内部结构：`nino12 + nino3`、`ENSO + nino3`、`nino3 + nino4`、`ENSO + nino12` 和 `ENSO + nino4` 都进入前 7。曲线峰值主要出现在 lead 6 到 10 个月，之后整体回落到约 `0.005-0.01` bits 或更低，说明整体未来状态的二阶协同也主要是短中期信号。

与 ENSO-only target 相比，all-mode target 会把印度洋相关背景也纳入同一个响应向量，所以 `ENSO + IOD`、`IOD + nino3`、`IOD + nino4` 进入前列。这不表示 IOD 单独支配整体未来状态；更稳妥的解释是，UniCM 的整体未来响应需要同时读赤道太平洋空间型态和部分印度洋背景态。所有 pair 的完整 lead 表见 `results/unicm_all_mode_target_pair_syn_cpu_bound4_n8192/all_mode_target_pair_syn_lead_summary.csv`。

## All-mode target PhiEID: 系统级联合增量在中期增强

在同一个 all-mode target 上进一步计算系统级

$$
\Phi^{EID}_{\ell}
= I(\mathbf{X}^{1:12}_{1:11};Y_{\ell}^{\mathrm{all}})
- \sum_{m=1}^{11} I(\mathbf{X}^{1:12}_{m};Y_{\ell}^{\mathrm{all}}).
$$

这里的 source partition 是 11 个 mode 的 singleton partition；每个 singleton source 是该 mode 的 12 个月历史。负的 raw 差值只保存在结果表中，图中报告非负 `max(0, raw PhiEID)`。这仍是 Gaussian log-det full-history 筛查口径，不等同于最终 transport-map PEID。

![All-mode target PhiEID leads](../../fig/unicm_all_mode_target_phi_eid_leads.png)

*图 9. All-mode target 的系统级 $\Phi^{EID}$ 随 lead 变化。上图为 $\Phi^{EID}$ 的 checkpoint seed mean 和 standard deviation；下图为 whole EI 与 singleton EI sum 的量级参照。*

曲线显示，whole EI 与 singleton EI sum 都随 lead 增长持续下降，但二者差值并不单调。`PhiEID` 在 lead 1..5 约 `0.05-0.07` bits，随后在 lead 7..10 增强，并在 lead 8 达到峰值 `0.183958 ± 0.042136` bits；lead 11..24 维持在约 `0.09-0.15` bits。也就是说，整体未来状态的系统级联合增量不是短 lead 最大，而是在中期更明显。

这个结果和上面的二源 Syn 曲线一致：单源或单 pair 对整体未来状态的解释在短 lead 已经很强，但不可约的多模态联合增量主要出现在 6 到 10 个月附近。完整逐 seed 表见 `results/unicm_all_mode_target_phi_eid_cpu_bound4_n8192/all_mode_target_phi_eid_rows.csv`。

## All-mode target PhiEID 的层级贪婪分解

进一步把每个 lead 的 all-mode `PhiEID` 按层级可加性分解。这个分解要回答的不是“唯一的高阶 PID 原子是什么”，而是一个更可读的问题：从全部 mode 开始，如果每一步都尽量把已经能由两个子模块解释的部分拆出去，那么还剩哪些模块必须被联合读取，才能解释当前的系统级 $\Phi^{EID}$？

### 层级贪婪分解怎么算

设全集为 $S=\{1,\ldots,11\}$，每个元素是一个 UniCM mode。对任意非空 mode 集合 $C\subseteq S$，令 $\mathbf{x}_C$ 表示集合 $C$ 中所有 mode 的 12 个月历史，$\mathbf{y}_{\ell}^{\mathrm{all}}$ 表示 lead $\ell$ 的 all-mode target。先定义这个集合自身的系统级增量：

$$
\Phi(C;\mathbf{y}_{\ell}^{\mathrm{all}})
= EI(\mathbf{x}_C;\mathbf{y}_{\ell}^{\mathrm{all}})
-\sum_{i\in C}EI(\mathbf{x}_i;\mathbf{y}_{\ell}^{\mathrm{all}}).
$$

这句话的直观意思是：先看整个集合 $C$ 一起读历史时能解释多少未来信息，再减掉每个 mode 单独读历史时能解释的信息。如果差值为正，说明 $C$ 的联合读出比单独读出之和多出一部分；这部分就是当前口径下的 $\Phi^{EID}$。

现在把当前节点 $C$ 拆成两个互不重叠、并且并起来等于 $C$ 的子块：

$$
L\cap R=\varnothing,\qquad L\cup R=C,\qquad L\neq\varnothing,\qquad R\neq\varnothing.
$$

对每个候选二分 $(L,R)$，先计算两个子块已经能解释的协同量：

$$
B(L,R;\mathbf{y}_{\ell}^{\mathrm{all}})
=\Phi(L;\mathbf{y}_{\ell}^{\mathrm{all}})
+\Phi(R;\mathbf{y}_{\ell}^{\mathrm{all}}).
$$

贪婪步骤选择 $B$ 最大的二分：

$$
(L^\star,R^\star)
=\underset{L\cup R=C,\ L\cap R=\varnothing}{\arg\max}\ 
\left[
\Phi(L;\mathbf{y}_{\ell}^{\mathrm{all}})
+\Phi(R;\mathbf{y}_{\ell}^{\mathrm{all}})
\right].
$$

选定这个二分后，父块 $C$ 还不能被两个子块解释的部分记为当前层的 residual atom：

$$
\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}})
=\Phi(C;\mathbf{y}_{\ell}^{\mathrm{all}})
-\Phi(L^\star;\mathbf{y}_{\ell}^{\mathrm{all}})
-\Phi(R^\star;\mathbf{y}_{\ell}^{\mathrm{all}}).
$$

因为 $L^\star$ 和 $R^\star$ 正好二分 $C$，单源项会相互抵消，所以上式也可以写成更直接的 EI 差：

$$
\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}})
=EI(\mathbf{x}_C;\mathbf{y}_{\ell}^{\mathrm{all}})
-EI(\mathbf{x}_{L^\star};\mathbf{y}_{\ell}^{\mathrm{all}})
-EI(\mathbf{x}_{R^\star};\mathbf{y}_{\ell}^{\mathrm{all}}).
$$

如果 $\gamma_C>0$，它表示：即使已经允许分别联合读取 $L^\star$ 和 $R^\star$，仍然有一部分信息只能通过把整个 $C$ 放在一起读出来。随后算法对 $L^\star$ 和 $R^\star$ 继续递归，直到子块为 singleton，或当前子块没有可继续解释的正协同。

因此，对根节点 $S$，最终得到一棵二分树和一组非负 atom。忽略数值容差时，这些 atom 闭合到总量：

$$
\Phi(S;\mathbf{y}_{\ell}^{\mathrm{all}})
=\sum_{C\in\mathcal{T}_{\ell}}\gamma_C(\mathbf{y}_{\ell}^{\mathrm{all}}),
$$

其中 $\mathcal{T}_{\ell}$ 是 lead $\ell$ 的贪婪二分树中被记录为正残差的节点集合。图里的 order 就是 $|C|$：order 2 是 pair residual，order 5 是五个 mode 必须一起读出的残差，`all 11 modes` 则是根节点没有被任何两个子块完全解释掉的全局残差。

需要强调的是，这个输出是 greedy hierarchy 下的非负残差分布，不是严格 Möbius 纯阶原子。它依赖每个节点选择到的二分路径，因此应解释为“沿这棵贪婪树，哪些模块集合仍需要联合读取”，而不是解释为唯一的高阶信息分解。

![UniCM all-mode PhiEID greedy decomposition](../../fig/unicm_phi_eid_greedy_decomposition.png)

*图 10. UniCM all-mode target 的 `PhiEID` 层级贪婪分解。左图为不同 order 的 greedy atom 随 lead 的堆叠分布，黑线为 atom sum；右图为按全部 seed/lead 平均值排序的 top 模块 heatmap。Peak lead 8 的细分分布见图 11。*

分解结果和上一节的总 `PhiEID` 完全闭合：每个 lead 上 `phi_atom_sum` 等于 `PhiEID`。峰值仍在 lead 8，`phi_atom_sum_mean=0.183958` bits。按全部 `3 seeds × 24 leads`、缺失视为 0 的平均贡献排序，最强模块是：

| Rank | Greedy module | order | mean atom bits | max atom bits | nonzero count |
|---:|---|---:|---:|---:|---:|
| 1 | ENSO + IOD + nino12 + nino3 + nino4 | 5 | 0.009840 | 0.041831 | 34/72 |
| 2 | ENSO + nino12 + nino3 + nino4 | 4 | 0.008738 | 0.050393 | 26/72 |
| 3 | all 11 modes | 11 | 0.006128 | 0.013373 | 71/72 |
| 4 | ENSO + nino3 + nino4 | 3 | 0.005791 | 0.049302 | 18/72 |
| 5 | ENSO + nino12 + nino3 | 3 | 0.005745 | 0.049969 | 21/72 |
| 6 | nino12 + nino3 | 2 | 0.005337 | 0.038964 | 26/72 |

把 peak lead 8 单独展开后，可以更清楚地看到分布集中度：

![UniCM lead-8 PhiEID atom distribution](../../fig/unicm_phi_eid_lead8_distribution.png)

*图 11. Lead 8 的 all-mode target `PhiEID` 层级 atom 分布。a 图显示按 seed mean 排序的 top 12 atoms，误差线为 checkpoint seed standard deviation，括号为占 lead-8 total PhiEID 的比例；b 图用 membership matrix 标出每个 atom 涉及的 source modes；c 图汇总不同 order 的 atom 质量；d 图给出 lead 8 的总量和 top atom 摘要。*

在 lead 8，top atom 是 `ENSO + IOD + nino12 + nino3 + nino4`，贡献 `0.032661` bits，占 total `PhiEID` 的 `17.8%`。Top 12 atoms 合计覆盖 `87.6%` 的 lead-8 Phi 质量。按阶数看，order 2 到 order 5 是主贡献区间，分别约占 `21%`、`22%`、`20%` 和 `18%`；order 6 以上主要是较小的跨块残差。

这个结果说明，系统级 `PhiEID` 的主要可解释层级仍集中在 ENSO 空间型态及 IOD 背景的嵌套组合上，而不是平均分散到全部 mode。`all 11 modes` 的 residual 几乎每个 seed/lead 都存在，但量级小，表示仍有弱的全局跨块残差。完整 atom 表见 `results/unicm_phi_eid_greedy_decomposition_cpu_bound4_n8192/unicm_phi_eid_greedy_atoms.csv`。



## IOD target 二源 Syn: 自身记忆与印度洋/ENSO 背景共同调制

作为对照，这里把 target 从 ENSO 换成 IOD，其他 full-history 最大熵干预口径保持一致：`8192` samples、checkpoint seeds `1, 2, 3`、lead `1..24`、intervention bound `[-4, 4]`，source modes 仍为 11 个 UniCM mode。图中展示按 IOD target 的 mean Syn 排名前 12 的 source pair，并保留同一组固定对照 pair。

| Rank | Source pair | mean Syn 1..24 | seed SD | positive seeds | joint EI | left EI | right EI |
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

![IOD target mode-pair Syn leads](../../fig/unicm_iod_mode_pair_syn_leads.png)

*图 12. IOD target 的二源 mode-pair Syn lead 曲线。实线为每个 lead 的 seed mean；同色浅虚线为该 pair 在 lead `1..24` 上的平均 Syn。*

IOD 结果的主信号与 ENSO target 不同：排名靠前的 pair 大多包含 IOD 自身历史，说明 IOD 未来状态的主要可预测部分仍由自身 12 个月历史提供；但 `IOD + SIOD`、`ENSO + IOD`、`IOD + nino4`、`NPMM + IOD` 等组合有正的额外二源增益。`IOD + SIOD` 在 lead 1 达峰，`ENSO + IOD` 和 `IOD + nino4` 在 lead 8 附近更强，说明印度洋内部结构和 ENSO/太平洋背景态主要影响短中期 IOD 演变。到 lead 15 后多数曲线贴近 0，不能支持长期稳定二源协同。

需要注意，`IOD + SIOD` 的 seed SD 仍接近均值，说明具体 rank 不宜过度解释。这里更稳妥的结论是：在当前 UniCM learned mechanism 中，IOD target 的二阶协同主要表现为 IOD 自身记忆与印度洋/ENSO 背景态的条件调制，而不是单个外部 mode 的独立强迫。

![Top mode-pair Syn curves](../../results/unicm_full_history_pair_syn_tm_degree1_n8192/fig/full_history_mode_pair_syn_top.png)

*图 13. 每个 target 按 1..24 lead 平均 Syn 排名前五的 source-mode pair 曲线；曲线为 checkpoint seed 均值。*

## 图表与数据索引

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
- Runge 多步 MLP+Ridge TM 重估 \(H=1\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H001_tm_reranked.png`
- Runge 多步 MLP+Ridge TM 重估 \(H=1\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H001_tm_reranked.csv`
- Runge 多步 MLP+Ridge TM 重估 \(H=10\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H010_tm_reranked.png`
- Runge 多步 MLP+Ridge TM 重估 \(H=10\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H010_tm_reranked.csv`
- Runge 多步 MLP+Ridge TM 重估 \(H=60\) top10 二阶候选图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H060_tm_reranked.png`
- Runge 多步 MLP+Ridge TM 重估 \(H=60\) top10 二阶候选表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_H060_tm_reranked.csv`
- Runge 多步 MLP+Ridge TM 重估候选结果目录：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_targeted`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) top10 汇总图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.png`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) top10 汇总表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked.csv`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) 复现次数表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_reranked_recurrence.csv`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) top10 距离表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_trends_top10_distances.csv`
- Runge 多步 MLP+Ridge TM 重估全 \(H\) 距离汇总表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/top10_order2_hyperedges_by_horizon_H001_H060_tm_trends_distance_summary.csv`
- Runge 多步 MLP+Ridge 代表超边强制 TM 趋势图：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.png`
- Runge 多步 MLP+Ridge 代表超边强制 TM 趋势表：`fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_targeted/forced_tm_edge_trends_H001_H060.csv`
- Runge 多步 MLP+Ridge 代表超边强制 TM 结果目录：`results/runge_slp_daily_1948_2026_20260628/mlp_tm_ei_lag04/results/runge/multistep_conditioned_ei_tm_forced_edges`

## 解释边界

- 本文只分析 frozen UniCM checkpoint 的 Modeformer learned mechanism，不使用 reanalysis 数据做预测复现，也不做单个历史事件归因。
- UniCM 的 overall EI 与 mode-pair Syn 使用 Gaussian log-det MI；这适合快速筛查，不等同于 transport-map PEID 的最终非线性分解。
- Syn 可以为负，表示 pair 的联合读数低于两个单源读数之和；除明确说明的 `PhiEID` 图外，本文不对 Syn 做非负截断。
- Overall EI 的 ENSO/nino 与 IOD target 均未通过 lead 排序的 seed 鲁棒性标准；因此应解释稳定方向和量级，不应解释单个 lead 的精细排序。
- Runge SLP 面板中的 PC-stable graph 仍不是原文 Fig. 4 的逐项复刻；当前 60 个 Varimax component 是在 1948-2026 扩展样本上重新拟合得到的，编号也不是官方固定标签，不能把未校准节点直接命名为确定气候过程。
- Runge 多步 MLP+Ridge 的 \(H=1,10,60\) 主图和跨 \(H=1,2,\ldots,10,15,20,30,40,50,60\) 汇总图均使用 TM 重估离散 top-1000 候选后的排序；离散 MI 只作为候选初筛，不作为最终 EI 数值。该口径仍未穷举全部 `102660` 条二阶候选，也没有经过 block-bootstrap 显著性筛选，不能替代前面的显著 PEID hypergraph 排名。
- 四条代表超边的强制 TM 趋势图绕过了离散 top-1000 初筛，但只验证指定边本身，不等同于全局候选穷举。
- Runge 超边的地理距离只按 component 空间中心的大圆距离计算；它是空间跨度诊断，不等同于两个模态完整 loading footprint 的物理距离。

## 参考文献

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
