# NYC Taxi：多区域预测与联合信息

## 结论

最新完成的 MGSTN 复现首先说明：**在论文兼容的 66 区、小时级 inflow/outflow 任务上，完整 MGSTN 的四项三随机种子均值全部落入原论文 ±5% 范围。** 在这个已经验收的非线性模型上，我们进一步对全部 66 区统一实施有限幅二阶 transport map（TM）干预，不再依赖无穷小 Jacobian。正式的 hurdle TM 将“是否有车流”和“正车流有多大”分开建模；结果显示，**近期数小时与日/周宏观节律的联合信息占目标级总 synergy 的 77.8%–83.9%，并在三个交通状态、三个模型种子中保持稳定。**

这个结论不是稀疏区域制造的假象。预先标记的 6 个稀疏区占全部区域数的 9.1%，但只贡献 **0.20%–0.30%** 的时间协同。普通二阶 TM 在 Randalls Island 出现 2 次显著非负性违规；hurdle TM 的 1,188 个 Syn 估计全部通过预设审计，因此 hurdle 版本被选为正式主估计，普通版本降为失败对照。

原有 30 分钟 pickup-only 实验说明的事情也保持不变：

1. **看全城，比只看本区更准。** Global Ridge 相对 Local Ridge 的测试 RMSE 降低 3.29%，并改善 59 个活跃区域中的 57 个。
2. **在原有任务和候选集中，更复杂的模型没有明显胜出。** Interaction Ridge 的误差最低，但只比 Global Ridge 低 0.021%；Extra Trees 和 MLP 也没有超过 Ridge。
3. **额外信息主要来自区域之间。** 在更正后的统一 EI 口径下，系统 $\Xi=8.1347$ bits，其中 91.0% 是跨区域联合信息，只有 9.0% 来自单一区域内部的多滞后联合读取。

因此，当前最稳妥的判断是：**曼哈顿出租车需求既有跨区域联合结构，也有更强、更稳健的跨时间尺度结构；下一小时的局部变化必须放进日/周生活节律中联合解释。** 原有 Ridge 与 MGSTN 的任务口径不同，不能直接用误差数值或 $\Xi$ 数值判定谁更好。

---

## 目录

- [0. MGSTN 论文精度复现](#0-mgstn-论文精度复现)
- [附录 G：旧局部 affine-TM 结果](#附录-g旧局部-affine-tm-结果)
- [1. 问题与数据](#1-问题与数据)
- [2. 主结果](#2-主结果)
- [3. 如何理解这些结果](#3-如何理解这些结果)
- [附录 A：实验设计](#附录-a实验设计)
- [附录 B：完整模型筛选](#附录-b完整模型筛选)
- [附录 C：EI 与 $\Xi$ 的定义和更正](#附录-cei-与-xi-的定义和更正)
- [附录 D：$\Xi$ 的详细结果与审计](#附录-dxi-的详细结果与审计)
- [附录 E：科学问题与后续实验](#附录-e科学问题与后续实验)
- [附录 F：复现资源与数据开放性](#附录-f复现资源与数据开放性)

---

## 0. MGSTN 论文精度复现

### 0.1 一句话结果

**复现成功：四项指标与论文均值的差异为 −0.2% 到 +4.5%，全部通过预先设定的 ±5% 验收线。**

![MGSTN 论文精度复现](../../fig/nyc_taxi_mgstn_reproduction.svg)

**附图 M1｜完整 MGSTN 的论文精度复现。** a，论文与本次三随机种子均值 ± 标准差；指标均为原始计数空间的每区域、每小时误差。b，本次均值相对论文均值的百分比差异，绿色区域为预先设定的 ±5% 验收带。c，三个随机种子的验证集 normalized MSE；圆点表示各自通过 early stopping 选中的最佳 epoch。所有图例均位于数据区域之外。

| 指标 | 原论文 | 本次复现 | 相对差异 | 验收 |
|---|---:|---:|---:|---:|
| Inflow MAE | 7.921 ± 0.026 | **8.029 ± 0.072** | +1.37% | 通过 |
| Inflow RMSE | 13.181 ± 0.078 | **13.157 ± 0.110** | −0.18% | 通过 |
| Outflow MAE | 8.215 ± 0.047 | **8.584 ± 0.157** | +4.49% | 通过 |
| Outflow RMSE | 15.407 ± 0.079 | **15.742 ± 0.283** | +2.17% | 通过 |

### 0.2 复现任务与数据口径

目标论文为 Shi 等发表于 *Applied Soft Computing* 的 [“MGSTN: A multi-granularity spatio-temporal network for citywide traffic forecasting”](https://doi.org/10.1016/j.asoc.2026.115896)（2026）。本次重新读取 2023 年 38,310,226 条 Yellow Taxi 原始行程，构建：

- 66 个有效 Manhattan Taxi Zones；从官方 Manhattan 69 区中排除无活动的 103、104、105 区。
- 8,760 个小时级时间点。
- Inflow 为跨区行程在终点区域、dropoff 时刻的计数；outflow 为起点区域、pickup 时刻的计数。
- 只保留起点和终点都属于这 66 区且起终点不同的行程。
- 最近 7 小时、前 5 日同小时、前 7 周同小时三个输入分支。
- 样本按时间顺序 70%/20%/10% 切分；标准化参数只由训练目标时间点估计。

论文没有明确说明是否保留区域边界行程。我们同时审计了三种口径；“66 区内部跨区流”的 HA 四项误差与论文 HA 最接近，差异约 2.2%–5.8%，因此在训练前固定为正式口径。这一选择没有读取 MGSTN 测试结果。

### 0.3 完整模型

![MGSTN 模型与信息分解框架](../../fig/nyc_taxi_mgstn_architecture.svg)

**框架图 1｜复现的完整 MGSTN 与事后信息读出。** 上部是预测模型：近期、日周期和周周期分别进入独立 STN 分支，每个分支同时抽取空间、时间和环境属性表示，再进行多粒度融合，输出 66 区下一小时 inflow/outflow。下部是训练完成后的有限幅信息读出，不参与模型训练：固定检查点，对历史车流实施经验分布干预，以二阶 hurdle TM 估计目标级时间与空间 Syn。

MGSTN 可以直观地理解为“三条时间支路、每条支路同时看空间和时间、最后再合流”：

1. **三种时间粒度。** Recent 分支读取连续 7 小时，Daily 分支读取前 5 日同小时，Weekly 分支读取前 7 周同小时。三支路参数互不共享，使模型可以分别学习短期惯性、日周期与周周期。
2. **每条支路的空间编码。** 距离图上的两层 residual GCN 传播邻近区域的流量；训练期流量相关图与区域类型进入两层 typed RGCN，提供不局限于地理相邻的语义联系。二者拼接后形成空间表示。
3. **每条支路的时间编码。** Non-Stationary Transformer 先按区域对序列平稳化，再由原始均值和标准差生成 de-stationary attention 的 $\tau$ 与 $\boldsymbol{\delta}$，经两层、四头注意力恢复状态依赖的时间关系。
4. **属性与融合。** 小时、星期、工作日、温度、降水和天气状态先进入属性网络，再与空间、时间表示融合。三个 96 维分支状态再次融合，并和目标时刻属性一起映射为 66 个区域各两个输出。

固定优化设置与论文一致：Adam、learning rate `1e-4`、weight decay `1e-3`、batch size 64、最多 100 epochs、early-stopping patience 10。论文未公开隐藏宽度、层数和注意力头数；本次只在验证集登记范围内比较 64 与 96 隐藏维度，最终固定 96 维、2 层 NST、2 层 GCN/RGCN、4 个注意力头，共 1,796,376 个参数。

三个随机种子的最佳 epoch 分别为 33、29、22；M1 Max 上正式三种子训练合计约 49.9 分钟。论文 RTX 4090D 报告约 5.87 分钟/seed，因此本机慢约 2.8 倍，但计算成本仍然较低。

### 0.4 正式方法：全区域有限幅二阶 hurdle TM

正式分析对 66 个区域使用完全相同的计算流程。对区域 $i$，近期块 $\mathbf{R}_i$ 包含最近 7 小时，宏观块 $\mathbf{M}_i=(\mathbf{D}_i,\mathbf{W}_i)$ 合并前 5 日同小时和前 7 周同小时；目标 $\mathbf{Y}_i$ 是该区域下一小时的 inflow/outflow。时间协同定义为

$$
\operatorname{Syn}^{\mathrm{time}}_i
=I(\mathbf{R}_i;\mathbf{M}_i\mid\mathbf{Y}_i),
\tag{0.1}
$$

空间对照则把本区完整历史 $\mathbf{H}_i$ 与其余 65 区历史 $\mathbf{H}_{-i}$ 配对：

$$
\operatorname{Syn}^{\mathrm{space}}_i
=I(\mathbf{H}_i;\mathbf{H}_{-i}\mid\mathbf{Y}_i).
\tag{0.2}
$$

两组源都从训练期经验窗口独立 bootstrap，分别经训练期 PCA 压缩为 2 维；固定 MGSTN 产生目标，并加入同一检查点的验证残差噪声。随后在 70% 干预样本上拟合二阶自回归 triangular TM，在留出的 30% 上估计条件互信息。每个区域、状态和模型种子使用 4,096 次有限幅干预。这里直接积分经验尺度上的非线性响应，**不计算 Jacobian，也不把某个区域换成不同阶数或不同样本量。**

hurdle 表示把每个计数拆为 $\mathbb{1}(x>0)$ 与 $\log(1+x)$ 两部分，再统一执行同一 PCA 与二阶 TM。它保留“零/非零状态切换”和“正流量幅度变化”两类机制，避免稀有的正样本在标准差归一化后被异常放大。

汇总量为

$$
q_{\mathrm{time}}
=\frac{\sum_i\operatorname{Syn}^{\mathrm{time}}_i}
{\sum_i\operatorname{Syn}^{\mathrm{time}}_i+
 \sum_i\operatorname{Syn}^{\mathrm{space}}_i}.
\tag{0.3}
$$

它是两个**目标级、二分块、有限幅**问题的相对量，不再声称是旧 66 块全输出 affine 分解的逐项替代。这里的 Syn 是冻结 MGSTN 所表达的预测机制，不是仅凭观测数据识别出的物理因果效应。

### 0.5 正式主结果

![NYC Taxi 社会系统多时间尺度耦合主图](../../fig/nyc_taxi_social_multiscale_main.svg)

**主图 1｜预测精度复现、区域分布与跨时间尺度协同地图。** a，复现模型的四项测试误差与三随机种子验证轨迹。b，正式 hurdle 二阶 TM 中，时间 Syn 占时间与空间目标级 Syn 之和的比例；柱为三种子均值，空心点为单种子。c，全部 66 区的 recent–macro Syn 分布；每点为区域的三种子均值，黑线表示四分位范围与中位数。d，三个交通状态下的全 66 区 Syn 地图，共用同一线性色标。方法名称、估计器阶数与零膨胀处理等技术信息只在图注中说明，不写入图面。

三个状态的正式结果为：

| 交通状态 | 时间 Syn 占比 | 三种子 SD | 稀疏 6 区贡献 |
|---|---:|---:|---:|
| 工作日高峰 | **83.9%** | 1.1 个百分点 | **0.30%** |
| 周末中午 | **82.3%** | 1.4 个百分点 | **0.20%** |
| 雨天高需求 | **77.8%** | 1.9 个百分点 | **0.20%** |

最简单的解释是：**MGSTN 要预测下一小时，不能只看最近几小时，也不能只查日/周周期；它必须判断“当前短时变化处在什么宏观生活节律中”。** 工作日、周末和雨天的绝对权重会变化，但三种状态下时间协同都占约八成，说明这不是某个单一时段的偶然模式。

### 0.6 hurdle 是否带来明显改进

答案是**有，但改进体现在估计有效性，而不是把 Syn 数值做大。** 普通标准化二阶 TM 在 seed 1 的 Randalls Island 上出现 2 个显著非负性违规，分别位于周末中午与雨天高需求状态；最小值为 −0.0729 bit，低于预先声明的 −0.05 bit 容差。按照 PEID 的非负定义，这两个普通 TM 结果被明确判为失败，没有裁剪为零，也没有进入汇总。

hurdle 二阶 TM 在同样的 66 区、3 状态、3 种子与 4,096 干预样本下，1,188 个 Syn 估计全部通过审计。其最小原始值为 −0.0147 bit；110 个位于 $[-0.05,0)$ 的估计按预声明的数值零记录，低于容差者为 0 个。由此，hurdle 不是可有可无的装饰，而是让零膨胀交通计数进入同一有限幅 TM 框架的必要稳健化步骤。

### 0.7 空间分布与证据边界

![NYC Taxi 跨时间尺度耦合地图](../../fig/nyc_taxi_temporal_coupling_map.svg)

**扩展图 M2｜全 66 区有限幅 recent–macro 协同地图。** 这是主图 1d 的独立放大版本：a–c 分别为工作日高峰、周末中午和雨天高需求；颜色是区域级 hurdle 二阶 TM Syn 的三种子均值，三图共享线性色标。它表示冻结模型中“近期变化与日/周节律必须联合读取”的信息量，不是客流量、预测误差或因果效应。

与旧 Jacobian 地图不同，新地图不再由 Highbridge Park、Randalls Island 等低活动区主导。高值主要分布于 Murray Hill、Union Sq、Lenox Hill East、East Chelsea 和 Midtown 等真实高活动区域，北部公园及边缘区接近零；三种状态的区域排名 Spearman 相关为 **0.992–0.996**，而雨天的全城强度下降。这与稀疏 6 区仅贡献 0.20%–0.30% 的汇总结果一致，直接排除了“前三个稀疏区抬高全局时间协同”的解释。

当前证据仍有限制：PCA 每块只保留 2 个主成分，二阶 TM 只表达至二次曲率，4,096 次干预对应有限的尾部支持。因此，正式结论是“在可审计的有限幅二阶近似下，MGSTN 的目标级联合信息以 recent–macro 时间耦合为主”，而不是模型全部高阶信息容量的封闭形式证明。

## 附录 G：旧局部 affine-TM 结果

以下结果保留用于方法敏感性对照，不再作为论文主结论。它们依赖状态点附近的无穷小 Jacobian，并曾在低活动区域产生异常高值；这正是本轮改用有限幅 hurdle 二阶 TM 的原因。

### G.1 旧 MGSTN EI 与 $\Xi$ 计算

#### 0.4.1 局部 TM 转移

MGSTN 的 66 个区域各有 38 个流量历史坐标：recent 为 $7\times2=14$ 维，daily 为 $5\times2=10$ 维，weekly 为 $7\times2=14$ 维，因此干预源总计 $66\times38=2{,}508$ 维；目标是 132 维下一小时 inflow/outflow。对每个已训练检查点和交通状态 $\mathbf{x}^{*}$，用一阶局部转移

$$
\mathbf{y}_{t+1}\approx f_{\boldsymbol{\theta}}(\mathbf{x}^{*})
+\mathbf{J}^{*}(\mathbf{x}_{t}-\mathbf{x}^{*})+\boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_{e}),
\tag{G.1}
$$

其中 $\mathbf{J}^{*}$ 是 $132\times2{,}508$ Jacobian，$\boldsymbol{\Sigma}_{e}$ 由同一检查点的 1,516 个验证残差通过 Ledoit–Wolf 收缩估计。天气和日期属性在每个分析中心固定，只干预相互独立的标准化流量历史 $\Delta\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$。这是 affine triangular transport map 的闭式读出，与 Brain/Earth 实验的仿射 log-determinant 口径一致。

记 $S$ 为一组源坐标，$\mathbf{J}^{*}_{S}$ 为对应列，$\boldsymbol{\Sigma}_{y}=\boldsymbol{\Sigma}_{e}+\mathbf{J}^{*}\mathbf{J}^{*\mathsf T}$。在同一个完整干预分布下，

$$
\operatorname{EI}(S\to\mathbf{y})
=\frac{1}{2}\log_{2}
\frac{\det\boldsymbol{\Sigma}_{y}}
{\det\!\left(\boldsymbol{\Sigma}_{y}-\mathbf{J}^{*}_{S}\mathbf{J}^{*\mathsf T}_{S}\right)}.
\tag{G.2}
$$

式 (G.2) 的分母仍保留其余源的变化，因此联合 EI 与各部分 EI 可在同一干预下相减，避免早先“每次单独重定义实验分布”造成的不可比问题。

#### 0.4.2 两层分解

先把第 $i$ 个区域的 38 维历史作为一个整体，定义跨区域联合信息

$$
\Xi_{\mathrm{region}}
=\operatorname{EI}(\mathrm{all}\to\mathbf{y})
-\sum_{i=1}^{66}\operatorname{EI}(i\to\mathbf{y}).
\tag{G.3}
$$

再把每个区域拆为 recent、daily、weekly 三组，定义区域 $i$ 内部的多时间尺度联合信息

$$
\Xi_{\mathrm{time},i}
=\operatorname{EI}(i\to\mathbf{y})
-\sum_{g\in\{r,d,w\}}\operatorname{EI}((i,g)\to\mathbf{y}).
\tag{G.4}
$$

于是最细分辨率的总联合信息满足

$$
\Xi_{\mathrm{fine}}
=\operatorname{EI}(\mathrm{all}\to\mathbf{y})
-\sum_{i=1}^{66}\sum_{g\in\{r,d,w\}}\operatorname{EI}((i,g)\to\mathbf{y})
=\Xi_{\mathrm{region}}+\sum_{i=1}^{66}\Xi_{\mathrm{time},i}.
\tag{G.5}
$$

为回答“宏观节律本身”和“微观变化放进宏观背景”各占多少，进一步固定一个有领域含义的层级：先将 daily 与 weekly 合成宏观节律块 $M=(D,W)$，再把 recent 与 $M$ 合成完整时间历史。于是区域 $i$ 的时间尺度 $\Xi$ 可严格写成两个非负二分块 Syn：

$$
\Xi_{\mathrm{macro},i}
=\operatorname{EI}((D_i,W_i)\to\mathbf{y})
-\operatorname{EI}(D_i\to\mathbf{y})
-\operatorname{EI}(W_i\to\mathbf{y}),
\tag{G.6}
$$

$$
\Xi_{\mathrm{micro\text{-}macro},i}
=\operatorname{EI}((R_i,D_i,W_i)\to\mathbf{y})
-\operatorname{EI}(R_i\to\mathbf{y})
-\operatorname{EI}((D_i,W_i)\to\mathbf{y}),
\tag{G.7}
$$

并满足

$$
\Xi_{\mathrm{time},i}
=\Xi_{\mathrm{macro},i}+\Xi_{\mathrm{micro\text{-}macro},i}.
\tag{G.8}
$$

这里的 $\Xi$ 是**已训练 MGSTN 所表达的局部预测机制**，不是仅凭观察数据就能识别出的物理因果效应。

### G.2 旧分解结果：局部多时间尺度耦合

![旧局部 affine-TM 分解](../../fig/nyc_taxi_mgstn_ei_decomposition.svg)

**旧附图 G1｜局部 affine-TM 分解。** 该图及下列百分比仅用于与有限幅结果比较。层级公式对应式 (G.6)–(G.8)，不再作为论文主结论。

这张图把 Taxi 作为社会系统代表所需要的两层证据放到了一起：

1. **预测模型已经通过复现验收。** 四项指标与论文均值的差异为 −0.2% 至 +4.5%，三个种子收敛到接近的验证误差，因此后面的机制读出不是建立在一个明显失配或未收敛的模型上。
2. **时间尺度是非加性联合信息的主体。** 工作日高峰、周末中午和雨天高需求中，区域内跨时间尺度部分分别占 fine $\Xi$ 的 **93.0%、92.4% 和 93.7%**；空间联合项只保留为灰色余量，不再作为主图叙事。
3. **真正占主导的是微观变化与宏观节律的耦合。** 在三个真实状态、三个种子合并后，$\Xi_{\mathrm{micro\text{-}macro}}$ 平均占区域内时间 $\Xi$ 的 **89.4% ± 5.0%**，daily–weekly 宏观节律自身的耦合占 **10.6% ± 5.0%**。分状态看，recent–macro 占比从工作日高峰的 84.5% 上升到周末中午的 90.1% 和雨天高需求的 93.6%。
4. **recent 同时需要日和周两个参照。** 两两诊断中，真实状态平均约 46.0% 来自 recent–daily，39.0% 来自 recent–weekly，14.9% 来自 daily–weekly；三种状态下三组均存在，但相对权重会改变。

因此，最简洁的领域解释不是“模型记住了几个周期”，而是：**模型先利用日与周信息识别城市所处的宏观生活节律，再用最近数小时判断当前状态相对该节律的位置与偏离，二者必须联合才能预测下一小时。** 这使 Taxi 部分与 Brain、Earth 两类系统形成互补：它展示的是社会活动、制度时间表与短时扰动之间的跨尺度耦合。

#### G.2.1 旧闭式分解的非负性和数值审计

所有 12 个 seed–state 单元同时审计了系统跨区域 $\Xi$、66 个区域内跨尺度 $\Xi$、66 个目标区跨区域 $\Xi$，以及每个目标的 2,145 个区域对 synergy。预先声明容差为 **$10^{-8}$ bit**，没有使用 `max(0, Syn)` 或其他裁剪。最小估计为 $-5.16\times10^{-12}$ bit；93,013 个负值均位于 $[-10^{-8},0)$，按数值零记录；低于容差的显著负值为 **0 个**。式 (G.5) 的最大绝对恒等误差为 $4.55\times10^{-13}$ bit。

新增时间层级分解沿用同一容差与干预。$\Xi_{\mathrm{macro}}$、$\Xi_{\mathrm{micro\text{-}macro}}$ 及三组两两 Syn 的显著负值均为 **0 个**；式 (G.8) 的最大绝对恒等误差为 $3.55\times10^{-15}$ bit。没有用裁剪制造图中的百分比。

### G.3 旧跨时间尺度耦合空间分布

旧地图使用式 (G.7) 的 Jacobian 读出和对数色标。它已经被 0.7 节的有限幅 hurdle 地图替代，不再单独展示，以免与正式主图混淆。

空间分布并不均匀，而且具有明显的跨状态稳定性：三种状态的区域排名 Spearman 相关为 **0.81–0.96**。Highbridge Park、Marble Hill 和 Inwood Hill Park 在三个状态中持续处于最高一组；工作日高峰还出现 Roosevelt Island 的高值。大多数中城和下城区域集中在约 $10^{-1}$ bits 的数量级。这个结果说明，micro–macro 联合读取不是全城均匀增加的模型能力，而是集中在一部分区域；但高值区域中包含公园和边缘小区，因此现阶段应把它解释为**模型机制敏感性的空间热点**，而不能直接解释成这些地区具有更强的客流、经济活动或物理因果作用。后续应把耦合强度与区域流量、残差方差和样本支持度做控制比较，以区分真实的跨尺度社会节律与低活动区域的标准化效应。

### G.4 旧方法的证据边界

这是**功能与精度复现**，不是逐行代码复现。论文未公开官方实现和完整架构参数；原 Foursquare check-in 入口也不可稳定取得。因此：

- 天气改用 Open-Meteo 的 LaGuardia 小时级历史天气；
- typed semantic graph 保持存在，但节点类型由训练期 mobility rhythm 聚类得到，而不是 Foursquare POI 主类别；
- 语义图只用训练期流量相关性构建，避免测试期泄漏。

尽管存在这些替代，四项测试指标仍全部落入论文 ±5%，说明当前模型已经足以作为后续非线性时空预测与区域联合信息研究的参照。它不能证明我们恢复了作者未公开的所有实现细节。

局部 affine TM 是当前 2,508 维源空间中计算可行、且能严格审计非负性的完整版本；它没有积分远离中心的强非线性曲率，也可能漏掉 XOR 类纯高阶效应。因此，当前结论是“MGSTN 在这些交通状态附近怎样整合信息”，不是网络的全局信息容量。下一步若要增强因果解释，应增加更密集的真实状态中心，并对天气、事件和区域流量实施显式反事实干预。

### G.5 与原有 Ridge 实验的关系

MGSTN 和后文 Ridge 实验不是同一个受控比较：MGSTN 使用 66 区、小时级、内部跨区 inflow/outflow；Ridge 使用 69 区、30 分钟、pickup-only。因此，`13.157` 不能与 Ridge 的标准化对数 RMSE `0.538` 直接比较。后续若比较二者或计算 MGSTN 的联合信息，必须先统一数据、目标、切分和评价尺度。

---

## 1. 问题与数据

我们把 2023 年 NYC Yellow Taxi 行程聚合为 **69 个曼哈顿区域、每 30 分钟一个时间点**的上车量序列，用过去 0.5 小时到 1 周的历史预测下一半小时。

这里依次回答三个问题：

1. 其他区域的历史能否改善一个区域的预测？
2. 更复杂的显式交互或非线性模型是否更合适？
3. 多个区域联合读取时，是否产生超出各自单独信息之和的有效信息？

三类证据不能混为一谈：预测增益表示“其他区域有用”，时序置乱表示“模型依赖区域间对齐”，只有统一干预分布下的 $\Xi$ 才表示“联合读取产生额外有效信息”。

数据规模如下：

| 项目 | 数值 |
|---|---:|
| 原始行程 | 38,310,226 条 |
| 保留的曼哈顿上车行程 | 33,806,164 条 |
| 空间单元 | 69 个 Taxi Zones |
| 时间分辨率 | 30 分钟 |
| 完整时间步 | 17,520 |
| 训练 / 验证 / 测试窗口 | 12,768 / 1,488 / 2,928 |

---

## 2. 主结果

![NYC Taxi 多区域预测与联合信息主图](../../fig/nyc_taxi_main_results.svg)

**主图 3｜Ridge 基线中，多区域历史有用，而联合信息主要来自区域之间。** a，Local、Global 和 Interaction Ridge 的测试标准化对数 RMSE。Global Ridge 明显优于 Local Ridge，Interaction Ridge 与 Global Ridge 几乎相同。b，Global Ridge 相对 Local Ridge 的逐区域 RMSE 改善；59 个活跃区域中有 57 个改善，中位数为 9.75%。每个点代表一个区域，箱线表示中位数和四分位范围。c，联合 EI 为 29.2635 bits，其中 21.1288 bits 可由标量单源 EI 之和解释，剩余的系统 $\Xi$ 为 8.1347 bits；系统 $\Xi$ 又由 7.4038 bits 跨区域项和 0.7309 bits 区域内跨滞后项构成。模型使用固定时间切分；Ridge 为确定性拟合，因此不报告随机种子误差条。

### 2.1 预测层面的结论

Global Ridge 同时读取 69 个区域的历史，相对只读取本区历史的 Local Ridge：

- 整体 RMSE 下降 3.29%；
- WAPE 与原始 MAE 均下降 8.42%；
- 57/59 个活跃区域改善；
- 逐区域 RMSE 改善的中位数为 9.75%。

这说明跨区域输入的价值广泛存在，并非只由少数中心区造成。

Interaction Ridge 的 RMSE 最低，但相对 Global Ridge 只改善 0.021%。因此，它是本轮的数值最优模型，却不是一个有实质领先幅度的赢家。现阶段更合理的主模型仍是 **Global Ridge**：精度几乎相同，结构更简单，而且能够进行严格、可审计的 affine transport-map（TM）信息分解。

### 2.2 联合信息层面的结论

Global Ridge 的联合 EI 为 29.2635 bits，其中系统 $\Xi$ 为 8.1347 bits，占联合 EI 的 27.8%。进一步分解得到：

- 跨区域 $\Xi=7.4038$ bits，占系统 $\Xi$ 的 91.0%；
- 区域内跨滞后 $\Xi=0.7309$ bits，占 9.0%。

直观地说：模型最有价值的“组合读取”主要是**同时看多个区域**，而不是只在同一区域里同时看多个历史时刻。

---

## 3. 如何理解这些结果

### 3.1 当前证据支持什么？

- 曼哈顿区域之间存在稳定、可预测的共同结构。
- 联合信息具有实质量级，并且主要发生在区域之间。
- 简单的全局线性模型已捕获大部分可预测结构。
- Midtown、Times Square、Penn Station、Upper East Side 和 West Chelsea 等区域是较强的联合信息接收端。

### 3.2 当前证据不支持什么？

- 跨区域置乱后的误差暴涨不能直接叫作 synergy；它同时破坏共同周期、冗余、唯一信息和联合信息。
- Interaction Ridge 的乘积系数不能直接解释为 PEID 原子。
- 分目标 $\Xi$ 排名不能确定唯一的来源区域组合，也不能证明交通传播方向或因果关系。
- Yellow Taxi 上车量是已实现行程，不等于不受车辆供给、拥堵和运营规则约束的潜在需求。
- 结果来自 2023 年和一次时间切分，尚不能声称跨年份或跨城市普适。

### 3.3 一句话的模型选择建议

**用 Global Ridge 做当前主模型和正式 $\Xi$ 读出；保留 Interaction Ridge 用于发现候选区域对；只有在非线性 TM 验证后，才讨论更复杂模型的正式 synergy。**

---

# 附录

## 附录 A：实验设计

### A.1 数据构造与开放性

数据来自 NYC Taxi and Limousine Commission（TLC）公开发布的 2023 Yellow Taxi Trip Records。每条行程用 `tpep_pickup_datetime` 确定半小时时间箱，用 `PULocationID` 确定上车区域，只保留曼哈顿上车记录。

TLC 提供可直接下载的月度 Parquet 文件、数据字典和 Taxi Zone lookup。本项目据此能够复现实验。官方页面没有给出可由本报告代为概括的明确开源许可证，因此“公开可下载”不应自动表述为“可不受限制地再分发”；正式发布衍生数据前仍需核对当时条款。

### A.2 输入、目标与时间切分

令 $\mathbf{x}_t\in\mathbb{R}^{69}$ 为时刻 $t$ 各区域经 `log1p` 变换后的上车量。滞后集合为

$$
\mathcal{L}=\{1,2,3,6,12,48,336\},
\tag{A.1}
$$

即 0.5、1、1.5、3、6、24 和 168 小时。预测任务为

$$
\widehat{\mathbf{x}}_t=
f_{\boldsymbol\theta}\!\left(
\mathbf{x}_{t-1},\mathbf{x}_{t-2},\mathbf{x}_{t-3},
\mathbf{x}_{t-6},\mathbf{x}_{t-12},\mathbf{x}_{t-48},
\mathbf{x}_{t-336},\mathbf{c}_t
\right),
\tag{A.2}
$$

其中 $\mathbf{c}_t\in\mathbb{R}^{4}$ 是日周期和周周期的正余弦特征。`log1p` 后的均值和标准差只用训练集估计。

时间切分固定为：1–9 月训练、10 月验证、11–12 月冻结测试。Ridge 正则强度只在验证集从 $\{0.1,1,10,100\}$ 中选择。

### A.3 模型条件

| 模型 | 输入与主要设置 | 角色 |
|---|---|---|
| Persistence | 前一半小时 | 短期基线 |
| Daily seasonal | 前 24 小时同时刻 | 日周期基线 |
| Weekly seasonal | 前 168 小时同时刻 | 周周期基线 |
| Local Ridge | 每个目标的 7 个自身滞后 + 4 个日历特征 | 单区域对照 |
| Global Ridge | $69\times7$ 个区域滞后 + 4 个日历特征，共 487 维 | 加性跨区域主模型 |
| Interaction Ridge | Global Ridge + 20 个高流量区域最近滞后的 190 个两两乘积 | 显式交互候选发现 |
| Extra Trees | 120 棵树，`min_samples_leaf=2`，`max_features=0.5` | 阈值型非线性模型 |
| MLP | 两层 128/64 ReLU，early stopping | 分布式非线性模型 |

Global Ridge 与 Interaction Ridge 均选择 $\alpha=100$；Local Ridge 的逐区域中位数为 $\alpha=10$。Extra Trees 和 MLP 使用相同随机种子 0、1、2。

### A.4 指标和受控比较

- 主指标：测试集标准化对数 RMSE，越低越好。
- 次指标：原始尺度 WAPE 与每区域、每半小时 MAE。
- 依赖诊断：保留目标区域自身历史，将其他区域测试历史整体循环平移 37 个半小时，计算 RMSE 增幅。
- 活跃区域：训练期平均每半小时至少 1 条行程，共 59 个。

所有模型共享相同数据、特征尺度、时间切分和指标。模型容量无法完全匹配，因此模型族差异是探索性筛选，不是严格的容量等价检验。

---

## 附录 B：完整模型筛选

| 模型 | 测试 log-RMSE ↓ | WAPE ↓ | 原始 MAE ↓ | 跨区域置乱损失 ↑ |
|---|---:|---:|---:|---:|
| Persistence | 0.7303 | 21.20% | 6.29 | 0 |
| Daily seasonal | 0.8270 | 30.95% | 9.18 | 0 |
| Weekly seasonal | 0.7925 | 30.29% | 8.99 | 0 |
| Local Ridge | 0.5563 | 18.49% | 5.49 | 0 |
| Global Ridge | 0.5380 | 16.93% | 5.02 | 299.4% |
| **Interaction Ridge** | **0.5379** | **16.68%** | **4.95** | 311.2% |
| Extra Trees | 0.5454 ± 0.0002 | 18.76% ± 0.01% | 5.57 ± 0.00 | 426.0% ± 0.7% |
| MLP | 0.5482 ± 0.0032 | 19.84% ± 0.67% | 5.89 ± 0.20 | 397.1% ± 10.0% |

![NYC Taxi 完整模型筛选](../../fig/nyc_taxi_synergy_model_screen.svg)

**图 A1｜完整模型筛选与跨区域对齐诊断。** a，测试标准化对数 RMSE；b，测试 WAPE；c，误差与跨区域置乱损失。Interaction Ridge 的误差最低，但与 Global Ridge 几乎重合。非线性模型报告 3 个种子的均值 ± 标准差；确定性基线只拟合一次。置乱损失用于确认模型依赖区域间时间对齐，不是正式 $\Xi$。

更细的比较为：

- Global Ridge 相对 Local Ridge：RMSE 降低 3.29%，WAPE/MAE 降低 8.42%。
- Interaction Ridge 相对 Global Ridge：RMSE 降低 0.021%，WAPE/MAE 降低约 1.50%；59 个活跃区域中 38 个改善，逐区域改善中位数为 0.19%。
- Global Ridge 逐区域改善均值为 8.85%，中位数为 9.75%，四分位范围为 6.22%–12.17%，范围为 −11.48%–16.74%。
- Extra Trees 和 MLP 均未超过两个全局 Ridge；更高模型容量没有自动转化为更低误差。

跨区域置乱损失很大，说明所有全区域模型确实读取了同步结构。但固定的 18.5 小时平移也会破坏日内相位、共同周期和其他统计关系，因此该数值只回答“模型是否依赖对齐”。

---

## 附录 C：EI 与 $\Xi$ 的定义和更正

### C.1 为什么使用 Global Ridge？

Interaction Ridge 包含由两个原始区域状态确定的乘积项。若把这些乘积和原始特征都当成相互独立的干预源，就会破坏确定性约束并改变科学问题。Global Ridge 与 Interaction Ridge 的 RMSE 只差 0.021%，同时具有清晰的线性转移，因此本轮用它作为严格的 affine-TM 读出。

日历变量作为固定背景，不进入源分解。正式源为 69 个区域乘以 7 个滞后，共 483 个标准化标量变量；目标是下一半小时 69 维联合区域状态。

### C.2 统一完整干预下的 EI

Global Ridge 的随机转移写为

$$
\mathbf{y}=\mathbf{A}\mathbf{x}+\boldsymbol\varepsilon,
\qquad
\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\qquad
\boldsymbol\varepsilon\sim
\mathcal{N}(\mathbf{0},\boldsymbol\Sigma_{\varepsilon}).
\tag{C.1}
$$

对任意源子集 $S$，其补集为 $S^c$。所有联合 EI 和部分 EI 必须来自同一个完整最大熵独立干预分布：

$$
\begin{aligned}
\boldsymbol\Sigma_y
&=\boldsymbol\Sigma_{\varepsilon}+\mathbf{A}\mathbf{A}^{\mathsf T},\\
\boldsymbol\Sigma_{\mathbf{y}\mid \mathbf{x}_S}
&=\boldsymbol\Sigma_{\varepsilon}
+\mathbf{A}_{S^c}\mathbf{A}_{S^c}^{\mathsf T},\\
EI(S\to\mathbf{y})
&=\frac{1}{2}\log_2
\frac{\det\boldsymbol\Sigma_y}
{\det\boldsymbol\Sigma_{\mathbf{y}\mid \mathbf{x}_S}}.
\end{aligned}
\tag{C.2}
$$

式（C.2）中，补集源没有被置零或从动力学中删除，而是继续变化，并在计算 $I(\mathbf{x}_S;\mathbf{y})$ 时被边缘化。先前出现负值的计算恰恰违反了这一点：它为每个子集生成了不同的目标分布，使联合 EI 和部分 EI 不再能直接相减。当前实现与仓库 Brain 和 Earth 实验的完整干预口径一致。

### C.3 非负性

将 483 个独立干预源记为 $x_1,\ldots,x_{483}$，则

$$
\begin{aligned}
\Xi_{\mathrm{system}}
&=EI(\mathbf{x}\to\mathbf{y})
-\sum_{j=1}^{483}EI(x_j\to\mathbf{y})\\
&=\sum_{j=1}^{483}H(x_j\mid\mathbf{y})
-H(\mathbf{x}\mid\mathbf{y})\\
&=TC(\mathbf{x}\mid\mathbf{y})\ge 0.
\end{aligned}
\tag{C.3}
$$

所以非负性不是靠裁剪得到的，而是独立源干预与统一目标分布共同保证的。数值实现声明 $10^{-10}$ bits 的非负容差；落在 $[-10^{-10},0)$ 的值只可视为数值零，低于该阈值则实验显式失败。当前没有负容差违规，也没有应用 `max(0, Xi)` 或其他静默投影。

### C.4 区域与滞后层级分解

把同一区域的 7 个滞后合成一个区域历史组，有

$$
\Xi_{\mathrm{system}}
=\Xi_{\mathrm{cross\text{-}region}}
+\sum_{z=1}^{69}\Xi_{\mathrm{within\text{-}region},z}.
\tag{C.4}
$$

式（C.4）是严格恒等式：第一项表示区域历史组之间的联合增量，第二项表示各区域内部多个滞后的联合增量。

---

## 附录 D：$\Xi$ 的详细结果与审计

### D.1 数值分解

| 量 | 数值（bits） | 含义 |
|---|---:|---|
| 联合 EI | 29.2635 | 483 个滞后共同指向 69 维未来的有效信息 |
| 标量单源 EI 之和 | 21.1288 | 483 个标量源分别与同一未来的 EI 之和 |
| 区域历史组 EI 之和 | 21.8597 | 69 个区域历史组分别与同一未来的 EI 之和 |
| 跨区域 $\Xi$ | **7.4038** | 联合 EI 超出区域组 EI 之和的部分 |
| 区域内跨滞后 $\Xi$ | **0.7309** | 各区域内部 7 个滞后的联合增量之和 |
| 系统 $\Xi$ | **8.1347** | 联合 EI 超出全部标量单源 EI 之和的部分 |

分解满足

$$
8.1347=7.4038+0.7309,
\tag{D.1}
$$

数值闭合误差为 $1.8\times10^{-15}$ bits。使用完整后验协方差独立计算 $TC(\mathbf{x}\mid\mathbf{y})$ 得到 8.13472304 bits，与 EI 差值法相差 $7.3\times10^{-8}$ bits。

![更正后的 NYC Taxi Ridge EI 与 Xi 分解](../../fig/nyc_taxi_corrected_xi_decomposition.svg)

**图 A2｜$\Xi$ 详细分解、位置与稳健性。** a，联合 EI 与系统 $\Xi$，以及系统 $\Xi$ 的跨区域/区域内分解。b，系统 $\Xi$ 较高的标量目标区域；这些值用于定位接收端，因目标之间存在相关结构，不能相加还原 69 维联合目标。c，各滞后在 69 个区域上的标量单源 EI 之和；它不是系统 $\Xi$ 的唯一可加归因。d，残差协方差 ridge 从 $10^{-8}$ 扫描到 $10^{-4}$；系统 $\Xi$ 最大漂移为 0.16%。

### D.2 联合信息较强的目标区域

| 目标区域 | 系统 $\Xi$ | 跨区域 $\Xi$ | 区域内 $\Xi$ |
|---|---:|---:|---:|
| West Chelsea/Hudson Yards | 0.4309 | 0.3680 | 0.0629 |
| Midtown Center | 0.4211 | 0.3845 | 0.0366 |
| Times Sq/Theatre District | 0.4116 | 0.3758 | 0.0358 |
| Upper East Side South | 0.4044 | 0.3883 | 0.0161 |
| Clinton East | 0.3816 | 0.3603 | 0.0213 |
| Upper East Side North | 0.3738 | 0.3538 | 0.0199 |
| Midtown East | 0.3677 | 0.3529 | 0.0148 |
| Penn Station/Madison Sq West | 0.3604 | 0.3227 | 0.0377 |

这些高值区域集中在商务、交通、娱乐和高强度活动区。一个待检验的解释是，它们的未来需求同时受多个城市功能区的同步状态影响；当前分析只定位联合信息的接收端，尚未识别具体来源组合。

### D.3 时间尺度的单源信息

| 滞后 | 标量单源 EI 之和（bits） |
|---|---:|
| 0.5 小时 | 5.5318 |
| 1 小时 | 3.1213 |
| 1.5 小时 | 2.6321 |
| 3 小时 | 2.0654 |
| 6 小时 | 1.5909 |
| 24 小时 | 2.8812 |
| 168 小时 | 3.3061 |

最近半小时提供最多单源信息；周周期、1 小时和日周期也有明显贡献。这组数值描述每个滞后单独与未来的关系，不应误写为 $\Xi$ 的时间归因。

### D.4 稳健性与边界

- 残差协方差 ridge 从 $10^{-8}$ 到 $10^{-4}$ 时，系统 $\Xi$ 为 8.1349–8.1218 bits，跨区域 $\Xi$ 为 7.4039–7.3920 bits。
- 所有区域内、分目标跨区域和分目标系统 $\Xi$ 均不低于 $10^{-10}$ bits 非负容差；违规数为 0。
- Global Ridge 的仿射高斯 TM 是解析读出。对 Interaction Ridge 或 MLP 的正式比较，必须在原始区域变量上保持乘积与非线性约束，并使用相同的干预支持。

---

## 附录 E：科学问题与后续实验

### E.1 可以继续追问的科学问题

1. **联合信息来自哪些区域组合？** 比较相邻区域、同功能区域和跨功能区域，寻找超出单区之和的稳定组合。
2. **联合信息是否领先于全城需求峰值？** 检查 $\Xi$ 是否在早晚高峰、节假日、极端天气或大型活动前上升。
3. **它是城市级共同因子，还是区域间传播？** 条件化全城总需求、主成分、天气和事件变量，观察跨区域 $\Xi$ 是否仍然存在。
4. **联合信息的时间尺度如何变化？** 扫描 0.5、1、2、4 小时预测时距，区分即时同步与更慢的空间传播。
5. **哪些区域是稳定的联合信息接收端或发送端？** 用滚动窗口、月份和年份比较排名稳定性，再讨论网络结构。

### E.2 建议的实验优先级

1. 加入全城总需求或第一主成分，排除简单共同因子解释。
2. 做相同星期、相同半小时槽内的季节保持置乱，避免同时破坏日/周周期。
3. 做多预测时距和月度滚动检验，给出 $\Xi$ 的时间稳定性。
4. 对候选区域组实施分组 TM；只有这一步才能从“系统存在联合信息”推进到“哪些组合贡献联合信息”。
5. 最后再用保持原始变量约束的非线性 TM 比较 Interaction Ridge 与 MLP。

---

## 附录 F：复现资源与数据开放性

### F.1 官方数据

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [NYC TLC Raw Data 说明](https://www.nyc.gov/site/tlc/about/raw-data.page)
- [Yellow Taxi 数据字典](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
- [Taxi Zone lookup table](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)
- [NYC Taxi Zones 边界](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/8meu-9t5y)

### F.2 代码、结果与图形

- [MGSTN 数据构造](../../scripts/prepare_nyc_taxi_mgstn.py)
- [MGSTN 训练与评价](../../scripts/train_nyc_taxi_mgstn.py)
- [MGSTN 全区域有限幅二阶 hurdle TM](../../scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py)
- [正式 TM 汇总与逐区域结果](../../results/nyc_taxi_mgstn_ei/finite_quadratic_tm/quadratic_tm_full_summary.json)
- [旧 MGSTN 局部 affine TM–PEID 分解](../../scripts/compute_nyc_taxi_mgstn_ei.py)
- [旧 MGSTN 时间尺度层级耦合](../../scripts/compute_nyc_taxi_mgstn_temporal_coupling.py)
- [MGSTN 框架图脚本](../../scripts/plot_nyc_taxi_mgstn_architecture.py)
- [MGSTN 信息分解图脚本](../../scripts/plot_nyc_taxi_mgstn_ei.py)
- [MGSTN 时间尺度耦合地图脚本](../../scripts/plot_nyc_taxi_temporal_coupling_map.py)
- [MGSTN 三随机种子预测结果](../../results/nyc_taxi_mgstn/summary.json)
- [旧 MGSTN 信息分解摘要](../../results/nyc_taxi_mgstn_ei/full_summary.json)
- [旧 MGSTN 完整分解数组](../../results/nyc_taxi_mgstn_ei/full_decomposition.npz)
- [旧 MGSTN 时间尺度耦合摘要](../../results/nyc_taxi_mgstn_ei/temporal_coupling_summary.json)
- [MGSTN 框架图 SVG](../../fig/nyc_taxi_mgstn_architecture.svg)（另有 PNG、PDF）
- [Taxi 社会系统多时间尺度主图 SVG](../../fig/nyc_taxi_social_multiscale_main.svg)（另有 PNG、PDF）
- [Taxi 时间尺度耦合地图 SVG](../../fig/nyc_taxi_temporal_coupling_map.svg)（另有 PNG、PDF）
- [数据聚合与模型筛选](../../scripts/nyc_taxi_synergy_model_screen.py)
- [更正后的 EI/$\Xi$ 计算](../../scripts/compute_nyc_taxi_ridge_xi.py)
- [主图脚本](../../scripts/plot_nyc_taxi_main_figure.py)
- [$\Xi$ 附录图脚本](../../scripts/plot_nyc_taxi_corrected_xi.py)
- [模型筛选结果](../../results/nyc_taxi_synergy_model_screen_metrics.json)
- [$\Xi$ 数值结果](../../results/nyc_taxi_global_ridge_xi.json)
- [聚合数据缓存](../../data/nyc_taxi_yellow_2023_30min_manhattan.npz)
- [主图 SVG](../../fig/nyc_taxi_main_results.svg)（另有 PNG、PDF）
- [模型筛选附录图 SVG](../../fig/nyc_taxi_synergy_model_screen.svg)（另有 PNG、PDF）
- [$\Xi$ 分解附录图 SVG](../../fig/nyc_taxi_corrected_xi_decomposition.svg)（另有 PNG、PDF）

所有图均由机器可读结果直接生成，图中数值没有手工改写。

### F.3 定义与文献说明

PEID 定义与非负性依据仓库中的 [Method](Method.md)，并与 Brain affine-TM 和 Earth 完整干预 EI 的实现口径对齐。本轮同时全文核对了本地 Zotero 条目 *Partial Effective Information Decomposition for Synergistic Causality*（item key `MYATYWAJ`，附件全文 26/26 页）：联合与部分 EI 均在同一个独立最大熵干预下定义，$\operatorname{Syn}_{P}=\operatorname{EI}(\mathrm{all}\to Y)-\sum_i\operatorname{EI}(i\to Y)$，并满足层级可加分解。正式 NYC Taxi 结果改用经验 bootstrap 干预与有限幅二阶 hurdle TM，因此结论限定为冻结 MGSTN 在所选状态、所选干预支持和二阶近似下的目标级预测机制。
