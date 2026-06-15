# 一般连续动力学中的 PID 协同计算前沿

## 结论摘要

截至 2026 年 6 月，对一般连续、非线性、可能高维的动力学系统，尚不存在一个同时满足
以下条件的公认 PID 协同计算方法：

1. 对任意连续分布均有明确理论定义；
2. PID 原子非负且满足所有常用公理；
3. 对变量的可逆变换不敏感；
4. 能稳定处理高维、小样本和强时间相关；
5. 能区分观测相关与动力学机制；
6. 有成熟、统一的软件实现。

目前最前沿的方法不是单一路线，而是按问题口径分成四类：

| 研究口径 | 当前优先方法 | 主要优势 | 主要限制 |
|---|---|---|---|
| 低维、一般连续、非参数完整 PID | Continuous SxPID | 直接定义连续冗余，支持非线性与多源 | kNN 维数灾难；依赖预处理；可能出现负原子 |
| 二源、一般连续、强调 Blackwell/决策语义 | Continuous BROJA unique information | 优化型定义清晰；适用于非线性连续分布 | 主要限于二源；神经优化昂贵且需收敛审计 |
| 高维、一般连续数据 | Flow-PID / Thin-PID | 2025 年最前沿的可扩展连续 PID；有开源实现 | 对非高斯数据依赖 latent Gaussian 近似 |
| 连续随机过程与时间记忆 | Partial Information Rate Decomposition, PIRD | 直接分解信息率而非单时刻互信息 | 当前实用实现主要依赖 Gaussian VAR/state-space |

对本项目的一般连续动力学，建议不要只选一个“最终 PID 数值”。主分析应采用固定随机
干预通道下的低维 `2→1` PID，并用至少一种不同定义做敏感性检查：

1. 主指标：独立干预样本上的 Continuous SxPID；
2. 二源理论敏感性：Continuous BROJA PID；
3. 高维状态或表征：Flow-PID；
4. 自然轨迹的过程级补充：Gaussian PIRD；
5. 始终同时报告 WMS/PEID、组成互信息、估计不确定性与采样口径。

## 1. 连续动力学首先需要定义随机通道

设两个当前源变量共同影响未来目标：

$$
(X_{1,t},X_{2,t})\to Y_{t+\tau}.
$$

PID 分解的对象不是微分方程本身，而是联合分布

$$
p(x_{1,t},x_{2,t},y_{t+\tau}).
$$

该分布可以来自自然轨迹，也可以来自干预分布与转移核：

$$
q(x_{1,t},x_{2,t})\,
p(y_{t+\tau}\mid x_{1,t},x_{2,t}).
$$

两者回答不同问题：

- 自然轨迹 PID：系统实际访问状态分布上的统计协同；
- 干预 PID：指定源分布下，动力学机制产生的联合约束。

### 1.1 无噪声连续确定性映射的发散问题

若

$$
Y=f(X_1,X_2)
$$

是无噪声连续确定性映射，则联合分布可能集中在低维流形上。此时条件微分熵可能为
\(-\infty\)，从而

$$
I(X_1,X_2;Y)
$$

可能发散。任何有限 PID 数值都隐含了某种有限分辨率或正则化。

因此，连续动力学 PID 必须预注册至少一种随机化口径：

- 显式过程噪声；
- 显式观测噪声；
- 概率预测模型给出的条件分布；
- 有限分辨率或分箱；
- 固定带宽的平滑核；
- 具有明确噪声尺度的干预响应分布。

不说明该尺度时，不同估计器得到的有限数值通常不能解释为同一个理论常数。

### 1.2 时间结构不能被忽略

将一条密集采样轨迹的每个时间点当作独立样本，会低估不确定性并混淆状态记忆与二源
协同。至少需要：

- 用多个独立初值或独立轨迹构造 transition pool；
- 固定并扫描预测时距 \(\tau\)；
- 对自然轨迹使用 block bootstrap；
- 检查嵌入历史是否足以近似 Markov 状态；
- 区分单步 PID 与过程级 information-rate PID。

## 2. Continuous SxPID：低维一般连续系统的首选完整 PID

Ehrlich et al. (2024) 提出了基于 shared exclusions 的连续 PID
（Continuous SxPID）。它将离散 SxPID 中“源事件共同排除了哪些目标事件”的思想推广
到连续概率密度，并给出基于 KSG/kNN 的估计器。

对二源一目标，它先估计连续冗余

$$
R_{\mathrm{sx}}(X_1,X_2;Y),
$$

再由 PID 一致性方程得到

$$
\begin{aligned}
U_1 &= I(X_1;Y)-R_{\mathrm{sx}},\\
U_2 &= I(X_2;Y)-R_{\mathrm{sx}},\\
S_{\mathrm{sx}}
&=I(X_1,X_2;Y)-U_1-U_2-R_{\mathrm{sx}}.
\end{aligned}
$$

### 优势

- 面向纯连续变量，不要求线性或高斯；
- 给出完整 PID，而不只是 WMS 或单个 unique information；
- 能推广到多个源变量及 PID redundancy lattice；
- 使用局部概率结构，能够识别非线性关系；
- 作者提供了连续估计器实现。

### 限制

- kNN 估计在维数升高后迅速变难；
- 结果受样本量与邻居参数 \(k\) 影响；
- 当前实用解析定义不对各变量的任意双射严格不变，需要预注册预处理；
- SxPID 允许负信息原子，其语义是平均 misinformation，而不是估计器必然出错；
- 对时间序列直接计算时，负原子的解释更困难。

### 适用判断

若源和目标均为低维连续变量，样本量足够，并且研究目标是一般非线性观测 PID，
Continuous SxPID 是当前最直接、最完整的前沿方法。它不应被描述成唯一正确 PID，
而应被描述成采用 shared-exclusion 冗余语义的 PID。

## 3. Continuous BROJA：二源连续 PID 的优化型路线

Pakman et al. (2021) 将 BROJA unique information 推广到连续变量。其核心是在保持两个
源-目标边缘分布不变的分布集合

$$
\Delta_p
=
\left\{
q:\ q(x_1,y)=p(x_1,y),\
q(x_2,y)=p(x_2,y)
\right\}
$$

上进行优化。例如，源 \(X_1\) 的 unique information 定义为

$$
U_1^{\mathrm{BROJA}}
=
\min_{q\in\Delta_p} I_q(X_1;Y\mid X_2).
$$

对二源 PID，一旦得到 unique information，即可由一致性方程恢复冗余与协同。连续
估计器结合 copula 分解和类似变分自编码器的优化技术。

### 优势

- 具有明确的 Blackwell/决策理论语义；
- 在定义层面适用于一般连续分布；
- 能处理非线性关系；
- 已用于混沌 rate-neuron 网络和循环神经网络。

### 限制

- 实用方法主要针对二源 PID；
- 需要优化分布，计算成本和训练不稳定性高于 kNN 读出；
- 结果需要多次随机初始化、约束满足度和收敛审计；
- 高维下仍然困难。

### 适用判断

若重点是“一个源相对于另一个源是否提供不可替代信息”，Continuous BROJA 是比 MMI-PID
更有操作语义的二源连续方法。对本项目，可将其作为 SxPID 的关键敏感性对照，而不是
唯一主指标。

## 4. Gaussian PID、Thin-PID 与 Flow-PID：高维连续路线

### 4.1 Gaussian PID

对联合高斯向量，PID 可以利用协方差结构与优化算法高效计算。Venkatesh et al. (2023)
进一步处理了高维 Gaussian PID 的有限样本偏差问题。该路线适合：

- 线性随机动力学；
- VAR 或 state-space 模型；
- 局部线性化后的机制；
- 高维数据的稳定基准。

但 Gaussian PID 只识别协方差层面的依赖。对乘积门控、相位耦合和多峰吸引子等一般
非线性结构，直接套用 Gaussian PID 会遗漏关键协同。

### 4.2 Thin-PID

Zhao et al. (2025) 证明：当两个源-目标边缘分布

$$
p(x_1,y),\qquad p(x_2,y)
$$

均为多元高斯时，Gaussian PID 优化存在联合高斯最优解。Thin-PID 据此使用梯度算法
高效求解高维 Gaussian PID。

在 pairwise Gaussian 假设成立时，Thin-PID 属于精确而可扩展的前沿方法。

### 4.3 Flow-PID

Flow-PID 使用分别作用于 \(X_1\)、\(X_2\) 与 \(Y\) 的可逆 normalizing flows，将一般
连续变量映射到近似 pairwise Gaussian 的潜空间，再运行 Thin-PID。可逆映射保留总互
信息，因而该方法试图在高维非高斯数据上获得可计算 PID。

### 优势

- 当前面向高维连续非高斯数据最前沿、最可扩展的 PID 路线之一；
- 适用于向量源和向量目标；
- 论文和代码均提供 Thin-PID、Flow-PID 与 Gaussian 基线；
- 可利用 GPU 和现代表示学习工具。

### 限制

- Thin-PID 只在 pairwise Gaussian 条件下精确；
- Flow-PID 对一般非高斯数据是近似方法；
- 误差取决于 normalizing flow 是否成功 Gaussianize 两个源-目标边缘分布；
- 复杂分布、有限样本和优化误差会造成偏差；
- 对真实数据缺少可验证的 PID ground truth。

### 适用判断

Flow-PID 是当前高维连续 PID 的首选前沿候选，但不能直接称为“一般连续 PID 的精确
解”。使用时必须同时报告：

- 潜空间 pairwise Gaussian 拟合诊断；
- 多次训练的方差；
- 与 Gaussian PID、SxPID 或低维投影结果的对照；
- 对流模型结构与正则权重的敏感性。

## 5. PIRD：连续动力学的过程级前沿

普通 PID 分解随机变量之间的互信息。对具有自相关与记忆的动力学，Partial Information
Rate Decomposition（PIRD）改为分解目标过程与源过程之间的 mutual information rate。

该方法可在频域和时间域区分 unique、redundant 与 synergistic information rates，避免
把动态网络错误地当成 i.i.d. 样本集合。2025 年工作给出了 Gaussian 随机过程的 spectral
实现，并有基于 VAR/state-space 的开源 MATLAB toolbox。

### 优势

- 研究对象与连续时间序列的动态性质更一致；
- 显式处理跨时间记忆和频带结构；
- 适合生理信号、气候信号和线性随机网络。

### 限制

- 当前实用实现主要是 Gaussian VAR/state-space；
- 不能直接处理任意非线性动力学机制；
- 主要是观测过程级信息分解，不自动具有干预因果语义。

因此，PIRD 是一般连续动力学的重要前沿方向，但当前更适合作为自然轨迹的过程级补充，
而不是非线性机制 PID 的统一解。

## 6. 其他重要但受限的前沿

### 6.1 仿射稳定分布与卷积闭合分布的解析 PID

Goswami and Merkley (2024) 将解析 PID 从高斯推广到稳定分布、卷积闭合分布和部分指数
族，包括 Cauchy、Poisson、Gamma 与 binomial 等。该路线提供严格解析结果，但要求明确
的仿射生成结构，不能覆盖一般非线性动力学。

### 6.2 MMI-PID

MMI-PID 易计算、稳定，适合 Gaussian scalar-target benchmark：

$$
S_{\mathrm{MMI}}
=
I(X_1,X_2;Y)
-
\max\{I(X_1;Y),I(X_2;Y)\}.
$$

但 MMI 的冗余定义较粗，尤其在向量目标、非高斯和复杂动力学中不应作为默认 ground
truth。它应保留为基准，而不是前沿主方法。

### 6.3 SURD 与其他因果分解

SURD 等方法面向因果影响的 unique、redundant 与 synergistic 分解，适合一般非线性
动力学，但它们不是经典 PID 冗余定义下的同一个数学对象。可以作为机制和因果敏感性
对照，不能与 PID 原子无条件等同。

## 7. 面向本项目的推荐协议

### 7.1 主问题定义

对每个候选二源关系，固定

$$
(X_{1,t},X_{2,t})\to Y_{t+\tau}
$$

及随机干预分布

$$
q^{\max}(x_{1,t})q^{\max}(x_{2,t}).
$$

通过已知动力学或冻结的概率预测模型生成

$$
p(y_{t+\tau}\mid x_{1,t},x_{2,t}).
$$

这使 PID 描述指定干预域上的机制协同，并避免自然轨迹源相关性主导冗余。

### 7.2 建议的三层计算

**第一层：低维正式 PID**

- 使用 Continuous SxPID 计算 \(R,U_1,U_2,S\)；
- 扫描 kNN 参数 \(k\)、样本量、噪声尺度与预测时距 \(\tau\)；
- 使用独立 transition pool 与 bootstrap 区间。

**第二层：定义敏感性**

- 对二源关系使用 Continuous BROJA；
- 同时报告 MMI-PID 与 WMS/PEID；
- 只将跨定义均稳定的排序解释为稳健结论。

**第三层：高维与过程级补充**

- 高维状态或 learned representation 使用 Flow-PID；
- 线性高斯自然轨迹使用 PIRD；
- 用 Gaussian PID 作为局部线性和估计偏差基准。

### 7.3 必须报告的诊断

每个结果至少报告：

| 类别 | 必须报告内容 |
|---|---|
| 随机通道 | 干预分布、噪声来源、噪声尺度、预测时距 |
| 样本结构 | 轨迹数、样本数、独立初值、时间间隔 |
| PID 定义 | SxPID、BROJA、Flow-PID、MMI 或 PIRD |
| 估计参数 | kNN 的 \(k\)、flow 结构、优化种子、VAR 阶数 |
| 稳定性 | bootstrap 区间、跨 seed 方差、样本量曲线 |
| 敏感性 | 预处理、可逆变换、噪声尺度、\(\tau\) 扫描 |
| 对照 | WMS/PEID、组成互信息、结构零点、已知机制 |

## 8. 最终判断

如果“最前沿”指理论上最直接面向一般低维连续分布的完整 PID，当前首选是 2024 年
Continuous SxPID。

如果“最前沿”指可扩展到高维连续非高斯数据，当前首选候选是 NeurIPS 2025
Flow-PID，但它是依赖 latent Gaussianization 质量的近似方法。

如果“最前沿”指真正尊重连续动力学的时间记忆结构，则 2025 年 PIRD 是最相关方向，
但当前实现仍主要限制在线性 Gaussian 随机过程。

因此，对一般连续非线性动力学，最严谨的方案不是押注单个 PID 估计器，而是：

$$
\text{明确随机通道}
\;+\;
\text{SxPID/BROJA 双定义}
\;+\;
\text{Flow-PID 高维检查}
\;+\;
\text{PIRD 过程级补充}.
$$

## 参考文献与实现

- Ehrlich, D. A., et al. (2024). Partial information decomposition for
  continuous variables based on shared exclusions: Analytical formulation and
  estimation. *Physical Review E*, 110, 014115.
  [论文](https://doi.org/10.1103/PhysRevE.110.014115)；
  [连续估计器](https://gitlab.gwdg.de/wibral/continuouspidestimator)。
- Pakman, A., et al. (2021). Estimating the Unique Information of Continuous
  Variables. *NeurIPS 2021*.
  [论文](https://proceedings.neurips.cc/paper/2021/hash/a9a1d5317a33ae8cef33961c34144f84-Abstract.html)。
- Venkatesh, P., et al. (2023). Gaussian Partial Information Decomposition:
  Bias Correction and Application to High-dimensional Data. *NeurIPS 2023*.
  [论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html)。
- Goswami, C., & Merkley, A. (2024). Analytically deriving Partial Information
  Decomposition for affine systems of stable and convolution-closed
  distributions. *NeurIPS 2024*.
  [论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9df56a345b2a9c64c294986a5a63b8a6-Abstract-Conference.html)。
- Zhao, W., et al. (2025). Partial Information Decomposition via Normalizing
  Flows in Latent Gaussian Distributions. *NeurIPS 2025*. Zotero item:
  `BYIY6AXJ`.
  [论文](https://arxiv.org/abs/2510.04417)；
  [代码](https://github.com/warrenzha/flow-pid)。
- Faes, L., et al. (2025). Partial Information Rate Decomposition.
  *Physical Review Letters*, 135, 187401.
  [论文](https://doi.org/10.1103/nrwj-n8lj)；
  [预印本](https://arxiv.org/abs/2502.04550)；
  [代码](https://github.com/LauraSparacino/PIRD_toolbox)。
