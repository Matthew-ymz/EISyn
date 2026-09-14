# EI—Koopman—干预控制：文献对照与阅读顺序

检索日期：2026-09-14。配套结论、推导及实验见 [探索报告](ei_koopman_control_exploration.md)。

## 1. 范围与证据边界

问题是：多尺度 EI 和 PEID 能否帮助学习可控制的表示、选择联合执行器，并降低指定宏观状态转移的成本？检索覆盖 EI/NIS+、信息论 Koopman 表示、empowerment、网络控制、控制抽象及技能学习，时间范围为基础工作至 2026-09-14。

首先只读检索本地 Zotero；以 `effective information Koopman causal emergence control empowerment NIS` 排序得到 35 个候选，另精确检索 PEID。选取 7 篇直接相关论文读取本地索引全文，并以公开学术页面核对关键题名、版本或发表信息。外部定向检索补齐控制和 RL 文献。下表保留 34 篇有明确关联的文献；不声称全部精读，也不将检索结果数量当作证据强度。

证据标签：**全文**表示读取了可访问正文中的相关章节，不代表逐页验证所有定理；**摘要**表示只据摘要或学术页面的摘要描述其关联；**仅元数据**不用于推断结果。未逐篇排除后续勘误，本文也不是穷尽式系统综述。

本地论文主键与附件键：

| 文献 | Zotero item key | 全文 attachment key | 本次使用部分 |
| --- | --- | --- | --- |
| PEID | MYATYWAJ | Y8GH3W58 | 二源定义、独立干预、分区 Syn 非负性及层级关系 |
| NIS+ | T3TKPLLG | JKTP95PB | 宏观均匀干预、表示学习目标、方法和鸟群实验 |
| 线性随机 EI | FTUE5FKU | GAGTIZ4M | 干预域、线性粗粒化、EI 表达式 |
| Information Shapes Koopman Representation | ZNEHTSSU | APVZWFXQ | 概率生成模型、信息目标、谱分配与表示塌缩 |
| Koopman 有限维不变子空间 | Y2EFJY5S | JL4329JZ | 不变子空间限制、控制动机 |
| Koopman 控制综述 | 5S7TB4JI | QQK736JS | 受控模型、双线性表示、误差与闭环保证 |
| 生物宏观尺度与控制 | AELDZBBX | WHXVBYZP | EI 的生物尺度选择动机 |

本地排名中其余相关论文只使用摘要，键在下表注明。没有向 Zotero 导入或修改记录。

## 2. 直接相关文献及各自作用

### EI、多尺度与分解

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 1 | Yang、Wang、Zhang，2026，[Partial Effective Information Decomposition for Synergistic Causality](https://arxiv.org/abs/2605.03267) | Emerging，预印本；全文；MYATYWAJ | 沿用干预和 Syn 定义；它不是控制性能定理 |
| 2 | Yang 等，2025，[Finding emergence in data by maximizing effective information](https://pmc.ncbi.nlm.nih.gov/articles/PMC11697982/) | Recent frontier；全文；T3TKPLLG | NIS+ 表示选择基础；需补显式动作条件机制 |
| 3 | Zhang、Liu，2022 在线发表，[Neural Information Squeezer for Causal Emergence](https://doi.org/10.3390/e25010026) | Foundational；摘要；K8NRVTXW | 可逆转换和信息丢弃的结构；卷期属于 2023 年 |
| 4 | Liu、Yuan、Zhang，2024，[An Exact Theory of Causal Emergence for Linear Stochastic Iteration Systems](https://arxiv.org/abs/2405.09207) | Recent frontier；全文；FTUE5FKU | 最直接解析先例；引用公式前核对均匀输入与噪声卷积 |
| 5 | Liu 等，2025，[SVD-based Causal Emergence for Gaussian Iterative Systems](https://arxiv.org/abs/2502.08261) | Recent frontier；摘要；C7479JCN | 谱、噪声协方差与 EI 的桥梁；采用此预印本题名 |
| 6 | Hoel、Albantakis、Tononi，2013，[Quantifying causal emergence shows that macro can beat micro](https://doi.org/10.1073/pnas.1314922110) | Foundational；摘要；G9D838F7 | 多尺度重新干预的基础，避免误用数据处理不等式 |
| 7 | Hoel、Levin，2020，[Emergence of informative higher scales in biological systems: a computational toolkit for optimal prediction and control](https://doi.org/10.1080/19420889.2020.1802914) | Foundational；全文；AELDZBBX | “EI 支持生物调控”已有明确动机，不能当全新想法 |
| 8 | Rosas 等，2020，[Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data](https://doi.org/10.1371/journal.pcbi.1008289) | Foundational；摘要；7LWNIPLK | 脑活动和鸟群的多尺度信息先例；不等于真实最小能量控制 |
| 9 | Varley、Hoel，2022，[Emergence as the conversion of information: a unifying theory](https://doi.org/10.1098/rsta.2021.0150) | Foundational；摘要；5V5LJEMM | 总信息与分解组成可有不同变化，支持区分 MI 和 Syn |

### Koopman 表示与控制

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 10 | Cheng 等，ICLR 2026，[Information Shapes Koopman Representation](https://arxiv.org/abs/2510.13025) | Recent frontier；全文；ZNEHTSSU | 最接近的信息论表示学习对照；已有 MI 与谱多样性的联合目标 |
| 11 | Brunton 等，2016，[Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control](https://pmc.ncbi.nlm.nih.gov/articles/PMC4769143/) | Foundational；全文；Y2EFJY5S | 区分有限维精确表示、近似表示和可重建原状态的要求 |
| 12 | Proctor、Brunton、Kutz，2016 预印本，[Generalizing Koopman Theory to allow for inputs and control](https://arxiv.org/abs/1602.07647) | Foundational；摘要 | 自主动力学表示不足以识别输入输出作用 |
| 13 | Korda、Mezić，2018，[Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control](https://arxiv.org/abs/1611.03537) | Foundational；全文相关章节 | 最合适的低成本控制基线：相同预测器容量和 MPC 预算 |
| 14 | Li 等，ICLR 2020，[Learning Compositional Koopman Operators for Model-Based Control](https://arxiv.org/abs/1910.08264) | Foundational；摘要 | 图结构/可组合表示已经用于控制，适合网络任务对照 |
| 15 | Strässer 等，2026，[An overview of Koopman-based control: From error bounds to closed-loop guarantees](https://doi.org/10.1016/j.arcontrol.2025.101035) | Recent frontier；全文；5S7TB4JI | 误差与稳定性条件是现代 Koopman 控制的重要部分；不能只比预测误差 |

### 信息论控制、网络控制和脑网络

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 16 | Klyubin、Polani、Nehaniv，2008，[Keep Your Options Open: An Information-Based Driving Principle for Sensorimotor Systems](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0004018) | Foundational；摘要 | 动作到感知结果的通道容量，是 EI 控制方向的直接邻域 |
| 17 | Salge、Glackin、Polani，2012，[Approximation of Empowerment in the Continuous Domain](https://doi.org/10.1142/S0219525912500798) | Foundational；摘要 | 连续 empowerment 的线性高斯近似，避免重复已有信道计算 |
| 18 | Salge、Glackin、Polani，2013 预印本，[Empowerment—An Introduction](https://arxiv.org/html/1310.1863) | Foundational；全文相关章节 | 任务无关的控制能力与指定目标效用不同；开放/闭环语义需区分 |
| 19 | Summers、Cortesi、Lygeros，2016，[On Submodularity and Controllability in Complex Dynamical Networks](https://arxiv.org/abs/1404.7665) | Foundational；摘要 | 执行器组合的强基线；具体子模结论必须逐指标核对，不能一概而论 |
| 20 | Baggio、Bassett、Pasqualetti，2021，[Data-Driven Control of Complex Networks](https://arxiv.org/abs/2003.12189) | Foundational；摘要 | 已有从实验输入输出直接求控制的路线；定理适用线性系统 |
| 21 | Gu 等，2015，[Controllability of structural brain networks](https://www.nature.com/articles/ncomms9414) | Foundational；摘要及页面方法片段 | 脑网络线性控制的基础参照，拓扑与真实非线性刺激机制需区分 |
| 22 | Betzel 等，2016，[Optimally controlling the human connectome: the role of network topology](https://www.nature.com/articles/srep30770) | Foundational；摘要 | 将起终状态与控制节点纳入问题定义 |
| 23 | Karrer 等，2019 预印本，[A practical guide to methodological considerations in the controllability of structural brain networks](https://arxiv.org/abs/1908.03514) | Foundational；摘要 | 控制时域、归一化、输入矩阵等方法选择；此处不混用正式发表年份 |
| 24 | 2024，[A network control theory pipeline for studying the dynamics of the structural connectome](https://www.nature.com/articles/s41596-024-01023-w) | Recent frontier；摘要 | 可复用控制能量和平均可控性的计算流程 |
| 25 | Ben Messaoud 等，2025，[Low-dimensional controllability of brain networks](https://doi.org/10.1371/journal.pcbi.1012691) | Recent frontier；摘要 | 极直接的宏观控制对照：谱投影与输出可控性；全文访问遇到验证码，未据全文推断细节 |

### 强化学习与集体调控

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 26 | Zhang 等，ICLR 2021，[Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742) | Foundational；摘要 | 用 bisimulation 保留任务相关信息，是控制抽象的直接理论对照 |
| 27 | Eysenbach 等，ICLR 2019，[Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070) | Foundational；摘要 | 信息论技能发现；技能可区分性不等于低成本目标转移 |
| 28 | Sharma 等，ICLR 2020，[Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657) | Foundational；摘要 | 技能的可预测结果和宏观规划，最贴近分层控制构想 |
| 29 | Park、Rybkin、Levine，ICLR 2024，[METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://arxiv.org/abs/2310.08887) | Recent frontier；摘要 | 时间距离约束的技能表示；纯 MI 目标的局限已有讨论 |
| 30 | Hansen、Su、Wang，ICLR 2024，[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) | Recent frontier；摘要 | 适合第二阶段的潜在世界模型与规划基线，非本轮重训练对象 |
| 31 | Hafner 等，2025，[Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) | Recent frontier；摘要 | Dreamer 的跨任务世界模型路线；规模较大，应在小实验成立后比较 |
| 32 | 2019 预印本，[Learning to flock through reinforcement](https://arxiv.org/abs/1911.01697) | Foundational；摘要 | 已有学习集体运动先例；“形成鸟群”与“低成本宏观转向”应分开 |
| 33 | 2021，[Learning to control active matter](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033291) | Foundational；摘要 | 用 RL 调控集体现象，适合作为复杂系统调控场景参照 |
| 34 | Chen 等，2023，[Optimal synchronization in pulse-coupled oscillator networks using reinforcement learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10109446/) | Recent frontier；摘要 | 振子网络的任务驱动调控，适合脑/同步任务的邻近基线 |

## 3. 综合判断与阅读顺序

**已有共识与先例。** 信息理论指导表示学习、动作到结果的容量、Koopman 预测器用于控制、低维脑网络输出控制，都已有直接文献。新工作不能仅将这几个词放在同一框架里。

**需认真区分的假设。** NIS+ 比较的宏观最大熵干预、Koopman 表示学习中的观测/生成分布、empowerment 优化的动作分布、最小能量控制的目标约束，彼此不相同。“预测更好”“EI 更高”“控制更便宜”因而不是可互换结论。

**值得探索但尚未证实的缺口。** 将物理可实施干预、受控动态闭合和成本保持同时纳入多尺度 EI 表示选择，再检验 PEID 是否能在相同预算下缩小执行器组合搜索。当前检索没有确立这一具体方案的首创性，更没有证明它优于现有算法。

建议阅读顺序：

1. 先对读 **NIS+、线性随机 EI、PEID**：明确自己的干预和分解对象。
2. 再读 **Information Shapes Koopman Representation、Koopman MPC、2026 Koopman 控制综述**：明确已经解决的表示问题，以及未解决的受控误差问题。
3. 接着读 **empowerment 综述、低维脑网络控制、执行器 Gramian 选择**：区分整体控制能力与指定目标成本。
4. 最后看 **DBC、DADS、METRA**，选择控制抽象或技能学习的连接点；TD-MPC2/Dreamer 留作后续预算允许时的算法对照。

## 4. 检索可复核性与访问限制

公共学术 API 的四个原始查询为：

- `Koopman effective information causal emergence control`，起始年份 2013。
- `empowerment controllability information energy`，起始年份 2005。
- `causal emergence neural information squeezer control`，起始年份 2013。
- `Koopman model predictive control reinforcement learning abstraction`，起始年份 2023。

每查询每源上限 12、超时 8 秒。返回的跨领域噪声较多，部分元数据年份甚至超过当前日期，因此没有直接使用自动排名的“最新”结果。后续采用题名和主题定向检索，并优先使用 arXiv、出版社、作者机构页面以及本地全文。

arXiv export API 与部分 Semantic Scholar 查询返回 HTTP 429 或超时；遇到这些结果后未重试同一访问路径。Baggio 论文的实验 HTML 入口失败，但摘要页可读；低维脑网络论文 PMC 正文访问触发验证码，降级为摘要证据。另一个非核心候选 Cornelius 等的 Nature 入口跳转认证失败，未将其作为已核验文献纳入上表。没有下载远程 PDF、绕过登录或导入 Zotero。

原始 API 记录暂存于 `/tmp/eisyn_control_search_1.json` 至 `/tmp/eisyn_control_search_4.json`，仅为本次检索审计；下方保存持久化的查询与失败链接，避免依赖临时文件重建检索。

### 公共 API 查询及失败记录

**查询 1**：Koopman effective information causal emergence control

- [openalex](https://api.openalex.org/works?search=Koopman+effective+information+causal+emergence+control&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2013-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+effective+information+causal+emergence+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+effective+information+causal+emergence+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=Koopman+effective+information+causal+emergence+control&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2013-01-01)
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+effective+information+causal+emergence+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — permission gate: HTTP 429

**查询 2**：empowerment controllability information energy

- [openalex](https://api.openalex.org/works?search=empowerment+controllability+information+energy&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2005-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=empowerment+controllability+information+energy&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2005-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3Aempowerment+controllability+information+energy&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=empowerment+controllability+information+energy&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2005-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=empowerment+controllability+information+energy&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2005-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3Aempowerment+controllability+information+energy&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — permission gate: HTTP 429

**查询 3**：causal emergence neural information squeezer control

- [openalex](https://api.openalex.org/works?search=causal+emergence+neural+information+squeezer+control&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2013-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=causal+emergence+neural+information+squeezer+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3Acausal+emergence+neural+information+squeezer+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=causal+emergence+neural+information+squeezer+control&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2013-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=causal+emergence+neural+information+squeezer+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3Acausal+emergence+neural+information+squeezer+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — network failure: The read operation timed out

**查询 4**：Koopman model predictive control reinforcement learning abstraction

- [openalex](https://api.openalex.org/works?search=Koopman+model+predictive+control+reinforcement+learning+abstraction&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2023-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2023-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+model+predictive+control+reinforcement+learning+abstraction&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2023-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2023-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+model+predictive+control+reinforcement+learning+abstraction&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — network failure: The read operation timed out
