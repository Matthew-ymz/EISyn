# UniCM：历史干预比较的文献指标核对

本报告保存分布修正前的定义核查。自然观测版已沿用文献公式、改用真实历史和真实未来，加入全阶 SURD；实现、结果及计时见[新比较报告](unicm_observational_prior_comparison.md)。下文“当前”指本报告记录的历史干预版。

2026-10-02。当前比较已移除 Conditional MI + self MI 及 MIM，保留新增的正向 stochastic interaction、causal density。MIM 使用同一最大熵干预分布，其读数等于 singleton EI，因此不作为普通观测互信息的独立对照。统一采用原 Ξ-Shapley 验证最优 α=30000、γ=3；[实测结果和图](unicm_information_prior_comparison.md)。

## 1. 被移除项是不是新构造

条件互信息和自历史—未来互信息本身都是既有信息量；此前使用的完整 prior 是本实验构造的接口：对目标自身历史使用普通 MI，对其他源使用仅条件于目标历史的 CMI。

$$
c_{j,\ell,m}^{\mathrm{old}}=
\begin{cases}
I(\mathbf h_j;Y_{j,\ell}),&m=j,\\
I(\mathbf h_m;Y_{j,\ell}\mid\mathbf h_j),&m\ne j.
\end{cases}
\tag{1.1}
$$

核对的原文没有将式（1.1）定义为名为“Conditional MI + self MI”的独立指标。因此它不应作为一项直接来自文献的竞争定义，现已从当前图、拟合方法列表和排序移除。这里不声称已经证明所有历史文章都不存在相同组合。旧缓存与[此前五项报告](../log/unicm_information_prior_comparison_five_methods_historical.md)保留追溯。

对所有保留方法也采用同一标准：**文献信息量定义**与**本实验的 attribution → ridge 惩罚映射**分开陈述。后者用于统一比较，不声称各原论文已经提出同样的冻结 Transformer 校准流程。

## 2. 当前新增指标与已移除的单源读数

所有信息量单位为 bit。$\mathbf h_m$ 是模态 $m$ 的完整12月历史；$Y_{j,\ell}$ 是同一冻结模型对目标 $j$、lead $\ell$ 的读出。各方法使用同一独立干预分布及 degree-1 Gaussian affine TM 代理，未改用不同观测估计器。

### 2.1 已移除的 MIM：在当前分布下等于单源 EI

$$
c^{\mathrm{MIM}}_{j,\ell,m}=I(\mathbf h_m;Y_{j,\ell}).
\tag{2.1}
$$

[Brown、Pocock、Zhao、Luján（2012）](https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf)式（1）明确采用该 relevance score，并列出其 Lewis（1992）来源。原方法按 MI 排序选特征；当前实验保持44维校准容量，仅将同一分数映射为源分组惩罚。它保留 target-specific 分辨率，不做 Shapley，也不减单源信息。

当前干预分布下，式（2.1）就是缓存中的 singleton EI，已从主图、比较图及当前方法列表移除；保留公式仅说明删除理由和历史缓存含义。它仍是互信息，删除的是其作为普通观测 MI 的独立竞争身份。它与既有 Univariate 不同：Univariate 只拟合目标自身冻结预测的一个系数和截距；历史 MIM 使用全部44个特征、按11个源的 singleton EI 分配惩罚；既有 EI-Shapley 使用全联盟归因，二者的惩罚和预测并非完全相同。也不能将普通 MI 无条件称为任意 PID 定义下的 unique information。

### 2.2 正向 stochastic interaction：ΦSI

[Kitazono、Kanai、Oizumi（2018）](https://doi.org/10.3390/e20030173)附录A式（A7）—（A10）给出正向条件分布分区及 stochastic interaction。对同一双源边采用

$$
q^{\mathrm{SI}}_{ab,\ell}
=H(Y_{a,\ell}\mid\mathbf h_a)+H(Y_{b,\ell}\mid\mathbf h_b)
-H(\boldsymbol y_{ab,\ell}\mid\mathbf h_a,\mathbf h_b),
\quad \boldsymbol y_{ab,\ell}=(Y_{a,\ell},Y_{b,\ell})^{\mathsf T}.
\tag{2.2}
$$

全部55对仍从完整同一 channel 边缘化其他源，每条边各半分给两端，随后广播到目标；与 ΦR/ΦWMS 使用相同配对和归因接口，不执行全系统 MIP 搜索。将12月历史作为过去源块、不同 lead 作为未来读出是本实验的应用约定。

当前 pair WMS 与式（2.2）满足

$$
q^{\mathrm{SI}}_{ab,\ell}=q^{\mathrm{WMS}}_{ab,\ell}
 +\operatorname{TC}(Y_{a,\ell},Y_{b,\ell}).
\tag{2.3}
$$

因此这个正向版本形成不同对照。它也会计入输出、包括相关残差造成的依赖，不能解释为纯预测协同或 PEID Syn。注意方向：Mediano等（2019）的 $\widetilde\Phi$ 使用过去给定未来的条件熵；其与 WMS 的差为过去 TC。当前源独立时，**反向版本**会与 WMS 重合，所以当前图明确标注 forward，不能把两种方向混称。

### 2.3 Causal density：条件信息传递的平均

[Seth、Barrett、Barnett（2011）](https://doi.org/10.1098/rsta.2011.0079)讨论 causal density；本次实现采用 [Mediano、Seth、Barrett（2019）](https://doi.org/10.3390/e21010017)式（29）—（31）的条件 transfer entropy 形式，条件中包括目标历史和其他全部源历史：

$$
T_{m\to j,\ell}=I(\mathbf h_m;Y_{j,\ell}\mid\mathbf h_{-m}),\quad m\ne j,
\qquad
\operatorname{CD}_{\ell}=\frac{1}{11\cdot10}\sum_{m\ne j}T_{m\to j,\ell}.
\tag{2.4}
$$

将原 CD 总量按 outgoing 发送源分配：

$$
c^{\mathrm{CD}}_{\ell,m}=\frac{1}{11\cdot10}\sum_{j\ne m}T_{m\to j,\ell},
\qquad \sum_m c^{\mathrm{CD}}_{\ell,m}=\operatorname{CD}_{\ell}.
\tag{2.5}
$$

式（2.5）是将已发表 CD 接入当前实验的归因选择，广播到各预测目标；不是新定义的“CD + self MI”。CD 原生排除自连接，本次没有填回对角 self MI。相对式（1.1），它还增加了对其他全部源的条件化。

这里使用有限历史、1—24月读出、冻结模型的干预 channel，不是从 ORAS5 原始观测序列估计的一步 TE，也不直接证明气候模态之间的观测因果关系。

### 2.4 连续变量计算

在共同代理中，$\boldsymbol y=\mathbf B^{\mathsf T}\boldsymbol x+\boldsymbol\eta$，$\operatorname{Cov}(\boldsymbol x)=\mathbf I$，$\operatorname{Cov}(\boldsymbol\eta)=\boldsymbol\Sigma$。令 $\mathbf G_m=\mathbf B_m^{\mathsf T}\mathbf B_m$。CD 使用

$$
T_{m\to j,\ell}=\frac12\log_2\left(1+\frac{(\mathbf G_m)_{jj}}{(\boldsymbol\Sigma)_{jj}}\right),\quad m\ne j.
\tag{2.6}
$$

历史 MIM 与当前 SI 均由同一 Gaussian 条件协方差的 log-determinant 信息量计算。非负容差1e-8 bit，超过负阈值失败；本轮容差内归零数、显著负值数均为0。CD对角置零来自排除自连接的定义，不是数值截断。

## 3. 扩展搜索中没有直接加入的候选

| 已发表候选 | 当前判断 | 理由 |
|---|---|---|
| Φ* mismatched decoding，Oizumi等2016 | 此设定下重复 ΦWMS | 独立过去块和逐块匹配 marginal decoder，使最优 mismatched information 等于分块 MI 之和 |
| 反向 integrated stochastic interaction | 重复 ΦWMS | 与WMS差为过去TC，独立干预源TC=0 |
| mRMR / MIFS | 退化为已移除的 singleton EI 加权 | 原始特征间MI冗余惩罚在当前独立源模型中为0；观测相关源时不成立 |
| 几何整合信息 ΦG | 有依据，下一批候选 | 需实现受限 Gaussian KL 优化并验证收敛，不能用SI或WMS的闭式表达冒充；本轮不启动此复杂工作流 |
| PED-Hsx synergy | 仍待统一表示 | 完整历史离散状态过多；只给PED压缩/二值化会改变比较因素，见[此前可行性报告](unicm_oinfo_wms_ped_feasibility.md) |
| ΦID synergy atoms / ψ | 本轮不新增 | 必须固定 redundancy 定义和所加原子；当前ΦR已经使用MMI冗余，MMI-PID也在用户排除范围内 |

Φ* 的来源为 [Oizumi等（2016）](https://doi.org/10.1371/journal.pcbi.1004654)。其重复关系是本实验条件下的推论：对于 $p(\boldsymbol h)=\prod_i p(\mathbf h_i)$、$q(\boldsymbol y\mid\boldsymbol h)=\prod_i p(Y_i\mid\mathbf h_i)$，mismatched 信息目标按部件分解。每项在β=1采用真实 marginal channel达到自身MI，故最优之和为 $\sum_i I(\mathbf h_i;Y_i)$，与WMS减项相同。这不宣称一般相关观测源下 Φ*=WMS。

ΦG 原始来源：[Oizumi、Tsuchiya、Amari（2016）](https://doi.org/10.1073/pnas.1603583113)，Gaussian数值条件亦见上述Kitazono附录A。不能把候选未补算解释为它表现不佳。O-Shapley、MMI-PID、target-averaged Ξ 按用户范围继续排除，未加回来；不新增单独二阶 PEID。

## 4. 搜索范围、证据及限制

先通过本地 Zotero 检索 PEID 和 integrated information；重新读取 PEID全文26/26页（MYATYWAJ/Y8GH3W58）、Mediano全文30/30页（X34436WI/AI552X6K）。公开原文核对覆盖经典整合信息、条件信息传递和信息论特征选择三支，不限最近年份。

检索词包括 `integrated information stochastic interaction geometric mismatched decoding`、`mutual information feature selection relevance Brown`、`causal density conditional transfer entropy information dynamics`；另外按作者、指标名和原文引用链查找 Φ*、ΦG、ΦSI、MIM、mRMR/MIFS、CD。Brown式（1）、Kitazono附录A、Mediano§2.2.8，以及Oizumi Φ*的定义/Methods均已读取。Seth2011使用PubMed元数据和摘要核对，CD数值定义采用已读Mediano全文。

公共API三组检索的自动排序出现明显无关结果、未来异常年份，已做人工语义筛除，不作为证据。Semantic Scholar及arXiv部分查询遇到429，另有arXiv超时；没有循环重试。Oizumi PMC副本遇到访问检查，改读可访问PLOS原文；Seth作者PDF未取得可靠全文，未以其支持具体公式。完整API失败URL记录保存在[本轮检索记录](../log/unicm_literature_search_provenance_20261002.json)中。搜索不是穷尽性系统综述，“本轮未找到”不等于“文献中从未存在”。

可复核原文：

- Brown等2012，JMLR13:27–66，[全文](https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf)及[期刊记录](https://www.jmlr.org/papers/v13/brown12a.html)。
- Kitazono等2018，Entropy20(3):173，[DOI](https://doi.org/10.3390/e20030173)及[公开全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512690/)。
- Mediano等2019，Entropy21(1):17，[DOI](https://doi.org/10.3390/e21010017)；本地Zotero完整PDF。
- Seth等2011，Philosophical Transactions A369:3748–3767，[DOI](https://doi.org/10.1098/rsta.2011.0079)及[PubMed](https://pubmed.ncbi.nlm.nih.gov/21893526/)。
- Oizumi等2016，PLOS Computational Biology12(1):e1004654，[PLOS全文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004654)。
- Oizumi、Tsuchiya、Amari2016，PNAS113(51):14817–14822，[DOI](https://doi.org/10.1073/pnas.1603583113)及[作者预印本](https://arxiv.org/abs/1510.04455)。
