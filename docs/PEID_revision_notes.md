# PEID 论文全文表述层面修改建议

> 范围：**仅表述/措辞/引文层面**，不涉及新增实验、新增证明、新增图表。
> 目标：在不改动方法和结果的前提下，把论文从"会被审稿人在事实、引文、因果话术上挑刺"的状态，调整为"边界清晰、引文忠实、表述自洽"的状态。
> 总工作量估计：约 2–3 小时。

---

## 总览：四类问题

通读全文，所有可在表述层面修复的问题归为四类：

1. **事实性错误**（必改）——站名、机构名、地名残缺或不准确。
2. **引文过度解读**（必改）——cited paper 没有说论文写的那么具体或那么强。
3. **因果话术过强**（强烈建议改）——在没有 do-calculus 保证的地方使用了"causal effect"。
4. **内部一致性 / 边界条件 / 学术规范**（锦上添花）——前后表述不冲突、假设说清楚、引用规范化。

下面按论文各部分顺序给出。

---

## A. Abstract、Highlights、Introduction（第 1 节）

### A1. Abstract 末句因果归因偏强

**当前：**
> "applying the framework to a machine-learning-based air quality forecasting task on KnowAir-V2, we demonstrate that PEID can extract interpretable inter-station causal structures from a learned dynamical model."

**问题：** "extract interpretable inter-station causal structures from a learned dynamical model"——读起来像是 PEID 提取了大气真实的因果结构。实际上 PEID 提取的是**学到的预测器在最大熵干预下的信息结构**，物理因果性需要额外假设。

**建议改为：**
> "applying the framework to a machine-learning-based air quality forecasting task on KnowAir-V2, we demonstrate that PEID can extract interpretable inter-station information-flow structures from a learned dynamical model, and discuss the assumptions under which such structures admit a causal interpretation."

### A2. Highlights 第 4 条

**当前：**
> "PEID extends to continuous nonlinear dynamics and enables interpretable causal analysis of machine-learning-based air quality forecasting models."

**建议改为：**
> "PEID extends to continuous nonlinear dynamics and enables interpretable analysis of information flow in machine-learning-based air quality forecasting models."

（把 "causal analysis" 弱化为 "analysis of information flow"，与 abstract 修改保持一致。）

### A3. Introduction 中关于 Granger causality 的两处小问题

**当前段落：**
> "the notions of causality adopted in these studies are still mostly grounded in the Granger-causality framework[1, 2, 4], rather than in an intervention-based definition of causation[3]. Granger causality characterizes a directed statistical dependence based on temporal precedence and predictive gain[1, 2]. However, in the absence of additional structural assumptions, this definition is not directly equivalent to causation in the underlying generative mechanism of the system, and it cannot automatically exclude spurious associations induced by latent confounders or common causes[11, 12]."

**问题：** 这段话本身没错，但把 [4]（Martínez-Sánchez et al. 2024, Nature Communications）一并归为"Granger-based"略有偏颇——该文的 Synergistic-Unique-Redundant 因果分解框架与 Granger 的关系比这里写的更微妙。建议把 [4] 单独拿出来表述。

**建议改为：**
> "...the notions of causality adopted in these studies are still mostly grounded in the Granger-causality framework[1, 2], or in information-decomposition extensions thereof[4], rather than in an intervention-based definition of causation[3]."

### A4. Introduction 中的"我们的主要贡献"段落

**当前：**
> "Our primary contribution is not to infer causal relations directly from observational data, but to provide an intervention-based framework for interpreting and analyzing dynamical models from a causal perspective."

**这一句写得很好，是论文最重要的范围声明。强烈建议把它在 abstract 和 conclusion 里也复述一遍**——这是回应审稿人因果质疑的最关键一句话，但目前只在 introduction 里出现一次，容易被忽略。

具体建议：
- 在 abstract 末尾加一句类似的范围声明
- 在 Section 4 开头第一段也复述一遍
- 在 Section 5 (Discussion) 第一段也提一次

---

## B. 第 2 节 Definition and Property

### B1. 2.1 节关于"qmax"的歧义

**当前：**
> "To the source variables at time t, we apply an intervention given by the maximum-entropy distribution, yielding the interventional distribution q^max(xt). For discrete systems, q^max is simply the uniform distribution over the input state space."

**问题：** "input state space" 指什么没有完全说清楚。在 multivariate 设定下，是 ∏ Ω_{X^(i)}（即各分量独立均匀），还是联合状态空间 Ω_{X_A}（这两者对离散等概率系统是一样的，但对连续或带约束系统不同）？

**建议改为：**
> "...yielding the interventional distribution q^max(xt) under which the source variables are mutually independent, each marginally uniform over its own state space. For discrete systems, q^max is simply the uniform distribution over the product of single-variable input spaces."

这一句改动很小，但能消除后面 Proposition 1 推导中"独立性"假设的来源歧义。

### B2. 2.2 节 Proposition 1 的陈述需要明确"在何种 PID 公理下"

**当前：**
> "Under an intervention that makes the source variables mutually independent, the PID redundancy term is zero..."

**问题：** 这个结论依赖 PID 的 Identity 公理 (Id)（Harder-Salge-Polani 2013，[9]），不是所有 PID 定义都自动满足。Williams-Beer 的 I_min 实际上**不**满足 Identity，所以这个命题对 I_min 不成立。

**建议改为：**
> "Under an intervention that makes the source variables mutually independent, and under any PID definition satisfying the standard axioms (S, I, M, LC, Id) including the Identity axiom (Eq. 56), the PID redundancy term is zero..."

并在 Appendix B 的开头明确写出："The proof relies on the Identity axiom; PID definitions that do not satisfy Identity (e.g., Williams & Beer's I_min) may not yield zero redundancy under maximum-entropy interventions."

### B3. 2.3 节 Definition 1 的命名

**当前：**
> "Definition 1 (unique effective information). For each part Mi ∈ P, the partition-level unique effective information from Mi to the target X^B_{t+1} is defined as Un^EID_P(Mi → X^B_{t+1}) := EI(Mi → X^B_{t+1})."

**问题：** 这个量在数值上**就是** EI(Mi → X^B_{t+1})，不是新概念，只是给它在分解里取了一个名字叫 "unique"。直接把它叫 "unique effective information" 而不解释，会让读者困惑——明明定义是 EI 本身，为什么叫 unique？

**建议加一句解释：**
> "Definition 1 (unique effective information). [...]
> 
> The name 'unique' reflects the role this term plays in the decomposition (Eq. 12): under the source-side maximum-entropy intervention, the EI of each part is fully attributable to that part itself, with no source-side redundancy. Numerically, Un^EID_P(Mi → X^B_{t+1}) coincides with EI(Mi → X^B_{t+1}), but conceptually it represents the unique-information atom in the PEID decomposition."

### B4. 2.4 节 EI causal graph 的命名

**当前论文从 Section 2.4 起反复使用 "EI causal graph"。**

**建议：** 在第一次出现的地方加一个 footnote 或括号说明：

> "The pairwise EI causal graph is defined as... [Footnote: We use 'causal graph' to refer to the graph whose edges are defined by intervention-based EI; whether this graph coincides with the structural causal graph of the underlying system depends on additional assumptions such as causal sufficiency and faithfulness, discussed in Section 5.]"

这样可以提前堵住审稿人的"这不是真正的 causal graph"之类的批评。

### B5. 2.6 节 Downward Causation 的术语

**当前：**
> "We define downward causation as the joint influence of the whole system on an individual at the next time step."

**问题：** "downward causation" 是哲学/复杂系统文献里有争议的术语。有 Rosas et al.（[14]）、Hoel et al.（[13]、[16]）、Bedau 等不同流派，定义不完全一致。

**建议：** 在第一次定义时加一句明确：

> "We define downward causation [in the sense of [14]] as the joint influence of the whole system on an individual at the next time step. Note that the precise mathematical content of 'downward causation' varies across the literature; the present definition (Eq. 42) follows the spirit of Rosas et al. [14] but is reformulated in terms of EI rather than partial information decomposition of mutual information."

---

## C. 第 3 节 Toy Examples

### C1. 3.1 节 Boolean networks 实验中"the whole is greater than the sum of its parts"用法

**当前：**
> "To verify that the measure Φ^EID(Xt) can appropriately quantify the property that a system as a whole is greater than the sum of its parts..."

**问题：** "the whole is greater than the sum of its parts" 在论文里被使用了至少 4 次，但严格来说 Φ^EID ≥ 0 只意味着 EI 不可加，不一定意味着哲学意义上的 emergence 或 holism。

**建议：** 第一次用时加引号、加限定，后面的复述就保持一致：

> "...quantify the property of dynamical non-additivity, often described informally as 'the whole being greater than the sum of its parts'..."

### C2. 3.2 节 Three-variable Boolean system 的小笔误

**当前：**
> "x0,t+1 = COPY(x2,t), x1,t+1 = AND(x0,t, x1,t), x2,t+1 = AND(x0,t, x1,t)."

**问题：** Equation 46 写了两个 AND，但根据后面的描述（"the 1 bit in the XOR gate is entirely synergistic effective information"），第三个应该是 XOR 不是 AND。

**这个看起来是 typo，请核对原始 Boolean 系统的定义并改正。** 大概率应该是：
> "x0,t+1 = COPY(x2,t), x1,t+1 = AND(x0,t, x1,t), x2,t+1 = XOR(x0,t, x1,t)."

### C3. 3.3 节关于"optimal coarse-graining"的强表述

**当前：**
> "EI-optimal coarse-graining yields macroscopic dynamics with a simpler causal structure"

**问题：** 这里"optimal"是指在所示的特定 partition 集合内最优，还是全局最优？没有明确证明全局最优性。

**建议改为：**
> "Among the partitions considered, the chosen coarse-graining (Eq. 48) maximizes macro-level total EI and yields macroscopic dynamics with a simpler causal structure."

### C4. 3.4 节 Downward Causation 关于 Rosas et al. 的措辞

**当前：**
> "Rosas et al. [14] previously provided two toy examples to explain dynamical decoupling and downward causation, whose designs are shown in Fig. 5(a) and (b)."
> ...
> "It is worth noting that Rosas et al. [14] did not distinguish the specific target variable when defining downward causation, whereas downward causation should evidently be defined as the influence of the whole system on a particular microscopic unit."

**问题：** "did not distinguish ... should evidently be defined" 这一句对前人工作的评价偏强，且 "should evidently" 是值判断而非论证。

**建议改为：**
> "Note that the original definition of downward causation in [14] was given at the level of the system as a whole, without distinguishing a specific target variable. Our formulation (Eq. 42) refines this by attributing downward causation to a particular microscopic unit, which we find both theoretically natural and empirically informative for the toy examples in Fig. 5."

### C5. 3.5 节关于连续 PEID 非负性的措辞

**当前（出现在 Section 5 Discussion）：**
> "In addition, for PEID with continuous variables, we cannot guarantee the non-negativity of synergistic effective information..."

**建议：** 这一句应当在 Section 3.5 第一次引入连续 PEID 时就提到，而不是只在 Discussion 里。

在 3.5 节 Eq. 51 之后加一句：
> "Note that, unlike the discrete case (Theorem 1), nonnegativity of Syn in the continuous setting is not guaranteed in general; small negative values may arise from finite-sample effects in transport-map density estimation. We treat such values as estimation artefacts and discuss the theoretical status further in Section 5."

---

## D. 第 4 节 Application（核心修改区）

### D1. 节首加一段范围声明

**建议在 Section 4 开头第一段后插入一段：**

> "Before presenting the results, we emphasize the scope of the analysis. The EI and Syn quantities reported in this section are computed on a learned MLP predictor under maximum-entropy interventions on its inputs. Their interpretation as causal quantities of the underlying atmospheric system requires the strong assumptions of causal sufficiency, faithfulness, and modular autonomy. Our primary aim is therefore to demonstrate that PEID can extract spatially interpretable information-flow structure from a learned forecasting model—not to claim that the extracted edges are validated atmospheric causal links. Section 5 discusses the relevant limitations in detail."

### D2. 站点身份与命名

#### D2.1 1228A 加院校沿革脚注

**当前：**
> "the strongest edges are highly concentrated among the outgoing edges from 1228A (Zhejiang Agricultural University)."

**建议加 footnote：**
> "Zhejiang Agricultural University (浙江农业大学) merged into Zhejiang University in 1998; the CNEMC station code 1228A continues to refer to the original campus, now Zhejiang University's Huajiachi Campus (浙江大学华家池校区), located in central-eastern Hangzhou near the Kaixuan Road traffic corridor. We retain the legacy station name to remain consistent with KnowAir-V2."

#### D2.2 3557A 站名残缺

**当前：**
> "3557A (Zhen No. 2 Middle School, Fuyang District)"

**问题：** "Zhen" 前面少了一个字（中文站名首字在英译过程中丢失）。

**修法：**
- 核对 KnowAir-V2 的原始站点列表或 CNEMC 名单，补全完整中文站名
- 推测可能是 "城厢镇第二中学" 或类似——务必核对再改
- 在英文里写为完整拼音 + (district)，例如 "[Town] No. 2 Middle School, Fuyang District" 或者直接保留中文："富阳区[XX]第二中学"

#### D2.3 全文范围内删除"background station / 背景"用法

**当前几处问题表述：**

| 出处 | 当前 | 建议改为 |
|---|---|---|
| 倒数第二段 | "Linping station represents a background with relatively strong ozone pollution in northeastern Hangzhou" | "Linping station represents a suburban site characterized by elevated ozone levels in northeastern Hangzhou" |
| 同段稍后 | "the neighboring scenic-background station 1227A" | "the neighboring scenic-area urban site 1227A" |

**理由：** 杭州在 CNEMC 国控背景点名单中**没有任何**站点；全国背景点只有约 16 个远郊点（长白山、神农架、洱海等）。Linping 和云栖在 Su et al. 2018 (AAQR) 等文献中被分类为 "suburban" 或 "urban"，不是 "background"。这是一个会被熟悉中国监测网络的审稿人立刻识别的错误。

### D3. 引文转述忠实度修复

#### D3.1 [45] Chen et al. 的转述需收紧

**当前：**
> "This is consistent with the interpretation that Linping station represents a background with relatively strong ozone pollution in northeastern Hangzhou [45]."

**问题：** [45] 只报告了杭州 12 站东高西低的粗略空间梯度，没有单独把临平定性，也没有用 "background" 一词。

**建议改为：**
> "This is consistent with [45], who report an east-higher–west-lower spatial gradient of ozone across Hangzhou stations, with the northeastern area exhibiting relatively elevated ozone levels."

#### D3.2 [46] Han et al. 的转述需大幅软化

**当前：**
> "Moreover, hourly local airflow in Hangzhou can transport ozone from Linping to other regions [46]."

**问题：** [46] 分析的是 2018 年 6 月 5 日单次个例下海陆风、城市热岛、山谷环流的耦合，**完全没有把临平识别为源区**。"from Linping to other regions" 是论文加上去的方向性论断，超出 [46] 的实际内容。

**建议改为：**
> "Moreover, [46] show that hourly local circulations in the Hangzhou metropolitan area, including sea–land breezes, urban heat-island circulations, and valley flows, can mediate inter-station ozone redistribution."

#### D3.3 [47] Lin et al. 的对应关系需弱化

**当前：**
> "Station 3558A (Lin'an Municipal Government Building) is located in a foothill basin-like environment, surrounded by mountains on three sides, where particulate matter and meteorological factors exhibit obvious lagged relationships [47]."

**问题：** 地理描述本身没错，"PM-气象滞后关系"也是 [47] 的核心结论，但 [47] 用的监测点不一定就是 3558A——这个对应关系无法直接核实。

**建议改为：**
> "Station 3558A is located in Lin'an, in a foothill area at the eastern margin of the Tianmu Mountain range, with mountains to the north, west, and south. In this region, [47] document hysteretic relationships between particulate matter and local meteorological factors, with PM responding to meteorological forcing on a lag of hours to days."

#### D3.4 [48] Wang et al. 的小词调整

**当前：**
> "Urban-core traffic, commercial activities, solvent use, and combustion sources jointly affect VOCs, NOx, secondary aerosols, and PM2.5..."

**问题：** [48] 给出的 PMF 因子是 solvent (24.1%)、combustion (22.7%)、vehicle exhaust (19.8%) 等；"commercial activities" 不是其中任何一项。最接近的应是 "industrial sources" 或 "stationary sources"。

**建议改为：**
> "Urban-core traffic, industrial and stationary sources, solvent use, and combustion jointly affect VOCs, NOx, secondary aerosols, and PM2.5..."

#### D3.5 [50] Dewan & Lakhani 引用范围限定

**当前：**
> "This is likely because O3 is a secondary pollutant whose lifetime in cities can be as short as several hours, so its cross-station influence depends strongly on wind fields, local circulation, and photochemical conditions [50]."

**问题 1：** "几小时"那部分 [50] 直接支持。但用 [50] 同时支持"风场、局地环流、光化学"略勉强——这部分更对口的引文是 Monks et al. 2015 (ACP) 或 Lu et al. 2019 (Curr. Pollut. Rep.)。

**问题 2：** "几小时"这个寿命描述与论文 12h 预测窗口的内部张力（见 D5）。

**建议改为：**
> "This is likely because urban tropospheric O3 has a strongly variable photochemical lifetime—on the order of hours near intense sources due to NO titration and HOx-rich conditions, but typically extending to one to two days under transport-favorable circulations [50]—so its cross-station influence depends strongly on wind fields, local circulation, and photochemical conditions."

如果方便，可以把 [50] 替换为更切题的引用（Monks 2015 或 Lu 2019）；如果不想动参考文献编号，至少把"几小时 vs 几天"这个区间写清楚。

### D4. 因果话术全面降级（这是 Section 4 最重要的修改）

**核心原则：** 不要在叙述具体边和具体站点关系时使用 "causal effect" / "causal influence"。这个词只保留给方法论定义部分（Section 2）和讨论限制部分（Section 5）。在 Section 4 描述结果时，统一改为：
- "model-based predictive influence"
- "information flow in the learned predictor"
- "intervention-based response of the surrogate model"

#### D4.1 "1231A behaves as a stable hub node"

**当前：**
> "This indicates that 1231A (Linping station) behaves more like a stable hub node that outputs O3 outward."

**建议改为：**
> "This indicates that, in the learned predictor, perturbations to input O3 at 1231A propagate to multiple downstream stations under maximum-entropy intervention, consistent with—but not by itself demonstrating—1231A acting as an upwind ozone source under prevailing circulations."

#### D4.2 "3558A 既是 receptor 又把臭氧传给 1227A"

**当前：**
> "Therefore, 3558A can act both as a receptor and, at the 12h scale, as a station that transmits ozone signals filtered by the basin and foothill circulation to the neighboring scenic-background station 1227A."

**建议改为：**
> "Therefore, in the learned predictor, 3558A appears both as a target of strong incoming edges and as a source of an outgoing edge to 1227A at the 12h scale, a pattern compatible with 3558A receiving ozone signals modulated by the foothill–basin circulation before they propagate to the neighboring scenic-area site."

#### D4.3 "1228A 的 PM2.5 对 3557A 的 O3 有独立因果效应"

**当前：**
> "This indicates that PM2.5 at 1228A has a relatively strong independent causal effect on the ozone level at 3557A, whereas ozone at 1228A has almost no causal effect on it."

**建议改为：**
> "This indicates that, in the learned predictor, PM2.5 at 1228A contributes a non-redundant unique predictive component to forecasted O3 at 3557A, whereas O3 at 1228A contributes essentially none."

#### D4.4 协同项与"协同管控"的因果跳跃

**当前：**
> "This indicates that the coordinated control of PM2.5 and O3 is an important issue, especially for ozone pollution caused by urban-core traffic."

**问题：** 预测模型中的统计协同 ≠ 排放干预层面的非可加耦合。从前者直接推到后者是因果跳跃。

**建议改为：**
> "This is consistent with the established atmospheric-chemistry view that PM2.5 and O3 are non-additively coupled in urban environments, which has motivated joint management strategies in the literature [48, 49]. We note, however, that synergy in the present sense is a statistical property of the learned predictor and does not by itself constitute evidence for the effectiveness of joint emission control."

### D5. 内部一致性：12h 窗口与 O3 寿命的张力

**问题：** 论文一方面用 [50] 论证"O3 in cities is short-lived (several hours)"，另一方面用 EI 分析 12h 站间输送。如果 O3 真只活几小时，12h 预测主要在学日变化气候态，这削弱了 EI 想揭示的站间结构。

**建议在 Section 4 第一段或方法描述处加一段：**

> "The 12-hour forecast horizon spans an entire diurnal photochemical cycle, during which urban O3 undergoes both production (daytime, NOx + VOC photochemistry) and removal (nighttime NO titration, dry deposition). While near-source urban O3 lifetime can be as short as a few hours, regional-scale O3 transported under stable circulations persists on a timescale of one to two days, which provides the physical basis for inter-station information flow at this horizon. The 12-hour choice is also motivated by its relevance as a policy-relevant warning lead time for ozone-pollution episodes."

### D6. 关于"无混杂"假设的措辞需更诚实

**当前：**
> "It is worth noting that, when using an MLP to fit the dynamics underlying the data, we assume that there are no other confounding factors affecting changes in O3 beyond the PM2.5 and O3 variables used here and the other eight meteorological variables. The specific geographical distribution and meteorological conditions of the Hangzhou region have already been incorporated into the information fitted by the MLP parameters, and they constitute part of the causal structure underlying air pollution."

**问题：** 这段话写得太轻描淡写。**特别是没提到 KnowAir-V2 实际包含的 MEIC 排放变量（NOx、VOC、SO2、NH3、PM2.5、PM10 emissions）在本实验中并未被使用**——审稿人会立刻问这个问题。

**建议改为：**

> "Several caveats apply to the interpretation of the inferred edges. First, the EI and Syn quantities reported here are properties of the learned MLP under input-side maximum-entropy interventions, and their interpretation as atmospheric causal links requires causal sufficiency. Second, our input features comprise 2 pollutants (O3, PM2.5) and 8 meteorological variables, whereas KnowAir-V2 also provides MEIC anthropogenic emission inventories (NOx, VOC, SO2, NH3, primary PM2.5 and PM10) that are direct exogenous drivers of both species; any inter-station edge mediated by a shared upwind emissions source could therefore be partially misattributed under the present configuration. Third, the spatial graph is restricted to 12 stations within Hangzhou's administrative boundary, so transport pathways from outside this window (e.g., from neighboring YRD cities) are not represented and are absorbed into the learned mapping. The geographical and meteorological context of Hangzhou is partially captured in the MLP parameters, but this should not be interpreted as a complete resolution of the confounding structure underlying air pollution. We return to these points in Section 5."

### D7. G.1 节补 12 站选取依据

**当前 G.1 节：**
> "In the present study, we use the preprocessed hourly records and extract the Hangzhou station network from the YRD portion of the dataset for the continuous effective information analysis."

**问题：** 没说"为什么是这 12 站"。KnowAir-V2 公开发布并没有规范的"杭州 12 站子集"，所以这是作者的子样本选取，必须文档化。

**建议加一句：**
> "The 12-station Hangzhou subset corresponds to all CNEMC monitoring stations within Hangzhou's administrative boundary (covering the urban core, Lin'an, Linping, Fuyang, Tonglu and Chun'an districts) that have data coverage above [X]% over the 2016–2023 KnowAir-V2 period, after the dataset's standard imputation procedure."

（具体阈值或选取规则按你实际操作填写。如果其实是用了一个固定的距离阈值或行政区划过滤，把那个规则写出来即可。）

### D8. G.2 节关于 baseline 的措辞

**当前：**
> "The baseline is a persistence model: it directly uses the current input snapshot of O3 and PM2.5 as the prediction for all future horizons."

**建议加一句：**
> "We note that persistence is a minimal baseline used here only to verify that the MLP captures non-trivial dynamics rather than reproducing the input. We do not claim that the MLP is state-of-the-art on KnowAir-V2; stronger surrogates such as PCDCNet [51] or graph-based models would likely yield lower forecast error. Our purpose is to obtain a smooth, differentiable, jointly trained multistation predictor on which intervention-based EI and Syn can be computed, not to advance forecast accuracy."

这段补充能预先回应"为什么不用更强的模型"这类审稿意见。

---

## E. 第 5 节 Discussion

### E1. 第一段开头加范围声明

**建议在 Section 5 第一段开头加一句：**
> "We reiterate that the framework proposed in this paper is intended for interpreting and analyzing dynamical models from a causal perspective, not for inferring atmospheric causal relations directly from observational data. The interpretive value of PEID for real systems depends on the degree to which a learned dynamical surrogate faithfully approximates the underlying mechanism, and on standard structural assumptions including causal sufficiency, faithfulness, and modular autonomy."

### E2. Limitations 段落需要扩展

**当前 Limitations 部分提到了三点：** (1) 真实数据的最大熵干预合理性、(2) target-side 冗余、(3) 连续 PEID 非负性、(4) Markov 假设。

**建议补充以下几点（都是表述层面、不需要新实验）：**

#### E2.1 Off-manifold intervention 问题

> "Fifth, in continuous systems with strongly correlated inputs—such as PM2.5 and O3 in air-quality data, which exhibit pronounced seasonal and meteorological anti-correlation—maximum-entropy interventions on each input dimension produce input combinations that lie outside the empirical joint distribution. Neural network predictors are known to extrapolate erratically off-manifold, which can bias EI estimates in direction-dependent ways across edges. Future work should examine on-manifold alternatives such as conditional permutation interventions, or weight EI estimates by the local in-distribution density."

#### E2.2 PID estimator 依赖性

> "Sixth, the PID atom decomposition is not uniquely defined: different redundancy measures (Williams & Beer's I_min [6], Bertschinger et al.'s I_BROJA [8], Ince's I_ccs, Finn & Lizier's I_PM [10]) can yield qualitatively different decompositions on the same data. The PEID framework as developed here inherits the redundancy-vanishing property under independent maximum-entropy interventions and is therefore robust to this choice in the source-side decomposition; however, any extension to target-side decomposition (see point 2 above) would need to specify and justify a particular estimator."

#### E2.3 Statistical inference

> "Seventh, the present analysis reports point estimates of EI and Syn without uncertainty quantification or multiple-comparison control. For applications involving many candidate edges (such as the 12-station Hangzhou case, which involves order 264 directed pairs), bootstrap or jackknife confidence intervals and false-discovery-rate control would be needed to distinguish robust structure from sampling artefacts. We leave a systematic statistical-inference treatment to future work."

### E3. 最后一段的措辞

**当前最后一段（"Overall, this paper provides a new method..."）写得很好**，唯一建议是把最后一句的"PEID is expected to become a general tool for analyzing multiscale causal structures and emergent mechanisms in complex systems"——"general tool" 略大，可以改为：

> "...PEID provides a principled and computable interventionist information-theoretic toolkit for analyzing multiscale causal structures and emergent mechanisms in complex systems, and we hope it will complement existing causal-discovery methodologies."

---

## F. Appendices

### F1. Appendix A 公理列表的小问题

**当前 Appendix A 在列出 PID axioms 时：**
- 列了 Symmetry (S)、Self-redundancy (I)、Monotonicity (M)、Identity (Id)
- 但 Lemma 1（Appendix B）的前提里出现了 "axioms (S, I, M, LC, Id)"，多出一个 LC

**问题：** Appendix A 没有定义 LC（Local Chain Rule / Target Chain Rule）公理。Appendix B 的 Lemma 1 证明里直接用了 LC 但没有先引入。

**修法：** 在 Appendix A 加一条：

> "**Local chain rule (LC).** For two source variables A1, A2 and a composite target (S1, S2):
> Red(A1, A2; S1, S2) = Red(A1, A2; S1) + Red(A1, A2; S2 | S1)."

或者，如果你想保持 Appendix A 的简洁，至少在 Lemma 1 引入 LC 时给出它的明确定义。

### F2. Appendix B 中 Eq. (60)–(61) 的下标小笔误

**核对一遍：**
> Red(U, V; U, V, T) = Red(U, V; U, V) + Red(U, V; T | U, V) (60)
> Red(U, V; U, V, T) = Red(U, V; T) + Red(U, V; U, V | T) (61)

这两个等式来自 LC 公理，符号方向需要核对——(60) 和 (61) 是把 LC 应用到不同的拆分上。建议检查一下哪个变量在条件位置上是否一致，特别是 (61) 是否应该是 Red(U, V; T) + Red(U, V; U, V | T) 还是 Red(U, V; T) + Red(U, V; U | V, T) + Red(U, V; V | U, T) 之类。

（这是数学细节，需要你或合作者重新过一遍——我无法替你验证，但这是审稿人会逐行核对的地方。）

### F3. Appendix E 的 typo

**当前 Appendix E 第一句：**
> "As a supplement to Fig. 7.4(b) in Sec. 6.4..."

**问题：** 论文里没有 Section 6.4，也没有 Fig. 7.4。这显然是从一个早期版本的 cross-reference 留下的。

**修法：** 改为正确的章节号（应该是 Sec. 3.3，Fig. 4 的对应位置）。

### F4. Appendix F 关于 transport map 的小问题

**当前：**
> "the basic form of the transport map, the density transformation formula, and the implementation details for estimating continuous EI are provided in Appendix F."

**这部分写得不错**，但有一处建议：

**当前：**
> "small negative values are treated as finite-sample or density-model mismatch artifacts."

**建议改为：**
> "small negative values, when observed, are reported as-is and interpreted as finite-sample or density-model mismatch artifacts; we do not clip them to zero, to preserve transparency about estimation uncertainty."

### F5. Appendix G.1 关于站点图构建的措辞

**当前：**
> "In the original PCDCNet benchmark, the station graph is constructed by connecting monitoring stations within a 200 km geodesic-distance threshold... In the present analysis, however, the graph visualized in Section 4 is not directly taken from this predefined spatial adjacency graph. Instead, station coordinates are used to place nodes at their geographic locations, while the displayed directed edges are inferred from the estimated EI_tm and Syn_tm quantities."

**这段写得很清楚，没有问题**——这正是审稿人会想确认的，保留即可。

---

## G. References / 引用规范

### G1. 几条引文格式不统一

通读 References，发现以下规范不一致：

- [4] Martínez-Sánchez 等的姓氏带重音：检查 LaTeX 源里是否正确编码（如 `\'a` 或 UTF-8 直接字符），避免 PDF 里出现乱码
- [20] Schölkopf 同上，注意 ö 的编码
- [11] Eichler 期刊名缩写为 "Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences"，但 [32] 同期刊用了相同全称——一致性 OK
- [25] Kořenek 的姓氏带 ř，确认编码
- [29] Varley 期刊 "Entropy" 卷期信息：26(10):883 — 这是 article number 还是页码？Entropy 是 article-number journal，应该写为 "26:883" 或 "26, Article 883"
- [30] Faes et al. arXiv:2603.07634 — 这个 arXiv ID 看起来不太对，arXiv ID 格式通常是 YYMM.NNNNN，2603 月份不存在。**核对原始引用，可能是 typo**
- [49] Tao et al. 2024：DOI 加上更好

### G2. arXiv 预印本与正式发表

- [5] Yuan et al. Entropy 2024 — 已正式发表，OK
- [17] Zhang et al. npj Complexity 2024 — 已正式发表，OK
- [38] Zhang Entropy 2022 — 已正式发表，OK
- [39] Yang et al. National Science Review 2025 — 已正式发表，OK
- [44] Wang et al. KnowAir-V2 Zenodo 2025 — OK
- [51] Wang et al. PCDCNet arXiv 2025 — 看是否已发表期刊版，如果有应替换
- [40] Yang, Pan, Zhang 2025 Digital Technologies Research and Applications — 核对 OA 期刊页码
- [41] Lyu, Clark, Raviv Physical Review E 2026 — 核对是否已正式刊出

### G3. 自引

论文自引了 [44] (KnowAir-V2)、[51] (PCDCNet)、[40] (Yang, Pan, Zhang 2025) 等——这是合理的，因为方法和数据都是基于自己之前工作的。但建议在第一次使用 KnowAir-V2 时明确说明 "we use KnowAir-V2 [44], introduced in our prior work [51]" 之类，避免审稿人误以为是隐藏自引。

---

## H. 全文一致性检查清单

最后，过一遍全文统一性：

- [ ] "causal effect" 在 Section 4 (Application) 全部替换为 "model-based predictive influence" / "intervention-based response" 等
- [ ] "background station" 全部删除，按 D2.3 替换
- [ ] 1231A 的描述前后一致：永远是 "suburban site with elevated ozone in NE Hangzhou"
- [ ] 3557A 站名补全
- [ ] 1228A 第一次出现时加院校沿革 footnote
- [ ] Φ^EID 的符号在全文统一（Section 2 用的是带 EID 上标的 Φ，Section 3.1 也是，但 Highlights 里用的是 "synergistic causal contributions" 散文表述，OK）
- [ ] EI_tm 和 Syn_tm 的下标 tm（transport map）在 Appendix F 第一次出现时定义清楚
- [ ] 所有 Eq. 编号顺序检查（Appendix E 的 cross-reference 错乱已经是一个例证）
- [ ] 图 7 caption 里的 self-loop 颜色条范围 [-0.10, 0.10] 在正文里应有解释——这是什么单位？bit？需要补图注

---

## 修改优先级矩阵

| 优先级 | 修改项 | 工作量 |
|---|---|---|
| 🔴 必改 | D2.2 补全 3557A 站名 | 1 分钟 |
| 🔴 必改 | D2.3 全文删除 "background" | 5 分钟 |
| 🔴 必改 | D3.2 软化对 [46] 的转述 | 2 分钟 |
| 🔴 必改 | D3.1 软化对 [45] 的转述 | 2 分钟 |
| 🔴 必改 | C2 核对 Eq. 46 的 AND/XOR typo | 5 分钟 |
| 🔴 必改 | F3 修正 Appendix E 的 cross-reference | 1 分钟 |
| 🟠 强烈建议 | D1 Section 4 节首加范围声明 | 5 分钟 |
| 🟠 强烈建议 | D4 因果话术降级（4 处） | 15 分钟 |
| 🟠 强烈建议 | D6 把假设说诚实，承认丢了 MEIC | 15 分钟 |
| 🟠 强烈建议 | A1 + A4 + E1 三处范围声明 | 10 分钟 |
| 🟠 强烈建议 | D5 化解 12h 与"几小时"的张力 | 5 分钟 |
| 🟠 强烈建议 | D7 补 12 站选取依据 | 5 分钟 |
| 🟡 锦上添花 | B1 + B2 + B3 + B4 + B5 概念边界澄清 | 20 分钟 |
| 🟡 锦上添花 | C1 + C3 + C4 + C5 toy example 措辞 | 15 分钟 |
| 🟡 锦上添花 | E2 扩展 Limitations | 20 分钟 |
| 🟡 锦上添花 | D2.1 1228A 院校沿革 footnote | 3 分钟 |
| 🟡 锦上添花 | D3.3 + D3.4 + D3.5 引文措辞微调 | 10 分钟 |
| 🟡 锦上添花 | D8 baseline 措辞 | 5 分钟 |
| 🟡 锦上添花 | F4 transport map 措辞 | 3 分钟 |
| 🟡 锦上添花 | A2 + E3 Highlights 和 Discussion 末句 | 5 分钟 |
| 🟡 锦上添花 | G References 一致性 | 15 分钟 |
| 🟡 锦上添花 | H 全文一致性扫一遍 | 20 分钟 |

**总计：**
- 仅做必改 + 强烈建议：约 1.5 小时
- 完整做完所有：约 3 小时

---

## 一句话总结

这篇论文的方法和实验是扎实的，但 **Section 4 的表述把"MLP 在均匀输入扰动下的属性"叙述成了"大气的因果结构"**，这是最大的可挑剔之处。把这一层措辞收紧、把几处引文转述还原到原文支持的范围内、修两三处事实小错（站名残缺、background 用词、Appendix E cross-ref），论文在不做任何新实验的前提下就能显著提高审稿通过率。

其他章节的修改大多是锦上添花——B、C、E 节的概念边界澄清和 Limitations 扩展，能让评审觉得作者对方法的边界有清晰认识；引用规范化是必要的工程性收尾。
