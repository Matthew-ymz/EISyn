# 83 ROI DMF 临界窗实验：数据来源、处理流程与复现说明

## 1. 文档目的与适用范围

本文说明“83 ROI Dynamic Mean Field（DMF）临界窗多尺度汇总”实验中所有数据对象的来源、含义、下载方式、清洗步骤、模拟过程、估计方法、功能网络映射和质量控制。本文面向未接触原始开发环境的研究人员，可独立阅读，不依赖任何个人计算机目录或本地绝对路径。

本文只说明该实验及其汇总图，不覆盖同一研究中另行开展的 HCP Schaefer-500、Schaefer-1000、工作记忆任务态或其他脑网络实验。

最重要的口径如下：

1. 实验真正下载并用于 DMF 耦合拓扑的数据来自 **F-TRACT atlas 21-12 版**的 Lausanne2008-33 数据，而不是 HCP 扩散 MRI 纤维追踪矩阵。
2. 实验使用的是 F-TRACT `probability` 条目所附的 **`count` 观测支持矩阵**，没有使用 `probability` 数值本身，也没有使用 `amplitude`、`latency`、`duration` 等响应特征。
3. F-TRACT 数据只提供 83 个脑区之间的经验代理耦合拓扑。图中的发放率、E/I 门控状态、EI、$\Phi^{EID}$ 和 observational WMS 都是 DMF 数值模拟及其后续估计的结果，不是受试者逐时刻脑活动记录。
4. Yeo-7 数据只用于把 68 个皮层 ROI 分成功能网络；fsaverage5 和 Desikan2006 只用于表面显示。它们不参与 DMF 动力学拟合。
5. 当前实验是对 Mediano 等（2025）DMF 分析的**代理数据近似复现和扩展分析**，不是原论文 HCP Lausanne-83 结构连接结果的精确复现。

## 2. 数据对象总览

| 数据对象 | 来源 | 是否为真实观测数据 | 在实验中的作用 |
|---|---|---:|---|
| F-TRACT Lausanne2008-33 `count` 矩阵 | F-TRACT atlas 21-12，CCEP 汇总 | 是，群体级汇总 | 构造 83×83 有向代理耦合矩阵 |
| Lausanne2008-33 ROI 标签 | 同一 F-TRACT 数据包 | 是，图谱元数据 | 确定 68 个皮层、14 个皮层下和 1 个脑干 ROI |
| Yeo2011 七网络注释 | Thomas Yeo Lab / CBIG | 是，1000 名被试静息态 fMRI 推导的群体图谱 | 将 68 个皮层 ROI 分入七个功能网络 |
| Desikan2006 皮层注释 | FreeSurfer / TemplateFlow | 群体模板 | 把 ROI 数值投影到皮层表面 |
| fsaverage5 表面网格 | FreeSurfer，经 nilearn 获取 | 群体模板 | 绘制左右半球外侧、内侧四视角 |
| 自发 DMF 发放率轨迹 | 数值模拟 | 否 | 提供全局耦合扫描的动力学参照 |
| 最大熵干预 source/target 样本 | 数值模拟 | 否 | 计算 whole EI、单变量 EI 之和和 $\Phi^{EID}$ |
| 自然稳态 source/target 样本 | 数值模拟 | 否 | 计算与干预实验对齐的 observational $\Phi^{WMS}$ |
| ROI、结构模块和 Yeo-7 分解结果 | 上述模拟数据的派生量 | 否 | 形成汇总图 D–G 面板 |

## 3. 外部原始数据的下载

### 3.1 F-TRACT atlas

本实验使用 F-TRACT atlas 2021 年 12 月发布的 **21-12 版**。官方记录说明，该版根据 780 名患者在立体脑电（SEEG）直接电刺激期间记录的 cortico-cortical evoked potentials（CCEP）估计脑区间连接概率及多种反应特征，覆盖 Lausanne2008 等多种分区。

- 数据记录页：[F-TRACT atlas, Zenodo record 7015415](https://zenodo.org/records/7015415)
- DOI：[10.5281/zenodo.7015415](https://doi.org/10.5281/zenodo.7015415)
- F-TRACT 数据说明页：[Functional Brain Tractography Project — Atlas](https://f-tract.eu/atlas/)
- 完整发布包直接下载：[f-tract_v2112.zip](https://zenodo.org/records/7015415/files/f-tract_v2112.zip?download=1)
- 完整发布包页面标注大小：915.3 MB
- 完整发布包官方 MD5：`df03571e8fce5b030482f239dd0d04ab`
- 发布日期：2021-12-09
- 版本：21-12

可使用以下通用命令下载并校验完整发布包：

```bash
curl -L -o f-tract_v2112.zip \
  "https://zenodo.org/records/7015415/files/f-tract_v2112.zip?download=1"
md5sum f-tract_v2112.zip
unzip f-tract_v2112.zip
```

macOS 可将 `md5sum` 替换为 `md5`。

完整发布包包含多种脑区分区和分辨率。本实验只使用其中的 `Lausanne2008-33.zip`。本次审计所用子归档具有以下特征：

| 项目 | 值 |
|---|---|
| 文件名 | `Lausanne2008-33.zip` |
| 精确字节数 | 14,394,702 bytes |
| SHA-256 | `8038a4cf2584acf0ece80531bfe55710657c9e826dffbf206ef533188de4b843` |
| 图谱标称规模 | 84 parcels |
| 最终实验规模 | 83 ROI |

### 3.2 Yeo2011 七网络注释

Yeo2011 七网络图谱来自 1000 名被试的静息态功能连接数据，使用表面配准和聚类得到七个大尺度皮层功能网络。官方 CBIG 发布同时提供 fsaverage5 表面空间的 7/17 网络注释。

- 官方代码与数据说明：[CBIG Yeo2011 fcMRI clustering](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering)
- 建议固定版本：[CBIG v0.19.2-Yeo2011_Schaefer2018](https://github.com/ThomasYeoLab/CBIG/releases/tag/v0.19.2-Yeo2011_Schaefer2018)
- 原始论文：[Yeo et al., 2011](https://doi.org/10.1152/jn.00338.2011)

七个网络为 Visual、Somatomotor、Dorsal attention、Salience/ventral attention、Limbic、Frontoparietal control 和 Default mode。Yeo-7 只覆盖大脑皮层，不给皮层下结构和脑干分配网络标签。

### 3.3 fsaverage5 与 Desikan2006 表面资源

脑表面图使用 fsaverage5 膨胀表面网格及 sulcal depth 背景。Desikan2006 注释通过 TemplateFlow 的 fsaverage、10k density、`aparc` 分割获取；10k density 对应约 10,242 个顶点/半球。

- TemplateFlow 资源浏览器：[TemplateFlow Archive](https://www.templateflow.org/browse/)
- TemplateFlow Python 客户端说明：[TemplateFlow](https://www.templateflow.org/)
- 图中使用的注释对象命名形式：`tpl-fsaverage_hemi-{L|R}_den-10k_atlas-Desikan2006_seg-aparc_dseg.label.gii`

这些表面资源只影响 G 面板的可视化，不改变任何 EI、$\Phi^{EID}$、相关系数或网络分解数值。

## 4. F-TRACT 子归档的内容与实际选取

### 4.1 归档内容

`Lausanne2008-33.zip` 同时包含：

- 一个 protocol-5 pickle 文件；
- 同一 pickle 的 gzip 压缩版本；
- 按特征导出的 CSV 和逐 ROI 文本文件；
- 84 个 ROI 的标签与方向说明。

导出的特征包括 probability、amplitude、peak/onset latency、duration、integral、fiber/euclidean distance、velocity、axonal conduction delay、synaptic excitatory/inhibitory delay 等。pickle 展平后含 19 个条目。

### 4.2 `count_00` 到底是什么

代码读取 pickle 展平后的第 0 个条目，并只取其中的 `count` 字段。通过把 pickle 中的统计矩阵与归档内各特征 CSV 逐元素比较，可以确认：

- 第 0 个条目的 `mean` 与 `probability.csv` 完全一致；
- 因此第 0 个条目是 **probability 条目**；
- 实验实际取的是 probability 条目附带的 `count`，不是 probability 的 `mean`；
- `count` 是每个刺激 ROI—记录 ROI 组合用于估计连接概率的观测支持数量；
- `probability` 数值、amplitude 及其他 18 个特征均未进入当前耦合矩阵。

所以，把当前矩阵称为“F-TRACT 结构连接矩阵”或“连接概率矩阵”都不够准确。最严格的名称是：

> **F-TRACT Lausanne2008-33 probability 条目的 CCEP 观测支持计数所派生的 83 ROI 代理耦合矩阵。**

它不是白质纤维数量、扩散 MRI tract count、概率值本身或个体级功能连接。

### 4.3 矩阵方向

F-TRACT CSV 的元数据明确写明：

- 行（纵向）是 stimulated parcels；
- 列（横向）是 recorded parcels。

因此，原始 `count[a,b]` 的官方方向是“刺激 ROI $a$，在 ROI $b$ 记录”。

当前实验在载入后没有转置矩阵，而 DMF 长程输入使用矩阵左乘状态向量，即第 $j$ 行被解释为目标 ROI $j$ 接收的输入。由此，**实际执行代码对 F-TRACT 官方刺激→记录方向作了反向解释**。这一点不会改变使用对称化矩阵计算的结构强度，但会改变有向 DMF 传播方向。该事实必须作为当前结果的复现口径和局限保留；如未来决定转置矩阵，应视为一个新的方向敏感性实验，不能与本图数值混用。

## 5. 从 84×84 `count` 到 83×83 代理耦合矩阵

### 5.1 标签读取

标签从 amplitude 导出的 CSV 表头读取。选择 amplitude CSV 只为读取共同 ROI 顺序，不表示使用了 amplitude 数值。标签总数为 84：

- 左半球皮层 34 个；
- 右半球皮层 34 个；
- 双侧皮层下结构 14 个；
- Brain-Stem 1 个；
- Unknown 1 个。

### 5.2 删除 `Unknown`

对行和列同时删除唯一的 `Unknown`，矩阵由 84×84 变为 83×83。代码要求恰好找到一个 `Unknown`；数量不是 1 时立即报错，不会静默删除其他标签。

### 5.3 缩放、对角线和缺失值处理

实际处理顺序为：

1. 把 `count` 转为浮点数组；
2. 删除 `Unknown` 的行与列；
3. 取删除 `Unknown` 后、**仍包含对角线**的 83×83 矩阵全局最大值；
4. 所有元素除以该最大值，再乘以 0.2；
5. 最后把主对角线强制设为 0；
6. 检查矩阵为方阵、元素有限、非负且总权重大于 0。

写成公式：

$$
\widetilde{C}_{jk}=0.2\frac{N_{jk}}{\max_{a,b}N_{ab}},
\qquad
C_{jk}=\begin{cases}
0,&j=k,\\
\widetilde{C}_{jk},&j\ne k.
\end{cases}
$$

其中 $N_{jk}$ 是删去 `Unknown` 后的 count 值。

这一顺序有一个容易忽略的结果：全局最大值 8,151 位于对角线上，对角线在缩放后才被清零；最大非对角 count 为 4,332。因此最终非对角最大权重不是 0.2，而是

$$
0.2\times\frac{4332}{8151}=0.1062937063.
$$

当前流程没有执行以下操作：

- 不对矩阵做转置；
- 不对矩阵做对称化；
- 不做行归一化或谱半径归一化；
- 不取对数、不做分位数缩放；
- 不设稀疏阈值；
- 不对非零 count 做二值化；
- 不用 probability、mean、median、std 或 MAD 替换 count；
- 不插补缺失值；所选 count 矩阵本身全部为有限整数。

### 5.4 处理前后的审计统计

| 指标 | 值 |
|---|---:|
| 原始矩阵 | 84×84 |
| 原始有限元素 | 7,056 / 7,056 |
| 原始非零元素 | 3,549 |
| 原始 count 范围 | 0–8,151 |
| 删除 `Unknown` 后矩阵 | 83×83 |
| 非对角可能边数 | 6,806 |
| 最终非零有向边数 | 3,477 |
| 非对角密度 | 51.0873% |
| 最终权重范围 | 0–0.1062937063 |
| 最终权重总和 | 16.2794503742 |
| 最大 $|C-C^\mathsf{T}|$ | 0.0252238989 |
| $\|C-C^\mathsf{T}\|_F/\|C\|_F$ | 0.2542994496 |
| 对应方向元素 Pearson 相关 | 0.9640250983 |
| 零入/出强度 ROI | Left-Accumbens-area、Right-Pallidum |

最终带标签的 83×83 数值矩阵 SHA-256 为：

`9745282a493e3dc58448aefbdeb2323e645c38d335cf22581a01e6402967678b`

## 6. 真实数据与模拟数据的边界

进入 DMF 后，唯一来自 F-TRACT 的对象是固定矩阵 $\mathbf{C}$ 及其 ROI 标签。每个 ROI 的兴奋性/抑制性门控状态、发放率和所有未来状态均由模型生成。

因此，以下说法是准确的：

- “使用经验人脑 CCEP 覆盖计数派生的网络拓扑驱动 DMF”；
- “在经验代理耦合网络上研究模型内临界动力学和协同”。

以下说法不准确：

- “在 780 名患者的时序上直接计算 $\Phi^{EID}$”；
- “在 HCP BOLD 数据上得到图 A–G”；
- “使用扩散 MRI 纤维连接计算图中的信息指标”；
- “图中的 2048 个样本对应 2048 名受试者或 2048 个实测时间点”。

## 7. DMF 动力学生成过程

### 7.1 状态和方程

每个 ROI $j$ 包含兴奋性 NMDA 门控变量 $S_j^{(E)}$ 和抑制性 GABA 门控变量 $S_j^{(I)}$。长程耦合只作用于兴奋性输入：

$$
I_j^{(E)}=W_EI_0+w_+J_{\mathrm{NMDA}}S_j^{(E)}
+GJ_{\mathrm{NMDA}}\sum_k C_{jk}S_k^{(E)}
-J_j^{\mathrm{FIC}}S_j^{(I)},
$$

$$
I_j^{(I)}=W_II_0+J_{\mathrm{NMDA}}S_j^{(E)}-S_j^{(I)}.
$$

电流经平滑 transfer function 转为发放率：

$$
r_j^{(q)}=
\frac{g_q\left(I_j^{(q)}-I_{\mathrm{thr}}^{(q)}\right)}
{1-\exp\left[-d_qg_q\left(I_j^{(q)}-I_{\mathrm{thr}}^{(q)}\right)\right]},
\qquad q\in\{E,I\}.
$$

门控变量按 Euler–Maruyama 更新：

$$
\mathrm{d}S_j^{(E)}=
\left[-\frac{S_j^{(E)}}{\tau_E}
+(1-S_j^{(E)})\gamma_Er_j^{(E)}\right]\mathrm{d}t
+\sigma\,\mathrm{d}W_j^{(E)},
$$

$$
\mathrm{d}S_j^{(I)}=
\left[-\frac{S_j^{(I)}}{\tau_I}+r_j^{(I)}\right]\mathrm{d}t
+\sigma\,\mathrm{d}W_j^{(I)}.
$$

### 7.2 固定模型参数

| 参数 | 值 |
|---|---:|
| $W_E$ | 1.0 |
| $W_I$ | 0.7 |
| $I_0$ | 0.382 |
| $w_+$ | 1.4 |
| $J_{\mathrm{NMDA}}$ | 0.15 |
| $g_E$ | 310 |
| $I_{\mathrm{thr}}^{(E)}$ | 0.403 |
| $d_E$ | 0.16 |
| $g_I$ | 615 |
| $I_{\mathrm{thr}}^{(I)}$ | 0.288 |
| $d_I$ | 0.087 |
| $\tau_E$ | 0.100 s |
| $\tau_I$ | 0.010 s |
| $\gamma_E$ | 0.641 |
| 动力学噪声 $\sigma$ | 0.01 |

模型来源可参考 [Deco et al. (2014)](https://doi.org/10.1523/JNEUROSCI.5068-13.2014) 和 [Deco et al. (2018)](https://doi.org/10.1016/j.cub.2018.07.083)。当前图所对照的 $\Phi$ID 研究见 [Mediano et al. (2025)](https://doi.org/10.1073/pnas.2423297122)。

### 7.3 JFIC 校准

每个 ROI 有一个 feedback inhibitory control 参数 $J_j^{\mathrm{FIC}}$。它在 $G=1.0$ 处校准一次，随后整条 $G$ 扫描复用同一 83 维向量，避免在每个 $G$ 重新拟合而混入额外处理因素。

校准设置：

- 目标区域平均兴奋性发放率：3.0 Hz；
- 允许最大区域误差：0.05 Hz；
- 初值：1.0；
- 更新步长：0.025；
- 合法范围：0.1–10.0；
- 最大迭代：12；
- 校准噪声：0；
- 当前结果在 7 次迭代收敛；
- 最大绝对区域误差：0.046379 Hz；
- 最终 JFIC 值范围：1.019464–1.468028。

### 7.4 自发发放率扫描

汇总图 A、C 的黑色曲线来自自发 DMF 轨迹：

- $G=1.0,1.1,\ldots,3.0$，共 21 点；
- 每点模拟 1.5 s；
- 积分步长 $10^{-4}$ s；
- 最短 burn-in 0.3 s；
- 初始 $S_E=S_I=0.001$；
- $G$ 点按升序 continuation，上一个 $G$ 的终态作为下一个 $G$ 的初态；
- 模拟状态每步裁剪到 $[0,1]$；
- 每个 $G$ 使用不同但确定的随机种子；
- 统计从检测到稳定的窗口开始，否则从最短 burn-in 开始；
- 平均发放率是所有 ROI 的兴奋性发放率在统计时间段和脑区上的均值。

稳定检测将轨迹分成 0.05 s 窗口；相邻窗口均值漂移不超过 0.05 Hz，且连续满足 3 次时判为稳定。当前 21 个点中 19 个检测到稳定；$G=1.7$ 和 $1.8$ 未检测到稳定时使用 0.2 s 起始点，$G=1.9$ 使用 0.45 s 起始点。

黑色曲线的 21 个均值从 $G=1.0$ 的 2.9971 Hz 单调上升到 $G=3.0$ 的 20.4393 Hz。离散最大斜率位于 $G=1.8$。图中 $G=1.6$–1.8 灰带是后续分析采用的临界平台，并不表示某个精确的人脑生理耦合常数。

对应自发扫描数据 SHA-256：

`a529e5014f7bad53d939cb8fb03b6edd42db09174acc671a18c7243f9d9260d0`

## 8. 最大熵干预数据：A、B、D、E、F、G 面板的共同底座

### 8.1 条件设计

| 项目 | 设定 |
|---|---|
| ROI 数 | 83 |
| source 维数 | 166 |
| target 维数 | 166 |
| source 状态 | 完整 $(\mathbf{s}_E,\mathbf{s}_I)$ |
| target 状态 | 300 步后的完整 $(\mathbf{s}_E,\mathbf{s}_I)$ |
| 每维干预分布 | 相互独立的 $U(0.30,0.70)$ |
| 每个 seed–$G$ 样本数 | 2,048 |
| seeds | 3–10，共 8 个 |
| 主扫描 $G$ | 1.0、1.3、1.4、1.5、1.6、1.7、1.8、1.9、2.2、3.0 |
| 层级分解 $G$ | 1.6、1.7、1.8 |
| 积分步长 | 0.001 s |
| horizon | 300 步，即 0.3 s |
| 动力学噪声 | 0.01 |
| 长程耦合 | direct excitatory coupling |
| 状态边界 | 不裁剪 |
| 估计器 ridge | $10^{-6}$ |

这里的“最大熵”是指：在预先声明的每维支持 $[0.30,0.70]$ 上，均匀分布具有最大微分熵。它不是整个物理区间 $[0,1]^{166}$ 上的绝对最大熵分布。较宽的 $U(0,1)^{166}$ 在先前对照中没有得到同样的临界峰，因此不能把本结论无条件外推到整个物理状态空间。

### 8.2 source 和 target 的实际列顺序

数组实际按以下顺序拼接：

$$
\mathbf{s}=
(S_1^{(E)},\ldots,S_{83}^{(E)},S_1^{(I)},\ldots,S_{83}^{(I)})^\mathsf{T}.
$$

ROI $r$ 的二元 source block 因而是列索引 $(r,r+83)$，不是在数组中相邻的 $(2r,2r+1)$。target 使用同样的顺序。

### 8.3 随机数日程

对 seed $s$ 和原 21 点 $G$ 网格索引 $g$：

- source 随机数种子：$100000s+1000g$；
- target 动力学噪声种子：$100000s+1000g+17$。

同一 seed–$G$ 条件的 source 和 target 可完全复算；不同条件使用不同随机流。Yeo-7 和结构模块分解严格复用主确认的 source、target、噪声和 JFIC，不重新抽取另一套样本。

### 8.4 标准化

source 和 target 分别按列标准化：

$$
z_{ni}=\frac{x_{ni}-\bar{x}_i}{s_i},
$$

其中 $s_i$ 使用样本标准差（`ddof=1`）。如果某列标准差不超过 $10^{-12}$，缩放因子替换为 1。source 的均值/标准差不会用于缩放 target，反之亦然。

### 8.5 Gaussian EI 估计

标准化后用最小二乘拟合一步条件映射：

$$
\mathbf{T}=\mathbf{A}\mathbf{S}+\boldsymbol{\varepsilon}.
$$

由于数据已中心化，拟合不再显式加入截距。最大熵干预口径下，把经验 source 协方差强制因子化为对角矩阵，再加 $10^{-6}\mathbf{I}$；残差协方差和构造出的 target 协方差也加入 ridge。对半正定矩阵计算对数行列式时，特征值下限截为 $10^{-12}$。

联合 EI 为

$$
EI(\mathbf{S};\mathbf{T})
=\frac{1}{2\ln2}
\left[
\log|\boldsymbol{\Sigma}_S|
-\log|\boldsymbol{\Sigma}_{S\mid T}|
\right].
$$

每个标量 source 的 EI 用对应 1×1 先验和条件协方差计算。系统量定义为

$$
\Phi^{EID}
=EI(\mathbf{S};\mathbf{T})
-\sum_{i=1}^{166}EI(S_i;\mathbf{T}).
$$

在独立 source 的 Gaussian 构造下，该量等于 source 在给定完整 target 后的 conditional total correlation，因此非负。

本实验未采用 transport map。原因是 166 维 source、166 维 target、每条件 2048 样本和多 seed–$G$ 条件下，高维 transport-map 估计的计算成本过高；Gaussian 估计器可使系统量、ROI 层级恒等式和 WMS 对照保持统一。代价是结果只刻画线性条件均值和二阶协方差结构，对强非高斯或高阶非线性依赖不敏感。

### 8.6 主确认结果的机器可读内容

主确认缓存应至少包含：

- `G`、原网格索引和 seeds；
- source/target 状态类型；
- 干预上下界；
- horizon、样本数和状态边界模式；
- whole EI、166 个 singleton EI 之和及 $\Phi^{EID}$；
- target variance retained、spatial SD、平均非对角相关；
- target entropy、联合条件熵和 singleton 条件熵之和。

对应主确认数据 SHA-256：

`f2416d59c358c30582c5c10bbe91692101af869eeea3e1f9edf1783b3c6a099c`

## 9. 自然稳态 observational WMS：C 面板

### 9.1 “observational” 的准确含义

C 面板的 observational WMS 不是患者或 HCP 的实测时序。它来自同一 DMF 在每个 $G$ 下的自然稳态轨迹，因此“observational”表示沿模型自然访问的状态分布取样，而不是从人为独立均匀干预分布取样。

### 9.2 与最大熵干预保持一致的项目

C 面板与 A/B 的干预实验共同固定：

- 同一 83 ROI 代理耦合矩阵；
- 同一 JFIC；
- 完整 166 维 E/I source；
- 完整 166 维未来 target；
- 300 步 horizon；
- 2048 个样本/条件；
- seeds 3–10；
- direct coupling；
- $dt=0.001$、$\sigma=0.01$；
- 无状态裁剪；
- 相同 target 噪声种子偏移 `+17`；
- 分列标准化；
- 线性 Gaussian 估计和 ridge $10^{-6}$。

唯一有意改变的处理因素是 source 分布：

- 干预条件：各维独立 $U(0.3,0.7)$，source 协方差因子化；
- observational 条件：自然稳态 DMF source，保留完整经验 source 协方差。

### 9.3 自然状态的抽样

每个 seed–$G$ 先生成一条 1.5 s 的自然 DMF 轨迹，最短 burn-in 为 0.3 s；稳定检测窗口 0.05 s、漂移阈值 0.15 Hz、连续确认 2 个窗口。从稳定段有放回抽取 2048 个索引，取得对应的 83 维 $S_E$ 和 83 维 $S_I$。

抽样前先消费与干预条件相同数量的均匀随机数，使后续随机数日程对齐；然后用同一 seed 体系抽取自然状态索引。由于是有放回抽样，每个条件的唯一自然状态数为 395–978，而不是 2048。

观测 WMS 使用同一 common-target whole-minus-sum 形式：

$$
\Phi^{WMS}
=I_{p_{\mathrm{obs}}}(\mathbf{S};\mathbf{T})
-\sum_{i=1}^{166}I_{p_{\mathrm{obs}}}(S_i;\mathbf{T}).
$$

它保留 source 之间的自然相关，因此允许为负。全部 168 个 seed–$G$ 条件均为负；$G=1.6$ 的 seed 均值最深，为 $-328.118\pm3.940$ bits。自然 source 协方差条件数中位数为 $5.37\times10^5$，所以 WMS 的绝对量级对 ridge 较敏感，应优先解释曲线形状和跨 seed 一致性。

C 面板数据 SHA-256：

`ad06b009776ca937beef3ca0a22e216267dad875fd82617d46ac589940f54c93`

## 10. 临界窗层级分解：D、E、F、G 面板

### 10.1 条件集合

层级分解只使用 $G\in\{1.6,1.7,1.8\}$ 和 seeds 3–10，共 $3\times8=24$ 个条件。每个条件仍有 2048 个 source–target 样本。

所有分解都从同一个 166×166 条件 source 协方差 $\boldsymbol{\Sigma}_{S\mid T}$ 出发。复算 $\Phi^{EID}$ 与主确认的最大绝对误差为 $1.79\times10^{-12}$ bits；层级加法恒等式误差不超过 $1.95\times10^{-13}$ bits。

### 10.2 ROI 内与跨 ROI 分解

每个 ROI 的 $E/I$ 二元 source block 为

$$
\mathbf{s}_r=(S_r^{(E)},S_r^{(I)})^\mathsf{T}.
$$

ROI 内项为

$$
\phi_{r,\mathrm{within}}
=I(S_r^{(E)};S_r^{(I)}\mid\mathbf{T}).
$$

跨 ROI 总量为把 83 个二元 ROI block 作为分块后得到的条件 total correlation。系统总量满足

$$
\Phi^{EID}_{166}
=\sum_{r=1}^{83}\phi_{r,\mathrm{within}}
+\Phi^{EID}_{\mathrm{between\ ROI}}.
$$

单个 ROI 的跨区 leverage 为删除该 ROI block 后，between-ROI conditional total correlation 的下降量。它等于该 ROI block 与其余 source blocks 在给定 target 后的条件互信息。各 ROI leverage 可重叠，**不能跨 ROI 求和并解释为互斥贡献**。

### 10.3 D 面板的结构强度

D 面板为避免有向入/出强度口径冲突，先对代理矩阵对称化：

$$
\mathbf{C}_{\mathrm{sym}}
=\frac{\mathbf{C}+\mathbf{C}^{\mathsf{T}}}{2},
$$

再把对角线设为 0，并以行和作为加权结构强度。该强度范围为 0–0.642068。横轴因此不保留刺激→记录的方向。

每个 ROI 的纵轴数值先在 24 个条件上取均值：

- 左轴：ROI 内 E/I 条件耦合；
- 右轴：ROI 与其余全脑的跨 ROI 条件耦合。

相关统计使用 83 个 ROI 的 Spearman 双侧检验。ROI 内项与结构强度 $\rho=-0.7534$；跨 ROI leverage 与结构强度 $\rho=0.9885$。这些是同一固定代理网络上的关联，不是跨受试者统计，也不构成结构强度的因果效应。

### 10.4 E 面板的比例

对每个 seed–$G$ 条件分别计算：

$$
f_{\mathrm{within}}=
\frac{\Phi_{\mathrm{within\ ROI}}}{\Phi_{166}},
\qquad
f_{\mathrm{cross}}=1-f_{\mathrm{within}}.
$$

然后对 24 个条件取均值。结果为：

- ROI 内：3.8652 bits，占 31.33%；
- 跨 ROI：8.4720 bits，占 68.67%；
- 24/24 个条件均为跨 ROI 大于 ROI 内。

误差线按 24 个条件的标准误计算。由于三个 $G$ 共享 seed，这 24 个值并非完全独立重复；误差线用于描述 seed–$G$ 条件波动，不等同于独立受试者层面的置信区间。

### 10.5 F 面板的 Yeo-7 分组

68 个皮层 ROI 依据 fsaverage5 表面上 Desikan–Killiany parcel 与 Yeo2011 七网络注释的最大顶点重叠分组。当前实验直接使用固定映射表，不在每次运行时重新下载 Yeo 注释或重算 overlap。15 个非皮层 ROI 全部归入独立的 Non-cortical 组。

| 功能组 | ROI 数 | 组内跨 ROI $\Phi^{EID}$ 均值（bits） |
|---|---:|---:|
| Visual | 11 | 0.121579 |
| Somatomotor | 11 | 0.385344 |
| Dorsal attention | 2 | 0.001235 |
| Salience / ventral attention | 11 | 0.122806 |
| Limbic | 12 | 0.091482 |
| Frontoparietal control | 3 | 0.004740 |
| Default mode | 18 | 0.388715 |
| Non-cortical | 15 | 0.130272 |

同一功能组内部的跨 ROI 总量为 1.2462 bits；不同功能组之间为 7.2258 bits。F 面板显示的是每组**总量**，不按 ROI 数、ROI 对数或网络体积归一化，不能据此直接比较“单位 ROI 效应”。

Yeo 层级数据 SHA-256：

`6cd08ff095003e9ba2c1726fbc346d5e497fe66111f617db1f373c875677a312`

### 10.6 G 面板的皮层表面投影

G 面板使用 D 面板同一个跨 ROI leverage，在 24 个条件上取 ROI 均值。投影步骤为：

1. 按 `ctx-lh-` 和 `ctx-rh-` 分离左右半球，各得到 34 个皮层 ROI；
2. 读取 TemplateFlow Desikan2006 10k GIFTI 注释；
3. 用注释 label table 的 parcel 名称与 ROI 名称匹配；
4. 把每个 ROI 数值复制到其所有表面顶点；
5. 背景和 medial-wall 顶点设为缺失，不着指标色；
6. 在 fsaverage5 inflated mesh 上绘制左右半球外侧和内侧视图；
7. 使用所有已投影皮层值的实际最小值和最大值建立共同色标。

15 个非皮层 ROI 仍参与系统 EI、层级分解、E/F 面板和 D 面板统计，但不会被投影到皮层表面。

## 11. 汇总图逐面板数据血缘

| 面板 | 输入 | 处理 | 输出含义 |
|---|---|---|---|
| A 黑线 | 自发 DMF 21 点 $G$ 扫描 | 每点稳定段平均 | 全脑平均兴奋性发放率 |
| A 紫线 | 8 seeds × 10 个 $G$ × 2048 最大熵干预 | Gaussian whole-minus-singletons | 全系统 $\Phi^{EID}$ |
| B 蓝线 | 同 A 紫线 | 联合 source 到完整 target 的 Gaussian EI | whole EI |
| B 橙线 | 同 A 紫线 | 166 个标量 E/I source EI 求和 | singleton EI sum；不是 83 个 ROI-block EI 之和 |
| C 黑线 | 与 A 相同的自发发放率扫描 | 同 A 黑线 | 动力学参照 |
| C 绿线 | 8 seeds × 21 个 $G$ × 2048 自然稳态 source | 保留完整 source 协方差的 Gaussian WMS | observational $\Phi^{WMS}$ |
| D | 临界窗条件协方差 + 对称化代理耦合矩阵 | 24 条件均值；ROI-level Spearman | 结构强度与局部/跨区条件耦合关系 |
| E | 临界窗 ROI 分解 | 每条件比例后取 24 条件均值和 SEM | ROI 内与跨 ROI 比例 |
| F | 临界窗条件协方差 + 固定 Yeo/Non-cortical 映射 | 各组内跨 ROI 项取 24 条件均值 | 功能组总贡献 |
| G | D 的跨 ROI leverage + Desikan/fsaverage5 | ROI 值扩展到表面顶点 | 68 个皮层 ROI 的空间分布 |

## 12. ROI 与 Yeo/Non-cortical 分组清单

| ROI | 分组 | ROI | 分组 |
|---|---|---|---|
| ctx-lh-cuneus | Visual | ctx-rh-cuneus | Visual |
| ctx-lh-lateraloccipital | Visual | ctx-rh-lateraloccipital | Visual |
| ctx-lh-lingual | Visual | ctx-rh-lingual | Visual |
| ctx-lh-pericalcarine | Visual | ctx-rh-pericalcarine | Visual |
| ctx-lh-bankssts | Default mode | ctx-rh-bankssts | Somatomotor |
| ctx-lh-entorhinal | Limbic | ctx-rh-entorhinal | Limbic |
| ctx-lh-fusiform | Visual | ctx-rh-fusiform | Visual |
| ctx-lh-inferiortemporal | Limbic | ctx-rh-inferiortemporal | Limbic |
| ctx-lh-middletemporal | Default mode | ctx-rh-middletemporal | Default mode |
| ctx-lh-parahippocampal | Default mode | ctx-rh-parahippocampal | Visual |
| ctx-lh-superiortemporal | Somatomotor | ctx-rh-superiortemporal | Somatomotor |
| ctx-lh-temporalpole | Limbic | ctx-rh-temporalpole | Limbic |
| ctx-lh-transversetemporal | Somatomotor | ctx-rh-transversetemporal | Somatomotor |
| ctx-lh-inferiorparietal | Default mode | ctx-rh-inferiorparietal | Default mode |
| ctx-lh-postcentral | Somatomotor | ctx-rh-postcentral | Somatomotor |
| ctx-lh-precuneus | Default mode | ctx-rh-precuneus | Default mode |
| ctx-lh-superiorparietal | Dorsal attention | ctx-rh-superiorparietal | Dorsal attention |
| ctx-lh-supramarginal | Salience / ventral attention | ctx-rh-supramarginal | Salience / ventral attention |
| ctx-lh-caudalanteriorcingulate | Salience / ventral attention | ctx-rh-caudalanteriorcingulate | Salience / ventral attention |
| ctx-lh-isthmuscingulate | Default mode | ctx-rh-isthmuscingulate | Default mode |
| ctx-lh-posteriorcingulate | Salience / ventral attention | ctx-rh-posteriorcingulate | Salience / ventral attention |
| ctx-lh-rostralanteriorcingulate | Default mode | ctx-rh-rostralanteriorcingulate | Default mode |
| ctx-lh-paracentral | Somatomotor | ctx-rh-paracentral | Somatomotor |
| ctx-lh-caudalmiddlefrontal | Default mode | ctx-rh-caudalmiddlefrontal | Frontoparietal control |
| ctx-lh-frontalpole | Limbic | ctx-rh-frontalpole | Limbic |
| ctx-lh-parsopercularis | Salience / ventral attention | ctx-rh-parsopercularis | Salience / ventral attention |
| ctx-lh-parstriangularis | Default mode | ctx-rh-parstriangularis | Salience / ventral attention |
| ctx-lh-precentral | Somatomotor | ctx-rh-precentral | Somatomotor |
| ctx-lh-rostralmiddlefrontal | Frontoparietal control | ctx-rh-rostralmiddlefrontal | Frontoparietal control |
| ctx-lh-superiorfrontal | Default mode | ctx-rh-superiorfrontal | Default mode |
| ctx-lh-lateralorbitofrontal | Limbic | ctx-rh-lateralorbitofrontal | Limbic |
| ctx-lh-medialorbitofrontal | Limbic | ctx-rh-medialorbitofrontal | Limbic |
| ctx-lh-parsorbitalis | Default mode | ctx-rh-parsorbitalis | Default mode |
| ctx-lh-insula | Salience / ventral attention | ctx-rh-insula | Salience / ventral attention |
| Left-Thalamus-Proper | Non-cortical | Right-Thalamus-Proper | Non-cortical |
| Left-Pallidum | Non-cortical | Right-Pallidum | Non-cortical |
| Left-Putamen | Non-cortical | Right-Putamen | Non-cortical |
| Left-Hippocampus | Non-cortical | Right-Hippocampus | Non-cortical |
| Left-Caudate | Non-cortical | Right-Caudate | Non-cortical |
| Left-Accumbens-area | Non-cortical | Right-Accumbens-area | Non-cortical |
| Left-Amygdala | Non-cortical | Right-Amygdala | Non-cortical |
| Brain-Stem | Non-cortical | — | — |

## 13. 数据质量控制与复现验收

建议重跑时至少检查以下项目：

1. F-TRACT 完整发布包 MD5 与官方记录一致。
2. `Lausanne2008-33.zip` SHA-256 与本文一致。
3. 标签数为 84，且唯一存在一个 `Unknown`。
4. pickle 展平条目数为 19；第 0 个条目的 `mean` 与 probability CSV 逐元素一致。
5. 第 0 个 `count` 为 84×84、有限、非负；删除 `Unknown` 后为 83×83。
6. 缩放分母为 8,151；清零对角后最大权重为 0.1062937063。
7. 最终非零有向边数为 3,477；Left-Accumbens-area 和 Right-Pallidum 为零强度节点。
8. 自发扫描有 21 个 $G$ 点，JFIC 在 $G=1.0$ 校准一次并整段复用。
9. 最大熵主确认有 8×10=80 个完整条件，无缺失值。
10. observational WMS 有 8×21=168 个完整条件，无缺失值。
11. Yeo 层级分解有 8×3=24 个条件，ROI 数为 83，条件协方差为 166×166。
12. 系统 $\Phi^{EID}$ 复算误差不超过 $10^{-7}$ bits；当前实际误差约 $10^{-12}$ bits。
13. 层级分解加法误差不超过 $10^{-7}$ bits；当前实际误差约 $10^{-13}$ bits。
14. 24/24 个条件均满足跨 ROI 大于 ROI 内。
15. 24/24 个条件均满足功能组间大于功能组内跨 ROI。
16. 表面投影时左右半球各准确匹配 34 个 Desikan parcel；任何缺失标签都应报错。

## 14. 主要限制和不得越界的解释

1. **代理矩阵不是原论文 HCP SC。**原论文使用 HCP 900 release 扩散 MRI 经 Lausanne-83 分区得到的结构连接；当前数据是 F-TRACT CCEP probability 条目的 count 支持矩阵。
2. **count 不等于连接强度。**count 很大可能表示该刺激—记录组合有更多观测支持，也可能受电极覆盖和临床取样不均衡影响；它不能直接解释为更强的生理连接概率或更多纤维。
3. **方向被反向解释。**F-TRACT 行是 stimulated、列是 recorded；当前 DMF 未转置矩阵。未来应做原矩阵与转置矩阵的成对方向敏感性分析。
4. **缩放受对角线主导。**8,151 的缩放最大值随后被清零，导致实际最大跨区权重只有 0.1063。若先清零再缩放，会得到不同的耦合尺度和临界区。
5. **有两个零强度 ROI。**它们仍保留在 83 ROI 系统和 Non-cortical 分组中，但没有代理长程边。
6. **没有个体层面重复。**F-TRACT 是群体汇总图谱；8 个 seeds 是模拟随机种子，不是 8 名受试者。
7. **Yeo 映射是粗粒度多数归属。**单个 Desikan parcel 内可能跨越多个 Yeo 网络；固定为一个网络会丢失空间异质性。
8. **网络总量受组大小影响。**F 面板未做规模归一化，不能比较单位 ROI 或单位 ROI-pair 效应。
9. **Gaussian 估计只保留二阶结构。**它不能捕捉全部非线性、非高斯或多模态依赖；ridge 也会影响高维绝对量。
10. **自然 WMS 仍是模型观测分布。**它不能替代真实脑时序的 observational 分析。
11. **临界窗是模型内操作性定义。**$G=1.6$–1.8 不对应人体固定生理常数，Kuramoto $K_c=1.5958$ 只提供形状参照。
12. **ROI leverage 不是可加 atom。**跨 ROI leverage 会重叠，不应按 ROI 相加或解释为互斥因果贡献。

## 15. 许可、引用与数据共享注意事项

Zenodo Rights 字段将 F-TRACT 21-12 标记为 **CC BY 4.0**，但数据集描述同时写有“free use for research use only”。这两处文字存在范围差异。研究性使用应完整引用数据集及推荐论文；如需商业使用、重新分发拆分后的子归档或公开派生矩阵，建议先向 F-TRACT 数据维护方确认许可边界。

建议至少引用：

1. David, O. et al. F-TRACT atlas, version 21-12. Zenodo. [doi:10.5281/zenodo.7015415](https://doi.org/10.5281/zenodo.7015415).
2. Trebaul, L. et al. Probabilistic functional tractography of the human cortex revisited. *NeuroImage* 181, 414–429 (2018). [doi:10.1016/j.neuroimage.2018.07.039](https://doi.org/10.1016/j.neuroimage.2018.07.039).
3. Lemaréchal, J.-D. et al. A brain atlas of axonal and synaptic delays based on modelling of cortico-cortical evoked potentials. *Brain* 145, 1653–1667 (2022). [doi:10.1093/brain/awab362](https://doi.org/10.1093/brain/awab362).
4. Cammoun, L. et al. Mapping the human connectome at multiple scales with diffusion spectrum MRI. *Journal of Neuroscience Methods* 203, 386–397 (2012). [doi:10.1016/j.jneumeth.2011.09.031](https://doi.org/10.1016/j.jneumeth.2011.09.031).
5. Yeo, B. T. T. et al. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology* 106, 1125–1165 (2011). [doi:10.1152/jn.00338.2011](https://doi.org/10.1152/jn.00338.2011).
6. Mediano, P. A. M. et al. Toward a unified taxonomy of information dynamics via Integrated Information Decomposition. *PNAS* 122, e2423297122 (2025). [doi:10.1073/pnas.2423297122](https://doi.org/10.1073/pnas.2423297122).

## 16. 一句话交付说明

本实验以 F-TRACT 21-12 的 Lausanne2008-33 probability 条目 `count` 观测支持矩阵构造 83 ROI 有向代理耦合网络，在该网络上运行 E/I DMF，分别从独立 $U(0.3,0.7)^{166}$ 干预分布和模型自然稳态分布生成 166 维 source–target 样本，再用统一的线性 Gaussian 信息估计器计算系统 EI、$\Phi^{EID}$、WMS 及 ROI/Yeo 层级分解；因此结果应解释为**经验代理拓扑上的模型内协同分析**，而非 HCP 结构连接、患者脑活动时序或个体级神经机制的直接测量。
