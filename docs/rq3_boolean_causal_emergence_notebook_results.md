# RQ3 Boolean Causal Emergence Notebook Results

## 摘要

本文档对应当前的 Hoel Figure 2 notebook 版本。与旧版不同，现在的 family 不再额外引入 shared noise 参数，而是直接在 Hoel 等（2013）Figure 2 的 micro mechanism 表上做一维采样。

更具体地说，当前 notebook 回答的问题是：

> 当 Hoel Figure 2 的 micro mechanism 从
> $P(1 \mid 00)=P(1 \mid 01)=P(1 \mid 10)=0.30$
> 推广为
> $P(1 \mid 00)=P(1 \mid 01)=P(1 \mid 10)=q_{\mathrm{off}}$
> 时，哪一种 Hoel 风格分解项最稳定地解释单个系统内部
> $EI(Z \to X^+)$
> 的 candidate 排序？

这次 family 统计的样本单位仍然是“系统”，但系统总数从旧版的 `441` 个降到了 `21` 个：每一个点对应一个具体的
$q_{\mathrm{off}} \in \{0, 0.05, \ldots, 1.00\}$，
并在该系统上完整枚举 `147` 个 `2+2` coarse-graining candidate 后得到一组 Spearman 相关系数。

## 1. 研究问题与实验层次

当前 notebook 仍然包含两个层次的分析。

第一层是单系统参考分析。它固定 Hoel Figure 2 toy example 本身，即
$q_{\mathrm{off}} = 0.30$，
在一个系统内部穷举全部 `147` 个 `2+2` 候选粗粒化，观察
$EI(Z \to X^+)$
与三项分解量之间的相关结构。

第二层是 family 统计分析。它把 Hoel 原例中的局部机制表沿着
$q_{\mathrm{off}}$
做一维扫描，在 `21` 个不同系统上重复第一层的完整 candidate 枚举，但不再为每个系统分别画散点图，而是直接汇总每个系统得到的三组相关系数，并展示这些相关系数在系统族上的分布。

因此：

$$
\text{单系统散点图的样本单位} = \text{candidate coarse-graining},
$$

$$
\text{family 箱图的样本单位} = \text{system } q_{\mathrm{off}}.
$$

## 2. 系统构造

### 2.1 Hoel Figure 2 的微观机制

沿用原文记号，微观系统记为
$S_m = \{A,B,C,D\}$，
四个元素都取二值状态。连线结构与 Hoel Figure 2 一致：每个微观节点都由两个输入决定，并实现同一类 noisy AND mechanism。

若某个微观节点在时刻 $t$ 的两个输入在时刻 $t-1$ 的联合状态记为
$ij \in \{00,01,10,11\}$，
则当前 notebook 采用的局部机制表是

$$
P(1 \mid 00) = P(1 \mid 01) = P(1 \mid 10) = q_{\mathrm{off}},
\qquad
P(1 \mid 11) = 1.
$$

等价地，

$$
P(0 \mid 00) = P(0 \mid 01) = P(0 \mid 10) = 1 - q_{\mathrm{off}},
\qquad
P(0 \mid 11) = 0.
$$

因此，$q_{\mathrm{off}}$ 精确刻画的是：当输入不是 `11` 时，微观节点在下一时刻错误地激活为 `1` 的概率。

Hoel Figure 2 的原始 toy example 对应于

$$
q_{\mathrm{off}} = 0.30.
$$

### 2.2 从局部机制表到微观 TPM

和 Hoel 原文一样，完整的微观转移概率矩阵由“对全部微观状态做均匀扰动”得到。也就是说，对
$S_m$
的全部 `16` 个微观状态从 `0000` 到 `1111` 逐一施加干预，并根据上面的局部机制表计算下一时刻的联合分布，从而得到一个 `16 \times 16` 的微观 TPM。

当前 family 并不改变连线结构，也不改变允许的粗粒化类型；它只改变这张局部机制表中的同一个参数
$q_{\mathrm{off}}$。
因此 family 枚举的参数网格是

$$
q_{\mathrm{off}} \in \{0, 0.05, \ldots, 1.00\},
$$

总系统数为

$$
21.
$$

### 2.3 宏观映射与 Hoel 原文的一致性

宏观层仍然采用 Hoel 等（2013）Figure 2 的构造：

- 分组固定在两块 `2+2` 候选上做完整枚举；
- 代表性的 planted coarse-graining 是
  $\{\{A,B\}, \{C,D\}\}$；
- 每个二比特块上的 planted macro mapping 都是
  $[00,01,10] \mapsto \mathrm{off}$，
  $[11] \mapsto \mathrm{on}$，
  即代码里的 `AND/off-on [0001]`。

这正对应 Hoel 原文允许的映射
$M : S_m \to S_M$：
宏观状态不能保留块内微观元素的身份信息，因此
`00/01/10 -> off, 11 -> on`
是允许的，而依赖 `01` 与 `10` 区分的映射则不允许。

## 3. 每个系统内部的 Hoel-style 统计量

对任一固定系统
$s = q_{\mathrm{off}}$，
我们都重新完整枚举全部 `147` 个 `2+2` coarse-graining candidate。设 candidate 集合记为
$\mathcal{C}$，
其中
$|\mathcal{C}| = 147$。

对于每个 candidate
$c \in \mathcal{C}$，
代码都会计算

$$
EI_s(Z \to X^+; c),
\qquad
\mathrm{Syn}_{\mathrm{micro},s}(c),
\qquad
\mathrm{Loss}_{\mathrm{sum},s}(c),
\qquad
\mathrm{Syn}_{\mathrm{macro},s}(c).
$$

这里 `Syn macro` 的计算口径是先构造
$Z \to X^+$
的 TPM，再在同一 TPM 上做块级边缘化：

$$
\mathrm{Syn}_{\mathrm{macro},s}(c)
=
EI_s(Z \to X^+; c)
-\sum_{\ell=1}^{2} EI_s\!\bigl(Z_\ell \to X^+; c\bigr),
$$

其中
$EI_s(Z_\ell \to X^+; c)$
不是直接由微观 TPM 取行平均，而是由
$Z \to X^+$
TPM 按另一个宏变量做均匀边缘化得到。这样三项都处在同一个
`do(Z_1, Z_2)` 干预口径下，和正文定理的非负性条件一致。

为了和 Hoel 单例散点图保持一致，family 统计比较的是
$-\mathrm{Syn}_{\mathrm{micro}}$、
$-\mathrm{Loss}_{\mathrm{sum}}$
和
$\mathrm{Syn}_{\mathrm{macro}}$
这三项。于是每个系统都会产生三组相关系数：

$$
\rho^{(\mathrm{neg\text{-}syn})}_s
=
\rho_{\mathrm{Spearman}}
\!\left(
\{-\mathrm{Syn}_{\mathrm{micro},s}(c)\}_{c \in \mathcal{C}},
\{EI_s(Z \to X^+; c)\}_{c \in \mathcal{C}}
\right),
$$

$$
\rho^{(\mathrm{neg\text{-}loss})}_s
=
\rho_{\mathrm{Spearman}}
\!\left(
\{-\mathrm{Loss}_{\mathrm{sum},s}(c)\}_{c \in \mathcal{C}},
\{EI_s(Z \to X^+; c)\}_{c \in \mathcal{C}}
\right),
$$

$$
\rho^{(\mathrm{macro})}_s
=
\rho_{\mathrm{Spearman}}
\!\left(
\{\mathrm{Syn}_{\mathrm{macro},s}(c)\}_{c \in \mathcal{C}},
\{EI_s(Z \to X^+; c)\}_{c \in \mathcal{C}}
\right).
$$

因此 family 箱图汇总的是

$$
\left\{
\rho^{(\mathrm{neg\text{-}syn})}_s,
\rho^{(\mathrm{neg\text{-}loss})}_s,
\rho^{(\mathrm{macro})}_s
\right\}_{s \in \mathcal{S}},
\qquad
|\mathcal{S}| = 21.
$$

## 4. Hoel 单例参考结果

单系统参考例固定在 Hoel Figure 2 toy example 本身，即

$$
q_{\mathrm{off}} = 0.30.
$$

在这个系统上，完整 candidate 枚举的结果为：

| 指标 | 数值 |
| --- | ---: |
| 微观 $EI(X \to X^+)$ | `1.148579` |
| 最优宏观 $EI(S_M)$ | `1.551829` |
| 最优 $EI(Z \to X^+)$ | `1.551829` |
| 候选 `2+2` 数量 | `147` |
| 最优划分 | `{A,B} | {C,D}` |
| 最优映射 | `AND/off-on [0001]`, `AND/off-on [0001]` |

![Hoel toy example scatter](../results/rq3_boolean_causal_emergence/hoel_fig2_toy_example_scatter.svg)

该参考系统上的三组相关系数为：

| 解释项 | Spearman `rho` |
| --- | ---: |
| `neg. syn` | `-0.077362` |
| `neg. loss` | `0.610869` |
| `Syn macro` | `0.504261` |

这个参考例说明，在 Hoel 基线系统内部，`neg. loss` 已经是最强的单调解释项，而 `Syn macro` 也保持了稳定的中等正相关；`neg. syn` 则接近于零。

## 5. Family-level 主结果

### 5.1 family 统计设计

对于当前 family 里的每一个系统，notebook 都重新执行和 Hoel 单例完全相同的 candidate 搜索与拟合流程。因此，family 图上的每一列对应的是 `21` 个系统级相关系数点，而不是某个固定宏观描述下的切片值。

进一步地，在这 `21` 个系统上，报出的最佳 candidate 始终都恢复了 Hoel 的 planted 描述：

$$
\{\{A,B\},\{C,D\}\}, \qquad
\phi_{AB} = \phi_{CD} = [0,0,0,1].
$$

这说明在当前这一维 family 里，真正发生变化的不是 planted candidate 是否被恢复，而是系统内部 `147` 个 candidate 的排序结构怎样随
$q_{\mathrm{off}}$
改变。

需要说明的是，这个“始终恢复”应理解为当前搜索结果上的最优返回值；像
$q_{\mathrm{off}} = 1$
这样的完全退化情形会导致大量 candidate 并列，此时 `{A,B}|{C,D}` 不应被解释为严格唯一的最优解。

### 5.2 相关系数分布

![Family per-system rho distribution](../results/rq3_boolean_causal_emergence/family_system_rho_distribution.svg)

三组系统级相关系数的分布统计如下：

| 解释项 | mean | std | min | Q1 | median | Q3 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neg. syn` | `-0.069588` | `0.020094` | `-0.089631` | `-0.077362` | `-0.077362` | `-0.065093` | `0.000000` |
| `neg. loss` | `0.666349` | `0.173219` | `0.000000` | `0.611233` | `0.641278` | `0.794777` | `0.826781` |
| `Syn macro` | `0.465201` | `0.107288` | `0.000000` | `0.463079` | `0.498875` | `0.504261` | `0.518542` |

这里有三个直接结论。

第一，`neg. loss` 仍然是最稳定、幅度也最大的正相关项。即使把 family 收缩到只扫描 Hoel 原始局部机制表中的一个参数，这个结论仍然保留。

第二，`Syn macro` 始终保持显著正相关，但整体弱于 `neg. loss`。这意味着宏观协同确实参与解释 candidate 排序，不过它不是这一组 Hoel-style 系统中的首要排序轴。

第三，`neg. syn` 大多接近于零且略为负值，说明微观协同项对
$EI(Z \to X^+)$
排序的解释力最弱。

另外，按最新口径把
$EI(Z_1 \to X^+)$、
$EI(Z_2 \to X^+)$
统一改为从同一
$Z \to X^+$
TPM 边缘化计算后，`Syn macro` 在所有 candidate 上都保持非负（仅存在浮点误差量级的接近零负值），与定理条件一致。

### 5.3 代表系统

为了让 family 的变化更容易把握，下面列出四个代表系统：

| 角色 | `q_off` | `neg. syn` | `neg. loss` | `Syn macro` | `mean |rho|` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hoel 原例 | `0.30` | `-0.077362` | `0.610869` | `0.504261` | `0.397498` |
| `neg. loss` 最高 | `0.80` | `-0.065093` | `0.826781` | `0.431132` | `0.441002` |
| `Syn macro` 最高 | `0.50` | `-0.077362` | `0.679355` | `0.518542` | `0.425086` |
| 完全退化端点 | `1.00` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |

这些系统共同说明：随着
$q_{\mathrm{off}}$
增大，Hoel 原例的局部 noisy AND 逐渐失去选择性，相关结构会发生幅度变化；但在绝大多数非退化区间里，`neg. loss` 仍然是最稳定的解释项，而 planted `2+2 + AND/off-on` 描述仍会被恢复。

## 6. 解释与结论

与旧版 shared-noise family 相比，当前版本的优点是解释路径更直接：

1. 它完全沿着 Hoel Figure 2 的原始故事推进，只改局部机制表中的一个概率参数。
2. 它避免把“共享噪声耦合”这一层额外设计混进系统构造，因此更容易把结果解释成“Hoel 原例对局部噪声强度变化的鲁棒性”。
3. 它更适合回答你真正关心的问题：在 Hoel 这张 micro mechanism 表附近做扰动时，`AB | CD` 与 `AND/off-on` 何时还能作为最优宏观描述出现。

因此，当前 notebook 最直接支持的结论是：

> 在 Hoel Figure 2 的一维 micro-mechanism family 中，只要系统没有退化到完全失效，Hoel 的 planted `2+2 + AND/off-on` 描述就会稳定地被恢复；同时，在系统内部 candidate 排序的解释上，`neg. loss` 依然比 `Syn macro` 和 `neg. syn` 更稳定。

如果后续要继续扩展实验，一个自然的下一步不是重新引入共享噪声参数，而是直接研究：

- 当局部机制表不再满足
  $P(1 \mid 00)=P(1 \mid 01)=P(1 \mid 10)$
  时，Hoel 的 planted 宏观描述在哪些方向上开始失效；
- 当
  $P(1 \mid 11)$
  也不再固定为 `1` 时，最优 coarse-graining 是否还会稳定落在 `{A,B}|{C,D}`。
