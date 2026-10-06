# 从二元 EI 分解到 N→N：单端可提取共信息与固定 KL 残余树

整理更新：2026-10-06。本文从 $2\to1$ 与 $1\to2$ 出发，直接采用已有的单端可提取共信息定义冗余，再由信息收支确定特有与协同。先给出二元定义、证明和例子，随后说明固定父项如何延拓到 $2\to2$，并用二分递归构造 $N\to M$ 非负分解；$N=M$ 即 $N\to N$。

理论范围为有限离散变量，单位为 bit，对数以 2 为底。二元定义适用于任意给定的联合分布，包括自然观测分布；所有量均在同一分布下计算。采用干预分布时，相应互信息具有 EI 的解释。二元冗余直接采用已有单端可提取共信息及其与内禀条件互信息的关系。[25](#ref-25) 高阶部分复用独立源的 SPT，并增加固定 KL 参照与条件读出，以保持同一父项的非负收支；贪婪搜索负责选择分解路径。当前 PEID 稿件依据、版本歧义和实现接口见第 6.4 节。

## 1. 2→1 与 1→2：定义、收支及完整证明

### 1.1 用单端可提取共信息定义冗余

在任意给定的联合分布下，设 $U,V,W$ 为三个有限离散变量，允许各自有多个状态。所有信息量省略分布下标。全文二元部分沿用这三个字母：$2\to1$ 写为 $(U,V)\to W$，$1\to2$ 写为 $W\to(U,V)$。$U,V$ 表示需要分配共有与特有信息的两变量，$W$ 是指定的局部处理端。共信息定义为

$$
\operatorname{CoI}(U,V,W)
:=I(U;W)+I(V;W)-I(U,V;W)
=I(U;V)-I(U;V\mid W).
\tag{1-1}
$$

共信息对三个变量完全对称，符号为“冗余减协同”。它可以为负；共有与协同还可能互相抵消，因此直接取共信息或其正部不足以识别全部共有部分。第 2.2 节给出具体例子。

只处理指定端 $W$，定义单端可提取共信息

$$
C_{\mathrm{ext}}(U,V;W)
:=\sup_{Z\leftarrow W}\operatorname{CoI}(U,V,Z)
=I(U;V)-\inf_{Z\leftarrow W}I(U;V\mid Z).
\tag{1-2}
$$

$Z\leftarrow W$ 表示局部随机通道，满足 Markov 链 $(U,V)-W-Z$，可行域包含常量、恒等处理。右侧下确界是内禀条件互信息 $I(U;V\downarrow W)$，此构件及上述关系已有来源。[25](#ref-25) 辅助状态数由第 1.9 节给出有限充分界。

本文直接用它作为二元冗余，记

$$
\boxed{R(U,V;W):=C_{\mathrm{ext}}(U,V;W).}
\tag{1-3}
$$

分号标明固定处理端。交换 $U,V$ 保持冗余不变，$W$ 的角色固定。对 $2\to1$ 处理单目标；对 $1\to2$ 处理单源。局部通道用于在给定联合分布内提取信息；因果解释由该分布的生成或干预方式决定。

### 1.2 2→1：两个源的冗余、特有与协同

对 $(U,V)\to W$，在给定联合分布下直接定义。特有信息记为 $\mathrm{Un}$，以区别于变量 $U$；$R,S$ 分别表示冗余与协同。

$$
\boxed{
\begin{aligned}
R&:=C_{\mathrm{ext}}(U,V;W),\\
\mathrm{Un}_{U}&:=I(U;W)-R,\qquad \mathrm{Un}_{V}:=I(V;W)-R,\\
S&:=I(U,V;W)-I(U;W)-I(V;W)+R.
\end{aligned}}
\tag{1-4}
$$

对目标 $W$ 作局部处理，提取两个源共有的 $R$；从每个单源互信息扣除共有部分得到特有，联合互信息的余额为协同。收支为

$$
\begin{aligned}
I(U;W)&=R+\mathrm{Un}_{U},\qquad I(V;W)=R+\mathrm{Un}_{V},\\
I(U,V;W)&=R+\mathrm{Un}_{U}+\mathrm{Un}_{V}+S.
\end{aligned}
\tag{1-5}
$$

这里不要求源独立，也不要求均匀分布。若另有 $U\perp V$，第 1.6 节将证明 $R=0$；当前二源 PEID 的因子化干预是这一独立情形的实例，可直接令冗余为零。

### 1.3 1→2：两个输出的冗余、特有与协同

对 $W\to(U,V)$，同样在给定联合分布下直接定义

$$
\boxed{
\begin{aligned}
R&:=C_{\mathrm{ext}}(U,V;W),\\
\mathrm{Un}_{U}&:=I(W;U)-R,\qquad \mathrm{Un}_{V}:=I(W;V)-R,\\
S&:=I(W;U,V)-I(W;U)-I(W;V)+R.
\end{aligned}}
\tag{1-6}
$$

对源 $W$ 作局部处理，提取两个输出共有的 $R$；特有是各输出扣除共有后的余额，协同是联合读取的剩余。对应收支为

$$
\begin{aligned}
I(W;U)&=R+\mathrm{Un}_{U},\qquad I(W;V)=R+\mathrm{Un}_{V},\\
I(W;U,V)&=R+\mathrm{Un}_{U}+\mathrm{Un}_{V}+S.
\end{aligned}
\tag{1-7}
$$

由互信息的对称性，式（1-6）与式（1-4）是同一组公式：两个方向共用 $R,\mathrm{Un}_U,\mathrm{Un}_V,S$，无需增加方向下标。交换两个源或两个输出只交换特有标签。两个输出可以相关，源分布也可以非均匀；分配使用实际给定的联合结构。

**表 1　二元性质概览。** 下表均采用 $R(U,V;W)=C_{\mathrm{ext}}(U,V;W)$，只处理分号后的 $W$。完整证明按右栏顺序给出。

| 性质 | 形式化表达与适用条件 | 证明 |
|---|---|---|
| 两变量交换对称 | $R(U,V;W)=R(V,U;W)$；处理端 $W$ 固定 | 第 1.4 节 |
| 四项非负与闭合 | $R,\mathrm{Un}_U,\mathrm{Un}_V,S\ge0$；$I(U,V;W)=R+\mathrm{Un}_U+\mathrm{Un}_V+S$ | 第 1.5 节 |
| 独立源退化为 PEID | $U\perp V\Rightarrow R=0,\ S=I(U;V\mid W)$ | 第 1.6 节 |
| 普通恒等性 | $H(W\mid U,V)=H(U,V\mid W)=0\Rightarrow R=I(U;V)$ | 第 1.7 节 |
| 确定性包含 | $H(U\mid V)=0\Rightarrow R=I(U;W),\ \mathrm{Un}_U=0$ | 第 1.7 节 |
| 单变量自冗余 | $R(U;W):=I(U;W)$，作为相容的定义性扩展 | 第 1.7 节 |
| 处理端数据处理 | $T\leftarrow W\Rightarrow R(U,V;T)\le R(U,V;W)$ | 第 1.7 节 |
| 冗余的附加噪声不变性 | 在任意两位置附加噪声对；噪声对与原 $(U,V,W)$ 联合独立，$R$ 不变；两噪声可相关 | 第 1.8 节 |
| 输出附加噪声不改四项 | $(N_U,N_V)\perp(U,V,W)$，$\widetilde{\boldsymbol u}=(U,N_U),\widetilde{\boldsymbol v}=(V,N_V)$：<br>$(R',\mathrm{Un}'_U,\mathrm{Un}'_V,S')=(R,\mathrm{Un}_U,\mathrm{Un}_V,S)$，其中 $U,V$ 为输出、$W$ 为源 | 第 1.8 节 |
| 有限字母表连续性 | 固定有限字母表，$p_k\to p\Rightarrow(R_k,\mathrm{Un}_{U,k},\mathrm{Un}_{V,k},S_k)\to(R,\mathrm{Un}_U,\mathrm{Un}_V,S)$ | 第 1.9 节 |
| 最优值可达 | $m=\lvert\operatorname{supp}(W)\rvert$ 时，$C_{\mathrm{ext}}(U,V;W)=\max_{Z\leftarrow W,\ \lvert\mathcal Z\rvert\le m+1}\operatorname{CoI}(U,V,Z)$ | 第 1.9 节 |
| 条件独立时无协同 | $U\perp V\mid W\Rightarrow R=I(U;V),\ S=0$ | 第 1.10 节 |
| 完整冗余链式法则 | 一般不成立；独立公平源的 XOR 条件化给出反例 | 第 1.11 节 |

表格独立导出：[PNG](../../fig/ei_decomposition_properties.png) · [SVG](../../fig/ei_decomposition_properties.svg)。

### 1.4 保证非负分配的上下界

**命题 1。** 对任意三个有限离散变量，

$$
\boxed{
\max\{0,\operatorname{CoI}(U,V,W)\}
\le R(U,V;W)
\le\min\{I(U;V),I(U;W),I(V;W)\}.
}
\tag{1-8}
$$

且 $R(U,V;W)=R(V,U;W)$。

**证明。** 常量处理给出共信息 0，恒等处理给出原共信息，故下界成立。对任意 $Z\leftarrow W$，

$$
\begin{aligned}
\operatorname{CoI}(U,V,Z)
&=I(U;V)-I(U;V\mid Z)\\
&=I(U;Z)-I(U;Z\mid V)\\
&=I(V;Z)-I(V;Z\mid U).
\end{aligned}
\tag{1-9}
$$

条件 MI 非负，共信息不超过 $I(U;V),I(U;Z),I(V;Z)$。数据处理给出 $I(U;Z)\le I(U;W)$、$I(V;Z)\le I(V;W)$，于是每个可行通道均受式（1-8）上界约束，取上确界后仍成立。交换 $U,V$ 不改变共信息或可行通道，因此两变量交换对称。证毕。

这组界来自单端优化本身。后面的四项非负、独立源退化及递归闭合只需要这些界，不要求源与目标完全交换对称。

### 1.5 四项非负、精确收支及并集信息

统一记两变量一侧为 $U,V$，单变量一侧为 $W$：

$$
R=C_{\mathrm{ext}}(U,V;W),\quad
\mathrm{Un}_U=I(U;W)-R,\quad \mathrm{Un}_V=I(V;W)-R,\quad
S=R-\operatorname{CoI}(U,V,W).
\tag{1-10}
$$

**命题 2。** 四项非负，且

$$
I(U;W)=R+\mathrm{Un}_U,\quad I(V;W)=R+\mathrm{Un}_V,\quad
I(U,V;W)=R+\mathrm{Un}_U+\mathrm{Un}_V+S.
\tag{1-11}
$$

**证明。** 任意采用该收支的二元分配，四项非负的充要条件为

$$
\max\{0,\operatorname{CoI}(U,V,W)\}
\le R\le\min\{I(U;W),I(V;W)\}.
\tag{1-12}
$$

这是分别展开 $R,\mathrm{Un}_U,\mathrm{Un}_V,S\ge0$ 的结果。命题 1 满足全部约束，故非负；直接代入相加得到收支。证毕。

定义并集信息量

$$
J:=R+\mathrm{Un}_U+\mathrm{Un}_V
=I(U;W)+I(V;W)-R(U,V;W).
\tag{1-13}
$$

其界为

$$
\max\{I(U;W),I(V;W)\}\le J
\le\min\{I(U,V;W),I(U;W)+I(V;W)\}.
\tag{1-14}
$$

证明分别使用 $R\le I(U;W),I(V;W)$ 与 $R\ge\operatorname{CoI},0$。共有只计一次后的并集覆盖每个单变量信息，不超出联合信息。

### 1.6 独立源精确退化为 PEID

**命题 3。** 若 $U\perp V$，则

$$
R=0,\quad \mathrm{Un}_U=I(U;W),\quad \mathrm{Un}_V=I(V;W),\quad
S=I(U,V;W)-I(U;W)-I(V;W)=I(U;V\mid W)\ge0.
\tag{1-15}
$$

**证明。** 命题 1 的上界含 $I(U;V)=0$，故 $R=0$。以 $p$ 记当前概率质量函数，代入共信息展开，

$$
S=I(U;V\mid W)
=\mathbb E_{p(W)}
D_{\mathrm{KL}}\!\left(p(U,V\mid W)
\middle\|p(U\mid W)p(V\mid W)\right)\ge0.
\tag{1-16}
$$

其余两项由定义得到。证毕。

证明只需要独立，不需要公平或均匀边缘。若采用当前 PEID 的因子化最大熵干预，源独立自然成立，式（1-15）精确恢复其二源数值定义。任意一对变量独立都会使 $R=0$，因此 $1\to2$ 若实际输出独立，也有零冗余；独立性均在当前给定分布下判断。

### 1.7 恒等性、包含、自冗余及重编码

**命题 4。** 本定义满足：

- 双向无损恒等性：$H(W\mid U,V)=H(U,V\mid W)=0$ 时，$R=I(U;V)$。
- 确定性包含：$H(U\mid V)=0$ 时，$R=I(U;W)$、$\mathrm{Un}_U=0$。
- 各变量分别作一一重编码时，冗余及对应四项不变。
- 处理端数据处理：若 $T\leftarrow W$ 是局部通道，则 $R(U,V;T)\le R(U,V;W)$。
- 单预测变量自冗余可相容地定义为 $R(U;W):=I(U;W)$。

**证明。** 双向无损意味着 $W$ 与源对在正概率支持上一一对应，给定 $W$ 后两变量确定，故 $\operatorname{CoI}=I(U;V)$；上下界相等得到恒等性，也可无损重编码到 $W=\langle U,V\rangle$。

若 $U=f(V)$，则 $I(U,V;W)=I(V;W)$、$\operatorname{CoI}=I(U;W)$；上界也不超过此值，得到包含等号。数据处理保证 $I(U;W)\le I(V;W)$。

各变量分别一一重编码时，MI、CMI 不变，可行局部通道也一一对应，故冗余及四项不变。若 $T\leftarrow W$，每个 $Z\leftarrow T$ 都可复合成 $Z\leftarrow W$，取上确界得到处理端数据处理。单预测变量情形按普通 MI 定义，与二元包含退化相容。证毕。

### 1.8 附加无关噪声分量不改变冗余

**命题 5。** 在任意两个位置附加与原 $(U,V,W)$ 联合独立的噪声分量，允许这两个噪声相关，冗余不变。具体记 $\widetilde{\boldsymbol u}=(U,N_U)$、$\widetilde{\boldsymbol v}=(V,N_V)$、$\widetilde{\boldsymbol w}=(W,N_W)$，则

$$
\begin{aligned}
R(\widetilde{\boldsymbol u},\widetilde{\boldsymbol v};W)&=R(U,V;W),\\
R(\widetilde{\boldsymbol u},V;\widetilde{\boldsymbol w})&=R(U,V;W).
\end{aligned}
\tag{1-17}
$$

每个等式分别要求所用噪声对与原三个变量联合独立。交换 $U,V$ 包含第三种位置组合。两个未处理位置均为输出时，四项全部不变；不据此要求三个位置同时附加任意相关噪声也不变。

**证明。** 先考虑两个未处理位置。对每个 $Z\leftarrow W$，噪声对与 $(U,V,W,Z)$ 独立，因此

$$
\begin{aligned}
I(\widetilde{\boldsymbol u};\widetilde{\boldsymbol v})&=I(U;V)+I(N_U;N_V),\\
I(\widetilde{\boldsymbol u};\widetilde{\boldsymbol v}\mid Z)&=I(U;V\mid Z)+I(N_U;N_V).
\end{aligned}
\tag{1-18}
$$

两个噪声项相减抵消，每个候选共信息不变，可行 $W\to Z$ 通道也不变，得到第一个等式。

再考虑一个未处理位置与处理端。令 $(N_U,N_W)\perp(U,V,W)$。任取 $Z\leftarrow\widetilde{\boldsymbol w}$，

$$
\begin{aligned}
\operatorname{CoI}(\widetilde{\boldsymbol u},V,Z)
&=I(U;V)-I(\widetilde{\boldsymbol u};V\mid Z)\\
&=I(U;V)-I(N_U;V\mid Z)-I(U;V\mid Z,N_U)\\
&\le I(U;V)-I(U;V\mid Z,N_U).
\end{aligned}
\tag{1-19}
$$

给定正概率 $N_U=n$，原三个变量的联合分布不变，而

$$
\kappa_n(Z\mid W)
:=\sum_{n_W}P(N_W=n_W\mid N_U=n)\kappa(Z\mid W,n_W)
\tag{1-20}
$$

是可行的原 $W\to Z$ 通道。因此每个条件态的 $I(U;V\mid Z,N_U=n)$ 都不小于 $\inf_{Z'\leftarrow W}I(U;V\mid Z')$，平均后仍成立，式（1-19）不超过 $R(U,V;W)$。对扩展通道取上确界得到一侧不等式。反向由忽略 $N_W$ 的原通道实现：此时 $N_U$ 与 $(U,V,Z)$ 独立，候选共信息不变。得到第二个等式。

若两个未处理位置是输出，附加噪声还保持 $I(W;\widetilde{\boldsymbol u})=I(W;U)$、$I(W;\widetilde{\boldsymbol v})=I(W;V)$、$I(W;\widetilde{\boldsymbol u},\widetilde{\boldsymbol v})=I(W;U,V)$，所以四项均不变。证毕。

这里的操作保留原变量并附加分量，不包括加法噪声、翻转、遮蔽或信号替换。扩展输出分布为 $p(\widetilde{\boldsymbol u},\widetilde{\boldsymbol v}\mid W)=p(U,V\mid W)p(N_U,N_V)$。

例如 $U=V=W$ 为公平比特，向两输出附加相同的独立公平噪声 $N$，输出间 MI 从 1 变为 2，但关于源的冗余仍为 1、四项不变。若改为 $U=W,V=W\oplus N$，第二输出与源独立，冗余变为 0，1 bit 归 $U$ 特有。一般翻转率 $\varepsilon\le1/2$ 时，$I(W;W\oplus N)=1-h_2(\varepsilon)$，仅在 $\varepsilon=1/2$ 时单输出信息消失。

### 1.9 有限状态最优值可达与连续性

**命题 6。** 被处理变量 $W$ 有 $m$ 个正概率状态时，

$$
C_{\mathrm{ext}}(U,V;W)
=\max_{Z\leftarrow W,\ |\mathcal Z|\le m+1}\operatorname{CoI}(U,V,Z).
\tag{1-21}
$$

固定有限字母表上 $p_k\to p$ 逐项收敛时，$(R_k,\mathrm{Un}_{U,k},\mathrm{Un}_{V,k},S_k)\to(R,\mathrm{Un}_U,\mathrm{Un}_V,S)$，包括支持边界。

**证明。**

设被处理变量 $W$ 有 $m$ 个正概率状态，去掉零概率状态后记其边缘概率向量为 $\boldsymbol{p}$。对辅助变量 $Z$ 的每个取值 $z$，记其概率为 $\lambda_z$，对应的 $W$ 后验概率向量为 $\boldsymbol{r}_z$。它们满足

$$
\lambda_z\ge0,\qquad\sum_z\lambda_z=1,\qquad
\sum_z\lambda_z\boldsymbol{r}_z=\boldsymbol{p}.
\tag{1-22}
$$

因为 $Z$ 仅由 $W$ 产生，对应的条件分布为 $\sum_{\bar w}r_z(\bar w)p(U,V\mid W=\bar w)$。定义单纯形上连续函数

$$
f(\boldsymbol{r}):=
I_{\sum_{\bar w}r(\bar w)p(U,V\mid W=\bar w)}(U;V).
\tag{1-23}
$$

则要最小化的条件 MI 为 $\sum_z\lambda_zf(\boldsymbol{r}_z)$。函数图像 $(\boldsymbol{r},f(\boldsymbol{r}))$ 位于 $m$ 维仿射空间：$\boldsymbol{r}$ 的自由维数为 $m-1$，函数值再占一维。其图像紧，有限维凸包亦紧。

最小值就是该凸包中第一坐标为 $\boldsymbol{p}$ 时的最低函数坐标。可行切片非空且紧，故最低值可达。Carathéodory 定理保证该点可由至多 $m+1$ 个图像点混合得到，因此 $|\mathcal Z|\le m+1$ 足够。

每个混合解都可还原成局部通道：对 $p(w)>0$ 的状态 $w$，

$$
p(z\mid w)=\frac{\lambda_zr_z(w)}{p(w)}.
\tag{1-24}
$$

式（1-22）保证每行和为 1，且该通道产生相同后验及目标值。这里给出充分界，不声称它在所有分布上最紧。

为证明连续性，固定整个 $W$ 字母表大小 $M$，统一使用 $M+1$ 个辅助状态，必要时增加零概率状态。所有行随机通道构成不依赖原分布的紧集合。联合概率与通道共同决定的有限联合分布连续，MI、CMI 和共信息在有限概率单纯形上连续，取该紧域上的最大值仍随原联合概率连续变化。因此单端冗余随原联合概率连续变化，其余项由连续 MI 加减得到。这个证明包括支持变化，不依赖在支持边界定义后验。

$m+1$ 为充分界，不一定最紧。二状态处理端最多三个辅助状态即可，但概率参数仍连续，不等于有限次确定性枚举，也不保证局部优化全局最优。

连续性指信息量对分布的稳定性，不保证最优通道唯一、连续或可微。经验概率一致收敛时，精确指标亦一致收敛；数值求解还须优化误差趋零，没有给出有限样本速度。例如 $U=W$ 公平、$V=W\oplus N$、翻转率 $\varepsilon$ 时，上下界相等，$R=1-h_2(\varepsilon)\to1$，不会在任意正噪声下跳到零。

### 1.10 条件独立时的分配与协同上界

**命题 7。** 若 $U\perp V\mid W$，则

$$
R(U,V;W)=I(U;V),\qquad S=0.
\tag{1-25}
$$

一般情况下，

$$
0\le S\le\min\{I(U;V\mid W),I(U;W\mid V),I(V;W\mid U)\}.
\tag{1-26}
$$

**证明。** 条件独立使 $\operatorname{CoI}=I(U;V)$，命题 1 上下界相等。一般有 $S=R-\operatorname{CoI}\le I(U;V)-\operatorname{CoI}=I(U;V\mid W)$；另两对变量同理。

这些结论直接来自指定端 $C_{\mathrm{ext}}$ 的上下界；无需跨角色取最大值。证毕。

因此 $1\to2$ 若输出分别是源的确定性函数，或给定源后独立，本定义的协同为零，观测收益分到冗余及特有。秘密共享的正协同依赖给定源后仍存在的联合输出随机结构；这是本单端定义的性质。

### 1.11 完整冗余链式法则的反例

条件冗余在每个正概率条件分布中重新用指定端 $C_{\mathrm{ext}}$，再按条件概率平均。完整目标链式法则（LC）要求

$$
R(U,V;\langle W,W'\rangle)
=R(U,V;W)+\sum_{\bar w}p(\bar w)R(U,V;W'\mid W=\bar w).
\tag{1-27}
$$

**命题 8。** $R(U,V;W)=C_{\mathrm{ext}}(U,V;W)$ 保留普通恒等性及逐条件非负，但不满足完整 LC。

**证明。**

取独立公平比特 $U,V$，$W=U\oplus V$、$W'=U$。对原源对，$I(U;V)=0$，所以

$$
R(U,V;W)=0,\qquad
R(U,V;\langle W,W'\rangle)=0.
\tag{1-28}
$$

给定 $W=\bar w$ 后，$V=U\oplus\bar w$，而 $U$ 仍公平。用包含性质，在条件分布上

$$
R(U,V;W'\mid W=\bar w)=I(U;U\mid W=\bar w)=1.
\tag{1-29}
$$

所以逐条件平均为 1，完整 LC 要求 $0=0+1$，矛盾。源在最初分布下独立，条件化后依赖；反例没有靠“不独立的原源”才成立。

对任意候选公理包也可看出同一问题：联合目标 $\langle W,W'\rangle$ 与 $(U,V)$ 一一对应，普通恒等性和无损目标重编码给出冗余 0；逐条件恒等性给出条件冗余 1；完整 LC 加冗余非负不能成立。这个直接反例的约定比一句“恒等与 LC 冲突”更完整，不能省略条件量定义与重编码假设。

Finn–Lizier 的已发表 Theorem 6 表述涉及目标链式法则、恒等性质和所有 PID 原子的 local positivity。[18](#ref-18) 本文在上述明确约定下展示直接矛盾，不把其原定理悄然改写成仅一个未限定的冗余非负命题。

后文使用 MI 或 KL 本身的链式法则，不对冗余强加式（1-27）。条件化会使最初独立的源相关；只声明原源独立不能回避该反例。二元噪声不变和连续性也不能未经证明移植到双侧树。

## 2. 二元机制例子：从复制到共有与协同并存

### 2.1 2→1：复制、AND 与 XOR

![XOR 的源协同与复制的输出冗余：各自只处理指定端](../../fig/ei_decomposition_xor_copy.png)

**图 1　两种二元方向使用同一个提取定义。** 左图 $(U,V)\to W$ 只处理目标 $W$，右图 $W\to(U,V)$ 只处理源 $W$。常量／恒等通道分别达到冗余上界，得到纯协同与纯冗余各 1 bit。箭头表示原生成机制，辅助通道只参与信息分配。图中结论由精确概率表得到；数值非负容差为 0 bit。[SVG](../../fig/ei_decomposition_xor_copy.svg)

$U,V$ 为独立公平比特。AND 门 $W=U\land V$ 有 $H(W)=h_2(1/4)$、$H(W\mid U)=H(W\mid V)=1/2$，所以

$$
(R,\mathrm{Un}_{U},\mathrm{Un}_{V},S)
=\left(0,h_2(1/4)-\tfrac12,h_2(1/4)-\tfrac12,1-h_2(1/4)\right)
\approx(0,0.311278,0.311278,0.188722).
\tag{2-1}
$$

零冗余来自源独立；单源信息仍为正，协同是联合 EI 的余额。XOR 门 $W=U\oplus V$ 的各单源与输出独立，联合源确定输出，故为 $(0,0,0,1)$。

无损复制 $W=\langle U,V\rangle$ 的单 EI 均为 1、联合 EI 为 2，分配为 $(0,1,1,0)$。“输出同时含两个源”不自动表示协同；XOR 要求联合两个源读取。

### 2.2 1→2：四个基本机制

下列 $U_0,V_0,C,T,N$ 相互独立公平，$\langle U_0,V_0\rangle=2U_0+V_0$ 为四状态标量编码。

- **复制：** $W=C,U=V=C$。上下界都为 1，四项 $(1,0,0,0)$。
- **分流：** $W=\langle U_0,V_0\rangle,U=U_0,V=V_0$。输出独立、单 EI 均 1、联合 EI 为 2，四项 $(0,1,1,0)$。
- **秘密共享：** $W=T,U=N,V=N\oplus T$。各输出与源独立、联合恢复源，四项 $(0,0,0,1)$。
- **共有与协同并存：** $W=\langle C,T\rangle,U=\langle C,N\rangle,V=\langle C,N\oplus T\rangle$。单 EI 均 1、联合 EI 为 2，原共信息零；处理 $W\mapsto C$ 显露 1 bit 共信息，达到 $I(U;V)=1$ 上界，四项 $(1,0,0,1)$。

最后一例解释局部提取的必要性：共有可被协同抵消，仅原共信息的正部不够。

![随机掩码的生成机制、条件输出组合及单端信息分解](../../fig/ei_decomposition_random_mask.png)

**图 2　秘密共享为何属于输出协同。** 图中 $W$ 对应上述源比特 $T$。只对 $W$ 作均匀干预，$N$ 保持独立随机背景。单独任一输出均不提供源信息，联合输出恢复 $W=U\oplus V$。冗余只允许处理源 $W$；常量通道达到零上界，四项为 $(0,0,0,1)$ bit。精确计算的非负容差为 0 bit。[SVG](../../fig/ei_decomposition_random_mask.svg)

### 2.3 带噪机制、共同无关噪声及解析最优性

下表基本比特 $U_0,V_0,C,T,N$ 相互独立公平；翻转噪声 $F,F_1,F_2$ 与这些比特独立，两个 $F_i$ 彼此独立。$\langle\cdot,\cdot\rangle$ 表示无损的有限状态标量编码。对每行仅干预源 $W$，背景随机性由原机制保留。

记 $h_2(p)=-p\log_2p-(1-p)\log_2(1-p)$、$k=1-h_2(0.1)\simeq0.531004$、$d=1-h_2(0.18)\simeq0.319923$。表中单位均为 bit。

**表 2　九种输出机制。** $R=C_{\mathrm{ext}}(U,V;W)$，只处理源 $W$；$\mathrm{Un}_U,\mathrm{Un}_V,S$ 对应式（1-6）的输出分配。

| 1→2 机制 | $R$ | $\mathrm{Un}_U$ | $\mathrm{Un}_V$ | $S$ | 解释 |
|---|---:|---:|---:|---:|---|
| $W=C$；$U=V=C$ | 1 | 0 | 0 | 0 | 完整复制 |
| $W=\langle U_0,V_0\rangle$；$U=U_0,V=V_0$ | 0 | 1 | 1 | 0 | 独立分流 |
| $W=T$；$U=N,V=N\oplus T$ | 0 | 0 | 0 | 1 | 秘密共享 |
| $W=\langle C,T\rangle$；$U=\langle C,N\rangle,V=\langle C,N\oplus T\rangle$ | 1 | 0 | 0 | 1 | 共有与协同同时存在 |
| $W=\langle U_0,V_0\rangle$；$U=\langle U_0,N\rangle,V=\langle V_0,N\rangle$ | 0 | 1 | 1 | 0 | 共同无关噪声不制造源冗余 |
| $W=C$；$U=C,V=C\oplus F$，翻转率 0.1 | $k$ | $1-k$ | 0 | 0 | 带噪重叠 |
| $W=\langle C,T\rangle$；$U=\langle C,N\rangle,V=\langle C\oplus F,N\oplus T\rangle$ | $k$ | $1-k$ | 0 | 1 | 带噪共有与秘密共享并存 |
| $W=\langle U_0,V_0\rangle$；$U=U_0,V=U_0\land V_0$ | 0.311278 | 0.688722 | 0.5 | 0 | 确定性概率重叠 |
| $W=C$；$U=C\oplus F_1,V=C\oplus F_2$，翻转率均为 0.1 | $d$ | $k-d$ | $k-d$ | 0 | 条件独立观测；由第 1.10 节解释 |

表格独立导出：[PNG](../../fig/ei_decomposition_mechanisms.png) · [SVG](../../fig/ei_decomposition_mechanisms.svg)。

这些精确最优值有解析依据：上下界相等、提取 $C$ 达到二元 MI 上界，或由噪声不变性归约得到。数值分组搜索只是核对，不能作为随机优化达到最优的唯一依据。解析依据如下。

还应保留“输出独立但边缘通道相同”的例子：$W$ 公平，给定 $W=0$ 的 $(U,V)$ 在 $00,01,10,11$ 上的概率为 $(1/2,1/4,1/4,0)$，给定 $W=1$ 时为 $(0,1/4,1/4,1/2)$。两个输出独立，两条通道均为交叉率 $1/4$ 的 BSC。于是 $I(W;U)=I(W;V)=0.188722$、$I(W;U,V)=0.5$；本定义给出 $(0,0.188722,0.188722,0.122556)$。这说明相同边缘通道并不强迫本定义给出正冗余。

复制、独立分流和纯秘密共享的值分别由包含、独立零冗余和 MI 收支决定。确定性重叠 $W=\langle U_0,V_0\rangle$、$U=U_0,V=U_0\land V_0$ 给定源后输出条件独立，命题 7 给出 $R=I(U;V)=0.311278\ldots$。

共有加秘密共享例中，$I(U;V)=1$；源处理 $W\mapsto C$ 得到共信息 1，达到上界。带噪版本同理有 $I(U;V)=k$，处理为 $C$ 后两输出给定 $C$ 独立，其共信息为 $k$，达到上界。共同无关噪声例由第 1.8 节归约为独立分流。

精确副本加带噪副本时 $I(W;U)=I(W;U,V)=1$、$I(W;V)=k$，共信息也为 $k$，界迫使 $R=k$。两个独立噪声副本给定 $W$ 条件独立，$R=d$、$S=0$。其联合 MI 可写为

$$
I(W;U,V)=1-\left[0.82h_2(1/82)+0.18\right]
\simeq0.742085859,
\tag{2-2}
$$

其中两个输出一致的概率为 0.82，一致时后验错误概率为 $0.01/0.82=1/82$；输出不一致时源后验公平。故 $I(W;U,V)-k=k-d=0.211081\ldots$。

上述构造给出可行局部通道达到理论上界或上下界相等，因而不仅确定性处理、随机处理也不能超出表中最优值。

两个 0.1 独立噪声副本的 $R=d,\mathrm{Un}_U=\mathrm{Un}_V=k-d\approx0.211081,S=0$；BROJA 给出 $R=k$、特有零、$S\approx0.211081$。[4](#ref-4) 这是不同共同内容语义。本定义不要求边缘通道相同或 Blackwell 等价时特有必为零。

**完整联合结构的重要性。** “共有加秘密共享”与“不同源比特加共同无关噪声”具有相同标量输入：

$$
I(W;U)=I(W;V)=1,\quad I(W;U,V)=2,\quad
H(U)=H(V)=2,\quad I(U;V)=1,\quad H(W)=2.
\tag{2-3}
$$

前者冗余 1，后者由噪声附加不变性得到冗余 0。仅依赖这些标量不能区分，局部通道及完整联合分布提供额外依据。

## 3. 2→2：把二元分解延拓到固定协同残余

### 3.1 延拓的逻辑：固定父项，再定义内部读出

高阶部分只给变量加索引：源记为 $U_i$，目标记为 $V_j$；源、目标向量分别记为粗体小写 $\boldsymbol u,\boldsymbol v$。在 $2\to2$ 中，令 $\boldsymbol{u}=(U_1,U_2)$、$\boldsymbol{v}=(V_1,V_2)$。本节复用独立源的 PEID／SPT，因此取 $U_1\perp U_2$；源边缘无需均匀，目标条件分布任意。把联合目标视为有限变量，第 1.2 节在这一独立情形下给出

$$
\begin{aligned}
I(\boldsymbol{u};\boldsymbol{v})
&=I(U_1;\boldsymbol{v})+I(U_2;\boldsymbol{v})+s,\\
s&=I(U_1;U_2\mid\boldsymbol{v})\ge0.
\end{aligned}
\tag{3-1}
$$

前两个父项仍是 MI，分别再用第 1.3 节得到各自四项。第三项 $s$ 是 KL 散度，不是可直接充当新源的随机变量；需给它的内部读出一个定义。

延拓保留同一分配逻辑：提取共同部分，扣除后得到特有，保留联合余额。两个 MI 行直接复用 $1\to2$ 的单端定义；协同行则需把读出换成固定参照 KL，并证明相应界。下面的反例说明这一额外设计为何必要。

### 3.2 为什么不能重新计算各子目标协同

独立公平源取

$$
V_1=U_1\oplus U_2,\quad V_2=U_1,\qquad
s(V_1)=1,\quad s(V_2)=0,\quad s(V_1,V_2)=0.
\tag{3-2}
$$

固定联合协同零若非负细分，子项只能全零，不可能同时恢复 $s(V_1)=1$。因此恢复全部边缘 PEID 与固定父项非负细分一般不相容。后文保持父项，另定义其读出。

若两个输出只复制 $U_1$，源协同也为零，整体源的目标冗余却可为 1，后者不能直接放入零协同行。整体目标冗余与固定协同内部冗余是不同量。

### 3.3 固定后验依赖的 KL 参照

沿用源向量 $\boldsymbol u$，以 $q$ 表示当前联合分布，定义

$$
\begin{aligned}
P(\boldsymbol u,\boldsymbol v)&:=q(U_1,U_2,\boldsymbol v),\\
Q(\boldsymbol u,\boldsymbol v)&:=q(\boldsymbol v)
q(U_1\mid\boldsymbol v)q(U_2\mid\boldsymbol v),\\
s&=D_{\mathrm{KL}}(P\|Q).
\end{aligned}
\tag{3-3}
$$

$Q$ 删除完整输出条件下的源间依赖，是信息论参照，不是另一份因子化干预机制，一般改变源边缘。

**命题 9。** $Q$ 归一化；有限状态下 $P\ll Q$，且 $D_{\mathrm{KL}}(P\|Q)=s$。

**证明。** 固定正概率目标后，两后验边缘乘积的和为 1，再对 $q(\boldsymbol v)$ 求和，故归一。任一 $P$ 正状态的目标边缘与两源后验边缘均正，故 $Q$ 也正。把比值代入 KL 求和就是条件 MI。证毕。

目标子集只能边缘化这份 $Q$，不能各自重新乘局部后验，否则又回到式（3-2）的不相容父预算。

### 3.4 定义固定残余的目标读出

保留参照源边缘差异

$$
b:=D_{\mathrm{KL}}(P_{\boldsymbol u}\|Q_{\boldsymbol u}).
\tag{3-4}
$$

对目标集合 $H$，定义

$$
\begin{aligned}
F(H)&:=D_{\mathrm{KL}}(P_{\boldsymbol u,\boldsymbol v_H}
\|Q_{\boldsymbol u,\boldsymbol v_H})-b\\
&=\mathbb E_{P(\boldsymbol u)}
D_{\mathrm{KL}}\!\left(P(\boldsymbol v_H\mid\boldsymbol u)
\middle\|Q(\boldsymbol v_H\mid\boldsymbol u)\right).
\end{aligned}
\tag{3-5}
$$

**命题 10。** 对 $G\subseteq H$，

$$
F(\varnothing)=0,\quad
0\le F(G)\le F(H)\le F(\{1,2\}),\qquad s=b+F(\{1,2\}).
\tag{3-6}
$$

**证明。** KL 链式法则给式（3-5）第二行，从而非负。逐 $\boldsymbol u$ 边缘化目标并用 KL 数据处理，再按 $P(\boldsymbol u)$ 平均，得包含单调。完整集合等式同样来自 KL 链式法则。证毕。

$F(H)$ 衡量给定源状态时，目标子块区分实际分布与固定参照的程度，是固定残余的目标可辨识性预算，不是子目标的新 PEID。$b$ 是参照源边缘偏移的未细分余额，不是输出无关噪声，也不能命名为某个目标的原子。

### 3.5 在 KL 残余内部提取共同读出

对不交目标块 $G,H$，将同一个局部核 $\kappa(z\mid\boldsymbol v_G)$ 施加到 $P,Q$，保持两分布之间的比较相容，并以式（3-5）定义 $F(Z)$、$F(Z,\boldsymbol v_H)$。定义

$$
\begin{aligned}
c_G(\kappa)&:=F(Z)+F(H)-F(Z,\boldsymbol v_H),\\
r_F(G,H)&:=\max\left\{
\sup_{Z\leftarrow\boldsymbol v_G}c_G(\kappa),
\sup_{Z\leftarrow\boldsymbol v_H}c_H(\kappa)\right\}.
\end{aligned}
\tag{3-7}
$$

这里的读出是 $F$，不能直接把标量 KL 余额当作随机变量套用 $C_{\mathrm{ext}}$。两个目标方向用于使无序块 $G,H$ 交换时共同读出不变；非负证明分别适用于每个方向，不需要处理源端。二元 MI 的噪声不变、连续性及辅助状态界需为该 KL 读出另作证明。

**命题 11。** 记 $GH=G\cup H$，则

$$
\boxed{\max\{0,F(G)+F(H)-F(GH)\}
\le r_F(G,H)\le\min\{F(G),F(H)\}.}
\tag{3-8}
$$

**证明。** 常量核给 $F(Z)=0,F(Z,\boldsymbol v_H)=F(H)$，候选零；恒等核给 $F(G)+F(H)-F(GH)$。同核条件 KL 数据处理给

$$
F(Z,\boldsymbol v_H)\ge F(Z),\quad
F(Z,\boldsymbol v_H)\ge F(H),\quad F(Z)\le F(G).
\tag{3-9}
$$

所以 $c_G\le F(H)$ 且 $c_G\le F(Z)\le F(G)$。另一端同理，取上确界及最大值保留所有界。证毕。

### 3.6 协同的五项分配与 13 项闭合

定义

$$
\begin{aligned}
R_s&:=r_F(G,H),\\
\mathrm{Un}_{s,G}&:=F(G)-r_F(G,H),\qquad \mathrm{Un}_{s,H}:=F(H)-r_F(G,H),\\
S_s&:=F(GH)-F(G)-F(H)+r_F(G,H).
\end{aligned}
\tag{3-10}
$$

**命题 12。** 四项非负并加总为 $F(GH)$。两块覆盖完整目标时，

$$
\boxed{s=b+R_s+\mathrm{Un}_{s,G}+\mathrm{Un}_{s,H}+S_s.}
\tag{3-11}
$$

**证明。** 共同读出非负来自下界零，特有非负来自两个上界，联合余额非负来自另一条下界。直接相加得到 $F(GH)$，再用命题 10。证毕。

记单源 $U_i\to(V_1,V_2)$ 的原四项为 $(R_i,\mathrm{Un}_{i,1},\mathrm{Un}_{i,2},S_i)$，完整闭合为

$$
\boxed{
I(\boldsymbol{u};\boldsymbol v)=\sum_{i=1}^2(R_i+\mathrm{Un}_{i,1}+\mathrm{Un}_{i,2}+S_i)
+b+R_s+\mathrm{Un}_{s,1}+\mathrm{Un}_{s,2}+S_s.
}
\tag{3-12}
$$

两个单源行各四项，协同行五项，共 13 个位置；$b=0$ 时省去该零项可为十二项。每行非负精确细分，数值收支不重复计账。尚未建立兼容全部边缘分解的 ΦID 表，也未证明信息内容互斥。[6](#ref-6)、[7](#ref-7)

**表 3　2→2 的三个固定父预算。**

| 父行 | 原预算 | 目标细分 | 非负与收支依据 |
|---|---|---|---|
| 单源 $U_1$ | $I(U_1;V_1,V_2)$ | $R_1,\mathrm{Un}_{1,1},\mathrm{Un}_{1,2},S_1$，只处理 $U_1$ | 原 $1\to2$ 定义，命题 1—2 |
| 单源 $U_2$ | $I(U_2;V_1,V_2)$ | $R_2,\mathrm{Un}_{2,1},\mathrm{Un}_{2,2},S_2$，只处理 $U_2$ | 原 $1\to2$ 定义，命题 1—2 |
| 源协同 | $s=D_{\mathrm{KL}}(P\Vert Q)$ | $b,R_s,\mathrm{Un}_{s,1},\mathrm{Un}_{s,2},S_s$，全行固定同一 $Q$ | KL 链与共同读出界，命题 10—12 |

三行先固定预算，再逐行细分；各行内部只计一次，总和即式（3-12）。

### 3.7 2→2 的逐步例子

例 1、2、4、5 均取 $U_1,U_2$ 为独立公平比特，辅助随机比特与源独立；例 3 另给出四状态源的构造。

**例 1：协同被两个输出复制。** $V_1=V_2=U_1\oplus U_2$。两个单源父项为零、$s=1$。给定输出时各源后验仍公平，$Q=P_{\boldsymbol u}P_{\boldsymbol v}$、$b=0$。于是 $F(1)=F(2)=F(12)=1$，恒等核达上界 $r_F=1$，协同行 $(b,R_s,\mathrm{Un}_{s,1},\mathrm{Un}_{s,2},S_s)=(0,1,0,0,0)$。

**例 2：协同需联合两个输出。** 独立公平 $N$，$V_1=N,V_2=N\oplus U_1\oplus U_2$。$s=1,b=0$，$F(1)=F(2)=0,F(12)=1$；上界迫使 $r_F=0$，协同行 $(0,0,0,0,1)$。

**例 3：共有与联合余额同时存在。** $A_1,A_2,B_1,B_2,N$ 全部独立公平，源为四状态编码 $U_1=\langle A_1,A_2\rangle,U_2=\langle B_1,B_2\rangle$。令 $C=A_1\oplus B_1,T=A_2\oplus B_2$，输出也用无损编码 $V_1=\langle C,N\rangle,V_2=\langle C,N\oplus T\rangle$。$s=2,b=0,F(1)=F(2)=1,F(12)=2$，恒等候选为零；处理 $V_1\mapsto C$ 给候选 1，达到上界，协同行 $(0,1,0,0,1)$。这承接第 2.2 节混合机制，说明不能仅取未经处理的正共同读出。

**例 4：XOR＋复制。** 式（3-2）的联合 $s=0$，协同行五项均零。单源父项均为 1：原 $1\to2$ 定义将第一行放在 $V_2$ 特有位置，第二行放在目标协同位置，两行 EI 合计 2。单独 XOR 目标的源协同 1 不要求从固定零协同行恢复。

**例 5：AND／OR 与非零参照边缘。** $V_1=U_1\land U_2,V_2=U_1\lor U_2$ 时，

$$
\begin{aligned}
s&=\tfrac12,\quad P_{\boldsymbol u}=(\tfrac14,\tfrac14,\tfrac14,\tfrac14),\\
Q_{\boldsymbol u}&=(\tfrac38,\tfrac18,\tfrac18,\tfrac38),\\
b&=\tfrac12\log_2\tfrac43\approx0.2075187496,\\
F(1)&=F(2)\approx0.1462406252,\qquad F(12)\approx0.2924812504.
\end{aligned}
\tag{3-13}
$$

只用常量／恒等候选时 $r_{F,\mathrm{lo}}=0$，协同行粗分配 $(0.2075187496,0,0.1462406252,0.1462406252,0)$ 恰好为 0.5。第 5.1 节定义此候选版；未证明随机通道最优冗余零。单源两行预算均 0.5，整体 EI 为 1.5。非零 $b$ 表明本构造不能总压缩成十二个目标类型位置。

## 4. N→1、1→N 与 N→N：二分算法及整体证明

### 4.1 从两个源推广到两个独立源块

本节复用原源 SPT，取源相互独立的 $n$ 源、$m$ 目标联合分布

$$
q(\boldsymbol{u},\boldsymbol{v})
=\left[\prod_{i=1}^n q_i(u_i)\right]
q(\boldsymbol{v}\mid\boldsymbol{u}),\quad
\boldsymbol{u}=(U_1,\ldots,U_n),\quad \boldsymbol{v}=(V_1,\ldots,V_m).
\tag{4-1}
$$

各源边缘 $q_i$ 和目标条件分布可以任意，均匀性与干预并非本节证明的条件。所有源目标子集均从同一联合分布边缘化。若计算当前 PEID 的 EI，则采用其共同因子化最大熵干预，并固定支持与时间跨度。定义

$$
\Xi(A):=I(\boldsymbol{u}_A;\boldsymbol v)-\sum_{i\in A}I(U_i;\boldsymbol v).
\tag{4-2}
$$

**命题 13。**

$$
\Xi(A)=\mathbb E_{q(\boldsymbol v)}
D_{\mathrm{KL}}\!\left(q(\boldsymbol{u}_A\mid\boldsymbol v)
\middle\|\prod_{i\in A}q(U_i\mid\boldsymbol v)\right),\quad
0\le\Xi(A)\le I(\boldsymbol{u}_A;\boldsymbol v).
\tag{4-3}
$$

**证明。** 源独立使 $H(\boldsymbol{u}_A)=\sum_iH(U_i)$。展开 MI 后无条件熵抵消，剩下 $\sum_iH(U_i\mid\boldsymbol v)-H(\boldsymbol{u}_A\mid\boldsymbol v)$，即条件总相关的 KL。KL 非负给下界，单源 MI 非负给上界。证毕。

$|A|=2$ 时是原协同；更多源时 $\Xi$ 不自动为纯阶原子。例如前两源 XOR、第三源无关时，三源 $\Xi$ 仍 1。

不交源块 $\boldsymbol{u}_L,\boldsymbol{u}_R$ 独立，命题 3 直接给出

$$
\begin{aligned}
s(L,R;\boldsymbol v)
&=I(\boldsymbol{u}_{L\cup R};\boldsymbol v)-I(\boldsymbol{u}_L;\boldsymbol v)-I(\boldsymbol{u}_R;\boldsymbol v)\\
&=I(\boldsymbol{u}_L;\boldsymbol{u}_R\mid\boldsymbol v)\ge0,\\
\Xi(L\cup R)&=\Xi(L)+\Xi(R)+s(L,R;\boldsymbol v).
\end{aligned}
\tag{4-4}
$$

这就是 $2\to1$ 在两块上的直接延拓，未另造多源冗余。

### 4.2 N→1：原 SPT 的贪婪源树

各非单例节点选

$$
(L_A^\star,R_A^\star)\in\arg\max_{L\dot\cup R=A}[\Xi(L)+\Xi(R)].
\tag{4-5}
$$

父 $\Xi(A)$ 固定，最大保留子块集成信息等价于最小源块协同。递归到单源叶，目标与跨度固定。

**命题 14。** 任意完整二叉源树 $\mathcal T_U$ 满足

$$
I(\boldsymbol{u};\boldsymbol v)
=\sum_i I(U_i;\boldsymbol v)
+\sum_{v\in\operatorname{Int}(\mathcal T_U)}s(L_v,R_v;\boldsymbol v),
\quad \Xi(\{1,\ldots,n\})=\sum_v s(L_v,R_v;\boldsymbol v).
\tag{4-6}
$$

**证明。** 逐节点使用式（4-4）。非根子块互信息在上层作为子项，在自身展开中被等量取代；最终只剩叶互信息和节点 $s$。等价地，逐节点 $\Xi$ 相加使内部非根项一正一负抵消，单例 $\Xi=0$。每项非负。证毕。

得到 $n$ 个单源父项、$n-1$ 个协同父项。贪婪只保证节点候选域内最优，不保证整树全局最优；节点阶数是路径标签，不是 Möbius 纯阶原子。

**三源 XOR 例：** $U_1,U_2,U_3$ 为独立公平比特，$V=U_1\oplus U_2\oplus U_3$。单／双源 EI 零，联合 EI 1。任一根二分的协同 1，子树协同零；相比二源 XOR，只是一侧变成两个源的块。

### 4.3 1→N：两块四项与条件目标细化

本节仍用 $W$ 表示单源；在 $N\to N$ 的第 $i$ 个单源行中，$W=U_i$。单源目标树适用于任意有限联合分布。将目标块 $G,H$ 当作两个有限变量，直接复用 $1\to2$：

$$
\begin{aligned}
R&:=C_{\mathrm{ext}}(\boldsymbol v_G,\boldsymbol v_H;W),\\
\mathrm{Un}_G&:=I(W;\boldsymbol v_G)-R,\quad
\mathrm{Un}_H:=I(W;\boldsymbol v_H)-R,\\
S&:=I(W;\boldsymbol v_{GH})
-I(W;\boldsymbol v_G)-I(W;\boldsymbol v_H)+R.
\end{aligned}
\tag{4-7}
$$

命题 1—2 直接保证四项非负和两块闭合。可在此停止，保留清楚的块标签；不同二分不能全部相加为一张原子表。

继续细化时，四项标量不能各自充当新随机源。保持条件 MI 预算，$\boldsymbol c$ 为已读目标，$J$ 与之不交：

$$
I(W;\boldsymbol v_J\mid\boldsymbol c)
=I(W;\boldsymbol v_G\mid\boldsymbol c)
+I(W;\boldsymbol v_H\mid\boldsymbol c,\boldsymbol v_G),
\qquad J=G\dot\cup H.
\tag{4-8}
$$

两子预算非负。单目标终端直接输出；二目标或选定二块终端，在每个 $q(\cdot\mid\boldsymbol c)$ 中用 $C_{\mathrm{ext}}(\boldsymbol v_G,\boldsymbol v_H;W)$，始终只处理本行源 $W$，再按 $q(\boldsymbol c)$ 平均四项。处理通道允许随上下文变化。$1\to2$ 根终端精确恢复第 1.3 节定义。

**命题 15。** 条件二块 MI 四项非负，和为 $I(W;\boldsymbol v_{GH}\mid\boldsymbol c)$。

**证明。** 命题 1—2 适用于每个正概率条件分布，无需源独立；各态非负闭合，概率平均后仍成立。证毕。

这是 MI 链式法则，不是完整冗余 LC。条件预算始终由原联合分布产生；在 EI 应用中，也不对各条件节点重新干预。

**三输出秘密共享：** 独立公平 $W,N_1,N_2$，$V_1=N_1,V_2=N_2,V_3=W\oplus N_1\oplus N_2$。单／双输出 EI 零，完整 EI 1。在 $\{1,2\}\mid\{3\}$ 停止，联合余额 1；继续链式读取得到 $I(W;V_3\mid V_1,V_2)=1$。它必须带上下文，不能叫 $V_3$ 特有，有效作用域是三个目标并集。

**缺少部分共享的反例：** $W$ 为公平比特，$(V_1,V_2,V_3)=(W,W,0)$。若只设一项全体冗余、三个特有、一项协同，第三输出使全体冗余零，前两项各 1 而联合 EI 仅 1。需要块级共享或路径归属，不能机械扩大二元公式。

### 4.4 N→N：逐行固定 KL，再二分目标

固定源树。单源行用第 4.3 节；协同节点 $v$ 令 $\boldsymbol u_v=(\boldsymbol{u}_{L_v},\boldsymbol{u}_{R_v})$，直接将式（3-3）的标量源替换为独立块。普通小写 $v$ 仅为源树节点索引，粗体 $\boldsymbol v$ 为目标向量。

$$
P_v=q(\boldsymbol u_v,\boldsymbol v),\quad
Q_v=q(\boldsymbol v)q(\boldsymbol{u}_{L_v}\mid\boldsymbol v)
q(\boldsymbol{u}_{R_v}\mid\boldsymbol v),\quad
s_v=D_{\mathrm{KL}}(P_v\|Q_v).
\tag{4-9}
$$

每节点有自己的固定参照，第 3 节命题 9—12 逐行适用。根只抽一次 $b_v=D_{\mathrm{KL}}(P_{v,\boldsymbol u_v}\|Q_{v,\boldsymbol u_v})$，之后保持

$$
F_{v,\boldsymbol c}(J):=\mathbb E_{P_v(\boldsymbol c)}
D_{\mathrm{KL}}\!\left(P_v(\boldsymbol v_J\mid\boldsymbol c)
\middle\|Q_v(\boldsymbol v_J\mid\boldsymbol c)\right),
\qquad \boldsymbol c_{\mathrm{root}}=\boldsymbol u_v.
\tag{4-10}
$$

**命题 16。** 有序二分 $J=G\dot\cup H$ 满足

$$
\boxed{F_{v,\boldsymbol c}(J)
=F_{v,\boldsymbol c}(G)+F_{v,(\boldsymbol c,\boldsymbol v_G)}(H).}
\tag{4-11}
$$

两子非负。$F_{v,\boldsymbol c}$ 是第 3 节读出 $F$ 加上源节点与上下文索引；在该行根上下文中，它就是式（3-5）的读出。以 $F_{v,\boldsymbol c}$ 替换第 3.5 节 $F$，同核局部处理的二块四项也非负闭合。

**证明。** $P_v\ll Q_v$ 保证所有 $P_v$ 正权上下文有条件参照。将条件联合概率写为前块条件概率乘后块条件概率，$P_v,Q_v$ 分别展开；对数比值拆成两项，按 $P_v$ 平均得到等式。条件 KL 非负；逐上下文的数据处理再平均，重复命题 11—12 的证明即得终端四项。证毕。

子节点不再抽新基线，全部条件分布来自本行根 $P_v,Q_v$。MI 终端只处理单源，用 $C_{\mathrm{ext}}$；KL 终端用 $r_F$。两类规则分别对应已有 MI 信息量和固定 KL 残余，均由各自的上下界保证非负。

### 4.5 目标二分的贪婪目标及算法

![从源 SPT 到逐行目标树：条件 MI、固定参照 KL 与终端汇总](../../fig/ei_decomposition_recursive_flow.png)

**图 3　二元规则如何延拓为 N→N。** 在源相互独立的联合分布下，源树把联合互信息分成非负单源行与协同行。MI 行通过条件 MI 链递归，二块终端只处理本行源；KL 行先保留一次基线，随后固定参照作条件 KL 递归，二块终端用共同读出。内部四项用于择路，最终仅汇总终端与基线。$N\to1$ 保留源树父项；单源且二目标时直接恢复原 $1\to2$。[SVG](../../fig/ei_decomposition_recursive_flow.svg)

KL 节点评估各二分

$$
\Delta_{v,\boldsymbol c}(G,H):=F_{v,\boldsymbol c}(J)
-F_{v,\boldsymbol c}(G)-F_{v,\boldsymbol c}(H)
+r_{F,\mathrm{lo}}(G,H\mid\boldsymbol c),\quad
(G^\star,H^\star)\in\arg\min_{G\dot\cup H=J}\Delta_{v,\boldsymbol c}(G,H).
\tag{4-12}
$$

$r_{F,\mathrm{lo}}$ 为第 5.1 节有限候选版，有精确优化时用 $r_F$。MI 行用条件 MI 和逐条件分布的单端 $C_{\mathrm{ext}}$／可行下界替换，局部处理端固定为本行源。该准则选择联合余额最小的候选划分，与源 SPT 择路动机一致；随后先读边缘预算较大块，平局按索引或预先声明的领域规则。它是默认择路规则；非负和闭合由局部分配与链式等式保证，任何可行划分都可使用。

**算法：先分源，再逐行分目标。**

1. 给定联合分布，源侧复用 SPT 时取相互独立的源。固定概率／估计协议、候选策略、终止粒度和容差；计算 EI 时另固定共同干预、支持与时间跨度。
2. 按式（4-5）建立源树，得到单源行及协同行。
3. 目标仅一个时直接报告源树原父项，恢复 $N\to1$。
4. 单源行以原 $q$ 开始 MI 目标树；协同行固定式（4-9）、保留一次 $b_v$，以 $F_{v,\boldsymbol u_v}$ 开始 KL 树。
5. 单目标输出预算；二目标以对应二元规则输出四项并停止。也可在选定二块节点输出四项提前停止。
6. 还需细化的节点按式（4-12）选划分，**真正子预算**用式（4-8）或式（4-11），递归更新上下文。
7. 内部四项若仅择路，不再加入总和；不能先输出它们，再无参照地递归标量。
8. 最后只汇总终端和各根基线，核对联合互信息闭合。保存每项的源节点、目标块、已读目标上下文、参照及候选信息。

从 $2\to2$ 到 $N\to N$，源块二分复用独立源二元定义，MI 行复用 $1\to2$，KL 行复用固定残余提取与条件链；更多变量只增加树节点、块标签及上下文。

### 4.6 整体非负与闭合定理

**定理 1。** 在源相互独立的有限离散联合分布、精确概率、固定参照及可行候选下，任意完整源二叉树（叶为单源）、目标择路及上述目标提前停止均输出非负项，总和为 $I(\boldsymbol{u};\boldsymbol v)$。采用当前 PEID 的共同干预分布时，这一总量即联合 EI。

**证明。** 命题 14 给非负源父项及总和。KL 行由命题 10 分成非负基线和根条件 KL。各非终端父预算由 MI／KL 链式等式精确取代为两非负子预算。终端预算本身非负，二块终端由命题 15／16 精确取代并保持非负。对有限树归纳，终端加基线和等于原联合互信息；内部择路诊断不进入总和。证毕。

非负不依赖找到最优划分，搜索影响路径而不影响可行证书。未证明内容互斥、树唯一、换侧不变或恢复所有子目标原 PEID。

两个目标时共 $4n+5(n-1)$ 个位置；细化到单目标／二目标终端时，每行最多 $2m$ 个终端项，协同行另有一项基线，表示规模为 $O(nm)$。联合状态、条件上下文和完整二分枚举仍可能指数增长；这不是多项式运行时间结论。

### 4.7 一个 3→3 的完整路径例子

独立公平源取

$$
V_1=U_1\oplus U_2,\qquad V_2=U_1\oplus U_2,\qquad V_3=U_3.
\tag{4-13}
$$

总 EI 为 2，单源父项 $(0,0,1)$，全体 $\Xi=1$。源根选 $\{1,2\}\mid\{3\}$，子块 $\Xi$ 为 $1,0$，和达最大值；根协同 $2-1-1=0$，子节点 $\{1,2\}$ 协同为 1。

为展示块级解释，目标取合法路径 $\{1,2\}\mid\{3\}$，也达到此例联合余额最小值 0：

- $U_3$ 行预算 1，先读 $\{3\}$ 得到 1，其余条件预算零，保留 $U_3\to V_3$ 贡献。
- 源节点 $\{1,2\}$ 协同行预算 1、$b_v=0$。先读目标 $\{1,2\}$ 得到 1，目标 $\{3\}$ 条件预算零；二目标终端按第 3.7 节例 1 得共同读出 1，其余三项零。
- 根协同行零，细分全零。

最后非零项为“单源 $U_3$ 对 $V_3$ 贡献 1”和“源块 $\{1,2\}$ 协同被 $V_1,V_2$ 共同读取 1”，总和 2。其他平局路径可改变标签，但父预算和总量保持。

## 5. 计算、数值有效性及连续接口

### 5.1 有限候选提供可算的非负粗分解

二元计算只评估指定 $W\to Z$ 通道，把常量和恒等纳入候选集，定义

$$
R_{\mathrm{lo}}:=\max\{0,\operatorname{CoI}(U,V,W),
\text{已评估 }Z\leftarrow W\text{ 的共信息值}\},\quad
R_{\mathrm{hi}}:=\min\{I(U;V),I(U;W),I(V;W)\}.
\tag{5-1}
$$

$R_{\mathrm{lo}}\le C_{\mathrm{ext}}(U,V;W)\le R_{\mathrm{hi}}$；用下界诱导的四项仍非负闭合，低估冗余及协同、高估特有，不能标作精确最优。

KL 行定义

$$
r_{F,\mathrm{lo}}:=\max\{0,F(G)+F(H)-F(GH),
c_G(\kappa_1),\ldots,c_H(\lambda_k)\}.
\tag{5-2}
$$

每候选受命题 11 上界约束，代入式（3-10）仍非负闭合。精确值满足 $r_{F,\mathrm{lo}}\le r_F\le\min(F(G),F(H))$，可报告区间给出的差距上界，未知精确差距应标未知。记录候选集、辅助状态数及策略。

最大值的零来自可行常量通道，不是裁剪估计 Syn。两个混合例子显示：只用常量／恒等虽保证非负，仍可能遗漏可提取共有。

### 5.2 搜索、成本与数值失败规则

$m$ 元素有 $2^{m-1}-1$ 个非空无序二分。小节点可枚举，大节点限制领域、谱排序或其他候选，只保证候选域最优。当前稿源 SPT 规则见第 6.4 节，不自动规定目标树最佳候选。

概率精确时保证来自定理；估计时分别记录密度／MI／KL 误差和优化误差：

- 声明原生单位非负容差 $\tau\ge0$。
- 记录 $[-\tau,0)$ 数量，可当数值零，不作负协同证据。
- 任一消耗的 Syn／非负预算低于 $-\tau$ 时显式失败，报告最小值、阈值和数量。
- 估计界、包含关系或闭合不成立时报告参照／估计不一致，不静默投影。

基线只抽一次、同核推送固定参照、内部择路四项不重复汇总是结构条件，不能靠容差修复。

### 5.3 已有有限概率表核对记录

以下保留前期核对记录，并记录本轮图示的精确复核。当前二元表的精确值由第 2 节指定端的解析证明支持。

本轮重新核对图 1—2 的 XOR、复制与随机掩码三个精确有理概率表，只计算当前指定端的可行通道，并验证达到理论冗余上界及四项闭合。非负容差为 0 bit，微小负值和显著负值均为 0，调整数为 0；没有进行随机通道数值优化或新增大规模实验。

旧版二元核对包含九个机制和两个独立源门，穷举单端确定性分组并检查六种变量置换。容差 $10^{-12}$ bit，11 个机制中 1 个原子处于 $[-10^{-12},0)$，来自含噪共有加秘密共享的零特有项；显著负值 0，最大置换误差 $4.44\times10^{-16}$ bit，没有裁剪或全局随机通道数值优化。这是旧版三处理端最大值的历史记录，不作为当前定向冗余的三位置对称证据。现有表例的最优处理恰好作用于当前指定端，或上下界相等，故本轮解析核对保留其数值。

KL 核对枚举 $4^4=256$ 个确定性二输入二输出机制，检查“基线＋四项”粗分解与“基线＋两项链式读出”，共 2,048 个值。容差 $10^{-12}$ bit，微小负值 0、显著负值 0、最小值 0、最大加和误差 0。粗四项只用常量／恒等；第 3.7 节前三例精确值另由显式核和上界确定。普遍结论来自证明，非有限枚举。

### 5.4 连续 EI 与尚待研究问题

有限状态的辅助字母表、可达性和连续性不能直接搬到连续变量。需指定支持／矩约束、干预密度、有限 MI／KL、局部核及优化紧性，连续 EI 仍优先 TM。

新 KL 树需共同参照密度及一致边缘／条件化；各目标另拟合后验乘积不能替代固定 $Q_V$ 推送。仿射高斯代理与精确理论不同，具体反例见附录 B；现有估计代码尚未实现本文目标树。

**正的精确集成信息不证明机制具有不可加交互。** 令 $U_1,U_2$ 独立单位方差高斯，$W=U_1+U_2+\varepsilon$，$\varepsilon$ 独立高斯、方差 $\sigma^2>0$。精确联合读出增益为

$$
\Xi=\tfrac12\log_2\frac{(1+\sigma^2)^2}{\sigma^2(2+\sigma^2)}>0.
\tag{5-3}
$$

$\sigma^2=1$ 时为 $0.207519\ldots$ bit，机制仍是线性可加。该例说明信息量的解释边界，不把有限状态树的可达性、连续性或计算保证延拓到连续空间。

仍待研究：$C_{\mathrm{ext}}$ 在未处理两端上的局部处理性质、扩大观测的变化、独立模块可加；KL 冗余的状态界、可达、连续及噪声性质；路径敏感性、基线解释、更细残余表示。基准可扩展到相关变量、非对称噪声、低概率、支持变化及独立模块。对照应符合各自操作语义，不以本指标作唯一真值。

## 6. 与已有工作的关系及当前稿件依据

### 6.1 直接采用已有量，说明分解用途

式（1-2）直接采用已有单端可提取共信息。Rauh 等已给出局部随机提取及其与内禀条件互信息的关系。[25](#ref-25) 第 1 节说明这个已有量如何满足当前 EI 分配所需的性质。

本文组织的内容是：在任意有限联合分布下，用单端冗余统一写出 $2\to1$ 与 $1\to2$ 四项收支；独立源侧复用当前 PEID／SPT；对源协同固定一份 KL 参照，再按条件 MI／KL 链式法则逐步细分。固定参照保证子预算属于同一父项，条件链保证递归不重复计账。非负与精确总和由这些结构保证，贪婪搜索只决定路径。

Pica 等关于不同目标选择与角色不变成分的工作保留为背景阅读。[26](#ref-26) 当前框架固定单端角色，目标是得到可计算的路径粗分解。下面的比较用于解释信息分配的具体取舍，而不以与既有方法不同作为设计要求。

### 6.2 常用定义在具体机制中的分配差异

以下比较固定联合分布、变量角色和 bit 单位，展示不同共有信息语义如何影响分配。机制及计算细节见第 2 节和附录 A。

| 已有函数／构造 | 机制 | 已有冗余 | $C_{\mathrm{ext}}(U,V;W)$ | 分配差异 |
|---|---|---:|---:|---|
| MMI：$\min(I(U;W),I(V;W))$ | 独立公平源，$W$ 为两源一一编码 | 1 | 0 | 不满足独立源零冗余及该复制恒等性 |
| 最小耦合／BROJA | 独立公平源 AND，$W=U\land V$ | 0.311278 | 0 | 非负，但与所选 PEID 分配不同 |
| RR | 第 2.3 节共有加秘密共享 | 0.5 | 1 | 标量插值削弱完整共有内容 |
| $I_{\mathrm{CCS}}$ | 独立公平源 AND | 0.103759 | 0 | 不普遍满足独立源零冗余 |
| 共同确定性 $I_{\wedge}$ | 精确副本与 0.1 带噪副本 | 0 | 0.531004 | 按四项收支产生负协同 |
| 双路径通道公式直接代入 | 同一副本例，纳入两个次序 | 0.319923 | 0.531004 | 串接噪声使冗余低于非负下界 |

MMI 见 [9](#ref-9)，BROJA 见 [4](#ref-4)，RR 见 [19](#ref-19)、[22](#ref-22)，CCS 采用 Ince 保持三对边缘的最大熵参照版本，见 [24](#ref-24)，共同确定性见 [20](#ref-20)，路径构造见 [23](#ref-23)。计算与适用假设见附录 A；不能混用不同 CCS 版本或路径裁选规则。

### 6.3 保留的性质与设计取舍

| 要求 | 本定义 | RR | MMI | 所核对的 CCS 版本 |
|---|---|---|---|---|
| 任意有限离散分布上有定义 | 是；涉及全局随机优化 | 是；闭式标量插值 | 是；直接取最小 | 是；涉及最大熵参照 |
| 源独立时零冗余 | 保证 | 保证 | 不保证 | 不保证 |
| 二元四项非负 | 保证 | 保证 | 保证 | 不保证；可有负特有 |
| 普通复制恒等性 | 保证 | 一般不成立 | 一般不成立 | 不满足；仅满足独立源复制的较弱恒等性 |
| 需要分配的两个变量交换对称 | 保证；处理端固定 | 保证 | 保证 | 保证；固定目标角色 |
| 输出附加共同无关噪声分量不改变分配 | 保证 | 有反例 | 该类附加操作下保留；仍有独立分流问题 | 本文不宣称一般结论 |
| 完整目标 LC | 不满足 | 不满足 | 不满足 | 不满足；独立 XOR 条件化已有反例 |
| 只依赖两条源—目标边缘 | 不要求；使用实际联合结构 | 否；还用源依赖 | 是 | 否；参照保留第三对边缘 |
| Blackwell 等价边缘通道的特有必为零 | 不要求，已有反例 | 不保证 | 同 MI 时为零，非完整 Blackwell 表征 | 不作为本文已证明的性质 |

选择 $C_{\mathrm{ext}}$ 的理由是它同时给出非负四项、独立源 PEID 退化、复制恒等性及附加无关噪声不变，并能区分共有与共同噪声。代价是需要局部通道优化，且不满足完整冗余 LC。它使用实际联合结构，不要求 Blackwell 等价边缘通道的特有为零。

MMI 对联合高斯、标量目标且冗余／特有仅依赖两条边缘的类别有特定理论支持，[9](#ref-9) 不能用一个离散反例否定该条件结论。CCS 区分逐状态共同信息变化，有自己的内容动机；负特有反例说明它不适合本研究要求的全分布非负四项分配，见附录 A.3。

其他路线仍有各自用途。本文选择下列基本构件，是为了满足明确的收支要求，而不累加额外公理。

| 路线 | 保留用途 | 未作本文主定义的原因 |
|---|---|---|
| 最小耦合／BROJA | 固定单通道能力、非负分配 | 独立 AND 仍有正冗余，不恢复所选 PEID |
| 两侧分别用 PEID／BROJA | 允许两侧不同规则的备选 | 本文直接复用同一单端提取量，减少独立定义 |
| RR | 独立源零冗余、简单 | 无法区分共有与共同无关噪声，私有噪声改变权重 |
| MMI | 特定高斯对照 | 不满足独立分流及复制恒等性 |
| CCS | 逐状态共同信息变化 | 独立 AND 冗余非零，另有负特有 |
| 共同确定性 | 无误共有内容 | 小噪声跳变，收支可能负协同 |
| 双路径公式直接代入 | 指定路径参照 | 噪声串接低于非负下界，需额外路径假设 |
| 原共信息正部 | 简单可行粗分配 | 共有被协同抵消，遗漏混合机制 |

Blackwell 共同退化量采用决策论语义，[5](#ref-5) 与这里的单端共信息提取有不同定义。旧最小耦合多目标汇总保留在附录 A.5。二元信息量直接沿用已有工作；本文需说明的是如何在固定父预算下组织一条非负递归路径。

### 6.4 当前可用稿件、版本歧义与实现差异

本轮重新通过 Zotero 核对 *Emergent hierarchical organization of causal interactions in complex systems*，父项 P6UJCVG8；重新列附件并读主文 DXGC7JEA（19/19 页索引全文）、补充 MWIWKSVG（28/28 页索引全文）相关部分。

主附件导入 2026-10-02、元数据修改 2026-10-04；补附件导入 2026-10-04。无明确稿件日期／版本号，元数据不足以判断版本更新先后，本文依据当前可用全文并保留歧义。

当前主稿的 EI 协议仍是共同因子化最大熵干预。本文先在任意有限联合分布下说明二元信息分配，再把独立源的源树证明写到其所需的一般性；非均匀独立源也满足这些信息恒等式。采用自然观测分布时，总量称互信息；本文没有修改主稿的干预 EI 定义。

| 全文位置 | 与本文关系 |
|---|---|
| 主文 Methods 第 15—17 页，式（5）—（12） | 共同干预、二源 PEID、$\Xi$、SPT 目标及闭合 |
| 补充 S2 第 5 页，式（S20）—（S27） | 对称公理仅要求源索引置换；另含恒等性、完整 LC、逐条件平均；本文采用单端定义并不要求完整 LC |
| 补充 S3.1 第 6 页，式（S29）—（S36） | 原公理桥接 PEID／PID；第 1.11 节反例限制普遍适用表述 |
| 补充 S3.2 第 7 页，式（S37）—（S45） | 条件总相关 KL 非负证明，对应命题 3、13 |
| 补充 S3.3 第 8 页，式（S46）—（S48） | 源分区加和，对应命题 14 |
| 补充 S5 第 8—9 页，式（S49）—（S51）、Algorithm S1 | 小节点精确、大节点谱／平均连接／原序候选，全树贪婪 |
| 补充 S1.2 第 3—4 页，式（S7）—（S15） | 连续约束及仿射代理，不自动继承精确非负 |

本文未修改当前 PDF；现有估计代码／notebook 尚未实现整套目标残余树。只对齐文档定义与推导，不声称实现全面一致。历史预印本 [1](#ref-1) 为补充，不替代当前主稿。

## 附录 A. 既有方案的必要定义与反例

### A.1 最小耦合：非负成立，独立源退化失败

对两变量一侧 $U,V$ 和单变量 $W$，定义

$$
\Delta(P):=\{Q:Q_{UW}=P_{UW},\ Q_{VW}=P_{VW}\},\qquad
J_{\mathrm{MC}}:=\min_{Q\in\Delta(P)}I_Q(U,V;W).
\tag{A-1}
$$

该量来源于 Griffith–Koch union information 及二源 BROJA。[3](#ref-3)、[4](#ref-4) 数据处理给出 $J_{\mathrm{MC}}\ge\max(I(U;W),I(V;W))$，原 $P$ 可行给出 $J_{\mathrm{MC}}\le I(U,V;W)$，条件独立参照 $Q_0(\bar U,\bar V,\bar w)=P(\bar w)P(\bar U\mid\bar w)P(\bar V\mid\bar w)$ 又给出 $J_{\mathrm{MC}}\le I(U;W)+I(V;W)$。因此

$$
R_{\mathrm{MC}}:=I(U;W)+I(V;W)-J_{\mathrm{MC}},\quad
\mathrm{Un}_U^{\mathrm{MC}}:=J_{\mathrm{MC}}-I(V;W),\quad
\mathrm{Un}_V^{\mathrm{MC}}:=J_{\mathrm{MC}}-I(U;W),\quad
S_{\mathrm{MC}}:=I(U,V;W)-J_{\mathrm{MC}}
\tag{A-2}
$$

四项非负。源独立时，PEID 与之的准确关系是

$$
S_{\mathrm{PEID}}=I(U,V;W)-I(U;W)-I(V;W)=S_{\mathrm{MC}}-R_{\mathrm{MC}}.
\tag{A-3}
$$

独立公平源 AND：$I(U;W)=I(V;W)=h_2(1/4)-1/2=0.311278\ldots$、$I(U,V;W)=h_2(1/4)=0.811278\ldots$。固定两条边缘允许 $Q$ 令两源完全相同，达到 $J_{\mathrm{MC}}=I(U;W)$，因此 $R_{\mathrm{MC}}=I(U;W)$、两项特有为 0、$S_{\mathrm{MC}}=1/2$；PEID 是 $R=0$、$\mathrm{Un}_U=\mathrm{Un}_V=I(U;W)$、$S=0.188722\ldots$。

即使强迫候选源仍独立，保留 AND 的两条边缘会迫使回到原分布，$J=I(U,V;W)$；若还套用式（A-2），冗余反而成为 $I(U;W)+I(V;W)-I(U,V;W)=-0.188722\ldots$。单加源独立约束不能修复原四项公式。依 PEID 原则可称其正冗余不符合所需分配，不能无条件宣布其通道操作语义“高估错误”。

### A.2 RR 与 MMI

Goodwell–Kumar RR 在有限离散二源情形取

$$
R_-:=\max(0,\operatorname{CoI}(U,V,W)),\quad R_+:=\min(I(U;W),I(V;W)),\quad
\alpha:=\frac{I(U;V)}{\min\{H(U),H(V)\}},\qquad
R_{\mathrm{RR}}:=R_-+\alpha(R_+-R_-).
\tag{A-4}
$$

分母为 0 时明确定义 $R_{\mathrm{RR}}=0$。$0\le\alpha\le1$ 保证四项非负；源独立使 $\operatorname{CoI}(U,V,W)\le0$、$\alpha=0$，故 $R=0$。这已经是发表过的独立源零冗余前例。[19](#ref-19)、[22](#ref-22)

但式（2-3）两例均有 $\alpha=1/2,R_-=0,R_+=1$，RR 都给出 $R=1/2$。在共有加秘密共享例中给每个输出再追加一个独立私有公平噪声后，单 EI、联合 EI及两输出 MI 不变，输出熵变为 3，RR 降为 $1/3$。其权重是依赖强度而非充分的内容判据。

MMI 取 $R_{\mathrm{MMI}}=\min(I(U;W),I(V;W))$，始终在式（1-12）区间内，四项非负。但独立复制目标有 $I(U;W)=I(V;W)=1,I(U,V;W)=2$，它给出 $R=S=1,\mathrm{Un}_U=\mathrm{Un}_V=0$，与所需独立分流不同；复制恒等性要求 $R=I(U;V)=0$。MMI 的特定高斯结论见 [9](#ref-9)，不作全类别推广。

### A.3 CCS 的版本与两个关键例子

这里的 $I_{\mathrm{CCS}}$ 按 Ince 第 4.2 节式（30）—（32）定义：构造保持三对边缘 $P_{UV},P_{UW},P_{VW}$ 的最大熵参照分布，再按逐状态信息变化的符号一致条件保留共信息项。[24](#ref-24) 不使用只保持两条目标边缘的另一变体替代它。

独立公平 AND 的这些边缘唯一确定原概率表。该定义给出

$$
R_{\mathrm{CCS}}=\tfrac14\log_2(4/3)=0.103759\ldots,\quad
\mathrm{Un}_U=\mathrm{Un}_V=0.207519\ldots,\quad S=0.292481\ldots.
\tag{A-5}
$$

这不符合独立源零冗余。但它在完整共享 $C$ 加秘密共享 $T$ 中可以识别 1 bit 共有内容，不能只凭独立 AND 就否定它所有内容解释。

完整复制恒等性也不成立：目标无损复制源对时，三个局部目标 MI 都是非负惊讶度，CCS 的符号规则只保留源间正局部 MI；存在负局部 MI 时，结果一般不等于普通 $I(U;V)$。原文第 4.4 节明确区分完整恒等性与独立源复制的较弱恒等性。完整 LC 则由独立 XOR、剩余目标 $U$ 的条件化给出 $0=0+1$，与第 1.11 节同型。

原文 Table 7 的概率 $P(0,0,0)=0.4$、$P(0,1,0)=0.1$、$P(1,1,1)=0.5$，变量依次为 $U,V,W$，有 $W=U$。$I(U;W)=I(U,V;W)=1,I(V;W)=0.6099865\ldots$，CCS 给出 $R=0.7684828\ldots$，因此 $\mathrm{Un}_V=-0.1584963\ldots$。本定义由上下界相等给出 $R=I(V;W),\mathrm{Un}_V=S=0$。这里是候选 PID 原子的负值，不能解释成允许负的 PEID Syn，也不能截断后宣称保持收支。

### A.4 共同确定性与路径参照的带噪副本反例

共同确定性冗余取 $R_{\wedge}=I(K;W)$，$K$ 是两预测变量均可无误恢复的最大共同变量。[20](#ref-20) 对 $W=C,U=C,V=C\oplus F$，$0<\varepsilon<1/2$，两输出支持图连通，所以 $K$ 为常量，$R_{\wedge}=0$。但 $I(W;U)=I(W;U,V)=1,I(W;V)=1-h_2(\varepsilon)$，按四项收支得到 $S_{\wedge}=-I(W;V)<0$。在 $\varepsilon=0$ 时冗余为 1，任意正噪声时降为 0，定义不连续。确定性重叠 $U=U_0,V=U_0\land V_0$ 也有同类负协同，并非仅噪声模型的问题。

双路径公式直接代入是明确的参照构造：

$$
\begin{aligned}
Q_{12}(u,v,w)&:=P(u)P(v\mid u)P(w\mid v),\\
Q_{21}(v,u,w)&:=P(v)P(u\mid v)P(w\mid u),\\
R_{\mathrm{path}}&:=\min\{I_{Q_{12}}(U;W),I_{Q_{21}}(V;W)\}.
\end{aligned}
\tag{A-6}
$$

互信息按诱导分布计算。副本例中的较差信道是两次 BSC 串接，交叉率 $\delta=2\varepsilon(1-\varepsilon)>\varepsilon$，故 $R_{\mathrm{path}}=1-h_2(\delta)<I(W;V)$、$S=h_2(\varepsilon)-h_2(\delta)<0$。$\varepsilon=0.1$ 时冗余为 $d$、协同为 $-0.211081\ldots$。

Sigtermans 原文还涉及图、路径存在性及 Markov 等解释条件。[23](#ref-23) 此反例检验的是纳入两个次序的直接通道公式，不宣称覆盖任何额外裁边规则。共同确定性及路径量都能区分第 2.3 节的共有内容与共同噪声；作为参照有价值，但未满足本研究的全分布非负四项要求。

### A.5 旧最小耦合多目标报告为何不等于新延拓

对目标索引集合 $G$，固定所有 $P_{WV_i}$ 定义 $J_{\mathrm{MC}}(G)=\min_QI_Q(W;\boldsymbol{v}_G)$，有

$$
\max_{i\in G}I(W;V_i)\le J_{\mathrm{MC}}(G)\le
\min\left\{I(W;\boldsymbol v_G),\sum_{i\in G}I(W;V_i)\right\}.
\tag{A-7}
$$

于是 $D(G):=\sum_{i\in G}I(W;V_i)-J_{\mathrm{MC}}(G)$ 及 $S_{\mathrm{MC}}(G):=I(W;\boldsymbol v_G)-J_{\mathrm{MC}}(G)$ 非负，满足 $I(W;\boldsymbol v_G)=\sum_{i\in G}I(W;V_i)-D(G)+S_{\mathrm{MC}}(G)$。$D$ 是重复计数总量，不是一个全体共享原子。它还可与适当共有量组合成三项非负报告，但其中的剩余部分不等于各目标纯特有。

报告 $n$ 个单 EI 及每个大小至少为 2 的子集的 $D,S_{\mathrm{MC}}$，数量为 $2\cdot2^n-n-2$；这是旧方案的报告规模，不是新定义完整原子数。对子集取 Möbius 差分也不能自动保证非负：三个完整副本的每对 $D=1$、三者 $D=2$，三阶差分为 $2-3=-1$。

子集协同亦不必单调：前两个输出秘密共享源，第三个输出直接给出完整源，则前两者 $S_{\mathrm{MC}}=1$，加入第三个后 $J_{\mathrm{MC}}=I(W;V_1,V_2,V_3)=1$，$S_{\mathrm{MC}}=0$。这些是汇总结构的边界，不是负的精确 PEID Syn。

完整 PID 按反链晶格计数，ΦID 按双侧乘积晶格计数；报告少量子集／二分量可减少表示规模，却不等于消除了状态空间增长或解决完整分解。SURD、PED、O-information 等另有各自对象及缩减策略，[10](#ref-10)、[11](#ref-11)、[12](#ref-12)、[15](#ref-15) 不把所有缩减思路归为本研究首次。

## 附录 B. 理论非负与仿射估计代理的差别

当前补充 S1.2 在特征空间使用仿射高斯 TM 代理。将式（S12）改写为本文记号，单源特征为 $\boldsymbol{\phi}_s(u)=(u,u^2,u^3)^{\mathsf T}$，联合特征为 $\boldsymbol{\phi}_j(u_1,u_2)=(u_1,u_2,u_1u_2,u_1^2,u_2^2)^{\mathsf T}$，后者没有两个单源的三次项。保留原坐标保证精确 MI 不变，不保证不同特征空间的高斯代理相容。

保留前期解析反例：$U_1,U_2$ 独立均匀于 $[-1,1]$，$W=U_1^3+\varepsilon$，噪声独立高斯、方差 $\sigma^2>0$。精确 PEID Syn 为 0；单源仿射特征模型的预测残差方差为 $\sigma^2$，联合模型只能线性利用 $U_1$，残差方差为 $\sigma^2+4/175$。因此总体高斯代理的差值为

$$
\Delta_G=-\tfrac12\log\left(1+\frac{4}{175\sigma^2}\right)<0.
\tag{B-1}
$$

这里保留自然对数，单位为 nat；$\sigma^2=0.01$ 时为 $-\tfrac12\log(23/7)\simeq-0.594792$ nat。这个偏差在样本无限和正则趋零时仍存在，不能统一归为有限样本误差。至少要让联合特征覆盖单源特征，再验证所有代理 MI 的共同模型相容性；加回三次项仅修复这个例子。

前期有限代码核对还发现：`exp/TM/transport_map_density.py` 的入口使用多项式三角 TM、返回 bit 且 `bias_correction=0.0`；`yrd/__init__.py` 的末尾导入覆盖同名仿射函数，`exp/network_revival/effective_information.py` 保留上述特征但调用该多项式入口。这是当时所核对调用路径的记录，本次未重新追踪全部入口、重跑实验或修改实现。不能把式（B-1）当作所有现有实验的输出，也不能声称当前 PDF 与代码已经一致。

## 参考文献与证据范围

保留前期检索的参考资料，正文按单端提取、PEID／SPT、递归分解及性质对照使用。全文、摘要及元数据证据层级逐条标出。Zotero key 为本地条目标识，不是 BibTeX 引用键。外部索引及出版社访问存在未完成的检索路径；目前证据足以支持列出的定义、反例及来源关系，不足以支持“覆盖全部工作”或全球首次声明。

<a id="ref-1"></a>

**[1]** Yang, M., Wang, S., & Zhang, J. (2026). *Partial Effective Information Decomposition for Synergistic Causality*. arXiv:2605.03267，预印本。[论文](https://arxiv.org/abs/2605.03267)。Zotero：`MYATYWAJ`；证据：全文，重点为第 2 节、Discussion 及附录 A–B。用途：历史 PEID 背景；当前方法以第 6.4 节记录的现行主稿为准。

<a id="ref-2"></a>

**[2]** Williams, P. L., & Beer, R. D. (2010). *Nonnegative Decomposition of Multivariate Information*. arXiv:1004.2515。[论文](https://arxiv.org/abs/1004.2515)。Zotero：`K9ZE68ZB`；前期核对证据：摘要与元数据，晶格计数同时由文献 [6–7] 的全文交叉核对。用途：PID 及反链晶格基础。

<a id="ref-3"></a>

**[3]** Griffith, V., & Koch, C. (2014). *Quantifying Synergistic Mutual Information*. arXiv:1205.4265，2014 修订版。[论文](https://arxiv.org/abs/1205.4265)。Zotero：`PIVMHCH6`；证据：全文，重点为 union information 定义及附录。用途：最小耦合联合信息 $J_{\mathrm{MC}}$ 及其协同的来源；本文单端冗余量的来源见 [25]。

<a id="ref-4"></a>

**[4]** Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). *Quantifying Unique Information*. **Entropy, 16**(4), 2161–2183。[DOI](https://doi.org/10.3390/e16042161)。Zotero：`TWXUCUH9`；证据：全文，重点为固定边缘优化、Lemma 4–5。用途：二变量四原子、凸性与非负性。

<a id="ref-5"></a>

**[5]** Kolchinsky, A. (2022). *A Novel Approach to the Partial Information Decomposition*. **Entropy, 24**(3), 403。[DOI](https://doi.org/10.3390/e24030403)；[作者全文](https://artemyk.github.io/assets/pdf/papers/Kolchinsky_2022_Novel_Approach_to_the_PID.pdf)。本地未匹配到条目；证据：全文第 4 节、第 5.2–5.4 节及附录 G。用途：共同退化通道、Blackwell union 与最小耦合的等价关系。

<a id="ref-6"></a>

**[6]** Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D., Seth, A. K., & Barrett, A. B. (2025). *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition*. **Proceedings of the National Academy of Sciences, 122**(39), e2423297122。[DOI](https://doi.org/10.1073/pnas.2423297122)。Zotero：`26Q48H8Y`；证据：正文及补充材料全文。用途：一般乘积晶格、2→2 的 15-for-free、输入分布选择及 MMI 负原子说明。

<a id="ref-7"></a>

**[7]** Varley, T. F. (2023). *Decomposing past and future: Integrated information decomposition based on shared probability mass exclusions*. **PLOS ONE, 18**(3), e0282950。[DOI](https://doi.org/10.1371/journal.pone.0282950)。外部检索；证据：全文第 1.2、2.3 节及 Discussion。用途：多目标双冗余函数与完整 ΦID 的组合规模。

<a id="ref-8"></a>

**[8]** Harder, M., Salge, C., & Polani, D. (2013). *Bivariate measure of redundant information*. **Physical Review E, 87**, 012130。[DOI](https://doi.org/10.1103/PhysRevE.87.012130)。Zotero：`VPWBNEBV`；证据：全文。用途：几何冗余与内容区分的二变量对照。

<a id="ref-9"></a>

**[9]** Barrett, A. B. (2015). *Exploration of synergistic and redundant information sharing in static and dynamical Gaussian systems*. **Physical Review E, 91**, 052802。[DOI](https://doi.org/10.1103/PhysRevE.91.052802)；[arXiv](https://arxiv.org/abs/1411.2832)；[公开全文](https://arxiv.org/pdf/1411.2832)。前期本地检索未匹配到同题条目；证据更新：29 页公开全文，第 4.2 节（PDF 第 11—13 页）及式（33）。用途：MMI 定义、联合高斯标量目标且冗余／特有信息只依赖源—目标边缘时的唯一分解结论。

<a id="ref-10"></a>

**[10]** Martínez-Sánchez, Á., et al. (2024). *Decomposing causality into its synergistic, unique, and redundant components*. **Nature Communications**。[DOI](https://doi.org/10.1038/s41467-024-53373-4)；[arXiv](https://arxiv.org/abs/2405.12411)。Zotero：`K69ZCNZG`；证据：本地全文，重点为补充材料 S1.3–S1.4；DOI 由出版页核对。用途：SURD 的分配原则及组合计算。

<a id="ref-11"></a>

**[11]** Ince, R. A. A. (2017). *The Partial Entropy Decomposition: Decomposing multivariate entropy and mutual information via pointwise common surprisal*. arXiv:1702.01591。[论文](https://arxiv.org/abs/1702.01591)。Zotero：`3IYZCCSM`；前期核对证据：摘要与元数据。用途：与无指定目标的熵分解作对象层面的区别。

<a id="ref-12"></a>

**[12]** Rosas, F. E., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019). *Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information*. **Physical Review E, 100**, 032305。[DOI](https://doi.org/10.1103/PhysRevE.100.032305)；[arXiv](https://arxiv.org/abs/1902.11239)。外部检索；证据：摘要与元数据，净平衡指标的定位另见文献 [15] 全文。用途：O-information 对照。

<a id="ref-13"></a>

**[13]** Jansma, A. (2025). *Decomposing Interventional Causality into Synergistic, Redundant, and Unique Components*. arXiv:2501.11447，预印本。[论文](https://arxiv.org/abs/2501.11447)。Zotero：`P6SS3GBX`；前期核对证据：摘要与元数据。用途：干预式分解的相关前例；不据此认定其效应量等价于 EI。

<a id="ref-14"></a>

**[14]** Matthias, P. H., Makkeh, A., Wibral, M., & Gutknecht, A. J. (2025). *Novel Inconsistency Results for Partial Information Decomposition*. arXiv:2512.16662，预印本。[论文](https://arxiv.org/abs/2512.16662)。Zotero：`8GG5JCLP`；证据：全文第 4–5 节。用途：非负性、链式法则及重编码不变性的公理取舍；其结论须在原文假设下解释。

<a id="ref-15"></a>

**[15]** Luppi, A. I., Rosas, F. E., Mediano, P. A. M., Menon, D. K., & Stamatakis, E. A. (2024). *Information decomposition and the informational architecture of the brain*. **Trends in Cognitive Sciences, 28**(4), 352–368。[DOI](https://doi.org/10.1016/j.tics.2023.11.005)。Zotero：`DPVZQMJZ`；证据：全文 Box 1。用途：主要方法谱系及既有缩减分解策略。

<a id="ref-16"></a>

**[16]** Galeano Muñoz, S. P., Bounoua, M., Franzese, G., Michiardi, P., & Filippone, M. (2026). *DIPHINE: Diffusion-based Φ-ID Neural Estimator*. arXiv:2606.18997，预印本。[论文](https://arxiv.org/abs/2606.18997)；[HTML 全文](https://arxiv.org/html/2606.18997v1)。外部检索；证据：全文第 2.3、4.4、5.2 节。用途：区分高维连续估计进展与逐变量完整晶格扩展。

<a id="ref-17"></a>

**[17]** Bertschinger, N., Rauh, J., Olbrich, E., & Jost, J. (2013). *Shared Information — New Insights and Problems in Decomposing Information in Complex Systems*. In *Proceedings of the European Conference on Complex Systems 2012*, pp. 251–269。[DOI](https://doi.org/10.1007/978-3-319-00395-5_35)；[作者预印本](https://arxiv.org/abs/1210.5902)。Zotero：`2GJNRG7M`；附件：`PSNHU4N7`；证据：20 页全文，第 4 节 $(LM)$、$(LC)$、$(Id2)$。用途：核对目标链式法则的来源、条件态定义及其与基础 PI 公理的区别。

<a id="ref-18"></a>

**[18]** Finn, C., & Lizier, J. T. (2018). *Pointwise Partial Information Decomposition Using the Specificity and Ambiguity Lattices*. **Entropy, 20**(4), 297。[DOI](https://doi.org/10.3390/e20040297)；[作者全文](https://arxiv.org/pdf/1801.09010)。Zotero：`9F9RKL4L`；附件：`5B2W6VTX`；证据：36 页期刊版全文，第 5.5 节 Theorem 6、Appendix B.3 式（A35）—（A37）。用途：二源目标链式法则、恒等性质与所有 PID 原子非负之间的已发表不相容性证明；注意无损重编码及条件量的约定。

<a id="ref-19"></a>

**[19]** Goodwell, A. E., & Kumar, P. (2017). *Temporal information partitioning: Characterizing synergy, uniqueness, and redundancy in interacting environmental variables*. **Water Resources Research, 53**(7), 5920–5942。[DOI 与全文](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016WR020216)。证据：出版页第 2.3—2.4 节，式（8）—（10）的数学图片另由 [22] 的完整公式交叉核对；首次上线 2017-06-27。用途：独立源零冗余、非负二源分解的已发表 RR 前例。

<a id="ref-20"></a>

**[20]** Griffith, V., Chong, E. K. P., James, R. G., Ellison, C. J., & Crutchfield, J. P. (2014). *Intersection Information based on Common Randomness*. **Entropy, 16**(4), 1985–2000。[DOI](https://doi.org/10.3390/e16041985)；[作者出版记录](https://csc.ucdavis.edu/~cmg/compmech/pubs/iifcr.htm)；[作者公开全文 v3](https://arxiv.org/pdf/1310.1538v3)。证据：第 4 节式（4）—（5）、第 5.1 节式（8）、第 5.3 节 ImperfectRdn；公开 v3 日期为 2015-06-10，期刊发表年为 2014。用途：共同确定性变量冗余及其相关源局限。

<a id="ref-21"></a>

**[21]** Liardi, A., Down, K. J. A., Blackburne, G., Neri, M., & Mediano, P. A. M. (2026). *The mathematical landscape of partial information decomposition: A comprehensive review of properties and measures*. arXiv:2603.06678v2（2026-06-01），预印本。[全文](https://arxiv.org/html/2603.06678v2)。证据：Appendix D.2 的 RR、common-information、causal-tensor 定义；E.2.2 Proposition 13 与 E.2.6 Proposition 25。用途：扩大主要 PID 定义的筛查范围并交叉核对独立源零冗余；不把综述的所有公理结论未经证明直接套用。

<a id="ref-22"></a>

**[22]** Dell’Oca, A., Guadagnini, A., & Riva, M. (2020). *Interpretation of multi-scale permeability data through an information theory perspective*. **Hydrology and Earth System Sciences, 24**, 3097–3109。[全文](https://hess.copernicus.org/articles/24/3097/2020/index.html)。证据：第 3.1 节式（8a）—（8b）；连续数据的 KDE 与离散概率计算见第 3.2 节。用途：交叉核对 Goodwell–Kumar RR 的完整公式和输入依赖权重。

<a id="ref-23"></a>

**[23]** Sigtermans, D. (2020). *A Path-Based Partial Information Decomposition*. **Entropy, 22**(9), 952。[DOI](https://doi.org/10.3390/e22090952)；[期刊全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7597237/)。证据：第 2 节假设、Definition 1—2、式（7）、（18）—（19）及 Discussion。用途：源独立时路径通道失去输入依赖的已发表冗余前例；保留原文解释假设和负原子限制。

<a id="ref-24"></a>

**[24]** Ince, R. A. A. (2017). *Measuring Multivariate Redundant Information with Pointwise Common Change in Surprisal*. **Entropy, 19**(7), 318。[DOI](https://doi.org/10.3390/e19070318)；[作者公开全文](https://arxiv.org/pdf/1602.05063)。Zotero 父项 `SU9SDUHQ`，附件 `BKCCGPHL`；前期重新读取 37/37 页期刊版索引全文。证据：第 4.2 节式（30）—（32），第 4.4 节与 Table 7（PDF 第 20—21 页），第 5.2.1 节 Tables 11—12。用途：CCS 的保源联合边缘版本、逐状态重叠、负特有信息及独立 AND 冗余。

<a id="ref-25"></a>

**[25]** Rauh, J., Banerjee, P. K., Olbrich, E., Jost, J., & Bertschinger, N. (2017). *On Extractable Shared Information*. **Entropy, 19**(7), 328。[DOI](https://doi.org/10.3390/e19070328)；[作者公开全文 v3](https://arxiv.org/html/1701.07805v3)（2017-11-10）；[作者出版记录](https://e5150pro.github.io/publications/)。前期本地 Zotero 检索未匹配到条目。证据：公开全文 Section III 的式（6）—（7）、可提取共信息与内禀条件互信息的关系、Section IV 的 Lemma 2。用途：本文直接采用的单端可提取共信息及其与内禀条件互信息的关系；第 1 节给出当前 EI 分配所需性质的推导。

<a id="ref-26"></a>

**[26]** Pica, G., Piasini, E., Chicharro, D., & Panzeri, S. (2017). *Invariant Components of Synergy, Redundancy, and Unique Information among Three Variables*. **Entropy, 19**(9), 451。[DOI](https://doi.org/10.3390/e19090451)；[作者预印本全文](https://arxiv.org/html/1706.08921)；[公开期刊 PDF](https://openaccess.city.ac.uk/id/eprint/27163/1/entropy-19-00451-v2.pdf)。本地 Zotero 未匹配到条目；前期读取作者预印本 v1（2017-06-27）第 3—4 节，期刊发表于 2017-08-28。用途：不同目标选择及角色不变成分的背景；本文固定处理端，不将预印本称为最新期刊版本。

建议阅读顺序：[25]（单端提取量）→第 1—2 节（二元分配与证明）→第 3—4 节（固定残余递归）→[17–18]（LC 及恒等性的边界）→[4,9,19,24]（分配对照）→[6–7]（双侧扩展）；[26] 作为角色不变成分的背景。当前 PEID 方法依据是第 6.4 节记录的主文与补充附件，历史 [1] 仅作背景。
