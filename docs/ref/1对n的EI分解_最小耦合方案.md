# 对称可提取共信息的 EI 分解：定义、性质与研究路线

> **主研究路线：** 用同一个三变量冗余函数连接 2→1 与 1→2；在共同因子化干预下恢复二源 PEID，在任意有限离散联合分布下保证四项非负与信息收支。正文先给定义和核心证明，再说明机制解释、已有工作的关系及下一步；长证明见附录 A，淘汰路线见第 6 节与附录 B。
>
> 本文把下面的函数作为**本研究提出的候选定义**。它的可提取信息构件已有文献来源；与已比较函数的不等价可由反例证明，但尚未完成足以宣称“学界首次”的优先权核查。整理及当前稿重核日期：2026-10-04。保留原文件路径，正文主方向已从最小耦合转为本定义。

## 1. 主定义：三个位置的单端可提取共信息

### 1.1 定义域与共信息

设 $A,B,T$ 为服从同一个联合分布 $P$ 的有限离散随机变量，字母表固定。信息单位为 bit，所有对数以 2 为底。变量可取有限元组的编码值，不要求为二元变量。先记

$$
a:=I_P(A;T),\qquad b:=I_P(B;T),\qquad j:=I_P(A,B;T),
$$

$$
c:=\operatorname{CoI}_P(A,B,T)
=a+b-j
=I_P(A;B)-I_P(A;B\mid T).
\tag{1-1}
$$

$\operatorname{CoI}$ 是对三个变量完全对称的共信息，采用“冗余减协同”的符号约定；它可以为负。下文省略 $P$ 下标时，仍指这个共同分布。

对任意三个这样的变量 $U,V,W$，定义**单端可提取共信息**

$$
\boxed{
\Phi(U,V;W)
:=\sup_{P_{Z\mid W}}\operatorname{CoI}(U,V,Z)
=I(U;V)-\inf_{P_{Z\mid W}}I(U;V\mid Z).
}
\tag{1-2}
$$

每个候选联合分布严格为 $P(u,v,w)P(z\mid w)$，即 $(U,V)-W-Z$ 构成 Markov 链。$Z$ 只能由 $W$ 局部随机处理得到，其额外随机性给定 $W$ 后与 $U,V$ 独立；不能访问另两个变量或重新选择原联合分布。允许恒等处理 $Z=W$ 及常量处理。这里一次只处理一个位置，其余两个变量保留。

右侧下确界是内禀条件互信息 $I(U;V\downarrow W)$。概率式局部提取及这一关系已有来源，见 Rauh 等 Section III。[25](#ref-25) 本文的新候选在于下一步的三位置构造及其 EI 分配，不能把式（1-2）的构件本身称为首次提出。

### 1.2 对称冗余与四项分解

定义**对称可提取共信息冗余**

$$
\boxed{
R_{\mathrm{sym}}(A,B,T)
:=\max\{\Phi(A,B;T),\ \Phi(A,T;B),\ \Phi(B,T;A)\}.
}
\tag{1-3}
$$

三个候选分别允许处理 $T$、$B$ 和 $A$，再取最大值。任意置换三个变量只会重排候选，因此 $R_{\mathrm{sym}}$ 完全置换对称。它与“同时任意处理三端”的优化不同；本文没有用后者替代式（1-3）。

以 $A,B$ 为两变量一侧、$T$ 为单变量一侧，统一定义

$$
\boxed{
R:=R_{\mathrm{sym}}(A,B,T),\qquad
U_A:=a-R,\qquad U_B:=b-R,\qquad
S:=j-a-b+R=R-c.
}
\tag{1-4}
$$

冗余 $R$ 衡量通过任一单端处理可显露的三变量共信息；特有项是对应单变量互信息扣除这份冗余后的余额；协同是联合互信息扣除一份冗余及两份特有后的余额。这是精确的数学定义，其机制解释及边界见第 3 节。

若沿用 union information 的收支术语，可定义

$$
J_{\mathrm{sym}}:=a+b-R_{\mathrm{sym}},\qquad
U_A=J_{\mathrm{sym}}-b,\quad
U_B=J_{\mathrm{sym}}-a,\quad S=j-J_{\mathrm{sym}}.
\tag{1-5}
$$

$J_{\mathrm{sym}}$ 是本定义诱导的并集量；它不是固定两条边缘通道后最小化联合互信息得到的 $J_{\mathrm{MC}}$。两者收支形式相同，优化问题不同。

### 1.3 在两个方向上的同一用法

| 分解方向 | 两变量一侧 $(A,B)$ | 单变量一侧 $T$ | 采用的冗余 |
|---|---|---|---|
| $(X_1,X_2)\to Y$ | $(X_1,X_2)$ | $Y$ | $R_{\mathrm{sym}}(X_1,X_2,Y)$ |
| $X\to(Y_1,Y_2)$ | $(Y_1,Y_2)$ | $X$ | $R_{\mathrm{sym}}(Y_1,Y_2,X)$ |

两个方向采用式（1-2）—（1-4）同一套构造。$R$ 与 $S$ 都对三个变量完全对称；特有项随所报告的单变量互信息而改变。这个数学对称性比“换向后使用同一术语及框架”更强。

**EI 的干预协议保持因果方向。** 1→2 时取

$$
P(x,y_1,y_2)=u(x)K(y_1,y_2\mid\mathrm{do}(x)),
\tag{1-6}
$$

其中 $u$ 是固定有限支持上的均匀干预分布，$K$ 是原联合输出机制。2→1 时取同一个共同干预协议下的

$$
P(x_1,x_2,y)=u_1(x_1)u_2(x_2)
K(y\mid\mathrm{do}(x_1,x_2)).
\tag{1-7}
$$

其他未被报告的输入仍按同一因子化干预分布边缘化；时间跨度也保持一致。单独 EI 与联合 EI 都从同一个 $P$ 取边缘，不能为单源项另换干预背景。置换泛函中的变量位置不意味着干预输出，也不意味着实际反向干预机制产生相同的 $P$。因果解释来自原干预分布，局部提取是信息论参照操作。

## 2. 已证明的核心性质

### 2.1 关键界、非负性与信息收支

**命题 1。** 对任意有限离散 $P$，

$$
\boxed{
\max(0,c)\le R_{\mathrm{sym}}
\le\min\{I(A;B),I(A;T),I(B;T)\}.
}
\tag{2-1}
$$

**简证。** 常量处理给出 0，恒等处理给出 $c$，故有下界。对任意 $Z\leftarrow W$，共信息不超过 $I(U;V)$、$I(U;Z)$、$I(V;Z)$；后两项再由数据处理不超过 $I(U;W)$、$I(V;W)$。因此每个单端提取量均受三个原始二元互信息限制，取最大值仍保留上界。展开证明见附录 A.1。

**推论 1。** 式（1-4）中的四项均非负，且

$$
a=R+U_A,\qquad b=R+U_B,\qquad
j=R+U_A+U_B+S.
\tag{2-2}
$$

非负性分别由 $R\ge0$、$R\le a,b$ 及 $R\ge c$ 得到，收支由代入直接成立。相应并集量满足

$$
\max(a,b)\le J_{\mathrm{sym}}\le\min(j,a+b).
\tag{2-3}
$$

这解决了当前要求中的非负性及两条单信息、一条联合信息的收支。它只证明本二元分解，不证明任意多源晶格上的所有原子非负。

### 2.2 独立源严格退化为 PEID

**命题 2。** 若 $A\perp B$，则

$$
\boxed{
R=0,\quad U_A=I(A;T),\quad U_B=I(B;T),\quad
S=I(A,B;T)-I(A;T)-I(B;T)=I(A;B\mid T)\ge0.
}
\tag{2-4}
$$

**证明。** 式（2-1）的上界含 $I(A;B)=0$，故 $R=0$；其余由式（1-1）、（1-4）推出。

在式（1-7）的共同因子化干预下，$X_1\perp X_2$，式（2-4）与当前二源 PEID 的数值定义严格一致。这个结论直接来自本函数的界，不依赖完整冗余 LC 公理。

由于冗余完全对称，任意一对变量独立都会使 $R=0$；因此 1→2 中若实际两个输出独立，也得到零冗余。独立应在所采用的共同 $P$ 下判断，不能用观测分布中的独立替代干预分布中的独立。

### 2.3 恒等性、包含、无关噪声与连续性

**命题 3。** 本定义满足以下性质，详细证明见附录 A.3—A.5。

| 性质 | 精确结论与适用条件 |
|---|---|
| 普通恒等性 | 若 $T$ 是 $(A,B)$ 的一一编码，则 $R=I(A;B)$ |
| 确定性包含 | 若 $A=f(B)$，则 $R=I(A;T)$，$U_A=0$ |
| 单预测变量自冗余 | 另定义为 $R(A;T)=I(A;T)$，与上述包含及二元上界相容 |
| 无关噪声不变 | 在任意两个位置追加与原三变量联合独立的噪声对，允许这两个噪声相关，$R$ 不变 |
| 输出噪声不改变原子 | 上述噪声追加在输出位置时，单输出及联合 EI 也不变，故四个原子均不变 |
| 有限字母表连续性 | 固定有限字母表时，$R$ 及四项对 $P$ 连续，包括支持边界 |
| 最优值可达 | 处理变量 $W$ 有 $m$ 个正概率状态时，$|\mathcal Z|\le m+1$ 足够；三次优化均可取到最优值 |

这里的单端随机优化一般不能只用确定性分组替代。有限维最优值存在，也不等于有快速的全局求解算法。

二元“增加第二个预测变量后冗余不超过单变量 MI”及确定性包含等号已经证明；任意多源晶格单调性、任意局部处理单调性和独立系统可加性尚未证明，见第 5 节。

### 2.4 明确不采用完整冗余 LC

本定义保留普通恒等性，且对每个条件分布单独应用时仍有相同恒等性与非负性；它**不满足完整目标链式法则**

$$
R(A,B;T,T')=R(A,B;T)
+\sum_tP(t)R_{P(\cdot\mid t)}(A,B;T').
\tag{2-5}
$$

独立公平源的 XOR 已能给出反例：取 $T=A\oplus B$、$T'=A$。原源独立，式（2-5）左侧和右侧第一项都为 0；给定每个 $T=t$ 后，$A,B$ 互相确定，条件冗余为 1，右侧成为 1。详细推导见附录 A.7。[17](#ref-17)、[18](#ref-18)

因此，最初源独立不使该分布族对条件化封闭，不能用“只研究独立源”避开这个完整 LC 反例。应把零冗余退化作为新定义的性质，避免再通过不相容的普遍公理包证明它。

## 3. 机制解释与直觉检验

### 3.1 已有解析例子

以下 $A,B,C,Z,N$ 是相互独立的公平比特；$F,F_1,F_2$ 是与它们独立的翻转噪声，出现两个 $F_i$ 时也相互独立。令 $\langle u,v\rangle:=2u+v$ 表示两个比特的一一编码，避免把四状态源误当作两个另行干预的源。

记 $h_2(p)=-p\log_2p-(1-p)\log_2(1-p)$、$k=1-h_2(0.1)\simeq0.531004$、$d=1-h_2(0.18)\simeq0.319923$。表中单位均为 bit。

| 1→2 机制 | $R$ | $U_1$ | $U_2$ | $S$ | 解释 |
|---|---:|---:|---:|---:|---|
| $X=C$；$Y_1=Y_2=C$ | 1 | 0 | 0 | 0 | 完整复制 |
| $X=\langle A,B\rangle$；$Y_1=A,Y_2=B$ | 0 | 1 | 1 | 0 | 独立分流 |
| $X=Z$；$Y_1=N,Y_2=N\oplus Z$ | 0 | 0 | 0 | 1 | 秘密共享 |
| $X=\langle C,Z\rangle$；$Y_1=\langle C,N\rangle,Y_2=\langle C,N\oplus Z\rangle$ | 1 | 0 | 0 | 1 | 共有与协同同时存在 |
| $X=\langle A,B\rangle$；$Y_1=\langle A,N\rangle,Y_2=\langle B,N\rangle$ | 0 | 1 | 1 | 0 | 共同无关噪声不制造源冗余 |
| $X=C$；$Y_1=C,Y_2=C\oplus F$，翻转率 0.1 | $k$ | $1-k$ | 0 | 0 | 带噪重叠 |
| $X=\langle C,Z\rangle$；$Y_1=\langle C,N\rangle,Y_2=\langle C\oplus F,N\oplus Z\rangle$ | $k$ | $1-k$ | 0 | 1 | 带噪共有与秘密共享并存 |
| $X=\langle A,B\rangle$；$Y_1=A,Y_2=A\land B$ | 0.311278 | 0.688722 | 0.5 | 0 | 确定性概率重叠 |
| $X=C$；$Y_i=C\oplus F_i$，翻转率均为 0.1 | $d$ | $k-d$ | $k-d$ | 0 | 条件独立观测；解释见第 3.3 节 |

这些精确最优值有解析依据：上下界相等、提取 $C$ 达到二元 MI 上界，或由噪声不变性归约得到。数值分组搜索只是核对，不能作为随机优化达到最优的唯一依据。具体见附录 A.8。

2→1 的独立 AND 门得到 $(R,U_1,U_2,S)=(0,0.311278,0.311278,0.188722)$；独立 XOR 门得到 $(0,0,0,1)$，恢复 PEID。

还应保留“输出独立但边缘通道相同”的例子：$X$ 公平，给定 $X=0$ 的 $(Y_1,Y_2)$ 在 $00,01,10,11$ 上的概率为 $(1/2,1/4,1/4,0)$，给定 $X=1$ 时为 $(0,1/4,1/4,1/2)$。两个输出独立，两条通道均为交叉率 $1/4$ 的 BSC。于是 $E_1=E_2=0.188722$、$E_{12}=0.5$；本定义给出 $(0,0.188722,0.188722,0.122556)$。这说明相同边缘通道并不强迫本定义给出正冗余。

### 3.2 为什么必须使用完整联合结构

表中的“共有加秘密共享”和“不同源比特加共同噪声”具有完全相同的标量输入：

$$
E_1=E_2=1,\quad E_{12}=2,\quad
H(Y_1)=H(Y_2)=2,\quad I(Y_1;Y_2)=1,\quad H(X)=2.
\tag{3-1}
$$

但前者冗余应为 1，后者在无关噪声不变要求下应为 0。任何只依赖式（3-1）这些标量的公式都不能同时区分两例。原共信息也都为 0，因此只取其正部仍会遗漏前者的共有 $C$。

式（1-3）利用局部处理与整个联合分布。第一例只需将源 $X=\langle C,Z\rangle$ 处理为 $C$ 即达到 1 bit；第二例由噪声不变性得到 0。它能把共同内容与掩盖这份内容的协同分开，也避免把输出间共同无关噪声等同于对源的共同信息。

### 3.3 完全对称性的代价：条件独立观测协同为零

**命题 4。** 任意三变量完全置换对称的冗余函数，只要在每个变量充当单变量一侧时四项均非负，就必有 $R\le\min\{I(A;B),I(A;T),I(B;T)\}$。若 $A\perp B\mid T$，则被迫满足

$$
\boxed{R=I(A;B),\qquad S=0.}
\tag{3-2}
$$

**简证。** 在三个角色下用特有项非负得到三个二元 MI 上界；条件独立给出 $c=I(A;B)$，协同非负又要求 $R\ge c$，因此相等。见附录 A.6。

更一般地，式（2-1）还给出

$$
0\le S\le\min\{I(A;B\mid T),I(A;T\mid B),I(B;T\mid A)\}.
\tag{3-3}
$$

特别地，若 1→2 的两个输出分别是源的确定性函数，给定源后它们自动条件独立，所以本定义的目标侧协同为零，信息分到冗余与特有。秘密共享例的正协同依赖联合输出中给定源仍存在的随机结构。这是所选强对称框架的实质解释，应在后续应用中保留。

两个 0.1 独立噪声副本的联合 EI 为 $0.742086$，单个 EI 为 $0.531004$；本定义将两个观测的增量分别归入 $0.211081$ bit 的特有项，协同为零。BROJA 则给出 $R=0.531004$、两份特有为零、$S=0.211081$。[4](#ref-4)

这是两种“共同内容／协同”语义的差别。采用本路线意味着接受式（3-2）的分配，不能把所有联合降噪收益都叫作正协同。当前选择以三个变量的共同结构和独立源零冗余为基础；不要求特有信息在边缘通道相同或 Blackwell 等价时必为零。

上述必要性针对**完全三变量置换对称和全角色非负**，不能推成“任何同公式换向构造都必须如此”。它也没有证明所有可能机制都已通过直觉检验；将新反例明确纳入第 5 节的检验任务。

## 4. 与已有函数的关系及可主张的新意

### 4.1 区分构件、函数与完整理论

本研究的候选函数是式（1-3），不是重新发明共信息、内禀条件互信息或局部可提取信息。Rauh 等已给出概率式提取以及单端可提取共信息与内禀条件互信息的关系。[25](#ref-25) 原文关于其他共享信息函数的性质，不能未经证明移植到这里的三位置最大值。

Pica 等 2017 年已比较三种目标选择下的 PID，并构造角色不变的信息成分；其基本做法是从所选择的 PID 原子建立跨角色关系。[26](#ref-26) 因此“同时看三个方向”和“寻找角色不变成分”都有先例。本研究目前的具体选择是直接优化三个单端可提取共信息，再取最大值，而不是先指定一个 PID 冗余后再拆分角色成分。两类构造是否可能数值等价，应作为优先核查问题。

可以主张：**本文提出一个以单端局部随机处理为基础、完全三变量对称的二元冗余候选，并证明其独立源 PEID 退化、四项非负、噪声不变与有限字母表连续性。** 目前不能主张它与所有已有信息量均不同，或已经给出完整 1→N／$m\to n$ 理论。

### 4.2 与六种已比较函数确有不等价

以下每一行都是同一分布、同一角色和同一信息单位下的差异。一个反例足以证明两个泛函不恒等；这不意味着任一函数在自己的操作语义下“算错”。

| 已有函数／构造 | 鉴别例 | 已有冗余 | $R_{\mathrm{sym}}$ | 对本研究要求的影响 |
|---|---|---:|---:|---|
| MMI：$\min(a,b)$ | 独立公平源，$T$ 为两源一一编码 | 1 | 0 | 不满足独立源零冗余及该复制恒等性 |
| 最小耦合／BROJA | 独立公平源 AND，$T=A\land B$ | 0.311278 | 0 | 非负，但与所选 PEID 分配不同 |
| RR | 第 3.1 节共有加秘密共享 | 0.5 | 1 | 标量插值削弱完整共有内容 |
| $I_{\mathrm{CCS}}$ | 独立公平源 AND | 0.103759 | 0 | 不普遍满足独立源零冗余 |
| 共同确定性 $I_{\wedge}$ | 精确副本与 0.1 带噪副本 | 0 | 0.531004 | 按四项收支产生负协同 |
| 双路径通道公式直接代入 | 同一副本例，纳入两个次序 | 0.319923 | 0.531004 | 串接噪声使冗余低于非负下界 |

MMI 见 [9](#ref-9)，BROJA 见 [4](#ref-4)，RR 见 [19](#ref-19)、[22](#ref-22)，CCS 采用 Ince 保持三对边缘的最大熵参照版本，见 [24](#ref-24)，共同确定性见 [20](#ref-20)，路径构造见 [23](#ref-23)。计算与适用假设见附录 B；不能混用不同 CCS 版本或路径裁选规则。

### 4.3 公理选择与相对优势

| 要求 | 本定义 | RR | MMI | 所核对的 CCS 版本 |
|---|---|---|---|---|
| 任意有限离散分布上有定义 | 是；涉及全局随机优化 | 是；闭式标量插值 | 是；直接取最小 | 是；涉及最大熵参照 |
| 源独立时零冗余 | 保证 | 保证 | 不保证 | 不保证 |
| 二元四项非负 | 保证 | 保证 | 保证 | 不保证；可有负特有 |
| 普通复制恒等性 | 保证 | 一般不成立 | 一般不成立 | 不满足；仅满足独立源复制的较弱恒等性 |
| 三个变量完全置换对称 | 保证 | 不保证 | 不保证 | 不作为本文已证明的性质 |
| 共同无关输出噪声不改变分配 | 保证 | 有反例 | 该类噪声下保留；仍有独立分流问题 | 本文不宣称一般结论 |
| 完整目标 LC | 不满足 | 不满足 | 不满足 | 不满足；独立 XOR 条件化已有反例 |
| 只依赖两条源—目标边缘 | 不要求；使用实际联合结构 | 否；还用源依赖 | 是 | 否；参照保留第三对边缘 |
| Blackwell 等价边缘通道的特有必为零 | 不要求，已有反例 | 不保证 | 同 MI 时为零，非完整 Blackwell 表征 | 不作为本文已证明的性质 |

本路线相对 RR 的优势是内容区分和噪声稳定性；相对 MMI 的优势是独立源退化与复制恒等性；相对 CCS 的优势是四项全分布非负以及所要求的独立源退化。它牺牲了完整 LC 与单纯边缘通道决策语义，而且计算比 RR、MMI 更复杂。这些具体取舍比笼统宣称“比其他 PID 更正确”更有说服力。

MMI 对联合高斯、标量目标且冗余／特有仅依赖两条边缘的类别有特定理论支持，[9](#ref-9) 不能用一个离散反例否定该条件结论。CCS 区分逐状态共同信息变化，有自己的内容动机；负特有反例说明它不适合本研究要求的全分布非负四项分配，见附录 B.3。

## 5. 后续研究路线

### 5.1 先完成二元理论与优先权核查

首先核查式（1-3）是否与已有对称共享信息、角色不变 PID 成分、内禀信息或其封闭构造等价。优先从 [25](#ref-25) 的引用链和 [26](#ref-26) 的跨角色构造进入，也要比较“对已知 PID 的三个角色冗余取最小值”等可行对称化。仅未检索到相同公式不足以宣称首次。

随后检验下列未决问题，每项都需要证明或最小反例，不能从第 2 节自动推断：

1. 任一变量经过确定性或随机局部处理后，$R_{\mathrm{sym}}$ 是否单调不增？单个 $\Phi$ 对被处理端的单调性不能直接证明三个候选的最大值具有此性质。
2. 加入目标信息或扩大观测变量时，冗余、特有与协同分别如何变化？明确所变的是哪一侧，避免把不同单调性混成一个公理。
3. 两个相互独立的信息模块合并后，$R$ 是否可加？允许一个局部通道联合处理模块时，优化可能耦合，不能仅靠逐模块候选证明等号。
4. 相同边缘通道、改变实际输出耦合时，本定义的变化是否与选定的共同结构解释一致？用第 3.1 节的独立同等通道和条件独立副本作为起点。

当前已经证明的性质是第 2 节与附录 A 的命题；这一清单是待研究事项。

### 5.2 可核验的有限状态求解

附录 A.5 给出有限辅助字母表界，因此可以先开发精确概率表上的二元求解器。每个变量分别搜索局部随机通道，报告可行解和全局最优差距；确定性分组适合作为下界与可解释候选，但未达到上界时不能当成完整随机优化。

一个有用的计算报告是

$$
R_{\mathrm{lo}}:=\max\{0,c,\text{已计算可行局部通道的共信息值}\},\qquad
R_{\mathrm{hi}}:=\min\{I(A;B),a,b\}.
\tag{5-1}
$$

其中 0 和 $c$ 来自明确可行的常量与恒等通道。理论上 $R_{\mathrm{lo}}\le R_{\mathrm{sym}}\le R_{\mathrm{hi}}$；可以进一步收紧上界。式（5-1）是明确的优化界，不是对估计 Syn 的裁剪。

由此有 $U_A\in[a-R_{\mathrm{hi}},a-R_{\mathrm{lo}}]$、$U_B\in[b-R_{\mathrm{hi}},b-R_{\mathrm{lo}}]$、$S\in[R_{\mathrm{lo}}-c,R_{\mathrm{hi}}-c]$。若临时报告 $R_{\mathrm{lo}}$ 诱导的分配，应明确它低估冗余及协同、高估特有，不能标作精确 $R_{\mathrm{sym}}$。

实现时区分优化误差和 MI 估计误差。每次消耗估计 Syn 都声明原生单位容差、记录处于 $[-\tau,0)$ 的数量；低于 $-\tau$ 时显式失败并报告最小值、阈值、数量。若估计量导致 $R_{\mathrm{lo}}>R_{\mathrm{hi}}$，也应报告不一致，不能静默投影。

### 5.3 基准问题与比较设计

先复用第 3 节的小状态机制，逐步加入相关源、非对称噪声、低概率状态、支持连通性变化及独立模块组合。对照包括 BROJA、RR、MMI、CCS、共同确定性与路径参照；评价应对应具体要求，而非以本指标自身输出作唯一真值。

重点报告：独立源零冗余、共有与秘密共享同时存在、共同无关噪声不变、低噪声连续性、四项收支和全角色非负；将条件独立观测零协同作为明确语义边界。当前尚无大规模随机分布或连续系统结果。

### 5.4 1→N 的第一步：两块分解

令 $X\to(Y_1,\ldots,Y_n)$，全部信息仍来自同一个干预 $P$。对不交非空目标索引块 $G,H$，记向量 $\boldsymbol{y}_G=(Y_i)_{i\in G}$、$\boldsymbol{y}_H=(Y_i)_{i\in H}$，以及 $E(G):=I_P(X;\boldsymbol{y}_G)$。直接沿用新函数定义

$$
\begin{aligned}
R_X(G\mid H)&:=R_{\mathrm{sym}}(\boldsymbol{y}_G,\boldsymbol{y}_H,X),\\
U_G&:=E(G)-R_X(G\mid H),\\
U_H&:=E(H)-R_X(G\mid H),\\
S_X(G\mid H)&:=E(G\cup H)-E(G)-E(H)+R_X(G\mid H).
\end{aligned}
\tag{5-2}
$$

有限状态下，把每个目标向量视为一个有限随机变量，命题 1 立即保证四项非负与两块收支。因此式（5-2）是已经有证明依据的延拓起点。若两块输出在 $P$ 下独立，则块间冗余为零。

源侧同理将独立干预源分成两块 $\boldsymbol{x}_G,\boldsymbol{x}_H$，用 $R_{\mathrm{sym}}(\boldsymbol{x}_G,\boldsymbol{x}_H,T)$；因子化干预保证两块独立，恢复当前 PEID 的块合并增益。这样两侧使用同一构造。

对全部 $n$ 个输出，非空无序二分共有 $2^{n-1}-1$ 个。报告这些二分量是在不同划分下看同一系统，**不是把它们相加后的全局非重叠原子分解**。可先选研究关心的分块，避免把枚举所有二分当成必需。

### 5.5 多块与树结构尚需新的收支规则

一般 $n\ge3$，不能只用“一个全体冗余 + 每个目标一个特有 + 一个协同”。取公平 $X$、$(Y_1,Y_2,Y_3)=(X,X,0)$，则单 EI 为 $(1,1,0)$、联合 EI 为 1。全体冗余因第三目标无信息被迫为 0，前两项特有各为 1，于是协同只能为 $-1$。问题在于缺少只属于前两个目标的部分共享项。

因此后续应先解决**不同二分之间如何兼容**：同一信息是否重复计算，加入无信息目标是否保留旧子集报告，改变树结构是否仅改变归属而保持总量。目标侧式（5-2）在每个节点增加冗余，不能照搬源独立时的望远镜式非负协同树；详细原因见附录 C.2。

真正的 $m\to n$ 延拓还需要源块与目标块的联合一致性。ΦID 已有双侧乘积晶格，[6](#ref-6)、[7](#ref-7) 2→2 也需要额外双冗余量。两边都选 $R_{\mathrm{sym}}$ 不能自动确定或保证其全部 16 个原子非负。本路线先研究可解释的二块报告，再决定是否构造完整原子或明确的压缩层级。

### 5.6 连续 EI 与现有 PEID 的接口

本文件的可达性及连续性定理以固定有限字母表为条件。连续变量需要另行指定支持、干预分布、局部随机核的类别、MI 有限性及优化紧性，不能把有限状态定理直接搬过去。

估计 EI 时遵循项目的 TM 优先规则。当前稿的仿射高斯特征代理与精确理论有不同保证，见附录 C.3；现有代码入口也未在本次完成全面对齐。求解器、连续估计器及 SPT 修订是后续任务，本次没有因整理 MD 而修改它们。

## 6. 已淘汰的主路线：保留关键理由与对照用途

| 路线 | 保留的价值 | 不再作为本研究主定义的原因 |
|---|---|---|
| 最小耦合／BROJA | 固定单通道后的信息能力比较，二元非负分配 | 独立 AND 仍有正冗余，不恢复所选 PEID |
| 源侧 PEID、目标侧 BROJA 分开定义 | 在允许两侧不同规则时可行 | 当前要求采用同一构造，因此降为备选 |
| 直接换位 RR | 全分布非负，独立预测变量零冗余，计算简单 | 无法区分共有加秘密共享与共同无关噪声；权重随私有噪声改变 |
| MMI | 简单，某些高斯类别有理论依据 | 独立分流误分冗余和协同，不满足所选恒等性 |
| CCS | 逐状态内容重叠，有明确解释 | 独立 AND 有正冗余，另有负特有反例 |
| 共同确定性变量 | 无误共有内容，能区分共有信号与共同噪声 | 小噪声导致跳变，四项收支可产生负协同 |
| 双路径公式直接代入 | 指定路径的信息传输参照 | 额外噪声串接低于冗余非负下界；还需路径选择假设 |
| 仅取正共信息 | 最简单的二元非负可行分配 | 共有信息会被协同抵消，不能保留混合机制中的共享 $C$ |

Blackwell 共同退化通道量有独立的决策论含义，与最小耦合并集量也有已知关系。[5](#ref-5) 它与本定义的实际联合结构语义不同，作为对照保留；不将其直接等同于本定义或单纯认定错误。

旧最小耦合的多目标子集报告仍可算作比较基线。它们的非负汇总和指数数量并未证明全局原子成立，不能将这些旧结论挪作新 $R_{\mathrm{sym}}$ 的多变量定理。摘要与反例见附录 B.5。

## 7. 当前稿依据、已知差异与证据范围

本次重新通过 Zotero 定位并核对题名 *Emergent hierarchical organization of causal interactions in complex systems*，父项为 `P6UJCVG8`；重新列出附件并读取当前可用的主文 `DXGC7JEA`（19 页）及补充材料 `MWIWKSVG`（28 页）。未发现明确稿件日期或版本号，附件元数据不足以确定更新先后；这里只依据当前可用附件，不声称已消除版本歧义。

| 位置 | 与本定义的关系 |
|---|---|
| 主文 Methods 第 15—16 页，式（5）—（8） | 所有 EI 来自同一因子化最大熵干预联合分布；二源 EI 差值与式（2-4）一致 |
| 补充 S2 第 5 页，式（S20）—（S27） | 明确采用普通恒等性、完整目标 LC 与逐条件分布平均；本定义保留恒等性而不保留完整 LC |
| 补充 S3.1 第 6 页，式（S29）—（S36） | 用该公理包桥接 PEID 与 PID；XOR 暴露其全分布适用问题，不能声称新定义验证了该桥接 |
| 补充 S3.2 第 7 页，式（S37）—（S44） | 条件总相关的 KL 非负性证明独立成立，见附录 C.1 |
| 补充 S3.3 第 8 页，式（S46）—（S48） | 源侧分块代数与独立块非负性可保留；目标侧新增冗余需要重新建立树收支 |
| 补充 S1.2 第 3—4 页，式（S11）—（S15） | 特征空间中的仿射高斯代理不自动继承精确 Syn 非负性，见附录 C.3 |

**方法修订方向：** 以后以共同干预 EI 及式（1-3）为二元定义入口，直接证明独立源退化；多源源侧保留条件总相关与独立块证明。相应改写当前稿中“在所列完整公理下必然零冗余”的桥接表述。当前 MD 记录了这个差异，主稿及实验代码本次未修改，不能称为已经全面一致。

历史 PEID 预印本 [1](#ref-1) 只作补充背景，不替代上述当前稿。前期已核对其他文献的证据层级保留在参考文献中；仅读过摘要的条目不用于证明具体公式或公理。

前期有限概率表核对涵盖第 3.1 节九个机制及两个独立源门，穷举单端确定性分组并检查六种变量置换；随机处理的最优值靠解析界和噪声命题确定，没有运行随机通道全局数值优化。容差为 $10^{-12}$ bit，11 个机制中有 1 个原子处于 $[-10^{-12},0)$，来自含噪共有加秘密共享机制的零特有项；低于 $-10^{-12}$ 的原子为 0，最大置换误差 $4.44\times10^{-16}$ bit。没有裁剪 Syn，没有连续 EI 估计或大规模实验。这些是前期验证记录，不表述为本次新增实验。

## 附录 A. 新定义的详细证明

### A.1 单端提取与三变量共同上界

任取 $Z\leftarrow W$。共信息的三种展开分别为

$$
\begin{aligned}
\operatorname{CoI}(U,V,Z)&=I(U;V)-I(U;V\mid Z),\\
&=I(U;Z)-I(U;Z\mid V),\\
&=I(V;Z)-I(V;Z\mid U).
\end{aligned}
\tag{A-1}
$$

条件互信息非负，得到它不超过右侧三个无条件 MI。Markov 链又给出

$$
I(U;Z)\le I(U;W),\qquad I(V;Z)\le I(V;W).
$$

因此

$$
\Phi(U,V;W)\le\min\{I(U;V),I(U;W),I(V;W)\}.
\tag{A-2}
$$

$Z$ 为常量时，共信息为 0；$Z=W$ 时，共信息为原始三变量 $c$。于是每个 $\Phi$ 都至少为 $\max(0,c)$。三个 $\Phi$ 的上界都是同一组二元 MI，下界也相同，取最大值得式（2-1）。

$\Phi(U,V;W)$ 对 $U,V$ 对称；置换 $A,B,T$ 将三个候选之间重排，式（1-3）的最大值不变。这同时证明完全三变量对称性。

### A.2 四项非负与独立源退化

任何采用式（1-4）收支的二元分配，四项非负的充要条件是

$$
\max(0,c)\le R\le\min(a,b).
\tag{A-3}
$$

式（2-1）满足这个条件。直接代入得到 $R\ge0$、$U_A\ge0$、$U_B\ge0$、$S=R-c\ge0$；相加给出式（2-2）。再由 $R\le a,b$、$R\ge c,0$ 分别得到 $J_{\mathrm{sym}}\ge a,b$、$J_{\mathrm{sym}}\le j,a+b$，证明式（2-3）。

源独立时，$I(A;B)=0$ 迫使 $R=0$。用共信息的源侧展开，

$$
S=-c=I(A;B\mid T)-I(A;B)=I(A;B\mid T).
\tag{A-4}
$$

条件互信息是 KL 散度的平均，故非负。这里不要求 $A,B$ 为公平比特；独立即可。最大熵且因子化干预是本项目产生这种独立的协议，均匀性不是该信息论等式的必要条件。

### A.3 恒等性、包含与重编码

若 $T=\langle A,B\rangle$ 为一一编码，给定 $T$ 后两个源完全确定，$I(A;B\mid T)=0$，因此 $c=I(A;B)$。式（2-1）的下界为 $I(A;B)$，上界也不超过它，所以 $R=I(A;B)$。

若 $A=f(B)$，则 $I(A,B;T)=I(B;T)$，故 $c=I(A;T)$；上界不超过 $I(A;T)$，所以 $R=I(A;T)$。数据处理保证 $I(A;T)\le I(B;T)$。

对每个变量作一一重编码，不改变 MI、CMI，也在局部通道之间产生一一对应，故 $R$ 及对应分配不变。这里指各变量各自的一一编码；把两个变量混成新变量会改变分解对象，不是该不变性。

单预测变量自冗余按 $I(A;T)$ 定义；包含等号及二元上界使它与二元构造相容。它没有指定三个及以上预测变量的冗余晶格函数。

### A.4 共同无关噪声不变性

令噪声对 $(N_A,N_B)$ 与原 $(A,B,T)$ 联合独立，允许 $N_A,N_B$ 相互相关。取 $\widetilde A=\langle A,N_A\rangle$、$\widetilde B=\langle B,N_B\rangle$；此处括号表示有限取值的一一编码。

**处理 $T$ 的候选。** 对每个 $Z\leftarrow T$，噪声与 $(A,B,T,Z)$ 独立，故

$$
\begin{aligned}
I(\widetilde A;\widetilde B)&=I(A;B)+I(N_A;N_B),\\
I(\widetilde A;\widetilde B\mid Z)&=I(A;B\mid Z)+I(N_A;N_B).
\end{aligned}
\tag{A-5}
$$

相减时噪声项抵消，因此 $\Phi(\widetilde A,\widetilde B;T)=\Phi(A,B;T)$。

**处理 $\widetilde A$ 的候选。** 任取 $Z\leftarrow\widetilde A$，由噪声独立性及条件链式法则，

$$
\begin{aligned}
\operatorname{CoI}(\widetilde B,T,Z)
&=I(B;T)-I(\widetilde B;T\mid Z)\\
&=I(B;T)-I(N_B;T\mid Z)-I(B;T\mid Z,N_B)\\
&\le I(B;T)-I(B;T\mid Z,N_B).
\end{aligned}
\tag{A-6}
$$

对每个正概率 $N_B=n$，条件原联合分布仍为 $P(A,B,T)$。此时

$$
P(z\mid A=a,N_B=n)
=\sum_{n_A}P(n_A\mid n)P(z\mid a,n_A)
\tag{A-7}
$$

是从原 $A$ 到 $Z$ 的可行随机通道。因此，记 $m_A:=\inf_{Z'\leftarrow A}I(B;T\mid Z')$，每个条件态都有 $I(B;T\mid Z,N_B=n)\ge m_A$，平均后也有该下界。式（A-6）于是不超过 $I(B;T)-m_A=\Phi(B,T;A)$。

对所有扩展通道取上确界，得到 $\Phi(\widetilde B,T;\widetilde A)\le\Phi(B,T;A)$。反向不等式由忽略 $N_A$ 的原通道得到；对于这种通道，$N_B$ 与原变量及 $Z$ 独立，添加它不改变候选共信息。故二者相等。处理 $\widetilde B$ 同理。

三个候选量分别不变，取最大值后 $R_{\mathrm{sym}}$ 不变。完全对称性使结论适用于任意两个位置；把其中一个噪声取常量也包括只加单个无关噪声。该命题未要求同时追加在三个位置的任意相关噪声都满足不变性。

若两个位置是输出，则 $I(X;\widetilde A)=I(X;A)$、$I(X;\widetilde B)=I(X;B)$、$I(X;\widetilde A,\widetilde B)=I(X;A,B)$，故由式（1-4）所有原子均不变。

### A.5 辅助字母表界、可达性与连续性

设被处理变量 $W$ 有 $m$ 个正概率状态，去掉零概率状态后记 $\mathbf p=P_W$。给每个 $Z=z$ 写权重 $\lambda_z=P(Z=z)$ 和后验概率向量 $\mathbf v_z=P(W\mid Z=z)$。向量使用粗体小写；它们满足

$$
\lambda_z\ge0,\qquad\sum_z\lambda_z=1,\qquad
\sum_z\lambda_z\mathbf v_z=\mathbf p.
\tag{A-8}
$$

因为 $Z$ 仅由 $W$ 产生，条件分布 $P(U,V\mid Z=z)$ 为 $\sum_wv_z(w)P(U,V\mid W=w)$。定义单纯形上连续函数

$$
F(\mathbf v):=
I_{\sum_wv(w)P(U,V\mid W=w)}(U;V).
\tag{A-9}
$$

则要最小化的条件 MI 为 $\sum_z\lambda_zF(\mathbf v_z)$。函数图像 $(\mathbf v,F(\mathbf v))$ 位于 $m$ 维仿射空间：$\mathbf v$ 的自由维数为 $m-1$，函数值再占一维。其图像紧，有限维凸包亦紧。

最小值就是该凸包中第一坐标为 $\mathbf p$ 时的最低函数坐标。可行切片非空且紧，故最低值可达。Carathéodory 定理保证该点可由至多 $m+1$ 个图像点混合得到，因此 $|\mathcal Z|\le m+1$ 足够。

每个混合解都可还原成局部通道：对 $p(w)>0$，

$$
P(z\mid w)=\frac{\lambda_zv_z(w)}{p(w)}.
\tag{A-10}
$$

式（A-8）保证每行和为 1，且该通道产生相同后验及目标值。这里给出充分界，不声称它在所有分布上最紧。

为证明连续性，固定整个 $W$ 字母表大小 $M$，统一使用 $M+1$ 个辅助状态，必要时增加零概率状态。所有行随机通道构成不依赖 $P$ 的紧集合。$P$ 与通道共同决定的有限联合分布连续，MI、CMI 和共信息在有限概率单纯形上连续，取该紧域上的最大值仍对 $P$ 连续。三个候选取最大值保持连续，其余原子由连续 MI 加减得到。这个证明包括支持变化，不依赖在支持边界定义后验。

### A.6 条件独立零协同的必要性

设某个三变量冗余 $R^*$ 完全置换对称，对每个角色均按四项收支定义非负原子。在 $T$ 为单变量时，特有非负给出 $R^*\le I(A;T),I(B;T)$。在 $A$ 为单变量时，又给出 $R^*\le I(A;B)$。因此

$$
R^*\le\min\{I(A;B),I(A;T),I(B;T)\}.
\tag{A-11}
$$

当 $A\perp B\mid T$ 时，$c=I(A;B)$；在原角色下协同非负给出 $R^*\ge c$。上下界相等，得式（3-2）。它适用于任何满足这些条件的函数，不仅是式（1-3）。

此外 $S=R-c\le I(A;B)-c=I(A;B\mid T)$；对另外两对变量作相同展开，得到式（3-3）。

### A.7 独立 XOR 对完整 LC 的反例

取独立公平比特 $A,B$，$T=A\oplus B$、$T'=A$。对原源对，$I(A;B)=0$，所以

$$
R_{\mathrm{sym}}(A,B,T)=0,\qquad
R_{\mathrm{sym}}(A,B,\langle T,T'\rangle)=0.
\tag{A-12}
$$

给定 $T=t$ 后，$B=A\oplus t$，而 $A$ 仍公平。用包含性质，在条件分布上

$$
R_{P(\cdot\mid T=t)}(A,B,T')=I(A;A\mid T=t)=1.
\tag{A-13}
$$

所以逐条件平均为 1，完整 LC 要求 $0=0+1$，矛盾。源在最初分布下独立，条件化后依赖；反例没有靠“不独立的原源”才成立。

对任意候选公理包也可看出同一问题：联合目标 $\langle T,T'\rangle$ 与 $(A,B)$ 一一对应，普通恒等性和无损目标重编码给出冗余 0；逐条件恒等性给出条件冗余 1；完整 LC 加冗余非负不能成立。这个直接反例的约定比一句“恒等与 LC 冲突”更完整，不能省略条件量定义与重编码假设。

Finn–Lizier 的已发表 Theorem 6 表述涉及目标链式法则、恒等性质和所有 PID 原子的 local positivity。[18](#ref-18) 本文在上述明确约定下展示直接矛盾，不把其原定理悄然改写成仅一个未限定的冗余非负命题。

### A.8 表中最优值的解析确定

复制、独立分流和纯秘密共享的值分别由包含、独立零冗余和 MI 收支决定。确定性重叠 $X=\langle A,B\rangle$、$Y_1=A,Y_2=A\land B$ 给定源后输出条件独立，命题 4 给出 $R=I(Y_1;Y_2)=0.311278\ldots$。

共有加秘密共享例中，$I(Y_1;Y_2)=1$；源处理 $X\mapsto C$ 得到共信息 1，达到上界。带噪版本同理有 $I(Y_1;Y_2)=k$，处理为 $C$ 后两输出给定 $C$ 独立，其共信息为 $k$，达到上界。共同无关噪声例由附录 A.4 归约为独立分流。

精确副本加带噪副本时 $a=j=1,b=k,c=k$，界迫使 $R=k$。两个独立噪声副本给定 $X$ 条件独立，$R=d$、$S=0$。其联合 MI 可写为

$$
E_{12}=1-\left[0.82h_2(1/82)+0.18\right]
\simeq0.742085859,
\tag{A-14}
$$

其中两个输出一致的概率为 0.82，一致时后验错误概率为 $0.01/0.82=1/82$；输出不一致时源后验公平。故 $E_{12}-k=k-d=0.211081\ldots$。

上述构造给出可行局部通道达到理论上界或上下界相等，因而不仅确定性处理、随机处理也不能超出表中最优值。

## 附录 B. 既有方案的必要定义与反例

### B.1 最小耦合：非负成立，独立源退化失败

对两变量一侧 $A,B$ 和单变量 $T$，定义

$$
\Delta(P):=\{Q:Q_{AT}=P_{AT},\ Q_{BT}=P_{BT}\},\qquad
J_{\mathrm{MC}}:=\min_{Q\in\Delta(P)}I_Q(A,B;T).
\tag{B-1}
$$

该量来源于 Griffith–Koch union information 及二源 BROJA。[3](#ref-3)、[4](#ref-4) 数据处理给出 $J_{\mathrm{MC}}\ge\max(a,b)$，原 $P$ 可行给出 $J_{\mathrm{MC}}\le j$，条件独立参照 $Q_0(a,b,t)=P(t)P(a\mid t)P(b\mid t)$ 又给出 $J_{\mathrm{MC}}\le a+b$。因此

$$
R_{\mathrm{MC}}:=a+b-J_{\mathrm{MC}},\quad
U_A^{\mathrm{MC}}:=J_{\mathrm{MC}}-b,\quad
U_B^{\mathrm{MC}}:=J_{\mathrm{MC}}-a,\quad
S_{\mathrm{MC}}:=j-J_{\mathrm{MC}}
\tag{B-2}
$$

四项非负。源独立时，PEID 与之的准确关系是

$$
S_{\mathrm{PEID}}=j-a-b=S_{\mathrm{MC}}-R_{\mathrm{MC}}.
\tag{B-3}
$$

独立公平源 AND：$a=b=h_2(1/4)-1/2=0.311278\ldots$、$j=h_2(1/4)=0.811278\ldots$。固定两条边缘允许 $Q$ 令两源完全相同，达到 $J_{\mathrm{MC}}=a$，因此 $R_{\mathrm{MC}}=a$、两项特有为 0、$S_{\mathrm{MC}}=1/2$；PEID 是 $R=0$、$U_A=U_B=a$、$S=0.188722\ldots$。

即使强迫候选源仍独立，保留 AND 的两条边缘会迫使回到原分布，$J=j$；若还套用式（B-2），冗余反而成为 $a+b-j=-0.188722\ldots$。单加源独立约束不能修复原四项公式。依 PEID 原则可称其正冗余不符合所需分配，不能无条件宣布其通道操作语义“高估错误”。

### B.2 RR 与 MMI

Goodwell–Kumar RR 在有限离散二源情形取

$$
R_-:=\max(0,c),\quad R_+:=\min(a,b),\quad
\alpha:=\frac{I(A;B)}{\min\{H(A),H(B)\}},\qquad
R_{\mathrm{RR}}:=R_-+\alpha(R_+-R_-).
\tag{B-4}
$$

分母为 0 时明确定义 $R_{\mathrm{RR}}=0$。$0\le\alpha\le1$ 保证四项非负；源独立使 $c\le0$、$\alpha=0$，故 $R=0$。这已经是发表过的独立源零冗余前例。[19](#ref-19)、[22](#ref-22)

但式（3-1）两例均有 $\alpha=1/2,R_-=0,R_+=1$，RR 都给出 $R=1/2$。在共有加秘密共享例中给每个输出再追加一个独立私有公平噪声后，单 EI、联合 EI及两输出 MI 不变，输出熵变为 3，RR 降为 $1/3$。其权重是依赖强度而非充分的内容判据。

MMI 取 $R_{\mathrm{MMI}}=\min(a,b)$，始终在式（A-3）区间内，四项非负。但独立复制目标有 $a=b=1,j=2$，它给出 $R=S=1,U_A=U_B=0$，与所需独立分流不同；复制恒等性要求 $R=I(A;B)=0$。MMI 的特定高斯结论见 [9](#ref-9)，不作全类别推广。

### B.3 CCS 的版本与两个关键例子

这里的 $I_{\mathrm{CCS}}$ 按 Ince 第 4.2 节式（30）—（32）定义：构造保持三对边缘 $P_{AB},P_{AT},P_{BT}$ 的最大熵参照分布，再按逐状态信息变化的符号一致条件保留共信息项。[24](#ref-24) 不使用只保持两条目标边缘的另一变体替代它。

独立公平 AND 的这些边缘唯一确定原概率表。该定义给出

$$
R_{\mathrm{CCS}}=\tfrac14\log_2(4/3)=0.103759\ldots,\quad
U_A=U_B=0.207519\ldots,\quad S=0.292481\ldots.
\tag{B-5}
$$

这不符合独立源零冗余。但它在完整共享 $C$ 加秘密共享 $Z$ 中可以识别 1 bit 共有内容，不能只凭独立 AND 就否定它所有内容解释。

完整复制恒等性也不成立：目标无损复制源对时，三个局部目标 MI 都是非负惊讶度，CCS 的符号规则只保留源间正局部 MI；存在负局部 MI 时，结果一般不等于普通 $I(A;B)$。原文第 4.4 节明确区分完整恒等性与独立源复制的较弱恒等性。完整 LC 则由独立 XOR、剩余目标 $A$ 的条件化给出 $0=0+1$，与附录 A.7 同型。

原文 Table 7 的概率 $P(0,0,0)=0.4$、$P(0,1,0)=0.1$、$P(1,1,1)=0.5$，变量依次为 $A,B,T$，有 $T=A$。$a=j=1,b=0.6099865\ldots$，CCS 给出 $R=0.7684828\ldots$，因此 $U_B=-0.1584963\ldots$。本定义由上下界相等给出 $R=b,U_B=S=0$。这里是候选 PID 原子的负值，不能解释成允许负的 PEID Syn，也不能截断后宣称保持收支。

### B.4 共同确定性与路径参照的带噪副本反例

共同确定性冗余取 $R_{\wedge}=I(K;T)$，$K$ 是两预测变量均可无误恢复的最大共同变量。[20](#ref-20) 对 $X=C,Y_1=C,Y_2=C\oplus F$，$0<\varepsilon<1/2$，两输出支持图连通，所以 $K$ 为常量，$R_{\wedge}=0$。但 $E_1=E_{12}=1,E_2=1-h_2(\varepsilon)$，按四项收支得到 $S_{\wedge}=-E_2<0$。在 $\varepsilon=0$ 时冗余为 1，任意正噪声时降为 0，定义不连续。确定性重叠 $Y_1=A,Y_2=A\land B$ 也有同类负协同，并非仅噪声模型的问题。

双路径公式直接代入是明确的参照构造：

$$
\begin{aligned}
Q_{12}(y_1,y_2,x)&:=P(y_1)P(y_2\mid y_1)P(x\mid y_2),\\
Q_{21}(y_2,y_1,x)&:=P(y_2)P(y_1\mid y_2)P(x\mid y_1),\\
R_{\mathrm{path}}&:=\min\{I_{Q_{12}}(Y_1;X),I_{Q_{21}}(Y_2;X)\}.
\end{aligned}
\tag{B-6}
$$

互信息按诱导分布计算。副本例中的较差信道是两次 BSC 串接，交叉率 $\delta=2\varepsilon(1-\varepsilon)>\varepsilon$，故 $R_{\mathrm{path}}=1-h_2(\delta)<E_2$、$S=h_2(\varepsilon)-h_2(\delta)<0$。$\varepsilon=0.1$ 时冗余为 $d$、协同为 $-0.211081\ldots$。

Sigtermans 原文还涉及图、路径存在性及 Markov 等解释条件。[23](#ref-23) 此反例检验的是纳入两个次序的直接通道公式，不宣称覆盖任何额外裁边规则。共同确定性及路径量都能区分第 3.2 节的共有内容与共同噪声；作为参照有价值，但未满足本研究的全分布非负四项要求。

### B.5 旧最小耦合多目标报告为何不等于新延拓

对目标索引集合 $G$，固定所有 $P_{XY_i}$ 定义 $J_{\mathrm{MC}}(G)=\min_QI_Q(X;\boldsymbol{y}_G)$，有

$$
\max_{i\in G}E_i\le J_{\mathrm{MC}}(G)\le
\min\left\{E(G),\sum_{i\in G}E_i\right\}.
\tag{B-7}
$$

于是 $D(G):=\sum_{i\in G}E_i-J_{\mathrm{MC}}(G)$ 及 $S_{\mathrm{MC}}(G):=E(G)-J_{\mathrm{MC}}(G)$ 非负，满足 $E(G)=\sum_{i\in G}E_i-D(G)+S_{\mathrm{MC}}(G)$。$D$ 是重复计数总量，不是一个全体共享原子。它还可与适当共有量组合成三项非负报告，但其中的剩余部分不等于各目标纯特有。

报告 $n$ 个单 EI 及每个大小至少为 2 的子集的 $D,S_{\mathrm{MC}}$，数量为 $2\cdot2^n-n-2$；这是旧方案的报告规模，不是新定义完整原子数。对子集取 Möbius 差分也不能自动保证非负：三个完整副本的每对 $D=1$、三者 $D=2$，三阶差分为 $2-3=-1$。

子集协同亦不必单调：前两个输出秘密共享源，第三个输出直接给出完整源，则前两者 $S_{\mathrm{MC}}=1$，加入第三个后 $J_{\mathrm{MC}}=E=1$，$S_{\mathrm{MC}}=0$。这些是汇总结构的边界，不是负的精确 PEID Syn。

完整 PID 按反链晶格计数，ΦID 按双侧乘积晶格计数；报告少量子集／二分量可减少表示规模，却不等于消除了状态空间增长或解决完整分解。SURD、PED、O-information 等另有各自对象及缩减策略，[10](#ref-10)、[11](#ref-11)、[12](#ref-12)、[15](#ref-15) 不把所有缩减思路归为本研究首次。

## 附录 C. 与现有 PEID 理论及估计的接口

### C.1 多源 EI 差值的独立 KL 证明

对共同因子化干预 $q$、固定目标 $T$ 和源集合 $G$，记 $\boldsymbol{x}_G=(X_i)_{i\in G}$，

$$
\Xi_q(G;T):=I_q(\boldsymbol{x}_G;T)-\sum_{i\in G}I_q(X_i;T).
\tag{C-1}
$$

源独立使无条件熵抵消，得到

$$
\begin{aligned}
\Xi_q(G;T)
&=\sum_{i\in G}H_q(X_i\mid T)-H_q(\boldsymbol{x}_G\mid T)\\
&=\mathbb E_{q(T)}D_{\mathrm{KL}}\left(
q(\boldsymbol{x}_G\mid T)\ \middle\|\ \prod_{i\in G}q(X_i\mid T)
\right)\ge0.
\end{aligned}
\tag{C-2}
$$

又因各单源 MI 非负，$\Xi_q\le I_q(\boldsymbol{x}_G;T)$。这是当前补充式（S37）—（S44）的实质证明，不需要冗余 LC。两源时与式（2-4）一致；多源时它是集成信息增益，不自动等同于纯 $|G|$ 阶 PID 原子。例如第三源与目标无关、前两源 XOR 时，三源 $\Xi$ 仍为 1。

### C.2 源侧树可保留，目标侧树须重建

对不交源块 $G,H$，因子化干预保证 $\boldsymbol{x}_G\perp\boldsymbol{x}_H$，合并增量为

$$
s_q(G,H;T):=I_q(\boldsymbol{x}_{G\cup H};T)
-I_q(\boldsymbol{x}_G;T)-I_q(\boldsymbol{x}_H;T)
=I_q(\boldsymbol{x}_G;\boldsymbol{x}_H\mid T)\ge0.
\tag{C-3}
$$

在任意固定二叉源树上，各节点增量望远镜式相加，

$$
I_q(\boldsymbol{x}_G;T)
=\sum_{i\in G}I_q(X_i;T)
+\sum_{v\in\operatorname{Int}(\mathcal T)}s_q(G_v,H_v;T).
\tag{C-4}
$$

这保留当前 PEID–SPT 的非负源侧收支；不证明树唯一或贪心整树全局最优。

目标侧式（5-2）则有 $E(G\cup H)=E(G)+E(H)-R_X(G\mid H)+S_X(G\mid H)$。在树上直接展开会同时累积冗余的减项和协同的加项；二元非负不保证这些是全局互不重叠原子。需要新的共享内容归属规则，不能只移植式（C-4）。

精确正 $\Xi$ 也不证明机制方程具有不可加交互：独立单位方差高斯源、$T=X_1+X_2+\varepsilon$、独立高斯噪声方差 $\sigma^2>0$ 时，

$$
\Xi=\tfrac12\log_2\frac{(1+\sigma^2)^2}{\sigma^2(2+\sigma^2)}>0.
\tag{C-5}
$$

它测量联合读出增益；$\sigma^2=1$ 时为 $0.207519\ldots$ bit，机制仍是线性可加。

### C.3 理论非负与仿射估计代理的差别

当前补充 S1.2 在特征空间使用仿射高斯 TM 代理。式（S12）的单源特征为 $\boldsymbol{\phi}_s(x)=(x,x^2,x^3)^{\mathsf T}$，联合特征为 $\boldsymbol{\phi}_j(x_1,x_2)=(x_1,x_2,x_1x_2,x_1^2,x_2^2)^{\mathsf T}$，后者没有两个单源的三次项。保留原坐标保证精确 MI 不变，不保证不同特征空间的高斯代理相容。

保留前期解析反例：$X_1,X_2$ 独立均匀于 $[-1,1]$，$T=X_1^3+\varepsilon$，噪声独立高斯、方差 $\sigma^2>0$。精确 PEID Syn 为 0；单源仿射特征模型的预测残差方差为 $\sigma^2$，联合模型只能线性利用 $X_1$，残差方差为 $\sigma^2+4/175$。因此总体高斯代理的差值为

$$
\Delta_G=-\tfrac12\log\left(1+\frac{4}{175\sigma^2}\right)<0.
\tag{C-6}
$$

这里保留自然对数，单位为 nat；$\sigma^2=0.01$ 时为 $-\tfrac12\log(23/7)\simeq-0.594792$ nat。这个偏差在样本无限和正则趋零时仍存在，不能统一归为有限样本误差。至少要让联合特征覆盖单源特征，再验证所有代理 MI 的共同模型相容性；加回三次项仅修复这个例子。

前期有限代码核对还发现：`exp/TM/transport_map_density.py` 的入口使用多项式三角 TM、返回 bit 且 `bias_correction=0.0`；`yrd/__init__.py` 的末尾导入覆盖同名仿射函数，`exp/network_revival/effective_information.py` 保留上述特征但调用该多项式入口。这是当时所核对调用路径的记录，本次未重新追踪全部入口、重跑实验或修改实现。不能把式（C-6）当作所有现有实验的输出，也不能声称当前 PDF 与代码已经一致。

## 参考文献与证据范围

保留前期检索的参考资料，避免丢失理论比较依据；正文仅使用与新主路线有关的条目。全文、摘要及元数据证据层级逐条标出。Zotero key 为本地条目标识，不是 BibTeX 引用键。外部索引及出版社访问存在未完成的检索路径；目前证据足以支持列出的定义、反例及来源关系，不足以支持“覆盖全部工作”或全球首次声明。

<a id="ref-1"></a>

**[1]** Yang, M., Wang, S., & Zhang, J. (2026). *Partial Effective Information Decomposition for Synergistic Causality*. arXiv:2605.03267，预印本。[论文](https://arxiv.org/abs/2605.03267)。Zotero：`MYATYWAJ`；证据：全文，重点为第 2 节、Discussion 及附录 A–B。用途：历史 PEID 背景；当前方法以第 7 节记录的现行主稿为准。

<a id="ref-2"></a>

**[2]** Williams, P. L., & Beer, R. D. (2010). *Nonnegative Decomposition of Multivariate Information*. arXiv:1004.2515。[论文](https://arxiv.org/abs/1004.2515)。Zotero：`K9ZE68ZB`；前期核对证据：摘要与元数据，晶格计数同时由文献 [6–7] 的全文交叉核对。用途：PID 及反链晶格基础。

<a id="ref-3"></a>

**[3]** Griffith, V., & Koch, C. (2014). *Quantifying Synergistic Mutual Information*. arXiv:1205.4265，2014 修订版。[论文](https://arxiv.org/abs/1205.4265)。Zotero：`PIVMHCH6`；证据：全文，重点为 union information 定义及附录。用途：被比较的最小耦合联合信息 $J_{\mathrm{MC}}$ 及其协同的来源，不是本文 $R_{\mathrm{sym}}$ 的来源。

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

**[25]** Rauh, J., Banerjee, P. K., Olbrich, E., Jost, J., & Bertschinger, N. (2017). *On Extractable Shared Information*. **Entropy, 19**(7), 328。[DOI](https://doi.org/10.3390/e19070328)；[作者公开全文 v3](https://arxiv.org/html/1701.07805v3)（2017-11-10）；[作者出版记录](https://e5150pro.github.io/publications/)。本地 Zotero 未匹配到条目。证据：公开全文 Section III 的式（6）—（7）、可提取共信息与内禀条件互信息的关系、Section IV 的 Lemma 2。用途：第 1 节定义的局部提取构件；三个位置取最大值的完整指标及附录 A 的证明须与原文已有结论区分。

<a id="ref-26"></a>

**[26]** Pica, G., Piasini, E., Chicharro, D., & Panzeri, S. (2017). *Invariant Components of Synergy, Redundancy, and Unique Information among Three Variables*. **Entropy, 19**(9), 451。[DOI](https://doi.org/10.3390/e19090451)；[作者预印本全文](https://arxiv.org/html/1706.08921)；[公开期刊 PDF](https://openaccess.city.ac.uk/id/eprint/27163/1/entropy-19-00451-v2.pdf)。本地 Zotero 未匹配到条目；本次读取作者预印本 v1（2017-06-27）第 3—4 节，期刊发表于 2017-08-28。用途：三种角色 PID 的不变信息成分及源冗余已有前例；本文与这些构造是否等价仍待核查，不将预印本称为最新期刊版本。

建议阅读顺序：[25]（局部可提取构件）→第 1—3 节及附录 A（本候选与证明）→[26]（跨角色已有结构）→[4,9,19,24]（主要对照）→[17–18]（LC 及恒等性的适用边界）→[6–7]（双侧扩展）。当前 PEID 方法依据是第 7 节记录的主文与补充附件，历史 [1] 仅作背景。
