# SPT 节点大小与不可约机制阶数：回应建议

核对日期：2026-10-27。范围为概念解释和回应位置；本轮未修改在线稿件、实验实现或图像。

## 判断

“SPT 节点包含 k 个源，并不自动证明不可约的 k 体机制”成立。应区分三个对象：父节点的源块数量、所选二分的组间信息贡献、生成动力学中直接作用项的机制阶数。它们不能仅凭节点大小相互替代。

对固定目标 B 和时域 ℓ，节点 A 的分割贡献为

\[
\operatorname{Syn}_{\ell}^{\mathrm{EID}}(L,R\to B)
=EI_{\ell}(A\to B)-EI_{\ell}(L\to B)-EI_{\ell}(R\to B),
\qquad A=L\mathbin{\dot\cup}R.
\]

这度量的是两个子组在该分割下的联合增量。它没有逐项去除所有跨越分割的低阶机制，也不是对所有源子集完成纯阶分解后的唯一 k 阶原子。节点大小 k 因而应解释为父组规模或明确约定的层级索引。

该限定不改变非负性及树上加和恒等式；它限制的是从分层信息贡献到机制阶数的解释。已知生成方程中的高阶项、受控消融或其他有针对性的模型比较，可以提供额外机制证据；SPT 节点本身不能取代这些证据。

## 一个与节点最优分割相容的解析反例

令 X1、X2、X3 为独立均匀二元源，固定共同目标为

\[
\boldsymbol{y}=(X_1\oplus X_2,\;X_2\oplus X_3).
\]

每个输出只由两个源产生，没有直接依赖三个源的输出机制。以二为底计算信息量，有

\[
I(X_1,X_2,X_3;\boldsymbol{y})=2,\qquad
I(X_i;\boldsymbol{y})=0,\qquad
I(X_i,X_j;\boldsymbol{y})=1\;(i\ne j).
\]

因此任意根二分 `{i} | {j,k}` 都有局部 Syn = 2−0−1 = 1。其子节点的两源贡献也为 1，树上总量为 2。所有候选根二分的分数相同，所以这一现象在全枚举最优节点搜索下仍存在，不依赖错误分割或谱近似。

本例反驳的是“正的三源节点推出不可约三体生成机制”。它不用于断言所有多源 PID 定义的三源原子在本例中均为零；信息分解阶数与生成方程阶数还需分别定义。

## 当前稿件及推荐位置

1. **正文 Results → Partial Effective Information Decomposition**：在以 `SPT uses this additive identity to construct a hierarchy` 开头的段落末尾，明确节点大小与机制阶数的区别。现句 `associated with specific variable groups and interaction orders` 建议改成 `indexed by source-group size along the selected hierarchy`。不要只在 Discussion 末尾限定。
2. **Methods → Synergy Partition Tree (SPT)**：在闭合式 `eq:greedy_closure` 后，调整现有 `The order of the SPT synergy ... is defined as |A|` 一段。该段已经说明不是 `Möbius-inverted pure-order PID atoms`，但还应直接说明节点大小不识别不可约 k 体机制，且“源”在应用中可能是 ROI、网络或完整历史块。
3. **附录 S4.1 → Node optimization and exact search**：在 `eq:supp_spt_score` 的分割分数／残差定义及其解释后，增加一段节点解释，并给出上述反例。S3.3 的加和证明不需重写。
4. **结果和图注**：将仅按节点大小汇总的 `interaction order`／`order mass` 明确称为 parent-node size／SPT contribution by parent-node size。当前 UniCM 52.9% 应写成“归属于包含 3–5 个源块的 SPT 节点的贡献”，不解释成纯三至五体机制的比例。部分 HCP 图注及脑实验方法已经使用 parent-node size 或注明不是纯阶原子，保留这些限定并统一正文用法即可。

正文可用的简短说明：

> Here, node size k counts the source blocks in the parent group. Local Syn quantifies joint information across its two children under the selected partition and does not, by itself, identify an irreducible k-body mechanism.

回应意见时宜承认这一区分，并解释修订将明确 SPT 的对象是分层信息组织，同时保留树上非负加和的理论结论。不要通过把节点大小重新命名为“阶数”来回避机制识别问题。

## 核验来源

本轮重新查询 Zotero 父项 `R68U4QXB`，核对题名 *Revealing hierarchies of synergistic causal interactions in complex systems*，重新读取正文 `GRX5IJJQ`（19/19 页）及附录 `UTEBTIWV`（34/34 页）的相关全文。正文 Methods 的节点式／闭合式在该附件中为式 (11)／(12)；附录相关位置为 S3.3、S4.1 及 S9.4 的节点解释。附件无明确稿件版本或日期，不能以元数据日期判定版本；同时直接重新读取了在线 `main.tex` 的首次 SPT 说明、Methods 末段及当前 52.9% 段落，并读取 `supplementary.tex` 的 S4.1。以上在线位置以源标签和段落开头定位，不依赖附件页码或旧缓存。
