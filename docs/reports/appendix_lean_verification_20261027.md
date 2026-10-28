# 附录数学推导：第一批 Lean 核验

核验日期：2026-10-27。范围为现行附录 S1–S4 的主要分解恒等式、层次可加性、两源 PID 条件推理及一个谱排序反例。初轮核验未修改论文和实验实现。

后续文字修正（同日）：按作者要求，已在在线附录 S4.2 将变号不变性限定于坐标互异的特征向量，并明确并列坐标采用原始源顺序。保留旧句注释，新句标红；附录编译 Errors 0、Warnings 1（原有 showhyphens 警告），已核对 PDF 第 8 页。实验实现及证明文件未改动。以下源码摘录与反例记录保留修订前的核验对象。

## 结果

已经在本地 Lean 4.29.0 中编译 35 条命名命题（包括辅助引理），另编译 6 条反例相关命题。两份文件共 41 条命题，编译无错误、无警告。逐条检查了公理依赖；没有未完成证明，也没有自定义公理，依赖不超出 Lean 的标准逻辑公理。

**已形式化的代数恒等式与条件推理有效；发现 S4.2 一句关于特征向量变号的无条件说明不成立。** 非负性证明的概率论基础尚未从分布、对数和积分定义端到端形式化，因此不能把本轮结果称为“附录所有数学推导均已完全验证”。

## 源码和版本

- Zotero 父条目：`R68U4QXB`，题名核实为 *Revealing hierarchies of synergistic causal interactions in complex systems*。
- 本轮重新列出并读取正文 `GRX5IJJQ`、附录 `UTEBTIWV` 的相关全文。两份附件没有明确稿件版本/日期，且附录保留旧 LC 证明，不能用作当前两源证明的底稿。
- 实际核验使用同日从 [Overleaf 项目](https://www.overleaf.com/project/6a52528e88192eff2e574fb8) 下载的 `supplementary.tex`。在线 S2 已采用 N、Identity 和 PLM，并给出相容性构造；S3.1 已采用复制源对的目标处理证明。
- 同次下载的 33 页附录 PDF 用于核对下面的公式编号。源文件及摘录的 SHA-256 见 [版本记录](../../formal/peid_appendix/source/provenance.json)。后续修订应重新对照源码，不能沿用本次编号而不核实。

## 已验证的对应关系

| 当前附录位置 | Lean 命题 | 本轮保证 |
|---|---|---|
| S1.1，式 S5、S6 | `normalized_determinism`、`determinism_degeneracy` | 归一化和非归一化两种 EI 分解的代数关系正确；归一化明确要求参考熵非零 |
| S1.3，式 S15 | `xi_singleton`、`ei_budget` | 单源减法定义下，标量单源的 Xi 为零，整体 EI 预算闭合 |
| S1.3，式 S16、S17 的排列视角 | `marginal_telescope`、`averaged_marginal_efficiency` | 每条完整加入顺序的增量望远镜求和，以及归一化有限平均后的总量守恒成立；未验证阶乘子集权重与所有排列的组合计数等价 |
| S1.3 与高低阶分配 | `marginal_linearity` | 分量相加时，每次边际增量也相加；不同分量分别归一化后的百分比不属于这条可加性结论 |
| S2，式 S26、S27 | `witness_pid_nonnegative` 等 | 给定普通互信息的非负性、数据处理不等式及共信息恒等式，min 型相容性构造产生的四个 PID 分量均非负 |
| S2，式 S21、S23–S25 的构造检查 | `witness_symmetry`、`minTwo_bounds`、`witness_identity`、`witness_source_inclusion`、`witness_target_processing` | 构造的源对称性、单调性界、Identity、源包含等号和 PLM 的数值推理成立；概率论前提逐条写在定理参数中 |
| S3.1，式 S29 | `two_source_information_identity`、`independent_two_source_identity` | 七个有限熵值展开后的恒等式，以及代入独立性后的消项正确 |
| S3.1，式 S30–S32 | `redundancy_zero`、`pid_correspondence` | N + 复制源对上的 PLM + Identity + 源独立性推出零冗余；代回 PID 恒等式得到两个唯一项及协同项 |
| S3.2，式 S34、S36、S37 | `sumOn_sub`、`entropy_cancellation` | 任意有限源数下，独立熵项消去后的条件总相关表达式正确 |
| S3.2，式 S38、S39 | `xi_nonnegative_from_entropy`、`xi_upper_bound` | 给定条件熵次可加性和单源 MI 非负性，正确推出下界与上界；尚未从 KL 定义重新证明这些基础事实 |
| S3.3，式 S42、S43 | `hierarchical_refinement` | 在分区块枚举的任意位置，把一个块换成其细分块后，残差可加性成立 |
| S3.3，式 S44 | `binary_xi_residual`、`binary_hierarchy` | 二分解中的单源项正确抵消，Xi 残差等于对应 EI 残差 |
| SPT 的逐层闭合关系 | `tree_budget` | 对任意有限二叉树，根值等于叶预算和全部内部残差之和；分组 E/I 叶的预算允许非零 |
| S4.1，式 S45、S46 | `score_residual_order` | 固定父节点时，最大化子节点总分等价于最小化残差 |

形式化采用有限列表枚举源和分区块。层次恒等式对任意这样的枚举都成立，因此也覆盖无重复、互不相交的合法分区。数值类型是任意满足相应条件的交换环/有序环/域，没有把连续信息量改成整数或浮点样本来代替一般证明。

## 发现的问题：S4.2 的变号不变性缺少条件

当前原文为：

> Eigenvector sign does not affect the unordered candidate set because reversing an order exchanges every prefix with its complement.

“把一条排序完全倒序”确实不改变无序前缀分割集合。但是，**特征向量变号后再稳定排序，不一定等于把原排序完全倒序**：相同坐标的内部顺序仍保留原始源顺序。

一个四节点精确例子取

\[
\boldsymbol{v}=(-1,-1,1,1)^{\mathsf T},\qquad
\mathbf W=\begin{pmatrix}
0&3&1&1\\3&0&1&1\\1&1&0&3\\1&1&3&0
\end{pmatrix}.
\]

全部加权度为 5，归一化拉普拉斯满足
\(\mathbf L\boldsymbol v=(4/5)\boldsymbol v\)。矩阵对称，非对角元素严格为正，向量非恒定。这些有限数值事实均已用 Lean 验证。

用 1 起始编号解释排序：

| 方向 | 稳定升序 | 无序前缀分割中的单节点侧 |
|---|---|---|
| \(\boldsymbol v\) | 1, 2, 3, 4 | {1}、{4} |
| \(-\boldsymbol v\) | 3, 4, 1, 2 | {3}、{2} |

两者都包含 {1,2}|{3,4}，但其他两个分割不同。`different_unordered_candidate_sets` 验证了这个差异。

为排除“合并其他排序后差异必然消失”的可能，另构造了一个连接的 16 节点正权重矩阵。用有理数逐项验证了其完整 Hadamard 特征基、互异特征值和所有加权度；将前四个非恒定谱排序、average-linkage 排序与原始顺序共同纳入候选池，仅改变第一谱方向的符号即可使候选数由 85 变为 84。分割 {0,2}|其余节点在变号后出现、变号前不存在。该有限候选池差异由 Lean 的 `full_pool_sign_change` 验证；矩阵与特征基、辅助排序的重现由 `check_spectral_sign.py` 检查，未把 Python 检查冒充为 Lean 的矩阵谱定理证明。

仓库 `scripts/spt.py` 的 `spectral_candidate_selector` 在第 484–486 行使用 `eigh` 和 `np.argsort(..., kind="stable")`，因此原文的并列值问题与实现相关。**这不构成现有实验树错误的证明**：本轮没有重跑任何论文实验，也没有量化当前缓存的候选是否因符号约定而变化。

最小文字修正建议为：

> Sources are stably sorted by each retained eigenvector, with ties resolved by the original source order. When all coordinates are distinct, changing an eigenvector's sign reverses its ordering and leaves the unordered prefix cuts unchanged.

如果以后要保证实现严格不受特征向量符号约定影响，可先统一每条向量的符号再排序，例如令绝对值最大分量为正，并固定并列时的选取顺序。这属于算法实现变更，需另行核对对既有结果的影响；本轮未采用这一变更。

## 非负性和 PID 核验的边界

1. `entropy_cancellation` 验证的是有限熵值之间的等式。源独立性如何给出熵可加性、条件熵次可加性如何从 KL 非负性推出，仍是本轮明确保留的概率论输入。
2. `redundancy_zero` 把 N、复制源对上的 PLM、Identity 和独立性全部列为参数，未把“冗余为零”直接假定为前提。随后 `pid_correspondence` 检查由零冗余到唯一项及协同项的推导。
3. `witness_pid_nonnegative` 等验证了 S2 相容性构造的代数/序关系步骤，没有把普通 MI 的数据处理、复制目标恒等式等基础事实重新从联合分布证明出来。其适用范围仍按现行文字限定为有限字母表；连续情形的 PID 对应保持原文所写的条件性陈述。
4. 连续情形仍需在下一层形式化中处理条件分布、可测性、绝对连续性以及信息差/微分熵表达式的合法性。本轮没有发现需要改动现行核心恒等式的依据。

尚未覆盖：精确 Shapley 子集权重的全维组合证明、连续积分与 TM 换元/雅可比证明、谱排序的一般矩阵理论、分割及表示数量的组合证明，以及后文各动力学模型和数值估计器的完整形式化。定义一个模型方程和证明该模型的经验结果，是不同的验证任务。

## 文件与复查

- [主证明](../../formal/peid_appendix/Appendix.lean)：35 条命名命题，包含辅助引理。
- [精确反例](../../formal/peid_appendix/Counterexamples.lean)：6 条命名命题。
- [验证输出](../../formal/peid_appendix/verification.json)：逐条公理依赖、版本、源码校验值和编译状态。
- [谱排序反例重现](../../formal/peid_appendix/check_spectral_sign.py)：精确有理数特征基检查与实际候选池构建。

在仓库根目录运行：

```sh
python formal/peid_appendix/verify.py
python formal/peid_appendix/check_spectral_sign.py
```

使用已安装的 Lean 4.29.0，没有安装 Mathlib、没有更改系统默认 Lean。初轮核验之后，仅按作者要求修正了上述 Overleaf 句子，实验代码未改动。
