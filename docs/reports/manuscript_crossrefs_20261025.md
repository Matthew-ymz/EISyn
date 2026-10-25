# 正文—附录交叉引用检查（2026-10-25）

检查对象：Overleaf 项目 EI-SYN 中当前 `main.tex` 与 `supplementary.tex` 的有效正文；排除已注释删除的旧稿。检查包括标签存在性、重复标签、显式占位符、附录章节覆盖及正文图子图编号。

## 已完成的修正

- 正文完成 10 处局部修改：非负性证明改用自动章节引用；SPT 表示规模比较；五种层次基线的具体实现；Shapley 及高低阶分配；SLP 数据方法；HCP 方法；城市交通数据预处理；连续变量 EI/TM 方法；定理自动编号；UniCM 的 `add SI section operation here` 占位符。
- 附录修正 7 处脑图引用：高阶图 `g,j,m` → `g,i,j`（3 处）；组织面板 `f--n` → `f--j`（2 处）；SPT `f,i,l` → `f,i,j`；网络归因 `h,k,n` → `h,i,j`。已下载并核对当前实际图文件。
- 所有修改保留相邻旧稿注释，并标红修订。未改实验结果、数值或方法定义。
- 两个文件的源代码差异已核对，仅有上述预期修改。有效正文中的引用标签全部有定义，无重复标签。正文和附录分别编译通过，均为 Errors 0、Warnings 1；唯一 warning 为已有的 `Command \showhyphens has changed`，另有既存排版提示。

## 需要补充或同步的内容

1. **城市交通（S10.3）是最明确的方法缺口。** 当前附录把 daily 和 weekly 合并为 macro-scale block，与正文分别使用 recent、daily、weekly 三组并报告三组共同贡献的分析不同。还缺少当前 198 叶 SPT 的来源/目标表示、干预与边际化方式、EI 估计和树搜索流程。已补的数据预处理入口不意味着这套新实验协议已写完整。
2. **UniCM 地理超图（S10.1.2）建议补一个短方法段落。** 现有文字给出了冻结模型、干预输入和 TM 估计；正文图注说明展示十条最强、单目标的两源超边。但附录尚未完整说明候选超边集合、跨 checkpoint 平均与筛选排序的顺序及模式的地理位置映射规则。不能仅凭图注反推实现。
3. **Discussion 有已删除实验的残留描述。** 句子 “In the mixed-order Kuramoto model, this reorganization appeared as a transition from separated pairwise and triadic modules to a nested cross-module hierarchy as coupling increased.” 对应原先改变团间耦合的实验；当前 S6 是三个独立基准，S8 是全局耦合 Kuramoto 同步转变，均不直接支持这句模块重组描述。建议删除，或改为 “Across the Kuramoto and Glauber benchmarks, SPT recovered separated and nested planted modules despite differences in interaction topology and dynamics (Fig.~\ref{fig:synergy_overview}d; Supplementary Section~\ref{app:six_node_hierarchy_comparison}).” 本轮未改动此结果陈述。
4. **气候应用开头还有作者占位标记。** “... across prediction horizons.???????” 是源文件中的字面问号，不是 LaTeX 未定义引用；本轮保留，需作者决定是否删除或补注。

## 尚未在正文提及的独立分析

**S10.1.1 内的 “Node-level causal effects and higher-order information contributions” 段落**（`app:slp_node_importance`）及其 **Supplementary Fig. 8**（`fig:slp_node_importance_comparison`）没有正文入口。S10.1.1 的总体数据方法已被引用，但正文并未讨论 ACE/HACE、ACS/HACS 的节点重要性结果。若希望保留为一项支撑结果，可在 SLP 扩展分析处加一句结果概述并引用此段及图；不建议为增加引用覆盖而硬塞进正文。

## 全部附录章节覆盖清单

“未单独引用”不等于没有关联：上级方法章节的引用可以覆盖其算法子节，而一个总标题也可通过相关下级小节进入正文。下表完整区分这几种情况。

| 章节 | 标题 | 正文引用状态 |
|---|---|---|
| S1 | Effective information: definition and evaluation | 通过下级小节引用，未单独引用此总标题 |
| S1.1 | Discrete variables and exact evaluation | 正文直接引用 |
| S1.2 | Continuous variables and transport-map estimation | 正文直接引用 |
| S1.3 | Exact Shapley attribution of grouped synergy | 正文直接引用 |
| S2 | PID axiomatic background | 正文直接引用 |
| S3 | Proofs of the PEID properties | 通过下级小节引用，未单独引用此总标题 |
| S3.1 | Proof of the two-source PEID--PID correspondence | 正文直接引用 |
| S3.2 | Proof of the non-negativity of integrated effective information | 正文直接引用 |
| S3.3 | Proof of hierarchical additivity | 正文直接引用 |
| S4 | Synergy Partition Tree implementation and split optimization | 正文直接引用 |
| S4.1 | Node optimization and exact search | 通过上级章节覆盖，未单独引用 |
| S4.2 | Spectral candidates for large coalitions | 通过上级章节覆盖，未单独引用 |
| S4.3 | Representation size of PEID and full PID | 正文直接引用 |
| S5 | Canonical nonlinear dynamical systems | 正文直接引用 |
| S5.1 | Coupled standard map | 通过上级章节覆盖，未单独引用 |
| S5.2 | Wilson--Cowan refractory map | 通过上级章节覆盖，未单独引用 |
| S5.3 | Kuramoto active-rotator phase model | 通过上级章节覆盖，未单独引用 |
| S5.4 | Controlled H\'enon-inspired response model | 通过上级章节覆盖，未单独引用 |
| S5.5 | Ikeda optical-cavity map | 通过上级章节覆盖，未单独引用 |
| S5.6 | Nicholson--Bailey host--parasitoid map | 通过上级章节覆盖，未单独引用 |
| S6 | Three six-node benchmarks for hierarchical recovery | 正文直接引用 |
| S7 | Comparison Methods | 正文直接引用 |
| S7.1 | Whole-minus-sum information (WMS) | 正文直接引用 |
| S7.2 | Minimum-mutual-information partial information decomposition (MMI-PID) | 通过上级章节覆盖，未单独引用 |
| S7.3 | Synergistic--unique--redundant decomposition (SURD) | 通过上级章节覆盖，未单独引用 |
| S7.4 | MLP-based SHAP interaction (MLP+SHAP) | 通过上级章节覆盖，未单独引用 |
| S7.5 | Hierarchical comparison methods | 正文直接引用 |
| S8 | Integrated-effective-information detection of the Kuramoto synchronization transition | 正文直接引用 |
| S9 | Supplementary results | 通过下级小节引用，未单独引用此总标题 |
| S9.1 | Forecast-scale reorganization in global sea-level pressure | 正文直接引用 |
| S9.2 | Task engagement reconfigures whole-brain synergy | 正文直接引用 |
| S9.3 | Low-order ROI attribution below 50 ROI blocks | 正文直接引用 |
| S9.4 | High-order synergistic causal interactions versus structural rankings | 正文直接引用 |
| S10 | Real-world datasets and preprocessing | 通过下级小节引用，未单独引用此总标题 |
| S10.1 | Earth-system datasets and models | 通过下级小节引用，未单独引用此总标题 |
| S10.1.1 | Global sea-level pressure | 正文直接引用 |
| S10.1.2 | Pretrained UniCM transition model | 正文直接引用 |
| S10.2 | Brain data and preprocessing | 通过下级小节引用，未单独引用此总标题 |
| S10.2.1 | Schaefer-100 structural connectome and dynamic mean-field model | 正文直接引用 |
| S10.2.2 | HCP Schaefer-1000 functional data and state comparisons | 正文直接引用 |
| S10.3 | Urban mobility data and preprocessing | 正文直接引用 |

未被正文单独引用的算法/模型子节集中为：**S4.1–S4.2、S5.1–S5.6、S7.2–S7.4**。它们均通过上级章节 S4、S5、S7 获得正文入口，无需为了逐节引用额外堆叠编号。

总标题 **S1、S3、S9、S10、S10.1、S10.2** 没有作为单独目标引用，但相关下级章节均有入口。S9 是无独立标签的汇总标题，其四个结果子节都已被正文引用。

## 稿件来源与界限

按项目约定核对了 Zotero 当前论文父条目 `R68U4QXB`，标题为 *Revealing hierarchies of synergistic causal interactions in complex systems*；附件 `GRX5IJJQ`（正文）及 `UTEBTIWV`（附录），附件录入时间为 2026-10-25。附件没有独立、明确的版本号或稿件日期，因此不以元数据编辑时间推断稿件版本；本次交叉引用与节号以实时下载的 Overleaf 源码和重新编译的 PDF 为准。没有据此声称完成所有实验实现或科学结论的全面审计。
