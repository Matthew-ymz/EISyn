# 两源冗余零值定理修订

本修订保留原结论，删除冗余的左链式法则（LC），改用非负性、Identity 和概率左单调性（PLM）。已完成本地 LaTeX 修订及编译，并于 2026-10-26 逐处合并到在线 Overleaf。

## 新证明

令两个源为 $U,V$，目标为 $Y$，并令中间目标为联合源的副本 $\boldsymbol{t}=(U,V)$。任意条件分布 $p(y\mid u,v)$ 都给出马尔可夫链 $(U,V)\to\boldsymbol{t}\to Y$。于是

$$
0\leq R(U,V;Y)\overset{\mathrm{PLM}}{\leq}R(U,V;(U,V))
\overset{\mathrm{Id}}{=}I(U;V)=0.
$$

因此，当 $U\perp V$ 时，冗余为零。该论证覆盖确定性和随机目标通道，也不限于二值变量。由两源 PID 恒等式，唯一信息分别等于两个单源互信息，协同等于联合互信息减两个单源互信息之和，即 $I(U;V\mid Y)$。

证明链为：复制联合源 → 对目标施加随机通道 → PLM 给出冗余上界 → Identity 将上界化为源间互信息 → 独立性使上下界同为零。

附录另给出有限字母表下的相容性构造：

$$
R_*(X_1,X_2;Y)=\min\{I(X_1;X_2),I(X_1;Y),I(X_2;Y)\}.
$$

它满足列出的两源性质，且诱导的四个 PID 分量均非负。该构造用于证明这组条件非空，不作为新的数值估计器。结论表述为适用于所选性质的冗余函数类，而非所有 PID 定义。多源 $\Xi$ 的非负性和层次可加性仍由原有信息恒等式证明，不依赖冗余的链式法则。

## 改动位置

- `supplementary.tex`：整体替换 S2、S3 引言和 S3.1；调整 S3.2 两处 PID 对应关系。新零冗余证明为式 S31，两源协同结论为式 S33。
- `main.tex`：引言、图 1a 图注、Discussion、命题 1 条件及命题后解释。正文式 6 保持不变。
- `sample.bib`：新增 `Rauh2017ExtractableSharedInformation`；复用已有 Harder、Bertschinger 和 Williams–Beer 条目。

所有被替换的原文均逐行保留为注释，新文字用现有红色修订样式标记。其余正文和实验结果未改动。

## 引文依据

1. Harder, Salge & Polani (2013), *Bivariate Measure of Redundant Information*, Physical Review E 87, 012130. [DOI](https://doi.org/10.1103/PhysRevE.87.012130)。已读[全文](https://arxiv.org/html/1207.2080v3) §III.1、式 5，支持 Identity。
2. Rauh, Banerjee, Olbrich, Jost & Bertschinger (2017), *On Extractable Shared Information*, Entropy 19(7), 328. [DOI](https://doi.org/10.3390/e19070328)。已读[全文](https://arxiv.org/html/1701.07805v3) §III、式 7 后的 PLM 定义，明确覆盖随机目标通道。引用的是该性质，未声称该文提出的具体冗余量同时满足本文全部性质。
3. Bertschinger et al. (2013), *Shared Information—New Insights and Problems in Decomposing Information in Complex Systems*. [DOI](https://doi.org/10.1007/978-3-319-00395-5_35)。已读[全文](https://arxiv.org/html/1210.5902v1) §4，用于说明确定性左单调性背景；不再采用其中的 LC。

## 版本与核验

- 已读取用户引用的“深度审稿”任务（`01a17730-3c9f-7272-bf05-cc5c4564b3cb`）。
- 本次重新查询 Zotero 父条目 `R68U4QXB`，核实标题为 *Revealing hierarchies of synergistic causal interactions in complex systems*。读取正文附件 `GRX5IJJQ` 和附录附件 `UTEBTIWV` 的相关全文；附件加入时间为 2026-10-25，两份 PDF 均未显示明确稿件版本或日期，不能仅凭元数据编辑时间认定最新版。
- 可编辑底稿来自该审稿任务在 2026-10-26 导出的 Overleaf 项目。重点核对了正文命题 1/式 6、附录 S2 及 S3.1；旧附录 S20–S27 为公理背景，S28–S36 为原两源证明。
- 已完整编译正文（18 页）和附录（33 页），新引文及全部交叉引用均解析；标签无重复。修订后的 S2、S3.1、S3.2、命题及图注已目视检查，红色标记与公式显示正常。日志仍有目录、图 1 所在页及其他位置的版面警告，未作为本次改动扩展处理。
- 在线独立浏览页显示 Restricted；已登录页面只能通过共享输入操作。在线源文和 Review 锚点尚未逐段重读或修改。应用本修订前应对照最新在线文字逐段合并，不能用此快照覆盖整份在线文件。

## 交付文件

- `source/main.tex`、`source/supplementary.tex`、`source/sample.bib`：完整修订源码，依赖原项目的其他文件。
- `supplementary_replacement.tex`：新的 S2、S3 引言及 S3.1，可独立审阅。
- `replacements.json`：八组精确原文与替换文本。
- `overleaf.patch`：相对于本次底稿的三文件补丁。
- `references.bib`：唯一新增的文献条目。
- `two_source_pid_sources.zip`：以上三份完整源码及原项目的 `topology.bib`，方便保留和对照。
- [修订页面预览](../../output/pdf/two_source_pid_revision.pdf)：正文第 2、3、12、14、15 页及附录第 5–7 页。

## 在线同步（2026-10-26）

- 已读取“正文精写2”最新修订，并在已登录浏览器的新项目页逐处同步；未整份覆盖在线文件。
- `main.tex` 的引言、图 1a 图注、Discussion、命题 1 条件及其解释已采用新公理表述；`supplementary.tex` 的 S2、S3 引言、S3.1 和 S3.2 两处限定已同步。新增 `Rauh2017ExtractableSharedInformation`，旧文逐行注释保留。
- Brain 的最终精简段落原已在线，无须重复写入；Earth 八个月结果段已采用“印太嵌套核心、三至五源贡献 52.9%”的版本，并保留原有耦合背景文献和动态图引用。
- 正文（18 页）与附录（33 页）在线编译均为 Errors 0、Warnings 1，唯一一般警告为 `Command \showhyphens has changed.`，无未解析引用。另保留原有排版类日志；未扩展为全文排版清理。
- 已在编译输出中核对红色修订、附录 PLM/相容性构造及新证明。现有未关闭 Review 仍为正文 39 条；Earth“这样写会不会更突出高阶来？”原线程已回复，未关闭任何评论。
- 本轮再次核对 Zotero 父项 `R68U4QXB` 题名与当前正文 `GRX5IJJQ`、附录 `UTEBTIWV` 的全文及附件清单。两份附件无明确稿件版本或日期；本次按被指定聊天中已经核验的修订合并，不据元数据时间宣称它们是新日期的稿件。
- `source/` 和压缩包仍是先前编译通过的本地修订底稿，并非本次全部在线内容的重新导出；在线项目的后续编辑应重新读取实时源码。
- 在线预览截图位于 `output/overleaf_sync/`。
