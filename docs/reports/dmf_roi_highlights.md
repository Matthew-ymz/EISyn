# 正文 SPT 高频 ROI 组合标注建议

2026-10-24；使用冻结缓存，仅修改绘图显示，不重跑模拟、EI/Syn、树搜索或频率筛选。

![局部标注预览](../../fig/dmf_schaefer100/dmf_main_roi_highlights_preview.png)

当前正文示例为平均 SC、G=1.3、seed 4 的树。高频统计来自 93 个个体 SC 模型各自的转变窗口，计数规则为六个 G 点中至少四个满足原有背景阈值及种子重复要求。该统计不是平均树在 93 人中的复现率，也不是协同信息强度或显著性。

| 编号 | ROI 组合 | 满足原入选准则的人数 | 当前树完整子树 | 现有标签对应的功能网络 |
|---|---|---:|---|---|
| — | {10,11,12} | 55/93 | 否 | 左侧 SM |
| A | {59,60,61} | 41/93 | 是 | 右侧 SM |
| D | {25,26,42} | 29/93 | 是 | 左侧 SA/VA 的额盖/岛叶分区 + DMN 前额叶分区 |
| C | {62,63,64,68} | 31/93 | 是 | 右侧 SM + DA 后部分区 |
| — | {10,11,12,25} | 15/93 | 否 | 左侧 SM + SA/VA 额盖/岛叶分区 |
| B | {59,60,61,76} | 13/93 | 是 | 右侧 SM + SA/VA 额盖/岛叶分区 |

## 图示建议

正文树仅加 A–D 四个定位标记。保留原树的叶序、成员、原始二叉拓扑及所有信息值，停止对这些节点的显示压缩。局部放大展示 A⊂B、C、D 三处分支即可；完整六组频率表及解剖标签放附录。预览保留 100 个 ROI 叶和全部 99 个原始内节点的表示。当前是可审阅局部预览，尚未替换 Overleaf。

## 可支持的解释

最清晰的是 A⊂B：模型内的右侧感觉运动三元协同组属于更大的感觉运动—显著性/腹侧注意协同组，直接体现网络内局部组织与跨网络高阶组织的嵌套关系。这里的父子关系是信息分解中的层级，不是时间上先后招募脑区或定向信号传播。

C 提供感觉运动与背侧注意分区的跨网络协同例子，可作为感觉运动—注意协调的候选组织解释。D 将额盖/岛叶分区与默认网络前额叶分区联系起来，可作为显著性/腹侧注意与默认网络协调的候选解释。已有 fronto-insular 网络研究提供背景，但没有验证这三个特定 ROI 的机制、方向性、认知任务作用或行为意义。

因此正文应强调“可重复出现的模型内协同组合具有可解释的功能网络组织，并形成嵌套层级”，而不是将人数排名等同于因果强度或赋予确定的认知功能。

候选英文句（需先确认原始 SC 行序）：

> The highlighted branches illustrate a nested organization of model-based synergy: the right somatomotor triplet {59,60,61} is embedded within a larger coalition including a fronto-opercular/insular parcel {59,60,61,76}. Additional recurrent groups span somatomotor and dorsal-attention parcels, or salience/ventral-attention and default-network parcels, suggesting candidate links between within-network integration and cross-network coordination.

## 解剖与证据边界

原 SC 归档没有显式脑区标签。preparation_summary.json 把行序记作 inferred，同序与外部 SC 的边相关为 0.44956，优于 2,000 次随机置换；这是支持性证据，尚非逐行标签确认。官方 Schaefer-100 名称已与本地标签逐个核对；这不能独立证明原 SC 行序。解剖解释因此暂为条件性解释。

原始模型是静息类 DMF 动力学，没有注意、运动、切换等任务操作或相应行为测量。文献功能只能帮助解释网络背景。

## 来源

- 频率：results/dmf_schaefer100/sc_spt_participant_frequency/summary.json。
- 平均树：results/dmf_schaefer100/xi_hierarchy_tree/examples/wide_G1000/summary.json 中 seed 4、G=1.3 的 cache。
- 逐项检查：tmp/dmf_roi_highlight_20261024/preview_audit.json。
- Zotero 本次重新核对 P6UJCVG8，标题 Emergent hierarchical organization of causal interactions in complex systems；读取正文 DXGC7JEA 的脑 DMF 部分，补充 MWIWKSVG 的 S5、S12.2.1（SPT 和 DMF 方法）。附件无明确修订日期，元数据 2026-10-04 不作为新版本日期。当前正文树与新增个体频率以现有冻结缓存为准。未宣称旧附件已经覆盖新增人群分析。
- [Schaefer 等，2018](https://doi.org/10.1093/cercor/bhx179)：分区方法。
- [官方 Schaefer-100 标签及质心](https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv)。
- [Yeo 等，2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3174820/)：感觉运动与注意等功能网络组织背景。
- [Sridharan 等，2008](https://doi.org/10.1073/pnas.0800005105)：right fronto-insular 与大尺度网络切换的任务/静息证据；不是本实验 ROI 组合的直接验证。

## 整张 Figure 2 排版预览

![完整排版](../../fig/dmf_schaefer100/dmf_brain_main_composite_roi_focus_preview.png)

190 × 255 mm，保持上方 a–e 曲线和命中率。中部 f 为放大至 110 mm 宽的转变区树和三个局部放大；g、h 为同一 G=1.3 的脑图和网络贡献。底部 i、j 分别压缩展示 G=0 与 G=10 的树、四视角脑图及网络贡献，各 G 的科学含义和数据保持独立。树保留 100 叶及 99 个原始内节点的表示，四个标注节点不再被压缩隐藏。相同 G 的组织证据仍归在同一视觉区域。未重估或汇总三个 G 的数据，未替换 Overleaf。

上方命中率脚注微调垂直位置以避开横轴标签；所有图例/色条边界、与数据轴重叠，以及文字超出画布的检查均通过，已查看完整导出图。高阶脑图色标、网络条形共用原比例；没有删去任一对照条件。

## 大尺度结构与 A–D 的连贯解释

2026-10-24，基于同一 G=1.3、seed-4 平均 SC 原始二叉树逐节点追溯（无模拟、信息估计或树搜索）。

- A {59,60,61} ⊂ B {59,60,61,76} ⊂ {59,60,61,75,76}（5 ROI）。
- C {62,63,64,68} 属于另一个 14-ROI 分支；它与上述 5-ROI 分支在 19-ROI 节点精确汇合。该 19-ROI 集合全部位于右半球，含 SM 8、DA 4、SA/VA 3、FPN 2、DMN 2。这个节点的局部 Syn 为 0.178037 nats（根 Xi 的 0.989636%）；整个子树 Xi 为 2.960197 nats（16.454502%）。局部节点贡献与子树累积量不可混用。
- D {25,26,42} ⊂ {25,26,42,43}（4 ROI，SA/VA 2、DMN 2），该分支与含上述 19-ROI 组织的 50-ROI 分支在 54-ROI 节点精确汇合。54-ROI 集合为左侧 18、右侧 36，含 VIS 6、SM 10、DA 11、SA/VA 8、FPN 8、DMN 11，覆盖六个网络但不含 LIM。该节点的局部 Syn 为 0.160993 nats（0.894894%）；子树 Xi 为 8.445125 nats（46.942934%）。

推荐故事：局部感觉运动协同组嵌套在感觉运动—注意组合中，再通过中尺度分支与显著性/默认等系统共同构成更广泛的跨网络信息层级。这里的“从小到大”是树的包含关系，不是时间顺序、定向因果传播或已证实的认知加工序列。

图上如需补充，可以用 E 定位 19-ROI 共同祖先、F 定位 54-ROI 共同祖先，仅标成员数和覆盖网络，避免把子树累计 Xi 当成该节点新增的 Syn。A–D 与 E/F 的角色不同：A–D 有个体窗口的筛选频率，E/F 只是当前代表树的解释性祖先。

更大节点尚未具备相同的跨个体复现证据。在三棵平均 SC 树的轻量检查中，B、C 的最小共同祖先大小分别为 9、19、21；B、C、D 的最小共同祖先为 73、54、51。不能将 19/54 的精确边界称为稳定人群模块。

原始树 99 个内节点中，91 次为单 ROI 分离，8 次两侧都至少有 2 ROI；图中虚线把连续单点分离压缩展示。大跨度横线不能据视觉宽度直接当成强协同模块，树高也不是时间。当前节点颜色尺度与大小说明应保留此区别。全部局部 Syn 和叶 Xi 闭合到根 Xi=17.990193 nats；逐项原始查询沿用既有 1e-8 bits 容差，未额外裁剪。

核验记录：tmp/dmf_roi_highlight_20261024/large_scale_ancestry.json。Zotero 本次重读同父条目和两附件相关正文、SPT 与 DMF 方法；旧稿中的 Yeo 约束树与当前无网络约束 ROI 树不同，不能用旧稿数字替代本次逐节点结果。脑区行序的 inferred 限制继续适用。

## E/F 图示与 93 人复现核对

已在完整 Figure 2 预览中用两处开放空间的折线引注标出 E（19 ROI）和 F（54 ROI），引线终点为精确原始内节点。E/F 未改变原树、叶序、Syn 颜色或数值，也未增加人群高频标记。图示仍是预览，尚未替换 Overleaf。

本次直接核对 93 人各自六个 G 点 × 三个种子的 1,674 棵既有窗口树及对应密度元数据；不重跑模拟、不估计新 EI/Syn、不搜索新树。目标完整成员在查询前由平均 SC、G=1.3、seed 4 的共同祖先固定。原始全 G 精确成员注册表独立复核了结果。

| 标记 | 成员数 | 窗口内任一次精确出现的人数 | 通过原窗口重复规则的人数 | 超背景且通过原规则 |
|---|---:|---:|---:|---:|
| E | 19 | 1/93（仅一个 G、一个种子） | 0/93 | 0/93 |
| F | 54 | 0/93 | 0/93 | 0/93 |

原规则：同一 G 至少 2/3 个种子重复，且六点个体转变窗口内至少 4/6 个 G 点满足。全 41-G 注册表中 E 也仅出现一次，F 完全未注册，因此并非只错过固定 G=1.3。两者不支持“跨个体高频完整模块”这一表述。

作为独立、探索性的边界相似性描述，逐树取与 E/F Jaccard 最高的原始非根节点，并用 Jaccard≥0.8 及相同重复/窗口规则核对，E/F 都为 0/93。这个相近匹配允许不同条件使用不同的集合，并不要求同一个近邻模块重复；不是精确身份或显著性检验。逐人 18 条件最近 Jaccard 中位数的群体中位数为 E=0.445、F=0.583，受集合大小和候选数量影响，不宜横向比较两个模块的稳定性。

检查的 165,726 个内节点及既有候选/成对/查询有效性记录均无显著负 Syn；沿用 1e-8 nats 容差，容差内负数计数均为 0，最大闭合误差 1.60e-14 nats。

这与 A–D 角色不同：A–D 来自个体频率筛选，E/F 为平均树上的解释性祖先。可用 E/F 说明该示例如何把小组组织到大尺度层级，但不能写成 93 人共享相同的 19/54 脑区模块。平均连接矩阵树不等于个体树的共识树；精确大集合匹配也比小集合更苛刻。本次没有验证大尺度功能组织的共识拓扑。

结果：results/dmf_schaefer100/roi_ancestor_recurrence/summary.json，含逐人窗口记录、原注册表计数、原文件指纹及有效性审计。可复核脚本：scripts/check_dmf_ancestor_recurrence.py。
