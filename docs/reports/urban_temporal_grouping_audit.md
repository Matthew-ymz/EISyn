# 交通实验的时间分组与正文一致性

核对日期：2026-10-26。本次只修订在线稿件描述，未修改计算方法、缓存数值或图像，未重跑实验。

## 合并的原因与适用范围

仓库记录和实际计算表明，regional finite-amplitude 分析采用 `recent | (daily, weekly)`，目的是比较近期变化与周期背景的两源协同，并与空间两源比较对应。daily 和 weekly 使用相同的经验窗口索引，保留二者在周期块内的经验依赖；recent 使用独立索引。两块各压缩为两个主成分后估计条件互信息。依据为 [区域干预构造](../../scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py:227)、[模型干预](../../scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py:330)及 [既有实验记录](nyc_taxi.md:64)。这是现有设计的功能解释，未发现证明该分组优于所有三组方案的比较实验。

MGSTN 本身始终有 recent、daily、weekly 三个分支；城市 SPT 也没有预合并 daily 和 weekly。它保留 66 区 × 3 时间尺度，共 198 个源块，以全城下一小时 132 维 inflow/outflow 为共同目标。时间块约束树比较三种合法根二分，而非固定指定 daily–weekly 配对；九个缓存条件均选中 `daily | (recent, weekly)`。依据为 [独立干预及表示](../../scripts/analyze_nyc_taxi_mgstn_spt.py:143)和 [时间块搜索](../../scripts/analyze_nyc_taxi_mgstn_spt.py:376)。

## 百分比的实际含义

在线正文原将 83.9%、82.3%、77.8% 称为三尺度共同贡献占比。实际 [绘图入口](../../scripts/plot_nyc_taxi_multiscale_spt_composite.py:220)调用 [比例绘制函数](../../scripts/plot_nyc_taxi_mgstn_ei.py:170)，读取 `results/nyc_taxi_mgstn_ei/finite_quadratic_tm/quadratic_tm_full_summary.json`。缓存定义及 [汇总代码](../../scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py:430)给出的比例是

\[
\frac{\sum_i S_i^{\mathrm{time}}}{\sum_i S_i^{\mathrm{time}}+\sum_i S_i^{\mathrm{space}}},
\]

先在每个检查点计算，再取三个检查点的均值。这里的时间项是 recent 与 daily–weekly 联合块的协同，不能解释为排除了所有两尺度作用后的纯三尺度贡献。三个百分比与缓存一致，修订只纠正其解释。

## 在线修订

- 附录 S10.3：先定义三个时间来源，再分别说明区域二分比较、有限幅干预及估计、198 源块城市 SPT；补齐空间比较及百分比分母。daily–weekly 联合块仅用于区域统计。
- 正文交通结果、图 4c 图注和相关结论：区分两源区域比较与三尺度 SPT；移除由这些比例推出“三尺度共同贡献占主导”的表述。
- 摘要：改为 recent 与 weekly 更紧密、daily 相对可分的层级发现。
- 保留精确旧文注释，新增文字标红。摘要 Weiwei Gu 评论的大范围高亮保持附着于修改后的正文；模型干预评论的 `for interventions on the frozen models` 原锚点未删除，评论内容与未解决状态保留。未新增回复或解决评论。

## 文稿来源与核验边界

本轮重新读取 Zotero 父项 `R68U4QXB`，核对题名 *Revealing hierarchies of synergistic causal interactions in complex systems*；当前附件为正文 `GRX5IJJQ`（19/19 页）和附录 `UTEBTIWV`（34/34 页），相关位置为正文城市交通部分及附录 S10.3／PDF 第 27–28 页。附件无明确版本或稿件日期，不能以元数据修改时间认定稿件版本；本次修订以重新读取的在线源码及实际计算、缓存定义为准。

核对范围为分组、抽样、目标、估计方法、比例分母和层级搜索，未声称验证了整个交通实验的全部方法假设，也未将区域二分分析改成新的三源实验。

在线正文（18 页）与附录（33 页）均编译为 Errors 0、Warnings 1，保留已有 `Command \showhyphens has changed.` 提示，没有未定义引用或文献警告。已核对正文第 11 页和附录第 26 页，修改标红、公式行内排版正常，后续参考文献恢复黑色。摘要和模型干预的原评论高亮均已重新打开并视觉核对，锚点、消息和未解决状态保留。

[正文核验截图](../../output/urban_temporal_consistency/main_page11.jpg)；[附录核验截图](../../output/urban_temporal_consistency/overleaf_urban_methods.jpg)。
