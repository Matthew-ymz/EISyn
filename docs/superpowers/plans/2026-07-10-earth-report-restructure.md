# Earth 报告重构实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `earth.md` 重构为 Runge 与 UniCM 并列、以高阶信息机制为主线的学术研究报告。

**Architecture:** 只修改目标 Markdown，不改实验结果与图片。先建立分层目录和统一方法定义，再分别组织 Runge 的尺度依赖超边证据与 UniCM 的 $\Phi^{\mathrm{EID}}$ 计算/分解，最后统一讨论、限制和证据索引。

**Tech Stack:** Markdown、LaTeX 数学公式、现有 PNG 实验图与 CSV/JSONL 结果。

---

## Chunk 1: 文档重写与校验

### Task 0: 固化当前工作区证据清单

**Files:**
- Read: `docs/reports/earth.md`
- Create: `/tmp/earth-before.md`（仅作本次校验，不纳入仓库）

- [ ] 在改写前复制当前工作区版本到 `/tmp/earth-before.md`，不得用 `HEAD` 替代，因为目标文件已有用户修改。
- [ ] 从该副本提取并记录 17 张图片、全部 Markdown 表格、13 条参考文献、关键数值、负面结果和限制；改写后逐项核对，预期零遗漏。

### Task 1: 建立论文式结构

**Files:**
- Modify: `docs/reports/earth.md`

- [ ] 将平铺目录改为“章—节—子节”结构。
- [ ] 新增研究问题、主要发现、统一方法与综合讨论。
- [ ] 保持 Runge 与 UniCM 的结果正文字符数比例在 `0.80-1.25`；用按二级标题切片后的 `wc -m` 统计，预期比例落入该区间。
- [ ] 按章节为每个展示公式添加 `\tag{章序号.公式序号}`，并在首次解释公式时使用“式（x.y）”交叉引用；结构调整后同步更新编号。

### Task 2: 重写 Runge 结果

**Files:**
- Modify: `docs/reports/earth.md`

- [ ] 将 ACE/ACS 压缩为节点级背景证据。
- [ ] 以 $H=1,10,60$、跨尺度重复超边和四条强制 TM 曲线为核心组织结果。
- [ ] 明确短期峰值、中期峰值、长尺度平台和长期增强四类行为。
- [ ] 提炼尺度依赖、空间跨度和候选遥相关的地球科学启示，并保留估计限制。

### Task 3: 重写 UniCM 结果

**Files:**
- Modify: `docs/reports/earth.md`

- [ ] 将整体 EI、单源 EI 和二源 Syn 降为 $\Phi^{\mathrm{EID}}$ 的基线与解释证据。
- [ ] 完整说明 $\Phi^{\mathrm{EID}}$ 的定义、计算口径、非负截断和 lead 8 峰值。
- [ ] 完整说明贪婪层级分解、闭合关系、主导模块和非唯一性。
- [ ] 以 ENSO 空间结构和 IOD 背景解释主要 atom。

### Task 4: 逐图强化结论

**Files:**
- Modify: `docs/reports/earth.md`

- [ ] 为每张主图、补充图和对照图写结论先行的中文图注。
- [ ] 在正文补齐观察、科学解释与限制。
- [ ] 统一中英文术语、图号和表头。
- [ ] 逐项验收 17 张图：Runge ACE/ACS、H1、H10、H60、跨 H 汇总、强制 TM；UniCM mode 地理图、ENSO overall EI、all-target overall EI、ENSO 单源 EI、all-mode self EI、ENSO 二源 Syn、系统级 Phi、Phi 贪婪分解、lead-8 atom、IOD 二源 Syn、all-target pair Syn。
- [ ] 每张图逐项确认：图注首句给出结论；面板/坐标、估计口径和不确定性有交代；正文包含观察、科学意义和限制。多面板图按面板复核。

### Task 5: 验证

**Files:**
- Verify: `docs/reports/earth.md`
- Create: `/tmp/check_earth_report.py`（只读校验器，不纳入仓库）

- [ ] 运行 `python -c 'from pathlib import Path; import re; p=Path("docs/reports/earth.md"); t=p.read_text(); xs=[x for x in re.findall(r"!\\[[^]]*\\]\\(([^)]+)\\)",t) if not (p.parent/x).resolve().exists()]; print("missing_images="+str(len(xs)), *xs, sep="\\n")'`；预期 `missing_images=0`。
- [ ] 用 `apply_patch` 创建 `/tmp/check_earth_report.py`：提取目录链接和 GitHub 风格标题锚点并比较，同时检查相邻标题级别不得增加超过 1；运行 `python /tmp/check_earth_report.py docs/reports/earth.md`，预期 `broken_toc_links=0`、`heading_level_errors=0`。
- [ ] 在同一只读校验器中统计 `$$` 是否配对；提取每个展示公式的 `\tag{x.y}`、所有“式（x.y）”引用，检查公式数等于 tag 数、tag 按所在二级章节编号且引用均可解析；预期 `unpaired_display_math=0`、`untagged_equations=0`、`misnumbered_equations=0`、`broken_equation_refs=0`。
- [ ] 对照原文确认图、表、关键数值、引用、负面结果和限制未丢失。
- [ ] 显式核对以下限制全部保留：PC-stable 非原文逐项复刻、Varimax 编号未完全校准、距离仅按空间中心、强制 TM 仅验证指定边、离散候选初筛且无全局显著性检验、Gaussian log-det 只作筛查、冻结 UniCM、无历史事件归因、overall EI lead 排序跨 seed 不稳定、Syn 可为负。
- [ ] 运行 `rg -n '^#{1,6} .*\b(Overall|target|mode-pair|All-mode|PhiEID)\b|^\*.*\b(Overall|target|mode-pair|All-mode)\b|\| *(Item|Value|Target|Source pair|rank|status) *\|' docs/reports/earth.md`；预期正文标题、图注和表头无未规范化英文（变量名和方法专名除外）。
- [ ] 人工通读，确认中文衔接自然，无“此前/这里已改/上游 manifest/脚本内置”等实验日志式叙述。
- [ ] 检查 Git diff，确保不覆盖用户的无关改动。
