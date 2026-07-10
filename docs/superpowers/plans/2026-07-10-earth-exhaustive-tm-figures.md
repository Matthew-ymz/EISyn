# Earth 全量 TM 超边图实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Earth 报告图 2—4 替换为全量 TM 重估的 (H=1,10,60) 全局 top-10 超边图，并使报告叙事一致。

**Architecture:** 在独立绘图模块中验证并读取每个尺度的完整穷举排名缓存，复用既有 Runge 世界地图超边编码输出图片和 CSV。该模块自行验证 NPZ schema 和候选宇宙，不依赖当前未提交的实验运行脚本。随后以生成资产与缓存中的数值为唯一证据更新 `earth.md`，将 shortlist 集中到图 5 的诊断说明。

**Tech Stack:** Python、NumPy、pandas、matplotlib、unittest、Markdown。

---

## Chunk 1: 全量排名驱动的空间网络图

### Task 1: 为全量 top-10 读取与导出建立回归测试

**Files:**
- Create: `tests/test_plot_runge_exhaustive_tm_maps.py`
- Create: `scripts/plot_runge_exhaustive_tm_maps.py`

- [ ] **Step 1: 写出失败测试**

使用临时的 `full_ranking.npz` 和匹配 `summary.json`，断言读取器只接受 `candidate_count=102660`、匹配 horizon/metadata、连续 `tm_rank` 的完整排名；并断言所有行满足 `source_a < source_b`、源与 target 不同、编号在 60 个 component 范围内、候选无重复且其 canonical tuple 顺序/hash 与完整候选宇宙一致。再断言导出的 top-10 保留 10 行、对应全局 rank `1..10`、转换为 paper component 编号。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest tests.test_plot_runge_exhaustive_tm_maps -v`

Expected: FAIL，因为全量空间图读取/导出模块尚不存在。

- [ ] **Step 3: 实现最小读取器和绘图 CLI**

实现独立脚本：读取 `summary.json` 的 `ranking_metadata` 并在本模块内解析 NPZ；拒绝缺失、损坏、非 `102660` 行、metadata 不匹配或不满足 canonical candidate-universe/hash 的缓存；从同一验证过的 ranking 取前十；复用 `plot_runge_multistep_ridge_node0_hyperedges.py` 的 component 中心、Mollweide 底图和“两源 → 紫色中介点 → 目标”绘制函数。为每个 horizon 写出 PNG、SVG、PDF 和 top-10 CSV；CSV 包含全局 `tm_rank`、local/paper 源与目标编号、`delta2_tm`、`joint_ei`、两个单源 EI 及来源指纹。

- [ ] **Step 4: 运行 GREEN**

Run: `python -m unittest tests.test_plot_runge_exhaustive_tm_maps -v`

Expected: PASS，并确认读取器不调用任何 TM 估计函数或未提交的实验运行脚本。

- [ ] **Step 5: 提交绘图模块与测试**

```bash
git add tests/test_plot_runge_exhaustive_tm_maps.py scripts/plot_runge_exhaustive_tm_maps.py
git commit -m "feat: plot exhaustive TM Runge maps"
```

### Task 2: 生成并视觉核查三张替换图

**Files:**
- Create: `fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H001_tm_exhaustive.{png,svg,pdf,csv}`
- Create: `fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H010_tm_exhaustive.{png,svg,pdf,csv}`
- Create: `fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/top10_order2_hyperedges_H060_tm_exhaustive.{png,svg,pdf,csv}`

- [ ] **Step 1: 生成图片和表格**

Run: `python scripts/plot_runge_exhaustive_tm_maps.py --horizons 1,10,60 --output-dir fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive`

Expected: 每个尺度产生 PNG/SVG/PDF/CSV，且命令只读取既有穷举缓存。

- [ ] **Step 2: 核对来源与排名**

读取三个 `summary.json`、`input_manifest.json` 和新 CSV；确认每个尺度 metadata 的 `candidate_count=102660`、input fingerprint 一致、CSV 的 `tm_rank` 为 `1..10`，并逐项与 `full_ranking.npz` 前十行一致。

- [ ] **Step 3: 做视觉检查**

逐张查看 PNG，确认沿用既有世界地图视觉语言：浅灰参考节点、活跃节点编号可读、紫色中介点与箭头均不裁切，且无图例覆盖超边或数据。

- [ ] **Step 4: 提交图片与表格**

```bash
git add fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive
git commit -m "fig: add exhaustive TM Runge hyperedge maps"
```

## Chunk 2: 报告叙事与引用同步

### Task 3: 以全量 TM 结果重写图 2—4 的证据链

**Files:**
- Create: `tests/test_earth_exhaustive_tm_report.py`
- Modify: `docs/reports/earth.md`

- [ ] **Step 1: 写出报告一致性失败测试**

为 `earth.md` 添加提交的只读测试，要求图 2—4 都指向 `multistep_conditioned_ei_tm_exhaustive` 资产，图注均写明全量 `102660` 条候选/全局 TM top-10，并且第 3.2 节、解释边界与图表索引不再把它们称为 shortlist 内部排序。测试同时检查图片存在、目录链接有效、展示公式 `\\tag{x.y}` 与“式（x.y）”引用可解析且编号一致。

- [ ] **Step 2: 运行 RED**

Run: `python -m unittest tests.test_earth_exhaustive_tm_report -v`

Expected: FAIL，因为当前图片路径和文字仍指向 shortlist 重排资产。

- [ ] **Step 3: 最小化编辑报告**

替换图 2—4 的图片路径、alt text 和结论先行图注；用新 CSV/全量排名更新每个尺度的前三边和数值；把第 3.2 节改为全量 TM 横截面，把 shortlist 的作用收束到图 5；同步表 1 前后衔接、第 3.3—3.4 的交叉叙事、第 5.1 的限制和第 6 节资产索引。保留未完成 block-bootstrap、季节分层和物理校准的限制；不声称已量化不确定性。

- [ ] **Step 4: 运行 GREEN 与 Markdown 校验**

Run: `python -m unittest tests.test_earth_exhaustive_tm_report -v && python -c 'from pathlib import Path; import re; p=Path("docs/reports/earth.md"); t=p.read_text(); xs=[x for x in re.findall(r"!\[[^]]*\]\(([^)]+)\)",t) if not (p.parent/x).resolve().exists()]; print("missing_images="+str(len(xs)), *xs, sep="\n")'`

Expected: `missing_images=0`，并通过图 2—4 口径、公式标签/交叉引用和目录链接检查。

- [ ] **Step 5: 保护既有报告改动并保留本次编辑**

```bash
git diff -- docs/reports/earth.md
git status --short docs/reports/earth.md
```

由于 `earth.md` 在本次工作开始前已处于修改状态，不使用 `git add docs/reports/earth.md` 或提交它，以免暂存/提交无关的用户改动；交付前在 diff 中人工确认新增的全量 TM 编辑没有覆盖原有内容。提交新增测试时，只暂存 `tests/test_earth_exhaustive_tm_report.py`。

### Task 4: 最终复核

**Files:**
- Verify: `tests/test_plot_runge_exhaustive_tm_maps.py`
- Verify: `tests/test_earth_exhaustive_tm_report.py`
- Verify: `docs/reports/earth.md`
- Verify: `fig/runge_slp_daily_1948_2026_20260628/multistep_conditioned_ei_tm_exhaustive/`

- [ ] **Step 1: 运行自动验证**

Run: `python -m unittest tests.test_plot_runge_exhaustive_tm_maps tests.test_earth_exhaustive_tm_report -v`

Expected: 所有测试和口径检查通过。

- [ ] **Step 2: 完成证据核对**

逐项比对三个 CSV 与全量缓存的前十行，并审阅 `git diff --check` 与 `git diff -- docs/reports/earth.md`，确认没有覆盖无关的用户修改。

- [ ] **Step 3: 最终图形 QA**

再次查看三张 PNG，确认标题、组件编号、箭头、中介点和边没有遮挡或裁切；图 2 的分散性与图 3、4 的 `No.0/No.1` 集中性均与正文一致。
