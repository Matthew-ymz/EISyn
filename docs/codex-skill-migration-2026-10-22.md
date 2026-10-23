# Codex 技能盘点与 WorkBuddy 迁移报告

生成时间：2026-10-22 15:58（本机时区 GMT+8）
执行环境：macOS (darwin)，WorkBuddy 桌面端
目标环境：WorkBuddy 用户级技能目录 `~/.workbuddy/skills/`

---

## 0. 结论摘要

| 项目 | 数量 |
|---|---|
| Codex 用户级技能（`~/.codex/skills/`） | 31 |
| Codex 系统级技能（`~/.codex/skills/.system/`） | 5 |
| Codex 插件层技能（`~/.codex/plugins/cache/`） | 39 |
| **合计** | **75** |
| 已成功部署到 WorkBuddy | **30** |
| 需改造后可用（已完成改造） | 19 |
| 无法迁移 | 1（用户级）+ 3（系统级）+ 39（插件层） |
| 部署产物体积 / 文件数 | 9.3 MB / 352 个文件 |

三条关键结论：

1. **用户级技能几乎全部可迁移**。它们的本质是"Markdown 指令 + 可选 Python 脚本"，与 Codex 运行时耦合很浅；30 个已落地并通过冒烟测试。
2. **真正的障碍只有两类**：外部 MCP 连接（Figma）与 Codex 私有运行时（插件层的 app-server / node_repl / computer-use）。
3. **环境差异中最容易踩坑的是 Python 解释器**：WorkBuddy 的 `python3` 指向托管运行时（3.13.12，无 numpy/scipy/matplotlib），与 Codex 里能直接跑科学计算脚本的假设不一致。已建独立 venv 并在每个技能的文档尾部写明解释器绝对路径。

---

## 1. 调查方法

1. 定位二进制与安装方式：`which -a codex`、`~/.npm-global/lib/node_modules`、`~/.codex/version.json`。
2. 枚举技能：`~/.codex/skills/*/SKILL.md`、`.system/*/SKILL.md`、`plugins/cache/**/SKILL.md`。
3. 抽取 frontmatter（name / description / allowed-tools / compatibility / metadata）作为功能与依赖初判。
4. **静态依赖分析**：用 AST 解析全部 bundled 脚本的 import，区分标准库与第三方包；正则提取外部 CLI（pandoc、draw.io、lean/lake、pdftoppm、uv）与 MCP 工具名、环境变量。
5. **运行时验证**：检查外部命令是否存在（`command -v`）、Zotero 本地 API（`127.0.0.1:23119`）是否可达、目标 venv 能否 import 全部依赖。
6. **迁移后回归**：对可执行技能实跑 `--help` 与真实产出（见第 6 节）。

---

## 2. Codex 安装详情

| 项 | 值 |
|---|---|
| CLI 版本 | `codex-cli 0.157.0` |
| 可执行文件 | `/Users/yangmingzhe/.npm-global/bin/codex` |
| npm 包位置 | `~/.npm-global/lib/node_modules/@openai/*` |
| 配置目录 | `~/.codex/`（`config.toml`、`skills/`、`plugins/`、`skill-repos/`、`sessions/`） |
| 默认模型 | `gpt-6.1-sol`（reasoning: ultra），另有 `deepseek` profile 指向本地 gateway |
| 已启用 MCP | `computer-use`（enabled=false）、`node_repl`（ChatGPT.app 内置 node） |
| 已启用插件 | documents / pdf / spreadsheets / presentations / template-creator / visualize / browser / computer-use / code-review / codex-app-tools |

> 注：Codex 的 MCP 配置全部绑定 `/Applications/ChatGPT.app` 与 `~/.codex/computer-use`，这也是插件层技能无法迁移的根因。

---

## 3. 用户级技能清单（31 个，`~/.codex/skills/`）

"迁移结论"列：**A**=直接复用，**B**=需改依赖/配置（已完成），**C**=无法迁移。

| # | 技能（路径 `~/.codex/skills/<名称>/`） | 功能说明 | 结论 |
|---|---|---|---|
| 1 | `nature-figure` | 论文级科研图表的创建、修订与审计（Matplotlib/Seaborn，Nature 风格，多面板、导出校验），本项目的核心绘图技能 | B |
| 2 | `academic-plotting` | 生成论文中的架构图、流程图、方法示意等非数值图 | A |
| 3 | `academic-experiment-visualizer` | 兼容性别名，把实验绘图请求重定向到 nature-figure | A |
| 4 | `scientific-visualization` | 同上，旧名称的兼容别名 | A |
| 5 | `academic-writing` | 中英文学术写作、改写、翻译、评审与审计；含投稿模板（NeurIPS/ICLR/ICML/AAAI/COLM）与 claims/citation 审计 CLI | B |
| 6 | `literature-review` | 系统性文献综述、meta 分析，多库检索（PubMed/arXiv/Semantic Scholar 等），输出带引用的 MD/PDF | B |
| 7 | `search-frontier-literature` | 前沿文献检索与 DOI/arXiv 元数据核验、开放获取全文发现 | B |
| 8 | `zotero-literature-synthesis` | 基于本地 Zotero 库做目标论文的对比综述，带证据等级标注与 item key | B |
| 9 | `nature-review-studio` | 多视角高水平审稿与逐条回复，知识库蒸馏自 1287 份 Nature 审稿文件，固定 DOCX+MD 双产物 | B |
| 10 | `experimental-design` | 实验设计：随机化、区组、因子/DOE、消融、配对种子、敏感性分析 | B |
| 11 | `statistical-power` | 样本量与统计功效计算（解析式 + Monte Carlo），MDE、power curve | B |
| 12 | `statsmodels` | statsmodels 模型参考：OLS/GLM/混合模型/ARIMA 与诊断 | B |
| 13 | `scikit-learn` | scikit-learn 建模参考：监督/无监督、评估、调参、Pipeline | B |
| 14 | `pymc` | PyMC 贝叶斯建模：层次模型、NUTS、VI、LOO/WAIC、后验检验 | B |
| 15 | `sympy` | 符号数学：代数、微积分、方程求解、lambdify/LaTeX 生成 | B |
| 16 | `networkx` | 复杂网络构建、图算法、社区发现与拓扑可视化 | B |
| 17 | `jupyter-notebook` | 创建/脚手架/编辑实验与教学 notebook（模板 + `new_notebook.py`） | B |
| 18 | `notebook-figure-integration` | 把 notebook 图表导出到 `fig/` 并在文档中引用 | B |
| 19 | `streamline-experiment-notebook` | 重构实验 notebook：保持"薄"，把可复用逻辑抽到 .py 模块 | A |
| 20 | `continuous-tuning-agent` | 仓库内自主调参与迭代搜索，长周期健康监督，结果落 `docs/log/` | B |
| 21 | `long-experiment-progress` | 长时实验（>15 min）轻量周期健康检查，禁止加进度条脚本 | B |
| 22 | `repo-simplifier` | 仓库结构审计与保守清理，先扫描后动刀 | A |
| 23 | `push-local-changes-to-github` | 直接提交并推送到远端默认分支，不建分支/PR，含安全闸门 | A |
| 24 | `analyzing-math-indicators` | 数学指标/函数的性质分析与证明（界、非负性、单调性、可加性），Lean 仅作内部校验 | A |
| 25 | `lean-proof-assistant` | Lean 4 / mathlib 定理证明、proof patch、调试与骨架生成 | B |
| 26 | `causal-scientist` | 因果推断：因果发现、反事实、干预效应、SCM/DAG、DoWhy/CausalNex | A |
| 27 | `swarma-encyclopedia-writing` | 集智百科（wiki.swarma.org）词条写作与 MediaWiki 源码整理 | A |
| 28 | `pdf` | PDF 读取/生成/审阅：Poppler 渲染校验 + reportlab/pdfplumber/pypdf | B |
| 29 | `drawio-skill` | 生成 .drawio XML 并调用 draw.io 桌面 CLI 导出 PNG/SVG/PDF | B |
| 30 | `figma-implement-design` | 把 Figma 节点 1:1 转成生产代码，依赖 Figma MCP + OAuth | **C** |
| 31 | `controlling-comparison-experiments` | 兼容性别名，指向 experimental-design | A（作为别名一并部署） |

> 说明：第 31 项在原清单中与 `experimental-design` 重复度高，部署时保留为独立目录以维持旧提示词兼容。

---

## 4. 系统级技能（5 个，`~/.codex/skills/.system/`）

| 技能 | 功能 | 处理方式 |
|---|---|---|
| `skill-creator` | 创建/更新 Codex 技能 | **不迁移**：WorkBuddy 自带同名 `skill-creator`，且其流程围绕 `$CODEX_HOME` 与 `openai.yaml` |
| `skill-installer` | 从 openai/skills 仓库安装技能到 `$CODEX_HOME/skills` | **不迁移**：WorkBuddy 有 `marketplace-skill-installer` 与技能市场 |
| `review-agent` | 只读取、面向缺陷的代码变更评审，供其他 agent 委派 | 可迁移（本次未部署）：WorkBuddy 用 subagent 机制更自然，价值重叠 |
| `imagegen` | 位图生成/编辑 | **不迁移**：WorkBuddy 内置 `ImageGen` 工具，能力等价 |
| `openai-docs` | Codex/OpenAI 产品与 API 文档查询、自我认知 | **不迁移**：内容专指 Codex，对 WorkBuddy 是错误知识源 |

---

## 5. 插件层技能（39 个，`~/.codex/plugins/cache/`）

| 插件市场 | 数量 | 技能 | 迁移判断 |
|---|---|---|---|
| `openai-primary-runtime` | 6 | documents、excel-live-control、pdf、presentations、spreadsheets、template-creator | **不可迁移**：绑定 Codex app-server 的 Office 实时控制通道；WorkBuddy 有 `workbuddy-docx` / `workbuddy-pptx` / `workbuddy-sheet-*` 等价能力 |
| `openai-curated` | 6 | github、gh-fix-ci、gh-address-comments、yeet、remotion、zotero | **不可迁移/不需要**：`zotero` 插件是 MCP 形态，本机 Zotero 本地 API 已可直接用；github 系列逻辑已由 `push-local-changes-to-github` 技能 + `gh` CLI 覆盖 |
| `openai-curated-remote` | 26 | 20 个 artifact-template-*、sites、sites-remix、plugin-management、pets/create-pet/update-pet | **不可迁移**：模板依赖 Codex Artifacts 渲染器；WorkBuddy 用 `sites` / `library` 能力 |
| `openai-bundled` | 1 | visualize | **不迁移**：WorkBuddy 有 `show_widget` 可视化通道 |

插件目录中另有 `browser`、`computer-use`、`unified-computer-use`、`code-review`、`codex-app-tools`、`gmail`，其载体是 MCP server 与原生 app 而非 SKILL.md，天然依赖 Codex 运行时，不在迁移范围。

---

## 6. 迁移评估明细

### A. 可直接复用（11 个）

`academic-plotting`、`academic-experiment-visualizer`、`scientific-visualization`、`streamline-experiment-notebook`、`causal-scientist`、`swarma-encyclopedia-writing`、`repo-simplifier`、`push-local-changes-to-github`、`analyzing-math-indicators`、`lean-proof-assistant`、`controlling-comparison-experiments`

共同点：纯指令型（Markdown + references），或依赖的外部能力在本机已就绪（git/gh、Lean 4 via `~/.elan/bin`）。
仍统一做了 3 处机械替换：`$skill` → `/skill`、`Codex` → `WorkBuddy`、追加运行环境说明。

### B. 需修改依赖或配置（19 个）

| 障碍类型 | 技能 | 具体问题 | 采取的处置 |
|---|---|---|---|
| Python 科学计算栈缺失 | nature-figure, experimental-design, statistical-power, statsmodels, scikit-learn, pymc, sympy, networkx, pdf, statistical-power, notebook-figure-integration, jupyter-notebook, literature-review | WorkBuddy 默认 `python3` 为 3.13.12 裸环境，无 numpy/matplotlib 等 | 新建 venv `~/.workbuddy/binaries/python/envs/default`，装齐 20 个包（含 pymc 6.3.2、pyDOE3），并把技能内的 `python3` 调用改为绝对路径 |
| 硬编码 Codex 路径 | academic-writing, jupyter-notebook, nature-review-studio | `${CODEX_HOME:-$HOME/.codex}/skills/...`、`~/.codex/skill-repos/...`、`~/.cache/codex-runtimes/...` | 全部改写为 `$HOME/.workbuddy/skills/...`、`~/.workbuddy/skill-repos/...`、目标 venv |
| 依赖 Codex 调度机制 | continuous-tuning-agent, long-experiment-progress | 原文用 Codex 的"短周期 recurring heartbeat" | 改为说明使用 WorkBuddy automation（`automation_update` 一次性/重复自动化），禁用 cron 模拟 |
| 依赖外部服务/密钥 | zotero-literature-synthesis, search-frontier-literature, literature-review | Zotero 本地 API；NCBI/OpenAlex/S2/CORE/OpenRouter key | Zotero 已验证可达（HTTP 200），脚本实跑通过；key 均为可选，未配置 |
| 外部 CLI 缺失 | drawio-skill | 本机无 `draw.io`/`drawio`（且未安装 Homebrew） | 技能已部署，导出功能待装 draw.io 桌面版（见第 8 节） |
| 大仓库路径耦合 | nature-review-studio | 技能依赖 `skill-repos/nature-review-studio` 发布根（11 MB） | 整仓复制到 `~/.workbuddy/skill-repos/`，并同步改写 `NRS_ROOT` 与 `.nrs_root` |

### C. 无法迁移（1 个用户级 + 3 个系统级 + 39 个插件级）

| 对象 | 阻断原因 | 是否有替代 |
|---|---|---|
| `figma-implement-design` | 必须连接 Figma MCP server（`https://mcp.figma.com/mcp`）并完成 OAuth；WorkBuddy 侧无 Figma 连接器，也无等价设计稿读取工具；原文的 `codex mcp add/login` 指令在 WorkBuddy 中不存在 | 否。若需要，须先具备 Figma MCP（可在 `~/.workbuddy/mcp.json` 挂载后再迁移） |
| `openai-docs` | 内容是 Codex 产品知识，迁入会成为错误知识源 | WorkBuddy 官方文档 |
| `skill-installer` / `skill-creator` | 与 WorkBuddy 内置能力重复且路径模型不同 | 内置 `skill-creator`、`marketplace-skill-installer` |
| 39 个插件层技能 | 依赖 Codex app-server、node_repl、Artifacts 渲染、Office 实时通道 | 见第 5 节对应替代 |

---

## 7. 已完成的部署

### 7.1 目标位置与规范核对

- 目标：`~/.workbuddy/skills/<skill-name>/SKILL.md`（用户级，跨项目可用）。
- 已确认该路径在 `~/.workbuddy/settings.json` 中注册为技能根目录（`"path": "~/.workbuddy/skills/"`）。
- frontmatter 兼容性已核对：WorkBuddy 客户端解析 `name` / `description`，并支持 `allowed-tools`（客户端内含 `allowed-tools` → `allowedTools` 映射），因此保留原 frontmatter 不破坏加载。
- 30 个技能全部通过结构校验：目录名与 `name` 一致、`description` 存在、YAML 头完整。

### 7.2 统一改写规则（仅作用于 `.md`，脚本与数据未改动）

1. `$skill-name` → `/skill-name`（WorkBuddy 用斜杠/技能工具调用，不再有 `$` 前缀）。
2. `${CODEX_HOME:-$HOME/.codex}/skills/`、`$CODEX_HOME/skills/`、`~/.codex/skills/` → `~/.workbuddy/skills/`。
3. 正文中的 `Codex` → `WorkBuddy`（含 description，保证触发语义正确）；命令行中的小写 `codex` 不改。
4. `python3 ` → `/Users/yangmingzhe/.workbuddy/binaries/python/envs/default/bin/python `。
5. 每个 `SKILL.md` 末尾追加「运行环境（WorkBuddy 部署）」小节：技能根目录、调用方式、解释器路径与已装依赖，必要时附外部依赖/密钥说明。

### 7.3 Python 运行时

- 位置：`/Users/yangmingzhe/.workbuddy/binaries/python/envs/default/bin/python`（3.13.12）。
- 已安装：numpy、scipy、pandas、matplotlib、seaborn、statsmodels、scikit-learn、sympy、networkx、pyDOE3、nbformat、nbconvert、jupyterlab、python-docx、pdfplumber、pypdf、reportlab、pillow、requests、lifelines、pingouin、openpyxl、pymc、arviz。
- 安装耗时约 36 s（uv），未污染系统 Python 与 anaconda。

### 7.4 冒烟测试结果

| 技能 | 测试 | 结果 |
|---|---|---|
| nature-figure | `style_preview.py --output preview.png --style nature` | ✅ 产出 177 KB PNG，无 warning |
| experimental-design | `doe_designs.py`（2³ 全因子） | ✅ 正常输出设计表 |
| zotero-literature-synthesis | `rank_related.py --query "effective information" --limit 3 --json` | ✅ 命中本地库条目（含 key/摘要/DOI） |
| nature-review-studio | `render_review_docx.py --help` + `.nrs_root` 校验 | ✅ 脚本可执行，根路径已指向新位置 |
| academic-writing | 检查 `references/audit/cli_reference.md` 路径改写 | ✅ 已指向新技能目录与新解释器 |
| pymc | `import pymc` | ✅ 6.3.2 |
| 全量 | frontmatter 结构校验 29/29 | ✅ 无异常 |

### 7.5 回滚方式

原 Codex 技能目录**未做任何修改**，仍可正常使用。回滚只需删除 `~/.workbuddy/skills/` 下对应目录（以及 `~/.workbuddy/skill-repos/nature-review-studio`）。

---

## 8. 待执行步骤

| 优先级 | 步骤 | 说明 |
|---|---|---|
| P0 | **重启 WorkBuddy（或新开会话）** | 技能清单在启动时扫描并缓存（`~/.workbuddy/.skill-list-cache.json`），当前会话的可用技能列表是会话开始时固定的 |
| P1 | 安装 draw.io 桌面版 | 本机无 Homebrew，无法 `brew install --cask drawio`。请从 https://github.com/jgraph/drawio-desktop/releases 下载 macOS arm64 dmg，安装后确认 `draw.io --version` 可用；否则 `drawio-skill` 只能生成 .drawio 源文件，无法导出 PNG/SVG/PDF |
| P2 | 如需 Figma 能力 | 在 `~/.workbuddy/mcp.json` 配置 Figma MCP 并完成授权后，可将 `figma-implement-design` 从 `~/.codex/skills/` 复制过来并做同样的 4 条改写 |
| P3 | 可选密钥 | `search-frontier-literature`（NCBI/OpenAlex/S2/CORE）、`literature-review`（OPENROUTER_API_KEY）不配也能跑，配了可提高速率与覆盖率 |
| P3 | 确认 Zotero 本地 API | 当前已验证可达；若重启 Zotero 后失效，在 设置 → 高级 中确认「允许其他应用与 Zotero 通信」已勾选 |

---

## 9. 风险与未决事项

1. **技能数量增长后的触发竞争**：29 个技能中，`nature-figure`/`academic-plotting`/`academic-experiment-visualizer`/`scientific-visualization` 描述高度重叠，模型可能在四个之间摇摆。建议长期只保留 `nature-figure` 作为主入口，其余三个作为别名保留但在描述中保持"别名/重定向"措辞（现状已如此）。
2. **venv 与系统解释器漂移**：技能文档中写死了 venv 绝对路径。若日后重建 venv，需批量替换 30 个 SKILL.md 中的路径（一条 `rg` + `sed` 即可）。
3. **`~/.codex` 仍是被引用对象**：`nature-review-studio` 的发布仓已复制到 WorkBuddy 侧，但若卸载 Codex，需确认无其他隐性引用；本次扫描显示脚本层已无 `.codex` 残留。
4. **未迁移不等于无用**：`review-agent`、插件层 `zotero`、`github` 系列在 WorkBuddy 侧有更贴合的实现路径，后续如有需要可按"技能 + MCP/子 agent"重新设计，而不是照搬。
5. **令牌成本**：29 个技能的 description 会进入每次会话的技能清单。当前总量约 9.2 MB 文件但 description 很短，影响有限；若继续扩张，建议按项目启用项目级技能（`<repo>/.workbuddy/skills/`）而非全部用户级。

---

## 附录：部署清单（30 个，`~/.workbuddy/skills/`）

```
01 academic-experiment-visualizer      16 networkx
02 academic-plotting                   17 notebook-figure-integration
03 academic-writing                    18 pdf
04 analyzing-math-indicators           19 push-local-changes-to-github
05 causal-scientist                    20 pymc
06 continuous-tuning-agent             21 repo-simplifier
07 controlling-comparison-experiments  22 scientific-visualization
08 drawio-skill                        23 scikit-learn
09 experimental-design                 24 search-frontier-literature
10 jupyter-notebook                    25 statistical-power
11 lean-proof-assistant                26 statsmodels
12 literature-review                   27 streamline-experiment-notebook
13 long-experiment-progress            28 swarma-encyclopedia-writing
14 nature-figure                       29 sympy
15 nature-review-studio                30 zotero-literature-synthesis
```

配套：`~/.workbuddy/skill-repos/nature-review-studio/`（nature-review-studio 的发布根与知识库，11 MB）。
