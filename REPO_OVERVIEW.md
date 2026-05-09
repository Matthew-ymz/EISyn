# EISyn 仓库说明

本说明文档对当前仓库进行一次完整清点，目的是让新加入的协作者或合作研究者能在不打开每个文件的前提下，快速理解：仓库在研究什么问题、代码与实验如何组织、哪些目录承担哪类职责，以及在哪里可以找到主线结论与产物。

## 1. 项目主题

仓库围绕一套统一的研究主线展开：**Partial Effective Information Decomposition (PEID) 与协同因果（synergistic causality）**。围绕这一主题，仓库同时承载理论推导与四类实验：

- 离散布尔与马尔可夫系统中的有效信息分解、协同测度、向下因果与 IIT 2.0 主复合体的关系；
- 线性高斯 / 八节点 Mediano 风格 benchmark 上的 EI 分解与 `Phi^EID` 比较；
- 连续动力学下基于 transport map 的互信息估计、L 敏感性与密度估计基线；
- 真实数据（长三角空气质量站点、美国 Compustat 上市公司年度面板）上的 PEID 因果图与协同超图实验。

研究框架的论文级稿件存放在 `docs/研究框架.md`（正文）和 `docs/研究框架_附录.md`（附录），两份文件之间通过分节编号交叉引用。

## 2. 顶层目录速览

```text
EISyn/
├── AGENTS.md              # 协作约定（绘图规则、长实验缓存约定、Zotero/技能触发条件）
├── REPO_OVERVIEW.md       # 本文件
├── utils.py               # 仓库主工具库（离散 EI/PID、TPM 构造、绘图、benchmark 工具）
├── data/                  # 原始与中间数据集
├── docs/                  # 研究框架、参考材料、设计文档与运行日志
├── exp/                   # 实验 notebook 及其缓存 (exp/cache/)
├── fig/                   # 实验产生的图表与中间数据
├── results/               # 结构化结果产物 (CSV/JSON/HTML)
├── scripts/               # 可执行脚本（CLI 入口、批量任务、PDF 导出）
├── tests/                 # pytest 测试，覆盖核心代码与 notebook 烟雾级流程
├── yrd/                   # 长三角实验的核心 Python 包（合并入 __init__.py）
├── .gitignore
└── .skill_staging/, .superpowers/, .claude/, .pytest_cache/   # 本地工具/缓存目录，详见第 12 节
```

`.gitignore` 显式排除了 `exp/cache/`、`results/`、`fig/**/artifacts/`、所有模型权重 (`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`)、`data/*.nc`、`data/*.csv`、`docs/log/**` 等大体量与本地化资产；它们当前存在于工作副本里，但不进入版本控制。

## 3. 顶层关键文件

- `AGENTS.md`：仓库级协作约定。包含三条对实验有强约束力的规则：
  1. 实验图例必须放在轴外（默认右侧），导出时使用 `bbox_inches="tight"`，避免遮挡数据；
  2. `docs/` 中引用的图必须以 `png` 或 `pdf` 作为最终插入格式，仅 `svg` 不允许进入研究框架文档；
  3. 长实验必须把结果落盘到 `exp/cache/`、`results/` 或同级缓存目录，便于后续直接复用。
- `utils.py`：仓库的主工具模块（约 1300 行，聚合了离散 TPM 构造、确定性/概率布尔系统枚举、Marshall Example 1 复现、PID 与协同分解、密度估计、SVG 因果图渲染、benchmark 数据集等）。所有 `exp/` 下的 notebook 都通过 `from utils import ...` 调用它。
- `utils.py` 在导入时显式设置 `MPLCONFIGDIR` 到一个临时目录，并把 matplotlib 的后端固定为 `Agg`、字体改为 `DejaVu Serif`，从而在 headless 与受限环境下也能稳定出图。

## 4. 核心代码包 `yrd/`

`yrd/` 是长三角空气质量实验的主代码库。它的源码已被合并为一个 `__init__.py`，但保留了原先按模块划分的注释段（`# --- Former yrd/<name>.py ---`）。核心子模块及职责如下：

- `yrd/config.py`：`YRDExperimentConfig` 数据类，定义数据路径、训练配置（`history_hours=24`、`horizons=(1,24)`、`target_variables=("O3","PM2.5")`、训练/验证/测试时间切分等）。
- `yrd/data.py`：从 `data/dataset_yrd.nc` 读取 xarray，对站点元数据 `data/stations_yrd.csv` 做筛选，构建滑窗样本与时间切分。
- `yrd/models.py`：`JointStationMLP` 等多站点联合预测网络（ResMLP / LayerNorm + SiLU 等结构）。
- `yrd/train.py`：训练循环、checkpoint 管理、`run_smoke_pipeline`、`rebuild_joint_model_from_checkpoint`、numpy 推理工具。
- `yrd/coupling.py`：基于 Jacobian 与 NIS 的协同度量（`compute_subset_nis_summary`、`estimate_residual_covariance`、`jacobian_for_target_subset` 等）。
- `yrd/transport_map.py`：仿射下三角 transport map 互信息估计器（`estimate_mutual_information_transport_map`、`lift_transport_source_features`），同时被 `utils.py` 引用。
- `yrd/intervention_sampling.py`：均匀盒式干预采样、support-cover `L_v` 计算等口径工具。
- `yrd/groups.py`：源变量分组（站点污染物、气象变量、跨站污染物等）。
- `yrd/analysis.py`、`yrd/plotting.py`：协同摘要、JSON/Markdown 报告输出，以及上海实验的横向对比图。
- `yrd/shanghai_notebook.py`：上海单城实验的 notebook 级薄封装（`build_default_shanghai_one_step_config`、`prepare_shanghai_one_step_bundle`、`run_or_load_one_step_predictions`、`compute_station_causal_graph_results`）。
- `yrd/air_search.py`、`yrd/air_search_notebook.py`：多城市（上海、南京、杭州、北京）粗搜 + 精搜 + 报告三阶段流水线，以及 notebook 级薄封装（`run_air_tm_notebook_case`、`save_combined_tm_graph_panel`）。

## 5. `scripts/`：CLI 与批处理入口

`scripts/` 提供命令行入口与离线批处理脚本：

- `scripts/run_yrd_experiment.py`：长三角实验的 CLI 入口，目前实现 `--smoke` 模式，依次跑训练、分析、绘图阶段。
- `scripts/run_air_search.py`：多城市 air search 实验的 `coarse / refine / report` 三阶段调度。
- `scripts/tm_stability_scan.py`：TM 稳定性扫描脚本（含 `phaseA`、`phaseB` 等批次）。
- `scripts/build_yrd_all_station_summary.py`：全站点汇总图与协同摘要的离线构建。
- `scripts/export_research_framework_pdf.py`：把 `docs/研究框架.md` 中的 svg 引用改写为 png/pdf，调用 pandoc + xelatex 生成 PDF。
- `scripts/company_ce/`：Compustat 公司财务数据子项目的脚本：
  - `build_qtpm.py`：构建无条件分位数转移概率矩阵（QTPM）；
  - `peid_causal_hypergraph.py`：PEID 因果超图主分析（核心 `PeidConfig` 类、`run_peid_analysis`、`plot_outputs`）；
  - `peid_sensitivity.py`：参数鲁棒性扫描（`SensitivityConfig`、`run_sensitivity_analysis`）；
  - `plot_peid_robustness_graphs.py`：把鲁棒性扫描的 CSV 渲染为热图/网格图。

## 6. `tests/`：覆盖代码与 notebook

`tests/` 下的测试既覆盖纯 Python 模块，也覆盖 notebook 流程（依赖环境变量 `YRD_NOTEBOOK_TEST_MODE=1` 等开关进入轻量化模式）：

- 模块级测试：`test_utils.py`、`test_main_complex_utils.py`、`test_yrd_data.py`、`test_yrd_groups.py`、`test_yrd_models.py`、`test_yrd_pipeline.py`、`test_yrd_plotting.py`、`test_yrd_coupling.py`、`test_yrd_air_search.py`、`test_company_ce_peid_sensitivity.py`、`test_tm_ei_l_baseline_support.py`、`test_export_research_framework_pdf.py`。
- Notebook 烟雾测试：`test_main_complex_notebook.py`、`test_marshall_example1_macro_search_notebook.py`、`test_boolean_motif_causal_graphs_notebook.py`、`test_discrete_benchmark_notebook.py`、`test_rq3_manual_case_notebook.py`、`test_rq3_causal_emergence_notebook.py`、`test_yrd_shanghai_notebook.py`、`test_yrd_single_station_notebook.py`、`test_yrd_air_search_notebook.py`、`test_company_ce_peid_notebook.py`。

## 7. `exp/`：实验 notebook

仓库的实验主线全部以 notebook 形式承载在 `exp/` 下。按研究问题归类：

### 7.1 离散系统与 IIT 2.0 对照

- `exp/main_complex.ipynb`：在 5 节点布尔系统（`A/B/C` 强耦合核心 + `D/E` 前馈 readout）上同时计算 IIT 2.0 的 `Phi` 与 EI 协同分数 `Phi^EID`，比较 main complex 选择口径。
- `exp/boolean_motif_causal_graphs.ipynb`：基于 3 节点布尔机制（COPY / AND / XOR）渲染 ground-truth 与重建因果图，并同时输出单目标和多目标协同超边表。
- `exp/marshall_example1_macro_search.ipynb`：复现 Marshall et al. (2024) Example 1 的 4 节点微观 TPM，在 588 个 `2+2` 候选粗粒化中检验论文 Figure 4C 是否仍是 EI 最大解。
- `exp/discrete_benchmark.ipynb`：六个八节点 Mediano 风格离散 benchmark 网络的总 EI 与 `Phi^EID` 对比，并提供 publication-style 重绘。
- `exp/experiment6_downward_causation.ipynb`：复现 Rosas (2020) Figure 1 的 XOR 与 causal-decoupling 例子，验证向下因果分数 `DC_j = flexibility + environment synergy` 的离散基线。

### 7.2 因果涌现 RQ3 系列

- `exp/rq3_manual_case.ipynb`：6 节点手工布尔系统的最优/非最优粗粒化对比（最优时宏观 EI=3.000、协同=0；非最优时宏观 EI=1.658、协同=0.393）。
- `exp/rq3_boolean_causal_emergence.ipynb`：Hoel Figure 2 toy example 与 micro-mechanism family（21 个 `q_off` 系统、每个系统枚举 147 个 `2+2` 候选粗粒化），输出三类相关系数分布。

### 7.3 连续动力学与 transport map

- `exp/linear_gaussian_benchmark.ipynb`：六个八节点 AR(1) 线性高斯网络上对比 `EI_full / Syn_high` 与 Mediano 指标族（`Phi`、`Phi_tilde`、`psi`、`CD`、`TDMI`、平均绝对相关）。
- `exp/tm_nonlinear.ipynb`：已知动力学下用 transport map 验证协同估计；包含外部噪声扫描、`L` 敏感性扫描、固定 `alpha` 噪声扫描。
- `exp/tm_ei_l_baseline.ipynb`：1D 动力学 `\{x, x^2, sin(x), exp(x)\}` 在不同 `L` 下的 `EI_raw(L)` 与 baseline (`identity`、`variance_matched` 等) 对比，并附带固定 `L` 的中心扫描。
- `exp/density_estimation_benchmark.ipynb`：仿射下三角 transport map vs KDE vs kNN 的密度估计基线（高斯 / GMM / banana 分布），含维度与样本量鲁棒性扫描和时间对比。

### 7.4 长三角空气质量

- `exp/yrd_hangzhou_tm_graph.ipynb`：杭州站点网络上的多步预测 + TM 因果图实验，封装在 `yrd.air_search_notebook` 之上，比较不同 `horizon`、`tm_sample_count`、随机种子和 `top_k` 下的三类因果图（O3 自身、PM2.5 → O3、O3+PM2.5 → O3）。

> 与上海相关的 notebook（如 shanghai single-step 实验）当前以测试桩 `tests/test_yrd_shanghai_notebook.py` 与 `tests/test_yrd_single_station_notebook.py` 形式存在；上海实验的实际 notebook 不在 `exp/` 中，主要逻辑在 `yrd/shanghai_notebook.py` 内，由测试 / 脚本驱动。

### 7.5 公司财务（company_ce）

- `exp/company_ce/qtpm_growth_heatmaps.ipynb`：从 `scripts/company_ce/build_qtpm.py` 输出读取四个变量（`at`、`revt`、`emp`、`dltt`）的分位数转移概率矩阵并绘制 2×2 热图。
- `exp/company_ce/peid_causal_hypergraph.ipynb`：在 `inf_at, inf_revt, emp, inf_lt, inf_ch, inf_ni, inf_cogs` 上做单尺度 PEID 因果超图分析，同时支持参数鲁棒性扫描。

### 7.6 实验缓存 `exp/cache/`

每个长实验都会把可重用的中间产物写到 `exp/cache/` 下：

- `exp/cache/density_benchmark/`：accuracy / dimension / sample_size 的原始 parquet 与汇总 csv。
- `exp/cache/rq3_boolean_causal_emergence/`：主实验汇总、Hoel family、micro-mechanism family、reverse-permutation summary、per-system 相关系数 csv。
- `exp/cache/rq3_manual_case/`：手工系统的最优/非最优粗粒化结果。
- `exp/cache/yrd_coupling/`：上海与杭州的联合模型 checkpoint、metrics、coupling samples、causal graph summary。其中 `air_search/<city>/<horizon>/` 下按 `seed` 和 `lmax / l<value>` 等 tag 组织多种子实验。
- `exp/cache/yrd_coupling/shanghai_full_v5_resmlp/` 与 `shanghai_one_step_o3_station_graph[_tm_causal_graph]/`：上海单城多 / 单步实验的关键 checkpoint 与 summary。

## 8. `data/`：原始数据集

- `data/dataset_yrd.nc`、`data/dataset_bthsa.nc`、`data/dataset_bthsa_yrd_aqi_mete_emis.nc`：长三角与京津冀+长三角的污染物 + 气象 + 排放面板（NetCDF）。
- `data/stations_yrd.csv`、`data/stations_bthsa.csv`：站点元数据（坐标、城市、类别）。
- `data/inf_compustat_anual_US_filter_feas.csv`：通胀调整后的美国上市公司 Compustat 年度面板（company_ce 子项目主输入）。

注：`data/*.nc` 与 `data/*.csv` 已被 `.gitignore` 排除，不进入版本控制；本地副本由原始来源单独维护。

## 9. `fig/`：实验图表与产物

`fig/` 按实验主题组织子目录，与 `exp/` 中的 notebook 一一对应：

- `fig/main_complex/`：5 节点布尔静态因果拓扑。
- `fig/boolean_motif_causal_graphs/`：ground-truth 机制图、pairwise EI、order0/1/2 协同超图、多目标超图、`manifest.json`。
- `fig/mediano_discrete_benchmark/` 与 `fig/mediano_discrete_benchmark_publication/`：六网络拓扑总图与 EI / `Phi^EID` 柱状图（含论文风格重绘 + 源数据 csv）。
- `fig/mediano_linear_gaussian_benchmark/`：双节点低/高机制对比、固定 `a` 时的相关曲线/协方差布局图。
- `fig/experiment6_downward_causation/`：causal decoupling vs downward causation 双栏图、混合机制图。
- `fig/rq3_boolean_causal_emergence/`：代表性 pack / split / pair 散点图、family term 分布、micro topology 概览、最优/非最优粗粒化对照。
- `fig/transport_map_mutual_information/`：transport map 与各 baseline 在密度估计、`L` 敏感性、center sweep 上的对比图（PNG）与 manifest（json）。
- `fig/yrd_shanghai/`：上海单城多站点时序、anchor 站点 24h 时间序列、joint metrics 比较、站点布局图、smoke 级摘要 md/png。
- `fig/yrd_shanghai/artifacts/`：以 run name 区分的训练 artifacts（`full_v5_resmlp`、`shanghai_one_step_o3_station_graph[_tm_causal_graph]`、`tm_stability_phase1[_variableL]/batch_*` 等）。
- `fig/yrd_air_search/<city>/<horizon>/`：杭州（含其它城市）多种子的 refine TM run manifest 与图。
- `fig/company_ce/qtpm/`、`fig/company_ce/peid*/`：QTPM 与 PEID（含 sensitivity 扫描）的图。

## 10. `results/`：结构化结果

`results/` 用于沉淀供论文/外部脚本直接引用的结构化结果：

- `results/company_ce/csv/qtpm/`：四个变量的概率矩阵、计数矩阵、growth pairs、quantile edges、汇总。
- `results/company_ce/csv/peid_smoke/`、`peid_bins5_smoke/`、`peid_notebook_smoke/`：开发与 notebook 用的 smoke 结果。
- `results/company_ce/csv/peid_sensitivity[_main]/`：按 `bins_*`、`min_source_count_*`、`alpha_*`、`winsor_*` 等设置组织的鲁棒性扫描产物，包含每个设置的全量 pairwise 边、协同超边、零分布样本与 `peid_run_config.json`。
- `results/rq3_boolean_causal_emergence/`：split / pair 散点 svg、相关系数分布、family term 分布、Hoel toy example 散点。
- `results/rq3_manual_case/`：手工案例的 metrics_table.html 与 summary.json。
- `results/hypernetx/hypernetx_smoke.png`：hypernetx 渲染烟雾测试。

## 11. `docs/`：文档与日志

`docs/` 同时承载“面向论文的研究框架”、参考材料与“面向开发的设计/计划/调参日志”。

### 11.1 研究框架与正文素材

- `docs/研究框架.md`：PEID 论文级正文，包含 1–6 节（问题设定、EI 分解、因果图、计算方式、向下因果、实验设计 RQ1–RQ6）+ 参考文献，并通过分层公式编号 `(k.m)` / `(k.l-m)` 引用附录公式。
- `docs/研究框架_附录.md`：附录 A–M，承载完整证明、连续情形推导、Jacobian-Covariance 展开、复杂度估算与扩展定理。
- `docs/其他材料.md`：5 节点布尔 main-complex 与 IIT 2.0 对照表等额外稿件素材。
- `docs/hangzhou.md`：杭州监测网与污染源综合调研报告。
- `docs/rq3_boolean_causal_emergence_notebook_results.md`：RQ3 notebook 的结果说明。
- `docs/company_ce/company_ce.md`：公司面板 PEID 子项目的研究问题、数据定义与结论组织。

### 11.2 参考材料

- `docs/ref/信息分解与涌现相关研究的学术讨论.md`：项目内部学术讨论纪要。
- `docs/ref/micro_macro_support_jacobian_derivation.md`：micro-macro support / Jacobian 推导参考。

### 11.3 设计与计划

设计文档与执行计划按照 `superpowers/{specs,plans}/` 双轨组织，文件名以 `YYYY-MM-DD-<topic>-{design,plan}.md` 命名，便于按时间线追踪：

- `docs/superpowers/specs/`、`docs/superpowers/plans/`：当前活跃的设计与执行计划（`air-causal-search`、`tm-alpha-synergy` 等）。
- `docs/log/superpowers/specs/`、`docs/log/superpowers/plans/`：归档的历史设计与计划（覆盖 `shanghai-one-step-joint-model`、`shanghai-conditional-synergy-graph`、`shanghai-global-l-ratio-graph`、`shanghai-uniform-box-causal-graph`、`shanghai-tm-causal-graph`、`transport-map-large-l`、`transport-map-notebook-simplification`、`marshall-example1-ei-validation`、`discrete-benchmark-notebook-slimming`、`rq2-three-variable`、`rq3-hoel-family-per-system-correlation`、`rq3-manual-case-notebook`、`streamline-experiment-notebook` 等主题）。
- `docs/log/2026-04-08-上海单城完整实验notebook设计.md`：上海单城实验的 notebook 设计稿。
- `docs/log/王硕论文_因果图待验证命题.md`：从外部博士论文中提炼的 YRD 因果图待验证命题。

### 11.4 调参与运行日志（air_tuning）

`docs/log/air_tuning/` 是 air search 调参主线的工作目录：

- `leaderboard.md`：当前 TM 稳定性 leaderboard（`StageA2_TM_VariableL_pollutant_x1.2_M8192` 居首）。
- `notes.md`、`next_steps.md`、`search_plan.md`、`repo_audit.md`：调参思路、下一步、搜索计划与仓库审计。
- `search_space.json`、`tm_l_profiles.json`、`run_history.jsonl`：搜索空间与运行历史的结构化记录。
- `docs/log/tuning_report.md`：阶段性调参汇总。

## 12. 本地工具与辅助目录

以下目录主要服务本地开发，不进入版本控制（已被 `.gitignore` 排除），但当前工作副本中存在，列出便于排查：

- `.skill_staging/notebook-figure-integration/`：notebook 图导出技能的 staging 区。
- `.superpowers/brainstorm/`：brainstorm 工具的本地状态（`.server.pid`、`.server-stopped`、`connection-check.html`）。
- `.claude/worktrees/elastic-nightingale-3c5d90/`：Claude 协作工具创建的 git worktree，是仓库的一份镜像副本，包含同步的 `scripts/`、`tests/`、`utils.py`、`yrd/__init__.py`、`docs/` 等内容；除非显式需要，应在主目录而不是 worktree 副本上修改文件。
- `.pytest_cache/`：pytest 缓存。

## 13. 复现与协作约定速查

- 主要交互方式：先打开对应的 `exp/<topic>.ipynb`，按需通过 `YRD_NOTEBOOK_TEST_MODE=1`、`PEID_NOTEBOOK_SMOKE=1` 等环境变量进入 smoke 模式；长实验默认从缓存读取，缓存缺失时自动重算。
- CLI 入口：长三角实验主要使用 `python -m scripts.run_yrd_experiment` 或 `python scripts/run_yrd_experiment.py`，多城市搜索使用 `scripts/run_air_search.py`，PDF 导出使用 `scripts/export_research_framework_pdf.py`。
- 测试入口：`pytest tests/` 可同时验证模块与 notebook smoke；notebook 测试依赖 smoke 模式开关，不会跑完整训练。
- 图与表的引用约定：研究框架文档只引用 `png` / `pdf`；`svg` 仅作中间编辑格式，导出后必须有同名的 `png`/`pdf` 伴随产物。
- 所有长实验都遵循“先写缓存再画图”的约定（详见 `AGENTS.md`），因此 `exp/cache/` 与 `fig/<topic>/artifacts/` 中常见 `loss_history.json`、`metrics_summary.json`、`config.json`、`run_manifest.json` 等结构化文件，可作为重跑或撰写论文时的权威来源。

如果需要在本说明之外得到更细粒度的清单（例如某个 notebook 调用了哪些 `utils.py` 中的函数、`yrd/` 中各子模块的接口签名、或 `docs/log/superpowers/` 中具体某条计划的执行状态），可以直接定位到对应文件再展开阅读。
