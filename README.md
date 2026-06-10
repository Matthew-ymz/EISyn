# EISyn

EISyn 是一个围绕 **Partial Effective Information Decomposition (PEID)**、effective information (EI) 与协同因果的研究仓库。仓库同时包含理论文档、可复现实验 notebook、命令行脚本、结果图表和 pytest 测试，用于研究离散系统、连续非线性动力学、气候时空网关、空气质量网络、脑网络与公司面板数据中的高阶因果结构。

## 研究主线

当前仓库围绕以下问题组织：

- PEID / EI 在离散布尔系统、马尔可夫系统与 IIT 2.0 main complex 对照中的定义、分解与可视化。
- 线性高斯、Mediano 风格 benchmark、Runge 时空 causal gateway / mediator 中的 EI 与协同因果读出。
- transport-map mutual information 在连续非线性动力学中的干预式 EI 估计，包括 Lorenz-96、已知动力学、密度估计和 L 敏感性实验。
- 真实数据应用，包括长三角 / 京津冀空气质量网络、NCEP/NCAR SLP Runge 分量、Lausanne 脑网络、microbiome network revival 与 Compustat 公司面板 PEID 超图。
- 高阶 PEID 协同超图、pairwise MLP/GRU/RNN 预测读出、Granger vs PEID 对照，以及与论文写作相关的研究框架文档。

论文级研究框架主要在 [docs/研究框架.md](docs/研究框架.md) 与 [docs/研究框架_附录.md](docs/研究框架_附录.md)。Runge 实验综述在 [docs/runge_experiments.md](docs/runge_experiments.md)。Lorenz-96 TM-EI 实验说明在 [exp/TM/lorzen/README.md](exp/TM/lorzen/README.md)。

## 仓库结构

```text
EISyn/
├── README.md              # 仓库入口说明
├── AGENTS.md              # 协作、绘图、长实验缓存和 PEID 文献规则
├── utils.py               # 离散 EI/PID、TPM、benchmark、绘图等通用工具
├── density_benchmark.py   # density / transport-map benchmark 支持代码
├── yrd/                   # 长三角空气质量实验 Python 包
├── scripts/               # CLI 入口、复现实验、批处理与文档导出脚本
├── exp/                   # notebook、实验模块与实验子项目
├── docs/                  # 研究框架、实验报告、调参日志与参考材料
├── fig/                   # 直接可查看的实验图表
├── results/               # 结构化结果产物，通常被 .gitignore 排除
├── data/                  # 本地数据集，通常被 .gitignore 排除
└── tests/                 # pytest 测试与 notebook smoke 测试
```

`.gitignore` 默认排除缓存、模型权重、大体量数据和多数结果目录，包括 `exp/cache/`、`results/`、`data/*.nc`、`data/*.csv`、`*.pt`、`*.pth`、`*.ckpt`、`*.onnx` 等。实验图优先以可直接查看的 PNG 作为文档引用资产。

## 主要代码模块

### `utils.py`

仓库的主工具库，包含：

- 离散状态空间、TPM 构造与布尔机制枚举。
- EI、PID、协同分解、main-complex / benchmark 相关工具。
- Marshall Example 1、Mediano benchmark、因果图与超图渲染。
- headless matplotlib 设置，便于在 CI / 远程环境中稳定出图。

### `yrd/`

长三角空气质量实验的核心包，当前代码集中在 `yrd/__init__.py`，保留原模块分段注释。主要职责包括：

- 数据读取、站点筛选、滑窗样本与 train/validation/test 切分。
- 多站点联合预测模型、训练循环、checkpoint 管理。
- Jacobian / NIS / transport-map MI 协同估计。
- 上海、杭州与多城市 air-search notebook 的轻量封装。
- 因果图、协同摘要、Markdown / JSON 报告和横向对比图生成。

### `exp/TM/`

transport-map EI 与连续动力学实验，包括：

- `tm_nonlinear.ipynb`：已知非线性动力学下的 TM-MI / EI 验证。
- `tm_ei_l_baseline.ipynb`：不同 intervention support `L` 与 baseline 的 EI 对照。
- `transport_map_density.py`、`transport_map_density_demo.ipynb`：TM density estimation demo。
- `lorzen_tm_ei.py`、`lorzen_synergy.py`、`lorzen/README.md`：Lorenz-96 TM-EI 因果恢复、lag sweep、MLP intervention 与二源协同验证。

## 实验入口

### 离散系统与因果涌现

- [exp/main_complex.ipynb](exp/main_complex.ipynb)：5 节点布尔系统中 IIT 2.0 `Phi` 与 `Phi^EID` 对照。
- [exp/boolean_motif_causal_graphs.ipynb](exp/boolean_motif_causal_graphs.ipynb)：COPY / AND / XOR 机制的因果图与协同超边。
- [exp/marshall_example1_macro_search.ipynb](exp/marshall_example1_macro_search.ipynb)：Marshall et al. Example 1 粗粒化搜索。
- [exp/discrete_benchmark.ipynb](exp/discrete_benchmark.ipynb)：六个八节点 Mediano 风格 benchmark。
- [exp/rq3_manual_case.ipynb](exp/rq3_manual_case.ipynb) 与 [exp/rq3_boolean_causal_emergence.ipynb](exp/rq3_boolean_causal_emergence.ipynb)：RQ3 因果涌现实验。

### 连续动力学与 TM-EI

- [exp/linear_gaussian_benchmark.ipynb](exp/linear_gaussian_benchmark.ipynb)：线性高斯 benchmark 与 Mediano 指标族比较。
- [exp/TM/tm_nonlinear.ipynb](exp/TM/tm_nonlinear.ipynb)：非线性动力学下 TM-MI 协同估计。
- [exp/TM/tm_ei_l_baseline.ipynb](exp/TM/tm_ei_l_baseline.ipynb)：EI 的 `L` 敏感性与 baseline 支持集对照。
- [exp/TM/lorzen/README.md](exp/TM/lorzen/README.md)：Lorenz-96 因果恢复实验，当前推荐 lag=3 观测路径、delta surrogate intervention 与 MLP intervention。

### Runge / 气候网关

- [docs/runge_experiments.md](docs/runge_experiments.md)：Runge 线性复现、MLP-TM-EI path-effect 与 PEID 二阶超图的汇总说明。
- [scripts/reproduce_runge2015_gateways.py](scripts/reproduce_runge2015_gateways.py)：Runge 2015 causal gateway / mediator 线性复现。
- [scripts/run_runge_pairwise_mlp_ei.py](scripts/run_runge_pairwise_mlp_ei.py)：pairwise MLP-EI 读出。
- [scripts/run_runge_peid_hypergraph.py](scripts/run_runge_peid_hypergraph.py)：Runge 分量上的 PEID 高阶协同超图。
- [scripts/run_runge_rnn_forecast_comparison.py](scripts/run_runge_rnn_forecast_comparison.py)：RNN / GRU history sweep 与预测对照。
- [exp/runge_slp_60d_visualization/](exp/runge_slp_60d_visualization/)：60 个 SLP Varimax 分量可视化与 synthetic GRU 支持代码。

### 空气质量与真实数据

- [yrd/yrd_hangzhou_tm_graph.ipynb](yrd/yrd_hangzhou_tm_graph.ipynb)：杭州站点 TM 因果图实验。
- [exp/o3_h1_tm_peid_results.ipynb](exp/o3_h1_tm_peid_results.ipynb)：O3 horizon-1 TM / PEID 结果整理。
- [exp/bthsa_pm25_deweather_shap.ipynb](exp/bthsa_pm25_deweather_shap.ipynb)：PM2.5 deweathering 与 SHAP 分析。
- [scripts/run_yrd_experiment.py](scripts/run_yrd_experiment.py)：长三角实验 smoke CLI。
- [scripts/run_air_search.py](scripts/run_air_search.py)：多城市 air-search 的 coarse / refine / report 调度。

### 其他应用

- [exp/company_ce/](exp/company_ce/) 与 [scripts/company_ce/](scripts/company_ce/)：Compustat 公司面板 QTPM 与 PEID 因果超图。
- [exp/network_revival/](exp/network_revival/)：network revival、microbiome、multistable attractor 与协同 ignition 实验。
- [exp/brain/](exp/brain/)：DMF Fig.6、Lausanne 脑网络与 downward causation notebook。
- [exp/granger_peid_mlp_comparison.ipynb](exp/granger_peid_mlp_comparison.ipynb)：Granger 与 PEID / MLP 因果读出对照。
- [scripts/classic_network_dynamics_benchmark.py](scripts/classic_network_dynamics_benchmark.py)：Kuramoto、耦合 Rössler、SIS 与 Wilson–Cowan 向量场上的 Granger / SHAP / SURD / PEID 对照。
- [docs/reports/classic_network_dynamics_benchmark.md](docs/reports/classic_network_dynamics_benchmark.md)：经典模型实验设计、干预协议、3-seed 结果与失败模式说明。
- [exp/mediated_peid_known_dynamics.ipynb](exp/mediated_peid_known_dynamics.ipynb)：已知动力学中的 mediated PEID 验证。

## 常用命令

运行测试：

```bash
pytest tests/
```

运行长三角 smoke pipeline：

```bash
python -m scripts.run_yrd_experiment --smoke
```

运行多城市 air-search：

```bash
python scripts/run_air_search.py coarse
python scripts/run_air_search.py refine
python scripts/run_air_search.py report
```

复现 Lorenz-96 推荐的 lag=3 TM-EI：

```bash
python -m exp.TM.lorzen_tm_ei exp/TM/lorzen/input \
  --output-dir exp/TM/lorzen/results_lag3 \
  --top-k 4 \
  --target-mode next \
  --estimator-mode observed \
  --lag 3
```

运行 Lorenz-96 MLP intervention：

```bash
python -m exp.TM.lorzen_tm_ei \
  --simulate-lorenz96-mlp \
  --output-dir exp/TM/lorzen/results_mlp \
  --top-k 4 \
  --lag 15 \
  --box-width 8.0 \
  --sample-count 3000 \
  --steps 8000 \
  --burn-in 1000 \
  --train-sample-count 5000 \
  --hidden-layer-sizes 128,64 \
  --max-iter 1000 \
  --seed 4
```

导出研究框架 PDF：

```bash
python scripts/export_research_framework_pdf.py
```

## 结果与图表

主要图表目录：

- `fig/runge/`：Runge gateway / mediator、pairwise MLP-EI、MLP-TM-EI、PEID hypergraph、RNN history sweep 与 SLP 分量可视化。
- `fig/transport_map_mutual_information/`：TM density、非线性协同、`L` sensitivity 与 baseline 对照。
- `fig/yrd_shanghai/`、`fig/yrd_air_search/`：上海 / 杭州空气质量因果图与 TM 稳定性结果。
- `fig/company_ce/`：公司面板 QTPM 与 PEID 鲁棒性扫描图。
- `fig/network_revival_microbiome/`、`exp/network_revival/figures/`：network revival 与 microbiome 结果图。
- `fig/dmf_fig6_brodmann82/`、`exp/brain/result_lausanne_fig6/`：脑网络复现实验产物。

结构化结果通常写入 `results/` 或实验子目录下的 `results_*`，长实验缓存写入 `exp/cache/` 或 notebook-adjacent cache。大体量结果默认不进入版本控制。

## 文档

- [docs/研究框架.md](docs/研究框架.md)：PEID 论文正文草稿。
- [docs/研究框架_附录.md](docs/研究框架_附录.md)：证明、连续情形推导与复杂度估算。
- [docs/runge_experiments.md](docs/runge_experiments.md)：Runge 线性复现、非线性 EI 与 PEID 超图实验报告。
- [docs/granger_peid_mlp_comparison.md](docs/granger_peid_mlp_comparison.md)：Granger / PEID / MLP 对照结果。
- [docs/iid_fig6_phi_eid_comparison.md](docs/iid_fig6_phi_eid_comparison.md)：IIT / `Phi^EID` 对照说明。
- [docs/高阶PEID协同定义.md](docs/高阶PEID协同定义.md)：高阶协同定义草稿。
- [docs/log/](docs/log/) 与 [docs/superpowers/](docs/superpowers/)：调参日志、设计文档与执行计划。

## 协作约定

见 [AGENTS.md](AGENTS.md)。关键规则包括：

- 实验图例不要遮挡线、点、柱或置信区间；密集图优先把 legend 放在轴外右侧。
- 实验结果默认先产出可直接查看的 PNG；除非文档构建或用户明确要求，不默认导出 PDF / SVG / TIFF 伴随文件。
- 长实验仅在重算成本高或用户要求时持久化机器可读缓存。
- PEID 理论相关任务应优先使用 Zotero 检索本地 PEID 文献；不可用时需明确说明。

## 数据说明

`data/` 中的数据文件通常是本地大文件，不随仓库提交。当前实验涉及的数据包括空气质量 / 气象 / 排放 NetCDF 与站点元数据、Compustat 年度面板、NCEP/NCAR SLP 相关派生产物等。需要复现实验时，应先确认对应 notebook 或脚本引用的数据路径是否存在。
