# Kuramoto 同步转变图的复现

本实验对应图中 **b Difference and synchronization**：绿色曲线先升后降，峰值位于 `K=1.7`，约 `193.82 nats`；灰色曲线为有限尺寸基线修正后的同步序参量。旧图记为 `Phi^EID`，当前代码和手稿使用 `Xi`，这里复现相同的数值定义。

## 代码位置与 GitHub 核对

计算和绘图均位于 [`scripts/classic_network_dynamics_benchmark.py`](../../scripts/classic_network_dynamics_benchmark.py)：

- `run_large_kuramoto_oracle_nsource_whole_state_phi_sweep`：生成独立均匀相位干预，积分已知 Kuramoto 方程，计算 EI 分解及独立初始化集合上的同步参照曲线。
- `_transport_nsource_phi` → `_gaussian_block_conditional_total_correlation`：本实验 `degree=1` 的估计路径。
- `_plot_large_kuramoto_n64_ei_decomposition`：绘制 EI 分量和截图所示 b 面板，纵轴由 bits 乘 `ln(2)` 转换为 nats。

2026-10-25 核对的 GitHub `main` 为 [`7e4a7bb72e1561c8d551248bcc198316f868590c`](https://github.com/Matthew-ymz/EISyn/tree/7e4a7bb72e1561c8d551248bcc198316f868590c)。上述计算和绘图代码已经存在。该版本未包含本地图对应的 `large_kuramoto_oracle_nsource_whole_state_phi_sweep_n64.json` 缓存及 `large_kuramoto_n64_ei_decomposition.png`，但下面命令可从头生成数据，无需外部数据、Zotero、GPU 或模型权重。

**不要直接使用原脚本的默认参数复现截图。** 原通用入口默认 `N=16`、三个种子，默认耦合网格也缺少截图中的 `0.05`、`0.1`。下面通过已有 Python 函数显式传入截图参数，可直接用于上述 GitHub 版本，无需新增计算脚本。

## 环境与运行方式

在终端中执行，或已有仓库时从进入仓库目录开始。为固定本次核对版本，可在新克隆的仓库内执行 `git checkout 7e4a7bb72e1561c8d551248bcc198316f868590c`；不需要切换已有工作目录。

```bash
git clone https://github.com/Matthew-ymz/EISyn.git
cd EISyn
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 pandas==2.1.4 matplotlib==3.11.1
```

本次验证环境为 Python 3.11.8，以及上述三个包的版本。本实验路径只需 NumPy、pandas 和 Matplotlib 及其安装依赖；不需要为其他 benchmark 安装 PyTorch。然后在**仓库根目录**运行完整命令：

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python - <<'PY'
import json
import math
from pathlib import Path

import numpy as np
from scripts.classic_network_dynamics_benchmark import (
    run_large_kuramoto_oracle_nsource_whole_state_phi_sweep,
    _plot_large_kuramoto_n64_ei_decomposition,
)

result_dir = Path("results/kuramoto_synchronization")
figure_dir = Path("fig/kuramoto_synchronization")
result_path = result_dir / "kuramoto_n64.json"

# 原始 Syn/Xi 的数值容差，单位 nats；保留原值，不做裁剪。
tolerance_nats = 1e-8
payload = run_large_kuramoto_oracle_nsource_whole_state_phi_sweep(
    mode="full",
    oscillator_count=64,
    couplings=(
        0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
        1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.6, 3.2, 4.0,
    ),
    seeds=(0, 1),
    tau=4.0,
    frequency_sigma=1.0,
    nsource_transport_degree=1,
    nsource_null_shuffles=1,
    result_path=result_path,
    figure_path=figure_dir / "oracle_sweep.png",
)

raw_nats = np.asarray([row["oracle_raw_phi"] for row in payload["rows"]]) * math.log(2)
bad_count = int(np.count_nonzero(raw_nats < -tolerance_nats))
validity = {
    "tolerance_nats": tolerance_nats,
    "min_raw_syn_nats": float(raw_nats.min()),
    "numerical_zero_count": int(np.count_nonzero((raw_nats < 0) & (raw_nats >= -tolerance_nats))),
    "significant_negative_count": bad_count,
    "values_clipped": False,
}
if bad_count:
    raise ValueError(
        f"Syn nonnegativity violation: minimum={raw_nats.min()} nats, "
        f"threshold={-tolerance_nats} nats, count={bad_count}"
    )
payload["syn_validity"] = validity
result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# 通用 sweep 入口不自动调用这个与截图对应的绘图函数，需显式执行。
figure_path = figure_dir / "ei_decomposition.png"
_plot_large_kuramoto_n64_ei_decomposition(payload, figure_path)
peak = max(payload["summary"], key=lambda row: row["oracle_phi_mean"])
print(f"Peak: K={peak['coupling']:g}, Xi={peak['oracle_phi_mean'] * math.log(2):.8f} nats")
print("Syn validity:", validity)
print("Data:", result_path)
print("Figure:", figure_path)
PY
```

## 参数与结果对应

| 项目 | 截图使用的设置 |
| --- | --- |
| 动力学 | 全局耦合 Kuramoto，`K/N` 归一化；已知方程的 Oracle 实验 |
| 振子数与种子 | `N=64`，种子 `0, 1` |
| 固有频率 | 每个种子重新生成高斯频率，再去均值并缩放至标准差 1 |
| EI 干预 | 每个条件 220 个独立相位配置，各振子 `Uniform(0, 2*pi)` |
| 同步参照 | 另外生成 360 个独立初态；同样推进至 `tau=4`，不是稳态扫描 |
| 时间积分 | `tau=4`、`dt=0.04`，沿用现有显式 Euler 实现 |
| 来源与目标 | 64 个来源块，每块为一个振子的 `(cos, sin)`；目标为全部振子的未来 `(cos, sin)`，共 128 维 |
| EI 估计 | `nsource_transport_degree=1`，线性高斯 / affine-TM 近似；源协方差按振子分块，ridge 为 `1e-6` |
| 绿色曲线 | `EI(all;Y) - sum_i EI(i;Y)`；缓存单位 bits，绘图乘 `ln(2)` 得 nats |
| shuffle 参照 | 每条件 1 次；图中使用原始 `oracle_phi`，不是 `oracle_null_corrected_phi` |
| 误差带 | 两个种子的均值及现有实现的 `std(ddof=0)/sqrt(2)` |
| 横轴 | `symlog`，`linthresh=0.05`，含 `K=0` |

全局配置字段 `transport_map.degree=3` 是其他 benchmark 的通用配置；本实验实际使用专门字段 **`nsource_transport_map_degree=1`** 及对应的 Gaussian 分支。复现时应依据后者。

主要输出为：

- `results/kuramoto_synchronization/kuramoto_n64.json`：38 个条件的逐种子结果、19 点汇总和数值有效性记录。
- `fig/kuramoto_synchronization/ei_decomposition.png`：a、b 两个面板；**右侧 b 面板对应截图**。当前标签为 `Xi (nats)`。
- 通用函数还会输出 `oracle_sweep.png` 和 determinism/degeneracy 辅助图。

`results/` 和 `fig/` 受现有 `.gitignore` 忽略；其他人运行后在本地生成这些文件。仅需重绘时，可以读取已生成的 JSON，再调用上述绘图函数，无需重新积分。

应得到的主要数值：

| K | Xi / nats | corrected order |
| --- | ---: | ---: |
| 0 | 数值零 | 0.027120 |
| 1.7 | 193.821488 | 0.226392 |
| 2.2 | 178.536940 | 0.506932 |
| 4 | 9.411491 | 0.960236 |

理论无限总体阈值为 `Kc=1.595769`；本组有限尺寸、有限时间的同步曲线最大斜率位于 `K=2.2`。该结果验证已知动力学及独立干预条件下的分解曲线，不能仅凭本图宣称已经验证观测数据的提前预警性能。

2026-10-25 已直接使用上述 GitHub 提交的计算代码，从头重算全部 38 个条件，本机耗时约 16 秒。与原本地缓存相比，逐条件 Xi 的最大绝对差为 `4.0e-9 nats`，同步序参量最大绝对差为 `1.12e-16`；38 个输入样本摘要均一致。本次原始 Xi 最小值为 0，`1e-8 nats` 容差内的负值计数和显著负值计数均为 0，没有裁剪。新生成的 nats 图已与截图逐轴核对。

## 手稿依据与验证边界

本次重新查询 Zotero 父条目 `R68U4QXB`，核实标题为 *Revealing hierarchies of synergistic causal interactions in complex systems*。可用正文附件为 `GRX5IJJQ`，附录为 `UTEBTIWV`；附件录入于 2026-10-25，未给出明确版本号或独立稿件日期，不能仅凭录入或修改时间认定更高版本。核对内容为正文 integrated effective information 的式 (7)–(8)，以及附录 **S8、式 (S78)–(S80)、Figure 3（PDF 第 17–18 页）**。

本说明区分已知非线性动力学和其信息量的 degree-one 估计近似；核对不构成对其他实验或更高阶估计器的全面一致性审计。
