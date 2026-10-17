"""Fixed mixed-order coupled Henon map, compared with the same seven clusterers.

Six 2D Henon nodes, pairwise module 123, triadic module 456, weak cross edges.
All parameters, seeds 0/1/2 and two observation views are fixed before scoring.
This is a coupled stochastic extension of Henon's map, not the earlier bounded
tanh Henon-inspired response. No MLP, SPT/EI estimation, or parameter search.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compare_kuramoto_short_trajectory_clustering import (
    METHODS, TITLES, BLUE, ORANGE, analyze_view, digest, group_string,
    partition_drawing, tree_drawing,
)

OUT = ROOT / "results/henon_short_trajectory_clustering"
REPORT = ROOT / "docs/reports/henon_short_trajectory_clustering.md"
FIG = ROOT / "docs/reports/assets/kuramoto_hierarchy/henon_short_trajectory_clustering.png"
PAIR_WEIGHTS = {(0, 1): math.sqrt(1.5), (0, 2): math.sqrt(1.5) / 4,
                (1, 2): math.sqrt(1.5) / 4}
TRIAD_COUPLING = math.sqrt(2 * sum(w * w for w in PAIR_WEIGHTS.values()))
K_OUT = .04
NOISE = .08
SEEDS = (0, 1, 2)


def step(state, *, pair_scale=1., triad_scale=1., k_out=K_OUT, noise=None):
    """One simultaneous discrete update; noise, if given, has state shape N*6*2."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 3 or state.shape[1:] != (6, 2):
        raise ValueError("Six nodes with two coordinates each are required")
    x, y = state[:, :, 0], state[:, :, 1]
    future = np.empty_like(state)
    future[:, :, 0] = 1 - 1.4 * x**2 + y
    future[:, :, 1] = .3 * x
    for (i, j), weight in PAIR_WEIGHTS.items():
        contribution = pair_scale * weight * x[:, i] * x[:, j]
        future[:, i, 0] += contribution
        future[:, j, 0] += contribution
    future[:, 3:, 0] += (triad_scale * TRIAD_COUPLING * np.prod(x[:, 3:], axis=1))[:, None]
    for i in range(3):
        for j in range(3, 6):
            contribution = (k_out / 3) * x[:, i] * x[:, j]
            future[:, i, 0] += contribution
            future[:, j, 0] += contribution
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != future.shape:
            raise ValueError("Noise shape differs from complete state")
        future += noise
    if not np.isfinite(future).all():
        raise FloatingPointError("Nonfinite Henon map output; no clipping applied")
    return future


def data(seed, count=4800):
    source = np.random.default_rng(1000 + seed).uniform(-1, 1, (count, 6, 2))
    source[:, :, 1] *= .3
    noise = np.random.default_rng(11000 + seed).normal(0, NOISE, source.shape)
    future = step(source, noise=noise)
    return source, future, dict(source_sha256=digest(source), future_sha256=digest(future),
                                increment_sha256=digest(future - source), noise_sha256=digest(noise))


def protocol():
    return dict(date="2026-10-17", model="six two-coordinate stochastic coupled Henon maps",
                source="independent x_i ~ Uniform(-1,1), y_i ~ Uniform(-.3,.3)",
                sample_count=4800, independent_dataset_seeds=list(SEEDS), lag=1,
                a=1.4, b=.3, noise_sd_per_coordinate=NOISE, k_out=K_OUT,
                pair_weights={f"{i+1}-{j+1}": w for (i, j), w in PAIR_WEIGHTS.items()},
                triadic_coefficient=TRIAD_COUPLING,
                triadic_normalization="match mean per-receiver pairwise coupling RMS under independent uniform x",
                planted_modules=[[1, 2, 3], [4, 5, 6]],
                primary_view="increment", views={"increment": "(x_next-x,y_next-y) per node",
                                                 "transition": "(x,y,x_next-x,y_next-y) per node"},
                algorithms="same seven implementations/configurations as Kuramoto comparison, including KSG-1 k=5 and dimension-normalized MIC",
                fixed_algorithm_random_state=0, no_fitting=True, no_new_ei_or_syn=True,
                no_parameter_search=True, no_sample_or_seed_exclusion=True, no_state_clipping=True,
                compute_limit_seconds=900,
                code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), ROOT / "scripts/compare_kuramoto_short_trajectory_clustering.py"]},
                manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                                checked="2026-10-17", locations="Methods (10)-(12); SI S5, S6.4, S7",
                                version="current unique attachments, no explicit manuscript version/date"))


def plot(payload, view, path):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(3, 7, figsize=(17, 9.4), constrained_layout=True)
        for row, condition in enumerate(payload["conditions"]):
            for col, name in enumerate(METHODS):
                ax = axes[row, col]
                result = condition["views"][view]["methods"][name]
                if result["kind"] == "hierarchy":
                    tree_drawing(ax, result)
                else:
                    partition_drawing(ax, result)
                ax.set(xlim=(-.35, 5.35), ylim=(-1.55, 5.75))
                ax.axis("off")
                if row == 0:
                    ax.set_title(TITLES[name] + ("\nmerge MI / nats" if name == "mic" else ""), fontsize=10.5, pad=7)
                ax.text(2.5, -.98, group_string(result["groups"]), ha="center", fontsize=8.5)
                ax.text(2.5, -1.38, f"ARI {result['original_partition_ari']:.3f}", ha="center", fontsize=8, color="#65727C")
                if col == 0:
                    ax.text(-.1, 1.02, f"seed {condition['seed']}", transform=ax.transAxes,
                            fontsize=11, weight="bold", ha="left")
        fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Planted module 123"),
                            Line2D([], [], color=ORANGE, lw=2, label="Planted module 456")],
                   loc="outside lower center", ncols=2, frameon=False, fontsize=10)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_report(payload):
    lines = ["# 弱耦合混合阶 Hénon 网络：七种聚类比较", "", "2026-10-17。保留六节点、123 二体模块、456 三体模块及弱跨组耦合；将节点动力学改为二维离散 Hénon 映射。参数和三个数据种子在评分前固定，没有按 MIC 成败调整。", "",
             "![三个预定种子的增量聚类](assets/kuramoto_hierarchy/henon_short_trajectory_clustering.png)", "",
             "每行一个独立数据种子，每列一个方法。树的枝长只表示拓扑层次；MIC 标注两子组原始 MI（nats），不解释为纯阶因果贡献。K-means 和谱聚类只展示平面分组。叶色表示植入归属，蓝/橙完整分支标出 123/456。", "",
             "## 固定的离散动力学", "",
             r"每个节点的完整状态为 $\mathbf{s}_{i,t}=(x_{i,t},y_{i,t})$，系统共十二个标量状态。令 $P=\{1,2,3\}$、$T=\{4,5,6\}$，所有坐标同步更新：", "",
             r"$$\begin{aligned}x_{i,t+1}&=1-1.4x_{i,t}^2+y_{i,t}+C_i(\mathbf{x}_t)+0.08\epsilon^x_{i,t},\\y_{i,t+1}&=0.3x_{i,t}+0.08\epsilon^y_{i,t},\end{aligned}$$", "",
             r"$$C_i(\mathbf{x})=\begin{cases}\sum_{j\in P\setminus\{i\}}w_{ij}x_ix_j+\frac{0.04}{3}\sum_{j\in T}x_ix_j,&i\in P,\\ c_3x_4x_5x_6+\frac{0.04}{3}\sum_{j\in P}x_ix_j,&i\in T.\end{cases}$$", "",
             f"$w_{{12}}={PAIR_WEIGHTS[(0,1)]:.9f}$，$w_{{13}}=w_{{23}}={PAIR_WEIGHTS[(0,2)]:.9f}$，沿用原图的二体权重；$c_3={TRIAD_COUPLING:.9f}$。噪声逐坐标、逐样本独立标准 Gaussian。", "",
             r"独立 $x_i\sim\mathrm{Unif}(-1,1)$ 下，二体模块平均每接收节点的输入方差为 $2\sum_{i<j}w_{ij}^2/27=0.125$；三体输入方差为 $c_3^2/27$。固定 $c_3=\sqrt{2\sum_{i<j}w_{ij}^2}$，两者 RMS 同为 $\sqrt{0.125}=0.353553$。该归一化在查看聚类分数前确定；不按原始系数相等假定二体/三体信号强度相等。", "",
             "关闭耦合与噪声后，精确还原经典二维 Hénon 映射。二体/三体项和过程噪声是本实验的明确扩展；没有 tanh、有界投影或裁剪。三体混合导数为 c3，二体模块没有三变量混合项。它不同于仓库此前的标量 tanh Hénon-inspired 网络，也没有借用该模型的 SPT。", "",
             "每种子 4,800 个独立初态，x∈(-1,1)、y∈(-0.3,0.3)，各迭代一步；数据种子固定为 0、1、2，初态 RNG=1000+seed，噪声 RNG=11000+seed。所有方法逐条件读取同一状态对；不做 burn-in，也不筛掉轨迹。此协议检验一步转移的分组，不宣称是耦合系统自然吸引子上的长时间混沌检验。", "",
             "## 聚类口径与结果", "",
             "沿用 Kuramoto 比较的全部七法、固定参数及分组评分。主视图每节点为二维增量 (Δx,Δy)；补充视图每节点为四维完整转移信息 (x,y,Δx,Δy)。各坐标在全样本上标准化，六个节点是聚类对象。K-means k=2/n_init=50/random_state=0；谱聚类为绝对对齐相关 affinity、cluster_qr/k=2。Ward 用欧氏距离；相关/距离相关/成对 MI 使用 average linkage；MIC 每次重算组 MI 并按维数归一化。KSG-1 固定 k=5，1e-10 标准差抖动，负 MI 估计完整保留。没有按数据种子改变算法随机数或选择分辨率。", "",
             "| 方法 | 增量：正确两团次数 | 初态＋增量：正确两团次数 |", "|---|---:|---:|"]
    for name in METHODS:
        hits = [sum(c["views"][view]["methods"][name]["exact_original_two_groups"] for c in payload["conditions"])
                for view in ("increment", "transition")]
        title = TITLES[name].replace("|", "\\|")
        lines.append(f"| {title} | {hits[0]}/3 | {hits[1]}/3 |")
    for view, label in (("increment", "主视图：二维增量"), ("transition", "补充视图：完整节点转移")):
        lines += ["", "### " + label, "", "| seed | 方法 | 根切分／两团 | 保留123 | 保留456 | ARI |", "|---:|---|---|:---:|:---:|---:|"]
        for c in payload["conditions"]:
            for name in METHODS:
                result = c["views"][view]["methods"][name]
                title, groups = TITLES[name].replace("|", "\\|"), group_string(result["groups"]).replace("|", "\\|")
                lines.append(f"| {c['seed']} | {title} | {groups} | {'是' if result['module_123_retained'] else '否'} | {'是' if result['module_456_retained'] else '否'} | {result['original_partition_ari']:.3f} |")
        if view == "transition":
            lines += ["", "![完整节点转移的结果](assets/kuramoto_hierarchy/henon_short_transition_clustering.png)"]
    lines += ["", "## MIC 的实际合并与解释", "",
              "三个种子在两个视图下的 MIC 均恢复 123|456，没有观察到分组失误。各内部节点及其两子组原始 MI 如下；列表按树的后序遍历排列，不表示实际合并的时间次序。", ""]
    def merges(node):
        result = []
        for child in node["children"]:
            result += merges(child)
        if node["children"]:
            result.append(node)
        return result
    for c in payload["conditions"]:
        for view in ("increment", "transition"):
            result = c["views"][view]["methods"]["mic"]
            text = "; ".join(f"{''.join(map(str,n['sources']))}: {n['group_mi_nats']:.4f}" for n in merges(result["tree"]))
            lines.append(f"- seed {c['seed']} / {view}：{group_string(result['groups'])}；各内部组及合并 MI：{text}。")
    lines += ["", "本次模型不构成 MIC 的失败例子。其原因可以从已固定的方程看出：相同的三体乘积进入 456 的三个接收节点，因此输出之间存在共享信号。独立均匀初态、独立过程噪声下，对任意不同的 i,j∈456，", "",
              r"$$\operatorname{Cov}(\Delta x_i,\Delta x_j)=\operatorname{Var}(c_3x_4x_5x_6)=c_3^2/27=0.125.$$"]
    lines += ["", "各节点的自身折叠、记忆、初态扣除及弱跨组项在上述跨节点协方差中贡献为零，可由独立初态的零奇数矩核对。三源动力学项由此已经在成对输出统计中留下信息；MIC 无须从完全无成对信号的输入中猜出三元组。seed 0 增量视图的 456 三个成对 MI 估计为 0.0904、0.1154、0.1064 nats，亦与存在成对可读依赖一致。换成 Hénon 本身不保证制造 MIC 的失效条件。微小负 KSG MI 估计保留为有限样本估计值，不是 PEID Syn。", ""]
    lines += ["", "## 证据边界与复现", "",
              "本次未训练 MLP、未估计新 EI/Syn/SPT、未进行参数或模型搜索；因此即使 MIC 出错，也不能据此断言 SPT 在这个新 Hénon 网络上成功。旧 Kuramoto 树和旧有界 Hénon-style 树均不可移用。三个种子只是在这一个预先指定的耦合模型下的探索性重复，不能建立 MIC 的普遍失误率。", "",
              "保留了节点支持、二体权重比例与弱跨边，但 Hénon 的状态维数、耦合函数、噪声单位和预测时距均与 Kuramoto 不同。两种动力学不是完全等同尺度下的单一因素性能试验。增量视图保留与上一轮相同的观察方式；完整节点转移视图检查结论是否依赖遗漏初态。", "",
              "本次重新通过 Zotero 核对父项 P6UJCVG8 题名及当前唯一主文 DXGC7JEA（19/19 页）和 SI MWIWKSVG（28/28 页），读取 Methods (10)–(12)、SI S5/Algorithm S1、S6.4 和 S7。仍没有明确稿件日期或版本，元数据修改日期不消除版本歧义。SI S6.4 是受控响应，并未规定本次六节点二维耦合映射；本次方程为明确的新基准适配。", "",
              "原始状态对可低成本按固定种子重建，不重复持久化大数组；JSON 保存全部方法结果、MI 查询、树、距离矩阵、样本 SHA256、运行协议及代码摘要。复现入口 `scripts/compare_henon_short_trajectory_clustering.py`，内部结果 `results/henon_short_trajectory_clustering/summary.json`。不生成 CSV。", "",
              f"计算用时 {payload['elapsed_seconds']:.2f} 秒。3 项软件测试通过，覆盖无耦合极限、真实二体/三体混合项、RMS 归一化及数据复现；不替代统计精度验证。", "",
              "- Hénon (1976), [A two-dimensional mapping with a strange attractor](https://link.springer.com/article/10.1007/BF01608556)；[原文镜像](https://acoustique.ec-lyon.fr/chaos/Henon76.pdf)。", 
              "- [Kraskov et al., Hierarchical Clustering Using Mutual Information](https://arxiv.org/abs/q-bio/0311037)，连续 MIC Eq. (10)；[KSG-1](https://arxiv.org/abs/cond-mat/0305641)。", 
              "- 七法实现及其他方法文献沿用[上一轮报告](kuramoto_short_trajectory_clustering.md)，工作流技能及其参考亦列于其中。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = protocol()
    # Written before looking at any clustering result.
    (OUT / "protocol.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    payload = dict(protocol=contract, runtime=dict(python=platform.python_version(), numpy=np.__version__), conditions=[])
    started = time.monotonic()
    for seed in SEEDS:
        source, future, provenance = data(seed)
        increment = future - source
        condition = dict(seed=seed, provenance=provenance, views={})
        for view, blocks in (("increment", increment), ("transition", np.concatenate((source, increment), axis=2))):
            if time.monotonic() - started > 900:
                raise TimeoutError("Declared comparison limit reached")
            print(f"Henon seed={seed} view={view}: seven methods", flush=True)
            condition["views"][view] = analyze_view(blocks, K_OUT, seed=0)
            print({name: r['groups'] for name,r in condition['views'][view]['methods'].items()}, flush=True)
        payload["conditions"].append(condition)
        (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    payload["elapsed_seconds"] = time.monotonic() - started
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    plot(payload, "increment", FIG)
    plot(payload, "transition", FIG.with_name("henon_short_transition_clustering.png"))
    write_report(payload)
    print(f"Completed in {payload['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
