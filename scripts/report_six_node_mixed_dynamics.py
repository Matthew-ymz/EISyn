"""Render the frozen six-site dynamics search results; no new experiments."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compare_kuramoto_short_trajectory_clustering import (
    METHODS, TITLES, BLUE, ORANGE, GREEN, tree_drawing, partition_drawing, group_string,
)
from scripts.search_six_node_mixed_dynamics import (
    OUT, LOG, observations, DiscreteInformation, spin_kernel, nontrivial_bipartitions,
)

FIG = ROOT/"docs/reports/assets/kuramoto_hierarchy/mixed_order_glauber_comparison.png"
REPORT = ROOT/"docs/reports/six_node_mixed_dynamics_search.md"
ORDER = ("kmeans", "ward", "spectral", "correlation_hac", "dcor_hac", "mi_hac", "mic", "spt")
LABELS = ["K-means", "Ward", "Spectral", "Correlation HAC", "Distance-corr. HAC", "Pairwise MI HAC", "Group MI (MIC)", "SPT"]


def recoveries(rows, layer, name):
    return sum((row["fitted_spt"] if name == "spt" else row[layer]["transition"]["methods"][name])
               ["exact_original_two_groups"] for row in rows)


def make_figure(uniform, stationary):
    rows = uniform["conditions"][1:]
    natural = stationary["conditions"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(15, 10.7), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=(1.7, 1.))
        trees = outer[0].subgridspec(2, 4, wspace=.08, hspace=.12)
        for index, name in enumerate(ORDER):
            ax = fig.add_subplot(trees[index//4, index%4])
            result = uniform["population_spt"] if name == "spt" else uniform["population"]["transition"]["methods"][name]
            if result["kind"] == "hierarchy":
                tree_drawing(ax, result)
            else:
                partition_drawing(ax, result)
            ax.set(xlim=(-.3, 5.3), ylim=(-1.2, 5.2))
            ax.axis("off")
            ax.set_title(LABELS[index], fontsize=10.5, pad=1, color=GREEN if name == "spt" else "#222222")
            ax.text(2.5, -.92, group_string(result["groups"]), ha="center", fontsize=9)
            if index == 0:
                ax.text(-.02, 1.15, "a  Exact-kernel partitions | random initial states, full transitions",
                        transform=ax.transAxes, ha="left", fontsize=11, weight="bold")
        bottom = outer[1].subgridspec(1, 3, width_ratios=(1.42, 1.12, 1.))
        ax = fig.add_subplot(bottom[0])
        scores = np.array([[np.nan if name == "spt" else recoveries(rows, "empirical", name),
                            recoveries(rows, "model", name),
                            np.nan if name == "spt" else recoveries(natural, "empirical", name),
                            recoveries(natural, "model", name)] for name in ORDER], dtype=float)
        ax.imshow(np.ma.masked_invalid(scores), cmap="YlGnBu", vmin=0, vmax=10, aspect="auto")
        for i in range(8):
            for j in range(4):
                value = scores[i, j]
                ax.text(j, i, "—" if np.isnan(value) else f"{int(value)}/10", ha="center", va="center",
                        fontsize=9, color="white" if value >= 7 else "#27313B")
        ax.set_yticks(range(8), LABELS, fontsize=9)
        ax.set_xticks(range(4), ["Random\nData", "Random\nFit", "Equilibrium\nData", "Equilibrium\nFit"], fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("b  Exact module recovery", loc="left", fontsize=11, weight="bold", pad=10)
        ax.text(0, -.21, "Fit: all methods share q(X)Pfit(Y|X), q(X)=1/64.\nSPT fits 4,800 endpoint pairs per dataset; no true module mask.",
                transform=ax.transAxes, va="top", fontsize=8, linespacing=1.5)
        ax = fig.add_subplot(bottom[1])
        codes, card = observations("transition")
        info = DiscreteInformation(codes, spin_kernel(**uniform["config"]).ravel()/64, card)
        pairs = [([4], [5]), ([1, 2, 3], [4]), ([4, 5], [6])]
        values = [info.mi(a, b)/info.h(a+b) for a, b in pairs]
        labels = ["4 vs 5", "123 vs 4", "45 vs 6\n(if 45 had formed)"]
        ax.barh(range(3), values, color=[ORANGE, "#A8B1BA", GREEN], height=.52)
        ax.set_yticks(range(3), labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1e-5, .3)
        ax.set_xlabel("MIC similarity I(A;B) / H(A,B)")
        for i, value in enumerate(values):
            ax.text(value*1.12, i, f"{value:.2g}", va="center", fontsize=8)
        ax.set_title("c  A greedy merge misses the triplet", loc="left", fontsize=11, weight="bold", pad=10)
        ax.text(0, -.21, "123 absorbs a triplet node before any triplet pair forms.\nThe strong group dependence is reached too late.",
                transform=ax.transAxes, va="top", fontsize=8, linespacing=1.5)
        ax = fig.add_subplot(bottom[2])
        modules = uniform["population_spt"]["tree"]["children"]
        total = np.array([node["xi_bits"] for node in modules])
        residual = np.array([node["syn_bits"] for node in modules])
        share = residual/total
        ax.bar([0, 1], 1-share, color="#C8D1D8", width=.55, label="Retained in child pair")
        ax.bar([0, 1], share, bottom=1-share, color=GREEN, width=.55, label="Split residual Syn")
        for i, value in enumerate(share):
            ax.text(i, 1-value/2, f"{value:.1%}", ha="center", va="center", color="white", weight="bold")
            ax.text(i, 1.04, f"{total[i]:.4f} bits", ha="center", fontsize=8.5)
        ax.set_xticks([0, 1], ["123: pairwise", "456: triadic"], fontsize=9)
        ax.set_ylabel("Fraction of module Xi")
        ax.set_ylim(0, 1.16)
        ax.set_yticks([0, .5, 1], ["0", "50%", "100%"])
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=8)
        ax.set_title("d  SPT preserves the order contrast", loc="left", fontsize=11, weight="bold", pad=10)
        ax.text(0, -.38, "Split residual is grouped synergy, not a pure third-order atom.",
                transform=ax.transAxes, va="top", fontsize=7.5)
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def make_report(uniform, stationary):
    rows, natural = uniform["conditions"][1:], stationary["conditions"]
    kernel = spin_kernel(**uniform["config"])
    codes, card = observations("transition")
    info = DiscreteInformation(codes, kernel.ravel()/64, card)
    modules = uniform["population_spt"]["tree"]["children"]
    lines = ["# 六节点混合阶动力学搜索：Glauber 自旋模型", "",
             "2026-10-17。**找到一个有条件满足目标的动力学：混合二体/三体 p-spin Hamiltonian 上的连续时间 Glauber 动力学。** 随机初态短转移中，SPT 在 10 个独立确认种子上全部恢复 123｜456，直接经验 MIC 为 0/10；同一个拟合转移模型也提供给 MIC 后，为 1/10。Ward 同样为 10/10。", "",
             "**自然稳态采样的限制：该动力学的平衡分布短轨迹中，直接经验 MIC 也达到 10/10。** 因此，本轮结果支持受控随机初态下的机制比较，尚未找到在自然稳态完整转移数据上稳定击败 MIC 的例子。不能把干预重建结果改称自然观测聚类失败。", "",
             "![完整比较](assets/kuramoto_hierarchy/mixed_order_glauber_comparison.png)", "",
             "图 a 为同一精确完整转移通道、均匀独立初态下的原生树/平面分组；枝长只表示层级拓扑。图 b 保留所有确认种子，Data 为直接观测聚类，Fit 为共享同一拟合通道后的均匀干预重建；SPT 仅在拟合通道上计算。图 c 显示 MIC 的实际贪心选择。图 d 为 SPT 模块内分配，不能解读为纯三阶信息原子。", "",
             "## 1. 动力学与不变拓扑", "",
             r"令 $P=\{1,2,3\}$，$T=\{4,5,6\}$，$x_i\in\{-1,+1\}$。采用传统 p-spin 的乘积型 Hamiltonian：", "",
             r"$$H(\boldsymbol{x})=-\sum_{i<j\in P}w_{ij}x_ix_j-J_3x_4x_5x_6-\frac{K_{\rm out}}{3}\sum_{i\in P,j\in T}x_ix_j.\tag{1}$$", "",
             r"$w_{12}=\sqrt{1.5}$，$w_{13}=w_{23}=\sqrt{1.5}/4$；$J_3=1$，$K_{\rm out}=0.04$，$\beta=1$。456 内部三个成对系数严格为零；跨组九条边都保留。采用乘积型三体项，避免“全体对齐”三体指标展开后夹带二体项。传统 p-spin 与全体对齐型高阶 Ising 的差别见 [Robiglio 等，补充材料 S4a–S4b](https://arxiv.org/html/2411.19618v1)。本实验是该模型族的六节点混合阶实例。", "",
             r"每个节点以单位尝试率进行单自旋 heat-bath 翻转，$\boldsymbol{x}^{(i)}$ 表示只翻转第 $i$ 位：", "",
             r"$$[\boldsymbol{Q}]_{\boldsymbol{x},\boldsymbol{x}^{(i)}}=\frac{1}{1+\exp[\beta(H(\boldsymbol{x}^{(i)})-H(\boldsymbol{x}))]},\quad [\boldsymbol{Q}]_{\boldsymbol{x},\boldsymbol{x}}=-\sum_i[\boldsymbol{Q}]_{\boldsymbol{x},\boldsymbol{x}^{(i)}},\quad \boldsymbol{P}_{\tau}=\exp(\tau\boldsymbol{Q}),\quad\tau=2.\tag{2}$$", "",
             "这是连续时间随机自旋动力学，用 64×64 矩阵指数精确求有限时间转移概率；没有时间步离散误差。原始 Glauber 动力学来源：[Glauber (1963)](https://doi.org/10.1063/1.1703954)。本实验将局部能量差扩展为式 (1) 的混合阶 Hamiltonian。不是 Kuramoto，也不是将三体项改名为成对作用。", "",
             "## 2. 数据、拟合与公平比较", "",
             "每个数据集均为 4,800 对六节点短转移端点，滞后 τ=2。所有方法读取相同端点；主要聚类对象为六个节点的完整 (初态,末态) 变量块。增量与仅末态视图完整保留为敏感性分析。", "",
             "两套独立采样：①随机初态：每位独立等概率 ±1，选择种子 0，冻结后确认种子 1–10；②自然平衡分布：初态从该动力学精确 Gibbs 稳态独立抽样，确认种子 1001–1010，随后沿同一通道演化。后者是真实平衡分布的独立短轨迹，不是一条长路径上相邻采样点。两套结果分开呈现。", "",
             "SPT 的因子化最大熵干预始终为 q(初态)=1/64，完整六节点末态为固定共同目标。直接使用观测联合分布计算 EI 会改变该定义，因此未这样处理自然平衡数据。自然数据先拟合完整转移核，再施加独立均匀干预。", "",
             "样本拟合使用通用的 41 参数自旋模型：六个单点场、全部 15 条成对项和全部 20 个三体项，全部开放估计；不提供模块标签，不固定真实稀疏掩码，也不做依赖分量分解。已知条件为 heat-bath 模型族、单位尝试率与滞后 τ。L2 系数 10 在确认种子前冻结；参数界为 [-4,4]，零初值，端点条件似然优化。该拟合使用了动力学模型族知识，不能宣称是无模型发现。没有训练 MLP。", "",
             "对照分两层：Data 直接在同一端点数据上聚类；Fit 将拟合所得相同 q(初态)Pfit(末态|初态) 提供给所有方法。Fit 因而控制了“只有 SPT 获得动力学模型”这一信息差。另用真实精确通道计算全部算法，排除有限样本 MI 偏差。", "",
             "离散 MIC 采用原文式 (9) 的 I(A;B)/H(A,B)，每次拼接变量并重新计算组间 MI；另登记式 (10) 风格的按维数归一化敏感性，未选择有利版本。参见 [Kraskov 等的 MIC 原文](https://arxiv.org/html/q-bio/0311037)。本实验的 MIC 是 mutual-information clustering，不是 maximal information coefficient。离散状态使用原生 Shannon 熵，未加抖动或使用连续 KSG。其他七法设置沿用此前比较：K-means k=2/n_init=50/random_state=0；Ward 欧氏；谱聚类绝对对齐相关 affinity/cluster_qr/k=2；相关、距离相关和成对 MI 为 average linkage。", "",
             "## 3. 完整转移视图：独立确认结果", "",
             "成功定义为树根或原生两类精确等于 123｜456。分母均为 10 个独立数据种子，不含选择种子 0。", "",
             "| 方法 | 随机初态 Data | 随机初态 Fit | 自然稳态 Data | 自然数据拟合后 Fit |", "|---|---:|---:|---:|---:|"]
    for name, label in zip(ORDER, LABELS):
        vals = ["—" if name == "spt" and layer == "empirical" else f"{recoveries(r,layer,name)}/10"
                for r, layer in ((rows,"empirical"),(rows,"model"),(natural,"empirical"),(natural,"model"))]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "SPT 不能解释为压过所有对照：Ward 在这两套数据上也全部正确，谱聚类也经常正确。按维数归一化 MIC 在随机初态完整转移视图同为 Data 0/10、Fit 1/10；在自然数据完整转移视图同为 Data 10/10、Fit 0/10。Fit 列描述均匀干预分布下的重建聚类，不能替代左侧自然观测结果。", "",
              "视图也会改变结论：随机初态的增量视图中，真实精确通道的 MIC **可以**正确分团，拟合模型 MIC 为 9/10；直接经验 MIC 为 0/10，说明此处还有组熵有限样本偏差。自然稳态的增量 MIC 为 Data 10/10、Fit 10/10。不能把主视图的失败扩大成所有表示下都失败。", "",
              "## 4. 为什么精确 MIC 仍可能分错", "",
              "以下数值全部来自随机独立初态、真实精确完整转移通道，单位 bits。先将 1、2 合并，再纳入 3 后，MIC 的下一步为：", "",
              f"- 三体团内任意两节点的 MI 约为 {info.mi([4],[5]):.9f}，归一化相似度 {info.mi([4],[5])/info.h([4,5]):.9g}。",
              f"- 已形成的 123 与节点 4 的 MI 为 {info.mi([1,2,3],[4]):.9f}，归一化相似度 {info.mi([1,2,3],[4])/info.h([1,2,3,4]):.9g}。",
              f"- 假如 45 已形成，I(45;6) 为 {info.mi([4,5],[6]):.9f}，相似度 {info.mi([4,5],[6])/info.h([4,5,6]):.9g}，远强于前两者。", "",
              "因此 MIC 先将 4 接入 123；一次错误合并不可撤回，随后才暴露出很强的三体组依赖。这是贪心起步的限制，不是互信息不能表达高阶依赖。精确总体 MIC 的错误两团为 12345｜6；对称节点的具体编号可随平局规则变化，但 456 已无法形成独立分支。", "",
              "在 Kout=0 的纯三体平衡分布中，p(x4,x5,x6)∝exp(βJ3 x4x5x6)，任意两位的边缘独立，而三位乘积有偏。这给出了机制直觉。实际比较保留非零弱耦合，且完整端点变量还含时间信息，不能把该理想极限的“两两完全独立”原封不动套用到实验。", "",
              "SPT 从完整六源集合出发枚举 31 个根二分，使用共同完整未来目标；最小分割损失发生在 123｜456。根 Syn 为 0.000035514 bits，严格保留了弱跨组作用，没有强制设零。第二优根分割的损失为 0.002815809 bits，最优间隔约 0.002780294 bits。", "",
              "## 5. 是否保留二体主导与三体主导的差别", "",
              "| 模块 | 模块总 Ξ / bits | 最优内部二节点子组的 Ξ / bits | 该模块分割 Syn / bits | 分割 Syn / 模块 Ξ |", "|---|---:|---:|---:|---:|"]
    for node in modules:
        total, syn = node["xi_bits"], node["syn_bits"]
        lines.append(f"| {''.join(map(str,node['sources']))} | {total:.9f} | {total-syn:.9f} | {syn:.9f} | {syn/total:.1%} |")
    lines += ["", "123 的大部分协同已保留在 12 子组，456 则主要体现为拆开整组三节点时损失的协同。该差别与已知 Hamiltonian 结构相符。这里的三节点 SPT 残差是分组协同，包含跨该二分的所有作用；不将其直接命名为纯三阶 PEID 原子。普通聚类树能够给出分团，但没有原生的因果阶数贡献输出。", "",
              "## 6. 搜索记录、数值审计与局限", "",
              "保留全部筛选结果，没有删掉成功对照。初始并行 heat-bath 48 组：完整转移总体 MIC 仅 1 组失败；该组 SPT 正确但总 Ξ 只有 0.000593676 bits，降低优先级。额外弱耦合强度敏感性 135 组中 MIC 失败 30 组，没有为它们新增 SPT。连续时间 Glauber 63 组中完整转移总体 MIC 失败 11 组；在这些失败组中按末态三位乘积矩的绝对值最大选定 β=J3=1、τ=2、Kout=.04。该排名未用 SPT 或 EI。只为上述两个参数配置计算 SPT，后续均为固定配置的数据重复。", "",
              "高阶 LV 的正值 Ricker 离散版本筛选 12 组（自抑制 .5/2，步长 .25/.5，对数噪声 .04/.15/.4），每组同样 4,800 条随机初态短转移：MIC 在完整转移与增量两视图均为 12/12 正确，未找到目标差异，因此未消耗 SPT 候选名额。LV 模型的本体含 NiNj 成对项与 NiNjNk 三体项，参照 [Malizia 等，Eq. (5)](https://www.nature.com/articles/s41467-024-49278-x)；本实验的乘法离散化及参数是明确改编。之前 Hénon 的 MIC 也保持 3/3，未改写为失败。", "",
              "另做 12 组相关翻转噪声通道的机制筛选，MIC 虽全部失败，但其三体结构在噪声联合律而非节点间状态力中；不满足本任务对固定动力学交互拓扑的强解释，未作为达标候选，也未计算 SPT。", "",
              "所有 SPT 使用完整枚举 EI，63 个非空源子集、全部 301 个二分，固定完整六节点未来目标。Syn 容差 1e-10 bits；任何低于 -容差的值直接失败。总体与全部 21 次拟合通道均无显著负值，也无容差内负值，未裁剪。总体最小候选 Syn 为 1.2955865e-6 bits。闭合误差均在数值精度内。没有连续 EI 积分，也没有依赖分量近似；有限离散状态直接枚举比 TM 更精确。", "",
              "通用模型的梯度用有限差分核对，最大绝对差 2.3e-7；检查了独立通道 EI 可加性、已知 Hamiltonian 与通用生成矩阵一致性、Gibbs 详细平衡、离散加权 dCor 与重复样本实现一致性。20 次预定随机重新编号均保持 SPT 正确和完整转移总体 MIC 错误，排除特定节点编号的偶然选择。5 项数值测试通过，全部拟合收敛且没有触及参数界。独立 4,800 对测试转移上，相对真实通道的平均额外负对数似然约 0.0045 nats/转移；该校验没有参与参数或超参数选择。", "",
              "研究性质为探索性筛选加冻结参数后的独立种子确认，不是预注册算法排名。10 个种子只支持此动力学、滞后、样本量和表示；未证明普适优越性。拟合结果依赖已知模型族；自然系统不满足 heat-bath 假设时不能照搬。自然平衡数据上的 MIC 成功意味着本轮尚未实现更强的自然稳态优势目标。", "",
              "## 7. 当前稿件一致性与复现", "",
              "本任务重新核对 Zotero 父项 P6UJCVG8：标题 Emergent hierarchical organization of causal interactions in complex systems；当前唯一主稿 DXGC7JEA、补充稿 MWIWKSVG，未发现明确稿件版本或日期，最新版本歧义仍保留。依据主稿 Methods (5)–(12)、SI S1 离散精确 EI、S5/Algorithm S1。共同因子化干预、共同完整未来目标、原始残差选分与递归到单点保持一致。未借用旧 PEID 稿件替代。", "",
              "实验实现：`scripts/search_six_node_mixed_dynamics.py`；仅从结果重绘与生成本报告：`scripts/report_six_node_mixed_dynamics.py`。筛选、确认、拟合系数、数据哈希与完整树保存在 `results/six_node_mixed_dynamics_search/` 的 JSON 中；协议与运行记录在 `docs/log/six_node_mixed_order_search/`。这些目录遵循现有忽略规则，属本地支持材料。", "",
              "复现时请先重新取得计算预算；脚本不自行延长已批准截止。生成报告仅读取现有结果，不重跑实验。"]
    REPORT.write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    uniform = json.loads((OUT/"ctmc_confirmation.json").read_text())
    stationary = json.loads((OUT/"ctmc_stationary_confirmation.json").read_text())
    make_figure(uniform, stationary)
    make_report(uniform, stationary)
    print(FIG)
    print(REPORT)
