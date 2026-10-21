"""Compose three existing six-node examples; read caches only, never refit.

Working figure contract: compare recovered module hierarchies on two interaction
topologies, with three rows and a common method order. Node percentages are local
split Syn / sum of all internal-node Syn, not fractions of a module's own Xi.
Use the original phase-increment view in I, endpoint blocks in II, and the
previously reported best-of-two-view baseline selection in III. These are
exploratory examples with different protocols, not a pooled algorithm ranking.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "docs/reports/assets/kuramoto_hierarchy/spt_three_example_comparison.png"
REPORT = ROOT / "docs/reports/spt_three_example_comparison.md"
WEAK = ROOT / "results/spt_third_case_weak_coupling"
INPUTS = {
    "original": ROOT / "results/kuramoto_short_trajectory_clustering/summary.json",
    "glauber": ROOT / "results/six_node_mixed_dynamics_search/ctmc_confirmation.json",
    "nested_spt": ROOT / "results/kuramoto_six_topology_search/generic_fourier_x4/B_2_seed0/spt.json",
    "nested_baselines": ROOT / "results/kuramoto_six_topology_search/generic_fourier_x4/B_2_seed0/baselines_raw.json",
}
TEAL, GREEN, RED, GRAY = "#237D73", "#26734D", "#BF4B45", "#7D8993"
BLUE, ORANGE, INK = "#4477A8", "#B56828", "#26333E"
FONT_BODY, FONT_HEADER, FONT_PANEL = 9.0, 10.5, 12.5
FIGURE_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                "font.size": FONT_BODY, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                "axes.unicode_minus": False, "text.color": INK, "axes.labelcolor": INK,
                "xtick.color": INK, "ytick.color": INK}
METHODS = ("mic", "mi_hac", "dcor_hac", "correlation_hac", "ward")
LABELS = ("MIC", "MI-HAC", "dCor-HAC", "AbsCorr-HAC", "Ward")
MODEL_EXPLANATION = r"""## Glauber 动力学的数学表达

此例采用连续时间、逐个自旋翻转的 Glauber（heat-bath）动力学。六个节点的状态为 $x_i\in\{-1,+1\}$，记 $P=\{1,2,3\}$、$T=\{4,5,6\}$，能量函数为

$$
H(\boldsymbol{x})=-\sum_{\substack{i<j\\i,j\in P}}w_{ij}x_ix_j
-J_3x_4x_5x_6
-\frac{K_{\mathrm{out}}}{3}\sum_{i\in P}\sum_{j\in T}x_ix_j.
$$

三项分别对应 123 内的成对作用、456 的三体作用，以及两团间的弱成对耦合。本例固定 $w_{12}=\sqrt{1.5}$、$w_{13}=w_{23}=\sqrt{1.5}/4$、$J_3=1$、$K_{\mathrm{out}}=0.04$ 和逆温度 $\beta=1$；456 内没有直接成对能量项。

令 $\boldsymbol{x}^{(i)}$ 表示仅翻转第 $i$ 个自旋后的状态。单位尝试率下，翻转速率为

$$
c_i(\boldsymbol{x})=
\frac{1}{1+\exp\!\left[\beta\bigl(H(\boldsymbol{x}^{(i)})-H(\boldsymbol{x})\bigr)\right]},
\qquad
\Pr\!\left[\boldsymbol{x}(t+\mathrm{d}t)=\boldsymbol{x}^{(i)}\mid\boldsymbol{x}(t)=\boldsymbol{x}\right]
=c_i(\boldsymbol{x})\,\mathrm{d}t+o(\mathrm{d}t).
$$

降低能量的翻转速率更高，升高能量的翻转仍有非零速率。它描述随机跳变过程，时间连续、状态离散。概率分布满足主方程

$$
\frac{\mathrm{d}p_t(\boldsymbol{x})}{\mathrm{d}t}
=\sum_{i=1}^{6}\left[
c_i(\boldsymbol{x}^{(i)})p_t(\boldsymbol{x}^{(i)})
-c_i(\boldsymbol{x})p_t(\boldsymbol{x})\right].
$$

相应的 $64\times64$ 生成矩阵满足 $[\boldsymbol{Q}]_{\boldsymbol{x},\boldsymbol{x}^{(i)}}=c_i(\boldsymbol{x})$，对角元为所有离开速率之负和，完整转移矩阵为 $\boldsymbol{P}_{\tau}=\exp(\tau\boldsymbol{Q})$；实验取 $\tau=2$。以上参数和方程对应本仓库的混合二体/三体扩展，不能将三体 Hamiltonian 本身归为原始一维 Ising 例子。实现位置：`scripts/search_six_node_mixed_dynamics.py` 的 `spin_kernel`，`family="glauber_ctmc"`。

## nested Kuramoto 中的 nested

**nested 就是“嵌套”：一个小作用模块完整包含在更大的作用模块内。** 此例同时设置三体超边 $\{1,2,3\}$ 和四体超边 $\{1,2,3,4\}$，即 $\{1,2,3\}\subset\{1,2,3,4\}$；另有独立于这两个支持集合的二体模块 $\{5,6\}$。这里“嵌套”描述植入的交互支持结构，不是只因输出树有多层才这样命名。

具体相位动力学为

$$
\mathrm{d}\theta_i=
\left[\omega_i+
\sum_{\substack{e\in\mathcal{E}\\i\in e}}
\kappa_e\sin\!\left(\sum_{j\in e\setminus\{i\}}\theta_j-(|e|-1)\theta_i\right)
\right]\mathrm{d}t+\sigma\,\mathrm{d}W_i,
\qquad
\mathcal{E}=\bigl\{\{1,2,3\},\{1,2,3,4\},\{5,6\}\bigr\}.
$$

所选配置中 $\kappa_{123}=\kappa_{56}\approx0.56695$、$\kappa_{1234}\approx1.13389$，$\sigma=0.08$，$\boldsymbol{\omega}=(-0.4,0,0.4,-0.4,0,0.4)^{\mathsf T}$。目标是同时恢复外层 $1234\mid56$ 和内层 $123\mid4$。只找对外层两团的方法仍可能漏掉内层 123。实现位置：`scripts/kuramoto_six_topology_core.py` 的 `SUPPORTS["B"]`、`configuration("B", 2)` 和 `drift`。
"""
POSITIONS = {1: (-1.12, .72), 2: (-1.25, -.62), 3: (-.28, -.05),
             4: (.28, -.05), 5: (1.25, -.62), 6: (1.12, .72)}


def load(path):
    return json.loads(path.read_text())


def nodes(tree):
    return [tree] + [n for child in tree["children"] for n in nodes(child)]


def scored_clades(tree, truth):
    """Keep true modules and unsupported merges; ignore primitive refinements.

    This is the existing nested benchmark metric. Common root and singleton
    leaves are unscored. Refinements inside a primitive planted module have no
    unique binary truth and are excluded, including true pairwise refinements.
    """
    truth = {frozenset(s) for s in truth}
    primitives = {s for s in truth if not any(t < s for t in truth)}
    return {frozenset(n["sources"]) for n in nodes(tree)
            if 1 < len(n["sources"]) < 6
            and not any(set(n["sources"]) < p for p in primitives)}


def module_f1(tree, truth):
    pred, true = scored_clades(tree, truth), {frozenset(s) for s in truth}
    return 2 * len(pred & true) / (len(pred) + len(true))


def validate_syn(tree, field, tolerance, audit, total=None):
    values = np.array([n[field] for n in nodes(tree) if n["children"]])
    if not np.isfinite(values).all():
        raise ArithmeticError("Non-finite cached Syn")
    count = int(np.sum(values < -tolerance))
    if count:
        raise ArithmeticError(f"Syn violation: minimum={values.min()}, threshold={-tolerance}, affected={count}")
    for key in ("violation_count", "violating_atom_count", "violating_partition_syn_count",
                "violating_non_singleton_xi_count", "xi_violation_count"):
        if audit.get(key, 0):
            raise ArithmeticError(f"Cached nonnegativity audit failed: {key}={audit[key]}")
    raw_total = float(values.sum())
    if total is not None and not np.isclose(raw_total, total, atol=1e-9):
        raise ArithmeticError(f"Tree closure failure: {raw_total} versus {total}")
    if raw_total <= 0:
        raise ArithmeticError("Cannot normalize a nonpositive total Syn")
    # No projection is used in this composite. Fail even for tolerance-scale
    # negative displayed nodes so that percentage colors cannot hide them.
    if np.any(values < 0):
        raise ArithmeticError(f"Negative displayed Syn needs explicit handling: min={values.min()}, "
                              f"tolerance={tolerance}, affected={int(np.sum(values < 0))}")
    return {"native_field": field, "tolerance": tolerance,
            "minimum_displayed_syn": float(values.min()),
            "negative_displayed_count": int(np.sum(values < 0)),
            "native_audit": audit, "total": raw_total, "clipping": False}


def collect():
    raw = {name: load(path) for name, path in INPUTS.items()}
    result_path = WEAK/'spt_projection.json' if (WEAK/'spt_projection.json').exists() else WEAK/'spt.json'
    weak_result = load(result_path) if result_path.exists() else None
    if weak_result and weak_result.get('tree'):
        raw['nested_spt'] = weak_result
        raw['nested_baselines'] = load(WEAK/'baselines_raw.json')
        raw['weak_configuration'] = load(WEAK/'configuration.json')
    original = next(c for c in raw["original"]["conditions"] if c["k_out"] == .04)
    g = next(c for c in raw["glauber"]["conditions"] if c["seed"] == 0)
    ref = original["spt_reference"]
    simple_truth = [[1, 2, 3], [4, 5, 6]]
    cases = [dict(id="I", name="Kuramoto", topology="A", truth=simple_truth,
                  note="4,800 transitions · seed 0", view_note="Baselines: phase increment",
                  spt=ref["tree"], field="syn_bits", approximate=False,
                  audit=validate_syn(ref["tree"], "syn_bits", .1,
                                     ref["syn_audit"], ref["root_xi_bits"]),
                  methods=copy.deepcopy(original["views"]["increment"]["methods"])),
             dict(id="II", name="Glauber", topology="A", truth=simple_truth,
                  note="4,800 transitions · seed 0", view_note="Baselines: initial + future",
                  spt=g["fitted_spt"]["tree"], field="syn_bits", approximate=False,
                  audit=validate_syn(g["fitted_spt"]["tree"], "syn_bits", 1e-10,
                                     g["fitted_spt"]["syn_audit"],
                                     g["fitted_spt"]["tree"]["xi_bits"]),
                  methods=copy.deepcopy(g["empirical"]["transition"]["methods"])),
             dict(id="III", name="Kuramoto", topology="B",
                  truth=[[1, 2, 3], [1, 2, 3, 4], [5, 6]],
                  note="24,576 training pairs · seed 0", view_note="Baselines: best of two views*",
                  spt=raw["nested_spt"]["tree"], field="syn_nats",
                  approximate=not raw['nested_spt'].get('numerically_resolved', False),
                  weak_bridges='weak_configuration' in raw,
                  audit=validate_syn(raw["nested_spt"]["tree"], "syn_nats", .01,
                                     raw["nested_spt"]["audit"]), methods={})]
    for name in METHODS:
        options = [(view, raw["nested_baselines"]["views"][view]["methods"][name])
                   for view in ("increment", "transition")]
        key = lambda pair: module_f1(pair[1]["tree"], cases[2]["truth"])
        view, selected = max(options, key=key)
        cases[2]["methods"][name] = dict(copy.deepcopy(selected), selected_view=view)
    for case in cases:
        case["methods"] = {name: case["methods"][name] for name in METHODS}
        case["scores"] = {"spt": module_f1(case["spt"], case["truth"])}
        case["scores"].update({name: module_f1(case["methods"][name]["tree"], case["truth"])
                               for name in METHODS})
    fit_mic = g["model"]["transition"]["methods"]["mic"]
    cases[1]["scores"]["mic_fit"] = module_f1(fit_mic["tree"], simple_truth)
    return raw, cases


def tint(color, amount):
    return tuple(1 - amount + amount * c for c in to_rgb(color))


def state(sources, truth):
    s, true = set(sources), [set(t) for t in truth]
    primitive = [t for t in true if not any(q < t for q in true)]
    return GREEN if s in true else GRAY if len(s) in (1, 6) or any(s < t for t in primitive) else RED


def draw_tree(ax, tree, truth, *, field=None, approximate=False):
    tree = copy.deepcopy(tree)
    def canonical(n):
        n["children"].sort(key=lambda c: min(c["sources"]))
        for c in n["children"]:
            canonical(c)
    canonical(tree)
    leaf_x, position, count = {}, {}, [0]
    def locate(n):
        children = [locate(c) for c in n["children"]]
        if children:
            x, y = np.mean([p[0] for p in children]), 1 + max(p[1] for p in children)
        else:
            x, y = count[0], 0
            leaf_x[n["sources"][0]] = x
            count[0] += 1
        position[tuple(n["sources"])] = (x, y)
        return x, y
    _, height = locate(tree)
    total = sum(n[field] for n in nodes(tree) if n["children"]) if field else 1.
    for n in nodes(tree):
        x, y = position[tuple(n["sources"])]
        for c in n["children"]:
            cx, cy = position[tuple(c["sources"])]
            ax.plot([x, cx], [y, cy], color="#B5BEC5", lw=1.05, zorder=1)
    for n in nodes(tree):
        x, y = position[tuple(n["sources"])]
        internal = bool(n["children"])
        label = "".join(map(str, n["sources"]))
        color = state(n["sources"], truth)
        fontsize = FONT_BODY
        if field and internal:
            share = 100 * n[field] / total
            # Keep a nonzero small share visibly distinct from an exact zero.
            value = "<0.1%" if 0 < share < .05 else f"{share:.1f}%"
            label += "\n" + ("≈" if approximate else "") + value
            # A common 0–100% scale across rows; border status is independent
            # of the quantitative fill, including for unscored refinements.
            face = tint(TEAL, .75 * share / 100)
        else:
            face = "white"
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                linespacing=1.12, color=INK, zorder=3,
                bbox=dict(boxstyle="round,pad=.22", fc=face, ec=color,
                          lw=1.15 if color in (GREEN, RED) else .75))
    ax.set(xlim=(-.46, 5.46), ylim=(-.42, height + .45))
    ax.axis("off")


def draw_topology(ax, kind, weak_bridges=False):
    def edge(i, j, color, lw, alpha=1):
        p = np.array([POSITIONS[i], POSITIONS[j]])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, alpha=alpha, zorder=1,
                solid_capstyle="round")
    if kind == "A":
        for i in (1, 2, 3):
            for j in (4, 5, 6):
                edge(i, j, "#BAC0C7", .65, .65)
        for i, j, w in ((1, 2, 3.8), (1, 3, 1.5), (2, 3, 1.5)):
            edge(i, j, BLUE, w)
        p = np.array([POSITIONS[i] for i in (4, 5, 6)])
        ax.add_patch(Polygon(p, fc=tint(ORANGE, .20), ec=tint(ORANGE, .60), lw=1.4, zorder=0))
    else:
        if weak_bridges:
            for i, j in ((1, 5), (4, 6)):
                edge(i, j, '#ADB8C1', .85, .85)
        # Node 3 is inside this hull and belongs to the four-body hyperedge.
        p = np.array([POSITIONS[i] for i in (1, 2, 4)])
        ax.add_patch(Polygon(p, fc=tint(BLUE, .08), ec=BLUE, ls="--", lw=1.3, zorder=0))
        p = np.array([POSITIONS[i] for i in (1, 2, 3)])
        ax.add_patch(Polygon(p, fc=tint(BLUE, .20), ec=BLUE, lw=1.45, zorder=1))
        edge(5, 6, ORANGE, 2.4)
    for i, (x, y) in POSITIONS.items():
        left = i <= (3 if kind == "A" else 4)
        color = BLUE if left else ORANGE
        ax.scatter([x], [y], s=210, fc=tint(color, .18), ec=color, lw=1.1, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=FONT_BODY, color=INK, zorder=5)
    ax.set(xlim=(-1.55, 1.55), ylim=(-.88, .97), aspect="equal")
    ax.axis("off")


def draw_comparison(fig, cases, *, bounds=(0., 0., 1., 1.), panel_letters=False,
                    panel_labels=("c", "d")):
    """Draw native axes within a region without scaling the physical text size."""
    x0, y0, width, height = bounds
    def point(x, y):
        return x0 + width * x, y0 + height * y
    def rectangle(x, y, w, h):
        return (*point(x, y), width * w, height * h)
    with plt.rc_context(FIGURE_STYLE):
        grid = fig.add_gridspec(3, 7, left=point(.018, 0)[0], right=point(.987, 0)[0],
                               top=point(0, .925)[1], bottom=point(0, .205)[1],
                               width_ratios=(1.12, 1.40, 1.10, 1.10, 1.10, 1.10, 1.10),
                               hspace=.16, wspace=.12)
        headers = ("", "SPT", *LABELS)
        for j, title in enumerate(headers):
            box = grid[0, j].get_position(fig)
            fig.text((box.x0 + box.x1) / 2, point(0, .953)[1], title, ha="center", va="center",
                     fontsize=FONT_HEADER, weight="bold", color=TEAL if j == 1 else INK)
        for i, case in enumerate(cases):
            cell = grid[i, 0].get_position(fig)
            topo = fig.add_axes([cell.x0, cell.y0 + .005 * height, cell.width, cell.height - .05 * height])
            draw_topology(topo, case["topology"], case.get('weak_bridges', False))
            fig.text(cell.x0, cell.y1 - .008 * height, f"{case['id']}   {case['name']}", fontsize=FONT_HEADER,
                     weight="bold", color=INK)
            ax = fig.add_subplot(grid[i, 1])
            draw_tree(ax, case["spt"], case["truth"], field=case["field"], approximate=case["approximate"])
            for j, name in enumerate(METHODS, 2):
                ax = fig.add_subplot(grid[i, j])
                draw_tree(ax, case["methods"][name]["tree"], case["truth"])
            if i < 2:
                y = (cell.y0 + grid[i + 1, 0].get_position(fig).y1) / 2
                fig.add_artist(Line2D([point(.018, 0)[0], point(.987, 0)[0]], [y, y], transform=fig.transFigure,
                                      color="#E0E5E9", lw=.7))

        fig.text(*point(.019, .168), "Module F1", fontsize=FONT_HEADER, weight="bold", color=INK)
        ax = fig.add_axes(rectangle(.018, .024, .969, .124))
        ax.axis("off")
        # Shared-fit MIC is a Glauber-only diagnostic, retained in the report
        # and manuscript supplement rather than the cross-example table.
        order = ("spt", "mic", "mi_hac", "dcor_hac", "correlation_hac", "ward")
        columns = ["Example", "SPT", *LABELS]
        rows = [[c["id"] + "  " + c["name"],
                 *[f"{c['scores'][name]:.3f}" + ("†" if c['approximate'] and name == "spt" else "")
                   if name in c["scores"] else "—" for name in order]] for c in cases]
        table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                         colWidths=[.18] + [.82 / len(order)] * len(order), bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_BODY)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(1.5)
            cell.set_facecolor("#EDF1F4" if r == 0 else ("#F4F7F8" if r % 2 else "#FAFBFC"))
            cell.set_text_props(color=INK, weight="bold" if r == 0 or c == 1 else "normal")
            if c == 0:
                cell.set_text_props(ha="left")
            if c == 1 and r > 0:
                cell.set_facecolor(tint(TEAL, .10))
                cell.set_text_props(color=TEAL, weight="bold")
        legend = [Patch(fc="white", ec=GREEN, lw=1.3, label="Recovered module"),
                  Patch(fc="white", ec=RED, lw=1.3, label="Incorrect merge"),
                  Patch(fc="white", ec=GRAY, lw=1.0, label="Unscored")]
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=point(.05, .995), ncol=3,
                   frameon=False, fontsize=FONT_BODY, handlelength=1.2, columnspacing=2)
        fig.text(*point(.986, .981), "SPT fill: Syn share (%)", ha="right", fontsize=FONT_BODY)
        if panel_letters:
            for letter, y in zip(panel_labels, (.998, .185)):
                fig.text(*point(.004, y), letter, fontsize=FONT_PANEL, weight="bold",
                         ha="left", va="top", color=INK)


def render(cases):
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(19.0, 9.1), facecolor="white")
        draw_comparison(fig, cases)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=260, facecolor="white")
        plt.close(fig)


def write_report(raw, cases):
    weak = cases[2].get('weak_bridges', False)
    intro = ("2026-10-17。I/II 沿用旧结果；III 在用户批准的 60 分钟预算内加入两条固定弱跨团边，重新生成共享数据、拟合模型、计算基线与完整六维 EI。未搜索有利参数。"
             if weak else "2026-10-17。仅重排已有缓存，没有重训、重算 EI、增加数值积分或搜索参数。")
    explanation = MODEL_EXPLANATION
    if weak:
        explanation = explanation.replace(
            r"\mathcal{E}=\bigl\{\{1,2,3\},\{1,2,3,4\},\{5,6\}\bigr\}",
            r"\mathcal{E}=\bigl\{\{1,2,3\},\{1,2,3,4\},\{5,6\},\{1,5\},\{4,6\}\bigr\}")
        explanation += "\n\n本轮另外固定 $\\kappa_{15}=\\kappa_{46}=0.02$，原三个团内系数不重新归一化。上式实际交互支持包含弱桥边；模块 F1 真值仍为 123、1234、56。\n"
    lines = ["# 三个六节点例子的统一比较", "", intro, "",
             "![三个例子的拓扑、SPT、基线输出和评分](assets/kuramoto_hierarchy/spt_three_example_comparison.png)", "",
             "三排分别为：I 混合阶 Kuramoto；II 同一交互拓扑上的 Glauber；III Kuramoto。拓扑 A 重画于前两排，以明确两个动力学共享同一交互支持，而不是相同动力学方程或参数。拓扑 B 为三体 123 嵌入四体 1234，另有二体 56。" +
             ("本轮还加入权重各 0.02 的 1–5、4–6 两条弱跨团边，两团不再严格断开。" if weak else "") +
             "拓扑 B 的四体面包含内部节点 3；没有把面边界解释为新增成对边。", "",
             "绿色边框为真实模块；红色边框为未被真实模块支持的合并；灰色边框为不评分的原生模块内部细分、共同根和叶子。边框分类与填色数值独立。SPT 使用全图共享的 0–100% 青绿色顺序色阶，零值为白色，填色深度线性编码节点 Syn 占比；基线框内不填色。颜色不表示纯阶 PID 原子。所有树只保留真实二叉拓扑，允许交换左右子树改善编号顺序；纵坐标不编码动力学时间或可跨方法比较的聚类距离。拓扑的蓝色与橙色区分模块身份，III 的 123 与 1234 均使用蓝色；线与填充面区分二体作用与高阶支持，不以颜色表示作用阶数。", "",
             "SPT 内部节点标注 `100 × 该节点 Syn / 全树 Ξ`，分母是全部内部节点 Syn 之和。叶子 Syn 为零，不重复写百分比；不是以各模块自身 Ξ 为分母。" +
             ("III 使用约等号标记尚未通过数值精度检查。" if cases[2]['approximate'] else "III 已通过声明的数值精度检查，去掉未收敛提示符；仍是学得通道的有限精度估计。") +
             "百分比经显示舍入后不必精确加至 100%。本版按用户要求隐藏全树总量 Ξ 的数值，只展示节点百分比，计算与归一化不变。", "",
             "## 图中小字的含义与删减", "",
             "图中只保留方法名、例子编号与动力学名称、节点编号、Syn 百分比、三项颜色图例和模块 F1 表格。以下文字移到本说明，图内不再重复：", "",
             "| 原标注 | 含义 |", "|---|---|",
             "| `Topology A/B`、`same as I` | 两种交互支持；I/II 共享同一拓扑，III 使用嵌套支持。 |",
             "| `pairwise / 2-body`、`triadic / 3-body`、`4-body` | 二体、三体和四体作用，分别连接两个、三个和四个节点；填充面表示高阶作用，不是面边界上的新增成对边。 |",
             "| `Kout=0.04`、`bridges 1–5, 4–6: 0.02` | 团间弱耦合参数；III 的两条桥边各为 0.02，图中的细灰边保留。 |",
             "| `4,800 transitions`、`24,576 training pairs`、`seed 0` | 图示实验的样本数和随机种子；表中得分来自这些输出，不是多种子均值。 |",
             "| `Baselines: phase increment / initial + future / best of two views` | 基线读取的变量表示：I 用相位增量，II 用初态与未来状态，III 对每种基线取预设两视图中得分较高者。 |",
             "| `learned-component approximation`、`generic spin model`、`numerical checks passed` | 拟合通道的实现方式及数值积分检查；具体限制保留在下文。 |",
             "| `10/10`、`0/10`、`1/10` | II 在十个新数据种子上的精确根恢复次数，属于独立确认结果，不能当作 F1。 |",
             "| `local Syn / whole-tree Ξ` | 节点百分比的定义；现在简写为 `Syn share (%)`。Ξ 是所有内部节点局部 Syn 之和，其绝对数值已从图中删除。 |", "",
             "图中的 `MIC (data)` 表示直接对样本计算组互信息并构树。`MIC (shared fit)` 使用与 SPT 相同的拟合通道，只在 II 做了这一附加控制；其得分保留在下表及论文补充材料中，不占主图比较表的一列。绿色表示恢复的真实模块，红色表示错误合并，灰色表示不计分的根、叶子或模块内部细分。F1 越大越好，1 为完全恢复；下表中的破折号表示该项未提供，不能读作零。", "",
             "## 统一评分", "",
             "模块 F1 = 2TP/(预测的可评分模块数＋真实模块数)。I/II 真值为 123、456；III 为 123、1234、56。共同六源根和单点叶子不评分；原生最小模块内部的二叉细分没有唯一真值，沿用嵌套实验口径而排除；其他错误合并计为假阳性。因而不能把 123 内部的任意配对标红。本版仅展示层级方法；右下角额外小树也已移除，共享拟合通道 MIC 的控制分数保留在本报告的下表中。", "",
             "| 方法 | I Kuramoto | II Glauber | III Kuramoto |", "|---|---:|---:|---:|"]
    for name, label in [("spt", "SPT：模块 F1"), *zip(METHODS, LABELS),
                        ("mic_fit", "MIC：共享拟合通道 F1")]:
        scores = [f"{c['scores'][name]:.3f}" if name in c["scores"] else "—" for c in cases]
        lines.append(f"| {label} | " + " | ".join(scores) + " |")
    lines += ["", "以上均为图示 seed 0 的实际输出得分，不是十种子平均分。II 的独立确认结果另列为精确根恢复次数：SPT 10/10、Ward 10/10、MIC Data 0/10、MIC Fit 1/10，不能将这些次数直接替换成 F1。II 自然平衡分布的 Data MIC 也为 10/10，本图只呈现随机独立初态实验。", "", explanation, "",
              "## 输入与解释边界", "",
              "I：原始 4,800 条短转移、seed 0、Kout=0.04。基线固定用原主视图（相位增量），不是逐算法择优。SPT 沿用已学通道的依赖分量近似、机制 Fourier context 和完整六维增量目标；不是只替换聚类规则的公平排名。弱跨边根 Syn=0 来自 learned-channel 可分近似，不代表物理跨边消失。", "",
              "II：选择种子 0 的同一 4,800 对端点。主树基线用经验完整转移块，SPT 用这些样本拟合的通用 41 参数 heat-bath CTMC 通道。表中另保留同一拟合通道下 MIC 的控制分数，以区分数据拟合与构树差别。不使用总体精确通道的节点百分比冒充样本拟合结果；拟合通道内部的离散 EI 是精确枚举，不适用连续 TM。", "",
              "III：总共 32,768 对共同样本，其中 24,576 训练、4,096 验证、4,096 测试；图中基线读取共同训练样本。" +
              (f"SPT 使用本轮已完成的最高档 {raw['nested_spt']['outer']}×{raw['nested_spt']['particles']}、四次 scramble 平均 EI。"
               if weak else "SPT 读取最高档 2048×512、两次 scramble 平均 EI 的缓存树。") +
              "沿用既有报告中对每个基线在两视图中择较高得分的口径（此举有利于基线），选取如下：", ""]
    for name in METHODS:
        lines.append(f"- {name}: {cases[2]['methods'][name]['selected_view']}")
    precision = ("III 的模块 F1=1 是当前平均估计树的事实；原 `numerically_resolved=False` 仍保留。最大 EI SEM=0.01796 nats，跨档变化=0.05445 nats，最差 ESS≈4.46，最大归一化积分权重≈0.466；未通过原准入。百分比和强度尚未收敛，也没有新数据种子确认。"
                 if not weak else "III 本轮精度检查：`" + json.dumps({k: raw['nested_spt'].get(k) for k in
                    ('numerically_resolved', 'maximum_ei_sem_nats', 'grid_change_nats', 'minimum_ess',
                     'maximum_weight', 'scramble_module_tree_stable', 'maximum_share_sem_percentage_points',
                     'maximum_share_grid_change_percentage_points')}, ensure_ascii=False) +
                 "`。物理数据仍只有 seed 0；四次 scramble 是数值积分重复，不是跨数据种子验证。配置、精度门槛和原约等号原因见 `docs/ref/spt_third_case_weak_coupling.md`。")
    if weak and raw['nested_spt'].get('estimator'):
        precision += ("本图使用完整条件 TM 积分加上解析零均值的噪声控制及独立数值校准得到的 Fourier 方差控制；"
                      "EI 的总体定义不变。原始 EI、原始精度、控制系数与全部校准候选并列保存；"
                      "不能将受控估计的收敛改写成原始估计已经通过。")
        inner_path = WEAK/'inner_precision_check.json'
        if inner_path.exists():
            inner = load(inner_path)
            precision += "固定外层样本、P=512→1024 的配对检查：`" + json.dumps({k: inner[k] for k in (
                'maximum_mean_ei_change_nats', 'maximum_ei_difference_sem_nats',
                'maximum_node_share_change_percentage_points', 'passed')}, ensure_ascii=False) + "`。"
    lines += ["", precision, "",
              "## Syn 非负检查", "",
              "仅读取原生单位的缓存。没有 max(0, Syn)、剪裁或投影；显著负值显式失败，显示树若出现任何负值也停止，避免颜色掩盖它。I 全部候选沿用原审计；II 为拟合离散表全部 301 个二分；III 沿用完整六维 EI 表审计。", ""]
    for c in cases:
        a = c["audit"]
        lines.append(f"- {c['id']}: 容差 {a['tolerance']} {a['native_field'].removeprefix('syn_')}；"
                     f"图示节点最小值 {a['minimum_displayed_syn']:.10g}；显示负值数 {a['negative_displayed_count']}；"
                     f"原审计 `{json.dumps(a['native_audit'], ensure_ascii=False)}`。")
    lines += ["", "## 当前稿件核对", "",
              "本任务重新查询本地 Zotero，核对父项 P6UJCVG8 标题为 *Emergent hierarchical organization of causal interactions in complex systems*；重新列举附件并读取正文 DXGC7JEA（19 页）的 Methods (5)–(12)，补充 MWIWKSVG（28 页）的 S1.1、S5/Algorithm S1 与 S7。当前唯一正文/补充附件均无明确稿件版本或稿件日期，元数据修改时间不视为新稿件版本，无法消除最新版本歧义。", "",
              "节点采用稿件式 (11) 局部二分残差，按式 (12) 全树闭合后归一化。I 的固定增量 target、learned-component 近似等与一般完整未来状态定义的差异保留说明；III 的完整未来相位 target 与 I 不相同。没有把不同目标、单位、样本量、模型先验和验证水平混成跨例子总排名。" +
              ("本轮更新第三例的动力学、共享数据模型和积分精度；没有据此宣称全部稿件/实现差异已解决。" if weak else "这里只统一展示和评分，不宣称全面解决稿件/实现差异。"), "",
              "## 可复现来源", "", "绘图入口：`scripts/compose_spt_three_examples.py`；使用仓库 `.venv/bin/python` 直接运行，仅依赖现有结果。原实验报告和结果均未覆盖。", ""]
    for name, path in INPUTS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"- {name}: `{path.relative_to(ROOT)}`；SHA256 `{digest}`")
    if weak:
        for name in ('configuration.json', 'model.json', 'data_manifest.json', 'baselines_raw.json', 'spt.json', 'protocol.json'):
            path = WEAK/name
            lines.append(f"- 本轮 {name}: `{path.relative_to(ROOT)}`；SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
        for name in ('spt_projection.json', 'spt_projection_levels.json', 'fourier_control.json', 'mps_ei_equivalence.json',
                     'inner_precision_check.json', 'jacobian_evaluation_equivalence.json'):
            path = WEAK/name
            if path.exists():
                lines.append(f"- 数值审计 {name}: `{path.relative_to(ROOT)}`；SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raw, cases = collect()
    render(cases)
    write_report(raw, cases)
    print(FIGURE)
    print(REPORT)
    print(json.dumps({c["id"]: c["scores"] for c in cases}, ensure_ascii=False))
