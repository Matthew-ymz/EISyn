"""Preview the Figure 1 redesign using existing cached evidence only.

Contract: 183 mm native publication width, a--e narrative composite and one
complete hierarchy-comparison supplement. PNG previews and editable PDF exports
are saved separately from the current manuscript assets. No fit, estimator,
hierarchy, scoring rule, or intervention protocol is changed.

Curves: cached means and sample SD (ddof=1) over the same four paired data
replicates, verified against raw runs; information is converted bits -> nats.
SHAP retains its native scale in a separate aligned axis. Cases I/III retain
compose_spt_three_examples.collect() and its Syn audits and data views. Case II
uses the first available paired 48,000-transition MLP/baseline confirmation
record, with matched sample hashes and the same scoring and Syn audit rules.
Node fill is local split
Syn / total tree Xi, with the same 0--100% normalization in every tree.
Tree heights encode topology only. These three examples are not pooled into a
statistical algorithm ranking. No clipping, smoothing, or projection is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import compose_part1_figure1 as original
from scripts import compose_spt_three_examples as hierarchy

OUT = ROOT / "paper_assets/figure1_review_preview"
INK = "#26333E"
TEAL = "#237D73"
BLUE = "#4477A8"
ORANGE = "#B56828"
PURPLE = "#8065A0"
GRAY = "#66737D"
PALE = "#B9C1C8"
RECOVERED = "#176A96"
INCORRECT = "#BA4F28"
SYN_CMAP = colors.LinearSegmentedColormap.from_list(
    "local_syn_share", ["#FFFFFF", "#A7CCC6", "#327F74"]
)
F1_CMAP = colors.LinearSegmentedColormap.from_list(
    "module_f1", ["#F3F5F6", "#B8D4D1", "#236F66"]
)
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": .5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": .45,
    "ytick.major.width": .45,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}


def load_current_hierarchies():
    """Read existing evidence; replace the superseded CTMC fit in case II.

    Select the lowest available confirmation seed with a paired SPT/baseline
    cache, independently of its score. No seed or method is tuned for this plot.
    """
    raw, cases = hierarchy.collect()
    directory = ROOT / "results/six_node_data_driven_tpm/confirmation"
    comparison = json.loads((directory / "clustering_baselines_two_seeds.json").read_text())
    eligible = [row for row in comparison["conditions"] if row["sample_count"] == 48000]
    if not eligible:
        raise FileNotFoundError("No paired 48,000-transition MLP/baseline cache for case II")
    paired = min(eligible, key=lambda row: row["seed"])
    record = json.loads((directory / f"n48000_seed{paired['seed']}.json").read_text())
    assert record["sample_count"] == paired["sample_count"] == 48000
    assert record["seed"] == paired["seed"]
    assert not record["mechanism_prior_in_data_driven_fits"]
    assert not record["conditional_output_factorization"]
    for key in ("source_sha256", "future_sha256"):
        if record[key] != paired[key]:
            raise ValueError(f"SPT/baseline sample mismatch in case II: {key}")
    model = record["methods"]["mlp"]
    if model != paired["spt_mlp"]:
        raise ValueError("Paired baseline cache does not match its MLP SPT record")
    result = model["spt"]
    case = cases[1]
    case.update(
        spt=copy.deepcopy(result["tree"]), field="syn_bits", approximate=False,
        note="48,000 transitions · generic joint-output MLP",
        view_note="Baselines: initial + future",
        methods={name: copy.deepcopy(paired["layers"]["data"]["methods"][name])
                 for name in hierarchy.METHODS},
        audit=hierarchy.validate_syn(result["tree"], "syn_bits", record["syn_tolerance_bits"],
                                     result["syn_audit"], model["root_xi_bits"]),
        displayed_seed=record["seed"], sample_count=record["sample_count"],
        evidence_path=str(directory / f"n48000_seed{record['seed']}.json"),
    )
    case["scores"] = {"spt": hierarchy.module_f1(case["spt"], case["truth"])}
    case["scores"].update({name: hierarchy.module_f1(case["methods"][name]["tree"], case["truth"])
                          for name in hierarchy.METHODS})
    assert case["scores"]["spt"] == model["module_f1"]
    for name in hierarchy.METHODS:
        assert case["scores"][name] == case["methods"][name]["module_f1"]
    raw["displayed_glauber_mlp"] = record
    print(f"Case II: paired 48,000-transition MLP record, seed {record['seed']}; "
          f"Syn tolerance={record['syn_tolerance_bits']} bits; "
          f"minimum candidate={result['syn_audit']['minimum_syn_bits']:.6g}; "
          "no violations or clipping")
    return raw, cases


def axes_mm(fig, box):
    w, h = fig.get_size_inches() * 25.4
    x, y, bw, bh = box
    return fig.add_axes((x / w, y / h, bw / w, bh / h))


def text_mm(fig, x, y, text, **kwargs):
    w, h = fig.get_size_inches() * 25.4
    return fig.text(x / w, y / h, text, **kwargs)


def letter(fig, x, y, s):
    text_mm(fig, x, y, s, fontsize=8, fontweight="bold", va="top")


def arrow(ax, start, end, *, color=INK, lw=.65, ls="-", rad=0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=5,
        connectionstyle=f"arc3,rad={rad}", lw=lw, color=color, linestyle=ls,
        shrinkA=1, shrinkB=1,
    ))


def draw_principle(fig):
    letter(fig, 2, 153, "a")
    text_mm(fig, 6, 150.5, "Factorized intervention", fontsize=6.3)
    ax = axes_mm(fig, (2.5, 123.5, 55.5, 26))
    ax.set(xlim=(0, 8.9), ylim=(-.55, 3.6), aspect="equal")
    ax.axis("off")
    for x1, x2, ty in [(1.3, 2.5, 1.9), (5.68, 7.86, 6.77)]:
        xx, yy = np.meshgrid(np.linspace(x1-1.05, x2+1.05, 320), np.linspace(-.05, 3.15, 210))
        s1 = (xx-x1)**2 + (yy-1)**2 <= 1.0**2
        s2 = (xx-x2)**2 + (yy-1)**2 <= 1.0**2
        target = (xx-ty)**2 + (yy-1.9)**2 <= 1.2**2
        cap = target & (yy > 2.35 - .45*((xx-ty)/1.2)**2)
        region = np.zeros_like(xx, dtype=int)
        region[s1], region[s2], region[s1 & s2] = 1, 2, 3
        region[target & ~s1 & ~s2] = 5
        region[target & s1 & ~s2] = 6
        region[target & s2 & ~s1] = 7
        region[target & s1 & s2] = 8
        region[cap] = 4
        ax.contourf(xx, yy, region, levels=np.arange(-.5, 9, 1), antialiased=True,
                    colors=["white", "#DFEEDB", "#F7E8D4", "#DFDEA8",
                            "#F5F7F9", "#CBDCEE", "#A5C9C5", "#C5B4DB", "#D8D99D"])
        for (x, y), radius, color in [((x1, 1), 1.0, "#50916A"),
                                      ((x2, 1), 1.0, "#D08B36"),
                                      ((ty, 1.9), 1.2, "#6586B2")]:
            ax.add_patch(Circle((x, y), radius, fc="none", ec=color, lw=.6))
        px = np.linspace(ty-1.2, ty+1.2, 120)
        ax.plot(px, 2.35-.45*((px-ty)/1.2)**2, color=GRAY, lw=.4, ls=":")
        shift = -.04 if x1 < 2 else .38
        labels = [(ty-.12, 3.43, "$Y$", 5.6),
                  (x1-.7, -.2, "$X_1$", 5.6), (x2+.7, -.2, "$X_2$", 5.6),
                  (ty, 2.60, r"$H(Y\mid X_1,X_2)$", 5.2),
                  (ty, 2.145, r"$\mathit{Syn}$", 5.2),
                  (x1+shift, 1.53, r"$\mathit{Un}_1$", 5.2),
                  (x2-shift, 1.53, r"$\mathit{Un}_2$", 5.2)]
        if x1 < 2:
            labels.append((ty, 1.22, r"$\mathit{Red}$", 5.2))
        for x, y, label, size in labels:
            ax.text(x, y, label, ha="center", va="center", fontsize=size)
    arrow(ax, (3.4, 1.7), (4.72, 1.7), lw=.6)
    ax.text(4.10, 2.16, r"$\mathrm{do}(\mathbf{x}\sim\mathcal{U})$", ha="center", fontsize=5.2)
    ax.text(4.10, 1.17, r"$X_1\perp X_2$", ha="center", fontsize=5.2)
    text_mm(fig, 29, 120.5, r"$q(X_1,X_2)=q(X_1)\,q(X_2)$", ha="center", fontsize=7)
    text_mm(fig, 10.7, 114.5, r"$\mathrm{Syn}\,=$", fontsize=7)
    text_mm(fig, 18.6, 114.5, r"$\mathrm{EI}((X_1,X_2)\to Y)$", fontsize=7)
    text_mm(fig, 18.6, 110.0, r"$-\mathrm{EI}(X_1\to Y)-\mathrm{EI}(X_2\to Y)$", fontsize=7)


def draw_control(fig):
    letter(fig, 59.5, 153, "b")
    text_mm(fig, 64, 150.5, "Controlled nonlinear system", fontsize=6.3)
    ax = axes_mm(fig, (60, 107.5, 37.5, 39.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    w, x, y, f, z = (.09, .49), (.35, .75), (.35, .22), (.65, .49), (.93, .49)
    dashed = (0, (2.3, 1.6))
    arrow(ax, (.13, .53), (.31, .72), color=BLUE, ls=dashed)
    arrow(ax, (.13, .45), (.31, .25), color=BLUE, ls=dashed)
    # Route the original common-driver arc explicitly above the x node.
    driver_path = MplPath(
        [(.09, .55), (.12, .92), (.36, .99), (.55, .96),
         (.76, .91), (.87, .73), (.91, .55)],
        [MplPath.MOVETO] + [MplPath.CURVE4]*6,
    )
    ax.add_patch(FancyArrowPatch(path=driver_path, arrowstyle="-|>",
                                 mutation_scale=5, lw=.65, color=BLUE, linestyle=dashed))
    ax.text(.24, .57, r"$0.8\beta$", fontsize=5.8, ha="center", va="center")
    ax.text(.24, .39, r"$0.8\beta$", fontsize=5.8, ha="center", va="center")
    ax.text(.70, .82, r"$0.1\beta$", fontsize=5.8, ha="center", va="center")
    arrow(ax, (.39, .73), (.56, .53), color=TEAL, lw=.9, rad=-.2)
    arrow(ax, (.39, .24), (.56, .45), color=TEAL, lw=.9, rad=.2)
    arrow(ax, (.74, .49), (.89, .49), color=TEAL, lw=.9)
    for pos, name in [(w, "$w$"), (x, "$x$"), (y, "$y$"), (z, "$z$")]:
        ax.add_patch(Ellipse(pos, .105, .10, ec=BLUE if name == "$w$" else GRAY,
                             fc="#EBF1F5", lw=.65))
        ax.text(*pos, name, ha="center", va="center", fontsize=6.4)
    ax.add_patch(Ellipse(f, .24, .22, ec=TEAL, fc="#DFEDE9", lw=.8))
    ax.text(*f, r"$\sin(xy)$", ha="center", va="center", fontsize=5.8)
    ax.text(.65, .30, "Fixed", ha="center", fontsize=5.3)
    text_mm(fig, 62, 107, "Common driver", fontsize=5.5)
    text_mm(fig, 83, 107, "Joint interaction", fontsize=5.5)
    fig.add_artist(Line2D([60/183, 61.5/183], [107.6/155]*2, color=BLUE, lw=.7, ls=dashed))
    fig.add_artist(Line2D([80.5/183, 82/183], [107.6/155]*2, color=TEAL, lw=.9))


SERIES = [
    ("mlp_peid_xy_synergy", "MLP+PEID", TEAL, "-", "^", True),
    ("observational_wms", "WMS", BLUE, "--", "D", True),
    ("mmi_pid_xy_synergy", "MMI-PID", PURPLE, "-.", "P", True),
    ("surd_xy_synergy", "SURD", GRAY, ":", "o", True),
    ("shap_xy_mean_abs_interaction", "MLP+SHAP", ORANGE, "--", "s", False),
]


def load_curves():
    data = json.loads(original.CONFOUNDER_READOUT_RESULT.read_text())["updated_full_result"]
    beta = np.array(sorted({row["beta"] for row in data["runs"]}))
    seeds = sorted({row["seed"] for row in data["runs"]})
    assert len(seeds) == 4 and data["config"]["intervention_samples"] == 5120
    lookup = {(row["beta"], row["seed"]): row for row in data["runs"]}
    summary = {row["beta"]: row for row in data["summary"]}
    result = {}
    for key, *_rest, information in SERIES:
        values = np.array([[lookup[b, s][key] for s in seeds] for b in beta])
        if not np.isfinite(values).all():
            raise ArithmeticError(f"Non-finite curve: {key}")
        mean, sd = values.mean(1), values.std(1, ddof=1)
        np.testing.assert_allclose(mean, [summary[b][key+"_mean"] for b in beta], atol=1e-12)
        np.testing.assert_allclose(sd, [summary[b][key+"_std"] for b in beta], atol=1e-12)
        scale = np.log(2.) if information else 1.
        result[key] = (mean*scale, sd*scale)
        if key == "mlp_peid_xy_synergy":
            tol = 0.0  # strict display tolerance, nats; no projection/clipping
            invalid = values*scale < -tol
            if invalid.any():
                raise ArithmeticError(f"Syn violation: min={values.min()*scale}, tolerance={tol}, count={invalid.sum()}")
            print(f"Curve Syn audit: tolerance={tol} nats; minimum={values.min()*scale:.6g}; violations={invalid.sum()}; no clipping")
    return beta, result


def draw_curves(fig, beta, values):
    letter(fig, 100.5, 153, "c")
    ax = axes_mm(fig, (109, 123, 71.5, 22.5))
    shap_ax = axes_mm(fig, (109, 107, 71.5, 11.5))
    handles = []
    for key, label, color, style, marker, information in SERIES:
        target = ax if information else shap_ax
        mean, sd = values[key]
        target.fill_between(beta, mean-sd, mean+sd, color=color, alpha=.10, linewidth=0)
        line, = target.plot(beta, mean, color=color, ls=style, marker=marker,
                            markevery=4, markersize=2.1, markeredgecolor="white",
                            markeredgewidth=.2, lw=1.1 if label=="MLP+PEID" else .7,
                            zorder=4 if label=="MLP+PEID" else 3, label=label)
        handles.append(line)
    ax.axhline(0, color="#9DA6AD", lw=.4, ls=":")
    ax.set(xlim=(-.015, 1.015), ylim=(-.10, .55), yticks=[0, .2, .4])
    ax.tick_params(labelbottom=False)
    shap_ax.set(xlim=(-.015, 1.015), ylim=(.19, .59), yticks=[.2, .4, .6])
    shap_ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Information (nats)", fontsize=5.8, labelpad=2)
    shap_ax.set_ylabel("Attribution\n(native scale)", fontsize=5.3, labelpad=3)
    shap_ax.set_xlabel(r"Common-driver strength $\beta$", fontsize=6, labelpad=1.5)
    for target in (ax, shap_ax):
        target.tick_params(labelsize=5.5)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(146/183, 154.5/155),
               ncol=3, fontsize=5.5, handlelength=1.7, columnspacing=1,
               handletextpad=.4, labelspacing=.35, borderaxespad=0)
    text_mm(fig, 180, 120, r"Mean $\pm$ SD, $n=4$", fontsize=5.2, ha="right")


def draw_topology(ax, case, *, small=False):
    pos = hierarchy.POSITIONS
    def edge(i, j, color, lw, alpha=1, ls="-"):
        ax.plot(*zip(pos[i], pos[j]), color=color, lw=lw, alpha=alpha,
                ls=ls, solid_capstyle="round", zorder=1)
    if case["topology"] == "A":
        for i in (1, 2, 3):
            for j in (4, 5, 6):
                edge(i, j, "#C4CBD0", .28, .65)
        for i, j, lw in [(1, 2, 1.52), (1, 3, .60), (2, 3, .60)]:
            edge(i, j, BLUE, lw)
        ax.add_patch(Polygon([pos[i] for i in (4, 5, 6)], ec=ORANGE,
                             fc="#F0DFCB", lw=.6, zorder=0))
    else:
        if case.get("weak_bridges"):
            for i, j in [(1, 5), (4, 6)]:
                edge(i, j, "#A5B1BA", .4)
        ax.add_patch(Polygon([pos[i] for i in (1, 2, 4)], ec=BLUE, fc="#EDF3F7",
                             lw=.7, ls=(0, (2.5, 1.7)), zorder=0))
        ax.add_patch(Polygon([pos[i] for i in (1, 2, 3)], ec=BLUE,
                             fc="#D1E1EF", lw=.7, zorder=1))
        edge(5, 6, ORANGE, 1)
    for i, (x, y) in pos.items():
        color = BLUE if i <= (3 if case["topology"] == "A" else 4) else ORANGE
        ax.scatter(x, y, s=28 if small else 35, fc=hierarchy.tint(color, .15),
                   ec=color, lw=.6, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=5.4, zorder=5)
    ax.set(xlim=(-1.5, 1.5), ylim=(-.85, .94), aspect="equal")
    ax.axis("off")


def draw_tree(ax, tree, truth, *, field=None, font=5.7):
    tree = copy.deepcopy(tree)
    positions, counter = {}, [0]
    def locate(node):
        children = sorted(node["children"], key=lambda x: min(x["sources"]))
        node["children"] = children
        cc = [locate(c) for c in children]
        if cc:
            xy = (np.mean([p[0] for p in cc]), 1 + max(p[1] for p in cc))
        else:
            xy = (counter[0], 0)
            counter[0] += 1
        positions[tuple(node["sources"])] = xy
        return xy
    _, depth = locate(tree)
    all_nodes = hierarchy.nodes(tree)
    total = sum(node[field] for node in all_nodes if node["children"]) if field else 1
    for node in all_nodes:
        x, y = positions[tuple(node["sources"])]
        for child in node["children"]:
            cx, cy = positions[tuple(child["sources"])]
            ax.plot([x, cx], [y, cy], lw=.5, color=PALE, zorder=1)
    for node in all_nodes:
        x, y = positions[tuple(node["sources"])]
        internal = bool(node["children"])
        label = "".join(str(s) for s in node["sources"])
        status = hierarchy.state(node["sources"], truth)
        edge = RECOVERED if status == hierarchy.GREEN else INCORRECT if status == hierarchy.RED else "#9EABB4"
        linestyle = (0, (2.2, 1.2)) if status == hierarchy.RED else "-"
        face = "white"
        if internal and field:
            share = 100*node[field]/total
            if not 0 <= share <= 100:
                raise ArithmeticError(f"Invalid unprojected Syn share: {share}")
            label += "\n" + ("<0.1%" if 0 < share < .05 else f"{share:.1f}%")
            face = SYN_CMAP(share/100)
        bbox = dict(boxstyle="round,pad=.17", fc=face, ec=edge,
                    lw=.65 if status != hierarchy.GRAY else .4,
                    linestyle=linestyle) if internal else None
        ax.text(x, y, label, fontsize=font, ha="center", va="center", linespacing=1.03,
                bbox=bbox, color=INK, zorder=3)
    ax.set(xlim=(-.45, 5.45), ylim=(-.24, depth+.32))
    ax.axis("off")


def draw_status_legend(fig, anchor, *, incorrect=True, fontsize=5.4):
    handles = [Patch(fc="white", ec=RECOVERED, lw=.7, label="Recovered module")]
    if incorrect:
        handles.append(Patch(fc="white", ec=INCORRECT, lw=.7, ls=(0, (2.2, 1.2)), label="Unsupported merge"))
    handles.append(Patch(fc="white", ec="#9EABB4", lw=.4, label="Unscored"))
    return fig.legend(handles=handles, loc="upper left", bbox_to_anchor=anchor,
                      ncol=len(handles), fontsize=fontsize, handlelength=1.3,
                      handletextpad=.4, columnspacing=1.15, borderaxespad=0)


def draw_main_hierarchies(fig, cases, *, with_baseline=False):
    letter(fig, 2, 96, "d")
    if with_baseline:
        # Use the same comparator in every row, including its successful case I.
        # All five baseline scores remain visible in e; all trees remain in SI.
        text_mm(fig, 57.5, 91, "SPT", ha="center", fontsize=6.5, fontweight="bold")
        text_mm(fig, 99.5, 91, "MIC", ha="center", fontsize=6.2, fontweight="bold")
    else:
        text_mm(fig, 81.5, 93, "SPT", ha="center", fontsize=6.5, fontweight="bold")
    draw_status_legend(fig, (29/183, 97/155), incorrect=with_baseline)
    for idx, case in enumerate(cases):
        bottom = 66.5 - idx*30
        name = f"{case['id']}   {case['name']}" + (" · nested" if case['id']=="III" else "")
        text_mm(fig, 4.5, bottom+23, name, fontsize=6.2, fontweight="bold")
        if idx == 2 and not with_baseline:
            text_mm(fig, 115, bottom+23, r"$\{1,2,3\}\subset\{1,2,3,4\}$", fontsize=5.4, ha="right")
        draw_topology(axes_mm(fig, (5, bottom+.5, 32, 20)), case)
        if with_baseline:
            draw_tree(axes_mm(fig, (40, bottom, 35, 21)), case["spt"], case["truth"],
                      field=case["field"], font=5.5)
            draw_tree(axes_mm(fig, (82, bottom, 35, 21)), case["methods"]["mic"]["tree"],
                      case["truth"], font=5.5)
        else:
            draw_tree(axes_mm(fig, (44, bottom, 75, 21)), case["spt"], case["truth"], field=case["field"])


def draw_scores(fig, cases):
    letter(fig, 117, 96, "e")
    text_mm(fig, 122, 93.5, r"Module $F_1$", fontsize=6.5)
    order = ("spt", *hierarchy.METHODS)
    labels = ["SPT", *hierarchy.LABELS]
    matrix = np.array([[case["scores"][name] for case in cases] for name in order])
    ax = axes_mm(fig, (136, 22.5, 44, 64))
    im = ax.imshow(matrix, cmap=F1_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax.set_xticks(range(3), ["I", "II", "III"])
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, labelsize=6, pad=3)
    for i, label in enumerate(labels):
        text_mm(fig, 134, 22.5+(5.5-i)*64/6, label, fontsize=6,
                ha="right", va="center", fontweight="bold" if i==0 else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.3f}", fontsize=6.4, ha="center", va="center",
                color="white" if value > .75 else INK)
    text_mm(fig, 129, 18.8, r"$F_1$", fontsize=6, ha="right", va="center")
    cax = axes_mm(fig, (136, 18, 44, 1.8))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)
    text_mm(fig, 129, 8.8, "Syn share", fontsize=5.7, ha="right", va="center")
    cax = axes_mm(fig, (136, 8, 44, 1.8))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 100), cmap=SYN_CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.set_xticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1.5)


def build_main(cases, beta, values, *, with_baseline=False):
    fig = plt.figure(figsize=(183/25.4, 155/25.4), facecolor="white")
    draw_principle(fig)
    draw_control(fig)
    draw_curves(fig, beta, values)
    draw_main_hierarchies(fig, cases, with_baseline=with_baseline)
    draw_scores(fig, cases)
    return fig


def build_supplement(cases):
    fig = plt.figure(figsize=(183/25.4, 135/25.4), facecolor="white")
    draw_status_legend(fig, (3/183, 133/135), incorrect=True, fontsize=5.7)
    left = [3, 29, 59, 83.3, 107.6, 131.9, 156.2]
    widths = [23, 27, 21.5, 21.5, 21.5, 21.5, 23.5]
    titles = ["", "SPT", *hierarchy.LABELS]
    for x, width, title in zip(left, widths, titles):
        text_mm(fig, x+width/2, 120.5, title, fontsize=6, fontweight="bold", ha="center", va="center")
    for i, case in enumerate(cases):
        bottom = 84 - i*36.4
        name = f"{case['id']}   {case['name']}" + (" · nested" if case['id']=="III" else "")
        text_mm(fig, 3, bottom+28.5, name, fontsize=6, fontweight="bold")
        draw_topology(axes_mm(fig, (left[0], bottom+1.5, widths[0], 24)), case, small=True)
        draw_tree(axes_mm(fig, (left[1], bottom, widths[1], 26)), case["spt"], case["truth"],
                  field=case["field"], font=5.3)
        for j, name in enumerate(hierarchy.METHODS, 2):
            draw_tree(axes_mm(fig, (left[j], bottom, widths[j], 26)),
                      case["methods"][name]["tree"], case["truth"], font=5.3)
    text_mm(fig, 3, 3.8, "Branch height encodes topology only.", fontsize=5.2)
    text_mm(fig, 113, 3.8, "SPT fill: local Syn / total Ξ (%)", fontsize=5.2)
    cax = axes_mm(fig, (151, 7, 28, 1.9))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 100), cmap=SYN_CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.1, length=1.4, pad=1)
    return fig


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches=None)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None)
    plt.close(fig)
    print(stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--with-baseline", action="store_true",
                        help="Preview paired SPT/MIC trees in all three main-figure rows")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, cases = load_current_hierarchies()
    beta, curves = load_curves()
    with plt.rc_context(STYLE):
        stem = "figure1_with_baseline" if args.with_baseline else "figure1_redesigned"
        save(build_main(cases, beta, curves, with_baseline=args.with_baseline), args.out_dir/stem)
        save(build_supplement(cases), args.out_dir/"figure_s_hierarchy_comparison")
    print("Existing MLP evidence and F1 scores verified; sample SD uses ddof=1; no fits or experiments.")


if __name__ == "__main__":
    main()
