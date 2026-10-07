"""Figure/report export for the frozen 979-person cohort.

The default presentation excludes the unverified REST inputs and compares the
four available tasks. Existing fitted arrays are reused without refitting.
"""
from __future__ import annotations

import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

LABELS = ["REST", "Emotion", "Language", "Motor", "WM"]
NET_LABELS = ["Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default"]
REST_COLOR = "#527FA8"
TASK_COLOR = "#CE8856"
LN2 = math.log(2)
TASK_STATES = ["EMOTION", "LANGUAGE", "MOTOR", "WM"]
SYN_TOLERANCE_BITS = 1e-4


def stars(q):
    return "***" if q<.001 else "**" if q<.01 else "*" if q<.05 else "ns"


def mark(axis, label):
    axis.text(-.08, 1.06, label, transform=axis.transAxes, fontsize=13, fontweight="bold")


def distribution(axis, values, labels, colors, seed):
    rng = np.random.default_rng(seed)
    bp = axis.boxplot(values, positions=np.arange(len(values)), widths=.52,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="#2F3336", linewidth=1.1),
                      whiskerprops=dict(color="#69737C"), capprops=dict(color="#69737C"))
    for i,(v,c,box) in enumerate(zip(values,colors,bp["boxes"])):
        box.set(facecolor=c, alpha=.19, edgecolor=c)
        axis.scatter(i+rng.uniform(-.17,.17,len(v)),v,s=4,color=c,alpha=.16,
                     linewidths=0,rasterized=True,zorder=2)
        axis.scatter(i,np.mean(v),s=38,marker="D",facecolor="white",edgecolor="#30383E",linewidth=.9,zorder=4)
    axis.set_xticks(np.arange(len(labels)),labels)
    axis.set_xlim(-.55,len(values)-.45)
    axis.spines[["top","right"]].set_visible(False)
    axis.tick_params(length=3)


def heatmap(fig, axis, values, xlabels, ylabels, *, label, cmap="viridis", vmin=0, vmax=None, percent=False, center=False):
    if center:
        lim = max(float(np.max(np.abs(values))),1e-8)
        vmin,vmax=-lim,lim
    image = axis.imshow(values,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    axis.set_xticks(np.arange(len(xlabels)),xlabels,rotation=28,ha="right")
    axis.set_yticks(np.arange(len(ylabels)),ylabels)
    axis.tick_params(length=0,pad=5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = image.cmap(image.norm(values[i,j]))
            luminance=.2126*color[0]+.7152*color[1]+.0722*color[2]
            text=f"{values[i,j]:.1f}%" if percent else f"{values[i,j]:+.2f}" if center else f"{values[i,j]:.2f}"
            axis.text(j,i,text,ha="center",va="center",fontsize=7.1,
                      color="white" if luminance<.46 else "#172127")
    fig.colorbar(image,ax=axis,fraction=.045,pad=.025,label=label)
    return image


def render_results(out, arrays, summary, *, include_rest=False):
    if include_rest:
        return render_results_with_rest(out, arrays, summary)
    return render_task_results(out, arrays, summary)


def render_task_results(out, arrays, summary):
    """A: task distributions/pairs; B: tree order mass; C: network shares."""
    out = Path(out)
    variants = list(np.asarray(arrays["variants"]).astype(str))
    indices = [variants.index(state) for state in TASK_STATES]
    labels = LABELS[1:]
    subjects = np.asarray(arrays["subjects"])
    n = len(subjects)
    if n != 979 or len(set(subjects.tolist())) != n:
        raise ValueError("Expected the frozen 979 distinct paired participants")
    xi_bits = np.asarray(arrays["system_xi_bits"])[indices]
    mass_bits = np.asarray(arrays["order_mass_bits"])[indices]
    percent = np.asarray(arrays["network_percent"])[indices]
    for name, values in (("system_xi", xi_bits), ("order_mass", mass_bits), ("network_percent", percent)):
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name} values")
    # Audit the existing native Syn estimates before any figure-only zero handling.
    if summary["numerical_audit"]["all_candidate_syn"]["significant_negative_count"]:
        raise ValueError("Source audit contains significant Syn nonnegativity violations")
    significant = mass_bits < -SYN_TOLERANCE_BITS
    if significant.any():
        raise ValueError(f"Syn/order mass violation: minimum={mass_bits.min():.12g} bits; "
                         f"threshold={-SYN_TOLERANCE_BITS}; count={significant.sum()}")
    numerical_zero_count = int((mass_bits < 0).sum())
    display_mass_bits = np.where(mass_bits < 0, 0, mass_bits)
    cross_bits = np.asarray(arrays["cross_xi_bits"])[indices]
    if not np.allclose(mass_bits.sum(axis=-1), cross_bits, atol=1e-8, rtol=0):
        raise ValueError("Tree order mass does not close to cross-network Xi")
    if not np.allclose(percent.sum(axis=-1), 100, atol=1e-6, rtol=0):
        raise ValueError("Participant network shares do not sum to 100%")
    xi = xi_bits * LN2
    mean_mass = display_mass_bits.mean(axis=1).T * LN2
    mean_percent = percent.mean(axis=1).T

    pairs = list(combinations(range(4), 2))
    p = np.asarray([float(wilcoxon(xi[i] - xi[j], alternative="two-sided", method="approx").pvalue)
                    if np.any(xi[i] != xi[j]) else 1.0 for i, j in pairs])
    rank = np.argsort(p)
    q = np.empty_like(p)
    q[rank] = np.minimum(1, np.minimum.accumulate((p[rank] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    comparisons = [dict(left=TASK_STATES[i], right=TASK_STATES[j],
                        mean_difference_nats=float((xi[i] - xi[j]).mean()),
                        p=float(pv), q=float(qv), annotation=stars(float(qv)))
                   for (i, j), pv, qv in zip(pairs, p, q)]

    # Preserve the earlier REST-containing presentation as diagnostic history.
    archive = out / "rest_diagnostic"
    archive.mkdir(exist_ok=True)
    for source, target in (("hcp_979_abc.png", "hcp_979_abc_with_unverified_rest.png"),
                           ("report.md", "report_with_unverified_rest.md")):
        if (out / source).exists() and not (archive / target).exists():
            if source == "report.md":
                previous_report = (out / source).read_text().replace(
                    str((out / "hcp_979_abc.png").resolve()),
                    str((archive / "hcp_979_abc_with_unverified_rest.png").resolve()))
                (archive / target).write_text(previous_report)
            else:
                shutil.copy2(out / source, archive / target)

    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.labelsize": 10, "axes.linewidth": .8,
                         "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(10.8, 8.2), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.15], hspace=.10, wspace=.12)
        a = fig.add_subplot(grid[0, :])
        b = fig.add_subplot(grid[1, 0])
        c = fig.add_subplot(grid[1, 1])
        distribution(a, list(xi), labels, [TASK_COLOR] * 4, 20261007)
        a.set_ylabel(r"System-level $\Xi$ (nats)")
        a.text(.99, 1.06, f"paired n={n} · six task pairs: Wilcoxon, BH · diamonds: means",
               transform=a.transAxes, ha="right", fontsize=8)
        # Every comparison is above the highest observation; no old REST stars survive.
        maximum = float(xi.max())
        step = .047 * maximum
        levels = {(0, 1): 0, (2, 3): 0, (1, 2): 1, (0, 2): 2, (1, 3): 3, (0, 3): 4}
        for (i, j), qv in zip(pairs, q):
            y = maximum + .045 * maximum + levels[i, j] * step
            h = .011 * maximum
            a.plot([i, i, j, j], [y, y + h, y + h, y], color="#626D75", lw=.65, clip_on=False)
            a.text((i + j) / 2, y + h + .004 * maximum, stars(float(qv)),
                   ha="center", va="bottom", fontsize=9)
        a.set_ylim(min(0, float(xi.min())), maximum * 1.31)
        heatmap(fig, b, mean_mass, labels, list(range(2, 8)), label="SPT order mass (nats)")
        b.set_ylabel("Source-network order")
        heatmap(fig, c, mean_percent, labels, NET_LABELS,
                label="Network attribution (%)", cmap="YlGnBu", vmin=0, percent=True)
        for axis, label in ((a, "a"), (b, "b"), (c, "c")):
            mark(axis, label)
        fig.savefig(out / "hcp_979_abc.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    task_summary = dict(
        n_subjects=n, states=TASK_STATES, data_source=str((out / "arrays.npz").resolve()),
        excluded_from_presentation="REST and all REST length-matched comparisons",
        sample_selection="Existing frozen 979 participants; no additional exclusions or expanded task cohort",
        units="nats (native bits multiplied by ln(2))",
        means_system_xi_nats=xi.mean(axis=1).tolist(),
        mean_order_mass_nats=mean_mass.T.tolist(),
        group_peak_orders=(mean_mass.argmax(axis=0) + 2).tolist(),
        mean_network_percent=mean_percent.T.tolist(),
        paired_comparisons=comparisons, multiplicity_family="All six task pairs",
        test="Two-sided paired Wilcoxon signed-rank, normal approximation; BH across six pairs",
        syn_tolerance_bits=SYN_TOLERANCE_BITS, figure_numerical_zero_count=numerical_zero_count,
        statistical_limitation="Participant-level descriptive/exploratory comparisons; family dependence, motion and unequal task duration not controlled",
        method_reference=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                              reread_date="2026-10-07", sections=["S5.1, S49-S50", "S12.2.2"],
                              explicit_manuscript_version="Unavailable; one main/supplement pair, metadata edits do not establish a version"))
    (out / "tasks_only_summary.json").write_text(json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n")
    report = ["# HCP 979 人：四种任务态 A–C", "",
              "日期：2026-10-07。按用户要求移除新REST及其全部对照，仅展示Emotion、Language、Motor、WM。沿用现有979名统一配对被试及缓存，未重新拟合、调参或扩大样本。", "",
              f"![四任务A–C]({(out / 'hcp_979_abc.png').resolve()})", "",
              "A：每点一名被试；箱体为四分位范围，横线为中位数，须线延伸至1.5倍IQR范围内的最远观测，白色菱形为均值。六组任务对比均为双侧配对Wilcoxon，跨六比较BH校正；*** q<0.001，** q<0.01，* q<0.05，ns q≥0.05。该检验不是均值差检验，星号本身不表示均值差方向。", "",
              "B：按每人所选SPT节点的来源网络数汇总Syn，再取被试均值；2–7阶质量之和为跨网络Ξ，不包含网络内部Ξ。阶数是来源功能网络数量，动力学历史阶数仍固定为3。C：先计算每人七网络对系统Ξ的归因百分比，再取被试均值；每列合计100%。", "",
              "| 任务 | 系统Ξ均值（nats） | 群体峰值阶数 | 6阶质量（nats） | 7阶质量（nats） |", "|---|---:|---:|---:|---:|"]
    for k, state in enumerate(TASK_STATES):
        report.append(f"| {state} | {xi[k].mean():.3f} | {task_summary['group_peak_orders'][k]} | {mean_mass[4,k]:.3f} | {mean_mass[5,k]:.3f} |")
    report.extend(["", "Motor系统Ξ均值最高；四任务的群体平均阶数质量均在6阶达到峰值。这不等于每名被试都以6阶为峰值，也不把网络阶数解释为单脑区阶数。", "",
                   "| 配对比较 | 左减右均值差（nats） | BH q |", "|---|---:|---:|"])
    for row in comparisons:
        report.append(f"| {row['left']} − {row['right']} | {row['mean_difference_nats']:+.4f} | {row['q']:.4g} |")
    report.extend(["", f"原始Syn估计已通过来源审计。绘图容差固定为{SYN_TOLERANCE_BITS:g} bits；容差内负阶数质量显示为零的数量为{numerical_zero_count}，低于负容差将显式失败。", "",
                   "任务时长不同（176、316、284、405帧）；本图为现有结果比较，未新增任务间等时长控制。被试统计尚未控制HCP家系依赖和头动。移除REST不自动解除这些任务比较的解释边界。", "",
                   "本次重新核对Zotero P6UJCVG8的题名及附件：正文DXGC7JEA，补充MWIWKSVG；读取S5.1（S49–S50）和S12.2.2。仅有一组正文/补充附件，显式稿件日期/版本仍不可用。本次只改变展示和配对比较族，模型及指标定义沿用原结果。", "",
                   "精确值、比较范围和图形处理记录保存在tasks_only_summary.json。先前含REST的图和报告作为诊断历史保留在rest_diagnostic目录，不进入本次呈现。"])
    (out / "report.md").write_text("\n".join(report) + "\n")
    return task_summary


def render_results_with_rest(out, arrays, summary):
    out=Path(out)
    with mpl.rc_context({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":9,"axes.labelsize":10,"axes.linewidth":.8,
                         "savefig.facecolor":"white","pdf.fonttype":42}):
        xi=arrays["system_xi_bits"]*LN2
        mass=arrays["order_mass_bits"]*LN2
        percent=arrays["network_percent"]
        # The native audit occurs before this declared figure-only numerical-zero rule.
        display_mass=np.where(mass<0,0,mass)
        fig=plt.figure(figsize=(11.2,7.8),layout="constrained")
        grid=fig.add_gridspec(2,2,height_ratios=[.82,1.15],hspace=.12,wspace=.13)
        a=fig.add_subplot(grid[0,:]); b=fig.add_subplot(grid[1,0]); c=fig.add_subplot(grid[1,1])
        distribution(a,list(xi[:5]),LABELS,[REST_COLOR]+[TASK_COLOR]*4,20261006)
        a.set_ylabel(r"System-level $\Xi$ (nats)")
        ymax=float(xi[:5].max()); ymin=min(0,float(xi[:5].min()))
        a.set_ylim(ymin,ymax*1.13)
        for i,row in enumerate(summary["full_length"]["panel_A"],1):
            a.text(i,ymax*1.045,stars(row["q"]),ha="center",fontsize=11)
        a.text(.99,1.06,"paired n=979 · Wilcoxon, BH · diamond: mean",transform=a.transAxes,ha="right",fontsize=8)
        a.axvline(.5,color="#9DA7AF",ls="--",lw=.8)
        mark(a,"a");mark(b,"b");mark(c,"c")
        heatmap(fig,b,display_mass[:5].mean(axis=1).T,LABELS,list(range(2,8)),
                label="SPT order mass (nats)")
        b.set_ylabel("Source-network order")
        heatmap(fig,c,percent[:5].mean(axis=1).T,LABELS,NET_LABELS,
                label="Network attribution (%)",cmap="YlGnBu",vmin=0,percent=True)
        fig.savefig(out/"hcp_979_abc.png",dpi=300,bbox_inches="tight")
        plt.close(fig)

        fig=plt.figure(figsize=(12.3,7.7),layout="constrained")
        grid=fig.add_gridspec(2,1,height_ratios=[.85,1.1],hspace=.13)
        top=grid[0].subgridspec(1,4,wspace=.12)
        bottom=grid[1].subgridspec(1,2,wspace=.16)
        axes=[fig.add_subplot(top[0,i]) for i in range(4)]
        matched_tests=summary["length_matched"]["panel_A"]
        highest=float(max(xi[1:5].max(),xi[5:].max()))
        for j,axis in enumerate(axes):
            distribution(axis,[xi[j+5],xi[j+1]],["REST",LABELS[j+1]],[REST_COLOR,TASK_COLOR],20261007+j)
            axis.set_ylim(0,highest*1.08)
            row=matched_tests[j]
            axis.text(.5,1.03,f"{stars(row['q'])}   Δ={row['mean_difference']:+.2f} nats",transform=axis.transAxes,ha="center",fontsize=8)
            if j==0:axis.set_ylabel(r"System-level $\Xi$ (nats)")
            else:axis.tick_params(labelleft=False)
        axes[0].text(0,1.22,"paired n=979 · one fixed centered REST window per task · diamonds: means",transform=axes[0].transAxes,fontsize=8)
        mark(axes[0],"a")
        b=fig.add_subplot(bottom[0,0]);c=fig.add_subplot(bottom[0,1])
        order_delta=(mass[1:5]-mass[5:9]).mean(axis=1).T
        pct_delta=(percent[1:5]-percent[5:9]).mean(axis=1).T
        heatmap(fig,b,order_delta,LABELS[1:],list(range(2,8)),label="Task − matched REST (nats)",cmap="RdBu_r",center=True)
        heatmap(fig,c,pct_delta,LABELS[1:],NET_LABELS,label="Task − matched REST (percentage points)",cmap="RdBu_r",center=True)
        b.set_ylabel("Source-network order")
        mark(b,"b");mark(c,"c")
        fig.savefig(out/"hcp_979_length_control.png",dpi=300,bbox_inches="tight")
        plt.close(fig)

    supported_full=[r["task"] for r in summary["full_length"]["panel_A"] if r["mean_difference"]>0 and r["q"]<.05]
    supported_matched=[r["task"] for r in summary["length_matched"]["panel_A"] if r["mean_difference"]>0 and r["q"]<.05]
    lower_full=[r["task"] for r in summary["full_length"]["panel_A"] if r["mean_difference"]<0 and r["q"]<.05]
    lower_matched=[r["task"] for r in summary["length_matched"]["panel_A"] if r["mean_difference"]<0 and r["q"]<.05]
    report=["# HCP 979 人 A–C 复现结果", "", "日期：2026-10-07。仅报告完整 979 人；五状态使用统一配对样本，另做四种固定中央 REST 等长窗口。", "",
            f"全长分析中，REST 的系统 Ξ 显著高于 {len(supported_full)}/4 个任务、低于 {len(lower_full)}/4 个任务；固定等长对照中分别为 {len(supported_matched)}/4 与 {len(lower_matched)}/4。方向取均值差，显著阈值为双侧 BH q<0.05。", "",
            f"![A–C]({(out/'hcp_979_abc.png').resolve()})", "", "## 系统整合信息", "",
            "差值为 REST 减任务，信息单位为 nats；区间为 5000 次被试重采样的均值差 95% 区间。双侧配对 Wilcoxon，每组四比较 BH 校正。图中 *** q<0.001，** q<0.01，* q<0.05；星号本身不表示方向。", "",
            "| 任务 | 全长差值 [95% CI] | q | 等长差值 [95% CI] | q |", "|---|---:|---:|---:|---:|"]
    for full,matched in zip(summary["full_length"]["panel_A"],summary["length_matched"]["panel_A"]):
        def effect(r):
            lo,hi=r["mean_difference_ci95"]
            return f"{r['mean_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        report.append(f"| {full['task']} | {effect(full)} | {full['q']:.3g} | {effect(matched)} | {matched['q']:.3g} |")
    report.extend(["", "## 阶数与网络归因", "",
        "B 按每人选中树节点的来源网络数求和，再取被试均值；2–7 阶之和为跨网络 Ξ。C 先计算每人的网络百分比，再取均值，每列合计 100%。", "",
        "| 状态 | 组均值峰值阶数 | 6 阶质量 | 7 阶质量 | 7 阶归一化占比 | Control 占比 | Limbic 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for state in ("REST","EMOTION","LANGUAGE","MOTOR","WM"):
        r=summary["by_variant"][state]
        report.append(f"| {state} | {r['group_peak_order']} | {r['mean_order_mass_nats'][4]:.3f} | {r['mean_order_mass_nats'][5]:.3f} | {100*r['mean_order_share'][5]:.1f}% | {r['mean_network_percent'][5]:.1f}% | {r['mean_network_percent'][4]:.1f}% |")
    report.extend(["", "7 阶原始质量与归一化占比是不同问题；各自的配对差值、区间与校正结果保存在 summary.json。阶数表示来源网络数量，动力学历史阶数固定为 3。", "",
                  f"![等长对照]({(out/'hcp_979_length_control.png').resolve()})", "",
                  "等长图 A 展示各任务对应的中央 REST 窗口；B、C 展示任务减等长 REST 的均值差。没有移动窗口选择结果。", "",
                  "## 校准和适用范围", ""])
    cal=json.loads((out/"calibration.json").read_text())
    report.append(f"全部旧 57 人、228 个任务的回归重建通过；最大相对误差 {cal['maximum_regression_relative_error']:.3g}。旧指标复算最大误差 {cal['maximum_legacy_metric_error_bits']:.3g} bits。新旧任务逐 ROI 仿射对应最大残差 {cal['maximum_roi_affine_error']:.3g}；新旧网络 PC1 最小绝对相关 {cal['minimum_pca_score_abs_correlation']:.4f}。缩放对 PCA 和指标的实际影响逐被试记录在 calibration.json。")
    rest_check=out/"rest_input_verification.json"
    if rest_check.exists():
        check=json.loads(rest_check.read_text())
        report.extend(["",f"静息输入复核（校准诊断，不另作子集推断）：旧 REST 同样可复算，最大旧缓存误差 {check['maximum_legacy_rest_metric_error_bits']:.3g} bits。在相同校准被试中，旧 REST 平均 Ξ 为 {check['mean_old_rest_xi_nats']:.3f} nats，新 REST 为 {check['mean_new_rest_xi_nats']:.3f} nats。这证明输入差异足以明显改变估计，不能将当前反向结果归因于样本扩大本身；具体预处理原因尚未确定。"])
    report.extend(["",f"共完成 {summary['fit_count']} 个模型。所有候选二分和各级量检查 Syn；容差为 1e-4 bits（约 6.931e-5 nats），显著违规数为 0，最大闭合误差 {summary['maximum_identity_error_bits']:.3g} bits。原始值全部保留；主图阶数质量仅将已审计的容差内负值显示为零，受影响记录数为 {int(np.sum(mass<0))}。"])
    limitations=[
        "新 REST 的 run 标识和去噪来源尚未明确。",
        "新增被试的脑区列序假定遵循提供的 Schaefer1000 标签；目前没有独立的列序来源证明。",
        "被试级统计未控制 HCP 家系依赖和头动，因此总体推断仍属初步。",
        "任务 OLS 使用整个 run；后 25% 仅用于动力学预测诊断，不能称为预处理全流程未触及的测试集。",
        "每任务只使用一个固定中央 REST 窗口，不能等同于旧 12 窗口平均对照。",
        "线性高斯估计提供联合信息的组织描述，不能单独证明不可由成对机制解释的物理高阶作用。"]
    for limitation in limitations:
        report.append("\n- "+limitation)
    report.extend(["", "方法依据：本次重新读取 Zotero 当前可用稿件 P6UJCVG8，正文 DXGC7JEA、补充 MWIWKSVG；正文 Eq. (2)–(3)，补充 S1.2、S5、S12.2.2。附件未提供明确稿件版本/日期，修改时间不能建立最新版本身份。内部 bits 已实际乘 ln(2) 转换为主图 nats。", "",
                  "这次结果检验现有输入上的扩样与方法迁移；新 REST 与旧 REST1_LR 不同，不能把结果称为相同预处理下的严格独立复现。"])
    gate_path=out/"rest_diagnostic/legacy_abc_gate.json"
    if gate_path.exists():
        gate=json.loads(gate_path.read_text())
        report[0]="# HCP 979 人现有输入分析（REST 可比性待确认）"
        report[4:4]=[
            "**复现状态：旧输入的57人八状态A–C核验通过；新REST未通过输入一致性核验。979人结果保留为探索性输出，待REST输入口径对齐后再接受为正式扩样复现。**", "",
            f"[新旧REST诊断与完整一致性核验]({(out/'rest_diagnostic/report.md').resolve()})", ""]
    (out/"report.md").write_text("\n".join(report)+"\n")
