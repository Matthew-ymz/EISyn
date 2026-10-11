#!/usr/bin/env python3
"""Lightweight, unadjusted Kendall tau-b screen of the frozen 979 HCP cohort.

Reuses grouped Xi and the five existing behavior fields. All 600 hypotheses
share Holm/BH correction. Raw Spearman is computed on exactly the same inputs
to distinguish the correlation statistic from the old covariate adjustment.
No fitted-rank residual is passed to Kendall's ordinary asymptotic test.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hcp_task_behavior_979 as source

OUTPUT = ROOT / "results/hcp_kendall_behavior_979"
SEED = 20261011
BOOTSTRAPS = 2000
LABELS = {
    "story_accuracy": ("Story accuracy", "Story accuracy (%)"),
    "math_accuracy": ("Math accuracy", "Math accuracy (%)"),
    "face_speed": ("Face response speed", r"Face response speed (s$^{-1}$)"),
    "motor_capacity": ("Motor capacity", "Motor capacity (frozen composite)"),
    "wm_accuracy": ("WM accuracy", "WM accuracy (%)"),
}


def fmt(p):
    return f"{p:.2e}" if p < .001 else f"{p:.3f}"


def pair_count_check(x, y):
    """Independent O(n^2) check, including one-sided and double ties."""
    i, j = np.triu_indices(len(y), 1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    concordant = int(np.sum(sx * sy > 0))
    discordant = int(np.sum(sx * sy < 0))
    tx, ty = int(np.sum(sx == 0)), int(np.sum(sy == 0))
    denominator = np.sqrt(float(len(i) - tx) * (len(i) - ty))
    if denominator == 0:
        raise ValueError("Undefined Kendall tau: a variable is constant")
    return dict(concordant_pairs=concordant, discordant_pairs=discordant,
                x_tied_pairs=tx, y_tied_pairs=ty,
                pairwise_tau=(concordant - discordant) / denominator)


def bootstrap(x, y, seed):
    """Pointwise percentile interval; subjects assumed independent."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(y), len(y))
        values.append(kendalltau(x[indices], y[indices], variant="b",
                                method="asymptotic").statistic)
    if not np.isfinite(values).all():
        raise ValueError("Undefined bootstrap coefficient")
    return list(map(float, np.quantile(values, [.025, .975])))


def make_scatter(rows, analyses, filename):
    with plt.rc_context(source.style()):
        fig, axes = plt.subplots(1, len(rows), figsize=(4.05 * len(rows), 4.05),
                                 layout="constrained", squeeze=False)
        for ax, row, letter in zip(axes.flat, rows, "abcde"):
            a = analyses[row["endpoint"]]
            x = a["x"][:, source.NAMES.index(row["coalition"])] * np.log(2)
            y = a["y"]
            color = source.COLORS[row["state"]]
            ax.scatter(x, y, s=12, alpha=.23, color=color, linewidths=0,
                       rasterized=True)
            ax.set_xlabel(r"Coalition $\Xi$ (nats)")
            ax.set_ylabel(LABELS[row["endpoint"]][1])
            ax.set_title(f"{LABELS[row['endpoint']][0]} · {source.compact(row['coalition'])}\n"
                         rf"$\tau_b$={row['tau_b']:+.3f}   Holm-600 P={fmt(row['p_holm_600'])}" + "\n"
                         f"raw association · n={row['n']}", loc="left", fontsize=9.5, pad=10)
            ax.text(-.16, 1.23, letter, transform=ax.transAxes,
                    fontsize=13, fontweight="bold")
            ax.grid(color="#EEEEEE", linewidth=.5)
            ax.set_axisbelow(True)
            ax.margins(x=.04, y=.07)
        fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def report(summary):
    winners, rows = summary["representatives"], summary["associations"]
    significant = sorted([r for r in rows if r["p_holm_600"] < .05],
                         key=lambda r: (r["endpoint"], r["p_holm_600"]))
    lines = ["# HCP 979 人：原始 Kendall τ-b 与行为关联", "",
             "2026-10-11。图片对应 Schaefer-1000/Yeo-7 的 979 人四任务数据；"
             "本次复用同一队列、同一 EI 缓存和五个既有行为字段，MOTOR 完整案例为 976 人。", "",
             f"**原始 Kendall τ-b 有 {len(significant)} 项通过 Holm-600："
             f"Story {summary['counts']['story_accuracy']['holm']}、Math {summary['counts']['math_accuracy']['holm']}、"
             f"Face speed {summary['counts']['face_speed']['holm']}；MOTOR 与 WM 均为 0。** "
             "这些是被试独立假设下的未调整关联，重叠联盟不代表独立机制。", "",
             "![三个 Holm-600 显著端点的代表散点](kendall_holm_representative_scatter.png)", "",
             "每个显著端点展示 |τ-b| 最大的组合。每点是一名被试，保留原始分数与并列值，"
             "不加抖动、不删满分值或离群点。图展示实际观测；Kendall 统计通过对象对顺序计算。", "",
             "| 端点 | 代表组合 | N | τ-b | 逐项 bootstrap 95% CI | Holm-600 P | BH-600 q | Holm 通过数 | BH 通过数 |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in winners:
        lo, hi = r["bootstrap_95_ci"]
        counts = summary["counts"][r["endpoint"]]
        lines.append(f"| {LABELS[r['endpoint']][0]} | {source.compact(r['coalition'])} | {r['n']} | "
                     f"{r['tau_b']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {fmt(r['p_holm_600'])} | "
                     f"{fmt(r['q_bh_600'])} | {counts['holm']} | {counts['bh']} |")
    lines += ["", "![MOTOR 与 WM 最强候选，均未通过 Holm-600](kendall_motor_wm_exploratory_scatter.png)", "",
              "MOTOR/WM 图只展示各端点 |τ-b| 最大的探索候选，均未通过 Holm-600；"
              "BH 通过仅表示 FDR 筛选层面的候选。所有区间为同一全样本选出代表后得到的逐项区间，"
              "采用 2,000 次被试 bootstrap，未作选择偏差或多重区间校正。", "",
              "## 与 Spearman 的公平比较", "",
              "同一原始数据、同一完整案例、同一 600 项家族下重新计算 Spearman，"
              "不能用当前未调整 Kendall 与旧调整后 Spearman 的显著项数量判断指标优劣。", "",
              "| 端点 | Kendall Holm / BH 通过数 | 原始 Spearman Holm / BH 通过数 |",
              "|---|---:|---:|"]
    for key, c in summary["counts"].items():
        lines.append(f"| {LABELS[key][0]} | {c['holm']} / {c['bh']} | {c['spearman_holm']} / {c['spearman_bh']} |")
    cmp = summary["metric_comparison"]
    lines += ["", f"两者 Holm 显著集合共有 {cmp['holm_intersection']} 项；"
              f"仅 Kendall 通过 {cmp['kendall_only_holm']} 项，仅 Spearman 通过 {cmp['spearman_only_holm']} 项。"
              "Kendall 的优势是基于对象对顺序、显式处理并列值的解释；当前数据没有显示更强的检出能力。"
              "τ 与 ρ 的数值尺度不同，不能以 τ 绝对值更小推断关联更弱。", "",
              "旧协变量调整 Spearman 的 Math 最强候选同为 DAN+VAN+L+D。"
              "WM 的旧优胜组合 L+C 在原始 Kendall 中仍为负向（τ-b=-0.0795），但未通过 Holm-600；"
              "原始两种指标的 WM 最强候选均为正向 DAN+C。改变的是优胜组合及显著性，"
              "不能把不同组合的正负方向当作同一组合发生方向反转。", "",
              "## 检验对象与统计口径", "",
              "组合量沿用 SI S16 的分组整合信息：", "",
              "$$", r"\Xi(S)=EI(\mathbf{x}_S\to\mathbf{y})-\sum_{i\in S}EI(\mathbf{x}_i\to\mathbf{y}).", "$$", "",
              "源为组合内网络的三阶历史，目标固定为完整七网络下一状态。"
              "它是组合联合 EI 超出单网络 EI 之和的量，不能称为纯 k 阶 PID 原子或 SPT 节点残差。"
              "沿用 NoGSR、每网络 PC1、delta Ridge p=3/α=1、独立标准高斯源干预、affine-Gaussian TM/logdet EI。"
              "内部单位 bits，散点转换为 nats；单调正单位换算不改变秩统计。", "",
              "五端点为 Story/Math 正确率、Face 反应速度、冻结旧权重的扫描外运动综合分、WM 总正确率。"
              "Face 速度取 1000/Face median RT（原 RT 单位 ms），数值越大越快；"
              "其排序与旧 -log(RT) 一致。这里没有调整 Shape 速度，故不使用 face-specific 表述。"
              "运动综合分仍使用旧 57 人固定均值和标准差，不因本次关联改权重。", "",
              "原始 Kendall τ-b 使用 SciPy 的双侧、含并列修正的渐近检验；全部 600 个 P 值统一 Holm 和 BH 校正，阈值 0.05。"
              "原始 Spearman 也使用双侧渐近检验与同样校正。Story 的 614/979 个满分保留；"
              "各端点取值数、并列对象对数、缺失排除 ID 及全部 600 项效应保存在 summary.json。", "",
              "本次是轻量原始相关筛查，没有新增年龄、性别、重建版本、来源或 Shape-speed 校正的 Kendall 置换推断。"
              "对秩回归残差直接使用普通 Kendall P 值不等价于有效的偏 Kendall 检验，故没有这样替代。"
              "旧调整后 Spearman 的效应与校正值逐项保留在 JSON 中作参考，不能借用其 P 值声明 Kendall 显著。", "",
              "HCP 家系、头动和新被试 atlas 独立提取来源的既有缺口继续保留。"
              "Holm 的家族错误率含义依赖单项 P 值有效，BH 另依赖其适用条件；"
              "存在亲属依赖与未调整混杂时，本筛查的显著性只作暂定探索证据，不证明因果或独立复现。", "",
              "## 全部 Holm-600 通过组合", "",
              "| 端点 | 组合 | 阶数 | τ-b | 原始 P | Holm-600 P | BH-600 q |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in significant:
        lines.append(f"| {LABELS[r['endpoint']][0]} | {source.compact(r['coalition'])} | {r['source_network_count']} | "
                     f"{r['tau_b']:+.4f} | {fmt(r['p_raw'])} | {fmt(r['p_holm_600'])} | {fmt(r['q_bh_600'])} |")
    audit = summary["numerical_audit"]
    lines += ["", "V=Visual，SM=Somatomotor，DAN=Dorsal Attention，VAN=Salience/Ventral Attention，"
              "L=Limbic，C=Control，D=Default；阶数是功能网络数。", "",
              "## 核验与稿件依据", "",
              "本次实际重算 manifest、records.jsonl、行为表与行为定义依赖文件的 SHA256，均与已核验的 2026-10-09 分析一致。"
              "未再次读取 24.59 GB 的上游原始时序，因此此身份检查不能替代独立提取来源证明。", "",
              f"组合 Ξ 非负容差为 1e-4 bits（{1e-4*np.log(2):.8g} nats）；"
              f"实际检查 {sum(audit[s]['checked_count'] for s in source.STATES):,} 个值，"
              f"最小为 {min(audit[s]['minimum_bits'] for s in source.STATES):.10g} bits，"
              "容差内负值和显著非负性违反均为 0，保留原值。"
              "五个代表组合另以直接对象对计数复核 τ-b，包含单侧及双侧并列；"
              "Holm/BH 的极小 P 值与并列边界用独立标量公式复核，均通过。", "",
              "2026-10-11 经 Zotero 本地 API 重新核验父条目 P6UJCVG8 题名 "
              "*Emergent hierarchical organization of causal interactions in complex systems*，"
              "重新查询附件并取得正文 DXGC7JEA（19/19 页）与补充 MWIWKSVG（28/28 页）的全文。"
              "当前仅各一份附件，没有明确稿件版本号或稿内修订日期；2026-10-04 附件修改时间不能确立版本。"
              "读取正文 Fig.3/行为结果（pp.8–9）、Methods 式（5）–（12）（pp.15–17）；"
              "SI S1.2–S1.3（pp.3–4，S7–S18）、S5.1（p.9，S49–S50）、S10（pp.15–16）、"
              "S12.2.2（pp.23–25）。本次只改变关联读出，沿用 979 人扩样、四任务和五行为字段；"
              "稿件原 57 人七任务、协变量调整 Spearman 及 SOCIAL max-T 的口径不被当前原始 Kendall 替代。", "",
              "复现脚本：[screen_hcp_kendall_behavior_979.py](../../scripts/screen_hcp_kendall_behavior_979.py)。"
              "完整数值：[summary.json](summary.json)。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def correction_check():
    p = np.array([1e-9, .1, 1e-9, .0001, .02, .0009, 1., 0., .05])
    order = np.argsort(p)
    expected_h, expected_b = np.empty_like(p), np.empty_like(p)
    for rank, index in enumerate(order):
        expected_h[index] = min(1., max((len(p)-j)*p[order[j]] for j in range(rank+1)))
        expected_b[index] = min(1., min(len(p)*p[order[j]]/(j+1) for j in range(rank,len(p))))
    np.testing.assert_allclose(source.holm(p), expected_h)
    np.testing.assert_allclose(source.bh(p), expected_b)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source.OUTPUT / "summary.json").read_text())
    config = previous["configuration"]
    fingerprints = {}
    for path, key in [(source.SOURCE/"manifest.json", "source_manifest_hash"),
                      (source.SOURCE/"records.jsonl", "source_records_hash"),
                      (source.BEHAVIOR, "behavior_sha256")]:
        actual = source.sha(path)
        if actual != config[key]:
            raise ValueError(f"Frozen data changed: {path}")
        fingerprints[str(path.relative_to(ROOT))] = actual
    for relative, expected in config["reference_file_sha256"].items():
        actual = source.sha(ROOT/relative)
        if actual != expected:
            raise ValueError(f"Frozen behavioral reference changed: {relative}")
        fingerprints[relative] = actual
    fingerprints["scripts/run_hcp_task_behavior_979.py"] = source.sha(Path(source.__file__))
    if fingerprints["scripts/run_hcp_task_behavior_979.py"] != config["script_sha256"]:
        raise ValueError("Frozen feature/behavior implementation changed")
    manifest = json.loads((source.SOURCE / "manifest.json").read_text())
    ids, features, audit = source.load_features(manifest)
    if len(ids) != 979 or len(set(ids)) != 979:
        raise ValueError("Expected 979 unique paired imaging subjects")
    frame = source.raw_behavior(ids)
    reference = {(r["endpoint"], r["coalition"]): r for r in previous["exploration"]}
    analyses, rows, endpoint_audits = {}, [], {}
    with threadpool_limits(limits=2):
        for definition in source.ENDPOINTS:
            a = source.prepare_analysis(definition, frame, features)
            original_key = a["key"]
            key = "face_speed" if original_key == "face_specific_speed" else original_key
            if key == "face_speed":
                a["y"] = 1000 / a["frame"]["Emotion_Task_Face_Median_RT"].to_numpy(float)
            analyses[key] = a
            y = a["y"]
            unique, counts = np.unique(y, return_counts=True)
            endpoint_audits[key] = dict(n=len(y), distinct_scores=len(unique),
                                       tied_behavior_pairs=int(np.sum(counts*(counts-1)//2)),
                                       excluded_subjects=list(frame.index[~a["mask"]]),
                                       maximum_score_count=int(counts[-1]), maximum_score=float(unique[-1]))
            for j, name in enumerate(source.NAMES):
                kt = kendalltau(a["x"][:, j], y, variant="b", method="asymptotic")
                sp = spearmanr(a["x"][:, j], y)
                if not np.isfinite([kt.statistic, kt.pvalue, sp.statistic, sp.pvalue]).all():
                    raise ValueError(f"Undefined association {key} {name}")
                old = reference[original_key, name]
                rows.append(dict(endpoint=key, state=a["state"], coalition=name,
                                 source_network_count=len(source.COMBOS[j]), n=len(y),
                                 tau_b=float(kt.statistic), p_raw=float(kt.pvalue),
                                 raw_spearman_rho=float(sp.statistic), raw_spearman_p=float(sp.pvalue),
                                 old_adjusted_spearman_rho=old["rho"],
                                 old_adjusted_spearman_p_holm_600=old["p_holm_600"],
                                 old_adjusted_spearman_q_bh_600=old["q_bh_600"]))
        correction_check()
        if len(rows) != 600:
            raise ValueError("Expected one 600-hypothesis family")
        for input_key, method, output_key in [
                ("p_raw", source.holm, "p_holm_600"), ("p_raw", source.bh, "q_bh_600"),
                ("raw_spearman_p", source.holm, "raw_spearman_p_holm_600"),
                ("raw_spearman_p", source.bh, "raw_spearman_q_bh_600")]:
            for r, v in zip(rows, method([r[input_key] for r in rows])):
                r[output_key] = float(v)
        counts, representatives = {}, []
        for i, key in enumerate(analyses):
            local = [r for r in rows if r["endpoint"] == key]
            counts[key] = dict(holm=sum(r["p_holm_600"] < .05 for r in local),
                               bh=sum(r["q_bh_600"] < .05 for r in local),
                               spearman_holm=sum(r["raw_spearman_p_holm_600"] < .05 for r in local),
                               spearman_bh=sum(r["raw_spearman_q_bh_600"] < .05 for r in local))
            top = dict(max(local, key=lambda r: abs(r["tau_b"])))
            a = analyses[key]
            x = a["x"][:, source.NAMES.index(top["coalition"])]
            check = pair_count_check(x, a["y"])
            if abs(check["pairwise_tau"] - top["tau_b"]) > 1e-12:
                raise ValueError(f"Independent pair-count mismatch {key}")
            top["pair_count_check"] = check
            top["bootstrap_95_ci"] = bootstrap(x, a["y"], SEED+i)
            representatives.append(top)
        kt = {(r["endpoint"], r["coalition"]) for r in rows if r["p_holm_600"] < .05}
        sp = {(r["endpoint"], r["coalition"]) for r in rows if r["raw_spearman_p_holm_600"] < .05}
        summary = dict(date="2026-10-11", statistic="unadjusted Kendall tau-b",
                       p_method="two-sided, tie-corrected asymptotic; independent subjects assumed",
                       family_size=600, threshold=.05, seed=SEED, bootstrap_repeats=BOOTSTRAPS,
                       input_hashes=fingerprints, script_sha256=source.sha(Path(__file__)),
                       endpoint_audit=endpoint_audits, numerical_audit=audit,
                       metric_comparison=dict(holm_intersection=len(kt & sp),
                                              kendall_only_holm=len(kt - sp), spearman_only_holm=len(sp - kt)),
                       counts=counts, representatives=representatives, associations=rows,
                       covariate_adjustment=False, family_dependence_adjustment=False,
                       independent_holdout=False,
                       manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG",
                                       checked_date="2026-10-11", explicit_draft_version=None,
                                       locations=["Main pp8-9, 15-17", "SI pp3-4, 9, 15-16, 23-25"]))
        source.dump(OUTPUT / "summary.json", summary)
        make_scatter([r for r in representatives if r["p_holm_600"] < .05], analyses,
                     "kendall_holm_representative_scatter.png")
        make_scatter([r for r in representatives if r["endpoint"] in ["motor_capacity", "wm_accuracy"]],
                     analyses, "kendall_motor_wm_exploratory_scatter.png")
        report(summary)
        print(json.dumps(dict(counts=counts, metric_comparison=summary["metric_comparison"],
                              representatives=representatives), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
