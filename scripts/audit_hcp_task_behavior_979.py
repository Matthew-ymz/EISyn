#!/usr/bin/env python3
"""Prespecified input/phenotype sensitivity checks and views of frozen results.

No new endpoints, hypotheses, p-values, model fits, or selection criteria are added.
The new 57-person effects are a pipeline check, not a separate replication cohort.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_hcp_task_behavior_979 as run


def overlap_effects(ids, features):
    from scripts import screen_hcp_motor_composite_scores_57 as motor
    from scripts import screen_hcp_emotion_performance_coalitions_57 as emotion
    lookup = {sid: i for i, sid in enumerate(ids)}
    gate = json.loads((run.OUTPUT / "legacy_gate.json").read_text())
    old = {(r["endpoint"], r["coalition"]): r for r in gate["rows"]}
    result = []
    with np.load(run.ROOT / "results/hcp_language_story_math_coalitions_57/language_coalition_synergy_57.npz") as archive:
        lang_ids = archive["subjects"].astype(str)
    behavior = run.raw_behavior(ids)
    for key, name, _ in run.FIXED[:2]:
        positions = [lookup[s.removeprefix("sub-")] for s in lang_ids]
        field = next(e[3] for e in run.ENDPOINTS if e[0] == key)
        y = behavior.loc[[s.removeprefix("sub-") for s in lang_ids], field].to_numpy(float)
        rho = float(spearmanr(features["LANGUAGE"][positions, run.NAMES.index(name)], y).statistic)
        result.append(dict(endpoint=key, coalition=name, n=57, old_rho=old[key, name]["rho"], new_input_57_rho=rho))
    with np.load(motor.CACHE) as archive:
        motor_ids = archive["subjects"].astype(str)
    positions = [lookup[s.removeprefix("sub-")] for s in motor_ids]
    scores = motor.load_scores(motor_ids)
    for name in run.MOTOR_NAMES:
        x = features["MOTOR"][positions, run.NAMES.index(name)]
        rho = float(motor.adjusted_rho(x, scores["composite"], scores))
        result.append(dict(endpoint="motor_capacity", coalition=name, n=57,
                           old_rho=old["motor_capacity", name]["rho"], new_input_57_rho=rho))
    with np.load(emotion.CACHE) as archive:
        emotion_ids = archive["subjects"].astype(str)
    positions = [lookup[s.removeprefix("sub-")] for s in emotion_ids]
    b = emotion.load_behavior(emotion_ids)
    shape = -np.log(b["Emotion_Task_Shape_Median_RT"])
    face = -np.log(b["Emotion_Task_Face_Median_RT"])
    design = np.column_stack([np.ones(57), rankdata(b["age"]), b["sex"], b["cohort"], rankdata(shape)])
    name = "Limbic+Cont+Default"
    rho = emotion.partial_rank_correlation(features["EMOTION"][positions, run.NAMES.index(name)], face, design)
    result.append(dict(endpoint="face_specific_speed", coalition=name, n=57,
                       old_rho=old["face_specific_speed", name]["rho"], new_input_57_rho=float(rho)))
    mapping = {(r["endpoint"], r["coalition"]): r for r in result}
    ordered = [mapping[k, n] for k, n, _ in run.FIXED]
    for r in ordered:
        r["rho_difference"] = r["new_input_57_rho"]-r["old_rho"]
    return ordered


def new_candidate_plot(analyses, summary):
    anchors = []
    for key in ["math_accuracy", "wm_accuracy"]:
        rows = [r for r in summary["exploration"] if r["endpoint"] == key and r["p_holm_600"] < .05]
        if rows:
            anchors.append(max(rows, key=lambda r:abs(r["rho"])))
    if not anchors:
        return
    with run.plt.rc_context(run.style()):
        fig, axes = run.plt.subplots(1, len(anchors), figsize=(10.0, 4.3), layout="constrained", squeeze=False)
        for ax, r, letter in zip(axes.flat, anchors, "ab"):
            a = next(a for a in analyses if a["key"] == r["endpoint"])
            c = run.NAMES.index(r["coalition"])
            x, y = a["yr"]/len(a["y"]), a["xr"][:, c]/len(a["y"])
            color = run.COLORS[r["state"]]
            ax.scatter(x, y, s=12, alpha=.34, color=color, linewidths=0)
            slope, intercept = np.polyfit(x, y, 1)
            ends = np.array([x.min(), x.max()])
            ax.plot(ends, intercept+slope*ends, color=color, linewidth=1.6)
            ax.set_title(f"{r['state']} · {run.compact(r['coalition'])}\n"
                         f"ρ={r['rho']:+.3f}   Holm-600 p={run.format_p(r['p_holm_600'])}   n={r['n']}", loc="left", fontsize=10, pad=10)
            ax.text(-.16, 1.18, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.set_xlabel(a["title"]+"\n(adjusted rank residual / N)")
            ax.set_ylabel("Coalition Ξ\n(adjusted rank residual / N)")
            ax.axvline(0, color="#DDDDDD", linewidth=.5, zorder=0)
            ax.axhline(0, color="#DDDDDD", linewidth=.5, zorder=0)
            ax.grid(color="#EEEEEE", linewidth=.6)
            ax.set_axisbelow(True)
        fig.savefig(run.OUTPUT / "hcp979_new_candidate_associations.png", dpi=300, bbox_inches="tight")
        run.plt.close(fig)


def main():
    with threadpool_limits(limits=2):
        summary = json.loads((run.OUTPUT / "summary.json").read_text())
        if run.sha(run.SOURCE / "manifest.json") != summary["configuration"]["source_manifest_hash"] or \
           run.sha(run.SOURCE / "records.jsonl") != summary["configuration"]["source_records_hash"] or \
           run.sha(run.BEHAVIOR) != summary["configuration"]["behavior_sha256"]:
            raise ValueError("Inference inputs changed before supplementary audit")
        for relative_path, expected in summary["configuration"]["reference_file_sha256"].items():
            if run.sha(ROOT / relative_path) != expected:
                raise ValueError(f"Frozen behavior reference changed: {relative_path}")
        manifest = json.loads((run.SOURCE / "manifest.json").read_text())
        ids, features, _ = run.load_features(manifest)
        frame = run.raw_behavior(ids)
        analyses = [run.prepare_analysis(e, frame, features) for e in run.ENDPOINTS]
        from statsmodels.stats.multitest import multipletests
        challenging_p = np.array([1e-5, .1, 1e-5, .0001, .02, .0009, 1., 0., .05])
        assert np.allclose(run.holm(challenging_p), multipletests(challenging_p, method="holm")[1])
        assert np.allclose(run.bh(challenging_p), multipletests(challenging_p, method="fdr_bh")[1])
        maximum_effect_error = 0.
        for a in analyses:
            stored = np.array([r["rho"] for r in summary["exploration"] if r["endpoint"] == a["key"]])
            maximum_effect_error = max(maximum_effect_error, float(np.abs(stored-a["observed"]).max()))
        assert maximum_effect_error < 1e-12, maximum_effect_error
        overlap = overlap_effects(ids, features)
        motor = next(a for a in analyses if a["key"] == "motor_capacity")
        raw = motor["frame"][["Endurance_AgeAdj", "Dexterity_AgeAdj", "Strength_AgeAdj"]].to_numpy()
        recalculated = ((raw-raw.mean(axis=0))/raw.std(axis=0, ddof=1)).mean(axis=1)
        yr = run.normalize(run.residual(rankdata(recalculated), motor["q"]))
        alternative_rho = motor["brain"].T @ yr
        sensitivity = dict(kind="Descriptive prespecified phenotype and overlapping-input checks; no new p-value families",
                           new_57_original_model_effects=overlap,
                           maximum_absolute_new_57_rho_difference=max(abs(r["rho_difference"]) for r in overlap),
                           maximum_recomputed_600_effect_error=maximum_effect_error,
                           low_p_ties_and_boundaries_multiple_comparisons_check="passed vs statsmodels Holm and BH",
                           motor_standardization=dict(n=976, score_rank_correlation=float(spearmanr(recalculated, motor["y"]).statistic),
                               reference="New complete-case 976-person sample SD; primary still uses old frozen 57-person parameters",
                               fixed_rhos=[dict(coalition=n, primary_rho=float(motor["observed"][run.NAMES.index(n)]),
                                                alternative_rho=float(alternative_rho[run.NAMES.index(n)])) for n in run.MOTOR_NAMES]),
                           supplementary_script_sha256=run.sha(Path(__file__)))
        summary["sensitivity_checks"] = sensitivity
        summary["exploration_summary"] = dict(
            holm_600_by_endpoint={e[0]:sum(r["p_holm_600"] < .05 and r["endpoint"] == e[0] for r in summary["exploration"]) for e in run.ENDPOINTS},
            bh_600_by_endpoint={e[0]:sum(r["q_bh_600"] < .05 and r["endpoint"] == e[0] for r in summary["exploration"]) for e in run.ENDPOINTS},
            holm_600_at_least_three_networks=sum(r["p_holm_600"] < .05 and r["source_network_count"] >= 3 for r in summary["exploration"]),
            permutation_resolution=1/100001, independent_new_mechanisms_count=None)
        selected = [r for r in summary["exploration"] if r["p_holm_600"] < .05]
        selected_math = [r for r in selected if r["endpoint"] == "math_accuracy"]
        membership = {n: sum(n in r["coalition"].split("+") for r in selected_math) for n in run.NETWORKS}
        run.dump(run.OUTPUT / "summary.json", summary)
        new_candidate_plot(analyses, summary)
        report = run.OUTPUT / "report.md"
        text = report.read_text().split("\n## 执行后补充核验与候选展示")[0]
        lines = ["", "## 执行后补充核验与候选展示", "",
                 "以下图只展示全 600 项 Holm 校正通过的新候选中，Math 与 WM 各自效应最大的组合。组合和效应在同一全样本内筛选、估计，不能当作独立验证效应。", "",
                 "![校正后新候选](hcp979_new_candidate_associations.png)", "",
                 f"{len(selected)} 个 Holm-600 候选中，{len(selected_math)} 个对应 Math accuracy、"
                 f"{sum(r['endpoint'] == 'wm_accuracy' for r in selected)} 个对应 WM accuracy；其中 "
                 f"{sum(r['source_network_count'] >= 3 for r in selected)} 个含至少三个源网络。"
                 f"它们是相互重叠的组合，不是 {len(selected)} 个独立神经机制。Math 候选中，"
                 f"{membership['SalVentAttn']} 个包含 SalVentAttn，{membership['Default']} 个包含 Default，"
                 f"{membership['Limbic']} 个包含 Limbic。这是描述性共同构成，不是这些网络独立必要性的统计证明。", "",
                 "同一批 57 人使用新 TASK 输入、旧行为终点和原统计调整后，12 项效应与旧值接近：", "",
                 "| 终点 | 组合 | 旧输入 ρ | 新输入同 57 人 ρ | 差值 |", "| --- | --- | --- | --- | --- |"]
        for r in overlap:
            lines.append(f"| {r['endpoint']} | {run.compact(r['coalition'])} | {r['old_rho']:+.4f} | {r['new_input_57_rho']:+.4f} | {r['rho_difference']:+.4f} |")
        lines += ["", f"最大绝对系数差为 {sensitivity['maximum_absolute_new_57_rho_difference']:.6f}。这支持重叠 57 人的任务处理与指标在行为关联层面也基本对齐；扩样旧效应减弱不能仅凭这一现象归结为 TASK 原始数据错误。小样本筛选、补充样本的选择和效应不稳定可能影响旧值；本轮不能区分这些原因的相对贡献，也不能替代新被试来源核实。", "",
                  f"MOTOR 在 976 个完整案例内重标准化后，综合分与冻结旧权重的分数秩相关为 {sensitivity['motor_standardization']['score_rank_correlation']:.6f}；固定 9 个组合的替代效应记录在 JSON 中。此检查是描述性敏感性结果，没有替换主权重或新增可择优报告的显著性检验。", "",
                  "100000 次置换的最小 P 值分辨率约 1e-5，几项强候选达到分辨率下限；接近 0.05 的校正结果存在 Monte Carlo 精度限制。报告保留冻结次数下的结果，后续若进行独立验证，可同时增加置换精度；本轮没有事后追加置换或搜索。"]
        content = text+"\n".join(lines)+"\n"
        for name in ["hcp979_fixed_replication_forest.png", "hcp979_fixed_association_scatter.png",
                     "hcp979_all_600_associations.png", "hcp979_new_candidate_associations.png", "summary.json"]:
            content = content.replace(f"]({name})", f"]({run.OUTPUT / name})")
        report.write_text(content)
        print(json.dumps(dict(status="passed", maximum_absolute_new_57_rho_difference=sensitivity["maximum_absolute_new_57_rho_difference"],
                              motor_score_rank_correlation=sensitivity["motor_standardization"]["score_rank_correlation"])))


if __name__ == "__main__":
    main()
