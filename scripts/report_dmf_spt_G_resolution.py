#!/usr/bin/env python3
"""Write the completed fine-G SPT findings and insert a compact brain.md view."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_dmf_spt_G_resolution import OUTPUT


def red(text):
    return '<span style="color: red;">'+text+'</span>'


def label(name):
    return name.replace("7Networks_LH_", "L-").replace("7Networks_RH_", "R-")


def intervals(row):
    pieces = [f"{r['first_G']:.2f}–{r['last_G']:.2f}" for r in row["intervals"]]
    shown = "、".join(pieces[:4])
    return shown+(f"；共 {len(pieces)} 段" if len(pieces)>4 else "") if pieces else "未选中"


def main():
    analysis = json.loads((OUTPUT / "analysis.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    if analysis["status"] != "complete" or summary["status"] != "complete" or len(summary["records"]) != 171:
        raise ValueError("Do not report an unfinished scan as complete")
    s = analysis["stats"]
    labels = analysis["labels"]
    unchanged = s["identical_full_tree_pairs"]
    small = s["identical_small_clade_sets_pairs"]
    count = s["adjacent_pair_count"]
    finding = (f"**0.01 的小步长下，信息量的变化与树结构的变化并不等同。** "
               f"{count} 次相邻比较中，忽略 ROI 身份的纯树形相同 {s['identical_unlabelled_shape_pairs']} 次，"
               f"保留 ROI 身份的完整树完全相同 {unchanged} 次，"
               f"2–20 ROI 团簇集合完全相同 {small} 次。"
               f"每步新选中的非根内部联盟中位数为 {s['new_full_clades_median']:.0f} / 98，"
               f"范围 {s['new_full_clades_range'][0]}–{s['new_full_clades_range'][1]}。"
               "大主干退出次序改变会连带改变许多嵌套联盟；这个计数不表示同样数量的独立团簇诞生。")
    selected = sorted(analysis["tracked"], key=lambda r: r["strength_rank"])[:4]
    names = "、".join(f"{r['id']}（{r['size']} ROI）" for r in selected)
    missing = [str(r["size"]) for r in analysis["original_reference"] if not r["selected_points"]]
    original_statement = ("按精确 ROI 集合追踪原图 n=6/9/15/24/40/59/85 的主干联盟。"
                          +(f"其中 n={('/'.join(missing))} 在这 171 棵配对树中没有被原样选中。" if missing else "它们的保留格点数见表。")
                          +"原图条件独立抽样，本扫描使用共同输入／噪声；两组样本不同，因此这项不复现不能全部归因于 G。")
    competition = (f"对 {count-unchanged} 次拓扑变化，在第一处发生切分差异的共同父联盟上，"
                   "对两侧获选切分作交叉评分："
                   f"{s['score_crossing_first_changes']} 次的两侧排序与候选分数交叉一致，"
                   f"{s['candidate_set_first_changes']} 次至少一侧存在未被当时谱候选纳入、却比获选切分更好的另一侧切分。"
                   "后一类变化体现候选集合的限制，不能直接解释为动力学组织转变；本轮未扩大搜索预算或重建替代树。")
    peak = (f"整体 Ξ 在当前单 seed 细网格的峰为 G={s['peak_overall_xi_G']:.2f}、"
            f"{s['peak_overall_xi_nats']:.4f} nats。它是当前近似估计下的细网格峰，不据此定义严格临界点。")
    interpretation = ("**团簇的“出现”在这里表示它第一次被独立搜索的树选中。** "
                      "同一个联盟可以消失、重现，或一直具有信息而未被选中。"
                      "树拓扑取离散值；若固定候选分数连续且最优切分有正间隔，切分在局部保持不变，"
                      "分数交叉或候选集合变化时则可跳变。"
                      f"图中的 {names} 冻结成员与参考二分后，在全部 G 上都有独立计算的原生 Syn 曲线，"
                      "未获选时没有填零。图高是 ROI 数的对数，不是出现时刻或 Syn 强度。")
    setup = ("固定 93 人平均 Schaefer100 SC、G=1 校准并冻结的 JFIC、seed 4；"
             "G=0–1.70、步长 0.01，共 171 条件。每个条件复用同一批 2,048 个因子化 "
             "U(0.30,0.70) E/I 干预及同一逐步噪声；完整 200 维 source/target，"
             "300 ms 时距、dt=1 ms、sigma=0.01、ridge=10⁻⁶、无状态裁剪。"
             "ROI 叶块保留 E/I 内部 Ξ；≤8 ROI 精确枚举，大联盟沿用原图谱候选，"
             "没有增加随机候选、功能网络约束、相邻树惩罚或平滑。"
             "ROI 名称沿用既有结构连接数据的推定行序（inferred），Yeo 标签仅作事后标识。")
    estimator = ("估计器沿用旧配对缓存的条件 Gaussian 矩模型：用均匀干预样本拟合线性转移，"
                 "将 source 协方差置为对角矩阵，残差与目标采用 Gaussian／ridge 近似。"
                 "没有新拟合非线性 TM；沿用高维例外是为保持 200 维 source/target 和原图口径，"
                 "不将这个近似称为精确均匀干预 EI。补充 S1.2 的低维特征提升与有限样本修正未实现。"
                 "G=0 的非零跨 ROI 量保留为有限样本及拟合背景，未扣除或裁剪。")
    verification = (f"G=1.3 重跑模拟与旧配对条件协方差最大差异为 0，199 个节点成员相同、Syn 最大差异为 0。"
                    f"全部树最大闭合误差为 {s['maximum_closure_error_bits']:.3g} bits。"
                    "非负容差为 10⁻⁸ bits（换算为 nats 时乘 ln 2）；"
                    f"选中树、搜索候选及固定联盟／交叉评分的容差内负值计数分别为 "
                    f"{s['selected_tree_tolerance_negative_count']}、{s['search_tolerance_negative_count']}、"
                    f"{s['fixed_evaluation_tolerance_negative_count']}，显著违规为 0。"
                    "保留原始信息值，没有静默非负投影。")
    source = ("本任务重新经 Zotero 核实父条目 P6UJCVG8，题名 "
              "*Emergent hierarchical organization of causal interactions in complex systems*；"
              "可访问正文 DXGC7JEA（19 页）与补充 MWIWKSVG（28 页）。"
              "阅读正文 Brain/Fig.2（第 6–7 页）、Methods 式(5)–(12)（第 15–17 页），"
              "补充 S1.2–S1.3（第 3–5 页）、S3.3/S5/Algorithm S1（第 8–9 页）、"
              "S12.2.1 式 S82–S87（第 23 页）。两附件无明确稿件日期或修订号，"
              "附件元数据修改时间不能建立新稿版本。正文 Fig.2b 为 Yeo 首层约束树，"
              "本轮沿用仓库无先验 ROI 树；本轮加密 G、配对输入及高维近似均明确记录。")
    assets = "../../fig/dmf_schaefer100/spt_G_resolution_seed04/"
    report = ["# 平均 SC：SPT 团簇随 G 的细网格变化", "", "日期：2026-10-08。状态：171 条件扫描、分析与制图完成。", "",
              finding, "", peak, "", "## 图与观察", "",
              f"![G 细扫描的信息、树变化和团簇保留]({assets}spt_G_resolution_overview.png)", "",
              "a：整体、跨 ROI 与 ROI 内信息预算；b：每个格点相比前一格新选中的联盟数；"
              "c：持久小联盟的精确成员保留；d：用户原图的具名主干联盟；e：冻结成员和二分的 Syn。"
              "c/d 每列是一个实际格点，色块宽度只表示采样网格，不保证格点间始终存在。", "",
              interpretation, "", competition, "", original_statement, "",
              f"[可拖动 G 的交互图]({assets}spt_G_resolution_explorer.html) · "
              f"[完整 171 帧动画]({assets}spt_G_resolution.gif) · "
              f"[六棵代表树]({assets}spt_G_resolution_representative_trees.png)", "",
              "## 可追踪的小团簇", "",
              "探索性选择 2–20 ROI、至少连续 5 个采样点（首末点相差至少 0.04 G）的联盟，"
              "再按其峰值局部 Syn 取前 12 个；这个筛选不检验显著性或跨 seed 重复性。"
              "ID 按第一次选中的 G 排序。表中区间只记录实际格点，更多分段与完整数值在交互图中。", "",
              "| ID | ROI 数 | 被选中格点 / 171 | 最长连续格点 | 保留区间 | 峰值局部 Syn / nats | 完整成员 |",
              "|---|---:|---:|---:|---|---:|---|"]
    for row in analysis["tracked"]:
        members = "；".join(label(labels[j]) for j in row["members"])
        report.append(f"| {row['id']} | {row['size']} | {row['selected_points']} | {row['longest_run_points']} | "
                      f"{intervals(row)} | {row['maximum_selected_syn_nats']:.4f} | {members} |")
    report += ["", "## 原图联盟的精确成员追踪", "", original_statement, "",
               "| 原图联盟 | 被选中格点 / 171 | 区间 |", "|---|---:|---|"]
    for row in analysis["original_reference"]:
        report.append(f"| n={row['size']} | {row['selected_points']} | {intervals(row)} |")
    report += ["", "## 协议、核对与解释边界", "", setup, "", estimator, "", verification, "",
               "本轮是一个固定模拟 seed 的探索性轨迹。树中的强度是路径依赖的层级二分残差，"
               "不是纯阶 PID 原子；选中区间不等于联盟的生物学形成区间。"
               "0.01 网格不能排除格点间短暂变化，本轮未开展 0.001 加密、多 seed 复核、样本量加倍或扩大搜索。", "",
               source, "", f"程序记录的扫描耗时（不含制图）为 {summary['elapsed_seconds']/60:.1f} 分钟。逐条件缓存可复用。"
               "[执行方案与版本记录](../log/dmf_spt_G_resolution_plan.md) · "
               "[扫描入口](../../scripts/run_dmf_spt_G_resolution.py) · "
               "[制图入口](../../scripts/plot_dmf_spt_G_resolution.py)", ""]
    (ROOT / "docs/reports/brain_spt_G_resolution.md").write_text("\n".join(report))
    start = "<!-- report-section:dmf-spt-G-resolution:start -->"
    end = "<!-- report-section:dmf-spt-G-resolution:end -->"
    excerpt = "\n".join([start, '<a id="dmf-spt-G-resolution"></a>', "",
              "#### 2.4.4 "+red("0.01 细扫描：团簇的保留、换枝与强度变化"), "",
              red(finding), "", red(setup), "",
              f"![平均 SC 的 G 细扫描与团簇保留]({assets}spt_G_resolution_overview.png)", "",
              red("图 Q2b 补充 D｜a–e 分别为信息预算、相邻格点联盟变更、持续小团簇、原图主干联盟与冻结二分 Syn。"
                  "绿色表示该精确成员集合被当前树选中，灰色表示未选中；曲线在联盟未选中时仍显示真实 Syn。"
                  "G 轴保留实际 0.01 网格，图例在数据区外。"), "",
              red(interpretation), "", red(competition), "", red(original_statement), "", red(peak), "",
              red("[拖动 G 查看完整树与成员]("+assets+"spt_G_resolution_explorer.html) · "
                  "[完整动画]("+assets+"spt_G_resolution.gif) · [代表树]("+assets+"spt_G_resolution_representative_trees.png) · "
                  "[具名团簇表、协议和核对](brain_spt_G_resolution.md)"), "",
              red(verification+" "+"本轮使用一个 seed，当前结果支持探索性组织追踪；未开展 0.001 加密或跨 seed 确认。"
                  "实际条件 Gaussian 矩估计与手稿版本边界见详细报告。"), "", end, ""])
    path = ROOT / "docs/reports/brain.md"
    text = path.read_text()
    if start in text:
        a, b = text.index(start), text.index(end)+len(end)
        text = text[:a]+excerpt.rstrip()+text[b:]
    else:
        anchor = "### 2.5 ROI Shapley 给出可相加的整体贡献"
        if text.count(anchor) != 1:
            raise ValueError("Cannot uniquely locate the existing brain.md section")
        text = text.replace(anchor, excerpt+"\n"+anchor, 1)
    path.write_text(text)
    print("REPORTS COMPLETE: brain.md section 2.4.4 and brain_spt_G_resolution.md")


if __name__ == "__main__":
    main()
