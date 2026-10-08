#!/usr/bin/env python3
"""Concise interpretation and complete top-20 bidirectional rank tables."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_sc_topology_comparison import METRIC_NAMES, OUTPUT

NAMES = {'internal_mean':'内部平均边权', 'clique_intensity':'闭合子图强度',
         'modularity_contribution':'模块度单组贡献'}
REPORT = ROOT/'docs/reports/brain_sc_topology_comparison.md'
RANKINGS = ROOT/'docs/reports/brain_sc_topology_rankings.md'


def members(row):
    return '、'.join(map(str,row['members_one_based']))


def rank(row, metric):
    r = row['sc'][metric]['cohort_midrank']
    return f'{r:g}'


def main():
    summary = json.loads((OUTPUT/'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Incomplete comparison')
    verified = json.loads((OUTPUT/'verification.json').read_text())
    if verified['status'] != 'passed':
        raise ValueError('Native-data verification failed')
    for name, expected in verified['verified_cache_sha256'].items():
        if hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Native-data verification is stale: {name}')
    orders = summary['orders']
    candidates = [r for k in ('3','4') for r in orders[k]['candidates']]
    lines = ['# 个体 SC 拓扑排名与转折窗口自然 SPT 的比较', '',
        '2026-10-08。93 人、全部三元组和四元组的比较已完成，共评估 **379,712,025 个个体–组合**。保持原生个体 SC、固定三项结构指标及原 SPT 搜索口径；未启动新 DMF 模拟或重新构树。', '',
        '结果同时支持两点：**C1–C3 具有突出的结构连接背景；结构排名与转折窗口协同复现排名存在明显差异。** 因此已有候选可以从结构背景理解，但不能将它们称为拓扑上普通、仅通过协同才发现的组合。动力学分析提供的是这些组合在什么工作点更常形成联合信息节点，以及哪些结构领先组合并未同样高频形成节点。', '',
        '## C1–C3 的同阶排名', '',
        '队列 SC 排名先在每人全部同阶组合中计算并列中间名次，再跨 93 人平均归一化名次；不是对平均 SC 排名。每人三元组 161,700 个、四元组 3,921,225 个。下表均为同阶排名，SPT 使用原冻结严格规则。', '',
        '| 集合 | ROI | SPT 名次 | SC：平均边权 | SC：闭合子图强度 | SC：模块度贡献 |',
        '|---|---|---:|---:|---:|---:|']
    for r in candidates:
        lines.append(f"| {r['display_id']} | {members(r)} | {r['spt_same_order_display_rank']} | "+
                     ' | '.join(rank(r,m) for m in METRIC_NAMES)+' |')
    lines += ['', '| 集合 | 指标 | 个体名次中位数［Q25，Q75］/ top % | 个体前 1% 人数 |',
              '|---|---|---:|---:|']
    for r in candidates:
        for m in METRIC_NAMES:
            q25,med,q75 = r['sc'][m]['individual_rank_percent_q25_median_q75']
            lines.append(f"| {r['display_id']} | {NAMES[m]} | {med:.4f} ［{q25:.4f}，{q75:.4f}］ | {r['sc'][m]['individual_top1_count']}/93 |")
    lines += ['', 'top % 越小越靠前；前 1% 是固定的描述性阈值，不是统计显著性水平。名次分布的中位数/IQR 与用于队列排行榜的平均名次是两个不同读出。', '',
        'C1 的三种结构指标均很突出。C2 在多数人的三角形强度上同样靠前，但完整图边权的几何平均对零边敏感：`sub-10280` 的 59–61 与 60–61 为零，其三角形强度在个人同阶排名约 top 53.05%；另一个被试约 top 2.04%。因此其 91/93 人仍在前 1%，而平均名次汇总的队列位置降到第 133。保留冻结平均排名规则，同时报告这一稀疏性影响。', '',
        'C3 的闭合强度也受到少数缺边个体影响：`sub-10638` 缺 62–64，`sub-10280` 缺 62–63、62–64、62–68，两人的完整四节点子图强度均为零，个人名次分别约 top 50.91% 和 50.73%。其余 91 人全部位于个人前 0.016%，93 人名次中位数为 top 0.0018%，但按冻结平均名次汇总仍为队列第 584。平均边权和模块度指标也排在约 392 万个四元组中的前千名；不能将这些数百名的位置理解成普通或弱连接。', '',
        '## 两类排行榜的异同', '',
        '| 阶数 | 结构指标 | 与 SPT 前 20 名重合 | Jaccard | 结构质量与窗口复现的 Spearman ρ |',
        '|---|---|---:|---:|---:|']
    for k in ('3','4'):
        for m in METRIC_NAMES:
            a=orders[k]['overlap'][m];rho=orders[k]['correlations'][m]['spearman_rho']
            lines.append(f"| {k} | {NAMES[m]} | {a['top20_intersection_count']}/20 | {a['jaccard']:.3f} | {rho:.3f} |")
    lines += ['', '相关仅在曾成为自然节点的 561 个三元组、779 个四元组内计算，结构质量定义为负的平均归一化名次；不代表全部可能集合的相关，也不报告选择后的未经校正 p 值。完整 SC 搜索保留未入树集合，以下双向表补充相关读出没有覆盖的结构候选。', '',
        '四元组 SPT 前 20 截断处的强复现数为 1/93，有 16 组在这一主指标并列，其中 13 组在全部排序条件上并列；前 20 重合数受冻结成员字典序展示规则影响，不能当作精确可区分的优劣。三元组截断处主指标为 3/93，主指标并列 4 组，全部排序条件并列 1 组。', '',
        '三元组 **29、37、50** 在三项 SC 指标中均为第 1，但背景以上窗口 ≥4/6 的复现只有 **2/93**，窗口任一格为 **11/93**；C1 的对应值为 **55/93、75/93**。这说明强结构组合不会自动成为转折邻域高频协同节点。', '',
        '四元组 **77、87、89、100** 同样在三项 SC 指标中均为第 1，强窗口 ≥4/6 仅 **1/93**，≥1/6 为 **3/93**；C3 的对应值为 **31/93、66/93**。', '',
        '### 纯结构排名前 3 的动力学表现', '',
        '| 阶数 | SC 指标 | SC 名次 | 完整 ROI | SPT 窗口 ≥4/6 | SPT 窗口 ≥1/6 | 曾入自然树 |',
        '|---|---|---:|---|---:|---:|---|']
    for k in ('3','4'):
        for m in METRIC_NAMES:
            for r in orders[k]['sc_top20'][m][:3]:
                d=r['above_background']
                lines.append(f"| {k} | {NAMES[m]} | {rank(r,m)} | {members(r)} | {d['critical_ge4']}/93 | {d['critical_ge1']}/93 | {'是' if r['ever_natural_node'] else '否'} |")
    lines += ['', '两阶、三指标的完整前 20 名及 SPT → SC 反向表见[完整排名表](brain_sc_topology_rankings.md)。表中的 SPT 强复现均要求同一人、同一 G、同一完整自然节点至少 2/3 seed 高于冻结同阶参考；原始入树计数另列，不能与附件图中的 seed 颜色混同。', '',
        '## 比较图', '',
        '![C1–C3 的逐个体 SC 同阶名次分布](../../fig/dmf_schaefer100/sc_topology_comparison/candidate_sc_rank_ecdf.png)', '',
        '**图 1｜候选的个体结构名次。** 每条 ECDF 包含 93 人，三列对应固定结构指标；C1/C2 在全部三元组中排名，C3 在全部四元组中排名。横轴为个人中间名次的 top 百分比，对数刻度，越小越好；所有值为正，不删零分或缺边个体。纵轴为累计被试比例，虚线为前 1% 的描述性阈值。没有平滑或额外插值，也不把跨人离散度当置信区间。', '',
        '![静态结构与自然 SPT 复现的双向比较](../../fig/dmf_schaefer100/sc_topology_comparison/sc_vs_spt_ranking.png)', '',
        '**图 2｜结构排名与转折复现的关系。** 上、下行分别为三元组和四元组；横轴为跨人平均个人名次的 top 百分比，对数刻度，越小越好；纵轴为背景以上窗口 ≥4/6 的复现人数。灰点包括全部曾入树同阶集合，紫色圈为 SPT 前 20，橙色菱形为本列 SC 前 20，彩色标记为 C1–C3。未入树的 SC 前列集合显示为复现人数 0，其 Syn 未填零。标题 ρ 是“结构质量”与复现的相关，结构质量等于负的横轴名次量。未对零复现人数加抖动。', '',
        '## 固定方法与验证', '',
        '三个指标为内部边权算术平均、全部内部边权几何平均（三角形/完整四节点子图强度），以及标准分辨率 1 的加权模块度单组贡献。零边保持零；不以 epsilon 补边，不截断 SC，不限制 Yeo 或空间。指标依据及完整公式见[冻结方案](brain_sc_topology_comparison_plan.md)，原始定义见 [Onnela 等，2005](https://doi.org/10.1103/PhysRevE.71.065103)与 [Newman，2004](https://doi.org/10.1103/PhysRevE.70.056131)。', '',
        'SC 得分按实际计算浮点值的精确相等处理并列；个人归一化中间名次 `(rank−0.5)/N` 的全组合平均必须为 0.5。队列展示并列时使用完整成员字典序，表中数值报告中间名次。SPT 前 20 截断处的主复现数及完整排序条件并列数量均记录在支持摘要中；不能把确定性展示顺序当作额外证据。', '',
        f"小图独立核对 {summary['verification']['direct_checks']} 项，包括直接边权计算、完整模块度矩阵式与 NetworkX 划分值、并列中间名次和统一缩放。每个被试的全部组合枚举数、名次平均、指标范围和零分数量记录在摘要中。输入/注册/科学协议的 SHA256 核对，并确认分析期间未改变；93 人身份按 subject 配对。", '',
        f"原生数据独立复核通过：在 93 人的全部登记三、四元组上用边列表及完整模块度矩阵重算 {verified['registered_score_values_checked']:,} 个指标值；另用首位被试的全部同阶组合直接计数复核 C1–C3 的九个个人名次，与排序缓存一致（归一化名次绝对容差 2×10⁻¹⁶）。从原展示缓存重计窗口出现数的 {verified['candidate_display_count_checks']} 项核对均一致。支持摘要保留检查内容及所验缓存哈希。", '',
        f"读取原始显示缓存的 Syn 审计：原生容差 {summary['selected_syn_audit']['tolerance_nats']:g} nats，最小有限值 {summary['selected_syn_audit']['minimum_selected_syn_nats']:.10g} nats；容差内负值 {summary['selected_syn_audit']['tolerance_negative_count']}，显著违反 {summary['selected_syn_audit']['violation_count']}，未裁剪。模块度的负值合法，不作为 Syn 违反。", '',
        f"完整三元组计算约 {orders['3']['elapsed_seconds']:.1f} s，四元组约 {orders['4']['elapsed_seconds']:.1f} s。逐人处理，没有保留全体个体–组合的大矩阵；仅缓存登记集合的个人读出与两类前列集合支持数据（NPZ/JSON），可由入口直接复用匹配缓存。", '',
        '## 结论的范围', '',
        '本次结果支持“突出结构为候选提供连接骨架，但静态拓扑排行榜不等同于转折窗口联合信息排行榜”。静态 SC 指标在全部边统一乘以正 G 时排名不变，不能直接产生转折窗口出现带。C1 的结构本身极突出，论文不能据此声称传统结构方法完全无法找到它；可讨论自然 SPT 如何定位其联合信息的动力学工作区间，以及区分结构第一与协同高频组合。', '',
        '完整子图强度偏向 clique；非闭合但有动力学联合影响的组合可能得分低。未实施随机网络显著性、空间/距离匹配或连接干预；未控制分区大小和 tractography 偏差。所有 93 人参与两类搜索，无独立确认集。排名不同并不证明纯三阶/四阶 PID 原子、原生多体耦合或结构之外的因果机制。', '',
        '2026-10-08 执行前再次查询 Zotero `P6UJCVG8` 并核验题名 *Emergent hierarchical organization of causal interactions in complex systems*，重新提取正文 `DXGC7JEA`（19/19 页）与补充 `MWIWKSVG`（28/28 页）。附件无明确修订日期/号，版本日期歧义保留。读取 SPT 式（10）–（12）、S5/Algorithm S1、S12.2.1；沿用冻结个体无先验扩展，保持原 target、时距、affine-TM、干预先验与共享群体 JFIC。稿件平均 SC/Yeo 首层约束与扩展的差异及 SC 行序 inferred 限制不因本次比较而消除。', '',
        '实现：[分析入口](../../scripts/analyze_dmf_sc_topology_comparison.py)、[绘图入口](../../scripts/plot_dmf_sc_topology_comparison.py)、[原生数据复核](../../scripts/verify_dmf_sc_topology_comparison.py)。支持数据：`results/dmf_schaefer100/sc_topology_comparison/`；运行记录：`docs/log/brain_critical_coalitions/sc_topology_comparison_run.log`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    tables = ['# SC 与 SPT 完整前 20 名双向表', '',
        '2026-10-08。读出来自全部 93 人及每人全部同阶组合；指标、排名与限制见[比较报告](brain_sc_topology_comparison.md)。SC 均为平均个人归一化名次的队列排名，中间名次越小越好；SPT 为既有严格同阶展示排序。完整成员必须相同才算重合。', '']
    for k in ('3','4'):
        for m in METRIC_NAMES:
            tables += [f"## {k} 阶 · SC {NAMES[m]}前 20", '',
                '| SC 名次 | 完整 ROI | 平均个人名次 / top % | 个体前 1% | 强窗口 ≥4/6 | 强窗口 ≥1/6 | 强节点格点频率：窗口/外 | 原始窗口 ≥4/6 | 曾入树 |',
                '|---:|---|---:|---:|---:|---:|---:|---:|---|']
            for r in orders[k]['sc_top20'][m]:
                sc=r['sc'][m]; d=r['above_background']; raw=r['natural_presence']
                tables.append(f"| {rank(r,m)} | {members(r)} | {sc['mean_individual_rank_percent']:.4f} | {sc['individual_top1_count']}/93 | {d['critical_ge4']}/93 | {d['critical_ge1']}/93 | {d['mean_critical_fraction']:.2%} / {d['mean_outside_fraction']:.2%} | {raw['critical_ge4']}/93 | {'是' if r['ever_natural_node'] else '否'} |")
            overlap=orders[k]['overlap'][m]
            tables += ['', f"与 SPT 前 20 的完整成员交集：{overlap['top20_intersection_count']}/20；Jaccard={overlap['jaccard']:.3f}。SC 截断分数并列数={overlap['sc_boundary']['tie_count']}；SPT 主复现数截断并列数={overlap['spt_boundary_primary_tie_count']}，完整排序条件并列数={overlap['spt_boundary_all_criteria_tie_count']}。", '']
        tables += [f'## {k} 阶 · SPT 前 20 在 SC 中的排名', '',
            '| SPT 名次 | 完整 ROI | 强窗口 ≥4/6 | 强窗口 ≥1/6 | SC 平均边权名次 | SC 闭合强度名次 | SC 模块度名次 |',
            '|---:|---|---:|---:|---:|---:|---:|']
        for r in orders[k]['spt_top20']:
            d=r['above_background']
            tables.append(f"| {r['spt_same_order_display_rank']} | {members(r)} | {d['critical_ge4']}/93 | {d['critical_ge1']}/93 | "+' | '.join(rank(r,m) for m in METRIC_NAMES)+' |')
        tables += ['', '强窗口计数使用冻结背景以上重复节点；原始入树另列。没有成为自然节点的集合的复现为零，不意味着其真实 Syn 为零。', '']
    RANKINGS.write_text('\n'.join(tables)+'\n')
    print(json.dumps({'report':str(REPORT),'rankings':str(RANKINGS)}))


if __name__ == '__main__':
    main()
