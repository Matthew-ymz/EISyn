#!/usr/bin/env python3
"""Export the completed cohort's exploratory evidence without inventing a success claim."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dmf_critical_coalitions import BASE
from scripts.run_dmf_subject_consistency import atomic_json, digest

REPORT = ROOT / 'docs/reports/brain_critical_coalition_all93.md'
FIGURE = ROOT / 'fig/dmf_schaefer100/critical_coalition_all93.png'


def table(rows, n, ids):
    text = ['| 编号与完整 ROI 集合（1-based） | 阶数 | 窗口 ≥4/6 | ≥3/6 | ≥1/6 | 低/高任一格泄漏 | 窗口外出现 | ≥4/6 且窗口专属 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        a = r['above_background']; name = ids.get(tuple(r['members']), '')
        label = (name + '：' if name else '') + '、'.join(map(str, r['members_one_based']))
        text.append(f'| {label} | {r["order"]} | {a["critical_ge4"]}/{n} | {a["critical_ge3"]}/{n} | {a["critical_ge1"]}/{n} | {a["low_any"]}/{n}；{a["high_any"]}/{n} | {a["outside_any"]}/{n} | {a["exclusive_ge4"]}/{n} |')
    return '\n'.join(text)


def build_report(s, c, base, figure, development_only=False):
    n = s['subject_count']; ranks = s['rankings']; audit = s['audit']
    ids = {tuple(r['members']):r['display_id'] for r in s['selected']}
    best = ranks['strict_preferred'][0]; a = best['above_background']
    relaxed = ranks['relaxed_preferred'][0]; b = relaxed['above_background']
    exclusive = ranks['exclusive_preferred'][0]; e = exclusive['above_background']
    gate = s['comparison_80percent_10percent']
    complete = json.loads((base/'all93_simulation_complete.json').read_text()) if (base/'all93_simulation_complete.json').exists() and not development_only else None
    elapsed = f'{complete["elapsed_seconds"]/3600:.2f} 小时' if complete else '开发缓存回归检查；不计为全 93 人完成'
    all_best = ranks['strict_all_orders'][0]
    positive = gate['matching_count'] > 0
    target_text = (f'有 {gate["matching_count"]} 个完整集合达到历史发生率参照（窗口 ≥4/6 至少 {gate["critical_min"]}/{n}，两端各不超过 {gate["low_max"]}/{n}）'
        if positive else f'没有集合达到历史发生率参照（窗口 ≥4/6 至少 {gate["critical_min"]}/{n}，两端各不超过 {gate["low_max"]}/{n}）')
    figure_rel = '../../fig/dmf_schaefer100/critical_coalition_all93.png' if not development_only else str(figure)
    labels_path = ROOT/'results/dmf_schaefer100/schaefer100_labels.txt'
    labels = labels_path.read_text().splitlines()
    label_notes = []
    for r in s['selected']:
        if r['order'] <= 10:
            label_notes.append(f'- {r["display_id"]}：'+ '；'.join(f'{i+1}=`{labels[i]}`' for i in r['members'])+'。')
    ci = a['critical_ge4_ci95_wilson']
    return f'''# 全 93 人 DMF 转折窗口的自然 SPT 完整组合搜索

{'开发缓存回归预览，只有 8 人；此文件不是 93 人结果。' if development_only else '93 人 × 41 个 G × 3 seed 的全部 11,439 条件已经计算与汇总；图形导出仍待最后目视复核。'}

## 完整扫描的候选证据

共登记 **{s['registered_high_order_sets']:,} 个三阶以上完整集合**，其中 **{s['registered_preferred_sets']:,} 个为 3–10 ROI**。所有 3–99 阶都参与登记与排序。全阶数的最高强窗口复现数为 **{s['maximum_strong_critical_ge4']}/{n}（至少 4/6 格）**；放宽为至少 3/6 和至少 1/6 格后，最高值分别为 **{s['maximum_strong_critical_ge3']}/{n}** 与 **{s['maximum_strong_critical_ge1']}/{n}**。这三个最大值可能来自不同集合，不能拼接为同一个候选的表现。

{target_text}，其中满足同等频率的全扫描窗口专属性者为 {gate['window_exclusive_count']} 个。这个历史参照仅用于对照，不是本轮扩展或停止的门槛。候选以下按复现、泄漏及完整身份直接排序，不使用可调权重总分。

优先查看 **ROI {'、'.join(map(str,best['members_one_based']))}（{best['order']} 阶）**：强窗口复现 ≥4/6 为 {a['critical_ge4']}/{n}，≥3/6 为 {a['critical_ge3']}/{n}，≥1/6 为 {a['critical_ge1']}/{n}；低/高 G 泄漏分别为 {a['low_any']}/{n}、{a['high_any']}/{n}。其 ≥4/6 发生率的描述性 Wilson 95% 区间为 {ci[0]*100:.1f}%–{ci[1]*100:.1f}%。窗口外出现于 {a['outside_any']}/{n} 人，严格窗口专属为 {a['exclusive_ge4']}/{n} 人。

![全部个体的自然 SPT 候选扫描]({figure_rel})

**图 1｜全个体完整成员搜索。** a，各 ROI 阶数所有完整集合中的最大个体复现数，分别显示原始入树 ≥4/6 及背景以上 ≥4/6、≥3/6、≥1/6；是选择后的最大值。b，严格排序前三个 3–10 ROI 候选的两端泄漏与三种窗口宽度命中要求，误差条为个体比例 Wilson 95% 描述性区间。c，C1 的逐个体完整扫描，按各自发放率转折中点排序和对齐；四种颜色区分全部 seed 缺失、单 seed、重复但未超过参考、至少 2/3 seed 超过参考，白色为未扫描范围；红虚线围住六个窗口格点的显示单元。d，前三个候选的背景以上逐点复现率，C1 阴影为描述性区间，每点分母是实际覆盖该相对 G 的人数，扫描边缘不固定为 {n}；连接实际格点，不平滑。e，C1 每人/G/seed 入树条件的原始局部 Syn，未入树保持缺失；虚线为同阶 G=0 参考。图例位于各数据区域之外。

### 严格窗口复现排序：优先 3–10 ROI

{table(ranks['strict_preferred'],n,ids)}

### 放宽为窗口任一格复现：优先 3–10 ROI

{table(ranks['relaxed_preferred'],n,ids)}

这里的放宽只改变汇总读出：仍要求同一人、同一 G 至少 2/3 seed 的**同一个完整自然节点**高于冻结背景，不放宽完整成员、不移动窗口，也不改变构树。放宽排序首位为 ROI {'、'.join(map(str,relaxed['members_one_based']))}，≥1/6 达 {b['critical_ge1']}/{n}，但 ≥4/6 为 {b['critical_ge4']}/{n}，窗口外出现为 {b['outside_any']}/{n}。

### 全扫描窗口专属性排序

{table(ranks['exclusive_preferred'][:3],n,ids)}

此排序首位为 ROI {'、'.join(map(str,exclusive['members_one_based']))}，在 {e['exclusive_ge4']}/{n} 人中同时满足 ≥4/6 命中与 35 个窗口外格点均不重复出现。专属性和覆盖人数是不同取舍；不能用“低/高两端少”替代“整个窗口外少”。

### 全部阶数及二阶参照

{table(ranks['strict_all_orders'][:3],n,ids)}

全阶严格首位为 ROI {'、'.join(map(str,all_best['members_one_based']))}（{all_best['order']} 阶）。二阶参照如下，未计入高阶候选：

{table(ranks['pair_reference'],n,ids)}

## 哪些候选适合继续抓住

严格排序首位的平均窗口格点发生比例为 {a['mean_critical_fraction']*100:.1f}%，低/高两端平均为 {a['mean_low_fraction']*100:.1f}% / {a['mean_high_fraction']*100:.1f}%，其他 35 格平均为 {a['mean_outside_fraction']*100:.1f}%。窗口减两端平均为 {a['critical_minus_endpoint_fraction']*100:.1f} 个百分点，窗口减全部非窗口平均为 {a['critical_minus_outside_fraction']*100:.1f} 个百分点。比例按每人的窗口 6 格、各端 4 格、窗口外 35 格分别归一化，再跨人平均。

当前可以直接依据这些结果挑选复现优先、窗口内至少一次出现优先和窗口专属性优先的候选。需要区分“在转折附近常出现”“转折附近更集中”和“只在转折附近出现”。所有 93 人都参与本轮选择，以上候选频率和区间属于搜索后的描述性证据；原来的 85 人不再是未接触的候选验证集。没有对选择后候选提供未经选择校正的显著性声明。是否形成论文主张，留待针对这些实际候选的下一步解释与独立检查。

## 冻结条件与本轮修订

本轮根据用户明确指示取消“8 人通过才计算其余 85 人”的停止规则，将全部 93 人纳入探索，不设置新的成功才继续门槛。已有 8 人 984 棵自然树、100 个独立 G=0 背景 seed 与匹配的密度缓存复用；其余条件补齐。旧 8 人的结果和协议保留为历史阶段，未改写为成功。

科学内核沿用原冻结协议：原生个体 SC，共享群体 SC 在 G=1 标定后固定的 JFIC，G=0–4、步长 0.1、seed 3/4/5；每 ROI 的 E/I 对为不可分叶块，完整 200 维未来 target；2,048 组独立 U(0.30,0.70) 输入，300-step、dt=0.001 s、sigma=0.01、ridge=10⁻⁶。同 seed 跨 G 和个体使用相同输入与噪声。只在同一个全系统联合密度内查询各子集。

采用 affine triangular TM：以匹配均匀干预矩的因子化 Gaussian 源先验和线性 Gaussian 残差构造联合密度。这是精确均匀干预 EI 的近似读出；不是另换非 TM 估计器。自然 SPT 小节点 ≤10 ROI 精确枚举全部无序非空切分，大节点采用补充 S5 的四个谱排序、平均链接与原顺序 prefix cuts，去重后最大化两个子集保留的增量。无 Yeo、空间、跨人或阶数先验；递归到 100 个叶。3–99 ROI 全登记，2 ROI 为参照，100 ROI 根不作为特定候选。完整节点身份用两字 uint64 的无碰撞 100-bit 编码保存，不使用模糊匹配。

99 个内部节点的预算闭合到**跨 ROI 增量**，ROI 内 E/I 增量保留在 100 个叶块。两者相加才是以 200 个独立标量为细分源的全系统 Ξ。内部节点的 Syn 仍为同一 target 下父块 EI 减两个子块 EI，不把叶块内增量重复加入节点强度。

相较补充 S1.2 式 S13–S14 的通用样本 TM 互信息估计路径，本实验继承已冻结的 DMF 共同密度 log-determinant 读出，未另外减去该节的维数相关有限样本偏差项。此处不是声称两种估计路径完全相同；保留原读出以便逐条件复现旧密集曲线，另用冻结背景描述有限样本量级，背景不能替代正式偏差校正。

每阶背景参考为 100 棵独立 G=0 树中逐树最大原始节点 Syn 的 95% 分位数，linear 规则，至少 20 棵出现该阶才标定。这是筛选参考，不是正式节点 p 值。共享 JFIC 的 G=0 动力学在 SC 间相同，93 个重复 SC 不能充当 93 个独立背景样本。

个人窗口由既有发放率最大正斜率区间 [L,U] 向两侧各扩 0.2，共六格；严格读出 ≥4/6，新增较宽松 ≥3/6 与 ≥1/6 同时呈现。低端 G=0–0.3，高端 G=3.7–4.0，分别四格；任一格泄漏与 ≥2/4 阶段出现分开存储。全部非窗口 35 格均保留，不只检查两端。

## 全条件有效性和完成证据

- {audit['tree_count']:,} 棵完整树与 {audit['density_count']:,} 份密度逐条件核验；共 {audit['internal_node_count']:,} 个内部节点，评估 {audit['candidate_count']:,} 个切分候选。
- 原密集扫描 Ξ 最大逐条件差异 {audit['max_dense_xi_difference_nats']:.3g} nats；最大记录闭合误差 {audit['max_closure_error_nats']:.3g} nats，另独立重算节点预算闭合。
- 原始 Syn 不裁剪。数值非负容差 10⁻⁸ nats；最小候选 Syn {audit['minimum_candidate_syn_nats']:.8g} nats；候选容差内负数 {audit['candidate_tolerance_negative_count']}，选中节点容差内负数 {audit['selected_tolerance_negative_count']}，pair/query 容差内负数 {audit['pair_tolerance_negative_count']}/{audit['query_tolerance_negative_count']}。低于 −10⁻⁸ nats 或非有限值会显式失败。状态越界计数 {audit['outside_state_count']}。
- 每份密度核对科学协议、共同输入 SHA256 与噪声 seed；{audit['imported_density_count']} 份历史 pilot 密度经过匹配后复用，其余由匹配模拟生成。树始终按本轮标准规则构建。
- 本轮扩展扫描四进程实际耗时 {elapsed}；统计耗时 {s['analysis_elapsed_seconds']:.1f} 秒。以前完成的 8 人和背景耗时另计。
- 新紧凑统计在已有 8 人的全部 74,991 个完整集合上逐一与旧统计对照，每集合 17 项已有读出完全一致；相关科学检查 23 项通过。全 93 人的最终图形另需目视复核，导出不等于目视检查完成。

## 稿件核对与解释边界

2026-10-07 重新查询 Zotero 父条目 `P6UJCVG8` 并核验题名 *Emergent hierarchical organization of causal interactions in complex systems*，重新读取正文 `DXGC7JEA`（19/19 页）和补充 `MWIWKSVG`（28/28 页）。两附件无明确修订号/稿件日期；附件元数据编辑不能确定版本日期。相关位置为正文 Brain/Fig.2 第 6–7 页、Methods 式（5）–（12）第 15–17 页；补充 S1.2 第 3–4 页、S5/Algorithm S1 第 8–9 页、S12.2.1/式 S82–S87 第 23 页。

稿件 Brain 主分析使用平均 SC、Yeo 首层约束、G=0–3；本实验扩展为原生个体 SC、无先验树、G=0–4。均匀输入经 moment-matched Gaussian affine-TM 近似，不能称为精确均匀干预信息。发放率转折为操作性定位，不自动等同于物理临界点；G=4 不自动是每人的强耦合极限。高阶节点 Syn 是完整树路径下两子块的层级残差，不是纯高阶 PID 原子或原生多体机制的单独证明。共享 Monte Carlo 输入使 seed 误差配对，3 个 seed 不当作额外个体；队列不是人体总体随机抽样。

继承 ROI 标签行序仍为 inferred，以下标签供候选定位，不宣称已完成个体解剖核验：

{chr(10).join(label_notes)}

## 复现与本地支持数据

- [全部条件执行入口](../../scripts/run_dmf_critical_coalitions_all93.py)，复用冻结科学内核并自动衔接统计、绘图、报告。
- [紧凑全组合统计](../../scripts/analyze_dmf_critical_coalitions_all93.py)，保存全部候选的完整身份和发生计数，不只保存展示名单。
- [绘图入口](../../scripts/plot_dmf_critical_coalitions_all93.py)。图例在数据区域外，缺失不补零。
- 结果目录 `results/dmf_schaefer100/critical_coalitions/`：`all93_extension_contract.json`、`all93_analysis_contract.json`、`all93_registry.npz`、`all93_summary.json`、`all93_display.npz`、逐条件 `density/` 与 `trees/`。
- 主日志 `docs/log/brain_critical_coalitions/all93_run.log`。NPZ/JSON 为昂贵结果复用支持；未生成 CSV。旧阶段见[8 人执行报告](brain_critical_coalition_results.md)。
'''


def export(base, output=REPORT, development_only=False, update_index=True):
    prefix='all93_devcheck' if development_only else 'all93'
    s=json.loads((base/(prefix+'_summary.json')).read_text())
    c=json.loads((base/'contract.json').read_text())
    if s['scientific_contract_sha256'] != digest(base/'contract.json'):
        raise ValueError('Report summary provenance mismatch')
    if not development_only and (s['subject_count']!=93 or s['condition_count']!=11439):
        raise ValueError('Do not report a partial cohort as all93 complete')
    figure=FIGURE if not development_only else Path('/tmp/eisyn_all93_devcheck.png')
    if not figure.exists():
        raise FileNotFoundError(figure)
    text=build_report(s,c,base,figure,development_only)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text)
    atomic_json(base/(prefix+'_report_metadata.json'),dict(output=str(output),
        exported_utc=datetime.now(timezone.utc).isoformat(),report_sha256=digest(output),
        summary_sha256=digest(base/(prefix+'_summary.json')),figure_sha256=digest(figure),
        reporting_sha256=digest(Path(__file__)),visual_review='pending'))
    if update_index and not development_only:
        index=ROOT/'docs/reports/brain.md'
        marker='<!-- dmf-critical-coalitions-all93 -->'
        end_marker='<!-- end-dmf-critical-coalitions-all93 -->'
        current=index.read_text()
        a=s['rankings']['strict_preferred'][0]
        addition=f'''\n{marker}
<a id="dmf-critical-coalitions-all93"></a>

### Q.4 全 93 人自然 SPT 组合探索

按后续授权，取消 8 人通过才扩展的门槛，已完成全部 **93×41×3=11,439** 棵自然树。完整登记 {s['registered_high_order_sets']:,} 个三阶以上集合；最高背景以上窗口复现数为 ≥4/6：**{s['maximum_strong_critical_ge4']}/93**，≥3/6：**{s['maximum_strong_critical_ge3']}/93**，≥1/6：**{s['maximum_strong_critical_ge1']}/93**。3–10 ROI 严格首位为 **{'、'.join(map(str,a['members_one_based']))}**，低/高 G 任一格泄漏 {a['above_background']['low_any']}/93、{a['above_background']['high_any']}/93，其他窗口外出现 {a['above_background']['outside_any']}/93，严格窗口专属 {a['above_background']['exclusive_ge4']}/93。

![全 93 人自然 SPT 组合搜索](../../fig/dmf_schaefer100/critical_coalition_all93.png)

*图 Q4｜原始入树与背景以上的严格、较宽松复现读出，首位候选的全部个体扫描与原始节点 Syn。全 93 人都参与选择，是探索性结果；原来的 85 人不再是未接触的验证集。完整候选、全阶排名、归一化窗口富集和解释边界见[93 人报告](brain_critical_coalition_all93.md)。图形导出待最后目视复核。*
{end_marker}
'''
        if marker in current:
            start=current.index(marker)
            if end_marker not in current[start:]:
                raise ValueError('Cannot safely locate end of owned all93 report section')
            end=current.index(end_marker,start)+len(end_marker)
            index.write_text(current[:start]+addition.lstrip('\n')+current[end:])
        else:
            index.write_text(current+addition)
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=REPORT)
    p.add_argument('--development-only',action='store_true')
    p.add_argument('--no-update-index',action='store_true')
    a=p.parse_args();export(a.output_dir,a.output,a.development_only,not a.no_update_index)


if __name__=='__main__':
    main()
