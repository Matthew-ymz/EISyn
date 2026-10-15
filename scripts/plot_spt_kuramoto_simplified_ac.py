"""Compare measured A/C diagnostics at matching queries and resolutions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_simplified_ac import LAYOUTS, queries, syn_audit
from scripts.spt_kuramoto_finite_tm import summarize_precision


def comparison(record, case):
    rows = [r for r in record.get('precision', [])
            if (r['outer'], r['particles']) in LAYOUTS]
    summary = summarize_precision(rows, queries(case), case) if rows else []
    complete = [r for r in summary if r['replicates'] == 4]
    result = {'summary': complete}
    if len(complete) != len(LAYOUTS):
        return result
    result['changes'] = [max(abs(a['mean_ei_nats'][str(m)] - b['mean_ei_nats'][str(m)])
                            for m in queries(case)) for a, b in zip(complete, complete[1:])]
    high = [r for r in rows if (r['outer'], r['particles']) == LAYOUTS[-1]]
    ds = [d for r in high for d in [r['diagnostics']['denominator'],
          *[v for k, v in r['diagnostics']['queries'].items() if int(k) in queries(case)]]
          if d['ess_minimum'] is not None]
    result.update(sem=complete[-1]['maximum_ei_sem_nats'],
                  ess=min(d['ess_minimum'] for d in ds),
                  weight=max(d['maximum_normalized_weight'] for d in ds),
                  newton=max(d['max_remaining_step_rad'] for d in ds),
                  syn=syn_audit(high, case))
    return result


def main():
    folder = ROOT / 'results/spt_local_kuramoto_tm'
    old = json.loads((folder / 'flow_mixture_dt0025.json').read_text())
    new = json.loads((folder / 'simplified_ac_light.json').read_text())
    families = [('Original', old, '#7A8088', 's'), ('Simplified', new, '#16836A', 'o')]
    measures = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'xtick.labelsize': 8, 'ytick.labelsize': 8}):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), layout='constrained')
        for row, case in enumerate(('A', 'C')):
            measures[case] = {}
            for label, payload, color, marker in families:
                record = next((r for r in payload['records'] if r['case'] == case), {})
                data = comparison(record, case)
                measures[case][label] = data
                points = [r for r in data['summary'] if r['particles'] == 256]
                if points:
                    axes[row, 0].plot([r['outer'] for r in points],
                        [r['maximum_ei_sem_nats'] for r in points],
                        marker=marker, color=color, lw=1.4, ms=4)
                if 'ess' in data:
                    x = 0 if label == 'Original' else 1
                    axes[row, 1].scatter(x, data['ess'], marker=marker, color=color, s=35, zorder=3)
                    axes[row, 2].scatter(x, data['weight'], marker=marker, color=color, s=35, zorder=3)
            axes[row, 0].axhline(.01, ls='--', lw=1, color='#B84A43')
            axes[row, 0].set(xscale='log', yscale='log', xlabel='Outer samples (256 particles)',
                            ylabel='Largest EI numerical SEM (nats)', ylim=(.0003, .05))
            axes[row, 0].set_xticks([256, 1024], ['256', '1024'])
            axes[row, 0].minorticks_off()
            axes[row, 1].axhline(10, ls='--', lw=1, color='#B84A43')
            axes[row, 1].set(yscale='log', ylabel='Minimum effective sample size', ylim=(.8, 400))
            axes[row, 2].axhline(.25, ls='--', lw=1, color='#B84A43')
            axes[row, 2].set(yscale='log', ylabel='Maximum normalized weight', ylim=(.003, 1.3))
            for column in (1, 2):
                axes[row, column].set_xticks([0, 1], ['Original', 'Simplified'])
                axes[row, column].set_xlim(-.45, 1.45)
                axes[row, column].set_xlabel('1024 outer samples, 256 particles')
            for column, title in enumerate(('EI uncertainty', 'Weight coverage', 'Weight concentration')):
                axes[row, column].set_title(f'{chr(97 + row * 3 + column)}   {case} · {title}',
                                            loc='left', weight='bold', fontsize=10)
        handles = [Line2D([], [], color=color, marker=marker, lw=1.4, label=label)
                   for label, _, color, marker in families]
        handles.append(Line2D([], [], color='#B84A43', ls='--', lw=1, label='Fixed diagnostic threshold'))
        fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False)
        image_path = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_simplified_ac.png'
        fig.savefig(image_path, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    report = ROOT / 'docs/reports/spt_kuramoto_simplified_ac.md'
    lines = [
        '# A/C 简化连边的 Kuramoto 小规模复算', '',
        '2026-10-15。只复用现有估计器进行约 5 分钟以内的小规模诊断；未启动六法完整比较或参数搜索。', '',
        '**结果：简化连边明显缓解了样本内的积分权重退化；两例的预设根切分完成 4096 样本加密并通过检查。** '
        '其余查询只计算到 1024 样本，部分 EI 在 256→1024 时仍变化超过 .01 nats；完整 SPT 恢复尚未验证。', '',
        '## 网络及固定条件', '',
        '| 案例 | 被替换的超边 → 二体桥（系数） | 保留的底层模块 |',
        '|---|---|---|',
        '| A | 12345→34 (.5)，6789→78 (.5)，123456789→56 (.25) | 123、45、67、89（各 1） |',
        '| C | 123456→34 (.5)，123456789→67 (.25) | 123、456、789（各 1） |', '',
        '数字串表示节点集合。九个节点、每节点一个相位、边数和逐项系数保持不变；最高作用阶数由 9 降为 3。'
        '系数继承原层级，不能按新边的包含关系重新赋值。高阶超边改为桥后，力的频谱、作用范围和总均方幅度也改变；'
        '因此这是网络简化方案的探索性对照，不能单独归因于“阶数下降”。旧嵌套支持不作为新网络的恢复真值。', '',
        '固定独立 Uniform(-π,π)^9 干预、完整未来九相位 target、τ=.20、dt=.0025、σ=.08、原九频率。'
        '训练/验证/测试为 16384/4096/4096，开发数据 seed 0。两例与旧实验共享实际初态及 Brownian 增量；'
        '重新生成旧转移并匹配旧训练/验证联合摘要，以验证真实配对。', '',
        '估计器仍为全 502 候选边的数据学习均值流加周期条件 TM，不向拟合器传入真边。'
        '独立、恒定浓度 von Mises 输出仍是拟合近似；不把条件均值拟合成功解释为完整 SDE 密度正确。'
        'EI 使用 nats；Syn 容差 .01 nats，保留 raw 值，不裁剪。低于 -.01 的估计明确失败并记录最小值、阈值和数量。', '',
        '## 同精度比较', '',
        '各例只查 8 个 EI：完整源、预设根切分两侧、1、2、12、3、123；直接核查根切分、1|2、12|3 共 3 个二分。'
        '积分档位为 (N,P)=(256,64)、(256,256)、(1024,256)，每档 4 个独立 scramble。'
        '数值 SEM 来自积分随机化，不是独立物理数据集的标准误。旧结果只取完全相同的查询和积分档位。', '',
        '| 案例/网络 | 内层 EI 最大变化 | 外层 EI 最大变化 | 最大 EI SEM | 最低 ESS | 最大权重 | 最大剩余 Newton 步 (rad) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    network_figure = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_simplified_networks.png'
    if network_figure.exists():
        insertion = lines.index('## 同精度比较')
        lines[insertion:insertion] = [
            '![简化后的 A/C 网络结构](assets/kuramoto_hierarchy/spt_kuramoto_simplified_networks.png)', '',
            '圆圈为九个相位振子，紫色 H₃ 菱形为三体相互作用，虚线仅表示超边成员关系；'
            '实线为二体相互作用，数字为耦合系数。A 有 6 条二体边和 1 条三体超边，'
            'C 有 2 条二体边和 3 条三体超边。'
            '这是已计算生成器的真实结构，不是推断得到的 SPT。'
            '[可编辑结构图](assets/kuramoto_hierarchy/spt_kuramoto_simplified_networks.drawio)。', '',
        ]
    for case in ('A', 'C'):
        for label, _, _, _ in families:
            d = measures[case][label]
            if 'ess' in d:
                lines.append(f"| {case}/{label} | {d['changes'][0]:.6f} | {d['changes'][1]:.6f} | "
                             f"{d['sem']:.6f} | {d['ess']:.3f} | {d['weight']:.6f} | {d['newton']:.3g} |")
            else:
                lines.append(f'| {case}/{label} | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |')
    lines += ['', 'EI/Syn 数值门槛保持 .01 nats；proposal 门槛为最低 ESS≥10、最大权重<.25、'
              '剩余 Newton 步<1e-3 rad。没有为新网络放宽阈值。', '',
              '![同档位数值诊断](assets/kuramoto_hierarchy/spt_kuramoto_simplified_ac.png)', '',
              '## 新网络结果及边界', '',
              f"总状态：`{new['status']}`；计算耗时 {new['elapsed_compute_seconds']:.2f} 秒。", '',
              '| 案例 | 已完成积分重复 | 轻量检查 | 步长 RMS (rad) | 已检查 raw Syn 最小值 (nats) | 容差内负值/显著违规 |',
              '|---|---:|---|---:|---:|---:|']
    for r in new['records']:
        audits = r.get('critical_syn_audits', [])
        minimum = min((a['minimum_raw_syn_nats'] for a in audits), default=None)
        small = sum(a['tolerance_negative_count'] for a in audits)
        bad = sum(a['significant_violation_count'] for a in audits)
        verdict = {True: '通过', False: '未通过'}.get(r.get('lightweight_checks_passed'), '未完成')
        minimum_text = f'{minimum:.6f}' if minimum is not None else '未计算'
        lines.append(f"| {r['case']} | {len(r['precision'])}/12 | {verdict} | "
                     f"{r['step_precision']['rms_difference_rad'][-1]:.3g} | {minimum_text} | {small}/{bad} |")
    if new['protocol'].get('root_refinement'):
        lines += ['', '### 剩余额度中的根切分加密', '',
                  '仅将完整源和预设根切分两侧这 3 项 EI 加密到 N=4096、P=256，仍使用同一模型及 4 个 scramble。'
                  '这次自适应追加由数值误差触发，不依据恢复分数；计算仍计入同一个 300 秒上限。', '',
                  '| 案例 | 完成重复 | 1024→4096 最大 EI 变化 (nats) | 最大 EI SEM (nats) | 根 Syn 均值 ± 数值 SEM (nats) | 最低 ESS | 最大权重 | 根查询门槛 |',
                  '|---|---:|---:|---:|---:|---:|---:|---|']
        notes = []
        for r in new['records']:
            d = r.get('root_refinement_summary')
            count = len(r.get('root_precision', []))
            if d:
                lines.append(f"| {r['case']} | {count}/4 | {d['maximum_ei_change_from_1024_nats']:.6f} | "
                    f"{d['maximum_ei_sem_nats']:.6f} | {d['root_syn_mean_nats']:.6f} ± {d['root_syn_sem_nats']:.6f} | "
                    f"{d['minimum_ess']:.3f} | {d['maximum_normalized_weight']:.6f} | "
                    f"{'通过' if r.get('root_precision_passed') else '未通过'} |")
                notes.append(f"{r['case']} 根加密 raw Syn 最小值 {d['minimum_raw_syn_nats']:.6f} nats，"
                    f"容差内负值/显著违规 {d['tolerance_negative_count']}/{d['significant_violation_count']}；"
                    f"根 Syn 加密变化 {d['root_syn_change_from_1024_nats']:.6f} nats。")
            else:
                lines.append(f'| {r["case"]} | {count}/4 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |')
        for note in notes:
            lines += ['', note]
    lines += ['', '这里的“通过”仅指上述有限查询、1024 样本的轻量诊断；即使根切分完成 4096 样本加密，也未覆盖其余查询的最高档、'
              '全部 511 个源子集、9330 个二分、候选切分排名或 group-target 边缘。'
              '根加密通过只认证预设切分的这 3 项查询；256→1024 的粗档变化仍完整保留。'
              '没有生成 SPT 树、六法 F1 或统计显著性结论。网络简化可能使成对基线也更容易恢复模块；不预设 SPT 必然领先。', '',
              '## 手稿依据与复现', '',
              '本任务重新通过 Zotero 本地 API 按题名核对父项 `P6UJCVG8`，'
              '*Emergent hierarchical organization of causal interactions in complex systems*，重新读取主文 `DXGC7JEA` '
              '及 SI `MWIWKSVG` 的相关全文：Methods (5)–(12)、SI S1.2 (S7)–(S15)、S5.1 (S49)–(S50)、'
              'S7 (S67)–(S71)。当前只有一组主文/SI，均无明确稿件日期或版本；附件 2026-10-04 的元数据修改时间不作为稿件版本日期。', '',
              '共同干预、固定完整 target、EI 定义和 raw Syn 遵循主文。二体及三体相互作用分别符合 SI (S68)、(S69) 的核形式，'
              '九节点网络及桥结构是本次试验设计。完整未来相位不同于 SI S7 的未来增量 target；本周期条件 TM 也不同于 '
              'SI S1.2 的 affine TM，不套用其 Wishart 修正。没有在小规模诊断中执行手稿要求的完整 SPT 精确切分。', '',
              '内部缓存：`results/spt_local_kuramoto_tm/simplified_ac_light.json`。'
              '独立文件保留原 A/C 缓存，未修改正在进行的 B 比较。', '',
              '复现入口：`scripts/spt_kuramoto_simplified_ac.py --output <new.json> --budget-seconds 300`；'
              '随后可用同一 output 及 `--refine-roots --budget-seconds 300` 在累计额度内追加根查询；'
              '固定结果绘图与报告：`scripts/plot_spt_kuramoto_simplified_ac.py`。', '']
    report.write_text('\n'.join(lines))
    print(image_path)
    print(report)


if __name__ == '__main__':
    main()
