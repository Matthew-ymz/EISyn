"""Collect computed A/C methods without turning missing/invalid results into zero."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_finite_tm import save_json
from scripts.spt import nontrivial_bipartitions
from scripts.spt_continuous_phase_benchmark import score
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS

FOLDER = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
METHODS = ('spt', 'pair_phi_r_hac', 'pair_xi_hac', 'future_mi_hac',
           'block_phi_r', 'normalized_block_phi_r')
LABELS = ('SPT', 'Pair ΦR + HAC', 'Pair Ξ + HAC', 'Future MI + HAC',
          'Block ΦR', 'Normalized Block ΦR')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def proposal_ok(data, full=False):
    p = data.get('proposal_audit', {})
    return (p.get('minimum_ess', 0) >= 10
            and p.get('maximum_normalized_weight', 1) < .25
            and (not full or p.get('maximum_remaining_newton_step_rad', 1) < 1e-3))


def high_proposals(data):
    diagnostics = [d for row in data.get('rows', []) for d in
                   [row.get('denominator', {}), *row.get('diagnostics', {}).values()]
                   if d.get('ess_minimum') is not None]
    if not diagnostics:
        return {}
    return {'proposal_audit': {
        'minimum_ess': min(d['ess_minimum'] for d in diagnostics),
        'maximum_normalized_weight': max(d['maximum_normalized_weight'] for d in diagnostics),
        'maximum_remaining_newton_step_rad': max(d.get('max_remaining_step_rad', float('inf'))
                                               for d in diagnostics)}}


def tree_signature(metrics):
    return ({frozenset(g) for g in metrics['root_split']},
            {frozenset(g) for g in metrics['clades']})


def repeat_trees_agree(record):
    metrics = record.get('metrics')
    repeats = record.get('numerical_replicate_metrics', [])
    return bool(metrics) and len(repeats) == 4 and all(
        'clades' in m and tree_signature(m) == tree_signature(metrics) for m in repeats)


def high_pool_spt(data, case):
    """Build mean and replicate trees from every available high-grid split.

    This is a candidate-pool result, not a mixture of high/low EI values.
    Any new parent without a represented split is explicitly unresolved.
    """
    rows = data['rows']
    available = set.intersection(*(set(map(int, row['ei_nats'])) for row in rows))
    raw_values = {}
    selected = []
    checked_parents = {c['parent'] for c in data.get('spt_selection_checks', [])}
    def nodes(mask):
        return tuple(i + 1 for i in range(9) if mask >> i & 1)
    def build(parent, replicate=None):
        if parent.bit_count() == 1:
            return (), []
        candidates = []
        for left, right in nontrivial_bipartitions(nodes(parent)):
            l = sum(1 << (i - 1) for i in left)
            r = sum(1 << (i - 1) for i in right)
            if not {parent, l, r} <= available:
                continue
            values = np.array([z['ei_nats'][str(parent)] - z['ei_nats'][str(l)]
                               - z['ei_nats'][str(r)] for z in rows])
            raw_values[parent, l, r] = values
            value = values.mean() if replicate is None else values[replicate]
            candidates.append((float(value), l, r))
        if not candidates:
            raise KeyError(f'No high-grid split available for {parent}')
        if parent not in checked_parents and len(candidates) != 2 ** (parent.bit_count() - 1) - 1:
            raise KeyError(f'New parent {parent} needs complete split expansion')
        value, left, right = min(candidates, key=lambda x: (x[0], nodes(x[1]), nodes(x[2])))
        if replicate is None:
            selected.append({'parent': parent, 'left': left, 'right': right,
                             'available_candidates': len(candidates), 'syn_nats': value})
        _, lc = build(left, replicate)
        _, rc = build(right, replicate)
        return (nodes(left), nodes(right)), [nodes(g) for g in (left, right) if g.bit_count() > 1] + lc + rc
    def scored(replicate=None):
        root, clades = build(511, replicate)
        return score({'status': 'exploratory_refined_candidate_pool',
                      'root_split': root, 'clades': clades}, SUPPORTS[case])
    result = {'metrics': scored(), 'numerical_replicate_metrics': [scored(i) for i in range(4)],
              'high_pool_selections': selected}
    raw = np.concatenate(list(raw_values.values()))
    result['pool_syn_audit'] = {'minimum_raw_syn_nats': float(raw.min()),
        'unique_partition_count': len(raw_values), 'numerical_replicate_count': 4,
        'threshold_nats': -.01, 'significant_violation_count': int(np.count_nonzero(raw < -.01)),
        'tolerance_negative_count': int(np.count_nonzero((raw < 0) & (raw >= -.01)))}
    return result


def invalidate(record, status, error):
    if record.get('metrics'):
        record['coarse_diagnostic_metrics'] = record.pop('metrics')
    record.update(status=status, error=error, numerically_stable=False)


def collect(case):
    folder = FOLDER / case
    analysis = read(folder / 'full_analysis.json')
    full = read(folder / 'full_n1024_p256.json')
    high = read(folder / 'full_refined_n4096_p256.json')
    group = read(folder / 'group_results_n256_p256.json')
    gh = read(folder / 'group_refinement.json')
    futures = [(p.name, read(p)) for p in sorted(folder.glob('future_mi*.json'))]
    methods = {}
    for name in METHODS:
        record = {'status': 'not_computed', 'numerically_stable': False}
        if name in ('spt', 'pair_xi_hac'):
            record = {**record, **analysis.get('methods', {}).get(name, {}),
                      'score_source': '1024-sample complete full-target EI table'}
            if name == 'pair_xi_hac' and high.get('status') == 'complete' and high.get(name, {}).get('metrics'):
                record.update(high[name], score_source='4096-sample pair queries')
                record['numerically_stable'] = (
                    high[name].get('maximum_matrix_sem_nats', 1) < .01
                    and high[name].get('matrix_change_nats', 1) < .01
                    and high.get('xi_audit', {}).get('significant_violation_count', 1) == 0
                    and repeat_trees_agree(record))
            if name == 'spt' and high.get('status') == 'complete':
                checks = high.get('spt_selection_checks', [])
                record['refinement_checks'] = checks
                try:
                    record.update(high_pool_spt(high, case),
                                  score_source='4096-sample available candidate pool')
                    record['numerically_stable'] = repeat_trees_agree(record)
                    if record['pool_syn_audit']['significant_violation_count']:
                        invalidate(record, 'significant_syn_violation', str(record['pool_syn_audit']))
                except KeyError as error:
                    record['refinement_status'] = 'new_parent_requires_expansion'
                    record['refinement_error'] = str(error)
                record['validation_scope'] = 'screened high-grid candidates; not all high-grid partitions'
            record['high_proposal_audit'] = high_proposals(high).get('proposal_audit')
            record['numerically_stable'] &= proposal_ok(analysis, full=True) and proposal_ok(high_proposals(high), full=True)
            if str(high.get('status', '')).startswith('significant_'):
                invalidate(record, high['status'], high.get('error'))
            if str(full.get('status', '')).startswith('significant_') or str(analysis.get('status', '')).startswith('significant_'):
                invalidate(record, analysis.get('status', full.get('status')),
                           analysis.get('error', full.get('error')))
        elif name == 'future_mi_hac':
            usable = [(n, d) for n, d in futures if d.get('metrics')]
            if usable:
                filename, d = usable[-1]
                record.update(status=d['status'], metrics=d['metrics'],
                    numerical_replicate_metrics=d.get('numerical_replicate_metrics', []),
                    numerically_stable=d.get('status') == 'complete' and bool(d.get('precision_passed')),
                    score_source=filename, changes_nats=d.get('changes_nats'),
                    maximum_matrix_sem_nats=d.get('maximum_matrix_sem_nats'),
                    available_future_runs={n: v.get('status') for n, v in futures})
                record['matrix_precision_passed'] = bool(d.get('precision_passed'))
                record['numerically_stable'] &= repeat_trees_agree(record)
        else:
            coarse = group.get('methods', {}).get(name, {})
            refined = gh.get('methods', {}).get(name, {})
            record.update(coarse, score_source='256-sample group marginals')
            if refined.get('metrics') and gh.get('status') == 'complete':
                record.update(refined, score_source='512-sample group marginal candidate validation')
                if name == 'pair_phi_r_hac':
                    record['numerically_stable'] = (
                        refined.get('maximum_sem_nats', 1) < .01
                        and refined.get('inner_change_nats', 1) < .01
                        and refined.get('outer_change_nats', 1) < .01
                        and repeat_trees_agree(record))
                else:
                    checks = refined.get('selection_checks', [])
                    record['numerically_stable'] = repeat_trees_agree(record) and bool(checks) and all(
                        c['matches_coarse'] and (c['gap_objective'] is None or
                        c['gap_objective'] > 3 * c['gap_sem_objective']) for c in checks)
                record['validation_scope'] = 'screened group candidate pool'
                record['numerically_stable'] &= proposal_ok(gh)
            elif refined.get('status'):
                record['refinement_status'] = refined['status']
                record['refinement_error'] = refined.get('error', refined.get('missing_parent'))
                if str(refined['status']).startswith('significant_'):
                    invalidate(record, refined['status'], refined.get('error'))
            record['proposal_audit'] = gh.get('proposal_audit', group.get('proposal_audit'))
        if record.get('metrics'):
            record['reporting_status'] = 'numerically_stable' if record['numerically_stable'] else 'exploratory_unresolved'
        else:
            record['reporting_status'] = record.get('refinement_status', record['status'])
        methods[name] = record
    return {'case': case, 'run': read(folder / 'run.json'), 'methods': methods,
            'full_table_status': full.get('status', 'not_started'),
            'full_table_error': full.get('error'), 'full_analysis_error': analysis.get('error'),
            'completed_full_subsets_per_scramble': [len(r['ei_nats']) for r in full.get('rows', [])],
            'full_syn_audit': full.get('audit'), 'full_replicate_audits': full.get('replicate_audits'),
            'full_proposal_audit': analysis.get('proposal_audit'),
            'refined_full_status': high.get('status', 'not_started'),
            'group_status': group.get('status', 'not_started'),
            'refined_group_status': gh.get('status', 'not_started')}


def plot(records):
    style = {'font.family': 'DejaVu Sans', 'font.size': 9,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.spines.left': False, 'axes.linewidth': .7}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for ax, data in zip(axes, records):
            for index, name in enumerate(METHODS):
                record = data['methods'][name]
                metrics = record.get('metrics')
                if not metrics:
                    ax.text(.025, index, 'Not evaluated / invalid', color='#7E8790', va='center', fontsize=8)
                    continue
                color = '#16836A' if name == 'spt' else '#3B7298'
                stable = record['numerically_stable']
                ax.barh(index, metrics['clade_f1'], height=.53, color=color,
                        alpha=.85 if stable else .42, edgecolor=color,
                        hatch=None if stable else '///', linewidth=.5)
                repeats = [m['clade_f1'] for m in record.get('numerical_replicate_metrics', [])
                           if 'clade_f1' in m]
                if repeats:
                    ax.scatter(repeats, index + np.linspace(-.12, .12, len(repeats)),
                               s=15, facecolors='white', edgecolors='#283943', linewidths=.7, zorder=3)
                ax.text(metrics['clade_f1'] + .018, index, f"{metrics['clade_f1']:.3f}",
                        ha='left', va='center', fontsize=8)
            ax.set_yticks(range(6), LABELS)
            ax.set(ylim=(5.65, -.65), xlim=(0, 1.12), xlabel='F1 against retained module reference')
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_title(data['case'] + '   Simplified Kuramoto', loc='left', fontsize=11, weight='bold')
        fig.get_layout_engine().set(rect=(0, .09, 1, 1))
        fig.text(.5, .05, 'Dots: numerical-scramble scores from one physical dataset · Hatching: unresolved numerical validation',
                 ha='center', color='#53616D', fontsize=8)
        fig.text(.5, .015, 'Missing or invalid methods are not assigned F1 = 0',
                 ha='center', color='#53616D', fontsize=8)
        output = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_ac_comparison.png'
        fig.savefig(output, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return output


def group_text(groups):
    return ' / '.join('{' + ','.join(map(str, g)) + '}' for g in groups)


def main():
    records = [collect(case) for case in ('A', 'C')]
    payload = {'cases': records, 'scope': 'one frozen physical dataset per case; no population significance'}
    save_json(FOLDER / 'comparison_summary.json', payload)
    image = plot(records)
    lines = ['| 案例 | 方法 | 模块参考 F1 | 根切分 | 数值状态 |', '|---|---|---:|---|---|']
    for data in records:
        for name, label in zip(METHODS, LABELS):
            r = data['methods'][name]
            m = r.get('metrics')
            f1 = f"{m['clade_f1']:.4f}" if m else '未评价'
            root = group_text(m['root_split']) if m else '—'
            lines.append(f"| {data['case']} | {label} | {f1} | {root} | {r['reporting_status']} |")
    lines += ['', '斜线柱为尚未通过数值稳定性核查的探索性分数；白点是同一物理数据集的数值 scramble 得分，不能解释为独立物理重复。'
              '未完成或显著违规的方法不补零。F1 评价相对于预声明的模块参考，不是所有物理连边的恢复率。', '',
              '![六方法模块参考相似度](assets/kuramoto_hierarchy/spt_kuramoto_ac_comparison.png)', '']
    for data in records:
        lines += [f"### {data['case']}：覆盖及原始结构", '',
                  f"运行状态 `{data['run'].get('status', 'not_started')}`；完整 EI 各重复覆盖 "
                  f"`{data['completed_full_subsets_per_scramble']}`/511。full/group/refined-full/refined-group 状态："
                  f"`{data['full_table_status']}` / `{data['group_status']}` / "
                  f"`{data['refined_full_status']}` / `{data['refined_group_status']}`。", '']
        audit = data.get('full_syn_audit')
        if audit:
            lines += [f"平均表全 {audit['partition_count']} 个二分的 raw Syn 最小值为 "
                      f"{audit['minimum_syn_nats']:.6f} nats；.01 nats 容差内负值 "
                      f"{audit['tolerance_negative_count']} 个，显著违规 {audit['significant_violation_count']} 个。", '']
            all_audits = [audit, *(data.get('full_replicate_audits') or [])]
            lines += [f"计入均值及各重复，最小 raw Syn 为 {min(a['minimum_syn_nats'] for a in all_audits):.6f} nats，"
                      f"显著违规合计 {sum(a['significant_violation_count'] for a in all_audits)} 个，阈值 -.01 nats。", '']
        if data.get('full_table_error') or data.get('full_analysis_error'):
            lines += [f"完整表/分析失败：{data.get('full_analysis_error') or data.get('full_table_error')}。", '']
        for name, label in zip(METHODS, LABELS):
            r = data['methods'][name]
            if r.get('metrics'):
                lines += [f"- {label}：原始 clades 为 `{r['metrics']['clades']}`；分数来自 {r.get('score_source')}。"]
            if r.get('error') or r.get('refinement_error'):
                lines += [f"- {label}：{r.get('refinement_error') or r.get('error')}。"]
        lines += ['']
    text = '\n'.join(lines)
    report = ROOT / 'docs/reports/spt_kuramoto_ac_comparison.md'
    content = report.read_text()
    start = '<!-- AC_RESULTS -->'
    end = '<!-- AC_RESULTS_END -->'
    if end in content:
        before, remainder = content.split(start, 1)
        _, after = remainder.split(end, 1)
        content = before + start + '\n\n' + text + '\n' + end + after
    else:
        content = content.replace(start, start + '\n\n' + text + '\n' + end)
    report.write_text(content)
    print(image)
    print(report)


if __name__ == '__main__':
    main()
