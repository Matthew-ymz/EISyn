"""Audit and plot the three retained intervention methods, preserving unresolved results."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_ac_selected import SelectedCase
from scripts.spt_kuramoto_finite_tm import save_json
from scripts.spt_local_hyperedge_kuramoto import SUPPORTS

FOLDER = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
FIGURE = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_kuramoto_ac_selected.png'
REPORT = ROOT / 'docs/reports/spt_kuramoto_ac_selected_comparison.md'
METHODS = ('spt', 'surd', 'block_phi_r')
LABELS = ('SPT', 'SURD +\nmax-weight tree', r'Block $\Phi_R$')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def signature(metrics):
    return (frozenset(frozenset(s) for s in metrics['root_split']),
            frozenset(frozenset(s) for s in metrics['clades']))


def module_signature(metrics, case):
    supports = SUPPORTS[case]
    primitive = [e for e in supports if not any(set(g) < set(e) for g in supports)]
    return (signature(metrics)[0], frozenset(frozenset(s) for s in metrics['clades']
            if not any(set(s) < set(p) for p in primitive)))


def stability(metrics, repeats, case):
    if not metrics or len(repeats) != 4:
        return {'raw_tree_agreement_count': 0, 'scored_clade_agreement_count': 0, 'numerical_f1_range': None}
    return {'raw_tree_agreement_count': sum(signature(m) == signature(metrics) for m in repeats),
            'scored_clade_agreement_count': sum(module_signature(m, case) == module_signature(metrics, case) for m in repeats),
            'numerical_f1_range': [min(m['clade_f1'] for m in repeats), max(m['clade_f1'] for m in repeats)]}


def proposals(rows):
    ds = [d for r in rows for d in [r.get('denominator', {}), *r.get('diagnostics', {}).values()]
          if d.get('ess_minimum') is not None]
    if not ds:
        return {}
    return {'minimum_ess': min(d['ess_minimum'] for d in ds),
            'maximum_normalized_weight': max(d['maximum_normalized_weight'] for d in ds),
            'maximum_remaining_newton_step_rad': max(d.get('max_remaining_step_rad', 0) for d in ds)}


def proposal_ok(p):
    return p.get('minimum_ess', 0) >= 10 and p.get('maximum_normalized_weight', 1) < .25


def collect_spt(case, folder):
    result = read(folder / 'spt_selected.json')
    full = read(folder / 'full_n1024_p256.json')
    high = read(folder / 'spt_selected_refined.json')
    record = {'status': result.get('status', 'not_computed'), 'score_source': 'N1024 full enumeration',
              'full_subsets_per_scramble': [len(r['ei_nats']) for r in full.get('rows', [])]}
    if not result.get('metrics'):
        return {**record, 'error': result.get('error')}
    # Re-audit all Xi as well as every Syn, including numerical repeats.
    context = SelectedCase(case)
    audited = [context.table_tree(v) for v in
               [full['mean_ei_nats'], *[r['ei_nats'] for r in full['rows']]]]
    record.update(metrics=result['metrics'], numerical_replicate_metrics=result['numerical_replicate_metrics'],
                  coarse_audits=[v['audit'] for v in audited], coarse_proposal=proposals(full['rows']),
                  closure_error_nats=result['closure_error_nats'])
    if high.get('status') == 'complete':
        record.update(metrics=high['metrics'], numerical_replicate_metrics=high['numerical_replicate_metrics'],
                      score_source='N4096 screened candidate pool', refinement_audit=high['audit'],
                      high_proposal=proposals(high['rows']), high_query_count=len(high['source_masks']))
    elif str(high.get('status', '')).startswith('significant_'):
        record['coarse_diagnostic_metrics'] = record.pop('metrics')
        audit = high.get('audit', {})
        return {**record, 'status': 'invalid_nonnegativity', 'refinement_audit': audit,
                'error': f"High-grid Syn/Xi minimum={min(audit.get('minimum_raw_syn_nats', 0), audit.get('minimum_xi_nats', 0))}; "
                         f"threshold=-.01; count={audit.get('significant_violation_count', 0) + audit.get('significant_xi_violation_count', 0)}"}
    record.update(stability(record['metrics'], record['numerical_replicate_metrics'], case))
    record['numerically_stable'] = (high.get('status') == 'complete'
        and record['scored_clade_agreement_count'] == 4 and proposal_ok(record['coarse_proposal'])
        and proposal_ok(record.get('high_proposal', {}))
        and record['coarse_proposal']['maximum_remaining_newton_step_rad'] < 1e-3
        and record.get('high_proposal', {}).get('maximum_remaining_newton_step_rad', 1) < 1e-3)
    record['validation_scope'] = 'Full coarse enumeration; heuristic high-grid candidate screening'
    return record


def collect_surd(folder, full):
    data = read(folder / 'surd_selected_n1024_p8192.json')
    refinement = read(folder / 'surd_selected_refined_n4096_p8192.json')
    if refinement.get('status') == 'complete':
        data = refinement
    outer = data.get('protocol', {}).get('outer', 1024)
    record = {'status': data.get('status', 'not_computed'), 'approximation': 'local Gaussian posterior TM',
              'error': data.get('error')}
    if not data.get('metrics'):
        return record
    record.update(metrics=data['metrics'], numerical_replicate_metrics=data['numerical_replicate_metrics'],
                  approximation_audits=data['approximation_audits'],
                  maximum_atom_sem_nats=data['maximum_atom_sem_nats'], score_source=f'N{outer} P8192 posterior TM',
                  refinement_status=refinement.get('status', 'not_started'))
    record.update(stability(record['metrics'], record['numerical_replicate_metrics'], folder.name))
    high = [next(z for z in r['layouts'] if (z['outer'], z['particles']) == (outer, 8192)) for r in data['records']]
    low = [next(z for z in r['layouts'] if (z['outer'], z['particles']) == (outer, 4096)) for r in data['records']]
    short = [next(z for z in r['layouts'] if (z['outer'], z['particles']) == (256 if outer == 1024 else 1024, 8192)) for r in data['records']]
    weights = np.asarray([r['synergy_atoms_nats'] for r in high])
    record['maximum_particle_atom_change_nats'] = float(abs(weights.mean(0) - np.mean([r['synergy_atoms_nats'] for r in low], 0)).max())
    record['maximum_outer_atom_change_nats'] = float(abs(weights.mean(0) - np.mean([r['synergy_atoms_nats'] for r in short], 0)).max())
    record['particle_raw_tree_agreement_count'] = sum(signature(a['metrics']) == signature(b['metrics']) for a, b in zip(high, low))
    record['outer_raw_tree_agreement_count'] = sum(signature(a['metrics']) == signature(b['metrics']) for a, b in zip(high, short))
    record['particle_scored_clade_agreement_count'] = sum(module_signature(a['metrics'], folder.name) == module_signature(b['metrics'], folder.name) for a, b in zip(high, low))
    record['outer_scored_clade_agreement_count'] = sum(module_signature(a['metrics'], folder.name) == module_signature(b['metrics'], folder.name) for a, b in zip(high, short))
    record['proposal_audits'] = [r['proposal_audit'] for r in data['records']]
    common_source = full
    common_masks = list(range(1, 512))
    if outer == 4096:
        common_source = read(folder / 'spt_selected_refined.json')
        common_masks = sorted(set.intersection(*(set(map(int, r['ei_nats'])) for r in common_source.get('rows', [])))) if len(common_source.get('rows', [])) == 4 else []
    if common_source.get('status') == 'complete' and len(common_source.get('rows', [])) == 4 and common_masks:
        common_rows = {r['scramble']: r for r in common_source['rows']}
        common = np.array([[common_rows[r['scramble']]['ei_nats'][str(m)] for m in common_masks] for r in data['records']])
        specific = np.array([[r['mean_specific_nats'][m] for m in common_masks] for r in high])
        differences = specific - common
        center = differences.mean(0)
        sem = differences.std(0, ddof=1) / 2
        record['common_ei_agreement'] = {
            'maximum_absolute_mean_difference_nats': float(abs(center).max()),
            'maximum_paired_numerical_sem_nats': float(sem.max()),
            'outside_01_plus_three_sem_count': int(np.count_nonzero(abs(center) > .01 + 3 * sem)),
            'source_subset_count': len(common_masks), 'outer': outer,
            'scope': 'Diagnostic per-subset agreement; no simultaneous statistical certificate'}
    density_ok = all(abs(a['full_posterior_gaussian_kl_mean_nats']) < .01
                     and a['kl_diagnostic_below_minus_01_count'] == 0 for a in data['approximation_audits'])
    record['density_diagnostic_passed'] = density_ok
    record['numerically_stable'] = (density_ok and record['scored_clade_agreement_count'] == 4
        and record['particle_scored_clade_agreement_count'] == 4 and record['outer_scored_clade_agreement_count'] == 4
        and record['maximum_particle_atom_change_nats'] < .01
        and record['maximum_outer_atom_change_nats'] < .01
        and record['maximum_atom_sem_nats'] < .01
        and record.get('common_ei_agreement', {}).get('outside_01_plus_three_sem_count', 1) == 0
        and record.get('common_ei_agreement', {}).get('maximum_paired_numerical_sem_nats', 1) < .01
        and all(proposal_ok(p) and p['maximum_six_sd_chart_extent_rad'] < np.pi for p in record['proposal_audits']))
    return record


def collect(case):
    folder = FOLDER / case
    methods = {}
    try:
        methods['spt'] = collect_spt(case, folder)
    except (ArithmeticError, KeyError, ValueError) as e:
        methods['spt'] = {'status': 'audit_failure', 'error': str(e)}
    methods['surd'] = collect_surd(folder, read(folder / 'full_n1024_p256.json'))
    block = read(folder / 'block_selected_n256_p256.json')
    refined = read(folder / 'block_selected_refinement.json')
    for name in METHODS[2:]:
        record = {**block.get('methods', {}).get(name, {'status': block.get('status', 'not_computed')}),
                  'proposal_audit': block.get('proposal_audit', {}), 'score_source': 'N256 P256 group marginals'}
        if record.get('metrics'):
            record.update(stability(record['metrics'], record['numerical_replicate_metrics'], case))
            checks = record['selection_diagnostics']
            record['numerically_stable'] = (record['scored_clade_agreement_count'] == 4
                and proposal_ok(record['proposal_audit'])
                and all(c['maximum_inner_change'] < .01 and c['maximum_outer_change'] < .01 for c in checks))
        high = refined.get('methods', {}).get(name, {})
        if record.get('status') == 'complete' and high.get('status') == 'complete':
            record.update(high, score_source='N512 P512 screened group candidates',
                          proposal_audit=refined.get('proposal_audit', {}))
            record.update(stability(record['metrics'], record['numerical_replicate_metrics'], case))
            primitive = [e for e in SUPPORTS[case] if not any(set(g) < set(e) for g in SUPPORTS[case])]
            relevant_checks = [c for c in record['selection_diagnostics']
                               if not any(set(i + 1 for i in range(9) if c['parent'] >> i & 1) <= set(p) for p in primitive)]
            record['numerically_stable'] = (record['scored_clade_agreement_count'] == 4
                and proposal_ok(record['proposal_audit'])
                and all(c['coarse_change_objective'] is not None and c['coarse_change_objective'] < .01
                        and (c['runner_gap_objective'] is None or c['runner_gap_objective'] > 3 * c['runner_gap_sem_objective'])
                        for c in relevant_checks))
        elif str(high.get('status', '')).startswith('significant_'):
            if record.get('metrics'):
                record['coarse_diagnostic_metrics'] = record.pop('metrics')
            record.update(status=high['status'], error=high.get('error'), refinement_audit=high.get('audit'))
        methods[name] = record
    for record in methods.values():
        record['reporting_status'] = ('stable' if record.get('numerically_stable') else 'sensitivity_unresolved') if record.get('metrics') else record['status']
    return {'case': case, 'methods': methods}


def plot(records):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': .6}):
        fig, ax = plt.subplots(figsize=(3.9, 3.0), layout='constrained')
        matrix = np.array([[r['methods'][m].get('metrics', {}).get('clade_f1', np.nan) for r in records] for m in METHODS])
        cmap = LinearSegmentedColormap.from_list('support', ['#faf5ef', '#b9d9cc', '#167663'])
        cmap.set_bad('#e9ecef')
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(2), ['A', 'C']); ax.xaxis.tick_top()
        ax.set_yticks(range(len(METHODS)), LABELS); ax.tick_params(length=0, pad=8)
        for i, name in enumerate(METHODS):
            for j, result in enumerate(records):
                record = result['methods'][name]; value = matrix[i, j]
                bad = any(w in record['status'] for w in ('failure', 'failed', 'violation', 'invalid', 'exhausted', 'unresolved'))
                label = ('N/A' if bad else 'pending') if np.isnan(value) else f'{value:.2f}'
                if record.get('metrics') and not record.get('numerically_stable'):
                    label += '*'
                ax.text(j, i, label, ha='center', va='center', color='white' if value > .65 else '#293740',
                        fontsize=10, weight='bold' if value > .65 else 'normal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel('Module-reference F1', labelpad=10)
        fig.savefig(FIGURE, dpi=320, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def write_results(records):
    lines = ['| 方法 | A：F1 | C：F1 |', '|---|---:|---:|']
    for name, label in zip(METHODS, ('SPT', 'SURD + max-weight tree', 'Block ΦR')):
        cells = []
        for r in records:
            v = r['methods'][name]
            cells.append(f"{v['metrics']['clade_f1']:.4f}" + ('' if v.get('numerically_stable') else ' *') if v.get('metrics') else v['status'])
        lines.append('| ' + ' | '.join([label, *cells]) + ' |')
    lines += ['', '* 表示数值或近似敏感性尚未全部解决；不将这些分数当作确定排名。失败/缺项不填零。',
              '', '![三法比较](assets/kuramoto_hierarchy/spt_kuramoto_ac_selected.png)', '']
    for r in records:
        lines.append(f"### {r['case']} 的结构与核查")
        for name in METHODS:
            v = r['methods'][name]
            lines += ['', f"- **{name}**：`{v['reporting_status']}`；"]
            if v.get('metrics'):
                lines[-1] += f"根切分 `{v['metrics']['root_split']}`；clades `{v['metrics']['clades']}`；数值 F1 范围 `{v['numerical_f1_range']}`；评分结构/原始树一致 {v['scored_clade_agreement_count']}/4、{v['raw_tree_agreement_count']}/4。"
            elif v.get('error'):
                lines[-1] += v['error']
            if name == 'surd' and v.get('metrics'):
                lines += [f"  粒子/outer 原子最大变化 {v['maximum_particle_atom_change_nats']:.5g}/{v['maximum_outer_atom_change_nats']:.5g} nats；同档原子最大数值 SEM {v['maximum_atom_sem_nats']:.5g} nats；full-KL 均值 `{[a['full_posterior_gaussian_kl_mean_nats'] for a in v['approximation_audits']]}`。",
                          f"  与共同 EI 的诊断：`{v.get('common_ei_agreement')}`。"]
            if name == 'spt' and v.get('coarse_audits'):
                lines.append(f"  评分来自 {v['score_source']}；五份完整粗档审计的最低 raw Syn 为 {min(a['minimum_syn_nats'] for a in v['coarse_audits']):.6g} nats，容差负值数 `{[a['tolerance_negative_count'] for a in v['coarse_audits']]}`，显著 Syn/Ξ 违规均为零；closure={v['closure_error_nats']:.3g} nats。")
                if v.get('refinement_audit'):
                    a = v['refinement_audit']
                    lines.append(f"  高档最低 raw Syn/Ξ={a['minimum_raw_syn_nats']:.6g}/{a['minimum_xi_nats']:.6g} nats；容差负 Syn/Ξ 数={a['tolerance_negative_count']}/{a['tolerance_negative_xi_count']}。")
            if name in METHODS[2:] and v.get('audit'):
                lines.append(f"  原生 ΦR 非负审计 `{v['audit']}`；proposal 核查 `{v['proposal_audit']}`。")
    text = REPORT.read_text()
    start, end = '<!-- SELECTED_RESULTS -->', '<!-- SELECTED_RESULTS_END -->'
    REPORT.write_text(text.split(start)[0] + start + '\n\n' + '\n'.join(lines) + '\n\n' + end + text.split(end)[1])


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        records = [collect(c) for c in 'AC']
        save_json(FOLDER / 'selected_summary.json', {'records': records})
        plot(records)
        write_results(records)
    print(json.dumps({r['case']: {m: {'status': v['reporting_status'], 'f1': v.get('metrics', {}).get('clade_f1')} for m, v in r['methods'].items()} for r in records}))
