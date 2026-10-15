"""Combine the five retained methods; the historical entry name is preserved."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_spt_kuramoto_ac_selected import collect, read, stability
from scripts.spt_kuramoto_finite_tm import save_json
from scripts.spt_kuramoto_natural_baselines import information_audit
from scripts.summarize_spt_glv_a_comparison import provisional_surd_result

AC = ROOT / 'results/spt_local_kuramoto_tm/ac_comparison'
GLV = ROOT / 'results/spt_local_glv_tm/a_comparison'
BASE_METHODS = ('spt', 'surd', 'block_phi_r')
NATURAL_METHODS = ('wms_natural', 'shap_profiles')
METHODS = BASE_METHODS + NATURAL_METHODS
LABELS = ('SPT', 'SURD +\nmax-weight tree', r'Block $\Phi_R$',
          'WMS tree\n(natural sampling)', 'SHAP output-\nprofile tree')
REPORT = ROOT / 'docs/reports/spt_glv_kuramoto_five_methods.md'
FIGURE = ROOT / 'docs/reports/assets/kuramoto_hierarchy/spt_glv_kuramoto_five_methods.png'


def natural_record(record, case, fallback_status):
    result = dict(record) if record else {'status': fallback_status}
    native_passed = result.get('status', '').endswith('screen_passed')
    if result.get('metrics'):
        reps = result.get('integration_replicates', result.get('numerical_replicate_metrics', []))
        native_passed &= len(reps) == 4
        if len(reps) == 4:
            result.update(stability(result['metrics'], reps, case))
            native_passed &= result['scored_clade_agreement_count'] == 4
    result['numerically_stable'] = bool(native_passed and result.get('metrics'))
    result['reporting_status'] = ('native_screen_passed' if result['numerically_stable'] else 'sensitivity_unresolved') if result.get('metrics') else result.get('status', 'not_computed')
    if result.get('provisional') and result.get('metrics'):
        result.update(reporting_status='provisional_completed_full_tables', numerically_stable=False)
    return result


def score_marker(record):
    return '†' if record.get('provisional') else '' if record.get('numerically_stable') else '*'


def missing_score_label(record):
    status = record.get('reporting_status', '')
    terminal = any(w in status for w in ('failed', 'failure', 'invalid', 'violation', 'unresolved', 'exhausted', 'cancelled'))
    return 'N/A' if terminal else 'pending'


def records():
    four = read(GLV / 'comparison_1024.json')
    native = four
    natural = read(GLV / 'natural_baselines.json')
    budget = read(GLV / 'budget.json')
    job_terminal = native.get('status') in ('cancelled', 'budget_exhausted', 'numerical_failure',
                                           'comparison_complete', 'complete_with_unresolved_numerics')
    budget_ended = job_terminal and time.time() >= budget.get('deadline_epoch', float('inf'))
    glv_methods = {}
    for name in METHODS:
        original = {'surd': 'surd_maximum_weight',
                    'normalized_block_phi_r': 'block_phi_r_normalized'}.get(name, name)
        source = four if name in BASE_METHODS else natural
        raw = source.get('methods', {}).get(original, native.get('methods', {}).get(original, {}))
        if (name in BASE_METHODS and budget_ended and not raw.get('metrics')
                and raw.get('status', 'not_computed') in ('running', 'not_yet_evaluated', 'not_computed', 'not_started', 'pending')):
            raw = {'status': 'not_evaluated_budget_exhausted', 'metrics': None,
                   'cached_information_query_count': native.get('query_count'),
                   'native_job_status': native.get('status'),
                   'error': 'Original gLV budget ended before this method had a complete valid result'}
        glv_methods[name] = natural_record(raw, 'A', source.get('status', 'not_computed'))
        glv_methods[name]['source_path'] = str((GLV / ('comparison_1024.json' if name in BASE_METHODS
                                                     else 'natural_baselines.json')).relative_to(ROOT))
    if glv_methods['surd'].get('status') != 'screen_passed':
        try:
            provisional = provisional_surd_result(read(GLV / 'surd_full.json'))
        except (ArithmeticError, ValueError, KeyError) as error:
            provisional = {'status': 'completed_surd_table_audit_failure', 'metrics': None, 'error': str(error)}
        if provisional is not None:
            glv_methods['surd'] = natural_record(provisional, 'A', 'not_computed')
            glv_methods['surd']['source_path'] = str((GLV / 'surd_full.json').relative_to(ROOT))
            glv_methods['surd'].update(native_job_status=native.get('status'),
                                      native_job_budget_ended=budget_ended,
                                      native_job_error=native.get('error'))
    rows = natural.get('wms_replicates', [])
    if len(rows) == 4 and all(len(r.get('information_nats', {})) == 511 for r in rows):
        audit = information_audit(np.array([[r['information_nats'][str(m)] for m in range(1, 512)] for r in rows]))
        record = glv_methods['wms_natural']
        record['information_audit'] = audit
        if not audit['passed']:
            if record.get('metrics'):
                record['diagnostic_metrics'] = record.pop('metrics')
            record.update(reporting_status='information_audit_failure', numerically_stable=False)
    output = [{'case': 'gLV A', 'methods': glv_methods, 'native_status': native.get('status'),
               'provenance': 'results/spt_local_glv_tm/a_comparison/comparison_1024.json'}]
    for case in 'AC':
        current = collect(case)
        raw = read(AC / case / 'natural_baselines.json')
        for name in NATURAL_METHODS:
            current['methods'][name] = natural_record(raw.get('methods', {}).get(name, {}), case, raw.get('status', 'not_computed'))
        current.update(case=f'Kuramoto {case}', provenance=f'results/spt_local_kuramoto_tm/ac_comparison/{case}/')
        output.append(current)
    return output


def render(data):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
        fig, ax = plt.subplots(figsize=(5.5, 4.7), layout='constrained')
        values = np.array([[r['methods'][m].get('metrics', {}).get('clade_f1', np.nan)
                            if r['methods'][m].get('metrics') else np.nan for r in data] for m in METHODS])
        cmap = LinearSegmentedColormap.from_list('module_f1', ['#faf5ef', '#b9d9cc', '#167663'])
        cmap.set_bad('#e9ecef')
        ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='equal')
        ax.set_xticks(range(3), ['gLV\nA (original)', 'Kuramoto\nA (simpl.)', 'Kuramoto\nC (simpl.)']); ax.xaxis.tick_top()
        ax.set_yticks(range(len(METHODS)), LABELS); ax.tick_params(length=0, pad=9)
        for i, name in enumerate(METHODS):
            for j, row in enumerate(data):
                record = row['methods'][name]; value = values[i, j]
                if np.isnan(value):
                    label = missing_score_label(record)
                else:
                    label = f'{value:.2f}' + score_marker(record)
                ax.text(j, i, label, ha='center', va='center', fontsize=10,
                        color='white' if value > .65 else '#293740', weight='bold' if value > .65 else 'normal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel('Module-reference F1', labelpad=11)
        ax.text(.5, -.105, '* Sensitivity unresolved; N/A: not evaluable\n† Provisional complete-table result; precision unconfirmed',
                transform=ax.transAxes, ha='center', va='top', fontsize=7)
        fig.savefig(FIGURE, dpi=320, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def write_report(data):
    rows = ['| 方法 | gLV A（原网络） | Kuramoto A（简化） | Kuramoto C（简化） |', '|---|---:|---:|---:|']
    for name, label in zip(METHODS, ('SPT', 'SURD + max-weight tree', 'Block ΦR',
                                    'WMS tree (natural sampling)', 'SHAP output-profile tree')):
        cells = []
        for case in data:
            r = case['methods'][name]
            cells.append(f"{r['metrics']['clade_f1']:.4f}" + score_marker(r)
                         if r.get('metrics') else 'N/A' if missing_score_label(r) == 'N/A' else '待完成')
        rows.append('| ' + ' | '.join([label, *cells]) + ' |')
    notes = []
    for case in data:
        notes.append(f"### {case['case']}")
        for name in METHODS:
            r = case['methods'][name]
            notes.append(f"- {name}：`{r['reporting_status']}`；来源 `{r.get('source_path', case['provenance'])}`。")
            if r.get('metrics'):
                notes[-1] += f"根切分 `{r['metrics']['root_split']}`；clades `{r['metrics']['clades']}`。"
            if r.get('numerical_f1_range') is not None:
                notes[-1] += (f"四次积分/背景子样本的 F1 范围 `{r['numerical_f1_range']}`；"
                              f"评分结构一致 {r.get('scored_clade_agreement_count', 0)}/4，"
                              f"原始树一致 {r.get('raw_tree_agreement_count', 0)}/4。")
            if r.get('failed_checks'):
                notes[-1] += f"未通过项 `{r['failed_checks']}`。"
            if r.get('error'):
                notes[-1] += f"错误：{r['error']}。"
            if r.get('reason'):
                notes[-1] += f"限制：{r['reason']}。"
            if r.get('available_information_queries') is not None:
                notes[-1] += (f"通用 MI 缓存 {r['available_information_queries']} 项；"
                              f"其中未分辨 {r.get('unresolved_information_queries', '未记录')} 项。")
            if r.get('provisional'):
                notes[-1] += (f"已完成完整表 {r['completed_integration_replicates']}/4，"
                              f"采用所有已完成 scramble `{r['selected_scrambles']}` 的平均，"
                              f"精度档 `{r['grid']}`；不按 F1 挑选重复。原生阶段 `{r['source_status']}`。")
                if r.get('source_error'):
                    notes[-1] += f"原生终止原因：{r['source_error']}。"
                if r.get('native_job_status'):
                    notes[-1] += f"主作业状态 `{r['native_job_status']}`。"
                if r.get('native_job_budget_ended'):
                    notes[-1] += "原预算已结束、主作业已停止。"
                    if r.get('source_status') == 'running':
                        notes[-1] += "全表检查点中的 running 头部是保留的旧状态，不代表仍在计算。"
            if r.get('cached_information_query_count') is not None:
                notes[-1] += f"已保留通用 MI 查询 {r['cached_information_query_count']} 项，未构成该方法完整结果。"
            if name == 'wms_natural' and r.get('maximum_information_sem_nats') is not None:
                notes.append(f"  最大 MI 数值 SEM={r['maximum_information_sem_nats']:.6g} nats；"
                    f"内层 64→256 最大变化={r['maximum_inner_change_nats']:.6g} nats；"
                    f"最小 ESS={r['minimum_ess']:.6g}；最大归一化权重={r['maximum_weight']:.6g}。"
                    "门槛依次为 <0.01、<0.01、≥10、<0.25；未通过时不将诊断候选树分数作为有效 F1。")
                if r.get('information_audit'):
                    a = r['information_audit']
                    notes.append(f"  MI 非负及源包含单调审计：`{a}`。WMS 有符号残差的负值合法，不据此判为 Syn 非负违规。")
        notes.append('')
    text = REPORT.read_text(); a = '<!-- SIX_RESULTS -->'; b = '<!-- SIX_RESULTS_END -->'
    body = '\n'.join(rows) + '\n\n* 表示数值或结构敏感性尚未完全解决；† 为已完成完整表的暂定结果，预定完整精度检查尚未通过或结束；N/A/未评价与实测零分区分。\n\n![三例五法](assets/kuramoto_hierarchy/spt_glv_kuramoto_five_methods.png)\n\n' + '\n'.join(notes)
    REPORT.write_text(text.split(a)[0] + a + '\n\n' + body + '\n\n' + b + text.split(b)[1])


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        data = records()
        save_json(ROOT / 'results/spt_glv_kuramoto_five_methods.json', {'records': data, 'generated_epoch_seconds': time.time()})
        render(data); write_report(data)
    print(json.dumps({r['case']: {m: v['reporting_status'] for m, v in r['methods'].items()} for r in data}))
