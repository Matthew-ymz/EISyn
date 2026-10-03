#!/usr/bin/env python3
"""Analyze COMPLETE frozen dense DMF caches; never analyze a selected subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from scripts.dmf_curve_shape import curve_shape, METHOD_COLORS, METHOD_LABELS
from scripts.dmf_dense_statistics import subject_landmarks, compare_cohort
from scripts.dmf_dense_timing import summarize_cost
from scripts.dmf_joint_readout import audit_nonnegative
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.report_sections import write_report_section

ROOT = Path(__file__).resolve().parents[1]


def load_complete(base):
    c = json.loads((base/'contract.json').read_text()); sha = digest(base/'contract.json')
    shape = (len(c['subject_ids']), len(c['seeds']), len(c['G']))
    values = {n: np.empty(shape) for n in ['xi', 'phi_r', 'wms', 'whole_ei', 'partial_ei_sum', 'rate', 'susceptibility']}
    timing_records = []
    audits = dict(condition_count=0, intervention_outside_count=0, diagnostic_outside_count=0,
        native_future_outside_count=0, native_boundary_hits=0, phi_floor_count=0,
        wms_floor_count=0, xi_numerical_negative_count=0, pilot_cache_reused_count=0,
        phi_numerical_zero_count=0, minimum_phi_raw_bits=None,
        minimum_natural_unique_timepoints=None, EI_closure_max_nats=0.)
    for pi, subject in enumerate(c['subject_ids']):
        for si, seed in enumerate(c['seeds']):
            for gi, g in enumerate(c['G']):
                p = base/'conditions'/f'{subject}_G{g:.2f}_seed{seed}.npz'
                if not p.exists():
                    raise FileNotFoundError(f'Dense analysis requires all {np.prod(shape)} conditions: {p.name} missing')
                with np.load(p) as a:
                    if str(a['contract_sha256']) != sha or str(a['subject']) != subject or float(a['G']) != g or int(a['seed']) != seed:
                        raise ValueError(f'Frozen condition mismatch: {p.name}')
                    metrics = json.loads(str(a['metrics_json'])); native = json.loads(str(a['native_metrics_json']))
                    timing_records.append(dict(subject=subject, G=g, seed=seed, **json.loads(str(a['timing_json']))))
                    for name in ['xi', 'whole_ei', 'partial_ei_sum']:
                        values[name][pi, si, gi] = metrics[name+'_nats']
                    values['phi_r'][pi, si, gi] = native['phi_r_bits']*np.log(2)
                    values['wms'][pi, si, gi] = native['wms_bits']*np.log(2)
                    values['rate'][pi, si, gi] = float(a['mean_rate_hz'])
                    values['susceptibility'][pi, si, gi] = float(a['susceptibility'])
                    ia = json.loads(str(a['intervention_diagnostics_json']))
                    fa = json.loads(str(a['native_future_diagnostics_json']))
                    na = json.loads(str(a['native_natural_json']))
                    pa = json.loads(str(a['native_phi_audit_json']))
                    wa = json.loads(str(a['native_wms_audit_json']))
                    xa = json.loads(str(a['xi_audit_json']))
                    audits['condition_count'] += 1
                    audits['intervention_outside_count'] += ia['outside_state_count']
                    audits['diagnostic_outside_count'] += int(float(a['boundary_fraction']) != 0.)
                    audits['native_future_outside_count'] += fa['outside_state_count']
                    audits['native_boundary_hits'] += na['boundary_hit_count']
                    audits['phi_floor_count'] += sum(pa[k] for k in ['source_eigenvalue_floor_count', 'target_eigenvalue_floor_count', 'joint_eigenvalue_floor_count'])
                    audits['wms_floor_count'] += wa['eigenvalue_floor_count']
                    audits['xi_numerical_negative_count'] += xa['tolerance_negative_count']
                    audits['pilot_cache_reused_count'] += int(a['pilot_cache_reused'])
                    audits['phi_numerical_zero_count'] += native['phi_r_numerical_zero_count']
                    raw = native['phi_r_minimum_raw_bits']
                    audits['minimum_phi_raw_bits'] = raw if audits['minimum_phi_raw_bits'] is None else min(raw, audits['minimum_phi_raw_bits'])
                    count = na['sampled_unique_timepoint_count']
                    audits['minimum_natural_unique_timepoints'] = count if audits['minimum_natural_unique_timepoints'] is None else min(count, audits['minimum_natural_unique_timepoints'])
                    audits['EI_closure_max_nats'] = max(audits['EI_closure_max_nats'], abs(metrics['whole_ei_nats']-metrics['partial_ei_sum_nats']-metrics['xi_nats']))
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ArithmeticError('Nonfinite full dense array')
    if any(audits[k] for k in ['intervention_outside_count', 'diagnostic_outside_count', 'native_future_outside_count']):
        raise ArithmeticError(f'State violations in complete dense caches: {audits}')
    if audits['EI_closure_max_nats'] > 1e-8:
        raise ArithmeticError('Complete EI decomposition does not close')
    audits['xi_nonnegative_audit'] = audit_nonnegative(values['xi'].ravel())
    return c, values, audits, timing_records


def summarize_shapes(rows):
    result = {}
    for name in ('xi', 'phi_r', 'wms'):
        all_shapes = [r['shape'][name] for r in rows]
        shapes = [r for r in all_shapes if not r['flat']]
        row = dict(subject_count=len(rows), nonflat_count=len(shapes), flat_count=len(rows)-len(shapes),
            weakly_unimodal_count=sum(r['weakly_unimodal'] for r in shapes),
            interior_weakly_unimodal_count=sum(r['weakly_unimodal'] and r['peak_interior'] for r in shapes),
            boundary_extremum_count=sum(not r['peak_interior'] for r in shapes),
            multiple_prominent_peaks_count=sum(r['interior_peak_counts']['0.05'] > 1 for r in shapes))
        for key in ('unimodality_violation', 'post_extremum_rebound_fraction', 'slope_total_variation'):
            vals = np.array([r[key] for r in shapes])
            row[key] = dict(mean=float(vals.mean()), median=float(np.median(vals)), maximum=float(vals.max())) if len(vals) else None
        result[name] = row
    return result


def analyze(c, values):
    rows = []; seed_rows = [[] for _ in c['seeds']]
    for pi, subject in enumerate(c['subject_ids'][:-1]):
        mean = {n: values[n][pi].mean(0) for n in ('xi', 'phi_r', 'wms')}
        row = subject_landmarks(c['G'], values['rate'][pi].mean(0), mean)
        row['subject'] = subject
        row['shape'] = {n: curve_shape(c['G'], v, -1 if n == 'wms' else 1) for n, v in mean.items()}
        rows.append(row)
        for si in range(len(c['seeds'])):
            r = subject_landmarks(c['G'], values['rate'][pi, si], {n: values[n][pi, si] for n in mean})
            r['shape'] = {n: curve_shape(c['G'], values[n][pi, si], -1 if n == 'wms' else 1) for n in mean}
            r['subject'] = subject; seed_rows[si].append(r)
    holdout = [r for r in rows if r['subject'] in c['holdout_subject_ids']]
    development = [r for r in rows if r['subject'] in c['development_subject_ids']]
    return dict(status='complete', subject_rows=rows, primary_holdout85=compare_cohort(holdout),
        all93=compare_cohort(rows), development8=compare_cohort(development),
        shape_summary=dict(primary_holdout85=summarize_shapes(holdout), all93=summarize_shapes(rows), development8=summarize_shapes(development)),
        per_seed_subject_rows=[dict(seed=seed, subjects=rs) for seed, rs in zip(c['seeds'], seed_rows)],
        per_seed_primary_sensitivity=[dict(seed=seed, **compare_cohort([r for r in rs if r['subject'] in c['holdout_subject_ids']])) for seed, rs in zip(c['seeds'], seed_rows)],
        success=compare_cohort(holdout)['both_comparisons_significant'],
        false_positive_rate=None, no_confirmed_absence_controls=True,
        protocol=c['analysis'], manuscript=c['manuscript'])


def draw(c, v, s, base, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(base/'inputs.npz') as a:
        rho = a['spectral_radius'][:-1]
    colors = plt.cm.viridis((rho-rho.min())/(rho.max()-rho.min()))
    g = np.asarray(c['G'])
    for names, filename, ylabel in [(['xi', 'phi_r', 'wms'], 'metric_curves.png', 'Information (nats)'),
                                    (['whole_ei', 'partial_ei_sum', 'xi'], 'ei_components.png', 'Information (nats)'),
                                    (['rate'], 'order_parameter_all93.png', 'Mean E firing rate (Hz)')]:
        fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 3.8), layout='constrained', squeeze=False)
        titles = dict(xi='Interventional $\\Xi$', phi_r='BOLD-like pairwise $\\Phi^R$', wms='Signed source WMS',
                      whole_ei='Whole EI', partial_ei_sum='Sum of partial EI', rate='Independent order curve')
        for ni, name in enumerate(names):
            ax = axes[0, ni]
            for i in range(93):
                ax.plot(g, v[name][i].mean(0), color=colors[i], alpha=.42, lw=.65)
            ax.plot(g, v[name][-1].mean(0), color='#222222', ls='--', lw=1.5, label='Mean SC (93)')
            ax.set(xlabel='$G$', ylabel=ylabel, title=titles[name])
            ax.text(-.1, 1.03, 'abc'[ni], transform=ax.transAxes, fontweight='bold')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', frameon=False)
        norm = plt.Normalize(rho.min(), rho.max())
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes.ravel().tolist(), label='Native SC spectral radius', shrink=.8)
        fig.savefig(output/filename, dpi=240, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.), layout='constrained')
    for ni, name in enumerate(('xi', 'phi_r', 'wms')):
        rates = [s[k]['methods'].get(name, {}).get('hit_rate', np.nan) for k in ['all93', 'primary_holdout85']]
        axes[0].bar(np.arange(2)+(ni-1)*.23, rates, width=.21, color=METHOD_COLORS[name], label=METHOD_LABELS[name])
    axes[0].set(xticks=[0, 1], xticklabels=['All 93', 'Unseen 85 (primary)'], ylabel='Hits / independently located subjects', ylim=(0, 1))
    comp = list(s['primary_holdout85']['comparisons'].items())
    if not comp:
        axes[1].text(.5, .5, 'No independently located primary subjects', transform=axes[1].transAxes, ha='center')
    for i, (name, row) in enumerate(comp):
        low, high = row['difference_ci95_percentile']; value = row['rate_difference']
        axes[1].errorbar(i, value, yerr=[[max(0., value-low)], [max(0., high-value)]], fmt='o', color=METHOD_COLORS[name], capsize=4)
        axes[1].text(i, 1.02, f"Holm p={row['p_holm_two']:.3g}", transform=axes[1].get_xaxis_transform(), ha='center', fontsize=8)
    axes[1].axhline(0, color='#aaa', lw=.7)
    axes[1].set(xticks=[0, 1], xticklabels=['Ξ − ΦR', 'Ξ − WMS'], ylabel='Paired hit-rate difference (primary 85)', xlim=(-.5, 1.5))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False)
    fig.savefig(output/'transition_correspondence.png', dpi=240, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), layout='constrained')
    for ax, key, label in zip(axes, ['unimodality_violation', 'post_extremum_rebound_fraction', 'slope_total_variation'],
                             ['Extra wrong-way movement U', 'Post-extremum rebound / amplitude', 'Slope roughness Q']):
        for i, name in enumerate(('xi', 'phi_r', 'wms')):
            y = np.array([r['shape'][name][key] for r in s['subject_rows'] if not r['shape'][name]['flat']])
            if not len(y):
                continue
            ax.boxplot([y], positions=[i], widths=.38, showfliers=False,
                       medianprops={'color': '#111111'}, whiskerprops={'color': '#555555'})
            ax.scatter(i+.16*np.sin(np.arange(len(y))*2.4), y, s=8, alpha=.5, color=METHOD_COLORS[name], linewidths=0)
        ax.set(xticks=[0, 1, 2], xticklabels=['Ξ', 'ΦR', 'WMS'], ylabel=label, xlim=(-.5, 2.5))
    fig.savefig(output/'curve_shape_summary.png', dpi=240, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9., 3.8), layout='constrained')
    for ax, suffix, label in zip(axes, ['estimator', 'pipeline'], ['Audited estimator time (s)', 'Required pipeline time (s)']):
        for i, name in enumerate(('xi', 'phi_r', 'wms')):
            row = s['cost']['groups'][name+'_'+suffix]['wall_seconds']
            # Raw spread is shown separately from the mean; it is not a CI.
            ax.vlines(i, row['q10'], row['q90'], color=METHOD_COLORS[name], lw=2)
            ax.scatter(i, row['mean'], color=METHOD_COLORS[name], s=35, zorder=3)
        ax.set(xticks=[0, 1, 2], xticklabels=['Ξ', 'ΦR', 'WMS'], ylabel=label, xlim=(-.5, 2.5))
        ax.set_ylim(bottom=0)
    fig.savefig(output/'computation_cost.png', dpi=240, bbox_inches='tight'); plt.close(fig)
    pages = []
    for start in range(0, 94, 8):
        indices = list(range(start, min(start+8, 94))); n = len(indices)
        columns, rows = (4, 2) if n == 8 else (3, 2)
        fig = plt.figure(figsize=(3.2*columns, 4.2*rows), layout='constrained')
        cells = fig.subfigures(rows, columns, wspace=.03, hspace=.08)
        for pi, cell in zip(indices, cells.ravel()):
            axes = cell.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
            r = v['rate'][pi]; m, sd = r.mean(0), r.std(0, ddof=1)
            axes[0].plot(g, m, color='#222222', lw=1.2)
            axes[0].fill_between(g, m-sd, m+sd, color='#222222', alpha=.15)
            axes[0].set(title=c['subject_ids'][pi], ylabel='Mean E rate (Hz)')
            landmark = subject_landmarks(g, m, {n: v[n][pi].mean(0) for n in ('xi', 'phi_r', 'wms')})
            t = landmark['transition']
            for ax in axes:
                ax.axvspan(*t['interval'], color='#BFC3C8', alpha=.25, hatch=None if t['located'] else '///', lw=0)
            for name in ('xi', 'phi_r', 'wms'):
                mean = v[name][pi].mean(0); spread = v[name][pi].std(0, ddof=1); scale = mean.std()
                if scale <= 1e-12*max(1., float(np.abs(mean).max())):
                    continue
                z = (mean-mean.mean())/scale
                axes[1].plot(g, z, color=METHOD_COLORS[name], lw=1.1)
                axes[1].fill_between(g, z-spread/scale, z+spread/scale, color=METHOD_COLORS[name], alpha=.10)
            axes[1].set(xlabel='$G$', ylabel='Metric shape (z score)')
        handles = [Line2D([], [], color=METHOD_COLORS[n], label=METHOD_LABELS[n]) for n in ('xi', 'phi_r', 'wms')]
        handles += [Patch(facecolor='#BFC3C8', alpha=.3, label='Independent rate-turn interval')]
        fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False, fontsize=8)
        name = f'subject_order_curves_{start//8+1:02d}.png'; pages.append(name)
        fig.savefig(output/name, dpi=240, bbox_inches='tight'); plt.close(fig)
    return pages


def write_report(c, s, path, pages):
    rows, tests, shapes = [], [], []
    for cohort, label in [('primary_holdout85', '新增85人（主要检验）'), ('all93', '全部93人（补充）'), ('development8', '已看过8人（开发）')]:
        data = s[cohort]
        if not data['eligible_count']:
            rows.append(f'| {label} | 全部指标 | 0/0 | 无法估计 | 无法估计 |')
        for name, row in s['shape_summary'][cohort].items():
            vals = [row[key] for key in ('unimodality_violation', 'post_extremum_rebound_fraction', 'slope_total_variation')]
            text = ['平坦' if v is None else f"{v['mean']:.4g}" for v in vals]
            shapes.append(f"| {label} | {name} | {row['weakly_unimodal_count']}/{row['nonflat_count']} | {' | '.join(text)} | {row['multiple_prominent_peaks_count']} |")
        for n, row in data['methods'].items():
            distance = row['mean_distance_to_interval_G']
            rows.append(f"| {label} | {n} | {row['hit_count']}/{row['denominator']} | {row['hit_rate']:.3f} | {distance if distance is not None else '未定义'} |")
        for n, row in data['comparisons'].items():
            low, high = row['difference_ci95_percentile']
            tests.append(f"| {label} | Ξ vs {n} | {row['xi_only']} / {row['comparator_only']} | {row['rate_difference']:.3f} | [{low:.3f}, {high:.3f}] | {row['p_holm_two']:.6g} |")
    outcome = ('在主要检验集上，Ξ对两项基线的命中率差均为正，且Holm校正后均显著。' if s['success'] else
               '主要检验未同时支持Ξ显著优于两项基线；完整结果保留，未改变扫描、转折或峰值规则。')
    images = '\n'.join(f'- [逐人对照第{i+1}页](assets/dmf_subject_dense/{name})' for i, name in enumerate(pages))
    cost_rows = []
    for name in ('xi', 'phi_r', 'wms'):
        e = s['cost']['groups'][name+'_estimator']; p = s['cost']['groups'][name+'_pipeline']
        cost_rows.append(f"| {name} | {e['wall_seconds']['mean']:.5g} | {e['cpu_seconds']['mean']:.5g} | {p['wall_seconds']['mean']:.5g} | {p['cpu_seconds']['mean']:.5g} |")
    shared = s['cost']['groups']['natural_shared']['wall_seconds']['mean']
    diagnostic = s['cost']['groups']['independent_diagnostic']['wall_seconds']['mean']
    write_report_section(path, 'dmf-subject-dense', f'''# 93人DMF细扫描：指标极值与独立序参量转折

{outcome}

全部93个原生SC及一个93人平均SC参照，G从0到{c['G'][-1]:g}、步长{c['G'][1]-c['G'][0]:g}，每点seed {c['seeds']}，共{s['audit']['condition_count']}条件。平均SC不进入统计。已看过的8人单列；新增85人为主要检验集。结果解释条件于固定JFIC、SC队列与同一3seed协议，不是独立生物相变真值或公式单因素比较。

![三项原始指标曲线](assets/dmf_subject_dense/metric_curves.png)

每条细线为一人的3seed均值，颜色对应原生SC谱半径；黑色虚线为平均SC独立模拟。未归一化SC、平滑或按指标峰位对齐。WMS保留原始符号。

![独立序参量曲线](assets/dmf_subject_dense/order_parameter_all93.png)

独立5s模拟最后2s的100ROI平均E发放率。每人的最大正斜率区间按原规则定位，边界最大斜率列为无法定位，而非“无相变”。

![转折命中及配对差异](assets/dmf_subject_dense/transition_correspondence.png)

| 队列 | 指标 | 命中/可定位 | 命中率 | 到转折区间的平均距离G |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

只按独立序参量决定可定位集合，三项指标共享同一分母。Ξ/ΦR取全局最大值、WMS取全局最小值；内部极值落在区间才算命中，指标边界极值和平坦曲线算未命中。区间含端点；相同极值取首格点，未按离转折最近的位置选峰。

| 队列 | 比较 | Ξ单独命中 / 基线单独命中 | 配对命中率差 | 95%配对bootstrap CI | Holm p |
|---|---|---:|---:|---:|---:|
{chr(10).join(tests)}

单侧精确McNemar在不一致被试上用[二项检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)实现，备择是Ξ命中率更高；两项比较作[Holm校正](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)，阈值0.05。CI为按被试共同重采样10000次的双侧95%百分位区间，未作多重校正；主要成功判据以校正p值和正效应为准，不以CI筛选。seed/G/4950 ROI对不是额外独立被试。全部93人与开发8人的检验是补充结果，不替代主要85人检验。

## 逐人对照与形状

{images}

每页上部为原始Hz序参量、下部为同人各指标按跨G均值/SD转换的形状；带为3seed SD，不是CI。灰带为独立转折候选，斜线为扫描边界。没有翻转WMS或插值。完整逐人极值、距离、U/Q、prominence敏感性和逐seed端点保存在[分析汇总](../../results/dmf_schaefer100/subject_curves_93_dense/summary.json)。形状描述继续采用[开发阶段固定定义](brain.md#dmf-subject-curves)：U测额外反向变化、Q测归一化相邻斜率总变差；陡峭单峰也可有较高Q。

![额外回摆、峰后回摆和粗糙度](assets/dmf_subject_dense/curve_shape_summary.png)

散点是93个人的3seed平均曲线，箱线表示四分位分布；平坦曲线的形状分数未定义并单列。WMS只在形状诊断中取负，以比较单谷对应的峰前/峰后方向。

| 队列 | 指标 | U≈0 / 非平坦人数 | 平均U | 平均峰后回摆 | 平均Q | >1个明显内部峰人数 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(shapes)}

U为全局峰前下降量加峰后回升量，除以全曲线幅度；峰后回摆单独保留。U≈0的相对容差为10⁻¹⁰，并不要求出现内部峰，边界单调曲线也可U=0。明显峰采用幅度5%的prominence，同时保存0%、1%、5%敏感性；该阈值不改变主要命中判断。Q为G和纵轴幅度归一化后相邻斜率的总变差，没有事后正确/错误阈值。均值和逐seed形状均保留，平均可能掩盖seed波动。

## 计算成本与复杂度

![平均计算成本](assets/dmf_subject_dense/computation_cost.png)

点为新计算条件的平均墙钟时间，竖线为条件耗时的10–90百分位，不是置信区间。每个条件是一人、一个G、一个seed；4进程并行、每进程BLAS单线程。共{s['cost']['condition_count']}个个体条件、{s['cost']['subject_count']}人进入计时，排除{s['cost']['excluded_cached_individual_conditions']}个复用条件和平均SC。缓存读取不当作估计器耗时。下表单位秒，同时记录每进程CPU时间以区分并行竞争。

| 指标 | 估计均值：墙钟 | 估计均值：CPU | 必需流程均值：墙钟 | 必需流程均值：CPU |
|---|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

估计耗时包含各自原生数值审计，Ξ还包括affine-TM拟合与密度查询；ΦR含pairwise MMI与特征值审计；WMS含标准化、原Gaussian拟合和审计。完整流程为估计加必需样本准备：Ξ的干预未来批次、ΦR的自然轨迹/BOLD转换、WMS的自然轨迹/未来批次。两基线共享的自然轨迹平均{shared:.5g}秒，分别列入依赖成本，因此不能将两项完整流程均值相加来估计总运行时间；独立序参量诊断平均{diagnostic:.5g}秒单列，不归入某项指标。初始化、共享均匀源生成、输入/缓存读写和最终绘图不进入这些计时。开发8人的原粗格点复用，故计时覆盖不同；汇总同时给出等被试权重均值、条件中位数、SD及10–90百分位。此处均值为条件权重，固定100ROI维度的实测不用于拟合渐近指数。

复杂度按当前实现分析。令R为ROI数，d=2R为源/未来状态维数，N为未来样本数，T为自然轨迹长度，H为未来积分步数。在相同源/未来维数、稠密SC下，Ξ的affine-TM拟合与查询为O(Nd²+d³)；原生WMS为O(Nd²+d³)，其标量循环复用一个逆矩阵，采用秩一行列式更新，并非每标量重做完整逆。ΦR为O(TR²+R²)，来自滞后协方差及全部ROI对的固定小矩阵运算。Ξ/WMS各自的未来批次为O(HNR²)，共享自然轨迹为O(TR²)、BOLD转换为O(TR)，独立诊断为O(LR²)、L=5000。未来批次与密度的主要工作内存为O(Nd+d²)，自然轨迹另占O(TR)。这些是当前代码的主要阶数，常数、稀疏实现和不同估计协议会影响实际成本。

## 单独的EI分解

![整体EI、部分EI之和及差值](assets/dmf_subject_dense/ei_components.png)

这三项按同一200标量源、同一完整200维未来、同一affine-TM密度计算，Ξ=整体EI−部分EI之和。whole EI不进入基线比较。

## 估计与边界

沿用2048个[0.3,0.7]因子化干预、300ms未来、固定JFIC及共同affine-TM Gaussian近似。Ξ非负容差10⁻⁸ nats，未投影；数值负值计数{s['audit']['xi_numerical_negative_count']}，全量分解最大闭合误差{s['audit']['EI_closure_max_nats']:.6g} nats。ΦR保留1.5s自然轨迹的BOLD-like转换、1ms延迟及原生特征值下限；WMS保留自然完整E/I状态、有放回抽2048次的相关源先验及原ridge。ΦR/WMS轨迹短与正则化限制解释，增加被试和G点不能自动解决估计器偏差。原生ΦR下限触发总数{s['audit']['phi_floor_count']}；WMS下限触发{s['audit']['wms_floor_count']}；最少不同自然采样时间点{s['audit']['minimum_natural_unique_timepoints']}。没有独立确认无转变对照，虚假峰率未定义。失败、边界和不利结果均保留，细扫描不保证Ξ获胜。

本任务重新读取Zotero父条目P6UJCVG8、唯一正文附件DXGC7JEA，*Emergent hierarchical organization of causal interactions in complex systems*，19页，Brain/Fig.2及Methods式（5）、（7）、（8）。稿件没有明确版本日期，补充附录不可用。正文把两观测基线写为BOLD-like，仓库原生WMS实际是完整E/I状态；这里保留原生协议并记录差别。

[固定执行与分析协议](../log/dmf_subject_dense_protocol.md)。计算支持断点复用，原8人结果保留在独立目录。
''')


def main(base):
    base = Path(base)
    c, v, audit, timing = load_complete(base)
    s = analyze(c, v); s['audit'] = audit; s['contract_sha256'] = digest(base/'contract.json')
    s['cost'] = summarize_cost(timing)
    output = ROOT/'docs/reports/assets/dmf_subject_dense'
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False}):
        pages = draw(c, v, s, base, output)
    atomic_json(base/'summary.json', s)
    report = ROOT/'docs/reports/brain.md'; write_report(c, s, report, pages)
    atomic_json(base/'completed.json', dict(status='complete', condition_count=audit['condition_count'],
        primary_superiority_supported=s['success'], contract_sha256=s['contract_sha256'],
        report_sha256=digest(report), summary_sha256=digest(base/'summary.json'),
        figures_sha256={p.name: digest(p) for p in output.glob('*.png')}))
    print(json.dumps({'analysis_complete': True, 'primary85': s['primary_holdout85'], 'all93': s['all93']}, indent=2), flush=True)
