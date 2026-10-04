#!/usr/bin/env python3
"""Present the user-fixed dense grid/window using existing, frozen outputs only.

Four aligned raw-curve panels contain all 93 subjects and the separately simulated
mean SC. Hit bars use the existing k=2 subject classifications; no new inference.
Whole EI and the sum of partial EI stay in a separate decomposition figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np

from scripts.dmf_curve_shape import METHOD_COLORS, METHOD_LABELS
from scripts.dmf_dense_statistics import subject_landmarks
from scripts.run_dmf_subject_consistency import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
OUTPUT = ROOT / 'docs/reports/assets/dmf_subject_dense'
REPORT = ROOT / 'docs/reports/brain.md'
NAMES = ('xi', 'phi_r', 'wms')
TITLE = {'rate': 'Independent order parameter', 'xi': r'Integrated EI $\Xi$',
         'phi_r': r'BOLD-like pairwise $\Phi^R$', 'wms': 'Signed source WMS',
         'whole_ei': 'Whole EI', 'partial_ei_sum': 'Sum of partial EI'}


def read_inputs(config):
    for name, expected in config['source_sha256'].items():
        if digest(BASE / name) != expected:
            raise ValueError(f'Protected source changed: {name}')
    contract = json.loads((BASE / 'contract.json').read_text())
    frozen = json.loads((BASE / 'summary.json').read_text())
    windows = json.loads((BASE / 'window_summary.json').read_text())
    assert config['G_step'] == .1 and config['k_each_side'] == 2
    with np.load(BASE / 'analysis_curves.npz') as a:
        ids = a['subject_ids'].tolist()
        g = a['G'].copy()
        assert ids == contract['subject_ids'] and ids[-1] == 'group_mean_93'
        assert np.array_equal(g, np.round(np.arange(41) * .1, 1))
        assert str(a['contract_sha256']) == digest(BASE / 'contract.json')
        values = {n: a[n].mean(axis=1) for n in (*NAMES, 'rate', 'whole_ei', 'partial_ei_sum')}
    assert all(y.shape == (94, 41) and np.isfinite(y).all() for y in values.values())
    tolerance = contract['syn_tolerance_nats']
    minimum = float(values['xi'].min())
    if minimum < -tolerance:
        count = int(np.count_nonzero(values['xi'] < -tolerance))
        raise ArithmeticError(f'Xi nonnegativity violation: min={minimum}, tolerance={tolerance}, count={count}')
    closure = float(np.abs(values['whole_ei'] - values['partial_ei_sum'] - values['xi']).max())
    if closure > 1e-8:
        raise ArithmeticError(f'EI closure exceeds 1e-8 nats: {closure}')
    for i, old in enumerate(frozen['subject_rows']):
        assert ids[i] == old['subject']
        recovered = subject_landmarks(g, values['rate'][i], {n: values[n][i] for n in NAMES})
        assert recovered['transition'] == old['transition']
        assert recovered['methods'] == old['methods']
    scenario = next(s for s in windows['scenarios'] if s['k_each_side'] == config['k_each_side'])
    assert len(scenario['subject_rows']) == 93
    assert scenario['cohorts']['primary85']['subject_count'] == 85
    assert scenario['cohorts']['all93']['subject_count'] == 93
    assert scenario['cohorts']['all93']['clipped_subject_count'] == 0
    for row in scenario['subject_rows']:
        lo, hi = row['independent_transition']['interval']
        assert np.allclose(row['window']['interval'], [max(0, lo-.2), min(4, hi+.2)], atol=1e-12, rtol=0)
        assert abs(row['window']['actual_width_G'] - .5) <= 1e-12
    with np.load(BASE / 'inputs.npz') as a:
        assert a['subject_ids'].tolist() == ids
        rho = a['spectral_radius'][:-1].copy()
    assert rho.shape == (93,) and np.isfinite(rho).all()
    return contract, g, values, rho, scenario, dict(
        all_93_landmarks_match_frozen=True, cache_contract_verified=True,
        mean_SC_is_separate_simulation=True, subject_count=93, mean_SC_reference_count=1,
        seed_count=3, G_node_count=41, normalization='none', smoothing='none',
        WMS_sign='unchanged', EI_closure_max_nats=closure, xi_minimum_nats=minimum,
        xi_tolerance_nats=tolerance, xi_negative_within_tolerance_count=int(np.count_nonzero(values['xi'] < 0)))


def curves(ax, g, y, colors, name, letter, zero=False):
    for i in range(93):
        ax.plot(g, y[i], color=colors[i], alpha=.50, lw=.70, zorder=1)
    ax.plot(g, y[-1], color='#202020', ls=(0, (5, 2.4)), lw=2.0,
            marker='o', ms=2.1, markevery=5, zorder=4)
    ax.set(xlim=(0, 4), xticks=np.arange(0, 4.1, 1), xlabel='$G$',
           ylabel='Mean E rate (Hz)' if name == 'rate' else 'Information (nats)', title=TITLE[name])
    if zero:
        ax.set_ylim(bottom=0)
    ax.margins(y=.06)
    ax.text(-.12, 1.06, letter, transform=ax.transAxes, fontweight='bold', fontsize=12)


def draw(g, values, rho, scenario):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Native SC property, independent of plotted metrics. Same individual color everywhere.
    palette = LinearSegmentedColormap.from_list('native_sc', plt.cm.viridis(np.linspace(.04, .90, 256)))
    norm = Normalize(float(rho.min()), float(rho.max()))
    colors = palette(norm(rho))
    individual = Line2D([], [], color=palette(.5), lw=.9, label='Individuals (n = 93; 3-seed means)')
    reference = Line2D([], [], color='#202020', ls=(0, (5, 2.4)), lw=2,
                       marker='o', ms=2, label='Mean SC (separate simulation)')
    style = {'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 10,
             'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
             'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}
    exports = []
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.8, 6.6), layout='constrained')
        left, right = fig.subfigures(1, 2, width_ratios=[2.18, 1], wspace=.045)
        axes = left.subplots(2, 2, sharex=True)
        for ax, name, letter in zip(axes.ravel(), ['rate', 'xi', 'phi_r', 'wms'], 'abcd'):
            curves(ax, g, values[name], colors, name, letter, zero=name in ('rate', 'xi'))
        for ax in axes[0]:
            ax.set_xlabel('')
        left.legend(handles=[individual, reference], loc='outside upper center', ncol=2,
                    frameon=False, fontsize=8.5)
        cb = left.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette), ax=axes.ravel().tolist(),
                           orientation='horizontal', shrink=.72, aspect=40, pad=.07)
        cb.set_label('Native SC spectral radius (same individual colors in all panels)', fontsize=8.5)
        cb.ax.tick_params(labelsize=8)

        ax = right.subplots()
        for position, name in zip([2, 1, 0], NAMES):
            m = scenario['cohorts']['all93']['methods'][name]
            assert m['denominator'] == 93
            rate = 100 * m['hit_rate']
            ax.barh(position, rate, height=.40, color=METHOD_COLORS[name], linewidth=0)
            ax.text(rate+1.7, position, f"{rate:.1f}%  {m['hit_count']}/93",
                    va='center', ha='left', fontsize=8.2)
        ax.axvline(50, color='#aaa', ls=':', lw=.9, zorder=0)
        ax.set(xlim=(0, 100), ylim=(-.6, 2.6), xticks=np.arange(0, 101, 25),
               yticks=[2, 1, 0], yticklabels=[METHOD_LABELS[n] for n in NAMES],
               xlabel='Hits within own transition window (%)',
               title='All 93 participants\nFixed 0.5 G window')
        ax.text(-.11, 1.06, 'e', transform=ax.transAxes, fontweight='bold', fontsize=12)
        for ext in ('png', 'pdf'):
            name = f'fixed_window_overview.{ext}'
            fig.savefig(OUTPUT/name, dpi=260, bbox_inches='tight')
            exports.append(name)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), layout='constrained')
        for ax, name, letter in zip(axes, ['whole_ei', 'partial_ei_sum', 'xi'], 'abc'):
            curves(ax, g, values[name], colors, name, letter, zero=True)
        for ax in axes[:2]:
            ax.set_ylim(0, 160)
        axes[-1].set_ylim(0, 23)
        fig.legend(handles=[individual, reference], loc='outside upper center', ncol=2,
                   frameon=False, fontsize=8.5)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette), ax=axes.tolist(),
                          orientation='horizontal', shrink=.60, aspect=45, pad=.06)
        cb.set_label('Native SC spectral radius (same colors as overview)', fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        for ext in ('png', 'pdf'):
            name = f'fixed_window_ei_decomposition.{ext}'
            fig.savefig(OUTPUT/name, dpi=260, bbox_inches='tight')
            exports.append(name)
        plt.close(fig)
    return exports


def write_report(config):
    # Only the canonical Appendix P changes; preserve every pre-existing result verbatim.
    text = REPORT.read_text()
    start, end = '<!-- report-section:dmf-subject-dense:start -->', '<!-- report-section:dmf-subject-dense:end -->'
    assert text.count(start) == text.count(end) == 1
    a, b = text.index(start), text.index(end)
    section = text[a:b]
    lo, hi = '<!-- dense-fixed-view:start -->', '<!-- dense-fixed-view:end -->'
    if lo in section:
        assert section.count(lo) == section.count(hi) == 1
        section = section[:section.index(lo)] + section[section.index(hi)+len(hi):]
    archive_start = '<!-- dense-fixed-archive:start -->\n<details>\n<summary>历史结果：原精确区间、粗网格分析与完整窗口搜索</summary>\n'
    archive_end = '</details>\n<!-- dense-fixed-archive:end -->'
    section = section.replace(archive_start, '').replace(archive_end, '')
    # Keep the dedicated rebound view outside the historical fold on later rerenders.
    rebound = ''
    rl, rh = '<!-- dense-rebound-view:start -->', '<!-- dense-rebound-view:end -->'
    if rl in section:
        assert section.count(rl) == section.count(rh) == 1
        ri, rj = section.index(rl), section.index(rh)+len(rh)
        rebound, section = section[ri:rj], section[:ri]+section[rj:]
    heading = '## 附录 P：93 人 DMF 细扫描\n'
    assert section.count(heading) == 1
    h = section.index(heading) + len(heading)
    old_content = section[h:].strip()
    body = f'''{lo}
<a id="dmf-subject-fixed-window"></a>

### 当前固定展示参数与总览（2026-10-04）

用户确认固定 **G=0–4、步长0.1**；以每个人独立平均E发放率曲线的**最大正斜率小区间**[L,U]为中心，左右各扩两格，命中窗口为[max(0,L−0.2),min(4,U+0.2)]。这里是发放率对G的最大正导数区间，不是发放率达到最高值的位置。名义总宽0.5 G；当前93人的窗口均无端点截断，实际宽也都是0.5 G。

Ξ/ΦR取原全局峰、保留符号的WMS取原全局谷；内部极值、平坦/边界处理、首并列点及含端点规则均保留。参数固定只停止本次继续选宽度，没有使此前同队列的事后选择成为预注册独立验证。原0.1精确定位与全部窗口搜索保留在下方历史记录。

![93人曲线、平均SC与固定窗口命中率](assets/dmf_subject_dense/fixed_window_overview.png)

**图P当前主图。** a为独立序参量（100ROI平均E发放率，Hz），b为整合有效信息Ξ，c为BOLD-like pairwise ΦR，d为保留符号的原生source WMS，e为三指标在各自独立转折窗口下的命中率。a–d每个面板都包含全部93人的原始41格点、同人3seed均值，无平滑、插值、峰位对齐或幅度标准化。个体颜色由原生SC谱半径决定，同一个人在四图中颜色一致；黑色虚线及小圆点是93人平均SC的**独立模拟**，不是93条个体曲线的算术平均，也不算第94位被试。不同子图使用各自原生纵轴范围，不据幅度大小比较指标优劣。

e按用户要求只展示**全部93人**的三项命中率：Ξ **65/93（69.9%）**，ΦR/WMS各 **28/93（30.1%）**，三项分母统一为93。柱高是样本比例，配对差及CI见历史窗口统计；50%竖线只对应用户的展示目标。窗口按各人的序参量单独确定，平均SC参照不进入人数统计。

**93人与85人的关系。** 93人是完整队列，其中8人此前已经用于开发和查看方案；原分析把后来新增的85人作为主要检验集，所以93=开发8人+新增85人。当前总览统一展示93人，原85人的配对检验与40项窗口搜索校正仍保存在下方历史记录。这个展示调整不改变被试成员、命中分类或参数，也不将全93人结果解释为一组重新独立验证的数据。

![93人整体EI、部分EI之和与整合有效信息分解](assets/dmf_subject_dense/fixed_window_ei_decomposition.png)

**EI分解单独展示。** 全部93人及平均SC参照的whole EI、部分EI之和、Ξ，数据与颜色规则同主图。它们满足 **Ξ=whole EI−部分EI之和**，由两个分量下降速度的差异形成Ξ的峰；不是两个分量相加。whole EI与部分EI之和不进入三指标命中率比较。前两幅纵轴都为0–160 nats，Ξ单列0–23 nats。此图保留低G与个别回摆，不把曲线概括为全域严格单调。

当前图用于先看总体趋势与个体差异；93条曲线的总体外观不能替代同人命中判断，也不能证明无相变时不会报峰。固定窗口宽为原0.1区间的5倍，仍是转折邻域对应，不能称精确相变点定位。此前原生WMS使用完整E/I自然状态、ΦR使用BOLD-like轨迹的协议差异保持并已记录。

本次重新通过Zotero按标题核实父条目P6UJCVG8及附件清单，当前唯一正文为DXGC7JEA（19页），读取Brain/Fig.2和Methods式（5）、（7）、（8）；仍无明确稿件日期/版本，补充附录不可用。图只复用冻结缓存，不变更稿件相关估计方法。当前固定参数及来源核验保存在[展示配置](../../results/dmf_schaefer100/subject_curves_93_dense/fixed_window_view.json)，重现：`.venv/bin/python -m scripts.plot_dmf_subject_fixed_window`。
{hi}'''
    section = section[:h] + '\n' + body + '\n\n' + (rebound+'\n\n' if rebound else '') + archive_start + '\n' + old_content + '\n\n' + archive_end + '\n'
    updated = text[:a] + section + text[b:]
    assert updated[:a] == text[:a] and updated.endswith(text[b:])
    REPORT.write_text(updated)
    return dict(path='docs/reports/brain.md#dmf-subject-fixed-window', sha256=digest(REPORT),
                historical_content_preserved=True, changes_outside_appendix_P=False)


def main():
    config = json.loads((BASE / 'fixed_window_view.json').read_text())
    contract, g, values, rho, scenario, audit = read_inputs(config)
    exports = draw(g, values, rho, scenario)
    report = write_report(config)
    for name, expected in config['source_sha256'].items():
        assert digest(BASE/name) == expected, f'Protected file changed: {name}'
    atomic_json(BASE/'fixed_window_view_review.json', dict(
        status='rendered_visual_review_pending', config_sha256=digest(BASE/'fixed_window_view.json'),
        implementation_sha256=digest(Path(__file__)), figures_sha256={n:digest(OUTPUT/n) for n in exports},
        report=report, audit=audit, native_G_step=config['G_step'], k_each_side=config['k_each_side'],
        hit_counts={key:{n:scenario['cohorts'][key]['methods'][n]['hit_count'] for n in NAMES}
                    for key in ('all93', 'primary85')},
        additional_simulations=0, additional_hypothesis_tests=0))
    print('Fixed view rendered: 93 individual curves + separate mean SC; G step .1, k=2, width .5.')
    print('Hits', {key:{n:scenario['cohorts'][key]['methods'][n]['hit_count'] for n in NAMES}
                   for key in ('all93', 'primary85')})


if __name__ == '__main__':
    main()
