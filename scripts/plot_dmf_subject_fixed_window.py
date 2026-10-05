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
    # Update only the two active blocks in their current body/appendix locations.
    # Never recreate Appendix P or move the identification results back to its end.
    text = REPORT.read_text()
    lo, hi = '<!-- dense-fixed-view:start -->', '<!-- dense-fixed-view:end -->'
    body = f'''{lo}
<a id="dmf-subject-fixed-window"></a>

### 1.1 个体转折定位与三指标命中率

**在93个体SC驱动的DMF模型中，Ξ的极值命中个体序参量转折邻域的比例为69.9%，ΦR与WMS均为30.1%。** 每个人分别使用自己的原生SC，G=0–4、步长0.1，每个格点取seed 3/4/5均值；另模拟93人平均SC作为参照，共11,562个完整条件。

任务是比较指标极值是否对应每个人独立发放率曲线的变化。以独立100ROI平均E发放率对G的**最大正斜率小区间**[L,U]为中心，左右各扩两格，命中窗口为[max(0,L−0.2),min(4,U+0.2)]。它由序参量决定，不由三个指标决定；不是最大发放率所在的位置。当前93人的实际窗口均为0.5 G，无端点截断。

Ξ/ΦR取原全局峰，保留符号的WMS取原全局谷；内部极值落入窗口才算命中，平坦或边界极值算未命中。保持首并列点及含端点规则，不平滑、插值或移动极值。93人均有可定位的独立转折，三项分母统一为93；平均SC不进入统计。

![93人曲线、平均SC与固定窗口命中率](assets/dmf_subject_dense/fixed_window_overview.png)

**识别总览图｜93人曲线与命中率。** a为独立序参量（Hz），b为整合有效信息Ξ，c为BOLD-like pairwise ΦR，d为保留符号的原生source WMS，e为全部93人的命中率。a–d均保留全部41格点的同人3seed均值，个体颜色按原生SC谱半径编码、跨图一致。黑色虚线及小圆点是平均SC的**独立模拟**，不是93条曲线的算术平均。各图使用原生纵轴范围，信息量单位为nats；没有幅度标准化或峰位对齐。图例在数据外。

e只展示93人结果：Ξ **65/93（69.9%）**，ΦR/WMS各 **28/93（30.1%）**。50%竖线对应用户的展示目标，柱高为样本比例。

原主要检验集为新增85人，另8人曾用于开发和查看方案。85人中Ξ为58/85（68.2%）、两基线各25/85（29.4%），两项配对增幅均为38.8个百分点；描述性95%bootstrap CI为[28.2,49.4]个百分点（未作多重/选窗校正），覆盖全部40项窗口比较的Holm p均4.6566×10⁻⁹。窗口0.5 G是同队列事后搜索后固定的容许范围，宽为原0.1区间的5倍；支持**转折邻域对应**，不能当作独立确认性结果或精确相变点定位。完整搜索、原精确区间与队列记录见[附录P](#dmf-subject-dense)。

{hi}'''
    el, eh = '<!-- dense-ei-view:start -->', '<!-- dense-ei-view:end -->'
    ei_body = f'''{el}
<a id="dmf-subject-ei-dense"></a>

### Q.1 93人细扫描：两个EI分量的下降速度形成整合峰

识别比较之后，进一步考察Ξ的代数来源。在同一93人、G步长0.1和三seed协议下，分别展示whole EI、部分EI之和与Ξ；whole EI和部分EI之和不进入前面的三指标命中比较。

![93人整体EI、部分EI之和与整合有效信息分解](assets/dmf_subject_dense/fixed_window_ei_decomposition.png)

**图 Q1｜联合读取与单变量读取。** 每条细线为一个体三seed均值，颜色与识别总览图一致，黑色虚线为平均SC独立模拟。两分量满足 **Ξ=whole EI−部分EI之和**。前两幅共用0–160 nats范围，Ξ单列0–23 nats；保留低G及个别回摆，不将两分量概括为全域严格单调。

部分EI之和下降更快时，差值Ξ增大；整体EI下降更快时，Ξ回落。峰值因而来自联合信息相对单变量信息的变化，不需要whole EI本身上升。当前共同affine-TM近似使用完整200维未来、同一干预和300ms时距，EI分解闭合检查通过。与下文八seed的早期经验Gaussian结果分开报告，不合并不同估计协议或bits/nats单位。
{eh}'''
    updated = text
    for lower, upper, value in [(lo, hi, body), (el, eh, ei_body)]:
        assert updated.count(lower) == updated.count(upper) == 1
        a, b = updated.index(lower), updated.index(upper)+len(upper)
        updated = updated[:a]+value+updated[b:]
    assert updated.index(lo) < updated.index('<a id="dmf-main">')
    assert updated.index('<a id="appendix-q">') < updated.index(el)
    REPORT.write_text(updated)
    return dict(path='docs/reports/brain.md#dmf-subject-fixed-window', sha256=digest(REPORT),
                historical_content_preserved=True, layout='identification_first_then_organization_with_ei_appendix',
                updated_blocks=['dense-fixed-view', 'dense-ei-view'])


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
