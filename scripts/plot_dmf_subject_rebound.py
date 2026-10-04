#!/usr/bin/env python3
"""Expose the previously frozen 93-subject U diagnostic as its own comparison.

No new estimator, threshold, simulation or hypothesis test. All original shapes
are recomputed from the small dense-curve cache to validate the stored definitions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.dmf_curve_shape import curve_shape, METHOD_COLORS, METHOD_LABELS, RELATIVE_SHAPE_TOLERANCE
from scripts.run_dmf_subject_consistency import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
OUTPUT = ROOT/'docs/reports/assets/dmf_subject_dense'
REPORT = ROOT/'docs/reports/brain.md'
NAMES = ('xi', 'phi_r', 'wms')


def validated_scores():
    config = json.loads((BASE/'fixed_window_view.json').read_text())
    for name, expected in config['source_sha256'].items():
        if digest(BASE/name) != expected:
            raise ValueError(f'Protected source changed: {name}')
    original = json.loads((BASE/'summary.json').read_text())
    contract = json.loads((BASE/'contract.json').read_text())
    results = {}
    with np.load(BASE/'analysis_curves.npz') as a:
        assert str(a['contract_sha256']) == digest(BASE/'contract.json')
        assert a['subject_ids'].tolist() == contract['subject_ids']
        g = a['G'].copy()
        assert np.array_equal(g, np.round(np.arange(41)*.1, 1))
        for n in NAMES:
            values = a[n][:-1].mean(axis=1)
            assert values.shape == (93, 41) and np.isfinite(values).all()
            if n == 'xi' and values.min() < -contract['syn_tolerance_nats']:
                count = int(np.count_nonzero(values < -contract['syn_tolerance_nats']))
                raise ArithmeticError(f'Xi below tolerance: min={values.min()}, '
                    f'threshold={-contract["syn_tolerance_nats"]} nats, affected_count={count}')
            scores = []
            for i, row in enumerate(original['subject_rows']):
                recovered = curve_shape(g, values[i], -1 if n == 'wms' else 1)
                assert recovered == row['shape'][n], (row['subject'], n)
                scores.append(dict(subject=row['subject'], U=recovered['unimodality_violation'],
                    U_post=recovered['post_extremum_rebound_fraction'],
                    U_pre=recovered['unimodality_violation']-recovered['post_extremum_rebound_fraction'],
                    weakly_unimodal=recovered['weakly_unimodal'], interior=recovered['peak_interior'],
                    flat=recovered['flat']))
            assert len(scores) == 93 and not any(s['flat'] for s in scores)
            u = np.array([s['U'] for s in scores])
            old = original['shape_summary']['all93'][n]
            assert float(u.mean()) == old['unimodality_violation']['mean']
            assert float(np.median(u)) == old['unimodality_violation']['median']
            assert float(u.max()) == old['unimodality_violation']['maximum']
            assert sum(s['weakly_unimodal'] for s in scores) == old['weakly_unimodal_count']
            results[n] = dict(subject_count=93, scores=scores, mean=float(u.mean()),
                median=float(np.median(u)), maximum=float(u.max()),
                no_rebound_count=sum(s['weakly_unimodal'] for s in scores),
                all_U_equal_U_post=all(s['U_pre'] == 0 for s in scores))
    # Scientific properties, not shape thresholds fitted to these subjects.
    assert curve_shape([0, .1, .2], [0, 10, 0])['unimodality_violation'] == 0
    alternating = np.array([0., 1., 0., 1., 0., 1., 0.])
    ag = np.arange(len(alternating))*.1
    assert curve_shape(ag, alternating)['unimodality_violation'] == 2
    assert np.isclose(curve_shape(ag, alternating)['unimodality_violation'],
                      curve_shape(ag, 7.3*alternating+4.1)['unimodality_violation'])
    assert curve_shape([0, .1, .2], [1, 1, 1])['unimodality_violation'] is None
    return config, results


def draw(results):
    with plt.rc_context({'font.family':'sans-serif', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42}):
        fig, ax = plt.subplots(figsize=(7.4, 4.0), layout='constrained')
        for i, n in enumerate(NAMES):
            y = 100*np.array([s['U'] for s in results[n]['scores']])
            ax.boxplot([y], positions=[i], widths=.38, showfliers=False,
                medianprops={'color':'#222222', 'lw':1.1},
                boxprops={'color':METHOD_COLORS[n]}, whiskerprops={'color':METHOD_COLORS[n]},
                capprops={'color':METHOD_COLORS[n]})
            ax.scatter(i+.15*np.sin(np.arange(93)*2.4), y, color=METHOD_COLORS[n],
                       s=13, alpha=.48, linewidths=0, zorder=2)
            ax.scatter(i, y.mean(), marker='D', s=37, color=METHOD_COLORS[n],
                       edgecolor='#202020', linewidth=.6, zorder=4)
            mean_text = r'$3.55\times10^{-5}\%$' if n == 'xi' else f'{y.mean():.2f}%'
            ax.text(i, 1.02, 'Mean: '+mean_text, transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=9)
        labels = [METHOD_LABELS[n]+f"\n{results[n]['no_rebound_count']}/93 with U ≤ tol." for n in NAMES]
        ax.set(xticks=range(3), xticklabels=labels, xlim=(-.5, 2.5), ylim=(-1.5, 65),
               ylabel='Cumulative rebound / curve amplitude (%)')
        ax.axhline(0, color='#ccc', lw=.6, zorder=0)
        ax.legend(handles=[Line2D([], [], marker='o', ls='none', color='#777', ms=4, label='Subject (3-seed mean)'),
                           Line2D([], [], marker='D', ls='none', color='#777', markeredgecolor='#222', ms=5, label='Mean across 93')],
                  loc='center left', bbox_to_anchor=(1.02, .5), frameon=False, fontsize=9)
        exports = []
        for ext in ('png', 'pdf'):
            name=f'fixed_window_rebound.{ext}'
            fig.savefig(OUTPUT/name, dpi=260, bbox_inches='tight')
            exports.append(name)
        plt.close(fig)
    return exports


def write_report(results):
    labels = {'xi':'Ξ', 'phi_r':'ΦR', 'wms':'WMS'}
    rows = '\n'.join(f"| {labels[n]} | {100*results[n]['mean']:.8g}% | "
        f"{100*results[n]['median']:.5g}% | {results[n]['no_rebound_count']}/93 | {100*results[n]['maximum']:.6g}% |"
        for n in NAMES)
    lo, hi = '<!-- dense-rebound-view:start -->', '<!-- dense-rebound-view:end -->'
    body = rf'''{lo}
<a id="dmf-subject-rebound"></a>

### 独立比较维度：额外回摆 U（93人）

用原协议已冻结的**累计反向变化量 U**专门区分单峰趋势和额外起伏；不另选一个有利阈值。先取原41个G点的同人3seed平均曲线，Ξ、ΦR用原值，WMS只在这个形状诊断中取负值以把单谷对应到单峰。主图WMS符号和原谷位不变。

对非平坦曲线 $mathbf{{y}}=(y_1,ldots,y_{{41}})^	op$，令 $p$ 为首个全局最大值的位置，$Delta_i=y_{{i+1}}-y_i$，$A=max_i y_i-min_i y_i>0$。定义

$$
U=U_{{m pre}}+U_{{m post}}
=rac{{sum_{{i<p}}[-Delta_i]_++sum_{{ige p}}[Delta_i]_+}}{{A}},qquad [z]_+=max(z,0).
$$

第一项累计峰前下降，第二项累计峰后回升；除以整条曲线幅度消除nats量级差异。显示为100U%，表示回摆量相当于曲线幅度的多少，而非发生回摆的被试比例。**U越低，越接近先升后降；陡峭但单峰的曲线仍有U=0。**

**性质及边界。** 对非平坦有限曲线，U非负；U=0当且仅当全局峰前弱单调上升、峰后弱单调下降，因为分子各项非负，和为0要求所有反向增量为0。该数学比值对正幅度缩放及平移不变；实际浮点输入仍须通过原平坦判据：幅度不超过10⁻¹²×max(1,最大绝对值)则单列为不可辨识。它不测峰位是否对应独立转折，也不要求存在内部峰：边界单调曲线也可得0分。累计回摆可超过100%，例如[0,1,0,1,0,1,0]有U=2，因此U不是0到1概率。平坦曲线的U未定义，当前93人三项均无平坦曲线。原弱单峰容差是无量纲10⁻¹⁰，不是Ξ非负容差10⁻⁸ nats。

![93人累计额外回摆的分布](assets/dmf_subject_dense/fixed_window_rebound.png)

**回摆分布。** 每个圆点是一人的3seed平均曲线，共93人；横向散开仅防重叠。箱线表示中位数与四分位分布、须为1.5 IQR范围内数据，全部个体点保留，菱形是93人算术均值；没有CI或显著性星号。三项共用线性纵轴，Ξ的微小回摆在这一尺度接近0，精确量级列在表中。图例位于数据外。

| 指标 | 平均累计回摆/幅度 | 中位数 | U≤10⁻¹⁰人数 | 最大累计回摆/幅度 |
|---|---:|---:|---:|---:|
{rows}

Ξ的89/93条曲线满足原弱单峰容差，另4条最大回摆也只为完整幅度的0.001789%；ΦR和WMS的93条曲线均有超容差回摆。**本次每个被试、每项指标的峰前反向量都为0，故整体U与峰后回摆逐项相等**，不再重复画一个数值相同的面板。

回摆与粗糙度Q分开：Q衡量相邻斜率的变化，可同时响应陡峭尖峰与频繁振荡；U更直接回答“有没有额外回摆”。当前结果支持固定SC＋DMF、原生估计协议下，Ξ的单峰趋势更稳定；不能将短自然轨迹、高维拟合或正则化导致的观测基线波动直接归为指标公式失效。这里只复用既有描述性形状结果，没有新增模拟、检验或改变G步长0.1/命中窗口0.5 G；3seed均值的平滑外观不代替逐seed稳健性。

本轮重新通过Zotero核实主稿父条目P6UJCVG8及唯一正文附件DXGC7JEA（19页），读取Brain/Fig.2与Methods式（7）、（8）；无明确稿件版本日期且补充附录不可用。U是原实验的形状诊断，不作为稿件新信息量或PEID分解项。原生WMS完整E/I自然状态与稿件BOLD-like基线措辞差别保持。复现：`.venv/bin/python -m scripts.plot_dmf_subject_rebound`。
{hi}'''
    current = REPORT.read_text()
    start, end = '<!-- report-section:dmf-subject-dense:start -->', '<!-- report-section:dmf-subject-dense:end -->'
    a, b = current.index(start), current.index(end)
    section = current[a:b]
    if lo in section:
        assert section.count(lo) == section.count(hi) == 1
        i, j = section.index(lo), section.index(hi)+len(hi)
        section = section[:i]+section[j:]
    anchor = '<!-- dense-fixed-view:end -->'
    assert section.count(anchor) == 1
    section = section.replace(anchor, anchor+'\n\n'+body, 1)
    updated = current[:a]+section+current[b:]
    assert updated[:a] == current[:a] and updated.endswith(current[b:])
    REPORT.write_text(updated)
    return dict(path='docs/reports/brain.md#dmf-subject-rebound', full_report_sha256=digest(REPORT),
                rebound_block_sha256=hashlib.sha256(body.encode()).hexdigest(), changes_outside_appendix_P=False)


def main():
    config, results = validated_scores()
    figures = draw(results)
    report = write_report(results)
    for name, expected in config['source_sha256'].items():
        assert digest(BASE/name) == expected, name
    atomic_json(BASE/'rebound_view_review.json', dict(status='rendered_visual_review_pending',
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), source_sha256=config['source_sha256'],
        implementation_sha256=digest(Path(__file__)), definition_module_sha256=digest(ROOT/'scripts/dmf_curve_shape.py'),
        original_93_shapes_exactly_reproduced=True, G_step=.1, node_count=41, seed_count=3,
        results=results, tolerance_U=RELATIVE_SHAPE_TOLERANCE,
        property_controls={'steep_single_peak_has_U_zero':True, 'repeated_rebound_can_exceed_one':True,
            'positive_scale_and_offset_invariant':True, 'flat_is_undefined':True},
        figures_sha256={n:digest(OUTPUT/n) for n in figures}, report=report,
        additional_simulations=0, additional_hypothesis_tests=0))
    print('Rebound view rendered from frozen 93-subject shapes.')
    print({n:dict(mean_percent=100*r['mean'], no_rebound_count=r['no_rebound_count'],
                  all_U_equal_U_post=r['all_U_equal_U_post']) for n,r in results.items()})


if __name__ == '__main__':
    main()
