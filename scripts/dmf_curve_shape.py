"""Descriptive finite-grid shape and independent rate-landmark diagnostics.

No smoothing, peak alignment, estimator changes, or additional simulations.
WMS is negated ONLY for single-trough diagnostics; figures retain its signed value.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

SYN_TOLERANCE_NATS = 1e-8
RELATIVE_SHAPE_TOLERANCE = 1e-10
PROMINENCE_FRACTIONS = (0., .01, .05)
METHOD_COLORS = {'xi': '#B64D64', 'phi_r': '#287C9C', 'wms': '#737B85'}
METHOD_LABELS = {'xi': '$\\Xi$', 'phi_r': '$\\Phi^R$', 'wms': 'WMS'}


def curve_shape(g, values, direction=1):
    """Scale-free descriptors on the ACTUAL nonuniform G grid.

    U is wrong-way movement on the two sides of the global maximum / range.
    It is zero for every weakly unimodal curve, including boundary-monotone ones.
    Q is total variation of normalized adjacent-interval slopes. Q can also be
    large for a legitimate sharp single peak, so it has no pass/fail threshold.
    """
    g = np.asarray(g, float)
    raw = np.asarray(values, float)
    if g.ndim != 1 or raw.shape != g.shape or len(g) < 3:
        raise ValueError('Expected matching one-dimensional curves with >=3 points')
    if not np.isfinite(g).all() or not np.isfinite(raw).all() or np.any(np.diff(g) <= 0):
        raise ValueError('Nonfinite curve or non-increasing G grid')
    if direction not in (-1, 1):
        raise ValueError('Diagnostic direction must be +1 or -1')
    y = direction * raw
    span = float(np.ptp(y))
    flat = span <= 1e-12 * max(1., float(np.abs(y).max()))
    result = dict(flat=flat, direction=direction, amplitude_nats=span,
                  peak_G=None if flat else float(g[y.argmax()]),
                  peak_interior=False if flat else bool(0 < y.argmax() < len(y)-1))
    if flat:
        return dict(result, unimodality_violation=None, wrong_way_nats=0.,
                    post_extremum_rebound_fraction=None, slope_total_variation=None,
                    weakly_unimodal=None, sign_reversal_count=0,
                    interior_peak_counts={str(p): 0 for p in PROMINENCE_FRACTIONS},
                    classification='flat: no identifiable extremum')
    k = int(y.argmax())
    dy = np.diff(y)
    pre = -dy[:k][dy[:k] < 0].sum()
    post = dy[k:][dy[k:] > 0].sum()
    u = float((pre + post) / span)
    x = (g - g[0]) / (g[-1] - g[0])
    z = (y - y.min()) / span
    slopes = np.diff(z) / np.diff(x)
    signs = np.sign(np.diff(z)[np.abs(np.diff(z)) > RELATIVE_SHAPE_TOLERANCE])
    counts = {str(p): int(len(find_peaks(z, prominence=p)[0])) for p in PROMINENCE_FRACTIONS}
    weak = u <= RELATIVE_SHAPE_TOLERANCE
    classification = ('single interior peak/trough' if result['peak_interior'] else 'boundary/monotone') if weak else 'non-unimodal'
    return dict(result, unimodality_violation=u, wrong_way_nats=float(pre + post),
                post_extremum_rebound_fraction=float(post / span),
                slope_total_variation=float(np.abs(np.diff(slopes)).sum()),
                weakly_unimodal=bool(weak), sign_reversal_count=int(np.count_nonzero(np.diff(signs))),
                interior_peak_counts=counts, classification=classification)


def analyze_shapes(d, summary, curves, base):
    for subject in d['ids']:
        with np.load(base/'dynamics'/f'{subject}.npz') as cache:
            if json.loads(str(cache['contract_json'])) != d['contract']:
                raise ValueError(f'Independent dynamics contract mismatch: {subject}')
            if not np.isfinite(cache['mean_rate_hz']).all() or np.any(cache['boundary_fraction'] != 0):
                raise ArithmeticError(f'Invalid independent order-parameter cache: {subject}')
    xi = np.asarray(curves['xi'])
    if not np.isfinite(xi).all():
        raise ArithmeticError('Nonfinite Xi in shape analysis')
    count = int(np.count_nonzero(xi < -SYN_TOLERANCE_NATS))
    if count:
        raise ArithmeticError(f'Xi nonnegativity violation: minimum={xi.min()} nats, '
                              f'threshold={-SYN_TOLERANCE_NATS} nats, count={count}')
    methods = {}
    for name, v in curves.items():
        direction = -1 if name == 'wms' else 1
        rows = []
        for i, subject in enumerate(d['ids'][:8]):
            row = curve_shape(d['g'], v[i].mean(0), direction)
            t = summary['metrics'][name]['extrema'][i]['transition']
            peak = row['peak_G']
            distance = None
            if t['located'] and peak is not None:
                low, high = t['interval']
                distance = float(max(low-peak, 0., peak-high))
            rows.append(dict(subject=subject, transition=t,
                             distance_to_transition_interval_G=distance,
                             mean_curve=row,
                             seeds=[curve_shape(d['g'], y, direction) for y in v[i]]))
        u = np.array([r['mean_curve']['unimodality_violation'] for r in rows], float)
        q = np.array([r['mean_curve']['slope_total_variation'] for r in rows], float)
        eligible = [r for r in rows if r['distance_to_transition_interval_G'] is not None]
        methods[name] = dict(subjects=rows, weakly_unimodal_count=sum(r['mean_curve']['weakly_unimodal'] is True for r in rows),
            interior_global_extremum_count=sum(r['mean_curve']['peak_interior'] for r in rows),
            exactly_one_prominent_interior_peak_count=sum(r['mean_curve']['interior_peak_counts']['0.05'] == 1 for r in rows),
            unimodality_violation_mean=float(np.nanmean(u)), unimodality_violation_sd=float(np.nanstd(u, ddof=1)),
            slope_total_variation_mean=float(np.nanmean(q)), slope_total_variation_sd=float(np.nanstd(q, ddof=1)),
            transition_interval_hit_count=sum(r['distance_to_transition_interval_G'] == 0. for r in eligible),
            eligible_transition_count=len(eligible),
            distance_to_interval_G_mean=float(np.mean([r['distance_to_transition_interval_G'] for r in eligible])) if eligible else None)
    return dict(methods=methods, G=d['g'].tolist(), shape_tolerance_relative=RELATIVE_SHAPE_TOLERANCE,
        prominence_sensitivity_fractions=list(PROMINENCE_FRACTIONS),
        seed_unit='Metrics computed per seed and on seed-mean curve; subject is the summary unit',
        direction='Xi/PhiR: maximum; WMS: minimum via negation for diagnostics only; signed plots retained',
        smoothing='none; slopes use the actual unequal G spacing; no interpolation',
        flat_rule='Range <= 1e-12 * max(1, maximum absolute value): shape scores undefined, peak absent',
        transition_rule='Existing independent mean E firing-rate maximum-slope interval; interior positive maximum is a candidate, not proof of a phase transition',
        no_transition_controls=0, false_positive_rate=None,
        boundary_censored_subjects=[r['subject'] for r in methods['xi']['subjects'] if not r['transition']['located']],
        xi_audit=dict(tolerance_nats=SYN_TOLERANCE_NATS, minimum_nats=float(xi.min()),
            numerical_negative_count=int(np.count_nonzero((xi < 0) & (xi >= -SYN_TOLERANCE_NATS))),
            significant_negative_count=count, projection='none'),
        manuscript_recheck=dict(parent='P6UJCVG8', attachment='DXGC7JEA', version_date=None,
            sections='Brain/Fig.2 and Methods Eqs.5,7,8; 19 pages',
            unresolved='No explicit manuscript date/version; supplementary appendices unavailable'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def draw_order_comparison(d, summary, curves, path):
    """Eight subject cells, each with independent order curve above metric shapes."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(12.6, 9.2), layout='constrained')
    cells = fig.subfigures(2, 4, wspace=.04, hspace=.08)
    for i, cell in enumerate(cells.ravel()):
        axes = cell.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1.5], 'hspace': .08})
        rates = d['dyn'][i]
        mean, sd = rates.mean(0), rates.std(0, ddof=1)
        axes[0].plot(d['g'], mean, color='#222222', marker='o', ms=2.5, lw=1.25)
        axes[0].fill_between(d['g'], mean-sd, mean+sd, color='#222222', alpha=.14)
        axes[0].set(title=d['ids'][i], ylabel='Mean E rate (Hz)')
        t = summary['metrics']['xi']['extrema'][i]['transition']
        for ax in axes:
            ax.axvspan(*t['interval'], color='#BFC3C8', alpha=.22, lw=0,
                       hatch='///' if not t['located'] else None, zorder=0)
            ax.set_xlim(-.07, 3.07)
        for name, v in curves.items():
            m, spread = v[i].mean(0), v[i].std(0, ddof=1)
            scale = m.std(ddof=0)
            if scale <= 1e-12 * max(1., float(np.abs(m).max())):
                continue
            z = (m-m.mean())/scale
            if np.any(z-spread/scale < -2.7) or np.any(z+spread/scale > 2.7):
                raise ValueError('Fixed metric shape axis would clip a seed SD band')
            axes[1].plot(d['g'], z, color=METHOD_COLORS[name], lw=1.2, marker='o', ms=2.3)
            axes[1].fill_between(d['g'], z-spread/scale, z+spread/scale, color=METHOD_COLORS[name], alpha=.10)
        axes[1].axhline(0, color='#dedede', lw=.6, zorder=0)
        axes[1].set(xlabel='$G$', ylabel='Metric shape (z score)', ylim=(-2.7, 2.7),
                    xticks=[0., 1., 2., 3.], yticks=[-2., 0., 2.])
    handles = [Line2D([], [], color='#222222', lw=1.3, label='Independent E firing rate')]
    handles += [Line2D([], [], color=METHOD_COLORS[n], lw=1.3, label=METHOD_LABELS[n]) for n in curves]
    handles += [Patch(facecolor='#BFC3C8', alpha=.3, label='Interior rate-turn interval'),
                Patch(facecolor='#BFC3C8', alpha=.3, hatch='///', label='Boundary: not located')]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False, fontsize=8)
    fig.savefig(path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def draw_shape_diagnostics(d, shape, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), layout='constrained', sharey=True)
    names = list(shape['methods'])
    for ni, name in enumerate(names):
        rows = shape['methods'][name]['subjects']
        y = np.arange(8) + (ni-(len(names)-1)/2)*.19
        for ax, key, factor in zip(axes, ['unimodality_violation', 'slope_total_variation'], [100., 1.]):
            ax.scatter([r['mean_curve'][key]*factor for r in rows], y, s=28,
                       color=METHOD_COLORS[name], marker=['o','s','^'][ni], label=METHOD_LABELS[name])
            ax.set(yticks=np.arange(8), yticklabels=d['ids'][:8])
            ax.axvline(0, color='#dedede', lw=.7, zorder=0)
    axes[0].set(xlabel='Wrong-way movement / curve range (%)')
    axes[1].set(xlabel='Total variation of normalized slopes')
    axes[0].invert_yaxis()
    for letter, ax in zip('ab', axes):
        ax.text(-.09, 1.03, letter, transform=ax.transAxes, fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False)
    fig.savefig(path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def shape_report(shape):
    zh = {'xi': 'Ξ', 'phi_r': 'ΦR', 'wms': 'WMS（单谷）'}
    rows, exceptions, seeds = [], [], []
    for name, v in shape['methods'].items():
        rows.append(f"| {zh[name]} | {v['weakly_unimodal_count']}/8 | "
            f"{v['interior_global_extremum_count']}/8 | {v['exactly_one_prominent_interior_peak_count']}/8 | "
            f"{100*v['unimodality_violation_mean']:.3f} ± {100*v['unimodality_violation_sd']:.3f}% | "
            f"{v['slope_total_variation_mean']:.3f} ± {v['slope_total_variation_sd']:.3f} |")
        for r in v['subjects']:
            m = r['mean_curve']
            if m['weakly_unimodal'] is False:
                exceptions.append(f"| {zh[name]} | {r['subject']} | {m['wrong_way_nats']:.6f} | "
                    f"{100*m['unimodality_violation']:.4f}% | {m['interior_peak_counts']['0.0']} / "
                    f"{m['interior_peak_counts']['0.01']} / {m['interior_peak_counts']['0.05']} |")
        for r in v['subjects']:
            seed_u = [100*s['unimodality_violation'] for s in r['seeds']]
            if any(u > 100*RELATIVE_SHAPE_TOLERANCE for u in seed_u):
                seeds.append(f"| {zh[name]} | {r['subject']} | "+' / '.join(f'{u:.4f}%' for u in seed_u)+' |')
    text = r'''令 $y_j$ 为某人的3seed均值曲线，$m=7$为格点数，$\Delta y_j=y_{j+1}-y_j$。Ξ、ΦR取原值；WMS仅在本节形状计算中取 $y_j=-WMS_j$，把单谷转为单峰，图中保留WMS原始符号。令 $k$ 是全局最大值的首个格点，$A=\max y-\min y$。对非平坦曲线，定义

$$
U=\frac{\sum_{j<k}[-\Delta y_j]_++\sum_{j\ge k}[\Delta y_j]_+}{A}
=\frac{\sum_j|\Delta y_j|-(y_k-y_0)-(y_k-y_{m-1})}{2A},\qquad [a]_+=\max(a,0).
$$

**U是单峰偏离量。** 分子累加主峰前的下降与主峰后的回升；除以该曲线的完整幅度，避免不同nats量级直接比较。弱单峰（先不降、后不升）、单调曲线和峰顶平台均为0；偏离越大，反向运动越大。这里的正部运算是描述量的定义，不是对Ξ/Syn做裁剪。相对形状零阈值为10⁻¹⁰；平坦曲线（幅度≤10⁻¹²×max(1,最大绝对值)）没有可识别极值，U/Q不定义，不算作理想单峰。

为单独描述折线的粗糙程度，把 $x_j=(G_j-G_0)/(G_{m-1}-G_0)$、$z_j=(y_j-\min y)/A$，定义

$$
Q=\sum_{j=0}^{m-3}\left|\frac{z_{j+2}-z_{j+1}}{x_{j+2}-x_{j+1}}-\frac{z_{j+1}-z_j}{x_{j+1}-x_j}\right|.
$$

**Q是归一化区间斜率的总变差。** 它使用真实不等距G格点，对幅度和平移不敏感。Q较大既可能来自锯齿，也可能来自真实的陡峭单峰，因此不给Q设置“正确/错误”阈值，更不将较小Q直接当作指标优势。7点只能描述观测折线，无法证明连续曲线平滑。

| 指标 | 弱单峰或单调 U≈0 | 主极值在内部 | 恰有1个显著内部峰/谷 | U个体均值±SD | Q个体均值±SD |
|---|---:|---:|---:|---:|---:|
@ROWS@

**表3。** 8人为统计单位，先平均3seed再计算形状。显著内部峰由格点prominence≥整条曲线幅度的5%定义，并保留0%、1%、5%敏感性结果；这是看过曲线后设定的探索性描述规则。局部prominence依赖扫描支持，边界不算内部峰，故恰有1个显著峰与U=0不能互相替代。Ξ的7个内部峰加1条边界单调上升曲线均有U=0。

![单峰偏离量与斜率总变差](assets/dmf_subject_consistency/curve_shape_diagnostics.png)

**图4。** 每个点是一人的3seed均值曲线。左为U×100%，右为Q；零值保留。WMS按单谷评价。量纲归一化只用于描述形状，没有重拟合、平滑或改动任何指标原值。

| 指标 | 被试 | 反向运动总量（nats） | U | 内部峰/谷数：0% / 1% / 5% |
|---|---|---:|---:|---:|
@EXCEPTIONS@

**表4。** 所有非单峰均值曲线，不按结果选择案例。ΦR在sub-10274的峰后回升约0.618 nats，U约30.7%，且在5%prominence规则下仍有2峰；这是明确的观测回弹。ΦR在sub-10228以及WMS的两个尾部回摆较小，不能把所有回摆都写成“强波动”。小幅回摆是否可重复还需对照seed；短自然轨迹、估计正则化和动力学差异也可能造成形状差异，本轮不能据此证明指标公式本身失效。

至少一个seed有回摆的全部被试如下（seed 3 / 4 / 5）；其余被试的各seed U均为0。完整逐seed U/Q、峰数、极值方向和边界分类保留在分析汇总curve_shape字段。Ξ的sub-10228在seed 4仍有0.0154%的小回摆，不能把8条均值曲线无回摆写成所有随机重复都严格单峰。

| 指标 | 被试 | 各seed U |
|---|---|---|
@SEEDS@

**表5。** 每人每seed独立描述，不把24个seed当作24个被试。
'''
    for key, value in {'@ROWS@': '\n'.join(rows), '@EXCEPTIONS@': '\n'.join(exceptions), '@SEEDS@': '\n'.join(seeds)}.items():
        text = text.replace(key, value)
    return text
