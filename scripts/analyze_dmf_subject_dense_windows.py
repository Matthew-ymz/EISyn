#!/usr/bin/env python3
"""Posthoc symmetric windows; all dense-grid landmarks and inputs stay fixed."""
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

from scripts.analyze_dmf_subject_dense_resolution import holm_family
from scripts.dmf_curve_shape import METHOD_COLORS, METHOD_LABELS
from scripts.dmf_dense_statistics import paired_hits, subject_landmarks
from scripts.run_dmf_subject_consistency import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
NAMES = ('xi', 'phi_r', 'wms')
LABELS = {'xi':'Ξ', 'phi_r':'ΦR', 'wms':'WMS'}
GRID_TOLERANCE_G = 1e-12  # Same inclusive-endpoint numerical rule as frozen primary.


def expanded_subject(row, k, step=.1, support=(0., 4.)):
    low, high = row['transition']['interval']
    uncut = (low-step*k, high+step*k)
    bounds = (max(support[0], uncut[0]), min(support[1], uncut[1]))
    output = dict(subject=row['subject'], independent_transition=row['transition'],
        window=dict(k_each_side=k, added_margin_G=step*k, nominal_width_G=step*(1+2*k),
                    interval=list(bounds), actual_width_G=bounds[1]-bounds[0],
                    lower_clipped=uncut[0] < support[0]-GRID_TOLERANCE_G,
                    upper_clipped=uncut[1] > support[1]+GRID_TOLERANCE_G), methods={})
    for n in NAMES:
        old = row['methods'][n]
        p = old['extremum_G']
        distance = None if p is None else float(max(bounds[0]-p, 0., p-bounds[1]))
        hit = bool(row['transition']['located'] and old['interior'] and not old['flat'] and
                   distance is not None and distance <= GRID_TOLERANCE_G)
        output['methods'][n] = dict(extremum_G=p, flat=old['flat'], interior=old['interior'],
            frozen_hit=old['hit'], hit=hit, distance_to_window_G=distance,
            frozen_distance_to_interval_G=old['distance_to_interval_G'])
    return output


def summarize_cohort(rows):
    if not rows:
        raise ValueError('Empty cohort')
    count = len(rows)
    widths = np.array([r['window']['actual_width_G'] for r in rows])
    valid = [r for r in rows if r['independent_transition']['located']]
    relative = np.array([r['window']['actual_width_G']/r['independent_transition']['midpoint']
                         for r in valid if r['independent_transition']['midpoint'] > 0])
    hits = {n: np.array([r['methods'][n]['hit'] for r in rows]) for n in NAMES}
    return dict(subject_count=count, located_count=len(valid),
        unlocated_subjects=[r['subject'] for r in rows if not r['independent_transition']['located']],
        width_G=dict(mean=float(widths.mean()), minimum=float(widths.min()), maximum=float(widths.max())),
        width_relative_to_transition_midpoint=dict(mean=float(relative.mean()), median=float(np.median(relative)),
            minimum=float(relative.min()), maximum=float(relative.max())),
        clipped_subject_count=sum(r['window']['lower_clipped'] or r['window']['upper_clipped'] for r in rows),
        lower_clipped_count=sum(r['window']['lower_clipped'] for r in rows),
        upper_clipped_count=sum(r['window']['upper_clipped'] for r in rows),
        methods={n:dict(hit_count=int(hits[n].sum()), denominator=count, hit_rate=float(hits[n].mean()),
            flat_count=sum(r['methods'][n]['flat'] for r in rows),
            boundary_extremum_count=sum(not r['methods'][n]['interior'] and not r['methods'][n]['flat'] for r in rows))
            for n in NAMES},
        comparisons={n:paired_hits(hits['xi'], hits[n]) for n in NAMES[1:]})


def compute(plan, frozen, contract):
    original = frozen['subject_rows']
    ids = [r['subject'] for r in original]
    if len(ids) != 93 or len(set(ids)) != 93 or set(ids) != set(contract['subject_ids'][:-1]):
        raise ValueError('Full subject membership changed')
    if contract['G'] != np.round(np.arange(41)*.1, 1).tolist():
        raise ValueError('Expected frozen 0.1 grid on 0..4')
    if any(not r['transition']['located'] for r in original):
        raise ValueError('Unexpected unlocated frozen subject; no silent denominator change')
    sets = dict(primary85=set(contract['holdout_subject_ids']), all93=set(ids),
                development8=set(contract['development_subject_ids']))
    assert len(sets['primary85']) == 85 and len(sets['development8']) == 8
    assert sets['primary85'].isdisjoint(sets['development8'])
    assert sets['primary85'] | sets['development8'] == sets['all93']
    scenarios = []
    previous = None
    for k in plan['k_values']:
        rows = [expanded_subject(r, k, plan['native_step_G'], plan['support_G']) for r in original]
        for source, new in zip(original, rows):
            assert source['transition'] == new['independent_transition']
            for n in NAMES:
                assert source['methods'][n]['extremum_G'] == new['methods'][n]['extremum_G']
                assert source['methods'][n]['interior'] == new['methods'][n]['interior']
                assert source['methods'][n]['flat'] == new['methods'][n]['flat']
                if k == 0:
                    assert source['methods'][n]['hit'] == new['methods'][n]['hit']
        if previous is not None:
            for old, new in zip(previous, rows):
                assert old['subject'] == new['subject']
                assert new['window']['interval'][0] <= old['window']['interval'][0]+GRID_TOLERANCE_G
                assert new['window']['interval'][1] >= old['window']['interval'][1]-GRID_TOLERANCE_G
                for n in NAMES:
                    assert not old['methods'][n]['hit'] or new['methods'][n]['hit']
        scenarios.append(dict(k_each_side=k, added_margin_G=plan['native_step_G']*k,
            nominal_width_G=plan['native_step_G']*(1+2*k), subject_rows=rows,
            cohorts={key:summarize_cohort([r for r in rows if r['subject'] in cohort]) for key, cohort in sets.items()}))
        previous = rows
    # Exact control for all cohorts, including original paired uncertainty/p-values.
    for key, old_key in [('primary85','primary_holdout85'), ('all93','all93'), ('development8','development8')]:
        first = scenarios[0]['cohorts'][key]
        original_summary = frozen[old_key]
        assert first['located_count'] == original_summary['eligible_count'] == first['subject_count']
        for n in NAMES:
            for field in ('hit_count', 'denominator', 'hit_rate', 'flat_count', 'boundary_extremum_count'):
                assert first['methods'][n][field] == original_summary['methods'][n][field]
        for n in NAMES[1:]:
            for field, value in first['comparisons'][n].items():
                assert value == original_summary['comparisons'][n][field]
            first['comparisons'][n]['p_holm_original_two'] = original_summary['comparisons'][n]['p_holm_two']
    family = [s['cohorts']['primary85']['comparisons'][n] for s in scenarios[1:] for n in NAMES[1:]]
    assert len(family) == 40
    for comparison, p in zip(family, holm_family([r['p_one_sided_exact'] for r in family])):
        comparison['p_holm_window_family40'] = float(p)
        comparison['significant_superiority'] = bool(comparison['rate_difference'] > 0 and p < .05)
    for scenario in scenarios:
        primary = scenario['cohorts']['primary85']
        scenario['qualifies'] = bool(scenario['k_each_side'] > 0 and primary['methods']['xi']['hit_rate'] > .5 and
            all(r['significant_superiority'] for r in primary['comparisons'].values()))
    qualified = [s for s in scenarios if s['qualifies']]
    return dict(status='complete', analysis_type='Posthoc fixed-dense-grid window sensitivity', plan=plan,
        scenarios=scenarios, smallest_qualifying_k=(qualified[0]['k_each_side'] if qualified else None),
        qualifying_k_values=[s['k_each_side'] for s in qualified],
        audit=dict(original_landmarks_unchanged=True, dense_grid_retained=True,
            k0_all_cohort_counts_and_paired_statistics_reproduced=True,
            subject_level_windows_nested=True, subject_level_hit_sets_nested=True,
            fixed_denominators=True, new_comparison_family_size=40, additional_simulations=0,
            syn_tolerance_nats=frozen['audit']['xi_nonnegative_audit']['tolerance_nats'],
            original_syn_audit=frozen['audit']['xi_nonnegative_audit'],
            original_EI_closure_max_nats=frozen['audit']['EI_closure_max_nats']))


def draw(result, frozen, contract):
    output = ROOT/'docs/reports/assets/dmf_subject_dense'
    output.mkdir(parents=True, exist_ok=True)
    selected = result['smallest_qualifying_k']
    figures = []
    with plt.rc_context({'font.family':'sans-serif', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42}):
        fig = plt.figure(figsize=(11.0, 4.6), layout='constrained')
        subs = fig.subfigures(1, 2, wspace=.05)
        axes = [s.subplots() for s in subs]
        ks = np.array(result['plan']['k_values'])
        for n, marker in zip(NAMES, ('o', 's', '^')):
            y = [100*s['cohorts']['primary85']['methods'][n]['hit_rate'] for s in result['scenarios']]
            axes[0].plot(ks, y, color=METHOD_COLORS[n], label=METHOD_LABELS[n], lw=1.4,
                marker=marker, ms=5 if n == 'phi_r' else 3.6,
                markerfacecolor='white' if n == 'phi_r' else METHOD_COLORS[n],
                ls='--' if n == 'phi_r' else '-')
        axes[0].axhline(50, color='#aaa', lw=.8, ls=':')
        axes[0].set(ylabel='Hit rate (primary 85, %)', ylim=(-1, 101))
        lo_all, hi_all = [], []
        for n, marker in zip(NAMES[1:], ('s', '^')):
            rows = [s['cohorts']['primary85']['comparisons'][n] for s in result['scenarios']]
            y = np.array([r['rate_difference'] for r in rows])*100
            ci = np.array([r['difference_ci95_percentile'] for r in rows])*100
            lo_all.extend(ci[:, 0]); hi_all.extend(ci[:, 1])
            axes[1].fill_between(ks, ci[:, 0], ci[:, 1], color=METHOD_COLORS[n], alpha=.13, lw=0)
            axes[1].plot(ks, y, color=METHOD_COLORS[n], label=f'Ξ − {LABELS[n]}',
                ls='--' if n == 'phi_r' else '-', marker=marker, ms=5 if n == 'phi_r' else 3.6,
                markerfacecolor='white' if n == 'phi_r' else METHOD_COLORS[n], lw=1.4)
        axes[1].axhline(0, color='#aaa', lw=.8)
        axes[1].set(ylabel='Paired hit-rate gain (percentage points)',
                    ylim=(min(-1, min(lo_all)-2), max(hi_all)+3))
        for i, (sub, ax) in enumerate(zip(subs, axes)):
            ax.set(xlabel='Added 0.1 intervals per side, k', xlim=(-.25,20.25), xticks=np.arange(0,21,2))
            if selected is not None:
                ax.axvline(selected, color='#787878', ls=':', lw=.9, zorder=0)
            ax.text(-.085,1.03,'ab'[i],transform=ax.transAxes,fontweight='bold')
            sub.legend(*ax.get_legend_handles_labels(), loc='outside upper center',
                       ncol=3 if i == 0 else 2, frameon=False)
        for ext in ('png','pdf'):
            name=f'window_sensitivity.{ext}'
            fig.savefig(output/name, dpi=240, bbox_inches='tight')
            figures.append(name)
        plt.close(fig)

        # The illustration subject is fixed by order, never chosen for a favorable hit.
        index = len(frozen['subject_rows'])//2
        subject = frozen['subject_rows'][index]['subject']
        row = frozen['subject_rows'][index]
        cache = BASE/'analysis_curves.npz'
        with np.load(cache) as a:
            if str(a['contract_sha256']) != digest(BASE/'contract.json'):
                raise ValueError('Illustration curve cache contract mismatch')
            g = a['G']; ids = a['subject_ids'].tolist()
            pi = ids.index(subject)
            y = {n:a[n][pi].mean(0) for n in NAMES}
            rate = a['rate'][pi].mean(0)
        recovered = subject_landmarks(g, rate, y)
        assert recovered['transition'] == row['transition']
        assert recovered['methods'] == row['methods']
        example_k = selected if selected is not None else 2
        expanded = expanded_subject(row, example_k)
        fig, axes = plt.subplots(2,1, figsize=(7.5,6.1), layout='constrained', sharex=True,
                                 gridspec_kw={'height_ratios':[1,1.25]})
        for ax in axes:
            ax.axvspan(*expanded['window']['interval'], color='#B9AB84', alpha=.22, lw=0, zorder=0)
            ax.axvspan(*row['transition']['interval'], color='#757A83', alpha=.3, lw=0, zorder=0)
            ax.set_xlim(-.03,4.03)
        axes[0].plot(g, rate, color='#252525', lw=1.25, marker='o', ms=2.4)
        axes[0].set(ylabel='Independent E rate (Hz)', title=subject)
        for n, marker in zip(NAMES, ('o','s','^')):
            if not np.isfinite(y[n]).all():
                raise ValueError('Nonfinite illustration curve')
            scale = float(y[n].std(ddof=0))
            if scale <= 1e-12*max(1.,float(np.abs(y[n]).max())):
                raise ValueError('Flat illustration curve')
            z=(y[n]-y[n].mean())/scale
            axes[1].plot(g,z,color=METHOD_COLORS[n],marker=marker,ms=2.5,lw=1.2)
            p=row['methods'][n]['extremum_G']; gi=int(np.flatnonzero(np.isclose(g,p))[0])
            axes[1].scatter([p],[z[gi]],color=METHOD_COLORS[n],s=38,marker='D',zorder=4)
        axes[1].axhline(0,color='#ddd',lw=.6,zorder=0)
        axes[1].set(xlabel='$G$ (all original 0.1 nodes retained)', ylabel='Metric shape (z score)')
        for ax,letter in zip(axes,'ab'):
            ax.text(-.10,1.03,letter,transform=ax.transAxes,fontweight='bold')
        handles=[Line2D([],[],color=METHOD_COLORS[n],label=METHOD_LABELS[n]) for n in NAMES]
        handles += [Patch(facecolor='#757A83',alpha=.3,label='Original rate-turn interval'),
                    Patch(facecolor='#B9AB84',alpha=.3,label=f'Expanded window (k={example_k})'),
                    Line2D([],[],color='#252525',marker='D',ls='none',label='Frozen extremum')]
        fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
        for ext in ('png','pdf'):
            name=f'window_example.{ext}'
            fig.savefig(output/name,dpi=240,bbox_inches='tight'); figures.append(name)
        plt.close(fig)
    result['illustration'] = dict(subject=subject, rule='Middle index of frozen93 ordering, no outcome selection',
        k_each_side=example_k, dense_curves_sha256=digest(cache),
        reconstructed_landmarks_match_frozen=True, z_scores='Display only; WMS signed; no peak change')
    return figures


def write_report(result):
    scenarios=result['scenarios']; selected=result['smallest_qualifying_k']
    table, pairs = [], []
    for s in scenarios:
        k=s['k_each_side']; p=s['cohorts']['primary85']; a=s['cohorts']['all93']
        hits = [f"{p['methods'][n]['hit_count']}/85（{100*p['methods'][n]['hit_rate']:.1f}%）" for n in NAMES]
        q=[p['comparisons'][n]['p_holm_original_two'] if k == 0 else p['comparisons'][n]['p_holm_window_family40'] for n in NAMES[1:]]
        all_hits=' / '.join(str(a['methods'][n]['hit_count']) for n in NAMES)
        table.append(f"| {k} | {s['nominal_width_G']:.1f} | "+' | '.join(hits)+
            ' | '+' / '.join(f'{v:.4g}' for v in q)+f" | {all_hits} | {p['clipped_subject_count']} / {a['clipped_subject_count']} |")
        for n in NAMES[1:]:
            c=p['comparisons'][n]; low,high=c['difference_ci95_percentile']
            q=c['p_holm_original_two'] if k == 0 else c['p_holm_window_family40']
            pairs.append(f"| {k} | Ξ−{LABELS[n]} | {100*c['rate_difference']:.1f} | "
                f"[{100*low:.1f}, {100*high:.1f}] | {c['xi_only']} / {c['comparator_only']} | "
                f"{c['p_one_sided_exact']:.4g} | {q:.4g} |")
    if selected is None:
        finding='在全部预先声明的k=0–20档位中，没有找到同时满足Ξ命中率严格超过50%且两项40检验Holm校正p<0.05的窗口。没有继续扩域或改阈值。'
        selected_text='不存在满足两项条件的最小窗口；所有未满足或不利结果仍完整列示。'
    else:
        s=scenarios[selected]; p=s['cohorts']['primary85']; a=s['cohorts']['all93']; d=s['cohorts']['development8']
        c=p['comparisons']['phi_r']; ci=c['difference_ci95_percentile']
        finding=(f"**最小合格窗口为左右各扩{selected}格，即各放宽{s['added_margin_G']:.1f} G，总宽度{s['nominal_width_G']:.1f} G。** "
            f"保留全部0.1格点和原峰位，主要85人Ξ为{p['methods']['xi']['hit_count']}/85（{100*p['methods']['xi']['hit_rate']:.1f}%），"
            f"ΦR/WMS各{p['methods']['phi_r']['hit_count']}/85（{100*p['methods']['phi_r']['hit_rate']:.1f}%）。"
            f"两项配对优势各{100*c['rate_difference']:.1f}个百分点，40检验Holm p均为{c['p_holm_window_family40']:.4g}。")
        selected_text=(f"该窗口两项比较的Ξ单独命中/基线单独命中均为{c['xi_only']}/{c['comparator_only']}，"
            f"共命中{c['both_hits']}人、共未命中{c['both_misses']}人；单侧精确McNemar p={c['p_one_sided_exact']:.4g}。"
            f"配对增幅的描述性95%bootstrap区间均为[{100*ci[0]:.1f}, {100*ci[1]:.1f}]个百分点，**未作多重或窗口选择校正**。"
            f"全部93人Ξ为{a['methods']['xi']['hit_count']}/93（{100*a['methods']['xi']['hit_rate']:.1f}%），"
            f"两基线各{a['methods']['phi_r']['hit_count']}/93（{100*a['methods']['phi_r']['hit_rate']:.1f}%）；"
            f"开发8人分别{d['methods']['xi']['hit_count']}/8、{d['methods']['phi_r']['hit_count']}/8、{d['methods']['wms']['hit_count']}/8。"
            f"后两队列仅补充描述，不能作为另一次独立验证。最小窗口无边界截断，总宽是原0.1区间的{1+2*selected}倍、"
            f"扫描域的{100*s['nominal_width_G']/4:.1f}%；主要85人的窗口/转折中点比值中位数为"
            f"{100*p['width_relative_to_transition_midpoint']['median']:.1f}%（范围"
            f"{100*p['width_relative_to_transition_midpoint']['minimum']:.1f}%–"
            f"{100*p['width_relative_to_transition_midpoint']['maximum']:.1f}%）。因此‘附近命中’没有恢复成0.1精确定位。")
    lead=f'''<!-- dense-window-lead:start -->
### 最新结果：保持0.1网格，对称放宽转折窗口

{finding}

当前优先解释以这一细网格窗口分析为准，原0.1精确区间检验与此前粗化敏感性均保留为独立记录；新窗口是看过结果后选择的容许误差超参，不能包装为预注册确认性终点。详细方案、全部档位与区间尺度见本附录末尾。
<!-- dense-window-lead:end -->'''
    body=f'''<!-- dense-window-review:start -->
<a id="dmf-subject-window"></a>

### 固定细网格上的对称窗口敏感性（2026-10-04）

{finding}

{selected_text}

用户在观察前次分辨率结果后提出本分析，并要求2026-10-04 06:00北京时间一次性实施。计算前记录[完整计划](../../results/dmf_schaefer100/subject_curves_93_dense/window_plan.json)：k=0–20的全部整数，三项共用同一独立转折窗口。对原最大正斜率区间[L,U]，窗口为[max(0,L−0.1k),min(4,U+0.1k)]；名义宽度0.1(1+2k)，真实宽度和截断人数单独保存。峰位、三seed平均、平坦/边界分类、原区间和0.1采样均冻结，含端点时采用原10⁻¹² G浮点容差。ΦR取全局峰，WMS取保留符号的全局谷。未重算模拟或估计器；没有平滑、插值、粗采样或按指标决定窗口。

![固定细网格上的原区间与扩充窗口](assets/dmf_subject_dense/window_example.png)

**窗口示意。** 个体按冻结93人顺序的中间索引选定（{result['illustration']['subject']}），没有按命中选择案例。上为独立发放率，下为三指标跨G的z score，仅作形状显示，保留WMS符号。全部41个0.1节点保留；菱形为原极值，灰带为原最大斜率区间，浅色带为k={result['illustration']['k_each_side']}的扩充窗口。背景宽度为容许范围，不是估计不确定性。缓存重建的原转折与极值逐项吻合冻结结果。

![命中率与配对优势随窗口扩充变化](assets/dmf_subject_dense/window_sensitivity.png)

**全部预定窗口。** 左为主要85人命中率，50%横线及最小合格k的竖线只标示选择规则；右为Ξ减去各基线的配对百分点差及未校正95%被试共同bootstrap区间。覆盖21个k，未删除大窗口饱和结果。三指标命中集合逐人嵌套，命中率全部单调不降；配对优势和显著性不具有这一单调性。两基线在k≥2的命中数和配对差相等，蓝色空方块与灰色三角用于显示重合，而非删去一条曲线。

#### 全部窗口命中与宽度代价

主要分母始终85；全部93人及开发8人的成员也固定。原始0.1独立转折在93人均可定位，本次没有重新定位，所有k无新增排除。表中的Holm两值依次对ΦR、WMS；k=0沿用原两项校正，k≥1的40项检验一起校正。

| 每侧扩k格 | 名义宽度G | Ξ主要85人 | ΦR主要85人 | WMS主要85人 | Holm p：ΦR / WMS | 全93人命中：Ξ / ΦR / WMS | 截断人数：85 / 93 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

满足两项条件的k为{result['qualifying_k_values']}，最小为{selected}。k=1的Ξ为37/85（43.5%），还未超过50%；k=2为68.2%。k=3、4、5时Ξ分别78.8%、90.6%、94.1%，但名义宽度也分别增加到0.7、0.9、1.1，配对增幅从35.3个百分点降至24.7、16.5个百分点。k=6时Ξ96.5%、两基线87.1%，校正p=0.1172，不再显著。k=13起三项在85人及93人均100%命中、差值为0：过宽窗口会机械地覆盖所有极值，不能将这种饱和当作定位能力。

端点截断从全93人k=7开始出现；主要85人同档2人被截断。所有人的真实上下界、真实宽度、单侧截断以及窗口相对独立转折中点的比例保存在汇总中。名义宽度到4.1并不表示每人的真实窗口覆盖整个扫描域；不能以名义宽度代替裁切后的个人宽度。

#### 全部配对效应与不确定性

同一被试为独立比较单位；G点、seed和ROI对不增加被试数。沿用单侧精确McNemar（不一致对子上的二项检验），备择是Ξ命中概率高于基线。新家族共20个k×2基线=40项，统一Holm控制，阈值0.05。95%CI为每档同人共同重采样10000次、seed20261003的双侧百分位区间，仅作描述；未做同时覆盖或最小合格窗口选择校正。CI不能代替40项校正p值判断显著性。“高很多”没有另造效果量阈值，实际配对百分点差及CI全部给出。

| k | 比较 | 配对增幅（百分点） | 描述性95%CI（未校正） | Ξ单独 / 基线单独命中 | 单侧精确p | Holm p |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(pairs)}

原k=0不仅复现主要85人15/85、4/85、4/85，还逐项复现全部93人与开发8人的命中、配对列联数、原bootstrap CI及原精确p。检查了每人的窗口包含关系、命中集合单调性、峰位/内部极值/平坦标签不变及固定分母。原summary.json、completed.json、contract.json与前次分辨率计划/汇总哈希均未变化；没有读取11,562个条件文件或追加模拟。仅用于示意的已提取曲线缓存约0.6MB。

#### 解释边界与来源

窗口宽度是在已经看过结果的同一队列上选择，40项校正控制本次声明家族的错误率，但不把该选择变成独立验证；选中窗口处的描述性CI也不是选择校正区间。此结果支持当前固定SC＋DMF和原生估计协议下，Ξ极值更接近各人的独立发放率转折，并以每侧0.2 G容许范围满足样本命中率超过50%。尚无独立确认无转变对照，不能据此证明真实相变点、无相变不报峰、一般生物学相变识别能力或未见队列性能。对不同人的相对转折尺度，固定0.5窗口并非同样严格。

本次重新通过Zotero核实父条目P6UJCVG8的标题和附件；唯一索引正文为DXGC7JEA、19页，读了Brain/Fig.2及Methods式（5）、（7）、（8）。无明确稿件日期/版本，条目元数据版本5375不当作稿件版本；补充附录仍不可用。沿用同一因子化干预、全200维未来、300ms和共同affine-TM密度定义，Ξ非负容差仍10⁻⁸ nats，无数值投影。保留已记录差异：正文把两观测基线称为BOLD-like，原生WMS实际用完整E/I自然状态；本任务不变更估计器。

[全量窗口与逐人结果](../../results/dmf_schaefer100/subject_curves_93_dense/window_summary.json) · [完成审核](../../results/dmf_schaefer100/subject_curves_93_dense/window_review.json)。重现：`.venv/bin/python -m scripts.analyze_dmf_subject_dense_windows`。
<!-- dense-window-review:end -->'''
    report=ROOT/'docs/reports/brain.md'; current=report.read_text()
    start='<!-- report-section:dmf-subject-dense:start -->'; end='<!-- report-section:dmf-subject-dense:end -->'
    if current.count(start)!=1 or current.count(end)!=1:
        raise ValueError('Expected one canonical dense report section')
    a,b=current.index(start),current.index(end)
    section=current[a:b]
    for name in ('dense-window-lead','dense-window-review'):
        lo,hi=f'<!-- {name}:start -->',f'<!-- {name}:end -->'
        if lo in section:
            if section.count(lo)!=1 or section.count(hi)!=1:
                raise ValueError('Repeated window report block')
            section=section[:section.index(lo)]+section[section.index(hi)+len(hi):]
    marker='## 附录 P：93 人 DMF 细扫描\n'
    if section.count(marker)!=1:
        raise ValueError('Dense report heading missing')
    section=section.replace(marker,marker+'\n'+lead+'\n',1).rstrip()+'\n\n'+body+'\n'
    updated=current[:a]+section+current[b:]
    assert updated[:a]==current[:a] and updated.endswith(current[b:])
    report.write_text(updated)
    return dict(path='docs/reports/brain.md#dmf-subject-window',
                full_report_sha256=digest(report), window_block_sha256=hashlib.sha256(body.encode()).hexdigest())


def main():
    plan = json.loads((BASE/'window_plan.json').read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(BASE/name) != expected:
            raise ValueError(f'Frozen or previous analysis changed: {name}')
    frozen = json.loads((BASE/'summary.json').read_text())
    contract = json.loads((BASE/'contract.json').read_text())
    result = compute(plan, frozen, contract)
    result['plan_sha256'] = digest(BASE/'window_plan.json')
    result['implementation_sha256'] = digest(Path(__file__))
    result['dependency_sha256']={name:digest(ROOT/name) for name in [
        'scripts/dmf_dense_statistics.py','scripts/analyze_dmf_subject_dense_resolution.py']}
    assert result['dependency_sha256']['scripts/dmf_dense_statistics.py'] == contract['implementation_sha256']['scripts/dmf_dense_statistics.py']
    figures=draw(result,frozen,contract)
    atomic_json(BASE/'window_summary.json', result)
    report=write_report(result)
    for name,expected in plan['input_sha256'].items():
        assert digest(BASE/name)==expected, f'Frozen input changed after analysis: {name}'
    atomic_json(BASE/'window_review.json', dict(status='analysis_complete_visual_review_pending',
        original_inputs_preserved=plan['input_sha256'], plan_sha256=digest(BASE/'window_plan.json'),
        summary_sha256=digest(BASE/'window_summary.json'), implementation_sha256=digest(Path(__file__)),
        figures_sha256={n:digest(ROOT/'docs/reports/assets/dmf_subject_dense'/n) for n in figures},
        report=report, original_dense_completed_status='complete',
        audit=result['audit'], smallest_qualifying_k=result['smallest_qualifying_k']))
    for s in result['scenarios']:
        p = s['cohorts']['primary85']
        print(json.dumps(dict(k=s['k_each_side'], width=s['nominal_width_G'],
            hits85={n:r['hit_count'] for n,r in p['methods'].items()},
            hits93={n:r['hit_count'] for n,r in s['cohorts']['all93']['methods'].items()},
            p_holm40={n:r.get('p_holm_window_family40') for n,r in p['comparisons'].items()},
            gains={n:r['rate_difference'] for n,r in p['comparisons'].items()},
            clipping85=p['clipped_subject_count'], qualifies=s['qualifies'])))
    print('Smallest qualifying k:', result['smallest_qualifying_k'])


if __name__ == '__main__':
    main()
