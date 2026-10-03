#!/usr/bin/env python3
"""Posthoc resolution sensitivity on complete caches; frozen primary is untouched."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_dmf_subject_dense import load_complete
from scripts.dmf_curve_shape import METHOD_COLORS, METHOD_LABELS
from scripts.dmf_dense_statistics import compare_cohort, paired_hits, subject_landmarks
from scripts.run_dmf_subject_consistency import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/dmf_schaefer100/subject_curves_93_dense'
NAMES = ('xi', 'phi_r', 'wms')


def holm_family(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected a nonempty valid p-value family')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return adjusted


def joint_cohort(rows):
    """Shared failures to locate stay in the fixed-cohort denominator."""
    conditional = compare_cohort(rows)
    hits = {n: [r['methods'][n]['hit'] is True for r in rows] for n in NAMES}
    return dict(subject_count=len(rows), located_count=conditional['eligible_count'],
        unlocated_subjects=conditional['unlocated_subjects'],
        methods={n: dict(hit_count=sum(hits[n]), denominator=len(rows),
                         joint_hit_rate=sum(hits[n])/len(rows)) for n in NAMES},
        comparisons={n: paired_hits(hits['xi'], hits[n]) for n in NAMES[1:]},
        conditional_on_located=conditional)


def merged_rows(dense_rows, g, stride, origin=0):
    """Merge native intervals, preserving dense extrema and independent location."""
    rows = copy.deepcopy(dense_rows)
    for r in rows:
        if not r['transition']['located']:
            continue
        i = int(np.argmin(np.abs(g-r['transition']['interval'][0])))
        start = ((i-origin)//stride)*stride+origin
        low, high = float(g[max(0, start)]), float(g[min(len(g)-1, start+stride)])
        r['transition'] = dict(r['transition'], interval=[low, high],
            midpoint=(low+high)/2, reason='Dense independent interval in a common merged G block')
        for n in NAMES:
            m = r['methods'][n]
            p = m['extremum_G']
            dist = None if p is None else max(low-p, 0., p-high)
            m.update(distance_to_interval_G=dist,
                     hit=bool(m['interior'] and dist is not None and dist <= 1e-12))
    return rows


def sliced_rows(c, mean, indices):
    g = np.asarray(c['G'])[indices]
    return [dict(subject_landmarks(g, mean['rate'][pi, indices],
        {n: mean[n][pi, indices] for n in NAMES}), subject=subject)
        for pi, subject in enumerate(c['subject_ids'][:-1])]


def summarize_components(c, mean):
    tol = 1e-8  # nats per adjacent G step; descriptive only, no projection.
    out = dict(adjacent_increase_tolerance_nats=tol, no_projection=True)
    for n in ('whole_ei', 'partial_ei_sum'):
        y = mean[n][:-1]
        dy = np.diff(y, axis=1)
        rising = dy > tol
        out[n] = dict(nonincreasing_subject_count=int(np.sum(~rising.any(1))),
            increasing_subject_count=int(np.sum(rising.any(1))),
            increasing_step_count=int(rising.sum()), total_steps=int(dy.size),
            maximum_increase_nats=float(dy.max()),
            maximum_cumulative_increase_nats=float(np.where(rising, dy, 0).sum(1).max()),
            nonincreasing_from_G0_5_subject_count=int(np.sum(~rising[:, np.asarray(c['G'])[:-1] >= .5].any(1))),
            maximum_increase_from_G0_5_nats=float(dy[:, np.asarray(c['G'])[:-1] >= .5].max()),
            subjects_with_increases=[s for s, yes in zip(c['subject_ids'][:-1], rising.any(1)) if yes])
    e = np.diff(mean['whole_ei'][:-1], axis=1)
    s = np.diff(mean['partial_ei_sum'][:-1], axis=1)
    x = np.diff(mean['xi'][:-1], axis=1)
    out['difference_of_increments_max_error_nats'] = float(np.max(np.abs(e-s-x)))
    if out['difference_of_increments_max_error_nats'] > tol:
        raise ArithmeticError('EI increment identity does not close')
    return out


def write_review(result):
    """Add an explicitly posthoc interpretation without rewriting frozen evidence."""
    frozen = json.loads((BASE/'summary.json').read_text())
    complete = json.loads((BASE/'completed.json').read_text())
    contract = json.loads((BASE/'contract.json').read_text())
    assert digest(BASE/'contract.json') == complete['contract_sha256']
    assert digest(BASE/'summary.json') == complete['summary_sha256']
    assert len(list((BASE/'conditions').glob('*.npz'))) == complete['condition_count'] == 11562
    report = ROOT/'docs/reports/brain.md'
    old_review = json.loads((BASE/'final_review.json').read_text()) if (BASE/'final_review.json').exists() else {}
    verification = json.loads((BASE/'completion_verification.json').read_text()) if (BASE/'completion_verification.json').exists() else {}
    original_report_verified = (digest(report) == complete['report_sha256'] or
                                old_review.get('original_report_snapshot_verified', False) or
                                (verification.get('original_report_snapshot_verified', False) and
                                 verification.get('verified_report_sha256') == complete['report_sha256'] and
                                 verification.get('original_completed_sha256') == digest(BASE/'completed.json')))
    if not original_report_verified:
        raise ValueError('Completion report snapshot has not been verified before editorial changes')
    analyzer = ROOT/'scripts/analyze_dmf_subject_dense.py'
    text = analyzer.read_text()
    reconstructed = text.replace('from scripts.report_sections import write_report_section\n', '').replace(
        "write_report_section(path, 'dmf-subject-dense', f'''", "path.write_text(f'''").replace(
        '(brain.md#dmf-subject-curves)', '(brain_dmf_subject_curve_comparison.md)').replace(
        "ROOT/'docs/reports/brain.md'", "ROOT/'docs/reports/brain_dmf_subject_dense.md'")
    original_analysis_hash = hashlib.sha256(reconstructed.encode()).hexdigest()
    expected = contract['implementation_sha256']['scripts/analyze_dmf_subject_dense.py']
    assert original_analysis_hash == expected
    code_audit = {}
    for name, sha in contract['implementation_sha256'].items():
        current = digest(ROOT/name)
        assert current == sha or (name == 'scripts/analyze_dmf_subject_dense.py' and original_analysis_hash == sha)
        code_audit[name] = dict(frozen_sha256=sha, current_sha256=current,
                               literal_match=current == sha, presentation_only_reconstruction=(current != sha))
    assets = ROOT/'docs/reports/assets/dmf_subject_dense'
    for name, sha in complete['figures_sha256'].items():
        assert digest(assets/name) == sha
    table = []
    for kind, label in [('coarse_rescan', '重算粗网格'), ('merged_intervals', '仅合并区间')]:
        for s in result['scenarios']:
            if s['kind'] != kind:
                continue
            row = s['cohorts']['primary85']
            rates = [f"{row['methods'][n]['hit_count']}/85（{100*row['methods'][n]['joint_hit_rate']:.1f}%）" for n in NAMES]
            allhits = ' / '.join(str(s['cohorts']['all93']['methods'][n]['hit_count']) for n in NAMES)
            pvals = ([frozen['primary_holdout85']['comparisons'][n]['p_holm_two'] for n in NAMES[1:]]
                     if s['nominal_width_G'] == .1 else [row['comparisons'][n]['p_holm_resolution_family20'] for n in NAMES[1:]])
            table.append(f"| {label} | {s['nominal_width_G']:g} | "+' | '.join(rates)+
                         ' | '+' | '.join(f'{p:.4g}' for p in pvals)+f' | {allhits} |')
    origins = []
    for s in result['scenarios']:
        if s['kind'] == 'merged_intervals' and s['nominal_width_G'] > .1:
            bounds = [f"{min(o['cohorts']['primary85'][n] for o in s['all_origin_hit_counts'])}–"
                      f"{max(o['cohorts']['primary85'][n] for o in s['all_origin_hit_counts'])} / 85" for n in NAMES]
            origins.append(f"| {s['nominal_width_G']:g} | "+' | '.join(bounds)+' |')
    component = result['ei_components_all93']
    lead = '''<!-- dense-final-interpretation:start -->
### 完成后的比较结论

11,562/11,562 条件及自动分析全部完成。主要新增85人中，0.1细网格的Ξ命中15/85（17.6%），ΦR/WMS各4/85（4.7%）；两项优势均为12.9个百分点，Holm p=0.0009766。**细分辨率下相对优势成立，但绝对定位命中率仍低。** 全部93人分别为18/93、4/93、5/93。Ξ的75/93个峰早于独立转折区间，平均中点偏移−0.258 G；ΦR/WMS分别为89/93、88/93个极值偏早，平均偏移约−0.436、−0.434 G。Ξ更接近转折，并不等于与转折精确重合。

按用户随后提出的粗化方案复用全部数据：0.8粗网格重新定位后，主要85人Ξ达到70/85（82.4%），ΦR为46/85（54.1%）、WMS为55/85（64.7%）；跨本次20项新比较的Holm p分别为1.97×10⁻⁵、0.001221。该结果支持**当前模型与原生协议下，较粗尺度上的转折对应更好**；它是事后分辨率敏感性结果，不能替代原0.1主要检验，也不证明精确相变定位。所有分辨率均在下文报告。

形状优势比精确定位更明显：93条均值Ξ曲线中89条严格满足固定弱单峰容差，余下4条最大回摆仅为全曲线幅度的0.00179%；没有任何Ξ曲线出现两个5% prominence明显峰。ΦR/WMS的平均额外回摆为24.13%/16.21%，78/93及58/93人有多个明显峰，Q均值约91.19/92.50，Ξ为15.68。当前协议下Ξ更接近单峰、回摆更少；短轨迹及正则化使我们不能把基线波动直接归为公式本身失效。

估计阶段每条件平均Ξ/ΦR/WMS耗时91.6/26.2/144.4 ms；Ξ比WMS少约36.5%，ΦR最快。计入必需样本生成后分别为5.77/0.133/5.67 s，Ξ没有完整流程成本优势。Ξ与WMS估计同为O(Nd²+d³)，当前100ROI实测没有识别渐近指数。
<!-- dense-final-interpretation:end -->'''
    supplement = f'''<!-- dense-resolution-review:start -->
### 分辨率敏感性：粗定位、区间合并与边界起点

本节是看过0.1结果后、由用户明确要求的事后分析。没有新增模拟、改估计器、改干预或选择被试；原始summary/completed/contract保持不变。计算前将全部方案记入[分辨率分析计划](../../results/dmf_schaefer100/subject_curves_93_dense/resolution_plan.json)。宽度为0.1、0.2、0.4、0.5、0.8、1.0：这些是原生0.1网格中能整除0–4扫描范围且保留至少5个节点的全部整数步长。每项均先平均同人的3seed，再找极值。

**重算粗网格：** 从G=0开始保留已有观测格点，重新以独立序参量最大正斜率定位区间，并在相同粗格点寻找Ξ/ΦR全局峰和WMS全局谷。没有插值、平滑或移动待比较指标峰去迎合转折。**仅合并区间：** 保留细网格的独立转折和精确极值，把包含该独立转折的相邻0.1区间按共同G=0起点合并；三项使用完全相同的新区间。因此第二项只改变容许分辨率，第一项还包含粗采样对峰位和最大斜率的量化。

![分辨率敏感性](assets/dmf_subject_dense/resolution_sensitivity.png)

图中为主要85人的命中率；两种分析的全部宽度均能定位全部85人，所以主要检验的固定队列分母与条件于可定位者的分母一致。全部93人的补充中，仅重算1.0粗网格有1人转折落在边界、92人可定位；表中其补充命中人数仍以固定93人为队列，另存条件于92人的命中率。其他档位全部93人可定位。无法定位不写成无相变。0.1行保留原两项Holm校正；其余行对5宽度×2处理×2基线共20项主要85人比较一起作Holm校正。全部93人是补充描述，不再用其p值宣称独立验证。

| 处理 | 宽度G | Ξ：主要85人 | ΦR：主要85人 | WMS：主要85人 | 对ΦR的Holm p | 对WMS的Holm p | 全部93人命中数：Ξ / ΦR / WMS |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

命中率**并非越粗越高**。例如重算粗网格从0.8到1.0，Ξ由82.4%降至69.4%，WMS由64.7%降至51.8%；最大斜率区间及极值格点都可能变化。仅合并区间到1.0时，Ξ为88.2%，两基线均为82.4%，配对优势仅5/85，20项校正p=0.0625，**高绝对命中率没有保留显著差异**。到0.5，重算粗网格Ξ为69.4%、两基线31.8%/32.9%，跨20项校正仍显著；到0.8则为82.4%、54.1%/64.7%。所有宽度一起显示，不按最高命中率挑选唯一“正确”尺度。

共同区间起点会影响仅合并结果。下表遍历每个宽度的所有0.1格点起点偏移，保留完整原扫描支持，并给出命中人数范围；范围内每个点均来自同一93人缓存。这里只报告起点敏感性，不根据起点的显著性选值，也不对这些界限给出新的确认性p值。

| 宽度G | Ξ主要85人：各起点范围 | ΦR主要85人 | WMS主要85人 |
|---|---:|---:|---:|
{chr(10).join(origins)}

原8人粗网格G=[0,0.5,1,1.3,1.6,2.2,3]也已在同一完整缓存上复现：7人可定位，Ξ5/7（71.4%）、两基线各2/7（28.6%）；将边界无法定位者保留在全部8人分母则为5/8、2/8、2/8。该小样本当时的两项Holm p=0.25，不能把高点估计当作已确立的显著优势。细扫描既缩窄区间，也将支持扩到4；两批命中率差不能只归因于分辨率或只归因于新增被试。

### 全量EI分解的补充解释

视觉上的总体下降并非严格全域单调。以每相邻G步增加超过10⁻⁸ nats计，whole EI有{component['whole_ei']['increasing_subject_count']}/93人出现增加，最大单步{component['whole_ei']['maximum_increase_nats']:.3f} nats；部分EI之和有{component['partial_ei_sum']['increasing_subject_count']}/93人出现增加，最大单步{component['partial_ei_sum']['maximum_increase_nats']:.3f} nats。主要是低G的轻微上升，另有4人在高G低信息尾部出现小幅回升；从G≥0.5开始，两分量各89/93人均不再上升。没有为了呈现单调而平滑或投影。

在实际有限格点上逐步验证ΔΞ=Δwhole EI−Δ部分EI之和，最大误差{component['difference_of_increments_max_error_nats']:.2g} nats。在两分量同时下降的区段，部分EI之和下降更快时差值Ξ上升；whole EI下降更快时Ξ下降。因此峰来自两项变化速度的交替，而不是whole EI本身必须有峰。此差值恒等式解释当前共同affine-TM估计中的峰，不单独证明相变机制。

### 完成证据与方法边界

11562个缓存全部通过有限性、状态支持、Ξ非负容差10⁻⁸ nats及EI闭合审计，原18张PNG与完成记录的哈希一致，已逐页检查图例位于数据外。原主分析脚本在运行期间随文档归并发生4处展示变化（报告路径、章节写入函数、导入和链接）；将这4处恢复后的源文本SHA256与冻结脚本完全相同，科学分析逻辑未改。完成时的brain.md哈希也与completed.json匹配；本节后续编辑另存[最终审核](../../results/dmf_schaefer100/subject_curves_93_dense/final_review.json)，不覆盖原完成证据。每小时临时检查已删除。

稿件核对沿用本任务新读取的P6UJCVG8/DXGC7JEA正文、Methods式（5）（7）（8）及Brain/Fig.2；版本日期和补充附录仍无法确定。当前“命中”是对应固定DMF发放率最大变化区间，既非独立生物相变真值，也非“无相变不报峰”的检验。尚无确认无转变对照；高命中与低回摆不足以证明普遍相变识别能力。

完整逐人、逐宽度、起点和配对结果：[事后分析汇总](../../results/dmf_schaefer100/subject_curves_93_dense/resolution_summary.json)。重现本节：`.venv/bin/python -m scripts.analyze_dmf_subject_dense_resolution`。高成本模拟无需重跑。
<!-- dense-resolution-review:end -->'''
    current = report.read_text()
    start, end = '<!-- report-section:dmf-subject-dense:start -->', '<!-- report-section:dmf-subject-dense:end -->'
    a, b = current.index(start), current.index(end)
    section = current[a:b]
    for label in ('dense-final-interpretation', 'dense-resolution-review'):
        lo, hi = f'<!-- {label}:start -->', f'<!-- {label}:end -->'
        if lo in section:
            first, last = section.index(lo), section.index(hi)+len(hi)
            section = section[:first]+section[last:]
    marker = '## 附录 P：93 人 DMF 细扫描\n'
    assert section.count(marker) == 1
    section = section.replace(marker, marker+'\n'+lead+'\n', 1).rstrip()+'\n\n'+supplement+'\n'
    # Verify unaffected sections are preserved byte for byte.
    updated = current[:a]+section+current[b:]
    assert updated[:a] == current[:a] and updated.endswith(current[b:])
    report.write_text(updated)
    review = dict(status='complete', condition_count=11562, temporary_automation='93-dmf deleted',
        original_report_snapshot_verified=original_report_verified,
        original_completed_sha256=digest(BASE/'completed.json'),
        frozen_contract_sha256=digest(BASE/'contract.json'), frozen_summary_sha256=digest(BASE/'summary.json'),
        frozen_source_audit=code_audit, original_figure_hashes_verified=complete['figures_sha256'],
        additional_figure_hashes={n:digest(assets/n) for n in ['resolution_sensitivity.png', 'resolution_sensitivity.pdf']},
        visual_review='All 12 subject pages, six original summary figures and new resolution figure checked; legends outside data.',
        canonical_report='docs/reports/brain.md#dmf-subject-dense', current_report_sha256=digest(report),
        current_dense_section_sha256=hashlib.sha256(section.encode()).hexdigest(),
        resolution_summary_sha256=digest(BASE/'resolution_summary.json'),
        resolution_implementation_sha256=digest(Path(__file__)))
    atomic_json(BASE/'final_review.json', review)


def main():
    plan = json.loads((BASE/'resolution_plan.json').read_text())
    frozen = json.loads((BASE/'summary.json').read_text())
    c, v, audit, _ = load_complete(BASE)
    mean = {n: y.mean(1) for n, y in v.items()}
    g = np.asarray(c['G'])
    dense_rows = sliced_rows(c, mean, np.arange(len(g)))
    # Meaningful control: reconstruct every original endpoint and paired primary.
    for a, b in zip(dense_rows, frozen['subject_rows']):
        assert a['subject'] == b['subject']
        assert a['transition'] == b['transition']
        assert a['methods'] == b['methods']
    cohorts = {'primary85': set(c['holdout_subject_ids']),
               'all93': set(c['subject_ids'][:-1]),
               'development8': set(c['development_subject_ids'])}
    scenarios = []
    for width in plan['widths_G']:
        stride = int(round(width/.1))
        assert (len(g)-1) % stride == 0
        for kind in ('coarse_rescan', 'merged_intervals'):
            rows = (sliced_rows(c, mean, np.arange(0, len(g), stride))
                    if kind == 'coarse_rescan' else merged_rows(dense_rows, g, stride))
            scenario = dict(kind=kind, nominal_width_G=width, origin_G=0.,
                G_nodes=g[::stride].tolist(), subject_rows=rows,
                cohorts={k: joint_cohort([r for r in rows if r['subject'] in ids])
                         for k, ids in cohorts.items()})
            if kind == 'merged_intervals':
                origin_counts = []
                for origin in range(stride):
                    rr = merged_rows(dense_rows, g, stride, origin)
                    origin_counts.append(dict(origin_G=float(g[origin]),
                        cohorts={k: {n: sum(r['methods'][n]['hit'] is True for r in rr
                                           if r['subject'] in ids) for n in NAMES}
                                 for k, ids in cohorts.items()}))
                scenario['all_origin_hit_counts'] = origin_counts
            scenarios.append(scenario)
    family = [s['cohorts']['primary85']['comparisons'][n]
              for s in scenarios if s['nominal_width_G'] > .1 for n in NAMES[1:]]
    assert len(family) == 20
    adjusted = holm_family([r['p_one_sided_exact'] for r in family])
    for r, q in zip(family, adjusted):
        r['p_holm_resolution_family20'] = float(q)
    historical_indices = np.array([np.flatnonzero(np.isclose(g, z))[0] for z in plan['historical_G']])
    historical = sliced_rows(c, mean, historical_indices)
    historical = dict(G=plan['historical_G'], subject_rows=historical,
        cohorts={k: joint_cohort([r for r in historical if r['subject'] in ids])
                 for k, ids in cohorts.items()})
    offsets = {}
    for n in NAMES:
        off = np.array([r['methods'][n]['extremum_G']-r['transition']['midpoint'] for r in dense_rows])
        offsets[n] = dict(mean_G=float(off.mean()), median_G=float(np.median(off)),
            before_interval_count=sum(r['methods'][n]['extremum_G'] < r['transition']['interval'][0]-1e-12 for r in dense_rows),
            after_interval_count=sum(r['methods'][n]['extremum_G'] > r['transition']['interval'][1]+1e-12 for r in dense_rows))
    result = dict(status='complete', type='posthoc_resolution_sensitivity',
        contract_sha256=digest(BASE/'contract.json'), frozen_summary_sha256=digest(BASE/'summary.json'),
        plan_sha256=digest(BASE/'resolution_plan.json'), implementation_sha256=digest(Path(__file__)),
        original_primary_reproduced=True, audit=audit, plan=plan, scenarios=scenarios,
        historical_pilot_grid=historical, dense_offsets_all93=offsets,
        ei_components_all93=summarize_components(c, mean))
    atomic_json(BASE/'resolution_summary.json', result)
    np.savez_compressed(BASE/'analysis_curves.npz', contract_sha256=result['contract_sha256'],
                        G=g, subject_ids=np.array(c['subject_ids']), **v)
    output = ROOT/'docs/reports/assets/dmf_subject_dense'
    with plt.rc_context({'font.family':'sans-serif', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout='constrained', sharey=True)
        for ax, kind, title, letter in zip(axes, ('coarse_rescan', 'merged_intervals'),
            ('Recompute on coarser observed grids', 'Merge intervals; retain dense extrema'), 'ab'):
            chosen = [s for s in scenarios if s['kind'] == kind]
            for n, marker in zip(NAMES, ('o','s','^')):
                ax.plot([s['nominal_width_G'] for s in chosen],
                    [100*s['cohorts']['primary85']['methods'][n]['joint_hit_rate'] for s in chosen],
                    color=METHOD_COLORS[n], marker=marker, ms=5 if n == 'phi_r' else 4,
                    markerfacecolor='white' if n == 'phi_r' else METHOD_COLORS[n],
                    ls='--' if n == 'phi_r' else '-', lw=1.4, label=METHOD_LABELS[n])
            ax.set(xlabel='G interval width', title=title, xticks=plan['widths_G'], ylim=(-2,102))
            ax.text(-.09, 1.05, letter, transform=ax.transAxes, fontweight='bold')
        axes[0].set_ylabel('Hit rate / all 85 subjects (%)')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False)
        fig.savefig(output/'resolution_sensitivity.png', dpi=240, bbox_inches='tight')
        fig.savefig(output/'resolution_sensitivity.pdf', bbox_inches='tight')
        plt.close(fig)
    write_review(result)
    for s in scenarios:
        co=s['cohorts']['primary85']; a=s['cohorts']['all93']
        print(json.dumps(dict(kind=s['kind'],width=s['nominal_width_G'],located85=co['located_count'],
            hits85={n:r['hit_count'] for n,r in co['methods'].items()},
            hits93={n:r['hit_count'] for n,r in a['methods'].items()},
            holm20={n:r.get('p_holm_resolution_family20') for n,r in co['comparisons'].items()})))
    print('historical development8',json.dumps(historical['cohorts']['development8'],ensure_ascii=False))
    print('EI components',json.dumps(result['ei_components_all93'],ensure_ascii=False))


if __name__ == '__main__':
    main()
