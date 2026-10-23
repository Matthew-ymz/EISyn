#!/usr/bin/env python3
"""Matched per-participant selection, cohort rankings and leave-one-person-out checks.

Reuses frozen simulations. The only new scores are the three existing SC metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_critical_coalitions_all93 import (
    M, empty_audit, mask_members, read_subject, word_mask,
)
from scripts.analyze_dmf_sc_topology_comparison import (
    BASE, INPUT, METRIC_NAMES, combination_index, combinations, load_data,
    save_json, score_arrays, sha, verify_metrics,
)
from scripts.run_dmf_paired_spt_pilot import paired_sources

OUT = ROOT / 'results/dmf_schaefer100/sc_spt_participant_frequency'
FIG = ROOT / 'fig/dmf_schaefer100/sc_topology_comparison/sc_spt_participant_frequency.png'
REPORT = ROOT / 'docs/reports/dmf_sc_spt_participant_frequency.md'
METHODS = ['spt', *METRIC_NAMES]
LABELS = ['SPT', 'Mean internal weight', 'Weighted clique intensity', 'Modularity contribution']
SEED = 20261023


def priority(group, subject=None):
    """Independent of method, scores and SPT identity; stable across Python runs."""
    text = f'{SEED}|{subject or "cohort"}|' + ','.join(map(str, group))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=16).digest(), 'big')


def choose(scores, groups, budget, subject):
    if not 0 <= budget <= len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores or selection budget')
    audit = dict(budget=budget, all_constant=bool(scores.min() == scores.max()),
                 zero_score_count=int((scores == 0).sum()), boundary_score=None,
                 boundary_tie_count=0, boundary_selected=0, zero_boundary_selected=False,
                 boundary_randomization_needed=False)
    if budget == 0:
        return [], audit
    threshold = float(np.partition(scores, len(scores) - budget)[len(scores) - budget])
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    remaining = budget - len(above)
    if len(tied) > remaining:
        tied = np.array(sorted(tied, key=lambda i: (priority(groups[i], subject), int(i))))
    chosen = np.sort(np.r_[above, tied[:remaining]])
    if len(chosen) != budget:
        raise ArithmeticError('SC selection budget was not matched')
    audit.update(boundary_score=threshold, boundary_tie_count=len(tied),
                 boundary_selected=remaining, zero_boundary_selected=bool(threshold == 0),
                 boundary_randomization_needed=bool(len(tied) > remaining))
    return [tuple(map(int, g)) for g in groups[chosen]], audit


def ranked_counts(selections):
    counts = Counter(g for person in selections for g in person)
    return sorted(counts, key=lambda g: (-counts[g], priority(g), g)), counts


def top_summary(selections):
    ranking, counts = ranked_counts(selections)
    top = ranking[:3]
    threshold = counts[top[-1]] if top else None
    return dict(top3=[dict(members_one_based=[v + 1 for v in g], count=counts[g]) for g in top],
                positive_frequency_groups=len(ranking), third_boundary_count=threshold,
                third_boundary_tie_count=sum(v == threshold for v in counts.values()) if top else 0,
                third_boundary_spans_selection=bool(len(ranking) > 3 and counts[ranking[3]] == threshold))


def evaluate(selections):
    """The held-out person's identities do not enter their discovery ranking."""
    n = len(selections)
    full = top_summary(selections)
    values, recommendations = [], []
    no_recommendation = 0
    tied_folds = 0
    for i, held in enumerate(selections):
        ranking, counts = ranked_counts(selections[:i] + selections[i + 1:])
        top = ranking[:3]
        hits = sum(g in held for g in top)
        tied_boundary = bool(len(ranking) > 3 and counts[ranking[2]] == counts[ranking[3]])
        tied_folds += tied_boundary
        values.append(hits / len(top) if top else 0.)
        no_recommendation += not bool(top)
        recommendations.append(dict(subject_index=i, groups_one_based=[[v + 1 for v in g] for g in top],
                                    hits=hits, recommendation_count=len(top),
                                    third_boundary_spans_selection=tied_boundary))
    full.update(loo_reselection_rate=float(np.mean(values)), loo_per_subject=values,
                loo_no_recommendation_subjects=no_recommendation,
                loo_third_boundary_tied_folds=tied_folds,
                loo_recommendations=recommendations,
                loo_subject_sd=float(np.std(values, ddof=1)) if n > 1 else None)
    return full


def summarize(selections):
    result = {m: evaluate(selections[m]) for m in METHODS}
    for m in METHODS[1:]:
        delta = np.array(result['spt']['loo_per_subject']) - result[m]['loo_per_subject']
        result[m]['paired_spt_minus_method'] = dict(mean=float(delta.mean()),
            subject_sd=float(delta.std(ddof=1)) if len(delta) > 1 else None,
            positive_count=int((delta > 0).sum()), negative_count=int((delta < 0).sum()),
            tied_count=int((delta == 0).sum()))
    overlap = []
    for i, m in enumerate(METHODS):
        for q in METHODS[i + 1:]:
            a = {tuple(x['members_one_based']) for x in result[m]['top3']}
            b = {tuple(x['members_one_based']) for x in result[q]['top3']}
            overlap.append(dict(method_a=m, method_b=q, exact_intersection=len(a & b),
                                jaccard=len(a & b) / len(a | b) if a | b else None))
    return dict(methods=result, top3_overlap=overlap)


def recover_spt(c, subjects, thresholds):
    source = {seed: hashlib.sha256(np.concatenate(
        paired_sources(seed, c['sample_count'], 100), axis=1).tobytes()).hexdigest()
              for seed in c['seeds']}
    audit, sources = empty_audit(), {}
    selected = {str(k): [] for k in (3, 4)}
    start = time.perf_counter()
    for si, subject in enumerate(subjects):
        window = c['windows'][subject]['grid_indices']
        if len(window) != 6 or c['windows'][subject]['required_hits'] != 4:
            raise ValueError('Frozen six-point / four-hit window changed')
        local = dict(c, G=[c['G'][i] for i in window])
        events = read_subject(BASE, local, subject, thresholds, sha(BASE / 'contract.json'), audit, source)
        for k in (3, 4):
            groups = []
            for mask, values in events.items():
                if mask.bit_count() != k:
                    continue
                repeat = [int(x).bit_count() >= c['repeat_seed_count'] for x in values[1]]
                if sum(repeat) >= 4:
                    groups.append(tuple(mask_members(mask)))
            selected[str(k)].append(sorted(groups))
        for g in local['G']:
            for seed in c['seeds']:
                stem = f'{subject}_G{g:.2f}_seed{seed}'
                for path in (BASE / 'trees' / f'{stem}.json', BASE / 'density' / f'{stem}.npz'):
                    sources[str(path.relative_to(ROOT))] = sha(path)
        if si == 0 or (si + 1) % 20 == 0 or si == len(subjects) - 1:
            print(f'SPT recovery {si + 1}/{len(subjects)}, {time.perf_counter() - start:.1f}s', flush=True)
    return selected, audit, sources


def verify_registry(selected, words, orders, counts):
    checked = {}
    for k in (3, 4):
        actual = Counter(g for person in selected[str(k)] for g in person)
        expected = {tuple(mask_members(word_mask(words[i]))): int(counts[i, 1, M['critical_ge4']])
                    for i in np.flatnonzero(orders == k)}
        if set(actual) - set(expected) or any(actual[g] != value for g, value in expected.items()):
            raise ArithmeticError(f'Order {k}: recovered SPT counts do not match the frozen registry')
        checked[str(k)] = dict(registered_groups_checked=len(expected), positive_groups=len(actual),
                               selection_count=sum(actual.values()), exact_match=True)
    return checked


def run(limit=None):
    start = time.perf_counter()
    c, subjects, words, orders, counts, original, fingerprints, _ = load_data()
    config = json.loads((BASE / 'contract.json').read_text())
    background = json.loads((BASE / 'background_reference.json').read_text())
    if (config['syn_tolerance_nats'] != 1e-8 or config['input_sha256'] != sha(INPUT)
            or background['contract_sha256'] != sha(BASE / 'contract.json')
            or original['background_reference_sha256'] != sha(BASE / 'background_reference.json')):
        raise ValueError('Frozen inputs, Syn tolerance or background provenance changed')
    if subjects != config['subject_ids'] or config['repeat_seed_count'] != 2:
        raise ValueError('Frozen participant identity or repeat rule changed')
    original_sc = json.loads((ROOT / 'results/dmf_schaefer100/sc_topology_comparison/contract.json').read_text())
    for path, expected in original_sc['provenance'].items():
        if not path.endswith('_sha256') and sha(ROOT / path) != expected:
            raise ValueError(f'Prior SC-comparison source changed: {path}')
    all_subjects = subjects.copy()
    if limit:
        c, subjects = c[:limit], subjects[:limit]
    spt, audit, source_hashes = recover_spt(config, subjects, background['thresholds'])
    registry_check = verify_registry(spt, words, orders, counts) if not limit else {'status': 'full cohort required'}
    results, sparse = {}, {}
    for k in (3, 4):
        order_start = time.perf_counter()
        groups = combinations(100, k)
        selections = {'spt': [set(person) for person in spt[str(k)]]}
        selections.update({m: [] for m in METRIC_NAMES})
        per_subject, valid = [], []
        for si, (subject, matrix) in enumerate(zip(subjects, c)):
            budget = len(selections['spt'][si])
            scores = score_arrays(matrix, groups)
            subject_audit = {'subject': subject, 'budget': budget, 'metrics': {}}
            common_valid = True
            for m, values in zip(METRIC_NAMES, scores):
                chosen, check = choose(values, groups, budget, subject)
                selections[m].append(set(chosen))
                subject_audit['metrics'][m] = check
                common_valid &= not (budget > 0 and (check['all_constant'] or check['zero_boundary_selected']))
            valid.append(common_valid)
            per_subject.append(subject_audit)
            if si == 0 or (si + 1) % 20 == 0 or si == len(subjects) - 1:
                print(f'SC order {k}: {si + 1}/{len(subjects)}, {time.perf_counter() - order_start:.1f}s', flush=True)
        del groups, scores, values
        for si in range(len(subjects)):
            if any(len(selections[m][si]) != len(selections['spt'][si]) for m in METHODS):
                raise ArithmeticError('Unequal individual selection counts')
        result = summarize(selections)
        budgets = [len(x) for x in selections['spt']]
        result.update(subject_count=len(subjects), active_subject_count=sum(x > 0 for x in budgets),
                      budgets=budgets, budget_distribution={str(v): budgets.count(v) for v in sorted(set(budgets))},
                      per_subject_selection_audit=per_subject,
                      common_valid_subject_ids=[s for s, v in zip(subjects, valid) if v])
        if not all(valid):
            if not any(valid):
                result['common_valid_sensitivity'] = {'status': 'no common valid participants'}
            else:
                result['common_valid_sensitivity'] = summarize(
                    {m: [x for x, v in zip(selections[m], valid) if v] for m in METHODS})
        else:
            result['common_valid_sensitivity'] = {'status': 'all participants valid; identical to primary'}
        results[str(k)] = result
        sparse[str(k)] = {m: [sorted(person) for person in selections[m]] for m in METHODS}
    fingerprints.update(source_hashes)
    for path in [BASE / 'background_reference.json', Path(__file__),
                 ROOT / 'scripts/analyze_dmf_sc_topology_comparison.py',
                 ROOT / 'scripts/analyze_dmf_critical_coalitions_all93.py']:
        fingerprints[str(path.relative_to(ROOT))] = sha(path)
    output = dict(status='smoke' if limit else 'complete', created_utc=datetime.now(timezone.utc).isoformat(),
                  subject_ids=subjects, frozen_cohort_subject_ids=all_subjects, orders=results,
                  registry_verification=registry_check, syn_audit=audit, metric_verification=verify_metrics(),
                  protocol=dict(seed=SEED, individual_budget='SPT above-background critical_ge4 count',
                      universe={'3': 161700, '4': 3921225},
                      individual_ties='Exact score equality; ascending BLAKE2b-128(seed|subject|members), then members',
                      cohort_ties='Decreasing own frequency; ascending BLAKE2b-128(seed|cohort|members), then members',
                      loo='Rank on other people only; average hits over positive-frequency top <=3, then over people; no recommendations=0',
                      inference='Exploratory within frozen upstream model; no confidence intervals or p values',
                      syn_tolerance_nats=1e-8, syn_clipping=False, new_simulations=0),
                  manuscript=dict(parent='P6UJCVG8', attachments=['DXGC7JEA', 'MWIWKSVG'],
                      rechecked_date='2026-10-23', explicit_version_or_date=None,
                      sections='Supplement S1.2 (S7-S8), S5.1-S5.2 (S49-S51), S12.2.1 (S82-S87)',
                      ambiguity='One main text and one supplement, without explicit version dates; metadata does not date the draft',
                      remaining_discrepancies=config['manuscript']['discrepancies'],
                      repository_design='docs/log/dmf_sc_spt_participant_frequency_design_20261023.md'),
                  elapsed_seconds=time.perf_counter() - start)
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = 'smoke_' if limit else ''
    save_json(OUT / f'{prefix}selections.json', dict(subject_ids=subjects, selections=sparse))
    save_json(OUT / f'{prefix}provenance.json', fingerprints)
    output['selection_sha256'] = sha(OUT / f'{prefix}selections.json')
    output['provenance_sha256'] = sha(OUT / f'{prefix}provenance.json')
    save_json(OUT / f'{prefix}summary.json', output)
    return output


def render(summary, *, top_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = ['#167E78', '#537EA5', '#C78C42', '#8A759C']
    text_size = 11 if top_only else 9
    style = {'font.family': 'DejaVu Sans', 'font.size': 11 if top_only else 10, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.spines.left': False,
             'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(10.4, 6.4) if top_only else (12, 7.5), layout='constrained')
        grid = fig.add_gridspec(1, 2) if top_only else fig.add_gridspec(2, 2, height_ratios=[2.6, 1])
        for col, k in enumerate((3, 4)):
            ax = fig.add_subplot(grid[0, col])
            result = summary['orders'][str(k)]
            for mi, m in enumerate(METHODS):
                top = result['methods'][m]['top3']
                y = np.arange(len(top)) + mi * 4
                counts = [x['count'] for x in top]
                ax.barh(y, counts, height=.67, color=colors[mi])
                for ti, (yy, value) in enumerate(zip(y, counts)):
                    tied_boundary = (ti == len(top) - 1 and
                                     result['methods'][m]['third_boundary_spans_selection'])
                    count_label = str(value) + (r'$^{\dagger}$' if tied_boundary else '')
                    ax.text(value + 1.2, yy, count_label, va='center', fontsize=text_size)
                metric_label = LABELS[mi] + ['', r' $W(A)$', r' $I(A)$', r' $Q(A)$'][mi]
                ax.text(0, mi * 4 - .65, metric_label, transform=ax.get_yaxis_transform(),
                        ha='left', va='bottom', fontsize=text_size, color=colors[mi])
            ticks = [mi * 4 + j for mi, m in enumerate(METHODS)
                     for j in range(len(result['methods'][m]['top3']))]
            labels = ['{' + ', '.join(map(str, x['members_one_based'])) + '}'
                      for m in METHODS for x in result['methods'][m]['top3']]
            ax.set_yticks(ticks, labels, fontsize=text_size)
            ax.tick_params(axis='y', length=0)
            ax.set_ylim(14.8, -1.45)
            ax.set_xlim(0, 65)
            ax.set_xticks([0, 20, 40, 60])
            ax.grid(axis='x', color='#E9ECEF', linewidth=.6)
            ax.set_axisbelow(True)
            ax.set_xlabel('Participants selecting the group (of 93)')
            ax.set_title(f'{k}-ROI groups', loc='left', fontsize=12 if top_only else 11, pad=10)
            ax.text(-.23, 1.04, 'ab'[col], transform=ax.transAxes, fontweight='bold', fontsize=12)
        if not top_only:
            ax = fig.add_subplot(grid[1, :])
            x = np.arange(len(METHODS))
            for k, dx, color, label in [(3, -.18, '#167E78', '3-ROI groups'),
                                        (4, .18, '#9AA8B3', '4-ROI groups')]:
                values = [summary['orders'][str(k)]['methods'][m]['loo_reselection_rate'] * 100 for m in METHODS]
                ax.bar(x + dx, values, width=.32, color=color)
                for xx, value in zip(x + dx, values):
                    ax.text(xx, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.set_xticks(x, LABELS)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Held-out reselection (%)')
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis='y', color='#E9ECEF', linewidth=.6)
            ax.set_axisbelow(True)
            ax.text(-.09, 1.04, 'c', transform=ax.transAxes, fontweight='bold', fontsize=12)
            ax.legend(handles=[Patch(facecolor='#167E78', label='3-ROI groups'),
                               Patch(facecolor='#9AA8B3', label='4-ROI groups')],
                      loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
        FIG.parent.mkdir(parents=True, exist_ok=True)
        output = FIG.with_name(FIG.stem + '_ab.png') if top_only else FIG
        fig.savefig(output, dpi=300 if top_only else 240, bbox_inches='tight')
        plt.close(fig)


def report(summary):
    lines = ['# SPT 与结构指标：匹配入选数量的跨个体对比', '',
             '执行日期：2026-10-23。93 人；三、四脑区分别分析；复用原模拟，不重估 EI/Syn。', '',
             '![对比结果](../../fig/dmf_schaefer100/sc_topology_comparison/sc_spt_participant_frequency.png)', '',
             '## 主要结果', '',
             '匹配每人入选数量后，SPT 与三个结构方法的三脑区、四脑区群体前三名均无完整成员重叠。SPT 的留出复现率没有普遍高于结构方法：三脑区略低于三个结构方法；四脑区略高于平均内部权重、低于加权团强度和模块度贡献。该结果支持共享结构组织与共享协同因果组织识别出不同组合，不能据此声称 SPT 具有更高跨个体一致性或更高因果准确率。', '',
             '| 方法 | 三脑区留出复现率 | 四脑区留出复现率 |', '|---|---:|---:|']
    for m, label in zip(METHODS, LABELS):
        v = [summary['orders'][str(k)]['methods'][m]['loo_reselection_rate'] * 100 for k in (3, 4)]
        lines.append(f'| {label} | {v[0]:.2f}% | {v[1]:.2f}% |')
    lines += ['', '留出率：每次在其余 92 人内按该方法自身入选人数选出前三名，再在留出者中检查入选情况；先对每人推荐组合求平均，再对 93 人求平均。空集合个体仍在分母内。', '',
              '## 入选名额及前三名', '']
    for k in (3, 4):
        r = summary['orders'][str(k)]
        lines += [f'### {k} 脑区', '',
                  f"非空集合：{r['active_subject_count']}/93 人；每人入选数量分布（数量:人数）：{r['budget_distribution']}。", '',
                  '| 方法 | 第一名：ROI / 人数 | 第二名：ROI / 人数 | 第三名：ROI / 人数 | 第三名计数并列组数 |',
                  '|---|---|---|---|---:|']
        for m, label in zip(METHODS, LABELS):
            q = r['methods'][m]
            cells = ['{' + ', '.join(map(str, x['members_one_based'])) + f"}} / {x['count']}" for x in q['top3']]
            cells += ['无'] * (3 - len(cells))
            lines.append('| ' + ' | '.join([label, *cells, str(q['third_boundary_tie_count'])]) + ' |')
        tied_methods = [LABELS[METHODS.index(m)] for m in METHODS
                        if r['methods'][m]['third_boundary_spans_selection']]
        if tied_methods:
            lines += ['', '第三名边界存在跨界并列的方法：' + '、'.join(tied_methods) + '；图中按预设共同优先序展示其中一个组合。']
        lines += ['', '留出排名第三名跨界并列的折数（SPT、平均内部权重、加权团强度、模块度贡献）：'
                  + '、'.join(str(r['methods'][m]['loo_third_boundary_tied_folds']) for m in METHODS)
                  + '。这些折的推荐与复现率条件于预先固定的并列规则，不应将微小方法差异解读为稳健优势。']
        lines += ['', '| 比较 | 前三名完整成员交集 | Jaccard | SPT − 结构方法留出率（百分点） | 配对正/负/相同人数 |',
                  '|---|---:|---:|---:|---:|']
        for row in r['top3_overlap']:
            if row['method_a'] != 'spt':
                continue
            m = row['method_b']; delta = r['methods'][m]['paired_spt_minus_method']
            lines.append(f"| SPT / {LABELS[METHODS.index(m)]} | {row['exact_intersection']} | {row['jaccard']:.3f} | {100 * delta['mean']:+.2f} | {delta['positive_count']}/{delta['negative_count']}/{delta['tied_count']} |")
        audits = [a['metrics'][m] for a in r['per_subject_selection_audit'] for m in METRIC_NAMES]
        lines += ['', f"需随机处理的个体截断边界并列：{sum(a['boundary_randomization_needed'] for a in audits)} 次；全常数分数：{sum(a['all_constant'] for a in audits)} 次；零分边界实际入选：{sum(a['zero_boundary_selected'] for a in audits)} 次。", '',
                  f"共同有效个体：{len(r['common_valid_subject_ids'])}/93。敏感性结果状态：{r['common_valid_sensitivity'].get('status', '已在结果 JSON 中报告')}。", '']
    a = summary['syn_audit']
    lines += ['## 核验与结论边界', '',
              '- 每人三个结构方法入选数量均等于该人的 SPT 数量；结构方法在全部 161,700/3,921,225 个组合中评分，只共享名额，不共享 SPT 成员身份。',
              '- 三、四阶 SPT 完整成员的 critical_ge4 人数均逐项恢复原注册表，包括原计数为零的组合。',
              f"- 读取 {a['tree_count']} 棵窗口树及相应密度缓存；Syn 容差 1e-8 nats。选中节点/候选/成对/查询的容差内负值计数分别为 {a['selected_tolerance_negative_count']}/{a['candidate_tolerance_negative_count']}/{a['pair_tolerance_negative_count']}/{a['query_tolerance_negative_count']}；所有有效性检查通过，无裁剪。", 
              '- 个体边界按精确原始分数判等；仅并列时按固定种子 20261023、个体身份和完整成员的 BLAKE2b-128 优先序选取。群体频率并列使用同一种子与完整成员的共同优先序；该优先序不依赖方法或效果。',
              '- 各方法群体排名仅按自身入选人数，不沿用旧图中的端点泄漏等附加排序。第三名跨边界并列均显式记录。ROI 编号从 1 开始，与模型原始顺序一致；不依赖尚待核实的解剖标签映射。',
              '- 这是冻结上游模型下的探索性跨个体复现检查。组平均 FIC 等上游参数未在留出折内重拟合，不能称为完全独立队列验证。SPT 入选预算来自留出者自身，因此评价也是给定个人预算的条件复现率。',
              '- 本次不报告置信区间或显著性检验；个体配对差值和符号计数描述样本内变化，不支持总体显著性结论。前三名已按同一数据筛选，不能用全样本人数证明泛化。',
              '- 跨个体共享程度与协同因果准确率是不同目标；结构方法与 SPT 成员不同本身并不构成因果有效性检验。', '',
              '## 手稿及来源', '',
              '重新读取 Zotero 父条目 P6UJCVG8，核对标题 Emergent hierarchical organization of causal interactions in complex systems；附件 DXGC7JEA（正文）与 MWIWKSVG（补充）均可读取全文，但均无明确稿件版本日期，不能据元数据修改时间断定最新修订。核对补充 S1.2（S7–S8）、S5.1–S5.2（S49–S51）及 S12.2.1（S82–S87）。补充 S9.4 是 SHAP 方法，并非当前仓库新增拓扑比较；本次具体方案以冻结设计和合同为准。', '',
              '沿用既有实验并保留以下手稿/实现差异：', '']
    lines += ['- ' + x for x in summary['manuscript']['remaining_discrepancies']]
    lines += ['', '输入来自原 93 人 SC 和既有窗口树、背景阈值、密度缓存；来源逐文件 SHA-256 保存于 results/dmf_schaefer100/sc_spt_participant_frequency/provenance.json。稀疏个体集合、并列记录、逐人留出推荐和配对差值均保存在同目录的 selections.json 与 summary.json。', '',
              f"总计算耗时 {summary['elapsed_seconds']:.1f} 秒。图中 a、b 柱高均为各方法自身入选人数，c 为留出个体重新入选比例；不显示误差条不代表不存在不确定性。", '']
    REPORT.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-subjects', type=int)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--top-only', action='store_true', help='Render cached panels a and b only')
    args = parser.parse_args()
    if args.top_only and not args.render_only:
        parser.error('--top-only requires --render-only; it never reruns the analysis')
    if args.smoke_subjects is not None and not 1 <= args.smoke_subjects <= 93:
        parser.error('--smoke-subjects must be between 1 and 93')
    if args.render_only:
        summary = json.loads((OUT / 'summary.json').read_text())
        if summary['status'] != 'complete' or summary['selection_sha256'] != sha(OUT / 'selections.json'):
            raise ValueError('Incomplete or changed result cache')
    else:
        summary = run(args.smoke_subjects)
    if not args.smoke_subjects:
        render(summary, top_only=args.top_only)
        if not args.top_only:
            report(summary)
    print(json.dumps(dict(status=summary['status'], elapsed_seconds=summary['elapsed_seconds'],
                          output=str(OUT), figure=str(FIG.with_name(FIG.stem + '_ab.png') if args.top_only else FIG)), ensure_ascii=False))


if __name__ == '__main__':
    main()
