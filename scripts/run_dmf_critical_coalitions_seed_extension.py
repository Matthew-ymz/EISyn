#!/usr/bin/env python3
"""Add authorized seeds 6/7; retain the frozen kernel and fixed C1–C3 identities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_dmf_critical_coalitions as kernel
from scripts.analyze_dmf_critical_coalitions_all93 import members_mask
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez

BASE = kernel.BASE / 'seed_extension5'
FIGURE_DIR = ROOT / 'fig/dmf_schaefer100'
REPORT = ROOT / 'docs/reports/brain_critical_coalition_seed5.md'
STATE = {}


def prepare(base):
    original = kernel.prepare(kernel.BASE)
    summary = json.loads((kernel.BASE / 'all93_summary.json').read_text())
    if summary['status'] != 'complete' or summary['condition_count'] != 11439:
        raise ValueError('The original full three-seed cohort must be complete')
    contract = dict(version='fixed-candidates-five-seeds-v1', authorization_date='2026-10-08',
        authorization='User explicitly requested two additional seeds and original three-seed presentation figures',
        scientific_kernel_contract_sha256=digest(kernel.BASE / 'contract.json'),
        original_summary_sha256=digest(kernel.BASE / 'all93_summary.json'),
        launcher_sha256=digest(Path(__file__)), subject_ids=original['subject_ids'], G=original['G'],
        reused_seeds=[3, 4, 5], added_seeds=[6, 7], all_seeds=[3, 4, 5, 6, 7],
        reused_conditions=11439, added_conditions=7626, total_conditions=19065,
        fixed_candidates=[dict(display_id=r['display_id'], members=r['members'],
                               members_one_based=r['members_one_based']) for r in summary['selected'][:3]],
        original_result_directory=str(kernel.BASE),
        contract_roles='contract.json is a byte-identical frozen scientific-kernel source contract; '
                       'this extension contract overrides its run seed schedule only, without modifying the kernel',
        controls='Native SC, fixed shared JFIC, G/window, full 200-dimensional target, 2048 uniform inputs, '
                 '300 steps, affine-TM readout, ridge/sigma, standard SPT and native Syn tolerance unchanged',
        pairing=original['pairing'],
        primary_readout='Exact natural-node presence count / seed count, with no Syn-strength reference',
        secondary_readouts='Strict majority: 2/3 original, 2/2 added-only, 3/5 combined; '
                           'report >=4/5 combined separately as a stricter sensitivity, not an alternative selected for success',
        inference='Fixed candidates from prior all93 exploration; new seeds assess simulation repetition, '
                  'not new subjects or an untouched cohort; no new candidate ranking',
        stopping='Complete all declared conditions regardless of candidate recurrence; numerical violations fail explicitly',
        new_seed_dense_reference='Original independent dense scan has only seeds 3/4/5. '
                                 'No dense Xi reproduction claim for seeds 6/7; instead audit native covariance, source/noise and all tree budgets',
        native_syn_tolerance_nats=kernel.TOL,
        manuscript=dict(parent='P6UJCVG8', attachments=['DXGC7JEA', 'MWIWKSVG'],
            checked_date='2026-10-08', explicit_version_or_date=None,
            ambiguity='No explicit draft revision/date; attachment metadata edits do not date the manuscript',
            locations='Main Methods Eqs10–12 pp16–17; supplement S5/Algorithm S1 pp8–9'))
    base.mkdir(parents=True, exist_ok=True)
    for directory in ['trees', 'density']:
        (base / directory).mkdir(exist_ok=True)
    frozen_copy = base / 'contract.json'
    original_bytes = (kernel.BASE / 'contract.json').read_bytes()
    if frozen_copy.exists() and frozen_copy.read_bytes() != original_bytes:
        raise ValueError('Extension scientific-kernel contract changed')
    if not frozen_copy.exists():
        frozen_copy.write_bytes(original_bytes)
    path = base / 'seed_extension_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen five-seed extension changed; do not silently resume')
    if not path.exists():
        atomic_json(path, contract)
    return original, contract


def initialize(base):
    kernel.initialize(base)
    STATE.update(base=Path(base), extension=json.loads((Path(base) / 'seed_extension_contract.json').read_text()),
                 extension_sha=digest(Path(base) / 'seed_extension_contract.json'))


def validate_diagnostics(diagnostics, key):
    fields = ['state_min', 'state_max', 'rate_min_hz', 'rate_max_hz',
              'outside_state_count', 'state_count', 'abnormal_rate_count']
    if not np.isfinite([diagnostics[field] for field in fields]).all():
        raise ArithmeticError(f'Nonfinite simulation diagnostics: {key}')
    if diagnostics['outside_state_count'] or diagnostics['abnormal_rate_count']:
        raise ArithmeticError(f'State/rate boundary violation: {key}: {diagnostics}')


def validate_tree(record, key):
    tree = record['tree']
    nodes = tree['nodes']
    check = kernel.audit_nonnegative([node['syn_nats_raw'] for node in nodes])
    seen = set()
    children = []
    for node in nodes:
        mask = members_mask(node['members'])
        left, right = members_mask(node['left']), members_mask(node['right'])
        if (mask in seen or mask.bit_count() != node['order'] or node['order'] < 2
                or not left or not right or left & right or (left | right) != mask):
            raise ValueError(f'Invalid exact tree membership: {key}')
        seen.add(mask)
        children.extend([left, right])
    if (len(seen) != 99 or members_mask(range(100)) not in seen
            or any(child.bit_count() > 1 and child not in seen for child in children)):
        raise ValueError(f'Incomplete natural tree: {key}')
    totals = record['totals']
    if not np.isfinite(list(totals.values()) + [tree['root_cross_roi_nats'], tree['closure_error_nats']]).all():
        raise ArithmeticError(f'Nonfinite global/tree budgets: {key}')
    closure = max(abs(sum(node['syn_nats_raw'] for node in nodes) - totals['cross_roi_nats']),
        abs(tree['root_cross_roi_nats'] - totals['cross_roi_nats']),
        abs(tree['closure_error_nats']), abs(totals['closure_error_nats']))
    if closure > kernel.TOL:
        raise ArithmeticError(f'Tree/global budget failure: {key}: {closure} nats')
    candidate = tree['candidate_audit']
    if not np.isfinite(candidate['minimum_candidate_syn']) or candidate['minimum_candidate_syn'] < -kernel.TOL:
        raise ArithmeticError(f"Candidate Syn violation: min={candidate['minimum_candidate_syn']} nats; "
                              f"threshold={-kernel.TOL} nats; affected>=1: {key}")
    for kind in ['selected_syn_audit', 'pair_audit', 'query_audit']:
        a = tree[kind]
        if (not np.isfinite(a['minimum_nats']) or a['minimum_nats'] < -kernel.TOL
                or a['violation_count'] or a['tolerance_nats'] != kernel.TOL or a['count'] < 1):
            raise ArithmeticError(f'Invalid nonnegativity audit {kind}: {key}: {a}')
    if tree['selected_syn_audit'] != check:
        raise ArithmeticError(f'Selected-node audit mismatch: {key}')
    return check, closure


def job(task):
    subject, gi, seed = task
    w = kernel.W
    c = w['c']
    g = c['G'][gi]
    key = f'{subject}_G{g:.2f}_seed{seed}'
    if seed not in STATE['extension']['added_seeds']:
        raise ValueError('Only declared new seeds may be simulated here')
    path = STATE['base'] / 'trees' / (key + '.json')
    if path.exists():
        r = json.loads(path.read_text())
        if (r['contract_sha256'] != w['sha'] or r['extension_contract_sha256'] != STATE['extension_sha']
                or r['subject'] != subject or r['G'] != g or r['seed'] != seed):
            raise ValueError(f'Existing extension condition mismatch: {path}')
        return dict(condition=key, reused=True, elapsed_seconds=0.)
    started = time.perf_counter()
    if seed not in w['sources']:
        w['sources'][seed] = np.concatenate(kernel.paired_sources(seed, c['sample_count'], 100), axis=1)
    x = w['sources'][seed]
    source_sha = hashlib.sha256(x.tobytes()).hexdigest()
    density_path = STATE['base'] / 'density' / (key + '.npz')
    simulation_seconds = 0.
    if density_path.exists():
        with np.load(density_path) as a:
            if (str(a['contract_sha256']) != w['sha'] or str(a['extension_contract_sha256']) != STATE['extension_sha']
                    or str(a['source_sha256']) != source_sha or int(a['noise_seed']) != kernel.noise_seed(seed)):
                raise ValueError(f'Existing density source/kernel mismatch: {density_path}')
            covariance = a['covariance'].copy()
            diagnostics = json.loads(str(a['diagnostics_json']))
    else:
        sc = w['matrices'][c['subject_ids'].index(subject)]
        t = time.perf_counter()
        y, diagnostics = kernel.simulate_sources(w['dmf'], x, sc, w['jf'], g, w['p'], kernel.noise_seed(seed))
        simulation_seconds = time.perf_counter() - t
        if diagnostics['outside_state_count'] or not np.isfinite(y).all():
            raise ArithmeticError(f'Nonfinite simulation/state boundary violation: {key}: {diagnostics}')
        covariance = kernel.fit_affine_joint(x, y, .2, ridge=c['ridge'])
        atomic_savez(density_path, covariance=covariance, contract_sha256=w['sha'],
            extension_contract_sha256=STATE['extension_sha'], source_sha256=source_sha,
            noise_seed=kernel.noise_seed(seed), diagnostics_json=json.dumps(diagnostics))
    validate_diagnostics(diagnostics, key)
    game = kernel.CommonTargetGame(covariance)
    totals = game.totals()
    t = time.perf_counter()
    tree = kernel.standard_tree(game)
    tree_seconds = time.perf_counter() - t
    closure = abs(sum(n['syn_nats_raw'] for n in tree['nodes']) - totals['cross_roi_nats'])
    if closure > kernel.TOL:
        raise ArithmeticError(f'Independent tree budget failure: {key}: {closure} nats')
    record = dict(subject=subject, G=g, seed=seed, background=False, contract_sha256=w['sha'],
        extension_contract_sha256=STATE['extension_sha'], tree=tree, totals=totals,
        dense_xi_difference_nats=None, dense_reference_status='Not available for added seed',
        independent_closure_error_nats=closure, simulation_seconds=simulation_seconds,
        tree_seconds=tree_seconds, elapsed_seconds=time.perf_counter() - started)
    validate_tree(record, key)
    atomic_json(path, record)
    return dict(condition=key, reused=False, elapsed_seconds=record['elapsed_seconds'],
                simulation_seconds=simulation_seconds, tree_seconds=tree_seconds)


def smoke(base, workers):
    c, extension = prepare(base)
    tasks = [(c['subject_ids'][0], 0, 6), (c['subject_ids'][0], 28, 6),
             (c['subject_ids'][-1], 13, 7), (c['subject_ids'][-1], 40, 7)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(str(base),)) as pool:
        for r in pool.map(job, tasks):
            print(json.dumps(dict(event='smoke_condition', **r)), flush=True)
    atomic_json(base / 'seed5_smoke_complete.json', dict(status='passed', conditions=len(tasks),
        extension_contract_sha256=digest(base / 'seed_extension_contract.json')))


def analyze(base, c, extension):
    sha = digest(base / 'contract.json')
    extension_sha = digest(base / 'seed_extension_contract.json')
    subjects, grid, seeds = c['subject_ids'], c['G'], extension['all_seeds']
    candidates = extension['fixed_candidates']
    masks = {members_mask(row['members']): i for i, row in enumerate(candidates)}
    values = np.full((len(candidates), len(subjects), len(grid), len(seeds)), np.nan)
    source_hashes = {seed: hashlib.sha256(np.concatenate(kernel.paired_sources(seed, c['sample_count'], 100),
                                                        axis=1).tobytes()).hexdigest() for seed in seeds}
    audit = dict(tree_count=0, density_count=0, selected_node_count=0, minimum_selected_syn_nats=None,
        selected_tolerance_negative_count=0, candidate_count=0, minimum_candidate_syn_nats=None,
        candidate_tolerance_negative_count=0, pair_tolerance_negative_count=0,
        query_tolerance_negative_count=0, outside_state_count=0, abnormal_rate_count=0,
        max_closure_error_nats=0.)
    for pi, subject in enumerate(subjects):
        for gi, g in enumerate(grid):
            for si, seed in enumerate(seeds):
                location = kernel.BASE if seed in extension['reused_seeds'] else base
                key = f'{subject}_G{g:.2f}_seed{seed}'
                r = json.loads((location / 'trees' / (key + '.json')).read_text())
                if (r['contract_sha256'] != sha or r['subject'] != subject or r['G'] != g
                        or r['seed'] != seed or r['background']):
                    raise ValueError(f'Tree identity/kernel mismatch: {key}')
                if seed in extension['added_seeds'] and r['extension_contract_sha256'] != extension_sha:
                    raise ValueError(f'Tree extension mismatch: {key}')
                with np.load(location / 'density' / (key + '.npz')) as a:
                    if (str(a['contract_sha256']) != sha or str(a['source_sha256']) != source_hashes[seed]
                            or int(a['noise_seed']) != kernel.noise_seed(seed)):
                        raise ValueError(f'Density source/noise mismatch: {key}')
                    if seed in extension['added_seeds'] and str(a['extension_contract_sha256']) != extension_sha:
                        raise ValueError(f'Density extension mismatch: {key}')
                    diagnostics = json.loads(str(a['diagnostics_json']))
                validate_diagnostics(diagnostics, key)
                nodes = r['tree']['nodes']
                check, closure = validate_tree(r, key)
                for node in nodes:
                    mask = members_mask(node['members'])
                    if mask in masks:
                        values[masks[mask], pi, gi, si] = node['syn_nats_raw']
                audit['tree_count'] += 1
                audit['density_count'] += 1
                audit['selected_node_count'] += 99
                audit['selected_tolerance_negative_count'] += check['tolerance_negative_count']
                minimum = audit['minimum_selected_syn_nats']
                audit['minimum_selected_syn_nats'] = check['minimum_nats'] if minimum is None else min(minimum, check['minimum_nats'])
                ca = r['tree']['candidate_audit']
                audit['candidate_count'] += ca['candidate_count']
                minimum = audit['minimum_candidate_syn_nats']
                audit['minimum_candidate_syn_nats'] = ca['minimum_candidate_syn'] if minimum is None else min(minimum, ca['minimum_candidate_syn'])
                audit['candidate_tolerance_negative_count'] += ca['tolerance_zero_count']
                audit['pair_tolerance_negative_count'] += r['tree']['pair_audit']['tolerance_negative_count']
                audit['query_tolerance_negative_count'] += r['tree']['query_audit']['tolerance_negative_count']
                audit['max_closure_error_nats'] = max(audit['max_closure_error_nats'], closure)
    if audit['tree_count'] != extension['total_conditions']:
        raise ValueError('Partial extension must not be called complete')
    with np.load(kernel.BASE / 'all93_display.npz') as a:
        if not np.array_equal(values[:, :, :, :3], a['raw_syn_nats'][:3], equal_nan=True):
            raise ValueError('Original three-seed display values changed')
    present = np.isfinite(values)
    rows = []
    for ci, candidate in enumerate(candidates):
        groups = {}
        for label, indices in [('original3', [0, 1, 2]), ('added2', [3, 4]), ('all5', [0, 1, 2, 3, 4])]:
            p = present[ci][:, :, indices]
            frequency = p.mean(2)
            repeated = p.sum(2) >= (len(indices) // 2 + 1)
            window_hits = np.array([repeated[pi, c['windows'][who]['grid_indices']].sum()
                                    for pi, who in enumerate(subjects)])
            window_p = np.array([frequency[pi, c['windows'][who]['grid_indices']].mean()
                                 for pi, who in enumerate(subjects)])
            outside_p = np.array([frequency[pi, np.setdiff1d(np.arange(len(grid)), c['windows'][who]['grid_indices'])].mean()
                                  for pi, who in enumerate(subjects)])
            outside_hits = np.array([repeated[pi, np.setdiff1d(np.arange(len(grid)), c['windows'][who]['grid_indices'])].sum()
                                     for pi, who in enumerate(subjects)])
            groups[label] = dict(seeds=[seeds[i] for i in indices], majority_required=len(indices)//2+1,
                window_any_subjects=int((window_hits >= 1).sum()), window_ge4_subjects=int((window_hits >= 4).sum()),
                outside_any_subjects=int((outside_hits > 0).sum()),
                mean_window_seed_presence_fraction=float(window_p.mean()),
                mean_outside_seed_presence_fraction=float(outside_p.mean()))
        strict = present[ci].sum(2) >= 4
        strict_hits = np.array([strict[pi, c['windows'][who]['grid_indices']].sum()
                                for pi, who in enumerate(subjects)])
        rows.append(dict(**candidate, groups=groups,
                         all5_at_least4seeds_window_ge4_subjects=int((strict_hits >= 4).sum())))
    summary = dict(status='complete', subject_count=93, condition_count=extension['total_conditions'],
        seeds=seeds, candidates=rows, audit=audit, native_syn_tolerance_nats=kernel.TOL,
        scientific_kernel_contract_sha256=sha, extension_contract_sha256=extension_sha,
        readout='Natural exact-node presence; original3, added2 and all5 separate; no strength reference',
        inference=extension['inference'])
    atomic_savez(base / 'seed5_display.npz', raw_syn_nats=values, presence=present,
        subject_ids=np.array(subjects), G=np.array(grid), seeds=np.array(seeds),
        members_json=json.dumps([row['members'] for row in candidates]),
        scientific_kernel_contract_sha256=sha, extension_contract_sha256=extension_sha)
    atomic_json(base / 'seed5_summary.json', summary)
    return summary, present


def export(base, c, extension, summary, present):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    subjects, grid = c['subject_ids'], np.array(c['G'])
    mids = np.array([c['windows'][p]['transition']['midpoint'] for p in subjects])
    permutation = sorted(range(93), key=lambda pi: mids[pi])
    xlim = (float((grid[0] - mids).min() - .05), float((grid[-1] - mids).max() + .05))
    colours = ['#236A91', '#BA5636', '#717A38']
    files = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'}):
        cmap = LinearSegmentedColormap.from_list('presence', ['#EBECEE', '#236A91'])
        for ci, row in enumerate(extension['fixed_candidates']):
            fig, ax = plt.subplots(figsize=(10.8, 6.8), layout='constrained')
            frequency = present[ci].mean(2) * 100
            for ri, pi in enumerate(permutation):
                edges = np.r_[grid - .05, grid[-1] + .05] - mids[pi]
                mesh = ax.pcolormesh(edges, [ri - .5, ri + .5], frequency[pi][None, :],
                                     cmap=cmap, norm=Normalize(0, 100), shading='flat')
            for edge in [-.3, .3]:
                ax.axvline(edge, color=colours[1], ls='--', lw=1)
            positions = list(range(0, 93, 6))
            ax.set_yticks(positions, [subjects[permutation[i]].replace('sub-', '') for i in positions])
            ax.set(xlim=xlim, ylim=(92.5, -.5), ylabel='Individual SC (93 rows; sorted by transition)',
                   xlabel=r'$\Delta G = G - G_i^{*}$',
                   title=row['display_id'] + ': ROI ' + ', '.join(map(str, row['members_one_based'])))
            colorbar = fig.colorbar(mesh, ax=ax, pad=.03, shrink=.75, ticks=[0, 20, 40, 60, 80, 100])
            colorbar.set_label('Seeds with exact natural node (%)\n5 seeds per subject/G')
            ax.legend(handles=[Line2D([], [], color=colours[1], ls='--', label='Six-point window (cell edges)')],
                      loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False)
            output = FIGURE_DIR / f'critical_coalition_seed5_{row["display_id"].lower()}.png'
            fig.savefig(output, dpi=240, bbox_inches='tight')
            plt.close(fig)
            files.append(output)
        fig, (ax, coverage_ax) = plt.subplots(2, 1, figsize=(10.8, 5.6), sharex=True,
            layout='constrained', gridspec_kw={'height_ratios': [3.5, 1], 'hspace': .05})
        for ci, row in enumerate(extension['fixed_candidates']):
            bucket = {}
            for pi in range(93):
                for gi, g in enumerate(grid):
                    key = int(round((g - mids[pi]) * 20))
                    bucket.setdefault(key, []).append(present[ci, pi, gi].astype(float))
            keys = sorted(bucket)
            x = np.array(keys)/20
            coverage = np.array([len(bucket[key]) for key in keys])
            per_seed = np.array([np.mean(bucket[key], axis=0) for key in keys]) * 100
            ax.plot(x, per_seed.mean(1), color=colours[ci], lw=1.5, marker=['o', 's', '^'][ci],
                    ms=3.5, ls=['-', '--', '-.'][ci], label=row['display_id'])
            if ci == 0:
                ax.fill_between(x, per_seed.min(1), per_seed.max(1), color=colours[0], alpha=.13, lw=0)
        for axes in (ax, coverage_ax):
            axes.axvspan(-.3, .3, color=colours[1], alpha=.08, lw=0)
            axes.set_xlim(*xlim)
        handles, labels = ax.get_legend_handles_labels()
        from matplotlib.patches import Patch
        handles += [Patch(facecolor=colours[0], alpha=.13)]
        labels += ['C1: min–max across 5 seeds\n(not a confidence interval)']
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
        ax.set(ylabel='Mean natural-node presence (%)', ylim=(-2, 102))
        coverage_ax.plot(x, coverage, color='#5A6066', lw=1.2)
        coverage_ax.set(ylabel='Covered\nsubjects', ylim=(0, 100), yticks=[0, 50, 93],
                        xlabel=r'$\Delta G = G - G_i^{*}$')
        output = FIGURE_DIR / 'critical_coalition_seed5_recurrence.png'
        fig.savefig(output, dpi=240, bbox_inches='tight')
        plt.close(fig)
        files.append(output)
    table = []
    for row in summary['candidates']:
        old, new, all5 = [row['groups'][key] for key in ['original3', 'added2', 'all5']]
        table.append(f"| {row['display_id']}：{'、'.join(map(str, row['members_one_based']))} | "
            f"{old['mean_window_seed_presence_fraction']*100:.1f}% | {new['mean_window_seed_presence_fraction']*100:.1f}% | "
            f"{all5['mean_window_seed_presence_fraction']*100:.1f}% | {all5['mean_outside_seed_presence_fraction']*100:.1f}% | "
            f"{all5['window_any_subjects']}/93 | {all5['window_ge4_subjects']}/93 |")
    text = '# 固定 C1–C3 的五 seed 自然节点复现\n\n'
    text += '原 seed 3、4、5 的 11,439 条件复用，新增 seed 6、7 的 7,626 条件完成；总计 93×41×5=19,065 条件。图形导出待目视复核。\n\n'
    text += '候选完整成员固定自原全 93 人探索。每格的主读出为入树 seed 数／seed 总数，不按 Syn 强度或 reference 筛选。下表的平均窗口入树频率先对每人的六格和 seed 平均，再跨人平均；窗口外按其余 35 格平均。\n\n'
    text += '| 完整候选 | 原3seed窗口频率 | 新2seed窗口频率 | 全5seed窗口频率 | 全5seed窗口外频率 | 多数seed窗口≥1/6 | 多数seed窗口≥4/6 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    text += '\n'.join(table) + '\n\n'
    text += '全五 seed 的多数规则为至少 3/5；原三 seed 为 2/3，新增两 seed 为两者都出现。主图保留连续频率，另存至少 4/5 的较严格读出，均不根据候选表现选择阈值。\n\n'
    for output in files:
        text += f'![{output.stem}](../../fig/dmf_schaefer100/{output.name})\n\n'
    text += '逐人热图使用共同 0–100% 入树频率色标，白色为未扫描；93 行同序、无平滑。复现曲线为实际覆盖个体的单 seed 入树比例，再对五 seed 等权平均；C1 阴影为五个 seed 的最小–最大范围，不是置信区间。扫描边缘按实际覆盖人数计算。\n\n'
    text += '新增 seed 检验固定候选的模拟重复性，不构成新增被试、未接触队列或消除估计偏差的证明。节点 Syn 仍是路径依赖的层级残差；发放率窗口仍为操作性定位。所有模型、干预、共同 target、时距、估计近似及 SPT 搜索保持原科学内核。\n\n'
    text += f"全部 {summary['audit']['tree_count']:,} 棵树与密度的身份、source/noise、完整成员、非负性及预算闭合已检查。Syn 容差为 10⁻⁸ nats，最小选中 Syn 为 {summary['audit']['minimum_selected_syn_nats']:.10g} nats，容差内负值为 {summary['audit']['selected_tolerance_negative_count']}；原 Syn 未裁剪。新增 seed 无既有独立 dense 扫描，不声称其 dense Ξ 逐点再现检查；原三 seed 的展示缓存逐值完全一致。\n"
    REPORT.write_text(text)
    atomic_json(base / 'seed5_figure_metadata.json', dict(figures=[dict(output=str(p), sha256=digest(p)) for p in files],
        summary_sha256=digest(base/'seed5_summary.json'), plotting_sha256=digest(Path(__file__)),
        report_sha256=digest(REPORT), visual_review='pending', legend_placement='Outside all data',
        readout='Five-seed natural-node presence; no strength reference',
        native_syn_tolerance_nats=kernel.TOL, tolerance_negative_count=summary['audit']['selected_tolerance_negative_count']))


def run(base, workers):
    c, extension = prepare(base)
    with (base / 'seed5_run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = []
        reused_new = 0
        for subject in c['subject_ids']:
            for gi, g in enumerate(c['G']):
                for seed in extension['added_seeds']:
                    path = base / 'trees' / f'{subject}_G{g:.2f}_seed{seed}.json'
                    if path.exists():
                        r = json.loads(path.read_text())
                        if r['extension_contract_sha256'] != digest(base / 'seed_extension_contract.json'):
                            raise ValueError(f'Existing extension mismatch: {path}')
                        reused_new += 1
                    else:
                        tasks.append((subject, gi, seed))
        started = time.perf_counter()
        launch = dict(pid=os.getpid(), workers=workers, scheduled_new_conditions=len(tasks),
            reused_added_conditions=reused_new, original_reused_conditions=11439, total_conditions=19065,
            started_utc=datetime.now(timezone.utc).isoformat(),
            extension_contract_sha256=digest(base/'seed_extension_contract.json'))
        atomic_json(base / 'seed5_launch.json', launch)
        print(json.dumps(dict(event='launch', **launch)), flush=True)
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize, initargs=(str(base),)) as pool:
                futures = [pool.submit(job, task) for task in tasks]
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        completed += 1
                        print(json.dumps(dict(completed=completed, scheduled=len(tasks), **result)), flush=True)
                except BaseException as exc:
                    atomic_json(base / 'seed5_failure.json', dict(status='failed', exception=type(exc).__name__,
                        message=str(exc), completed_new_conditions=completed, launch=launch))
                    for future in futures:
                        future.cancel()
                    raise
            atomic_json(base / 'seed5_simulation_complete.json', dict(status='complete',
                added_conditions=completed+reused_new, elapsed_seconds=time.perf_counter()-started,
                extension_contract_sha256=launch['extension_contract_sha256']))
            summary, present = analyze(base, c, extension)
            export(base, c, extension, summary, present)
            atomic_json(base / 'seed5_pipeline_complete.json', dict(status='computation_analysis_exports_complete',
                elapsed_seconds=time.perf_counter()-started, visual_review='pending agent inspection',
                extension_contract_sha256=launch['extension_contract_sha256']))
        except BaseException as exc:
            atomic_json(base / 'seed5_failure.json', dict(status='failed', exception=type(exc).__name__,
                message=str(exc), completed_new_conditions=completed, launch=launch))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=BASE)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--postprocess-only', action='store_true',
                   help='Recheck complete caches and regenerate exports without running simulations')
    a = p.parse_args()
    if a.workers < 1 or (a.smoke and a.postprocess_only):
        p.error('Positive worker count and at most one special mode are required')
    if a.postprocess_only:
        c = kernel.prepare(kernel.BASE)
        extension = json.loads((a.output_dir / 'seed_extension_contract.json').read_text())
        if (digest(a.output_dir / 'contract.json') != extension['scientific_kernel_contract_sha256']
                or digest(kernel.BASE / 'contract.json') != extension['scientific_kernel_contract_sha256']):
            raise ValueError('Scientific-kernel source changed')
        summary, present = analyze(a.output_dir, c, extension)
        export(a.output_dir, c, extension, summary, present)
    elif a.smoke:
        smoke(a.output_dir, a.workers)
    else:
        run(a.output_dir, a.workers)


if __name__ == '__main__':
    main()
