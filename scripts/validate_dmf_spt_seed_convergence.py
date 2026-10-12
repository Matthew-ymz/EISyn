#!/usr/bin/env python3
"""Fixed G=1.3: eight-seed SPT convergence with nested sample/noise prefixes.

Only sample count changes. Keep the current Gaussian/affine estimator (the
documented 200-dimensional TM cost exception), exact <=8 ROI enumeration,
and deterministic spectral candidates. Batch 0 exactly replays the existing
2048-sample paired protocol; subsequent independent batches use seed+1000*b.
Compare full binary trees, never their visually compressed dendrograms.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_critical_phi_hierarchy_topology import conditional_source_covariance
from scripts.analyze_dmf_schaefer100_xi_hierarchy_tree import _tree_metrics, build_scalable_hierarchy, pairwise_syn_affinity
from scripts.analyze_runge_slp_pc60_xi_hierarchy import _node_record
from scripts.compare_runge_slp_pc60_xi_horizons import _node_from_record
from scripts.plot_dmf_schaefer100_tree_examples import CachedScalarLogdetXiOracle, audit_tree, lca_sizes, nontrivial_clades
from scripts.run_dmf_diffusive_fullstate_control import rollout
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.spt import flatten_nodes
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, standardize

SOURCE = ROOT / 'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
BASE_DIR = ROOT / 'results/dmf_schaefer100/unconstrained_spt_wide/shards'
EXAMPLES = ROOT / 'results/dmf_schaefer100/xi_hierarchy_tree/examples/wide/summary.json'
DEFAULT_OUT = ROOT / 'results/dmf_schaefer100/xi_hierarchy_tree/sampling_convergence_G130'
DEFAULT_FIGURE = ROOT / 'docs/reports/assets/dmf_spt_seed_sampling_convergence.png'
DEFAULT_TREE_FIGURE = ROOT / 'docs/reports/assets/dmf_spt_seed_sampling_trees.png'
TOL = 1e-8
BLOCK = 2048
G = 1.3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def native_audit(values, label):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f'{label}: nonfinite information')
    bad = values < -TOL
    if bad.any():
        raise RuntimeError(f'{label}: minimum={values.min():.12g} bits; threshold={-TOL}; affected_count={bad.sum()}')
    return {'minimum_bits': float(values.min()), 'tolerance_bits': TOL,
            'numerical_zero_band_count': int(((values < 0) & ~bad).sum()), 'significant_violation_count': 0}


def protocol(seeds, counts):
    return {'version': 1, 'G': G, 'seeds': seeds, 'sample_counts': counts,
            'source_sha256': digest(SOURCE), 'batch_size': BLOCK, 'batch_seed_stride': 1000,
            'support': [.3, .7], 'source_and_target': 'all 100 E + 100 I scalar states',
            'horizon_steps': 300, 'dt': .001, 'sigma': .01, 'ridge': 1e-6,
            'boundary': 'none', 'JFIC_reference_G': 1., 'exact_max_size': 8,
            'candidate_policy': 'deterministic spectral/average-linkage/original-order prefixes; no supplements',
            'estimator': 'existing Gaussian conditional covariance; documented 200-D TM cost exception',
            'estimator_limitation': 'tests convergence of this affine/Gaussian approximation, not exact nonlinear EI',
            'syn_tolerance_bits': TOL, 'source_seed_offset': 31000, 'noise_seed_offset': 31017,
            'method_sha256': {name: digest(ROOT / name) for name in (
                'scripts/spt.py', 'scripts/analyze_runge_slp_pc60_xi_hierarchy.py',
                'scripts/analyze_dmf_critical_phi_hierarchy_topology.py',
                'scripts/run_dmf_diffusive_fullstate_control.py', 'scripts/run_dmf_paired_spt_pilot.py',
                'scripts/run_dmf_fixed_uniform_multihorizon.py', 'scripts/validate_dmf_83_region_oracle_phi_eid.py',
                'scripts/plot_dmf_schaefer100_tree_examples.py',
                'exp/brain/dmf_fig6.py')},
            'practical_negligibility_rule': {'max_unmatched_clades_per_seed_pair': 1,
                'max_roi_pair_lca_size_difference_percent': 1.,
                'max_frozen_syn_L1_difference_percent_of_root': 1., 'overall_xi_seed_cv_percent': 1.},
            'inference': 'descriptive across 8 independent simulation seeds; 28 overlapping pairs are not independent replicates'}


def generate_covariances(seed, counts, config, out, dmf, sc, jf):
    paths = {n: out / f'cov_seed{seed:02d}_N{n:06d}.npz' for n in counts}
    cached = {}
    for n, path in paths.items():
        if path.exists():
            with np.load(path) as a:
                if json.loads(str(a['config_json'].item())) != config:
                    raise RuntimeError(f'Covariance cache protocol changed: {path}')
                cached[n] = a['conditional_covariance'].copy()
    if len(cached) == len(counts):
        return paths
    bases = list(BASE_DIR.glob(f'seeds*/covariance/seed{seed:02d}_G1.30.npz'))
    if len(bases) != 1:
        raise RuntimeError(f'Expected one existing baseline covariance for seed={seed}')
    with np.load(bases[0]) as a:
        base = a['conditional_covariance'].copy()
        old = json.loads(str(a['config_json'].item()))
        baseline_source_sha = str(a['input_sha256'].item())
    for key in ('source_sha256', 'support', 'dt', 'sigma', 'ridge', 'boundary', 'JFIC_reference_G'):
        if old[key] != config[key]:
            raise RuntimeError(f'Baseline simulation control differs: {key}')
    if old['sample_count'] != BLOCK or old['horizon'] != 300:
        raise RuntimeError('Unexpected baseline sample count/horizon')
    xs, ys = [], []
    params = dmf.DMFParameters(t_total=1, burn_in=0, dt=.001, sigma=.01)
    started = time.monotonic()
    for batch in range(max(counts) // BLOCK):
        batch_seed = seed + batch * 1000
        se, si = paired_sources(batch_seed, BLOCK, 100)
        x = np.concatenate((se, si), axis=1)
        if batch == 0 and hashlib.sha256(x.tobytes()).hexdigest() != baseline_source_sha:
            raise RuntimeError('Original intervention prefix changed')
        te, ti = rollout(dmf, se, si, connectivity=sc, coupling_g=G, j_fic=jf, parameters=params,
                         mode='direct', state_boundary='none', horizon=300,
                         rng=np.random.default_rng(noise_seed(batch_seed)))
        y = np.concatenate((te, ti), axis=1)
        if not np.isfinite(y).all():
            raise RuntimeError(f'Nonfinite target: seed={seed}, batch={batch}')
        xs.append(x)
        ys.append(y)
        n = len(xs) * BLOCK
        if n in counts:
            source, target = np.concatenate(xs), np.concatenate(ys)
            _, cov, _ = conditional_source_covariance(standardize(source)[0], standardize(target)[0], ridge=1e-6)
            if n == BLOCK:
                prefix_error = float(np.max(np.abs(cov - base)))
                if prefix_error > 1e-9:
                    raise RuntimeError(f'Baseline target/noise prefix changed: error={prefix_error}')
            if n in cached and not np.allclose(cov, cached[n], atol=1e-12, rtol=0):
                raise RuntimeError(f'Cached nested prefix changed at N={n}')
            if n not in cached:
                np.savez_compressed(paths[n], conditional_covariance=cov, sample_count=n, seed=seed, G=G,
                    config_json=json.dumps(config), baseline_covariance_path=str(bases[0]),
                    baseline_covariance_sha256=digest(bases[0]), baseline_prefix_error=prefix_error,
                    input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                    target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                    cumulative_simulation_seconds=time.monotonic() - started)
            print(f'covariance seed={seed} N={n} prefix_error={prefix_error:.3g}', flush=True)
    return paths


def frozen_values(oracle, reference):
    internals = [n for n in flatten_nodes(reference) if n.children]
    syn = np.asarray([oracle.xi(n.indices) - sum(oracle.xi(c.indices) for c in n.children) for n in internals])
    leaves = np.asarray([oracle.xi((i,)) for i in range(100)])
    audit = native_audit(np.concatenate((syn, leaves)), 'Frozen reference tree')
    error = float(syn.sum() + leaves.sum() - oracle.xi(range(100)))
    if abs(error) > TOL:
        raise RuntimeError(f'Frozen tree budget failed: {error} bits')
    audit['closure_error_bits'] = error
    return syn.tolist(), audit


def build_condition(seed, count, covariance_path, config, out, reference, old_rows):
    path = out / f'tree_seed{seed:02d}_N{count:06d}.json'
    cov_sha = digest(covariance_path)
    if path.exists():
        record = json.loads(path.read_text())
        if record['config'] != config or record['covariance_sha256'] != cov_sha:
            raise RuntimeError(f'Tree cache protocol changed: {path}')
        audit_tree(_node_from_record(record['tree']), record['overall_xi_bits'])
        return record
    with np.load(covariance_path) as a:
        conditional = a['conditional_covariance'].copy()
        prefix_error = float(a['baseline_prefix_error'])
    started = time.monotonic()
    oracle = CachedScalarLogdetXiOracle(conditional, [(i, i + 100) for i in range(100)])
    reused = None
    if count == BLOCK and seed in old_rows:
        old_path = Path(old_rows[seed]['cache'])
        old = json.loads(old_path.read_text())
        if old['config']['exact_max_size'] != 8:
            raise RuntimeError('Old tree uses another exact search limit')
        for name in ('scripts/spt.py', 'scripts/analyze_runge_slp_pc60_xi_hierarchy.py'):
            if old['config']['search_source_sha256'][name] != config['method_sha256'][name]:
                raise RuntimeError('Cached baseline tree search code changed')
        tree = _node_from_record(old['tree'])
        # Audit all original atoms against the reconstructed covariance oracle.
        errors = [abs(n.xi_bits - oracle.xi(n.indices)) for n in flatten_nodes(tree)]
        if max(errors) > TOL:
            raise RuntimeError('Old tree does not match the replayed baseline oracle')
        reused = {'path': str(old_path), 'sha256': digest(old_path), 'maximum_node_xi_error_bits': max(errors)}
        pair_zeros, search = old['pair_tolerance_zero_count'], {'candidate_count': old['candidate_split_count'],
                                                               'tolerance_zero_count': old['split_tolerance_zero_count']}
    else:
        affinity, pair_zeros = pairwise_syn_affinity(oracle, 100, tolerance=TOL)
        search = {}
        tree = build_scalable_hierarchy(tuple(range(100)), oracle, affinity, exact_max_size=8,
                                       tolerance=TOL, audit=search)
    overall = oracle.xi(range(100))
    validation = audit_tree(tree, overall)
    frozen_syn, frozen_audit = frozen_values(oracle, reference)
    record = {'seed': seed, 'G': G, 'sample_count': count, 'config': config,
        'covariance_path': str(covariance_path), 'covariance_sha256': cov_sha,
        'baseline_prefix_error': prefix_error, 'reused_tree': reused,
        'overall_xi_bits': overall, 'overall_xi_nats': overall * math.log(2),
        'tree': _node_record(tree), 'tree_metrics': _tree_metrics(tree), 'validation': validation,
        'pair_tolerance_zero_count': pair_zeros, 'search_audit': search,
        'scalar_cache_check_error_bits': oracle.legacy_maximum_error_bits,
        'frozen_reference_syn_bits': frozen_syn, 'frozen_reference_audit': frozen_audit,
        'elapsed_seconds': time.monotonic() - started}
    save_json(path, record)
    print(f'tree seed={seed} N={count} Xi={record["overall_xi_nats"]:.4f} nats '
          f'depth={record["tree_metrics"]["maximum_depth"]} seconds={record["elapsed_seconds"]:.1f}', flush=True)
    return record


def shape_signature(tree):
    if not tree.children:
        return 'L'
    return '(' + ','.join(sorted(shape_signature(c) for c in tree.children)) + ')'


def pair_metrics(first, second):
    ta, tb = _node_from_record(first['tree']), _node_from_record(second['tree'])
    a, b = nontrivial_clades(ta), nontrivial_clades(tb)
    small_a, small_b = {c for c in a if len(c) <= 20}, {c for c in b if len(c) <= 20}
    mean_root = (first['overall_xi_bits'] + second['overall_xi_bits']) / 2
    return {'first_seed': first['seed'], 'second_seed': second['seed'],
        'unmatched_clades': 98 - len(a & b), 'clade_agreement_percent': 100 * len(a & b) / 98,
        'small_module_jaccard_percent': 100 * len(small_a & small_b) / len(small_a | small_b),
        'lca_difference_percent': float(np.mean(np.abs(lca_sizes(ta) - lca_sizes(tb)))),
        'frozen_syn_L1_percent': float(100 * np.abs(np.asarray(first['frozen_reference_syn_bits'])
                                                   - np.asarray(second['frozen_reference_syn_bits'])).sum() / mean_root),
        'same_unlabelled_shape': shape_signature(ta) == shape_signature(tb),
        'absolute_depth_difference': abs(first['tree_metrics']['maximum_depth'] - second['tree_metrics']['maximum_depth'])}


def summarize(records, config):
    by_count = []
    for count in config['sample_counts']:
        current = [r for r in records if r['sample_count'] == count]
        pairs = [pair_metrics(a, b) for a, b in itertools.combinations(current, 2)]
        xi = np.asarray([r['overall_xi_nats'] for r in current])
        row = {'sample_count': count, 'seed_count': len(current), 'pairs': pairs,
               'overall_xi_mean_nats': float(xi.mean()), 'overall_xi_sd_nats': float(xi.std(ddof=1)),
               'overall_xi_cv_percent': float(100 * xi.std(ddof=1) / xi.mean()),
               'exact_shape_pair_agreement_percent': 100 * np.mean([p['same_unlabelled_shape'] for p in pairs]),
               'depth_by_seed': {r['seed']: r['tree_metrics']['maximum_depth'] for r in current},
               'colless_by_seed': {r['seed']: r['tree_metrics']['normalized_colless_imbalance'] for r in current}}
        for field in ('clade_agreement_percent', 'unmatched_clades', 'small_module_jaccard_percent',
                      'lca_difference_percent', 'frozen_syn_L1_percent', 'absolute_depth_difference'):
            v = [p[field] for p in pairs]
            row[field] = {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))}
        rule = config['practical_negligibility_rule']
        row['practically_negligible'] = (row['unmatched_clades']['max'] <= rule['max_unmatched_clades_per_seed_pair']
            and row['lca_difference_percent']['max'] <= rule['max_roi_pair_lca_size_difference_percent']
            and row['frozen_syn_L1_percent']['max'] <= rule['max_frozen_syn_L1_difference_percent_of_root']
            and row['overall_xi_cv_percent'] <= rule['overall_xi_seed_cv_percent'])
        by_count.append(row)
    within_seed = []
    for seed in config['seeds']:
        current = sorted((r for r in records if r['seed'] == seed), key=lambda r: r['sample_count'])
        for a, b in zip(current, current[1:]):
            within_seed.append({'seed': seed, 'first_N': a['sample_count'], 'second_N': b['sample_count'], **pair_metrics(a, b)})
    return {'status': 'complete', 'config': config, 'by_sample_count': by_count, 'within_seed_prefix_comparisons': within_seed,
        'condition_paths': [r['covariance_path'].replace('cov_seed', 'tree_seed').replace('.npz', '.json') for r in records]}


def plot_results(summary, records, output):
    counts = summary['config']['sample_counts']
    with mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
        fig.subplots_adjust(left=.085, right=.87, top=.88, bottom=.12, hspace=.39, wspace=.34)
        seed_count = len(summary['config']['seeds'])
        pair_count = seed_count * (seed_count - 1) // 2
        specs = [('clade_agreement_percent', 'Shared ROI subtrees (%)', '#267A70'),
                 ('lca_difference_percent', 'ROI-pair hierarchy difference (%)', '#446B8A'),
                 (None, r'Overall $\Xi$ (nats)', None),
                 ('frozen_syn_L1_percent', r'Frozen split Syn difference (% of $\Xi$)', '#A66B35')]
        for letter, ax, (field, ylabel, color) in zip('abcd', axes.flat, specs):
            ax.text(-.16, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=13)
            ax.set_xscale('log', base=2)
            ax.set_xticks(counts, [f'{n:,}' for n in counts])
            ax.set_xlabel('Intervention samples')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', color='#E8ECEF', linewidth=.6)
            if field:
                rows = summary['by_sample_count']
                means = [r[field]['mean'] for r in rows]
                low, high = [r[field]['min'] for r in rows], [r[field]['max'] for r in rows]
                ax.fill_between(counts, low, high, color=color, alpha=.13)
                ax.plot(counts, means, '-o', color=color, linewidth=1.5, markersize=5)
                if field == 'clade_agreement_percent':
                    ax.set_ylim(0, 100)
                else:
                    ax.set_ylim(bottom=0)
                ax.set_title(f'{pair_count} seed pairs: mean and range', fontsize=9, color='#53646C', pad=8)
            else:
                colors = mpl.colormaps['tab10'](np.arange(8))
                for color, seed in zip(colors, summary['config']['seeds']):
                    ys = [next(r['overall_xi_nats'] for r in records if r['seed'] == seed and r['sample_count'] == n) for n in counts]
                    ax.plot(counts, ys, '-o', color=color, linewidth=1.1, markersize=3.5, label=f'Seed {seed}')
                ax.set_title(f'{seed_count} independent simulation seeds', fontsize=9, color='#53646C', pad=8)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.89, .32), frameon=False, fontsize=8)
        fig.suptitle('G = 1.3  |  Sample-size convergence of the same binary SPT', fontsize=12, x=.48, y=.965)
        fig.text(.085, .035, 'Nested intervention and noise prefixes; unchanged Gaussian estimator and deterministic split search.\n'
                 f'Shaded ranges are descriptive: the {pair_count} overlapping seed pairs are not independent replicates.',
                 fontsize=8, color='#53646C', linespacing=1.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=400, bbox_inches='tight', facecolor='white',
                    metadata={'SPT sampling convergence': json.dumps(summary, allow_nan=False)})
        plt.close(fig)


def plot_tree_examples(summary, records, output):
    """Show the first three declared seeds, without selecting for agreement."""
    from matplotlib.patches import Patch
    from scripts.plot_dmf_spt_collapsed_tree import render_collapsed_tree, add_colorbar
    from scripts.plot_dmf_schaefer100_tree_examples import NETWORK_COLORS
    with np.load(ROOT / 'results/dmf_schaefer100/full/critical_yeo7.npz', allow_pickle=True) as a:
        labels, membership, names = a['region_labels'].tolist(), a['network_membership'], a['network_names'].tolist()
    seeds, counts = summary['config']['seeds'][:3], summary['config']['sample_counts']
    trees = {(r['seed'], r['sample_count']): _node_from_record(r['tree']) for r in records}
    maximum = max(100 * n.syn_bits / tree.xi_bits for tree in trees.values()
                  for n in flatten_nodes(tree) if n.children)
    color_max = max(3., math.ceil(maximum * 2) / 2)
    with mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans']}):
        fig, axes = plt.subplots(len(seeds), len(counts), figsize=(13, 3.1 * len(seeds) + 1.6), squeeze=False)
        fig.subplots_adjust(left=.035, right=.985, top=.845, bottom=.085, wspace=.12, hspace=.22)
        audits = []
        for row, seed in enumerate(seeds):
            for column, count in enumerate(counts):
                ax, tree = axes[row, column], trees[seed, count]
                audit = render_collapsed_tree(tree, axis=ax, labels=labels, network_membership=membership,
                    network_names=names, seed=seed, coupling_g=G, color_max=color_max,
                    show_heading=False, show_roi_labels=False, node_label_limit=0)
                ax.set_ylim(-.08, 1.1)
                ax.set_title(f'Seed {seed}  |  N={count:,}  |  '+rf'$\Xi={tree.xi_bits * math.log(2):.2f}$ nats',
                             fontsize=9, pad=9)
                if column == 0:
                    ax.text(-.012, .5, f'Seed {seed}', transform=ax.transAxes,
                            rotation=90, va='center', ha='right', fontsize=9)
                audits.append({'seed': seed, 'N': count, **audit})
        fig.suptitle(f'G = 1.3  |  The first {len(seeds)} prespecified seeds at increasing sample sizes',
                     fontsize=12, y=.99)
        fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(NETWORK_COLORS, names)],
                   loc='upper left', bbox_to_anchor=(.03, .955), ncol=4, frameon=False, fontsize=8.5)
        add_colorbar(fig, fig.add_axes([.76, .915, .20, .012]), color_max, fontsize=7)
        fig.text(.035, .025, 'Every ROI and branching module retained. Consecutive singleton exits share dashed display forks (peak local Syn).\n'
                 f'Vertical spacing is display layout; statistical comparisons use the complete original binary trees and all {len(summary["config"]["seeds"])} seeds.',
                 fontsize=8, color='#53646C', linespacing=1.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=350, bbox_inches='tight', facecolor='white',
                    metadata={'Sampling tree examples': json.dumps(audits, allow_nan=False)})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(3, 11)))
    parser.add_argument('--sample-counts', type=int, nargs='+', default=[2048, 8192, 32768])
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--figure', type=Path, default=DEFAULT_FIGURE)
    parser.add_argument('--tree-figure', type=Path, default=DEFAULT_TREE_FIGURE)
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2
            or args.sample_counts != sorted(set(args.sample_counts)) or args.sample_counts[0] != BLOCK
            or any(n % BLOCK for n in args.sample_counts)):
        parser.error('Unique >=2 seeds; increasing sample counts must start at 2048 and be multiples of 2048')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol(args.seeds, args.sample_counts)
    contract = args.output_dir / 'contract.json'
    if contract.exists() and json.loads(contract.read_text()) != config:
        raise RuntimeError('Existing experiment contract changed; use a new output directory')
    save_json(contract, config)
    old_summary = json.loads(EXAMPLES.read_text())
    old_rows = {r['seed']: r for r in old_summary['rows'] if r['coupling_g'] == G}
    reference_path = Path(old_rows[4]['cache'])
    reference = _node_from_record(json.loads(reference_path.read_text())['tree'])
    reference_record = {'seed': 4, 'G': G, 'path': str(reference_path),
                        'sha256': digest(reference_path), 'tree': _node_record(reference)}
    reference_file = args.output_dir / 'frozen_reference.json'
    if reference_file.exists() and json.loads(reference_file.read_text()) != reference_record:
        raise RuntimeError('Frozen comparison reference changed')
    save_json(reference_file, reference_record)
    if args.analysis_only:
        records = [json.loads((args.output_dir / f'tree_seed{s:02d}_N{n:06d}.json').read_text())
                   for s in args.seeds for n in args.sample_counts]
    else:
        print(f'PID={os.getpid()} seeds={args.seeds} sample_counts={args.sample_counts}', flush=True)
        with np.load(SOURCE) as a:
            sc, allg, jfic = a['connectivity'], a['G'], a['j_fic']
        index = np.flatnonzero(np.isclose(allg, 1.))
        if len(index) != 1 or sc.shape != (100, 100) or not np.array_equal(jfic, np.broadcast_to(jfic[index[0]], jfic.shape)):
            raise RuntimeError('SC / frozen JFIC controls differ')
        dmf, jf = load_dmf_module(), jfic[index[0]]
        records = []
        for seed in args.seeds:
            paths = generate_covariances(seed, args.sample_counts, config, args.output_dir, dmf, sc, jf)
            for count in args.sample_counts:
                records.append(build_condition(seed, count, paths[count], config, args.output_dir, reference, old_rows))
    summary = summarize(records, config)
    save_json(args.output_dir / 'summary.json', summary)
    plot_results(summary, records, args.figure)
    plot_tree_examples(summary, records, args.tree_figure)
    print('COMPLETE: ' + str(args.figure), flush=True)
    for row in summary['by_sample_count']:
        print(json.dumps({k: v for k, v in row.items() if k not in {'pairs', 'depth_by_seed', 'colless_by_seed'}}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
