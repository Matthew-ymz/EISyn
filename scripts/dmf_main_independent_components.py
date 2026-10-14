"""Refresh the main plate's G=0 readout without changing historical experiments.

Only three paired G=0 responses need replaying: all participants share the same
local parameters and samples. Positive-G connected components use exactly the
original affine model, so their covariance/attribution caches can be reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dmf_joint_readout import CommonTargetGame, audit_nonnegative
from scripts.dmf_response_benchmark import (
    DMF_AFFINE_DENSITY_VERSION, dmf_source_components, fit_affine_joint,
    fit_dmf_affine_joint, simulate_sources, refit_affine_joint_from_moments,
)
from scripts.dmf_subject_consistency import network_values
from scripts.plot_dmf_roi_shapley_three_g import (
    sampling_diagnostics, comparison_statistics, surface_plate,
)
from scripts.run_dmf_paired_spt_pilot import paired_sources, noise_seed
from scripts.run_dmf_subject_consistency import atomic_json, digest
from scripts.run_dmf_subject_curve_baselines import atomic_savez
from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module

BASE = ROOT / 'results/dmf_schaefer100/main_independent_components_v2'
ROI = BASE / 'roi'
FIGURE = ROOT / 'fig/dmf_schaefer100/main_independent_components_v2'
OLD_ROI = ROOT / 'results/dmf_schaefer100/roi_shapley_three_G_G1000'
DENSE = ROOT / 'results/dmf_schaefer100/subject_curves_93_dense'
PILOT = ROOT / 'results/dmf_schaefer100/subject_consistency_pilot'
SOURCE = ROOT / 'results/dmf_schaefer100/source/group_mean_native_mean_rate.npz'
SURFACE = ROOT / 'results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz'
TOL = 1e-8  # native nats; preserve all raw values
VERSION = 'main-independent-components-readout-v2'
LEGACY_DENSITY = ROOT / 'results/dmf_schaefer100/critical_coalitions/density'


def _archive(path):
    with np.load(path) as a:
        return {k: a[k].copy() for k in a.files}


def refit_disconnected_curves(dense, inputs, identity_sha):
    """Reuse OLS sufficient statistics; cache only the costly derived metrics."""
    source_moments, source_hashes = {}, {}
    for seed in (3, 4, 5):
        x = np.concatenate(paired_sources(seed, 2048, 100), axis=1)
        xz = (x-x.mean(0))/x.std(0, ddof=1)
        source_moments[seed] = np.cov(xz, rowvar=False)
        source_hashes[seed] = hashlib.sha256(x.tobytes()).hexdigest()
    legacy_contract = digest(LEGACY_DENSITY.parent / 'contract.json')
    rows = []
    for pi, (subject, sc) in enumerate(zip(inputs['subject_ids'], inputs['connectivity'])):
        blocks = dmf_source_components(sc, 1.)
        if len(blocks) == 1:
            continue
        subject = str(subject)
        output = BASE / 'individual_metrics' / (subject + '.npz')
        if output.exists():
            saved = _archive(output)
            if str(saved['correction_identity_sha256']) != identity_sha:
                raise ValueError('Individual refit belongs to another correction')
            data = saved['metrics']
            audit = json.loads(str(saved['audit_json']))
            legacy_hashes = saved['legacy_cache_sha256'].tolist()
        else:
            data = np.empty((3, 40, 3)); queried = []; legacy_hashes = []
            for si, seed in enumerate((3, 4, 5)):
                for gi, coupling in enumerate(dense['G'][1:]):
                    path = LEGACY_DENSITY / f'{subject}_G{coupling:.2f}_seed{seed}.npz'
                    with np.load(path) as a:
                        if (str(a['contract_sha256']) != legacy_contract
                                or str(a['source_sha256']) != source_hashes[seed]
                                or int(a['noise_seed']) != noise_seed(seed)):
                            raise ValueError(f'Legacy density identity differs: {path}')
                        covariance = a['covariance'].copy()
                    old = CommonTargetGame(covariance).totals()
                    np.testing.assert_allclose(old['xi_nats'], dense['xi'][pi, si, gi+1], atol=TOL, rtol=0)
                    corrected = refit_affine_joint_from_moments(covariance, source_moments[seed], blocks)
                    game = CommonTargetGame(corrected); metrics = game.totals()
                    # The complete source partition by dynamical components
                    # must carry no cross-component information increment.
                    component_ei = sum(game.ei(tuple(j for j in block if j < 100)) for block in blocks)
                    component_residual = metrics['whole_ei_nats'] - component_ei
                    if abs(component_residual) > TOL:
                        raise ArithmeticError(f'Cross-component residual: {component_residual} nats')
                    queried.extend([*game.audited, *metrics.values(), component_residual])
                    data[si, gi] = [metrics['xi_nats'], metrics['whole_ei_nats'], game.scalar_ei.sum()]
                    legacy_hashes.append(digest(path))
            audit = audit_nonnegative(queried, TOL)
            output.parent.mkdir(exist_ok=True)
            atomic_savez(output, metrics=data, audit_json=json.dumps(audit),
                legacy_cache_sha256=np.asarray(legacy_hashes), subject=subject,
                correction_identity_sha256=identity_sha)
        for fi, field in enumerate(['xi', 'whole_ei', 'partial_ei_sum']):
            dense[field][pi, :, 1:] = data[..., fi]
        rows.append(dict(subject=subject, component_roi_sizes=[len(b)//2 for b in blocks],
            refitted_conditions=120, metrics_sha256=digest(output), audit=audit,
            legacy_density_cache_count=len(legacy_hashes)))
        print(f'Individual components: {subject}; {len(rows)} subjects complete', flush=True)
    return rows


def refresh():
    old_summary = json.loads((OLD_ROI / 'summary.json').read_text())
    old_roi = _archive(OLD_ROI / 'roi_shapley.npz')
    dense = _archive(DENSE / 'analysis_curves.npz')
    inputs = _archive(DENSE / 'inputs.npz')
    pilot = _archive(PILOT / 'inputs.npz')
    contract = json.loads((DENSE / 'contract.json').read_text())
    if digest(SOURCE) != contract['reference_source_sha256']:
        raise ValueError('Frozen SC/JFIC source changed')
    if (contract['sample_count'] != 2048 or contract['support'] != [.3, .7]
            or contract['horizon_steps'] != 300 or contract['sigma'] != .01
            or contract['dt_seconds'] != .001 or contract['ridge'] != 1e-6):
        raise ValueError('Unexpected native response protocol')
    if dense['subject_ids'].tolist() != contract['subject_ids']:
        raise ValueError('Dense subject identities changed')
    disconnected = [str(s) for s, sc in zip(inputs['subject_ids'], inputs['connectivity'])
                    if len(dmf_source_components(sc, 1.)) > 1]
    mean_index = pilot['subject_ids'].tolist().index('group_mean_93')
    sc, jf = pilot['connectivity'][mean_index], pilot['j_fic']
    np.testing.assert_array_equal(jf, inputs['j_fic'])
    with np.load(SOURCE) as a:
        np.testing.assert_array_equal(sc, a['connectivity'])
        np.testing.assert_array_equal(a['j_fic'], np.broadcast_to(jf, a['j_fic'].shape))
    g, seeds = old_roi['G'], old_roi['seeds']
    if g.tolist() != [0., 1.3, 10.] or seeds.tolist() != [3, 4, 5]:
        raise ValueError('Unexpected main-plate G/seed set')
    labels, membership = old_roi['region_labels'], old_roi['network_membership']
    groups = [tuple(np.flatnonzero(membership == i)) for i in range(7)]
    identity = dict(version=VERSION, density_version=DMF_AFFINE_DENSITY_VERSION,
        old_curves_sha256=digest(DENSE / 'analysis_curves.npz'),
        old_roi_sha256=digest(OLD_ROI / 'roi_shapley.npz'),
        old_roi_summary_sha256=digest(OLD_ROI / 'summary.json'),
        fixed_window_sha256=digest(DENSE / 'fixed_window_view.json'),
        source_sha256=digest(SOURCE), inputs_sha256=digest(DENSE / 'inputs.npz'),
        kernel_sha256=digest(ROOT / 'scripts/dmf_response_benchmark.py'),
        updater_sha256=digest(Path(__file__)),
        positive_G_rule='Connected SCs reuse the original fit; disconnected SCs refit from OLS sufficient statistics',
        disconnected_subject_ids=disconnected,
        legacy_density_contract_sha256=digest(LEGACY_DENSITY.parent/'contract.json'),
        G0_rule='Refit each E/I pair and its residual covariance; assemble one joint TM',
        tolerance_nats=TOL)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    BASE.mkdir(parents=True, exist_ok=True); ROI.mkdir(exist_ok=True)
    manifest = BASE / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity:
            raise ValueError('Correction provenance changed; use a new versioned output')
    conditions, totals, games = [], [], {}
    dmf = load_dmf_module(); parameters = dmf.DMFParameters(dt=.001, sigma=.01)
    for gi, coupling in enumerate(g):
        row_totals = []
        for si, seed in enumerate(seeds):
            old = next(r for r in old_summary['provenance']['conditions']
                       if r['G'] == coupling and r['seed'] == seed)
            old_path = ROOT / old['cache']
            if digest(old_path) != old['cache_sha256']:
                raise ValueError(f'Legacy density changed: {old_path}')
            x = np.concatenate(paired_sources(int(seed), 2048, 100), axis=1)
            input_sha = hashlib.sha256(x.tobytes()).hexdigest()
            with np.load(old_path) as a:
                legacy_covariance = a['covariance'].copy()
                if input_sha != str(a['source_sha256']) or int(a['noise_seed']) != noise_seed(seed):
                    raise ValueError('Actual response source/noise identity differs')
            path = BASE / 'conditions' / old_path.name
            if path.exists():
                with np.load(path) as a:
                    if str(a['correction_identity_sha256']) != identity_sha:
                        raise ValueError('Correction condition belongs to another protocol')
                    covariance = a['covariance'].copy()
                    diagnostics = json.loads(str(a['diagnostics_json']))
            elif coupling == 0:
                y, diagnostics = simulate_sources(dmf, x, sc, jf, 0., parameters, noise_seed(seed))
                np.testing.assert_allclose(fit_affine_joint(x, y, .2), legacy_covariance,
                                           atol=1e-12, rtol=0)
                if diagnostics['outside_state_count'] or diagnostics['abnormal_rate_count']:
                    raise ArithmeticError(f'Invalid G=0 response: {diagnostics}')
                covariance = fit_dmf_affine_joint(x, y, .2, connectivity=sc, coupling=0.)
            else:
                covariance = legacy_covariance
                diagnostics = dict(reused_connected_component=True)
            game = CommonTargetGame(covariance); metrics = game.totals()
            within, between, network_total, network_audit = network_values(game, groups)
            audit = audit_nonnegative([*game.audited, *metrics.values(), network_total], TOL)
            if coupling == 0 and max(abs(metrics['cross_roi_nats']), abs(network_total)) > TOL:
                raise ArithmeticError('Corrected G=0 mechanism fails its independence control')
            if coupling != 0:
                np.testing.assert_array_equal(covariance, legacy_covariance)
                np.testing.assert_allclose(metrics['xi_nats'], old_roi['xi_nats'][gi, si], atol=TOL, rtol=0)
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                atomic_savez(path, covariance=covariance, conditional=game.conditional,
                    source_sha256=input_sha, noise_seed=noise_seed(seed),
                    G=float(coupling), seed=int(seed), density_version=DMF_AFFINE_DENSITY_VERSION,
                    correction_identity_sha256=identity_sha, legacy_cache_sha256=old['cache_sha256'],
                    diagnostics_json=json.dumps(diagnostics), metrics_json=json.dumps(metrics),
                    within_network_xi=within, between_network_shapley=between,
                    network_audit_json=json.dumps(network_audit))
            conditions.append(dict(G=float(coupling), seed=int(seed), cache=str(path.relative_to(ROOT)),
                cache_sha256=digest(path), legacy_cache=old['cache'], legacy_cache_sha256=old['cache_sha256'],
                source_sha256=input_sha, noise_seed=noise_seed(seed),
                refitted=bool(coupling == 0), density_audit=audit, network_audit=network_audit))
            games[gi, si] = game
            row_totals.append([metrics[k] for k in ['xi_nats', 'roi_local_xi_nats', 'cross_roi_nats']])
            print(f'G={coupling:g} seed={seed}: Xi={metrics["xi_nats"]:.9f}; between={network_total:.3g} nats', flush=True)
        totals.append(row_totals)
    totals = np.asarray(totals)
    local = np.array([[games[gi, si].local_xi for si in range(3)] for gi in range(3)])
    samples = old_roi['paired_samples_nats'].copy()
    samples[:, 0] = local[0]  # Analytic Shapley of the additive independent-ROI game.
    attribution = samples.mean(0); attribution[0] = local[0]
    cross = attribution - local
    pp, condition_se, mean_se, diagnostics = sampling_diagnostics(samples, totals[..., 0])
    shares = pp.mean(0); shares[0] = 100*local[0]/totals[0, :, 0, None]
    condition_se[0] = 0; mean_se[0] = 0  # Exact game solution; not a seed-variance claim.
    np.testing.assert_allclose(shares.sum(-1), 100, atol=TOL, rtol=0)
    np.testing.assert_array_equal(samples[:, 1:], old_roi['paired_samples_nats'][:, 1:])
    np.testing.assert_array_equal(totals[1:, :, 0], old_roi['xi_nats'][1:])
    gi0 = int(np.flatnonzero(dense['G'] == 0)[0])
    old_peaks = dense['xi'].argmax(-1).copy()
    for field in ['xi', 'whole_ei', 'partial_ei_sum']:
        np.testing.assert_allclose(dense[field][:, :, gi0],
            np.broadcast_to(dense[field][-1, :, gi0], dense[field][:, :, gi0].shape), atol=TOL, rtol=0)
    for si in range(3):
        game = games[0, si]
        dense['xi'][:, si, gi0] = totals[0, si, 0]
        dense['whole_ei'][:, si, gi0] = game.totals()['whole_ei_nats']
        dense['partial_ei_sum'][:, si, gi0] = game.scalar_ei.sum()
    individual_rows = refit_disconnected_curves(dense, inputs, identity_sha)
    original = _archive(DENSE / 'analysis_curves.npz')
    changed = {}
    for field in ['xi', 'whole_ei', 'partial_ei_sum']:
        mask = np.ones(dense[field].shape, bool); mask[:, :, gi0] = False
        for pi, subject in enumerate(dense['subject_ids']):
            if str(subject) in disconnected:
                mask[pi] = False
        np.testing.assert_array_equal(dense[field][mask], original[field][mask])
        changed[field] = int(np.count_nonzero(dense[field] != original[field]))
    for field in ['rate', 'phi_r', 'wms', 'susceptibility']:
        np.testing.assert_array_equal(dense[field], original[field])
    closure = float(np.abs(dense['whole_ei']-dense['partial_ei_sum']-dense['xi']).max())
    if closure > TOL:
        raise ArithmeticError(f'Corrected dense EI budget fails: {closure} nats')
    dense['base_contract_sha256'] = dense.pop('contract_sha256')
    dense['correction_identity_sha256'] = np.array(identity_sha)
    dense['density_version'] = np.array(DMF_AFFINE_DENSITY_VERSION)
    atomic_savez(BASE / 'analysis_curves.npz', **dense)
    sampling = json.loads(str(old_roi['sampling_json']))
    sampling.update(G0_method='Exact additive local E/I attribution; no permutation estimation',
                    inherited_positive_G_sampling_sha256=digest(OLD_ROI/'roi_shapley.npz'))
    atomic_savez(ROI / 'roi_shapley.npz', paired_samples_nats=samples,
        roi_shapley_nats=attribution, xi_nats=totals[..., 0], roi_local_xi_nats=local,
        G=g, seeds=seeds, region_labels=labels, network_membership=membership,
        network_names=old_roi['network_names'], sampling_json=json.dumps(sampling),
        identity_json=json.dumps(identity))
    delta_samples = np.array([pp[:, b]-pp[:, a] for a, b in [(0, 1), (1, 2), (0, 2)]])
    atomic_savez(ROI / 'roi_tables.npz', G=g, seeds=seeds, region_labels=labels,
        roi_shapley_nats=attribution, share_percent=shares, mean_share_percent=shares.mean(1),
        condition_mc_se_pp=condition_se, mean_mc_se_pp=mean_se, roi_local_xi_nats=local,
        roi_cross_shapley_nats=cross, xi_nats=totals[..., 0],
        condition_paired_difference_mc_se_pp=delta_samples.std(1, ddof=1)/np.sqrt(len(samples)),
        mean_paired_difference_mc_se_pp=delta_samples.mean(2).std(1, ddof=1)/np.sqrt(len(samples)),
        contrast_G=np.array([[0., 1.3], [1.3, 10.], [0., 10.]]))
    summary = dict(version=VERSION, G=g.tolist(), seeds=seeds.tolist(),
        provenance=dict(protocol={**old_summary['provenance']['protocol'],
            'density_version':DMF_AFFINE_DENSITY_VERSION, 'density_correction_identity':identity_sha},
            conditions=conditions, migration=identity),
        nonnegative_audits=dict(xi=audit_nonnegative(totals[..., 0], TOL),
            local=audit_nonnegative(local, TOL), attribution=audit_nonnegative(attribution, TOL),
            cross_attribution=audit_nonnegative(cross, TOL)),
        normalization=old_summary['normalization'], sampling=sampling, diagnostics=diagnostics,
        analysis=comparison_statistics(pp, labels),
        state_totals=[dict(G=float(coupling), xi_mean_nats=float(totals[i, :, 0].mean()),
            xi_seed_sd_nats=float(totals[i, :, 0].std(ddof=1)),
            local_mean_nats=float(totals[i, :, 1].mean()), cross_mean_nats=float(totals[i, :, 2].mean()))
            for i, coupling in enumerate(g)],
        manuscript_recheck={**old_summary['manuscript_recheck'], 'checked_date':'2026-10-14',
            'consistency':'Known independent native DMF components now factorize. Uniform-to-Gaussian moment approximation and within-component finite-sample bias remain; no feature lift or S14 correction.'},
        surface_sha256=digest(SURFACE),
        G0_hierarchy=dict(seed=4, method='Independent ROI leaves; no identifiable cross-ROI hierarchy',
            local_roi_xi_nats=local[0, 1].tolist(), full_xi_nats=float(totals[0, 1, 0]),
            cross_roi_nats=float(totals[0, 1, 2]), leaf_count=100))
    state_notes = [f"$\\Xi$ = {r['xi_mean_nats']:.2f} $\\pm$ {r['xi_seed_sd_nats']:.2f} nats"
                   for r in summary['state_totals']]
    summary['plot'] = surface_plate(shares.mean(1), labels, [rf'$G={v:g}$' for v in g],
        state_notes, FIGURE/'roi_share_three_G', observed_range=True,
        colorbar_label=r'ROI attribution to overall $\Xi$ (%)',
        footer='G=0: local E/I only · connected G>0: original cached attributions · seeds 3, 4, 5')
    atomic_json(ROI / 'summary.json', summary)
    result = dict(identity=identity, identity_sha256=identity_sha,
        corrected_curves_sha256=digest(BASE/'analysis_curves.npz'),
        corrected_roi_sha256=digest(ROI/'roi_shapley.npz'),
        corrected_roi_tables_sha256=digest(ROI/'roi_tables.npz'),
        corrected_roi_summary_sha256=digest(ROI/'summary.json'),
        corrected_surface_sha256=digest(FIGURE/'roi_share_three_G.png'),
        curve_changed_counts=changed,
        changed_seed_peak_count=int(np.count_nonzero(old_peaks != dense['xi'].argmax(-1))),
        disconnected_subject_count=len(disconnected), individual_refits=individual_rows,
        dense_closure_max_nats=closure,
        current_main_scope='Corrected current main plate only; historical appendix figures/caches are preserved and not relabeled')
    atomic_json(manifest, result)
    print(json.dumps(summary['state_totals']), flush=True)


def load_manifest():
    record = json.loads((BASE / 'manifest.json').read_text())
    for path, key in [(DENSE/'analysis_curves.npz', 'old_curves_sha256'),
                      (ROOT/'scripts/dmf_response_benchmark.py', 'kernel_sha256'),
                      (Path(__file__), 'updater_sha256')]:
        if digest(path) != record['identity'][key]:
            raise ValueError(f'Correction input changed: {path}')
    for path, key in [(BASE/'analysis_curves.npz', 'corrected_curves_sha256'),
                      (ROI/'roi_shapley.npz', 'corrected_roi_sha256'),
                      (ROI/'roi_tables.npz', 'corrected_roi_tables_sha256'),
                      (ROI/'summary.json', 'corrected_roi_summary_sha256'),
                      (FIGURE/'roi_share_three_G.png', 'corrected_surface_sha256')]:
        if digest(path) != record[key]:
            raise ValueError(f'Corrected output changed: {path}')
    return record


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    with threadpool_limits(limits=1):
        refresh()
