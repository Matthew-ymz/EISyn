"""Natural fixed-full-target WMS and periodic-output SHAP for simplified A/C."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.special import i0e, i1e, logsumexp
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_kuramoto_ac_selected import SelectedCase, labels, mask
from scripts.spt_kuramoto_finite_tm import (
    TWO_PI, DEFENSIVE, WIDE_FRACTION, WIDE_SCALE, proposal_sample,
    proposal_log_prob, check_time, save_json,
)
from scripts.spt_kuramoto_natural_source import RotationalSource, fit_source
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_local_hyperedge_kuramoto import wrap
from scripts.spt import nontrivial_bipartitions

VERSION = 'kuramoto-natural-periodic-vector-magnitude-v2'
MEMBERSHIP = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(bool)


def natural_data(context):
    path = context.folder / 'natural_chains.npz'
    network = SimplifiedKuramoto(context.case)
    meta = {'version': VERSION, 'case': context.case, 'seed': 0,
        'stream': [20261015, 0, 90001], 'chains': 64, 'burn': 256, 'retained_steps': 512,
        'tau': .2, 'dt': .0025, 'noise': .08,
        'network': [{'edge': list(e), 'weight': w} for e, w in network.weights.items()],
        'trajectory': 'adjacent natural states; identical stored data for WMS and SHAP'}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data['meta'])) != meta:
                raise ValueError('Natural trajectory protocol changed')
            return data['states'].copy(), meta
    rng = np.random.default_rng(np.random.SeedSequence(meta['stream']))
    state = network.sample_source(64, rng)
    for _ in range(meta['burn']):
        check_time(context.deadline()); state = network.step(state, rng, dt=.0025)
    print(json.dumps({'case': context.case, 'stage': 'natural burn complete'}), flush=True)
    states = [state.copy()]
    for _ in range(meta['retained_steps']):
        check_time(context.deadline()); state = network.step(state, rng, dt=.0025)
        states.append(state.copy())
    states = np.asarray(states)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, states=states, meta=json.dumps(meta))
    temporary.replace(path)
    return states, meta


def natural_proposal(model, prior, source, target, missing):
    """Joint mode proposal; its local component is only an efficiency device."""
    value = source.copy()
    value[:, missing] = target[:, missing]  # Never initialize with missing truth.
    qmm = prior.precision[np.ix_(missing, missing)]
    for _ in range(12):
        mean, jac = model.mean(value, jacobian=True)
        derivative = jac[:, :, missing]
        precision = np.einsum('nti,t,ntj->nij', derivative, model.kappa, derivative) + qmm
        prior_residual = wrap(value - prior.local_component(value))
        gradient = (np.einsum('nti,t,nt->ni', derivative, model.kappa, np.sin(wrap(target - mean)))
                    - prior_residual @ prior.precision[missing].T)
        step = np.linalg.solve(precision, gradient)
        # Bounded proposal optimization steps; no information value is clipped.
        scale = np.minimum(1., .5 / np.maximum(abs(step).max(1), 1e-300))
        value[:, missing] = wrap(value[:, missing] + step * scale[:, None])
    mean, jac = model.mean(value, jacobian=True)
    derivative = jac[:, :, missing]
    precision = np.einsum('nti,t,ntj->nij', derivative, model.kappa, derivative) + qmm
    chol = np.linalg.cholesky(1.5 * np.linalg.inv(precision))
    return value[:, missing], chol


class NaturalInformation:
    def __init__(self, context, prior, *, outer, seed):
        self.context = context; self.prior = prior; self.outer = outer
        self.model, _, _ = context.load_model()
        points = qmc.Sobol(20, scramble=True, seed=seed, bits=30).random_base2(int(np.log2(outer))) + .5 / 2**30
        self.source = prior.sample(points[:, :11])
        self.target = self.model.sample(self.source, points[:, 11:])
        self.inner = (qmc.Sobol(10, scramble=True, seed=seed + 1, bits=30).random_base2(
            int(np.log2(outer * 256))) + .5 / 2**30).reshape(outer, 256, 10)
        self.meta = {'version': VERSION, 'model_sha256': context.digest,
                     'source_density': prior.payload(), 'outer': outer, 'max_particles': 256, 'seed': seed}
        token = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode()).hexdigest()[:24]
        self.directory = context.folder / 'natural_marginals' / token
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def marginal(self, source_mask, particles):
        key = source_mask, particles
        if key in self.cache:
            return self.cache[key]
        check_time(self.context.deadline())
        path = self.directory / f'{source_mask}_{particles}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data['meta'])) != self.meta:
                    raise ValueError('Natural marginal provenance mismatch')
                result = data['log_marginal'].copy(), json.loads(str(data['diagnostic']))
            self.cache[key] = result
            return result
        missing = np.flatnonzero(~MEMBERSHIP[source_mask])
        if not len(missing):
            result = self.model.log_prob(self.source, self.target), {'ess_minimum': None, 'maximum_weight': None}
        else:
            center, chol = natural_proposal(self.model, self.prior, self.source, self.target, missing)
            log_source_marginal = self.prior.log_prob(self.source, source_mask)
            values, ess, largest = [], [], []
            batch = max(1, 16384 // particles)
            for start in range(0, self.outer, batch):
                check_time(self.context.deadline()); stop = min(self.outer, start + batch)
                u = self.inner[start:stop, :particles]
                c, l = center[start:stop], chol[start:stop]
                normal = u[:, :, 1 + missing]
                sample = proposal_sample(c, l, normal)
                wide = proposal_sample(c, WIDE_SCALE * l, normal)
                branch = u[:, :, 0]
                sample = np.where(((branch >= DEFENSIVE) & (branch < DEFENSIVE + WIDE_FRACTION))[:, :, None], wide, sample)
                sample = np.where((branch < DEFENSIVE)[:, :, None], normal * TWO_PI - np.pi, sample)
                logq = np.logaddexp(np.logaddexp(
                    np.log(1 - DEFENSIVE - WIDE_FRACTION) + proposal_log_prob(sample, c, l),
                    np.log(WIDE_FRACTION) + proposal_log_prob(sample, c, WIDE_SCALE * l)),
                    np.log(DEFENSIVE) - len(missing) * np.log(TWO_PI))
                x = np.repeat(self.source[start:stop, None, :], particles, axis=1)
                x[:, :, missing] = sample
                flat = x.reshape(-1, 9)
                y = np.repeat(self.target[start:stop], particles, axis=0)
                # q(x_M|x_A)=q(x)/q_A(x_A); both densities are of the same
                # normalized rotational source mixture, not shuffled priors.
                weights = ((self.model.log_prob(flat, y) + self.prior.log_prob(flat))
                    .reshape(stop - start, particles) - log_source_marginal[start:stop, None] - logq)
                normalizer = logsumexp(weights, axis=1)
                w = np.exp(weights - normalizer[:, None])
                values.append(normalizer - np.log(particles))
                ess.extend(1 / (w * w).sum(1)); largest.extend(w.max(1))
            result = np.concatenate(values), {'ess_minimum': float(min(ess)), 'maximum_weight': float(max(largest))}
        if not np.isfinite(result[0]).all():
            raise ArithmeticError('Nonfinite natural marginal')
        np.savez_compressed(path, log_marginal=result[0], diagnostic=json.dumps(result[1]), meta=json.dumps(self.meta))
        self.cache[key] = result
        return result

    def information(self, source_mask, particles):
        a, da = self.marginal(source_mask, particles); b, db = self.marginal(0, particles)
        pointwise = a - b
        return float(pointwise.mean()), pointwise, (da, db)


def fixed_target_wms(context, table):
    clades, selected, residuals = [], [], []
    def visit(parent):
        if len(parent) == 1:
            return ()
        choices = [(table[mask(parent)] - table[mask(l)] - table[mask(r)], l, r)
                   for l, r in nontrivial_bipartitions(parent)]
        residuals.extend(z[0] for z in choices)
        raw, left, right = min(choices)
        selected.append({'parent': parent, 'left': left, 'right': right, 'wms_nats': raw})
        for child in (left, right):
            if len(child) > 1:
                clades.append(child)
            visit(child)
        return left, right
    root = visit(tuple(range(1, 10)))
    closure = table[511] - sum(table[1 << i] for i in range(9)) - sum(z['wms_nats'] for z in selected)
    if abs(closure) > 1e-9:
        raise ArithmeticError('Signed WMS telescoping failed')
    return context.scored(root, clades), {'nodes': selected, 'minimum_search_wms_nats': min(residuals),
        'negative_search_count': sum(v < 0 for v in residuals), 'below_minus_01_count': sum(v < -.01 for v in residuals),
        'tolerance_band_negative_count': sum(-.01 <= v < 0 for v in residuals),
        'negative_values_are_valid': True, 'clipping': False, 'closure_error_nats': closure}


def information_audit(values, tolerance=.01):
    """Audit MI itself; a signed WMS residual need not be nonnegative."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 511 or not np.isfinite(values).all():
        raise ArithmeticError('Finite complete MI tables are required')
    tables = np.column_stack((np.zeros(len(values) + 1),
                              np.vstack((values, values.mean(0)))))
    increments = np.array([tables[:, m | (1 << i)] - tables[:, m]
                           for m in range(512) for i in range(9) if not (m >> i & 1)])
    def record(x):
        return {'minimum_nats': float(x.min()),
                'tolerance_negative_count': int(np.count_nonzero((x < 0) & (x >= -tolerance))),
                'significant_violation_count': int(np.count_nonzero(x < -tolerance))}
    mi, monotone = record(tables[:, 1:]), record(increments)
    return {'tolerance_nats': tolerance, 'clipping': False,
            'tables': 'each numerical scramble and the canonical mean',
            'mutual_information': mi, 'source_inclusion_increment': monotone,
            'passed': mi['significant_violation_count'] == 0 and monotone['significant_violation_count'] == 0}


def shap_profile(model, evaluation, background, deadline):
    # The complete nine future phases are represented by their conditional
    # E(cos Y_j),E(sin Y_j), avoiding an arbitrary angle-chart discontinuity.
    resultant = i1e(model.kappa) / i0e(model.kappa)
    game = np.empty((len(evaluation), 512, 18))
    for coalition, bits in enumerate(MEMBERSHIP):
        check_time(deadline())
        x = np.where(bits[None, None], evaluation[:, None], background[None])
        mu = model.mean(x.reshape(-1, 9))
        prediction = np.concatenate((np.cos(mu) * resultant, np.sin(mu) * resultant), axis=1)
        game[:, coalition] = prediction.reshape(len(evaluation), len(background), 18).mean(1)
    values = np.zeros((len(evaluation), 9, 18))
    for i in range(9):
        absent = np.flatnonzero(~MEMBERSHIP[:, i])
        weights = np.array([1 / (9 * math.comb(8, int(m).bit_count())) for m in absent])
        values[:, i] = np.einsum('m,nmo->no', weights, game[:, absent | (1 << i)] - game[:, absent])
    error = float(abs(values.sum(1) - game[:, 511] + game[:, 0]).max())
    if error > 1e-9:
        raise ArithmeticError(f'Shapley efficiency failed: {error}')
    magnitude = np.hypot(values[:, :, :9], values[:, :, 9:])
    return magnitude.mean(0), {'efficiency_error_output_units': error,
        'coalitions': 512, 'players': 9, 'outputs': 'complete nine future phases in 18 circular moment coordinates',
        'profile': 'mean Euclidean magnitude of each circular-output attribution pair; nine output profiles',
        'evaluation_samples': len(evaluation), 'background_samples': len(background)}


def profile_tree(context, profile):
    distances = pdist(profile, metric='cosine')
    if not np.isfinite(distances).all():
        raise ArithmeticError('Undefined SHAP output-profile distance')
    clusters = linkage(distances, method='single')
    groups = {i: (i + 1,) for i in range(9)}; clades = []
    for i, row in enumerate(clusters):
        left, right = (groups[int(j)] for j in row[:2])
        group = tuple(sorted(left + right)); groups[9 + i] = group
        if len(group) < 9:
            clades.append(group)
    return context.scored((left, right), clades)


def run(case):
    context = SelectedCase(case)
    # The old six-hour ledger is immutable. A later explicitly approved
    # natural-stage ledger may extend only this newly requested stage.
    extra = context.folder.parent / 'natural_budget.json'
    if not extra.exists():
        raise ValueError('Start the approved shared natural-stage ledger after the original four methods end')
    authorized = json.loads(extra.read_text())
    if (authorized.get('cases') != ['A', 'C']
            or authorized['deadline_epoch_seconds'] - authorized['start_epoch_seconds'] != 4 * 3600
            or authorized.get('pilot_sha256') != context.budget['pilot_sha256']):
        raise ValueError('Natural-stage authorization/provenance mismatch')
    for name, expected in authorized['source_sha256'].items():
        if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Natural-stage implementation changed: {name}')
    endpoint = authorized['deadline_epoch_seconds']
    context.deadline = lambda: time.monotonic() + endpoint - time.time()
    folder = context.folder
    path = folder / 'natural_baselines.json'
    result = json.loads(path.read_text()) if path.exists() else {'version': VERSION, 'case': case, 'methods': {}}
    if result['version'] != VERSION or result['case'] != case:
        raise ValueError('Natural baseline version/case differs')
    if result.get('model_sha256', context.digest) != context.digest:
        raise ValueError('Natural baseline belongs to a different frozen transition model')
    result.update(status='running', stage='natural_sampling', model_sha256=context.digest)
    save_json(path, result)
    try:
        states, meta = natural_data(context)
        digest = hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()
        if result.get('natural_data_sha256', digest) != digest:
            raise ValueError('Natural baseline belongs to different natural trajectories')
        result.update(natural_sampling=meta, natural_data_sha256=digest)
        model, _, _ = context.load_model()
        source_path = folder / 'natural_source.json'
        if source_path.exists():
            fitted = json.loads(source_path.read_text())
            if fitted['natural_data_sha256'] != digest or fitted.get('version') != VERSION:
                raise ValueError('Natural density fitted to different trajectories')
            prior = RotationalSource(fitted['source']['centers'], fitted['source']['kappa'])
            source_audit = fitted['audit']
        else:
            prior, source_audit = fit_source(states)
            save_json(source_path, {'version': VERSION, 'natural_data_sha256': digest, 'source': prior.payload(), 'audit': source_audit})
        result['source_density_audit'] = source_audit
        result['stage'] = 'shap_profiles'; save_json(path, result)
        if 'shap_profiles' not in result['methods']:
            train = states[:256].reshape(-1, 9); hold = states[256:512].reshape(-1, 9)
            records = result.setdefault('shap_replicates', [])
            for r in range(4):
                if any(v['repeat'] == r for v in records):
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([20261015, 0, 91000 + r]))
                foreground = hold[rng.choice(len(hold), 256, replace=False)]
                background = train[rng.choice(len(train), 128, replace=False)]
                profile, audit = shap_profile(model, foreground, background, context.deadline)
                records.append({'repeat': r, 'profile': profile.tolist(), 'audit': audit})
                save_json(path, result)
                print(json.dumps({'case': case, 'stage': 'SHAP profiles', 'repeat': r}), flush=True)
            profiles = np.asarray([v['profile'] for v in records])
            result['methods']['shap_profiles'] = {'status': 'screen_passed',
                'metrics': profile_tree(context, profiles.mean(0)),
                'integration_replicates': [profile_tree(context, p) for p in profiles],
                'profiles': profiles.mean(0).tolist(), 'checks': [v['audit'] for v in records],
                'repeat_unit': 'natural foreground/background subsamples; one frozen fitted model',
                'tree_adaptation': 'mean per-output circular attribution vector magnitude; cosine/single linkage',
                'source_data_sha256': digest}
            save_json(path, result)
        result['stage'] = 'wms_natural'; save_json(path, result)
        if not source_audit['diagnostic_passed']:
            result['methods']['wms_natural'] = {'status': 'natural_source_diagnostics_failed', 'metrics': None}
        elif 'wms_natural' not in result['methods']:
            rows = result.setdefault('wms_replicates', [])
            for r in range(4):
                row = next((v for v in rows if v['scramble'] == r), None)
                if row is None:
                    row = {'scramble': r, 'information_nats': {}, 'diagnostics': {}}; rows.append(row)
                integral = NaturalInformation(context, prior, outer=1024, seed=20261015 + 1000 * r)
                for m in range(1, 512):
                    if str(m) in row['information_nats']:
                        continue
                    value, _, diagnostics = integral.information(m, 64)
                    row['information_nats'][str(m)] = value; row['diagnostics'][str(m)] = diagnostics
                    save_json(path, result)
                    if m % 64 == 0:
                        print(json.dumps({'case': case, 'stage': 'natural WMS', 'scramble': r, 'subsets': m}), flush=True)
            values = np.array([[r['information_nats'][str(m)] for m in range(1, 512)] for r in rows])
            maximum_sem = float((values.std(0, ddof=1) / 2).max())
            ds = [d for row in rows for pair in row['diagnostics'].values() for d in pair if d['ess_minimum'] is not None]
            minimum_ess = min(d['ess_minimum'] for d in ds); maximum_weight = max(d['maximum_weight'] for d in ds)
            changes = []
            for r in range(4):
                integral = NaturalInformation(context, prior, outer=1024, seed=20261015 + 1000 * r)
                for m in (1, 3, 7, 24, 31, 96, 384, 480, 511):
                    value, _, _ = integral.information(m, 256)
                    changes.append(abs(value - rows[r]['information_nats'][str(m)]))
            issues = []
            mi_audit = information_audit(values)
            if not mi_audit['passed']:
                issues.append('MI nonnegativity or source-inclusion monotonicity failed')
            if maximum_sem >= .01:
                issues.append('information numerical SEM >= .01 nats')
            if max(changes) >= .01:
                issues.append('inner 64-to-256 change >= .01 nats')
            if minimum_ess < 10 or maximum_weight >= .25:
                issues.append('importance weights failed')
            metrics, audit = fixed_target_wms(context, {m: float(values[:, m-1].mean()) for m in range(1, 512)})
            record = {'status': 'numerically_unresolved' if issues else 'screen_passed',
                'metrics': None if issues else metrics, 'failed_checks': issues,
                'integration_replicates': [fixed_target_wms(context, {m: row['information_nats'][str(m)] for m in range(1, 512)})[0] for row in rows],
                'audit': audit, 'information_audit': mi_audit, 'maximum_information_sem_nats': maximum_sem,
                'maximum_inner_change_nats': max(changes), 'minimum_ess': minimum_ess,
                'maximum_weight': maximum_weight, 'source_data_sha256': digest,
                'target': 'fixed complete nine future phases', 'transition': 'same frozen nonlinear conditional TM',
                'source_approximation': 'normalized rotational von Mises mixture fitted to shared natural trajectories'}
            if issues:
                record['diagnostic_metrics'] = metrics
            result['methods']['wms_natural'] = record
        result['status'] = 'complete'
    except (TimeoutError, ArithmeticError, ValueError, np.linalg.LinAlgError) as e:
        result.update(status='budget_exhausted' if isinstance(e, TimeoutError) else 'numerical_failure', error=str(e))
    save_json(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'C'), required=True)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        output = run(args.case)
    print(json.dumps({'case': args.case, 'status': output['status']}))
