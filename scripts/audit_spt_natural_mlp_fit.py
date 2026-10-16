"""Small, truth-only held-out audit; no training, model selection, or EI/SPT.

Repeated simulator transitions estimate a noisy conditional-mean reference.
Known interaction supports are used ONLY to measure excitation after fitting.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spt_kuramoto_natural_mlp import NaturalMLPConditionalTM
from scripts.spt_real_natural_mlp import NaturalRealMLPConditionalTM
from scripts.spt_kuramoto_simplified_ac import SimplifiedKuramoto
from scripts.spt_glv_simplified_a_pilot import MatchedSimplifiedGLV
from scripts.spt_local_hyperedge_kuramoto import wrap, edge_force as circular_force
from scripts.spt_local_hyperedge_glv import edge_force as real_force
from scripts.spt_kuramoto_finite_tm import save_json, digest

FOLDER = ROOT / 'results/spt_natural_mlp_three_cases'
OUTPUT = ROOT / 'results/spt_natural_mlp_fit_audit/diagnostics.json'


def effective_rank(x):
    covariance = np.cov(x, rowvar=False)
    return float(np.trace(covariance) ** 2 / np.sum(covariance ** 2))


def reference_error(generator, model, x, rng, circular, repeats=32):
    y = generator.step(np.repeat(x, repeats, axis=0), rng, dt=.0025).reshape(len(x), repeats, 9)
    if circular:
        reference = np.angle(np.mean(np.exp(1j * y), axis=1))
        residual = wrap(y - reference[:, None, :])
        error = wrap(reference - model.mean(x))
        base_error = wrap(reference - x)
    else:
        reference = y.mean(1)
        residual = y - reference[:, None, :]
        error = reference - model.mean(x)
        base_error = reference - x
    mc_variance = float(np.mean(residual ** 2) / (repeats - 1))
    return dict(initial_states=len(x), independent_repeats=repeats,
        source_sha256=digest(x), mlp_to_mc_mean_rmse=float(np.sqrt(np.mean(error ** 2))),
        persistence_to_mc_mean_rmse=float(np.sqrt(np.mean(base_error ** 2))),
        mc_mean_noise_rms_approx=float(np.sqrt(mc_variance)),
        mean_mse_minus_mc_variance=float(np.mean(error ** 2) - mc_variance),
        diagnostic_only=True, used_for_training_or_selection=False,
        circular_mean_variance_is_local_small_noise_approximation=circular)


def main():
    started, deadline = time.monotonic(), time.monotonic() + 60
    data = json.loads((FOLDER / 'summary.json').read_text())
    out = dict(scope='existing frozen MLPs; held-out truth diagnostics only; no new fit/EI/F1',
               seed=20261017, records=[])
    rng = np.random.default_rng(20261017)
    for record in data['records']:
        if time.monotonic() >= deadline:
            out['status'] = 'audit_time_limit'
            break
        circular = record['family'] == 'circular'
        cls = NaturalMLPConditionalTM if circular else NaturalRealMLPConditionalTM
        model = cls.load(json.loads((FOLDER / record['name'] / 'model.json').read_text()))
        metadata = record['natural_data']
        with np.load(ROOT / metadata['pool_path'], allow_pickle=False) as z:
            states = z['states'].copy()
        chain_ids = metadata['splits']['test']['chains']
        natural = states[:-1, chain_ids].reshape(-1, 9)
        q = rng.uniform(-np.pi, np.pi, (4096, 9))
        if circular:
            generator = SimplifiedKuramoto(record['native_reference'])
            features = np.concatenate((np.sin(natural), np.cos(natural)), axis=1)
            qfeatures = np.concatenate((np.sin(q), np.cos(q)), axis=1)
            force = circular_force
        else:
            weights = {tuple(row['edge']): row['weight'] for row in json.loads((ROOT / 'results/spt_local_kuramoto_tm/ac_comparison/A/frozen_model.json').read_text())['network']}
            generator = MatchedSimplifiedGLV(weights)
            features, qfeatures, force = natural, q, real_force
        excitation = []
        for edge in generator.supports:
            indices = np.asarray(edge) - 1
            a, b = force(natural, edge)[:, indices], force(q, edge)[:, indices]
            natural_rms, intervention_rms = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
            natural_variation = float(np.sqrt(np.mean((a - a.mean(0)) ** 2)))
            intervention_variation = float(np.sqrt(np.mean((b - b.mean(0)) ** 2)))
            excitation.append(dict(edge=list(edge), natural_force_rms=natural_rms,
                intervention_force_rms=intervention_rms, ratio=natural_rms / intervention_rms,
                natural_centered_force_rms=natural_variation,
                intervention_centered_force_rms=intervention_variation,
                centered_excitation_ratio=natural_variation / intervention_variation))
        x = natural[rng.choice(len(natural), 128, replace=False)]
        ix = q[:128]
        result = dict(case=record['case'], model_sha256=model.sha256(),
            feature_effective_rank_natural=effective_rank(features),
            feature_effective_rank_intervention=effective_rank(qfeatures),
            median_source_sd_natural=float(np.median(natural.std(0))),
            median_source_sd_intervention=float(np.median(q.std(0))),
            true_edge_excitation_diagnostic=excitation,
            natural_reference=reference_error(generator, model, x, rng, circular),
            intervention_reference=reference_error(generator, model, ix, rng, circular))
        out['records'].append(result)
        out['elapsed_seconds'] = time.monotonic() - started
        save_json(OUTPUT, out)
        print(json.dumps(dict(case=result['case'], natural_mc_rmse=result['natural_reference']['mlp_to_mc_mean_rmse'],
            intervention_mc_rmse=result['intervention_reference']['mlp_to_mc_mean_rmse'],
            natural_effective_rank=result['feature_effective_rank_natural'],
            intervention_effective_rank=result['feature_effective_rank_intervention'],
            excitation_ratio=[min(e['ratio'] for e in excitation), max(e['ratio'] for e in excitation)],
            elapsed_seconds=out['elapsed_seconds'])), flush=True)
    out.setdefault('status', 'complete')
    save_json(OUTPUT, out)


if __name__ == '__main__':
    main()
