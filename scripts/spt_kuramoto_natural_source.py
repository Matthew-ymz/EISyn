"""Normalized rotational von-Mises mixture for a correlated natural phase prior.

This is an explicit source-density approximation, not a new transition model.
Integrating the latent common rotation analytically gives every source marginal.
The same natural trajectories supply its centers and the SHAP background.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e, i1e, logsumexp

from scripts.spt_kuramoto_finite_tm import LOG_TWO_PI, PeriodicConditionalTM
from scripts.spt_local_hyperedge_kuramoto import wrap


def log_i0(x):
    return np.log(i0e(x)) + abs(x)


class RotationalSource:
    def __init__(self, centers, kappa):
        self.centers = np.asarray(centers, dtype=float)
        self.kappa = float(kappa)
        if self.centers.ndim != 2 or self.centers.shape[1] != 9 or not len(self.centers):
            raise ValueError('Natural source centers must have shape (K,9)')
        if not np.isfinite(self.centers).all() or not np.isfinite(kappa) or kappa <= 0:
            raise ValueError('Finite circular centers and positive concentration required')
        self.center_phase = np.exp(-1j * self.centers)
        self.precision = self.kappa * (np.eye(9) - np.ones((9, 9)) / 9)
        self.noise = PeriodicConditionalTM([np.empty((0, 9)) for _ in range(9)],
            [np.empty(0) for _ in range(9)], [np.empty(0) for _ in range(9)],
            np.zeros(9), np.full(9, self.kappa))

    def log_prob(self, x, mask=511):
        x = np.asarray(x)
        indices = np.array([i for i in range(9) if mask >> i & 1], dtype=int)
        if len(indices) == 0:
            return np.zeros(len(x))
        # Integral over a shared uniform rotation: I0(|sum_j k exp(i delta_j)|).
        output = np.empty(len(x))
        for start in range(0, len(x), 4096):
            z = np.exp(1j * x[start:start + 4096, indices])
            resultant = abs(self.kappa * z @ self.center_phase[:, indices].T)
            output[start:start + 4096] = (logsumexp(log_i0(resultant), axis=1)
                - np.log(len(self.centers)) - len(indices) * (LOG_TWO_PI + log_i0(self.kappa)))
        return output

    def sample(self, uniforms):
        # 1 mixture coordinate, 1 uniform rotation, and 9 independent VM noises.
        u = np.asarray(uniforms)
        if u.shape[1] != 11 or np.any((u <= 0) | (u >= 1)):
            raise ValueError('Open-interval (N,11) source uniforms required')
        component = np.floor(u[:, 0] * len(self.centers)).astype(int)
        location = self.centers[component] + (u[:, 1, None] - .5) * 2 * np.pi
        return self.noise.sample(location, u[:, 2:])

    def local_component(self, x):
        # Used only to construct a proposal. The importance weight uses the
        # full mixture density, including all components and rotations.
        z = np.exp(1j * x) @ self.center_phase.T
        component = np.argmax(abs(z), axis=1)
        rotation = np.angle(z[np.arange(len(x)), component])
        return wrap(self.centers[component] + rotation[:, None])

    def payload(self):
        return {'family': 'rotational_von_mises_mixture', 'centers': self.centers.tolist(),
                'kappa': self.kappa, 'common_rotation': 'analytically integrated uniform(-pi,pi)',
                'marginals': 'exact for the fitted source mixture'}


def fit_source(states):
    """Fixed, data-only size/bandwidth selection on held-out natural states."""
    train = states[:256]
    hold = states[256:512].reshape(-1, 9)[::4]
    candidates = []
    for per_chain in (1, 2, 4):
        times = np.linspace(32, 223, per_chain).round().astype(int)
        centers = train[times].reshape(-1, 9)
        for kappa in (25., 100., 400., 1600.):
            source = RotationalSource(centers, kappa)
            nll = float(-source.log_prob(hold).mean())
            candidates.append((nll, per_chain, kappa, centers))
    nll, count, kappa, centers = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    selected = RotationalSource(centers, kappa)
    first, second = states[:256].reshape(-1, 9), states[256:512].reshape(-1, 9)
    p1 = np.mean(np.exp(1j * (first[:, :, None] - first[:, None, :])), axis=0)
    p2 = np.mean(np.exp(1j * (second[:, :, None] - second[:, None, :])), axis=0)
    predicted = ((i1e(kappa) / i0e(kappa)) ** 2
                 * np.mean(np.exp(1j * (centers[:, :, None] - centers[:, None, :])), axis=0))
    np.fill_diagonal(predicted, 1.)
    moment_error = float(abs(predicted - p2).max())
    audit = {'source_density_approximation': 'rotational von Mises mixture; not a Gaussian phase chart',
        'heldout_nll_nats': nll, 'components': len(centers), 'centers_per_chain': count,
        'kappa': kappa, 'maximum_pair_phase_moment_shift': float(abs(p1 - p2).max()),
        'maximum_heldout_pair_phase_moment_error': moment_error,
        'diagnostic_passed': bool(abs(p1 - p2).max() < .25 and moment_error < .25),
        'diagnostic_scope': 'stationarity and pair circular moments; not a proof of full density accuracy',
        'bandwidth_selection': 'minimum heldout negative log likelihood; no F1 input',
        'candidates': [{'heldout_nll_nats': v, 'centers_per_chain': c, 'kappa': k} for v, c, k, _ in candidates]}
    return selected, audit
