"""Boundary-aware triangular proposals for the existing full Gaussian TM channel.

Only numerical integration changes. Each proposal is a normalized sequentially
truncated Gaussian transport on the original source box, with its exact density.
The full learned channel, common factorized prior, future target, and EI definition
are unchanged. No decomposition into planted components or Syn projection.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri, ndtri_exp
from scipy.stats import qmc

from scripts.kuramoto_six_topology_core import JointIntegrator, MLPChannel, N, check, universal_frequencies


def zero_mean_projection_controls(model, x, z):
    """Conditional-zero-mean Gaussian quadratic controls, coefficient fixed at 1.

    With whitened missing-source Jacobian B(x), P=I-QQ^T is a projector.
    Since z is independent N(0,I), E[(z^T P(x) z-tr(P(x)))/2 | x]=0.
    Thus this changes finite-sample variance, never the EI population target.
    """
    _, jacobian = model.mean(x, True)
    whitened = np.linalg.solve(model.chol[None], jacobian)
    controls = []
    for mask in range(1, 64):
        missing = [i for i in range(6) if not mask >> i & 1]
        if missing:
            q, _ = np.linalg.qr(whitened[:, :, missing])
            residual = z-np.einsum('nik,nk->ni', q, np.einsum('nik,ni->nk', q, z))
        else:
            residual = z
        controls.append(.5*(np.sum(residual*residual, axis=1)-(6-len(missing))))
    return np.array(controls)


class DoublePrecisionMLPChannel(MLPChannel):
    """Same frozen model, vectorized CPU float64 inference; original Jacobian."""
    def __init__(self, payload):
        super().__init__(payload)
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.tw = [torch.tensor(w, dtype=torch.float64) for w in self.weights]
        self.tb = [torch.tensor(b, dtype=torch.float64) for b in self.biases]
        self.tc = {name: torch.tensor(getattr(self, name), dtype=torch.float64)
                   for name in ('feature_mean', 'feature_scale', 'output_mean', 'output_scale')}
        self.tf = torch.tensor(universal_frequencies(), dtype=torch.float64)

    def mean(self, x, jacobian=False):
        torch = self.torch
        with torch.inference_mode():
            value = torch.as_tensor(x, dtype=torch.float64)
            columns = [a for h in (1, 2, 3) for a in (torch.sin(h*value), torch.cos(h*value))]
            if jacobian:
                derivatives = [torch.diag_embed(a) for harmonic in (1, 2, 3)
                               for a in (harmonic*torch.cos(harmonic*value), -harmonic*torch.sin(harmonic*value))]
            if self.payload.get('feature_kind', 'node') == 'universal_fourier':
                phase = value @ self.tf.T
                columns.extend((torch.sin(phase), torch.cos(phase)))
                if jacobian:
                    derivatives.extend((torch.cos(phase)[:, :, None]*self.tf[None],
                                        -torch.sin(phase)[:, :, None]*self.tf[None]))
            h = (torch.cat(columns, 1)-self.tc['feature_mean'])/self.tc['feature_scale']
            if jacobian:
                j = (torch.cat(derivatives, 1)/self.tc['feature_scale'][None, :, None]).transpose(1, 2)
            for i, (w, b) in enumerate(zip(self.tw, self.tb)):
                h = h @ w + b
                if jacobian:
                    j = j @ w
                if i < 2:
                    if jacobian:
                        sigmoid = torch.sigmoid(h)
                        j = j*(sigmoid+h*sigmoid*(1-sigmoid))[:, None, :]
                    h = torch.nn.functional.silu(h)
            result = (value+h*self.tc['output_scale']+self.tc['output_mean']).numpy()
            if jacobian:
                return result, (torch.eye(6)[None]+j.transpose(1, 2)*self.tc['output_scale'][None, :, None]).numpy()
            return result


class MPSDensityMLPChannel(DoublePrecisionMLPChannel):
    """GPU forward pass only; Gaussian log density and all proposals stay float64.

    Used only after whole-integral comparison against the CPU float64 channel.
    The outer joint draws and Jacobians deliberately keep the CPU implementation.
    """
    def __init__(self, payload):
        super().__init__(payload)
        torch = self.torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS device unavailable')
        self.gw = [w.to(device='mps', dtype=torch.float32) for w in self.tw]
        self.gb = [b.to(device='mps', dtype=torch.float32) for b in self.tb]
        self.gc = {key: value.to(device='mps', dtype=torch.float32) for key, value in self.tc.items()}
        self.gf = self.tf.to(device='mps', dtype=torch.float32)

    def gpu_mean(self, x):
        torch = self.torch
        with torch.inference_mode():
            value = torch.as_tensor(x, device='mps', dtype=torch.float32)
            columns = [a for h in (1, 2, 3) for a in (torch.sin(h*value), torch.cos(h*value))]
            if self.payload.get('feature_kind', 'node') == 'universal_fourier':
                phase = value @ self.gf.T
                columns.extend((torch.sin(phase), torch.cos(phase)))
            h = (torch.cat(columns, 1)-self.gc['feature_mean'])/self.gc['feature_scale']
            for i, (w, b) in enumerate(zip(self.gw, self.gb)):
                h = h @ w + b
                if i < 2:
                    h = torch.nn.functional.silu(h)
            return (value+h*self.gc['output_scale']+self.gc['output_mean']).cpu().numpy().astype(float)

    def log_prob(self, x, y):
        residual = y-self.gpu_mean(x)
        return self.lognorm-.5*np.einsum('ni,ij,nj->n', residual, self.precision, residual, optimize=True)


def log_interval_mass(a, b):
    """Stable log(Phi(b)-Phi(a)), including intervals in either tail."""
    from scipy.special import ndtr
    result = np.empty_like(a)
    middle = (a <= 0) & (b >= 0)
    result[middle] = np.log(ndtr(b[middle])-ndtr(a[middle]))
    tail = ~middle
    right = a[tail] > 0
    big = np.where(right, log_ndtr(-a[tail]), log_ndtr(b[tail]))
    small = np.where(right, log_ndtr(-b[tail]), log_ndtr(a[tail]))
    result[tail] = big + np.log(-np.expm1(small-big))
    return result


def bounded_sample(u, center, chol, lower=-np.pi, upper=np.pi):
    batch, particles, dimensions = u.shape
    z = np.zeros_like(u)
    value = np.empty_like(u)
    logdensity = np.zeros((batch, particles))
    u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
    for j in range(dimensions):
        mu = center[:, None, j] + np.einsum('bpk,bk->bp', z[:, :, :j], chol[:, j, :j])
        sd = chol[:, j, j, None]
        a, b = (lower - mu) / sd, (upper - mu) / sd
        # Direct CDF arithmetic is stable when the interval crosses zero;
        # only one-sided tails need the more expensive log-CDF inverse.
        from scipy.special import ndtr
        middle = (a <= 0) & (b >= 0)
        lo, hi = ndtr(a[middle]), ndtr(b[middle])
        z[:, :, j][middle] = ndtri(lo+u[:, :, j][middle]*(hi-lo))
        tail = ~middle
        at, bt, ut = a[tail], b[tail], u[:, :, j][tail]
        logp = np.logaddexp(np.log1p(-ut)+log_ndtr(at), np.log(ut)+log_ndtr(bt))
        logs = np.logaddexp(np.log1p(-ut)+log_ndtr(-at), np.log(ut)+log_ndtr(-bt))
        z[:, :, j][tail] = np.where(at > 0, -ndtri_exp(logs), ndtri_exp(logp))
        value[:, :, j] = mu + sd * z[:, :, j]
        logdensity += -.5 * z[:, :, j]**2 - .5 * np.log(2*np.pi) - np.log(sd) - log_interval_mass(a, b)
    if not np.isfinite(value).all() or np.any(value < lower - 1e-10) or np.any(value > upper + 1e-10):
        raise ArithmeticError("Invalid bounded transport draw; no sample projection")
    return value, logdensity


def bounded_logpdf(value, center, chol, lower=-np.pi, upper=np.pi):
    # Forward substitution reuses each triangular factor for every particle;
    # broadcasting linalg.solve would refactor the same matrix P times.
    z = np.zeros_like(value)
    result = np.zeros(value.shape[:2])
    for j in range(value.shape[-1]):
        sd = chol[:, j, j, None]
        mu = center[:, None, j] + np.einsum('bpk,bk->bp', z[:, :, :j], chol[:, j, :j])
        z[:, :, j] = (value[:, :, j] - mu) / sd
        a, b = (lower - mu) / sd, (upper - mu) / sd
        result += -.5*z[:, :, j]**2 - .5*np.log(2*np.pi) - np.log(sd) - log_interval_mass(a, b)
    return np.where(np.all((value >= lower) & (value <= upper), axis=-1), result, -np.inf)


class BoundedJointIntegrator(JointIntegrator):
    """Same EI oracle, bounded transport mixtures, streamed Sobol completions."""
    def __init__(self, model, outer, particles, seed, deadline, prior='uniform'):
        self.model, self.outer, self.particles = model, outer, particles
        self.seed, self.deadline, self.prior = seed, deadline, prior
        for value in (outer, particles):
            if value < 1 or value & (value - 1):
                raise ValueError("Sobol sizes must be powers of two")
        if prior != 'uniform':
            raise ValueError("This boundary-aware proposal is for the registered uniform box")
        u = qmc.Sobol(12, scramble=True, seed=seed).random_base2(int(np.log2(outer)))
        self.x = (u[:, :N] - .5) * 2*np.pi
        self.y = model.sample(self.x, ndtri(u[:, N:]))
        self.marginals, self.diagnostics = {}, {}

    def marginal(self, mask):
        check(self.deadline)
        if mask in self.marginals:
            return self.marginals[mask]
        if mask == 63:
            value = self.model.log_prob(self.x, self.y)
            self.marginals[mask] = value
            return value
        started = time.time()
        missing = np.array([i for i in range(N) if not mask >> i & 1])
        d = len(missing)
        center, chol, remaining = self.proposal(missing)
        # Restart for every marginal: common random numbers, no giant N*P array.
        sequence = qmc.Sobol(7, scramble=True, seed=self.seed + 1)
        values, all_ess, all_maximum = [], [], []
        chunk = max(1, 32768 // self.particles)
        for start in range(0, self.outer, chunk):
            check(self.deadline)
            stop = min(self.outer, start + chunk)
            u = sequence.random((stop - start) * self.particles).reshape(stop-start, self.particles, 7)
            c, l = center[start:stop], chol[start:stop]
            normal, _ = bounded_sample(u[:, :, 1:d+1], c, l)
            wide, _ = bounded_sample(u[:, :, 1:d+1], c, l * 2)
            prior = (u[:, :, 1:d+1] - .5) * 2*np.pi
            proposed = np.where((u[:, :, 0] < .05)[:, :, None], prior,
                                np.where((u[:, :, 0] < .35)[:, :, None], wide, normal))
            logprior = -d * np.log(2*np.pi)
            logproposal = logsumexp(np.stack((
                np.log(.65) + bounded_logpdf(proposed, c, l),
                np.log(.30) + bounded_logpdf(proposed, c, l*2),
                np.full((stop-start, self.particles), np.log(.05) + logprior))), axis=0)
            full = np.repeat(self.x[start:stop, None, :], self.particles, axis=1)
            full[:, :, missing] = proposed
            target = np.repeat(self.y[start:stop], self.particles, axis=0)
            logweights = self.model.log_prob(full.reshape(-1, N), target).reshape(stop-start, self.particles)
            logweights += logprior - logproposal
            normalizer = logsumexp(logweights, axis=1)
            normalized = logweights - normalizer[:, None]
            values.extend(normalizer - np.log(self.particles))
            all_ess.extend(np.exp(-logsumexp(2*normalized, axis=1)))
            all_maximum.extend(np.exp(normalized.max(axis=1)))
        value = np.asarray(values)
        if not np.isfinite(value).all():
            raise ArithmeticError(f"Non-finite bounded integral, mask={mask}")
        self.marginals[mask] = value
        self.diagnostics[str(mask)] = dict(ess_min=float(np.min(all_ess)),
            ess_median=float(np.median(all_ess)), maximum_weight=float(np.max(all_maximum)),
            remaining_mode_step=remaining, seconds=time.time()-started,
            proposal='65/30/5 bounded triangular Gaussian, 2x-wide bounded Gaussian, uniform')
        return value
