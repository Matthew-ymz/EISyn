"""Data-only nonlinear conditional triangular TM for the finite-data SPT pilot.

Real outputs: learned dense affine preconditioning followed by autoregressive
rational-quadratic splines with linear tails. Circular outputs: bounded splines
to a uniform reference, tied endpoint derivatives, periodic context and history.
Neither the model nor its marginal integrator accepts simulator parameters.
"""
from __future__ import annotations

import math
import copy
import time

import numpy as np
import torch
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc
from torch import nn
from torch.nn import functional as F
from nflows.transforms.splines.rational_quadratic import (
    rational_quadratic_spline, unconstrained_rational_quadratic_spline,
)

TWO_PI = 2 * np.pi
LOG_TWO_PI = float(np.log(TWO_PI))
BINS = 8
HIDDEN = 128
SYN_TOLERANCE = .03


def tensor(array):
    return torch.tensor(np.asarray(array), dtype=torch.float32)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Approved computation budget reached')


class MaskedLinear(nn.Linear):
    def __init__(self, inputs, outputs, mask):
        super().__init__(inputs, outputs)
        self.register_buffer('mask', torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class SplineConditioner(nn.Module):
    """Unrestricted source context plus strictly earlier-output history."""
    def __init__(self, dimension, domain, hidden=HIDDEN, bins=BINS):
        super().__init__()
        self.dimension = dimension
        self.domain = domain
        self.multiplier = 3 * bins if domain == 'phase' else 3 * bins - 1
        context_width = dimension * (4 if domain == 'phase' else 1)
        history_width = dimension * (2 if domain == 'phase' else 1)
        self.context = nn.Sequential(
            nn.Linear(context_width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dimension * self.multiplier),
        )
        input_degrees = np.tile(np.arange(1, dimension + 1),
                                2 if domain == 'phase' else 1)
        hidden_degrees = np.arange(hidden) % max(dimension, 1)
        output_degrees = np.repeat(np.arange(1, dimension + 1), self.multiplier)
        self.history = nn.Sequential(
            MaskedLinear(history_width, hidden,
                         hidden_degrees[:, None] >= input_degrees[None, :]),
            nn.SiLU(),
            MaskedLinear(hidden, hidden,
                         hidden_degrees[:, None] >= hidden_degrees[None, :]),
            nn.SiLU(),
            MaskedLinear(hidden, dimension * self.multiplier,
                         output_degrees[:, None] > hidden_degrees[None, :]),
        )
        # Identity is an explicit validation candidate, not an implicit fallback.
        for head in (self.context[-1], self.history[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, source, previous):
        if self.domain == 'phase':
            context = torch.cat((source.sin(), source.cos(),
                                 (2 * source).sin(), (2 * source).cos()), dim=-1)
            angle = previous * TWO_PI
            history = torch.cat((angle.sin(), angle.cos()), dim=-1)
        else:
            context, history = source, previous
        return (self.context(context) + self.history(history)).reshape(
            -1, self.dimension, self.multiplier)


class ConditionalSplineTM(nn.Module):
    def __init__(self, dimension, domain, *, coefficient=None, intercept=None,
                 cholesky=None, hidden=HIDDEN, bins=BINS):
        super().__init__()
        if domain not in ('real', 'phase'):
            raise ValueError('Unknown coordinate domain')
        self.dimension, self.domain, self.bins = dimension, domain, bins
        self.conditioner = SplineConditioner(dimension, domain, hidden, bins)
        self.register_buffer('coefficient', tensor(np.eye(dimension) * 0
                             if coefficient is None else coefficient))
        self.register_buffer('intercept', tensor(np.zeros(dimension)
                             if intercept is None else intercept))
        self.register_buffer('cholesky', tensor(np.eye(dimension)
                             if cholesky is None else cholesky))
        self.register_buffer('inverse_cholesky', torch.linalg.inv(self.cholesky))
        self.register_buffer('logdet', self.cholesky.diag().log().sum())
        self.derivative_identity = math.log(math.expm1(1 - .001))

    def prepare(self, source, target):
        if self.domain == 'phase':
            if torch.any((target < 0) | (target > TWO_PI)):
                raise ValueError('Circular observation outside registered support')
            return target / TWO_PI
        return (target - source @ self.coefficient.T - self.intercept) @ self.inverse_cholesky.T

    def spline(self, values, parameters, inverse=False):
        k = self.bins
        widths, heights = parameters[..., :k], parameters[..., k:2*k]
        derivative = parameters[..., 2*k:] + self.derivative_identity
        if self.domain == 'phase':
            # Exact endpoint derivative tie makes the conditional density periodic.
            derivative = torch.cat((derivative, derivative[..., :1]), dim=-1)
            return rational_quadratic_spline(values, widths, heights, derivative,
                                             inverse=inverse)
        return unconstrained_rational_quadratic_spline(
            values, widths, heights, derivative, inverse=inverse,
            tails='linear', tail_bound=4.)

    def log_prob_components(self, source, target):
        prepared = self.prepare(source, target)
        reference, logjac = self.spline(
            prepared, self.conditioner(source, prepared))
        if self.domain == 'phase':
            log_prob = logjac.sum(-1) - self.dimension * LOG_TWO_PI
            return log_prob, torch.full_like(log_prob, -self.dimension * LOG_TWO_PI)
        normal = -.5 * (reference.square() + LOG_TWO_PI).sum(-1)
        base = -.5 * (prepared.square() + LOG_TWO_PI).sum(-1) - self.logdet
        return normal + logjac.sum(-1) - self.logdet, base

    def log_prob(self, source, target):
        return self.log_prob_components(source, target)[0]

    @torch.no_grad()
    def sample(self, source, reference):
        generated = torch.zeros_like(reference)
        for j in range(self.dimension):
            parameters = self.conditioner(source, generated)[:, j]
            value, _ = self.spline(reference[:, j], parameters, inverse=True)
            generated[:, j] = value
        if self.domain == 'phase':
            return generated * TWO_PI
        return (generated @ self.cholesky.T + source @ self.coefficient.T
                + self.intercept)


@torch.no_grad()
def batched_log_prob(model, source, target, batch_size=512):
    values = []
    for start in range(0, len(source), batch_size):
        values.append(model.log_prob(source[start:start+batch_size],
                                     target[start:start+batch_size]).double().numpy())
    return np.concatenate(values)


def fit_spline(source, target, validation_source, validation_target, *, domain,
               seed, epochs, deadline, checkpoint_callback=None):
    """Only observations/domain enter this fitter; no support/oracle arguments."""
    from scripts.spt_finite_data_tm import FittedConditionalTM
    torch.manual_seed(seed)
    n, _, width = source.shape
    if domain == 'real':
        affine = FittedConditionalTM.fit(source, target, domain=domain)
        kwargs = dict(coefficient=affine.coefficient, intercept=affine.intercept,
                      cholesky=affine.noise_cholesky)
    else:
        kwargs = {}
    model = ConditionalSplineTM(9 * width, domain, **kwargs)
    x, y = tensor(source.reshape(n, -1)), tensor(target.reshape(n, -1))
    vx, vy = tensor(validation_source.reshape(len(validation_source), -1)), tensor(
        validation_target.reshape(len(validation_target), -1))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    initial = float(-batched_log_prob(model, vx, vy).mean())
    best, best_epoch = initial, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = [{'epoch': 0, 'validation_nll_nats': initial}]
    rng = np.random.default_rng(seed + 1901)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        loss_sum = 0.
        for start in range(0, n, 512):
            check_deadline(deadline)
            idx = torch.as_tensor(order[start:start+512])
            loss = -model.log_prob(x[idx], y[idx]).mean()
            if not torch.isfinite(loss):
                raise ArithmeticError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        if epoch % 5 and epoch != epochs:
            continue
        model.eval()
        validation = float(-batched_log_prob(model, vx, vy).mean())
        if not np.isfinite(validation):
            raise ArithmeticError('Nonfinite validation loss')
        if validation < best - 1e-4:
            best, best_epoch = validation, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = {'epoch': epoch, 'train_nll_nats': loss_sum / n,
               'validation_nll_nats': validation,
               'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        print(row, flush=True)
        if checkpoint_callback is not None:
            checkpoint_callback(row)
        if epoch - best_epoch >= 15:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'initial_validation_nll_nats': initial,
                   'best_validation_nll_nats': best, 'best_epoch': best_epoch,
                   'trained_epochs': epoch, 'history': history,
                   'train_seconds': time.monotonic() - started,
                   'selected_identity_spline': best_epoch == 0,
                   'parameters': sum(p.numel() for p in model.parameters())}


class MarginalIntegrator:
    """RQMC of one fitted joint, sharing outer samples/inner prefixes per seed.

Real-domain Gaussian conditioning is a learned importance proposal only. Exact
likelihood-ratio weights correct it to the nonlinear spline model. The registered
Gaussian intervention is unchanged. Circular-domain integration uses that
registered uniform prior directly.
"""
    def __init__(self, model, *, node_width, outer, max_particles, seed, deadline,
                 density_device='cpu'):
        self.model = model
        self.width, self.outer = node_width, outer
        self.max_particles, self.deadline = max_particles, deadline
        self.dimension = model.dimension
        self.cache = {}
        self.density_device = density_device if model.domain == 'phase' else 'cpu'
        self.density_model = (model if self.density_device == 'cpu' else
                              copy.deepcopy(model).to(self.density_device).eval())
        self.batch_rows = 2048 if self.density_device == 'cpu' else 8192
        self.identity_spline = all(torch.count_nonzero(parameter).item() == 0
            for head in (model.conditioner.context[-1], model.conditioner.history[-1])
            for parameter in (head.weight, head.bias))
        d = self.dimension
        point = qmc.Sobol(2*d, scramble=True, seed=seed).random_base2(int(np.log2(outer)))
        if np.any((point <= 0) | (point >= 1)):
            raise ArithmeticError('RQMC reached inverse-CDF boundary')
        source = point[:, :d] * TWO_PI if model.domain == 'phase' else ndtri(point[:, :d])
        reference = point[:, d:] if model.domain == 'phase' else ndtri(point[:, d:])
        self.source = source.astype(np.float32)
        check_deadline(deadline)
        self.target = model.sample(tensor(source), tensor(reference)).numpy()
        if not np.isfinite(self.target).all():
            raise ArithmeticError('Nonfinite model-generated target')
        uniform = qmc.Sobol(d, scramble=True, seed=seed+1).random_base2(
            int(np.log2(outer*max_particles)))
        if np.any((uniform <= 0) | (uniform >= 1)):
            raise ArithmeticError('RQMC reached inverse-CDF boundary')
        values = uniform * TWO_PI if model.domain == 'phase' else ndtri(uniform)
        self.inner = values.astype(np.float32).reshape(outer, max_particles, d)
        if model.domain == 'real':
            self.coefficient = model.coefficient.double().numpy()
            self.intercept = model.intercept.double().numpy()
            self.cholesky = model.cholesky.double().numpy()
            self.whitened_coefficient = solve_triangular(
                self.cholesky, self.coefficient, lower=True)

    def columns(self, mask):
        return np.array([j for i in range(9) if mask >> i & 1
                         for j in range(i*self.width, (i+1)*self.width)], dtype=int)

    def gaussian_proposal(self, mask):
        missing = self.columns(511 ^ mask)
        kept = self.columns(mask)
        f = self.coefficient[:, missing]
        b = self.whitened_coefficient[:, missing]
        precision = np.eye(len(missing)) + b.T @ b
        factor = np.linalg.cholesky(precision)
        covariance = cho_solve((factor, True), np.eye(len(missing)))
        posterior_cholesky = np.linalg.cholesky(covariance)
        residual = (self.target - self.intercept
                    - self.source[:, kept] @ self.coefficient[:, kept].T)
        whitened = solve_triangular(self.cholesky, residual.T, lower=True).T
        mean = whitened @ b @ covariance
        marginal_cholesky = np.linalg.cholesky(self.cholesky @ self.cholesky.T + f @ f.T)
        z = solve_triangular(marginal_cholesky, residual.T, lower=True).T
        base_log_prob = (-.5 * (z*z + LOG_TWO_PI).sum(1)
                         - np.log(np.diag(marginal_cholesky)).sum())
        return missing, mean, posterior_cholesky, base_log_prob

    def marginal(self, mask, particles):
        key = (mask, particles)
        if key in self.cache:
            return self.cache[key]
        check_deadline(self.deadline)
        if mask == 511:
            values = batched_log_prob(self.model, tensor(self.source), tensor(self.target))
            result = values, {'ess_median': None, 'ess_minimum': None}
            self.cache[key] = result
            return result
        kept, missing = self.columns(mask), self.columns(511 ^ mask)
        if self.model.domain == 'real':
            missing, means, chol, base = self.gaussian_proposal(mask)
            if self.identity_spline:
                # Exact importance identity: every correction weight is one.
                # Keep the same finite outer EI estimate; skip only redundant
                # neural evaluations of identically unit inner weights.
                result = base, {'ess_median': float(particles),
                                'ess_minimum': float(particles),
                                'exact_unit_weights': True}
                self.cache[key] = result
                return result
        else:
            base = np.zeros(self.outer)
        log_means, ess = [], []
        for start in range(0, self.outer, max(1, self.batch_rows // particles)):
            check_deadline(self.deadline)
            stop = min(self.outer, start + max(1, self.batch_rows // particles))
            completed = self.inner[start:stop, :particles].copy()
            if self.model.domain == 'real':
                completed[:, :, missing] = (means[start:stop, None, :]
                    + completed[:, :, missing] @ chol.T)
            completed[:, :, kept] = self.source[start:stop, None, kept]
            target = np.repeat(self.target[start:stop], particles, axis=0)
            with torch.no_grad():
                log_prob, gaussian = self.density_model.log_prob_components(
                    tensor(completed.reshape(-1, self.dimension)).to(self.density_device),
                    tensor(target).to(self.density_device))
                weights = ((log_prob-gaussian) if self.model.domain == 'real'
                           else log_prob).cpu().double().numpy().reshape(stop-start, particles)
            log_means.append(logsumexp(weights, axis=1)-np.log(particles))
            normalized = weights - logsumexp(weights, axis=1)[:, None]
            ess.extend(np.exp(-logsumexp(2*normalized, axis=1)))
        values = base + np.concatenate(log_means)
        result = values, {'ess_median': float(np.median(ess)),
                          'ess_minimum': float(np.min(ess))}
        self.cache[key] = result
        return result

    def information(self, masks, particles):
        denominator, denominator_diagnostic = self.marginal(0, particles)
        values, diagnostics = {}, {}
        for mask in masks:
            numerator, diagnostic = self.marginal(mask, particles)
            pointwise = numerator-denominator
            values[mask] = float(pointwise.mean())
            diagnostics[mask] = diagnostic
        return values, {'denominator': denominator_diagnostic, 'queries': diagnostics}
