"""A data-trained periodic MLP shared by intervention and attribution adapters.

The network predicts a one-step phase displacement from sin/cos coordinates.
A fitted product-von-Mises residual distribution supplies a normalized conditional
CDF transport, sample(), and log_prob(); the target remains the full future phase.
No generator supports or coefficients are accepted by the fitter.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from scripts.spt_kuramoto_finite_tm import (
    PeriodicConditionalTM, check_time, digest, fit_concentration,
)
from scripts.spt_local_hyperedge_kuramoto import wrap

FAMILY = 'natural_periodic_mlp_v1'


def chain_partitions(states, *, seed=20261016):
    """Disjoint chains, retaining every adjacent source/future pair per chain."""
    states = np.asarray(states, dtype=float)
    if states.shape != (513, 64, 9) or not np.isfinite(states).all():
        raise ValueError('Expected the registered finite (513,64,9) trajectory pool')
    order = np.random.default_rng(seed).permutation(64)
    groups = dict(train=order[:48], validation=order[48:56], test=order[56:])
    data, manifest = {}, {'seed': seed, 'split_unit': 'independent trajectory',
                          'pair_row_id': 'time_index * 64 + chain_index', 'splits': {}}
    for name, indices in groups.items():
        x = states[:-1, indices].reshape(-1, 9).copy()
        y = states[1:, indices].reshape(-1, 9).copy()
        data[name] = x, y
        manifest['splits'][name] = dict(chains=indices.tolist(), pairs=len(x),
                                        paired_arrays_sha256=digest(x, y))
    manifest['shared_by'] = ['SPT MLP', 'SHAP MLP', 'WMS data adapter']
    return data, manifest


class NaturalMLPConditionalTM(PeriodicConditionalTM):
    """Numpy inference/Jacobian for a frozen tanh MLP plus circular residual TM."""

    def __init__(self, weights, biases, kappa):
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.kappa = np.asarray(kappa, dtype=float)
        if len(self.weights) != 3 or len(self.biases) != 3:
            raise ValueError('Expected exactly two hidden layers and one output layer')
        width = 18
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError('Incompatible MLP layer dimensions')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite MLP parameters')
            width = w.shape[1]
        if width != 9 or self.kappa.shape != (9,) or np.any(self.kappa <= 0) or not np.isfinite(self.kappa).all():
            raise ValueError('Expected nine outputs and finite positive concentrations')

    def mean(self, source, *, jacobian=False):
        x = np.asarray(source, dtype=float)
        if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
            raise ValueError('Expected finite (samples,9) source states')
        h = np.concatenate((np.sin(x), np.cos(x)), axis=1)
        if jacobian:
            j = np.zeros((len(x), 18, 9))
            for k in range(9):
                j[:, k, k] = np.cos(x[:, k])
                j[:, k + 9, k] = -np.sin(x[:, k])
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if jacobian:
                j = np.einsum('nki,ko->noi', j, w, optimize=True)
            if layer < 2:
                h = np.tanh(h)
                if jacobian:
                    j *= (1 - h * h)[:, :, None]
        value = x + h
        return (value, j + np.eye(9)[None]) if jacobian else value

    def payload(self):
        return dict(family=FAMILY, weights=[w.tolist() for w in self.weights],
                    biases=[b.tolist() for b in self.biases], kappa=self.kappa.tolist(),
                    preprocessing='concatenate(sin(x),cos(x)); identity residual skip',
                    residual='independent constant-concentration von Mises; fitted on training residuals')

    @classmethod
    def load(cls, payload):
        if payload.get('family') != FAMILY:
            raise ValueError('Unexpected MLP model family')
        return cls(payload['weights'], payload['biases'], payload['kappa'])

    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def fit(cls, source, target, validation_source, validation_target, *, deadline,
            seed=20261016, progress=None):
        """One fixed architecture/optimizer; validation chooses the epoch only."""
        import torch
        from torch import nn

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, 64),
                            nn.Tanh(), nn.Linear(64, 9)).double()
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        tensor = lambda a: torch.tensor(a, dtype=torch.float64)
        features = lambda a: np.concatenate((np.sin(a), np.cos(a)), axis=1)
        x, y = tensor(features(source)), tensor(wrap(target - source))
        vx, vy = tensor(features(validation_source)), tensor(wrap(validation_target - validation_source))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=1e-5)
        history, best, best_loss, best_epoch = [], None, float('inf'), 0
        started = time.monotonic()
        training_end = min(deadline - 90, started + 480)
        for epoch in range(1, 251):
            if time.monotonic() >= training_end:
                break
            net.train()
            permutation = torch.randperm(len(x))
            for indices in permutation.split(1024):
                check_time(deadline)
                prediction = net(x[indices])
                loss = (1 - torch.cos(prediction - y[indices])).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                value = float((1 - torch.cos(net(vx) - vy)).mean())
            history.append(dict(epoch=epoch, validation_circular_loss=value))
            if value < best_loss - 1e-8:
                best_loss, best_epoch = value, epoch
                best = {key: a.detach().clone() for key, a in net.state_dict().items()}
            if epoch % 10 == 0 and progress:
                progress(dict(epoch=epoch, best_epoch=best_epoch, validation_circular_loss=value))
            if epoch - best_epoch >= 25:
                break
        if best is None:
            raise TimeoutError('No completed MLP epoch within approved budget')
        net.load_state_dict(best)
        linear = [layer for layer in net if isinstance(layer, nn.Linear)]
        weights = [layer.weight.detach().numpy().T.copy() for layer in linear]
        biases = [layer.bias.detach().numpy().copy() for layer in linear]
        model = cls(weights, biases, np.ones(9))
        residual = wrap(target - model.mean(source))
        model.kappa = np.array([fit_concentration(residual[:, j]) for j in range(9)])
        with torch.no_grad():
            torch_prediction = source[:32] + net(x[:32]).numpy()
        error = float(abs(torch_prediction - model.mean(source[:32])).max())
        if error > 1e-10:
            raise ArithmeticError(f'Torch/Numpy prediction discrepancy: {error}')
        return model, dict(seed=seed, architecture=[18, 64, 64, 9], activation='tanh',
            identity_skip=True, optimizer='AdamW', learning_rate=.003, weight_decay=1e-5,
            batch_size=1024, maximum_epochs=250, patience=25, epochs=len(history),
            selected_epoch=best_epoch, selection='validation circular loss only',
            best_validation_circular_loss=best_loss, history=history,
            training_seconds=time.monotonic() - started, numpy_torch_maximum_error_rad=error,
            training_time_capped=time.monotonic() >= training_end)
