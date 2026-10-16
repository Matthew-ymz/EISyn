"""Natural-data residual MLP with a complete, label-free periodic pair basis."""
from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations

import numpy as np
from scipy.special import expit

from scripts.spt_kuramoto_finite_tm import PeriodicConditionalTM, fit_concentration
from scripts.spt_local_hyperedge_kuramoto import wrap

PAIRS = np.array(list(combinations(range(9), 2)))
FAMILY = 'natural_periodic_pair_mlp_v1'


def features(x, jacobian=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
        raise ValueError('Expected finite (n,9) phases')
    delta = x[:, PAIRS[:, 1]] - x[:, PAIRS[:, 0]]
    value = np.concatenate([np.sin(delta), np.cos(delta),
                            np.sin(2*delta), np.cos(2*delta)], axis=1)
    if not jacobian:
        return value
    derivative = np.concatenate([np.cos(delta), -np.sin(delta),
                                 2*np.cos(2*delta), -2*np.sin(2*delta)], axis=1)
    incidence = np.zeros((144, 9))
    for block in range(4):
        for p, (i, j) in enumerate(PAIRS):
            incidence[block*36+p, i] = -1
            incidence[block*36+p, j] = 1
    return value, derivative[:, :, None] * incidence[None]


class PairMLPConditionalTM(PeriodicConditionalTM):
    """Complete future phases; independent von Mises residual is an approximation."""
    def __init__(self, weights, biases, feature_center, feature_scale,
                 output_center, output_scale, kappa):
        self.weights = [np.asarray(w, float) for w in weights]
        self.biases = [np.asarray(b, float) for b in biases]
        for key, value in [('feature_center', feature_center), ('feature_scale', feature_scale),
                           ('output_center', output_center), ('output_scale', output_scale), ('kappa', kappa)]:
            setattr(self, key, np.asarray(value, float))
        if len(self.weights) != 3 or len(self.biases) != 3:
            raise ValueError('Expected two hidden layers')
        width = 144
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError('Invalid network dimensions')
            width = w.shape[1]
        if width != 9 or self.feature_center.shape != (144,) or self.feature_scale.shape != (144,):
            raise ValueError('Invalid feature/output dimensions')
        if any(a.shape != (9,) for a in [self.output_center, self.output_scale, self.kappa]):
            raise ValueError('Expected nine targets')
        arrays = self.weights+self.biases+[self.feature_center, self.feature_scale,
                                          self.output_center, self.output_scale, self.kappa]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Nonfinite parameters')
        if any(np.any(a <= 0) for a in [self.feature_scale, self.output_scale, self.kappa]):
            raise ValueError('Nonpositive scale/concentration; no silent floor')

    def mean(self, source, *, jacobian=False):
        x = np.asarray(source, float)
        if jacobian:
            h, j = features(x, True)
            j = j/self.feature_scale[None, :, None]
        else:
            h = features(x)
        h = (h-self.feature_center)/self.feature_scale
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h@w+b
            if jacobian:
                j = np.einsum('nki,ko->noi', j, w, optimize=True)
            if layer < 2:
                sig = expit(h)
                if jacobian:
                    j *= (sig*(1+h*(1-sig)))[:, :, None]
                h = h*sig
        value = x+self.output_center+self.output_scale*h
        return (value, np.eye(9)[None]+j*self.output_scale[None, :, None]) if jacobian else value

    def payload(self):
        return dict(family=FAMILY, weights=[w.tolist() for w in self.weights],
            biases=[b.tolist() for b in self.biases], feature_center=self.feature_center.tolist(),
            feature_scale=self.feature_scale.tolist(), output_center=self.output_center.tolist(),
            output_scale=self.output_scale.tolist(), kappa=self.kappa.tolist(),
            residual='independent constant-concentration von Mises fitted on training residuals; not full covariance',
            features='all 36 pairs, sin/cos phase difference at k=1,2; no support labels')

    @classmethod
    def load(cls, payload):
        if payload.get('family') != FAMILY:
            raise ValueError('Unexpected model family')
        return cls(**{k: payload[k] for k in ['weights', 'biases', 'feature_center',
                     'feature_scale', 'output_center', 'output_scale', 'kappa']})

    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def fit(cls, source, target, validation_source, validation_target, *, deadline,
            seed=20261016, progress=None):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        f, y = features(source), wrap(target-source)
        fc, fs, yc, ys = f.mean(0), f.std(0), y.mean(0), y.std(0)
        if np.any(fs <= 0) or np.any(ys <= 0):
            raise ValueError('Degenerate training standardization')
        tensor = lambda a: torch.tensor(a, dtype=torch.float64)
        x, y = tensor((f-fc)/fs), tensor((y-yc)/ys)
        vx = tensor((features(validation_source)-fc)/fs)
        vy = tensor((wrap(validation_target-validation_source)-yc)/ys)
        net = nn.Sequential(nn.Linear(144, 128), nn.SiLU(), nn.Linear(128, 128),
                            nn.SiLU(), nn.Linear(128, 9)).double()
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=1e-5)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=3e-5)
        best, best_loss, best_epoch, history = None, float('inf'), 0, []
        started = time.monotonic()
        for epoch in range(1, 801):
            if time.monotonic() >= deadline-90:
                break
            net.train()
            complete = True
            for idx in torch.randperm(len(x)).split(1024):
                if time.monotonic() >= deadline-90:
                    complete = False
                    break
                loss = ((net(x[idx])-y[idx])**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not complete:
                break
            net.eval()
            with torch.no_grad():
                value = float(((net(vx)-vy)**2).mean())
            history.append(dict(epoch=epoch, validation_standardized_increment_mse=value))
            if value < best_loss-1e-8:
                best_loss, best_epoch = value, epoch
                best = {k: a.detach().clone() for k, a in net.state_dict().items()}
            schedule.step()
            if epoch % 10 == 0 and progress:
                progress(dict(epoch=epoch, best_epoch=best_epoch, validation_standardized_increment_mse=value))
            if epoch >= 200 and epoch-best_epoch >= 100:
                break
        if best is None:
            raise TimeoutError('No completed enhanced MLP epoch')
        net.load_state_dict(best)
        layers = [a for a in net if isinstance(a, nn.Linear)]
        model = cls([a.weight.detach().numpy().T.copy() for a in layers],
                    [a.bias.detach().numpy().copy() for a in layers], fc, fs, yc, ys, np.ones(9))
        residual = wrap(target-model.mean(source))
        model.kappa = np.array([fit_concentration(residual[:, j]) for j in range(9)])
        with torch.no_grad():
            prediction = source[:32]+yc+ys*net(x[:32]).numpy()
        error = float(abs(prediction-model.mean(source[:32])).max())
        if error > 1e-10:
            raise ArithmeticError(f'Inference mismatch {error}')
        return model, dict(seed=seed, architecture=[144,128,128,9], activation='SiLU',
            training_standardization_only=True, maximum_epochs=800, minimum_epochs_before_early_stop=200,
            patience=100, learning_rate=.003, cosine_final_lr=3e-5, weight_decay=1e-5, batch_size=1024,
            selection='natural validation standardized increment MSE only', selected_epoch=best_epoch,
            epochs=len(history), best_validation_loss=best_loss, history=history,
            training_seconds=time.monotonic()-started, numpy_torch_maximum_error_rad=error,
            training_time_capped=time.monotonic() >= deadline-90)
