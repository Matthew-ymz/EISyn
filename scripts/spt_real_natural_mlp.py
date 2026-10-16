"""Natural-only real-coordinate MLP with a diagonal Gaussian conditional TM."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from scripts.spt_glv_finite_tm import RealConditionalTM
from scripts.spt_kuramoto_finite_tm import check_time


class NaturalRealMLPConditionalTM(RealConditionalTM):
    def __init__(self, weights, biases, center, scale, variance):
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.center, self.scale, self.variance = map(lambda a: np.asarray(a, dtype=float), (center, scale, variance))
        if len(self.weights) != 3 or len(self.biases) != 3:
            raise ValueError('Expected two hidden layers and one output layer')
        width = 9
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Invalid real MLP layer')
            width = w.shape[1]
        if width != 9 or any(a.shape != (9,) or not np.isfinite(a).all() for a in (self.center, self.scale, self.variance)) or np.any(self.scale <= 0) or np.any(self.variance <= 0):
            raise ValueError('Expected finite parameters, positive scales and residual variances')
        self.precision = 1 / self.variance

    def mean(self, source, *, jacobian=False):
        x = np.asarray(source, dtype=float)
        if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
            raise ValueError('Expected finite (samples,9) real states')
        h = (x - self.center) / self.scale
        if jacobian:
            jac = np.broadcast_to(np.diag(1 / self.scale), (len(x), 9, 9)).copy()
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if jacobian:
                jac = np.einsum('nki,ko->noi', jac, w, optimize=True)
            if layer < 2:
                h = np.tanh(h)
                if jacobian:
                    jac *= (1 - h * h)[:, :, None]
        return (x + h, jac + np.eye(9)[None]) if jacobian else x + h

    def mean_selected(self, source, targets, variables=None):
        if variables is not None:
            raise ValueError('Real MLP requires the complete nine-coordinate source')
        return self.mean(source)[:, targets]

    mean_numpy = mean

    def payload(self):
        return dict(family='natural_real_mlp_v1', weights=[w.tolist() for w in self.weights],
                    biases=[b.tolist() for b in self.biases], center=self.center.tolist(),
                    scale=self.scale.tolist(), variance=self.variance.tolist())

    @classmethod
    def load(cls, p):
        if p.get('family') != 'natural_real_mlp_v1':
            raise ValueError('Unexpected model family')
        return cls(p['weights'], p['biases'], p['center'], p['scale'], p['variance'])

    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def fit(cls, x, y, vx, vy, *, deadline, seed=20261016):
        import torch
        from torch import nn

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        center, scale = x.mean(0), x.std(0)
        if np.any(scale <= 0):
            raise ArithmeticError('Degenerate natural source coordinate; no silent scale floor')
        net = nn.Sequential(nn.Linear(9, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 9)).double()
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        tensor = lambda a: torch.tensor(a, dtype=torch.float64)
        tx, ty, tvx, tvy = map(tensor, ((x - center) / scale, y - x, (vx - center) / scale, vy - vx))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=1e-5)
        best, best_loss, best_epoch, history = None, float('inf'), 0, []
        started = time.monotonic()
        training_end = min(deadline - 90, started + 480)
        for epoch in range(1, 251):
            if time.monotonic() >= training_end:
                break
            for ids in torch.randperm(len(tx)).split(1024):
                check_time(deadline)
                loss = ((net(tx[ids]) - ty[ids]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                value = float(((net(tvx) - tvy) ** 2).mean())
            history.append(dict(epoch=epoch, validation_mse=value))
            if value < best_loss - 1e-8:
                best, best_loss, best_epoch = {k: v.detach().clone() for k, v in net.state_dict().items()}, value, epoch
            if epoch % 25 == 0:
                print(json.dumps(dict(stage='real_mlp_fit', epoch=epoch, validation_mse=value)), flush=True)
            if epoch - best_epoch >= 25:
                break
        if best is None:
            raise TimeoutError('No completed MLP epoch')
        net.load_state_dict(best)
        linear = [a for a in net if isinstance(a, nn.Linear)]
        model = cls([a.weight.detach().numpy().T.copy() for a in linear],
                    [a.bias.detach().numpy().copy() for a in linear], center, scale, np.ones(9))
        model.variance = ((y - model.mean(x)) ** 2).mean(0)
        if np.any(model.variance <= 0) or not np.isfinite(model.variance).all():
            raise ArithmeticError('Invalid fitted noise variance; no noise floor added')
        model.precision = 1 / model.variance
        with torch.no_grad():
            difference = float(abs((x[:32] + net(tx[:32]).numpy()) - model.mean(x[:32])).max())
        if difference > 1e-10:
            raise ArithmeticError('Numpy/Torch inference mismatch')
        return model, dict(architecture=[9, 64, 64, 9], activation='tanh', identity_skip=True,
            preprocessing='training-coordinate mean/std only', optimizer='AdamW', learning_rate=.003,
            weight_decay=1e-5, batch_size=1024, maximum_epochs=250, patience=25,
            epochs=len(history), selected_epoch=best_epoch, history=history,
            loss='mean squared real displacement error', training_seconds=time.monotonic() - started,
            numpy_torch_maximum_error=difference, seed=seed)
