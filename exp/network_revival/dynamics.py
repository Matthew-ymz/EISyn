"""
dynamics.py
===========
Nonlinear interaction dynamics for complex network revival experiments.

Translated from MATLAB KindOfDynamics.m (Sanhedrai et al., Nature Physics 2022).

Each model returns a dict with callable M0, M1, M2, and optionally Ginv / Rinv.
The governing ODE for node i is:
    dx_i/dt = M0(x_i) + M1(x_i) * sum_j [A_ij * M2(x_j)]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def michaelis_menten(a: float = 1, h: float = 2) -> dict:
    """
    Gene regulatory / cellular dynamics (Michaelis-Menten).
    MM model: M0(x)=-x^a, M1(x)=1, M2(x)=x^h/(1+x^h)
    """
    def M0(x):
        x = np.asarray(x, dtype=float)
        return -x ** a

    def M1(x):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x)

    def M2(x):
        x = np.asarray(x, dtype=float)
        xh = x ** h
        return xh / (1.0 + xh)

    def Ginv(y):
        """Inverse of M2: M2(x)=y → x = (y/(1-y))^(1/h)"""
        y = np.clip(np.asarray(y, dtype=float), 1e-15, 1 - 1e-15)
        return (y / (1.0 - y)) ** (1.0 / h)

    def Rinv(y):
        """Inverse of R(x)=-M0(x)/M1(x)=x^a → x = y^(1/a)"""
        y = np.asarray(y, dtype=float)
        return y ** (1.0 / a)

    return dict(name="MM", M0=M0, M1=M1, M2=M2, Ginv=Ginv, Rinv=Rinv,
                x_range=(0.0, 100.0), x_th=0.1)


def neural(mu: float = 10.0, delta: float = 1.0) -> dict:
    """
    Wilson-Cowan neuronal dynamics.
    M0(x)=-x, M1(x)=1, M2(x)=1/(1+exp(-delta*x+mu))

    mu controls the bistability threshold (paper uses mu=10).
    """
    def M0(x):
        x = np.asarray(x, dtype=float)
        return -x

    def M1(x):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x)

    def M2(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-delta * x + mu))

    def Ginv(y):
        """Inverse of sigmoid: y = 1/(1+exp(-delta*x+mu)) → x = (mu - ln((1-y)/y))/delta"""
        y = np.clip(np.asarray(y, dtype=float), 1e-15, 1 - 1e-15)
        return (mu - np.log((1.0 - y) / y)) / delta

    def Rinv(y):
        """Inverse of R(x)=x → x=y"""
        return np.asarray(y, dtype=float)

    return dict(name="Neural", M0=M0, M1=M1, M2=M2, Ginv=Ginv, Rinv=Rinv,
                mu=mu, delta=delta, x_range=(0.0, 300.0), x_th=1.0)


def ecological(F: float = 5.0, B: float = 3.0, C: float = 3.0, K: float = 10.0) -> dict:
    """
    Microbial / ecological dynamics (Allee + Lotka-Volterra diffusion).
    M0(x)=F+Bx(1-x/C)(x-K), M1(x)=x, M2(x)=x
    """
    def M0(x):
        x = np.asarray(x, dtype=float)
        return F + B * x * (1 - x / C) * (x - K)

    def M1(x):
        x = np.asarray(x, dtype=float)
        return x

    def M2(x):
        x = np.asarray(x, dtype=float)
        return x

    def Ginv(y):
        return np.asarray(y, dtype=float)

    return dict(name="Eco", M0=M0, M1=M1, M2=M2, Ginv=Ginv,
                x_range=(0.0, 50.0), x_th=0.5)


def sis(beta: float = 1.0) -> dict:
    """
    SIS epidemic dynamics.
    M0(x)=-beta*x, M1(x)=1-x, M2(x)=x
    """
    def M0(x):
        x = np.asarray(x, dtype=float)
        return -beta * x

    def M1(x):
        x = np.asarray(x, dtype=float)
        return 1.0 - x

    def M2(x):
        x = np.asarray(x, dtype=float)
        return x

    return dict(name="SIS", M0=M0, M1=M1, M2=M2,
                x_range=(0.0, 1.0), x_th=0.05)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS = {
    "MM": michaelis_menten,
    "Neural": neural,
    "Eco": ecological,
    "SIS": sis,
}


def get_model(name: str, **kwargs) -> dict:
    """Convenience factory: get_model('MM'), get_model('Neural', mu=10)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)
