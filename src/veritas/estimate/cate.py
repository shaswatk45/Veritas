"""Heterogeneous treatment effect (CATE) estimators — *who* benefits, not just
the average. This is where uplift modeling lives; Veritas contains it.

* ``xlearner_cate`` — Kunzel et al. X-learner (native), robust under imbalance.
* ``causal_forest_cate`` — EconML Causal Forest (honest, nonparametric CATE).
* ``dml_cate`` — EconML Linear DML (debiased, linear-in-X CATE with CIs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from veritas.estimate.nuisance import _to_np


@dataclass
class CATEResult:
    name: str
    tau: np.ndarray                 # per-unit estimated effect
    est: object = field(default=None, repr=False)

    @property
    def ate(self) -> float:
        return float(np.mean(self.tau))


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                         max_depth=4, random_state=seed)


def _clf(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=4, random_state=seed)


def xlearner_cate(X, t, y, random_state: int = 42, clip: float = 0.02) -> CATEResult:
    """Native X-learner CATE (continuous outcome)."""
    Xn = _to_np(X)
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    Xt, Xc = Xn[t == 1], Xn[t == 0]
    yt, yc = y[t == 1], y[t == 0]

    mu1, mu0 = _reg(random_state), _reg(random_state)
    mu1.fit(Xt, yt)
    mu0.fit(Xc, yc)

    d_treated = yt - mu0.predict(Xt)     # effect imputed for treated
    d_control = mu1.predict(Xc) - yc     # effect imputed for control
    tau_t, tau_c = _reg(random_state), _reg(random_state)
    tau_t.fit(Xt, d_treated)
    tau_c.fit(Xc, d_control)

    g = _clf(random_state)
    g.fit(Xn, t)
    classes = list(g.classes_)
    p = g.predict_proba(Xn)[:, classes.index(1)] if 1 in classes else np.full(len(Xn), t.mean())
    p = np.clip(p, clip, 1 - clip)
    tau = p * tau_c.predict(Xn) + (1 - p) * tau_t.predict(Xn)
    return CATEResult("X-learner", tau, est=(tau_t, tau_c))


def causal_forest_cate(X, t, y, n_folds: int = 5, random_state: int = 42) -> CATEResult:
    """EconML Causal Forest DML (raises if econml missing)."""
    from econml.dml import CausalForestDML

    est = CausalForestDML(
        model_y=_reg(random_state), model_t=_clf(random_state),
        discrete_treatment=True, n_estimators=400, min_samples_leaf=10,
        cv=n_folds, random_state=random_state,
    )
    est.fit(np.asarray(y, dtype=float), np.asarray(t).astype(int), X=_to_np(X))
    tau = np.asarray(est.effect(_to_np(X))).ravel()
    return CATEResult("Causal Forest", tau, est=est)


def dml_cate(X, t, y, n_folds: int = 5, random_state: int = 42) -> CATEResult:
    """EconML Linear DML CATE (raises if econml missing)."""
    from econml.dml import LinearDML

    est = LinearDML(
        model_y=_reg(random_state), model_t=_clf(random_state),
        discrete_treatment=True, cv=n_folds, random_state=random_state,
    )
    est.fit(np.asarray(y, dtype=float), np.asarray(t).astype(int), X=_to_np(X))
    tau = np.asarray(est.effect(_to_np(X))).ravel()
    return CATEResult("Linear DML", tau, est=est)
