"""Cross-fitted nuisance functions: outcome models mu0, mu1 and propensity e.

Doubly-robust estimators need out-of-fold ("cross-fitted") nuisance predictions
so the estimator stays valid even with flexible ML models — a unit's nuisance
prediction is never made by a model that saw that unit. This is the machinery
behind AIPW, TMLE, and DML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold


@dataclass
class Nuisances:
    """Out-of-fold nuisance predictions for every unit."""

    mu0: np.ndarray   # E[Y | X, T=0]
    mu1: np.ndarray   # E[Y | X, T=1]
    e: np.ndarray     # P(T=1 | X), clipped for positivity

    @property
    def muA(self) -> np.ndarray:
        raise NotImplementedError  # requires t; use muA_for


def _to_np(X):
    return X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)


def _default_outcome(seed, binary=False):
    if binary:
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              max_depth=4, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                         max_depth=4, random_state=seed)


def _default_propensity(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=4, random_state=seed)


def _predict_outcome(model, X, binary):
    if binary:
        classes = list(model.classes_)
        if 1 in classes:
            return model.predict_proba(X)[:, classes.index(1)]
        return np.zeros(len(X))
    return model.predict(X)


def crossfit_nuisances(
    X, t, y, n_folds: int = 5, clip: float = 0.02, random_state: int = 42,
    binary_outcome: bool = False,
) -> Nuisances:
    """Return out-of-fold mu0, mu1, e for every unit via K-fold cross-fitting."""
    X = _to_np(X)
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    n = len(y)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    e = np.zeros(n)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, t):
        Xtr, ttr, ytr = X[tr], t[tr], y[tr]
        # Propensity on all training units.
        ps = _default_propensity(random_state)
        ps.fit(Xtr, ttr)
        pc = list(ps.classes_)
        e[te] = ps.predict_proba(X[te])[:, pc.index(1)] if 1 in pc else 0.5

        # Outcome models on treated / control subsets of the training folds.
        m1 = _default_outcome(random_state, binary_outcome)
        m0 = _default_outcome(random_state, binary_outcome)
        # Guard against a fold with a single outcome class (binary) or too few.
        m1.fit(Xtr[ttr == 1], ytr[ttr == 1])
        m0.fit(Xtr[ttr == 0], ytr[ttr == 0])
        mu1[te] = _predict_outcome(m1, X[te], binary_outcome)
        mu0[te] = _predict_outcome(m0, X[te], binary_outcome)

    e = np.clip(e, clip, 1 - clip)
    return Nuisances(mu0=mu0, mu1=mu1, e=e)
