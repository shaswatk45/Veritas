"""Average treatment effect estimators, from naive to doubly-robust.

The story these tell together: the **naive** difference in means is biased under
confounding; **IPW** reweights by the propensity to remove it; **AIPW** and
**TMLE** are *doubly robust* — consistent if EITHER the outcome model or the
propensity model is right — and come with honest influence-function confidence
intervals. **DML** (EconML) is the debiased-ML cross-fit estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from scipy import stats

from veritas.estimate.nuisance import crossfit_nuisances, Nuisances, _to_np


@dataclass
class ATEResult:
    name: str
    ate: float
    se: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    note: str = ""

    def as_row(self) -> dict:
        return {
            "estimator": self.name,
            "ATE": round(self.ate, 4),
            "SE": round(self.se, 4) if np.isfinite(self.se) else None,
            "95% CI": (f"[{self.ci_low:.3f}, {self.ci_high:.3f}]"
                       if np.isfinite(self.ci_low) else "—"),
            "note": self.note,
        }


def _ci(ate, se):
    if not np.isfinite(se):
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.975)
    return ate - z * se, ate + z * se


# --------------------------------------------------------------------------- #
def naive_ate(t, y) -> ATEResult:
    """Difference in means. Correct only under randomization; biased otherwise."""
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    lo, hi = _ci(ate, se)
    return ATEResult("Naive diff-in-means", ate, se, lo, hi, "biased under confounding")


def ipw_ate(t, y, nu: Nuisances) -> ATEResult:
    """Stabilized (Hajek) inverse-propensity weighting."""
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    e = nu.e
    w1, w0 = t / e, (1 - t) / (1 - e)
    m1 = np.sum(w1 * y) / np.sum(w1)
    m0 = np.sum(w0 * y) / np.sum(w0)
    ate = m1 - m0
    # Influence-function SE for the Hajek estimator.
    ic = w1 * (y - m1) / np.mean(w1) - w0 * (y - m0) / np.mean(w0)
    se = np.std(ic, ddof=1) / np.sqrt(len(y))
    lo, hi = _ci(ate, se)
    return ATEResult("IPW (stabilized)", ate, se, lo, hi, "reweights by propensity")


def aipw_ate(t, y, nu: Nuisances) -> ATEResult:
    """Augmented IPW (doubly robust). ATE = mean of the efficient influence fn."""
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    mu1, mu0, e = nu.mu1, nu.mu0, nu.e
    psi = (mu1 - mu0
           + t * (y - mu1) / e
           - (1 - t) * (y - mu0) / (1 - e))
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(y)))
    lo, hi = _ci(ate, se)
    return ATEResult("AIPW (doubly robust)", ate, se, lo, hi, "consistent if outcome OR propensity right")


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def tmle_ate(t, y, nu: Nuisances, outcome_type: str = "continuous") -> ATEResult:
    """Targeted Maximum Likelihood Estimation (doubly robust, plug-in + targeting).

    Bounds the (continuous) outcome to [0,1], fits a one-parameter logistic
    fluctuation with the clever covariate H = T/e - (1-T)/(1-e), then reads off
    the updated plug-in ATE. SE from the efficient influence curve.
    """
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    mu1, mu0, e = nu.mu1.copy(), nu.mu0.copy(), nu.e

    if outcome_type == "binary":
        a, b = 0.0, 1.0
    else:
        a, b = float(y.min()), float(y.max())
    span = (b - a) if b > a else 1.0
    ys = (y - a) / span
    Q1, Q0 = np.clip((mu1 - a) / span, 1e-4, 1 - 1e-4), np.clip((mu0 - a) / span, 1e-4, 1 - 1e-4)
    QA = np.where(t == 1, Q1, Q0)

    # Clever covariate and Newton fluctuation for epsilon.
    H1, H0 = 1.0 / e, -1.0 / (1 - e)
    HA = np.where(t == 1, H1, H0)
    off = _logit(QA)
    eps = 0.0
    for _ in range(100):
        p = _expit(off + eps * HA)
        score = np.sum(HA * (ys - p))
        hess = -np.sum(HA * HA * p * (1 - p))
        if abs(hess) < 1e-12:
            break
        step = score / hess
        eps -= step
        if abs(step) < 1e-8:
            break

    Q1s = _expit(_logit(Q1) + eps * H1)
    Q0s = _expit(_logit(Q0) + eps * H0)
    QAs = np.where(t == 1, Q1s, Q0s)
    psi_scaled = float(np.mean(Q1s - Q0s))
    ate = psi_scaled * span

    # Efficient influence curve (on scaled outcome), rescaled.
    ic = (HA * (ys - QAs) + (Q1s - Q0s) - psi_scaled) * span
    se = float(np.std(ic, ddof=1) / np.sqrt(len(y)))
    lo, hi = _ci(ate, se)
    return ATEResult("TMLE (doubly robust)", ate, se, lo, hi, "plug-in + targeting step")


def dml_ate(X, t, y, n_folds: int = 5, random_state: int = 42) -> ATEResult:
    """Debiased/Double ML ATE via EconML LinearDML (raises if econml missing)."""
    from econml.dml import LinearDML
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    est = LinearDML(
        model_y=HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=random_state),
        model_t=HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=random_state),
        discrete_treatment=True, cv=n_folds, random_state=random_state,
    )
    est.fit(np.asarray(y, dtype=float), np.asarray(t).astype(int), X=_to_np(X))
    ate = float(est.ate(_to_np(X)))
    try:
        lo, hi = est.ate_interval(_to_np(X), alpha=0.05)
        lo, hi = float(lo), float(hi)
        se = (hi - lo) / (2 * 1.959964)
    except Exception:
        se, lo, hi = float("nan"), float("nan"), float("nan")
    return ATEResult("DML (EconML)", ate, se, lo, hi, "debiased ML, cross-fit")


def all_ate_estimators(
    data, n_folds: int = 5, clip: float = 0.02, random_state: int = 42,
) -> List[ATEResult]:
    """Run the full ATE suite on a CausalData object, sharing cross-fit nuisances."""
    X, t, y = data.X, data.t, data.y
    binary = getattr(data, "outcome_type", "continuous") == "binary"
    nu = crossfit_nuisances(X, t, y, n_folds=n_folds, clip=clip,
                            random_state=random_state, binary_outcome=binary)
    results = [
        naive_ate(t, y),
        ipw_ate(t, y, nu),
        aipw_ate(t, y, nu),
        tmle_ate(t, y, nu, outcome_type="binary" if binary else "continuous"),
    ]
    try:
        results.append(dml_ate(X, t, y, n_folds=n_folds, random_state=random_state))
    except Exception as exc:  # econml missing / fit issue
        results.append(ATEResult("DML (EconML)", float("nan"), note=f"skipped: {exc!r}"[:60]))
    return results
