"""Sensitivity analysis for unmeasured confounding — is the finding honest?

No observational study can rule out a hidden confounder. Instead of pretending
otherwise, we quantify how strong one would have to be to overturn the result.

* **E-value** (VanderWeele & Ding, 2017): the minimum strength of association —
  on the risk-ratio scale — that an unmeasured confounder would need with *both*
  treatment and outcome to fully explain away the observed effect. Reported for
  the point estimate and for the CI limit nearest the null.

* **Simulated confounder**: inject an unmeasured confounder U of increasing
  strength and re-estimate, finding the strength that drives the effect to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EValueResult:
    rr: float
    e_value_point: float
    e_value_ci: Optional[float]
    note: str = ""

    def summary(self) -> dict:
        return {
            "risk_ratio_scale": round(self.rr, 3),
            "E-value (point)": round(self.e_value_point, 3),
            "E-value (CI limit)": round(self.e_value_ci, 3) if self.e_value_ci is not None else None,
            "interpretation": self.note,
        }


def e_value(rr: float) -> float:
    """E-value for a risk ratio (or a risk ratio-scaled association)."""
    rr = float(rr)
    if rr < 1:  # protective effect: evaluate on the >1 side by inverting
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def _rr_from_smd(d: float) -> float:
    """Approximate a risk ratio from a standardized mean difference (VanderWeele
    & Ding's continuous-outcome bridge): RR ~= exp(0.91 * d)."""
    return float(np.exp(0.91 * d))


def e_value_from_estimate(
    ate: float, se: float, y_sd: Optional[float] = None, outcome_type: str = "continuous",
    rr: Optional[float] = None,
) -> EValueResult:
    """Compute E-values for an effect estimate.

    For a continuous outcome, the ATE is converted to a standardized mean
    difference (ATE / SD of outcome) and then to an approximate risk ratio. For a
    binary/RR effect, pass ``rr`` directly.
    """
    if rr is None:
        if y_sd is None or y_sd == 0:
            raise ValueError("continuous outcome needs y_sd to standardize")
        d = ate / y_sd
        rr_point = _rr_from_smd(d)
        # CI limit on RR nearest the null, via the SMD CI.
        d_ci = (ate - 1.959964 * se) / y_sd if ate > 0 else (ate + 1.959964 * se) / y_sd
        rr_ci = _rr_from_smd(d_ci)
    else:
        rr_point = rr
        rr_ci = None

    ev_point = e_value(rr_point)
    ev_ci = None
    if rr_ci is not None:
        # If the CI crosses the null, the effect isn't robust: E-value = 1.
        crosses = (rr_point > 1 and rr_ci <= 1) or (rr_point < 1 and rr_ci >= 1)
        ev_ci = 1.0 if crosses else e_value(rr_ci)

    note = (f"an unmeasured confounder would need RR>={ev_point:.2f} with both "
            f"treatment and outcome to explain away the point estimate")
    return EValueResult(rr=rr_point, e_value_point=ev_point, e_value_ci=ev_ci, note=note)


def simulate_unmeasured_confounder(
    X, t, y, estimator, strengths=None, random_state: int = 42,
):
    """Erosion curve: how the estimate moves toward the null as an unmeasured
    confounder aligned with the treatment residual grows in strength.

    We fit a propensity e(x), form the treatment residual r = t - e (the part of
    treatment not explained by the measured covariates), and subtract ``gamma * r``
    from the outcome — i.e. attribute an increasing share of the treatment-outcome
    association to a hidden confounder — then re-estimate. As gamma grows the
    adjusted effect erodes monotonically toward zero: the tipping point is where a
    confounder of that strength would fully explain the result away.

    ``estimator`` is a callable (X, t, y) -> ate. Returns [{gamma, ate}].
    """
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    Xn = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t).astype(int)

    # Out-of-fold propensity -> treatment residual.
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=random_state)
    e = cross_val_predict(clf, Xn, t, cv=3, method="predict_proba")[:, 1]
    e = np.clip(e, 0.02, 0.98)
    r = t - e

    cols = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(Xn.shape[1])]
    ate0 = estimator(pd.DataFrame(Xn, columns=cols), t, y)
    if strengths is None:
        # Sweep the bias from 0 up to ~1.4x the effect (past the tipping point).
        scale = abs(ate0) / (np.var(r) if np.var(r) > 0 else 1.0)
        strengths = [round(m, 3) for m in np.linspace(0, 1.4 * scale, 8)]

    out = []
    for g in strengths:
        y_adj = y - g * r
        ate = estimator(pd.DataFrame(Xn, columns=cols), t, y_adj)
        # Report gamma as the induced bias in outcome units (interpretable).
        out.append({"gamma": float(g * np.var(r)), "ate": float(ate)})
    return out
