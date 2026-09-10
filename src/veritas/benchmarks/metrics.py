"""Ground-truth evaluation metrics for causal estimators.

Because benchmarks carry the true individual effect, we can measure accuracy the
way prediction never lets us in the real world:

* **PEHE** — Precision in Estimation of Heterogeneous Effect: the RMSE between the
  estimated and true individual treatment effects. The headline CATE metric.
* **ATE error** — |estimated ATE - true ATE|, absolute and relative. This is what
  exposes confounding bias: the naive estimate is far off, doubly-robust is close.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def pehe(tau_hat, tau_true) -> float:
    """sqrt(mean((tau_hat - tau_true)^2)) — RMSE of individual treatment effects."""
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    tau_true = np.asarray(tau_true, dtype=float).ravel()
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_error(ate_hat: float, ate_true: float) -> float:
    """Absolute error of an ATE estimate."""
    return float(abs(ate_hat - ate_true))


def ate_metrics(ate_hat: float, ate_true: float) -> Dict[str, float]:
    """Absolute, signed, and relative ATE error in one call."""
    err = ate_hat - ate_true
    rel = abs(err) / abs(ate_true) if ate_true != 0 else float("nan")
    return {
        "ate_hat": float(ate_hat),
        "ate_true": float(ate_true),
        "abs_error": float(abs(err)),
        "signed_error": float(err),
        "rel_error": float(rel),
    }


def policy_risk(policy, tau_true) -> float:
    """Expected regret of a 0/1 treatment policy vs the oracle (per unit).

    The oracle treats exactly the units with positive true effect. Regret is the
    true effect lost by disagreeing with it.
    """
    policy = np.asarray(policy, dtype=float).ravel()
    tau_true = np.asarray(tau_true, dtype=float).ravel()
    oracle_value = np.mean(np.maximum(tau_true, 0.0))
    policy_value = np.mean(policy * tau_true)
    return float(oracle_value - policy_value)
