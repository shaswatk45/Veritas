"""Policy learning: turn per-unit CATE into a *whom-to-treat* decision and value it.

A treatment rule pi(x) = 1[tau(x) > cost] treats a unit only when its estimated
benefit clears the cost of treating. We value a rule the honest, doubly-robust
way (AIPW): the estimate stays valid if either nuisance model is right, and on
benchmarks we can also compute the rule's TRUE value from ground-truth outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from veritas.estimate.nuisance import Nuisances


@dataclass
class PolicyResult:
    name: str
    value: float           # estimated expected outcome under the policy
    treated_frac: float
    se: float = float("nan")
    true_value: float = float("nan")

    def as_row(self) -> dict:
        row = {
            "policy": self.name,
            "treated_%": round(100 * self.treated_frac, 1),
            "policy_value (DR)": round(self.value, 4),
        }
        if np.isfinite(self.se):
            row["SE"] = round(self.se, 4)
        if np.isfinite(self.true_value):
            row["true_value"] = round(self.true_value, 4)
        return row


def treatment_rule(tau, cost: float = 0.0) -> np.ndarray:
    """Treat where estimated effect exceeds the per-treatment cost."""
    return (np.asarray(tau, dtype=float) > cost).astype(int)


def dr_policy_value(policy, t, y, nu: Nuisances):
    """Doubly-robust (AIPW) value of a fixed 0/1 policy: E[Y under policy]."""
    policy = np.asarray(policy, dtype=float)
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    mu1, mu0, e = nu.mu1, nu.mu0, nu.e
    # AIPW outcome under treatment / control, then select per the policy.
    q1 = mu1 + t * (y - mu1) / e
    q0 = mu0 + (1 - t) * (y - mu0) / (1 - e)
    scores = policy * q1 + (1 - policy) * q0
    value = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(y)))
    return value, se


def true_policy_value(policy, mu0, mu1) -> float:
    """Ground-truth value of a policy (benchmarks only)."""
    policy = np.asarray(policy, dtype=float)
    return float(np.mean(policy * np.asarray(mu1) + (1 - policy) * np.asarray(mu0)))


def evaluate_policies(
    tau_hat, t, y, nu: Nuisances, cost: float = 0.0,
    mu0=None, mu1=None,
) -> List[PolicyResult]:
    """Compare the learned rule against treat-all and treat-none."""
    n = len(y)
    learned = treatment_rule(tau_hat, cost)
    policies = {
        "Learned rule (CATE)": learned,
        "Treat everyone": np.ones(n, dtype=int),
        "Treat no one": np.zeros(n, dtype=int),
    }
    out = []
    for name, pi in policies.items():
        value, se = dr_policy_value(pi, t, y, nu)
        tv = true_policy_value(pi, mu0, mu1) if mu0 is not None else float("nan")
        out.append(PolicyResult(name, value, float(pi.mean()), se, tv))
    return out
