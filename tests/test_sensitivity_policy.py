"""Tests for the sensitivity (E-value), policy-value, and benchmark-metric code."""
import numpy as np
import pytest

from veritas.benchmarks import make_synthetic_confounded, policy_risk
from veritas.estimate import crossfit_nuisances
from veritas.policy import treatment_rule, dr_policy_value, true_policy_value, evaluate_policies
from veritas.sensitivity import e_value, e_value_from_estimate


def test_e_value_formula():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2), rel=1e-6)  # 3.414
    # Protective effects invert to the same scale (symmetry).
    assert e_value(0.5) == pytest.approx(e_value(2.0), rel=1e-6)
    # Larger effects need a stronger confounder to explain away.
    assert e_value(3.0) > e_value(2.0)


def test_e_value_ci_crossing_null_is_one():
    # A CI that crosses the null → not robust → E-value at the CI limit is 1.
    res = e_value_from_estimate(ate=0.05, se=0.05, y_sd=1.0)
    assert res.e_value_point > 1.0
    assert res.e_value_ci == pytest.approx(1.0)


def test_policy_value_and_rule():
    data = make_synthetic_confounded(n=3000, random_state=1)
    nu = crossfit_nuisances(data.X, data.t, data.y, n_folds=5, random_state=1)
    rows = evaluate_policies(data.tau, data.t, data.y, nu, cost=0.0,
                             mu0=data.mu0, mu1=data.mu1)
    by = {r.name: r for r in rows}
    # With mostly-positive effects, treating everyone yields more outcome than no one.
    assert by["Treat everyone"].value > by["Treat no one"].value
    # The oracle rule's true value is at least treat-none's true value.
    assert by["Learned rule (CATE)"].true_value >= by["Treat no one"].true_value - 1e-9


def test_policy_risk_oracle_is_zero():
    tau = np.array([1.0, -0.5, 0.2, -1.0])
    oracle = (tau > 0).astype(int)
    assert policy_risk(oracle, tau) == pytest.approx(0.0)
    # Treating no one loses all the positive effects.
    assert policy_risk(np.zeros_like(tau), tau) > 0


def test_treatment_rule_respects_cost():
    tau = np.array([0.05, 0.5, -0.2, 0.15])
    r = treatment_rule(tau, cost=0.1)
    assert r.tolist() == [0, 1, 0, 1]
