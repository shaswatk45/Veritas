"""Correctness tests: on a confounded DGP with KNOWN effect, naive must be biased
and the doubly-robust estimators must recover the truth."""
import numpy as np
import pytest

from veritas.benchmarks import make_synthetic_confounded, pehe, ate_metrics
from veritas.estimate import (
    crossfit_nuisances, naive_ate, ipw_ate, aipw_ate, tmle_ate, xlearner_cate,
)


@pytest.fixture(scope="module")
def data():
    return make_synthetic_confounded(n=4000, confounding=1.0, random_state=0)


@pytest.fixture(scope="module")
def nu(data):
    return crossfit_nuisances(data.X, data.t, data.y, n_folds=5, random_state=0)


def test_confounding_makes_naive_biased(data):
    true = data.true_ate
    naive = naive_ate(data.t, data.y).ate
    # The naive estimate is materially off from the truth (positive bias here).
    assert abs(naive - true) > 0.3
    assert naive > true  # confounders inflate the naive comparison


def test_nuisances_in_range(data, nu):
    assert np.all((nu.e > 0) & (nu.e < 1))
    assert len(nu.mu0) == len(nu.mu1) == len(data)


def test_doubly_robust_recovers_truth(data, nu):
    true = data.true_ate
    for est in (ipw_ate(data.t, data.y, nu),
                aipw_ate(data.t, data.y, nu),
                tmle_ate(data.t, data.y, nu)):
        err = abs(est.ate - true)
        assert err < 0.25, f"{est.name} off by {err:.3f} (true={true:.3f})"
    # And they must beat naive.
    assert abs(aipw_ate(data.t, data.y, nu).ate - true) < abs(naive_ate(data.t, data.y).ate - true)


def test_aipw_ci_covers_truth(data, nu):
    est = aipw_ate(data.t, data.y, nu)
    assert est.ci_low <= data.true_ate <= est.ci_high


def test_xlearner_pehe_beats_constant(data):
    c = xlearner_cate(data.X, data.t, data.y, random_state=0)
    pehe_model = pehe(c.tau, data.tau)
    pehe_const = pehe(np.full_like(data.tau, data.true_ate), data.tau)
    # Modeling heterogeneity beats predicting the average effect for everyone.
    assert pehe_model < pehe_const


def test_ate_metrics():
    m = ate_metrics(1.2, 1.0)
    assert m["abs_error"] == pytest.approx(0.2)
    assert m["rel_error"] == pytest.approx(0.2)
    assert m["signed_error"] == pytest.approx(0.2)
