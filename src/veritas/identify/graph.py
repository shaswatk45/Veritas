"""Explicit causal assumptions (a DAG) + identification + refutation, via DoWhy.

ZS-grade RWE is *defended*, not just computed: state the assumptions as a graph,
identify the estimand (backdoor adjustment), estimate it, then try to break it.

Refutation tests we run:
* **Placebo treatment** — replace treatment with noise; a valid effect collapses to ~0.
* **Random common cause** — add an independent covariate; the estimate should not move.
* **Data subset** — re-estimate on a random 80%; the estimate should be stable.

If DoWhy is not installed the pipeline skips this stage (``DOWHY_AVAILABLE``).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from dowhy import CausalModel  # noqa: F401
    DOWHY_AVAILABLE = True
    _DOWHY_ERR = None
except Exception as exc:  # pragma: no cover
    DOWHY_AVAILABLE = False
    _DOWHY_ERR = exc


def causal_dag_edges(feature_names, treatment="T", outcome="Y", max_conf=6):
    """Human-readable DAG edges: confounders -> T, confounders -> Y, T -> Y."""
    conf = list(feature_names)[:max_conf]
    edges = []
    for c in conf:
        edges.append((c, treatment))
        edges.append((c, outcome))
    edges.append((treatment, outcome))
    return edges


def _build_df(data):
    df = data.X.copy()
    df["T"] = np.asarray(data.t).astype(int)
    df["Y"] = np.asarray(data.y, dtype=float)
    return df


def build_causal_model(data):
    """Construct a DoWhy CausalModel with all covariates as common causes."""
    from dowhy import CausalModel

    df = _build_df(data)
    return CausalModel(
        data=df, treatment="T", outcome="Y",
        common_causes=list(data.feature_names),
    )


def identify_and_estimate(data, method_name="backdoor.linear_regression") -> Dict:
    """Identify the backdoor estimand and produce a DoWhy estimate."""
    model = build_causal_model(data)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified, method_name=method_name)
    return {
        "model": model,
        "identified": identified,
        "estimate": estimate,
        "estimand_type": str(getattr(identified, "estimand_type", "")),
        "ate": float(estimate.value),
    }


def run_refutations(data, ie: Dict, random_state: int = 42, n_sim: int = 20) -> List[Dict]:
    """Run placebo / random-common-cause / subset refuters and grade each.

    ``n_sim`` caps each refuter's simulations (default 20 — plenty for a pass/fail
    signal, and far faster than DoWhy's default of 100).
    """
    model, identified, estimate = ie["model"], ie["identified"], ie["estimate"]
    base = float(estimate.value)
    tests = []

    def grade(name, new_effect, expect_zero, tol_rel=0.25, tol_abs=None):
        new_effect = float(new_effect)
        if expect_zero:
            thresh = tol_abs if tol_abs is not None else max(0.15 * abs(base), 1e-6)
            passed = abs(new_effect) <= thresh
            expect = "≈ 0"
        else:
            passed = abs(new_effect - base) <= tol_rel * abs(base) + 1e-6
            expect = f"≈ {base:.3f}"
        return {"test": name, "estimate": base, "refuted_estimate": new_effect,
                "expected": expect, "passed": bool(passed)}

    try:
        r = model.refute_estimate(identified, estimate,
                                  method_name="placebo_treatment_refuter",
                                  placebo_type="permute", num_simulations=n_sim,
                                  random_seed=random_state)
        tests.append(grade("Placebo treatment", r.new_effect, expect_zero=True))
    except Exception as exc:
        tests.append({"test": "Placebo treatment", "error": repr(exc)[:80], "passed": None})

    try:
        r = model.refute_estimate(identified, estimate,
                                  method_name="random_common_cause",
                                  num_simulations=n_sim, random_seed=random_state)
        tests.append(grade("Random common cause", r.new_effect, expect_zero=False))
    except Exception as exc:
        tests.append({"test": "Random common cause", "error": repr(exc)[:80], "passed": None})

    try:
        r = model.refute_estimate(identified, estimate,
                                  method_name="data_subset_refuter",
                                  subset_fraction=0.8, num_simulations=n_sim,
                                  random_seed=random_state)
        tests.append(grade("Data subset (80%)", r.new_effect, expect_zero=False))
    except Exception as exc:
        tests.append({"test": "Data subset (80%)", "error": repr(exc)[:80], "passed": None})

    return tests
