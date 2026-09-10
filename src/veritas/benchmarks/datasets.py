"""Benchmark and case-study datasets with a single interface.

* ``make_synthetic_confounded`` — a confounded DGP with a *known* individual
  treatment effect. Treatment and outcome share confounders, so the naive
  difference-in-means is biased; because we know the truth, it is the oracle for
  the estimator-correctness tests.
* ``load_ihdp`` — the IHDP semi-synthetic benchmark (Hill 2011): real covariates,
  simulated outcomes with ground-truth potential outcomes mu0/mu1 (so true ITE
  and ATE are known). 100 replications.

Every loader returns a :class:`CausalData`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass
class CausalData:
    """An observational dataset, optionally with ground-truth effects."""

    X: pd.DataFrame
    t: np.ndarray                 # 0/1 treatment
    y: np.ndarray                 # observed outcome
    name: str
    feature_names: list
    # Ground truth (benchmarks only)
    tau: Optional[np.ndarray] = None   # individual treatment effect mu1 - mu0
    mu0: Optional[np.ndarray] = None
    mu1: Optional[np.ndarray] = None
    propensity_true: Optional[np.ndarray] = None
    outcome_type: str = "continuous"   # "continuous" or "binary"

    def __len__(self) -> int:
        return len(self.y)

    @property
    def true_ate(self) -> Optional[float]:
        if self.tau is not None:
            return float(np.mean(self.tau))
        return None

    @property
    def naive_ate(self) -> float:
        """Difference in means — biased under confounding."""
        return float(self.y[self.t == 1].mean() - self.y[self.t == 0].mean())


# --------------------------------------------------------------------------- #
# Synthetic confounded DGP (known ground truth)
# --------------------------------------------------------------------------- #
def make_synthetic_confounded(
    n: int = 2000,
    p: int = 8,
    confounding: float = 1.0,
    random_state: int = 42,
) -> CausalData:
    """Confounded observational data with a known heterogeneous effect.

    X0..X2 are confounders: they drive BOTH the propensity to be treated and the
    baseline outcome, so treated and control differ systematically. The naive
    difference in means therefore overstates the effect; a valid estimator must
    adjust for X to recover ``true_ate``.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, p))
    cols = [f"x{i}" for i in range(p)]

    # Propensity depends on confounders X0..X2 (strength scaled by `confounding`).
    ps_logit = confounding * (0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])
    e = 1.0 / (1.0 + np.exp(-ps_logit))
    e = np.clip(e, 0.05, 0.95)
    t = (rng.random(n) < e).astype(int)

    # Baseline outcome also depends on the same confounders -> confounding bias.
    mu0 = 2.0 + X[:, 0] + 0.5 * X[:, 1] ** 2 - 0.8 * X[:, 2] + 0.3 * X[:, 3]
    # Heterogeneous treatment effect (who benefits): larger for high x0, x3>0.
    tau = 1.0 + 0.6 * X[:, 0] + 0.5 * (X[:, 3] > 0).astype(float)
    mu1 = mu0 + tau

    noise = rng.normal(scale=1.0, size=n)
    y = np.where(t == 1, mu1, mu0) + noise

    return CausalData(
        X=pd.DataFrame(X, columns=cols), t=t, y=y,
        name="synthetic-confounded", feature_names=cols,
        tau=tau, mu0=mu0, mu1=mu1, propensity_true=e, outcome_type="continuous",
    )


# --------------------------------------------------------------------------- #
# IHDP benchmark
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def _load_ihdp_npz(cache_dir: Path):
    from veritas.config import IHDP_TRAIN_URL, DATA_DIR

    cache_dir = Path(cache_dir or DATA_DIR)
    path = cache_dir / "ihdp_npci_1-100.train.npz"
    if not path.exists():
        _download(IHDP_TRAIN_URL, path)
    return np.load(path)


def load_ihdp(replication: int = 0, cache_dir: Optional[Path] = None) -> CausalData:
    """Load one IHDP replication with ground-truth potential outcomes.

    The npz stores arrays shaped (n_units, n_replications). For replication ``i``
    we take column ``i`` and reconstruct the true ITE as mu1 - mu0.
    """
    from veritas.config import DATA_DIR

    d = _load_ihdp_npz(Path(cache_dir or DATA_DIR))
    i = replication
    X = d["x"][:, :, i]
    t = d["t"][:, i].astype(int)
    yf = d["yf"][:, i]
    mu0 = d["mu0"][:, i]
    mu1 = d["mu1"][:, i]
    cols = [f"x{j}" for j in range(X.shape[1])]
    return CausalData(
        X=pd.DataFrame(X, columns=cols), t=t, y=yf,
        name=f"IHDP[{i}]", feature_names=cols,
        tau=mu1 - mu0, mu0=mu0, mu1=mu1, outcome_type="continuous",
    )


def ihdp_replications(n: int = 20, cache_dir: Optional[Path] = None) -> Iterator[CausalData]:
    """Yield the first ``n`` IHDP replications (for the validation harness)."""
    from veritas.config import DATA_DIR

    d = _load_ihdp_npz(Path(cache_dir or DATA_DIR))
    total = d["t"].shape[1]
    for i in range(min(n, total)):
        X = d["x"][:, :, i]
        cols = [f"x{j}" for j in range(X.shape[1])]
        yield CausalData(
            X=pd.DataFrame(X, columns=cols), t=d["t"][:, i].astype(int),
            y=d["yf"][:, i], name=f"IHDP[{i}]", feature_names=cols,
            tau=d["mu1"][:, i] - d["mu0"][:, i], mu0=d["mu0"][:, i],
            mu1=d["mu1"][:, i], outcome_type="continuous",
        )
