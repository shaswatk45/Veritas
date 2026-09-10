"""Central configuration: paths, dataset URLs, and run defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# IHDP benchmark (Hill 2011), 100 semi-synthetic replications with ground-truth
# potential outcomes. Fredrik Johansson's widely-used mirror.
IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

# RHC (Right Heart Catheterization; Connors et al. 1996) — the textbook
# observational-confounding case study. Vanderbilt Biostatistics mirror.
RHC_URL = "https://hbiostat.org/data/repo/rhc.csv"


@dataclass
class Config:
    """Run-time knobs for a Veritas experiment."""

    random_state: int = 42
    n_folds: int = 5              # cross-fitting folds for doubly-robust estimators
    propensity_clip: float = 0.02  # trim propensities to [clip, 1-clip] (positivity)

    # Benchmark harness
    ihdp_replications: int = 20   # how many IHDP replications to average over
    synthetic_n: int = 2000       # rows for the synthetic confounded benchmark

    # Policy
    treat_cost: float = 0.0       # cost (in outcome units) charged per treatment

    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)
    reports_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)

    def ensure_dirs(self) -> None:
        for d in (self.figures_dir, self.reports_dir, self.data_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
