"""Real observational case study: Right Heart Catheterization (Connors et al. 1996).

The clinical question: **does inserting a right-heart (Swan-Ganz) catheter in
critically ill ICU patients change 30-day mortality?** Sicker patients are far
more likely to be catheterized, so the naive comparison is badly confounded — the
textbook RWE problem. 5,735 patients, ~50 baseline covariates.

This mirrors ZS pharma RWE work exactly: does the intervention help in the real
population, and for whom — recovered from observational data, not a trial.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from veritas.benchmarks.datasets import CausalData, _download

RHC_QUESTION = ("Does right-heart catheterization (RHC) change 30-day mortality "
                "in critically ill ICU patients?")

# Columns that are outcomes, treatment, identifiers, dates, or post-treatment —
# never valid confounders to adjust for.
_DROP = {
    "", "Unnamed: 0", "ptid", "swang1", "death", "dth30", "t3d30",
    "sadmdte", "dschdte", "dthdte", "lstctdte", "surv2md1", "cat2",
}


def load_rhc(cache_dir: Optional[Path] = None) -> CausalData:
    """Load and preprocess the RHC dataset into a binary-outcome CausalData."""
    from veritas.config import RHC_URL, DATA_DIR

    cache_dir = Path(cache_dir or DATA_DIR)
    path = cache_dir / "rhc.csv"
    if not path.exists():
        _download(RHC_URL, path)
    df = pd.read_csv(path)

    # Treatment: RHC within 24h of admission.
    t = (df["swang1"].astype(str).str.strip() == "RHC").astype(int).to_numpy()
    # Outcome: death within 30 days (1 = died).
    if "dth30" in df.columns:
        y = (df["dth30"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"])).astype(int).to_numpy()
    else:
        y = (df["death"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()

    covars = df.drop(columns=[c for c in df.columns if c in _DROP], errors="ignore")
    # One-hot encode categoricals; median-fill numeric gaps.
    cat_cols = covars.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(covars, columns=cat_cols, dummy_na=False, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    # Drop any all-NaN / constant columns that survived.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.astype(float).reset_index(drop=True)

    return CausalData(
        X=X, t=t, y=y, name="RHC (Connors 1996)",
        feature_names=list(X.columns), outcome_type="binary",
    )
