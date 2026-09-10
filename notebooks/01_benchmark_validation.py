# %% [markdown]
# # Veritas — benchmark validation
# The claim "our estimator is accurate" is only meaningful against ground truth.
# Here we use a confounded DGP with a KNOWN treatment effect and show the naive
# estimate is biased while doubly-robust estimators recover the truth.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
from veritas.benchmarks import make_synthetic_confounded, pehe, ate_metrics
from veritas.estimate import crossfit_nuisances, naive_ate, ipw_ate, aipw_ate, tmle_ate, xlearner_cate

data = make_synthetic_confounded(n=4000, confounding=1.0, random_state=0)
print("true ATE:", round(data.true_ate, 3), "| naive diff:", round(data.naive_ate, 3))

# %% [markdown]
# ## ATE estimators vs the truth
# Naive is off because confounders inflate the treated group; IPW/AIPW/TMLE adjust.

# %%
nu = crossfit_nuisances(data.X, data.t, data.y, n_folds=5, random_state=0)
rows = []
for est in (naive_ate(data.t, data.y), ipw_ate(data.t, data.y, nu),
            aipw_ate(data.t, data.y, nu), tmle_ate(data.t, data.y, nu)):
    m = ate_metrics(est.ate, data.true_ate)
    rows.append({"estimator": est.name, "ATE": round(est.ate, 3),
                 "abs error": round(m["abs_error"], 3), "rel error": f"{m['rel_error']*100:.0f}%"})
pd.DataFrame(rows)

# %% [markdown]
# ## Heterogeneous effects (CATE) — PEHE vs ground truth

# %%
cate = xlearner_cate(data.X, data.t, data.y, random_state=0)
print("X-learner PEHE:", round(pehe(cate.tau, data.tau), 3))
print("constant-ATE PEHE:", round(pehe(np.full_like(data.tau, data.true_ate), data.tau), 3))

# %% [markdown]
# ## IHDP benchmark (real covariates, simulated ground truth)

# %%
from veritas.benchmarks import ihdp_replications
errs = {"Naive diff-in-means": [], "AIPW (doubly robust)": [], "TMLE (doubly robust)": []}
for d in ihdp_replications(10):
    nu = crossfit_nuisances(d.X, d.t, d.y, n_folds=5, random_state=0)
    errs["Naive diff-in-means"].append(abs(naive_ate(d.t, d.y).ate - d.true_ate))
    errs["AIPW (doubly robust)"].append(abs(aipw_ate(d.t, d.y, nu).ate - d.true_ate))
    errs["TMLE (doubly robust)"].append(abs(tmle_ate(d.t, d.y, nu).ate - d.true_ate))
pd.DataFrame({k: [round(np.mean(v), 3)] for k, v in errs.items()}, index=["mean |ATE error|"])
