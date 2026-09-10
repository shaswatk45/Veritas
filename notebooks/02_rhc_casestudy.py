# %% [markdown]
# # Veritas — RHC case study
# Does right-heart catheterization (RHC) change 30-day mortality in ICU patients?
# Sicker patients are catheterized, so the naive comparison is confounded.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
from veritas.casestudy import load_rhc
from veritas.estimate import crossfit_nuisances, all_ate_estimators, xlearner_cate
from veritas.sensitivity import e_value

data = load_rhc()
p0 = data.y[data.t == 0].mean(); p1 = data.y[data.t == 1].mean()
print(f"n={len(data)}  treated={data.t.mean():.1%}  mortality RHC={p1:.3f} vs no-RHC={p0:.3f}")

# %% [markdown]
# ## Naive vs adjusted 30-day mortality effect

# %%
rows = [{"estimator": r.name, "risk diff (pp)": round(r.ate * 100, 2),
         "95% CI (pp)": (f"[{r.ci_low*100:.1f}, {r.ci_high*100:.1f}]" if np.isfinite(r.ci_low) else "—")}
        for r in all_ate_estimators(data)]
pd.DataFrame(rows)

# %% [markdown]
# ## E-value — how strong must a hidden confounder be to overturn it?

# %%
from veritas.estimate import aipw_ate
nu = crossfit_nuisances(data.X, data.t, data.y, n_folds=5, binary_outcome=True)
ate = aipw_ate(data.t, data.y, nu).ate
rr = (p0 + ate) / p0
print("adjusted risk difference:", round(ate * 100, 2), "pp")
print("risk ratio:", round(rr, 3), "| E-value:", round(e_value(rr), 2))

# %% [markdown]
# ## Who is most affected? (CATE quartiles)

# %%
tau = xlearner_cate(data.X, data.t, data.y).tau
q = np.digitize(tau, np.quantile(tau, [.25, .5, .75]))
pd.DataFrame({"quartile": ["Q1", "Q2", "Q3", "Q4"],
              "mean effect (pp)": [round(tau[q == k].mean() * 100, 2) for k in range(4)]})
