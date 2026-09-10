"""Matplotlib figures for the report and dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, savepath):
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


def forest_plot(rows: List[dict], true_ate=None, title="ATE estimates", savepath=None):
    """Bias-vs-corrected: point estimate + 95% CI per estimator; truth as a line."""
    rows = [r for r in rows if np.isfinite(r.get("ate", np.nan))]
    names = [r["name"] for r in rows]
    est = [r["ate"] for r in rows]
    los = [r.get("ci_low", np.nan) for r in rows]
    his = [r.get("ci_high", np.nan) for r in rows]
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(names) + 1.6))
    for i, (e, lo, hi) in enumerate(zip(est, los, his)):
        yi = y[i]
        color = "#c1443c" if "aive" in names[i] else "#1f7a5a"
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, lw=2, alpha=.8)
        ax.plot(e, yi, "o", color=color, ms=9, zorder=3)
    if true_ate is not None:
        ax.axvline(true_ate, color="#2b2b2b", ls="--", lw=1.4, label=f"true ATE = {true_ate:.2f}")
        ax.legend(frameon=False, loc="lower right")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Average treatment effect")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.25)
    return _save(fig, savepath)


def overlap_plot(e, t, savepath=None):
    """Propensity overlap (positivity check): treated vs control score histograms."""
    e = np.asarray(e); t = np.asarray(t).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(e[t == 1], bins=bins, alpha=.6, color="#1f7a5a", label="treated", density=True)
    ax.hist(e[t == 0], bins=bins, alpha=.6, color="#c1443c", label="control", density=True)
    ax.set_xlabel("Estimated propensity  P(T=1 | X)")
    ax.set_ylabel("density")
    ax.set_title("Overlap / positivity")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    return _save(fig, savepath)


def cate_hist(tau, savepath=None):
    """Distribution of individual treatment effects."""
    tau = np.asarray(tau)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(tau, bins=40, color="#3a6ea5", alpha=.85)
    ax.axvline(0, color="#c1443c", lw=1.4, ls="--", label="no effect")
    ax.axvline(tau.mean(), color="#1f7a5a", lw=1.6, label=f"mean = {tau.mean():.3f}")
    ax.set_xlabel("Estimated individual treatment effect (CATE)")
    ax.set_ylabel("count")
    ax.set_title("Who benefits? Heterogeneous effects")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    return _save(fig, savepath)


def benchmark_bar(err_by_estimator: Dict[str, float], title="ATE error vs ground truth", savepath=None):
    """Absolute ATE error per estimator (lower = better); naive highlighted."""
    names = list(err_by_estimator.keys())
    vals = [err_by_estimator[k] for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#c1443c" if "aive" in n else "#1f7a5a" for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("|estimated − true ATE|")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.25)
    return _save(fig, savepath)


def sensitivity_plot(curve: List[dict], savepath=None):
    """ATE as an unmeasured confounder of increasing strength is subtracted."""
    g = [c["gamma"] for c in curve]
    a = [c["ate"] for c in curve]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(g, a, "-o", color="#1f7a5a")
    ax.axhline(0, color="#c1443c", ls="--", lw=1.3, label="null effect")
    ax.set_xlabel("Unmeasured-confounder strength  γ")
    ax.set_ylabel("Adjusted ATE")
    ax.set_title("Sensitivity to hidden confounding")
    ax.legend(frameon=False)
    ax.grid(alpha=.25)
    return _save(fig, savepath)
