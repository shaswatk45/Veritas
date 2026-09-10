"""End-to-end Veritas run: validate on ground-truth benchmarks, then apply the
whole causal pipeline to a real observational case study.

    python -m veritas.pipeline                 # benchmarks + RHC case study
    python -m veritas.pipeline --reps 20       # more IHDP replications
    python -m veritas.pipeline --no-casestudy  # benchmarks only (offline-friendly)

Writes results.json, a markdown report, and figures into reports/.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from veritas.config import Config
from veritas.benchmarks import (
    make_synthetic_confounded, ihdp_replications, pehe, ate_metrics,
)
from veritas.estimate import (
    all_ate_estimators, crossfit_nuisances, aipw_ate,
    xlearner_cate, causal_forest_cate,
)
from veritas.policy import dr_policy_value, treatment_rule
from veritas.sensitivity import e_value, simulate_unmeasured_confounder
from veritas import plots

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Benchmark validation
# --------------------------------------------------------------------------- #
def _bench_one(data, cfg):
    ate_rows = []
    for r in all_ate_estimators(data, n_folds=cfg.n_folds, clip=cfg.propensity_clip,
                                random_state=cfg.random_state):
        m = ate_metrics(r.ate, data.true_ate) if np.isfinite(r.ate) else None
        ate_rows.append({"name": r.name, "ate": r.ate, "se": r.se,
                         "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "abs_error": m["abs_error"] if m else float("nan"),
                         "rel_error": m["rel_error"] if m else float("nan")})
    cate_rows = []
    for fn in (xlearner_cate, causal_forest_cate):
        try:
            c = fn(data.X, data.t, data.y, random_state=cfg.random_state) \
                if fn is xlearner_cate else \
                fn(data.X, data.t, data.y, n_folds=cfg.n_folds, random_state=cfg.random_state)
            cate_rows.append({"name": c.name, "pehe": pehe(c.tau, data.tau),
                              "ate_error": abs(c.ate - data.true_ate)})
        except Exception as exc:
            cate_rows.append({"name": getattr(fn, "__name__", "cate"),
                              "pehe": float("nan"), "error": repr(exc)[:60]})
    return ate_rows, cate_rows


def run_benchmarks(cfg: Config) -> dict:
    print("[1/3] Benchmark validation (ground truth known)")
    out = {}

    # --- synthetic confounded ---
    syn = make_synthetic_confounded(n=cfg.synthetic_n, random_state=cfg.random_state)
    ate_rows, cate_rows = _bench_one(syn, cfg)
    out["synthetic"] = {
        "name": "Synthetic confounded", "true_ate": syn.true_ate,
        "naive_ate": syn.naive_ate, "n": len(syn),
        "estimators": ate_rows, "cate": cate_rows,
    }
    print(f"      synthetic: true ATE={syn.true_ate:.3f}, naive={syn.naive_ate:.3f}")
    for r in ate_rows:
        print(f"        {r['name']:<26} ATE={r['ate']:.3f}  |err|={r['abs_error']:.3f}")

    # --- IHDP replications ---
    try:
        ate_err = defaultdict(list); ate_rel = defaultdict(list); ate_val = defaultdict(list)
        cate_pehe = defaultdict(list)
        cover = defaultdict(list)
        reps = list(ihdp_replications(cfg.ihdp_replications, cache_dir=cfg.data_dir))
        for k, data in enumerate(reps):
            arows, crows = _bench_one(data, cfg)
            for r in arows:
                if np.isfinite(r["abs_error"]):
                    ate_err[r["name"]].append(r["abs_error"])
                    ate_rel[r["name"]].append(r["rel_error"])
                    ate_val[r["name"]].append(r["ate"])
                    if np.isfinite(r["ci_low"]):
                        cover[r["name"]].append(int(r["ci_low"] <= data.true_ate <= r["ci_high"]))
            for c in crows:
                if np.isfinite(c.get("pehe", np.nan)):
                    cate_pehe[c["name"]].append(c["pehe"])
            print(f"      IHDP rep {k+1}/{len(reps)} done", end="\r")
        print()
        est_rows = [{"name": n, "ate_abs_error_mean": float(np.mean(v)),
                     "ate_abs_error_std": float(np.std(v)),
                     "rel_error_mean": float(np.mean(ate_rel[n])),
                     "coverage": float(np.mean(cover[n])) if cover[n] else None}
                    for n, v in ate_err.items()]
        cate_rows = [{"name": n, "pehe_mean": float(np.mean(v)),
                      "pehe_std": float(np.std(v))} for n, v in cate_pehe.items()]
        out["ihdp"] = {"name": "IHDP", "n_reps": len(reps),
                       "true_ate_mean": float(np.mean([d.true_ate for d in reps])),
                       "estimators": est_rows, "cate": cate_rows}
        for r in est_rows:
            print(f"        {r['name']:<26} mean|err|={r['ate_abs_error_mean']:.3f} "
                  f"rel={r['rel_error_mean']*100:.1f}% cov={r['coverage']}")
    except Exception as exc:
        print(f"      IHDP skipped: {exc!r}")
        out["ihdp"] = {"error": repr(exc)[:120]}

    return out


# --------------------------------------------------------------------------- #
# Case study: RHC
# --------------------------------------------------------------------------- #
def run_casestudy(cfg: Config) -> dict:
    print("[2/3] Case study — Right Heart Catheterization (observational)")
    from veritas.casestudy import load_rhc, RHC_QUESTION

    data = load_rhc(cache_dir=cfg.data_dir)
    n = len(data)
    p0 = float(data.y[data.t == 0].mean())
    p1 = float(data.y[data.t == 1].mean())
    print(f"      n={n:,}  treated={data.t.mean():.1%}  "
          f"30d mortality: RHC={p1:.3f} vs no-RHC={p0:.3f}")

    nu = crossfit_nuisances(data.X, data.t, data.y, n_folds=cfg.n_folds,
                            clip=cfg.propensity_clip, random_state=cfg.random_state,
                            binary_outcome=True)
    ate_results = all_ate_estimators(data, n_folds=cfg.n_folds, clip=cfg.propensity_clip,
                                     random_state=cfg.random_state)
    ate_rows = [{"name": r.name, "ate": r.ate, "se": r.se,
                 "ci_low": r.ci_low, "ci_high": r.ci_high, "note": r.note}
                for r in ate_results]
    for r in ate_results:
        print(f"        {r.name:<26} risk diff={r.ate:+.4f}")

    # --- adjusted effect (AIPW) → E-value on the risk-ratio scale ---
    aipw = next(r for r in ate_results if r.name.startswith("AIPW"))
    rr_point = (p0 + aipw.ate) / p0 if p0 > 0 else float("nan")
    lo_rd, hi_rd = aipw.ci_low, aipw.ci_high
    crosses = lo_rd <= 0 <= hi_rd
    rr_near = (p0 + (lo_rd if aipw.ate > 0 else hi_rd)) / p0
    ev_point = e_value(rr_point)
    ev_ci = 1.0 if crosses else e_value(rr_near)
    evalue = {"risk_ratio": rr_point, "e_value_point": ev_point,
              "e_value_ci": ev_ci, "adjusted_risk_diff": aipw.ate,
              "ci": [lo_rd, hi_rd], "crosses_null": bool(crosses)}
    print(f"        E-value (point)={ev_point:.2f}, RR={rr_point:.3f}")

    # --- CATE + subgroups ---
    cate = xlearner_cate(data.X, data.t, data.y, random_state=cfg.random_state)
    tau = cate.tau
    q = np.quantile(tau, [0.25, 0.5, 0.75])
    quart = np.digitize(tau, q)
    subgroups = []
    agecol = next((c for c in data.X.columns if c.lower() == "age"), None)
    apscol = next((c for c in data.X.columns if c.lower() in ("aps1", "apache")), None)
    for k in range(4):
        m = quart == k
        row = {"quartile": ["Q1 (most helped)", "Q2", "Q3", "Q4 (most harmed)"][k],
               "n": int(m.sum()), "mean_cate": float(tau[m].mean()),
               "mortality_rhc": float(data.y[m & (data.t == 1)].mean()) if (m & (data.t == 1)).any() else float("nan"),
               "mortality_norhc": float(data.y[m & (data.t == 0)].mean()) if (m & (data.t == 0)).any() else float("nan")}
        if agecol:
            row["mean_age"] = float(data.X[agecol].to_numpy()[m].mean())
        if apscol:
            row["mean_apache"] = float(data.X[apscol].to_numpy()[m].mean())
        subgroups.append(row)

    # --- policy: whom to catheterize to MINIMIZE mortality (lower outcome better) ---
    beneficial = treatment_rule(-tau, cost=cfg.treat_cost)  # treat where RHC reduces mortality
    policies = {"Learned rule (treat only if RHC helps)": beneficial,
                "Catheterize everyone": np.ones(n, int),
                "Catheterize no one": np.zeros(n, int)}
    policy_rows = []
    for name, pi in policies.items():
        val, se = dr_policy_value(pi, data.t, data.y, nu)
        policy_rows.append({"policy": name, "treated_%": round(100*pi.mean(), 1),
                            "expected_mortality": round(val, 4), "se": round(se, 4)})
        print(f"        policy '{name[:34]:<34}' mortality={val:.4f} ({pi.mean():.0%} treated)")

    # --- sensitivity curve (AIPW re-estimated as hidden confounder grows) ---
    def _aipw_estimator(X, t, y):
        # The sensitivity sim residualizes the binary outcome into a continuous
        # one, so the outcome nuisance must be a regressor (binary_outcome=False).
        nn = crossfit_nuisances(X, t, y, n_folds=cfg.n_folds, clip=cfg.propensity_clip,
                                random_state=cfg.random_state, binary_outcome=False)
        return aipw_ate(t, y, nn).ate
    print("        running sensitivity simulation...")
    sens = simulate_unmeasured_confounder(data.X, data.t, data.y, _aipw_estimator,
                                          random_state=cfg.random_state)

    # --- DoWhy identification + refutation ---
    refutation = []
    dag_edges = []
    try:
        from veritas.identify import identify_and_estimate, run_refutations, causal_dag_edges, DOWHY_AVAILABLE
        dag_edges = [[a, b] for a, b in causal_dag_edges(data.feature_names)]
        if DOWHY_AVAILABLE:
            print("        running DoWhy refutation tests...")
            ie = identify_and_estimate(data)
            refutation = run_refutations(data, ie, random_state=cfg.random_state)
            for t_ in refutation:
                if "passed" in t_:
                    print(f"          {t_['test']:<22} {'PASS' if t_['passed'] else 'FAIL'}")
    except Exception as exc:
        print(f"        DoWhy skipped: {exc!r}")

    # figures
    plots.forest_plot([{"name": r["name"], "ate": r["ate"], "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"]} for r in ate_rows],
                      title="RHC: naive vs adjusted 30-day mortality effect",
                      savepath=cfg.figures_dir / "rhc_forest.png")
    plots.overlap_plot(nu.e, data.t, savepath=cfg.figures_dir / "rhc_overlap.png")
    plots.cate_hist(tau, savepath=cfg.figures_dir / "rhc_cate.png")
    plots.sensitivity_plot(aipw.ate, e_value=ev_point,
                           savepath=cfg.figures_dir / "rhc_sensitivity.png")

    return {
        "name": data.name, "question": RHC_QUESTION, "n": n,
        "treated_frac": float(data.t.mean()),
        "mortality_treated": p1, "mortality_control": p0,
        "ate": ate_rows, "evalue": evalue, "subgroups": subgroups,
        "policy": policy_rows, "sensitivity": sens,
        "refutation": refutation, "dag_edges": dag_edges,
        "cate_values": [round(float(v), 4) for v in tau[:5000]],
        "cate_mean": float(tau.mean()),
        "propensity": {"treated": [round(float(v), 4) for v in nu.e[data.t == 1][:5000]],
                       "control": [round(float(v), 4) for v in nu.e[data.t == 0][:5000]]},
    }


# --------------------------------------------------------------------------- #
def run(cfg: Config, do_casestudy: bool = True) -> dict:
    cfg.ensure_dirs()
    t0 = time.time()
    bench = run_benchmarks(cfg)

    # benchmark figures (synthetic forest + error bar)
    syn = bench["synthetic"]
    plots.forest_plot([{"name": r["name"], "ate": r["ate"], "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"]} for r in syn["estimators"]],
                      true_ate=syn["true_ate"],
                      title="Synthetic benchmark: recovering the true ATE",
                      savepath=cfg.figures_dir / "bench_forest.png")
    if "estimators" in bench.get("ihdp", {}):
        err = {r["name"]: r["ate_abs_error_mean"] for r in bench["ihdp"]["estimators"]}
        plots.benchmark_bar(err, title="IHDP: mean |ATE error| by estimator",
                            savepath=cfg.figures_dir / "bench_ihdp.png")

    case = run_casestudy(cfg) if do_casestudy else None

    print("[3/3] Writing report + figures")
    headline = _headline(bench, case)
    results = {"headline": headline, "benchmarks": bench, "casestudy": case,
               "runtime_sec": round(time.time() - t0, 1)}
    (cfg.reports_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    _write_markdown(cfg, results)
    print(f"\nDONE in {results['runtime_sec']}s. See reports/report.md")
    return results


def _headline(bench, case):
    h = {}
    syn = bench["synthetic"]
    naive = next(r for r in syn["estimators"] if "aive" in r["name"])
    aipw = next(r for r in syn["estimators"] if r["name"].startswith("AIPW"))
    h["synthetic_true_ate"] = syn["true_ate"]
    h["synthetic_naive_relerr"] = naive["rel_error"]
    h["synthetic_aipw_relerr"] = aipw["rel_error"]
    h["naive_bias_x"] = (naive["abs_error"] / aipw["abs_error"]) if aipw["abs_error"] > 1e-9 else None
    if "estimators" in bench.get("ihdp", {}):
        ih = bench["ihdp"]["estimators"]
        h["ihdp_naive_relerr"] = next(r["rel_error_mean"] for r in ih if "aive" in r["name"])
        best_dr = min([r for r in ih if r["name"].startswith(("AIPW", "TMLE"))],
                      key=lambda r: r["ate_abs_error_mean"])
        h["ihdp_best_dr"] = best_dr["name"]
        h["ihdp_best_dr_relerr"] = best_dr["rel_error_mean"]
        cate = bench["ihdp"].get("cate", [])
        if cate:
            best_cate = min(cate, key=lambda r: r["pehe_mean"])
            h["ihdp_best_pehe"] = best_cate["pehe_mean"]
            h["ihdp_best_pehe_model"] = best_cate["name"]
    if case:
        naive_c = next(r for r in case["ate"] if "aive" in r["name"])
        aipw_c = next(r for r in case["ate"] if r["name"].startswith("AIPW"))
        h["rhc_naive_rd"] = naive_c["ate"]
        h["rhc_adjusted_rd"] = aipw_c["ate"]
        h["rhc_evalue"] = case["evalue"]["e_value_point"]
    return h


def _write_markdown(cfg, results):
    h = results["headline"]
    bench = results["benchmarks"]; case = results["casestudy"]
    syn = pd.DataFrame([{"estimator": r["name"], "ATE": round(r["ate"], 3),
                         "|error|": round(r["abs_error"], 3),
                         "rel error": f"{r['rel_error']*100:.0f}%"}
                        for r in bench["synthetic"]["estimators"]])
    lines = [f"# Veritas — results\n",
             f"*Doubly-robust causal inference validated on ground truth, then applied to real observational data.*\n",
             "## Headline\n",
             f"- **Synthetic benchmark (true ATE = {h['synthetic_true_ate']:.2f}):** the naive "
             f"difference-in-means is off by **{h['synthetic_naive_relerr']*100:.0f}%**; "
             f"AIPW recovers the truth within **{h['synthetic_aipw_relerr']*100:.0f}%** "
             f"— a **{h['naive_bias_x']:.1f}× reduction in error**." if h.get("naive_bias_x") else "",
             ]
    if "ihdp_best_dr" in h:
        lines.append(f"- **IHDP ({bench['ihdp']['n_reps']} reps):** naive mean relative ATE error "
                     f"**{h['ihdp_naive_relerr']*100:.0f}%** vs **{h['ihdp_best_dr_relerr']*100:.0f}%** "
                     f"for {h['ihdp_best_dr']}; best CATE PEHE **{h.get('ihdp_best_pehe',float('nan')):.2f}** "
                     f"({h.get('ihdp_best_pehe_model','')}).")
    if case:
        lines.append(f"- **RHC case study:** naive says catheterization changes 30-day mortality by "
                     f"**{h['rhc_naive_rd']*100:+.1f} pp**; doubly-robust adjustment gives "
                     f"**{h['rhc_adjusted_rd']*100:+.1f} pp**, with an **E-value of {h['rhc_evalue']:.2f}** "
                     f"(strength a hidden confounder would need to overturn it).")
    lines += ["\n![Synthetic benchmark](figures/bench_forest.png)\n",
              "## Synthetic benchmark — estimator accuracy\n", syn.to_markdown(index=False), "\n"]
    if "estimators" in bench.get("ihdp", {}):
        ih = pd.DataFrame([{"estimator": r["name"],
                            "mean |ATE err|": round(r["ate_abs_error_mean"], 3),
                            "rel err": f"{r['rel_error_mean']*100:.0f}%",
                            "95% CI coverage": r["coverage"]}
                           for r in bench["ihdp"]["estimators"]])
        lines += ["## IHDP benchmark (averaged over replications)\n", ih.to_markdown(index=False),
                  "\n![IHDP error](figures/bench_ihdp.png)\n"]
    if case:
        ac = pd.DataFrame([{"estimator": r["name"], "risk diff (pp)": round(r["ate"]*100, 2),
                            "95% CI": f"[{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}]"
                            if np.isfinite(r["ci_low"]) else "—"} for r in case["ate"]])
        lines += [f"## Case study — {case['name']}\n", f"**Question:** {case['question']}\n",
                  f"Naive 30-day mortality: RHC **{case['mortality_treated']*100:.1f}%** vs "
                  f"no-RHC **{case['mortality_control']*100:.1f}%** ({case['treated_frac']*100:.0f}% treated, n={case['n']:,}).\n",
                  ac.to_markdown(index=False),
                  f"\n**E-value:** {case['evalue']['e_value_point']:.2f} (point), "
                  f"{case['evalue']['e_value_ci']:.2f} (CI limit).\n",
                  "![RHC forest](figures/rhc_forest.png)",
                  "![RHC overlap](figures/rhc_overlap.png)",
                  "![RHC CATE](figures/rhc_cate.png)",
                  "![RHC sensitivity](figures/rhc_sensitivity.png)\n"]
        if case["refutation"]:
            rf = pd.DataFrame([{"refutation test": t["test"],
                                "estimate": round(t.get("estimate", float('nan')), 4),
                                "refuted": round(t.get("refuted_estimate", float('nan')), 4),
                                "passed": t.get("passed")} for t in case["refutation"]])
            lines += ["## DoWhy refutation tests\n", rf.to_markdown(index=False), "\n"]
    lines.append(f"\n---\n*Reproduce:* `python -m veritas.pipeline --reps {cfg.ihdp_replications}`\n")
    (cfg.reports_dir / "report.md").write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")


def _parse_args() -> tuple:
    p = argparse.ArgumentParser(description="Run the Veritas causal-inference pipeline.")
    p.add_argument("--reps", type=int, default=15, help="IHDP replications")
    p.add_argument("--synthetic-n", type=int, default=2000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--no-casestudy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(random_state=a.seed, n_folds=a.folds, ihdp_replications=a.reps,
                  synthetic_n=a.synthetic_n), (not a.no_casestudy)


if __name__ == "__main__":
    cfg, do_case = _parse_args()
    run(cfg, do_casestudy=do_case)
