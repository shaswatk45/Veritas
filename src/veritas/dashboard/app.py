"""Veritas dashboard (Streamlit).

    streamlit run src/veritas/dashboard/app.py

Loads reports/results.json written by `python -m veritas.pipeline`. For the
polished, self-contained version see reports/veritas_dashboard.html
(`python -m veritas.export_frontend`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from veritas.config import REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="Veritas — RWE engine", layout="wide")


@st.cache_data
def load():
    p = REPORTS_DIR / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


results = load()
st.title("Veritas — recovering the truth from confounded data")
st.caption("Doubly-robust causal inference, validated on ground truth, applied to a real clinical question.")

if results is None:
    st.warning("No results yet. Run `python -m veritas.pipeline` first.")
    st.stop()

h = results["headline"]
B = results["benchmarks"]
C = results.get("casestudy")

c1, c2, c3, c4 = st.columns(4)
if h.get("naive_bias_x"):
    c1.metric("Naive bias (synthetic)", f"×{h['naive_bias_x']:.1f}", help="naive error vs doubly-robust")
c2.metric("AIPW accuracy", f"±{h.get('synthetic_aipw_relerr',0)*100:.0f}%")
if C:
    c3.metric("RHC E-value", f"{h.get('rhc_evalue',float('nan')):.2f}")
if h.get("ihdp_best_pehe") is not None:
    c4.metric("Best CATE PEHE (IHDP)", f"{h['ihdp_best_pehe']:.2f}", help=h.get("ihdp_best_pehe_model", ""))

st.divider()
st.subheader("Benchmark — synthetic (true ATE known)")
syn = B["synthetic"]
st.write(f"True ATE = **{syn['true_ate']:.3f}**, naive diff = **{syn['naive_ate']:.3f}**")
st.dataframe(pd.DataFrame([{"estimator": r["name"], "ATE": round(r["ate"], 3),
                            "|error|": round(r["abs_error"], 3),
                            "rel error": f"{r['rel_error']*100:.0f}%"} for r in syn["estimators"]],
                          ), hide_index=True, use_container_width=True)
st.image(str(REPORTS_DIR / "figures" / "bench_forest.png"))

if "estimators" in B.get("ihdp", {}):
    st.subheader(f"Benchmark — IHDP ({B['ihdp']['n_reps']} replications)")
    st.dataframe(pd.DataFrame([{"estimator": r["name"],
                                "mean |ATE err|": round(r["ate_abs_error_mean"], 3),
                                "rel err": f"{r['rel_error_mean']*100:.0f}%",
                                "CI coverage": r["coverage"]} for r in B["ihdp"]["estimators"]]),
                 hide_index=True, use_container_width=True)

if C:
    st.divider()
    st.subheader(f"Case study — {C['name']}")
    st.write(f"**{C['question']}**")
    st.write(f"Unadjusted mortality: RHC {C['mortality_treated']*100:.1f}% vs no-RHC {C['mortality_control']*100:.1f}%")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(REPORTS_DIR / "figures" / "rhc_forest.png"))
        st.metric("E-value", f"{C['evalue']['e_value_point']:.2f}",
                  help="confounder strength (RR) needed to overturn the effect")
    with col2:
        st.image(str(REPORTS_DIR / "figures" / "rhc_overlap.png"))
    st.image(str(REPORTS_DIR / "figures" / "rhc_cate.png"))
    st.image(str(REPORTS_DIR / "figures" / "rhc_sensitivity.png"))
    if C.get("refutation"):
        st.subheader("DoWhy refutation tests")
        st.dataframe(pd.DataFrame(C["refutation"]), hide_index=True, use_container_width=True)
    st.subheader("Policy — whom to treat")
    st.dataframe(pd.DataFrame(C["policy"]), hide_index=True, use_container_width=True)
