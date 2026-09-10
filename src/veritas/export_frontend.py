"""Build the self-contained HTML frontend from the pipeline's results.json.

Precomputes histogram bins for the overlap and CATE plots so the page embeds only
a few KB. Writes reports/frontend_data.json and inlines it into the template to
produce reports/veritas_dashboard.html (fully self-contained, no server).

    python -m veritas.export_frontend
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from veritas.config import REPORTS_DIR, REPO_ROOT


def _hist(values, bins=24, lo=None, hi=None):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"edges": [], "counts": []}
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def build_payload() -> dict:
    results = json.loads((REPORTS_DIR / "results.json").read_text())
    case = results.get("casestudy")

    payload = {
        "headline": results["headline"],
        "benchmarks": results["benchmarks"],
        "runtime_sec": results.get("runtime_sec"),
    }

    if case:
        prop = case.get("propensity", {})
        t_all = (prop.get("treated", []) + prop.get("control", []))
        lo = min(t_all) if t_all else 0.0
        hi = max(t_all) if t_all else 1.0
        overlap = {
            "treated": _hist(prop.get("treated", []), bins=24, lo=lo, hi=hi),
            "control": _hist(prop.get("control", []), bins=24, lo=lo, hi=hi),
        }
        cate_vals = case.get("cate_values", [])
        cate_hist = _hist(cate_vals, bins=30)
        slim = {k: v for k, v in case.items()
                if k not in ("propensity", "cate_values")}
        slim["overlap_hist"] = overlap
        slim["cate_hist"] = cate_hist
        payload["casestudy"] = slim

    return payload


def main():
    payload = build_payload()
    data_str = json.dumps(payload, separators=(",", ":"), default=float)
    (REPORTS_DIR / "frontend_data.json").write_text(data_str)
    print(f"wrote frontend_data.json ({len(data_str)/1024:.1f} KB)")

    template = REPO_ROOT / "frontend" / "template.html"
    if template.exists():
        html = template.read_text(encoding="utf-8").replace("__VERITAS_DATA__", data_str)
        page = REPORTS_DIR / "veritas_dashboard.html"
        page.write_text(html, encoding="utf-8")
        print(f"wrote {page} ({page.stat().st_size/1024:.1f} KB, self-contained)")

        # Also publish as the GitHub Pages site (docs/index.html).
        docs = REPO_ROOT / "docs"
        docs.mkdir(exist_ok=True)
        (docs / "index.html").write_text(html, encoding="utf-8")
        (docs / ".nojekyll").write_text("", encoding="utf-8")
        print(f"wrote {docs/'index.html'} (GitHub Pages site)")


if __name__ == "__main__":
    main()
