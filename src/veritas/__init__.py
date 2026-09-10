"""Veritas: a causal-inference / real-world-evidence engine.

Randomized trials are the gold standard, but most real decisions rely on
*observational* data where treated and untreated groups differ systematically
(confounding). Naive correlation gives the wrong answer — often the wrong *sign*.
Veritas recovers the true causal effect anyway: doubly-robust ATE estimation,
heterogeneous effects (CATE), sensitivity analysis for hidden bias, and a
whom-to-treat policy — all validated against ground-truth causal benchmarks.
"""

__version__ = "0.1.0"

from veritas.config import Config  # noqa: E402

__all__ = ["Config", "__version__"]
