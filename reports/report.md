# Veritas — results

*Doubly-robust causal inference validated on ground truth, then applied to real observational data.*

## Headline

- **Synthetic benchmark (true ATE = 1.23):** the naive difference-in-means is off by **45%**; AIPW recovers the truth within **5%** — a **8.4× reduction in error**.
- **IHDP (12 reps):** naive mean relative ATE error **4%** vs **3%** for TMLE (doubly robust); best CATE PEHE **2.04** (X-learner).
- **RHC case study:** naive says catheterization changes 30-day mortality by **+7.4 pp**; doubly-robust adjustment gives **+4.1 pp**, with an **E-value of 1.52** (strength a hidden confounder would need to overturn it).

![Synthetic benchmark](figures/bench_forest.png)

## Synthetic benchmark — estimator accuracy

| estimator            |   ATE |   |error| | rel error   |
|:---------------------|------:|----------:|:------------|
| Naive diff-in-means  | 1.785 |     0.557 | 45%         |
| IPW (stabilized)     | 1.151 |     0.078 | 6%          |
| AIPW (doubly robust) | 1.295 |     0.066 | 5%          |
| TMLE (doubly robust) | 1.307 |     0.078 | 6%          |
| DML (EconML)         | 1.283 |     0.054 | 4%          |


## IHDP benchmark (averaged over replications)

| estimator            |   mean |ATE err| | rel err   |   95% CI coverage |
|:---------------------|-----------------:|:----------|------------------:|
| Naive diff-in-means  |            0.23  | 4%        |          0.916667 |
| IPW (stabilized)     |            0.19  | 4%        |          1        |
| AIPW (doubly robust) |            0.256 | 5%        |          1        |
| TMLE (doubly robust) |            0.125 | 3%        |          1        |
| DML (EconML)         |            0.427 | 8%        |          0.666667 |

![IHDP error](figures/bench_ihdp.png)

## Case study — RHC (Connors 1996)

**Question:** Does right-heart catheterization (RHC) change 30-day mortality in critically ill ICU patients?

Naive 30-day mortality: RHC **38.0%** vs no-RHC **30.6%** (38% treated, n=5,735).

| estimator            |   risk diff (pp) | 95% CI      |
|:---------------------|-----------------:|:------------|
| Naive diff-in-means  |             7.36 | [4.8, 9.9]  |
| IPW (stabilized)     |             4.07 | [0.8, 7.3]  |
| AIPW (doubly robust) |             4.12 | [1.4, 6.8]  |
| TMLE (doubly robust) |             4    | [1.3, 6.7]  |
| DML (EconML)         |             2.76 | [-0.1, 5.6] |

**E-value:** 1.52 (point), 1.27 (CI limit).

![RHC forest](figures/rhc_forest.png)
![RHC overlap](figures/rhc_overlap.png)
![RHC CATE](figures/rhc_cate.png)
![RHC sensitivity](figures/rhc_sensitivity.png)

## DoWhy refutation tests

| refutation test     |   estimate |   refuted | passed   |
|:--------------------|-----------:|----------:|:---------|
| Placebo treatment   |     0.0584 |   -0.0005 | True     |
| Random common cause |     0.0584 |    0.0584 | True     |
| Data subset (80%)   |     0.0584 |    0.0587 | True     |



---
*Reproduce:* `python -m veritas.pipeline --reps 12`
