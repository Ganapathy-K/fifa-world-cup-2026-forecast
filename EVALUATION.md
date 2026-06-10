# FIFA WC 2026 - Baseline Model Evaluation

> Time-split: trained on 45,806 matches before 2023-01-01, graded on 3,475 held-out matches after (no leakage). Lower is better on all metrics. Actual test outcomes: home 47.0% / draw 23.0% / away 30.0%.

## Metrics: Poisson model vs dumb base-rate benchmark

| Metric | Poisson model | Base-rate benchmark | Improvement % |
|---|---|---|---|
| Brier | 0.5703 | 0.6371 | 10.5 |
| Log loss | 0.9625 | 1.0553 | 8.8 |
| RPS | 0.1978 | 0.2299 | 14.0 |

**Verdict:** the baseline beats a team-blind benchmark by 9-14% - it has learned real signal. RPS 0.198 is in the normal band for simple football models. Not elite; the room for improvement is what Elo / Dixon-Coles / a talent prior are for.

## Calibration (predicted home-win prob vs actual home-win rate)

| Bin matches | Mean predicted home | Actual home rate | Gap (actual - predicted) |
|---|---|---|---|
| 25 | 0.065 | 0.040 | -0.025 |
| 167 | 0.162 | 0.102 | -0.060 |
| 635 | 0.258 | 0.208 | -0.050 |
| 1100 | 0.352 | 0.405 | +0.053 |
| 940 | 0.445 | 0.580 | +0.134 |
| 436 | 0.540 | 0.768 | +0.229 |
| 132 | 0.640 | 0.909 | +0.269 |
| 28 | 0.735 | 0.929 | +0.193 |
| 11 | 0.844 | 1.000 | +0.156 |
| 1 | 0.907 | 1.000 | +0.093 |

**Reading it:** a positive gap means the model UNDER-predicted home wins - expected, because the model has no home-advantage term. This is the single clearest motivation for adding one.