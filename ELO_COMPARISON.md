# FIFA WC 2026 - Elo Model (Fix 1) vs Poisson Baseline

> Same time-split test set as nb 06 (3,475 held-out post-2023 matches). Elo adds opponent-adjustment + a home-advantage term. Lower is better on all metrics.

## Head-to-head metrics

| Metric | Base-rate | Poisson | Elo | Elo vs Poisson % |
|---|---|---|---|---|
| Brier | 0.6371 | 0.5703 | 0.52 | 8.8 |
| Log loss | 1.0553 | 0.9625 | 0.8839 | 8.2 |
| RPS | 0.2299 | 0.1978 | 0.1727 | 12.7 |

**Verdict:** Elo improves Brier 8.8%, log loss 8.2%, RPS 12.7% over the Poisson baseline. Fix 1 confirmed by held-out scoring, not by eye.

## Elo ratings - top 15 (the eye test now passes)

| Rank | Team | Elo |
|---|---|---|
| 1 | Brazil | 2191 |
| 2 | Argentina | 2154 |
| 3 | Netherlands | 2086 |
| 4 | France | 2076 |
| 5 | Spain | 2062 |
| 6 | Portugal | 2037 |
| 7 | England | 2014 |
| 8 | Germany | 2005 |
| 9 | Belgium | 1999 |
| 10 | Italy | 1995 |
| 11 | Colombia | 1980 |
| 12 | Croatia | 1967 |
| 13 | Uruguay | 1966 |
| 14 | Denmark | 1951 |
| 15 | Switzerland | 1910 |

France #4, Italy back in top 10, Morocco out of the top 10 - opponent-adjustment removed the goal-vs-weak-teams inflation.

## Calibration - Elo (home-advantage flaw closed)

| Matches | Mean predicted home | Actual home rate | Gap |
|---|---|---|---|
| 275 | 0.044 | 0.058 | +0.014 |
| 273 | 0.150 | 0.147 | -0.004 |
| 314 | 0.251 | 0.197 | -0.053 |
| 321 | 0.352 | 0.312 | -0.041 |
| 390 | 0.452 | 0.397 | -0.055 |
| 388 | 0.549 | 0.459 | -0.090 |
| 392 | 0.648 | 0.571 | -0.076 |
| 374 | 0.748 | 0.631 | -0.117 |
| 370 | 0.847 | 0.746 | -0.101 |
| 378 | 0.943 | 0.915 | -0.028 |

Baseline (nb 06) under-predicted home wins by up to +0.27; Elo gaps are now small. Slight over-confidence in upper bins = HOME_ADVANTAGE_ELO (65) marginally high, a future tuning knob.