# Fitted presets (held-out RPS, continuous optimizer)

> Every constant below was replaced by the value that minimises Ranked Probability
> Score on out-of-sample matches, instead of being hand-picked. The two squad knobs
> are fitted with `scipy.optimize.minimize_scalar` (bounds [0,1] = a weight's natural
> range) so there is no hand-set candidate menu at all - the data walks to the optimum.
> RPS is the football-standard ordered metric (lower = better).

## Live spine (scripts 09-12) - squad prior

Held-out test set: **464 internationals after the EA FC 26 snapshot (2025-09-19)**, both sides FC26-rated and with >=30 prior games.

| Knob | Old preset | Fitted | How |
|---|---|---|---|
| Best-XI weight (vs top-23 depth) | 0.70 | **0.58** | continuous optimizer (minimize_scalar) |
| Squad blend weight (talent vs form) | 0.35 | **0.54** | continuous optimizer (minimize_scalar) |

Form-Elo-only held-out RPS = 0.15897; best squad-blended RPS = 0.15488 (+2.57%).

**Honest reading:** the surface is flat - any blend in **0.40-0.65** scores within RPS noise (0.0003) of the best. Report as **≈0.5**; the exact decimal is not signal the 445-match sample can support. Production is locked at the optimizer's point.

### Blend sensitivity around the optimum (characterisation, not selection)

| Blend weight | Held-out RPS | |
|---|---|---|
| 0.20 | 0.15652 |  |
| 0.25 | 0.15608 |  |
| 0.30 | 0.15570 |  |
| 0.35 | 0.15539 |  |
| 0.40 | 0.15516 | flat |
| 0.45 | 0.15499 | flat |
| 0.50 | 0.15490 | flat |
| 0.55 | 0.15488 | flat |
| 0.60 | 0.15493 | flat |
| 0.65 | 0.15506 | flat |
| 0.70 | 0.15526 |  |
| 0.75 | 0.15553 |  |
| 0.80 | 0.15588 |  |

## Archived ratio baseline only (NOT the live forecast) - recency half-life

The live Elo spine has no half-life knob (Elo's recency is implicit in walk-forward
updating). This fit hardens only the archived ratio-Poisson baseline (nb02/03/05/06).

| Knob | Old preset | Fitted | How |
|---|---|---|---|
| Half-life (years) | 4 | **15** | grid search, held-out RPS (ratio baseline) |

| Half-life (yr) | Held-out RPS |
|---|---|
| 1 | 0.21438 |
| 2 | 0.20514 |
| 3 | 0.20064 |
| 4 | 0.19793 |
| 5 | 0.19614 |
| 6 | 0.19490 |
| 8 | 0.19339 |
| 10 | 0.19261 |
| 15 | 0.19199 |
| 20 | 0.19205 |
| 30 | 0.19257 |
| 50 | 0.19352 |
