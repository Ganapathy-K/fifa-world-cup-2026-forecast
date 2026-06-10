# FIFA World Cup 2026 — Champion Predictor

A probabilistic forecast of the 2026 World Cup (48 teams, 104 matches), built from
~49,000 international results since 1872. It rates every team, simulates the whole
tournament 10,000 times, and reports each team's odds of reaching each round.

**Current headline forecast** (10,000 simulations, frozen at kickoff):

| Team | Champion % | Reach Final % | Reach Semi % |
|---|---|---|---|
| Spain | 20.2 | 32.0 | 48.4 |
| France | 17.5 | 28.4 | 45.4 |
| Argentina | 13.7 | 24.6 | 39.5 |
| Brazil | 11.2 | 20.7 | 34.8 |
| England | 9.4 | 18.2 | 32.4 |
| Portugal | 7.0 | 14.5 | 27.8 |
| Germany | 5.9 | 12.3 | 24.6 |

Full table: [CHAMPION_ODDS_FINAL.md](CHAMPION_ODDS_FINAL.md).

---

## The idea in two sentences

Two ideas plus one loop. **Elo** gives every team a single strength number that goes
up when you beat strong teams and down when you lose to weak ones. **Poisson** turns
the strength gap between two teams into a full distribution of possible scorelines
(not one prediction — the whole spread). Then a **Monte Carlo loop** plays the entire
bracket 10,000 times using those scoreline probabilities and counts how often each
team lifts the trophy.

Why simulate instead of just picking winners? Because the single most-likely bracket
has a near-zero chance of being *exactly* right, yet the favourite still wins ~20% of
the time — there are thousands of routes to the title, not the one everyone draws.
The forecast captures all of them; a single bracket captures one.

---

## How it's built (and how it got there)

The interesting part of this project isn't the final number — it's the trail of
**flaws found and fixed**. Each stage was scored on held-out matches before it was
kept, so "this is better" means measurably better, not better by eye.

| Stage | What it does | Why it was needed |
|---|---|---|
| **Baseline** | Recency-weighted attack/defence ratios → Poisson | First shippable model. Rated *Morocco #1* — a tell that something was wrong. |
| **Fix 1 — Elo** | Opponent-adjusted ratings + home-advantage term | The ratio model rewarded scoring against weak opponents. Elo fixed it: **RPS −12.7%** on held-out games; Morocco dropped out of the top 10, France/Spain rose. |
| **Fix 2 — Confederation calibration** | Per-confederation Elo offsets from 2,100+ inter-confederation matches | Raw Elo can't compare a UEFA team to an AFC team fairly (they rarely play). Offsets calibrate across confederations. |
| **Fix 3 — Squad-strength prior** | EA Sports FC 26 ratings (best-XI weighted), blended with form | Pure match form under-rates teams with elite talent and a thin recent schedule. The talent prior is an independent check. |
| **Fix 4 — Neutral-venue correction** | Home advantage applied *only* to host nations | A bug applied home advantage to whoever was nominally "home" in each fixture — making **France** wrongly favourite. WC games are neutral; fixing it put **Spain** (the top-rated team) correctly #1. |
| **Dixon–Coles** | Low-score draw correction on the scoreline grid | Plain Poisson slightly under-counts 0-0/1-1 draws. The standard football fix. |
| **Annex C** | Exact FIFA third-place qualification table (495 rows) | The 8 best third-placed teams advance via a specific bracket allocation — modelled exactly, not approximated. |

Two more fixes worth flagging because they're the kind of thing that quietly corrupts
results:

- **Reproducibility bug:** the third-place allocation iterated a Python `set`, whose
  order is randomised per process — so champion % drifted ~0.3pp run-to-run *despite a
  fixed seed*. Sorting the options made consecutive runs bit-identical.
- **Data-quality / name-map bug:** EA FC 26 names some nations differently from the
  results dataset (*Czechia* → Czech Republic, *Cabo Verde* → Cape Verde, *Türkiye* →
  Turkey). The mismatch silently dropped their talent prior. An audit of all 48 teams
  caught it; all 48 now resolve.

---

## Fit, don't preset

Every tunable constant is set to the value that **minimises [Ranked Probability
Score](https://en.wikipedia.org/wiki/Probabilistic_forecasting) on out-of-sample
matches**, not picked by hand. The two squad-blend knobs are fitted with a continuous
optimiser (`scipy.optimize.minimize_scalar`, bounds `[0,1]` = a weight's natural range),
so there's no hand-set candidate menu at all — the data walks to the optimum.

**Honest reading of the fit:** the RPS surface is *flat*. Any talent-blend weight in
`0.40–0.65` scores within noise of the best, on a 464-match test set. So the defensible
claim is "**talent weight ≈ 0.5**" — the exact decimal isn't signal this sample can
support. Knowing the difference is the point. See [FITTED_PRESETS.md](FITTED_PRESETS.md).

---

## How good is it, honestly?

Scored on a clean time-split (train < 2023, test ≥ 2023, no leakage):

| Metric | Base-rate | Poisson baseline | + Elo | 
|---|---|---|---|
| RPS (lower better) | 0.230 | 0.198 | **0.173** |
| Brier | 0.637 | 0.570 | **0.520** |
| Log loss | 1.055 | 0.963 | **0.884** |

It beats a team-blind benchmark by 9–14% → it has learned real signal. RPS ~0.17 is a
**solid amateur model, not Vegas-grade.** Full evaluation incl. a calibration curve:
[EVALUATION.md](EVALUATION.md), [ELO_COMPARISON.md](ELO_COMPARISON.md).

**What to trust, and what not to:**

- ✅ **Tiers and direction** — Spain/France/Argentina/Brazil on top, the minnows at the
  bottom, and "X is more likely than Y" calls are real signal.
- ⚠️ **Near-ties are noise** — any gap under ~10–15 Elo (the table still prints a
  confident order it can't justify).
- ⚠️ **Don't read the decimals** — "France 17.5%" really means *roughly 15–20%*.
- ⚠️ **It's a snapshot of average strength** — no injuries, squad news, momentum, or
  match-day reality.

A real bug (the home-advantage leak that wrongly made France favourite) was found
*during* this build — so the right description is "good and improving," not "finished."

---

## Repository layout

```
fifa_wc_2026_poisson/
├── match_engine.py              # core goal model: Elo→expected goals→Poisson grid→W/D/L, + bracket logic
├── 09_confederation_adjustment.py
├── 10_squad_strength_blend.py   # FC26 talent prior + the 10k Monte Carlo (live forecast)
├── 11_example_tournament.py     # replays ONE simulated tournament (a single 1-of-10k run)
├── 12_most_likely_bracket.py    # the deterministic "chalk" bracket (for curiosity, not the forecast)
├── 13_fit_presets.py            # fits every preset by held-out RPS (fit, don't preset)
├── build_annex_c_table.py       # scrapes FIFA's exact third-place allocation table
├── annex_c_third_allocation.csv # the 495-row Annex C reference data
├── notebooks/                   # 01 data ingestion, 04 WC 2026 draw
├── archive/                     # earlier build notebooks (02,03,05,06,07,08) — the teaching trail
└── *.md                         # readable forecast + evaluation reports
```

The pipeline is: **data → Elo → +confederation → +squad → Poisson → draw → Monte Carlo → odds.**

## Running it

```bash
# Python 3.13, pandas, numpy, scipy, pyarrow
python 10_squad_strength_blend.py   # regenerates the forecast + CHAMPION_ODDS_FINAL.md
python 13_fit_presets.py            # re-fits the presets and reports the RPS surface
```

`data/processed/*.parquet` are regenerated by the scripts and are not tracked.

## Data sources

- **[martj42/international_results](https://github.com/martj42/international_results)** — every men's international 1872→2026 (~49k matches).
- **EA Sports FC 26 player ratings** — squad-strength talent prior (Sept-2025 snapshot).
- **FIFA WC 2026 draw + Annex C** — the official 48-team bracket and third-place allocation.

---

*The model is frozen for the duration of the tournament — no mid-event tweaks. That's
the credibility test: the predictions stand or fall as they were made before kickoff.*
</content>
</invoke>
